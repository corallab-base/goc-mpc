"""Constraint-satisfying waypoint seed for `SmallContinuousVRPSolver`'s
discrete skeleton search.

For every enumerated `(assignment, aux)` skeleton, one Levenberg-Marquardt
solve of the problem's OWN compiled node/edge constraints (the eq/ineq
residuals `local_refine`'s AL sees, masked the same way) over the free wp
columns, every node started at the live `x0` -- so the routing DP prices
configurations that actually satisfy each node's constraints (e.g. an
FK-reach node's arm IK) instead of the x0/interpolated template, and a
skeleton whose constraints the seed can't satisfy ranks last.

The routing objective is deliberately absent here -- the DP prices routes
on the result, `local_refine` then trades route cost against the
constraints jointly. A proximal pull toward the start configuration,
annealed to ~0 over the iterations, selects the solution nearest the start
without leaving a bias in the converged residual.

Projections are applied at branch 0 / `psi = 0` during the solve; the grid
re-resolves every branch on top of the seeded template afterwards."""

import jax
import jax.numpy as jnp
import numpy as np

from .problem import apply_anchor, apply_projections
from .solver import _accepts_node_rank, _call_residual
from ..logic_based_benders_solver.structure import warm_start_wp_parts


def _active_columns(problem):
    """Flat `(node*state_dim + col)` wp indices the seed moves: the free
    columns some residual reads, or every free column if any residual's read
    set isn't statically known."""
    free = np.asarray(problem.wp_free_idx)
    read_lists = list(problem.eq_read_cols) + list(problem.ineq_read_cols)
    if any(rc is None for rc in read_lists):
        return free
    S = problem.state_dim
    read = {int(n) * S + int(c) for rc in read_lists for (n, c) in rc}
    return np.asarray([i for i in free if int(i) in read], dtype=int)


def make_kinematic_seed(problem, combos, aux, t_seed, *, n_iters=50,
                        rest_weight=(1e-1, 1e-8), lm_lambda0=1e-2,
                        ineq_margin=1e-4, eq_eps=1e-4):
    """Returns `seed(x0, params, anchor) -> (wp (NC*NA, N, S), cv (NC*NA,))`,
    jittable, row `c*NA + a` for `combos[c]` (`(NC, n_variables)` agent ids)
    and `aux[a]` (`(NA, n_cond)` 0/1). `cv` is the seeded wp's constraint
    violation (solver.py's `_calc_cv_jax` convention). `t_seed` `(N,)`: the
    node rank order-dependent residuals are evaluated at (one linear
    extension of the hard edges)."""
    N, S = problem.n_nodes, problem.state_dim
    J = problem.n_agents
    NC, NA = int(combos.shape[0]), int(aux.shape[0])
    G = NC * NA
    const, from_x0 = (jnp.asarray(x) for x in warm_start_wp_parts(problem))

    active = _active_columns(problem)
    D = int(active.shape[0])
    eq_fns, ineq_fns = problem._eq_constraints, problem._ineq_constraints
    n_rows = problem.n_eq_extra + problem.n_ineq_extra

    def template(x0):
        return jnp.where(from_x0, x0[None, :], const)

    if D == 0 or n_rows == 0:
        def seed_trivial(x0, params, anchor):
            return jnp.broadcast_to(template(x0)[None], (G, N, S)), jnp.zeros((G,))
        return seed_trivial

    active_j = jnp.asarray(active)
    wp_lo = jnp.asarray(problem.xl[problem.wp_offset:problem.psi_offset])[active_j]
    wp_hi = jnp.asarray(problem.xu[problem.wp_offset:problem.psi_offset])[active_j]
    eq_mask = jnp.asarray(problem.eq_free_mask, dtype=float)
    ineq_mask = jnp.asarray(problem.ineq_free_mask, dtype=float)
    eq_rank = [_accepts_node_rank(fn) for fn in eq_fns]
    ineq_rank = [_accepts_node_rank(fn) for fn in ineq_fns]
    t1 = jnp.asarray(t_seed, dtype=float)[None]
    psi1 = jnp.zeros((1, problem.n_psi))
    pb1 = jnp.zeros((1, problem.n_branch))
    combos_j = jnp.asarray(combos, dtype=jnp.int32)
    cond_all = jnp.tile(jnp.asarray(aux, dtype=float), (NC, 1))  # (G, n_cond)
    combo_idx = jnp.repeat(jnp.arange(NC), NA)                   # (G,)
    w0, w1 = rest_weight
    w_sched = jnp.asarray(w0 * (w1 / w0) ** (np.arange(n_iters) / max(1, n_iters - 1)))
    use_rev = n_rows <= D

    def constraint_rows(z, base, assign1, cond1, x0, params, anchor):
        """`(h (n_eq,), g (n_ineq,))` at wp = base with `active` set to `z`."""
        wp = base.reshape(-1).at[active_j].set(z).reshape(1, N, S)
        wp = apply_projections(problem, wp, psi1, pb1, params, assign=assign1[None], anchor=anchor,
                               cond_binary=cond1[None], t=t1, node_active=anchor.node_active, x0=x0)
        assign_eff, wp_f, wp_l = apply_anchor(problem, assign1[None], wp, anchor, x0)

        def rows(fns, wants):
            return jnp.concatenate(
                [_call_residual(fn, w, assign_eff, cond1[None], t1, wp_f, wp_l,
                                anchor.node_active, x0, params, None)
                 for fn, w in zip(fns, wants)], axis=1)[0]
        h = rows(eq_fns, eq_rank) * eq_mask if eq_fns else jnp.zeros((0,))
        g = rows(ineq_fns, ineq_rank) * ineq_mask if ineq_fns else jnp.zeros((0,))
        return h, g

    def seed_one(assign1, cond1, base, x0, params, anchor):
        def r_fn(z):
            h, g = constraint_rows(z, base, assign1, cond1, x0, params, anchor)
            return jnp.concatenate([h, jnp.maximum(0.0, g + ineq_margin)])

        jac = jax.jacrev(r_fn) if use_rev else jax.jacfwd(r_fn)
        z0 = base.reshape(-1)[active_j]

        def merit(z, r, w):
            return r @ r + w * jnp.sum((z - z0) ** 2)

        def step(carry, w):
            z, lam = carry
            r = r_fn(z)
            Jm = jac(z)                                              # (R, D)
            b = Jm.T @ r + w * (z - z0)
            mu = w + lam
            if use_rev:   # Woodbury: (J^T J + mu I)^-1 b via an R x R solve
                inner = jnp.linalg.solve(Jm @ Jm.T + mu * jnp.eye(n_rows), Jm @ b)
                delta = -(b - Jm.T @ inner) / mu
            else:
                delta = -jnp.linalg.solve(Jm.T @ Jm + mu * jnp.eye(D), b)
            z_new = jnp.clip(z + delta, wp_lo, wp_hi)
            m_new = merit(z_new, r_fn(z_new), w)
            accept = jnp.isfinite(m_new) & (m_new < merit(z, r, w))
            z = jnp.where(accept, z_new, z)
            lam = jnp.clip(jnp.where(accept, lam * 0.3, lam * 5.0), 1e-9, 1e9)
            return (z, lam), None

        (z, _), _ = jax.lax.scan(step, (z0, jnp.asarray(lm_lambda0, dtype=float)), w_sched)
        h, g = constraint_rows(z, base, assign1, cond1, x0, params, anchor)
        cv = jnp.sum(jnp.maximum(0.0, g)) + jnp.sum(jnp.maximum(0.0, jnp.abs(h) - eq_eps))
        return base.reshape(-1).at[active_j].set(z).reshape(N, S), cv

    def seed(x0, params, anchor):
        base = template(x0)
        if problem.n_variables:
            slot_agent = jnp.where(anchor.var_committed[None, :], anchor.var_anchor[None, :],
                                   combos_j)
            assign_all = jax.nn.one_hot(slot_agent, J)[combo_idx]    # (G, n_var, J)
        else:
            assign_all = jnp.zeros((G, 0, J))
        return jax.vmap(seed_one, (0, 0, None, None, None, None))(
            assign_all, cond_all, base, x0, params, anchor)

    return seed

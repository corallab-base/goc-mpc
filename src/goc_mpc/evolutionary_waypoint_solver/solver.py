"""Augmented-Lagrangian + batched-L-BFGS + GA-operator solver for
GraphOrderingRelaxed (problem.py): a single jax.lax.scan generation loop
combining Lamarckian AL+L-BFGS local refinement of `wp`, order-crossover +
2-opt/Or-opt local search of `t`, and smooth combined-score ranking.

`build_lamarckian_ga` returns a jitted `step(carry_in, x0, params, anchor) ->
carry_out` over an arbitrary carry (X, mu, lam, rho, F, CV, key, best_X,
best_F, best_CV) -- it does not build its own initial population. Building
the *first* carry (random population + precedence-heuristic seeding) is
`build_initial_carry_fn`; `carry_from_population` wraps an existing
population (e.g. a previous call's `Result.pop[0]`) into a fresh carry for a
new (but same-shaped) problem, letting a caller resume a previous run's
population/AL-state/PRNG stream -- via `run_lamarckian_al(_init_carry=...)`
-- instead of cold-restarting the search every call (see
EvolutionaryWaypointSolver in mpc.py, which warm-starts every MPC tick this
way).

`x0` (the full configuration -- state_dim-wide, agent depot then object
depot, see problem.apply_anchor), `params` (GraphOfConstraints.
view_param_values() -- every declared add_param(...)'s current value, read
by any compiled constraint referencing a param(id) placeholder, see
spec.py's _make_row_resolver), and `anchor` (problem.AnchorState) are
genuine arguments of `step`/`problem._batched`, not values baked into
`problem`'s kernel closures at construction time, so a compiled `step` (and
everything it closes over) stays valid across depot/param drift and
remaining_vertices changes between MPC ticks without retracing, as long as
`problem`'s structure (shapes, edges, constraints, and DECLARED param count)
is unchanged -- set_param overwrites a value in place without changing that
count, so it never forces a rebuild/retrace, only add_param (a genuinely new
placeholder) does. `x0` is both what problem.apply_anchor substitutes in for
a "live"-mode constraint's passed-node reads, and (sliced down to its
(n_agents, dim) agent-only view via problem.agent_depot, right where
kernel.py's routing math needs it) the routing depot -- one real-state value
flowing through this whole stack; `params` is a second, independent one (no
anchoring/frozen-live distinction applies to it -- a param is a plain
caller-supplied constant, not a per-node quantity). Only
`build_initial_carry_fn`'s cold-start `init(key)` uses a fixed `problem.x0`/
`problem.params` (the eager numpy precedence-seeding heuristic can't run
under a dynamic-x0 trace, and `problem.x0` is itself only ever agent-shaped
-- see problem.pad_to_state_dim) -- acceptable since it only runs once per
distinct problem shape, never on the warm-started hot path.
"""

import time
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from .problem import apply_anchor, agent_depot, pad_to_state_dim

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Batched Augmented Lagrangian merit function
# ---------------------------------------------------------------------------

def _make_merit_batched(problem, wp_shape):
    n_nodes, state_dim = wp_shape
    batched_kernel = problem._batched
    eq_fns, ineq_fns = problem._eq_constraints, problem._ineq_constraints

    def merit(wp_flat, assign, cond_binary, t, mu, lam, rho, x0, params, anchor):
        wp = wp_flat.reshape(-1, n_nodes, state_dim)
        # anchor splices remaining_vertices state in once here: a node/
        # variable no longer in remaining_vertices reads back as either its
        # last-committed planned constant (wp_eff_frozen) or the current call's
        # real state (wp_eff_live, straight from `x0` -- see problem.apply_anchor)
        # -- whichever a given eq/ineq fn was compiled to read -- not the free
        # wp_flat/assign value being optimized. Routing (batched_kernel) never
        # reads a passed row either way, so it's handed wp_eff_frozen
        # arbitrarily; it only ever wants x0's agent-depot slice regardless
        # (problem.agent_depot).
        assign_eff, wp_eff_frozen, wp_eff_live = apply_anchor(problem, assign, wp, anchor, x0)
        F, _g_kernel = batched_kernel(assign_eff, cond_binary, t, wp_eff_frozen, agent_depot(problem, x0),
                                       anchor.node_active)

        total = F
        if eq_fns:
            h = jnp.concatenate(
                [fn(assign_eff, cond_binary, t, wp_eff_frozen, wp_eff_live, anchor.node_active, x0, params)
                 for fn in eq_fns], axis=1)
            total = total + jnp.sum(mu * h, axis=1) + 0.5 * rho * jnp.sum(h * h, axis=1)
        if ineq_fns:
            g = jnp.concatenate(
                [fn(assign_eff, cond_binary, t, wp_eff_frozen, wp_eff_live, anchor.node_active, x0, params)
                 for fn in ineq_fns], axis=1)
            z = jnp.maximum(0.0, lam + rho[:, None] * g)
            total = total + (jnp.sum(z * z, axis=1) - jnp.sum(lam * lam, axis=1)) / (2.0 * rho)
        return total

    def merit_and_grad(wp_flat, assign, cond_binary, t, mu, lam, rho, x0, params, anchor):
        value, vjp_fn = jax.vjp(
            lambda w: merit(w, assign, cond_binary, t, mu, lam, rho, x0, params, anchor), wp_flat)
        (grad,) = vjp_fn(jnp.ones_like(value))
        return value, grad

    return merit_and_grad


def _eval_residuals_batched(fns, assign, cond_binary, t, wp_frozen, wp_live, node_active, x0, params):
    pop = wp_frozen.shape[0]
    if not fns:
        return jnp.zeros((pop, 0))
    return jnp.concatenate(
        [fn(assign, cond_binary, t, wp_frozen, wp_live, node_active, x0, params) for fn in fns], axis=1)


# ---------------------------------------------------------------------------
# Batched L-BFGS: two-loop recursion + Armijo backtracking
# ---------------------------------------------------------------------------

def _lbfgs_direction(grad, s_hist, y_hist, lb_rho):
    pop, m, d = s_hist.shape

    def loop1(q, idx):
        s_i, y_i, rho_i = s_hist[:, idx, :], y_hist[:, idx, :], lb_rho[:, idx]
        a = rho_i * jnp.sum(s_i * q, axis=1)
        return q - a[:, None] * y_i, a

    q, alpha_rev = jax.lax.scan(loop1, grad, jnp.arange(m - 1, -1, -1))
    alpha_store = alpha_rev[::-1]

    s_last, y_last = s_hist[:, -1, :], y_hist[:, -1, :]
    num = jnp.sum(s_last * y_last, axis=1)
    den = jnp.sum(y_last * y_last, axis=1)
    gamma = jnp.where(den > 1e-12, num / jnp.where(den > 1e-12, den, 1.0), 1.0)
    r0 = gamma[:, None] * q

    def loop2(r, xs):
        idx, alpha_i = xs
        y_i, s_i, rho_i = y_hist[:, idx, :], s_hist[:, idx, :], lb_rho[:, idx]
        b = rho_i * jnp.sum(y_i * r, axis=1)
        return r + (alpha_i - b)[:, None] * s_i, None

    r, _ = jax.lax.scan(loop2, r0, (jnp.arange(m), alpha_store))
    return -r


def _backtrack(f, x0, f0, g0, direction, max_trials, alpha0=1.0, c1=1e-4, beta=0.5):
    pop = x0.shape[0]
    d0 = jnp.sum(g0 * direction, axis=1)

    def body(carry, _):
        alpha, x_best, f_best, g_best, done, _x_last, _f_last, _g_last = carry
        x_try = x0 + alpha[:, None] * direction
        f_try, g_try = f(x_try)
        armijo = f_try <= f0 + c1 * alpha * d0
        accept = armijo & (~done)
        x_best = jnp.where(accept[:, None], x_try, x_best)
        f_best = jnp.where(accept, f_try, f_best)
        g_best = jnp.where(accept[:, None], g_try, g_best)
        done_new = done | armijo
        alpha_new = jnp.where(done_new, alpha, alpha * beta)
        return (alpha_new, x_best, f_best, g_best, done_new, x_try, f_try, g_try), None

    init = (jnp.full((pop,), alpha0), x0, f0, g0, jnp.zeros((pop,), dtype=bool), x0, f0, g0)
    (_, x_best, f_best, g_best, done, x_last, f_last, g_last), _ = jax.lax.scan(
        body, init, xs=None, length=max_trials)

    x_best = jnp.where(done[:, None], x_best, x_last)
    f_best = jnp.where(done, f_best, f_last)
    g_best = jnp.where(done[:, None], g_best, g_last)
    return x_best, f_best, g_best


def _lbfgs_solve(merit_and_grad_fixed, wp_flat0, m, inner_maxiter, max_ls_trials):
    pop, d = wp_flat0.shape
    s_hist0 = jnp.zeros((pop, m, d))
    y_hist0 = jnp.zeros((pop, m, d))
    lb_rho0 = jnp.zeros((pop, m))
    f0, g0 = merit_and_grad_fixed(wp_flat0)

    def body(carry, _):
        x, f, g, s_hist, y_hist, lb_rho = carry
        direction = _lbfgs_direction(g, s_hist, y_hist, lb_rho)
        x_new, f_new, g_new = _backtrack(merit_and_grad_fixed, x, f, g, direction, max_ls_trials)

        s = x_new - x
        y = g_new - g
        sy = jnp.sum(s * y, axis=1)
        curvature_ok = sy > 1e-10
        new_rho = jnp.where(curvature_ok, 1.0 / jnp.where(curvature_ok, sy, 1.0), 0.0)

        keep = curvature_ok[:, None, None]
        s_hist_new = jnp.concatenate([s_hist[:, 1:, :], jnp.where(keep, s[:, None, :], 0.0)], axis=1)
        y_hist_new = jnp.concatenate([y_hist[:, 1:, :], jnp.where(keep, y[:, None, :], 0.0)], axis=1)
        lb_rho_new = jnp.concatenate([lb_rho[:, 1:], jnp.where(curvature_ok, new_rho, 0.0)[:, None]], axis=1)

        return (x_new, f_new, g_new, s_hist_new, y_hist_new, lb_rho_new), None

    init = (wp_flat0, f0, g0, s_hist0, y_hist0, lb_rho0)
    (x_final, _, _, _, _, _), _ = jax.lax.scan(body, init, xs=None, length=inner_maxiter)
    return x_final


# ---------------------------------------------------------------------------
# Batched AL outer loop around the L-BFGS solve
# ---------------------------------------------------------------------------

def make_batched_local_refine(problem, outer_iters, inner_maxiter, rho_growth, rho_max,
                                lbfgs_history=10, ls_max_trials=10):
    n_nodes, state_dim = problem.n_nodes, problem.state_dim
    eq_fns, ineq_fns = problem._eq_constraints, problem._ineq_constraints
    merit_and_grad = _make_merit_batched(problem, (n_nodes, state_dim))

    lo = jnp.asarray(problem.xl[problem.wp_offset:])
    hi = jnp.asarray(problem.xu[problem.wp_offset:])

    def batched_local_refine(wp0, assign, cond_binary, t, mu, lam, rho, x0, params, anchor, cv_tol):
        """`cv_tol` (per-generation-annealed, from _score_schedule/
        _calibrate_score_scale -- the SAME absolute yardstick the GA's own
        selection score already penalizes violation against) is what grows
        `rho` here, not a per-call relative "did this outer iteration's own
        refinement shrink violation enough" comparison: that self-relative
        version compares each iteration's post-refinement violation against
        wherever THAT SAME iteration started, which -- fed a warm-started wp
        that's already a little better each call -- can keep clearing its
        own bar indefinitely while the absolute gap barely moves, so `rho`
        never grows and the AL penalty never actually ramps up enough to
        close a genuinely infeasible point. Comparing against `cv_tol`
        instead means growth is judged against a fixed target, not a moving
        one that resets every call. Uses _calc_cv_jax (the same summed
        violation _evaluate_population_jax computes CV0/cv_tol's own
        calibration from), not a max-norm, so the comparison is scale-
        consistent with what `cv_tol` was actually calibrated against."""
        pop = wp0.shape[0]
        wp_flat = wp0.reshape(pop, -1)

        def merit_and_grad_fixed(w):
            return merit_and_grad(w, assign, cond_binary, t, mu, lam, rho, x0, params, anchor)

        for _ in range(outer_iters):
            wp_flat = _lbfgs_solve(merit_and_grad_fixed, wp_flat, lbfgs_history, inner_maxiter, ls_max_trials)
            wp_flat = jnp.clip(wp_flat, lo, hi)

            wp = wp_flat.reshape(pop, n_nodes, state_dim)
            assign_eff, wp_eff_frozen, wp_eff_live = apply_anchor(problem, assign, wp, anchor, x0)
            h1 = _eval_residuals_batched(eq_fns, assign_eff, cond_binary, t, wp_eff_frozen, wp_eff_live,
                                          anchor.node_active, x0, params)
            g1 = _eval_residuals_batched(ineq_fns, assign_eff, cond_binary, t, wp_eff_frozen, wp_eff_live,
                                          anchor.node_active, x0, params)
            v1 = _calc_cv_jax(pop, g1, h1)

            if eq_fns:
                mu = mu + rho[:, None] * h1
            if ineq_fns:
                lam = jnp.maximum(0.0, lam + rho[:, None] * g1)
            rho = jnp.where(v1 <= cv_tol, rho, jnp.minimum(rho * rho_growth, rho_max))

            def merit_and_grad_fixed(w, assign=assign, cond_binary=cond_binary, t=t, mu=mu, lam=lam, rho=rho,
                                      x0=x0, params=params, anchor=anchor):
                return merit_and_grad(w, assign, cond_binary, t, mu, lam, rho, x0, params, anchor)

        return wp_flat.reshape(pop, n_nodes, state_dim), mu, lam, rho

    return batched_local_refine


def _write_wp_batch_jax(problem, X, wp):
    pop = X.shape[0]
    return X.at[:, problem.wp_offset:].set(wp.reshape(pop, -1))


def _write_t_batch_jax(problem, X, t):
    return X.at[:, problem.t_offset:problem.wp_offset].set(t)


# ---------------------------------------------------------------------------
# Precedence-respecting construction heuristic
# ---------------------------------------------------------------------------

def _seed_precedence_permutation(problem):
    n_nodes = problem.n_nodes
    hard_edges = list(getattr(problem, "hard_edges", []))

    indeg = [0] * n_nodes
    succ = [[] for _ in range(n_nodes)]
    for u, v in hard_edges:
        succ[u].append(v)
        indeg[v] += 1

    node_targets = None
    for fn in problem._eq_constraints:
        node_targets = getattr(fn, "node_targets", None)
        if node_targets is not None:
            break

    visited = [False] * n_nodes
    available = [n for n in range(n_nodes) if indeg[n] == 0]
    order = []
    pos = np.mean(problem.x0, axis=0) if node_targets is not None else None

    while len(order) < n_nodes:
        if not available:
            available = [n for n in range(n_nodes) if not visited[n]]
        if node_targets is not None:
            dists = [np.linalg.norm(node_targets[n] - pos) for n in available]
            pick = available.pop(int(np.argmin(dists)))
            pos = node_targets[pick]
        else:
            available.sort()
            pick = available.pop(0)
        visited[pick] = True
        order.append(pick)
        for v in succ[pick]:
            if visited[v]:
                continue
            indeg[v] -= 1
            if indeg[v] == 0:
                available.append(v)

    seed_t = np.zeros(n_nodes)
    seed_t[np.array(order, dtype=int)] = np.arange(n_nodes)
    seed_wp = np.asarray(node_targets, dtype=float) if node_targets is not None else None
    return seed_t, seed_wp


# ---------------------------------------------------------------------------
# Order crossover (OX) on the decoded global node permutation
# ---------------------------------------------------------------------------

def _ox_single(key, perm1, perm2):
    n = perm1.shape[0]
    key_a, key_len = jax.random.split(key)
    a = jax.random.randint(key_a, (), 0, n)
    length = jax.random.randint(key_len, (), 1, n)
    b = (a + length) % n

    idx = jnp.arange(n)
    rel = (idx - a) % n
    kept_mask = rel < length

    rolled_idx = (idx + b) % n
    perm2_rolled = perm2[rolled_idx]
    kept_mask_rolled = kept_mask[rolled_idx]

    is_value_kept = jnp.zeros(n, dtype=bool).at[perm1].set(kept_mask)
    p2_val_is_new = ~is_value_kept[perm2_rolled]
    fill_key = jnp.where(p2_val_is_new, idx, n + idx)
    fill_order = jnp.argsort(fill_key)
    compacted_p2 = perm2_rolled[fill_order]

    cum_empty = jnp.cumsum(~kept_mask_rolled) - 1
    fill_value_rolled = compacted_p2[jnp.clip(cum_empty, 0, n - 1)]
    fill_value_at_position = fill_value_rolled[(idx - b) % n]

    return jnp.where(kept_mask, perm1, fill_value_at_position)


def _ox_crossover_batched(key, perm1, perm2):
    pop = perm1.shape[0]
    keys = jax.random.split(key, pop)
    return jax.vmap(_ox_single)(keys, perm1, perm2)


# ---------------------------------------------------------------------------
# Precedence-respecting 2-opt + Or-opt local search
# ---------------------------------------------------------------------------

def _or_opt_single(key, perm, max_seg_len):
    n = perm.shape[0]
    key_s, key_len, key_d = jax.random.split(key, 3)
    L = jax.random.randint(key_len, (), 1, max_seg_len + 1)
    s = jax.random.randint(key_s, (), 0, n - max_seg_len + 1)

    idx = jnp.arange(n)
    in_seg = (idx >= s) & (idx < s + L)

    remaining_rank = jnp.cumsum(~in_seg) - 1
    d = jnp.clip(
        jnp.floor(jax.random.uniform(key_d) * (n - L + 1).astype(jnp.float32)).astype(jnp.int32),
        0, n - L)

    seg_rank = idx - s
    new_rank_remaining = remaining_rank + jnp.where(remaining_rank >= d, L, 0)
    new_rank_seg = d + seg_rank
    new_rank = jnp.where(in_seg, new_rank_seg, new_rank_remaining)

    return jnp.zeros(n, dtype=perm.dtype).at[new_rank].set(perm)


def _or_opt_batched(key, perm, max_seg_len):
    pop = perm.shape[0]
    keys = jax.random.split(key, pop)
    return jax.vmap(_or_opt_single, in_axes=(0, 0, None))(keys, perm, max_seg_len)


def _routing_local_search_batched(problem, key, assign, cond_binary, t, wp, x0, node_active, n_trials,
                                   or_opt_prob=0.3, max_or_opt_seg_len=3):
    """`assign`/`wp` are expected to already be anchor-substituted
    (assign_eff/wp_eff, see problem.apply_anchor) -- callers compute that
    once per generation and pass the effective values straight through here,
    since this function's own problem._batched calls need it every trial.
    `x0` is the full configuration (see problem.apply_anchor) -- sliced down
    to its agent-depot view once here since that's all problem._batched
    (routing) ever reads."""
    pop, n_nodes = t.shape
    max_or_opt_seg_len = max(1, min(max_or_opt_seg_len, n_nodes - 1)) if n_nodes > 1 else 0
    or_opt_prob = or_opt_prob if max_or_opt_seg_len > 0 else 0.0
    agent_x0 = agent_depot(problem, x0)

    def body(carry, k):
        t_cur, F_cur = carry
        k_type, k_2opt, k_oropt = jax.random.split(k, 3)
        perm = jnp.argsort(t_cur, axis=1)

        k_a, k_len = jax.random.split(k_2opt)
        a = jax.random.randint(k_a, (pop,), 0, n_nodes)
        length = jax.random.randint(k_len, (pop,), 2, n_nodes + 1)
        idx = jnp.arange(n_nodes)[None, :]
        rel = (idx - a[:, None]) % n_nodes
        seg_mask = rel < length[:, None]
        mirror_pos = (a[:, None] + (length[:, None] - 1 - rel)) % n_nodes
        gather_idx = jnp.where(seg_mask, mirror_pos, idx)
        perm_2opt = jnp.take_along_axis(perm, gather_idx, axis=1)

        perm_oropt = _or_opt_batched(k_oropt, perm, max_or_opt_seg_len)

        do_or_opt = jax.random.uniform(k_type, (pop,)) < or_opt_prob
        new_perm = jnp.where(do_or_opt[:, None], perm_oropt, perm_2opt)

        row_idx = jnp.arange(pop)[:, None]
        rank = jnp.broadcast_to(jnp.arange(n_nodes), (pop, n_nodes))
        t_new = jnp.zeros((pop, n_nodes)).at[row_idx, new_perm].set(rank)

        F_new, _ = problem._batched(assign, cond_binary, t_new, wp, agent_x0, node_active)
        accept = F_new < F_cur
        t_next = jnp.where(accept[:, None], t_new, t_cur)
        F_next = jnp.where(accept, F_new, F_cur)
        return (t_next, F_next), None

    F0, _ = problem._batched(assign, cond_binary, t, wp, agent_x0, node_active)
    keys = jax.random.split(key, n_trials)
    (t_final, _), _ = jax.lax.scan(body, (t, F0), keys)
    return t_final


# ---------------------------------------------------------------------------
# Batched constraint-violation aggregation
# ---------------------------------------------------------------------------

def _calc_cv_jax(pop, G, H, eq_eps=1e-4):
    total = jnp.zeros(pop)
    if G is not None:
        total = total + jnp.sum(jnp.maximum(0.0, G), axis=1)
    if H is not None:
        total = total + jnp.sum(jnp.maximum(0.0, jnp.abs(H) - eq_eps), axis=1)
    return total


def _evaluate_population_jax(problem, X, x0, params, anchor):
    pop = X.shape[0]
    assign, cond_binary, t, wp = problem._extract_batch(X)
    assign_eff, wp_eff_frozen, wp_eff_live = apply_anchor(problem, assign, wp, anchor, x0)
    F, _G_kernel = problem._batched(assign_eff, cond_binary, t, wp_eff_frozen, agent_depot(problem, x0),
                                     anchor.node_active)
    G = (jnp.concatenate(
            [fn(assign_eff, cond_binary, t, wp_eff_frozen, wp_eff_live, anchor.node_active, x0, params)
             for fn in problem._ineq_constraints], axis=1)
         if problem._ineq_constraints else None)
    H = (jnp.concatenate(
            [fn(assign_eff, cond_binary, t, wp_eff_frozen, wp_eff_live, anchor.node_active, x0, params)
             for fn in problem._eq_constraints], axis=1)
         if problem._eq_constraints else None)
    CV = _calc_cv_jax(pop, G, H)
    return F, CV


# ---------------------------------------------------------------------------
# Smooth soft-penalty ranking/selection
# ---------------------------------------------------------------------------

def _combined_score(F, CV, w, cv_tol):
    return F + w * jnp.maximum(0.0, CV - cv_tol)


def _calibrate_score_scale(F0, CV0, w_frac, cv_tol_frac):
    F_scale = jnp.median(jnp.abs(F0)) + 1e-8
    CV_scale = jnp.median(CV0) + 1e-8
    w = w_frac * F_scale / CV_scale
    cv_tol = cv_tol_frac * CV_scale
    return w, cv_tol


def _score_schedule(gen, n_gen, w0, cv_tol0, w_growth, cv_tol_floor_frac):
    frac = gen / max(n_gen - 1, 1)
    frac = jnp.clip(frac, 0.0, 1.0)
    w_t = w0 * (1.0 + (w_growth - 1.0) * frac)
    cv_tol_t = cv_tol0 * (1.0 - (1.0 - cv_tol_floor_frac) * frac)
    return w_t, cv_tol_t


def _rank_jax(S):
    return jnp.argsort(S)


def _tournament_select_jax(key, S, pop_size, k=2):
    idx = jax.random.randint(key, (pop_size, k), 0, pop_size)
    best = idx[:, 0]
    for j in range(1, k):
        cand = idx[:, j]
        take_cand = S[cand] < S[best]
        best = jnp.where(take_cand, cand, best)
    return best


# ---------------------------------------------------------------------------
# Whole-run GA as a single jax.lax.scan over generations
# ---------------------------------------------------------------------------

def _make_gen_step_fn(problem, local_refine, pop_size, n_gen, mut_sigma, cx_prob,
                       ox_prob, n_2opt_trials, or_opt_prob, max_or_opt_seg_len,
                       tournament_k, w0, cv_tol0, w_growth, cv_tol_floor_frac, x0, params, anchor):
    n_var = problem.n_var
    n_nodes = problem.n_nodes
    xl, xu = jnp.asarray(problem.xl), jnp.asarray(problem.xu)

    def gen_step(carry, gen):
        X, mu, lam, rho, F, CV, key, best_X, best_F, best_CV = carry
        w, cv_tol = _score_schedule(gen, n_gen, w0, cv_tol0, w_growth, cv_tol_floor_frac)
        S = _combined_score(F, CV, w, cv_tol)
        key, k_p1, k_p2, k_cx_mask, k_cx_alpha, k_mut, k_ox_mask, k_ox, k_2opt = jax.random.split(key, 9)

        p1 = _tournament_select_jax(k_p1, S, pop_size, k=tournament_k)
        p2 = _tournament_select_jax(k_p2, S, pop_size, k=tournament_k)

        do_cx = jax.random.uniform(k_cx_mask, (pop_size,)) < cx_prob
        alpha = jax.random.uniform(k_cx_alpha, (pop_size, n_var), minval=-0.5, maxval=1.5)
        blended = X[p1] + alpha * (X[p2] - X[p1])
        child_X = jnp.where(do_cx[:, None], blended, X[p1])
        noise = jax.random.normal(k_mut, (pop_size, n_var), dtype=xl.dtype) * mut_sigma
        child_X = child_X + noise * (xu - xl)
        child_X = jnp.clip(child_X, xl, xu)

        _, _, t_p1, _ = problem._extract_batch(X[p1])
        _, _, t_p2, _ = problem._extract_batch(X[p2])
        perm1, perm2 = jnp.argsort(t_p1, axis=1), jnp.argsort(t_p2, axis=1)
        child_perm_ox = _ox_crossover_batched(k_ox, perm1, perm2)
        row_idx = jnp.arange(pop_size)[:, None]
        rank = jnp.broadcast_to(jnp.arange(n_nodes), (pop_size, n_nodes))
        t_ox = jnp.zeros((pop_size, n_nodes)).at[row_idx, child_perm_ox].set(rank)
        do_ox = jax.random.uniform(k_ox_mask, (pop_size,)) < ox_prob
        _, _, t_blx, _ = problem._extract_batch(child_X)
        child_X = _write_t_batch_jax(problem, child_X, jnp.where(do_ox[:, None], t_ox, t_blx))

        parent = jnp.where(S[p1] < S[p2], p1, p2)
        child_mu, child_lam, child_rho = mu[parent], lam[parent], rho[parent]

        child_assign, child_cond_binary, child_t, child_wp0 = problem._extract_batch(child_X)
        child_wp_star, off_mu, off_lam, off_rho = local_refine(
            child_wp0, child_assign, child_cond_binary, child_t, child_mu, child_lam, child_rho, x0, params, anchor,
            cv_tol)
        off_X = _write_wp_batch_jax(problem, child_X, child_wp_star)

        off_assign, off_cond_binary, off_t, off_wp = problem._extract_batch(off_X)
        # Routing local search never reads a passed row's value either way
        # (masked out via node_active regardless -- see kernel.py), so
        # off_wp_eff_frozen is handed to it arbitrarily; off_wp_eff_live is
        # unused here.
        off_assign_eff, off_wp_eff_frozen, _off_wp_eff_live = apply_anchor(problem, off_assign, off_wp, anchor, x0)
        off_t = _routing_local_search_batched(problem, k_2opt, off_assign_eff, off_cond_binary, off_t,
                                               off_wp_eff_frozen, x0, anchor.node_active, n_2opt_trials,
                                               or_opt_prob=or_opt_prob, max_or_opt_seg_len=max_or_opt_seg_len)
        off_X = _write_t_batch_jax(problem, off_X, off_t)

        off_F, off_CV = _evaluate_population_jax(problem, off_X, x0, params, anchor)

        pool_X = jnp.concatenate([X, off_X], axis=0)
        pool_mu = jnp.concatenate([mu, off_mu], axis=0)
        pool_lam = jnp.concatenate([lam, off_lam], axis=0)
        pool_rho = jnp.concatenate([rho, off_rho], axis=0)
        pool_F = jnp.concatenate([F, off_F], axis=0)
        pool_CV = jnp.concatenate([CV, off_CV], axis=0)
        pool_S = _combined_score(pool_F, pool_CV, w, cv_tol)

        keep = _rank_jax(pool_S)[:pop_size]
        X_new, mu_new, lam_new, rho_new = pool_X[keep], pool_mu[keep], pool_lam[keep], pool_rho[keep]
        F_new, CV_new, S_new = pool_F[keep], pool_CV[keep], pool_S[keep]

        best_S = _combined_score(best_F, best_CV, w, cv_tol)
        improved = S_new[0] < best_S
        best_X_new = jnp.where(improved, X_new[0], best_X)
        best_F_new = jnp.where(improved, F_new[0], best_F)
        best_CV_new = jnp.where(improved, CV_new[0], best_CV)

        return (X_new, mu_new, lam_new, rho_new, F_new, CV_new, key,
                best_X_new, best_F_new, best_CV_new)

    return gen_step


def _seed_initial_population(problem, key, X0, n_seed, seed_jitter_t, seed_jitter_wp_frac):
    if n_seed <= 0:
        return X0
    n_nodes = problem.n_nodes
    seed_t_np, seed_wp_np = _seed_precedence_permutation(problem)
    seed_t = jnp.asarray(seed_t_np, dtype=X0.dtype)

    key, k_seed_t = jax.random.split(key)
    jitter_t = jax.random.uniform(k_seed_t, (n_seed, n_nodes), minval=-seed_jitter_t, maxval=seed_jitter_t)
    seed_block = _write_t_batch_jax(problem, X0[:n_seed], seed_t[None, :] + jitter_t)

    if seed_wp_np is not None:
        # Structurally dead in practice: `target_eq_constraint`'s `.node_target`
        # attribute is set on the raw per-instance fn, but problem._eq_constraints
        # holds the wrapped batched closures from _batch_python_constraint_fn,
        # which never copies it over -- so _seed_precedence_permutation's
        # getattr(fn, "node_targets", None) lookup above never matches (pre-
        # existing behavior, unrelated to the wp row layout below). Kept
        # dimensionally consistent with it regardless, in case that's fixed.
        n_nodes, state_dim = problem.n_nodes, problem.state_dim
        wp_lo = jnp.asarray(problem.xl[problem.wp_offset:].reshape(n_nodes, state_dim))
        wp_hi = jnp.asarray(problem.xu[problem.wp_offset:].reshape(n_nodes, state_dim))
        key, k_seed_wp = jax.random.split(key)
        jitter_wp = jax.random.normal(k_seed_wp, (n_seed, n_nodes, state_dim), dtype=X0.dtype) * (seed_jitter_wp_frac * (wp_hi - wp_lo))
        seed_wp = jnp.asarray(seed_wp_np, dtype=X0.dtype)
        seeded_wp = jnp.clip(seed_wp[None, :, :] + jitter_wp, wp_lo, wp_hi)
        seed_block = _write_wp_batch_jax(problem, seed_block, seeded_wp)

    return X0.at[:n_seed].set(seed_block)


def _carry_from_population(problem, key, X, pop_size, x0, params, anchor, mu=None, lam=None, rho=None, rho0=1.0):
    """Wraps a decision-variable population `X` -- freshly random, or reused
    from a previous run against a different (but same-shaped) problem -- into
    a full GA carry: (X, mu, lam, rho, F, CV, key, best_X, best_F, best_CV).

    `mu`/`lam`/`rho` (the AL multiplier state) default to zero/rho0 when not
    given (the fresh-random-population case, where there's nothing to reuse).
    When given (a previous carry's own state, reused against a same-shaped
    new problem), they're passed through as-is rather than reset: they
    represent how strongly the constraints have needed to be enforced so
    far, which is usually still a good estimate after a small problem
    perturbation (e.g. a shifted depot x0) and matters at least as much as
    the population itself for how quickly the search re-tightens onto a
    good, feasible point -- resetting them to rho0 every call would force
    the AL penalty to re-ramp from scratch each time regardless of how good
    X already is.

    F0/CV0/best-so-far are always (re-)evaluated fresh under `problem`:
    X's *values* stay meaningful across a same-shaped problem change, but
    its fitness does not. The best-so-far seed here only needs to be a
    reasonable individual to track from generation 0: build_lamarckian_ga's
    own step() recalibrates the real w/cv_tol-scaled score from this same
    F0/CV0 and will overwrite it the moment anything scores better under
    that schedule.
    """
    if mu is None or lam is None or rho is None:
        n_eq, n_ineq = problem.n_eq_constr, problem.n_ieq_constr
        mu = jnp.zeros((pop_size, n_eq)) if mu is None else mu
        lam = jnp.zeros((pop_size, n_ineq)) if lam is None else lam
        rho = jnp.full((pop_size,), rho0) if rho is None else rho

    F0, CV0 = _evaluate_population_jax(problem, X, x0, params, anchor)

    order0 = _rank_jax(F0 + 1e6 * jnp.maximum(0.0, CV0))
    best_X0, best_F0, best_CV0 = X[order0[0]], F0[order0[0]], CV0[order0[0]]

    return (X, mu, lam, rho, F0, CV0, key, best_X0, best_F0, best_CV0)


def carry_from_population(problem, X, x0, key, anchor, params=None, mu=None, lam=None, rho=None, pop_size=None,
                           rho0=1.0):
    """Public entry point for warm-starting a solve from a previous run's
    final population (e.g. `Result.pop[0]`) and, when given, its AL
    multiplier state too (e.g. `Result.pop[1:4]`) -- see
    `_carry_from_population`'s docstring for what is and isn't preserved.
    `x0` is the *current* full configuration (state_dim-wide -- see
    problem.apply_anchor; may differ from whatever `problem` was originally
    built with -- F0/CV0/best are always freshly evaluated here under this
    `x0`, never a stale one). `params` is likewise the *current*
    GraphOfConstraints.view_param_values() (defaults to problem.params, its
    structural build-time snapshot, when not given). `anchor` is likewise the
    *current* remaining_vertices state (problem.AnchorState, see
    EvolutionaryWaypointSolver._compute_anchor in mpc.py) -- reused
    population values stay meaningful across a remaining_vertices change (a
    node passing doesn't invalidate the search), but which nodes/variables
    are anchored-vs-free can differ from whatever anchor produced this X, so
    F0/CV0/best are always re-evaluated fresh under it here too."""
    X = jnp.asarray(X)
    x0 = jnp.asarray(x0)
    params = jnp.asarray(params) if params is not None else jnp.asarray(problem.params)
    pop_size = pop_size if pop_size is not None else X.shape[0]
    return _carry_from_population(problem, key, X, pop_size, x0, params, anchor, mu=mu, lam=lam, rho=rho, rho0=rho0)


def build_resume_carry_fn(problem, pop_size):
    """Builds a jitted `resume(X, x0, key, anchor, params, mu, lam, rho) ->
    carry`, compiled once per `problem` -- the same re-evaluate-under-
    current-(x0, anchor)-and-keep-(mu, lam, rho) logic `carry_from_population`/
    `_carry_from_population` implement (see their docstrings), but for
    EvolutionaryWaypointSolver's own repeated warm-resume call (solve()'s
    `self._carry is not None` branch, taken on every real call after the
    first) instead of a one-off external caller.

    `carry_from_population` stays a plain, un-jitted function deliberately
    -- it accepts `problem` as an ordinary argument (not a valid JAX
    tracer/pytree leaf) and tolerates `mu`/`lam`/`rho` being `None` (built
    fresh from `pop_size`/`rho0` instead), which suits a convenient one-off
    public entry point but means `_evaluate_population_jax`'s constraint
    evaluations -- one un-fused eager op per registered constraint, not a
    single cached compiled program -- are re-dispatched from scratch on
    EVERY call. For a problem with real per-constraint compute (e.g.
    forward-kinematics-heavy pick/place constraints), that eager-dispatch
    cost is genuine repeated compute, not one-time overhead: measured on
    one such problem, this was the actual dominant cost of `solve()` --
    ~620ms EVERY call, dwarfing the GA/AL/L-BFGS computation itself
    (~10ms) -- not a first-call-only compile artifact. Closing over
    `problem`/`pop_size` here (mirroring `build_lamarckian_ga`/
    `build_initial_carry_fn`'s own factory-returns-a-jitted-closure
    pattern) and requiring `mu`/`lam`/`rho` as plain (non-`None`) arguments
    -- always true for EvolutionaryWaypointSolver's own resume call, which
    only ever takes this branch once a carry, and so real `mu`/`lam`/`rho`,
    already exist -- makes this compilable, fixing that."""
    def resume(X, x0, key, anchor, params, mu, lam, rho):
        F0, CV0 = _evaluate_population_jax(problem, X, x0, params, anchor)
        order0 = _rank_jax(F0 + 1e6 * jnp.maximum(0.0, CV0))
        best_X0, best_F0, best_CV0 = X[order0[0]], F0[order0[0]], CV0[order0[0]]
        return (X, mu, lam, rho, F0, CV0, key, best_X0, best_F0, best_CV0)
    return jax.jit(resume)


def build_initial_carry_fn(problem, pop_size, anchor, rho0=1.0,
                            n_seed_individuals=None, seed_jitter_t=1.0, seed_jitter_wp_frac=0.05):
    """Returns a jitted `init(key) -> carry` building a fresh (cold) random
    population (plus a small precedence-heuristic-seeded subset) -- used the
    first time a given problem shape is solved, before any previous carry
    exists to resume/reuse from. `anchor` is baked into this closure exactly
    like `problem.x0` already is: this cold-start path runs exactly once per
    solver instance (see EvolutionaryWaypointSolver._ensure_built, mpc.py),
    at which point remaining_vertices == every graph node (problem.
    full_active_anchor), so there's no drift to track across calls the way
    build_lamarckian_ga's `step` must track x0/anchor drift. `problem.x0` is
    only ever agent-shaped (the structural x0 build_graph_ordering_problem
    was built from, see problem.GraphOrderingRelaxed), so it's zero-padded
    out to the full state_dim width every other `x0` here now carries (problem.
    pad_to_state_dim) -- inert padding, since full_active_anchor never
    actually reads it (nothing's passed yet)."""
    n_var = problem.n_var
    xl, xu = jnp.asarray(problem.xl), jnp.asarray(problem.xu)
    x0 = pad_to_state_dim(jnp.asarray(problem.x0).reshape(-1), problem.state_dim)
    params = jnp.asarray(problem.params)
    n_seed = pop_size // 10 if n_seed_individuals is None else n_seed_individuals
    n_seed = max(0, min(n_seed, pop_size))

    def init(key):
        key, k_init, k_seed = jax.random.split(key, 3)
        X0 = jax.random.uniform(k_init, (pop_size, n_var), minval=xl, maxval=xu, dtype=xl.dtype)
        X0 = _seed_initial_population(problem, k_seed, X0, n_seed, seed_jitter_t, seed_jitter_wp_frac)
        return _carry_from_population(problem, key, X0, pop_size, x0, params, anchor, rho0=rho0)

    return jax.jit(init)


def build_lamarckian_ga(problem, pop_size, n_gen, outer_iters=1, inner_maxiter=20,
                         rho_growth=10.0, rho_max=1e6,
                         mut_sigma=0.1, cx_prob=0.9,
                         lbfgs_history=10, ls_max_trials=10, tournament_k=2,
                         w=None, cv_tol=None, w_frac=1.0, cv_tol_frac=0.05,
                         w_growth=10.0, cv_tol_floor_frac=0.0,
                         ox_prob=0.5, n_2opt_trials=5, or_opt_prob=0.3, max_or_opt_seg_len=3):
    """Builds a jitted `step(carry_in, x0, params, anchor) -> carry_out`
    running `n_gen` generations of the Lamarckian GA+AL loop starting from an
    arbitrary carry (see `build_initial_carry_fn`/`carry_from_population`),
    rather than always cold-randomizing X0 internally -- this is what lets a
    caller resume a previous call's final population/AL-state/PRNG stream
    instead of restarting the search from scratch every time. `x0` (the
    depot), `params` (GraphOfConstraints.view_param_values(), read by any
    constraint referencing a param(id) placeholder), and `anchor`
    (remaining_vertices state, problem.AnchorState) are genuine traced
    arguments, not baked into this compiled function, so the same `step`
    stays valid across depot/param/remaining_vertices drift between calls
    without retracing -- only `problem`'s structure (which this function's
    tracing *does* bake in, and which is now fixed for the whole graph
    regardless of remaining_vertices -- see spec.py's class docstring)
    requires a rebuild when it changes.

    Note: the w/cv_tol anneal (`_score_schedule`) still ramps from gen=0 to
    gen=n_gen-1 *within this call*, recalibrated from carry_in's own F0/CV0
    every time step() runs. Chaining several step() calls together (e.g.
    many short n_gen chunks polled from a driving loop for early-exit
    purposes) re-runs this local anneal each chunk rather than continuing
    one schedule across chunks -- fine for warm-starting the population/
    AL-state between calls, but worth knowing before relying on it for a
    single continuous schedule across chunk boundaries.
    """
    local_refine = make_batched_local_refine(
        problem, outer_iters=outer_iters, inner_maxiter=inner_maxiter,
        rho_growth=rho_growth, rho_max=rho_max,
        lbfgs_history=lbfgs_history, ls_max_trials=ls_max_trials)

    def step(carry_in, x0, params, anchor):
        X0, mu0, lam0, rho_arr0, F0, CV0, key, best_X0, best_F0, best_CV0 = carry_in

        w_auto, cv_tol_auto = _calibrate_score_scale(F0, CV0, w_frac, cv_tol_frac)
        w_val = w_auto if w is None else jnp.asarray(w, dtype=F0.dtype)
        cv_tol_val = cv_tol_auto if cv_tol is None else jnp.asarray(cv_tol, dtype=F0.dtype)

        gen_step = _make_gen_step_fn(problem, local_refine, pop_size, n_gen, mut_sigma, cx_prob,
                                      ox_prob, n_2opt_trials, or_opt_prob, max_or_opt_seg_len,
                                      tournament_k, w_val, cv_tol_val, w_growth, cv_tol_floor_frac, x0, params,
                                      anchor)

        init_carry = (X0, mu0, lam0, rho_arr0, F0, CV0, key, best_X0, best_F0, best_CV0)
        final_carry, _ = jax.lax.scan(lambda c, g: (gen_step(c, g), None),
                                       init_carry, xs=jnp.arange(n_gen), length=n_gen)
        return final_carry

    return jax.jit(step)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

@dataclass
class Result:
    F: np.ndarray
    CV: np.ndarray
    X: np.ndarray
    exec_time: float
    pop: object = field(default=None)


def run_lamarckian_al(problem, anchor, pop_size=30, n_gen=60, seed=1,
                       outer_iters=1, inner_maxiter=20,
                       rho0=1.0, rho_growth=10.0, rho_max=1e6,
                       mut_sigma=0.1, cx_prob=0.9,
                       lbfgs_history=10, ls_max_trials=10,
                       w=None, cv_tol=None, w_frac=1.0, cv_tol_frac=0.05,
                       w_growth=10.0, cv_tol_floor_frac=0.0,
                       ox_prob=0.5, n_2opt_trials=5, or_opt_prob=0.3, max_or_opt_seg_len=3,
                       n_seed_individuals=None, seed_jitter_t=1.0, seed_jitter_wp_frac=0.05,
                       _ga_fn=None, _init_carry=None, x0=None, params=None):
    """`x0` (the full configuration -- state_dim-wide, see problem.
    apply_anchor) defaults to `problem.x0` (zero-padded out from its
    agent-only structural shape, see problem.pad_to_state_dim) when not
    given -- pass it explicitly to run against a real state that may differ
    from whatever `problem` was originally built with (the whole point of
    `_ga_fn`/`step` treating x0 as a live argument rather than a baked-in
    constant; see build_lamarckian_ga's docstring). `params` (GraphOfConstraints.
    view_param_values()) works the same way, defaulting to `problem.params`.
    `anchor` (problem.AnchorState) is required explicitly -- unlike x0/params
    it has no meaningful problem-level default, since silently falling back
    to "everything active" would mask remaining_vertices being dropped by a
    caller by mistake; see EvolutionaryWaypointSolver._compute_anchor
    (mpc.py) for how to build it from a live remaining_vertices set, or
    problem.full_active_anchor for the "every node still remaining"
    bootstrap case."""
    start = time.perf_counter()
    x0_arr = jnp.asarray(x0) if x0 is not None else pad_to_state_dim(
        jnp.asarray(problem.x0).reshape(-1), problem.state_dim)
    params_arr = jnp.asarray(params) if params is not None else jnp.asarray(problem.params)
    ga_fn = _ga_fn if _ga_fn is not None else build_lamarckian_ga(
        problem, pop_size, n_gen, outer_iters=outer_iters, inner_maxiter=inner_maxiter,
        rho_growth=rho_growth, rho_max=rho_max,
        mut_sigma=mut_sigma, cx_prob=cx_prob,
        lbfgs_history=lbfgs_history, ls_max_trials=ls_max_trials,
        w=w, cv_tol=cv_tol, w_frac=w_frac, cv_tol_frac=cv_tol_frac,
        w_growth=w_growth, cv_tol_floor_frac=cv_tol_floor_frac,
        ox_prob=ox_prob, n_2opt_trials=n_2opt_trials, or_opt_prob=or_opt_prob,
        max_or_opt_seg_len=max_or_opt_seg_len)

    if _init_carry is not None:
        carry_in = _init_carry
    else:
        init_fn = build_initial_carry_fn(
            problem, pop_size, anchor, rho0=rho0, n_seed_individuals=n_seed_individuals,
            seed_jitter_t=seed_jitter_t, seed_jitter_wp_frac=seed_jitter_wp_frac)
        carry_in = init_fn(jax.random.PRNGKey(seed))

    carry_out = ga_fn(carry_in, x0_arr, params_arr, anchor)
    jax.block_until_ready(carry_out)
    exec_time = time.perf_counter() - start

    best_X, best_F, best_CV = carry_out[-3], carry_out[-2], carry_out[-1]
    return Result(F=np.array([float(best_F)]), CV=np.array([float(best_CV)]),
                  X=np.asarray(best_X), exec_time=exec_time, pop=carry_out)


def warmup_lamarckian_al(problem, anchor, pop_size, n_gen, outer_iters=1, inner_maxiter=20,
                          rho0=1.0, rho_growth=10.0, rho_max=1e6,
                          mut_sigma=0.1, cx_prob=0.9,
                          lbfgs_history=10, ls_max_trials=10,
                          w=None, cv_tol=None, w_frac=1.0, cv_tol_frac=0.05,
                          w_growth=10.0, cv_tol_floor_frac=0.0,
                          ox_prob=0.5, n_2opt_trials=5, or_opt_prob=0.3, max_or_opt_seg_len=3,
                          n_seed_individuals=None, seed_jitter_t=1.0, seed_jitter_wp_frac=0.05,
                          x0=None, params=None):
    """`x0` (the full configuration) defaults to `problem.x0` (zero-padded,
    see run_lamarckian_al's docstring) when not given; `params` likewise
    defaults to `problem.params`; `anchor` is required -- see
    run_lamarckian_al's docstring."""
    ga_fn = build_lamarckian_ga(
        problem, pop_size, n_gen, outer_iters=outer_iters, inner_maxiter=inner_maxiter,
        rho_growth=rho_growth, rho_max=rho_max,
        mut_sigma=mut_sigma, cx_prob=cx_prob,
        lbfgs_history=lbfgs_history, ls_max_trials=ls_max_trials,
        w=w, cv_tol=cv_tol, w_frac=w_frac, cv_tol_frac=cv_tol_frac,
        w_growth=w_growth, cv_tol_floor_frac=cv_tol_floor_frac,
        ox_prob=ox_prob, n_2opt_trials=n_2opt_trials, or_opt_prob=or_opt_prob,
        max_or_opt_seg_len=max_or_opt_seg_len)
    init_fn = build_initial_carry_fn(
        problem, pop_size, anchor, rho0=rho0, n_seed_individuals=n_seed_individuals,
        seed_jitter_t=seed_jitter_t, seed_jitter_wp_frac=seed_jitter_wp_frac)
    x0_arr = jnp.asarray(x0) if x0 is not None else pad_to_state_dim(
        jnp.asarray(problem.x0).reshape(-1), problem.state_dim)
    params_arr = jnp.asarray(params) if params is not None else jnp.asarray(problem.params)

    start = time.perf_counter()
    carry_in = init_fn(jax.random.PRNGKey(0))
    carry_out = ga_fn(carry_in, x0_arr, params_arr, anchor)
    jax.block_until_ready(carry_out)
    compile_time = time.perf_counter() - start
    return compile_time, ga_fn, carry_out

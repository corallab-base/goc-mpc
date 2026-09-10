r"""`SmallContinuousVRPSolver`: `LamarckianGA` (lamarckian_ga.py) with the
DISCRETE part of the problem -- agent ASSIGNMENT (`assign`), conditional
binaries (`cond_binary`), node ORDER (`t`) and per-projection BRANCH
selection (`proj_branch`) -- solved EXACTLY, in one shot, by the vectorized
discrete solver (`logic_based_benders_solver.dp_master_jax.make_dp_master_
jax`) instead of being evolutionarily searched by crossover/mutation.

`dp_master_jax` enumerates every `(assignment, aux, ordering)` skeleton,
prices each with the same per-agent obstacle-aware `edge_cost_fn` kernel.py's
routing objective uses (a branch Viterbi for avg/minmax, the coupled
makespan forward pass otherwise), reduces `min` over orderings, and returns
the `k` cheapest DISTINCT skeletons -- each as a ready-to-splice genome
fragment `(assign one-hot, cond_binary, node rank `t`, proj_branch one-hot)`
plus its analytically-resolved waypoints.

`_ask` is deliberately minimal: MASTER + LOCAL REFINE, nothing else. It
seeds generation 0 from those `k` skeletons (member `i` -> skeleton `i % k`,
cost-ordered; an infeasible skeleton falls back to the best feasible one),
taking each member's discrete genome AND its waypoints from the skeleton,
then every generation runs the Lamarckian AL+L-BFGS `local_refine` on the
carried population. NO tournament, NO crossover, NO mutation, NO routing
2-opt. The discrete genome is frozen after generation 0; only the
continuous `wp`/`psi` and the AL multiplier state evolve.

`_tell` (inherited, mu+lambda elitist truncation) keeps the better of each
parent and its refined self, so a bad discrete skeleton is not a dead end:
the top-`k` skeletons all sit in the population at once and the best
continuously-feasible one wins ("keep the top-k members with different
assignments, optimise the continuous problem for each, reject the infeasible
ones, keep the best").

Still a `LamarckianGA` subclass: same Params / State / `_tell`, same
`local_refine`. Reintroducing search over the continuous axes, or
resampling skeletons per generation, are later steps.

Scope: whatever `make_dp_master_jax` accepts -- a jnp-traceable
`edge_cost_fn`, an assignment space within `max_assign_combos`, orderings
within `max_orders`, branch combos within `max_branch_combos`, and no two
multi-branch projections writing the same node. It raises (not silently
degrades) otherwise.

Known limitations:
  - The population is seeded once, at generation 0. Across MPC cycles that
    reuse the carried population the skeletons are NOT re-derived for the
    new `x0` -- fine on a static / teleport scene (the discrete optimum is
    `x0`-stable), stale if the scene moves substantially or the remaining
    subgraph changes shape.
  - The `wp_template` fed to the skeleton search is built once with an
    `x0`-at-origin fallback (`warm_start_wp` with a zero `x0`); a node that
    is routed through but carries no projection at all -- rare in the
    single-chain scenes this class targets -- is priced from the origin in
    the initial skeleton ranking (`local_refine` then fixes its columns).
  - IK-feasibility is not checked (`make_dp_master_jax` inherits
    `solve_dp_master`'s limitation): analytic IK returns a best-effort `q`
    for an out-of-reach target, so a skeleton can look cheap yet be
    continuously infeasible. Downstream continuous feedback -- rejecting
    such an individual on its real CV and letting the next skeleton win --
    is exactly what the top-`k` population is for.
"""

import jax
import jax.numpy as jnp
import numpy as np

from .lamarckian_ga import LamarckianGA
from .evosax_ga import _split_genome, _join_genome
from .solver import _write_wp_batch_jax, _write_psi_batch_jax
from ..logic_based_benders_solver.dp_master_jax import make_dp_master_jax
from ..logic_based_benders_solver.structure import warm_start_wp

jax.config.update("jax_enable_x64", True)


class SmallContinuousVRPSolver(LamarckianGA):
    """See module docstring. Same construction / Params / State / `_tell` as
    `LamarckianGA` -- only `_ask` differs: discrete axes + waypoints come
    from `dp_master_jax`, then `local_refine`."""

    def __init__(self, population_size, solution, problem,
                 max_assign_combos=4096, max_orders=20000, max_branch_combos=4096,
                 **kwargs):
        super().__init__(population_size, solution, problem, **kwargs)
        self._dp = make_dp_master_jax(
            problem, objective=problem.objective, edge_cost_fn=problem.edge_cost_fn,
            max_assign_combos=max_assign_combos, max_orders=max_orders,
            max_branch_combos=max_branch_combos)
        # one skeleton per population member, capped at the number of
        # distinct (assignment, aux) skeletons that actually exist.
        self._n_skeletons = int(min(population_size, self._dp.NC * self._dp.NA))
        self._skeleton_fn = self._dp.skeleton_grid_fn(self._n_skeletons)
        # static wp template for the skeleton search -- see module docstring
        # (nodes unreachable from any projection anchor fall back to x0=0).
        self._wp_template = jnp.asarray(
            warm_start_wp(problem, np.zeros(problem.state_dim)))

    def _seed_population(self, params, pop_size):
        """`(disc (pop, wp_offset), wp (pop, n_nodes*state_dim))` -- for
        member `i`, cost-ordered skeleton `i % n_skeletons`'s discrete genome
        (`assign | cond_binary | proj_branch | t`) and its analytically-
        resolved waypoints. An infeasible skeleton is remapped to the best
        feasible one."""
        problem = self.problem
        J = problem.n_agents
        X0 = jnp.broadcast_to(params.x0[None, :], (J, problem.state_dim))
        anchor = params.anchor
        obj_s, assign_s, cond_s, t_s, pb_s, wp0_s, _cell = self._skeleton_fn(
            params.problem_params, self._wp_template, params.x0, X0,
            anchor.node_active, anchor.var_committed, anchor.var_anchor)
        Sn = obj_s.shape[0]
        feas = jnp.isfinite(obj_s)
        remap = jnp.where(feas, jnp.arange(Sn), jnp.argmax(feas))     # -> best feasible
        src = remap[jnp.arange(pop_size) % Sn]                        # (pop,)

        assign_flat = assign_s[src].reshape(pop_size, -1)             # (pop, n_assign_vars)
        disc = jnp.concatenate(
            [assign_flat, cond_s[src], pb_s[src], t_s[src]], axis=1)  # (pop, wp_offset)
        wp = wp0_s[src].reshape(pop_size, -1)                         # (pop, n_nodes*state_dim)
        return disc, wp

    def _ask(self, key, state, params):
        problem = self.problem
        pop_size = self.population_size
        wp_offset, psi_offset = problem.wp_offset, problem.psi_offset

        X, mu, lam, rho = _split_genome(problem, state.population)

        # Generation 0: discrete genome + waypoints come from the exact
        # dp_master_jax skeletons. Later generations carry each individual's
        # own frozen skeleton and just keep refining its continuous part.
        seed = state.generation_counter == 0
        skel_disc, skel_wp = self._seed_population(params, pop_size)
        child_X = X
        child_X = child_X.at[:, :wp_offset].set(
            jnp.where(seed, skel_disc, X[:, :wp_offset]))
        child_X = child_X.at[:, wp_offset:psi_offset].set(
            jnp.where(seed, skel_wp, X[:, wp_offset:psi_offset]))

        assign, cond_binary, proj_branch, t, wp0, psi0 = problem._extract_batch(child_X)
        wp_star, psi_star, off_mu, off_lam, off_rho = self.local_refine(
            wp0, psi0, assign, cond_binary, proj_branch, t, mu, lam, rho,
            params.x0, params.problem_params, params.anchor, params.cv_tol)
        off_X = _write_wp_batch_jax(problem, child_X, wp_star)
        off_X = _write_psi_batch_jax(problem, off_X, psi_star)

        child_genome = _join_genome(off_X, off_mu, off_lam, off_rho)
        return child_genome, state

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

The top-`k` `(assignment, aux)` skeletons are enumerated ONCE, in `init`
(from `problem.x0` / the initial `anchor`), and stashed on the algorithm
`State` (`seed_disc`, `seed_wp`) -- NOT re-derived per `_ask` and NOT
re-derived across MPC cycles that reuse the carried population. This is an
interim step, not the intended steady state: the eventual behavior is to
periodically evict the population's bottom-`k` members and reset them from
a fresh top-`k` discrete-solver call, so a moved `x0` / shrunk remaining
subgraph gets fed back in without discarding the whole carried population.
See "Known limitations" for what's missing until that lands.

`_ask` is deliberately minimal: SEED (generation 0 only) + LOCAL REFINE,
nothing else. Generation 0 splices each member's frozen discrete genome and
analytic waypoints from skeleton `i % k` (cost-ordered; an infeasible
skeleton falls back to the best feasible one) and zeros its `psi`; every
generation then runs the Lamarckian AL+L-BFGS `local_refine` on the carried
population. NO tournament, NO crossover, NO mutation, NO routing 2-opt. The
discrete genome is frozen after generation 0; only the continuous
`wp`/`psi` and the AL multiplier state evolve.

`local_refine` only ever moves the wp columns NO projection writes
(`problem.wp_free_idx`, via `gather_free_wp`/`scatter_free_wp`) plus `psi`
-- a projection-pinned column carries no real degree of freedom. After the
refine, `_ask` re-runs `apply_projections` with the just-optimized `psi` so
every particle's pinned wp columns land exactly on their analytic targets
before the genome is stored (matters only when a projection actually
declares `psi`; a no-`psi` scene like `dual_ur5e_block_stacking` gets the
identical result either way).

`_tell` (inherited, mu+lambda elitist truncation) keeps the better of each
parent and its refined self, so a bad discrete skeleton is not a dead end:
the top-`k` skeletons all sit in the population at once and the best
continuously-feasible one wins ("keep the top-k members with different
assignments, optimise the continuous problem for each, reject the infeasible
ones, keep the best").

Still a `LamarckianGA` subclass: same Params / `_tell`, same
`local_refine`. `State` is extended with the once-enumerated skeletons
(`seed_disc`/`seed_wp`). Reintroducing search over the continuous axes, or
periodic bottom-`k` eviction/reseeding (see above), are later steps.

Scope: whatever `make_dp_master_jax` accepts -- a jnp-traceable
`edge_cost_fn`, an assignment space within `max_assign_combos`, orderings
within `max_orders`, branch combos within `max_branch_combos`, and no two
multi-branch projections writing the same node. It raises (not silently
degrades) otherwise.

Known limitations:
  - The skeletons are enumerated ONCE, in `init` (against `problem.x0` and
    the `init`-time `anchor`), and reused for every `_ask` and every
    subsequent MPC cycle that resumes the carried population -- they are
    NOT re-derived for a new `x0` or a shrunk remaining subgraph. Correct
    only until the intended periodic bottom-`k` eviction/reseeding (module
    docstring) lands; until then, a scene whose `x0` moves or whose
    remaining subgraph shrinks substantially runs on an increasingly stale
    discrete skeleton.
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

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from .lamarckian_ga import LamarckianGA, State as _LamarckianState
from .evosax_ga import _split_genome, _join_genome
from .problem import jit_apply_projections
from .solver import _write_wp_batch_jax, _write_psi_batch_jax
from ..logic_based_benders_solver.dp_master_jax import make_dp_master_jax
from ..logic_based_benders_solver.structure import warm_start_wp

jax.config.update("jax_enable_x64", True)


@struct.dataclass
class State(_LamarckianState):
    # The top-`k` skeletons, enumerated ONCE in `init` and remapped one per
    # population member (member `i` -> cost-ordered skeleton `i % k`, an
    # infeasible one falling back to the best feasible). `_ask` splices these
    # verbatim at generation 0 and never touches them again -- see module
    # docstring.
    seed_disc: jax.Array   # (pop, wp_offset)   assign | cond_binary | proj_branch | t
    seed_wp: jax.Array      # (pop, n_nodes * state_dim)   analytic waypoints


class SmallContinuousVRPSolver(LamarckianGA):
    """See module docstring. Same construction / Params / `_tell` as
    `LamarckianGA`; `State` is extended with the once-enumerated skeletons
    (`seed_disc`/`seed_wp`). `init` enumerates them; `_ask` seeds generation
    0 from them, then runs `local_refine` on the continuous axes only."""

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

    def _init(self, key, params):
        base = super()._init(key, params)
        problem = self.problem
        pop = self.population_size
        return State(
            **{f.name: getattr(base, f.name) for f in dataclasses.fields(base)},
            seed_disc=jnp.zeros((pop, problem.wp_offset)),
            seed_wp=jnp.zeros((pop, problem.n_nodes * problem.state_dim)),
        )

    def init(self, key, population, fitness, params):
        """Enumerate the top-`k` `(assignment, aux)` skeletons ONCE here --
        against the live `params.x0`/`problem_params`/`anchor` threaded in by
        evosax_ga.py's `build_initial_carry_fn` -- and stash them (remapped
        one per population member) on the `State` for every later `_ask` and
        every resumed MPC cycle. See module docstring."""
        state = super().init(key, population, fitness, params)
        seed_disc, seed_wp = self._seed_population(params, self.population_size)
        return state.replace(seed_disc=seed_disc, seed_wp=seed_wp)

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
        wp_offset, psi_offset = problem.wp_offset, problem.psi_offset

        X, mu, lam, rho = _split_genome(problem, state.population)

        # Generation 0: frozen discrete genome + analytic waypoints come from
        # the skeletons `init` already enumerated (state.seed_disc/seed_wp);
        # psi starts at zero. Every later generation carries each individual's
        # own frozen skeleton and only refines its continuous part.
        seed = state.generation_counter == 0
        child_X = X
        child_X = child_X.at[:, :wp_offset].set(
            jnp.where(seed, state.seed_disc, X[:, :wp_offset]))
        child_X = child_X.at[:, wp_offset:psi_offset].set(
            jnp.where(seed, state.seed_wp, X[:, wp_offset:psi_offset]))
        child_X = child_X.at[:, psi_offset:].set(
            jnp.where(seed, jnp.zeros_like(X[:, psi_offset:]), X[:, psi_offset:]))

        assign, cond_binary, proj_branch, t, wp0, psi0 = problem._extract_batch(child_X)
        # local_refine moves only the non-projection wp columns
        # (problem.wp_free_idx) and psi -- a projection-pinned column has no
        # real freedom (apply_projections overwrites it).
        wp_star, psi_star, off_mu, off_lam, off_rho = self.local_refine(
            wp0, psi0, assign, cond_binary, proj_branch, t, mu, lam, rho,
            params.x0, params.problem_params, params.anchor, params.cv_tol)
        # Re-resolve every projection-pinned column against the just-optimized
        # psi so each stored particle's waypoints are internally consistent
        # (a no-op on pinned values for a scene whose projections declare no
        # psi, e.g. dual_ur5e_block_stacking). jit_apply_projections (cached,
        # compiled once per problem -- problem.py), not the eager
        # apply_projections: this runs once per _ask, not in a hot per-
        # iteration loop, so it wants the cheap-after-first-compile jitted
        # path rather than eager per-call tracing overhead.
        wp_star = jit_apply_projections(problem)(
            wp_star, psi_star, proj_branch, params.problem_params, assign, cond_binary, t,
            params.anchor.node_active, params.x0, params.anchor.var_committed, params.anchor.var_anchor)
        off_X = _write_wp_batch_jax(problem, child_X, wp_star)
        off_X = _write_psi_batch_jax(problem, off_X, psi_star)

        child_genome = _join_genome(off_X, off_mu, off_lam, off_rho)
        return child_genome, state

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
re-derived across MPC cycles that reuse the carried population; `_ask`
only ever splices `state.seed_disc`/`seed_wp` at true generation 0. A
moved `x0` / shrunk remaining subgraph is instead fed back in by `reseed`
(below), called once per external `step()` call (the `reseed` duck-typed
hook evosax_ga.py's `step` checks for): it periodically evicts the
population's `n_evict` worst members (hard CV-dominant score) and
replaces each with a fresh top-`k` discrete-solver candidate re-derived
against the LIVE `x0`/`anchor` -- but only when that candidate actually
beats the incumbent it would replace, so an unchanged scene's already-
converged individuals are never needlessly reset. See `reseed`'s own
docstring for the full mechanism and why the beat-the-incumbent gate
matters.

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
(`seed_disc`/`seed_wp`). Reintroducing search over the continuous axes is
a later step; periodic bottom-`k` eviction/reseeding is `reseed`, above.

Scope: whatever `make_dp_master_jax` accepts -- a jnp-traceable
`edge_cost_fn`, an assignment space within `max_assign_combos`, orderings
within `max_orders`, branch combos within `max_branch_combos`, and no two
multi-branch projections writing the same node. It raises (not silently
degrades) otherwise.

Known limitations:
  - `_ask`'s own generation-0 seed (`state.seed_disc`/`seed_wp`) is still
    the ONE-TIME `init`-time skeleton search -- correct at true generation
    0 by construction. `reseed` bounds later staleness by re-evaluating
    the WHOLE population fresh every call and re-deriving `n_evict`
    candidates for whichever slots currently rank worst, so a slot whose
    skeleton has actually gone bad (the motivating moved-`x0` case) always
    surfaces into `n_evict` and gets challenged -- but only the `n_evict`
    currently-worst slots get a fresh challenger per call; a slot that is
    merely mediocre rather than worst is re-scored, not re-challenged,
    until it is.
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
from .solver import _write_wp_batch_jax, _write_psi_batch_jax, _evaluate_population_jax
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
                 reseed_frac=0.2, reseed_rho0=1.0, reseed_cv_tol=1e-4,
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
        # `reseed`'s own default eviction count/AL-reset value -- see that
        # method's docstring.
        self._n_evict = int(max(1, min(population_size, round(reseed_frac * population_size))))
        self._reseed_rho0 = reseed_rho0
        # Fixed (not annealed, unlike `_combined_score`'s own `cv_tol`) hard-
        # score slack -- see reseed's docstring for why a bare CV-dominant
        # comparison, with no slack at all, isn't what's wanted here. 1e-4
        # matches _calc_cv_jax's own eq_eps: the same "quantization floor" a
        # closed-form/analytic residual already carries elsewhere in this
        # module, not a new number invented for this purpose.
        self._reseed_cv_tol = reseed_cv_tol

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

    def reseed(self, key, state, params, n_evict=None):
        """Periodic bottom-`n_evict` eviction + fresh top-k discrete reseed
        -- the "intended steady state" the module docstring names as missing
        from the once-in-`init` skeleton search. Called once per EXTERNAL
        `step()` call (evosax_ga.py's `step`, via the duck-typed `reseed`
        hook it checks for via `getattr`/`hasattr`, exactly like that
        module's own `_live_params`), NOT once per generation inside the
        n_gen-generation `lax.scan`: `params.x0`/`anchor`/`problem_params`
        are invariant across that whole scan (only `w`/`cv_tol` anneal
        internally), so a fresh discrete search there would return the
        identical skeletons every generation for `n_gen`x the cost.

        Ranks the CURRENT population by a hard CV-dominant score (`F + 1e6
        * max(0, CV - reseed_cv_tol)` -- mirrors solver.py's own
        `_carry_from_population` seed-selection, not the softly-annealed
        `_combined_score` `_tell` uses for tournament/elitism: this decision
        wants a confident feasible-beats-infeasible split, not something
        still mid-anneal). `reseed_cv_tol` (default 1e-4, matching
        _calc_cv_jax's own eq_eps) is exactly this score's `cv_tol`: two
        candidates both under it are compared on `F` alone, so an
        incumbent whose only "violation" is analytic/quantization-floor
        noise below that line isn't evicted just because some other
        candidate randomly rolled a CV a few floating-point ulps lower --
        without it, CV noise this small would dominate every ranking
        decision here even though it's invisible to `_combined_score`'s own
        softly-annealed ranking everywhere else in this class. Takes the
        `n_evict` worst slots by that score, and re-runs `_seed_population`
        against the LIVE `params.x0`/`anchor` (unlike `state.seed_disc`/
        `seed_wp`, frozen from `init`, `_seed_population` was always
        parameterized by live params -- it was just never called again)
        to get `n_evict` fresh top-k discrete candidates.

        Each fresh candidate starts with `mu=0`/`lam=0`/`rho=reseed_rho0`
        (like a true generation-0 individual -- its multipliers belong to a
        DIFFERENT discrete skeleton's active-constraint set than whatever
        occupied that slot before, so the old slot's mu/lam/rho carries no
        meaningful warm-start, unlike `_carry_from_population`'s own
        same-skeleton small-perturbation case) and gets exactly one
        `local_refine` call -- the same one every other offspring gets this
        generation -- before it's allowed to compete.

        A fresh candidate only REPLACES its slot if it actually beats the
        incumbent on the same hard score; otherwise the incumbent is left
        untouched. This is not optional bookkeeping: without it, an
        unchanged scene would re-enumerate the SAME best skeleton every
        call and unconditionally stomp an already-converged incumbent's
        accumulated local_refine progress and AL ramp back to a fresh,
        under-refined start -- at population_size=1 in particular, that
        would mean never letting anything converge. The gate is a cheap
        pre-filter, not the final word: a candidate that wins the gate but
        is still not fully refined gets the same `n_gen` further `_ask`/
        `_tell` generations as everyone else, later this call, to prove
        itself for real."""
        problem = self.problem
        pop = self.population_size
        n_evict = self._n_evict if n_evict is None else n_evict
        n_evict = int(max(0, min(n_evict, pop)))
        if n_evict == 0:
            return state

        def hard_score(F, CV):
            return F + 1e6 * jnp.maximum(0.0, CV - self._reseed_cv_tol)

        X_old = _split_genome(problem, state.population)[0]
        F_old, CV_old = _evaluate_population_jax(
            problem, X_old, params.x0, params.problem_params, params.anchor)
        hard_old = hard_score(F_old, CV_old)
        worst = jnp.argsort(-hard_old)[:n_evict]                      # worst-first slots

        seed_disc, seed_wp = self._seed_population(params, n_evict)
        wp_offset, psi_offset = problem.wp_offset, problem.psi_offset
        cand_X = jnp.zeros((n_evict, problem.n_var))
        cand_X = cand_X.at[:, :wp_offset].set(seed_disc)
        cand_X = cand_X.at[:, wp_offset:psi_offset].set(seed_wp)      # psi columns stay zero

        n_eq, n_ineq = problem.n_eq_constr, problem.n_ieq_constr
        cand_mu = jnp.zeros((n_evict, n_eq))
        cand_lam = jnp.zeros((n_evict, n_ineq))
        cand_rho = jnp.full((n_evict,), self._reseed_rho0)

        assign, cond_binary, proj_branch, t, wp0, psi0 = problem._extract_batch(cand_X)
        wp_star, psi_star, cand_mu, cand_lam, cand_rho = self.local_refine(
            wp0, psi0, assign, cond_binary, proj_branch, t, cand_mu, cand_lam, cand_rho,
            params.x0, params.problem_params, params.anchor, params.cv_tol)
        wp_star = jit_apply_projections(problem)(
            wp_star, psi_star, proj_branch, params.problem_params, assign, cond_binary, t,
            params.anchor.node_active, params.x0, params.anchor.var_committed, params.anchor.var_anchor)
        cand_X = _write_wp_batch_jax(problem, cand_X, wp_star)
        cand_X = _write_psi_batch_jax(problem, cand_X, psi_star)
        cand_genome = _join_genome(cand_X, cand_mu, cand_lam, cand_rho)

        F_cand, CV_cand = _evaluate_population_jax(
            problem, cand_X, params.x0, params.problem_params, params.anchor)
        hard_cand = hard_score(F_cand, CV_cand)

        take_cand = (hard_cand < hard_old[worst])[:, None]
        new_slots = jnp.where(take_cand, cand_genome, state.population[worst])
        population = state.population.at[worst].set(new_slots)
        return state.replace(population=population)

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

"""Custom evosax `PopulationBasedAlgorithm` reproducing solver.py's own
`_make_gen_step_fn` operators exactly (tournament selection, BLX-alpha
crossover, order-crossover on `t`, the `wp_mut_scale`-gated freeze of `wp`
back to the fitter parent's own value, mu/lam/rho carried verbatim from
the fitter parent, the Lamarckian AL+L-BFGS local refine, the routing
2-opt/Or-opt local search, and mu+lambda elitist truncation), so
`build_evosax_ga(problem, LamarckianGA, ..., algo_kwargs=dict(problem=
problem))` (evosax_ga.py) is a faithful, ask/tell-shaped stand-in for
`run_lamarckian_al`. Any OTHER evosax algorithm (`SimpleGA`, `PSO`, ...) is
a genuinely different, equally pluggable alternative that only ever gets
the plain merit/CV-based fitness evosax_ga.py computes generically -- see
that module's docstring.

Where the Lamarckian local refine and routing local search live: inside
`_ask`, not in a generic post-`ask` evaluate step and not in `_tell`.
`ask` is where a candidate SOLUTION gets constructed -- for this
algorithm, a candidate's true `wp`/`psi`/`t` values ARE the gradient-
refined/locally-searched ones, not whatever raw crossover+mutation first
produced, exactly the way evosax's own PSO computes its candidate via a
velocity update inside `_ask` rather than as a generic operator applied
afterward. Putting it after `ask` would force every OTHER `algo_cls` used
through evosax_ga.py to pay for it (the shared evaluate step is meant to
stay algorithm-agnostic); putting it in `_tell` would hide "what a
candidate solution actually is" inside book-keeping that's only supposed
to decide who survives.

Live per-call/per-generation values (`x0`, the problem's `params`, the
current `anchor`, and the annealed `w`/`cv_tol`) that the local refine/
local search/scoring need but the plain tournament+BLX+OX operators don't
-- travel in via evosax's own `params` argument (`Params`, below),
refreshed each call/generation by evosax_ga.py's `step` before every
`ask`/`tell` -- the same channel evosax already uses for per-call
hyperparameters (see e.g. PSO's own `Params.inertia_coeff` etc.), just
extended to also carry these live values since evosax has no separate
channel for that.

Why `_ask`/`_tell` recompute raw (F, CV) fresh via `_evaluate_population_
jax` instead of trusting `state.fitness`/the `fitness` argument evosax's
own `tell()` passes through: `w`/`cv_tol` anneal generation-to-generation
(`_score_schedule`, evosax_ga.py) AND get recalibrated fresh every
EXTERNAL `step()` call (`_calibrate_score_scale`, from that call's own
carried-in population), so a combined score computed at one generation (or
call) is not comparable to one computed at a later generation/call under a
different `w`/`cv_tol` -- exactly why solver.py's own `_make_gen_step_fn`
carries the RAW `(F, CV)` in its carry and recomputes `_combined_score`
fresh every generation (`S = _combined_score(F, CV, w, cv_tol)`,
`pool_S = _combined_score(pool_F, pool_CV, w, cv_tol)`, and, for its own
best-so-far, `best_S = _combined_score(best_F, best_CV, w, cv_tol)`)
rather than reusing a stored score from a previous generation. evosax's
`state.fitness`/the `fitness` handed to `tell()` are exactly such frozen,
schedule-specific values -- correct for comparing SAME-generation
offspring against each other (tournament/pooling, above), but wrong
against anything scored under a DIFFERENT schedule.

This is why `_tell` below also carries its OWN `best_F`/`best_CV` (raw,
mirroring solver.py's `best_X`/`best_F`/`best_CV`) instead of trusting
evosax's base-class `best_solution`/`best_fitness` tracking
(`EvolutionaryAlgorithm.tell`'s `update_best_solution_and_fitness`, which
runs BEFORE `_tell` and compares a fresh offspring score against a stale
`state.best_fitness` frozen under whatever schedule was live whenever it
was last set -- confirmed empirically: across many small `step()` calls,
as production actually drives this solver, `state.best_solution` got
stuck reporting an infeasible individual (CV~4) generations after the
population's own current best had already reached CV=0). `_tell`
overwrites `best_solution`/`best_fitness` with its own correct,
freshly-rescaled comparison, so evosax's built-in tracking is computed
(harmlessly) but never relied upon.
"""

import jax
import jax.numpy as jnp
from flax import struct

from evosax.algorithms.population_based.base import (
    Params as BaseParams,
    PopulationBasedAlgorithm,
    State as BaseState,
    metrics_fn,
)
from evosax.core.fitness_shaping import identity_fitness_shaping_fn

from .evosax_ga import _split_genome, _join_genome
from .problem import apply_anchor, jit_apply_projections, full_active_anchor
from .solver import (
    make_batched_local_refine,
    _routing_local_search_batched,
    _write_wp_batch_jax,
    _write_psi_batch_jax,
    _write_t_batch_jax,
    _tournament_select_jax,
    _ox_crossover_batched,
    _rank_jax,
    _evaluate_population_jax,
    _evaluate_projection_cv_jax,
    _combined_score,
)

jax.config.update("jax_enable_x64", True)


@struct.dataclass
class State(BaseState):
    # Raw best-so-far (F, CV) -- see module docstring for why `_tell`
    # tracks these itself instead of trusting the base class's own
    # best_solution/best_fitness.
    best_F: jax.Array
    best_CV: jax.Array
    # Read-only post-projection CV (_evaluate_projection_cv_jax) for the
    # SAME best_solution, updated in lockstep with best_F/best_CV in
    # `_tell` -- tracked here (computed once per generation, inside this
    # already-jitted step) so a caller wanting it (mpc.py's solve(), for
    # its own diagnostic self._last_CV_proj) can just read it off the
    # returned state instead of re-running the eager, uncached
    # `_evaluate_projection_cv_jax(problem, best_X[None, :], ...)` call
    # that used to pay a full XLA retrace/recompile on every external
    # solve() call (confirmed via profiling: ~1.4s/cycle, dwarfing the
    # rest of the search).
    best_CV_proj: jax.Array


@struct.dataclass
class Params(BaseParams):
    cx_prob: float
    mut_sigma: float
    ox_prob: float
    # Live values, refreshed by evosax_ga.py's `step` every call/generation
    # -- see module docstring. Their values here (in `_default_params`)
    # only need to be STRUCTURALLY valid (right shape/dtype); they're
    # always overwritten via `.replace(...)` before first real use.
    x0: jax.Array
    problem_params: jax.Array
    anchor: object
    w: float
    cv_tol: float


class LamarckianGA(PopulationBasedAlgorithm):
    """Reproduces solver.py's `_make_gen_step_fn` ask/tell exactly (see
    module docstring)."""

    def __init__(self, population_size, solution, problem, tournament_k=2, wp_mut_scale=0.0,
                 outer_iters=1, inner_maxiter=20, rho_growth=10.0, rho_max=1e6,
                 lbfgs_history=10, ls_max_trials=10, optimizer=None,
                 n_2opt_trials=5, or_opt_prob=0.3, max_or_opt_seg_len=3,
                 fitness_shaping_fn=identity_fitness_shaping_fn, metrics_fn=metrics_fn):
        super().__init__(population_size, solution, fitness_shaping_fn, metrics_fn)
        self.problem = problem
        self.tournament_k = tournament_k
        self.wp_mut_scale = wp_mut_scale
        # Local-search knobs that fix array shapes/scan lengths (n_2opt_
        # trials, max_or_opt_seg_len) must stay plain Python ints, not
        # traced Params fields -- see _routing_local_search_batched/
        # _or_opt_batched, which size a jax.random.split/lax.scan from
        # them. or_opt_prob is only ever compared against, so it could be
        # a Params field, but is kept alongside its shape-fixing siblings
        # here for the same reason tournament_k/wp_mut_scale are: they're
        # solver-tuning choices fixed for this algorithm instance, not
        # values any caller anneals generation-to-generation.
        self.n_2opt_trials = n_2opt_trials
        self.or_opt_prob = or_opt_prob
        self.max_or_opt_seg_len = max_or_opt_seg_len
        self.local_refine = make_batched_local_refine(
            problem, outer_iters=outer_iters, inner_maxiter=inner_maxiter,
            rho_growth=rho_growth, rho_max=rho_max,
            lbfgs_history=lbfgs_history, ls_max_trials=ls_max_trials, optimizer=optimizer)

    @property
    def _default_params(self):
        problem = self.problem
        return Params(
            cx_prob=0.9, mut_sigma=0.1, ox_prob=0.5,
            x0=jnp.zeros((problem.state_dim,)),
            problem_params=jnp.zeros((problem.n_params,)),
            anchor=full_active_anchor(problem),
            w=1.0,
            cv_tol=1.0,
        )

    def _init(self, key, params):
        genome_dim = self.num_dims
        return State(
            population=jnp.full((self.population_size, genome_dim), jnp.nan),
            fitness=jnp.full((self.population_size,), jnp.inf),
            best_solution=jnp.full((genome_dim,), jnp.nan),
            best_fitness=jnp.inf,
            best_F=jnp.inf,
            best_CV=jnp.inf,
            best_CV_proj=jnp.inf,
            generation_counter=0,
        )

    def init(self, key, population, fitness, params):
        """Seeds `best_F`/`best_CV`/`best_CV_proj` (raw) for the SAME
        individual the base class's `init` already picked as
        `best_solution`/`best_fitness` (`argmin` over `fitness`,
        NaN-safe) -- `params.x0`/`problem_params`/`anchor` must already be
        the real, live values by this point (see evosax_ga.py's
        `build_initial_carry_fn`, which threads them in before calling
        `algo.init`), not this class's own placeholder `_default_params`."""
        state = super().init(key, population, fitness, params)
        problem = self.problem
        X0 = _split_genome(problem, population)[0]
        F0, CV0 = _evaluate_population_jax(problem, X0, params.x0, params.problem_params, params.anchor)
        CV_proj0 = _evaluate_projection_cv_jax(problem, X0, params.x0, params.problem_params, params.anchor)
        best_idx = jnp.argmin(jnp.where(jnp.isnan(fitness), jnp.inf, fitness))
        return state.replace(best_F=F0[best_idx], best_CV=CV0[best_idx], best_CV_proj=CV_proj0[best_idx])

    def _ask(self, key, state, params):
        problem = self.problem
        pop_size = self.population_size
        n_var = problem.n_var
        wp_offset, t_offset = problem.wp_offset, problem.t_offset
        n_nodes = problem.n_nodes

        X, mu, lam, rho = _split_genome(problem, state.population)
        # Recomputed fresh from raw (F, CV), not `state.fitness` -- see
        # module docstring for why a stored/passed-through combined score
        # isn't comparable across an annealed w/cv_tol schedule.
        F_parent, CV_parent = _evaluate_population_jax(
            problem, X, params.x0, params.problem_params, params.anchor)
        S = _combined_score(F_parent, CV_parent, params.w, params.cv_tol)

        key, k_p1, k_p2, k_cx_mask, k_cx_alpha, k_mut, k_ox_mask, k_ox, k_2opt = jax.random.split(key, 9)

        p1 = _tournament_select_jax(k_p1, S, pop_size, k=self.tournament_k)
        p2 = _tournament_select_jax(k_p2, S, pop_size, k=self.tournament_k)
        parent = jnp.where(S[p1] < S[p2], p1, p2)

        do_cx = jax.random.uniform(k_cx_mask, (pop_size,)) < params.cx_prob
        alpha = jax.random.uniform(k_cx_alpha, (pop_size, n_var), minval=-0.5, maxval=1.5)
        blended = X[p1] + alpha * (X[p2] - X[p1])
        child_X = jnp.where(do_cx[:, None], blended, X[parent])
        xl, xu = jnp.asarray(problem.xl), jnp.asarray(problem.xu)
        noise = jax.random.normal(k_mut, (pop_size, n_var), dtype=X.dtype) * params.mut_sigma
        child_X = child_X + noise * (xu - xl)
        child_X = jnp.clip(child_X, xl, xu)

        if self.wp_mut_scale < 1.0:
            parent_wp = X[parent][:, wp_offset:]
            child_X = child_X.at[:, wp_offset:].set(
                parent_wp + self.wp_mut_scale * (child_X[:, wp_offset:] - parent_wp))

        t_p1 = X[p1][:, t_offset:wp_offset]
        t_p2 = X[p2][:, t_offset:wp_offset]
        perm1, perm2 = jnp.argsort(t_p1, axis=1), jnp.argsort(t_p2, axis=1)
        child_perm_ox = _ox_crossover_batched(k_ox, perm1, perm2)
        row_idx = jnp.arange(pop_size)[:, None]
        rank = jnp.broadcast_to(jnp.arange(n_nodes), (pop_size, n_nodes))
        t_ox = jnp.zeros((pop_size, n_nodes), dtype=X.dtype).at[row_idx, child_perm_ox].set(rank)
        do_ox = jax.random.uniform(k_ox_mask, (pop_size,)) < params.ox_prob
        t_blx = child_X[:, t_offset:wp_offset]
        child_X = child_X.at[:, t_offset:wp_offset].set(jnp.where(do_ox[:, None], t_ox, t_blx))

        child_mu, child_lam, child_rho = mu[parent], lam[parent], rho[parent]

        # Lamarckian write-back: this offspring's TRUE wp/psi is the
        # AL+L-BFGS-refined value, not the raw crossover+mutation result
        # above -- see module docstring.
        assign, cond_binary, proj_branch, t, wp0, psi0 = problem._extract_batch(child_X)
        wp_star, psi_star, off_mu, off_lam, off_rho = self.local_refine(
            wp0, psi0, assign, cond_binary, proj_branch, t, child_mu, child_lam, child_rho,
            params.x0, params.problem_params, params.anchor, params.cv_tol)
        off_X = _write_wp_batch_jax(problem, child_X, wp_star)
        off_X = _write_psi_batch_jax(problem, off_X, psi_star)

        # Routing 2-opt/Or-opt local search on `t`, using the just-refined
        # `wp` -- also part of what this offspring's candidate solution IS,
        # not a post-hoc evaluation step (see module docstring).
        off_assign, off_cond_binary, off_proj_branch, off_t, off_wp, off_psi = problem._extract_batch(off_X)
        # jit_apply_projections (cached, compiled once per problem --
        # problem.py) instead of eager apply_projections: this is one of
        # several distinct textual call sites of the same projection chain
        # across this module/solver.py, each of which would otherwise
        # independently re-trace it (see _evaluate_population_jax's own
        # comment, solver.py).
        off_wp = jit_apply_projections(problem)(
            off_wp, off_psi, off_proj_branch, params.problem_params, off_assign, off_cond_binary, off_t,
            params.anchor.node_active, params.x0, params.anchor.var_committed, params.anchor.var_anchor)
        off_assign_eff, off_wp_eff_frozen, _off_wp_eff_live = apply_anchor(
            problem, off_assign, off_wp, params.anchor, params.x0)
        off_t = _routing_local_search_batched(
            problem, k_2opt, off_assign_eff, off_cond_binary, off_t, off_wp_eff_frozen,
            params.x0, params.anchor.node_active, self.n_2opt_trials,
            or_opt_prob=self.or_opt_prob, max_or_opt_seg_len=self.max_or_opt_seg_len)
        off_X = _write_t_batch_jax(problem, off_X, off_t)

        child_genome = _join_genome(off_X, off_mu, off_lam, off_rho)
        return child_genome, state

    def _tell(self, key, population, fitness, state, params):
        """mu+lambda elitist truncation: pool the current population
        (parents) with the just-`_ask`-produced offspring, rank by fitness,
        keep the best `population_size` -- exactly solver.py's own
        `pool_X`/`pool_S`/`keep` logic in `_make_gen_step_fn`.

        Ranks by raw (F, CV) recomputed fresh for BOTH sides and rescaled
        under the CURRENT `w`/`cv_tol` -- not the passed-in `fitness`
        (correct only for the offspring, under whatever schedule was live
        when evosax_ga.py's `step` computed it) and not `state.fitness`
        (frozen from whenever the surviving parents were themselves last
        scored) -- see module docstring."""
        problem = self.problem
        X_old = _split_genome(problem, state.population)[0]
        X_new = _split_genome(problem, population)[0]
        F_old, CV_old = _evaluate_population_jax(problem, X_old, params.x0, params.problem_params, params.anchor)
        F_new, CV_new = _evaluate_population_jax(problem, X_new, params.x0, params.problem_params, params.anchor)
        # Read-only diagnostic, not part of the ranking below -- see
        # State.best_CV_proj's own comment for why this is tracked here
        # (batched over the pool, in the already-jitted step) rather than
        # left for a caller to recompute eagerly on just the winner.
        CV_proj_old = _evaluate_projection_cv_jax(problem, X_old, params.x0, params.problem_params, params.anchor)
        CV_proj_new = _evaluate_projection_cv_jax(problem, X_new, params.x0, params.problem_params, params.anchor)

        pool_pop = jnp.concatenate([state.population, population], axis=0)
        pool_F = jnp.concatenate([F_old, F_new])
        pool_CV = jnp.concatenate([CV_old, CV_new])
        pool_CV_proj = jnp.concatenate([CV_proj_old, CV_proj_new])
        pool_S = _combined_score(pool_F, pool_CV, params.w, params.cv_tol)

        keep = _rank_jax(pool_S)[:self.population_size]
        population_new, F_kept, CV_kept, CV_proj_kept, S_kept = (
            pool_pop[keep], pool_F[keep], pool_CV[keep], pool_CV_proj[keep], pool_S[keep])

        # Best-so-far: raw (F, CV, CV_proj), rescaled fresh -- see module
        # docstring for why this overrides (rather than trusts) the base
        # class's own best_solution/best_fitness. `state.best_solution`
        # itself may not be IN `state.population` any more (mu+lambda
        # truncation only guarantees survival within a generation's own
        # top-`population_size`; the carried-over best is tracked
        # separately and can persist for many generations/external calls
        # after being evicted from the visible population) -- so re-
        # evaluate it fresh here too, against THIS call's x0/anchor/params,
        # rather than trusting `state.best_F`/`best_CV`/`best_CV_proj`
        # (which would otherwise stay frozen at whatever scene state was
        # live the last time something actually beat it, silently going
        # stale across an x0/anchor change -- e.g. a moved block -- until
        # some new individual happens to beat it again). This is also
        # exactly the fresh-under-current-scene value mpc.py's solve()
        # wants for its own diagnostic self._last_F/_last_CV/_last_CV_proj
        # readout, computed once here instead of via a separate eager,
        # uncached call there.
        X_best_old = _split_genome(problem, state.best_solution[None, :])[0]
        F_best_old, CV_best_old = _evaluate_population_jax(
            problem, X_best_old, params.x0, params.problem_params, params.anchor)
        CV_proj_best_old = _evaluate_projection_cv_jax(
            problem, X_best_old, params.x0, params.problem_params, params.anchor)
        best_S_old = _combined_score(F_best_old[0], CV_best_old[0], params.w, params.cv_tol)
        improved = S_kept[0] < best_S_old
        best_solution = jnp.where(improved, population_new[0], state.best_solution)
        best_F = jnp.where(improved, F_kept[0], F_best_old[0])
        best_CV = jnp.where(improved, CV_kept[0], CV_best_old[0])
        best_CV_proj = jnp.where(improved, CV_proj_kept[0], CV_proj_best_old[0])
        best_fitness = jnp.where(improved, S_kept[0], best_S_old)

        return state.replace(
            population=population_new, fitness=S_kept,
            best_solution=best_solution, best_fitness=best_fitness,
            best_F=best_F, best_CV=best_CV, best_CV_proj=best_CV_proj,
        )

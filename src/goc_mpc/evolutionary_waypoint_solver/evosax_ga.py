"""Generic evosax `ask`/`evaluate`/`tell` driver for GraphOrderingRelaxed
(problem.py): any evosax `PopulationBasedAlgorithm` (`algo_cls`) can be
plugged in to search the extended `[X | mu | lam | rho]` genome (see
below) -- this module only owns the parts that are the SAME regardless of
which algorithm is chosen: the genome layout/bounds, the merit/
constraint-violation evaluation (`_evaluate_population_jax`, from
solver.py), and the smooth combined-score fitness (`_combined_score`/
`_calibrate_score_scale`/`_score_schedule`, also from solver.py) that
whatever `fitness` value gets handed to `algo.tell` is computed from.

What this module does NOT own: any algorithm's own `_ask`/`_tell` operators
-- in particular, the Lamarckian AL+L-BFGS local refine and the routing
2-opt/Or-opt local search that solver.py's original GA applies to every
offspring live inside `lamarckian_ga.LamarckianGA._ask`, not here, because
they define what a CANDIDATE SOLUTION actually is (exactly like evosax's
own PSO computes its candidate via a velocity update inside `_ask`, not as
some universal post-processing step) -- see that module's docstring. Using
`algo_cls=lamarckian_ga.LamarckianGA` here reproduces solver.py's original
algorithm; any other evosax algorithm (`SimpleGA`, `PSO`, ...) is a
genuinely different, equally pluggable alternative that only ever gets the
plain merit/CV-based fitness below.

Genome layout: `[X | mu | lam | rho]`, one flat vector per individual --
`X` is the ordinary GraphOrderingRelaxed decision vector (assign |
cond_binary | proj_branch | t | wp | psi); `mu`/`lam`/`rho` are the
Augmented-Lagrangian multiplier state, folded in alongside it so that
whatever process an `algo_cls` uses to produce/select individuals carries
them along automatically, exactly like `wp`/`psi` already do (there is no
per-offspring "parent index" evosax's own ask/tell exposes to track them
separately by lineage the way solver.py's `mu[parent], lam[parent],
rho[parent]` could).

Live per-call/per-generation values (`x0`, the problem's `params`, the
current `anchor`, and the annealed `w`/`cv_tol`) that an algorithm's own
`_ask`/`_tell` may need (e.g. `LamarckianGA`'s local refine and its own
raw-(F, CV) rescaling, see that module's docstring for why it doesn't
trust a stored/passed-through combined score across generations) but
generic algorithms (`SimpleGA`, `PSO`, ...) don't -- travel in via evosax's own `params`
argument: `step`, below, threads them into a fresh `Params.replace(...)`
before every `ask`/`tell` call, but ONLY for algorithms whose `Params`
dataclass actually declares matching field names (checked via `hasattr`,
so this is a no-op, not a special case, for any algorithm that doesn't
need them).

Bounds: evosax algorithms generally do not clip to a decision-variable box
on their own (SimpleGA's mutation is unbounded Gaussian noise, for
instance) -- every `ask()` output is clipped here before use, to
`problem.xl`/`problem.xu` for the `X` block and to AL-appropriate ranges
for `mu`/`lam`/`rho` (lam >= 0, rho in (0, rho_max], mu unconstrained but
kept finite).
"""

import jax
import jax.numpy as jnp

from .problem import pad_to_state_dim
from .solver import (
    _evaluate_population_jax,
    _combined_score,
    _calibrate_score_scale,
    _score_schedule,
    _seed_initial_population,
)

jax.config.update("jax_enable_x64", True)


def _genome_dims(problem):
    return problem.n_var, problem.n_eq_constr, problem.n_ieq_constr


def _split_genome(problem, genome):
    n_var, n_eq, n_ineq = _genome_dims(problem)
    X = genome[:, :n_var]
    mu = genome[:, n_var:n_var + n_eq]
    lam = genome[:, n_var + n_eq:n_var + n_eq + n_ineq]
    rho = genome[:, -1]
    return X, mu, lam, rho


def _join_genome(X, mu, lam, rho):
    return jnp.concatenate([X, mu, lam, rho[:, None]], axis=1)


def _genome_bounds(problem, rho_max):
    """(lo, hi) for the extended [X | mu | lam | rho] genome -- X's own
    box (problem.xl/xu), mu left effectively unconstrained (a Lagrange
    multiplier on an equality residual has no natural sign or scale, but
    is kept finite to guard against runaway crossover/mutation noise
    compounding over many generations), lam >= 0 (AL theory: an inequality
    multiplier is never negative -- local_refine's own update already
    re-clips it this way too, this just keeps ask()'s raw output sane
    before that), rho in (a small positive floor, rho_max] (an AL penalty
    weight must stay strictly positive)."""
    n_var, n_eq, n_ineq = _genome_dims(problem)
    mu_bound = 1e6
    lo = jnp.concatenate([
        jnp.asarray(problem.xl),
        jnp.full((n_eq,), -mu_bound),
        jnp.zeros((n_ineq,)),
        jnp.array([1e-8]),
    ])
    hi = jnp.concatenate([
        jnp.asarray(problem.xu),
        jnp.full((n_eq,), mu_bound),
        jnp.full((n_ineq,), mu_bound),
        jnp.array([rho_max]),
    ])
    return lo, hi


def _live_params(algo_params, **live):
    """Merges `live` (x0/problem_params/anchor/cv_tol, whichever the caller
    has ready) into `algo_params`, restricted to whichever of those fields
    `algo_params`'s own dataclass actually declares -- a no-op passthrough
    for any algorithm (SimpleGA, PSO, ...) that doesn't declare them, so
    this stays generic rather than special-cased on algorithm identity."""
    present = {k: v for k, v in live.items() if hasattr(algo_params, k)}
    return algo_params.replace(**present) if present else algo_params


def build_evosax_ga(problem, algo_cls, pop_size, n_gen, algo_kwargs=None, algo_params=None,
                     rho_max=1e6, w=None, cv_tol=None, w_frac=1.0, cv_tol_frac=0.05,
                     w_growth=10.0, cv_tol_floor_frac=0.0):
    """Builds (algo, algo_params, step): `algo` is the constructed evosax
    algorithm instance (population_size=pop_size, solution=the extended
    genome template) -- kept around for its `init` (see build_initial_carry_
    fn below); `algo_params` is its (possibly caller-overridden) Params
    pytree; `step(carry_in, x0, params, anchor) -> carry_out` is a jitted
    `jax.lax.scan` over `n_gen` generations, mirroring solver.py's
    `build_lamarckian_ga`'s external contract (an arbitrary carry in,
    x0/params/anchor as live per-call arguments, same carry shape out) --
    except the carry here is `(evosax_state, key)`, not solver.py's
    10-tuple: evosax's own State already tracks population/fitness/
    best_solution/best_fitness/generation_counter, so there is nothing left
    for this module's own carry to duplicate (see module docstring for why
    mu/lam/rho are folded into the genome instead of carried separately,
    closing the last gap). `algo_cls` is any evosax `PopulationBasedAlgorithm`
    subclass -- this is the whole point of this module: swapping it is a
    constructor argument, not a rewrite. Pass `algo_cls=lamarckian_ga.
    LamarckianGA` (with `algo_kwargs=dict(problem=problem, ...)`) to
    reproduce solver.py's own algorithm exactly; any other algorithm gets
    only the plain merit/CV-based fitness below, with no Lamarckian
    refinement or local search (those live inside `LamarckianGA._ask`
    itself, see this module's and that module's docstrings).
    """
    algo_kwargs = {} if algo_kwargs is None else algo_kwargs
    n_var, n_eq, n_ineq = _genome_dims(problem)
    genome_dim = n_var + n_eq + n_ineq + 1
    lo, hi = _genome_bounds(problem, rho_max)

    algo = algo_cls(population_size=pop_size, solution=jnp.zeros(genome_dim), **algo_kwargs)
    algo_params = algo.default_params if algo_params is None else algo_params

    def step(carry_in, x0, params, anchor):
        state0, key0 = carry_in
        # w/cv_tol calibrated from the carried-in state's OWN best/current
        # fitness the same way solver.py's build_lamarckian_ga's step
        # recalibrates every call -- see that function's docstring for why
        # this only anneals WITHIN this call's n_gen generations, not
        # across separately-chunked calls.
        F0, CV0 = _evaluate_population_jax(
            problem, _split_genome(problem, state0.population)[0], x0, params, anchor)
        w0, cv_tol0 = _calibrate_score_scale(F0, CV0, w_frac, cv_tol_frac)
        w_val = w0 if w is None else jnp.asarray(w, dtype=F0.dtype)
        cv_tol_val = cv_tol0 if cv_tol is None else jnp.asarray(cv_tol, dtype=F0.dtype)

        # x0/params/anchor are invariant across this whole n_gen-generation
        # scan (only cv_tol anneals per generation, below) -- threaded in
        # once here rather than re-merged every generation.
        base_params = _live_params(algo_params, x0=x0, problem_params=params, anchor=anchor)

        def gen_step(carry, gen):
            state, key = carry
            key, key_ask, key_tell = jax.random.split(key, 3)
            w_t, cv_tol_t = _score_schedule(gen, n_gen, w_val, cv_tol_val, w_growth, cv_tol_floor_frac)
            gen_params = _live_params(base_params, w=w_t, cv_tol=cv_tol_t)

            genome, state = algo.ask(key_ask, state, gen_params)
            genome = jnp.clip(genome, lo, hi)

            X = _split_genome(problem, genome)[0]
            F, CV = _evaluate_population_jax(problem, X, x0, params, anchor)
            fitness = _combined_score(F, CV, w_t, cv_tol_t)

            state, _metrics = algo.tell(key_tell, genome, fitness, state, gen_params)
            return (state, key), None

        (state_out, key_out), _ = jax.lax.scan(
            gen_step, (state0, key0), xs=jnp.arange(n_gen), length=n_gen)
        return state_out, key_out

    return algo, algo_params, jax.jit(step)


def build_initial_carry_fn(problem, algo, algo_params, pop_size, anchor, x0=None, params=None,
                            rho0=1.0, n_seed_individuals=None, seed_jitter_t=1.0, seed_jitter_wp_frac=0.05):
    """Returns a jitted `init(key) -> carry` -- the evosax-native analogue
    of solver.py's `build_initial_carry_fn`: a fresh (cold) random genome
    population (plus the same small precedence-heuristic-seeded subset,
    `_seed_initial_population` reused verbatim -- it only ever writes into
    the `X` block, so it composes with the extended genome unchanged),
    mu/lam/rho initialized to zero/rho0 (folded into the genome, see module
    docstring), raw (un-refined) fitness evaluated once, then handed to
    `PopulationBasedAlgorithm.init` (which -- unlike the base class's plain
    `init(key, params)` -- takes the initial population/fitness directly,
    since a population-based algorithm's very first generation of parents
    IS this initial population, not something it generates itself).
    `algo_params` is threaded with the real x0/params/anchor (via
    `_live_params`, a no-op for algorithms that don't declare those fields)
    before this call, so an algorithm whose own `init` override needs them
    (e.g. `lamarckian_ga.LamarckianGA`, to seed its raw best-so-far; or
    `small_continuous_vrp.SmallContinuousVRPSolver`, whose `init` override
    makes an irrevocable discrete choice from them) sees live values, not
    `_default_params`'s structural placeholders.

    `x0`/`params` default to `None`, which falls back to a COLD value
    derived from `problem.x0`/`problem.params` -- the problem-BUILD-time
    snapshot (`GraphOfConstraintsMPC`'s first `_ensure_built` call), zero-
    padded to `state_dim` (`pad_to_state_dim`) for any column `problem.x0`
    doesn't itself carry (e.g. object/non-agent state) -- backward-
    compatible with every existing call site, but WRONG for an `init`
    override that needs an object's real current position (a moved block,
    not the zero `pad_to_state_dim` fills in): pass the caller's own live
    `x0_arr`/`params_arr` (mpc.py's `warmup`/`solve` already compute them,
    the same values handed to `step_fn` right after this) to get that."""
    n_var, n_eq, n_ineq = _genome_dims(problem)
    xl, xu = jnp.asarray(problem.xl), jnp.asarray(problem.xu)
    if x0 is None:
        x0 = pad_to_state_dim(jnp.asarray(problem.x0).reshape(-1), problem.state_dim)
    if params is None:
        params = jnp.asarray(problem.params)
    n_seed = pop_size // 10 if n_seed_individuals is None else n_seed_individuals
    n_seed = max(0, min(n_seed, pop_size))

    def init(key):
        key, k_init, k_seed, k_algo = jax.random.split(key, 4)
        X0 = jax.random.uniform(k_init, (pop_size, n_var), minval=xl, maxval=xu, dtype=xl.dtype)
        X0 = _seed_initial_population(problem, k_seed, X0, n_seed, seed_jitter_t, seed_jitter_wp_frac)
        mu0 = jnp.zeros((pop_size, n_eq))
        lam0 = jnp.zeros((pop_size, n_ineq))
        rho_arr0 = jnp.full((pop_size,), rho0)
        genome0 = _join_genome(X0, mu0, lam0, rho_arr0)

        F0, CV0 = _evaluate_population_jax(problem, X0, x0, params, anchor)
        w0, cv_tol0 = _calibrate_score_scale(F0, CV0, 1.0, 0.05)
        fitness0 = _combined_score(F0, CV0, w0, cv_tol0)

        live_algo_params = _live_params(algo_params, x0=x0, problem_params=params, anchor=anchor)
        state0 = algo.init(k_algo, genome0, fitness0, live_algo_params)
        return state0, key

    return jax.jit(init)

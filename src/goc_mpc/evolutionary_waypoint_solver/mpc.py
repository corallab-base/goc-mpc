"""EvolutionaryWaypointSolver: a GraphWaypointMPC-compatible (duck-typed)
wrapper around spec.build_graph_ordering_problem + evosax_ga.build_evosax_ga.
`GraphOfConstraintsMPC` (goc_mpc.py) accepts any object satisfying this
protocol via its `waypoint_mpc=` constructor argument, with no isinstance
check, so no changes are needed there.

The underlying algorithm is a constructor choice, not a fixed
implementation: `algorithm="lamarckian_al"` (default, lamarckian_ga.
LamarckianGA -- reproduces solver.py's original GA+AL algorithm exactly)
or `algorithm="small_continuous_vrp"` (small_continuous_vrp.
SmallContinuousVRPSolver -- additionally replaces analytic-projection
branch selection with an exact per-generation DP; see that module's own
docstring for what it fixes and its scope limits). Both are evosax
`PopulationBasedAlgorithm` subclasses, built through the SAME generic
evosax_ga.py driver (`build_evosax_ga`/`build_initial_carry_fn`) -- see
this module's `_ALGORITHMS`/`_ensure_built`.

Output buffer shapes/conventions match GraphWaypointMPC's exactly:
  _waypoints:       (num_nodes, graph.total_dim) -- agent columns followed by
                     object columns, one row per node, mirroring
                     MILP's per-node joint configuration `W`
                     (graph_of_constraints.cpp:79's `total_dim`). view_
                     waypoints() keeps returning this whole row (matching
                     GraphWaypointMPC's contract, which callers already slice
                     down to the agent columns themselves -- see goc_mpc.py's
                     step()); view_object_waypoints() returns the object-
                     column slice as a convenience view. A node not yet
                     covered by any solve() call (still at construction-time
                     init) reads as NaN; every node in remaining_vertices as
                     of the last solve() call carries that solve's real,
                     GA-solved value, whether or not any constraint
                     referenced it there (e.g. an un-held object's value at
                     a node tied to it only via the auto-generated "stay
                     stationary" invariant, spec.py's
                     _resolve_stationary_objects, not a NaN sentinel).
  _assignments:     (num_phis,)
  _var_assignments: (num_variables,)
  _t_by_node_id:    (num_graph_nodes,)
so GraphOfConstraintsMPC.step()/_solve_for_waypoints/_solve_for_timing
consume it identically to MILPWaypointMPC.

The underlying JAX kernel models a full per-node configuration (every
agent's AND every referenced object's position, one row per node -- see
spec.py's module docstring), the same shape of information MILPWaypointMPC's
joint `W` carries, just solved via GA+AL instead of MILP.

Build once, reuse forever, remaining_vertices as a runtime input
------------------------------------------------------------------
Unlike MILPWaypointMPC (which rebuilds a SubgraphOfConstraints fresh every
solve from remaining_vertices), build_graph_ordering_problem/
GraphOrderingRelaxed now span the WHOLE graph unconditionally (see spec.py's
build_graph_ordering_problem docstring) -- their shape (n_var, n_eq_constr,
n_ieq_constr, ...) is fixed for the lifetime of a GraphOfConstraints,
independent of remaining_vertices. So `_ensure_built` constructs `_problem`/
`_ga_fn` exactly ONCE, on the first solve()/warmup() call, and every later
call reuses them unconditionally
-- no cache-key/shape-key bookkeeping, since the shape can never mismatch
again. `add_python_constraint` must be called before that first call (raises
otherwise): constraints are baked into `_problem`'s fixed structure once,
not re-applied per solve.

remaining_vertices' actual effect is now a per-call runtime input instead of
a structural one: `_compute_anchor` builds a `problem.AnchorState` from
remaining_vertices plus this solver's own persisted `_waypoints`/
`_var_assignments` (a passed node's row, or a committed variable's
assignment, is exactly the last value THIS solver wrote there while it was
still active, and is left untouched by the write-back loops below once it's
no longer in remaining_vertices -- the same "frozen last-committed value"
MILP substitutes via `previous_X.row(u_node)`). `problem.apply_anchor`/
kernel.py's `node_active` argument then splice that in wherever a constraint
or routing decision reads a node's position or a variable's owner: a node/
edge constraint anchored at an already-passed node stays correctly enforced
against that known constant instead of silently disappearing (the bug this
design fixes -- see spec.py's module docstring), while ordering edges
touching a passed node are (correctly) dropped, since precedence against an
already-visited node is moot.

A passed node's "known constant" is one of TWO choices, not one: `anchor_wp`
above (the frozen, last-committed-while-active value -- "the planned
position") or this call's real state (the full-configuration `x0` given to
solve()/warmup(), read straight off by problem.apply_anchor -- "the live
position"). Which one a given EDGE constraint's `u`-side residual reads is a
per-constraint choice made where the constraint is defined
(GraphOfConstraints.add_edge_constraint(..., live=True) --
graph_of_constraints.hpp's docstring) -- build_graph_ordering_problem reads
`graph.live_edge_phis` straight off the graph itself, not something it
threads through as its own argument (there is nothing for a caller here to
legitimately override; see build_graph_ordering_problem's own docstring
comment), nor something it rebinds or tracks per node -- there is no "rewrite
this row because the node just passed" step anywhere here, deliberately: a
constraint that wants the mobile real state (e.g. a rigid-transport edge,
where the *real* carried-object position should keep tracking the *real*
carrying-agent position even after the grasp node's nominal pass point, not
the grasp node's originally-planned position) should be registered live;
anything else defaults to frozen, matching `main` branch's
DeferredEdgeOp.waypoint_builder x_u argument (previous_X.row(u_node)) with no
artificial rebind hack layered on top. Node constraints have no live/frozen
choice to make at all (graph_of_constraints.hpp's live_edge_phis comment) --
spec.py's _resolve_symbolic_constraints hardcodes "frozen" for every one of
them.

Because AnchorState is a fixed-shape pytree (n_nodes/n_variables never
change), passing a different `anchor` value on each call -- like `x0` -- adds
no retracing, only a cheap-in-Python `_compute_anchor` call. Population reuse
(`_carry`) is therefore unconditional after the first build too: `solve()`
always resumes from the previous call's population by handing `_carry`
straight back into `self._step_fn` (built once per problem by
`_ensure_built`, see `evosax_ga.build_evosax_ga`) -- unlike solver.py's old
carry, evosax's own State stores no raw per-individual F/CV that could go
stale across an x0/anchor change (its `_ask`/`_tell` recompute those fresh
from whatever x0/anchor `self._step_fn` is called with, every call -- see
evosax_ga.py's own module docstring), so there's no separate resume-refresh
step to call first. Call `warmup()` once up front to pay the first trace/
compile outside of a timed run and seed `_carry` from that run's own result.
"""

import dataclasses
import time

import jax
import numpy as np
import jax.numpy as jnp

from .spec import (
    build_graph_ordering_problem, _agent_widths, _slot_width,
    _object_widths, _object_slot_width,
)
from .problem import AnchorState, jit_apply_projections
from .evosax_ga import (
    build_evosax_ga, build_initial_carry_fn as _build_evosax_initial_carry_fn,
    _split_genome,
)
from .lamarckian_ga import LamarckianGA
from .small_continuous_vrp import SmallContinuousVRPSolver
from .solver import _evaluate_population_jax, _evaluate_projection_cv_jax

# `algorithm=` constructor choices -- both evosax PopulationBasedAlgorithm
# subclasses (lamarckian_ga.py/small_continuous_vrp.py), built through the
# SAME generic evosax_ga.py driver (build_evosax_ga/build_initial_carry_fn)
# below, so switching between them is exactly this one constructor kwarg,
# not a different code path. "lamarckian_al" (the default) reproduces the
# original algorithm exactly (see LamarckianGA's own module docstring);
# "small_continuous_vrp" additionally replaces `proj_branch`'s GA search
# with an exact per-generation DP (see SmallContinuousVRPSolver's own
# module docstring for what that fixes and its scope limits).
_ALGORITHMS = {
    "lamarckian_al": LamarckianGA,
    "small_continuous_vrp": SmallContinuousVRPSolver,
}

# Params that only affect *building the initial carry* (a fresh random
# population, or re-wrapping a reused one) -- as opposed to the algorithm
# itself. Split out of **lamarckian_kwargs in __init__ so each downstream
# call only ever receives the subset of kwargs it actually accepts.
_CARRY_KWARGS = ("rho0", "n_seed_individuals", "seed_jitter_t", "seed_jitter_wp_frac")
# LamarckianGA/SmallContinuousVRPSolver constructor kwargs (both share the
# same __init__ signature -- SmallContinuousVRPSolver only adds its own
# static-chain setup on top, see that module). The last three
# (reseed_frac/reseed_rho0/reseed_cv_tol) are SmallContinuousVRPSolver-only
# -- LamarckianGA's own __init__ has no **kwargs sink, so passing one of
# these under algorithm="lamarckian_al" fails loudly with a plain
# TypeError from that constructor, rather than silently doing nothing.
_ALGO_KWARGS = ("tournament_k", "wp_mut_scale", "outer_iters", "inner_maxiter",
                "rho_growth", "lbfgs_history", "ls_max_trials", "optimizer",
                "n_2opt_trials", "or_opt_prob", "max_or_opt_seg_len",
                "reseed_frac", "reseed_rho0", "reseed_cv_tol")
# build_evosax_ga's own top-level annealed-schedule kwargs.
_SCHEDULE_KWARGS = ("w", "cv_tol", "w_frac", "cv_tol_frac", "w_growth", "cv_tol_floor_frac")
# LamarckianGA.Params fields with no constructor-kwarg home -- applied via
# Params.replace(...) once the algorithm's own default_params exist (see
# _ensure_built). `rho_max` is deliberately in NEITHER bucket above: it's a
# genuine constructor kwarg (LamarckianGA.__init__) AND a build_evosax_ga
# top-level kwarg (genome bounds) at once, so it's read once and passed to
# both explicitly instead of living in one list and silently missing the
# other.
_PARAMS_KWARGS = ("mut_sigma", "cx_prob", "ox_prob")


@dataclasses.dataclass
class PopulationDiagnostics:
    """Per-current-population-member diagnostics -- see
    EvolutionaryWaypointSolver.get_population_diagnostics()'s own
    docstring. Plain numpy, not jax arrays -- a debug/visualization
    accessor, not a hot path."""
    F: np.ndarray               # (pop,)
    CV: np.ndarray               # (pop,)
    CV_proj: np.ndarray          # (pop,)
    owner_variable: np.ndarray   # (pop, n_variables) int -- assigned agent id per var slot
    hard_score: np.ndarray       # (pop,)
    rank: np.ndarray             # (pop,) int, 0 = best (lowest hard_score)
    will_be_evicted: np.ndarray  # (pop,) bool -- the n_evict worst by hard_score
    n_evict: int
    # Per-VARIABLE (not per-member -- one committed/anchor state governs the
    # WHOLE population, see AnchorState in problem.py), read straight off the
    # anchor this diagnostic snapshot was evaluated against. A committed slot
    # is not searched at all by the discrete solver (dp_master_jax's
    # `slot_agent = jnp.where(var_committed, var_anchor, A)`): every skeleton
    # it can offer already has that slot pinned to `var_anchor`, so a
    # population where every member shares the same owner for a slot is
    # expected, not a search failure, exactly when that slot is committed.
    var_committed: np.ndarray   # (n_variables,) bool
    var_anchor: np.ndarray      # (n_variables,) int -- committed agent id (meaningless where not committed)


class EvolutionaryWaypointSolver:
    def __init__(self, graph, splines, objective="avg", edge_cost_fn=None,
                 wp_bounds=(-10.0, 10.0), pop_size=30, n_gen=60,
                 algorithm="lamarckian_al", **lamarckian_kwargs):
        if algorithm not in _ALGORITHMS:
            raise ValueError(
                f"algorithm must be one of {sorted(_ALGORITHMS)}, got {algorithm!r}")
        self._algo_cls = _ALGORITHMS[algorithm]

        self._graph = graph
        self._splines = splines
        self._objective = objective
        # One shared callable (a, b) -> scalar, or a per-agent list/tuple of
        # them (one entry per graph agent) for per-agent obstacle costs --
        # see kernel.make_graph_kernel's docstring.
        self._edge_cost_fn = edge_cost_fn
        self._wp_bounds = wp_bounds
        self._pop_size = pop_size
        self._n_gen = n_gen
        # `seed` selects the PRNGKey for a from-scratch initial carry (only
        # used when no previous population is available to resume from).
        self._seed = lamarckian_kwargs.pop("seed", 1)
        self._carry_kwargs = {k: lamarckian_kwargs.pop(k) for k in _CARRY_KWARGS
                               if k in lamarckian_kwargs}
        # rho_max is a genuine kwarg of BOTH the algorithm's own constructor
        # and build_evosax_ga's top-level genome-bounds argument -- read
        # once, applied to both explicitly in _ensure_built (see
        # _PARAMS_KWARGS' own comment above for why it lives in neither
        # bucket list).
        self._rho_max = lamarckian_kwargs.pop("rho_max", 1e6)
        self._algo_kwargs = {k: lamarckian_kwargs.pop(k) for k in _ALGO_KWARGS
                              if k in lamarckian_kwargs}
        self._schedule_kwargs = {k: lamarckian_kwargs.pop(k) for k in _SCHEDULE_KWARGS
                                  if k in lamarckian_kwargs}
        self._params_kwargs = {k: lamarckian_kwargs.pop(k) for k in _PARAMS_KWARGS
                                if k in lamarckian_kwargs}
        if lamarckian_kwargs:
            raise TypeError(
                f"unrecognized keyword argument(s) for EvolutionaryWaypointSolver: "
                f"{sorted(lamarckian_kwargs)}")
        # Kept for backward compatibility with callers/scripts that read
        # this attribute directly (e.g. to rebuild an equivalent evosax call
        # elsewhere) -- the algorithm-constructor bucket specifically, since
        # that's the one such callers have actually wanted so far.
        self._lamarckian_kwargs = dict(self._algo_kwargs)
        self._python_constraints = []  # (node, fn, kind, name)

        num_nodes = graph.structure.num_nodes
        agents_width = graph.agent_col_offsets[-1]
        # One row per node -- agent columns then object columns,
        # mirroring MILP's per-node joint configuration `W` (see module
        # docstring). Object columns start NaN (never-referenced sentinel,
        # see view_object_waypoints()); agent columns start 0, overwritten
        # with a real depot value before ever being read (solve()'s
        # per-active-node reset, the only writer -- see module docstring).
        self._waypoints = np.zeros((num_nodes, graph.total_dim))
        self._waypoints[:, agents_width:] = np.nan
        self._assignments = -np.ones(graph.num_phis, dtype=int)
        self._var_assignments = -np.ones(graph.num_variables, dtype=int)
        self._t_by_node_id = np.zeros(num_nodes)
        self._last_solve_time = 0.0
        self._last_F = None
        self._last_CV = None
        self._last_CV_proj = None
        # Stashed every solve() so get_population_diagnostics() can
        # re-evaluate the CURRENT population against the SAME scene state
        # get_last_fitness()'s own (F, CV, CV_proj) was computed against
        # (see that method's comment for why staleness matters here).
        self._last_x0_arr = None
        self._last_params_arr = None
        self._last_anchor = None

        # Built once, on the first solve()/warmup() call -- see module
        # docstring -- and never rebuilt again.
        self._problem = None
        self._algo = None
        self._algo_params = None
        self._step_fn = None
        self._carry = None
        self._last_solve_was_warm = False
        self._last_population_reused = False
        self._last_compile_time = 0.0

    def add_python_constraint(self, node, fn, kind="eq", name=None):
        """Registers fn(wp_row, assign) -> (k,) on `node`'s whole row (see
        spec.build_graph_ordering_problem's docstring), baked into
        `_problem`'s structure the first time it's built (the first
        solve()/warmup() call). Must be called before that first call --
        structure is built once and reused for this solver's whole lifetime
        (see module docstring), so there's no later point at which a
        newly-registered constraint could take effect."""
        if self._problem is not None:
            raise RuntimeError(
                "add_python_constraint() called after the first solve()/warmup() -- "
                "structure is built once, from every constraint registered up to that "
                "point, and reused for the solver's whole lifetime; register all "
                "python constraints before the first solve()/warmup() call")
        self._python_constraints.append((node, fn, kind, name))

    def _check_x0(self, x0):
        """`x0` is always the single full-configuration state vector
        (graph.total_dim,) -- agent columns then object columns, PACKED
        (graph.agent_col_offsets/object_col_offsets' cumulative convention),
        exactly what GraphOfConstraintsMPC.step() passes. solve()/warmup()
        convert it to the padded-slot layout (via _to_padded_row) before
        handing it to the JAX kernel, where it's what problem.apply_anchor's
        wp_eff_live substitution reads as one global joint state. There is
        no separate agent-only depot convention: a caller with no real
        object state to report (e.g. a fresh solve before anything's been
        grasped) zero-fills those columns itself."""
        x0 = np.asarray(x0)
        graph = self._graph
        assert x0.size == graph.total_dim, (
            f"x0.size ({x0.size}) != graph.total_dim ({graph.total_dim}) -- "
            "x0 must be the full agent+object configuration vector")
        return x0

    def _ensure_built(self, x0):
        if self._problem is not None:
            return
        graph = self._graph
        # Per-agent padded-slot layout (see spec.py's _agent_widths/
        # _slot_width docstrings) -- build_graph_ordering_problem derives
        # the SAME agent_widths/slot_width internally and requires this
        # x0 argument's shape to agree with it; a plain reshape only works
        # when every agent happens to share one width, which is no longer
        # guaranteed.
        agent_widths = _agent_widths(graph)
        slot_width = _slot_width(agent_widths)
        agent_offsets = graph.agent_col_offsets
        x0_per_agent = np.zeros((graph.num_agents, slot_width))
        for k in range(graph.num_agents):
            x0_per_agent[k, :agent_widths[k]] = x0[agent_offsets[k]:agent_offsets[k + 1]]

        # Stashed for _to_padded_row/_to_packed_row below -- the packed
        # (graph.agent_col_offsets/object_col_offsets cumulative convention,
        # graph.total_dim-wide -- _waypoints'/x0's own layout) <-> padded-
        # slot (spec.py's _slot_width/_object_slot_width layout, problem.
        # state_dim-wide -- anchor_wp's/wp's/the JAX kernel's own x0
        # argument's layout) row conversion every caller of those needs.
        # Structural (graph.structure never changes after construction), so
        # computed once here alongside the rest of _ensure_built's one-time
        # setup.
        self._agent_widths = agent_widths
        self._slot_width = slot_width
        self._object_widths = _object_widths(graph)
        self._object_slot_width = _object_slot_width(self._object_widths)
        self._agent_offsets = agent_offsets
        self._object_offsets = graph.object_col_offsets

        problem = build_graph_ordering_problem(
            graph, x0_per_agent, self._wp_bounds,
            objective=self._objective, edge_cost_fn=self._edge_cost_fn,
            python_constraints=self._python_constraints)

        self._problem = problem
        # build_evosax_ga's own step(carry, x0, params, anchor) -> carry IS
        # the warm-resume path too (unlike solver.py's old carry, evosax's
        # State carries only the population + a combined `fitness`/
        # `best_fitness` -- no raw per-individual F/CV that could go stale
        # across an x0/anchor change -- _ask/_tell recompute those fresh
        # from the CURRENT x0/params/anchor every single call, see
        # evosax_ga.py's own module docstring), so there's no separate
        # resume-carry function to build/warm here the way solver.py's
        # build_resume_carry_fn used to be.
        self._algo, self._algo_params, self._step_fn = build_evosax_ga(
            problem, self._algo_cls, self._pop_size, self._n_gen,
            algo_kwargs=dict(problem=problem, rho_max=self._rho_max, **self._algo_kwargs),
            rho_max=self._rho_max, **self._schedule_kwargs)
        if self._params_kwargs:
            self._algo_params = self._algo_params.replace(**self._params_kwargs)

    def _to_padded_row(self, packed_row):
        """Expands a (graph.total_dim,) packed row -- graph.agent_col_offsets/
        object_col_offsets' cumulative convention, the layout _waypoints and
        the x0 callers pass in both use -- out to a (problem.state_dim,)
        padded-slot row: every agent/object gets an equal-width slot_width/
        object_slot_width column span (spec.py's _slot_width docstring),
        real config left-justified within it and zero past its own width.
        This is the layout anchor_wp/wp/the JAX kernel's own x0 argument
        all actually operate in -- see mpc.py's module docstring and
        problem.apply_anchor's."""
        agent_widths, slot_width = self._agent_widths, self._slot_width
        object_widths, object_slot_width = self._object_widths, self._object_slot_width
        agent_offsets, object_offsets = self._agent_offsets, self._object_offsets
        agents_width = len(agent_widths) * slot_width
        out = np.zeros(agents_width + len(object_widths) * object_slot_width)
        for k, w in enumerate(agent_widths):
            out[k * slot_width:k * slot_width + w] = packed_row[agent_offsets[k]:agent_offsets[k] + w]
        for k, w in enumerate(object_widths):
            col0 = agents_width + k * object_slot_width
            out[col0:col0 + w] = packed_row[object_offsets[k]:object_offsets[k] + w]
        return out

    def _to_packed_row(self, padded_row):
        """Inverse of _to_padded_row: strips a (problem.state_dim,) padded-
        slot row back down to a (graph.total_dim,) packed row -- _waypoints'
        own layout (what view_waypoints()/view_object_waypoints()/
        get_agent_paths callers expect, matching GraphOfConstraintsMPC.
        step()'s own agent_col_offsets slicing)."""
        agent_widths, slot_width = self._agent_widths, self._slot_width
        object_widths, object_slot_width = self._object_widths, self._object_slot_width
        agent_offsets, object_offsets = self._agent_offsets, self._object_offsets
        agents_width = len(agent_widths) * slot_width
        out = np.zeros(self._graph.total_dim)
        for k, w in enumerate(agent_widths):
            out[agent_offsets[k]:agent_offsets[k] + w] = padded_row[k * slot_width:k * slot_width + w]
        for k, w in enumerate(object_widths):
            col0 = agents_width + k * object_slot_width
            out[object_offsets[k]:object_offsets[k] + w] = padded_row[col0:col0 + w]
        return out

    def _compute_anchor(self, remaining_vertices):
        """Builds this solve()/warmup() call's AnchorState from
        remaining_vertices plus this solver's own persisted _waypoints/
        _var_assignments -- see module docstring. Cheap (small numpy loops
        over problem's static, graph-global structure), not part of any
        JIT-traced path. Only ever populates anchor_wp -- the "frozen,
        planned" reading; the "live, real state" reading is `x0` itself,
        supplied straight to problem.apply_anchor by the caller (solve()/
        warmup()), not something this method builds."""
        graph, problem = self._graph, self._problem
        remaining_set = set(remaining_vertices)
        node_list = range(problem.n_nodes)

        node_active = np.array([node in remaining_set for node in node_list], dtype=bool)

        anchor_wp = np.zeros((problem.n_nodes, problem.state_dim))
        for node in node_list:
            if node in remaining_set:
                continue
            anchor_wp[node, :] = self._to_padded_row(np.nan_to_num(self._waypoints[node], nan=0.0))

        var_nodes = {}
        for node, (kind, val) in problem.instance_list:
            if kind == "var":
                var_nodes.setdefault(val, []).append(node)

        var_committed = np.zeros(problem.n_variables, dtype=bool)
        var_anchor = np.zeros(problem.n_variables, dtype=np.int32)
        for var_id, slot in problem.var_id_to_slot.items():
            nodes = var_nodes.get(var_id, [])
            if any(node not in remaining_set for node in nodes):
                assert self._var_assignments[var_id] != -1, (
                    f"variable {var_id} has an already-passed instance but was never assigned")
                var_committed[slot] = True
                var_anchor[slot] = self._var_assignments[var_id]

        return AnchorState(
            node_active=jnp.asarray(node_active),
            anchor_wp=jnp.asarray(anchor_wp),
            var_committed=jnp.asarray(var_committed),
            var_anchor=jnp.asarray(var_anchor),
        )

    def warmup(self, remaining_vertices, x0):
        """Builds (on the first call) and JIT-compiles the algorithm
        (against the given x0/remaining_vertices, though the compiled step
        remains valid for any x0/remaining_vertices afterwards -- see
        module docstring) and seeds `_carry` with the resulting (already
        optimized) population, so this cost is paid once up front instead
        of on the first timed solve() -- which then also starts from a
        genuinely warm population rather than a random one.

        Also exercises `problem._decode_node_rank` (called only by solve()'s
        own write-back, to report the visiting-order rank) so a real
        solve() call right after this one doesn't still pay its first-call
        compile. Unlike solver.py's old carry, `self._step_fn` itself IS
        the warm-resume path (see _ensure_built's own comment) -- there is
        no separate resume function left to warm here.

        Returns the compile time."""
        x0 = self._check_x0(x0)
        self._ensure_built(x0)
        x0_arr = jnp.asarray(self._to_padded_row(x0))
        params_arr = jnp.asarray(self._graph.view_param_values())
        anchor = self._compute_anchor(remaining_vertices)

        init_fn = _build_evosax_initial_carry_fn(
            self._problem, self._algo, self._algo_params, self._pop_size, anchor,
            x0=x0_arr, params=params_arr, **self._carry_kwargs)

        start = time.perf_counter()
        carry_in = init_fn(jax.random.PRNGKey(self._seed))
        carry_out = self._step_fn(carry_in, x0_arr, params_arr, anchor)
        jax.block_until_ready(carry_out)

        state_out, _key_out = carry_out
        best_X = np.asarray(state_out.best_solution[:self._problem.n_var])
        assign, cond_binary, _proj_branch, t, _wp, _psi = self._problem._extract_single(best_X)
        owner_variable = (np.argmax(assign, axis=-1) if self._problem.n_variables > 0
                          else np.zeros(0, dtype=int))
        node_rank = self._problem._decode_node_rank(
            owner_variable, np.asarray(cond_binary), t, np.asarray(anchor.node_active))
        jax.block_until_ready(node_rank)

        compile_time = time.perf_counter() - start

        self._carry = carry_out
        self._last_compile_time = compile_time
        return compile_time

    def solve(self, remaining_vertices, x0) -> bool:
        start = time.perf_counter()
        graph = self._graph
        x0 = self._check_x0(x0)
        was_already_built = self._problem is not None
        self._ensure_built(x0)
        problem = self._problem
        x0_arr = jnp.asarray(self._to_padded_row(x0))
        params_arr = jnp.asarray(self._graph.view_param_values())
        remaining_set = set(remaining_vertices)

        anchor = self._compute_anchor(remaining_vertices)
        # numpy-side copies for the write-back loops below (cheap, small
        # arrays -- avoids repeated jnp->python scalar coercion per instance).
        var_committed_np = np.asarray(anchor.var_committed)
        var_anchor_np = np.asarray(anchor.var_anchor)
        # Stashed for get_population_diagnostics() -- see its own comment.
        self._last_x0_arr = x0_arr
        self._last_params_arr = params_arr
        self._last_anchor = anchor

        self._last_solve_was_warm = was_already_built

        if self._carry is not None:
            # Resume the previous solve's population directly -- unlike
            # solver.py's old carry, evosax's State stores no raw per-
            # individual F/CV that could go stale across an x0/anchor
            # change: self._step_fn's own _ask/_tell recompute those fresh
            # from the CURRENT x0/anchor every call (see _ensure_built's own
            # comment), so there's no separate resume-refresh step needed
            # here the way build_resume_carry_fn used to be.
            carry_in = self._carry
            self._last_population_reused = True
        else:
            init_fn = _build_evosax_initial_carry_fn(
                problem, self._algo, self._algo_params, self._pop_size, anchor,
                x0=x0_arr, params=params_arr, **self._carry_kwargs)
            carry_in = init_fn(jax.random.PRNGKey(self._seed))
            self._last_population_reused = False

        carry_out = self._step_fn(carry_in, x0_arr, params_arr, anchor)
        self._carry = carry_out
        state_out, _key_out = carry_out
        best_X = np.asarray(state_out.best_solution[:problem.n_var])

        # `state_out.best_fitness` is a SOFT combined score frozen under
        # whatever w/cv_tol schedule was live when this individual last
        # improved (see lamarckian_ga.py's own State docstring) -- not the
        # raw, CURRENT-scene (F, CV) a caller wants to inspect (e.g. "does
        # the solver even know this pick is now infeasible after I moved a
        # block"). `state_out.best_F`/`best_CV`/`best_CV_proj` ARE that raw,
        # current-scene readout: `_tell` (lamarckian_ga.py) re-evaluates the
        # carried-over `best_solution` fresh against THIS call's x0/anchor
        # every generation (not just when a new individual beats it), for
        # exactly this reason -- so no separate recomputation is needed
        # here. This used to be a bare eager `_evaluate_population_jax`/
        # `_evaluate_projection_cv_jax` call at batch size 1, which (being
        # outside any persistent jax.jit) silently repaid a full XLA
        # retrace/recompile of `decode_rank_batched`'s `_topological_rank`
        # scan on every solve() call -- confirmed via profiling to cost
        # ~1.4s/cycle, dwarfing the GA search's own ~1s. Reading it off
        # `state_out` instead means it's computed once, batched, inside the
        # already-jitted `_step_fn`, exactly like every other per-
        # generation quantity here.
        self._last_F = float(state_out.best_F)
        self._last_CV = float(state_out.best_CV)
        self._last_CV_proj = float(state_out.best_CV_proj)

        assign, cond_binary, proj_branch, t, wp, psi = problem._extract_single(best_X)
        # A projected node's own pinned columns are never actually driven
        # anywhere by local refinement (apply_projections overwrites them
        # before anything reads them, every solve -- see its own docstring)
        # -- so the raw wp value extracted above is not the real answer for
        # those columns; substitute it in now, once, on the winning
        # individual, before persisting anything to self._waypoints.
        wp = jit_apply_projections(problem)(
            jnp.asarray(wp)[None, :, :], jnp.asarray(psi)[None, :],
            jnp.asarray(proj_branch)[None, :], params_arr,
            jnp.asarray(assign)[None, :, :], jnp.asarray(cond_binary)[None, :],
            jnp.asarray(t)[None, :], anchor.node_active, jnp.asarray(x0_arr),
            anchor.var_committed, anchor.var_anchor)[0]
        wp = np.asarray(wp)
        owner_variable = (np.argmax(np.asarray(assign), axis=-1) if problem.n_variables > 0
                          else np.zeros(0, dtype=int))

        # Copy the whole solved row through unconditionally for every
        # remaining node -- agent AND object columns alike, exactly like
        # MILPWaypointMPC.view_waypoints() (which likewise always returns
        # its whole solved W row, not filtered by which agent/object "owns"
        # each node). Downstream readers only ever index a node's row for an
        # agent/object a constraint or graph.get_agent_paths actually routes
        # through, so a column at a node that never references it is simply
        # never read, whatever it holds. Restricted to remaining_vertices
        # (problem.var_id_to_slot's write-back below is too, for the same
        # reason) so an already-passed node's frozen anchor is never
        # overwritten with this solve's masked-out/inert GA output. A node
        # id is directly its own row index into wp/t (spec.py gives every
        # graph node a row, matching MILP), so no node-id -> row-index
        # conversion is needed anywhere below.
        for node in remaining_set:
            self._waypoints[node, :] = self._to_packed_row(wp[node, :])

        # Report the DECODED visiting-order rank, not the raw priority `t`:
        # `t` only carries a meaningful order after kernel.py's topological
        # decode (see kernel.py's module docstring) -- writing back the raw
        # value here would silently reintroduce the exact bug that decoder
        # exists to fix, one level up (GraphOfConstraints.get_agent_paths
        # sorts nodes by view_t_by_node() too, and needs the same
        # already-topologically-valid order the route-cost computation
        # actually used). One value per node (kernel.py), so no
        # per-instance collapsing is needed here -- just index it directly.
        node_rank = np.asarray(problem._decode_node_rank(
            owner_variable, np.asarray(cond_binary), t, np.asarray(anchor.node_active)))
        for node in remaining_set:
            self._t_by_node_id[node] = float(node_rank[node])

        for var_id, slot in problem.var_id_to_slot.items():
            if var_committed_np[slot]:
                continue  # already correctly frozen; don't overwrite with this solve's inert output
            self._var_assignments[var_id] = int(owner_variable[slot])

        for phi_id in range(graph.num_phis):
            if phi_id in graph.phi_to_variable_map:
                var_id = graph.phi_to_variable_map[phi_id]
                self._assignments[phi_id] = self._var_assignments[var_id]
            elif phi_id in graph.phi_to_static_assignment_map:
                self._assignments[phi_id] = graph.phi_to_static_assignment_map[phi_id]
            else:
                self._assignments[phi_id] = -1

        self._last_solve_time = time.perf_counter() - start
        return True

    def view_waypoints(self):
        return self._waypoints

    def view_object_waypoints(self):
        """(num_nodes, total object width) solved object positions -- a view
        onto _waypoints' object-column slice (see view_waypoints()),
        matching its node-indexed convention. NaN for any node not yet
        covered by any solve() call; every node in the last solve()'s
        remaining_vertices carries a real, GA-solved value regardless of
        whether any constraint referenced it there."""
        return self._waypoints[:, self._graph.agent_col_offsets[-1]:]

    def view_assignments(self):
        return self._assignments

    def view_var_assignments(self):
        return self._var_assignments

    def view_t_by_node(self):
        return self._t_by_node_id

    def get_last_fitness(self):
        """(F, CV, CV_proj) of the population member solve() last selected,
        raw and freshly evaluated against that solve() call's own x0/anchor
        (see the comment at its computation in solve() for why this can't
        just read `state.best_fitness`). CV_proj (_evaluate_projection_cv_
        jax, solver.py) is a SEPARATE read-only check of whatever
        constraints were resolved by a `proj=` analytic projection instead
        of an ordinary AL residual -- CV alone is structurally blind to one
        of those silently failing to hold (e.g. a clipped/degenerate
        analytic-IK branch on an out-of-reach target -- see that function's
        own docstring). None, None, None before the first solve()."""
        return self._last_F, self._last_CV, self._last_CV_proj

    def get_population_diagnostics(self):
        """Per-population-member (F, CV, CV_proj, owner_variable,
        hard_score, rank, will_be_evicted) as of the last solve()/warmup()
        call -- unlike get_last_fitness() (which reports only the single
        selected/best member), this reports the WHOLE population
        `state.population` carries, freshly evaluated against that call's
        own x0/anchor (same never-trust-stale-fitness stance as
        get_last_fitness()/reseed() -- see get_last_fitness()'s own
        comment for why). A debug/visualization accessor, not a hot path
        -- returns a PopulationDiagnostics of plain numpy arrays, not jax.

        `hard_score`/`rank`/`will_be_evicted` mirror
        SmallContinuousVRPSolver.reseed()'s own CV-dominant eviction
        ranking exactly (small_continuous_vrp.py: `F + 1e6 * max(0,
        CV_total - reseed_cv_tol)`, `CV_total = CV + CV_proj`), read off
        the active algorithm instance via getattr so this degrades
        gracefully under algorithm="lamarckian_al" (no reseed mechanism at
        all): `will_be_evicted` is then all False (n_evict=0) and
        `hard_score` uses reseed_cv_tol=0 (a plain hard CV-dominant
        score).

        `owner_variable[i]` is member `i`'s assigned agent id per
        assignment-variable slot (`argmax` of that member's one-hot
        `assign`) -- e.g. `[0, 1]` means variable slot 0 went to agent 0
        ("r0") and slot 1 to agent 1 ("r1").

        `var_committed`/`var_anchor` are read straight off this call's
        anchor (problem.AnchorState) -- see PopulationDiagnostics' own
        comment for why a committed slot means the WHOLE population sharing
        one owner for it is expected, not a search failure: a committed
        slot is never actually offered as free by the discrete solver in
        the first place.

        None before the first solve()/warmup() call."""
        if self._carry is None:
            return None
        problem = self._problem
        state_out, _key_out = self._carry
        X, _mu, _lam, _rho = _split_genome(problem, state_out.population)

        F, CV = _evaluate_population_jax(
            problem, X, self._last_x0_arr, self._last_params_arr, self._last_anchor)
        CV_proj = _evaluate_projection_cv_jax(
            problem, X, self._last_x0_arr, self._last_params_arr, self._last_anchor)
        F = np.asarray(F)
        CV = np.asarray(CV)
        CV_proj = np.asarray(CV_proj)
        pop = F.shape[0]

        assign, _cond_binary, _proj_branch, _t, _wp0, _psi0 = problem._extract_batch(X)
        owner_variable = (np.argmax(np.asarray(assign), axis=-1) if problem.n_variables > 0
                          else np.zeros((pop, 0), dtype=int))

        reseed_cv_tol = getattr(self._algo, "_reseed_cv_tol", 0.0)
        n_evict = int(getattr(self._algo, "_n_evict", 0))
        hard_score = F + 1e6 * np.maximum(0.0, (CV + CV_proj) - reseed_cv_tol)
        rank = np.argsort(np.argsort(hard_score))       # 0 = best (lowest hard_score)
        will_be_evicted = np.zeros(pop, dtype=bool)
        if n_evict > 0:
            worst_first = np.argsort(-hard_score)        # mirrors reseed()'s own selection
            will_be_evicted[worst_first[:n_evict]] = True

        var_committed = np.asarray(self._last_anchor.var_committed, dtype=bool)
        var_anchor = np.asarray(self._last_anchor.var_anchor, dtype=int)

        return PopulationDiagnostics(
            F=F, CV=CV, CV_proj=CV_proj, owner_variable=owner_variable,
            hard_score=hard_score, rank=rank, will_be_evicted=will_be_evicted,
            n_evict=n_evict, var_committed=var_committed, var_anchor=var_anchor)

    def get_last_solve_time(self):
        return self._last_solve_time

    def was_last_solve_warm(self):
        """True if the last solve() reused the already-compiled GA (i.e. it
        wasn't the first solve()/warmup() call ever made on this solver
        instance -- x0/remaining_vertices may differ freely either way,
        since the compiled GA is valid for any of either, see module
        docstring)."""
        return self._last_solve_was_warm

    def was_last_population_reused(self):
        """True if the last solve() started from a previous carry's
        population (its own _ask/_tell re-score it fresh under the current
        x0/anchor, see module docstring) rather than a cold random one."""
        return self._last_population_reused

    def get_last_compile_time(self):
        """Compile time from the last warmup() call (0.0 if warmup() was
        never called)."""
        return self._last_compile_time

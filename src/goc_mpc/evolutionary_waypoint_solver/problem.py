"""GraphOrderingRelaxed: the real-relaxed decision-variable layout and problem
metadata the GA/L-BFGS solver (solver.py) optimizes over, built by
spec.build_graph_ordering_problem() (spec.py).

The decision vector `x` is four contiguous blocks:
  assign      (n_variables, n_agents)   -- one-hot/continuous-relaxed agent
                                            choice per assignable routing
                                            variable.
  cond_binary (n_cond_vars,)            -- continuous-relaxed [0, 1] score per
                                            GraphOfConstraints binary_cond_sym_var,
                                            GA-searched exactly like `assign`;
                                            eq/ineq constraint functions never
                                            need it, only problem._batched does
                                            (kernel.make_graph_kernel's gates).
  t           (n_nodes,)                -- ordering-priority score, one per
                                            graph node -- purely a tie-break,
                                            decoded into an actual visiting
                                            order by a topological sort (see
                                            kernel.py's module docstring).
  wp          (n_nodes, state_dim)      -- per-node configuration,
                                            `wp[node] = [agent_0 | agent_1 |
                                            ... | object_0 | ...]`, mirroring
                                            MILP's per-node joint configuration
                                            `W` (graph_of_constraints.cpp:79's
                                            `total_dim`).
Ordering edges are `(u, v, gate_fn)` node-index pairs (kernel.make_graph_kernel's
docstring); `gate_fn=None` means an always-active hard edge, otherwise
`gate_fn(owner_variable, cond_binary) -> bool`.
"""

import weakref
from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np

from .kernel import make_graph_kernel, decode_rank_batched


# One analytic-elimination substitution (projection.ProjOperator, resolved
# by spec.py's _resolve_projections) -- see apply_projections' docstring for
# how a list of these gets applied.
#   write_node: node id whose row gets written.
#   pinned_cols: (w,) int array -- ABSOLUTE column indices within
#       write_node's row to overwrite. Only meaningful when
#       owner_var_slot is None (a plain agent_q(k)/object_q(k) pin, static
#       columns known at spec-build time); ignored otherwise.
#   owner_var_slot / owner_cols_per_agent: the var_agent_q(var_id) pin case
#       (spec.py's _resolve_pin_columns "dynamic" branch) -- None/None for
#       an ordinary static pin. `owner_var_slot` indexes `assign`'s own
#       `n_variables` axis: which assignable variable's own GA-searched
#       choice decides WHICH agent's columns actually get written, decoded
#       via argmax exactly like every other var_agent_q consumer in this
#       solver (spec.py's _make_row_resolver/_resolve_holds), respecting
#       anchor.var_committed/var_anchor once that variable's owner has been
#       committed. `owner_cols_per_agent` is a spec-build-time-precomputed
#       `(n_agents, w)` int array -- row k lists the ABSOLUTE columns this
#       pin would write if agent k turned out to be the owner (k*slot_width
#       + each local offset) -- apply_projections gathers row `owner` out
#       of it per population member, so no `slot_width` arithmetic is
#       needed at runtime. Unlike a static pin's fixed `pinned_cols`, WHICH
#       columns get written varies per individual/generation (whichever
#       agent that individual's own assign currently favors) -- this is
#       exactly why these columns can never join wp_free_idx's static
#       shrink (see gather_free_wp's docstring): no single column is ever
#       unconditionally dead.
#   node_locals: node id(s) whose rows read_fn is called with (len 1 for a
#       node constraint, 2 (u, v) for a relational edge constraint).
#   read_cols / write_cols: frozensets of (node, col) this projection's
#       `func` reads / its pin could ever write (a var_agent_q(...) dynamic
#       pin's write_cols is the union over every candidate agent's slot).
#       spec.py's _resolve_projections builds the projection list in an
#       order where every entry follows the ones whose write_cols meet its
#       read_cols (a data-dependency topo sort -- apply_projections threads
#       wp through its walk, so a reader must come after its writer); these
#       are kept on the entry so other consumers (small_continuous_vrp,
#       future search-space analysis) can see the dependency structure
#       without re-deriving it. A param(id) read is not a wp column and
#       contributes nothing here; an FK (agent_link_pos/_rot) read
#       contributes that agent's whole config slice at node_locals[0].
#   read_fn: UNBATCHED callable(*rows) -> tuple of arrays, one per the
#       ProjOperator's own `reads` entries, in order (empty tuple if
#       `reads=()`) -- built by spec.py from the same per-placeholder
#       resolution every other compiled constraint in this module uses,
#       restricted to the static/FK cases (see spec.py's
#       _resolve_read_component).
#   func: UNBATCHED callable(*read_values, psi, branch) -> value (see
#       projection.ProjOperator's own docstring) -- None when `table` is
#       set instead (the reads=() and continuous_params=0 case: spec.py
#       precomputes every branch's value once, in plain numpy, rather than
#       calling this on every batched pass).
#   continuous_params: width of this projection's own slice of the global
#       `psi` block.
#   psi_bounds: (lo, hi) box for this projection's own psi slice (copied
#       from ProjOperator.psi_bounds; ignored when continuous_params == 0).
#   psi_slice / branch_slice: this projection's own slice of the global
#       flat `psi` / `proj_branch` blocks (GraphOrderingRelaxed below).
#   discrete_params: cardinality of this projection's own branch selector.
#   table: (discrete_params, len(pinned_cols)) numpy array, or None -- see
#       `func` above.
#   is_static: True iff this projection's value cannot change across any of
#       one local_refine call's L-BFGS iterations/backtracking trials or
#       outer AL rounds -- continuous_params == 0 (no psi to gradient-
#       refine) and every read `func` actually consumes (an underscore-
#       prefixed func parameter name is the documented "kept only for the
#       constraint's free-variable-coverage check, never truly read"
#       convention -- see spec.py's _classify_static_projection) resolves to
#       a param(id) placeholder, itself runtime-constant for the whole
#       step() call. Always False when `table` is set -- a tabled entry is
#       already an O(1) gather, precompute_static_projections has nothing
#       to add for it. See precompute_static_projections/apply_projections'
#       own docstrings for how this is used to avoid recomputing a
#       potentially expensive analytic elimination (e.g. an 8-branch
#       closed-form IK) on every one of a generation's many merit calls.
#   gate_fn: None (unconditional -- the pin always applies) OR an UNBATCHED-
#       over-nothing callable `gate_fn(rank) -> (pop,)` returning a hard
#       {0.0, 1.0} mask, `rank` the decoded node-rank `(pop, n_nodes)` int
#       array (spec.py's _decode_rank_batched). apply_projections then
#       BLENDS -- `g*value + (1-g)*cur` -- so per individual the write is
#       either exact (`value`, gradient zero, a real pin) or a no-op
#       (`cur`, the column stays a free decision variable whatever residual
#       still governs it drives). Used for the auto-derived stationary-object
#       / rigid-carry substitutions (spec.py's _resolve_stationary_objects /
#       _resolve_holds), whose applicability depends on where the solved
#       schedule places a node relative to a hold's span. A gated entry is
#       never `is_static` and its columns never join wp_pinned_mask's static
#       shrink (they're live for the gate-off case, exactly like a dynamic
#       pin's). Not supported together with a var_agent_q(...) dynamic pin
#       (owner_var_slot is not None) yet -- apply_projections raises.
#   owner_aware: True iff `func` takes a trailing `owner` (resolved agent
#       id) arg -- projection.ProjOperator.owner_aware, DYNAMIC pins only,
#       for an elimination whose form differs per agent (e.g. per-arm
#       analytic IK). apply_projections passes owner_variable[owner_var_slot]
#       to `func`. Forces is_static/table off (the value depends on a
#       runtime assignment).
ProjectionEntry = namedtuple(
    "ProjectionEntry",
    ["write_node", "pinned_cols", "node_locals", "read_fn", "func",
     "continuous_params", "psi_slice", "psi_bounds", "branch_slice",
     "discrete_params", "table", "is_static",
     "owner_var_slot", "owner_cols_per_agent",
     "read_cols", "write_cols", "gate_fn", "owner_aware"],
    defaults=[None, False])  # gate_fn (unconditional), owner_aware


# Bundles remaining_vertices' runtime effect on an otherwise-fixed-size
# problem (see spec.py's module docstring) into one pytree,
# threaded alongside x0 through decode_and_cost/_batched/merit/local_refine/
# _routing_local_search_batched/gen_step/step (solver.py, kernel.py) -- built
# fresh each solve() call by EvolutionaryWaypointSolver._compute_anchor
# (mpc.py) from remaining_vertices plus the solver's own persisted
# _waypoints/_var_assignments history. The current call's real state lives in
# the separate `x0` argument threaded alongside `anchor` everywhere (see
# apply_anchor) rather than inside AnchorState itself -- x0 is now the FULL
# configuration (state_dim,), the same value routing (kernel.py, sliced down
# to its agent-depot view via problem.agent_depot) already needed every call,
# so there is exactly one "current real state" flowing through this whole
# stack, not two.
#   node_active: (n_nodes,) bool -- is this node still in
#       remaining_vertices (a free decision variable) or already passed (an
#       anchored constant)?
#   anchor_wp: (n_nodes, state_dim) float -- for a passed node, its
#       last-committed row -- i.e. whatever _waypoints held for it as
#       of the last solve() where it was still active (irrelevant/unused
#       where node_active=True). This is the "planned" reading: a genuinely
#       per-node historical value, frozen at whatever the GA last solved it
#       to.
#   var_committed: (n_variables,) bool -- has ANY instance using this
#       assignable variable already passed? If so its assignment must stay
#       pinned (mirrors MILP's "don't let the routing solve reassign this
#       hold's holder mid-grasp").
#   var_anchor: (n_variables,) int -- the committed real-agent id for a
#       var_committed variable (irrelevant/unused otherwise).
AnchorState = namedtuple("AnchorState", ["node_active", "anchor_wp", "var_committed", "var_anchor"])


def agent_depot(problem, x0):
    """Slices the full-configuration runtime `x0` (state_dim,) down to the
    (n_agents, dim) agent-depot view kernel.py's routing math expects -- the
    object suffix, if any, is irrelevant to routing (see kernel.py's module
    docstring: an agent's route cost/arrival time only ever depends on agent
    positions) and simply dropped here."""
    return x0[: problem.n_agents * problem.dim].reshape(problem.n_agents, problem.dim)


def pad_to_state_dim(agent_x0_flat, state_dim):
    """Zero-pads a flat agent-only x0 (n_agents*dim,) out to the full
    state_dim width -- used wherever only a structural/default x0 is
    available (no real object state to report), e.g. build_initial_carry_fn's
    cold-start bootstrap or run_lamarckian_al's x0=None default (solver.py).
    Safe to zero-fill rather than track precisely: both call sites only ever
    run under an anchor with node_active all-True (nothing passed yet), so
    apply_anchor's wp_eff_live substitution -- the only reader of x0's object
    columns -- is never actually selected for any row there regardless of
    what these zeros hold."""
    n = agent_x0_flat.shape[-1]
    if n == state_dim:
        return agent_x0_flat
    return jnp.concatenate([agent_x0_flat, jnp.zeros(state_dim - n, dtype=agent_x0_flat.dtype)])


def apply_anchor(problem, assign, wp, anchor, x0):
    """Splices `anchor` into batched (pop, ...) `assign`/`wp` decision
    variables, returning (assign_eff, wp_eff_frozen, wp_eff_live): a node no
    longer in remaining_vertices reads back as either its last-committed
    "planned" constant (wp_eff_frozen, anchor.anchor_wp) or this call's real
    state (wp_eff_live, the full-configuration `x0` argument -- state_dim
    wide, broadcast across every node since x0 is one global joint
    state, not a per-node history) instead of the free GA/L-BFGS value --
    spec.py's build_graph_ordering_problem compiles each constraint to read
    from whichever of the two it was registered against (live_phi_ids), so
    which one "wins" for a given passed node is a per-constraint choice,
    not a per-node one (a variable no longer in remaining_vertices has no
    such choice -- assign_eff is always its committed one-hot). Callers
    should use assign_eff/
    wp_eff_frozen/wp_eff_live everywhere a constraint or routing decision is
    made (never the raw assign/wp) -- this is the single substitution point
    that lets a node/edge constraint anchored at an already-passed node stay
    correctly enforced (against a known constant) rather than being dropped,
    mirroring MILP's boundary-edge substitution, with no changes needed to
    spec.py's constraint compilation itself beyond the frozen/live pick.
    jnp.where's VJP naturally zeroes gradient into anchored wp slots, so
    solver.py's L-BFGS local refinement needs no extra masking. Routing
    (kernel.py) never reads a passed node's row either way (masked out via
    node_active regardless of value), so callers feeding it wp can pass
    either variant -- solver.py consistently uses wp_eff_frozen there, an
    arbitrary pick."""
    wp_eff_frozen = jnp.where(anchor.node_active[None, :, None], wp, anchor.anchor_wp[None, :, :])
    wp_eff_live = jnp.where(anchor.node_active[None, :, None], wp, x0[None, None, :])
    if problem.n_variables > 0:
        anchor_one_hot = jax.nn.one_hot(anchor.var_anchor, problem.n_agents, dtype=assign.dtype)
        assign_eff = jnp.where(anchor.var_committed[None, :, None], anchor_one_hot[None, :, :], assign)
    else:
        assign_eff = assign
    return assign_eff, wp_eff_frozen, wp_eff_live


def gather_free_wp(problem, wp):
    """Drops every projection-pinned column out of `(pop, n_nodes,
    state_dim)` wp, returning `(pop, n_wp_free)` -- the columns no
    projection ever writes (problem.wp_free_idx, GraphOrderingRelaxed.
    __init__). A pinned column's incoming value is always overwritten by
    apply_projections before anything else ever reads it (see that
    function's docstring), so it carries no real degree of freedom;
    local_refine gradient-descends only this reduced vector instead of
    wastefully carrying those dead columns through every L-BFGS iteration.
    Identity (mod reshape) when problem has no projections at all
    (wp_free_idx == arange(n_nodes*state_dim))."""
    pop = wp.shape[0]
    return wp.reshape(pop, -1)[:, problem.wp_free_idx]


def scatter_free_wp(problem, wp_free, base):
    """Inverse of gather_free_wp: splices `(pop, n_wp_free)` free-column
    values into `base`'s `(pop, n_nodes, state_dim)` shape at their
    original positions, leaving every pinned column exactly as `base`
    already held it there -- irrelevant regardless, since apply_projections
    always overwrites a pinned column right after (see gather_free_wp) --
    so callers reconstructing a fresh full wp from scratch may pass
    `base=jnp.zeros((pop, n_nodes, state_dim))` and rely on that overwrite
    rather than tracking a real placeholder value."""
    pop = wp_free.shape[0]
    flat = base.reshape(pop, -1).at[:, problem.wp_free_idx].set(wp_free)
    return flat.reshape(pop, problem.n_nodes, problem.state_dim)


def precompute_static_projections(problem, wp0, proj_branch, params):
    """Evaluates every "generation-static" projection's value ONCE
    (ProjectionEntry.is_static -- continuous_params == 0 and every read
    `func` actually consumes is a param(id) placeholder), rather than
    letting apply_projections recompute it from scratch on every one of a
    single local_refine call's many merit_and_grad calls (~1000s per
    generation for a tuned outer_iters/inner_maxiter/ls_max_trials budget --
    see the profiling this followed from). Safe because a static entry's
    value provably cannot change across any of those calls: `proj_branch`/
    `params` are themselves fixed for local_refine's whole call (proj_branch
    is GA-searched once per generation, not L-BFGS-refined; params is a
    plain runtime constant), and continuous_params == 0 means there's no
    psi for L-BFGS to move underneath it either.

    `wp0` supplies whatever `rows` a static entry's read_fn needs for its
    documented-unused (underscore-prefixed func parameter) reads -- e.g.
    UR5e's analytic-IK projection reads its own about-to-be-pinned FK
    placeholders only to satisfy the constraint's free-variable-coverage
    check, then discards them (see spec.py's _classify_static_projection) --
    so ANY value works there, including wp0's own pre-refinement one; a
    real param(id) read (the only kind that can affect an is_static entry's
    actual value) never touches `rows` at all.

    Returns a `{id(entry): value}` cache for apply_projections' own
    `static_cache` argument -- keyed by identity since `problem.projections`
    is a plain fixed-length Python list, walked identically (in the same
    order) by both functions.

    `problem.projections` is dependency-ordered (spec.py's
    _resolve_projections), so a chained static entry -- one reading a column
    an earlier projection pins -- is handled by threading a working `wp`
    through this walk: each static entry's value is spliced back in before a
    later entry reads that column, exactly as apply_projections does. Tabled
    entries (value known at build time) are spliced too. An entry whose
    value this function CAN'T reproduce (non-static + non-tabled, or a
    var_agent_q(...) dynamic pin needing `assign`) never has a static
    descendant reading its columns -- _resolve_projections demotes any such
    descendant to non-static up front."""
    cache = {}
    wp = wp0
    for entry in problem.projections:
        if entry.table is not None:
            # Splice the branch's tabled value so a later chained static
            # entry reads it -- only for a static write target; a
            # var_agent_q(...) tabled pin's target needs `assign` (absent
            # here), and _resolve_projections has already demoted any static
            # entry downstream of it.
            if entry.owner_var_slot is None:
                branch = jnp.argmax(proj_branch[:, entry.branch_slice], axis=-1)  # (pop,)
                value = jnp.asarray(entry.table)[branch]  # (pop, w)
                wp = wp.at[:, entry.write_node, entry.pinned_cols].set(value)
            continue
        if not entry.is_static:
            continue
        branch = jnp.argmax(proj_branch[:, entry.branch_slice], axis=-1)  # (pop,)
        rows = tuple(wp[:, node, :] for node in entry.node_locals)
        psi_i = jnp.zeros((branch.shape[0], entry.continuous_params))  # always 0-width: is_static implies continuous_params == 0
        # is_static entries only ever read param(id) placeholders (see
        # _classify_static_projection), so read_fn's owner_variable / x0 args
        # are unused here -- zero placeholders are fine.
        ov = jnp.zeros((branch.shape[0], problem.n_variables), dtype=jnp.int32)
        x0z = jnp.zeros((problem.state_dim,))

        def _one(rows_1, psi_1, branch_1, ov_1, entry=entry, params=params, x0z=x0z):
            read_vals = entry.read_fn(rows_1, params, ov_1, x0z)
            return jnp.asarray(entry.func(*read_vals, psi_1, branch_1))

        value = jax.vmap(_one, in_axes=(0, 0, 0, 0))(rows, psi_i, branch, ov)
        cache[id(entry)] = value
        if entry.owner_var_slot is None:
            wp = wp.at[:, entry.write_node, entry.pinned_cols].set(value)
    return cache


def apply_projections(problem, wp, psi, proj_branch, params, assign=None, anchor=None, static_cache=None,
                      cond_binary=None, t=None, node_active=None, x0=None, only_entries=None,
                      var_committed=None, var_anchor=None):
    """Splices every registered analytic-elimination substitution
    (spec.py's _resolve_projections, projection.ProjOperator) into batched
    `(pop, n_nodes, state_dim)` wp, reading batched `psi`
    `(pop, n_psi)`/`proj_branch` `(pop, n_branch)` for whatever continuous/
    discrete parameters each one needs, and the POPULATION-INVARIANT
    `params` array (GraphOfConstraints.view_param_values(), same value
    every other compiled constraint reads a param(id) placeholder from) for
    any projection whose `reads` includes one -- set_param overwrites it in
    place between solves, so a param-reading projection always reflects the
    CURRENT value, never a stale one baked in at spec-build time.

    `assign`/`anchor` (batched `(pop, n_variables, n_agents)` / problem.
    AnchorState): only needed when some projection pins a var_agent_q(...)
    row instead of a plain agent_q(k)/object_q(k) one (ProjectionEntry.
    owner_var_slot is not None -- see its own docstring) -- such a pin's
    WRITE TARGET, not just its value, is dynamic: which agent's columns
    actually get overwritten is that variable's own GA-searched choice,
    decoded via `jnp.argmax(assign[:, owner_var_slot, :], axis=-1)` exactly
    like every other var_agent_q consumer in this solver (spec.py's
    _make_row_resolver/_resolve_holds), honoring `anchor.var_committed`/
    `var_anchor` once that variable has been committed -- this function
    runs BEFORE apply_anchor (see below), so it can't just read an
    already-computed assign_eff and must apply that same substitution
    itself. Omit both (the default) for any problem with no var_agent_q
    pin at all -- every existing caller that never touches this feature is
    unaffected; passing an entry that needs them without providing them
    raises rather than silently writing into the wrong (or every) agent's
    row.

    `static_cache` (optional, from precompute_static_projections): when an
    entry's value is already cached there (its own `is_static` says it's
    safe to -- see that function's docstring), this reuses the cached value
    instead of recomputing it, so a caller invoking apply_projections
    repeatedly within one local_refine call (make_batched_local_refine, the
    ~1000s-of-merit_and_grad-calls hot path) pays for each static
    projection's real work exactly once per generation rather than on every
    call. Omitted entirely (the default), every entry is recomputed from
    scratch every time -- the original, always-correct behavior every
    other caller (mpc.py's write-back, this function's own doctests) keeps
    using unchanged.

    Call this FIRST, on the free GA-searched wp, and feed its return value
    as apply_anchor's own `wp` argument (every call site in solver.py does
    exactly this) -- an already-passed node's frozen anchor is always the
    true ground truth regardless of whether it was ever projected, so
    apply_anchor's substitution must win where the two overlap. mpc.py's
    write-back applies this same function (pop=1) to the WINNING
    individual before persisting a node's row, for the identical reason: a
    pinned column is never actually driven anywhere by local refinement --
    apply_projections overwrites it before any constraint/kernel/gradient
    ever reads it, so its gradient there is exactly zero (see
    _resolve_projections' docstring) -- so the raw searched wp value
    sitting in that column is not the real answer; this substitution is.

    Each projection's own branch is decoded via argmax over its
    `branch_slice` of proj_branch, exactly like `assign`'s one-hot-over-
    agents decode elsewhere in this module. A tabled projection (`entry.
    table` set -- see ProjectionEntry's docstring) is a plain gather, no
    grad needed since it never depends on a traced value; an untabled one
    vmaps its (read_fn, func) pair over the population axis, matching how
    every other compiled constraint in this module is built for a single
    example and vmapped by its caller.

    `problem.projections` is dependency-ordered (spec.py's
    _resolve_projections): this loop threads `wp` through, so an entry whose
    `reads` reference a column an earlier entry `pins` sees that pinned value
    (its `rows` are sliced from the running `wp`, not the original).

    `cond_binary`/`t`/`node_active`: needed only when some entry is GATED
    (ProjectionEntry.gate_fn -- the auto-derived stationary/rigid-carry
    substitutions). The decoded node-rank `(pop, n_nodes)` is computed ONCE
    here (kernel.decode_rank_batched, via problem._decode_node_rank -- the
    same decoder the gated residuals use) and each gate_fn maps it to a hard
    {0,1} mask; the pin is then BLENDED in
    (`g*value + (1-g)*cur`) rather than written unconditionally. `node_active`
    falls back to `anchor.node_active`, then to all-True. A gated entry with
    none of these available raises.

    `only_entries`: apply just this subsequence of `problem.projections`
    instead of all of it -- structure.node_candidates uses it to resolve one
    self-contained projection's candidate rows without running the rest of
    the chain (and, when the subset carries no gate, without the rank
    decode). The caller is responsible for passing a subset that is
    self-contained (reads nothing another, un-included, entry pins)."""
    # `only_entries`: apply just this subset (structure.node_candidates
    # resolving ONE self-contained projection's row -- skips the rest of the
    # chain and, when the subset is gate-free, the rank decode entirely).
    entries = problem.projections if only_entries is None else list(only_entries)

    pop = wp.shape[0]
    if x0 is None:
        x0 = jnp.zeros((problem.state_dim,))
    if problem.n_variables > 0 and assign is not None:
        owner_variable = jnp.argmax(assign, axis=-1)  # (pop, n_variables)
    else:
        owner_variable = jnp.zeros((pop, problem.n_variables), dtype=jnp.int32)

    rank = None
    if any(e.gate_fn is not None for e in entries):
        if cond_binary is None or t is None or assign is None:
            raise ValueError(
                "a gated projection (ProjectionEntry.gate_fn) needs "
                "apply_projections' `assign`, `cond_binary` and `t` arguments to "
                "decode the node-rank its gate is evaluated against")
        na = node_active
        if na is None:
            na = anchor.node_active if anchor is not None else jnp.ones((problem.n_nodes,), dtype=bool)
        rank = decode_rank_batched(problem._decode_node_rank, assign, cond_binary, t, na)  # (pop, n_nodes)

    for entry in entries:
        if static_cache is not None and id(entry) in static_cache:
            value = static_cache[id(entry)]
        elif entry.table is not None:
            branch_scores = proj_branch[:, entry.branch_slice]
            branch = jnp.argmax(branch_scores, axis=-1)  # (pop,)
            value = jnp.asarray(entry.table)[branch]  # (pop, w)
        else:
            branch_scores = proj_branch[:, entry.branch_slice]
            branch = jnp.argmax(branch_scores, axis=-1)  # (pop,)
            rows = tuple(wp[:, node, :] for node in entry.node_locals)  # each (pop, state_dim)
            psi_i = psi[:, entry.psi_slice]  # (pop, continuous_params)

            def _one(rows_1, psi_1, branch_1, ov_1, entry=entry, params=params, x0=x0):
                read_vals = entry.read_fn(rows_1, params, ov_1, x0)
                if entry.owner_aware:
                    # DYNAMIC pin whose elimination differs per agent -- hand
                    # `func` the resolved owner id (see ProjOperator.func).
                    return jnp.asarray(entry.func(
                        *read_vals, psi_1, branch_1, ov_1[entry.owner_var_slot]))
                return jnp.asarray(entry.func(*read_vals, psi_1, branch_1))

            value = jax.vmap(_one, in_axes=(0, 0, 0, 0))(rows, psi_i, branch, owner_variable)

        if entry.gate_fn is not None:
            if entry.owner_var_slot is not None:
                raise NotImplementedError(
                    "a gated projection combined with a var_agent_q(...) dynamic "
                    "pin (ProjectionEntry.owner_var_slot) is not supported")
            g = entry.gate_fn(rank)[:, None]  # (pop, 1) hard {0, 1}
            cur = wp[:, entry.write_node, entry.pinned_cols]  # (pop, w)
            value = g * value + (1.0 - g) * cur

        if entry.owner_var_slot is None:
            wp = wp.at[:, entry.write_node, entry.pinned_cols].set(value)
        else:
            if assign is None:
                raise ValueError(
                    "a var_agent_q(...)-pinned projection (ProjectionEntry."
                    "owner_var_slot is not None) needs apply_projections' own "
                    "`assign` argument to resolve which agent's row to write "
                    "into -- see this function's docstring")
            owner = jnp.argmax(assign[:, entry.owner_var_slot, :], axis=-1)  # (pop,)
            vc = var_committed if var_committed is not None else (
                anchor.var_committed if anchor is not None else None)
            vanc = var_anchor if var_anchor is not None else (
                anchor.var_anchor if anchor is not None else None)
            if vc is not None:
                owner = jnp.where(vc[entry.owner_var_slot], vanc[entry.owner_var_slot], owner)
            cols = jnp.asarray(entry.owner_cols_per_agent)[owner]  # (pop, w) -- per-individual ABSOLUTE cols
            row = wp[:, entry.write_node, :]
            new_row = jax.vmap(lambda r, c, v: r.at[c].set(v))(row, cols, value)
            wp = wp.at[:, entry.write_node, :].set(new_row)
    return wp


_apply_projections_jit = weakref.WeakKeyDictionary()  # problem -> {only_key -> jitted fn}


def jit_apply_projections(problem, only_entries=None):
    """A `jax.jit`-compiled `apply_projections` bound to `problem` (and, if
    given, `only_entries`) -- both closed over as static -- cached per
    problem instance. The returned callable takes `(wp, psi, proj_branch,
    params, assign, cond_binary, t, node_active, x0)`, all traced arrays,
    every one required (no None defaults under jit), plus optional
    `var_committed`/`var_anchor` traced arrays -- pass them (from `anchor`)
    when a committed var_agent_q(...) pin must write into the agent it was
    frozen to rather than this genome's searched owner.

    For a caller hitting `apply_projections` repeatedly on ONE problem at
    fixed array shapes but changing values -- structure.node_candidates
    every MPC cycle, mpc.py's per-step write-back -- eager execution of the
    projection chain (an 8-branch analytic IK vmapped over its branch
    population, plus the topological-rank decode) costs ~hundreds of ms to
    seconds per call; the jitted version pays that once as a compile and
    then runs in milliseconds. `static_cache` / `anchor` aren't exposed
    here (a jitted resolve is always fresh); splice a passed-node freeze
    into `node_active` beforehand if needed."""
    only_key = None if only_entries is None else tuple(id(e) for e in only_entries)
    per_problem = _apply_projections_jit.setdefault(problem, {})
    f = per_problem.get(only_key)
    if f is None:
        oe = None if only_entries is None else tuple(only_entries)

        def _run(wp, psi, proj_branch, params, assign, cond_binary, t, node_active, x0,
                 var_committed=None, var_anchor=None):
            return apply_projections(problem, wp, psi, proj_branch, params, assign=assign,
                                     cond_binary=cond_binary, t=t, node_active=node_active,
                                     x0=x0, only_entries=oe,
                                     var_committed=var_committed, var_anchor=var_anchor)
        f = jax.jit(_run)
        per_problem[only_key] = f
    return f


def full_active_anchor(problem):
    """AnchorState with every node "active" (free, not yet passed) and no
    committed variable assignments -- the correct anchor for the very first
    solve of a fresh problem, before anything has been solved/written back.
    EvolutionaryWaypointSolver._compute_anchor (mpc.py) naturally reduces to
    this whenever remaining_vertices == every graph node (in particular, the
    genuine first-ever call, since GraphOfConstraintsMPC always starts
    remaining_phases as every node -- goc_mpc.py)."""
    return AnchorState(
        node_active=jnp.ones((problem.n_nodes,), dtype=bool),
        anchor_wp=jnp.zeros((problem.n_nodes, problem.state_dim)),
        var_committed=jnp.zeros((problem.n_variables,), dtype=bool),
        var_anchor=jnp.zeros((problem.n_variables,), dtype=jnp.int32),
    )


def _infer_constraint_size(fn, n_variables, n_agents, n_nodes, state_dim, n_cond_vars, n_params):
    """Output width of `fn` on this problem's shapes, without ever running it:
    `jax.eval_shape` traces `fn` abstractly from the dummy args' shape/dtype
    alone. A real (eager) call here would dispatch every un-fused jnp
    primitive inside `fn` (an eq/ineq constraint can embed a full FK/IK
    chain) as its own tiny XLA program -- dozens of one-off compiles just to
    learn a shape."""
    dummy_assign = np.zeros((1, n_variables, n_agents))
    dummy_cond_binary = np.zeros((1, n_cond_vars))
    dummy_t = np.zeros((1, n_nodes))
    dummy_wp = np.zeros((1, n_nodes, state_dim))
    dummy_node_active = np.ones((n_nodes,), dtype=bool)
    dummy_x0 = np.zeros((state_dim,))
    dummy_params = np.zeros((n_params,))
    out_shape = jax.eval_shape(fn, dummy_assign, dummy_cond_binary, dummy_t, dummy_wp, dummy_wp,
                                dummy_node_active, dummy_x0, dummy_params)
    return out_shape.shape[1]


class GraphOrderingRelaxed:
    def __init__(self, instance_sources, n_variables, ordering_edges, x0, wp_bounds,
                 instance_node, n_nodes, state_dim,
                 n_cond_vars=0, objective="avg", edge_cost_fn=None,
                 eq_constraints=(), ineq_constraints=(), params=None,
                 instance_list=(), var_id_to_slot=None, projections=(),
                 categorical_ne=()):
        self.instance_sources = list(instance_sources)
        # Raw (node, (kind, val)) routing-instance pairs and the GA-slot
        # assignment for each assignable variable id -- unlike
        # instance_sources/instance_node above (already slot-translated,
        # for kernel.py's consumption), these keep the graph's own node/
        # var_id numbering, so a caller with only `problem` in hand
        # (EvolutionaryWaypointSolver._compute_anchor, mpc.py) can recover
        # which nodes reference a given assignable variable without
        # build_graph_ordering_problem needing to stay alive as a separate
        # object past this call (see that function's docstring).
        self.instance_list = list(instance_list)
        self.var_id_to_slot = {} if var_id_to_slot is None else dict(var_id_to_slot)
        # Categorical inequality constraints between assignment SLOTS: each
        # `(slot_a, slot_b)` says those two assignable variables must resolve
        # to different agents. Planners emit these ("robot0 != robot1"); the
        # discrete solvers use them to prune the assignment enumeration (and,
        # for dp_master_jax, to allow a two-arm handoff node whose per-arm
        # branch owners are only guaranteed distinct BY such a constraint).
        # Slot indices, already translated from the graph's var ids by
        # build_graph_ordering_problem.
        self.categorical_ne = [(int(a), int(b)) for (a, b) in categorical_ne]
        self.n_variables = n_variables
        self.n_agents = x0.shape[0]
        self.dim = x0.shape[1]
        self.x0 = x0
        # Structural default for GraphOrderingRelaxed's own params-less
        # call sites (build_initial_carry_fn's cold start, run_lamarckian_al/
        # warmup_lamarckian_al's params=None default) -- mirrors problem.x0's
        # own role exactly; a live caller (EvolutionaryWaypointSolver.solve/
        # warmup, mpc.py) always supplies the graph's CURRENT
        # view_param_values() instead, the same way it always supplies a
        # live x0 rather than relying on this default.
        self.params = np.zeros(0) if params is None else np.asarray(params)
        self.n_params = self.params.shape[0]
        self.objective = objective
        self.n_cond_vars = n_cond_vars

        self.instance_node = np.asarray(instance_node, dtype=int)
        self.n_nodes = n_nodes
        self.state_dim = state_dim

        # Stashed as plain problem-instance data so callers with only
        # `problem` in hand (e.g. the seed heuristic in solver.py) can
        # recover graph structure. Only edges with gate_fn=None (always
        # active) count as "hard" for that heuristic's precedence DAG.
        self.ordering_edges = list(ordering_edges)
        self.hard_edges = [(u, v) for u, v, gate in self.ordering_edges if gate is None]

        # edge_cost_fn may be one shared callable (priced against every
        # agent's route) or a per-agent list/tuple, one entry per agent --
        # see make_graph_kernel's docstring. Validate the per-agent length
        # here so the error names the mismatch rather than surfacing deep in
        # a traced kernel.
        if isinstance(edge_cost_fn, (list, tuple)) and len(edge_cost_fn) != self.n_agents:
            raise ValueError(
                f"per-agent edge_cost_fn has {len(edge_cost_fn)} entries but the "
                f"graph has {self.n_agents} agents")
        # Kept alongside the built kernel so a consumer with only `problem` in
        # hand (SmallContinuousVRPSolver's branch-selection DP) can price its
        # own transitions with the SAME field kernel.py's routing objective
        # uses, instead of a plain Euclidean surrogate. `None` (Euclidean),
        # one shared callable, or a per-agent list -- see make_graph_kernel.
        self.edge_cost_fn = edge_cost_fn
        kernel_kwargs = {} if edge_cost_fn is None else {"edge_cost_fn": edge_cost_fn}
        self._decode_and_cost, self._batched, self._decode_node_rank = make_graph_kernel(
            self.instance_sources, self.n_variables, self.ordering_edges,
            self.instance_node, self.dim, self.n_nodes,
            objective, **kernel_kwargs)

        self._eq_constraints = list(eq_constraints)
        self._ineq_constraints = list(ineq_constraints)
        sizes = lambda fns: sum(_infer_constraint_size(fn, self.n_variables, self.n_agents,
                                                         self.n_nodes, self.state_dim, self.n_cond_vars,
                                                         self.n_params)
                                 for fn in fns)
        self.n_eq_extra = sizes(self._eq_constraints)
        self.n_ineq_extra = sizes(self._ineq_constraints)
        self.n_eq_constr = self.n_eq_extra
        self.n_ieq_constr = self.n_ineq_extra

        # Analytic-elimination substitutions (problem.ProjectionEntry, from
        # spec.py's _resolve_projections) -- each one claims a slice of two
        # new flat blocks, `proj_branch` (grouped with assign/cond_binary:
        # real-relaxed, GA-searched-only, decoded via argmax, never
        # gradient-refined) and `psi` (grouped with wp: gradient-refined by
        # local_refine, since it's a genuine continuous free parameter --
        # see apply_projections' docstring). Both are 0-width when there
        # are no projections, so an unprojected graph's layout/n_var is
        # completely unchanged from before this feature existed.
        self.projections = list(projections)
        self.n_branch = sum(p.discrete_params for p in self.projections)
        self.n_psi = sum(p.continuous_params for p in self.projections)

        # Which (node, column) wp entries are fully determined by some
        # STATIC projection (problem.ProjectionEntry.write_node/pinned_cols,
        # owner_var_slot is None) -- a pinned column's raw value is always
        # thrown away by apply_projections before anything reads it (see
        # its docstring), regardless of whether that projection also
        # carries a psi/branch, so it's never a real degree of freedom for
        # local_refine's L-BFGS to search. spec.py's own registration
        # already guarantees at most one projection claims any given
        # column (see _resolve_projections' docstring), so a plain union
        # here is exact -- no column is ever double-counted. wp_free_idx
        # indexes into a flattened (n_nodes*state_dim,) row; empty
        # `projections` (the common case) makes it exactly
        # arange(n_nodes*state_dim), i.e. every column free, identical to
        # this feature not existing.
        #
        # A DYNAMIC (var_agent_q-pinned, owner_var_slot is not None) entry
        # is deliberately excluded here -- see ProjectionEntry's own
        # docstring: which agent's columns it actually writes varies per
        # population member/generation (whichever that individual's own
        # assign currently favors), so no single column at that node is
        # EVER unconditionally dead the way a static pin's columns are --
        # every candidate agent's slot there stays a genuine, searched
        # decision variable for the individuals that don't currently
        # select it. JAX/XLA need one fixed shape per compiled function, so
        # there is no way to shrink this per-individual.
        # A GATED entry (gate_fn is not None) is likewise excluded: its pin
        # only fires for the individuals/generations whose decoded schedule
        # satisfies the gate; for the rest the column is still governed by
        # whatever residual the gated projection shadows, so it stays a real
        # searched variable.
        wp_pinned_mask = np.zeros((n_nodes, state_dim), dtype=bool)
        for p in self.projections:
            if p.owner_var_slot is not None or p.gate_fn is not None:
                continue
            wp_pinned_mask[p.write_node, p.pinned_cols] = True
        self.wp_pinned_mask = wp_pinned_mask
        self.wp_free_idx = np.flatnonzero(~wp_pinned_mask.reshape(-1))
        self.n_wp_free = int(self.wp_free_idx.shape[0])

        self.n_assign_vars = self.n_variables * self.n_agents
        self.cond_offset = self.n_assign_vars
        self.branch_offset = self.cond_offset + self.n_cond_vars
        self.t_offset = self.branch_offset + self.n_branch
        self.wp_offset = self.t_offset + self.n_nodes
        self.psi_offset = self.wp_offset + self.n_nodes * self.state_dim
        n_var = self.psi_offset + self.n_psi
        self.n_var = n_var

        xl = np.zeros(n_var)
        xu = np.ones(n_var)
        wp_lo, wp_hi = wp_bounds
        xl[self.wp_offset:self.psi_offset] = wp_lo
        xu[self.wp_offset:self.psi_offset] = wp_hi
        xl[self.t_offset:self.wp_offset] = 0.0
        xu[self.t_offset:self.wp_offset] = float(self.n_nodes - 1)
        # proj_branch (branch_offset:t_offset) keeps the [0, 1] default,
        # same convention assign/cond_binary already use -- a real-relaxed
        # per-branch score, decoded via argmax, not a bounded physical
        # quantity.
        for p in self.projections:
            if p.continuous_params == 0:
                continue
            lo, hi = p.psi_bounds
            start = self.psi_offset + p.psi_slice.start
            stop = self.psi_offset + p.psi_slice.stop
            xl[start:stop] = lo
            xu[start:stop] = hi

        # Statically-fixed instances no longer occupy any position in x at
        # all (no synthetic vagent/one-hot row to pin) -- their real agent
        # id flows directly into kernel.py's owner_instance, bypassing
        # `assign` entirely.
        self.xl, self.xu = xl, xu

    def _extract_single(self, x):
        assign = x[:self.n_assign_vars].reshape(self.n_variables, self.n_agents)
        cond_binary = x[self.cond_offset:self.branch_offset]
        proj_branch = x[self.branch_offset:self.t_offset]
        t = x[self.t_offset:self.wp_offset]
        wp = x[self.wp_offset:self.psi_offset].reshape(self.n_nodes, self.state_dim)
        psi = x[self.psi_offset:]
        return assign, cond_binary, proj_branch, t, wp, psi

    def _extract_batch(self, X):
        pop = len(X)
        assign = X[:, :self.n_assign_vars].reshape(pop, self.n_variables, self.n_agents)
        cond_binary = X[:, self.cond_offset:self.branch_offset]
        proj_branch = X[:, self.branch_offset:self.t_offset]
        t = X[:, self.t_offset:self.wp_offset]
        wp = X[:, self.wp_offset:self.psi_offset].reshape(pop, self.n_nodes, self.state_dim)
        psi = X[:, self.psi_offset:]
        return assign, cond_binary, proj_branch, t, wp, psi

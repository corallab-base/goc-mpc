"""build_graph_ordering_problem: derives a GraphOrderingRelaxed problem
instance (see problem.py) from a real C++ GraphOfConstraints, so one graph
definition can drive either the MILP (MILPWaypointMPC) or this JAX-native
evolutionary solver.

Built ONCE per GraphOfConstraints and reused for the solver's whole lifetime,
spanning the WHOLE graph unconditionally (see build_graph_ordering_
problem's own docstring): remaining_vertices is a runtime AnchorState (mpc.py/problem.py)
fed into the already-compiled GraphOrderingRelaxed at solve time, so a node
dropping out of remaining_vertices never removes its wps row, routing
instance, or any constraint touching it -- it only switches that node from a
free decision variable to a known constant (its last-committed value),
mirroring how MILP substitutes a passed node's frozen waypoint into a
boundary edge constraint rather than dropping it. Node/edge constraints
added via the graph's unified symbolic API (add_constraint /
add_assignable_constraint / add_edge_constraint) are auto-derived from
graph.phi_to_formula_map/edge_phi_to_formula_map and compiled via
formula_compiler.compile_relational_formula -- the same Formula that drives
MILPWaypointMPC also drives this solver, no manual duplication.

Every graph node gets a configuration row, unconditionally (whether or not
any phi references it -- matching MILP's own `W`, one row per node in its
subgraph regardless of constraint presence) -- `[agent_0 | agent_1 | ... |
agent_{n_agents-1} | object_0 | ... | object_{n_objects-1}]`, `state_dim =
num_agents*slot_width + num_objects*object_slot_width` wide -- mirroring
MILP's per-node joint configuration `W` (graph_of_constraints.cpp's
`total_dim`, though MILP's own `W` uses a packed cumulative offset per
agent/object rather than this module's padded slots -- see _slot_width's
docstring for why). Each agent's/object's own slot is `slot_width`/
`object_slot_width` wide, derived from each agent's/object's own declared
config (_agent_widths/_object_widths) -- agents/objects need not share a
width with each other or with any other agent/object.
A node id is therefore directly its own row index into wp/t/node_active --
no separate local/compacted node numbering anywhere in this solver. So a
node/edge constraint can reference ANY agent's or
object's placeholder (agent_q(k), object_q(k), var_agent_q(var), and their
u_-/v_-prefixed edge counterparts) via a plain column offset into that row
(see _make_row_resolver) -- no per-formula "which sparse row" bookkeeping,
plus agent_link_pos(agent_id, link_name)/agent_link_rot(agent_id, link_name)
(see _make_link_pos_map/_make_link_rot_map), forward-kinematics placeholders
resolved by calling a registered graph.set_robot_fk function directly rather
than by column offset, and no restriction on how many distinct agents/objects
one formula mixes
(e.g. a rigid-attachment/grasp formula tying agent_q(k) to object_q(j) at a
node, or a *relational* edge formula tying v_object_q - u_object_q to
v_agent_q - u_agent_q). An edge constraint built from the plain, un-prefixed
placeholders instead means "along the edge" -- an invariant applied
independently at both endpoints (see edge_phi_to_along_edge_map and
_resolve_symbolic_constraints below), not a relation between them.

Routing/ordering is a separate concern layered on top, and operates at TWO
different granularities. A routing "instance" -- `("fixed", agent_id)` (a
literal agent_q(k) constraint) or `("var", var_id)` (an assignable
var_agent_q(var) constraint) at a given node -- exists purely to record
which agent(s) a node's route-cost/ordering must count toward; its own wp is
gathered from its node's row (kernel.py), and it owns no arrival-time
state of its own. Arrival time (`t`) and ordering edges are genuinely
per-NODE, matching MILP's own per-routing-node `t` variable
(milp_waypoint_mpc.cpp): a node visited by more than one agent (e.g. a
shared start or rendezvous node) is one shared schedulable event, not one
per agent -- kernel.py reduces each agent's own computed arrival time back
down to a single per-node value (a max over whichever instances share that
node) before any precedence check runs. Object references never create
routing instances or need any ownership -- an object's column just sits in
whichever node rows reference it, constrained only by whatever symbolic
formulas mention it.

A constraint referencing a placeholder genuinely outside this scope (e.g.
u_agent_q/v_agent_q inside a *node* constraint, which has no u/v side) raises
at spec-construction time. For anything outside this solver's symbolic scope
entirely (e.g. a non-symbolic/black-box cost), add_python_constraint remains
available as an escape hatch, scoped to a single node's whole row (no
routing instance required there -- see add_python_constraint's docstring)."""

import jax
import jax.numpy as jnp
import numpy as np

from .default_fk import resolve_link_fk
from .formula_compiler import as_variable, compile_condition, compile_relational_formula
from .kernel import build_decode_node_rank
from .problem import GraphOrderingRelaxed


def target_eq_constraint(target):
    """Convenience constraint: pins a node's WHOLE row to a fixed target
    exactly (H = wp_row - target, feasible at H=0). `target` must match the
    node's full row width (state_dim) -- every agent's slot plus every
    object's column -- not just one agent's own slice (see
    add_python_constraint)."""
    target = np.asarray(target, dtype=float)

    def fn(wp_row, assign):
        return wp_row - target
    fn.node_target = target
    return fn


def _placeholder_id_map(vec):
    """vec: array of Expression (e.g. graph.agent_q(k) or
    graph.var_agent_q(var)). Returns {Variable.get_id(): component_index}."""
    out = {}
    for j, expr in enumerate(vec):
        v = as_variable(expr)
        if v is not None:
            out[v.get_id()] = j
    return out


def _agent_widths(graph):
    """Per-agent ambient config width, one entry per graph.num_agents,
    derived from graph._robot_specs (the actual declared Block sizes) --
    agents need not share a width with each other (GraphOfConstraints no
    longer requires it, and there is no single graph-wide `dim` any more).
    See _slot_width's docstring for the row layout this feeds into."""
    return [sum(b.size for b in spec) for spec in graph._robot_specs]


def _slot_width(agent_widths):
    """The per-agent PADDED-SLOT width this module's wp row layout uses:
    every agent gets an equal-width column slot (max over agent_widths),
    with its real config occupying only the slot's first agent_widths[k]
    columns -- exactly the zero-fill convention mpc.py's/problem.py's own
    x0 padding already uses elsewhere, just applied to the row layout
    itself instead of to x0. This keeps every per-agent column offset a
    plain `k * slot_width` (or, for a GA-searched dynamic agent, `agent *
    slot_width`) -- a single static stride valid for every agent -- rather
    than a packed cumulative offset that would depend on which agents
    precede k, which a GA-searched dynamic selection can't index without
    itself becoming per-candidate-width-aware. A static, agent-known
    reference (agent_q(k), agent_link_pos(k, ...), a hold's static
    robot_ag) instead slices its own agent_widths[k] columns out of that
    slot, never the padded remainder -- see _make_row_resolver/
    _resolve_holds."""
    return max(agent_widths) if agent_widths else 0


def _object_widths(graph):
    """Per-object ambient config width, mirroring _agent_widths above."""
    return [sum(b.size for b in spec) for spec in graph._object_specs]


def _object_slot_width(object_widths):
    """Per-object padded-slot width, mirroring _slot_width above -- objects
    don't have agents' GA-searched-dynamic-candidate complication (an
    object is always referenced by a fixed index, never resolved via
    jax.lax.dynamic_slice the way var_agent_q is), so a packed cumulative
    offset would work just as well here; this module uses the same padded-
    slot shape for both anyway, for one consistent row-layout convention
    rather than two."""
    return max(object_widths) if object_widths else 0


def _unsupported_placeholder(var):
    raise ValueError(
        f"Symbolic constraint references placeholder variable {var!r} that "
        "isn't representable here -- expected one of agent_q(k)/object_q(k)/"
        "var_agent_q(var)/param(id)/agent_link_pos(agent_id, link_name)/"
        "agent_link_rot(agent_id, link_name) (node constraints, or an edge "
        "constraint's \"along the edge\" form) or "
        "u_agent_q(k)/u_object_q(k) (an edge constraint's u side) or "
        "v_agent_q(k)/v_object_q(k) (an edge constraint's v side; "
        "u_/v_-prefixed placeholders are not valid inside a node constraint, "
        "which has no u/v side). If this is an agent_link_pos(agent_id, "
        "link_name)/agent_link_rot(agent_id, link_name) placeholder, check "
        "that a forward-kinematics function is registered for that "
        "(agent_id, link_name) via graph.set_robot_fk -- this solver "
        "resolves it by calling that function directly, in Python, and has "
        "no fallback for an unregistered link. Otherwise, use "
        "add_python_constraint for anything genuinely outside this scope.")


def _make_link_pos_map(graph):
    """{Variable.get_id(): (agent_id, link_name, fk_fn, component_index)} for
    every agent_link_pos(agent_id, link_name) placeholder backed by a
    registered forward-kinematics function (graph.robot_fk_registry, set via
    graph.set_robot_fk) -- the same (agent_id, link_name) -> fk_fn registry
    link_pose consults in C++, read here directly so this solver can call
    fk_fn itself, in Python, under jax.jit/vmap tracing (see set_robot_fk's
    doc comment: fk_fn is expected to be written in jax.numpy so it works
    both ways). A placeholder with no registered fk_fn simply doesn't appear
    here -- referencing it later correctly falls through to
    _unsupported_placeholder rather than silently doing nothing."""
    out = {}
    for (agent_id, link_name), fk_fn in graph.robot_fk_registry.items():
        for j, expr in enumerate(graph.agent_link_pos(agent_id, link_name)):
            v = as_variable(expr)
            if v is not None:
                out[v.get_id()] = (agent_id, link_name, fk_fn, j)
    return out


def _make_link_rot_map(graph):
    """Rotation counterpart of _make_link_pos_map: {Variable.get_id():
    (agent_id, link_name, fk_fn, flat_index)} for every
    agent_link_rot(agent_id, link_name) placeholder. flat_index indexes into
    fk_fn(q)[1].flatten() -- row-major, matching
    GraphOfConstraints::agent_link_rot's C++-side flattening convention (and
    jax.numpy's default flatten() order), so component j here always lines
    up with rotation entry (j // workspace_dim, j % workspace_dim)."""
    out = {}
    for (agent_id, link_name), fk_fn in graph.robot_fk_registry.items():
        for j, expr in enumerate(graph.agent_link_rot(agent_id, link_name)):
            v = as_variable(expr)
            if v is not None:
                out[v.get_id()] = (agent_id, link_name, fk_fn, j)
    return out


def _make_row_resolver(graph, var_id_to_slot, agent_widths, slot_width,
                        object_widths, object_slot_width):
    """Builds resolve(var, n_row_slots) -> Callable[*rows, owner_variable,
    params] for the unified symbolic constraint API, over the per-node row
    layout described in this module's docstring.

    agent_q(k)/object_q(k) (side 0 -- a node constraint's own row, or an
    "along the edge" edge constraint's row, applied once per endpoint by the
    caller -- see _resolve_symbolic_constraints), u_agent_q(k)/u_object_q(k)
    (also side 0 -- a *relational* edge constraint's u row; a distinct
    placeholder from plain agent_q/object_q that happens to resolve to the
    same column), and v_agent_q(k)/v_object_q(k) (side 1, a relational edge
    constraint's v row) are all STATIC column offsets, known at spec-build
    time. var_agent_q(var) is the only DYNAMIC case: which agent's column it
    reads depends on the GA-searched assignment (`owner_variable`, threaded
    as the second-to-last argument after every row by
    _batch_symbolic_constraint_fn), resolved via a differentiable
    jax.lax.dynamic_slice so gradient-based local refinement of wp still
    works through it. It only ever binds side 0 (mirrors add_edge_constraint's
    C++ substitution, which never binds var_agent_q on the v side of a
    relational edge constraint, and an "along the edge" constraint has no v
    side to bind at all -- it's just a single node-scoped placeholder set,
    applied once per endpoint).

    agent_link_pos(agent_id, link_name)/agent_link_rot(agent_id, link_name)
    are a third, distinct case: a STATIC side-0 column slice (agent_id is
    fixed at authoring time, same as plain agent_q -- see
    graph.add_edge_constraint's has_plain classification), but resolved via
    the registered fk_fn instead of a bare column read -- calls fk_fn
    directly, in Python (see _make_link_pos_map/_make_link_rot_map), on that
    agent's dim-wide slice, tracing cleanly under jax.jit/vmap as long as
    fk_fn is written in jax.numpy (see set_robot_fk's doc comment).

    param(id) is a fourth case: a runtime-editable scalar constant
    (GraphOfConstraints.add_param/set_param), resolved by reading the
    trailing `params` argument (always LAST, after owner_variable) -- a
    genuine jax runtime array threaded the same way x0 already is, so
    set_param never forces a retrace (see this module's param_map).

    n_row_slots: 1 for a node constraint or an "along the edge" edge
    constraint (only side 0 valid -- u_/v_agent_q / u_/v_object_q correctly
    raise via _unsupported_placeholder), 2 for a relational edge constraint
    (side 0 = u, side 1 = v).

    agent_widths/slot_width (and object_widths/object_slot_width, the same
    idea applied to objects): this module's padded-slot row layout (see
    _slot_width's docstring) -- agent k's/object k's slot starts at `k *
    slot_width`/`agents_width + k * object_slot_width` regardless of its
    own real width, so every static offset below is one of those plus `j`,
    never a packed cumulative offset.
    """
    num_agents = graph.num_agents
    num_objects = graph.num_objects
    agents_width = num_agents * slot_width

    static_map = {}  # var_id -> (side, col)
    for k in range(num_agents):
        for j, expr in enumerate(graph.agent_q(k)):
            v = as_variable(expr)
            if v is not None:
                static_map[v.get_id()] = (0, k * slot_width + j)
        for j, expr in enumerate(graph.u_agent_q(k)):
            v = as_variable(expr)
            if v is not None:
                static_map[v.get_id()] = (0, k * slot_width + j)
    for k in range(num_objects):
        for j, expr in enumerate(graph.object_q(k)):
            v = as_variable(expr)
            if v is not None:
                static_map[v.get_id()] = (0, agents_width + k * object_slot_width + j)
        for j, expr in enumerate(graph.u_object_q(k)):
            v = as_variable(expr)
            if v is not None:
                static_map[v.get_id()] = (0, agents_width + k * object_slot_width + j)
    for k in range(num_agents):
        for j, expr in enumerate(graph.v_agent_q(k)):
            v = as_variable(expr)
            if v is not None:
                static_map[v.get_id()] = (1, k * slot_width + j)
    for k in range(num_objects):
        for j, expr in enumerate(graph.v_object_q(k)):
            v = as_variable(expr)
            if v is not None:
                static_map[v.get_id()] = (1, agents_width + k * object_slot_width + j)

    var_map = {}  # var_id -> (slot, component j)
    for var_id, slot in var_id_to_slot.items():
        for j, expr in enumerate(graph.var_agent_q(var_id)):
            v = as_variable(expr)
            if v is not None:
                var_map[v.get_id()] = (slot, j)

    link_pos_map = _make_link_pos_map(graph)  # var_id -> (agent_id, link_name, fk_fn, j)
    link_rot_map = _make_link_rot_map(graph)  # var_id -> (agent_id, link_name, fk_fn, flat_j)

    # Shared FK-result cache. Every agent_link_pos(a, link) / agent_link_rot(
    # a, link) COMPONENT placeholder resolves to its own closure, and each
    # would otherwise call the full fk_fn(q) itself -- so a node with a
    # position+orientation EE target evaluates fk_fn ~12x per residual pass
    # (and XLA does NOT reliably CSE a non-trivial articulated FK across
    # those separately-built closures: a 6-DOF UR5e FK measured ~linear in
    # the placeholder count, ~4x slower for pos+rot vs pos-only, plus a much
    # larger HLO -> long compile). Compute fk_fn(q) ONCE per (agent, link)
    # per row and share. Keyed by id() of the row array, with the array
    # itself pinned in the entry so its id can't be recycled while the
    # entry is live; entries accumulate only across (rare) retraces.
    _fk_cache: dict = {}

    def _link_fk(row, agent_id, link_name, col0, w, fk_fn):
        entry = _fk_cache.get(id(row))
        if entry is None or entry[0] is not row:
            entry = (row, {})
            _fk_cache[id(row)] = entry
        got = entry[1].get((agent_id, link_name))
        if got is None:
            got = fk_fn(row[col0:col0 + w])
            entry[1][(agent_id, link_name)] = got
        return got

    # param(id) -- runtime-editable scalar placeholders (GraphOfConstraints.
    # add_param/param/set_param). Every compiled residual fn is called as
    # fn(*rows, owner_variable, params) (see _batch_symbolic_constraint_fn et
    # al.) -- params is therefore always the LAST positional argument, a
    # genuine jax runtime array (not baked into the compiled closure), so
    # set_param never forces a retrace.
    param_map = {}  # var_id -> component index into params
    for i in range(graph.num_params()):
        v = as_variable(graph.param(i))
        if v is not None:
            param_map[v.get_id()] = i

    def resolve(var, n_row_slots):
        vid = var.get_id()
        if vid in static_map:
            side, col = static_map[vid]
            if side >= n_row_slots:
                _unsupported_placeholder(var)
            return lambda *args, side=side, col=col: args[side][col]
        if vid in var_map:
            slot, j = var_map[vid]

            # `agent` is GA-searched (traced, one per population member), so
            # this offset must be a single static stride valid for whichever
            # agent gets selected -- slot_width (see _slot_width's
            # docstring), not that agent's own (potentially narrower) real
            # width. `j` only ever ranges over var_agent_q(var)'s own real
            # (per-variable) width -- resolved in C++ from that variable's
            # candidate agents, which add_variable_constraint requires to
            # share one width -- so it never exceeds any candidate's real
            # width regardless of how wide slot_width itself is.
            def fn(*args, slot=slot, j=j, slot_width=slot_width):
                owner_variable = args[-2]
                agent = owner_variable[slot]
                return jax.lax.dynamic_slice_in_dim(args[0], agent * slot_width + j, 1)[0]
            return fn
        if vid in param_map:
            idx = param_map[vid]
            return lambda *args, idx=idx: args[-1][idx]
        if vid in link_pos_map:
            agent_id, link_name, fk_fn, j = link_pos_map[vid]
            col0 = agent_id * slot_width
            w = agent_widths[agent_id]

            def fn(*args, agent_id=agent_id, link_name=link_name, col0=col0, w=w, fk_fn=fk_fn, j=j):
                pos, _rot = _link_fk(args[0], agent_id, link_name, col0, w, fk_fn)
                return pos[j]
            return fn
        if vid in link_rot_map:
            agent_id, link_name, fk_fn, j = link_rot_map[vid]
            col0 = agent_id * slot_width
            w = agent_widths[agent_id]

            def fn(*args, agent_id=agent_id, link_name=link_name, col0=col0, w=w, fk_fn=fk_fn, j=j):
                _pos, rot = _link_fk(args[0], agent_id, link_name, col0, w, fk_fn)
                return jnp.reshape(rot, (-1,))[j]
            return fn
        _unsupported_placeholder(var)
    return resolve


def _decode_rank_batched(decode_node_rank, assign, cond_binary, t, node_active):
    """vmaps a spec-level decode_node_rank(owner_variable, cond_binary, t,
    node_active) -> node_rank (kernel.build_decode_node_rank) over the
    population axis of assign/cond_binary/t -- node_active is shared across
    the whole population (in_axes=None), exactly mirroring how kernel.py's
    own `batched` vmaps decode_and_cost (in_axes=(0, 0, 0, 0, None, None)).
    Returns (pop, n_nodes) int32 -- an EXACT, per-individual topologically
    valid visiting-order rank, unlike raw `t` (see build_decode_node_rank's
    docstring for why raw t alone is unsafe to gate on)."""
    owner_variable = jnp.argmax(assign, axis=-1)
    return jax.vmap(decode_node_rank, in_axes=(0, 0, 0, None))(owner_variable, cond_binary, t, node_active)


def _batch_symbolic_constraint_fn(fn, node_locals, mode="frozen"):
    """Wraps a compiled symbolic-constraint residual fn(*rows, owner_variable,
    params) -> (k,) into the batched (assign, cond_binary, t, wp_frozen,
    wp_live, node_active, x0, params) -> (pop, k) contract solver.py expects,
    via jax.vmap over the population. `wp_frozen`/`wp_live` are both the
    per-node tensor (pop, n_nodes, state_dim) -- identical wherever a row's
    node is still active, and differing only for an already-passed node's row
    (problem.apply_anchor); `mode` (a plain Python str, closed over at
    spec-build time via build_graph_ordering_problem's live_phi_ids, never a traced
    value) picks which one this particular constraint reads for ALL of its
    rows. Each entry of `node_locals` is a node index (this constraint's
    node, or (u, v) for an edge constraint) -- NOT a routing-instance id (see
    module docstring). `cond_binary`/`node_active`/`x0` are accepted only to
    match the uniform 8-arg contract every eq/ineq constraint closure shares
    (see _batch_along_edge_interior_fn/_batch_relational_interior_fn/
    _batch_stationary_edge_fn for closures that actually need node_active,
    via decode_node_rank, and _batch_depot_stationary_fn for the one that
    actually needs x0). `params` (GraphOfConstraints.add_param/set_param) IS
    read here, by any compiled residual referencing a param(id) placeholder
    -- population-invariant (in_axes=None, unlike every row/owner_variable
    arg above it), so set_param between solve() calls never forces a
    retrace, only changes what this same jitted vmapped(...) reads."""
    vmapped = jax.vmap(fn, in_axes=(0,) * len(node_locals) + (0, None))

    def batched(assign, cond_binary, t, wp_frozen, wp_live, node_active, x0, params,
                node_locals=node_locals, vmapped=vmapped, mode=mode):
        wp = wp_live if mode == "live" else wp_frozen
        rows = [wp[:, nl, :] for nl in node_locals]
        owner_variable = jnp.argmax(assign, axis=-1)
        return vmapped(*rows, owner_variable, params)
    return batched


def _batch_along_edge_interior_fn(fn, kind, u, v, decode_node_rank, mode="frozen"):
    """JAX analogue of MILP's betweenness-gated interior_builder
    re-application (see milp_waypoint_mpc.cpp's Constraint 13b): re-applies
    an "along the edge" formula's residual at every OTHER node whose
    DECODED visiting-order rank (kernel.build_decode_node_rank -- an exact,
    per-individual topologically valid permutation, NOT raw `t`; see that
    function's docstring for why raw t alone can't be trusted for this)
    currently falls between u's and v's.

    Unlike the exact per-node registrations _resolve_symbolic_constraints
    makes for u/v themselves (each its own persistent AL multiplier --
    unaffected by this function), every OTHER between-node's contribution is
    aggregated into a SINGLE non-negative "total interior violation" per
    residual component -- summing masked per-node relu(residual) (ineq) or
    residual**2 (eq) -- and always registered as one ineq-style constraint
    (feasible at <=0; since it's a sum of non-negative terms that's
    equivalent to "exactly zero everywhere between"), regardless of the
    formula's own eq/ineq kind. This trades exact per-node multipliers
    (which would need a per-individual, dynamically-sized multiplier count
    -- mu/lam are fixed-size for the solver's whole lifetime, see
    problem.GraphOrderingRelaxed) for a single shared multiplier whose
    "identity" drifts as the between-set changes generation to generation.

    `mode` picks wp_frozen vs wp_live (see _batch_symbolic_constraint_fn's
    docstring) -- the same choice made for this formula's exact u/v
    registrations (both come from the same phi's live_phi_ids membership),
    applied uniformly to every other between-node's row too."""
    vmapped_pop = jax.vmap(fn, in_axes=(0, 0, None))

    def batched(assign, cond_binary, t, wp_frozen, wp_live, node_active, x0, params, u=u, v=v,
                kind=kind, vmapped_pop=vmapped_pop, mode=mode, decode_node_rank=decode_node_rank):
        wp = wp_live if mode == "live" else wp_frozen
        owner_variable = jnp.argmax(assign, axis=-1)

        def per_node(node_rows):  # node_rows: (pop, state_dim) -- wp[:, nd, :]
            return vmapped_pop(node_rows, owner_variable, params)  # (pop, k)

        # vmap over wp's node axis (1) -- (pop, n_nodes, state_dim)
        # -> (n_nodes, pop, k).
        all_residuals = jax.vmap(per_node, in_axes=1, out_axes=0)(wp)
        viol = jnp.maximum(0.0, all_residuals) if kind == "ineq" else all_residuals ** 2

        rank = _decode_rank_batched(decode_node_rank, assign, cond_binary, t, node_active)
        lo = jnp.minimum(rank[:, u], rank[:, v])
        hi = jnp.maximum(rank[:, u], rank[:, v])
        between = (rank >= lo[:, None]) & (rank <= hi[:, None])
        between = between.at[:, u].set(False)
        between = between.at[:, v].set(False)

        masked = viol * jnp.transpose(between)[:, :, None]  # (n_nodes, pop, k)
        return jnp.sum(masked, axis=0)  # (pop, k)
    return batched


def _batch_relational_interior_fn(fn, kind, u, v, decode_node_rank, mode="frozen"):
    """The *relational* (two-row) analogue of _batch_along_edge_interior_fn
    above -- used for hold-derived rigid-carry constraints (see
    spec.py's _resolve_holds), the JAX side of MILP's Constraint 14a
    interior-reinforcement loop (milp_waypoint_mpc.cpp: AddHoldRigidity*Gated,
    gated by GetOrAddBetweennessIndicator). `fn(row_ref, row_other,
    owner_variable) -> (k,)` is the same compiled residual already registered
    at the hold's own (u, v) endpoints (side 0 = u's placeholders, side 1 =
    v's) -- here reapplied with u's OWN row held fixed as the side-0
    reference and every OTHER node w's row substituted in as side 1, gated by
    whether w's DECODED visiting-order rank (see _batch_along_edge_interior_fn's
    docstring) currently falls between u's and v's. Needed because a node
    scheduled between a hold's endpoints by some OTHER agent's route would
    otherwise leave the held object's value there completely unconstrained --
    the object column is a real, independently optimized decision variable
    at every node (see this module's docstring), not automatically pinned by
    the hold's own (u, v) registration.

    Aggregation follows _batch_along_edge_interior_fn's convention exactly:
    every between-node's contribution is summed into a single non-negative
    "total interior violation" (masked relu(residual) for an ineq formula,
    residual**2 for an eq one) and registered as one ineq-style constraint,
    trading exact per-node multipliers for a single shared one."""
    vmapped_pop = jax.vmap(fn, in_axes=(0, 0, 0, None))

    def batched(assign, cond_binary, t, wp_frozen, wp_live, node_active, x0, params, u=u, v=v,
                kind=kind, vmapped_pop=vmapped_pop, mode=mode, decode_node_rank=decode_node_rank):
        wp = wp_live if mode == "live" else wp_frozen
        owner_variable = jnp.argmax(assign, axis=-1)
        row_u = wp[:, u, :]

        def per_node(node_rows):  # node_rows: (pop, state_dim) -- wp[:, nd, :]
            return vmapped_pop(row_u, node_rows, owner_variable, params)  # (pop, k)

        # vmap over wp's node axis (1) -- (pop, n_nodes, state_dim)
        # -> (n_nodes, pop, k).
        all_residuals = jax.vmap(per_node, in_axes=1, out_axes=0)(wp)
        viol = jnp.maximum(0.0, all_residuals) if kind == "ineq" else all_residuals ** 2

        rank = _decode_rank_batched(decode_node_rank, assign, cond_binary, t, node_active)
        lo = jnp.minimum(rank[:, u], rank[:, v])
        hi = jnp.maximum(rank[:, u], rank[:, v])
        between = (rank >= lo[:, None]) & (rank <= hi[:, None])
        between = between.at[:, u].set(False)
        between = between.at[:, v].set(False)

        masked = viol * jnp.transpose(between)[:, :, None]  # (n_nodes, pop, k)
        return jnp.sum(masked, axis=0)  # (pop, k)
    return batched


def _batch_stationary_edge_fn(u, v, seg_slice, hold_node_pairs, decode_node_rank, mode="live"):
    """Gated stationary-object residual for one (structural edge, object)
    pair -- the JAX analogue of MILP's Constraint 14b (milp_waypoint_mpc.cpp's
    `stationary_edge` lambda, gated by GetOrAddIntervalOverlapIndicator).
    Ties the object's segment at u and v together UNLESS the edge's own
    DECODED-rank interval [rank_u, rank_v] overlaps some declared hold's own
    interval [rank_hu, rank_hv] on this object (`hold_node_pairs`, from
    spec.py's _resolve_stationary_objects) -- via decode_node_rank
    (see _batch_along_edge_interior_fn's docstring for why the EXACT decoded
    rank is used here rather than raw `t`: unlike that function's bonus/
    best-effort reinforcement on top of an already-exact registration, this
    gate directly toggles the PRIMARY equality, so a noisy signal would
    create false gate-offs -- verified directly while building this: a
    t-gated version of this exact function failed to re-pin an object's
    position on the edge immediately following a hold's release).

    Uses a STRICT interval-overlap test (a0 < b1 and b0 < a1) rather than
    the inclusive one MILP's GetOrAddIntervalOverlapIndicator uses: with
    EXACT decoded ranks (a true bijective permutation), two structural edges
    can only share a boundary VALUE by sharing an actual node (e.g. a hold
    ending exactly at v with the very next edge starting at v) -- MILP's
    inclusive reading treats that as "overlapping" too, which would leave
    the release edge immediately following any hold ungated (the one case
    parity most needs to enforce), so it's deliberately not replicated. A
    hold's OWN edge (checked against itself when this stationary edge IS
    that hold's declared edge) still registers as a genuine overlap under
    the strict test, since a real hold spans hu strictly before hv.

    `hold_node_pairs` empty (no hold anywhere ever touches this object)
    means overlap_any is vacuously always False -- collapses to an
    unconditional equality, exactly mirroring MILP's `any_gate` branch.

    Registered mode="live" (see spec.py's _resolve_holds' docstring
    for the same reasoning applied to hold rigidity): once u has passed, an
    untouched object's real current position (x0) is the ground truth going
    forward, not whatever was merely planned there."""
    def batched(assign, cond_binary, t, wp_frozen, wp_live, node_active, x0, params, u=u, v=v, seg_slice=seg_slice,
                hold_node_pairs=hold_node_pairs, decode_node_rank=decode_node_rank, mode=mode):
        wp = wp_live if mode == "live" else wp_frozen
        residual = wp[:, u, seg_slice] - wp[:, v, seg_slice]
        viol = jnp.sum(residual ** 2, axis=-1, keepdims=True)  # (pop, 1)

        rank = _decode_rank_batched(decode_node_rank, assign, cond_binary, t, node_active)
        edge_lo = jnp.minimum(rank[:, u], rank[:, v])
        edge_hi = jnp.maximum(rank[:, u], rank[:, v])
        overlap_any = jnp.zeros(rank.shape[0], dtype=bool)
        for hu, hv in hold_node_pairs:
            hold_lo = jnp.minimum(rank[:, hu], rank[:, hv])
            hold_hi = jnp.maximum(rank[:, hu], rank[:, hv])
            overlap_any = overlap_any | ((edge_lo < hold_hi) & (hold_lo < edge_hi))

        gate = jnp.where(overlap_any, 0.0, 1.0)[:, None]  # (pop, 1)
        return viol * gate  # feasible at <=0: forces exact equality unless gated off
    return batched


def _batch_depot_stationary_fn(v, seg_slice, hold_node_pairs, decode_node_rank, mode="live"):
    """Depot analogue of _batch_stationary_edge_fn: ties a graph SOURCE
    node's (no incoming hard edge) object segment directly to the object's
    REAL depot value -- the runtime `x0` GraphOfConstraintsMPC.step() passes
    each cycle, not another node's solved row. This is the JAX side of
    MILP's Constraint 14b depot loop (milp_waypoint_mpc.cpp: `for (v14b :
    subgraph.structure.sources())`), needed because _resolve_stationary_
    objects' regular edge-to-edge chaining alone has nothing to anchor a
    source node to: an object with no explicit node/hold constraint
    anywhere is otherwise only tied to ITSELF across edges (internally
    consistent at some arbitrary GA-found value), never to where it's
    actually sitting.

    The depot plays the role of "the previous waypoint" for a node with no
    real structural predecessor: same residual/gating shape as
    _batch_stationary_edge_fn, just with x0 substituted for wp[u] and the
    depot's own decoded rank fixed at -1 -- strictly before every real
    node's rank (always >= 0, see kernel.build_decode_node_rank) -- so the
    interval-overlap gate against each hold's own [rank_hu, rank_hv] span
    still correctly turns this off wherever a hold reaches v before the
    depot's (definitionally always-earliest) claim would apply."""
    def batched(assign, cond_binary, t, wp_frozen, wp_live, node_active, x0, params, v=v, seg_slice=seg_slice,
                hold_node_pairs=hold_node_pairs, decode_node_rank=decode_node_rank, mode=mode):
        wp = wp_live if mode == "live" else wp_frozen
        residual = x0[None, seg_slice] - wp[:, v, seg_slice]
        viol = jnp.sum(residual ** 2, axis=-1, keepdims=True)  # (pop, 1)

        rank = _decode_rank_batched(decode_node_rank, assign, cond_binary, t, node_active)
        edge_lo = -jnp.ones_like(rank[:, v])
        edge_hi = rank[:, v]
        overlap_any = jnp.zeros(rank.shape[0], dtype=bool)
        for hu, hv in hold_node_pairs:
            hold_lo = jnp.minimum(rank[:, hu], rank[:, hv])
            hold_hi = jnp.maximum(rank[:, hu], rank[:, hv])
            overlap_any = overlap_any | ((edge_lo < hold_hi) & (hold_lo < edge_hi))

        gate = jnp.where(overlap_any, 0.0, 1.0)[:, None]  # (pop, 1)
        return viol * gate  # feasible at <=0: forces exact equality unless gated off
    return batched


def _batch_python_constraint_fn(fn, node):
    """Wraps a single-node add_python_constraint fn(wp_row, assign) -> (k,)
    (see target_eq_constraint) into the batched (assign, cond_binary, t,
    wp_frozen, wp_live, node_active, x0, params) -> (pop, k) contract every
    eq/ineq constraint closure shares, via jax.vmap over the population.
    `wp_row` is `node`'s WHOLE row (state_dim,) -- every agent's slot plus
    every object's column, unsliced -- and `assign` is the raw one-hot
    assignment tensor (n_variables, num_agents) for that population member,
    so fn does its own gather (e.g. jnp.argmax(assign, axis=-1) for
    owner_variable, then index into wp_row by agent*slot_width) if it needs
    one, rather than this wrapper picking an agent slice on `node`'s behalf
    -- unlike kernel.py's routing-instance gather, this constraint isn't
    scoped to any one instance. Always reads wp_frozen (add_python_
    constraint has no live/frozen choice -- it's a single-node escape hatch,
    not a u_/v_-prefixed relational edge formula, so there's no "which side
    is passed" question for it to answer); wp_live/cond_binary/node_active/
    x0/params are accepted only to match the uniform 8-arg contract every
    eq/ineq constraint closure shares."""
    vmapped = jax.vmap(fn)

    def batched(assign, cond_binary, t, wp_frozen, wp_live, node_active, x0, params, node=node, vmapped=vmapped):
        row = wp_frozen[:, node, :]
        return vmapped(row, assign)
    return batched


def build_graph_ordering_problem(graph, x0, wp_bounds,
                                  objective="avg", edge_cost_fn=None,
                                  python_constraints=()):
    """Derives a GraphOrderingRelaxed problem instance (see problem.py)
    from a real C++ GraphOfConstraints in one pass, so one graph
    definition can drive either the MILP (MILPWaypointMPC) or this
    JAX-native evolutionary solver. Called exactly once per
    GraphOfConstraints, by EvolutionaryWaypointSolver._ensure_built
    (mpc.py) -- the returned GraphOrderingRelaxed's shape is fixed for the
    solver's whole lifetime, so there is no later point at which a change
    here could take effect.

    Spans the WHOLE graph unconditionally, independent of any
    receding-horizon remaining_vertices set (see module docstring above):
    remaining_vertices is a runtime AnchorState (mpc.py/problem.py) fed
    into the already-built GraphOrderingRelaxed at solve time, so a node
    dropping out of remaining_vertices never removes its wp row, routing
    instance, or any constraint touching it -- it only switches that node
    from a free decision variable to a known constant (its last-committed
    value), mirroring how MILP substitutes a passed node's frozen waypoint
    into a boundary edge constraint rather than dropping it. For an edge
    constraint, that constant is one of TWO choices, picked per constraint
    via `live_phi_ids` (edge phi ids only -- see graph_of_constraints.hpp's
    live_edge_phis comment for why a node constraint has no such choice to
    make) -- see _resolve_symbolic_constraints and problem.apply_anchor's
    docstring.

    `python_constraints`: an iterable of (node, fn, kind, name) tuples --
    see EvolutionaryWaypointSolver.add_python_constraint (mpc.py), the
    single-node-whole-row Python escape hatch for anything outside this
    module's symbolic scope entirely (e.g. a non-symbolic/black-box cost);
    fn(wp_row, assign) -> (k,), kind="eq" feeds pymoo/AL's H (feasible at
    0), kind="ineq" feeds G (feasible at <=0) -- see target_eq_constraint/
    _batch_python_constraint_fn. Every node/edge constraint added via the
    graph's unified symbolic API (add_constraint/add_assignable_constraint/
    add_edge_constraint) is instead auto-derived below from graph.phi_to_
    formula_map/edge_phi_to_formula_map and compiled via formula_compiler.
    compile_relational_formula -- the same Formula that drives
    MILPWaypointMPC also drives this solver, no manual duplication."""
    # Edge phi ids (graph.edge_phi_to_formula_map keys, populated by
    # GraphOfConstraints.add_edge_constraint(..., live=True)) whose u-side
    # reads should track the current call's real x0 (wp_live) instead of
    # the last-committed planned value (wp_frozen, the default for
    # anything not listed here) -- see _resolve_symbolic_constraints and
    # problem.apply_anchor's docstring. Read straight off `graph` (not a
    # separate argument): the live/frozen choice lives with the constraint
    # that defines it (graph_of_constraints.hpp's live_edge_phis comment),
    # not with whichever solver happens to run it, so there is nothing for
    # a caller here to legitimately override. Node phi ids are never
    # meaningful here (_resolve_symbolic_constraints hardcodes "frozen"
    # for every node constraint), so any present would simply be inert.
    live_phi_ids = frozenset(graph.live_edge_phis)
    symbolic_constraints = []  # (node_locals_tuple, fn, kind, mode, name)
    interior_constraints = []  # (batched_fn, name) -- see _batch_along_edge_interior_fn
    stationary_constraints = []  # (batched_fn, name) -- see _batch_stationary_edge_fn

    # -- structure derivation ---------------------------------------------

    # Every graph node gets a row, unconditionally -- matching MILP, which
    # gives every node in its (remaining_vertices-scoped) subgraph a W row
    # regardless of whether any phi references it. This solver spans the
    # whole graph rather than just remaining_vertices (see above), so
    # "every node" here means every node in graph.structure, not just
    # remaining_vertices. There is therefore exactly one node-numbering
    # space in this solver: a real graph node id (from graph.structure) IS
    # its own row index into wp/t/node_active/anchor_wp -- no separate
    # local/compacted numbering.
    node_list = list(range(graph.structure.num_nodes))
    n_nodes = len(node_list)
    # Per-agent/per-object padded-slot row layout -- see _slot_width's
    # docstring. `agent_widths`/`slot_width`/`object_widths`/
    # `object_slot_width` (no leading underscore) name the VALUES here,
    # deliberately distinct from the _agent_widths/_slot_width/
    # _object_widths/_object_slot_width module-level functions that compute
    # them -- every reference to any of these names below (including inside
    # the nested functions further down) means this same set of values,
    # never the function.
    agent_widths = _agent_widths(graph)
    slot_width = _slot_width(agent_widths)
    object_widths = _object_widths(graph)
    object_slot_width = _object_slot_width(object_widths)
    state_dim = graph.num_agents * slot_width + graph.num_objects * object_slot_width

    def _resolve_phi_agent_source(phi_id):
        """Resolves ONE phi/constraint's own agent source -- ("fixed",
        agent_id), ("var", var_id), or None if it doesn't establish a
        routing instance (e.g. an object-only formula) -- independently of
        any other phi on the same node, purely for ROUTING/ordering purposes
        (see module docstring; unrelated to whether the formula itself can
        be compiled, which the waypoint row layout always supports)."""
        if phi_id in graph.phi_to_variable_map:
            return ("var", graph.phi_to_variable_map[phi_id])
        if phi_id in graph.phi_to_static_assignment_map:
            return ("fixed", graph.phi_to_static_assignment_map[phi_id])
        formula = graph.phi_to_formula_map.get(phi_id)
        if formula is None:
            return None
        # No entry in either map: either a plain literal-agent_q(k) node
        # constraint (add_constraint's C++ side only records var_agent_q
        # placeholders anywhere, never literal ones -- see add_constraint's
        # own free-variable scan, graph_of_constraints.cpp), an object-only
        # formula (no agent_q(k) reference at all -- correctly falls through
        # to None, no routing instance), or a multi-variable-disjunction phi
        # (references var_agent_q for >1 variable, matches nothing below,
        # correctly falls through to None).
        free_var_ids = {v.get_id() for v in formula.GetFreeVariables()}
        matched = [k for k in range(graph.num_agents)
                   if not _placeholder_id_map(graph.agent_q(k)).keys().isdisjoint(free_var_ids)]
        if len(matched) == 1:
            return ("fixed", matched[0])
        if len(matched) > 1:
            raise ValueError(
                f"phi {phi_id}'s constraint formula references multiple distinct "
                f"literal agent_q(k) placeholders {matched} -- not representable "
                "as a single agent source for routing purposes")
        return None

    # Routing instances: (node, resolved agent source) pairs -- purely a
    # routing/ordering concept (which real agent visits this node, and in
    # what order relative to its other nodes). A node may carry several
    # (distinct agent targets); object references never contribute one
    # (see _resolve_phi_agent_source). Discovered over every node with
    # phis at all -- NOT restricted to Formula-based ones, since a routing
    # instance can be established purely via phi_to_variable_map/phi_to_
    # static_assignment_map (e.g. the older add_robot_linear_eq-style
    # helpers, which register a static assignment but no drake::symbolic
    # Formula at all -- see add_python_constraint/target_eq_constraint's
    # use in examples/pointmass_example.py).
    instance_list = []
    instance_local_id = {}  # (node, src) -> local index, dedup only -- not read past this loop
    for node in node_list:
        for phi_id in graph.node_to_phis_map.get(node, []):
            src = _resolve_phi_agent_source(phi_id)
            if src is None:
                continue
            key = (node, src)
            if key not in instance_local_id:
                instance_local_id[key] = len(instance_list)
                instance_list.append(key)

    # One slot per distinct assignable variable id actually referenced
    # here; statically-fixed instances consume no slot at all (their real
    # agent id is already known, no GA search needed) -- this can
    # legitimately leave n_variables at 0.
    #
    # Also folds in every assignable hold's var_id (hold.var_id, see
    # _resolve_holds below) even though a hold contributes no routing
    # instance of its own (add_assignable_hold's edge carries no node phi
    # -- see get_agent_paths' HoldOwningAgents, goc-mpc's C++ side, for the
    # ownership-resolution counterpart of this same gap). In every current
    # experiment a hold's var_id is ALSO referenced by its u_node's own
    # Pick-style constraint (e.g. pyrobosim_gymnasium's _pick_add pins
    # agent_q via the same var_agent_q(var_id)), so this is redundant
    # there -- but nothing enforces that pairing, and without it a hold
    # whose variable is referenced nowhere else would KeyError in
    # _resolve_holds instead of getting a GA slot.
    var_ids = sorted({src[1] for _, src in instance_list if src[0] == "var"}
                     | {hold.var_id for hold in graph.hold_ops.values()
                        if hold.var_id is not None})
    var_id_to_slot = {v: i for i, v in enumerate(var_ids)}
    n_variables = len(var_ids)

    instance_sources = [
        (kind, var_id_to_slot[val] if kind == "var" else val)
        for _, (kind, val) in instance_list
    ]
    instance_node = [node for node, _ in instance_list]

    # Ordering edges are node pairs, not instance pairs: kernel.py's
    # topological decoder (_topological_rank) computes a visiting-order
    # rank for EVERY node unconditionally, regardless of whether that node
    # carries a routing instance -- unlike the older arrival-time scheme
    # this replaced (where node_arrival_time was only ever populated for
    # nodes an agent's own route actually visited, via instance_node -- a
    # node with no routing instance stayed stuck at its -inf init value,
    # so an edge touching one was genuinely unconstrainable then). So an
    # ordering edge no longer needs either endpoint to carry a routing
    # instance: it's entirely normal, and common in this package's own
    # experiments (e.g. a "release"/"place" node whose agent position is
    # established purely by a transport EDGE constraint, not a node
    # constraint -- see e.g. object_grasp_experiment.py's n_place), for a
    # node that must still participate in precedence to have no routing
    # instance of its own.

    # Hard edges: conditional edges never enter `structure` (see
    # add_conditional_edge_ordering's doc -- "invisible to BFS/routing"),
    # so every structure edge is an unconditional precedence constraint.
    hard_edges = []
    for u in node_list:
        for e in graph.structure.neighbors(u):
            hard_edges.append((u, e.to))

    # Conditional ordering edges: compile each Formula into a gate_fn
    # closed over owner_variable/cond_binary index maps.
    var_sym_ids = {graph.assignment_sym(v).get_id(): var_id_to_slot[v] for v in var_ids}
    cond_binary_vars = graph.binary_cond_sym_vars
    cond_sym_ids = {v.get_id(): i for i, v in enumerate(cond_binary_vars)}
    n_cond_vars = len(cond_binary_vars)

    cond_edges = []
    for (u, v), formula in graph.conditional_ordering_map.items():
        gate = compile_condition(formula, var_sym_ids, cond_sym_ids)
        cond_edges.append((u, v, gate))

    # Needed here (not just below, where symbolic/hold/stationary
    # constraint resolution also uses it) since gated interior/stationary
    # constraints (_batch_along_edge_interior_fn, _batch_relational_
    # interior_fn, _batch_stationary_edge_fn) need the SAME decode_node_
    # rank the routing kernel itself will use, to gate on an exact
    # per-individual topologically valid order instead of raw t (see
    # kernel.build_decode_node_rank's docstring).
    ordering_edges = [(u, v, None) for u, v in hard_edges] + list(cond_edges)
    decode_node_rank = build_decode_node_rank(ordering_edges, n_nodes)

    row_resolver = _make_row_resolver(graph, var_id_to_slot, agent_widths, slot_width,
                                       object_widths, object_slot_width)

    # -- symbolic constraints -----------------------------------------------

    def _resolve_symbolic_constraints():
        """Auto-derives eq/ineq residual constraints from the graph's unified
        symbolic API (add_constraint / add_assignable_constraint /
        add_edge_constraint), compiling each stored Formula the same way MILP
        does structurally. Raises (via _make_row_resolver, through
        _unsupported_placeholder) if a formula references a placeholder
        genuinely outside that scope (e.g. a u-/ v-side placeholder inside a
        node constraint). An edge constraint's stored formula compiles
        differently depending on graph.edge_phi_to_along_edge_map: relationally
        (once, over both endpoint rows) or "along the edge" (once, applied
        independently to each endpoint's own row) -- see the branch below. Each
        phi's `mode` ("live" if its id is in live_phi_ids, else "frozen")
        threads through to build_graph_ordering_problem's _batch_symbolic_
        constraint_fn/_batch_along_edge_interior_fn calls below -- see
        problem.apply_anchor's docstring for what the two modes mean."""
        node_formulas = graph.phi_to_formula_map
        for node in node_list:
            for phi_id in graph.node_to_phis_map.get(node, []):
                if phi_id not in node_formulas:
                    continue  # not a Formula-based (symbolic) constraint
                formula = node_formulas[phi_id]
                # Always frozen, never checked against live_phi_ids: a
                # node constraint can only reference its own node's
                # placeholders (see graph_of_constraints.hpp's live_edge_phis
                # comment), so it has no live/frozen distinction to make --
                # and node phi ids and edge phi ids are independent counters
                # that can collide numerically, so checking a node phi_id
                # against a set that only ever holds EDGE phi ids would risk
                # spuriously matching an unrelated edge's id.
                mode = "frozen"
                resolver = lambda var, row_resolver=row_resolver: row_resolver(var, 1)
                fn, kind = compile_relational_formula(formula, resolver)
                symbolic_constraints.append(
                    ((node,), fn, kind, mode, f"phi_{phi_id}"))

        edge_formulas = graph.edge_phi_to_formula_map
        edge_along_edge = graph.edge_phi_to_along_edge_map
        for (u, v), phi_ids in graph.edge_to_phis_map.items():
            for phi_id in phi_ids:
                if phi_id not in edge_formulas:
                    continue  # not a Formula-based (symbolic) edge constraint
                formula = edge_formulas[phi_id]
                mode = "live" if phi_id in live_phi_ids else "frozen"

                if edge_along_edge.get(phi_id, False):
                    # "Along the edge" -- built from the plain agent_q/
                    # object_q/var_agent_q placeholders (same as a node
                    # constraint), so compile it ONCE against a single-slot
                    # (node-scoped) resolver, then register the resulting fn
                    # at each endpoint's own row (exact, its own persistent AL
                    # multiplier each), PLUS a best-effort aggregate
                    # re-application at any OTHER node whose `t` currently
                    # falls between the endpoints' -- see
                    # _batch_along_edge_interior_fn.
                    resolver = lambda var, row_resolver=row_resolver: row_resolver(var, 1)
                    fn, kind = compile_relational_formula(formula, resolver)
                    symbolic_constraints.append(
                        ((u,), fn, kind, mode, f"edge_phi_{phi_id}_u"))
                    symbolic_constraints.append(
                        ((v,), fn, kind, mode, f"edge_phi_{phi_id}_v"))
                    interior_batched = _batch_along_edge_interior_fn(
                        fn, kind, u, v, decode_node_rank, mode=mode)
                    interior_constraints.append(
                        (interior_batched, f"edge_phi_{phi_id}_interior"))
                    continue

                resolver = lambda var, row_resolver=row_resolver: row_resolver(var, 2)
                fn, kind = compile_relational_formula(formula, resolver)
                symbolic_constraints.append(
                    ((u, v), fn, kind, mode, f"edge_phi_{phi_id}"))

    def _resolve_holds():
        """Auto-derives rigid-carry (translation-only) constraints from the
        graph's canonical hold registry (graph.hold_ops -- add_hold/
        add_assignable_hold) -- the JAX analogue of MILP's Constraint 14a
        (milp_waypoint_mpc.cpp).

        For each held point, enforces that the object's held position moves
        exactly as the holding robot's end-effector (its "ee" link, or
        whatever fk_fn is registered for it) moves in world space:
            v_object_q(oid)[:workspace_dim] - u_object_q(oid)[:workspace_dim]
                == fk(v_agent_q(robot_ag)) - fk(u_agent_q(robot_ag))
        (position-only -- workspace_dim, not the object's own full width --
        matching "translation-only" above). This is deliberately NOT the
        raw agent_q(robot_ag) delta the old version of this method used:
        agent_q is a robot's raw CONFIGURATION row (e.g. joint angles for an
        articulated arm), which has no fixed relation to its end-effector's
        world-space delta, and (for any robot whose config width != the
        object's own width, e.g. any articulated arm) doesn't even have a
        matching shape to
        subtract against the object's row -- eq() would throw on the length
        mismatch. fk_fn is resolved via default_fk.resolve_link_fk: a
        graph.set_robot_fk-registered override for (robot_ag, "ee") if
        present, else a built-in closed-form pose keyed by the robot's
        inferred configuration-space kind (point-mass/pos+yaw/pos+quaternion
        /pos+rotation-matrix -- see that module's docstring), matching
        GraphOfConstraints::link_pose's own registry-then-builtin fallback.
        Raises (via default_fk_for) if robot_ag's configuration space is
        articulated and has no registered fk_fn -- there's no default for
        that case, a real fk_fn is required.

        Static holds (hold.robot_ag set) pin `agent_col0`/`fk_fn` to a plain
        Python int/closure at spec-build time -- robot_ag is already known,
        no search needed. Assignable holds (hold.var_id set) instead need
        the SAME GA-searched agent resolved consistently on BOTH row_u and
        row_v -- which side of a relational formula var_agent_q(var) binds
        is fixed per placeholder in the generic resolver (_make_row_
        resolver: "it only ever binds side 0"), so that path can't express
        this. Handled here instead, mirroring MILP's own resolution
        (milp_waypoint_mpc.cpp: "For each possible agent k, construct the
        same residual as static-HoldSpec(robot_ag=k)"): build one static
        branch per candidate agent k (exactly the static-hold computation
        above, just not yet selected), stack their (v_pos - u_pos) results,
        and gather the one at `owner_variable[slot]` via
        jax.lax.dynamic_index_in_dim -- the same hard-select-then-
        dynamic-slice convention _make_row_resolver's var_map branch
        already uses for a single placeholder, just applied to two rows and
        gathering a precomputed delta instead of a raw column. `slot` comes
        from `var_id_to_slot`, which folds in every hold's var_id even when
        (as here) nothing else references it -- see the instance-discovery
        comment above.

        Since fk_fn is called directly here rather than substituted through
        the unified symbolic placeholder API (agent_link_pos has no u_/v_
        relational counterpart -- see spec.py's module docstring and
        _make_row_resolver's), this bypasses compile_relational_formula's
        eq()-formula path entirely and builds the residual fn by hand,
        matching its exact fn(row_u, row_v, owner_variable) -> (k,) jnp
        array contract (compile_relational_formula's own docstring) so it
        still plugs into the same registration/interior-reinforcement
        machinery below unmodified. Registered mode="live" -- unlike an
        ordinary edge constraint (frozen by default), a hold's u-side should
        track the REAL state once u has passed, not the frozen plan: see
        mpc.py's module docstring -- pyrobosim's actual grasp model snaps
        agent and object to the same real pose the instant a grasp happens,
        so a frozen reading would keep propagating the by-then-fictional
        planned offset forward forever.

        Also reinforces the same relation, betweenness-gated, at every OTHER
        node the solved route might schedule between the hold's u and v, via
        _batch_relational_interior_fn -- the JAX analogue of Constraint 14a's
        own interior-reinforcement loop -- since another agent's route might
        otherwise schedule a node in between that leaves the held object's
        value there completely unconstrained."""
        agents_width = graph.num_agents * slot_width
        workspace_dim = graph.workspace_dim

        for hold_id, hold in graph.hold_ops.items():
            u, v = hold.u_node, hold.v_node
            mode = "live"

            for oid in hold.held_point_ids:
                obj_col0 = agents_width + oid * object_slot_width

                if hold.robot_ag is not None:
                    fk_fn = resolve_link_fk(graph, hold.robot_ag)
                    agent_col0 = hold.robot_ag * slot_width
                    agent_w = agent_widths[hold.robot_ag]

                    def fn(row_u, row_v, owner_variable, params, fk_fn=fk_fn,
                           agent_col0=agent_col0, agent_w=agent_w, obj_col0=obj_col0,
                           workspace_dim=workspace_dim):
                        del owner_variable, params  # robot_ag is static, not GA-searched; no param(id) here
                        u_pos, _u_rot = fk_fn(row_u[agent_col0:agent_col0 + agent_w])
                        v_pos, _v_rot = fk_fn(row_v[agent_col0:agent_col0 + agent_w])
                        obj_u = row_u[obj_col0:obj_col0 + workspace_dim]
                        obj_v = row_v[obj_col0:obj_col0 + workspace_dim]
                        return (obj_v - obj_u) - (v_pos - u_pos)
                else:
                    slot = var_id_to_slot[hold.var_id]
                    # One (agent_col0, agent_w, fk_fn) triple per candidate
                    # agent, Python-time (fk_fn may genuinely differ per
                    # agent -- heterogeneous configuration kinds -- so this
                    # can't be a single traced closure the way agent_col0
                    # alone could be); which branch actually applies is
                    # resolved at trace time below via owner_variable[slot].
                    # Each branch slices its own candidate's real width
                    # (agent_w), not the shared slot_width stride -- fk_fn
                    # expects that agent's genuine config, no padding.
                    branch_cols = [k * slot_width for k in range(graph.num_agents)]
                    branch_widths = [agent_widths[k] for k in range(graph.num_agents)]
                    branch_fks = [resolve_link_fk(graph, k)
                                  for k in range(graph.num_agents)]

                    def fn(row_u, row_v, owner_variable, params,
                           branch_cols=branch_cols, branch_widths=branch_widths,
                           branch_fks=branch_fks, slot=slot,
                           obj_col0=obj_col0, workspace_dim=workspace_dim):
                        del params  # no param(id) placeholder in this residual
                        # Every candidate's (v_pos - u_pos), stacked -- same
                        # per-candidate enumeration MILP does explicitly via
                        # binary indicators (milp_waypoint_mpc.cpp), here
                        # just gathered instead of gated. num_agents is
                        # always small, so this is a handful of extra fk
                        # evaluations, not a real cost.
                        deltas = jnp.stack([
                            fk(row_v[col:col + w])[0] - fk(row_u[col:col + w])[0]
                            for col, w, fk in zip(branch_cols, branch_widths, branch_fks)
                        ])  # (num_agents, workspace_dim)
                        agent = owner_variable[slot]
                        # Hard select, same convention _make_row_resolver's
                        # var_map branch uses (argmax + dynamic_slice):
                        # differentiable w.r.t. wp (row_u/row_v feed the
                        # stacked deltas), not w.r.t. which agent -- that's
                        # already true of every other assignable-var
                        # resolution in this file, not a new tradeoff.
                        delta = jax.lax.dynamic_index_in_dim(deltas, agent, axis=0, keepdims=False)
                        obj_u = row_u[obj_col0:obj_col0 + workspace_dim]
                        obj_v = row_v[obj_col0:obj_col0 + workspace_dim]
                        return (obj_v - obj_u) - delta

                kind = "eq"
                name = f"hold_{hold_id}_obj_{oid}"
                symbolic_constraints.append(((u, v), fn, kind, mode, name))
                interior_batched = _batch_relational_interior_fn(
                    fn, kind, u, v, decode_node_rank, mode=mode)
                interior_constraints.append((interior_batched, f"{name}_interior"))

    def _resolve_stationary_objects():
        """Auto-derives the default "an object not currently being held must
        not move" invariant -- the JAX analogue of MILP's Constraint 14b
        (milp_waypoint_mpc.cpp), always enforced (unlike 14a/_resolve_holds,
        which is translation-only-static-holds for now): 14b only ever
        compares plain object segments with no rotation involved, so it has
        no rotation-dependent case to defer the way 14a's rigid-carry
        relation does.

        Unlike every other constraint in this module, this one is never
        authored by a graph builder -- there is no add_edge_constraint call
        to discover via graph.edge_to_phis_map. It's a default applied to
        EVERY (structural edge, object) pair unconditionally, mirroring
        MILP's own unconditional `for (int obj = 0; ...) for (edge : ...)`
        loop -- synthesized directly from hard_edges x
        range(graph.num_objects), gated per pair via _batch_stationary_edge_fn
        against every hold (graph.hold_ops) declared anywhere on that
        object. An object with zero declared holds anywhere still gets the
        (now always-on) equality -- matching MILP's `if (!any_gate)
        AddLinearEqualityConstraint` branch, since the gate is vacuously
        always-off with no hold_node_pairs to overlap against.

        The gate is dynamic (per solved individual), via this function's
        enclosing decode_node_rank -- an EXACT topologically valid
        visiting-order rank (kernel.build_decode_node_rank), not raw `t` (see
        that function's docstring, and _batch_stationary_edge_fn's, for why
        raw t alone is unsafe here: it's a pure GA-searched sort key with no
        pressure to stay magnitude-monotonic once decode fixes the true
        order -- empirically confirmed to drift even on a trivial 3-node
        chain). An EARLIER version of this function used a static,
        t-independent structural-reachability exemption instead (an edge is
        exempt iff it lies on some path from a hold's u to its v) -- correct
        for the deterministic/nested case (verified against
        test_interior_reinforcement, examples/test_hold_registry.py: hold
        n0->n2 spanning structural n0->n1->n2 exempts both (n0,n1) and
        (n1,n2) either way), but WRONG in general: whether a structural edge
        genuinely overlaps a hold's span can depend on the solved schedule
        itself when the two are structurally unordered (concurrent
        branches, no DAG precedence either way) -- there, which interleaving
        the GA settles on IS part of what's being searched over, so no
        static, t-independent answer exists. Decoded rank resolves this
        correctly in both cases: it's the exact, valid schedule that
        particular individual represents, deterministic or concurrent alike,
        so a betweenness/overlap test built on it is correct for that
        individual specifically (this is exactly the same generalization
        _batch_along_edge_interior_fn/_batch_relational_interior_fn now use
        too -- one unified decode-rank-based gating mechanism for all of
        this module's edge constraints, per-edge along-edge/relational
        registration plus decode-rank-gated interior/stationary
        reinforcement alike).

        Uses EVERY hold in the registry for gating (both static AND
        assignable -- _resolve_holds' rigid-carry relation handles both too,
        see its docstring) -- the gate only needs to know WHEN a hold's own
        node interval falls, not which agent resolves it, so an assignable
        hold's span is real, usable gating information regardless of how
        its own rigid-carry formula resolves the agent.

        Registered mode="live" (see _resolve_holds' docstring for the same
        reasoning applied to hold rigidity): once u has passed, an untouched
        object's real current position (x0) is the ground truth going
        forward, not whatever was merely planned there.

        Edge-to-edge chaining alone has nothing to anchor a graph SOURCE
        node (no incoming hard edge) to: with no real predecessor, it's only
        ever tied to itself via the equality above, so its value is free to
        drift to whatever the GA/AL search finds convenient rather than the
        object's actual position. MILP's own Constraint 14b covers this with
        a second loop over `subgraph.structure.sources()`, feeding the real
        depot `x0` in as each source's "previous waypoint"
        (milp_waypoint_mpc.cpp: `for (v14b : subgraph.structure.sources())`)
        -- mirrored below via _batch_depot_stationary_fn, one source node at
        a time, gated by the same hold-overlap logic (see that function's
        docstring for how the depot's own always-earliest rank makes the
        existing gate math work unmodified)."""
        agents_width = graph.num_agents * slot_width

        holds_by_object = {}
        for hold in graph.hold_ops.values():
            for oid in hold.held_point_ids:
                holds_by_object.setdefault(oid, []).append((hold.u_node, hold.v_node))

        # Source nodes: no incoming hard edge -- the only ones with no real
        # structural predecessor for the depot loop below to anchor.
        edge_targets = {v for _u, v in hard_edges}
        source_nodes = [n for n in node_list if n not in edge_targets]

        for oid in range(graph.num_objects):
            obj_col0 = agents_width + oid * object_slot_width
            seg_slice = slice(obj_col0, obj_col0 + object_widths[oid])
            hold_node_pairs = holds_by_object.get(oid, [])
            for u, v in hard_edges:
                batched = _batch_stationary_edge_fn(
                    u, v, seg_slice, hold_node_pairs, decode_node_rank, mode="live")
                stationary_constraints.append((batched, f"stationary_obj_{oid}_{u}_{v}"))
            for v in source_nodes:
                depot_batched = _batch_depot_stationary_fn(
                    v, seg_slice, hold_node_pairs, decode_node_rank, mode="live")
                stationary_constraints.append((depot_batched, f"stationary_obj_{oid}_depot_{v}"))

    _resolve_symbolic_constraints()
    _resolve_holds()
    _resolve_stationary_objects()

    # -- python constraints + build -----------------------------------------

    eq_constraints, ineq_constraints = [], []
    for node, fn, kind, _name in python_constraints:
        if kind not in ("eq", "ineq"):
            raise ValueError(f"Unknown constraint kind {kind!r}, expected 'eq' or 'ineq'")
        batched = _batch_python_constraint_fn(fn, node)
        (eq_constraints if kind == "eq" else ineq_constraints).append(batched)
    for node_locals, fn, kind, mode, _name in symbolic_constraints:
        batched = _batch_symbolic_constraint_fn(fn, node_locals, mode=mode)
        (eq_constraints if kind == "eq" else ineq_constraints).append(batched)
    for batched, _name in interior_constraints:
        ineq_constraints.append(batched)
    for batched, _name in stationary_constraints:
        ineq_constraints.append(batched)

    return GraphOrderingRelaxed(
        instance_sources=instance_sources,
        n_variables=n_variables,
        ordering_edges=ordering_edges,
        x0=np.asarray(x0),
        wp_bounds=wp_bounds,
        instance_node=instance_node,
        n_nodes=n_nodes,
        state_dim=state_dim,
        n_cond_vars=n_cond_vars,
        objective=objective,
        edge_cost_fn=edge_cost_fn,
        eq_constraints=eq_constraints,
        ineq_constraints=ineq_constraints,
        # Structural shape only (n_params) -- fixes the jitted GA's
        # `params` argument width for this problem's whole lifetime, same
        # as x0 fixes n_agents/dim. The actual VALUES a live solve() reads
        # come fresh from graph.view_param_values() each call (mpc.py),
        # same as x0 -- this initial snapshot only matters for params-less
        # structural call sites (build_initial_carry_fn's cold start,
        # run_lamarckian_al/warmup_lamarckian_al's default).
        params=np.asarray(graph.view_param_values()),
        # Plain problem-instance data, stashed the same way instance_
        # sources/instance_node/ordering_edges already are (see
        # GraphOrderingRelaxed's own comment on that) so a caller with only
        # `problem` in hand -- EvolutionaryWaypointSolver._compute_anchor,
        # mpc.py -- can recover which nodes reference a given GA-searched
        # variable, without this module needing to stay alive as a
        # separate object past this call.
        instance_list=instance_list,
        var_id_to_slot=var_id_to_slot,
    )

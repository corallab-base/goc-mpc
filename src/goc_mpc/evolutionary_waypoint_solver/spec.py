"""GraphOrderingSpec: adapter that derives a GraphOrderingRelaxed problem
instance (see problem.py) from a real C++ GraphOfConstraints, so one graph
definition can drive either the MILP (MILPWaypointMPC) or this JAX-native
evolutionary solver.

Built ONCE per GraphOfConstraints and reused for the solver's whole lifetime
-- unlike SubgraphOfConstraints, which MILP rebuilds fresh every solve from
remaining_vertices. This solver instead spans the WHOLE graph unconditionally
(see GraphOrderingSpec's class docstring) and takes remaining_vertices as a
runtime AnchorState (mpc.py/problem.py) fed into the already-compiled
GraphOrderingRelaxed at solve time -- a node dropping out of remaining_
vertices never removes its dense row, routing instance, or any constraint
touching it; it only switches that node from a free decision variable to a
known constant (its last-committed value), exactly mirroring how MILP
substitutes a passed node's frozen waypoint into a boundary edge constraint
rather than dropping it. Node/edge constraints added via the graph's unified
symbolic API
(add_constraint / add_assignable_constraint / add_edge_constraint) are
auto-derived from graph.phi_to_formula_map/edge_phi_to_formula_map and
compiled via formula_compiler.compile_relational_formula -- the same Formula
that drives MILPWaypointMPC also drives this solver, no manual duplication.

Each relevant graph node gets a DENSE per-node row -- `[agent_0 | agent_1 |
... | agent_{n_agents-1} | object_0 | ... | object_{n_objects-1}]`,
`state_dim = num_agents*dim + num_objects*non_robot_dim` wide -- mirroring
MILP's per-node joint configuration `W` (graph_of_constraints.cpp:79's
`total_dim`). So a node/edge constraint can reference ANY agent's or
object's placeholder (agent_q(k), object_q(k), var_agent_q(var), and their
u_-/v_-prefixed edge counterparts) via a plain column offset into that row
(see _make_dense_resolver) -- no per-formula "which sparse row" bookkeeping,
and no restriction on how many distinct agents/objects one formula mixes
(e.g. a rigid-attachment/grasp formula tying agent_q(k) to object_q(j) at a
node, or a *relational* edge formula tying v_object_q - u_object_q to
v_agent_q - u_agent_q). An edge constraint built from the plain, un-prefixed
placeholders instead means "along the edge" -- an invariant applied
independently at both endpoints (see edge_phi_to_along_edge_map and
_resolve_symbolic_constraints below), not a relation between them.

Routing/ordering is a separate, unchanged concern layered on top: a routing
"instance" (node, resolved agent source) still exists exactly as before --
`("fixed", agent_id)` (a literal agent_q(k) constraint) or `("var", var_id)`
(an assignable var_agent_q(var) constraint) -- and still drives which real
agent's route includes which node, in what GA-searched order (t). An
instance's own wp is now simply gathered from its node's dense row (see
kernel.py) rather than owning independent storage. Object references never
create routing instances or need any special "ownership" -- an object's
column just sits in whichever node rows reference it, constrained only by
whatever symbolic formulas mention it, with no route/ordering semantics of
its own.

A constraint referencing a placeholder genuinely outside this scope (e.g.
u_agent_q/v_agent_q inside a *node* constraint, which has no u/v side) raises
at spec-construction time. For anything outside this solver's symbolic scope
entirely (e.g. a non-symbolic/black-box cost), add_python_constraint remains
available as an escape hatch (single agent-instance scope only, same as
before).
"""

import jax
import jax.numpy as jnp
import numpy as np

from .formula_compiler import as_variable, compile_condition, compile_relational_formula
from .problem import GraphOrderingRelaxed


def target_eq_constraint(target):
    """Convenience single-instance constraint: pins a node's waypoint to a
    fixed target exactly (H = wp_row - target, feasible at H=0)."""
    target = np.asarray(target, dtype=float)

    def fn(wp_row):
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


def _unsupported_placeholder(var):
    raise ValueError(
        f"Symbolic constraint references placeholder variable {var!r} that "
        "isn't representable here -- expected one of agent_q(k)/object_q(k)/"
        "var_agent_q(var) (node constraints, or an edge constraint's \"along "
        "the edge\" form) or u_agent_q(k)/u_object_q(k) (an edge constraint's "
        "u side) or v_agent_q(k)/v_object_q(k) (an edge constraint's v side; "
        "u_/v_-prefixed placeholders are not valid inside a node constraint, "
        "which has no u/v side). Use add_python_constraint for anything "
        "genuinely outside this scope.")


def _object_ids_referenced(formula, graph, placeholder_fn):
    """Which object ids (0..graph.num_objects-1) `formula` references via
    `placeholder_fn` (graph.object_q or graph.v_object_q) -- used only for
    write-back bookkeeping (mpc.py's view_object_waypoints()), never for
    constraint compilation itself (see _make_dense_resolver: object columns
    are always addressable regardless of which formula, if any, references
    them at a given node)."""
    free_var_ids = {v.get_id() for v in formula.GetFreeVariables()}
    return [oid for oid in range(graph.num_objects)
            if not _placeholder_id_map(placeholder_fn(oid)).keys().isdisjoint(free_var_ids)]


def _make_dense_resolver(graph, var_id_to_slot):
    """Builds resolve(var, n_row_slots) -> Callable[*rows, owner_variable]
    for the unified symbolic constraint API, over the dense per-node row
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
    as the trailing argument after every row by _batch_symbolic_constraint_
    fn), resolved via a differentiable jax.lax.dynamic_slice so gradient-
    based local refinement of wp still works through it. It only ever binds
    side 0 (mirrors add_edge_constraint's C++ substitution, which never
    binds var_agent_q on the v side of a relational edge constraint, and an
    "along the edge" constraint has no v side to bind at all -- it's just a
    single node-scoped placeholder set, applied once per endpoint).

    n_row_slots: 1 for a node constraint or an "along the edge" edge
    constraint (only side 0 valid -- u_/v_agent_q / u_/v_object_q correctly
    raise via _unsupported_placeholder), 2 for a relational edge constraint
    (side 0 = u, side 1 = v).
    """
    dim, num_agents = graph.dim, graph.num_agents
    num_objects, non_robot_dim = graph.num_objects, graph.non_robot_dim
    agents_width = num_agents * dim

    static_map = {}  # var_id -> (side, col)
    for k in range(num_agents):
        for j, expr in enumerate(graph.agent_q(k)):
            v = as_variable(expr)
            if v is not None:
                static_map[v.get_id()] = (0, k * dim + j)
        for j, expr in enumerate(graph.u_agent_q(k)):
            v = as_variable(expr)
            if v is not None:
                static_map[v.get_id()] = (0, k * dim + j)
    for k in range(num_objects):
        for j, expr in enumerate(graph.object_q(k)):
            v = as_variable(expr)
            if v is not None:
                static_map[v.get_id()] = (0, agents_width + k * non_robot_dim + j)
        for j, expr in enumerate(graph.u_object_q(k)):
            v = as_variable(expr)
            if v is not None:
                static_map[v.get_id()] = (0, agents_width + k * non_robot_dim + j)
    for k in range(num_agents):
        for j, expr in enumerate(graph.v_agent_q(k)):
            v = as_variable(expr)
            if v is not None:
                static_map[v.get_id()] = (1, k * dim + j)
    for k in range(num_objects):
        for j, expr in enumerate(graph.v_object_q(k)):
            v = as_variable(expr)
            if v is not None:
                static_map[v.get_id()] = (1, agents_width + k * non_robot_dim + j)

    var_map = {}  # var_id -> (slot, component j)
    for var_id, slot in var_id_to_slot.items():
        for j, expr in enumerate(graph.var_agent_q(var_id)):
            v = as_variable(expr)
            if v is not None:
                var_map[v.get_id()] = (slot, j)

    def resolve(var, n_row_slots):
        vid = var.get_id()
        if vid in static_map:
            side, col = static_map[vid]
            if side >= n_row_slots:
                _unsupported_placeholder(var)
            return lambda *args, side=side, col=col: args[side][col]
        if vid in var_map:
            slot, j = var_map[vid]

            def fn(*args, slot=slot, j=j, dim=dim):
                owner_variable = args[-1]
                agent = owner_variable[slot]
                return jax.lax.dynamic_slice_in_dim(args[0], agent * dim + j, 1)[0]
            return fn
        _unsupported_placeholder(var)
    return resolve


def _batch_symbolic_constraint_fn(fn, node_locals):
    """Wraps a compiled symbolic-constraint residual fn(*rows, owner_variable)
    -> (k,) into the batched (assign, t, wp) -> (pop, k) contract solver.py
    expects, via jax.vmap over the population. `wp` is the dense per-node
    tensor (pop, n_dense_nodes, state_dim); each entry of `node_locals` is a
    dense-node index (this constraint's node, or (u, v) for an edge
    constraint) -- NOT a routing-instance id (see module docstring)."""
    vmapped = jax.vmap(fn)

    def batched(assign, t, wp, node_locals=node_locals, vmapped=vmapped):
        rows = [wp[:, nl, :] for nl in node_locals]
        owner_variable = jnp.argmax(assign, axis=-1)
        return vmapped(*rows, owner_variable)
    return batched


def _batch_along_edge_interior_fn(fn, kind, u_local, v_local, node_repr_instance, node_has_instance):
    """Best-effort JAX analogue of MILP's betweenness-gated interior_builder
    re-application (see milp_waypoint_mpc.cpp's Constraint 13b / kernel.py's
    module docstring on `t`): re-applies an "along the edge" formula's
    residual at every OTHER dense node whose representative routing
    instance's `t` currently falls between u_local's and v_local's --
    cheap here since `t` is a real-valued, directly comparable arrival-time
    surrogate (`perm = jnp.argsort(t)` in kernel.py), unlike MILP's
    big-M-encoded betweenness binary.

    Unlike the exact per-node registrations _resolve_symbolic_constraints
    makes for u_local/v_local themselves (each its own persistent AL
    multiplier -- unaffected by this function), every OTHER between-node's
    contribution is aggregated into a SINGLE non-negative "total interior
    violation" per residual component -- summing masked per-node
    relu(residual) (ineq) or residual**2 (eq) -- and always registered as
    one ineq-style constraint (feasible at <=0; since it's a sum of
    non-negative terms that's equivalent to "exactly zero everywhere
    between"), regardless of the formula's own eq/ineq kind. This trades
    exact per-node multipliers (which would need a per-individual,
    dynamically-sized multiplier count -- mu/lam are fixed-size for the
    solver's whole lifetime, see problem.GraphOrderingRelaxed) for a single
    shared multiplier whose "identity" drifts as the between-set changes
    generation to generation. Acceptable since this only ADDS coverage that
    was previously entirely missing (endpoints keep their exact per-node
    registrations, not touched here).

    A node with several routing instances (multiple agents/vars
    independently constrained there) uses one arbitrary representative
    instance's `t` as a stand-in for "was this node visited around now" --
    fine, since the along-edge formula is evaluated against the node's
    WHOLE dense row regardless of which instance triggered inclusion. A
    node with no routing instance at all (e.g. a pure object_q-only node)
    never participates (excluded via node_has_instance)."""
    vmapped_pop = jax.vmap(fn)

    def batched(assign, t, wp, u_local=u_local, v_local=v_local,
                node_repr_instance=node_repr_instance, node_has_instance=node_has_instance,
                kind=kind, vmapped_pop=vmapped_pop):
        owner_variable = jnp.argmax(assign, axis=-1)

        def per_node(node_rows):  # node_rows: (pop, state_dim) -- wp[:, nd, :]
            return vmapped_pop(node_rows, owner_variable)  # (pop, k)

        # vmap over wp's dense-node axis (1) -- (pop, n_dense_nodes, state_dim)
        # -> (n_dense_nodes, pop, k).
        all_residuals = jax.vmap(per_node, in_axes=1, out_axes=0)(wp)
        viol = jnp.maximum(0.0, all_residuals) if kind == "ineq" else all_residuals ** 2

        node_t = t[:, node_repr_instance]  # (pop, n_dense_nodes)
        lo = jnp.minimum(node_t[:, u_local], node_t[:, v_local])
        hi = jnp.maximum(node_t[:, u_local], node_t[:, v_local])
        between = (node_t >= lo[:, None]) & (node_t <= hi[:, None]) & node_has_instance[None, :]
        between = between.at[:, u_local].set(False)
        between = between.at[:, v_local].set(False)

        masked = viol * jnp.transpose(between)[:, :, None]  # (n_dense_nodes, pop, k)
        return jnp.sum(masked, axis=0)  # (pop, k)
    return batched


def _batch_python_constraint_fn(fn, dense_node, kind, val, dim, n_variables):
    """Wraps a single-instance add_python_constraint fn(wp_row) -> (k,) (see
    target_eq_constraint) into the batched (assign, t, wp) -> (pop, k)
    contract, gathering that instance's own dim-wide agent slice out of its
    node's dense row -- the same kind of gather kernel.py's routing performs
    for every instance at once, just for this one instance."""
    vmapped = jax.vmap(fn)
    is_var = (kind == "var")
    fixed_agent = val if kind == "fixed" else 0
    var_slot = val if kind == "var" else 0

    def batched(assign, t, wp, dense_node=dense_node, is_var=is_var, fixed_agent=fixed_agent,
                var_slot=var_slot, dim=dim, n_variables=n_variables, vmapped=vmapped):
        node_row = wp[:, dense_node, :]
        if n_variables > 0 and is_var:
            owner_variable = jnp.argmax(assign, axis=-1)
            agent = owner_variable[:, var_slot]
        else:
            agent = jnp.full((wp.shape[0],), fixed_agent, dtype=jnp.int32)
        col = agent * dim
        idx = col[:, None] + jnp.arange(dim)[None, :]
        row = jnp.take_along_axis(node_row, idx, axis=1)
        return vmapped(row)
    return batched


class GraphOrderingSpec:
    """Spans the WHOLE graph, independent of any receding-horizon
    remaining_vertices set -- built once per GraphOfConstraints and reused
    for every solve() (see EvolutionaryWaypointSolver in mpc.py, which feeds
    remaining_vertices in as a runtime AnchorState instead: a node no longer
    in remaining_vertices keeps its dense row and any routing instance, but
    solver.py's apply_anchor() substitutes its last-committed value in place
    of the free decision variable there, and kernel.py masks it out of
    routing/ordering -- so a node/edge constraint anchored at an already-
    passed node is never dropped, only its "passed" side becomes a known
    constant instead of a free variable)."""

    def __init__(self, graph, x0, wp_bounds,
                 objective="avg", edge_cost_fn=None, penalty_weight=20.0):
        self.graph = graph
        self.x0 = x0
        self.wp_bounds = wp_bounds
        self.objective = objective
        self.edge_cost_fn = edge_cost_fn
        self.penalty_weight = penalty_weight
        self._python_constraints = []  # (instance_local_idx, fn, kind, name)
        self._symbolic_constraints = []  # (dense_node_locals_tuple, fn, kind, name)
        self._interior_constraints = []  # (batched_fn, name) -- see _batch_along_edge_interior_fn

    @classmethod
    def from_graph_of_constraints(cls, graph, x0, wp_bounds,
                                   objective="avg", edge_cost_fn=None, penalty_weight=20.0):
        spec = cls(graph, x0, wp_bounds, objective, edge_cost_fn, penalty_weight)
        spec._resolve_structure()
        return spec

    # -- structure derivation -------------------------------------------

    def _resolve_phi_agent_source(self, phi_id):
        """Resolves ONE phi/constraint's own agent source -- ("fixed",
        agent_id), ("var", var_id), or None if it doesn't establish a
        routing instance (e.g. an object-only formula) -- independently of
        any other phi on the same node, purely for ROUTING/ordering purposes
        (see module docstring; unrelated to whether the formula itself can
        be compiled, which the dense row layout always supports)."""
        if phi_id in self.graph.phi_to_variable_map:
            return ("var", self.graph.phi_to_variable_map[phi_id])
        if phi_id in self.graph.phi_to_static_assignment_map:
            return ("fixed", self.graph.phi_to_static_assignment_map[phi_id])
        formula = self.graph.phi_to_formula_map.get(phi_id)
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
        matched = [k for k in range(self.graph.num_agents)
                   if not _placeholder_id_map(self.graph.agent_q(k)).keys().isdisjoint(free_var_ids)]
        if len(matched) == 1:
            return ("fixed", matched[0])
        if len(matched) > 1:
            raise ValueError(
                f"phi {phi_id}'s constraint formula references multiple distinct "
                f"literal agent_q(k) placeholders {matched} -- not representable "
                "as a single agent source for routing purposes")
        return None

    def _resolve_structure(self):
        node_list = [n for n in range(self.graph.structure.num_nodes)
                     if self.graph.node_to_phis_map.get(n)]

        # Routing instances: (node, resolved agent source) pairs -- purely a
        # routing/ordering concept (which real agent visits this node, and
        # in what order relative to its other nodes). A node may carry
        # several (distinct agent targets); object references never
        # contribute one (see _resolve_phi_agent_source). Discovered over
        # every node with phis at all -- NOT restricted to Formula-based
        # ones, since a routing instance can be established purely via
        # phi_to_variable_map/phi_to_static_assignment_map (e.g. the older
        # add_robot_linear_eq-style helpers, which register a static
        # assignment but no drake::symbolic Formula at all -- see
        # add_python_constraint/target_eq_constraint's use in
        # examples/pointmass_example.py).
        instance_list = []
        instance_local_id = {}
        node_instances = {}
        phi_instance = {}
        node_has_formula = set()
        for node in node_list:
            for phi_id in self.graph.node_to_phis_map.get(node, []):
                if phi_id in self.graph.phi_to_formula_map:
                    node_has_formula.add(node)
                src = self._resolve_phi_agent_source(phi_id)
                if src is None:
                    continue
                key = (node, src)
                if key not in instance_local_id:
                    instance_local_id[key] = len(instance_list)
                    instance_list.append(key)
                    node_instances.setdefault(node, []).append(instance_local_id[key])
                phi_instance[phi_id] = instance_local_id[key]

        # A node needs a dense row iff it has a routing instance (so
        # add_python_constraint / routing / write-back have somewhere to
        # read and write its position, even with zero Formula-based phis),
        # a Formula-based node phi (so object_q/mixed symbolic constraints
        # can address it, even with zero routing instances -- e.g. a purely
        # object_q-pinning node no agent ever visits), OR is an endpoint of
        # a Formula-based EDGE phi (edge_to_phis_map/edge_phi_to_formula_map
        # are independent of node_to_phis_map -- an edge constraint's
        # endpoint may carry no node-level phi of its own at all, e.g. a
        # pure transport-only relation). Spans the whole graph (see class
        # docstring) -- remaining_vertices no longer participates here at
        # all; a node that's since been passed keeps its dense row and
        # instance(s), just anchored to a known value at solve time instead
        # of dropped (mpc.py's AnchorState).
        edge_formula_nodes = set()
        for (u, v), phi_ids in self.graph.edge_to_phis_map.items():
            if any(phi_id in self.graph.edge_phi_to_formula_map for phi_id in phi_ids):
                edge_formula_nodes.add(u)
                edge_formula_nodes.add(v)
        self._node_list = sorted(set(node_instances) | node_has_formula | edge_formula_nodes)
        self._node_local_id = {g: i for i, g in enumerate(self._node_list)}
        self.n_dense_nodes = len(self._node_list)
        self.state_dim = self.graph.num_agents * self.graph.dim + self.graph.num_objects * self.graph.non_robot_dim

        self._instance_list = instance_list
        self._instance_local_id = instance_local_id
        self._node_instances = node_instances
        self._phi_instance = phi_instance

        # One slot per distinct assignable variable id actually referenced
        # here; statically-fixed instances consume no slot at all (their
        # real agent id is already known, no GA search needed) -- this can
        # legitimately leave n_variables at 0.
        var_ids = sorted({src[1] for _, src in instance_list if src[0] == "var"})
        var_id_to_slot = {v: i for i, v in enumerate(var_ids)}
        self._var_id_to_slot = var_id_to_slot
        self.n_variables = len(var_ids)

        self._instance_sources = [
            (kind, var_id_to_slot[val] if kind == "var" else val)
            for _, (kind, val) in instance_list
        ]
        self._instance_dense_node = [self._node_local_id[node] for node, _ in instance_list]

        # Per-dense-node representative routing instance, for
        # _batch_along_edge_interior_fn's betweenness mask: t (instance-
        # indexed) needs a per-NODE stand-in to compare against. First
        # instance registered at a node wins as its representative (see
        # _batch_along_edge_interior_fn's docstring); nodes with zero
        # instances (e.g. pure object_q-only nodes) never participate.
        node_repr_instance = np.zeros(self.n_dense_nodes, dtype=np.int32)
        node_has_instance = np.zeros(self.n_dense_nodes, dtype=bool)
        for inst_id, node_local in enumerate(self._instance_dense_node):
            if not node_has_instance[node_local]:
                node_repr_instance[node_local] = inst_id
                node_has_instance[node_local] = True
        self._node_repr_instance = jnp.asarray(node_repr_instance)
        self._node_has_instance = jnp.asarray(node_has_instance)

        # Hard edges: conditional edges never enter `structure` (see
        # add_conditional_edge_ordering's doc -- "invisible to BFS/routing"),
        # so every structure edge is an unconditional precedence constraint.
        # A node may carry several instances (distinct agent targets); the
        # edge is expanded to the full cross product of endpoint instances --
        # "everything at u finishes before anything at v begins".
        hard_edges = []
        for u in self._node_list:
            if u not in node_instances:
                continue
            for e in self.graph.structure.neighbors(u):
                v = e.to
                if v in node_instances:
                    for iu in node_instances[u]:
                        for iv in node_instances[v]:
                            hard_edges.append((iu, iv))
        self._hard_edges = hard_edges

        # Conditional ordering edges: compile each Formula into a gate_fn
        # closed over owner_variable/cond_binary index maps.
        var_sym_ids = {self.graph.assignment_sym(v).get_id(): var_id_to_slot[v] for v in var_ids}
        cond_binary_vars = self.graph.binary_cond_sym_vars
        cond_sym_ids = {v.get_id(): i for i, v in enumerate(cond_binary_vars)}
        self.n_cond_vars = len(cond_binary_vars)

        cond_edges = []
        for (u, v), formula in self.graph.conditional_ordering_map.items():
            if u not in node_instances or v not in node_instances:
                continue
            gate = compile_condition(formula, var_sym_ids, cond_sym_ids)
            for iu in node_instances[u]:
                for iv in node_instances[v]:
                    cond_edges.append((iu, iv, gate))
        self._cond_edges = cond_edges

        self._resolver = _make_dense_resolver(self.graph, var_id_to_slot)
        self._node_objects = {}  # node -> {object_id, ...} referenced there (write-back bookkeeping only)
        self._resolve_symbolic_constraints()

    def _resolve_symbolic_constraints(self):
        """Auto-derives eq/ineq residual constraints from the graph's unified
        symbolic API (add_constraint / add_assignable_constraint /
        add_edge_constraint), compiling each stored Formula the same way MILP
        does structurally, just against this solver's dense per-node row
        layout instead of Drake decision variables. Raises (via
        _make_dense_resolver, through _unsupported_placeholder) if a formula
        references a placeholder genuinely outside that scope (e.g. a u-/
        v-side placeholder inside a node constraint). An edge constraint's
        stored formula compiles differently depending on
        graph.edge_phi_to_along_edge_map: relationally (once, over both
        endpoint rows) or "along the edge" (once, applied independently to
        each endpoint's own row) -- see the branch below."""
        node_formulas = self.graph.phi_to_formula_map
        for node in self._node_list:
            node_local = self._node_local_id[node]
            for phi_id in self.graph.node_to_phis_map.get(node, []):
                if phi_id not in node_formulas:
                    continue  # not a Formula-based (symbolic) constraint
                formula = node_formulas[phi_id]
                resolver = lambda var, self=self: self._resolver(var, 1)
                fn, kind = compile_relational_formula(formula, resolver)
                self._symbolic_constraints.append(
                    ((node_local,), fn, kind, f"phi_{phi_id}"))
                for oid in _object_ids_referenced(formula, self.graph, self.graph.object_q):
                    self._node_objects.setdefault(node, set()).add(oid)

        edge_formulas = self.graph.edge_phi_to_formula_map
        edge_along_edge = self.graph.edge_phi_to_along_edge_map
        for (u, v), phi_ids in self.graph.edge_to_phis_map.items():
            # _node_list/_node_local_id are graph-global (see class
            # docstring) and already include both endpoints of any edge that
            # carries a Formula-based phi (edge_formula_nodes, above) -- so
            # this only ever skips an edge whose phis are ALL non-Formula
            # (e.g. purely a conditional-ordering edge, handled separately
            # via _cond_edges), never one dropped due to remaining_vertices.
            if u not in self._node_local_id or v not in self._node_local_id:
                continue
            u_local, v_local = self._node_local_id[u], self._node_local_id[v]
            for phi_id in phi_ids:
                if phi_id not in edge_formulas:
                    continue  # not a Formula-based (symbolic) edge constraint
                formula = edge_formulas[phi_id]

                if edge_along_edge.get(phi_id, False):
                    # "Along the edge" -- built from the plain agent_q/
                    # object_q/var_agent_q placeholders (same as a node
                    # constraint), so compile it ONCE against a single-slot
                    # (node-scoped) resolver, then register the resulting fn
                    # at each endpoint's own dense row (exact, its own
                    # persistent AL multiplier each), PLUS a best-effort
                    # aggregate re-application at any OTHER dense node whose
                    # `t` currently falls between the endpoints' -- see
                    # _batch_along_edge_interior_fn.
                    resolver = lambda var, self=self: self._resolver(var, 1)
                    fn, kind = compile_relational_formula(formula, resolver)
                    self._symbolic_constraints.append(
                        ((u_local,), fn, kind, f"edge_phi_{phi_id}_u"))
                    self._symbolic_constraints.append(
                        ((v_local,), fn, kind, f"edge_phi_{phi_id}_v"))
                    interior_batched = _batch_along_edge_interior_fn(
                        fn, kind, u_local, v_local,
                        self._node_repr_instance, self._node_has_instance)
                    self._interior_constraints.append(
                        (interior_batched, f"edge_phi_{phi_id}_interior"))
                    for oid in _object_ids_referenced(formula, self.graph, self.graph.object_q):
                        self._node_objects.setdefault(u, set()).add(oid)
                        self._node_objects.setdefault(v, set()).add(oid)
                    continue

                resolver = lambda var, self=self: self._resolver(var, 2)
                fn, kind = compile_relational_formula(formula, resolver)
                self._symbolic_constraints.append(
                    ((u_local, v_local), fn, kind, f"edge_phi_{phi_id}"))
                for oid in _object_ids_referenced(formula, self.graph, self.graph.u_object_q):
                    self._node_objects.setdefault(u, set()).add(oid)
                for oid in _object_ids_referenced(formula, self.graph, self.graph.v_object_q):
                    self._node_objects.setdefault(v, set()).add(oid)

    # -- constraints -------------------------------------------------------

    def add_python_constraint(self, node, fn, kind="eq", name=None):
        """Registers a single-instance constraint fn(wp_row) -> (k,) on
        `node`'s waypoint. kind="eq" feeds pymoo/AL's H (feasible at 0),
        kind="ineq" feeds G (feasible at <=0)."""
        instances = self._node_instances.get(node, [])
        if not instances:
            raise ValueError(f"node {node} has no resolvable agent source "
                              "(no phi registered establishing a routing instance there)")
        if len(instances) > 1:
            raise ValueError(
                f"node {node} has {len(instances)} distinct agent-source instances -- "
                "add_python_constraint can't disambiguate which waypoint row to target; "
                "use the graph's symbolic add_constraint/add_assignable_constraint API instead")
        if kind not in ("eq", "ineq"):
            raise ValueError(f"Unknown constraint kind {kind!r}, expected 'eq' or 'ineq'")
        self._python_constraints.append((instances[0], fn, kind, name))

    # -- build ---------------------------------------------------------

    def build_problem(self):
        ordering_edges = [(u, v, None) for u, v in self._hard_edges]
        ordering_edges += list(self._cond_edges)

        eq_constraints, ineq_constraints = [], []
        for instance_local, fn, kind, _name in self._python_constraints:
            node, source = self._instance_list[instance_local]
            dense_node = self._node_local_id[node]
            src_kind, val = source
            batched = _batch_python_constraint_fn(fn, dense_node, src_kind, val, self.graph.dim, self.n_variables)
            (eq_constraints if kind == "eq" else ineq_constraints).append(batched)
        for node_locals, fn, kind, _name in self._symbolic_constraints:
            batched = _batch_symbolic_constraint_fn(fn, node_locals)
            (eq_constraints if kind == "eq" else ineq_constraints).append(batched)
        for batched, _name in self._interior_constraints:
            ineq_constraints.append(batched)

        return GraphOrderingRelaxed(
            instance_sources=self._instance_sources,
            n_variables=self.n_variables,
            ordering_edges=ordering_edges,
            x0=np.asarray(self.x0),
            wp_bounds=self.wp_bounds,
            instance_dense_node=self._instance_dense_node,
            n_dense_nodes=self.n_dense_nodes,
            state_dim=self.state_dim,
            n_cond_vars=self.n_cond_vars,
            objective=self.objective,
            penalty_weight=self.penalty_weight,
            edge_cost_fn=self.edge_cost_fn,
            eq_constraints=eq_constraints,
            ineq_constraints=ineq_constraints,
        )

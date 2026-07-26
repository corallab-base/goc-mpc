"""JAX-native graph-ordering kernel: port of
examples/new_p1_experiments/jax_waypoint_eval.py's make_graph_kernel /
_masked_sort_perm / _one_agent_route (the general graph-ordering kernel only --
the older task-only decode_and_cost/p1_pymoo-specific kernel stays in
examples/, unused here).

Generalization over the ported original: hard_edges/cond_pairs (two separate,
narrower registries) collapse into one `ordering_edges` list of
(u, v, gate_fn) instance-pairs, where `gate_fn(owner_variable, cond_binary) ->
bool` decides whether that edge's "u must arrive no later than v" penalty is
currently active. A plain hard edge is `gate_fn=None` (always active). This is
what lets a GraphOfConstraints' _conditional_ordering_map -- arbitrary
compiled Formula gates, not just a fixed "same real agent" block-ordering
special case -- plug in directly (see formula_compiler.compile_condition),
while remaining a strict superset of the old always-true hard-edge case.

Second generalization: the routing/ordering unit is a "(node, agent source)
instance" (see spec.GraphOrderingSpec's module docstring), not a bare graph
node -- a node can carry several instances when multiple, independently-
resolved agents are constrained there. Each instance's agent source is either
`("fixed", agent_id)` (known, no GA search needed) or `("var", var_slot)`
(GA-searched via the `assign` one-hot block, one row per distinct real
`variable`). `n_variables` -- the number of GA-searched rows -- can
legitimately be 0 when every instance is statically fixed.

Third generalization: the wp DECISION VARIABLE is no longer one row per
routing instance. It's one DENSE row per graph node -- `wp_dense[node] =
[agent_0 | agent_1 | ... | agent_{n_agents-1} | object_0 | ... |
object_{n_objects-1}]`, mirroring MILP's per-node joint configuration `W`
(see graph_of_constraints.cpp:79's `total_dim`) -- so symbolic node/edge
constraints referencing agent_q(k)/object_q(k)/var_agent_q(var) (see
spec.py's _make_dense_resolver) can address any agent's or object's position
at a node via a plain column offset, with no per-formula "which sparse row"
bookkeeping. Routing/ordering, however, still only cares about each routing
instance's OWN agent slice -- `decode_and_cost` below gathers that
`dim`-wide slice out of each instance's node row (`instance_dense_node`)
before handing off to the unchanged _one_agent_route/_masked_sort_perm
routing math. A "fixed" instance's slice offset (`agent_id * dim`) is static;
a "var" instance's is dynamic (`owner_variable[var_slot] * dim`, GA-decided).
Object-only node rows (no agent instance references them) are simply never
gathered by anything here -- they incur zero routing cost and need no
sentinel owner, unlike agent instances which are always exactly one real
agent's responsibility.

See jax_waypoint_eval.py's module docstring for the underlying masked-sort /
real-arrival-time design this reuses verbatim.

Fourth generalization: `wp_dense`/`n_dense_nodes` now span the WHOLE graph,
not just whatever's left in the current MPC horizon (see spec.py's class
docstring) -- remaining_vertices membership is instead threaded in as the
`node_active` runtime argument of `decode_and_cost`/`batched`, exactly
parallel to `x0` (see make_graph_kernel's docstring), so an already-passed
node's routing instance is excluded from route cost/arrival time and its
ordering edges are gated to zero, without changing this kernel's fixed
shape or forcing a retrace when a node is passed.
"""

import functools

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def _euclidean_edge_cost(a, b):
    return jnp.linalg.norm(b - a)


def _masked_sort_perm(agent_id, owner, order, active):
    """`active` (n,) bool -- excludes instances no longer in
    remaining_vertices (already passed) from this agent's route/arrival-time
    computation entirely, the same way an instance owned by a DIFFERENT
    agent is already excluded via `owner == agent_id` -- a passed instance
    contributes no route cost and gets arrival_by_node == 0 (see
    _one_agent_route), regardless of what value sits in its wp row."""
    n = owner.shape[0]
    include = (owner == agent_id) & active
    key = jnp.where(include, order, jnp.inf)
    perm = jnp.argsort(key, stable=True)
    count = jnp.sum(include)
    mask = jnp.arange(n) < count
    return perm, mask


def _one_agent_route(agent_id, owner, order, wp, depot, active, edge_cost_fn=_euclidean_edge_cost):
    """Returns (cost, arrival_by_node) for one agent -- see
    jax_waypoint_eval._one_agent_route's docstring for the full derivation."""
    n = owner.shape[0]
    perm, mask = _masked_sort_perm(agent_id, owner, order, active)
    wp_sorted = wp[perm]
    seq = jnp.concatenate([depot[None, :], wp_sorted], axis=0)
    seg_dist = jax.vmap(edge_cost_fn)(seq[:-1], seq[1:])
    cost = jnp.sum(jnp.where(mask, seg_dist, 0.0))

    cum_time_sorted = jnp.cumsum(seg_dist)
    arrival_by_node = jnp.zeros(n, dtype=seg_dist.dtype).at[perm].set(
        jnp.where(mask, cum_time_sorted, 0.0))
    return cost, arrival_by_node


def make_graph_kernel(instance_sources, n_variables, ordering_edges, instance_dense_node, dim,
                       objective="avg", penalty_weight=20.0, margin=0.05, edge_cost_fn=_euclidean_edge_cost):
    """Factory: builds (decode_and_cost, batched) closed over one problem
    instance's fixed structure.

    instance_sources: (n_instances,) list of ("fixed", agent_id) or
        ("var", var_slot) -- one entry per routing instance (see spec.py's
        module docstring for "instance" = (node, resolved agent source)).
        var_slot indexes into `assign`'s rows (0..n_variables-1).
    n_variables: number of GA-searched rows in `assign` -- may be 0 when
        every instance is statically fixed (nothing for the GA to search).
    instance_dense_node: (n_instances,) int array -- which row of the dense
        per-node wp tensor (see module docstring) each routing instance's
        position is gathered from.
    dim: per-agent state width (the gathered slice width -- a strict
        sub-width of the dense row's own `state_dim`, which also carries
        object columns this routing math never looks at).
    ordering_edges: iterable of (u, v, gate_fn), instance u must arrive no
        later than instance v IN REAL TIME whenever gate_fn(owner_variable,
        cond_binary) holds (or unconditionally if gate_fn is None) --
        penalized as relu(arrival_time[u] - arrival_time[v] + margin), same
        margin reasoning as jax_waypoint_eval.make_graph_kernel.

    Deliberately does NOT close over x0 (n_agents, dim depots): x0 is a
    per-call runtime value (the MPC depot drifts every tick) while
    everything else here is fixed problem structure, so x0 is instead a
    genuine argument of decode_and_cost/batched below -- that's what lets a
    compiled `batched` (and everything built on top of it in solver.py) stay
    valid across depot drift without retracing, as long as this structure
    doesn't change.

    Same treatment for `node_active` (n_dense_nodes,): remaining_vertices
    membership is also a per-call runtime value (problem.AnchorState, built
    fresh each solve() from remaining_vertices -- see mpc.py), not fixed
    structure, so it's threaded as a genuine argument of decode_and_cost/
    batched too rather than baked in here -- this is what lets a node
    dropping out of remaining_vertices (an already-passed node) change from
    "included in this agent's route" to "excluded" without retracing:
    `instance_active = node_active[instance_dense_node]` gathers it down to
    per-routing-instance granularity and excludes inactive instances from
    _one_agent_route's arrival-time/cost computation and from ordering-edge
    violations, exactly the way an instance owned by a different agent is
    already excluded -- see _masked_sort_perm. Callers should pass wp_dense
    as the anchor-substituted `wp_eff` (problem.apply_anchor) for
    consistency with the constraint-evaluation call sites in solver.py,
    though decode_and_cost's own routing math never reads an inactive
    instance's row anyway (masked out regardless of its value).
    """
    instance_sources = list(instance_sources)
    n_instances = len(instance_sources)
    instance_is_fixed = jnp.asarray([kind == "fixed" for kind, _ in instance_sources])
    instance_fixed_agent = jnp.asarray(
        [val if kind == "fixed" else 0 for kind, val in instance_sources], dtype=jnp.int32)
    instance_var_slot = jnp.asarray(
        [val if kind == "var" else 0 for kind, val in instance_sources], dtype=jnp.int32)
    instance_dense_node = jnp.asarray(instance_dense_node, dtype=jnp.int32)

    ordering_edges = list(ordering_edges)
    edge_u = jnp.array([u for u, v, _ in ordering_edges], dtype=jnp.int32) \
        if ordering_edges else jnp.zeros((0,), dtype=jnp.int32)
    edge_v = jnp.array([v for u, v, _ in ordering_edges], dtype=jnp.int32) \
        if ordering_edges else jnp.zeros((0,), dtype=jnp.int32)
    # gate_fn=None -> always-active; represented as a constant-True gate so
    # every edge can be evaluated uniformly below.
    gate_fns = [gate if gate is not None else (lambda ov, cb: jnp.asarray(True))
                for _, _, gate in ordering_edges]

    def decode_and_cost(assign, cond_binary, t, wp_dense, x0, node_active):
        """
        assign: (n_variables, n_agents) -- one-hot Binary (mixed) or
                continuous relaxed scores (relaxed). May be empty
                (n_variables==0) when every instance is statically fixed.
                Callers should pass the anchor-substituted `assign_eff`
                (problem.apply_anchor) so a variable committed by an
                already-passed instance routes to its pinned agent.
        cond_binary: (n_cond,) -- continuous relaxed scores in [0, 1] for each
                free binary conditional-ordering switch (GraphOfConstraints'
                binary_cond_sym_vars); GA-decided, same as `assign`.
        t: (n_instances,) -- ordering/time score per instance.
        wp_dense: (n_dense_nodes, state_dim) -- dense per-node configuration
                (see module docstring); routing gathers each instance's own
                dim-wide agent slice out of it below.
        x0: (n_agents, dim) -- depots; a runtime argument (see make_graph_kernel's
                docstring), not baked into this closure.
        node_active: (n_dense_nodes,) bool -- remaining_vertices membership,
                a runtime argument for the same reason x0 is (see
                make_graph_kernel's docstring).
        """
        n_agents = x0.shape[0]
        instance_active = node_active[instance_dense_node]              # (n_instances,)

        # n_variables is a static Python int closed over at kernel-build
        # time (not a traced value) -- branching on it here is safe and is
        # what avoids ever gathering into a possibly-empty `assign`-derived
        # array (an empty `assign` has no well-defined index dtype).
        if n_variables > 0:
            owner_variable = jnp.argmax(assign, axis=-1)                # (n_variables,)
            var_owner_for_instance = owner_variable[instance_var_slot]  # (n_instances,)
            owner_instance = jnp.where(instance_is_fixed, instance_fixed_agent, var_owner_for_instance)
        else:
            owner_variable = jnp.zeros((0,), dtype=jnp.int32)
            owner_instance = instance_fixed_agent

        # Gather each routing instance's dim-wide position slice out of its
        # node's dense row -- the column is owner_instance*dim (a "fixed"
        # instance's column is static in spirit but computed the same way
        # here since owner_instance already equals instance_fixed_agent for
        # those; a "var" instance's column is genuinely dynamic, decided by
        # this population member's own `assign`).
        node_rows = wp_dense[instance_dense_node]                      # (n_instances, state_dim)
        col = owner_instance * dim                                     # (n_instances,)
        gather_idx = col[:, None] + jnp.arange(dim)[None, :]           # (n_instances, dim)
        wp = jnp.take_along_axis(node_rows, gather_idx, axis=1)        # (n_instances, dim)

        agent_ids = jnp.arange(n_agents)
        per_agent_costs, per_agent_arrival = jax.vmap(
            functools.partial(_one_agent_route, edge_cost_fn=edge_cost_fn),
            in_axes=(0, None, None, None, 0, None)
        )(agent_ids, owner_instance, t, wp, x0, instance_active)
        route_cost = jnp.max(per_agent_costs) if objective == "minmax" else jnp.mean(per_agent_costs)

        # Real, cross-agent-comparable arrival time per instance (see
        # jax_waypoint_eval.make_graph_kernel's docstring for why summing,
        # not picking, is correct: owner_instance partitions the instances).
        arrival_time = jnp.sum(per_agent_arrival, axis=0)   # (n_instances,)

        if ordering_edges:
            gate_vals = jnp.stack([g(owner_variable, cond_binary) for g in gate_fns]).astype(t.dtype)
            # Ordering a node against one that's already passed is moot --
            # unlike value constraints, precedence edges are correctly
            # dropped (not anchored) once either endpoint leaves
            # remaining_vertices, composing with the existing conditional
            # gate via plain multiplication.
            edge_active = (instance_active[edge_u] & instance_active[edge_v]).astype(t.dtype)
            raw_violation = jax.nn.relu(arrival_time[edge_u] - arrival_time[edge_v] + margin)
            ordering_violation = jnp.sum(gate_vals * edge_active * raw_violation)
        else:
            ordering_violation = jnp.asarray(0.0, dtype=t.dtype)

        g = jnp.abs(jnp.sum(assign, axis=-1) - 1.0) - 1e-6   # (n_variables,)
        F = route_cost + penalty_weight * ordering_violation
        return F, g

    batched = jax.jit(jax.vmap(decode_and_cost, in_axes=(0, 0, 0, 0, None, None)))
    return decode_and_cost, batched

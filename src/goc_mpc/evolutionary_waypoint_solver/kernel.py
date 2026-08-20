"""JAX-native graph-ordering kernel: builds a jitted (decode_and_cost, batched)
pair closed over one problem's fixed structure (assignment/ordering-edge/
routing-instance layout), used to score a population of candidate decision
vectors under the Augmented-Lagrangian GA (solver.py).

Decision variables decode_and_cost consumes:
  assign: (n_variables, n_agents) one-hot (or continuous-relaxed) -- which
      real agent each assignable routing variable resolves to.
  cond_binary: (n_cond,) continuous-relaxed [0, 1] scores -- GraphOfConstraints'
      binary_cond_sym_vars, gating conditional ordering edges.
  t: (n_nodes,) -- one ordering-priority score per graph node (not per
      routing instance -- see "Routing instances" below). Purely a
      tie-breaking priority, not an arrival time -- see "Ordering edges"
      below for how it actually determines visiting order.
  wp: (n_nodes, state_dim) -- per-node configuration,
      `[agent_0 | agent_1 | ... | object_0 | ...]`, mirroring MILP's per-node
      joint configuration `W` (graph_of_constraints.cpp:79's `total_dim`).
`x0` (n_agents, dim depots) and `node_active` (n_nodes,) are genuine
runtime arguments of decode_and_cost/batched rather than values closed over
at kernel-build time: `x0` drifts every MPC tick and `node_active` changes
whenever a node leaves remaining_vertices, so keeping both as arguments lets
one compiled `batched` stay valid across both without retracing, as long as
the problem's structure (shapes, edges, constraints) is unchanged.

Routing instances
------------------
A routing instance is a (node, resolved agent source) pair -- `("fixed",
agent_id)` (a literal agent_q(k) constraint) or `("var", var_slot)` (an
assignable var_agent_q(var) constraint), one entry per distinct agent source
a node's phis establish (see spec.build_graph_ordering_problem). A
node can carry several instances when more than one agent is independently
constrained there (e.g. a shared start/rendezvous node). Instances exist
solely to partition nodes into "which agent must visit this node" for each
agent's own route-cost computation (_one_agent_route below) -- they carry no
ordering state of their own; visiting order is a per-NODE quantity
(node_rank, see below), gathered down to instance granularity
(`node_rank[instance_node]`) only for the duration of each agent's own
masked-sort route computation.

Ordering edges
---------------
`ordering_edges` (u, v, gate_fn) are NODE pairs: node u must be visited no
later than node v whenever `gate_fn(owner_variable, cond_binary)` holds (or
unconditionally if `gate_fn` is None).

This precedence is enforced BY CONSTRUCTION, not by a penalty term a GA
search has to discover on its own: `_topological_rank` decodes the raw
per-node priority vector `t` into `node_rank`, a per-node visiting-order
rank, via a priority-based topological sort (a standard technique in the
scheduling-GA literature -- e.g. the "activity list + serial schedule
generation scheme" used for RCPSP GAs, or "random-key" DAG-constrained
sequencing) -- greedily, at each step, picking the smallest-`t` node among
those whose every (gated-active) predecessor has already been scheduled.
Every real-valued `t` decodes into SOME topologically valid order this way,
so the GA's search space is naturally restricted to precedence-respecting
schedules; there is no way for a decoded order to violate `ordering_edges`,
so no penalty/constraint-violation bookkeeping is needed for it at all.
(An earlier version of this kernel folded a `relu(node_arrival_time[u] -
node_arrival_time[v] + margin)` term into the cost with a fixed, non-adaptive
weight -- unlike a true Augmented-Lagrangian constraint, that weight never
grew, so the GA could and did trade away correctness for a shorter route
whenever that was cheaper than the fixed penalty; worse, since it was folded
into the objective rather than reported as a constraint, the solver's own
CV -- constraint violation -- metric stayed near zero even when precedence
was being violated, giving false confidence. Both problems are structural
to any fixed-weight penalty scheme, not a tuning issue -- hence the
by-construction decoder instead.)
"""

import functools

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def _euclidean_edge_cost(a, b):
    return jnp.linalg.norm(b - a)


def _topological_rank(t, node_active, edge_u, edge_v, edge_active, n_nodes):
    """Decodes per-node priorities `t` into a topologically valid per-node
    visiting rank (0-indexed, one distinct value per node), respecting the
    directed precedence edges `edge_u[i] -> edge_v[i]` selected by
    `edge_active[i]` (already folded with that edge's conditional gate and
    both endpoints' node_active -- see decode_node_rank below). Ties among
    nodes that are simultaneously ready to schedule are broken by `t`
    (smaller first) -- see module docstring for the general technique.

    A node not in `node_active` (already passed) is pre-marked scheduled: it
    can neither block anything, and never occupies a slot in the returned
    order among currently-active nodes (every consumer of THIS decoder's
    original use -- kernel.py's own routing -- already masks passed
    instances out via instance_active regardless of rank value, matching the
    older scheme's "ordering a node against an already-passed one is moot").
    Its returned rank is nonetheless well-defined, not leftover scatter-init
    garbage: -1, uniformly, for every passed node -- guaranteed less than
    every active node's real rank (0..num_active-1) -- rather than the
    ambiguous 0 an unpatched scatter would leave it at, which -- for a
    caller that (unlike routing) DOES compare a passed node's rank directly,
    e.g. spec.py's decode-rank-gated interior/stationary constraints reading
    a hold endpoint that has since left remaining_vertices -- would
    otherwise collide with whichever active node legitimately earns rank 0,
    corrupting any betweenness/overlap test involving it (verified directly:
    an anchored hold's own edge, endpoints (passed, active), stopped being
    recognized as self-overlapping once its passed endpoint's rank
    coincided with its active endpoint's).

    Runs exactly `n_nodes` greedy steps (a jax.lax.scan, static length, so
    this stays jit/vmap-compatible) regardless of how many nodes actually
    still need scheduling: once every remaining node has been scheduled, the
    remaining steps have no real candidate left and are made inert (their
    scan output is an out-of-bounds sentinel node id, dropped by the final
    scatter's `mode="drop"`) rather than being allowed to arbitrarily
    re-pick and corrupt an already-assigned node's rank.

    A cycle among the currently-active edges (only possible via conditional
    gates disagreeing with each other for a given individual -- `ordering_edges`
    is otherwise a real DAG) would otherwise stall every remaining node as
    permanently "not ready"; guarded against by falling back to ignoring the
    blocked filter (picking purely by `t`) whenever no node is ready but
    nodes remain, so this always terminates with every active node ranked,
    never NaNs or an infinite loop.
    """
    adj = jnp.zeros((n_nodes, n_nodes), dtype=bool).at[edge_u, edge_v].set(
        edge_active, mode="drop")

    def step(scheduled, _):
        blocked = jnp.any(adj & (~scheduled)[:, None], axis=0)
        ready = (~scheduled) & (~blocked)
        any_ready = jnp.any(ready)
        candidate = jnp.where(any_ready, ready, ~scheduled)
        productive = jnp.any(candidate)
        key = jnp.where(candidate, t, jnp.inf)
        chosen = jnp.argmin(key)
        scheduled_next = jnp.where(productive, scheduled.at[chosen].set(True), scheduled)
        out_node = jnp.where(productive, chosen, n_nodes)  # n_nodes: OOB sentinel, dropped below
        return scheduled_next, out_node

    init_scheduled = ~node_active
    _, order = jax.lax.scan(step, init_scheduled, xs=None, length=n_nodes)
    active_rank = jnp.zeros((n_nodes,), dtype=jnp.int32).at[order].set(
        jnp.arange(n_nodes, dtype=jnp.int32), mode="drop")
    return jnp.where(node_active, active_rank, -1)


def build_decode_node_rank(ordering_edges, n_nodes):
    """Builds decode_node_rank(owner_variable, cond_binary, t, node_active)
    -> node_rank straight from a plain (u, v, gate_fn) ordering-edge list and
    node count -- exactly the construction make_graph_kernel uses internally
    for its own routing decode (edge_u/edge_v/gate_fns -> _topological_rank),
    factored out so any OTHER caller needing the same exact topologically
    valid per-node order can build one without a full routing kernel around
    it. In particular: spec.py's gated edge constraints (hold rigidity,
    stationary-object, and along-edge interior reinforcement) need this to
    decide which OTHER nodes/edges a given individual's schedule currently
    places between two reference nodes -- raw `t` alone is NOT usable for
    that (see make_graph_kernel's module docstring: it's a pure GA-searched
    sort key, under no pressure to stay magnitude-monotonic with true
    visiting order once decode fixes that order; empirically verified to
    drift, e.g. a plain 3-node chain n0->n1->n2 producing raw t=[1, 2, 0]).
    `node_rank` is the real thing to gate on: an exact, per-individual-valid
    permutation, so a betweenness/overlap test built on it is correct for
    that individual's actual (possibly still-evolving) schedule, including
    genuinely concurrent/unordered branches where the relative order itself
    is part of what the GA is searching over -- not just an approximation of
    the deterministic/nested case."""
    ordering_edges = list(ordering_edges)
    edge_u = jnp.array([u for u, v, _ in ordering_edges], dtype=jnp.int32) \
        if ordering_edges else jnp.zeros((0,), dtype=jnp.int32)
    edge_v = jnp.array([v for u, v, _ in ordering_edges], dtype=jnp.int32) \
        if ordering_edges else jnp.zeros((0,), dtype=jnp.int32)
    gate_fns = [gate if gate is not None else (lambda ov, cb: jnp.asarray(True))
                for _, _, gate in ordering_edges]

    def decode_node_rank(owner_variable, cond_binary, t, node_active):
        if ordering_edges:
            gate_vals = jnp.stack([g(owner_variable, cond_binary) for g in gate_fns]).astype(t.dtype)
            edge_active = (gate_vals > 0.5) & node_active[edge_u] & node_active[edge_v]
        else:
            edge_active = jnp.zeros((0,), dtype=bool)
        return _topological_rank(t, node_active, edge_u, edge_v, edge_active, n_nodes)

    return decode_node_rank


def _masked_sort_perm(agent_id, owner, order, active):
    """`active` (n,) bool -- excludes instances no longer in
    remaining_vertices (already passed) from this agent's route computation
    entirely, the same way an instance owned by a DIFFERENT agent is already
    excluded via `owner == agent_id` -- a passed instance contributes no
    route cost and gets arrival_by_node == 0 (see _one_agent_route),
    regardless of what value sits in its wp row. `order` is expected to
    already be a topologically valid rank (node_rank gathered to instance
    granularity, see decode_node_rank) -- ties are impossible by
    construction, so this plain argsort just recovers each agent's own
    subsequence of that already-valid global order."""
    n = owner.shape[0]
    include = (owner == agent_id) & active
    key = jnp.where(include, order, jnp.inf)
    perm = jnp.argsort(key, stable=True)
    count = jnp.sum(include)
    mask = jnp.arange(n) < count
    return perm, mask


def _one_agent_route(agent_id, owner, order, wp, depot, active, edge_cost_fn=_euclidean_edge_cost):
    """Returns (cost, arrival_by_node) for one agent: `cost` is the total
    depot-to-first-stop-to-...-to-last-stop distance over this agent's own
    instances (sorted by `order`, masked to just this agent's active ones);
    `arrival_by_node` is the cumulative distance at each of this agent's own
    instances (0 for any instance excluded by `active`/`owner`), aligned back
    to `owner`'s original (unsorted) index order."""
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


def make_graph_kernel(instance_sources, n_variables, ordering_edges, instance_node, dim,
                       n_nodes, objective="avg", edge_cost_fn=_euclidean_edge_cost):
    """Factory: builds (decode_and_cost, batched, decode_node_rank) closed
    over one problem instance's fixed structure -- see module docstring for
    the decision variables/routing-instance/ordering-edge design this
    implements.

    instance_sources: (n_instances,) list of ("fixed", agent_id) or
        ("var", var_slot) -- one entry per routing instance. var_slot
        indexes into `assign`'s rows (0..n_variables-1).
    n_variables: number of GA-searched rows in `assign` -- may be 0 when
        every instance is statically fixed (nothing for the GA to search).
    instance_node: (n_instances,) int array -- which row of the
        per-node `wp` tensor each routing instance's position is
        gathered from, and which entry of `t`/node_rank its order is
        gathered from.
    dim: per-agent state width (the gathered slice width -- a strict
        sub-width of a waypoint row's own `state_dim`, which also carries
        object columns this routing math never looks at).
    n_nodes: total graph-node count `t`/`wp` are indexed over.
    ordering_edges: iterable of (u, v, gate_fn) node pairs -- see module
        docstring.

    `decode_node_rank(owner_variable, cond_binary, t, node_active) ->
    node_rank` is exposed (not just used internally by decode_and_cost) so a
    caller holding a single already-solved individual (mpc.py's solve(), on
    its GA's best_X) can recover the SAME topologically valid per-node order
    the route-cost computation actually used, for reporting via
    view_t_by_node() -- reporting the raw `t` priority there instead would
    reintroduce exactly the bug this decoder exists to fix, one level up
    (GraphOfConstraints.get_agent_paths sorts nodes by t_by_node too).
    """
    instance_sources = list(instance_sources)
    instance_is_fixed = jnp.asarray([kind == "fixed" for kind, _ in instance_sources])
    instance_fixed_agent = jnp.asarray(
        [val if kind == "fixed" else 0 for kind, val in instance_sources], dtype=jnp.int32)
    instance_var_slot = jnp.asarray(
        [val if kind == "var" else 0 for kind, val in instance_sources], dtype=jnp.int32)
    instance_node = jnp.asarray(instance_node, dtype=jnp.int32)

    decode_node_rank = build_decode_node_rank(ordering_edges, n_nodes)

    def decode_and_cost(assign, cond_binary, t, wp, x0, node_active):
        n_agents = x0.shape[0]
        instance_active = node_active[instance_node]              # (n_instances,)

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

        node_rank = decode_node_rank(owner_variable, cond_binary, t, node_active)
        rank_per_instance = node_rank[instance_node].astype(t.dtype)   # (n_instances,)

        # Gather each routing instance's dim-wide position slice out of its
        # node's row -- the column is owner_instance*dim (a "fixed"
        # instance's column is static in spirit but computed the same way
        # here since owner_instance already equals instance_fixed_agent for
        # those; a "var" instance's column is genuinely dynamic, decided by
        # this population member's own `assign`).
        node_rows = wp[instance_node]                                  # (n_instances, state_dim)
        col = owner_instance * dim                                     # (n_instances,)
        gather_idx = col[:, None] + jnp.arange(dim)[None, :]           # (n_instances, dim)
        wp = jnp.take_along_axis(node_rows, gather_idx, axis=1)        # (n_instances, dim)

        agent_ids = jnp.arange(n_agents)
        per_agent_costs, _per_agent_arrival = jax.vmap(
            functools.partial(_one_agent_route, edge_cost_fn=edge_cost_fn),
            in_axes=(0, None, None, None, 0, None)
        )(agent_ids, owner_instance, rank_per_instance, wp, x0, instance_active)
        route_cost = jnp.max(per_agent_costs) if objective == "minmax" else jnp.mean(per_agent_costs)

        g = jnp.abs(jnp.sum(assign, axis=-1) - 1.0) - 1e-6   # (n_variables,)
        return route_cost, g

    batched = jax.jit(jax.vmap(decode_and_cost, in_axes=(0, 0, 0, 0, None, None)))
    return decode_and_cost, batched, jax.jit(decode_node_rank)

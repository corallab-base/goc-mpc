"""Derives a CP-SAT-ready, solver-agnostic view of a graph's routing/
precedence/discrete-candidate structure from an already-built
GraphOrderingRelaxed (evolutionary_waypoint_solver.spec.build_graph_ordering_
problem), reusing its one-time parse of the real GraphOfConstraints instead
of re-deriving node/edge/projection structure a second time. Nothing here
touches GraphOrderingRelaxed's own flat-vector/kernel machinery (that stays
GA-specific) -- this module only reads problem.hard_edges, problem.
instance_node/instance_sources, and problem.projections, and reuses problem.
apply_projections/apply_anchor to materialize concrete wp rows.

Call order for a fresh solve:
  1. warm_start_wp(problem, x0) -- fixed/interpolated wp template, no
     projections applied yet.
  2. node_candidates(problem, wp_template, params) -- per node, the finite
     set of concrete candidate rows a CP-SAT model can pick between.
  3. build_edge_cost_table(problem, candidates, wp_template, edge_cost_fn,
     pairs) / build_depot_cost_table(candidates, wp_template, x0_row,
     nodes, edge_cost_fn) -- numeric cost tables CP-SAT's per-agent
     circuit arcs are looked up from, so the discrete model never touches
     wp/edge_cost_fn directly. See cpsat_model.py for which pairs a
     per-agent routing circuit actually needs (all pairs among that
     agent's own nodes, NOT just problem.hard_edges).
"""

from collections import namedtuple
import weakref

import jax
import jax.numpy as jnp
import numpy as np
import pydrake.symbolic as sym

from ..evolutionary_waypoint_solver.problem import jit_apply_projections
from ..evolutionary_waypoint_solver.kernel import _euclidean_edge_cost
from ..evolutionary_waypoint_solver.formula_compiler import as_variable


# One node's finite candidate set for the discrete solve.
#   rows: (n_candidates, state_dim) numpy -- concrete, fully-resolved wp
#       rows (only meaningful where this node has no free/continuous
#       remainder; see node_candidates' docstring for the self-contained-
#       projection-only scope of this first cut).
#   branch_ids: (n_candidates,) int, or None for a node with no
#       projection (a single implicit candidate: wp_template[node] as-is).
NodeCandidates = namedtuple("NodeCandidates", ["node", "rows", "branch_ids"])


def node_instances(problem):
    """dict[node] -> list of (kind, agent_or_slot) routing instances at that
    node, straight off problem.instance_node/instance_sources (spec.py's own
    parse -- see GraphOrderingRelaxed's docstring for the ("fixed", agent_id)
    / ("var", slot) shape). Tells the assignment layer which agent(s) a node
    actually constrains, before any discrete-branch or agent-choice search
    happens."""
    out = {}
    for node, src in zip(problem.instance_node, problem.instance_sources):
        out.setdefault(int(node), []).append(src)
    return out


def warm_start_wp(problem, x0):
    """Builds a (n_nodes, state_dim) wp template by propagating known
    values along problem.hard_edges rather than leaving free nodes at an
    arbitrary constant (never initialize a free node to 0 -- see the design
    discussion this module follows from: an uninformative default invents
    fake-cheap routes a discrete solve can then get permanently stuck
    behind).

    A node is "known" here if it is the write_node of some self-contained
    node projection (len(entry.node_locals) == 1 -- resolved per-branch,
    not yet chosen, so its template row is only a placeholder, overwritten
    again in node_candidates) or if it has no projection at all and at
    least one already-known neighbor to copy from. Every other node
    (including one written by an edge projection, which genuinely needs
    its other endpoint's row first) is filled by linear interpolation
    between its nearest known predecessor and successor along hard_edges
    (BFS in both directions); a node with no known node reachable at all
    falls back to the flattened x0 (agent_depot padded to state_dim), i.e.
    "stay where you already are" rather than the origin.
    """
    n_nodes, state_dim = problem.n_nodes, problem.state_dim
    wp = np.zeros((n_nodes, state_dim))
    known = np.zeros(n_nodes, dtype=bool)

    x0_flat = np.zeros(state_dim)
    n = min(state_dim, np.asarray(x0).reshape(-1).shape[0])
    x0_flat[:n] = np.asarray(x0).reshape(-1)[:n]
    wp[:] = x0_flat  # fallback for nodes unreachable from any known anchor

    for entry in problem.projections:
        if len(entry.node_locals) != 1:
            continue  # edge projection -- needs the other endpoint first; skip here
        node = entry.write_node
        # Placeholder: branch 0's row (via its own table if precomputed,
        # else a zero psi/branch-0 apply_projections call would also work --
        # a plain zero-fill is enough here since node_candidates below
        # re-resolves every branch's actual row for the discrete search,
        # this call only needs to mark the node "known" for interpolation).
        # A DYNAMIC (var_agent_q) entry is skipped for the write -- its
        # `pinned_cols` is empty and which agent's columns it targets isn't
        # known without an assignment -- but the node is still marked known.
        if entry.table is not None and entry.owner_var_slot is None:
            wp[node, np.asarray(entry.pinned_cols)] = np.asarray(entry.table)[0]
        known[node] = True

    u = np.array([e[0] for e in problem.hard_edges], dtype=int)
    v = np.array([e[1] for e in problem.hard_edges], dtype=int)

    # Fixed-point relaxation: repeatedly average each unknown node with its
    # already-known hard-edge neighbors, propagating known values inward a
    # hop at a time. Converges in at most n_nodes passes (a diameter bound);
    # cheap at this graph's size (<=25 nodes) so no need for a smarter
    # single-pass BFS/topological scheme here.
    for _ in range(n_nodes):
        changed = False
        for a, b in zip(u, v):
            if known[a] and not known[b]:
                wp[b] = wp[a]
                known[b] = True
                changed = True
            elif known[b] and not known[a]:
                wp[a] = wp[b]
                known[a] = True
                changed = True
            elif known[a] and known[b]:
                # both known already -- nothing to propagate
                continue
        if not changed:
            break

    return wp


def static_entry_owner(problem, entry):
    """The routing-instance OWNER a static (owner_var_slot is None)
    projection entry belongs to, derived from which column band its
    `pinned_cols` fall in: an agent id `c0 // problem.dim` for an
    agent-config band, or `("object", c0)` for an object band. This is the
    same column->track logic SmallContinuousVRPSolver._build_static_chain
    uses (`by_track`).

    Raises on a DYNAMIC (var_agent_q-pinned) entry -- its `pinned_cols` is
    empty and its write target moves with the assignment; use
    `entry_owner(problem, entry, owner_vagent)` there instead."""
    if entry.owner_var_slot is not None:
        raise ValueError(
            "static_entry_owner called on a dynamic (var_agent_q-pinned) "
            "projection entry -- use entry_owner(problem, entry, owner_vagent)")
    c0 = int(min(int(c) for c in entry.pinned_cols))
    if c0 < problem.n_agents * problem.dim:
        return c0 // problem.dim
    return ("object", c0)


def entry_owner(problem, entry, owner_vagent=None):
    """The routing-instance OWNER a projection entry writes into, for either
    kind of pin. A STATIC entry (owner_var_slot is None) defers to
    `static_entry_owner` (column-band derivation). A DYNAMIC (var_agent_q)
    entry resolves to the agent the current assignment binds its variable to
    -- `int(owner_vagent[entry.owner_var_slot])`; `owner_vagent` is the
    `(n_variables,)` resolved-agent vector for the assignment being priced
    (`dp_master`'s enumerated combo, or a solved result's `assignment`).
    Raises if it's missing for a dynamic entry."""
    if entry.owner_var_slot is None:
        return static_entry_owner(problem, entry)
    if owner_vagent is None:
        raise ValueError(
            "entry_owner needs owner_vagent for a dynamic (var_agent_q-pinned) "
            "projection entry -- its write target moves with the assignment")
    return int(np.asarray(owner_vagent)[entry.owner_var_slot])


def node_candidates(problem, wp_template, params, allow_unresolved=False, active_nodes=None):
    """dict[node] -> dict[owner] -> NodeCandidates, one inner entry per
    *self-contained static* projection (problem.projections,
    len(entry.node_locals) == 1 and entry.owner_var_slot is None) writing
    that node, keyed by `static_entry_owner`. This is the per-(agent,node)
    branch choice `B(j,v)` of the discrete formulation: a two-arm handoff
    node carrying one projection per arm contributes one inner entry per
    arm's owning agent, each with its own independent branch set.

    `allow_unresolved=True`: an EDGE / GATED / CHAINED / DYNAMIC projection
    is SKIPPED (not raised on) -- the caller (solve_dp_master with
    `_needs_full_resolve`) resolves those per enumerated schedule via
    apply_projections instead of from a precomputed table. The self-contained
    static entries still get their table here (the fast path).

    Reuses apply_projections directly (pop dimension = discrete_params, one
    candidate per branch) rather than re-deriving branch-to-row resolution,
    so this stays a mechanical lookup. Every branch's row is well-defined
    from wp_template alone (a self-contained entry's `reads` only ever pull
    from write_node's own pre-overwrite template row or from
    problem.params).

    A (node, owner) with no projection is simply absent -- it has one
    implicit candidate, wp_template[node], which the CP-SAT layer treats as
    a constant. Raises on:
      * an EDGE projection (len(node_locals) == 2) -- needs the other
        endpoint's candidate first (the LBBD continuous subproblem's job);
      * a DYNAMIC projection (owner_var_slot is not None) -- which agent's
        columns it writes moves with the assignment, not supported here yet
        (matches SmallContinuousVRPSolver's own fence);
      * two projections pinning the SAME owner's band at one node (chained);
      * a CHAINED projection -- one whose `reads` column another projection
        `pins` (spec.py's _resolve_projections allows these and orders them,
        but this per-candidate enumeration assumes each entry resolves from
        wp_template alone).
    """
    # A projection that READS a column another projection PINS can't be
    # tabulated from wp_template alone (the predecessor's substitution isn't
    # reflected there). A projection that is merely READ BY a downstream one
    # is fine -- its own candidate rows still resolve independently.
    chained_readers = {id(b) for a in problem.projections for b in problem.projections
                       if a is not b and (a.write_cols & b.read_cols)}
    if chained_readers and not allow_unresolved:
        raise NotImplementedError(
            "chained projections (one projection's `reads` column is pinned by "
            "another) are not supported by the discrete solver -- resolving that "
            "dependency belongs to the continuous subproblem (pass "
            "allow_unresolved=True to defer them to solve_dp_master)")
    out = {}
    n_nodes = problem.n_nodes
    # `active_nodes` (an anchor's remaining vertices): skip resolving a
    # projection whose write_node is already committed -- solve_dp_master
    # never consults a passed node's candidate rows (it routes only the
    # future subgraph), so the analytic-IK resolve there is wasted work.
    active_set = None if active_nodes is None else set(active_nodes)
    # Reused across entries (all traced args are per-entry; these are not).
    _dummy_assign1 = jnp.zeros((1, problem.n_variables, problem.n_agents))
    _dummy_cb1 = jnp.zeros((1, problem.n_cond_vars))
    _dummy_t1 = jnp.arange(n_nodes, dtype=float)[None]
    _na_all = jnp.ones((n_nodes,), dtype=bool)
    _x0_dummy = jnp.asarray(np.asarray(wp_template).reshape(-1)[:problem.state_dim])
    _params = jnp.asarray(params)
    _wp1 = jnp.asarray(wp_template)[None]
    for entry in problem.projections:
        if active_set is not None and entry.write_node not in active_set:
            continue
        deferred = (len(entry.node_locals) != 1 or entry.owner_var_slot is not None
                    or entry.gate_fn is not None or id(entry) in chained_readers)
        if deferred:
            if allow_unresolved:
                continue
            raise NotImplementedError(
                f"node {entry.write_node}: edge/gated/dynamic projection is not "
                "a self-contained per-branch table -- pass allow_unresolved=True "
                "to defer it to solve_dp_master's per-schedule resolution")
        owner = static_entry_owner(problem, entry)
        node = entry.write_node
        if owner in out.get(node, {}):
            raise NotImplementedError(
                f"node {node}: two projections pin owner {owner!r}'s band "
                "(chained projections aren't supported)")
        k = entry.discrete_params
        branch_ids = np.arange(k)
        proj_branch_pop = jnp.zeros((k, problem.n_branch)).at[
            branch_ids, entry.branch_slice.start + branch_ids].set(1.0)
        # Resolve ONLY this self-contained entry (only_entries) -- skips the
        # rest of the projection chain and, since a tabulated entry is never
        # gated, the topological-rank decode too; jitted + cached per
        # (problem, entry) so the analytic-IK cost is a one-time compile.
        resolve = jit_apply_projections(problem, only_entries=(entry,))
        resolved = np.asarray(resolve(
            jnp.broadcast_to(_wp1, (k, n_nodes, problem.state_dim)),
            jnp.zeros((k, problem.n_psi)), proj_branch_pop, _params,
            jnp.broadcast_to(_dummy_assign1, (k, problem.n_variables, problem.n_agents)),
            jnp.broadcast_to(_dummy_cb1, (k, problem.n_cond_vars)),
            jnp.broadcast_to(_dummy_t1, (k, n_nodes)), _na_all, _x0_dummy))
        rows = resolved[:, node, :]
        out.setdefault(node, {})[owner] = NodeCandidates(
            node=node, rows=rows, branch_ids=branch_ids)
    return out


# cost_fn -> jax.jit(jax.vmap(cost_fn)), weakref-keyed so it evicts once
# `cost_fn` (and whatever problem/edge_cost_fn holds it) is gone.
# `build_edge_cost_table`/`build_depot_cost_table` used to call `cost_fn`
# once per (a, b) pair in a python loop and force each result to a python
# float -- for a jax-traceable `cost_fn` (e.g. an NTField's `travel_time`,
# objectives/ntfield.py) that's one eager JAX dispatch per pair, hundreds of
# ms to seconds for a graph with any real branch fan-out. Batching every
# pair a table needs into one vmapped call amortizes that to a single
# dispatch (a one-time compile per distinct batch size, milliseconds after).
# Relies on `cost_fn` being the SAME object across calls at a fixed
# (agent_id, dim) -- see cpsat_model._agent_sliced_cost_fn's own cache.
_batched_cost_fn_jit = weakref.WeakKeyDictionary()


def _batched_cost_fn(cost_fn):
    f = _batched_cost_fn_jit.get(cost_fn)
    if f is None:
        f = jax.jit(jax.vmap(cost_fn))
        _batched_cost_fn_jit[cost_fn] = f
    return f


def build_edge_cost_table(rows_by_node, wp_template, pairs, edge_cost_fn=None):
    """dict[(u, v)] -> (n_cand_u, n_cand_v) numpy cost matrix, one per
    `pairs`. `rows_by_node`: dict[node] -> (n_cand, state_dim) already
    resolved for ONE agent (its own instance's candidate rows at that
    node); a node absent from it is treated as its single wp_template row.
    `edge_cost_fn` is the already-agent-sliced callable(a, b) -> scalar
    (cpsat_model._agent_sliced_cost_fn); default kernel._euclidean_edge_cost.

    Every (a, b) pair across every entry in `pairs` is gathered into one
    flat batch and run through `cost_fn` in a SINGLE `jax.vmap`+`jax.jit`
    call (cached per `cost_fn`, see `_batched_cost_fn`) rather than one
    eager call per pair."""
    cost_fn = edge_cost_fn if edge_cost_fn is not None else _euclidean_edge_cost
    if not pairs:
        return {}
    shapes = []
    a_chunks, b_chunks = [], []
    for u, v in pairs:
        rows_u = rows_by_node[u] if u in rows_by_node else wp_template[u][None, :]
        rows_v = rows_by_node[v] if v in rows_by_node else wp_template[v][None, :]
        nu, nv = rows_u.shape[0], rows_v.shape[0]
        a_chunks.append(np.repeat(np.asarray(rows_u), nv, axis=0))
        b_chunks.append(np.tile(np.asarray(rows_v), (nu, 1)))
        shapes.append((u, v, nu, nv))
    costs = np.asarray(_batched_cost_fn(cost_fn)(
        jnp.asarray(np.concatenate(a_chunks, axis=0)),
        jnp.asarray(np.concatenate(b_chunks, axis=0))))
    table = {}
    offset = 0
    for u, v, nu, nv in shapes:
        n = nu * nv
        table[(u, v)] = costs[offset:offset + n].reshape(nu, nv)
        offset += n
    return table


def build_depot_cost_table(rows_by_node, wp_template, x0_row, nodes, edge_cost_fn=None):
    """dict[node] -> (n_cand,) numpy cost array: the cost from the REAL
    current state `x0_row` (state_dim,) to each of `node`'s candidate rows
    (the depot-to-first-stop leg). `rows_by_node` as in
    build_edge_cost_table; needed for every node in `nodes` (any of the
    agent's own nodes could be first in the solved order). Batched into one
    `cost_fn` call the same way build_edge_cost_table is."""
    cost_fn = edge_cost_fn if edge_cost_fn is not None else _euclidean_edge_cost
    if not nodes:
        return {}
    x0_row = np.asarray(x0_row)
    shapes = []
    a_chunks, b_chunks = [], []
    for node in nodes:
        rows = np.asarray(rows_by_node[node] if node in rows_by_node else wp_template[node][None, :])
        n = rows.shape[0]
        a_chunks.append(np.broadcast_to(x0_row, (n,) + x0_row.shape))
        b_chunks.append(rows)
        shapes.append((node, n))
    costs = np.asarray(_batched_cost_fn(cost_fn)(
        jnp.asarray(np.concatenate(a_chunks, axis=0)),
        jnp.asarray(np.concatenate(b_chunks, axis=0))))
    out = {}
    offset = 0
    for node, n in shapes:
        out[node] = costs[offset:offset + n]
        offset += n
    return out


# What a discrete solver needs to know about one projection entry before it
# can decide WHEN that entry's rows are fixed -- see projection_dependencies.
#   gate: "none" (ungated); "on" / "off" (gated, but provably constant under
#       every schedule hard precedence allows, anchors included); or
#       "varying".
#   upstream: frozenset of entry indices this entry reads the output of,
#       transitively (a writer's write_cols meets a reader's read_cols).
#   varying_gates: sorted tuple of indices -- this entry itself and/or its
#       upstream closure -- whose gate is "varying": the pins that make this
#       entry's value depend on the visiting order.
#   order_nodes: union of those gates' gate_nodes -- the nodes whose relative
#       order this entry's value can depend on. None if some varying gate
#       has unknown gate_nodes (assume the whole order); empty if none.
#   robot_reads: frozenset of (node, col) agent-configuration columns read
#       by this entry or anything upstream of it -- the robot rows (hence,
#       for a branched writer, the branch choices) its value depends on.
ProjectionDeps = namedtuple(
    "ProjectionDeps",
    ["gate", "upstream", "varying_gates", "order_nodes", "robot_reads"])


def _hard_reach(n_nodes, hard_edges):
    """`reach[u]` = every node that must come after `u` under the hard
    precedence edges (transitive)."""
    succ = [[] for _ in range(n_nodes)]
    for u, v in hard_edges:
        succ[int(u)].append(int(v))
    reach = []
    for u in range(n_nodes):
        seen, stack = set(), list(succ[u])
        while stack:
            w = stack.pop()
            if w not in seen:
                seen.add(w)
                stack.extend(succ[w])
        reach.append(seen)
    return reach


def _gate_rank_batch(n_nodes, gate_nodes, reach, cap):
    """Every rank vector a gate over `gate_nodes` can actually be evaluated
    against, restricted to those nodes, as an `(B, n_nodes)` int32 array --
    or None if there are more than `cap`.

    `kernel._topological_rank` gives active nodes distinct ranks in a
    topological order and EVERY passed node the same rank -1, and in MPC the
    passed set is always down-closed under hard precedence (execution
    respected it). So the reachable relative orders of `gate_nodes` are:
    a down-closed subset D (tied at -1) x a linear extension of the rest.
    Only hard edges are used -- conditional ones only ever remove orders, so
    this is a superset, which is the safe direction for proving a gate
    constant. Nodes outside `gate_nodes` get distinct ranks above every
    gate node; by the gate_nodes contract they are never read."""
    G = sorted(int(n) for n in gate_nodes)
    k = len(G)
    if k > 20:
        return None
    before = {x: {y for y in G if x in reach[y]} for x in G}   # y must precede x
    base = np.arange(n_nodes, dtype=np.int32) + n_nodes
    rows = []

    def extensions(rest, prefix, out):
        if len(out) > cap:
            return
        if not rest:
            out.append(list(prefix))
            return
        for x in sorted(rest):
            if before[x] & rest:
                continue
            prefix.append(x)
            extensions(rest - {x}, prefix, out)
            prefix.pop()

    for mask in range(1 << k):
        D = {G[i] for i in range(k) if (mask >> i) & 1}
        if any(not before[x] <= D for x in D):
            continue                                   # not down-closed
        exts = []
        extensions(set(G) - D, [], exts)
        for order in exts:
            r = base.copy()
            for x in D:
                r[x] = -1
            for pos, x in enumerate(order):
                r[x] = pos
            rows.append(r)
            if len(rows) > cap:
                return None
    return np.stack(rows) if rows else np.zeros((0, n_nodes), np.int32)


def _classify_gate(entry, n_nodes, reach, cap):
    if entry.gate_fn is None:
        return "none"
    if entry.gate_nodes is None:
        return "varying"                  # unknown reads: assume the worst
    ranks = _gate_rank_batch(n_nodes, entry.gate_nodes, reach, cap)
    if ranks is None:
        return "varying"                  # too many orders to prove anything
    if ranks.shape[0] == 0:
        ranks = np.zeros((1, n_nodes), np.int32)
    g = np.asarray(entry.gate_fn(jnp.asarray(ranks)))
    if np.all(g == g[0]):
        return "on" if g[0] > 0.5 else "off"
    return "varying"


def projection_dependencies(problem, max_gate_orders=20000):
    """`list[ProjectionDeps]`, one per `problem.projections` entry: what each
    entry's value depends on that a routing solver DECIDES -- the visiting
    order (through gated pins) and robot configurations (through agent-
    column reads) -- traced transitively through the entries it reads.

    A gated pin is only order-dependent if its gate can actually change
    value. Each gate is evaluated over every rank it can really be handed
    (`_gate_rank_batch`: hard-precedence-consistent orders of its
    `gate_nodes`, with any down-closed subset tied at the passed-node rank
    -1). If the value never changes it is classified "on"/"off" and is not
    a source of order dependence at all. That is exact for the gate as
    written -- it runs the real `gate_fn` -- and sound for every assignment,
    aux vector and anchor, since it only ever considers a superset of the
    reachable orders. A gate with unknown `gate_nodes`, or too many orders
    to enumerate (`max_gate_orders`), is conservatively "varying".

    This is the analysis a discrete solver needs to decide when a node's
    candidate rows are fixed: rows with empty `varying_gates` and empty
    `robot_reads` can be resolved once per (assignment, aux), independent
    of the order and of every branch choice."""
    entries = list(problem.projections)
    n_nodes = int(problem.n_nodes)
    hard = [(u, v) for (u, v, g) in problem.ordering_edges if g is None]
    reach = _hard_reach(n_nodes, hard)
    agent_hi = int(problem.n_agents) * int(problem.dim)

    gate = [_classify_gate(e, n_nodes, reach, max_gate_orders) for e in entries]
    direct = [{j for j, w in enumerate(entries) if j != i and (w.write_cols & e.read_cols)}
              for i, e in enumerate(entries)]

    memo = {}

    def closure(i, stack=()):
        if i in memo:
            return memo[i]
        if i in stack:
            raise ValueError(f"projection_dependencies: entries {stack + (i,)} "
                             "read each other's output in a cycle")
        up = set()
        for j in direct[i]:
            up.add(j)
            up |= closure(j, stack + (i,))
        memo[i] = frozenset(up)
        return memo[i]

    out = []
    for i, e in enumerate(entries):
        up = closure(i)
        scope = sorted(up | {i})
        varying = tuple(k for k in scope if gate[k] == "varying")
        if any(entries[k].gate_nodes is None for k in varying):
            order_nodes = None
        else:
            order_nodes = frozenset().union(*(entries[k].gate_nodes for k in varying))
        robot = frozenset((int(n), int(c)) for k in scope
                          for (n, c) in entries[k].read_cols if c < agent_hi)
        out.append(ProjectionDeps(gate[i], up, varying, order_nodes, robot))
    return out


def conditional_edge_data(graph, problem):
    """(cond_formulas, var_sym_ids, cond_sym_ids) for CP-SAT's conditional
    ordering edges -- everything compile_gate_cpsat needs, read straight off
    the graph the same way spec.build_graph_ordering_problem does:
      * cond_formulas: dict[(u, v)] -> drake symbolic Formula
        (graph.conditional_ordering_map, verbatim).
      * var_sym_ids: assignment_sym(var).get_id() -> assignment slot index
        (index into `assign_bool`).
      * cond_sym_ids: binary_cond_sym_var.get_id() -> aux-binary index
        (index into `aux_bool`).
    """
    var_ids = sorted(problem.var_id_to_slot, key=problem.var_id_to_slot.get)
    var_sym_ids = {graph.assignment_sym(v).get_id(): problem.var_id_to_slot[v]
                   for v in var_ids}
    cond_sym_ids = {v.get_id(): i for i, v in enumerate(graph.binary_cond_sym_vars)}
    return dict(graph.conditional_ordering_map), var_sym_ids, cond_sym_ids


def _is_const(expr):
    return expr.get_kind() == sym.ExpressionKind.Constant


def compile_gate_cpsat(f, model, assign_bool, aux_bool, var_sym_ids, cond_sym_ids,
                       n_agents, name, _and_lit):
    """Compile a drake symbolic Formula (graph.conditional_ordering_map value)
    into a CP-SAT BoolVar `g` reified `g <=> formula`. The CP-SAT analogue of
    milp_waypoint_mpc.cpp's CompileConditionToBinary (and of
    formula_compiler.compile_condition, which emits a jnp bool instead).

    Recognised atoms:
      * VarEq   -- assignment_sym(s0) == assignment_sym(s1): the two vars
        resolve to the same real agent. `g <=> OR_a (A[s0,a] AND A[s1,a])`.
      * AssignEq -- assignment_sym(s) == <int j>: var s is assigned to agent
        j. `g == assign_bool[s][j]` directly (no new var).
      * BinVarEq -- bv == 0 / 1: a free auxiliary binary. Aliases aux_bool
        (or its negation).
    Compounds: And / Or / Not. `_and_lit` is cpsat_model._and_lit, passed in
    to avoid an import cycle."""
    kind = f.get_kind()

    if kind == sym.FormulaKind.Eq:
        _, (lhs, rhs) = f.Unapply()
        lv, rv = as_variable(lhs), as_variable(rhs)

        if lv is not None and rv is not None \
           and lv.get_id() in var_sym_ids and rv.get_id() in var_sym_ids:
            s0, s1 = var_sym_ids[lv.get_id()], var_sym_ids[rv.get_id()]
            same = [_and_lit(model, assign_bool[s0][a], assign_bool[s1][a],
                             f"{name}_same{a}") for a in range(n_agents)]
            g = model.NewBoolVar(name)
            model.AddMaxEquality(g, same)
            return g

        for ve, ce in ((lv, rhs), (rv, lhs)):
            if ve is not None and ve.get_id() in var_sym_ids and _is_const(ce):
                return assign_bool[var_sym_ids[ve.get_id()]][int(round(ce.Evaluate()))]

        for ve, ce in ((lv, rhs), (rv, lhs)):
            if ve is not None and ve.get_id() in cond_sym_ids:
                b = aux_bool[cond_sym_ids[ve.get_id()]]
                return b if ce.Evaluate() >= 0.5 else b.Not()

        raise ValueError(f"unsupported Eq atom in conditional edge {name}")

    if kind == sym.FormulaKind.And:
        subs = [compile_gate_cpsat(s, model, assign_bool, aux_bool, var_sym_ids,
                                   cond_sym_ids, n_agents, f"{name}_a{i}", _and_lit)
                for i, s in enumerate(f.Unapply()[1])]
        g = model.NewBoolVar(name)
        model.AddMinEquality(g, subs)
        return g

    if kind == sym.FormulaKind.Or:
        subs = [compile_gate_cpsat(s, model, assign_bool, aux_bool, var_sym_ids,
                                   cond_sym_ids, n_agents, f"{name}_o{i}", _and_lit)
                for i, s in enumerate(f.Unapply()[1])]
        g = model.NewBoolVar(name)
        model.AddMaxEquality(g, subs)
        return g

    if kind == sym.FormulaKind.Not:
        (sub,) = f.Unapply()[1]
        return compile_gate_cpsat(sub, model, assign_bool, aux_bool, var_sym_ids,
                                  cond_sym_ids, n_agents, f"{name}_n", _and_lit).Not()

    raise ValueError(f"unsupported FormulaKind {kind} in conditional edge {name}")

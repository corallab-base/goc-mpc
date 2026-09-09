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


def node_candidates(problem, wp_template, params, allow_unresolved=False):
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
    # Reused across entries (all traced args are per-entry; these are not).
    _dummy_assign1 = jnp.zeros((1, problem.n_variables, problem.n_agents))
    _dummy_cb1 = jnp.zeros((1, problem.n_cond_vars))
    _dummy_t1 = jnp.arange(n_nodes, dtype=float)[None]
    _na_all = jnp.ones((n_nodes,), dtype=bool)
    _x0_dummy = jnp.asarray(np.asarray(wp_template).reshape(-1)[:problem.state_dim])
    _params = jnp.asarray(params)
    _wp1 = jnp.asarray(wp_template)[None]
    for entry in problem.projections:
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

"""Exact enumerate-orders x DP-branches solver for the *same* discrete
routing / assignment / per-(agent,node)-branch problem `cpsat_model.py`
encodes -- an alternative to the CP-SAT search, meant to be faster on the
strongly-precedence-constrained instances real scenes produce (block
stacking, hand-offs) where the number of feasible global orders is tiny.

Equivalence to `build_cpsat_model` (see that module's docstring for the
formulation).  For a fixed assignment `A` and auxiliary-binary vector
`aux`, the conditional ordering edges resolve to a concrete precedence DAG
`P = hard_edges u {gated edges whose formula is true}`.  Every feasible
CP-SAT solution induces a *global linear extension* of `P` (sort the nodes
by `time[v]`), and every linear extension is realizable (set `time` to the
DAG longest-path values).  So enumerating linear extensions of `P` loses
nothing.  Given one linear extension:

  * each agent's route is its own subsequence of that order (AddCircuit's
    "visit each owned node once", made concrete);
  * `avg` / `minmax` are separable -- agent routes share no variables and
    `edge_cost_fn` reads only the agent's own column slice -- so the score
    is `sum_j` / `max_j` of a per-agent branch Viterbi along that
    subsequence from the real depot state (exactly `_solve_branch_dp`'s
    per-track shortest path);
  * `makespan` couples agents only through the shared `time[v]` on
    ordering edges, so once the order and every node's branch are fixed the
    arrival times are a deterministic DAG forward pass
    `arr[v] = max(max_{(p,v) in P} arr[p],  arr[prev_own] + c,  depot_c)`
    -- `min t(n+1)` with exact cross-agent ordering-induced waiting.  The
    branch choice is not separable from the waiting, so we brute-force the
    branch combinations (small: `prod_v k_{owner,v}`).

`node_candidates`'s edge tables are reused verbatim
(`structure.build_edge_cost_table` / `build_depot_cost_table`), so the
per-arc costs are byte-for-byte what CP-SAT sees -- and here they stay
float (no `_SCALE` rounding).

Scope / fences (raise `NotImplementedError`, never silently approximate):
  * more than `max_assign_combos` assignment combinations, or more than
    `max_orders` linear extensions of some resolved `P` -- the regime where
    this enumeration stops being the right algorithm (Held-Karp per agent
    for avg/minmax, or CP-SAT, takes over);
  * the same `node_candidates` fences (edge / dynamic / chained
    projections) propagate up unchanged.

Multi-instance disjunctive ownership (`exists i in I_v : A(i,j)=1`): a
`(node, agent)` visit is gated on the OR over instances of `v` that
resolve to `j`; here that is just set membership, so a node owned by the
same agent through two instances is one stop (CP-SAT currently raises on
this -- a known, tracked divergence to retrofit).
"""

import itertools

import numpy as np

from .structure import build_edge_cost_table, build_depot_cost_table, entry_owner
from .cpsat_model import _agent_sliced_cost_fn


def _resolve_precedence(problem, ordering_edges, owner_vagent, aux):
    """`set[(u, v)]` -- hard edges plus every conditional edge whose gate is
    true under this `(A, aux)`.  `owner_vagent`: (n_variables,) resolved
    agent id per assignment slot; `aux`: (n_cond_vars,) 0/1."""
    ov = np.asarray(owner_vagent)
    cb = np.asarray(aux, dtype=float)
    P = set()
    for u, v, gate in ordering_edges:
        if gate is None or bool(gate(ov, cb)):
            P.add((int(u), int(v)))
    return P


def _transitive_closure(n_nodes, P):
    reach = [set() for _ in range(n_nodes)]
    for u, v in P:
        reach[u].add(v)
    changed = True
    while changed:
        changed = False
        for u in range(n_nodes):
            add = set()
            for w in reach[u]:
                add |= reach[w]
            if not add <= reach[u]:
                reach[u] |= add
                changed = True
    return reach


def _linear_extensions(nodes, P, cap):
    """All total orders over `nodes` consistent with `P` (all topological
    sorts).  Returns `list[tuple[int]]`; raises if there would be more than
    `cap`.  `None` if `P` (restricted to `nodes`) has a cycle.

    `nodes` is the set of node ids to order -- pass `range(n_nodes)` for
    the whole graph, or an `anchor`'s remaining vertices to enumerate only
    the future subgraph (edges of `P` touching a node outside `nodes` are
    ignored: an already-committed predecessor imposes no constraint on the
    order the remaining work is done in)."""
    nodes = list(nodes)
    nset = set(nodes)
    n = len(nodes)
    succ = {i: [] for i in nodes}
    indeg = {i: 0 for i in nodes}
    for u, v in P:
        if u in nset and v in nset:
            succ[u].append(v)
            indeg[v] += 1
    if n and all(d > 0 for d in indeg.values()):
        return None  # every node has a predecessor -> cycle

    out = []
    order = []
    used = set()
    deg = dict(indeg)

    def rec():
        if len(out) > cap:
            raise NotImplementedError(
                f"dp_master: precedence DAG has > {cap} linear extensions "
                "-- too loosely ordered for order enumeration (use Held-Karp "
                "per agent for avg/minmax, or CP-SAT)")
        if len(order) == n:
            out.append(tuple(order))
            return
        ready = [i for i in nodes if i not in used and deg[i] == 0]
        if not ready:
            return  # cycle among the remainder
        for i in ready:
            used.add(i)
            order.append(i)
            for j in succ[i]:
                deg[j] -= 1
            rec()
            for j in succ[i]:
                deg[j] += 1
            order.pop()
            used.discard(i)

    rec()
    return out


def _extensions_matrix(nodes, P, cap, bucket=True):
    """Static-shape linear extensions for the vectorized kernels.

    `_linear_extensions` returns a variable-length `list[tuple]`; a `vmap` /
    `scan` over it needs a fixed leading axis. This wraps it into

      `(mat, n_valid, M)`

    where `mat` is `(M, L)` int32 (`L == len(nodes)`), rows `[:n_valid]` are
    the linear extensions of `P` over `nodes` (node ids in order) and rows
    `[n_valid:M]` repeat `mat[0]` -- harmless filler so a vmapped kernel
    never evaluates an invalid permutation; mask its results with
    `arange(M) < n_valid`. `n_valid` is the real extension count (`0` iff `P`
    restricted to `nodes` has a cycle). `M` is `n_valid` rounded up to the
    next power of two when `bucket` (so the kernels see O(log cap) distinct
    trace shapes instead of one per count), else exactly `max(n_valid, 1)`.

    Raises (via `_linear_extensions`) if there would be more than `cap`
    extensions."""
    nodes = list(nodes)
    L = len(nodes)
    exts = _linear_extensions(nodes, P, cap)
    if not exts:  # None (cycle) or [] (never -- kept defensive)
        return np.zeros((1, L), dtype=np.int32), 0, 1
    n_valid = len(exts)
    M = (1 << (n_valid - 1).bit_length()) if bucket else n_valid
    mat = np.zeros((M, L), dtype=np.int32)
    for i, e in enumerate(exts):
        mat[i, :] = e
    mat[n_valid:, :] = mat[0, :]
    return mat, n_valid, M


def _agent_owned_nodes(problem, instances, owner_vagent):
    """`dict[agent_id] -> set[node]` -- every node the agent visits under
    this assignment, disjunctively over that node's instances."""
    owned = {}
    for node, insts in instances.items():
        for kind, val in insts:
            j = int(val) if kind == "fixed" else int(owner_vagent[val])
            owned.setdefault(j, set()).add(int(node))
    return owned


def _inst_key(problem, instances, node, agent_id, owner_vagent):
    """The `(node, inst_key)` key `candidates` / CP-SAT `branch` use for
    `agent_id` at `node` -- the agent id itself for a fixed instance,
    `("var", slot)` for an assignable one it wins."""
    for kind, val in instances.get(node, []):
        if kind == "fixed" and int(val) == agent_id:
            return agent_id
        if kind == "var" and int(owner_vagent[val]) == agent_id:
            return ("var", val)
    return agent_id


def _agent_tables(problem, candidates, wp_template, x0_row, nodes, owner,
                  edge_cost_fn):
    """`(edge_table, depot_table, branch_counts)` for one agent over its
    `nodes`, reusing `structure`'s builders so the numbers match CP-SAT
    exactly.  `owner`: the candidate-dict key for this agent's instance at
    each node (agent id, or the `('var', slot)` it resolves)."""
    sliced = _agent_sliced_cost_fn(edge_cost_fn, owner if isinstance(owner, int) else 0,
                                   problem.dim)
    rows_by_node = {}
    branch_counts = {}
    for n in nodes:
        nc = candidates.get(n, {}).get(owner if isinstance(owner, int) else None)
        if nc is not None:
            rows_by_node[n] = nc.rows
            branch_counts[n] = nc.rows.shape[0]
        else:
            branch_counts[n] = 1
    pairs = [(u, v) for u in nodes for v in nodes if u != v]
    et = build_edge_cost_table(rows_by_node, wp_template, pairs, edge_cost_fn=sliced)
    dt = build_depot_cost_table(rows_by_node, wp_template, x0_row, nodes, edge_cost_fn=sliced)
    return et, dt, branch_counts


def _branch_viterbi(seq, edge_table, depot_table, branch_counts):
    """Min routed path cost + per-node branch argmin for one agent whose
    ordered stops are `seq` (from the depot).  Exactly `_solve_branch_dp`'s
    per-track shortest path, over an already-fixed visiting order."""
    if not seq:
        return 0.0, {}
    k0 = branch_counts[seq[0]]
    cost = np.array(depot_table[seq[0]][:k0], dtype=float)
    back = []
    for j in range(1, len(seq)):
        u, v = seq[j - 1], seq[j]
        mat = edge_table[(u, v)]  # (k_u, k_v)
        total = cost[:, None] + mat
        back.append(np.argmin(total, axis=0))
        cost = np.min(total, axis=0)
    choice = [0] * len(seq)
    choice[-1] = int(np.argmin(cost))
    for j in range(len(seq) - 2, -1, -1):
        choice[j] = int(back[j][choice[j + 1]])
    return float(cost[choice[-1]]), {n: choice[i] for i, n in enumerate(seq)}


def _makespan_forward(ext, P_pred, agent_seq, agent_of_node, branch_of_node,
                      edge_tables, depot_tables):
    """Arrival time of every node for one linear extension `ext` and a fixed
    branch assignment `branch_of_node`.  `arr[v] = max(precedence preds,
    own-agent travel-from-previous | depot leg)`."""
    arr = {}
    prev_own = {}  # agent -> last node placed
    for v in ext:
        t = 0.0
        for p in P_pred[v]:
            t = max(t, arr[p])
        js = agent_of_node.get(v, [])
        for j in js:
            bv = branch_of_node[(v, j)]
            if j in prev_own:
                u = prev_own[j]
                bu = branch_of_node[(u, j)]
                t = max(t, arr[u] + edge_tables[j][(u, v)][bu, bv])
            else:
                t = max(t, depot_tables[j][v][bv])
        arr[v] = t
        for j in js:
            prev_own[j] = v
    return arr


def _needs_full_resolve(problem):
    """True iff some projection that AFFECTS AN AGENT COLUMN can't be
    reduced to a self-contained per-branch table (structure.node_candidates)
    -- a GATED, DYNAMIC (var_agent_q), EDGE, or chained-reader entry writing
    into the routing-relevant agent band. Those need per-(assignment, aux,
    order, branch-combo) resolution via apply_projections.

    A gated/chained entry that only pins OBJECT columns (the auto-derived
    stationary / rigid-carry ones) does NOT trigger this: object columns
    never enter an agent's route cost or arrival time, so the discrete solve
    (assignment + branch + order + times) is unaffected -- the continuous
    subproblem applies those substitutions when it materializes W."""
    p = list(problem.projections)
    agent_hi = problem.n_agents * problem.dim
    readers = {id(b) for a in p for b in p if a is not b and (a.write_cols & b.read_cols)}
    for e in p:
        deferred = (e.gate_fn is not None or e.owner_var_slot is not None
                    or len(e.node_locals) == 2 or id(e) in readers)
        if deferred and any(c < agent_hi for (_n, c) in e.write_cols):
            return True
    return False


def _order_t(ext, n_nodes):
    """A `t` sort key whose decode is exactly the linear extension `ext`
    (position in the order) -- what apply_projections' gate machinery needs
    instead of the GA's continuous `t`."""
    t = np.zeros(n_nodes)
    for pos, nd in enumerate(ext):
        t[nd] = float(pos)
    return t


def _resolve_schedule_wp(problem, wp_pop, x0_full, owner_vagent, aux, proj_branch_pop, t_vec,
                         only_entries=None, node_active=None):
    """`(pop, n_nodes, state_dim)` wp for one enumerated schedule: the
    (`only_entries` subset of) projections spliced in by apply_projections
    against this assignment (`owner_vagent` one-hots), aux, per-member branch
    vectors and the order-derived `t` (shared across the pop). jitted +
    cached per (problem, only_entries). `wp_pop` and `proj_branch_pop` carry
    the population axis; everything else is broadcast.

    `node_active` (n_nodes,) bool -- an `anchor`'s remaining-vertex mask, so
    apply_projections freezes already-passed columns and the gate machinery
    decodes ranks consistent with the committed set (a passed node is
    pre-scheduled at rank -1, its `t` value ignored). Defaults to all-active."""
    import jax.numpy as jnp
    from ..evolutionary_waypoint_solver.problem import jit_apply_projections
    pop = wp_pop.shape[0]
    n_var, n_agents = problem.n_variables, problem.n_agents
    assign = np.zeros((pop, n_var, n_agents))
    for s in range(n_var):
        assign[:, s, int(owner_vagent[s])] = 1.0
    cb = (np.broadcast_to(np.asarray(aux, dtype=float), (pop, problem.n_cond_vars))
          if problem.n_cond_vars else np.zeros((pop, 0)))
    na = (np.ones((problem.n_nodes,), dtype=bool) if node_active is None
          else np.asarray(node_active, dtype=bool))
    out = jit_apply_projections(problem, only_entries=only_entries)(
        jnp.asarray(wp_pop), jnp.zeros((pop, problem.n_psi)),
        jnp.asarray(proj_branch_pop), jnp.asarray(problem.params),
        jnp.asarray(assign), jnp.asarray(cb),
        jnp.broadcast_to(jnp.asarray(t_vec, dtype=float), (pop, problem.n_nodes)),
        jnp.asarray(na), jnp.asarray(x0_full))
    return np.asarray(out)


def _branch_viterbi_wp(seq, rows_by_node, wp0, x0_row, sliced):
    """Per-track branch DP: min routed-path cost + per-node branch argmin for
    one agent's ordered stops `seq`, from the depot. `rows_by_node[n]` is a
    `(k, state_dim)` array of branch candidates for node n (a node not in it
    has a single candidate, `wp0[n]`). Exact for `avg` / `minmax` -- an
    agent's route cost is separable from every other agent's."""
    if not seq:
        return 0.0, {}

    def cands(n):
        return rows_by_node[n] if n in rows_by_node else wp0[n][None, :]

    r0 = cands(seq[0])
    cost = np.array([sliced(x0_row, row) for row in r0], dtype=float)
    back = []
    for i in range(1, len(seq)):
        ru, rv = cands(seq[i - 1]), cands(seq[i])
        mat = np.array([[sliced(a, b) for b in rv] for a in ru])  # (k_u, k_v)
        total = cost[:, None] + mat
        back.append(np.argmin(total, axis=0))
        cost = np.min(total, axis=0)
    choice = [0] * len(seq)
    choice[-1] = int(np.argmin(cost))
    for i in range(len(seq) - 2, -1, -1):
        choice[i] = int(back[i][choice[i + 1]])
    return float(cost[choice[-1]]), {n: choice[i] for i, n in enumerate(seq)}


def _select_wp(ext, rows_by_node, wp0, branch_of_node):
    """Materialize `(n_nodes, state_dim)` for one branch choice: each
    projected node's chosen candidate row, `wp0` elsewhere."""
    wp = np.array(wp0, copy=True)
    for n in ext:
        if n in rows_by_node:
            wp[n] = rows_by_node[n][branch_of_node.get(n, 0)]
    return wp


def _makespan_forward_wp(ext, P_pred, agent_of_node, wp_res, x0_of, sliced_of):
    arr = {}
    prev_own = {}
    for v in ext:
        t = 0.0
        for p in P_pred[v]:
            t = max(t, arr[p])
        for j in agent_of_node.get(v, []):
            if j in prev_own:
                u = prev_own[j]
                t = max(t, arr[u] + float(sliced_of[j](wp_res[u], wp_res[v])))
            else:
                t = max(t, float(sliced_of[j](x0_of[j], wp_res[v])))
        arr[v] = t
        for j in agent_of_node.get(v, []):
            prev_own[j] = v
    return arr


def _ext_nonfull_avg_minmax(objective, ext, P_pred, agent_of_node, owned, owner_vagent, aux,
                            agent_keys, edge_tables, depot_tables, branch_counts, best, best_sol):
    """One `ext` of the non-full avg/minmax loop: per-agent branch Viterbi
    over the pre-built cost tables, keep it if it beats `best`. Returns the
    (possibly updated) `(best, best_sol)`."""
    agent_seq = {j: [n for n in ext if n in owned.get(j, ())] for j in owned}
    per_agent = {}
    br = {}
    for j, seq in agent_seq.items():
        c, ch = _branch_viterbi(seq, edge_tables[j], depot_tables[j], branch_counts[j])
        per_agent[j] = c
        for n, b in ch.items():
            br[(n, agent_keys[j][n])] = b
    score = (sum(per_agent.values()) if objective == "avg"
             else max(per_agent.values(), default=0.0))
    if score < best:
        arr = _makespan_forward(
            ext, P_pred, agent_seq, agent_of_node,
            {(n, j): br[(n, agent_keys[j][n])] for n in ext
             for j in agent_of_node.get(n, [])},
            edge_tables, depot_tables)
        best = score
        best_sol = (dict(br), owner_vagent.copy(), aux, arr, agent_seq, per_agent)
    return best, best_sol


def _ext_nonfull_makespan(ext, P_pred, agent_of_node, owned, owner_vagent, aux,
                          agent_keys, edge_tables, depot_tables, branch_counts, best, best_sol):
    """One `ext` of the non-full makespan loop: brute-force every branch
    combo for this order, DAG forward pass per combo, keep the best."""
    agent_seq = {j: [n for n in ext if n in owned.get(j, ())] for j in owned}
    proj_nodes = [(n, j) for n in ext for j in agent_of_node.get(n, [])]
    ranges = [range(branch_counts[j][n]) for (n, j) in proj_nodes]
    for combo in itertools.product(*ranges):
        b_of = {proj_nodes[i]: combo[i] for i in range(len(proj_nodes))}
        arr = _makespan_forward(ext, P_pred, agent_seq, agent_of_node,
                                b_of, edge_tables, depot_tables)
        ms = max(arr.values(), default=0.0)
        if ms < best:
            br = {(n, agent_keys[j][n]): b_of[(n, j)] for (n, j) in proj_nodes}
            per_agent = {}
            for j, seq in agent_seq.items():
                pc = 0.0
                for i in range(1, len(seq)):
                    u, v = seq[i - 1], seq[i]
                    pc += edge_tables[j][(u, v)][b_of[(u, j)], b_of[(v, j)]]
                if seq:
                    pc += depot_tables[j][seq[0]][b_of[(seq[0], j)]]
                per_agent[j] = pc
            best = ms
            best_sol = (br, owner_vagent.copy(), aux, arr, agent_seq, per_agent)
    return best, best_sol


def _full_ext_resolve(problem, ext, n_nodes, owned, owner_vagent, aux, x0_full,
                      layer0, branched, wp_template, node_active=None):
    """Shared per-`ext` resolve for the full path: (1) the branch-free
    projection layer once against this order's `t`, (2) each branched
    entry's k candidate rows against that layer. Returns
    `(agent_seq, tvec, wp0, rows_by_node, node_owner_of)`.

    `node_active` (n_nodes,) bool -- an `anchor`'s remaining-vertex mask,
    threaded into the resolve so passed columns freeze and gates decode
    against the committed set."""
    agent_seq = {j: [n for n in ext if n in owned.get(j, ())] for j in owned}
    tvec = _order_t(ext, n_nodes)
    wp0 = _resolve_schedule_wp(
        problem, wp_template[None], x0_full, owner_vagent, aux,
        np.zeros((1, problem.n_branch)), tvec, only_entries=layer0,
        node_active=node_active)[0]

    rows_by_node = {}
    node_owner_of = {}          # projected node -> (entry, owner-agent)
    for e in branched:
        k = e.discrete_params
        pb = np.zeros((k, problem.n_branch))
        pb[np.arange(k), e.branch_slice.start + np.arange(k)] = 1.0
        res = _resolve_schedule_wp(
            problem, np.broadcast_to(wp0[None], (k,) + wp0.shape),
            x0_full, owner_vagent, aux, pb, tvec, only_entries=(e,),
            node_active=node_active)
        rows_by_node[int(e.write_node)] = res[:, int(e.write_node), :]
        # For a DYNAMIC (var_agent_q) entry the owner is whichever agent
        # THIS assignment binds its variable to -- fixed inside this
        # `for A in assign_combos` iteration, so the entry prices exactly
        # like a static multi-branch one from here on.
        node_owner_of[int(e.write_node)] = (e, entry_owner(problem, e, owner_vagent))
    return agent_seq, tvec, wp0, rows_by_node, node_owner_of


def _full_ext_resolve_batched(problem, mat, n_valid, n_nodes, owned, owner_vagent, aux,
                              x0_full, layer0, branched, wp_template, node_active=None):
    """All `n_valid` extensions' full-path resolve in `(1 + len(branched))`
    `apply_projections` launches instead of 2 per ext.

    Each `_resolve_schedule_wp` call already broadcasts assignment / aux /
    x0 / node_active over the population axis and takes a per-member `t`, so
    the only per-ext input is the order-derived `t` -- stack those and run
    the whole batch at once.

    Returns `(resolved_list, node_owner_of, wp0_all, rows_all)`:
    `resolved_list[m]` is the same `(agent_seq, tvec, wp0, rows_by_node,
    node_owner_of)` tuple `_full_ext_resolve` returns for `mat[m]` (only the
    first `n_valid` are real); `wp0_all` is `(M, n_nodes, S)` and `rows_all`
    is `dict[node -> (M, k, S)]` -- the raw tensors the batched scorer needs
    without re-slicing per ext."""
    M = mat.shape[0]
    S = wp_template.shape[1]
    exts = [tuple(int(n) for n in mat[m]) for m in range(M)]
    T = np.stack([_order_t(exts[m], n_nodes) for m in range(M)])       # (M, n_nodes)

    wp0_all = _resolve_schedule_wp(
        problem, np.broadcast_to(wp_template[None], (M, n_nodes, S)),
        x0_full, owner_vagent, aux, np.zeros((M, problem.n_branch)), T,
        only_entries=layer0, node_active=node_active)                  # (M, n_nodes, S)

    rows_all = {}                       # node -> (M, k, S)
    node_owner_of = {}
    for e in branched:
        k = e.discrete_params
        pb = np.zeros((k, problem.n_branch))
        pb[np.arange(k), e.branch_slice.start + np.arange(k)] = 1.0
        res = _resolve_schedule_wp(
            problem, np.repeat(wp0_all, k, axis=0), x0_full, owner_vagent, aux,
            np.tile(pb, (M, 1)), np.repeat(T, k, axis=0), only_entries=(e,),
            node_active=node_active).reshape(M, k, n_nodes, S)
        rows_all[int(e.write_node)] = res[:, :, int(e.write_node), :]
        node_owner_of[int(e.write_node)] = (e, entry_owner(problem, e, owner_vagent))

    resolved_list = []
    for m in range(M):
        agent_seq = {j: [n for n in exts[m] if n in owned.get(j, ())] for j in owned}
        rows_by_node = {nd: np.asarray(rows_all[nd][m]) for nd in rows_all}
        resolved_list.append((agent_seq, T[m], np.asarray(wp0_all[m]),
                              rows_by_node, node_owner_of))
    return resolved_list, node_owner_of, np.asarray(wp0_all), rows_all


def _ext_full_avg_minmax(problem, objective, ext, P_pred, agent_of_node, n_nodes, owned,
                         owner_vagent, aux, x0_full, x0_of, sliced_of, layer0, branched,
                         wp_template, best, best_sol, node_active=None):
    """One `ext` of the full-resolve avg/minmax loop: separable per-agent
    branch DP over the resolved candidate rows."""
    resolved = _full_ext_resolve(
        problem, ext, n_nodes, owned, owner_vagent, aux, x0_full, layer0, branched,
        wp_template, node_active=node_active)
    return _ext_full_avg_minmax_score(
        objective, ext, P_pred, agent_of_node, owned, owner_vagent, aux,
        x0_of, sliced_of, resolved, best, best_sol)


def _ext_full_avg_minmax_score(objective, ext, P_pred, agent_of_node, owned, owner_vagent,
                               aux, x0_of, sliced_of, resolved, best, best_sol):
    """Score one already-resolved ext (`resolved` = a `_full_ext_resolve`
    tuple) for the full-resolve avg/minmax path -- split out so the batched
    caller can resolve every ext in bulk and only score here."""
    agent_seq, _tvec, wp0, rows_by_node, node_owner_of = resolved
    per_agent, choice = {}, {}
    for j, seq in agent_seq.items():
        c, ch = _branch_viterbi_wp(seq, rows_by_node, wp0, x0_of[j], sliced_of[j])
        per_agent[j] = c
        choice.update(ch)
    score = (sum(per_agent.values()) if objective == "avg"
             else max(per_agent.values(), default=0.0))
    if score < best:
        wp_sel = _select_wp(ext, rows_by_node, wp0, choice)
        arr = _makespan_forward_wp(ext, P_pred, agent_of_node, wp_sel, x0_of, sliced_of)
        br = {(n, node_owner_of[n][1]): int(choice.get(n, 0)) for n in rows_by_node}
        best = score
        best_sol = (br, owner_vagent.copy(), aux, arr, agent_seq, per_agent)
    return best, best_sol


def _ext_full_makespan(problem, ext, P_pred, agent_of_node, n_nodes, owned, owner_vagent, aux,
                       x0_full, x0_of, sliced_of, layer0, branched, wp_template, best, best_sol,
                       node_active=None):
    """One `ext` of the full-resolve makespan loop: branch choice couples
    with cross-agent waiting, so enumerate combos over per-agent cost
    matrices built once from the cached rows."""
    resolved = _full_ext_resolve(
        problem, ext, n_nodes, owned, owner_vagent, aux, x0_full, layer0, branched,
        wp_template, node_active=node_active)
    return _ext_full_makespan_score(
        ext, P_pred, agent_of_node, owned, owner_vagent, aux, x0_of, sliced_of,
        resolved, best, best_sol)


def _ext_full_makespan_score(ext, P_pred, agent_of_node, owned, owner_vagent, aux,
                             x0_of, sliced_of, resolved, best, best_sol):
    """Score one already-resolved ext for the full-resolve makespan path."""
    agent_seq, _tvec, wp0, rows_by_node, node_owner_of = resolved
    et, dt, bc = {}, {}, {}
    for j, seq in agent_seq.items():
        rj = {n: (rows_by_node[n] if n in rows_by_node else wp0[n][None, :]) for n in seq}
        bc[j] = {n: rj[n].shape[0] for n in seq}
        dt[j] = {n: np.array([sliced_of[j](x0_of[j], r) for r in rj[n]]) for n in seq}
        et[j] = {(u, v): np.array([[sliced_of[j](a, b) for b in rj[v]] for a in rj[u]])
                 for u, v in zip(seq, seq[1:])}
    pnodes = [(n, j) for n in ext for j in agent_of_node.get(n, [])]
    for combo in itertools.product(*[range(bc[j][n]) for (n, j) in pnodes]):
        b_of = dict(zip(pnodes, combo))
        arr = _makespan_forward(ext, P_pred, agent_seq, agent_of_node, b_of, et, dt)
        ms = max(arr.values(), default=0.0)
        if ms < best:
            per_agent = {}
            for j, seq in agent_seq.items():
                pc = dt[j][seq[0]][b_of[(seq[0], j)]] if seq else 0.0
                for u, v in zip(seq, seq[1:]):
                    pc += et[j][(u, v)][b_of[(u, j)], b_of[(v, j)]]
                per_agent[j] = float(pc)
            br = {(n, node_owner_of[n][1]): int(b_of[(n, j)])
                  for (n, j) in pnodes if n in node_owner_of}
            best = ms
            best_sol = (br, owner_vagent.copy(), aux, arr, agent_seq, per_agent)
    return best, best_sol


# ----------------------------------------------------------------------------
# Vectorized (JAX) non-full ext loop.
#
# The scalar `_ext_nonfull_*` bodies above stay the reference. These score
# EVERY linear extension (and, for makespan, every branch combo) in one
# batched vmap, pick the winner, and hand that single ext back to the scalar
# body for an exact byte-identical reconstruction of `best_sol`. So the fast
# path only ever REPLACES the O(#exts x #combos) Python loop with one kernel
# launch; it never changes which solution is returned.
# ----------------------------------------------------------------------------

# OFF by default: on the real dual_ur5e scenes the per-(assignment, aux) vmap
# trace/dispatch overhead is a 4-9x REGRESSION vs the scalar per-ext loop
# (block_stacking/avg 106ms -> 417ms; tabletop/makespan 14ms -> 124ms). The
# fully-vectorized replacement is dp_master_jax.make_dp_master_jax
# (DpMasterWaypointSolver(dp_backend="jax")); these flags stay for A/B / a
# future size-gated reuse.
_DP_MASTER_BATCH_NONFULL = False
_DP_MASTER_BATCH_FULL = False
_DP_MASTER_BATCH_CAP = 200_000    # #exts x #combos above which we stay scalar


def _ensure_x64():
    import jax
    if not jax.config.read("jax_enable_x64"):
        jax.config.update("jax_enable_x64", True)


_BIG = 1e12


def _pack_nonfull_tables(n_agents, n_nodes, owned, edge_tables, depot_tables,
                         branch_counts):
    """Dense `(EDGE, DEPOT, KC, max_k)` from the per-agent dict tables:
      EDGE  (n_agents, n_nodes, n_nodes, max_k, max_k)  -- `_BIG` where unset
      DEPOT (n_agents, n_nodes, max_k)                  -- `_BIG` where unset
      KC    (n_agents, n_nodes) int                     -- branch count, >=1
    so the kernels are pure gather + reduce."""
    max_k = 1
    for j in owned:
        for k in branch_counts[j].values():
            max_k = max(max_k, int(k))
    EDGE = np.full((n_agents, n_nodes, n_nodes, max_k, max_k), _BIG)
    DEPOT = np.full((n_agents, n_nodes, max_k), _BIG)
    KC = np.ones((n_agents, n_nodes), dtype=np.int32)
    for j in owned:
        for n, k in branch_counts[j].items():
            KC[j, int(n)] = int(k)
        for n, d in depot_tables[j].items():
            d = np.asarray(d, float)
            DEPOT[j, int(n), :d.shape[0]] = d
        for (u, v), m in edge_tables[j].items():
            m = np.asarray(m, float)
            EDGE[j, int(u), int(v), :m.shape[0], :m.shape[1]] = m
    return EDGE, DEPOT, KC, max_k


def _agent_stops(mat, n_valid, owned, n_agents):
    """`(STOPS (M, n_agents, L) int, NSTOP (M, n_agents) int)` -- for each
    (extension, agent), the agent's ordered subsequence of that extension,
    zero-padded to `L`."""
    M, L = mat.shape
    STOPS = np.zeros((M, n_agents, L), dtype=np.int32)
    NSTOP = np.zeros((M, n_agents), dtype=np.int32)
    osets = {j: set(int(n) for n in owned.get(j, ())) for j in range(n_agents)}
    for m in range(n_valid):
        row = [int(n) for n in mat[m]]
        for j in range(n_agents):
            s = [n for n in row if n in osets[j]]
            STOPS[m, j, :len(s)] = s
            NSTOP[m, j] = len(s)
    return STOPS, NSTOP


def _branch_combos(remaining, owned, KC, n_agents):
    """`(pnodes, BR_batch)` -- `pnodes` the list of branched `(node, agent)`
    pairs, `BR_batch` an `(C, n_nodes, n_agents)` int tensor with every
    combination of their branch indices (0 elsewhere)."""
    n_nodes = KC.shape[1]
    pnodes = [(int(n), j) for n in remaining for j in range(n_agents)
              if int(n) in owned.get(j, ()) and KC[j, int(n)] > 1]
    ranges = [range(int(KC[j, n])) for (n, j) in pnodes]
    combos = list(itertools.product(*ranges)) if pnodes else [()]
    BR = np.zeros((len(combos), n_nodes, n_agents), dtype=np.int32)
    for c, combo in enumerate(combos):
        for i, (n, j) in enumerate(pnodes):
            BR[c, n, j] = combo[i]
    return pnodes, BR


def _viterbi_cost_jax(stops, nstop, EDGE_j, DEPOT_j):
    """min routed-path cost for one agent's padded `stops` (0 if `nstop==0`).
    Forward min-DP only -- the scalar body redoes the argmin backtrace on the
    winning ext."""
    import jax.numpy as jnp
    from jax import lax
    L = stops.shape[0]

    def step(cost, s):
        v = stops[s]
        prev_v = stops[jnp.maximum(s - 1, 0)]
        edge_c = jnp.min(cost[:, None] + EDGE_j[prev_v, v], axis=0)
        new = jnp.where(s == 0, DEPOT_j[v], edge_c)
        return jnp.where(s < nstop, new, cost), None

    cost, _ = lax.scan(step, jnp.zeros(EDGE_j.shape[-1]), jnp.arange(L))
    return jnp.where(nstop == 0, 0.0, jnp.min(cost))


def _makespan_arr_jax(ext_row, BR, EDGE, DEPOT, OWN, PRED, n_agents):
    """DAG forward pass -> `arr` (n_nodes,) for one extension `ext_row` (a
    permutation of the remaining node ids) and branch tensor `BR`
    (n_nodes, n_agents)."""
    import jax.numpy as jnp
    from jax import lax
    ag = jnp.arange(n_agents)
    NEG = -_BIG

    def step(carry, v):
        arr, prev_own, prev_br = carry
        t = jnp.maximum(0.0, jnp.max(jnp.where(PRED[v], arr, NEG)))
        has_prev = prev_own >= 0
        u_safe = jnp.where(has_prev, prev_own, 0)
        bv = BR[v]
        edge_leg = arr[u_safe] + EDGE[ag, u_safe, v, prev_br, bv]
        leg = jnp.where(has_prev, edge_leg, DEPOT[ag, v, bv])
        own_v = OWN[v]
        t = jnp.maximum(t, jnp.max(jnp.where(own_v, leg, NEG)))
        arr = arr.at[v].set(t)
        return (arr, jnp.where(own_v, v, prev_own),
                jnp.where(own_v, bv, prev_br)), None

    init = (jnp.zeros(PRED.shape[0]), -jnp.ones(n_agents, jnp.int32),
            jnp.zeros(n_agents, jnp.int32))
    (arr, _, _), _ = lax.scan(step, init, ext_row)
    return arr


def _ext_nonfull_batched(objective, mat, n_valid, remaining, P, P_pred, agent_of_node,
                         owned, owner_vagent, aux, agent_keys, edge_tables, depot_tables,
                         branch_counts, n_agents, n_nodes, best, best_sol, max_branch_combos):
    """Score all `n_valid` extensions of `mat` in one batched kernel, then
    reconstruct the winner exactly via the scalar `_ext_nonfull_*` body.
    Returns the (possibly updated) `(best, best_sol)`, or `None` to signal
    'fall back to the scalar per-ext loop' (batch too large / degenerate)."""
    import jax
    import jax.numpy as jnp
    _ensure_x64()
    M, L = mat.shape
    EDGE, DEPOT, KC, _mk = _pack_nonfull_tables(
        n_agents, n_nodes, owned, edge_tables, depot_tables, branch_counts)
    OWN = np.zeros((n_nodes, n_agents), dtype=bool)
    for j, ns in owned.items():
        for n in ns:
            OWN[int(n), j] = True
    PRED = np.zeros((n_nodes, n_nodes), dtype=bool)
    for (u, v) in P:
        PRED[int(v), int(u)] = True
    rem_ids = np.asarray(sorted(int(n) for n in remaining), dtype=np.int32)
    EXTS = jnp.asarray(mat)

    if objective in ("avg", "minmax"):
        STOPS, NSTOP = _agent_stops(mat, n_valid, owned, n_agents)
        f_ag = jax.vmap(lambda st, ns, Ej, Dj: _viterbi_cost_jax(st, ns, Ej, Dj),
                        in_axes=(0, 0, 0, 0))
        f_all = jax.vmap(f_ag, in_axes=(0, 0, None, None))
        costs = np.asarray(f_all(jnp.asarray(STOPS), jnp.asarray(NSTOP),
                                 jnp.asarray(EDGE), jnp.asarray(DEPOT)))  # (M, n_agents)
        score = costs.sum(1) if objective == "avg" else costs.max(1)
        score = np.where(np.arange(M) < n_valid, score, np.inf)
        m_star = int(np.argmin(score))
        if not np.isfinite(score[m_star]) or score[m_star] >= best:
            return best, best_sol
        return _ext_nonfull_avg_minmax(
            objective, tuple(int(n) for n in mat[m_star]), P_pred, agent_of_node,
            owned, owner_vagent, aux, agent_keys, edge_tables, depot_tables,
            branch_counts, best, best_sol)

    # makespan
    pnodes, BR = _branch_combos(remaining, owned, KC, n_agents)
    C = BR.shape[0]
    if n_valid * C > _DP_MASTER_BATCH_CAP or C > max_branch_combos:
        return None
    kern = lambda ext_row, br: jnp.max(_makespan_arr_jax(
        ext_row, br, jnp.asarray(EDGE), jnp.asarray(DEPOT),
        jnp.asarray(OWN), jnp.asarray(PRED), n_agents)[rem_ids])
    grid = jax.vmap(jax.vmap(kern, in_axes=(None, 0)), in_axes=(0, None))
    ms = np.asarray(grid(EXTS, jnp.asarray(BR)))  # (M, C)
    ms = np.where((np.arange(M) < n_valid)[:, None], ms, np.inf)
    m_star = int(np.argmin(ms) // C)
    if not np.isfinite(ms.min()) or ms.min() >= best:
        return best, best_sol
    return _ext_nonfull_makespan(
        tuple(int(n) for n in mat[m_star]), P_pred, agent_of_node, owned,
        owner_vagent, aux, agent_keys, edge_tables, depot_tables, branch_counts,
        best, best_sol)


def _ecf_traceable(sliced_fn, state_dim):
    """True iff `sliced_fn` (one agent's `_agent_sliced_cost_fn`) can run
    under `jax.vmap` -- the full-resolve scoring kernels need a jnp cost fn.
    A `float(...)`-wrapped or `np.asarray`-based fn fails here and the caller
    stays on the scalar per-ext scorer."""
    import jax
    import jax.numpy as jnp
    try:
        jax.eval_shape(lambda a, b: jnp.asarray(sliced_fn(a, b)),
                       jnp.zeros(state_dim), jnp.zeros(state_dim))
        return True
    except Exception:
        return False


def _ext_full_batched(objective, mat, n_valid, remaining, P, P_pred, agent_of_node,
                      owned, owner_vagent, aux, x0_of, sliced_of, n_agents, n_nodes,
                      state_dim, resolved_list, wp0_all, rows_all, best, best_sol,
                      max_branch_combos):
    """Score every batched-resolved extension in one vmapped kernel (dense
    per-ext EDGE/DEPOT built from the resolved candidate rows via the jnp
    `sliced_of`), pick the winner, reconstruct it exactly via the scalar
    `_ext_full_*_score` body. `None` -> caller stays on the scalar scorer
    (cost fn not jnp-traceable, or batch too large)."""
    import jax
    import jax.numpy as jnp
    _ensure_x64()
    if not _ecf_traceable(sliced_of[0], state_dim):
        return None
    M = mat.shape[0]
    exts = [tuple(int(n) for n in mat[m]) for m in range(M)]

    max_k = max([1] + [rows_all[n].shape[1] for n in rows_all])
    CAND = np.repeat(np.asarray(wp0_all)[:, :, None, :], max_k, axis=2)  # (M,N,max_k,S)
    KC_node = np.ones(n_nodes, dtype=np.int32)
    for n, r in rows_all.items():
        k = r.shape[1]
        KC_node[int(n)] = k
        CAND[:, int(n), :k, :] = r
        CAND[:, int(n), k:, :] = r[:, :1, :]

    OWN = np.zeros((n_nodes, n_agents), dtype=bool)
    for j, ns in owned.items():
        for n in ns:
            OWN[int(n), j] = True
    KC = np.where(OWN.T, KC_node[None, :], 1).astype(np.int32)  # (J, N)
    PRED = np.zeros((n_nodes, n_nodes), dtype=bool)
    for (u, v) in P:
        PRED[int(v), int(u)] = True
    rem_ids = np.asarray(sorted(int(n) for n in remaining), dtype=np.int32)
    X0 = np.stack([np.asarray(x0_of[j], dtype=float) for j in range(n_agents)])  # (J,S)

    CANDj = jnp.asarray(CAND)
    validk = jnp.asarray(np.arange(max_k)[None, :] < KC_node[:, None])  # (N, max_k)

    def tables_j(j):
        sj = sliced_of[j]
        x0j = jnp.asarray(X0[j])
        dep = jax.vmap(jax.vmap(jax.vmap(lambda r: jnp.asarray(sj(x0j, r)))))(CANDj)  # (M,N,K)
        gk = jax.vmap(jax.vmap(lambda ru, rv: jnp.asarray(sj(ru, rv)), (None, 0)), (0, None))
        gn = jax.vmap(jax.vmap(gk, (None, 0)), (0, None))
        edg = jax.vmap(gn)(CANDj, CANDj)  # (M,N,N,K,K)
        dep = jnp.where(validk[None], dep, _BIG)
        edg = jnp.where(validk[None, :, None, :, None], edg, _BIG)
        edg = jnp.where(validk[None, None, :, None, :], edg, _BIG)
        return dep, edg

    DEP, EDG = zip(*(tables_j(j) for j in range(n_agents)))
    DEPOT = jnp.stack(DEP, axis=1)   # (M, J, N, K)
    EDGE = jnp.stack(EDG, axis=1)    # (M, J, N, N, K, K)

    if objective in ("avg", "minmax"):
        STOPS, NSTOP = _agent_stops(mat, n_valid, owned, n_agents)
        f_all = jax.vmap(jax.vmap(_viterbi_cost_jax, in_axes=(0, 0, 0, 0)),
                         in_axes=(0, 0, 0, 0))
        costs = np.asarray(f_all(jnp.asarray(STOPS), jnp.asarray(NSTOP), EDGE, DEPOT))
        score = costs.sum(1) if objective == "avg" else costs.max(1)
        score = np.where(np.arange(M) < n_valid, score, np.inf)
        m_star = int(np.argmin(score))
        if not np.isfinite(score[m_star]) or score[m_star] >= best:
            return best, best_sol
        return _ext_full_avg_minmax_score(
            objective, exts[m_star], P_pred, agent_of_node, owned, owner_vagent,
            aux, x0_of, sliced_of, resolved_list[m_star], best, best_sol)

    pnodes, BR = _branch_combos(remaining, owned, KC, n_agents)
    C = BR.shape[0]
    if n_valid * C > _DP_MASTER_BATCH_CAP or C > max_branch_combos:
        return None
    OWNj, PREDj = jnp.asarray(OWN), jnp.asarray(PRED)

    def kern(ext_row, br, edge_m, depot_m):
        return jnp.max(_makespan_arr_jax(ext_row, br, edge_m, depot_m,
                                         OWNj, PREDj, n_agents)[rem_ids])

    grid = jax.vmap(jax.vmap(kern, in_axes=(None, 0, None, None)),
                    in_axes=(0, None, 0, 0))
    ms = np.asarray(grid(jnp.asarray(mat), jnp.asarray(BR), EDGE, DEPOT))  # (M, C)
    ms = np.where((np.arange(M) < n_valid)[:, None], ms, np.inf)
    if not np.isfinite(ms.min()) or ms.min() >= best:
        return best, best_sol
    m_star = int(np.argmin(ms) // C)
    return _ext_full_makespan_score(
        exts[m_star], P_pred, agent_of_node, owned, owner_vagent, aux,
        x0_of, sliced_of, resolved_list[m_star], best, best_sol)


def solve_dp_master(problem, candidates, wp_template, x0_by_agent, instances,
                    ordering_edges=None, edge_cost_fn=None, objective="makespan",
                    max_assign_combos=4096, max_orders=20000, x0_full=None,
                    max_branch_combos=4096, anchor=None):
    """Solve the discrete problem by enumerating (assignment, aux, global
    order) and running an exact branch DP / forward pass inside.  Return
    dict mirrors `solve_cpsat`: `status`, `objective`, `branch`
    (dict[(node, inst_key)] -> idx), `assignment` (dict[slot] -> agent),
    `aux` (dict[k] -> 0/1), `time` (dict[node] -> float), `routes`
    (dict[agent] -> [node,...]), `wall_time`.

    `ordering_edges`: `problem.ordering_edges` (default) -- list of
    `(u, v, gate)`, `gate is None` for a hard edge else a
    `formula_compiler.compile_condition` closure `gate(owner_vagent,
    cond_binary) -> bool`.

    `anchor`: optional `evolutionary_waypoint_solver.problem.AnchorState`
    (node_active, anchor_wp, var_committed, var_anchor) -- the MPC-cycle
    commitment state. When given, the discrete problem is solved over the
    FUTURE subgraph only:
      * assignment slots with `var_committed[slot]` are pinned to
        `var_anchor[slot]`; the enumeration products only over the free
        slots;
      * linear extensions are generated over the remaining vertices only
        (`node_active`), edges of `P` touching a passed node dropped -- an
        already-committed predecessor imposes no constraint on the order the
        remaining work is done in;
      * passed nodes are excluded from every route / branch DP / forward
        pass, and from the returned `time` / `branch` / `routes` dicts (the
        caller keeps their committed rows);
      * `node_active` is threaded into the per-schedule projection resolve
        so passed columns freeze and the gate machinery decodes ranks
        against the committed set.
    Depot legs start from the live `x0_by_agent` / `x0_full` (each agent's
    current real position after its committed prefix), so no separate
    committed-position bookkeeping is needed here.
    """
    import time as _t
    t0 = _t.perf_counter()
    if objective not in ("makespan", "minmax", "avg"):
        raise ValueError(f"unknown objective {objective!r}")
    if ordering_edges is None:
        ordering_edges = problem.ordering_edges
    n_nodes = problem.n_nodes
    n_agents = problem.n_agents
    n_var = problem.n_variables
    n_cond = problem.n_cond_vars

    if anchor is not None:
        node_active = np.asarray(anchor.node_active, dtype=bool)
        var_committed = np.asarray(anchor.var_committed, dtype=bool)
        var_anchor = np.asarray(anchor.var_anchor, dtype=int)
    else:
        node_active = np.ones(n_nodes, dtype=bool)
        var_committed = np.zeros(n_var, dtype=bool)
        var_anchor = np.zeros(n_var, dtype=int)
    remaining = [n for n in range(n_nodes) if node_active[n]]
    remaining_set = set(remaining)

    if isinstance(x0_by_agent, dict):
        x0_of = dict(x0_by_agent)
    else:
        x0_of = {a: np.asarray(x0_by_agent) for a in range(n_agents)}

    ecf = edge_cost_fn if edge_cost_fn is not None else getattr(problem, "edge_cost_fn", None)

    full = _needs_full_resolve(problem)
    if full and x0_full is None:
        # Best effort: agent columns from x0_of, objects left at 0. A caller
        # with real object state should pass x0_full=(state_dim,) explicitly
        # -- the gated depot projections read x0[object segment].
        x0_full = np.zeros(problem.state_dim)
        for j, row in x0_of.items():
            x0_full = x0_full + np.asarray(row)
    if full:
        sliced_of = {j: _agent_sliced_cost_fn(ecf, j, problem.dim) for j in range(n_agents)}
        proj_entries = list(problem.projections)
        # Two layers: branch-FREE entries (gated stationary / rigid-carry /
        # object pins -- resolved once per order) and BRANCHED ones (the
        # analytic-IK ProjOperators -- one branch choice per node, resolved
        # per branch against the branch-free layer's wp). A branched entry
        # reading another branched entry's column would couple the branch DP
        # -- not implemented (block stacking is a flat 2-layer chain).

        # EXAMPLE: an analytic IK proj on one robot that constraints him to be
        # in one of a few places, and then another analytic IK proj that
        # constrains another robot to be next to the first robot. Maybe it is
        # okay to keep it out of scope, but I would like to leave nothing missing

        layer0 = tuple(e for e in proj_entries if e.discrete_params == 1)
        branched = [e for e in proj_entries if e.discrete_params > 1]
        # A branched entry reading ANOTHER branched entry's pinned column
        # would couple the per-track branch DP (its candidate rows depend on
        # the upstream branch choice) -- not implemented. A self-read (an IK
        # entry's own documented-unused FK placeholder) doesn't count.
        for e in branched:
            others = set().union(*(o.write_cols for o in branched if o is not e))
            if e.read_cols & others:
                raise NotImplementedError(
                    "dp_master full-resolve: a multi-branch projection reads a "
                    "column another multi-branch projection pins -- the coupled "
                    "branch DP that needs is not implemented")
        makespan_combo_cap = max(max_branch_combos, 100000)
        n_makespan_combos = int(np.prod([e.discrete_params for e in branched], dtype=object)) if branched else 1
        if objective == "makespan" and n_makespan_combos > makespan_combo_cap:
            raise NotImplementedError(
                f"dp_master full-resolve makespan: {n_makespan_combos} branch combos "
                f"> {makespan_combo_cap} -- makespan couples branch choice with "
                "cross-agent waiting, so it still enumerates combos (over CACHED "
                "per-branch rows, but still exponential); avg/minmax use the "
                "separable per-track DP")

    # 2a: enumerate only the assignment slots not already committed by the
    # anchor -- a committed slot is pinned to `var_anchor[slot]`.
    free_slots = [s for s in range(n_var) if not var_committed[s]]
    n_assign = n_agents ** len(free_slots) if free_slots else 1
    if n_assign > max_assign_combos:
        raise NotImplementedError(
            f"dp_master: {n_agents}**{len(free_slots)} = {n_assign} assignment combos "
            f"> max_assign_combos={max_assign_combos}")

    free_combos = (list(itertools.product(range(n_agents), repeat=len(free_slots)))
                   if free_slots else [()])
    aux_combos = list(itertools.product((0, 1), repeat=n_cond)) if n_cond else [()]

    best = np.inf
    best_sol = None

    for free_vals in free_combos:
        owner_vagent = var_anchor.copy()
        for s, val in zip(free_slots, free_vals):
            owner_vagent[s] = val
        owned = _agent_owned_nodes(problem, instances, owner_vagent)
        # Restrict to the future subgraph: a passed node is never re-routed.
        owned = {j: {n for n in ns if n in remaining_set} for j, ns in owned.items()}
        # candidate-dict key per (agent, node)
        agent_keys = {j: {n: _inst_key(problem, instances, n, j, owner_vagent) for n in ns}
                      for j, ns in owned.items()}

        for aux in aux_combos:
            P = _resolve_precedence(problem, ordering_edges, owner_vagent, aux)
            # Drop edges touching a passed node -- an already-committed
            # predecessor imposes no constraint on the remaining order.
            P = {(u, v) for (u, v) in P if u in remaining_set and v in remaining_set}
            exts = _linear_extensions(remaining, P, max_orders)
            if exts is None:
                continue  # cyclic precedence for this (A, aux)
            P_pred = {v: set() for v in range(n_nodes)}
            for u, v in P:
                P_pred[v].add(u)

            # per-agent tables (keyed by the single candidate key each agent
            # uses; all its nodes share one instance kind in supported scenes)
            # -- skipped entirely for the full-resolve path, which prices off
            # apply_projections directly.
            edge_tables, depot_tables, branch_counts = {}, {}, {}
            if not full:
                for j, ns in owned.items():
                    ns = sorted(ns)
                    key0 = next(iter(agent_keys[j].values())) if agent_keys[j] else j
                    et, dt, bc = _agent_tables(problem, candidates, wp_template,
                                               x0_of[j], ns, key0, ecf)
                    edge_tables[j], depot_tables[j], branch_counts[j] = et, dt, bc

            agent_of_node = {}
            for j, ns in owned.items():
                for n in ns:
                    agent_of_node.setdefault(n, []).append(j)

            # Non-full: score every extension (and branch combo) in one
            # batched kernel, reconstruct the winner via the scalar body.
            if not full and _DP_MASTER_BATCH_NONFULL and exts:
                mat, n_valid, _M = _extensions_matrix(remaining, P, max_orders)
                r = _ext_nonfull_batched(
                    objective, mat, n_valid, remaining, P, P_pred, agent_of_node,
                    owned, owner_vagent, aux, agent_keys, edge_tables, depot_tables,
                    branch_counts, n_agents, n_nodes, best, best_sol, max_branch_combos)
                if r is not None:
                    best, best_sol = r
                    continue

            # Full-resolve: resolve every extension's projection layers in
            # one batched `apply_projections` launch (per branched entry),
            # then score each ext with the scalar body.
            if full and _DP_MASTER_BATCH_FULL and exts:
                mat, n_valid, _M = _extensions_matrix(remaining, P, max_orders)
                resolved_list, _now, wp0_all, rows_all = _full_ext_resolve_batched(
                    problem, mat, n_valid, n_nodes, owned, owner_vagent, aux, x0_full,
                    layer0, branched, wp_template, node_active=node_active)
                r = _ext_full_batched(
                    objective, mat, n_valid, remaining, P, P_pred, agent_of_node,
                    owned, owner_vagent, aux, x0_of, sliced_of, n_agents, n_nodes,
                    problem.state_dim, resolved_list, wp0_all, rows_all,
                    best, best_sol, max_branch_combos)
                if r is not None:
                    best, best_sol = r
                    continue
                # cost fn not jnp-traceable / batch too big -- score the
                # (still batch-resolved) exts with the scalar body.
                for m in range(n_valid):
                    ext = tuple(int(n) for n in mat[m])
                    if objective in ("avg", "minmax"):
                        best, best_sol = _ext_full_avg_minmax_score(
                            objective, ext, P_pred, agent_of_node, owned, owner_vagent,
                            aux, x0_of, sliced_of, resolved_list[m], best, best_sol)
                    else:
                        best, best_sol = _ext_full_makespan_score(
                            ext, P_pred, agent_of_node, owned, owner_vagent, aux,
                            x0_of, sliced_of, resolved_list[m], best, best_sol)
                continue

            # One `ext` at a time -- dispatch to the loop body for this
            # (full/non-full) x (avg-minmax/makespan) combination. Both
            # switches are loop-invariant across the whole enumeration, so
            # the per-`ext` branch here is free.
            for ext in exts:
                if full and objective in ("avg", "minmax"):
                    best, best_sol = _ext_full_avg_minmax(
                        problem, objective, ext, P_pred, agent_of_node, n_nodes, owned,
                        owner_vagent, aux, x0_full, x0_of, sliced_of, layer0, branched,
                        wp_template, best, best_sol, node_active=node_active)
                elif full:
                    best, best_sol = _ext_full_makespan(
                        problem, ext, P_pred, agent_of_node, n_nodes, owned, owner_vagent,
                        aux, x0_full, x0_of, sliced_of, layer0, branched, wp_template,
                        best, best_sol, node_active=node_active)
                elif objective in ("avg", "minmax"):
                    best, best_sol = _ext_nonfull_avg_minmax(
                        objective, ext, P_pred, agent_of_node, owned, owner_vagent, aux,
                        agent_keys, edge_tables, depot_tables, branch_counts, best, best_sol)
                else:
                    best, best_sol = _ext_nonfull_makespan(
                        ext, P_pred, agent_of_node, owned, owner_vagent, aux, agent_keys,
                        edge_tables, depot_tables, branch_counts, best, best_sol)

    wall = _t.perf_counter() - t0
    if best_sol is None:
        return dict(status="INFEASIBLE", objective=None, branch={}, assignment={},
                    aux={}, time={}, routes={}, agent_cost={}, wall_time=wall)

    br, ov, aux, arr, agent_seq, per_agent = best_sol
    return dict(
        status="OPTIMAL",
        objective=float(best),
        branch={k: int(v) for k, v in br.items()},
        assignment={s: int(ov[s]) for s in range(n_var)},
        aux={k: int(aux[k]) for k in range(n_cond)},
        time={int(n): float(t) for n, t in arr.items()},
        t_end=float(max(arr.values(), default=0.0)),
        agent_cost={int(j): float(c) for j, c in per_agent.items()},
        routes={int(j): list(seq) for j, seq in agent_seq.items()},
        wall_time=wall,
    )

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


def _linear_extensions(n_nodes, P, cap):
    """All total orders over `range(n_nodes)` consistent with `P` (all
    topological sorts).  Returns `list[tuple[int]]`; raises if there would
    be more than `cap`.  `None` if `P` has a cycle."""
    succ = [[] for _ in range(n_nodes)]
    indeg = [0] * n_nodes
    for u, v in P:
        succ[u].append(v)
        indeg[v] += 1
    if sum(indeg) and all(d > 0 for d in indeg):
        return None  # every node has a predecessor -> cycle

    out = []
    order = []
    used = [False] * n_nodes
    deg = indeg[:]

    def rec():
        if len(out) > cap:
            raise NotImplementedError(
                f"dp_master: precedence DAG has > {cap} linear extensions "
                "-- too loosely ordered for order enumeration (use Held-Karp "
                "per agent for avg/minmax, or CP-SAT)")
        if len(order) == n_nodes:
            out.append(tuple(order))
            return
        ready = [i for i in range(n_nodes) if not used[i] and deg[i] == 0]
        if not ready:
            return  # cycle among the remainder
        for i in ready:
            used[i] = True
            order.append(i)
            for j in succ[i]:
                deg[j] -= 1
            rec()
            for j in succ[i]:
                deg[j] += 1
            order.pop()
            used[i] = False

    rec()
    return out


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
                         only_entries=None):
    """`(pop, n_nodes, state_dim)` wp for one enumerated schedule: the
    (`only_entries` subset of) projections spliced in by apply_projections
    against this assignment (`owner_vagent` one-hots), aux, per-member branch
    vectors and the order-derived `t` (shared across the pop). jitted +
    cached per (problem, only_entries). `wp_pop` and `proj_branch_pop` carry
    the population axis; everything else is broadcast."""
    import jax.numpy as jnp
    from ..evolutionary_waypoint_solver.problem import jit_apply_projections
    pop = wp_pop.shape[0]
    n_var, n_agents = problem.n_variables, problem.n_agents
    assign = np.zeros((pop, n_var, n_agents))
    for s in range(n_var):
        assign[:, s, int(owner_vagent[s])] = 1.0
    cb = (np.broadcast_to(np.asarray(aux, dtype=float), (pop, problem.n_cond_vars))
          if problem.n_cond_vars else np.zeros((pop, 0)))
    out = jit_apply_projections(problem, only_entries=only_entries)(
        jnp.asarray(wp_pop), jnp.zeros((pop, problem.n_psi)),
        jnp.asarray(proj_branch_pop), jnp.asarray(problem.params),
        jnp.asarray(assign), jnp.asarray(cb),
        jnp.broadcast_to(jnp.asarray(t_vec, dtype=float), (pop, problem.n_nodes)),
        jnp.ones((problem.n_nodes,), dtype=bool), jnp.asarray(x0_full))
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


def solve_dp_master(problem, candidates, wp_template, x0_by_agent, instances,
                    ordering_edges=None, edge_cost_fn=None, objective="makespan",
                    max_assign_combos=4096, max_orders=20000, x0_full=None,
                    max_branch_combos=4096):
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

    n_assign = n_agents ** n_var if n_var else 1
    if n_assign > max_assign_combos:
        raise NotImplementedError(
            f"dp_master: {n_agents}**{n_var} = {n_assign} assignment combos "
            f"> max_assign_combos={max_assign_combos}")
    assign_combos = (list(itertools.product(range(n_agents), repeat=n_var))
                     if n_var else [()])
    aux_combos = list(itertools.product((0, 1), repeat=n_cond)) if n_cond else [()]

    best = np.inf
    best_sol = None

    for A in assign_combos:
        owner_vagent = np.asarray(A, dtype=int)
        owned = _agent_owned_nodes(problem, instances, owner_vagent)
        # candidate-dict key + sliced tables per agent
        agent_keys = {j: _inst_key(problem, instances, next(iter(ns)), j, owner_vagent)
                      if ns else j
                      for j, ns in owned.items()}
        agent_keys = {j: {n: _inst_key(problem, instances, n, j, owner_vagent) for n in ns}
                      for j, ns in owned.items()}

        for aux in aux_combos:
            P = _resolve_precedence(problem, ordering_edges, owner_vagent, aux)
            exts = _linear_extensions(n_nodes, P, max_orders)
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

            if full:
                # Per-schedule resolution, staged: (1) resolve the branch-
                # free projection layer once per order (gates see this
                # order's `t`); (2) resolve each branched (analytic-IK)
                # entry's k candidate rows against that layer; (3) per-agent
                # branch DP over those rows (avg/minmax) or a cached-row
                # combo enumeration (makespan).
                for ext in exts:
                    agent_seq = {j: [n for n in ext if n in owned.get(j, ())] for j in owned}
                    tvec = _order_t(ext, n_nodes)
                    wp0 = _resolve_schedule_wp(
                        problem, wp_template[None], x0_full, owner_vagent, aux,
                        np.zeros((1, problem.n_branch)), tvec, only_entries=layer0)[0]

                    rows_by_node = {}
                    node_owner_of = {}          # projected node -> (entry, owner-agent)
                    for e in branched:
                        k = e.discrete_params
                        pb = np.zeros((k, problem.n_branch))
                        pb[np.arange(k), e.branch_slice.start + np.arange(k)] = 1.0
                        res = _resolve_schedule_wp(
                            problem, np.broadcast_to(wp0[None], (k,) + wp0.shape),
                            x0_full, owner_vagent, aux, pb, tvec, only_entries=(e,))
                        rows_by_node[int(e.write_node)] = res[:, int(e.write_node), :]
                        # For a DYNAMIC (var_agent_q) entry the owner is
                        # whichever agent THIS assignment binds its variable
                        # to -- fixed inside this `for A in assign_combos`
                        # iteration, so the entry prices exactly like a
                        # static multi-branch one from here on.
                        node_owner_of[int(e.write_node)] = (e, entry_owner(problem, e, owner_vagent))

                    if objective in ("avg", "minmax"):
                        per_agent, choice = {}, {}
                        for j, seq in agent_seq.items():
                            c, ch = _branch_viterbi_wp(seq, rows_by_node, wp0, x0_of[j], sliced_of[j])
                            per_agent[j] = c
                            choice.update(ch)
                        score = (sum(per_agent.values()) if objective == "avg"
                                 else max(per_agent.values(), default=0.0))
                        if score < best:
                            wp_sel = _select_wp(ext, rows_by_node, wp0, choice)
                            arr = _makespan_forward_wp(ext, P_pred, agent_of_node, wp_sel,
                                                       x0_of, sliced_of)
                            br = {(n, node_owner_of[n][1]): int(choice.get(n, 0))
                                  for n in rows_by_node}
                            best = score
                            best_sol = (br, owner_vagent.copy(), aux, arr, agent_seq, per_agent)
                        continue

                    # makespan: branch choice couples with cross-agent
                    # waiting, so enumerate combos -- but over per-agent
                    # (node-pair) cost MATRICES built once from the cached
                    # rows (pure arithmetic per combo, reusing the fast
                    # path's `_makespan_forward`), not a wp rebuild.
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
                continue

            for ext in exts:
                agent_seq = {j: [n for n in ext if n in owned.get(j, ())]
                             for j in owned}

                if objective in ("avg", "minmax"):
                    per_agent = {}
                    br = {}
                    for j, seq in agent_seq.items():
                        c, ch = _branch_viterbi(seq, edge_tables[j], depot_tables[j],
                                                branch_counts[j])
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
                        best_sol = (dict(br), owner_vagent.copy(), aux, arr, agent_seq,
                                    per_agent)
                    continue

                # makespan: brute-force branch combos for this order
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

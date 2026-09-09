"""Monolithic exact CP-SAT (OR-Tools) solver for the discrete routing /
assignment / branch-selection problem that the logic-based Benders
decomposition's master will eventually be a relaxation of.

This is a faithful encoding of the written CP-SAT formulation (minimize
`t(n+1)`; per-agent routing indicators `Z`, agent assignment `A`,
per-(agent,node) discrete branch `B(j,v)`, node timings `t`), built on
OR-Tools' `AddCircuit` global constraint instead of spelling out flow
conservation + subtour elimination.

Correspondence to the formulation
---------------------------------
* `A(i,j)`  -> `assign_bool[slot]`: one BoolVar per real agent, `AddExactlyOne`.
              Created only for nodes carrying a `("var", slot)` routing
              instance (structure.node_instances); a `("fixed", agent)` node
              needs no variable.
* `B(j,v)`  -> `branch_bool[(v, inst_key)]`: one BoolVar per candidate row of
              node `v`'s instance owned by `inst_key` (an agent id for a
              fixed instance, `("var", slot)` for an assignable one),
              `AddExactlyOne`. Candidate rows come from
              structure.node_candidates, one per projection branch. Two arms
              of a handoff node each get their own `inst_key` and their own
              independent branch set.
* `Z^j_{ab}`-> arc literals in one `AddCircuit` per agent, over vertices
              expanded per (node, that agent's branch). Circuit index 0 is a
              synthetic depot; real depot->node arc costs come from that
              agent's own current state `x0[j]`; every node->depot return arc
              costs 0, turning the closed tour AddCircuit needs into the open
              path `depot -> first -> ... -> last`. Flow conservation, "visit
              each owned node once", single depot exit / single sink entry,
              and "no return to 0 / no leave from n+1" all fall out of
              AddCircuit + the generalized depot self-loop.
* `t(v)`    -> `time[v]`: one shared scaled-int per graph node (shared across
              agents -- this is what couples them). For each chosen routing
              arc `(u,i)->(v,j)` of any agent, `time[v] >= time[u] +
              round(delta_uv(B(u),B(v)) * SCALE)` reified on the arc literal
              (CP-SAT native reification, no big-M); for each chosen depot
              arc, `time[v] >= round(depot_cost * SCALE)`. `t(0)=0` is
              implicit. `s_{ab}` is omitted deliberately.
* `t(a) <= t(b)` for every ordering edge, including cross-agent: `time[a] <=
              time[b]`. Hard edges (problem.hard_edges) always; conditional
              edges (graph.conditional_ordering_map) gated on a BoolVar `g`
              reified `g <=> formula` by structure.compile_gate_cpsat -- an
              arbitrary And/Or/Not formula over assignment-equality atoms
              (`assignment_sym(i) == assignment_sym(j)` or `== <agent j>`,
              reified against `assign_bool`) and free auxiliary binaries
              `aux_bool` (one per graph.binary_cond_sym_vars -- e.g. the
              direction selector for disjunctive block ordering). A
              cross-agent edge forces the downstream agent to idle until the
              upstream node's time -- real waiting.
* `min t(n+1)` -> `T_end >= time[v]` for every node, `Minimize(T_end)`.
              Return legs cost 0, so `T_end == max_v time[v] == t(n+1)`.

`rank[v]` (a shared `AddAllDifferent` permutation, `rank[u] < rank[v]` on
hard edges and reified on every chosen node-to-node arc) is an OPTIONAL
redundant-propagation layer, `use_rank=` (default False). `time` is the
authoritative ordering; the AllDifferent permutation is pure overhead for a
makespan solve and measurably hurts loosely-ordered instances, so it's off
unless a caller finds it helps their graph shape.

Objective flag: `"makespan"` (default) minimizes `T_end`; `"minmax"`
minimizes `max_j agent_cost[j]` (the written first formulation's `L`,
ignores cross-agent waiting); `"avg"` minimizes `sum_j agent_cost[j]`.

Scaling: real costs are floats, CP-SAT is integer-only, so every cost is
`round(x * _SCALE)`. Fine for optimality (relative order is preserved);
bump `_SCALE` for finer resolution.

Assignable / multi-instance nodes: for each `slot` referenced by a `("var",
slot)` instance, `assign[slot][*]` BoolVars + `AddExactlyOne` (every agent
eligible -- structure.py carries no per-slot eligibility yet). A node with
such an instance is a candidate in every agent's circuit, its presence there
gated on `branch_bool AND assign_bool[slot][agent]` (`_and_lit`, a genuine
biconditional -- the self-loop literal must be its exact negation). Two
instances at the same node must resolve to different agents (per-node, not
per-instance, vertex expansion); raises otherwise.
"""

import time as _time
import weakref

from ortools.sat.python import cp_model
import numpy as np

from .structure import (
    build_edge_cost_table, build_depot_cost_table, conditional_edge_data,
    compile_gate_cpsat, node_instances, node_candidates, warm_start_wp,
)
from ..evolutionary_waypoint_solver.kernel import _euclidean_edge_cost


def _all_pairs(nodes):
    return [(u, v) for u in nodes for v in nodes if u != v]


# fn -> {(agent_id, dim): sliced closure}, weakref-keyed on the underlying
# edge_cost_fn so it evicts itself once that fn (and the problem holding it)
# is gone. `_agent_sliced_cost_fn` is called fresh every MPC cycle with the
# SAME `fn`; returning the SAME `sliced` object each time (rather than a
# fresh closure) lets `structure.build_edge_cost_table`/`build_depot_cost_
# table` cache the vmapped+jitted version of it across cycles too -- see
# their own `_batched_cost_fn`.
_sliced_cost_fn_cache = weakref.WeakKeyDictionary()


def _agent_sliced_cost_fn(edge_cost_fn, agent_id, dim):
    """Wraps `edge_cost_fn` (a single shared callable, or a per-agent
    list/tuple -- GraphOrderingRelaxed's edge_cost_fn contract) so it's
    called on just agent `agent_id`'s own `[agent_id*dim : (agent_id+1)*dim]`
    slice of each row. An agent's route cost only ever depends on its own
    columns; skipping this slice lets other agents'/objects' columns
    contaminate the distance.

    Plain array slicing (not `np.asarray(...)[...]`) so the result stays
    traceable under `jax.vmap`/`jax.jit` when `a`/`b` are jax arrays --
    `structure.py`'s cost-table builders batch every (a, b) pair across a
    whole table into one vmapped call rather than one eager call per pair."""
    fn = edge_cost_fn
    if isinstance(fn, (list, tuple)):
        fn = fn[agent_id]
    if fn is None:
        fn = _euclidean_edge_cost
    per_fn = _sliced_cost_fn_cache.setdefault(fn, {})
    key = (agent_id, dim)
    sliced = per_fn.get(key)
    if sliced is None:
        lo, hi = agent_id * dim, (agent_id + 1) * dim

        def sliced(a, b, lo=lo, hi=hi, fn=fn):
            return fn(a[lo:hi], b[lo:hi])

        per_fn[key] = sliced
    return sliced


def _and_lit(model, a, b, name):
    """A BoolVar `lit` fully reified as `lit <=> a AND b` (both directions --
    a self-loop literal must be the EXACT negation of this, not just an
    upper bound)."""
    lit = model.NewBoolVar(name)
    model.AddBoolAnd([a, b]).OnlyEnforceIf(lit)
    model.AddBoolOr([a.Not(), b.Not()]).OnlyEnforceIf(lit.Not())
    return lit


_SCALE = 1000


def _to_int(x):
    return int(round(float(x) * _SCALE))


def build_cpsat_model(problem, candidates, wp_template, x0_by_agent, instances,
                      cond_formulas=None, var_sym_ids=None, cond_sym_ids=None,
                      edge_cost_fn=None, objective="makespan", use_rank=False):
    """Build the CP-SAT model. See module docstring for the formulation.

    `candidates`: structure.node_candidates output -- `dict[node] ->
        dict[owner] -> NodeCandidates`.
    `instances`: structure.node_instances output -- `dict[node] -> list of
        (kind, agent_or_slot)`.
    `cond_formulas`, `var_sym_ids`, `cond_sym_ids`:
        structure.conditional_edge_data output -- raw conditional-ordering
        Formulas + the symbol-id maps compile_gate_cpsat needs. One free
        `aux_bool` BoolVar is created per `graph.binary_cond_sym_vars`.
    `x0_by_agent`: a single (state_dim,) row (broadcast) or dict[agent_id] ->
        (state_dim,) row.
    Returns `(model, handles)`.
    """
    if objective not in ("makespan", "minmax", "avg"):
        raise ValueError(f"unknown objective {objective!r}")
    model = cp_model.CpModel()
    n_nodes = problem.n_nodes
    n_agents = problem.n_agents
    dim = problem.dim

    if isinstance(x0_by_agent, dict):
        x0_of_agent = dict(x0_by_agent)
    else:
        x0_of_agent = {a: x0_by_agent for a in range(n_agents)}

    # -- per-node routing instances: dict[agent_id] -> dict[node] ->
    #    (inst_key, gate) -----------------------------------------------------
    # inst_key: the agent id (fixed instance) or ("var", slot). gate: None
    # (unconditional) or assign_bool[slot][agent] (assignable).
    var_slots = sorted({val for insts in instances.values()
                        for kind, val in insts if kind == "var"})
    assign_bool = {}
    for slot in var_slots:
        bs = [model.NewBoolVar(f"assign_{slot}_{a}") for a in range(n_agents)]
        model.AddExactlyOne(bs)
        assign_bool[slot] = bs

    # Nodes with no routing instance participate only in precedence (`time`
    # + ordering edges + T_end); they join no agent's circuit. spec.py
    # explicitly allows these (e.g. a "place" node whose position is set by
    # a transport edge, not a node constraint).
    agent_nodes = {}  # agent_id -> {node: (inst_key, gate)}
    precedence_only = []
    for node in range(n_nodes):
        insts = instances.get(node, [])
        if not insts:
            precedence_only.append(node)
            continue
        seen = {}
        for kind, val in insts:
            if kind == "fixed":
                per_agent = {val: (val, None)}
            elif kind == "var":
                per_agent = {a: (("var", val), assign_bool[val][a]) for a in range(n_agents)}
            else:
                raise NotImplementedError(f"node {node}: unknown instance kind {kind!r}")
            for agent_id, ik_gate in per_agent.items():
                if agent_id in seen:
                    raise NotImplementedError(
                        f"node {node}: agent {agent_id} reachable via more than one "
                        "instance -- needs per-instance vertex expansion")
                seen[agent_id] = ik_gate
        for agent_id, ik_gate in seen.items():
            agent_nodes.setdefault(agent_id, {})[node] = ik_gate

    # -- per-(node, inst_key) branch choice B(j,v) --------------------------
    def _cand_rows(node, inst_key):
        owner = inst_key if isinstance(inst_key, int) else None
        nc = candidates.get(node, {}).get(owner) if owner is not None else None
        return nc.rows if nc is not None else wp_template[node][None, :]

    branch_bool = {}
    branch_idx = {}
    inst_keys_of_node = {}
    for agent_id, node_map in agent_nodes.items():
        for node, (inst_key, _gate) in node_map.items():
            inst_keys_of_node.setdefault(node, set()).add(inst_key)
    for node, iks in inst_keys_of_node.items():
        for inst_key in iks:
            k = _cand_rows(node, inst_key).shape[0]
            bvars = [model.NewBoolVar(f"branch_{node}_{inst_key}_{b}") for b in range(k)]
            model.AddExactlyOne(bvars)
            branch_bool[(node, inst_key)] = bvars
            idx = model.NewIntVar(0, max(k - 1, 0), f"branch_idx_{node}_{inst_key}")
            for b, bv in enumerate(bvars):
                model.Add(idx == b).OnlyEnforceIf(bv)
            branch_idx[(node, inst_key)] = idx

    # -- shared topological order (redundant propagation; off by default --
    #    `time` + AddCircuit already enforce ordering, and the AllDifferent
    #    permutation is pure overhead for a makespan solve) ------------------
    rank = None
    if use_rank:
        rank = [model.NewIntVar(0, n_nodes - 1, f"rank_{v}") for v in range(n_nodes)]
        model.AddAllDifferent(rank)
        for u, v in problem.hard_edges:
            model.Add(rank[u] < rank[v])

    # -- shared metric timing --------------------------------------------
    # Generous upper bound: everyone visits every node via the most
    # expensive candidate pair, plus the most expensive depot leg.
    all_pair_max = 0.0
    all_depot_max = 0.0
    slice_fns = {a: _agent_sliced_cost_fn(edge_cost_fn, a, dim) for a in agent_nodes}
    per_agent_rows = {}
    per_agent_edge_table = {}
    per_agent_depot_table = {}
    for agent_id, node_map in agent_nodes.items():
        nodes = list(node_map.keys())
        rows_by_node = {n: _cand_rows(n, node_map[n][0]) for n in nodes}
        per_agent_rows[agent_id] = rows_by_node
        pairs = _all_pairs(nodes)
        et = build_edge_cost_table(rows_by_node, wp_template, pairs, edge_cost_fn=slice_fns[agent_id])
        dt = build_depot_cost_table(rows_by_node, wp_template, x0_of_agent[agent_id], nodes,
                                    edge_cost_fn=slice_fns[agent_id])
        per_agent_edge_table[agent_id] = et
        per_agent_depot_table[agent_id] = dt
        for mat in et.values():
            all_pair_max += float(mat.max()) if mat.size else 0.0
        for arr in dt.values():
            all_depot_max = max(all_depot_max, float(arr.max()) if arr.size else 0.0)
    time_ub = _to_int(all_pair_max + all_depot_max) + n_nodes + 1
    time = [model.NewIntVar(0, time_ub, f"time_{v}") for v in range(n_nodes)]
    T_end = model.NewIntVar(0, time_ub, "T_end")

    # -- per-agent circuits -------------------------------------------------
    agent_cost = {}
    agent_cost_ub = {}
    for agent_id, node_map in agent_nodes.items():
        nodes = list(node_map.keys())
        edge_table = per_agent_edge_table[agent_id]
        depot_table = per_agent_depot_table[agent_id]

        idx_of = {}
        next_idx = 1
        for node in nodes:
            inst_key = node_map[node][0]
            k = len(branch_bool[(node, inst_key)])
            idx_of[node] = list(range(next_idx, next_idx + k))
            next_idx += k

        # active[node][b]: this (node, branch) vertex is really visited by
        # this agent.
        active = {}
        for node in nodes:
            inst_key, gate = node_map[node]
            bvars = branch_bool[(node, inst_key)]
            if gate is None:
                active[node] = list(bvars)
            else:
                active[node] = [_and_lit(model, bv, gate, f"active_{node}_{b}_{agent_id}")
                                for b, bv in enumerate(bvars)]

        arcs = []  # (tail, head, BoolVar, int_cost)

        # depot -> (node, branch): own literal gated by active (never
        # `active` itself -- only one vertex is ever the true first stop).
        for node in nodes:
            for b in range(len(active[node])):
                enter = model.NewBoolVar(f"arc_depot_{node}_{b}_{agent_id}")
                model.AddImplication(enter, active[node][b])
                c = _to_int(depot_table[node][b])
                arcs.append((0, idx_of[node][b], enter, c))
                model.Add(time[node] >= c).OnlyEnforceIf(enter)

        # (node, branch) -> depot: return leg, cost 0.
        for node in nodes:
            for b in range(len(active[node])):
                ret = model.NewBoolVar(f"arc_return_{node}_{b}_{agent_id}")
                model.AddImplication(ret, active[node][b])
                arcs.append((idx_of[node][b], 0, ret, 0))

        # (u, i) -> (v, j): real branch-pair cost.
        for u, v in _all_pairs(nodes):
            mat = edge_table[(u, v)]
            for i in range(len(active[u])):
                for j in range(len(active[v])):
                    lit = model.NewBoolVar(f"arc_{u}_{i}_{v}_{j}_{agent_id}")
                    c = _to_int(mat[i, j])
                    arcs.append((idx_of[u][i], idx_of[v][j], lit, c))
                    if use_rank:
                        model.Add(rank[u] < rank[v]).OnlyEnforceIf(lit)
                    model.Add(time[v] >= time[u] + c).OnlyEnforceIf(lit)

        # self-loops.
        for node in nodes:
            for b in range(len(active[node])):
                arcs.append((idx_of[node][b], idx_of[node][b], active[node][b].Not(), 0))

        all_active = [act for node in nodes for act in active[node]]
        depot_self = model.NewBoolVar(f"depot_self_loop_{agent_id}")
        model.AddBoolOr(all_active + [depot_self])
        for act in all_active:
            model.AddImplication(depot_self, act.Not())
        arcs.append((0, 0, depot_self, 0))

        model.AddCircuit([(t, h, lit) for t, h, lit, _c in arcs])

        cost_ub = sum(c for *_r, c in arcs if c > 0) + 1
        cost_terms = [lit * c for _t, _h, lit, c in arcs if c != 0]
        cost_var = model.NewIntVar(0, cost_ub, f"cost_agent_{agent_id}")
        model.Add(cost_var == sum(cost_terms))
        agent_cost[agent_id] = cost_var
        agent_cost_ub[agent_id] = cost_ub

    # -- ordering edges on `time` (and `rank`) --------------------------
    for u, v in problem.hard_edges:
        model.Add(time[u] <= time[v])

    cond_formulas = cond_formulas or {}
    aux_bool = [model.NewBoolVar(f"bv_{k}") for k in range(problem.n_cond_vars)]
    for (u, v), formula in cond_formulas.items():
        g = compile_gate_cpsat(formula, model, assign_bool, aux_bool,
                               var_sym_ids or {}, cond_sym_ids or {}, n_agents,
                               f"cond_{u}_{v}", _and_lit)
        if use_rank:
            model.Add(rank[u] < rank[v]).OnlyEnforceIf(g)
        model.Add(time[u] <= time[v]).OnlyEnforceIf(g)

    # -- objective -------------------------------------------------------
    for v in range(n_nodes):
        model.Add(T_end >= time[v])

    if objective == "makespan":
        objective_var = T_end
    elif objective == "minmax":
        objective_var = model.NewIntVar(0, sum(agent_cost_ub.values()) + 1, "objective")
        for c in agent_cost.values():
            model.Add(objective_var >= c)
    else:  # avg / sum
        objective_var = model.NewIntVar(0, sum(agent_cost_ub.values()) + 1, "objective")
        model.Add(objective_var == sum(agent_cost.values()))
    model.Minimize(objective_var)

    handles = dict(
        branch_bool=branch_bool, branch_idx=branch_idx, rank=rank, time=time,
        T_end=T_end, assign_bool=assign_bool, aux_bool=aux_bool,
        agent_nodes=agent_nodes, agent_cost=agent_cost,
        objective_var=objective_var, scale=_SCALE, objective=objective,
    )
    return model, handles


def solve_cpsat(model, handles, time_limit=10.0, hint=None, workers=8):
    """Solve `model`. Returns a dict: `status`, `branch` (dict[(node,
    inst_key)] -> chosen candidate index), `rank` (dict[node] -> int),
    `time` (dict[node] -> float, unscaled), `t_end` (float), `assignment`
    (dict[slot] -> agent id), `agent_cost` (dict[agent] -> float),
    `objective` (float, unscaled), `wall_time`.

    `hint`: an optional prior `solve_cpsat` result dict (or just its `branch`
    sub-dict). Its `branch` / `assignment` / `aux` / `rank` entries are fed to
    `CpModel.AddHint` -- the cheap warm-start substitute, since CP-SAT has no
    incremental re-solve API and `build_cpsat_model` rebuilds the model each
    call. A complete, still-feasible hint lets CP-SAT accept it as the initial
    incumbent (immediate upper bound); an infeasible-under-new-data hint is
    just ignored, never wrong."""
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers

    if hint:
        if "branch" not in hint and "assignment" not in hint:
            hint = {"branch": hint}  # bare branch sub-dict
        hv, hval = [], []
        for key, chosen in hint.get("branch", {}).items():
            for b, bv in enumerate(handles["branch_bool"].get(key, [])):
                hv.append(bv)
                hval.append(1 if b == chosen else 0)
        for slot, agent in hint.get("assignment", {}).items():
            for a, bv in enumerate(handles["assign_bool"].get(slot, [])):
                hv.append(bv)
                hval.append(1 if a == agent else 0)
        for k, val in hint.get("aux", {}).items():
            if k < len(handles["aux_bool"]):
                hv.append(handles["aux_bool"][k])
                hval.append(int(val))
        if handles["rank"] is not None:
            for node, rk in hint.get("rank", {}).items():
                if node < len(handles["rank"]):
                    hv.append(handles["rank"][node])
                    hval.append(int(rk))
        for v, val in zip(hv, hval):
            model.AddHint(v, val)

    status = solver.Solve(model)
    status_name = solver.StatusName(status)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return dict(status=status_name, branch={}, rank={}, time={}, t_end=None,
                    assignment={}, aux={}, agent_cost={}, objective=None,
                    wall_time=solver.WallTime())

    scale = handles["scale"]
    branch = {key: solver.Value(idx) for key, idx in handles["branch_idx"].items()}
    rank = ({node: solver.Value(r) for node, r in enumerate(handles["rank"])}
            if handles["rank"] is not None else {})
    node_time = {node: solver.Value(t) / scale for node, t in enumerate(handles["time"])}
    assignment = {slot: next(a for a, bv in enumerate(bvars) if solver.Value(bv))
                  for slot, bvars in handles["assign_bool"].items()}
    agent_cost = {a: solver.Value(c) / scale for a, c in handles["agent_cost"].items()}
    aux = {k: solver.Value(b) for k, b in enumerate(handles["aux_bool"])}
    return dict(status=status_name, branch=branch, rank=rank, time=node_time, aux=aux,
                t_end=solver.Value(handles["T_end"]) / scale, assignment=assignment,
                agent_cost=agent_cost, objective=solver.ObjectiveValue() / scale,
                wall_time=solver.WallTime())


def _decode_routes(problem, handles, result):
    """dict[agent_id] -> list of node ids in visiting order, from
    `handles['agent_nodes']` + `result['assignment']`, ordered by solved
    `rank` when present else by node `time`."""
    routes = {}
    order_key = result["rank"] if result["rank"] else result["time"]
    assignment = result["assignment"]
    for agent_id, node_map in handles["agent_nodes"].items():
        owned = []
        for node, (inst_key, _gate) in node_map.items():
            if isinstance(inst_key, tuple) and inst_key[0] == "var":
                if assignment.get(inst_key[1]) != agent_id:
                    continue
            owned.append(node)
        routes[agent_id] = sorted(owned, key=lambda n: order_key[n])
    return routes


def solve_discrete_graph(graph, x0_by_agent, *, objective="makespan",
                         edge_cost_fn=None, params=None, wp_bounds=(-10.0, 10.0),
                         time_limit=10.0, workers=8, hint=None, use_rank=False):
    """End-to-end monolithic discrete solve from a GraphOfConstraints:
    spec.build_graph_ordering_problem -> structure.* -> build_cpsat_model
    -> solve_cpsat. `x0_by_agent`: (n_agents, state) array / row, or
    dict[agent_id] -> row. Returns the solve_cpsat dict plus `n_nodes`,
    `n_agents`, `build_time`, `solve_time`, `routes` (dict[agent] -> node
    order).

    `hint`: a prior result dict from this function -- its branch / assignment
    / aux / rank are fed to the CP-SAT solver as a starting incumbent (see
    solve_cpsat). NOTE: this call rebuilds `problem` + candidates + model
    every time; only the CP-SAT search is warm-started, not the extraction.
    A caller doing repeated solves on the same graph should build `problem`
    and `node_candidates` once and call build_cpsat_model / solve_cpsat
    directly."""
    from ..evolutionary_waypoint_solver.spec import build_graph_ordering_problem

    if params is None:
        params = np.asarray(graph.view_param_values())
    params = np.asarray(params, dtype=float)

    n_agents = graph.num_agents if hasattr(graph, "num_agents") else len(x0_by_agent)
    if isinstance(x0_by_agent, dict):
        raw = [np.asarray(x0_by_agent[a], dtype=float).reshape(-1) for a in range(n_agents)]
    else:
        raw = [np.asarray(r, dtype=float).reshape(-1) for r in x0_by_agent]
    slot_w = max(len(r) for r in raw)
    x0_spec = np.zeros((n_agents, slot_w))
    for a, r in enumerate(raw):
        x0_spec[a, :len(r)] = r

    t0 = _time.perf_counter()
    problem = build_graph_ordering_problem(
        graph, x0_spec, wp_bounds=wp_bounds,
        objective=objective if objective != "makespan" else "avg",
        edge_cost_fn=edge_cost_fn)

    # Full-width (state_dim,) current-state rows, one per agent: the agent's
    # own config placed at its absolute column band [a*dim : ...].
    dim = problem.dim
    x0_rows = {}
    for a, r in enumerate(raw):
        row = np.zeros(problem.state_dim)
        row[a * dim: a * dim + len(r)] = r
        x0_rows[a] = row

    wp_template = warm_start_wp(problem, x0_spec)
    cands = node_candidates(problem, wp_template, params)
    instances = node_instances(problem)
    cond_formulas, var_sym_ids, cond_sym_ids = conditional_edge_data(graph, problem)

    ecf = edge_cost_fn if edge_cost_fn is not None else getattr(problem, "edge_cost_fn", None)
    model, handles = build_cpsat_model(
        problem, cands, wp_template, x0_rows, instances,
        cond_formulas=cond_formulas, var_sym_ids=var_sym_ids, cond_sym_ids=cond_sym_ids,
        edge_cost_fn=ecf, objective=objective, use_rank=use_rank)
    build_time = _time.perf_counter() - t0

    t1 = _time.perf_counter()
    result = solve_cpsat(model, handles, time_limit=time_limit, hint=hint, workers=workers)
    result["solve_time"] = _time.perf_counter() - t1
    result["build_time"] = build_time
    result["n_nodes"] = problem.n_nodes
    result["n_agents"] = problem.n_agents
    if result["status"] in ("OPTIMAL", "FEASIBLE"):
        result["routes"] = _decode_routes(problem, handles, result)
    else:
        result["routes"] = {}
    return result

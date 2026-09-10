"""`make_dp_master_jax` -- piece 1: the exhaustive static enumeration plus
the plain-Python `run_python` loop must reproduce `dp_master.solve_dp_master`
exactly, and the precomputed gate-activation / extension-feasibility tensors
must agree with `_linear_extensions` on the resolved precedence graph for
every (assignment, aux) combo.

Run: python examples/test_dp_master_jax.py
"""

import numpy as np
import jax.numpy as jnp
from pydrake.math import eq, ge

from goc_mpc import GraphOfConstraints
from goc_mpc._ext.configuration_spline import Block
from goc_mpc.evolutionary_waypoint_solver.projection import ProjOperator
from goc_mpc.evolutionary_waypoint_solver.spec import build_graph_ordering_problem
from goc_mpc.logic_based_benders_solver.structure import (
    node_instances, node_candidates, warm_start_wp)
from goc_mpc.logic_based_benders_solver.dp_master import (
    solve_dp_master, _resolve_precedence, _linear_extensions)
from goc_mpc.logic_based_benders_solver.dp_master_jax import make_dp_master_jax

DIM = 2
EU = lambda a, b: jnp.sqrt(jnp.sum((jnp.asarray(b) - jnp.asarray(a)) ** 2))


def _lit(g, agent, xy):
    xy = np.asarray(xy, float)
    return ProjOperator(pins=g.agent_q(agent), reads=(), continuous_params=0,
                        discrete_params=1, func=lambda psi, b, v=xy: v)


def _ik(g, agent, xys):
    xys = np.asarray(xys, float)
    return ProjOperator(pins=g.agent_q(agent), reads=(), continuous_params=0,
                        discrete_params=len(xys), func=lambda psi, b, v=xys: v[b])


def _lit_var(g, var, xy):
    xy = np.asarray(xy, float)
    return ProjOperator(pins=g.var_agent_q(var), reads=(), continuous_params=0,
                        discrete_params=1, func=lambda psi, b, v=xy: v)


# -- scene A: dynamic var_agent_q multi-branch pin (full-resolve, NC=2) ----
def scene_A():
    cands_xy = np.array([[9.0, 9.0], [1.0, 0.0], [-8.0, 7.0]])
    g = GraphOfConstraints([[Block.R(DIM)], [Block.R(DIM)]], [],
                           state_lower_bound=-20.0, state_upper_bound=20.0,
                           robot_names=["r0", "r1"])
    n0, n1 = g.structure.add_nodes(2)
    g.structure.add_edge(n0, n1, True)
    var = g.add_variable()
    g.add_constraint(n0, ge(g.var_agent_q(var), np.array([-20.0, -20.0])),
                     proj=ProjOperator(pins=g.var_agent_q(var), reads=(), continuous_params=0,
                                       discrete_params=3,
                                       func=lambda psi, b: jnp.asarray(cands_xy)[b]))
    g.add_constraint(n1, eq(g.var_agent_q(var), np.array([2.0, 2.0])),
                     proj=ProjOperator(pins=g.var_agent_q(var), reads=(),
                                       func=lambda psi, b: np.array([2.0, 2.0])))
    x0 = np.concatenate([cands_xy[1], np.array([40.0, 40.0])])
    problem = build_graph_ordering_problem(g, x0.reshape(2, DIM), wp_bounds=(-20.0, 20.0),
                                           objective="avg", edge_cost_fn=EU)
    x0_full = np.zeros(problem.state_dim)
    x0_full[:len(x0)] = x0
    x0_rows = {0: np.pad(cands_xy[1], (0, problem.state_dim - DIM)),
               1: np.pad(np.array([40.0, 40.0]), (DIM, problem.state_dim - 2 * DIM))}
    return problem, x0_full, x0_rows


# -- scene B: conditional ordering edges on a free aux binary (NA=2) ------
def scene_B():
    g = GraphOfConstraints([[Block.R(DIM)]], [], state_lower_bound=-50.0,
                           state_upper_bound=50.0, robot_names=["robot"])
    n0, n1, n2 = g.structure.add_nodes(3)
    g.structure.add_edge(n0, n1, True)
    g.structure.add_edge(n0, n2, True)
    bv = g.add_binary_cond_var()
    g.add_edge(n1, n2, cond=(bv == 1))
    g.add_edge(n2, n1, cond=(bv == 0))
    g.add_constraint(n0, eq(g.agent_q(0), np.zeros(DIM)), proj=_lit(g, 0, [0.0, 0.0]))
    g.add_constraint(n1, eq(g.agent_q(0), np.zeros(DIM)), proj=_lit(g, 0, [1.0, 0.0]))
    g.add_constraint(n2, eq(g.agent_q(0), np.zeros(DIM)), proj=_lit(g, 0, [8.0, 0.0]))
    problem = build_graph_ordering_problem(g, np.zeros((1, DIM)), wp_bounds=(-50.0, 50.0),
                                           objective="makespan", edge_cost_fn=EU)
    x0_full = np.zeros(problem.state_dim)
    return problem, x0_full, {0: np.zeros(problem.state_dim)}


# -- scene C: two arms, cross-agent edge, per-arm branch (makespan) -------
def scene_C():
    g = GraphOfConstraints([[Block.R(DIM)], [Block.R(DIM)]], [],
                           state_lower_bound=-50.0, state_upper_bound=50.0,
                           robot_names=["r0", "r1"])
    n = g.structure.add_nodes(3)
    g.structure.add_edge(n[0], n[1], True)
    g.structure.add_edge(n[1], n[2], True)
    g.add_constraint(n[0], eq(g.agent_q(0), np.zeros(DIM)),
                     proj=ProjOperator(pins=g.agent_q(0), reads=(), continuous_params=0,
                                       discrete_params=2,
                                       func=lambda psi, b: jnp.array([[3.0, 0.0], [0.0, 3.0]])[b]))
    g.add_constraint(n[1], eq(g.agent_q(0), np.zeros(DIM)), proj=_lit(g, 0, [6.0, 0.0]))
    g.add_constraint(n[2], eq(g.agent_q(1), np.zeros(DIM)), proj=_lit(g, 1, [10.0, 0.0]))
    problem = build_graph_ordering_problem(g, np.zeros((2, DIM)), wp_bounds=(-50.0, 50.0),
                                           objective="makespan", edge_cost_fn=EU)
    return problem, np.zeros(problem.state_dim), {0: np.zeros(problem.state_dim),
                                                 1: np.zeros(problem.state_dim)}


def _cmp(a, b):
    for k in ("status", "assignment", "aux", "time", "routes"):
        assert a.get(k) == b.get(k), (k, a.get(k), b.get(k))
    assert abs((a["objective"] or 0.0) - (b["objective"] or 0.0)) < 1e-9, \
        ("objective", a["objective"], b["objective"])
    # `branch` differs only in whether single-branch nodes get an explicit 0
    # (non-full path lists them, full-machinery path doesn't) -- compare with
    # a 0 default over the union of keys.
    for k in set(a.get("branch", {})) | set(b.get("branch", {})):
        assert a["branch"].get(k, 0) == b["branch"].get(k, 0), ("branch", k, a["branch"], b["branch"])


def _check_scene(name, scene, objectives, anchor=None):
    problem, x0_full, x0_rows = scene()
    params = np.asarray(problem.params)
    wp = warm_start_wp(problem, x0_full)
    inst = node_instances(problem)
    na = None if anchor is None else anchor[0]
    for obj in objectives:
        cands = node_candidates(problem, wp, params, allow_unresolved=True)
        ref = solve_dp_master(problem, cands, wp, x0_rows, inst,
                              ordering_edges=problem.ordering_edges, edge_cost_fn=EU,
                              objective=obj, x0_full=x0_full,
                              anchor=None if anchor is None else _AnchorShim(*anchor))
        jm = make_dp_master_jax(problem, objective=obj, edge_cost_fn=EU)
        kw = dict(node_active=na,
                  var_committed=None if anchor is None else anchor[1],
                  var_anchor=None if anchor is None else anchor[2])
        _cmp(ref, jm.run_python(params, wp, x0_full, x0_rows, **kw))
        _cmp(ref, jm.run_vec(params, wp, x0_full, x0_rows, **kw))

        # run_topk: first == run_vec, objectives non-decreasing, (assignment,
        # aux) distinct across the returned members.
        top = jm.run_topk(5, params, wp, x0_full, x0_rows, **kw)
        assert top, (name, obj, "run_topk empty")
        _cmp(ref, top[0])
        objs = [d["objective"] for d in top]
        assert objs == sorted(objs), (name, obj, "run_topk not sorted", objs)
        seen = [(tuple(sorted(d["assignment"].items())),
                 tuple(sorted(d["aux"].items()))) for d in top]
        assert len(seen) == len(set(seen)), (name, obj, "run_topk dup skeleton", seen)

        # skeleton_grid_fn: the jittable genome tuple for the top skeleton must
        # decode to the same discrete solution solve_dp_master found.
        X0 = jnp.stack([jnp.asarray(x0_rows[j] if isinstance(x0_rows, dict) else x0_rows,
                                    float) for j in range(problem.n_agents)])
        node_active = np.ones(problem.n_nodes, bool) if na is None else np.asarray(na, bool)
        vc = np.zeros(problem.n_variables, bool) if anchor is None else np.asarray(anchor[1], bool)
        va = np.zeros(problem.n_variables, int) if anchor is None else np.asarray(anchor[2], int)
        gobj, gassign, gcond, gt, gpb, gwp0, gcell = jm.skeleton_grid_fn(3)(
            jnp.asarray(params), jnp.asarray(wp), jnp.asarray(x0_full), X0,
            jnp.asarray(node_active), jnp.asarray(vc), jnp.asarray(va))
        gobj = np.asarray(gobj)
        assert abs(float(gobj[0]) - ref["objective"]) < 1e-9, (name, obj, "grid obj", gobj[0])
        if problem.n_variables:
            got_assign = {s: int(np.argmax(np.asarray(gassign)[0, s])) for s in range(problem.n_variables)}
            assert got_assign == ref["assignment"], (name, obj, "grid assign", got_assign)
        got_aux = {k: int(round(v)) for k, v in enumerate(np.asarray(gcond)[0])}
        assert got_aux == ref["aux"], (name, obj, "grid aux", got_aux, ref["aux"])
        # proj_branch one-hot decodes per branched entry to the ref branch
        for e in problem.projections:
            if e.discrete_params <= 1:
                continue
            sl = e.branch_slice
            picked = int(np.argmax(np.asarray(gpb)[0, sl.start:sl.start + e.discrete_params]))
            owner = [o for (n, o) in ref["branch"] if n == e.write_node]
            if owner:
                assert ref["branch"][(e.write_node, owner[0])] == picked, \
                    (name, obj, "grid branch", e.write_node, picked, ref["branch"])
        # t is a valid linearisation of the hard precedence graph
        gt0 = np.asarray(gt)[0]
        for (u, v) in problem.hard_edges:
            if node_active[u] and node_active[v]:
                assert gt0[u] < gt0[v], (name, obj, "grid t not topological", u, v, gt0)

        # gate-activation / feasibility tensors vs _linear_extensions
        node_active = np.ones(problem.n_nodes, bool) if na is None else np.asarray(na, bool)
        remaining = [n for n in range(problem.n_nodes) if node_active[n]]
        for c in range(jm.NC):
            ov = np.asarray(jm.A[c], int)
            for aidx in range(jm.NA):
                aux = tuple(int(x) for x in jm.AUX[aidx])
                P = _resolve_precedence(problem, jm.ordering_edges, ov, aux)
                P = {(u, v) for (u, v) in P if node_active[u] and node_active[v]}
                exts = _linear_extensions(remaining, P, jm.max_orders)
                want = set() if exts is None else {tuple(e) for e in exts}
                mask = jm.feasible_ext_mask(c, aidx, node_active)
                have = {tuple(int(x) for x in jm.EXT[e] if node_active[int(x)])
                        for e in range(jm.E) if mask[e]}
                assert have == want, (name, obj, c, aux, have, want)
    print(f"[{name}] run_python & run_vec == solve_dp_master; feasibility tensors == "
          f"_linear_extensions ({','.join(objectives)}) -- OK")


class _AnchorShim:
    def __init__(self, node_active, var_committed, var_anchor):
        import jax.numpy as _j
        self.node_active = _j.asarray(node_active)
        self.var_committed = _j.asarray(var_committed)
        self.var_anchor = _j.asarray(var_anchor)
        self.anchor_wp = _j.zeros((len(node_active), 1))


# -- scene D: two STATIC analytic-IK entries at ONE node, one per arm -----
# (a two-arm handoff -- each arm picks its own branch; the grid must track
#  the branch per (node, agent), not collapse to one per node).
def scene_D():
    g = GraphOfConstraints([[Block.R(DIM)], [Block.R(DIM)]], [],
                           state_lower_bound=-50.0, state_upper_bound=50.0,
                           robot_names=["r0", "r1"])
    n0, n1 = g.structure.add_nodes(2)
    g.structure.add_edge(n0, n1, True)
    g.add_constraint(n0, eq(g.agent_q(0), np.zeros(DIM)), proj=_ik(g, 0, [[3.0, 0.0], [0.0, 9.0]]))
    g.add_constraint(n0, eq(g.agent_q(1), np.zeros(DIM)), proj=_ik(g, 1, [[0.0, 8.0], [10.0, 0.0]]))
    g.add_constraint(n1, eq(g.agent_q(0), np.zeros(DIM)), proj=_lit(g, 0, [6.0, 0.0]))
    problem = build_graph_ordering_problem(g, np.zeros((2, DIM)), wp_bounds=(-50.0, 50.0),
                                           objective="avg", edge_cost_fn=EU)
    return problem, np.zeros(problem.state_dim), {0: np.zeros(problem.state_dim),
                                                 1: np.zeros(problem.state_dim)}


def test_shared_static_node():
    """Two static IK entries at node 0 (arm 0 and arm 1). `run_vec` must
    match `solve_dp_master` (whose non-full path keys branches per
    (node, owner)) -- arm 0 picks [3,0] (closer than [0,9]), arm 1 picks
    [0,8] (closer than [10,0])."""
    problem, x0_full, x0_rows = scene_D()
    params = np.asarray(problem.params)
    wp = warm_start_wp(problem, x0_full)
    inst = node_instances(problem)
    for obj in ("avg", "minmax", "makespan"):
        cands = node_candidates(problem, wp, params, allow_unresolved=True)
        ref = solve_dp_master(problem, cands, wp, x0_rows, inst,
                              ordering_edges=problem.ordering_edges, edge_cost_fn=EU,
                              objective=obj, x0_full=x0_full)
        jm = make_dp_master_jax(problem, objective=obj, edge_cost_fn=EU)
        got = jm.run_vec(params, wp, x0_full, x0_rows)
        _cmp(ref, got)
        # both arms' branches came through, independently
        assert got["branch"].get((0, 0)) == 0, ("D", obj, "arm0 branch", got["branch"])
        assert got["branch"].get((0, 1)) == 0, ("D", obj, "arm1 branch", got["branch"])
    print("[D shared-static-node] run_vec == solve_dp_master; per-arm branch kept -- OK")


def test_categorical_ne_prunes_assignment():
    """`problem.categorical_ne` drops the assignment combos where the two
    named slots land on the same agent -- from `_DpMasterJax.A` and hence
    from every discrete solution."""
    g = GraphOfConstraints([[Block.R(DIM)], [Block.R(DIM)]], [],
                           state_lower_bound=-50.0, state_upper_bound=50.0,
                           robot_names=["r0", "r1"])
    n0, n1 = g.structure.add_nodes(2)
    g.structure.add_edge(n0, n1, True)
    va, vb = g.add_variable(), g.add_variable()
    # both variables want to sit at the same cheap spot -> unconstrained
    # optimum puts both on whichever arm is closer; the constraint forbids it
    g.add_constraint(n0, eq(g.var_agent_q(va), np.zeros(DIM)), proj=_lit_var(g, va, [1.0, 0.0]))
    g.add_constraint(n1, eq(g.var_agent_q(vb), np.zeros(DIM)), proj=_lit_var(g, vb, [1.0, 0.0]))
    problem = build_graph_ordering_problem(g, np.zeros((2, DIM)), wp_bounds=(-50.0, 50.0),
                                           objective="avg", edge_cost_fn=EU)
    x0f = np.zeros(problem.state_dim)

    base = make_dp_master_jax(problem, objective="avg", edge_cost_fn=EU)
    assert base.NC == 4, base.NC                                   # 2 agents ** 2 slots

    problem.categorical_ne = [(0, 1)]
    jm = make_dp_master_jax(problem, objective="avg", edge_cost_fn=EU)
    assert jm.NC == 2, jm.NC                                       # (0,1) and (1,0) only
    assert all(int(r[0]) != int(r[1]) for r in jm.A), jm.A
    got = jm.run_vec(np.asarray(problem.params), warm_start_wp(problem, x0f), x0f, x0f)
    assert got["assignment"][0] != got["assignment"][1], got["assignment"]
    print("[categorical_ne] assignment enumeration pruned to distinct-agent combos -- OK")


if __name__ == "__main__":
    _check_scene("A dynamic-multibranch", scene_A, ("avg", "minmax", "makespan"))
    _check_scene("B conditional-edges", scene_B, ("avg", "minmax", "makespan"))
    _check_scene("C coupled-makespan", scene_C, ("avg", "minmax", "makespan"))
    # anchor: node 0 committed in scene A
    _check_scene("A +anchor", scene_A, ("avg", "makespan"),
                 anchor=(np.array([False, True]), np.array([False]), np.array([0])))
    test_shared_static_node()
    test_categorical_ne_prunes_assignment()
    print("\nAll checks passed.")

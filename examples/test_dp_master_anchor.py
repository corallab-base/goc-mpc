"""solve_dp_master's `anchor` (MPC-cycle commitment) awareness:

  A. remaining-subgraph routing -- a passed node is dropped from the route,
     the branch DP and the returned time/branch/routes dicts; the future
     legs are priced from the live depot x0.
  B. committed assignment slots (2a) -- a var_committed slot is pinned to
     var_anchor and never re-enumerated, even when a different agent would
     be globally cheaper.
  C. committed-prefix linear extensions (2b) -- edges of a partial order
     touching a passed node are dropped, so the remaining order is free.

Run: python examples/test_dp_master_anchor.py
"""

import numpy as np
import jax.numpy as jnp
from pydrake.math import eq, ge

from goc_mpc import GraphOfConstraints
from goc_mpc._ext.configuration_spline import Block
from goc_mpc.evolutionary_waypoint_solver.projection import ProjOperator
from goc_mpc.evolutionary_waypoint_solver.problem import AnchorState
from goc_mpc.evolutionary_waypoint_solver.spec import build_graph_ordering_problem
from goc_mpc.logic_based_benders_solver.structure import (
    node_instances, node_candidates, warm_start_wp)
from goc_mpc.logic_based_benders_solver.dp_master import (
    solve_dp_master, _linear_extensions, _extensions_matrix)

DIM = 2


def _euclid(a, b):
    """jax-traceable (vmapped by the non-full cost-table path) and fine on
    plain numpy (full-resolve path calls it eagerly)."""
    return jnp.sqrt(jnp.sum((jnp.asarray(b) - jnp.asarray(a)) ** 2))


def _e(a, b):
    return float(np.linalg.norm(np.asarray(b) - np.asarray(a)))


def _lit(g, agent, value):
    value = np.asarray(value, dtype=float)
    return ProjOperator(pins=g.agent_q(agent), reads=(), continuous_params=0,
                        discrete_params=1, func=lambda psi, b, v=value: v)


def _all_active_anchor(problem):
    return AnchorState(
        node_active=np.ones(problem.n_nodes, dtype=bool),
        anchor_wp=np.zeros((problem.n_nodes, problem.state_dim)),
        var_committed=np.zeros(problem.n_variables, dtype=bool),
        var_anchor=np.zeros(problem.n_variables, dtype=int))


# --------------------------------------------------------------------------
# A. remaining-subgraph routing
# --------------------------------------------------------------------------
def scene_A():
    pts = [np.array([0.0, 0.0]), np.array([1.0, 0.0]),
           np.array([1.0, 2.0]), np.array([4.0, 2.0])]
    g = GraphOfConstraints([[Block.R(DIM)]], [[Block.R(DIM)]],
                           state_lower_bound=-20.0, state_upper_bound=20.0,
                           robot_names=["r0"], object_names=["o0"])
    nodes = g.structure.add_nodes(4)
    for a, b in zip(nodes, nodes[1:]):
        g.structure.add_edge(a, b, True)
    for n, p in zip(nodes, pts):
        g.add_constraint(n, eq(g.agent_q(0), np.zeros(DIM)), proj=_lit(g, 0, p))
    problem = build_graph_ordering_problem(g, pts[0].reshape(1, DIM), wp_bounds=(-20.0, 20.0))
    return problem, pts


def test_A():
    problem, pts = scene_A()
    params = np.asarray(problem.params)
    wp = warm_start_wp(problem, pts[0].reshape(1, DIM))
    inst = node_instances(problem)

    def run(anchor, x0_pt):
        x0_full = np.zeros(problem.state_dim)
        x0_full[:DIM] = x0_pt
        cands = node_candidates(problem, wp, params, allow_unresolved=True,
                                active_nodes=[n for n in range(problem.n_nodes)
                                              if anchor.node_active[n]])
        return solve_dp_master(problem, cands, wp, {0: np.pad(x0_pt, (0, problem.state_dim - DIM))},
                               inst, ordering_edges=problem.ordering_edges,
                               edge_cost_fn=_euclid, objective="avg",
                               x0_full=x0_full, anchor=anchor)

    full = run(_all_active_anchor(problem), pts[0])
    assert full["status"] == "OPTIMAL"
    assert full["routes"] == {0: [0, 1, 2, 3]}, full["routes"]

    # node 0 committed; agent now sits at pts[1] (its committed position).
    anc = _all_active_anchor(problem)._replace(
        node_active=np.array([False, True, True, True]))
    part = run(anc, pts[1])
    assert part["status"] == "OPTIMAL"
    assert part["routes"] == {0: [1, 2, 3]}, part["routes"]
    assert set(part["time"]) == {1, 2, 3}, part["time"]
    assert 0 not in part["branch"] and all(k[0] != 0 for k in part["branch"])
    # future legs only: (x0=pts[1])->1 [==0] + 1->2 + 2->3
    exp = _e(pts[1], pts[1]) + _e(pts[1], pts[2]) + _e(pts[2], pts[3])
    assert abs(part["objective"] - exp) < 1e-6, (part["objective"], exp)
    # = full (0+1+2+3) minus the already-paid 0->1 leg
    assert abs(part["objective"] - (full["objective"] - _e(pts[0], pts[1]))) < 1e-6
    print("[A] remaining-subgraph routing: passed node dropped, legs priced from live x0 -- OK")


# --------------------------------------------------------------------------
# B. committed assignment slot
# --------------------------------------------------------------------------
def scene_B():
    """One assignable node whose var_agent_q pin lands both arms on the same
    target; agent 0 starts on it (cost 0), agent 1 starts far."""
    target = np.array([0.5, 0.0])
    g = GraphOfConstraints([[Block.R(DIM)], [Block.R(DIM)]], [],
                           state_lower_bound=-20.0, state_upper_bound=20.0,
                           robot_names=["r0", "r1"])
    (n0,) = g.structure.add_nodes(1)
    var = g.add_variable()
    g.add_constraint(n0, ge(g.var_agent_q(var), np.array([-20.0, -20.0])),
                     proj=ProjOperator(pins=g.var_agent_q(var), reads=(), continuous_params=0,
                                       discrete_params=1, func=lambda psi, b: target))
    problem = build_graph_ordering_problem(g, np.zeros((2, DIM)), wp_bounds=(-20.0, 20.0),
                                           objective="avg", edge_cost_fn=_euclid)
    return problem, problem.var_id_to_slot[var], target


def test_B():
    problem, slot, target = scene_B()
    params = np.asarray(problem.params)
    wp = warm_start_wp(problem, np.zeros((2, DIM)))
    inst = node_instances(problem)
    x0_full = np.zeros(problem.state_dim)
    x0_full[:DIM] = target            # agent 0 sits on the target -> leg 0
    x0_full[DIM:2 * DIM] = [10.0, 10.0]
    x0_rows = {0: np.pad(target, (0, problem.state_dim - DIM)),
               1: np.pad(np.array([10.0, 10.0]), (DIM, problem.state_dim - 2 * DIM))}

    def run(anchor):
        return solve_dp_master(problem, node_candidates(problem, wp, params, allow_unresolved=True),
                               wp, x0_rows, inst, ordering_edges=problem.ordering_edges,
                               edge_cost_fn=_euclid, objective="avg",
                               x0_full=x0_full, anchor=anchor)

    free = run(_all_active_anchor(problem))
    assert free["status"] == "OPTIMAL"
    assert free["assignment"][slot] == 0, free["assignment"]

    committed = run(_all_active_anchor(problem)._replace(
        var_committed=np.array([True]), var_anchor=np.array([1])))
    assert committed["status"] == "OPTIMAL"
    assert committed["assignment"][slot] == 1, committed["assignment"]
    assert committed["routes"] == {1: [0]}, committed["routes"]
    print("[B] committed assignment slot pinned to var_anchor, not re-enumerated -- OK")


# --------------------------------------------------------------------------
# C. committed-prefix linear extensions
# --------------------------------------------------------------------------
def test_C():
    # partial order over 4 nodes: 0->3 only. full graph: many linear
    # extensions. With 0 passed, the (0,3) edge drops -> {1,2,3} fully free.
    P = {(0, 3)}
    full = _linear_extensions(range(4), P, 10000)
    rem = _linear_extensions([1, 2, 3], P, 10000)
    assert len(rem) == 6, len(rem)                       # 3! free
    assert all(0 not in e for e in rem)
    assert len(full) > len(rem)
    print(f"[C] linear extensions: full={len(full)} -> remaining-only={len(rem)} (edge to passed node dropped) -- OK")


# --------------------------------------------------------------------------
# D. static-shape linear extensions (for the vectorized kernels)
# --------------------------------------------------------------------------
def test_D():
    P = {(0, 3)}
    exts = _linear_extensions(range(4), P, 10000)          # 12 orders
    mat, n_valid, M = _extensions_matrix(range(4), P, 10000)
    assert n_valid == len(exts) == 12
    assert M == 16 and mat.shape == (16, 4)               # 12 -> next pow2
    assert {tuple(r) for r in mat[:n_valid]} == set(exts)
    assert all(tuple(mat[i]) == tuple(mat[0]) for i in range(n_valid, M))  # filler
    # every real row is a permutation of the 4 nodes, and respects 0 -> 3
    for r in mat[:n_valid]:
        assert sorted(r) == [0, 1, 2, 3]
        assert list(r).index(0) < list(r).index(3)

    # remaining-only subgraph: 3! free, edge to passed node dropped
    rmat, rn, rM = _extensions_matrix([1, 2, 3], P, 10000)
    assert rn == 6 and rM == 8 and rmat.shape == (8, 3)
    assert {tuple(r) for r in rmat[:rn]} == {tuple(e) for e in
                                             _linear_extensions([1, 2, 3], P, 10000)}

    # bucket=False -> exact count
    emat, en, eM = _extensions_matrix([1, 2, 3], P, 10000, bucket=False)
    assert en == eM == 6 and emat.shape == (6, 3)

    # cycle -> n_valid == 0, shape still static
    cmat, cn, cM = _extensions_matrix(range(2), {(0, 1), (1, 0)}, 10000)
    assert cn == 0 and cmat.shape == (1, 2)
    print(f"[D] extensions matrix: {n_valid} orders -> static ({M}, 4) pow2-bucketed, "
          "filler rows masked by n_valid -- OK")


if __name__ == "__main__":
    test_A()
    test_B()
    test_C()
    test_D()
    print("\nAll checks passed.")

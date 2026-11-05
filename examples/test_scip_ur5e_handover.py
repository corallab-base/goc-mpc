"""
SCIPWaypointMPC on the dual-UR5e table handover, in joint space
================================================================
po-goc-mpc's dual_ur5e_tabletop experiment (DualUR5eTableJointEnv): robot0
picks blockA off the left table, hands it to robot1 over the divider, and
robot1 places it on the right table. The graph is what that experiment's
PCOP plan translates to, rebuilt by hand with the same generators' formulas:

  Pick(robot0)    fk_0(q) == blockA + standoff, top-down
  Handoff         blockA == HANDOFF; fk_0(q) == HANDOFF, gripper along +x;
                  fk_1(q) == HANDOFF + (dx, 0, 0), gripper along -x
  Place(robot1)   blockA in the right-table band at place_z;
                  fk_1(q) == blockA + standoff, top-down

chained Pick -> Handoff -> Place, with holds Pick -> Handoff (robot0) and
Handoff -> Place (robot1), and every arm's joint box at every node
(routes=False). The plan grounds each action's robot, so nothing is left to
assign: the only discrete choices are the IK branches, and blockA's placement
in its band is continuous. The objective is joint-space route length from
the current configurations.

Scene constants come from data/ur5e_table_handover.npz, dumped from
po-goc-mpc's DualUR5eTableJointEnv (the reset state is x0), each arm's FK
checked here against its jax FK on 20 random configurations.

SCIPWaypointMPC only locally optimizes the FK constraints, from the robots'
current configurations, and this graph has no discrete decision, so the
solve is one local NLP. Reported per objective, beside it: the same problem
written directly as one NLP (Ipopt from x0) -- a reference local optimum.

Usage:
  python test_scip_ur5e_handover.py [time_limit_seconds]
"""

import sys
import time
from pathlib import Path

import numpy as np
from pydrake.math import eq, ge
from pydrake.solvers import IpoptSolver, MathematicalProgram
from pydrake.symbolic import cos, logical_and, sin, sqrt

from goc_mpc import GraphOfConstraints, SCIPWaypointMPC, WaypointObjective
from goc_mpc._ext.configuration_spline import Block, CubicConfigurationSpline

DATA = Path(__file__).parent / "data" / "ur5e_table_handover.npz"
LINK = "ee"
BLOCK_A = 0


def load():
    d = dict(np.load(DATA))
    for k in ("base_rot", "link_rot", "ee_rot", "axis", "base_pos", "link_pos", "ee_pos",
              "R_top", "R_plus_x", "R_minus_x"):
        v = np.where(np.abs(d[k]) < 1e-12, 0.0, d[k])
        d[k] = np.where(np.abs(np.abs(v) - 1.0) < 1e-12, np.sign(v), v)
    return d


def make_fk(d, arm, sin_, cos_):
    """po-goc-mpc's `build_ur5e_fk` chain for `arm`'s base placement."""
    cy, sy = np.cos(d[f"base_yaw{arm}"]), np.sin(d[f"base_yaw{arm}"])
    cy, sy = (0.0 if abs(cy) < 1e-12 else cy), (0.0 if abs(sy) < 1e-12 else sy)
    yaw = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    R0 = yaw @ d["base_rot"]
    p0 = np.array([*d[f"base_xy{arm}"], d[f"base_z{arm}"]]) + yaw @ d["base_pos"]

    def rot(axis, theta):
        a = axis / np.linalg.norm(axis)
        K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
        return np.eye(3) + sin_(theta) * K + (1.0 - cos_(theta)) * (K @ K)

    def fk(q):
        p, R = p0, R0
        for i in range(6):
            p = p + R @ d["link_pos"][i]
            R = R @ d["link_rot"][i] @ rot(d["axis"][i], q[i])
        return p + R @ d["ee_pos"], R @ d["ee_rot"]

    return fk


def check_fk(d):
    for arm in (0, 1):
        fk = make_fk(d, arm, np.sin, np.cos)
        for q, p, R in zip(d[f"check_q{arm}"], d[f"check_p{arm}"], d[f"check_R{arm}"]):
            pn, Rn = fk(q)
            assert np.allclose(pn, p, atol=1e-9) and np.allclose(Rn, R, atol=1e-9), arm


def build_graph(d):
    graph = GraphOfConstraints([[Block.R(6)]] * 2, [[Block.R(3)]] * 2, state_lower_bound=-d["state_bound"],
                               state_upper_bound=d["state_bound"], robot_names=["robot0", "robot1"],
                               object_names=["blockA", "blockC"], workspace_dim=3)
    for arm in (0, 1):
        graph.set_robot_fk(arm, LINK, make_fk(d, arm, np.sin, np.cos), symbolic_fk_fn=make_fk(d, arm, sin, cos))
    pick, handoff, place = graph.structure.add_nodes(3)
    graph.structure.add_edge(pick, handoff, True)
    graph.structure.add_edge(handoff, place, True)
    graph.add_hold(pick, handoff, 0, [BLOCK_A])
    graph.add_hold(handoff, place, 1, [BLOCK_A])

    def pin(node, arm, xyz, R):
        graph.add_constraint(node, np.concatenate([eq(graph.agent_link_pos(arm, LINK), xyz),
                                                   eq(graph.agent_link_rot(arm, LINK), R.flatten())]))

    block = graph.object_q(BLOCK_A)
    standoff = np.array([0.0, 0.0, float(d["grasp_standoff"])])
    pin(pick, 0, block + standoff, d["R_top"])

    hxyz = d["handoff_xyz"]
    graph.add_constraint(handoff, eq(block, hxyz))
    pin(handoff, 0, hxyz, d["R_plus_x"])
    pin(handoff, 1, hxyz + np.array([float(d["receiver_dx"]), 0.0, 0.0]), d["R_minus_x"])

    xmin, ymin, xmax, ymax = (float(v) for v in d["right_table"])
    graph.add_constraint(place, logical_and(block[0] >= xmin, block[1] >= ymin, block[0] <= xmax, block[1] <= ymax))
    graph.add_constraint(place, block[2] == float(d["place_z"]))
    pin(place, 1, block + standoff, d["R_top"])

    for arm in (0, 1):
        q = graph.agent_q(arm)
        lo, hi = d[f"joint_ranges{arm}"].T
        for node in (pick, handoff, place):
            graph.add_constraint(node, logical_and(*[q[k] >= float(lo[k]) for k in range(6)]), routes=False)
            graph.add_constraint(node, logical_and(*[q[k] <= float(hi[k]) for k in range(6)]), routes=False)
    return graph


def route_length(qs, ord_):
    return sum(float(np.linalg.norm(b - a, ord_)) for a, b in zip(qs, qs[1:]))


def local_solve(d, starts, ord_):
    """The same problem as one NLP, Ipopt from `starts` = (q0 pick, q0
    handoff, q1 handoff, q1 place). Orientation as three equations + trace
    >= 1 (Ipopt rejects the nine-equation pin: more equalities than joints
    at each pinned node). L2 route length smoothed as sqrt(|.|^2 + 1e-8)."""
    x0 = d["x0"]
    prog = MathematicalProgram()
    Q = [prog.NewContinuousVariables(6, n) for n in ("q0_pick", "q0_hand", "q1_hand", "q1_place")]
    xy = prog.NewContinuousVariables(2, "place_xy")
    for k, q in enumerate(Q):
        lo, hi = d[f"joint_ranges{0 if k < 2 else 1}"].T
        prog.AddBoundingBoxConstraint(lo, hi, q)
        prog.SetInitialGuess(q, starts[k])
    xmin, ymin, xmax, ymax = d["right_table"]
    prog.AddBoundingBoxConstraint([xmin, ymin], [xmax, ymax], xy)
    prog.SetInitialGuess(xy, [(xmin + xmax) / 2, (ymin + ymax) / 2])
    fks = [make_fk(d, a, sin, cos) for a in (0, 1)]
    standoff = np.array([0.0, 0.0, float(d["grasp_standoff"])])
    hxyz = d["handoff_xyz"]
    place_xyz = np.array([xy[0], xy[1], float(d["place_z"])]) + standoff
    targets = [(0, Q[0], x0[12:15] + standoff, d["R_top"]),
               (0, Q[1], hxyz, d["R_plus_x"]),
               (1, Q[2], hxyz + np.array([float(d["receiver_dx"]), 0, 0]), d["R_minus_x"]),
               (1, Q[3], place_xyz, d["R_top"])]
    for arm, q, xyz, Rt in targets:
        p, R = fks[arm](q)
        M = Rt.T @ R
        for e in [*(p - xyz), M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]]:
            prog.AddConstraint(e, 0.0, 0.0)
        prog.AddConstraint(M[0, 0] + M[1, 1] + M[2, 2], 1.0, 3.0)
    legs = [(x0[:6], Q[0]), (Q[0], Q[1]), (x0[6:12], Q[2]), (Q[2], Q[3])]
    for a, b in legs:
        diff = b - a
        if ord_ == 2:
            prog.AddCost(sqrt(diff @ diff + 1e-8))
        else:
            t = prog.NewContinuousVariables(6, "t")
            prog.AddLinearConstraint(ge(t - diff, 0))
            prog.AddLinearConstraint(ge(t + diff, 0))
            prog.AddLinearCost(np.ones(6), t)
            prog.SetInitialGuess(t, np.ones(6))
    tic = time.perf_counter()
    result = IpoptSolver().Solve(prog)
    wall = time.perf_counter() - tic
    if not result.is_success():
        return None, wall
    qs = [result.GetSolution(q) for q in Q]
    value = route_length([x0[:6], qs[0], qs[1]], ord_) + route_length([x0[6:12], qs[2], qs[3]], ord_)
    return value, wall


def main():
    time_limit = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    d = load()
    check_fk(d)
    x0 = d["x0"]
    fk_num = [make_fk(d, a, np.sin, np.cos) for a in (0, 1)]
    print(f"x0 robot0 {np.round(x0[:6], 3)}, robot1 {np.round(x0[6:12], 3)}, blockA {x0[12:15]}")

    for objective, ord_ in ((WaypointObjective.kAvgL2, 2), (WaypointObjective.kAvgL1, 1)):
        print(f"\n{objective}")
        local, wall = local_solve(d, [x0[:6], x0[:6], x0[6:12], x0[6:12]], ord_)
        print(f"   local solve from x0: " + (f"{local:.4f}" if local is not None else "failed")
              + f" in {wall * 1e3:.1f} ms")

        graph = build_graph(d)
        scip = SCIPWaypointMPC(graph, [CubicConfigurationSpline([Block.R(6)]) for _ in range(2)],
                               objective=objective, time_limit=time_limit)
        tic = time.perf_counter()
        ok = scip.solve([0, 1, 2], x0)
        wall = time.perf_counter() - tic
        s = scip.view_stats()
        times = np.asarray(s.solution_times)
        if not ok:
            print(f"   SCIP: no solution ({s.status}, {s.nodes} nodes, {wall:.2f}s; {s.local_solves} local solves, "
                  f"{s.local_failures} failed, {s.local_solve_time:.3f}s)")
            continue
        W = np.asarray(scip.view_waypoints())
        routes = [list(r) for r in scip.view_routes()]
        assert routes == [[0, 1], [1, 2]], routes
        q0p, q0h, q1h, q1p = W[0, :6], W[1, :6], W[1, 6:12], W[2, 6:12]
        block_place = W[2, 12 + 3 * BLOCK_A:15 + 3 * BLOCK_A]
        standoff = np.array([0.0, 0.0, float(d["grasp_standoff"])])
        for arm, q, xyz, Rt in ((0, q0p, x0[12:15] + standoff, d["R_top"]), (0, q0h, d["handoff_xyz"], d["R_plus_x"]),
                                (1, q1h, d["handoff_xyz"] + [float(d["receiver_dx"]), 0, 0], d["R_minus_x"]),
                                (1, q1p, block_place + standoff, d["R_top"])):
            p, R = fk_num[arm](q)
            assert np.linalg.norm(p - xyz) < 1e-5 and np.linalg.norm(R - Rt) < 1e-5, (arm, p, xyz)
        got = route_length([x0[:6], q0p, q0h], ord_) + route_length([x0[6:12], q1h, q1p], ord_)
        print(f"   SCIP: route length {got:.4f} ({s.status}, {s.nodes} nodes, {len(times)} solutions, first at "
              f"{times[0]:.3f}s); {s.local_solves} local solves ({s.local_failures} failed) in "
              f"{s.local_solve_time:.3f}s; solve {s.solving_time:.3f}s, wall {wall:.3f}s")
        print(f"         blockA placed at {np.round(block_place, 3)}")


if __name__ == "__main__":
    main()

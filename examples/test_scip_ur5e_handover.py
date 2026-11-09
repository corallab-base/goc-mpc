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

chained Pick -> Handoff -> Place, with every arm's joint box at every node
(routes=False). Unlike the PCOP plan, which grounds each action's robot, the
giver and receiver are assignment variables (giver != receiver), with
assignable holds Pick -> Handoff (giver) and Handoff -> Place (receiver): the
one discrete decision is robot0 -> robot1 or robot1 -> robot0. blockA's
placement in its band is continuous. The objective is joint-space route
length from the current configurations.

Scene constants come from data/ur5e_table_handover.npz, dumped from
po-goc-mpc's DualUR5eTableJointEnv (the reset state is x0), each arm's FK
checked here against its jax FK on 20 random configurations. The joint
boxes are the model's own stops less the env's 0.05 margin (+-2pi on five
joints, +-pi on the elbow), not the env's +-pi clamp, which exists only for
its NTField's training box.

SCIPWaypointMPC only locally optimizes the FK constraints, from the robots'
current configurations. Reported per objective: SCIP's assignment, total
solve time, B&B nodes and time per node, and its local NLP solves; beside
it, each assignment written directly as one NLP (Ipopt from x0) -- a
reference local optimum.

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
    giver, receiver = graph.add_variable(), graph.add_variable()
    graph.add_variable_ineq_constraint(giver, receiver)
    graph.add_assignable_hold(pick, handoff, giver, [BLOCK_A])
    graph.add_assignable_hold(handoff, place, receiver, [BLOCK_A])

    def pin(node, role, xyz, R):
        graph.add_constraint(node, np.concatenate([eq(graph.var_agent_link_pos(role, LINK), xyz),
                                                   eq(graph.var_agent_link_rot(role, LINK), R.flatten())]))

    block = graph.object_q(BLOCK_A)
    standoff = np.array([0.0, 0.0, float(d["grasp_standoff"])])
    pin(pick, giver, block + standoff, d["R_top"])

    hxyz = d["handoff_xyz"]
    graph.add_constraint(handoff, eq(block, hxyz))
    pin(handoff, giver, hxyz, d["R_plus_x"])
    pin(handoff, receiver, hxyz + np.array([float(d["receiver_dx"]), 0.0, 0.0]), d["R_minus_x"])

    xmin, ymin, xmax, ymax = (float(v) for v in d["right_table"])
    graph.add_constraint(place, logical_and(block[0] >= xmin, block[1] >= ymin, block[0] <= xmax, block[1] <= ymax))
    graph.add_constraint(place, block[2] == float(d["place_z"]))
    pin(place, receiver, block + standoff, d["R_top"])

    for arm in (0, 1):
        q = graph.agent_q(arm)
        lo, hi = d[f"joint_ranges{arm}"].T
        for node in (pick, handoff, place):
            graph.add_constraint(node, logical_and(*[q[k] >= float(lo[k]) for k in range(6)]), routes=False)
            graph.add_constraint(node, logical_and(*[q[k] <= float(hi[k]) for k in range(6)]), routes=False)
    return graph


def route_length(qs, ord_):
    return sum(float(np.linalg.norm(b - a, ord_)) for a, b in zip(qs, qs[1:]))


def local_solve(d, giver, receiver, ord_):
    """The problem under one assignment as one NLP, Ipopt from x0. Orientation as three equations + trace
    >= 1 (Ipopt rejects the nine-equation pin: more equalities than joints
    at each pinned node). L2 route length smoothed as sqrt(|.|^2 + 1e-8)."""
    x0 = d["x0"]
    arms = (giver, giver, receiver, receiver)
    prog = MathematicalProgram()
    Q = [prog.NewContinuousVariables(6, n) for n in ("qg_pick", "qg_hand", "qr_hand", "qr_place")]
    xy = prog.NewContinuousVariables(2, "place_xy")
    for k, q in enumerate(Q):
        lo, hi = d[f"joint_ranges{arms[k]}"].T
        prog.AddBoundingBoxConstraint(lo, hi, q)
        prog.SetInitialGuess(q, x0[6 * arms[k]:6 * arms[k] + 6])
    xmin, ymin, xmax, ymax = d["right_table"]
    prog.AddBoundingBoxConstraint([xmin, ymin], [xmax, ymax], xy)
    prog.SetInitialGuess(xy, [(xmin + xmax) / 2, (ymin + ymax) / 2])
    fks = [make_fk(d, a, sin, cos) for a in (0, 1)]
    standoff = np.array([0.0, 0.0, float(d["grasp_standoff"])])
    hxyz = d["handoff_xyz"]
    place_xyz = np.array([xy[0], xy[1], float(d["place_z"])]) + standoff
    targets = [(giver, Q[0], x0[12:15] + standoff, d["R_top"]),
               (giver, Q[1], hxyz, d["R_plus_x"]),
               (receiver, Q[2], hxyz + np.array([float(d["receiver_dx"]), 0, 0]), d["R_minus_x"]),
               (receiver, Q[3], place_xyz, d["R_top"])]
    for arm, q, xyz, Rt in targets:
        p, R = fks[arm](q)
        M = Rt.T @ R
        for e in [*(p - xyz), M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]]:
            prog.AddConstraint(e, 0.0, 0.0)
        prog.AddConstraint(M[0, 0] + M[1, 1] + M[2, 2], 1.0, 3.0)
    qg0, qr0 = x0[6 * giver:6 * giver + 6], x0[6 * receiver:6 * receiver + 6]
    legs = [(qg0, Q[0]), (Q[0], Q[1]), (qr0, Q[2]), (Q[2], Q[3])]
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
    value = route_length([qg0, qs[0], qs[1]], ord_) + route_length([qr0, qs[2], qs[3]], ord_)
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
        for giver, receiver in ((0, 1), (1, 0)):
            local, wall = local_solve(d, giver, receiver, ord_)
            print(f"   local solve from x0, robot{giver} -> robot{receiver}: "
                  + (f"{local:.4f}" if local is not None else "failed") + f" in {wall * 1e3:.1f} ms")

        graph = build_graph(d)
        scip = SCIPWaypointMPC(graph, [CubicConfigurationSpline([Block.R(6)]) for _ in range(2)],
                               objective=objective, time_limit=time_limit)
        tic = time.perf_counter()
        ok = scip.solve([0, 1, 2], x0)
        wall = time.perf_counter() - tic
        s = scip.view_stats()
        local_times = np.array([r.total_time for r in s.local_log])
        per_local = (f"mean {local_times.mean() * 1e3:.2f} ms, median {np.median(local_times) * 1e3:.2f} ms, "
                     f"max {local_times.max() * 1e3:.2f} ms" if local_times.size else "none")
        print(f"   SCIP: {s.status}, wall {wall:.3f}s (SCIP solving time {s.solving_time:.3f}s); "
              f"{s.nodes} nodes, {s.solving_time / max(s.nodes, 1) * 1e3:.2f} ms per node")
        print(f"         {s.local_solves} local solves ({s.local_failures} failed), {s.local_solve_time:.3f}s total; "
              f"per solve {per_local}")
        for k, r in enumerate(s.local_log):
            print(f"           local solve {k}: {'ok    ' if r.success else 'failed'} {r.total_time * 1e3:7.2f} ms, "
                  f"outer {r.outer}, inner {r.inner}, rho {r.rho:.0e}")
        if not ok:
            print(f"         no solution")
            continue
        giver, receiver = (int(a) for a in np.asarray(scip.view_var_assignments()))
        W = np.asarray(scip.view_waypoints())
        routes = [list(r) for r in scip.view_routes()]
        assert routes[giver] == [0, 1] and routes[receiver] == [1, 2], routes
        g, r = slice(6 * giver, 6 * giver + 6), slice(6 * receiver, 6 * receiver + 6)
        qgp, qgh, qrh, qrp = W[0, g], W[1, g], W[1, r], W[2, r]
        block_place = W[2, 12 + 3 * BLOCK_A:15 + 3 * BLOCK_A]
        standoff = np.array([0.0, 0.0, float(d["grasp_standoff"])])
        for arm, q, xyz, Rt in ((giver, qgp, x0[12:15] + standoff, d["R_top"]),
                                (giver, qgh, d["handoff_xyz"], d["R_plus_x"]),
                                (receiver, qrh, d["handoff_xyz"] + [float(d["receiver_dx"]), 0, 0], d["R_minus_x"]),
                                (receiver, qrp, block_place + standoff, d["R_top"])):
            p, R = fk_num[arm](q)
            assert np.linalg.norm(p - xyz) < 1e-5 and np.linalg.norm(R - Rt) < 1e-5, (arm, p, xyz)
        got = route_length([x0[g], qgp, qgh], ord_) + route_length([x0[r], qrh, qrp], ord_)
        print(f"         robot{giver} -> robot{receiver}, route length {got:.4f}; blockA placed at "
              f"{np.round(block_place, 3)}")


if __name__ == "__main__":
    main()

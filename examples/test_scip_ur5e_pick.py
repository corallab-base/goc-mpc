"""
SCIPWaypointMPC on a UR5e pick, through the FK interface
========================================================
One node: robot0 of po-goc-mpc's dual_ur5e_block_stacking scene (joint-space
planning) picks blockB from the top -- the gripper's pinch site at the
block's centre plus the grasp standoff, pointing straight down. The target
reaches the graph exactly as that experiment writes it: an FK node
constraint,

    set_robot_fk(0, "ee", fk_fn, symbolic_fk_fn=...)
    add_constraint(node, eq(agent_link_pos(0, "ee"), xyz))
    add_constraint(node, eq(agent_link_rot(0, "ee"), R.flatten()))

plus the experiment's joint-limit box (routes=False). The waypoint objective
is the joint-space distance from HOME, so the solve picks among the arm's IK
branches (shoulder, elbow, wrist) the one closest to HOME.

The FK constants (link frames, base pose, joint ranges, target) come from
data/ur5e_block_stacking_pick.npz, dumped from po-goc-mpc's own
`ur5e_fk._extract_link_frames` and checked here against its jax FK on 20
random configurations.

Reported per objective:
  * a plain local solve (Ipopt from HOME) -- what one local optimum costs;
  * a multistart of that local solve -- the best value known;
  * SCIP: time to its first solution, time to within 1% of the best known,
    and the final status/bounds at the time limit.

Usage:
  python test_scip_ur5e_pick.py [time_limit_seconds]
"""

import sys
import time
from pathlib import Path

import numpy as np
from pydrake.math import eq, ge
from pydrake.solvers import IpoptSolver, MathematicalProgram
from pydrake.symbolic import cos, logical_and, sin

from goc_mpc import GraphOfConstraints, SCIPWaypointMPC, WaypointObjective
from goc_mpc._ext.configuration_spline import Block, CubicConfigurationSpline

DATA = Path(__file__).parent / "data" / "ur5e_block_stacking_pick.npz"
STATE_BOUND = 7.0  # DualUR5eBlockStackingJointEnv.STATE_BOUND
LINK = "ee"


def load():
    """The dumped scene, with link-frame constants snapped to exact 0/+-1
    where they're that up to quaternion round-off (keeps the symbolic FK
    free of 1e-16 terms)."""
    d = dict(np.load(DATA))
    for k in ("base_rot", "link_rot", "ee_rot", "axis", "base_pos", "link_pos", "ee_pos"):
        v = np.where(np.abs(d[k]) < 1e-12, 0.0, d[k])
        d[k] = np.where(np.abs(np.abs(v) - 1.0) < 1e-12, np.sign(v), v)
    return d


def make_fk(d, sin_, cos_):
    """`fk(q) -> (position (3,), rotation (3, 3))`, po-goc-mpc's
    `build_ur5e_fk` chain; numeric with (np.sin, np.cos), symbolic with
    pydrake's."""
    cy, sy = np.cos(d["base_yaw"]), np.sin(d["base_yaw"])
    yaw = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    R0 = yaw @ d["base_rot"]
    p0 = np.array([*d["base_xy"], d["base_z"]]) + yaw @ d["base_pos"]

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


def build_graph(d):
    graph = GraphOfConstraints([[Block.R(6)]], [], state_lower_bound=-STATE_BOUND,
                               state_upper_bound=STATE_BOUND, robot_names=["robot0"], object_names=[],
                               workspace_dim=3)
    graph.set_robot_fk(0, LINK, make_fk(d, np.sin, np.cos), symbolic_fk_fn=make_fk(d, sin, cos))
    (node,) = graph.structure.add_nodes(1)
    graph.add_constraint(node, eq(graph.agent_link_pos(0, LINK), d["target_xyz"]))
    graph.add_constraint(node, eq(graph.agent_link_rot(0, LINK), d["target_R"].flatten()))
    q = graph.agent_q(0)
    lo, hi = d["joint_ranges"].T
    graph.add_constraint(node, logical_and(*[q[k] >= float(lo[k]) for k in range(6)]), routes=False)
    graph.add_constraint(node, logical_and(*[q[k] <= float(hi[k]) for k in range(6)]), routes=False)
    return graph


def distance(q, home, ord_):
    return float(np.linalg.norm(q - home, ord_))


def local_solve(d, fk, q_start, ord_):
    """min |q - HOME|_ord s.t. fk(q) == target, joint box -- Ipopt from q_start.
    L2 as its square (same minimizer), L1 through slacks. The orientation as
    three equations (the skew part of R_target^T R, with trace >= 1 ruling
    out the half-turn rotations that also zero it): Ipopt rejects the
    graph's nine-equation form outright, having more equalities than joints."""
    prog = MathematicalProgram()
    q = prog.NewContinuousVariables(6, "q")
    lo, hi = d["joint_ranges"].T
    prog.AddBoundingBoxConstraint(lo, hi, q)
    p, R = fk(q)
    M = d["target_R"].T @ R
    for e in [*(p - d["target_xyz"]), M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]]:
        prog.AddConstraint(e, 0.0, 0.0)
    prog.AddConstraint(M[0, 0] + M[1, 1] + M[2, 2], 1.0, 3.0)
    home = d["home_q"]
    if ord_ == 2:
        prog.AddQuadraticErrorCost(np.eye(6), home, q)
    else:
        t = prog.NewContinuousVariables(6, "t")
        prog.AddLinearConstraint(ge(t - (q - home), 0))
        prog.AddLinearConstraint(ge(t + (q - home), 0))
        prog.AddLinearCost(np.ones(6), t)
        prog.SetInitialGuess(t, np.abs(q_start - home))
    prog.SetInitialGuess(q, q_start)
    tic = time.perf_counter()
    result = IpoptSolver().Solve(prog)
    wall = time.perf_counter() - tic
    if not result.is_success():
        return None, wall
    q_opt = result.GetSolution(q)
    p_opt, R_opt = make_fk(d, np.sin, np.cos)(q_opt)
    assert np.linalg.norm(p_opt - d["target_xyz"]) < 1e-6 and np.linalg.norm(R_opt - d["target_R"]) < 1e-6
    return q_opt, wall


def check_symbolic_fk(d):
    fk = make_fk(d, np.sin, np.cos)
    for q, p, R in zip(d["check_q"], d["check_p"], d["check_R"]):
        pn, Rn = fk(q)
        assert np.allclose(pn, p, atol=1e-9) and np.allclose(Rn, R, atol=1e-9)


def main():
    time_limit = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    d = load()
    check_symbolic_fk(d)
    fk_sym = make_fk(d, sin, cos)
    fk_num = make_fk(d, np.sin, np.cos)
    home = d["home_q"]
    print(f"target {d['target_xyz']}, HOME {home}")

    for objective, ord_ in ((WaypointObjective.kAvgL2, 2), (WaypointObjective.kAvgL1, 1)):
        print(f"\n{objective}")
        q_local, wall = local_solve(d, fk_sym, home, ord_)
        local = distance(q_local, home, ord_) if q_local is not None else float("nan")
        print(f"   local solve from HOME: {local:.4f} in {wall * 1e3:.1f} ms"
              + ("" if q_local is not None else "  (failed)"))

        rng = np.random.default_rng(0)
        lo, hi = d["joint_ranges"].T
        best, n_ok, walls = np.inf, 0, []
        for start in rng.uniform(lo, hi, size=(200, 6)):
            q, w = local_solve(d, fk_sym, start, ord_)
            walls.append(w)
            if q is not None:
                n_ok += 1
                best = min(best, distance(q, home, ord_))
        print(f"   multistart (200 random starts, {n_ok} converged, median {np.median(walls) * 1e3:.1f} ms each): "
              f"best {best:.4f}")

        graph = build_graph(d)
        scip = SCIPWaypointMPC(graph, [CubicConfigurationSpline([Block.R(6)])], objective=objective,
                               time_limit=time_limit)
        tic = time.perf_counter()
        ok = scip.solve([0], home)
        wall = time.perf_counter() - tic
        s = scip.view_stats()
        times, objs = np.asarray(s.solution_times), np.asarray(s.solution_objectives)
        if not ok:
            print(f"   SCIP: no solution ({s.status}, {s.nodes} nodes, {wall:.2f}s)")
            continue
        q = np.asarray(scip.view_waypoints())[0]
        p, R = fk_num(q)
        assert np.linalg.norm(p - d["target_xyz"]) < 1e-5 and np.linalg.norm(R - d["target_R"]) < 1e-5
        within = times[objs <= best * 1.01 + 1e-9]
        print(f"   SCIP: first solution {objs[0]:.4f} at {times[0]:.2f}s; "
              f"within 1% of best known at {within[0]:.2f}s" if within.size else
              f"   SCIP: first solution {objs[0]:.4f} at {times[0]:.2f}s; never within 1% of best known")
        print(f"         final {s.primal_bound:.4f} (dual bound {s.dual_bound:.4f}, gap {s.gap:.2%}), "
              f"{s.status}, {s.nodes} nodes, {len(times)} solutions; solve {s.solving_time:.2f}s, wall {wall:.2f}s")
        print(f"         q = {np.round(q, 4)}")


if __name__ == "__main__":
    main()

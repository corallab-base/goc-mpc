"""
SCIPWaypointMPC with nonlinear (forward-kinematics) node constraints
====================================================================
Planar 2-link arms whose configuration is the joint angles q = (q0, q1).
Every task node pins the end effector, fk(q) == p, written directly with
pydrake.symbolic sin/cos. Each such target has two IK solutions (elbow up
and elbow down), so the waypoint problem is nonconvex: which branch to take
at each node is a discrete choice hidden inside the continuous variables.

Ground truth: the closed-form IK gives both branches of every target, and
pinning each node to one branch makes the problem an ordinary GCS one.
GCSWaypointMPC (Gurobi) solves every branch combination; the best of those
is the global optimum, and SCIP's spatial branch-and-bound must match it.

  1. one arm, two nodes in sequence, under kAvgL1 and kMinMaxL2;
  2. two identical arms, three role tasks (assignment, order and branch
     all chosen together), under kMinMaxL2 and kAvgL2;
  3. through the FK interface instead (set_robot_fk with a symbolic_fk_fn,
     var_agent_link_pos/var_agent_link_rot): two arms at different bases,
     one role target, each candidate's own FK -- against enumerating every
     (arm, branch).

Usage:
  python test_scip_arm_fk.py
"""

import itertools
import time

import numpy as np
from pydrake.math import eq
from pydrake.symbolic import cos, sin

from goc_mpc import GCSWaypointMPC, GraphOfConstraints, SCIPWaypointMPC, WaypointObjective, WaypointSolver
from goc_mpc._ext.configuration_spline import Block, CubicConfigurationSpline

L1, L2 = 1.0, 0.8
DIM = 2


def fk(q):
    return np.array([L1 * cos(q[0]) + L2 * cos(q[0] + q[1]),
                     L1 * sin(q[0]) + L2 * sin(q[0] + q[1])])


def fk_num(q):
    return np.array([L1 * np.cos(q[0]) + L2 * np.cos(q[0] + q[1]),
                     L1 * np.sin(q[0]) + L2 * np.sin(q[0] + q[1])])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def ik(p):
    """Both IK solutions of target p, joint angles wrapped to [-pi, pi)."""
    c1 = (p @ p - L1**2 - L2**2) / (2 * L1 * L2)
    assert abs(c1) < 1, f"{p} out of reach"
    sols = []
    for q1 in (np.arccos(c1), -np.arccos(c1)):
        q0 = np.arctan2(p[1], p[0]) - np.arctan2(L2 * np.sin(q1), L1 + L2 * np.cos(q1))
        q = wrap(np.array([q0, q1]))
        assert np.allclose(fk_num(q), p)
        sols.append(q)
    return sols


def build(n_agents, targets, pins=None):
    """One node per target. n_agents == 1: a chain of agent-0 nodes;
    otherwise one role per target. pins (one IK branch per target) replaces
    fk(q) == p by q == that branch's configuration."""
    graph = GraphOfConstraints([[Block.R(DIM)]] * n_agents, [], state_lower_bound=-np.pi,
                               state_upper_bound=np.pi, robot_names=[f"arm{j}" for j in range(n_agents)],
                               object_names=[])
    nodes = graph.structure.add_nodes(len(targets))
    for k, (node, p) in enumerate(zip(nodes, targets)):
        if n_agents == 1:
            q = graph.agent_q(0)
            if k > 0:
                graph.structure.add_edge(nodes[k - 1], node, True)
        else:
            q = graph.var_agent_q(graph.add_variable())
        graph.add_constraint(node, eq(q, pins[k]) if pins is not None else eq(fk(q), p))
    return graph


def splines(n_agents):
    return [CubicConfigurationSpline([Block.R(DIM)]) for _ in range(n_agents)]


def value(mpc, objective, x0):
    """The objective's value: makespan (min-max) or total route length (avg)."""
    if "MinMax" in str(objective):
        return float(np.asarray(mpc.view_t())[-1])
    ord_ = 1 if "L1" in str(objective) else 2
    W = np.asarray(mpc.view_waypoints())
    total = 0.0
    for j, route in enumerate(mpc.view_routes()):
        pts = [x0[DIM * j:DIM * (j + 1)]] + [W[v, DIM * j:DIM * (j + 1)] for v in route]
        total += sum(np.linalg.norm(b - a, ord_) for a, b in zip(pts, pts[1:]))
    return total


def brute_force(n_agents, targets, objective, x0):
    branches = [ik(p) for p in targets]
    best = None
    for combo in itertools.product(range(2), repeat=len(targets)):
        graph = build(n_agents, targets, pins=[branches[k][b] for k, b in enumerate(combo)])
        mpc = GCSWaypointMPC(graph, splines(n_agents), solver=WaypointSolver.kGurobi, objective=objective)
        assert mpc.solve(list(range(len(targets))), x0)
        v = value(mpc, objective, x0)
        if best is None or v < best[0] - 1e-9:
            best = (v, combo, [list(r) for r in mpc.view_routes()])
    return best


def check(name, n_agents, targets, objective, x0):
    best, combo, routes = brute_force(n_agents, targets, objective, x0)
    graph = build(n_agents, targets)
    scip = SCIPWaypointMPC(graph, splines(n_agents), objective=objective)
    tic = time.perf_counter()
    assert scip.solve(list(range(len(targets))), x0), name
    wall = time.perf_counter() - tic
    W = np.asarray(scip.view_waypoints())
    got = value(scip, objective, x0)
    stats = scip.view_stats()

    # Every waypoint reaches its target, on the brute-force optimum's branch.
    branches = [ik(p) for p in targets]
    got_combo = []
    for k, p in enumerate(targets):
        j = next(j for j, r in enumerate(scip.view_routes()) if k in r)
        q = W[k, DIM * j:DIM * (j + 1)]
        assert np.linalg.norm(fk_num(q) - p) < 1e-5, (name, k, fk_num(q), p)
        got_combo.append(int(np.argmin([np.linalg.norm(wrap(q - b)) for b in branches[k]])))
    print(f"{name}: scip {got:.6f} vs brute force {best:.6f}  branches {tuple(got_combo)} vs {combo}  "
          f"routes {[list(r) for r in scip.view_routes()]} vs {routes}  "
          f"[{stats.status}, {stats.nodes} nodes, {wall:.2f}s]")
    assert abs(got - best) < 1e-3 * max(1.0, best), (name, got, best)
    assert tuple(got_combo) == combo, (name, got_combo, combo)


def fk_interface_case(objective):
    """Case 3: arm j's base at BASES[j], target p in world frame."""
    BASES = [np.array([0.0, 0.0]), np.array([1.5, 0.3])]
    x0 = np.array([0.4, 1.0, 2.6, -0.8])
    p = np.array([0.8, 0.9])

    def make_fk(base, sin_, cos_):
        def fk(q):
            a = q[0] + q[1]
            return (base + np.array([L1 * cos_(q[0]) + L2 * cos_(a), L1 * sin_(q[0]) + L2 * sin_(a)]),
                    np.array([[cos_(a), -sin_(a)], [sin_(a), cos_(a)]]))
        return fk

    graph = GraphOfConstraints([[Block.R(DIM)]] * 2, [], state_lower_bound=-np.pi, state_upper_bound=np.pi,
                               robot_names=["arm0", "arm1"], object_names=[], workspace_dim=2)
    for j, base in enumerate(BASES):
        graph.set_robot_fk(j, "ee", make_fk(base, np.sin, np.cos), symbolic_fk_fn=make_fk(base, sin, cos))
    (node,) = graph.structure.add_nodes(1)
    r = graph.add_variable()
    graph.add_constraint(node, eq(graph.var_agent_link_pos(r, "ee"), p))

    ord_ = 1 if "L1" in str(objective) else 2
    best = min((np.linalg.norm(q - x0[DIM * j:DIM * (j + 1)], ord_), j)
               for j, base in enumerate(BASES) for q in ik(p - base))
    scip = SCIPWaypointMPC(graph, splines(2), objective=objective)
    assert scip.solve([0], x0)
    j = int(np.asarray(scip.view_var_assignments())[0])
    q = np.asarray(scip.view_waypoints())[0, DIM * j:DIM * (j + 1)]
    got = np.linalg.norm(q - x0[DIM * j:DIM * (j + 1)], ord_)
    assert np.linalg.norm(make_fk(BASES[j], np.sin, np.cos)(q)[0] - p) < 1e-5
    print(f"FK interface {objective}: scip {got:.6f} on arm {j} vs brute force {best[0]:.6f} on arm {best[1]}  "
          f"[{scip.view_stats().status}, {scip.view_stats().nodes} nodes]")
    assert abs(got - best[0]) < 1e-3 * max(1.0, best[0]) and j == best[1]


def main():
    one_arm = [np.array([1.2, 0.6]), np.array([-0.4, 1.3])]
    x0_one = np.array([-0.3, -1.5])
    for objective in (WaypointObjective.kAvgL1, WaypointObjective.kMinMaxL2):
        check(f"1 arm {objective}", 1, one_arm, objective, x0_one)

    two_arms = [np.array([1.2, 0.6]), np.array([-0.4, 1.3]), np.array([0.3, -1.1])]
    x0_two = np.array([-0.3, -1.5, 2.0, 1.0])
    for objective in (WaypointObjective.kMinMaxL2, WaypointObjective.kAvgL2):
        check(f"2 arms {objective}", 2, two_arms, objective, x0_two)

    for objective in (WaypointObjective.kAvgL2, WaypointObjective.kAvgL1):
        fk_interface_case(objective)
    print("all checks passed")


if __name__ == "__main__":
    main()

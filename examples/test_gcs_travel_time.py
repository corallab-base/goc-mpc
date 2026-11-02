"""
GCSWaypointMPC travel-time objectives
=====================================
Checks kMinMaxTravelTime / kAvgTravelTime against the L2 objectives with a
Euclidean ArcTravelTimeModel, T(x, y) = |y - x| / speed:

  1. Every waypoint pinned: the linearization is exact, so the solve must
     match kMinMaxL2 / kAvgL2 -- same routes and assignments, objective / speed.
  2. Some waypoints confined to small boxes: still solves, and stays close to
     the L2 solution (the tangent is exact at the box centre only).
  3. max_tangent_halfwidth below a box's half-width raises.

Usage:
  python test_gcs_travel_time.py
"""

import numpy as np
from pydrake.math import eq, ge, le

from goc_mpc import (
    ArcTravelTimeModel,
    GCSWaypointMPC,
    GraphOfConstraints,
    WaypointObjective,
    WaypointSolver,
)
from goc_mpc._ext.configuration_spline import Block, CubicConfigurationSpline

DIM = 2
N_AGENTS = 2
SPEED = 2.0
LB, UB = -10.0, 10.0
X0 = np.array([-1.0, 0.0, 1.0, 0.0])

# One role per (node, location); each role's holder must be at its location.
ROLE_SITES = [
    (np.array([-2.0, 1.0]), np.array([-2.0, 3.0])),
    (np.array([0.5, 2.0]), np.array([0.0, 4.0])),
    (np.array([2.0, 1.0]), np.array([3.0, 2.5])),
]


class EuclideanTravelTime(ArcTravelTimeModel):
    def __init__(self, speed):
        lower, upper = np.full(DIM, LB), np.full(DIM, UB)
        super().__init__(lower, upper, np.linalg.norm(upper - lower) / speed)
        self.speed = speed

    def linearize(self, X, Y):
        d = Y - X
        n = np.linalg.norm(d, axis=1)
        g = np.where(n[:, None] > 0, d / (self.speed * np.maximum(n, 1e-12)[:, None]), 0.0)
        return n / self.speed, -g, g


def build_graph(box_halfwidth=None):
    """Three two-node tasks, each one role with a pinned (or, for the first
    node of each task when box_halfwidth is given, boxed) location."""
    graph = GraphOfConstraints([[Block.R(DIM)]] * N_AGENTS, [], state_lower_bound=LB,
                               state_upper_bound=UB, robot_names=["a0", "a1"], object_names=[])
    for first, second in ROLE_SITES:
        n_a, n_b = graph.structure.add_nodes(2)
        graph.structure.add_edge(n_a, n_b, True)
        r = graph.add_variable()
        q = graph.var_agent_q(r)
        if box_halfwidth is None:
            graph.add_constraint(n_a, eq(q, first))
        else:
            graph.add_constraint(n_a, ge(q, first - box_halfwidth))
            graph.add_constraint(n_a, le(q, first + box_halfwidth))
        graph.add_constraint(n_b, eq(q, second))
    return graph


def solve(graph, objective, models=(), **kwargs):
    splines = [CubicConfigurationSpline([Block.R(DIM)]) for _ in range(N_AGENTS)]
    mpc = GCSWaypointMPC(graph, splines, solver=WaypointSolver.kGurobi, objective=objective,
                         travel_time_models=list(models), **kwargs)
    remaining = list(range(graph.structure.num_nodes))
    assert mpc.solve(remaining, X0), f"{objective} solve failed"
    t = np.asarray(mpc.view_t())
    return {
        "routes": [list(r) for r in mpc.view_routes()],
        "var_assignments": np.asarray(mpc.view_var_assignments()).tolist(),
        "makespan": t[-1],
        "waypoints": np.asarray(mpc.view_waypoints()).copy(),
    }


def total_route_length(res):
    total = 0.0
    for j, route in enumerate(res["routes"]):
        pts = [X0[DIM * j:DIM * (j + 1)]] + [res["waypoints"][v, DIM * j:DIM * (j + 1)] for v in route]
        total += sum(np.linalg.norm(b - a) for a, b in zip(pts, pts[1:]))
    return total


def main():
    models = [EuclideanTravelTime(SPEED) for _ in range(N_AGENTS)]

    print("1. pinned waypoints")
    graph = build_graph()
    for l2, tt in ((WaypointObjective.kMinMaxL2, WaypointObjective.kMinMaxTravelTime),
                   (WaypointObjective.kAvgL2, WaypointObjective.kAvgTravelTime)):
        ref = solve(graph, l2)
        res = solve(graph, tt, models)
        assert res["routes"] == ref["routes"], (res["routes"], ref["routes"])
        assert res["var_assignments"] == ref["var_assignments"]
        len_ref, len_res = total_route_length(ref), total_route_length(res)
        print(f"   {tt}: routes {res['routes']}, makespan {res['makespan']:.4f} "
              f"(L2 {ref['makespan']:.4f} / speed = {ref['makespan'] / SPEED:.4f}), "
              f"route length {len_res:.4f} (L2 {len_ref:.4f})")
        assert abs(res["makespan"] - ref["makespan"] / SPEED) < 1e-4
        assert abs(len_res - len_ref) < 1e-4

    print("2. waypoints confined to boxes of half-width 0.1")
    graph = build_graph(box_halfwidth=0.1)
    ref = solve(graph, WaypointObjective.kMinMaxL2)
    res = solve(graph, WaypointObjective.kMinMaxTravelTime, models, max_tangent_halfwidth=0.2)
    print(f"   routes {res['routes']} (L2 {ref['routes']}), makespan {res['makespan']:.4f} "
          f"(L2 / speed {ref['makespan'] / SPEED:.4f})")
    assert res["routes"] == ref["routes"]
    assert abs(res["makespan"] - ref["makespan"] / SPEED) < 0.1 / SPEED * 4

    print("3. max_tangent_halfwidth below the box half-width raises")
    try:
        solve(graph, WaypointObjective.kMinMaxTravelTime, models, max_tangent_halfwidth=0.05)
    except RuntimeError as e:
        print(f"   raised: {e}")
    else:
        raise AssertionError("expected max_tangent_halfwidth to raise")

    print("all checks passed")


if __name__ == "__main__":
    main()

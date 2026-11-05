"""
SCIPWaypointMPC vs GCSWaypointMPC (Gurobi)
==========================================
Same graphs as test_gcs_travel_time.py. For each objective, SCIPWaypointMPC
must match GCSWaypointMPC's routes, assignments and optimal value (makespan
under min-max objectives, total route length under average ones, which is
exact on pinned sites):

  1. a cold solve over every node,
  2. a second, warm-started solve,
  3. a solve with the first task's first node passed (MPC-style shrinking).

Usage:
  python test_scip_waypoint_mpc.py
"""

import time

import numpy as np

from goc_mpc import GCSWaypointMPC, SCIPWaypointMPC, WaypointObjective, WaypointSolver
from goc_mpc._ext.configuration_spline import Block, CubicConfigurationSpline

from test_gcs_travel_time import DIM, N_AGENTS, SPEED, X0, EuclideanTravelTime, build_graph, total_route_length

CASES = [
    ("pinned kMinMaxL2", False, WaypointObjective.kMinMaxL2, {}),
    ("pinned kAvgL2", False, WaypointObjective.kAvgL2, {}),
    ("pinned kAvgL1", False, WaypointObjective.kAvgL1, {}),
    ("pinned kMinMaxL1", False, WaypointObjective.kMinMaxL1, {}),
    ("pinned kMinMaxTravelTime", False, WaypointObjective.kMinMaxTravelTime, {}),
    ("boxed kMinMaxTravelTime", True, WaypointObjective.kMinMaxTravelTime, {"max_tangent_halfwidth": 0.2}),
]


def make(cls, graph, objective, tt, kwargs):
    splines = [CubicConfigurationSpline([Block.R(DIM)]) for _ in range(N_AGENTS)]
    models = [EuclideanTravelTime(SPEED) for _ in range(N_AGENTS)] if tt else []
    extra = {"solver": WaypointSolver.kGurobi} if cls is GCSWaypointMPC else {}
    return cls(graph, splines, objective=objective, travel_time_models=models, **extra, **kwargs)


def summary(mpc, remaining):
    t = np.asarray(mpc.view_t())
    return {
        "makespan": float(t[-1]),
        "routes": [list(r) for r in mpc.view_routes()],
        "var_assignments": np.asarray(mpc.view_var_assignments()).tolist(),
        "waypoints": np.asarray(mpc.view_waypoints()).copy(),
    }


def compare(label, ref, res, minmax):
    if minmax:
        assert abs(res["makespan"] - ref["makespan"]) < 1e-4, (label, res["makespan"], ref["makespan"])
    else:
        assert abs(total_route_length(res) - total_route_length(ref)) < 1e-4, label
    assert res["routes"] == ref["routes"], (label, res["routes"], ref["routes"])
    assert res["var_assignments"] == ref["var_assignments"], label


def main():
    for name, boxed, objective, kwargs in CASES:
        tt = objective in (WaypointObjective.kMinMaxTravelTime, WaypointObjective.kAvgTravelTime)
        graph = build_graph(box_halfwidth=0.1 if boxed else None)
        gcs = make(GCSWaypointMPC, graph, objective, tt, kwargs)
        scip = make(SCIPWaypointMPC, graph, objective, tt, kwargs)
        all_nodes = list(range(graph.structure.num_nodes))
        print(name)
        for step, remaining in (("cold", all_nodes), ("warm", all_nodes), ("node 0 passed", all_nodes[1:])):
            assert gcs.solve(remaining, X0)
            tic = time.perf_counter()
            assert scip.solve(remaining, X0)
            wall = time.perf_counter() - tic
            ref, res = summary(gcs, remaining), summary(scip, remaining)
            minmax = "MinMax" in str(objective)
            compare(f"{name} / {step}", ref, res, minmax)
            value, ref_value = ((res["makespan"], ref["makespan"]) if minmax
                                else (total_route_length(res), total_route_length(ref)))
            print(f"   {step:14s} {'makespan' if minmax else 'length'} {value:.6f} (gurobi {ref_value:.6f})  "
                  f"routes {res['routes']}  "
                  f"scip {wall:.2f}s / gurobi {gcs.get_last_solve_time():.2f}s  "
                  f"[{scip.view_stats().status}, {scip.view_stats().nodes} nodes]")
    print("all checks passed")


if __name__ == "__main__":
    main()

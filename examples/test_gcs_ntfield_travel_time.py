"""
GCSWaypointMPC with a TensorRT NTField travel time
==================================================
One UR5e (joint space, R^6) visits N configurations in any order, with
kAvgTravelTime and the fast-ntfields pair-gradient engine as its
ArcTravelTimeModel (through fast_ntfields.pair_grad_ctypes, so no torch here).

  1. Every configuration pinned: the route's time must equal T summed along
     it, and its order must be the brute-force optimum under T. Also reports
     whether the plain-L2 order differs.
  2. Configurations confined to boxes of half-width BOX: reports the
     linearized route time against T re-evaluated at the solution's
     waypoints.

Usage:
  python test_gcs_ntfield_travel_time.py [--fast-ntfields DIR] [--engines DIR] [--n 6] [--seeds 20]
"""

import argparse
import itertools
import pathlib
import sys
import time

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

DIM = 6
BOX = 0.05


class NTFieldTravelTime(ArcTravelTimeModel):
    def __init__(self, engine, lower, upper, max_travel_time):
        super().__init__(lower, upper, max_travel_time)
        self.engine = engine
        self.calls = []

    def linearize(self, X, Y):
        t0 = time.perf_counter()
        out = self.engine.evaluate(X, Y)
        self.calls.append((len(X), time.perf_counter() - t0))
        return out


def build_graph(targets, lower, upper, box=None):
    graph = GraphOfConstraints([[Block.R(DIM)]], [], state_lower_bound=float(lower.min()),
                               state_upper_bound=float(upper.max()), robot_names=["ur5e"], object_names=[])
    nodes = graph.structure.add_nodes(len(targets))
    q = graph.agent_q(0)
    for n, target in zip(nodes, targets):
        if box is None:
            graph.add_constraint(n, eq(q, target))
        else:
            graph.add_constraint(n, ge(q, target - box))
            graph.add_constraint(n, le(q, target + box))
    return graph


def solve(graph, x0, objective, models=(), **kwargs):
    mpc = GCSWaypointMPC(graph, [CubicConfigurationSpline([Block.R(DIM)])], solver=WaypointSolver.kGurobi,
                         objective=objective, travel_time_models=list(models), **kwargs)
    t0 = time.perf_counter()
    assert mpc.solve(list(range(graph.structure.num_nodes)), x0), f"{objective} solve failed"
    wall = time.perf_counter() - t0
    route = list(mpc.view_routes()[0])
    wps = np.asarray(mpc.view_waypoints())[:, :DIM].copy()
    return route, np.asarray(mpc.view_t()), wps, wall


def route_time(engine, x0, points):
    pts = np.vstack([x0, points])
    return float(engine.evaluate(pts[:-1], pts[1:])[0].sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast-ntfields", default="/home/tassos/phd/software/fast-ntfields")
    parser.add_argument("--engines", default=None, help="default: <fast-ntfields>/engines/ur5e_pair_grad")
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()
    root = pathlib.Path(args.fast_ntfields)
    sys.path.insert(0, str(root / "src"))
    from fast_ntfields.pair_grad_ctypes import PairGradEngine

    import json
    engine_dir = pathlib.Path(args.engines or root / "engines" / "ur5e_pair_grad")
    engine = PairGradEngine(root / "cpp" / "build" / "libfast_ntfields_c.so", engine_dir)
    bounds = np.asarray(json.loads((engine_dir / "meta.json").read_text())["bounds"])
    lower, upper = bounds[:, 0], bounds[:, 1]

    rng = np.random.default_rng(0)
    pairs = rng.uniform(lower, upper, (20000, 2, DIM))
    t_max = float(engine.evaluate(pairs[:, 0], pairs[:, 1])[0].max())
    model = NTFieldTravelTime(engine, lower, upper, 2.0 * t_max)
    print(f"field domain {lower} .. {upper}; sampled max T {t_max:.3f}, max_travel_time {model.max_travel_time:.3f}")

    print(f"1. pinned: {args.seeds} random instances of {args.n} configurations")
    # Keep targets away from the domain edge so BOX-sized boxes stay inside it.
    inner_lo, inner_hi = lower + BOX, upper - BOX
    differs = 0
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        x0 = rng.uniform(inner_lo, inner_hi)
        targets = rng.uniform(inner_lo, inner_hi, (args.n, DIM))
        graph = build_graph(targets, lower, upper)
        route, t, _, wall = solve(graph, x0, WaypointObjective.kAvgTravelTime, [model])
        solved = route_time(engine, x0, targets[route])
        best = min(itertools.permutations(range(args.n)), key=lambda p: route_time(engine, x0, targets[list(p)]))
        best_time = route_time(engine, x0, targets[list(best)])
        l2_route, *_ = solve(graph, x0, WaypointObjective.kAvgL2)
        differs += l2_route != route
        print(f"   seed {seed:2d}: route {route} T-sum {solved:.4f} = model {t[-1]:.4f}; brute force "
              f"{list(best)} {best_time:.4f}; L2 route {'differs' if l2_route != route else 'same'} "
              f"({route_time(engine, x0, targets[l2_route]):.4f}); solve {wall * 1e3:.0f} ms")
        assert abs(solved - t[-1]) < 1e-3 * max(1.0, solved), "model time != T along the route"
        assert solved <= best_time + 1e-3 * max(1.0, best_time), "route is not the T-optimal order"
    print(f"   L2 order differs from the T order on {differs}/{args.seeds} instances")

    print(f"2. confined to boxes of half-width {BOX}")
    gaps = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        x0 = rng.uniform(inner_lo, inner_hi)
        targets = rng.uniform(inner_lo, inner_hi, (args.n, DIM))
        graph = build_graph(targets, lower, upper, box=BOX)
        route, t, wps, wall = solve(graph, x0, WaypointObjective.kAvgTravelTime, [model],
                                    max_tangent_halfwidth=2 * BOX)
        true_time = route_time(engine, x0, wps[route])
        gaps.append((t[-1] - true_time) / true_time)
        print(f"   seed {seed:2d}: route {route} model {t[-1]:.4f} vs T at solution {true_time:.4f} "
              f"(gap {gaps[-1]:+.2%}); solve {wall * 1e3:.0f} ms")
    print(f"   relative gap model - T: median {np.median(gaps):+.2%}, worst {min(gaps, key=abs):+.2%} .. "
          f"{max(gaps, key=abs):+.2%}")

    sizes, secs = zip(*model.calls)
    print(f"linearize calls: {len(sizes)}, batch {min(sizes)}..{max(sizes)}, "
          f"median {np.median(secs) * 1e6:.0f} us per call")
    print("all checks passed")


if __name__ == "__main__":
    main()

"""GCSWaypointMPC / SCIPWaypointMPC with `fields=`.

Agent 0 (`Block.R(2)`) visits N pinned points; its arcs are priced by a small
TD-NTFields field trained around a sphere (`AgentField`). Agent 1 has no field
(None) and is priced by the L2 distance. Checks:

  1. `FieldTravelTimeModel`'s gradients match finite differences of its times,
     which are the field's times in the agent's units.
  2. With every point pinned, the solver's visiting order for agent 0 is the
     brute-force optimum under the field's own travel time.
  3. `fields=` with a non-travel-time objective, or together with
     travel_time_models=, raises.

Needs the `ntfields` extra (torch). Run:
    python examples/test_waypoint_fields.py
"""

import itertools

import numpy as np
import ntfields
from ntfields.envs import SphereBoxEnv
from ntfields.speed import Clipped

from goc_mpc import AgentField, GCSWaypointMPC, GraphOfConstraints, SCIPWaypointMPC, WaypointObjective
from goc_mpc._ext.configuration_spline import CubicConfigurationSpline
from goc_mpc.fields import FieldTravelTimeModel
from goc_mpc.splines import Block

N = 5
OFFSET, SCALE = np.array([1.0, -0.5]), 2.0


def train_field():
    problem = ntfields.Problem(
        SphereBoxEnv.build(bounds=[(-1, 1)] * 2, spheres=[((0.0, 0.0), 0.3)]), Clipped(0.01, 0.1))
    train_cfg = ntfields.TrainConfig(n_pairs=200_000, batch_size=2000, batches_per_epoch=5, out_dir=None)
    return ntfields.train("td_ntfields_2d", problem, ntfields.Epochs(200), train_cfg=train_cfg)


def build_graph(targets0, targets1):
    spec = [Block.R(2)]
    graph = GraphOfConstraints([spec, spec], [], state_lower_bound=-4.0, state_upper_bound=4.0,
                               robot_names=["robot0", "robot1"], object_names=[])
    graph.structure.add_nodes(2 * N)
    for agent, targets in ((0, targets0), (1, targets1)):
        q = graph.agent_q(agent)
        for i, p in enumerate(targets):
            for k in range(2):
                graph.add_constraint(agent * N + i, q[k] == float(p[k]))
    return graph, spec


def main():
    field = AgentField(train_field(), OFFSET, SCALE)
    model = FieldTravelTimeModel(field, n_samples=512)

    rng = np.random.default_rng(0)
    X = OFFSET + SCALE * rng.uniform(-0.8, 0.8, (6, 2))
    Y = OFFSET + SCALE * rng.uniform(-0.8, 0.8, (6, 2))
    t, g_x, g_y = model.linearize(X, Y)
    for k in range(2):
        e = np.zeros(2)
        e[k] = 1e-3  # central differences; smaller steps hit float32 round-off
        fd_x = (model.linearize(X + e, Y)[0] - model.linearize(X - e, Y)[0]) / 2e-3
        fd_y = (model.linearize(X, Y + e)[0] - model.linearize(X, Y - e)[0]) / 2e-3
        assert np.allclose(g_x[:, k], fd_x, rtol=1e-2, atol=1e-3)
        assert np.allclose(g_y[:, k], fd_y, rtol=1e-2, atol=1e-3)
    print(f"gradients match finite differences; max_travel_time {model.max_travel_time:.3f}")

    targets0 = OFFSET + SCALE * rng.uniform(-0.8, 0.8, (N, 2))
    targets1 = rng.uniform(-3.0, 3.0, (N, 2))
    x0 = np.concatenate([OFFSET + SCALE * rng.uniform(-0.8, 0.8, 2), rng.uniform(-3.0, 3.0, 2)])
    T = lambda a, b: float(model.linearize(a[None], b[None])[0][0])

    def route_time(order):
        pts = [x0[:2]] + [targets0[i] for i in order]
        return sum(T(pts[k], pts[k + 1]) for k in range(N))

    best = min(itertools.permutations(range(N)), key=route_time)
    for name, cls, extra in (("SCIP", SCIPWaypointMPC, {}), ("GCS", GCSWaypointMPC, {"mip_gap": 1e-6})):
        graph, spec = build_graph(targets0, targets1)
        mpc = cls(graph, [CubicConfigurationSpline(spec)] * 2, fields=[field, None],
                  objective=WaypointObjective.kAvgTravelTime, **extra)
        assert mpc.solve(list(range(2 * N)), x0), f"{name} solve failed"
        route = list(mpc.view_routes()[0])
        print(f"{name}: agent 0 route {route} ({route_time(route):.3f}); brute force {list(best)} "
              f"({route_time(best):.3f}); agent 1 route {list(mpc.view_routes()[1])}")
        assert route == list(best), f"{name} route is not the field-optimal order"

    graph, spec = build_graph(targets0, targets1)
    for bad in ({"objective": WaypointObjective.kMinMaxL1}, {"travel_time_models": []}):
        try:
            SCIPWaypointMPC(graph, [CubicConfigurationSpline(spec)] * 2, fields=[field, None], **bad)
        except ValueError:
            continue
        raise AssertionError(f"fields= with {bad} did not raise")
    print("PASS")


if __name__ == "__main__":
    main()

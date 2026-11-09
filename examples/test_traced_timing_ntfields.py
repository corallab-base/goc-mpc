"""TracedTimingMPC with an ntfields field.

A point mass (`Block.R(3)`) moves between two waypoints on opposite sides of
a sphere in the z = 1 plane, so the straight chord between them runs through
it. A small TD-NTFields field is trained over the plane (field coordinates
`(x, y) / 1.0`, the sphere at the origin), and the agent's z column is
interpolated along each planned path. The traced spline must stay out of the
sphere, while plain GraphTimingMPC's chord goes through it. A second agent with
no field (`None`) gets straight segments.

Needs the `ntfields` extra (torch). Run:
    python examples/test_traced_timing_ntfields.py
"""

import numpy as np
import ntfields
from ntfields.envs import SphereBoxEnv
from ntfields.speed import Clipped

from goc_mpc import GraphOfConstraints, GraphTimingMPC
from goc_mpc._ext.configuration_spline import CubicConfigurationSpline
from goc_mpc.splines import Block
from goc_mpc.traced_timing_mpc import AgentField, TracedTimingMPC

RADIUS = 0.3
Z = 1.0
START = np.array([-0.6, 0.0, Z])
GOAL = np.array([0.6, 0.0, Z])


def train_field():
    problem = ntfields.Problem(
        SphereBoxEnv.build(bounds=[(-1, 1)] * 2, spheres=[((0.0, 0.0), RADIUS)]), Clipped(0.01, 0.1))
    train_cfg = ntfields.TrainConfig(n_pairs=500_000, batch_size=2000, batches_per_epoch=5, out_dir=None)
    return ntfields.train("td_ntfields_2d", problem, ntfields.Epochs(500), train_cfg=train_cfg)


def build_graph():
    spec = [Block.R(3)]
    graph = GraphOfConstraints(
        [spec, spec], [], state_lower_bound=-3.0, state_upper_bound=3.0,
        robot_names=["robot0", "robot1"], object_names=[])
    graph.structure.add_nodes(2)
    for agent, node in ((0, 0), (1, 1)):
        q = graph.agent_q(agent)
        for k in range(3):
            graph.add_constraint(node, q[k] == float(GOAL[k] if agent == 0 else GOAL[k] + 2.0))
    return graph, spec


def sampled(timing, spec, agent, x0, v0):
    splines = [CubicConfigurationSpline(spec), CubicConfigurationSpline(spec)]
    timing.fill_cubic_splines(splines, x0, v0)
    s = splines[agent]
    return np.array([s.eval(t)[0] for t in np.linspace(s.begin(), s.end(), 200)])


def inside(points):
    return float((np.linalg.norm(points[:, :2], axis=1) < RADIUS).mean())


def main():
    field = train_field()
    graph, spec = build_graph()
    x0 = np.concatenate([START, START + 2.0])
    v0 = np.zeros(6)
    waypoints = np.stack([np.concatenate([GOAL, GOAL + 2.0])] * 2)
    assignments = np.array([], dtype=np.int32)

    chord = GraphTimingMPC(graph, [CubicConfigurationSpline(spec)] * 2, time_cost=1.0)
    assert chord.solve(x0, v0, [0, 1], waypoints, assignments)
    chord_inside = inside(sampled(chord, spec, 0, x0, v0))
    print(f"chord: {chord_inside:.2f} of samples inside the sphere")
    assert chord_inside > 0.2

    traced = TracedTimingMPC(
        graph, [CubicConfigurationSpline(spec)] * 2,
        [AgentField(field, offset=(0.0, 0.0), scale=1.0), None], time_cost=1.0)
    assert traced.solve(x0, v0, [0, 1], waypoints, assignments)
    points = sampled(traced, spec, 0, x0, v0)
    print(f"traced: {inside(points):.3f} inside, z in [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}], "
          f"{traced.last_failed_segments} failed segments, {len(traced.view_wps_list()[0])} dense waypoints")
    assert inside(points) < 0.02 and np.allclose(points[:, 2], Z, atol=1e-6)
    assert np.allclose(points[-1], GOAL, atol=1e-6)
    other = sampled(traced, spec, 1, x0, v0)
    assert np.allclose(other[-1], GOAL + 2.0, atol=1e-6)
    print("PASS")


if __name__ == "__main__":
    main()

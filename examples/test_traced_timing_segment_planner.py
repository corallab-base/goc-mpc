"""TracedTimingMPC with a segment planner, on a planar base `[R(2), Torus(1)]`.

The planner routes the agent's (x, y) through a via point, which must become
a traced interior waypoint (node id -1). The heading starts at -pi and ends at
+pi, the same angle: the interpolated heading must stay at +-pi rather than
sweep a full turn.

Run:
    python examples/test_traced_timing_segment_planner.py
"""

import numpy as np

from goc_mpc import GraphOfConstraints
from goc_mpc.splines import Block
from goc_mpc.traced_timing_mpc import TracedTimingMPC

START = np.array([-2.0, 0.0, -np.pi])
GOAL = np.array([2.0, 0.0, np.pi])
VIA = np.array([0.0, 1.0])


def route(start, goal):
    return np.array([start[:2], VIA, goal[:2]])


def main():
    spec = [Block.R(2), Block.Torus(1)]
    graph = GraphOfConstraints([spec], [], state_lower_bound=-5.0, state_upper_bound=5.0,
                               robot_names=["base"], object_names=[], workspace_dim=2)
    q = graph.agent_q(0)
    nodes = graph.structure.add_nodes(1)
    for k in range(3):
        graph.add_constraint(nodes[0], q[k] == float(GOAL[k]))

    from goc_mpc._ext.configuration_spline import CubicConfigurationSpline
    timing = TracedTimingMPC(graph, [CubicConfigurationSpline(spec)], [None], segment_planners=[route])
    traced = timing.plan_segment(0, START, GOAL)
    assert np.allclose(traced[:, :2], route(START, GOAL)), traced
    assert np.allclose(np.abs(traced[:, 2]), np.pi), f"heading sweeps instead of staying at +-pi: {traced[:, 2]}"
    assert traced[0, 2] == START[2] and traced[-1, 2] == GOAL[2]

    waypoints = np.zeros((graph.structure.num_nodes, 3))
    waypoints[nodes[0]] = GOAL
    assert timing.solve(START, np.zeros(3), [nodes[0]], waypoints, np.array([], dtype=int))
    wps = np.asarray(timing.view_wps_list()[0])
    ids = list(timing.view_agent_nodes_list()[0])
    assert ids == [-1, nodes[0]], ids
    assert np.allclose(wps[0, :2], VIA), wps
    print("OK")


if __name__ == "__main__":
    main()

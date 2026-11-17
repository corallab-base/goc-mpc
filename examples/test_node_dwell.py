"""Node dwells: `graph.set_node_dwell(node, seconds)`.

The agent visiting a dwell node holds still there for `seconds`: the timing
solve fixes its knot velocity to zero, every later arrival time in its path
shifts by the dwell (cross-agent orderings included -- a LESS_THAN source
counts as left at the end of its dwell, an EQUAL node as reached at the
start), and the spline gets a constant piece of that length. Typical use: an
arm pauses at a pick or place pose while the executor runs the grasp there.

Run: python examples/test_node_dwell.py
"""

import numpy as np
from pydrake.math import eq

from goc_mpc import GraphOfConstraints, GraphOfConstraintsMPC
from goc_mpc._ext.configuration_spline import CubicConfigurationSpline
from goc_mpc.splines import Block
from goc_mpc.traced_timing_mpc import TracedTimingMPC

DWELL = 1.5


def chain(graph, agent, targets):
    """Nodes in a hard chain, node k pinning `agent` at targets[k]."""
    q = graph.agent_q(agent)
    nodes = list(graph.structure.add_nodes(len(targets)))
    for u, w in zip(nodes, nodes[1:]):
        graph.structure.add_edge(u, w, True)
    for node, target in zip(nodes, targets):
        graph.add_constraint(node, eq(q, np.asarray(target, dtype=float)))
    return nodes


def solve(graph, x0):
    mpc = GraphOfConstraintsMPC(graph, time_cost=0.05, acceleration_cost=1.0, phi_tolerance=0.02)
    mpc.reset()
    assert mpc._solve_for_waypoints(x0) and mpc._solve_for_timing(x0, np.zeros_like(x0))
    return mpc


def arrival(mpc, agent, node):
    """When `agent` reaches `node` (dwells before it included, its own not)."""
    nodes = list(mpc.timing_mpc.view_agent_nodes_list()[agent])
    k = nodes.index(node)
    deltas = np.asarray(mpc.timing_mpc.view_time_deltas_list()[agent])
    dwells = np.asarray(mpc.timing_mpc.view_agent_dwells_list()[agent])
    return deltas[:k + 1].sum() + dwells[:k].sum()


def assert_holds(spline, t0, t1, position):
    for t in np.linspace(t0, t1, 25):
        q, v, _a = spline.eval(float(t))
        assert np.allclose(q, position, atol=1e-9), (t, q, position)
        assert np.allclose(v, 0.0, atol=1e-9), (t, v)


def single_agent_graph():
    graph = GraphOfConstraints([[Block.R(2)]], [], state_lower_bound=-10.0, state_upper_bound=10.0,
                               robot_names=["r"], object_names=[], workspace_dim=2)
    nodes = chain(graph, 0, [(1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    return graph, nodes


def test_single_agent_holds():
    graph, nodes = single_agent_graph()
    graph.set_node_dwell(nodes[1], DWELL)
    assert graph.get_node_dwell(nodes[1]) == DWELL and graph.get_node_dwell(nodes[0]) == 0.0
    mpc = solve(graph, np.zeros(2))

    dwells = np.asarray(mpc.timing_mpc.view_agent_dwells_list()[0])
    assert np.allclose(dwells, [0.0, DWELL, 0.0]), dwells
    assert np.allclose(mpc.timing_mpc.view_vs_list()[0][1], 0.0), "zero knot velocity at the dwell node"

    spline = mpc.last_cycle_splines[0]
    deltas = np.asarray(mpc.timing_mpc.view_time_deltas_list()[0])
    assert np.isclose(spline.end() - spline.begin(), deltas.sum() + DWELL), (spline.end(), deltas)
    t1 = arrival(mpc, 0, nodes[1])
    assert_holds(spline, t1, t1 + DWELL, [1.0, 1.0])
    q, _v, _a = spline.eval(float(spline.end()))
    assert np.allclose(q, [0.0, 1.0]), q

    graph.set_node_dwell(nodes[1], 0.0)
    assert not graph.node_dwell_map, "seconds <= 0 removes the dwell"
    print("test_single_agent_holds: all checks passed.")


def two_agent_graph():
    """Agent 0 visits a0 -> a1; agent 1 visits b0 -> b1, with a1 -> b1 a
    cross-agent ordering (LESS_THAN) and agent 1 starting right at b1's
    neighbourhood so it would otherwise arrive first."""
    graph = GraphOfConstraints([[Block.R(2)], [Block.R(2)]], [], state_lower_bound=-10.0,
                               state_upper_bound=10.0, robot_names=["r0", "r1"], object_names=[],
                               workspace_dim=2)
    a = chain(graph, 0, [(1.0, 0.0), (2.0, 0.0)])
    b = chain(graph, 1, [(0.0, 3.1), (0.0, 3.2)])
    graph.structure.add_edge(a[1], b[1], True)
    return graph, a, b


def test_less_than():
    x0 = np.array([0.0, 0.0, 0.0, 3.0])
    graph, a, b = two_agent_graph()
    mpc = solve(graph, x0)
    gap = arrival(mpc, 1, b[1]) - arrival(mpc, 0, a[1])
    assert gap < DWELL - 0.1, f"without a dwell, b1 needn't wait {DWELL}s after a1 ({gap})"

    graph, a, b = two_agent_graph()
    graph.set_node_dwell(a[1], DWELL)
    mpc = solve(graph, x0)
    gap = arrival(mpc, 1, b[1]) - arrival(mpc, 0, a[1])
    assert gap >= DWELL - 1e-3, f"b1 waits for agent 0 to leave a1, a dwell later ({gap})"
    print("test_less_than: all checks passed.")


def test_equal():
    """A node both agents pin is an EQUAL interaction: both reach it at the
    same time, including agent 0's dwell at an earlier node of its own."""
    graph = GraphOfConstraints([[Block.R(2)], [Block.R(2)]], [], state_lower_bound=-10.0,
                               state_upper_bound=10.0, robot_names=["r0", "r1"], object_names=[],
                               workspace_dim=2)
    a = chain(graph, 0, [(1.0, 0.0)])
    shared = list(graph.structure.add_nodes(1))[0]
    graph.add_constraint(shared, eq(graph.agent_q(0), np.array([2.0, 0.0])))
    graph.add_constraint(shared, eq(graph.agent_q(1), np.array([2.0, 0.5])))
    graph.structure.add_edge(a[0], shared, True)
    graph.set_node_dwell(a[0], DWELL)
    mpc = solve(graph, np.array([0.0, 0.0, 2.0, 0.4]))
    t0, t1 = arrival(mpc, 0, shared), arrival(mpc, 1, shared)
    assert abs(t0 - t1) < 1e-3, (t0, t1)
    assert t0 >= DWELL, t0
    print("test_equal: all checks passed.")


def test_traced():
    """solve_dense: a segment planner adds a traced interior point (-1) before
    the dwell node; the dwell lands on the real node only."""
    spec = [Block.R(2)]
    graph = GraphOfConstraints([spec], [], state_lower_bound=-10.0, state_upper_bound=10.0,
                               robot_names=["r"], object_names=[], workspace_dim=2)
    nodes = chain(graph, 0, [(2.0, 0.0), (2.0, 2.0)])
    graph.set_node_dwell(nodes[0], DWELL)
    via = np.array([1.0, 1.0])
    timing = TracedTimingMPC(graph, [CubicConfigurationSpline(spec)], [None],
                             segment_planners=[lambda s, g: np.array([s[:2], via, g[:2]])])
    waypoints = np.array([[2.0, 0.0], [2.0, 2.0]])
    assert timing.solve(np.zeros(2), np.zeros(2), list(nodes), waypoints, np.array([], dtype=int))
    ids = list(timing.view_agent_nodes_list()[0])
    dwells = np.asarray(timing.view_agent_dwells_list()[0])
    assert -1 in ids and len(dwells) == len(ids), (ids, dwells)
    assert np.allclose(dwells, [DWELL if i == nodes[0] else 0.0 for i in ids]), (ids, dwells)

    spline = CubicConfigurationSpline(spec)
    timing.fill_cubic_splines([spline], np.zeros(2), np.zeros(2))
    k = ids.index(nodes[0])
    deltas = np.asarray(timing.view_time_deltas_list()[0])
    t_arrive = deltas[:k + 1].sum()
    assert_holds(spline, t_arrive, t_arrive + DWELL, [2.0, 0.0])
    assert np.isclose(spline.end(), deltas.sum() + DWELL), (spline.end(), deltas)
    print("test_traced: all checks passed.")


if __name__ == "__main__":
    test_single_agent_holds()
    test_less_than()
    test_equal()
    test_traced()

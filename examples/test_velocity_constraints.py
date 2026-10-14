"""Velocity constraints: `add_constraint(node, f)` with `agent_v(i)` placeholders.

An agent's velocity at a node (its tangent space) constrains the timing
solve's knot velocity there, not the waypoints. A single-component row is a
bound on that knot-velocity variable (an equality fixes it); a row in several
components is a slack-relaxed QP row, like the cross-agent interaction rows.
Typical use: a mobile manipulator's base stops at a node (velocity zero) while
its arm flows through -- unconstrained, the cubic overshoots a part that
should stay put.

Run: python examples/test_velocity_constraints.py
"""

import numpy as np
from pydrake.math import eq

from goc_mpc import GraphOfConstraints, GraphOfConstraintsMPC
from goc_mpc.splines import Block

X0 = np.zeros(3)


def build(add_velocity=None):
    """One agent: a 2-D base and a 1-D arm. The base goes to x = 1 by node 1
    and stays; the arm keeps moving through every node."""
    graph = GraphOfConstraints(
        [[Block.R(2), Block.R(1)]], [], state_lower_bound=-10.0, state_upper_bound=10.0,
        robot_names=["r"], object_names=[], workspace_dim=2)
    q, v = graph.agent_q(0), graph.agent_v(0)
    nodes = list(graph.structure.add_nodes(4))
    for u, w in zip(nodes, nodes[1:]):
        graph.structure.add_edge(u, w, True)
    for node, (base_x, arm) in zip(nodes, [(0.5, 0.5), (1.0, 1.0), (1.0, 1.5), (1.0, 2.0)]):
        graph.add_constraint(node, eq(q, np.array([base_x, 0.0, arm])))
    if add_velocity is not None:
        add_velocity(graph, nodes, v)
    return graph


def solve(graph):
    """The interior knot velocities (one row per knot), and the spline."""
    mpc = GraphOfConstraintsMPC(graph, time_cost=0.05, acceleration_cost=1.0, phi_tolerance=0.02)
    mpc.reset()
    assert mpc._solve_for_waypoints(X0) and mpc._solve_for_timing(X0, np.zeros(3))
    return mpc.timing_mpc.view_vs_list()[0][:3], mpc.last_cycle_splines[0]


def max_base_x(spline):
    return max(spline.eval(t)[0][0] for t in np.linspace(spline.begin(), spline.end(), 400))


def test_base_stops():
    vs, spline = solve(build())
    assert max_base_x(spline) > 1.0 + 1e-3, "unconstrained, the base overshoots x = 1"
    vs, spline = solve(build(lambda g, n, v: g.add_constraint(n[1], eq(v[:2], np.zeros(2)))))
    assert np.allclose(vs[1, :2], 0.0), vs
    assert abs(vs[1, 2]) > 1e-2, "the arm still moves through node 1"
    assert max_base_x(spline) <= 1.0 + 1e-6, "stopped at node 1, the base doesn't overshoot"
    print("test_base_stops: all checks passed.")


def test_inequalities():
    vs, _ = solve(build(lambda g, n, v: g.add_constraint(n[2], v[2] <= 0.05)))
    assert vs[2, 2] <= 0.05 + 1e-6, vs
    vs, _ = solve(build(lambda g, n, v: g.add_constraint(n[0], v[0] + v[2] <= 0.1)))
    assert vs[0, 0] + vs[0, 2] <= 0.1 + 1e-4, vs
    print("test_inequalities: all checks passed.")


def test_rejected():
    cases = {
        "mixed with positions": lambda g, n, v: g.add_constraint(n[0], v[0] == g.agent_q(0)[0]),
        "nonlinear": lambda g, n, v: g.add_constraint(n[0], v[0] * v[1] <= 1.0),
        "on an edge": lambda g, n, v: g.add_edge_constraint(n[0], n[1], v[0] == 0.0),
    }
    for name, add in cases.items():
        try:
            build(add)
        except Exception:
            continue
        raise AssertionError(f"a velocity constraint {name} wasn't rejected")
    print("test_rejected: all checks passed.")


if __name__ == "__main__":
    test_base_stops()
    test_inequalities()
    test_rejected()

"""`GraphOfConstraintsMPC(waypoint_stages=...)`: a staged kinematic solve.

Each stage sees the waypoints the solve just produced, may edit the graph's
runtime PARAMETERS, and returns whether that edit is worth re-solving for.
The motivating case is a loco-manipulator's grasp -- pass 1 solves for where
to STAND, a stage reads that stance and switches on the arm constraints
(a `ready` parameter multiplying them), pass 2 solves the arm -- but the
mechanism is just "solve, edit params, solve again", tested here on a bare
two-node graph whose node target IS a parameter.

Run: python examples/test_waypoint_stages.py
"""

import numpy as np
from pydrake.math import eq

from goc_mpc import GraphOfConstraints, GraphOfConstraintsMPC
from goc_mpc._ext.configuration_spline import Block, CubicConfigurationSpline
from goc_mpc.evolutionary_waypoint_solver import EvolutionaryWaypointSolver

DIM = 2
FIRST = np.array([1.0, 0.0])     # the target the graph is BUILT with
SECOND = np.array([3.0, -2.0])   # what the stage re-points it at
X0 = np.zeros(DIM)
#: Loose vs. the 2 m between FIRST and SECOND -- this tests WHICH target the
#: solve chased, not how precisely a 12-member, 2-generation solve converged.
TOL = 2e-2


def build():
    """One agent, two nodes; node 1 pins it at `param`-held `FIRST`."""
    graph = GraphOfConstraints(
        [[Block.R(DIM)]], [], state_lower_bound=-20.0, state_upper_bound=20.0,
        robot_names=["r0"], object_names=[], workspace_dim=DIM)
    n0, n1 = graph.structure.add_nodes(2)
    graph.structure.add_edge(n0, n1, True)
    param_ids = [graph.add_param(float(FIRST[k])) for k in range(DIM)]
    graph.add_constraint(
        n1, eq(graph.agent_q(0), np.array([graph.param(p) for p in param_ids])))
    return graph, param_ids


def controller(graph, stages, max_passes=4):
    solver = EvolutionaryWaypointSolver(
        graph, [CubicConfigurationSpline(spec) for spec in graph._robot_specs],
        wp_bounds=(-20.0, 20.0), pop_size=12, n_gen=2, seed=0)
    return GraphOfConstraintsMPC(graph, waypoint_mpc=solver, waypoint_stages=stages,
                                 max_waypoint_passes=max_passes)


def test_stage_reruns_the_solve():
    """A stage that re-points the target once: the waypoints the controller
    ends the cycle with must satisfy the SECOND target, not the first -- i.e.
    the stage's parameter edit was followed by a real re-solve."""
    graph, param_ids = build()
    seen = []

    def stage(g, waypoints):
        seen.append(np.asarray(waypoints)[1, :DIM].copy())
        if len(seen) > 1:
            return False        # already re-pointed; nothing more to change
        for pid, v in zip(param_ids, SECOND):
            g.set_param(pid, float(v))
        return True

    ctrl = controller(graph, [stage])
    assert ctrl._solve_for_waypoints(X0)
    final = np.asarray(ctrl.last_cycle_waypoints)[1, :DIM]
    print(f"[stages] pass 1 saw {np.round(seen[0], 3).tolist()}, "
          f"pass 2 saw {np.round(seen[1], 3).tolist()}, final {np.round(final, 3).tolist()}")
    assert len(seen) == 2, seen
    assert np.allclose(seen[0], FIRST, atol=TOL), seen[0]
    assert np.allclose(final, SECOND, atol=TOL), final


def test_max_passes_bounds_a_stage_that_never_settles():
    """A stage that always claims it changed something stops at
    `max_waypoint_passes` solves rather than looping forever."""
    graph, _param_ids = build()
    calls = []

    def stage(g, waypoints):
        calls.append(None)
        return True

    ctrl = controller(graph, [stage], max_passes=3)
    assert ctrl._solve_for_waypoints(X0)
    print(f"[stages] never-settling stage ran {len(calls)} times under max_waypoint_passes=3")
    assert len(calls) == 3, calls


def test_no_stages_solves_once():
    """The default (no stages) keeps the single-solve behavior exactly."""
    graph, _param_ids = build()
    ctrl = controller(graph, [])
    assert ctrl._solve_for_waypoints(X0)
    final = np.asarray(ctrl.last_cycle_waypoints)[1, :DIM]
    print(f"[stages] no stages -> {np.round(final, 3).tolist()}")
    assert np.allclose(final, FIRST, atol=TOL), final


if __name__ == "__main__":
    test_stage_reruns_the_solve()
    test_max_passes_bounds_a_stage_that_never_settles()
    test_no_stages_solves_once()
    print("\nAll checks passed.")

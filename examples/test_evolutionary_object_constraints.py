"""Validation script for object_q / rigid-attachment support in
EvolutionaryWaypointSolver.

1 robot + 1 object, dim=2 (equal agent/object width). Two graph nodes:

  n0 (grasp): object_q(0) == OBJECT_START            (pure node-object pin)
              agent_q(0) == object_q(0) + GRASP_OFFSET (mixed node constraint --
                rigid attachment: agent_q AND object_q in one Formula)
  n1 (place): object_q(0) == OBJECT_TARGET            (pure node-object pin)
              agent_q(0) >= -20  (loose bound, purely to establish a routing
                instance for the robot at n1 -- see spec.py's module
                docstring: an edge constraint's v_agent_q(0) reference can
                only resolve against an ALREADY-established routing instance
                at that node, same restriction as before this feature)

  edge n0->n1: graph.add_hold(n0, n1, 0, [0]) -- object is carried rigidly by
               robot 0 (spec.py's _resolve_holds auto-derives the
               same v_object_q - u_object_q == v_agent_q - u_agent_q relation
               this used to hand-write via add_edge_constraint; see
               examples/test_hold_registry.py for the canonical-API test this
               was migrated to match). Using the canonical hold registry
               here (rather than a bare add_edge_constraint) matters now
               that the solver auto-derives a default "object stationary
               unless held" invariant on every OTHER structural edge
               (spec.py's Constraint-14b analogue) -- only an edge declared
               through add_hold/add_assignable_hold is recognized as exempt
               from that default.

Physically well-determined: object starts pinned, robot grasps it with a
fixed offset, then the edge constraint forces the object's node1 position to
equal the robot's node1 position (since both start coincident up to the same
offset and move together) -- so node1's object_q pin fully determines the
robot's node1 position too, transitively.

Run directly: python examples/test_evolutionary_object_constraints.py
"""

import numpy as np
from pydrake.math import eq, ge

from goc_mpc import EvolutionaryWaypointSolver, GraphOfConstraints
from goc_mpc._ext.configuration_spline import Block, CubicConfigurationSpline

DIM = 2
OBJECT_START = np.array([2.0, 0.0])
OBJECT_TARGET = np.array([5.0, 3.0])
GRASP_OFFSET = np.array([0.3, 0.0])
TOL = 0.15


def build_graph() -> GraphOfConstraints:
    robot_spec = [Block.R(DIM)]
    object_spec = [Block.R(DIM)]
    graph = GraphOfConstraints(
        [robot_spec], [object_spec],
        state_lower_bound=-20.0, state_upper_bound=20.0,
        robot_names=["robot"], object_names=["cube"],
    )
    # Equal-width here purely for this script's convenience (GRASP_OFFSET
    # arithmetic below) -- the solver itself does NOT require dim ==
    # non_robot_dim: each node's dense row concatenates independently-sized
    # agent (dim-wide) and object (non_robot_dim-wide) blocks, so mismatched
    # widths work out of the box (verified separately, not exercised here).
    assert graph.object_ambient_dim(0) == graph.robot_ambient_dim(0)

    n0, n1 = graph.structure.add_nodes(2)
    graph.structure.add_edge(n0, n1, True)

    q0, obj0 = graph.agent_q(0), graph.object_q(0)

    graph.add_constraint(n0, eq(obj0, OBJECT_START))
    graph.add_constraint(n0, eq(q0, obj0 + GRASP_OFFSET))

    graph.add_constraint(n1, eq(graph.object_q(0), OBJECT_TARGET))
    graph.add_constraint(n1, ge(graph.agent_q(0), np.array([-20.0, -20.0])))

    # Canonical hold declaration -- see this function's docstring above.
    graph.add_hold(n0, n1, 0, [0])

    return graph, n0, n1


def test_single_solve():
    """Both edge endpoints present for the one and only solve() call --
    node_active is all-True and wp_eff/assign_eff reduce to the raw
    decision-variable values everywhere (see problem.apply_anchor), so this
    exercises the ordinary (non-receding-horizon) path only."""
    graph, n0, n1 = build_graph()
    splines = [CubicConfigurationSpline([Block.R(DIM)])]

    solver = EvolutionaryWaypointSolver(
        graph, splines, objective="avg",
        wp_bounds=(-20.0, 20.0), pop_size=60, n_gen=150,
        outer_iters=3, inner_maxiter=30,
    )

    remaining = list(range(graph.structure.num_nodes))
    # Full configuration -- robot depot (away from the object) then the
    # object's own real starting position (nothing's been grasped yet, see
    # test_receding_horizon's docstring for why this must be the real state,
    # not zero-padded, once EvolutionaryWaypointSolver.solve()/warmup()
    # require x0 to always be the single full graph.total_dim vector).
    x0 = np.concatenate([np.array([0.0, 0.0]), OBJECT_START])
    ok = solver.solve(remaining, x0)
    assert ok, "EvolutionaryWaypointSolver.solve() returned False"

    waypoints = solver.view_waypoints()
    object_waypoints = solver.view_object_waypoints()

    obj_n0 = object_waypoints[n0]
    obj_n1 = object_waypoints[n1]
    agent_n0 = waypoints[n0, 0:DIM]
    agent_n1 = waypoints[n1, 0:DIM]

    print(f"solve time: {solver.get_last_solve_time() * 1e3:.1f} ms")
    print(f"object @ n0 (grasp): {obj_n0}  (target {OBJECT_START})")
    print(f"object @ n1 (place): {obj_n1}  (target {OBJECT_TARGET})")
    print(f"agent  @ n0 (grasp): {agent_n0}  (expected {OBJECT_START + GRASP_OFFSET})")
    print(f"agent  @ n1 (place): {agent_n1}  (expected {OBJECT_TARGET + GRASP_OFFSET})")

    edge_residual = (obj_n1 - obj_n0) - (agent_n1 - agent_n0)
    print(f"edge residual (should be ~0): {edge_residual}")

    assert not np.any(np.isnan(obj_n0)), "object waypoint at n0 was never written back (still NaN)"
    assert not np.any(np.isnan(obj_n1)), "object waypoint at n1 was never written back (still NaN)"

    assert np.allclose(obj_n0, OBJECT_START, atol=TOL), \
        f"object @ n0 = {obj_n0}, expected {OBJECT_START}"
    assert np.allclose(obj_n1, OBJECT_TARGET, atol=TOL), \
        f"object @ n1 = {obj_n1}, expected {OBJECT_TARGET}"
    assert np.allclose(agent_n0, OBJECT_START + GRASP_OFFSET, atol=TOL), \
        f"agent @ n0 = {agent_n0}, expected {OBJECT_START + GRASP_OFFSET} (rigid attachment)"
    assert np.allclose(edge_residual, 0.0, atol=TOL), \
        f"edge residual {edge_residual} not close to 0 (object not carried rigidly)"

    print("\nAll checks passed.")


def test_receding_horizon():
    """Regression test for the bug this solver's anchor mechanism fixes:
    an edge constraint used to silently vanish once its "u" endpoint left
    remaining_vertices (spec.py used to rebuild its structure --
    including which edges compiled at all -- fresh from remaining_vertices
    every solve()). Here n0 is dropped from remaining_vertices on a second
    solve() call, mirroring what GraphOfConstraintsMPC's pass_node()/
    _solve_for_timing() do once n0's phis are verified satisfied at runtime.

    n1's own node constraints (object_q(n1) == OBJECT_TARGET, and a loose
    agent_q(0) >= -20 bound purely to establish a routing instance -- see
    build_graph()'s docstring) do NOT by themselves determine the robot's
    n1 position -- only the n0->n1 edge constraint does, transitively
    through n0's value. So checking agent_n1 against OBJECT_TARGET +
    GRASP_OFFSET after n0 has left remaining_vertices is exactly the
    assertion that distinguishes "edge constraint still anchored at n0's
    frozen value" (fixed) from "edge constraint silently dropped" (bug):
    under the bug this would be an unconstrained free variable instead.

    Every x0 passed to solve()/warmup() must be the FULL configuration
    (agent AND object), never agent-only: spec.py's _resolve_holds
    registers a hold's rigid-carry relation mode="live" unconditionally (see
    its docstring -- a hold's u-side should track the real current state,
    not the frozen plan, once u has passed), and problem.apply_anchor's
    wp_eff_live substitution reads x0 as ONE global joint state for every
    passed row, object columns included -- a short, agent-only x0 would read
    back as "object at the origin" instead of "object still at the robot's
    grasp offset", corrupting the very relation being tested here (mpc.py's
    EvolutionaryWaypointSolver.solve()/warmup() assert x0.size ==
    graph.total_dim up front specifically to catch this). So x0_after_grasp
    below reports the object's real physical position too (agent position
    minus GRASP_OFFSET, since the robot is still holding it) -- as does the
    FIRST solve()'s own x0 above, for the same reason.
    """
    graph, n0, n1 = build_graph()
    splines = [CubicConfigurationSpline([Block.R(DIM)])]

    solver = EvolutionaryWaypointSolver(
        graph, splines, objective="avg",
        wp_bounds=(-20.0, 20.0), pop_size=60, n_gen=150,
        outer_iters=3, inner_maxiter=30,
    )

    # Full configuration -- robot depot then the object's own real starting
    # position (nothing's been grasped yet); see this function's docstring
    # for why x0_after_grasp below must likewise be the real full state.
    x0 = np.concatenate([np.array([0.0, 0.0]), OBJECT_START])
    ok = solver.solve([n0, n1], x0)
    assert ok, "first solve() (both nodes remaining) returned False"

    problem_after_first, ga_fn_after_first = solver._problem, solver._step_fn
    waypoints_n0_first = solver.view_waypoints()[n0].copy()
    object_waypoints_n0_first = solver.view_object_waypoints()[n0].copy()
    agent_n0_first = waypoints_n0_first[0:DIM]

    # n0 has been "passed": the robot is now at its committed n0 position,
    # still holding the object at GRASP_OFFSET -- see this function's
    # docstring for why the object's real position must be included here.
    x0_after_grasp = np.concatenate([agent_n0_first, agent_n0_first - GRASP_OFFSET])
    ok = solver.solve([n1], x0_after_grasp)
    assert ok, "second solve() (n0 passed, only n1 remaining) returned False"

    assert solver._problem is problem_after_first, \
        "structure was rebuilt on the second solve() -- should be reused unconditionally " \
        "once remaining_vertices no longer determines problem shape"
    assert solver._step_fn is ga_fn_after_first, \
        "GA was recompiled on the second solve() -- should be reused unconditionally"

    waypoints = solver.view_waypoints()
    object_waypoints = solver.view_object_waypoints()

    np.testing.assert_array_equal(
        waypoints[n0], waypoints_n0_first,
        err_msg="n0's frozen waypoint was overwritten by a solve() call where n0 is no "
                "longer in remaining_vertices")
    np.testing.assert_array_equal(
        object_waypoints[n0], object_waypoints_n0_first,
        err_msg="n0's frozen object waypoint was overwritten by a solve() call where n0 "
                "is no longer in remaining_vertices")

    agent_n1 = waypoints[n1, 0:DIM]
    obj_n1 = object_waypoints[n1]

    print(f"\n[receding-horizon] solve time: {solver.get_last_solve_time() * 1e3:.1f} ms")
    print(f"[receding-horizon] object @ n1 (place): {obj_n1}  (target {OBJECT_TARGET})")
    print(f"[receding-horizon] agent  @ n1 (place): {agent_n1}  (expected {OBJECT_TARGET + GRASP_OFFSET})")

    edge_residual = (obj_n1 - object_waypoints_n0_first) - (agent_n1 - agent_n0_first)
    print(f"[receding-horizon] edge residual against frozen n0 (should be ~0): {edge_residual}")

    assert np.allclose(obj_n1, OBJECT_TARGET, atol=TOL), \
        f"object @ n1 = {obj_n1}, expected {OBJECT_TARGET}"
    assert np.allclose(agent_n1, OBJECT_TARGET + GRASP_OFFSET, atol=TOL), \
        f"agent @ n1 = {agent_n1}, expected {OBJECT_TARGET + GRASP_OFFSET} -- the n0->n1 " \
        "edge constraint, anchored at n0's frozen value, did not survive n0 leaving " \
        "remaining_vertices"
    assert np.allclose(edge_residual, 0.0, atol=TOL), \
        f"edge residual {edge_residual} not close to 0 against frozen n0 values"

    print("\n[receding-horizon] All checks passed.")


def main():
    test_single_solve()
    test_receding_horizon()


if __name__ == "__main__":
    main()

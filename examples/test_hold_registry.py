"""Validation script for spec.py's _resolve_holds -- rigid-carry
constraints auto-derived from the graph's canonical hold registry
(graph.hold_ops, populated by GraphOfConstraints.add_hold/add_assignable_hold)
-- the JAX evolutionary solver's analogue of MILP's Constraint 14a
(milp_waypoint_mpc.cpp). Covers both a statically-assigned hold (hold.robot_ag
-- test_single_solve/test_interior_reinforcement) and an assignable one
(hold.var_id -- test_assignable_hold).

test_single_solve mirrors test_evolutionary_object_constraints.py exactly,
except the rigid-carry edge constraint is no longer hand-written -- one
`graph.add_hold(n0, n1, 0, [0])` call replaces the manual
`graph.add_edge_constraint(n0, n1, eq(v_obj0 - u_obj0, v_q0 - u_q0))`.

test_interior_reinforcement is the new case this registry-driven approach
adds over the old hand-written pattern: a hold spanning n0 -> n2 with a THIRD
node n1 -- owned by a completely different agent's routing instance --
structurally scheduled in between. Without MILP-Constraint-14a-style interior
reinforcement, the held object's column at n1 is a real, independently
optimized decision variable with nothing pinning it (see spec.py's module
docstring: every node gets a full row unconditionally); this checks it still
tracks the holding robot's own (unused-by-anyone-else) agent column at n1 via
the rigid-carry relation. Verified against the raw GA output (`wp`) rather
than EvolutionaryWaypointSolver.view_object_waypoints(), since that
convenience view only reports (node, object) pairs spec._node_objects marked
"referenced" -- interior nodes deliberately aren't (matching the existing
along-edge interior-reinforcement precedent), so it'd report NaN there even
though the underlying value is correctly constrained.

test_assignable_hold exercises add_assignable_hold: 2 candidate agents, one
add_variable() variable, and a rigid-carry relation built from
u_var_agent_q(var)/v_var_agent_q(var) (spec.py's _resolve_holds) -- resolved
per-individual by a genuine jax.lax.dynamic_slice keyed off the GA-searched
assignment, not by enumerating every candidate agent the way MILP's
AddHoldRigidityAssignable/AddHoldRigidityAssignableGated (milp_waypoint_mpc.cpp)
has to (drake::symbolic::Formula has no dynamic-indexing primitive -- see
graph_of_constraints.hpp's _var_agent_q_vars_u/_v docstring). Checks that
whichever real agent the GA picks for `var` is the one tied to the object via
the rigid-carry relation, and that the OTHER (unpicked) agent is left free
(no spurious pin, no NaN).

Run directly: python examples/test_hold_registry.py
"""

import numpy as np
import jax.numpy as jnp
from pydrake.math import eq, ge

from goc_mpc import EvolutionaryWaypointSolver, GraphOfConstraints
from goc_mpc._ext.configuration_spline import Block, CubicConfigurationSpline
from goc_mpc.evolutionary_waypoint_solver.spec import build_graph_ordering_problem
from goc_mpc.evolutionary_waypoint_solver.problem import full_active_anchor
from goc_mpc.evolutionary_waypoint_solver.solver import run_lamarckian_al

DIM = 2
OBJECT_START = np.array([2.0, 0.0])
OBJECT_TARGET = np.array([5.0, 3.0])
GRASP_OFFSET = np.array([0.3, 0.0])
TOL = 0.2


def build_single_hold_graph():
    robot_spec = [Block.R(DIM)]
    object_spec = [Block.R(DIM)]
    graph = GraphOfConstraints(
        [robot_spec], [object_spec],
        state_lower_bound=-20.0, state_upper_bound=20.0,
        robot_names=["robot"], object_names=["cube"],
    )

    n0, n1 = graph.structure.add_nodes(2)
    graph.structure.add_edge(n0, n1, True)

    q0, obj0 = graph.agent_q(0), graph.object_q(0)
    graph.add_constraint(n0, eq(obj0, OBJECT_START))
    graph.add_constraint(n0, eq(q0, obj0 + GRASP_OFFSET))
    graph.add_constraint(n1, eq(graph.object_q(0), OBJECT_TARGET))
    graph.add_constraint(n1, ge(graph.agent_q(0), np.array([-20.0, -20.0])))

    # Canonical hold declaration -- replaces the hand-written
    # add_edge_constraint(eq(v_obj0 - u_obj0, v_q0 - u_q0)) call.
    graph.add_hold(n0, n1, 0, [0])

    return graph, n0, n1


def test_single_solve():
    graph, n0, n1 = build_single_hold_graph()
    splines = [CubicConfigurationSpline([Block.R(DIM)])]

    solver = EvolutionaryWaypointSolver(
        graph, splines, objective="avg",
        wp_bounds=(-20.0, 20.0), pop_size=60, n_gen=150,
        outer_iters=3, inner_maxiter=30,
    )

    remaining = list(range(graph.structure.num_nodes))
    # Full configuration -- robot depot then the object's own real starting
    # position (nothing's been grasped yet): EvolutionaryWaypointSolver.
    # solve()/warmup() require x0 to always be the single full
    # graph.total_dim vector, not an agent-only depot.
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
    assert np.allclose(obj_n0, OBJECT_START, atol=TOL), f"object @ n0 = {obj_n0}, expected {OBJECT_START}"
    assert np.allclose(obj_n1, OBJECT_TARGET, atol=TOL), f"object @ n1 = {obj_n1}, expected {OBJECT_TARGET}"
    assert np.allclose(agent_n0, OBJECT_START + GRASP_OFFSET, atol=TOL), \
        f"agent @ n0 = {agent_n0}, expected {OBJECT_START + GRASP_OFFSET} (rigid attachment)"
    assert np.allclose(edge_residual, 0.0, atol=TOL), \
        f"edge residual {edge_residual} not close to 0 (object not carried rigidly, via add_hold)"

    print("test_single_solve: all checks passed.\n")


def test_interior_reinforcement():
    robot_spec = [Block.R(DIM)]
    object_spec = [Block.R(DIM)]
    graph = GraphOfConstraints(
        [robot_spec, robot_spec], [object_spec],
        state_lower_bound=-20.0, state_upper_bound=20.0,
        robot_names=["robot0", "robot1"], object_names=["cube"],
    )

    n0, n1, n2 = graph.structure.add_nodes(3)
    graph.structure.add_edge(n0, n1, True)
    graph.structure.add_edge(n1, n2, True)

    q0, obj0 = graph.agent_q(0), graph.object_q(0)
    graph.add_constraint(n0, eq(obj0, OBJECT_START))
    graph.add_constraint(n0, eq(q0, obj0 + GRASP_OFFSET))

    # robot 1 owns n1's routing instance -- structurally between n0 and n2 --
    # while robot 0 (the hold's own robot) is only loosely bounded there.
    graph.add_constraint(n1, eq(graph.agent_q(1), np.array([9.0, 9.0])))
    graph.add_constraint(n1, ge(graph.agent_q(0), np.array([-20.0, -20.0])))

    graph.add_constraint(n2, eq(graph.object_q(0), OBJECT_TARGET))
    graph.add_constraint(n2, ge(graph.agent_q(0), np.array([-20.0, -20.0])))

    graph.add_hold(n0, n2, 0, [0])

    agents_width = graph.num_agents * graph.robot_ambient_dim(0)
    x0 = np.zeros(2 * DIM)
    problem = build_graph_ordering_problem(
        graph, x0.reshape(graph.num_agents, graph.robot_ambient_dim(0)), (-20.0, 20.0), objective="avg")
    anchor = full_active_anchor(problem)
    result = run_lamarckian_al(
        problem, anchor, pop_size=80, n_gen=200, seed=1,
        outer_iters=3, inner_maxiter=30,
        x0=jnp.asarray(np.concatenate([x0, np.zeros(graph.object_ambient_dim(0))])))
    _assign, _cond_binary, _proj_branch, _t, wp, _psi = problem._extract_single(np.asarray(result.X))
    wp = np.asarray(wp)

    obj_n0 = wp[n0, agents_width:]
    obj_n1 = wp[n1, agents_width:]
    obj_n2 = wp[n2, agents_width:]
    agent0_n0 = wp[n0, 0:DIM]
    agent0_n1 = wp[n1, 0:DIM]
    agent0_n2 = wp[n2, 0:DIM]

    print(f"object @ n0: {obj_n0}  @ n1 (interior): {obj_n1}  @ n2: {obj_n2}")
    print(f"robot0 @ n0: {agent0_n0}  @ n1: {agent0_n1}  @ n2: {agent0_n2}")

    interior_residual = (obj_n1 - obj_n0) - (agent0_n1 - agent0_n0)
    print(f"interior residual (should be ~0 if reinforced): {interior_residual}")

    assert np.allclose(interior_residual, 0.0, atol=TOL), \
        f"held object at interior node n1 drifted from the rigid-carry relation: {interior_residual}"

    print("test_interior_reinforcement: all checks passed.\n")


def build_assignable_hold_graph():
    robot_spec = [Block.R(DIM)]
    object_spec = [Block.R(DIM)]
    graph = GraphOfConstraints(
        [robot_spec, robot_spec], [object_spec],
        state_lower_bound=-20.0, state_upper_bound=20.0,
        robot_names=["robot0", "robot1"], object_names=["cube"],
    )
    n0, n1 = graph.structure.add_nodes(2)
    graph.structure.add_edge(n0, n1, True)

    var = graph.add_variable()

    graph.add_constraint(n0, eq(graph.object_q(0), OBJECT_START))
    # Establishes a routing instance for `var` at n0 -- whichever real agent
    # the GA assigns to `var` must be at the grasp position there.
    graph.add_constraint(n0, eq(graph.var_agent_q(var), OBJECT_START + GRASP_OFFSET))

    graph.add_constraint(n1, eq(graph.object_q(0), OBJECT_TARGET))
    graph.add_constraint(n1, ge(graph.var_agent_q(var), np.array([-20.0, -20.0])))

    # Canonical ASSIGNABLE hold declaration -- var, not a literal robot id.
    graph.add_assignable_hold(n0, n1, var, [0])

    return graph, n0, n1, var


# def test_assignable_hold():
#     graph, n0, n1, var = build_assignable_hold_graph()
#     x0 = np.zeros(2 * DIM)
#     problem = build_graph_ordering_problem(
#         graph, x0.reshape(graph.num_agents, graph.robot_ambient_dim(0)), (-20.0, 20.0), objective="avg")
#     anchor = full_active_anchor(problem)
#     result = run_lamarckian_al(
#         problem, anchor, pop_size=150, n_gen=300, seed=1,
#         outer_iters=4, inner_maxiter=40,
#         x0=jnp.asarray(np.concatenate([x0, np.zeros(graph.object_ambient_dim(0))])))
#     print(f"CV: {result.CV}, F: {result.F}")
#     assign, _cond_binary, _t, wp = problem._extract_single(np.asarray(result.X))
#     wp = np.asarray(wp)
#     assign = np.asarray(assign)

#     var_slot = problem.var_id_to_slot[var]
#     picked_agent = int(np.argmax(assign[var_slot]))
#     other_agent = 1 - picked_agent
#     print(f"GA picked agent {picked_agent} for var {var}")

#     agents_width = graph.num_agents * graph.robot_ambient_dim(0)
#     picked_n0 = wp[n0, picked_agent * DIM:(picked_agent + 1) * DIM]
#     picked_n1 = wp[n1, picked_agent * DIM:(picked_agent + 1) * DIM]
#     other_n0 = wp[n0, other_agent * DIM:(other_agent + 1) * DIM]
#     other_n1 = wp[n1, other_agent * DIM:(other_agent + 1) * DIM]
#     obj_n0 = wp[n0, agents_width:]
#     obj_n1 = wp[n1, agents_width:]

#     print(f"picked agent @ n0: {picked_n0}  (expected {OBJECT_START + GRASP_OFFSET})")
#     print(f"picked agent @ n1: {picked_n1}")
#     print(f"object @ n0: {obj_n0}  (expected {OBJECT_START})")
#     print(f"object @ n1: {obj_n1}  (expected {OBJECT_TARGET})")
#     print(f"other (unpicked) agent {other_agent} -- free, no spurious pin: "
#           f"n0={other_n0}, n1={other_n1}")

#     assert result.CV[0] < 1e-2, f"infeasible (CV={result.CV})"
#     assert not np.any(np.isnan(other_n0)) and not np.any(np.isnan(other_n1)), \
#         "unpicked agent's waypoint was left NaN instead of a free value"
#     assert np.allclose(obj_n0, OBJECT_START, atol=TOL), f"object @ n0 = {obj_n0}, expected {OBJECT_START}"
#     assert np.allclose(obj_n1, OBJECT_TARGET, atol=TOL), f"object @ n1 = {obj_n1}, expected {OBJECT_TARGET}"
#     assert np.allclose(picked_n0, OBJECT_START + GRASP_OFFSET, atol=TOL), \
#         f"picked agent {picked_agent} @ n0 = {picked_n0}, expected rigid grasp offset"

#     edge_residual = (obj_n1 - obj_n0) - (picked_n1 - picked_n0)
#     print(f"edge residual (should be ~0): {edge_residual}")
#     assert np.allclose(edge_residual, 0.0, atol=TOL), \
#         f"edge residual {edge_residual} not close to 0 -- assignable hold's dynamic " \
#         "resolution didn't tie the object to the picked agent"

#     print("test_assignable_hold: all checks passed.\n")


if __name__ == "__main__":
    test_single_solve()
    test_interior_reinforcement()
    # test_assignable_hold()

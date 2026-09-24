"""Gated (schedule-conditional) auto-derived projections -- spec.py's
_resolve_stationary_objects (gated depot / edge stationary pins) and
_resolve_holds (rigid-carry pin), applied via problem.apply_projections'
`gate_fn` blend.

Run directly: python examples/test_gated_projections.py
"""

import numpy as np
import jax.numpy as jnp
from pydrake.math import eq

from goc_mpc import GraphOfConstraints
from goc_mpc._ext.configuration_spline import Block
from goc_mpc.evolutionary_waypoint_solver.projection import ProjOperator
from goc_mpc.evolutionary_waypoint_solver.spec import build_graph_ordering_problem
from goc_mpc.evolutionary_waypoint_solver.solver import run_lamarckian_al
from goc_mpc.evolutionary_waypoint_solver.problem import full_active_anchor, apply_projections

DIM = 2
TOL = 1e-3


def _solve(graph, x0_full, n_agents, pop=60, n_gen=120):
    problem = build_graph_ordering_problem(
        graph, x0_full[: n_agents * DIM].reshape(n_agents, DIM), (-20.0, 20.0), objective="avg")
    anchor = full_active_anchor(problem)
    result = run_lamarckian_al(problem, anchor, pop_size=pop, n_gen=n_gen, seed=1,
                                outer_iters=2, inner_maxiter=20, x0=jnp.asarray(x0_full))
    assign, cond_binary, proj_branch, t, wp, psi = problem._extract_single(np.asarray(result.X))
    return problem, np.asarray(wp), assign


def test_static_hold_stationary_and_rigid_carry():
    """n0 -> n1 -> n2, static hold n0->n2. Grasp (n0) and place (n2) EE
    targets are `proj=` pins (analytic, like a real Pick/Place), so with the
    gated depot + rigid-carry projections every object column is determined
    by the projection chain -- no continuous search left. Expect:
      * obj@n0 == depot x0 (gated depot projection, gate ON -- n0 before the
        hold span),
      * agent@n0 == obj@n0 + grasp_off (grasp IK reads the pinned obj@n0),
      * obj@n2 == obj@n0 + (agent@n2 - agent@n0) (rigid-carry projection),
      * which, since agent@n2 is the place pin, lands obj@n2 at the place
        target."""
    obj_start = np.array([2.0, 0.0])
    grasp_off = np.array([0.3, 0.0])
    place_target = np.array([5.0, 3.0])

    graph = GraphOfConstraints(
        [[Block.R(DIM)]], [[Block.R(DIM)]], state_lower_bound=-20.0, state_upper_bound=20.0,
        robot_names=["r0"], object_names=["obj0"])
    n0, n1, n2 = graph.structure.add_nodes(3)
    graph.structure.add_edge(n0, n1, True)
    graph.structure.add_edge(n1, n2, True)

    graph.add_constraint(
        n0, eq(graph.agent_q(0), graph.object_q(0) + grasp_off),
        proj=ProjOperator(pins=graph.agent_q(0), reads=(graph.object_q(0),),
                          func=lambda o, psi, b: o + grasp_off))
    graph.add_constraint(
        n2, eq(graph.agent_q(0), place_target + grasp_off),
        proj=ProjOperator(pins=graph.agent_q(0), reads=(),
                          func=lambda psi, b: place_target + grasp_off))
    graph.add_hold(n0, n2, 0, [0])

    x0 = np.concatenate([obj_start + grasp_off, obj_start])  # agent depot, object depot
    problem, wp, _ = _solve(graph, x0, n_agents=1, pop=20, n_gen=15)

    aw = graph.num_agents * graph.robot_ambient_dim(0)
    obj_n0, obj_n2 = wp[n0, aw:], wp[n2, aw:]
    ee_n0, ee_n2 = wp[n0, :DIM], wp[n2, :DIM]
    print(f"[static] obj@n0 {obj_n0} (depot {obj_start})  obj@n2 {obj_n2} (place {place_target})")

    assert np.allclose(obj_n0, obj_start, atol=TOL), obj_n0
    assert np.allclose(ee_n0, obj_start + grasp_off, atol=TOL), ee_n0
    assert np.allclose(obj_n2, obj_n0 + (ee_n2 - ee_n0), atol=TOL), (
        f"rigid carry violated: obj@n2 {obj_n2} != obj@n0 + ee-delta {obj_n0 + (ee_n2 - ee_n0)}")
    assert np.allclose(obj_n2, place_target, atol=TOL), obj_n2
    print("[static] OK")


def test_assignable_hold_rigid_carry():
    """Same shape, 2 candidate agents, an ASSIGNABLE hold -- the rigid-carry
    projection must resolve the FK'd agent from the GA's assignment
    (owner_variable) and still determine the object column."""
    obj_start = np.array([1.0, 1.0])
    grasp_off = np.array([0.3, 0.0])
    place_target = np.array([4.0, -2.0])

    graph = GraphOfConstraints(
        [[Block.R(DIM)], [Block.R(DIM)]], [[Block.R(DIM)]],
        state_lower_bound=-20.0, state_upper_bound=20.0,
        robot_names=["r0", "r1"], object_names=["obj0"])
    n0, n1 = graph.structure.add_nodes(2)
    graph.structure.add_edge(n0, n1, True)
    var = graph.add_variable()
    graph.add_constraint(n0, eq(graph.var_agent_q(var), graph.object_q(0) + grasp_off))
    graph.add_constraint(n1, eq(graph.var_agent_q(var), place_target + grasp_off))
    graph.add_constraint(n1, eq(graph.object_q(0), place_target))  # place target -> obj@n1 residual
    graph.add_assignable_hold(n0, n1, var, [0])

    x0 = np.concatenate([obj_start + grasp_off, np.zeros(DIM), obj_start])
    problem, wp, assign = _solve(graph, x0, n_agents=2, pop=60, n_gen=150)

    aw = graph.num_agents * graph.robot_ambient_dim(0)
    obj_n0, obj_n1 = wp[n0, aw:], wp[n1, aw:]
    owner = int(np.argmax(assign[0]))
    ee_n0 = wp[n0, owner * DIM:owner * DIM + DIM]
    ee_n1 = wp[n1, owner * DIM:owner * DIM + DIM]
    print(f"[assignable] owner=agent{owner}  obj@n0 {obj_n0}  obj@n1 {obj_n1} (place {place_target})")

    assert np.allclose(obj_n0, obj_start, atol=TOL), obj_n0
    assert np.allclose(obj_n1, obj_n0 + (ee_n1 - ee_n0), atol=TOL), (
        f"assignable rigid carry violated: {obj_n1} != {obj_n0 + (ee_n1 - ee_n0)}")
    assert np.allclose(obj_n1, place_target, atol=TOL), obj_n1
    print("[assignable] OK")


def test_gated_projection_entries_present():
    """Structural check: a hold graph gets exactly the expected gated auto
    projections, and a hold-free object graph gets NONE (feature can't
    perturb a hold-free scene)."""
    graph = GraphOfConstraints(
        [[Block.R(DIM)]], [[Block.R(DIM)]], state_lower_bound=-9.0, state_upper_bound=9.0,
        robot_names=["r0"], object_names=["o0"])
    n0, n1 = graph.structure.add_nodes(2)
    graph.structure.add_edge(n0, n1, True)
    graph.add_constraint(n0, eq(graph.object_q(0), np.zeros(DIM)))
    problem = build_graph_ordering_problem(graph, np.zeros((1, DIM)), (-9.0, 9.0))
    assert len(problem.projections) == 0, [e.write_node for e in problem.projections]

    graph.add_hold(n0, n1, 0, [0])
    problem = build_graph_ordering_problem(graph, np.zeros((1, DIM)), (-9.0, 9.0))
    gated = [e for e in problem.projections if e.gate_fn is not None]
    # depot @ n0, rigid-carry @ n1 (the n0->n1 stationary edge yields to the
    # rigid-carry claim on n1's object columns).
    assert len(gated) == 2, [(e.write_node, list(e.pinned_cols)) for e in problem.projections]
    assert {e.write_node for e in gated} == {n0, n1}
    print("[entries] OK")


if __name__ == "__main__":
    test_gated_projection_entries_present()
    test_static_hold_stationary_and_rigid_carry()
    test_assignable_hold_rigid_carry()
    print("\nAll checks passed.")

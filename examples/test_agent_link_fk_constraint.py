"""Validation script for agent_link_pos/agent_link_rot -- constraints written
against a registered forward-kinematics function, resolved by the JAX
evolutionary solver by calling that function directly (never MILP, which has
no way to represent arbitrary FK in a drake::symbolic::Formula).

1 robot (a 2-link planar arm, dim=2: [theta1, theta2] -- deliberately NOT
representable in closed form by GraphOfConstraints' built-in RobotKind
dispatch, so this genuinely exercises arbitrary registered FK) + 1 object
(dim=3: a plain 3D position). One graph node:

  n0: object_q(0) == TARGET                              (pure node-object pin)
      agent_link_pos(0, "ee") == object_q(0)              (FK-based constraint --
        the arm's end-effector position, via the registered two_link_fk,
        must land on the object's position)

two_link_fk is registered once via graph.set_robot_fk(0, "ee", two_link_fk)
and is written in jax.numpy -- per set_robot_fk's doc comment, that single
registration is polymorphic: called from C++ (graph.link_pose, with a
concrete numpy array) it runs eagerly, and called directly in Python by the
evolutionary solver (spec.py's _make_link_pos_map/_make_link_rot_map) it
traces cleanly under jax.jit/vmap.

test_orientation_constraint() exercises agent_link_rot the same way, against
a target end-effector orientation (the arm tip's rotation about z, a function
of theta1+theta2) instead of a target position -- same registered fk_fn,
just its rotation half.

Also confirms MILPWaypointMPC raises a clear, actionable error (not an
opaque Drake failure) if pointed at a graph using either placeholder.

test_2d_workspace() repeats the position+orientation idea against a
genuinely 2D graph (GraphOfConstraints(..., workspace_dim=2)) -- the arm's FK
naturally outputs (x, y) and a 2x2 rotation with no z-padding trick needed,
and agent_link_pos/agent_link_rot/link_pose size themselves to workspace_dim
throughout (link_pose's built-in RobotKind fallback for kPointMass/kPosYaw
honors it too, though that path isn't exercised by this script since both
graphs here register a custom fk_fn).

Run directly: python examples/test_agent_link_fk_constraint.py
"""

import jax.numpy as jnp
import numpy as np
from pydrake.math import eq

from goc_mpc import EvolutionaryWaypointSolver, GraphOfConstraints, MILPWaypointMPC
from goc_mpc._ext.configuration_spline import Block, CubicConfigurationSpline

AGENT_DIM = 2     # [theta1, theta2]
OBJECT_DIM = 3    # plain 3D position
L1, L2 = 1.0, 1.0
TARGET = np.array([1.2, 0.6, 0.0])  # within the arm's reach (|TARGET| < L1 + L2)
TARGET_ANGLE = 0.9  # ee orientation about z (theta1 + theta2), radians
_c, _s = np.cos(TARGET_ANGLE), np.sin(TARGET_ANGLE)
TARGET_ROT = np.array([[_c, -_s, 0.0], [_s, _c, 0.0], [0.0, 0.0, 1.0]])
TOL = 0.1
WP_BOUNDS = (-3.5, 3.5)  # covers both joint angles (+-pi) and TARGET's magnitude


def two_link_fk(q):
    """(theta1, theta2) -> (ee position (3,), ee rotation (3,3)). The ee
    frame is rotated about z by theta1+theta2 (the arm tip's total heading),
    so agent_link_rot is a genuine function of q, not a constant. Written in
    jax.numpy so it works both as link_pose's concrete (C++/pybind) backend
    and, called directly in Python, as the evolutionary solver's JAX-traced
    backend -- see this module's docstring and set_robot_fk's doc comment."""
    theta1, theta2 = q[0], q[1]
    x = L1 * jnp.cos(theta1) + L2 * jnp.cos(theta1 + theta2)
    y = L1 * jnp.sin(theta1) + L2 * jnp.sin(theta1 + theta2)
    pos = jnp.array([x, y, 0.0])
    phi = theta1 + theta2
    c, s = jnp.cos(phi), jnp.sin(phi)
    rot = jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return pos, rot


def build_graph() -> GraphOfConstraints:
    robot_spec = [Block.R(AGENT_DIM)]
    object_spec = [Block.R(OBJECT_DIM)]
    graph = GraphOfConstraints(
        [robot_spec], [object_spec],
        state_lower_bound=-20.0, state_upper_bound=20.0,
        robot_names=["arm"], object_names=["target_point"],
    )
    graph.set_robot_fk(0, "ee", two_link_fk)

    (n0,) = graph.structure.add_nodes(1)

    graph.add_constraint(n0, eq(graph.object_q(0), TARGET))
    graph.add_constraint(n0, eq(graph.agent_link_pos(0, "ee"), graph.object_q(0)))

    return graph, n0


def test_evolutionary_solver_reaches_target():
    graph, n0 = build_graph()
    splines = [CubicConfigurationSpline([Block.R(AGENT_DIM)])]

    solver = EvolutionaryWaypointSolver(
        graph, splines, objective="avg",
        wp_bounds=WP_BOUNDS, pop_size=60, n_gen=150,
        outer_iters=3, inner_maxiter=30,
    )

    remaining = list(range(graph.structure.num_nodes))
    x0 = np.zeros(AGENT_DIM + OBJECT_DIM)
    ok = solver.solve(remaining, x0)
    assert ok, "EvolutionaryWaypointSolver.solve() returned False"

    waypoints = solver.view_waypoints()
    q_n0 = waypoints[n0, 0:AGENT_DIM]

    ee_pos_jax, _ = two_link_fk(jnp.asarray(q_n0))
    ee_pos_jax = np.asarray(ee_pos_jax)

    # Independent confirmation via the concrete (C++/pybind) path -- same
    # registered function, called through graph.link_pose instead of
    # directly in Python, to confirm the "one polymorphic fk_fn" premise.
    x_full = np.concatenate([q_n0, TARGET])
    ee_pos_concrete, _ = graph.link_pose(0, x_full, "ee")

    print(f"solve time: {solver.get_last_solve_time() * 1e3:.1f} ms")
    print(f"solved q (joint angles): {q_n0}")
    print(f"ee position (JAX path):      {ee_pos_jax}")
    print(f"ee position (concrete path): {ee_pos_concrete}")
    print(f"target: {TARGET}")

    assert np.allclose(ee_pos_jax, ee_pos_concrete, atol=1e-6), \
        "JAX-traced and concrete calls to the SAME registered fk_fn disagree"
    assert np.allclose(ee_pos_jax, TARGET, atol=TOL), \
        f"end-effector position {ee_pos_jax} not close to target {TARGET}"

    print("\nAll checks passed.")


def build_graph_rotation() -> GraphOfConstraints:
    """Orientation-only counterpart of build_graph() -- constrains
    agent_link_rot(0, "ee") instead of agent_link_pos, so the ee's total
    heading (theta1+theta2) must reach TARGET_ANGLE. Position is left free
    (no constraint on object_q here), avoiding the elbow-up/elbow-down
    ambiguity a combined position+orientation target would introduce for a
    2-DOF arm (position alone already fully determines the reachable
    headings, up to that ambiguity)."""
    robot_spec = [Block.R(AGENT_DIM)]
    object_spec = [Block.R(OBJECT_DIM)]
    graph = GraphOfConstraints(
        [robot_spec], [object_spec],
        state_lower_bound=-20.0, state_upper_bound=20.0,
        robot_names=["arm"], object_names=["target_point"],
    )
    graph.set_robot_fk(0, "ee", two_link_fk)

    (n0,) = graph.structure.add_nodes(1)

    graph.add_constraint(n0, eq(graph.agent_link_rot(0, "ee"), TARGET_ROT.flatten()))

    return graph, n0


def test_orientation_constraint():
    graph, n0 = build_graph_rotation()
    assert graph.agent_link_rot(0, "ee").shape == (9,), \
        "agent_link_rot should be 9-wide (3x3 flattened) for a workspace_dim=3 graph"

    splines = [CubicConfigurationSpline([Block.R(AGENT_DIM)])]

    solver = EvolutionaryWaypointSolver(
        graph, splines, objective="avg",
        wp_bounds=WP_BOUNDS, pop_size=60, n_gen=150,
        outer_iters=3, inner_maxiter=30,
    )

    remaining = list(range(graph.structure.num_nodes))
    x0 = np.zeros(AGENT_DIM + OBJECT_DIM)
    ok = solver.solve(remaining, x0)
    assert ok, "EvolutionaryWaypointSolver.solve() returned False (orientation)"

    waypoints = solver.view_waypoints()
    q_n0 = waypoints[n0, 0:AGENT_DIM]

    _pos_jax, ee_rot_jax = two_link_fk(jnp.asarray(q_n0))
    ee_rot_jax = np.asarray(ee_rot_jax)

    # Independent confirmation via the concrete (C++/pybind) path.
    x_full = np.concatenate([q_n0, np.zeros(OBJECT_DIM)])
    _pos_concrete, ee_rot_concrete = graph.link_pose(0, x_full, "ee")

    print(f"\n[rot] solve time: {solver.get_last_solve_time() * 1e3:.1f} ms")
    print(f"[rot] solved q (joint angles): {q_n0}")
    print(f"[rot] ee rotation (JAX path):\n{ee_rot_jax}")
    print(f"[rot] ee rotation (concrete path):\n{ee_rot_concrete}")
    print(f"[rot] target rotation:\n{TARGET_ROT}")

    assert np.allclose(ee_rot_jax, ee_rot_concrete, atol=1e-6), \
        "JAX-traced and concrete calls to the SAME registered fk_fn disagree (rotation)"
    assert np.allclose(ee_rot_jax, TARGET_ROT, atol=TOL), \
        f"end-effector rotation {ee_rot_jax} not close to target {TARGET_ROT}"

    print("\n[rot] All checks passed.")


TARGET_2D = np.array([1.2, 0.6])  # within the arm's reach, no z-padding needed
TARGET_ANGLE_2D = 0.9
_c2, _s2 = np.cos(TARGET_ANGLE_2D), np.sin(TARGET_ANGLE_2D)
TARGET_ROT_2D = np.array([[_c2, -_s2], [_s2, _c2]])


def two_link_fk_2d(q):
    """2D analogue of two_link_fk -- (theta1, theta2) -> (ee position (2,),
    ee rotation (2,2), a rotation-by-(theta1+theta2) matrix, same heading
    convention as two_link_fk's 3D z-rotation). Same registration/tracing
    story as two_link_fk, just without the z=0 padding a workspace_dim=3
    graph would otherwise force."""
    theta1, theta2 = q[0], q[1]
    x = L1 * jnp.cos(theta1) + L2 * jnp.cos(theta1 + theta2)
    y = L1 * jnp.sin(theta1) + L2 * jnp.sin(theta1 + theta2)
    pos = jnp.array([x, y])
    phi = theta1 + theta2
    c, s = jnp.cos(phi), jnp.sin(phi)
    rot = jnp.array([[c, -s], [s, c]])
    return pos, rot


def build_graph_2d() -> GraphOfConstraints:
    robot_spec = [Block.R(AGENT_DIM)]
    object_spec = [Block.R(2)]
    graph = GraphOfConstraints(
        [robot_spec], [object_spec],
        state_lower_bound=-20.0, state_upper_bound=20.0,
        robot_names=["arm"], object_names=["target_point"],
        workspace_dim=2,
    )
    graph.set_robot_fk(0, "ee", two_link_fk_2d)

    (n0,) = graph.structure.add_nodes(1)

    graph.add_constraint(n0, eq(graph.object_q(0), TARGET_2D))
    graph.add_constraint(n0, eq(graph.agent_link_pos(0, "ee"), graph.object_q(0)))

    return graph, n0


def test_2d_workspace():
    graph, n0 = build_graph_2d()
    assert graph.agent_link_pos(0, "ee").shape == (2,), \
        "agent_link_pos should be 2-wide for a workspace_dim=2 graph"
    assert graph.agent_link_rot(0, "ee").shape == (4,), \
        "agent_link_rot should be 4-wide (2x2 flattened) for a workspace_dim=2 graph"

    splines = [CubicConfigurationSpline([Block.R(AGENT_DIM)])]
    solver = EvolutionaryWaypointSolver(
        graph, splines, objective="avg",
        wp_bounds=WP_BOUNDS, pop_size=60, n_gen=150,
        outer_iters=3, inner_maxiter=30,
    )

    remaining = list(range(graph.structure.num_nodes))
    x0 = np.zeros(AGENT_DIM + 2)
    ok = solver.solve(remaining, x0)
    assert ok, "EvolutionaryWaypointSolver.solve() returned False (2D workspace)"

    waypoints = solver.view_waypoints()
    q_n0 = waypoints[n0, 0:AGENT_DIM]

    ee_pos_jax, _ = two_link_fk_2d(jnp.asarray(q_n0))
    ee_pos_jax = np.asarray(ee_pos_jax)

    x_full = np.concatenate([q_n0, TARGET_2D])
    ee_pos_concrete, R_concrete = graph.link_pose(0, x_full, "ee")

    print(f"\n[2D] solve time: {solver.get_last_solve_time() * 1e3:.1f} ms")
    print(f"[2D] solved q (joint angles): {q_n0}")
    print(f"[2D] ee position (JAX path):      {ee_pos_jax}")
    print(f"[2D] ee position (concrete path): {ee_pos_concrete}")
    print(f"[2D] target: {TARGET_2D}")

    assert ee_pos_concrete.shape == (2,) and R_concrete.shape == (2, 2), \
        f"link_pose should return a 2D pose for a workspace_dim=2 graph, got shapes " \
        f"{ee_pos_concrete.shape}/{R_concrete.shape}"
    assert np.allclose(ee_pos_jax, ee_pos_concrete, atol=1e-6), \
        "JAX-traced and concrete calls to the SAME registered fk_fn disagree (2D)"
    assert np.allclose(ee_pos_jax, TARGET_2D, atol=TOL), \
        f"end-effector position {ee_pos_jax} not close to target {TARGET_2D} (2D)"

    # Rotation cross-check (no orientation constraint in this graph -- just
    # confirms the JAX and concrete paths agree on agent_link_rot's 2D
    # flattening, same premise as the 3D orientation test above).
    _pos_jax_r, ee_rot_jax = two_link_fk_2d(jnp.asarray(q_n0))
    ee_rot_jax = np.asarray(ee_rot_jax)
    assert np.allclose(ee_rot_jax, R_concrete, atol=1e-6), \
        "JAX-traced and concrete calls to the SAME registered fk_fn disagree (2D rotation)"

    print("\n[2D] All checks passed.")


def test_milp_raises_clear_error():
    graph, _n0 = build_graph()
    splines = [CubicConfigurationSpline([Block.R(AGENT_DIM)])]
    milp = MILPWaypointMPC(graph, splines)

    remaining = list(range(graph.structure.num_nodes))
    x0 = np.zeros(AGENT_DIM + OBJECT_DIM)
    try:
        milp.solve(remaining, x0)
    except RuntimeError as e:
        assert "agent_link_pos" in str(e), f"unexpected error message: {e}"
        print(f"\nMILPWaypointMPC raised as expected: {e}")
        return
    raise AssertionError("MILPWaypointMPC.solve() should have raised for an agent_link_pos constraint")


def test_milp_raises_clear_error_rotation():
    graph, _n0 = build_graph_rotation()
    splines = [CubicConfigurationSpline([Block.R(AGENT_DIM)])]
    milp = MILPWaypointMPC(graph, splines)

    remaining = list(range(graph.structure.num_nodes))
    x0 = np.zeros(AGENT_DIM + OBJECT_DIM)
    try:
        milp.solve(remaining, x0)
    except RuntimeError as e:
        assert "agent_link_rot" in str(e), f"unexpected error message: {e}"
        print(f"\nMILPWaypointMPC raised as expected: {e}")
        return
    raise AssertionError("MILPWaypointMPC.solve() should have raised for an agent_link_rot constraint")


def main():
    test_evolutionary_solver_reaches_target()
    test_orientation_constraint()
    test_2d_workspace()
    test_milp_raises_clear_error()
    test_milp_raises_clear_error_rotation()


if __name__ == "__main__":
    main()

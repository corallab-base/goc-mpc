"""Zero-config forward kinematics for the JAX evolutionary solver, mirroring
utils.hpp's PoseFromRow/RobotKind dispatch (the same closed-form math
MILP's C++ constraint builders and GraphOfConstraints::link_pose use) so a
robot whose configuration space is one of the built-in kinds -- point-mass,
pos+yaw, pos+quaternion, pos+rotation-matrix -- gets a working
agent_id -> (position, rotation) function without the caller ever having to
call graph.set_robot_fk.

RobotKind itself isn't exposed to Python (graph_of_constraints.hpp keeps
`_robot_kinds` private, inferred once at construction time from each agent's
spline spec), but that inference is a pure function of the spec -- the same
`robot_specs` list Python already has via `graph._robot_specs` -- so
`_default_kind` below re-derives it directly instead of needing a new C++
binding. `kArticulated` (no closed-form pose) has no default: joint-space
configuration bears no fixed relation to world-frame link pose, so a caller
must register a real forward-kinematics function via graph.set_robot_fk for
that agent.

`resolve_link_fk` is the actual entry point other modules should use: it
checks graph.robot_fk_registry first (a caller-registered fk_fn always wins,
same as GraphOfConstraints::link_pose) and only falls back to the built-in
dispatch here if nothing is registered -- and, matching link_pose's own
fallback exactly, `link_name` is ignored once it falls back, since none of
the built-in kinds model more than one link.
"""

import jax.numpy as jnp

from .._ext.configuration_spline import BlockType


def _default_kind(spec):
    """Ports graph_of_constraints.cpp's RobotKind inference (has_quat/
    has_rot_mat/has_torus/all_eucl over a Block spec) to Python -- see this
    module's docstring for why this is re-derived here rather than read off
    a C++ method. Returns one of "pos_quat"/"pos_rot_mat"/"pos_yaw"/
    "point_mass"/"articulated", the same precedence order the C++ side uses.
    """
    has_quat = any(b.type == BlockType.SO3Quat for b in spec)
    has_rot_mat = any(b.type == BlockType.SO3Mat for b in spec)
    has_torus = any(b.type == BlockType.Torus for b in spec)
    all_eucl = all(b.type == BlockType.R for b in spec)

    if has_quat:
        return "pos_quat"
    if has_rot_mat:
        return "pos_rot_mat"
    if has_torus:
        return "pos_yaw"
    if all_eucl:
        return "point_mass"
    return "articulated"


def _rot_from_quat(w, x, y, z):
    """Branchless quaternion -> rotation matrix, assuming a unit quaternion
    -- same formula as RotFromQuatBranchless (utils.hpp)."""
    return jnp.array([
        [1 - 2 * (y * y + z * z),     2 * (x * y - w * z),     2 * (x * z + w * y)],
        [    2 * (x * y + w * z), 1 - 2 * (x * x + z * z),     2 * (y * z - w * x)],
        [    2 * (x * z - w * y),     2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _pose_point_mass(q, workspace_dim):
    """PoseFromRow_PointMass: leading workspace_dim-wide position, identity
    rotation (a point mass has no orientation)."""
    return q[:workspace_dim], jnp.eye(workspace_dim)


def _pose_pos_yaw(q, workspace_dim):
    """PoseFromRow_PosYaw: [x, y, θ] (workspace_dim==2) or [x, y, z, θ]
    (workspace_dim==3) -- yaw about the plane/world-Z axis, immediately
    after the position block."""
    pos = q[:workspace_dim]
    theta = q[workspace_dim]
    c, s = jnp.cos(theta), jnp.sin(theta)
    if workspace_dim == 2:
        rot = jnp.array([[c, -s], [s, c]])
    else:
        rot = jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return pos, rot


def _pose_pos_quat(q):
    """PoseFromRow_PosQuat: [x, y, z, w, qx, qy, qz] -- 3D position only,
    same restriction as the C++ side (quaternion rotation has no 2D
    reduction)."""
    return q[:3], _rot_from_quat(q[3], q[4], q[5], q[6])


def _pose_pos_rot_mat(q):
    """PoseFromRow_PosRotMatrix: [x, y, z, R00..R22] (row-major) -- 3D
    position only, same restriction as the C++ side."""
    return q[:3], q[3:12].reshape(3, 3)


def default_fk_for(graph, agent_id):
    """Returns a fk_fn(q) -> (position, rotation) for `agent_id`'s built-in
    RobotKind (inferred from graph._robot_specs[agent_id] via
    _default_kind), matching utils.hpp's PoseFromRow dispatch -- `q` is
    that agent's own dim-wide row (already sliced, offset 0), the same
    convention a graph.set_robot_fk-registered fk_fn uses. Raises for
    "articulated" -- there is no closed-form default, a real fk_fn must be
    registered via graph.set_robot_fk for that agent."""
    kind = _default_kind(graph._robot_specs[agent_id])
    workspace_dim = graph.workspace_dim

    if kind == "point_mass":
        return lambda q: _pose_point_mass(q, workspace_dim)
    if kind == "pos_yaw":
        return lambda q: _pose_pos_yaw(q, workspace_dim)
    if kind == "pos_quat":
        if workspace_dim != 3:
            raise ValueError(
                f"default_fk_for: agent {agent_id} is pos_quat, which requires "
                f"workspace_dim == 3 (quaternion rotation has no 2D reduction), "
                f"got workspace_dim={workspace_dim}")
        return _pose_pos_quat
    if kind == "pos_rot_mat":
        if workspace_dim != 3:
            raise ValueError(
                f"default_fk_for: agent {agent_id} is pos_rot_mat, which requires "
                f"workspace_dim == 3 (3x3 rotation matrix has no 2D reduction), "
                f"got workspace_dim={workspace_dim}")
        return _pose_pos_rot_mat
    raise ValueError(
        f"default_fk_for: agent {agent_id} has no closed-form forward "
        "kinematics (articulated configuration space) -- register one via "
        "graph.set_robot_fk(agent_id, link_name, fk_fn) first.")


def resolve_link_fk(graph, agent_id, link_name="ee"):
    """The fk_fn `_resolve_holds`/agent_link_pos-style consumers should
    call: a registered override (graph.robot_fk_registry) for (agent_id,
    link_name) wins if present; otherwise falls back to default_fk_for,
    exactly matching GraphOfConstraints::link_pose's own fallback --
    `link_name` is ignored once it falls back, since none of the built-in
    RobotKinds model more than one link."""
    fk_fn = graph.robot_fk_registry.get((agent_id, link_name))
    if fk_fn is not None:
        return fk_fn
    return default_fk_for(graph, agent_id)

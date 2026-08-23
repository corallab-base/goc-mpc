"""
Interactive obstacle demo for the short-horizon MPC (stage 2 of 3).

A single point-mass agent has a FIXED nominal reference path (straight line,
start -> goal, drawn in gray). A draggable obstacle (sphere or box, pick via
the GUI) sits in the browser (viser transform-controls gizmo). Every time you
drag it -- or change a GUI control -- GraphShortPathMPC.solve() is re-run
from scratch against the (unchanged) nominal reference, and the resulting
short-horizon trajectory (blue line) is redrawn.

A second agent can be toggled on ("Enable second agent"). Its own nominal
path is a fixed-length segment attached to a SECOND, draggable-AND-
rotatable gizmo -- dragging moves the whole path, rotating re-orients it,
so you can steer it to cross agent 1's fixed path. It defaults to crossing
agent 1's path immediately (positioned at agent 1's own midpoint, identity
rotation) so the interesting case is visible the moment you tick the box,
before touching anything. Both agents' optimized short-horizon paths are
drawn (agent 1 blue, agent 2 orange), and the "Agent radius" slider controls
their inter-agent avoidance margin.

GraphShortPathMPC (see graph_short_path_mpc.hpp) is a Riemannian trust-region
SQP solver over qpOASES. Every obstacle constraint is a slack-relaxed
exact-penalty row, so the QP subproblem is feasible by construction at every
outer iteration, plus a closed-form safety-projection final pass gives a
hard clearance guarantee regardless of SQP convergence.

This deliberately calls the solver class directly (not the full
`GraphOfConstraintsMPC.step()` closed loop) and always resolves from the same
fixed start state(s) -- that's what "a nominal spline that's fixed" means
here, and it isolates the thing being demonstrated (does the short path
react correctly to a nearby obstacle/agent?) from a separate, known
closed-loop issue: repeatedly feeding a solved trajectory's own state back
as the next cycle's x0 can make the agent drift/stall near an obstacle
rather than pass it, which needs further investigation and isn't what this
script exercises.

Run:
    python examples/short_horizon_collision_avoidance.py
Then open the printed URL, drag the red obstacle (and, if enabled, the
second agent's path gizmo), and play with the GUI controls on the left.
"""

import time

import numpy as np
import viser

from goc_mpc import GraphOfConstraints, GraphShortPathMPC, ObstacleSet
from goc_mpc._ext.configuration_spline import CubicConfigurationSpline, Block

DIM = 3
START = np.array([0.0, 0.0, 1.0])
GOAL = np.array([0.0, 1.0, 1.0])

# Agent 2's nominal path is a fixed-length segment carried by its own
# transform-controls gizmo: local (-L/2,0,0) -> (L/2,0,0), mapped to world
# coordinates by the gizmo's own position+rotation. Same length as agent 1's
# path. Default gizmo pose (identity rotation, centered on agent 1's own
# midpoint) puts agent 2's default path through (-0.5,0.5,1)->(0.5,0.5,1),
# which crosses agent 1's path exactly at its midpoint -- deliberately, so
# ticking the checkbox shows a real conflict immediately.
AGENT2_PATH_LENGTH = 1.0
AGENT2_INITIAL_POSITION = np.array([0.0, 0.5, 1.0])
AGENT2_INITIAL_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])

SPHERE_RADIUS = 0.15
BOX_HALF_EXTENTS = np.array([0.25, 0.25, 0.25])
OBSTACLE_MARGIN = 0.05
OBSTACLE_INITIAL_CENTER = np.array([0.6, 0.5, 1.0])  # off to the side initially

NUM_STEPS = 10
TIME_PER_STEP = 0.1


def make_graph(num_agents: int) -> GraphOfConstraints:
    graph = GraphOfConstraints(
        [[Block.R(DIM)]] * num_agents, [],
        state_lower_bound=-10.0, state_upper_bound=10.0,
        robot_names=[f"agent{i}" for i in range(num_agents)], object_names=[],
    )
    graph.structure.add_nodes(1)
    return graph


def make_linear_spline(start: np.ndarray, goal: np.ndarray) -> CubicConfigurationSpline:
    """A FIXED reference: a straight line from `start` to `goal`."""
    spline = CubicConfigurationSpline([Block.R(DIM)])
    spline.set_linear(True)
    spline.set(np.stack([start, goal]), np.zeros((2, DIM)), np.array([0.0, 1.0]))
    return spline


def quat_rotate(wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate 3-vector `v` by unit quaternion `wxyz=(w,x,y,z)` (viser's own
    convention). Standard quaternion-rotation formula, done by hand rather
    than pulling in a new dependency for one vector rotation."""
    w = wxyz[0]
    q = wxyz[1:]
    t = 2.0 * np.cross(q, v)
    return v + w * t + np.cross(q, t)


def main():
    server = viser.ViserServer()

    second_agent_checkbox = server.gui.add_checkbox("Enable second agent", initial_value=False)
    kind_dropdown = server.gui.add_dropdown("Obstacle kind", ("sphere", "box"), initial_value="box")
    weight_slider = server.gui.add_slider(
        "Penalty weight", min=0.1, max=2000.0, step=0.1, initial_value=5.0)
    # log10(acceleration_weight) -- the smoothness term has a large BUILT-IN
    # stiffness relative to tracking (its own coefficients scale as
    # ~1/time_per_step^2, squared again in the Hessian), so a *linear* slider
    # can't usefully span the range where tracking becomes competitive; a
    # slider over the exponent can. 0 (10^0=1.0) reproduces
    # GraphShortPathMPC's default byte-for-byte.
    accel_log_slider = server.gui.add_slider(
        "Acceleration weight (log10)", min=-7.0, max=1.0, step=0.1, initial_value=0.0)
    agent_radius_slider = server.gui.add_slider(
        "Agent radius (inter-agent avoidance)", min=0.0, max=0.5, step=0.01, initial_value=0.1)
    status_text = server.gui.add_text("Status", initial_value="")

    # Agent 1: fixed nominal path + endpoint markers (unchanged from before
    # the second agent existed).
    spline1 = make_linear_spline(START, GOAL)
    server.scene.add_spline_catmull_rom(
        "/nominal_path", points=np.stack([START, GOAL]),
        color=(150, 150, 150), line_width=2)
    server.scene.add_icosphere("/start_marker", radius=0.03, color=(40, 160, 40),
                                position=tuple(START))
    server.scene.add_icosphere("/goal_marker", radius=0.03, color=(40, 90, 220),
                                position=tuple(GOAL))
    path_vis1 = server.scene.add_spline_catmull_rom(
        "/short_horizon_path", points=np.stack([START, GOAL]),
        color=(30, 160, 220), line_width=4)

    # Agent 2: draggable + rotatable path gizmo, hidden until the checkbox
    # is ticked. Its nominal path/markers are recomputed from the gizmo's
    # current transform every resolve() call (see agent2_start_goal below),
    # unlike agent 1's, which is genuinely fixed for the process lifetime.
    agent2_gizmo = server.scene.add_transform_controls(
        "/agent2_gizmo", scale=0.3, position=tuple(AGENT2_INITIAL_POSITION),
        wxyz=tuple(AGENT2_INITIAL_WXYZ))
    agent2_gizmo.visible = False
    agent2_nominal_vis = server.scene.add_spline_catmull_rom(
        "/agent2_nominal_path", points=np.stack([START, GOAL]),
        color=(150, 150, 150), line_width=2)
    agent2_nominal_vis.visible = False
    agent2_start_marker = server.scene.add_icosphere(
        "/agent2_start_marker", radius=0.03, color=(160, 40, 160), position=tuple(START))
    agent2_start_marker.visible = False
    agent2_goal_marker = server.scene.add_icosphere(
        "/agent2_goal_marker", radius=0.03, color=(220, 140, 40), position=tuple(GOAL))
    agent2_goal_marker.visible = False
    path_vis2 = server.scene.add_spline_catmull_rom(
        "/short_horizon_path_2", points=np.stack([START, GOAL]),
        color=(230, 120, 20), line_width=4)
    path_vis2.visible = False

    # Draggable obstacle gizmo. Both a sphere and a box mesh live under it at
    # all times; only the one matching `kind_dropdown` is shown, so switching
    # kinds doesn't need to re-create the gizmo itself.
    gizmo = server.scene.add_transform_controls(
        "/obstacle_gizmo", scale=0.3, disable_rotations=True,
        position=tuple(OBSTACLE_INITIAL_CENTER))
    sphere_mesh = server.scene.add_icosphere(
        "/obstacle_gizmo/sphere", radius=SPHERE_RADIUS,
        color=(220, 60, 60), opacity=0.6)
    sphere_margin_mesh = server.scene.add_icosphere(
        "/obstacle_gizmo/sphere_margin", radius=SPHERE_RADIUS + OBSTACLE_MARGIN,
        color=(220, 60, 60), opacity=0.15)
    box_mesh = server.scene.add_box(
        "/obstacle_gizmo/box", dimensions=tuple(2.0 * BOX_HALF_EXTENTS),
        color=(220, 60, 60), opacity=0.6)
    box_margin_mesh = server.scene.add_box(
        "/obstacle_gizmo/box_margin",
        dimensions=tuple(2.0 * (BOX_HALF_EXTENTS + OBSTACLE_MARGIN)),
        color=(220, 60, 60), opacity=0.15)

    # GraphShortPathMPC takes its tuning knobs (INCLUDING agent count, since
    # that's baked into the graph too) at CONSTRUCTION time, not per-solve --
    # so toggling the second agent, like changing weight/acceleration weight,
    # means rebuilding both the graph and the solver instance. Dragging a
    # gizmo does NOT rebuild -- it reuses the same instance and just calls
    # .solve() again, which is what lets the solver's own cross-cycle warm
    # start (previous _points/_vels seed the next solve's initial guess)
    # actually kick in as you drag, instead of starting cold on every frame.
    obstacles = ObstacleSet()
    mpc_holder = {"mpc": None, "num_agents": 1}

    def rebuild_mpc():
        num_agents = 2 if second_agent_checkbox.value else 1
        graph = make_graph(num_agents)
        agent_radius_slider.visible = num_agents == 2
        # Empty (not per-agent-filled) when there's only one agent --
        # GraphShortPathMPC treats that identically to all-zero radii, and
        # there's no pair to avoid anyway.
        agent_radii = (np.full(num_agents, agent_radius_slider.value)
                        if num_agents == 2 else np.array([]))
        mpc_holder["mpc"] = GraphShortPathMPC(
            graph, NUM_STEPS, num_agents, TIME_PER_STEP, obstacles, agent_radii,
            acceleration_weight=10.0 ** accel_log_slider.value,
            penalty_weight=weight_slider.value)
        mpc_holder["num_agents"] = num_agents

    def agent2_start_goal():
        half = np.array([AGENT2_PATH_LENGTH / 2.0, 0.0, 0.0])
        pos = np.array(agent2_gizmo.position)
        wxyz = np.array(agent2_gizmo.wxyz)
        return pos + quat_rotate(wxyz, -half), pos + quat_rotate(wxyz, half)

    def resolve():
        obstacles.clear()
        center = np.array(gizmo.position)
        is_box = kind_dropdown.value == "box"
        if is_box:
            obstacles.add_box(center, BOX_HALF_EXTENTS, OBSTACLE_MARGIN)
        else:
            obstacles.add_sphere(center, SPHERE_RADIUS, OBSTACLE_MARGIN)
        sphere_mesh.visible = not is_box
        sphere_margin_mesh.visible = not is_box
        box_mesh.visible = is_box
        box_margin_mesh.visible = is_box

        num_agents = mpc_holder["num_agents"]
        two_agents = num_agents == 2
        agent2_gizmo.visible = two_agents
        agent2_nominal_vis.visible = two_agents
        agent2_start_marker.visible = two_agents
        agent2_goal_marker.visible = two_agents
        path_vis2.visible = two_agents

        splines = [spline1]
        starts = [START]
        if two_agents:
            start2, goal2 = agent2_start_goal()
            splines.append(make_linear_spline(start2, goal2))
            starts.append(start2)
            agent2_nominal_vis.points = np.stack([start2, goal2])
            agent2_start_marker.position = tuple(start2)
            agent2_goal_marker.position = tuple(goal2)

        x0 = np.concatenate(starts)
        v0 = np.zeros(DIM * num_agents)

        mpc = mpc_holder["mpc"]
        ok = mpc.solve(x0, v0, np.array([], dtype=np.int32), [], splines)
        if not ok:
            status_text.value = "solve FAILED (obstacle/agents likely blocking all feasible paths)"
            return
        pts = mpc.view_points()
        path_vis1.points = pts[:, 0:DIM]

        if is_box:
            q = np.abs(pts[:, 0:DIM] - center) - BOX_HALF_EXTENTS
            clearance = float(np.min(np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)))
        else:
            clearance = float(np.linalg.norm(pts[:, 0:DIM] - center, axis=1).min()) - SPHERE_RADIUS
        msg = (f"solve ok  |  solve_time={mpc.get_last_solve_time() * 1000:.2f} ms  |  "
               f"agent-1 obstacle clearance={clearance:.3f} (required >= {OBSTACLE_MARGIN:.3f})")

        if two_agents:
            path_vis2.points = pts[:, DIM:2 * DIM]
            inter_dist = float(np.linalg.norm(pts[:, 0:DIM] - pts[:, DIM:2 * DIM], axis=1).min())
            required = 2.0 * agent_radius_slider.value
            msg += f"  |  min inter-agent distance={inter_dist:.3f} (required >= {required:.3f})"
        status_text.value = msg

    def rebuild_and_resolve():
        rebuild_mpc()
        resolve()

    @gizmo.on_update
    def _(_):
        resolve()

    @agent2_gizmo.on_update
    def _(_):
        resolve()

    @second_agent_checkbox.on_update
    def _(_):
        rebuild_and_resolve()

    @kind_dropdown.on_update
    def _(_):
        rebuild_and_resolve()

    @weight_slider.on_update
    def _(_):
        rebuild_and_resolve()

    @accel_log_slider.on_update
    def _(_):
        rebuild_and_resolve()

    @agent_radius_slider.on_update
    def _(_):
        rebuild_and_resolve()

    rebuild_and_resolve()
    print("Open the printed URL above. Drag the red obstacle (and, once "
          "'Enable second agent' is ticked, the second path gizmo -- drag to "
          "move it, use the rotation rings to re-orient it) and use the GUI "
          "controls on the left to compare solve behavior live.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()

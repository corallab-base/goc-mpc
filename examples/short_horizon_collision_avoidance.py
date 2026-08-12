"""
Interactive sphere-obstacle demo for the short-horizon MPC (stage 2 of 3).

A single point-mass agent has a FIXED nominal reference path (straight line,
start -> goal, drawn in gray). A red sphere obstacle is draggable in the
browser (viser transform-controls gizmo). Every time you drag it,
`GraphShortPathMPC.solve()` is re-run from scratch against the (unchanged)
nominal reference, and the resulting short-horizon trajectory (blue line) is
redrawn -- showing live how the hard sphere-clearance constraint + soft
Lorentzian repulsion cost deflect the path as the obstacle gets close.

This deliberately calls `GraphShortPathMPC` directly (not the full
`GraphOfConstraintsMPC.step()` closed loop) and always resolves from the same
fixed start state -- that's what "a nominal spline that's fixed" means here,
and it isolates the thing being demonstrated (does the short path react
correctly to a nearby obstacle?) from a separate, known closed-loop issue:
repeatedly feeding a solved trajectory's own state back as the next cycle's
x0 can make the agent drift/stall near an obstacle rather than pass it,
which needs further investigation and isn't what this script exercises.

Run:
    python examples/short_horizon_collision_avoidance.py
Then open the printed URL and drag the red sphere.
"""

import time

import numpy as np
import viser

from goc_mpc import GraphOfConstraints, GraphShortPathMPC, ObstacleSet
from goc_mpc._ext.configuration_spline import CubicConfigurationSpline, Block

DIM = 3
START = np.array([0.0, 0.0, 1.0])
GOAL = np.array([0.0, 1.0, 1.0])

OBSTACLE_RADIUS = 0.15
OBSTACLE_MARGIN = 0.05
OBSTACLE_INITIAL_CENTER = np.array([0.6, 0.5, 1.0])  # off to the side initially
REPULSION_WEIGHT = 20.0

NUM_STEPS = 10
TIME_PER_STEP = 0.1


def make_graph() -> GraphOfConstraints:
    graph = GraphOfConstraints(
        [[Block.R(DIM)]], [],
        state_lower_bound=-10.0, state_upper_bound=10.0,
        robot_names=["agent"], object_names=[],
    )
    graph.structure.add_nodes(1)
    return graph


def make_nominal_spline() -> CubicConfigurationSpline:
    """The FIXED reference: a straight line from START to GOAL. Never
    changes as the obstacle moves -- only the short-horizon MPC's reaction
    to it does."""
    spline = CubicConfigurationSpline([Block.R(DIM)])
    spline.set_linear(True)
    spline.set(np.stack([START, GOAL]), np.zeros((2, DIM)), np.array([0.0, 1.0]))
    return spline


def main():
    graph = make_graph()
    spline = make_nominal_spline()

    obstacles = ObstacleSet()
    obstacles.add_sphere(OBSTACLE_INITIAL_CENTER, OBSTACLE_RADIUS, OBSTACLE_MARGIN)
    mpc = GraphShortPathMPC(graph, NUM_STEPS, 1, DIM, TIME_PER_STEP, obstacles, REPULSION_WEIGHT)

    server = viser.ViserServer()

    # Fixed nominal path + endpoint markers.
    server.scene.add_spline_catmull_rom(
        "/nominal_path", points=np.stack([START, GOAL]),
        color=(150, 150, 150), line_width=2)
    server.scene.add_icosphere("/start_marker", radius=0.03, color=(40, 160, 40),
                                position=tuple(START))
    server.scene.add_icosphere("/goal_marker", radius=0.03, color=(40, 90, 220),
                                position=tuple(GOAL))

    # Draggable obstacle -- the sphere itself, plus a translucent outer shell
    # showing the registered margin (the actual clearance boundary is
    # radius + margin, not just the solid sphere).
    gizmo = server.scene.add_transform_controls(
        "/obstacle_gizmo", scale=0.3, disable_rotations=True,
        position=tuple(OBSTACLE_INITIAL_CENTER))
    server.scene.add_icosphere(
        "/obstacle_gizmo/sphere", radius=OBSTACLE_RADIUS,
        color=(220, 60, 60), opacity=0.6)
    server.scene.add_icosphere(
        "/obstacle_gizmo/margin", radius=OBSTACLE_RADIUS + OBSTACLE_MARGIN,
        color=(220, 60, 60), opacity=0.15)

    # Live, reactive short-horizon path.
    path_vis = server.scene.add_spline_catmull_rom(
        "/short_horizon_path", points=np.stack([START, GOAL]),
        color=(30, 160, 220), line_width=4)
    status_text = server.gui.add_text("Status", initial_value="")

    def resolve():
        obstacles.clear()
        obstacles.add_sphere(np.array(gizmo.position), OBSTACLE_RADIUS, OBSTACLE_MARGIN)
        ok = mpc.solve(START, np.zeros(DIM), np.array([], dtype=np.int32), [], [spline])
        if not ok:
            status_text.value = "solve FAILED (obstacle likely blocking all feasible paths)"
            return
        pts = mpc.view_points()
        path_vis.points = pts
        dmin = float(np.linalg.norm(pts - np.array(gizmo.position), axis=1).min())
        required = OBSTACLE_RADIUS + OBSTACLE_MARGIN
        status_text.value = (
            f"solve ok  |  solve_time={mpc.get_last_solve_time() * 1000:.1f} ms  |  "
            f"min clearance={dmin:.3f}  (required >= {required:.3f})")

    @gizmo.on_update
    def _(_):
        resolve()

    resolve()
    print("Open the printed URL above, then drag the red sphere and watch "
          "the blue short-horizon path react.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()

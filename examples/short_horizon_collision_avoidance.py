"""
Interactive obstacle demo for the short-horizon MPC (stage 2 of 3).

A single point-mass agent has a FIXED nominal reference path (straight line,
start -> goal, drawn in gray). A draggable obstacle (sphere or box, pick via
the GUI) sits in the browser (viser transform-controls gizmo). Every time you
drag it -- or change a GUI control -- the selected solver's `.solve()` is
re-run from scratch against the (unchanged) nominal reference, and the
resulting short-horizon trajectory (blue line) is redrawn.

Four solvers, picked live via the "Solver" dropdown:
  - SLSQP (hard constraint): GraphShortPathMPC, use_hard_constraints=True.
    A real clearance guarantee when it converges, but can report the whole
    solve infeasible with several obstacles simultaneously active.
  - SLSQP (soft penalty): GraphShortPathMPC, use_hard_constraints=False.
    Structurally can't report infeasible -- SLSQP with no AddConstraint
    calls just does local quasi-Newton minimization of a smooth function --
    but clearance depends on the weight slider: too low and the tracking/
    acceleration cost can pull the path inside the obstacle.
  - ADMM: AdmmShortPathMPC. Also structurally can't report infeasible (a
    closed-form projection can't fail), but gets there differently -- see
    that class's own doc comment (admm_short_path_mpc.hpp): the smooth cost
    is solved as a plain (unconstrained) per-axis linear system, factorized
    once and reused, with obstacle avoidance handled entirely by cheap
    closed-form projections outside the linear solve. Typically the fastest
    of the three (microseconds to low-milliseconds even with a point cloud)
    and, unlike the SLSQP-soft-penalty path, its "rho" doesn't trade away
    speed for clearance the way that path's weight does.
  - SQP: SqpShortPathMPC (see sqp_short_path_mpc.hpp) -- Riemannian
    trust-region SQP over qpOASES. Every obstacle constraint is a
    slack-relaxed exact-penalty row, so the QP subproblem is feasible by
    construction at every outer iteration (fixing SLSQP's hard-constraint
    infeasibility failure directly, not by tuning), plus a closed-form
    safety-projection final pass gives the same hard clearance guarantee
    ADMM has regardless of SQP convergence. The weight slider maps to its
    penalty weight.

This deliberately calls the solver classes directly (not the full
`GraphOfConstraintsMPC.step()` closed loop) and always resolves from the same
fixed start state -- that's what "a nominal spline that's fixed" means here,
and it isolates the thing being demonstrated (does the short path react
correctly to a nearby obstacle?) from a separate, known closed-loop issue:
repeatedly feeding a solved trajectory's own state back as the next cycle's
x0 can make the agent drift/stall near an obstacle rather than pass it,
which needs further investigation and isn't what this script exercises.

Run:
    python examples/short_horizon_collision_avoidance.py
Then open the printed URL, drag the red obstacle, and play with the GUI
controls on the left.
"""

import time

import numpy as np
import viser

from goc_mpc import (GraphOfConstraints, GraphShortPathMPC, AdmmShortPathMPC, SqpShortPathMPC,
                      ObstacleSet)
from goc_mpc._ext.configuration_spline import CubicConfigurationSpline, Block

DIM = 3
START = np.array([0.0, 0.0, 1.0])
GOAL = np.array([0.0, 1.0, 1.0])

SPHERE_RADIUS = 0.15
BOX_HALF_EXTENTS = np.array([0.25, 0.25, 0.25])
OBSTACLE_MARGIN = 0.05
OBSTACLE_INITIAL_CENTER = np.array([0.6, 0.5, 1.0])  # off to the side initially

NUM_STEPS = 10
TIME_PER_STEP = 0.1

SOLVERS = ("ADMM", "SQP", "SLSQP (hard constraint)", "SLSQP (soft penalty)")


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

    server = viser.ViserServer()

    kind_dropdown = server.gui.add_dropdown("Obstacle kind", ("sphere", "box"), initial_value="box")
    solver_dropdown = server.gui.add_dropdown("Solver", SOLVERS, initial_value="ADMM")
    weight_slider = server.gui.add_slider(
        "Repulsion weight (SLSQP) / rho (ADMM) / penalty weight (SQP)",
        min=0.1, max=2000.0, step=0.1, initial_value=5.0)
    admm_iters_slider = server.gui.add_slider(
        "ADMM iterations", min=1, max=50, step=1, initial_value=15)
    status_text = server.gui.add_text("Status", initial_value="")

    # Fixed nominal path + endpoint markers.
    server.scene.add_spline_catmull_rom(
        "/nominal_path", points=np.stack([START, GOAL]),
        color=(150, 150, 150), line_width=2)
    server.scene.add_icosphere("/start_marker", radius=0.03, color=(40, 160, 40),
                                position=tuple(START))
    server.scene.add_icosphere("/goal_marker", radius=0.03, color=(40, 90, 220),
                                position=tuple(GOAL))

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

    # Live, reactive short-horizon path.
    path_vis = server.scene.add_spline_catmull_rom(
        "/short_horizon_path", points=np.stack([START, GOAL]),
        color=(30, 160, 220), line_width=4)

    # Every solver class takes its tuning knobs at CONSTRUCTION time (not
    # per-solve), so changing solver/weight/iterations from the GUI means
    # rebuilding the instance. Dragging the gizmo does NOT rebuild -- it
    # reuses the same instance and just calls .solve() again, which is what
    # lets each class's own cross-cycle warm start (previous _points/_vels
    # seed the next solve's initial guess) actually kick in as you drag,
    # instead of starting cold on every frame.
    obstacles = ObstacleSet()
    mpc_holder = {"mpc": None, "is_admm": False}

    def rebuild_mpc():
        solver = solver_dropdown.value
        is_admm = solver == "ADMM"
        admm_iters_slider.visible = is_admm
        if is_admm:
            mpc_holder["mpc"] = AdmmShortPathMPC(
                graph, NUM_STEPS, 1, DIM, TIME_PER_STEP, obstacles,
                rho=weight_slider.value, num_iterations=int(admm_iters_slider.value))
        elif solver == "SQP":
            mpc_holder["mpc"] = SqpShortPathMPC(
                graph, NUM_STEPS, 1, DIM, TIME_PER_STEP, obstacles,
                penalty_weight=weight_slider.value)
        else:
            use_hard = solver == "SLSQP (hard constraint)"
            mpc_holder["mpc"] = GraphShortPathMPC(
                graph, NUM_STEPS, 1, DIM, TIME_PER_STEP, obstacles,
                weight_slider.value, use_hard)
        mpc_holder["is_admm"] = is_admm

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

        mpc = mpc_holder["mpc"]
        ok = mpc.solve(START, np.zeros(DIM), np.array([], dtype=np.int32), [], [spline])
        if not ok:
            status_text.value = "solve FAILED (obstacle likely blocking all feasible paths)"
            return
        pts = mpc.view_points()
        path_vis.points = pts
        if is_box:
            q = np.abs(pts - center) - BOX_HALF_EXTENTS
            clearance = float(np.min(np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)))
        else:
            clearance = float(np.linalg.norm(pts - center, axis=1).min()) - SPHERE_RADIUS
        status_text.value = (
            f"solve ok  |  solve_time={mpc.get_last_solve_time() * 1000:.2f} ms  |  "
            f"min clearance={clearance:.3f}  (required >= {OBSTACLE_MARGIN:.3f})")

    def rebuild_and_resolve():
        rebuild_mpc()
        resolve()

    @gizmo.on_update
    def _(_):
        resolve()

    @kind_dropdown.on_update
    def _(_):
        rebuild_and_resolve()

    @solver_dropdown.on_update
    def _(_):
        rebuild_and_resolve()

    @weight_slider.on_update
    def _(_):
        rebuild_and_resolve()

    @admm_iters_slider.on_update
    def _(_):
        rebuild_and_resolve()

    rebuild_and_resolve()
    print("Open the printed URL above. Drag the red obstacle and use the "
          "GUI controls (obstacle kind, solver, weight/rho, ADMM "
          "iterations) to compare solve behavior live.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()

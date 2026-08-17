"""
Interactive demo for SqpShortPathMPC's Stage 4 (R^3 x SO(3) quaternion
manifold support) with a VARIABLE number of agents and obstacles, and a
rigorous, reproducible comparison of its solve time against the SAME
solver restricted to R^3 (position only) -- the config space it originally
shipped with (Stage 1/2).

Up to MAX_AGENTS agents and MAX_OBSTACLES obstacles exist as scene nodes
from startup (toggled visible/invisible, not created/destroyed at
runtime); "Num agents"/"Num obstacles" pick how many are ACTIVE (used in
solve()). Each agent has a FIXED default start (arranged on a ring so
multiple agents' nominal paths cross near the center -- a real stress case
for inter-agent avoidance, not an easy non-interacting one) and a
draggable+rotatable GOAL gizmo (position + desired final orientation).
Each obstacle has a draggable gizmo (sphere or box, picked globally). All
gizmos keep whatever position you last dragged them to even while hidden,
so growing/shrinking the counts doesn't lose your setup.

  - "Config space" ("Position only (R^3)" / "Position + orientation (R^3 x
    SO3Quat)"): applies to every ACTIVE agent uniformly -- SqpShortPathMPC
    requires every agent to share the same ambient/tangent width (see its
    own constructor comment), so a per-agent mix isn't supported here
    either.

Two ways to get real numbers (not just eyeballing a single drag's solve
time, which is noisy):
  - "Run timing benchmark": solves the CURRENT scene (however many agents/
    obstacles are active, at their current gizmo positions) `repeats`
    times, COLD (fresh SqpShortPathMPC instance every call) and WARM (one
    instance, repeated `.solve()` -- steady-state MPC operation, see
    SqpShortPathMPC's own "warm start" comment), for both config spaces.
  - "Run scaling sweep": fixes the OTHER count at its current value and
    varies obstacle count 0..(current) and agent count 1..(current) one at
    a time, WARM only, for both config spaces -- directly shows how solve
    time scales with each, using a PREFIX of the currently active/
    positioned agents and obstacles at each step (not a separately
    regenerated layout). Worth knowing going in: neither registered
    sphere/box obstacle rows nor inter-agent pair rows are distance-pruned
    in this solver (see LinearizeObstacleConstraints/
    LinearizeAgentPairConstraints) -- every registered obstacle and every
    agent pair gets QP rows at every step REGARDLESS of proximity, so the
    sweep's exact positions don't materially change the timing scaling
    it's measuring, only whether the constraints are actually deflecting
    anything (visible in the path/clearance readout, not the benchmark).

Run:
    python examples/so3_short_horizon_collision_avoidance.py
"""

import time

import numpy as np
import viser

from goc_mpc import GraphOfConstraints, SqpShortPathMPC, ObstacleSet
from goc_mpc._ext.configuration_spline import CubicConfigurationSpline, Block

MAX_AGENTS = 6
MAX_OBSTACLES = 8

AGENT_RING_RADIUS = 1.2
OBSTACLE_JITTER_RADIUS = 0.5
Z_HEIGHT = 1.0
IDENTITY_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])

AGENT_COLORS = [
    (30, 160, 220), (230, 120, 20), (140, 70, 200),
    (40, 180, 120), (220, 60, 140), (200, 170, 30),
]

SPHERE_RADIUS = 0.15
BOX_HALF_EXTENTS = np.array([0.2, 0.2, 0.2])
OBSTACLE_MARGIN = 0.05

NUM_STEPS = 10
TIME_PER_STEP = 0.1
NUM_TRAJ_FRAMES = (NUM_STEPS + 1) // 2  # orientation frames per agent, every other step

MODES = ("Position only (R^3)", "Position + orientation (R^3 x SO3Quat)")


def default_agent_start(i: int) -> np.ndarray:
    angle = 2.0 * np.pi * i / MAX_AGENTS
    return np.array([AGENT_RING_RADIUS * np.cos(angle), AGENT_RING_RADIUS * np.sin(angle), Z_HEIGHT])


def default_agent_goal(i: int) -> np.ndarray:
    # Diametrically opposite the start -- every agent's nominal path passes
    # near the center by default, so inter-agent avoidance is exercised
    # immediately, not just position-vs-obstacle avoidance.
    angle = 2.0 * np.pi * i / MAX_AGENTS + np.pi
    return np.array([AGENT_RING_RADIUS * np.cos(angle), AGENT_RING_RADIUS * np.sin(angle), Z_HEIGHT])


def default_obstacle_position(j: int) -> np.ndarray:
    rng = np.random.default_rng(1000 + j)  # deterministic, distinct per slot
    offset = rng.uniform(-OBSTACLE_JITTER_RADIUS, OBSTACLE_JITTER_RADIUS, size=2)
    return np.array([offset[0], offset[1], Z_HEIGHT])


def make_graph(num_agents: int, with_orientation: bool) -> GraphOfConstraints:
    spec = [Block.R(3), Block.SO3Quat()] if with_orientation else [Block.R(3)]
    graph = GraphOfConstraints(
        [spec] * num_agents, [], state_lower_bound=-10.0, state_upper_bound=10.0,
        robot_names=[f"agent{i}" for i in range(num_agents)], object_names=[],
    )
    graph.structure.add_nodes(1)
    return graph


def make_ref_spline(with_orientation: bool, p0: np.ndarray, p1: np.ndarray,
                     q0: np.ndarray, q1: np.ndarray) -> CubicConfigurationSpline:
    """A FIXED reference from (p0,q0) to (p1,q1) -- q0/q1 are ignored when
    `with_orientation` is False (position-only config space)."""
    if with_orientation:
        spline = CubicConfigurationSpline([Block.R(3), Block.SO3Quat()])
        spline.set_linear(False)  # "linear" ambient interpolation isn't meaningful for a quaternion block
        pos = np.stack([np.concatenate([p0, q0]), np.concatenate([p1, q1])])
        vel = np.zeros((2, 6))  # tangent dim: 3 (R) + 3 (SO3Quat)
    else:
        spline = CubicConfigurationSpline([Block.R(3)])
        spline.set_linear(True)
        pos = np.stack([p0, p1])
        vel = np.zeros((2, 3))
    spline.set(pos, vel, np.array([0.0, 1.0]))
    return spline


def main():
    server = viser.ViserServer()

    mode_dropdown = server.gui.add_dropdown("Config space", MODES, initial_value=MODES[0])
    num_agents_number = server.gui.add_number("Num agents", initial_value=1, min=1, max=MAX_AGENTS, step=1)
    num_obstacles_number = server.gui.add_number(
        "Num obstacles", initial_value=1, min=0, max=MAX_OBSTACLES, step=1)
    kind_dropdown = server.gui.add_dropdown("Obstacle kind", ("sphere", "box"), initial_value="box")
    agent_radius_slider = server.gui.add_slider(
        "Agent radius (inter-agent avoidance)", min=0.0, max=0.5, step=0.01, initial_value=0.1)
    penalty_slider = server.gui.add_slider(
        "SQP penalty weight", min=0.1, max=2000.0, step=0.1, initial_value=1000.0)
    tracking_log_slider = server.gui.add_slider(
        "SQP tracking weight (log10)", min=0.0, max=4.0, step=0.1, initial_value=0.0)
    status_text = server.gui.add_text("Status", initial_value="", multiline=True)

    server.gui.add_markdown("---\n**Timing benchmark** (current scene)")
    repeats_number = server.gui.add_number("Repeats per regime", initial_value=20, min=1, max=200, step=1)
    benchmark_button = server.gui.add_button("Run timing benchmark")
    benchmark_text = server.gui.add_text("Benchmark result", initial_value="(not run yet)", multiline=True)

    server.gui.add_markdown("---\n**Scaling sweep** (warm-started only)")
    sweep_button = server.gui.add_button("Run scaling sweep")
    sweep_text = server.gui.add_text("Sweep result", initial_value="(not run yet)", multiline=True)

    # --- Per-agent scene nodes: all MAX_AGENTS created up front, toggled
    # visible/invisible by num_agents_number, never destroyed -- gizmo
    # position/rotation state persists across visibility toggles, so
    # growing/shrinking the count never loses a drag.
    start_markers, start_frames = [], []
    goal_gizmos, goal_frames = [], []
    nominal_path_vis, path_vis = [], []
    traj_frames = []
    for i in range(MAX_AGENTS):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        start = default_agent_start(i)
        goal = default_agent_goal(i)
        start_markers.append(server.scene.add_icosphere(
            f"/agent{i}/start_marker", radius=0.04, color=color, position=tuple(start)))
        start_frames.append(server.scene.add_frame(
            f"/agent{i}/start_frame", wxyz=tuple(IDENTITY_WXYZ), position=tuple(start),
            axes_length=0.12, axes_radius=0.007))
        goal_gizmos.append(server.scene.add_transform_controls(
            f"/agent{i}/goal_gizmo", scale=0.25, position=tuple(goal), wxyz=tuple(IDENTITY_WXYZ)))
        goal_frames.append(server.scene.add_frame(
            f"/agent{i}/goal_gizmo/frame", axes_length=0.12, axes_radius=0.007))
        nominal_path_vis.append(server.scene.add_spline_catmull_rom(
            f"/agent{i}/nominal_path", points=np.stack([start, goal]), color=(150, 150, 150), line_width=1))
        path_vis.append(server.scene.add_spline_catmull_rom(
            f"/agent{i}/short_horizon_path", points=np.stack([start, goal]), color=color, line_width=4))
        traj_frames.append([
            server.scene.add_frame(f"/agent{i}/traj_frame_{k}", axes_length=0.08, axes_radius=0.004,
                                    visible=False)
            for k in range(NUM_TRAJ_FRAMES)
        ])

    # --- Per-obstacle scene nodes: same up-front/toggle-visible pattern.
    obstacle_gizmos, sphere_meshes, sphere_margin_meshes, box_meshes, box_margin_meshes = [], [], [], [], []
    for j in range(MAX_OBSTACLES):
        pos = default_obstacle_position(j)
        obstacle_gizmos.append(server.scene.add_transform_controls(
            f"/obstacle{j}/gizmo", scale=0.25, disable_rotations=True, position=tuple(pos)))
        sphere_meshes.append(server.scene.add_icosphere(
            f"/obstacle{j}/gizmo/sphere", radius=SPHERE_RADIUS, color=(220, 60, 60), opacity=0.6))
        sphere_margin_meshes.append(server.scene.add_icosphere(
            f"/obstacle{j}/gizmo/sphere_margin", radius=SPHERE_RADIUS + OBSTACLE_MARGIN,
            color=(220, 60, 60), opacity=0.15))
        box_meshes.append(server.scene.add_box(
            f"/obstacle{j}/gizmo/box", dimensions=tuple(2.0 * BOX_HALF_EXTENTS),
            color=(220, 60, 60), opacity=0.6))
        box_margin_meshes.append(server.scene.add_box(
            f"/obstacle{j}/gizmo/box_margin", dimensions=tuple(2.0 * (BOX_HALF_EXTENTS + OBSTACLE_MARGIN)),
            color=(220, 60, 60), opacity=0.15))

    mpc_holder = {"mpc": None, "num_agents": 1, "with_orientation": False}

    def current_weights():
        return dict(penalty_weight=penalty_slider.value, tracking_weight=10.0 ** tracking_log_slider.value)

    def agent_radii_for(n: int) -> np.ndarray:
        return np.full(n, agent_radius_slider.value) if n >= 2 else np.array([])

    def build_mpc(num_agents: int, with_orientation: bool, obstacles: ObstacleSet) -> SqpShortPathMPC:
        graph = make_graph(num_agents, with_orientation)
        dim = 6 if with_orientation else 3
        return SqpShortPathMPC(graph, NUM_STEPS, num_agents, dim, TIME_PER_STEP, obstacles,
                                agent_radii_for(num_agents), **current_weights())

    def build_obstacles(num_obstacles: int) -> ObstacleSet:
        obstacles = ObstacleSet()
        is_box = kind_dropdown.value == "box"
        for j in range(num_obstacles):
            center = np.array(obstacle_gizmos[j].position)
            if is_box:
                obstacles.add_box(center, BOX_HALF_EXTENTS, OBSTACLE_MARGIN)
            else:
                obstacles.add_sphere(center, SPHERE_RADIUS, OBSTACLE_MARGIN)
        return obstacles

    def build_refs(num_agents: int, with_orientation: bool):
        refs, starts = [], []
        for i in range(num_agents):
            start = default_agent_start(i)
            goal = np.array(goal_gizmos[i].position)
            goal_wxyz = np.array(goal_gizmos[i].wxyz)
            goal_frames[i].wxyz = tuple(goal_wxyz)
            nominal_path_vis[i].points = np.stack([start, goal])
            refs.append(make_ref_spline(with_orientation, start, goal, IDENTITY_WXYZ, goal_wxyz))
            starts.append(np.concatenate([start, IDENTITY_WXYZ]) if with_orientation else start)
        return refs, starts

    obstacles_live = ObstacleSet()

    def update_visibility(num_agents: int, num_obstacles: int, with_orientation: bool):
        is_box = kind_dropdown.value == "box"
        for i in range(MAX_AGENTS):
            active = i < num_agents
            start_markers[i].visible = active
            start_frames[i].visible = active
            goal_gizmos[i].visible = active
            nominal_path_vis[i].visible = active
            path_vis[i].visible = active
            for frame in traj_frames[i]:
                frame.visible = active and with_orientation
        for j in range(MAX_OBSTACLES):
            active = j < num_obstacles
            obstacle_gizmos[j].visible = active
            sphere_meshes[j].visible = active and not is_box
            sphere_margin_meshes[j].visible = active and not is_box
            box_meshes[j].visible = active and is_box
            box_margin_meshes[j].visible = active and is_box

    def rebuild_mpc():
        num_agents = int(num_agents_number.value)
        with_orientation = mode_dropdown.value == MODES[1]
        obstacles_live.clear()
        num_obstacles = int(num_obstacles_number.value)
        is_box = kind_dropdown.value == "box"
        for j in range(num_obstacles):
            center = np.array(obstacle_gizmos[j].position)
            if is_box:
                obstacles_live.add_box(center, BOX_HALF_EXTENTS, OBSTACLE_MARGIN)
            else:
                obstacles_live.add_sphere(center, SPHERE_RADIUS, OBSTACLE_MARGIN)
        mpc_holder["mpc"] = build_mpc(num_agents, with_orientation, obstacles_live)
        mpc_holder["num_agents"] = num_agents
        mpc_holder["with_orientation"] = with_orientation

    def resolve():
        num_agents = mpc_holder["num_agents"]
        with_orientation = mpc_holder["with_orientation"]
        num_obstacles = int(num_obstacles_number.value)
        update_visibility(num_agents, num_obstacles, with_orientation)

        # Obstacles may have been DRAGGED since rebuild_mpc() last ran --
        # update the live ObstacleSet in place (SqpShortPathMPC stores a
        # POINTER to it, see its own doc comment, so this is picked up
        # without reconstructing the solver).
        obstacles_live.clear()
        is_box = kind_dropdown.value == "box"
        for j in range(num_obstacles):
            center = np.array(obstacle_gizmos[j].position)
            if is_box:
                obstacles_live.add_box(center, BOX_HALF_EXTENTS, OBSTACLE_MARGIN)
            else:
                obstacles_live.add_sphere(center, SPHERE_RADIUS, OBSTACLE_MARGIN)

        refs, starts = build_refs(num_agents, with_orientation)
        dim = 6 if with_orientation else 3
        x0 = np.concatenate(starts)
        v0 = np.zeros(dim * num_agents)

        mpc = mpc_holder["mpc"]
        ok = mpc.solve(x0, v0, np.array([], dtype=np.int32), [], refs)
        if not ok:
            status_text.value = "solve FAILED"
            return
        pts = mpc.view_points()
        ambient = 7 if with_orientation else 3

        min_clearance = np.inf
        for i in range(num_agents):
            pts_i = pts[:, i * ambient:(i + 1) * ambient]
            pos_i = pts_i[:, 0:3]
            path_vis[i].points = pos_i
            for j in range(num_obstacles):
                center = np.array(obstacle_gizmos[j].position)
                if is_box:
                    q = np.abs(pos_i - center) - BOX_HALF_EXTENTS
                    clr = float(np.min(np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)))
                else:
                    clr = float(np.linalg.norm(pos_i - center, axis=1).min()) - SPHERE_RADIUS
                min_clearance = min(min_clearance, clr)

            if with_orientation:
                quat_i = pts_i[:, 3:7]
                for k, frame in enumerate(traj_frames[i]):
                    step = min(2 * k, NUM_STEPS - 1)
                    q = quat_i[step]
                    q = q / np.linalg.norm(q)
                    frame.wxyz = tuple(q)
                    frame.position = tuple(pos_i[step])

        min_inter_agent = np.inf
        for i in range(num_agents):
            for i2 in range(i + 1, num_agents):
                pos_i = pts[:, i * ambient:i * ambient + 3]
                pos_i2 = pts[:, i2 * ambient:i2 * ambient + 3]
                min_inter_agent = min(min_inter_agent, float(np.linalg.norm(pos_i - pos_i2, axis=1).min()))

        mode_label = "R^3 x SO3Quat" if with_orientation else "R^3 only"
        msg = (f"mode={mode_label}  agents={num_agents}  obstacles={num_obstacles}  |  "
               f"solve_time={mpc.get_last_solve_time() * 1000:.3f} ms  |  iterations={mpc.get_last_iterations()}")
        if num_obstacles > 0:
            msg += f"\nmin obstacle clearance={min_clearance:.3f} (required >= {OBSTACLE_MARGIN:.3f})"
        if num_agents > 1:
            required = 2.0 * agent_radius_slider.value
            msg += f"\nmin inter-agent distance={min_inter_agent:.3f} (required >= {required:.3f})"
        status_text.value = msg

    def rebuild_and_resolve():
        rebuild_mpc()
        resolve()

    def run_timing_benchmark():
        num_agents = int(num_agents_number.value)
        num_obstacles = int(num_obstacles_number.value)
        obstacles = build_obstacles(num_obstacles)
        repeats = int(repeats_number.value)
        weights = current_weights()

        results = {}
        for with_orientation in (False, True):
            dim = 6 if with_orientation else 3
            refs, starts = build_refs(num_agents, with_orientation)
            x0 = np.concatenate(starts)
            v0 = np.zeros(dim * num_agents)

            cold_times, cold_iters = [], []
            for _ in range(repeats):
                mpc = build_mpc(num_agents, with_orientation, obstacles)
                if mpc.solve(x0, v0, np.array([], dtype=np.int32), [], refs):
                    cold_times.append(mpc.get_last_solve_time() * 1000.0)
                    cold_iters.append(mpc.get_last_iterations())

            warm_mpc = build_mpc(num_agents, with_orientation, obstacles)
            warm_times, warm_iters = [], []
            for _ in range(repeats):
                if warm_mpc.solve(x0, v0, np.array([], dtype=np.int32), [], refs):
                    warm_times.append(warm_mpc.get_last_solve_time() * 1000.0)
                    warm_iters.append(warm_mpc.get_last_iterations())

            results[with_orientation] = dict(
                cold_mean=np.mean(cold_times), cold_std=np.std(cold_times), cold_iters=np.mean(cold_iters),
                warm_mean=np.mean(warm_times), warm_std=np.std(warm_times), warm_iters=np.mean(warm_iters),
            )

        r3, r6 = results[False], results[True]
        lines = [
            f"agents={num_agents} obstacles={num_obstacles} repeats={repeats} "
            f"penalty={weights['penalty_weight']:.1f} tracking={weights['tracking_weight']:.3f}",
            "",
            f"R^3 only         cold: {r3['cold_mean']:.3f} +/- {r3['cold_std']:.3f} ms "
            f"({r3['cold_iters']:.1f} iters)",
            f"                 warm: {r3['warm_mean']:.3f} +/- {r3['warm_std']:.3f} ms "
            f"({r3['warm_iters']:.1f} iters)",
            f"R^3 x SO3Quat    cold: {r6['cold_mean']:.3f} +/- {r6['cold_std']:.3f} ms "
            f"({r6['cold_iters']:.1f} iters)",
            f"                 warm: {r6['warm_mean']:.3f} +/- {r6['warm_std']:.3f} ms "
            f"({r6['warm_iters']:.1f} iters)",
            "",
            f"slowdown (cold): {r6['cold_mean'] / max(r3['cold_mean'], 1e-9):.2f}x   "
            f"slowdown (warm): {r6['warm_mean'] / max(r3['warm_mean'], 1e-9):.2f}x",
        ]
        benchmark_text.value = "\n".join(lines)

    def warm_solve_time(num_agents: int, num_obstacles: int, with_orientation: bool, repeats: int) -> float:
        obstacles = build_obstacles(num_obstacles)
        dim = 6 if with_orientation else 3
        refs, starts = build_refs(num_agents, with_orientation)
        x0 = np.concatenate(starts)
        v0 = np.zeros(dim * num_agents)
        mpc = build_mpc(num_agents, with_orientation, obstacles)
        times = []
        for _ in range(repeats):
            if mpc.solve(x0, v0, np.array([], dtype=np.int32), [], refs):
                times.append(mpc.get_last_solve_time() * 1000.0)
        return float(np.mean(times)) if times else float("nan")

    def run_scaling_sweep():
        fixed_agents = int(num_agents_number.value)
        fixed_obstacles = int(num_obstacles_number.value)
        repeats = max(3, int(repeats_number.value) // 4)  # sweep has many points -- fewer reps each

        lines = ["Obstacle sweep (agents fixed at {}):".format(fixed_agents),
                 f"{'obstacles':>10}  {'R^3 (ms)':>10}  {'R^3xQuat (ms)':>14}  {'ratio':>7}"]
        for m in range(0, fixed_obstacles + 1):
            t3 = warm_solve_time(fixed_agents, m, False, repeats)
            t6 = warm_solve_time(fixed_agents, m, True, repeats)
            lines.append(f"{m:>10}  {t3:>10.3f}  {t6:>14.3f}  {t6 / max(t3, 1e-9):>6.2f}x")

        lines.append("")
        lines.append("Agent sweep (obstacles fixed at {}):".format(fixed_obstacles))
        lines.append(f"{'agents':>10}  {'R^3 (ms)':>10}  {'R^3xQuat (ms)':>14}  {'ratio':>7}")
        for n in range(1, fixed_agents + 1):
            t3 = warm_solve_time(n, fixed_obstacles, False, repeats)
            t6 = warm_solve_time(n, fixed_obstacles, True, repeats)
            lines.append(f"{n:>10}  {t3:>10.3f}  {t6:>14.3f}  {t6 / max(t3, 1e-9):>6.2f}x")

        sweep_text.value = "\n".join(lines)

    for gizmo in obstacle_gizmos:
        @gizmo.on_update
        def _(_):
            resolve()

    for goal_gizmo in goal_gizmos:
        @goal_gizmo.on_update
        def _(_):
            resolve()

    @mode_dropdown.on_update
    def _(_):
        rebuild_and_resolve()

    @num_agents_number.on_update
    def _(_):
        rebuild_and_resolve()

    @num_obstacles_number.on_update
    def _(_):
        rebuild_and_resolve()

    @kind_dropdown.on_update
    def _(_):
        rebuild_and_resolve()

    @agent_radius_slider.on_update
    def _(_):
        rebuild_and_resolve()

    @penalty_slider.on_update
    def _(_):
        rebuild_and_resolve()

    @tracking_log_slider.on_update
    def _(_):
        rebuild_and_resolve()

    @benchmark_button.on_click
    def _(_):
        run_timing_benchmark()

    @sweep_button.on_click
    def _(_):
        run_scaling_sweep()

    rebuild_and_resolve()
    print("Open the printed URL above. Set 'Num agents'/'Num obstacles', drag "
          "goal gizmos (rotate too, in orientation mode) and obstacles, and use "
          "'Run timing benchmark' / 'Run scaling sweep' for real numbers.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()

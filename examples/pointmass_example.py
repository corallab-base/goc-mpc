import argparse
import os
import time
import imageio
import numpy as np
import mujoco as mj
import matplotlib.pyplot as plt

from goc_mpc import (
    GraphOfConstraints,
    GraphOfConstraintsMPC,
    WaypointSolver,
    WaypointObjective,
    MILPWaypointMPC,
    EvolutionaryWaypointSolver,
)
from goc_mpc.splines import Block
from goc_mpc.evolutionary_waypoint_solver import target_eq_constraint
from goc_mpc._ext.configuration_spline import CubicConfigurationSpline

# Deferred: only the interactive meshcat demo needs these, and
# goc_mpc.simple_drake_env pulls in optional asset/rendering dependencies
# (e.g. corallab_assets) that --benchmark mode has no use for.


# ---------------------------------------------------------------------------
# Problem geometry: 3-node sequential task for a single 3-D point mass.
# ---------------------------------------------------------------------------
ORIGIN = np.array([0.0, 0.0, 0.0])
NODE_TARGETS = {
    0: ORIGIN + np.array([0.0, 0.1, 0.0]),
    1: ORIGIN + np.array([0.0, -0.1, 0.0]),
    2: ORIGIN + np.array([0.0, 0.0, 0.1]),
}


def build_graph() -> GraphOfConstraints:
    """Builds the point-mass task graph shared by both waypoint solvers.

    Uses `add_robot_linear_eq` (the per-robot form, with an explicit
    `robot_id`) rather than `add_robots_linear_eq` (the joint-over-all-agents
    form the original example used): only the former registers each phi in
    `phi_to_static_assignment_map`, which `EvolutionaryWaypointSolver` needs
    to resolve which agent owns each node. For this single-agent graph the
    two forms constrain the same thing (A=I over the one agent's dim), so
    this is a no-op change for MILPWaypointMPC.
    """
    robot_spec = [Block.R(3)]
    graph = GraphOfConstraints([robot_spec], [],
                               state_lower_bound=-10.0, state_upper_bound=10.0,
                               robot_names=["point_mass"],
                               object_names=[])

    graph.structure.add_nodes(3)
    graph.structure.add_edge(0, 1, True)
    graph.structure.add_edge(1, 2, True)

    for node, target in NODE_TARGETS.items():
        graph.add_robot_linear_eq(node, 0, np.eye(graph.robot_ambient_dim(0)), target)

    return graph


def build_evolutionary_waypoint_mpc(graph, spline_spec, pop_size=40, n_gen=80,
                                     warm_start=True):
    """Builds an EvolutionaryWaypointSolver configured for `graph`, with the
    same per-node targets as build_graph() (re-supplied here since constraint
    targets live in opaque C++ closures the JAX side can't see -- see
    EvolutionaryWaypointSolver's module docstring).

    If `warm_start`, pays the JIT trace/compile cost immediately via
    `warmup()` against the full remaining_vertices/x0=0 configuration, so the
    first real `solve()` call in the control loop is already warm.
    """
    splines = [CubicConfigurationSpline(spline_spec)]
    waypoint_mpc = EvolutionaryWaypointSolver(
        graph, splines, objective="avg",
        wp_bounds=(-10.0, 10.0), pop_size=pop_size, n_gen=n_gen)

    for node, target in NODE_TARGETS.items():
        waypoint_mpc.add_python_constraint(node, target_eq_constraint(target), kind="eq")

    if warm_start:
        remaining = list(range(graph.structure.num_nodes))
        x0 = np.zeros(graph.num_agents * graph.robot_ambient_dim(0))
        compile_time = waypoint_mpc.warmup(remaining, x0)
        print(f"[evolutionary] warm-up compile time: {compile_time * 1e3:.1f} ms")

    return waypoint_mpc


def pointmass_example_setup(use_evolutionary: bool = False, warm_start: bool = True):
    graph = build_graph()
    spline_spec = [Block.R(3)]

    waypoint_mpc = None
    if use_evolutionary:
        waypoint_mpc = build_evolutionary_waypoint_mpc(graph, spline_spec, warm_start=warm_start)

    # GoC-MPC
    goc_mpc = GraphOfConstraintsMPC(graph, spline_spec,
                                    # for waypoint solver (ignored when waypoint_mpc is set):
                                    waypoint_solver = WaypointSolver.kGurobi,
                                    waypoint_objective = WaypointObjective.kMinMaxL2,
                                    waypoint_enforce_rigidity = False,
                                    waypoint_mpc = waypoint_mpc,
                                    # for timing solver:
                                    time_delta_cutoff = 0.3,
                                    short_path_time_per_step = 0.1,
                                    phi_tolerance = 0.05,
                                    # max_vel/max_acc/max_jerk hard bounds are not supported by
                                    # GraphTimingMPC's current (trust-region SQP) implementation --
                                    # see graph_timing_mpc.hpp's own doc comment.
                                    linear_interpolation = True)

    # goc_mpc.reset()

    return graph, goc_mpc


# ---------------------------------------------------------------------------
# Benchmark: MILP vs. evolutionary waypoint solver, over many iterations.
# ---------------------------------------------------------------------------
def _timing_stats(times_s):
    arr = np.asarray(times_s) * 1e3
    return dict(mean=arr.mean(), median=np.median(arr), std=arr.std(),
                min=arr.min(), max=arr.max())


def _print_row(name, stats):
    print(f"{name:26s}{stats['mean']:>10.2f}{stats['median']:>10.2f}"
          f"{stats['std']:>10.2f}{stats['min']:>10.2f}{stats['max']:>10.2f}")


def benchmark_waypoint_solvers(n_iters=50, pop_size=40, n_gen=80, n_cold_iters=5):
    """Compares the original approach (MILPWaypointMPC, solved via Gurobi)
    against EvolutionaryWaypointSolver (JAX GA + augmented-Lagrangian) on the
    point-mass task graph, solving the identical waypoint problem repeatedly.

    Three configurations are timed:
      - MILP:                 the pre-existing baseline.
      - Evolutionary (cold):  a fresh EvolutionaryWaypointSolver instance per
        call, so every call pays JAX trace/compile from scratch -- this is
        what the solver did before warm-start caching was added.
      - Evolutionary (warm):  one instance, warmed up once via warmup(), then
        solve() reuses the compiled GA for every following call against the
        same (remaining_vertices, x0).
    """
    graph = build_graph()
    spline_spec = [Block.R(3)]
    remaining = list(range(graph.structure.num_nodes))
    x0 = np.zeros(graph.num_agents * graph.robot_ambient_dim(0))

    # --- MILP baseline ---
    milp = MILPWaypointMPC(graph, [CubicConfigurationSpline(spline_spec)],
                            solver=WaypointSolver.kGurobi,
                            objective=WaypointObjective.kMinMaxL2,
                            enforce_rigidity=False)
    milp_times = []
    for _ in range(n_iters):
        if not milp.solve(remaining, x0):
            raise RuntimeError("MILP waypoint solve failed")
        milp_times.append(milp.get_last_solve_time())
    milp_waypoints = milp.view_waypoints().copy()

    # --- Evolutionary, cold every call (no warm-start reuse) ---
    cold_times = []
    for _ in range(n_cold_iters):
        evo_cold = build_evolutionary_waypoint_mpc(graph, spline_spec, pop_size=pop_size,
                                                     n_gen=n_gen, warm_start=False)
        if not evo_cold.solve(remaining, x0):
            raise RuntimeError("Evolutionary waypoint solve failed")
        cold_times.append(evo_cold.get_last_solve_time())
    evo_cold_waypoints = evo_cold.view_waypoints().copy()

    # --- Evolutionary, warmed up once then reused ---
    evo_warm = build_evolutionary_waypoint_mpc(graph, spline_spec, pop_size=pop_size,
                                                n_gen=n_gen, warm_start=True)
    compile_time = evo_warm.get_last_compile_time()
    warm_times = []
    for _ in range(n_iters):
        if not evo_warm.solve(remaining, x0):
            raise RuntimeError("Evolutionary waypoint solve failed")
        assert evo_warm.was_last_solve_warm(), "expected cached GA to be reused"
        warm_times.append(evo_warm.get_last_solve_time())
    evo_warm_waypoints = evo_warm.view_waypoints().copy()

    milp_s = _timing_stats(milp_times)
    cold_s = _timing_stats(cold_times)
    warm_s = _timing_stats(warm_times)

    print("\n" + "=" * 76)
    print(f"Waypoint solver benchmark -- MILP: {n_iters} iters, "
          f"evolutionary cold: {n_cold_iters} iters, evolutionary warm: {n_iters} iters")
    print("=" * 76)
    print(f"{'':26s}{'mean':>10s}{'median':>10s}{'std':>10s}{'min':>10s}{'max':>10s}   (ms)")
    _print_row("MILP (Gurobi)", milp_s)
    _print_row("Evolutionary (cold)", cold_s)
    _print_row("Evolutionary (warm)", warm_s)

    print(f"\nEvolutionary JIT warm-up compile time : {compile_time * 1e3:.1f} ms (paid once)")
    print(f"Cold-call overhead avoided by warm-start : "
          f"{cold_s['mean'] - warm_s['mean']:.1f} ms/call "
          f"({cold_s['mean'] / warm_s['mean']:.1f}x)")
    if warm_s['mean'] < milp_s['mean']:
        print(f"Warm evolutionary vs MILP : {milp_s['mean'] / warm_s['mean']:.1f}x faster")
    else:
        print(f"Warm evolutionary vs MILP : {warm_s['mean'] / milp_s['mean']:.1f}x slower")

    print(f"\nMax |waypoint| discrepancy, MILP vs evolutionary (cold) : "
          f"{np.max(np.abs(milp_waypoints - evo_cold_waypoints)):.4f}")
    print(f"Max |waypoint| discrepancy, MILP vs evolutionary (warm) : "
          f"{np.max(np.abs(milp_waypoints - evo_warm_waypoints)):.4f}")
    print("=" * 76)

    return dict(milp=milp_times, evolutionary_cold=cold_times, evolutionary_warm=warm_times,
                compile_time=compile_time)


def main():
    parser = argparse.ArgumentParser(description="Point-mass GoC-MPC example")
    parser.add_argument("--benchmark", action="store_true",
                         help="Run the MILP-vs-evolutionary waypoint solver "
                              "benchmark instead of the interactive meshcat demo")
    parser.add_argument("--iters", type=int, default=50,
                         help="Number of solve() calls to time in --benchmark mode")
    parser.add_argument("--evolutionary", action="store_true",
                         help="Use EvolutionaryWaypointSolver instead of "
                              "MILPWaypointMPC in the interactive demo")
    args = parser.parse_args()

    if args.benchmark:
        benchmark_waypoint_solvers(n_iters=args.iters)
        return

    from pydrake.geometry import Meshcat
    from goc_mpc.simple_drake_env import SimpleDrakeGym

    meshcat = Meshcat(port=7000)
    env = SimpleDrakeGym(["point_mass"], [], meshcat=meshcat)

    graph, goc_mpc = pointmass_example_setup(use_evolutionary=args.evolutionary)

    input("Continue?")

    while True:
        obs, _ = env.reset()

        breakpoint()

        goc_mpc.reset()

        goc_mpc.unpause()

        t = 0.0
        teleport = False

        step = 1
        for k in range(0, 2000, step):
            x, x_dot = obs

            try:
                xi_h, xi_dot_h, times_h = goc_mpc.step(t, x, x_dot, teleport=teleport)
            except RuntimeError as e:
                # print(e)
                # xi_h, _, _ = goc_mpc.last_cycle_short_path
                # if xi_h.shape[0] > 1:
                #     xi_h = xi_h[1:]
                print("RuntimeError, not overwriting xi_h")
                pass
            except TypeError as e:
                # print(e)
                # xi_h, _, _ = goc_mpc.last_cycle_short_path
                # if xi_h.shape[0] > 1:
                #     xi_h = xi_h[1:]
                print("TypeError, not overwriting xi_h")
                pass


            # get target point on xi_h and associated time
            qpos = xi_h[step]
            time_delta = times_h[step]

            # jump to target point
            obs, rew, done, trunc, info = env.step(qpos, grasp_cmds=goc_mpc.last_grasp_commands)

            # advance time by the associated time delta
            t += time_delta

            if len(goc_mpc.remaining_phases) == 0:
                print("Finished!")
                goc_mpc.pause()
                break

            if teleport:
                time.sleep(1.0)
            else:
                time.sleep(0.1)

        resp = input("Repeat?")
        if resp == 'q':
            break


if __name__ == "__main__":
    main()

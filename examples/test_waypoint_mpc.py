"""
Waypoint-MPC test
=================
Two 3-D point-mass agents travel through a 4-node sequential task.

  Node 0  — start (both agents constrained)
  Node 1  — task A: agent 0 goes to P0, agent 1 goes to P1
  Node 2  — task B: agent 0 goes to P2, agent 1 goes to P3
  Node 3  — home (both agents constrained)

After solving:
  1. Print waypoints and timing solution
  2. Visualise piecewise-linear paths via matplotlib (x-y plane)

Usage:
  python test_waypoint_mpc.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pydrake.math import eq

from goc_mpc import (
    GraphOfConstraints,
    MILPWaypointMPC,
    EvolutionaryWaypointSolver,
    GraphTimingMPC,
    WaypointSolver,
    WaypointObjective,
)
from goc_mpc._ext.configuration_spline import CubicConfigurationSpline, Block

from pydrake.symbolic import logical_and, logical_not


# ---------------------------------------------------------------------------
# Problem geometry
# ---------------------------------------------------------------------------
DIM = 2
N_AGENTS = 2

START_0 = np.array([-1.0,  0.0])
START_1 = np.array([ 1.0,  0.0])

TASK_A_0 = np.array([-1.5,  1.0])   # agent 0 at node 1
TASK_A_1 = np.array([ 1.5,  1.0])   # agent 1 at node 1

TASK_B_0 = np.array([-0.5,  2.0])   # agent 0 at node 2
TASK_B_1 = np.array([ 0.5,  2.0])   # agent 1 at node 2

HOME_0 = np.array([-1.0,  3.0])
HOME_1 = np.array([ 1.0,  3.0])

X0 = np.concatenate([START_0, START_1])
V0 = np.zeros(N_AGENTS * DIM)


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
def build_simple_graph() -> GraphOfConstraints:
    robot_spec = [Block.R(DIM)]
    graph = GraphOfConstraints(
        [robot_spec, robot_spec], [],
        state_lower_bound=-10.0,
        state_upper_bound=10.0,
        robot_names=["agent_0", "agent_1"],
        object_names=[],
    )

    n0, n1, n2, n3 = graph.structure.add_nodes(4)
    graph.structure.add_edge(n0, n1, True)
    graph.structure.add_edge(n1, n2, True)
    graph.structure.add_edge(n2, n3, True)

    q0 = graph.agent_q(0)
    q1 = graph.agent_q(1)

    graph.add_constraint(n0, eq(q0, START_0))
    graph.add_constraint(n0, eq(q1, START_1))

    graph.add_constraint(n1, eq(q0, TASK_A_0))
    graph.add_constraint(n1, eq(q1, TASK_A_1))

    graph.add_constraint(n2, eq(q0, TASK_B_0))
    graph.add_constraint(n2, eq(q1, TASK_B_1))

    graph.add_constraint(n3, eq(q0, HOME_0))
    graph.add_constraint(n3, eq(q1, HOME_1))

    return graph


def build_graph_with_assignments() -> GraphOfConstraints:
    robot_spec = [Block.R(DIM)]
    graph = GraphOfConstraints(
        [robot_spec, robot_spec], [],
        state_lower_bound=-10.0,
        state_upper_bound=10.0,
        robot_names=["agent_0", "agent_1"],
        object_names=[],
    )

    n0, n1, n2, n3, n4, n5 = graph.structure.add_nodes(6)
    graph.structure.add_edge(n0, n1, True)
    graph.structure.add_edge(n2, n3, True)
    graph.structure.add_edge(n4, n5, True)

    # graph.structure.add_edge(n1, n2, True)

    r0 = graph.add_variable()
    r1 = graph.add_variable()
    r2 = graph.add_variable()
    # graph.add_variable_ineq_constraint(r0, r1)
    # graph.add_variable_ineq_constraint(r1, r2)

    # v1 = graph.add_binary_cond_var()
    # r0s, r1s = graph.assignment_sym(r0), graph.assignment_sym(r1)
    # graph.add_edge(n1, n2, logical_and(r0s == r1s, v1 == 1))
    # graph.add_edge(n3, n0, logical_and(r0s == r1s, v1 == 0))

    v0 = graph.add_binary_cond_var()
    r0s, r1s = graph.assignment_sym(r0), graph.assignment_sym(r1)
    graph.add_edge(n1, n2, logical_and(r0s == r1s, v0 == 1))
    graph.add_edge(n3, n0, logical_and(r0s == r1s, logical_not(v0 == 1)))

    v1 = graph.add_binary_cond_var()
    r1s, r2s = graph.assignment_sym(r1), graph.assignment_sym(r2)
    graph.add_edge(n3, n4, logical_and(r1s == r2s, v1 == 1))
    graph.add_edge(n5, n2, logical_and(r1s == r2s, logical_not(v1 == 1)))

    v2 = graph.add_binary_cond_var()
    r0s, r2s = graph.assignment_sym(r0), graph.assignment_sym(r2)
    graph.add_edge(n1, n4, logical_and(r0s == r2s, v2 == 1))
    graph.add_edge(n5, n0, logical_and(r0s == r2s, logical_not(v2 == 1)))

    q0 = graph.var_agent_q(r0)
    q1 = graph.var_agent_q(r1)
    q2 = graph.var_agent_q(r2)

    graph.add_constraint(n0, eq(q0, np.array([-2.0,  1.0])))
    graph.add_constraint(n1, eq(q0, np.array([-2.0,  2.0])))

    graph.add_constraint(n2, eq(q1, np.array([ 0.0,  1.0])))
    graph.add_constraint(n3, eq(q1, np.array([ 0.0,  2.0])))

    graph.add_constraint(n4, eq(q2, np.array([ 2.0,  1.0])))
    graph.add_constraint(n5, eq(q2, np.array([ 2.0,  2.0])))

    return graph


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------
def solve(graph: GraphOfConstraints):
    robot_spec = [Block.R(DIM)]
    splines = [CubicConfigurationSpline(robot_spec) for _ in range(N_AGENTS)]
    remaining = list(range(graph.structure.num_nodes))

    # --- Waypoint MPC ---
    waypoint_mpc = MILPWaypointMPC(
        graph, splines,
        solver=WaypointSolver.kGurobi,
        objective=WaypointObjective.kAvgL2,
        enforce_rigidity=False,
    )

    ok = waypoint_mpc.solve(remaining, X0)
    if not ok:
        raise RuntimeError("Waypoint MPC failed")

    waypoints    = waypoint_mpc.view_waypoints()
    assignments  = waypoint_mpc.view_assignments()
    var_asgn     = waypoint_mpc.view_var_assignments()
    solve_time   = waypoint_mpc.get_last_solve_time()

    print(f"Waypoint MPC  solve time : {solve_time*1e3:.1f} ms")
    print(f"Waypoints ({waypoints.shape}):\n{waypoints}")
    print(f"Assignments  : {assignments}")
    print(f"Var assignments: {var_asgn}")

    # --- Timing MPC (graph traversal / per-agent ordering) ---
    timing_mpc = GraphTimingMPC(
        graph, splines,
        time_cost=1.0, time_cost2=0.0, acceleration_cost=1.0,
    )

    t_by_node = waypoint_mpc.view_t_by_node()
    ok = timing_mpc.solve(X0, V0, remaining, waypoints, var_asgn, t_by_node)
    if not ok:
        raise RuntimeError("Timing MPC failed")

    wps_list         = timing_mpc.view_wps_list()
    agent_nodes_list = timing_mpc.view_agent_nodes_list()
    time_deltas_list = timing_mpc.view_time_deltas_list()

    print(f"\nTiming MPC solve time : {timing_mpc.get_last_solve_time()*1e3:.1f} ms")
    for i, (wps, nodes, tds) in enumerate(zip(wps_list, agent_nodes_list, time_deltas_list)):
        print(f"Agent {i}: nodes={nodes}, time_deltas={np.round(tds, 3)}")
        print(f"  waypoints:\n{wps}")

    return waypoints, wps_list, agent_nodes_list


# ---------------------------------------------------------------------------
# JAX-native evolutionary solver, routed through EvolutionaryWaypointSolver
# instead of MILP. Works against either graph: build_simple_graph() (every
# node's agent targets are literal/fixed -- no add_variable() calls at all)
# or build_graph_with_assignments() (each node owned by one GA-searched
# variable). Node targets are added to the graph via add_constraint() (a real
# pydrake.symbolic Formula) and auto-derived by spec.py from
# graph.phi_to_formula_map -- no manual re-supply needed, unlike before this
# formula was opaque to the JAX side.
# ---------------------------------------------------------------------------
def solve_evolutionary(graph: GraphOfConstraints):
    robot_spec = [Block.R(DIM)]
    splines = [CubicConfigurationSpline(robot_spec) for _ in range(N_AGENTS)]
    remaining = list(range(graph.structure.num_nodes))

    waypoint_mpc = EvolutionaryWaypointSolver(
        graph, splines,
        objective="avg",
        wp_bounds=(-10.0, 10.0),
        pop_size=30, n_gen=60,
    )

    ok = waypoint_mpc.solve(remaining, X0)
    if not ok:
        raise RuntimeError("Evolutionary waypoint solver failed")

    waypoints = waypoint_mpc.view_waypoints()
    assignments = waypoint_mpc.view_assignments()
    var_asgn = waypoint_mpc.view_var_assignments()
    solve_time = waypoint_mpc.get_last_solve_time()

    print(f"\nEvolutionary solver solve time : {solve_time*1e3:.1f} ms")
    print(f"Waypoints ({waypoints.shape}):\n{waypoints}")
    print(f"Assignments  : {assignments}")
    print(f"Var assignments: {var_asgn}")

    return waypoints, assignments, var_asgn


# ---------------------------------------------------------------------------
# Visualise
# ---------------------------------------------------------------------------
def visualise_2d(waypoints: np.ndarray,
                 wps_list: list,
                 agent_nodes_list: list):
    fig, ax = plt.subplots(1, 1, figsize=(13, 6))
    colors = ["tab:blue", "tab:orange"]
    labels = ["Agent 0", "Agent 1"]

    # Build per-agent paths: prepend x0 so the line starts at the initial position.
    agent_paths = []
    for i, (wps, nodes) in enumerate(zip(wps_list, agent_nodes_list)):
        start = X0[i * DIM:(i + 1) * DIM]
        if len(nodes) == 0:
            path = start.reshape(1, -1)
        else:
            path = np.vstack([start, wps])
        agent_paths.append((path, nodes))

    (xi, yi, xlabel, ylabel, title) = (0, 1, "x", "y", "Top view (x-y)")

    for i, (path, nodes) in enumerate(agent_paths):
        ax.plot(path[:, xi], path[:, yi], "-o",
                color=colors[i], linewidth=2, markersize=8,
                label=labels[i], zorder=3)

        # Mark initial position with a square
        ax.scatter(path[0, xi], path[0, yi],
                   marker="s", s=120, color=colors[i],
                   edgecolors="black", linewidths=1.0, zorder=5)

        # Annotate each waypoint with its node id
        for j, node_id in enumerate(nodes):
            ax.annotate(
                f" N{node_id}",
                xy=(path[j + 1, xi], path[j + 1, yi]),
                fontsize=9, color=colors[i],
            )

    # Also mark all waypoints from the raw waypoint matrix as grey crosses
    # n_nodes = waypoints.shape[0]
    # for k in range(n_nodes):
    #     for i in range(N_AGENTS):
    #         wp = waypoints[k, i * DIM:(i + 1) * DIM]
    #         ax.scatter(wp[xi], wp[yi], marker="x", s=40,
    #                    color="gray", alpha=0.5, zorder=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    fig.suptitle("Graph Waypoint MPC — piecewise-linear paths", fontsize=13)
    plt.tight_layout()
    out = "waypoint_mpc_test.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved plot to {out}")
    plt.show()


def visualise_3d(waypoints: np.ndarray,
                 wps_list: list,
                 agent_nodes_list: list):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    colors = ["tab:blue", "tab:orange"]
    labels = ["Agent 0", "Agent 1"]

    # Build per-agent paths: prepend x0 so the line starts at the initial position.
    agent_paths = []
    for i, (wps, nodes) in enumerate(zip(wps_list, agent_nodes_list)):
        start = X0[i * DIM:(i + 1) * DIM]
        path = np.vstack([start, wps])
        agent_paths.append((path, nodes))

    for ax, (xi, yi, xlabel, ylabel, title) in zip(
        axes,
        [(0, 1, "x", "y", "Top view (x-y)"),
         (0, 2, "x", "z", "Side view (x-z)")],
    ):
        for i, (path, nodes) in enumerate(agent_paths):
            ax.plot(path[:, xi], path[:, yi], "-o",
                    color=colors[i], linewidth=2, markersize=8,
                    label=labels[i], zorder=3)

            # Mark initial position with a square
            ax.scatter(path[0, xi], path[0, yi],
                       marker="s", s=120, color=colors[i],
                       edgecolors="black", linewidths=1.0, zorder=5)

            # Annotate each waypoint with its node id
            for j, node_id in enumerate(nodes):
                ax.annotate(
                    f" N{node_id}",
                    xy=(path[j + 1, xi], path[j + 1, yi]),
                    fontsize=9, color=colors[i],
                )

        # Also mark all waypoints from the raw waypoint matrix as grey crosses
        n_nodes = waypoints.shape[0]
        for k in range(n_nodes):
            for i in range(N_AGENTS):
                wp = waypoints[k, i * DIM:(i + 1) * DIM]
                ax.scatter(wp[xi], wp[yi], marker="x", s=40,
                           color="gray", alpha=0.5, zorder=2)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

    fig.suptitle("Graph Waypoint MPC — piecewise-linear paths", fontsize=13)
    plt.tight_layout()
    out = "waypoint_mpc_test.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved plot to {out}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    graph = build_simple_graph()

    waypoints, wps_list, agent_nodes_list = solve(graph)

    evo_waypoints, evo_assignments, evo_var_asgn = solve_evolutionary(graph)
    print("\n--- MILP vs Evolutionary comparison ---")
    print(f"MILP waypoints:\n{waypoints}")
    print(f"Evolutionary waypoints:\n{evo_waypoints}")
    print(f"Evolutionary var assignments: {evo_var_asgn}")

    # Also exercise the GA-searched-assignment path (build_graph_with_
    # assignments() -- each node owned by one add_variable() variable) so
    # both n_variables==0 (above) and n_variables>0 are covered.
    assign_graph = build_graph_with_assignments()
    _, _, assign_var_asgn = solve_evolutionary(assign_graph)
    print(f"\nbuild_graph_with_assignments() var assignments: {assign_var_asgn}")

    visualise_2d(waypoints, wps_list, agent_nodes_list)


if __name__ == "__main__":
    main()

"""
Interactive waypoint/timing demo, comparing GnTimingMPC and SqpTimingMPC
live.

Each of N_AGENTS agents has a chain of M_WAYPOINTS draggable 3-D waypoints
(viser transform-controls gizmos, one color per agent). Dragging any gizmo
-- or changing a GUI control -- rebuilds the graph from scratch, resolves it
(MILPWaypointMPC for the trivial literal-target waypoint assignment, then
whichever timing solver the "Solver" dropdown picks, for the cubic-Hermite
timing/spline), and redraws each agent's resulting spline. Unlike
short_horizon_collision_avoidance.py's reused solver instance + live
obstacle parameter, every waypoint target here is baked into a Drake `eq()`
Formula at `add_constraint()` time, so there's no live-reference shortcut --
a full rebuild-and-resolve is genuinely required on every drag, not just a
convenience choice.

The two solvers (see their own header comments -- gn_timing_mpc.hpp,
sqp_timing_mpc.hpp) solve the IDENTICAL problem (same gn_timing::
ProblemLayout/AssembleObjective) two different ways:
  - GnTimingMPC: one raw-IPOPT solve, given an exact Gauss-Newton Hessian.
  - SqpTimingMPC: a trust-region Gauss-Newton outer loop, qpOASES QP
    subproblems (hot-started).
A stress test across randomized multi-agent scenarios with active timing
constraints found GnTimingMPC failing (IPOPT MAXITER/RESTORATION_FAILURE)
on the large majority of harder cases while SqpTimingMPC kept succeeding
and stayed roughly 20-30x faster -- this dropdown is for seeing that
difference live on whatever scenario you drag together, not just trusting
the aggregate numbers.

Cross-agent timing constraints (LESS_THAN / EQUAL between two agents' own
waypoint arrivals) are expressed the same way every other constraint in this
graph is -- via GraphOfConstraints.add_constraint on graph nodes -- rather
than through any separate solver-level API. GnTimingMPC.solve() only ever
enforces the LESS_THAN/EQUAL interactions GraphOfConstraints::get_agent_paths
derives from the graph itself (see that function, graph_of_constraints.cpp):
  - A node CO-OWNED by >1 agent (i.e. add_constraint'd on that same node for
    more than one agent) automatically becomes an EQUAL interaction between
    those agents -- the same pattern test_waypoint_mpc.py's
    build_simple_graph already uses for its shared nodes, just applied here
    to a dedicated node instead of a node the main path also passes through.
  - An edge between two DISJOINTLY-owned nodes (each owned by exactly one,
    different, agent) automatically becomes a LESS_THAN interaction.
So enabling the "timing constraint" GUI control adds EXTRA, dedicated graph
nodes on top of each agent's own private waypoint chain -- one shared node
for EQUAL, two edge-joined nodes for LESS_THAN -- each constrained via
add_constraint to the CURRENT position of whichever existing draggable
waypoint the GUI has selected (so it tracks that point live, same as
everything else here being rebuilt from the gizmos' current state). These
extra nodes are singly-/co-owned exactly like any other node -- nothing
about GnTimingMPC or get_agent_paths needed to change to support this.

One consequence worth knowing: since the constraint nodes are owned by the
same agent(s) whose waypoint they're pinned to, they become one MORE segment
for that agent to visit -- at a position identical to the waypoint it
mirrors, so the added segment is a near-zero-length hop (floored at the
timing solver's own tau lower bound), not a visible detour.

Run:
    python examples/gn_timing_waypoints.py
Then open the printed URL, drag any waypoint gizmo, and try enabling the
timing constraint to see it change the resulting spline's timing (reported
in the status line as each agent's own cumulative arrival time at the
constrained waypoint).
"""

import time

import numpy as np
import viser
from pydrake.math import eq

from goc_mpc import GraphOfConstraints, MILPWaypointMPC, WaypointSolver, WaypointObjective
from goc_mpc._ext.configuration_spline import CubicConfigurationSpline, Block
from goc_mpc._ext.goc_mpc import GnTimingMPC, SqpTimingMPC

DIM = 3
N_AGENTS = 2
M_WAYPOINTS = 3

AGENT_COLORS = [(30, 160, 220), (220, 120, 30)]

X0 = [np.array([-1.0, 0.0, 1.0]), np.array([1.0, 0.0, 1.0])]
V0 = np.zeros(N_AGENTS * DIM)

INITIAL_WP = [
    [np.array([-1.2, 1.0, 1.0]), np.array([-0.6, 2.0, 1.3]), np.array([-1.0, 3.0, 1.0])],
    [np.array([1.2, 1.0, 1.0]), np.array([0.6, 2.0, 0.7]), np.array([1.0, 3.0, 1.0])],
]

CONSTRAINT_KINDS = ("LESS_THAN (A arrives <= B)", "EQUAL (A arrives == B)")

# SqpTimingMPC first/default: the stress test (this file's own module
# docstring) found it reliably beating GnTimingMPC on harder scenarios, not
# just matching it on easy ones -- so "compare both live" starts from the
# one that's actually expected to work, with IPOPT available to switch to
# for comparison.
SOLVERS = ("SqpTimingMPC (qpOASES)", "GnTimingMPC (IPOPT)")


def main():
    server = viser.ViserServer()

    solver_dropdown = server.gui.add_dropdown("Solver", SOLVERS, initial_value=SOLVERS[0])
    enable_checkbox = server.gui.add_checkbox("Enable timing constraint", initial_value=False)
    agent_a_dropdown = server.gui.add_dropdown(
        "Agent A", tuple(str(i) for i in range(N_AGENTS)), initial_value="0")
    wp_a_dropdown = server.gui.add_dropdown(
        "Agent A waypoint idx", tuple(str(k) for k in range(M_WAYPOINTS)), initial_value="1")
    agent_b_dropdown = server.gui.add_dropdown(
        "Agent B", tuple(str(i) for i in range(N_AGENTS)), initial_value="1")
    wp_b_dropdown = server.gui.add_dropdown(
        "Agent B waypoint idx", tuple(str(k) for k in range(M_WAYPOINTS)), initial_value="0")
    kind_dropdown = server.gui.add_dropdown(
        "Constraint kind", CONSTRAINT_KINDS, initial_value=CONSTRAINT_KINDS[0])
    status_text = server.gui.add_text("Status", initial_value="")

    # Fixed start markers (not draggable -- X0/V0 feed GnTimingMPC.solve()
    # directly, not through the graph, so there's no add_constraint to
    # rebuild for them).
    for i in range(N_AGENTS):
        server.scene.add_icosphere(
            f"/start_marker_{i}", radius=0.05, color=AGENT_COLORS[i], position=tuple(X0[i]))

    # Draggable waypoint gizmos: one transform_controls + a small marker
    # mesh per (agent, waypoint index).
    gizmos = [[None] * M_WAYPOINTS for _ in range(N_AGENTS)]
    for i in range(N_AGENTS):
        for k in range(M_WAYPOINTS):
            gizmo = server.scene.add_transform_controls(
                f"/agent_{i}/wp_{k}", scale=0.25, disable_rotations=True,
                position=tuple(INITIAL_WP[i][k]))
            server.scene.add_icosphere(
                f"/agent_{i}/wp_{k}/marker", radius=0.06, color=AGENT_COLORS[i], opacity=0.85)
            gizmos[i][k] = gizmo

    # Live, reactive per-agent splines.
    path_vis = [
        server.scene.add_spline_catmull_rom(
            f"/spline_{i}", points=np.stack([X0[i], INITIAL_WP[i][0]]),
            color=AGENT_COLORS[i], line_width=4)
        for i in range(N_AGENTS)
    ]

    def resolve():
        wp_positions = [[np.array(gizmos[i][k].position) for k in range(M_WAYPOINTS)]
                         for i in range(N_AGENTS)]

        # Every constraint target is baked into a Drake Formula at
        # add_constraint() time (no live-reference shortcut), so this graph
        # -- and both solves below -- are rebuilt from scratch every call.
        graph = GraphOfConstraints(
            [[Block.R(DIM)]] * N_AGENTS, [],
            state_lower_bound=-10.0, state_upper_bound=10.0,
            robot_names=[f"agent_{i}" for i in range(N_AGENTS)], object_names=[],
        )

        node_ids = graph.structure.add_nodes(N_AGENTS * M_WAYPOINTS)
        nodes = [list(node_ids[i * M_WAYPOINTS:(i + 1) * M_WAYPOINTS]) for i in range(N_AGENTS)]
        for i in range(N_AGENTS):
            qi = graph.agent_q(i)
            for k in range(M_WAYPOINTS):
                graph.add_constraint(nodes[i][k], eq(qi, wp_positions[i][k]))
                if k > 0:
                    graph.structure.add_edge(nodes[i][k - 1], nodes[i][k], True)

        # See this file's own module docstring for why these extra nodes
        # (not a separate solver-level API) are how a manual cross-agent
        # timing constraint gets expressed.
        sync_agents = None
        sync_node_a = sync_node_b = None
        if enable_checkbox.value:
            a, b = int(agent_a_dropdown.value), int(agent_b_dropdown.value)
            ka, kb = int(wp_a_dropdown.value), int(wp_b_dropdown.value)
            if a == b:
                status_text.value = "timing constraint needs two DIFFERENT agents -- ignoring"
            else:
                is_equal = kind_dropdown.value.startswith("EQUAL")
                qa, qb = graph.agent_q(a), graph.agent_q(b)
                if is_equal:
                    (sync_node,) = graph.structure.add_nodes(1)
                    graph.add_constraint(sync_node, eq(qa, wp_positions[a][ka]))
                    graph.add_constraint(sync_node, eq(qb, wp_positions[b][kb]))
                    sync_node_a = sync_node_b = sync_node
                else:
                    sync_node_a, sync_node_b = graph.structure.add_nodes(2)
                    graph.add_constraint(sync_node_a, eq(qa, wp_positions[a][ka]))
                    graph.add_constraint(sync_node_b, eq(qb, wp_positions[b][kb]))
                    graph.structure.add_edge(sync_node_a, sync_node_b, True)
                sync_agents = (a, b, is_equal)

        remaining = list(range(graph.structure.num_nodes))
        x0_flat = np.concatenate(X0)

        wp_splines = [CubicConfigurationSpline([Block.R(DIM)]) for _ in range(N_AGENTS)]
        waypoint_mpc = MILPWaypointMPC(
            graph, wp_splines, solver=WaypointSolver.kGurobi,
            objective=WaypointObjective.kAvgL2, enforce_rigidity=False)
        if not waypoint_mpc.solve(remaining, x0_flat):
            status_text.value = "waypoint MPC FAILED"
            return

        waypoints = waypoint_mpc.view_waypoints()
        assignments = waypoint_mpc.view_assignments()
        t_by_node = waypoint_mpc.view_t_by_node()

        solver_name = solver_dropdown.value
        is_sqp = solver_name.startswith("SqpTimingMPC")
        solver_cls = SqpTimingMPC if is_sqp else GnTimingMPC

        timing_splines = [CubicConfigurationSpline([Block.R(DIM)]) for _ in range(N_AGENTS)]
        timing_mpc = solver_cls(
            graph, timing_splines, time_cost=1.0, time_cost2=0.0, acceleration_cost=1.0)
        if not timing_mpc.solve(x0_flat, V0, remaining, waypoints, assignments, t_by_node):
            status_text.value = f"{solver_name} FAILED"
            return

        fill_splines = [CubicConfigurationSpline([Block.R(DIM)]) for _ in range(N_AGENTS)]
        timing_mpc.fill_cubic_splines(fill_splines, x0_flat, V0)
        for i in range(N_AGENTS):
            sp = fill_splines[i]
            times = np.linspace(sp.begin(), sp.end(), 60)
            pts, _ = sp.eval_multiple(times)
            path_vis[i].points = pts

        msg = f"{solver_name} ok  |  solve_time={timing_mpc.get_last_solve_time() * 1000:.2f} ms"
        if is_sqp:
            msg += (f"  |  iters={timing_mpc.get_last_iterations()}"
                    f"  trust_radius={timing_mpc.get_last_trust_radius():.2g}")
        if sync_agents is not None:
            a, b, is_equal = sync_agents
            time_deltas = timing_mpc.view_time_deltas_list()
            agent_nodes = timing_mpc.view_agent_nodes_list()

            def cumulative_arrival(agent, node):
                idx = agent_nodes[agent].index(node)
                return float(np.sum(time_deltas[agent][:idx + 1]))

            t_a = cumulative_arrival(a, sync_node_a)
            t_b = cumulative_arrival(b, sync_node_b)
            relation = "==" if is_equal else "<="
            holds = (abs(t_a - t_b) < 1e-3) if is_equal else (t_a <= t_b + 1e-3)
            msg += (f"  |  agent {a} @ {t_a:.3f}s {relation} agent {b} @ {t_b:.3f}s"
                    f"  ({'OK' if holds else 'VIOLATED'})")
        status_text.value = msg

    for i in range(N_AGENTS):
        for k in range(M_WAYPOINTS):
            @gizmos[i][k].on_update
            def _(_):
                resolve()

    for control in (solver_dropdown, enable_checkbox, agent_a_dropdown, wp_a_dropdown,
                    agent_b_dropdown, wp_b_dropdown, kind_dropdown):
        @control.on_update
        def _(_):
            resolve()

    resolve()
    print("Open the printed URL above. Drag any waypoint gizmo, switch the "
          "Solver dropdown to compare GnTimingMPC vs SqpTimingMPC live, and "
          "use the timing-constraint GUI controls to add/change a "
          "cross-agent LESS_THAN or EQUAL arrival-time constraint.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()

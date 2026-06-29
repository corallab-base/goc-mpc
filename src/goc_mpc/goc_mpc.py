import pickle
import threading
import time
import numpy as np

from goc_mpc.graphs import Graph

from ._ext.configuration_spline import CubicConfigurationSpline, Block
from ._ext.goc_mpc import (
    GraphOfConstraints,
    WaypointSolver,
    WaypointObjective,
    GraphWaypointMPC,
    GraphTimingMPC,
    GraphShortPathMPC
)


class GraphOfConstraintsMPC():

    def __init__(
            self,
            graph: GraphOfConstraints,
            spline_spec: list[Block],
            # waypoint mpc hyperparameters
            waypoint_solver: WaypointSolver = WaypointSolver.kGurobi,
            waypoint_objective: WaypointObjective = WaypointObjective.kSquaredDistance,
            waypoint_enforce_rigidity: bool = False,
            # timing mpc hyperparameters
            time_cost: float = 1.0,
            time_cost2: float = 0.0,
            acceleration_cost: float = 0.0,
            energy_cost: float = 0.0,
            arclength_cost: float = 1.0,
            time_delta_cutoff: float = 0.4,
            phi_tolerance: float = 0.03,
            max_vel: float = -1.0,
            max_acc: float = -1.0,
            max_jerk: float = -1.0,
            # short path mpc hyperparameters
            short_path_length: int = 10,
            short_path_time_per_step: float = 0.05,
            # misc. options
            solve_for_waypoints_once: bool = False,
            linear_interpolation: bool = False,
    ):
        # problem definition data
        num_agents = graph.num_agents
        dim = graph.dim

        # persistent data
        self.graph = graph
        self.last_cycle_time = 0.0
        self.last_cycle_splines = [CubicConfigurationSpline(spline_spec) for _ in range(num_agents)]
        for s in self.last_cycle_splines:
            s.set_linear(linear_interpolation)
        self.last_cycle_waypoints = None
        self.last_cycle_var_assignments = None
        self.last_cycle_short_path = None
        self.last_cycle_backtracked_phases = set()
        self.last_grasp_commands = []
        self.completed_phases = set()
        self.remaining_phases = list(range(graph.structure.num_nodes))

        # configuration
        self.time_delta_cutoff = time_delta_cutoff
        self.phi_tolerance = phi_tolerance
        self.solve_for_waypoints_once = solve_for_waypoints_once
        self.time_cost = time_cost
        self.time_cost2 = time_cost2
        self.short_path_length = short_path_length
        self.acceleration_cost = acceleration_cost
        self.energy_cost = energy_cost
        self.arclength_cost = arclength_cost
        self.short_path_time_per_step = short_path_time_per_step

        # solvers
        self.waypoint_mpc = GraphWaypointMPC(graph, self.last_cycle_splines,
                                             solver = waypoint_solver,
                                             objective = waypoint_objective,
                                             enforce_rigidity = waypoint_enforce_rigidity)
        self.timing_mpc = GraphTimingMPC(graph, self.last_cycle_splines,
                                         time_cost, time_cost2, acceleration_cost,
                                         energy_cost, arclength_cost,
                                         max_vel, max_acc, max_jerk)
        self.short_path_mpc = GraphShortPathMPC(graph, short_path_length,
                                                num_agents, dim, short_path_time_per_step)

        # double-buffered splines: timing writes to pending, short path reads committed
        self.committed_splines = self.last_cycle_splines
        self.pending_splines = [CubicConfigurationSpline(spline_spec) for _ in range(num_agents)]
        for s in self.pending_splines:
            s.set_linear(linear_interpolation)

        # shared solver inputs (written by step(), read by threads)
        self._latest_x = None
        self._latest_x_dot = None
        self._latest_remaining_phases = list(self.remaining_phases)

        # shared solver outputs (written by background threads)
        self._latest_waypoints = None
        self._latest_assignments = None
        self._latest_var_assignments = None
        self._latest_short_path = None

        # synchronization
        self._phases_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._timing_ready_event = threading.Event()   # set after first timing solve fills splines
        self._first_result_event = threading.Event()

        # start perpetual solver threads
        self._waypoint_thread = threading.Thread(target=self._waypoint_loop, daemon=True)
        self._timing_thread = threading.Thread(target=self._timing_loop, daemon=True)
        self._short_path_thread = threading.Thread(target=self._short_path_loop, daemon=True)
        self._waypoint_thread.start()
        self._timing_thread.start()
        self._short_path_thread.start()

    def _waypoint_loop(self):
        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                time.sleep(0.001)
                continue
            if self._latest_x is None:
                time.sleep(0.001)
                continue
            if self.solve_for_waypoints_once and self._latest_waypoints is not None:
                time.sleep(0.01)
                continue
            x_snap = self._latest_x.copy()
            with self._phases_lock:
                phases_snap = list(self._latest_remaining_phases)
            success = self.waypoint_mpc.solve(phases_snap, x_snap)
            if success:
                self._latest_waypoints = np.array(self.waypoint_mpc.view_waypoints())
                self._latest_assignments = np.array(self.waypoint_mpc.view_assignments())
                self._latest_var_assignments = np.array(self.waypoint_mpc.view_var_assignments())

    def _apply_passed_nodes(self, passed_nodes, x_snap, asgn_snap):
        with self._phases_lock:
            for node in passed_nodes:
                if node in self.graph.unpassable_nodes:
                    continue
                all_phis_satisfied = all(
                    self.graph.evaluate_phi(phi_id, x_snap, asgn_snap, self.phi_tolerance)
                    for phi_id in self.graph.get_phi_ids(node))
                if all_phis_satisfied:
                    print(f"Completed {node}")
                    self.completed_phases |= {node}
                    if node in self.remaining_phases:
                        self.remaining_phases.remove(node)
                    self.last_grasp_commands.extend(
                        self.graph.get_grasp_changes(node, asgn_snap))
                else:
                    print(f"Did not complete {node}")
            self._latest_remaining_phases = list(self.remaining_phases)

    def _timing_loop(self):
        last_t = time.monotonic()
        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                time.sleep(0.001)
                continue
            if self._latest_waypoints is None or self._latest_x is None:
                time.sleep(0.001)
                continue
            now = time.monotonic()
            delta = now - last_t
            last_t = now
            x_snap = self._latest_x.copy()
            x_dot_snap = self._latest_x_dot.copy()
            wps_snap = self._latest_waypoints
            asgn_snap = self._latest_assignments
            with self._phases_lock:
                phases_snap = list(self._latest_remaining_phases)
            if len(phases_snap) > 0 and delta > 0.0:
                passed = self.timing_mpc.set_progressed_time(delta, self.time_delta_cutoff)
                self._apply_passed_nodes(passed, x_snap, asgn_snap)
                with self._phases_lock:
                    phases_snap = list(self._latest_remaining_phases)

            # only solve timing problem when there are remaining phases
            if len(phases_snap) > 0:
                success = self.timing_mpc.solve(x_snap, x_dot_snap, phases_snap, wps_snap, asgn_snap)

            if success:
                self.timing_mpc.fill_cubic_splines(self.pending_splines, x_snap, x_dot_snap)
                self.committed_splines, self.pending_splines = \
                    self.pending_splines, self.committed_splines
                self._timing_ready_event.set()

    def _short_path_loop(self):
        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                time.sleep(0.001)
                continue
            if self._latest_var_assignments is None or self._latest_x is None:
                time.sleep(0.001)
                continue
            if not self._timing_ready_event.is_set():
                time.sleep(0.001)
                continue
            x_snap = self._latest_x.copy()
            x_dot_snap = self._latest_x_dot.copy()
            var_assign_snap = self._latest_var_assignments
            splines_snap = self.committed_splines
            with self._phases_lock:
                phases_snap = list(self._latest_remaining_phases)
            success = self.short_path_mpc.solve(
                x_snap, x_dot_snap, var_assign_snap, phases_snap, splines_snap)
            if success:
                self._latest_short_path = (
                    np.array(self.short_path_mpc.view_points()),
                    np.array(self.short_path_mpc.view_vels()),
                    np.array(self.short_path_mpc.view_times()))
                self._first_result_event.set()

    def pass_node(self, node: int, assignments: np.ndarray):
        with self._phases_lock:
            self.completed_phases |= {node}
            self.remaining_phases.remove(node)
            self.last_grasp_commands.extend(self.graph.get_grasp_changes(node, assignments))
            self._latest_remaining_phases = list(self.remaining_phases)

    def _backtrack(self, x, x_dot):
        self.last_cycle_backtracked_phases = {}

        if len(self.remaining_phases) == 0:
            return

        remaining_phases_changed = True
        while remaining_phases_changed:
            remaining_phases_changed = False
            for edge_phi_id, op in self.graph.get_next_edge_ops(self.remaining_phases).items():
                if not self.graph.evaluate_edge_phi(edge_phi_id, x, self.last_cycle_var_assignments, 0.00):
                    print(f"violated path constraint on {op.u_node}->{op.v_node} (edge phi id: {edge_phi_id})! backtracking.")
                    if edge_phi_id in self.graph.backtrack_map:
                        for node in self.graph.backtrack_map[edge_phi_id]:
                            self.completed_phases -= {node}
                            if node not in self.remaining_phases:
                                self.remaining_phases.append(node)
                            backtracked_agent = self.graph.get_edge_phi_agent(edge_phi_id, self.last_cycle_var_assignments)
                            self.last_cycle_backtracked_phases[backtracked_agent] = op.u_node
                    else:
                        self.completed_phases -= {op.u_node}
                        self.remaining_phases.append(op.u_node)
                        backtracked_agent = self.graph.get_edge_phi_agent(edge_phi_id, self.last_cycle_var_assignments)
                        self.last_cycle_backtracked_phases[backtracked_agent] = op.u_node
                    remaining_phases_changed = True

    def reset(self):
        with self._phases_lock:
            self.last_cycle_time = 0.0
            self.remaining_phases = list(range(self.graph.structure.num_nodes))

            # clear shared solver inputs
            self._latest_x = None
            self._latest_x_dot = None
            self._latest_remaining_phases = list(self.remaining_phases)

            # clear shared solver outputs
            self._latest_waypoints = None
            self._latest_assignments = None
            self._latest_var_assignments = None
            self._latest_short_path = None

            # clear internal buffers of timing MPC to prevent premature node passing
            self.timing_mpc.reset()

            # clear flag indicating computation of first timing and first whole solution
            self._timing_ready_event.clear()
            self._first_result_event.clear()

    def step(self, t, x, x_dot, teleport=False):
        "Returns the short horizon for the controller to execute."

        assert x.size == self.graph.total_dim, f"x.size ({x.size}) != self.graph.total_dim ({self.graph.total_dim})"

        self.last_cycle_time = t
        self.last_grasp_commands = []

        # update shared inputs (all three threads pick these up on next iteration)
        self._latest_x = x
        self._latest_x_dot = x_dot

        # backtrack: fast Python, modifies remaining_phases under lock
        if self._latest_var_assignments is not None:
            with self._phases_lock:
                self._backtrack(x, x_dot)
                self._latest_remaining_phases = list(self.remaining_phases)

        self.last_cycle_var_assignments = self._latest_var_assignments

        if teleport:
            # teleport reads waypoint/timing solver outputs; these may lag by one solve cycle
            wps = self._latest_waypoints
            if wps is None:
                self._first_result_event.wait()
                wps = self._latest_waypoints

            next_agent_states = []
            next_agent_deltas = []

            nodes_and_timings = list(zip(
                self.timing_mpc.view_agent_nodes_list(),
                self.timing_mpc.view_time_deltas_list()
            ))

            for i, (agent_nodes, timings) in enumerate(nodes_and_timings):
                next_agent_node = next(iter(agent_nodes), -1)
                next_agent_delta = next(iter(timings), 0.0)
                if next_agent_node == -1:
                    next_agent_state = x[i*self.graph.dim:(i+1)*self.graph.dim].copy()
                else:
                    next_agent_state = wps[next_agent_node, i*self.graph.dim:(i+1)*self.graph.dim].copy()
                next_agent_state[3:7] /= np.linalg.norm(next_agent_state[3:7])
                next_agent_states.append(next_agent_state)
                next_agent_deltas.append(next_agent_delta)

            next_agent_states = np.expand_dims(np.concatenate(next_agent_states), 0)
            next_agent_states = np.tile(next_agent_states, (self.short_path_length, 1))

            next_agent_times = np.array(max(next_agent_deltas))
            next_agent_times = np.tile(next_agent_times, (self.short_path_length,))

            return next_agent_states, None, next_agent_times

        # block only on the very first call until short path has produced a result
        self._first_result_event.wait()

        return self._latest_short_path

    def stop(self):
        self._stop_event.set()
        self._waypoint_thread.join()
        self._timing_thread.join()
        self._short_path_thread.join()

    def pause(self):
        self._pause_event.set()

    def unpause(self):
        self._pause_event.clear()

    def register_obstacle(self, obs_type, pos, *,
                          radius: float = 0.0,
                          half_sizes=None,
                          agent_id: int = -1,
                          agent_radius: float = 0.05,
                          weight: float = 10.0):
        """Register a static obstacle for the short-path collision cost.

        Args:
            obs_type:     ObsType.SPHERE or ObsType.BOX
            pos:          obstacle center in world frame, shape (3,)
            radius:       sphere radius (SPHERE only)
            half_sizes:   axis-aligned half-extents, shape (3,) (BOX only)
            agent_id:     which agent to apply to (-1 = all agents)
            agent_radius: bounding-sphere radius of the agent (default 0.05 m)
            weight:       cost weight (larger → stronger avoidance)
        """
        from goc_mpc import ObsType  # local import to avoid circular dep
        pos = np.asarray(pos, dtype=float)
        if obs_type == ObsType.SPHERE:
            self.short_path_mpc.add_sphere_obstacle(
                agent_id, pos, agent_radius, float(radius), float(weight))
        elif obs_type == ObsType.BOX:
            hs = np.asarray(half_sizes, dtype=float)
            self.short_path_mpc.add_box_obstacle(
                agent_id, pos, hs, agent_radius, float(weight))
        else:
            raise ValueError(f"Unknown ObsType: {obs_type}")

    def clear_obstacles(self):
        """Remove all registered obstacles from the short-path planner."""
        self.short_path_mpc.clear_obstacles()

    #
    # utils
    #

    def dump(self, f, x, x_dot):
        pickle.dump({
            "x": x,
            "x_dot": x_dot,
            "whole_waypoints": self.waypoint_mpc.view_waypoints(),
            "wps_list": self.timing_mpc.view_wps_list(),
            "vs_list": self.timing_mpc.view_vs_list(),
            "time_deltas_list": self.timing_mpc.view_time_deltas_list(),
            "agent_nodes_list": self.timing_mpc.view_agent_nodes_list(),
            "agent_spline_length_map": self.timing_mpc.view_agent_spline_length_map(),
        }, f)

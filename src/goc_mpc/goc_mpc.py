import pickle
import numpy as np

from goc_mpc.graphs import Graph

from ._ext.splines import CubicSpline
from ._ext.configuration_spline import CubicConfigurationSpline, Block
from ._ext.goc_mpc import (
    GraphOfConstraints,
    GraphWaypointMPC,
    GraphTimingMPC,
    GraphShortPathMPC
)


class GraphOfConstraintsMPC():

    def __init__(
            self,
            graph: GraphOfConstraints,
            spline_spec: list[Block],
            time_cost: float = 1.0,
            time_cost2: float = 0.0,
            acceleration_cost: float = 0.0,
            energy_cost: float = 0.0,
            arclength_cost: float = 1.0,
            time_delta_cutoff: float = 0.4,
            phi_tolerance: float = 0.03,
            short_path_length: int = 10,
            short_path_time_per_step: float = 0.05,
            solve_for_waypoints_once: bool = False,
            max_vel: float = -1.0,
            max_acc: float = -1.0,
            max_jerk: float = -1.0,
    ):
        # problem definition data
        num_agents = graph.num_agents
        dim = graph.dim

        # persistent data
        self.graph = graph
        self.last_cycle_time = 0.0
        self.last_cycle_splines = [CubicConfigurationSpline(spline_spec) for _ in range(num_agents)]
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
        self.waypoint_mpc = GraphWaypointMPC(graph, self.last_cycle_splines)
        self.timing_mpc = GraphTimingMPC(graph, self.last_cycle_splines, time_cost, time_cost2,
                                         acceleration_cost, energy_cost, arclength_cost, max_vel, max_acc, max_jerk)
        self.short_path_mpc = GraphShortPathMPC(graph, short_path_length, num_agents, dim, short_path_time_per_step)

    def _solve_for_waypoints(self, x: np.ndarray):
        if (self.solve_for_waypoints_once and self.last_cycle_waypoints is not None):
            return True
        else:
            success = self.waypoint_mpc.solve(self.remaining_phases, x)
            self.last_cycle_waypoints = self.waypoint_mpc.view_waypoints()
            return success

    def pass_node(self, node: int, assignments: np.ndarray):
        self.completed_phases |= {node}
        self.remaining_phases.remove(node)
        self.last_grasp_commands.extend(self.graph.get_grasp_changes(node, assignments))

    def _solve_for_timing(self, time_delta, x, x_dot):

        # get references to the stored waypoints and assignments solutions from waypoint_mpc
        waypoints = self.waypoint_mpc.view_waypoints()
        assignments = self.waypoint_mpc.view_assignments()
        var_assignments = self.waypoint_mpc.view_var_assignments()
        self.last_cycle_var_assignments = var_assignments

        # PROGRESSION: progress time and potentially change phase
        # shift timing
        if len(self.remaining_phases) > 0 and time_delta > 0.0:
            passed_nodes = self.timing_mpc.set_progressed_time(time_delta, self.time_delta_cutoff)

            for node in passed_nodes:
                if node in self.graph.unpassable_nodes:
                    continue

                all_phis_satisfied = all(
                    [self.graph.evaluate_phi(phi_id, x, assignments, self.phi_tolerance)
                     for phi_id in self.graph.get_phi_ids(node)])

                if all_phis_satisfied:
                    print(f"Completed {node}")
                    # breakpoint()
                    self.completed_phases |= {node}
                    self.remaining_phases.remove(node)
                    self.last_grasp_commands.extend(self.graph.get_grasp_changes(node, assignments))
                else:
                    print(f"Did not complete {node}")

        # if not self.timing_mpc.done():
        #     # if the closest next phase is further than time_delta_cutoff seconds into the future
        #     if self.timing_mpc.current_minimum_time_delta() > self.time_delta_cutoff:
        #         # resolve the timing problem
        #         # TODO: understand if there is something to do with ctrlErr

        success = self.timing_mpc.solve(x, x_dot, self.remaining_phases, waypoints, assignments)
        if success:
            self.timing_mpc.fill_cubic_splines(self.last_cycle_splines, x, x_dot)
            return True
        else:
            return False

    def _solve_for_short_path(self, x, x_dot):
        var_assignments = self.waypoint_mpc.view_var_assignments()

        success = self.short_path_mpc.solve(x, x_dot,
                                            var_assignments,
                                            self.remaining_phases,
                                            self.last_cycle_splines)

        if success:
            points = self.short_path_mpc.view_points()
            vels = self.short_path_mpc.view_vels()
            times = self.short_path_mpc.view_times()
            self.last_cycle_short_path = (points, vels, times)

        return success

    def _backtrack(self, x, x_dot):
        self.last_cycle_backtracked_phases = {}

        # BACKTRACKING: if the task has been finished
        if len(self.remaining_phases) == 0:
            # TODO: support final edge phis
            pass
        else:
            remaining_phases_changed = True

            # otherwise,
            while remaining_phases_changed:
                remaining_phases_changed = False

                for edge_phi_id, op in self.graph.get_next_edge_ops(self.remaining_phases).items():
                    if not self.graph.evaluate_edge_phi(edge_phi_id, x, self.last_cycle_var_assignments, 0.00):
                        print(f"violated path constraint on {op.u_node}->{op.v_node} (edge phi id: {edge_phi_id})! backtracking.")

                        if edge_phi_id in self.graph.backtrack_map:
                            for node in self.graph.backtrack_map[edge_phi_id]:
                                child_nodes = self.graph.structure.dfs(node)
                                self.completed_phases -= set(child_nodes)
                                self.remaining_phases = list(set(self.remaining_phases) | set(child_nodes))
                                # TODO: This is meant to open the gripper for
                                # the right agent when backtracking. Replace it
                                # with edge constraint for gripper preceeding actions
                                backtracked_agent = self.graph.get_edge_phi_agent(edge_phi_id, self.last_cycle_var_assignments)
                                self.last_cycle_backtracked_phases[backtracked_agent] = op.u_node
                        else:
                            self.completed_phases -= {op.u_node}
                            self.remaining_phases.append(op.u_node)

                            backtracked_agent = self.graph.get_edge_phi_agent(edge_phi_id, self.last_cycle_var_assignments)
                            self.last_cycle_backtracked_phases[backtracked_agent] = op.u_node

                        remaining_phases_changed = True

            # while not self.timing_mpc.at_the_start() and phi.maxError(C, 0.5+timingMPC.phase+subSeqStart) > opt.precision:
            #     # back track appropriately
            #     self.timing_mpc.update_backtrack();
            #     phase_changed = True

    def reset(self):
        self.last_cycle_time = 0.0
        self.remaining_phases = list(range(self.graph.structure.num_nodes))

    def step(self, t, x, x_dot, teleport=False):
        "Returns the short horizon for the controller to execute."

        assert x.size == self.graph.total_dim

        delta = t - self.last_cycle_time
        self.last_cycle_time = t

        self.last_grasp_commands = []

        if self.last_cycle_var_assignments is not None:
            self._backtrack(x, x_dot)

        success = self._solve_for_waypoints(x)

        if not success:
            raise RuntimeError("WaypointsMPC Failed!")

        success = self._solve_for_timing(delta, x, x_dot)

        if teleport:
            wps = self.waypoint_mpc.view_waypoints()
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

        if not success:
            raise RuntimeError("TimingMPC Failed!")

        success = self._solve_for_short_path(x, x_dot)

        if not success:
            raise RuntimeError("ShortPathMPC Failed!")

        # tuple:
        # points: n by d_pos
        # vels: n by d_vel
        # times: n
        return self.last_cycle_short_path

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

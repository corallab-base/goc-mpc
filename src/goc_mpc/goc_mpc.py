import numpy as np

from goc_mpc.graphs import Graph

from ._ext.splines import CubicSpline
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
            time_delta_cutoff: float = 0.5,
            short_path_length: int = 10,
            short_path_time_per_step: float = 0.05,
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
        self.last_cycle_splines = [CubicSpline()] * num_agents
        self.last_cycle_waypoints = None
        self.last_cycle_short_path = None
        self.completed_phases = set()
        self.remaining_phases = list(range(graph.structure.num_nodes))
        
        # configuration
        self.time_delta_cutoff = time_delta_cutoff
        self.time_cost = 1.0
        self.ctrl_cost = 1.0

        # solvers
        self.waypoint_mpc = GraphWaypointMPC(graph)
        self.timing_mpc = GraphTimingMPC(graph, 1.0, 1.0, max_vel, max_acc, max_jerk)
        self.short_path_mpc = GraphShortPathMPC(short_path_length, num_agents, dim, short_path_time_per_step)

    def _solve_for_waypoints(self, x: np.ndarray):
        success = self.waypoint_mpc.solve(self.remaining_phases, x)
        return success

    def _solve_for_timing(self, time_delta, x, x_dot):

        # get references to the stored waypoints and assignments solutions from waypoint_mpc
        waypoints = self.waypoint_mpc.view_waypoints()
        assignments = self.waypoint_mpc.view_assignments()

        # PROGRESSION: progress time and potentially change phase
        # shift timing
        if len(self.remaining_phases) > 0 and time_delta > 0.0:
            passed_nodes = self.timing_mpc.set_progressed_time(time_delta, self.time_delta_cutoff)

            for node in passed_nodes:
                all_phis_satisfied = all(
                    [self.graph.evaluate_phi(phi_id, x, assignments[phi_id], 0.2)
                     for phi_id in self.graph.get_phi_ids(node)])

                if all_phis_satisfied:
                    self.completed_phases |= {node}
                    self.remaining_phases.remove(node)
                    breakpoint()


        # BACKTRACKING: if the task has been finished
        # if len(self.remaining_phases) == 0:
        #     # and if any of the final constraint are violated
        #     if phi.maxError(C, timingMPC.phase+subSeqStart) > opt.precision):
        #         # back track appropriately
        #         self.timing_mpc.update_backtrack();
        #         phase_changed = True
        # else:
        #     # otherwise,
        #     while not self.timing_mpc.at_the_start() and phi.maxError(C, 0.5+timingMPC.phase+subSeqStart) > opt.precision:
        #         back track  appropriately
        #         self.timing_mpc.update_backtrack();
        #         phase_changed = True

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
        success = self.short_path_mpc.solve(x, x_dot, self.last_cycle_splines)

        if success:
            points = self.short_path_mpc.view_points()
            times = self.short_path_mpc.view_times()
            self.last_cycle_short_path = (points, times)

        return success

    def step(self, t, x, x_dot):
        "Returns the short horizon for the controller to execute."

        delta = t - self.last_cycle_time
        self.last_cycle_time = t

        success = self._solve_for_waypoints(x)
        success = self._solve_for_timing(delta, x, x_dot)
        success = self._solve_for_short_path(x, x_dot)
        
        return self.last_cycle_short_path

    #
    # task definition helpers
    #

    def add_linear_eq(self, node: int, A: np.ndarray, b: np.ndarray):
        return self.waypoint_mpc.add_linear_eq(node, A, b)

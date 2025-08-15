import numpy as np

from goc_mpc.graphs import Graph

from ._goc_mpc_cpp.splines import CubicSpline
from ._goc_mpc_cpp.goc_mpc import (
    GraphOfConstraints,
    GraphWaypointMPC,
    GraphTimingMPC,
    GraphShortPathMPC
)


class GraphOfConstraintsMPC():

    def __init__(
            self,
            graph: GraphOfConstraints,
            time_delta_cutoff: float = 0.5
    ):
        # problem definition data
        state_lower_bound = np.ones(dim) * -10.0
        state_upper_bound = np.ones(dim) * 10.0
        short_path_length = 10
        short_path_time_per_step = 0.1

        # persistent data
        self.main_graph = main_graph
        self.last_cycle_time = 0.0
        self.last_cycle_spline = CubicSpline()
        self.last_cycle_waypoints = None
        self.last_cycle_short_path = None
        self.completed_phases = set()
        self.remaining_phases = list(range(main_graph.num_nodes()))
        
        # configuration
        self.time_delta_cutoff = time_delta_cutoff
        self.time_cost = 1.0
        self.ctrl_cost = 1.0

        # solvers
        self.waypoint_mpc = GraphWaypointMPC(main_graph, num_agents, dim, state_lower_bound, state_upper_bound)
        self.timing_mpc = GraphTimingMPC(num_agents, dim, 1.0, 1.0)
        self.short_path_mpc = GraphShortPathMPC(short_path_length, dim, short_path_time_per_step)
        
    def _solve_for_waypoints(self, x: np.ndarray):
        waypoints, assignments = self.waypoint_mpc.solve(self.remaining_phases, x)
        return waypoints, assignments

    def _current_graph(self, assignments: np.ndarray) -> np.ndarray:
        # TODO: Implement support for conditional edges
        return self.main_graph

    def _solve_for_timing(self, time_delta, x, x_dot, waypoints, assignments):

        # PROGRESSION: progress time and potentially change phase
        # if not self.timing_mpc.done() and time_delta > 0.0:
        #     passed_waypoints = self.timing_mpc.set_progressed_time(time_delta, self.time_delta_cutoff);
        # else:
        #     passed_waypoints = set()

        # BACKTRACKING: if the task has been finished
        # if self.timing_mpc.done():
        # and if any of the final constraint are violated
        # if phi.maxError(C, timingMPC.phase+subSeqStart) > opt.precision):
        #     back track appropriately
        #     self.timing_mpc.update_backtrack();
        #     phase_changed = True
        #     pass
        # else:
        # otherwise,
        # while not self.timing_mpc.at_the_start() and phi.maxError(C, 0.5+timingMPC.phase+subSeqStart) > opt.precision:
        #     back track  appropriately
        #     self.timing_mpc.update_backtrack();
        #     phase_changed = True
        # pass

        # if not self.timing_mpc.done():
        #     # if the closest next phase is further than time_delta_cutoff seconds into the future
        #     if self.timing_mpc.current_minimum_time_delta() > self.time_delta_cutoff:
        #         # resolve the timing problem
        #         # TODO: understand if there is something to do with ctrlErr

        graph = self._current_graph(assignments)
        self.timing_mpc.solve(graph, x, x_dot, waypoints, assignments)
        self.timing_mpc.fill_cubic_spline(self.last_cycle_spline, x, x_dot)

    def _solve_for_short_path(self, x, x_dot):
        self.short_path_mpc.solve(x, x_dot, self.last_cycle_spline)
        ps = self.short_path_mpc.get_points()
        ts = self.short_path_mpc.get_times()
        return (ps, ts)

    def step(self, t, x, x_dot):
        "Returns the short horizon for the controller to execute."

        delta = t - self.last_cycle_time
        self.last_cycle_time = t

        waypoints, assignments = self._solve_for_waypoints(x)
        self._solve_for_timing(delta, x, x_dot, waypoints, assignments)
        xi_h, ts = self._solve_for_short_path(x, x_dot)

        # update state
        self.last_cycle_waypoints = waypoints
        self.last_cycle_short_path = (xi_h)

        return xi_h, ts

    #
    # task definition helpers
    #

    def add_linear_eq(self, node: int, A: np.ndarray, b: np.ndarray):
        return self.waypoint_mpc.add_linear_eq(node, A, b)

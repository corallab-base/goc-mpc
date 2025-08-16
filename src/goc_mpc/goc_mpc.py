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
            time_delta_cutoff: float = 0.5
    ):
        # problem definition data
        num_agents = graph.num_agents
        dim = graph.dim
        short_path_length = 10
        short_path_time_per_step = 0.1

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
        self.timing_mpc = GraphTimingMPC(graph, 1.0, 1.0)
        # self.short_path_mpc = GraphShortPathMPC(short_path_length, dim, short_path_time_per_step)

    def _solve_for_waypoints(self, x: np.ndarray):
        success = self.waypoint_mpc.solve(self.remaining_phases, x)
        return success

    def _solve_for_timing(self, time_delta, x, x_dot):

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

        # get references to the stored waypoints and assignments solutions from waypoint_mpc
        waypoints = self.waypoint_mpc.view_waypoints()
        assignments = self.waypoint_mpc.view_assignments()

        breakpoint()
        
        success = self.timing_mpc.solve(x, x_dot, self.remaining_phases, waypoints, assignments)
        if success:
            self.timing_mpc.fill_cubic_splines(self.last_cycle_splines, x, x_dot)
            return True
        else:
            return False

    # def _solve_for_short_path(self, x, x_dot):
        # self.short_path_mpc.solve(x, x_dot, self.last_cycle_spline)
        # ps = self.short_path_mpc.get_points()
        # ts = self.short_path_mpc.get_times()
        # return (ps, ts)

    def step(self, t, x, x_dot):
        "Returns the short horizon for the controller to execute."

        delta = t - self.last_cycle_time
        self.last_cycle_time = t

        success = self._solve_for_waypoints(x)
        success = self._solve_for_timing(delta, x, x_dot)
        # xi_h, ts = self._solve_for_short_path(x, x_dot)

        # # update state
        # self.last_cycle_waypoints = waypoints
        # self.last_cycle_short_path = (xi_h)

        # return xi_h, ts
        return None, None

    #
    # task definition helpers
    #

    def add_linear_eq(self, node: int, A: np.ndarray, b: np.ndarray):
        return self.waypoint_mpc.add_linear_eq(node, A, b)

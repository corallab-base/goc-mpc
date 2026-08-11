import pickle
import numpy as np

from goc_mpc.graphs import Graph

from ._ext.configuration_spline import CubicConfigurationSpline, Block
from ._ext.goc_mpc import (
    GraphOfConstraints,
    WaypointSolver,
    WaypointObjective,
    GraphWaypointMPC,
    MILPWaypointMPC,
    GraphTimingMPC,
    GraphShortPathMPC,
    EdgeCostFunctor,
    RegularGridInterpolant,
)


class GraphOfConstraintsMPC():

    def __init__(
            self,
            graph: GraphOfConstraints,
            spline_spec: list[Block],
            # waypoint mpc hyperparameters
            waypoint_solver: WaypointSolver = WaypointSolver.kGurobi,
            waypoint_objective: WaypointObjective = WaypointObjective.kMinMaxL1,
            waypoint_enforce_rigidity: bool = False,
            edge_cost_fn: EdgeCostFunctor | None = None,
            # pass an already-constructed GraphWaypointMPC (e.g. an
            # EvolutionaryWaypointSolver, which is duck-typed to this
            # protocol rather than a real C++ subclass) to use it directly;
            # waypoint_solver/waypoint_objective/
            # waypoint_enforce_rigidity/edge_cost_fn are then ignored, since the
            # instance is already configured. Leave unset (default) to keep
            # auto-building a MILPWaypointMPC from those args, as before.
            waypoint_mpc: GraphWaypointMPC | None = None,
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
            # Convex alternative/complement to acceleration_cost: penalizes
            # ||xJ-xJm1||^2/tau^3 + ||vJ-vJm1||^2/tau per segment instead of
            # acceleration_cost's coast-corrected ||(xJ-xJm1) -
            # 0.5*tau*(vJm1+vJ)||^2/tau^3 -- the latter's cross term can be
            # non-convex whenever the current velocity already points
            # roughly toward the target, which is what let NLopt's timing
            # solve land in two different local optima cycle to cycle (see
            # graph_timing_mpc.hpp's _stability_cost doc comment).
            stability_cost: float = 0.0,
            # short path mpc hyperparameters
            short_path_length: int = 10,
            short_path_time_per_step: float = 0.05,
            # misc. options
            solve_for_waypoints_once: bool = False,
            linear_interpolation: bool = False,
            # Runtime drift check for add_hold/add_assignable_hold spans (see
            # _hold_violated): how far (per-axis, same units as x) a held
            # point may stray from where the holding robot's *current*
            # end-effector pose (graph.link_pose's forward kinematics)
            # predicts it should be, given the nominal end-effector -> point
            # offset captured when the hold was established
            # (_maybe_start_holds) -- before _backtrack treats the hold as
            # broken (e.g. a real grasp slip) and reopens the hold's u_node.
            # This is a coarse "did it fall out of the hand" sanity check,
            # not a rigidity tolerance -- plans no longer need to hand-roll
            # their own proximity edge constraint (formerly add_holding_box/
            # add_robot_holding_cube_constraint + add_manual_backtrack_links)
            # just to get this.
            hold_drift_tolerance: float = 0.3,
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
        # Nominal end-effector -> held-point transform for each currently
        # active hold (see _maybe_start_holds/_hold_violated), keyed by
        # hold_id. Populated the instant a hold's u_node completes; dropped
        # once its v_node completes or its u_node gets reopened (backtrack).
        self._hold_nominal_offsets = {}

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
        self.stability_cost = stability_cost
        self.short_path_time_per_step = short_path_time_per_step
        self.hold_drift_tolerance = hold_drift_tolerance

        # solvers
        if waypoint_mpc is not None:
            # caller is responsible for having constructed waypoint_mpc with
            # splines built from the same spline_spec/agent count as this
            # instance (each C++ solver keeps its own copy of the splines
            # passed at construction, same as the auto-built path below).
            self.waypoint_mpc = waypoint_mpc
        else:
            waypoint_mpc_kwargs = {}
            if edge_cost_fn is not None:
                waypoint_mpc_kwargs["edge_cost_fn"] = edge_cost_fn
            self.waypoint_mpc = MILPWaypointMPC(graph, self.last_cycle_splines,
                                                solver = waypoint_solver,
                                                objective = waypoint_objective,
                                                enforce_rigidity = waypoint_enforce_rigidity,
                                                **waypoint_mpc_kwargs)
        self.timing_mpc = GraphTimingMPC(graph, self.last_cycle_splines,
                                         time_cost, time_cost2, acceleration_cost,
                                         energy_cost, arclength_cost,
                                         max_vel, max_acc, max_jerk, stability_cost)
        self.short_path_mpc = GraphShortPathMPC(graph, short_path_length,
                                                num_agents, dim, short_path_time_per_step)

    def _solve_for_waypoints(self, x: np.ndarray):
        if (self.solve_for_waypoints_once and self.last_cycle_waypoints is not None):
            return True
        else:
            success = self.waypoint_mpc.solve(self.remaining_phases, x)
            self.last_cycle_waypoints = self.waypoint_mpc.view_waypoints()
            return success

    def pass_node(self, node: int, assignments: np.ndarray):
        print(f"Completed {self.graph.get_node_name(node)}")
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
                    print(f"Completed {self.graph.get_node_name(node)}")
                    # breakpoint()
                    self.completed_phases |= {node}
                    self.remaining_phases.remove(node)
                    self.last_grasp_commands.extend(self.graph.get_grasp_changes(node, assignments))
                    self._maybe_commit(node, var_assignments)
                    self._maybe_start_holds(node, x)
                    self._maybe_end_holds(node)
                else:
                    print(f"Did not complete {self.graph.get_node_name(node)}")

        # if not self.timing_mpc.done():
        #     # if the closest next phase is further than time_delta_cutoff seconds into the future
        #     if self.timing_mpc.current_minimum_time_delta() > self.time_delta_cutoff:
        #         # resolve the timing problem
        #         # TODO: understand if there is something to do with ctrlErr

        if len(self.remaining_phases) > 0:
            t_by_node = self.waypoint_mpc.view_t_by_node()
            success = self.timing_mpc.solve(x, x_dot, self.remaining_phases, waypoints, assignments, t_by_node)
            if success:
                self.timing_mpc.fill_cubic_splines(self.last_cycle_splines, x, x_dot)
                return True
            else:
                return False
        else:
            return True


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

    def _maybe_commit(self, node, var_assignments) -> None:
        # If `node` is a registered commit trigger (add_variable_commit, or
        # auto-registered by add_assignable_hold at its pick-up node), pin
        # the variable to whatever agent it just resolved to -- the routing
        # solve can no longer reassign it on subsequent cycles (see
        # Constraint 8b in milp_waypoint_mpc.cpp) until _maybe_clear_commit
        # reopens this same node via backtracking.
        var = self.graph.get_commit_trigger_var(node)
        if var is not None:
            self.graph.commit_variable_assignment(var, int(var_assignments[var]))

    def _maybe_clear_commit(self, node) -> None:
        # Symmetric undo for _maybe_commit: reopening a commit-trigger node
        # (backtracking past a broken grasp/placement) un-pins its variable
        # so the next MILP solve is free to resolve it fresh -- possibly to
        # a different agent -- rather than staying stuck on the agent that
        # just failed.
        var = self.graph.get_commit_trigger_var(node)
        if var is not None:
            self.graph.clear_variable_commitment(var)

    def _hold_agent(self, hold) -> int:
        if hold.robot_ag is not None:
            return hold.robot_ag
        return int(self.last_cycle_var_assignments[hold.var_id])

    def _maybe_start_holds(self, node, x) -> None:
        # Any hold (add_hold/add_assignable_hold) whose pick-up (u_node) is
        # `node` becomes active now -- capture the nominal end-effector pose
        # (via graph.link_pose's forward kinematics -- a Python-registered
        # override if the holding robot has one via graph.set_robot_fk,
        # otherwise the built-in per-robot-kind dispatch; never a raw
        # workspace-point read of x) and, from it, each held point's offset
        # expressed in the end-effector's own frame at this instant. This is
        # what _hold_violated later measures drift against, instead of
        # comparing raw positions against a fixed absolute tolerance -- see
        # hold_drift_tolerance's doc comment on __init__.
        for hold_id, hold in self.graph.hold_ops.items():
            if hold.u_node != node:
                continue
            agent_id = self._hold_agent(hold)
            p_we, R_we = self.graph.link_pose(agent_id, x)
            self._hold_nominal_offsets[hold_id] = {
                point_id: R_we.T @ (self.graph.point_position(point_id, x) - p_we)
                for point_id in hold.held_point_ids
            }

    def _maybe_end_holds(self, node) -> None:
        # Symmetric undo for _maybe_start_holds: once a hold's release
        # (v_node) completes, its nominal offset is no longer meaningful.
        for hold_id, hold in self.graph.hold_ops.items():
            if hold.v_node == node:
                self._hold_nominal_offsets.pop(hold_id, None)

    def _maybe_clear_holds(self, node) -> None:
        # Reopening a hold's pick-up node via backtracking invalidates
        # whatever nominal offset was captured there -- the next time this
        # node completes (possibly with a different agent/grasp), a fresh
        # one must be captured via _maybe_start_holds.
        for hold_id, hold in self.graph.hold_ops.items():
            if hold.u_node == node:
                self._hold_nominal_offsets.pop(hold_id, None)

    def _hold_violated(self, hold_id, hold, x) -> bool:
        # Coarse "did it fall out of the hand" sanity check (not a rigidity
        # tolerance): has each held point drifted more than
        # hold_drift_tolerance (per axis) from where it should be given the
        # holding robot's *current* end-effector pose and the nominal
        # offset captured when the hold was established (_maybe_start_holds)?
        # See hold_drift_tolerance's doc comment on __init__.
        offsets = self._hold_nominal_offsets.get(hold_id)
        if offsets is None:
            # Hold just became current this cycle, before its u_node's
            # completion handler ran (or a fresh offset hasn't been
            # captured yet after a backtrack) -- nothing to compare against.
            return False
        agent_id = self._hold_agent(hold)
        p_we, R_we = self.graph.link_pose(agent_id, x)
        for point_id in hold.held_point_ids:
            predicted = p_we + R_we @ offsets[point_id]
            actual = self.graph.point_position(point_id, x)
            if np.any(np.abs(actual - predicted) > self.hold_drift_tolerance):
                return True
        return False

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

                for (u_node, v_node), edge_phi_id in self.graph.get_next_edge_phis(self.remaining_phases).items():
                    if not self.graph.evaluate_edge_phi(edge_phi_id, x, self.last_cycle_var_assignments, 0.00):
                        print(f"Violated path constraint on {self.graph.get_node_name(u_node)}->{self.graph.get_node_name(v_node)} (edge phi id: {edge_phi_id})! backtracking.")

                        if edge_phi_id in self.graph.backtrack_map:
                            for node in self.graph.backtrack_map[edge_phi_id]:
                                self.completed_phases -= {node}
                                if node not in self.remaining_phases:
                                    self.remaining_phases.append(node)
                                self._maybe_clear_commit(node)
                                self._maybe_clear_holds(node)
                                # TODO: This is meant to open the gripper for
                                # the right agent when backtracking. Replace it
                                # with edge constraint for gripper preceeding actions
                                backtracked_agent = self.graph.get_edge_phi_agent(edge_phi_id, self.last_cycle_var_assignments)
                                self.last_cycle_backtracked_phases[backtracked_agent] = u_node
                        else:
                            self.completed_phases -= {u_node}
                            self.remaining_phases.append(u_node)
                            self._maybe_clear_commit(u_node)
                            self._maybe_clear_holds(u_node)

                            backtracked_agent = self.graph.get_edge_phi_agent(edge_phi_id, self.last_cycle_var_assignments)
                            self.last_cycle_backtracked_phases[backtracked_agent] = u_node

                        remaining_phases_changed = True

                # Holds (add_hold/add_assignable_hold) currently in progress:
                # plans no longer need to hand-roll a proximity edge
                # constraint (formerly add_holding_box + add_manual_
                # backtrack_links) just to detect a dropped/slipped grasp --
                # backtrack straight to the hold's own u_node instead.
                for hold_id, hold in self.graph.get_current_holds(self.remaining_phases).items():
                    if self._hold_violated(hold_id, hold, x):
                        print(f"Violated hold on {self.graph.get_node_name(hold.u_node)}->{self.graph.get_node_name(hold.v_node)} (hold id: {hold_id})! backtracking.")

                        self.completed_phases -= {hold.u_node}
                        if hold.u_node not in self.remaining_phases:
                            self.remaining_phases.append(hold.u_node)
                        self._maybe_clear_commit(hold.u_node)
                        self._maybe_clear_holds(hold.u_node)

                        self.last_cycle_backtracked_phases[self._hold_agent(hold)] = hold.u_node
                        remaining_phases_changed = True

            # while not self.timing_mpc.at_the_start() and phi.maxError(C, 0.5+timingMPC.phase+subSeqStart) > opt.precision:
            #     # back track appropriately
            #     self.timing_mpc.update_backtrack();
            #     phase_changed = True

    def reset(self):
        self.last_cycle_time = 0.0
        self.remaining_phases = list(range(self.graph.structure.num_nodes))
        self._hold_nominal_offsets = {}

    def step(self, t, x, x_dot, teleport=False):
        "Returns the short horizon for the controller to execute."

        assert x.size == self.graph.total_dim, f"x.size ({x.size}) != self.graph.total_dim ({self.graph.total_dim})"

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

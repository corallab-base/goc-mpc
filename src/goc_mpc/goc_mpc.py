import logging
import pickle
import time
import numpy as np

from goc_mpc.graphs import Graph

# Configure verbosity via logging.getLogger("goc_mpc").setLevel(...).
logger = logging.getLogger(__name__)

from ._ext.configuration_spline import CubicConfigurationSpline, Block, BlockType
from ._ext.goc_mpc import (
    GraphOfConstraints,
    WaypointSolver,
    WaypointObjective,
    GraphWaypointMPC,
    MILPWaypointMPC,
    GraphTimingMPC,
    GraphShortPathMPC,
    ObstacleSet,
)


def _quat_block_slice(spec):
    """Ambient (offset, size) of `spec`'s first Block.SO3Quat block, relative
    to that agent's OWN ambient slice -- or None if it has none. Mirrors
    default_fk.py's _default_kind (same has_quat-over-Block-list check),
    used here so step()'s teleport branch renormalizes whichever slice is
    actually a quaternion for THIS agent instead of a hardcoded [3:7]."""
    offset = 0
    for block in spec:
        if block.type == BlockType.SO3Quat:
            return offset, block.size
        offset += block.size
    return None


class GraphOfConstraintsMPC():

    def __init__(
            self,
            graph: GraphOfConstraints,
            # waypoint mpc hyperparameters
            waypoint_solver: WaypointSolver = WaypointSolver.kGurobi,
            waypoint_objective: WaypointObjective = WaypointObjective.kMinMaxL1,
            waypoint_enforce_rigidity: bool = False,
            # pass an already-constructed GraphWaypointMPC (e.g. an
            # EvolutionaryWaypointSolver, which is duck-typed to this
            # protocol rather than a real C++ subclass) to use it directly;
            # waypoint_solver/waypoint_objective/
            # waypoint_enforce_rigidity are then ignored, since the
            # instance is already configured. Leave unset (default) to keep
            # auto-building a MILPWaypointMPC from those args, as before.
            waypoint_mpc: GraphWaypointMPC | None = None,
            # pass an already-constructed timing solver (e.g. a
            # TracedTimingMPC, duck-typed to GraphTimingMPC's public
            # surface rather than a real C++ subclass -- see
            # goc_mpc.traced_timing_mpc) to use it directly; time_cost/...
            # below are then ignored, since the instance is already
            # configured. Leave unset (default) to keep auto-building a
            # GraphTimingMPC from those args, as before.
            timing_mpc=None,
            # timing mpc hyperparameters
            time_cost: float = 1.0,
            time_cost2: float = 0.0,
            # Default ON (1.0): GraphTimingMPC is a trust-region Gauss-
            # Newton/qpOASES solver (graph_timing_mpc.hpp), not the earlier
            # Drake/IPOPT implementation this default used to be tuned
            # around -- that implementation needed acceleration_cost OFF by
            # default (and a separate "stability_cost" convex proxy ON
            # instead) specifically because its non-convex coast-corrected
            # residual could make IPOPT fail outright. The trust-region
            # solver converges to a local optimum regardless of convexity
            # (see graph_timing_mpc.hpp's own doc comment), so that
            # workaround -- and the stability_cost parameter itself -- no
            # longer exists; acceleration_cost is just the real min-
            # acceleration cost, on by default.
            acceleration_cost: float = 1.0,
            # energy_cost/arclength_cost: NOT supported by GraphTimingMPC's
            # current (trust-region SQP) implementation -- constructing it
            # with either nonzero throws. arclength_cost in particular
            # contains terms bilinear in (tau, velocity) before being
            # normed (see the per-quadrature-point sqrt(...) in
            # CubicConfigurationSpline::compute_arclength_cost), and was a
            # real contributor to both Ipopt IterationLimit failures AND
            # severe per-cycle slowdowns under the old implementation.
            energy_cost: float = 0.0,
            arclength_cost: float = 0.0,
            time_delta_cutoff: float = 0.4,
            phi_tolerance: float = 0.03,
            # Currently defunct
            max_vel: float | list[float] = -1.0,
            max_acc: float | list[float] = -1.0,
            max_jerk: float | list[float] = -1.0,
            # short path mpc hyperparameters
            short_path_length: int = 10,
            short_path_time_per_step: float = 0.05,
            # Static (per-episode) obstacle geometry consumed by the short
            # path MPC (sphere/box obstacles, point clouds, per-agent SDF
            # grids -- see ObstacleSet's own doc comment, obstacle_set.hpp).
            # Pass an ObstacleSet you've already registered geometry on, or
            # leave unset for an empty one.
            # GraphShortPathMPC stores it BY POINTER, not by copy, so
            # further mutation of the SAME object after construction remains
            # visible to every future solve() -- see `self.obstacles` below,
            # which is what keeps it alive.
            obstacles: ObstacleSet | None = None,
            # Remaining short path mpc hyperparameters mirror
            # GraphShortPathMPC's own constructor 1:1, same defaults -- see
            # its doc comment (graph_short_path_mpc.hpp) for the full
            # rationale behind each. agent_radii: one entry per agent,
            # default EMPTY meaning every agent has radius 0 (point agents
            # that still must not occupy the same position at the same
            # step); a pair's combined inter-agent avoidance radius is the
            # sum of both agents' own.
            agent_radii: np.ndarray | None = None,
            tracking_weight: float = 1.0,
            velocity_tracking_weight: float = 1.0,
            acceleration_weight: float = 1.0,
            penalty_weight: float = 1.0e3,
            max_iterations: int = 30,
            initial_trust_radius: float = 0.5,
            max_trust_radius: float = 5.0,
            min_trust_radius: float = 1.0e-6,
            grad_tol: float = 1.0e-6,
            constraint_prune_margin: float = 1.0,
            # Caller-supplied short-path solver, bypassing the
            # GraphShortPathMPC auto-construction below entirely -- mirrors
            # waypoint_mpc/timing_mpc's own override pattern. The override
            # need only expose the same solve()/view_points/view_vels/
            # view_times/view_obstacles/get_last_solve_time surface (see
            # GraphShortPathMPC's own doc comment) so this class's call site
            # (_solve_for_short_path) works unmodified. As with
            # waypoint_mpc/timing_mpc, the caller is responsible for having
            # built it against the same graph/specs/short_path_length,
            # and for keeping its own ObstacleSet alive (this class's
            # `self.obstacles` below is only populated in the
            # auto-constructed path).
            short_path_mpc=None,
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

        # persistent data
        self.graph = graph
        self.last_cycle_time = 0.0
        self.last_cycle_splines = [CubicConfigurationSpline(spec) for spec in graph._robot_specs]
        for s in self.last_cycle_splines:
            s.set_linear(linear_interpolation)
        self.last_cycle_waypoints = None
        self.last_cycle_var_assignments = None
        self.last_cycle_short_path = None
        self.last_cycle_backtracked_phases = set()
        self.last_grasp_commands = []
        # Wall-clock cost of the last step() call, broken down by the
        # sub-solve that spent it -- "backtrack"/"waypoint"/"timing"/
        # "short_path"/"total" (a phase absent from a given cycle, e.g.
        # "short_path" when short_path_mpc is None, or "timing" once
        # remaining_phases is empty, is simply left out of the dict rather
        # than recorded as 0.0). Callers (e.g. drive_loop.py's per-cycle
        # log line) read this right after step() returns to show where a
        # slow cycle's time actually went, instead of only ever seeing
        # step()'s combined total.
        self.last_cycle_solve_times = {}
        self.completed_phases = set()
        self.remaining_phases = list(range(graph.structure.num_nodes))
        # Nominal end-effector -> held-point transform for each currently
        # active hold (see _maybe_start_holds/_hold_violated), keyed by
        # hold_id. Populated the instant a hold's u_node completes; dropped
        # once its v_node completes or its u_node gets reopened (backtrack).
        self._hold_nominal_offsets = {}
        # Previous cycle's first dense waypoint per agent, DEBUG-logging
        # only (_log_timing_progression) -- lets that log flag whether the
        # near-term target is jittering cycle to cycle instead of holding
        # still while the robot closes in on it.
        self._debug_prev_first_wp = {}

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
        self.hold_drift_tolerance = hold_drift_tolerance

        # solvers
        if waypoint_mpc is not None:
            # caller is responsible for having constructed waypoint_mpc with
            # splines built from the same spec/agent count as this
            # instance (each C++ solver keeps its own copy of the splines
            # passed at construction, same as the auto-built path below).
            self.waypoint_mpc = waypoint_mpc
        else:
            self.waypoint_mpc = MILPWaypointMPC(graph, self.last_cycle_splines,
                                                solver = waypoint_solver,
                                                objective = waypoint_objective,
                                                enforce_rigidity = waypoint_enforce_rigidity)
        if timing_mpc is not None:
            # caller is responsible for having constructed timing_mpc with
            # splines built from the same spec/agent count as this
            # instance (each C++ solver keeps its own copy of the splines
            # passed at construction, same as the auto-built path below) --
            # fill_cubic_splines is called with self.last_cycle_splines
            # regardless, so that's the buffer that actually ends up filled.
            self.timing_mpc = timing_mpc
        else:
            self.timing_mpc = GraphTimingMPC(graph, self.last_cycle_splines,
                                             time_cost, time_cost2, acceleration_cost,
                                             energy_cost, arclength_cost,
                                             max_vel, max_acc, max_jerk)
        if short_path_mpc is not None:
            self.short_path_mpc = short_path_mpc
        else:
            # self.obstacles keeps the ObstacleSet alive for as long as this
            # GraphOfConstraintsMPC lives (mirrors self.graph = graph above)
            # -- required since GraphShortPathMPC stores it by pointer, not
            # by value (see obstacle_set.hpp's doc comment).
            self.obstacles = obstacles if obstacles is not None else ObstacleSet()
            self.short_path_mpc = GraphShortPathMPC(graph, short_path_length,
                                                    num_agents, short_path_time_per_step,
                                                    self.obstacles,
                                                    agent_radii if agent_radii is not None else np.array([]),
                                                    tracking_weight, velocity_tracking_weight,
                                                    acceleration_weight, penalty_weight,
                                                    max_iterations, initial_trust_radius,
                                                    max_trust_radius, min_trust_radius,
                                                    grad_tol, constraint_prune_margin)

    def _solve_for_waypoints(self, x: np.ndarray):
        if (self.solve_for_waypoints_once and self.last_cycle_waypoints is not None):
            return True
        else:
            success = self.waypoint_mpc.solve(self.remaining_phases, x)
            self.last_cycle_waypoints = self.waypoint_mpc.view_waypoints()
            return success

    def pass_node(self, node: int, assignments: np.ndarray):
        logger.info("Completed %s", self.graph.get_node_name(node))
        self.completed_phases |= {node}
        self.remaining_phases.remove(node)
        self.last_grasp_commands.extend(self.graph.get_grasp_changes(node, assignments))

    def _log_timing_progression(self, time_delta, x):
        """DEBUG-only: per agent, walks the same cumulative-tau rows
        set_progressed_time is about to walk, marking the first row it
        won't consider passed yet -- shows directly whether/where that
        walk falls short of time_delta_cutoff, instead of only seeing the
        aggregate passed_nodes result. Also reports the real distance from
        `x` to the current first dense waypoint, and whether that
        waypoint's own position moved since last cycle -- a tau that stays
        flat while distance is genuinely shrinking points at the timing
        solve itself; a first waypoint that jitters cycle to cycle instead
        of holding still points at the retraced target, not set_progressed_
        time."""
        if not logger.isEnabledFor(logging.DEBUG):
            return
        agent_offsets = self.graph.agent_col_offsets
        agent_nodes_list = self.timing_mpc.view_agent_nodes_list()
        time_deltas_list = self.timing_mpc.view_time_deltas_list()
        wps_list = self.timing_mpc.view_wps_list()
        for i, (nodes, taus) in enumerate(zip(agent_nodes_list, time_deltas_list)):
            cumulative = 0.0
            rows = []
            for node_id, tau in zip(nodes, taus):
                cumulative += tau
                label = self.graph.get_node_name(node_id) if node_id >= 0 else "interior"
                not_reached = time_delta < cumulative - self.time_delta_cutoff
                rows.append(f"{label}(tau={tau:.4f},cum={cumulative:.4f})"
                            + (" [NOT REACHED]" if not_reached else " [passed]"))
                if not_reached:
                    break
            logger.debug("agent %d progression: delta=%.4f tau_cutoff=%.4f rows=%s",
                         i, time_delta, self.time_delta_cutoff, rows)

            if wps_list[i].shape[0] == 0:
                continue
            first_wp = wps_list[i][0].copy()
            x_i = x[agent_offsets[i]:agent_offsets[i + 1]]
            distance = np.linalg.norm(x_i - first_wp)
            prev = self._debug_prev_first_wp.get(i)
            jitter = np.linalg.norm(first_wp - prev) if prev is not None else 0.0
            logger.debug("agent %d first_wp: distance_to_x=%.4f jitter_since_last_cycle=%.4f",
                         i, distance, jitter)
            self._debug_prev_first_wp[i] = first_wp

    def _solve_for_timing(self, time_delta, x, x_dot):

        # get references to the stored waypoints and assignments solutions from waypoint_mpc
        waypoints = self.waypoint_mpc.view_waypoints()
        assignments = self.waypoint_mpc.view_assignments()
        var_assignments = self.waypoint_mpc.view_var_assignments()
        self.last_cycle_var_assignments = var_assignments

        # PROGRESSION: progress time and potentially change phase
        # shift timing
        if len(self.remaining_phases) > 0 and time_delta > 0.0:
            self._log_timing_progression(time_delta, x)
            passed_nodes = self.timing_mpc.set_progressed_time(time_delta, self.time_delta_cutoff)
            logger.debug("passed_nodes: %s", passed_nodes)

            for node in passed_nodes:
                if node in self.graph.unpassable_nodes:
                    continue

                phi_results = {phi_id: self.graph.evaluate_phi(phi_id, x, assignments, self.phi_tolerance)
                               for phi_id in self.graph.get_phi_ids(node)}
                all_phis_satisfied = all(phi_results.values())

                if all_phis_satisfied:
                    logger.info("Completed %s", self.graph.get_node_name(node))
                    self.completed_phases |= {node}
                    self.remaining_phases.remove(node)
                    self.last_grasp_commands.extend(self.graph.get_grasp_changes(node, assignments))
                    self._maybe_commit(node, var_assignments)
                    self._maybe_start_holds(node, x)
                    self._maybe_end_holds(node)
                else:
                    failed_phi_ids = [phi_id for phi_id, ok in phi_results.items() if not ok]
                    logger.info("Did not complete %s -- failed phi id(s): %s",
                                self.graph.get_node_name(node), failed_phi_ids)

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
        if len(self.remaining_phases) > 0:
            remaining_phases_changed = True

            # otherwise,
            while remaining_phases_changed:
                remaining_phases_changed = False

                for (u_node, v_node), edge_phi_id in self.graph.get_next_edge_phis(self.remaining_phases).items():
                    if not self.graph.evaluate_edge_phi(edge_phi_id, x, self.last_cycle_var_assignments, 0.00):
                        logger.warning("Violated path constraint on %s->%s (edge phi id: %d)! backtracking.",
                                       self.graph.get_node_name(u_node), self.graph.get_node_name(v_node), edge_phi_id)

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
                        logger.warning("Violated hold on %s->%s (hold id: %d)! backtracking.",
                                       self.graph.get_node_name(hold.u_node), self.graph.get_node_name(hold.v_node), hold_id)

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

        # Reset rather than mutate in place: a phase this cycle doesn't
        # reach (e.g. "short_path" while teleport=True, or anything after
        # a RuntimeError below) should read as absent, not carry over a
        # stale value from the previous step() call.
        self.last_cycle_solve_times = {}
        step_start = time.perf_counter()

        if self.last_cycle_var_assignments is not None:
            phase_start = time.perf_counter()
            self._backtrack(x, x_dot)
            self.last_cycle_solve_times["backtrack"] = time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        success = self._solve_for_waypoints(x)
        self.last_cycle_solve_times["waypoint"] = time.perf_counter() - phase_start

        if not success:
            self.last_cycle_solve_times["total"] = time.perf_counter() - step_start
            raise RuntimeError("WaypointsMPC Failed!")

        phase_start = time.perf_counter()
        success = self._solve_for_timing(delta, x, x_dot)
        self.last_cycle_solve_times["timing"] = time.perf_counter() - phase_start

        if teleport:
            wps = self.waypoint_mpc.view_waypoints()
            next_agent_states = []
            next_agent_deltas = []

            nodes_and_timings = list(zip(
                self.timing_mpc.view_agent_nodes_list(),
                self.timing_mpc.view_time_deltas_list()
            ))

            agent_offsets = self.graph.agent_col_offsets
            for i, (agent_nodes, timings) in enumerate(nodes_and_timings):
                next_agent_node = next(iter(agent_nodes), -1)
                next_agent_delta = next(iter(timings), 0.0)
                lo, hi = agent_offsets[i], agent_offsets[i + 1]
                if next_agent_node == -1:
                    next_agent_state = x[lo:hi].copy()
                else:
                    next_agent_state = wps[next_agent_node, lo:hi].copy()
                quat_slice = _quat_block_slice(self.graph._robot_specs[i])
                if quat_slice is not None:
                    qoff, qsize = quat_slice
                    next_agent_state[qoff:qoff + qsize] /= np.linalg.norm(next_agent_state[qoff:qoff + qsize])
                next_agent_states.append(next_agent_state)
                next_agent_deltas.append(next_agent_delta)

            next_agent_states = np.expand_dims(np.concatenate(next_agent_states), 0)
            next_agent_states = np.tile(next_agent_states, (self.short_path_length, 1))

            next_agent_times = np.array(max(next_agent_deltas))
            next_agent_times = np.tile(next_agent_times, (self.short_path_length,))

            self.last_cycle_solve_times["total"] = time.perf_counter() - step_start
            return next_agent_states, None, next_agent_times

        if not success:
            self.last_cycle_solve_times["total"] = time.perf_counter() - step_start
            raise RuntimeError("TimingMPC Failed!")

        phase_start = time.perf_counter()
        success = self._solve_for_short_path(x, x_dot)
        self.last_cycle_solve_times["short_path"] = time.perf_counter() - phase_start

        if not success:
            self.last_cycle_solve_times["total"] = time.perf_counter() - step_start
            raise RuntimeError("ShortPathMPC Failed!")

        self.last_cycle_solve_times["total"] = time.perf_counter() - step_start

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

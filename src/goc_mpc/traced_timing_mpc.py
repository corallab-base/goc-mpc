"""TracedTimingMPC: a GraphTimingMPC-compatible (duck-typed) timing solver
that traces an obstacle-avoiding path between each pair of consecutive
waypoints per agent before optimizing the timed spline through it, instead
of GraphTimingMPC's straight-line chord between graph nodes.

`GraphOfConstraintsMPC` (goc_mpc.py) accepts any object satisfying
GraphTimingMPC's public surface via its `timing_mpc=` constructor argument
(no isinstance check), the same extension point already used for
EvolutionaryWaypointSolver on the waypoint side -- see
goc_mpc.evolutionary_waypoint_solver.

A single `GraphTimingMPC` instance backs this class -- NOT a coarse/traced
pair of separate MPC objects (that was this class's original design; see
git history if you need it). `GraphOfConstraints.get_agent_paths` resolves
ordering and cross-agent LESS_THAN/EQUAL timing interactions exactly as
`GraphTimingMPC.solve` itself does internally (both call the same C++
method) -- calling it directly here, rather than running a full separate
coarse `GraphTimingMPC.solve()` first, means there's exactly one place
(`GraphTimingMPC`) that knows how to time a sequence of fixed points,
whether that sequence is the graph's real nodes (`solve()`) or a denser one
with traced interior waypoints spliced in between them (`solve_dense()`).
Each consecutive pair of positions on the resolved per-agent path (x0 ->
first node -> second node -> ...) is traced through a `TimeToGoField`
-protocol `field` (see time_to_go_field.py) into a dense obstacle-avoiding
polyline; `GraphOfConstraints.reindex_agent_interactions` then remaps the
cross-agent interactions' depths (originally indices into the real-node-only
list `get_agent_paths` returns) onto that denser sequence, so
`solve_dense()`'s cross-agent LESS_THAN/EQUAL constraints are exact --
unlike the old coarse/traced split, which only enforced them approximately
(against the coarse per-edge time budget, not the refined per-segment one).
"""

import numpy as np

from .goc_mpc import GraphOfConstraints, GraphTimingMPC


def _trim_trailing_duplicates(path, eps=1e-9):
    """Drops trailing near-duplicate points from a traced (K, dim)
    polyline, keeping everything up to (and one past) the last point where
    consecutive positions actually differ, plus forcing the true final
    point back on unconditionally.

    `EdgeCostTimeToGoField` (time_to_go_field.py) traces via a *fixed*
    number of jax.lax.scan steps -- once the descent converges within
    goal_tol it freezes, so the raw path is padded with the converged
    position repeated out to max_steps regardless of how close start
    already was to goal. Left untrimmed, `_resample_by_arclength` below
    would always see a long path and always spend most of its output
    budget on synthetic interior points, even for an already-converged
    (near-zero-length) segment -- inflating the QP's size (every extra
    dense point is another free velocity/time-delta pair, see
    add_agent_timing_segments) for no benefit. Trimming restores what a
    variable-length trace (e.g. the old finite-difference tracer, or
    NTFieldSolver's own early-exit loop) gives for free: an already-short
    /degenerate raw path once start is close to goal.
    """
    path = np.asarray(path)
    if len(path) <= 2:
        return path

    diffs = np.linalg.norm(np.diff(path, axis=0), axis=1)
    moving = np.where(diffs > eps)[0]
    if len(moving) == 0:
        return path[[0, -1]]

    trimmed = path[:moving[-1] + 2]
    if not np.array_equal(trimmed[-1], path[-1]):
        trimmed = np.concatenate([trimmed, path[-1:]], axis=0)
    return trimmed


def _rdp(path, epsilon):
    """Ramer-Douglas-Peucker simplification: keeps only the points needed
    to represent a (K, dim) polyline's actual SHAPE within `epsilon`
    perpendicular distance of the straight chord spanning each kept span
    -- endpoints always kept, an interior point only if some point in its
    span deviates from that chord by more than `epsilon`. Iterative
    (explicit stack, not recursion) since a raw trace can be a few hundred
    points long.

    Replaces a flat per-segment point count (`_resample_by_arclength`)
    as the primary way `_agent_dense_wps_and_ids` thins a traced polyline:
    a near-straight segment (no real obstacle interaction, or one that's
    shrunk down to almost nothing as the agent approaches its goal)
    collapses to just its two endpoints regardless of how many raw
    descent steps produced it, while a segment that genuinely bends
    around an obstacle keeps the points that capture the bend. A flat
    count instead pads a near-straight/near-zero-length segment out to
    the same point budget as a real detour -- and since each point is a
    free velocity/time-delta pair with its own effectively nonzero
    minimum tau (add_agent_timing_segments / stability_cost's 1/tau
    term), that inflates the minimum time required to traverse a segment
    far past what its real shape needs -- see `_agent_dense_wps_and_ids`
    for how this showed up as the timing MPC stalling indefinitely just
    short of a real waypoint.
    """
    path = np.asarray(path)
    n = len(path)
    if n <= 2:
        return path

    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        a, b = path[start], path[end]
        ab = b - a
        ab_norm = np.linalg.norm(ab)
        if ab_norm < 1e-12:
            dists = np.linalg.norm(path[start + 1:end] - a, axis=1)
        else:
            ab_unit = ab / ab_norm
            proj = np.clip((path[start + 1:end] - a) @ ab_unit, 0.0, ab_norm)
            closest = a + np.outer(proj, ab_unit)
            dists = np.linalg.norm(path[start + 1:end] - closest, axis=1)
        idx = int(np.argmax(dists))
        if dists[idx] > epsilon:
            split = start + 1 + idx
            keep[split] = True
            stack.append((start, split))
            stack.append((split, end))

    return path[keep]


def _resample_by_arclength(path, n_points):
    """Downsamples a traced (K, dim) polyline to `n_points` evenly spaced
    (by world arc length) points, endpoints included exactly. A raw
    gradient-descent trace takes one point per fixed-size descent step,
    which is far denser than the timing QP needs -- every extra point is
    another free velocity/time-delta pair (see add_agent_timing_segments)
    -- so this keeps the QP a fixed, small size per segment regardless of
    trace resolution. Mirrors goc_ntfields's NTFieldSolver.trace_and_sample.
    """
    path = np.asarray(path)
    if len(path) <= n_points:
        return path

    diffs = np.diff(path, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum_length = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = cum_length[-1]

    if total_length == 0.0:
        return path[[0, -1]]

    sample_distances = np.linspace(0.0, total_length, n_points)
    return np.stack([
        np.interp(sample_distances, cum_length, path[:, d])
        for d in range(path.shape[1])
    ], axis=1)


class TracedTimingMPC:

    def __init__(
            self,
            graph,
            splines,
            field,
            time_cost: float = 1.0,
            time_cost2: float = 0.0,
            # Default ON (1.0) -- see GraphOfConstraintsMPC's matching
            # acceleration_cost comment (goc_mpc.py) for why: GraphTimingMPC
            # is a trust-region Gauss-Newton/qpOASES solver, which converges
            # to a local optimum regardless of convexity, so the old
            # default-off/stability_cost-proxy workaround (needed only
            # because that non-convex residual could make IPOPT fail
            # outright) no longer applies or exists.
            acceleration_cost: float = 1.0,
            # energy_cost/arclength_cost: NOT supported by GraphTimingMPC's
            # current (trust-region SQP) implementation -- constructing it
            # with either nonzero throws. See GraphOfConstraintsMPC's
            # matching comment (goc_mpc.py).
            energy_cost: float = 0.0,
            arclength_cost: float = 0.0,
            # A bare float broadcasts to every block in splines[0]'s spec; a
            # list[float] must have exactly one entry per block, in spec
            # order. NOT supported by GraphTimingMPC's current (trust-region
            # SQP) implementation -- constructing it with an actual (> 0)
            # bound throws; the unbounded sentinel <= 0 (the default) is fine.
            max_vel: float | list[float] = -1.0,
            max_acc: float | list[float] = -1.0,
            max_jerk: float | list[float] = -1.0,
            # Primary mechanism for thinning each segment's traced polyline
            # before handing it to the timing QP: Ramer-Douglas-Peucker
            # simplification (see _rdp's doc comment) with epsilon set to
            # `rdp_tolerance` FRACTION of that segment's own straight-line
            # chord length (‖goal - start‖), not an absolute world-unit
            # distance -- keeps the tolerance scale-free across differently
            # -scaled scenarios and automatically tight on a segment that's
            # shrunk down toward its goal. 0.02 (2%) is an empirically
            # reasonable starting point, not a derived value -- tune lower
            # for more shape fidelity, higher for fewer points. None
            # disables simplification entirely, falling back to the old
            # flat-count `_resample_by_arclength(traced,
            # max_points_per_segment)` behavior.
            rdp_tolerance: float | None = 0.02,
            # Safety cap, not the primary mechanism: if RDP still returns
            # more than this many points (a genuinely convoluted segment),
            # arc-length-resample ITS output down to this count so QP size
            # stays bounded in the worst case. None disables the cap (use
            # RDP's output as-is). Also doubles as the flat point count
            # `_resample_by_arclength` uses when rdp_tolerance is None.
            max_points_per_segment: int | None = 8,
    ):
        self.graph = graph
        self.field = field
        self.rdp_tolerance = rdp_tolerance
        self.max_points_per_segment = max_points_per_segment

        cost_args = (time_cost, time_cost2, acceleration_cost, energy_cost,
                     arclength_cost, max_vel, max_acc, max_jerk)

        # Own internal spline copy -- fill_cubic_splines is never called on
        # this; it's only used inside solve_dense() for ambient_dim()/
        # tangent_dim() and the cost functors. The caller's real output
        # splines are whatever is passed to this class's own
        # fill_cubic_splines(splines, x0, v0), delegated straight to
        # self._timing below.
        self._timing = GraphTimingMPC(graph, list(splines), *cost_args)

    def _agent_dense_wps_and_ids(self, agent, x0_i, coarse_node_ids, coarse_wps_i):
        """Traces every consecutive (position, position) pair on this
        agent's coarse path -- starting at its current position `x0_i`,
        through each real graph-node waypoint in order (resolved by
        `graph.get_agent_paths`, positions looked up from `waypoints` --
        see `solve` below) -- and concatenates the results into one dense
        (K, dim) waypoint matrix plus a parallel length-K list of real
        graph-node ids (-1 for a synthetic traced interior point). `x0_i`
        itself is never included in the returned arrays, matching
        build_dense_graph_timing_problem's wps_i convention (x0/v0 are
        solve_dense()'s own separate boundary-condition arguments).
        """
        if len(coarse_node_ids) == 0:
            return np.zeros((0, self.graph.robot_ambient_dim(agent))), []

        positions = [x0_i] + [coarse_wps_i[j] for j in range(len(coarse_node_ids))]
        node_ids = [None] + list(coarse_node_ids)

        dense_positions = []
        dense_ids = []
        for k in range(len(positions) - 1):
            start, goal = positions[k], positions[k + 1]
            goal_node_id = node_ids[k + 1]

            traced = np.asarray(self.field.trace_path(agent, start, goal))
            traced = _trim_trailing_duplicates(traced)
            if self.rdp_tolerance is not None:
                chord = np.linalg.norm(np.asarray(goal) - np.asarray(start))
                traced = _rdp(traced, self.rdp_tolerance * chord)
                if (self.max_points_per_segment is not None
                        and len(traced) > self.max_points_per_segment):
                    traced = _resample_by_arclength(traced, self.max_points_per_segment)
            elif self.max_points_per_segment is not None:
                traced = _resample_by_arclength(traced, self.max_points_per_segment)
            # traced[0] == start (already accounted for -- either dropped
            # here for k==0 since x0_i isn't part of the dense array, or
            # already appended as the previous segment's goal); keep
            # everything after it. A zero-length segment (start == goal,
            # e.g. the agent is already sitting exactly at this node's
            # target) collapses trace_path to a single point == goal --
            # keep that one point rather than dropping it, since there's
            # nothing to drop it as a duplicate "start" of.
            interior = traced[1:] if len(traced) > 1 else traced
            for point in interior[:-1]:
                dense_positions.append(np.asarray(point))
                dense_ids.append(-1)
            dense_positions.append(np.asarray(interior[-1]))  # == goal, exact
            dense_ids.append(-1 if goal_node_id is None else int(goal_node_id))

        return np.asarray(dense_positions), dense_ids

    def solve(self, x0, v0, remaining_vertices, waypoints, assignments, t_by_node=None):
        if t_by_node is None:
            t_by_node = np.array([])

        # Resolves ordering + cross-agent LESS_THAN/EQUAL interactions --
        # the same call GraphTimingMPC.solve makes internally -- directly,
        # instead of running a full separate coarse GraphTimingMPC.solve()
        # first (this class's original design; see module docstring).
        _parents, agent_nodes, agent_interactions = self.graph.get_agent_paths(
            remaining_vertices, assignments, t_by_node)

        agent_dense_wps = []
        agent_dense_node_ids = []
        agent_offsets = self.graph.agent_col_offsets
        for i in range(self.graph.num_agents):
            lo, hi = agent_offsets[i], agent_offsets[i + 1]
            x0_i = np.asarray(x0[lo:hi])
            node_ids_i = agent_nodes[i]
            # get_agent_paths resolves WHICH real nodes/order, not their
            # positions -- look those up from `waypoints` ourselves, the
            # same indexing build_graph_timing_problem's wps_i uses.
            if len(node_ids_i) == 0:
                wps_i = np.zeros((0, hi - lo))
            else:
                wps_i = np.asarray([
                    waypoints[node, lo:hi]
                    for node in node_ids_i
                ])
            dense_wps, dense_ids = self._agent_dense_wps_and_ids(i, x0_i, node_ids_i, wps_i)
            agent_dense_wps.append(dense_wps)
            agent_dense_node_ids.append(dense_ids)

        # agent_interactions' depths are indices into agent_nodes (real
        # nodes only) -- reindex them onto the now-denser
        # agent_dense_node_ids, or a cross-agent LESS_THAN/EQUAL constraint
        # would sum too few segments and under-constrain the refined
        # timing (see GraphOfConstraints.reindex_agent_interactions).
        agent_interactions = GraphOfConstraints.reindex_agent_interactions(
            agent_interactions, agent_dense_node_ids)

        success = self._timing.solve_dense(
            x0, v0, agent_dense_wps, agent_dense_node_ids, agent_interactions)
        return success

    # GraphTimingMPC-compatible duck-typed surface -- delegates straight to
    # self._timing, which holds both solve() and solve_dense()'s state.

    def get_agent_spline_length(self, agent):
        return self._timing.get_agent_spline_length(agent)

    def get_agent_spline_nodes(self, agent):
        return self._timing.get_agent_spline_nodes(agent)

    def set_progressed_time(self, delta, tau_cutoff):
        return self._timing.set_progressed_time(delta, tau_cutoff)

    def fill_cubic_splines(self, splines, x0, v0):
        return self._timing.fill_cubic_splines(splines, x0, v0)

    def get_next_taus(self):
        return self._timing.get_next_taus()

    def get_next_nodes(self):
        return self._timing.get_next_nodes()

    def view_wps_list(self):
        return self._timing.view_wps_list()

    def view_vs_list(self):
        return self._timing.view_vs_list()

    def view_time_deltas_list(self):
        return self._timing.view_time_deltas_list()

    def view_agent_nodes_list(self):
        return self._timing.view_agent_nodes_list()

    def view_agent_spline_length_map(self):
        return self._timing.view_agent_spline_length_map()

    def get_last_solve_time(self):
        return self._timing.get_last_solve_time()

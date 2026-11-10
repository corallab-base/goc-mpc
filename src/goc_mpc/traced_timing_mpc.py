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
first node -> second node -> ...) is planned through the agent's `ntfields`
`Field` (see `AgentField`) into a dense obstacle-avoiding polyline, or kept as
a straight segment for an agent without one;
`GraphOfConstraints.reindex_agent_interactions` then remaps the
cross-agent interactions' depths (originally indices into the real-node-only
list `get_agent_paths` returns) onto that denser sequence, so
`solve_dense()`'s cross-agent LESS_THAN/EQUAL constraints are exact --
unlike the old coarse/traced split, which only enforced them approximately
(against the coarse per-edge time budget, not the refined per-segment one).

Every segment of a solve is known once the waypoints are, so all of them are
planned before any is timed, each field's segments in one batch. The `mppi`
planner runs on `fast_ntfields`' CUDA planner (`fast_ntfields.plan.mppi_batch`)
when it is installed and supports the field (one loaded from a TD-NTFields
checkpoint), and otherwise, with a warning, on `ntfields.plan.mppi` one
segment at a time.

`ntfields` and `fast_ntfields` are optional dependencies (the `ntfields` and
`fast-ntfields` extras), imported only when some agent has a field.
"""

import functools
import warnings
from collections.abc import Mapping, Sequence

import numpy as np

from .fields import AgentField
from .goc_mpc import GraphOfConstraints, GraphTimingMPC


def _trim_trailing_duplicates(path, eps=1e-9):
    """Drops trailing near-duplicate points from a planned (K, dim)
    polyline, keeping everything up to (and one past) the last point where
    consecutive positions actually differ, plus forcing the true final
    point back on unconditionally.

    A planner that converges early can repeat its final position; left
    untrimmed, `_resample_by_arclength` below would spend its output budget
    on those points -- inflating the QP's size (every extra dense point is
    another free velocity/time-delta pair, see add_agent_timing_segments)
    for no benefit.
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
    trace resolution.
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


def _interpolate_trailing(lead, start, goal):
    """`lead` (K, dim), a path over the leading `dim` columns from `start[:dim]`
    to `goal[:dim]`, with the trailing columns of `start`/`goal` interpolated
    linearly in `lead`'s arc length."""
    dim = lead.shape[1]
    if start.shape[0] == dim:
        return lead
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(lead, axis=0), axis=1))])
    alpha = cum / cum[-1] if cum[-1] > 0.0 else np.linspace(0.0, 1.0, len(lead))
    trailing = start[dim:] + alpha[:, None] * (goal[dim:] - start[dim:])
    return np.concatenate([lead, trailing], axis=1)


@functools.cache
def _fast_ntfields_plan():
    """`fast_ntfields.plan`, or None (with a warning) if fast_ntfields is not
    installed."""
    try:
        from fast_ntfields import plan
    except ImportError as e:
        warnings.warn(f"fast_ntfields is not available ({e}); planning mppi segments with "
                      "ntfields.plan.mppi, one at a time", stacklevel=3)
        return None
    return plan


def _fast_mppi_batch(field):
    """`fast_ntfields.plan.mppi_batch` if it can plan on `field`, else None
    (with a warning)."""
    plan = _fast_ntfields_plan()
    if plan is None:
        return None
    if not plan.supports(field):
        warnings.warn(f"fast_ntfields cannot plan on this {type(field).__name__} (not loaded from a TD-NTFields "
                      "checkpoint); planning its mppi segments with ntfields.plan.mppi, one at a time",
                      stacklevel=3)
        return None
    return plan.mppi_batch


class TracedTimingMPC:

    def __init__(
            self,
            graph,
            splines,
            # Per-agent `AgentField`, or None for a straight segment: each
            # segment is planned by an `ntfields` planner
            # (`ntfields.plan.PLANNERS`; `mppi` on fast_ntfields when it is
            # installed, see the module docstring).
            fields: Sequence[AgentField | None],
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
            *,
            # `None` uses each field's method's own default planner.
            planner: str | None = None,
            planner_kwargs: Mapping | None = None,
    ):
        if len(fields) != graph.num_agents:
            raise ValueError(f"fields has {len(fields)} entries for {graph.num_agents} agents")
        self.graph = graph
        self.fields = list(fields)
        self.planner = planner
        self.planner_kwargs = dict(planner_kwargs or {})
        #: Segments in the last `solve` whose planner did not reach its goal;
        #: their paths end with a jump to the goal.
        self.last_failed_segments = 0
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

    def plan_segment(self, agent, start, goal):
        """`start` -> `goal` (this agent's full rows) through its `AgentField`
        (see `plan_segments`)."""
        return self.plan_segments([(agent, start, goal)])[0]

    def plan_segments(self, segments):
        """Each `(agent, start, goal)` of `segments` (the agent's full rows)
        planned through that agent's `AgentField`: the leading columns in
        field coordinates by the `ntfields` planner, mapped back, endpoints
        exact, trailing columns interpolated. Segments on the same field are
        planned together. An agent whose entry is None gets the straight
        segment."""
        segments = [(agent, np.asarray(start, dtype=float), np.asarray(goal, dtype=float))
                    for agent, start, goal in segments]
        traced = [None] * len(segments)
        groups = {}
        for k, (agent, start, goal) in enumerate(segments):
            spec = self.fields[agent]
            if spec is None:
                traced[k] = np.stack([start, goal])
                continue
            planner = self.planner or getattr(getattr(spec.field, "method", None), "default_planner", "mppi")
            groups.setdefault((id(spec.field), planner), []).append(k)
        if not groups:
            return traced

        import torch
        from ntfields.plan import PLANNERS

        for (_, planner), ks in groups.items():
            specs = [self.fields[segments[k][0]] for k in ks]
            field = specs[0].field

            def to_field(spec, q):
                offset = np.asarray(spec.offset, dtype=float)
                return (q[:offset.shape[0]] - offset) / float(spec.scale)

            starts = torch.as_tensor(np.stack([to_field(sp, segments[k][1]) for sp, k in zip(specs, ks)]),
                                     dtype=torch.float32, device=field.device)
            goals = torch.as_tensor(np.stack([to_field(sp, segments[k][2]) for sp, k in zip(specs, ks)]),
                                    dtype=torch.float32, device=field.device)
            fast = _fast_mppi_batch(field) if planner == "mppi" else None
            if fast is not None:
                results = fast(field, starts, goals, **self.planner_kwargs)
            else:
                results = [PLANNERS[planner](field, s, g, **self.planner_kwargs) for s, g in zip(starts, goals)]
            for k, spec, result in zip(ks, specs, results):
                traced[k] = self._map_back(spec, segments[k][1], segments[k][2], result)
        return traced

    def _map_back(self, spec, start, goal, result):
        offset = np.asarray(spec.offset, dtype=float)
        dim = offset.shape[0]
        lead = result.path.detach().cpu().numpy().astype(float) * float(spec.scale) + offset
        lead[0] = start[:dim]
        if result.success:
            lead[-1] = goal[:dim]
        else:
            self.last_failed_segments += 1
            lead = np.concatenate([lead, goal[None, :dim]], axis=0)
        return _interpolate_trailing(lead, start, goal)

    @staticmethod
    def _agent_segments(x0_i, coarse_wps_i):
        """The (start, goal) pairs along an agent's coarse path: its current
        position `x0_i`, then each real graph-node waypoint in order."""
        positions = [x0_i] + list(coarse_wps_i)
        return [(positions[k], positions[k + 1]) for k in range(len(positions) - 1)]

    def _agent_dense_wps_and_ids(self, agent, coarse_node_ids, segments, traced_segments):
        """Concatenates an agent's traced segments (`segments`' (start, goal)
        pairs, see `_agent_segments`, as planned by `plan_segments`) into one
        dense (K, dim) waypoint matrix plus a parallel length-K list of real
        graph-node ids (-1 for a synthetic traced interior point). The
        agent's current position itself is never included in the returned
        arrays, matching build_dense_graph_timing_problem's wps_i convention
        (x0/v0 are solve_dense()'s own separate boundary-condition
        arguments).
        """
        if len(coarse_node_ids) == 0:
            return np.zeros((0, self.graph.robot_ambient_dim(agent))), []

        node_ids = [None] + list(coarse_node_ids)
        dense_positions = []
        dense_ids = []
        for k, (start, goal) in enumerate(segments):
            goal_node_id = node_ids[k + 1]

            traced = np.asarray(traced_segments[k])
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
            # target) collapses the plan to a single point == goal --
            # keep that one point rather than dropping it, since there's
            # nothing to drop it as a duplicate "start" of.
            interior = traced[1:] if len(traced) > 1 else traced
            for point in interior[:-1]:
                dense_positions.append(np.asarray(point))
                dense_ids.append(-1)
            dense_positions.append(np.asarray(interior[-1]))  # == goal, exact
            dense_ids.append(-1 if goal_node_id is None else int(goal_node_id))

        return np.asarray(dense_positions), dense_ids

    def solve(self, x0, v0, remaining_vertices, waypoints, var_assignments, t_by_node=None):
        if t_by_node is None:
            t_by_node = np.array([])
        self.last_failed_segments = 0

        # Resolves ordering + cross-agent LESS_THAN/EQUAL interactions --
        # the same call GraphTimingMPC.solve makes internally -- directly,
        # instead of running a full separate coarse GraphTimingMPC.solve()
        # first (this class's original design; see module docstring).
        _parents, agent_nodes, agent_interactions = self.graph.get_agent_paths(
            remaining_vertices, var_assignments, t_by_node)

        agent_segments = []
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
            agent_segments.append(self._agent_segments(x0_i, wps_i) if len(node_ids_i) else [])

        traced = self.plan_segments([(i, start, goal)
                                     for i, segments in enumerate(agent_segments)
                                     for start, goal in segments])
        agent_dense_wps = []
        agent_dense_node_ids = []
        for i, segments in enumerate(agent_segments):
            traced_i, traced = traced[:len(segments)], traced[len(segments):]
            dense_wps, dense_ids = self._agent_dense_wps_and_ids(i, agent_nodes[i], segments, traced_i)
            agent_dense_wps.append(dense_wps)
            agent_dense_node_ids.append(dense_ids)

        # agent_interactions' depths are indices into agent_nodes (real
        # nodes only) -- reindex them onto the now-denser
        # agent_dense_node_ids, or a cross-agent LESS_THAN/EQUAL constraint
        # would sum too few segments and under-constrain the refined
        # timing (see GraphOfConstraints.reindex_agent_interactions).
        agent_interactions = GraphOfConstraints.reindex_agent_interactions(
            agent_interactions, agent_dense_node_ids)

        # var_assignments is forwarded (not just used above for
        # get_agent_paths) so fill_cubic_splines' per-node knot mask can
        # resolve var_agent_q(...) constraints -- otherwise a real node
        # pinned only by an assignable-agent constraint looks unconstrained
        # and the spline wrongly bridges past it.
        success = self._timing.solve_dense(
            x0, v0, agent_dense_wps, agent_dense_node_ids, agent_interactions,
            var_assignments)
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

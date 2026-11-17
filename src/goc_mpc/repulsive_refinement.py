"""Space-time repulsive refinement of the timing layer's splines.

`RepulsiveRefinementConfig` configures goc-mpc's C++ `RepulsiveRefiner`
(repulsive_refiner.hpp): inter-agent repulsion between the agents' collision
spheres through space-time, plus obstacle repulsion against the controller's
`ObstacleSet`, minimized over the splines' knots. `SplineRefiner` runs it on
one cycle: it reads each agent's knots back from the timing solve, refines
them, and writes them into the splines the short-path solver tracks.

Spheres come from the graph's registered collision models
(`set_agent_collision_model`); an agent without one is a single sphere at
q[:workspace_dim] of radius `agent_radii[agent]`.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Sequence

import numpy as np

from ._ext.goc_mpc import RefinerKnots, RefinerOptions, RepulsiveRefiner, SpaceTimeKernel


@dataclasses.dataclass(frozen=True)
class RepulsiveRefinementConfig:
    """Kernel (see spacetime_repulsion.hpp): tangent-point exponents `alpha`,
    `beta`; `time_scale` (m/s) lifts time into space; the radial barrier acts
    on the sphere gap, is zero from `cutoff` on, quadratic below `relax`, and
    has force 1 at `reference_gap`.

    Speed: `speed_weight` times the time integral of the squared excess of
    each spline block's speed over `max_speed`; 0 disables it.

    Descent (see repulsive_refiner.hpp): L-BFGS on the knots, at most
    `max_iterations` steps of at most `max_step` per knot entry, stopping once
    the clearances reach `target_clearance`. `metric_scale` weighs the
    (position, velocity, time) directions. Each knot interval is sampled at
    fixed fractions about `sample_dt` apart; agents stay parked after their
    last knot until the latest end time plus `horizon_margin`."""
    alpha: float = 2.0
    beta: float = 4.5
    time_scale: float = 1.0
    cutoff: float = 0.3
    relax: float = 0.05
    reference_gap: float = 0.1
    repulsion_weight: float = 1.0
    obstacle_weight: float = 1.0
    speed_weight: float = 0.0
    max_speed: float = 0.0
    sample_dt: float = 0.1
    horizon_margin: float = 2.0
    max_iterations: int = 100
    target_clearance: float = 0.02
    max_step: float = 0.1
    metric_scale: tuple = (1.0, 1.0, 0.25)
    min_knot_dt: float = 0.01
    memory: int = 8
    max_backtracks: int = 12
    armijo: float = 1e-4
    agent_radii: Optional[Sequence[float]] = None

    def options(self) -> RefinerOptions:
        o = RefinerOptions()
        o.kernel = SpaceTimeKernel(alpha=self.alpha, beta=self.beta, time_scale=self.time_scale,
                                   cutoff=self.cutoff, relax=self.relax, reference_gap=self.reference_gap)
        for name in ("repulsion_weight", "obstacle_weight", "speed_weight", "max_speed", "sample_dt", "horizon_margin", "max_iterations", "target_clearance",
                     "max_step", "min_knot_dt", "memory", "max_backtracks", "armijo"):
            setattr(o, name, getattr(self, name))
        o.metric_scale = np.asarray(self.metric_scale, dtype=float)
        o.agent_radii = np.asarray(self.agent_radii if self.agent_radii is not None else [], dtype=float)
        return o


def agent_knots(timing_mpc, graph, spline, agent_id: int, x: np.ndarray, x_dot: np.ndarray) -> RefinerKnots:
    """Agent `agent_id`'s knots as `fill_cubic_splines` assembled
    `spline` this cycle: x0/v0 prepended to the timing solve's waypoints and
    velocities, a zero final velocity, times from the solve's deltas.

    Free to move: velocities and times at every row but the first (the
    current state) and, for velocities, the last (the stop at the final
    waypoint); positions only at rows the traced timing solve inserted
    between graph nodes (node id -1), never at a waypoint the waypoint
    solve chose. An agent with no spline this cycle is parked at x0, fully
    frozen: still an obstacle to the others."""
    n = timing_mpc.get_agent_spline_length(agent_id)
    lo, hi = graph.agent_col_offsets[agent_id], graph.agent_col_offsets[agent_id + 1]
    x0, v0 = np.asarray(x[lo:hi], dtype=float), np.asarray(x_dot[lo:hi], dtype=float)
    blocks = len(graph._robot_specs[agent_id])
    if n <= 1:
        frozen = np.zeros(2, dtype=bool)
        return RefinerKnots(np.array([0.0, 1.0]), np.vstack([x0, x0]), np.vstack([v0, np.zeros_like(v0)]),
                            np.ones((2, blocks), dtype=np.int32), frozen, frozen, frozen)
    wps = np.vstack([x0[None, :], np.asarray(timing_mpc.view_wps_list()[agent_id])])
    vs = np.vstack([v0[None, :], np.asarray(timing_mpc.view_vs_list()[agent_id]), np.zeros((1, hi - lo))])
    times = np.concatenate([[0.0], np.cumsum(np.asarray(timing_mpc.view_time_deltas_list()[agent_id])[:n - 1])])
    nodes = timing_mpc.view_agent_nodes_list()[agent_id]
    wps_free = np.array([False] + [node == -1 for node in nodes])
    vs_free = np.ones(n, dtype=bool)
    vs_free[[0, -1]] = False
    times_free = np.ones(n, dtype=bool)
    times_free[0] = False
    active = np.asarray(spline.block_active_mask(), dtype=np.int32)
    if active.size == 0:
        active = np.ones((n, blocks), dtype=np.int32)
    return RefinerKnots(times, wps, vs, active, wps_free, vs_free, times_free)


class SplineRefiner:
    """A `RepulsiveRefiner` for `graph`, built once and run every cycle."""

    def __init__(self, graph, config: RepulsiveRefinementConfig, obstacles=None):
        self.graph = graph
        self.config = config
        self.obstacles = obstacles
        self._refiner = RepulsiveRefiner(graph, obstacles, config.options())

    def refine(self, timing_mpc, splines, x: np.ndarray, x_dot: np.ndarray) -> Optional[dict]:
        """Refines this cycle's `splines` (filled by `fill_cubic_splines`
        from the same `x`, `x_dot`) in place. Returns the refiner's report
        as a dict, or None for a graph of fewer than 2 agents."""
        if self.graph.num_agents < 2:
            return None
        knots = [agent_knots(timing_mpc, self.graph, splines[a], a, x, x_dot) for a in range(self.graph.num_agents)]
        refined, report = self._refiner.refine(knots)
        if report.moved:
            for a, k in enumerate(refined):
                if timing_mpc.get_agent_spline_length(a) <= 1:
                    continue
                splines[a].set_block_active_mask(np.asarray(k.active))
                splines[a].set(np.asarray(k.wps), np.asarray(k.vs), np.asarray(k.times))
        return report_dict(report)


def report_dict(report) -> dict:
    return {
        "moved": report.moved,
        "iterations": report.iterations,
        "evaluations": report.evaluations,
        "energy_initial": report.energy_initial,
        "energy_final": report.energy_final,
        "clearance_initial": (report.clearance_initial.inter_agent, report.clearance_initial.obstacle),
        "clearance_final": (report.clearance_final.inter_agent, report.clearance_final.obstacle),
        "pairs_initial": report.pairs_initial,
        "pairs_final": report.pairs_final,
        "time": report.time,
        "time_sampling": report.time_sampling,
        "time_energy": report.time_energy,
        "time_pullback": report.time_pullback,
        "time_clearance": report.time_clearance,
        "stop_reason": report.stop_reason,
    }

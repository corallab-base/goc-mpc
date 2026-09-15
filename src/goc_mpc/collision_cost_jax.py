"""Step 2 of PLAN_collision_aware_timing.md: the collision-cost forward
pass over the timing layer's own full-horizon splines -- sample every
agent's spline (spline_jax.eval, Step 1) on one shared absolute time grid,
push through each agent's sphereized FK, and penalize inter-agent sphere
overlap plus low environment speed, regularized against a reference
trajectory. One plain jax function of (per-agent spline params), meant to
be wrapped in the caller's own `jax.jit(jax.value_and_grad(...))` -- no
jit/vmap decoration happens in here, so composing this with an outer
optimization loop (Step 2's still-unwired second half) doesn't require
re-tracing this module's internals.

Deliberately robot- and environment-agnostic: `fk_fn`/`speed_fn` are
caller-supplied callables, not implemented here. The b1_multiroom SE(2)
sphere FK the plan calls out as "does not exist yet" is real B1 geometry
(sphere offsets/radii matching its actual body), which belongs in
po_goc_mpc alongside its ur5e_fk.py, not in this general-purpose library --
see PLAN_collision_aware_timing.md's still-open "which package owns the
JAX code" question. This module is the generic half of that split: the
sampling + collision-cost math that's the same regardless of which robot
or environment field is plugged in.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Optional, Sequence

import numpy as np
import jax
import jax.numpy as jnp

from . import spline_jax


class AgentGeometry(NamedTuple):
    """Static (non-differentiated) per-agent configuration -- spec/fk_fn/
    sphere_radii don't change cycle to cycle, so these are plain Python
    objects closed over by `build_cost_fn`, not part of the params pytree
    `jax.grad` differentiates through."""
    spec: Sequence               # goc_mpc.splines.Block list (R/Torus only)
    fk_fn: Callable[[jnp.ndarray], jnp.ndarray]  # q_row (ambient_dim,) -> (n_spheres, 3)
    sphere_radii: jnp.ndarray    # (n_spheres,)
    active_mask: Optional[jnp.ndarray] = None    # see spline_jax.eval


class SplineParams(NamedTuple):
    """The differentiated per-agent decision variables -- exactly the
    (times, wps, vs) arrays CubicConfigurationSpline::set takes (see
    spline_jax.eval), i.e. GraphTimingMPC's view_time_deltas_list (as
    absolute cumulative times)/view_wps_list/view_vs_list. Freeze
    `times` for the smaller first cut the plan's open questions flag (no
    -v(t) cross-segment coupling) via `freeze_times` below."""
    times: jnp.ndarray
    wps: jnp.ndarray
    vs: jnp.ndarray


def freeze_times(params: SplineParams) -> SplineParams:
    """Stops gradient flow through `params.times`, matching the plan's
    noted safe first-cut simplification (optimize wps/vs only).

    MUST be called on the SAME traced value `jax.grad`/`jax.value_and_grad`
    is differentiating -- i.e. from inside the function being
    differentiated (`cost_fn(params) = base_cost_fn([freeze_times(p) for p
    in params])`), or composed into `build_cost_fn`'s own `freeze_times=`
    flag below. Calling this on `params` BEFORE handing it to
    `jax.grad(cost_fn, argnums=0)` as the argnums=0 value itself is a
    no-op: `jax.lax.stop_gradient` only severs the specific edge it sits on
    within the trace it's called in, and a concrete array passed in as a
    differentiation variable carries no memory of an earlier stop_gradient
    -- `jax.grad` will still return that leaf's real, non-zero gradient.
    (Verified empirically: `grad(cost_fn)(freeze_times(params))` and
    `grad(cost_fn)(params)` are numerically identical -- both nonzero --
    while only `grad(lambda p: cost_fn([freeze_times(x) for x in
    p]))(params)` actually gives 0.)"""
    return params._replace(times=jax.lax.stop_gradient(params.times))


# build_cost_fn's own `freeze_times` PARAMETER shadows this function inside
# its body -- this alias is what that body actually calls.
_freeze_times = freeze_times


def sample_configs(geom: AgentGeometry, params: SplineParams, t_grid: jnp.ndarray,
                    active_mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """One agent's ambient-space samples on the shared time grid -- (T,
    ambient_dim(geom.spec)). `active_mask`, if given, OVERRIDES
    `geom.active_mask` -- the cross-cycle refinement orchestration below
    needs this to vary CALL to call (the timing solve's own node-bridging
    pattern changes cycle to cycle) without rebuilding `geom` -- see
    CollisionRefiner's own doc comment for why that distinction matters
    for jit reuse. Plain one-off callers (e.g. test_collision_cost_jax.py)
    can keep using the simpler `geom.active_mask`-only form."""
    mask = geom.active_mask if active_mask is None else active_mask
    q, _v, _a = spline_jax.eval(
        geom.spec, params.times, params.wps, params.vs, t_grid,
        active_mask=mask)
    return q


def sphere_centers(geom: AgentGeometry, q_samples: jnp.ndarray) -> jnp.ndarray:
    """(T, ambient_dim) -> (T, n_spheres, 3) via geom.fk_fn, vmapped over
    the T samples (fk_fn itself operates on one ambient-space row)."""
    return jax.vmap(geom.fk_fn)(q_samples)


def inter_agent_cost(centers_list: Sequence[jnp.ndarray],
                      radii_list: Sequence[jnp.ndarray],
                      eps: float = 1e-9) -> jnp.ndarray:
    """sum_{a<b} relu(R_a + R_b - dist(C_a, C_b))^2 over every sphere pair
    and every shared-grid time sample -- `centers_list[i]` is (T, s_i, 3),
    `radii_list[i]` is (s_i,). One vectorized broadcast per agent pair
    (agent count is small -- a handful -- so the Python double loop over
    pairs unrolls into a fixed, still-jittable graph; the actual
    O(spheres^2 * T) cost is the broadcast inside each pair, not the loop).
    `eps` keeps the distance gradient finite when two centers coincide
    exactly (measure-zero in practice, but not worth a NaN over)."""
    total = jnp.asarray(0.0)
    n = len(centers_list)
    for a in range(n):
        for b in range(a + 1, n):
            ca, cb = centers_list[a], centers_list[b]
            ra, rb = radii_list[a], radii_list[b]
            diff = ca[:, :, None, :] - cb[:, None, :, :]  # (T, s_a, s_b, 3)
            dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1) + eps)
            overlap = jax.nn.relu(ra[None, :, None] + rb[None, None, :] - dist)
            total = total + jnp.sum(overlap * overlap)
    return total


def environment_cost(speed_fn: Callable[[jnp.ndarray], jnp.ndarray],
                      q_samples: jnp.ndarray,
                      eps: float = 1e-6) -> jnp.ndarray:
    """sum -log(speed_fn(q) + eps) over one agent's samples, vmapped over
    T. Swap for the literal sphere-vs-obstacle-box analytic distance (see
    the plan) if this proves too flat near `speed_fn`'s min-speed floor."""
    speeds = jax.vmap(speed_fn)(q_samples)
    return -jnp.sum(jnp.log(speeds + eps))


def regularizer_cost(q_samples: jnp.ndarray, q0_samples: jnp.ndarray) -> jnp.ndarray:
    """w * sum (Q - Q0)^2 against a reference trajectory -- the weight
    itself is applied by the caller (build_cost_fn), not here."""
    diff = q_samples - q0_samples
    return jnp.sum(diff * diff)


class CostWeights(NamedTuple):
    inter_agent: float = 1.0
    environment: float = 1.0
    regularizer: float = 1.0


def build_cost_fn(geometries: Sequence[AgentGeometry],
                   t_grid: jnp.ndarray,
                   speed_fn: Callable[[jnp.ndarray], jnp.ndarray],
                   weights: CostWeights = CostWeights(),
                   freeze_times: bool = False):
    """Returns `cost_fn(params_list, q0_samples_list) -> (total, breakdown)`,
    the single forward pass Step 2 calls for `jax.value_and_grad` (`jit` is
    the caller's responsibility, per this module's own doc comment --
    typically `jax.jit(jax.value_and_grad(cost_fn, has_aux=True))`).
    `params_list`/`q0_samples_list` are per-agent, same order as
    `geometries`; `q0_samples_list[i]` is agent i's PRE-refinement samples
    on `t_grid` (`sample_configs` applied once, before any optimization
    step, held fixed as the regularizer's reference). Optional
    `active_masks_list` (per-agent, same order) OVERRIDES each geometry's
    own `active_mask` -- see `sample_configs`'s doc comment; omit it (or
    pass a per-agent None) to fall back to `geometries[i].active_mask`.

    `freeze_times=True` applies the module-level `freeze_times` INSIDE
    `cost_fn`'s own body, so `jax.grad(cost_fn, argnums=0)(params_list,
    ...)[i].times` is genuinely 0 -- the foolproof way to get that, since
    calling the standalone `freeze_times` on `params_list` yourself before
    handing it to `jax.grad` does NOT work (see that function's own doc
    comment: stop_gradient only severs an edge inside the trace it runs
    in, and there's no trace yet at that point)."""

    def cost_fn(params_list: Sequence[SplineParams],
                q0_samples_list: Sequence[jnp.ndarray],
                active_masks_list: Optional[Sequence[Optional[jnp.ndarray]]] = None):
        if freeze_times:
            params_list = [_freeze_times(p) for p in params_list]
        if active_masks_list is None:
            active_masks_list = [None] * len(geometries)
        q_samples_list = [sample_configs(g, p, t_grid, am)
                           for g, p, am in zip(geometries, params_list, active_masks_list)]
        centers_list = [sphere_centers(g, q)
                         for g, q in zip(geometries, q_samples_list)]
        radii_list = [g.sphere_radii for g in geometries]

        inter = inter_agent_cost(centers_list, radii_list)
        env = sum(environment_cost(speed_fn, q) for q in q_samples_list)
        reg = sum(regularizer_cost(q, q0)
                  for q, q0 in zip(q_samples_list, q0_samples_list))

        total = (weights.inter_agent * inter
                 + weights.environment * env
                 + weights.regularizer * reg)
        return total, {"inter_agent": inter, "environment": env, "regularizer": reg}

    return cost_fn


# ==========================================================================
# Orchestration: extracting one cycle's raw GraphTimingMPC/TracedTimingMPC
# arrays, running a jax.lax.scan refinement pass over them, and writing the
# result back into the CubicConfigurationSpline objects
# GraphOfConstraintsMPC.step() already built (self.last_cycle_splines) --
# what goc_mpc.py's _solve_for_timing calls when configured with a
# CollisionRefinementConfig (opt-in, mirrors short_path_mpc=None's own
# pattern: absent by default, nothing about this module runs otherwise).
# ==========================================================================

class CollisionRefinementConfig(NamedTuple):
    """fk_fns/sphere_radii: one entry per agent, same order as
    graph._robot_specs (q_row (ambient_dim,) -> (n_spheres, 3) world-frame
    sphere centers; (n_spheres,) radii) -- real robot geometry, supplied by
    the caller (see collision_cost_jax's own module doc comment for why
    this stays out of goc-mpc itself). speed_fn is shared across agents
    (one environment). weights/n_steps/step_size/t_samples default to the
    values empirically tuned against the real b1_multiroom experiment
    (project_collision_aware_timing_jax memory): regularizer and
    environment both 0 -- the timing layer's own splines are each agent's
    INDIVIDUALLY optimal route, not a jointly-optimal multi-agent one (the
    plan's own motivation), so penalizing distance from them fights this
    pass's whole purpose; inter_agent=8 with a small fixed step reliably
    resolves the large majority of inter-agent overlap within a few
    hundred steps on that scene -- 25 is deliberately much smaller (a
    per-cycle NUDGE under a receding-horizon replan budget, not a
    from-scratch solve; see n_steps' own note below).

    `max_nodes`: a FIXED per-agent knot-row buffer size, padded up to by
    `extract_agent_refinement_data` every cycle (see `_pad_stationary`) so
    every agent's arrays have the SAME shape every cycle regardless of how
    many knots `timing_mpc.get_agent_spline_length()` actually produced
    this time (which varies often -- agents progress through the graph,
    dense interior points appear/disappear as the traced path is
    re-simplified) -- this is what lets `CollisionRefiner` compile its
    scan ONCE and reuse it every controller.step() cycle instead of
    retracing whenever any agent's knot count changes. Must be >= the
    largest `get_agent_spline_length()` this config's graph/scene can ever
    produce for any single agent; `extract_agent_refinement_data` raises
    (not silently truncates -- truncating would silently drop the tail of
    an agent's trajectory from collision consideration) if a cycle ever
    exceeds it. 64 is a generous default for a handful of rooms' worth of
    RDP-simplified dense waypoints (b1_multiroom's own traced paths stay
    well under that); pick a real bound for your own scene rather than
    trusting this blindly."""
    fk_fns: Sequence[Callable[[jnp.ndarray], jnp.ndarray]]
    sphere_radii: Sequence[jnp.ndarray]
    speed_fn: Callable[[jnp.ndarray], jnp.ndarray]
    weights: CostWeights = CostWeights(inter_agent=8.0, environment=0.0, regularizer=0.0)
    # A handful of steps per controller.step() cycle, not a run-to-
    # convergence solve -- this pass reruns every cycle on the freshly
    # re-timed splines (no warm start across cycles yet -- seeding
    # `params` below from the PREVIOUS cycle's refined result, time-
    # shifted, is the natural next step the plan itself notes, not done
    # here), so partial progress each cycle compounds over the receding
    # horizon rather than needing to fully resolve overlap in one shot.
    n_steps: int = 25
    step_size: float = 3e-4
    t_samples: int = 150
    max_nodes: int = 64


def _cumsum_with_zero(deltas: np.ndarray, n: int) -> np.ndarray:
    """Python port of timing_gn_layout.cpp's CumsumWithZero(x, n): n+1
    absolute times from the first n deltas, times[0] == 0."""
    deltas = np.asarray(deltas)[:n]
    y = np.zeros(n + 1)
    y[1:] = np.cumsum(deltas)
    return y


# Eigen::MatrixXi (configuration_spline.hpp's knot_block_active_) always
# comes back through pybind as int32, confirmed directly (CubicConfiguration
# Spline.block_active_mask() -- both the empty-matrix and populated case).
# Every FABRICATED (not read off a real spline) active_mask below must
# match this dtype exactly, not just numerically equal it: with
# jax_enable_x64 True (which `eikonax` -- imported for b1_multiroom's
# default "sweep" edge-cost objective, among others -- flips on PROCESS-
# WIDE the moment it's first imported), an int64 numpy array survives
# `jnp.array(...)` as int64, while the real (non-fabricated) mask stays
# int32 -- a genuine per-agent pytree-leaf DTYPE mismatch the instant an
# agent alternates between "real mask this cycle" and "fabricated
# (empty-mask fallback or parked) this cycle", which forces CollisionRefiner's
# compiled scan to retrace. Confirmed as the actual cause of intermittent
# per-cycle timing spikes seen running the full b1_multiroom experiment.
_ACTIVE_MASK_DTYPE = np.int32

# Arbitrary but deliberately not tiny -- matches fill_cubic_splines' own
# dummy-spline convention of parking an agent and "coming to a stop after 1
# second" (graph_timing_mpc.cpp), and keeps every padded knot-interval's
# duration comfortably away from any float-precision edge case. The exact
# value doesn't affect correctness: every padded segment has zero
# displacement and zero endpoint velocity by construction (see
# _pad_stationary), so it evaluates to the same parked configuration
# everywhere in the padded range regardless of how that range is spaced,
# and spline_jax.eval clamps anything past the padded array's own last
# knot to that same parked value too.
_PAD_DT = 1.0


def _pad_stationary(times: np.ndarray, wps: np.ndarray, vs: np.ndarray,
                     active_mask: np.ndarray, max_nodes: int):
    """Pads (times, wps, vs, active_mask) from their real `n` rows out to
    exactly `max_nodes` rows with a STATIONARY continuation anchored at the
    real last knot -- position held at the last waypoint, velocity 0,
    times strictly increasing by `_PAD_DT` per pad row (spline_jax.eval
    divides by each piece's own duration, so pad rows must stay strictly
    increasing -- a zero-length segment would divide by zero; a
    zero-DISPLACEMENT, zero-velocity one, which this is, evaluates to
    exactly 0/tau^k = 0 safely for any tau > 0).

    This is not merely a shape trick: a real agent's OWN spline, sampled
    past its own horizon, already looks exactly like this (eval's clamping
    alone would return the same parked value) -- padding exists only so
    every agent's arrays share one fixed `max_nodes` shape ACROSS an MPC
    cycle where per-agent knot counts otherwise vary, letting
    `CollisionRefiner` compile its scan once and reuse it every cycle
    instead of retracing whenever any agent's knot count changes.

    No-ops (returns the inputs unchanged) if `n == max_nodes` already;
    raises if `n > max_nodes` (see CollisionRefinementConfig.max_nodes'
    own doc comment -- silently truncating would silently drop the tail of
    an agent's trajectory from collision consideration, which this
    codebase's own convention is to fail loud on instead)."""
    n = times.shape[0]
    if n > max_nodes:
        raise ValueError(
            f"collision_cost_jax: agent spline has {n} knots, exceeding "
            f"CollisionRefinementConfig.max_nodes={max_nodes} -- raise "
            "max_nodes rather than silently truncating collision-relevant "
            "trajectory data.")
    pad = max_nodes - n
    if pad == 0:
        return times, wps, vs, active_mask
    pad_times = times[-1] + _PAD_DT * np.arange(1, pad + 1)
    times_p = np.concatenate([times, pad_times])
    wps_p = np.concatenate([wps, np.tile(wps[-1], (pad, 1))], axis=0)
    vs_p = np.concatenate([vs, np.zeros((pad, vs.shape[1]))], axis=0)
    active_mask_p = np.concatenate(
        [active_mask, np.ones((pad, active_mask.shape[1]), dtype=active_mask.dtype)], axis=0)
    return times_p, wps_p, vs_p, active_mask_p


def _pad_free_mask(mask: np.ndarray, max_nodes: int) -> np.ndarray:
    """Pads a (n,) bool free-mask out to (max_nodes,), False (frozen) at
    every pad row -- a padded knot is a fabricated stationary continuation,
    never a real decision variable."""
    n = mask.shape[0]
    if n == max_nodes:
        return mask
    return np.concatenate([mask, np.zeros(max_nodes - n, dtype=bool)])


class _AgentRefinementData(NamedTuple):
    agent_id: int
    n_real: int                 # true (unpadded) knot count; 0 == "parked/dummy, don't write back"
    real_end_time: float        # times[n_real - 1] before padding (1.0 for the parked/dummy case)
    active_mask: jnp.ndarray    # (max_nodes, len(spec)) -- ALWAYS concrete, never None (jit pytree stability)
    params: SplineParams        # times/wps/vs padded to (max_nodes, ...)
    wps_free_mask: np.ndarray   # (max_nodes,) bool -- see extract_agent_refinement_data
    vs_free_mask: np.ndarray    # (max_nodes,) bool
    times_free_mask: np.ndarray  # (max_nodes,) bool


def extract_agent_refinement_data(timing_mpc, graph, spline, agent_id: int,
                                   x: np.ndarray, x_dot: np.ndarray,
                                   max_nodes: int) -> _AgentRefinementData:
    """Rebuilds agent `agent_id`'s (spec, active_mask, SplineParams) exactly
    as GraphTimingMPC::fill_cubic_splines (graph_timing_mpc.cpp) assembled
    them for GraphOfConstraintsMPC.last_cycle_splines[agent_id] THIS cycle
    -- i.e. the literal decision variables the timing solve produced,
    x0/v0-prepended, not a re-fit -- plus the three per-row freeze masks
    this refinement pass must respect:

    - `wps_free_mask`: False at row 0 (x0, the agent's CURRENT position --
      a live boundary condition, not a decision variable at any layer) and
      at every row `timing_mpc.view_agent_nodes_list()[agent_id]` marks
      with a REAL graph-node id (a waypoint `waypoint_mpc` actually solved
      for -- moving it here would silently disagree with the waypoint
      solve). True only at a SYNTHETIC (-1) row -- a dense interior point
      `TracedTimingMPC` invented by tracing between real nodes, which is
      exactly what this refinement pass exists to reshape.
    - `vs_free_mask`: False only at row 0 (v0, the current measured
      velocity -- also a live boundary condition) and the last row (the
      terminal velocity fill_cubic_splines always forces to exactly 0 --
      preserving "come to rest at the final waypoint" is not something
      this pass should silently relax). True everywhere else, INCLUDING
      real-node rows: unlike position, the velocity the timing solve
      passes through a real waypoint at was never a `waypoint_mpc`
      decision, so it isn't protected by the same "don't disagree with
      waypoint_mpc" concern the position mask is about.
    - `times_free_mask`: False only at row 0 (times[0] == 0 by
      construction -- CubicConfigurationSpline's own relative-time
      convention, begin() == 0, that other code depends on). True
      everywhere else, at every row real or synthetic: arrival TIME at a
      real waypoint was never a `waypoint_mpc` decision either, and
      leaving it free is the whole "wait/hurry" mechanism this plan is
      about (verified empirically to matter: project_collision_aware_
      timing_jax memory).

    `spline` is `controller.last_cycle_splines[agent_id]` -- its own
    `block_active_mask()` (set by fill_cubic_splines just before this
    runs) is read back as-is, not recomputed.

    Valid immediately after a `controller.step(t, x, x_dot)` call using
    the SAME `x`/`x_dot` (`v0_for_spline == x_dot` exactly on the
    controller's first-ever step; goc_mpc.py's own `_solve_for_timing`
    passes `v0_for_spline` here for every cycle, matching what
    `fill_cubic_splines` was actually called with).

    ALWAYS returns a value now (never None): an agent with no dense/real
    spline this cycle (spline_length <= 1 -- fill_cubic_splines' own
    dummy-spline branch, "stays at x0 and comes to a stop after 1 second")
    gets a PARKED entry instead -- the same dummy spline fill_cubic_splines
    itself would present, fully frozen (every free-mask False, `n_real=0`
    so `CollisionRefiner` skips writing it back). This is deliberate, not
    just a shape-uniformity concession: a parked/finished agent is still a
    real physical obstacle other agents' free waypoints should route
    around, so including it as a static point is MORE correct than
    dropping it from consideration entirely (the previous behavior).

    Every returned array is padded to exactly `max_nodes` rows (see
    `_pad_stationary`/`_pad_free_mask`) regardless of the real per-agent
    knot count `n` -- this is what lets the SAME compiled scan run every
    cycle; `_AgentRefinementData.n_real` carries the true count for
    write-back trimming.
    """
    spec = graph._robot_specs[agent_id]
    n = timing_mpc.get_agent_spline_length(agent_id)
    lo, hi = graph.agent_col_offsets[agent_id], graph.agent_col_offsets[agent_id + 1]
    x0_i = np.asarray(x[lo:hi])
    v0_i = np.asarray(x_dot[lo:hi])
    d = hi - lo
    num_blocks = len(spec)

    if n <= 1:
        # Mirrors fill_cubic_splines' own dummy-spline branch exactly
        # (graph_timing_mpc.cpp): parked at x0, decelerating to a stop
        # over 1 second, then (via padding below) parked forever after.
        times = np.array([0.0, 1.0])
        wps = np.vstack([x0_i[None, :], x0_i[None, :]])
        vs = np.vstack([v0_i[None, :], np.zeros((1, d))])
        active_mask = np.ones((2, num_blocks), dtype=_ACTIVE_MASK_DTYPE)
        times_p, wps_p, vs_p, active_mask_p = _pad_stationary(times, wps, vs, active_mask, max_nodes)
        wps_free = _pad_free_mask(np.zeros(2, dtype=bool), max_nodes)
        vs_free = _pad_free_mask(np.zeros(2, dtype=bool), max_nodes)
        times_free = _pad_free_mask(np.zeros(2, dtype=bool), max_nodes)
        params = SplineParams(times=jnp.array(times_p), wps=jnp.array(wps_p), vs=jnp.array(vs_p))
        return _AgentRefinementData(agent_id, 0, 1.0, jnp.array(active_mask_p),
                                     params, wps_free, vs_free, times_free)

    wps_rest = np.asarray(timing_mpc.view_wps_list()[agent_id])
    vs_rest = np.asarray(timing_mpc.view_vs_list()[agent_id])
    deltas = np.asarray(timing_mpc.view_time_deltas_list()[agent_id])
    agent_nodes = timing_mpc.view_agent_nodes_list()[agent_id]
    assert wps_rest.shape[0] == n - 1 and vs_rest.shape[0] == n - 2 and len(agent_nodes) == n - 1, (
        f"extract_agent_refinement_data: agent {agent_id}'s raw GraphTimingMPC arrays "
        f"don't match spline_length={n} (wps={wps_rest.shape[0]}, vs={vs_rest.shape[0]}, "
        f"nodes={len(agent_nodes)}) -- is `timing_mpc` mid-cycle (call right after "
        f"fill_cubic_splines, before anything else touches it)?")

    wps = np.vstack([x0_i[None, :], wps_rest])
    vs = np.vstack([v0_i[None, :], vs_rest, np.zeros((1, d))])
    times = _cumsum_with_zero(deltas, n - 1)
    real_end_time = float(times[-1])

    wps_free = np.array([False] + [node == -1 for node in agent_nodes])
    vs_free = np.ones(n, dtype=bool)
    vs_free[0] = False
    vs_free[-1] = False
    times_free = np.ones(n, dtype=bool)
    times_free[0] = False

    raw_mask = np.asarray(spline.block_active_mask())
    # Always a concrete array, never None -- see spline_jax.eval's own doc
    # comment ("pass an explicit all-ones array rather than None under
    # jax.jit") and _AgentRefinementData.active_mask's own field comment:
    # a None-vs-array pytree-structure flip across cycles would itself
    # force a retrace even with every array's SHAPE otherwise fixed. The
    # fabricated fallback's dtype must match `raw_mask`'s own (see
    # _ACTIVE_MASK_DTYPE's own comment) -- NOT a numerically-equal but
    # differently-typed array, which would retrace just the same.
    active_mask = raw_mask if raw_mask.size else np.ones((n, num_blocks), dtype=_ACTIVE_MASK_DTYPE)

    times_p, wps_p, vs_p, active_mask_p = _pad_stationary(times, wps, vs, active_mask, max_nodes)
    wps_free_p = _pad_free_mask(wps_free, max_nodes)
    vs_free_p = _pad_free_mask(vs_free, max_nodes)
    times_free_p = _pad_free_mask(times_free, max_nodes)
    params = SplineParams(times=jnp.array(times_p), wps=jnp.array(wps_p), vs=jnp.array(vs_p))

    return _AgentRefinementData(agent_id, n, real_end_time, jnp.array(active_mask_p),
                                 params, wps_free_p, vs_free_p, times_free_p)


def _run_scan(cost_fn, params_list: Sequence[SplineParams],
              q0_samples_list: Sequence[jnp.ndarray],
              active_masks_list: Sequence[jnp.ndarray],
              free_masks: Sequence[jnp.ndarray],
              step_size: float, n_steps: int):
    """The actual optimization loop, as one `jax.lax.scan` -- not a Python
    `for` loop over `n_steps` calls to a jitted step function (which would
    trace/compile `n_steps` times over, or unroll into an `n_steps`-times-
    larger XLA graph if jitted as a whole): `scan` compiles the step body
    ONCE and loops it on-device, the efficient form for "a fixed handful of
    gradient steps" this plan's own Step 2 section asks for.

    `active_masks_list[i]` is agent i's (fixed for the duration of this
    scan) block-active mask, threaded through to `cost_fn` on every step --
    not baked into `cost_fn`'s own closure, since (unlike `geometries`)
    it's the one piece of "static-seeming" per-agent data that actually
    DOES change cycle to cycle at the `CollisionRefiner` call site.

    `free_masks[i]` is `(wps_free, vs_free, times_free)` for agent i (each
    a bool array over that agent's own knot rows, see
    extract_agent_refinement_data's own doc comment) -- multiplied
    elementwise into that agent's gradient every step, so a frozen row's
    value is provably invariant across the whole scan (its gradient is
    exactly 0 on every iteration), not just "moved back afterward"."""
    grad_fn = jax.value_and_grad(cost_fn, argnums=0, has_aux=True)

    def step(carry, _):
        (total, bd), grads = grad_fn(carry, q0_samples_list, active_masks_list)
        new_carry = [
            SplineParams(
                times=p.times - step_size * g.times * fm[2],
                wps=p.wps - step_size * g.wps * fm[0][:, None],
                vs=p.vs - step_size * g.vs * fm[1][:, None],
            )
            for p, g, fm in zip(carry, grads, free_masks)
        ]
        return new_carry, (total, bd)

    final_params, (cost_history, bd_history) = jax.lax.scan(
        step, params_list, xs=None, length=n_steps)
    return final_params, cost_history, bd_history


def _make_compiled_refiner(geometries: Sequence[AgentGeometry],
                            speed_fn: Callable[[jnp.ndarray], jnp.ndarray],
                            weights: CostWeights, n_steps: int, step_size: float,
                            t_samples: int):
    """Builds and jits the WHOLE per-cycle refinement step exactly once:
    `run(params_list, active_masks_list, free_masks, t_max) -> (final_params,
    initial_bd, final_bd)`. `geometries`/`speed_fn`/`weights`/`n_steps`/
    `step_size`/`t_samples` are closed over as compile-time constants
    (true for the lifetime of one `CollisionRefiner` -- fk_fns/sphere_radii/
    speed_fn/weights/n_steps/step_size/t_samples all come from one
    unchanging `CollisionRefinementConfig`, and `geometries`' `spec`s come
    from `graph._robot_specs`, fixed for the graph's lifetime too).
    EVERYTHING that legitimately varies cycle to cycle -- knot values,
    each agent's active_mask (the timing solve's own node-bridging pattern
    changes cycle to cycle), which rows are free to move, and how far out
    the shared time grid needs to reach -- is a real argument to `run`,
    not baked into a closure, which is what makes reusing this ONE
    `jax.jit` object across calls actually correct (not just shape-
    compatible by accident): a value closed over at first-trace time would
    silently stay frozen at that first cycle's value forever after."""

    def run(params_list, active_masks_list, free_masks, t_max):
        t_grid = jnp.linspace(0.0, t_max, t_samples)
        cost_fn = build_cost_fn(geometries, t_grid, speed_fn, weights)
        q0_samples_list = [sample_configs(g, p, t_grid, am)
                           for g, p, am in zip(geometries, params_list, active_masks_list)]
        _, initial_bd = cost_fn(params_list, q0_samples_list, active_masks_list)
        final_params, _cost_history, bd_history = _run_scan(
            cost_fn, params_list, q0_samples_list, active_masks_list,
            free_masks, step_size, n_steps)
        final_bd = jax.tree_util.tree_map(lambda v: v[-1], bd_history)
        return final_params, initial_bd, final_bd

    return jax.jit(run)


class CollisionRefiner:
    """Top-level Step 2 entry point, built ONCE per (graph,
    CollisionRefinementConfig) and reused every controller.step() cycle --
    the fix for "we can't re-jit every cycle": `_make_compiled_refiner` is
    called exactly once, in `__init__`, and every `.refine()` call reuses
    that same compiled `jax.jit` object. This only works because (a) every
    agent's per-cycle arrays are padded to a fixed `config.max_nodes`
    shape (`extract_agent_refinement_data`/`_pad_stationary`), so the
    shapes handed to the compiled function never change, and (b) the graph
    always has `graph.num_agents` agents in the same order every cycle --
    a finished/parked agent gets a degenerate PARKED entry instead of
    being dropped (see extract_agent_refinement_data's own doc comment),
    so the per-cycle agent LIST length is fixed too, not just each agent's
    array shapes.

    `jax.jit` itself is lazy -- constructing this does no tracing or
    compilation; that happens on the first `.refine()` call, and every
    call after reuses the result as long as `config.max_nodes`/
    `config.t_samples`/agent count don't change (they don't, for one
    `CollisionRefiner` instance's lifetime)."""

    def __init__(self, graph, config: CollisionRefinementConfig):
        self.graph = graph
        self.config = config
        self.geometries = [
            AgentGeometry(spec=graph._robot_specs[i], fk_fn=config.fk_fns[i],
                          sphere_radii=config.sphere_radii[i])
            for i in range(graph.num_agents)
        ]
        self._compiled = _make_compiled_refiner(
            self.geometries, config.speed_fn, config.weights,
            config.n_steps, config.step_size, config.t_samples)

    def refine(self, timing_mpc, splines, x: np.ndarray, x_dot: np.ndarray) -> Optional[dict]:
        """Extracts every agent's current-cycle spline data (padded to
        `config.max_nodes`), runs the cached jax.lax.scan collision-cost
        refinement, and writes the result back into `splines`
        (`controller.last_cycle_splines`) via `set_block_active_mask`/
        `set` -- trimmed back to each agent's REAL (unpadded) knot count,
        the same construction path `fill_cubic_splines` itself uses, so
        every downstream consumer (short-path solver included) sees the
        refined splines with no other code path needing to change. Call
        this immediately after `timing_mpc.fill_cubic_splines(splines, x,
        x_dot)`, with the SAME `x`/`x_dot` that call used.

        Returns a diagnostics dict (`initial`/`final` cost breakdowns,
        each `{"inter_agent", "environment", "regularizer"}`) for logging,
        or None if the graph has fewer than 2 agents total (nothing
        meaningful for inter-agent avoidance to ever do here)."""
        if self.graph.num_agents < 2:
            return None

        extracted = [extract_agent_refinement_data(
                        timing_mpc, self.graph, splines[i], i, x, x_dot, self.config.max_nodes)
                     for i in range(self.graph.num_agents)]

        params_list = [e.params for e in extracted]
        active_masks_list = [e.active_mask for e in extracted]
        free_masks = [(e.wps_free_mask, e.vs_free_mask, e.times_free_mask) for e in extracted]
        t_max = max(e.real_end_time for e in extracted)

        final_params, initial_bd, final_bd = self._compiled(
            params_list, active_masks_list, free_masks, t_max)

        for e, p in zip(extracted, final_params):
            if e.n_real <= 1:
                continue  # parked/dummy this cycle -- nothing real to write back
            n = e.n_real
            splines[e.agent_id].set_block_active_mask(np.asarray(e.active_mask[:n]))
            splines[e.agent_id].set(np.asarray(p.wps[:n]), np.asarray(p.vs[:n]), np.asarray(p.times[:n]))

        return {"initial": {k: float(v) for k, v in initial_bd.items()},
                "final": {k: float(v) for k, v in final_bd.items()}}


def refine_splines(timing_mpc, graph, splines, x: np.ndarray, x_dot: np.ndarray,
                    config: CollisionRefinementConfig) -> Optional[dict]:
    """One-shot convenience wrapper over `CollisionRefiner` for a single
    call site that doesn't need cross-cycle reuse (e.g. a one-off
    visualization script). Builds (and immediately discards) a fresh
    `CollisionRefiner`, so it pays the one-time jit trace/compile on EVERY
    call -- `GraphOfConstraintsMPC` does NOT use this; it holds its own
    `CollisionRefiner` instance across cycles instead (see goc_mpc.py's
    own constructor)."""
    return CollisionRefiner(graph, config).refine(timing_mpc, splines, x, x_dot)

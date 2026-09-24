"""Validates goc_mpc.collision_cost_jax's Step-2 optimization loop
(`_run_scan`/`extract_agent_refinement_data`/`CollisionRefiner`), the
production wiring `_solve_for_timing` calls when configured with a
`CollisionRefinementConfig`. Four parts:

1. A white-box check of `_run_scan`'s freeze-mask handling on synthetic
   data: rows marked NOT free must come out of the scan bit-identical to
   what went in (every iteration, not just the last), and rows marked free
   must actually move under a nonzero gradient.
2. An end-to-end smoke test: a real 2-agent `GraphOfConstraintsMPC` +
   `TracedTimingMPC` (so real dense/synthetic interior waypoints actually
   exist, matching production), stepped once, refined via
   `collision_cost_jax.refine_splines`, checking inter-agent cost actually
   falls and that every agent's CURRENT position/velocity (row 0, always a
   live boundary condition) and each real waypoint's POSITION (frozen,
   per this session's "don't move what waypoint_mpc solved for" ask) are
   exactly preserved in the refined splines.
3. The same, but through the real `GraphOfConstraintsMPC.collision_refinement=`
   constructor/`_solve_for_timing` wiring.
4. `CollisionRefiner` reuses ONE compiled scan across cycles whose agents
   have different REAL knot counts (padded to a fixed `max_nodes`) rather
   than retracing every cycle -- the fix for "we can't re-jit every
   cycle" (see CollisionRefiner's own doc comment).

Run: python examples/test_collision_refinement_loop.py
"""

import numpy as np
import jax.numpy as jnp

from goc_mpc import GraphOfConstraints, GraphOfConstraintsMPC
from goc_mpc._ext.configuration_spline import CubicConfigurationSpline
from goc_mpc.splines import Block
from goc_mpc.traced_timing_mpc import TracedTimingMPC
from goc_mpc import collision_cost_jax as ccj
from goc_mpc.collision_cost_jax import (
    AgentGeometry, SplineParams, CostWeights, CollisionRefinementConfig,
    build_cost_fn, sample_configs,
)


def check_scan_mask_invariance():
    n0, n1 = 5, 6
    rng = np.random.default_rng(0)

    def make_params(n, seed):
        r = np.random.default_rng(seed)
        times = jnp.array(np.sort(r.uniform(0, 5, n)))
        wps = jnp.array(r.uniform(-3, 3, (n, 2)))
        vs = jnp.array(r.uniform(-1, 1, (n, 2)))
        return SplineParams(times=times, wps=wps, vs=vs)

    p0, p1 = make_params(n0, 1), make_params(n1, 2)

    def fk(q):
        return jnp.array([[q[0], q[1], 0.0]])

    geom0 = AgentGeometry(spec=[Block.R(2)], fk_fn=fk, sphere_radii=jnp.array([0.8]))
    geom1 = AgentGeometry(spec=[Block.R(2)], fk_fn=fk, sphere_radii=jnp.array([0.8]))
    t_grid = jnp.linspace(0.0, 5.0, 30)
    q0_list = [sample_configs(geom0, p0, t_grid), sample_configs(geom1, p1, t_grid)]
    weights = CostWeights(inter_agent=8.0, environment=0.0, regularizer=0.0)
    cost_fn = build_cost_fn([geom0, geom1], t_grid, lambda q: 1.0, weights)

    # Deliberately-mixed mask per agent so both free and frozen rows exist
    # for wps/vs/times independently.
    wps_free0 = np.array([False, True, False, True, False])
    vs_free0 = np.array([False, True, True, True, False])
    times_free0 = np.array([False, True, True, True, True])
    wps_free1 = np.array([False, True, True, False, True, False])
    vs_free1 = np.array([False, True, True, True, True, False])
    times_free1 = np.array([False, True, True, True, True, True])
    free_masks = [
        (jnp.array(wps_free0), jnp.array(vs_free0), jnp.array(times_free0)),
        (jnp.array(wps_free1), jnp.array(vs_free1), jnp.array(times_free1)),
    ]

    n_steps = 25
    active_masks_list = [None, None]
    final_params, _cost_hist, _bd_hist = ccj._run_scan(
        cost_fn, [p0, p1], q0_list, active_masks_list, free_masks, step_size=1e-3, n_steps=n_steps)

    ok = True
    for label, p_before, p_after, (wf, vf, tf) in zip(
            ["agent0", "agent1"], [p0, p1], final_params, free_masks):
        frozen_wps_ok = bool(jnp.all(p_after.wps[~np.asarray(wf)] == p_before.wps[~np.asarray(wf)]))
        frozen_vs_ok = bool(jnp.all(p_after.vs[~np.asarray(vf)] == p_before.vs[~np.asarray(vf)]))
        frozen_times_ok = bool(jnp.all(p_after.times[~np.asarray(tf)] == p_before.times[~np.asarray(tf)]))
        moved_wps = bool(jnp.any(p_after.wps[np.asarray(wf)] != p_before.wps[np.asarray(wf)]))
        print(f"[{label}] frozen wps unchanged: {'OK' if frozen_wps_ok else 'FAIL'}; "
              f"frozen vs unchanged: {'OK' if frozen_vs_ok else 'FAIL'}; "
              f"frozen times unchanged: {'OK' if frozen_times_ok else 'FAIL'}; "
              f"free wps actually moved: {'OK' if moved_wps else 'FAIL'}")
        ok &= frozen_wps_ok and frozen_vs_ok and frozen_times_ok and moved_wps
    return ok


def build_two_agent_graph():
    """Two Block.R(2) agents, each with one real interior waypoint node
    (start -> mid -> goal), on paths crossing near the middle -- mirrors
    test_collision_cost_jax.py's synthetic scenario but through a REAL
    GraphOfConstraints/TracedTimingMPC solve instead of hand-built
    SplineParams, so dense/synthetic (-1) interior rows genuinely exist."""
    spec = Block.R(2)
    graph = GraphOfConstraints(
        [[spec], [spec]], [],
        state_lower_bound=-10.0, state_upper_bound=10.0,
        robot_names=["a0", "a1"], object_names=[],
        workspace_dim=2,
    )
    graph.structure.add_nodes(4)  # nodes 0,1: agent0 mid,goal; 2,3: agent1 mid,goal
    graph.structure.add_edge(0, 1, True)
    graph.structure.add_edge(2, 3, True)

    starts = {0: np.array([0.0, 0.0]), 1: np.array([4.0, 0.0])}
    mids = {0: np.array([2.0, 0.0]), 1: np.array([2.0, 0.0])}
    goals = {0: np.array([4.0, 0.0]), 1: np.array([0.0, 0.0])}

    q0, q1 = graph.agent_q(0), graph.agent_q(1)
    for k in range(2):
        graph.add_constraint(0, q0[k] == float(mids[0][k]))
        graph.add_constraint(1, q0[k] == float(goals[0][k]))
        graph.add_constraint(2, q1[k] == float(mids[1][k]))
        graph.add_constraint(3, q1[k] == float(goals[1][k]))
    return graph, [spec, spec], starts, goals


class LinearTracer:
    """Minimal field.trace_path stand-in: a ZIGZAG path -- each interior
    sample alternates a perpendicular offset sign, so consecutive points
    are never colinear and TracedTimingMPC's default RDP simplification
    keeps (approximately) all `n_interior` of them, scaling the real
    per-agent knot count with `n_interior` (a single bow at the midpoint,
    tried first, doesn't do this: each of its two halves is perfectly
    straight, so RDP -- correctly -- collapses every interior point on
    each half down to just its shared endpoint, giving the SAME real knot
    count regardless of `n_interior`)."""

    def __init__(self, n_interior=4, bow=0.5):
        self.n_interior = n_interior
        self.bow = bow

    def trace_path(self, agent, start, goal):
        del agent
        start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
        chord = goal - start
        length = np.linalg.norm(chord)
        if length < 1e-9:
            return start[None, :]
        perp = np.array([-chord[1], chord[0]]) / length
        s = np.linspace(0.0, 1.0, self.n_interior + 2)
        signs = np.where(np.arange(len(s)) % 2 == 0, 1.0, -1.0)
        signs[0] = 0.0
        signs[-1] = 0.0  # endpoints pinned exactly to start/goal
        path = start[None, :] + s[:, None] * chord[None, :] + (self.bow * signs)[:, None] * perp[None, :]
        return path


def two_sphere_ish_fk(q):
    return jnp.array([[q[0], q[1], 0.0]])


def check_end_to_end():
    """Drives TracedTimingMPC directly (goc_mpc.collision_cost_jax.
    refine_splines only needs a timing_mpc/graph/splines triple, not a
    full GraphOfConstraintsMPC) -- mirrors
    traced_timing_obstacle_avoidance.py's own compare_chord_vs_traced()
    pattern, extended to 2 agents, so this test doesn't depend on a
    waypoint-MPC backend (Gurobi/IPOPT) being available/licensed at all."""
    graph, specs, starts, goals = build_two_agent_graph()
    field = LinearTracer(n_interior=4)
    splines = [CubicConfigurationSpline([s]) for s in specs]
    timing_mpc = TracedTimingMPC(graph, splines, field, time_cost=1.0)

    x = np.concatenate([starts[0], starts[1]])
    x_dot = np.zeros(4)
    total_dim = graph.agent_col_offsets[-1]
    waypoints = np.zeros((4, total_dim))
    waypoints[0, 0:2] = np.array([2.0, 0.0])   # agent0 mid
    waypoints[1, 0:2] = goals[0]                # agent0 goal
    waypoints[2, 2:4] = np.array([2.0, 0.0])   # agent1 mid
    waypoints[3, 2:4] = goals[1]                 # agent1 goal
    var_assignments = np.array([], dtype=np.int32)

    ok_solve = timing_mpc.solve(x, x_dot, [0, 1, 2, 3], waypoints, var_assignments)
    assert ok_solve, "TracedTimingMPC.solve failed"
    timing_mpc.fill_cubic_splines(splines, x, x_dot)

    # Confirm dense (-1) rows actually exist for at least one agent --
    # otherwise this test isn't exercising the thing it's meant to.
    any_dense = any(-1 in timing_mpc.view_agent_nodes_list()[i] for i in range(2))
    print(f"[setup] dense/synthetic interior rows present: {'OK' if any_dense else 'FAIL'}")

    radii = [jnp.array([0.6]), jnp.array([0.6])]
    config = CollisionRefinementConfig(
        fk_fns=[two_sphere_ish_fk, two_sphere_ish_fk], sphere_radii=radii,
        speed_fn=lambda q: 1.0,
        weights=CostWeights(inter_agent=8.0, environment=0.0, regularizer=0.0),
        n_steps=25, step_size=3e-4, t_samples=60,
    )

    # Snapshot each agent's row-0 (current x/v) and last-knot (goal)
    # position BEFORE refinement, straight off the freshly-fit splines.
    pre_x0 = [np.asarray(splines[i].eval(0.0)[0]) for i in range(2)]
    pre_goal = [np.asarray(splines[i].eval(splines[i].end())[0]) for i in range(2)]

    diagnostics = ccj.refine_splines(timing_mpc, graph, splines, x, x_dot, config)
    ok = diagnostics is not None
    print(f"[refine] ran: {'OK' if ok else 'FAIL (returned None)'}")
    if not ok:
        return False

    print(f"[refine] inter_agent cost: {diagnostics['initial']['inter_agent']:.4f} -> "
          f"{diagnostics['final']['inter_agent']:.4f}")
    improved = diagnostics["final"]["inter_agent"] < diagnostics["initial"]["inter_agent"]
    print(f"[refine] inter_agent cost decreased: {'OK' if improved else 'FAIL'}")
    ok &= improved

    post_x0 = [np.asarray(splines[i].eval(0.0)[0]) for i in range(2)]
    post_goal = [np.asarray(splines[i].eval(splines[i].end())[0]) for i in range(2)]
    for i in range(2):
        x0_ok = np.allclose(pre_x0[i], post_x0[i], atol=1e-10)
        goal_ok = np.allclose(pre_goal[i], post_goal[i], atol=1e-10)
        print(f"[refine] agent {i}: current position preserved: {'OK' if x0_ok else 'FAIL'}; "
              f"goal waypoint position preserved: {'OK' if goal_ok else 'FAIL'}")
        ok &= x0_ok and goal_ok

    return ok


class _StubWaypointMPC:
    """Just enough of GraphWaypointMPC's protocol for _solve_for_timing --
    duck-typed the same way an EvolutionaryWaypointSolver is (see
    GraphOfConstraintsMPC's own waypoint_mpc= doc comment) -- so this check
    exercises the REAL GraphOfConstraintsMPC.collision_refinement wiring
    (constructor + _solve_for_timing) without needing a licensed MILP
    solver just to resolve an already-known, hand-pinned waypoint matrix."""

    def __init__(self, waypoints, var_assignments):
        self._waypoints = waypoints
        self._var_assignments = var_assignments

    def view_waypoints(self):
        return self._waypoints

    def view_var_assignments(self):
        return self._var_assignments

    def view_t_by_node(self):
        return np.array([])


def check_controller_wiring():
    """GraphOfConstraintsMPC(collision_refinement=...) actually runs the
    refinement inside _solve_for_timing and records last_cycle_collision_
    refinement -- the real production call path this session wired up."""
    graph, specs, starts, goals = build_two_agent_graph()
    field = LinearTracer(n_interior=4)
    timing_mpc = TracedTimingMPC(graph, [CubicConfigurationSpline([s]) for s in specs],
                                  field, time_cost=1.0)

    x = np.concatenate([starts[0], starts[1]])
    x_dot = np.zeros(4)
    total_dim = graph.agent_col_offsets[-1]
    waypoints = np.zeros((4, total_dim))
    waypoints[0, 0:2] = np.array([2.0, 0.0])
    waypoints[1, 0:2] = goals[0]
    waypoints[2, 2:4] = np.array([2.0, 0.0])
    waypoints[3, 2:4] = goals[1]
    var_assignments = np.array([], dtype=np.int32)
    stub_waypoint_mpc = _StubWaypointMPC(waypoints, var_assignments)

    config = CollisionRefinementConfig(
        fk_fns=[two_sphere_ish_fk, two_sphere_ish_fk],
        sphere_radii=[jnp.array([0.6]), jnp.array([0.6])],
        speed_fn=lambda q: 1.0,
        weights=CostWeights(inter_agent=8.0, environment=0.0, regularizer=0.0),
        n_steps=25, step_size=3e-4, t_samples=60,
    )
    controller = GraphOfConstraintsMPC(
        graph, waypoint_mpc=stub_waypoint_mpc, timing_mpc=timing_mpc,
        collision_refinement=config, short_path_time_per_step=0.1, phi_tolerance=0.05)

    ok = controller.last_cycle_collision_refinement is None
    print(f"[controller] no refinement recorded before any solve: {'OK' if ok else 'FAIL'}")

    success = controller._solve_for_timing(x, x_dot)
    ok &= success
    diag = controller.last_cycle_collision_refinement
    ran = diag is not None
    print(f"[controller] _solve_for_timing succeeded: {'OK' if success else 'FAIL'}; "
          f"last_cycle_collision_refinement populated: {'OK' if ran else 'FAIL'}")
    ok &= ran
    if ran:
        improved = diag["final"]["inter_agent"] < diag["initial"]["inter_agent"]
        print(f"[controller] inter_agent cost: {diag['initial']['inter_agent']:.4f} -> "
              f"{diag['final']['inter_agent']:.4f} {'OK' if improved else 'FAIL'}")
        ok &= improved
    return ok


def check_no_retrace_across_cycles():
    """The actual point of CollisionRefiner: one instance's compiled scan
    is reused across cycles whose agents have DIFFERENT real knot counts
    (both padded to the same config.max_nodes), rather than retracing
    whenever a cycle's node count differs from the last -- verified by
    counting how many times `_run_scan`'s BODY actually executes (that
    only happens at jax.jit trace time, once per distinct compiled
    cache entry) across two `.refine()` calls built from deliberately
    different `LinearTracer` densities."""
    trace_count = [0]
    real_run_scan = ccj._run_scan

    def counting_run_scan(*args, **kwargs):
        trace_count[0] += 1
        return real_run_scan(*args, **kwargs)

    ccj._run_scan = counting_run_scan
    try:
        graph, specs, starts, goals = build_two_agent_graph()
        splines = [CubicConfigurationSpline([s]) for s in specs]
        config = CollisionRefinementConfig(
            fk_fns=[two_sphere_ish_fk, two_sphere_ish_fk],
            sphere_radii=[jnp.array([0.6]), jnp.array([0.6])],
            speed_fn=lambda q: 1.0,
            weights=CostWeights(inter_agent=8.0, environment=0.0, regularizer=0.0),
            n_steps=25, step_size=3e-4, t_samples=60, max_nodes=16,
        )
        refiner = ccj.CollisionRefiner(graph, config)

        x = np.concatenate([starts[0], starts[1]])
        x_dot = np.zeros(4)
        total_dim = graph.agent_col_offsets[-1]
        waypoints = np.zeros((4, total_dim))
        waypoints[0, 0:2] = np.array([2.0, 0.0])
        waypoints[1, 0:2] = goals[0]
        waypoints[2, 2:4] = np.array([2.0, 0.0])
        waypoints[3, 2:4] = goals[1]
        var_assignments = np.array([], dtype=np.int32)

        results = []
        for n_interior in (2, 6):  # different densities -> different real knot counts
            field = LinearTracer(n_interior=n_interior)
            timing_mpc = TracedTimingMPC(graph, splines, field, time_cost=1.0)
            ok_solve = timing_mpc.solve(x, x_dot, [0, 1, 2, 3], waypoints, var_assignments)
            assert ok_solve, "TracedTimingMPC.solve failed"
            timing_mpc.fill_cubic_splines(splines, x, x_dot)
            ns = [timing_mpc.get_agent_spline_length(i) for i in range(2)]
            diag = refiner.refine(timing_mpc, splines, x, x_dot)
            results.append((ns, diag))

        ns0, diag0 = results[0]
        ns1, diag1 = results[1]
        different_shapes = ns0 != ns1
        both_ran = diag0 is not None and diag1 is not None
        traced_once = trace_count[0] == 1
        print(f"[no-retrace] real knot counts differed across cycles ({ns0} vs {ns1}): "
              f"{'OK' if different_shapes else 'FAIL (test not exercising the thing it checks)'}")
        print(f"[no-retrace] both cycles refined successfully: {'OK' if both_ran else 'FAIL'}")
        print(f"[no-retrace] scan body traced exactly once across both cycles "
              f"(actual: {trace_count[0]}): {'OK' if traced_once else 'FAIL'}")
        return different_shapes and both_ran and traced_once
    finally:
        ccj._run_scan = real_run_scan


def check_active_mask_dtype_consistency():
    """The actual cause of intermittent per-cycle retracing seen running
    the full b1_multiroom experiment: a FABRICATED (not read off a real
    spline) `active_mask` used `np.int64` while pybind's real
    `CubicConfigurationSpline.block_active_mask()` always returns int32
    (`Eigen::MatrixXi`) -- invisible under jax's default (x64-disabled)
    mode, which silently downcasts both to int32 on `jnp.array(...)`, but
    a REAL per-agent pytree-leaf dtype mismatch once `jax_enable_x64` is
    True, which `eikonax` flips PROCESS-WIDE the moment b1_multiroom's
    default "sweep" edge-cost objective imports it -- an int64 numpy
    array then survives conversion as int64 while a real mask stays
    int32, forcing a retrace whenever an agent alternates between a real
    and a fabricated mask cycle to cycle. Deliberately enables x64 here
    to actually exercise the failure mode this guards against, and checks
    BOTH fabrication paths (the empty-raw-mask fallback and the
    parked/dummy-spline entry) against a real mask's own dtype."""
    import jax
    jax.config.update("jax_enable_x64", True)

    graph, specs, starts, goals = build_two_agent_graph()
    field = LinearTracer(n_interior=4)
    splines = [CubicConfigurationSpline([s]) for s in specs]
    timing_mpc = TracedTimingMPC(graph, splines, field, time_cost=1.0)

    x = np.concatenate([starts[0], starts[1]])
    x_dot = np.zeros(4)
    total_dim = graph.agent_col_offsets[-1]
    waypoints = np.zeros((4, total_dim))
    waypoints[0, 0:2] = np.array([2.0, 0.0])
    waypoints[1, 0:2] = goals[0]
    waypoints[2, 2:4] = np.array([2.0, 0.0])
    waypoints[3, 2:4] = goals[1]
    var_assignments = np.array([], dtype=np.int32)
    ok_solve = timing_mpc.solve(x, x_dot, [0, 1, 2, 3], waypoints, var_assignments)
    assert ok_solve, "TracedTimingMPC.solve failed"
    timing_mpc.fill_cubic_splines(splines, x, x_dot)

    real_data = ccj.extract_agent_refinement_data(timing_mpc, graph, splines[0], 0, x, x_dot, 16)
    real_dtype = np.asarray(real_data.active_mask).dtype

    # A brand-new, never-solved TracedTimingMPC: get_agent_spline_length
    # returns 0 (graph_timing_mpc.cpp) -- exactly the n<=1 parked/dummy
    # branch, exercised WITHOUT going through a real solve.
    fresh_timing_mpc = TracedTimingMPC(graph, [CubicConfigurationSpline([s]) for s in specs],
                                        field, time_cost=1.0)
    fresh_spline = CubicConfigurationSpline([specs[0]])
    parked_data = ccj.extract_agent_refinement_data(fresh_timing_mpc, graph, fresh_spline, 0,
                                                      x, x_dot, 16)
    parked_dtype = np.asarray(parked_data.active_mask).dtype

    ok = real_dtype == parked_dtype == np.dtype(np.int32)
    print(f"[dtype] real-mask dtype: {real_dtype}, parked/fabricated-mask dtype: {parked_dtype} "
          f"(expect both int32 under x64): {'OK' if ok else 'FAIL'}")
    return ok


def main():
    ok = True
    ok &= check_scan_mask_invariance()
    ok &= check_end_to_end()
    ok &= check_controller_wiring()
    ok &= check_no_retrace_across_cycles()
    ok &= check_active_mask_dtype_consistency()
    print("\nALL OK" if ok else "\nFAILURES ABOVE")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

"""Checks goc_mpc.repulsive_refinement and the C++ RepulsiveRefiner under it:

1. spacetime_repulsion's gradients (sphere centres and sample times, with
   obstacles) against central differences.
2. RepulsiveRefiner.energy_and_gradient (energy through the collision
   models and the Hermite pieces to the knots) against central differences.
3. GraphOfConstraintsMPC(collision_refinement=...): the refinement runs in
   _solve_for_timing, lowers the energy, and leaves every agent's current
   state and its graph waypoints where they were.
4. Agents already clear of each other: the refiner stops before its first
   step and leaves the splines alone.

Run: python examples/test_repulsive_refinement.py
"""

import numpy as np

from goc_mpc import GraphOfConstraints, GraphOfConstraintsMPC, ObstacleSet
from goc_mpc._ext.configuration_spline import CubicConfigurationSpline
from goc_mpc._ext.goc_mpc import RefinerKnots, RepulsiveRefiner, SpaceTimeKernel, spacetime_repulsion
from goc_mpc.repulsive_refinement import RepulsiveRefinementConfig, agent_knots
from goc_mpc.splines import Block
from goc_mpc.traced_timing_mpc import TracedTimingMPC


def check_energy_gradient():
    rng = np.random.default_rng(0)
    T = 40
    base = np.linspace(0.0, 4.0, T)
    times = [base, base * 1.03 + 0.02, base * 0.97]

    def agent(start, end, offsets, t):
        c = start + (end - start) * (t[:, None] / t[-1])
        return np.array([c[s] + o + 0.05 * rng.standard_normal(2) for s in range(T) for o in offsets])

    centers = [agent(np.array([-2.0, 0.0]), np.array([2.0, 0.1]), [np.zeros(2), np.array([0.2, 0.0])], times[0]),
               agent(np.array([2.0, 0.2]), np.array([-2.0, -0.1]), [np.zeros(2)], times[1]),
               agent(np.array([0.0, -2.0]), np.array([0.1, 2.0]), [np.zeros(2), np.array([0.0, 0.25])], times[2])]
    radii = [np.array([0.3, 0.25]), np.array([0.35]), np.array([0.3, 0.2])]
    obstacles = ObstacleSet()
    obstacles.add_box(np.array([0.0, 1.2]), np.array([1.0, 0.1]), 0.05)
    obstacles.add_sphere(np.array([-1.0, -0.6]), 0.3, 0.0)
    ok = True
    for alpha, beta in ((2.0, 4.5), (3.0, 6.0)):
        kernel = SpaceTimeKernel(alpha=alpha, beta=beta)
        w = 0.7

        def total(cs, ts):
            inter, obs, *_ = spacetime_repulsion(ts, cs, radii, obstacles, kernel, w, False)
            return inter + w * obs

        inter, obs, grads, tgrads, pairs = spacetime_repulsion(times, centers, radii, obstacles, kernel, w, True)
        h, worst = 1e-6, 0.0
        for a in range(3):
            for idx in rng.choice(centers[a].size, 40, replace=False):
                r, c = divmod(idx, 2)
                cp, cm = [x.copy() for x in centers], [x.copy() for x in centers]
                cp[a][r, c] += h
                cm[a][r, c] -= h
                fd = (total(cp, times) - total(cm, times)) / (2 * h)
                worst = max(worst, abs(fd - grads[a][r, c]) / max(1.0, abs(fd)))
            for s in rng.choice(T, 10, replace=False):
                tp, tm = [x.copy() for x in times], [x.copy() for x in times]
                tp[a][s] += h
                tm[a][s] -= h
                fd = (total(centers, tp) - total(centers, tm)) / (2 * h)
                worst = max(worst, abs(fd - tgrads[a][s]) / max(1.0, abs(fd)))
        good = pairs > 0 and inter > 0 and obs > 0 and worst < 1e-5
        print(f"[energy] alpha={alpha} beta={beta}: {pairs} pairs, worst relative gradient error {worst:.1e}: "
              f"{'OK' if good else 'FAIL'}")
        ok &= good
    return ok


def build_two_agent_graph():
    """Two Block.R(2) agents swapping ends of a line, each through a midpoint
    node of its own, so their paths meet head on."""
    spec = Block.R(2)
    graph = GraphOfConstraints([[spec], [spec]], [], state_lower_bound=-10.0, state_upper_bound=10.0,
                               robot_names=["a0", "a1"], object_names=[], workspace_dim=2)
    graph.structure.add_nodes(4)
    graph.structure.add_edge(0, 1, True)
    graph.structure.add_edge(2, 3, True)
    starts = {0: np.array([0.0, 0.0]), 1: np.array([4.0, 0.2])}
    mids = {0: np.array([2.0, 0.0]), 1: np.array([2.0, 1.5])}
    goals = {0: np.array([4.0, 0.0]), 1: np.array([0.0, 0.2])}
    q0, q1 = graph.agent_q(0), graph.agent_q(1)
    for k in range(2):
        graph.add_constraint(0, q0[k] == float(mids[0][k]))
        graph.add_constraint(1, q0[k] == float(goals[0][k]))
        graph.add_constraint(2, q1[k] == float(mids[1][k]))
        graph.add_constraint(3, q1[k] == float(goals[1][k]))
    return graph, [spec, spec], starts, mids, goals


class ZigzagTimingMPC(TracedTimingMPC):
    """TracedTimingMPC whose segments zigzag, so the timing solve keeps
    inserted knots between graph nodes for the refiner to move."""

    def __init__(self, graph, splines, n_interior=4, bow=0.3, **kwargs):
        super().__init__(graph, splines, [None] * graph.num_agents, **kwargs)
        self.n_interior, self.bow = n_interior, bow

    def plan_segments(self, segments):
        out = []
        for _, start, goal in segments:
            start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
            chord = goal - start
            length = np.linalg.norm(chord)
            if length < 1e-9:
                out.append(start[None, :])
                continue
            perp = np.array([-chord[1], chord[0]]) / length
            s = np.linspace(0.0, 1.0, self.n_interior + 2)
            signs = np.where(np.arange(len(s)) % 2 == 0, 1.0, -1.0)
            signs[[0, -1]] = 0.0
            out.append(start + s[:, None] * chord + (self.bow * signs)[:, None] * perp)
        return out


class StubWaypointMPC:
    def __init__(self, waypoints):
        self._waypoints = waypoints

    def view_waypoints(self):
        return self._waypoints

    def view_var_assignments(self):
        return np.array([], dtype=np.int32)

    def view_t_by_node(self):
        return np.array([])


def waypoints_for(graph, mids, goals):
    wp = np.zeros((4, graph.agent_col_offsets[-1]))
    wp[0, 0:2], wp[1, 0:2] = mids[0], goals[0]
    wp[2, 2:4], wp[3, 2:4] = mids[1], goals[1]
    return wp


def check_knot_gradient(label="knots", **config_kwargs):
    graph, specs, starts, mids, goals = build_two_agent_graph()
    splines = [CubicConfigurationSpline([s]) for s in specs]
    timing = ZigzagTimingMPC(graph, splines, time_cost=1.0)
    x, x_dot = np.concatenate([starts[0], starts[1]]), np.zeros(4)
    assert timing.solve(x, x_dot, [0, 1, 2, 3], waypoints_for(graph, mids, goals), np.array([], dtype=np.int32))
    timing.fill_cubic_splines(splines, x, x_dot)
    knots = [agent_knots(timing, graph, splines[a], a, x, x_dot) for a in range(2)]
    config = RepulsiveRefinementConfig(agent_radii=[0.5, 0.5], **config_kwargs)
    refiner = RepulsiveRefiner(graph, None, config.options())
    E, g = refiner.energy_and_gradient(knots)
    entries = []
    for a, k in enumerate(knots):
        n = k.times.shape[0]
        wf, vf, tf = (np.asarray(m) for m in (k.wps_free, k.vs_free, k.times_free))
        entries += [(a, 0, i, j) for i in range(n) if wf[i] for j in range(2)]
        entries += [(a, 1, i, j) for i in range(n) if vf[i] for j in range(2)]
        entries += [(a, 2, i, 0) for i in range(n) if tf[i]]

    def shifted(entry, delta):
        a, kind, i, j = entry
        ks = list(knots)
        k = ks[a]
        t, w, v = np.array(k.times), np.array(k.wps), np.array(k.vs)
        if kind == 0:
            w[i, j] += delta
        elif kind == 1:
            v[i, j] += delta
        else:
            t[i] += delta
        ks[a] = RefinerKnots(t, w, v, np.asarray(k.active), np.asarray(k.wps_free), np.asarray(k.vs_free),
                             np.asarray(k.times_free))
        return refiner.energy_and_gradient(ks)[0]

    h, worst = 1e-6, 0.0
    for idx, entry in enumerate(entries):
        fd = (shifted(entry, h) - shifted(entry, -h)) / (2 * h)
        worst = max(worst, abs(fd - g[idx]) / max(1.0, abs(fd), abs(g[idx])))
    ok = E > 0 and len(entries) == g.size and worst < 1e-5
    print(f"[{label}] energy {E:.3f}, {g.size} free entries, worst relative gradient error {worst:.1e}: "
          f"{'OK' if ok else 'FAIL'}")
    return ok


def check_controller_wiring():
    graph, specs, starts, mids, goals = build_two_agent_graph()
    timing = ZigzagTimingMPC(graph, [CubicConfigurationSpline([s]) for s in specs], time_cost=1.0)
    controller = GraphOfConstraintsMPC(
        graph, waypoint_mpc=StubWaypointMPC(waypoints_for(graph, mids, goals)), timing_mpc=timing,
        collision_refinement=RepulsiveRefinementConfig(agent_radii=[0.5, 0.5]),
        short_path_time_per_step=0.1, phi_tolerance=0.05)
    x, x_dot = np.concatenate([starts[0], starts[1]]), np.zeros(4)
    ok = controller.last_cycle_collision_refinement is None
    ok &= controller._solve_for_timing(x, x_dot)
    report = controller.last_cycle_collision_refinement
    if report is None:
        print("[controller] refinement didn't run: FAIL")
        return False
    lowered = report["energy_final"] < report["energy_initial"]
    print(f"[controller] {report['iterations']} iterations, stop '{report['stop_reason']}', energy "
          f"{report['energy_initial']:.3f} -> {report['energy_final']:.3f}, clearance "
          f"{report['clearance_initial'][0]:+.3f} -> {report['clearance_final'][0]:+.3f}: "
          f"{'OK' if lowered and report['moved'] else 'FAIL'}")
    ok &= lowered and report["moved"]
    nodes = timing.view_agent_nodes_list()
    for a in range(2):
        sp = controller.last_cycle_splines[a]
        start_ok = np.allclose(sp.eval(0.0)[0], x[2 * a:2 * a + 2], atol=1e-9)
        wps = np.asarray(timing.view_wps_list()[a])
        # The refiner moves knot times but never the positions of graph
        # waypoints, so the refined spline still passes through each.
        real = [i for i, node in enumerate(nodes[a]) if node != -1]
        hits = [min(np.linalg.norm(sp.eval(t)[0] - wps[i]) for t in np.linspace(0.0, sp.end(), 4001)) for i in real]
        wp_ok = max(hits) < 2e-3
        print(f"[controller] agent {a}: current state kept: {'OK' if start_ok else 'FAIL'}; passes its "
              f"{len(real)} graph waypoints (worst {max(hits):.1e} m): {'OK' if wp_ok else 'FAIL'}")
        ok &= start_ok and wp_ok
    return ok


def check_early_exit():
    graph, specs, starts, mids, goals = build_two_agent_graph()
    far = {1: np.array([0.0, 6.0])}
    mids, goals = dict(mids), dict(goals)
    mids[1], goals[1] = np.array([2.0, 6.0]), np.array([4.0, 6.0])
    splines = [CubicConfigurationSpline([s]) for s in specs]
    timing = ZigzagTimingMPC(graph, splines, time_cost=1.0)
    x, x_dot = np.concatenate([starts[0], far[1]]), np.zeros(4)
    wp = waypoints_for(graph, mids, goals)
    assert timing.solve(x, x_dot, [0, 1, 2, 3], wp, np.array([], dtype=np.int32))
    timing.fill_cubic_splines(splines, x, x_dot)
    before = [np.array([sp.eval(t)[0] for t in np.linspace(0, sp.end(), 50)]) for sp in splines]
    # The graph pins a1's waypoints elsewhere; only the splines matter here.
    from goc_mpc.repulsive_refinement import SplineRefiner
    report = SplineRefiner(graph, RepulsiveRefinementConfig(agent_radii=[0.5, 0.5])).refine(timing, splines, x, x_dot)
    after = [np.array([sp.eval(t)[0] for t in np.linspace(0, sp.end(), 50)]) for sp in splines]
    same = all(np.array_equal(b, a) for b, a in zip(before, after))
    ok = report["iterations"] == 0 and report["stop_reason"] == "clear" and not report["moved"] and same
    print(f"[early exit] stop '{report['stop_reason']}' after {report['iterations']} iterations in "
          f"{report['time'] * 1e3:.2f} ms, splines unchanged: {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    results = [check_energy_gradient(), check_knot_gradient(),
               check_knot_gradient("knots+speed", speed_weight=10.0, max_speed=0.2), check_controller_wiring(), check_early_exit()]
    print("ALL OK" if all(results) else "SOME CHECKS FAILED")
    raise SystemExit(0 if all(results) else 1)

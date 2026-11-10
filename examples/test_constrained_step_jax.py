"""Checks for `goc_mpc.constrained_step_jax`, walls as hard constraints in the P2 refiner.

Covers the projected direction (Eq 22) and its sign check, back-projection (Eq 23) from a start inside a
wall, a repulsion that would push an agent into a wall, pins, and the refiner hook.

Run: python examples/test_constrained_step_jax.py
"""

import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from goc_mpc import collision_cost_jax as ccj
from goc_mpc import constrained_step_jax as cs
from goc_mpc.collision_cost_jax import AgentGeometry, CostWeights, SplineParams, build_cost_fn
from goc_mpc.constrained_step_jax import WallConstraint
from goc_mpc.spacetime_repulsion_jax import SpaceTimeKernel
from goc_mpc.splines import Block
from test_collision_refinement_loop import refinement_config, solved_two_agent_problem

KERNEL = SpaceTimeKernel(time_scale=1.0, cutoff=0.3, relax=0.05)
WALL_Y, RADIUS = 0.75, 0.3


def report(name, ok):
    print(f"[constrained] {name}: {'OK' if ok else 'FAIL'}")
    return bool(ok)


def check_direction():
    rng = np.random.default_rng(0)
    n = 8
    g = jnp.asarray(rng.normal(size=n))
    D = jnp.asarray(rng.uniform(0.5, 2.0, n)).at[3].set(0.0)       # entry 3 pinned
    con = WallConstraint()
    none = cs.projected_direction(g, jnp.zeros((2, n)), D, jnp.array([False, False]), con)
    ok = report("no held rows gives -D g", np.allclose(none, -D * g, atol=1e-14))
    c = jnp.asarray(rng.normal(size=n))
    if float(c @ (-D * g)) > 0.0:
        c = -c                                                       # make the row block the free step
    d = cs.projected_direction(g, c[None, :], D, jnp.array([True]), con)
    ok &= report("a blocking row is held at C d = 0", abs(float(c @ d)) < 1e-7 * float(jnp.linalg.norm(c) * jnp.linalg.norm(d)))
    ok &= report("the held step still descends", float(g @ d) < 0.0)
    ok &= report("pinned entries do not move", float(d[3]) == 0.0)
    away = cs.projected_direction(g, -c[None, :], D, jnp.array([True]), con)
    ok &= report("a row the step leaves anyway is released", np.allclose(away, -D * g, atol=1e-12))
    return ok


def _scene():
    """One agent passing a parked one at the origin, a wall at y = WALL_Y above the pass."""
    spec = [Block.R(2)]
    fk = lambda q: jnp.concatenate([q, jnp.zeros(1)])[None, :]
    geometry = [AgentGeometry(spec=spec, fk_fn=fk, sphere_radii=jnp.array([RADIUS])) for _ in range(2)]
    n = 8
    x = jnp.linspace(-3.0, 3.0, n)
    moving = SplineParams(times=jnp.linspace(0.0, 6.0, n), wps=jnp.stack([x, jnp.full(n, 0.15)], 1),
                          vs=jnp.zeros((n, 2)).at[1:-1, 0].set(1.0))
    parked = SplineParams(times=jnp.linspace(0.0, 6.0, n), wps=jnp.zeros((n, 2)), vs=jnp.zeros((n, 2)))
    masks = [jnp.ones((n, 1), dtype=jnp.int32) for _ in range(2)]
    interior = jnp.ones(n, dtype=bool).at[0].set(False).at[-1].set(False)
    zero = jnp.zeros(n, dtype=bool)
    clearance = lambda q: WALL_Y - q[1] - RADIUS
    return geometry, [moving, parked], masks, [(interior, interior, interior), (zero, zero, zero)], clearance


def _run(params, constraint, steps=60):
    geometry, _, masks, free, clearance = _scene()
    t_grid = jnp.linspace(0.0, 6.0, 200)
    weights = CostWeights(inter_agent=0.0, environment=0.0, regularizer=0.0, repulsive=8.0)
    cost = build_cost_fn(geometry, t_grid, lambda q: 1.0, weights, repulsive_kernel=KERNEL)
    q0 = [ccj.sample_configs(g, p, t_grid, m) for g, p, m in zip(geometry, params, masks)]
    if constraint is None:
        final, hist, _ = ccj._run_line_search(cost, params, q0, masks, free, steps, 1e-3, 8, (1.0, 1.0, 0.25),
                                              max_step=0.1)
    else:
        final, hist, _ = cs._run_constrained(cost, geometry, clearance, t_grid, params, q0, masks, free, steps,
                                             1e-3, 8, (1.0, 1.0, 0.25), constraint, max_step=0.1,
                                             project_times=ccj._project_knot_times)
    q = ccj.sample_configs(geometry[0], final[0], jnp.linspace(0.0, 6.0, 2000), masks[0])
    wall = float(jnp.min(jax.vmap(clearance)(q)))
    gap = float(jnp.min(jnp.linalg.norm(q, axis=1))) - 2 * RADIUS
    _, bd = cost(final, q0, masks)
    _, bd0 = cost(params, q0, masks)
    return final, wall, gap, float(bd0["repulsive"]), float(bd["repulsive"])


def check_wall_conflict():
    _, params, *_ = _scene()
    _, wall_free, gap_free, e0, e_free = _run(params, None)
    final, wall, gap, _, e = _run(params, WallConstraint(margin=0.0, band=0.05))
    ok = report(f"unconstrained repulsion enters the wall ({wall_free:+.3f} m, gap {gap_free:+.3f})", wall_free < -0.02)
    ok &= report(f"constrained stays out of the wall ({wall:+.4f} m, dense samples)", wall > -5e-3)
    ok &= report(f"constrained still lowers the energy ({e0:.3f} -> {e:.3f})", e < e0)
    ok &= report("parked agent does not move", all(np.array_equal(x, y) for x, y in zip(final[1], params[1])))
    ends = all(np.array_equal(np.asarray(a)[[0, -1]], np.asarray(b)[[0, -1]]) for a, b in zip(final[0], params[0]))
    return ok & report("pinned end knots do not move", ends)


def check_restore():
    _, params, *_ = _scene()
    inside = params[0]._replace(wps=params[0].wps.at[2:6, 1].set(0.65))        # 0.2 m into the wall
    start = [inside, params[1]]
    geometry, _, masks, _, clearance = _scene()
    q = ccj.sample_configs(geometry[0], inside, jnp.linspace(0.0, 6.0, 2000), masks[0])
    before = float(jnp.min(jax.vmap(clearance)(q)))
    _, wall, *_ = _run(start, WallConstraint(margin=0.0), steps=10)
    return report(f"a start inside the wall is pulled out ({before:+.3f} -> {wall:+.4f} m)",
                  before < -0.15 and wall > -5e-3)


def check_refiner():
    graph, splines, timing_mpc, x, x_dot = solved_two_agent_problem()
    weights = CostWeights(inter_agent=0.0, environment=0.0, regularizer=0.0, repulsive=2.0)
    clearance = lambda q: 10.0 - jnp.abs(q[1])                                  # far walls
    config = refinement_config(repulsive_kernel=KERNEL, weights=weights, line_search=True, max_step=0.1,
                               clearance_fn=clearance, wall_constraint=WallConstraint())
    diag = ccj.refine_splines(timing_mpc, graph, splines, x, x_dot, config)
    ok = report(f"refiner runs with walls as constraints ({diag['initial']['repulsive']:.2f} -> "
                f"{diag['final']['repulsive']:.2f})",
                diag["accepted"] and diag["final"]["repulsive"] < diag["initial"]["repulsive"])
    refused = 0
    for bad in (dict(clearance_fn=None), dict(line_search=False), dict(max_step=float("inf")),
                dict(wall_constraint=WallConstraint(rows_per_agent=0)), dict(wall_constraint=WallConstraint(dual_iters=0))):
        try:
            ccj.CollisionRefiner(graph, config._replace(**bad))
        except ValueError:
            refused += 1
    return ok & report("unusable constraint configs are refused at build", refused == 5)


def check_direction_inequalities():
    """Several held rows: C d >= 0 on every one, and mu = 0 where the free step already leaves."""
    rng = np.random.default_rng(1)
    con = WallConstraint()
    worst = 0.0
    for _ in range(50):
        n, K = 10, 4
        g = jnp.asarray(rng.normal(size=n))
        D = jnp.asarray(rng.uniform(0.5, 2.0, n))
        C = jnp.asarray(rng.normal(size=(K, n)))
        d = cs.projected_direction(g, C, D, jnp.ones(K, bool), con)
        worst = min(worst, float(jnp.min(C @ d)) / float(jnp.linalg.norm(d) + 1e-300))
    return report(f"held rows are never pushed into (worst C d / |d| {worst:+.1e})", worst > -1e-6)


def check_parallel_rows_restore():
    """Two nearly parallel violated rows with different targets: one small lift, not metres."""
    n = 6
    c = jnp.arange(1.0, n + 1.0)
    C = jnp.stack([c, c + 1e-5 * jnp.ones(n)])
    target = jnp.array([0.01, 0.002])
    D = jnp.ones(n)
    nu = cs.solve_dual(C, D, target, jnp.array([True, True]), WallConstraint())
    delta = D * (C.T @ nu)
    ok = report("nearly parallel rows lift by the binding one", bool(jnp.all(C @ delta >= target - 1e-9)))
    return ok & report(f"and the lift stays small ({float(jnp.linalg.norm(delta)):.4f})",
                       float(jnp.linalg.norm(delta)) < 1.01 * 0.01 / float(jnp.linalg.norm(c)))


def check_movable_samples():
    n = 6
    p = SplineParams(times=jnp.arange(n, dtype=float), wps=jnp.array([[0., 0], [1, 0], [2, 0], [3, 0], [3, 0], [3, 0]]),
                     vs=jnp.zeros((n, 2)).at[1:3, 0].set(1.0))
    free = jnp.array([False, True, True, False, False, False])
    t = jnp.array([0.0, 0.5, 2.5, 3.5, 4.5])
    m = np.asarray(cs.movable_samples(p, (free, free, free), t))
    return report(f"pinned start and the goal hold are not candidates ({m.astype(int).tolist()})",
                  m.tolist() == [False, True, True, False, False])


def _b1_like(y0, vy0):
    """The wall scene with B1's masks: row 0 pinned, the last row's position and velocity pinned."""
    geometry, params, masks, _, clearance = _scene()
    n = params[0].times.shape[0]
    moving = params[0]._replace(wps=params[0].wps.at[0, 1].set(y0), vs=params[0].vs.at[0].set(jnp.array([1.0, vy0])))
    wf = jnp.ones(n, bool).at[0].set(False).at[-1].set(False)
    tf = jnp.ones(n, bool).at[0].set(False)
    zero = jnp.zeros(n, bool)
    return geometry, [moving, params[1]], masks, [(wf, wf, tf), (zero, zero, zero)], clearance


def check_start_near_margin():
    """Pinned start velocity into the wall from just inside the margin: the step must still make progress."""
    ok = True
    for y0, vy0 in ((0.44, 0.3), (0.47, 0.3)):
        geometry, params, masks, free, clearance = _b1_like(y0, vy0)
        t_grid = jnp.linspace(0.0, 6.0, 241)
        weights = CostWeights(inter_agent=0.0, environment=0.0, regularizer=0.0, repulsive=8.0)
        cost = build_cost_fn(geometry, t_grid, lambda q: 1.0, weights, repulsive_kernel=KERNEL)
        q0 = [ccj.sample_configs(g, p, t_grid, m) for g, p, m in zip(geometry, params, masks)]
        final, hist, _ = cs._run_constrained(cost, geometry, clearance, t_grid, params, q0, masks, free, 25, 1e-3, 8,
                                             (1.0, 1.0, 0.25), WallConstraint(), max_step=0.1,
                                             project_times=ccj._project_knot_times)
        v = lambda p: jax.nn.relu(-jax.vmap(clearance)(ccj.sample_configs(geometry[0], p, t_grid, masks[0])))
        e0, e1 = float(cost(params, q0, masks)[0]), float(cost(final, q0, masks)[0])
        ok &= report(f"start {y0} m, vy0 {vy0}: energy falls ({e0:.1f} -> {e1:.1f})", e1 < 0.5 * e0)
        ok &= report(f"start {y0} m, vy0 {vy0}: violation does not grow "
                     f"({float(jnp.max(v(params[0]))):.4f} -> {float(jnp.max(v(final[0]))):.4f})",
                     float(jnp.max(v(final[0]))) <= float(jnp.max(v(params[0]))) + 1e-9
                     and float(jnp.sum(v(final[0]))) <= float(jnp.sum(v(params[0]))) + 1e-9)
    return ok


def check_stuck_agent_does_not_freeze_others():
    """Agent A starts inside the margin with no way out at t0; agent B, far away, must still move."""
    spec = [Block.R(2)]
    geometry = [AgentGeometry(spec=spec, fk_fn=lambda q: jnp.concatenate([q, jnp.zeros(1)])[None, :],
                              sphere_radii=jnp.array([0.1]))] * 2
    n = 6
    x = jnp.linspace(0.0, 5.0, n)
    a = SplineParams(times=jnp.linspace(0.0, 5.0, n), wps=jnp.stack([x, jnp.full(n, 0.85).at[0].set(0.885)], 1),
                     vs=jnp.zeros((n, 2)).at[1:-1, 0].set(1.0))
    b = a._replace(wps=jnp.stack([x, jnp.full(n, -3.0)], 1))
    masks = [jnp.ones((n, 1), jnp.int32)] * 2
    free = (jnp.ones(n, bool).at[0].set(False).at[-1].set(False), jnp.ones(n, bool).at[0].set(False),
            jnp.ones(n, bool).at[0].set(False))

    def cost(params, q0, am):
        e = jnp.sum((params[0].wps[1:-1, 1] - 0.5) ** 2) + jnp.sum((params[1].wps[1:-1, 1] + 2.5) ** 2)
        return e, {"e": e}

    out, _, _ = cs._run_constrained(cost, geometry, lambda q: 1.0 - q[1], jnp.linspace(0.0, 5.0, 150), [a, b], None,
                                    masks, [free, free], 20, 1e-3, 8, (1.0, 1.0, 1.0), WallConstraint(margin=0.12),
                                    max_step=0.1, project_times=ccj._project_knot_times)
    by = float(jnp.max(jnp.abs(out[1].wps[1:-1, 1] + 2.5)))
    return report(f"an agent stuck in the margin does not hold back another (B off target {by:.3f} m)", by < 0.05)


def check_refiner_active_wall():
    """Through the real refiner, with a wall the unconstrained repulsion runs into."""
    graph, splines, timing_mpc, x, x_dot = solved_two_agent_problem()
    weights = CostWeights(inter_agent=0.0, environment=0.0, regularizer=0.0, repulsive=2.0)
    base = refinement_config(repulsive_kernel=KERNEL, weights=weights, line_search=True, max_step=0.1, n_steps=40)
    e = [ccj.extract_agent_refinement_data(timing_mpc, graph, splines[i], i, x, x_dot, base.max_nodes)
         for i in range(2)]
    t = jnp.linspace(0.0, max(ev.real_end_time for ev in e), 400)
    ys = [ccj.sample_configs(AgentGeometry(graph._robot_specs[i], None, None), ev.params, t, ev.active_mask)[:, 1]
          for i, ev in enumerate(e)]
    wall = float(max(jnp.max(jnp.abs(y)) for y in ys)) + 0.02
    clearance = lambda q: wall - jnp.abs(q[1])

    def worst(cfg):
        g2, sp, tm, x2, xd2 = solved_two_agent_problem()
        diag = ccj.refine_splines(tm, g2, sp, x2, xd2, cfg)
        lows = []
        for i in range(2):
            w = sp[i].written
            p = SplineParams(*(jnp.asarray(a) for a in w)) if w is not None else e[i].params
            mask = jnp.asarray(sp[i].block_active_mask()) if w is not None else e[i].active_mask
            q = ccj.sample_configs(AgentGeometry(g2._robot_specs[i], None, None), p, t, mask)
            lows.append(float(jnp.min(jax.vmap(clearance)(q))))
        return min(lows), diag

    free_low, _ = worst(base._replace(clearance_fn=clearance))
    low, diag = worst(base._replace(clearance_fn=clearance, wall_constraint=WallConstraint()))
    ok = report(f"unconstrained refiner crosses the wall ({free_low:+.3f} m)", free_low < -0.01)
    return ok & report(f"constrained refiner keeps it ({low:+.4f} m, energy {diag['initial']['repulsive']:.2f} -> "
                       f"{diag['final']['repulsive']:.2f})",
                       low > -5e-3 and diag["accepted"] and diag["final"]["repulsive"] < diag["initial"]["repulsive"])


def main():
    ok = True
    for check in (check_direction, check_direction_inequalities, check_parallel_rows_restore, check_movable_samples,
                  check_wall_conflict, check_restore, check_start_near_margin, check_stuck_agent_does_not_freeze_others,
                  check_refiner, check_refiner_active_wall):
        ok &= check()
    print("\nALL OK" if ok else "\nFAILURES ABOVE")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

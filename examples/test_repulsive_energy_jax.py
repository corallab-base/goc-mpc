"""Checks for `goc_mpc.repulsive_energy_jax`, the same-time repulsive energy.

Covers the barrier's joins and normalization, gradients at and through
contact, the sample weights, a finite-difference check of the energy, and
the term running inside `CollisionRefiner`, including a moving agent being
pushed off a parked one.

Run: python examples/test_repulsive_energy_jax.py
"""

import os
import sys

import jax

jax.config.update("jax_enable_x64", True)  # the barrier spans too many decades for float32

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from goc_mpc import collision_cost_jax as ccj
from goc_mpc._ext.configuration_spline import CubicConfigurationSpline
from goc_mpc.collision_cost_jax import AgentGeometry, CostWeights, SplineParams, build_cost_fn
from goc_mpc.repulsive_energy_jax import RepulsiveKernel, barrier, repulsive_energy, sample_weights
from goc_mpc.splines import Block
from goc_mpc.traced_timing_mpc import TracedTimingMPC
from test_collision_refinement_loop import (
    LinearTracer, _line_search_problem, build_two_agent_graph, refinement_config, solved_two_agent_problem,
    two_sphere_ish_fk)

KERNEL = RepulsiveKernel(power=2.5, cutoff=0.25, relax=0.02, reference_gap=0.1)


def report(name, ok):
    print(f"[repulsive] {name}: {'OK' if ok else 'FAIL'}")
    return bool(ok)


def check_barrier_joins():
    value = lambda g: barrier(jnp.asarray(g), KERNEL)
    slope = jax.grad(value)
    curve = jax.grad(slope)
    h = 1e-7
    ok = report("zero and flat at and past the cutoff",
                float(value(0.25)) == 0.0 and float(value(1.0)) == 0.0 and float(slope(0.3)) == 0.0)
    below, above = 0.25 - h, 0.25 + h
    ok &= report("value and slope reach zero at the cutoff",
                 abs(float(value(below))) < 1e-9 and abs(float(slope(below))) < 1e-5 and float(slope(above)) == 0.0)
    a, b = 0.02 - 1e-9, 0.02 + 1e-9
    close = lambda x, y: abs(float(x) - float(y)) <= 1e-5 * max(abs(float(x)), 1.0)
    ok &= report("value, slope and curvature match at relax",
                 close(value(a), value(b)) and close(slope(a), slope(b)) and close(curve(a), curve(b)))
    gaps = jnp.linspace(-1.0, 0.25, 500)
    values = np.asarray(barrier(gaps, KERNEL))
    ok &= report("strictly decreasing up to the cutoff", np.all(np.diff(values) < 0.0))
    ok &= report("force is 1 at the reference gap", abs(float(slope(0.1)) + 1.0) < 1e-10)
    for relax in (0.1, 0.15):
        wide = RepulsiveKernel(relax=relax)
        force = float(jax.grad(lambda g: barrier(g, wide))(jnp.asarray(0.1)))
        ok &= report(f"force is 1 at the reference gap with relax {relax}", abs(force + 1.0) < 1e-10)
    tie = float(slope(0.02))
    return ok & report("slope is continuous exactly at relax", abs(tie - float(slope(0.02 + 1e-10))) < 1e-3 * abs(tie))


def check_barrier_closed_form():
    """Every field differs from the default, and the formula is written out
    here rather than taken from the module."""
    p, cut, relax, ref = 1.0, 0.3, 0.05, 0.15
    kernel = RepulsiveKernel(power=p, cutoff=cut, relax=relax, reference_gap=ref)
    shifted = lambda g: g ** -p - cut ** -p + p * cut ** -(p + 1.0) * (g - cut)
    force = lambda g: p * g ** -(p + 1.0) - p * cut ** -(p + 1.0)
    at_power = shifted(0.1) / force(ref)
    at_quadratic = (shifted(relax) + force(relax) * relax + 0.5 * p * (p + 1.0) * relax ** -(p + 2.0) * relax ** 2) / force(ref)
    ok = report("power branch by formula", abs(float(barrier(jnp.asarray(0.1), kernel)) - at_power) < 1e-10)
    return ok & report("quadratic branch by formula", abs(float(barrier(jnp.asarray(0.0), kernel)) - at_quadratic) < 1e-10)


def check_bad_kernels_raise():
    ok = True
    for bad in (RepulsiveKernel(relax=0.3), RepulsiveKernel(reference_gap=0.3), RepulsiveKernel(power=0.0)):
        try:
            barrier(jnp.asarray(0.1), bad)
            ok = False
        except ValueError:
            pass
    return report("invalid kernels raise", ok)


def check_gradient_through_contact():
    grad = jax.grad(lambda g: jnp.sum(barrier(g, KERNEL)))
    gaps = jnp.array([-2.0, -0.5, 0.0, 1e-12, 0.02, 0.1, 0.25, 3.0])
    values = grad(gaps)
    ok = report("gradient finite from deep overlap to past the cutoff", np.all(np.isfinite(values)))
    ok &= report("deeper overlap pushes harder", abs(float(values[0])) > abs(float(values[1])) > abs(float(values[2])))

    c = jnp.zeros((3, 1, 3))
    r = jnp.array([0.3])
    t = jnp.linspace(0.0, 2.0, 3)
    coincident = jax.grad(lambda x: repulsive_energy([x, c], [r, r], t, KERNEL))(c)
    return ok & report("coincident centers give a finite gradient", np.all(np.isfinite(coincident)))


def check_sample_weights():
    uniform = np.asarray(sample_weights(jnp.linspace(0.0, 4.0, 5)))
    ok = report("uniform grid", np.allclose(uniform, [0.625, 1.25, 1.25, 1.25, 0.625]))
    t = jnp.array([0.0, 1.0, 3.0])
    nonuniform = np.asarray(sample_weights(t))
    return ok & report("nonuniform grid follows the spacing, mean 1",
                       np.allclose(nonuniform, np.array([0.5, 1.5, 1.0]) * 3.0 / 3.0) and abs(nonuniform.mean() - 1.0) < 1e-12)


def check_energy_by_hand():
    a = jnp.array([[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]])
    b = jnp.array([[[0.7, 0.0, 0.0]], [[5.0, 0.0, 0.0]]])
    r = jnp.array([0.3])
    t = jnp.array([0.0, 1.0])
    energy = float(repulsive_energy([a, b], [r, r], t, KERNEL))
    expected = 0.031141219423370652  # second sample is past the cutoff
    return report("two spheres at a known gap", abs(energy - expected) < 1e-6 * expected)


def check_energy_sums_everything():
    """Three agents with different sphere counts and radii on a nonuniform
    grid, against a plain loop over agent pairs, samples and sphere pairs."""
    rng = np.random.default_rng(9)
    t = jnp.array([0.0, 0.2, 0.7, 0.9, 2.0, 2.5])
    centers = [jnp.asarray(rng.normal(size=(6, n, 3)) * 0.35) for n in (2, 3, 1)]
    radii = [jnp.array([0.2, 0.3]), jnp.array([0.1, 0.25, 0.15]), jnp.array([0.28])]
    weights = np.asarray(sample_weights(t))
    expected = 0.0
    for a, b in ((0, 1), (0, 2), (1, 2)):
        for k in range(6):
            for i in range(centers[a].shape[1]):
                for j in range(centers[b].shape[1]):
                    distance = np.sqrt(np.sum(np.asarray(centers[a][k, i] - centers[b][k, j]) ** 2) + 1e-9)
                    gap = distance - float(radii[a][i] + radii[b][j])
                    expected += weights[k] * float(barrier(jnp.asarray(gap), KERNEL))
    energy = float(repulsive_energy(centers, radii, t, KERNEL))
    return report("agents, sphere pairs and sample weights all counted", abs(energy - expected) < 1e-9 * max(abs(expected), 1.0))


def check_finite_differences():
    rng = np.random.default_rng(4)
    a = jnp.asarray(rng.normal(size=(12, 2, 3)) * 0.4)
    b = jnp.asarray(rng.normal(size=(12, 3, 3)) * 0.4)
    ra, rb = jnp.array([0.2, 0.25]), jnp.array([0.2, 0.1, 0.25])
    t = jnp.asarray(np.cumsum(rng.uniform(0.1, 0.5, 12)))
    energy = lambda x: repulsive_energy([x, b], [ra, rb], t, KERNEL)
    grad = np.asarray(jax.grad(energy)(a))
    worst = 0.0
    h = 1e-6  # central differences bottom out here: smaller is roundoff, larger is truncation
    for index in np.ndindex(a.shape):
        fd = (float(energy(a.at[index].add(h))) - float(energy(a.at[index].add(-h)))) / (2.0 * h)
        worst = max(worst, abs(fd - grad[index]) / max(1.0, abs(fd)))
    return report(f"gradient against finite differences (worst relative {worst:.1e})", worst < 1e-6)


def _straight_pass():
    """A moving agent on a line through a parked one, in the plane."""
    spec = [Block.R(2)]
    fk = lambda q: jnp.concatenate([q, jnp.zeros(1)])[None, :]
    radii = jnp.array([0.3])
    geometry = [AgentGeometry(spec=spec, fk_fn=fk, sphere_radii=radii) for _ in range(2)]
    n = 6
    x = jnp.linspace(-3.0, 3.0, n)
    wps = jnp.stack([x, jnp.full(n, 0.02)], axis=1)
    vs = jnp.zeros((n, 2)).at[1:-1, 0].set(1.0)
    times = jnp.linspace(0.0, 6.0, n)
    moving = SplineParams(times=times, wps=wps, vs=vs)
    parked = SplineParams(times=times, wps=jnp.zeros((n, 2)), vs=jnp.zeros((n, 2)))
    masks = [jnp.ones((n, 1), dtype=jnp.int32) for _ in range(2)]
    one = jnp.ones(n)
    interior = jnp.ones(n).at[0].set(0.0).at[-1].set(0.0)
    free = [(interior, interior, interior), (0.0 * one, 0.0 * one, 0.0 * one)]
    return geometry, [moving, parked], masks, free


def check_parked_agent_repels():
    geometry, params, masks, free = _straight_pass()
    t_grid = jnp.linspace(0.0, 6.0, 300)
    weights = CostWeights(inter_agent=0.0, environment=0.0, regularizer=0.0, repulsive=4.0)
    cost = build_cost_fn(geometry, t_grid, lambda q: 1.0, weights, repulsive_kernel=KERNEL)
    q0 = [ccj.sample_configs(g, p, t_grid, m) for g, p, m in zip(geometry, params, masks)]
    final, history, _ = ccj._run_line_search(cost, params, q0, masks, free, 60, 1e-3, 8, (1.0, 1.0, 0.25),
                                             max_step=0.1)

    total = lambda w: cost_with(w)(params, q0, masks)

    def cost_with(weight):
        return build_cost_fn(geometry, t_grid, lambda q: 1.0, weights._replace(repulsive=weight), repulsive_kernel=KERNEL)

    with_term, breakdown = total(4.0)
    without_term, _ = total(0.0)
    weighted = abs(float(with_term - without_term) - 4.0 * float(breakdown["repulsive"])) < 1e-6 * float(with_term)

    def gap(p):
        q = ccj.sample_configs(geometry[0], p, t_grid, masks[0])
        return float(jnp.min(jnp.linalg.norm(q, axis=1))) - 0.6

    before, after = gap(params[0]), gap(final[0])
    ok = report(f"moving agent clears a parked one, without overshooting ({before:+.3f} -> {after:+.3f} m)",
                before < 0.0 and 0.05 < after < 0.3)
    ok &= report("the term weight scales the cost", weighted)
    ok &= report("parked agent does not move", all(np.array_equal(a, b) for a, b in zip(final[1], params[1])))
    return ok & report("cost never rises", np.all(np.diff(np.asarray(history)) <= 1e-9))


class _ParkedStub:
    """A timing solve in which agent 1 has finished, as the refiner sees it."""

    def __init__(self, timing_mpc):
        self._timing_mpc = timing_mpc

    def get_agent_spline_length(self, agent):
        return 1 if agent == 1 else self._timing_mpc.get_agent_spline_length(agent)

    def __getattr__(self, name):
        return getattr(self._timing_mpc, name)


def check_parked_agent_in_refiner():
    graph, splines, timing_mpc, x, x_dot = solved_two_agent_problem()
    x, x_dot = x.copy(), x_dot.copy()
    x[2:4], x_dot[2:4] = (1.0, 0.9), (0.5, 0.0)
    weights = CostWeights(inter_agent=0.0, environment=0.0, regularizer=0.0, repulsive=4.0)
    config = refinement_config(repulsive_kernel=KERNEL, weights=weights, line_search=True, max_step=0.1, n_steps=30)
    diag = ccj.refine_splines(_ParkedStub(timing_mpc), graph, splines, x, x_dot, config)
    ok = report("parked agent is not written back", splines[1].written is None and diag["accepted"])
    ok &= report("moving agent is written back", splines[0].written is not None and diag["moved"])
    return ok & report(f"refiner sees the parked agent ({diag['initial']['repulsive']:.1f} -> "
                       f"{diag['final']['repulsive']:.1f})",
                       0.0 < diag["final"]["repulsive"] < diag["initial"]["repulsive"])


def check_line_search_cap():
    cost, params, free, _target = _line_search_problem()
    run = lambda **kw: ccj._run_line_search(cost, params, [None], [None], free, 1, 1e-3, 8, (1.0, 1.0, 1.0), **kw)[0][0]
    largest = lambda p: float(jnp.max(jnp.abs(p.wps - params[0].wps)))
    ok = report("no cap by default", largest(run()) > 1.5)
    ok &= report("a capped step stops at the cap", abs(largest(run(max_step=0.05)) - 0.05) < 1e-9)

    start = SplineParams(times=jnp.array([0.0, 1.0, 2.0]), wps=jnp.array([[0.0], [3.0], [0.0]]), vs=jnp.zeros((3, 1)))
    only_middle = jnp.array([0.0, 1.0, 0.0])
    steep = lambda params_list, _q0, _masks: (jnp.exp(10.0 * params_list[0].wps[1, 0]), {})
    final, _, _ = ccj._run_line_search(steep, [start], [None], [None], [(only_middle, 0.0 * only_middle, 0.0 * only_middle)],
                                       20, 1e-3, 8, (1.0, 1.0, 1.0), max_step=0.1)
    travelled = float(start.wps[1, 0] - final[0].wps[1, 0])
    ok &= report(f"being capped does not shrink later steps (moved {travelled:.2f} of 2.00)", abs(travelled - 2.0) < 1e-6)

    hold = lambda params_list, q0, masks: (cost(params_list, q0, masks)[0] + 10.0 * (params_list[0].times[0] - 0.7) ** 2, {})
    final, _, _ = ccj._run_line_search(hold, params, [None], [None], free, 5, 1e-3, 8, (1.0, 1.0, 1.0), max_step=0.1)
    return ok & report("a pinned row stays put under a real gradient", float(final[0].times[0]) == 0.0)


def check_guards():
    graph, *_ = solved_two_agent_problem()
    weights = CostWeights(inter_agent=0.0, environment=0.0, regularizer=0.0, repulsive=1.0)
    cases = [dict(weights=weights, line_search=True, max_step=0.1),
             dict(weights=weights, repulsive_kernel=KERNEL, max_step=0.1),
             dict(weights=weights, repulsive_kernel=KERNEL, line_search=True)]
    ok = True
    for overrides in cases:
        try:
            ccj.CollisionRefiner(graph, refinement_config(**overrides))
            ok = False
        except ValueError:
            pass
    return report("an unsafe or incomplete repulsive config is refused", ok)


def check_refiner():
    def run(**overrides):
        graph, splines, timing_mpc, x, x_dot = solved_two_agent_problem()
        return ccj.refine_splines(timing_mpc, graph, splines, x, x_dot, refinement_config(**overrides)), splines

    plain_diag, plain = run()
    off_diag, off = run(repulsive_kernel=KERNEL)
    same = all(np.array_equal(getattr(a.written, g), getattr(b.written, g))
               for a, b in zip(plain, off) for g in ("times", "wps", "vs"))
    ok = report("weight 0 leaves the refiner unchanged", same and off_diag["final"]["repulsive"] == 0.0)

    weights = CostWeights(inter_agent=0.0, environment=0.0, regularizer=0.0, repulsive=2.0)
    graph, splines, timing_mpc, x, x_dot = solved_two_agent_problem()
    before = [ccj.extract_agent_refinement_data(timing_mpc, graph, spline, i, x, x_dot, 16)
              for i, spline in enumerate(splines)]
    config = refinement_config(repulsive_kernel=KERNEL, weights=weights, line_search=True, max_step=0.1)
    on_diag = ccj.refine_splines(timing_mpc, graph, splines, x, x_dot, config)
    ok &= report(f"descent lowers the energy ({on_diag['initial']['repulsive']:.2f} -> "
                 f"{on_diag['final']['repulsive']:.2f})",
                 0.0 <= on_diag["final"]["repulsive"] < on_diag["initial"]["repulsive"])
    ok &= report("result accepted and moved", on_diag["accepted"] and on_diag["moved"])
    furthest = max(float(np.max(np.abs(s.written.wps - np.asarray(e.params.wps[:e.n_real])))) for s, e in zip(splines, before))
    ok &= report(f"knots stay in reach (moved {furthest:.2f} m)", 0.0 < furthest < 1.0)
    ok &= report("no end-time overrun", on_diag["end_time_overrun"] == 0.0)

    graph, splines, timing_mpc, x, x_dot = solved_two_agent_problem()
    before = [ccj.extract_agent_refinement_data(timing_mpc, graph, spline, i, x, x_dot, 16)
              for i, spline in enumerate(splines)]
    single = refinement_config(repulsive_kernel=KERNEL, weights=weights, line_search=True, max_step=0.05,
                               n_steps=1, initial_step=1.0)
    ccj.refine_splines(timing_mpc, graph, splines, x, x_dot, single)
    largest = max(float(np.max(np.abs(np.asarray(a) - b)))
                  for s, e in zip(splines, before)
                  for a, b in ((e.params.wps[:e.n_real], s.written.wps), (e.params.vs[:e.n_real], s.written.vs),
                               (e.params.times[:e.n_real], s.written.times)))
    ok &= report("max_step reaches the refiner from its config", abs(largest - 0.05) < 1e-9)

    bad_diag, _ = run(speed_fn=lambda q: jnp.nan * jnp.sum(q), weights=CostWeights(inter_agent=8.0, environment=1.0, regularizer=0.0))
    return ok & report("a rejected result reports moved=False", not bad_diag["accepted"] and not bad_diag["moved"])


def check_one_compile_across_knot_counts():
    """Runs `run` once per trace, so the counter reads compiles, across two
    cycles whose real knot counts differ."""
    traces = [0]
    real = ccj._run_line_search

    def counting(*args, **kwargs):
        traces[0] += 1
        return real(*args, **kwargs)

    ccj._run_line_search = counting
    try:
        graph, specs, starts, goals = build_two_agent_graph()
        splines = [CubicConfigurationSpline([s]) for s in specs]
        weights = CostWeights(inter_agent=0.0, environment=0.0, regularizer=0.0, repulsive=2.0)
        refiner = ccj.CollisionRefiner(graph, refinement_config(repulsive_kernel=KERNEL, weights=weights,
                                                                line_search=True, max_step=0.1))
        x, x_dot = np.concatenate([starts[0], starts[1]]), np.zeros(4)
        waypoints = np.zeros((4, graph.agent_col_offsets[-1]))
        waypoints[0, 0:2] = (2.0, 0.0)
        waypoints[1, 0:2] = goals[0]
        waypoints[2, 2:4] = (2.0, 0.0)
        waypoints[3, 2:4] = goals[1]
        counts = []
        for n_interior in (2, 6):
            timing_mpc = TracedTimingMPC(graph, splines, LinearTracer(n_interior=n_interior), time_cost=1.0)
            assert timing_mpc.solve(x, x_dot, [0, 1, 2, 3], waypoints, np.array([], dtype=np.int32))
            timing_mpc.fill_cubic_splines(splines, x, x_dot)
            counts.append([timing_mpc.get_agent_spline_length(i) for i in range(2)])
            refiner.refine(timing_mpc, splines, x, x_dot)
        return report(f"one compile across knot counts {counts}", counts[0] != counts[1] and traces[0] == 1)
    finally:
        ccj._run_line_search = real


def main():
    ok = True
    for check in (check_barrier_joins, check_barrier_closed_form, check_bad_kernels_raise,
                  check_gradient_through_contact, check_sample_weights, check_energy_by_hand,
                  check_energy_sums_everything, check_finite_differences, check_parked_agent_repels,
                  check_parked_agent_in_refiner, check_line_search_cap, check_guards, check_refiner,
                  check_one_compile_across_knot_counts):
        ok &= check()
    print("\nALL OK" if ok else "\nFAILURES ABOVE")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

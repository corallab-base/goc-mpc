"""Checks for `goc_mpc.sobolev_metric_jax`, the Sobolev-Slobodeckij metric in the constrained P2 step.

Covers the Gram matrix against a loop transcribed from the authors' `SobolevCurves::SobolevGramMatrix`, its
null space and scaling, the pulled-back metric, and the metric inside the constrained step and the refiner.

Run: python examples/test_sobolev_metric_jax.py
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
from goc_mpc import sobolev_metric_jax as sm
from goc_mpc.collision_cost_jax import CostWeights, build_cost_fn
from goc_mpc.constrained_step_jax import WallConstraint
from goc_mpc.sobolev_metric_jax import SobolevMetric
from test_collision_refinement_loop import refinement_config, solved_two_agent_problem
from test_constrained_step_jax import KERNEL, _scene

METRIC = SobolevMetric()


def report(name, ok):
    print(f"[sobolev] {name}: {'OK' if ok else 'FAIL'}")
    return bool(ok)


def _reference_gram(x, alpha, beta, mass):
    """Loop form of SobolevCurves::SobolevGramMatrix (sobo_slobo.cpp:178-257), without the mass term."""
    T = len(x)
    A = np.zeros((T, T))
    s = (beta - 1) / alpha
    p = 2 * (s - 1) + 1
    for i in range(T - 1):
        for j in range(T - 1):
            if abs(i - j) <= 1:
                continue
            li, lj = np.linalg.norm(x[i + 1] - x[i]), np.linalg.norm(x[j + 1] - x[j])
            ti, tj = (x[i + 1] - x[i]) / li, (x[j + 1] - x[j]) / lj
            mi, mj = 0.5 * (x[i] + x[i + 1]), 0.5 * (x[j] + x[j + 1])
            d = mi - mj
            r = np.linalg.norm(d)
            high = 1.0 / r ** p
            low = 0.5 * (np.linalg.norm(np.cross(ti, d)) ** 2 + np.linalg.norm(np.cross(tj, d)) ** 2) / r ** (4 + p)
            verts = (i, i + 1, j, j + 1)

            def hat_grad(edge, v, t, ell):
                return (t / ell if v == edge + 1 else -t / ell) if v in (edge, edge + 1) else np.zeros(3)

            def hat_mid(edge, v):
                return 0.5 if v in (edge, edge + 1) else 0.0

            for u in verts:
                for v in verts:
                    gu = hat_grad(i, u, ti, li) - hat_grad(j, u, tj, lj)
                    gv = hat_grad(i, v, ti, li) - hat_grad(j, v, tj, lj)
                    A[u, v] += gu @ gv * high * li * lj
                    A[u, v] += (hat_mid(i, u) - hat_mid(j, u)) * (hat_mid(i, v) - hat_mid(j, v)) * low * li * lj
    return A


def check_gram():
    rng = np.random.default_rng(0)
    ok = True
    for T in (6, 9):
        x = np.cumsum(rng.normal(size=(T, 3)), axis=0)
        ref = _reference_gram(x, METRIC.alpha, METRIC.beta, 0.0)
        A = np.asarray(sm.gram(jnp.asarray(x), METRIC._replace(eps=1e-30)))
        err = np.max(np.abs(A - ref)) / np.max(np.abs(ref))
        ok &= report(f"Gram matches the authors' loop, T = {T} (rel {err:.1e})", err < 1e-10)
    x = jnp.asarray(np.cumsum(rng.normal(size=(12, 3)), axis=0))
    S = sm.gram(x, METRIC._replace(eps=1e-30))
    ok &= report("constants are in the null space of B + B0", float(jnp.max(jnp.abs(S @ jnp.ones(12)))) < 1e-10 * float(jnp.max(jnp.abs(S))))
    w = np.linalg.eigvalsh(np.asarray(sm.gram(x, METRIC)))
    ok &= report(f"with the mass term it is positive definite (min eig {w[0]:.2e})", w[0] > 0)
    # Scaling: l_I l_J |m|^-(2 sigma + 1) |D u|^2 with D ~ 1 / length gives lambda^(-2 sigma - 1).
    lam = 3.0
    u = jnp.arange(12.0)
    r = float(jnp.vdot(u, sm.gram(lam * x, METRIC._replace(eps=1e-30)) @ u) / jnp.vdot(u, S @ u))
    expect = lam ** (-(2 * sm.sigma(METRIC) + 1))
    return ok & report(f"B scales as length^-(2 sigma + 1) ({r:.4f} vs {expect:.4f})", abs(r / expect - 1) < 0.05)


def check_pullback():
    geometry, params, masks, free, _ = _scene()
    t_grid = jnp.linspace(0.0, 6.0, 120)
    p = params[0]
    dm = cs.metric_diagonal(p, free[0], (1.0, 1.0, 0.25))

    def points(x):
        pp = cs._unflat(x, p)
        q = ccj.sample_configs(geometry[0], pp, t_grid, masks[0])
        return jnp.mean(jax.vmap(geometry[0].fk_fn)(q), axis=1)

    chol = sm.pulled_back(points, cs._flat(p), t_grid, dm, METRIC)
    ok = report("pulled-back metric factors (finite Cholesky)", bool(jnp.all(jnp.isfinite(chol))))
    g = jnp.asarray(np.random.default_rng(1).normal(size=dm.shape[0]))
    d = -sm.apply_inverse(chol, (dm > 0).astype(float), g)
    ok &= report("pinned entries get no step", float(jnp.max(jnp.abs(jnp.where(dm > 0, 0.0, d))) ) == 0.0)
    return ok & report("the step descends", float(g @ d) < 0.0)


def _run(params, sobolev, steps):
    geometry, _, masks, free, clearance = _scene()
    t_grid = jnp.linspace(0.0, 6.0, 200)
    weights = CostWeights(inter_agent=0.0, environment=0.0, regularizer=0.0, repulsive=8.0)
    cost = build_cost_fn(geometry, t_grid, lambda q: 1.0, weights, repulsive_kernel=KERNEL)
    q0 = [ccj.sample_configs(g, p, t_grid, m) for g, p, m in zip(geometry, params, masks)]
    final, hist, _ = cs._run_constrained(cost, geometry, clearance, t_grid, params, q0, masks, free, steps, 1e-3, 8,
                                         (1.0, 1.0, 0.25), WallConstraint(), max_step=0.1,
                                         project_times=ccj._project_knot_times, sobolev=sobolev)
    q = ccj.sample_configs(geometry[0], final[0], jnp.linspace(0.0, 6.0, 2000), masks[0])
    return float(jnp.min(jax.vmap(clearance)(q))), np.asarray(hist), float(cost(final, q0, masks)[0])


def check_constrained_with_metric():
    _, params, *_ = _scene()
    wall, hist, final = _run(params, METRIC, 30)
    ok = report(f"metric step stays out of the wall ({wall:+.4f} m)", wall > -5e-3)
    ok &= report(f"and lowers the energy ({hist[0]:.2f} -> {final:.3f})", final < hist[0])
    for steps in (5, 10, 20):
        e_unit = _run(params, None, steps)[2]
        e_sob = _run(params, METRIC, steps)[2]
        print(f"[sobolev]   energy after {steps:2d} steps: unit {e_unit:8.3f}, Sobolev {e_sob:8.3f}")
    return ok


def check_refiner():
    graph, splines, timing_mpc, x, x_dot = solved_two_agent_problem()
    weights = CostWeights(inter_agent=0.0, environment=0.0, regularizer=0.0, repulsive=2.0)
    config = refinement_config(repulsive_kernel=KERNEL, weights=weights, line_search=True, max_step=0.1,
                               clearance_fn=lambda q: 10.0 - jnp.abs(q[1]), wall_constraint=WallConstraint(),
                               sobolev_metric=METRIC)
    diag = ccj.refine_splines(timing_mpc, graph, splines, x, x_dot, config)
    ok = report(f"refiner runs with the metric ({diag['initial']['repulsive']:.2f} -> {diag['final']['repulsive']:.2f})",
                diag["accepted"] and diag["final"]["repulsive"] < diag["initial"]["repulsive"])
    refused = 0
    for bad in (dict(wall_constraint=None), dict(sobolev_metric=METRIC._replace(beta=2.0)),
                dict(sobolev_metric=METRIC._replace(strength=0.0))):
        try:
            ccj.CollisionRefiner(graph, config._replace(**bad))
        except ValueError:
            refused += 1
    return ok & report("unusable metric configs are refused at build", refused == 3)


def main():
    ok = True
    for check in (check_gram, check_pullback, check_constrained_with_metric, check_refiner):
        ok &= check()
    print("\nALL OK" if ok else "\nFAILURES ABOVE")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

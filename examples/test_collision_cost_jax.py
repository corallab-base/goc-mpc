"""Validates goc_mpc.collision_cost_jax's forward pass (Step 2 of
PLAN_collision_aware_timing.md): builds two synthetic agents on
deliberately crossing paths, checks the forward pass is finite/jit/grad-
compatible, cross-checks jax autodiff against a central finite difference
of the whole (spline -> FK -> cost) composition, and runs a short plain-
gradient-descent demo showing the inter-agent term actually falls as the
refinement pushes the two agents apart.

Uses a synthetic sphere FK and speed field for this validation only -- the
real per-robot FK/environment field are robot/environment-specific and
live in po_goc_mpc (see collision_cost_jax.py's own doc comment), not
here.

Run: python examples/test_collision_cost_jax.py
"""

import numpy as np
import jax
import jax.numpy as jnp

from goc_mpc.splines import Block
from goc_mpc import spline_jax
from goc_mpc.collision_cost_jax import (
    AgentGeometry, SplineParams, CostWeights, build_cost_fn, sample_configs,
)

SPEC = [Block.R(2), Block.Torus(1)]  # matches every b1_multiroom agent
BODY_OFFSETS = jnp.array([[-0.3, 0.0], [0.3, 0.0]])  # two spheres, body frame
SPHERE_RADII = jnp.array([0.25, 0.25])


def two_sphere_fk(q):
    """Toy SE(2) rigid-sphere FK for validation only: two spheres at fixed
    body-frame offsets along local x, rotated by yaw q[2] and translated by
    q[:2]. Ambient (x, y, yaw) -> (2, 3) world-frame sphere centers
    (z fixed at 0)."""
    pos, yaw = q[:2], q[2]
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    rot = jnp.array([[c, -s], [s, c]])
    centers_2d = pos[None, :] + BODY_OFFSETS @ rot.T
    return jnp.concatenate([centers_2d, jnp.zeros((2, 1))], axis=-1)


def gaussian_dip_speed(q, center=jnp.array([2.5, 0.5]), sigma=0.6):
    """Toy environment speed field: dips near `center`, floor at 0.1."""
    d2 = jnp.sum((q[:2] - center) ** 2)
    return 0.1 + 0.9 * (1.0 - jnp.exp(-d2 / (2 * sigma ** 2)))


def make_agent(y0, y1, n=6, seed=0):
    """A near-straight-line trajectory along x, from (0, y0) to (5, y1),
    small per-knot yaw/velocity noise so it's a generic cubic, not a
    degenerate straight segment."""
    r = np.random.default_rng(seed)
    times = jnp.array(np.linspace(0.0, 5.0, n))
    xs = np.linspace(0.0, 5.0, n)
    ys = np.linspace(y0, y1, n)
    yaw = np.arctan2(np.gradient(ys), np.gradient(xs)) + r.normal(0, 0.02, n)
    wps = jnp.array(np.stack([xs, ys, yaw], axis=-1))
    vx = np.gradient(xs, times)
    vy = np.gradient(ys, times)
    vyaw = np.gradient(yaw, times)
    vs = jnp.array(np.stack([vx, vy, vyaw], axis=-1))
    return SplineParams(times=times, wps=wps, vs=vs)


def main():
    all_ok = True

    geom = AgentGeometry(spec=SPEC, fk_fn=two_sphere_fk, sphere_radii=SPHERE_RADII)
    geometries = [geom, geom]

    # Two agents on paths that cross near the middle of the horizon.
    params_a = make_agent(y0=-1.0, y1=1.0, seed=1)
    params_b = make_agent(y0=1.0, y1=-1.0, seed=2)
    params_list = [params_a, params_b]

    t_grid = jnp.linspace(0.0, 5.0, 40)
    q0_samples_list = [sample_configs(g, p, t_grid) for g, p in zip(geometries, params_list)]

    weights = CostWeights(inter_agent=5.0, environment=0.1, regularizer=0.05)
    cost_fn = build_cost_fn(geometries, t_grid, gaussian_dip_speed, weights)

    # --- finiteness + breakdown shape ---
    total, breakdown = cost_fn(params_list, q0_samples_list)
    finite = bool(jnp.isfinite(total)) and all(bool(jnp.isfinite(v)) for v in breakdown.values())
    breakdown_f = {k: float(v) for k, v in breakdown.items()}
    print(f"[forward] total={float(total):.4f} breakdown={breakdown_f}")
    print(f"[forward] finite: {'OK' if finite else 'FAIL'}")
    all_ok &= finite
    assert breakdown["inter_agent"] > 0, "expected the two crossing paths to actually overlap initially"

    # --- jit + value_and_grad compiles and produces gradients of the right shape ---
    vg = jax.jit(jax.value_and_grad(cost_fn, argnums=0, has_aux=True))
    (total_j, breakdown_j), grads = vg(params_list, q0_samples_list)
    shapes_ok = all(g.wps.shape == p.wps.shape and g.vs.shape == p.vs.shape and g.times.shape == p.times.shape
                     for g, p in zip(grads, params_list))
    print(f"[jit+grad] compiles, matches eager: {'OK' if jnp.allclose(total_j, total) else 'FAIL'}; "
          f"grad shapes match params: {'OK' if shapes_ok else 'FAIL'}")
    all_ok &= bool(jnp.allclose(total_j, total)) and shapes_ok

    # --- finite-difference cross-check of the whole composition's gradient ---
    eps = 1e-5
    rng = np.random.default_rng(3)
    n_probes = 8
    max_err = 0.0
    for _ in range(n_probes):
        ai = rng.integers(0, 2)
        i, j = rng.integers(0, params_list[ai].wps.shape[0]), rng.integers(0, 3)

        def cost_with_perturbed_wps(delta):
            perturbed = params_list[ai].wps.at[i, j].add(delta)
            plist = list(params_list)
            plist[ai] = params_list[ai]._replace(wps=perturbed)
            total, _ = cost_fn(plist, q0_samples_list)
            return total

        fd = (float(cost_with_perturbed_wps(eps)) - float(cost_with_perturbed_wps(-eps))) / (2 * eps)
        analytic = float(grads[ai].wps[i, j])
        err = abs(fd - analytic)
        max_err = max(max_err, err)
    fd_ok = max_err < 1e-3
    print(f"[fd-check] max |analytic - central-diff| over {n_probes} wps probes: {max_err:.3e} "
          f"{'OK' if fd_ok else 'FAIL'}")
    all_ok &= fd_ok

    # --- mini end-to-end demo: gradient descent should shrink inter-agent overlap ---
    # freeze_times=True here (not a bare freeze_times(p) call before vg) --
    # see freeze_times's own doc comment: applying it to params BEFORE
    # handing them to jax.grad as the differentiated argument is a no-op
    # (verified: it does not actually zero grad.times), only applying it
    # INSIDE the traced cost function (what build_cost_fn's flag does) does.
    step = 2e-3
    cost_fn_frozen = build_cost_fn(geometries, t_grid, gaussian_dip_speed, weights, freeze_times=True)
    vg_frozen = jax.jit(jax.value_and_grad(cost_fn_frozen, argnums=0, has_aux=True))
    params_gd = list(params_list)
    inter_history = []
    for it in range(40):
        (_, bd), g = vg_frozen(params_gd, q0_samples_list)
        inter_history.append(float(bd["inter_agent"]))
        params_gd = [SplineParams(times=p.times,
                                   wps=p.wps - step * gr.wps,
                                   vs=p.vs - step * gr.vs)
                     for p, gr in zip(params_gd, g)]
    improved = inter_history[-1] < 0.5 * inter_history[0]
    print(f"[gd-demo] inter_agent cost: {inter_history[0]:.4f} -> {inter_history[-1]:.4f} "
          f"{'OK (reduced)' if improved else 'FAIL (did not shrink)'}")
    all_ok &= improved

    print("\nALL OK" if all_ok else "\nFAILURES ABOVE")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

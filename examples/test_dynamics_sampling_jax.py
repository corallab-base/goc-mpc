"""Regression test for collision_cost_jax.dynamics_cost's `sampling` option.

A cubic Hermite spline is C1, so its acceleration jumps at knots. With
sampling="grid" the dynamics hinge is scored on fixed grid times and at the
knots, so when a knot sits on a grid sample the smallest move of that knot
flips the sample into the other piece and the cost jumps. On B1 cycle 0 this
made the Armijo line search reject every step for all agents. sampling=
"segment" scores fixed fractions inside each knot interval, which keeps the
cost smooth in the knots.

Checks:
  1. "grid" is the default, in dynamics_cost, build_cost_fn and
     CollisionRefinementConfig, so existing runs are unchanged.
  2. With a knot on a grid sample, "grid" jumps by O(1) across a 1e-7 move of
     that knot, while "segment" changes by O(1e-7).
  3. "segment" autodiff matches a central finite difference in every knot
     variable at that same point.
  4. "segment" samples only inside knot intervals and never at a knot.
  5. The line search lowers the "segment" cost from that start.
  6. CollisionRefiner refuses an unknown sampling name.

Run: python examples/test_dynamics_sampling_jax.py
"""

import inspect

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from goc_mpc.splines import Block  # noqa: E402
from goc_mpc import collision_cost_jax as ccj  # noqa: E402
from goc_mpc.collision_cost_jax import (  # noqa: E402
    AgentGeometry, SplineParams, CostWeights, CollisionRefinementConfig, build_cost_fn, dynamics_cost,
    sample_configs, segment_times, SEGMENT_FRACTIONS,
)

MAX_ACCEL = 0.5


def accel_hinge(q, v, a):
    return jnp.maximum(jnp.sqrt(jnp.sum(a[:2] ** 2) + 1e-12) - MAX_ACCEL, 0.0) ** 2


def zigzag():
    """A 2D path with sharp turns, so the acceleration is well above the cap and differs on both sides of
    every interior knot."""
    times = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
    wps = jnp.array([[0.0, 0.0], [1.0, 0.8], [2.0, -0.6], [3.0, 0.7], [4.0, 0.0]])
    vs = jnp.array([[0.0, 0.0], [1.2, -0.3], [0.9, 0.4], [1.1, -0.2], [0.0, 0.0]])
    return SplineParams(times=times, wps=wps, vs=vs)


def with_time(p, k, t):
    return SplineParams(times=p.times.at[k].set(t), wps=p.wps, vs=p.vs)


def main():
    ok_all = True

    def check(name, ok, detail=""):
        nonlocal ok_all
        ok_all &= bool(ok)
        print(f"[{'OK' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))

    geom = AgentGeometry(spec=[Block.R(2)], fk_fn=lambda q: jnp.concatenate([q, jnp.zeros(1)])[None, :],
                         sphere_radii=jnp.array([0.2]))
    p0 = zigzag()
    t_grid = jnp.linspace(0.0, 4.0, 41)  # step 0.1, so knot 2 at t = 2.0 is grid sample 20

    # 1. defaults
    defaults = [inspect.signature(dynamics_cost).parameters["sampling"].default,
                inspect.signature(build_cost_fn).parameters["dynamics_sampling"].default,
                CollisionRefinementConfig._field_defaults["dynamics_sampling"]]
    check("grid is the default everywhere", defaults == ["grid"] * 3, str(defaults))

    # 2. jump across a knot that sits on a grid sample
    k, eps = 2, 1e-7
    assert float(jnp.min(jnp.abs(t_grid - p0.times[k]))) == 0.0

    def cost(p, sampling):
        return dynamics_cost(accel_hinge, geom, p, t_grid, sampling=sampling)

    jumps = {}
    for sampling in ("grid", "segment"):
        lo = float(cost(with_time(p0, k, p0.times[k] - eps), sampling))
        hi = float(cost(with_time(p0, k, p0.times[k] + eps), sampling))
        jumps[sampling] = abs(hi - lo)
    check("grid cost jumps across a 1e-7 knot move", jumps["grid"] > 1e-2, f"{jumps['grid']:.3e}")
    check("segment cost is continuous there", jumps["segment"] < 1e-4, f"{jumps['segment']:.3e}")

    # 3. segment gradient matches finite differences at the same point
    flat0, unravel = jax.flatten_util.ravel_pytree(p0)
    f = jax.jit(lambda x: cost(unravel(x), "segment"))
    g = np.asarray(jax.grad(f)(flat0))
    h = 1e-6
    fd = np.array([(float(f(flat0.at[i].add(h))) - float(f(flat0.at[i].add(-h)))) / (2 * h)
                   for i in range(flat0.shape[0])])
    err = float(np.max(np.abs(g - fd)) / max(1.0, float(np.max(np.abs(fd)))))
    check("segment autodiff matches central FD", err < 1e-5, f"rel err {err:.2e}")

    # 4. segment samples stay strictly inside knot intervals
    t = np.asarray(segment_times(p0.times, SEGMENT_FRACTIONS)).reshape(len(p0.times) - 1, -1)
    lo, hi = np.asarray(p0.times[:-1])[:, None], np.asarray(p0.times[1:])[:, None]
    check("segment samples lie strictly inside each interval", bool(np.all((t > lo) & (t < hi))))

    # 5. the line search makes progress on the segment cost from this start
    weights = CostWeights(inter_agent=0.0, environment=0.0, regularizer=0.0, dynamics=1.0)
    free = (jnp.array([False, True, True, True, False]), jnp.array([False, True, True, True, False]),
            jnp.array([False, True, True, True, True]))
    q0 = [sample_configs(geom, p0, t_grid)]
    for sampling in ("segment",):
        cost_fn = build_cost_fn([geom], t_grid, lambda q: 1.0, weights, dynamics_fn=accel_hinge,
                                dynamics_sampling=sampling)
        final, hist, _ = ccj._run_line_search(cost_fn, [p0], q0, [None], [free], n_steps=20, initial_step=1e-2,
                                              max_backtracks=12, metric_scale=(1.0, 1.0, 0.25), max_step=0.1)
        before = float(cost_fn([p0], q0)[0])
        after = float(cost_fn(final, q0)[0])
        check(f"line search lowers the {sampling} cost", after < 0.9 * before, f"{before:.4f} -> {after:.4f}")

    # 6. config validation
    try:
        ccj.CollisionRefiner.__init__(object.__new__(ccj.CollisionRefiner), None,
                                      CollisionRefinementConfig(fk_fns=[], sphere_radii=[], speed_fn=None,
                                                                dynamics_sampling="knots"))
        refused = False
    except ValueError as e:
        refused = "dynamics_sampling" in str(e)
    except Exception as e:  # noqa: BLE001
        refused = False
        print("   unexpected", type(e).__name__, e)
    check("refiner refuses an unknown sampling", refused)

    print("\nALL OK" if ok_all else "\nFAILURES ABOVE")
    raise SystemExit(0 if ok_all else 1)


if __name__ == "__main__":
    import jax.flatten_util  # noqa: F401
    main()

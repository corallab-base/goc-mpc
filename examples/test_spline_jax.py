"""Validates goc_mpc.spline_jax.eval against the C++
CubicConfigurationSpline::eval it reimplements -- PLAN_collision_aware_
timing.md's Step 1 validation criterion: value agreement plus a finite-
difference check of the jax-autodiff gradient (w.r.t. wps/vs/times) against
central differences of the C++ eval. Covers both the plain case and the
per-block knot-bridging (`block_active_mask`) case, and both block types
this module supports (R, Torus).

Run: python examples/test_spline_jax.py
"""

import numpy as np
import jax
import jax.numpy as jnp

from goc_mpc.splines import Block, CubicConfigurationSpline
from goc_mpc import spline_jax

rng = np.random.default_rng(0)


def make_case(n=6, mask=None, seed=0):
    r = np.random.default_rng(seed)
    spec = [Block.R(2), Block.Torus(1)]
    times = np.sort(r.uniform(0.0, 1.0, size=n)) * 5.0
    times += np.arange(n) * 1e-2  # guarantee strictly increasing after sort
    wps = r.uniform(-3.0, 3.0, size=(n, 3))
    vs = r.uniform(-1.0, 1.0, size=(n, 3))

    spline = CubicConfigurationSpline(spec)
    if mask is not None:
        spline.set_block_active_mask(mask)
    spline.set(wps, vs, times)
    return spec, times, wps, vs, spline


def cpp_eval_batch(spline, ts):
    qs, vs_, as_ = [], [], []
    for t in ts:
        q, v, a = spline.eval(float(t))
        qs.append(q); vs_.append(v); as_.append(a)
    return np.stack(qs), np.stack(vs_), np.stack(as_)


def check_values(label, spec, times, wps, vs, spline, mask=None):
    t_samples = np.concatenate([
        times,  # exact knot times (piece-boundary edge cases)
        rng.uniform(times[0], times[-1], size=40),
        [times[0] - 0.5, times[-1] + 0.5],  # outside range -> clamping
    ])

    cpp_q, cpp_v, cpp_a = cpp_eval_batch(spline, t_samples)
    active_mask = None if mask is None else jnp.array(mask)
    jax_q, jax_v, jax_a = spline_jax.eval(
        spec, jnp.array(times), jnp.array(wps), jnp.array(vs),
        jnp.array(t_samples), active_mask=active_mask)

    ok = True
    for name, cpp_arr, jax_arr in [("q", cpp_q, jax_q), ("v", cpp_v, jax_v), ("a", cpp_a, jax_a)]:
        err = np.max(np.abs(cpp_arr - np.asarray(jax_arr)))
        passed = err < 1e-8
        ok &= passed
        print(f"[{label}] {name}: max abs err = {err:.3e} {'OK' if passed else 'FAIL'}")
    return ok


def check_gradients(label, spec, times, wps, vs, mask=None, t_fixed=None, n_probes=6, eps=1e-6):
    """Central-difference the C++ eval's q(t) w.r.t. random single entries of
    wps/vs/times, compare against jax.jacobian of spline_jax.eval at the
    same entries."""
    n = times.shape[0]
    if t_fixed is None:
        t_fixed = rng.uniform(times[0], times[-1])
    t_arr = jnp.array([t_fixed])
    active_mask = None if mask is None else jnp.array(mask)

    def q_of(times_, wps_, vs_):
        q, _, _ = spline_jax.eval(spec, times_, wps_, vs_, t_arr, active_mask=active_mask)
        return q[0]

    jac_times = jax.jacobian(q_of, argnums=0)(jnp.array(times), jnp.array(wps), jnp.array(vs))
    jac_wps = jax.jacobian(q_of, argnums=1)(jnp.array(times), jnp.array(wps), jnp.array(vs))
    jac_vs = jax.jacobian(q_of, argnums=2)(jnp.array(times), jnp.array(wps), jnp.array(vs))

    def cpp_q_at(times_, wps_, vs_, t):
        spline = CubicConfigurationSpline(spec)
        if mask is not None:
            spline.set_block_active_mask(mask)
        spline.set(wps_, vs_, times_)
        q, _, _ = spline.eval(float(t))
        return q

    ok = True
    for _ in range(n_probes):
        kind = rng.choice(["times", "wps", "vs"])
        if kind == "times":
            i = rng.integers(1, n - 1)  # keep endpoints away from strict-increasing edge cases
            base = times.copy()
            plus, minus = base.copy(), base.copy()
            plus[i] += eps; minus[i] -= eps
            fd = (cpp_q_at(plus, wps, vs, t_fixed) - cpp_q_at(minus, wps, vs, t_fixed)) / (2 * eps)
            analytic = np.asarray(jac_times[:, i])
        elif kind == "wps":
            i, j = rng.integers(0, n), rng.integers(0, wps.shape[1])
            plus, minus = wps.copy(), wps.copy()
            plus[i, j] += eps; minus[i, j] -= eps
            fd = (cpp_q_at(times, plus, vs, t_fixed) - cpp_q_at(times, minus, vs, t_fixed)) / (2 * eps)
            analytic = np.asarray(jac_wps[:, i, j])
        else:
            i, j = rng.integers(0, n), rng.integers(0, vs.shape[1])
            plus, minus = vs.copy(), vs.copy()
            plus[i, j] += eps; minus[i, j] -= eps
            fd = (cpp_q_at(times, wps, plus, t_fixed) - cpp_q_at(times, wps, minus, t_fixed)) / (2 * eps)
            analytic = np.asarray(jac_vs[:, i, j])

        err = np.max(np.abs(fd - analytic))
        passed = err < 1e-4
        ok &= passed
        print(f"[{label}] d(q)/d({kind}) probe: max abs err = {err:.3e} {'OK' if passed else 'FAIL'}")
    return ok


def main():
    all_ok = True

    spec, times, wps, vs, spline = make_case(n=6, seed=1)
    all_ok &= check_values("plain", spec, times, wps, vs, spline)
    all_ok &= check_gradients("plain", spec, times, wps, vs)

    # Bridging case: skip an interior knot for the Torus block, another for
    # the R block, endpoints forced active regardless of mask content.
    n = 7
    mask = np.ones((n, 2), dtype=np.int32)
    mask[2, 1] = 0  # bridge Torus block across knot 2
    mask[4, 0] = 0  # bridge R block across knot 4
    mask[0, :] = 0  # should be force-overridden back to active
    mask[-1, :] = 0
    spec, times, wps, vs, spline = make_case(n=n, mask=mask, seed=2)
    all_ok &= check_values("bridged", spec, times, wps, vs, spline, mask=mask)
    all_ok &= check_gradients("bridged", spec, times, wps, vs, mask=mask)

    print("\nALL OK" if all_ok else "\nFAILURES ABOVE")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

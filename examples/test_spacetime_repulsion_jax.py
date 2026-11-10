"""Checks for `goc_mpc.spacetime_repulsion_jax`, the space-time tangent-point energy.

Covers the radial factor's joins, Eq 18 against a brute-force loop, convergence to the continuum
energy of two parked agents, banding, invariances, gradients, the cross-time pairs the same-time
barrier cannot see, and the term inside the refiner.

Run: python examples/test_spacetime_repulsion_jax.py
"""

import itertools
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from goc_mpc import collision_cost_jax as ccj
from goc_mpc import spacetime_repulsion_jax as st
from goc_mpc.collision_cost_jax import AgentGeometry, CostWeights, SplineParams, build_cost_fn
from goc_mpc.repulsive_energy_jax import RepulsiveKernel, repulsive_energy
from goc_mpc.spacetime_repulsion_jax import SpaceTimeKernel, radial, spacetime_energy
from goc_mpc.splines import Block
from test_collision_refinement_loop import refinement_config, solved_two_agent_problem

KERNEL = SpaceTimeKernel(alpha=2.0, beta=4.5, time_scale=1.0, cutoff=0.25, relax=0.02, reference_gap=0.1)


def report(name, ok):
    print(f"[spacetime] {name}: {'OK' if ok else 'FAIL'}")
    return bool(ok)


def close(x, y, rel=1e-9):
    return abs(float(x) - float(y)) <= rel * max(abs(float(x)), abs(float(y)), 1e-300)


def check_radial_joins():
    value = lambda g: radial(jnp.asarray(g), KERNEL)
    slope, curve = jax.grad(value), jax.grad(jax.grad(value))
    ok = report("zero and flat at and past the cutoff",
                float(value(0.25)) == 0.0 and float(value(1.0)) == 0.0 and float(slope(0.3)) == 0.0)
    h = 1e-7
    ok &= report("value and slope reach zero at the cutoff",
                 abs(float(value(0.25 - h))) < 1e-9 and abs(float(slope(0.25 - h))) < 1e-4)
    a, b = 0.02 - 1e-9, 0.02 + 1e-9
    near = lambda x, y: abs(float(x) - float(y)) <= 1e-5 * max(abs(float(x)), 1.0)
    ok &= report("value, slope and curvature match at relax", near(value(a), value(b)) and near(slope(a), slope(b))
                 and near(curve(a), curve(b)))
    values = np.asarray(radial(jnp.linspace(-1.0, 0.25, 600), KERNEL))
    ok &= report("strictly decreasing up to the cutoff", np.all(np.diff(values) < 0.0))
    ok &= report("force is 1 at the reference gap", abs(float(slope(0.1)) + 1.0) < 1e-10)
    return ok & report("finite gradient through deep overlap", np.isfinite(float(slope(-0.5))) and float(slope(-0.5)) < 0)


def check_bad_kernels_raise():
    bad = [dict(beta=2.0), dict(time_scale=0.0), dict(relax=0.3), dict(reference_gap=0.3), dict(band=-1), dict(alpha=-1.0)]
    ok = True
    for overrides in bad:
        try:
            st.check_kernel(KERNEL._replace(**overrides))
            ok = False
        except ValueError:
            pass
    return report("invalid kernels are refused", ok)


def _random_pair(seed, T=7, sa=2, sb=3, dim=3, spread=0.3):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(T, sa, dim)) * spread
    b = rng.normal(size=(T, sb, dim)) * spread + 0.1
    t = np.cumsum(rng.uniform(0.05, 0.2, T))
    return a, b, rng.uniform(0.05, 0.12, sa), rng.uniform(0.05, 0.12, sb), t


def _brute(a, b, ra, rb, t, k):
    """Eq 17-18 written out as loops over edges, spheres and endpoints, in numpy."""
    c = k.time_scale
    P = lambda x, i: np.concatenate([x[i], [c * t[i]]])

    def rad(g):
        return float(radial(jnp.asarray(g), k))

    def sinp(d, T):
        s2 = max(0.0, 1.0 - (d @ T) ** 2 / (d @ d + 1e-12))
        return s2 ** (k.alpha / 2)

    total = 0.0
    for I, J in itertools.product(range(len(t) - 1), repeat=2):
        if k.band is not None and abs(I - J) > k.band:
            continue
        for si, sj in itertools.product(range(a.shape[1]), range(b.shape[1])):
            ea = P(a[:, si], I + 1) - P(a[:, si], I)
            eb = P(b[:, sj], J + 1) - P(b[:, sj], J)
            la, lb = np.linalg.norm(ea), np.linalg.norm(eb)
            for i, j in itertools.product((I, I + 1), (J, J + 1)):
                d = P(a[:, si], i) - P(b[:, sj], j)
                g = np.sqrt(d @ d + 1e-12) - ra[si] - rb[sj]
                total += 0.25 * (sinp(d, ea / la) + sinp(d, eb / lb)) * rad(g) * la * lb
    return total / c


def check_against_brute_force():
    ok = True
    for seed, kernel in ((0, KERNEL), (1, KERNEL._replace(alpha=3.0, beta=6.0, time_scale=0.7)),
                         (2, KERNEL._replace(band=2))):
        a, b, ra, rb, t = _random_pair(seed)
        got = float(spacetime_energy([jnp.asarray(a), jnp.asarray(b)], [jnp.asarray(ra), jnp.asarray(rb)],
                                     jnp.asarray(t), kernel))
        want = _brute(a, b, ra, rb, t, kernel)
        ok &= report(f"Eq 18 matches a brute-force loop (seed {seed}, {got:.6g} vs {want:.6g})",
                     want > 0.0 and close(got, want, 1e-9))
    return ok


def _parked(gap, L, T, kernel, radius=0.1):
    t = jnp.linspace(0.0, L, T)
    a = jnp.zeros((T, 1, 2))
    b = jnp.zeros((T, 1, 2)).at[..., 0].set(gap + 2 * radius)
    r = jnp.array([radius])
    return float(spacetime_energy([a, b], [r, r], t, kernel))


def _parked_continuum(gap, L, kernel, radius=0.1):
    """E = int (L - |u| / c) K(u) du for two vertical lines at spatial distance s, u = c * dt."""
    s = gap + 2 * radius
    reach = np.sqrt((kernel.cutoff + 2 * radius) ** 2 - s ** 2)
    u = np.linspace(-reach, reach, 200001)
    dist = np.sqrt(s * s + u * u)
    K = (s * s / (dist * dist)) ** (kernel.alpha / 2) * 2 * np.asarray(radial(jnp.asarray(dist - 2 * radius), kernel))
    w = L - np.abs(u) / kernel.time_scale
    return float(np.trapezoid(w * K, u))


def check_continuum_limit():
    ok = True
    for c in (1.0, 4.0):
        k = KERNEL._replace(time_scale=c)
        want = _parked_continuum(0.08, 3.0, k)
        errs = [abs(_parked(0.08, 3.0, T, k) - want) / want for T in (int(60 * c), int(120 * c), int(480 * c))]
        ok &= report(f"parked pair converges to the continuum energy at c={c:g} (errors {', '.join(f'{e:.1e}' for e in errs)})",
                     errs[2] < 2e-3 and errs[2] < errs[1] < errs[0])
    # Large c is the same-time limit: the band of cross-time pairs shrinks but E stays finite.
    e1, e4 = _parked_continuum(0.08, 30.0, KERNEL), _parked_continuum(0.08, 30.0, KERNEL._replace(time_scale=4.0))
    return ok & report(f"energy tends to a finite same-time limit as c grows ({e1:.4f}, {e4:.4f})", abs(e1 - e4) / e4 < 0.01)


def check_band():
    a, b, ra, rb, t = _random_pair(3, T=40, spread=0.2)
    t = jnp.linspace(0.0, 4.0, 40)
    args = ([jnp.asarray(a), jnp.asarray(b)], [jnp.asarray(ra), jnp.asarray(rb)], t)
    dense = float(spacetime_energy(*args, KERNEL))
    exact = [k for k in range(1, 30) if st.band_is_exact(t, KERNEL._replace(band=k), float(ra.max() + rb.max()))][0]
    banded = float(spacetime_energy(*args, KERNEL._replace(band=exact)))
    short = float(spacetime_energy(*args, KERNEL._replace(band=0)))
    ok = report(f"band {exact} reported exact and equals dense", close(dense, banded, 1e-12))
    ok &= report("a band reported inexact can differ", not st.band_is_exact(t, KERNEL._replace(band=0), 0.2)
                 and short != dense)
    # Short last intervals: the dropped pairs (0,3), (0,4), (1,4) are 1.1 s or more apart, past 0.8.
    uneven = np.array([0.0, 1.0, 2.0, 3.0, 3.1, 3.2])
    return ok & report("band check ignores edges past the end", st.band_is_exact(uneven, KERNEL._replace(time_scale=1.0, cutoff=0.3, band=2), 0.5))


def check_reference_below_relax():
    k = KERNEL._replace(relax=0.15, reference_gap=0.1)
    slope = jax.grad(lambda g: radial(jnp.asarray(g), k))
    return report("force is 1 at a reference gap below relax", abs(float(slope(0.1)) + 1.0) < 1e-10)


def check_invariances():
    a, b, ra, rb, t = _random_pair(5, T=10)
    E = lambda a_, b_, t_: float(spacetime_energy([jnp.asarray(a_), jnp.asarray(b_)], [jnp.asarray(ra), jnp.asarray(rb)],
                                                   jnp.asarray(t_), KERNEL))
    base = E(a, b, t)
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    ok = report("spatial translation", close(base, E(a + [1.0, -2.0, 0.5], b + [1.0, -2.0, 0.5], t), 1e-10))
    ok &= report("spatial rotation", close(base, E(a @ R.T, b @ R.T, t), 1e-10))
    ok &= report("time shift", close(base, E(a, b, t + 5.0), 1e-10))
    swapped = float(spacetime_energy([jnp.asarray(b), jnp.asarray(a)], [jnp.asarray(rb), jnp.asarray(ra)], jnp.asarray(t), KERNEL))
    ok &= report("agent order", close(base, swapped, 1e-12))
    return ok & report("energy is positive here", base > 0.0)


def check_finite_differences():
    a, b, ra, rb, t = _random_pair(6, T=8)
    for kernel in (KERNEL, KERNEL._replace(alpha=3.0, beta=6.0)):
        energy = lambda x: spacetime_energy([x, jnp.asarray(b)], [jnp.asarray(ra), jnp.asarray(rb)], jnp.asarray(t), kernel)
        x0 = jnp.asarray(a)
        grad = np.asarray(jax.grad(energy)(x0))
        worst, h = 0.0, 1e-6
        for index in np.ndindex(a.shape):
            fd = (float(energy(x0.at[index].add(h))) - float(energy(x0.at[index].add(-h)))) / (2 * h)
            worst = max(worst, abs(fd - grad[index]) / max(1.0, abs(fd)))
        if not report(f"gradient against finite differences, alpha {kernel.alpha:g} (worst relative {worst:.1e})",
                      worst < 1e-6 and np.abs(grad).max() > 0):
            return False
    return True


def check_far_and_overlap():
    t = jnp.linspace(0.0, 2.0, 30)
    r = jnp.array([0.1])
    a = jnp.zeros((30, 1, 2))
    far = jnp.zeros((30, 1, 2)).at[..., 0].set(1.0)
    e_far, g_far = jax.value_and_grad(lambda x: spacetime_energy([x, far], [r, r], t, KERNEL))(a)
    ok = report("zero energy and gradient past the cutoff", float(e_far) == 0.0 and float(jnp.abs(g_far).max()) == 0.0)
    over = jnp.zeros((30, 1, 2)).at[..., 0].set(0.1)  # gap -0.1
    e, g = jax.value_and_grad(lambda x: spacetime_energy([x, over], [r, r], t, KERNEL))(a)
    # Moving a toward -x separates the pair, so the gradient along x must be positive.
    return ok & report(f"overlap is finite and pushes apart (E {float(e):.1f})",
                       np.isfinite(float(e)) and float(g[..., 0].sum()) > 0 and np.isfinite(np.asarray(g)).all())


def _crossing(delay, c=1.0, T=400):
    """Agent a crosses the origin along x at t = 2, agent b along y at t = 2 + delay."""
    t = jnp.linspace(0.0, 4.0, T)
    a = jnp.stack([t - 2.0, jnp.zeros(T)], -1)[:, None]
    b = jnp.stack([jnp.zeros(T), t - 2.0 - delay], -1)[:, None]
    r = jnp.array([0.1])
    k = KERNEL._replace(time_scale=c)
    same_time = float(repulsive_energy([a, b], [r, r], t, RepulsiveKernel(power=2.5, cutoff=0.25, relax=0.02)))
    return float(spacetime_energy([a, b], [r, r], t, k)), same_time


def check_cross_time_pairs():
    # Closest approach is D / sqrt(2) at equal times and c D / sqrt(1 + 2 c^2) over all pairs, so at c = 1
    # delays between 0.64 and 0.78 s are out of the same-time barrier's reach (0.45 m centre to centre) but not this one's.
    energies = [_crossing(d) for d in (0.65, 0.7, 0.75, 2.0)]
    ok = report("same-time barrier ignores crossings 0.65-0.75 s apart", all(e[1] == 0.0 for e in energies))
    ok &= report(f"space-time energy sees it and fades with the delay ({', '.join(f'{e:.2g}' for e, _ in energies)})",
                 energies[0][0] > energies[1][0] > energies[2][0] > 0.0 and energies[3][0] == 0.0)
    return ok & report("a larger c shrinks the time reach", _crossing(0.7, c=4.0)[0] == 0.0)


def _straight_pass():
    spec = [Block.R(2)]
    fk = lambda q: jnp.concatenate([q, jnp.zeros(1)])[None, :]
    radii = jnp.array([0.3])
    geometry = [AgentGeometry(spec=spec, fk_fn=fk, sphere_radii=radii) for _ in range(2)]
    n = 6
    x = jnp.linspace(-3.0, 3.0, n)
    moving = SplineParams(times=jnp.linspace(0.0, 6.0, n), wps=jnp.stack([x, jnp.full(n, 0.02)], 1),
                          vs=jnp.zeros((n, 2)).at[1:-1, 0].set(1.0))
    parked = SplineParams(times=jnp.linspace(0.0, 6.0, n), wps=jnp.zeros((n, 2)), vs=jnp.zeros((n, 2)))
    masks = [jnp.ones((n, 1), dtype=jnp.int32) for _ in range(2)]
    interior = jnp.ones(n).at[0].set(0.0).at[-1].set(0.0)
    zero = jnp.zeros(n)
    return geometry, [moving, parked], masks, [(interior, interior, interior), (zero, zero, zero)]


def check_parked_agent_repels():
    geometry, params, masks, free = _straight_pass()
    t_grid = jnp.linspace(0.0, 6.0, 150)
    k = KERNEL._replace(time_scale=1.0)
    weights = CostWeights(inter_agent=0.0, environment=0.0, regularizer=0.0, repulsive=4.0)
    cost = build_cost_fn(geometry, t_grid, lambda q: 1.0, weights, repulsive_kernel=k)
    q0 = [ccj.sample_configs(g, p, t_grid, m) for g, p, m in zip(geometry, params, masks)]
    _, bd = cost(params, q0, masks)
    centers = [jax.vmap(g.fk_fn)(q) for g, q in zip(geometry, q0)]
    direct = float(spacetime_energy(centers, [g.sphere_radii for g in geometry], t_grid, k))
    ok = report("the refiner's cost dispatches to the space-time energy", close(bd["repulsive"], direct, 1e-12))
    final, history, _ = ccj._run_line_search(cost, params, q0, masks, free, 60, 1e-3, 8, (1.0, 1.0, 0.25), max_step=0.1)

    def gap(p):
        q = ccj.sample_configs(geometry[0], p, t_grid, masks[0])
        return float(jnp.min(jnp.linalg.norm(q, axis=1))) - 0.6

    before, after = gap(params[0]), gap(final[0])
    ok &= report(f"moving agent clears a parked one ({before:+.3f} -> {after:+.3f} m)", before < 0.0 < after < 0.3)
    ok &= report("parked agent does not move", all(np.array_equal(x, y) for x, y in zip(final[1], params[1])))
    return ok & report("cost never rises", np.all(np.diff(np.asarray(history)) <= 1e-9))


def check_refiner():
    weights = CostWeights(inter_agent=0.0, environment=0.0, regularizer=0.0, repulsive=2.0)
    graph, splines, timing_mpc, x, x_dot = solved_two_agent_problem()
    config = refinement_config(repulsive_kernel=KERNEL, weights=weights, line_search=True, max_step=0.1)
    diag = ccj.refine_splines(timing_mpc, graph, splines, x, x_dot, config)
    ok = report(f"refiner lowers the space-time energy ({diag['initial']['repulsive']:.2f} -> {diag['final']['repulsive']:.2f})",
                0.0 <= diag["final"]["repulsive"] < diag["initial"]["repulsive"] and diag["accepted"])
    refused = 0
    for kernel in (KERNEL._replace(beta=1.0),):
        try:
            ccj.CollisionRefiner(graph, refinement_config(repulsive_kernel=kernel, weights=weights, line_search=True,
                                                          max_step=0.1))
        except ValueError:
            refused += 1
    return ok & report("an invalid space-time kernel is refused at build", refused == 1)


def main():
    ok = True
    for check in (check_radial_joins, check_bad_kernels_raise, check_against_brute_force, check_continuum_limit,
                  check_band, check_reference_below_relax, check_invariances, check_finite_differences, check_far_and_overlap,
                  check_cross_time_pairs, check_parked_agent_repels, check_refiner):
        ok &= check()
    print("\nALL OK" if ok else "\nFAILURES ABOVE")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

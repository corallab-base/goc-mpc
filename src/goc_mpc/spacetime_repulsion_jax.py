"""Space-time tangent-point repulsion between agents (Repulsive Curves, Yu, Schumacher and Crane 2021).

Each sphere centre of each agent traces a curve in space-time, gamma(t) = (p(t), c * t), with `c` the
time scale in m/s (Sec 8.5). Two agents repel through the tangent-point kernel of the paper,

    k(x, y, T) = |T x (x - y)|^alpha / |x - y|^beta,

summed over both orderings of each inter-agent point pair, so the energy is the inter-agent part of the
paper's double integral over the union of curves. Self-energy is left out: each curve is a graph over t,
so it cannot fold back on itself, and nothing here should resist an agent's own path.

Rewritten as `sin(theta)^alpha * |x - y|^-(beta - alpha)`, with theta the angle between T and x - y,
the kernel gets three changes so it fits spheres and a planner:

- the radial factor uses the sphere gap `g = |x - y| - (ra + rb)` instead of `|x - y|`, so it blows up at
  contact rather than at coincident centres;
- it is cut off smoothly at `cutoff` (window `(1 - (g / cutoff)^2)^2`, value and slope 0 there);
- below `relax` it continues as the quadratic with matching value, slope and curvature, so an overlapping
  start stays finite and still pushes apart.

The time lift is what the same-time barrier in `repulsive_energy_jax` lacks. Two agents at the same time
have x - y purely spatial, while a tangent (v, c) / |(v, c)| leans along time, so sin(theta) stays near 1
and the energy behaves like a barrier. Pairs at different times count too, through the c * dt part of
x - y, so an agent that will reach a spot shortly after another has left it still sees it. A parked agent
is a vertical line, which a moving agent's curve can pass at any spatial distance, so parked agents are
ordinary obstacles here. As c grows, the cross-time pairs shrink to a band of width ~cutoff / c and the
energy (divided by c, see below) tends to the same-time barrier integrated over the band.

Contact is space-time contact. For a parked partner the nearest lifted point is at equal time, so it is
ordinary contact. For two agents moving together at speed v the lifted curves are closer than their
equal-time distance by a factor c / sqrt(v^2 + c^2), so each body reads as stretched along its motion by
up to sqrt(1 + v^2 / c^2): a follower is asked for a time headway of about (ra + rb) / c on top of the
spatial gap. Choose c at or above the speed cap to keep that stretch below sqrt(2).

Discretization is the paper's Eq 17-18: curves are polylines through the samples on `t_grid`, each edge I
carries its unit tangent T_I and length l_I, and an edge pair scores
`(1/4) sum_{i in I, j in J} [k(x_i, y_j, T_I) + k(y_j, x_i, T_J)] l_I l_J` over the four endpoint pairs.
The quadrature resolves the kernel only while edges are short against the cutoff, so
`resolution(t_grid, kernel)` reports `c * max(dt) / cutoff`; keep it below about 0.5.

The total is divided by `c` (an edge length is about c * dt, and the band of contributing cross-time pairs
is about 1 / c wide, so the raw sum grows like c), and by the radial factor's force at `reference_gap`.

Cost is dense: every edge of one agent against every edge of the other. `band` (number of samples) limits
pairs to |I - J| <= band, which is exact when c * (t[k + band] - t[k]) >= cutoff + ra + rb for every k.
`band_is_exact` checks that for a given grid.
"""

from __future__ import annotations

import functools
from typing import NamedTuple, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np


class SpaceTimeKernel(NamedTuple):
    alpha: float = 2.0
    beta: float = 4.5
    time_scale: float = 1.0      # c, m/s
    cutoff: float = 0.25         # in sphere gap, m
    relax: float = 0.02          # quadratic below this gap, m
    reference_gap: float = 0.1   # radial force is 1 here
    band: Optional[int] = None   # None: all edge pairs


def check_kernel(kernel: SpaceTimeKernel) -> None:
    if kernel.alpha < 0.0:
        raise ValueError(f"alpha must be non-negative, got {kernel.alpha}")
    if not kernel.beta > kernel.alpha:
        raise ValueError(f"need beta > alpha so the kernel decays, got {kernel.alpha}, {kernel.beta}")
    if not kernel.time_scale > 0.0:
        raise ValueError(f"time_scale must be positive, got {kernel.time_scale}")
    if not 0.0 < kernel.relax < kernel.cutoff:
        raise ValueError(f"need 0 < relax < cutoff, got {kernel.relax}, {kernel.cutoff}")
    if not 0.0 < kernel.reference_gap < kernel.cutoff:
        raise ValueError(f"need 0 < reference_gap < cutoff, got {kernel.reference_gap}, {kernel.cutoff}")
    if kernel.band is not None and kernel.band < 0:
        raise ValueError(f"band must be None or >= 0, got {kernel.band}")


def _windowed_power(g, p, cutoff):
    w = 1.0 - (g / cutoff) ** 2
    return g ** -p * w * w


@functools.lru_cache(maxsize=None)
def _radial_constants(p: float, cutoff: float, relax: float, reference_gap: float):
    """Value, slope and curvature of the windowed power at `relax`, and the normalizing force (minus the
    slope of the piecewise factor at `reference_gap`, on whichever branch it falls), as floats. Cached, so
    tracing does not redo the derivatives on every call."""
    f = lambda g: _windowed_power(g, p, cutoff)
    d1, d2 = jax.grad(f), jax.grad(jax.grad(f))
    with jax.ensure_compile_time_eval():
        h0, h1, h2 = (float(fn(jnp.asarray(relax, jnp.float64))) for fn in (f, d1, d2))
        if reference_gap >= relax:
            force = -float(d1(jnp.asarray(reference_gap, jnp.float64)))
        else:
            force = -(h1 + h2 * (reference_gap - relax))
    return h0, h1, h2, force


def radial(gap: jnp.ndarray, kernel: SpaceTimeKernel) -> jnp.ndarray:
    """Radial factor of the kernel at sphere gap `gap`, normalized to force 1 at `reference_gap`."""
    p, cutoff, relax = kernel.beta - kernel.alpha, kernel.cutoff, kernel.relax
    h0, h1, h2, force = _radial_constants(float(p), float(cutoff), float(relax), float(kernel.reference_gap))
    # Clamped first, or the unselected power branch would put NaN into the gradient through jnp.where.
    inside = jnp.clip(gap, relax, cutoff)
    power = _windowed_power(inside, p, cutoff)
    off = gap - relax
    quadratic = h0 + h1 * off + 0.5 * h2 * off * off
    value = jnp.where(gap >= cutoff, 0.0, jnp.where(gap > relax, power, quadratic))
    return value / force


def lift(centers: jnp.ndarray, t_grid: jnp.ndarray, time_scale: float) -> jnp.ndarray:
    """(T, s, dim) sphere centres to (T, s, dim + 1) space-time points (p, c t)."""
    tt = jnp.broadcast_to((time_scale * t_grid)[:, None, None], centers.shape[:2] + (1,))
    return jnp.concatenate([centers, tt.astype(centers.dtype)], axis=-1)


def _edges(points: jnp.ndarray, eps: float):
    e = points[1:] - points[:-1]
    length = jnp.sqrt(jnp.sum(e * e, axis=-1) + eps)
    return points[:-1], points[1:], e / length[..., None], length


def _sin_power(d, tangent, alpha, eps):
    """sin(theta)^alpha between `tangent` (unit) and `d`, broadcast over leading axes."""
    dd = jnp.sum(d * d, axis=-1) + eps
    along = jnp.sum(d * tangent, axis=-1)
    s2 = jnp.clip(1.0 - along * along / dd, 0.0, 1.0)
    if alpha == 2.0:
        return s2
    return (s2 + eps) ** (0.5 * alpha)


def _pair_terms(xa, ta, la, xb, tb, lb, rho, kernel, eps):
    """Sum over the four endpoint pairs of edges (I in a, J in b). Inputs broadcast against each other:
    `xa = (start, end)` points, `ta` tangents, `la` lengths, likewise for b; `rho` the radius sum."""
    total = 0.0
    for pa in xa:
        for pb in xb:
            d = pa - pb
            dist = jnp.sqrt(jnp.sum(d * d, axis=-1) + eps)
            r = radial(dist - rho, kernel)
            total = total + (_sin_power(d, ta, kernel.alpha, eps) + _sin_power(d, tb, kernel.alpha, eps)) * r
    return 0.25 * total * la * lb


def pair_energy(points_a, radii_a, points_b, radii_b, kernel: SpaceTimeKernel, eps: float = 1e-12):
    """Energy between two agents' lifted sphere curves, `(T, s, dim + 1)` each, before the 1 / c scaling."""
    a0, a1, ta, la = _edges(points_a, eps)        # (E, sa, D), (E, sa)
    b0, b1, tb, lb = _edges(points_b, eps)
    if kernel.band is None:
        # (E_a, sa, 1, 1, D) against (1, 1, E_b, sb, D)
        A = lambda x: x[:, :, None, None]
        B = lambda x: x[None, None]
        rho = radii_a[None, :, None, None] + radii_b[None, None, None, :]
        terms = _pair_terms((A(a0), A(a1)), A(ta), A(la), (B(b0), B(b1)), B(tb), B(lb), rho, kernel, eps)
        return jnp.sum(terms)
    # Banded: edge I of a against edges J = I + m of b, |m| <= band, gathered in one shot as
    # (E, W, sa, sb) so the graph does not grow with the band. Out-of-range J are clipped to a real
    # edge (finite values) and zeroed by `valid`.
    E = a0.shape[0]
    offsets = jnp.arange(-kernel.band, kernel.band + 1)
    j = jnp.arange(E)[:, None] + offsets[None, :]                          # (E, W)
    valid = ((j >= 0) & (j < E)).astype(a0.dtype)[:, :, None, None]
    j = jnp.clip(j, 0, E - 1)
    A = lambda x: x[:, None, :, None]                                      # (E, 1, sa, 1, ...)
    B = lambda x: x[j][:, :, None, :]                                      # (E, W, 1, sb, ...)
    rho = radii_a[None, None, :, None] + radii_b[None, None, None, :]
    terms = _pair_terms((A(a0), A(a1)), A(ta), A(la), (B(b0), B(b1)), B(tb), B(lb), rho, kernel, eps)
    return jnp.sum(terms * valid)


def spacetime_energy(centers_list: Sequence[jnp.ndarray], radii_list: Sequence[jnp.ndarray],
                     t_grid: jnp.ndarray, kernel: SpaceTimeKernel, eps: float = 1e-12) -> jnp.ndarray:
    """Inter-agent space-time tangent-point energy. `centers_list[i]` is agent i's `(T, s_i, dim)` sphere
    centres on `t_grid`, `radii_list[i]` its `(s_i,)` radii, as in `inter_agent_cost`."""
    check_kernel(kernel)
    lifted = [lift(c, t_grid, kernel.time_scale) for c in centers_list]
    total = jnp.asarray(0.0, dtype=lifted[0].dtype)
    for a in range(len(lifted)):
        for b in range(a + 1, len(lifted)):
            total = total + pair_energy(lifted[a], radii_list[a], lifted[b], radii_list[b], kernel, eps)
    return total / kernel.time_scale


def resolution(t_grid, kernel: SpaceTimeKernel) -> float:
    """Longest space-time edge from time alone, over the cutoff. Above about 0.5 the edge quadrature
    no longer resolves the kernel."""
    return float(kernel.time_scale * np.max(np.diff(np.asarray(t_grid))) / kernel.cutoff)


def band_is_exact(t_grid, kernel: SpaceTimeKernel, max_radius_sum: float) -> bool:
    """Whether `kernel.band` drops only edge pairs that are past the cutoff for any motion: their time
    gap alone puts them beyond cutoff + radius sum."""
    if kernel.band is None:
        return True
    t = np.asarray(t_grid)
    k = kernel.band
    if k + 1 >= len(t) - 1:
        return True
    # Edges I and I + k + 1 (I + k + 1 <= n - 2), the nearest dropped pair, are at least t[I + k + 1] - t[I + 1]
    # apart in time.
    gap_t = t[k + 1:len(t) - 1] - t[1:len(t) - k - 1]
    return bool(kernel.time_scale * gap_t.min() >= kernel.cutoff + max_radius_sum)

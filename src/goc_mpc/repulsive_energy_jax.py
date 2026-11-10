"""Repulsive energy between agents, for `collision_cost_jax`'s refiner.

Starts from the same-time limit of the space-time energy in Repulsive Curves
(Yu, Schumacher, Crane, 2021, Sec 8.5): as the time scale grows, only sample
pairs at equal time survive, so the double integral becomes a single sum of
a barrier on each sphere pair's gap over the shared time grid. There is no
tangent factor in this limit, and no pair of different times, so this tests
the barrier's shape on its own. Finite time scales would add the cross-time
pairs on top. That holds while the time scale times the sample spacing stays
above the pair's reach. On a finer grid the cross-time pairs dominate and act
like an integral of the barrier over time, so a power and weight tuned here
will not carry over.

`barrier` has three branches in the gap `g = |ca - cb| - (ra + rb)`:

- `g >= cutoff`: zero, so distant pairs cost nothing.
- `relax <= g < cutoff`: `g^-p`, shifted so value and slope reach zero at the
  cutoff.
- `g < relax`: the quadratic with matching value, slope and curvature at
  `relax`, which stays finite through contact and overlap. An overlapping
  start has no well-defined side of a true barrier.

The result is divided by the barrier's force at `reference_gap`, so on a
uniform grid a term weight of `w` means a force of `w` per sample at that gap.
That makes weights comparable with the hinge, whose force at gap `g` is
`2 * w * (margin - g)`. On a nonuniform grid the barrier integrates over time
while the hinge still counts samples, so their weights no longer compare.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp


class RepulsiveKernel(NamedTuple):
    power: float = 2.5
    cutoff: float = 0.25
    relax: float = 0.02
    reference_gap: float = 0.1


def _check(kernel: RepulsiveKernel) -> None:
    if kernel.power <= 0.0:
        raise ValueError(f"power must be positive, got {kernel.power}")
    if not 0.0 < kernel.relax < kernel.cutoff:
        raise ValueError(f"need 0 < relax < cutoff, got {kernel.relax} and {kernel.cutoff}")
    if not 0.0 < kernel.reference_gap < kernel.cutoff:
        raise ValueError(f"need 0 < reference_gap < cutoff, got {kernel.reference_gap} and {kernel.cutoff}")


def _raw_force(gap: float, kernel: RepulsiveKernel) -> float:
    """Minus the slope of the shifted power branch at `gap`."""
    p, cutoff = kernel.power, kernel.cutoff
    return p * gap ** -(p + 1.0) - p * cutoff ** -(p + 1.0)


def _force(gap: float, kernel: RepulsiveKernel) -> float:
    """Minus the slope of the barrier before normalization, on either branch."""
    if gap >= kernel.relax:
        return _raw_force(gap, kernel)
    p, relax = kernel.power, kernel.relax
    return _raw_force(relax, kernel) + p * (p + 1.0) * relax ** -(p + 2.0) * (relax - gap)


def _shifted_power(gap, kernel: RepulsiveKernel):
    p, cutoff = kernel.power, kernel.cutoff
    return gap ** -p - cutoff ** -p + p * cutoff ** -(p + 1.0) * (gap - cutoff)


def barrier(gap: jnp.ndarray, kernel: RepulsiveKernel) -> jnp.ndarray:
    """Normalized barrier value at each `gap`, in metres, negative in overlap."""
    _check(kernel)
    relax = kernel.relax
    # Clamped first, or the unselected power branch at a gap <= 0 would turn
    # into a NaN gradient through jnp.where.
    inside = jnp.clip(gap, relax, kernel.cutoff)
    power = _shifted_power(inside, kernel)
    offset = gap - relax
    p = kernel.power
    quadratic = (_shifted_power(relax, kernel) - _raw_force(relax, kernel) * offset
                 + 0.5 * p * (p + 1.0) * relax ** -(p + 2.0) * offset * offset)
    # `>` rather than `>=`: jnp.clip halves its derivative at an exact tie.
    value = jnp.where(gap >= kernel.cutoff, 0.0, jnp.where(gap > relax, power, quadratic))
    return value / _force(kernel.reference_gap, kernel)


def sample_weights(t_grid: jnp.ndarray) -> jnp.ndarray:
    """Trapezoid weights for `t_grid`, scaled to average 1, so a uniform grid
    scores like a plain sum and a nonuniform one integrates over time."""
    spacing = jnp.diff(t_grid)
    weights = 0.5 * (jnp.concatenate([spacing, jnp.zeros(1)]) + jnp.concatenate([jnp.zeros(1), spacing]))
    return weights / jnp.mean(weights)


def repulsive_energy(centers_list: Sequence[jnp.ndarray], radii_list: Sequence[jnp.ndarray],
                     t_grid: jnp.ndarray, kernel: RepulsiveKernel, eps: float = 1e-9) -> jnp.ndarray:
    """Same-time energy over every sphere pair of every agent pair, weighted
    by `sample_weights`. `centers_list[i]` is `(T, s_i, dim)`, `radii_list[i]`
    `(s_i,)`, as in `inter_agent_cost`. Parked agents count in full: an agent
    that has finished is still something to stay clear of."""
    weights = sample_weights(t_grid)
    total = jnp.asarray(0.0, dtype=weights.dtype)
    n = len(centers_list)
    for a in range(n):
        for b in range(a + 1, n):
            ca, cb = centers_list[a], centers_list[b]
            ra, rb = radii_list[a], radii_list[b]
            diff = ca[:, :, None, :] - cb[:, None, :, :]  # (T, s_a, s_b, dim)
            dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1) + eps)
            gap = dist - (ra[None, :, None] + rb[None, None, :])
            total = total + jnp.sum(weights * jnp.sum(barrier(gap, kernel), axis=(1, 2)))
    return total

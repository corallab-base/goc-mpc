"""Ground-truth trajectory metrics for comparing collision refinement variants.

Plain numpy, and deliberately independent of collision_cost_jax's cost: a
variant must not be able to score well just because it minimized the same
sampled quantity it is measured by. In particular `inter_agent_cost` only sees
spheres at its grid samples, so two agents can pass through each other between
samples; `swept_pair_clearance` checks the whole interval instead.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def swept_pair_clearance(centers_a: np.ndarray, centers_b: np.ndarray,
                         radii_a: np.ndarray, radii_b: np.ndarray) -> np.ndarray:
    """Minimum signed clearance between two agents over each sample interval,
    `(T - 1,)`, from `(T, s, dim)` centers and `(s,)` radii. Negative overlaps.

    Each sphere is taken to move linearly between samples, so the closest
    approach of a sphere pair is the closed-form minimum of |d0 + s * e| over
    s in [0, 1], with d0 their offset at the interval's start and e its change
    across the interval."""
    d = centers_a[:, :, None, :] - centers_b[:, None, :, :]  # (T, s_a, s_b, dim)
    d0, e = d[:-1], d[1:] - d[:-1]
    ee = np.sum(e * e, axis=-1)
    s = -np.sum(d0 * e, axis=-1) / np.where(ee > 0.0, ee, 1.0)
    s = np.clip(np.where(ee > 0.0, s, 0.0), 0.0, 1.0)
    dist = np.linalg.norm(d0 + s[..., None] * e, axis=-1)
    return np.min(dist - (radii_a[:, None] + radii_b[None, :]), axis=(1, 2))


def inter_agent_metrics(centers_list: Sequence[np.ndarray], radii_list: Sequence[np.ndarray],
                        t_grid: np.ndarray, margins: Sequence[float] = (0.0, 0.1, 0.2)) -> dict:
    """Worst swept clearance over all agent pairs, and how long pairs spend
    below each margin, in pair-seconds summed over pairs."""
    dt = np.diff(t_grid)
    below = {m: 0.0 for m in margins}
    worst, worst_pair = np.inf, None
    n = len(centers_list)
    for a in range(n):
        for b in range(a + 1, n):
            sep = swept_pair_clearance(centers_list[a], centers_list[b], radii_list[a], radii_list[b])
            if sep.min() < worst:
                worst, worst_pair = float(sep.min()), (a, b)
            for m in margins:
                below[m] += float(np.sum(dt[sep < m]))
    return {"min_swept_clearance_m": worst, "worst_pair": worst_pair,
            "pair_seconds_below": {str(m): v for m, v in below.items()}}


def space_time_clearance(centers_list: Sequence[np.ndarray], radii_list: Sequence[np.ndarray],
                         t_grid: np.ndarray, c: float) -> float:
    """Minimum over agent pairs and over ALL pairs of times of
    sqrt(|p_a(i) - p_b(j)|^2 + c^2 (t_i - t_j)^2) - (r_a + r_b).

    `c` (m/s) is the exchange rate between lateral and along-track error: a
    plan with clearance D keeps agents apart under any combined error with
    sqrt(|e_x|^2 + c^2 e_t^2) < D. O(T^2), so pass a decimated grid."""
    worst = np.inf
    n = len(centers_list)
    for a in range(n):
        for b in range(a + 1, n):
            r = radii_list[a][:, None] + radii_list[b][None, :]
            for i, t_i in enumerate(t_grid):
                d = centers_list[a][i][None, :, None, :] - centers_list[b][:, None, :, :]
                st = np.sum(d * d, axis=-1) + (c * (t_i - t_grid))[:, None, None] ** 2
                worst = min(worst, float(np.min(np.sqrt(st) - r)))
    return worst


def path_deviation(points: np.ndarray, reference: np.ndarray) -> float:
    """Largest distance from any of `points` (P, dim) to the polyline
    `reference` (R, dim), ignoring time: how far a path moved geometrically,
    which retiming alone would leave at zero."""
    a, e = reference[:-1], reference[1:] - reference[:-1]
    ee = np.sum(e * e, axis=-1)
    d = points[:, None, :] - a[None, :, :]
    s = np.sum(d * e[None, :, :], axis=-1) / np.where(ee > 0.0, ee, 1.0)
    s = np.clip(np.where(ee > 0.0, s, 0.0), 0.0, 1.0)
    return float(np.max(np.min(np.linalg.norm(d - s[..., None] * e[None, :, :], axis=-1), axis=1)))

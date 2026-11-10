"""Sobolev-Slobodeckij metric for the P2 refiner (Repulsive Curves, Yu, Schumacher and Crane 2021, Sec 4-5).

The paper's flow descends the tangent-point energy in the fractional Sobolev inner product of the energy's
own order, s = (beta - 1) / alpha, rather than in L2. For a polyline with vertices x_i, edges I of length
l_I, midpoints m_I and hat-function gradients g_{I,i} (vector, +-T_I / l_I), the discrete Gram matrix is
(Eq 21, matching `SobolevCurves::SobolevGramMatrix` in the authors' code):

    B(u, v)  = sum_{I, J not touching} l_I l_J |m_I - m_J|^-(2 sigma + 1) (D u_I - D u_J) . (D v_I - D v_J)
    B0(u, v) = sum_{I, J not touching} l_I l_J k(m_I, m_J) (u(m_I) - u(m_J)) (v(m_I) - v(m_J))
    A        = B + B0 + eps * diag(dual lengths)

with sigma = s - 1, D u_I = sum_i u_i g_{I,i} (a vector, so B also sees the turn between edges), u(m_I) the
edge-midpoint average, and k the symmetrized tangent-point kernel with exponents (2, 4 + 2 sigma + 1). A is
scalar per vertex pair and applies to each coordinate.

Here the curve is each agent's space-time path, the centroid of its sphere centres lifted to (p, c t) on the
refiner's time grid. The refiner's variables are knot times, positions and velocities, not polyline
vertices, so A is pulled back through the spline: with J = d p(t_k) / d x (the time coordinate is fixed on
the grid), G_s = sum_d J_d^T A J_d. Pinned entries are removed, and the metric used is

    G = U + strength * (tr U / tr G_s) * G_s,

U the unit metric (1 / metric_scale on free entries). Some knot variables barely move the sampled curve
(velocities and times on short or slow pieces), so G_s is nearly singular there. Blending U in with a small
weight made those directions far cheaper than under U, and the step went into them; with U as the base
they keep their unit cost, while the directions the curve does see are smoothed as in the paper.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class SobolevMetric(NamedTuple):
    alpha: float = 2.0
    beta: float = 4.5        # s = (beta - 1) / alpha, sigma = s - 1
    time_scale: float = 1.0  # c, m/s, of the space-time lift
    eps: float = 1e-2        # mass term, relative to the mean diagonal of B + B0
    strength: float = 10.0   # weight of the trace-normalized Sobolev part on top of the unit metric


def check_metric(m: SobolevMetric) -> None:
    s = (m.beta - 1.0) / m.alpha if m.alpha > 0 else float("nan")
    if not (m.alpha > 0 and 1.0 < s < 2.5):
        raise ValueError(f"need alpha > 0 and 1 < s = (beta - 1) / alpha < 2.5, got {m.alpha}, {m.beta}")
    if not m.time_scale > 0:
        raise ValueError(f"time_scale must be positive, got {m.time_scale}")
    if not (m.eps > 0 and m.strength > 0):
        raise ValueError(f"need eps > 0 and strength > 0, got {m.eps}, {m.strength}")


def sigma(m: SobolevMetric) -> float:
    return (m.beta - 1.0) / m.alpha - 1.0


def _kernel_sym(d, t1, t2, a, b, eps):
    """(|t1 x d|^a + |t2 x d|^a) / (2 |d|^b), via |t x d|^2 = |d|^2 - (t . d)^2 for unit t."""
    dd = jnp.sum(d * d, axis=-1) + eps
    c1 = jnp.clip(dd - jnp.sum(d * t1, axis=-1) ** 2, 0.0)
    c2 = jnp.clip(dd - jnp.sum(d * t2, axis=-1) ** 2, 0.0)
    return 0.5 * (c1 ** (0.5 * a) + c2 ** (0.5 * a)) / dd ** (0.5 * b)


def gram(points: jnp.ndarray, m: SobolevMetric, eps: float = 1e-12) -> jnp.ndarray:
    """(T, T) Gram matrix A of the polyline through `points` (T, D)."""
    T = points.shape[0]
    e = points[1:] - points[:-1]
    ell = jnp.sqrt(jnp.sum(e * e, axis=-1) + eps)
    tan = e / ell[:, None]
    mid = 0.5 * (points[1:] + points[:-1])
    E = T - 1
    I, J = jnp.arange(E)[:, None], jnp.arange(E)[None, :]
    far = (jnp.abs(I - J) > 1).astype(points.dtype)               # not the same edge, not sharing a vertex
    d = mid[:, None, :] - mid[None, :, :]
    r2 = jnp.sum(d * d, axis=-1) + eps
    p = 2.0 * sigma(m) + 1.0
    lw = ell[:, None] * ell[None, :] * far
    W = lw * r2 ** (-0.5 * p)
    W0 = lw * _kernel_sym(d, tan[:, None, :], tan[None, :, :], 2.0, 4.0 + p, eps)

    # Hat gradients: vertex I gets -T/l, vertex I + 1 gets +T/l, so D u_I = (u_{I+1} - u_I) T_I / l_I.
    g = tan / ell[:, None]                                       # (E, D)
    rows = jnp.arange(E)
    hat = jnp.zeros((E, T), points.dtype).at[rows, rows].set(-1.0).at[rows, rows + 1].set(1.0)
    lap = lambda w: jnp.diag(jnp.sum(w, axis=1)) - w             # sum_{IJ} w (a_I - a_J)^2 = 2 a^T lap a
    # D u_I . D u_J = (hat u)_I (hat u)_J (g_I . g_J), so B = 2 hat^T (diag(W 1) |g|^2 - W (g g^T)) hat.
    gg = g @ g.T
    B = 2.0 * hat.T @ (jnp.diag(jnp.sum(W, axis=1) * jnp.diag(gg)) - W * gg) @ hat
    avg = jnp.abs(hat) * 0.5
    B0 = 2.0 * avg.T @ lap(W0) @ avg
    dual = jnp.zeros(T, points.dtype).at[:-1].add(0.5 * ell).at[1:].add(0.5 * ell)
    S = B + B0
    mass = m.eps * jnp.mean(jnp.diag(S)) / jnp.mean(dual)
    return 0.5 * (S + S.T) + jnp.diag(mass * dual)


def pulled_back(points_fn, x: jnp.ndarray, t_grid: jnp.ndarray, unit_diag: jnp.ndarray,
                m: SobolevMetric) -> jnp.ndarray:
    """Cholesky factor of the metric on one agent's flat variables `x`. `points_fn(x)` gives the (T, D)
    spatial curve on `t_grid`; `unit_diag` is the unit metric's inverse diagonal (metric_scale times the
    free mask, 0 on pinned entries). Pinned rows and columns become identity, and only free entries get
    a step: solve with the factor, then multiply by the free mask."""
    p = points_fn(x)
    lifted = jnp.concatenate([p, (m.time_scale * t_grid)[:, None]], axis=1)
    A = gram(lifted, m)
    Jac = jax.jacfwd(points_fn)(x)                               # (T, D, n)
    G = jnp.einsum("tdn,ts,sdk->nk", Jac, A, Jac)
    free = (unit_diag > 0).astype(x.dtype)
    unit = jnp.where(unit_diag > 0, 1.0 / jnp.where(unit_diag > 0, unit_diag, 1.0), 0.0)
    G = G * free[:, None] * free[None, :]
    scale = jnp.sum(unit) / jnp.maximum(jnp.trace(G), 1e-300)
    G = m.strength * scale * G + jnp.diag(unit) + jnp.diag(1.0 - free)
    return jnp.linalg.cholesky(0.5 * (G + G.T))


def apply_inverse(chol: jnp.ndarray, free: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """G^-1 v on the free entries (0 on pinned ones), for v of shape (n,) or (n, k)."""
    f = free if v.ndim == 1 else free[:, None]
    return jax.scipy.linalg.cho_solve((chol, True), v * f) * f

"""JAX reimplementation of CubicConfigurationSpline::eval (configuration_
spline.hpp), differentiable w.r.t. (wps, vs, times) via ordinary jax
autodiff rather than a hand-derived analytic Jacobian -- see
PLAN_collision_aware_timing.md's "Language decision" section for why: the
step-2 refinement pass's other two pieces (per-robot sphereized FK, the
environment speed field) are JAX-only already, so this is the piece that
has to speak JAX for the whole forward+backward pass to be one jit+vmap'd
function.

Takes `spec` as the same `list[goc_mpc.splines.Block]` object
CubicConfigurationSpline itself is constructed from -- there is exactly one
Block/BlockType representation in this codebase, not a second parallel one
(matches default_fk.py's own convention of importing BlockType directly
rather than redefining it).

R and Torus blocks only, matching CubicConfigurationSpline::eval's own
power-basis math for those two types exactly, including per-block knot
bridging (`block_active_mask`). SO3Quat/SO3Mat raise NotImplementedError --
the same scope limit PLAN_collision_aware_timing.md states for step 1
(every current b1_multiroom agent is [Block.R(2), Block.Torus(1)]; SO3
routes through so3::left_jacobian/Exp and is real extra work, deferred
until an SO3-carrying agent actually needs this) -- rather than silently
producing a wrong answer for a block type this doesn't handle.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Sequence

import jax
import jax.numpy as jnp

from ._ext.configuration_spline import BlockType

_SUPPORTED_TYPES = (BlockType.R, BlockType.Torus)


class BlockLayout(NamedTuple):
    type: BlockType
    offset: int  # ambient == tangent offset (R/Torus only)
    size: int    # ambient == tangent size (R/Torus only)


def block_layout(spec: Sequence) -> tuple[BlockLayout, ...]:
    """Static per-block offsets for `spec`, mirroring
    CubicConfigurationSpline::set_spec's BlockOffset bookkeeping for the
    R/Torus case (ambient_size == tangent_size, so one offset serves both).
    Raises for SO3Quat/SO3Mat -- see module docstring."""
    layout = []
    off = 0
    for b in spec:
        if b.type not in _SUPPORTED_TYPES:
            raise NotImplementedError(
                f"spline_jax: block type {b.type} not supported yet (only "
                "R and Torus -- see module docstring)")
        layout.append(BlockLayout(b.type, off, b.size))
        off += b.size
    return tuple(layout)


def ambient_dim(spec: Sequence) -> int:
    return sum(b.size for b in spec)


tangent_dim = ambient_dim  # R/Torus only: ambient == tangent, same sum


def wrap_pi(a):
    """torus::wrap_pi (configuration_spline.hpp) ported to jax: wraps to
    (-pi, pi], forcing the exact multiple-of-2pi boundary to +pi (matching
    the C++ side's `r <= 0` correction, which prefers +pi over -pi there)."""
    two_pi = 2.0 * jnp.pi
    r = jnp.mod(a + jnp.pi, two_pi)
    r = jnp.where(r == 0.0, two_pi, r)
    return r - jnp.pi


def _active_mask_column(active_mask, bi: int, n: int):
    """Knot 0 and knot n-1 are always active regardless of the mask --
    matches CubicConfigurationSpline::set()'s `is_active` computation."""
    if active_mask is None:
        return jnp.ones(n, dtype=bool)
    col = active_mask[:, bi].astype(bool)
    return col.at[0].set(True).at[-1].set(True)


def _piece_bounds(active):
    """For one block's per-knot active flags (length N, endpoints already
    forced True), returns (lo, hi): length-(N-1) arrays where `lo[k]`/
    `hi[k]` are the start/end knot indices of the Hermite piece this block
    bridges base knot-interval k with. Mirrors
    CubicConfigurationSpline::set()'s per-block "consecutive active knots"
    piece construction and eval()'s walk-back-to-the-last-filled-slot loop
    -- as two vectorized cummax/cummin scans (jittable, fixed shape for
    fixed N) instead of a per-query Python while loop."""
    n = active.shape[0]
    idx = jnp.arange(n)
    lo_full = jax.lax.cummax(jnp.where(active, idx, -1))
    hi_src = jnp.where(active, idx, n)
    hi_full = jax.lax.cummin(hi_src[::-1])[::-1]
    lo = lo_full[: n - 1]
    hi = jnp.take(hi_full, lo + 1)
    return lo, hi


def eval(spec: Sequence, times, wps, vs, t, active_mask: Optional[jnp.ndarray] = None):
    """JAX counterpart to CubicConfigurationSpline::eval(t), batched over a
    query-time array `t` (shape (T,)) instead of one scalar per call, and
    differentiable w.r.t. wps/vs/times via ordinary jax autodiff (no
    analytic Jacobian needed -- see module docstring).

    `times` (N,), `wps` (N, ambient_dim(spec)), `vs` (N, tangent_dim(spec))
    are exactly the arrays CubicConfigurationSpline::set(pts, vels, times)
    takes -- `times` strictly increasing, N >= 2. `active_mask`, if given,
    is the same (N, len(spec)) 0/1 array `set_block_active_mask` takes
    (pass an explicit all-ones array rather than None under `jax.jit`,
    since None isn't a valid traced argument). Query times outside
    [times[0], times[-1]] are clamped, matching eval()'s own clamping.

    Returns (q, v, a) of shape (T, ambient_dim(spec)) / (T, tangent_dim(spec))
    / (T, tangent_dim(spec)) -- ambient velocity/acceleration, i.e. the
    tangent-space rates CubicConfigurationSpline::eval returns as
    v_tangent/a_tangent.
    """
    layout = block_layout(spec)
    n = times.shape[0]
    t_c = jnp.clip(t, times[0], times[-1])
    k = jnp.clip(jnp.searchsorted(times, t_c, side="right") - 1, 0, n - 2)

    q_parts, v_parts, a_parts = [], [], []
    for bi, blk in enumerate(layout):
        active = _active_mask_column(active_mask, bi, n)
        lo, hi = _piece_bounds(active)
        s = jnp.take(lo, k)
        e = jnp.take(hi, k)

        t_s = jnp.take(times, s)
        tau = jnp.take(times, e) - t_s
        tt = jnp.clip(t_c - t_s, 0.0, tau)[:, None]
        tau_col = tau[:, None]

        off, sz = blk.offset, blk.size
        x0 = jnp.take(wps, s, axis=0)[:, off:off + sz]
        x1 = jnp.take(wps, e, axis=0)[:, off:off + sz]
        v0 = jnp.take(vs, s, axis=0)[:, off:off + sz]
        v1 = jnp.take(vs, e, axis=0)[:, off:off + sz]

        if blk.type == BlockType.R:
            disp = x1 - x0
            d0 = x0
        else:  # Torus
            disp = wrap_pi(x1 - x0)
            d0 = jnp.zeros_like(x0)

        c0 = v0
        b0 = (3.0 * disp - tau_col * (2.0 * v0 + v1)) / tau_col ** 2
        a0 = (-2.0 * disp + tau_col * (v0 + v1)) / tau_col ** 3

        q_blk = d0 + tt * (c0 + tt * (b0 + tt * a0))
        v_blk = c0 + tt * (2.0 * b0 + tt * 3.0 * a0)
        a_blk = 2.0 * b0 + tt * 6.0 * a0
        if blk.type == BlockType.Torus:
            q_blk = wrap_pi(x0 + q_blk)

        q_parts.append(q_blk)
        v_parts.append(v_blk)
        a_parts.append(a_blk)

    return (jnp.concatenate(q_parts, axis=-1),
            jnp.concatenate(v_parts, axis=-1),
            jnp.concatenate(a_parts, axis=-1))

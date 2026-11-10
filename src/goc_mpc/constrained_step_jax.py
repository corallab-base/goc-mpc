"""Constrained projected descent for the P2 refiner (Repulsive Curves, Yu, Schumacher and Crane 2021, Sec 5.3).

Walls become hard rows instead of a penalty. Per agent and sample time t_k on the refiner's grid,

    h_k(x) = clearance_fn(q(t_k; x)) - margin >= 0.

Each iteration:

- Direction (Eq 22). With D the diagonal metric (`metric_scale` times the free masks, so pins stay
  pinned) and C the Jacobian of the near rows (h < band), take the step of

      min g^T d + 1/2 d^T D^-1 d   subject to   C d >= 0,

  d = -D (g - C^T mu), with mu >= 0 the solution of the dual, a non-negative least squares problem in
  S = C D C^T solved by a primal-dual active-set iteration. Rows the free step leaves anyway get mu = 0.
- Back-projection (Eq 23, as inequalities). After the trial step x + alpha d, the rows that ended up
  violated are lifted by the smallest step in the D^-1 norm with C delta >= restore_target - h, the same
  dual. Treating these rows as inequalities matters: as equalities, two nearly parallel rows with
  different targets ask for metres of nearly cancelling motion.
- Acceptance, per agent. An agent keeps its back-projected trial only if that does not raise its grid
  violation (sum and max of relu(-h)) over the plain step, and keeps either one only if it does not raise
  its violation over where it was; otherwise that agent stays put this iteration. From a feasible iterate
  this is the paper's "no new collision" bound on the grid. The joint result is then kept if the energy
  passes Armijo on the agents that moved, or if total violation fell. Otherwise alpha halves, as in
  `_run_line_search`. So an agent stuck inside the margin cannot hold back the others.

Rows are chosen per agent so a parked agent near a wall cannot take the others' slots, over windows of
`window` samples (the closest sample of each window) so neighbouring rows do not fill the budget, and
only among samples that can move: not the pinned start, and not stretches between two pinned rows or of
a stationary piece (the goal hold and the pads). The step is first order, and a wall reached between
samples is not seen, as with the hinge.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Sequence

import jax
import jax.numpy as jnp


class WallConstraint(NamedTuple):
    margin: float = 0.0            # h = clearance - margin >= 0
    band: float = 0.05             # rows with h < band are held in the direction solve
    rows_per_agent: int = 24
    window: int = 5                # samples per candidate window
    restore_iters: int = 2         # back-projection steps per trial
    restore_target: float = 1e-3   # violated rows are lifted to this h
    initial_restore_iters: int = 4
    dual_iters: int = 8            # primal-dual active-set iterations on the dual
    reg: float = 1e-9              # relative Tikhonov term on C D C^T
    tol: float = 1e-9              # slack in the no-new-penetration test


def check_constraint(c: WallConstraint) -> None:
    if not c.band >= 0.0:
        raise ValueError(f"band must be >= 0, got {c.band}")
    if c.rows_per_agent < 1 or c.window < 1:
        raise ValueError(f"rows_per_agent and window must be >= 1, got {c.rows_per_agent}, {c.window}")
    if c.restore_iters < 0 or c.initial_restore_iters < 0 or c.dual_iters < 1:
        raise ValueError("restore iteration counts must be >= 0 and dual_iters >= 1")
    if not c.restore_target >= 0.0:
        raise ValueError(f"restore_target must be >= 0, got {c.restore_target}")


def _flat(p) -> jnp.ndarray:
    return jnp.concatenate([p.times, p.wps.reshape(-1), p.vs.reshape(-1)])


def _unflat(x: jnp.ndarray, like):
    n, nw = like.times.shape[0], like.wps.size
    return like._replace(times=x[:n], wps=x[n:n + nw].reshape(like.wps.shape),
                         vs=x[n + nw:].reshape(like.vs.shape))


def metric_diagonal(p, free_mask, metric_scale) -> jnp.ndarray:
    """Flattened D for one agent: the scale of each group times its free mask (0 on pinned entries)."""
    wps_free, vs_free, times_free = free_mask
    s_w, s_v, s_t = metric_scale
    dt = s_t * jnp.asarray(times_free, p.times.dtype)
    dw = jnp.broadcast_to((s_w * jnp.asarray(wps_free, p.wps.dtype))[:, None], p.wps.shape)
    dv = jnp.broadcast_to((s_v * jnp.asarray(vs_free, p.vs.dtype))[:, None], p.vs.shape)
    return jnp.concatenate([dt, dw.reshape(-1), dv.reshape(-1)])


def movable_samples(p, free_mask, t_grid) -> jnp.ndarray:
    """Which grid samples can move under the free rows of `p` as it stands now. A sample on piece j moves
    only if row j or j + 1 has a free entry and the piece is not stationary (same position at both ends,
    zero velocities: the goal hold and the pads, which stay put whatever their times). The pinned start
    pose, at t <= times[0] with row 0 fully pinned, cannot move either."""
    wps_free, vs_free, times_free = free_mask
    free = jnp.asarray(wps_free) | jnp.asarray(vs_free) | jnp.asarray(times_free)
    n = p.times.shape[0]
    j = jnp.clip(jnp.searchsorted(p.times, t_grid, side="right") - 1, 0, n - 2)
    stationary = (jnp.all(p.wps[1:] == p.wps[:-1], axis=1) & jnp.all(p.vs[1:] == 0.0, axis=1)
                  & jnp.all(p.vs[:-1] == 0.0, axis=1))
    moves = (free[j] | free[j + 1]) & ~stationary[j]
    return moves & ~((t_grid <= p.times[0]) & ~free[0])


def select_rows(h: jnp.ndarray, movable: jnp.ndarray, constraint: WallConstraint):
    """Sample indices of the `rows_per_agent` lowest windows (closest movable sample of each) and their h.
    Samples that cannot move read as +inf."""
    T, w = h.shape[0], constraint.window
    nw = -(-T // w)
    hm = jnp.where(movable, h, jnp.inf)
    hm = jnp.pad(hm, (0, nw * w - T), constant_values=jnp.inf).reshape(nw, w)
    arg, hmin = jnp.argmin(hm, axis=1), jnp.min(hm, axis=1)
    k = min(constraint.rows_per_agent, nw)
    _, win = jax.lax.top_k(-hmin, k)
    return jnp.clip(win * w + arg[win], 0, T - 1), hmin[win]


def solve_dual(C, d_metric, r, active, constraint: WallConstraint):
    """mu >= 0 minimising 1/2 mu^T S mu - mu^T r over the active rows (0 elsewhere), S = C D C^T plus a
    relative Tikhonov term, by a primal-dual active-set iteration
    (`dual_iters` small solves; exact once the active set settles, clamped to mu >= 0 otherwise). This is the dual of
    min 1/2 x^T D^-1 x subject to C x >= r', for both the direction (r = C D g) and back-projection
    (r = target - h). Rows with no reach through D (pinned entries only) are treated as inactive."""
    return solve_dual_op(C, C * d_metric[None, :], r, active, constraint)


def solve_dual_op(C, CD, r, active, constraint: WallConstraint):
    """`solve_dual` for a general metric: `CD` is C M^-1 (M the metric), (K, n)."""
    K = C.shape[0]
    reach = jnp.sum(CD * C, axis=1)
    active = active & (reach > 1e-14 * jnp.maximum(jnp.max(reach), 1e-300))
    a = active.astype(C.dtype)
    S = (CD @ C.T) * a[:, None] * a[None, :]
    S = S + constraint.reg * jnp.maximum(jnp.max(jnp.diag(S)), 1e-30) * jnp.eye(K, dtype=C.dtype)
    r = jnp.where(active, r, 0.0)
    eye = jnp.eye(K, dtype=C.dtype)

    def iterate(_, mu):
        # Primal-dual active set: hold the rows whose mu - (S mu - r) is positive, solve S_AA mu_A = r_A.
        A = active & (mu - (S @ mu - r) > 0.0)
        f = A.astype(C.dtype)
        SA = S * f[:, None] * f[None, :] + jnp.diag(1.0 - f)
        return jnp.linalg.solve(SA, r * f) * f

    mu = jax.lax.fori_loop(0, constraint.dual_iters, iterate, jnp.zeros(K, C.dtype))
    return jnp.maximum(mu, 0.0) * a


def projected_direction(g, C, d_metric, held, constraint: WallConstraint):
    """Eq 22 with C d >= 0 on the held rows: d = -D (g - C^T mu), mu >= 0."""
    return projected_direction_op(g, C, lambda v: d_metric[:, None] * v if v.ndim == 2 else d_metric * v,
                                  held, constraint)


def projected_direction_op(g, C, minv, held, constraint: WallConstraint):
    """Eq 22 for a general metric M, given `minv(v) = M^-1 v` for v of shape (n,) or (n, k)."""
    CD = minv(C.T).T
    mg = minv(g)
    mu = solve_dual_op(C, CD, C @ mg, held, constraint)
    return -(mg - CD.T @ mu)


def _where_tree(cond, a, b):
    return jax.tree_util.tree_map(lambda x, y: jnp.where(cond, x, y), a, b)


def _run_constrained(cost_fn, geometries, clearance_fn: Callable, t_grid: jnp.ndarray,
                     params_list, q0_samples_list, active_masks_list, free_masks,
                     n_steps: int, initial_step: float, max_backtracks: int,
                     metric_scale: Sequence[float], constraint: WallConstraint,
                     armijo: float = 1e-4, max_step: float = float("inf"),
                     project_times: Callable = lambda t: t, sobolev=None):
    """Drop-in for `_run_line_search` with walls as constraints. Returns (params, cost_history,
    breakdown_history), the history entries taken before each step. `sobolev` (a
    `sobolev_metric_jax.SobolevMetric`) replaces the diagonal metric by the Sobolev-Slobodeckij one in the
    descent direction, rebuilt at each iterate. Back-projection keeps the diagonal metric: it only has to
    reach the walls, and the stiff metric spreads a local lift over the whole path, which then fails the
    no-new-penetration test (on B1 cycle 0 agents started 0.35 m into a wall stayed there)."""
    from .spline_jax import eval as spline_eval
    from . import sobolev_metric_jax as sm

    n_agents = len(geometries)
    d_metrics = [metric_diagonal(p, fm, metric_scale) for p, fm in zip(params_list, free_masks)]

    def metric_op(i, p):
        """v -> M^-1 v for agent i at `p`."""
        dm = d_metrics[i]
        if sobolev is None:
            return lambda v: dm[:, None] * v if v.ndim == 2 else dm * v

        def points(x):
            pp = _unflat(x, p)
            q, _v, _a = spline_eval(geometries[i].spec, pp.times, pp.wps, pp.vs, t_grid,
                                    active_mask=active_masks_list[i])
            return jnp.mean(jax.vmap(geometries[i].fk_fn)(q), axis=1)

        chol = sm.pulled_back(points, _flat(p), t_grid, dm, sobolev)
        free = (dm > 0).astype(dm.dtype)
        return lambda v: sm.apply_inverse(chol, free, v)

    def clearance_at(i, p, t):
        q, _v, _a = spline_eval(geometries[i].spec, p.times, p.wps, p.vs, t,
                                active_mask=active_masks_list[i])
        return jax.vmap(clearance_fn)(q) - constraint.margin

    def rows_and_jacobian(i, p):
        h = clearance_at(i, p, t_grid)
        rows, h_sel = select_rows(h, movable_samples(p, free_masks[i], t_grid), constraint)
        t_sel = t_grid[rows]
        J = jax.jacrev(lambda x: clearance_at(i, _unflat(x, p), t_sel))(_flat(p))
        return h_sel, J

    def agent_violation(i, p):
        v = jax.nn.relu(-clearance_at(i, p, t_grid))
        return jnp.sum(v), jnp.max(v)

    def no_worse(new, old):
        return (new[0] <= old[0] + constraint.tol) & (new[1] <= old[1] + constraint.tol)

    def with_times(p):
        return p._replace(times=project_times(p.times))

    def finite(p):
        return jnp.all(jnp.isfinite(_flat(p)))

    def restore_agent(i, p, iters, minv):
        """Back-projection of one agent, each step kept only if it does not raise that agent's violation."""
        for _ in range(iters):
            h_sel, C = rows_and_jacobian(i, p)
            CD = minv(C.T).T
            nu = solve_dual_op(C, CD, constraint.restore_target - h_sel, h_sel < 0.0, constraint)
            delta = CD.T @ nu
            big = jnp.max(jnp.abs(delta))
            delta = delta * jnp.minimum(1.0, max_step / jnp.maximum(big, 1e-300))
            q = with_times(_unflat(_flat(p) + delta, p))
            keep = finite(q) & no_worse(agent_violation(i, q), agent_violation(i, p))
            p = _where_tree(keep, q, p)
        return p

    def value(p):
        return cost_fn(p, q0_samples_list, active_masks_list)[0]

    value_and_grad = jax.value_and_grad(lambda p: cost_fn(p, q0_samples_list, active_masks_list), has_aux=True)

    unit_ops = [(lambda dm: (lambda v: dm[:, None] * v if v.ndim == 2 else dm * v))(dm) for dm in d_metrics]
    start = [restore_agent(i, p, constraint.initial_restore_iters, unit_ops[i]) for i, p in enumerate(params_list)]

    def step(carry, _):
        params, alpha = carry
        (total, bd), grads = value_and_grad(params)
        old_viol = [agent_violation(i, p) for i, p in enumerate(params)]
        directions, minvs = [], [metric_op(i, p) for i, p in enumerate(params)]
        for i, (p, g) in enumerate(zip(params, grads)):
            h_sel, C = rows_and_jacobian(i, p)
            d = projected_direction_op(_flat(g), C, minvs[i], h_sel < constraint.band, constraint)
            directions.append(jnp.where(jnp.all(jnp.isfinite(d)), d, 0.0))
        slopes = [jnp.dot(_flat(g), d) for g, d in zip(grads, directions)]
        largest = jnp.max(jnp.stack([jnp.max(jnp.abs(d)) for d in directions]))
        first_alpha = jnp.minimum(alpha, max_step / jnp.maximum(largest, 1e-300))
        old_sum = sum(v[0] for v in old_viol)

        def score(a):
            new, slope, new_sum, moved = [], 0.0, 0.0, False
            for i, (p, d) in enumerate(zip(params, directions)):
                plain = with_times(_unflat(_flat(p) + a * d, p))
                lifted = restore_agent(i, plain, constraint.restore_iters, unit_ops[i])
                v_plain, v_lift = agent_violation(i, plain), agent_violation(i, lifted)
                trial_p = _where_tree(finite(lifted) & no_worse(v_lift, v_plain), lifted, plain)
                v_trial = agent_violation(i, trial_p)
                keep = finite(trial_p) & no_worse(v_trial, old_viol[i])
                new.append(_where_tree(keep, trial_p, p))
                slope = slope + jnp.where(keep, slopes[i], 0.0)
                new_sum = new_sum + jnp.where(keep, v_trial[0], old_viol[i][0])
                moved = moved | keep
            return value(new), slope, new_sum, moved, new

        def enough(a, s):
            cost, slope, new_sum, moved, _ = s
            better = (cost <= total + armijo * a * slope) | (new_sum < old_sum - constraint.tol)
            return moved & jnp.isfinite(cost) & better

        def keep_looking(state):
            a, s, tries = state
            return (tries < max_backtracks) & ~enough(a, s)

        def backtrack(state):
            a, _, tries = state
            a = 0.5 * a
            return a, score(a), tries + 1

        final_alpha, final_score, _ = jax.lax.while_loop(keep_looking, backtrack,
                                                         (first_alpha, score(first_alpha), 0))
        accepted = enough(final_alpha, final_score)
        new_params = _where_tree(accepted, final_score[4], list(params))
        capped_only = accepted & (first_alpha < alpha) & (final_alpha == first_alpha)
        next_alpha = jnp.where(capped_only, alpha, jnp.where(accepted, 2.0 * final_alpha, final_alpha))
        return (new_params, next_alpha), (total, bd)

    (final_params, _alpha), (cost_history, bd_history) = jax.lax.scan(
        step, (start, jnp.asarray(initial_step, dtype=params_list[0].times.dtype)), xs=None, length=n_steps)
    return final_params, cost_history, bd_history

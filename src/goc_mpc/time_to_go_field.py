"""Point-to-point obstacle-aware path tracing via gradient descent on a
caller-supplied differentiable edge-cost (time-to-go) function.

Every adapter in this module implements the same duck-typed protocol:

    trace_path(agent: int, start: np.ndarray, goal: np.ndarray) -> np.ndarray

taking and returning `(dim,)` / `(K, dim)` points in world/graph
coordinates (whatever frame the caller's waypoints are already in -- no
adapter-internal coordinate remapping to undo). This mirrors
`goc_ntfields.solvers.ntfield_solver.NTFieldSolver.trace_path`.

`EdgeCostTimeToGoField` traces by gradient descent on an
`edge_cost_fn(agent, a, b) -> float` supplied entirely by the caller -- this
module has no opinion on how that function is built. It may be backed by an
all-pairs Fast Marching Method tensor (e.g. ported from
`po_goc_mpc/experiments/objectives/fmm.py::build_all_pairs_fmm_field`), a
trained NTField, or anything else with the same (agent, a, b) -> cost shape
already used elsewhere in this codebase for `edge_cost_fn=`
(EvolutionaryWaypointSolver). Treating `a` as the free/traced position and
`b` as the fixed goal, `edge_cost_fn(agent, a, goal)` is the cost-to-go from
`a`, so descending its negative gradient in `a` traces a path toward the
goal -- the same procedure as descending a single-source arrival-time
field's gradient, just phrased as a partial derivative of the two-point
cost instead of a method on a one-point field.

The gradient is exact `jax.grad`, not finite differences, so `edge_cost_fn`
must be JAX-jit/grad compatible (written in jax.numpy, no untraceable
Python control flow on array values -- `agent` may still drive Python-level
branching, since it's a static jit argument here, see
`EdgeCostTimeToGoField`'s doc comment). If a backend can't meet that bar but
can still supply exact gradients some other way (e.g. an autograd-based
NTFieldSolver, whose own `trace_path` already does this internally), prefer
delegating to it directly -- see `NTFieldTimeToGo` below.

If `edge_cost_fn` itself interpolates a discretized field (e.g. via
`jax.scipy.ndimage.map_coordinates(..., order=1)`, the common case for every
`edge_cost_fn` built in `po_goc_mpc/experiments/objectives/`), its EXACT
gradient is only piecewise well-behaved: `order=1` interpolation is
multilinear, which is C0 continuous (the value has no jumps) but NOT C1 --
the gradient is a different constant vector per grid cell, and jumps
discontinuously at every cell face. A plain per-step descent (this module's
`momentum=0.0` default) follows that jump the instant it crosses a
boundary, which shows up as a visibly jittery/zigzagging traced path near
any region where the field curves (confirmed directly: neither a finer x/y
grid nor more angular resolution reduced it, in
`po_goc_mpc/experiments/aniso_obstacle_routing_experiment.py`/
`aniso_obstacle_avoidance_drive_experiment.py` -- consistent with a
per-cell gradient discontinuity, which persists at any grid resolution
since `step_size` is conventionally scaled with resolution too, see
`po_goc_mpc.experiments.solvers.edge_cost_time_to_go_field`). `momentum`
(an exponential moving average over each step's descent direction, not
just the current step's local one) trades a bit of path-following fidelity
for smoothing that out -- see `_build_tracer`'s own comment.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

# Positions here are physical (metre-scale) coordinates fed through many
# fixed-step gradient-descent iterations; JAX's default float32 loses that
# precision fast. Same convention as evolutionary_waypoint_solver/kernel.py
# and solver.py.
jax.config.update("jax_enable_x64", True)


def _build_tracer(edge_cost_fn, max_steps, step_size, goal_tol, momentum):
    """Builds a jax.jit-compiled `trace(agent, start, goal) -> path`
    closure over one `edge_cost_fn`/hyperparameter set. `max_steps` must be
    a fixed Python int (jax.lax.scan's output shape is static -- there is
    no variable-length early exit inside a compiled trace); instead, once a
    step's distance to `goal` is within `goal_tol`, that step and every one
    after it become a no-op (masked by `active`), so the returned path
    simply repeats its converged tail past however many steps convergence
    actually took.

    `agent` is a static jit argument (`static_argnums=(0,)`): the tracer
    recompiles once per distinct `agent` value seen, in exchange for
    letting `edge_cost_fn` branch on it with ordinary Python control flow
    (e.g. selecting a per-agent obstacle mask) rather than requiring a
    fully vectorized/traced dispatch over agents.
    """
    grad_fn = jax.grad(edge_cost_fn, argnums=1)

    def step(carry, _):
        pos, goal, agent, active, velocity = carry
        grad = grad_fn(agent, pos, goal)
        grad_norm = jnp.linalg.norm(grad)
        descent_direction = -grad / jnp.maximum(grad_norm, 1e-12)

        # Exponential moving average of recent per-step descent directions,
        # not just the current step's local one -- see this module's
        # docstring for why a plain per-step direction (momentum=0.0, the
        # default) zigzags on an edge_cost_fn built by interpolating a
        # discretized field: its exact gradient is piecewise CONSTANT per
        # grid cell and jumps at every cell boundary crossed, so a single
        # cell's jump gets outvoted by the last ~1/(1-momentum) steps'
        # worth of direction instead of being followed immediately.
        # Re-normalized (not just averaged) so `capped_step` below still
        # produces a step of the intended length regardless of `momentum`
        # -- an un-renormalized EMA would shrink whenever recent directions
        # partially cancel, silently slowing descent near a direction
        # change instead of just smoothing which way it goes.
        new_velocity = momentum * velocity + (1.0 - momentum) * descent_direction
        velocity_norm = jnp.linalg.norm(new_velocity)
        move_direction = new_velocity / jnp.maximum(velocity_norm, 1e-12)

        # Clamp the step to the remaining distance to goal: a fixed
        # step_size larger than goal_tol would otherwise overshoot past the
        # goal every time it gets within one step of it, oscillating back
        # and forth around the goal for the rest of the descent instead of
        # settling inside goal_tol (this bit for real -- verified via a
        # dist-to-goal trace on a quadratic-bowl edge_cost_fn before this
        # clamp was added).
        capped_step = jnp.minimum(step_size, jnp.linalg.norm(goal - pos))
        new_pos = jnp.where(active, pos + capped_step * move_direction, pos)
        still_moving = jnp.logical_and(
            jnp.linalg.norm(goal - new_pos) > goal_tol, velocity_norm > 1e-12)
        new_active = jnp.logical_and(active, still_moving)
        return (new_pos, goal, agent, new_active, new_velocity), new_pos

    @functools.partial(jax.jit, static_argnums=(0,))
    def trace(agent, start, goal):
        init_active = jnp.linalg.norm(goal - start) > goal_tol
        init_velocity = jnp.zeros_like(start)
        carry0 = (start, goal, agent, init_active, init_velocity)
        _, path = jax.lax.scan(step, carry0, None, length=max_steps)
        full_path = jnp.concatenate([start[None, :], path], axis=0)
        # Force the exact endpoint: the masked descent only guarantees
        # landing within goal_tol, not bit-exact -- matches every other
        # adapter's path[-1] == goal contract.
        return full_path.at[-1].set(goal)

    return trace


class EdgeCostTimeToGoField:
    """Traces an obstacle-aware path from `start` to `goal` by jax.grad
    gradient descent on `edge_cost_fn(agent, a, b)`, varying `a` with `b`
    fixed at `goal`. See module docstring for the expected edge_cost_fn
    contract and `_build_tracer` for the fixed-length/JIT descent shape.

    The whole descent is compiled once (per distinct `agent`, see
    `_build_tracer`) at construction time and reused for every subsequent
    `trace_path` call -- `max_steps`/`step_size`/`goal_tol` are therefore
    fixed hyperparameters of the instance, not per-call overrides (changing
    them would mean rebuilding a new instance / triggering a recompile).

    Args:
        edge_cost_fn: callable(agent, a, b) -> float, cost-to-go from `a`
            to `b`, JAX-jit/grad compatible. Supplied entirely by the
            caller.
        max_steps: fixed number of descent steps (static for JIT).
        step_size: gradient-descent step size, in the same units as `a`/`b`.
        goal_tol: world-space distance to `goal` at which a step freezes.
        momentum: exponential-moving-average weight (in `[0, 1)`) applied to
            each step's descent direction, `0.0` (the default) reproducing
            the original plain per-step behavior exactly. See this module's
            docstring/`_build_tracer`'s own comment for why a nonzero value
            smooths out the zigzag a piecewise-multilinear `edge_cost_fn`
            can otherwise produce.
    """

    def __init__(self, edge_cost_fn, max_steps=2000, step_size=None, goal_tol=1e-3, momentum=0.0):
        if step_size is None:
            raise ValueError("step_size must be set")
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        self.edge_cost_fn = edge_cost_fn
        self.max_steps = max_steps
        self.step_size = step_size
        self.goal_tol = goal_tol
        self.momentum = momentum
        self._trace_jit = _build_tracer(edge_cost_fn, max_steps, step_size, goal_tol, momentum)

    def trace_path(self, agent, start, goal):
        """Returns path: (max_steps + 2, dim) numpy array, `path[0] ==
        start`, `path[-1] == goal` exactly. Interior points repeat the
        converged position past wherever the descent reached goal_tol.
        """
        start = jnp.asarray(start, dtype=jnp.float64)
        goal = jnp.asarray(goal, dtype=jnp.float64)
        return np.asarray(self._trace_jit(agent, start, goal))


class NTFieldTimeToGo:
    """Thin adapter around an already-constructed
    `goc_ntfields.solvers.ntfield_solver.NTFieldSolver`, satisfying this
    module's `trace_path(agent, start, goal)` protocol by delegating
    straight to the solver's own `trace_path(start, goal)` -- it already
    does gradient descent on a (trained, not numeric) field via autograd.
    `goc_ntfields`/`torch` are optional dependencies of goc-mpc, so this
    adapter is only imported/instantiated on demand, never at module load.
    """

    def __init__(self, ntfield_solver):
        self._solver = ntfield_solver

    def trace_path(self, agent, start, goal, **kwargs):
        del agent  # single shared field for now, same limitation as EdgeCostTimeToGoField
        return np.asarray(self._solver.trace_path(start, goal, **kwargs))

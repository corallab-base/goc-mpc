"""PymanoptShortPathMPC: a from-scratch, manifold-respecting short-horizon
tracking solver, duck-typed to GraphShortPathMPC's/AdmmShortPathMPC's public
surface (solve/view_points/view_vels/view_times) so it's a drop-in
`short_path_mpc=` for GraphOfConstraintsMPC -- the same extension point
already used for TracedTimingMPC and EvolutionaryWaypointSolver.

Motivation: neither GraphShortPathMPC (SLSQP) nor AdmmShortPathMPC track a
Torus (e.g. yaw) reference correctly across its own wraparound -- both
optimize a flat Euclidean state with no notion that some components are
periodic. This solver instead defines an actual Riemannian manifold matching
the robot's real configuration space (Block.R -> Euclidean, Block.Torus ->
a product of circles) and optimizes ON it via pymanopt, so "smooth" and
"close to the reference" are measured correctly near a wrap boundary by
construction, not patched after the fact.

Each Torus component is represented as a point on the unit circle (cos, sin)
-- pymanopt has no dedicated Torus/circle manifold, but Sphere(2) IS exactly
S^1 embedded in R^2, the standard trick. This also sidesteps pymanopt's own
manifold methods (`.log()`/`.dist()`), which are hardcoded to plain numpy
internally and break under jax tracing (confirmed empirically -- calling
Sphere.log() from inside a jax-decorated cost raises
TracerArrayConversionError): every cost here is instead a plain AMBIENT
(chordal) squared distance in the embedding, computed in pure jax. Chordal
distance is also the more numerically robust choice regardless -- geodesic
(arccos-based) distance has a singular derivative at zero separation, i.e.
exactly where tracking is supposed to converge.

Scope of this first version (deliberately smaller than AdmmShortPathMPC):
- Block.R / Block.Torus only -- SO3Quat/SO3Mat would need their own pymanopt
  factor (Sphere(4) / SpecialOrthogonalGroup(3)) and aren't wired up yet.
- No final obstacle-projection pass (AdmmShortPathMPC's own hard-feasibility
  safety net) -- soft repulsion cost only.
- No inter-agent collision cost -- solved independently per agent, same as
  AdmmShortPathMPC's own per-agent loop.
- Obstacle geometry: registered spheres/boxes (ObstacleSet.obstacles()) only,
  not point clouds.
- pymanopt rebuilds its manifold/cost/problem from scratch every solve()
  call (references/obstacles differ every cycle), so jax retraces every
  cycle -- no cross-cycle JIT caching. This is optimizing for correctness
  first, not "microseconds fast" like AdmmShortPathMPC; keeping the manifold
  shape static and threading references through as arguments (instead of
  closed-over Python values) would let jax cache the compiled
  gradient across cycles, if this needs to get faster later.

Obstacle avoidance is evaluated against a link's WORLD position via
default_fk.resolve_link_fk (registry override, else the built-in
closed-form pose for the robot's configuration-space kind) -- not the raw
leading ambient columns AdmmShortPathMPC/GraphShortPathMPC use, which is
only a valid "position" for a point-mass/pos-yaw robot in the first place.
For those built-in kinds resolve_link_fk's default IS exactly the leading
`workspace_dim` columns, so this is a strict generalization: zero behavior
change for the robots this repo currently exercises, correct obstacle
geometry the moment a real fk_fn is registered for an articulated one.
Reconstructing a raw angle from its (cos, sin) factor (via atan2, only
needed to build the ambient row fk_fn expects) reintroduces a branch-cut
discontinuity at the one specific configuration antipodal to wherever
atan2's own cut sits -- a real but narrow, well-scoped limitation of calling
an angle-based fk_fn from a circle-embedded representation; it does not
affect the tracking/smoothness terms above, which never form an angle.
"""

import logging
import time

import jax.numpy as jnp
import numpy as np
import pymanopt
from pymanopt.manifolds import Euclidean, Product, Sphere
from pymanopt.optimizers import SteepestDescent

from ._ext.configuration_spline import BlockType
from ._ext.goc_mpc import ObstacleKind
from .evolutionary_waypoint_solver.default_fk import resolve_link_fk

logger = logging.getLogger(__name__)


def _agent_factor_layout(blocks):
    """One entry per pymanopt manifold factor for a single agent's config,
    in ambient column order: (block, component_or_None) -- component is
    None for a whole Block.R factor, else the index of a Block.Torus block's
    angle. Raises for any other block type -- see module docstring."""
    layout = []
    for block in blocks:
        if block.type == BlockType.R:
            layout.append((block, None))
        elif block.type == BlockType.Torus:
            for component in range(block.size):
                layout.append((block, component))
        else:
            raise ValueError(
                f"PymanoptShortPathMPC: unsupported block type {block.type!r} -- "
                "only Block.R and Block.Torus are supported so far (see this "
                "module's docstring).")
    return layout


def _agent_manifold(layout, num_steps):
    """Flat (pymanopt disallows nested Product) manifold for `num_steps`
    copies of this agent's per-config factor layout -- Euclidean(block.size)
    for an R block, Sphere(2) (the unit circle, embedding a single angle as
    (cos, sin)) per Torus component."""
    factors = []
    for _ in range(num_steps):
        for block, component in layout:
            factors.append(Euclidean(block.size) if component is None else Sphere(2))
    return Product(factors)


def _row_to_factors(row, layout):
    """Ambient config row (dim,) -> list of per-factor arrays, matching
    _agent_manifold's factor order for ONE step."""
    factors = []
    col = 0
    for block, component in layout:
        if component is None:
            factors.append(row[col:col + block.size])
            col += block.size
        else:
            theta = row[col + component]
            factors.append(jnp.stack([jnp.cos(theta), jnp.sin(theta)]))
            if component == block.size - 1:
                col += block.size
    return factors


def _factors_to_row(factors_by_step, layout, dim):
    """Inverse of the per-step half of _row_to_factors, for every step at
    once: (num_steps, dim) ambient array from pymanopt's flat result.point
    (grouped num_steps-major, matching _agent_manifold's factor order)."""
    n_factors_per_step = len(layout)
    num_steps = len(factors_by_step) // n_factors_per_step
    out = np.zeros((num_steps, dim))
    for step in range(num_steps):
        col = 0
        for f, (block, component) in enumerate(layout):
            value = np.asarray(factors_by_step[step * n_factors_per_step + f])
            if component is None:
                out[step, col:col + block.size] = value
                col += block.size
            else:
                out[step, col + component] = np.arctan2(value[1], value[0])
                if component == block.size - 1:
                    col += block.size
    return out


def _wrap_pi(delta):
    """Same convention as splines.hpp's torus::wrap_pi -- wraps to (-pi, pi]."""
    r = np.fmod(delta + np.pi, 2.0 * np.pi)
    r = np.where(r <= 0, r + 2.0 * np.pi, r)
    return r - np.pi


def _finite_difference_vels(rows, layout, dt):
    """Wrap-aware finite-difference velocity estimate from a (num_steps,
    dim) ambient position sequence -- pure post-processing (not part of the
    differentiable optimization above), mirroring torus::shortest_delta's
    convention for a Torus component instead of a raw (2*pi-vulnerable)
    difference. First row repeats the second row's estimate (no earlier
    sample to difference against)."""
    num_steps, dim = rows.shape
    vels = np.zeros_like(rows)
    for step in range(1, num_steps):
        col = 0
        for block, component in layout:
            if component is None:
                vels[step, col:col + block.size] = (
                    rows[step, col:col + block.size] - rows[step - 1, col:col + block.size]) / dt
                col += block.size
            else:
                vels[step, col + component] = _wrap_pi(
                    rows[step, col + component] - rows[step - 1, col + component]) / dt
                if component == block.size - 1:
                    col += block.size
    if num_steps > 1:
        vels[0] = vels[1]
    return vels


def _safe_norm(vec, eps=1e-12):
    """jnp.linalg.norm's gradient is NaN exactly at ||vec||==0 (0/0 in
    d(sqrt(sum(x^2)))/dx) -- a real singularity here, not a corner case:
    gradient-descending a repulsion cost routinely pushes an iterate
    straight through zero separation from the obstacle center/surface on
    its way past. The epsilon keeps the gradient finite everywhere with a
    negligible bias (sqrt(1e-12) = 1e-6) instead of poisoning the whole
    trajectory's gradient with a single NaN term."""
    return jnp.sqrt(jnp.sum(vec ** 2) + eps)


def _obstacle_cost(world_pos, obstacles, margin):
    """Smooth hinge-squared repulsion against every registered sphere/box
    obstacle (ObstacleSet.obstacles()), evaluated at a single workspace_dim
    world position. Point-cloud obstacles aren't supported yet -- see this
    module's docstring."""
    total = 0.0
    for obstacle in obstacles.obstacles():
        d = world_pos.shape[0]
        center = jnp.asarray(obstacle.params[:d])
        pad = margin + obstacle.margin
        if obstacle.kind == ObstacleKind.kSphere:
            radius = obstacle.params[d]
            dist = _safe_norm(world_pos - center)
            total += jnp.maximum(0.0, radius + pad - dist) ** 2
        elif obstacle.kind == ObstacleKind.kBox:
            half_extents = jnp.asarray(obstacle.params[d:2 * d])
            # Axis-aligned box SDF: 0 inside, true distance outside.
            offset = jnp.abs(world_pos - center) - half_extents
            dist = _safe_norm(jnp.maximum(offset, 0.0))
            total += jnp.maximum(0.0, pad - dist) ** 2
    return total


class PymanoptShortPathMPC:

    def __init__(
            self,
            graph,
            num_steps: int,
            num_agents: int,
            dim: int,
            time_per_step: float,
            obstacles,
            tracking_weight: float = 1.0,
            smoothness_weight: float = 0.1,
            obstacle_weight: float = 1.0,
            obstacle_margin: float = 0.0,
            link_name: str = "ee",
            max_iterations: int = 50,
    ):
        self.graph = graph
        self.num_steps = num_steps
        self.num_agents = num_agents
        self.dim = dim
        self.time_per_step = time_per_step
        self.obstacles = obstacles
        self.tracking_weight = tracking_weight
        self.smoothness_weight = smoothness_weight
        self.obstacle_weight = obstacle_weight
        self.obstacle_margin = obstacle_margin
        self.link_name = link_name
        self.max_iterations = max_iterations

        self._times = np.array([(i + 1) * time_per_step for i in range(num_steps)])
        self._points = np.zeros((num_steps, num_agents * dim))
        self._vels = np.zeros((num_steps, num_agents * dim))
        self._last_solve_time = 0.0

    def solve(self, x0, v0, var_assignments, remaining_vertices, references):
        # var_assignments/remaining_vertices unused -- matches
        # AdmmShortPathMPC's own signature (agents solved independently,
        # no per-node/graph-topology dependence at the short-path level).
        #
        # x0/v0 unused too, unlike AdmmShortPathMPC (which explicitly
        # anchors its first optimized step to them): `references` is
        # assumed already fit from the real current state each cycle (the
        # same x0/v0 GraphOfConstraintsMPC._solve_for_timing just passed to
        # TracedTimingMPC/GraphTimingMPC.solve), so tracking `references`
        # itself keeps this implicitly anchored -- a simplification, not
        # (yet) a hard boundary condition; a caller feeding a stale/frozen
        # reference would need this to change.
        del var_assignments, remaining_vertices, x0, v0
        start = time.monotonic()
        try:
            points = np.zeros((self.num_steps, self.num_agents * self.dim))
            vels = np.zeros((self.num_steps, self.num_agents * self.dim))

            for agent_id in range(self.num_agents):
                layout = _agent_factor_layout(self.graph._robot_specs[agent_id])
                manifold = _agent_manifold(layout, self.num_steps)

                ref_pos, _ref_vel = references[agent_id].eval_multiple(self._times)
                ref_pos = np.asarray(ref_pos)
                ref_factors = [
                    jnp.asarray(f)
                    for step in range(self.num_steps)
                    for f in _row_to_factors(ref_pos[step], layout)
                ]

                fk_fn = resolve_link_fk(self.graph, agent_id, self.link_name)
                n_per_step = len(layout)
                tracking_weight = self.tracking_weight
                smoothness_weight = self.smoothness_weight
                obstacle_weight = self.obstacle_weight
                obstacle_margin = self.obstacle_margin
                obstacles = self.obstacles
                dim = self.dim
                num_steps = self.num_steps

                @pymanopt.function.jax(manifold)
                def cost(*args):
                    total = 0.0
                    prev_step_factors = None
                    for step in range(num_steps):
                        step_factors = args[step * n_per_step:(step + 1) * n_per_step]

                        for factor, ref_factor in zip(
                                step_factors, ref_factors[step * n_per_step:(step + 1) * n_per_step]):
                            total += tracking_weight * jnp.sum((factor - ref_factor) ** 2)

                        if obstacle_weight > 0.0:
                            row = jnp.concatenate([
                                factor if component is None else
                                jnp.array([jnp.arctan2(factor[1], factor[0])])
                                for factor, (block, component) in zip(step_factors, layout)
                            ])
                            world_pos, _rot = fk_fn(row)
                            total += obstacle_weight * _obstacle_cost(
                                world_pos, obstacles, obstacle_margin)

                        if prev_step_factors is not None and smoothness_weight > 0.0:
                            for factor, prev_factor in zip(step_factors, prev_step_factors):
                                total += smoothness_weight * jnp.sum((factor - prev_factor) ** 2)
                        prev_step_factors = step_factors
                    return total

                problem = pymanopt.Problem(manifold, cost)
                optimizer = SteepestDescent(max_iterations=self.max_iterations, verbosity=0)
                result = optimizer.run(problem, initial_point=[np.asarray(f) for f in ref_factors])

                agent_rows = _factors_to_row(result.point, layout, self.dim)
                col0 = agent_id * self.dim
                points[:, col0:col0 + self.dim] = agent_rows
                vels[:, col0:col0 + self.dim] = _finite_difference_vels(
                    agent_rows, layout, self.time_per_step)

            self._points = points
            self._vels = vels
            self._last_solve_time = time.monotonic() - start
            return True
        except Exception as e:
            logger.warning("PymanoptShortPathMPC.solve failed: %s", e)
            self._last_solve_time = time.monotonic() - start
            return False

    def view_points(self):
        return self._points

    def view_vels(self):
        return self._vels

    def view_times(self):
        return self._times

    def get_last_solve_time(self):
        return self._last_solve_time

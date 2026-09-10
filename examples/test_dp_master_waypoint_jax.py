"""`DpMasterWaypointSolver(dp_backend="jax")` -- the build-once jitted grid
solver (dp_master_jax.make_dp_master_jax) wired behind the same
`waypoint_mpc=` interface -- must produce the same waypoint matrix /
assignments / node times as the default `dp_backend="python"`
(dp_master.solve_dp_master), through several `.solve()` cycles including a
committed anchor.

Run: python examples/test_dp_master_waypoint_jax.py
"""

import time

import numpy as np
import jax.numpy as jnp
from pydrake.math import eq, ge

from goc_mpc import GraphOfConstraints
from goc_mpc._ext.configuration_spline import Block, CubicConfigurationSpline
from goc_mpc.evolutionary_waypoint_solver.projection import ProjOperator
from goc_mpc.logic_based_benders_solver import DpMasterWaypointSolver

DIM = 2
EU = lambda a, b: jnp.sqrt(jnp.sum((jnp.asarray(b) - jnp.asarray(a)) ** 2))


def build_scene():
    """Two arms, a var_agent_q multi-branch pin at n0 (full-resolve +
    branched), a cross-agent ordering edge, plus a fixed-agent leg."""
    cands_xy = np.array([[9.0, 9.0], [1.0, 0.0], [-8.0, 7.0]])
    g = GraphOfConstraints([[Block.R(DIM)], [Block.R(DIM)]], [],
                           state_lower_bound=-25.0, state_upper_bound=25.0,
                           robot_names=["r0", "r1"])
    n0, n1, n2 = g.structure.add_nodes(3)
    g.structure.add_edge(n0, n1, True)
    g.structure.add_edge(n1, n2, True)
    var = g.add_variable()
    g.add_constraint(n0, ge(g.var_agent_q(var), np.array([-25.0, -25.0])),
                     proj=ProjOperator(pins=g.var_agent_q(var), reads=(), continuous_params=0,
                                       discrete_params=3,
                                       func=lambda psi, b: jnp.asarray(cands_xy)[b]))
    g.add_constraint(n1, eq(g.var_agent_q(var), np.array([2.0, 2.0])),
                     proj=ProjOperator(pins=g.var_agent_q(var), reads=(),
                                       func=lambda psi, b: np.array([2.0, 2.0])))
    g.add_constraint(n2, eq(g.agent_q(1), np.zeros(DIM)),
                     proj=ProjOperator(pins=g.agent_q(1), reads=(), continuous_params=0,
                                       discrete_params=1, func=lambda psi, b: np.array([7.0, -3.0])))
    return g, var


def _solver(backend, objective):
    g, var = build_scene()
    splines = [CubicConfigurationSpline(s) for s in g._robot_specs]
    return DpMasterWaypointSolver(g, splines, objective=objective, edge_cost_fn=EU,
                                  wp_bounds=(-25.0, 25.0), dp_backend=backend), g, var


def _views(dp):
    return (np.array(dp.view_waypoints()), np.array(dp.view_assignments()),
            np.array(dp.view_var_assignments()), np.array(dp.view_t_by_node()))


def _cycle(dp, remaining, x0):
    assert dp.solve(list(remaining), x0), "solve returned False"
    return _views(dp)


def test_backend_parity():
    n_nodes = 3
    x0 = np.zeros(2 * DIM)
    x0[:DIM] = [1.0, 0.0]              # r0 sits on branch-1 candidate
    x0[DIM:] = [12.0, 12.0]

    for obj in ("avg", "minmax", "makespan"):
        dpp, _, _ = _solver("python", obj)
        dpj, _, _ = _solver("jax", obj)

        t = time.perf_counter()
        wp_p, as_p, va_p, tt_p = _cycle(dpp, range(n_nodes), x0)
        tpy = time.perf_counter() - t

        t = time.perf_counter()
        wp_j, as_j, va_j, tt_j = _cycle(dpj, range(n_nodes), x0)
        tj1 = time.perf_counter() - t
        t = time.perf_counter()
        wp_j2, *_ = _cycle(dpj, range(n_nodes), x0)
        tj2 = time.perf_counter() - t

        assert np.allclose(wp_p, wp_j, atol=1e-5, equal_nan=True), (obj, "waypoints")
        assert np.array_equal(as_p, as_j), (obj, "assignments", as_p, as_j)
        assert np.array_equal(va_p, va_j), (obj, "var_assignments", va_p, va_j)
        assert np.allclose(tt_p, tt_j, atol=1e-6), (obj, "t_by_node", tt_p, tt_j)
        assert np.allclose(wp_j, wp_j2, atol=1e-5, equal_nan=True), (obj, "jax not deterministic")

        # committed anchor: node 0 passed, r0 now parked at its n0 row
        owner = int(va_p[0])
        x0b = x0.copy()
        x0b[owner * DIM:owner * DIM + DIM] = wp_p[0, owner * DIM:owner * DIM + DIM]
        wp_p2, as_p2, va_p2, tt_p2 = _cycle(dpp, (1, 2), x0b)
        wp_j2b, as_j2b, va_j2b, tt_j2b = _cycle(dpj, (1, 2), x0b)
        assert np.allclose(wp_p2, wp_j2b, atol=1e-5, equal_nan=True), (obj, "anchor waypoints")
        assert np.array_equal(va_p2, va_j2b), (obj, "anchor var_assignments")
        assert np.allclose(tt_p2, tt_j2b, atol=1e-6), (obj, "anchor t_by_node")

        print(f"[{obj:8}] python vs jax: waypoints / assign / t identical  "
              f"(py {tpy*1e3:.0f}ms | jax compile {tj1*1e3:.0f}ms, warm {tj2*1e3:.1f}ms)")


if __name__ == "__main__":
    test_backend_parity()
    print("\nAll checks passed.")

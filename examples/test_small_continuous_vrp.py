"""`SmallContinuousVRPSolver` (algorithm="small_continuous_vrp") -- the
minimal "exact discrete master (dp_master_jax) + Lamarckian local refine"
waypoint solver. Its discrete output (assignment / var assignment / node
order / branch) must match what `make_dp_master_jax(...).run_vec` picks for
the same problem, and its waypoints must land on the analytic projection
targets (i.e. the local refine actually drives the seeded skeleton to a
feasible continuous solution).

Run: python examples/test_small_continuous_vrp.py
"""

import numpy as np
import jax.numpy as jnp
from pydrake.math import eq, ge

from goc_mpc import EvolutionaryWaypointSolver, GraphOfConstraints
from goc_mpc._ext.configuration_spline import Block, CubicConfigurationSpline
from goc_mpc.evolutionary_waypoint_solver.projection import ProjOperator
from goc_mpc.evolutionary_waypoint_solver.spec import build_graph_ordering_problem
from goc_mpc.logic_based_benders_solver.structure import warm_start_wp
from goc_mpc.logic_based_benders_solver.dp_master_jax import make_dp_master_jax

DIM = 2
EU = lambda a, b: jnp.sqrt(jnp.sum((jnp.asarray(b) - jnp.asarray(a)) ** 2))
CANDS = np.array([[9.0, 9.0], [1.0, 0.0], [-8.0, 7.0]])


def build_scene():
    g = GraphOfConstraints([[Block.R(DIM)], [Block.R(DIM)]], [],
                           state_lower_bound=-25.0, state_upper_bound=25.0,
                           robot_names=["r0", "r1"])
    n0, n1, n2 = g.structure.add_nodes(3)
    g.structure.add_edge(n0, n1, True)
    g.structure.add_edge(n1, n2, True)
    var = g.add_variable()
    g.add_constraint(n0, ge(g.var_agent_q(var), np.array([-25.0, -25.0])),
                     proj=ProjOperator(pins=g.var_agent_q(var), reads=(), continuous_params=0,
                                       discrete_params=3, func=lambda psi, b: jnp.asarray(CANDS)[b]))
    g.add_constraint(n1, eq(g.var_agent_q(var), np.array([2.0, 2.0])),
                     proj=ProjOperator(pins=g.var_agent_q(var), reads=(),
                                       func=lambda psi, b: np.array([2.0, 2.0])))
    g.add_constraint(n2, eq(g.agent_q(1), np.zeros(DIM)),
                     proj=ProjOperator(pins=g.agent_q(1), reads=(), continuous_params=0,
                                       discrete_params=1, func=lambda psi, b: np.array([7.0, -3.0])))
    return g, var


def _dp_ref(g, objective, x0_full, node_active=None, var_committed=None, var_anchor=None):
    problem = build_graph_ordering_problem(g, np.zeros((2, DIM)), wp_bounds=(-25.0, 25.0),
                                           objective=objective, edge_cost_fn=EU)
    wp = warm_start_wp(problem, np.zeros(problem.state_dim))
    jm = make_dp_master_jax(problem, objective=objective, edge_cost_fn=EU)
    return jm.run_vec(np.asarray(problem.params), wp, x0_full, x0_full,
                      node_active=node_active, var_committed=var_committed, var_anchor=var_anchor)


def _solver(objective):
    g, var = build_scene()
    splines = [CubicConfigurationSpline(s) for s in g._robot_specs]
    return EvolutionaryWaypointSolver(g, splines, objective=objective, edge_cost_fn=EU,
                                      wp_bounds=(-25.0, 25.0), pop_size=8, n_gen=25,
                                      algorithm="small_continuous_vrp"), g


def _check_targets(wp, va):
    owner = int(va[0])
    assert np.allclose(wp[1, owner * DIM:owner * DIM + DIM], [2.0, 2.0], atol=1e-3), wp[1]
    assert np.allclose(wp[2, DIM:2 * DIM], [7.0, -3.0], atol=1e-3), wp[2]
    # n0 got one of the branch candidates for its owner
    row0 = wp[0, owner * DIM:owner * DIM + DIM]
    assert min(np.linalg.norm(row0 - c) for c in CANDS) < 1e-3, row0


def test_matches_dp_master_and_is_feasible():
    x0 = np.zeros(2 * DIM)
    x0[:DIM] = [1.0, 0.0]           # r0 sits on branch-1 candidate
    x0[DIM:] = [12.0, 12.0]

    for obj in ("avg", "minmax", "makespan"):
        dp, g = _solver(obj)
        assert dp.solve([0, 1, 2], x0), (obj, "solve returned False")
        wp = np.array(dp.view_waypoints())
        va = np.array(dp.view_var_assignments())
        tt = np.array(dp.view_t_by_node())

        ref = _dp_ref(g, obj, x0)
        assert va[0] == ref["assignment"][0], (obj, "assignment", va, ref["assignment"])
        # node order agrees with the reference ranks
        ref_rank = [ref["time"][n] for n in range(3)]
        assert list(np.argsort(tt)) == list(np.argsort(ref_rank)), (obj, "order", tt, ref_rank)
        _check_targets(wp, va)

        # committed anchor: node 0 passed, its owner parked on its n0 row
        owner = int(va[0])
        x0b = x0.copy()
        x0b[owner * DIM:owner * DIM + DIM] = wp[0, owner * DIM:owner * DIM + DIM]
        assert dp.solve([1, 2], x0b), (obj, "anchor solve returned False")
        wp2 = np.array(dp.view_waypoints())
        assert np.allclose(wp2[1, owner * DIM:owner * DIM + DIM], [2.0, 2.0], atol=1e-3), (obj, wp2[1])
        assert np.allclose(wp2[2, DIM:2 * DIM], [7.0, -3.0], atol=1e-3), (obj, wp2[2])
        print(f"[{obj:8}] discrete == dp_master_jax, waypoints on target, anchor cycle OK")


if __name__ == "__main__":
    test_matches_dp_master_and_is_feasible()
    print("\nAll checks passed.")

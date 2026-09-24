"""`SmallContinuousVRPSolver(kinematic_seed=True)` -- the per-skeleton
constraint-satisfying wp seed (kinematic_seed.py).

Two planar 2-link arms on different bases, two assignable reach nodes with
NO projection (a plain FK-equality residual, like the G1 joint-space scene):
  * node A's target is reachable only by arm 0 -- the seed must certify the
    arm-0 skeletons (CV ~ 0) and flag the arm-1 ones (CV > 0);
  * node B's target is reachable by both -- routing decides.

Checks the seed directly, then an MPC-budget solve (pop 12, n_gen 1,
inner_maxiter 10, as po-goc-mpc runs it) with and without the seed.

Run: python examples/test_kinematic_seed.py
"""

import numpy as np
import jax.numpy as jnp
from pydrake.math import eq

from goc_mpc import EvolutionaryWaypointSolver, GraphOfConstraints
from goc_mpc._ext.configuration_spline import Block, CubicConfigurationSpline
from goc_mpc.evolutionary_waypoint_solver.kinematic_seed import make_kinematic_seed
from goc_mpc.evolutionary_waypoint_solver.problem import full_active_anchor
from goc_mpc.evolutionary_waypoint_solver.spec import build_graph_ordering_problem
from goc_mpc.logic_based_benders_solver.dp_master_jax import make_dp_master_jax

DIM = 2
BASES = np.array([[-1.5, 0.0], [1.5, 0.0]])
TARGET_A = np.array([-2.8, 0.8])   # arm 0 only (|.-B1| = 4.4 > reach 2)
TARGET_B = np.array([0.0, 0.8])    # both arms
X0 = np.array([0.3, 1.0, 2.0, 1.0])


def fk(base):
    def f(q):
        a, b = q[0], q[1]
        pos = jnp.asarray(base) + jnp.array([jnp.cos(a) + jnp.cos(a + b),
                                             jnp.sin(a) + jnp.sin(a + b)])
        c, s = jnp.cos(a + b), jnp.sin(a + b)
        return pos, jnp.array([[c, -s], [s, c]])
    return f


def build_scene():
    g = GraphOfConstraints([[Block.R(DIM)], [Block.R(DIM)]], [],
                           state_lower_bound=-np.pi, state_upper_bound=np.pi,
                           robot_names=["arm0", "arm1"], workspace_dim=2)
    for j in range(2):
        g.set_robot_fk(j, "ee", fk(BASES[j]))
    na, nb = g.structure.add_nodes(2)
    v, w = g.add_variable(), g.add_variable()
    g.add_constraint(na, eq(g.var_agent_link_pos(v, "ee"), TARGET_A))
    g.add_constraint(nb, eq(g.var_agent_link_pos(w, "ee"), TARGET_B))
    return g, (na, nb)


def ee(q, j):
    return np.asarray(fk(BASES[j])(jnp.asarray(q))[0])


def test_seed_certifies_reachable_skeletons():
    g, (na, nb) = build_scene()
    problem = build_graph_ordering_problem(g, X0.reshape(2, DIM), wp_bounds=(-np.pi, np.pi),
                                           objective="avg")
    dp = make_dp_master_jax(problem, objective="avg")
    seed = make_kinematic_seed(problem, dp.A, dp.AUX, dp.POS[0])
    wp, cv = seed(jnp.asarray(X0), jnp.asarray(problem.params), full_active_anchor(problem))
    wp, cv = np.asarray(wp), np.asarray(cv)
    for g_idx, (va, vb) in enumerate(np.asarray(dp.A)):
        print(f"  skeleton v->{va} w->{vb}: seed CV {cv[g_idx]:.2e}")
        # an unreachable node shares the LM's damping with the rest, so a
        # reachable row in an infeasible skeleton converges more slowly
        tol = 1e-4 if va == 0 else 1e-3
        assert np.allclose(ee(wp[g_idx, nb, vb * DIM:(vb + 1) * DIM], vb), TARGET_B, atol=tol)
        if va == 0:
            assert cv[g_idx] < 1e-6, (va, vb, cv[g_idx])
            assert np.allclose(ee(wp[g_idx, na, 0:DIM], 0), TARGET_A, atol=1e-4)
        else:
            assert cv[g_idx] > 1.0, (va, vb, cv[g_idx])
    print("[seed] arm-0 skeletons certified, arm-1 skeletons flagged")


def solve_cycles(kinematic_seed, n_cycles=4):
    g, (na, nb) = build_scene()
    splines = [CubicConfigurationSpline(s) for s in g._robot_specs]
    solver = EvolutionaryWaypointSolver(
        g, splines, objective="avg", wp_bounds=(-np.pi, np.pi),
        pop_size=12, n_gen=1, outer_iters=1, inner_maxiter=10,
        algorithm="small_continuous_vrp", kinematic_seed=kinematic_seed)
    cvs = []
    for _ in range(n_cycles):
        solver.solve([na, nb], X0)
        cvs.append(float(solver.get_last_fitness()[1]))
    return solver, cvs, (na, nb)


def test_mpc_budget_solve():
    results = {}
    for ks in (False, True):
        solver, cvs, (na, nb) = solve_cycles(ks)
        va = np.asarray(solver.view_var_assignments())
        wp = np.asarray(solver.view_waypoints())
        results[ks] = (va, cvs)
        print(f"[kinematic_seed={ks!s:5}] var assignment {va.tolist()}, "
              f"CV per cycle {[f'{c:.1e}' for c in cvs]}, "
              f"ee A {ee(wp[na, va[0] * DIM:(va[0] + 1) * DIM], va[0])}")
    va, cvs = results[True]
    assert va[0] == 0, va
    assert cvs[0] < 1e-3, cvs


if __name__ == "__main__":
    test_seed_certifies_reachable_skeletons()
    test_mpc_budget_solve()
    print("\nAll checks passed.")

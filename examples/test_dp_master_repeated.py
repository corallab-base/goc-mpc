"""Repeated-cycle behaviour for the LBBD discrete solve -- node_candidates
+ solve_dp_master called several times on ONE problem, as an MPC loop would.
jit_apply_projections (problem.py) must amortise the projection-chain
compile after cycle 0 so cycles 1+ are milliseconds, and the answer must be
stable across cycles. Synthetic hold chain (no mujoco).

Run: python examples/test_dp_master_repeated.py
"""

import time
import numpy as np
from pydrake.math import eq

from goc_mpc import GraphOfConstraints
from goc_mpc._ext.configuration_spline import Block
from goc_mpc.evolutionary_waypoint_solver.projection import ProjOperator
from goc_mpc.evolutionary_waypoint_solver.spec import build_graph_ordering_problem
from goc_mpc.logic_based_benders_solver.structure import (
    node_instances, node_candidates, warm_start_wp)
from goc_mpc.logic_based_benders_solver.dp_master import solve_dp_master

DIM = 2


def _euclid(a, b):
    return float(np.linalg.norm(np.asarray(b) - np.asarray(a)))


def _build():
    """Two 2-branch self-contained agent projections (exercise the
    node_candidates per-entry jit) + a hold on the object (exercise the
    gated stationary / rigid-carry entries that ride along)."""
    grasp = np.array([1.0, 0.0])
    a_close = np.array([1.0, 0.0])
    a_far = np.array([9.0, 9.0])
    g = GraphOfConstraints([[Block.R(DIM)]], [[Block.R(DIM)]],
                           state_lower_bound=-30.0, state_upper_bound=30.0,
                           robot_names=["r0"], object_names=["o0"])
    n0, n1 = g.structure.add_nodes(2)
    g.structure.add_edge(n0, n1, True)
    g.add_constraint(n0, eq(g.agent_q(0), grasp))
    # a 2-branch self-contained pin on the agent at n1: the DP must pick the
    # near branch.
    proj = ProjOperator(pins=g.agent_q(0), reads=(), continuous_params=0, discrete_params=2,
                        func=lambda psi, b: np.where(b == 0, a_far, a_close))
    g.add_constraint(n1, eq(g.agent_q(0), np.zeros(DIM)), proj=proj)
    g.add_hold(n0, n1, 0, [0])
    return g, (n0, n1), grasp


def main():
    g, (n0, n1), grasp = _build()
    problem = build_graph_ordering_problem(g, grasp.reshape(1, DIM), wp_bounds=(-30.0, 30.0))
    params = np.asarray(g.view_param_values())
    inst = node_instances(problem)

    x0_full = np.zeros(problem.state_dim)
    x0_full[:DIM] = grasp
    x0_rows = {0: np.pad(grasp, (0, problem.state_dim - DIM))}

    times = []
    objs = []
    for cyc in range(4):
        t = time.perf_counter()
        wp = warm_start_wp(problem, grasp.reshape(1, DIM))
        cands = node_candidates(problem, wp, params, allow_unresolved=True)
        r = solve_dp_master(problem, cands, wp, x0_rows, inst,
                            ordering_edges=problem.ordering_edges, edge_cost_fn=_euclid,
                            objective="makespan", x0_full=x0_full)
        dt = (time.perf_counter() - t) * 1e3
        times.append(dt)
        objs.append(r["objective"])
        print(f"cycle {cyc}: {dt:8.1f} ms   obj={r['objective']:.4f}   "
              f"branch={r['branch']}   routes={r['routes']}")
        assert r["status"] == "OPTIMAL"

    # DP picks the near branch (cost 0 from the depot at grasp, then 0 edge).
    assert all(abs(o - objs[0]) < 1e-6 for o in objs), objs
    assert abs(objs[0]) < 1e-6, objs[0]
    # cycle 0 pays the compile; steady-state cycles must be far cheaper.
    assert times[2] < 0.25 * times[0], (times, "jit not amortising across cycles")
    assert times[3] < 50.0, times
    print(f"\nsteady-state cycle: {times[2]:.1f} ms  (cycle 0 with compile: {times[0]:.0f} ms) -- OK")


if __name__ == "__main__":
    main()

"""solve_dp_master's per-schedule projection resolution (dp_master.py's
`_needs_full_resolve` path) on a small hold graph -- gated stationary +
rigid-carry projections resolved via apply_projections per enumerated
(assignment, order, branch) schedule, no precomputed cost tables.

Run: python examples/test_dp_master_gated.py
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
from goc_mpc.logic_based_benders_solver.dp_master import solve_dp_master, _needs_full_resolve

DIM = 2


def _euclid(a, b):
    return float(np.linalg.norm(np.asarray(b) - np.asarray(a)))


def build():
    go = np.array([0.3, 0.0])
    ps = np.array([5.0, 3.0])
    os_ = np.array([2.0, 0.0])
    g = GraphOfConstraints([[Block.R(DIM)]], [[Block.R(DIM)]],
                           state_lower_bound=-20.0, state_upper_bound=20.0,
                           robot_names=["r0"], object_names=["o0"])
    n0, n1, n2 = g.structure.add_nodes(3)
    g.structure.add_edge(n0, n1, True)
    g.structure.add_edge(n1, n2, True)
    g.add_constraint(n0, eq(g.agent_q(0), g.object_q(0) + go),
                     proj=ProjOperator(pins=g.agent_q(0), reads=(g.object_q(0),),
                                       func=lambda o, psi, b: o + go))
    g.add_constraint(n2, eq(g.agent_q(0), ps + go),
                     proj=ProjOperator(pins=g.agent_q(0), reads=(),
                                       func=lambda psi, b: ps + go))
    g.add_hold(n0, n2, 0, [0])
    return g, (n0, n1, n2), os_, go, ps


def main():
    g, (n0, n1, n2), os_, go, ps = build()
    problem = build_graph_ordering_problem(g, (os_ + go).reshape(1, DIM), wp_bounds=(-20.0, 20.0))
    params = np.asarray(g.view_param_values())

    assert _needs_full_resolve(problem), "expected gated/chained projections -> full resolve path"

    wp = warm_start_wp(problem, (os_ + go).reshape(1, DIM))
    cands = node_candidates(problem, wp, params, allow_unresolved=True)
    inst = node_instances(problem)

    x0_full = np.zeros(problem.state_dim)
    x0_full[:DIM] = os_ + go          # agent depot
    x0_full[problem.state_dim - DIM:] = os_   # object depot (object seg is last)
    x0_rows = {0: np.pad(os_ + go, (0, problem.state_dim - DIM))}

    for obj in ("avg", "minmax", "makespan"):
        t = time.perf_counter()
        r = solve_dp_master(problem, cands, wp, x0_rows, inst,
                            ordering_edges=problem.ordering_edges,
                            edge_cost_fn=_euclid, objective=obj, x0_full=x0_full)
        dt = (time.perf_counter() - t) * 1e3
        print(f"[{obj:8}] {r['status']}  obj={r['objective']:.4f}  routes={r['routes']}  "
              f"({dt:.1f} ms)")
        assert r["status"] == "OPTIMAL"
        # agent visits n0 (grasp) then n2 (place): route cost = depot->grasp + grasp->place
        # grasp EE = os_+go = depot -> first leg 0; place EE = ps+go.
        expected = _euclid(os_ + go, os_ + go) + _euclid(os_ + go, ps + go)
        assert abs(r["objective"] - expected) < 1e-6, (r["objective"], expected)

    print("\nOK")


if __name__ == "__main__":
    main()

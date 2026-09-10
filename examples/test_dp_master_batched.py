"""The vectorized (JAX) non-full ext loop in `solve_dp_master` returns the
byte-identical solution the scalar per-ext loop does -- it only replaces the
O(#exts x #branch-combos) Python loop with one batched kernel launch.

A/B every scene of discrete_solver_benchmark.py with
`_DP_MASTER_BATCH_NONFULL` toggled, plus a hand makespan scene where the
branch choice genuinely couples with cross-agent waiting.

Run: python examples/test_dp_master_batched.py
"""

import io
import re
import contextlib
import runpy

import numpy as np

import goc_mpc.logic_based_benders_solver.dp_master as dm


def _bench_lines():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        runpy.run_path("examples/discrete_solver_benchmark.py", run_name="__main__")
    scrub = re.compile(r"\s*(solve_s=\S+|\(\s*[\d.]+\s*ms\))")
    return [scrub.sub("", ln) for ln in buf.getvalue().splitlines()
            if any(k in ln for k in ("objective=", "status=", "obj=", "route"))]


def _script_lines(path):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        runpy.run_path(path, run_name="__main__")
    scrub = re.compile(r"\s*(solve_s=\S+|\(\s*[\d.]+\s*ms\))")
    return [scrub.sub("", ln) for ln in buf.getvalue().splitlines()]


def test_full_resolve_ab():
    """The batched full-resolve path (one `apply_projections` launch per
    branched entry across all exts) returns what the 2-calls-per-ext scalar
    path does -- A/B the gated + dynamic/chained projection scenes."""
    scripts = ["examples/test_dp_master_gated.py", "examples/test_proj_operator.py"]
    dm._DP_MASTER_BATCH_FULL = True
    batched = [_script_lines(s) for s in scripts]
    dm._DP_MASTER_BATCH_FULL = False
    scalar = [_script_lines(s) for s in scripts]
    dm._DP_MASTER_BATCH_FULL = True
    for s, a, b in zip(scripts, batched, scalar):
        assert a == b, f"{s}:\n" + "\n".join(
            f"{'  ' if x == y else 'X '}{x}" + ("" if x == y else f"   <=>   {y}")
            for x, y in zip(a, b))
    print(f"[full-resolve] {sum(map(len, batched))} lines identical batched vs scalar "
          f"across {len(scripts)} scripts -- OK")


def test_full_resolve_jax_scoring_ab():
    """A full-resolve scene (dynamic `var_agent_q` multi-branch pin) with a
    jnp-traceable edge cost fn -- so `_ext_full_batched` runs its vmapped
    scoring kernel (not the scalar fallback). A/B every objective vs the
    scalar path."""
    import jax.numpy as jnp
    from pydrake.math import eq, ge
    from goc_mpc import GraphOfConstraints
    from goc_mpc._ext.configuration_spline import Block
    from goc_mpc.evolutionary_waypoint_solver.projection import ProjOperator
    from goc_mpc.evolutionary_waypoint_solver.spec import build_graph_ordering_problem
    from goc_mpc.logic_based_benders_solver.structure import (
        node_instances, node_candidates, warm_start_wp)

    D = 2
    euclid = lambda a, b: jnp.sqrt(jnp.sum((jnp.asarray(b) - jnp.asarray(a)) ** 2))
    cands_xy = np.array([[9.0, 9.0], [1.0, 0.0], [-8.0, 7.0]])
    g = GraphOfConstraints([[Block.R(D)], [Block.R(D)]], [],
                           state_lower_bound=-20.0, state_upper_bound=20.0,
                           robot_names=["r0", "r1"])
    n0, n1 = g.structure.add_nodes(2)
    g.structure.add_edge(n0, n1, True)
    var = g.add_variable()
    g.add_constraint(n0, ge(g.var_agent_q(var), np.array([-20.0, -20.0])),
                     proj=ProjOperator(pins=g.var_agent_q(var), reads=(), continuous_params=0,
                                       discrete_params=3,
                                       func=lambda psi, branch: jnp.asarray(cands_xy)[branch]))
    g.add_constraint(n1, eq(g.var_agent_q(var), np.array([2.0, 2.0])),
                     proj=ProjOperator(pins=g.var_agent_q(var), reads=(),
                                       func=lambda psi, b: np.array([2.0, 2.0])))
    x0 = np.concatenate([cands_xy[1], np.array([40.0, 40.0])])
    problem = build_graph_ordering_problem(g, x0.reshape(2, D), wp_bounds=(-20.0, 20.0),
                                           objective="avg", edge_cost_fn=euclid)
    params = np.asarray(g.view_param_values())
    x0_full = np.zeros(problem.state_dim)
    x0_full[:len(x0)] = x0
    wpt = warm_start_wp(problem, x0)
    cands = node_candidates(problem, wpt, params, allow_unresolved=True)
    inst = node_instances(problem)
    x0_rows = {0: np.pad(cands_xy[1], (0, problem.state_dim - D)),
               1: np.pad(np.array([40.0, 40.0]), (D, problem.state_dim - 2 * D))}

    def solve(obj):
        return dm.solve_dp_master(problem, cands, wpt, x0_rows, inst,
                                  ordering_edges=problem.ordering_edges, edge_cost_fn=euclid,
                                  objective=obj, x0_full=x0_full)

    for obj in ("avg", "minmax", "makespan"):
        dm._DP_MASTER_BATCH_FULL = True
        a = solve(obj)
        dm._DP_MASTER_BATCH_FULL = False
        b = solve(obj)
        dm._DP_MASTER_BATCH_FULL = True
        for k in ("status", "objective", "branch", "assignment", "time", "routes"):
            assert a[k] == b[k], (obj, k, a[k], b[k])
        assert a["status"] == "OPTIMAL"
        assert a["assignment"][problem.var_id_to_slot[var]] == 0 and a["branch"] == {(0, 0): 1}, a
    print("[full-resolve jax] avg/minmax/makespan: vmapped scoring == scalar, "
          "picks (var->agent0, branch 1) -- OK")


def test_benchmark_ab():
    dm._DP_MASTER_BATCH_NONFULL = True
    batched = _bench_lines()
    dm._DP_MASTER_BATCH_NONFULL = False
    scalar = _bench_lines()
    dm._DP_MASTER_BATCH_NONFULL = True
    assert batched == scalar, "\n".join(
        f"{'  ' if a == b else 'X '}{a}" + ("" if a == b else f"   <=>   {b}")
        for a, b in zip(batched, scalar))
    print(f"[benchmark] {len(batched)} result lines identical batched vs scalar -- OK")


def test_coupled_makespan_ab():
    """2 agents, a shared ordering edge, a 2-branch analytic-IK-style node on
    each arm -- the makespan-optimal branch on one arm depends on when the
    other arm's edge lets it start. Non-full (literal-target branches)."""
    import jax.numpy as jnp
    from pydrake.math import eq
    from goc_mpc import GraphOfConstraints
    from goc_mpc._ext.configuration_spline import Block
    from goc_mpc.evolutionary_waypoint_solver.projection import ProjOperator
    from goc_mpc.evolutionary_waypoint_solver.spec import build_graph_ordering_problem
    from goc_mpc.logic_based_benders_solver.structure import (
        node_instances, node_candidates, warm_start_wp)

    DIM = 2

    def euclid(a, b):
        return jnp.sqrt(jnp.sum((jnp.asarray(b) - jnp.asarray(a)) ** 2))

    g = GraphOfConstraints([[Block.R(DIM)], [Block.R(DIM)]], [],
                           state_lower_bound=-50.0, state_upper_bound=50.0,
                           robot_names=["r0", "r1"])
    n = g.structure.add_nodes(3)
    g.structure.add_edge(n[0], n[1], True)        # r0: 0 -> 1
    g.structure.add_edge(n[1], n[2], True)        # cross: 1 (r0) -> 2 (r1)
    g.add_constraint(n[0], eq(g.agent_q(0), np.zeros(DIM)),
                     proj=ProjOperator(pins=g.agent_q(0), reads=(), continuous_params=0,
                                       discrete_params=2,
                                       func=lambda psi, b: (jnp.array([[3.0, 0.0], [0.0, 3.0]])[b])))
    g.add_constraint(n[1], eq(g.agent_q(0), np.zeros(DIM)),
                     proj=ProjOperator(pins=g.agent_q(0), reads=(), continuous_params=0,
                                       discrete_params=1, func=lambda psi, b: np.array([6.0, 0.0])))
    g.add_constraint(n[2], eq(g.agent_q(1), np.zeros(DIM)),
                     proj=ProjOperator(pins=g.agent_q(1), reads=(), continuous_params=0,
                                       discrete_params=1, func=lambda psi, b: np.array([10.0, 0.0])))
    problem = build_graph_ordering_problem(g, np.zeros((2, DIM)), wp_bounds=(-50.0, 50.0),
                                           objective="makespan", edge_cost_fn=euclid)
    params = np.asarray(problem.params)
    wp = warm_start_wp(problem, np.zeros((2, DIM)))
    inst = node_instances(problem)
    cands = node_candidates(problem, wp, params, allow_unresolved=True)
    x0 = {0: np.zeros(problem.state_dim), 1: np.zeros(problem.state_dim)}

    def solve():
        return dm.solve_dp_master(problem, cands, wp, x0, inst,
                                  ordering_edges=problem.ordering_edges,
                                  edge_cost_fn=euclid, objective="makespan",
                                  x0_full=np.zeros(problem.state_dim))

    dm._DP_MASTER_BATCH_NONFULL = True
    a = solve()
    dm._DP_MASTER_BATCH_NONFULL = False
    b = solve()
    dm._DP_MASTER_BATCH_NONFULL = True
    for k in ("status", "objective", "branch", "assignment", "time", "routes"):
        assert a[k] == b[k], (k, a[k], b[k])
    assert a["status"] == "OPTIMAL"
    print(f"[coupled] makespan obj={a['objective']:.4f} branch={a['branch']} "
          "identical batched vs scalar -- OK")


if __name__ == "__main__":
    test_benchmark_ab()
    test_coupled_makespan_ab()
    test_full_resolve_ab()
    test_full_resolve_jax_scoring_ab()
    print("\nAll checks passed.")

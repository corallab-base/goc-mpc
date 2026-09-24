"""`make_dp_master_jax(route_backend="coupled")` must agree, exactly, with
the enumerate backend and with `dp_master.solve_dp_master`.

Both are exact solvers for the same makespan problem -- the coupled one just
stops enumerating linear extensions and branch combinations -- so any
disagreement in the objective is a bug in one of them.  Reuses the scenes
`test_dp_master_jax.py` already defines.

Run: python examples/test_coupled_backend.py
"""

import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from test_dp_master_jax import EU, _AnchorShim, scene_A, scene_B, scene_C

from goc_mpc.logic_based_benders_solver.dp_master import solve_dp_master
from goc_mpc.logic_based_benders_solver.dp_master_jax import make_dp_master_jax
from goc_mpc.logic_based_benders_solver.structure import (
    node_candidates, node_instances, warm_start_wp)

FAILS = 0


def _fail(msg):
    global FAILS
    FAILS += 1
    print(f"[FAIL] {msg}")


def check(name, scene, anchor=None):
    problem, x0_full, x0_rows = scene()
    params = np.asarray(problem.params)
    wp = warm_start_wp(problem, x0_full)
    inst = node_instances(problem)
    na = None if anchor is None else anchor[0]
    kw = dict(node_active=na,
              var_committed=None if anchor is None else anchor[1],
              var_anchor=None if anchor is None else anchor[2])

    cands = node_candidates(problem, wp, params, allow_unresolved=True)
    ref = solve_dp_master(problem, cands, wp, x0_rows, inst,
                          ordering_edges=problem.ordering_edges, edge_cost_fn=EU,
                          objective="makespan", x0_full=x0_full,
                          anchor=None if anchor is None else _AnchorShim(*anchor))

    jm_e = make_dp_master_jax(problem, objective="makespan", edge_cost_fn=EU)
    jm_c = make_dp_master_jax(problem, objective="makespan", edge_cost_fn=EU,
                              route_backend="coupled")

    ve = jm_e.run_vec(params, wp, x0_full, x0_rows, **kw)
    vc = jm_c.run_vec(params, wp, x0_full, x0_rows, **kw)

    for tag, got in (("enumerate", ve), ("coupled", vc)):
        if got["status"] != ref["status"]:
            _fail(f"{name}: {tag} status {got['status']} != ref {ref['status']}")
        o_ref, o_got = ref["objective"] or 0.0, got["objective"] or 0.0
        if abs(o_ref - o_got) > 1e-9:
            _fail(f"{name}: {tag} objective {o_got} != ref {o_ref}")

    # the coupled backend's own reconstruction must be self-consistent: the
    # times it reports have to reproduce the objective it claims.
    if vc["status"] == "OPTIMAL" and vc["time"]:
        if abs(max(vc["time"].values()) - vc["objective"]) > 1e-9:
            _fail(f"{name}: coupled objective {vc['objective']} != max time "
                  f"{max(vc['time'].values())}")

    # `t_k` (the node-rank genome the GA is seeded with) is now scattered
    # from the winning order rather than read out of `POS`. For the enumerate
    # backend those must be identical -- this pins the rewrite.
    import jax.numpy as jnp
    X0 = jnp.stack([jnp.asarray(x0_rows[j], float) for j in sorted(x0_rows)])
    na_j = jnp.ones(problem.n_nodes, bool) if na is None else jnp.asarray(na)
    vc_j = (jnp.zeros(problem.n_variables, bool) if kw["var_committed"] is None
            else jnp.asarray(kw["var_committed"]))
    va_j = (jnp.zeros(problem.n_variables, jnp.int32) if kw["var_anchor"] is None
            else jnp.asarray(kw["var_anchor"], jnp.int32))
    g_args = (jnp.asarray(params), jnp.asarray(wp), jnp.asarray(x0_full), X0,
              na_j, vc_j, va_j)
    _o, _a, _c, t_e, _p, _w, cell_e = jm_e.skeleton_grid_fn(3)(*g_args)
    pos_ref = np.asarray(jm_e.POS)[np.asarray(cell_e[:, 2])]
    if not np.allclose(np.asarray(t_e), pos_ref):
        _fail(f"{name}: enumerate t_k != POS[e_sel]")
    # and for the coupled backend `t_k` must rank exactly the order it stored
    # in `cell`, and that order must respect every resolved precedence edge.
    _o, _a, _c, t_c, _p, _w, cell_c = jm_c.skeleton_grid_fn(3)(*g_args)
    t_c, cell_c = np.asarray(t_c), np.asarray(cell_c)
    for i in range(t_c.shape[0]):
        if not np.allclose(t_c[i][cell_c[i, 3:]], np.arange(problem.n_nodes)):
            _fail(f"{name}: coupled t_k does not rank its own stored order")

    te = [d["objective"] for d in jm_e.run_topk(5, params, wp, x0_full, x0_rows, **kw)]
    tc = [d["objective"] for d in jm_c.run_topk(5, params, wp, x0_full, x0_rows, **kw)]
    if len(te) != len(tc) or any(abs(a - b) > 1e-9 for a, b in zip(te, tc)):
        _fail(f"{name}: run_topk objectives differ\n    enumerate={te}\n    coupled  ={tc}")

    print(f"[ok ] {name}: obj={vc['objective']:.6f} (ref {ref['objective']:.6f}), "
          f"top-{len(tc)} identical, structures cached={len(jm_c._coupled_cache)}, "
          f"front={jm_c.coupled_front} exact={jm_c.coupled_exact}, "
          f"orders enumerated: {jm_e.E} -> {jm_c.E}")
    return jm_e, jm_c, (params, wp, x0_full, x0_rows, kw)


def main():
    check("scene_B (1 agent, 3 nodes, conditional edge)", scene_B)
    check("scene_C (2 agents, cross-agent edge, branch)", scene_C)

    # an anchor changes node_active -> a DIFFERENT coupled structure, built
    # and cached on demand without retracing anything.
    na = np.array([False, True, True])
    check("scene_C + anchor (node 0 passed)", scene_C,
          anchor=(na, np.zeros(0, bool), np.zeros(0, int)))

    # scene_A has a DYNAMIC (var_agent_q) multi-branch projection: its rows
    # depend on the ASSIGNMENT (a grid axis `_score_grid_core` already
    # resolves per cell), not on the visiting order.  So the coupled backend
    # must accept it -- and must then agree with enumerate + solve_dp_master,
    # which is the evidence that narrowing the fence was sound.
    check("scene_A (dynamic var_agent_q branch, NC=2)", scene_A)
    check("scene_A + anchor (node 0 passed)", scene_A,
          anchor=(np.array([False, True]), np.array([False]), np.array([0])))
    for committed_to in (0, 1):
        check(f"scene_A + anchor (slot committed to agent {committed_to})",
              scene_A, anchor=(np.array([False, True]), np.array([True]),
                               np.array([committed_to])))

    problem, _x0, _rows = scene_A()
    try:
        make_dp_master_jax(problem, objective="avg", edge_cost_fn=EU,
                           route_backend="coupled")
        _fail("scene_A: coupled backend accepted a non-makespan objective")
    except NotImplementedError as exc:
        print(f"[ok ] non-makespan objective refused: {str(exc)[:70]}...")
    # the order-dependence fence itself is tested in
    # examples/test_projection_dependencies.py

    # per-cycle cost: the structure cache must make repeat solves cheap.
    _e, jm_c, (params, wp, x0_full, x0_rows, kw) = check(
        "scene_C (timing warm-up)", scene_C)
    t0 = time.perf_counter()
    for _ in range(5):
        jm_c.run_vec(params, wp, x0_full, x0_rows, **kw)
    dt = (time.perf_counter() - t0) / 5
    print(f"\ncoupled backend steady-state run_vec: {dt * 1e3:.1f} ms/cycle "
          f"({len(jm_c._coupled_cache)} cached structures, 0 rebuilds)")

    print(f"\nFAILURES: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())

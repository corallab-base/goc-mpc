"""`route_backend="lazy"` (logic_based_benders_solver.lazy_dp) against
references it does not share code with.

  1. Scenes A/B/C (+ anchors) from test_dp_master_jax.py: lazy == enumerate
     == solve_dp_master on makespan.
  2. Scene O -- ORDER-dependent rows: robot 1's IK target is an object whose
     depot pin is gated on whether robot 0's hold has started.  The coupled
     backend must refuse; lazy must match enumerate (which re-resolves per
     ordering) and brute force.
  3. Scene R -- UPSTREAM-BRANCH-dependent rows: robot 0's 2-branch pick sets,
     through the rigid-carry pin, where the object ends up; robot 1's IK then
     targets it.  Brute force is the only trustworthy reference; the
     enumerate backend's answer is printed for information.

The brute-force oracle: every linear extension x every branch combination,
each resolved with ALL projections under its true rank, scored by a plain
numpy forward pass.  It shares nothing with the DP but apply_projections and
the ownership / precedence helpers.

Run: python examples/test_lazy_dp.py
"""

import itertools
import sys
import time

import numpy as np
import jax.numpy as jnp
from pydrake.math import eq

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from test_dp_master_jax import EU, _AnchorShim, scene_A, scene_B, scene_C

from goc_mpc import GraphOfConstraints
from goc_mpc._ext.configuration_spline import Block
from goc_mpc.evolutionary_waypoint_solver.projection import ProjOperator
from goc_mpc.evolutionary_waypoint_solver.spec import build_graph_ordering_problem
from goc_mpc.logic_based_benders_solver.dp_master import (
    _agent_owned_nodes, _resolve_precedence, solve_dp_master)
from goc_mpc.logic_based_benders_solver.dp_master_jax import make_dp_master_jax
from goc_mpc.logic_based_benders_solver.lazy_dp import _resolver
from goc_mpc.logic_based_benders_solver.structure import (
    node_candidates, node_instances, warm_start_wp)

DIM = 2
FAILS = 0


def _fail(msg):
    global FAILS
    FAILS += 1
    print(f"[FAIL] {msg}")


def grid(jm, problem, wpt, x0, na=None, vc=None, va=None, k=4):
    X0 = jnp.broadcast_to(jnp.asarray(x0)[None], (problem.n_agents, problem.state_dim))
    na = jnp.ones(problem.n_nodes, bool) if na is None else jnp.asarray(na)
    vc = jnp.zeros(problem.n_variables, bool) if vc is None else jnp.asarray(vc)
    va = jnp.zeros(problem.n_variables, jnp.int32) if va is None else jnp.asarray(va, jnp.int32)
    out = jm._skeleton_grid(jnp.asarray(problem.params), jnp.asarray(wpt), jnp.asarray(x0),
                            X0, na, vc, va, k)
    return [np.asarray(o) for o in out]


# ------------------------------------------------------------ brute force
def brute_force(problem, wpt, x0):
    """min makespan over every order x every branch combo (no variables, no
    aux, every node active)."""
    N, J, D = problem.n_nodes, problem.n_agents, problem.dim
    ov = np.zeros(problem.n_variables, int)
    owned = _agent_owned_nodes(problem, node_instances(problem), ov)
    owners = [[j for j in range(J) if v in owned.get(j, ())] for v in range(N)]
    P = _resolve_precedence(problem, list(problem.ordering_edges), ov, np.zeros(0))
    preds = [[u for (u, w) in P if w == v] for v in range(N)]
    exts = [perm for perm in itertools.permutations(range(N))
            if all(perm.index(u) < perm.index(v) for (u, v) in P)]
    ents = list(problem.projections)
    br = [i for i, e in enumerate(ents) if e.discrete_params > 1]
    combos = list(itertools.product(*[range(ents[i].discrete_params) for i in br]))
    RK, PB, meta = [], [], []
    for ext in exts:
        rank = np.zeros(N, np.int32)
        for pos, v in enumerate(ext):
            rank[v] = pos
        for cb in combos:
            pb = np.zeros(problem.n_branch)
            for i, e in enumerate(ents):
                b = cb[br.index(i)] if i in br else 0
                pb[e.branch_slice.start + b] = 1.0
            RK.append(rank)
            PB.append(pb)
            meta.append((ext, cb))
    M = len(meta)
    out = np.asarray(_resolver(problem, tuple(range(len(ents))))(
        jnp.asarray(np.repeat(np.asarray(wpt)[None], M, 0)), jnp.asarray(np.stack(PB)),
        jnp.asarray(problem.params), jnp.zeros((M, problem.n_variables, J)),
        jnp.zeros((M, problem.n_cond_vars)), jnp.asarray(np.stack(RK)),
        jnp.ones(N, bool), jnp.asarray(x0)))
    best = (np.inf, None)
    for m, (ext, cb) in enumerate(meta):
        wp = out[m]
        t, last, mk = {}, {}, 0.0
        for v in ext:
            tv = max([0.0] + [t[u] for u in preds[v]])
            for j in owners[v]:
                sl = slice(j * D, (j + 1) * D)
                a = x0[sl] if j not in last else wp[last[j], sl]
                tv = max(tv, (t[last[j]] if j in last else 0.0)
                         + float(np.linalg.norm(wp[v, sl] - a)))
                last[j] = v
            t[v] = tv
            mk = max(mk, tv)
        if mk < best[0]:
            best = (mk, (ext, cb))
    return best


# ---------------------------------------------------------------- scenes
GRASP = np.array([0.3, 0.0])


def _two_robots_one_object():
    return GraphOfConstraints([[Block.R(DIM)], [Block.R(DIM)]], [[Block.R(DIM)]],
                              state_lower_bound=-20.0, state_upper_bound=20.0,
                              robot_names=["r0", "r1"], object_names=["o0"])


def scene_O():
    """r0: n0 pick -> n1 place (hold n0->n1, explicit place pin).  r1: n2 goes
    to the object, UNORDERED w.r.t. the hold, with a 2-branch approach.  The
    depot pin at source node n2 is gated on the hold not having started, so
    r1's target is the depot pose only if n2 is ranked before n0."""
    g = _two_robots_one_object()
    n0, n1, n2 = g.structure.add_nodes(3)
    g.structure.add_edge(n0, n1, True)
    place = np.array([-4.0, 3.0])
    offs = jnp.array([[0.4, 0.0], [-0.4, 0.0]])
    g.add_constraint(n0, eq(g.agent_q(0), g.object_q(0) + GRASP),
                     proj=ProjOperator(pins=g.agent_q(0), reads=(g.object_q(0),),
                                       func=lambda o, psi, b: o + GRASP))
    g.add_constraint(n1, eq(g.agent_q(0), place + GRASP),
                     proj=ProjOperator(pins=g.agent_q(0), reads=(),
                                       func=lambda psi, b: place + GRASP))
    g.add_constraint(n1, eq(g.object_q(0), place),
                     proj=ProjOperator(pins=g.object_q(0), reads=(),
                                       func=lambda psi, b: place))
    g.add_constraint(n2, eq(g.agent_q(1), g.object_q(0) + GRASP),
                     proj=ProjOperator(pins=g.agent_q(1), reads=(g.object_q(0),),
                                       continuous_params=0, discrete_params=2,
                                       func=lambda o, psi, b: o + offs[b]))
    g.add_hold(n0, n1, 0, [0])
    x0 = np.concatenate([[1.5, 0.0], [2.5, 0.5], [2.0, 0.0]])     # r0, r1, object
    problem = build_graph_ordering_problem(g, x0[:4].reshape(2, DIM), (-20.0, 20.0),
                                           objective="makespan", edge_cost_fn=EU)
    wpt = np.asarray(warm_start_wp(problem, x0)).copy()
    wpt[:, 2 * DIM:] = [8.0, -8.0]   # an ungated object column is far away:
    return problem, wpt, x0          # the order decision really matters


def scene_R():
    """r0: n0 pick (2-branch grasp) -> n1 place (literal EE, NO explicit object
    pin -> the rigid-carry pin sets obj@n1 = obj@n0 + (q1 - q0), which depends
    on r0's branch at n0).  r1: n2, after n1, a 2-branch IK on the object."""
    g = _two_robots_one_object()
    n0, n1, n2 = g.structure.add_nodes(3)
    g.structure.add_edge(n0, n1, True)
    g.structure.add_edge(n1, n2, True)
    place_ee = np.array([-3.0, 2.0])
    grasp = jnp.array([[0.6, 0.0], [-0.6, 0.0]])
    offs = jnp.array([[0.0, 0.5], [0.0, -0.5]])
    g.add_constraint(n0, eq(g.agent_q(0), g.object_q(0) + GRASP),
                     proj=ProjOperator(pins=g.agent_q(0), reads=(g.object_q(0),),
                                       continuous_params=0, discrete_params=2,
                                       func=lambda o, psi, b: o + grasp[b]))
    g.add_constraint(n1, eq(g.agent_q(0), place_ee),
                     proj=ProjOperator(pins=g.agent_q(0), reads=(),
                                       func=lambda psi, b: place_ee))
    g.add_constraint(n2, eq(g.agent_q(1), g.object_q(0) + GRASP),
                     proj=ProjOperator(pins=g.agent_q(1), reads=(g.object_q(0),),
                                       continuous_params=0, discrete_params=2,
                                       func=lambda o, psi, b: o + offs[b]))
    g.add_hold(n0, n1, 0, [0])
    # r1 starts FAR away, so its leg to the object sets the makespan: r0's
    # grasp branch 0 is locally worse for r0 (t(n1) 7.6 vs 5.2) but leaves the
    # object 1.2 closer to r1 (makespan ~8.41 vs ~9.61).  Anything that picks
    # r0's branch by r0's own cost, or prices r1 against the wrong object
    # pose, gets this wrong.
    x0 = np.concatenate([[1.0, 0.0], [-12.0, 2.0], [2.0, 0.0]])
    problem = build_graph_ordering_problem(g, x0[:4].reshape(2, DIM), (-20.0, 20.0),
                                           objective="makespan", edge_cost_fn=EU)
    return problem, np.asarray(warm_start_wp(problem, x0)), x0


# ------------------------------------------------------------------ checks
def check_ab(name, scene, anchor=None):
    problem, x0_full, x0_rows = scene()
    params = np.asarray(problem.params)
    wp = warm_start_wp(problem, x0_full)
    na = None if anchor is None else anchor[0]
    kw = dict(node_active=na, var_committed=None if anchor is None else anchor[1],
              var_anchor=None if anchor is None else anchor[2])
    cands = node_candidates(problem, wp, params, allow_unresolved=True)
    ref = solve_dp_master(problem, cands, wp, x0_rows, node_instances(problem),
                          ordering_edges=problem.ordering_edges, edge_cost_fn=EU,
                          objective="makespan", x0_full=x0_full,
                          anchor=None if anchor is None else _AnchorShim(*anchor))
    jm_e = make_dp_master_jax(problem, objective="makespan", edge_cost_fn=EU)
    jm_l = make_dp_master_jax(problem, objective="makespan", edge_cost_fn=EU,
                              route_backend="lazy")
    ge = grid(jm_e, problem, wp, x0_full, na, kw["var_committed"], kw["var_anchor"])
    gl = grid(jm_l, problem, wp, x0_full, na, kw["var_committed"], kw["var_anchor"])
    fe, fl = np.isfinite(ge[0]), np.isfinite(gl[0])
    if not (np.array_equal(fe, fl) and np.allclose(ge[0][fe], gl[0][fl], atol=1e-9)):
        _fail(f"{name}: skeleton objectives differ\n  enumerate={ge[0]}\n  lazy     ={gl[0]}")
    if abs(float(gl[0][0]) - (ref["objective"] or 0.0)) > 1e-9:
        _fail(f"{name}: lazy best {gl[0][0]} != solve_dp_master {ref['objective']}")
    vl = jm_l.run_vec(params, wp, x0_full, x0_rows, **kw)
    if abs((vl["objective"] or 0.0) - (ref["objective"] or 0.0)) > 1e-9:
        _fail(f"{name}: lazy run_vec {vl['objective']} != ref {ref['objective']}")
    print(f"[ok ] {name}: lazy == enumerate == solve_dp_master "
          f"(best {float(gl[0][0]):.6f}), exact={jm_l.lazy_exact}")


def check_vs_brute(name, scene, expect_enumerate_valid, expect_coupled_refuses):
    problem, wpt, x0 = scene()
    bf, (ext, cb) = brute_force(problem, wpt, x0)
    t0 = time.perf_counter()
    jm_l = make_dp_master_jax(problem, objective="makespan", edge_cost_fn=EU,
                              route_backend="lazy")
    gl = grid(jm_l, problem, wpt, x0)
    t_l = time.perf_counter() - t0
    lazy = float(gl[0][0])
    if abs(lazy - bf) > 1e-7 * max(1.0, bf):
        _fail(f"{name}: lazy {lazy} != brute force {bf} (brute order {ext}, branches {cb})")
    lazy_order = tuple(int(v) for v in gl[6][0, 3:])
    lazy_pb = np.asarray(gl[4][0])
    brs = [int(np.argmax(lazy_pb[e.branch_slice])) for e in problem.projections
           if e.discrete_params > 1]
    note = ""
    try:
        make_dp_master_jax(problem, objective="makespan", edge_cost_fn=EU,
                           route_backend="coupled")
        if expect_coupled_refuses:
            _fail(f"{name}: coupled backend accepted order/branch-dependent rows")
        note += " coupled accepted;"
    except NotImplementedError:
        note += " coupled refused (as it must);"
    try:
        ge = grid(make_dp_master_jax(problem, objective="makespan", edge_cost_fn=EU),
                  problem, wpt, x0)
        enum = float(ge[0][0])
        agree = abs(enum - bf) <= 1e-7 * max(1.0, bf)
        if expect_enumerate_valid and not agree:
            _fail(f"{name}: enumerate {enum} != brute force {bf}")
        note += f" enumerate={enum:.6f} ({'matches' if agree else 'DIFFERS from'} brute force)"
    except NotImplementedError as exc:
        note += f" enumerate refused: {str(exc)[:60]}"
    st = jm_l.lazy_stats[0] if jm_l.lazy_stats else {}
    print(f"[ok ] {name}: lazy {lazy:.6f} == brute force {bf:.6f}, order {lazy_order} "
          f"(brute {ext}), branches {brs} (brute {list(cb)}), {st.get('contexts')} contexts, "
          f"{st.get('resolver_calls')} "
          f"resolver calls, {st.get('cost_evals')} cost evals, {t_l:.2f}s incl. compile;"
          f"{note}")
    return problem, wpt, x0, jm_l


def main():
    check_ab("scene_A (dynamic var_agent_q branch)", scene_A)
    check_ab("scene_A + anchor (node 0 passed)", scene_A,
             anchor=(np.array([False, True]), np.array([False]), np.array([0])))
    check_ab("scene_A + anchor (slot committed to agent 1)", scene_A,
             anchor=(np.array([False, True]), np.array([True]), np.array([1])))
    check_ab("scene_B (conditional edges)", scene_B)
    check_ab("scene_C (cross-agent edge, branch)", scene_C)
    check_ab("scene_C + anchor", scene_C,
             anchor=(np.array([False, True, True]), np.zeros(0, bool), np.zeros(0, int)))

    check_vs_brute("scene_O (order-dependent IK target)", scene_O,
                   expect_enumerate_valid=True, expect_coupled_refuses=True)
    problem, wpt, x0, jm_l = check_vs_brute(
        "scene_R (rigid carry -> IK, upstream branch)", scene_R,
        expect_enumerate_valid=False, expect_coupled_refuses=True)

    # warm re-solve: rows are per-solve, but plans and the value-keyed cost
    # cache persist -- a repeated cycle should re-evaluate no legs at all.
    grid(jm_l, problem, wpt, x0)
    before = jm_l._lazy.__dict__["_cost"]
    n_before = sum(len(c) for c in before)
    t0 = time.perf_counter()
    grid(jm_l, problem, wpt, x0)
    dt = time.perf_counter() - t0
    st = jm_l.lazy_stats[0]
    if st["cost_evals"] != 0:
        _fail(f"warm re-solve re-evaluated {st['cost_evals']} legs (cache holds {n_before})")
    print(f"[ok ] warm re-solve of scene_R: {dt * 1e3:.1f} ms, 0 legs re-evaluated "
          f"({n_before} cached)")

    print(f"\nFAILURES: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())

"""Validation for `proj=` (projection.ProjOperator) on GraphOfConstraints.
add_constraint -- the evolutionary waypoint solver's analytic-elimination
mechanism (spec.py's _resolve_projections/problem.apply_projections).

Four cases, in increasing generality:
  1. A fully literal (reads=(), continuous_params=0, discrete_params=1)
     elimination -- spec.py precomputes this into a plain lookup table at
     build time (no `func` call at all during solve). Checks the solved
     position matches the literal target EXACTLY (float precision, not a
     residual-convergence tolerance -- there is no residual for this
     constraint at all).
  2. A `reads`-based elimination -- the pinned agent reads another
     placeholder (an object's own column at the same node) live every
     batched pass, exercising the untabled apply_projections branch and the
     _read_array_fn/_resolve_read_component resolution chain end-to-end.
  3. A `reads`-based elimination off a param(id) placeholder (Graph
     OfConstraints.add_param/set_param) -- the runtime-editable-target
     pattern MujocoEnv.pin_ee_pose actually uses. Confirms a `set_param`
     between solves is reflected immediately, with no rebuild/retrace.
  4. A multi-branch (discrete_params=2) elimination with one "wrong"
     (expensive-to-route-to) and one "right" (cheap) candidate value, wired
     up with a real edge_cost_fn -- confirms the GA actually learns to
     select the cost-minimizing branch via `proj_branch`'s argmax decode,
     not just that a table gathers correctly in isolation.

Run directly: python examples/test_proj_operator.py
"""

import numpy as np
import jax.numpy as jnp
from pydrake.math import eq, ge

from goc_mpc import GraphOfConstraints
from goc_mpc._ext.configuration_spline import Block, CubicConfigurationSpline
from goc_mpc.evolutionary_waypoint_solver.projection import ProjOperator
from goc_mpc.evolutionary_waypoint_solver.spec import build_graph_ordering_problem
from goc_mpc.evolutionary_waypoint_solver.solver import run_lamarckian_al
from goc_mpc.evolutionary_waypoint_solver.problem import full_active_anchor

DIM = 2
TOL = 1e-6


def _euclidean(a, b):
    return jnp.linalg.norm(b - a)


def test_literal_table_elimination():
    target = np.array([1.75, -2.5])
    robot_spec = [Block.R(DIM)]
    graph = GraphOfConstraints([robot_spec], [], state_lower_bound=-20.0, state_upper_bound=20.0,
                               robot_names=["r0"])
    (n0,) = graph.structure.add_nodes(1)

    proj = ProjOperator(
        pins=graph.agent_q(0),
        reads=(),
        continuous_params=0,
        discrete_params=1,
        func=lambda psi, branch: target,
    )
    # formula only needs to cover the same free variables proj.pins/reads
    # do -- never compiled as a residual once proj is attached.
    graph.add_constraint(n0, eq(graph.agent_q(0), np.zeros(DIM)), proj=proj)

    x0 = np.zeros((1, DIM))
    problem = build_graph_ordering_problem(graph, x0, wp_bounds=(-20.0, 20.0))
    assert problem.n_eq_constr == 0 and problem.n_ieq_constr == 0, (
        "a projected constraint must register no residual at all")
    assert problem.n_branch == 1 and problem.n_psi == 0

    anchor = full_active_anchor(problem)
    result = run_lamarckian_al(problem, anchor, pop_size=10, n_gen=5, seed=1,
                                x0=jnp.asarray(x0.reshape(-1)))
    _assign, _cond_binary, _proj_branch, _t, wp, _psi = problem._extract_single(np.asarray(result.X))
    from goc_mpc.evolutionary_waypoint_solver.problem import apply_projections
    wp_proj = np.asarray(apply_projections(
        problem, jnp.asarray(wp)[None], jnp.asarray(_psi)[None], jnp.asarray(_proj_branch)[None],
        jnp.asarray(problem.params)))[0]

    solved = wp_proj[n0]
    assert np.allclose(solved, target, atol=1e-9), f"solved {solved} != literal target {target}"
    print(f"[literal] solved {solved}  (exact target {target}) -- OK")


def test_reads_based_elimination():
    # Two agents, no objects -- deliberately avoids _resolve_stationary_
    # objects' auto-generated "an unheld object must not move" invariant
    # can't legally reference agent_q(k) for TWO distinct agents in one
    # formula either (routing-instance discovery needs a single owner per
    # node phi -- see _resolve_phi_agent_source) -- an object column is the
    # framework's actual mechanism for "another entity's value, read at
    # this same node", so this test pins the object to `obj_target` both
    # ways (an ordinary residual AND -- since this single-node, no-hold
    # graph makes it a routing "source" -- _resolve_stationary_objects'
    # always-on depot invariant, matching x0's own object depot below) and
    # has the agent's projection read that column live.
    offset = np.array([0.3, -0.1])
    obj_target = np.array([2.0, 4.0])
    # Depot already at the true answer -- a projected pin creates a REAL
    # routing instance for its agent (pins must reference agent_q(k), which
    # necessarily satisfies _resolve_phi_agent_source), so this node's F
    # includes a genuine route cost from depot to the solved position. A far
    # depot pits that cost against CV in the combined selection score (an
    # AL/GA scheduling interaction pre-existing to this whole module, not
    # particular to projections -- confirmed by reproducing the identical
    # stuck-at-a-CV-tol-shielded-local-optimum symptom with an ordinary,
    # non-projected two-placeholder node constraint on this same toy single-
    # node graph shape); starting the depot at the answer removes that
    # tension so this test isolates the projection mechanism itself.
    agent_depot = obj_target + offset
    robot_spec = [Block.R(DIM)]
    object_spec = [Block.R(DIM)]
    graph = GraphOfConstraints([robot_spec], [object_spec], state_lower_bound=-20.0, state_upper_bound=20.0,
                               robot_names=["r0"], object_names=["obj0"])
    (n0,) = graph.structure.add_nodes(1)

    graph.add_constraint(n0, eq(graph.object_q(0), obj_target))

    proj = ProjOperator(
        pins=graph.agent_q(0),
        reads=(graph.object_q(0),),
        continuous_params=0,
        discrete_params=1,
        func=lambda obj_pos, psi, branch: obj_pos + offset,
    )
    graph.add_constraint(n0, eq(graph.agent_q(0), graph.object_q(0)), proj=proj)

    agents_width = graph.agent_col_offsets[-1]
    x0 = np.concatenate([agent_depot, obj_target])
    problem = build_graph_ordering_problem(graph, agent_depot.reshape(1, DIM), wp_bounds=(-20.0, 20.0))
    # Only the object's own eq constraint remains -- the agent's is projected.
    assert problem.n_eq_constr == DIM, problem.n_eq_constr

    anchor = full_active_anchor(problem)
    result = run_lamarckian_al(problem, anchor, pop_size=80, n_gen=200, seed=1,
                                outer_iters=3, inner_maxiter=30,
                                x0=jnp.asarray(x0))
    _assign, _cond_binary, proj_branch, _t, wp, psi = problem._extract_single(np.asarray(result.X))
    from goc_mpc.evolutionary_waypoint_solver.problem import apply_projections
    wp_proj = np.asarray(apply_projections(
        problem, jnp.asarray(wp)[None], jnp.asarray(psi)[None], jnp.asarray(proj_branch)[None],
        jnp.asarray(problem.params)))[0]

    obj_solved = wp_proj[n0, agents_width:]
    agent_solved = wp_proj[n0, :agents_width]
    assert np.allclose(obj_solved, obj_target, atol=1e-3), obj_solved
    assert np.allclose(agent_solved, obj_solved + offset, atol=1e-9), (
        f"agent {agent_solved} != object {obj_solved} + offset {offset}")
    print(f"[reads]   object {obj_solved}  agent {agent_solved}  (== object + {offset}) -- OK")


def test_param_read_elimination():
    """`reads` may include a param(id) placeholder (GraphOfConstraints.
    add_param/set_param) -- the runtime-editable-target pattern every EE
    pose pin in the mujoco experiments actually uses (`MujocoEnv.
    pin_ee_pose`). Confirms the projection tracks a live `set_param` value
    (never tabled, since a param can change between solves) rather than
    baking in whatever it was at spec-build time."""
    target1 = np.array([1.2, -0.4])
    target2 = np.array([-0.6, 0.9])
    robot_spec = [Block.R(DIM)]
    graph = GraphOfConstraints([robot_spec], [], state_lower_bound=-20.0, state_upper_bound=20.0,
                               robot_names=["r0"])
    (n0,) = graph.structure.add_nodes(1)

    pid_x = graph.add_param(float(target1[0]))
    pid_y = graph.add_param(float(target1[1]))
    proj = ProjOperator(
        pins=graph.agent_q(0),
        reads=(graph.param(pid_x), graph.param(pid_y)),
        continuous_params=0,
        discrete_params=1,
        func=lambda px, py, psi, branch: jnp.stack([px, py]),
    )
    graph.add_constraint(n0, eq(graph.agent_q(0), np.zeros(DIM)), proj=proj)

    problem = build_graph_ordering_problem(graph, target1.reshape(1, DIM), wp_bounds=(-20.0, 20.0))
    assert problem.n_eq_constr == 0 and problem.n_ieq_constr == 0

    from goc_mpc.evolutionary_waypoint_solver.problem import apply_projections

    def _solve_and_extract(x0):
        anchor = full_active_anchor(problem)
        result = run_lamarckian_al(problem, anchor, pop_size=10, n_gen=5, seed=1,
                                    x0=jnp.asarray(x0))
        _assign, _cond_binary, proj_branch, _t, wp, psi = problem._extract_single(np.asarray(result.X))
        wp_proj = np.asarray(apply_projections(
            problem, jnp.asarray(wp)[None], jnp.asarray(psi)[None], jnp.asarray(proj_branch)[None],
            jnp.asarray(graph.view_param_values())))[0]
        return wp_proj[n0]

    solved1 = _solve_and_extract(target1)
    assert np.allclose(solved1, target1, atol=1e-9), f"solved {solved1} != param target {target1}"
    print(f"[param]   solved {solved1}  (== initial param values {target1}) -- OK")

    # A runtime hook moving the target (set_param) -- no graph rebuild, no
    # retrace -- must be reflected immediately, same as an ordinary
    # param(id) residual would.
    graph.set_param(pid_x, float(target2[0]))
    graph.set_param(pid_y, float(target2[1]))
    solved2 = _solve_and_extract(target2)
    assert np.allclose(solved2, target2, atol=1e-9), f"solved {solved2} != updated param target {target2}"
    print(f"[param]   solved {solved2}  (== updated param values {target2}, via set_param) -- OK")


def test_multi_branch_ga_selects_cheap_branch():
    depot = np.array([0.0, 0.0])
    wrong_target = np.array([50.0, 50.0])   # far -- expensive to route to
    right_target = np.array([1.0, 0.0])     # close -- cheap
    robot_spec = [Block.R(DIM)]
    graph = GraphOfConstraints([robot_spec], [], state_lower_bound=-100.0, state_upper_bound=100.0,
                               robot_names=["r0"])
    n0, n1 = graph.structure.add_nodes(2)
    graph.structure.add_edge(n0, n1, True)
    graph.add_constraint(n0, eq(graph.agent_q(0), depot))

    proj = ProjOperator(
        pins=graph.agent_q(0),
        reads=(),
        continuous_params=0,
        discrete_params=2,
        func=lambda psi, branch: jnp.where(branch == 0, jnp.asarray(wrong_target), jnp.asarray(right_target)),
    )
    graph.add_constraint(n1, eq(graph.agent_q(0), np.zeros(DIM)), proj=proj)

    x0 = depot.reshape(1, DIM)
    problem = build_graph_ordering_problem(graph, x0, wp_bounds=(-100.0, 100.0), edge_cost_fn=_euclidean)
    assert problem.n_branch == 2

    anchor = full_active_anchor(problem)
    result = run_lamarckian_al(problem, anchor, pop_size=40, n_gen=60, seed=1,
                                x0=jnp.asarray(x0.reshape(-1)))
    _assign, _cond_binary, proj_branch, _t, wp, psi = problem._extract_single(np.asarray(result.X))
    from goc_mpc.evolutionary_waypoint_solver.problem import apply_projections
    wp_proj = np.asarray(apply_projections(
        problem, jnp.asarray(wp)[None], jnp.asarray(psi)[None], jnp.asarray(proj_branch)[None],
        jnp.asarray(problem.params)))[0]

    solved = wp_proj[n1]
    assert np.allclose(solved, right_target, atol=1e-6), (
        f"GA picked {solved}, expected the cheap branch {right_target}")
    print(f"[branch]  GA selected {solved}  (cheap target {right_target}, "
          f"rejected {wrong_target}) -- OK")


def test_chained_projection():
    """Projection B's `reads` pull a column projection A `pins` -- spec.py's
    _resolve_projections must apply A before B (apply_projections threads wp
    through its walk), even though B's constraint is added FIRST here. A pins
    the object to its depot; B (an analytic-IK-shaped elimination) then pins
    the agent to `object + offset`, reading the value A just wrote."""
    offset = np.array([0.25, -0.15])
    obj_depot = np.array([1.5, 3.0])
    agent_depot = obj_depot + offset
    graph = GraphOfConstraints([[Block.R(DIM)]], [[Block.R(DIM)]],
                               state_lower_bound=-20.0, state_upper_bound=20.0,
                               robot_names=["r0"], object_names=["obj0"])
    (n0,) = graph.structure.add_nodes(1)

    # B added FIRST -- discovery order is [B, A], the dependency sort flips it.
    proj_b = ProjOperator(
        pins=graph.agent_q(0),
        reads=(graph.object_q(0),),
        continuous_params=0, discrete_params=1,
        func=lambda obj_pos, psi, branch: obj_pos + offset,
    )
    graph.add_constraint(n0, eq(graph.agent_q(0), graph.object_q(0)), proj=proj_b)

    proj_a = ProjOperator(
        pins=graph.object_q(0),
        reads=(),
        continuous_params=0, discrete_params=1,
        func=lambda psi, branch: obj_depot,
    )
    graph.add_constraint(n0, eq(graph.object_q(0), np.zeros(DIM)), proj=proj_a)

    x0 = np.concatenate([agent_depot, obj_depot])
    problem = build_graph_ordering_problem(graph, agent_depot.reshape(1, DIM), wp_bounds=(-20.0, 20.0))
    assert problem.n_eq_constr == 0, problem.n_eq_constr
    # A (reads=(), continuous_params=0) is tabled; B has a real column read
    # so it isn't -- their order in problem.projections is the dependency one.
    assert [e.table is not None for e in problem.projections] == [True, False], (
        "expected dependency order [A (tabled), B (untabled)]")

    anchor = full_active_anchor(problem)
    result = run_lamarckian_al(problem, anchor, pop_size=20, n_gen=10, seed=1,
                                x0=jnp.asarray(x0))
    _assign, _cb, proj_branch, _t, wp, psi = problem._extract_single(np.asarray(result.X))
    from goc_mpc.evolutionary_waypoint_solver.problem import apply_projections
    wp_proj = np.asarray(apply_projections(
        problem, jnp.asarray(wp)[None], jnp.asarray(psi)[None],
        jnp.asarray(proj_branch)[None], jnp.asarray(problem.params)))[0]

    agents_width = graph.agent_col_offsets[-1]
    obj_solved = wp_proj[n0, agents_width:]
    agent_solved = wp_proj[n0, :agents_width]
    assert np.allclose(obj_solved, obj_depot, atol=1e-9), obj_solved
    assert np.allclose(agent_solved, obj_depot + offset, atol=1e-9), (
        f"agent {agent_solved} != object {obj_depot} + offset {offset} "
        "(B did not see A's pinned column)")
    print(f"[chain]   object {obj_solved}  agent {agent_solved}  "
          f"(== object + {offset}) -- OK")


def test_chained_projection_cycle():
    """Two projections that each `reads` the other's `pins` -- no valid
    single application order exists, so build must raise."""
    graph = GraphOfConstraints([[Block.R(DIM)]], [[Block.R(DIM)]],
                               state_lower_bound=-20.0, state_upper_bound=20.0,
                               robot_names=["r0"], object_names=["obj0"])
    (n0,) = graph.structure.add_nodes(1)
    proj_a = ProjOperator(pins=graph.agent_q(0), reads=(graph.object_q(0),),
                          continuous_params=0, discrete_params=1,
                          func=lambda o, psi, b: o)
    graph.add_constraint(n0, eq(graph.agent_q(0), graph.object_q(0)), proj=proj_a)
    proj_b = ProjOperator(pins=graph.object_q(0), reads=(graph.agent_q(0),),
                          continuous_params=0, discrete_params=1,
                          func=lambda a, psi, b: a)
    graph.add_constraint(n0, eq(graph.object_q(0), graph.agent_q(0)), proj=proj_b)
    try:
        build_graph_ordering_problem(graph, np.zeros((1, DIM)), wp_bounds=(-20.0, 20.0))
    except ValueError as e:
        assert "cycle" in str(e).lower(), e
        print(f"[cycle]   raised as expected -- OK")
    else:
        raise AssertionError("expected a ValueError for the projection dependency cycle")


def test_dynamic_var_agent_q_pin():
    """`proj.pins` may be a var_agent_q(var_id) row -- a DYNAMIC pin whose
    write TARGET (which agent's slot) is decided by the solved assignment,
    not spec-build time. Exercised here through BOTH solvers that honour
    `proj`:
      * the evolutionary solver -- apply_projections decodes the owner from
        `assign` and writes owner_cols_per_agent[owner];
      * solve_dp_master / DpMasterWaypointSolver -- _needs_full_resolve
        catches the agent-column dynamic entry, node_candidates defers it,
        and the per-schedule resolution re-applies it against each
        enumerated assignment.

    A hold (assignable) is present too, so the auto rigid-carry / depot
    gated OBJECT projections coexist with the dynamic AGENT pin (different
    columns, so no gate/dynamic clash)."""
    from goc_mpc.logic_based_benders_solver.structure import (
        node_instances, node_candidates, warm_start_wp)
    from goc_mpc.logic_based_benders_solver.dp_master import solve_dp_master, _needs_full_resolve
    from goc_mpc.logic_based_benders_solver import DpMasterWaypointSolver

    D = 3  # robot (x, y, yaw); object is (x, y)
    t0 = np.array([1.0, 0.0, 0.5])
    t1 = np.array([4.0, 1.0, -0.3])
    rspec = [Block.R(2), Block.Torus(1)]
    g = GraphOfConstraints([rspec, rspec], [[Block.R(2)]],
                           state_lower_bound=-20.0, state_upper_bound=20.0,
                           robot_names=["r0", "r1"], object_names=["obj0"], workspace_dim=2)
    n0, n1 = g.structure.add_nodes(2)
    g.structure.add_edge(n0, n1, True)
    var = g.add_variable()
    g.add_constraint(n0, eq(g.object_q(0), np.array([1.0, 0.0])))
    g.add_constraint(n1, eq(g.object_q(0), np.array([4.0, 1.0])))
    g.add_assignable_hold(n0, n1, var, [0])

    p0 = [g.add_param(float(v)) for v in t0]
    g.add_constraint(
        n0, eq(g.var_agent_q(var), np.array([g.param(i) for i in p0])),
        proj=ProjOperator(pins=g.var_agent_q(var), reads=tuple(g.param(i) for i in p0),
                          func=lambda a, b, c, psi, br: jnp.stack([a, b, c])))
    p1 = [g.add_param(float(v)) for v in t1]
    g.add_constraint(
        n1, eq(g.var_agent_q(var), np.array([g.param(i) for i in p1])),
        proj=ProjOperator(pins=g.var_agent_q(var), reads=tuple(g.param(i) for i in p1),
                          func=lambda a, b, c, psi, br: jnp.stack([a, b, c])))

    # r0 depot at t0, r1 depot far -> both solvers must assign var -> r0.
    x0_agents = np.concatenate([t0, np.array([50.0, 50.0, 0.0])])
    x0 = np.concatenate([x0_agents, np.array([1.0, 0.0])])
    problem = build_graph_ordering_problem(
        g, x0_agents.reshape(2, D), wp_bounds=(-20.0, 20.0), objective="avg",
        edge_cost_fn=_euclidean)
    slot = problem.var_id_to_slot[var]
    dyn = [e for e in problem.projections if e.owner_var_slot is not None]
    assert len(dyn) == 2 and all(e.owner_var_slot == slot for e in dyn), (
        [(e.write_node, e.owner_var_slot) for e in problem.projections])
    # Both var_agent_q pins are projected -> no AL residual for them; what's
    # left is the two object-eq constraints + the assignable hold's own
    # rigid-carry residual (2 + 2 + 2).
    assert problem.n_eq_constr == 6, problem.n_eq_constr

    anchor = full_active_anchor(problem)
    result = run_lamarckian_al(problem, anchor, pop_size=40, n_gen=60, seed=1,
                                x0=jnp.asarray(x0))
    assign, _cb, _pb, _t, wp, _psi = problem._extract_single(np.asarray(result.X))
    wp = np.asarray(wp)  # local_refine returns the already-projected wp
    owner = int(np.argmax(np.asarray(assign)[slot]))
    assert owner == 0, owner
    assert np.allclose(wp[n0, owner * D:owner * D + D], t0, atol=1e-6), wp[n0, :2 * D]
    assert np.allclose(wp[n1, owner * D:owner * D + D], t1, atol=1e-6), wp[n1, :2 * D]
    print(f"[dynamic] evo: var -> agent{owner}, owner@n0 {wp[n0, :D]} owner@n1 {wp[n1, :D]} -- OK")

    assert _needs_full_resolve(problem), "an agent-column dynamic pin must force the full-resolve path"
    params = np.asarray(g.view_param_values())
    x0_full = np.zeros(problem.state_dim)
    x0_full[:len(x0)] = x0
    wpt = warm_start_wp(problem, x0)
    cands = node_candidates(problem, wpt, params, allow_unresolved=True)
    inst = node_instances(problem)
    x0_rows = {0: np.pad(t0, (0, problem.state_dim - D)),
               1: np.pad(np.array([50.0, 50.0, 0.0]), (D, problem.state_dim - 2 * D))}
    ecf = lambda a, b: float(_euclidean(a, b))
    for obj in ("avg", "minmax", "makespan"):
        r = solve_dp_master(problem, cands, wpt, x0_rows, inst,
                            ordering_edges=problem.ordering_edges, edge_cost_fn=ecf,
                            objective=obj, x0_full=x0_full)
        assert r["status"] == "OPTIMAL" and r["assignment"][slot] == 0, (obj, r["status"], r["assignment"])

    splines = [CubicConfigurationSpline(s) for s in g._robot_specs]
    dp = DpMasterWaypointSolver(g, splines, objective="avg", edge_cost_fn=ecf,
                                wp_bounds=(-20.0, 20.0))
    assert dp.solve(list(range(problem.n_nodes)), x0_full)
    W = dp.view_waypoints()
    owner = int(dp.view_var_assignments()[slot])
    assert owner == 0, owner
    assert np.allclose(W[n0, owner * D:owner * D + D], t0, atol=1e-5), W[n0, :2 * D]
    assert np.allclose(W[n1, owner * D:owner * D + D], t1, atol=1e-5), W[n1, :2 * D]
    print(f"[dynamic] dp_master: var -> agent{owner}, W resolves owner pin at n0/n1 -- OK")


def test_dynamic_multibranch_pin_dp_master():
    """A MULTI-branch dynamic pin (`discrete_params > 1` AND
    `pins=var_agent_q(...)`) through solve_dp_master / DpMasterWaypointSolver
    -- the analytic-IK shape (which arm AND which branch are both discrete
    choices). Inside `for A in assign_combos` the entry's owner is fixed, so
    it prices exactly like a static multi-branch entry (structure.entry_owner
    resolves the owner from the enumerated assignment; static_entry_owner's
    `min(pinned_cols)` would raise on the dynamic entry's empty pinned_cols).

    3 candidate positions at n0; only branch 1 sits at agent0's depot, so the
    cheap discrete solution is (var -> agent0, branch 1)."""
    from goc_mpc.logic_based_benders_solver.structure import (
        node_instances, node_candidates, warm_start_wp)
    from goc_mpc.logic_based_benders_solver.dp_master import solve_dp_master
    from goc_mpc.logic_based_benders_solver import DpMasterWaypointSolver

    D = 2
    cands_xy = np.array([[9.0, 9.0], [1.0, 0.0], [-8.0, 7.0]])  # branch 1 is the cheap one
    g = GraphOfConstraints([[Block.R(D)], [Block.R(D)]], [],
                           state_lower_bound=-20.0, state_upper_bound=20.0,
                           robot_names=["r0", "r1"])
    n0, n1 = g.structure.add_nodes(2)
    g.structure.add_edge(n0, n1, True)
    var = g.add_variable()
    g.add_constraint(
        n0, ge(g.var_agent_q(var), np.array([-20.0, -20.0])),
        proj=ProjOperator(
            pins=g.var_agent_q(var), reads=(), continuous_params=0, discrete_params=3,
            func=lambda psi, branch: jnp.asarray(cands_xy)[branch]))
    g.add_constraint(n1, eq(g.var_agent_q(var), np.array([2.0, 2.0])),
                     proj=ProjOperator(pins=g.var_agent_q(var), reads=(),
                                       func=lambda psi, b: np.array([2.0, 2.0])))

    x0_agents = np.concatenate([cands_xy[1], np.array([40.0, 40.0])])  # r0 at branch-1 pos
    x0 = x0_agents
    problem = build_graph_ordering_problem(
        g, x0_agents.reshape(2, D), wp_bounds=(-20.0, 20.0), objective="avg",
        edge_cost_fn=_euclidean)
    slot = problem.var_id_to_slot[var]
    mb = [e for e in problem.projections if e.owner_var_slot is not None and e.discrete_params > 1]
    assert len(mb) == 1 and mb[0].discrete_params == 3

    params = np.asarray(g.view_param_values())
    x0_full = np.zeros(problem.state_dim)
    x0_full[:len(x0)] = x0
    wpt = warm_start_wp(problem, x0)                       # no crash on the tabled dynamic entry
    cands = node_candidates(problem, wpt, params, allow_unresolved=True)
    inst = node_instances(problem)
    x0_rows = {0: np.pad(cands_xy[1], (0, problem.state_dim - D)),
               1: np.pad(np.array([40.0, 40.0]), (D, problem.state_dim - 2 * D))}
    ecf = lambda a, b: float(_euclidean(a, b))
    for obj in ("avg", "minmax", "makespan"):
        r = solve_dp_master(problem, cands, wpt, x0_rows, inst,
                            ordering_edges=problem.ordering_edges, edge_cost_fn=ecf,
                            objective=obj, x0_full=x0_full)
        assert r["status"] == "OPTIMAL", (obj, r["status"])
        assert r["assignment"][slot] == 0, (obj, r["assignment"])
        assert r["branch"].get((n0, 0)) == 1, (obj, r["branch"])

    splines = [CubicConfigurationSpline(s) for s in g._robot_specs]
    dp = DpMasterWaypointSolver(g, splines, objective="avg", edge_cost_fn=ecf,
                                wp_bounds=(-20.0, 20.0))
    assert dp.solve(list(range(problem.n_nodes)), x0_full)
    W = dp.view_waypoints()
    owner = int(dp.view_var_assignments()[slot])
    assert owner == 0, owner
    assert np.allclose(W[n0, owner * D:owner * D + D], cands_xy[1], atol=1e-5), W[n0, :2 * D]
    print(f"[dyn-mb] dp_master picked (var -> agent{owner}, branch 1), "
          f"W@n0 {W[n0, owner * D:owner * D + D]} == {cands_xy[1]} -- OK")


def test_owner_aware_dynamic_pin():
    """`ProjOperator(owner_aware=True)` -- a DYNAMIC pin whose elimination
    itself differs per agent (the two-arms-different-bases -> different
    analytic-IK case). `func` gets a trailing `owner` arg; here it just
    indexes a per-agent target, so a broken owner thread (always agent 0)
    would land agent 1's row on agent 0's target and blow the route cost.
    Checked through the evolutionary solver and DpMasterWaypointSolver."""
    from goc_mpc.logic_based_benders_solver import DpMasterWaypointSolver

    D = 2
    per_agent = jnp.asarray([[1.0, 0.0], [-1.0, 0.0]])  # agent 0's vs agent 1's n0 target
    g = GraphOfConstraints([[Block.R(D)], [Block.R(D)]], [],
                           state_lower_bound=-20.0, state_upper_bound=20.0,
                           robot_names=["r0", "r1"])
    n0, n1 = g.structure.add_nodes(2)
    g.structure.add_edge(n0, n1, True)
    var = g.add_variable()
    g.add_constraint(
        n0, ge(g.var_agent_q(var), np.array([-20.0, -20.0])),
        proj=ProjOperator(pins=g.var_agent_q(var), reads=(), continuous_params=0,
                          discrete_params=1, owner_aware=True,
                          func=lambda psi, branch, owner: per_agent[owner]))
    g.add_constraint(n1, eq(g.var_agent_q(var), np.array([0.0, 3.0])),
                     proj=ProjOperator(pins=g.var_agent_q(var), reads=(),
                                       func=lambda psi, b: np.array([0.0, 3.0])))

    # r0 far, r1 at its own n0 target -> both solvers must assign var -> r1,
    # and n0's row must come out as agent 1's target, not agent 0's.
    x0_agents = np.concatenate([np.array([30.0, 30.0]), np.asarray(per_agent[1])])
    problem = build_graph_ordering_problem(
        g, x0_agents.reshape(2, D), wp_bounds=(-20.0, 20.0), objective="avg",
        edge_cost_fn=_euclidean)
    slot = problem.var_id_to_slot[var]
    e0 = next(e for e in problem.projections if e.write_node == n0)
    assert e0.owner_aware and e0.table is None and e0.is_static is False

    anchor = full_active_anchor(problem)
    result = run_lamarckian_al(problem, anchor, pop_size=30, n_gen=40, seed=1,
                                x0=jnp.asarray(x0_agents))
    assign, _cb, _pb, _t, wp, _psi = problem._extract_single(np.asarray(result.X))
    wp, owner = np.asarray(wp), int(np.argmax(np.asarray(assign)[slot]))
    assert owner == 1, owner
    assert np.allclose(wp[n0, owner * D:owner * D + D], np.asarray(per_agent[1]), atol=1e-6), wp[n0]
    print(f"[owner]   evo: var -> agent{owner}, n0 row {wp[n0, owner*D:owner*D+D]} "
          f"== agent-{owner} target {np.asarray(per_agent[1])} -- OK")

    ecf = lambda a, b: float(_euclidean(a, b))
    splines = [CubicConfigurationSpline(s) for s in g._robot_specs]
    dp = DpMasterWaypointSolver(g, splines, objective="avg", edge_cost_fn=ecf,
                                wp_bounds=(-20.0, 20.0))
    x0_full = np.zeros(problem.state_dim)
    x0_full[:len(x0_agents)] = x0_agents
    assert dp.solve(list(range(problem.n_nodes)), x0_full)
    W, owner = dp.view_waypoints(), int(dp.view_var_assignments()[slot])
    assert owner == 1, owner
    assert np.allclose(W[n0, owner * D:owner * D + D], np.asarray(per_agent[1]), atol=1e-5), W[n0]
    print(f"[owner]   dp_master: var -> agent{owner}, W@n0 {W[n0, owner*D:owner*D+D]} "
          f"== agent-{owner} target -- OK")


def test_var_agent_link_fk_assignable():
    """`graph.var_agent_link_pos(var, link)` -- the assignable-arm FK
    placeholder (goc-mpc C++). A projected constraint pins var_agent_q(var)'s
    joint columns and carries an FK-equality Formula in terms of
    var_agent_link_pos(var, "ee"); the analytic-IK `func` (owner_aware) picks
    the per-arm inverse. Checks:
      * build passes (the Formula's var_agent_link_pos placeholder is
        "covered" by the var_agent_q pin -- spec.py's var_fk_slot);
      * evaluate_phi resolves the FK against the ASSIGNED arm (violation ~0
        for the right assignment, large for the wrong one);
      * evolutionary + dp_master land the assigned arm's joints on its own IK
        solution.
    """
    import jax
    from goc_mpc.logic_based_benders_solver import DpMasterWaypointSolver

    W = 2
    base1 = np.array([10.0, 0.0])     # agent 1's FK has an offset base; agent 0's is identity
    target = np.array([1.0, 2.0])
    ik = [target, target - base1]     # per-arm inverse of fk_a(q) = q (+ base1 for a==1)

    def fk0(q):
        return jnp.asarray(q)[:W], jnp.eye(W)

    def fk1(q):
        return jnp.asarray(q)[:W] + jnp.asarray(base1), jnp.eye(W)

    g = GraphOfConstraints([[Block.R(W)], [Block.R(W)]], [],
                           state_lower_bound=-40.0, state_upper_bound=40.0,
                           robot_names=["r0", "r1"], workspace_dim=W)
    g.set_robot_fk(0, "ee", fk0)
    g.set_robot_fk(1, "ee", fk1)
    (n0,) = g.structure.add_nodes(1)
    var = g.add_variable()
    g.add_constraint(
        n0, eq(g.var_agent_link_pos(var, "ee"), target),
        proj=ProjOperator(
            pins=g.var_agent_q(var), reads=(), continuous_params=0, discrete_params=1,
            owner_aware=True,
            func=lambda psi, branch, owner: jax.lax.switch(
                owner, [lambda: jnp.asarray(ik[0]), lambda: jnp.asarray(ik[1])])))

    # --- evaluate_phi against each assignment ---
    phi_id = g.node_to_phis_map[n0][0]
    x_wrong = np.concatenate([target, ik[1]])   # r0 joints = target, r1 joints = its IK
    assign_to0 = np.zeros(g.num_phis, dtype=int); assign_to0[phi_id] = 0
    assign_to1 = np.zeros(g.num_phis, dtype=int); assign_to1[phi_id] = 1
    v0 = g.evaluate_phi(phi_id, x_wrong, assign_to0, 1e-6)   # r0 joints=target -> fk0=target -> ok
    v1 = g.evaluate_phi(phi_id, x_wrong, assign_to1, 1e-6)   # r1 joints=ik1 -> fk1=target -> ok
    assert v0 and v1, (v0, v1)
    x_bad = np.concatenate([np.array([5.0, 5.0]), np.array([5.0, 5.0])])
    assert not g.evaluate_phi(phi_id, x_bad, assign_to0, 1e-6), "wrong joints must violate the FK equality"
    print("[vfk]     evaluate_phi resolves var_agent_link_pos against the assigned arm -- OK")

    # --- evolutionary + dp_master: r1's depot at its own IK solution, r0 far ---
    x_agents = np.concatenate([np.array([25.0, 25.0]), ik[1]])
    problem = build_graph_ordering_problem(
        g, x_agents.reshape(2, W), wp_bounds=(-40.0, 40.0), objective="avg",
        edge_cost_fn=_euclidean)
    slot = problem.var_id_to_slot[var]
    assert problem.n_eq_constr == 0, problem.n_eq_constr  # the FK equality is projected, not a residual

    anchor = full_active_anchor(problem)
    res = run_lamarckian_al(problem, anchor, pop_size=30, n_gen=40, seed=1,
                             x0=jnp.asarray(x_agents))
    a, _cb, _pb, _t, wp, _psi = problem._extract_single(np.asarray(res.X))
    wp, owner = np.asarray(wp), int(np.argmax(np.asarray(a)[slot]))
    assert owner == 1, owner
    assert np.allclose(wp[n0, owner * W:owner * W + W], ik[1], atol=1e-6), wp[n0]
    print(f"[vfk]     evo: var -> agent{owner}, joints {wp[n0, owner*W:owner*W+W]} == IK {ik[1]} -- OK")

    ecf = lambda p, q: float(_euclidean(p, q))
    splines = [CubicConfigurationSpline(s) for s in g._robot_specs]
    dp = DpMasterWaypointSolver(g, splines, objective="avg", edge_cost_fn=ecf,
                                wp_bounds=(-40.0, 40.0))
    x0_full = np.zeros(problem.state_dim); x0_full[:len(x_agents)] = x_agents
    assert dp.solve([n0], x0_full)
    Wp, owner = dp.view_waypoints(), int(dp.view_var_assignments()[slot])
    assert owner == 1, owner
    assert np.allclose(Wp[n0, owner * W:owner * W + W], ik[1], atol=1e-5), Wp[n0]
    print(f"[vfk]     dp_master: var -> agent{owner}, joints == IK {ik[1]} -- OK")


def test_chained_static_demotion():
    """A downstream projection whose `func` only truly consumes a param(id)
    (so _classify_static_projection would call it static) is demoted to
    non-static when it also `reads` a column an upstream NON-static
    projection pins -- precompute_static_projections can't reproduce that
    upstream value, so the downstream one must recompute each pass."""
    graph = GraphOfConstraints([[Block.R(DIM)], [Block.R(DIM)]], [[Block.R(DIM)]],
                               state_lower_bound=-20.0, state_upper_bound=20.0,
                               robot_names=["r0", "r1"], object_names=["obj0"])
    (n0,) = graph.structure.add_nodes(1)
    pid = graph.add_param(0.7)

    # upstream: non-static (reads agent 1's real column, which nothing pins),
    # pins the object.
    up = ProjOperator(pins=graph.object_q(0), reads=(graph.agent_q(1),),
                      continuous_params=0, discrete_params=1,
                      func=lambda a1, psi, b: a1)
    graph.add_constraint(n0, eq(graph.object_q(0), graph.agent_q(1)), proj=up)

    # downstream: func consumes only the param; the object read is
    # documented-unused (_obj), but it IS a read of the column `up` pins.
    down = ProjOperator(pins=graph.agent_q(0),
                        reads=(graph.param(pid), graph.object_q(0)),
                        continuous_params=0, discrete_params=1,
                        func=lambda pv, _obj, psi, b: jnp.stack([pv, pv]))
    graph.add_constraint(n0, eq(graph.agent_q(0), graph.object_q(0)), proj=down)

    problem = build_graph_ordering_problem(graph, np.zeros((2, DIM)), wp_bounds=(-20.0, 20.0))
    up_e, down_e = problem.projections  # dependency order: up first
    assert up_e.write_node == n0 and up_e.is_static is False
    assert down_e.is_static is False, (
        "downstream projection should be demoted to non-static (reads an "
        "upstream non-static projection's pinned column)")
    print("[demote]  downstream projection demoted to non-static -- OK")


if __name__ == "__main__":
    test_literal_table_elimination()
    test_reads_based_elimination()
    test_param_read_elimination()
    test_multi_branch_ga_selects_cheap_branch()
    test_dynamic_var_agent_q_pin()
    test_dynamic_multibranch_pin_dp_master()
    test_owner_aware_dynamic_pin()
    test_var_agent_link_fk_assignable()
    test_chained_projection()
    test_chained_projection_cycle()
    test_chained_static_demotion()
    print("\nAll checks passed.")

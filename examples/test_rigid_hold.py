"""`add_hold(..., rigid=True)`: the held point keeps its offset in the
END-EFFECTOR's own frame, so a carry survives the holder turning between
pick-up and release -- unlike the default translation-only hold (MILP's
linear Constraint 14a), which fixes the offset in world axes.

A pos+yaw robot picks a point up 1 m in front of it at the origin and puts
it down at (2, 2) after turning 90 degrees:
  * rigid=True  is satisfied by the point at (2, 3)  (1 m in front again),
  * rigid=False is satisfied by the point at (3, 2)  (same world offset).
Checked on both compiled forms of the relation: the residual (what the
solver drives to zero) and the auto-derived carry projection (what it pins
the released object's column to). MILP must refuse a rigid hold.

Run: python examples/test_rigid_hold.py
"""

import numpy as np
import jax.numpy as jnp

from goc_mpc import GraphOfConstraints, MILPWaypointMPC
from goc_mpc._ext.configuration_spline import Block, CubicConfigurationSpline
from goc_mpc.evolutionary_waypoint_solver.problem import apply_projections, full_active_anchor
from goc_mpc.evolutionary_waypoint_solver.spec import build_graph_ordering_problem

ROBOT_SPEC = [Block.R(2), Block.Torus(1)]   # (x, y) + yaw -> the built-in pos_yaw FK
AGENT_W, OBJ_W = 3, 2
ROT_CARRY = np.array([2.0, 3.0])            # holder turned 90 deg: still 1 m ahead
WORLD_CARRY = np.array([3.0, 2.0])          # same world-frame offset as at pick-up
#: The live state the plan starts from: holder at the origin, point 1 m ahead.
#: The point's own value here matters -- before the hold starts, the object is
#: pinned stationary AT IT (_resolve_stationary_objects), and the carry
#: projection below reads that pinned pick-up row, not the raw wp.
X0 = np.array([0.0, 0.0, 0.0, 1.0, 0.0])


def build(rigid):
    g = GraphOfConstraints([ROBOT_SPEC], [[Block.R(OBJ_W)]], state_lower_bound=-10.0,
                           state_upper_bound=10.0, robot_names=["r0"],
                           object_names=["obj"], workspace_dim=2)
    u, v = g.structure.add_nodes(2)
    g.structure.add_edge(u, v, True)
    g.add_hold(u, v, 0, [0], rigid=rigid)
    return g, (u, v)


def scene(g, nodes, obj_v):
    """`(problem, wp)`: the holder at the origin facing +x with the point 1 m
    in front, then at (2, 2) turned 90 degrees with the point at `obj_v`."""
    problem = build_graph_ordering_problem(g, np.zeros((1, AGENT_W)), wp_bounds=(-10.0, 10.0),
                                           objective="avg")
    u, v = nodes
    wp = np.zeros((problem.n_nodes, problem.state_dim))
    wp[u, :AGENT_W] = [0.0, 0.0, 0.0]
    wp[u, AGENT_W:AGENT_W + OBJ_W] = [1.0, 0.0]
    wp[v, :AGENT_W] = [2.0, 2.0, np.pi / 2]
    wp[v, AGENT_W:AGENT_W + OBJ_W] = obj_v
    return problem, wp


def residual_cv(g, nodes, obj_v):
    """Violation of the hold's own residual at `obj_v` -- the compiled eq
    constraints read directly, NOT through _evaluate_population_jax, whose
    apply_projections would overwrite the released object's column with the
    carry pin below before any residual sees it."""
    problem, wp = scene(g, nodes, obj_v)
    n = problem.n_nodes
    args = (jnp.zeros((1, problem.n_variables, problem.n_agents)), jnp.zeros((1, problem.n_cond_vars)),
            jnp.arange(n, dtype=float)[None], jnp.asarray(wp)[None], jnp.asarray(wp)[None],
            jnp.ones((n,), bool), jnp.asarray(X0), jnp.asarray(problem.params))
    rows = np.concatenate([np.asarray(fn(*args))[0] for fn in problem._eq_constraints])
    return float(np.sum(np.abs(rows)))


def carried_point(g, nodes):
    """Where the hold's auto-derived carry projection pins the released
    object's column, given the same two holder poses."""
    problem, wp = scene(g, nodes, np.zeros(OBJ_W))
    n = problem.n_nodes
    out = apply_projections(
        problem, jnp.asarray(wp)[None], jnp.zeros((1, problem.n_psi)),
        jnp.zeros((1, problem.n_branch)), jnp.asarray(problem.params),
        assign=jnp.zeros((1, problem.n_variables, problem.n_agents)),
        cond_binary=jnp.zeros((1, problem.n_cond_vars)),
        t=jnp.arange(n, dtype=float)[None], node_active=jnp.ones((n,), bool),
        x0=jnp.asarray(X0))
    return np.asarray(out)[0, nodes[1], AGENT_W:AGENT_W + OBJ_W]


def test_rigid_vs_translation():
    for rigid, good, bad in ((True, ROT_CARRY, WORLD_CARRY), (False, WORLD_CARRY, ROT_CARRY)):
        g, nodes = build(rigid)
        r_good, r_bad, pinned = (residual_cv(g, nodes, good), residual_cv(g, nodes, bad),
                                 carried_point(g, nodes))
        print(f"[rigid={rigid!s:5}] residual at {good.tolist()} = {r_good:.2e}, "
              f"at {bad.tolist()} = {r_bad:.2e}; carry pin -> {np.round(pinned, 3).tolist()}")
        assert r_good < 1e-9, (rigid, good, r_good)
        assert r_bad > 1.0, (rigid, bad, r_bad)
        assert np.allclose(pinned, good, atol=1e-9), (rigid, pinned, good)


def test_milp_rejects_rigid():
    g, nodes = build(True)
    milp = MILPWaypointMPC(g, [CubicConfigurationSpline(ROBOT_SPEC)])
    try:
        milp.solve(list(nodes), X0)
    except RuntimeError as e:
        if "Gurobi" in str(e):   # no usable licence here; nothing to assert
            print(f"[milp] skipped: {str(e).splitlines()[0][:80]}")
            return
        assert "rigid" in str(e), e
        print(f"[milp] refused a rigid hold: {str(e).splitlines()[0][:96]}...")
        return
    raise AssertionError("MILPWaypointMPC accepted a rigid hold")


if __name__ == "__main__":
    test_rigid_vs_translation()
    test_milp_rejects_rigid()
    print("\nAll checks passed.")

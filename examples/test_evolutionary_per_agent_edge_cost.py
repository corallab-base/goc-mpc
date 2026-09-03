"""Validation for per-agent `edge_cost_fn` in the evolutionary waypoint solver.

`make_graph_kernel` / `GraphOrderingRelaxed` / `EvolutionaryWaypointSolver`
accept `edge_cost_fn` either as ONE shared callable `(a, b) -> scalar`
(priced against every agent's route, the historical behavior) or as a
per-agent list/tuple, one entry per graph agent -- each agent's route cost
is then priced by its own edge cost. The routing/assignment objective is
`mean`/`max` of the per-agent route costs, so a shared field misprices the
assignment whenever two agents' reachable free space differs (both dual-arm
UR5e experiments, UR5e vs G1). See kernel.make_graph_kernel's docstring.

2 robots, dim 2, no objects. Two nodes, one per robot, each pinning its own
`agent_q(k)` to a fixed target -- so each agent's route is exactly
depot_k -> target_k and its cost is `edge_cost_fn[k](depot_k, target_k)`.
That makes the expected `route_cost` a closed form to check the kernel
against:

  * shared `f = ||.||`                 -> mean(||d0||, ||d1||)
  * per-agent `[f, f]`                 -> identical to the shared case
  * per-agent `[f, 3*f]`               -> mean(||d0||, 3*||d1||)

Run directly: python examples/test_evolutionary_per_agent_edge_cost.py
"""

import numpy as np
import jax.numpy as jnp
from pydrake.math import eq

from goc_mpc import EvolutionaryWaypointSolver, GraphOfConstraints
from goc_mpc._ext.configuration_spline import Block, CubicConfigurationSpline
from goc_mpc.evolutionary_waypoint_solver.spec import (
    build_graph_ordering_problem, _agent_widths, _slot_width,
)

DIM = 2
DEPOT0 = np.array([0.0, 0.0])
DEPOT1 = np.array([0.0, 0.0])
TARGET0 = np.array([3.0, 0.0])   # ||d0|| = 3
TARGET1 = np.array([0.0, 4.0])   # ||d1|| = 5
TOL = 1e-6


def _euclidean(a, b):
    return jnp.linalg.norm(b - a)


def _scaled_euclidean(scale):
    return lambda a, b: scale * jnp.linalg.norm(b - a)


def _build_problem(edge_cost_fn):
    robot_spec = [Block.R(DIM)]
    graph = GraphOfConstraints(
        [robot_spec, robot_spec], [],
        state_lower_bound=-20.0, state_upper_bound=20.0,
        robot_names=["r0", "r1"],
    )
    n0, n1 = graph.structure.add_nodes(2)
    graph.structure.add_edge(n0, n1, True)
    graph.add_constraint(n0, eq(graph.agent_q(0), TARGET0))
    graph.add_constraint(n1, eq(graph.agent_q(1), TARGET1))

    agent_widths = _agent_widths(graph)
    slot_width = _slot_width(agent_widths)
    x0_per_agent = np.zeros((graph.num_agents, slot_width))
    x0_per_agent[0, :DIM] = DEPOT0
    x0_per_agent[1, :DIM] = DEPOT1

    return graph, build_graph_ordering_problem(
        graph, x0_per_agent, (-20.0, 20.0),
        objective="avg", edge_cost_fn=edge_cost_fn,
    ), n0, n1


def _route_cost(problem, n0, n1):
    """Price the fully-determined waypoints (each agent pinned to its own
    target) through the built kernel's own decode_and_cost."""
    state_dim = problem.state_dim
    wp = np.zeros((problem.n_nodes, state_dim))
    # instance for agent k gathers columns [k*dim : k*dim+dim] of its node's
    # row (kernel.py's `col = owner_instance * dim`).
    wp[n0, 0:DIM] = TARGET0
    wp[n1, DIM:2 * DIM] = TARGET1
    assign = np.zeros((problem.n_variables, problem.n_agents))
    cond_binary = np.zeros((problem.n_cond_vars,))
    t = np.array([0.0, 1.0])
    x0 = np.stack([DEPOT0, DEPOT1])
    node_active = np.ones((problem.n_nodes,), dtype=bool)
    route_cost, _g = problem._decode_and_cost(assign, cond_binary, t, wp, x0, node_active)
    return float(route_cost)


def main():
    d0 = np.linalg.norm(TARGET0 - DEPOT0)   # 3.0
    d1 = np.linalg.norm(TARGET1 - DEPOT1)   # 5.0

    graph, prob_shared, n0, n1 = _build_problem(_euclidean)
    got_shared = _route_cost(prob_shared, n0, n1)
    exp_shared = 0.5 * (d0 + d1)
    print(f"shared f=||.||        route_cost {got_shared:.6f}  expected {exp_shared:.6f}")
    assert abs(got_shared - exp_shared) < TOL, (got_shared, exp_shared)

    _g, prob_list_same, n0, n1 = _build_problem([_euclidean, _euclidean])
    got_list_same = _route_cost(prob_list_same, n0, n1)
    print(f"per-agent [f, f]      route_cost {got_list_same:.6f}  expected {exp_shared:.6f}")
    assert abs(got_list_same - got_shared) < TOL, (got_list_same, got_shared)

    _g, prob_list_scaled, n0, n1 = _build_problem([_euclidean, _scaled_euclidean(3.0)])
    got_list_scaled = _route_cost(prob_list_scaled, n0, n1)
    exp_scaled = 0.5 * (d0 + 3.0 * d1)
    print(f"per-agent [f, 3*f]    route_cost {got_list_scaled:.6f}  expected {exp_scaled:.6f}")
    assert abs(got_list_scaled - exp_scaled) < TOL, (got_list_scaled, exp_scaled)

    # End-to-end: the full GA + Lamarckian gradient refine runs through the
    # unrolled per-agent loop (jax.vjp through a Python loop of
    # _one_agent_route calls) and still lands each agent on its pinned
    # target.
    graph, _prob, n0, n1 = _build_problem([_euclidean, _scaled_euclidean(3.0)])
    splines = [CubicConfigurationSpline([Block.R(DIM)]),
               CubicConfigurationSpline([Block.R(DIM)])]
    solver = EvolutionaryWaypointSolver(
        graph, splines, objective="avg",
        edge_cost_fn=[_euclidean, _scaled_euclidean(3.0)],
        wp_bounds=(-20.0, 20.0), pop_size=40, n_gen=80,
        outer_iters=3, inner_maxiter=30,
    )
    x0 = np.concatenate([DEPOT0, DEPOT1])
    ok = solver.solve(list(range(graph.structure.num_nodes)), x0)
    assert ok, "solve() with a per-agent edge_cost_fn list returned False"
    wp = solver.view_waypoints()
    agent0_n0 = wp[n0, 0:DIM]
    agent1_n1 = wp[n1, DIM:2 * DIM]
    print(f"\n[e2e] agent0 @ n0 {agent0_n0}  (target {TARGET0})")
    print(f"[e2e] agent1 @ n1 {agent1_n1}  (target {TARGET1})")
    assert np.allclose(agent0_n0, TARGET0, atol=0.15), (agent0_n0, TARGET0)
    assert np.allclose(agent1_n1, TARGET1, atol=0.15), (agent1_n1, TARGET1)

    # Wrong-length list is a clean error, not a deep trace failure.
    try:
        _build_problem([_euclidean, _euclidean, _euclidean])
    except ValueError as e:
        print(f"length check OK: {e}")
    else:
        raise AssertionError("a 3-entry edge_cost_fn list for a 2-agent graph should raise")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()

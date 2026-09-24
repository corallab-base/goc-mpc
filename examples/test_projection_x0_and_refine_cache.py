"""`ProjOperator(reads_x0=True, refine_cached=True)`.

* reads_x0: `func` gets the live full-state x0 as its last arg -- the pinned
  row must follow x0 between solves.
* refine_cached: `func` runs once per local_refine call instead of on every
  merit evaluation -- same solution, far fewer evaluations (counted at
  runtime with jax.debug.callback).

Run: python examples/test_projection_x0_and_refine_cache.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from pydrake.math import eq

from goc_mpc import EvolutionaryWaypointSolver, GraphOfConstraints
from goc_mpc._ext.configuration_spline import Block, CubicConfigurationSpline
from goc_mpc.evolutionary_waypoint_solver.projection import ProjOperator

DIM = 2
OFFSET = np.array([1.5, -0.5])
EU = lambda a, b: jnp.sqrt(jnp.sum((jnp.asarray(b) - jnp.asarray(a)) ** 2) + 1e-12)
CALLS = [0]


def _count(n):
    CALLS[0] += int(np.asarray(n).size)


def build(refine_cached):
    g = GraphOfConstraints([[Block.R(DIM)], [Block.R(DIM)]], [],
                           state_lower_bound=-25.0, state_upper_bound=25.0,
                           robot_names=["r0", "r1"])
    n0, n1 = g.structure.add_nodes(2)
    g.structure.add_edge(n0, n1, True)

    def func(psi, branch, x0):
        jax.debug.callback(_count, branch)
        return x0[0:DIM] + jnp.asarray(OFFSET)

    g.add_constraint(n0, eq(g.agent_q(0), np.zeros(DIM)),
                     proj=ProjOperator(pins=g.agent_q(0), reads=(), func=func,
                                       reads_x0=True, refine_cached=refine_cached))
    g.add_constraint(n1, eq(g.agent_q(1), np.array([4.0, 4.0])))
    return g, (n0, n1)


def solve(refine_cached, x0s):
    g, nodes = build(refine_cached)
    splines = [CubicConfigurationSpline(s) for s in g._robot_specs]
    solver = EvolutionaryWaypointSolver(g, splines, objective="avg", edge_cost_fn=EU,
                                        wp_bounds=(-25.0, 25.0), pop_size=6, n_gen=3,
                                        outer_iters=2, inner_maxiter=20,
                                        algorithm="small_continuous_vrp")
    rows = []
    for x0 in x0s:
        assert solver.solve(list(nodes), x0)
        rows.append(np.array(solver.view_waypoints()))
    return rows, nodes


def main():
    x0s = [np.array([0.0, 0.0, 9.0, 9.0]), np.array([2.0, -1.0, 9.0, 9.0])]
    out = {}
    for rc in (False, True):
        CALLS[0] = 0
        rows, (n0, n1) = solve(rc, x0s)
        jax.effects_barrier()
        out[rc] = (rows, CALLS[0])
        for x0, wp in zip(x0s, rows):
            assert np.allclose(wp[n0, 0:DIM], x0[0:DIM] + OFFSET, atol=1e-9), (rc, wp[n0], x0)
            assert np.allclose(wp[n1, DIM:2 * DIM], [4.0, 4.0], atol=1e-3), (rc, wp[n1])
        print(f"[refine_cached={rc!s:5}] pinned row follows x0; func evaluations: {CALLS[0]}")
    for a, b in zip(out[False][0], out[True][0]):
        assert np.allclose(a, b, atol=1e-6), (a, b)
    assert out[True][1] * 5 < out[False][1], (out[True][1], out[False][1])
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()

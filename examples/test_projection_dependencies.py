"""`ProjectionEntry.gate_nodes` (spec.py) and the dependency analysis built
on it (`structure.projection_dependencies`, and the coupled backend's
`dp_master_jax._order_dependent_rows`).

  A. gate_nodes contract, on real hold graphs built through spec.py: every
     gated entry declares its nodes, and the gate's value really does depend
     on nothing else (ranks outside gate_nodes are scrambled, including
     passed-node ties at -1, and the gate must not notice).
  B. classification on those graphs: a gate hard precedence already fixes is
     proved constant; one it doesn't is "varying".
  C. the coupled backend's fence, on stand-in entries covering each way a
     row can depend on what the DP decides -- and each way it can't.

Run: python examples/test_projection_dependencies.py
"""

import sys
from types import SimpleNamespace as NS

import numpy as np
import jax.numpy as jnp
from pydrake.math import eq

from goc_mpc import GraphOfConstraints
from goc_mpc._ext.configuration_spline import Block
from goc_mpc.evolutionary_waypoint_solver.spec import (
    build_graph_ordering_problem, _make_interval_overlap_gate,
    _interval_overlap_gate_nodes, _always_on_gate)
from goc_mpc.logic_based_benders_solver.structure import (
    projection_dependencies, _gate_rank_batch, _hard_reach)
from goc_mpc.logic_based_benders_solver.dp_master import _needs_full_resolve
from goc_mpc.logic_based_benders_solver.dp_master_jax import _order_dependent_rows

DIM = 2
FAILS = 0


def _fail(msg):
    global FAILS
    FAILS += 1
    print(f"[FAIL] {msg}")


# --------------------------------------------------------------- A + B
def chain_hold_graph():
    """n0 -> n1 -> n2 -> n3, one robot, one object, hold over (n1, n2):
    precedence fixes every hold span relative to every pin."""
    g = GraphOfConstraints([[Block.R(DIM)]], [[Block.R(DIM)]], state_lower_bound=-9.0,
                           state_upper_bound=9.0, robot_names=["r0"], object_names=["o0"])
    n = g.structure.add_nodes(4)
    for a, b in zip(n, n[1:]):
        g.structure.add_edge(a, b, True)
    g.add_constraint(n[0], eq(g.object_q(0), np.zeros(DIM)))
    g.add_hold(n[1], n[2], 0, [0])
    return build_graph_ordering_problem(g, np.zeros((1, DIM)), (-9.0, 9.0))


def parallel_hold_graph():
    """n1 -> n2 holds the object; n0 is a separate source node with NO
    precedence relative to the hold, so the depot pin at n0 genuinely
    depends on whether the hold has started by then."""
    g = GraphOfConstraints([[Block.R(DIM)]], [[Block.R(DIM)]], state_lower_bound=-9.0,
                           state_upper_bound=9.0, robot_names=["r0"], object_names=["o0"])
    n = g.structure.add_nodes(3)
    g.structure.add_edge(n[1], n[2], True)
    g.add_constraint(n[0], eq(g.object_q(0), np.zeros(DIM)))
    g.add_hold(n[1], n[2], 0, [0])
    return build_graph_ordering_problem(g, np.zeros((1, DIM)), (-9.0, 9.0)), n


def check_contract(name, problem, rng, n_samples=400):
    N = problem.n_nodes
    gated = [(i, e) for i, e in enumerate(problem.projections) if e.gate_fn is not None]
    if not gated:
        _fail(f"{name}: expected gated projections, found none")
        return
    for i, e in gated:
        if e.gate_nodes is None:
            _fail(f"{name}: gated entry #{i} @node {e.write_node} has no gate_nodes")
            continue
        others = [n for n in range(N) if n not in e.gate_nodes]
        R = np.stack([rng.permutation(N) for _ in range(n_samples)]).astype(np.int32)
        R[rng.random(R.shape) < 0.3] = -1                       # passed-node ties
        R2 = R.copy()
        if others:
            for row in R2:
                row[others] = rng.permutation(len(others)) + 1000
                row[[o for o in others if rng.random() < 0.3]] = -1
        g1 = np.asarray(e.gate_fn(jnp.asarray(R)))
        g2 = np.asarray(e.gate_fn(jnp.asarray(R2)))
        if not np.array_equal(g1, g2):
            _fail(f"{name}: gate #{i} @node {e.write_node} reads a node outside "
                  f"gate_nodes={sorted(e.gate_nodes)}")
    print(f"[ok ] {name}: all {len(gated)} gated entries declare gate_nodes, and "
          f"scrambling every other node's rank ({n_samples} samples) never changes a gate")


def check_classification():
    p = chain_hold_graph()
    deps = projection_dependencies(p)
    gated = [(i, e, d) for i, (e, d) in enumerate(zip(p.projections, deps))
             if e.gate_fn is not None]
    bad = [(i, d.gate) for i, e, d in gated if d.gate == "varying"]
    if bad:
        _fail(f"chain hold graph: precedence fixes every span, yet gates {bad} are 'varying'")
    else:
        print(f"[ok ] chain hold graph: all {len(gated)} gates proved constant "
              f"({sorted({d.gate for _i, _e, d in gated})})")

    p, n = parallel_hold_graph()
    deps = projection_dependencies(p)
    depot = [(i, d) for i, (e, d) in enumerate(zip(p.projections, deps))
             if e.gate_fn is not None and int(e.write_node) == int(n[0])]
    if not depot:
        _fail("parallel hold graph: no gated pin at the unordered source node")
    elif any(d.gate != "varying" for _i, d in depot):
        _fail(f"parallel hold graph: depot pin at unordered n0 classified "
              f"{[d.gate for _i, d in depot]}, want 'varying'")
    else:
        print("[ok ] parallel hold graph: the depot pin at the node unordered "
              "w.r.t. the hold is 'varying'")


# ------------------------------------------------------------------- C
AGENT_COLS, OBJ_A, OBJ_B = (0, 1), 4, 5          # n_agents=2, dim=2 -> agent band 0..3


def entry(node, cols, reads=(), gate=None, gate_nodes=None, slot=None):
    return NS(write_node=node, node_locals=(node,), gate_fn=gate, gate_nodes=gate_nodes,
              write_cols=frozenset((node, c) for c in cols),
              read_cols=frozenset(reads), owner_var_slot=slot)


def prob(n_nodes, edges, *entries):
    return NS(projections=list(entries), n_agents=2, dim=2, n_nodes=n_nodes,
              ordering_edges=[(u, v, None) for (u, v) in edges])


def ik(node, reads):
    return entry(node, AGENT_COLS, reads=reads)


def depot(v, hold):
    return (_make_interval_overlap_gate(None, v, [hold]),
            _interval_overlap_gate_nodes(None, v, [hold]))


def check_fence():
    g_dep, n_dep = depot(2, (0, 1))
    edge_gate = _make_interval_overlap_gate(1, 3, [(0, 2)])
    edge_nodes = _interval_overlap_gate_nodes(1, 3, [(0, 2)])
    cases = [
        # name, problem, flagged entry indices, substring expected in reason
        ("IK reads a depot pin whose hold span is unordered w.r.t. it",
         prob(3, [], entry(2, [OBJ_A], gate=g_dep, gate_nodes=n_dep),
              ik(2, [(2, OBJ_A)])), [1], "order of nodes [0, 1, 2]"),
        ("same pin, but hard precedence puts the node before the hold",
         prob(3, [(2, 0), (0, 1)], entry(2, [OBJ_A], gate=g_dep, gate_nodes=n_dep),
              ik(2, [(2, OBJ_A)])), [], None),
        ("anchor ties: constant over every strict order, flips once 0,1,2 are passed",
         prob(4, [(0, 1), (1, 2), (2, 3)],
              entry(3, [OBJ_A], reads=[(1, OBJ_A)], gate=edge_gate, gate_nodes=edge_nodes),
              ik(3, [(3, OBJ_A)])), [1], "order of nodes [0, 1, 2, 3]"),
        ("gate with unknown gate_nodes",
         prob(3, [(2, 0), (0, 1)], entry(2, [OBJ_A], gate=g_dep, gate_nodes=None),
              ik(2, [(2, OBJ_A)])), [1], "the whole order"),
        ("always-on gate (rigid-carry style)",
         prob(2, [], entry(1, [OBJ_A], gate=_always_on_gate, gate_nodes=frozenset()),
              ik(1, [(1, OBJ_A)])), [], None),
        ("transitive: IK <- ungated pin <- varying gated pin",
         prob(3, [], entry(2, [OBJ_A], gate=g_dep, gate_nodes=n_dep),
              entry(2, [OBJ_B], reads=[(2, OBJ_A)]), ik(2, [(2, OBJ_B)])), [2], "pin(s) [0]"),
        ("robot read: IK <- rigid-carry pin reading two robot rows",
         prob(3, [(0, 1), (1, 2)],
              entry(1, [OBJ_A], reads=[(0, 0), (1, 0), (0, OBJ_A)],
                    gate=_always_on_gate, gate_nodes=frozenset()),
              ik(2, [(1, OBJ_A)])), [1], "robot configuration at node(s) [0, 1]"),
        ("dynamic (var_agent_q) pin: depends on the assignment only",
         prob(1, [], entry(0, AGENT_COLS, slot=0)), [], None),
        ("varying gate on an object column nobody routing-relevant reads",
         prob(3, [], entry(2, [OBJ_A], gate=g_dep, gate_nodes=n_dep)), [], None),
    ]
    for name, p, want, substr in cases:
        got = _order_dependent_rows(p)
        got_idx = [i for i, _n, _w in got]
        if got_idx != want:
            _fail(f"fence '{name}': flagged {got}, want entries {want}")
        elif substr is not None and substr not in got[0][2]:
            _fail(f"fence '{name}': reason {got[0][2]!r} lacks {substr!r}")
    # the tie case must really be tie-driven: over strict orders alone the
    # gate is constant, and only the passed-node enumeration exposes it.
    reach = _hard_reach(4, [(0, 1), (1, 2), (2, 3)])
    strict = np.asarray([[0, 1, 2, 3]], np.int32)
    ties = _gate_rank_batch(4, edge_nodes, reach, 1000)
    g_strict = np.asarray(edge_gate(jnp.asarray(strict)))
    g_all = np.asarray(edge_gate(jnp.asarray(ties)))
    if not (len(set(g_strict.tolist())) == 1 and len(set(g_all.tolist())) == 2):
        _fail(f"tie case premise: strict gates {g_strict}, with ties {sorted(set(g_all))}")
    # dynamic pins are the one case _needs_full_resolve over-counts
    if not _needs_full_resolve(prob(1, [], entry(0, AGENT_COLS, slot=0))):
        _fail("premise: _needs_full_resolve should count a dynamic agent pin")
    print(f"[ok ] fence: {len(cases)} stand-in cases -- order dependence (incl. "
          f"transitive, unknown gate_nodes, and one exposed only by passed-node ties), "
          f"robot-row dependence; and not flagged for precedence-fixed, always-on, "
          f"dynamic, or object-only pins")


def main():
    rng = np.random.default_rng(0)
    check_contract("chain hold graph", chain_hold_graph(), rng)
    check_contract("parallel hold graph", parallel_hold_graph()[0], rng)
    check_classification()
    check_fence()
    print(f"\nFAILURES: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())

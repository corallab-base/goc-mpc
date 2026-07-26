"""Port of examples/new_p1_experiments/problem_formulations.py's
GraphOrderingBase/GraphOrderingRelaxed (the real-relaxed formulation only --
the mixed-variable pymoo GraphOrderingProblem is comparison tooling for the
experiment suite, not needed by the production solver, and stays in
examples/).

Generalizes the ported original in three ways:
  1. hard_edges/cond_pairs -> ordering_edges (see kernel.make_graph_kernel's
     docstring): a single (u, v, gate_fn) registry, gate_fn=None for an
     always-active hard edge.
  2. A new cond_binary decision-variable block (one continuous-relaxed [0, 1]
     score per GraphOfConstraints binary_cond_sym_var) sits between the
     assign block and the (t, wp) blocks. This is exactly parallel to
     `assign` -- GA-searched, not gradient-refined -- see
     kernel.make_graph_kernel's decode_and_cost for how compiled gates read
     it. `eq_constraints`/`ineq_constraints` functions never need it (they
     only ever look at wp), so this package's solver.py calls them with the
     original (assign, t, wp) 3-arg contract, unchanged; only
     problem._batched picks up the extra cond_binary argument.
  3. wp is no longer one row per routing instance. It's a DENSE per-node
     tensor -- `wp[node] = [agent_0 | agent_1 | ... | object_0 | ...]`,
     `state_dim`-wide, mirroring MILP's per-node joint configuration `W`
     (graph_of_constraints.cpp:79's `total_dim`) -- so symbolic node/edge
     constraints referencing any agent's or object's placeholder at a node
     (spec.py's _make_dense_resolver) are plain column offsets into ONE
     shared row, with no per-formula sparse-instance bookkeeping. `t`
     (the ordering/arrival-time score) stays indexed by routing instance,
     same as before -- see kernel.py's module docstring for how
     `instance_dense_node` bridges the two. `n_nodes` therefore keeps its
     old meaning (routing-instance count -- solver.py's t-block/2-opt/OX
     machinery is entirely generic over it and needs no changes); the new
     `n_dense_nodes`/`state_dim` describe the separate wp block.
"""

from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np

from .kernel import make_graph_kernel


# Bundles remaining_vertices' runtime effect on an otherwise-fixed-size
# problem (see GraphOrderingSpec's module docstring) into one pytree,
# threaded alongside x0 through decode_and_cost/_batched/merit/local_refine/
# _routing_local_search_batched/gen_step/step (solver.py, kernel.py) -- built
# fresh each solve() call by EvolutionaryWaypointSolver._compute_anchor
# (mpc.py) from remaining_vertices plus the solver's own persisted
# _waypoints/_object_waypoints/_var_assignments history.
#   node_active: (n_dense_nodes,) bool -- is this node still in
#       remaining_vertices (a free decision variable) or already passed (an
#       anchored constant)?
#   anchor_wp: (n_dense_nodes, state_dim) float -- for a passed node, its
#       last-committed dense row (irrelevant/unused where node_active=True).
#   var_committed: (n_variables,) bool -- has ANY instance using this
#       assignable variable already passed? If so its assignment must stay
#       pinned (mirrors MILP's "don't let the routing solve reassign this
#       hold's holder mid-grasp").
#   var_anchor: (n_variables,) int -- the committed real-agent id for a
#       var_committed variable (irrelevant/unused otherwise).
AnchorState = namedtuple("AnchorState", ["node_active", "anchor_wp", "var_committed", "var_anchor"])


def apply_anchor(problem, assign, wp, anchor):
    """Splices `anchor` into batched (pop, ...) `assign`/`wp` decision
    variables, returning (assign_eff, wp_eff): a node/variable no longer in
    remaining_vertices reads back as its last-committed constant here
    instead of the free GA/L-BFGS value. Callers should use assign_eff/
    wp_eff everywhere a constraint or routing decision is made (never the
    raw assign/wp) -- this is the single substitution point that lets a
    node/edge constraint anchored at an already-passed node stay correctly
    enforced (against a known constant) rather than being dropped, mirroring
    MILP's boundary-edge substitution, with no changes needed to spec.py's
    constraint compilation itself. jnp.where's VJP naturally zeroes gradient
    into anchored wp slots, so solver.py's L-BFGS local refinement needs no
    extra masking."""
    wp_eff = jnp.where(anchor.node_active[None, :, None], wp, anchor.anchor_wp[None, :, :])
    if problem.n_variables > 0:
        anchor_one_hot = jax.nn.one_hot(anchor.var_anchor, problem.n_agents, dtype=assign.dtype)
        assign_eff = jnp.where(anchor.var_committed[None, :, None], anchor_one_hot[None, :, :], assign)
    else:
        assign_eff = assign
    return assign_eff, wp_eff


def full_active_anchor(problem):
    """AnchorState with every node "active" (free, not yet passed) and no
    committed variable assignments -- the correct anchor for the very first
    solve of a fresh problem, before anything has been solved/written back.
    EvolutionaryWaypointSolver._compute_anchor (mpc.py) naturally reduces to
    this whenever remaining_vertices == every graph node (in particular, the
    genuine first-ever call, since GraphOfConstraintsMPC always starts
    remaining_phases as every node -- goc_mpc.py)."""
    return AnchorState(
        node_active=jnp.ones((problem.n_dense_nodes,), dtype=bool),
        anchor_wp=jnp.zeros((problem.n_dense_nodes, problem.state_dim)),
        var_committed=jnp.zeros((problem.n_variables,), dtype=bool),
        var_anchor=jnp.zeros((problem.n_variables,), dtype=jnp.int32),
    )


def _infer_constraint_size(fn, n_variables, n_agents, n_instances, n_dense_nodes, state_dim):
    dummy_assign = np.zeros((1, n_variables, n_agents))
    dummy_t = np.zeros((1, n_instances))
    dummy_wp = np.zeros((1, n_dense_nodes, state_dim))
    return np.asarray(fn(dummy_assign, dummy_t, dummy_wp)).shape[1]


class GraphOrderingRelaxed:
    def __init__(self, instance_sources, n_variables, ordering_edges, x0, wp_bounds,
                 instance_dense_node, n_dense_nodes, state_dim,
                 n_cond_vars=0, objective="avg", penalty_weight=20.0, edge_cost_fn=None,
                 eq_constraints=(), ineq_constraints=()):
        self.instance_sources = list(instance_sources)
        # Routing-instance count (NOT the dense-node count -- see module
        # docstring). Kept as `n_nodes` since solver.py treats this purely as
        # an opaque t-block row count, generic over the node-vs-instance
        # distinction -- renaming it would be a many-site mechanical change
        # there for no functional benefit.
        self.n_nodes = len(self.instance_sources)
        self.n_variables = n_variables
        self.n_agents = x0.shape[0]
        self.dim = x0.shape[1]
        self.x0 = x0
        self.objective = objective
        self.n_cond_vars = n_cond_vars

        self.instance_dense_node = np.asarray(instance_dense_node, dtype=int)
        self.n_dense_nodes = n_dense_nodes
        self.state_dim = state_dim

        # Stashed as plain problem-instance data so callers with only
        # `problem` in hand (e.g. the seed heuristic in solver.py) can
        # recover graph structure. Only edges with gate_fn=None (always
        # active) count as "hard" for that heuristic's precedence DAG.
        self.ordering_edges = list(ordering_edges)
        self.hard_edges = [(u, v) for u, v, gate in self.ordering_edges if gate is None]

        kernel_kwargs = {} if edge_cost_fn is None else {"edge_cost_fn": edge_cost_fn}
        self._decode_and_cost, self._batched = make_graph_kernel(
            self.instance_sources, self.n_variables, self.ordering_edges,
            self.instance_dense_node, self.dim,
            objective, penalty_weight, **kernel_kwargs)

        self._eq_constraints = list(eq_constraints)
        self._ineq_constraints = list(ineq_constraints)
        sizes = lambda fns: sum(_infer_constraint_size(fn, self.n_variables, self.n_agents,
                                                         self.n_nodes, self.n_dense_nodes, self.state_dim)
                                 for fn in fns)
        self.n_eq_extra = sizes(self._eq_constraints)
        self.n_ineq_extra = sizes(self._ineq_constraints)
        self.n_eq_constr = self.n_eq_extra
        self.n_ieq_constr = self.n_ineq_extra

        self.n_assign_vars = self.n_variables * self.n_agents
        self.cond_offset = self.n_assign_vars
        self.t_offset = self.cond_offset + self.n_cond_vars
        self.wp_offset = self.t_offset + self.n_nodes
        n_var = self.wp_offset + self.n_dense_nodes * self.state_dim
        self.n_var = n_var

        xl = np.zeros(n_var)
        xu = np.ones(n_var)
        wp_lo, wp_hi = wp_bounds
        xl[self.wp_offset:] = wp_lo
        xu[self.wp_offset:] = wp_hi
        xl[self.t_offset:self.wp_offset] = 0.0
        xu[self.t_offset:self.wp_offset] = float(self.n_nodes - 1)

        # Statically-fixed instances no longer occupy any position in x at
        # all (no synthetic vagent/one-hot row to pin) -- their real agent
        # id flows directly into kernel.py's owner_instance, bypassing
        # `assign` entirely.
        self.xl, self.xu = xl, xu

    def _extract_single(self, x):
        assign = x[:self.n_assign_vars].reshape(self.n_variables, self.n_agents)
        cond_binary = x[self.cond_offset:self.t_offset]
        t = x[self.t_offset:self.wp_offset]
        wp = x[self.wp_offset:].reshape(self.n_dense_nodes, self.state_dim)
        return assign, cond_binary, t, wp

    def _extract_batch(self, X):
        pop = len(X)
        assign = X[:, :self.n_assign_vars].reshape(pop, self.n_variables, self.n_agents)
        cond_binary = X[:, self.cond_offset:self.t_offset]
        t = X[:, self.t_offset:self.wp_offset]
        wp = X[:, self.wp_offset:].reshape(pop, self.n_dense_nodes, self.state_dim)
        return assign, cond_binary, t, wp

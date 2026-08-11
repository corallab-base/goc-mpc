"""GraphOrderingRelaxed: the real-relaxed decision-variable layout and problem
metadata the GA/L-BFGS solver (solver.py) optimizes over, built by
GraphOrderingSpec.build_problem() (spec.py).

The decision vector `x` is four contiguous blocks:
  assign      (n_variables, n_agents)   -- one-hot/continuous-relaxed agent
                                            choice per assignable routing
                                            variable.
  cond_binary (n_cond_vars,)            -- continuous-relaxed [0, 1] score per
                                            GraphOfConstraints binary_cond_sym_var,
                                            GA-searched exactly like `assign`;
                                            eq/ineq constraint functions never
                                            need it, only problem._batched does
                                            (kernel.make_graph_kernel's gates).
  t           (n_nodes,)                -- ordering-priority score, one per
                                            graph node -- purely a tie-break,
                                            decoded into an actual visiting
                                            order by a topological sort (see
                                            kernel.py's module docstring).
  wp          (n_nodes, state_dim)      -- per-node configuration,
                                            `wp[node] = [agent_0 | agent_1 |
                                            ... | object_0 | ...]`, mirroring
                                            MILP's per-node joint configuration
                                            `W` (graph_of_constraints.cpp:79's
                                            `total_dim`).
Ordering edges are `(u, v, gate_fn)` node-index pairs (kernel.make_graph_kernel's
docstring); `gate_fn=None` means an always-active hard edge, otherwise
`gate_fn(owner_variable, cond_binary) -> bool`.
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
# _waypoints/_var_assignments history. The current call's real state lives in
# the separate `x0` argument threaded alongside `anchor` everywhere (see
# apply_anchor) rather than inside AnchorState itself -- x0 is now the FULL
# configuration (state_dim,), the same value routing (kernel.py, sliced down
# to its agent-depot view via problem.agent_depot) already needed every call,
# so there is exactly one "current real state" flowing through this whole
# stack, not two.
#   node_active: (n_nodes,) bool -- is this node still in
#       remaining_vertices (a free decision variable) or already passed (an
#       anchored constant)?
#   anchor_wp: (n_nodes, state_dim) float -- for a passed node, its
#       last-committed row -- i.e. whatever _waypoints held for it as
#       of the last solve() where it was still active (irrelevant/unused
#       where node_active=True). This is the "planned" reading: a genuinely
#       per-node historical value, frozen at whatever the GA last solved it
#       to.
#   var_committed: (n_variables,) bool -- has ANY instance using this
#       assignable variable already passed? If so its assignment must stay
#       pinned (mirrors MILP's "don't let the routing solve reassign this
#       hold's holder mid-grasp").
#   var_anchor: (n_variables,) int -- the committed real-agent id for a
#       var_committed variable (irrelevant/unused otherwise).
AnchorState = namedtuple("AnchorState", ["node_active", "anchor_wp", "var_committed", "var_anchor"])


def agent_depot(problem, x0):
    """Slices the full-configuration runtime `x0` (state_dim,) down to the
    (n_agents, dim) agent-depot view kernel.py's routing math expects -- the
    object suffix, if any, is irrelevant to routing (see kernel.py's module
    docstring: an agent's route cost/arrival time only ever depends on agent
    positions) and simply dropped here."""
    return x0[: problem.n_agents * problem.dim].reshape(problem.n_agents, problem.dim)


def pad_to_state_dim(agent_x0_flat, state_dim):
    """Zero-pads a flat agent-only x0 (n_agents*dim,) out to the full
    state_dim width -- used wherever only a structural/default x0 is
    available (no real object state to report), e.g. build_initial_carry_fn's
    cold-start bootstrap or run_lamarckian_al's x0=None default (solver.py).
    Safe to zero-fill rather than track precisely: both call sites only ever
    run under an anchor with node_active all-True (nothing passed yet), so
    apply_anchor's wp_eff_live substitution -- the only reader of x0's object
    columns -- is never actually selected for any row there regardless of
    what these zeros hold."""
    n = agent_x0_flat.shape[-1]
    if n == state_dim:
        return agent_x0_flat
    return jnp.concatenate([agent_x0_flat, jnp.zeros(state_dim - n, dtype=agent_x0_flat.dtype)])


def apply_anchor(problem, assign, wp, anchor, x0):
    """Splices `anchor` into batched (pop, ...) `assign`/`wp` decision
    variables, returning (assign_eff, wp_eff_frozen, wp_eff_live): a node no
    longer in remaining_vertices reads back as either its last-committed
    "planned" constant (wp_eff_frozen, anchor.anchor_wp) or this call's real
    state (wp_eff_live, the full-configuration `x0` argument -- state_dim
    wide, broadcast across every node since x0 is one global joint
    state, not a per-node history) instead of the free GA/L-BFGS value --
    GraphOrderingSpec compiles each constraint to read from whichever of the
    two it was registered against (live_phi_ids), so which one "wins" for a
    given passed node is a per-constraint choice, not a per-node one (a
    variable no longer in remaining_vertices has no such choice -- assign_eff
    is always its committed one-hot). Callers should use assign_eff/
    wp_eff_frozen/wp_eff_live everywhere a constraint or routing decision is
    made (never the raw assign/wp) -- this is the single substitution point
    that lets a node/edge constraint anchored at an already-passed node stay
    correctly enforced (against a known constant) rather than being dropped,
    mirroring MILP's boundary-edge substitution, with no changes needed to
    spec.py's constraint compilation itself beyond the frozen/live pick.
    jnp.where's VJP naturally zeroes gradient into anchored wp slots, so
    solver.py's L-BFGS local refinement needs no extra masking. Routing
    (kernel.py) never reads a passed node's row either way (masked out via
    node_active regardless of value), so callers feeding it wp can pass
    either variant -- solver.py consistently uses wp_eff_frozen there, an
    arbitrary pick."""
    wp_eff_frozen = jnp.where(anchor.node_active[None, :, None], wp, anchor.anchor_wp[None, :, :])
    wp_eff_live = jnp.where(anchor.node_active[None, :, None], wp, x0[None, None, :])
    if problem.n_variables > 0:
        anchor_one_hot = jax.nn.one_hot(anchor.var_anchor, problem.n_agents, dtype=assign.dtype)
        assign_eff = jnp.where(anchor.var_committed[None, :, None], anchor_one_hot[None, :, :], assign)
    else:
        assign_eff = assign
    return assign_eff, wp_eff_frozen, wp_eff_live


def full_active_anchor(problem):
    """AnchorState with every node "active" (free, not yet passed) and no
    committed variable assignments -- the correct anchor for the very first
    solve of a fresh problem, before anything has been solved/written back.
    EvolutionaryWaypointSolver._compute_anchor (mpc.py) naturally reduces to
    this whenever remaining_vertices == every graph node (in particular, the
    genuine first-ever call, since GraphOfConstraintsMPC always starts
    remaining_phases as every node -- goc_mpc.py)."""
    return AnchorState(
        node_active=jnp.ones((problem.n_nodes,), dtype=bool),
        anchor_wp=jnp.zeros((problem.n_nodes, problem.state_dim)),
        var_committed=jnp.zeros((problem.n_variables,), dtype=bool),
        var_anchor=jnp.zeros((problem.n_variables,), dtype=jnp.int32),
    )


def _infer_constraint_size(fn, n_variables, n_agents, n_nodes, state_dim, n_cond_vars):
    dummy_assign = np.zeros((1, n_variables, n_agents))
    dummy_cond_binary = np.zeros((1, n_cond_vars))
    dummy_t = np.zeros((1, n_nodes))
    dummy_wp = np.zeros((1, n_nodes, state_dim))
    dummy_node_active = np.ones((n_nodes,), dtype=bool)
    return np.asarray(fn(dummy_assign, dummy_cond_binary, dummy_t, dummy_wp, dummy_wp,
                          dummy_node_active)).shape[1]


class GraphOrderingRelaxed:
    def __init__(self, instance_sources, n_variables, ordering_edges, x0, wp_bounds,
                 instance_node, n_nodes, state_dim,
                 n_cond_vars=0, objective="avg", edge_cost_fn=None,
                 eq_constraints=(), ineq_constraints=()):
        self.instance_sources = list(instance_sources)
        self.n_variables = n_variables
        self.n_agents = x0.shape[0]
        self.dim = x0.shape[1]
        self.x0 = x0
        self.objective = objective
        self.n_cond_vars = n_cond_vars

        self.instance_node = np.asarray(instance_node, dtype=int)
        self.n_nodes = n_nodes
        self.state_dim = state_dim

        # Stashed as plain problem-instance data so callers with only
        # `problem` in hand (e.g. the seed heuristic in solver.py) can
        # recover graph structure. Only edges with gate_fn=None (always
        # active) count as "hard" for that heuristic's precedence DAG.
        self.ordering_edges = list(ordering_edges)
        self.hard_edges = [(u, v) for u, v, gate in self.ordering_edges if gate is None]

        kernel_kwargs = {} if edge_cost_fn is None else {"edge_cost_fn": edge_cost_fn}
        self._decode_and_cost, self._batched, self._decode_node_rank = make_graph_kernel(
            self.instance_sources, self.n_variables, self.ordering_edges,
            self.instance_node, self.dim, self.n_nodes,
            objective, **kernel_kwargs)

        self._eq_constraints = list(eq_constraints)
        self._ineq_constraints = list(ineq_constraints)
        sizes = lambda fns: sum(_infer_constraint_size(fn, self.n_variables, self.n_agents,
                                                         self.n_nodes, self.state_dim, self.n_cond_vars)
                                 for fn in fns)
        self.n_eq_extra = sizes(self._eq_constraints)
        self.n_ineq_extra = sizes(self._ineq_constraints)
        self.n_eq_constr = self.n_eq_extra
        self.n_ieq_constr = self.n_ineq_extra

        self.n_assign_vars = self.n_variables * self.n_agents
        self.cond_offset = self.n_assign_vars
        self.t_offset = self.cond_offset + self.n_cond_vars
        self.wp_offset = self.t_offset + self.n_nodes
        n_var = self.wp_offset + self.n_nodes * self.state_dim
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
        wp = x[self.wp_offset:].reshape(self.n_nodes, self.state_dim)
        return assign, cond_binary, t, wp

    def _extract_batch(self, X):
        pop = len(X)
        assign = X[:, :self.n_assign_vars].reshape(pop, self.n_variables, self.n_agents)
        cond_binary = X[:, self.cond_offset:self.t_offset]
        t = X[:, self.t_offset:self.wp_offset]
        wp = X[:, self.wp_offset:].reshape(pop, self.n_nodes, self.state_dim)
        return assign, cond_binary, t, wp

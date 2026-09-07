"""`SmallContinuousVRPSolver`: `LamarckianGA` (lamarckian_ga.py) with discrete
BRANCH selection (`proj_branch`) replaced by an exact per-generation DP,
instead of letting crossover/mutation evolutionarily search it.

Motivation (see the ur5e_block_stacking suboptimality investigation this
class grew out of): `diag_path_length.py` proved the GA settles on a total
joint-space path length ~2x the true optimum, even though the true optimum
picks the SAME analytic-IK branch at every node -- a single, globally
consistent discrete choice the continuous-relaxed, argmax-decoded
`proj_branch` block (crossed over and Gaussian-mutated like any other real-
valued gene) demonstrably fails to converge onto reliably.

The combinatorial structure decomposes into three largely independent axes,
of very different tractability:
  - BRANCH (`proj_branch`, a projection's own discrete choice): given a
    fixed node visiting order, this is a SHORTEST PATH over a layered graph
    (one layer per chain node, one state per branch candidate) -- exactly
    the DP diag_path_length.py already validates against, O(n_nodes * K^2),
    trivially vmappable. This is the axis this class actually replaces.
  - ORDERING (`t`): a routing problem among each agent's own nodes. Left
    UNCHANGED here (still OX-crossover + 2-opt/Or-opt local search, exactly
    LamarckianGA's own) -- every scene built so far has `t` fully
    determined by problem.hard_edges anyway (no real freedom to search).
  - ASSIGNMENT (`assign`, which agent owns an assignable variable): the
    genuinely hard axis (NP-hard in general), but small in every scene
    built so far. Left UNCHANGED here too (still BLX-crossover + mutation).

So this class is a SCOPED first phase, not a general VRP solver (hence
"Small" in the name): it requires `problem.hard_edges` to form a single
total order over EVERY node (a genuinely single-chain, e.g. single-agent,
graph -- raises NotImplementedError otherwise, rather than silently
producing a meaningless order for a graph with real parallel/multi-agent
structure), and at most one projection per node. Extending it to genuine
multi-agent ordering (Held-Karp DP or the existing local search, per agent)
and to assignment (brute-force-enumerate the -- typically single-digit --
combos, Boltzmann-sampled for population diversity rather than a
deterministic argmin) is future work, deliberately deferred until a real
scene exercises either axis.
"""

import jax
import jax.numpy as jnp

from .lamarckian_ga import LamarckianGA
from .evosax_ga import _split_genome, _join_genome
from .problem import apply_anchor, apply_projections
from .solver import (
    _routing_local_search_batched,
    _write_wp_batch_jax,
    _write_psi_batch_jax,
    _write_t_batch_jax,
    _write_proj_branch_batch_jax,
    _tournament_select_jax,
    _ox_crossover_batched,
    _evaluate_population_jax,
    _combined_score,
)

jax.config.update("jax_enable_x64", True)


def _build_static_chain(problem):
    """(chain_order, node_entry, relevant_cols): `chain_order` is a topological sort of
    `problem.hard_edges` over every node -- raises if it doesn't cover all
    `problem.n_nodes` (a genuinely branching/multi-agent graph, out of
    scope for now -- see this module's docstring). Same plain Kahn's-
    algorithm-over-a-possibly-redundant/transitive edge set this module's
    own diag_path_length.py validation script uses (ur5e_block_stacking's
    real `hard_edges` include transitive pairs like (2,0) AND (2,3) AND
    (0,3), not just adjacent-node pairs, so a stricter
    "every node has at most one successor" check would incorrectly reject
    this exact, already-validated scene).

    `node_entry`: node id -> its own ProjectionEntry (problem.py), for
    every node that has one -- raises if a node has more than one (multiple
    projections per node aren't supported by this class' DP yet) or if any
    projection pins a var_agent_q(...) row (ProjectionEntry.owner_var_slot
    -- ambient-space Part C's dynamic write target is a different axis
    entirely from this class' branch-selection DP, and combining them isn't
    supported yet).

    `relevant_cols`: the sorted union of every registered projection's own
    `pinned_cols` -- the DP's distance metric (_solve_branch_dp) is
    restricted to exactly these columns, both between consecutive chain
    nodes AND from the real depot/start state (`params.x0`) to the first
    chain node. This matters for two reasons: (1) it's what makes the DP's
    answer match diag_path_length.py's own ground truth EXACTLY (that
    script's metric is `norm(q - q_home)` over just the 6 joint columns,
    never the whole row); (2) using the WHOLE row instead would let a
    node's other, genuinely-free wp columns (this scene has 9 of them per
    node alongside the 6 pinned ones -- see Part A's wp-shrink) inject
    unrelated GA-search noise into the branch decision, and skipping the
    depot term entirely (an earlier, now-fixed bug in this class) silently
    ignores the real cost of the FIRST move away from the actual start
    state. Assumes every projected node's pins describe the SAME physical
    quantity (true for this validated scene -- always `agent_q[0:6]`); a
    future scene mixing position/orientation-only pins across different
    nodes would need a more careful per-node relevant-columns notion."""
    n_nodes = problem.n_nodes
    adj = {i: [] for i in range(n_nodes)}
    indeg = {i: 0 for i in range(n_nodes)}
    for u, v in problem.hard_edges:
        adj[u].append(v)
        indeg[v] += 1
    order = [i for i, d in indeg.items() if d == 0]
    frontier = list(order)
    while frontier:
        u = frontier.pop(0)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                order.append(v)
                frontier.append(v)
    if len(order) != n_nodes:
        raise NotImplementedError(
            f"SmallContinuousVRPSolver needs problem.hard_edges to form a single "
            f"total order over every node (topological sort only covered "
            f"{len(order)}/{n_nodes}) -- likely a genuinely branching/multi-agent "
            "graph, which this class doesn't support yet (see its own docstring "
            "for the phased plan)")

    node_entry = {}
    for entry in problem.projections:
        if entry.owner_var_slot is not None:
            raise NotImplementedError(
                "SmallContinuousVRPSolver doesn't support var_agent_q(...) "
                "(dynamic-owner) projections yet -- that's a different axis "
                "(WHICH agent's row gets written) from this class' branch-"
                "selection DP (WHAT value gets written)")
        if entry.write_node in node_entry:
            raise NotImplementedError(
                f"node {entry.write_node} has more than one projection -- "
                "SmallContinuousVRPSolver only handles a single projection per "
                "node for now")
        node_entry[entry.write_node] = entry

    relevant_cols = sorted({int(c) for entry in node_entry.values() for c in entry.pinned_cols})
    return order, node_entry, relevant_cols


def _node_candidates(problem, nid, entry, wp0, params_arr):
    """`(pop, k, state_dim)` candidate rows for one chain node -- `k=1`
    (just `wp0`'s own current row) if it has no projection at all;
    otherwise every one of its projection's `discrete_params` branch
    candidates, spliced into that row's `pinned_cols` (mirrors problem.
    apply_projections' own per-entry substitution, just evaluated at EVERY
    branch instead of only the GA/argmax-selected one -- reuses `entry.
    table` verbatim when set, the same O(1) gather apply_projections
    itself uses for a tabled entry). `entry.continuous_params` is assumed
    0 here (real for every projection with a genuine discrete branch
    choice built so far, e.g. UR5e's analytic IK -- see projection.
    ProjOperator's own docstring); a future projection combining a real
    psi with discrete_params > 1 would only get an approximate (psi=0)
    candidate value here, not a fully exact one."""
    pop = wp0.shape[0]
    base_row = wp0[:, nid, :]
    if entry is None:
        return base_row[:, None, :]
    k = entry.discrete_params
    if entry.table is not None:
        values = jnp.broadcast_to(jnp.asarray(entry.table), (pop, k, len(entry.pinned_cols)))
    else:
        node_rows = tuple(wp0[:, n, :] for n in entry.node_locals)

        def _one(rows_1, entry=entry, params=params_arr):
            read_vals = entry.read_fn(rows_1, params)
            psi_1 = jnp.zeros((entry.continuous_params,))
            return jnp.stack([jnp.asarray(entry.func(*read_vals, psi_1, b)) for b in range(k)])

        values = jax.vmap(_one)(node_rows)  # (pop, k, w)
    rows = jnp.broadcast_to(base_row[:, None, :], (pop, k, problem.state_dim))
    col_idx = jnp.asarray(entry.pinned_cols)
    return rows.at[:, :, col_idx].set(values)


def _solve_branch_dp(problem, chain_order, chain_node_entry, relevant_cols, wp0, x0, params_arr):
    """Exact per-individual DP over `chain_order`'s static node chain: picks
    the branch index at every discrete-choice projection minimizing total
    L2 path length through the chain's candidate rows -- a plain
    shortest-path-over-a-layered-graph, the same algorithm diag_path_
    length.py's own ground truth uses, just run once per generation inside
    `_ask` instead of after the fact in numpy. A plain Python loop over
    `chain_order` (a small, spec-build-time-fixed list) unrolls this at
    trace time -- no jax.lax.scan/backtrack bookkeeping needed, since
    successive steps' candidate counts (`k`) can differ freely.

    Every distance -- including the FIRST chain node's cost, from the real
    depot/start state `x0` -- is computed over `relevant_cols` only (see
    _build_static_chain's own docstring for why: matching diag_path_
    length.py's metric exactly, and not letting unrelated free wp columns'
    GA-search noise leak into the branch decision). Omitting the `x0` term
    entirely was a real, confirmed bug in an earlier version of this
    function -- it silently ignored the cost of the first move away from
    the actual start state, and the DP would then optimize a path that
    looked good in isolation but wasn't actually the globally shortest one
    INCLUDING getting there from `x0`.

    Returns `(pop, n_branch)`: a plain one-hot per projection's own
    `branch_slice` (apply_projections' argmax decode then exactly recovers
    the chosen branch); a node with no real choice (discrete_params == 1)
    gets an all-zero row of width 1 -- argmax of a length-1 slice is
    always index 0 regardless of its value, so this is already correct."""
    pop = wp0.shape[0]
    cols = jnp.asarray(relevant_cols)
    rows_by_node = [(nid, chain_node_entry.get(nid), _node_candidates(problem, nid, chain_node_entry.get(nid),
                                                                        wp0, params_arr))
                    for nid in chain_order]

    x0_rel = x0[cols]  # (len(relevant_cols),)
    first_rel = rows_by_node[0][2][:, :, cols]  # (pop, k0, len(relevant_cols))
    costs = [jnp.linalg.norm(first_rel - x0_rel[None, None, :], axis=-1)]  # (pop, k0)
    backptrs = []
    for i in range(1, len(rows_by_node)):
        prev_rows = rows_by_node[i - 1][2][:, :, cols]
        cur_rows = rows_by_node[i][2][:, :, cols]
        diff = prev_rows[:, :, None, :] - cur_rows[:, None, :, :]
        dist = jnp.linalg.norm(diff, axis=-1)  # (pop, k_prev, k_cur)
        total = costs[-1][:, :, None] + dist
        costs.append(jnp.min(total, axis=1))
        backptrs.append(jnp.argmin(total, axis=1))  # (pop, k_cur)

    choices = [None] * len(rows_by_node)
    choices[-1] = jnp.argmin(costs[-1], axis=1)  # (pop,)
    for i in range(len(rows_by_node) - 2, -1, -1):
        choices[i] = jnp.take_along_axis(backptrs[i], choices[i + 1][:, None], axis=1)[:, 0]

    proj_branch = jnp.zeros((pop, problem.n_branch))
    for i, (_nid, entry, _rows) in enumerate(rows_by_node):
        if entry is None:
            continue
        onehot = jax.nn.one_hot(choices[i], entry.discrete_params)
        proj_branch = proj_branch.at[:, entry.branch_slice].set(onehot)
    return proj_branch


class SmallContinuousVRPSolver(LamarckianGA):
    """See module docstring. Same construction/Params/State/`_tell` as
    `LamarckianGA` (unchanged) -- only `_ask` differs, and only in how
    `proj_branch` is produced."""

    def __init__(self, population_size, solution, problem, **kwargs):
        super().__init__(population_size, solution, problem, **kwargs)
        self._chain_order, self._chain_node_entry, self._chain_relevant_cols = _build_static_chain(problem)

    def _ask(self, key, state, params):
        problem = self.problem
        pop_size = self.population_size
        n_var = problem.n_var
        wp_offset, t_offset = problem.wp_offset, problem.t_offset
        n_nodes = problem.n_nodes

        X, mu, lam, rho = _split_genome(problem, state.population)
        F_parent, CV_parent = _evaluate_population_jax(
            problem, X, params.x0, params.problem_params, params.anchor)
        S = _combined_score(F_parent, CV_parent, params.w, params.cv_tol)

        key, k_p1, k_p2, k_cx_mask, k_cx_alpha, k_mut, k_ox_mask, k_ox, k_2opt = jax.random.split(key, 9)

        p1 = _tournament_select_jax(k_p1, S, pop_size, k=self.tournament_k)
        p2 = _tournament_select_jax(k_p2, S, pop_size, k=self.tournament_k)
        parent = jnp.where(S[p1] < S[p2], p1, p2)

        # Same BLX-alpha crossover + Gaussian mutation as LamarckianGA, over
        # the WHOLE X -- including proj_branch's own slice, harmlessly:
        # whatever crossover/mutation leaves there is about to be entirely
        # overwritten by the exact DP below, so there's no need to carve it
        # out of this generic operator (it's cheaper to let it run and
        # discard the result than to special-case the slice away).
        do_cx = jax.random.uniform(k_cx_mask, (pop_size,)) < params.cx_prob
        alpha = jax.random.uniform(k_cx_alpha, (pop_size, n_var), minval=-0.5, maxval=1.5)
        blended = X[p1] + alpha * (X[p2] - X[p1])
        child_X = jnp.where(do_cx[:, None], blended, X[parent])
        xl, xu = jnp.asarray(problem.xl), jnp.asarray(problem.xu)
        noise = jax.random.normal(k_mut, (pop_size, n_var), dtype=X.dtype) * params.mut_sigma
        child_X = child_X + noise * (xu - xl)
        child_X = jnp.clip(child_X, xl, xu)

        if self.wp_mut_scale < 1.0:
            parent_wp = X[parent][:, wp_offset:]
            child_X = child_X.at[:, wp_offset:].set(
                parent_wp + self.wp_mut_scale * (child_X[:, wp_offset:] - parent_wp))

        t_p1 = X[p1][:, t_offset:wp_offset]
        t_p2 = X[p2][:, t_offset:wp_offset]
        perm1, perm2 = jnp.argsort(t_p1, axis=1), jnp.argsort(t_p2, axis=1)
        child_perm_ox = _ox_crossover_batched(k_ox, perm1, perm2)
        row_idx = jnp.arange(pop_size)[:, None]
        rank = jnp.broadcast_to(jnp.arange(n_nodes), (pop_size, n_nodes))
        t_ox = jnp.zeros((pop_size, n_nodes), dtype=X.dtype).at[row_idx, child_perm_ox].set(rank)
        do_ox = jax.random.uniform(k_ox_mask, (pop_size,)) < params.ox_prob
        t_blx = child_X[:, t_offset:wp_offset]
        child_X = child_X.at[:, t_offset:wp_offset].set(jnp.where(do_ox[:, None], t_ox, t_blx))

        child_mu, child_lam, child_rho = mu[parent], lam[parent], rho[parent]

        assign, cond_binary, _proj_branch_ga, t, wp0, psi0 = problem._extract_batch(child_X)

        # THE ACTUAL CHANGE: replace the crossed-over/mutated proj_branch
        # (discarded, see the comment above) with the exact DP-optimal
        # branch choice given this individual's own CURRENT wp0 -- computed
        # BEFORE local_refine runs, so local_refine's own AL residual/
        # routing computations use the correct branch throughout, not a
        # stale one patched in after the fact.
        proj_branch = _solve_branch_dp(problem, self._chain_order, self._chain_node_entry,
                                        self._chain_relevant_cols, wp0, params.x0, params.problem_params)

        wp_star, psi_star, off_mu, off_lam, off_rho = self.local_refine(
            wp0, psi0, assign, cond_binary, proj_branch, t, child_mu, child_lam, child_rho,
            params.x0, params.problem_params, params.anchor, params.cv_tol)
        off_X = _write_wp_batch_jax(problem, child_X, wp_star)
        off_X = _write_psi_batch_jax(problem, off_X, psi_star)
        off_X = _write_proj_branch_batch_jax(problem, off_X, proj_branch)

        off_assign, off_cond_binary, off_proj_branch, off_t, off_wp, off_psi = problem._extract_batch(off_X)
        off_wp = apply_projections(problem, off_wp, off_psi, off_proj_branch, params.problem_params,
                                    assign=off_assign, anchor=params.anchor)
        off_assign_eff, off_wp_eff_frozen, _off_wp_eff_live = apply_anchor(
            problem, off_assign, off_wp, params.anchor, params.x0)
        off_t = _routing_local_search_batched(
            problem, k_2opt, off_assign_eff, off_cond_binary, off_t, off_wp_eff_frozen,
            params.x0, params.anchor.node_active, self.n_2opt_trials,
            or_opt_prob=self.or_opt_prob, max_or_opt_seg_len=self.max_or_opt_seg_len)
        off_X = _write_t_batch_jax(problem, off_X, off_t)

        child_genome = _join_genome(off_X, off_mu, off_lam, off_rho)
        return child_genome, state

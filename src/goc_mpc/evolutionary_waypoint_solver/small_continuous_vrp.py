r"""`SmallContinuousVRPSolver`: `LamarckianGA` (lamarckian_ga.py) with the
DISCRETE parts of the problem -- branch selection (`proj_branch`) and agent
ASSIGNMENT (`assign`) -- replaced by exact enumeration/DP instead of letting
crossover/mutation evolutionarily search them, while keeping a real
population (so a bad discrete choice this generation isn't a dead end --
selection over the next generation's fresh, independently re-sampled/re-
solved choices is the recovery mechanism, not a single deterministic
solve-once-and-commit).

The TRUE problem this class is working towards an exact solve of, in full
(this is the target -- see "Current status" below for what's actually
implemented vs. still GA-searched):

    minimize_{A, B, Z}  t(n+1)

    for all i:                sum_j A(i, j) = 1
    for all v in V:            B(v) in {0, ..., b_v}
    for all i, j:              A(i, j) in {0, 1}
    for all j, (a, b):         Z^j_{ab} in {0, 1}
    for all v in V:             exists i in I_v : A(i, j) = 1
                                     ==> sum_{a} Z^j_{av} = 1
    for all j, v in V:         sum_a Z^j_{av} = sum_b Z^j_{vb}
    for all j:                 sum_v Z^j_{0v} = 1
    for all j:                 sum_v Z^j_{v,n+1} = 1
    for all j, v in V:         Z^j_{v0} = 0
    for all j, v in V:         Z^j_{n+1,v} = 0
    for all j, a, b (a != b):  Z^j_{ab} = 1 ==>
                                    t(b) >= t(a) + delta_{ab}(B(a), B(b)) + s_{ab}
    for all (a, b) in E:       t(a) <= t(b)
                                t(0) = 0

(`A`: assign, `B`: proj_branch, `Z`: per-agent routing/edge-selection,
`t`: node timing/precedence, `delta_{ab}(B(a), B(b))`: the branch-dependent
edge cost -- e.g. joint-space distance between node a's and node b's own
chosen IK solution, `s_{ab}`: a fixed per-edge service/setup time. `t` here
plays the same MTZ subtour-elimination role solver.py's own MILP baseline
uses.)

Current status -- what's exact vs. still GA-searched:
  - B (BRANCH, `proj_branch`): EXACT, `_solve_branch_dp` -- a shortest-path
    DP over the static node chain, one layer per chain node, one state per
    combined branch candidate (the Cartesian product over every projection
    writing to that node, see `_node_candidates` -- correct because
    different projections at one node write to disjoint `pinned_cols`,
    problem.apply_projections applies each independently). This is exactly
    `min sum delta_{ab}(B(a), B(b))` along the fixed chain, given `Z`/`A`
    fixed to their current (this individual's) values -- `s_{ab}` (a fixed
    per-edge constant, not yet modeled) and `A`'s own choice are the two
    remaining gaps between this and the true joint objective above.
  - A (ASSIGNMENT, `assign`): EXACT ENUMERATION + Boltzmann sampling,
    `_sample_assign` -- every one of the `n_agents ** n_variables` discrete
    combinations (typically single-digit, per this class' own scope) is
    scored by its REAL combined score (`_combined_score(F, CV, ...)`, this
    individual's own current wp/branch/t/cond_binary held fixed), then one
    is Gumbel-sampled per population member. Not a deterministic argmin --
    that would collapse the population's assignment diversity to one value
    every generation, exactly the local-minimum failure mode a population
    exists to avoid; sampling proportional to real quality keeps several
    live hypotheses while still being informed, not random-walk GA noise.
  - Z (ROUTING/edges) and `t` (ORDERING): still UNCHANGED from LamarckianGA
    -- OX-crossover + 2-opt/Or-opt local search on `t`, decoded into edges
    by kernel.py's own topological-sort/gate machinery, same as ever. Left
    alone because every scene built so far has `t` fully determined by
    `problem.hard_edges` (no real ordering freedom to search) -- converting
    this to an exact per-agent DP (Held-Karp over each agent's own node
    subset) is real future work, deliberately deferred until a scene
    actually exercises that freedom, per this class' own existing
    "scoped, not general" philosophy (see below).
  - `s_{ab}` (fixed edge service/setup time) and any coupling between A's
    choice and `Z`'s FEASIBILITY (rather than just its cost) beyond what
    `problem.hard_edges`/routing local search already encodes: not modeled
    yet.

Scope requirements (raise `NotImplementedError` rather than silently do
something wrong otherwise -- see `_build_static_chain`/`_build_assign_
combos`): `problem.hard_edges` must form a single total order over every
node (a genuinely single-chain, e.g. single-agent, graph); no projection
may pin a `var_agent_q(...)` row (`ProjectionEntry.owner_var_slot` --
ambient-space Part C's dynamic write target is a different axis entirely,
not combined with either DP here yet); the assignment axis must enumerate
to at most `max_assign_combos` combinations. A node MAY carry several
projections at once (e.g. a two-arm handoff node pinning both agents' own
analytic-IK branch) -- `_node_candidates` handles this via the Cartesian-
product generalization described above.
"""

import itertools

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
    _write_assign_batch_jax,
    _tournament_select_jax,
    _ox_crossover_batched,
    _evaluate_population_jax,
    _combined_score,
)

jax.config.update("jax_enable_x64", True)


def _build_static_chain(problem):
    """(chain_order, node_entries, relevant_cols): `chain_order` is a
    topological sort of `problem.hard_edges` over every node -- raises if
    it doesn't cover all `problem.n_nodes` (a genuinely branching/multi-
    agent graph, out of scope for now -- see this module's docstring). Same
    plain Kahn's-algorithm-over-a-possibly-redundant/transitive edge set
    this module's own diag_path_length.py validation script uses
    (ur5e_block_stacking's real `hard_edges` include transitive pairs like
    (2,0) AND (2,3) AND (0,3), not just adjacent-node pairs, so a stricter
    "every node has at most one successor" check would incorrectly reject
    this exact, already-validated scene).

    `node_entries`: node id -> the (possibly several) ProjectionEntry
    objects (problem.py) that write to it, for every node that has at
    least one -- raises if any projection pins a var_agent_q(...) row
    (ProjectionEntry.owner_var_slot -- ambient-space Part C's dynamic write
    target is a different axis entirely from this class' branch-selection
    DP, and combining them isn't supported yet). Several entries at the
    same node (e.g. a two-arm handoff node pinning both agents' own
    analytic-IK branch) are fine -- see _node_candidates.

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
    quantity (true for every validated scene so far -- always
    `agent_q[0:6]`, one range per agent at a multi-projection node); a
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

    node_entries = {}
    for entry in problem.projections:
        if entry.owner_var_slot is not None:
            raise NotImplementedError(
                "SmallContinuousVRPSolver doesn't support var_agent_q(...) "
                "(dynamic-owner) projections yet -- that's a different axis "
                "(WHICH agent's row gets written) from this class' branch-"
                "selection DP (WHAT value gets written)")
        node_entries.setdefault(entry.write_node, []).append(entry)

    relevant_cols = sorted({int(c) for entries in node_entries.values()
                             for entry in entries for c in entry.pinned_cols})
    return order, node_entries, relevant_cols


def _one_entry_candidates(problem, entry, wp0, params_arr):
    """`(pop, k, state_dim)` candidate rows for ONE projection entry, `k =
    entry.discrete_params` -- every one of its branch candidates, spliced
    into that row's `pinned_cols` (mirrors problem.apply_projections' own
    per-entry substitution, just evaluated at EVERY branch instead of only
    the GA/argmax-selected one -- reuses `entry.table` verbatim when set,
    the same O(1) gather apply_projections itself uses for a tabled entry).
    `entry.continuous_params` is assumed 0 here (real for every projection
    with a genuine discrete branch choice built so far, e.g. UR5e's
    analytic IK -- see projection.ProjOperator's own docstring); a future
    projection combining a real psi with discrete_params > 1 would only get
    an approximate (psi=0) candidate value here, not a fully exact one."""
    pop = wp0.shape[0]
    base_row = wp0[:, entry.write_node, :]
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


def _node_candidates(problem, nid, entries, wp0, params_arr):
    """`(pop, K, state_dim)` candidate rows for one chain node, plus the
    per-entry branch-count tuple `dims` used to decode a chosen flat index
    back into each entry's own branch (_solve_branch_dp). `entries` is the
    (possibly empty) list of every ProjectionEntry that writes to `nid`:
      - no entries -> K=1, just wp0's own current row (no discrete choice).
      - one entry -> exactly the original single-projection behaviour,
        K = entry.discrete_params.
      - several entries -> the CARTESIAN PRODUCT of every entry's own
        branch candidates, K = prod(discrete_params) -- correct because
        every projection writes to its OWN disjoint pinned_cols
        (apply_projections applies each entry independently; see its own
        docstring), so any combination of branch choices across different
        projections at the same node composes without conflict. K stays
        small in practice (e.g. two 8-branch analytic-IK pins at a handoff
        node -> 64), and is a static Python int (discrete_params is fixed
        at spec-build time), so this is a plain unrolled Python loop, not a
        traced shape."""
    pop = wp0.shape[0]
    base_row = wp0[:, nid, :]
    if not entries:
        return base_row[:, None, :], ()
    per_entry_rows = [_one_entry_candidates(problem, entry, wp0, params_arr) for entry in entries]
    dims = tuple(entry.discrete_params for entry in entries)
    if len(entries) == 1:
        return per_entry_rows[0], dims

    combined = []
    for combo in itertools.product(*[range(d) for d in dims]):
        row = base_row
        for entry, entry_rows, b in zip(entries, per_entry_rows, combo):
            cols = jnp.asarray(entry.pinned_cols)
            row = row.at[:, cols].set(entry_rows[:, b, :][:, cols])
        combined.append(row)
    return jnp.stack(combined, axis=1), dims


def _solve_branch_dp(problem, chain_order, chain_node_entries, relevant_cols, wp0, x0, params_arr):
    """Exact per-individual DP over `chain_order`'s static node chain: picks
    the branch index at every discrete-choice projection minimizing total
    L2 path length through the chain's candidate rows -- a plain
    shortest-path-over-a-layered-graph, the same algorithm diag_path_
    length.py's own ground truth uses, just run once per generation inside
    `_ask` instead of after the fact in numpy. A plain Python loop over
    `chain_order` (a small, spec-build-time-fixed list) unrolls this at
    trace time -- no jax.lax.scan/backtrack bookkeeping needed, since
    successive steps' candidate counts (`K`) can differ freely.

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
    always index 0 regardless of its value, so this is already correct.
    A node with SEVERAL projections decodes its combined flat choice back
    into each entry's own branch via `jnp.unravel_index` (the inverse of
    `_node_candidates`' `itertools.product` enumeration -- both use
    default C/row-major order, so they agree)."""
    pop = wp0.shape[0]
    # dtype=int32 explicitly: relevant_cols is empty for a scene with no
    # projections at all (e.g. pick_place_task, whose Pick/Place targets
    # are plain FK-residual equality constraints, not analytic-IK
    # ProjOperators) -- jnp.asarray([]) defaults to float64 with nothing
    # else to infer a dtype from, and float-indexing x0[cols] below raises.
    # An empty int array here is correct either way: every node then has
    # entries=[] too (no projection anywhere in the problem), so
    # _node_candidates' own K=1 trivial-candidate path is what actually
    # makes the whole DP a no-op, not this array's shape.
    cols = jnp.asarray(relevant_cols, dtype=jnp.int32)
    rows_by_node = []
    for nid in chain_order:
        entries = chain_node_entries.get(nid, [])
        rows, dims = _node_candidates(problem, nid, entries, wp0, params_arr)
        rows_by_node.append((entries, dims, rows))

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
    for i, (entries, dims, _rows) in enumerate(rows_by_node):
        if not entries:
            continue
        per_entry_idx = (choices[i],) if len(entries) == 1 else jnp.unravel_index(choices[i], dims)
        for entry, branch_idx in zip(entries, per_entry_idx):
            onehot = jax.nn.one_hot(branch_idx, entry.discrete_params)
            proj_branch = proj_branch.at[:, entry.branch_slice].set(onehot)
    return proj_branch


def _build_assign_combos(problem, max_assign_combos):
    """`(n_combos, n_variables, n_agents)` one-hot array -- every possible
    discrete `assign` combination, brute-force enumerated (the ASSIGNMENT
    axis this module's own docstring calls "the genuinely hard axis...
    but small in every scene built so far"). `n_combos = n_agents **
    n_variables`; raises rather than silently paying for a combinatorial
    blowup if that exceeds `max_assign_combos` (a real scene with a large
    assignment space needs a different algorithm -- Held-Karp-style or
    branch-and-bound over THIS axis specifically -- not brute force).
    `n_variables == 0` (no assignable variable at all, e.g. any single-
    agent scene) degenerates to exactly one, empty combo -- this class'
    assignment handling is then a no-op, matching its original single-
    agent-only behaviour exactly."""
    n_variables, n_agents = problem.n_variables, problem.n_agents
    n_combos = n_agents ** n_variables if n_variables > 0 else 1
    if n_combos > max_assign_combos:
        raise NotImplementedError(
            f"SmallContinuousVRPSolver enumerates every assign combination "
            f"({n_agents}**{n_variables} = {n_combos}) -- exceeds "
            f"max_assign_combos={max_assign_combos}; this scene's assignment axis "
            "is too large for brute-force enumeration (see this module's docstring)")
    if n_variables == 0:
        return jnp.zeros((1, 0, n_agents))
    combos = jnp.asarray(list(itertools.product(range(n_agents), repeat=n_variables)))
    return jax.nn.one_hot(combos, n_agents)  # (n_combos, n_variables, n_agents)


def _sample_assign(problem, combos, child_X, key, params, temperature):
    """Boltzmann/Gumbel-samples ONE `assign` combo per population member
    from `combos` (every discrete combination, enumerated once at
    construction -- see _build_assign_combos), weighted by each combo's
    own REAL combined score (`_combined_score(F, CV, params.w,
    params.cv_tol)` -- the SAME F/CV every other selection/ranking step in
    this module already uses, not a hand-rolled cheap proxy) evaluated
    with this individual's OWN current branch/ordering/wp (`child_X`) held
    fixed and just the assign block swapped per combo. This is what keeps
    the ASSIGNMENT axis a real population-diversity axis rather than
    collapsing to a single deterministic argmin every generation: a
    combo that scores poorly for one individual this generation can still
    be resampled (by a different individual, or the same one after
    tournament selection picks a different parent) next generation, so a
    bad early commitment isn't a dead end the way a single alternating
    exact-solve-and-fix would be. `temperature` (this class' own
    `assign_temperature` constructor kwarg) trades exploration (high) for
    exploitation (low, -> argmin) directly.

    `n_combos == 1` (no real assignable variable, see _build_assign_combos)
    always selects that one combo -- an inert no-op costing one extra,
    trivially cheap batched evaluation."""
    pop, n_var = child_X.shape
    n_combos = combos.shape[0]
    n_assign_vars = problem.n_assign_vars
    combo_flat = combos.reshape(n_combos, n_assign_vars)
    trial_X = jnp.broadcast_to(child_X[:, None, :], (pop, n_combos, n_var))
    trial_X = trial_X.at[:, :, :n_assign_vars].set(
        jnp.broadcast_to(combo_flat[None], (pop, n_combos, n_assign_vars)))
    F, CV = _evaluate_population_jax(problem, trial_X.reshape(pop * n_combos, n_var),
                                      params.x0, params.problem_params, params.anchor)
    F, CV = F.reshape(pop, n_combos), CV.reshape(pop, n_combos)
    S = _combined_score(F, CV, params.w, params.cv_tol)
    chosen = jax.random.categorical(key, -S / temperature, axis=-1)  # (pop,)
    return combos[chosen]  # (pop, n_variables, n_agents)


class SmallContinuousVRPSolver(LamarckianGA):
    """See module docstring. Same construction/Params/State/`_tell` as
    `LamarckianGA` (unchanged) -- only `_ask` differs, in how `assign` and
    `proj_branch` are produced."""

    def __init__(self, population_size, solution, problem, assign_temperature=1.0,
                 max_assign_combos=4096, **kwargs):
        super().__init__(population_size, solution, problem, **kwargs)
        self._chain_order, self._chain_node_entries, self._chain_relevant_cols = _build_static_chain(problem)
        self._assign_combos = _build_assign_combos(problem, max_assign_combos)
        self._assign_temperature = assign_temperature

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

        key, k_p1, k_p2, k_cx_mask, k_cx_alpha, k_mut, k_ox_mask, k_ox, k_2opt, k_assign = \
            jax.random.split(key, 10)

        p1 = _tournament_select_jax(k_p1, S, pop_size, k=self.tournament_k)
        p2 = _tournament_select_jax(k_p2, S, pop_size, k=self.tournament_k)
        parent = jnp.where(S[p1] < S[p2], p1, p2)

        # Same BLX-alpha crossover + Gaussian mutation as LamarckianGA, over
        # the WHOLE X -- including assign's and proj_branch's own slices,
        # harmlessly: whatever crossover/mutation leaves there is about to
        # be entirely overwritten by the exact sample/DP below, so there's
        # no need to carve them out of this generic operator (it's cheaper
        # to let it run and discard the result than to special-case the
        # slices away).
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

        # THE ACTUAL CHANGE (1/2): replace the crossed-over/mutated assign
        # (discarded, see the comment above) with a Boltzmann/Gumbel sample
        # over every REAL discrete combination, scored by this individual's
        # own actual combined score -- exact enumeration + informed
        # sampling, not GA noise, while still keeping the population's
        # ability to recover from a bad sample (see _sample_assign's own
        # docstring).
        assign_sampled = _sample_assign(problem, self._assign_combos, child_X, k_assign, params,
                                         self._assign_temperature)
        child_X = _write_assign_batch_jax(problem, child_X, assign_sampled)

        assign, cond_binary, _proj_branch_ga, t, wp0, psi0 = problem._extract_batch(child_X)

        # THE ACTUAL CHANGE (2/2): replace the crossed-over/mutated
        # proj_branch (discarded, see the comment above) with the exact
        # DP-optimal branch choice given this individual's own CURRENT wp0
        # -- computed BEFORE local_refine runs, so local_refine's own AL
        # residual/routing computations use the correct branch throughout,
        # not a stale one patched in after the fact.
        proj_branch = _solve_branch_dp(problem, self._chain_order, self._chain_node_entries,
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

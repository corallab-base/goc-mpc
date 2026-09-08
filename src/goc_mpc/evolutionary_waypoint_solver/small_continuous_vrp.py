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
edge cost -- `problem.edge_cost_fn` between node a's and node b's own chosen
IK solution (the agent's obstacle-aware field, the SAME one kernel.py's
routing objective is built from), `s_{ab}`: a fixed per-edge service/setup
time. `t` here plays the same MTZ subtour-elimination role solver.py's own
MILP baseline uses.)

Current status -- what's exact vs. still GA-searched:
  - B (BRANCH, `proj_branch`): EXACT, `_solve_branch_dp` -- one independent
    shortest-path per "track" (an agent, or an object; see
    `_build_static_chain`'s `track_plan`) over that track's own ordered
    projection nodes, from the real depot state, each hop priced by that
    track's `problem.edge_cost_fn`. The tracks are separable (disjoint
    columns, per-agent cost fns), so summing the per-track minima is exactly
    `min sum delta_{ab}(B(a), B(b))` given `Z`/`A` fixed to this
    individual's values -- `s_{ab}` (a fixed per-edge constant, not yet
    modeled) and `A`'s own choice are the two remaining gaps between this
    and the true joint objective above.
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
to at most `max_assign_combos` combinations; a given track's projections
must all pin the same column set. A node MAY carry several projections at
once (e.g. a two-arm handoff node pinning both agents' own analytic-IK
branch) -- each is simply its own track in the per-track DP.
"""

import itertools

import jax
import jax.numpy as jnp

from .lamarckian_ga import LamarckianGA
from .evosax_ga import _split_genome, _join_genome
from .kernel import _euclidean_edge_cost
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
    """(chain_order, node_entries, track_plan): `chain_order` is a
    topological sort of `problem.hard_edges` over every node -- raises if
    it doesn't cover all `problem.n_nodes` (a genuinely branching/multi-
    agent graph, out of scope for now -- see this module's docstring). Same
    plain Kahn's-algorithm-over-a-possibly-redundant/transitive edge set
    (ur5e_block_stacking's real `hard_edges` include transitive pairs like
    (2,0) AND (2,3) AND (0,3), not just adjacent-node pairs, so a stricter
    "every node has at most one successor" check would incorrectly reject
    this exact, already-validated scene).

    `node_entries`: node id -> the (possibly several) ProjectionEntry
    objects (problem.py) that write to it, for every node that has at
    least one -- raises if any projection pins a var_agent_q(...) row
    (ProjectionEntry.owner_var_slot -- ambient-space Part C's dynamic write
    target is a different axis entirely from this class' branch-selection
    DP, and combining them isn't supported yet).

    `track_plan`: one entry per "track" -- an agent (its config columns) or
    an object -- that any projection pins, as
    `(cost_fn, cols, [ProjectionEntry, ...])`:

    - `cols`: the ABSOLUTE columns this track's projections write (identical
      across a track's own entries -- raises otherwise; a multi-arm handoff
      node just contributes one entry per arm, each its own track).
    - `[ProjectionEntry, ...]`: this track's entries, ordered by their
      node's position in `chain_order` -- the ordered stops on this track's
      own path.
    - `cost_fn`: how a step on this track is priced. For an agent track,
      `problem.edge_cost_fn` for that agent (its own obstacle-aware field,
      the SAME one kernel.py's routing objective uses) -- so branch
      selection agrees with the continuous objective instead of using a
      plain-Euclidean surrogate. Euclidean for an object track, or whenever
      `problem.edge_cost_fn` is `None`.

    `_solve_branch_dp` then runs one independent shortest-path per track
    over its own stops (from the real depot state `params.x0`), which is
    exact: the per-track costs are separable (disjoint columns, per-agent
    cost fns), so the joint minimum is the sum of the per-track minima, and
    it needs no assumption about how the tracks interleave in `chain_order`."""
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

    if any(a.write_cols & b.read_cols
           for a in problem.projections for b in problem.projections if a is not b):
        raise NotImplementedError(
            "SmallContinuousVRPSolver doesn't support chained projections (one "
            "projection's `reads` column pinned by another) -- its per-track "
            "shortest-path DP assumes the tracks are independent")

    node_entries = {}
    for entry in problem.projections:
        if entry.owner_var_slot is not None:
            raise NotImplementedError(
                "SmallContinuousVRPSolver doesn't support var_agent_q(...) "
                "(dynamic-owner) projections yet -- that's a different axis "
                "(WHICH agent's row gets written) from this class' branch-"
                "selection DP (WHAT value gets written)")
        node_entries.setdefault(entry.write_node, []).append(entry)

    order_index = {nid: i for i, nid in enumerate(order)}
    n_agents, dim = problem.n_agents, problem.dim
    ecf = getattr(problem, "edge_cost_fn", None)

    by_track = {}  # track key -> list of ProjectionEntry
    for entry in problem.projections:
        cols = tuple(int(c) for c in entry.pinned_cols)
        c0 = min(cols)
        key = ("agent", c0 // dim) if c0 < n_agents * dim else ("object", c0)
        by_track.setdefault(key, []).append(entry)

    track_plan = []
    for key, entries in by_track.items():
        entries = sorted(entries, key=lambda e: order_index[e.write_node])
        cols = tuple(int(c) for c in entries[0].pinned_cols)
        if any(tuple(int(c) for c in e.pinned_cols) != cols for e in entries):
            raise NotImplementedError(
                f"SmallContinuousVRPSolver branch DP: track {key} has projections "
                "pinning different column sets across nodes; the per-track path "
                "cost needs one fixed column set per track")
        if key[0] == "agent" and isinstance(ecf, (list, tuple)):
            cost_fn = ecf[key[1]]
        elif key[0] == "agent" and callable(ecf):
            cost_fn = ecf
        else:
            cost_fn = _euclidean_edge_cost
        track_plan.append((cost_fn, jnp.asarray(cols, dtype=jnp.int32), tuple(entries)))
    return order, node_entries, track_plan


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


def _apply_edge_cost(cost_fn, a, b):
    """`cost_fn(a, b) -> scalar` broadcast over the leading dims of `a`/`b`
    (each `(..., w)`); returns an array of the broadcast leading shape. Same
    `jax.vmap(cost_fn)` contract kernel.py's `_one_agent_route` uses, so a
    per-agent NTField closure priced here costs exactly what it does there."""
    lead = jnp.broadcast_shapes(jnp.shape(a)[:-1], jnp.shape(b)[:-1])
    w = jnp.shape(a)[-1]
    a = jnp.broadcast_to(a, lead + (w,)).reshape(-1, w)
    b = jnp.broadcast_to(b, lead + (w,)).reshape(-1, w)
    return jax.vmap(cost_fn)(a, b).reshape(lead)


def _solve_branch_dp(problem, track_plan, wp0, x0, params_arr):
    """Exact per-individual branch selection: for every projection with a
    discrete branch choice (e.g. UR5e analytic IK's 8 solutions), pick the
    branch minimizing this individual's total routed path cost -- run once
    per generation inside `_ask`, BEFORE local_refine, so local_refine's own
    AL/routing passes see the right branch throughout rather than a stale
    one patched in after.

    One INDEPENDENT shortest-path per track (`_build_static_chain`'s
    `track_plan`): a track's stops are its own ordered projection nodes, its
    path starts at the real depot state `x0`, and each hop is priced by that
    track's `cost_fn` -- an agent track's own obstacle-aware field, the SAME
    one kernel.py's routing objective uses, so branch selection agrees with
    the continuous objective instead of a plain-Euclidean surrogate;
    Euclidean for an object track or when `problem.edge_cost_fn` is `None`.
    The tracks are separable (disjoint columns, per-agent cost fns), so
    solving each in isolation and summing is exactly `min sum delta_{ab}`
    over the joint branch choice -- no assumption about how tracks interleave
    in the chain, and (unlike the earlier union-column L2 metric) no free
    wp column from an unrelated node leaking GA-search noise into the
    decision, since each hop reads only this track's own pinned columns.

    Including the `x0 -> first-stop` hop matters: omitting it (a real, fixed
    bug in an earlier version) ignored the cost of the first move away from
    the actual start state, so the DP optimized a path good in isolation but
    not the shortest one INCLUDING getting there from `x0`.

    Returns `(pop, n_branch)`: a one-hot per projection's own `branch_slice`
    (apply_projections' argmax decode recovers the chosen branch); a
    projection with no real choice (`discrete_params == 1`) gets an all-zero
    width-1 slice, whose argmax is 0 regardless -- already correct."""
    pop = wp0.shape[0]
    proj_branch = jnp.zeros((pop, problem.n_branch))

    for cost_fn, cols, entries in track_plan:
        # (pop, k_j, w) candidate positions for this track at each of its
        # ordered stops -- k_j = entries[j].discrete_params.
        stops = [_one_entry_candidates(problem, e, wp0, params_arr)[:, :, cols]
                 for e in entries]

        depot = x0[cols]  # (w,)
        cost = _apply_edge_cost(cost_fn, depot[None, None, :], stops[0])  # (pop, k0)
        backptrs = []
        for j in range(1, len(stops)):
            a = stops[j - 1][:, :, None, :]  # (pop, k_prev, 1, w)
            b = stops[j][:, None, :, :]      # (pop, 1, k_cur, w)
            total = cost[:, :, None] + _apply_edge_cost(cost_fn, a, b)  # (pop, k_prev, k_cur)
            cost = jnp.min(total, axis=1)
            backptrs.append(jnp.argmin(total, axis=1))  # (pop, k_cur)

        choices = [None] * len(stops)
        choices[-1] = jnp.argmin(cost, axis=1)  # (pop,)
        for j in range(len(stops) - 2, -1, -1):
            choices[j] = jnp.take_along_axis(backptrs[j], choices[j + 1][:, None], axis=1)[:, 0]

        for entry, branch_idx in zip(entries, choices):
            proj_branch = proj_branch.at[:, entry.branch_slice].set(
                jax.nn.one_hot(branch_idx, entry.discrete_params))

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
        self._chain_order, self._chain_node_entries, self._chain_track_plan = _build_static_chain(problem)
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
        proj_branch = _solve_branch_dp(problem, self._chain_track_plan, wp0,
                                        params.x0, params.problem_params)

        wp_star, psi_star, off_mu, off_lam, off_rho = self.local_refine(
            wp0, psi0, assign, cond_binary, proj_branch, t, child_mu, child_lam, child_rho,
            params.x0, params.problem_params, params.anchor, params.cv_tol)
        off_X = _write_wp_batch_jax(problem, child_X, wp_star)
        off_X = _write_psi_batch_jax(problem, off_X, psi_star)
        off_X = _write_proj_branch_batch_jax(problem, off_X, proj_branch)

        off_assign, off_cond_binary, off_proj_branch, off_t, off_wp, off_psi = problem._extract_batch(off_X)
        off_wp = apply_projections(problem, off_wp, off_psi, off_proj_branch, params.problem_params,
                                    assign=off_assign, anchor=params.anchor,
                                    cond_binary=off_cond_binary, t=off_t, node_active=params.anchor.node_active, x0=params.x0)
        off_assign_eff, off_wp_eff_frozen, _off_wp_eff_live = apply_anchor(
            problem, off_assign, off_wp, params.anchor, params.x0)
        off_t = _routing_local_search_batched(
            problem, k_2opt, off_assign_eff, off_cond_binary, off_t, off_wp_eff_frozen,
            params.x0, params.anchor.node_active, self.n_2opt_trials,
            or_opt_prob=self.or_opt_prob, max_or_opt_seg_len=self.max_or_opt_seg_len)
        off_X = _write_t_batch_jax(problem, off_X, off_t)

        child_genome = _join_genome(off_X, off_mu, off_lam, off_rho)
        return child_genome, state

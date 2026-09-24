"""`make_dp_master_jax(problem, ...)` -- a build-once / call-many form of
`dp_master.solve_dp_master` whose returned `run(...)` is a single
`jax.jit`ted function.

`solve_dp_master` loops in Python over `(assignment combo, aux combo, linear
extension, branch combo)` and scores each. Here every one of those discrete
axes is enumerated EXHAUSTIVELY at build time (bounded by the same `max_*`
caps `solve_dp_master` raises on):

  * assignment: all `n_agents ** n_variables` slot->agent maps (not just the
    anchor-free ones -- an anchored slot is masked at run time);
  * aux: all `2 ** n_cond_vars`;
  * order: all linear extensions of the HARD precedence graph (the gated
    edges only ever ADD constraints, so every feasible order under any
    `(A, aux)` is one of these -- run time masks the ones a resolved
    gated edge would violate);
  * branch: all `prod_e discrete_params` combos over the multi-branch
    projection entries.

The returned `run` takes the cycle's continuous data + anchor arrays, masks
the infeasible grid cells, resolves + scores the rest in one vmapped pass,
and returns the winner as plain arrays. `solve_dp_master` stays the
reference implementation and the A/B oracle (see
`examples/test_dp_master_jax.py`).

`make_dp_master_jax(problem, ...)` returns a `_DpMasterJax`; call it (or its
`run_vec`) with the cycle's continuous data. `run_topk(k, ...)` returns the
`k` best discrete skeletons (distinct `(assignment, aux)`, ascending by
objective) instead of just the winner; `skeleton_grid_fn(k)` is the same
computation as a jittable pure function returning genome tensors, for
`SmallContinuousVRPSolver._ask` to seed its population from.

The grid search -- `_score_grid_core`: resolve every `(assignment, aux,
ordering)` cell via one batched `apply_projections` per projection layer,
build dense EDGE/DEPOT, price each with a per-agent branch Viterbi
(`_agent_route_dp`, avg/minmax -- returns the chosen branch per node too, not
just the cost) or the coupled makespan forward pass (`_makespan_arr_jax`,
over an explicit `BC` branch-combo axis), bias that cost by each cell's
`CV_proj` (did the now-known winning branch's spliced waypoints actually
satisfy their own analytic projections, e.g. an out-of-reach analytic IK --
same `cost + 1e6*max(0, CV_proj - cv_proj_tol)` hard-score convention
`SmallContinuousVRPSolver.reseed` uses), reduce `min` over orderings so each
skeleton has one (biased) score, then `lax.top_k` -- is one `jax.jit` per
`k`. Only the winner -> `solve_dp_master` dict reconstruction (routes,
times) stays in Python (`_reconstruct_cell`), on the single winning cell.

Requires a jnp-traceable `edge_cost_fn`. Needs no `node_candidates` -- the
grid search resolves everything through `apply_projections`, so the
non-full / full-resolve distinction `solve_dp_master` draws does not apply
here.
"""

import heapq
import itertools

import numpy as np

from .structure import (
    build_depot_cost_table, build_edge_cost_table, entry_owner, node_instances)
from .cpsat_model import _agent_sliced_cost_fn
from .coupled_dp import build_coupled_dp
from .dp_master import (
    _BIG, _agent_owned_nodes, _ensure_x64, _ext_full_avg_minmax_score,
    _ext_full_makespan_score, _ext_nonfull_avg_minmax, _ext_nonfull_makespan,
    _extensions_matrix, _full_ext_resolve, _inst_key, _linear_extensions,
    _needs_full_resolve,
    _makespan_arr_jax, _resolve_precedence, _resolve_schedule_wp,
)


def _agent_route_dp(order, act, EDGE_j, DEPOT_j):
    """One agent's minimum routed-path cost AND the branch index it picks at
    each node it owns.

    `order` (N,) is the full node visiting order; `act` (N,) bool flags the
    stops this agent owns. `EDGE_j` (N,N,K,K) / `DEPOT_j` (N,K) are this
    agent's branch-resolved edge / depot costs -- entry `[u, v, bu, bv]` is
    the cost of hopping from node `u`'s branch `bu` to node `v`'s branch `bv`
    (invalid branch slots already sit at `_BIG`, so they never win an argmin).

    A Viterbi over the K-branch lattice along the agent's own stops, carrying
    backpointers so the chosen branch per node comes back too, not just the
    cost (`dp_master._viterbi_cost_jax` returns cost only -- fine for its
    scalar dispatch, but the grid needs the full discrete solution to seed a
    population). `branch_of_node` is 0 at every node this agent does not own
    and at every owned node with no real branch choice (K_valid == 1)."""
    import jax.numpy as jnp
    from jax import lax
    K = EDGE_j.shape[-1]
    N = order.shape[0]

    def fwd(carry, i):
        cost, prev, seen = carry
        v, a = order[i], act[i]
        first = a & (~seen)
        trans = cost[:, None] + EDGE_j[prev, v]              # (K_prev, K_v)
        edge_c = jnp.min(trans, axis=0)                      # (K_v,)
        bp = jnp.argmin(trans, axis=0).astype(jnp.int32)     # (K_v,) best prev-branch
        new = jnp.where(first, DEPOT_j[v], edge_c)
        return (jnp.where(a, new, cost), jnp.where(a, v, prev), seen | a), (bp, a, v)

    (cost, _, seen), (BP, A, V) = lax.scan(
        fwd, (jnp.zeros(K), order[0], jnp.bool_(False)), jnp.arange(N))

    def bwd(cur_b, i):
        a = A[i]
        return jnp.where(a, BP[i, cur_b], cur_b), jnp.where(a, cur_b, 0)

    _, b_rev = lax.scan(bwd, jnp.argmin(cost).astype(jnp.int32), jnp.arange(N)[::-1])
    b_fwd = b_rev[::-1]                                      # branch chosen at each step
    branch_of_node = jnp.zeros(N, jnp.int32).at[V].set(jnp.where(A, b_fwd, 0))
    return jnp.where(seen, jnp.min(cost), 0.0), branch_of_node


def _order_dependent_rows(problem):
    """The projections writing an AGENT column whose candidate rows depend on
    something the coupled DP decides, as `[(index, write_node, reason), ...]`
    -- empty when the coupled backend is safe.  Built on
    `structure.projection_dependencies`, which traces each entry's inputs
    transitively and says exactly what they depend on:

      * the VISITING ORDER, through a gated pin whose gate can actually change
        value.  A gate is evaluated over every rank it can really be handed,
        anchors included; one that never changes (e.g. a stationary-object pin
        whose hold span hard precedence already fixes) is not a source of
        order dependence, however it reads.
      * a ROBOT CONFIGURATION, through an agent-column read (e.g. a rigid-carry
        pin, `obj_u + FK(q_v) - FK(q_u)`) -- i.e. on another node's branch.

    The coupled backend resolves rows once per (assignment, aux) against a
    single representative order and the warm-start robot rows, so either
    dependence would price the wrong rows.  Dependence on the ASSIGNMENT
    (a dynamic `var_agent_q` pin) is fine: the assignment is a grid axis and
    every cell is resolved with its own."""
    from .structure import projection_dependencies
    agent_hi = problem.n_agents * problem.dim
    out = []
    for i, (e, d) in enumerate(zip(problem.projections, projection_dependencies(problem))):
        if not any(c < agent_hi for (_n, c) in e.write_cols):
            continue
        why = []
        if d.varying_gates:
            nodes = "the whole order" if d.order_nodes is None else sorted(d.order_nodes)
            why.append(f"the order of nodes {nodes} via gated pin(s) {list(d.varying_gates)}")
        if d.robot_reads:
            why.append(f"robot configuration at node(s) "
                       f"{sorted({n for (n, _c) in d.robot_reads})}")
        if why:
            out.append((i, int(e.write_node), " and ".join(why)))
    return out


def _one_topo_order(n_nodes, hard_edges):
    """One topological order of the hard precedence edges (Kahn, lowest id
    first so it is deterministic).  The coupled backend chooses the real
    visiting order itself, so it needs a single representative order only --
    to feed `apply_projections` a `t`, which it is fenced to never read --
    and never enumerates the factorially many others."""
    succ = {i: [] for i in range(n_nodes)}
    indeg = [0] * n_nodes
    for (u, v) in hard_edges:
        succ[u].append(v)
        indeg[v] += 1
    q = [i for i in range(n_nodes) if indeg[i] == 0]
    heapq.heapify(q)
    out = []
    while q:
        u = heapq.heappop(q)
        out.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(q, v)
    if len(out) != n_nodes:
        raise ValueError("dp_master_jax: the hard precedence edges contain a "
                         "cycle -- no schedule exists")
    return out


class _DpMasterJax:
    """Holds the exhaustive static enumeration for one `problem` + objective
    + caps. `run` (piece 1: Python; later: jitted) consumes a cycle's
    continuous data against it."""

    def __init__(self, problem, objective, edge_cost_fn, max_assign_combos,
                 max_orders, max_branch_combos, cv_proj_bias=False, cv_proj_tol=1e-4,
                 route_backend="enumerate", max_coupled_structures=512,
                 coupled_kwargs=None, lazy_kwargs=None):
        # "enumerate" (default): all linear extensions x, for makespan, all
        # branch combos -- the original path.  "coupled": coupled_dp.py's
        # exact DP chooses order AND branch per (assignment, aux), so neither
        # axis is enumerated at all; `max_orders` / `max_branch_combos` stop
        # binding.  See `_coupled_host`.
        # "lazy": lazy_dp.py's order-aware engine -- resolves every node's
        # candidate rows when the DP schedules it, so pins whose value
        # depends on the order or on an upstream branch are priced exactly.
        self.route_backend = route_backend
        self._coupled = route_backend == "coupled"
        self._lazy_backend = route_backend == "lazy"
        if route_backend not in ("enumerate", "coupled", "lazy"):
            raise ValueError(f"unknown route_backend {route_backend!r}")
        if self._lazy_backend and objective != "makespan":
            raise NotImplementedError(
                "dp_master_jax: route_backend='lazy' is a makespan DP -- "
                f"objective={objective!r} is already exact and cheap on the "
                "per-agent Viterbi the enumerate backend uses")
        if self._coupled:
            if objective != "makespan":
                raise NotImplementedError(
                    "dp_master_jax: route_backend='coupled' is a makespan DP "
                    f"-- objective={objective!r} is already exact and cheap on "
                    "the per-agent Viterbi the enumerate backend uses")
            bad = _order_dependent_rows(problem)
            if bad:
                listed = ", ".join(f"#{i} @node {n} ({why})" for i, n, why in bad)
                raise NotImplementedError(
                    "dp_master_jax: route_backend='coupled' resolves each "
                    "node's candidate rows once per (assignment, aux), so they "
                    "must not depend on what the DP itself decides -- the "
                    "visiting order or another node's robot configuration. "
                    f"These agent-column projections do: {listed}")
        self._coupled_cache = {}
        self._coupled_kwargs = dict(coupled_kwargs or {})
        self.max_coupled_structures = int(max_coupled_structures)
        self.coupled_front = 0
        self.coupled_exact = True
        self._prepped = None
        self._jitted = {}
        self.problem = problem
        self.objective = objective
        self.max_orders = max_orders
        self.max_branch_combos = max_branch_combos
        # Opt-in (default off): biases `_score_grid_core`'s top-k selection
        # away from cells whose analytic projections didn't actually hold
        # (`CV_proj`), same hard-score convention `SmallContinuousVRPSolver.
        # reseed` uses (F + 1e6*max(0, CV_total - tol)) -- that class turns
        # this on and threads its own `reseed_cv_tol` in as `cv_proj_tol` so
        # one number governs both places. Off by default so `run_vec`/
        # `run_topk` stay a pure-cost equivalence oracle against
        # `solve_dp_master` for callers that want exactly that (e.g.
        # logic_based_benders_solver/mpc.py, test_dp_master_jax.py).
        self.cv_proj_bias = bool(cv_proj_bias)
        self.cv_proj_tol = float(cv_proj_tol)
        n_nodes = problem.n_nodes
        n_agents = problem.n_agents
        n_var = problem.n_variables
        n_cond = problem.n_cond_vars
        self.n_nodes, self.n_agents = n_nodes, n_agents
        self.n_var, self.n_cond = n_var, n_cond

        self.instances = node_instances(problem)
        self.ordering_edges = list(problem.ordering_edges)
        ecf = edge_cost_fn if edge_cost_fn is not None else getattr(problem, "edge_cost_fn", None)
        self.ecf = ecf
        self.sliced_of = {j: _agent_sliced_cost_fn(ecf, j, problem.dim) for j in range(n_agents)}

        proj = list(problem.projections)
        self.layer0 = tuple(e for e in proj if e.discrete_params == 1)
        self.branched = [e for e in proj if e.discrete_params > 1]
        self.branch_static = [e.owner_var_slot is None for e in self.branched]
        self.branch_cols = [np.asarray(sorted(int(c) for c in e.pinned_cols), dtype=int)
                            for e in self.branched]
        for e in self.branched:
            if self._lazy_backend:
                break      # lazy_dp resolves a reader after its writer's branch
            others = set().union(*(o.write_cols for o in self.branched if o is not e)) \
                if len(self.branched) > 1 else set()
            if e.read_cols & others:
                raise NotImplementedError(
                    "dp_master_jax: a multi-branch projection reads a column "
                    "another multi-branch projection pins -- coupled branch DP "
                    "not implemented")
        # branched entries sharing a write_node (a two-arm handoff): each
        # arm's branch is tracked per (node, OWNER), so the two owners must
        # be GUARANTEED distinct -- either static-and-different, or forced
        # apart by a `problem.categorical_ne` constraint. If not, raise (the
        # graph needs the inequality) rather than silently pruning combos.
        _wn_share = {}
        for bi, e in enumerate(self.branched):
            _wn_share.setdefault(int(e.write_node), []).append(bi)
        self._shared_wn = [g for g in _wn_share.values() if len(g) > 1]

        # -- assignment combos over the free slots, minus the ones a
        #    `categorical_ne` constraint forbids ------------------------
        n_assign = n_agents ** n_var if n_var else 1
        if n_assign > max_assign_combos:
            raise NotImplementedError(
                f"dp_master_jax: {n_agents}**{n_var} = {n_assign} assignment "
                f"combos > max_assign_combos={max_assign_combos}")
        ne_pairs = [(int(a), int(b)) for (a, b) in getattr(problem, "categorical_ne", ())]
        _combos = [c for c in itertools.product(range(n_agents), repeat=n_var)
                   if not any(c[a] == c[b] for (a, b) in ne_pairs)] if n_var else [()]
        if not _combos:
            raise NotImplementedError(
                f"dp_master_jax: categorical_ne={ne_pairs} rules out every "
                "assignment combo")
        self.A = (np.array(_combos, dtype=np.int32)
                  if n_var else np.zeros((1, 0), dtype=np.int32))            # (NC, n_var)

        # every surviving combo must keep the owners of a shared branched
        # node apart -- the grid tracks one branch per (node, owner), so two
        # entries there landing on the same agent would need that agent's
        # branch product.
        for grp in ([] if self._lazy_backend else self._shared_wn):
            wnn = int(self.branched[grp[0]].write_node)
            slots = sorted({self.branched[bi].owner_var_slot for bi in grp
                            if self.branched[bi].owner_var_slot is not None})
            for ov in self.A:
                owners = [entry_owner(problem, self.branched[bi], np.asarray(ov, int))
                          for bi in grp]
                if len(set(owners)) == len(owners):
                    continue
                hint = (f"add a categorical_ne constraint between slots {slots}"
                        if slots else
                        "these projections target the same fixed agent")
                raise NotImplementedError(
                    f"dp_master_jax: {len(grp)} multi-branch projections write "
                    f"node {wnn} but can share an owner (e.g. assignment "
                    f"{list(map(int, ov))}); {hint} so their owners are "
                    "guaranteed distinct")
        self.AUX = (np.array(list(itertools.product((0, 1), repeat=n_cond)), dtype=np.int32)
                    if n_cond else np.zeros((1, 0), dtype=np.int32))         # (NA, n_cond)
        self.NC, self.NA = self.A.shape[0], self.AUX.shape[0]

        # -- hard-graph linear extensions ------------------------------
        self.hard_edges = [(int(u), int(v)) for (u, v, g) in self.ordering_edges if g is None]
        self.gated = [(int(u), int(v), g) for (u, v, g) in self.ordering_edges if g is not None]
        if self._coupled or self._lazy_backend:
            self.EXT = np.asarray([_one_topo_order(n_nodes, self.hard_edges)],
                                  dtype=np.int32)
            self.n_ext, self.E = 1, 1
        else:
            self.EXT, self.n_ext, self.E = _extensions_matrix(
                range(n_nodes), set(self.hard_edges), max_orders)           # (E, n_nodes)
        # position of each node in each extension row -> feasibility + `t`
        self.POS = np.zeros((self.E, n_nodes), dtype=np.int32)
        for e in range(self.E):
            for pos, nd in enumerate(self.EXT[e]):
                self.POS[e, int(nd)] = pos

        # -- gate activation per (assignment, aux) --------------------
        self.GACT = np.zeros((self.NC, self.NA, len(self.gated)), dtype=bool)
        for c in range(self.NC):
            ov = self.A[c]
            for a in range(self.NA):
                cb = self.AUX[a].astype(float)
                for gi, (_u, _v, gate) in enumerate(self.gated):
                    self.GACT[c, a, gi] = bool(gate(ov, cb))
        # each gated edge's "extension respects it" flag
        self.GFEAS = np.ones((self.E, len(self.gated)), dtype=bool)
        for gi, (u, v, _g) in enumerate(self.gated):
            self.GFEAS[:, gi] = self.POS[:, u] < self.POS[:, v]

        # -- branch combo grid over the multi-branch entries ----------
        ks = [int(e.discrete_params) for e in self.branched]
        n_bc = int(np.prod(ks, dtype=object)) if ks else 1
        self.branch_ks = ks
        if (objective == "makespan" and not (self._coupled or self._lazy_backend)
                and n_bc > max(max_branch_combos, 100000)):
            raise NotImplementedError(
                f"dp_master_jax makespan: {n_bc} branch combos > "
                f"{max(max_branch_combos, 100000)} -- makespan couples the branch "
                "choice with cross-agent waiting, so it still enumerates combos")
        self.BC_grid = (np.array(list(itertools.product(*[range(k) for k in ks])), dtype=np.int32)
                        if ks else np.zeros((1, 0), dtype=np.int32))         # (BC, n_branched)
        self.n_bc = self.BC_grid.shape[0]
        self._branched_idx = [i for i, e in enumerate(proj) if e.discrete_params > 1]
        self._lazy = None
        if self._lazy_backend:
            from .lazy_dp import LazyMakespanDP
            self._lazy = LazyMakespanDP(problem, edge_cost_fn=ecf, **dict(lazy_kwargs or {}))
        self.lazy_exact, self.lazy_front, self.lazy_stats = True, 0, []

    # ------------------------------------------------------------------
    def feasible_ext_mask(self, c, a, node_active):
        """Boolean `(E,)` -- which hard-graph extensions are consistent with
        `(A[c], AUX[a])`'s resolved gated edges, restricted to active nodes.
        This is the vectorizable stand-in for `_linear_extensions(remaining,
        resolved_P)` -- validated against it in the test."""
        act = np.ones(self.E, dtype=bool)
        for gi, (u, v, _g) in enumerate(self.gated):
            if not self.GACT[c, a, gi]:
                continue
            if not (node_active[u] and node_active[v]):
                continue
            act &= self.GFEAS[:, gi]
        return act

    # ------------------------------------------------------------------
    def _combo_allowed(self, c, var_committed, var_anchor):
        ov = self.A[c]
        return all((not var_committed[s]) or int(ov[s]) == int(var_anchor[s])
                   for s in range(self.n_var))

    def run_python(self, params, wp_template, x0_full, x0_by_agent,
                   node_active=None, var_committed=None, var_anchor=None):
        """Piece-1 reference: the same enumeration `solve_dp_master` walks,
        but combos span every slot and the gated-edge / anchor handling goes
        through the precomputed tensors. Returns `solve_dp_master`'s dict."""
        p = self.problem
        n_nodes, n_agents, n_var, n_cond = self.n_nodes, self.n_agents, self.n_var, self.n_cond
        node_active = (np.ones(n_nodes, bool) if node_active is None
                       else np.asarray(node_active, bool))
        var_committed = (np.zeros(n_var, bool) if var_committed is None
                         else np.asarray(var_committed, bool))
        var_anchor = (np.zeros(n_var, int) if var_anchor is None
                      else np.asarray(var_anchor, int))
        remaining = [n for n in range(n_nodes) if node_active[n]]
        x0_of = (dict(x0_by_agent) if isinstance(x0_by_agent, dict)
                 else {a: np.asarray(x0_by_agent) for a in range(n_agents)})

        best, best_sol = np.inf, None
        for c in range(self.NC):
            if not self._combo_allowed(c, var_committed, var_anchor):
                continue
            ov = np.asarray(self.A[c], dtype=int)
            owned = _agent_owned_nodes(p, self.instances, ov)
            owned = {j: {n for n in ns if node_active[n]} for j, ns in owned.items()}
            for a in range(self.NA):
                aux = tuple(int(x) for x in self.AUX[a])
                P = _resolve_precedence(p, self.ordering_edges, ov, aux)
                P = {(u, v) for (u, v) in P if node_active[u] and node_active[v]}
                exts = _linear_extensions(remaining, P, self.max_orders)
                if exts is None:
                    continue
                P_pred = {v: set() for v in range(n_nodes)}
                for u, v in P:
                    P_pred[v].add(u)
                agent_of_node = {}
                for j, ns in owned.items():
                    for n in ns:
                        agent_of_node.setdefault(n, []).append(j)
                for ext in exts:
                    resolved = _full_ext_resolve(
                        p, ext, n_nodes, owned, ov, aux, x0_full,
                        self.layer0, self.branched, wp_template, node_active=node_active)
                    if self.objective in ("avg", "minmax"):
                        best, best_sol = _ext_full_avg_minmax_score(
                            self.objective, ext, P_pred, agent_of_node, owned, ov, aux,
                            x0_of, self.sliced_of, resolved, best, best_sol)
                    else:
                        best, best_sol = _ext_full_makespan_score(
                            ext, P_pred, agent_of_node, owned, ov, aux,
                            x0_of, self.sliced_of, resolved, best, best_sol)

        if best_sol is None:
            return dict(status="INFEASIBLE", objective=None, branch={}, assignment={},
                        aux={}, time={}, routes={}, agent_cost={})
        br, ovv, auxx, arr, agent_seq, per_agent = best_sol
        return dict(
            status="OPTIMAL",
            objective=float(best),
            branch={k: int(v) for k, v in br.items()},
            assignment={s: int(ovv[s]) for s in range(n_var)},
            aux={k: int(auxx[k]) for k in range(n_cond)},
            time={int(n): float(t) for n, t in arr.items()},
            t_end=float(max(arr.values(), default=0.0)),
            agent_cost={int(j): float(cc) for j, cc in per_agent.items()},
            routes={int(j): list(seq) for j, seq in agent_seq.items()},
        )


    # ------------------------------------------------------------------
    # piece 2: one vmapped resolve + score pass over the whole
    # (assignment x aux x extension [x branch]) grid.
    # ------------------------------------------------------------------
    def _prep(self):
        """Build + cache the jnp static tensors the vectorized pass needs."""
        if getattr(self, "_prepped", None) is not None:
            return self._prepped
        import jax.numpy as jnp
        p, N, J = self.problem, self.n_nodes, self.n_agents
        NC = self.NC

        owner_per_combo = np.zeros((NC, len(self.branched)), dtype=np.int32)
        for c in range(NC):
            for bi, e in enumerate(self.branched):
                owner_per_combo[c, bi] = entry_owner(p, e, np.asarray(self.A[c], int))

        OWN = np.zeros((NC, N, J), dtype=bool)
        for c in range(NC):
            for j, ns in _agent_owned_nodes(p, self.instances, np.asarray(self.A[c], int)).items():
                for n in ns:
                    OWN[c, int(n), j] = True

        hard_pred = np.zeros((N, N), dtype=bool)   # [v, u]
        for (u, v) in self.hard_edges:
            hard_pred[v, u] = True
        gated_vu = np.zeros((len(self.gated), N, N), dtype=bool)
        gated_ends = np.zeros((len(self.gated), 2), dtype=np.int32)
        for gi, (u, v, _g) in enumerate(self.gated):
            gated_vu[gi, v, u] = True
            gated_ends[gi] = (u, v)

        wn = np.array([int(e.write_node) for e in self.branched], dtype=np.int32)
        starts = np.array([int(e.branch_slice.start) for e in self.branched], dtype=np.int32)
        max_k = max([1] + self.branch_ks)

        # grid index arrays
        gc, ga, ge = (x.reshape(-1) for x in np.meshgrid(
            np.arange(NC), np.arange(self.NA), np.arange(self.E), indexing="ij"))

        self._prepped = dict(
            A=jnp.asarray(self.A), AUX=jnp.asarray(self.AUX),
            POS=jnp.asarray(self.POS), EXT=jnp.asarray(self.EXT.astype(np.int32)),
            GACT=jnp.asarray(self.GACT), GFEAS=jnp.asarray(self.GFEAS),
            owner_per_combo=jnp.asarray(owner_per_combo), OWN=jnp.asarray(OWN),
            hard_pred=jnp.asarray(hard_pred), gated_vu=jnp.asarray(gated_vu),
            gated_ends=jnp.asarray(gated_ends), wn=wn, wn_j=jnp.asarray(wn),
            starts=starts, max_k=max_k,
            BC_grid=jnp.asarray(self.BC_grid),
            gc=jnp.asarray(gc), ga=jnp.asarray(ga), ge=jnp.asarray(ge),
            n_ext=self.n_ext,
        )
        return self._prepped

    def skeleton_grid_fn(self, k):
        """A jittable pure function
        `(params, wp_template, x0_full, X0, node_active, var_committed,
        var_anchor, skel_penalty=None) -> (obj, assign_oh, cond, t,
        proj_branch, wp0, cell)` for the `k` best discrete skeletons (distinct
        `(assignment, aux)`, ascending by objective). Safe to call inside
        another `jax.jit` trace (e.g. `SmallContinuousVRPSolver._ask`). See
        `_score_grid_core`."""
        from functools import partial
        _ensure_x64()
        self._prep()
        return partial(self._score_grid_core, n_top=int(k))

    def _skeleton_grid(self, params, wp_template, x0_full, X0,
                       node_active, var_committed, var_anchor, k):
        """`skeleton_grid_fn(k)` compiled + cached per `k` and run eagerly.
        Returns the 7-tuple `(obj (k,), assign_oh (k,n_var,J), cond (k,n_cond),
        t (k,N), proj_branch (k,n_branch), wp0 (k,N,S), cell (k,3))` -- `cell`
        is `[assignment_combo, aux_combo, extension]` for `_reconstruct_cell`.
        `obj` is `+inf` for a skeleton with no feasible ordering."""
        import jax
        _ensure_x64()
        self._prep()
        if k not in self._jitted:
            self._jitted[k] = jax.jit(self.skeleton_grid_fn(k))
        return self._jitted[k](params, wp_template, x0_full, X0,
                               node_active, var_committed, var_anchor)

    def _score_grid_core(self, params, wp_template, x0_full, X0,
                         node_active, var_committed, var_anchor, n_top, skel_penalty=None):
        """The whole grid search, one `jax.jit` per `n_top`. Enumerates every
        `(assignment, aux, ordering)` cell (`G = NC*NA*E`), resolves each
        through `apply_projections`, prices it with a per-agent branch Viterbi
        (`_agent_route_dp`, avg/minmax) or the coupled makespan forward pass
        (`_makespan_arr_jax`, over the `BC` branch-combo axis), then reduces
        `min` over orderings so each `(assignment, aux)` skeleton has one
        score, and returns the `n_top` cheapest skeletons with the genome
        pieces needed to seed a GA individual. See `_skeleton_grid` for the
        return shape.

        `wp_template` is one `(N, S)` template shared by every cell, or one
        per `(assignment, aux)` skeleton, `(NC*NA, N, S)` (row `c*NA + a`).
        `skel_penalty` (optional, `(NC*NA,)`) is added to each skeleton's
        cells' score before the top-k."""
        if self._lazy_backend:
            return self._score_grid_lazy(params, wp_template, x0_full, X0, node_active,
                                         var_committed, var_anchor, n_top, skel_penalty)
        import jax
        import jax.numpy as jnp
        from ..evolutionary_waypoint_solver.problem import jit_apply_projections
        st = self._prep()
        p, N, J = self.problem, self.n_nodes, self.n_agents
        NC, NA, E = self.NC, self.NA, self.E
        n_var, n_cond = self.n_var, self.n_cond
        n_psi, n_branch = p.n_psi, p.n_branch
        S = wp_template.shape[-1]
        gc, ga, ge = st["gc"], st["ga"], st["ge"]
        G = gc.shape[0]
        max_k = st["max_k"]

        # -- discrete masks ------------------------------------------------
        slot_agent = jnp.where(var_committed[None, :], var_anchor[None, :], st["A"]) \
            if n_var else jnp.zeros((NC, 0), jnp.int32)
        combo_ok = (jnp.all(jnp.where(var_committed[None, :],
                                      st["A"] == var_anchor[None, :], True), axis=1)
                    if n_var else jnp.ones((NC,), bool))
        assign_oh = (jax.nn.one_hot(slot_agent, J) if n_var
                     else jnp.zeros((NC, 0, J)))
        cond = st["AUX"].astype(float) if n_cond else jnp.zeros((NA, 0))

        ends = st["gated_ends"]
        if ends.shape[0]:
            end_ok = node_active[ends[:, 0]] & node_active[ends[:, 1]]     # (n_gated,)
            active_g = st["GACT"] & end_ok[None, None, :]                  # (NC,NA,n_gated)
            feas = jnp.all(jnp.where(active_g[:, :, None, :],
                                     st["GFEAS"][None, None, :, :], True), axis=3)
        else:
            active_g = jnp.zeros((NC, NA, 0), bool)
            feas = jnp.ones((NC, NA, E), bool)
        feas = (feas & combo_ok[:, None, None]
                & (jnp.arange(E) < st["n_ext"])[None, None, :])
        if self._coupled:
            # The coupled DP's states ARE order ideals of the resolved
            # precedence DAG, so it enforces the gated edges itself and
            # reports infeasibility as an infinite makespan.  The "does
            # extension e respect them" mask is meaningless here: there is a
            # single representative extension and it is never used as the
            # visiting order.
            feas = jnp.broadcast_to(combo_ok[:, None, None], (NC, NA, E))
        feas_g = feas[gc, ga, ge]                                         # (G,)

        # PRED (NC,NA,N,N) via active gated edges
        pred = jnp.broadcast_to(st["hard_pred"], (NC, NA, N, N))
        if ends.shape[0]:
            pred = pred | jnp.any(active_g[:, :, :, None, None]
                                  & st["gated_vu"][None, None, :, :, :], axis=2)
        pred = pred & node_active[None, None, :, None] & node_active[None, None, None, :]

        OWN_act = st["OWN"] & node_active[None, :, None]                   # (NC,N,J)
        n_br = len(self.branched)

        # -- resolve layer0 over the grid -------------------------------
        assign_g = assign_oh[gc]                                          # (G,n_var,J)
        cond_g = cond[ga]                                                 # (G,n_cond)
        T_g = st["POS"][ge].astype(float)                                 # (G,N)
        f0 = jit_apply_projections(p, only_entries=self.layer0)
        wp_t_g = (wp_template[gc * NA + ga] if wp_template.ndim == 3
                  else jnp.broadcast_to(wp_template[None], (G, N, S)))
        wp0_g = f0(wp_t_g,
                   jnp.zeros((G, n_psi)), jnp.zeros((G, n_branch)), params,
                   assign_g, cond_g, T_g, node_active, x0_full)           # (G,N,S)

        # Per-agent branch candidate rows: agent j's row at node n is `wp0`
        # spliced with the branch candidates of the (single) branched entry
        # that writes n AND is owned by j. Two entries at one shared node
        # have distinct owners (guaranteed in __init__), so they land in
        # different `CAND[j]`. `Kof[g, j, n]` is that entry's branch count
        # (1 where j has no branched entry at n) -> masks the padding.
        own_per_g = st["owner_per_combo"][gc] if n_br else jnp.zeros((G, 0), jnp.int32)
        CAND = [jnp.repeat(wp0_g[:, :, None, :], max_k, axis=2) for _ in range(J)]
        Kof = jnp.ones((G, J, N), jnp.int32)
        res_rows = []                                                     # per bi: (G,k,S)
        for bi, k in enumerate(self.branch_ks):
            wn = int(st["wn"][bi])
            start = int(st["starts"][bi])
            pb = jax.nn.one_hot(start + jnp.arange(k), n_branch)          # (k,n_branch)
            fb = jit_apply_projections(p, only_entries=(self.branched[bi],))
            res = fb(jnp.repeat(wp0_g, k, axis=0),
                     jnp.zeros((G * k, n_psi)), jnp.tile(pb, (G, 1)), params,
                     jnp.repeat(assign_g, k, axis=0), jnp.repeat(cond_g, k, axis=0),
                     jnp.repeat(T_g, k, axis=0), node_active, x0_full
                     ).reshape(G, k, N, S)
            rows = res[:, :, wn, :]                                       # (G,k,S)
            res_rows.append(rows)
            padded = jnp.concatenate(
                [rows, jnp.broadcast_to(rows[:, :1], (G, max_k - k, S))], axis=1)  # (G,max_k,S)
            owner_bi = own_per_g[:, bi]                                   # (G,)
            Kof = Kof.at[jnp.arange(G), owner_bi, wn].set(k)
            for j in range(J):
                sel = (owner_bi == j)[:, None, None]                     # (G,1,1)
                CAND[j] = CAND[j].at[:, wn, :, :].set(
                    jnp.where(sel, padded, CAND[j][:, wn, :, :]))

        # -- dense EDGE/DEPOT per grid cell, per agent -----------------
        DEP, EDG = [], []
        for j in range(J):
            sj = self.sliced_of[j]
            x0j = X0[j]
            Cj = CAND[j]                                                 # (G,N,max_k,S)
            dep = jax.vmap(jax.vmap(jax.vmap(
                lambda r: jnp.asarray(sj(x0j, r)))))(Cj)                 # (G,N,K)
            gk = jax.vmap(jax.vmap(lambda ru, rv: jnp.asarray(sj(ru, rv)),
                                   (None, 0)), (0, None))
            gn = jax.vmap(jax.vmap(gk, (None, 0)), (0, None))
            edg = jax.vmap(gn)(Cj, Cj)                                   # (G,N,N,K,K)
            vkj = jnp.arange(max_k)[None, None, :] < Kof[:, j, :, None]  # (G,N,max_k)
            dep = jnp.where(vkj, dep, _BIG)
            edg = jnp.where(vkj[:, :, None, :, None], edg, _BIG)
            edg = jnp.where(vkj[:, None, :, None, :], edg, _BIG)
            DEP.append(dep)
            EDG.append(edg)
        DEPOT = jnp.stack(DEP, axis=1)                                    # (G,J,N,K)
        EDGE = jnp.stack(EDG, axis=1)                                     # (G,J,N,N,K,K)

        order_g = st["EXT"][ge]                                           # (G,N)
        wn_j = st["wn_j"]                                                 # (n_br,)

        # -- cost + branch pick per grid cell --------------------------
        if self.objective in ("avg", "minmax"):
            actmask = OWN_act[gc[:, None], order_g]                       # (G,N,J)
            route = jax.vmap(lambda o, am, ed, dp: jax.vmap(
                _agent_route_dp, (None, 1, 0, 0))(o, am, ed, dp))         # over J
            costs, br_gj = route(order_g, actmask, EDGE, DEPOT)           # (G,J), (G,J,N)
            cost_g = costs.sum(1) if self.objective == "avg" else costs.max(1)
            score_g = jnp.where(feas_g, cost_g, jnp.inf)                  # (G,)
            order_dp = order_g
            if n_br:
                own_g = st["owner_per_combo"][gc]                        # (G,n_br)
                branch_of_entry = br_gj[jnp.arange(G)[:, None], own_g,
                                        wn_j[None, :]]                   # (G,n_br)
            else:
                branch_of_entry = jnp.zeros((G, 0), jnp.int32)
        elif self._coupled:
            # coupled_dp.py: ONE exact DP per (assignment, aux) picks the
            # visiting order and every branch together, so neither the
            # extension axis nor the BC axis exists any more.  Its state graph
            # depends on the RESOLVED ownership / precedence / branch counts,
            # and `node_active` is a traced value here, so the structure is
            # built and cached host-side behind a `pure_callback`.  That keeps
            # every shape in this trace fixed, which is what preserves
            # evosax_ga's "a new anchor never retraces `step`" contract -- see
            # `_coupled_host`.
            out_shape = (jax.ShapeDtypeStruct((G,), jnp.float64),
                         jax.ShapeDtypeStruct((G, N), jnp.int32),
                         jax.ShapeDtypeStruct((G, n_br), jnp.int32))
            score_g, order_dp, branch_of_entry = jax.pure_callback(
                self._coupled_host, out_shape,
                EDGE, DEPOT, Kof, OWN_act[gc], pred[gc, ga], own_per_g)
            score_g = jnp.where(feas_g, score_g, jnp.inf)                 # (G,)
        else:
            # makespan: branch combos couple with cross-agent waiting, so the
            # branch combo stays an explicit grid axis and its winner is the
            # BC argmin (not a per-agent Viterbi pick).
            BC = st["BC_grid"].shape[0]
            BR = jnp.zeros((NC, BC, N, J), jnp.int32)
            for bi in range(n_br):
                wn = int(st["wn"][bi])
                own_c = st["owner_per_combo"][:, bi]                      # (NC,)
                BR = BR.at[jnp.arange(NC)[:, None], jnp.arange(BC)[None, :],
                           wn, own_c[:, None]].set(st["BC_grid"][None, :, bi])
            BR_g = BR[gc]                                                 # (G,BC,N,J)

            def cell(order, br, edge_m, depot_m, own_c, pred_ca):
                arr = _makespan_arr_jax(order, br, edge_m, depot_m, own_c, pred_ca, J)
                return jnp.max(jnp.where(node_active, arr, -_BIG))

            ms = jax.vmap(jax.vmap(cell, (None, 0, None, None, None, None)),
                          (0, 0, 0, 0, 0, 0))(
                order_g, BR_g, EDGE, DEPOT, OWN_act[gc], pred[gc, ga])    # (G,BC)
            ms = jnp.where(feas_g[:, None], ms, jnp.inf)
            bc_star_g = jnp.argmin(ms, axis=1)                            # (G,)
            score_g = jnp.min(ms, axis=1)                                 # (G,)
            branch_of_entry = (st["BC_grid"][bc_star_g] if n_br
                               else jnp.zeros((G, 0), jnp.int32))         # (G,n_br)
            order_dp = order_g

        # -- CV_proj bias: every cell's fully-resolved wp (layer0 +
        # each branched entry spliced at its now-known winning branch,
        # same splice `_reconstruct_cell`/the kk-gather below do post-
        # selection, done here for every grid cell instead) -- then bias
        # `score_g` away from cells whose analytic projections didn't
        # actually hold (e.g. ur5e_ik.py's clipped out-of-reach IK),
        # mirroring `SmallContinuousVRPSolver.reseed`'s own hard_score
        # exactly: cost + 1e6*max(0, CV_proj - cv_proj_tol). Without this,
        # `lax.top_k` below is purely cost-ordered and a cheaper-but-
        # analytically-broken skeleton can starve a feasible one out of
        # the top-k entirely -- reseed can only re-rank what made the cut.
        #
        # `apply_anchor`'s frozen/live substitution is skipped here (no
        # `anchor.anchor_wp` at this layer): frozen == live == wp_full_g at
        # every ACTIVE node (the only nodes this grid varies), and
        # node_active/var_committed/x0 are fixed for the whole grid, so any
        # divergence at a passed node is the same constant for every cell
        # -- it cannot change which cell ranks best.
        #
        # Opt-in (`self.cv_proj_bias`, see __init__) and zero-cost whenever
        # this problem declares no `proj=` constraints (no CV_proj term at
        # all, same as `_evaluate_projection_cv_jax`'s own all-zero
        # convention) -- the splice below is skipped too, so a caller that
        # doesn't want this (or a scene with nothing to check) doesn't pay
        # for it.
        if self.cv_proj_bias and (p._proj_ineq_constraints or p._proj_eq_constraints):
            from ..evolutionary_waypoint_solver.solver import _calc_cv_jax
            wp_full_g = wp0_g
            for bi, kb in enumerate(self.branch_ks):
                wn = int(st["wn"][bi])
                ch = branch_of_entry[:, bi]                              # (G,)
                row_bi = res_rows[bi][jnp.arange(G), ch]                 # (G,S)
                if self.branch_static[bi] and self.branch_cols[bi].shape[0]:
                    ci = jnp.asarray(self.branch_cols[bi])
                    wp_full_g = wp_full_g.at[:, wn, ci].set(row_bi[:, ci])
                else:
                    owner_bi_g = own_per_g[:, bi]                        # (G,)
                    band = owner_bi_g[:, None] * p.dim + jnp.arange(p.dim)[None, :]  # (G,dim)
                    wp_full_g = wp_full_g.at[jnp.arange(G)[:, None], wn, band].set(
                        jnp.take_along_axis(row_bi, band, axis=1))
            cv_G = (jnp.concatenate(
                    [fn(assign_g, cond_g, T_g, wp_full_g, wp_full_g, node_active, x0_full, params)
                     for fn in p._proj_ineq_constraints], axis=1)
                  if p._proj_ineq_constraints else None)
            cv_H = (jnp.concatenate(
                    [fn(assign_g, cond_g, T_g, wp_full_g, wp_full_g, node_active, x0_full, params)
                     for fn in p._proj_eq_constraints], axis=1)
                  if p._proj_eq_constraints else None)
            cv_proj_g = _calc_cv_jax(G, cv_G, cv_H)                       # (G,)
            score_g = score_g + 1e6 * jnp.maximum(0.0, cv_proj_g - self.cv_proj_tol)
        if skel_penalty is not None:
            score_g = score_g + skel_penalty[gc * NA + ga]

        # -- reduce over orderings: one score per (assignment, aux) ----
        s2 = score_g.reshape(NC * NA, E)
        e_star = jnp.argmin(s2, axis=1).astype(jnp.int32)                 # (NC*NA,)
        skel_score = jnp.min(s2, axis=1)                                  # (NC*NA,)
        kk = min(int(n_top), NC * NA)
        neg, s_star = jax.lax.top_k(-skel_score, kk)                      # (kk,)
        obj = -neg
        c_star = (s_star // NA).astype(jnp.int32)
        a_star = (s_star % NA).astype(jnp.int32)
        e_sel = e_star[s_star]                                           # (kk,)
        g_sel = (s_star * E + e_sel).astype(jnp.int32)                   # flat G index

        # -- gather genome pieces for the kk best skeletons ------------
        assign_k = assign_oh[c_star] if n_var else jnp.zeros((kk, 0, J))  # (kk,n_var,J)
        cond_k = (st["AUX"][a_star].astype(float) if n_cond
                  else jnp.zeros((kk, 0)))                                # (kk,n_cond)
        # rank-per-node: for the enumerate backend `order_k` IS `EXT[e_sel]`
        # and this scatter reproduces `POS[e_sel]` exactly; for the coupled
        # backend it is the sequence the DP itself chose.
        order_k = order_dp[g_sel]                                        # (kk,N)
        t_k = jnp.zeros((kk, N)).at[jnp.arange(kk)[:, None], order_k].set(
            jnp.broadcast_to(jnp.arange(N, dtype=float)[None, :], (kk, N)))
        be_k = branch_of_entry[g_sel]                                    # (kk,n_br)
        own_k = st["owner_per_combo"][c_star] if n_br else None          # (kk,n_br)
        proj_branch_k = jnp.zeros((kk, n_branch))
        wp0_k = wp0_g[g_sel]                                            # (kk,N,S)
        dim = p.dim
        for bi, kb in enumerate(self.branch_ks):
            wn = int(st["wn"][bi])
            start = int(st["starts"][bi])
            ch = be_k[:, bi]                                             # (kk,)
            proj_branch_k = proj_branch_k.at[:, start:start + kb].set(
                jax.nn.one_hot(ch, kb))
            row_bi = res_rows[bi][g_sel, ch]                            # (kk,S) full spliced row
            # write only the columns this entry controls, so two entries at a
            # shared node don't clobber each other's band.
            if self.branch_static[bi] and self.branch_cols[bi].shape[0]:
                ci = jnp.asarray(self.branch_cols[bi])                  # (w,)
                wp0_k = wp0_k.at[:, wn, ci].set(row_bi[:, ci])
            else:                                                       # dynamic: owner's dim-band
                band = own_k[:, bi:bi + 1] * dim + jnp.arange(dim)[None, :]  # (kk,dim)
                wp0_k = wp0_k.at[jnp.arange(kk)[:, None], wn, band].set(
                    jnp.take_along_axis(row_bi, band, axis=1))
        # (kk, 3+N): `[assignment, aux, extension]` as before, then the
        # winning visiting order -- `_reconstruct_cell` reads the order from
        # here rather than from `EXT[e]`, which is the same thing for the
        # enumerate backend and the only correct thing for the coupled one.
        cell_k = jnp.concatenate(
            [jnp.stack([c_star, a_star, e_sel], axis=1), order_k.astype(jnp.int32)],
            axis=1)
        return obj, assign_k, cond_k, t_k, proj_branch_k, wp0_k, cell_k

    def __call__(self, *a, **k):
        return self.run_vec(*a, **k)

    def run_vec(self, params, wp_template, x0_full, x0_by_agent,
                node_active=None, var_committed=None, var_anchor=None):
        """Run the jitted grid search and reconstruct the winning
        `(assignment, aux, extension)` cell into a `solve_dp_master`-shaped
        dict. Thin wrapper over `run_topk(1, ...)`."""
        top = self.run_topk(1, params, wp_template, x0_full, x0_by_agent,
                            node_active=node_active, var_committed=var_committed,
                            var_anchor=var_anchor)
        if not top:
            return dict(status="INFEASIBLE", objective=None, branch={}, assignment={},
                        aux={}, time={}, routes={}, agent_cost={})
        return top[0]

    def run_topk(self, k, params, wp_template, x0_full, x0_by_agent,
                 node_active=None, var_committed=None, var_anchor=None):
        """The `k` best discrete skeletons (distinct `(assignment, aux)`),
        each a `solve_dp_master`-shaped dict, ascending by objective. Returns
        fewer than `k` dicts if the feasible grid is smaller, `[]` if
        infeasible.

        `_skeleton_grid` already reduces `min` over orderings, so the returned
        skeletons are distinct by construction -- this is one `lax.top_k` plus
        one `_reconstruct_cell` per skeleton. Intended as the population seed
        for `SmallContinuousVRP`."""
        import jax.numpy as jnp
        import numpy as _np
        n_nodes, n_agents, n_var = self.n_nodes, self.n_agents, self.n_var
        node_active = (_np.ones(n_nodes, bool) if node_active is None
                       else _np.asarray(node_active, bool))
        var_committed = (_np.zeros(n_var, bool) if var_committed is None
                         else _np.asarray(var_committed, bool))
        var_anchor = (_np.zeros(n_var, int) if var_anchor is None
                      else _np.asarray(var_anchor, int))
        x0_of = (dict(x0_by_agent) if isinstance(x0_by_agent, dict)
                 else {a: _np.asarray(x0_by_agent) for a in range(n_agents)})
        X0 = jnp.stack([jnp.asarray(x0_of[j], float) for j in range(n_agents)])

        k = max(1, int(k))
        obj, _ak, _ck, _tk, _pk, _wk, cell = self._skeleton_grid(
            jnp.asarray(params), jnp.asarray(wp_template), jnp.asarray(x0_full), X0,
            jnp.asarray(node_active), jnp.asarray(var_committed),
            jnp.asarray(var_anchor), k)
        obj = _np.asarray(obj)
        cell = _np.asarray(cell)

        out = []
        for i in range(len(obj)):
            if not _np.isfinite(float(obj[i])):
                break
            c, a, e = int(cell[i, 0]), int(cell[i, 1]), int(cell[i, 2])
            d = self._reconstruct_cell(c, a, e, wp_template, x0_full, x0_of,
                                       node_active, var_committed, var_anchor,
                                       order=cell[i, 3:])
            if d is not None:
                out.append(d)
        return out

    def _score_grid_lazy(self, params, wp_template, x0_full, X0, node_active,
                         var_committed, var_anchor, n_top, skel_penalty=None):
        """`_score_grid_core` for `route_backend="lazy"`.  One exact
        `lazy_dp` solve per (assignment, aux) cell, run host-side behind a
        `pure_callback` (the engine's state graph is data-dependent, and
        this keeps every traced shape fixed so evosax_ga's jitted `step`
        still never retraces on an anchor change).  The engine returns each
        cell's order, branches and FULLY RESOLVED `wp` under that order --
        so the CV_proj bias and the seeded waypoints use the winner's own
        rank, not a representative one.  Same 7-tuple as `_skeleton_grid`."""
        import jax
        import jax.numpy as jnp
        st = self._prep()
        p, N, J = self.problem, self.n_nodes, self.n_agents
        NC, NA = self.NC, self.NA
        n_var, n_cond, n_branch = self.n_var, self.n_cond, p.n_branch
        S = wp_template.shape[-1]
        G = NC * NA
        n_br = len(self.branched)
        slot_agent = (jnp.where(var_committed[None, :], var_anchor[None, :], st["A"])
                      if n_var else jnp.zeros((NC, 0), jnp.int32))
        combo_ok = (jnp.all(jnp.where(var_committed[None, :],
                                      st["A"] == var_anchor[None, :], True), axis=1)
                    if n_var else jnp.ones((NC,), bool))
        assign_oh = jax.nn.one_hot(slot_agent, J) if n_var else jnp.zeros((NC, 0, J))
        cond = st["AUX"].astype(float) if n_cond else jnp.zeros((NA, 0))

        out_shape = (jax.ShapeDtypeStruct((G,), jnp.float64),
                     jax.ShapeDtypeStruct((G, N), jnp.int32),
                     jax.ShapeDtypeStruct((G, n_br), jnp.int32),
                     jax.ShapeDtypeStruct((G, N, S), jnp.float64))
        score_g, order_g, boe_g, wp_g = jax.pure_callback(
            self._lazy_host, out_shape, params, wp_template, x0_full, X0,
            node_active, slot_agent.astype(jnp.int32), combo_ok)
        gc = jnp.arange(G) // NA
        ga = jnp.arange(G) % NA
        score_g = jnp.where(combo_ok[gc], score_g, jnp.inf)
        assign_g, cond_g = assign_oh[gc], cond[ga]
        T_g = jnp.zeros((G, N)).at[jnp.arange(G)[:, None], order_g].set(
            jnp.broadcast_to(jnp.arange(N, dtype=float)[None, :], (G, N)))

        if self.cv_proj_bias and (p._proj_ineq_constraints or p._proj_eq_constraints):
            from ..evolutionary_waypoint_solver.solver import _calc_cv_jax
            cv_G = (jnp.concatenate(
                    [fn(assign_g, cond_g, T_g, wp_g, wp_g, node_active, x0_full, params)
                     for fn in p._proj_ineq_constraints], axis=1)
                  if p._proj_ineq_constraints else None)
            cv_H = (jnp.concatenate(
                    [fn(assign_g, cond_g, T_g, wp_g, wp_g, node_active, x0_full, params)
                     for fn in p._proj_eq_constraints], axis=1)
                  if p._proj_eq_constraints else None)
            cv_proj_g = _calc_cv_jax(G, cv_G, cv_H)
            score_g = score_g + 1e6 * jnp.maximum(0.0, cv_proj_g - self.cv_proj_tol)
        if skel_penalty is not None:
            score_g = score_g + skel_penalty

        kk = min(int(n_top), G)
        neg, s_star = jax.lax.top_k(-score_g, kk)
        obj = -neg
        c_star = (s_star // NA).astype(jnp.int32)
        a_star = (s_star % NA).astype(jnp.int32)
        assign_k = assign_oh[c_star] if n_var else jnp.zeros((kk, 0, J))
        cond_k = st["AUX"][a_star].astype(float) if n_cond else jnp.zeros((kk, 0))
        proj_branch_k = jnp.zeros((kk, n_branch))
        for bi, kb in enumerate(self.branch_ks):
            start = int(st["starts"][bi])
            proj_branch_k = proj_branch_k.at[:, start:start + kb].set(
                jax.nn.one_hot(boe_g[s_star, bi], kb))
        cell_k = jnp.concatenate(
            [jnp.stack([c_star, a_star, jnp.zeros_like(c_star)], axis=1),
             order_g[s_star].astype(jnp.int32)], axis=1)
        return obj, assign_k, cond_k, T_g[s_star], proj_branch_k, wp_g[s_star], cell_k

    def _lazy_host(self, params, wp_template, x0_full, X0, node_active, slot_agent,
                   combo_ok):
        """Host side of `route_backend="lazy"`: one `lazy_dp.solve_cell` per
        allowed (assignment, aux) cell.  An infeasible cell is `inf`.
        Records `lazy_exact` / `lazy_front` / `lazy_stats` on `self` as
        side-channel diagnostics (not part of the traced value)."""
        params = np.asarray(params)
        wpt = np.asarray(wp_template, float)
        x0 = np.asarray(x0_full, float)
        X0 = np.asarray(X0, float)
        na = np.asarray(node_active, bool)
        slot_agent = np.asarray(slot_agent, int)
        combo_ok = np.asarray(combo_ok, bool)
        NC, NA, N = self.NC, self.NA, self.n_nodes
        G = NC * NA
        n_br = len(self._branched_idx)
        score = np.full(G, np.inf)
        order = np.tile(np.arange(N, dtype=np.int32), (G, 1))
        boe = np.zeros((G, n_br), np.int32)
        wpt_g = wpt if wpt.ndim == 3 else np.broadcast_to(wpt[None], (G,) + wpt.shape)
        wp = np.array(wpt_g)
        self.lazy_exact, self.lazy_front, self.lazy_stats = True, 0, []
        for c in range(NC):
            if not combo_ok[c]:
                continue
            for a in range(NA):
                g = c * NA + a
                r = self._lazy.solve_cell(slot_agent[c], self.AUX[a], params, wpt_g[g], x0, X0, na)
                if r is None:
                    continue
                score[g] = r.makespan
                order[g] = r.order
                wp[g] = r.wp
                boe[g] = [r.branch.get(i, 0) for i in self._branched_idx]
                self.lazy_exact &= r.exact
                self.lazy_front = max(self.lazy_front, r.max_front)
                self.lazy_stats.append(r.stats)
        return score, order, boe, wp

    def _coupled_host(self, EDGE, DEPOT, Kof, OWN_act, PRED, own_per_g):
        """Host side of `route_backend="coupled"`: price every grid cell with
        `coupled_dp`'s exact DP.

        One cell is one `(assignment, aux)` -- the extension axis is gone --
        and its DP *structure* is fixed by the three resolved tensors this
        gets handed: `OWN_act[g]` (who visits what, already masked by
        `node_active`), `PRED[g]` (the resolved precedence DAG, likewise) and
        `Kof[g]` (branch counts).  Those bytes are the cache key, so the
        structure is built once per distinct scene shape and every later
        cycle only re-prices it -- which is the whole point of
        `coupled_dp.build_coupled_dp` keeping cost lookups as indices.

        Returns `(score (G,), order (G,N), branch_of_entry (G,n_br))`.  An
        infeasible cell comes back as `inf` and keeps its placeholder order.

        Called through `jax.pure_callback`: pure in its outputs, but it does
        memoise structures and record `coupled_front` / `coupled_exact` as
        diagnostics on `self`, which are side-channel and not part of the
        traced value."""
        EDGE = np.asarray(EDGE)
        DEPOT = np.asarray(DEPOT)
        Kof = np.asarray(Kof)
        OWN_act = np.asarray(OWN_act)
        PRED = np.asarray(PRED)
        own_per_g = np.asarray(own_per_g)

        G, N, J = OWN_act.shape[0], self.n_nodes, self.n_agents
        n_br = len(self.branched)
        wn = [int(en.write_node) for en in self.branched]
        max_k = int(EDGE.shape[-1])

        score = np.full(G, np.inf)
        order = np.tile(np.arange(N, dtype=np.int32), (G, 1))
        be = np.zeros((G, n_br), np.int32)
        self.coupled_front, self.coupled_exact = 0, True

        for g in range(G):
            own_g, pred_g, k_g = OWN_act[g], PRED[g], Kof[g]
            key = (own_g.tobytes(), pred_g.tobytes(), k_g.tobytes())
            dp = self._coupled_cache.get(key)
            if dp is None:
                if len(self._coupled_cache) >= self.max_coupled_structures:
                    raise NotImplementedError(
                        "dp_master_jax: route_backend='coupled' has built "
                        f"{len(self._coupled_cache)} distinct DP structures "
                        "(> max_coupled_structures) -- the scene shape is "
                        "changing far more than an MPC anchor should, so the "
                        "build cost is no longer amortized")
                # PRED[v, u] is "u precedes v" (dp_master's `hard_pred`).
                dp = build_coupled_dp(
                    N, J,
                    [[j for j in range(J) if own_g[v, j]] for v in range(N)],
                    [[u for u in range(N) if pred_g[v, u]] for v in range(N)],
                    k_g, max_k=max_k, **self._coupled_kwargs)
                self._coupled_cache[key] = dp

            best, o, bc, ow, front = dp._route_jit()(EDGE[g], DEPOT[g])
            best = float(best)
            self.coupled_front = max(self.coupled_front, int(front))
            self.coupled_exact &= int(front) <= dp.max_labels
            if not np.isfinite(best) or best >= 1e11:
                continue
            score[g] = best
            o, bc, ow = np.asarray(o), np.asarray(bc), np.asarray(ow)
            order[g] = o
            if n_br:
                pick = {}
                for r in range(o.shape[0]):
                    for m in range(ow.shape[1]):
                        if ow[r, m] >= 0:
                            pick[(int(o[r]), int(ow[r, m]))] = int(bc[r, m])
                for bi in range(n_br):
                    # A branched entry whose owning agent does not actually
                    # VISIT its write node has no routed branch to read, so
                    # it falls back to 0.  Its branch cannot affect the
                    # makespan (no leg touches it); it can still move the
                    # spliced `wp0` row, and hence `CV_proj`, where the
                    # enumerate backend would have taken whichever branch its
                    # BC argmin happened to land on.  Deterministic here,
                    # arbitrary there -- neither is a minimization.
                    be[g, bi] = pick.get((wn[bi], int(own_per_g[g, bi])), 0)

        return score, order, be

    def _reconstruct_cell(self, c, a, e, wp_template, x0_full, x0_of,
                          node_active, var_committed, var_anchor, order=None):
        """Reconstruct one grid cell `(A[c], AUX[a], EXT[e])` into a
        `solve_dp_master`-shaped dict, or `None` if the cell has no feasible
        branch DP solution."""
        import numpy as _np
        p = self.problem
        n_nodes, n_var, n_cond = self.n_nodes, self.n_var, self.n_cond
        ov = _np.where(var_committed, var_anchor, _np.asarray(self.A[c], int))
        aux = tuple(int(x) for x in self.AUX[a])
        seq = self.EXT[e] if order is None else order
        ext = tuple(int(x) for x in seq if node_active[int(x)])
        owned = {j: {n for n in ns if node_active[n]}
                 for j, ns in _agent_owned_nodes(p, self.instances, ov).items()}
        P = {(u, v) for (u, v) in _resolve_precedence(p, self.ordering_edges, ov, aux)
             if node_active[u] and node_active[v]}
        P_pred = {v: set() for v in range(n_nodes)}
        for u, v in P:
            P_pred[v].add(u)
        agent_of_node = {}
        for j, ns in owned.items():
            for n in ns:
                agent_of_node.setdefault(n, []).append(j)
        _as0, tvec, wp0, _rbn, _now = _full_ext_resolve(
            p, ext, n_nodes, owned, ov, aux, x0_full,
            self.layer0, self.branched, wp_template, node_active=node_active)

        # `_full_ext_resolve`'s `rows_by_node` keeps only the LAST branched
        # entry at a shared node; re-resolve each entry keyed by its own
        # owner so a two-arm handoff prices against the right columns.
        rows_by_owner = {}                             # (write_node, owner) -> (k, S)
        for be in self.branched:
            kb = be.discrete_params
            pb = _np.zeros((kb, p.n_branch))
            pb[_np.arange(kb), be.branch_slice.start + _np.arange(kb)] = 1.0
            res = _resolve_schedule_wp(
                p, _np.broadcast_to(wp0[None], (kb,) + wp0.shape), x0_full, ov, aux,
                pb, tvec, only_entries=(be,), node_active=node_active)
            wnn = int(be.write_node)
            rows_by_owner[(wnn, int(entry_owner(p, be, ov)))] = res[:, wnn, :]

        # Price the winning cell from numpy cost tables built in ONE batched
        # vmap per agent (structure.build_*_cost_table), then run the numpy
        # `_ext_nonfull_*` scorer -- NOT `_ext_full_*_score`, whose eager
        # per-(u,v,branch) `sliced_of` loop is ~1s when `sliced_of` is a jnp
        # fn with a few hundred branch combos (the dp_backend="jax" case).
        edge_tables, depot_tables, branch_counts, agent_keys = {}, {}, {}, {}
        for j, ns in owned.items():
            ns = sorted(ns)
            rj = {n: rows_by_owner[(n, j)] for n in ns if (n, j) in rows_by_owner}
            pairs = [(u, v) for u in ns for v in ns if u != v]
            edge_tables[j] = build_edge_cost_table(rj, wp0, pairs, edge_cost_fn=self.sliced_of[j])
            depot_tables[j] = build_depot_cost_table(rj, wp0, x0_of[j], ns,
                                                    edge_cost_fn=self.sliced_of[j])
            branch_counts[j] = {n: (rj[n].shape[0] if n in rj else 1) for n in ns}
            agent_keys[j] = {n: _inst_key(p, self.instances, n, j, ov) for n in ns}
        best, best_sol = _np.inf, None
        if self.objective in ("avg", "minmax"):
            best, best_sol = _ext_nonfull_avg_minmax(
                self.objective, ext, P_pred, agent_of_node, owned, ov, aux, agent_keys,
                edge_tables, depot_tables, branch_counts, best, best_sol)
        else:
            best, best_sol = _ext_nonfull_makespan(
                ext, P_pred, agent_of_node, owned, ov, aux, agent_keys,
                edge_tables, depot_tables, branch_counts, best, best_sol)
        if best_sol is None:
            return None
        br_raw, ovv, auxx, arr, agent_seq, per_agent = best_sol
        # `_ext_nonfull_*` keys `branch` by (node, _inst_key) for every routed
        # node; solve_dp_master's full path (and mpc.py's materialisation) key
        # only the multi-branch entries by (write_node, entry_owner-int). Remap.
        br = {}
        for (n, owner) in rows_by_owner:
            k = (n, agent_keys.get(owner, {}).get(n))
            if k in br_raw:
                br[(n, owner)] = br_raw[k]
        return dict(
            status="OPTIMAL", objective=float(best),
            branch={k: int(v) for k, v in br.items()},
            assignment={s: int(ovv[s]) for s in range(n_var)},
            aux={k: int(auxx[k]) for k in range(n_cond)},
            time={int(n): float(t) for n, t in arr.items()},
            t_end=float(max(arr.values(), default=0.0)),
            agent_cost={int(j): float(cc) for j, cc in per_agent.items()},
            routes={int(j): list(seq) for j, seq in agent_seq.items()})


def make_dp_master_jax(problem, *, objective="makespan", edge_cost_fn=None,
                       max_assign_combos=4096, max_orders=20000, max_branch_combos=4096,
                       cv_proj_bias=False, cv_proj_tol=1e-4,
                       route_backend="enumerate", max_coupled_structures=512,
                       coupled_kwargs=None, lazy_kwargs=None):
    """Build the exhaustive static enumeration for `problem` and return an
    object whose `run_python` (piece 1) / `run` (piece 3) solves the discrete
    subproblem for a cycle's continuous data. See the module docstring."""
    if objective not in ("makespan", "minmax", "avg"):
        raise ValueError(f"unknown objective {objective!r}")
    return _DpMasterJax(problem, objective, edge_cost_fn, max_assign_combos,
                        max_orders, max_branch_combos,
                        cv_proj_bias=cv_proj_bias, cv_proj_tol=cv_proj_tol,
                        route_backend=route_backend,
                        max_coupled_structures=max_coupled_structures,
                        coupled_kwargs=coupled_kwargs, lazy_kwargs=lazy_kwargs)

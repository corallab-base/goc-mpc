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
`run_vec`) with the cycle's continuous data. The grid search --
`_score_grid_core`: resolve every `(assignment, aux, extension)` cell via
one batched `apply_projections` per projection layer, build dense
EDGE/DEPOT, score with `_viterbi_cost_masked` (avg/minmax) or
`_makespan_arr_jax` (makespan), mask infeasible cells, global argmin -- is
one `jax.jit` compiled once per problem. Only the final winner -> dict
reconstruction (routes per agent, branch per (node, owner)) stays in Python,
on the single winning extension (same trick as
`dp_master._ext_full_batched`).

Requires a jnp-traceable `edge_cost_fn`. Needs no `node_candidates` -- the
grid search resolves everything through `apply_projections`, so the
non-full / full-resolve distinction `solve_dp_master` draws does not apply
here.
"""

import itertools

import numpy as np

from .structure import (
    build_depot_cost_table, build_edge_cost_table, entry_owner, node_instances)
from .cpsat_model import _agent_sliced_cost_fn
from .dp_master import (
    _BIG, _agent_owned_nodes, _ensure_x64, _ext_full_avg_minmax_score,
    _ext_full_makespan_score, _ext_nonfull_avg_minmax, _ext_nonfull_makespan,
    _extensions_matrix, _full_ext_resolve, _inst_key, _linear_extensions,
    _makespan_arr_jax, _resolve_precedence,
)


def _viterbi_cost_masked(order, act, EDGE_j, DEPOT_j):
    """min routed-path cost for the agent whose visited positions along the
    full node order `order` are flagged by `act` (bool, same length). Like
    `dp_master._viterbi_cost_jax` but consumes the un-compacted extension +
    an ownership mask, so the caller need not compact per (combo, agent)."""
    import jax.numpy as jnp
    from jax import lax
    K = EDGE_j.shape[-1]

    def step(carry, i):
        cost, prev, seen = carry
        v = order[i]
        a = act[i]
        first = a & (~seen)
        edge_c = jnp.min(cost[:, None] + EDGE_j[prev, v], axis=0)
        new = jnp.where(first, DEPOT_j[v], edge_c)
        return (jnp.where(a, new, cost), jnp.where(a, v, prev), seen | a), None

    (cost, _, seen), _ = lax.scan(
        step, (jnp.zeros(K), order[0], jnp.bool_(False)), jnp.arange(order.shape[0]))
    return jnp.where(seen, jnp.min(cost), 0.0)


class _DpMasterJax:
    """Holds the exhaustive static enumeration for one `problem` + objective
    + caps. `run` (piece 1: Python; later: jitted) consumes a cycle's
    continuous data against it."""

    def __init__(self, problem, objective, edge_cost_fn, max_assign_combos,
                 max_orders, max_branch_combos):
        self._prepped = None
        self._jitted = None
        self.problem = problem
        self.objective = objective
        self.max_orders = max_orders
        self.max_branch_combos = max_branch_combos
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
        for e in self.branched:
            others = set().union(*(o.write_cols for o in self.branched if o is not e)) \
                if len(self.branched) > 1 else set()
            if e.read_cols & others:
                raise NotImplementedError(
                    "dp_master_jax: a multi-branch projection reads a column "
                    "another multi-branch projection pins -- coupled branch DP "
                    "not implemented")

        # -- assignment combos over ALL slots ----------------------------
        n_assign = n_agents ** n_var if n_var else 1
        if n_assign > max_assign_combos:
            raise NotImplementedError(
                f"dp_master_jax: {n_agents}**{n_var} = {n_assign} assignment "
                f"combos > max_assign_combos={max_assign_combos}")
        self.A = (np.array(list(itertools.product(range(n_agents), repeat=n_var)), dtype=np.int32)
                  if n_var else np.zeros((1, 0), dtype=np.int32))            # (NC, n_var)
        self.AUX = (np.array(list(itertools.product((0, 1), repeat=n_cond)), dtype=np.int32)
                    if n_cond else np.zeros((1, 0), dtype=np.int32))         # (NA, n_cond)
        self.NC, self.NA = self.A.shape[0], self.AUX.shape[0]

        # -- hard-graph linear extensions ------------------------------
        self.hard_edges = [(int(u), int(v)) for (u, v, g) in self.ordering_edges if g is None]
        self.gated = [(int(u), int(v), g) for (u, v, g) in self.ordering_edges if g is not None]
        self.EXT, self.n_ext, self.E = _extensions_matrix(
            range(n_nodes), set(self.hard_edges), max_orders)               # (E, n_nodes)
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
        if objective == "makespan" and n_bc > max(max_branch_combos, 100000):
            raise NotImplementedError(
                f"dp_master_jax makespan: {n_bc} branch combos > "
                f"{max(max_branch_combos, 100000)} -- makespan couples the branch "
                "choice with cross-agent waiting, so it still enumerates combos")
        self.BC_grid = (np.array(list(itertools.product(*[range(k) for k in ks])), dtype=np.int32)
                        if ks else np.zeros((1, 0), dtype=np.int32))         # (BC, n_branched)
        self.n_bc = self.BC_grid.shape[0]

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
        KC_node = np.ones(N, dtype=np.int32)
        for bi, k in enumerate(self.branch_ks):
            KC_node[wn[bi]] = k

        # grid index arrays
        gc, ga, ge = (x.reshape(-1) for x in np.meshgrid(
            np.arange(NC), np.arange(self.NA), np.arange(self.E), indexing="ij"))

        self._prepped = dict(
            A=jnp.asarray(self.A), AUX=jnp.asarray(self.AUX),
            POS=jnp.asarray(self.POS), EXT=jnp.asarray(self.EXT.astype(np.int32)),
            GACT=jnp.asarray(self.GACT), GFEAS=jnp.asarray(self.GFEAS),
            owner_per_combo=jnp.asarray(owner_per_combo), OWN=jnp.asarray(OWN),
            hard_pred=jnp.asarray(hard_pred), gated_vu=jnp.asarray(gated_vu),
            gated_ends=jnp.asarray(gated_ends), wn=wn, starts=starts,
            KC_node=jnp.asarray(KC_node), max_k=max_k,
            BC_grid=jnp.asarray(self.BC_grid),
            gc=jnp.asarray(gc), ga=jnp.asarray(ga), ge=jnp.asarray(ge),
            n_ext=self.n_ext,
        )
        return self._prepped

    def _score_grid(self, params, wp_template, x0_full, X0,
                    node_active, var_committed, var_anchor):
        """`(g_star, bc_star, best_obj)` for the optimal grid cell -- the flat
        index (into `gc`/`ga`/`ge`) and branch-combo index, and its
        objective. `jax.jit`ed once per `_DpMasterJax` (all args are traced
        arrays of problem-fixed shape, so one compile serves every cycle)."""
        import jax
        if self._jitted is None:
            _ensure_x64()
            self._prep()
            self._jitted = jax.jit(self._score_grid_core)
        return self._jitted(params, wp_template, x0_full, X0,
                            node_active, var_committed, var_anchor)

    def _score_grid_core(self, params, wp_template, x0_full, X0,
                         node_active, var_committed, var_anchor):
        import jax
        import jax.numpy as jnp
        from ..evolutionary_waypoint_solver.problem import jit_apply_projections
        st = self._prep()
        p, N, J = self.problem, self.n_nodes, self.n_agents
        NC, NA, E = self.NC, self.NA, self.E
        n_var, n_cond = self.n_var, self.n_cond
        n_psi, n_branch = p.n_psi, p.n_branch
        S = wp_template.shape[1]
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
        feas_g = feas[gc, ga, ge]                                         # (G,)

        # PRED (NC,NA,N,N) via active gated edges
        pred = jnp.broadcast_to(st["hard_pred"], (NC, NA, N, N))
        if ends.shape[0]:
            pred = pred | jnp.any(active_g[:, :, :, None, None]
                                  & st["gated_vu"][None, None, :, :, :], axis=2)
        pred = pred & node_active[None, None, :, None] & node_active[None, None, None, :]

        OWN_act = st["OWN"] & node_active[None, :, None]                   # (NC,N,J)

        # -- resolve layer0 over the grid -------------------------------
        assign_g = assign_oh[gc]                                          # (G,n_var,J)
        cond_g = cond[ga]                                                 # (G,n_cond)
        T_g = st["POS"][ge].astype(float)                                 # (G,N)
        f0 = jit_apply_projections(p, only_entries=self.layer0)
        wp0_g = f0(jnp.broadcast_to(wp_template[None], (G, N, S)),
                   jnp.zeros((G, n_psi)), jnp.zeros((G, n_branch)), params,
                   assign_g, cond_g, T_g, node_active, x0_full)           # (G,N,S)

        CAND = jnp.repeat(wp0_g[:, :, None, :], max_k, axis=2)            # (G,N,max_k,S)
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
            CAND = CAND.at[:, wn, :k, :].set(rows)
            CAND = CAND.at[:, wn, k:, :].set(rows[:, :1, :])

        validk = jnp.arange(max_k)[None, :] < st["KC_node"][:, None]      # (N,max_k)

        # -- dense EDGE/DEPOT per grid cell -----------------------------
        DEP, EDG = [], []
        for j in range(J):
            sj = self.sliced_of[j]
            x0j = X0[j]
            dep = jax.vmap(jax.vmap(jax.vmap(
                lambda r: jnp.asarray(sj(x0j, r)))))(CAND)                # (G,N,K)
            gk = jax.vmap(jax.vmap(lambda ru, rv: jnp.asarray(sj(ru, rv)),
                                   (None, 0)), (0, None))
            gn = jax.vmap(jax.vmap(gk, (None, 0)), (0, None))
            edg = jax.vmap(gn)(CAND, CAND)                                # (G,N,N,K,K)
            dep = jnp.where(validk[None], dep, _BIG)
            edg = jnp.where(validk[None, :, None, :, None], edg, _BIG)
            edg = jnp.where(validk[None, None, :, None, :], edg, _BIG)
            DEP.append(dep)
            EDG.append(edg)
        DEPOT = jnp.stack(DEP, axis=1)                                    # (G,J,N,K)
        EDGE = jnp.stack(EDG, axis=1)                                     # (G,J,N,N,K,K)

        order_g = st["EXT"][ge]                                           # (G,N)

        if self.objective in ("avg", "minmax"):
            actmask = OWN_act[gc[:, None], order_g]                       # (G,N,J)
            f_g = jax.vmap(lambda o, am, ed, dp: jax.vmap(
                _viterbi_cost_masked, (None, 1, 0, 0))(o, am, ed, dp))    # over J
            costs = f_g(order_g, actmask, EDGE, DEPOT)                    # (G,J)
            score = costs.sum(1) if self.objective == "avg" else costs.max(1)
            score = jnp.where(feas_g, score, jnp.inf)
            g_star = jnp.argmin(score)
            return g_star, jnp.int32(0), score[g_star]

        # makespan: branch combos couple with cross-agent waiting
        BC = st["BC_grid"].shape[0]
        BR = jnp.zeros((NC, BC, N, J), jnp.int32)
        for bi in range(len(self.branched)):
            wn = int(st["wn"][bi])
            own_c = st["owner_per_combo"][:, bi]                          # (NC,)
            BR = BR.at[jnp.arange(NC)[:, None], jnp.arange(BC)[None, :],
                       wn, own_c[:, None]].set(st["BC_grid"][None, :, bi])
        BR_g = BR[gc]                                                     # (G,BC,N,J)

        def cell(order, br, edge_m, depot_m, own_c, pred_ca):
            arr = _makespan_arr_jax(order, br, edge_m, depot_m, own_c, pred_ca, J)
            return jnp.max(jnp.where(node_active, arr, -_BIG))

        ms = jax.vmap(jax.vmap(cell, (None, 0, None, None, None, None)),
                      (0, 0, 0, 0, 0, 0))(
            order_g, BR_g, EDGE, DEPOT, OWN_act[gc], pred[gc, ga])        # (G,BC)
        ms = jnp.where(feas_g[:, None], ms, jnp.inf)
        flat = jnp.argmin(ms)
        g_star, bc_star = flat // BC, flat % BC
        return g_star, bc_star, ms[g_star, bc_star]

    def __call__(self, *a, **k):
        return self.run_vec(*a, **k)

    def run_vec(self, params, wp_template, x0_full, x0_by_agent,
                node_active=None, var_committed=None, var_anchor=None):
        """Run the jitted grid search to pick the optimal (assignment, aux,
        extension) cell, then reconstruct the full `solve_dp_master`-shaped
        result dict via the scalar `_ext_full_*_score` body on that one cell
        (same trick as `dp_master._ext_full_batched`)."""
        import jax.numpy as jnp
        import numpy as _np
        p = self.problem
        n_nodes, n_agents, n_var, n_cond = self.n_nodes, self.n_agents, self.n_var, self.n_cond
        node_active = (_np.ones(n_nodes, bool) if node_active is None
                       else _np.asarray(node_active, bool))
        var_committed = (_np.zeros(n_var, bool) if var_committed is None
                         else _np.asarray(var_committed, bool))
        var_anchor = (_np.zeros(n_var, int) if var_anchor is None
                      else _np.asarray(var_anchor, int))
        x0_of = (dict(x0_by_agent) if isinstance(x0_by_agent, dict)
                 else {a: _np.asarray(x0_by_agent) for a in range(n_agents)})
        X0 = jnp.stack([jnp.asarray(x0_of[j], float) for j in range(n_agents)])

        g_star, bc_star, best_obj = self._score_grid(
            jnp.asarray(params), jnp.asarray(wp_template), jnp.asarray(x0_full), X0,
            jnp.asarray(node_active), jnp.asarray(var_committed), jnp.asarray(var_anchor))
        g_star, bc_star = int(g_star), int(bc_star)
        st = self._prep()
        c = int(_np.asarray(st["gc"])[g_star])
        a = int(_np.asarray(st["ga"])[g_star])
        e = int(_np.asarray(st["ge"])[g_star])
        if not _np.isfinite(float(best_obj)):
            return dict(status="INFEASIBLE", objective=None, branch={}, assignment={},
                        aux={}, time={}, routes={}, agent_cost={})

        ov = _np.where(var_committed, var_anchor, _np.asarray(self.A[c], int))
        aux = tuple(int(x) for x in self.AUX[a])
        remaining = [n for n in range(n_nodes) if node_active[n]]
        ext = tuple(int(x) for x in self.EXT[e] if node_active[int(x)])
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
        _as0, _tv, wp0, rows_by_node, node_owner_of = _full_ext_resolve(
            p, ext, n_nodes, owned, ov, aux, x0_full,
            self.layer0, self.branched, wp_template, node_active=node_active)

        # Price the winning cell from numpy cost tables built in ONE batched
        # vmap per agent (structure.build_*_cost_table), then run the numpy
        # `_ext_nonfull_*` scorer -- NOT `_ext_full_*_score`, whose eager
        # per-(u,v,branch) `sliced_of` loop is ~1s when `sliced_of` is a jnp
        # fn with a few hundred branch combos (the dp_backend="jax" case).
        edge_tables, depot_tables, branch_counts, agent_keys = {}, {}, {}, {}
        for j, ns in owned.items():
            ns = sorted(ns)
            rj = {n: rows_by_node[n] for n in ns if n in rows_by_node}
            pairs = [(u, v) for u in ns for v in ns if u != v]
            edge_tables[j] = build_edge_cost_table(rj, wp0, pairs, edge_cost_fn=self.sliced_of[j])
            depot_tables[j] = build_depot_cost_table(rj, wp0, x0_of[j], ns,
                                                    edge_cost_fn=self.sliced_of[j])
            branch_counts[j] = {n: (rows_by_node[n].shape[0] if n in rows_by_node else 1)
                                for n in ns}
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
            return dict(status="INFEASIBLE", objective=None, branch={}, assignment={},
                        aux={}, time={}, routes={}, agent_cost={})
        br_raw, ovv, auxx, arr, agent_seq, per_agent = best_sol
        # `_ext_nonfull_*` keys `branch` by (node, _inst_key) for every routed
        # node; solve_dp_master's full path (and mpc.py's materialisation) key
        # only the multi-branch entries by (write_node, entry_owner-int). Remap.
        br = {}
        for n, (_e, owner) in node_owner_of.items():
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
                       max_assign_combos=4096, max_orders=20000, max_branch_combos=4096):
    """Build the exhaustive static enumeration for `problem` and return an
    object whose `run_python` (piece 1) / `run` (piece 3) solves the discrete
    subproblem for a cycle's continuous data. See the module docstring."""
    if objective not in ("makespan", "minmax", "avg"):
        raise ValueError(f"unknown objective {objective!r}")
    return _DpMasterJax(problem, objective, edge_cost_fn, max_assign_combos,
                        max_orders, max_branch_combos)

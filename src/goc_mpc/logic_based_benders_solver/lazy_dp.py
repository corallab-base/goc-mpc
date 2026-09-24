"""Lazy, order-aware exact coupled-makespan DP.

`coupled_dp.py` solves the order and every branch together, but it needs a
cost table `EDGE[j, u, v, b_u, b_v]` built BEFORE the DP runs, and that table
assumes a node's candidate rows are fixed by `(node, branch)`.  On real
scenes they are not:

  * ORDER.  The auto-derived stationary / rigid-carry pins (spec.py's
    `_resolve_stationary_objects` / `_resolve_holds`) are gated on the
    decoded node rank, so the object pose an IK pin reads -- and with it all
    of that IK's branch rows -- can change with the visiting order.
  * UPSTREAM BRANCHES.  A rigid-carry pin sets a carried object's pose to
    `obj_u + FK(q_v) - FK(q_u)`: it reads ROBOT rows, so a later IK target
    depends on which branch another robot chose earlier (a hand-off).

This engine resolves every node's rows AT THE MOMENT THE DP SCHEDULES IT,
when everything they depend on is known, and evaluates costs lazily.

The state
---------
`(S, last, carried, mem)`:
  S        the scheduled active nodes (an order ideal of the resolved DAG);
  last[j]  agent j's last node and a key for its RESOLVED robot row -- this
           replaces coupled_dp's "branch", so a row that differs by context
           and a row that differs by branch are the same kind of thing;
  carried  the resolved values, at already-scheduled nodes, that a pin at a
           not-yet-scheduled node still reads (the "row frontier" -- the
           analogue, for values, of the time frontier);
  mem      the schedule order of the scheduled nodes that a still-pending
           varying gate references.  `rank[x] < rank[y]` is decided for any
           two scheduled nodes in `mem`, which is all a gate needs.
Labels are exactly coupled_dp's: Pareto sets of times over the precedence
open frontier plus the running makespan.

Laziness, and what keeps it cheap
---------------------------------
Only the RELEVANT projections are ever resolved: those that write an agent
column some owner routes through, plus everything they read, transitively
(`structure.projection_dependencies`).  Everything else cannot move a cost.
Each DP level runs in three batched phases: (A) resolve every new
(node, context) through `apply_projections` with a constructed
`precomputed_rank`, grouped per node; (B) evaluate every leg cost not
already cached, grouped per agent; (C) expand and Pareto-merge.  Both caches
are keyed by VALUE -- rows by the resolved values and rank facts they were
computed from, costs by the robot rows themselves -- so two histories that
produce the same target share rows, costs and DP states.  The cost cache
persists across solves (it depends only on `edge_cost_fn`); the row cache is
per solve (rows depend on params / x0).

A gate is only "varying" if `projection_dependencies` could not prove it
constant.  When a node is scheduled, its varying gates may still reference
nodes that are NOT yet scheduled.  The engine then evaluates the pin under
every precedence-consistent completion of those nodes; if the value does not
depend on the completion it is used, and if it does, the pin's value is
genuinely not decided yet at that point in the schedule and this raises
`NotImplementedError` rather than guess.

Correctness guard
-----------------
After the DP, the winner is re-resolved with ALL projections under its true
rank and branches, and re-scored by an independent forward pass that prices
each leg from the cost cache under the keys of THOSE rows.  A leg whose row
the DP never priced, or a total that differs from the DP value, means
incremental resolution disagreed with whole-schedule resolution, and raises
`AssertionError`.  That resolved `wp` is also what is returned, so the
waypoints a caller seeds from are exactly the ones that were priced.

The check deliberately does not re-evaluate the cost function: a float32
cost (po_goc_mpc's NTField travel time) is not batch-size invariant -- up to
2.4e-4 relative at batch 64 -- so re-evaluating would conflate that noise
with a real disagreement.  Costs are evaluated in fixed chunks of
`cost_batch` (default 8, measured to match single-pair evaluation to ~2e-6)
so they are reproducible.  Comparing this engine's makespan with another
backend's is only meaningful to that precision, unless both are re-scored
one pair at a time.

Scope / fences (raise, never approximate):
  * a varying gate with unknown `gate_nodes`;
  * a relevant pin reading a value at an active node that is not yet
    scheduled (its branch would have to be chosen early);
  * a relevant branched pin at an already-passed node whose output an active
    node still reads (the committed branch is not known here);
  * `max_states`.
Truncating a Pareto front at `max_labels` is reported (`exact`), not fenced.
"""

import itertools
import weakref
from dataclasses import dataclass, field

import numpy as np

from .cpsat_model import _agent_sliced_cost_fn
from .dp_master import _agent_owned_nodes, _resolve_precedence
from .structure import (_batched_cost_fn, _hard_reach, node_instances,
                        projection_dependencies)

_BIG = 1e12
_SENT = -1


def _ensure_x64():
    import jax
    if not jax.config.read("jax_enable_x64"):
        jax.config.update("jax_enable_x64", True)


def _bucket(n):
    """Next power of two >= n: batched calls are padded to a few fixed sizes
    so a changing batch does not recompile every time."""
    return 1 << max(0, (int(n) - 1).bit_length())


_resolver_cache = weakref.WeakKeyDictionary()


def _resolver(problem, idx):
    """`apply_projections` over just the entries `idx` (problem order), jitted
    and cached per (problem, idx), taking an explicit `precomputed_rank`:
    `(wp, proj_branch, params, assign, cond, rank, node_active, x0) -> wp`."""
    import jax
    import jax.numpy as jnp
    from ..evolutionary_waypoint_solver.problem import apply_projections
    per = _resolver_cache.setdefault(problem, {})
    f = per.get(idx)
    if f is None:
        oe = tuple(problem.projections[i] for i in idx)

        def run(wp, pb, params, assign, cond, rank, node_active, x0):
            psi = jnp.zeros((wp.shape[0], problem.n_psi))
            return apply_projections(problem, wp, psi, pb, params, assign=assign,
                                     cond_binary=cond, node_active=node_active, x0=x0,
                                     only_entries=oe, precomputed_rank=rank)
        f = jax.jit(run)
        per[idx] = f
    return f


def _extensions(nodes, reach, cap):
    """All orders of `nodes` consistent with `reach` (u -> must-come-after
    set); None if more than `cap`."""
    nodes = sorted(nodes)
    before = {x: {y for y in nodes if x in reach[y]} for x in nodes}
    out = []

    def rec(rest, prefix):
        if len(out) > cap:
            return
        if not rest:
            out.append(tuple(prefix))
            return
        for x in sorted(rest):
            if before[x] & rest:
                continue
            prefix.append(x)
            rec(rest - {x}, prefix)
            prefix.pop()

    rec(set(nodes), [])
    return None if len(out) > cap else out


@dataclass
class _Plan:
    """Everything the DP needs about one (assignment, aux, node_active) that
    does not depend on cycle data.  Cached, so an unchanged scene shape
    re-plans nothing across MPC cycles."""
    active: np.ndarray
    act_nodes: tuple
    owners: list          # per node: agents routing through it (active only)
    preds: list           # per node: active precedence predecessors
    succs: list
    reach: list           # resolved-precedence reach, for gate completions
    owned_active: list    # per agent: frozenset of active nodes it visits
    entries_at: list      # per node: relevant entry indices, problem order
    branched_at: list     # per node: [(entry idx, K)]
    combos_at: list       # per node: list of branch tuples
    reads_at: list        # per node: sorted ((u, col)) read from other nodes
    carry_out: list       # per node: sorted cols read at this node by others
    readers: list         # per node: frozenset of nodes that read it
    gnodes_at: list       # per node: union gate_nodes of its varying relevant pins


@dataclass
class LazyResult:
    """One cell's optimum.  `order` is every node (passed ones first, then the
    DP's sequence), `branch` maps entry index -> chosen branch for every
    branched entry the DP resolved, `wp` is the full resolved waypoint array
    under that order and those branches -- exactly what was priced."""
    makespan: float
    order: list
    branch: dict
    wp: np.ndarray
    exact: bool
    max_front: int
    stats: dict = field(default_factory=dict)


class LazyMakespanDP:
    """Build once per problem; `solve_cell(...)` per (assignment, aux) cell per
    cycle.  See the module docstring."""

    def __init__(self, problem, edge_cost_fn=None, max_labels=16,
                 max_states=500_000, dominance_tol=1e-9,
                 max_gate_completions=5040, cost_batch=8):
        _ensure_x64()
        p = problem
        self.problem = p
        self.N, self.J, self.dim = int(p.n_nodes), int(p.n_agents), int(p.dim)
        self.entries = list(p.projections)
        self.deps = projection_dependencies(p)
        ecf = edge_cost_fn if edge_cost_fn is not None else getattr(p, "edge_cost_fn", None)
        self.sliced = [_agent_sliced_cost_fn(ecf, j, self.dim) for j in range(self.J)]
        self.batched = [_batched_cost_fn(s) for s in self.sliced]
        self.instances = node_instances(p)
        self.ordering_edges = list(p.ordering_edges)
        self.max_labels = int(max_labels)
        self.max_states = int(max_states)
        self.tol = float(dominance_tol)
        self.max_gate_completions = int(max_gate_completions)
        self.cost_batch = int(cost_batch)
        self._plans = {}
        # value-keyed across solves: agent j's (row_u slice, row_v slice) -> cost
        self._cost = [dict() for _ in range(self.J)]
        self._keys = {}          # bytes -> small int
        self._key_rows = {}      # small int -> a full representative row

    # ------------------------------------------------------------------
    def _intern(self, b, row=None):
        k = self._keys.get(b)
        if k is None:
            k = len(self._keys)
            self._keys[b] = k
        if row is not None and k not in self._key_rows:
            # copy: the cost cache outlives this solve, and a caller's array
            # (e.g. a pure_callback input buffer) can be reused after it
            self._key_rows[k] = (np.array(row, copy=True) if isinstance(row, np.ndarray)
                                 else row)
        return k

    def _slice(self, j):
        return slice(j * self.dim, (j + 1) * self.dim)

    def _agent_key(self, j, row):
        return self._intern((j, row[self._slice(j)].tobytes()), row)

    # ------------------------------------------------------------------
    def _plan(self, ov, aux, node_active):
        key = (tuple(int(x) for x in ov), tuple(int(x) for x in aux),
               np.asarray(node_active, bool).tobytes())
        hit = self._plans.get(key)
        if hit is not None:
            return hit
        p, N, J = self.problem, self.N, self.J
        active = np.asarray(node_active, bool)
        owned = _agent_owned_nodes(p, self.instances, np.asarray(ov, int))
        owners = [tuple(j for j in range(J) if active[v] and v in owned.get(j, ()))
                  for v in range(N)]
        P = {(u, v) for (u, v) in _resolve_precedence(p, self.ordering_edges,
                                                      np.asarray(ov, int), aux)
             if active[u] and active[v]}
        preds = [tuple(sorted(u for (u, w) in P if w == v)) for v in range(N)]
        succs = [tuple(sorted(w for (u, w) in P if u == v)) for v in range(N)]
        reach = _hard_reach(N, sorted(P))

        needed = {(v, c) for v in range(N) for j in owners[v]
                  for c in range(j * self.dim, (j + 1) * self.dim)}
        rel = {i for i, e in enumerate(self.entries) if e.write_cols & needed}
        for i in list(rel):
            rel |= self.deps[i].upstream
        entries_at = [tuple(i for i in sorted(rel) if int(self.entries[i].write_node) == v)
                      for v in range(N)]
        writer = {}
        for i in sorted(rel):
            for nc in self.entries[i].write_cols:
                writer[(int(nc[0]), int(nc[1]))] = i
        reads_at = [set() for _ in range(N)]
        carry_out = [set() for _ in range(N)]
        readers = [set() for _ in range(N)]
        for i in rel:
            v = int(self.entries[i].write_node)
            for (u, c) in self.entries[i].read_cols:
                u, c = int(u), int(c)
                if u != v and (u, c) in writer:
                    reads_at[v].add((u, c))
                    carry_out[u].add(c)
                    readers[u].add(v)
        gnodes_at = []
        for v in range(N):
            g = set()
            for i in entries_at[v]:
                if self.deps[i].gate != "varying":
                    continue
                if self.entries[i].gate_nodes is None:
                    raise NotImplementedError(
                        f"lazy_dp: pin #{i} at node {v} has a varying gate with "
                        "unknown gate_nodes -- cannot tell what order information "
                        "it needs")
                g |= set(int(n) for n in self.entries[i].gate_nodes)
            gnodes_at.append(frozenset(g))
        branched_at = [[(i, int(self.entries[i].discrete_params)) for i in entries_at[v]
                        if int(self.entries[i].discrete_params) > 1] for v in range(N)]
        combos_at = [list(itertools.product(*[range(k) for _i, k in branched_at[v]]))
                     for v in range(N)]
        for v in range(N):
            if not active[v] and branched_at[v] and readers[v] & set(np.flatnonzero(active)):
                raise NotImplementedError(
                    f"lazy_dp: passed node {v} has a branched pin whose output an "
                    "active node still reads -- its committed branch is not "
                    "known to the skeleton search")
        plan = _Plan(
            active=active, act_nodes=tuple(int(v) for v in np.flatnonzero(active)),
            owners=owners, preds=preds, succs=succs, reach=reach,
            owned_active=[frozenset(v for v in owned.get(j, ()) if active[v])
                          for j in range(J)],
            entries_at=entries_at, branched_at=branched_at, combos_at=combos_at,
            reads_at=[tuple(sorted(r)) for r in reads_at],
            carry_out=[tuple(sorted(c)) for c in carry_out],
            readers=[frozenset(r) for r in readers], gnodes_at=gnodes_at)
        self._plans[key] = plan
        return plan

    # ------------------------------------------------------------------
    @staticmethod
    def _topo(plan, nodes, chain=()):
        """A linear extension of `nodes` under the resolved precedence plus
        the extra chain `chain[0] -> chain[1] -> ...` (Kahn, lowest id first).
        Every rank a pin is evaluated against must be a REAL linear extension:
        `projection_dependencies` only proved gates constant over those, and
        a gate handed a precedence-violating rank can flip."""
        import heapq
        nodes = set(nodes)
        pred = {n: {u for u in plan.preds[n] if u in nodes} for n in nodes}
        for a, b in zip(chain, chain[1:]):
            pred[b].add(a)
        succ = {n: set() for n in nodes}
        for n, ps in pred.items():
            for u in ps:
                succ[u].add(n)
        indeg = {n: len(ps) for n, ps in pred.items()}
        q = [n for n in nodes if indeg[n] == 0]
        heapq.heapify(q)
        out = []
        while q:
            n = heapq.heappop(q)
            out.append(n)
            for w in succ[n]:
                indeg[w] -= 1
                if indeg[w] == 0:
                    heapq.heappush(q, w)
        if len(out) != len(nodes):
            raise AssertionError("lazy_dp: precedence + schedule chain has a cycle")
        return out

    def _ranks(self, plan, mem, S_nodes, v, gnodes):
        """`(Q, N)` rank vectors for resolving node `v` (or the passed prefix
        when `v` is None), each a genuine linear extension of the resolved
        precedence: passed nodes -1; the scheduled nodes in an order that
        keeps `mem` in its TRUE schedule order; then `v`; then the
        unscheduled nodes, once per precedence-consistent completion of the
        ones `gnodes` references (the rest interleaved topologically)."""
        N = self.N
        r = np.zeros(N, np.int32)
        r[~plan.active] = -1
        pos = 0
        for n in self._topo(plan, S_nodes, tuple(mem)):
            r[n] = pos
            pos += 1
        if v is not None:
            r[v] = pos
            pos += 1
        done = set(S_nodes) | ({v} if v is not None else set())
        unsched = [n for n in plan.act_nodes if n not in done]
        ug = sorted(n for n in unsched if n in gnodes)
        comps = _extensions(ug, plan.reach, self.max_gate_completions) if ug else [()]
        if comps is None:
            raise NotImplementedError(
                f"lazy_dp: > {self.max_gate_completions} completions of "
                f"unscheduled gate nodes {ug} at node {v}")
        out = []
        for comp in comps:
            rr = r.copy()
            for q, n in enumerate(self._topo(plan, unsched, tuple(comp)), start=pos):
                rr[n] = q
            out.append(rr)
        return np.stack(out)

    def _resolve(self, plan, v, contexts, ctx_args, run):
        """Phase A for one node: resolve every listed context of node `v` in
        ONE batched call.  `contexts[k]` -> `(carried values, mem, S)`.
        Fills `run['ctx'][key] = [(branch tuple, {agent: key}, carry key)]`."""
        import jax.numpy as jnp
        p, N = self.problem, self.N
        wp_t, params, x0, assign1, cond1 = ctx_args
        combos = plan.combos_at[v]
        B = len(combos)
        idx = plan.entries_at[v]
        rows_meta = []            # per context: (key, Q)
        WP, PB, RK = [], [], []
        for key, (cvals, mem, S_nodes) in contexts:
            ranks = self._ranks(plan, mem, S_nodes, v, plan.gnodes_at[v])
            Q = ranks.shape[0]
            base = wp_t.copy()
            for (u, cols, vals) in cvals:
                base[u, list(cols)] = vals
            for b in combos:
                pb = np.zeros(p.n_branch)
                for (i, _k), bi in zip(plan.branched_at[v], b):
                    pb[self.entries[i].branch_slice.start + bi] = 1.0
                for q in range(Q):
                    WP.append(base)
                    PB.append(pb)
                    RK.append(ranks[q])
            rows_meta.append((key, Q))
        n = len(WP)
        if idx:
            m = _bucket(n)
            pad = m - n
            WPa = np.stack(WP + [WP[-1]] * pad)
            PBa = np.stack(PB + [PB[-1]] * pad)
            RKa = np.stack(RK + [RK[-1]] * pad)
            out = np.asarray(_resolver(p, idx)(
                jnp.asarray(WPa), jnp.asarray(PBa), jnp.asarray(params),
                jnp.broadcast_to(assign1, (m,) + assign1.shape[1:]),
                jnp.broadcast_to(cond1, (m,) + cond1.shape[1:]),
                jnp.asarray(RKa), jnp.asarray(plan.active), jnp.asarray(x0)))[:n]
            run["resolver_calls"] += 1
        else:
            out = np.stack(WP)
        rows = out[:, v, :]
        # what the DP consumes at v: the owners' robot slices and the carried cols
        watch = sorted({c for j in plan.owners[v]
                        for c in range(j * self.dim, (j + 1) * self.dim)}
                       | set(plan.carry_out[v]))
        k0 = 0
        for key, Q in rows_meta:
            res = []
            for bi, b in enumerate(combos):
                blk = rows[k0 + bi * Q: k0 + (bi + 1) * Q]
                if Q > 1 and not np.all(blk[:, watch] == blk[:1, watch]):
                    raise NotImplementedError(
                        f"lazy_dp: the pins at node {v} depend on the order of "
                        f"nodes not yet scheduled when {v} is ({sorted(plan.gnodes_at[v])}) "
                        "-- their value is not decided at that point in the schedule")
                row = np.array(blk[0])
                akeys = {j: self._agent_key(j, row) for j in plan.owners[v]}
                cols = plan.carry_out[v]
                ck = (self._intern(("c", v, cols, row[list(cols)].tobytes()),
                                   (cols, row[list(cols)].copy())) if cols else None)
                res.append((tuple(b), akeys, ck))
            run["ctx"][key] = res
            run["contexts"] += 1
            k0 += B * Q

    # ------------------------------------------------------------------
    def _costs(self, pending):
        """Phase B: evaluate every uncached `(j, key_u, key_v)` leg, in chunks
        of exactly `cost_batch` pairs per agent (the last one padded).

        A FIXED chunk size, not "as big as possible": a float32 cost is not
        batch-size invariant.  Measured on po_goc_mpc's NTField travel time,
        batches of <= 8 match single-pair evaluation to ~2e-6 relative, while
        a batch of 64 moves every pair by up to 2.4e-4 relative -- enough to
        shift a makespan by ~1e-3 and reorder near-ties.  One fixed shape also
        means one compile per agent."""
        import jax.numpy as jnp
        by_j = {}
        for (j, ku, kv) in pending:
            if (ku, kv) not in self._cost[j]:
                by_j.setdefault(j, set()).add((ku, kv))
        n_eval, cb = 0, self.cost_batch
        for j, pairs in by_j.items():
            pairs = sorted(pairs)
            for c0 in range(0, len(pairs), cb):
                chunk = pairs[c0:c0 + cb]
                A = np.stack([self._key_rows[ku] for ku, _ in chunk])
                Bv = np.stack([self._key_rows[kv] for _, kv in chunk])
                pad = cb - len(chunk)
                if pad:
                    A = np.concatenate([A, np.repeat(A[-1:], pad, 0)])
                    Bv = np.concatenate([Bv, np.repeat(Bv[-1:], pad, 0)])
                c = np.asarray(self.batched[j](jnp.asarray(A), jnp.asarray(Bv)))[:len(chunk)]
                for (ku, kv), cv in zip(chunk, c):
                    self._cost[j][(ku, kv)] = float(cv)
            n_eval += len(pairs)
        return n_eval

    # ------------------------------------------------------------------
    def _tf(self, plan, S, agents, memo):
        """Sorted time-frontier nodes of a state: scheduled nodes with an
        unscheduled precedence successor, plus every active agent's last."""
        k = (S, agents)
        hit = memo.get(k)
        if hit is None:
            tf = {u for u in plan.act_nodes if (S >> u) & 1
                  and any(not (S >> w) & 1 for w in plan.succs[u])}
            tf |= {a[0] for a in agents if a != _SENT}
            hit = tuple(sorted(tf))
            memo[k] = hit
        return hit

    def _pareto(self, cands):
        tol, P = self.tol, self.max_labels
        n = len(cands)
        labs = [c[0] for c in cands]
        keep = []
        for i in range(n):
            li = labs[i]
            dom = False
            for j in range(n):
                if j == i:
                    continue
                lj = labs[j]
                if all(a <= b + tol for a, b in zip(lj, li)) and (
                        any(a < b - tol for a, b in zip(lj, li)) or j < i):
                    dom = True
                    break
            if not dom:
                keep.append(cands[i])
        keep.sort(key=lambda c: c[0][-1])
        return keep[:P], len(keep)

    # ------------------------------------------------------------------
    def solve_cell(self, ov, aux, params, wp_template, x0_full, X0, node_active):
        """Exact makespan optimum for one (assignment `ov`, aux) cell, or None
        if it has no feasible schedule.  All arrays numpy."""
        import jax.numpy as jnp
        p, N, J = self.problem, self.N, self.J
        plan = self._plan(ov, aux, node_active)
        wp_t = np.asarray(wp_template, float)
        X0 = np.asarray(X0, float)
        assign1 = jnp.asarray(np.eye(J)[np.asarray(ov, int)][None]
                              if len(ov) else np.zeros((1, 0, J)))
        cond1 = jnp.asarray(np.asarray(aux, float)[None])
        ctx_args = (wp_t, np.asarray(params), np.asarray(x0_full), assign1, cond1)
        run = dict(ctx={}, contexts=0, resolver_calls=0, cost_evals=0, states=0)

        # -- the passed prefix: resolved once, its carried values seed the DP
        carried0 = {}
        inact = [v for v in range(N) if not plan.active[v]]
        pidx = tuple(i for v in inact for i in plan.entries_at[v])
        if inact and any(plan.carry_out[v] for v in inact):
            pidx = tuple(sorted(pidx))
            gn = frozenset().union(*(plan.gnodes_at[v] for v in inact))
            ranks = self._ranks(plan, (), set(), None, gn)
            Q = ranks.shape[0]
            if pidx:
                out = np.asarray(_resolver(p, pidx)(
                    jnp.asarray(np.repeat(wp_t[None], Q, 0)),
                    jnp.zeros((Q, p.n_branch)), jnp.asarray(params),
                    jnp.broadcast_to(assign1, (Q,) + assign1.shape[1:]),
                    jnp.broadcast_to(cond1, (Q,) + cond1.shape[1:]),
                    jnp.asarray(ranks), jnp.asarray(plan.active), jnp.asarray(x0_full)))
            else:
                out = np.repeat(wp_t[None], Q, 0)
            for u in inact:
                cols = plan.carry_out[u]
                if not cols or not (plan.readers[u] & set(plan.act_nodes)):
                    continue
                vals = out[:, u, list(cols)]
                if Q > 1 and not np.all(vals == vals[:1]):
                    raise NotImplementedError(
                        f"lazy_dp: passed node {u}'s pins depend on the order of "
                        "still-active nodes")
                carried0[u] = self._intern(("c", u, cols, vals[0].tobytes()),
                                           (cols, vals[0].copy()))

        tf_memo = {}
        init_carried = tuple(sorted(carried0.items()))
        init = (0, tuple(_SENT for _ in range(J)), init_carried, ())
        levels = [{init: [((0.0,), None)]}]
        n_act = len(plan.act_nodes)
        act_set = set(plan.act_nodes)
        max_front, truncated = 1, False

        for _c in range(n_act):
            cur = levels[-1]
            # enumerate (state, v) moves once; phases reuse them
            moves = []
            for key in cur:
                S, agents, carried, mem = key
                S_nodes = {u for u in plan.act_nodes if (S >> u) & 1}
                cmap = dict(carried)
                for v in plan.act_nodes:
                    if (S >> v) & 1 or any(not (S >> u) & 1 for u in plan.preds[v]):
                        continue
                    for (u, _c2) in plan.reads_at[v]:
                        if u in act_set and u not in S_nodes:
                            raise NotImplementedError(
                                f"lazy_dp: a pin at node {v} reads node {u}'s "
                                f"value, but {u} can be scheduled after {v}")
                    reads_u = sorted({u for (u, _c2) in plan.reads_at[v]})
                    gn = plan.gnodes_at[v]
                    gsig = tuple(n for n in mem if n in gn) if gn else None
                    ckey = (v, tuple((u, cmap[u]) for u in reads_u), gsig)
                    moves.append((key, S_nodes, v, ckey, reads_u))

            # Phase A: resolve new contexts, one batched call per node
            need = {}
            for key, S_nodes, v, ckey, reads_u in moves:
                if ckey in run["ctx"] or ckey in need.get(v, {}):
                    continue
                cmap = dict(key[2])
                cvals = []
                for u in reads_u:
                    cols, vals = self._key_rows[cmap[u]]
                    want = [c for (uu, c) in plan.reads_at[v] if uu == u]
                    sel = [cols.index(c) for c in want]
                    cvals.append((u, tuple(want), vals[sel]))
                need.setdefault(v, {})[ckey] = (cvals, key[3], S_nodes)
            for v, ctxs in need.items():
                self._resolve(plan, v, list(ctxs.items()), ctx_args, run)

            # Phase B: lazily evaluate the legs these moves need
            pending = set()
            for key, _S, v, ckey, _r in moves:
                agents = key[1]
                for (_b, akeys, _ck) in run["ctx"][ckey]:
                    for j in plan.owners[v]:
                        ku = (self._intern((j, "depot", X0[j][self._slice(j)].tobytes()), X0[j])
                              if agents[j] == _SENT else agents[j][1])
                        pending.add((j, ku, akeys[j]))
            run["cost_evals"] += self._costs(pending)

            # Phase C: expand and merge
            nxt = {}
            for key, S_nodes, v, ckey, _r in moves:
                S, agents, carried, mem = key
                tf_old = self._tf(plan, S, agents, tf_memo)
                S2 = S | (1 << v)
                S2_nodes = S_nodes | {v}
                pending_gn = frozenset().union(
                    *(plan.gnodes_at[w] for w in plan.act_nodes if w not in S2_nodes)) \
                    if plan.act_nodes else frozenset()
                mem2 = tuple(n for n in mem + (v,) if n in pending_gn)
                for ci, (b, akeys, ck) in enumerate(run["ctx"][ckey]):
                    legs, agents2 = {}, list(agents)
                    for j in plan.owners[v]:
                        ku = (self._intern((j, "depot", X0[j][self._slice(j)].tobytes()), X0[j])
                              if agents[j] == _SENT else agents[j][1])
                        legs[j] = self._cost[j][(ku, akeys[j])]
                        agents2[j] = (_SENT if plan.owned_active[j] <= S2_nodes
                                      else (v, akeys[j]))
                    agents2 = tuple(agents2)
                    # a carried value lives until its last active reader is scheduled
                    car2 = [(u, k) for (u, k) in carried
                            if (plan.readers[u] & act_set) - S2_nodes]
                    if ck is not None and (plan.readers[v] & act_set) - S2_nodes:
                        car2.append((v, ck))
                    car2 = tuple(sorted(set(car2)))
                    key2 = (S2, agents2, car2, mem2)
                    tf_new = self._tf(plan, S2, agents2, tf_memo)
                    bucket = nxt.setdefault(key2, [])
                    for li, (lab, _bp) in enumerate(cur[key]):
                        tmap = dict(zip(tf_old, lab[:-1]))
                        t = 0.0
                        for u in plan.preds[v]:
                            t = max(t, tmap[u])
                        for j, c in legs.items():
                            base = 0.0 if agents[j] == _SENT else tmap[agents[j][0]]
                            t = max(t, base + c)
                        tmap[v] = t
                        lab2 = tuple(tmap[u] for u in tf_new) + (max(lab[-1], t),)
                        bucket.append((lab2, (key, li, v, ckey, ci)))
            merged = {}
            for key2, cands in nxt.items():
                kept, n_front = self._pareto(cands)
                max_front = max(max_front, n_front)
                truncated |= n_front > self.max_labels
                merged[key2] = kept
            run["states"] += len(merged)
            if run["states"] > self.max_states:
                raise NotImplementedError(
                    f"lazy_dp: > {self.max_states} DP states -- instance too large")
            levels.append(merged)

        final = levels[-1]
        best = None
        for key, labs in final.items():
            for li, (lab, _bp) in enumerate(labs):
                if best is None or lab[-1] < best[0]:
                    best = (lab[-1], key, li)
        if best is None:
            return None
        mk, key, li = best

        # -- backtrack
        path, steps = [], []
        for lv in range(len(levels) - 1, 0, -1):
            lab, bp = levels[lv][key][li]
            pkey, pli, v, ckey, ci = bp
            b, _ak, _ck = run["ctx"][ckey][ci]
            path.append((v, {i: bi for (i, _k), bi in zip(plan.branched_at[v], b)}))
            steps.append((v, ckey, ci))
            key, li = pkey, pli
        path.reverse()
        steps.reverse()
        order = sorted(int(v) for v in range(N) if not plan.active[v]) + [v for v, _ in path]
        branch = {}
        for _v, bmap in path:
            branch.update(bmap)

        wp_full = self._full_resolve(plan, path, branch, ctx_args)
        # kept for diagnosis: which context / row each step of the winner used
        self.last_solve = dict(plan=plan, run=run, path=path, steps=steps, wp=wp_full,
                               claimed=float(mk), X0=np.array(X0, copy=True))
        chk = self._rescore(plan, path, wp_full, X0)
        if abs(chk - mk) > 1e-6 * max(1.0, abs(mk)):
            raise AssertionError(
                f"lazy_dp: whole-schedule re-resolution scores {chk}, the DP "
                f"claimed {mk} -- incremental resolution disagreed (a bug, not a "
                "fence)")
        return LazyResult(float(mk), order, branch, wp_full, not truncated, max_front,
                          dict(states=run["states"], contexts=run["contexts"],
                               resolver_calls=run["resolver_calls"],
                               cost_evals=run["cost_evals"]))

    # ------------------------------------------------------------------
    def _full_resolve(self, plan, path, branch, ctx_args):
        """ALL projections under the winner's true rank and branches."""
        import jax.numpy as jnp
        p, N = self.problem, self.N
        wp_t, params, x0, assign1, cond1 = ctx_args
        rank = np.full(N, -1, np.int32)
        for pos, (v, _b) in enumerate(path):
            rank[v] = pos
        pb = np.zeros(p.n_branch)
        for i, e in enumerate(self.entries):
            pb[e.branch_slice.start + branch.get(i, 0)] = 1.0
        idx = tuple(range(len(self.entries)))
        if not idx:
            return wp_t.copy()
        return np.asarray(_resolver(p, idx)(
            jnp.asarray(wp_t[None]), jnp.asarray(pb[None]), jnp.asarray(params),
            assign1, cond1, jnp.asarray(rank[None]), jnp.asarray(plan.active),
            jnp.asarray(x0)))[0]

    def _rescore(self, plan, path, wp, X0):
        """Independent forward pass over the WHOLE-SCHEDULE resolution `wp`,
        pricing each leg by looking it up in the cost cache under the keys
        of `wp`'s own rows.  A missing key means incremental resolution
        produced a different row than whole-schedule resolution did -- the
        bug this guards against -- and raises.  Using the cached numbers
        (rather than re-evaluating the cost function) makes the check exact:
        it verifies rows and DP bookkeeping, and is not confused by a cost
        function whose float32 value moves with batch size (see `_costs`)."""
        t, last, mk = {}, {}, 0.0
        for v, _b in path:
            tv = max([0.0] + [t[u] for u in plan.preds[v]])
            for j in plan.owners[v]:
                if j in last:
                    ku = self._keys.get((j, wp[last[j]][self._slice(j)].tobytes()))
                else:
                    ku = self._keys.get((j, "depot", X0[j][self._slice(j)].tobytes()))
                kv = self._keys.get((j, wp[v][self._slice(j)].tobytes()))
                c = self._cost[j].get((ku, kv))
                if c is None:
                    raise AssertionError(
                        f"lazy_dp: the whole-schedule resolution of node {v} gives "
                        f"agent {j} a row the DP never priced -- incremental "
                        "resolution disagreed (a bug, not a fence)")
                tv = max(tv, (t[last[j]] if j in last else 0.0) + c)
                last[j] = v
            t[v] = tv
            mk = max(mk, tv)
        return mk

"""Exact fully-coupled multi-agent makespan DP, vectorised in JAX.

The problem this solves, for one *already fixed* assignment `A` and one
fixed auxiliary vector (so the conditional ordering edges have resolved to
a concrete precedence DAG `P`): choose every agent's visiting order over
the nodes it owns, and every `(node, agent)` branch, to minimise
`t(n+1) = max_v t(v)` under

    t(v) >= release(v)                                   external / committed
    t(v) >= t(u)                       for (u, v) in P    precedence
    t(v) >= t(prev_j(v)) + delta_j(...)  for each owner j  routing

`dp_master.py` handles this by enumerating *all linear extensions* of `P`
and, for `makespan`, *all branch combinations* on an explicit `BC` axis.
Both are factorial / exponential in the wrong quantity.  This module
replaces them with a single dynamic program.

The recursion
-------------
Held-Karp is

    g(S, e) = min_s  g(S \\ {e}, s) + d(s, e)

Four changes turn that into an exact coupled multi-agent makespan DP:

1. **Branches.**  Carry the chosen configuration of the last node in the
   state: `g(S, e, b) = min_{s,b_s} g(S\\{e}, s, b_s) + delta(s^b_s, e^b)`.

2. **Precedence.**  Restrict `S` to *order ideals* (downsets) of `P`, and
   only allow `e` with `pred(e) subset of S\\{e}`.  This is where the
   precedence structure buys back the exponent -- for a strongly ordered
   scene the ideals are a tiny fraction of `2^N`.

3. **Cost becomes time.**  `+` becomes `max`:
   `g(S,e,b) = min_{s,b_s} max( g(S\\{e},s,b_s) + delta, release(e) )`.
   Still one scalar per state -- this is the per-agent DP with frozen
   release times.

4. **Coupling.**  The release of `e` is `max_{u in pred(e)} t(u)`, and `u`
   generally belongs to *another* agent, so the M per-agent DPs cannot be
   separated.  The state becomes `(S, l, b)` with `l_j` each agent's last
   node and `b_j` its branch, and the *value becomes a vector*:

       theta = ( t(u) for u in F(S) ;  t(l_j) for each agent j ;  mk )

   where `F(S) = { u in S : some (u,w) in P has w not in S }` is the **open
   frontier** -- the already-done nodes that still gate unscheduled work --
   and `mk = max over scheduled v of t(v)` is the running makespan.
   `min` becomes `ParetoMin` over these vectors.

   Carrying only the per-agent completion times is *wrong*: two prefixes
   can reach the same `(S, l, b)` with identical `t(l_j)` for every agent
   and different `t(u)` on the frontier, and those choose different global
   makespans.  Every predecessor of an eligible `e` is guaranteed to lie in
   `F(S)` (it has the pending successor `e`), so the frontier is exactly
   the right set to carry -- no more, no less.

Exactness rests on three facts, all of which hold here:
  * every timing constraint is a difference constraint `t(b) - t(a) >= c`
    with `c >= 0` and free waiting, so all downstream times are monotone
    non-decreasing in every label component -- componentwise dominance is
    sound;
  * sorting any feasible solution's nodes by `t(v)` (ties broken by a fixed
    topological order of `P`) makes every prefix an ideal and respects each
    agent's route order, so no feasible solution is missed;
  * the objective is monotone in every label component.

The DP is therefore exact **iff no Pareto front was truncated at
`max_labels`** -- that is reported as `CoupledResult.exact`, not assumed.
A node owned by several agents simultaneously (a hand-off) is handled
natively: scheduling it advances all of its owners at once.

`collapse_finished` (default True) merges states that differ only in where
an agent that has *finished* all its work ended up -- it can never move
again, so its last node and branch cannot affect the future and its
completion time is already inside `mk`.  Measured on random instances this
is a 3-6x reduction in states for no loss of exactness, and it is on by
default.  It does cost one clean certificate: the merged states now differ
in `mk` alone, so `max_front == 1` is no longer implied by the absence of
cross-agent precedence.  With `collapse_finished=False` that implication
holds again (verified in `examples/test_coupled_dp.py`), which makes the
uncollapsed mode the one to use when you want the decoupling certificate
rather than the smaller state graph.

Representation
--------------
A label is stored dense, width `W = n + 1`: slots `0..n-1` hold `t(v)`
*canonicalised to 0 wherever `v` is irrelevant at this state*, slot `n`
holds `mk`.  Canonicalising instead of masking means plain componentwise
comparison is already exact frontier dominance, and identical prefixes
dedupe bitwise.

Every transition is fully described by static data -- which node is added,
its predecessors, and each owner's `(from-slot, hop cost)` -- because the
only run-time quantity is `t` itself.  So the whole level update is gathers
plus a fixed-shape Pareto merge, with no data-dependent control flow.  A
sentinel slot `n` holding the constant `0.0` is appended to the time vector
for padding, which also makes the depot leg fall out for free: an agent
that has not started has `l_j = n`, so `t(l_j) + hop = 0 + DEPOT[j,v,b]`.

Scope / fences (raise `NotImplementedError`, never silently approximate):
`max_states`, `max_transitions` and `max_label_bytes`.  Truncation of a
Pareto front is *not* a fence -- it degrades gracefully and is reported.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

_BIG = 1e12
_HALF_BIG = _BIG / 2.0


def _ensure_x64():
    import jax
    if not jax.config.read("jax_enable_x64"):
        jax.config.update("jax_enable_x64", True)


@dataclass
class CoupledResult:
    """`makespan` under `order` / `branch_of`, with `arr` the arrival time of
    every scheduled node (global ids throughout).

    `exact` is False iff some Pareto front hit `max_labels`, in which case
    `makespan` is an upper bound.  `max_front` is the largest front actually
    seen -- `max_front == 1` certifies the agents never traded off against
    each other, and `max_front < max_labels` certifies `exact`."""

    makespan: float
    order: list
    branch_of: dict
    arr: dict
    exact: bool
    max_front: int
    n_states: int
    n_transitions: int
    n_ideals: int


def _order_ideals(n, predmask, cap):
    """Every downset of the precedence poset, grouped by cardinality.

    `levels[c]` is the list of bitmasks `S` with `popcount(S) == c` such that
    `v in S => pred(v) subset of S`.  Returns `None` if the poset has a cycle
    (no ideal of full cardinality is reachable)."""
    levels = [[0]]
    seen = {0}
    total = 1
    for _ in range(n):
        nxt = []
        for S in levels[-1]:
            for v in range(n):
                if (S >> v) & 1:
                    continue
                if predmask[v] & ~S:
                    continue
                S2 = S | (1 << v)
                if S2 not in seen:
                    seen.add(S2)
                    nxt.append(S2)
                    total += 1
                    if total > cap:
                        raise NotImplementedError(
                            f"coupled_dp: precedence poset has > {cap} order "
                            "ideals -- too loosely ordered for the exact "
                            "coupled DP (raise max_ideals, or fall back to a "
                            "beam / per-agent decomposition)")
        levels.append(nxt)
        if not nxt:
            break
    if len(levels) <= n or not levels[n]:
        return None
    return levels


class _CoupledDP:
    """Built state graph for one resolved instance; `solve()` runs the DP."""

    def __init__(self, n_nodes, n_agents, owners, preds, KC, *, max_k=None,
                 nodes=None, release=None, max_labels=16, max_states=500_000,
                 max_transitions=4_000_000, max_ideals=1_000_000,
                 max_label_bytes=2 << 30, dominance_tol=1e-9,
                 collapse_finished=True):
        self.max_labels = int(max_labels)
        self.dominance_tol = float(dominance_tol)
        self.collapse_finished = bool(collapse_finished)

        gnodes = list(range(n_nodes)) if nodes is None else [int(v) for v in nodes]
        self.gnodes = gnodes
        n = len(gnodes)
        self.n = n
        self.n_agents = int(n_agents)
        g2l = {g: i for i, g in enumerate(gnodes)}

        self.N_global = int(n_nodes)
        self.KC = np.asarray(KC, np.int64)
        # branch stride of the cost tensors this structure will be priced
        # against -- EDGE is (J, N, N, max_k, max_k), DEPOT is (J, N, max_k).
        self.max_k = int(max_k) if max_k is not None else int(max(1, self.KC.max()))

        if release is None:
            self.release = np.zeros(n)
        else:
            r = np.asarray(release, float)
            self.release = np.array([float(r[g]) for g in gnodes])

        # --- per-node static data, in local indices -----------------------
        self.owners = [sorted(int(j) for j in owners[g]) for g in gnodes]
        predmask = np.zeros(n, dtype=np.int64)
        succmask = np.zeros(n, dtype=np.int64)
        predlist = []
        for i, g in enumerate(gnodes):
            pl = [g2l[int(u)] for u in preds[g] if int(u) in g2l]
            predlist.append(pl)
            for u in pl:
                predmask[i] |= (1 << u)
                succmask[u] |= (1 << i)
        self.predlist = predlist
        self.predmask = predmask
        self.succmask = succmask

        ownmask = np.zeros(self.n_agents, dtype=np.int64)
        for i in range(n):
            for j in self.owners[i]:
                ownmask[j] |= (1 << i)
        self.ownmask = ownmask

        # branch combinations of each node's owner set (usually a single owner)
        self.bcs = [
            list(itertools.product(*[range(max(1, int(self.KC[j, gnodes[i]])))
                                     for j in self.owners[i]])) or [()]
            for i in range(n)
        ]

        levels = _order_ideals(n, predmask, max_ideals)
        if levels is None:
            raise ValueError("coupled_dp: precedence graph has a cycle "
                             "(no order ideal reaches every node) -- infeasible")
        self.n_ideals = sum(len(L) for L in levels)

        self._enumerate_states(levels, max_states)
        self._enumerate_transitions(max_transitions)

        W = n + 1
        nbytes = (self.n_states + 2) * self.max_labels * W * 8
        if nbytes > max_label_bytes:
            raise NotImplementedError(
                f"coupled_dp: label table would be {nbytes / 2**30:.2f} GiB "
                f"({self.n_states} states x {self.max_labels} labels x {W}) "
                f"-- over max_label_bytes; lower max_labels or shrink the "
                "instance")

    # ------------------------------------------------------------------
    def _enumerate_states(self, levels, max_states):
        """States are `(S, l, b)`: an ideal, each agent's last node, and its
        branch.  Two canonicalisations keep the count down and are exact:

          * an agent with no *remaining* work (`V_j subset of S`) can never
            move again, so its `l_j` and `b_j` are irrelevant to the future
            and its completion time is already folded into `mk`; collapse
            both to the sentinel.  An agent that has not started collapses
            to the same sentinel, and the depot leg then falls out of the
            generic formula.
          * `b_j` is only meaningful when `l_j` is a real node.
        """
        n, M = self.n, self.n_agents
        SENT = n
        gn = self.gnodes
        self.SENT = SENT

        index = {}
        state_key = []
        level_of = []
        for c, L in enumerate(levels):
            for S in L:
                # per-agent choices of (l_j, b_j)
                choices = []
                for j in range(M):
                    om = int(self.ownmask[j])
                    vis = S & om
                    if vis == 0 or ((om & ~S) == 0 and self.collapse_finished):
                        choices.append([(SENT, 0)])
                        continue
                    opts = []
                    for i in range(n):
                        if (vis >> i) & 1:
                            k = max(1, int(self.KC[j, gn[i]]))
                            opts.extend((i, b) for b in range(k))
                    choices.append(opts)
                for combo in itertools.product(*choices):
                    key = (S, combo)
                    index[key] = len(state_key)
                    state_key.append(key)
                    level_of.append(c)
                    if len(state_key) > max_states:
                        raise NotImplementedError(
                            f"coupled_dp: > {max_states} DP states -- the "
                            "instance is too large for the exact coupled DP "
                            "(raise max_states, or use a beam)")

        self.index = index
        self.state_key = state_key
        self.n_states = len(state_key)
        self.level_of = np.asarray(level_of, np.int32)
        self.n_levels = len(levels)

        # relevance mask: t(u) must be carried iff u still gates unscheduled
        # work, or is some still-active agent's last node.  Everything else
        # is already absorbed into mk.
        REL = np.zeros((self.n_states + 2, n), dtype=bool)
        for s, (S, combo) in enumerate(state_key):
            for i in range(n):
                if (S >> i) & 1 and (int(self.succmask[i]) & ~S):
                    REL[s, i] = True
            for (li, _b) in combo:
                if li != SENT:
                    REL[s, li] = True
        self.REL = REL

        lvl = [[] for _ in range(self.n_levels)]
        for s in range(self.n_states):
            lvl[self.level_of[s]].append(s)
        self.levels_states = lvl
        self.L_max = max(len(x) for x in lvl)
        self.init_state = self.index[(0, tuple((SENT, 0) for _ in range(M)))]

    # ------------------------------------------------------------------
    def _enumerate_transitions(self, max_transitions):
        """One transition per `(state, eligible node, branch combo)`.

        Everything the run-time step needs is static: which slots to read
        (`T_pred`, `T_frm`), which cost-tensor entry to add (`T_eidx` /
        `T_didx`, flat indices into `EDGE` / `DEPOT`), and where to write
        (`T_node`, `T_dst`).  The only run-time quantities are the hop costs
        -- gathered from whatever `(EDGE, DEPOT)` this cycle supplies -- and
        `t` itself.  Storing indices rather than the costs themselves is what
        makes the structure reusable: an MPC cycle re-prices the same state
        graph instead of rebuilding it."""
        n, M, SENT = self.n, self.n_agents, self.SENT
        gn = self.gnodes
        Pmax = max(1, max(len(p) for p in self.predlist)) if n else 1
        Mmax = max(1, max(len(o) for o in self.owners)) if n else 1
        self.Pmax, self.Mmax = Pmax, Mmax

        NG, K = self.N_global, self.max_k
        src, dst, node, rel = [], [], [], []
        pidx, fidx, bcv, ownv = [], [], [], []
        eidx, didx, isdep, hval = [], [], [], []

        for s, (S, combo) in enumerate(self.state_key):
            for i in range(n):
                if (S >> i) & 1:
                    continue
                if int(self.predmask[i]) & ~S:
                    continue
                S2 = S | (1 << i)
                gi = gn[i]
                ow = self.owners[i]
                for bc in self.bcs[i]:
                    fs, ei, di, isd = [], [], [], []
                    for m, j in enumerate(ow):
                        bv = bc[m]
                        lj, bj = combo[j]
                        if lj == SENT:
                            # not started: the sentinel time slot holds 0.0,
                            # so `0 + DEPOT[j,v,b]` is the depot leg.
                            fs.append(SENT)
                            di.append((j * NG + gi) * K + bv)
                            ei.append(0)
                            isd.append(True)
                        else:
                            fs.append(lj)
                            u = gn[lj]
                            ei.append(((((j * NG + u) * NG + gi) * K) + bj) * K + bv)
                            di.append(0)
                            isd.append(False)
                    nc = list(combo)
                    for m, j in enumerate(ow):
                        om = int(self.ownmask[j])
                        nc[j] = ((SENT, 0)
                                 if ((om & ~S2) == 0 and self.collapse_finished)
                                 else (i, bc[m]))
                    d = self.index[(S2, tuple(nc))]

                    src.append(s)
                    dst.append(d)
                    node.append(i)
                    rel.append(float(self.release[i]))
                    pidx.append(self.predlist[i] + [SENT] * (Pmax - len(self.predlist[i])))
                    fidx.append(fs + [SENT] * (Mmax - len(fs)))
                    pad = Mmax - len(ei)
                    eidx.append(ei + [0] * pad)
                    didx.append(di + [0] * pad)
                    isdep.append(isd + [False] * pad)
                    hval.append([True] * len(ei) + [False] * pad)
                    bcv.append(list(bc) + [0] * (Mmax - len(bc)))
                    ownv.append(list(ow) + [-1] * (Mmax - len(ow)))
                    if len(src) > max_transitions:
                        raise NotImplementedError(
                            f"coupled_dp: > {max_transitions} DP transitions "
                            "-- instance too large for the exact coupled DP")

        E = len(src)
        self.n_transitions = E
        DUMMY_SRC, DUMMY_DST = self.n_states, self.n_states + 1
        # trailing padding transition: reads the always-invalid dummy source,
        # writes the dummy sink.
        self.T_src = np.asarray(src + [DUMMY_SRC], np.int32)
        self.T_dst = np.asarray(dst + [DUMMY_DST], np.int32)
        self.T_node = np.asarray(node + [0], np.int32)
        self.T_rel = np.asarray(rel + [0.0], float)
        self.T_pred = np.asarray(pidx + [[SENT] * Pmax], np.int32)
        self.T_frm = np.asarray(fidx + [[SENT] * Mmax], np.int32)
        self.T_eidx = np.asarray(eidx + [[0] * Mmax], np.int32)
        self.T_didx = np.asarray(didx + [[0] * Mmax], np.int32)
        self.T_isdep = np.asarray(isdep + [[False] * Mmax], bool)
        self.T_hvalid = np.asarray(hval + [[False] * Mmax], bool)
        self.T_bc = np.asarray(bcv + [[0] * Mmax], np.int32)
        self.T_own = np.asarray(ownv + [[-1] * Mmax], np.int32)

        # Incoming transitions, grouped per *level*.  A single global
        # `(n_states, max_indeg)` table would pad every state out to the
        # worst in-degree anywhere in the graph -- measured at 3-6x wasted
        # merge slots on real instances -- so each level carries its own
        # width instead, at the cost of one kernel compile per distinct
        # `(n_level_states, level_max_indeg)` shape.
        counts = np.zeros(self.n_states + 2, np.int64)
        for d in dst:
            counts[d] += 1
        self.max_indeg = int(max(1, counts.max()))
        inc = [[] for _ in range(self.n_states + 2)]
        for e, d in enumerate(dst):
            inc[d].append(e)

        self.level_dst, self.level_in = [], []
        for ss in self.levels_states:
            if not ss:
                self.level_dst.append(np.zeros(0, np.int32))
                self.level_in.append(np.zeros((0, 1), np.int32))
                continue
            D = max(1, max(len(inc[s]) for s in ss))
            tab = np.full((len(ss), D), E, np.int32)
            for r, s in enumerate(ss):
                tab[r, :len(inc[s])] = inc[s]
            self.level_dst.append(np.asarray(ss, np.int32))
            self.level_in.append(tab)

    # ------------------------------------------------------------------
    def _kernels(self):
        """Device tables + one jitted level kernel per distinct
        `(n_level_states, level_max_indeg)` shape, memoised on the instance.
        Repeated pricings of the same structure -- an MPC cycle re-solving
        with new costs -- rebuild nothing and recompile nothing."""
        if getattr(self, "_kern", None) is not None:
            return self._kern

        import jax
        import jax.numpy as jnp
        from jax import lax, vmap
        _ensure_x64()

        n, P, W = self.n, self.max_labels, self.n + 1
        tol = self.dominance_tol

        T_src = jnp.asarray(self.T_src)
        T_dst = jnp.asarray(self.T_dst)
        T_node = jnp.asarray(self.T_node)
        T_rel = jnp.asarray(self.T_rel)
        T_pred = jnp.asarray(self.T_pred)
        T_frm = jnp.asarray(self.T_frm)
        REL = jnp.asarray(self.REL)
        ar_n = jnp.arange(n)
        ar_P = jnp.arange(P, dtype=jnp.int32)
        ar_2P = jnp.arange(2 * P)

        self._jt = dict(
            T_src=T_src, T_node=T_node,
            T_eidx=jnp.asarray(self.T_eidx), T_didx=jnp.asarray(self.T_didx),
            T_isdep=jnp.asarray(self.T_isdep),
            T_hvalid=jnp.asarray(self.T_hvalid),
            T_bc=jnp.asarray(self.T_bc), T_own=jnp.asarray(self.T_own),
            gnodes=jnp.asarray(np.asarray(self.gnodes, np.int32)),
            term=jnp.asarray(np.asarray(self.levels_states[n], np.int32)),
        )

        def expand(LAB, tr, HOP):
            """Candidate labels produced by one incoming transition per state.
            `tr` (L,) -> `(L, P, W)`; the whole max-plus step is gathers."""
            th = LAB[T_src[tr]]                                   # (L,P,W)
            arr, mk = th[..., :n], th[..., n]
            ae = jnp.concatenate(
                [arr, jnp.zeros(arr.shape[:-1] + (1,), arr.dtype)], -1)

            pi = jnp.broadcast_to(T_pred[tr][:, None, :],
                                  arr.shape[:-1] + (self.Pmax,))
            pr = jnp.take_along_axis(ae, pi, -1).max(-1)          # precedence
            fi = jnp.broadcast_to(T_frm[tr][:, None, :],
                                  arr.shape[:-1] + (self.Mmax,))
            lg = (jnp.take_along_axis(ae, fi, -1)
                  + HOP[tr][:, None, :]).max(-1)                  # routing legs

            t = jnp.maximum(jnp.maximum(T_rel[tr][:, None], pr), lg)
            oh = ar_n[None, None, :] == T_node[tr][:, None, None]
            arr2 = jnp.where(REL[T_dst[tr]][:, None, :],
                             jnp.where(oh, t[..., None], arr), 0.0)
            cand = jnp.concatenate([arr2, jnp.maximum(mk, t)[..., None]], -1)
            return jnp.where((mk >= _HALF_BIG)[..., None], _BIG, cand)

        def merge(A, Atr, Asp, B, Btr, Bsp):
            """Pareto-min of two size-P label sets, truncated back to P.

            `dom[i,j]` = "j dominates i": componentwise <= within `tol`, and
            either strictly better somewhere or lower index (which breaks
            ties so exactly one of a duplicate group survives)."""
            C = jnp.concatenate([A, B], 0)                        # (2P,W)
            Ctr = jnp.concatenate([Atr, Btr], 0)
            Csp = jnp.concatenate([Asp, Bsp], 0)
            ok = C[:, -1] < _HALF_BIG
            d = C[None, :, :] - C[:, None, :]                     # d[i,j]=C[j]-C[i]
            dom = ((d <= tol).all(-1)
                   & ((d < -tol).any(-1) | (ar_2P[None, :] < ar_2P[:, None]))
                   & ok[None, :])
            keep = ok & ~dom.any(1)
            perm = jnp.argsort(jnp.where(keep, C[:, -1], 2 * _BIG))[:P]
            k = keep[perm]
            return (jnp.where(k[:, None], C[perm], _BIG),
                    jnp.where(k, Ctr[perm], -1), jnp.where(k, Csp[perm], -1),
                    keep.sum().astype(jnp.int32))

        merge_v = vmap(merge)
        cache = {}

        def make_step(L, D):
            hit = cache.get((L, D))
            if hit is not None:
                return hit

            @jax.jit
            def step(LAB, BTR, BSP, dst_ids, in_tr, front, HOP):
                init = (jnp.full((L, P, W), _BIG),
                        jnp.full((L, P), -1, jnp.int32),
                        jnp.full((L, P), -1, jnp.int32), front)

                def slot(carry, i):
                    cur, ctr, csp, fr = carry
                    tr = in_tr[:, i]
                    cand = expand(LAB, tr, HOP)
                    ntr = jnp.broadcast_to(tr[:, None], (L, P)).astype(jnp.int32)
                    nsp = jnp.broadcast_to(ar_P[None, :], (L, P))
                    a, b, c, nk = merge_v(cur, ctr, csp, cand, ntr, nsp)
                    return (a, b, c, jnp.maximum(fr, nk.max())), None

                (lab, btr, bsp, front), _ = lax.scan(slot, init, jnp.arange(D))
                return (LAB.at[dst_ids].set(lab), BTR.at[dst_ids].set(btr),
                        BSP.at[dst_ids].set(bsp), front)
            cache[(L, D)] = step
            return step

        self._kern = make_step
        return make_step

    # ------------------------------------------------------------------
    def hop_table(self, EDGE, DEPOT):
        """`(n_transitions+1, Mmax)` hop costs for this cycle, gathered from
        `EDGE`/`DEPOT` through the flat indices baked at build time.  Padding
        slots get `-_BIG` so they lose the routing-leg `max`."""
        import jax.numpy as jnp
        self._kernels()
        jt = self._jt
        E, D = jnp.asarray(EDGE), jnp.asarray(DEPOT)
        want_e = (self.n_agents, self.N_global, self.N_global,
                  self.max_k, self.max_k)
        want_d = (self.n_agents, self.N_global, self.max_k)
        if tuple(E.shape) != want_e or tuple(D.shape) != want_d:
            raise ValueError(
                "coupled_dp: cost tensors do not match the structure this DP "
                f"was built for -- got EDGE{tuple(E.shape)} DEPOT"
                f"{tuple(D.shape)}, expected EDGE{want_e} DEPOT{want_d}")
        h = jnp.where(jt["T_isdep"], D.reshape(-1)[jt["T_didx"]],
                      E.reshape(-1)[jt["T_eidx"]])
        return jnp.where(jt["T_hvalid"], h, -_BIG)

    def route(self, EDGE, DEPOT):
        """Price this structure against one cycle's cost tensors.  Jittable:
        no host sync and no Python branching on traced values, so it is safe
        to call from inside another `jax.jit` trace.

        Returns `(makespan, order, bc, own, front)`:
          `order` (n,)        GLOBAL node ids in scheduled sequence
          `bc`, `own` (n,Mmax) branch chosen at each stop and the agent it
                              belongs to (`own == -1` in a padding slot)
          `front` ()          largest Pareto front seen; `front <= max_labels`
                              certifies the answer is exact

        An infeasible instance comes back with `makespan >= _BIG/2` rather
        than raising -- a traced caller cannot branch on it, so the check
        belongs to whoever consumes the value.  `solve()` does raise."""
        import jax.numpy as jnp
        from jax import lax
        make_step = self._kernels()
        jt = self._jt
        n, P, W = self.n, self.max_labels, self.n + 1
        nS = self.n_states
        HOP = self.hop_table(EDGE, DEPOT)

        LAB = jnp.full((nS + 2, P, W), _BIG).at[self.init_state, 0].set(
            jnp.zeros(W))
        BTR = jnp.full((nS + 2, P), -1, jnp.int32)
        BSP = jnp.full((nS + 2, P), -1, jnp.int32)
        front = jnp.asarray(1, jnp.int32)

        for c in range(1, self.n_levels):
            di, ti = self.level_dst[c], self.level_in[c]
            if di.size == 0:
                continue
            LAB, BTR, BSP, front = make_step(*ti.shape)(
                LAB, BTR, BSP, jnp.asarray(di), jnp.asarray(ti), front, HOP)

        term = jt["term"]
        mk = LAB[term][:, :, n].reshape(-1)
        flat = jnp.argmin(mk)
        best = mk[flat]

        def back(carry, _):
            st, sl = carry
            e = jnp.maximum(BTR[st, sl], 0)     # clamp: a broken chain only
            return (jt["T_src"][e], BSP[st, sl]), e   # happens when infeasible

        _, es = lax.scan(back, (term[flat // P], (flat % P).astype(jnp.int32)),
                         None, length=n)
        es = es[::-1]
        return (best, jt["gnodes"][jt["T_node"][es]], jt["T_bc"][es],
                jt["T_own"][es], front)

    def _route_jit(self):
        import jax
        if getattr(self, "_rj", None) is None:
            self._rj = jax.jit(self.route)
        return self._rj

    def solve(self, EDGE, DEPOT):
        """Eager `route()` plus host-side reconstruction, an independent
        re-score of the reconstructed schedule, and loud failure."""
        import jax.numpy as jnp
        _ensure_x64()
        n, P, nS = self.n, self.max_labels, self.n_states
        if n == 0:
            return CoupledResult(0.0, [], {}, {}, True, 1, nS,
                                 self.n_transitions, self.n_ideals)

        best, order_j, bc_j, own_j, front = self._route_jit()(
            jnp.asarray(EDGE), jnp.asarray(DEPOT))
        best = float(best)
        if not np.isfinite(best) or best >= _HALF_BIG:
            raise ValueError(
                "coupled_dp: no feasible schedule -- every terminal label is "
                "invalid (an unreachable branch, or a precedence/routing "
                "combination with no completion)")

        order = [int(g) for g in np.asarray(order_j)]
        bc, own = np.asarray(bc_j), np.asarray(own_j)
        branch_of = {}
        for r, g in enumerate(order):
            for m in range(self.Mmax):
                j = int(own[r, m])
                if j >= 0:
                    branch_of[(g, j)] = int(bc[r, m])

        arr = self._forward(order, branch_of, EDGE, DEPOT)
        chk = max(arr.values()) if arr else 0.0
        if abs(chk - best) > 1e-6 * max(1.0, abs(best)):
            raise AssertionError(
                f"coupled_dp: reconstruction disagrees with the DP value "
                f"({chk} vs {best}) -- this is a bug, not a fence")

        mf = int(front)
        return CoupledResult(best, order, branch_of, arr, mf <= P, mf,
                             nS, self.n_transitions, self.n_ideals)

    def _forward(self, order, branch_of, EDGE, DEPOT):
        """Independent DAG forward pass over the reconstructed schedule --
        the arbiter that `solve()` checks its own DP value against."""
        EDGE, DEPOT = np.asarray(EDGE, float), np.asarray(DEPOT, float)
        g2l = {g: i for i, g in enumerate(self.gnodes)}
        arr, prev = {}, {}
        for g in order:
            i = g2l[g]
            t = float(self.release[i])
            for u in self.predlist[i]:
                t = max(t, arr[self.gnodes[u]])
            for j in self.owners[i]:
                bv = branch_of[(g, j)]
                if j in prev:
                    pu, pb = prev[j]
                    t = max(t, arr[pu] + float(EDGE[j, pu, g, pb, bv]))
                else:
                    t = max(t, float(DEPOT[j, g, bv]))
            arr[g] = t
            for j in self.owners[i]:
                prev[j] = (g, branch_of[(g, j)])
        return arr


def build_coupled_dp(n_nodes, n_agents, owners, preds, KC, **kw):
    """Build the state graph ONCE for one fixed assignment / resolved
    precedence DAG / node subset.  The result is reusable: `.route(EDGE,
    DEPOT)` (jittable) or `.solve(EDGE, DEPOT)` (eager, verified) price it
    against any cycle's costs without rebuilding or recompiling.

    `owners[v]` -> agent ids visiting node `v`; `preds[v]` -> its precedence
    predecessors; `KC` (n_agents, n_nodes) int -> branch count, >= 1.
    `max_k` is the branch stride of the cost tensors (default `KC.max()`).
    Optional: `nodes`, `release`, `max_labels`, `dominance_tol`,
    `collapse_finished`, and the `max_*` fences -- see the module docstring.
    """
    return _CoupledDP(n_nodes, n_agents, owners, preds, KC, **kw)


def solve_coupled_makespan(n_nodes, n_agents, owners, preds, EDGE, DEPOT, KC,
                           **kw):
    """One-shot build + solve.  `EDGE` (n_agents, n_nodes, n_nodes, K, K) and
    `DEPOT` (n_agents, n_nodes, K) hold `_BIG` where invalid.  Use
    `build_coupled_dp` instead when the same structure is priced repeatedly
    (every MPC cycle), which is what it is designed for."""
    kw.setdefault("max_k", int(np.asarray(EDGE).shape[-1]))
    return _CoupledDP(n_nodes, n_agents, owners, preds, KC,
                      **kw).solve(EDGE, DEPOT)

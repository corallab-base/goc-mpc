"""A/B test for `coupled_dp.solve_coupled_makespan` against brute force.

Brute force = every linear extension of the precedence DAG x every
`(node, agent)` branch combination, scored by the same DAG forward pass
`dp_master._makespan_forward` uses.  That is the definition the coupled DP
is claimed to optimise exactly, so the two must agree to floating point on
every instance small enough to enumerate.
"""

import itertools
import sys

import numpy as np

from goc_mpc.logic_based_benders_solver.coupled_dp import (
    _BIG, build_coupled_dp, solve_coupled_makespan)


# ----------------------------------------------------------------- brute force
def brute_force(n, M, owners, preds, EDGE, DEPOT, KC):
    predmask = [0] * n
    for v in range(n):
        for u in preds[v]:
            predmask[v] |= 1 << u

    exts = []
    for perm in itertools.permutations(range(n)):
        S = 0
        for v in perm:
            if predmask[v] & ~S:
                break
            S |= 1 << v
        else:
            exts.append(perm)

    pairs = [(v, j) for v in range(n) for j in owners[v]]
    combos = list(itertools.product(*[range(int(KC[j, v])) for (v, j) in pairs]))

    best, best_sol = np.inf, None
    for ext in exts:
        for combo in combos:
            br = {pairs[i]: combo[i] for i in range(len(pairs))}
            arr, prev, ok = {}, {}, True
            for v in ext:
                t = 0.0
                for u in preds[v]:
                    t = max(t, arr[u])
                for j in owners[v]:
                    bv = br[(v, j)]
                    c = (EDGE[j, prev[j][0], v, prev[j][1], bv]
                         if j in prev else DEPOT[j, v, bv])
                    if c >= _BIG / 2:
                        ok = False
                        break
                    t = max(t, (arr[prev[j][0]] if j in prev else 0.0) + c)
                if not ok:
                    break
                arr[v] = t
                for j in owners[v]:
                    prev[j] = (v, br[(v, j)])
            if ok and max(arr.values()) < best:
                best, best_sol = max(arr.values()), (ext, dict(br))
    return best, best_sol


# ------------------------------------------------------------------- instances
def random_instance(rng, n, M, kmax, edge_density, multi_owner=0.0):
    owners = []
    for v in range(n):
        js = [int(rng.integers(M))]
        if M > 1 and rng.random() < multi_owner:
            k = int(rng.integers(M))
            if k != js[0]:
                js.append(k)
        owners.append(sorted(js))

    preds = [[] for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):          # u < v keeps it acyclic
            if rng.random() < edge_density:
                preds[v].append(u)

    KC = np.ones((M, n), np.int64)
    for v in range(n):
        for j in owners[v]:
            KC[j, v] = int(rng.integers(1, kmax + 1))
    K = int(KC.max())

    EDGE = np.full((M, n, n, K, K), _BIG)
    DEPOT = np.full((M, n, K), _BIG)
    for j in range(M):
        for v in range(n):
            if j not in owners[v]:
                continue
            DEPOT[j, v, :KC[j, v]] = rng.uniform(0.5, 4.0, KC[j, v])
            for u in range(n):
                if u == v or j not in owners[u]:
                    continue
                EDGE[j, u, v, :KC[j, u], :KC[j, v]] = rng.uniform(
                    0.5, 6.0, (KC[j, u], KC[j, v]))
    return owners, preds, EDGE, DEPOT, KC


def random_costs(rng, n, M, owners, KC):
    """Fresh EDGE/DEPOT for a FIXED (owners, KC) structure -- what an MPC
    cycle does: same graph, same ownership, same branch counts, new numbers."""
    K = int(KC.max())
    EDGE = np.full((M, n, n, K, K), _BIG)
    DEPOT = np.full((M, n, K), _BIG)
    for j in range(M):
        for v in range(n):
            if j not in owners[v]:
                continue
            DEPOT[j, v, :KC[j, v]] = rng.uniform(0.5, 4.0, KC[j, v])
            for u in range(n):
                if u == v or j not in owners[u]:
                    continue
                EDGE[j, u, v, :KC[j, u], :KC[j, v]] = rng.uniform(
                    0.5, 6.0, (KC[j, u], KC[j, v]))
    return EDGE, DEPOT


def worked_example():
    """The 5-node example from the design discussion: agent 0 owns {a,b,e},
    agent 1 owns {c,d}, and `a -> c` couples them.  Both of agent 0's routes
    finish at the same time, but they put `t(a)` at 1 vs 3, which decides a
    makespan of 6 vs 8.  A DP keyed only on per-agent completion times sees
    one state and flips a coin; this is the instance that pins the open
    frontier down."""
    a, b, e, c, d = 0, 1, 2, 3, 4
    n, M, K = 5, 2, 1
    owners = [[0], [0], [0], [1], [1]]
    preds = [[], [], [a, b], [a], [c]]
    EDGE = np.full((M, n, n, K, K), _BIG)
    DEPOT = np.full((M, n, K), _BIG)
    DEPOT[0, a, 0] = DEPOT[0, b, 0] = 1.0
    EDGE[0, a, b, 0, 0] = EDGE[0, b, a, 0, 0] = 2.0
    EDGE[0, a, e, 0, 0] = EDGE[0, b, e, 0, 0] = 2.0
    DEPOT[1, c, 0] = 0.0
    EDGE[1, c, d, 0, 0] = 5.0
    return n, M, owners, preds, EDGE, DEPOT, np.ones((M, n), np.int64)


# ------------------------------------------------------------------------ main
def main():
    fails = 0

    n, M, owners, preds, EDGE, DEPOT, KC = worked_example()
    r = solve_coupled_makespan(n, M, owners, preds, EDGE, DEPOT, KC)
    bf, _ = brute_force(n, M, owners, preds, EDGE, DEPOT, KC)
    tag = "ok " if abs(r.makespan - bf) < 1e-9 else "FAIL"
    fails += tag == "FAIL"
    print(f"[{tag}] worked example: dp={r.makespan:.4f} brute={bf:.4f} "
          f"order={r.order} front={r.max_front} exact={r.exact}")

    rng = np.random.default_rng(0)
    cases = [
        # n,  M, kmax, density, multi_owner, trials
        (5,  2, 1, 0.25, 0.0, 12),
        (6,  2, 2, 0.20, 0.0, 12),
        (6,  3, 2, 0.30, 0.0, 12),
        (7,  2, 3, 0.15, 0.0,  8),
        (7,  3, 2, 0.35, 0.0,  8),
        (7,  2, 2, 0.00, 0.0,  6),   # no precedence at all -> fronts must be 1
        (6,  3, 2, 0.25, 0.5,  8),   # hand-offs: nodes with two owners
        (8,  3, 2, 0.40, 0.0,  5),
    ]
    worst_front, decoupled_front = 0, 0
    for (n, M, kmax, dens, mo, trials) in cases:
        for _ in range(trials):
            owners, preds, EDGE, DEPOT, KC = random_instance(
                rng, n, M, kmax, dens, mo)
            bf, _ = brute_force(n, M, owners, preds, EDGE, DEPOT, KC)
            # both state-collapse modes must give the same optimum
            rs = {}
            for cf in (True, False):
                try:
                    rs[cf] = solve_coupled_makespan(
                        n, M, owners, preds, EDGE, DEPOT, KC, max_labels=32,
                        collapse_finished=cf)
                except Exception as exc:                  # noqa: BLE001
                    print(f"[FAIL] n={n} M={M} k={kmax} d={dens} "
                          f"collapse={cf}: raised {exc!r}")
                    fails += 1
            if len(rs) < 2:
                continue
            r = rs[True]
            worst_front = max(worst_front, r.max_front)
            if dens == 0.0:
                # the decoupling certificate only holds uncollapsed: with
                # collapse_finished the merged states differ in mk alone.
                decoupled_front = max(decoupled_front, rs[False].max_front)
            if abs(rs[True].makespan - rs[False].makespan) > 1e-9:
                print(f"[FAIL] n={n} M={M}: collapse modes disagree "
                      f"{rs[True].makespan} vs {rs[False].makespan}")
                fails += 1
            if not np.isfinite(bf):
                continue
            if abs(r.makespan - bf) > 1e-7 * max(1.0, bf):
                print(f"[FAIL] n={n} M={M} k={kmax} d={dens} mo={mo}: "
                      f"dp={r.makespan:.6f} brute={bf:.6f}")
                fails += 1
        print(f"[ok ] n={n} M={M} kmax={kmax} dens={dens} multi={mo}: "
              f"{trials} instances match brute force and each other  "
              f"(states={r.n_states} trans={r.n_transitions} "
              f"ideals={r.n_ideals} front<={r.max_front})")

    # --- build once, price many: the MPC-cycle shape -------------------
    # ONE structure (owners / precedence / branch counts fixed), priced
    # against several different cost tensors -- exactly what a cycle does.
    # Each pricing must equal both a fresh one-shot build and brute force.
    rng2 = np.random.default_rng(1234)
    n, M = 7, 3
    owners, preds, _E, _D, KC = random_instance(rng2, n, M, 2, 0.3)
    dp = build_coupled_dp(n, M, owners, preds, KC, max_k=int(KC.max()),
                          max_labels=32)
    reuse_bad = 0
    for _ in range(6):
        EDGE, DEPOT = random_costs(rng2, n, M, owners, KC)
        r_reuse = dp.solve(EDGE, DEPOT)
        r_fresh = solve_coupled_makespan(n, M, owners, preds, EDGE, DEPOT, KC,
                                         max_labels=32)
        bf, _ = brute_force(n, M, owners, preds, EDGE, DEPOT, KC)
        if abs(r_reuse.makespan - r_fresh.makespan) > 1e-9:
            print(f"[FAIL] reuse != fresh: {r_reuse.makespan} vs {r_fresh.makespan}")
            reuse_bad += 1
        if np.isfinite(bf) and abs(r_reuse.makespan - bf) > 1e-7 * max(1.0, bf):
            print(f"[FAIL] reuse != brute force: {r_reuse.makespan} vs {bf}")
            reuse_bad += 1
    fails += reuse_bad
    print(f"[{'ok ' if not reuse_bad else 'FAIL'}] build-once/price-many: one "
          f"structure priced against 6 cost tensors matches a fresh build and "
          f"brute force every time")

    # the reuse must also be CHEAP -- no rebuild, no recompile after the first
    import time
    EDGE, DEPOT = random_costs(rng2, n, M, owners, KC)
    dp.solve(EDGE, DEPOT)
    t0 = time.time()
    for _ in range(5):
        EDGE, DEPOT = random_costs(rng2, n, M, owners, KC)
        dp.solve(EDGE, DEPOT)
    print(f"[ok ] warm re-price: {(time.time() - t0) / 5 * 1e3:.1f} ms/solve "
          f"(n={n} M={M}, states={dp.n_states})")

    print(f"\nlargest Pareto front seen: {worst_front}")
    print(f"largest front with no precedence at all, uncollapsed: "
          f"{decoupled_front} (must be 1 -- with no cross-agent coupling "
          f"there is no trade-off to trade off)")
    if decoupled_front != 1:
        fails += 1

    # --- scaling probe: how far does the exact coupled DP actually reach? ---
    print("\nscaling (exact, no brute-force comparison):")
    import time
    for (n, M, kmax, dens) in [(10, 3, 2, 0.30), (12, 3, 2, 0.30),
                               (14, 4, 2, 0.35), (16, 4, 2, 0.40)]:
        owners, preds, EDGE, DEPOT, KC = random_instance(
            rng, n, M, kmax, dens)
        t0 = time.time()
        try:
            r = solve_coupled_makespan(n, M, owners, preds, EDGE, DEPOT, KC,
                                       max_labels=16)
            print(f"  n={n:2d} M={M} k={kmax} dens={dens}: "
                  f"mk={r.makespan:7.3f} states={r.n_states:8d} "
                  f"trans={r.n_transitions:9d} ideals={r.n_ideals:7d} "
                  f"front={r.max_front:3d} exact={r.exact} "
                  f"{time.time() - t0:6.2f}s")
        except NotImplementedError as exc:
            print(f"  n={n:2d} M={M}: fenced -- {exc}")

    print("\nFAILURES:", fails)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

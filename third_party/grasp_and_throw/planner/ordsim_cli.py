"""ordsim_cli: streams, training data and evaluation for the picking-order network (existing recipe: 5 flow kinds, 20 objects per stream, roll-K expert, DAgger).
 data: python ordsim_cli.py data --streams 600 --seed 0 --y-vis 1.4 [--policy m0.npz] --out r0.npz      (F, mask, y, group, adv — the format of learning/train_order_policy.py)
 eval: python ordsim_cli.py eval --streams 40 --seed 41 --y-vis 1.4 --policies fifo roll nn --policy m1.npz"""
import os, sys, argparse, time
import numpy as np
import multiprocessing as mp
import ordsim as O
import ordfull as F
CONFIGS = [("uniform", (0.0, 0.05), None, None, None), ("uniform", (0.03, 0.10), None, None, None), ("uniform", (0.05, 0.14), None, None, None),   # learning/order_policy_data.CONFIGS
           ("burst", (0.0, 0.0), (3, 6), 0.10, (0.5, 1.0)), ("burst", (0.0, 0.0), (5, 8), 0.15, (0.4, 0.8))]
NAMES = ["dense", "tight", "normal", "burst", "big burst"]; X_RANGE = (0.25, 0.595); Y_ENTER = 0.90; MIN_SEP = 0.08; TMODEL = os.environ.get("ORD_TMODEL") or os.path.join(O.HERE, "logs", "tmodel.npz")
ZR = tuple(float(v) for v in os.environ["ORD_Z"].split(",")) if os.environ.get("ORD_Z") else None      # 11-18: object tops uniform in ORD_Z = "lo,hi"; not set = every object at the default top


def stream(rng, n, cfg, code):
    """learning/order_stream.random_stream: the first object at the line y = Y_ENTER at time 0, the others upstream."""
    mode, gap, burst, blen, bgap = cfg; ys = []; y = Y_ENTER
    if mode == "burst":
        while len(ys) < n:
            c = int(rng.integers(burst[0], burst[1] + 1)); ys += sorted((y + rng.uniform(0.0, blen, size=c)).tolist()); y = ys[-1] + float(rng.uniform(*bgap))
        ys = ys[:n]
    else:
        for k in range(n):
            if k:
                y += float(rng.uniform(*gap))
            ys.append(y)
    xy = []
    for yy in ys:
        for _ in range(50):
            x = float(rng.uniform(*X_RANGE))
            if all((x - a) ** 2 + (yy - b) ** 2 >= MIN_SEP ** 2 for a, b in xy):
                break
        xy.append((x, yy))
    b = rng.integers(0, 3, size=n); z = rng.uniform(ZR[0], ZR[1], size=n) if ZR else None                # tops drawn last: positions and bins of a seed are the same with and without heights
    return O.Stream([p[0] for p in xy], [p[1] for p in xy], b, code=code, z=z)


def _data_job(a):
    sid, seed, n, y_vis, policy, expert = a; tm = O.TM(TMODEL); rng = np.random.default_rng(seed); S = stream(rng, n, CONFIGS[sid % 5], tm.bin_code); rows = []; last = [0]; hits = [0, 0]

    def rec(S, a_, ts, vis, ok, passed):
        Fe, mask = O.features(S, a_, ts, vis, ok, passed); st = []; v = F.full_values(S, tm, a_, ts, vis, ok, stat=st) if expert == "full" else O.roll_values(S, tm, a_, ts, vis, ok)        # expert values: every order (full) or candidate-then-FIFO (roll)
        last[0] = int(min(range(len(v)), key=lambda i: (-v[i][0], v[i][1]))); hits[0] += 1; hits[1] += int(bool(st and st[0][1])); rows.append((Fe, mask, last[0], O.advantage(v)))

    r = O.run(S, tm, O.make_nn(O.MLP(policy)) if policy else (lambda *_a: last[0]), y_vis, on_decision=rec); return sid, rows, r["n_done"], hits


def _eval_job(a):
    sid, seed, n, y_vis, pols, policy = a; tm = O.TM(TMODEL); S = stream(np.random.default_rng(seed), n, CONFIGS[sid % 5], tm.bin_code); out = {}
    for p in pols:
        ch = O.choose_fifo if p == "fifo" else O.choose_roll if p == "roll" else F.choose_full if p == "full" else O.make_nn(O.MLP(policy)); t0 = time.time(); r = O.run(S, tm, ch, y_vis); out[p] = (r["n_done"], r["n_rest"], r["t"], time.time() - t0)
    return sid, out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["data", "eval"]); ap.add_argument("--streams", type=int, default=600); ap.add_argument("--n", type=int, default=20); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--y-vis", type=float, default=1.4); ap.add_argument("--policy", default=""); ap.add_argument("--policies", nargs="+", default=["fifo", "roll"]); ap.add_argument("--out", default=""); ap.add_argument("--procs", type=int, default=14); ap.add_argument("--expert", default="full", choices=["full", "roll"])
    a = ap.parse_args(); t0 = time.time()
    with mp.get_context("fork").Pool(a.procs) as pool:
        if a.cmd == "data":
            Fs, Mk, Y, G, A = [], [], [], [], []; nh = [0, 0]
            for sid, rows, _n, hits in pool.imap_unordered(_data_job, [(s, a.seed + s, a.n, a.y_vis, a.policy, a.expert) for s in range(a.streams)]):
                nh[0] += hits[0]; nh[1] += hits[1]
                for f, m, y, adv in rows:
                    Fs.append(f); Mk.append(m); Y.append(y); G.append(sid); A.append(adv)
            np.savez_compressed(a.out, F=np.array(Fs, np.float32), mask=np.array(Mk, bool), y=np.array(Y, np.int64), group=np.array(G, np.int64), adv=np.array(A, np.float32)); y = np.array(Y); nc = np.array(Mk).sum(1)
            print(f"decisions {len(Y)} -> {a.out} [{time.time() - t0:.0f} s]. expert rank {dict((k, int((y == k).sum())) for k in range(O.K))}; candidates per decision mean {nc.mean():.2f}, with >= 2: {(nc >= 2).mean() * 100:.0f} %; expert {a.expert}, search cut by the state cap in {nh[1]} of {nh[0]} decisions")
        else:
            tot = {p: np.zeros(5) for p in a.policies}; rest = {p: np.zeros(5) for p in a.policies}; dec = {p: [] for p in a.policies}
            for sid, out in pool.imap_unordered(_eval_job, [(s, a.seed * 100000 + s, a.n, a.y_vis, a.policies, a.policy) for s in range(a.streams * 5)]):
                for p, (n, nr, _t, w) in out.items():
                    tot[p][sid % 5] += n; rest[p][sid % 5] += nr; dec[p].append(w)
            print(f"eval: {a.streams} streams x 5 kinds x {a.n} objects, visible line {a.y_vis}, seed {a.seed} [{time.time() - t0:.0f} s]")
            for p in a.policies:
                print(f"  {p:5s} " + "  ".join(f"{NAMES[k]} {tot[p][k]:.0f}" for k in range(5)) + f"  | total {tot[p].sum():.0f} ({(tot[p].sum() / tot[a.policies[0]].sum() - 1) * 100:+.1f} %)  to the wait point {rest[p].sum():.0f}  ({np.mean(dec[p]) * 1e3:.0f} ms per stream)")


if __name__ == "__main__":
    main()

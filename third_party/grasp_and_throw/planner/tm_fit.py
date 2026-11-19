"""tm_fit: cycle-time model (the 'throw time model' of the existing method, for the realized cycles): MLP (xa, ya, xb, yb at the cycle start, A's bin one-hot) -> cycle time.
Data = solved cells of the 5 cm DB (logs/real_D5, whole reach ring) + random off-grid solves (logs/tm_rand, logs/tm_rand2; half of them held out for validation).  -> logs/tmodel.npz
usage: python tm_fit.py [epochs]"""
import os, sys, glob, json
os.environ["LPP_VIN"] = "1"
import numpy as np
import db_vin as V


def load(folder):
    X = []; y = []
    for f in sorted(glob.glob(os.path.join(folder, "c_*.npz"))):
        if len(os.path.basename(f)[2:-4].split("_")) != 5:
            continue
        I = json.loads(str(np.load(f, allow_pickle=True)["info"]))
        if I["status"] in ("Solve_Succeeded", "Solved_To_Acceptable_Level") and I["viol"] <= 1e-5:
            k = I["key"]; oh = [0.0, 0.0, 0.0]; oh[int(k[4])] = 1.0; X.append(k[:4] + oh); y.append(I["t_cycle"])
    return np.array(X), np.array(y)


Xd, yd = load(os.environ.get("TM_DB", "logs/real_D5")); RR = [load(f) for f in os.environ.get("TM_RAND", "logs/tm_rand,logs/tm_rand2").split(",") if os.path.isdir(f)]; Xr = np.vstack([r[0] for r in RR]); yr = np.concatenate([r[1] for r in RR]); rng = np.random.default_rng(0); p = rng.permutation(len(Xr)); h = len(Xr) // 2 if len(Xr) >= 2000 else len(Xr); va = p[:h]; tr = p[h:]      # 11-18 (user: off-grid tasks for validation only, a few): fewer than 2,000 -> none of them in the training set
X = np.vstack([Xd, Xr[tr]]); y = np.r_[yd, yr[tr]]; xm = X.mean(0); xs = X.std(0); ym = y.mean(); Xn = (X - xm) / xs; dims = [7, 64, 64, 1]; ep = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
Ws = [rng.uniform(-1, 1, (dims[i], dims[i + 1])) / np.sqrt(dims[i]) for i in range(3)]; bs = [np.zeros(dims[i + 1]) for i in range(3)]; bs[-1][0] = ym; P = Ws + bs; m1 = [np.zeros_like(q) for q in P]; m2 = [np.zeros_like(q) for q in P]; step = 0


def fwd(Z):
    hs = [Z]
    for k in range(3):
        z = hs[-1] @ Ws[k] + bs[k]; hs.append(np.tanh(z) if k < 2 else z)
    return hs


pred = lambda Xq: fwd((Xq - xm) / xs)[-1][:, 0]
for e in range(ep):
    lr = 3e-3 * 0.5 * (1 + np.cos(np.pi * e / ep)); perm = rng.permutation(len(Xn))
    for k0 in range(0, len(perm), 256):
        idx = perm[k0:k0 + 256]; hs = fwd(Xn[idx]); d = ((hs[-1][:, 0] - y[idx]) / len(idx))[:, None]; gW = [None] * 3; gb = [None] * 3
        for k in (2, 1, 0):
            gW[k] = hs[k].T @ d; gb[k] = d.sum(0)
            if k:
                d = (d @ Ws[k].T) * (1 - hs[k] ** 2)
        step += 1
        for j, (q, g) in enumerate(zip(P, gW + gb)):
            m1[j] = 0.9 * m1[j] + 0.1 * g; m2[j] = 0.999 * m2[j] + 0.001 * g * g; q -= lr * (m1[j] / (1 - 0.9 ** step)) / (np.sqrt(m2[j] / (1 - 0.999 ** step)) + 1e-8)
    if e % 500 == 0 or e == ep - 1:
        print(f"epoch {e}: train MAE {np.abs(pred(X) - y).mean() * 1e3:.2f} ms, held-out off-grid MAE {np.abs(pred(Xr[va]) - yr[va]).mean() * 1e3:.2f} ms", flush=True)
ev = np.abs(pred(Xr[va]) - yr[va]) * 1e3; code = np.array([float(np.asarray(V.M.BINS[b], float)[1]) / 0.33 for b in range(3)])
np.savez(os.environ.get("TM_OUT", "logs/tmodel.npz"), n_layers=3, W0=Ws[0], W1=Ws[1], W2=Ws[2], b0=bs[0], b1=bs[1], b2=bs[2], x_mean=xm, x_std=xs, bin_code=code)
print(f"cells {len(Xd)} + off-grid {len(tr)} for training, {len(va)} off-grid held out: MAE {ev.mean():.2f} ms, 90 % {np.percentile(ev, 90):.2f}, max {ev.max():.2f} | cycle time mean {y.mean():.3f} s (min {y.min():.3f}, max {y.max():.3f}) | bin codes {code.tolist()}")

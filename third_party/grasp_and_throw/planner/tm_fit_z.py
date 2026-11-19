"""tm_fit_z (2026-11-18, user: object height as an input of the cycle-time model): tm_fit.py with the tops of A and B as two more inputs.
MLP (xa, ya, xb, yb at the cycle start, bin of A one-hot, zA, zB) -> cycle time, 9-64-64-1, same training as tm_fit.py.
Data: the 5 cm library TM_DB (logs/real_A6, both tops at the default top) + the off-grid tasks TM_RAND (logs/tm_rand_a6, default top; split as in tm_fit.py)
      + the off-grid tasks with random tops TM_RANDZ (logs/tm_randz_a6; TM_HOLD of them held out, default 2,000).   -> TM_OUT (logs/tmodel_a6_z.npz), read by ordsim.TM.
Also prints the error of the model without heights (TM_OLD, logs/tmodel_a6.npz) on the held-out tasks with heights.
usage: python tm_fit_z.py [epochs]"""
import os, sys, glob, json
os.environ["LPP_VIN"] = "1"
import numpy as np
import db_vin as V
ZTOP = float(V.D.ZTOP)


def load(folder, n):
    X = []; y = []
    for f in sorted(glob.glob(os.path.join(folder, "c_*.npz"))):
        if len(os.path.basename(f)[2:-4].split("_")) != n:
            continue
        I = json.loads(str(np.load(f, allow_pickle=True)["info"]))
        if I["status"] in ("Solve_Succeeded", "Solved_To_Acceptable_Level") and I["viol"] <= 1e-5:
            k = I["key"]; oh = [0.0, 0.0, 0.0]; oh[int(k[4])] = 1.0; X.append(list(k[:4]) + oh + ([float(k[5]), float(k[6])] if n == 7 else [ZTOP, ZTOP])); y.append(I["t_cycle"])
    return np.array(X), np.array(y)


Xd, yd = load(os.environ.get("TM_DB", "logs/real_A6"), 5); Xr, yr = load(os.environ.get("TM_RAND", "logs/tm_rand_a6"), 5); Xz, yz = load(os.environ.get("TM_RANDZ", "logs/tm_randz_a6"), 7)
rng = np.random.default_rng(0); p = rng.permutation(len(Xr)); h = len(Xr) // 2 if len(Xr) >= 2000 else len(Xr); va = p[:h]; tr = p[h:]; pz = np.random.default_rng(1).permutation(len(Xz)); nh = int(os.environ.get("TM_HOLD", "2000")); vz = pz[:nh]; tz = pz[nh:]
X = np.vstack([Xd, Xr[tr], Xz[tz]]); y = np.r_[yd, yr[tr], yz[tz]]; xm = X.mean(0); xs = X.std(0); ym = y.mean(); Xn = (X - xm) / xs; dims = [9, 64, 64, 1]; ep = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
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
        print(f"epoch {e}: train MAE {np.abs(pred(X) - y).mean() * 1e3:.2f} ms, held-out default top {np.abs(pred(Xr[va]) - yr[va]).mean() * 1e3:.2f} ms, held-out with heights {np.abs(pred(Xz[vz]) - yz[vz]).mean() * 1e3:.2f} ms", flush=True)
code = np.array([float(np.asarray(V.M.BINS[b], float)[1]) / 0.33 for b in range(3)])
np.savez(os.environ.get("TM_OUT", "logs/tmodel_a6_z.npz"), n_layers=3, W0=Ws[0], W1=Ws[1], W2=Ws[2], b0=bs[0], b1=bs[1], b2=bs[2], x_mean=xm, x_std=xs, bin_code=code)
Z = np.load(os.environ.get("TM_OLD", "logs/tmodel_a6.npz")); ho = (Xz[vz][:, :7] - Z["x_mean"]) / Z["x_std"]
for k in range(3):
    ho = ho @ Z[f"W{k}"] + Z[f"b{k}"]; ho = np.tanh(ho) if k < 2 else ho
e0 = np.abs(pred(Xr[va]) - yr[va]) * 1e3; ez = np.abs(pred(Xz[vz]) - yz[vz]) * 1e3; eo = np.abs(ho[:, 0] - yz[vz]) * 1e3
print(f"cells {len(Xd)} + off-grid {len(tr)} + off-grid with heights {len(tz)} for training (solved with heights {len(Xz)}) | held out default top {len(va)}: MAE {e0.mean():.2f} ms, 90 % {np.percentile(e0, 90):.2f}, max {e0.max():.2f} | "
      f"held out with heights {len(vz)}: MAE {ez.mean():.2f} ms, 90 % {np.percentile(ez, 90):.2f}, max {ez.max():.2f} | model without heights on them: MAE {eo.mean():.2f} ms, 90 % {np.percentile(eo, 90):.2f}, max {eo.max():.2f} | "
      f"cycle time with heights mean {yz.mean():.3f} s (min {yz.min():.3f}, max {yz.max():.3f}) | default top {ZTOP}")

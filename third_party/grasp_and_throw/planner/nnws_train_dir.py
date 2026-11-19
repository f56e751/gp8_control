"""nnws_train2 (2026-11-13 night, user: "신경망 2개로 DB를 대체하되 신경망을 좀더 overfitting 하면 반복 수 일정 초과 건들이 줄어드는지 확인"):
two networks — one for the 4 durations, one for Q, V, A, J (1746 numbers) — fitted to the library more or less tightly.
Same data and test tasks as nnws_train.py (68,811 cells + 1,500 off-grid for training; the other 1,500 off-grid tasks for the test).
usage: ~/isaacsim/python.sh nnws_train2.py NAME EPOCHS WIDTH DEPTH     (durations network: 512 x 4, same number of epochs; no weight decay)
-> logs/nnws/pred_NAME.npz and one line 'NN2 NAME ...' with the fit on the training cells and the error on the test tasks."""
import os, sys, json, time
ND = os.environ.get("NNWS_DIR", "logs/nnws")            # 11-16: copy of nnws_train2.py with the data/output folder from NNWS_DIR (one folder per belt speed)
import numpy as np
import torch
name = sys.argv[1]; ep = int(sys.argv[2]); W = int(sys.argv[3]); DEP = int(sys.argv[4]); dev = "cuda"; torch.manual_seed(0)
G = np.load(ND + "/grid.npz"); R = np.load(ND + "/rand.npz"); VU = G["VU"]


ZIN = os.environ.get("NNWS_Z") == "1"                                             # 11-18 (user): the tops of A and B as two more inputs, random-height tasks in the training set


def feats(key):
    X = np.zeros((len(key), 9 if ZIN else 7)); X[:, :4] = key[:, :4]; X[np.arange(len(key)), 4 + key[:, 4].astype(int)] = 1.0
    if ZIN:
        X[:, 7:9] = key[:, 5:7] if key.shape[1] >= 7 else 0.02
    return X


def targ(D):
    n = len(D["key"]); return np.concatenate([D["T"], D["Q"].reshape(n, -1), D["V"].reshape(n, -1), D["A"].reshape(n, -1), D["J"].reshape(n, -1)], 1).astype(np.float64)


GR = {"T": slice(0, 4), "Q": slice(4, 442), "V": slice(442, 880), "A": slice(880, 1318), "J": slice(1318, 1750)}
ntr = int(os.environ.get("NNWS_RAND_TRAIN", "1500" if len(R["key"]) >= 3000 else "0"))      # 11-18 (user: off-grid tasks for validation only, a few): none of them in the training set when there are fewer than 3,000
perm = np.random.default_rng(0).permutation(len(R["key"])); itr, ite = perm[:ntr], perm[ntr:]
Xg, Yg = feats(G["key"]), targ(G); Xr, Yr = feats(R["key"]), targ(R)
Xtr = np.vstack([Xg, Xr[itr]]); Ytr = np.vstack([Yg, Yr[itr]]); Xte, Yte = Xr[ite], Yr[ite]; kte = R["key"][ite]; nte0 = len(Xte)
if ZIN:
    RZ = np.load(ND + "/randz.npz"); Xz, Yz = feats(RZ["key"]), targ(RZ); pz = np.random.default_rng(2).permutation(len(Xz)); nh = int(os.environ.get("NNWS_Z_HOLD", "1500")); hz, tz = pz[:nh], pz[nh:]
    Xtr = np.vstack([Xtr, Xz[tz]]); Ytr = np.vstack([Ytr, Yz[tz]]); Xte = np.vstack([Xte, Xz[hz]]); Yte = np.vstack([Yte, Yz[hz]]); kz = RZ["key"][hz]; kte = np.vstack([np.c_[kte[:, :5], np.full((len(kte), 2), 0.02)] if kte.shape[1] < 7 else kte, kz])
xm, xs = Xtr.mean(0), Xtr.std(0) + 1e-9; ym, ys = Ytr.mean(0), np.maximum(Ytr.std(0), 1e-6)
tX = torch.tensor((Xtr - xm) / xs, dtype=torch.float32, device=dev); tY = torch.tensor((Ytr - ym) / ys, dtype=torch.float32, device=dev); tXe = torch.tensor((Xte - xm) / xs, dtype=torch.float32, device=dev)
sub = np.random.default_rng(1).permutation(len(Xg))[:5000]                    # training cells on which the fit is reported


def train(sl, w, dep):
    L = []; n = tX.shape[1]
    for _ in range(dep):
        L += [torch.nn.Linear(n, w), torch.nn.SiLU()]; n = w
    net = torch.nn.Sequential(*L, torch.nn.Linear(n, sl.stop - sl.start)).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.0); nb = 1024; steps = ep * (len(tX) // nb + 1)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3 if w > 1024 else 2e-3, total_steps=steps, pct_start=0.03, final_div_factor=1e3); Y = tY[:, sl]; t0 = time.time()
    for e in range(ep):
        idx = torch.randperm(len(tX), device=dev)
        for i in range(0, len(tX), nb):
            j = idx[i:i + nb]; loss = torch.nn.functional.mse_loss(net(tX[j]), Y[j]); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
    net.eval()
    with torch.no_grad():
        P = net(tXe).double().cpu().numpy() * ys[sl] + ym[sl]; Ptr = net(tX[sub]).double().cpu().numpy() * ys[sl] + ym[sl]; ltr = float(torch.nn.functional.mse_loss(net(tX[:20000]), Y[:20000]))
    return P, Ptr, time.time() - t0, ltr, sum(p.numel() for p in net.parameters()), [p.detach().cpu().numpy().astype(np.float32) for p in net.parameters()]


def metrics(P, Y):
    n = len(P); E = P - Y; out = {}; ec = np.abs(E[:, GR["T"]].sum(1)) * 1e3; out["cycle_ms"] = [round(float(ec.mean()), 3), round(float(ec.max()), 2)]
    for g, nn_, unit in (("Q", 73, None), ("V", 73, VU), ("A", 73, VU), ("J", 72, VU)):
        e = np.abs(E[:, GR[g]]).reshape(n, nn_, 6); e = np.degrees(e) if unit is None else e / unit; mx = e.max((1, 2)); out[g] = [round(float(np.median(mx)), 4), round(float(np.percentile(mx, 90)), 4), round(float(mx.max()), 3)]
    return out


P = np.zeros_like(Yte); Ptr = np.zeros((len(sub), 1750)); PT, PTtr, t1, l1, n1, wT = train(GR["T"], 512, 4); P[:, GR["T"]] = PT; Ptr[:, GR["T"]] = PTtr
PX, PXtr, t2, l2, n2, wX = train(slice(4, 1750), W, DEP); P[:, 4:] = PX; Ptr[:, 4:] = PXtr; n = len(P)
np.savez(ND + "/pred_%s.npz" % name, key=kte, T=P[:, GR["T"]], Q=P[:, GR["Q"]].reshape(n, 73, 6), V=P[:, GR["V"]].reshape(n, 73, 6), A=P[:, GR["A"]].reshape(n, 73, 6), J=P[:, GR["J"]].reshape(n, 72, 6))
np.savez(ND + "/w2_%s.npz" % name, xm=xm, xs=xs, ymT=ym[:4], ysT=ys[:4], ymX=ym[4:], ysX=ys[4:], **{"t%d" % i: w for i, w in enumerate(wT)}, **{"x%d" % i: w for i, w in enumerate(wX)})
print("NN2", name, json.dumps(dict(width=W, depth=DEP, epochs=ep, params_M=round((n1 + n2) / 1e6, 2), train_s=round(t1 + t2), loss_T=round(l1, 6), loss_QVAJ=round(l2, 6),
                                    on_training_cells=metrics(Ptr, Yg[sub]), on_test_tasks=metrics(P[:nte0], Yte[:nte0]) if nte0 else None, n_train=len(Xtr), n_test_offgrid=nte0,
                                    on_test_heights=(metrics(P[nte0:], Yte[nte0:]) if len(P) > nte0 else None), inputs=int(tX.shape[1]))), flush=True)

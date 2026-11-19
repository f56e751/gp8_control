"""rn2: the cycle optimization of rnlp.py (same variables, constraints, objective, limits) — built once per process and reused.
The object positions and the bin enter as parameters, so a call only looks up the start (nearest solved cell of RN_DB) and solves.
Stop rule: RN_MAXT seconds of wall clock from the call; the iterate at that moment is returned as it is (user rule 11-11).
usage (through srv.py): rn2.py "xa,ya,xb,yb,bin[,za,zb]" out.npz      (positions at the moment A is touched; za, zb = tops of A and of the next object B, default 0.02)"""
import os, sys, json, time, glob, builtins
os.environ["LPP_VIN"] = "1"
import numpy as np
import casadi as ca
import db_vin as V
t_call = time.time(); D = V.D; M = V.M; H = V.H; VB = np.asarray(V.VB, float); G = float(M._GRAV); TOP = D.ZTOP; XMIN = 0.25; HW = 0.05; W1 = 100.0; WACC = 1e6; WOM = 1.0
CACHE = builtins.__dict__.setdefault("_rn2_cache", {})


class Stop(ca.Callback):                                                       # asks the solver to stop when the wall clock passes .deadline
    def __init__(self, nx, ng, npar):
        ca.Callback.__init__(self); self.nx, self.ng, self.npar = nx, ng, npar; self.deadline = 1e18; self.construct("stop", {})

    def get_n_in(self): return ca.nlpsol_n_out()
    def get_n_out(self): return 1
    def get_name_in(self, i): return ca.nlpsol_out(i)
    def get_name_out(self, i): return "ret"

    def get_sparsity_in(self, i):
        n = ca.nlpsol_out(i)
        return ca.Sparsity.scalar() if n == "f" else ca.Sparsity.dense(self.nx) if n in ("x", "lam_x") else ca.Sparsity.dense(self.ng) if n in ("g", "lam_g") else ca.Sparsity.dense(self.npar)

    def eval(self, arg):
        return [1 if time.time() > self.deadline else 0]


def build(lim, NK, Wa, a0):
    VU = np.asarray(M.V.QD, float) / float(os.environ.get("GP8_LIM_SCALE", "0.9")); VEL, ACC, JRK = lim[0] * VU, lim[1] * VU, lim[2] * VU; QLO = np.asarray(M.V.Q_LO, float) + 0.02; QHI = np.asarray(M.V.Q_HI, float) - 0.02
    P = ca.MX.sym("p", 22); qA, vA, A0, B0, tg, c6, zA, zB = P[0:6], P[6:12], P[12:14], P[14:16], P[16:19], P[19], P[20], P[21]      # zA, zB: object tops (11-13: perceived height per object)
    Tv = ca.MX.sym("T", 4); Tk = [Tv[0], Tv[1], HW, HW, Tv[2], Tv[3]]; NT = sum(NK); n1 = NT + 1
    Q = ca.MX.sym("Q", 6, n1); Vv = ca.MX.sym("V", 6, n1); Aa = ca.MX.sym("A", 6, n1); Jj = ca.MX.sym("J", 6, NT); Sa = ca.MX.sym("S", 6, n1); g = []; lbg = []; ubg = []; hs = []; tnode = [0]
    for k, n in enumerate(NK):
        for _ in range(n):
            hs.append(Tk[k] / n); tnode.append(tnode[-1] + Tk[k] / n)
    for m in range(NT):
        h = hs[m]; g += [Q[:, m + 1] - (Q[:, m] + Vv[:, m] * h + Aa[:, m] * h ** 2 / 2 + Jj[:, m] * h ** 3 / 6), Vv[:, m + 1] - (Vv[:, m] + Aa[:, m] * h + Jj[:, m] * h ** 2 / 2), Aa[:, m + 1] - (Aa[:, m] + Jj[:, m] * h)]
        lbg += [np.zeros(18)]; ubg += [np.zeros(18)]
    iL = NK[0]; iW0 = NK[0] + NK[1]; iR = iW0 + NK[2]; iW1 = iR + NK[3]; iH = iW1 + NK[4]; iE = NT; tL = tnode[iL]; tH = tnode[iH]; tE = tnode[iE]

    def eq(e, n):
        g.append(e); lbg.append(np.zeros(n)); ubg.append(np.zeros(n))

    def hover(idx, o, t):
        q = Q[:, idx]; tw = ca.mtimes(M.F_JT(q), Vv[:, idx]); eq(M.F_TCP(q) - ca.vertcat(o[0] + VB[0] * t, o[1] + VB[1] * t, H), 3); eq(M.F_AX(q)[0:2], 2); eq(tw[0:2] - ca.DM(VB[:2]), 2); eq(tw[3:6], 3)

    eq(Q[:, 0] - qA, 6); eq(Vv[:, 0] - vA, 6)
    if a0:
        eq(Aa[:, 0], 6)
    hover(iL, A0, tL); hover(iH, B0, tH)
    qr = Q[:, iR]; p = M.F_P(qr); vv = M.F_V(qr, Vv[:, iR]); d = tg[0:2] - p[0:2]; Rr = ca.norm_2(d); hh = tg[2] - p[2]; th = np.pi / 4 + 0.5 * ca.atan2(hh, Rr); spd = ca.sqrt(G * (hh + ca.sqrt(Rr ** 2 + hh ** 2)))
    eq(vv - ca.vertcat(spd * ca.cos(th) * d / Rr, spd * ca.sin(th)), 3)
    qe = Q[:, iE]; twe = ca.mtimes(M.F_JT(qe), Vv[:, iE]); eq(M.F_TCP(qe) - ca.vertcat(B0[0] + VB[0] * tE, B0[1] + VB[1] * tE, zB), 3); eq(M.F_AX(qe)[0:2], 2); eq(twe[0:3] - ca.DM(VB), 3); eq(twe[3:6], 3); eq(qe[5] - c6, 1)
    for m in range(1, NT + 1):
        pz = M.F_TCP(Q[:, m]); lo = zA if m <= iL else zB if m > iH else H - 0.001              # floor: top of A while lifting, travel height, top of B while descending
        if m not in (iL, iH, iE):
            g += [pz[2] - lo, pz[0]]; lbg += [np.array([0.0]), np.array([XMIN])]; ubg += [np.array([np.inf]), np.array([np.inf])]
    sc = ca.repmat(ca.DM(ACC), 1, n1); g += [ca.vec(Sa) - ca.vec(Aa / sc), ca.vec(Sa) + ca.vec(Aa / sc)]; lbg += [np.zeros(6 * n1)] * 2; ubg += [np.full(6 * n1, np.inf)] * 2
    Jacc = 0
    for m in range(NT):
        Jacc = Jacc + hs[m] * (ca.sum1(Sa[:, m]) + ca.sum1(Sa[:, m + 1])) / 2
    wn = list(range(iW0, iW1 + 1)); wts = np.full(len(wn), 2 * HW / (len(wn) - 1)); wts[[0, -1]] *= 0.5; Jw = 0
    for m, wm in zip(wn, wts):
        e = M.F_LVEC(Q[:, m], Vv[:, m], tg); om = M.F_W(Q[:, m], Vv[:, m]); Jw = Jw + wm * (WACC * ca.sumsqr(e) + WOM * ca.sumsqr(om))
    w = ca.vertcat(Tv, ca.vec(Q), ca.vec(Vv), ca.vec(Aa), ca.vec(Jj), ca.vec(Sa)); rep = lambda a, n: np.tile(np.asarray(a, float), n)
    lbw = np.r_[[0.02, 0.0, 0.02, 0.02], rep(QLO, n1), rep(-VEL, n1), rep(-ACC, n1), rep(-JRK, NT), np.zeros(6 * n1)]; ubw = np.r_[[1.0, 1.0, 1.5, 1.0], rep(QHI, n1), rep(VEL, n1), rep(ACC, n1), rep(JRK, NT), np.full(6 * n1, 1.0 + 1e-9)]
    G_ = ca.vertcat(*g); LB = np.concatenate([np.atleast_1d(a).ravel() for a in lbg]); UB = np.concatenate([np.atleast_1d(a).ravel() for a in ubg]); cb = Stop(w.shape[0], G_.shape[0], 22)
    opts = {"ipopt.max_iter": 3000, "ipopt.tol": 1e-6, "ipopt.acceptable_tol": 1e-1, "ipopt.acceptable_constr_viol_tol": 1e-6, "ipopt.acceptable_dual_inf_tol": 1e3, "ipopt.acceptable_compl_inf_tol": 1e-2, "ipopt.acceptable_obj_change_tol": 2e-5,
            "ipopt.acceptable_iter": 12, "ipopt.print_level": 0, "print_time": 0, "ipopt.mu_strategy": "adaptive", "iteration_callback": cb}
    return dict(sol=ca.nlpsol("rn", "ipopt", dict(x=w, p=P, f=W1 * (tE + Wa * Jacc) + Jw, g=G_), opts), cb=cb, LB=LB, UB=UB, lbw=lbw, ubw=ubw, ACC=ACC, NT=NT, iR=iR, iH=iH)


def cells(folder, b):                                                          # solved cells of one bin: positions and files (listed once)
    k = ("db", folder, b)
    if k not in CACHE or os.environ.get("RN_DB_REFRESH"):
        fs = [f for f in glob.glob(os.path.join(folder, "c_*_%d.npz" % b)) if len(os.path.basename(f)[2:-4].split("_")) == 5]
        CACHE[k] = (np.array([[float(x) for x in os.path.basename(f)[2:-4].split("_")[:4]] for f in fs]), fs)
    return CACHE[k]


def nn_start(path, b, key, ACC):                                              # start from the two networks (durations; Q V A J) instead of the library (user 11-14)
    if ("nn", path) not in CACHE:
        Z = np.load(path); nt = len([k for k in Z.files if k[0] == "t" and k[1:].isdigit()]) // 2; nx = len([k for k in Z.files if k[0] == "x" and k[1:].isdigit()]) // 2
        CACHE[("nn", path)] = (Z["xm"].astype(np.float32), Z["xs"].astype(np.float32), [(Z["t%d" % (2 * i)], Z["t%d" % (2 * i + 1)]) for i in range(nt)], Z["ymT"], Z["ysT"],
                               [(Z["x%d" % (2 * i)], Z["x%d" % (2 * i + 1)]) for i in range(nx)], Z["ymX"], Z["ysX"])
    xm, xs, LT, ymT, ysT, LX, ymX, ysX = CACHE[("nn", path)]; x = np.zeros(len(xm), np.float32); x[:4] = key[:4]; x[4 + b] = 1.0
    if len(xm) >= 9:                                                           # 11-18 (user): networks trained with the tops of A and B
        x[7], x[8] = (key[5], key[6]) if len(key) >= 7 else (TOP, TOP)
    x = (x - xm) / xs

    def fwd(L):
        h = x
        for i, (W_, b_) in enumerate(L):
            h = W_ @ h + b_
            if i < len(L) - 1:
                h = h / (1.0 + np.exp(-h))                                     # SiLU
        return h.astype(float)
    T = np.clip(fwd(LT) * ysT + ymT, [0.02, 0.0, 0.02, 0.02], [1.0, 1.0, 1.5, 1.0]); y = fwd(LX) * ysX + ymX; A = y[876:1314].reshape(-1, 6)      # y = Q (438), V (438), A (438), J (432)
    return np.r_[T, y, (np.abs(A) / ACC + 1e-3).ravel()], "nn:" + os.path.basename(path), 0.0


def start(folder, b, key, ACC):
    K, fs = cells(folder, b); d = np.sum((K - np.array(key[:4])) ** 2, axis=1)
    d[d < 1e-10] = 1e9                                                         # never start from the cell itself (re-solves)
    for i in np.argsort(d)[:20]:
        if ("w0", fs[i]) not in CACHE or (CACHE[("w0", fs[i])] is None and os.environ.get("RN_DB_REFRESH")):
            Z = np.load(fs[i], allow_pickle=True); I = json.loads(str(Z["info"])); T = I["T"]; ok = I["status"] in ("Solve_Succeeded", "Solved_To_Acceptable_Level") and I["viol"] <= 1e-5
            CACHE[("w0", fs[i])] = np.r_[[T[0], T[1], T[4], T[5]], Z["Q"].ravel(), Z["V"].ravel(), Z["A"].ravel(), Z["J"].ravel(), (np.abs(Z["A"]) / ACC + 1e-3).ravel()] if ok else None
        if CACHE[("w0", fs[i])] is not None:
            return CACHE[("w0", fs[i])], fs[i], float(d[i]) ** 0.5
    raise RuntimeError("no solved cell near " + str(key))


lim = tuple(float(x) for x in os.environ.get("RN_LIM", "0.8,8,160").split(",")); NK = [int(x) for x in os.environ.get("RN_NK", "10,10,6,6,30,10").split(",")]; Wa = float(os.environ.get("RN_WA", "0.05")); a0 = bool(os.environ.get("RN_A0"))
sig = ("nlp", lim, tuple(NK), Wa, a0); t0 = time.time()
if sig not in CACHE:
    CACHE[sig] = build(lim, NK, Wa, a0)
S = CACHE[sig]; t1 = time.time(); key = [float(v) for v in sys.argv[1].split(",")]; out = sys.argv[2]; b = int(key[4]); tg = np.array(M.BINS[b], float); A0 = np.array(key[0:2]); B0 = np.array(key[2:4]); zA, zB = (key[5], key[6]) if len(key) >= 7 else (TOP, TOP)
qA = np.asarray(M.ik(np.r_[A0, zA]), float); vA = np.asarray(M.qd_of(qA, VB), float); w0, f0, dist = nn_start(os.environ["RN_NN"], b, key, S["ACC"]) if os.environ.get("RN_NN") else start(os.environ["RN_DB"], b, key, S["ACC"])
S["cb"].deadline = t_call + float(os.environ.get("RN_MAXT", "600")); t2 = time.time()
R = S["sol"](x0=w0, p=np.r_[qA, vA, A0, B0, tg, qA[5], zA, zB], lbx=S["lbw"], ubx=S["ubw"], lbg=S["LB"], ubg=S["UB"]); st = S["sol"].stats(); t3 = time.time()
x = np.array(R["x"]).ravel(); gv = np.array(R["g"]).ravel(); viol = float(max(np.max(S["LB"] - gv), np.max(gv - S["UB"]), 0.0)); NT = S["NT"]; n1 = NT + 1; Ts = x[:4]; o = 4
Qs = x[o:o + 6 * n1].reshape(n1, 6); o += 6 * n1; Vs = x[o:o + 6 * n1].reshape(n1, 6); o += 6 * n1; As = x[o:o + 6 * n1].reshape(n1, 6); o += 6 * n1; Js = x[o:o + 6 * NT].reshape(NT, 6)
Tk = [Ts[0], Ts[1], HW, HW, Ts[2], Ts[3]]; tt = np.r_[0, np.cumsum(np.concatenate([np.full(n, Tk[k] / n) for k, n in enumerate(NK)]))]
info = dict(key=key, zA=float(zA), zB=float(zB), belt=float(-VB[1]), status=st["return_status"], iters=int(st["iter_count"]), build_s=t1 - t0, prep_s=t2 - t1 + t0 - t_call, solve_s=t3 - t2, viol=viol, T=[float(v) for v in Tk], t_release=float(tt[S["iR"]]), t_hover=float(tt[S["iH"]]),
            t_cycle=float(tt[-1]), lim=list(lim), Wa=Wa, NK=NK, start_cell=os.path.basename(f0), start_dist=dist)
np.savez(out, tt=tt, Q=Qs, V=Vs, A=As, J=Js, A0=A0, B0=B0, tg=tg, info=json.dumps(info)); print("RNLP", json.dumps(info), flush=True)

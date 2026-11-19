"""11-05 시험: 실제 궤적 (관절 5 차 B-스플라인, 실제 시각 r ∈ [0, t_c]) 을 직접 최적화 — 실제 속도·가속도 ≤ 한계 90 % (명령은 보지 않음).
그다음 필터를 정확히 역계산 (상자 둘 재귀: u[k] = u[k−n] + n (y[k] − y[k−1])) 해 명령을 만들고 100 % 한계 위반을 잰다.
조건·목적·과제는 cyc2 케이스 A 와 같음 (이상 이력). 시작: 실제 q(0) = 이력 실제, 1~4 계 도함수도 이력 실제와 같게 (명령이 이어지도록).
사용: python real1.py case_index out.json"""
import os, sys, json, time
import numpy as np
from scipy.interpolate import BSpline
import cyc2 as C
from cyc2 import CY, T, SG, DT, H, NP_PATH, NP_HI, BAND_XY, BAND_V, BAND_W
from gp8_control.grasping.workspace_gate import tcp_batch

K = 5; N = C.N
KN = np.concatenate([np.zeros(K), np.linspace(0.0, 1.0, N - K + 1), np.ones(K)])
DB = [BSpline(KN, np.eye(N), K)]
DB += [DB[0].derivative(nu) for nu in range(1, 5)]
FD, (FW1, FW2) = T.FILTER
N1, N2 = int(round(FW1 / DT)), int(round(FW2 / DT))
QD_FULL = T.QD_LIM / T.LIM_SCALE; QDD_FULL = T.QDD_LIM / T.LIM_SCALE
CMDLIM = os.environ.get("REAL_CMDLIM", "0") == "1"              # 1: 역계산 명령 (위치·속도·가속도) ≤ 계획 한계 (× LIM_SCALE) 를 제약으로
TC_MAX = 1.69; NMAX = int(round(TC_MAX / DT)) + 1                    # 명령 샘플 행 수 고정 (t_c 이후 행은 mask 로 끔)


def mder(nu):
    """nu 계 도함수의 제어점 = mder(nu) @ P^T (s 기준) — 볼록 껍질로 구간 전체 한계."""
    M = np.asarray(DB[nu].c)[:N - nu]
    s = np.linspace(0, 1, 37); chk = BSpline(KN[nu:len(KN) - nu], M, K - nu)(s)
    assert np.abs(chk - DB[nu](s)).max() < 1e-6 * max(1.0, np.abs(DB[nu](s)).max())
    return M


def build():
    import casadi as cs
    c = CY._consts(); NW = T.N_WIN
    HW_LO = np.maximum(T.GP8_Q_MIN, T.Q_LO_REAL) + CY.Q_MARGIN; HW_HI = np.minimum(T.GP8_Q_MAX, T.Q_HI_REAL) - CY.Q_MARGIN
    o = cs.Opti(); Pf = o.variable(6, N - 1)
    par = {k: o.parameter(*((n,) if isinstance(n, int) else n)) for k, n in dict(
        q0=6, v0=6, t_c=1, tgt=3, j6=1, zi=1, zh=1, vbar=1, S0=(6, 4),
        Mu=(NMAX, N - 1), Mv=(NMAX, N - 1), Ma=(NMAX, N - 1), hu=(6, NMAX), hv=(6, NMAX), ha=(6, NMAX), mask=(6, NMAX),
        Aw=(N, NW), Adw=(N, NW), hw=(6, NW), hdw=(6, NW),
        Ap=(N, NP_PATH), Apd=(N, NP_PATH), hp=(6, NP_PATH), hpd=(6, NP_PATH), Op=(2, NP_PATH),
        Ah=(N, NP_HI), hh=(6, NP_HI),
        Ak=N, Akd=N, hk_=6, hkd=6, ok=3,
        Ae=N, Aed=N, he=6, hed=6, oe=2).items()}
    t_c = par["t_c"]; P = cs.horzcat(par["q0"], Pf)
    for k in range(1, N):
        o.subject_to(o.bounded(HW_LO, P[:, k], HW_HI))
    M1, M2 = mder(1), mder(2)
    V = cs.mtimes(P, cs.DM(M1.T)) / t_c; A = cs.mtimes(P, cs.DM(M2.T)) / t_c ** 2      # 실제 속도·가속도 제어점 (planner 규약)
    if not CMDLIM:
        o.subject_to(o.bounded(np.repeat(-T.QD_LIM[:, None], N - 1, 1), V, np.repeat(T.QD_LIM[:, None], N - 1, 1)))
        o.subject_to(o.bounded(np.repeat(-T.QDD_LIM[:, None], N - 2, 1), A, np.repeat(T.QDD_LIM[:, None], N - 2, 1)))
    else:                                                                     # 명령 (planner 규약, 코드 τ = k DT, k = 1..n) = h + Pf·M — 선형
        mk = par["mask"]; rp_ = lambda v: np.repeat(v[:, None], NMAX, 1)
        HL = np.maximum(T.GP8_Q_MIN, T.Q_LO_REAL) + CY.Q_MARGIN; HH = np.minimum(T.GP8_Q_MAX, T.Q_HI_REAL) - CY.Q_MARGIN
        o.subject_to(o.bounded(rp_(HL), mk * (par["hu"] + cs.mtimes(Pf, par["Mu"].T)), rp_(HH)))
        o.subject_to(o.bounded(rp_(-T.QD_LIM), mk * (par["hv"] + cs.mtimes(Pf, par["Mv"].T)), rp_(T.QD_LIM)))
        o.subject_to(o.bounded(rp_(-T.QDD_LIM), mk * (par["ha"] + cs.mtimes(Pf, par["Ma"].T)), rp_(T.QDD_LIM)))
    for nu in range(1, 5):                                                    # 시작 도함수 = 이력 실제
        o.subject_to(cs.mtimes(P, cs.DM(DB[nu](0.0))) / t_c ** nu - par["S0"][:, nu - 1] == 0)
    F_p, F_a, F_pd, _ = T.end_kinematics_functions()
    vbc = cs.DM(C.VB)
    C.GROUPS.clear(); C.GROUPS.append(("lim", 0, o.ng)); mark = [o.ng]

    def grp(name):
        C.GROUPS.append((name, mark[0], o.ng)); mark[0] = o.ng
    for m in range(NP_PATH):
        q = cs.mtimes(P, par["Ap"][:, m]); qd = cs.mtimes(P, par["Apd"][:, m])
        p = F_p(q); v = F_pd(q, qd); w = T.fk_omega_sym(q, qd)
        o.subject_to(p[2] - par["zi"] >= 0)
        for e, b in ((p[0:2] - par["Op"][:, m], BAND_XY), (v[0:2] - vbc[0:2], BAND_V), (w, BAND_W)):
            o.subject_to(o.bounded(-b, e, b))
    grp("path")
    q = cs.mtimes(P, par["Ak"]); qd = cs.mtimes(P, par["Akd"])
    o.subject_to(F_p(q) - par["ok"] == 0); o.subject_to(F_pd(q, qd) - vbc == 0); grp("k")
    for m in range(NP_HI):
        o.subject_to(F_p(cs.mtimes(P, par["Ah"][:, m]))[2] - par["zh"] >= 0)
    grp("zh")
    q = cs.mtimes(P, par["Ae"]); qd = cs.mtimes(P, par["Aed"])
    p = F_p(q); v = F_pd(q, qd); a = F_a(q)
    o.subject_to(p[0:2] - par["oe"] == 0); o.subject_to(p[2] - par["zh"] == 0); grp("endp")
    o.subject_to(a[0] == 0); o.subject_to(a[1] == 0); o.subject_to(a[2] <= -0.5); o.subject_to(q[5] - par["j6"] == 0); grp("endrot")
    o.subject_to(T.fk_omega_sym(q, qd) == 0); grp("endw")
    o.subject_to(v[0:2] - vbc[0:2] == 0); o.subject_to(o.bounded(-par["vbar"], v[2], -1e-3)); grp("endv")
    mark[0] = o.ng
    F_err, F_z, _ = T.make_objective_functions_sym(); dragc = cs.DM(np.zeros(3))
    w_tr = np.full(NW, c["RT"] / (NW - 1)); w_tr[[0, -1]] *= 0.5
    J_win = 0; taus = []
    for m in range(NW):
        q = cs.mtimes(P, par["Aw"][:, m]); qd = cs.mtimes(P, par["Adw"][:, m])
        tau = o.variable(); taus.append(tau)
        o.subject_to(o.bounded(0.05, tau, 1.5)); o.subject_to(F_z(q, qd, tau, par["tgt"], dragc) == 0)
        J_win = J_win + w_tr[m] * (T.W_ACC * F_err(q, qd, tau, par["tgt"], dragc) + T.W_OMEGA_LADDER[0] * cs.sumsqr(T.fk_omega_sym(q, qd)))
    J = T.W1 * t_c + T.W2 * J_win
    o.minimize(J)
    ip = {"max_iter": 3000, "print_level": 0, "sb": "yes", "tol": 1e-6, "acceptable_tol": 1e-4, "acceptable_iter": 10,
          "mu_strategy": "adaptive", "warm_start_init_point": "yes", "warm_start_bound_push": 1e-9,
          "warm_start_mult_bound_push": 1e-9, "warm_start_slack_bound_push": 1e-9}
    o.solver("ipopt", {"print_time": False, "expand": True}, ip)
    Lf = cs.Function("L", [o.x, o.p, o.lam_g], [o.f + cs.dot(o.lam_g, o.g)])
    return dict(opti=o, Pf=Pf, taus=taus, par=par, J=J, J_win=J_win, L=Lf)


def W(t_c, r):
    s = np.clip(np.asarray(r, float) / t_c, 0.0, 1.0); m = len(s)
    return DB[0](s), DB[1](s) / t_c, np.zeros((6, m)), np.zeros((6, m))


class RRun(C.Run):
    def __init__(self, task, hist):
        super().__init__(task, hist)
        s = C.S(); s["opti"].set_value(s["par"]["S0"], task["S0"])

    def set_theta(self, th):
        t_c, t_st, t_s, t_k = (float(v) for v in th); c = CY._consts(); s = C.S(); o, p = s["opti"], s["par"]
        if CMDLIM:
            for k, v in cmd_maps(t_c, self.t).items():
                o.set_value(p[k], v)
        rp = np.linspace(0.0, t_s, NP_PATH + 1)[1:]; rh = np.linspace(t_s, t_c, NP_HI)
        Aw, Adw, hw, hdw = W(t_c, t_st + c["OFFS"]); Ap, Apd, hp, hpd = W(t_c, rp); Ah, _, hh, _ = W(t_c, rh)
        Ak, Akd, hk_, hkd = W(t_c, [t_k]); Ae, Aed, he, hed = W(t_c, [t_c])
        oi = self.t["oi"]; on = self.t["on"]
        vals = dict(t_c=t_c, Aw=Aw.T, Adw=Adw.T, hw=hw, hdw=hdw, Ap=Ap.T, Apd=Apd.T, hp=hp, hpd=hpd,
                    Op=np.stack([oi(r)[:2] for r in rp], 1), Ah=Ah.T, hh=hh, Ak=Ak[0], Akd=Akd[0], hk_=hk_[:, 0], hkd=hkd[:, 0],
                    ok=np.r_[oi(t_k)[:2], self.t["zi"]], Ae=Ae[0], Aed=Aed[0], he=he[:, 0], hed=hed[:, 0], oe=on(t_c)[:2])
        for k, v in vals.items():
            o.set_value(p[k], v)


def cmd_maps(t_c, task):
    """명령 = 실제의 정확한 역계산 (선형): u_pl(k) = hu + Pf·Mu (k = 1..NMAX), 속도·가속도는 차분 (k = 1 은 이력 끝 명령과)."""
    u_h, y_h, a_h = task["HR"]; q0 = np.asarray(task["q0"], float)
    r = np.arange(1, NMAX + 1) * DT; B = DB[0](np.minimum(r / t_c, 1.0))[:, 1:]
    Z = np.zeros((400, N - 1)); L = np.asarray(inverse(Z, Z, Z, B))[400:]                 # 0 이력에서 실제 증분 → 명령
    h = inverse(u_h, y_h, a_h, np.repeat(a_h[-1:], NMAX, 0))[len(u_h):] * SG              # 이력 + 실제 정지 유지 (planner)
    hu = (h - (L @ np.ones(N - 1))[:, None] * q0[None]).T                                 # B(P − q0) = B[:,1:](Pf − q0)
    Mf = np.vstack([np.zeros((2, N - 1)), L]); hf = np.hstack([(u_h[-2:] * SG).T, hu])
    n = int(round(t_c / DT)); mask = np.zeros((6, NMAX)); mask[:, :n] = 1.0
    return dict(Mu=L, Mv=np.diff(Mf, axis=0)[1:] / DT, Ma=np.diff(Mf, 2, axis=0) / DT ** 2, hu=hu,
                hv=np.diff(hf, axis=1)[:, 1:] / DT, ha=np.diff(hf, 2, axis=1) / DT ** 2, mask=mask)


def boxf(x, n):
    xp = np.vstack([np.repeat(x[:1], n, 0), x]); cs_ = np.cumsum(xp, 0)
    return (cs_[n:] - cs_[:-n]) / n


def hist_realized(Uh):
    """이력 명령 (로봇 규약, 마지막 = 코드 τ 0) → u, y = 상자 W2 (u), a = 상자 W1 (y). a[-1] = 실제 r = 0 (지연 D 뒤)."""
    u = np.vstack([np.repeat(Uh[:1], 400, 0), Uh]); y = boxf(u, N2); a = boxf(y, N1)
    return u, y, a


def inverse(u, y, a, a_new):
    """실제 a_new[k] (k = 1..n, 실제 r = k DT) 를 정확히 내는 명령 u[k] (코드 τ = k DT)."""
    u = list(u); y = list(y); a = list(a)
    for an in a_new:
        a.append(an); y.append(y[-N1] + N1 * (a[-1] - a[-2])); u.append(u[-N2] + N2 * (y[-1] - y[-2]))
    return np.array(u)


def hist_exact(o_p, vin, T0=-0.60, Ta=-0.55, Tb=-0.30):
    """이력: 실제 툴이 정지 (r ≤ Ta) → 매끄럽게 (7 차 smootherstep, C³) vin 까지 가속하며 o 위 직선 하강, r = 0 에 h.
    명령 = 그 실제 궤적의 정확한 역계산 (정지 상태에서 시작) — 이력 실제가 매끄러워 r = 0 의 도함수가 정확하다."""
    P_ = C.prim(); rr = np.arange(int(round(T0 / DT)), 1) * DT
    S = lambda u: (lambda u: 35 * u ** 4 - 84 * u ** 5 + 70 * u ** 6 - 20 * u ** 7)(np.clip(u, 0.0, 1.0))
    fine = np.linspace(T0, 0.0, 30001); sv = S((fine - Ta) / (Tb - Ta))  # xy (벨트) 와 z (vin) 모두 정지 → 같은 램프
    F = np.concatenate([[0.0], np.cumsum((sv[1:] + sv[:-1]) / 2 * np.diff(fine))])
    g = F[-1] - np.interp(rr, fine, F)                                        # ∫_r^0 램프 (r ≥ Tb 에서 −r)
    o0 = o_p(0.0)
    pts = np.array([np.r_[o0[:2] - C.VB[:2] * gg, H + vin * gg] for gg in g])
    q_mid = P_._pick_ik(pts[-1])
    Q = C.TS._ik_path(P_.robot, pts[::-1], q_mid, P_.j6)[::-1]; Q = C.yaw_fix(Q, q_mid[0])
    pad = np.repeat(Q[:1], 400, 0)
    u = inverse(pad, pad, pad, Q[1:])
    Uh = u[-(len(Q) + 200):]
    _, _, a_c = hist_realized(Uh); C._PR["hist_err"] = float(np.abs(a_c[-len(Q):] - Q).max())
    return Uh


def real_init(task, hist_U, vin0):
    P_ = C.prim(); oi, on = task["oi"], task["on"]; zi = task["zi"]; out = []
    variants = [v for v in C.INIT_VARIANTS if v is not None][:3]
    for kv, (v, t_k, dl) in enumerate([(v, tk, dl) for tk, dl in ((0.15, 0.18), (0.22, 0.22), (C.MEM - C.D + 0.02, C.MEM - C.D + 0.02)) for v in variants]):
        t_s = t_k + dl; tf = float(np.clip(v["T0"], 0.7, 1.0)); t_c = t_s + tf
        try:
            rr = np.arange(0.0, t_s + 1e-9, DT)
            pts = np.array([np.r_[oi(r)[:2], C.zprof(r, t_k, t_s, zi, vin0)] for r in rr])
            Qp = TS_ik(P_, pts, hist_U[-1]); Qp = C.yaw_fix(Qp, Qp[0, 0])
            q_h = P_._pick_ik(np.r_[on(t_c)[:2], H], q_init=Qp[-1])
            if q_h is None:
                continue
            Pt = T.init_ctrl_points(Qp[-1] * SG, np.asarray(v["dq_swing"], float), float(v["chi0"]), p_f=q_h * SG)
            n1 = int(round(tf / DT)); Qt = BSpline(T.KNOTS, Pt.T, T.DEGREE)(np.arange(1, n1 + 1) / n1) * SG
            Qr = np.vstack([Qp, Qt]); n_c = int(round(t_c / DT)); Qr = np.vstack([Qr, np.repeat(Qr[-1:], 5, 0)])[:n_c + 1] * SG
            B = DB[0](np.minimum(np.arange(n_c + 1) * DT / t_c, 1.0)); q0 = np.asarray(task["q0"], float)
            X = np.linalg.lstsq(B[1:, 1:], Qr[1:] - B[1:, :1] * q0[None], rcond=None)[0]
            out.append((f"real{kv}", dict(P=np.vstack([q0[None], X]).T, theta=(t_c, t_s + float(v["chi0"]) * tf, t_s, t_k))))
        except ValueError:
            continue
    return out


def cmd_init(task, Uh, vin0):
    """cyc2 의 명령 공간 초기값 (3 차 명령 스플라인) → 필터한 실제 → 5 차 실제 스플라인 맞춤."""
    out = []
    for tg, w in C.path_init(task, Uh, vin0):
        t_c = float(w["theta"][0]); U = CY.sample(w["P"], t_c) * SG; n = int(round(t_c / DT))
        _, _, a = hist_realized(np.vstack([Uh[:-1], U])); ar = a[-len(U):][1:n + 1] * SG      # 실제 r = k DT, k = 1..n
        Bm = DB[0](np.minimum(np.arange(1, n + 1) * DT / t_c, 1.0)); q0 = np.asarray(task["q0"], float)
        X = np.linalg.lstsq(Bm[:, 1:], ar - Bm[:, :1] * q0[None], rcond=None)[0]
        out.append(("c" + tg, dict(P=np.vstack([q0[None], X]).T, theta=w["theta"])))
    return out


def TS_ik(P_, pts, seed):
    return C.TS._ik_path(P_.robot, pts, seed, P_.j6)


if __name__ == "__main__":
    ci = int(sys.argv[1]); out = sys.argv[2]
    (xi, yi), (xn, yn), b, _, _ = C.CASES[ci]
    oi = C.moving([xi, yi, C.ZI]); on = C.moving([xn, yn, C.ZI])
    vin0 = 0.6 * C.vbar_of((xi, yi), C.ZI)
    Uh = hist_exact(oi, vin0)
    hv = np.abs(np.diff(Uh, axis=0) / DT) / QD_FULL; ha = np.abs(np.diff(Uh, 2, axis=0) / DT ** 2) / QDD_FULL
    print("이력 재현 오차", C._PR["hist_err"], "이력 명령 속도비", hv.max(0).round(3), "가속도비", ha.max(0).round(3), flush=True)
    # 필터 확인: 상자 둘 재귀 = 실제 커널 (지연 제외)
    hk = C.hk; rnd = np.random.default_rng(0).normal(size=(600, 6)).cumsum(0) * 0.01
    ref = np.stack([np.convolve(np.vstack([np.repeat(rnd[:1], 400, 0), rnd])[:, j], hk)[400:1000] for j in range(6)], -1)
    _, _, a_r = hist_realized(rnd)
    filt_err = float(np.abs(ref[C.ND:] - a_r[400:1000 - C.ND]).max())
    task, hd = C.make_task(Uh, oi, on, b, C.ZI, C.ZI)
    u_h, y_h, a_h = hist_realized(Uh)
    rr = np.arange(-40, 1) * DT; S0 = np.zeros((6, 4))
    for j in range(6):                                                        # 이력 실제의 r = 0 왼쪽 도함수 (5 차 다항식 맞춤)
        cf = np.polyfit(rr, a_h[-41:, j] * SG[j], 5)
        for nu in range(1, 5):
            S0[j, nu - 1] = np.polyval(np.polyder(cf, nu), 0.0)
    task["q0"] = a_h[-1] * SG; task["S0"] = S0
    task["HR"] = (u_h, y_h, a_h)
    C._S["S"] = build(); C.Run = RRun
    if CMDLIM:
        _cl = C.clamp; C.clamp = lambda th: (lambda t: np.r_[min(t[0], TC_MAX - 0.01), t[1:]])(_cl(th))
    mode = os.environ.get("REAL_INIT", "real")
    t0 = time.time(); inits = real_init(task, Uh, vin0) if mode == "real" else cmd_init(task, Uh, vin0)
    res = C.optimize(task, hd, inits)
    log = dict(case=ci, filt_err=filt_err, hist_err=C._PR.get("hist_err"), s=time.time() - t0, n_init=len(inits),
               hist_cmd_vel_ratio=hv.max(0).round(3).tolist(), hist_cmd_acc_ratio=ha.max(0).round(3).tolist())
    if not res["ok"]:
        log.update(ok=False, why=res["why"]); json.dump(log, open(out, "w"), ensure_ascii=False, indent=1); print(log); sys.exit(0)
    t_c, t_st, t_s, t_k = (float(v) for v in res["theta"]); Pr = res["P"]
    n = int(round(t_c / DT)); r = np.arange(1, n + 1) * DT
    a_new = (DB[0](np.minimum(r / t_c, 1.0)) @ Pr.T) * SG                    # 실제 (로봇 규약)
    u = inverse(u_h, y_h, a_h, a_new); L = len(u_h)
    U = u[L - 1:]                                                             # 코드 τ = 0 (이력 끝) .. t_c
    # 확인: 역계산 명령을 다시 필터하면 실제 궤적이 나오는가
    _, _, a_chk = hist_realized(u[400:]); recon_err = float(np.abs(a_chk[-n:] - a_new).max())
    vel = np.diff(U, axis=0) / DT; acc = np.diff(U, 2, axis=0) / DT ** 2
    rv = np.abs(vel) / QD_FULL; ra = np.abs(acc) / QDD_FULL
    HL = np.maximum(T.GP8_Q_MIN, T.Q_LO_REAL); HH = np.minimum(T.GP8_Q_MAX, T.Q_HI_REAL); Upl = U * SG
    pos_over = float(np.maximum(np.maximum(HL - Upl, Upl - HH), 0).max())
    # 실제 궤적 쪽 (스플라인): 속도·가속도 비 (≤ 0.9 이어야), jerk·4 계 (필터가 낼 수 있는 상한 2 q̈/W2, 4 q̈/(W1 W2) 대비)
    rs = np.linspace(0, 1, 2001)
    rv_r = (np.abs(DB[1](rs) @ Pr.T) / t_c / QD_FULL).max(0); ra_r = (np.abs(DB[2](rs) @ Pr.T) / t_c ** 2 / QDD_FULL).max(0)
    jr = (np.abs(DB[3](rs) @ Pr.T) / t_c ** 3).max(0) / (2 * QDD_FULL / FW2)
    sr = (np.abs(DB[4](rs) @ Pr.T) / t_c ** 4).max(0) / (4 * QDD_FULL / (FW1 * FW2))
    pc = tcp_batch(U); vc = np.linalg.norm(np.diff(pc, axis=0), axis=1) / DT
    pr_ = tcp_batch(a_new); vr = np.linalg.norm(np.diff(pr_, axis=0), axis=1) / DT
    over_v = rv > 1.0; over_a = ra > 1.0
    log.update(ok=True, theta=[t_c, t_st, t_s, t_k], J=res["J"], it=res["it"], recon_err=recon_err,
               cmd_vel_ratio_max=rv.max(0).round(3).tolist(), cmd_acc_ratio_max=ra.max(0).round(3).tolist(),
               cmd_vel_over_frac=over_v.mean(0).round(3).tolist(), cmd_acc_over_frac=over_a.mean(0).round(3).tolist(),
               cmd_vel_tmax=(np.argmax(rv, 0) * DT).round(3).tolist(), cmd_acc_tmax=(np.argmax(ra, 0) * DT).round(3).tolist(),
               cmd_pos_over_rad=pos_over, real_vel_ratio_max=rv_r.round(3).tolist(), real_acc_ratio_max=ra_r.round(3).tolist(),
               real_jerk_vs_filter=jr.round(2).tolist(), real_snap_vs_filter=sr.round(2).tolist(),
               cmd_tcp_speed_max=float(vc.max()), real_tcp_speed_max=float(vr.max()))
    np.savez(out.replace(".json", ".npz"), U=U, a_new=a_new, P=Pr, theta=res["theta"])
    json.dump(log, open(out, "w"), ensure_ascii=False, indent=1); print(json.dumps(log, ensure_ascii=False))

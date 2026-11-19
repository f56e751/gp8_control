"""11-05 사용자 새 사이클 정식화 시험 (앱 코드와 분리). 사이클 = 물체 i 잡기 (h 에서 하강 시작) + 던지기, 끝 = 물체 i+1 위 h 하강 시작.
모든 시각 = 실제 궤적 기준 r (사이클 시작 0). 명령 시각 s = r − D 이고 코드 축 τ = r + D 에서 cycle_nlp.weights 를 쓴다 (명령 [0, t_c]).
  [0, t_s]: z ≥ z_i, xy = o_i, xy 속도 = 벨트, ω = 0   (구간 조건은 노드에서 — 등식은 띠 폭 BAND_* 로, 0 이면 정확 등식)
  t_k     : 위치 = o_i + 윗면, 속도 = (벨트, 0)
  [t_s, t_c]: z ≥ h
  t_c     : 위치 = (o_{i+1} xy, h), 툴 수직 + J6, ω = 0, xy 속도 = 벨트, −v̄ ≤ ż ≤ −1e-3
목적 = 창 + W1 t_c (cycle_nlp 와 같음). θ = (t_c, t*, t_s, t_k) 바깥 BFGS.
사용: python cyc2.py case_index out.json"""
import os, sys, json, time
os.environ.setdefault("GP8_CYC_NCTRL", "40")
import numpy as np

from gp8_control.throwing import cycle_nlp as CY
from gp8_control.throwing.nlp import throw_nlp as T
from gp8_control.throwing.nlp.planner import PLANNER_SIGN as SG, INIT_VARIANTS
from gp8_control.throwing import sep_throw as ST
from gp8_control.grasping import sep_pick as SP
from gp8_control.grasping import throw_sequence as TS
from gp8_control.grasping.workspace_gate import tcp_batch

DT = 0.002
DELTA = float(os.environ.get("CYC2_DELTA", "0.02"))
BAND_XY = float(os.environ.get("CYC2_BAND_XY", "0.001"))
BAND_V = float(os.environ.get("CYC2_BAND_V", "0.01"))
BAND_W = float(os.environ.get("CYC2_BAND_W", "0.02"))
N = CY.N_CTRL
_BX = float(__import__("os").environ.get("RGE_BIN_X", "1.35"))                    # bin distance; RGE_BIN_X=1.1 for the near-bin test (user 11-13)
_BZ = float(__import__("os").environ.get("RGE_BIN_Z", "-0.06"))                   # height of the bin targets; RGE_BIN_Z=-0.11 (user 11-18)
BINS = [(_BX, 0.33, _BZ), (_BX, 0.0, _BZ), (_BX, -0.33, _BZ)]
VB = np.array([0.0, -float(__import__("os").environ.get("RGE_BELT_V", "0.12")), 0.0])            # belt speed; RGE_BELT_V = speed measured on the first object (real robot), default 0.12
H = SP.Z_H
hk = SP._h(); ND = int(np.argmax(np.abs(hk) > 1e-15)); D = ND * DT; LAG = float(np.arange(len(hk)) @ hk) * DT
MEM = len(hk) * DT
NP_PATH = 40; NP_HI = 40
OFF = set(x for x in os.environ.get("CYC2_OFF", "").split(",") if x)   # 진단: 뺄 조건 묶음 (xy,v,w,zi,k,zh,endp,endrot,endw,endv)
GROUPS = []                                                    # 노드 수 상한 (θ 바뀌어도 식 꼴 고정: 노드 수 고정)


def build():
    import casadi as cs
    c = CY._consts(); NW = T.N_WIN
    HW_LO = np.maximum(T.GP8_Q_MIN, T.Q_LO_REAL) + CY.Q_MARGIN; HW_HI = np.minimum(T.GP8_Q_MAX, T.Q_HI_REAL) - CY.Q_MARGIN
    o = cs.Opti(); Pf = o.variable(6, N - 1)
    par = {k: o.parameter(*((n,) if isinstance(n, int) else n)) for k, n in dict(
        q0=6, v0=6, t_c=1, tgt=3, j6=1, zi=1, zh=1, vbar=1,
        Aw=(N, NW), Adw=(N, NW), hw=(6, NW), hdw=(6, NW),
        Ap=(N, NP_PATH), Apd=(N, NP_PATH), hp=(6, NP_PATH), hpd=(6, NP_PATH), Op=(2, NP_PATH),
        Ah=(N, NP_HI), hh=(6, NP_HI),
        Ak=N, Akd=N, hk_=6, hkd=6, ok=3,
        Ae=N, Aed=N, he=6, hed=6, oe=2).items()}
    t_c = par["t_c"]; P = cs.horzcat(par["q0"], Pf)
    for k in range(1, N):
        o.subject_to(o.bounded(HW_LO, P[:, k], HW_HI))
    for k in range(N - 1):
        span = c["KN"][k + CY.DEG + 1] - c["KN"][k + 1]
        o.subject_to(o.bounded(-T.QD_LIM, CY.DEG * (P[:, k + 1] - P[:, k]) / span / t_c, T.QD_LIM))
    qd0 = CY.DEG * (P[:, 1] - P[:, 0]) / (c["U_HEAD"] * t_c)
    o.subject_to(o.bounded(-T.QDD_LIM * 0.002, qd0 - par["v0"], T.QDD_LIM * 0.002))
    for i in range(c["DDB"].shape[0]):
        o.subject_to(o.bounded(-T.QDD_LIM, cs.mtimes(P, c["DDB"][i]) / t_c ** 2, T.QDD_LIM))
    F_p, F_a, F_pd, _ = T.end_kinematics_functions()
    vbc = cs.DM(VB)
    GROUPS.clear(); GROUPS.append(("cmd", 0, o.ng)); mark = [o.ng]

    def grp(name):
        GROUPS.append((name, mark[0], o.ng)); mark[0] = o.ng
    cons = lambda name, c_: None if name in OFF else o.subject_to(c_)
    # [0, t_s] 노드: z ≥ z_i, xy = o_i, xy 속도 = 벨트, ω = 0
    for m in range(NP_PATH):
        q = cs.mtimes(P, par["Ap"][:, m]) + par["hp"][:, m]; qd = cs.mtimes(P, par["Apd"][:, m]) + par["hpd"][:, m]
        p = F_p(q); v = F_pd(q, qd); w = T.fk_omega_sym(q, qd)
        cons("zi", p[2] - par["zi"] >= 0)
        for nm, e, b in (("xy", p[0:2] - par["Op"][:, m], BAND_XY), ("v", v[0:2] - vbc[0:2], BAND_V), ("w", w, BAND_W)):
            cons(nm, e == 0) if b <= 0 else cons(nm, o.bounded(-b, e, b))
    grp("path")
    # 접촉 t_k
    q = cs.mtimes(P, par["Ak"]) + par["hk_"]; qd = cs.mtimes(P, par["Akd"]) + par["hkd"]
    cons("k", F_p(q) - par["ok"] == 0); cons("k", F_pd(q, qd) - vbc == 0); grp("k")
    # [t_s, t_c] 노드: z ≥ h
    for m in range(NP_HI):
        cons("zh", F_p(cs.mtimes(P, par["Ah"][:, m]) + par["hh"][:, m])[2] - par["zh"] >= 0)
    grp("zh")
    # 끝 t_c
    q = cs.mtimes(P, par["Ae"]) + par["he"]; qd = cs.mtimes(P, par["Aed"]) + par["hed"]
    p = F_p(q); v = F_pd(q, qd); a = F_a(q)
    cons("endp", p[0:2] - par["oe"] == 0); cons("endp", p[2] - par["zh"] == 0); grp("endp")
    cons("endrot", a[0] == 0); cons("endrot", a[1] == 0); cons("endrot", a[2] <= -0.5); cons("endrot", q[5] - par["j6"] == 0); grp("endrot")
    cons("endw", T.fk_omega_sym(q, qd) == 0); grp("endw")
    cons("endv", v[0:2] - vbc[0:2] == 0); cons("endv", o.bounded(-par["vbar"], v[2], -1e-3)); grp("endv")
    # 창
    mark[0] = o.ng
    F_err, F_z, _ = T.make_objective_functions_sym(); dragc = cs.DM(np.zeros(3))
    w_tr = np.full(NW, c["RT"] / (NW - 1)); w_tr[[0, -1]] *= 0.5
    J_win = 0; taus = []
    for m in range(NW):
        q = cs.mtimes(P, par["Aw"][:, m]) + par["hw"][:, m]; qd = cs.mtimes(P, par["Adw"][:, m]) + par["hdw"][:, m]
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


_S = {}


def S():
    if "S" not in _S:
        _S["S"] = build()
    return _S["S"]


def clamp(th):
    t_c, t_st, t_s, t_k = (float(v) for v in th); RT = CY._consts()["RT"]
    t_c = float(np.clip(t_c, 0.9, 2.6))
    t_k = float(np.clip(t_k, 0.08, 0.6))
    t_s = float(np.clip(t_s, t_k + 0.06, t_c - RT - 0.05))
    t_st = float(np.clip(t_st, t_s + RT / 2, t_c - RT / 2))
    return np.array([t_c, t_st, t_s, t_k])


class Run:
    def __init__(self, task, hist):
        self.t = task; self.hist = hist; s = S(); o, p = s["opti"], s["par"]
        for k in ("q0", "v0", "tgt", "j6", "zi", "zh", "vbar"):
            o.set_value(p[k], task[k])
        self.n_it = 0; self.n_solve = 0

    def set_theta(self, th):
        t_c, t_st, t_s, t_k = (float(v) for v in th); c = CY._consts(); s = S(); o, p = s["opti"], s["par"]
        W = lambda r: CY.weights(t_c, np.asarray(r, float) + D, self.hist)
        rp = np.linspace(0.0, t_s, NP_PATH + 1)[1:]                          # r=0 은 이력만으로 정해짐
        rh = np.linspace(t_s, t_c, NP_HI)
        Aw, Adw, hw, hdw = W(t_st + c["OFFS"])
        Ap, Apd, hp, hpd = W(rp)
        Ah, _, hh, _ = W(rh)
        Ak, Akd, hk_, hkd = W([t_k]); Ae, Aed, he, hed = W([t_c])
        oi = self.t["oi"]; on = self.t["on"]
        vals = dict(t_c=t_c, Aw=Aw.T, Adw=Adw.T, hw=hw, hdw=hdw, Ap=Ap.T, Apd=Apd.T, hp=hp, hpd=hpd,
                    Op=np.stack([oi(r)[:2] for r in rp], 1), Ah=Ah.T, hh=hh, Ak=Ak[0], Akd=Akd[0], hk_=hk_[:, 0], hkd=hkd[:, 0],
                    ok=np.r_[oi(t_k)[:2], self.t["zi"]], Ae=Ae[0], Aed=Aed[0], he=he[:, 0], hed=hed[:, 0], oe=on(t_c)[:2])
        for k, v in vals.items():
            o.set_value(p[k], v)

    def solve(self, th, init):
        s = S(); o = s["opti"]; self.set_theta(th)
        if "x" in init:
            o.set_initial(o.x, init["x"]); o.set_initial(o.lam_g, init["lam"])
        else:
            o.set_initial(o.lam_g, np.zeros(o.ng)); o.set_initial(s["Pf"], np.asarray(init["P"], float)[:, 1:])
            for tau in s["taus"]:
                o.set_initial(tau, 0.45)
        t0 = time.time()
        try:
            sol = o.solve(); st = "Solve_Succeeded"
        except RuntimeError as e_:
            sol = o.debug
            try:
                st = o.return_status()
            except RuntimeError:
                st = str(e_).splitlines()[-1]
        try:
            it = int(o.stats().get("iter_count", 0))
        except RuntimeError:                                                 # 풀이 시작 전 실패 (초기점 NaN 등)
            return dict(ok=False, status=f"시작 실패 ({str(st)[:80]})", J=np.nan, it=0, dt=time.time() - t0)
        self.n_it += it; self.n_solve += 1
        ok = st in ("Solve_Succeeded", "Solved_To_Acceptable_Level")
        if not ok:
            try:
                g = np.array(sol.value(o.g)).ravel(); lb = np.array(sol.value(o.lbg)).ravel(); ub = np.array(sol.value(o.ubg)).ravel()
                vio = np.maximum(lb - g, 0) + np.maximum(g - ub, 0)
                st = str(st) + " 위반[" + ", ".join(f"{nm} {vio[a:b].max():.3g}" for nm, a, b in GROUPS if b > a) + f", 창 {vio[GROUPS[-1][2]:].max() if len(vio) > GROUPS[-1][2] else 0:.3g}]"
            except Exception as e_:                                          # noqa: BLE001
                st = str(st) + f" (위반 계산 실패 {e_})"[:80]
        return dict(ok=ok, status=str(st), J=float(sol.value(s["J"])), x=np.array(sol.value(o.x)).ravel(),
                    lam=np.array(sol.value(o.lam_g)).ravel(), Pf=np.array(sol.value(s["Pf"])), it=it, dt=time.time() - t0)

    def grad(self, th, r):
        o = S()["opti"]; g = []
        for d in np.eye(4) * 2e-4:
            Ls = []
            for sg in (1, -1):
                self.set_theta(th + sg * d); Ls.append(float(S()["L"](r["x"], o.value(o.p), r["lam"])))
            g.append((Ls[0] - Ls[1]) / 4e-4)
        self.set_theta(th)
        return np.array(g)


def optimize(task, hist, inits, max_outer=4):
    R = Run(task, hist); r = None; th = None; fails = []; tag = None; last = None
    for tg, w in inits:
        th = clamp(w["theta"]); rr = R.solve(th, w)
        if rr["ok"]:
            r, tag = rr, tg; break
        fails.append(f"{tg}: {rr['status']} ({rr['it']})"); last = rr
    if r is None:
        out = dict(ok=False, why="; ".join(fails), it=R.n_it)
        if last is not None and "Pf" in last and "User_Requested_Stop" in str(last.get("status")):   # 11-08 마감 반복값 (호출자가 gate 로 검사)
            out["partial"] = dict(P=np.hstack([np.asarray(task["q0"], float)[:, None], np.asarray(last["Pf"], float)]), theta=np.asarray(th, float),
                                  J=float(last["J"]), it=int(last["it"]), x=last["x"], lam=last["lam"])
        return out
    g = R.grad(th, r); Hm = np.eye(4) * 200.0
    for _k in range(max_outer):
        step = -np.linalg.solve(Hm, g); n = np.linalg.norm(step)
        if n > 0.05:
            step *= 0.05 / n
        acc = False
        for _ls in range(5):
            th_n = clamp(th + step); rn = R.solve(th_n, dict(x=r["x"], lam=r["lam"]))
            if rn["ok"] and rn["J"] < r["J"] - 1e-9:
                acc = True; break
            step *= 0.5
        if not acc:
            break
        g_n = R.grad(th_n, rn); s_ = th_n - th; y = g_n - g
        if s_ @ y > 1e-12:
            Hs = Hm @ s_; Hm = Hm - np.outer(Hs, Hs) / (s_ @ Hs) + np.outer(y, y) / (s_ @ y)
        dJ = r["J"] - rn["J"]; th, r, g = th_n, rn, g_n
        if np.linalg.norm(s_) < 3e-4 or dJ < 1e-6 * abs(r["J"]):
            break
    P = np.hstack([np.asarray(task["q0"], float)[:, None], r["Pf"]])
    return dict(ok=True, P=P, theta=th, J=r["J"], tag=tag, it=R.n_it, n_solve=R.n_solve)


# ---------------------------------------------------------------- 과제·초기값
_PR = {}


def prim():
    if "P" not in _PR:
        _PR["P"] = TS.ThrowPrimitive(0.12)
    return _PR["P"]


def yaw_c():
    """수직 툴에서 ω_z = 0 이 되는 J6 비율 c (로봇 규약 q̇6 = c q̇1)."""
    if "c" not in _PR:
        import casadi as cs
        q = cs.MX.sym("q", 6); qd = cs.MX.sym("qd", 6); Fw = cs.Function("w", [q, qd], [T.fk_omega_sym(q, qd)])
        q0 = prim()._pick_ik(np.array([0.4, 0.0, H]))
        w1 = float(np.array(Fw(q0 * SG, np.eye(6)[0] * SG)).ravel()[2]); w6 = float(np.array(Fw(q0 * SG, np.eye(6)[5] * SG)).ravel()[2])
        _PR["c"] = -w1 / w6
    return _PR["c"]


def yaw_fix(Q, q1_ref):
    """하강·상승: 툴 수직축 회전 0 — J6 = j6 + c (q1 − q1_ref) (q1_ref 에서 J6 = j6 = 잡기 자세)."""
    Q = np.array(Q, float); Q[:, 5] = prim().j6 + yaw_c() * (Q[:, 0] - q1_ref)
    return Q


def moving(p0):
    p0 = np.asarray(p0, float)
    return lambda r: p0 + VB * r


def vbar_of(xy, z):
    """hover 높이 h 의 하강 속도 척도 v̄ (λ_z 정규화). 수직선 표가 없는 반지름 (이 로봇 ≈ 0.18 아래·0.72 위) 은 표가 있는
    가장 가까운 반지름으로 대신한다 (11-07 g10: 다음 물체가 τ=0 에 r 0.71–0.81 이면 과제 생성이 실패하던 것)."""
    r_ = round(float(np.hypot(*xy)), 2); step = -0.01 if r_ > 0.45 else 0.01
    for rt in np.round(np.arange(r_, 0.45 + step, step), 2):
        try:
            u_up = float(SP.line_bounds(float(rt), z_lo=-0.18)[0](H))
        except (TypeError, ValueError, IndexError):
            continue
        return SP.vbar(H - z, u_up)
    raise ValueError(f"v̄: 수직선 표 없음 r {r_}")


def zprof(r, t_k, t_s, zi, vin):
    """실제 z 초기 모양: h (−vin) → z_i (0) at t_k → h + 0.02 (+1.0 m/s) at t_s, Hermite."""
    def herm(s, T_, p0, v0, p1, v1):
        u = s / T_; h00 = 2*u**3 - 3*u**2 + 1; h10 = u**3 - 2*u**2 + u; h01 = -2*u**3 + 3*u**2; h11 = u**3 - u**2
        return h00 * p0 + h10 * T_ * v0 + h01 * p1 + h11 * T_ * v1
    if r <= 0:
        return H - vin * r
    if r <= t_k:
        return herm(r, t_k, H, -vin, zi, 0.0)
    return herm(min(r, t_s) - t_k, t_s - t_k, zi, 0.0, H + 0.02, 1.0)


def path_init(task, hist_U, vin0):
    """실제 경로 모양을 L−D 앞당긴 명령 → B-스플라인 초기값 (명령 [0, t_c]). 여러 던지기 변형."""
    P_ = prim(); oi, on = task["oi"], task["on"]; zi = task["zi"]; out = []
    lead = LAG - D; q_seed = hist_U[-1]
    W12 = MEM - D                                                            # 실제 속도는 명령을 W1+W2 평균 — 하강을 세우는 데 그 이상 걸린다
    for kv, (v, t_k) in enumerate([(v, tk) for tk in (W12 + 0.02, W12 + 0.08) for v in [v for v in INIT_VARIANTS if v is not None][:3]]):
        t_s = t_k + W12 + 0.02
        tf = float(np.clip(v["T0"], 0.7, 1.0)); t_c = t_s + tf
        try:
            rr = np.arange(0.0, t_s + 1e-9, DT)
            pts = np.array([np.r_[oi(r)[:2], zprof(r, t_k, t_s, zi, vin0)] for r in rr])
            Qp = TS._ik_path(P_.robot, pts, q_seed, P_.j6); Qp = yaw_fix(Qp, Qp[0, 0])
            q_h = P_._pick_ik(np.r_[on(t_c)[:2], H], q_init=Qp[-1])
            if q_h is None:
                continue
            Pt = T.init_ctrl_points(Qp[-1] * SG, np.asarray(v["dq_swing"], float), float(v["chi0"]), p_f=q_h * SG)
            from scipy.interpolate import BSpline
            sp = BSpline(T.KNOTS, Pt.T, T.DEGREE); n1 = int(round(tf / DT)); Qt = sp(np.arange(1, n1 + 1) / n1) * SG
            Qr = np.vstack([Qp, Qt, np.repeat(q_h[None], int(round(lead / DT)) + 2, 0)])   # 실제 경로 r = 0 ..
            nl = int(round(lead / DT)); n_c = int(round(t_c / DT))
            Qc = Qr[nl:nl + n_c + 1]                                          # 명령 s = r − (L − D)... 코드축 명령 [0, t_c]
            Qc[0] = hist_U[-1]
            out.append((f"cold{kv}", dict(P=CY.fit(Qc * SG, t_c, hist_U[-1] * SG), theta=(t_c, t_s + float(v["chi0"]) * tf, t_s, t_k))))
        except ValueError:
            continue
    return out


def ideal_hist(o_p, zi_unused, vin, n_iter=60):
    """실제 툴이 r ≤ 0 에서 o 위 직선 (벨트, −vin) 하강이고 r = 0 에 h 를 지나도록 하는 명령 이력 (로봇 규약, 코드축 τ ≤ 0 = 실제 r ≤ −D).
    목표 관절 = 직선 IK (J6 는 수직축 회전 0 규칙, r = 0 에서 J6 = j6), 명령 = 필터 역보정 (U += 오차를 L 앞당겨 더하기, 반복)."""
    P_ = prim(); n1 = int(round(0.56 / DT)); n2 = int(round(0.4 / DT))
    tau = np.arange(-n1, n2) * DT; rr = tau - D                              # 코드축 τ 의 실제 시각 r
    def zline(r):                                                           # 하강 속력: r ≤ −0.75 정지 → −0.35 부터 vin (매끄러운 가속, 필터 기억 0.32 s 밖)
        if r >= -0.33:
            return H - vin * r
        a_, b_ = -0.50, -0.33; u = np.clip((np.linspace(max(r, a_), b_, 200) - a_) / (b_ - a_), 0, 1)
        sv = u * u * (3 - 2 * u); dz = np.trapz(vin * sv, np.linspace(max(r, a_), b_, 200))
        return H + vin * 0.33 + dz
    line = lambda r: np.r_[o_p(r)[:2], zline(r)]
    q_mid = P_._pick_ik(line(0.0))
    if q_mid is None:
        raise ValueError("이력 IK")
    k0 = int(np.argmin(np.abs(rr)))
    Qa = TS._ik_path(P_.robot, np.array([line(r) for r in rr[k0::-1]]), q_mid, P_.j6)[::-1]
    Qb = TS._ik_path(P_.robot, np.array([line(r) for r in rr[k0:]]), q_mid, P_.j6)[1:]
    Qt = yaw_fix(np.vstack([Qa, Qb]), q_mid[0])
    nl = int(round(LAG / DT)); pad = len(hk)

    def filt(U):
        Up = np.vstack([np.repeat(U[:1], pad, 0), U])
        return np.stack([np.convolve(Up[:, j], hk)[pad:pad + len(U)] for j in range(6)], -1)
    n = len(Qt); A = np.zeros((n, n))                                     # 실제 = A U (앞쪽 명령은 첫 명령 유지), 매끄러움 벌점 최소제곱
    for m, hm in enumerate(hk):
        if hm != 0.0:
            A[np.arange(n), np.maximum(np.arange(n) - m, 0)] += hm
    U0 = np.vstack([Qt[nl:], np.repeat(Qt[-1:], nl, 0)])                  # 단순 앞당김 명령 — 이동평균 영점의 진동을 막는 기준
    lam = float(os.environ.get("CYC2_HLAM", "0.03"))
    U = np.linalg.lstsq(np.vstack([A, lam * np.eye(n)]), np.vstack([Qt, lam * U0]), rcond=None)[0]
    E = Qt - filt(U); sl = slice(k0 - int(round(0.1 / DT)), k0 + nl)      # 확인: 실제 r ∈ [−0.1, L−D]
    Pt = tcp_batch(Qt[sl]); Pr = tcp_batch(filt(U)[sl])
    _PR["hist_err"] = float(np.abs(Pt - Pr).max())                            # 이력 보정 뒤 실제 TCP 오차 (r ≤ L−D)
    kz = int(np.argmin(np.abs(tau)))
    return U[:kz + 1][-(len(ST.G_H) + 12):]


def make_task(Uh, oi, on, b, zi, zn):
    hd = ST.hist_data(Uh, Uh[-1:], 0.0, 0.1)
    q0 = Uh[-1] * SG; v0 = (Uh[-1] - Uh[-2]) / DT * SG
    return dict(q0=q0, v0=v0, tgt=np.array(BINS[b]), j6=float(prim().j6) * SG[5], zi=zi, zh=H, vbar=vbar_of(on(0.0)[:2], zn),
                oi=oi, on=on), hd


def evaluate(res, Uh, task):
    """수치 실제 궤적: 끝 상태, 조건 위반, 꼬리 명령, 끝난 뒤 (명령 유지) 실제."""
    t_c, t_st, t_s, t_k = res["theta"]
    U = CY.sample(res["P"], t_c) * SG                                        # 코드축 [0, t_c] 명령
    pad = len(hk) + ND + 5
    Uf = np.vstack([Uh[:-1], U, np.repeat(U[-1:], pad, 0)])
    qr = np.stack([np.convolve(Uf[:, j], hk)[:len(Uf)] for j in range(6)], -1)
    k0 = len(Uh) - 1                                                          # 코드 τ=0
    pr = tcp_batch(qr); pc = tcp_batch(Uf)
    kr = lambda r: k0 + int(round((r + D) / DT))                              # 실제 r → 인덱스
    oi, on = task["oi"], task["on"]
    rr = np.arange(0, t_s, DT)
    e_xy = max(float(np.abs(pr[kr(r), :2] - oi(r)[:2]).max()) for r in rr)
    z_min_path = float(min(pr[kr(r), 2] - task["zi"] for r in rr))
    z_min_hi = float(min(pr[kr(r), 2] for r in np.arange(t_s, t_c, DT)) - H)
    ve = (pr[kr(t_c) + 1] - pr[kr(t_c) - 1]) / (2 * DT)
    ncmd = int(round(MEM / DT))
    vc = np.linalg.norm(np.diff(pc[k0 + len(U) - ncmd:k0 + len(U)], axis=0), axis=1) / DT  # 마지막 필터 기억 명령 TCP 속력
    # 끝난 뒤 명령 유지 시 실제 (게이트 방식): 다음 물체 xy 와 차이·z 최저
    post = np.arange(t_c, t_c + MEM, DT)
    e_post = max(float(np.linalg.norm(pr[kr(r), :2] - on(r)[:2])) for r in post)
    z_post = float(min(pr[kr(r), 2] for r in post))
    x_min = float(pr[kr(0):kr(t_c), 0].min())
    return dict(theta=[float(v) for v in res["theta"]], J=res["J"], it=res["it"], e_xy_path=e_xy, z_path_minus_zi=z_min_path,
                z_hi_minus_h=z_min_hi, end_p=pr[kr(t_c)].round(4).tolist(), end_v=ve.round(3).tolist(),
                tail_cmd_speed_max=float(vc.max()), tail_cmd_speed_end=float(vc[-1]), post_hold_exy=e_post, post_hold_zmin=z_post,
                x_min=x_min, U=U, v_in=float(-ve[2]))


CASES = [  # (o_i xy, o_n xy, bin, B 의 다음 xy, B bin)
    ((0.40, 0.00), (0.40, 0.15), 1, (0.35, 0.30), 0),
    ((0.30, -0.15), (0.45, 0.00), 0, (0.30, 0.20), 2),
    ((0.50, 0.10), (0.35, 0.20), 2, (0.45, 0.35), 1),
    ((0.35, -0.30), (0.40, -0.15), 1, (0.50, 0.00), 0),
    ((0.45, 0.20), (0.30, 0.30), 0, (0.40, 0.45), 2),
]
ZI = 0.02


if __name__ == "__main__":
    ci = int(sys.argv[1]); out = sys.argv[2]
    (xi, yi), (xn, yn), b, (x2, y2), b2 = CASES[ci]
    log = dict(case=ci)
    oi = moving([xi, yi, ZI]); on = moving([xn, yn, ZI])
    vin0 = 0.6 * vbar_of((xi, yi), ZI)
    UhA = ideal_hist(oi, ZI, vin0); log["hist_err_A"] = _PR.get("hist_err"); print("이력 오차", _PR.get("hist_err"), "yaw c", yaw_c(), flush=True)
    taskA, hdA = make_task(UhA, oi, on, b, ZI, ZI)
    t0 = time.time(); inits = path_init(taskA, UhA, vin0)
    if os.environ.get("CYC2_NINIT"):
        inits = inits[::max(1, len(inits) // int(os.environ["CYC2_NINIT"]))][:int(os.environ["CYC2_NINIT"])]
    rA = optimize(taskA, hdA, inits)
    log["A_time"] = time.time() - t0; log["A_inits"] = len(inits)
    if not rA["ok"]:
        log["A"] = dict(ok=False, why=rA["why"]); json.dump(log, open(out, "w"), ensure_ascii=False, indent=1); print(log); sys.exit(0)
    eA = evaluate(rA, UhA, taskA); UA = eA.pop("U"); log["A"] = dict(ok=True, **eA)
    print("A", json.dumps(log["A"], ensure_ascii=False), flush=True)
    if os.environ.get("CYC2_AONLY"):
        json.dump(log, open(out, "w"), ensure_ascii=False, indent=1); sys.exit(0)
    # B: 다음 사이클 (물체 i+1 잡기 + 던지기), 시작 = A 의 실제 끝. 시간 원점 = A 의 t_c
    tcA = rA["theta"][0]
    oiB = lambda r: on(r + tcA); onB = moving([x2, y2 - 0.0, ZI])
    onB2 = lambda r: onB(r + tcA)
    vinB = max(eA["v_in"], 0.05)
    for tag, Uh in (("B_from_A", np.vstack([UhA[:-1], UA])[-(len(ST.G_H) + 12):]), ("B_ideal", ideal_hist(oiB, ZI, vinB))):
        taskB, hdB = make_task(Uh, oiB, onB2, b2, ZI, ZI)
        t0 = time.time(); inits = path_init(taskB, Uh, vinB)
        rB = optimize(taskB, hdB, inits)
        if rB["ok"]:
            eB = evaluate(rB, Uh, taskB); eB.pop("U"); log[tag] = dict(ok=True, s=time.time() - t0, n_init=len(inits), **eB)
        else:
            log[tag] = dict(ok=False, s=time.time() - t0, n_init=len(inits), why=rB["why"])
        print(tag, json.dumps(log[tag], ensure_ascii=False), flush=True)
    json.dump(log, open(out, "w"), ensure_ascii=False, indent=1)

"""통합 사이클 NLP (2026-11-04 사용자 "통합 NLP를 그냥 할까 제약식을 최대한 간단하게", 논문 version2 식 6).

한 사이클 = 던지기 i + 잡기 i+1, 명령 B-스플라인 하나 u(t) = Σ P_k N_k(t/t_c), 경계 = 잡은 뒤 상승 중 (앞 사이클 마지막 명령들이 실제를
D + W₁ + W₂ 동안 끌고 가므로, 정밀 조건이 없는 상승 중에 끊는다). 명령에는 𝒰 만, 등식은 모두 필터를 지난 실제 궤적:
  접촉 t_k: 실제 위치 = 물체 윗면 (xy 는 그 시각 물체), 실제 속도 = (벨트 xy, 0), 잡기 자세 (툴 축 수직 + J6)
  접촉 근처 σ ∈ 𝒮: 실제 z ≥ 윗면, |실제 xy − 물체 xy| ≤ s̄
  끝 t_c + D: 실제 z = h, 실제 ż ≥ 0
  방출 창 [t* ± ΔT/2]: 착지 오차 평균 + 각속도 (던지기 NLP 와 같은 비용), 목적 = W2·창 + W1·t_c
구조는 sep_alt 와 같다 — 안쪽: θ = (t_c, t*, t_k) 숫자 고정 → 실제 (q, q̇)(t) = P·a(t) + h(t) 가 제어점의 선형식, NLP 안 비선형은 FK 뿐.
바깥: θ 를 BFGS (기울기 = 포락선 정리 dJ*/dθ = ∂L/∂θ 중앙 차분)."""
import os
import time

import numpy as np
from scipy.interpolate import BSpline

from gp8_control.throwing.nlp import throw_nlp as T

N_CTRL = int(os.environ.get("GP8_CYC_NCTRL", "20"))                 # 관절당 제어점 수
DEG = 3
S_BAR = float(os.environ.get("GP8_SEP_V3_SBAR", "0.005"))           # 접촉 근처 실제 xy 허용 [m]
SIGMAS = np.array([-0.10, -0.05, -0.02, 0.02, 0.05, 0.10])           # 접촉 근처 검사 시각 [s]
TC_BOUNDS = (0.6, 2.2)
Q_MARGIN = np.deg2rad(0.2)                                            # 관절 위치 범위 안쪽 여유
MAX_OUTER = int(os.environ.get("GP8_CYC_MAXOUT", "20"))
STEP_MAX = 0.05
H0 = 200.0
N_LS = 6
GRAD_H = 2e-4
_C = {}


def _consts():
    if "c" not in _C:
        kn = np.concatenate([np.zeros(DEG), np.linspace(0.0, 1.0, N_CTRL - DEG + 1), np.ones(DEG)])
        BS = BSpline(kn, np.eye(N_CTRL), DEG)
        D, Ws = T.FILTER
        W1F, W2F = Ws
        uk = np.unique(kn)
        _C["c"] = dict(KN=kn, B0=BS, B1=BS.antiderivative(1), B2=BS.antiderivative(2), DDB=BS.derivative(2)(uk),
                       U_HEAD=kn[DEG + 1], D=D, TAPS=np.array([D, D + W1F, D + W2F, D + W1F + W2F]),
                       SG=np.array([1.0, -1.0, -1.0, 1.0]) / (W1F * W2F), RT=T.RELEASE_TIME,
                       OFFS=np.linspace(-T.RELEASE_TIME / 2, T.RELEASE_TIME / 2, T.N_WIN))
    return _C["c"]


def weights(t_c: float, times, hist: dict):
    """실제 (q, q̇)(t) = P·a(t) + h(t) — 리턴 Aq (m,N), Aqd (m,N), hq (6,m), hqd (6,m). sep_alt.weights 와 같은 식 (이 모듈의 매듭)."""
    from gp8_control.throwing import sep_throw as ST
    c = _consts(); H_S = ST.H_S
    tau = np.asarray(times, float)[:, None] - c["TAPS"][None, :]
    pos = tau >= 0
    s = np.clip(tau / t_c, 0.0, 1.0)
    b1 = c["B1"](s) * t_c
    b2 = c["B2"](s) * t_c ** 2
    b1[~pos] = 0.0
    b2[~pos] = 0.0
    sh = np.maximum(np.minimum(tau, 0.0), -H_S) / H_S
    V = sh[..., None] ** np.arange(np.asarray(hist["hF1"]).shape[1])
    h1 = V @ np.asarray(hist["hF1"]).T
    h2 = V @ np.asarray(hist["hF2"]).T
    h1[pos] = 0.0
    h2[pos] = 0.0
    return (np.einsum("k,mkn->mn", c["SG"], b2), np.einsum("k,mkn->mn", c["SG"], b1),
            np.einsum("k,mkj->jm", c["SG"], h2), np.einsum("k,mkj->jm", c["SG"], h1))


def weights_acc(t_c: float, times, hist: dict):
    """실제 q̈(t) = P·a(t) + h(t) (상자 둘의 2 차 미분 = 명령 u 의 탭 넷 부호 합). 리턴 A (m,N), h (6,m)."""
    from gp8_control.throwing import sep_throw as ST
    c = _consts(); H_S = ST.H_S
    tau = np.asarray(times, float)[:, None] - c["TAPS"][None, :]
    pos = tau >= 0
    b0 = c["B0"](np.clip(tau / t_c, 0.0, 1.0))
    b0[~pos] = 0.0
    hU = np.asarray(hist.get("hU", np.zeros((6, 1))))
    sh = np.maximum(np.minimum(tau, 0.0), -H_S) / H_S
    h0 = (sh[..., None] ** np.arange(hU.shape[1])) @ hU.T
    h0[pos] = 0.0
    return np.einsum("k,mkn->mn", c["SG"], b0), np.einsum("k,mkj->jm", c["SG"], h0)


def _build():
    import casadi as cs
    c = _consts(); N = N_CTRL; NS = len(SIGMAS); NW = T.N_WIN
    HW_LO = np.maximum(T.GP8_Q_MIN, T.Q_LO_REAL) + Q_MARGIN; HW_HI = np.minimum(T.GP8_Q_MAX, T.Q_HI_REAL) - Q_MARGIN   # 2 ms 샘플·gate 여유
    opti = cs.Opti()
    Pf = opti.variable(6, N - 1)
    par = {k: opti.parameter(*((n,) if isinstance(n, int) else n)) for k, n in dict(
        q0=6, v0=6, t_c=1, tgt=3, j6=1, ztop=1, zh=1, tgtc=3, vbc=3, On=(2, NS),
        Aw=(N, NW), Adw=(N, NW), hw=(6, NW), hdw=(6, NW), Ac=N, Acd=N, hc=6, hcd=6, An=(N, NS), hn=(6, NS),
        Ae=N, Aed=N, he=6, hed=6, Aa=N, ha=6, oe=2).items()}
    t_c = par["t_c"]
    P = cs.horzcat(par["q0"], Pf)
    for k in range(1, N):
        opti.subject_to(opti.bounded(HW_LO, P[:, k], HW_HI))
    for k in range(N - 1):
        span = c["KN"][k + DEG + 1] - c["KN"][k + 1]
        opti.subject_to(opti.bounded(-T.QD_LIM, DEG * (P[:, k + 1] - P[:, k]) / span / t_c, T.QD_LIM))
    qd0 = DEG * (P[:, 1] - P[:, 0]) / (c["U_HEAD"] * t_c)                   # 경계: 명령 속도 변화 ≤ 가속 한계 × 2 ms (𝒰)
    opti.subject_to(opti.bounded(-T.QDD_LIM * 0.002, qd0 - par["v0"], T.QDD_LIM * 0.002))
    for i in range(c["DDB"].shape[0]):
        opti.subject_to(opti.bounded(-T.QDD_LIM, cs.mtimes(P, c["DDB"][i]) / t_c ** 2, T.QDD_LIM))
    F_p, F_a, F_pd, F_ad = T.end_kinematics_functions()
    # 접촉 t_k (실제): 위치 = 윗면, 속도 = (벨트 xy, 0), 잡기 자세
    q_c = cs.mtimes(P, par["Ac"]) + par["hc"]; qd_c = cs.mtimes(P, par["Acd"]) + par["hcd"]
    opti.subject_to(F_p(q_c) - par["tgtc"] == 0)
    opti.subject_to(F_pd(q_c, qd_c) - par["vbc"] == 0)
    a_c = F_a(q_c)
    opti.subject_to(a_c[0] == 0); opti.subject_to(a_c[1] == 0); opti.subject_to(a_c[2] <= -0.5); opti.subject_to(q_c[5] - par["j6"] == 0)
    # 접촉 근처 (실제): z ≥ 윗면, |xy − 물체| ≤ s̄
    for m in range(NS):
        p_n = F_p(cs.mtimes(P, par["An"][:, m]) + par["hn"][:, m])
        opti.subject_to(p_n[2] - par["ztop"] >= 0)
        opti.subject_to(opti.bounded(-S_BAR, p_n[0:2] - par["On"][:, m], S_BAR))
    # 끝 t_c + D (실제): z = h, ż ≥ 0
    q_e = cs.mtimes(P, par["Ae"]) + par["he"]; qd_e = cs.mtimes(P, par["Aed"]) + par["hed"]
    p_e = F_p(q_e); v_e = F_pd(q_e, qd_e)
    opti.subject_to(p_e[2] - par["zh"] == 0)
    opti.subject_to(v_e[2] >= 0)
    # 11-04 RT 사이클 시뮬: z 만 걸면 끝 직전 명령이 멋대로 (h 위 0.7 m, 베이스 쪽) — 다음 사이클·정지가 끌려간다. 실제 툴이 물체 위를
    # 벨트와 함께 곧게 올라가는 상태로 넘긴다: 실제 xy = 물체, 실제 xy 속도 = 벨트, 실제 관절 가속 0
    opti.subject_to(p_e[0:2] - par["oe"] == 0)
    opti.subject_to(v_e[0:2] - par["vbc"][0:2] == 0)
    opti.subject_to(cs.mtimes(P, par["Aa"]) + par["ha"] == 0)
    # 방출 창 (실제): 착지 오차 + 각속도, 마디마다 비행 시간 τ
    F_err, F_z, _ = T.make_objective_functions_sym(); dragc = cs.DM(np.zeros(3))
    w_tr = np.full(NW, c["RT"] / (NW - 1)); w_tr[[0, -1]] *= 0.5
    J_win = 0; taus = []
    for m in range(NW):
        q_i = cs.mtimes(P, par["Aw"][:, m]) + par["hw"][:, m]; qd_i = cs.mtimes(P, par["Adw"][:, m]) + par["hdw"][:, m]
        tau = opti.variable(); taus.append(tau)
        opti.subject_to(opti.bounded(0.05, tau, 1.5)); opti.subject_to(F_z(q_i, qd_i, tau, par["tgt"], dragc) == 0)
        J_win = J_win + w_tr[m] * (T.W_ACC * F_err(q_i, qd_i, tau, par["tgt"], dragc) + T.W_OMEGA_LADDER[0] * cs.sumsqr(T.fk_omega_sym(q_i, qd_i)))
    J = T.W1 * t_c + T.W2 * J_win
    opti.minimize(J)
    ip = {"max_iter": 3000, "print_level": 0, "sb": "yes", "tol": 1e-6, "acceptable_tol": 1e-4, "acceptable_iter": 10,
          "mu_strategy": "adaptive", "warm_start_init_point": "yes", "warm_start_bound_push": 1e-9,
          "warm_start_mult_bound_push": 1e-9, "warm_start_slack_bound_push": 1e-9}
    cb = T._DeadlineStop(opti.nx, opti.ng)
    opti.solver("ipopt", {"print_time": False, "expand": True, "iteration_callback": cb}, ip)
    Lf = cs.Function("L", [opti.x, opti.p, opti.lam_g], [opti.f + cs.dot(opti.lam_g, opti.g)])
    return dict(opti=opti, Pf=Pf, taus=taus, par=par, J=J, J_win=J_win, L=Lf, cb=cb)


def solver():
    if "S" not in _C:
        _C["S"] = _build()
    return _C["S"]


def prewarm() -> float:
    """조립 + IPOPT 풀이기 생성 (첫 o.solve, ~1.5 s) 을 앱 시작 때 치른다: 마감을 지난 시각으로 걸어 첫 반복에서 멈추는 가짜 풀이 한 번
    (11-05 실시간 시뮬: 꼬리 첫 사이클이 0.72 s 마감 안에서 이 비용을 치러 0 반복으로 멈췄다)."""
    from gp8_control.throwing import sep_throw as ST
    t0 = time.time(); solver()
    q0 = np.array([0.6, 0.5, -0.1, 0.0, 1.0, 1.1]); Uh = np.repeat(q0[None], len(ST.G_H) + 2, 0)
    fixed = dict(q0=q0, v0=np.zeros(6), tgt=np.array([1.35, 0.0, -0.06]), j6=1.1, ztop=0.0, zh=0.12, vbc=np.array([0.0, -0.12, 0.0]))
    keep = T.SOLVE_DEADLINE
    try:
        T.SOLVE_DEADLINE = time.time() - 1.0
        _Run(fixed, ST.hist_data(Uh, Uh[-1:], 0.0, 0.1), lambda t: np.array([0.4, 0.2 - 0.12 * t, 0.0])).solve(
            np.array([0.9, 0.3, 0.8]), dict(P=np.repeat(q0[:, None], N_CTRL, 1), theta=(0.9, 0.3, 0.8)))
    except Exception:                                                        # noqa: BLE001 — 조립만 목적
        pass
    finally:
        T.SOLVE_DEADLINE = keep
    return time.time() - t0


class _Run:
    """fixed: q0, v0 (planner 규약), tgt, j6, ztop, zh, vbc; obj_at(t) → 다음 물체 흡착점 (사이클 시작 기준 t, 로봇 xy 와 같은 좌표)."""
    def __init__(self, fixed: dict, hist: dict, obj_at):
        self.S = solver(); self.hist = hist; self.obj_at = obj_at; o, p = self.S["opti"], self.S["par"]
        for k in ("q0", "v0", "tgt", "j6", "ztop", "zh", "vbc"):
            o.set_value(p[k], fixed[k])
        self.fixed = fixed; self.n_solve = 0; self.n_iter = 0

    def set_theta(self, th):
        t_c, t_s, t_k = (float(v) for v in th)
        c = _consts(); o, p = self.S["opti"], self.S["par"]
        Aw, Adw, hw, hdw = weights(t_c, t_s + c["OFFS"], self.hist)
        Ac, Acd, hc, hcd = weights(t_c, np.array([t_k]), self.hist)
        An, _, hn, _ = weights(t_c, t_k + SIGMAS, self.hist)
        Ae, Aed, he, hed = weights(t_c, np.array([t_c + c["D"]]), self.hist)
        Aa, ha = weights_acc(t_c, np.array([t_c + c["D"]]), self.hist)
        oc = np.asarray(self.obj_at(t_k), float)
        On = np.stack([np.asarray(self.obj_at(t_k + sg), float)[:2] for sg in SIGMAS], 1)
        for k, v in dict(t_c=t_c, Aw=Aw.T, Adw=Adw.T, hw=hw, hdw=hdw, Ac=Ac[0], Acd=Acd[0], hc=hc[:, 0], hcd=hcd[:, 0],
                         An=An.T, hn=hn, Ae=Ae[0], Aed=Aed[0], he=he[:, 0], hed=hed[:, 0], Aa=Aa[0], ha=ha[:, 0],
                         oe=np.asarray(self.obj_at(t_c + c["D"]), float)[:2],
                         tgtc=np.r_[oc[:2], self.fixed["ztop"]], On=On).items():
            o.set_value(p[k], v)

    def solve(self, th, init: dict) -> dict:
        S = self.S; o = S["opti"]; self.set_theta(th)
        if "x" in init:
            o.set_initial(o.x, init["x"]); o.set_initial(o.lam_g, init["lam"])
        else:
            o.set_initial(o.lam_g, np.zeros(o.ng))
            o.set_initial(S["Pf"], np.asarray(init["P"], float)[:, 1:])
            for tau in S["taus"]:
                o.set_initial(tau, 0.45)
        t0 = time.time()
        try:
            sol = o.solve(); status = "Solve_Succeeded"
        except RuntimeError:
            sol = o.debug; status = o.return_status()
        st = o.stats(); it = int(st.get("iter_count", 0)); self.n_solve += 1; self.n_iter += it
        ok = bool(st.get("success")) or status in ("Solve_Succeeded", "Solved_To_Acceptable_Level")
        return dict(ok=ok, status=str(status), J=float(sol.value(S["J"])), J_win=float(sol.value(S["J_win"])),
                    x=np.array(sol.value(o.x)).ravel(), lam=np.array(sol.value(o.lam_g)).ravel(), Pf=np.array(sol.value(S["Pf"])),
                    it=it, dt=time.time() - t0)

    def grad(self, th, r) -> np.ndarray:
        o = self.S["opti"]; g = []
        for d in np.eye(3) * GRAD_H:
            Ls = []
            for sg in (1, -1):
                self.set_theta(th + sg * d)
                Ls.append(float(self.S["L"](r["x"], o.value(o.p), r["lam"])))
            g.append((Ls[0] - Ls[1]) / (2 * GRAD_H))
        self.set_theta(th)
        return np.array(g)


def clamp(th):
    c = _consts(); RT, D = c["RT"], c["D"]
    t_c = float(np.clip(th[0], *TC_BOUNDS))
    t_s = float(np.clip(th[1], D + RT / 2, t_c))
    t_k = float(np.clip(th[2], t_s + RT / 2, t_c + D - 0.02))
    t_s = min(t_s, t_k - RT / 2)
    return np.array([t_c, t_s, t_k])


def run(fixed: dict, hist: dict, obj_at, inits, logger=None, max_outer: int = None) -> dict:
    """inits: [(tag, dict(P (6,N) planner 규약, theta=(t_c, t*, t_k)) 또는 dict(x, lam, theta) — DB 해 전체)]. max_outer: 바깥 BFGS 횟수
    (None = MAX_OUTER; 0 = θ 고정 안쪽 한 번 — DB x·λ warm start 면 30~50 반복). 리턴 dict(P, theta, J, J_win, tag, it, n_solve, dt, steps). 실패면 ValueError."""
    t0 = time.time(); R = _Run(fixed, hist, obj_at)
    r = None; tag = None; th = None; fails = []
    for tg, w in inits:
        th = clamp(np.asarray(w["theta"], float))
        rr = R.solve(th, w)
        if rr["ok"]:
            r, tag = rr, tg
            break
        fails.append(f"{tg}: {rr['status']} ({rr['it']} 반복)")
    if r is None:
        raise ValueError("사이클 NLP 실패 — " + "; ".join(fails))
    steps = [f"안쪽0 {r['it']}반복 {r['dt'] * 1e3:.0f}ms J {r['J']:.4f}"]
    g = R.grad(th, r); Hm = np.eye(3) * H0
    for k in range(1, (MAX_OUTER if max_outer is None else max_outer) + 1):
        step = -np.linalg.solve(Hm, g); n = np.linalg.norm(step)
        if n > STEP_MAX:
            step *= STEP_MAX / n
        acc = False; rn = None; th_n = th
        for _ls in range(N_LS):
            th_n = clamp(th + step)
            rn = R.solve(th_n, dict(x=r["x"], lam=r["lam"]))
            if rn["ok"] and rn["J"] < r["J"] - 1e-9:
                acc = True
                break
            step *= 0.5
        steps.append(f"바깥{k} {rn['it']}반복 J {rn['J']:.4f} {'채택' if acc else '거부'}")
        if not acc:
            break
        g_n = R.grad(th_n, rn); s_ = th_n - th; y = g_n - g
        if s_ @ y > 1e-12:
            Hs = Hm @ s_
            Hm = Hm - np.outer(Hs, Hs) / (s_ @ Hs) + np.outer(y, y) / (s_ @ y)
        dJ = r["J"] - rn["J"]; th, r, g = th_n, rn, g_n
        if np.linalg.norm(s_) < 3e-4 or dJ < 1e-6 * abs(r["J"]):
            break
    P = np.hstack([np.asarray(fixed["q0"], float)[:, None], r["Pf"]])
    out = dict(P=P, theta=th, J=r["J"], J_win=r["J_win"], tag=tag, it=R.n_iter, n_solve=R.n_solve, dt=time.time() - t0, steps=steps,
               x=r["x"], lam=r["lam"])                                       # 해 전체 (원·쌍대) — DB 에 두면 다음 풀이 warm start
    if logger is not None:
        logger.info(f"  사이클 NLP ({tag}): t_c {th[0]:.3f} t* {th[1]:.3f} t_k {th[2]:.3f} J {r['J']:.4f}, 안쪽 {R.n_solve}번 {R.n_iter}반복 "
                    f"{out['dt']:.1f}s — " + " | ".join(steps[-3:]))
    return out


def sample(P, t_c: float, dt: float = 0.002):
    """제어점 (planner 규약) → 격자 명령 (planner 규약) 0..⌈t_c⌉."""
    c = _consts(); n = int(np.ceil(t_c / dt - 1e-9))
    return c["B0"](np.minimum(np.arange(n + 1) * dt / t_c, 1.0)) @ np.asarray(P, float).T


def fit(Q_samples, t_c: float, q0, dt: float = 0.002) -> np.ndarray:
    """격자 명령 (planner 규약, 0..n) → 제어점 (6, N), 첫 점 = q0 고정 최소제곱."""
    c = _consts(); Q = np.asarray(Q_samples, float); n = len(Q) - 1
    B = c["B0"](np.minimum(np.arange(1, n + 1) * dt / t_c, 1.0))
    X = np.linalg.lstsq(B[:, 1:], Q[1:] - B[:, :1] * np.asarray(q0)[None], rcond=None)[0]
    return np.vstack([np.asarray(q0)[None], X]).T

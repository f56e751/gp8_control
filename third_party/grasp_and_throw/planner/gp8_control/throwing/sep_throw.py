"""분리 방식 던지기 (2026-10-30 사용자): 하강·상승은 잡기 LP (grasping.sep_pick) 로 후계산하고, 던지기 NLP 에 상승 끝 수직 속력 v_out
(물체 i) 과 다음 물체 하강 시작 수직 속력 v_in (물체 i+1) 을 변수로 둔다.

  목적  W_t·(T_상승(r_i, v_out) + t_f + T_하강(r_{i+1}, v_in)) + 릴리즈 창 평균 [W_e‖e‖² + W_ω‖ω‖²]   (실제 궤적 = 필터 통과)
  시작  물체 i 위 z_top 픽 자세, q̇ = J⁻¹(v_belt + ẑ v_out) = A + B v_out,  이력 = 1차 연장 (hist = (z_top − 윗면)/v_out)
  종착  물체 i+1 위 hover (t_ref 기준 벨트 이동, 정확식), 속도 v_belt − ẑ v_in, 툴 수직·J6, ω = 0
        정지 종점 판 (11-03 사용자, stop_end): 다음 물체가 아직 도달 범위 밖이면 대기점 hover 에 정지 (끝 위치 고정, 속도 0, v_in = 0) —
        매개변수 mv (1 = 벨트 따라 움직이는 종점, 0 = 정지) 하나로 같은 풀이기를 쓴다
  한계  관절 위치 = 실기 hull (상자 없음), 속도 hull, 가속 knot, 명령 툴 상자 (창 끝까지)
v_out 을 고정하려면 v_out_fixed (이미 잡아 올린 상태에서 출발하는 꼬리). 초기값 = 직전 해 (같은 타겟·가까운 시작) 또는 양끝 자유 warm DB
(GP8_THR_WARM_DB) 최근접 entry. 결과는 throw_planner.ThrowPlan (앱 던지기 세그먼트와 같은 꼴) + (v_out, v_in).
"""
import os
import time
from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import BSpline

from gp8_control.throwing.nlp import throw_nlp as T
from gp8_control.throwing.nlp import planner as _planner
from gp8_control.throwing.throw_planner import (ThrowPlan, ThrowBoundary, to_planner, to_robot, tcp_of, belt_joint_velocity,
                                                _predict_landing, PLANNER_SIGN, TRAVEL_Z)

GRID_DT = 0.002
UP = np.array([0.0, 0.0, 1.0])
CPU_S = float(os.environ.get("GP8_SEP_CPU_S", "60"))
WARM_OPTS: bool = os.environ.get("GP8_SEP_WARM_OPTS", "1") not in ("0", "", "false")
JIT: bool = os.environ.get("GP8_SEP_JIT", "0") not in ("0", "", "false")
ALT: bool = os.environ.get("GP8_SEP_ALT", "0") not in ("0", "", "false")
# 콜드 (10-30 사용자: 시간 가중치를 바꿔 "멈췄다가 콜드로 풀고 다시 시작"): DB·직전 해를 쓰지 않고 종전 NLP 의 고정 초기해
# (planner.INIT_VARIANTS 의 꺾은선: 시작 → 스윙 끝 → 종점) 로 모두 끝까지 풀어 목적함수가 가장 작은 해. 시뮬 정지 풀이 전용 (시간 무관).
COLD: bool = os.environ.get("GP8_SEP_COLD", "0") not in ("0", "", "false")     # 실제 이력 판을 번갈아 풀기로 (sep_alt, 2026-10-30)
FIXT: bool = os.environ.get("GP8_SEP_FIXT", "0") not in ("0", "", "false")   # 실험: t_f·t*·v_out 을 매개변수로 (안쪽 풀이 속도 확인)
X_MIN = float(os.environ.get("GP8_SEP_X_MIN", "0.24"))   # 명령 툴 상자 x 하한 (실제 궤적 게이트 x ≥ 0.22 보다 2 cm 안쪽 여유)
_SOLVER = {}

# 실제 이력 판 (2026-10-30 사용자: "이전 명령을 고려해서 … 명령을 바로 이어받는거야"): 잡기 LP 가 실제로 낸 명령을 t<0 이력으로 넣고,
# 상승 중 처짐 방지를 명령 하한(8d, LP 가정 이어짐) 대신 **실제 TCP 높이**에 직접 건다. v_out 은 잡기 LP 에 쓴 값으로 고정.
H_S = 0.36                                   # 이력 길이 [s] ≥ 필터 기억 D + W₁ + W₂ (0.32 s)
G_H = np.linspace(-H_S, 0.0, int(round(H_S / GRID_DT)) + 1)
TL_CHECK = np.arange(0.06, 0.341, 0.02)      # 실제 높이 하한을 확인하는 시각 (던지기 시작 기준) — 0.05 s 전 실제 높이는 던지기 명령과 무관
# NLP 안에서 거는 시각. 기본 = TL_CHECK 전부 (종전 15 곳). GP8_SEP_HIST_TL="0.06,0.10,0.14" 면 그 시각만 걸고 (2026-10-30 사용자: 접촉 근처 +
# 0.06–0.14 s 의 약 6 곳), 나머지는 풀이 뒤 TL_CHECK 전부에서 숫자로 다시 재어 lift["chk_margin"] 에 남긴다 (start:stop:step 또는 쉼표 목록).
def _parse_tl(s: str) -> np.ndarray:
    if ":" in s:
        a, b, c = (float(v) for v in s.split(":"))
        return np.arange(a, b, c)
    return np.array([float(v) for v in s.split(",") if v.strip()])
TL_LIFT = _parse_tl(os.environ.get("GP8_SEP_HIST_TL", "0.06:0.341:0.02"))
N_TC = 3                                     # 접촉 근처 추가 시각 (t_c − 8 ms, t_c, t_c + 8 ms)
T_INFL = 0.05                                # 이보다 이른 시각의 실제 높이는 던지기 명령이 (거의) 못 바꾼다 → 제약 풀어 둠
H_DEG = int(os.environ.get("GP8_SEP_HIST_DEG", "12"))   # 이력 적분의 다항식 차수 (s = τ/H_S ∈ [−1, 0])
Z_EPS = float(os.environ.get("GP8_SEP_HIST_EPS", "0.0005"))   # 실제 높이 하한·접촉 상한 여유 [m] (뒤 보정 QP 는 ±0.3 mm)


def _deadline_project(P, t_f: float):
    """마감에서 멈춘 반복값을 그대로 쓰되 한계 안으로 (throw_nlp._partial_iterate 와 같은 규칙 — 2026-09-30 사용자, 10-30 분리 방식에도:
    "마감에서 그때 값을 무조건 제출"). 시작 P₀·P₁ 은 식이 고정해 늘 이어진다. 나머지 제어점은 분리 NLP 의 관절 범위로 자르고, 속도·가속이
    한계를 넘으면 관절마다 가장 조금 옮겨 맞춘다 (끝은 자유, 가중치 10). 남은 제약 위반·착지 오차는 받아들인다 — 못 옮기면 ValueError."""
    from scipy.optimize import minimize
    P0 = np.asarray(P, float)
    if not (np.all(np.isfinite(P0)) and np.isfinite(t_f) and T.T_BOUNDS[0] - 1e-6 <= t_f <= T.T_BOUNDS[1] + 1e-6):
        raise ValueError(f"분리 던지기: 마감 반복값이 쓸 수 없는 값 (t_f {t_f})")
    lo = np.maximum(T.GP8_Q_MIN, T.Q_LO_REAL) + 1e-4; hi = np.minimum(T.GP8_Q_MAX, T.Q_HI_REAL) - 1e-4
    D, A = T._deriv_mats(); N = P0.shape[1]; free = list(range(2, N))
    Pc = P0.copy(); Pc[:, 2:] = np.clip(P0[:, 2:], lo[:, None], hi[:, None])

    def ratio(Q):
        return max(float(np.max(np.abs(D @ Q.T).T / (T.QD_LIM[:, None] * t_f))), float(np.max(np.abs(A @ Q.T).T / (T.QDD_LIM[:, None] * t_f ** 2))))
    r0 = ratio(Pc)
    if r0 > 1.0 and T.DEADLINE_FIX == "clamp":                              # 10-30 사용자: 투영 대신 clamp (앞에서부터 자르기, ~1 ms)
        Pc = T.clamp_limits(Pc, t_f, lo, hi, margin=1e-4)
    elif r0 > 1.0:
        for j in range(6):
            p0 = Pc[j].copy(); x0 = p0[free]; w = np.ones(len(free)); w[-2:] = 10.0
            vmax = float(T.QD_LIM[j]) * t_f * (1.0 - 1e-4); amax = float(T.QDD_LIM[j]) * t_f ** 2 * (1.0 - 1e-4)
            Df, Af = D[:, free], A[:, free]; bd = D @ p0 - Df @ x0; ba = A @ p0 - Af @ x0
            cons = [dict(type="ineq", fun=lambda x: vmax - (bd + Df @ x), jac=lambda x: -Df), dict(type="ineq", fun=lambda x: vmax + (bd + Df @ x), jac=lambda x: Df),
                    dict(type="ineq", fun=lambda x: amax - (ba + Af @ x), jac=lambda x: -Af), dict(type="ineq", fun=lambda x: amax + (ba + Af @ x), jac=lambda x: Af)]
            r = minimize(lambda x: float(np.sum(w * (x - x0) ** 2)), x0, jac=lambda x: 2.0 * w * (x - x0), bounds=[(lo[j], hi[j])] * len(free),
                         constraints=cons, method="SLSQP", options=dict(maxiter=300, ftol=1e-12))
            Pc[j, free] = np.clip(r.x, lo[j], hi[j])
    r1 = ratio(Pc)
    if r1 > 1.001:
        raise ValueError(f"분리 던지기: 마감 반복값을 속도·가속 한계 안으로 못 옮김 (한계 대비 {r1:.2f})")
    return Pc, dict(clamp_deg=float(np.rad2deg(np.max(np.abs(Pc - P0)))), lim_ratio_before=r0, lim_ratio_after=r1)


def _grid_ceil(t: float) -> float:
    return float(np.ceil(t / GRID_DT - 1e-9) * GRID_DT)


class PwHist:
    """2 ms 샘플 명령 (조각 선형) 의 정확한 1·2 차 적분. F1(0) = F2(0) = 0 (τ = 0 = 던지기 시작) — realized_qs 의 스플라인 쪽과 같은 원점."""

    def __init__(self, g: np.ndarray, U: np.ndarray):
        self.g = np.asarray(g, float); self.U = np.asarray(U, float); h = np.diff(self.g)[:, None]
        u0, u1 = self.U[:-1], self.U[1:]
        F1 = np.vstack([np.zeros((1, self.U.shape[1])), np.cumsum(h * (u0 + u1) / 2, axis=0)])
        F2 = np.vstack([np.zeros((1, self.U.shape[1])), np.cumsum(F1[:-1] * h + u0 * h ** 2 / 2 + (u1 - u0) * h ** 2 / 6, axis=0)])
        self.F1r, self.F2r = F1, F2
        self._c1, self._c2 = self._raw(0.0)[1:]                               # τ = 0 에서 0 이 되게

    def _raw(self, tau: float):
        g, U = self.g, self.U
        if tau <= g[0]:                                                       # 이력 앞 = 첫 샘플에 정지
            s = tau - g[0]
            return U[0], self.F1r[0] + U[0] * s, self.F2r[0] + self.F1r[0] * s + U[0] * s ** 2 / 2
        k = min(int((tau - g[0]) / (g[1] - g[0])), len(g) - 2)
        h = g[k + 1] - g[k]; s = min(tau - g[k], h) if tau <= g[-1] else tau - g[k]
        du = (U[k + 1] - U[k]) / h
        return (U[k] + du * s, self.F1r[k] + U[k] * s + du * s ** 2 / 2,
                self.F2r[k] + self.F1r[k] * s + U[k] * s ** 2 / 2 + du * s ** 3 / 6)

    def __call__(self, tau: float):
        u, f1, f2 = self._raw(float(tau))
        return u, f1 - self._c1, f2 - self._c2 - self._c1 * float(tau)


def realized_numeric(H: "PwHist", t: float):
    """PwHist (τ 전 구간 명령) 의 필터 통과 실제 상태 (q, q̇) — realized_qs 와 같은 연속 모델 (상자 둘)."""
    D, (W1, W2) = T.FILTER
    F = lambda tau: H(tau)
    q = (F(t - D)[2] - F(t - D - W1)[2] - F(t - D - W2)[2] + F(t - D - W1 - W2)[2]) / (W1 * W2)
    qd = (F(t - D)[1] - F(t - D - W1)[1] - F(t - D - W2)[1] + F(t - D - W1 - W2)[1]) / (W1 * W2)
    return q, qd


REAL_END = os.environ.get("GP8_SEP_REAL_END", "0") not in ("0", "", "false")   # 11-04 사용자: 끝 xy 위치·속도를 실제 궤적에 (논문 식 10c)
# 11-04 사용자 (version2): 상승 하한 (잡기 LP 의 0.05 s 상승 가정 — "휴리스틱") 과 툴 박스를 NLP 에서 뺀다 → 둘 다 gate (DB 에서 미리 거름)
V2 = os.environ.get("GP8_SEP_V2", "0") not in ("0", "", "false")
# 11-04 사용자 (논문 version2 식 9): 실제 시간축 — 끝 조건 12 개 모두 실제 t_f + D 에, P_1 자유 (이음 가속 부등식), 방출 창 ≥ D + ΔT/2, v_in ≤ v̄ 해석식
RT = os.environ.get("GP8_SEP_RT", "0") not in ("0", "", "false")
_VBAR = {}


def _build(mode: str = "ramp"):
    """mode "ramp": 종전 (이력 = v_out 직선 근사, v_out 변수 가능, 8d 명령 하한) — DB 빌더·대체 경로.
    mode "hist": 실제 이력 판 (이력 = 매개변수 데이터, v_out 매개변수, 실제 TCP 높이 하한·접촉 상한)."""
    import casadi as cs
    data = mode == "hist"
    N = T.N_CTRL; u_head = T.KNOTS[T.DEGREE + 1]; u_tail = 1.0 - T.KNOTS[len(T.KNOTS) - T.DEGREE - 2]
    HW_LO = np.maximum(T.GP8_Q_MIN, T.Q_LO_REAL); HW_HI = np.minimum(T.GP8_Q_MAX, T.Q_HI_REAL)
    opti = cs.Opti()
    Pf = opti.variable(6, N - 2); v_in = opti.variable()
    if FIXT:                                                                 # 안쪽 풀이: 시간 변수 고정 (매개변수) — 실제 상태가 제어점의 선형식
        t_f = opti.parameter(); t_s = opti.parameter(); v_out = opti.parameter()
    else:
        t_f = opti.variable(); t_s = opti.variable(); v_out = opti.parameter() if data else opti.variable()
    par = {k: opti.parameter(n) for k, n in dict(q0=6, A0=6, B0=6, endp=3, vb=3, tref=1, j6=1, tgt=3, zgap=1, cl=2, cd=2,
                                                vlo=1, vhi=1, vmi=1, vmi_s=1, rref=1, hfix=1, hvar=1, ztop=1, tflo=1, tfhi=1, mv=1).items()}
    opti.set_value(par["tflo"], T.T_BOUNDS[0]); opti.set_value(par["tfhi"], T.T_BOUNDS[1])
    hist_fn = None
    if data:
        for k in ("hU", "hF1", "hF2"):                                       # 이력 (u, ∫u, ∫∫u) 의 다항식 계수 — 매개변수 B-spline 보간기는
            par[k] = opti.parameter(6, H_DEG + 1)                             # 반복당 ~10 s 였다 (10-30 실측)
        par["zlo"] = opti.parameter(len(TL_LIFT) + N_TC); par["zhi"] = opti.parameter(); par["tc"] = opti.parameter(N_TC)

        def _poly(C, s):
            v = C[:, H_DEG]
            for k in range(H_DEG - 1, -1, -1):
                v = v * s + C[:, k]
            return v

        def hist_fn(tau):
            s = cs.fmin(cs.fmax(tau, -H_S), 0.0) / H_S
            return _poly(par["hU"], s), _poly(par["hF1"], s), _poly(par["hF2"], s)
    v0 = par["A0"] + par["B0"] * v_out
    P = cs.horzcat(par["q0"], par["q0"] + (u_head / T.DEGREE) * t_f * v0, Pf)
    hist_s = par["hfix"] + par["hvar"] * par["zgap"] / cs.fmax(v_out, 0.05)    # 이력 길이: 변수 모드 = (z_top − 윗면)/v_out, 고정 모드 = 상수
    for k in range(2, N):
        opti.subject_to(opti.bounded(HW_LO, P[:, k], HW_HI))
    for k in range(N - 1):
        span = T.KNOTS[k + T.DEGREE + 1] - T.KNOTS[k + 1]
        opti.subject_to(opti.bounded(-T.QD_LIM * t_f, T.DEGREE * (P[:, k + 1] - P[:, k]) / span, T.QD_LIM * t_f))
    for u_k in np.unique(T.KNOTS):
        opti.subject_to(opti.bounded(-T.QDD_LIM, T.spline_qs(P, float(u_k), t_f)[2], T.QDD_LIM))
    F_p, F_a, F_pd, F_ad = T.end_kinematics_functions()
    q_f = P[:, N - 1]; qd_f = T.DEGREE * (P[:, N - 1] - P[:, N - 2]) / (u_tail * t_f)
    p_tgt = par["endp"] + par["mv"] * par["vb"] * (t_f - par["tref"])
    if REAL_END:                                                             # 움직이는 끝: 실제 xy 위치·속도 = 물체, 명령은 높이·하강 속력
        q_rf, qd_rf = T.realized_qs(P, t_f, t_f, par["q0"], v0, hist_s, hist_fn=hist_fn)
        p_r = F_p(q_rf); v_r = F_pd(q_rf, qd_rf); p_c = F_p(q_f); v_c = F_pd(q_f, qd_f); mv = par["mv"]
        opti.subject_to(mv * (p_r[0:2] - p_tgt[0:2]) + (1 - mv) * (p_c[0:2] - p_tgt[0:2]) == 0)
        opti.subject_to(p_c[2] - p_tgt[2] == 0)
        opti.subject_to(mv * (v_r[0:2] - par["vb"][0:2]) + (1 - mv) * v_c[0:2] == 0)
        opti.subject_to(v_c[2] + v_in == 0)
    else:
        opti.subject_to(F_p(q_f) == p_tgt)                                   # mv 0: 대기점 고정 (정지 종점)
    a_f = F_a(q_f); ad_f = F_ad(q_f, qd_f)
    opti.subject_to(a_f[0] == 0); opti.subject_to(a_f[1] == 0); opti.subject_to(a_f[2] <= -0.5); opti.subject_to(q_f[5] == par["j6"])
    if not REAL_END:
        opti.subject_to(F_pd(q_f, qd_f) == par["mv"] * par["vb"] - cs.DM(UP) * v_in)
    opti.subject_to(ad_f[0] == 0); opti.subject_to(ad_f[1] == 0); opti.subject_to(qd_f[5] == 0)
    if V2:
        pass
    elif data:
        # 실제 이력 판: 필터를 지난 실제 TCP 높이가 윗면 아래로 안 들어가고 (하한 zlo), 접촉 시각 t_c 에 실제로 닿아 있다 (상한 zhi).
        # 하한·상한은 plan() 이 잡기 LP 의 가정 이어짐(실제 FK)으로 계산 — 그 이어짐이 늘 가능한 해이므로 문제가 비지 않는다.
        t_pts = [float(t) for t in TL_LIFT] + [par["tc"][j] for j in range(N_TC)]
        for i, tk in enumerate(t_pts):
            zr = T.fk_tcp_sym(T.realized_qs(P, t_f, tk, par["q0"], v0, None, hist_fn=hist_fn)[0])[2]
            opti.subject_to(zr >= par["zlo"][i])
            if i == len(TL_LIFT) + N_TC // 2:
                opti.subject_to(zr <= par["zhi"])
    else:
        # 상승 중 처짐 방지: 던지기 명령의 TCP z 가 처음 0.34 s 동안 잡기 LP 가 가정한 이어짐(z_top + v_out·min(t, 0.05 s)) 이상.
        # 실제 궤적은 명령의 양의 가중 평균이라, 명령이 LP 가정 이상이면 실제 z 도 LP 가 보장한 윗면 이상이다 (10-30 DB 1 차: 5 % 가 처짐).
        for tk in np.arange(0.02, 0.341, 0.02):
            zk = T.fk_tcp_sym(T.spline_qs(P, cs.fmin(tk / t_f, 1.0), t_f)[0])[2]
            opti.subject_to(zk >= par["ztop"] + v_out * min(float(tk), 0.05))
    rt = T.RELEASE_TIME
    u_re = (t_s + rt / 2) / t_f + 0.08
    for u_c in (np.linspace(0, 1, 26)[1:-1] if not V2 else []):
        tcp = T.fk_col_points_sym(T.spline_qs(P, float(u_c), t_f)[0])[-1]
        slack = (1.0 - 1.0 / (1.0 + cs.exp(40.0 * (float(u_c) - u_re)))) * 10.0
        opti.subject_to(tcp[0] >= max(T.CART_MIN_X + T.CART_PLAN_MARGIN, X_MIN) - slack)
        opti.subject_to(tcp[2] >= T.CART_MIN_Z + T.CART_PLAN_MARGIN - slack)
    if not FIXT:
        opti.subject_to(t_s >= rt / 2); opti.subject_to(t_s <= t_f - rt / 2); opti.subject_to(opti.bounded(par["tflo"], t_f, par["tfhi"]))
        # tflo/tfhi (10-31 사용자 "접근 불가능한 곳을 왜 가"): 끝 hover 와 거기서 잡은 뒤 던지기 시작 자세가 관절 범위 안인 던지기 시간만
        if not data:
            opti.subject_to(opti.bounded(par["vlo"], v_out, par["vhi"]))
    # 다음 물체 하강 시작 속력 ≤ 그 종점 반경에서 잡기 LP 가 풀리는 상한 (sep_vlim 표, 감속 거리·실제 z ≥ 윗면까지 고려). 종점은 t_f 에 따라
    # 벨트를 따라 움직이므로 반경도 변수 — 표를 기준 반경 rref 둘레 1차식으로.
    p_end = par["endp"] + par["mv"] * par["vb"] * (t_f - par["tref"]); r_end = cs.sqrt(p_end[0] ** 2 + p_end[1] ** 2)
    opti.subject_to(v_in >= 0.0); opti.subject_to(v_in <= par["vmi"] + par["vmi_s"] * (r_end - par["rref"]))
    F_err, F_z, _ = T.make_objective_functions_sym(); dragc = cs.DM(np.zeros(3))
    offs = np.linspace(-rt / 2, rt / 2, T.N_WIN); w_tr = np.full(T.N_WIN, rt / (T.N_WIN - 1)); w_tr[[0, -1]] *= 0.5
    J_win = 0; taus = []
    for off, w in zip(offs, w_tr):
        q_i, qd_i = T.realized_qs(P, t_f, t_s + off, par["q0"], v0, hist_s, hist_fn=hist_fn)
        tau = opti.variable(); taus.append(tau)
        opti.subject_to(opti.bounded(0.05, tau, 1.5)); opti.subject_to(F_z(q_i, qd_i, tau, par["tgt"], dragc) == 0)
        J_win = J_win + w * (T.W_ACC * F_err(q_i, qd_i, tau, par["tgt"], dragc) + T.W_OMEGA_LADDER[0] * cs.sumsqr(T.fk_omega_sym(q_i, qd_i)))
    J_time = t_f + par["cl"][0] * v_out + par["cl"][1] * v_out ** 2 + par["cd"][0] * v_in + par["cd"][1] * v_in ** 2
    opti.minimize(T.W1 * J_time + T.W2 * J_win)
    ip = {"max_iter": 3000, "print_level": 0, "sb": "yes", "tol": 1e-6, "acceptable_tol": 1e-4, "acceptable_iter": 10, "max_cpu_time": CPU_S}
    if os.environ.get("GP8_SEP_HESS", "exact") != "exact":                  # 실험: 헤시안 근사 (반복당 시간의 62 % 가 정확 헤시안)
        ip.update({"hessian_approximation": "limited-memory", "limited_memory_max_history": int(os.environ.get("GP8_SEP_LBFGS_M", "20"))})
    if WARM_OPTS:                                                            # 이웃 해 재시동 (throw_nlp.POLISH_IPOPT_OPTS 와 같은 규칙: 적응 장벽 + warm start)
        ip.update({"mu_strategy": "adaptive", "warm_start_init_point": "yes", "warm_start_bound_push": 1e-9,
                   "warm_start_mult_bound_push": 1e-9, "warm_start_slack_bound_push": 1e-9})
    # 마감 (앱 꼬리: planner.set_deadline → throw_nlp.SOLVE_DEADLINE): 지나면 IPOPT 를 그 반복에서 멈춘다 (기존 polish 와 같은 콜백)
    cb = T._DeadlineStop(opti.nx, opti.ng)
    so = {"print_time": False, "expand": True, "iteration_callback": cb}
    if JIT:                                                                  # 목적·제약·미분·헤시안을 C 로 컴파일 (반복당 시간의 62 % 가 헤시안)
        so.update({"jit": True, "compiler": "shell", "jit_temp_suffix": False, "jit_name": f"sep_nlp_{mode}_{os.getpid()}",
                   "jit_options": {"flags": [os.environ.get("GP8_SEP_JIT_OPT", "-O1")], "verbose": False, "compiler": "gcc"}})
    opti.solver("ipopt", so, ip)
    return dict(cb=cb, opti=opti, Pf=Pf, t_f=t_f, t_s=t_s, v_out=v_out, v_in=v_in, taus=taus, par=par, P=P, J_win=J_win, J_time=J_time,
                mode=mode)


def solver(mode: str = "ramp"):
    if mode not in _SOLVER:
        _SOLVER[mode] = _build(mode)
    return _SOLVER[mode]


def prewarm(mode: str = "ramp") -> float:
    """조립 + IPOPT 풀이기 생성 비용 (첫 solve 때 expand·함수 생성, 10-30 실시간 시뮬에서 6.4 s) 을 시작 때 치른다:
    그럴듯한 매개변수로 마감을 이미 지난 시각에 걸어 첫 반복에서 멈추는 가짜 풀이 한 번 (sep_alt.prewarm 과 같은 방식)."""
    t0 = time.time(); S = solver(mode); opti, par = S["opti"], S["par"]
    q0 = np.array([0.6, 0.5, -0.1, 0.0, 1.0, 1.1])
    val = dict(q0=q0, endp=np.array([0.45, 0.3, 0.1]), vb=np.array([0.0, -0.12, 0.0]), tref=0.6, j6=1.1, tgt=np.array([1.35, 0.0, -0.06]),
               zgap=0.1, vlo=0.3, vhi=3.0, vmi=1.0, rref=0.5, hfix=0.05, ztop=0.12, zhi=10.0, tc=np.array([0.07, 0.078, 0.086]),
               tflo=T.T_BOUNDS[0], tfhi=T.T_BOUNDS[1], mv=1.0)
    for k, p in par.items():
        v = val.get(k)
        if v is None:
            v = np.full(p.shape, -10.0) if k == "zlo" else np.zeros(p.shape)
        opti.set_value(p, v)
    keep = T.SOLVE_DEADLINE
    try:
        opti.set_initial(S["Pf"], np.repeat(q0[:, None], S["Pf"].shape[1], 1))
        if not FIXT:
            opti.set_initial(S["t_f"], 0.65); opti.set_initial(S["t_s"], 0.45)
        T.SOLVE_DEADLINE = time.time() - 1.0
        opti.solve()
    except Exception:                                                        # noqa: BLE001 — 생성 비용만 목적 (멈춤·실패는 무시)
        pass
    finally:
        T.SOLVE_DEADLINE = keep
    return time.time() - t0


def hist_data(U_hist, U_cont, z_obj: float, t_c: float, logger=None) -> dict:
    """실제 이력 판의 입력. U_hist: 던지기 시작 전 명령 (로봇 규약, 2 ms, 마지막 = 던지기 시작 자세) — 앞쪽이 H_S 보다 짧으면 첫 샘플로 채운다.
    U_cont: 잡기 LP 가 가정한 이어짐 (로봇 규약, 2 ms, 첫 샘플 = 던지기 시작 자세). t_c: 실제 접촉 시각 (던지기 시작 기준 [s]).
    하한·상한 = 가정 이어짐의 실제 TCP 높이 (정확 FK) 와 윗면 중 낮은/높은 쪽 ∓ Z_EPS — 그 이어짐이 늘 제약을 만족한다."""
    U_hist = np.asarray(U_hist, float)[:, :6]; nh = len(G_H)
    if len(U_hist) < nh:
        U_hist = np.vstack([np.repeat(U_hist[:1], nh - len(U_hist), 0), U_hist])
    U_hist = U_hist[-nh:]
    U_cont = np.asarray(U_cont, float)[:, :6]
    g_all = np.concatenate([G_H, np.arange(1, len(U_cont)) * GRID_DT])
    H_all = PwHist(g_all, np.vstack([U_hist, U_cont[1:]]))
    t_tc = np.clip(t_c + np.array([-0.008, 0.0, 0.008]), 0.0, None)
    t_pts = np.concatenate([TL_LIFT, t_tc])
    z_cont = np.array([tcp_of(realized_numeric(H_all, float(t))[0])[2] for t in t_pts])
    zlo = np.where(t_pts >= T_INFL, np.minimum(z_obj, z_cont) - Z_EPS, -10.0)
    t_chk = np.concatenate([TL_CHECK, t_tc])                                  # 풀이 뒤 확인용 (NLP 에는 안 건다) — 같은 규칙의 하한
    z_cont_chk = np.array([tcp_of(realized_numeric(H_all, float(t))[0])[2] for t in t_chk])
    zlo_chk = np.where(t_chk >= T_INFL, np.minimum(z_obj, z_cont_chk) - Z_EPS, -10.0)
    zhi = max(z_obj, float(z_cont[len(TL_LIFT) + N_TC // 2])) + Z_EPS if t_c >= T_INFL else 10.0
    Up = U_hist * PLANNER_SIGN                                               # planner 규약 (부호만 — 적분도 같은 부호)
    Hp = PwHist(G_H, Up)
    g_f = np.linspace(-H_S, 0.0, 4 * len(G_H) - 3)                           # 다항식 맞춤 (샘플 사이 값까지)
    V = np.vander(g_f / H_S, H_DEG + 1, increasing=True)
    vals = [np.array([Hp(t)[i] for t in g_f]) for i in range(3)]
    coef = [np.linalg.lstsq(V, y, rcond=None)[0].T for y in vals]            # (6, H_DEG+1) 씩
    fit_err = [float(np.abs(V @ c.T - y).max()) for c, y in zip(coef, vals)]
    return dict(hU=coef[0], hF1=coef[1], hF2=coef[2], zlo=zlo, zhi=zhi, tc=t_pts[len(TL_LIFT):], fit_err=fit_err,
                H=Hp, z_cont=z_cont, t_pts=t_pts, z_obj=float(z_obj), t_c=float(t_c), t_chk=t_chk, zlo_chk=zlo_chk)


def realized_eval_hist(Pv, t_f: float, Hp: PwHist):
    """수치 해의 실제 궤적 (planner 규약) — τ ≥ 0 은 스플라인 부정적분 (T.realized_eval 과 같은 식), τ < 0 은 실제 이력."""
    D, (W1, W2) = T.FILTER
    spl = [BSpline(T.KNOTS, np.asarray(Pv[j], float), T.DEGREE) for j in range(6)]
    a1 = [s_.antiderivative(1) for s_ in spl]; a2 = [s_.antiderivative(2) for s_ in spl]

    def F(tau):
        if tau >= 0:
            s = min(max(tau / t_f, 0.0), 1.0)
            return None, t_f * np.array([a(s) for a in a1]), t_f ** 2 * np.array([a(s) for a in a2])
        return Hp(tau)
    return (lambda t: (F(t - D)[2] - F(t - D - W1)[2] - F(t - D - W2)[2] + F(t - D - W1 - W2)[2]) / (W1 * W2),
            lambda t: (F(t - D)[1] - F(t - D - W1)[1] - F(t - D - W2)[1] + F(t - D - W1 - W2)[1]) / (W1 * W2))


def warm_entry(p_start_xy, p_end_xy, p_target, logger=None) -> Optional[dict]:
    """양끝 자유 warm DB 에서 같은 타겟, 시작+종점 xy 거리 최소인 exact entry."""
    db = _planner.load_warm_db(logger)
    if not db:
        return None
    A = _planner._warm_db_arrays(db)
    d_t = np.linalg.norm(A["target_xy"] - np.asarray(p_target, float)[:2], axis=1)
    idx = np.flatnonzero(d_t < d_t.min() + 1e-6)
    ok = A["end_mode"][idx] == "exact"
    if ok.any():
        idx = idx[ok]
    pn = np.nan_to_num(A["p_next_xy"][idx], nan=1e3)
    d = np.linalg.norm(A["p_start_xy"][idx] - np.asarray(p_start_xy, float)[:2], axis=1) + np.linalg.norm(pn - np.asarray(p_end_xy, float)[:2], axis=1)
    return db[int(idx[int(np.argmin(d))])]


_SEP_DB = {}


def sep_entry(p_start_xy, p_end_xy, p_target) -> Optional[dict]:
    """분리 방식 DB (GP8_SEP_DB — sep_db 빌더의 게이트 통과 해 목록) 에서 같은 타겟, 시작+종점 xy 거리 최소인 해. 없으면 None."""
    path = os.environ.get("GP8_SEP_DB", "")
    if not path or not os.path.exists(path):
        return None
    if path not in _SEP_DB:
        import pickle
        with open(path, "rb") as f:
            es = pickle.load(f)
        _SEP_DB[path] = (es, np.array([e["target"][:2] for e in es], float), np.array([e["p_start"][:2] for e in es], float),
                         np.array([e["p_next"][:2] for e in es], float))
    es, tg, ps, pn = _SEP_DB[path]
    d_t = np.linalg.norm(tg - np.asarray(p_target, float)[:2], axis=1)
    idx = np.flatnonzero(d_t < d_t.min() + 1e-6)
    d = np.linalg.norm(ps[idx] - np.asarray(p_start_xy, float)[:2], axis=1) + np.linalg.norm(pn[idx] - np.asarray(p_end_xy, float)[:2], axis=1)
    return es[int(idx[int(np.argmin(d))])]


def _sample(P: np.ndarray, t_f: float):
    sp = [BSpline(T.KNOTS, P[j], T.DEGREE) for j in range(6)]
    d1 = [s.derivative(1) for s in sp]; d2 = [s.derivative(2) for s in sp]
    q_of = lambda t: np.array([s(min(max(t / t_f, 0.0), 1.0)) for s in sp])
    qd_of = lambda t: np.array([s(min(max(t / t_f, 0.0), 1.0)) for s in d1]) / t_f
    qdd_of = lambda t: np.array([s(min(max(t / t_f, 0.0), 1.0)) for s in d2]) / t_f ** 2
    return q_of, qd_of, qdd_of


def plan(start_q, p_target, end: ThrowBoundary, v_belt, z_obj: float, r_i: float, r_n: float, table, *,
         v_out_fixed: Optional[float] = None, start_qd=None, warm: Optional[dict] = None, logger=None, vel_basis=None,
         hist: Optional[dict] = None, tf_win: Optional[Tuple[float, float]] = None, z_n: Optional[float] = None,
         stop_end: bool = False):
    """(ThrowPlan, v_out, v_in, raw). 실패면 ValueError. start_q: 로봇 규약 (물체 i 위 z_top 픽 자세).
    tf_win: 던지기 시간 범위 (하한, 상한) — 끝 hover·잡은 뒤 자세가 관절 범위 안인 시간 (없으면 T_BOUNDS).
    v_out_fixed: 이미 정해진 시작 상태에서 출발 (start_qd = 그 관절 속도, 로봇 규약 — 정지 출발이면 0). 시작 속도를 그대로 쓰고 상승 시간 항은 상수.
    hist: hist_data(…) — 실제 이력 판 (v_out_fixed·start_qd 필요). 이력 = 잡기 LP 명령, 실제 TCP 높이 하한·접촉 상한.
    stop_end: 정지 종점 — end.p_tcp (대기점 hover TCP) 에 속도 0 으로 끝 (v_in = 0, 다음 물체 하강 시간 항 없음, r_n·z_n 안 씀)."""
    if hist is not None and (v_out_fixed is None or start_qd is None):
        raise ValueError("실제 이력 판은 v_out_fixed·start_qd 가 필요")
    use_alt = hist is not None and ALT
    if not use_alt:
        S = solver("hist" if hist is not None else "ramp"); opti, par = S["opti"], S["par"]
    start_q = np.asarray(start_q, float)[:6]
    q0 = to_planner(start_q)
    if vel_basis is not None:                                                # 잡기 IK 경로(J6 고정)의 미분 — 잡기 LP 끝 속도와 정확히 같다
        A0, B0 = to_planner(vel_basis[0]), to_planner(vel_basis[1])
    else:
        A0 = to_planner(belt_joint_velocity(start_q, v_belt)); B0 = to_planner(belt_joint_velocity(start_q, UP)) - to_planner(belt_joint_velocity(start_q, np.zeros(3)))
    ri = table.row(r_i, z_obj)                                                  # 잡기 표는 반경 × 물체 윗면 높이 (z_n: 다음 물체, 없으면 0.02)
    rn = None if stop_end else table.row(r_n, z_n)
    if ri is None or (rn is None and not stop_end):
        raise ValueError(f"잡기 표에 반지름이 없음 ({r_i:.3f} / {r_n:.3f})")
    li = table.vout_lim(r_i, z_obj)                                             # 잡기 LP 가 풀리는 끝 속력 상한 (표에 없으면 vmax − 0.10)
    ln = (0.0, 0.0) if stop_end else table.vin_lim(r_n, z_n)                    # 정지 종점: v_in ∈ [0, 0]
    vmo, (vmi, vmi_s) = li[0], ln
    if V2 and not stop_end:                                                     # 11-04 사용자 "상한 없애": v_in ≤ v̄ 없음 (식 수는 그대로, 상한만 크게)
        vmi, vmi_s = 10.0, 0.0
    if RT and not stop_end:                                                     # 11-04 사용자 "상한 넣고 analytic하게": v̄ = Δ⁻¹(h − z) (논문 식 12)
        from gp8_control.grasping import sep_pick as SP
        key = (round(float(r_n), 2), round(float(z_n if z_n is not None else 0.02), 3))
        if key not in _VBAR:                                                    # u = 그 반경 수직선의 최대 상승 속력 (1 cm·1 mm 칸 캐시)
            u_up = float(SP.line_bounds(key[0], z_lo=-0.18)[0](SP.Z_H))
            _VBAR[key] = SP.vbar(SP.Z_H - key[1], u_up)
        vmi, vmi_s = _VBAR[key], 0.0
    hfix, hvar = 0.0, 1.0
    if v_out_fixed is not None:
        if start_qd is not None:                                              # 시작 관절 속도 그대로 (벨트 추종이 아닐 수도 — 정지 출발)
            A0 = to_planner(np.asarray(start_qd, float)[:6]); B0 = np.zeros(6)
        else:
            A0 = A0 + B0 * float(v_out_fixed); B0 = np.zeros(6)
        vlo = vhi = 0.0
        hfix, hvar = (TRAVEL_Z - float(z_obj)) / max(float(v_out_fixed), 0.05), 0.0
    else:
        vlo, vhi = 0.3, vmo
    t_ref_end = float(end.t) if end.t is not None else 0.0
    vals = dict(q0=q0, A0=A0, B0=B0, endp=np.asarray(end.p_tcp, float), vb=np.asarray(v_belt, float), tref=t_ref_end,
                j6=float(end.j6) * float(PLANNER_SIGN[5]), tgt=np.asarray(p_target, float), zgap=TRAVEL_Z - float(z_obj),
                cl=ri[1][3:5], cd=np.zeros(2) if stop_end else rn[1][1:3], vlo=vlo, vhi=vhi, vmi=vmi, vmi_s=vmi_s, rref=float(r_n), hfix=hfix,
                hvar=hvar, ztop=float(tcp_of(start_q)[2]), tflo=float(T.T_BOUNDS[0]), tfhi=float(T.T_BOUNDS[1]), mv=0.0 if stop_end else 1.0)
    if tf_win is not None:
        vals["tflo"] = float(max(T.T_BOUNDS[0], tf_win[0])); vals["tfhi"] = float(min(T.T_BOUNDS[1], max(tf_win[1], vals["tflo"])))
    if hist is not None:
        vals.update({k: hist[k] for k in ("hU", "hF1", "hF2", "zlo", "zhi", "tc")})
    if not use_alt:
        for k, v in vals.items():
            opti.set_value(par[k], v)
    inits = []
    if warm is not None and warm.get("P") is not None:
        inits.append(("warm-prev", warm))
    se = sep_entry(tcp_of(start_q)[:2], np.asarray(end.p_tcp, float)[:2], p_target)
    if se is not None:
        inits.append(("sepdb", dict(P=np.asarray(se["P"], float), t_f=float(se["t_f"]), t_star=float(se["t_star"]), v_out=float(se["v_out"]),
                                    v_in=float(se["v_in"]))))
    ent = warm_entry(tcp_of(start_q)[:2], np.asarray(end.p_tcp, float)[:2], p_target, logger)
    if ent is not None:
        inits.append(("db", dict(P=np.asarray(ent["P"], float), t_f=float(ent["t_f"]), t_star=float(ent["t_star"]),
                                 v_out=float(ent.get("lift_speed", 2.0)), v_in=0.5)))
    if COLD:
        from gp8_control.throwing.nlp.planner import INIT_VARIANTS
        qe = to_planner(np.asarray(end.q, float)[:6]); rt0 = T.RELEASE_TIME
        inits = []
        for k_, v_ in enumerate([v for v in INIT_VARIANTS if v is not None]):
            tf0 = float(np.clip(v_["T0"], *T.T_BOUNDS))
            inits.append((f"cold{k_}", dict(P=T.init_ctrl_points(q0, np.asarray(v_["dq_swing"], float), float(v_["chi0"]), p_f=qe),
                                            t_f=tf0, t_star=float(np.clip(v_["chi0"] * tf0, rt0 / 2, tf0 - rt0 / 2)), v_out=2.0, v_in=0.5)))
    if not inits:
        raise ValueError("분리 던지기: 초기값 없음 (warm DB 없음)")
    if use_alt:                                                              # 번갈아 풀기 (θ 고정 안쪽 + 바깥 BFGS, 마감 대응)
        from gp8_control.throwing import sep_alt as SA
        fixed = dict(q0=q0, v0=A0, **{k: vals[k] for k in ("endp", "vb", "tref", "j6", "tgt", "cd", "vmi", "vmi_s", "rref", "zlo", "zhi",
                                                          "tflo", "tfhi", "mv")})
        if COLD:
            ra = None; errs = []
            for ini in inits:
                try:
                    r_ = SA.run(fixed, hist, [ini], logger=None)
                except ValueError as e_:
                    errs.append(str(e_)); continue
                if ra is None or r_["J"] < ra["J"]:
                    ra = r_
            if ra is None:
                raise ValueError("분리 던지기 alt 콜드 전부 실패 — " + "; ".join(errs)[:300])
            if logger is not None:
                logger.info(f"  분리 alt 콜드: {len(inits)}개 중 최선 {ra['tag']} J {ra['J']:.4f}")
        else:
            ra = SA.run(fixed, hist, inits, logger=logger)
        Pv, t_f, t_star, v_in = ra["P"], ra["t_f"], ra["t_s"], ra["v_in"]; v_out = float(v_out_fixed); v0_sol = A0
        tag, it, dt, partial, fails = ra["tag"], ra["it"], ra["dt"], ra["partial"], ra["fails"]
        J_val, Jwin_val, mode = ra["J"], ra["J_win"], "alt"
    else:
        best = None; fails = []; partial = False; cold_ok = []

        def _set_init(w):
            opti.set_initial(S["Pf"], np.asarray(w["P"], float)[:, 2:])
            _set = opti.set_value if FIXT else opti.set_initial                   # 시간 고정 판: 이웃 해의 t_f·t*·v_out 을 그대로 매개변수로
            _set(S["t_f"], float(w["t_f"])); _set(S["t_s"], float(w["t_star"]))
            (opti.set_value if (FIXT or hist is not None) else opti.set_initial)(
                S["v_out"], 0.0 if v_out_fixed is not None else float(np.clip(w.get("v_out", 2.0), vlo, vhi)))
            opti.set_initial(S["v_in"], float(np.clip(w.get("v_in", 0.5), 0.0, vmi)))
            for tau in S["taus"]:
                opti.set_initial(tau, 0.45)
        for tag, w in inits:
            if best is None and fails and T.SOLVE_DEADLINE is not None and time.time() > T.SOLVE_DEADLINE:
                break                                                            # 마감이 지났으면 다음 초기값은 풀지 않는다
            _set_init(w)
            t0 = time.time()
            try:
                sol = opti.solve(); status = "ok"
            except RuntimeError:
                sol = opti.debug; status = opti.return_status()
            it = opti.stats().get("iter_count")
            if status in ("ok", "Solve_Succeeded", "Solved_To_Acceptable_Level"):
                if COLD:                                                     # 콜드: 전부 풀어 가장 작은 J 를 고른다
                    cold_ok.append((float(sol.value(opti.f)), tag, w)); continue
                best = (sol, tag, it, time.time() - t0); break
            if "User_Requested_Stop" in str(status) and T.ACCEPT_PARTIAL:
                # 마감에서 멈춘 반복값 (2026-09-30 사용자 규칙, 10-30 분리 방식에도): 제약 위반·착지와 무관하게 그때 값으로 —
                # 아래 _deadline_project 가 한계 안으로 옮긴다 (0 반복이면 초기값 = DB 이웃 해)
                best = (sol, tag + "+마감", it, time.time() - t0); partial = True; break
            fails.append(f"{tag}: {status} ({it} 반복)")
        if best is None and cold_ok:
            Jc, tag_c, w_c = min(cold_ok, key=lambda x: x[0])
            _set_init(w_c); t0 = time.time()
            try:
                sol = opti.solve(); best = (sol, tag_c, opti.stats().get("iter_count"), time.time() - t0)
            except RuntimeError:
                fails.append(f"{tag_c}: 콜드 최선 재풀이 실패")
            if logger is not None:
                logger.info(f"  분리 콜드: {len(inits)}개 중 {len(cold_ok)}개 수렴, 최선 {tag_c} J {Jc:.4f}")
        if best is None:
            raise ValueError("분리 던지기 NLP 실패 — " + "; ".join(fails))
        sol, tag, it, dt = best
        Pv = np.array(sol.value(S["P"])); t_f = float(sol.value(S["t_f"])); t_star = float(sol.value(S["t_s"]))
        v_out = float(sol.value(S["v_out"])) if v_out_fixed is None else float(v_out_fixed); v_in = float(sol.value(S["v_in"]))
        v0_sol = A0 + B0 * (float(sol.value(S["v_out"])))
        J_val = float(sol.value(S["J_time"])) * T.W1 + float(sol.value(S["J_win"])) * T.W2; Jwin_val = float(sol.value(S["J_win"])); mode = S["mode"]
    proj = None
    if partial:                                                              # 마감 값: 한계 안으로 (못 옮기면 ValueError)
        Pv, proj = _deadline_project(Pv, t_f)
    q_of, qd_of, qdd_of = _sample(Pv, t_f)
    t_end = _grid_ceil(t_f)
    ts = np.arange(int(round(t_end / GRID_DT)) + 1) * GRID_DT
    q_f, qd_f = q_of(t_f), qd_of(t_f)
    Q = np.array([q_of(t) if t <= t_f else q_f + qd_f * (t - t_f) for t in ts]) * PLANNER_SIGN
    Qd = np.array([qd_of(t) if t <= t_f else qd_f for t in ts]) * PLANNER_SIGN
    Qdd = np.array([qdd_of(t) if t <= t_f else np.zeros(6) for t in ts]) * PLANNER_SIGN
    rt = T.RELEASE_TIME
    w0, w1 = t_star - rt / 2, t_star + rt / 2
    i_rel = int(np.argmin(np.abs(ts - t_star)))
    if hist is not None:
        qr = realized_eval_hist(Pv, t_f, hist["H"])
        z_r = np.array([tcp_of(qr[0](float(t)) * PLANNER_SIGN)[2] for t in hist["t_pts"]])   # 실제 높이 (제약 시각) — 기록용
        lift = dict(z_min=float(np.min(z_r[np.asarray(hist["t_pts"]) >= T_INFL]) - hist["z_obj"]),
                    z_tc=float(z_r[len(TL_LIFT) + N_TC // 2] - hist["z_obj"]), t_c=hist["t_c"],
                    z_cont_tc=float(hist["z_cont"][len(TL_LIFT) + N_TC // 2] - hist["z_obj"]), fit_err=hist.get("fit_err"))
        if hist.get("t_chk") is not None:                                    # NLP 에 안 건 시각까지 포함한 확인 (실제 − 하한, 음수 = 위반)
            t_chk = np.asarray(hist["t_chk"]); m = t_chk >= T_INFL
            z_chk = np.array([tcp_of(qr[0](float(t)) * PLANNER_SIGN)[2] for t in t_chk])
            marg = z_chk - np.asarray(hist["zlo_chk"])
            k = int(np.argmin(np.where(m, marg, np.inf)))
            lift.update(z_min=float(np.min(z_chk[m]) - hist["z_obj"]), chk_margin=float(marg[k]), chk_t=float(t_chk[k]), n_con=int(len(hist["t_pts"])))
    else:
        qr = T.realized_eval(Pv, t_f, q0, v0_sol, (TRAVEL_Z - float(z_obj)) / max(v_out, 0.05)); lift = None
    p_launch, v_launch, p_land, err = _predict_landing(qr[0](t_star), qr[1](t_star), np.asarray(p_target, float))
    p_goal = np.asarray(end.p_tcp, float) + vals["mv"] * np.asarray(v_belt, float) * (t_f - t_ref_end)
    info = dict(model="sep", t_f=t_f, t_star=t_star, J=J_val,
                J_win=Jwin_val, release_time=rt, tag=f"sep-{tag}", n_fail=len(fails), iters=it, solve_s=dt,
                t_end=float(t_end), end_exact=True, end_extrap_s=t_f - t_ref_end, stop_end=bool(stop_end),
                end_tcp_err_m=float(np.linalg.norm(tcp_of(q_f * PLANNER_SIGN) - p_goal)), v_out=v_out, v_in=v_in, mode=mode, lift=lift, deadline_proj=proj)
    start = ThrowBoundary(start_q, to_robot(v0_sol))
    end_b = ThrowBoundary(Q[-1], Qd[-1], end.t, end.q_rate, end.p_tcp, end.v_tcp_exact, end.j6)
    raw = dict(P=Pv, t_f=t_f, t_star=t_star, v_out=v_out, v_in=v_in)
    tp = ThrowPlan(t=ts, q=Q, qd=Qd, qdd=Qdd, t_f=t_f, t_release=t_star, release_window=(w0, w1), i_release=i_rel, t_suction_on=t_f,
                   start=start, end=end_b, p_target=np.asarray(p_target, float), p_launch=p_launch, v_launch=v_launch, p_land=p_land,
                   land_err=err, info=info, raw=raw)
    if logger is not None:
        lf = "" if lift is None else (f", 실제 이력: 실제 높이 최저 {lift['z_min'] * 1e3:+.2f} mm, 접촉 t_c {lift['t_c']:.3f}s 높이 "
                                      f"{lift['z_tc'] * 1e3:+.2f} mm (가정 이어짐 {lift['z_cont_tc'] * 1e3:+.2f}), 이력 맞춤 오차 {lift['fit_err']}"
                                      + ("" if "chk_margin" not in lift else
                                         f", 하한 여유 최소 {lift['chk_margin'] * 1e3:+.2f} mm @ {lift['chk_t']:.3f}s (검사 {lift['n_con']} 곳)"))
        logger.info(f"  분리 던지기{' 정지 종점' if stop_end else ''} ({tag}, {it} 반복, {dt:.1f}s): t_f {t_f:.3f}, t* {t_star:.3f}, v_상승 {v_out:.2f}, v_하강 {v_in:.2f} m/s, "
                    f"착지 예측 오차 {err * 100:.1f} cm{lf}")
    return tp, v_out, v_in, raw


def call(fn, *a, **k):
    """시뮬 풀이 중 정지(GP8_SIM_PAUSE_SOLVE)가 켜져 있으면 그 안에서 부른다."""
    try:
        from gp8_control.sim import pause_clock
        return pause_clock.call("sep", fn, *a, **k)
    except ImportError:
        return fn(*a, **k)

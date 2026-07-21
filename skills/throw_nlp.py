"""
GP8 투척 궤적 NLP — CasADi(IPOPT) + clamped cubic B-Spline.
실행: .venv/bin/python throw_nlp.py → plots/throw_nlp_debug.png

문제 정식화
===========
결정 변수:
  P (6×N_CTRL) : 관절별 B-Spline 제어점 (P₁, Pₙ₋₁은 경계조건으로 소거 → 독립 변수 아님)
  t_f          : 전체 궤적 종료 시간
  t_star       : release 윈도우 중심 시점

제약 (전부 hard — penalty 아님):
  1. 경계 기구학 조건 (p0, v0, vf는 '주어지는' 입력; 이번 데모는 예시로 v0=vf=0):
       q(0) = P₀ = q_start(p0의 IK)                       … 첫 제어점 고정
       q̇(0) = p/(t_f·u_{p+1})·(P₁−P₀) = v0   ⟹  P₁ = P₀ + (u_{p+1}/p)·t_f·v0
       q̇(t_f) = p/(t_f·(1−u_{m−p−1}))·(Pₙ−Pₙ₋₁) = vf ⟹ Pₙ₋₁ = Pₙ − ((1−u_{m−p−1})/p)·t_f·vf
     → P₁, Pₙ₋₁을 독립 결정변수에서 '소거'하고 t_f에 종속된 식으로 구성.
       auxiliary 등식 제약 없이 임의의 경계조건을 구조적으로 만족
       (v0=vf=0이면 P₀=P₁, Pₙ₋₁=Pₙ으로 환원).
  2. |q̈| ≤ 5·v_max : cubic B-Spline의 q̈는 u에 대해 구간별 '선형'·연속 (단순 knot에서
                    C²) → 모든 knot에서만 제약해도 [0,T] 전 구간 exact.
  3. 관절 위치 한계 (q_B≥10°, U 팔올림≤45°, |R|≤80° 포함): **제어점 convex hull**
     — 모든 제어점을 [Q_LO, Q_HI]로 bound하면 B-spline의 볼록껍질 성질로 q(u)
     전 구간 만족이 'solve마다' 수학적으로 보장됨 (2026-07-16 사용자 요구
     "매번 확실하게"). 약간 보수적이지만 노드 사이 침범이 원천 불가능.
     dense 검증 + refinement 루프는 안전망으로 유지 (발동할 일 없음).
  4. 시간 변수    : release_time/2 ≤ t* ≤ t_f − release_time/2 (윈도우가 [0,t_f] 안).

목적 함수 (하드웨어 한계만 hard, 정확도는 penalty — 스펙 원형):
  J = w1·t_f + w2·∫_{t*−rt/2}^{t*+rt/2} (W_ACC·J_acc + W_SENS·J_sens) dt
  J_acc  = ‖x_land(q,q̇) − target‖²  포물선 궤적 오차 (calc_landing_error).
           W_ACC=1e7 (실측 선정): 속도 hard constraint가 문제를 정칙화해서
           착탄 0.01~0.02mm를 6~10s에 달성 (구 1e9 시절 42s의 원인이던
           스케일 폭주 없음).
  J_sens = ‖∂x_land/∂q‖² + ‖∂x_land/∂q̇‖²   (calc_throwing_sensitivity —
           CasADi 자동미분 jacobian; 논문 Eq.(1)의 Ballistic×Kinematic chain과 동일 값)
  적분은 윈도우를 N_WIN개 노드로 이산화해 trapezoid 합산.
  t*가 변수라 윈도우 노드 시각 u_i = (t*+offset_i)/t_f 가 심볼릭 —
  B-Spline basis를 심볼릭 u에서 평가(Cox–de Boor + if_else)해 자동미분 가능.

  5. 관절 속도 한계 (2026-07-15 활성화): |q̇| ≤ q̇_max — 도함수 스플라인 제어점
     convex hull로 전 구간 hard 보장 (실기 실행가능성; 하드웨어 스펙 한계는
     위치·속도·가속도 전부 hard constraint).
"""

import casadi as cs
import numpy as np
from scipy.interpolate import BSpline

from throwing import (
    G,               # 9.81
    GP8_Q_MAX,       # 관절 위치 상한
    GP8_Q_MIN,       # 관절 위치 하한
    GP8_QD_MAX,      # 관절 속도 한계 (가속도 한계 5×의 기준)
    _CHAIN,          # FK 체인 정의 (translation, 회전축) — 심볼릭 FK 포팅에 사용
    fk_frames,       # 수치 FK (초기자세 계산·검증·플롯용)
    ik_position,     # 위치 IK (시작 자세)
)

# ---------------------------------------------------------------------------
# 파라미터
# ---------------------------------------------------------------------------
N_CTRL = 12                     # 관절당 B-Spline 제어점 수
DEGREE = 3                      # cubic (q̈가 구간별 선형 → knot 제약 = exact)
N_WIN = 7                       # release 윈도우 적분 이산화 노드 수
RELEASE_TIME = 0.05             # 윈도우 총 길이 (s). 긴 윈도우는 W_ACC가 너무 크면
# 실패 — 스윙 ~3 m/s × 0.05s = 12~15cm 구간 전체에 mm 정확도를 강요해 basin이
# 죽음 (1e7에서 최악 쌍 0/6 수렴). W_ACC를 1e4로 낮추면 같은 쌍 5/6 수렴,
# 윈도우 오차 max ~6mm (bin 개구부 대비 무시 가능) — 2026-07-16 스윕.
B_LO_DEG = 10.0                 # q_B ≥ 10° (위치 하한의 B 성분 — 2026-07-15 사용자 수정)
QDD_LIM = 5.0 * GP8_QD_MAX      # 가속도 한계 = 5×속도한계
W1, W2 = 0.5, 1.0               # 시간 / 윈도우 정확도 가중치 (스펙의 w1, w2)
W_SENS = 10.0                   # sensitivity 비용 스케일
W_ACC = 1e5                     # 착탄오차 penalty 스케일 (m² 오차 → 비용).
# 사용자 요구 (2026-07-15): 정확도는 hard constraint가 아니라 목적함수 penalty로.
# 너무 크면 사실상 hard로 작동해 수렴성(조건수)을 죽임 — rt=0.05 구스윕 (2026-07-16,
# 최악 쌍 수렴/윈도우오차): 1e7 0/6, 1e5 2/6·1.8mm, 1e4 5/6·6.2mm, 1e3 6/6·41mm.
# 2026-07-21 재스윕 (정확도 상향 요구; 실제 정적 테스트 조합 4종, 실기와 같은
# multistart 기준): 1e4 4/4·6.7mm/5.2s, 3e4 4/4·(1건 나쁜 basin inf), 1e5
# 4/4·2.1mm/6.8s → 1e5로 상향. 나쁜 basin은 어느 가중치서든 가능하고 착탄
# 게이트(30mm)+multistart가 걸러낸다. (rt=0.01 시절엔 1e7이 0.01mm — stiff해도 수렴.)
T_BOUNDS = (0.3, 1.5)           # t_f 탐색 범위
REFINE_TOL = 1e-6               # 위치 한계 dense 검증 허용 위반 (rad) — IPOPT의
                                #   제약 잔차(~1e-8)보다 느슨해야 활성 bound에서
                                #   가짜 미수렴 경고가 안 남 (물리적으론 6e-5°)
MAX_REFINE = 6                  # adaptive refinement 최대 반복

# 기둥(베이스+shoulder yoke) 회피 hard constraint (2026-07-16): wrist·그리퍼 로드
# 50/75%·TCP 4점이 기둥 축 세그먼트(z∈[0, COL_H])에서 COL_R 이상 떨어질 것.
# 캘리브레이션 (충돌/통과 실측 궤적 32개): 충돌 해는 기둥 관통(r 0.00~0.04) 또는
# yoke 스침(로드 r 0.11~0.15, z 0.34~0.53), 통과 해는 지상에서 r ≥ 0.29 —
# 지하 dip(z<0)은 세그먼트 거리의 z항이 슬랙이라 안 걸림. 시작 자세 wrist r ≥ 0.38.
# (예전 TCP '바닥' hard 제약 0/148 전멸과 다른 점: 초기해가 이 제약을 위반하지 않음.)
COL_R = 0.18
COL_H = 0.55

# clamped uniform knot vector (정규화 시간 u∈[0,1])
KNOTS = np.concatenate([np.zeros(DEGREE),
                        np.linspace(0.0, 1.0, N_CTRL - DEGREE + 1),
                        np.ones(DEGREE)])

# 위치 한계: B(index 4) 하한을 10°로 올려 q_B≥10°를 위치 한계에 통합 (제약 3)
POS_LIMIT_MODE = "hull"   # 제어점 convex hull (2026-07-16) — warm DB 무효화 마커
Q_LO = GP8_Q_MIN.copy(); Q_LO[4] = np.deg2rad(B_LO_DEG)
Q_HI = GP8_Q_MAX.copy()
# U(팔 2번째 pitch 관절, index 2) 위로 꺾임 ≤ 45° — 팔을 너무 높이 들지 않도록 (사용자 규칙).
# 플래너 컨벤션은 '위 = 음수 q_U' (corr(z_TCP, −q_U)=0.99) → 하한 −45°.
# (URDF/Yaskawa 부호 q_U_urdf = −q_U_planner 로는 q_U ≤ +45°에 해당)
Q_LO[2] = np.deg2rad(-45.0)
# R(전완 roll, index 3) |q_R| ≤ 80° — 전완이 뒤집히지 않도록 (2026-07-15 사용자 규칙).
Q_LO[3], Q_HI[3] = np.deg2rad(-80.0), np.deg2rad(80.0)


# ---------------------------------------------------------------------------
# 모듈 1: B-Spline basis 평가 (Cox–de Boor, 심볼릭/수치 겸용)
# ---------------------------------------------------------------------------
def bspline_basis(u, knots=KNOTS, p=DEGREE):
    """
    u(스칼라 — float 또는 CasADi MX 식)에서 basis 3종 평가:
      N(u), N'(u), N''(u)  — 각각 길이 N_CTRL 리스트 (u에 대한 미분).
    knots는 수치 상수라 재귀 구조(0-span 생략, 분모)는 파이썬에서 결정되고,
    u만 심볼릭으로 흘러 CasADi가 자동미분 가능. u=1은 마지막 span에 ≤로 포함.
    """
    u = u if isinstance(u, (cs.MX, cs.SX)) else cs.MX(float(u))
    m = len(knots) - 1
    # degree 0: 구간 지시함수 (if_else — 0-span은 항등 0)
    table = [[]]
    for k in range(m):
        if knots[k + 1] > knots[k]:
            hi_ok = (u <= knots[k + 1]) if knots[k + 1] >= knots[-1] - 1e-12 \
                else (u < knots[k + 1])
            table[0].append(cs.if_else(cs.logic_and(u >= knots[k], hi_ok), 1.0, 0.0))
        else:
            table[0].append(cs.MX(0.0))
    # degree 1..p: Cox–de Boor 재귀 (0-분모 항은 생략)
    for d in range(1, p + 1):
        row = []
        for k in range(m - d):
            term = cs.MX(0.0)
            den1 = knots[k + d] - knots[k]
            if den1 > 0:
                term = term + (u - knots[k]) / den1 * table[d - 1][k]
            den2 = knots[k + d + 1] - knots[k + 1]
            if den2 > 0:
                term = term + (knots[k + d + 1] - u) / den2 * table[d - 1][k + 1]
            row.append(term)
        table.append(row)

    def deriv(lower, dd):  # 차수 dd basis의 도함수 (하위차수 basis로 표현)
        out = []
        for k in range(m - dd):
            t1 = cs.MX(0.0); t2 = cs.MX(0.0)
            den1 = knots[k + dd] - knots[k]
            if den1 > 0:
                t1 = dd / den1 * lower[k]
            den2 = knots[k + dd + 1] - knots[k + 1]
            if den2 > 0:
                t2 = dd / den2 * lower[k + 1]
            out.append(t1 - t2)
        return out

    dN = deriv(table[p - 1], p)                 # N'  (degree p-1 basis로부터)
    ddN = deriv(deriv(table[p - 2], p - 1), p)  # N'' (degree p-2 → p-1' → p'')
    return table[p], dN, ddN


def spline_qs(P, u, t_f):
    """제어점 P(6×N_CTRL)와 u에서 (q, q̇, q̈) — 시간 미분은 1/t_f 체인룰."""
    N, dN, ddN = bspline_basis(u)
    q = sum(P[:, k] * N[k] for k in range(N_CTRL))
    qd = sum(P[:, k] * dN[k] for k in range(N_CTRL)) / t_f
    qdd = sum(P[:, k] * ddN[k] for k in range(N_CTRL)) / t_f**2
    return q, qd, qdd


# ---------------------------------------------------------------------------
# 모듈 2: 운동학 (CasADi 심볼릭 FK — throwing._CHAIN 포팅)
# ---------------------------------------------------------------------------
def _rot_sym(axis, t):
    c, s = cs.cos(t), cs.sin(t)
    if axis == "x":
        return cs.vertcat(cs.horzcat(1, 0, 0), cs.horzcat(0, c, -s), cs.horzcat(0, s, c))
    if axis == "y":
        return cs.vertcat(cs.horzcat(c, 0, s), cs.horzcat(0, 1, 0), cs.horzcat(-s, 0, c))
    return cs.vertcat(cs.horzcat(c, -s, 0), cs.horzcat(s, c, 0), cs.horzcat(0, 0, 1))


def fk_tcp_sym(q):
    """TCP 위치 p(q) — throwing.fk_frames와 동일 체인 (TCP = 마지막 링크 원점)."""
    R = cs.MX.eye(3)
    p = cs.MX.zeros(3)
    for (xyz, ax), i in zip(_CHAIN, range(6)):
        p = p + R @ cs.DM(list(xyz))
        R = R @ _rot_sym(ax, q[i])
    return p


def fk_col_points_sym(q):
    """기둥 회피 제약용 포인트 4개: wrist(관절 B 원점), 그리퍼 로드 50/75%, TCP."""
    R = cs.MX.eye(3)
    p = cs.MX.zeros(3)
    origins = []
    for (xyz, ax), i in zip(_CHAIN, range(6)):
        p = p + R @ cs.DM(list(xyz))
        origins.append(p)
        R = R @ _rot_sym(ax, q[i])
    wrist, tcp = origins[4], origins[5]
    return [wrist, wrist + 0.5 * (tcp - wrist), wrist + 0.75 * (tcp - wrist), tcp]


# ---------------------------------------------------------------------------
# 모듈 3: 목적 함수 구성 요소 (CasADi Function으로 모듈화)
# ---------------------------------------------------------------------------
_OBJ_SYM = None


def make_objective_functions_sym():
    """calc_landing_error(q,q̇,tgt), calc_throwing_sensitivity(q,q̇,tgt) —
    target을 세 번째 '입력'으로 받는 심볼릭 버전 (모듈 캐시). 파라메트릭 polish
    솔버(target = opti.parameter)와 고정-target 래퍼가 같은 몸체를 공유한다.

    독립 심볼 (q, q̇)로 식을 세우고 Function으로 감싸므로, NLP에서는 제어점의
    식(q_i(P,t*,t_f))을 그대로 넣어도 CasADi가 chain rule로 자동미분한다.
    """
    global _OBJ_SYM
    if _OBJ_SYM is not None:
        return _OBJ_SYM
    q = cs.MX.sym("q", 6)
    qd = cs.MX.sym("qd", 6)
    tgt = cs.MX.sym("tgt", 3)

    p = fk_tcp_sym(q)                    # TCP 위치
    v = cs.jtimes(p, q, qd)              # TCP 속도 = Jv(q)·q̇ (자동미분)

    # ---- 탄도 모델: 착탄점 x_land(q, q̇) ----
    # TODO: 사용자 파일 수식 삽입 지점 — 현재는 논문 Eq.(2),(3) (항력 무시 포물선,
    #       착지 평면 z = tgt[2], 비행시간 = 이차방정식의 양의 근)
    disc = v[2]**2 + 2.0 * G * (p[2] - tgt[2])
    s = cs.sqrt(cs.fmax(disc, 1e-9))     # 판별식 guard (반복 중 일시적 음수 방지)
    tau = (v[2] + s) / G                 # 비행시간
    x_land = p[0:2] + v[0:2] * tau       # 착탄점 (x, y)

    # J_acc: 포물선 궤적 오차 (target까지의 착탄 오차 제곱)
    err2 = cs.sumsqr(x_land - tgt[0:2])

    # J_sens: 투척 민감도 = ‖∂x_land/∂q‖² + ‖∂x_land/∂q̇‖²
    # TODO: 사용자 파일 수식 삽입 지점 — 현재는 CasADi jacobian(자동미분)으로 계산.
    #       논문 Eq.(1)의 Ballistic×Kinematic chain과 수학적으로 동일한 값.
    S_q = cs.jacobian(x_land, q)         # ∂x_land/∂q   (2×6)
    S_qd = cs.jacobian(x_land, qd)       # ∂x_land/∂q̇  (2×6)
    sens = cs.sumsqr(S_q) + cs.sumsqr(S_qd)

    _OBJ_SYM = (cs.Function("calc_landing_error", [q, qd, tgt], [err2]),
                cs.Function("calc_throwing_sensitivity", [q, qd, tgt], [sens]))
    return _OBJ_SYM


def make_objective_functions(p_target):
    """고정-target 래퍼: calc_landing_error(q,q̇), calc_throwing_sensitivity(q,q̇)."""
    F_err, F_sens = make_objective_functions_sym()
    q = cs.MX.sym("q", 6)
    qd = cs.MX.sym("qd", 6)
    t = cs.DM(np.asarray(p_target, float))
    return (cs.Function("calc_landing_error", [q, qd], [F_err(q, qd, t)]),
            cs.Function("calc_throwing_sensitivity", [q, qd], [F_sens(q, qd, t)]))


# ---------------------------------------------------------------------------
# NLP 조립 + 풀이 (adaptive refinement 루프 포함)
# ---------------------------------------------------------------------------
def _assemble_nlp(opti, P_free, t_f, t_star, q0, v0c, vfc, tgt, u_pos_nodes, rt):
    """NLP 몸체 조립 (제약 1~5 + 기둥 회피 + 목적함수) — cold(_build_and_solve)와
    캐시된 파라메트릭 polish 솔버(_polish_solve)가 공유. q0/v0c/vfc/tgt는 DM 상수
    또는 opti.parameter — 동일 수식이라 파라미터화가 그대로 성립한다.
    전체 제어점 행렬 P(6×N_CTRL) 식을 리턴."""
    u_head = KNOTS[DEGREE + 1]
    u_tail = 1.0 - KNOTS[len(KNOTS) - DEGREE - 2]

    # -- 제약 1: 경계 기구학 조건 — 변수 소거 (auxiliary 등식 없음) --
    P1c = q0 + (u_head / DEGREE) * t_f * v0c                # q̇(0)=v0
    Pn = P_free[:, -1]
    Pn1 = Pn - (u_tail / DEGREE) * t_f * vfc                # q̇(t_f)=vf
    P = cs.horzcat(q0, P1c, P_free[:, :-1], Pn1, Pn)        # 전체 6×N_CTRL

    # -- 제약 3a: 위치 한계(+q_B≥10°, U≥−45°, |R|≤80°) — 제어점 convex hull --
    # B-spline은 제어점들의 convex hull 안에 있으므로, '모든 제어점'을
    # [Q_LO, Q_HI]로 bound하면 q(u)의 전 구간 만족이 solve마다 수학적으로 보장됨
    # (2026-07-16 사용자 요구 "매번 확실하게" — 구 collocation 31노드+refinement는
    # 노드 사이 침범을 사후에 잡는 방식이었음). 약간 보수적(궤적이 한계에 정확히
    # 닿으려면 제어점이 몰려야 함)이지만 침범이 원천 불가능. refinement 루프는
    # 순수 안전망으로 강등 (발동할 일 없음).
    # P₀(=q_start)는 상수/파라미터라 제약 불가 — 입력 검증이 담당. P₁은 v0=0이면
    # 상수로 접혀 결정변수가 없으므로 depends_on으로 가드 (여기의 vars는 opti.x가
    # 아니라 실제 변수 핸들이라 안전).
    _vars = cs.veccat(cs.vec(P_free), t_f, t_star)
    for k in range(1, N_CTRL):
        if not cs.depends_on(P[:, k], _vars):
            continue
        opti.subject_to(opti.bounded(Q_LO, P[:, k], Q_HI))

    # -- 제약 3b: 기둥 회피 — collocation 노드 (비선형 FK 제약이라 hull 불가) --
    # 기둥: wrist·로드·TCP 4점의 기둥 축 세그먼트(z∈[0,COL_H]) 거리 ≥ COL_R.
    # z<0(지하 dip)·z>COL_H는 z항이 슬랙. 활성 구간은 release 창 끝(+버퍼)까지만 —
    # 감속 꼬리는 지하로 다이브하며 축 근처를 지나는 게 일상이고 (dry-run 꼬리
    # 교체가 처리), 전 구간에 걸면 무충돌 basin까지 잘라 multistart가 전멸한다
    # (2026-07-16 실측: bin9/11 96샘플 전멸). t*·t_f가 변수라 노드별 활성 여부가
    # 심볼릭 → smooth sigmoid 게이트 (act≈1: 창 이전, act≈0: 꼬리).
    # (TCP '바닥' 클리어런스는 hard로 걸면 0/148 전멸 — 초기해가 깊게 위반한 채
    #  시작해 restoration 즉사. 바닥 여유는 sim dry-run 가드가 담당 — 2026-07-15)
    u_re = (t_star + rt / 2) / t_f + 0.08          # release 창 끝 + 버퍼 (u 단위)
    for u_c in u_pos_nodes:
        if float(u_c) == 0.0:
            continue    # u=0: q=P₀=q_start 상수/파라미터 (결정변수 없음 — Opti가 거부)
        q_c, _, _ = spline_qs(P, float(u_c), t_f)
        act = 1.0 / (1.0 + cs.exp(40.0 * (float(u_c) - u_re)))
        for pt in fk_col_points_sym(q_c):
            d2 = (pt[0]**2 + pt[1]**2 + cs.fmax(pt[2] - COL_H, 0)**2
                  + cs.fmax(-pt[2], 0)**2)
            opti.subject_to(d2 >= COL_R**2 * act)

    # -- 제약 2: 가속도 한계 — knot 노드 (cubic이라 전 구간 exact) --
    for u_k in np.unique(KNOTS):
        _, _, qdd_k = spline_qs(P, float(u_k), t_f)
        opti.subject_to(opti.bounded(-QDD_LIM, qdd_k, QDD_LIM))

    # -- 제약 4: 시간 변수 --
    opti.subject_to(t_star >= rt / 2)
    opti.subject_to(t_star <= t_f - rt / 2)
    opti.subject_to(opti.bounded(T_BOUNDS[0], t_f, T_BOUNDS[1]))

    # -- 제약 5: 관절 속도 한계 (2026-07-15 활성화) — 도함수 스플라인 제어점
    # convex hull로 전 구간 hard 보장 (보수적). 미활성 시 IPOPT가 하드웨어 정격
    # 초과(qd 1.04×) 해를 수렴시키는 사례가 있어 실기-실행가능성 기준으로 포함.
    for k in range(N_CTRL - 1):
        span = KNOTS[k + DEGREE + 1] - KNOTS[k + 1]
        opti.subject_to(opti.bounded(-GP8_QD_MAX * t_f,
                        DEGREE * (P[:, k + 1] - P[:, k]) / span, GP8_QD_MAX * t_f))

    # -- 목적: J = w1·t_f + w2·∫윈도우 (W_ACC·J_acc + W_SENS·J_sens) dt --
    # 착탄 정확도는 penalty로 목적함수에 포함 (사용자 요구 — 스펙 원형).
    # 하드웨어 한계(위치/속도/가속도)와 기둥 회피만 hard constraint.
    # t*가 변수 → 노드 시각 u_i = (t*+offset)/t_f 가 심볼릭 (가변 시간 맵핑).
    F_err, F_sens = make_objective_functions_sym()
    offsets = np.linspace(-rt / 2, rt / 2, N_WIN)
    trap_w = np.full(N_WIN, rt / (N_WIN - 1)); trap_w[[0, -1]] *= 0.5
    J_win = cs.MX(0.0)
    for off, w in zip(offsets, trap_w):
        # clamp: 제약4는 '해'에서만 u_i∈[0,1] 보장 — IPOPT 중간 iterate가 범위를
        # 벗어나면 basis≡0이 되어 목적이 붕괴(gradient 소실)하므로 방어적 clamp
        u_i = cs.fmin(cs.fmax((t_star + off) / t_f, 0.0), 1.0)
        q_i, qd_i, _ = spline_qs(P, u_i, t_f)
        J_win = J_win + w * (W_ACC * F_err(q_i, qd_i, tgt)
                             + W_SENS * F_sens(q_i, qd_i, tgt))
    opti.minimize(W1 * t_f + W2 * J_win)
    return P


_POLISH_SOLVERS = {}   # (rt, u_pos 노드 집합) → 조립된 파라메트릭 솔버 (재사용)


def _polish_solve(u_pos, rt, q_start, v0, vf, p_target, warm_data):
    """캐시된 파라메트릭 솔버로 full warm start polish 한 번 풀기.
    CasADi 그래프를 (rt, 노드 집합)당 한 번만 조립하고 이후 호출은 parameter
    (q_start/v0/vf/target)와 초기값(primal+lam_g)만 갈아끼움 — 조립 오버헤드
    ~0.3–0.5s 제거 (docs/warmstart_speedup.md §5 '현실적 하한 ~1초')."""
    key = (round(float(rt), 9), tuple(np.round(u_pos, 12)))
    h = _POLISH_SOLVERS.get(key)
    if h is None:
        opti = cs.Opti()
        P_free = opti.variable(6, N_CTRL - 3)
        t_f = opti.variable()
        t_star = opti.variable()
        q0 = opti.parameter(6)
        v0p = opti.parameter(6)
        vfp = opti.parameter(6)
        tgt = opti.parameter(3)
        P = _assemble_nlp(opti, P_free, t_f, t_star, q0, v0p, vfp, tgt, u_pos, rt)
        opti.solver("ipopt", {"print_time": False, "expand": True},
                    {"max_iter": 3000, "print_level": 0, "sb": "yes",
                     "tol": 1e-4, "mu_init": 1e-6, "warm_start_init_point": "yes",
                     "warm_start_bound_push": 1e-9,
                     "warm_start_mult_bound_push": 1e-9,
                     "warm_start_slack_bound_push": 1e-9})
        h = dict(opti=opti, P_free=P_free, t_f=t_f, t_star=t_star,
                 q0=q0, v0=v0p, vf=vfp, tgt=tgt, P=P)
        _POLISH_SOLVERS[key] = h
    opti = h["opti"]
    opti.set_value(h["q0"], np.asarray(q_start, float))
    opti.set_value(h["v0"], np.asarray(v0, float))
    opti.set_value(h["vf"], np.asarray(vf, float))
    opti.set_value(h["tgt"], np.asarray(p_target, float))
    free_idx = list(range(2, N_CTRL - 2)) + [N_CTRL - 1]
    opti.set_initial(h["P_free"], warm_data["P"][:, free_idx])
    opti.set_initial(h["t_f"], warm_data["t_f"])
    opti.set_initial(h["t_star"], warm_data["t_star"])
    lam = warm_data.get("lam_g")
    if lam is not None and np.size(lam) == opti.ng:  # 제약 구성 동일할 때만 유효
        opti.set_initial(opti.lam_g, np.asarray(lam, float).ravel())
    sol = opti.solve()   # 실패 시 예외 전파 — 호출측이 cold multistart로 fallback
    return dict(P=np.array(sol.value(h["P"])), t_f=float(sol.value(h["t_f"])),
                t_star=float(sol.value(h["t_star"])), J=float(sol.value(opti.f)),
                lam_g=np.array(sol.value(opti.lam_g)).ravel())


def _spline_eval(P, t_f):
    """수치 해 (P, t_f)로 scipy BSpline 평가기 (q, q̇, q̈)(t) 생성 — 검증·플롯용."""
    spl = [BSpline(KNOTS, P[j], DEGREE) for j in range(6)]
    q_of = lambda t: np.array([s(t / t_f) for s in spl])
    qd_of = lambda t: np.array([s.derivative(1)(t / t_f) for s in spl]) / t_f
    qdd_of = lambda t: np.array([s.derivative(2)(t / t_f) for s in spl]) / t_f**2
    return q_of, qd_of, qdd_of


def solve_throw_nlp(p_grasp, p_target, v0=None, vf=None,
                    release_time=RELEASE_TIME, q_start=None, verbose=False,
                    init=None, warm_data=None):
    """
    입력 (주어지는 값): p_grasp(=p0, 물체를 쥔 TCP 위치), p_target(착지 목표),
                        v0/vf(시작/종료 관절속도 6-vector — 기본 0, 일반형).
    q_start: 시작 관절자세를 직접 지정 (시뮬/실기에서 로봇의 실제 자세로 출발할 때).
             None이면 p_grasp의 IK로 계산 (p_grasp는 이때만 사용, None 허용).
    init: multistart용 초기해 변형 dict — dq_swing(스윙 끝자세 오프셋 6벡터),
          T0(t_f 초기값), chi0(t*/t_f 초기 비율). IPOPT는 local solver라 한
          초기해의 basin이 infeasible하면 실패하므로, 호출측에서 init을 바꿔가며
          다른 local minima를 탐색하는 용도. None이면 기본 초기해.
    warm_data: offline DB entry로 full warm start polish (docs/warmstart_speedup.md)
          — dict(P, t_f, t_star, lam_g, u_pos). primal+dual을 함께 넘겨 barrier를
          끝 상태에서 재시동, cold 6~10s → 1~2s. p0/q_start가 entry와 수십 cm
          달라도 유효 (경계조건이 파라미터화에 내장 — §5.5). 실패 시 예외를 그대로
          던지므로 호출측이 cold multistart로 fallback.
          (주의: primal만 넘기는 순진한 warm start는 역효과라 지원 안 함.)
    리턴: dict(P, t_f, t_star, lam_g, u_pos, knots, degree, ...) — B-Spline 궤적
          정의 일체 + warm DB entry 재료.
    """
    p_target = np.asarray(p_target, float)
    v0 = np.zeros(6) if v0 is None else np.asarray(v0, float)
    vf = np.zeros(6) if vf is None else np.asarray(vf, float)
    rt = release_time

    if q_start is None:
        # 시작 자세: p_grasp를 잡는 IK 해 중 B가 하한에서 떨어진 것
        for seedB in (0.6, 0.9, 0.3, 1.2):
            qs, ok = ik_position(np.asarray(p_grasp, float),
                                 np.array([0.2, 0.5, -0.2, 0.0, seedB, 0.0]))
            if ok and qs[4] >= Q_LO[4] + 0.01:   # B 하한(10°)에서 살짝 떨어진 해만
                q_start = qs
                break
        assert q_start is not None, "B≥10° 시작자세 IK 실패"
    else:
        q_start = np.asarray(q_start, float)
        if not (np.all(q_start >= Q_LO - 1e-9) and np.all(q_start <= Q_HI + 1e-9)):
            raise ValueError("q_start가 위치 한계(B≥10°, U 팔올림≤45° 포함) 밖")

    free_idx = list(range(2, N_CTRL - 2)) + [N_CTRL - 1]   # 독립 제어점 열 인덱스

    def _build_and_solve(u_pos_nodes, warm):
        """제약 3의 위치 collocation 노드 집합을 받아 cold NLP 한 번 풀기.
        (full warm start polish는 캐시된 파라메트릭 솔버 _polish_solve가 담당.)"""
        opti = cs.Opti()
        P_free = opti.variable(6, N_CTRL - 3)    # 독립 제어점: P₂..Pₙ₋₂, Pₙ
        t_f = opti.variable()
        t_star = opti.variable()
        P = _assemble_nlp(opti, P_free, t_f, t_star, cs.DM(q_start), cs.DM(v0),
                          cs.DM(vf), cs.DM(np.asarray(p_target, float)),
                          u_pos_nodes, rt)

        # -- 초기값: 선형 보간 스윙 (refinement 재풀이 시엔 이전 해 warm start) --
        if warm is None:
            # 기본 초기해: 스윙 S(yaw) 오프셋은 target y 부호에 맞춤 (고정 +0.13이면
            # +y target이 나쁜 basin으로 흘러가는 비대칭). init dict로 override 가능.
            cfg = init or {}
            s_yaw = 0.13 if p_target[1] <= 0 else -0.13
            dq = np.asarray(cfg.get("dq_swing",
                                    [s_yaw, 0.7, 0.6, 0.0, 0.4, 0.0]), float)
            T0 = float(cfg.get("T0", 0.9))
            chi0 = float(cfg.get("chi0", 0.5))
            q_end0 = q_start + dq
            P0_full = np.linspace(q_start, q_end0, N_CTRL).T
            opti.set_initial(P_free, P0_full[:, free_idx])
            opti.set_initial(t_f, T0)
            opti.set_initial(t_star, chi0 * T0)
        else:
            opti.set_initial(P_free, warm["P"][:, free_idx])
            opti.set_initial(t_f, warm["t_f"])
            opti.set_initial(t_star, warm["t_star"])

        # expand: MX→SX 그래프 전개 — 같은 수식·같은 iterate 경로로 평가만 ~2× 빨라짐
        # (2026-07-15 실측 14.0→7.1s, 해 완전 동일). jit은 컴파일 10분+라 부적합.
        opti.solver("ipopt", {"print_time": False, "expand": True},
                    {"max_iter": 3000, "print_level": 5 if verbose else 0, "sb": "yes"})
        sol = opti.solve()   # 실패 시 예외 전파 (자진신고)
        return dict(P=np.array(sol.value(P)), t_f=float(sol.value(t_f)),
                    t_star=float(sol.value(t_star)), J=float(sol.value(opti.f)),
                    lam_g=np.array(sol.value(opti.lam_g)).ravel())

    # ---- adaptive refinement 루프 (제약 3을 보수성 없이 전 구간 보장) ----
    u_pos = np.unique(np.concatenate([np.linspace(0.0, 1.0, 25), np.unique(KNOTS)]))
    if warm_data is not None and warm_data.get("u_pos") is not None:
        u_pos = np.unique(np.asarray(warm_data["u_pos"], float))  # 제약 구성 일치용
    out, warm = None, None
    for it in range(MAX_REFINE):
        first = out is None
        try:
            if first and warm_data is not None:
                # full warm start polish — 캐시된 파라메트릭 솔버 (조립 생략).
                # refinement 재풀이(드묾)는 노드가 늘어 구성이 달라지므로 아래
                # cold 경로(primal warm start)로.
                out = _polish_solve(u_pos, rt, q_start, v0, vf, p_target, warm_data)
            else:
                out = _build_and_solve(u_pos, warm)
        except RuntimeError:
            if first:
                raise          # 첫 풀이 실패는 그대로 자진신고 (fallback 해가 없음)
            # refinement warm start 정체 (직전 해가 제약 경계에 활성으로 붙으면
            # interior-point 재시동이 잘 실패함) → 같은 노드 집합 cold restart.
            try:
                out = _build_and_solve(u_pos, None)
            except RuntimeError:
                # cold도 실패 — 직전 해로 fallback (dense 위반 잔존, 경고와 함께)
                import warnings
                warnings.warn("refinement 재풀이 실패 — 직전 해 반환 (pos_viol_dense 확인 필요)")
                out = warm
                break
        # dense 검증: 위반 지점 수집 (관절×상/하한별 최악 지점) — BSpline 배열
        # 평가로 벡터화 (스칼라 3000회 루프는 polish 시간의 ~15%를 먹음)
        u_dense = np.linspace(0.0, 1.0, 3000)
        Qd_ = np.stack([BSpline(KNOTS, out["P"][j], DEGREE)(u_dense)
                        for j in range(6)], axis=1)
        new_nodes = []
        for j in range(6):
            lo_viol = Q_LO[j] - Qd_[:, j]
            hi_viol = Qd_[:, j] - Q_HI[j]
            for viol in (lo_viol, hi_viol):
                if viol.max() > REFINE_TOL:
                    new_nodes.append(u_dense[int(np.argmax(viol))])
        if not new_nodes:
            break
        u_pos = np.unique(np.concatenate([u_pos, new_nodes]))
        warm = out
    else:
        import warnings
        warnings.warn(f"위치 한계 refinement 미수렴 ({MAX_REFINE}회)")
        it = MAX_REFINE
    # 최종 해의 dense 위치 위반을 결과에 기록 (0이어야 정상; fallback 시 진단용)
    Qf = np.stack([BSpline(KNOTS, out["P"][j], DEGREE)(np.linspace(0.0, 1.0, 3000))
                   for j in range(6)], axis=1)
    out["pos_viol_dense"] = float(max(0.0, np.max(Qf - Q_HI), np.max(Q_LO - Qf)))
    out["refine_iters"], out["pos_nodes"] = it, len(u_pos)
    out["u_pos"] = u_pos            # warm DB entry용 (lam_g는 _build_and_solve가 기록)

    out.update(knots=KNOTS.copy(), degree=DEGREE, q_start=q_start,
               p_target=p_target, release_time=rt, v0=v0, vf=vf)
    return out


# ---------------------------------------------------------------------------
# 검증 + 데모 (scipy BSpline으로 '독립' 재구성해 교차검증)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from throwing import G_VEC, jacobian, landing_error

    P0_GRASP = np.array([0.30, 0.0, 0.50])
    TGT = np.array([1.80, 0.20, 0.0])

    # --- FK 포팅 자가검증: CasADi FK vs throwing.fk_frames (랜덤 100자세) ---
    qsym = cs.MX.sym("q", 6)
    fk_fun = cs.Function("fk", [qsym], [fk_tcp_sym(qsym)])
    rng = np.random.default_rng(1)
    fk_err = max(np.max(np.abs(np.array(fk_fun(qr)).ravel() - fk_frames(qr)[2]))
                 for qr in rng.uniform(-1.5, 1.5, (100, 6)))
    print(f"FK 교차검증 (CasADi vs numpy): 최대 오차 {fk_err:.2e}")

    res = solve_throw_nlp(P0_GRASP, TGT)          # v0=vf=0 (예시)
    P, t_f, t_st = res["P"], res["t_f"], res["t_star"]
    rt = res["release_time"]
    print(f"IPOPT 수렴: J={res['J']:.4f}, t_f={t_f:.3f}s, t*={t_st:.3f}s, "
          f"윈도우=[{t_st-rt/2:.3f}, {t_st+rt/2:.3f}], "
          f"refine {res['refine_iters']}회/노드 {res['pos_nodes']}개")

    q_of, qd_of, qdd_of = _spline_eval(P, t_f)

    # --- 제약 검증 (dense 4000점, scipy 독립 재구성) ---
    ts = np.linspace(0.0, t_f, 4000)
    Q = np.array([q_of(t) for t in ts])
    Qd = np.array([qd_of(t) for t in ts])
    Qdd = np.array([qdd_of(t) for t in ts])
    print(f"검증: |q̇(0)−v0|={np.linalg.norm(qd_of(0)-res['v0']):.1e}, "
          f"|q̇(t_f)−vf|={np.linalg.norm(qd_of(t_f)-res['vf']):.1e}")
    print(f"      가속도 max|q̈|/(5v_max) = {np.max(np.abs(Qdd)/QDD_LIM):.4f} (≤1 이어야)")
    print(f"      위치 위반 = {max(0.0, np.max(Q - Q_HI), np.max(Q_LO - Q)):.1e}, "
          f"B_min = {np.rad2deg(Q[:, 4].min()):.3f}°")
    print(f"      속도 max|q̇|/v_max = {np.max(np.abs(Qd)/GP8_QD_MAX):.2f} (hard, ≤1 이어야)")

    # --- 윈도우 착탄 오차 + 민감도 ---
    _, f_sens = make_objective_functions(TGT)
    for tr in t_st + np.linspace(-rt / 2, rt / 2, 5):
        q, qd = q_of(tr), qd_of(tr)
        p = fk_frames(q)[2]; v = jacobian(q)[0] @ qd
        print(f"  release t={tr:.4f}s → 착탄오차 {1e3*landing_error(p, v, TGT):6.2f} mm, "
              f"J_sens={float(f_sens(q, qd)):.2f}")

    # --- 착탄오차 프로파일: t*가 argmin에 앉았는지 ---
    prof_t, prof = [], []
    for t in np.linspace(0.02, t_f, 500):
        q, qd = q_of(t), qd_of(t)
        p = fk_frames(q)[2]; v = jacobian(q)[0] @ qd
        if v[2]**2 + 2 * G * p[2] > 0 and np.dot((TGT - p)[:2], v[:2]) > 0:
            prof_t.append(t); prof.append(landing_error(p, v, TGT))
    prof_t, prof = np.array(prof_t), np.array(prof)
    t_amin = prof_t[np.argmin(prof)]
    print(f"착탄오차 argmin t={t_amin:.3f}s (t*와 차이 {abs(t_amin-t_st)*1e3:.1f}ms)")

    # --- 플롯 ---
    tcp = np.array([fk_frames(q)[2] for q in Q])
    i_w0, i_w1 = np.searchsorted(ts, [t_st - rt / 2, t_st + rt / 2])

    def arm_pts(q):
        """팔 linkage 스냅샷: base→각 관절 원점→TCP 꺾은선 좌표."""
        o, _, p = fk_frames(q)
        return np.array([[0, 0, 0]] + [oo.tolist() for oo in o[1:]] + [p.tolist()])

    t_snap = np.linspace(0.0, t_f, 9)          # 균일 시간 간격 팔 자세 스냅샷
    fig = plt.figure(figsize=(16, 5))
    for k, (i, j, lbl) in enumerate([(0, 2, "z (m)"), (0, 1, "y (m)")]):
        ax = fig.add_subplot(1, 3, k + 1)
        ax.plot(tcp[:i_w0, i], tcp[:i_w0, j], "r-", lw=2, label="swing")
        ax.plot(tcp[i_w0:i_w1, i], tcp[i_w0:i_w1, j], "b-", lw=4, label="release window")
        ax.plot(tcp[i_w1:, i], tcp[i_w1:, j], color="orange", lw=2, label="decelerate")
        for m, ts_ in enumerate(t_snap):        # 시간 순서대로 진해짐 (release 근처는 파랑)
            pts = arm_pts(q_of(ts_))
            near_rel = abs(ts_ - t_st) <= 0.06
            ax.plot(pts[:, i], pts[:, j], "o-",
                    color="tab:blue" if near_rel else "k", ms=2.5, lw=1.1,
                    alpha=0.2 + 0.6 * m / (len(t_snap) - 1))
        for tr in t_st + np.linspace(-rt / 2, rt / 2, 5):
            q, qd = q_of(tr), qd_of(tr)
            p = fk_frames(q)[2]; v = jacobian(q)[0] @ qd
            tf_ = np.linspace(0, 0.7, 80)[:, None]
            fl = p[None] + v[None] * tf_ + 0.5 * G_VEC[None] * tf_**2
            fl = fl[fl[:, 2] >= -0.01]
            ax.plot(fl[:, i], fl[:, j], "g-", alpha=0.45)
        ax.plot(P0_GRASP[i], P0_GRASP[j], "ko", ms=6, label="grasp")
        ax.plot(TGT[i], TGT[j], "r*", ms=14, label="target")
        ax.set_xlabel("x (m)"); ax.set_ylabel(lbl)
        ax.grid(alpha=0.3); ax.axis("equal")
        ax.set_title("side view" if j == 2 else "top view")
        if k == 0:
            ax.legend(fontsize=7)
    ax = fig.add_subplot(1, 3, 3)
    ax.semilogy(prof_t, prof, "b-", lw=1.5)
    ax.axvspan(t_st - rt / 2, t_st + rt / 2, color="blue", alpha=0.2, label="window")
    ax.axvline(t_amin, color="r", ls="--", lw=1, label=f"argmin={t_amin:.3f}s")
    ax.set_xlabel("t (s)"); ax.set_ylabel("landing error (m)")
    ax.set_title("landing-error profile")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(f"CasADi/IPOPT B-spline NLP: t_f={t_f:.2f}s, t*={t_st:.3f}s, "
                 f"v0=vf=0, B>=10deg, |q..|<=5v_max", fontsize=11)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots",
                       "throw_nlp_debug.png")
    fig.savefig(out, dpi=130)
    print("플롯 저장:", out)

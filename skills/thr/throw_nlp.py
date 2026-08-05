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
  3. 관절 위치 한계 (q_B≥10°, U 팔올림≤30°, L∈[0,30]°, |R|≤80° 포함): **제어점 convex hull**
     — 모든 제어점을 [Q_LO, Q_HI]로 bound하면 B-spline의 볼록껍질 성질로 q(u)
     전 구간 만족이 'solve마다' 수학적으로 보장됨 (2026-07-16 사용자 요구
     "매번 확실하게"). 약간 보수적이지만 노드 사이 침범이 원천 불가능.
     dense 검증 + refinement 루프는 안전망으로 유지 (발동할 일 없음).
  4. 시간 변수    : release_time/2 ≤ t* ≤ t_f − release_time/2 (윈도우가 [0,t_f] 안).

목적 함수 (하드웨어 한계만 hard, 정확도는 penalty — 스펙 원형):
  J = w1·t_f + w2·∫_{t*−rt/2}^{t*+rt/2} W_ACC·J_acc dt,  윈도우 노드 ω_tcp=0 hard
  (구 W_SENS·J_sens 페널티는 2026-07-22 ω=0 제약으로 대체)
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

from .throwing import (
    G,               # 9.81
    GP8_Q_MAX,       # 관절 위치 상한
    GP8_Q_MIN,       # 관절 위치 하한
    GP8_QD_MAX,      # 관절 속도 한계 (가속도 한계 3×의 기준)
    TCP_X_MIN,       # Cartesian 엔벨로프 (아래 CART_MIN_X/Z 별칭의 단일 출처)
    TCP_Z_MIN,
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
RELEASE_TIME = 0.1              # 윈도우 총 길이 (s) — 2026-07-31 사용자 0.05→0.1.
# (이력) rt=0.1은 윈도우 전 구간 착탄이 ~20-30mm 하한에 걸리고 W_ACC를 올려도 안
# 줄었음(3e5~3e7 스윕 19~22mm 평평) — rt=0.05면 TCP 이동이 ~15cm라 윈도우 착탄
# ~1.4mm. 즉 0.1은 정확도를 내주고 release 타이밍 강건성을 사는 선택이다.
B_LO_DEG = 10.0                 # q_B ≥ 10° (위치 하한의 B 성분 — 2026-07-15 사용자 수정)
# 가속도 한계 = **3×속도한계** (2026-08-04 사용자 지시로 5.0 → 3.0).
# DT/Physics 비교 실험에서 세 모델이 같은 한계를 쓰도록 통일한 값이다.
# URDF velocity limit(L/U/B = 386.7/520/550 deg/s)이 GP8_QD_MAX와 사실상 같아
# URDF 기준이든 상수 기준이든 1160/1560/1650 deg/s²로 동일.
# 주의: 이 값이 바뀌면 warm DB가 무효화된다 (_warm_db_params의 qdd_lim).
QDD_LIM = 3.0 * GP8_QD_MAX
W1, W2 = 5.0, 1.0               # 시간 / 윈도우 정확도 가중치 (스펙의 w1, w2)
# W1 0.5→5.0 (2026-07-24 ablation): 구 0.5는 ω 페널티(기여 ~1.4)에 압도돼 사실상
# 무효(제거해도 t_f −1.9%)였음. W1 스윕에서 5.0이 스윗스팟 — t_f −40%(1.15→0.69s)
# 와 착탄 −23%(1.41→1.09mm)가 동시에 개선(시간 압박이 스윙을 짧고 단정하게).
# 대가는 ω 1.70→2.59 rad/s 증가. W1=100은 착탄 1.81mm로 다시 악화.
W_ACC = 1e6                     # 착탄오차 penalty 스케일 (m² 오차 → 비용).
# 2026-07-22 1e5→1e6 상향 (사용자 "랜딩 오차=0 제약식 가중치를 높여" — 정확도
# 최우선). ω를 hard→penalty로 완화(아래)해 문제가 부드러워진 덕에 1e6도 수렴 가능.
# 이력: 1e4(2026-07-16)→1e5(2026-07-21)→1e6(2026-07-22). 2026-07-24 1e7 시도했다
# 복귀 — rt=0.1 랜덤 release에서 W_ACC 상향이 조건수만 나빠뜨려 착탄 최악값이
# 오히려 증가(1e6 27.5→1e7 31.5mm). rt=0.05 복귀와 함께 1e6이 최적.
# 너무 크면 사실상 hard로 작동해 수렴성(조건수)을 죽임 — rt=0.05 스윕 (2026-07-16,
# 최악 쌍 수렴/윈도우오차): 1e7 0/6, 1e5 2/6·1.8mm, 1e4 5/6·6.2mm, 1e3 6/6·41mm.
# (rt=0.01 시절엔 1e7이 0.01mm/6~10s — 짧은 윈도우라 stiff해도 수렴했음.)
# ω_tcp → 0 은 hard 등식이 아니라 ω² penalty (2026-07-22 사용자): 윈도우 노드마다
# W_OMEGA·‖ω_tcp‖² 를 목적에 더한다. W_OMEGA는 opti.parameter라 아래 '가중치 사다리'로
# 강→약 시도 — 강한 가중치로 안 풀리면 다음 rung으로 내려가고, 마지막 0.0은 ω를
# 포기(정확도+시간만)해 ω 도입 이전의 검증된 공식화로 환원 → 항상 해 확보.
# **약한 penalty로 운영** (2026-07-22 사용자 "ω penalty 약하게 유지"): 실측상
# ω→0은 GRIP_OFF(발사점=CoM, v=jtimes로 ω×r 이미 정확 보정)와 중복이면서 wrist
# snap을 막아 정확도(33→46mm)·먼 코너 도달성을 오히려 해쳤음. rung 10 = 과한
# 회전만 살짝 억제(ω~1 rad/s 허용), 안 풀리면 0(GRIP_OFF만).
W_OMEGA_LADDER = (10.0, 0.0)
W_SENS = f"omega-pen-ladder-{int(W_OMEGA_LADDER[0])}"  # warm DB 유효성 마커 (w_sens 키)
T_BOUNDS = (0.3, 1.5)           # t_f 탐색 범위
REFINE_TOL = 1e-6               # 위치 한계 dense 검증 허용 위반 (rad) — IPOPT의
                                #   제약 잔차(~1e-8)보다 느슨해야 활성 bound에서
                                #   가짜 미수렴 경고가 안 남 (물리적으론 6e-5°)
MAX_REFINE = 6                  # adaptive refinement 최대 반복

# Cartesian 안전 엔벨로프 (2026-08-05 사용자 지시) — **TCP 한 점의 반평면 2개**.
#   x > CART_MIN_X : 로봇 앞쪽에 머문다   z > CART_MIN_Z : 바닥을 긁지 않는다
# **값은 throwing.py 한 곳에만 둔다** (TCP_X_MIN/TCP_Z_MIN). 여기·dt_gp8_env·
# 실기 게이트가 각자 복제하면 모델마다 다른 제약으로 심사받는다 — 2026-08-05
# 실측으로 확인했다 (구 원기둥 기준으로 거른 DT 학습데이터가 반평면 게이트에
# 걸렸다). 아래는 이 모듈 안에서 쓰던 이름을 유지하기 위한 별칭일 뿐이다.
CART_MIN_X, CART_MIN_Z = TCP_X_MIN, TCP_Z_MIN

# (구) 기둥 회피 원기둥 파라미터 — 2026-08-05 부터 **어떤 제약에도 쓰지 않는다**.
# 기준이 "wrist·로드 50/75%·TCP 4점이 기둥 축 세그먼트(z∈[0,COL_H])에서 COL_R 이상"
# 이었고, 위 엔벨로프와 포함관계가 아니다 (원기둥은 팔 전체 vs 기둥, 엔벨로프는
# TCP vs 앞쪽·바닥). 되살릴 일이 있을까 싶어 값만 남긴다.
COL_R = 0.18
COL_H = 0.55

# **카타시안(작업공간) 제약 전체 스위치** — 2026-07-31 사용자 "카타시안 제한은 모두
# 없애". False면 위의 Cartesian 엔벨로프가 NLP에서 빠지고, 남는 hard constraint는 전부
# 관절공간(위치/속도/가속도 한계)과 시간 변수뿐이다. 이 값은 warm DB 유효성 키
# (nlp_planner._warm_db_params의 'col')에 들어가므로, 되돌리면 예전 config의
# entry가 다시 선택된다. 주의: 기둥/바닥 침범 궤적이 그대로 해로 나올 수 있고,
# 시뮬 dry-run은 이제 그것을 '보고만' 하고 던지기를 계속한다
# (nlp_planner.CART_CHECK_BLOCKING=False).
CART_CONSTRAINTS = False

# 공기저항 flight 모델 (2026-07-21 사용자 스펙): 물체의 top-down point cloud에서
# 뽑은 축별 유효 단면적 기반 '비등방 2차 항력' — v̇ = g − c ⊙ (|v|·v),
# c_i = ρ_air·C_d·A_i/(2m) [1/m]. 폐형 포물선 대신 RK4 N_FLIGHT 스텝으로 비행을
# 적분하고, 비행시간 τ를 윈도우 노드별 '변수'로 두고 z(τ)=z_tgt 등식으로 정의.
# c=0이면 RK4가 등가속도를 '정확히' 적분하므로 (2차 다항) 기존 포물선과 일치.
# 목적은 항력 최소화가 아니라 '보상' — 항력 포함 착지점이 target과 일치하는 궤적.
N_FLIGHT = 8
GRIP_OFF = 0.02   # 발사점 = TCP + 로드 축 방향 2cm (흡착 물체 CoM 근사 —
#   2026-07-24 사용자 1cm→2cm 복귀. 판정이 CoM 기준이라 TCP 조준 시 ω×r 오버슛 bias가
#   생기는 문제의 계획단 보정: p_eff를 심볼릭으로 만들고 v=jtimes(p_eff)라
#   그 점의 강체 속도(ω×r 포함)가 자동으로 비행 초기조건이 됨)
FLIGHT_MODEL = f"rk4x{N_FLIGHT}-anisodrag-grip{int(1e3*GRIP_OFF)}mm"  # 유효성 마커

# clamped uniform knot vector (정규화 시간 u∈[0,1])
KNOTS = np.concatenate([np.zeros(DEGREE),
                        np.linspace(0.0, 1.0, N_CTRL - DEGREE + 1),
                        np.ones(DEGREE)])

# 위치 한계 = URDF(GP8_Q_MIN/MAX) ∩ 실기 한계 ∩ 사용자 규칙 (아래 순서대로 적용)
POS_LIMIT_MODE = "hull"   # 제어점 convex hull (2026-07-16) — warm DB 무효화 마커

# (1) 실기(실물 로봇) 관절 한계 — 2026-07-23 사용자 제공, planner 부호로 변환.
#     시뮬 URDF와 실기 '둘 다' 만족해야 실제 실행 가능하므로 교집합을 취한다.
#       관절   URDF          실기          교집합 지배
#       J1   [-170,+170]  [-170,+170]   동일
#       J2   [ -65,+145]  [ -65,+145]   동일
#       J3   [ -70,+190]  [ -70,+190]   동일
#       J5   [-135,+135]  [-135,+60.8]  ← 실기 (planner 하한 -60.8)
#       J6   [±360.6   ]  [±360     ]   ← 실기
Q_LO_REAL = np.deg2rad([-170.0, -65.0, -190.0, -190.0, -60.8, -360.0])
Q_HI_REAL = np.deg2rad([170.0, 145.0, 70.0, 190.0, 135.0, 360.0])
Q_LO = np.maximum(GP8_Q_MIN, Q_LO_REAL)
Q_HI = np.minimum(GP8_Q_MAX, Q_HI_REAL)

# (2) 사용자 규칙 — 관절별로 추가 축소 (전부 hard 위치 한계로 들어감)
Q_LO[4] = np.deg2rad(B_LO_DEG)      # B(손목) ≥ 10° — 손목이 바닥 쪽을 향하게
# S(베이스 yaw, index 0): |q_S| ≤ 60° (2026-07-23 사용자 규칙 — 실기 ±170°에서
# 축소). S는 SIGN=+1이라 planner/URDF 부호가 같아 그대로 적용된다.
Q_LO[0], Q_HI[0] = np.deg2rad(-60.0), np.deg2rad(60.0)
# L(어깨 pitch, index 1): URDF 기준 [−15°, +45°] (2026-07-31 사용자). L은 SIGN=+1이라
# planner 부호도 동일하다. FK 확인: q_L 음수 = 뒤로 젖힘 (TCP x 감소; −15°에서 x ~0.55m).
# (이력) 같은 날 [0,30] → [−15,30] → [−15,45]. 하한 0°는 무의미했다 (시작자세 IK가
# 실제로 쓰는 L은 15~28°라 하한이 걸리지 않음 — 무작위 시작점 실패율 2.0%로 동일).
# 걸리던 건 항상 상한이라 45°로 올려야 무더기 가장자리 물체가 도달 가능해진다.
Q_LO[1] = np.deg2rad(-15.0)
Q_HI[1] = np.deg2rad(45.0)
# U(팔 2번째 pitch 관절, index 2): **URDF 기준 [−45°, +30°]** (2026-07-31 사용자).
# 플래너 컨벤션은 부호가 뒤집힌다 (q_U_urdf = −q_U_planner, '위 = 음수 q_U',
# corr(z_TCP, −q_U)=0.99) → planner 범위 [−30°, +45°].
#   URDF +30° (팔올림 한도)  → planner 하한 −30°
#   URDF −45° (전완 내림 한도) → planner 상한 +45°
# (이력) 직전까지는 URDF·planner 모두 [−45,+45] 대칭이라 컨벤션 구분이 무의미했음.
# 이번 변경으로 팔올림이 45°→30°로 15° 더 조여진다.
Q_LO[2] = np.deg2rad(-30.0)
Q_HI[2] = np.deg2rad(45.0)
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


def fk_omega_sym(q, qd):
    """TCP 각속도 ω(q,q̇) = Σ q̇ᵢ·zᵢ(q) (base frame) — throwing.jacobian의 Jw·q̇."""
    axis_v = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}
    R = cs.MX.eye(3)
    w = cs.MX.zeros(3)
    for (xyz, ax), i in zip(_CHAIN, range(6)):
        w = w + qd[i] * (R @ cs.DM(axis_v[ax]))
        R = R @ _rot_sym(ax, q[i])
    return w


# ---------------------------------------------------------------------------
# 모듈 3: 목적 함수 구성 요소 (CasADi Function으로 모듈화)
# ---------------------------------------------------------------------------
_OBJ_SYM = None


def make_objective_functions_sym():
    """(q, q̇, τ, tgt, c) 입력의 심볼릭 Function 3종 (모듈 캐시):
      calc_landing_error(q,q̇,τ,tgt,c)        — 항력 포함 RK4 비행 후 ‖xy−tgt_xy‖²
      calc_z_residual(q,q̇,τ,tgt,c)           — z(τ) − tgt_z (비행시간 τ 정의 등식)
      calc_throwing_sensitivity(q,q̇,τ,tgt,c) — ‖∂xy/∂q‖²+‖∂xy/∂q̇‖² (τ 고정 근사)
    c(3-vector) = ρ·C_d·A_axis/(2m): top-down PC에서 산출 (sim의 drag_coeffs_from_pc).
    c=0이면 등가속 비행 = 구 폐형 포물선과 동일 (RK4가 2차 다항을 정확히 적분).
    TODO: 사용자 파일 수식 삽입 지점 — 항력/탄도 모델 교체는 f(state)만 바꾸면 됨.
    """
    global _OBJ_SYM
    if _OBJ_SYM is not None:
        return _OBJ_SYM
    q = cs.MX.sym("q", 6)
    qd = cs.MX.sym("qd", 6)
    tau = cs.MX.sym("tau")
    tgt = cs.MX.sym("tgt", 3)
    cdrag = cs.MX.sym("cdrag", 3)

    pts = fk_col_points_sym(q)           # [wrist, ·, ·, tcp]
    wrist, tcp = pts[0], pts[3]
    rod_len = _CHAIN[5][0][0]            # d6+tool (로드 축 길이 — 상수)
    p = tcp + (GRIP_OFF / rod_len) * (tcp - wrist)   # 발사점 = TCP+2cm(로드 축)
    v = cs.jtimes(p, q, qd)              # 발사점 강체 속도 (ω×r 자동 포함)

    g_vec = cs.DM([0.0, 0.0, -G])

    def f(state):                        # 비행 동역학: ṗ=v, v̇=g − c⊙(|v|v)
        vv = state[3:6]
        sp = cs.sqrt(cs.sumsqr(vv) + 1e-9)
        return cs.vertcat(vv, g_vec - cdrag * sp * vv)

    s = cs.vertcat(p, v)
    h = tau / N_FLIGHT
    for _ in range(N_FLIGHT):            # 고정 스텝 RK4
        k1 = f(s)
        k2 = f(s + h / 2 * k1)
        k3 = f(s + h / 2 * k2)
        k4 = f(s + h * k3)
        s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    xy = s[0:2]
    err2 = cs.sumsqr(xy - tgt[0:2])      # J_acc: 착지 xy 오차²
    zres = s[2] - tgt[2]                 # τ 정의: z(τ) = tgt_z
    S_q = cs.jacobian(xy, q)             # J_sens (τ 고정 근사 — 지배항)
    S_qd = cs.jacobian(xy, qd)
    sens = cs.sumsqr(S_q) + cs.sumsqr(S_qd)

    _OBJ_SYM = (cs.Function("calc_landing_error", [q, qd, tau, tgt, cdrag], [err2]),
                cs.Function("calc_z_residual", [q, qd, tau, tgt, cdrag], [zres]),
                cs.Function("calc_throwing_sensitivity",
                            [q, qd, tau, tgt, cdrag], [sens]))
    return _OBJ_SYM


def make_objective_functions(p_target, drag=None):
    """고정-target 래퍼: calc_landing_error(q,q̇,τ), calc_throwing_sensitivity(q,q̇,τ)."""
    F_err, F_z, F_sens = make_objective_functions_sym()
    q = cs.MX.sym("q", 6)
    qd = cs.MX.sym("qd", 6)
    tau = cs.MX.sym("tau")
    t = cs.DM(np.asarray(p_target, float))
    d = cs.DM(np.zeros(3) if drag is None else np.asarray(drag, float))
    return (cs.Function("calc_landing_error", [q, qd, tau], [F_err(q, qd, tau, t, d)]),
            cs.Function("calc_throwing_sensitivity", [q, qd, tau],
                        [F_sens(q, qd, tau, t, d)]))


# ---------------------------------------------------------------------------
# NLP 조립 + 풀이 (adaptive refinement 루프 포함)
# ---------------------------------------------------------------------------
def _assemble_nlp(opti, P_free, t_f, t_star, q0, v0c, vfc, tgt, u_pos_nodes, rt,
                  dragc, womega):
    """NLP 몸체 조립 (제약 1~5 + 기둥 회피 + 목적함수) — cold(_build_and_solve)와
    캐시된 파라메트릭 polish 솔버(_polish_solve)가 공유. q0/v0c/vfc/tgt/womega는
    DM 상수 또는 opti.parameter — 동일 수식이라 파라미터화가 그대로 성립한다.
    womega는 ω² penalty 가중치 (가중치 사다리로 조절 — parameter로 두면 그래프
    재조립 없이 rung만 갈아끼움). 전체 제어점 행렬 P(6×N_CTRL) 식을 리턴."""
    u_head = KNOTS[DEGREE + 1]
    u_tail = 1.0 - KNOTS[len(KNOTS) - DEGREE - 2]

    # -- 제약 1: 경계 기구학 조건 — 변수 소거 (auxiliary 등식 없음) --
    P1c = q0 + (u_head / DEGREE) * t_f * v0c                # q̇(0)=v0
    Pn = P_free[:, -1]
    Pn1 = Pn - (u_tail / DEGREE) * t_f * vfc                # q̇(t_f)=vf
    P = cs.horzcat(q0, P1c, P_free[:, :-1], Pn1, Pn)        # 전체 6×N_CTRL

    # -- 제약 3a: 위치 한계(+q_B≥10°, U∈[−30,45]°, L∈[0,30]°, |R|≤80°) — 제어점 hull --
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

    # -- 제약 3b: Cartesian 엔벨로프 — collocation 노드 (비선형 FK 제약이라 hull 불가) --
    # TCP 가 x > CART_MIN_X, z > CART_MIN_Z. 활성 구간은 release 창 끝(+버퍼)까지만 —
    # 감속 꼬리는 지하로 다이브하며 축 근처를 지나는 게 일상이고 (dry-run 꼬리
    # 교체가 처리), 전 구간에 걸면 무충돌 basin까지 잘라 multistart가 전멸한다
    # (2026-07-16 실측: bin9/11 96샘플 전멸). t*·t_f가 변수라 노드별 활성 여부가
    # 심볼릭 → smooth sigmoid 게이트 (act≈1: 창 이전, act≈0: 꼬리).
    # (TCP '바닥' 클리어런스는 hard로 걸면 0/148 전멸 — 초기해가 깊게 위반한 채
    #  시작해 restoration 즉사. 바닥 여유는 sim dry-run 가드가 담당 — 2026-07-15)
    # CART_CONSTRAINTS=False (2026-07-31 사용자)면 이 블록 전체가 빠진다 — 남는
    # hard constraint는 관절공간 한계와 시간 변수뿐 (u_pos_nodes는 그때 무의미).
    if CART_CONSTRAINTS:
        # 2026-08-05 사용자 지시: 카타시안 제약을 **TCP 안전 엔벨로프 하나로 통일**.
        # 구현은 구 기둥 원기둥(COL_R/COL_H, wrist·로드·TCP 4점)을 대체한 것이고,
        # 실기 게이트(gp8_control skills/thr_throw_skill.CART_MIN_X/Z)·DT 학습데이터
        # 필터(dt_gp8_env.trajectory_ok_cartesian)와 **같은 기준**이다.
        # 활성 구간도 실기 게이트와 맞춘다 — 게이트는 아크를 release 까지만 보므로
        # 여기서도 release 창 끝까지만 걸고 감속 꼬리는 푼다 (꼬리까지 hard 로 걸면
        # 무충돌 basin 까지 잘려 multistart 가 전멸한다 — 구 주석과 같은 이유).
        u_re = (t_star + rt / 2) / t_f + 0.08      # release 창 끝 + 버퍼 (u 단위)
        for u_c in u_pos_nodes:
            if float(u_c) == 0.0:
                continue  # u=0: q=P₀=q_start 상수/파라미터 (결정변수 없음 — Opti가 거부)
            q_c, _, _ = spline_qs(P, float(u_c), t_f)
            act = 1.0 / (1.0 + cs.exp(40.0 * (float(u_c) - u_re)))
            tcp = fk_col_points_sym(q_c)[-1]       # 4점 중 마지막 = TCP
            slack = (1.0 - act) * 10.0             # act→0 이면 사실상 비활성
            opti.subject_to(tcp[0] >= CART_MIN_X - slack)
            opti.subject_to(tcp[2] >= CART_MIN_Z - slack)

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
    F_err, F_z, F_sens = make_objective_functions_sym()
    offsets = np.linspace(-rt / 2, rt / 2, N_WIN)
    trap_w = np.full(N_WIN, rt / (N_WIN - 1)); trap_w[[0, -1]] *= 0.5
    J_win = cs.MX(0.0)
    taus = []
    for off, w in zip(offsets, trap_w):
        # clamp: 제약4는 '해'에서만 u_i∈[0,1] 보장 — IPOPT 중간 iterate가 범위를
        # 벗어나면 basis≡0이 되어 목적이 붕괴(gradient 소실)하므로 방어적 clamp
        u_i = cs.fmin(cs.fmax((t_star + off) / t_f, 0.0), 1.0)
        q_i, qd_i, _ = spline_qs(P, u_i, t_f)
        # 비행시간 τ_i: 변수 + z(τ)=z_tgt 등식으로 정의 (항력 때문에 폐형 근 없음)
        tau_i = opti.variable()
        taus.append(tau_i)
        opti.subject_to(opti.bounded(0.05, tau_i, 1.5))
        opti.subject_to(F_z(q_i, qd_i, tau_i, tgt, dragc) == 0)
        # ω_tcp → 0 은 penalty (2026-07-22 사용자 — 구 hard 등식 완화): 윈도우 노드
        # 무회전 release로 ω×r 오차원을 줄이되, hard로 걸면 basin이 죽어 안 풀리므로
        # womega·‖ω‖²를 목적에 더한다 (womega는 parameter — 가중치 사다리로 강→약).
        J_win = J_win + w * (W_ACC * F_err(q_i, qd_i, tau_i, tgt, dragc)
                             + womega * cs.sumsqr(fk_omega_sym(q_i, qd_i)))
    opti.minimize(W1 * t_f + W2 * J_win)
    return P, taus


_POLISH_SOLVERS = {}   # (rt, u_pos 노드 집합) → 조립된 파라메트릭 솔버 (재사용)


def _polish_solve(u_pos, rt, q_start, v0, vf, p_target, warm_data, drag):
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
        dragp = opti.parameter(3)
        womega = opti.parameter()
        P, taus = _assemble_nlp(opti, P_free, t_f, t_star, q0, v0p, vfp, tgt,
                                u_pos, rt, dragp, womega)
        opti.solver("ipopt", {"print_time": False, "expand": True},
                    {"max_iter": 3000, "print_level": 0, "sb": "yes",
                     "tol": 1e-4, "mu_init": 1e-6, "warm_start_init_point": "yes",
                     "warm_start_bound_push": 1e-9,
                     "warm_start_mult_bound_push": 1e-9,
                     "warm_start_slack_bound_push": 1e-9})
        h = dict(opti=opti, P_free=P_free, t_f=t_f, t_star=t_star,
                 q0=q0, v0=v0p, vf=vfp, tgt=tgt, drag=dragp, womega=womega,
                 P=P, taus=taus)
        _POLISH_SOLVERS[key] = h
    opti = h["opti"]
    opti.set_value(h["q0"], np.asarray(q_start, float))
    opti.set_value(h["v0"], np.asarray(v0, float))
    opti.set_value(h["vf"], np.asarray(vf, float))
    opti.set_value(h["tgt"], np.asarray(p_target, float))
    opti.set_value(h["drag"], np.asarray(drag, float))
    free_idx = list(range(2, N_CTRL - 2)) + [N_CTRL - 1]
    lam = warm_data.get("lam_g")
    # ω² penalty 가중치 사다리 (강→약, 첫 수렴 rung 채택; 마지막 0.0은 항상 수렴)
    last_exc = None
    for w_om in W_OMEGA_LADDER:
        opti.set_value(h["womega"], w_om)
        opti.set_initial(h["P_free"], warm_data["P"][:, free_idx])
        opti.set_initial(h["t_f"], warm_data["t_f"])
        opti.set_initial(h["t_star"], warm_data["t_star"])
        for t_i in h["taus"]:
            opti.set_initial(t_i, 0.45)
        if lam is not None and np.size(lam) == opti.ng:  # 제약 구성 동일할 때만 유효
            opti.set_initial(opti.lam_g, np.asarray(lam, float).ravel())
        try:
            sol = opti.solve()   # 실패 시 다음 rung으로
        except RuntimeError as e:
            last_exc = e
            continue
        return dict(P=np.array(sol.value(h["P"])), t_f=float(sol.value(h["t_f"])),
                    t_star=float(sol.value(h["t_star"])), J=float(sol.value(opti.f)),
                    lam_g=np.array(sol.value(opti.lam_g)).ravel(), w_omega=float(w_om))
    raise last_exc   # 사다리 전부 실패 — 호출측이 cold multistart로 fallback


def _spline_eval(P, t_f):
    """수치 해 (P, t_f)로 scipy BSpline 평가기 (q, q̇, q̈)(t) 생성 — 검증·플롯용."""
    spl = [BSpline(KNOTS, P[j], DEGREE) for j in range(6)]
    q_of = lambda t: np.array([s(t / t_f) for s in spl])
    qd_of = lambda t: np.array([s.derivative(1)(t / t_f) for s in spl]) / t_f
    qdd_of = lambda t: np.array([s.derivative(2)(t / t_f) for s in spl]) / t_f**2
    return q_of, qd_of, qdd_of


# position-only IK seed 목록 (yaw, L, U, B) — yaw=None이면 atan2(y, x)로 대체.
# 앞의 4개는 구버전 seed 그대로라, 그것으로 한계 안 해가 나오던 케이스는 자세가
# 바뀌지 않는다. 뒤의 6개는 L을 낮추고 U를 흔들어 좁아진 L∈[0,30]°·U(URDF)
# ∈[−45,30]° 안으로 들어오는 자세를 찾는 폴백.
_IK_START_SEEDS = (
    (0.2, 0.5, -0.2, 0.6), (0.2, 0.5, -0.2, 0.9),
    (0.2, 0.5, -0.2, 0.3), (0.2, 0.5, -0.2, 1.2),
    (None, 0.2, 0.0, 0.6), (None, 0.2, 0.3, 0.9), (None, 0.0, 0.3, 0.6),
    (None, 0.35, -0.4, 0.9), (None, 0.1, 0.5, 0.6), (None, 0.0, 0.0, 1.2),
)


def ik_start_pose(p_grasp):
    """p_grasp를 잡는 던지기 시작 자세 — **전 관절이 [Q_LO, Q_HI] 안**인 IK 해.

    position-only IK는 3차원 여유가 있어 seed를 바꾸면 다른 자세가 나온다.
    2026-07-31 이전 구현은 B 하한만 검사했는데, L∈[0,30]°·U(URDF)∈[−45,30]°로
    좁힌 뒤로는 한계 밖 해가 흔해졌다. q_start는 P₀로 '고정'되는 상수라 제어점
    convex hull 제약이 잡아주지 못하고, dense 검증에서 전 후보가 같은 값의
    '위치한계 잔존 위반'으로 기각돼 multistart가 통째로 전멸한다 (실측: 무작위
    배치 12던지기 중 3건). 여기서 한계 안 자세를 고르는 것이 근본 해결이다.

    한계 안 해가 하나도 없으면 ValueError — 그 (물체 위치, 한계) 조합에서는
    애초에 실행 가능한 시작 자세가 없다는 뜻이므로 조기에 자진신고한다.
    """
    p = np.asarray(p_grasp, float)
    yaw = float(np.arctan2(p[1], p[0]))
    best = None
    for s_yaw, s_l, s_u, s_b in _IK_START_SEEDS:
        qs, ok = ik_position(p, np.array([yaw if s_yaw is None else s_yaw,
                                          s_l, s_u, 0.0, s_b, 0.0]))
        if not ok or qs[4] < Q_LO[4] + 0.01:   # B 하한에서 살짝 떨어진 해만
            continue
        viol = float(max(np.max(Q_LO - qs), np.max(qs - Q_HI), 0.0))
        if viol <= 1e-9:
            return qs
        if best is None or viol < best[0]:
            best = (viol, qs)
    if best is None:
        raise ValueError(f"시작자세 IK 실패 (B≥{B_LO_DEG}° 해 없음): "
                         f"p_grasp={np.round(p, 3)}")
    j = int(np.argmax(np.maximum(Q_LO - best[1], best[1] - Q_HI)))
    raise ValueError(
        f"시작자세가 관절 한계 밖 — p_grasp={np.round(p, 3)}, "
        f"최소 위반 {np.rad2deg(best[0]):.2f}° (관절 {'SLURBT'[j]}, "
        f"q={np.round(np.rad2deg(best[1]), 1)}, 한계 "
        f"[{np.rad2deg(Q_LO[j]):.0f}, {np.rad2deg(Q_HI[j]):.0f}]°)")


def solve_throw_nlp(p_grasp, p_target, v0=None, vf=None,
                    release_time=RELEASE_TIME, q_start=None, verbose=False,
                    init=None, warm_data=None, drag=None):
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
    drag = np.zeros(3) if drag is None else np.asarray(drag, float)  # c=0 → 무항력
    rt = release_time

    if q_start is None:
        q_start = ik_start_pose(p_grasp)
    else:
        q_start = np.asarray(q_start, float)
        if not (np.all(q_start >= Q_LO - 1e-9) and np.all(q_start <= Q_HI + 1e-9)):
            raise ValueError("q_start가 위치 한계(B≥10°, U 팔올림≤30°, L≥0° 포함) 밖")

    free_idx = list(range(2, N_CTRL - 2)) + [N_CTRL - 1]   # 독립 제어점 열 인덱스

    def _build_and_solve(u_pos_nodes, warm):
        """제약 3의 위치 collocation 노드 집합을 받아 cold NLP 한 번 풀기.
        (full warm start polish는 캐시된 파라메트릭 솔버 _polish_solve가 담당.)"""
        opti = cs.Opti()
        P_free = opti.variable(6, N_CTRL - 3)    # 독립 제어점: P₂..Pₙ₋₂, Pₙ
        t_f = opti.variable()
        t_star = opti.variable()
        womega = opti.parameter()                # ω² penalty 가중치 (사다리로 조절)
        P, taus = _assemble_nlp(opti, P_free, t_f, t_star, cs.DM(q_start),
                                cs.DM(v0), cs.DM(vf),
                                cs.DM(np.asarray(p_target, float)),
                                u_pos_nodes, rt, cs.DM(drag), womega)

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
            P_init = np.linspace(q_start, q_start + dq, N_CTRL).T[:, free_idx]
            tf_init, ts_init = T0, chi0 * T0
        else:
            P_init = warm["P"][:, free_idx]
            tf_init, ts_init = warm["t_f"], warm["t_star"]

        # expand: MX→SX 그래프 전개 — 같은 수식·같은 iterate 경로로 평가만 ~2× 빨라짐
        # (2026-07-15 실측 14.0→7.1s, 해 완전 동일). jit은 컴파일 10분+라 부적합.
        opti.solver("ipopt", {"print_time": False, "expand": True},
                    {"max_iter": 3000, "print_level": 5 if verbose else 0, "sb": "yes"})
        # ω² penalty 가중치 사다리 (강→약, 첫 수렴 rung 채택; 마지막 0.0은 정확도+시간만
        # → ω 도입 이전 검증된 공식화라 basin이 feasible하면 항상 수렴)
        last_exc = None
        for w_om in W_OMEGA_LADDER:
            opti.set_value(womega, w_om)
            opti.set_initial(P_free, P_init)
            opti.set_initial(t_f, tf_init)
            opti.set_initial(t_star, ts_init)
            for t_i in taus:
                opti.set_initial(t_i, 0.45)
            try:
                sol = opti.solve()   # 실패 시 다음 rung으로
            except RuntimeError as e:
                last_exc = e
                continue
            return dict(P=np.array(sol.value(P)), t_f=float(sol.value(t_f)),
                        t_star=float(sol.value(t_star)), J=float(sol.value(opti.f)),
                        lam_g=np.array(sol.value(opti.lam_g)).ravel(),
                        w_omega=float(w_om))
        raise last_exc   # 사다리 전부 실패 (마지막 0.0도 실패 = 진짜 나쁜 basin)

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
                out = _polish_solve(u_pos, rt, q_start, v0, vf, p_target,
                                    warm_data, drag)
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
               p_target=p_target, release_time=rt, v0=v0, vf=vf, drag=drag)
    return out


# ---------------------------------------------------------------------------
# 검증 + 데모 (scipy BSpline으로 '독립' 재구성해 교차검증)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .throwing import G_VEC, jacobian, landing_error, launch_state

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

    # --- 윈도우 착탄 오차 + TCP 각속도 (ω=0 hard — 2026-07-22 대체) ---
    for tr in t_st + np.linspace(-rt / 2, rt / 2, 5):
        q, qd = q_of(tr), qd_of(tr)
        p, v = launch_state(q, qd)           # 발사점(TCP+GRIP_OFF) — planner와 동일
        w_tcp = jacobian(q)[1] @ qd
        print(f"  release t={tr:.4f}s → 착탄오차 {1e3*landing_error(p, v, TGT):6.2f} mm, "
              f"|ω|={np.linalg.norm(w_tcp):.4f} rad/s")

    # --- 착탄오차 프로파일: t*가 argmin에 앉았는지 ---
    prof_t, prof = [], []
    for t in np.linspace(0.02, t_f, 500):
        q, qd = q_of(t), qd_of(t)
        p, v = launch_state(q, qd)
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
            p, v = launch_state(q, qd)
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

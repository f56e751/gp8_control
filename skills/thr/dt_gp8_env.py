"""
THR GP8 rig용 Decision Transformer 학습 환경 (Thr_DT 논문 절차의 env 교체판).

왜 필요한가 — Thr_DT(/PublicSSD/ryugaeun/Thr_DT)의 사전학습 가중치를 THR sim에
그대로 이식하면 0/12다 (2026-07-30 측정). 원인 두 가지를 진단했다:
  ① Thr_DT의 평면 시뮬 env(env/throw_env.py)에는 관절 '위치' 한계가 없다
     (속도 한계만). 학습된 정책은 평면 t₃를 −187°까지 휘두르는데, 이는 GP8 U축
     한계(planner 프레임 +70°)를 릴리즈 2스텝 전부터 위반한다. 한계로 클립하면
     스윙이 망가져 물체가 로봇 뒤(x<0)로 날아간다.
  ② 그 평면 시뮬의 도달 능력 자체가 THR rig에 못 미친다. GP8 위치 한계를 걸고
     무작위 스윙 600회를 굴리면 최대 착지 0.945m (한계 없이도 1.31m)인데,
     THR bin은 1.20~1.70m다 — 학습을 잘 해도 구조적으로 도달 불가.
논문 §5.1은 시뮬 동역학 파라미터가 "arbitrary while in the same scale of the
real one"이라고 명시하므로, Thr_DT의 평면 시뮬은 THR의 GP8 rig와 '다른 로봇'인
셈이다. 따라서 가중치를 이식하는 대신 **논문의 방법론(무작위 OU 데이터 수집 →
HER 목표 재라벨 → DT 오프라인 학습)을 THR rig에 그대로 적용**한다. 이 파일은
그때 쓰는 env이고, Thr_DT의 model/trainer/HER/OU 코드는 수정 없이 재사용한다
(dt_train_gp8.py가 env만 주입).

논문 규약 유지 (Thr_DT/config.py와 동일 출처):
  · 10 Hz 제어, 최대 1초 = 10 스텝                                  [§5.1]
  · 평면 던지기: GP8의 2·3·5번 관절(L/U/B)만 사용, 던지는 방향은
    1번 관절(S)로 해석적 설정 — 이 env는 S=0 평면에서 학습한다      [§5.1]
  · 상태 (θ_L, θ_U, θ_B, θ_gr) + 목표거리 d_g, 행동 (ω_L, ω_U, ω_B, a_gr) [§4.1]
  · 위치 명령 θ_t = θ_{t−1} + ω_t/f (Taylor 근사) + 1차 actuator lag  [식 5.1]
  · 그리퍼는 a_gr ≤ τ에서 열림 (τ = 학습셋 그리퍼 액션 평균)         [§5.1]
  · 희소 보상 ±1 (반경 ρ), 착지점은 첫 낙하 지점                     [식 4.2, §4.1]
  · 비행은 무항력 포물선 — 시뮬(THR)은 quadratic drag이므로 갭이 남지만,
    NLP(PLAN_WITH_DRAG=False)·Physics-only도 같은 무항력 가정이라 공정하다.

THR rig로 교체되는 부분 (모두 실제 GP8 값):
  · FK/Jacobian: throwing.py의 6축 GP8 체인 (S=0, R=T=0 평면 단면)
  · 최대 속도: GP8_QD_MAX[L,U,B] = (385, 520, 550) deg/s
    — 논문이 명시한 값과 정확히 일치하므로 규약 변경이 아니다
  · 위치 한계: GP8_Q_MIN/MAX[L,U,B] = L[−65,145] U[−190,70] B[−135,135]°
    (Thr_DT env에 없던 제약 — 이걸 넣는 것이 ①의 수정)
  · 목표 거리 범위: THR bin 실측 범위를 포함하도록 DG_MIN/DG_MAX
"""

import os

import numpy as np

from .throwing import (G, GP8_Q_MAX, GP8_Q_MIN, GP8_QD_MAX, TCP_X_MIN,
                      TCP_Z_MIN, fk_pos, fk_frames,
                      jacobian)

# planner 프레임 6축 중 평면 던지기에 쓰는 3축 (L, U, B) — 논문 §5.1의 2·3·5번
PLANAR_IDX = [1, 2, 4]

# 위치 한계 두 가지 (2026-08-03 사용자 지시로 기본을 NLP과 동일하게 맞춤).
#   'urdf' : GP8 URDF 원 한계 — 기구학적으로 가능한 전 범위
#   'nlp'  : THR NLP 플래너가 실제로 쓰는 한계 (throw_nlp.Q_LO/Q_HI의 평면 3축).
#            URDF ∩ 실기 한계에 사용자 작업 규칙까지 적용한 값이라 훨씬 좁다.
#            여기에 하드코딩한 이유: throw_nlp는 casadi를 import하는데 DT 실행
#            인터프리터(Thr_Phy/.venv)에는 casadi가 없다. throw_nlp.py를 고치면
#            아래 값도 같이 고쳐야 한다(검증: tests 없이 수동 대조 2026-08-03).
#              L(idx1) [-15, 45] deg   ← throw_nlp Q_LO[1]/Q_HI[1]
#              U(idx2) [-30, 45] deg   ← throw_nlp Q_LO[2]/Q_HI[2] (URDF 기준
#                                        [-45,+30]의 부호 반전 — planner 컨벤션)
#              B(idx4) [ 10,135] deg   ← B_LO_DEG=10, 실기 상한 135
# ---------------------------------------------------------------------------
# 카티시안 작업영역 제약 (2026-08-05 사용자 지정): TCP가 x > 0.20 m, z > 0.02 m.
# 로봇 앞쪽(기둥 반대편)에 머물고 바닥을 긁지 않는다는 뜻이다. NLP도 같은 조건을
# 받도록 맞춘다. 판정은 궤적의 모든 스텝에서.
def tcp_xz(q3):
    """평면 3축 자세의 TCP (x, z) [m]. yaw=0 기준 — 회전은 x를 반경으로 보내므로
    x 하한 판정에 그대로 유효하다."""
    p = fk_pos(q6(q3))
    return float(p[0]), float(p[2])


def state_ok_cartesian(q3):
    x, z = tcp_xz(np.asarray(q3, float)[:3])
    return x > TCP_X_MIN and z > TCP_Z_MIN


def trajectory_ok_cartesian(states):
    """궤적의 모든 자세가 TCP x > 0.20, z > 0.02 를 만족하는가."""
    return all(state_ok_cartesian(s) for s in states)


def trajectory_limits(Qp, dt, cfg=None):
    """노드 궤적 → **컨트롤러가 실제로 실행하는 곡선**에서 관절각·속도·가속도·
    Cartesian 을 한 번에 검사한다 (2026-08-05 사용자 지시).

    10 Hz 노드만 봐서는 부족하다: 노드 사이는 MotoROS 3차 Hermite 가 그리므로
    위치가 노드 밖으로 넘칠 수 있고, 속도·가속도는 노드에 아예 나타나지 않는다.
    실측(v9, d_g=1.35): 노드 기준 평균가속도는 한계의 1.18배인데 실행 곡선의
    순간가속도는 4.55배였다 — 3차 곡선이 구간 시작에서 4배로 튀기 때문이다.

    데모 수집이 이 함수를 통과한 궤적만 담아야 학습 분포 자체가 실행 가능해진다.
    리턴: dict(ok, q, qd, qdd, cart, 각 항목의 한계 대비 최대비).
    """
    cfg = GP8Config() if cfg is None else cfg
    Qp = np.asarray(Qp, float)
    qdd_lim = float(getattr(cfg, "qdd_factor", 3.0)) * np.asarray(QD_MAX, float)
    if len(Qp) < 2:
        return dict(ok=True, q=0.0, qd=0.0, qdd=0.0, cart=True)
    t, Q, Qd = controller_curve(Qp, dt)
    Qdd = np.gradient(Qd, t, axis=0)

    # 위치: 한계 안쪽이어야 한다. Q_LO/Q_HI 는 대칭이 아니므로 여유분으로 잰다.
    # 위치: 초과량을 rad 로 직접 잰다. 노드가 한계에 얹히면 3차 곡선이 1e-12
    # 수준으로 넘칠 수 있어 상대비로 재면 전부 위반이 된다 — 0.01° 허용.
    # 허용오차. 가속도는 상자가 한계에 **정확히** 얹히도록 투영하므로 비율이
    # 1.0000000x 로 나온다 — 상대 1e-6 을 준다. 위치는 구간 내부 극값 때문에
    # 노드 밖으로 0.2~0.5° 넘칠 수 있었는데, 상자가 Q_MARGIN(1°) 안쪽에서만
    # 뽑도록 고쳐 실제 한계는 넘지 않는다 — 그래서 여기는 엄격히 본다.
    Q_TOL = np.deg2rad(0.01)
    A_TOL = 1.0 + 1e-6
    r_q = float(np.maximum(Q_LO - Q, Q - Q_HI).max())               # rad
    r_qd = float((np.abs(Qd) / np.asarray(QD_MAX, float)).max())
    r_qdd = float((np.abs(Qdd) / qdd_lim).max())
    cart = all(state_ok_cartesian(q) for q in Q)
    return dict(ok=bool(r_q <= Q_TOL and r_qd <= A_TOL and r_qdd <= A_TOL and cart),
                q=r_q, qd=r_qd, qdd=r_qdd, cart=cart)


def trajectory_ok_limits(Qp, dt, cfg=None):
    """`trajectory_limits` 의 bool 축약."""
    return trajectory_limits(Qp, dt, cfg)["ok"]


LIMIT_MODE = os.environ.get("DT_GP8_LIMITS", "nlp")
Q_LO_NLP = np.deg2rad([-15.0, -30.0, 10.0])
Q_HI_NLP = np.deg2rad([45.0, 45.0, 135.0])
if LIMIT_MODE == "nlp":
    Q_LO, Q_HI = Q_LO_NLP, Q_HI_NLP
else:
    Q_LO = GP8_Q_MIN[PLANAR_IDX]
    Q_HI = GP8_Q_MAX[PLANAR_IDX]
QD_MAX = GP8_QD_MAX[PLANAR_IDX]          # (385, 520, 550) deg/s — 논문과 일치

# 목표 거리 범위: THR bin 조준점 수평거리는 1.20~1.70 m. 학습 범위는 그보다
# 넓게 잡아 경계 근처의 외삽을 피한다 (논문 [0.5, 2.0] m 정신 유지).
DG_MIN, DG_MAX = 0.6, 2.0


SIM_DT = 1.0 / 240.0        # THR sim 스텝 (sim_env.DT와 같아야 함)

# ---------------------------------------------------------------------------
# 노드 적분 규약 (2026-08-05).
#
# 논문 식 (5.1)은 직사각 적분 q_k = q_{k−1} + ω_k·Δt 다. 그러면 구간 k의
# **평균** 속도가 ω_k인데, GP8 공식 보간(MotoROS 3차 Hermite)에 넘기는 waypoint
# 속도는 양 끝이 (ω_{k−1}, ω_k)다. 곡선은 이 둘을 동시에 맞추려고 구간 중간에서
# 부풀 수밖에 없고, 그 대가가
#     q̈(0) = 4·(ω_k − ω_{k−1})/Δt      ← 명령한 가속도의 **4배**
# 와 노드 밖 위치 오버슛이다 (실측: 무작위 120개 중 가속도로 전멸, 가중치를
# 4로 낮추면 이번엔 위치로 64/120 탈락).
#
# 사다리꼴 적분 q_k = q_{k−1} + (ω_{k−1} + ω_k)·Δt/2 로 바꾸면 Hermite 계수가
#     a1 = (ω_k − ω_{k−1})/Δt ,  a2 = 0
# 이 되어 구간 내부가 **정확히 등가속도**다. 증폭이 0이고, 속도는 노드에서
# 정확히 ω_k, 가속도는 정확히 명령한 변화율이다. 즉 수집 단계에서 건 한계가
# 실행 곡선에서 그대로 성립한다 — 사용자 요구(그 지점 속도 = DT 출력 각속도 +
# 데모 자체가 한계를 지킬 것)를 동시에 만족하는 유일한 규약이다.
#
# 논문과의 차이: 적분 규칙만 1차 → 2차 정확도로 올린 것이고, 상태·행동·보상·
# 학습 절차는 그대로다. 직사각으로 되돌리려면 TRAPEZOID=False.
# waypoint 속도 규약 (2026-08-05 사용자 지시: 공식 두 규약을 다 만들고 실기 비교).
# 공식 저장소 MaxorPaxor/ThrowBot 안에 규약이 두 개 있고, 어느 쪽도 이 rig 에서
# 검증된 적이 없다. 환경변수 DT_GP8_WP_VEL 로 고른다.
#
#   'zero'      공식 **DT 실행 경로** — real_robot/robot_env_dt_real.py:170-175
#               위치는 식 (5.1) 직사각, waypoint 속도는 전부 0.
#               실측(CEM 1.36 m 궤적): 릴리즈 각속도 0, 가속도 한계의 9.13배,
#               실제 사거리 0.680 m. 논문은 이 격차를 실기 파인튜닝으로 흡수한다
#               ("far-from-reality simulation", 초록).
#   'segment'   공식 **녹화 재생 경로** — control_real_gp8.py:107-115
#               위치는 직사각, waypoint 속도는 (p − p_old)/dt.
#               실측: 릴리즈 (59.6, −145.5, −247.5)°/s, 가속도 3.75배, 사거리 1.482 m.
#   'trapezoid' 여기서 유도한 규약 — 위치는 사다리꼴, waypoint 속도는 ω_k 복원.
#               MotoROS 계수에서 3차항이 0이 되어 구간 내부가 정확히 등가속도.
#               실측: 릴리즈 (85.4, −67.5, −330.0)°/s, 가속도 1.00배, 사거리 1.279 m.
#
# 셋 다 보간은 동일한 MotoROS 3차 Hermite(sim/gp8_interp.py)를 지난다 — 저자 포크
# MaxorPaxor/motoman_ps 의 MotionServer.c:1414-1420 과 계수가 일치함을 확인했다.
# 기본값 = 'segment' (2026-08-05 사용자 지시). 공식 저장소의 두 규약 중
# 실제로 던져지는 쪽이고, 식 (5.1) 직사각 적분을 그대로 쓴다.
WP_VEL_MODE = os.environ.get("DT_GP8_WP_VEL", "segment")
assert WP_VEL_MODE in ("zero", "segment", "trapezoid"), WP_VEL_MODE

# 적분 규칙은 waypoint 규약과 짝이다 — 공식 두 경로는 식 (5.1) 직사각을 쓴다.
TRAPEZOID = (WP_VEL_MODE == "trapezoid")


# ---------------------------------------------------------------------------
# 실현 가능한 액션 상자 (2026-08-05 사용자 지시: "데모를 애초에 관절 각도·속도·
# 가속도 한계까지 고려해서 뽑아라").
#
# 종전 수집기는 액션을 먼저 뽑아 **그대로 저장**하고 한계는 step() 안에서 사후에
# 클립했다. 실측(무작위 300 에피소드, 1555 스텝):
#     가속도 한계가 명령을 바꾼 스텝 60.5% / 위치 한계 64.2%
#     |기록 액션 − 실현 액션| 평균 0.222, 최대 1.000 (라벨은 최대속도인데
#     실제로는 관절이 한계에 박혀 정지)
# 즉 라벨의 절반이 실행되지 않은 값이라, DT는 존재하지 않는 행동↔결과 대응을
# 배우고 있었다. 그래서 **한계를 액션 생성 단계로 옮긴다** — 아래 상자 안에서만
# 액션을 뽑으면 명령이 곧 실현값이므로 라벨이 정확해진다.
#
# 상자는 세 한계의 교집합이다:
#   속도   |ω| ≤ ω_max                                       (a ∈ [−1,1])
#   가속도 ω ∈ ω_prev ± a_max·Δt,  a_max = qdd_factor·ω_max
#   위치   ① 한 스텝 안: q + ω·Δt ∈ [Q_LO, Q_HI]
#          ② 제동 여유: 이번 스텝 뒤에도 a_max로 한계 전에 멈출 수 있어야 한다
#             ω²/(2a) ≤ (Q_HI − q) − ωΔt  ⇒  ω ≤ −aΔt + √((aΔt)² + 2a(Q_HI−q))
#          ②가 없으면 전속으로 한계에 접근한 뒤 그 자리에서 멈출 수밖에 없어
#          (=가속도 한계 위반) 종전의 '박히는' 궤적이 그대로 재현된다.
# 위치 한계 안전 여유 [rad]. 상자는 **노드** 위치를 묶지만, 관절이 구간 안에서
# 부호를 바꾸면 정점이 노드가 아니라 구간 내부에 생긴다 (등가속 포물선의 꼭짓점).
# 그 초과량은 ω_prev 가 작을 때만 남고 실측 최대 0.52° 였다 — 1° 여유로 덮는다.
Q_MARGIN = np.deg2rad(1.0)


def feasible_qdot_box(q3, qdot_prev, dt, lo=None, hi=None, cfg=None):
    """자세 q3, 직전 명령속도 qdot_prev에서 이번 스텝에 낼 수 있는 ω 구간."""
    lo = (Q_LO + Q_MARGIN) if lo is None else lo
    hi = (Q_HI - Q_MARGIN) if hi is None else hi
    cfg = GP8Config() if cfg is None else cfg
    q = np.asarray(q3, float)[:3]
    wp = np.asarray(qdot_prev, float)[:3]
    # 실행 곡선 기준 (ACC_RECON_GAIN 참고). 사다리꼴 적분에서는 3차 곡선이
    # 정확히 등가속도라 증폭이 없어 1.0이다. 제동 여유도 같은 a_max 로
    # 계산해야 '낼 수 있는 감속' 과 '멈출 수 있다는 가정' 이 일치한다.
    a_max = float(cfg.qdd_factor) * QD_MAX / ACC_RECON_GAIN
    d_hi = np.maximum(hi - q, 0.0)
    d_lo = np.maximum(q - lo, 0.0)
    # 제동 여유: 이번 스텝을 ω로 끝낸 뒤에도 a_max로 한계 전에 멈출 수 있어야
    # 한다.  사다리꼴이면 이번 스텝 이동이 (ω_prev + ω)Δt/2 이므로
    #     (ω_prev + ω)Δt/2 + ω²/(2a) ≤ d_hi
    #  ⇒  ω² + aΔt·ω + aΔt·ω_prev − 2a·d_hi ≤ 0
    #  ⇒  ω ≤ −aΔt/2 + √((aΔt/2)² − aΔt·ω_prev + 2a·d_hi)
    # 이 조건이 한 스텝 위치 조건(ω²/(2a) ≥ 0 이므로)을 이미 포함한다.
    # 직사각이면 이동이 ωΔt 이라 √ 안이 (aΔt)² + 2a·d_hi 인 종전 식이 된다.
    if TRAPEZOID:
        disc_hi = (0.5 * a_max * dt) ** 2 - a_max * dt * wp + 2.0 * a_max * d_hi
        disc_lo = (0.5 * a_max * dt) ** 2 + a_max * dt * wp + 2.0 * a_max * d_lo
        brake_hi = -0.5 * a_max * dt + np.sqrt(np.maximum(disc_hi, 0.0))
        brake_lo = -0.5 * a_max * dt + np.sqrt(np.maximum(disc_lo, 0.0))
        # 판별식이 음수 = 이미 되돌릴 수 없는 상태 → 최대 감속만 허용
        brake_hi = np.where(disc_hi < 0.0, wp - a_max * dt, brake_hi)
        brake_lo = np.where(disc_lo < 0.0, -(wp + a_max * dt), brake_lo)
    else:
        brake_hi = -a_max * dt + np.sqrt((a_max * dt) ** 2 + 2.0 * a_max * d_hi)
        brake_lo = -a_max * dt + np.sqrt((a_max * dt) ** 2 + 2.0 * a_max * d_lo)
    w_hi = np.minimum.reduce([QD_MAX, wp + a_max * dt, brake_hi])
    w_lo = np.maximum.reduce([-QD_MAX, wp - a_max * dt, -brake_lo])
    # 교집합이 빌 수 있는 경우(수치 오차, 또는 이전 스텝이 이미 위반) —
    # 하드웨어 진실인 가속도 상자를 우선하고 그 안에서 위치에 가장 가까운 값.
    bad = w_lo > w_hi
    if np.any(bad):
        mid = 0.5 * (w_lo + w_hi)
        w_lo = np.where(bad, mid, w_lo)
        w_hi = np.where(bad, mid, w_hi)
    return w_lo, w_hi

# release 후 팔로우스루 노드 배수 — 궤적이 release에서 끝나면 지터의 '+' 방향이
# 궤적 밖으로 클립되어 무효가 되므로 coast 1스텝 + 감속 2스텝을 덧붙인다.
# dt_cem.traj_from_seq과 **같은 규칙**을 써야 학습(env)과 실행(시뮬)이 일치한다.
FOLLOWTHROUGH = (1.0, 0.5, 0.15)


def followthrough_nodes(q_last, qd_seg, dt, lo=None, hi=None, cfg=None):
    """release 직후 스텝 노드들.

    감속 배수 FOLLOWTHROUGH를 그대로 쓰면 Δω가 최대 0.5·ω_max라 가속도 한계
    (0.3·ω_max/스텝)를 넘는다 — 릴리즈 직후 구간만 한계를 면제받는 셈이라
    앞뒤가 맞지 않는다. 그래서 **감속도 실현 가능한 액션 상자 안에서** 만든다
    (2026-08-05). 위치 한계는 제동거리까지 고려한 상자가 이미 지키므로 여기서
    별도 클립은 하지 않는다."""
    # 상자와 **같은** 마진을 쓴다 — 여기만 전 범위를 쓰면 감속 구간에서 구간
    # 내부 극값이 한계를 0.5° 넘겼다 (실측).
    lo = (Q_LO + Q_MARGIN) if lo is None else lo
    hi = (Q_HI - Q_MARGIN) if hi is None else hi
    cfg = GP8Config() if cfg is None else cfg
    q = np.asarray(q_last, float).copy()
    w0 = np.asarray(qd_seg, float)
    w = w0.copy()
    out = []
    for f in FOLLOWTHROUGH:
        w_lo, w_hi = feasible_qdot_box(q, w, dt, lo=lo, hi=hi, cfg=cfg)
        w_new = np.clip(w0 * f, w_lo, w_hi)
        dq = ((w + w_new) * 0.5 if TRAPEZOID else w_new) * dt
        q = np.clip(q + dq, lo, hi)
        w = w_new
        out.append(q.copy())
    return out


# ---------------------------------------------------------------------------
# 10 Hz 노드 → 실행 궤적: **GP8 공식 보간법**(MotoROS 3차 Hermite)을 쓴다.
# (2026-08-05 사용자 지시. 종전 선형보간은 위치는 연속이나 구간마다 속도가
# 상수여서 노드에서 속도가 계단식으로 점프했고, 재생 가속도가 38,560 deg/s²까지
# 튀었다 — 관절 한계 1,650의 23배이고 이것이 렌더링의 "툭툭 끊김"이다.)
#
# waypoint = DT의 10 Hz 노드 그 자체이고, 각 waypoint의 속도는 **그 지점에서
# DT가 출력한 각속도** ω_k 다. tau_act=0이므로 ω_k = (q_k − q_{k−1})/dt로
# 정확히 일치한다(실측 100%). 즉 보간이 속도를 왜곡하지 않고, 노드에서의
# 곡선 도함수가 DT 출력과 같다.
#
# 구현은 sim/gp8_interp.py(실기 MotoROS `Ros_MotionServer_JointTrajDataToIncQueue`
# 이식)를 그대로 호출한다 — **env와 planner가 같은 함수를 써야** 학습 착탄과
# 실행 착탄이 어긋나지 않는다.


FOLLOWTHROUGH_SMOOTH = True     # False면 구 선형보간 (재현/비교용)

# 3차 보간의 가속도 증폭 배수 (2026-08-05).
# 구간 [k,k+1] 의 waypoint 속도가 V_k = s_{k−1}, V_{k+1} = s_k (s=구간 기울기)
# 인데 곡선의 **평균** 속도는 s_k 여야 한다. 시작이 s_{k−1} 로 느리므로 중간에
# s_k 위로 부풀 수밖에 없고, 그래서
#     q̈(u) = [(6−12u)·Δq/dt + (6u−4)·V_k + (6u−2)·V_{k+1}] / dt
#     q̈(0) = (6s_k − 4s_{k−1} − 2s_k)/dt = **4·(s_k − s_{k−1})/dt**
# 노드 평균가속도의 4배다 (4 ms 샘플이면 u=0.04 → 3.76배, 실측 3.75와 일치).
#
# 사다리꼴 적분(TRAPEZOID)이면 3차 곡선이 구간 내 등가속도라 증폭이 없다 → 1.0.
# 직사각 적분으로 되돌리면 위 유도대로 4.0이어야 한다.
ACC_RECON_GAIN = 1.0 if TRAPEZOID else 4.0


def node_velocities(Qp, dt):
    """노드 k에서의 각속도 ω_k — **그 지점에서 DT가 출력한 값**.

    사다리꼴 적분에서는 q_k − q_{k−1} = (ω_{k−1} + ω_k)·Δt/2 이므로 팔은 항상
    정지에서 출발한다(ω_0 = 0)는 사실로부터 정확히 복원된다:
        ω_k = 2(q_k − q_{k−1})/Δt − ω_{k−1}
    직사각 적분이면 ω_k = (q_k − q_{k−1})/Δt 로 그대로 읽힌다.
    """
    Qp = np.asarray(Qp, float)
    V = np.zeros_like(Qp)
    if len(Qp) < 2:
        return V
    if WP_VEL_MODE == "zero":
        return V                                  # 공식 DT 경로: 전부 0
    if WP_VEL_MODE == "segment":
        # 공식 재생 경로. 직사각 적분에서 (q_k − q_{k−1})/Δt = ω_k 이므로
        # 이게 곧 "그 노드에서 DT 가 출력한 각속도"다. V[0] 은 0 — 팔은 정지에서
        # 출발하므로(ω_0 = 0), V[1] 을 복사하면 t=0 에 이미 움직이는 셈이 된다.
        V[1:] = (Qp[1:] - Qp[:-1]) / dt
        return V
    for k in range(1, len(Qp)):                   # 사다리꼴: ω_k 복원
        V[k] = 2.0 * (Qp[k] - Qp[k - 1]) / dt - V[k - 1]
    return V


def controller_curve(Qp, dt, out_dt=SIM_DT, period=None, V=None):
    """노드 → GP8 컨트롤러가 실제로 그리는 궤적 (t, q, qd).
    waypoint 주기 = 노드 주기(1/dt Hz), 보간 주기 = INTERP_PERIOD(4 ms).
    V를 주면 그 노드 속도를 그대로 쓰고, 없으면 node_velocities로 복원한다."""
    from .gp8_interp import INTERP_PERIOD, controller_track
    Qp = np.asarray(Qp, float)
    t_nodes = np.arange(len(Qp)) * dt
    V = node_velocities(Qp, dt) if V is None else np.asarray(V, float)
    return controller_track(t_nodes, Qp, V, traj_hz=1.0 / dt,
                            period=INTERP_PERIOD if period is None else period,
                            out_dt=out_dt)


def q6(q3, yaw=0.0):
    """평면 3축 (θ_L, θ_U, θ_B) → GP8 planner 6축 (S=yaw, R=T=0)."""
    q = np.zeros(6)
    q[0] = yaw
    q[PLANAR_IDX] = np.asarray(q3, float)
    return q


class GP8Config:
    """Thr_DT SimConfig과 같은 필드 이름을 갖는 THR rig 설정 (duck typing).
    Thr_DT의 collect_data/train_dt_offline/evaluate_dt가 cfg에서 읽는 값들을
    그대로 제공한다."""

    # --- 제어 루프 [§5.1] ---
    update_rate = 10.0
    total_time = 1.0
    smooth_factor = 0.0              # 속도 상보필터 비활성 [repo]

    # --- 관절 [§5.1] + GP8 실측 한계 ---
    joint_names = ('joint_2_l', 'joint_3_u', 'joint_5_b', 'finger_joint')
    max_speed_deg = tuple(np.rad2deg(QD_MAX))     # (385, 520, 550)
    max_speed_factor = 1.0

    # --- 목표/보상 ---
    dg_min = DG_MIN
    dg_max = DG_MAX
    target_radius = 0.1             # ρ [repo env/robot_env_dt.py:31]
    gripper_thresh = 0.8385         # τ 초기값 (학습셋 평균으로 대체됨) [§5.1]

    # --- 시작 자세 (planner 프레임 L, U, B) ---
    # Thr_DT home (t₂,t₃,t₅)=(0.5,−0.3,−1.5)의 GP8 대응 자세 (부호 규약
    # q_L=t₂, q_U=−t₃, q_B=−t₅ — dt_planner.py의 FK 대조로 검증). GP8에서
    # TCP=(0.271, 0, 0.193)로 물체를 든 시작 높이(P_START_Z=0.20)와 맞고,
    # 세 관절 모두 위치 한계 안이다.
    home_pose = (0.5, 0.3, 1.5)

    # --- 적분/구동 동역학 ---
    substeps = 10
    # tau_act 제거 (2026-08-05 사용자 지시). 종전 0.05 s의 1차 지연 때문에
    # 실현 구간속도가 명령 ω의 89.3%로 감쇠했다 — 식 (5.1) θ_t = θ_{t−1} + ω_t/f
    # 를 그대로 따르면 그럴 이유가 없다. 0이면 명령이 그대로 실현된다
    # (관절 위치/속도/가속도 한계만 걸린다).
    tau_act = 0.0
    gravity = G                      # 9.81 — THR sim과 동일

    # 가속도 한계: |ω_t − ω_{t−1}| ≤ qdd_max·Δt, qdd_max = 5×속도한계.
    # 논문에도 공식 코드에도 없다 (Gazebo에선 링크 관성이 알아서 강제). 실제
    # 드라이브는 100 ms 안에 속도를 불연속으로 못 바꾸므로 명시적으로 건다.
    # 계수 3.0 = **속도 한계(URDF)의 3배** (2026-08-03 사용자 지정).
    # URDF(assets/yaskawa/yaskawa.urdf)의 velocity limit L/U/B = 386.7/520/550
    # deg/s가 여기 쓰는 GP8_QD_MAX(385/520/550)와 사실상 같아, URDF 기준이든
    # 코드 상수 기준이든 1160/1560/1650 deg/s²로 동일하다.
    # THR NLP 규약은 5.0이지만 NLP은 매끄러움 목적항(W_ACC)이 있어 그 여유를
    # 다 쓰지 않는다. DT는 OU 노이즈 데이터를 모방하므로 허용된 만큼 스텝마다
    # 가속/감속을 뒤집어 눈에 보이는 떨림이 생긴다(가속↔감속 전환 관절당
    # 3~6회/던지기, 2026-08-03 측정). 3.0이면 스텝당 Δω 허용이 193→116 deg/s.
    qdd_factor = 3.0
    enforce_acceleration_limit = True

    # 착지 기준면 z [m]. DT는 "어떤 수평면 위의 목표"에 던지도록 학습되는데
    # (§4.1 "a position on some horizontal plane"), THR의 bin은 pit 안에 있어
    # 조준점 12개가 전부 z=−0.08 이다 (sim_env.bin_targets()로 확인). 여기를
    # 0으로 두면 DT가 바닥(z=0) 기준으로 계산해 8 cm 더 낙하하는 동안 수평으로
    # 120~155 mm 더 나가는 계통 오버슛이 생긴다 (2026-08-03 실측: 상수 보정만
    # 걸어도 명중 30.6%→50.0%). 학습 자체를 실제 조준면에서 하는 것이 정답.
    z_land = -0.08

    # --- 물체 ---
    # THR의 던질 물체는 4cm 큐브~17cm 망치. 착지 판정 높이는 논문처럼 물체
    # 반대각(여기서는 4cm 큐브 기준)으로 두고, 실제 bin 판정은 THR sim이 한다.
    object_half_diag = 0.5 * 0.04 * np.sqrt(2)

    @property
    def dt(self):
        return 1.0 / self.update_rate

    @property
    def number_steps(self):
        return int(self.total_time * self.update_rate)     # = 10

    @property
    def max_speed_rad(self):
        return np.asarray(QD_MAX, float)


class GP8ThrowArm:
    """GP8 평면 던지기 arm — Thr_DT env/throw_env.RoboticArm과 같은 공개 API
    (reset / update_target / get_state / step / reward_sparse / 속성들)라서
    Thr_DT의 수집·학습·평가 코드를 그대로 쓸 수 있다."""

    def __init__(self, cfg=None, rng=None, interp=None):
        """interp: None이면 시뮬이 계획 궤적을 그대로 재생한다고 가정.
        dict(traj_hz=…, period=…)를 주면 **시뮬과 똑같이** Yaskawa 컨트롤러
        보간(MotoROS 3차 Hermite)을 거친 궤적에서 release 상태를 계산한다.
        학습 데이터 수집·평가 환경이 실행 환경과 어긋나면(보간 유무 불일치)
        DT가 배운 착탄 라벨이 실제와 달라지므로, 보간 조건마다 데이터를 따로
        수집해야 한다 (2026-07-31 사용자 지시)."""
        self.cfg = cfg if cfg is not None else GP8Config()
        self.rng = rng if rng is not None else np.random.default_rng()
        self.interp = interp
        self.UPDATE_RATE = self.cfg.update_rate
        self.total_time = self.cfg.total_time
        self.number_steps = self.cfg.number_steps
        self.no_rotation = True
        self.smooth_factor = self.cfg.smooth_factor
        self.number_states = 1
        self.her = True
        self.target_radius = self.cfg.target_radius
        self.joints = np.array(self.cfg.joint_names)
        self.max_speed = np.append(np.asarray(self.cfg.max_speed_deg, float), 1.0)
        self.max_speed_factor = self.cfg.max_speed_factor
        self.gripper_thresh = self.cfg.gripper_thresh
        self.target = np.array([1.0, 0.0, 0.0])
        self.reset()
        self.initial_pos = self.object_position.copy()

    # ------------------------------------------------------------------ FK
    def tcp(self, q3=None):
        """평면 단면의 TCP (x, z) — GP8 6축 FK에서 y=0."""
        p = fk_pos(q6(self.q if q3 is None else q3))
        return np.array([p[0], p[2]])

    def tcp_velocity(self, q3, qd3):
        """TCP 선속도 (vx, vz) — GP8 geometric Jacobian (S=R=T=0 평면)."""
        Jv, _ = jacobian(q6(q3))
        qd = np.zeros(6)
        qd[PLANAR_IDX] = np.asarray(qd3, float)
        v = Jv @ qd
        return np.array([v[0], v[2]])

    # --------------------------------------------------------------- reset
    def reset(self):
        self.q = np.array(self.cfg.home_pose, dtype=np.float64)
        self.qdot = np.zeros(3)
        self.qdot_seg = np.zeros(3)     # 스텝 구간 평균 속도 (release용)
        self.qdot_cmd_prev = np.zeros(3)  # 가속도 rate limit용 직전 명령속도
        self.qdot_node = np.zeros(3)      # 노드에서의 각속도 ω_k (팔은 정지 출발)
        self.action_exec = np.zeros(4)    # 직전 스텝에 실제로 실현된 정규화 액션
        self.q_hist = [self.q.copy()]   # 스텝 노드 이력 (보간 release 계산용)
        self.w_hist = [np.zeros(3)]     # 같은 노드의 각속도 이력 (waypoint 속도)
        self.gripper_closed = True
        self.velocity = np.zeros(4)
        self.curr_step = 0
        self.curr_time = 0.0
        tip = self.tcp()
        self.object_position = np.array([tip[0], 0.0, tip[1]])
        self.object_height = tip[1]

    # -------------------------------------------------------------- target
    def update_target(self, target):
        self.target = np.asarray(target, dtype=np.float64)

    # --------------------------------------------------------------- state
    def get_state(self):
        """(θ_L, θ_U, θ_B, θ_gr) — θ_gr는 이진 (1=닫힘) [§4.1]."""
        return np.array([self.q[0], self.q[1], self.q[2],
                         1.0 if self.gripper_closed else 0.0])

    # -------------------------------------------------------------- reward
    def reward_sparse(self, obj_pos=None, target=None):
        """희소 보상 [식 4.2] — 착지가 시작 위치보다 앞이어야 성공 [repo]."""
        target = self.target if target is None else target
        obj_pos = self.object_position if obj_pos is None else obj_pos
        d = np.hypot(obj_pos[0] - target[0], obj_pos[1] - target[1])
        if d <= self.target_radius and obj_pos[0] > self.initial_pos[0]:
            return 1.0
        return -1.0

    # ------------------------------------------------------------- helpers
    def smooth_velocity(self, new_velocity):
        old = np.asarray(self.velocity)
        sm = old * self.smooth_factor + new_velocity * (1 - self.smooth_factor)
        sm[-1] = new_velocity[-1]
        return sm

    def proj_on_max_speed(self, velocity_vector):
        """[-1,1] 행동 → rad/s 명령 (그리퍼 성분은 그대로) [repo]."""
        v = velocity_vector * self.max_speed * self.max_speed_factor
        v = v * np.pi / 180.0
        v[-1] = velocity_vector[-1]
        return v

    @property
    def max_speed_rad_arm(self):
        return np.asarray(QD_MAX, float) * self.max_speed_factor

    def _integrate_control_step(self, qdot_cmd):
        """1/f 한 주기 적분: 위치 명령 θ_t = θ_{t−1} + ω_t/f [식 5.1]를 1차
        actuator lag로 추종. Thr_DT env와 같은 적분기에 **GP8 위치 한계**를
        추가한다 (한계에 닿은 관절은 속도 0 — 실기의 리미트 정지).

        self.qdot(순간 속도)과 별도로 self.qdot_seg(이 스텝 구간의 평균 속도
        = (q_k − q_{k−1})/Δt)를 기록한다. release 속도로는 **후자**를 쓴다:
        THR sim은 10 Hz 스텝 궤적을 시뮬 그리드로 보간해 kinematic replay하고
        release 상태를 그 궤적의 수치 미분에서 취하므로(sim_env
        _release_state_from_plan), 구간 평균이 시뮬에서 실제로 발현되는 속도다.
        순간 속도를 쓰면 env가 1.70 m로 예측한 던지기가 시뮬에서 1.04 m로
        떨어지는 계통 편향이 생긴다 (2026-07-30 측정)."""
        c = self.cfg
        h = c.dt / c.substeps
        q_prev = self.q.copy()
        w_prev = self.qdot_node.copy()
        # 사다리꼴이면 이번 스텝 이동량 = (ω_{k−1} + ω_k)·Δt/2 (TRAPEZOID 참고).
        dq = ((w_prev + qdot_cmd) * 0.5 if TRAPEZOID else qdot_cmd) * c.dt
        q_cmd = np.clip(self.q + dq, Q_LO, Q_HI)
        if c.tau_act <= 0.0:                 # 지연 없음 — 명령을 그대로 실현
            self.q = q_cmd
            # 노드에서의 속도는 **DT가 그 지점에 출력한 ω_k** 자체다. 공식
            # 보간이 이 값을 waypoint 속도로 그대로 쓰므로 시뮬에서 발현되는
            # 순간속도와도 일치한다 (선형보간 시절의 '구간 평균'이 아니다).
            # 만에 하나 위치 클립이 걸리면 **적분 항등식에서 ω를 역산**한다 —
            # 그래야 노드 위치만으로 ω를 복원하는 node_velocities와 어긋나지
            # 않고, 저장되는 액션도 실제로 실현된 값이 된다.
            if TRAPEZOID:
                self.qdot_node = 2.0 * (q_cmd - q_prev) / c.dt - w_prev
            else:
                self.qdot_node = (q_cmd - q_prev) / c.dt
            self.qdot = self.qdot_node.copy()
            self.qdot_seg = self.qdot_node.copy()
            self.q_hist.append(self.q.copy())
            self.w_hist.append(self.qdot_node.copy())
            return self.tcp()
        for _ in range(c.substeps):
            qdot_des = (q_cmd - self.q) / c.tau_act
            qdot_des = np.clip(qdot_des, -self.max_speed_rad_arm,
                               self.max_speed_rad_arm)
            q_new = np.clip(self.q + qdot_des * h, Q_LO, Q_HI)
            self.qdot = (q_new - self.q) / h      # 한계에 걸리면 실제 속도도 0
            self.q = q_new
        self.qdot_seg = (self.q - q_prev) / c.dt
        self.qdot_node = self.qdot_seg.copy()
        self.q_hist.append(self.q.copy())
        self.w_hist.append(self.qdot_node.copy())
        return self.tcp()

    def release_state(self):
        """release 시각의 평면 TCP (위치, 속도) — **시뮬과 같은 파이프라인**.
        interp가 없으면 현재 자세 + 구간 평균 속도(시뮬의 궤적 미분과 동형),
        있으면 스텝 궤적(+팔로우스루)을 시뮬 그리드로 편 뒤 컨트롤러 보간을
        거친 궤적에서 t_rel의 자세·속도를 읽는다."""
        if not self.interp:
            if not FOLLOWTHROUGH_SMOOTH:
                return self.tcp(), self.tcp_velocity(self.q, self.qdot_seg)
            # planner가 내보낼 것과 **같은** 공식 보간 곡선에서 읽는다.
            nodes = np.asarray(list(self.q_hist) + followthrough_nodes(
                self.q, self.qdot_seg, self.cfg.dt), float)
            t_c, Q_c, V_c = controller_curve(nodes, self.cfg.dt)
            t_rel = self.curr_step * self.cfg.dt
            q_r = np.array([np.interp(t_rel, t_c, Q_c[:, j]) for j in range(3)])
            qd_r = np.array([np.interp(t_rel, t_c, V_c[:, j]) for j in range(3)])
            return self.tcp(q_r), self.tcp_velocity(q_r, qd_r)
        from .gp8_interp import controller_track

        nodes = list(self.q_hist) + followthrough_nodes(
            self.q, self.qdot_seg, self.cfg.dt)
        Qp = np.asarray(nodes, float)
        t_nodes = np.arange(len(Qp)) * self.cfg.dt
        # 시뮬(dt_planner.gp8_dt_traj_fn)이 만드는 것과 같은 240 Hz 궤적
        ts = np.arange(0.0, t_nodes[-1] + SIM_DT / 2, SIM_DT)
        Q = np.column_stack([np.interp(ts, t_nodes, Qp[:, j]) for j in range(3)])
        Qd = np.gradient(Q, ts, axis=0)
        t_c, Q_c, V_c = controller_track(
            ts, Q, Qd, traj_hz=self.interp.get("traj_hz"),
            period=self.interp.get("period"), out_dt=SIM_DT)
        t_rel = self.curr_step * self.cfg.dt          # 스텝 실행 직후 시각
        q_r = np.array([np.interp(t_rel, t_c, Q_c[:, j]) for j in range(3)])
        qd_r = np.array([np.interp(t_rel, t_c, V_c[:, j]) for j in range(3)])
        return self.tcp(q_r), self.tcp_velocity(q_r, qd_r)

    def _ballistic_landing(self, pos, vel):
        """release 상태 (pos=(x,z), vel=(vx,vz))의 첫 지면 도달 x [§4.1].
        무항력 포물선 — 논문 env와 동일 가정."""
        g = self.cfg.gravity
        # 착지 판정 높이 = **조준면 그 자체**. 논문 env는 물체 바닥이 바닥면에
        # 닿는 시점(z_land + 반대각)으로 봤지만, THR 시뮬은 물체 **중심**이
        # 조준면 z=−0.08을 지나는 지점을 기록하고(sim_env.step의 crossings),
        # NLP·Physics도 질점이 목표 z에 도달하는 지점으로 조준한다. DT만 28 mm
        # 높은 곳에서 궤적을 끊으면 그만큼 더 날아가 계통 오버슛이 된다 —
        # 실측 반경 편향 DT +62 mm vs NLP +12 mm (2026-08-05). 세 모델의 착지
        # 규약을 통일한다.
        z0 = pos[1] - getattr(self.cfg, 'z_land', 0.0)
        if z0 <= 0.0:
            return pos[0]
        vz = vel[1]
        t_land = (vz + np.sqrt(vz ** 2 + 2.0 * g * z0)) / g
        return pos[0] + vel[0] * t_land

    # ----------------------------------------------------------------- step
    def step(self, velocity_vector):
        """10 Hz 제어 한 스텝 — Thr_DT env.step과 같은 반환 규약:
        (reward, done, termination_reason, obj_pos, success)."""
        velocity_vector = np.asarray(velocity_vector, dtype=np.float64)
        velocity_vector = self.proj_on_max_speed(velocity_vector)
        velocity_vector = self.smooth_velocity(velocity_vector)
        # 속도·가속도·위치(제동 여유 포함) 한계의 교집합으로 **먼저** 투영한다.
        # 종전처럼 가속도만 사후 클립하고 위치는 적분기에서 잘라내면 기록 액션이
        # 실현값과 달라진다 (실측 60~64% 스텝에서 불일치).
        # 기준은 **직전 노드에서 실제로 실현된** ω 다. 명령값(qdot_cmd_prev)을
        # 쓰면 위치 클립이 걸린 스텝에서 둘이 갈라져, 실행 곡선의 가속도가
        # 상자를 넘는다 (실측 무작위 60개 중 8개).
        w_lo, w_hi = feasible_qdot_box(self.q, self.qdot_node,
                                       self.cfg.dt, cfg=self.cfg)
        velocity_vector[:3] = np.clip(velocity_vector[:3], w_lo, w_hi)
        self.qdot_cmd_prev = velocity_vector[:3].copy()
        self.velocity = velocity_vector
        qdot_cmd = velocity_vector[:3]
        gripper = velocity_vector[-1]

        tip = self._integrate_control_step(qdot_cmd)
        # 실제로 실현된 정규화 액션 — 수집기는 **이 값**을 데모로 저장한다.
        # 상자 투영 덕에 보통 명령과 같지만, 수치적으로도 일치를 보장한다.
        self.action_exec = np.append(self.qdot_seg / np.asarray(QD_MAX, float),
                                     gripper)
        self.curr_time += self.cfg.dt
        self.curr_step += 1

        if self.gripper_closed:
            self.object_position = np.array([tip[0], 0.0, tip[1]])
            self.object_height = tip[1]

        if self.curr_step >= self.number_steps or gripper < self.gripper_thresh:
            done = True
            if gripper < self.gripper_thresh:
                termination_reason = f"Gripper was opened with value: {gripper}"
                # release 자세·속도는 시뮬과 같은 파이프라인으로 (보간 반영)
                tip, tip_vel = self.release_state()
                x_land = self._ballistic_landing(tip, tip_vel)
                self.gripper_closed = False
                self.object_position = np.array(
                    [x_land, 0.0, self.cfg.object_half_diag])
                self.object_height = self.object_position[2]
                reward = self.reward_sparse()
                success = True
            else:
                termination_reason = f"Time is up: {self.curr_time}"
                reward = -1.0
                success = False
        else:
            if self.object_height <= self.cfg.object_half_diag:
                termination_reason = ("Object is too close to ground: "
                                     f"{self.object_height}")
                done = True
                reward = -1.0
                success = False
            else:
                termination_reason = None
                done = False
                reward = 0.0
                success = False
        return (reward, done, termination_reason, self.object_position.copy(),
                success)

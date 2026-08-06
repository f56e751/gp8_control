"""effort-PID 동역학 — 공식 Gazebo 스택의 평면 3링크 재현 (2026-08-06 사용자 지시).

공식 논문·코드에는 가속도 한계가 없다. 시뮬(Gazebo)에서는
  `effort_controllers/JointTrajectoryController` 의 PID 가 **토크**를 계산하고
  Gazebo 가 URDF `<limit effort>` 로 클램프, 가속도는 q̈ = M⁻¹(τ−…) 로 창발한다.
이 모듈이 그 사슬을 그대로 옮긴다. 수치는 전부 공식 저장소에서 가져왔다:

  · 관성: `gp8_macro_gazebo.xacro` <inertial> — **모든 링크 COM 이 관절 원점**
    (origin 0 0 0), iyy 만 유효 (평면 y축 회전). 조악하지만 그게 공식이다.
  · PID: `gp8_gazebo_controller.yaml` 활성 게인 ("Works good with 10Hz simulation")
       L 8000/1000/100(clamp 1000), U 5000/500/70(clamp 100), B 300/10/4(clamp 10)
  · effort: L 176.4, U 107.56, B 32.68 N·m
  · 물리 스텝: 1 kHz (`camera_world.world` max_step_size 0.001)
  · 설정점: JTC 방식 — 10 Hz 위치 명령 사이를 선형 램프

동역학 정식화 (도출 실수를 피하려고 점질량 야코비안 합성으로 구현):
  평면(x-z, y축 회전) 3자유도 (L,U,B). 각 질점 p (몸체 b, 피벗에서 r):
      p(q) = Σ_{i<b} a_i·u(Φ_i) + r·u(Φ_b),   Φ_i = Σ_{j≤i} s_j q_j + o_i
  M(q)      = Σ m J_pᵀJ_p + Σ I_k 1_k1_kᵀ        (1_k = Φ_k 의 ∂/∂q 행)
  편향력    = Σ m J_pᵀ (J̇_p q̇)                    (순수 관성항은 속도항 없음)
  중력      = Σ m g ∂p_z/∂q
  부호 s_j 와 오프셋 o_i 는 THR `throwing.fk_frames` 에 수치 캘리브레이션한다
  (아래 _calibrate) — 프레임 규약 실수를 원천 차단.

우리 rig 추가분 (공식 URDF 에 없는 것): 흡착 툴 질량 M_TOOL 을 TCP 에 점질량으로.
"""

from __future__ import annotations

import numpy as np

G = 9.81

# ── 공식 수치 (ThrowBot_official) ──────────────────────────────────────────
# 평면 3몸체: body0 = link_2_l (L 구동), body1 = link_3_u + link_4_r (U 구동),
#             body2 = link_5_b + link_6_t + tool (B 구동)
EFFORT = np.array([176.4, 107.56, 32.68])          # τ 상한 [N·m] (L, U, B)
PID_P = np.array([8000.0, 5000.0, 300.0])
PID_D = np.array([1000.0, 500.0, 10.0])
PID_I = np.array([100.0, 70.0, 4.0])
PID_ICLAMP = np.array([1000.0, 100.0, 10.0])
H_SUB = 1.0e-3                                     # Gazebo max_step_size

M_TOOL = 0.4    # 흡착 툴+어댑터 추정 [kg] — 공식엔 Robotiq(≈1 kg)이 그 자리

# 질점 목록: (body_idx, r_along_link[m], mass[kg])  — COM=관절 원점(공식 xacro)
# body1 의 link_4_r 은 U 피벗에서 전완 방향 0.04 지점(R 관절 원점).
# body2 의 link_6_t·툴: THR 체인의 TCP 는 B 피벗에서 0.32 (플랜지 0.08 + 툴
# 0.24 를 합친 체인 병합 — throwing._CHAIN 인덱스 5). 그 끝점에 몰아 놓는다.
_POINTS = [(0, 0.0, 7.98), (1, 0.0, 4.8), (1, 0.04, 3.575),
           (2, 0.0, 0.5), (2, 0.32, 0.1), (2, 0.32, M_TOOL)]
_IYY = np.array([1.407, 0.236 + 0.353, 0.003 + 0.001])   # 몸체별 순수 회전관성

_CAL = {}       # {'a': [a1,a2], 's': 부호(3,), 'o': 오프셋(3,), 'p0': L피벗 xz}


def _fk_pivots(q3):
    """throwing.fk_frames 로 L/U/B 피벗과 TCP 의 (x,z) — 캘리브레이션 전용."""
    from .dt_gp8_env import q6
    from .throwing import fk_frames
    origins, axes, tcp = fk_frames(q6(np.asarray(q3, float)))
    # 체인 인덱스: 1=L, 2=U, 4=B (throwing._CHAIN 순서), TCP = 체인 끝
    pv = [np.asarray(origins[i], float) for i in (1, 2, 4)]
    return [np.array([v[0], v[2]]) for v in pv], np.array([tcp[0], tcp[2]])


def _calibrate():
    """부호 s_i·오프셋 o_i·링크길이 a_i 를 FK 에 수치 맞춤."""
    if _CAL:
        return _CAL
    pv0, _ = _fk_pivots([0.0, 0.0, 0.0])
    a1 = float(np.linalg.norm(pv0[1] - pv0[0]))
    a2 = float(np.linalg.norm(pv0[2] - pv0[1]))
    seg_ang0 = [float(np.arctan2(*(pv0[1] - pv0[0])[::-1][::-1])) for _ in (0,)]

    def seg_angle(p_from, p_to):
        d = p_to - p_from
        return float(np.arctan2(d[1], d[0]))       # x-z 평면 각 (x축 기준)

    o = np.zeros(3)
    o[0] = seg_angle(pv0[0], pv0[1])
    o[1] = seg_angle(pv0[1], pv0[2])
    _, tcp0 = _fk_pivots([0.0, 0.0, 0.0])
    o[2] = seg_angle(pv0[2], tcp0)
    # 부호: 관절 j 만 δ 돌렸을 때 해당 세그먼트 각 변화로 결정
    s = np.zeros(3)
    d = 1e-4
    for j in range(3):
        q = np.zeros(3); q[j] = d
        pv, tcp = _fk_pivots(q)
        if j == 0:
            s[0] = np.sign(seg_angle(pv[0], pv[1]) - o[0]) or 1.0
        elif j == 1:
            s[1] = np.sign(seg_angle(pv[1], pv[2]) - o[1]) or 1.0
        else:
            s[2] = np.sign(seg_angle(pv[2], tcp) - o[2]) or 1.0
    _CAL.update(a=np.array([a1, a2]), s=s, o=o, p0=pv0[0])
    return _CAL


def _phis(q3, cal):
    """몸체 절대각 Φ_i 와 ∂Φ_i/∂q_j 행렬 (3×3 하삼각·부호 반영)."""
    s, o = cal["s"], cal["o"]
    cum = np.cumsum(s * np.asarray(q3, float))
    phi = o + cum
    W = np.zeros((3, 3))
    for i in range(3):
        W[i, :i + 1] = s[:i + 1]
    return phi, W


def dynamics(q3, qd3, tau, cal=None):
    """q̈ = M(q)⁻¹ (τ − 편향 − 중력).  전부 평면 (x-z, y축 회전) 폐형."""
    cal = cal or _calibrate()
    a = cal["a"]
    phi, W = _phis(q3, cal)
    c, s_ = np.cos(phi), np.sin(phi)
    u = np.stack([c, s_], 1)                        # 링크 방향 (x,z)
    du = np.stack([-s_, c], 1)                      # dΦ 방향
    qd = np.asarray(qd3, float)
    phid = W @ qd

    M = np.zeros((3, 3)); bias = np.zeros(3); grav = np.zeros(3)
    for b, r, m in _POINTS:
        # p = Σ_{i<b} a_i u_i + r u_b ;  J = Σ_{i<b} a_i du_i W_i행 + r du_b W_b행
        J = np.zeros((2, 3)); Jd = np.zeros((2, 3))
        for i in range(b):
            J += a[i] * np.outer(du[i], W[i]); Jd += a[i] * np.outer(-u[i] * phid[i], W[i])
        J += r * np.outer(du[b], W[b]); Jd += r * np.outer(-u[b] * phid[b], W[b])
        M += m * (J.T @ J)
        bias += m * (J.T @ (Jd @ qd))
        grav += m * G * J[1]                        # z행 = 높이 기울기
    for k in range(3):
        M += _IYY[k] * np.outer(W[k], W[k])
    return np.linalg.solve(M, np.asarray(tau, float) - bias - grav)


class EffortPID:
    """공식 JTC 대응: 10 Hz 위치 명령 → 1 kHz PID 토크 → effort 클램프 → 적분."""

    def __init__(self, q0, q_lo, q_hi):
        self.q = np.asarray(q0, float).copy()
        self.qd = np.zeros(3)
        self.ei = np.zeros(3)
        self.q_lo, self.q_hi = np.asarray(q_lo, float), np.asarray(q_hi, float)
        self.cal = _calibrate()
        self.sp_prev = self.q.copy()

    def track(self, q_cmd, dt):
        """한 제어 스텝(dt): 설정점을 sp_prev→q_cmd 선형 램프하며 서브스텝 적분.
        리턴: (q, qd) — 스텝 끝의 **실측** 상태 (공식처럼 상태는 측정값)."""
        n = max(1, int(round(dt / H_SUB)))
        sp0, sp1 = self.sp_prev, np.asarray(q_cmd, float)
        for k in range(n):
            al = (k + 1) / n
            sp = sp0 + (sp1 - sp0) * al
            spd = (sp1 - sp0) / dt
            e = sp - self.q
            self.ei = np.clip(self.ei + e * H_SUB * PID_I, -PID_ICLAMP, PID_ICLAMP)
            tau = np.clip(PID_P * e + PID_D * (spd - self.qd) + self.ei,
                          -EFFORT, EFFORT)
            qdd = dynamics(self.q, self.qd, tau, self.cal)
            self.qd = self.qd + qdd * H_SUB
            self.q = self.q + self.qd * H_SUB
            # URDF 하드스톱 (Gazebo 대응): 한계에서 정지
            hit_lo = self.q < self.q_lo; hit_hi = self.q > self.q_hi
            self.q = np.clip(self.q, self.q_lo, self.q_hi)
            self.qd[hit_lo | hit_hi] = 0.0
        self.sp_prev = sp1
        return self.q.copy(), self.qd.copy()

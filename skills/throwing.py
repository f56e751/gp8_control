"""
GP8 kinematics + ballistics 코어 라이브러리.

궤적 생성은 throw_nlp.py(CasADi/IPOPT B-Spline NLP)가 담당하고, 이 모듈은
그 밑단의 공용 유틸만 제공한다:
  - GP8 치수/하드웨어 한계 상수, joint chain 정의 (_CHAIN)
  - fk_frames / fk_pos / jacobian (수치 FK — throw_nlp의 심볼릭 FK와 교차검증됨)
  - ik_position (position-only IK — 시작 자세 계산용)
  - landing_error (release 상태 → 착탄 오차, 검증용)

(구 quintic/manifold planner — ThrowProblem, plan.py 파이프라인 — 는
 2026-07-15 throw_nlp 통합 때 제거. 세션 백업: scratchpad THR_deleted_backup/)
"""

import numpy as np

G = 9.81
G_VEC = np.array([0.0, 0.0, -G])

# ---------------------------------------------------------------------------
# Yaskawa GP8 kinematics
# ---------------------------------------------------------------------------
# datasheet 치수 (m): base→L축 높이 d1, S→L offset a1, 상완 a2,
# U→R offset a3, 전완 d4, 손목→flange d6.  reach 검증: a1+a2+d4 ≈ 0.725 (spec 0.727)
# tool: flange에 단 suction gripper 길이 — TCP(=물체 위치)는 그리퍼 끝
# 0.24 = IITP URDF 확정 (gripper 로드 24cm, TCP = link6에서 0.32; 2026-07-15 교정 v2)
GP8_DIMS = dict(d1=0.330, a1=0.040, a2=0.345, a3=0.040, d4=0.340, d6=0.080, tool=0.240)

# GP8 hardware limits (datasheet). 실기 파라미터 확인 후 필요시 교체.
GP8_Q_MIN = np.deg2rad([-170.0, -65.0, -113.0, -190.0, -135.0, -360.0])
GP8_Q_MAX = np.deg2rad([170.0, 150.0, 255.0, 190.0, 135.0, 360.0])
GP8_QD_MAX = np.deg2rad([455.0, 385.0, 520.0, 550.0, 550.0, 1000.0])

# joint chain: (직전 translation xyz, 회전축) — zero pose에서 상완 수직, 전완 +x 수평
_CHAIN = [
    ((0.0, 0.0, 0.0), "z"),                                  # S
    ((GP8_DIMS["a1"], 0.0, GP8_DIMS["d1"]), "y"),            # L
    ((0.0, 0.0, GP8_DIMS["a2"]), "y"),                       # U
    ((0.0, 0.0, GP8_DIMS["a3"]), "x"),                       # R (전완 roll)
    ((GP8_DIMS["d4"], 0.0, 0.0), "y"),                       # B (손목 pitch)
    ((GP8_DIMS["d6"] + GP8_DIMS["tool"], 0.0, 0.0), "x"),    # T (roll, TCP = 그리퍼 끝)
]

_AXIS = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0])}


def _rot(axis, t):
    c, s = np.cos(t), np.sin(t)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def fk_frames(q):
    """각 joint의 원점/회전축(base frame) + TCP 위치를 리턴."""
    R = np.eye(3)
    p = np.zeros(3)
    origins, axes = [], []
    for (xyz, ax), qi in zip(_CHAIN, q):
        p = p + R @ np.asarray(xyz)
        origins.append(p)
        axes.append(R @ _AXIS[ax])
        R = R @ _rot(ax, qi)
    return origins, axes, p  # 마지막 회전은 원점을 안 움직이므로 TCP = joint T 원점


def fk_pos(q):
    return fk_frames(q)[2]


def jacobian(q):
    """geometric Jacobian: v = Jv qd, ω = Jw qd."""
    origins, axes, p = fk_frames(q)
    Jv = np.stack([np.cross(a, p - o) for a, o in zip(axes, origins)], axis=1)
    Jw = np.stack(axes, axis=1)
    return Jv, Jw


def ik_position(p_des, q_seed, iters=300, tol=1e-10):
    """position-only IK (damped least squares). orientation은 free."""
    q = np.asarray(q_seed, float).copy()
    for _ in range(iters):
        e = p_des - fk_pos(q)
        if np.linalg.norm(e) < tol:
            return q, True
        Jv, _ = jacobian(q)
        dq = Jv.T @ np.linalg.solve(Jv @ Jv.T + 1e-8 * np.eye(3), e)
        q += np.clip(dq, -0.2, 0.2)
    return q, np.linalg.norm(p_des - fk_pos(q)) < 1e-8


def landing_error(p, v, p_target):
    """
    (p,v)에서 release 했을 때 target 높이 통과점의 최소 수평 오차 (m).
    상승/하강 두 근 모두 τ>0이면 후보 (target이 위에 있으면 상승 중 명중도 유효).
    τ>0 근이 없으면 inf.
    """
    disc = v[2] ** 2 - 2 * G * (p_target[2] - p[2])
    if disc < 0:
        return np.inf
    taus = [(v[2] + s * np.sqrt(disc)) / G for s in (1.0, -1.0)]
    taus = [t for t in taus if t > 1e-9]
    if not taus:
        return np.inf
    return min(np.linalg.norm(p[:2] + v[:2] * t - p_target[:2]) for t in taus)

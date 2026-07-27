"""Throw skill: grasp an object at the intercept, then fling it (NLP throw).

NN throw를 THR 통합 NLP planner (throw_nlp.solve_throw_nlp — CasADi/IPOPT
B-Spline)로 교체한 버전 (2026-07-15: 구 manifold plan.plan_throw 대체).
궤적은 3-세그먼트:
  lift   — 정지(프레스 자세 q_press, 폴백 시 grasp_joint)에서 던지기 시작
           자세(q_lift, 정지)까지 time-optimal
           (grasp 위치 + THROW_LIFT 상승 — 벨트 clearance)
  throw  — NLP 단일 B-spline이 스윙+release 윈도우(t*±T/2)+감속을 통합 최적화.
           v0=vf=0 구조적 보장, 윈도우 내내 착지 최적 (아무 때나 release 가능),
           하드웨어 한계 전 구간 hard: q̈≤5·q̇_max(knot exact) + |q̇|≤q̇_max
           (derivative hull) + 위치(제어점 hull — B≥10°, U≤45°, |R|≤80° 포함)
           + 기둥 회피(윈도우까지).
  chain  — 궤적이 '정지'로 끝나므로 park까지는 rest-to-rest 이동 (감속 세그먼트
           불필요 — NLP가 이미 감속 포함).

NN 버전 대비 구조 차이:
  * compute_throw_params / new_trajectory / 속도 clamp 루프 / windup 역적분
    제거 — 전부 NLP 하나가 담당.
  * release는 윈도우 중앙(t*) — 양쪽 ±T/2 타이밍 여유 (suction 밸브 jitter 면역).
  * 계획: ① 세션 warm 캐시 → ② 파일 warm DB(warm_db.pkl) → ③ cold
    multistart(순차) 순으로 시도, 모든 해는 _solution_gates(위치/속도/착탄
    독립 재검증 — 착탄 게이트 30mm)를 통과해야 dispatch.
※ gp8_control 환경에서만 실행 가능 — 이 마이그레이션은 실기/HIL 미검증.
"""

from __future__ import annotations

import datetime
import os
import pickle
import sys
import threading
import time
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from gp8_control.skills.base import ManipulationSkill, SkillResult
from gp8_control.trajectory.trajectory_primitive import trajectory

# THR NLP planner 모듈(throw_nlp.py / throwing.py)은 이 패키지(skills/)에 복사돼
# 있다. 둘은 서로를 bare import(`from throwing import ...`)로 찾으므로 skills/
# 디렉터리 자체를 sys.path에 올린다 (THR 원본 레포와 파일을 그대로 공유하기 위해
# 모듈 내부 import는 패키지 경로로 고치지 않는다).
_THR_DIR = str(Path(__file__).resolve().parent)
if _THR_DIR not in sys.path:
    sys.path.insert(0, _THR_DIR)
import throw_nlp  # noqa: E402
from throw_nlp import Q_HI, Q_LO, _spline_eval, solve_throw_nlp  # noqa: E402
from throwing import (GP8_DIMS, GP8_QD_MAX, fk_pos, ik_position,  # noqa: E402
                      jacobian, landing_error, launch_state)

if TYPE_CHECKING:
    from gp8_control.skills.context import PickRequest
    from gp8_control.tracking import TrackedObject

# 플래너(throwing/throw_nlp) ↔ 로봇(robots/gp8 = 하드웨어/스트림) 관절 부호 변환.
# GP8 PoE 축은 J3..J6 = -y,-x,-y,-x 인데 THR 체인은 +y,+x,+y,+x — 관절 3~6 부호
# 반전. 수치 검증: fk_pos(S*q) vs GP8.forward_kinematics(q) 오차 0.000mm (300
# 랜덤 자세; 2026-07-21 툴 길이 URDF 통일 후 — 구 tool=0.240 시절엔 상수 5mm).
# S는 자기 역원(S*S=1)이라 양방향 변환에 같은 벡터를 쓴다. 이 변환 없이 플래너
# 해를 그대로 스트리밍하면 U/R/B/T가 반전된 계획 밖 스윙이 나간다 (2026-07-20
# 적대 리뷰 확정 — 플래너의 한계/기둥회피 보장도 로봇 규약에선 무효).
_PLANNER_SIGN = np.array([1.0, 1.0, -1.0, -1.0, -1.0, -1.0])


# =========================================================================
# Throw policy
# =========================================================================

# Throw target bin (base-frame XY, m). The throw HEADING is re-aimed PER OBJECT
# by the manifold planner itself (v0 points from the object's throw-start
# position to the bin), so no per-object theta is needed for the trajectory —
# theta below is only used by the legacy plan_throw_landing fallback.
# THROW_BIN_X: float = 1.1
# THROW_BIN_Y: float = -0.25

# Per-class throw bin TARGET (absolute base-frame XYZ, m). When a target's
# class is in this map, the landing point is OVERRIDDEN with these coordinates
# AND the NLP solve starts in the BACKGROUND at execute() start (fixed target →
# 계획을 이동/대기와 겹침). 카메라 클래스명 기준: 페트병 = "transparent"
# (perception_client.py — "plastic"이라는 클래스는 없음). 좌표는 운영자 제공
# 실측(2026-07-20), z=0 = 바닥 높이 bin.
THROW_BIN_TARGET_MAP: dict[str, tuple] = {
    "metal": (0.9, 0.16, -0.11),
    "transparent": (0.9, -0.16, -0.11),
}

# 던지기 시작 TCP = grasp 위치 + 이만큼 lift (벨트/주변 clearance 확보).
THROW_LIFT: float = 0.10

# HOVER_DESCEND 픽 대기 높이: grasp TCP + 이 값 [m]. 도착 전 벨트 위 물체가
# 파킹된 컵과 충돌하지 않도록 위에서 기다린다.
HOVER_ABOVE: float = 0.05
# 프레스 목표 TCP Z [m, base 절대 높이]. 물체 도착 순간 이 높이까지 내려 찍는다.
# 운영자 지정(2026-07-20). 주의: 벨트 접촉 실측 TCP Z(GRASP_Z 0.062, 터치 0.067)
# 보다 ~3cm 낮음 — 컵 bellows/물체 압축으로 흡수되는 것을 전제로 한 값이므로,
# 픽 미스(빈 벨트)에 프레스가 나가면 컵이 벨트를 강하게 누른다. 조정은 이 상수.
PRESS_Z: float = -0.005
# 프레스 후 유지 시간 [s]: 컵이 PRESS_Z에서 물체를 누른 채 이만큼 기다린 뒤
# 던지기로 넘어간다 (진공 실링 확보). 이 구간 동안 팔은 정지가 아니라 벨트와
# 같은 속도로 물체를 따라간다(_build_belt_follow) — 정지 유지면 물체가 컵 밑에서
# v·PICK_TIME(0.25m/s·1s = 25cm)만큼 끌려나가 실링이 깨진다. NLP 선계획이 아직
# 안 끝났으면 join이 이 구간과 겹쳐 흡수되므로 시간은 공짜에 가깝다.
PICK_TIME: float = 0.5
# 추종 마무리 감속 시간 [s]: 추종 끝에서 정지까지 (lift가 정지 출발 전제).
# 이 구간의 뒤처짐(≈v·FOLLOW_DECEL_T/2)은 이미 실링된 뒤라 무해 — 컵이 물체를
# 잡고 있으면 벨트가 밑에서 미끄러진다.
FOLLOW_DECEL_T: float = 0.1
# 프레스 선행 시간 [s] — 하강 소요시간(자동 계산) 위에 얹는 고정 보정.
# 대기 종료를 이만큼 더 앞당겨 접촉을 앞으로 당긴다. 실기 관측(2026-07-20):
# 프레스가 도착보다 일정하게 ~0.2s 늦어 0.2로 설정. 벨트 속도와 무관하게 일정
# 시간 어긋날 때 쓰는 노브이고, 속도에 비례해 어긋나면 대신 카메라 쪽
# GP8_PERCEPTION_LATENCY_S(v×delay 역보정)를 조정한다. env GP8_PRESS_LEAD.
PRESS_LEAD: float = float(os.environ.get("GP8_PRESS_LEAD", "0.2"))
# release 윈도우 길이 (s) — throw_nlp 기본값(0.05)과 동일하게 유지해야
# W_ACC 스윕이 검증한 조합과 warm entry(제약 구성 키)가 그대로 성립한다.
# 실기 밸브 지연의 '평균'은 RELEASE_LEAD 선행 명령이 보정하고, 잔여 jitter는
# ±window/2 = ±25ms 안이면 착지 보장 (구버전 0.08은 미검증 조합이라 폐기).
THROW_WINDOW_T: float = float(throw_nlp.RELEASE_TIME)

# 착탄 게이트 (m): 윈도우 dense 착탄오차 최대가 이걸 넘으면 그 해를 거부.
# penalty 공식화라 착탄 정확도는 solver 보장이 아님 — 나쁜 basin은 윈도우
# 일부에서 착탄 불능(inf)이 나올 수 있어 dispatch 전 반드시 확인 (실기 안전).
LANDING_GATE: float = 0.03

# Dispatch 직전 Cartesian 안전 엔벨로프 (base frame TCP, m) — 운영자 지정
# 2026-07-23. 던지기 궤적의 TCP가 이 밖으로 나가면 **실행하지 않고 에러로 보고**한다.
#   x ≤ MIN_TCP_X : 기둥/베이스 쪽으로 파고듦
#   z ≤ MIN_TCP_Z : 바닥/벨트 충돌
#   z ≥ MAX_TCP_Z : 팔을 너무 높이 듦 (2026-07-27 사용자 — DB 빌더 게이트와 동일 0.85)
# 이 게이트가 유일한 방어선인 이유: throw_nlp의 기둥 회피는 release 창 끝까지만
# 활성이고(감속 꼬리 전 구간에 걸면 무충돌 basin까지 잘려 multistart가 전멸),
# TCP 바닥 클리어런스는 아예 hard 제약이 아니다("hard로 걸면 0/148 전멸" —
# throw_nlp.py 제약 3b 주석). 즉 감속 꼬리가 지하로 다이브하는 해가 정상 수렴한다.
# MAX_TCP_Z 는 tools/build_warm_db._gates 와 동일 기준으로, 스윙 아크가 이보다
# 높이 올라가는 해를 후보·dispatch 양쪽에서 거른다 (빌더와 런타임 정합).
MIN_TCP_X: float = 0.20
MIN_TCP_Z: float = 0.04
MAX_TCP_Z: float = 0.85

# cold multistart 초기해 변형 (lift 높이 시작 z≈0.7 기준). IPOPT는 local
# solver라 한 초기해의 basin이 infeasible하면 실패만 반환 → 순차 재시도.
# (sim처럼 fork 병렬화하지 않는 이유: ROS2 노드에서 fork는 DDS 스레드와
#  충돌해 hang 위험 — 실기에서는 안전이 latency보다 우선.)
INIT_VARIANTS_ROS = (
    None,                                                             # 기본
    dict(dq_swing=[0.0, 0.9, 0.3, 0.0, 0.7, 0.0], T0=0.6),            # 어깨 위주
    dict(dq_swing=[0.0, 0.5, 0.9, 0.0, 0.2, 0.0], T0=1.1, chi0=0.7),  # elbow 위주
    dict(dq_swing=[-0.13, 0.7, 0.6, 0.0, 0.4, 0.0]),                  # 반대 yaw
)

# offline warm DB (선택): THR 루트의 warm_db.pkl — 있으면 최근접 entry로
# full warm start polish (~1-2s). 공식화 파라미터가 다르면 자동 무시.
THROW_WARM_DB: str = os.environ.get(
    "GP8_THROW_WARM_DB", str(Path(_THR_DIR) / "warm_db.pkl"))

_WARM_DB_CACHE: "Optional[list]" = None   # 파일 DB 로드 캐시 (None=미로드)


def _formulation_params() -> dict:
    """warm entry 유효성 기준이 되는 공식화 파라미터 (sim의 _warm_db_params와
    같은 키 구성 — sim이 만든 DB에 sim 전용 키(tilt 등)가 더 있어도 여기 키들만
    일치하면 entry의 basin/제약 구성이 호환된다)."""
    return dict(rt=throw_nlp.RELEASE_TIME, w_acc=throw_nlp.W_ACC,
                n_ctrl=throw_nlp.N_CTRL, n_win=throw_nlp.N_WIN,
                q_lo=throw_nlp.Q_LO.tolist(), q_hi=throw_nlp.Q_HI.tolist(),
                qd_max=GP8_QD_MAX.tolist(),
                col=(throw_nlp.COL_R, throw_nlp.COL_H),
                pos_mode=getattr(throw_nlp, "POS_LIMIT_MODE", "colloc"),
                # 2026-07-21 세분화: 해(특히 dual lam_g)의 유효성에 영향을 주는
                # 나머지 공식화 요소 전부 — 목적함수 가중치(w1/w2/w_sens), 가속도
                # 한계, t_f 범위, 스플라인 차수, 운동학 치수(툴 길이 포함). 이 중
                # 하나라도 바뀌면 기존 entry는 무효 (빌더
                # tools/build_warm_db._formulation_params_standalone 과 키를
                # 반드시 동일하게 유지할 것; sim이 만드는 DB도 마찬가지).
                w1=throw_nlp.W1, w2=throw_nlp.W2, w_sens=throw_nlp.W_SENS,
                qdd_lim=throw_nlp.QDD_LIM.tolist(),
                t_bounds=tuple(throw_nlp.T_BOUNDS),
                degree=throw_nlp.DEGREE,
                dims=dict(GP8_DIMS),
                # flight 모델 마커 (rk4x8-anisodrag-grip20mm): 항력/RK4 비행·발사점
                # (GRIP_OFF)·τ 변수가 목적함수와 제약 구성(opti.ng)을 바꾸므로
                # entry의 basin/dual 유효성 키에 포함. 없는 구버전 = "parabola".
                flight=getattr(throw_nlp, "FLIGHT_MODEL", "parabola"))


def _load_warm_db(log) -> list:
    """warm_db.pkl 로드 (한 번만). 공식화 불일치·손상 시 빈 리스트 (cold 진행)."""
    global _WARM_DB_CACHE
    if _WARM_DB_CACHE is None:
        _WARM_DB_CACHE = []
        if os.path.exists(THROW_WARM_DB):
            try:
                with open(THROW_WARM_DB, "rb") as f:
                    db = pickle.load(f)
                ours = _formulation_params()
                db_params = db.get("params", {})
                if all(db_params.get(k) == v for k, v in ours.items()):
                    _WARM_DB_CACHE = db["entries"]
                    log.info(f"warm DB 로드: entry {len(_WARM_DB_CACHE)}개")
                else:
                    log.warn("warm DB 공식화 불일치 — cold로 진행 (재구축 필요)")
            except Exception as e:  # 손상 파일이 실기 스킬을 죽이면 안 됨
                log.warn(f"warm DB 로드 실패({type(e).__name__}) — cold로 진행")
    return _WARM_DB_CACHE


class PickWaitMode(Enum):
    """How the arm waits at the ambush intercept before suction fires.

    Map object classes to a wait mode in ``PICK_WAIT_MODE``; unlisted classes
    use ``DEFAULT_PICK_WAIT_MODE``.
    """

    WAIT_AT_GRASP = "wait_at_grasp"   # cup parked at grasp height; suction on arrival
    HOVER_DESCEND = "hover_descend"   # hover HOVER_ABOVE above; press to PRESS_Z on arrival


# Per-class wait mode (class_name -> PickWaitMode). Classes not listed use
# DEFAULT_PICK_WAIT_MODE.
PICK_WAIT_MODE: dict[str, PickWaitMode] = {}
DEFAULT_PICK_WAIT_MODE = PickWaitMode.HOVER_DESCEND


class RobustThrowSkill(ManipulationSkill):
    """Pick (suction) at the intercept and throw via the NLP (CasADi/IPOPT) planner.

    Drop-in alternative to ``throw_skill.ThrowSkill`` (the NN thrower), registered
    under its own name so a run can pick it: ``--skill robust_throw`` /
    ``GP8_FORCE_SKILL=robust_throw`` / launch ``skill:=robust_throw``.

    ``execute`` runs the full ambush cycle (position → wait → suction → throw →
    chain to next intercept). The individual stages —
    :meth:`plan_throw_landing`, :meth:`plan_nlp_throw`,
    :meth:`build_throw_trajectory` — are public so external code can reuse
    the throw planning.
    """

    name = "robust_throw"

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._last_throw_meta: dict = {}
        # 세션 warm 캐시: target(반올림) → 직전 성공 해 (P, t_f, t*, lam_g, u_pos).
        # 같은 bin으로 재던지기가 실전의 대부분 — 두 번째부터는 polish ~1-2s.
        self._warm_cache: dict = {}

    def t_to_contact(self, move_time: float) -> float:
        """Throw pick budget = positioning estimate + a GUARANTEED parked
        vacuum-forming hold (``cfg.MIN_SUCTION_HOLD``).

        Adding the hold to the base ``move_time * PICK_FEASIBILITY_FACTOR`` makes
        ``earliest_reachable_intercept`` place the grasp far enough DOWNSTREAM that
        the arm reaches it ~``MIN_SUCTION_HOLD`` before the object arrives, so the
        vacuum seals during the parked wait instead of firing with ~0 lead (the
        backed-up 2nd+ object miss). An object that can't be caught that far
        downstream is dropped by the same solver rather than grabbed with no hold.
        """
        base = move_time * self.ctx.cfg.PICK_FEASIBILITY_FACTOR
        return base + self.ctx.cfg.MIN_SUCTION_HOLD

    def arrival_lead(self) -> float:
        """대기를 끝내는 선행 시간 = 공용 디스패치 예산 + PRESS_LEAD.

        HOVER_DESCEND 픽은 대기 종료 후 '하강해서 눌러야' 접촉이므로, 하강
        소요시간(execute가 궤적에서 계산해 더함) 위에 잔여 지연 보정으로
        PRESS_LEAD를 얹는다. base와 달리 이 스킬에만 적용된다.
        """
        return super().arrival_lead() + PRESS_LEAD

    def _build_belt_follow(self, T_press: np.ndarray, press_joint: np.ndarray,
                           v_belt: float):
        """PICK_TIME 동안 벨트와 같은 속도로 물체를 따라가는 직교 직선 세그먼트.

        TCP는 PRESS_Z를 유지한 채 벨트 진행 방향(-Y)으로 이동한다. 속도 프로파일은
        [등속 v_belt] → [FOLLOW_DECEL_T 동안 정지까지 감속]: 실링이 걸리는 앞부분은
        물체와 상대속도 0이고, 뒷부분 감속은 lift의 정지 출발 계약을 맞춘다.

        각 waypoint는 직전 해를 시드로 IK를 풀어 같은 wrist branch를 유지한다
        (push 스트로크와 동일 — 시드 없이 풀면 J5 부호 교차에서 4/6축이 π 튄다).
        IK가 중간에 실패하면 거기서 잘라 반환한다 (도달 한계 → 짧게 추종).

        리턴: (traj (6,n), vel (6,n), ts (n,)) 또는 None (벨트 정지/생성 불가).
        """
        ctx = self.ctx
        if v_belt < 1e-3 or PICK_TIME <= 0.0:
            return None                      # 벨트 정지 → 추종할 것이 없다
        dt = 1.0 / ctx.cfg.TRAJ_HZ
        n_steps = max(2, int(round(PICK_TIME / dt)))
        t_dec = min(FOLLOW_DECEL_T, PICK_TIME)
        t_cruise = PICK_TIME - t_dec

        waypoints = []
        q_seed = np.asarray(press_joint, dtype=float)
        for i in range(n_steps + 1):
            t = min(i * dt, PICK_TIME)
            if t <= t_cruise:
                s = v_belt * t               # 등속 추종 (물체와 상대속도 0)
            else:
                tau = t - t_cruise           # 선형 감속 v→0
                s = v_belt * (t_cruise + tau - 0.5 * tau * tau / t_dec)
            T_wp = np.asarray(T_press, dtype=float).copy()
            T_wp[1, 3] -= s                  # 벨트는 -Y로 진행
            T_wp[2, 3] = PRESS_Z
            ik = ctx.robot.inverse_kinematics(T_wp, q_init=q_seed)
            if ik is None:
                ctx.log.warn(
                    f"belt-follow IK 실패 @ {i}/{n_steps} (s={s * 1000:.0f}mm) — "
                    f"추종을 {len(waypoints)} waypoint로 자름"
                )
                break
            q_seed = np.asarray(ik, dtype=float)
            waypoints.append(q_seed)

        if len(waypoints) < 3:
            ctx.log.warn("belt-follow 세그먼트 생성 실패 — 정지 유지로 폴백")
            return None

        n = len(waypoints)
        traj = np.column_stack(waypoints)                  # (6, n)
        ts = np.linspace(0.0, dt * (n - 1), n)
        vel = np.zeros_like(traj)
        vel[:, 1:-1] = (traj[:, 2:] - traj[:, :-2]) / (2.0 * dt)
        # 시작은 벨트 속도로 진입(하강 세그먼트가 이 속도로 착지하도록 전달),
        # 끝은 정지 (lift가 정지 출발).
        vel[:, 0] = (traj[:, 1] - traj[:, 0]) / dt
        vel[:, -1] = 0.0

        # 관절 속도 한계 확인. 시간을 늘리면 벨트 추종이 깨지므로 stretch하지 않고
        # 경고만 남긴다 (벨트 속도에서 넘칠 일은 없지만, 넘으면 추종 자체가 불가).
        seg = np.abs(np.diff(traj, axis=1)) / dt
        ratio = float(np.max(seg / np.asarray(ctx.M1[:6], dtype=float)[:, None]))
        if ratio > 1.0:
            ctx.log.warn(
                f"belt-follow 관절속도 한계 초과 (max {ratio:.2f}×) — 추종 포기")
            return None
        return traj, vel, ts

    def _tcp_z_joint(self, T_grasp: np.ndarray, z_abs: float, seed_joint: np.ndarray):
        """grasp TCP의 XY/자세를 유지한 채 절대 높이 ``z_abs``의 IK 해 (로봇 규약,
        seed_joint 시드 → 같은 IK branch). 실패 시 None (호출측이 폴백)."""
        T_h = np.asarray(T_grasp, dtype=float).copy()
        T_h[2, 3] = float(z_abs)
        q = self.ctx.robot.inverse_kinematics(
            T_h, q_init=np.asarray(seed_joint, dtype=float)[:6]
        )
        return None if q is None else np.asarray(q, dtype=float)

    # ------------------------------------------------------------------
    # Skill entry point (ambush strategy)
    # ------------------------------------------------------------------
    def execute(self, request: "PickRequest") -> SkillResult:
        """Pre-position (mode-dependent), wait for arrival + suction, then throw."""
        ctx = self.ctx
        target = request.target
        current_joint = request.current_joint
        aim_joint = request.aim_joint
        grasp_joint = request.grasp_joint
        T_aim = request.T_aim
        T_grasp = request.T_grasp
        secondary = request.secondary

        mode = PICK_WAIT_MODE.get(target.class_name, DEFAULT_PICK_WAIT_MODE)

        # Start clean (uniform per-object flow: no cross-cycle suction hand-off).
        ctx.traj_ctrl.suction_off()

        # HOVER_DESCEND(기본): grasp + HOVER_ABOVE 높이에서 대기하고, 물체 도착
        # 순간 절대 높이 PRESS_Z까지 하강해 위에서 찍어 누른다. 대기 종료
        # (start_lead)를 하강 시간만큼 앞당겨 컵-물체 접촉이 도착 시각에 오도록
        # 한다. suction은 position_and_prime이 hover에 파킹된 상태에서 프라임 →
        # 접촉과 동시 실링. 프레스가 grasp보다 낮게 끝나므로 던지기 lift의 시작
        # 관절(start_q)도 프레스 자세로 맞춘다 (아니면 던지기 시작에서 점프).
        # WAIT_AT_GRASP(폴백/클래스별): grasp 높이에 파킹하고 도착을 받는다.
        wait_joint = np.asarray(grasp_joint, dtype=float)
        start_q = np.asarray(grasp_joint, dtype=float)   # 던지기 lift 시작 관절
        T_lift_ref = np.asarray(T_grasp, dtype=float)    # p_lift 기준 (추종 후 이동)
        descend = None                       # (d_traj, d_vel, d_ts) | None
        follow = None                        # (f_traj, f_vel, f_ts) | None
        if mode == PickWaitMode.HOVER_DESCEND:
            q_hover = self._tcp_z_joint(
                T_grasp, T_grasp[2, 3] + HOVER_ABOVE, grasp_joint)
            q_press = self._tcp_z_joint(T_grasp, PRESS_Z, grasp_joint)
            if q_hover is None or q_press is None:
                ctx.log.warn("hover/press IK 실패 — WAIT_AT_GRASP로 폴백")
            else:
                zero6 = np.zeros(6)
                # 벨트 추종 세그먼트를 먼저 만들어, 하강이 '벨트 속도로 착지'하게
                # 그 진입 관절속도를 하강의 종료속도로 넘긴다 — 접촉 순간 컵과
                # 물체의 상대속도가 0이 되어 실링 중 끌림이 없다.
                follow = self._build_belt_follow(
                    T_grasp, q_press, float(ctx.conveyor.current))
                dq_press = follow[1][:, 0] if follow is not None else zero6
                descend = trajectory(
                    q_hover, zero6, q_press, dq_press,
                    ctx.M1[:6], ctx.M2[:6], hertz=ctx.cfg.TRAJ_HZ,
                )
                wait_joint, start_q = q_hover, q_press
                if follow is not None:
                    # 추종이 끝난 자세에서 던지기가 출발한다. p_lift 기준도 그만큼
                    # 하류로 옮기되 lift 높이 컨벤션(grasp z + THROW_LIFT)은 유지.
                    start_q = follow[0][:, -1]
                    T_end = ctx.robot.forward_kinematics(start_q)
                    T_lift_ref = np.asarray(T_grasp, dtype=float).copy()
                    T_lift_ref[0, 3], T_lift_ref[1, 3] = T_end[0, 3], T_end[1, 3]
                    ctx.log.info(
                        f"belt-follow: {float(ctx.conveyor.current):.3f} m/s로 "
                        f"{float(follow[2][-1]):.2f}s 추종 "
                        f"({(T_grasp[1, 3] - T_end[1, 3]) * 1000:.0f}mm 하류)"
                    )

        # 계획-대기 겹치기: 고정 bin 클래스는 착지점이 시간 불변이므로 intercept가
        # 확정된 지금 NLP solve를 백그라운드로 시작한다 — 이동+hover 대기 시간과
        # 겹쳐서, 프레스 직후 join만 하면 됨 (eta > solve 시간이면 체감 지연 ~0).
        # casadi solve는 GIL을 해제(실측: solve 중 메인 스레드 처리율 101%)하므로
        # 250 Hz 스트림/대기 루프를 방해하지 않는다. bin 미등록 클래스는 착지점이
        # 시간 종속(fallback)이라 기존대로 프레스 후 순차로 푼다.
        bin_xyz = THROW_BIN_TARGET_MAP.get(target.class_name)
        plan_box: dict = {}
        plan_thread = None
        if bin_xyz is not None:
            p_target_fixed = np.asarray(bin_xyz, dtype=float)
            ctx.log.info(
                f"Throw target for {target.class_name}: fixed bin "
                f"({p_target_fixed[0]:+.3f}, {p_target_fixed[1]:+.3f}, "
                f"{p_target_fixed[2]:+.3f}) m — NLP 선계획 백그라운드 시작"
            )

            def _plan_bg(_q=start_q, _T=T_lift_ref) -> None:
                try:
                    plan_box["planned"] = self.plan_nlp_throw(
                        _q, _T, p_target_fixed)
                except Exception as e:   # 스레드 예외는 join 후 회수 (넘기면 유실)
                    plan_box["error"] = e

            plan_thread = threading.Thread(
                target=_plan_bg, name="nlp-preplan", daemon=True)
            plan_thread.start()

        start_lead = self.arrival_lead()
        if descend is not None:
            t_desc = float(descend[2][-1])
            # position_and_prime은 suction cap(arrival - SUCTION_LEAD)보다 일찍
            # 리턴하지 않으므로, 하강+디스패치가 SUCTION_LEAD 안에 들어와야
            # 접촉이 도착 정시에 온다 (넘치면 그만큼 늦게 눌린다).
            if t_desc + start_lead > ctx.cfg.SUCTION_LEAD:
                ctx.log.warn(
                    f"press 하강 {t_desc:.2f}s + lead {start_lead:.2f}s > "
                    f"SUCTION_LEAD {ctx.cfg.SUCTION_LEAD:.2f}s — 접촉 지연"
                )
            start_lead += t_desc

        ctx.set_status("POSITIONING", target.class_name)
        ctx.position_and_prime(
            current_joint, aim_joint, wait_joint, target, T_grasp[1, 3],
            start_lead=start_lead,
        )
        if descend is not None:
            # 도착 순간 프레스 하강 + (있으면) 벨트 추종을 하나의 궤적으로 이어
            # 디스패치한다: 두 번 나눠 보내면 사이에 디스패치 간극이 생기고 그동안
            # 물체가 계속 흘러간다. 추종의 첫 열은 하강의 마지막(q_press)과 중복이라
            # 버리고 시계만 이어 붙인다 (push 스트로크와 동일한 접합 방식).
            # 스트림이 실시간 페이싱하므로 이 호출 자체가 하강+PICK_TIME을 소모.
            d_traj, d_vel, d_ts = descend
            if follow is not None:
                f_traj, f_vel, f_ts = follow
                d_traj = np.concatenate((d_traj, f_traj[:, 1:]), axis=1)
                d_vel = np.concatenate((d_vel, f_vel[:, 1:]), axis=1)
                d_ts = np.concatenate((d_ts, f_ts[1:] + d_ts[-1]))
            ctx.traj_ctrl.send_trajectory_queue(
                d_traj, d_vel, d_ts, final_joint=start_q,
            )
            if follow is None:
                # 벨트 정지/IK 불가 → 그 자리에서 유지 (JGPC가 마지막 명령을
                # zero-order-hold; sleep_until은 신규 detection 인입을 계속).
                ctx.sleep_until(time.time() + PICK_TIME)

        # DIAGNOSTIC: object vs intercept at the instant the lift/throw fires.
        ctx.log_action_timing(target, T_grasp[1, 3], "throw-lift")

        # Lift + throw.
        ctx.set_status("THROWING", target.class_name)

        # 계획 회수: 선계획 스레드가 있으면 join (이동+대기와 겹쳐 이미 풀렸으면
        # 즉시 반환), 없으면(legacy fallback 착지점 — 시간 종속) 여기서 순차로 푼다.
        if plan_thread is not None:
            t_join0 = time.time()
            plan_thread.join()
            join_s = time.time() - t_join0
            err = plan_box.get("error")
            if err is not None:
                ctx.log.error(
                    f"NLP 선계획 스레드 예외: {type(err).__name__}: {err}")
            planned = plan_box.get("planned")
            ctx.log.info(f"NLP 선계획 회수: press 후 추가 대기 {join_s:.2f}s")
        else:
            # legacy fallback (secondary 예측 위치 또는 T_aim hover). theta는
            # fallback 전용 — planner는 방향을 p_object→p_target으로 스스로 잡음.
            theta = float(np.arctan2(
                THROW_BIN_Y - T_grasp[1, 3], THROW_BIN_X - T_grasp[0, 3],
            ))
            T_aim2 = self.plan_throw_landing(T_grasp, theta, T_aim, time.time(), secondary)
            p_target = T_aim2[:3, 3].copy()
            planned = self.plan_nlp_throw(start_q, T_grasp, p_target)

        # NLP throw 계획: lift + 통합 궤적. 불가능하면 던지지 않고 정리.
        if planned is None:
            ctx.log.warn("Throw infeasible; dropping in place and parking")
            ctx.traj_ctrl.suction_off()
            ctx.set_active_target(None)
            ctx.set_status("IDLE", "")
            return SkillResult(False, "throw infeasible for this target")
        res, lift = planned

        # Chain the follow-through toward the NEXT object's grasp (best-effort) so the arm
        # OVERLAPS the next approach with this throw instead of parking far and re-driving
        # serially. Symmetric + stateless: the next epoch still SELECTS + DRIVES fresh from
        # this closer pose (no commit/preposition). None -> lifted-standby park.
        throw_T = float(lift[2][-1] + res["t_f"])  # lift + NLP 전체 궤적
        # next_chain_target은 (grasp_joint(6,), cand) tuple 또는 None을 반환한다
        # (push 머지로 바뀐 계약 — 예전엔 grasp_joint만 줬다). NLP 스로워는 아직
        # chain_park_joint(다음 물체의 스킬이 지정하는 park 자세)를 쓰지 않으므로
        # grasp만 취한다. throw_skill처럼 park까지 존중하려면 cand로
        # ctx.skill_obj_for(cand).chain_park_joint(...)를 부르면 된다.
        nxt = ctx.next_chain_target(grasp_joint, throw_T)
        next_grasp = nxt[0] if nxt is not None else None
        self.build_throw_trajectory(
            grasp_joint, res, lift, next_grasp=next_grasp,
        )
        ctx.traj_ctrl.suction_off()   # release the object after the throw
        self._log_throw_cycle(target)
        ctx.set_active_target(None)
        ctx.set_status("IDLE", "")
        return SkillResult(True, "throw complete")

    # ------------------------------------------------------------------
    # Throw planning
    # ------------------------------------------------------------------
    def plan_throw_landing(
        self,
        T_grasp1: np.ndarray,
        theta: float,
        T_aim1_fallback: np.ndarray,
        now: float,
        secondary: "Optional[TrackedObject]",
    ) -> np.ndarray:
        """Aim throw at ``secondary`` if feasible; else drop in place.

        ``secondary`` is captured by the caller *before* the lock step so
        this method does not depend on the queue's mutation order.
        """
        ctx = self.ctx
        if secondary is None:
            return T_aim1_fallback.copy()

        T_aim2, _, _, neg_wait2 = ctx.planner.plan_throw_landing(
            T_grasp1,
            secondary.T_aim_base.copy(),
            theta,
            secondary.detect_time,
            ctx.conveyor.current,
            now,
            fixed_delay=ctx.cfg.FIXED_DELAY_THROW,
        )
        infeasible = (
            neg_wait2 is not None
            or T_aim2[0, 3] < 0.1
            or T_aim2[2, 3] < 0.0
            or T_aim2[2, 3] > ctx.cfg.MAX_REACH
        )
        if infeasible:
            return T_aim1_fallback.copy()
        return T_aim2

    def plan_nlp_throw(
        self,
        start_joint: np.ndarray,
        T_grasp: np.ndarray,
        p_target: np.ndarray,
    ):
        """THR 통합 NLP planner 호출 + lift 세그먼트 생성.

        ``start_joint``: lift가 출발하는 실제 관절 자세 (로봇 규약) — HOVER_DESCEND
        픽에서는 프레스 자세(q_press, TCP z=PRESS_Z), 폴백에서는 grasp_joint.
        리턴: (res dict, (lift_traj (6,n), lift_vel (6,n), lift_ts (n,)))
        또는 None (실현 불가 — 사유는 로그).
        던지기 시작 TCP는 grasp 위치 + THROW_LIFT (벨트 clearance).
        q_lift IK seed = start_joint이라 현재 자세 근처 branch로 잡히고,
        NLP는 q_start=q_lift에서 정지 출발 (v0=0) — lift가 정지로 끝나므로 연속.
        """
        ctx = self.ctx
        p_lift = T_grasp[:3, 3] + np.array([0.0, 0.0, THROW_LIFT])
        p_target = np.asarray(p_target, dtype=float)

        # lift 자세: NLP 위치 한계(B≥10°, U≤45°, |R|≤80° 포함) 안의 IK 해.
        # ik_position/Q_LO/Q_HI는 플래너 규약이므로 로봇 규약인 start_joint를
        # _PLANNER_SIGN으로 변환해 시드로 쓴다 (고정 시드들은 원래 플래너 규약).
        # lift IK seed: 목표 방향으로 S 를 잡고(atan2(y,x)) 어깨 L 을 낮게·손목 B 를
        # 넓게 뿌린다. 구 seed(start_joint + b∈{0.6,0.9,1.2}, S 고정)는 조인 한계
        # (J2≤30/J3≤45)에서 코너 자세(예 0.5,0.3)를 전부 L>30 basin 으로 흘려보내
        # '한계 내 해 없음'을 냈지만(2026-07-24 실기 재현), 해 자체는 존재한다 —
        # 아래 seed 집합은 실측상 12개 lift 자세 12/12 를 한계 안에서 찾는다.
        q_lift = None
        s_yaw = float(np.clip(np.arctan2(p_lift[1], p_lift[0]), Q_LO[0], Q_HI[0]))
        seeds = [np.asarray(start_joint, dtype=float) * _PLANNER_SIGN]
        seeds += [np.array([s_yaw, L, U, 0.0, b, 0.0])
                  for L in (0.2, 0.4) for U in (0.1, -0.2) for b in (0.2, 0.5, 0.9, 1.2)]
        for seed in seeds:
            q, ok = ik_position(p_lift, seed)
            if ok and np.all(q >= Q_LO) and np.all(q <= Q_HI):
                q_lift = q
                break
        if q_lift is None:
            ctx.log.warn("NLP throw planning failed: lift 자세 IK 불가 (한계 내 해 없음)")
            return None

        # ---- 해 탐색: ① 세션 warm 캐시 → ② 파일 warm DB → ③ cold multistart ----
        # warm은 full warm start polish (~1-2s), cold는 초기해 순차 재시도 (수 초/개).
        # 모든 해는 dispatch 전 _solution_gates(착탄/한계/속도 독립 재검증)를 통과해야 함.
        t_plan0 = time.time()
        cache_key = tuple(np.round(p_target, 3))
        warm_cands = []
        if cache_key in self._warm_cache:
            warm_cands.append(("cache", self._warm_cache[cache_key]))
        db = _load_warm_db(ctx.log)
        res, how = None, ""

        # ---- ① exact-match 고속 경로: 저장된 최적 궤적을 그대로 사용 (재풀이 생략) --
        # 이 (시작자세 p_lift, target) 쌍의 해가 DB에 이미 있으면 NLP 재풀이(실측
        # 3~14s)를 건너뛰고 저장 궤적(P, t_f, t_star)을 그대로 dispatch 한다 —
        # 결정적 static test(런타임 쌍 = 빌더 PAIRED 쌍)에선 계획 시간이 ~0.
        # 저장 해는 빌더 게이트를 이미 통과했지만 실기 안전상 _solution_gates 로
        # 재검증하고, q_lift 를 entry 의 q_start 로 맞춰 lift 가 저장 arc(P[0]=q_start)
        # 와 연속이 되게 한다. (매칭 허용 2mm — 같은 산술로 만든 좌표라 실질 정확.)
        if db:
            for e in db:
                ps = e.get("p_start")
                if ps is None:
                    continue
                if (float(np.linalg.norm(np.asarray(e["target"]) - p_target)) < 2e-3
                        and float(np.linalg.norm(np.asarray(ps, float) - p_lift)) < 2e-3):
                    cand = dict(
                        P=np.asarray(e["P"], float), t_f=float(e["t_f"]),
                        t_star=float(e["t_star"]), J=float(e["J"]),
                        release_time=THROW_WINDOW_T, lam_g=e.get("lam_g"),
                        u_pos=e.get("u_pos"),
                        pos_viol_dense=float(e.get("pos_viol_dense", 0.0)))
                    why = self._solution_gates(cand, p_target)
                    if why is None:
                        res, how = cand, "exact-DB"
                        q_lift = np.asarray(e["q_start"], dtype=float)
                        break
                    ctx.log.info(f"exact-DB 게이트 기각: {why} — polish 로 진행")

        # ---- ② 세션 캐시 + 파일 DB 최근접으로 warm start polish (재풀이) ----
        if res is None:
            if db:
                # 후보 선정: target XY 근접이 1차, 시작 자세(p_start) 근접이 2차.
                # 정확 일치가 아닐 때(쌍이 어긋난 경우) 그나마 가까운 start 를 고른다.
                def _cost(e):
                    dt = float(np.linalg.norm(np.asarray(e["target"])[:2] - p_target[:2]))
                    ps = e.get("p_start")
                    ds = 0.0 if ps is None else float(
                        np.linalg.norm(np.asarray(ps, float) - p_lift))
                    return dt + 0.3 * ds
                near = sorted(db, key=_cost)[:2]
                warm_cands += [(f"db#{i}", e) for i, e in enumerate(near)]
            for tag, ent in warm_cands:
                try:
                    cand = solve_throw_nlp(None, p_target, release_time=THROW_WINDOW_T,
                                           q_start=q_lift, warm_data=ent)
                except (RuntimeError, ValueError, AssertionError) as e:
                    ctx.log.info(f"warm[{tag}] 불발: {type(e).__name__}")
                    continue
                why = self._solution_gates(cand, p_target)
                if why is None:
                    res, how = cand, f"warm[{tag}]"
                    break
                ctx.log.info(f"warm[{tag}] 게이트 기각: {why}")

        if res is None:
            for k, init in enumerate(INIT_VARIANTS_ROS):
                try:
                    cand = solve_throw_nlp(None, p_target, release_time=THROW_WINDOW_T,
                                           q_start=q_lift, init=init)
                except (RuntimeError, ValueError, AssertionError) as e:
                    ctx.log.info(f"cold[init{k}] 불발: {type(e).__name__}")
                    continue
                why = self._solution_gates(cand, p_target)
                if why is None:
                    res, how = cand, f"cold[init{k}]"
                    break
                ctx.log.info(f"cold[init{k}] 게이트 기각: {why}")

        if res is None:
            ctx.log.warn("NLP throw planning failed: warm/cold 전 후보 기각")
            return None

        # 성공 해를 세션 warm 캐시에 저장 — 같은 bin 재던지기는 다음부터 polish
        self._warm_cache[cache_key] = dict(
            target=p_target.copy(), p_start=p_lift.copy(), P=res["P"],
            t_f=res["t_f"], t_star=res["t_star"], lam_g=res.get("lam_g"),
            u_pos=res.get("u_pos"), J=res["J"])

        # lift: 정지(start_joint) → 정지(q_lift). trajectory()가 M1/M2 안에서
        # time-optimal 세그먼트를 만들어 줌. 스트리밍되는 세그먼트는 로봇 규약
        # 이어야 하므로 q_lift(플래너 규약)를 변환한 끝점을 쓴다 — NLP의
        # q_start=q_lift(플래너 규약)와는 별개다.
        zero6 = np.zeros(6)
        l_traj, l_vel, l_ts = trajectory(
            np.asarray(start_joint, dtype=float), zero6,
            q_lift * _PLANNER_SIGN, zero6,
            ctx.M1[:6], ctx.M2[:6], hertz=ctx.cfg.TRAJ_HZ,
        )
        ctx.log.info(
            f"NLP throw [{how}, {time.time() - t_plan0:.1f}s]: "
            f"t_f={res['t_f']:.3f}s t*={res['t_star']:.3f}s "
            f"window={res['release_time']:.3f}s J={res['J']:.3f} "
            f"lift={l_ts[-1]:.3f}s (release anytime in window)"
        )
        return res, (l_traj, l_vel, l_ts)

    def _solution_gates(self, res: dict, p_target: np.ndarray) -> "Optional[str]":
        """dispatch 전 독립 재검증 (scipy/numpy 경로 — solver를 믿지 않는다).
        통과 시 None, 기각 시 사유 문자열."""
        # ① 위치 한계 dense 잔존 위반 (hull이 보장하지만 이중 확인; 0.06° 허용)
        if res.get("pos_viol_dense", 0.0) > 1e-3:
            return f"위치한계 잔존 위반 {res['pos_viol_dense']:.1e}"
        q_of, qd_of, _ = _spline_eval(res["P"], res["t_f"])
        # ② 속도 한계 (NLP에서 hard지만 실기 안전상 독립 재확인)
        qd_ratio = max(np.max(np.abs(qd_of(t)) / GP8_QD_MAX)
                       for t in np.linspace(0.0, res["t_f"], 400))
        if qd_ratio > 1.0:
            return f"속도한계 초과 max|q̇|/limit={qd_ratio:.2f}"
        # ③ 착탄 게이트: 윈도우 dense 착탄오차 — penalty 공식화라 solver 보장이
        #    아니고, 나쁜 basin은 윈도우 일부에서 inf가 나옴 (실기 필수 게이트).
        #    발사점은 NLP와 동일하게 launch_state(TCP + 로드축 GRIP_OFF, ω×r 포함
        #    강체속도) — bare TCP로 재검증하면 2cm/ω×r 만큼 어긋나 정상 해가 착탄
        #    게이트에 잘못 걸린다. drag는 실물 미배선(None=무항력)이라 게이트도
        #    항력 없는 폐형 landing_error로 일관 (NLP도 drag=0 = 포물선 등가).
        rt = res["release_time"]
        errs = []
        for t in res["t_star"] + np.linspace(-rt / 2, rt / 2, 11):
            q, qd = q_of(t), qd_of(t)
            p_eff, v_eff = launch_state(q, qd)
            errs.append(landing_error(p_eff, v_eff, p_target))
        e_max = float(np.max(errs))
        if not np.isfinite(e_max) or e_max > LANDING_GATE:
            return f"착탄 게이트 초과 (window max {e_max * 1e3:.0f}mm > {LANDING_GATE * 1e3:.0f}mm)"
        # ④ Cartesian 안전 엔벨로프: 최적화가 끝난 '궤적'을 보고 기둥/베이스
        #    (x ≤ MIN_TCP_X)나 바닥/벨트(z ≤ MIN_TCP_Z)를 침범하면 이 후보를 기각.
        #    NLP는 바닥 클리어런스를 hard 제약으로 걸지 않고(걸면 basin이 전멸)
        #    기둥 회피도 release 창까지만 활성이라, 감속 꼬리가 지하로 다이브하는
        #    해가 정상 수렴한다 — 그래서 사후 검사로 거른다.
        #    **여기서 기각해야 plan_nlp_throw가 다음 warm/cold 후보를 재시도한다.**
        #    dispatch 직전 게이트(build_throw_trajectory)만 있으면 재시도 없이 그
        #    사이클이 버려진다. 빌더(tools/build_warm_db._gates)와 동일 기준.
        #    fk_pos는 플래너 규약 FK (로봇 FK와 0.000mm 일치 검증됨).
        P_arc = np.array([fk_pos(q_of(t))
                          for t in np.linspace(0.0, res["t_f"], 300)])
        x_min, z_min, z_max = (float(P_arc[:, 0].min()), float(P_arc[:, 2].min()),
                               float(P_arc[:, 2].max()))
        if x_min <= MIN_TCP_X or z_min <= MIN_TCP_Z or z_max > MAX_TCP_Z:
            return (f"Cartesian 엔벨로프 위반 (x_min={x_min:+.3f}m, "
                    f"z_min={z_min:+.3f}m, z_max={z_max:+.3f}m; "
                    f"한계 x>{MIN_TCP_X:.2f}, {MIN_TCP_Z:.2f}<z<{MAX_TCP_Z:.2f})")
        return None

    # ------------------------------------------------------------------
    # Throw trajectory build + dispatch
    # ------------------------------------------------------------------
    def build_throw_trajectory(
        self,
        grasp_joint: np.ndarray,
        res: dict,
        lift,
        next_grasp: "Optional[np.ndarray]" = None,
        chain_park: "Optional[np.ndarray]" = None,
    ) -> None:
        """Build and dispatch: lift + NLP 통합 궤적 + chain-to-park.

        NLP 궤적은 solver가 이미 보장함 —
          * 윈도우(t*±T/2) 내내 착지 조건 최적 (아무 때나 release해도 착지)
          * 관절 위치·속도·가속도 한계 + 기둥 회피 전 구간 hard 만족, v0=vf=0
        이라서 NN 버전의 사후 velocity-clamp 루프가 필요 없음. lift/chain은
        trajectory()가 M1/M2로 bound하는 rest-to-rest 이동 (NLP가 감속을
        포함하므로 chain은 정지에서 출발).

        release는 윈도우 중앙 t* 샘플에 걸어서 suction 밸브 지연/jitter가
        ±window/2 안이면 착지가 보장되게 함 (RELEASE_LEAD로 명령 시점 선행).
        """
        ctx = self.ctx
        l_traj, l_vel, l_ts = lift
        dt = 1.0 / ctx.cfg.TRAJ_HZ

        # NLP 전체 궤적 샘플 (첫 샘플은 lift 끝(q_lift, 정지)과 동일 상태라 drop).
        # NLP 해는 플래너 규약이므로 스트리밍 전에 _PLANNER_SIGN으로 로봇 규약
        # 변환 (lift 끝 = q_lift*S 와 연속 — 첫 drop 샘플이 정확히 그 점).
        q_of, qd_of, _ = _spline_eval(res["P"], res["t_f"])
        t_nlp = np.arange(0.0, res["t_f"] + dt / 2, dt)
        arc_traj = _PLANNER_SIGN[:, None] * np.stack(
            [q_of(t) for t in t_nlp], axis=1)[:, 1:]                   # (6, n)
        arc_vel = _PLANNER_SIGN[:, None] * np.stack(
            [qd_of(t) for t in t_nlp], axis=1)[:, 1:]
        arc_ts = t_nlp[1:] + l_ts[-1]

        # 윈도우 [t*−T/2, t*+T/2]의 concat 기준 index 범위
        rt = res["release_time"]
        off = l_traj.shape[1] - 1                  # t_nlp[k] → concat index (drop 보정)
        win_lo = off + max(1, int(np.ceil((res["t_star"] - rt / 2) / dt)))
        win_hi = off + int(np.floor((res["t_star"] + rt / 2) / dt))
        win_hi = max(win_hi, win_lo)

        # release: 물리적 release가 윈도우 중앙(t*)에 오도록 RELEASE_LEAD만큼
        # 앞당겨 명령. 어느 쪽으로 밀려도 윈도우 안(±T/2)이면 착지 보장.
        lead_steps = int(round(ctx.cfg.RELEASE_LEAD * ctx.cfg.TRAJ_HZ))
        release_idx = (win_lo + win_hi) // 2 - lead_steps
        release_idx = max(win_lo, min(release_idx, win_hi))

        # Chain: NLP 끝(정지)에서 park까지 rest-to-rest 이동. 목적지 우선순위는
        # NN 버전과 동일 (next skill park > over next grasp > idle > standby).
        if chain_park is not None:
            chain_target = np.asarray(chain_park, dtype=float).copy()
            chain_dest = "next action-start park"
        elif next_grasp is not None:
            # Park OVER the next grasp (its XY raised to home Z), NOT at belt
            # height — a belt-height endpoint would sweep the TCP low across
            # the belt on the next drive (floor-dip / grazing).
            chain_target = ctx.lifted_standby_joint(next_grasp)
            chain_dest = "over next grasp"
        elif not ctx.queue:
            chain_target = self.idle_target().copy()
            chain_dest = "home/idle (queue empty)"
        else:
            chain_target = ctx.lifted_standby_joint(arc_traj[:, -1])
            chain_dest = "lifted standby"

        zero6 = np.zeros(6)
        traj_chain, vel_chain, ts_chain = trajectory(
            arc_traj[:, -1], zero6,               # NLP 끝은 vf=0 (구조적 보장)
            chain_target[:6], zero6,
            ctx.M1[:6], ctx.M2[:6], hertz=ctx.cfg.TRAJ_HZ,
        )
        # Drop the chain's first column — it duplicates the NLP's last
        # sample. Slice traj/vel/ts the SAME way so the three stay equal
        # length (a lone-ts slice would misalign _build_queue_waypoints).
        traj_chain = traj_chain[:, 1:]
        vel_chain = vel_chain[:, 1:]
        ts_chain_shifted = ts_chain[1:] + arc_ts[-1]

        traj_throw = np.concatenate((l_traj, arc_traj, traj_chain), axis=1)  # (6, total)
        vel_throw = np.concatenate((l_vel, arc_vel, vel_chain), axis=1)
        timestep_throw = np.concatenate((l_ts, arc_ts, ts_chain_shifted))
        assert traj_throw.shape[1] == vel_throw.shape[1] == timestep_throw.shape[0], (
            f"throw traj/vel/timestep length mismatch: "
            f"{traj_throw.shape[1]}/{vel_throw.shape[1]}/{timestep_throw.shape[0]}"
        )
        final_joint = chain_target

        # SAFETY GATE (로봇 규약): 변환·연결된 전체 궤적이 실기 관절 한계
        # (robots/gp8.joint_limits — J5 상한 +60.8°는 펜던트 실측 충돌 한계)
        # 안인지 독립 확인 후에만 dispatch. 플래너의 위치 한계 보장은 플래너
        # 규약 기준이라 변환 실수·모델 불일치를 잡지 못한다.
        jl = np.asarray(ctx.robot.joint_limits, dtype=float)   # (6, 2) lo/hi
        viol = (traj_throw < jl[:, :1] - 1e-6) | (traj_throw > jl[:, 1:2] + 1e-6)
        if viol.any():
            j, k = (int(v) for v in np.argwhere(viol)[0])
            ctx.log.error(
                f"NLP throw ABORTED (not dispatched): robot-frame joint {j + 1} "
                f"limit violation at sample {k} "
                f"({np.rad2deg(traj_throw[j, k]):+.1f}°, "
                f"limits [{np.rad2deg(jl[j, 0]):+.1f}, {np.rad2deg(jl[j, 1]):+.1f}]°)"
            )
            return

        # SAFETY GATE 2 (Cartesian, 로봇 규약 FK): TCP가 기둥/베이스(x ≤ MIN_TCP_X)나
        # 바닥/벨트(z ≤ MIN_TCP_Z)로 들어가거나 너무 높이(z > MAX_TCP_Z) 뜨면
        # 디스패치하지 않는다.
        # z는 픽 자세(프레스 TCP z=PRESS_Z, 0.04보다 낮음)에서 출발하므로, lift가
        # 처음 MIN_TCP_Z를 넘어선 '이후' 구간에만 적용한다 — 출발점 자체는 위반이
        # 아니고, 한 번 벗어난 뒤 다시 내려오는 것(=꼬리 다이브)만 잡는다.
        # x는 전 구간 적용 (픽/던지기 어느 단계에서도 베이스에 파고들 이유가 없다).
        n_s = traj_throw.shape[1]
        tcp = np.stack(
            [np.asarray(ctx.robot.forward_kinematics(traj_throw[:, k]),
                        dtype=float)[:3, 3] for k in range(n_s)], axis=1)   # (3, n)
        bad: list[tuple[int, str]] = []
        cleared = np.nonzero(tcp[2] > MIN_TCP_Z)[0]
        if cleared.size == 0:
            bad.append((0, f"궤적 전체가 z ≤ {MIN_TCP_Z:.3f}m (lift가 바닥을 못 벗어남)"))
        else:
            k0 = int(cleared[0])
            dip = np.nonzero(tcp[2, k0:] <= MIN_TCP_Z)[0]
            if dip.size:
                k = k0 + int(dip[0])
                bad.append((k, f"TCP z={tcp[2, k]:+.4f}m ≤ {MIN_TCP_Z:.3f}m (바닥/벨트)"))
        near = np.nonzero(tcp[0] <= MIN_TCP_X)[0]
        if near.size:
            k = int(near[0])
            bad.append((k, f"TCP x={tcp[0, k]:+.4f}m ≤ {MIN_TCP_X:.3f}m (기둥/베이스)"))
        # z 상한(팔 과다 상승)은 전 구간 적용 — 출발/lift 어디서도 0.85 위로 갈 이유 없다.
        high = np.nonzero(tcp[2] > MAX_TCP_Z)[0]
        if high.size:
            k = int(high[0])
            bad.append((k, f"TCP z={tcp[2, k]:+.4f}m > {MAX_TCP_Z:.3f}m (팔 과다 상승)"))
        if bad:
            k, why = min(bad)                      # 가장 이른 위반 지점
            n_viol = int(np.count_nonzero(
                (tcp[0] <= MIN_TCP_X) | (tcp[2] <= MIN_TCP_Z) | (tcp[2] > MAX_TCP_Z)))
            T_bad = np.asarray(ctx.robot.forward_kinematics(traj_throw[:, k]),
                               dtype=float)
            seg = ("lift" if k < l_traj.shape[1]
                   else "throw" if k < l_traj.shape[1] + arc_traj.shape[1] else "chain")
            ctx.log.error(
                f"NLP throw ABORTED (not dispatched): Cartesian 안전 엔벨로프 위반 — "
                f"{why} @ sample {k}/{n_s} (t={timestep_throw[k]:.3f}s, {seg} 구간, "
                f"위반 샘플 {n_viol}개)"
            )
            ctx.log.error(
                "  위반 지점 TCP SE3 (base frame):\n"
                + "\n".join("    [" + "  ".join(f"{v:+10.5f}" for v in row) + "]"
                            for row in T_bad)
            )
            ctx.log.error(
                "  위반 지점 관절 (robot frame, deg): "
                + ", ".join(f"J{j + 1}={np.rad2deg(traj_throw[j, k]):+.1f}"
                            for j in range(6))
            )
            return

        ctx.log.info(
            f"NLP throw dispatch: lift {l_traj.shape[1]} + throw "
            f"{arc_traj.shape[1]} + chain {traj_chain.shape[1]} samples, "
            f"release step {release_idx} (window [{win_lo},{win_hi}], "
            f"timing margin ±{0.5 * rt:.3f}s, "
            f"lead {ctx.cfg.RELEASE_LEAD:.2f}s, chain->{chain_dest})"
        )

        ctx.traj_ctrl.send_trajectory_queue_with_timed_release(
            traj_throw, vel_throw, timestep_throw,
            final_joint=final_joint,
            release_index=release_idx,
        )
        self._last_throw_meta = {
            "T": rt,
            "lift_T": float(l_ts[-1]),
            "t_f": res["t_f"],
            "t_star": res["t_star"],
            "J": res["J"],
            "release_idx": release_idx,
            "n_steps": traj_throw.shape[1],
        }

    # ------------------------------------------------------------------
    # Per-cycle timing log
    # ------------------------------------------------------------------
    def _log_throw_cycle(self, target: "TrackedObject") -> None:
        """Append one pick-cycle timing row to PICK_LOG_CSV: suction-on ->
        throw-start -> release, for offline analysis of the release timing."""
        ctx = self.ctx
        path = ctx.cfg.PICK_LOG_CSV
        if not path:
            return
        lt = getattr(ctx.traj_ctrl, "last_throw", {}) or {}
        meta = self._last_throw_meta or {}
        son = getattr(ctx.traj_ctrl, "last_suction_on_t", None)
        t0 = lt.get("throw_start")
        trel = lt.get("release_wall")

        def _d(a, b):
            return round(a - b, 4) if (a is not None and b is not None) else ""

        row = {
            "iso_time": datetime.datetime.now().isoformat(timespec="milliseconds"),
            "class": target.class_name,
            "belt_mps": round(ctx.conveyor.current, 4),
            "suction_on_t": round(son, 4) if son else "",
            "throw_start_t": round(t0, 4) if t0 else "",
            "release_t": round(trel, 4) if trel else "",
            "on_to_throwstart_s": _d(t0, son),
            "throwstart_to_release_s": _d(trel, t0),
            "window_T_s": round(meta.get("T", 0.0), 3),
            "lift_T_s": round(meta.get("lift_T", 0.0), 3),
            "t_f_s": round(meta.get("t_f", 0.0), 3),
            "t_star_s": round(meta.get("t_star", 0.0), 3),
            "J": round(meta.get("J", 0.0), 4),
            "release_idx": meta.get("release_idx", ""),
            "n_steps": meta.get("n_steps", ""),
            "io_ms": round(lt["io_ms"], 1) if lt.get("io_ms") is not None else "",
        }
        self._append_csv_row(path, row, ctx.log)

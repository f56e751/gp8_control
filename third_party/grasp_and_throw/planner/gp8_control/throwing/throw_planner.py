"""던지기 플래너 공개 API — **관절 경계 상태 → 관절 궤적 + suction 타이밍** (NLP, Ryu et al.).

입력 (전부 로봇/URDF 관절 규약, rad · rad/s · m — `robot/gp8.py` 와 같은 부호):

    start : ThrowBoundary(q, qd)         시작점 = 현재 잡은 물체 위치에서 lift 높이만큼 올라간 자세 (q),
                                          qd = **제자리에서 잡고 위로 올라가는** 관절 속도 = J⁻¹·[0, 0, +LIFT_SPEED]
    end   : ThrowBoundary(q, qd, t, q_rate) | None
                                          종착점 = 다음 잡을 물체 (q = 다음 물체를 잡는 자세), qd = **그 물체 위에서
                                          아래로 내려오며 벨트 속도에 맞춰 움직이는** 관절 속도 = J⁻¹·[v_belt + (0,0,−DESCEND_SPEED)].
                                          q_rate = 물체(=종점 자세)의 이동률 J⁻¹·[v_belt] — 시간 t 를 주면 종점은
                                          q + q_rate·(t_f − t) 로 t_f 와 함께 움직인다 (아래 '순환 해소').
                                          None 이면 종점 자세 자유·정지.
    p_target : 착지 조준점 (x, y, z) [m] — bin 개구부 위 조준면 (NlpThrowSkill: bin z + AIM_UP)

출력 ThrowPlan:
    t / q / qd / qdd     240 Hz 관절 궤적 (N,), (N,6) — t=0 이 start, t=t_f 가 end
    t_release            **suction OFF(릴리즈) 명령 시각** = 릴리즈 윈도우 중심 t*  (i_release = 격자 인덱스)
    release_window       (t*−ΔT/2, t*+ΔT/2): 이 안 어디서 놓아도 착탄 정확도가 유지되도록 최적화된 구간
    t_suction_on         다음 물체 도착(=t_f)에 맞춘 **suction ON 시각** (end 를 준 경우; 아니면 None)
    p_launch / v_launch / p_land / land_err   릴리즈 시점 발사 상태와 무항력 포물선 예측 착지·오차 (진단)

    plan.suction_events() → [(t_release, False), (t_f, True)]  — 시간순 (시각, 석션 on?)

순환 해소 (2026-09-18): 던지기 시간 t_f 를 알아야 다음 물체의 도착 위치(종점)를 알고, 종점을 알아야 t_f 가 나온다.
    물체는 벨트 위를 등속으로 흐르므로 "그 물체를 잡는 관절 자세" 도 시간에 대해 (거의) 선형이고 그 변화율이 바로
    벨트 추종 관절속도 qd 다. 그래서 종점을 상수가 아니라  q_f(t_f) = q(t_ref) + qd·(t_f − t_ref)  로 NLP 에 넣으면
    solver 가 t_f 와 종점을 동시에 결정한다 — 한 번의 solve, 반복 없음 (nlp/throw_nlp._assemble_nlp 참고).
    1차 근사 오차는 (t_f − t_ref)² 차수: 벨트 0.12 m/s, 0.15 s 어긋나도 1 mm 미만. plan.info["end_extrap_s"],
    ["end_tcp_err_m"] 로 확인한다.

물체 위치·속도에서 경계 상태를 만드는 헬퍼:
    object_boundary(p_obj, v_obj, v_tcp=…, lift=…, ik=…)
        TCP 위치(+lift) 의 관절 자세, qd = J⁻¹·[v_tcp; 0] (기본 v_obj), q_rate = J⁻¹·[v_obj; 0] (물체 이동률)
    plan_throw_between_objects(p_cur, v_cur, p_target, p_next, v_next, lift=…)
        현재 물체 → 다음 물체 체인 계획. 다음 물체의 지금 위치·속도를 받아 위의 움직이는 종점으로 한 번에 푼다.

프레임: NLP 내부는 planner 프레임(U·R·B·T 부호 반대, `nlp/planner.PLANNER_SIGN`) — 여기서 왕복 변환한다.
기하: link6→TCP 0.320 m (`nlp/throwing.GP8_DIMS`, robot/gp8.py home_ee 와 일치).
ROS 무관 (numpy/casadi) — 단독 실행 예: `python -m gp8_control.throwing.throw_planner`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from gp8_control.throwing.nlp import planner as _planner
from gp8_control.throwing.nlp import throw_nlp
from gp8_control.throwing.nlp.throwing import G, fk_pos, jacobian, launch_state

PLANNER_SIGN = _planner.PLANNER_SIGN        # q_robot = q_planner · SIGN (속도도 동일)
GRID_DT: float = _planner.DT                 # 궤적 샘플 격자 = 로봇 스트림 2 ms (robot_base.STREAM_DT 와 같아야 한다)
END_MODE: str = os.environ.get("GP8_THR_END_MODE", "exact")   # 움직이는 종점: exact(FK 등식) | linear(관절 1차 근사)


def grid_ceil(t: float) -> float:
    """t 를 GRID_DT 배수로 올림 (부동소수 잡음 1e-6 s 는 내림)."""
    return float(np.ceil(float(t) / GRID_DT - 1e-6) * GRID_DT)


def grid_round(t: float) -> float:
    return float(np.round(float(t) / GRID_DT) * GRID_DT)


# 움직이는 종점 재선형화 문턱 [m]: NLP 가 낸 t_f 와 선형화 기준 t_ref 의 차이 동안 벨트가 흐른 거리 |t_f − t_ref|·v_belt 가
# 이보다 크면 t_ref = t_f 로 종점을 다시 잡고 (직전 해로 warm) 한 번 더 푼다. 1차 근사 오차는 거리의 제곱에 비례해
# 16 cm 외삽에서 J6 20° 였으니(2026-09-21, 벨트 0.3) 1 cm 면 ≈ 0.1° — 이음새 불연속이 사라진다. 런타임·DB 빌더 공통.
RELIN_M: float = float(os.environ.get("GP8_THR_RELIN_M", "0.01"))
DT = _planner.DT                            # 궤적 샘플 주기 (1/240 s)

# 시작점 lift: 잡은 물체 위치에서 +z 로 이 만큼 올라간 상태가 던지기 시작점 (사용자 규약 2026-09-18, env GP8_THR_START_LIFT).
START_LIFT = _planner.START_LIFT
# 벨트 진행 방향 (로봇 프레임 단위벡터) — 로봇 PC 앱 규약: 물체는 −Y 로 흐른다 (track_descend 와 동일).
BELT_DIR = np.array([0.0, -1.0, 0.0])
# 시작 속도: 제자리에서 잡고 +z 로 올라가는 TCP 속도 [m/s] (2026-09-18 사용자 규약). 종점 하강 속도는 픽의
# 벨트 추종 하강 속도(Config.TRACK_Z_SPEED 0.3)와 같은 값을 기본으로 둔다.
LIFT_SPEED: float = float(os.environ.get("GP8_THR_LIFT_SPEED", "1.2"))       # 2026-09-23 사용자: 0.6 → 1.2 m/s
# 상승이 끝나고(던지기 시작) · 하강이 시작되는(아크 종점·대기점) TCP 의 **절대 높이** [m, 로봇 base] — 2026-09-21 사용자:
# 물체 높이 기준 상대값(START_LIFT/hover 5 cm) 대신 고정. 물체 윗면이 이보다 높으면 윗면 + TRAVEL_MIN_CLEAR.
TRAVEL_Z: float = float(os.environ.get("GP8_TRAVEL_Z", "0.12"))   # 2026-09-23 사용자: 0.20 → 0.12
TRAVEL_MIN_CLEAR: float = 0.02


def travel_top(p, z_travel: float = None) -> np.ndarray:
    """물체 흡착점 p → 그 위 상승/하강 높이의 점 (z = TRAVEL_Z, 단 물체가 높으면 윗면 + TRAVEL_MIN_CLEAR)."""
    p = np.asarray(p, float).reshape(3)
    zt = TRAVEL_Z if z_travel is None else float(z_travel)
    return np.array([p[0], p[1], max(zt, p[2] + TRAVEL_MIN_CLEAR)])
DESCEND_SPEED: float = float(os.environ.get("GP8_THR_DESCEND_SPEED", "1.2"))
UP = np.array([0.0, 0.0, 1.0])


_PICK_IK = None


def default_pick_ik():
    """기본 IK = 로봇 모델(robot/gp8.GP8)의 픽 자세 IK (도구 아래 방향, J6 = Config.PICK_WRIST_J6) — 스킬·warm DB 빌더와
    같은 팔 배치. 이걸 써야 DB entry 와 관절공간에서 가까워 warm start 가 먹는다 (position-only IK 는 배치가 달라 11 s).
    로봇 모델을 못 불러오면 None (→ object_boundary 가 position-only IK 로 폴백)."""
    global _PICK_IK
    if _PICK_IK is None:
        try:
            from gp8_control.config import Config
            from gp8_control.robot.gp8 import GP8
            robot, j6 = GP8(), float(Config().PICK_WRIST_J6)
        except Exception:                                   # noqa: BLE001 — ROS 없는 환경 등
            _PICK_IK = False
        else:
            def _ik(p):
                q = _planner.pick_joint(robot, p)
                if q is None:
                    return None
                q = np.asarray(q, float)[:6]; q[-1] = j6
                return q
            _PICK_IK = _ik
    return _PICK_IK or None


# ---------------------------------------------------------------------------
# 프레임 변환 · 경계 상태
# ---------------------------------------------------------------------------
def to_planner(x) -> np.ndarray:
    """로봇 관절 (자세 또는 속도) → planner 프레임."""
    return np.asarray(x, float).reshape(6) * PLANNER_SIGN


def to_robot(x) -> np.ndarray:
    return np.asarray(x, float).reshape(6) * PLANNER_SIGN


def tcp_of(q_robot) -> np.ndarray:
    """로봇 관절 자세 → TCP 위치 (m, 로봇 프레임) — NLP 와 같은 FK."""
    return fk_pos(to_planner(q_robot))


def belt_joint_velocity(q_robot, v_tcp) -> np.ndarray:
    """TCP 가 v_tcp (m/s, 로봇 프레임) 로 병진하고 회전은 0 이 되게 하는 관절 속도 (rad/s, 로봇 규약).
    6×6 기하 자코비안 [Jv; Jw] 의 최소제곱 해 — 특이자세 근처에서도 발산하지 않게 lstsq."""
    v = np.asarray(v_tcp, float).reshape(3)
    if not np.any(v):
        return np.zeros(6)
    Jv, Jw = jacobian(to_planner(q_robot))
    J = np.vstack([Jv, Jw])
    qd_p, *_ = np.linalg.lstsq(J, np.concatenate([v, np.zeros(3)]), rcond=None)
    return to_robot(qd_p)


@dataclass
class ThrowBoundary:
    """관절 경계 상태 (로봇 규약).
    p_tcp / v_tcp / j6 (선택): 종점이 벨트를 따라 움직일 때 정확한 Cartesian 식용 — t 시각의 TCP 위치, TCP 속도(벨트),
    T 관절 값. 있으면 plan_throw 가 관절 1차 근사 대신 FK 등식(툴 수직 + J6 고정)으로 종점을 건다 (2026-09-21).

    t      : 이 상태가 유효한 궤적 시계 시각 [s] (None = 시각 무관/고정). 종점에 t 를 주면 물체를 따라 움직이는
             종점으로 취급돼 q(t_f) = q + q_rate·(t_f − t) 가 된다 (시작점의 t 는 무시).
    q_rate : 자세의 이동률 [rad/s] (물체가 벨트를 따라 흐르는 속도의 관절 표현). None 이면 qd 와 같다고 본다.
             qd(도착 관절속도, 하강 성분 포함)와 구분한다 — 하강 성분으로 종점 자세를 외삽하면 안 되기 때문."""
    q: np.ndarray                      # (6,) rad
    qd: np.ndarray = field(default_factory=lambda: np.zeros(6))   # (6,) rad/s
    t: Optional[float] = None
    q_rate: Optional[np.ndarray] = None
    p_tcp: Optional[np.ndarray] = None     # (3,) m — t 시각의 TCP (정확한 종점식용)
    v_tcp_exact: Optional[np.ndarray] = None   # (3,) m/s — 종점 TCP 속도 (벨트)
    j6: Optional[float] = None             # T 관절 값 [rad, 로봇 규약]

    def __post_init__(self):
        self.q = np.asarray(self.q, float).reshape(6)
        self.qd = np.asarray(self.qd, float).reshape(6)
        if self.q_rate is not None:
            self.q_rate = np.asarray(self.q_rate, float).reshape(6)
        if self.p_tcp is not None:
            self.p_tcp = np.asarray(self.p_tcp, float).reshape(3)
        if self.v_tcp_exact is not None:
            self.v_tcp_exact = np.asarray(self.v_tcp_exact, float).reshape(3)

    @property
    def exact(self) -> bool:
        """정확한 Cartesian 종점식을 쓸 수 있는가 (p_tcp·v_tcp_exact·j6 가 있고 시각 t 가 있음)."""
        return self.t is not None and self.p_tcp is not None and self.v_tcp_exact is not None and self.j6 is not None

    @property
    def rate(self) -> np.ndarray:
        return self.qd if self.q_rate is None else self.q_rate

    def at(self, t: float) -> "ThrowBoundary":
        """선형 외삽: 시각 t 의 상태 (t 가 없으면 그대로)."""
        if self.t is None:
            return self
        return ThrowBoundary(self.q + self.rate * (t - self.t), self.qd.copy(), t, self.rate.copy())

    @property
    def tcp(self) -> np.ndarray:
        return tcp_of(self.q)

    @property
    def v_tcp(self) -> np.ndarray:
        Jv, _ = jacobian(to_planner(self.q))
        return Jv @ to_planner(self.qd)


def object_boundary(p_obj, v_obj=None, *, v_tcp=None, lift: float = 0.0,
                    ik: "Optional[Callable[[np.ndarray], Optional[np.ndarray]]]" = None,
                    t: Optional[float] = None, logger=None) -> ThrowBoundary:
    """물체 위치·속도 → 관절 경계 상태.

    p_obj : 물체(흡착점) 위치 (m, 로봇 프레임).  TCP = p_obj + [0, 0, lift]
    v_obj : 물체 속도 (m/s, 로봇 프레임; 보통 벨트 속도 벡터 = speed·BELT_DIR). None/0 = 정지. → q_rate
    v_tcp : 이 경계에서의 TCP 속도 (m/s). None 이면 v_obj (물체를 그대로 따라감). → qd
            시작점: [0, 0, +LIFT_SPEED] (제자리에서 위로), 종점: v_belt + [0, 0, −DESCEND_SPEED] (물체 위에서 내려오며 추종)
    ik    : TCP 위치 → 로봇 관절 자세. None 이면 default_pick_ik()(로봇 모델 픽 자세 IK). 실패·NLP 한계 밖이면
            position-only IK(`throw_nlp.ik_start_pose`, 한계 안 해 탐색)로 대체.
    t     : p_obj 가 유효한 궤적 시계 시각 (종점을 움직이는 종점으로 쓸 때).
    """
    p = np.asarray(p_obj, float).reshape(3) + np.array([0.0, 0.0, float(lift)])
    if ik is None:
        ik = default_pick_ik()
    q = None
    if ik is not None:
        q_ik = ik(p)
        if q_ik is not None:
            q_ik = np.asarray(q_ik, float).reshape(-1)[:6]
            if _planner.q_start_in_limits(to_planner(q_ik)):
                q = q_ik
            else:
                _planner._log(logger, f"  boundary: IK 자세가 NLP 한계 밖 → 같은 점의 한계 안 자세로 대체 "
                                      f"(TCP {np.round(p, 3).tolist()})")
    if q is None:
        q = to_robot(throw_nlp.ik_start_pose(p))           # ValueError = 이 점은 한계 안 자세가 없음
    q_rate = np.zeros(6) if v_obj is None else belt_joint_velocity(q, v_obj)
    qd = q_rate.copy() if v_tcp is None else belt_joint_velocity(q, v_tcp)
    v_exact = (np.zeros(3) if v_obj is None else np.asarray(v_obj, float).reshape(3)) if v_tcp is None \
        else np.asarray(v_tcp, float).reshape(3)
    return ThrowBoundary(q, qd, t, q_rate, p_tcp=p, v_tcp_exact=v_exact, j6=float(q[5]))


# ---------------------------------------------------------------------------
# 계획 결과
# ---------------------------------------------------------------------------
@dataclass
class ThrowPlan:
    t: np.ndarray            # (N,) s — 0 = start, 스트림 격자(GRID_DT) 위, t[-1] = t_end = t_f 를 2 ms 로 올린 값
    q: np.ndarray            # (N,6) rad, 로봇 규약
    qd: np.ndarray           # (N,6) rad/s
    qdd: np.ndarray          # (N,6) rad/s²
    t_f: float               # 궤적 종료 = end 상태 도달 시각 (NLP 값; 샘플 끝은 info["t_end"] ≥ t_f)
    t_release: float         # suction OFF 명령 시각 (릴리즈 윈도우 중심 t*)
    release_window: tuple    # (t0, t1)
    i_release: int           # t_release 에 해당하는 격자 인덱스
    t_suction_on: Optional[float]   # end(다음 물체) 를 줬을 때 t_f, 아니면 None
    start: ThrowBoundary
    end: Optional[ThrowBoundary]    # 입력 end (None 이면 자유 종점 — q[-1] 이 플래너가 고른 자세)
    p_target: np.ndarray
    p_launch: np.ndarray     # 릴리즈 시점 발사점 (TCP+GRIP_OFF, m)
    v_launch: np.ndarray     # 발사 속도 (m/s)
    p_land: np.ndarray       # 무항력 포물선으로 p_target 높이를 (하강하며) 지나는 점
    land_err: float          # |p_land − p_target| 수평 오차 (m)
    info: dict               # NLP 진단 (J, t_star, tag, n_fail, release_time, …)
    raw: Optional[dict] = field(default=None, repr=False)   # solver 출력 (P, t_f, t_star, lam_g, u_pos …) — warm_init 용

    def suction_events(self) -> list:
        """[(시각 s, on?)] 시간순 — OFF 는 릴리즈, ON 은 다음 물체 도착."""
        ev = [(self.t_release, False)]
        if self.t_suction_on is not None:
            ev.append((self.t_suction_on, True))
        return ev

    @property
    def arc(self):
        """(q (6,N), qd (6,N), t (N,)) — 로봇 PC 디스패치 배열 규약 (관절 × 샘플)."""
        return self.q.T.copy(), self.qd.T.copy(), self.t.copy()

    def summary(self) -> str:
        w0, w1 = self.release_window
        s = (f"t_f={self.t_f:.3f}s  release t*={self.t_release:.3f}s "
             f"[{w0:.3f}, {w1:.3f}] (idx {self.i_release})  "
             f"|v_launch|={np.linalg.norm(self.v_launch):.2f} m/s  "
             f"착지 예측 ({self.p_land[0]:+.3f}, {self.p_land[1]:+.3f}) 오차 {self.land_err * 100:.1f} cm")
        if self.t_suction_on is not None:
            s += f"  suction ON @ {self.t_suction_on:.3f}s (다음 물체 도착)"
        if "end_extrap_s" in self.info:
            s += (f"  [종점 외삽 {self.info['end_extrap_s']:+.3f}s, "
                  f"근사오차 {self.info['end_tcp_err_m'] * 1e3:.2f} mm]")
        return s


def _predict_landing(q_p, qd_p, p_target):
    """planner 프레임 릴리즈 상태 → (p_launch, v_launch, p_land, err). 하강 근 우선, 없으면 상승 근."""
    p, v = launch_state(q_p, qd_p, off=throw_nlp.GRIP_OFF)
    disc = v[2] ** 2 - 2.0 * G * (p_target[2] - p[2])
    if disc < 0:
        return p, v, np.full(3, np.nan), float("inf")
    taus = [(v[2] + s * np.sqrt(disc)) / G for s in (1.0, -1.0)]
    taus = [tau for tau in taus if tau > 1e-9]
    if not taus:
        return p, v, np.full(3, np.nan), float("inf")
    tau = max(taus)                                    # 하강하며 통과하는 근
    p_land = np.array([p[0] + v[0] * tau, p[1] + v[1] * tau, p_target[2]])
    return p, v, p_land, float(np.linalg.norm(p_land[:2] - p_target[:2]))


# ---------------------------------------------------------------------------
# 계획
# ---------------------------------------------------------------------------
def plan_throw(start: ThrowBoundary, p_target, end: "Optional[ThrowBoundary]" = None,
               *, warm_init=None, logger=None) -> ThrowPlan:
    """시작 상태(관절 자세·속도) → [릴리즈] → 종점 상태(다음 물체의 관절 자세·속도) 궤적 + suction 타이밍.

    end=None 이면 종점 자세는 플래너가 고르고(자유) 정지로 끝난다.
    warm_init : 직전 ThrowPlan.raw — 같은 edge 를 t_ref 만 바꿔 다시 풀 때 그 해로 먼저 polish (재선형화, RELIN_M).
    실패(시작/종점 자세가 NLP 한계 밖, multistart 전멸)면 ValueError.
    """
    p_target = np.asarray(p_target, float).reshape(3)
    moving = end is not None and end.t is not None and bool(np.any(end.rate))
    # 정확한 종점식 (2026-09-21 사용자 "근사 말고 실제 식"): 움직이는 종점이고 Cartesian 정보가 있으면 FK 등식으로.
    # GP8_THR_END_MODE=linear 로 옛 관절 1차 근사 유지 (A/B 비교용).
    exact = moving and end.exact and END_MODE != "linear"
    end_cart = None
    if exact:
        end_cart = dict(p=end.p_tcp, v=end.v_tcp_exact, t_ref=float(end.t),
                        j6=float(end.j6) * float(PLANNER_SIGN[5]))          # 위치는 프레임 공통, 관절만 부호
    # 필터 공식화의 이력 길이: 들어올리기 = 물체에서 START_LIFT 만큼 올라온 등속 구간으로 근사 (그 전은 정지)
    v_up = float(start.v_tcp[2]) if np.any(start.qd) else 0.0
    hist_s = (float(START_LIFT) / max(v_up, 0.05)) if v_up > 1e-3 else None
    res = _planner.nlp_plan(
        to_planner(start.q), p_target, v0=to_planner(start.qd), hist_s=hist_s,
        p_f=None if end is None else to_planner(end.q),
        vf=None if end is None else to_planner(end.qd),
        p_f_rate=(to_planner(end.rate) if moving else None),
        t_ref=(float(end.t) if moving else 0.0),
        end_cart=end_cart, warm_init=warm_init,
        logger=logger)
    t_f = float(res["info"]["t_f"])
    # 샘플을 스트림 격자(GRID_DT = 2 ms) 위에 정확히 놓는다 — 스트리머가 보간 없이 그대로 내보내도록 (2026-09-21 사용자).
    # t_f 는 NLP 가 정한 임의 실수라 마지막 샘플은 t_f 를 2 ms 배수로 올린 t_end 에 두고, t_f 이후(≤ 2 ms)는 종점 상태
    # (자세 + 종점 속도 = 벨트 추종)에서 선형 외삽한다 — 오차 ½·q̈·Δ² ≤ 0.02°.
    t_end = grid_ceil(t_f)
    ts = np.arange(int(round(t_end / GRID_DT)) + 1) * GRID_DT
    q_of, qd_of, qdd_of = res["q_of"], res["qd_of"], res["qdd_of"]
    q_f, qd_f = q_of(t_f), qd_of(t_f)
    Q = np.array([q_of(t) if t <= t_f else q_f + qd_f * (t - t_f) for t in ts]) * PLANNER_SIGN   # (N,6) → 로봇 규약
    Qd = np.array([qd_of(t) if t <= t_f else qd_f for t in ts]) * PLANNER_SIGN
    Qdd = np.array([qdd_of(t) if t <= t_f else np.zeros(6) for t in ts]) * PLANNER_SIGN
    w0, w1 = (float(x) for x in res["release_window"])
    t_star = 0.5 * (w0 + w1)
    i_rel = int(np.argmin(np.abs(ts - t_star)))
    p_launch, v_launch, p_land, err = _predict_landing(q_of(t_star), qd_of(t_star), p_target)
    info = dict(res["info"])
    info["t_end"] = float(t_end)
    if moving:
        # 종점이 t_f 와 함께 얼마나 외삽됐는지 + 종점 오차 (도착 자세 TCP vs 물체 위 hover 점, 진단)
        info["end_extrap_s"] = t_f - float(end.t)
        p_goal = (end.p_tcp + end.v_tcp_exact * (t_f - float(end.t))) if end.exact else end.at(t_f).tcp
        info["end_tcp_err_m"] = float(np.linalg.norm(tcp_of(q_f * PLANNER_SIGN) - p_goal))
        info["end_exact"] = bool(exact)
    plan = ThrowPlan(
        t=ts, q=Q, qd=Qd, qdd=Qdd, t_f=t_f,
        t_release=t_star, release_window=(w0, w1), i_release=i_rel,
        t_suction_on=(t_f if end is not None else None),
        start=start, end=end, p_target=p_target,
        p_launch=p_launch, v_launch=v_launch, p_land=p_land, land_err=err,
        info=info, raw=res.get("raw"))
    _planner._log(logger, "  plan_throw: " + plan.summary())
    return plan


def plan_throw_between_objects(p_cur, v_cur, p_target, p_next=None, v_next=None, *,
                               lift: float = START_LIFT, lift_speed: float = LIFT_SPEED,
                               descend_speed: float = DESCEND_SPEED, t_next_offset: float = 0.0,
                               ik=None, t_ref: float = 1.0, logger=None) -> ThrowPlan:
    """현재 물체에서 lift 만큼 올라간 시작점 → 다음 물체 종점 체인 계획 — solve 한 번.

    p_cur, v_cur   : 현재 잡은 물체의 위치·속도 (로봇 프레임). 시작 TCP = p_cur + [0,0,lift],
                     시작 관절속도 = J⁻¹·[0, 0, +lift_speed] (제자리에서 잡고 위로 올라가는 속도; v_cur 는 미사용)
    p_next, v_next : 다음 물체의 **지금** 위치·속도 (같은 프레임). None 이면 종점 자유.
                     종점 관절속도 = J⁻¹·(v_next + [0, 0, −descend_speed]) (물체 위에서 내려오며 벨트 추종),
                     종점 이동률 = J⁻¹·v_next
    t_next_offset  : '지금' 부터 궤적 시작(t=0)까지 걸리는 시간 [s] (계획·lift 시간). 다음 물체는 그 사이에도 흐른다.
    t_ref          : 종점 자세를 선형화할 기준 시각 (궤적 시계) — 대략적인 t_f 예상값이면 충분 (1차 근사라 정확할
                     필요 없음). 종점은 IK(p_next + v_next·(t_next_offset + t_ref)) 에서 qd·(t_f − t_ref) 만큼 외삽된다.
    ik             : TCP 위치 → 로봇 관절 (픽 자세 IK). None 이면 position-only IK.
    """
    start = object_boundary(p_cur, None, v_tcp=UP * lift_speed, lift=lift, ik=ik, logger=logger)
    if p_next is None:
        return plan_throw(start, p_target, None, logger=logger)
    p_next = np.asarray(p_next, float).reshape(3)
    v_next = np.zeros(3) if v_next is None else np.asarray(v_next, float).reshape(3)
    p_ref = p_next + v_next * (t_next_offset + t_ref)          # 기준 시각의 다음 물체 위치
    end = object_boundary(p_ref, v_next, v_tcp=v_next - UP * descend_speed, lift=0.0, ik=ik, t=t_ref,
                          logger=logger)
    return plan_throw(start, p_target, end, logger=logger)


# ---------------------------------------------------------------------------
if __name__ == "__main__":                       # 단독 확인: 정지 벨트 두 물체 체인
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("throw_planner")
    speed = 0.12                                                     # 벨트 0.12 m/s
    v_belt = speed * BELT_DIR
    plan = plan_throw_between_objects(
        p_cur=[0.45, 0.10, 0.05], v_cur=v_belt, p_target=[1.35, 0.075, -0.03],
        p_next=[0.45, 0.35, 0.05], v_next=v_belt, t_next_offset=0.3, logger=log)
    print(plan.summary())
    print("suction events:", plan.suction_events())
    print("q(0)   deg:", np.round(np.rad2deg(plan.q[0]), 1))
    print("q(t_f) deg:", np.round(np.rad2deg(plan.q[-1]), 1), " end TCP:", np.round(tcp_of(plan.q[-1]), 3))
    print("qd(0)  deg/s:", np.round(np.rad2deg(plan.qd[0]), 1), " qd(t_f):", np.round(np.rad2deg(plan.qd[-1]), 1))

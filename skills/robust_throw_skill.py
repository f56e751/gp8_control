"""Throw skill: grasp an object at the intercept, then fling it (NLP throw).

NN throw를 THR 통합 NLP planner (throw_nlp.solve_throw_nlp — CasADi/IPOPT
B-Spline)로 교체한 버전 (2026-07-15: 구 manifold plan.plan_throw 대체).
궤적은 3-세그먼트:
  lift   — 정지(grasp_joint)에서 던지기 시작 자세(q_lift, 정지)까지 time-optimal
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
from throwing import (GP8_QD_MAX, fk_pos, ik_position,  # noqa: E402
                      jacobian, landing_error)

if TYPE_CHECKING:
    from gp8_control.skills.context import PickRequest
    from gp8_control.tracking import TrackedObject

# 플래너(throwing/throw_nlp) ↔ 로봇(robots/gp8 = 하드웨어/스트림) 관절 부호 변환.
# GP8 PoE 축은 J3..J6 = -y,-x,-y,-x 인데 THR 체인은 +y,+x,+y,+x — 관절 3~6 부호
# 반전. 수치 검증(2026-07-20, 300 랜덤 자세): fk_pos(S*q) vs GP8.forward_kinematics(q)
# 오차 = 전 자세 상수 5.0mm(툴 길이 0.320 vs 0.325m 차이 — throwing.py:27 참고).
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
THROW_BIN_X: float = 1.1
THROW_BIN_Y: float = -0.25

# Per-class throw bin TARGET (absolute base-frame XYZ, m). When a target's
# class is in this map, the landing point is OVERRIDDEN with these coordinates.
# Empty default — fill in with measured bin coords (e.g., from terminal_debug).
THROW_BIN_TARGET_MAP: dict[str, tuple] = {}

# 던지기 시작 TCP = grasp 위치 + 이만큼 lift (벨트/주변 clearance 확보).
THROW_LIFT: float = 0.10
# release 윈도우 길이 (s) — throw_nlp 기본값(0.05)과 동일하게 유지해야
# W_ACC 스윕이 검증한 조합과 warm entry(제약 구성 키)가 그대로 성립한다.
# 실기 밸브 지연의 '평균'은 RELEASE_LEAD 선행 명령이 보정하고, 잔여 jitter는
# ±window/2 = ±25ms 안이면 착지 보장 (구버전 0.08은 미검증 조합이라 폐기).
THROW_WINDOW_T: float = float(throw_nlp.RELEASE_TIME)

# 착탄 게이트 (m): 윈도우 dense 착탄오차 최대가 이걸 넘으면 그 해를 거부.
# penalty 공식화라 착탄 정확도는 solver 보장이 아님 — 나쁜 basin은 윈도우
# 일부에서 착탄 불능(inf)이 나올 수 있어 dispatch 전 반드시 확인 (실기 안전).
LANDING_GATE: float = 0.03

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
THROW_WARM_DB: str = str(Path(_THR_DIR) / "warm_db.pkl")

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
                pos_mode=getattr(throw_nlp, "POS_LIMIT_MODE", "colloc"))


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

    Extension point: map object classes to a wait mode in ``PICK_WAIT_MODE``
    so e.g. fragile classes can hover-and-descend while flat ones park at
    grasp height. Only WAIT_AT_GRASP is implemented today; HOVER_DESCEND
    falls back to it with a warning until added.
    """

    WAIT_AT_GRASP = "wait_at_grasp"   # cup parked at grasp height; suction on arrival
    HOVER_DESCEND = "hover_descend"   # park above, descend + suction on arrival (TODO)


# Per-class wait mode (class_name -> PickWaitMode). Classes not listed use
# DEFAULT_PICK_WAIT_MODE.
PICK_WAIT_MODE: dict[str, PickWaitMode] = {}
DEFAULT_PICK_WAIT_MODE = PickWaitMode.WAIT_AT_GRASP


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
        if mode == PickWaitMode.HOVER_DESCEND:
            ctx.log.warn(
                "HOVER_DESCEND wait mode not implemented yet; using WAIT_AT_GRASP"
            )
            mode = PickWaitMode.WAIT_AT_GRASP

        # Start clean (uniform per-object flow: no cross-cycle suction hand-off).
        ctx.traj_ctrl.suction_off()

        # WAIT_AT_GRASP: drive to the grasp pose and prime suction SUCTION_LEAD before
        # the object's arrival. Returns once the object has reached the intercept.
        ctx.set_status("POSITIONING", target.class_name)
        ctx.position_and_prime(
            current_joint, aim_joint, grasp_joint, target, T_grasp[1, 3],
            start_lead=self.arrival_lead(),
        )

        # DIAGNOSTIC: object vs intercept at the instant the lift/throw fires.
        ctx.log_action_timing(target, T_grasp[1, 3], "throw-lift")

        # Lift + throw.
        ctx.set_status("THROWING", target.class_name)

        # 착지 목표 결정. bin map에 있으면 절대좌표 사용, 없으면 legacy fallback
        # (secondary 예측 위치 또는 T_aim hover). theta는 legacy fallback 전용 —
        # manifold planner는 던지는 방향을 p_object→p_target으로 스스로 잡음.
        bin_xyz = THROW_BIN_TARGET_MAP.get(target.class_name)
        if bin_xyz is not None:
            p_target = np.asarray(bin_xyz, dtype=float)
            ctx.log.info(
                f"Throw target for {target.class_name}: fixed bin "
                f"({p_target[0]:+.3f}, {p_target[1]:+.3f}, {p_target[2]:+.3f}) m"
            )
        else:
            theta = float(np.arctan2(
                THROW_BIN_Y - T_grasp[1, 3], THROW_BIN_X - T_grasp[0, 3],
            ))
            T_aim2 = self.plan_throw_landing(T_grasp, theta, T_aim, time.time(), secondary)
            p_target = T_aim2[:3, 3].copy()

        # NLP throw 계획: lift + 통합 궤적. 불가능하면 던지지 않고 정리.
        planned = self.plan_nlp_throw(grasp_joint, T_grasp, p_target)
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
        # 이 브랜치의 next_chain_target은 다음 intercept의 grasp_joint(6,) 또는
        # None을 반환한다 (throw_skill/push_skill과 동일 계약 — (grasp, cand)
        # tuple을 주던 push 브랜치 인터페이스가 아님; chain_park_joint도 이
        # 브랜치엔 없다).
        next_grasp = ctx.next_chain_target(grasp_joint, throw_T)
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
        grasp_joint: np.ndarray,
        T_grasp: np.ndarray,
        p_target: np.ndarray,
    ):
        """THR 통합 NLP planner 호출 + lift 세그먼트 생성.

        리턴: (res dict, (lift_traj (6,n), lift_vel (6,n), lift_ts (n,)))
        또는 None (실현 불가 — 사유는 로그).
        던지기 시작 TCP는 grasp 위치 + THROW_LIFT (벨트 clearance).
        q_lift IK seed = grasp_joint이라 현재 자세 근처 branch로 잡히고,
        NLP는 q_start=q_lift에서 정지 출발 (v0=0) — lift가 정지로 끝나므로 연속.
        """
        ctx = self.ctx
        p_lift = T_grasp[:3, 3] + np.array([0.0, 0.0, THROW_LIFT])
        p_target = np.asarray(p_target, dtype=float)

        # lift 자세: NLP 위치 한계(B≥10°, U≤45°, |R|≤80° 포함) 안의 IK 해.
        # ik_position/Q_LO/Q_HI는 플래너 규약이므로 로봇 규약인 grasp_joint를
        # _PLANNER_SIGN으로 변환해 시드로 쓴다 (고정 시드들은 원래 플래너 규약).
        q_lift = None
        seeds = [np.asarray(grasp_joint, dtype=float) * _PLANNER_SIGN]
        seeds += [np.array([seeds[0][0], 0.5, -0.2, 0.0, b, 0.0]) for b in (0.6, 0.9, 1.2)]
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
        if db:
            near = sorted(db, key=lambda e: float(np.linalg.norm(
                np.asarray(e["target"])[:2] - p_target[:2])))[:2]
            warm_cands += [(f"db#{i}", e) for i, e in enumerate(near)]

        res, how = None, ""
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

        # lift: 정지(grasp_joint) → 정지(q_lift). trajectory()가 M1/M2 안에서
        # time-optimal 세그먼트를 만들어 줌. 스트리밍되는 세그먼트는 로봇 규약
        # 이어야 하므로 q_lift(플래너 규약)를 변환한 끝점을 쓴다 — NLP의
        # q_start=q_lift(플래너 규약)와는 별개다.
        zero6 = np.zeros(6)
        l_traj, l_vel, l_ts = trajectory(
            np.asarray(grasp_joint, dtype=float), zero6,
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
        #    아니고, 나쁜 basin은 윈도우 일부에서 inf가 나옴 (실기 필수 게이트)
        rt = res["release_time"]
        errs = []
        for t in res["t_star"] + np.linspace(-rt / 2, rt / 2, 11):
            q, qd = q_of(t), qd_of(t)
            errs.append(landing_error(fk_pos(q), jacobian(q)[0] @ qd, p_target))
        e_max = float(np.max(errs))
        if not np.isfinite(e_max) or e_max > LANDING_GATE:
            return f"착탄 게이트 초과 (window max {e_max * 1e3:.0f}mm > {LANDING_GATE * 1e3:.0f}mm)"
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

"""Throw skill: grasp an object at the intercept, then fling it (manifold throw).

NN throw를 THR ballistic-manifold planner (plan.plan_throw)로 교체한 버전.
궤적은 3-세그먼트:
  windup  — 정지 상태(grasp_joint)에서 윈도우 시작 상태(q0, qd0)까지 time-optimal
  window  — quintic joint 궤적. 윈도우 내내 TCP (p,v)가 ballistic manifold 위
            → 윈도우 안 '아무 때나' release해도 target 착지 (release jitter 면역)
  chain   — 윈도우 끝 상태(q(T), q̇(T))에서 park까지 time-optimal 감속

NN 버전 대비 구조 차이:
  * compute_throw_params / new_trajectory / 속도 clamp 루프 제거 — 윈도우는
    planner가 joint limit을 전 구간 exact하게 보장하고 (다항식 근 기반),
    windup/chain은 trajectory()가 M1/M2 안에서 time-optimal로 생성.
  * release는 한 점(eta)이 아니라 윈도우 중앙 — 양쪽 ±T/2 타이밍 여유.
  * J6도 planner가 구동 (∫||ω_tcp||² 최소화 = 던지는 동안 물체 orientation이
    아예 안 돎 — J6를 잡아두는 것보다 suction에 유리).
"""

from __future__ import annotations

import datetime
import sys
import time
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from gp8_control.skills.base import ManipulationSkill, SkillResult
from gp8_control.trajectory.trajectory_primitive import trajectory

# THR manifold planner (이 파일 기준 상위 폴더 = /PublicSSD/ryugaeun/THR)
_THR_DIR = str(Path(__file__).resolve().parents[1])
if _THR_DIR not in sys.path:
    sys.path.insert(0, _THR_DIR)
from plan import plan_throw  # noqa: E402
from throwing import fk_pos  # noqa: E402

if TYPE_CHECKING:
    from gp8_control.skills.context import PickRequest
    from gp8_control.tracking import TrackedObject


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

# 윈도우 시작 TCP = grasp 위치 + 이만큼 lift (벨트/주변 clearance 확보).
THROW_LIFT: float = 0.10
# manifold 윈도우 길이 (s). 이 안 아무 때나 release해도 착지.
THROW_WINDOW_T: float = 0.08


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


class ThrowSkill(ManipulationSkill):
    """Pick (suction) at the intercept and throw via the manifold trajectory.

    ``execute`` runs the full ambush cycle (position → wait → suction → throw →
    chain to next intercept). The individual stages —
    :meth:`plan_throw_landing`, :meth:`plan_manifold_throw`,
    :meth:`build_throw_trajectory` — are public so external code can reuse
    the throw planning.
    """

    name = "throw"

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._last_throw_meta: dict = {}

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

        # manifold throw 계획: windup + 윈도우. 불가능하면 던지지 않고 정리.
        planned = self.plan_manifold_throw(grasp_joint, T_grasp, p_target)
        if planned is None:
            ctx.log.warn("Throw infeasible; dropping in place and parking")
            ctx.traj_ctrl.suction_off()
            ctx.set_active_target(None)
            ctx.set_status("IDLE", "")
            return SkillResult(False, "throw infeasible for this target")
        plan, windup = planned

        # Chain the follow-through toward the NEXT object's grasp (best-effort) so the arm
        # OVERLAPS the next approach with this throw instead of parking far and re-driving
        # serially. Symmetric + stateless: the next epoch still SELECTS + DRIVES fresh from
        # this closer pose (no commit/preposition). None -> lifted-standby park.
        throw_T = float(windup[2][-1] + plan.release_time)  # windup + 윈도우
        nxt = ctx.next_chain_target(grasp_joint, throw_T)
        next_grasp, next_cand = nxt if nxt is not None else (None, None)
        # Chain PARK: ask the NEXT object's skill where to park (push → its
        # backswing pose; default None → lifted standby over the grasp).
        chain_park = None
        if next_cand is not None:
            chain_park = ctx.skill_obj_for(next_cand).chain_park_joint(
                next_grasp, next_cand
            )
        self.build_throw_trajectory(
            grasp_joint, plan, windup,
            next_grasp=next_grasp, chain_park=chain_park,
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

    def plan_manifold_throw(
        self,
        grasp_joint: np.ndarray,
        T_grasp: np.ndarray,
        p_target: np.ndarray,
    ):
        """THR manifold planner 호출 + windup 세그먼트 생성.

        리턴: (ThrowPlan, (windup_traj (6,n), windup_vel (6,n), windup_ts (n,)))
        또는 None (실현 불가 — 사유는 로그).
        윈도우 시작 TCP는 grasp 위치 + THROW_LIFT (벨트 clearance),
        IK seed = grasp_joint이라 q0가 현재 자세 근처 branch로 잡힘.
        """
        ctx = self.ctx
        p_object = T_grasp[:3, 3] + np.array([0.0, 0.0, THROW_LIFT])
        try:
            plan = plan_throw(
                p_object, p_target,
                release_time=THROW_WINDOW_T,
                q_seed=np.asarray(grasp_joint, dtype=float),
            )
        except ValueError as e:
            ctx.log.warn(f"manifold throw planning failed: {str(e).splitlines()[0]}")
            return None

        # windup: 정지(grasp_joint) → 윈도우 시작 상태(q0, qd0). trajectory()가
        # M1/M2 (속도/가속 한계) 안에서 time-optimal 세그먼트를 만들어 줌.
        zero6 = np.zeros(6)
        w_traj, w_vel, w_ts = trajectory(
            np.asarray(grasp_joint, dtype=float), zero6,
            plan.q0, plan.qd0,
            ctx.M1[:6], ctx.M2[:6], hertz=ctx.cfg.TRAJ_HZ,
        )
        ctx.log.info(
            f"Manifold throw: solver={plan.solver} tau0={plan.tau0:.2f}s "
            f"windup={w_ts[-1]:.3f}s window={plan.release_time:.3f}s "
            f"landing_err_max={1e3 * plan.landing_err_max:.2f}mm "
            f"(release anytime in window)"
        )
        return plan, (w_traj, w_vel, w_ts)

    # ------------------------------------------------------------------
    # Throw trajectory build + dispatch
    # ------------------------------------------------------------------
    def build_throw_trajectory(
        self,
        grasp_joint: np.ndarray,
        plan,
        windup,
        next_grasp: "Optional[np.ndarray]" = None,
        chain_park: "Optional[np.ndarray]" = None,
    ) -> None:
        """Build and dispatch: windup + manifold window + chain-to-park.

        window 구간은 planner가 이미 보장함 —
          * TCP가 ballistic manifold 위 (윈도우 안 아무 때나 release해도 착지)
          * joint 위치/속도 한계 전 구간 exact 만족 (다항식 근 기반 검증)
        이라서 NN 버전의 사후 velocity-clamp 루프가 필요 없음. windup/chain은
        trajectory()가 M1/M2로 bound함.

        release는 윈도우 '중앙' 샘플에 걸어서 suction 밸브 지연/jitter가
        ±window/2 안이면 착지가 보장되게 함 (RELEASE_LEAD로 명령 시점 선행).
        """
        ctx = self.ctx
        w_traj, w_vel, w_ts = windup
        dt = 1.0 / ctx.cfg.TRAJ_HZ

        # manifold 윈도우 샘플 (첫 샘플은 windup 끝과 동일 상태라 drop).
        t_win, Q, Qd, _ = plan.sample(dt=dt)
        arc_traj = Q.T[:, 1:]                      # (6, n_win)
        arc_vel = Qd.T[:, 1:]
        arc_ts = t_win[1:] + w_ts[-1]
        win_lo = w_traj.shape[1]                   # 윈도우 첫 샘플 index (concat 기준)
        win_hi = win_lo + arc_traj.shape[1] - 1    # 윈도우 마지막 샘플 index

        # release: 물리적 release가 윈도우 중앙에 오도록 RELEASE_LEAD만큼 앞당겨
        # 명령. 어느 쪽으로 밀려도 윈도우 안(±window/2)이면 착지 보장.
        lead_steps = int(round(ctx.cfg.RELEASE_LEAD * ctx.cfg.TRAJ_HZ))
        release_idx = (win_lo + win_hi) // 2 - lead_steps
        release_idx = max(win_lo, min(release_idx, win_hi))

        # Chain: 윈도우 끝 상태(속도 있음)에서 park까지 감속. 목적지 우선순위는
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
            arc_traj[:, -1], arc_vel[:, -1],
            chain_target[:6], zero6,
            ctx.M1[:6], ctx.M2[:6], hertz=ctx.cfg.TRAJ_HZ,
        )
        # Drop the chain's first column — it duplicates the window's last
        # sample. Slice traj/vel/ts the SAME way so the three stay equal
        # length (a lone-ts slice would misalign _build_queue_waypoints).
        traj_chain = traj_chain[:, 1:]
        vel_chain = vel_chain[:, 1:]
        ts_chain_shifted = ts_chain[1:] + arc_ts[-1]

        traj_throw = np.concatenate((w_traj, arc_traj, traj_chain), axis=1)  # (6, total)
        vel_throw = np.concatenate((w_vel, arc_vel, vel_chain), axis=1)
        timestep_throw = np.concatenate((w_ts, arc_ts, ts_chain_shifted))
        assert traj_throw.shape[1] == vel_throw.shape[1] == timestep_throw.shape[0], (
            f"throw traj/vel/timestep length mismatch: "
            f"{traj_throw.shape[1]}/{vel_throw.shape[1]}/{timestep_throw.shape[0]}"
        )
        final_joint = chain_target

        ctx.log.info(
            f"Manifold throw dispatch: windup {w_traj.shape[1]} + window "
            f"{arc_traj.shape[1]} + chain {traj_chain.shape[1]} samples, "
            f"release step {release_idx} (window [{win_lo},{win_hi}], "
            f"timing margin ±{0.5 * plan.release_time:.3f}s, "
            f"lead {ctx.cfg.RELEASE_LEAD:.2f}s, chain->{chain_dest})"
        )

        ctx.traj_ctrl.send_trajectory_queue_with_timed_release(
            traj_throw, vel_throw, timestep_throw,
            final_joint=final_joint,
            release_index=release_idx,
        )
        self._last_throw_meta = {
            "T": plan.release_time,
            "windup_T": float(w_ts[-1]),
            "tau0": plan.tau0,
            "solver": plan.solver,
            "landing_err_mm": 1e3 * plan.landing_err_max,
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
            "windup_T_s": round(meta.get("windup_T", 0.0), 3),
            "tau0_s": round(meta.get("tau0", 0.0), 3),
            "solver": meta.get("solver", ""),
            "landing_err_mm": round(meta.get("landing_err_mm", 0.0), 3),
            "release_idx": meta.get("release_idx", ""),
            "n_steps": meta.get("n_steps", ""),
            "io_ms": round(lt["io_ms"], 1) if lt.get("io_ms") is not None else "",
        }
        self._append_csv_row(path, row, ctx.log)

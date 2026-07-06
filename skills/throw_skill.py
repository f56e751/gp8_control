"""Throw skill: grasp an object at the intercept, then fling it (NN throw)."""

from __future__ import annotations

import datetime
import time
from enum import Enum
from typing import TYPE_CHECKING, Optional

import numpy as np

from gp8_control.skills.base import ManipulationSkill, SkillResult
from gp8_control.trajectory.trajectory_primitive import (
    new_trajectory,
    trajectory,
    pad,
)

if TYPE_CHECKING:
    from gp8_control.skills.context import PickRequest
    from gp8_control.tracking import TrackedObject


# =========================================================================
# Throw policy (was app-level policy in app.py)
# =========================================================================

# Throw target bin (base-frame XY, m). The throw HEADING (theta) is computed PER
# OBJECT as the bearing from that object's grasp to this bin, so the swing re-aims
# from any grab position. Y is downstream-negative (belt -Y). Tune to the measured
# bin centre.
THROW_BIN_X: float = 1.1
THROW_BIN_Y: float = -0.25


# Per-class throw bin TARGET (absolute base-frame XYZ, m). When a target's
# class is in this map, T_aim2 is OVERRIDDEN with these coordinates so the
# NN throw aims at a fixed bin location instead of secondary/T_aim hover.
# Empty default — fill in with measured bin coords (e.g., from terminal_debug).
THROW_BIN_TARGET_MAP: dict[str, tuple] = {}


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
    """Pick (suction) at the intercept and throw the object via the NN trajectory.

    ``execute`` runs the full ambush cycle (position → wait → suction → throw →
    chain to next intercept). The individual stages —
    :meth:`plan_throw_landing`, :meth:`build_throw_trajectory` — are public so
    external code can reuse the throw planning.
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
        # delta < 0 -> the object already passed the grasp point and we suction
        # empty belt / lift behind it (the multi-object symptom). See
        # SkillContext.log_action_timing.
        ctx.log_action_timing(target, T_grasp[1, 3], "throw-lift")

        # Lift + throw.
        ctx.set_status("THROWING", target.class_name)

        # Throw heading = bearing from THIS object's grasp to the fixed bin,
        # recomputed per object so the swing re-aims from any grab position (a
        # fixed per-class angle only matched a y=0 grasp). theta then tilts the
        # throw swing toward the bin. See THROW_BIN_X/Y.
        theta = float(np.arctan2(
            THROW_BIN_Y - T_grasp[1, 3], THROW_BIN_X - T_grasp[0, 3],
        ))

        # Throw target. If the class has a fixed bin coord in THROW_BIN_TARGET_MAP,
        # override T_aim2 with that absolute base-frame XYZ so the NN aims at the
        # bin. Otherwise fall back to the legacy plan_throw_landing (secondary's
        # predicted position, or T_aim hover when no secondary).
        bin_xyz = THROW_BIN_TARGET_MAP.get(target.class_name)
        if bin_xyz is not None:
            T_aim2 = np.eye(4)
            T_aim2[:3, :3] = T_aim[:3, :3]
            T_aim2[:3, 3] = np.asarray(bin_xyz, dtype=float)
            ctx.log.info(
                f"Throw target for {target.class_name}: fixed bin "
                f"({bin_xyz[0]:+.3f}, {bin_xyz[1]:+.3f}, {bin_xyz[2]:+.3f}) m"
            )
        else:
            T_aim2 = self.plan_throw_landing(T_grasp, theta, T_aim, time.time(), secondary)

        aim_joint2 = ctx.robot.inverse_kinematics(T_aim2)
        if aim_joint2 is None:
            ctx.log.warn("Throw IK failed after grab; lifting in place")
            aim_joint2, T_aim2 = aim_joint, T_aim
        # Hold the pick wrist through the throw — the NN drives joints 1-5
        # only, so J6 just parks wherever the grasp left it (PICK_WRIST_J6
        # baseline) instead of snapping to 0 and back around every throw.
        aim_joint2 = np.asarray(aim_joint2, dtype=float)
        aim_joint2[-1] = float(grasp_joint[-1])

        params = ctx.planner.compute_throw_params(T_grasp, T_aim2, theta)
        # Chain the follow-through toward the NEXT object's grasp (best-effort) so the arm
        # OVERLAPS the next approach with this throw instead of parking far and re-driving
        # serially. Symmetric + stateless: the next epoch still SELECTS + DRIVES fresh from
        # this closer pose (no commit/preposition). None -> lifted-standby park.
        nxt = ctx.next_chain_target(grasp_joint, float(params.T))
        next_grasp, next_cand = nxt if nxt is not None else (None, None)
        # Chain PARK: ask the NEXT object's skill where to park (push → its
        # backswing pose; default None → lifted standby over the grasp).
        chain_park = None
        if next_cand is not None:
            chain_park = ctx.skill_obj_for(next_cand).chain_park_joint(
                next_grasp, next_cand
            )
        self.build_throw_trajectory(
            grasp_joint, aim_joint2, params,
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

    # ------------------------------------------------------------------
    # Throw trajectory build + dispatch
    # ------------------------------------------------------------------
    def build_throw_trajectory(
        self,
        grasp_joint: np.ndarray,
        aim_joint2: np.ndarray,
        params,
        next_grasp: "Optional[np.ndarray]" = None,
        chain_park: "Optional[np.ndarray]" = None,
    ) -> None:
        """Build and dispatch throw trajectory using already-decoded ThrowParams.

        The NN throw arc (grasp → release → aim_joint2) is generated and
        velocity-clamped exactly as the trained swing, so the motion UP TO the
        release sample is identical to the original throw — same path, same
        velocity, same release timing. After the release the full arc is
        kept up to aim_joint2 (8 cm hover, dq(T)=0) and chained from rest to a
        safe park — a lifted-standby, or the shared idle pose when the queue is
        empty — so the arm parks high; the next pick is then selected and driven
        fresh next epoch (no cross-object pre-position).

        The timed suction release still fires at ``release_idx``
        (= eta_idx - lead_steps); in the cut case that is the last arc
        waypoint, so release timing/position/velocity are unchanged.
        """
        ctx = self.ctx
        throw_T = float(params.T)
        n_steps = max(2, int(throw_T * ctx.cfg.TRAJ_HZ))
        s = np.linspace(0.0, 1.0, n_steps + 1)

        traj_ext, vel_ext, _, _, ts_ext = new_trajectory(
            s, grasp_joint[:5], aim_joint2[:5], params.w, throw_T,
        )
        # traj_ext, vel_ext: (n_steps+1, 5); ts_ext: (n_steps+1,)

        # Clamp the throw to the robot's joint velocity limits. The NN throw is
        # time-parameterised (params.T) with NO joint-speed bound, so a large
        # joint sweep in a short T can command a queued segment faster than the
        # controller allows -> Yaskawa alarm 4414 "excessive segment velocity"
        # (seen on S/L). new_trajectory's path depends only on s; velocity ∝
        # 1/T, so stretching T by the over-limit ratio brings every segment
        # under the limit in one rescale (2nd pass guards the n_steps re-sample).
        m1_5 = np.asarray(ctx.M1[:5], dtype=float)
        for _ in range(2):
            dt_seg = np.maximum(np.diff(ts_ext), 1e-9)[:, None]
            seg_vel = np.abs(np.diff(traj_ext, axis=0)) / dt_seg   # (n,5) rad/s
            ratio = float(np.max(seg_vel / m1_5[None, :]))
            if ratio <= 1.0:
                break
            throw_T = throw_T * ratio * 1.05    # +5% margin
            n_steps = max(2, int(throw_T * ctx.cfg.TRAJ_HZ))
            s = np.linspace(0.0, 1.0, n_steps + 1)
            traj_ext, vel_ext, _, _, ts_ext = new_trajectory(
                s, grasp_joint[:5], aim_joint2[:5], params.w, throw_T,
            )
            ctx.log.warn(
                f"Throw clamped: seg vel ratio {ratio:.2f} > 1 "
                f"-> T {params.T:.3f}->{throw_T:.3f}s (n_steps {n_steps})"
            )

        eta_idx = int(round(params.eta * n_steps))
        eta_idx = max(0, min(eta_idx, n_steps))
        lead_steps = int(round(ctx.cfg.RELEASE_LEAD * ctx.cfg.TRAJ_HZ))
        release_idx = max(0, min(eta_idx - lead_steps, n_steps))

        # The arc + velocity clamp above are UNCHANGED, so the swing up to the
        # release sample is byte-identical to the original throw. Only the
        # post-release tail differs (see docstring):
        #   * next pick known -> CUT at the release sample and fly straight to
        #     it from the actual release state (large velocity); drop the
        #     wasted release→aim_joint2 follow-through.
        #   * no next pick    -> keep the full arc to aim_joint2 (8 cm hover,
        #     at rest) and chain to the idle/standby pose (idle_target()).
        # Either branch then appends one fresh time-optimal trajectory() from
        # (start_q5, start_dq5) to chain_target — only the start state / target
        # differ, so the concat/dispatch code below stays shared.
        # Uniform flow: always keep the FULL arc to aim_joint2 (8 cm hover, at rest) and
        # chain the follow-through to a safe park — home/idle when the queue is EMPTY, else
        # a lifted-standby near the last pose so the next pick approaches fresh from near the
        # belt (no wasted home round-trip). No cut-at-release / pre-position of a next object
        # (that throw-only chaining is what tangled push<->throw alternation). copy() so the
        # shared ctx.idle_joint is never mutated.
        traj_pre_5 = traj_ext.T                    # (5, n_steps+1) full arc
        vel_pre_5 = vel_ext.T
        ts_pre = ts_ext
        start_q5 = traj_ext[-1]                     # aim_joint2 at rest
        start_dq5 = vel_ext[-1]                     # ~0 (NN boundary condition)
        if chain_park is not None:
            # The NEXT object's skill supplied its own action-start park (push:
            # its backswing pose + clearance). Full 6-DOF pose, wrist included.
            chain_target = np.asarray(chain_park, dtype=float).copy()
            chain_dest = "next action-start park"
        elif next_grasp is not None:
            # Park OVER the next grasp (its XY raised to home Z), NOT at belt height. A
            # belt-height chain endpoint would leave the arm low, and the next epoch's
            # drive to a different-lane grasp would sweep the TCP low across the belt
            # (floor-dip / grazing). lifted_standby_joint keeps the XY, raises Z, and
            # parks the wrist at the shared PICK_WRIST_J6 baseline.
            chain_target = ctx.lifted_standby_joint(next_grasp)
            chain_dest = "over next grasp"
        elif not ctx.queue:
            chain_target = self.idle_target().copy()
            chain_dest = "home/idle (queue empty)"
        else:
            chain_target = ctx.lifted_standby_joint(aim_joint2)
            chain_dest = "lifted standby"

        # 6-DOF chain: the ARC is 5-DOF (the NN drives joints 1-5; J6 stays
        # parked at the pick wrist), but the chain must be able to ROTATE the
        # wrist toward the park target (e.g. the next push's backswing J6) —
        # so it runs on all six joints, starting from the arc end with J6 at
        # the held wrist and zero wrist velocity.
        held_wrist = float(grasp_joint[-1])
        start_q6 = np.append(start_q5, held_wrist)
        start_dq6 = np.append(start_dq5, 0.0)
        zero6 = np.zeros(6)
        traj_chain, vel_chain, ts_chain = trajectory(
            start_q6, start_dq6,
            chain_target[:6], zero6,
            ctx.M1[:6], ctx.M2[:6], hertz=ctx.cfg.TRAJ_HZ,
        )
        # Drop the chain's first column — it's start_q6 (the last sample of the
        # kept arc: aim_joint2 + held wrist), so it would be a duplicate. Slice
        # traj/vel/ts the SAME way so the three stay equal length: if the chain
        # degenerates to a single sample (chain_target ≈ start_q6 -> opt_time ≈
        # 0) all three become empty and only the arc remains. (Slicing ts alone
        # would leave traj/vel one column longer, and zip() in
        # _build_queue_waypoints would then silently drop a waypoint and bind
        # final_joint to the wrong timestamp.)
        traj_chain = traj_chain[:, 1:]
        vel_chain = vel_chain[:, 1:]
        ts_chain_shifted = ts_chain[1:] + ts_pre[-1]

        # Pad the ARC to 6-DOF (J6 held at the pick wrist, zero velocity —
        # zero-POSITION padding would snap the wrist to 0 at the throw start),
        # then append the 6-DOF chain.
        traj_pre_6 = pad(traj_pre_5.T, fill=held_wrist).T
        vel_pre_6 = pad(vel_pre_5.T).T
        traj_throw = np.concatenate((traj_pre_6, traj_chain), axis=1)   # (6, total)
        vel_throw = np.concatenate((vel_pre_6, vel_chain), axis=1)
        timestep_throw = np.concatenate((ts_pre, ts_chain_shifted))
        assert traj_throw.shape[1] == vel_throw.shape[1] == timestep_throw.shape[0], (
            f"throw traj/vel/timestep length mismatch: "
            f"{traj_throw.shape[1]}/{vel_throw.shape[1]}/{timestep_throw.shape[0]}"
        )
        final_joint = chain_target

        ctx.log.info(
            f"Throw T={params.T:.3f}s eta={params.eta:.3f} -> release step "
            f"{release_idx}/{traj_throw.shape[1] - 1} (eta step {eta_idx}, "
            f"lead {ctx.cfg.RELEASE_LEAD:.2f}s, full arc->{chain_dest})"
        )

        ctx.traj_ctrl.send_trajectory_queue_with_timed_release(
            traj_throw, vel_throw, timestep_throw,
            final_joint=final_joint,
            release_index=release_idx,
        )
        self._last_throw_meta = {
            "T": params.T, "eta": params.eta,
            "release_idx": release_idx, "n_steps": n_steps,
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
            "throw_T_s": round(meta.get("T", 0.0), 3),
            "eta": round(meta.get("eta", 0.0), 3),
            "release_idx": meta.get("release_idx", ""),
            "n_steps": meta.get("n_steps", ""),
            "io_ms": round(lt["io_ms"], 1) if lt.get("io_ms") is not None else "",
        }
        self._append_csv_row(path, row, ctx.log)

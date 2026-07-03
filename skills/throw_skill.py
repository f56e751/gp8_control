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

# Class-specific throw-plane angle (radians, rotation about +Z). Keys are
# SAM class names; classes not listed here throw at theta=0. This connects
# perception output to throw geometry — it lives with the throw skill so the
# throw owner edits it without touching app.py.
THETA_MAP = {
    "transparent": -np.pi / 12.0,
    "metal":       -np.pi * 25.0 / 180.0,
}

# Throw target bin (base-frame XY, m). The throw HEADING (theta) is computed PER
# OBJECT as the bearing from that object's grasp to this bin, so the swing re-aims
# from any grab position — replacing the old fixed per-class THETA_MAP angle, which
# only matched when grabbing on the y=0 line. Y is downstream-negative (belt -Y).
# Tune to the measured bin centre. (THETA_MAP is still used by the legacy "moving"
# strategy in app.py.)
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
    :meth:`plan_throw_landing`, :meth:`solve_keyframe_joints`,
    :meth:`build_throw_trajectory` — are also public so the legacy "moving"
    pick strategy in ``app.py`` can reuse the same throw code.
    """

    name = "throw"

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._last_throw_meta: dict = {}

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

        # Start clean — but NOT if the prior throw's return already primed this
        # pick's suction (vacuum intentionally ON); clearing it would re-open the
        # back-to-back blind gap we just closed.
        if not ctx.suction_primed_for_pick:
            ctx.traj_ctrl.suction_off()

        # Is this object the previous throw's committed return target? If so the
        # return swing already streamed the arm to its grasp pose, so we must NOT
        # re-drive (same spot) — just wait + grab. Only drive when the arm actually
        # needs to move there (first pick, a different object). Decided by object
        # identity upstream (PickRequest.prepositioned), not a distance guess.
        # (adv4ncr 250Hz stream: there is no point-queue mode to (re)enter, so the
        # old ~0.4s re-entry and the persistent-queue workaround it needed are gone.)
        prepositioned = request.prepositioned

        # WAIT_AT_GRASP: drive to the grasp pose (unless already there) and prime
        # suction SUCTION_LEAD before the object's arrival. Returns once the
        # object has reached the intercept.
        ctx.set_status("POSITIONING", target.class_name)
        ctx.position_and_prime(
            current_joint, aim_joint, grasp_joint, target, T_grasp[1, 3],
            skip_move=prepositioned,
            start_lead=self.arrival_lead(),
        )

        # Lift + throw.
        ctx.set_status("THROWING", target.class_name)

        # Throw heading = bearing from THIS object's grasp to the fixed bin,
        # recomputed per object so the swing re-aims from any grab position (the
        # old fixed per-class THETA_MAP angle only matched a y=0 grasp). theta then
        # tilts the throw swing toward the bin. See THROW_BIN_X/Y.
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
        aim_joint2 = np.asarray(aim_joint2, dtype=float); aim_joint2[-1] = 0.0

        # Decode the throw first so the chain selection below knows how long the
        # swing takes (the arm can't start chaining to the next object until the
        # throw finishes).
        params = ctx.planner.compute_throw_params(T_grasp, T_aim2, theta)

        # Chain target for the throw's post-release motion: the next object the
        # pick will ACTUALLY complete — same feasibility gate as the main pick,
        # plus the throw_time the arm must finish first. So the arm only flies to
        # an intercept it will then pick (no "went there but never picked"), and
        # ``next_suction_at`` primes THAT object's vacuum during the return chain.
        next_intercept_joint, next_suction_at = ctx.scan_next_intercept(
            grasp_joint, float(params.T),
        )
        primed_next = self.build_throw_trajectory(
            grasp_joint, aim_joint2, params,
            next_intercept_joint=next_intercept_joint,
            next_suction_at=next_suction_at,
        )
        # If the return chain primed the next pick's suction (vacuum ON), hand
        # that off to the next cycle instead of clearing it. Otherwise it's a
        # safety release in case the throw push failed with suction still on.
        if primed_next:
            ctx.suction_primed_for_pick = True
        else:
            ctx.traj_ctrl.suction_off()
        self._log_throw_cycle(target)
        ctx.set_active_target(None)
        ctx.set_status("IDLE", "")
        return SkillResult(True, "throw complete")

    # ------------------------------------------------------------------
    # Throw planning (shared with the legacy "moving" pick strategy)
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

    def solve_keyframe_joints(
        self,
        T_aim1: np.ndarray,
        T_grasp1: np.ndarray,
        T_aim2: np.ndarray,
    ):
        """IK for the aim/grasp/aim2 keyframes; wrist (joint 6) zeroed. None on IK fail."""
        return self._ik_keyframes((T_aim1, T_grasp1, T_aim2), wrist=0.0)

    # ------------------------------------------------------------------
    # Throw trajectory build + dispatch
    # ------------------------------------------------------------------
    def build_throw_trajectory(
        self,
        grasp_joint: np.ndarray,
        aim_joint2: np.ndarray,
        params,
        next_intercept_joint: "Optional[np.ndarray]" = None,
        next_suction_at: "Optional[float]" = None,
    ) -> bool:
        """Build and dispatch throw trajectory using already-decoded ThrowParams.

        The NN throw arc (grasp → release → aim_joint2) is generated and
        velocity-clamped exactly as the trained swing, so the motion UP TO the
        release sample is identical to the original throw — same path, same
        velocity, same release timing. Only the post-release tail differs,
        depending on whether a next pick is known:

          - ``next_intercept_joint`` supplied -> CUT the arc at the release
            sample and fly straight to that intercept via a time-optimal move
            that STARTS FROM THE ACTUAL RELEASE STATE (position + the large
            throw velocity). The follow-through release→aim_joint2 is dropped
            (wasted motion once the object is gone), so the arm heads to the
            next pick immediately instead of parking at the 8 cm hover first.
          - no next pick -> keep the full arc up to aim_joint2 (8 cm hover,
            dq(T)=0) and chain to the shared idle/standby pose
            (``idle_target()``) from rest, so the arm parks high instead of at
            this object's grasp.

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
        if next_intercept_joint is not None:
            cut = release_idx
            traj_pre_5 = traj_ext[:cut + 1].T         # arc up to (incl.) release
            vel_pre_5 = vel_ext[:cut + 1].T
            ts_pre = ts_ext[:cut + 1]
            start_q5 = traj_ext[cut]                   # release-sample pose
            # Release-sample velocity is large (mid-swing). Clip to the joint
            # speed limit so opt_time/_trajectory_1d stay in their feasible
            # region; affects only the appended move's start, never the throw.
            start_dq5 = np.clip(vel_ext[cut], -ctx.M1[:5], ctx.M1[:5])
            chain_target = np.asarray(next_intercept_joint, dtype=float)
            chained_to_next = True
            chain_dest = "next intercept"
        else:
            traj_pre_5 = traj_ext.T                    # (5, n_steps+1) full arc
            vel_pre_5 = vel_ext.T
            ts_pre = ts_ext
            start_q5 = traj_ext[-1]                     # aim_joint2 at rest
            start_dq5 = vel_ext[-1]                     # ~0 (NN boundary condition)
            # No committed next pick. Only trek all the way to the home/standby pose
            # when the queue is EMPTY (no next object detected yet, ①a). When a next
            # object DOES exist but wasn't committed (different skill, or unreachable
            # after the throw — ①b/②), don't go home: lift the follow-through
            # (aim_joint2) to home Z and park there, so the next skill approaches
            # fresh from near the belt instead of after a wasted home round-trip.
            # copy() so the shared ctx.idle_joint is never mutated.
            if not ctx.queue:
                chain_target = self.idle_target().copy()
                chain_dest = "home/idle (queue empty)"
            else:
                chain_target = ctx.lifted_standby_joint(aim_joint2)
                chain_dest = "lifted standby (uncommitted next)"
            chained_to_next = False

        zero5 = np.zeros(5)
        traj_chain_5, vel_chain_5, ts_chain = trajectory(
            start_q5, start_dq5,
            chain_target[:5], zero5,
            ctx.M1[:5], ctx.M2[:5], hertz=ctx.cfg.TRAJ_HZ,
        )
        # Drop the chain's first column — it's start_q5 (the last sample of the
        # kept arc: release pose when cut, else aim_joint2), so it would be a
        # duplicate. Slice traj/vel/ts the SAME way so the three stay equal
        # length: if the chain degenerates to a single sample (chain_target ≈
        # start_q5 -> opt_time ≈ 0) all three become empty and only the arc
        # remains. (Slicing ts alone would
        # leave traj/vel one column longer, and zip() in _build_queue_waypoints
        # would then silently drop a waypoint and bind final_joint to the wrong
        # timestamp.)
        traj_chain_5 = traj_chain_5[:, 1:]
        vel_chain_5 = vel_chain_5[:, 1:]
        ts_chain_shifted = ts_chain[1:] + ts_pre[-1]

        traj_full_5 = np.concatenate((traj_pre_5, traj_chain_5), axis=1)   # (5, total)
        vel_full_5 = np.concatenate((vel_pre_5, vel_chain_5), axis=1)
        ts_full = np.concatenate((ts_pre, ts_chain_shifted))
        assert traj_full_5.shape[1] == vel_full_5.shape[1] == ts_full.shape[0], (
            f"throw traj/vel/timestep length mismatch: "
            f"{traj_full_5.shape[1]}/{vel_full_5.shape[1]}/{ts_full.shape[0]}"
        )

        # Pad to 6-dof, reorient to (6, total) as the queue expects.
        traj_throw = pad(traj_full_5.T).T
        vel_throw = pad(vel_full_5.T).T
        timestep_throw = ts_full
        final_joint = chain_target

        ctx.log.info(
            f"Throw T={params.T:.3f}s eta={params.eta:.3f} -> release step "
            f"{release_idx}/{traj_throw.shape[1] - 1} (eta step {eta_idx}, "
            f"lead {ctx.cfg.RELEASE_LEAD:.2f}s, "
            f"{'cut@release' if chained_to_next else 'full arc'}->{chain_dest})"
        )

        primed_next = ctx.traj_ctrl.send_trajectory_queue_with_timed_release(
            traj_throw, vel_throw, timestep_throw,
            final_joint=final_joint,
            release_index=release_idx,
            suction_on_at=next_suction_at,
        )
        self._last_throw_meta = {
            "T": params.T, "eta": params.eta,
            "release_idx": release_idx, "n_steps": n_steps,
        }
        return primed_next

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

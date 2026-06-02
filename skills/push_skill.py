"""Push skill: position at the intercept, wait for the object, and push it off the belt.

Mirrors the full structure of ``ThrowSkill`` — ambush at the intercept, wait
for the object, then execute a contact push.  Key differences from throw:

  * **No suction**: the gripper/TCP physically pushes the object.
  * **Rule-based push stroke**: a constant-speed Cartesian straight line on the
    belt plane (constant Z), NOT an NN-generated arc.

Trajectory segments (all concatenated into one dispatch):
  1. **Descent** (aim → grasp): time-optimal joint interpolation via
     ``trajectory()``.
  2. **Push stroke** (grasp → push_end): Cartesian interpolation at
     ``PUSH_SPEED`` m/s for ``PUSH_DISTANCE`` m, parallel to the belt
     surface.  Direction = T_grasp1 → T_aim2 projected onto XY.
  3. **Chain** (push_end → next intercept): time-optimal transition so the
     arm flows to the next pick cycle.

Shared primitives live on ``self.ctx`` (a ``SkillContext``):
  * ``ctx.robot`` (FK/IK), ``ctx.traj_ctrl``, ``ctx.M1``/``ctx.M2``, ``ctx.cfg``
  * ``ctx.move_through`` / ``ctx.wait_for_arrival_and_suction``
  * ``ctx.scan_next_intercept``
  * ``ctx.set_status(status, detail)``, ``ctx.set_active_target(None)``
"""

from __future__ import annotations

import csv
import datetime
import os
import time
from typing import TYPE_CHECKING, Optional

import numpy as np

from gp8_control.skills.base import ManipulationSkill, SkillResult
from gp8_control.trajectory.trajectory_primitive import (
    trajectory,
    pad,
)

if TYPE_CHECKING:
    from gp8_control.skills.context import PickRequest
    from gp8_control.tracking import TrackedObject


# =========================================================================
# Push policy (mirrors throw_skill's THETA_MAP / THROW_BIN_TARGET_MAP)
# =========================================================================

# Classes routed to push instead of throw by the ActionSelector. Objects whose
# class_name is in this set are handled by PushSkill.can_handle.
PUSH_CLASSES: set[str] = {
    "transparent",
    "metal",
}

# Per-class push-plane angle (radians, rotation about +Z in base frame).
# Determines the sweep direction on the belt plane. Same class keys as
# throw_skill's THETA_MAP — the push might need a different angle than the
# throw for the same class.
PUSH_THETA_MAP: dict[str, float] = {
    "transparent": -np.pi / 12.0,
    "metal":       -np.pi * 25.0 / 180.0,
}

# Per-class push bin TARGET (absolute base-frame XYZ, m). When a target's
# class is in this map, T_aim2 is OVERRIDDEN with these coordinates so the
# push aims at a fixed bin/chute location instead of computing from secondary.
PUSH_BIN_TARGET_MAP: dict[str, tuple] = {}


# ---- Push stroke parameters ------------------------------------------------

# TCP speed during the push stroke (m/s). The arm sweeps at this speed
# parallel to the belt surface. Tune to balance impact force vs. control
# stability; too fast may exceed joint velocity limits.
PUSH_SPEED: float = 1.0

# Push stroke distance (m). How far the TCP travels from T_grasp1 in the
# push direction. Must be long enough to clear the object off the belt but
# short enough to stay within the workspace.
PUSH_DISTANCE: float = 0.2

# Retreat distance (m) for the high wait pose.  The TCP waits on the line
# from the push target through the intercept point (T_grasp), but offset
# PUSH_RETREAT_DISTANCE behind T_grasp — i.e. in the direction *opposite*
# to the push.  This keeps TCP, object, and target collinear while giving
# the arm room to accelerate into the push stroke.
PUSH_RETREAT_DISTANCE: float = 0.20

# Minimum TCP X position (m) after retreat.  If the full retreat would
# place the TCP at X < PUSH_RETREAT_MIN_X, the retreat distance is scaled
# down proportionally so X stays at exactly this limit.  Prevents the arm
# from over-reaching toward the base.
PUSH_RETREAT_MIN_X: float = 0.27

# 6th joint angle (rad) for all push keyframes.  π/2 ≈ 90° clockwise
# (viewed from above) so the TCP faces the push direction.
PUSH_JOINT6_ANGLE: float = - np.pi / 2.0

# Minimum time gap (seconds) between consecutive queued trajectory
# points.  ``queue_traj_point`` round-trip through the URDF→raw
# bridge takes ~60-100 ms; 150 ms gives a comfortable margin so
# ``_push_waypoints`` never falls behind the robot's execution clock.
_MIN_QUEUE_GAP: float = 0.15

FIXED_DELAY_PUSH = 0.18


class PushSkill(ManipulationSkill):
    """Position at the intercept, wait for the object, and push it off the belt.

    ``execute`` runs the full push cycle — identical flow to ``ThrowSkill``:

      1. **POSITIONING** — ``move_through(current, aim, grasp)`` to drive to
         the grasp pose and park there.
      2. **WAITING** — ``wait_for_arrival_and_suction`` blocks until the object
         arrives.  (Suction fires but is immediately turned off since push is
         contact-based.)
      3. **PUSHING** — re-enter queue mode, compute push target, dispatch the
         descent + push stroke + chain trajectory.

    Public planning/build methods mirror ``ThrowSkill`` so external code
    (e.g. the legacy moving strategy in ``app.py``) can reuse push logic.
    """

    name = "push"

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._last_push_meta: dict = {}

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
    def can_handle(self, target: "TrackedObject") -> bool:
        """Accept objects whose class_name is in PUSH_CLASSES."""
        return target.class_name in PUSH_CLASSES

    # ------------------------------------------------------------------
    # Skill entry point (ambush strategy)
    # ------------------------------------------------------------------
    def execute(self, request: "PickRequest") -> SkillResult:
        """Pre-position (with retreat), wait for arrival, then push.

        Follows the same flow as ``ThrowSkill.execute``:

        1. ``enter_queue_mode`` → ``move_through(current, aim, wait_joint)``
           (POSITIONING).  ``wait_joint`` is the retreat pose — offset
           behind T_grasp opposite to the push direction, at aim height.
        2. ``wait_for_arrival_and_suction`` (WAITING) — blocks until the
           object reaches the intercept line.
        3. ``enter_queue_mode`` → dispatch descent + push stroke + chain
           trajectory (PUSHING).
        4. Cleanup.
        """
        ctx = self.ctx
        target = request.target
        current_joint = request.current_joint
        aim_joint = request.aim_joint
        grasp_joint = request.grasp_joint
        T_aim = request.T_aim
        T_grasp = request.T_grasp
        secondary = request.secondary

        # Push does NOT use suction — ensure it's off from any prior cycle.
        ctx.traj_ctrl.suction_off()

        theta = PUSH_THETA_MAP.get(target.class_name, 0.0)

        # ---- Compute push target (T_aim2) early ----
        # We need the push direction *before* positioning so we can place
        # the wait pose on the line behind T_grasp, opposite to the push.
        bin_xyz = PUSH_BIN_TARGET_MAP.get(target.class_name)
        if bin_xyz is not None:
            T_aim2 = np.eye(4)
            T_aim2[:3, :3] = T_aim[:3, :3]
            T_aim2[:3, 3] = np.asarray(bin_xyz, dtype=float)
            ctx.log.info(
                f"Push target for {target.class_name}: fixed bin "
                f"({bin_xyz[0]:+.3f}, {bin_xyz[1]:+.3f}, {bin_xyz[2]:+.3f}) m"
            )
        else:
            T_aim2 = self.plan_push_target(
                T_grasp, theta, T_aim, time.time(), secondary,
            )

        # ---- Compute retreat wait pose ----
        wait_joint = self._compute_retreat_wait_joint(
            T_grasp, T_aim2, T_aim, aim_joint,
        )

        # Re-enter point queue mode each cycle (same as ThrowSkill).
        if not ctx.traj_ctrl.enter_queue_mode():
            ctx.log.error("Failed to (re)enter queue mode; skipping this pick")
            return SkillResult(False, "enter_queue_mode (pick) failed")

        # ---- 1. POSITIONING: drive to retreat wait pose and park ----
        ctx.set_status("POSITIONING", target.class_name)
        ctx.move_through(current_joint, aim_joint, wait_joint)

        # ---- 2. WAITING: block until the object arrives ----
        ctx.set_status("WAITING", target.class_name)
        ctx.wait_for_arrival(target, T_grasp[1, 3], offset = FIXED_DELAY_PUSH)

        # ---- 3. PUSHING: re-enter queue mode and dispatch push traj ----
        ctx.set_status("PUSHING", target.class_name)
        if not ctx.traj_ctrl.enter_queue_mode():
            ctx.log.error("Failed to (re)enter queue mode for push; dropping object")
            ctx.traj_ctrl.suction_off()
            ctx.set_active_target(None)
            ctx.set_status("IDLE", "")
            return SkillResult(False, "enter_queue_mode (push) failed")

        # Chain target for post-push transition (next pick's intercept).
        next_intercept_joint = ctx.scan_next_intercept()

        # Build & dispatch: descent (wait→grasp) + push stroke + chain.
        self.build_push_trajectory(
            wait_joint, grasp_joint, T_grasp, T_aim2, theta,
            next_intercept_joint=next_intercept_joint,
        )

        # ---- 4. Cleanup ----
        # Safety: turn suction off in case wait_for_arrival_and_suction
        # left it on (push is contact-based, no suction needed).
        ctx.traj_ctrl.suction_off()
        self._log_push_cycle(target)
        ctx.set_active_target(None)
        ctx.set_status("IDLE", "")
        return SkillResult(True, "push complete")

    # ------------------------------------------------------------------
    # Retreat wait pose computation
    # ------------------------------------------------------------------
    def _compute_retreat_wait_joint(
        self,
        T_grasp: np.ndarray,
        T_aim2: np.ndarray,
        T_aim: np.ndarray,
        aim_joint: np.ndarray,
    ) -> np.ndarray:
        """Compute retreat wait joint: behind T_grasp opposite the push, at aim height.

        Returns ``wait_joint`` (6-DOF). Falls back to ``aim_joint`` on IK failure.
        """
        ctx = self.ctx
        push_dir = self._compute_push_direction(T_grasp, T_aim2)
        dx = PUSH_RETREAT_DISTANCE * push_dir[0]
        dy = PUSH_RETREAT_DISTANCE * push_dir[1]

        # Clamp so X doesn't go below PUSH_RETREAT_MIN_X.
        if T_grasp[0, 3] - dx < PUSH_RETREAT_MIN_X and dx > 0:
            available = max(0.0, T_grasp[0, 3] - PUSH_RETREAT_MIN_X)
            if available <= 0:
                return aim_joint.copy()
            scale = available / dx
            dx *= scale
            dy *= scale

        T_wait = T_grasp.copy()
        T_wait[0, 3] -= dx
        T_wait[1, 3] -= dy
        T_wait[2, 3] = T_aim[2, 3]  # aim height

        ik = ctx.robot.inverse_kinematics(T_wait)
        if ik is None:
            ctx.log.warn("Retreat wait IK failed; falling back to aim_joint")
            return aim_joint.copy()

        wait_joint = np.asarray(ik, dtype=float)
        wait_joint[-1] = PUSH_JOINT6_ANGLE
        return wait_joint

    # ------------------------------------------------------------------
    # Push target planning (mirrors throw_skill.plan_throw_landing)
    # ------------------------------------------------------------------
    def plan_push_target(
        self,
        T_grasp1: np.ndarray,
        theta: float,
        T_aim1_fallback: np.ndarray,
        now: float,
        secondary: "Optional[TrackedObject]",
    ) -> np.ndarray:
        """Aim push at ``secondary`` if feasible; else use T_aim1 fallback.

        Same logic as ``ThrowSkill.plan_throw_landing`` — reuses the planner's
        ``plan_throw_landing`` to compute where the secondary object will be,
        then validates feasibility. Falls back to ``T_aim1_fallback`` (the
        aim hover) when no secondary is available or the target is out of
        reach.
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
    # Keyframe IK solver (mirrors throw_skill.solve_keyframe_joints)
    # ------------------------------------------------------------------
    def solve_keyframe_joints(
        self,
        T_aim1: np.ndarray,
        T_grasp1: np.ndarray,
        T_aim2: np.ndarray,
    ):
        """IK for aim, grasp, and push-target keyframes; zero last joint.

        Returns ``(aim_joint, grasp_joint, push_target_joint)`` or ``None``
        on any IK failure.
        """
        ctx = self.ctx
        aim_joint1 = ctx.robot.inverse_kinematics(T_aim1)
        grasp_joint1 = ctx.robot.inverse_kinematics(T_grasp1)
        aim_joint2 = ctx.robot.inverse_kinematics(T_aim2)
        if aim_joint1 is None or grasp_joint1 is None or aim_joint2 is None:
            ctx.log.warn("Push IK failed for keyframes; aborting")
            return None
        aim_joint1 = np.asarray(aim_joint1, dtype=float)
        grasp_joint1 = np.asarray(grasp_joint1, dtype=float)
        aim_joint2 = np.asarray(aim_joint2, dtype=float)
        aim_joint1[-1] = PUSH_JOINT6_ANGLE
        grasp_joint1[-1] = PUSH_JOINT6_ANGLE
        aim_joint2[-1] = PUSH_JOINT6_ANGLE
        return aim_joint1, grasp_joint1, aim_joint2

    # ------------------------------------------------------------------
    # Push trajectory build + dispatch
    # ------------------------------------------------------------------
    def build_push_trajectory(
        self,
        aim_joint: np.ndarray,
        grasp_joint: np.ndarray,
        T_grasp: np.ndarray,
        T_aim2: np.ndarray,
        theta: float,
        next_intercept_joint: "Optional[np.ndarray]" = None,
    ) -> None:
        """Build and dispatch the 3-segment push trajectory.

        Segments:

        1. **Descent** (aim_joint → grasp_joint): time-optimal joint
           interpolation via ``trajectory()``. Brings the arm from the high
           hover down to the near-object pose at belt height.

        2. **Push stroke** (grasp → push_end): rule-based Cartesian straight
           line at ``PUSH_SPEED`` m/s for ``PUSH_DISTANCE`` m, parallel to
           the belt surface (constant Z). Direction is from T_grasp towards
           T_aim2, projected onto the XY plane. Each Cartesian waypoint is
           converted to joint space via IK. Boundary velocities are zero for
           smooth concatenation with the adjacent segments.

        3. **Chain** (push_end → next_intercept or grasp): time-optimal
           transition so the arm flows directly to the next pick cycle,
           same pattern as ``ThrowSkill.build_throw_trajectory``.

        No suction release is scheduled — push is contact-based, so
        ``send_trajectory_queue`` is used instead of
        ``send_trajectory_queue_with_timed_release``.
        """
        ctx = self.ctx
        zero5 = np.zeros(5)

        # ================================================================
        # Segment 1: Descent  (aim → grasp, time-optimal trapezoidal)
        # ================================================================
        traj_desc_5, vel_desc_5, ts_desc = trajectory(
            aim_joint[:5], zero5,
            grasp_joint[:5], zero5,
            ctx.M1[:5], ctx.M2[:5], hertz=ctx.cfg.TRAJ_HZ,
        )
        # traj_desc_5, vel_desc_5: (5, n_desc); ts_desc: (n_desc,)

        # ================================================================
        # Segment 2: Push stroke  (Cartesian, belt-parallel, constant speed)
        # ================================================================
        push_dir = self._compute_push_direction(T_grasp, T_aim2)
        traj_stroke_5, vel_stroke_5, ts_stroke = self._build_push_stroke(
            T_grasp, push_dir, grasp_joint,
        )

        # Clamp segment velocities against robot joint limits (Yaskawa alarm
        # 4414 prevention). Same 2-pass rescale approach as throw_skill.
        traj_stroke_5, vel_stroke_5, ts_stroke = self._clamp_stroke_velocity(
            traj_stroke_5, vel_stroke_5, ts_stroke,
        )

        # Shift stroke timestamps to follow descent; drop stroke's first
        # sample (it duplicates descent's last = grasp_joint).
        ts_stroke_shifted = ts_stroke[1:] + ts_desc[-1]
        traj_stroke_5 = traj_stroke_5[:, 1:]
        vel_stroke_5 = vel_stroke_5[:, 1:]

        # ================================================================
        # Segment 3: Chain  (push_end → next intercept or grasp)
        # ================================================================
        push_end_q5 = traj_stroke_5[:, -1]      # last waypoint of stroke
        push_end_dq5 = vel_stroke_5[:, -1]      # ~0 (boundary condition)

        chain_target = (
            np.asarray(next_intercept_joint, dtype=float)
            if next_intercept_joint is not None
            else np.asarray(grasp_joint, dtype=float)
        )
        chain_target[-1] = PUSH_JOINT6_ANGLE
        chained_to_next = next_intercept_joint is not None

        traj_chain_5, vel_chain_5, ts_chain = trajectory(
            push_end_q5, push_end_dq5,
            chain_target[:5], zero5,
            ctx.M1[:5], ctx.M2[:5], hertz=ctx.cfg.TRAJ_HZ,
        )
        # Drop chain's first sample (duplicate of stroke's last), shift ts.
        stroke_end_t = (
            ts_stroke_shifted[-1]
            if len(ts_stroke_shifted) > 0
            else ts_desc[-1]
        )
        traj_chain_5 = traj_chain_5[:, 1:]
        vel_chain_5 = vel_chain_5[:, 1:]
        ts_chain_shifted = ts_chain[1:] + stroke_end_t

        # ================================================================
        # Concatenate all segments
        # ================================================================
        traj_full_5 = np.concatenate(
            (traj_desc_5, traj_stroke_5, traj_chain_5), axis=1,
        )
        vel_full_5 = np.concatenate(
            (vel_desc_5, vel_stroke_5, vel_chain_5), axis=1,
        )
        ts_full = np.concatenate((ts_desc, ts_stroke_shifted, ts_chain_shifted))
        assert traj_full_5.shape[1] == vel_full_5.shape[1] == ts_full.shape[0], (
            f"push traj/vel/ts length mismatch: "
            f"{traj_full_5.shape[1]}/{vel_full_5.shape[1]}/{ts_full.shape[0]}"
        )

        # Pad to 6-DOF, then set the 6th joint to PUSH_JOINT6_ANGLE.
        traj_push = pad(traj_full_5.T).T
        traj_push[5, :] = PUSH_JOINT6_ANGLE
        vel_push = pad(vel_full_5.T).T       # 6th vel = 0 is correct
        final_joint = chain_target

        n_desc = traj_desc_5.shape[1]
        n_stroke = traj_stroke_5.shape[1]
        ctx.log.info(
            f"Push traj: descent {n_desc} + stroke {n_stroke} steps "
            f"(d={PUSH_DISTANCE:.3f}m @ {PUSH_SPEED:.2f}m/s, "
            f"θ={np.degrees(theta):.1f}°), "
            f"{'chain→next intercept' if chained_to_next else 'chain→current grasp'}"
        )

        # Dispatch. No timed release — push is contact-based, no suction.
        ctx.traj_ctrl.send_trajectory_queue(
            traj_push, vel_push, ts_full, final_joint=final_joint,
        )

        self._last_push_meta = {
            "n_descent": n_desc,
            "n_stroke": n_stroke,
            "push_distance": PUSH_DISTANCE,
            "push_speed": PUSH_SPEED,
            "theta": theta,
            "chained_to_next": chained_to_next,
        }

    # ------------------------------------------------------------------
    # Queue-safe decimation
    # ------------------------------------------------------------------

    @staticmethod
    def _decimate_for_queue(
        traj_5: np.ndarray,
        vel_5: np.ndarray,
        ts: np.ndarray,
        min_gap: float = _MIN_QUEUE_GAP,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Thin a trajectory so consecutive points are ≥ *min_gap* apart.

        MotoROS2 ``queue_traj_point`` is a synchronous service call whose
        round-trip through the URDF→raw bridge takes ~60-100 ms.  If
        consecutive trajectory points are closer in time than that latency,
        the robot consumes queued points faster than ``_push_waypoints``
        can supply them → the internal buffer drains → MotoROS2 auto-exits
        point-queue mode → code 2 "Must call start_point_queue_mode".

        Decimation keeps the **first** and **last** point unconditionally
        and retains interior points only when they are ≥ *min_gap* from
        the most-recently-kept point.  Joint velocities are recomputed
        via central finite differences at the new (wider) spacing so that
        MotoROS2's interpolation stays consistent.

        This is applied as a post-processing step after the full
        trajectory (startup + positioning + hold + descent + stroke +
        chain) is concatenated, so it is guaranteed to cover **all**
        segments regardless of their original sample rate.
        """
        n = traj_5.shape[1]
        if n <= 2:
            return traj_5, vel_5, ts

        # Greedy decimation: walk interior points, keep those ≥ min_gap
        # from the last kept point.  First and last are always kept.
        keep = [0]
        for i in range(1, n - 1):
            if ts[i] - ts[keep[-1]] >= min_gap:
                keep.append(i)
        keep.append(n - 1)

        # If the last interior point is too close to the final point,
        # drop it so the invariant (ALL gaps ≥ min_gap) holds at the end.
        while len(keep) > 2 and ts[keep[-1]] - ts[keep[-2]] < min_gap:
            keep.pop(-2)

        idx = np.array(keep)
        traj_dec = traj_5[:, idx]
        ts_dec = ts[idx]

        # Recompute velocities via central finite differences.
        m = len(idx)
        vel_dec = np.zeros_like(traj_dec)
        if m > 2:
            for j in range(1, m - 1):
                dt2 = ts_dec[j + 1] - ts_dec[j - 1]
                if dt2 > 1e-9:
                    vel_dec[:, j] = (
                        traj_dec[:, j + 1] - traj_dec[:, j - 1]
                    ) / dt2
        # Boundary velocities stay zero (start/end at rest).

        return traj_dec, vel_dec, ts_dec

    # ------------------------------------------------------------------
    # Push stroke helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_push_direction(
        T_grasp: np.ndarray,
        T_aim2: np.ndarray,
    ) -> np.ndarray:
        """Unit direction from T_grasp to T_aim2 projected onto the XY belt plane.

        Returns a 3D unit vector with Z=0. Falls back to +X if T_grasp and
        T_aim2 coincide in XY.
        """
        delta = T_aim2[:3, 3] - T_grasp[:3, 3]
        delta[2] = 0.0                         # project onto belt plane
        norm = np.linalg.norm(delta)
        if norm < 1e-6:
            # Degenerate case: push along +X as a safe default.
            return np.array([1.0, 0.0, 0.0])
        return delta / norm

    def _build_push_stroke(
        self,
        T_grasp: np.ndarray,
        direction: np.ndarray,
        grasp_joint: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Constant-speed Cartesian trajectory parallel to the belt surface.

        Generates ``n_steps + 1`` waypoints along a straight line:

        * Start: ``T_grasp`` (= T_grasp1, at belt height)
        * End:   ``T_grasp + PUSH_DISTANCE * direction`` (same Z)

        Each waypoint is converted to joint space via IK. If IK fails
        mid-stroke, the trajectory is truncated with a warning.

        Joint velocities are computed via central finite differences with
        boundary velocities forced to zero so the stroke concatenates
        smoothly with the descent (v=0 at end) and chain (v=0 at start)
        segments.

        Returns ``(traj_5, vel_5, ts)`` with shapes ``(5, n), (5, n), (n,)``.
        """
        ctx = self.ctx
        total_time = PUSH_DISTANCE / max(PUSH_SPEED, 1e-6)
        n_steps = max(2, int(total_time * ctx.cfg.TRAJ_HZ))
        dt = total_time / n_steps

        # ---- Cartesian waypoints → joint space via IK ----
        waypoints: list[np.ndarray] = []
        for i in range(n_steps + 1):
            alpha = i / n_steps
            T_wp = T_grasp.copy()
            T_wp[0, 3] += alpha * PUSH_DISTANCE * direction[0]
            T_wp[1, 3] += alpha * PUSH_DISTANCE * direction[1]
            # Z unchanged — belt-parallel motion.
            ik = ctx.robot.inverse_kinematics(T_wp)
            if ik is None:
                ctx.log.warn(
                    f"Push stroke IK failed at step {i}/{n_steps}; "
                    f"truncating stroke to {len(waypoints)} waypoints"
                )
                break
            q = np.asarray(ik, dtype=float)
            # Only q[:5] is used below; 6th joint is handled by
            # build_push_trajectory after pad().
            waypoints.append(q[:5])

        # Fallback: if fewer than 2 waypoints, return a zero-motion segment
        # so the caller can still concatenate without crashing.
        if len(waypoints) < 2:
            ctx.log.warn("Push stroke degenerate (< 2 IK solutions); no-op segment")
            q0 = np.asarray(grasp_joint[:5], dtype=float)
            traj = np.column_stack([q0, q0])
            vel = np.zeros_like(traj)
            ts = np.array([0.0, dt])
            return traj, vel, ts

        n = len(waypoints)
        traj = np.column_stack(waypoints)              # (5, n)
        ts = np.linspace(0.0, dt * (n - 1), n)        # (n,)

        # ---- Joint velocities via central finite differences ----
        vel = np.zeros_like(traj)                      # (5, n)
        if n > 2:
            vel[:, 1:-1] = (traj[:, 2:] - traj[:, :-2]) / (2.0 * dt)
        # Boundary: start and end at rest so the stroke concatenates
        # smoothly with descent (dq=0 at grasp) and chain (dq=0 at start).
        vel[:, 0] = 0.0
        vel[:, -1] = 0.0

        return traj, vel, ts

    def _clamp_stroke_velocity(
        self,
        traj: np.ndarray,
        vel: np.ndarray,
        ts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stretch stroke timestamps if any segment exceeds joint velocity limits.

        Same 2-pass rescale as ``ThrowSkill.build_throw_trajectory``. Because
        the push stroke is Cartesian-interpolated, a fast ``PUSH_SPEED`` can
        produce joint-space segment velocities that exceed ``M1`` in certain
        arm configurations → Yaskawa alarm 4414 "excessive segment velocity".

        Stretching all timestamps by the over-limit ratio uniformly slows the
        stroke while preserving the Cartesian path.
        """
        ctx = self.ctx
        m1_5 = np.asarray(ctx.M1[:5], dtype=float)

        for _ in range(2):
            dt_seg = np.maximum(np.diff(ts), 1e-9)
            seg_vel = np.abs(np.diff(traj, axis=1)) / dt_seg[None, :]  # (5, n-1)
            ratio = float(np.max(seg_vel / m1_5[:, None]))
            if ratio <= 1.0:
                break
            scale = ratio * 1.05                       # +5% margin
            ts = ts * scale

            # Re-derive joint velocities at the stretched time scale.
            n = traj.shape[1]
            new_dt = ts[1] - ts[0] if n > 1 else 1e-9  # still uniform
            vel = np.zeros_like(traj)
            if n > 2:
                vel[:, 1:-1] = (traj[:, 2:] - traj[:, :-2]) / (2.0 * new_dt)
            vel[:, 0] = 0.0
            vel[:, -1] = 0.0

            ctx.log.warn(
                f"Push stroke clamped: seg vel ratio {ratio:.2f} > 1 "
                f"→ time scale ×{scale:.2f}"
            )
        return traj, vel, ts

    # ------------------------------------------------------------------
    # Per-cycle timing log (mirrors throw_skill._log_throw_cycle)
    # ------------------------------------------------------------------
    def _log_push_cycle(self, target: "TrackedObject") -> None:
        """Append one push-cycle timing row to PICK_LOG_CSV for offline analysis.

        Logs push-specific parameters (speed, distance, angle, step counts)
        alongside the shared conveyor/timing fields.
        """
        ctx = self.ctx
        path = ctx.cfg.PICK_LOG_CSV
        if not path:
            return

        # Reuse throw timing dict if available (same traj_ctrl attributes).
        lt = getattr(ctx.traj_ctrl, "last_throw", {}) or {}
        meta = self._last_push_meta or {}

        row = {
            "iso_time": datetime.datetime.now().isoformat(timespec="milliseconds"),
            "skill": "push",
            "class": target.class_name,
            "belt_mps": round(ctx.conveyor.current, 4),
            "push_speed_mps": meta.get("push_speed", ""),
            "push_distance_m": meta.get("push_distance", ""),
            "theta_deg": round(np.degrees(meta.get("theta", 0.0)), 1),
            "n_descent": meta.get("n_descent", ""),
            "n_stroke": meta.get("n_stroke", ""),
            "chained": meta.get("chained_to_next", ""),
            "io_ms": round(lt["io_ms"], 1) if lt.get("io_ms") is not None else "",
        }
        try:
            new_file = not os.path.exists(path) or os.path.getsize(path) == 0
            with open(path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if new_file:
                    w.writeheader()
                w.writerow(row)
        except OSError as e:
            ctx.log.warn(f"push-log write failed: {e}")

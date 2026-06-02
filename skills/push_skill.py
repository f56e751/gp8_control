"""Push skill: position above the intercept, descend, and push the object off the belt.

Mirrors the full structure of ``ThrowSkill`` — ambush at the intercept, wait
for the object, then execute a contact push. Key differences from throw:

  * **No suction**: the gripper/TCP physically pushes the object.
  * **Wait at T_aim1 (high)**: instead of parking at grasp height, the arm
    hovers above the intercept and descends only when the object approaches.
  * **Rule-based push stroke**: a constant-speed Cartesian straight line on the
    belt plane (constant Z), NOT an NN-generated arc.

Trajectory segments (all concatenated into one dispatch):
  1. **Descent** (aim → grasp): time-optimal joint interpolation via
     ``trajectory()``. Timed so the arm arrives at T_grasp1 exactly when
     the object reaches the intercept.
  2. **Push stroke** (grasp → push_end): Cartesian interpolation at
     ``PUSH_SPEED`` m/s for ``PUSH_DISTANCE`` m, parallel to the belt
     surface. Direction = T_grasp1 → T_aim2 projected onto XY.
  3. **Chain** (push_end → next intercept): time-optimal transition so the
     arm flows to the next pick cycle.

Shared primitives live on ``self.ctx`` (a ``SkillContext``):
  * ``ctx.robot`` (FK/IK), ``ctx.traj_ctrl``, ``ctx.M1``/``ctx.M2``, ``ctx.cfg``
  * ``ctx.move_through`` / ``ctx.sleep_until`` / ``ctx.scan_next_intercept``
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
PUSH_DISTANCE: float = 0.3

# Small lead time (s) subtracted from the computed descent-start time to
# compensate for trajectory dispatch latency (queue setup, ROS transport).
PUSH_DESCENT_LEAD: float = 0.1


class PushSkill(ManipulationSkill):
    """Position above the intercept, descend, and push the object off the belt.

    ``execute`` runs the full push cycle:

      1. **POSITIONING** — drive to T_aim1 (high hover above intercept), park.
      2. **WAITING** — block until it's time to begin the descent so the arm
         reaches T_grasp1 (low, near-object pose) exactly when the object
         arrives. No suction is used.
      3. **PUSHING** — dispatch the 3-segment trajectory: descent (aim→grasp)
         + push stroke (Cartesian, belt-parallel, constant speed) + chain
         (push_end → next intercept).

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
        """Full push cycle: position high → wait → descend + push → chain."""
        ctx = self.ctx
        target = request.target
        current_joint = request.current_joint
        aim_joint = request.aim_joint        # T_aim1: high wait pose
        grasp_joint = request.grasp_joint    # T_grasp1: low, near-object pose
        T_aim = request.T_aim               # 4×4 SE3 of aim pose
        T_grasp = request.T_grasp           # 4×4 SE3 of grasp pose
        secondary = request.secondary

        # Push does NOT use suction — ensure it's off from any prior cycle.
        ctx.traj_ctrl.suction_off()

        # ---- 1. Queue mode entry (positioning) ----
        # Re-enter queue mode each cycle (same MotoROS2 workaround as throw).
        if not ctx.traj_ctrl.enter_queue_mode():
            ctx.log.error("Failed to enter queue mode for push positioning")
            return SkillResult(False, "enter_queue_mode (positioning) failed")

        # ---- 2. POSITIONING: drive to T_aim1 (high hover) and park ----
        # Unlike throw (which parks at grasp height), push parks HIGH so the
        # gripper clears the belt while waiting. move_through's 3rd arg is the
        # destination, so pass aim_joint to park at the high hover.
        ctx.set_status("POSITIONING", target.class_name)
        ctx.move_through(current_joint, aim_joint, aim_joint)

        # ---- 3. WAITING: hold at T_aim1 until time to descend ----
        # No suction — we compute ETA, subtract the descent duration, and
        # sleep so the subsequent trajectory dispatch starts the descent at
        # exactly the right moment for the push stroke to coincide with
        # object arrival at the intercept.
        ctx.set_status("WAITING", target.class_name)
        self._wait_for_approach(target, T_grasp, aim_joint, grasp_joint)

        # ---- 4. PUSHING ----
        ctx.set_status("PUSHING", target.class_name)
        # Re-enter queue mode (MotoROS2 left it after positioning drained).
        if not ctx.traj_ctrl.enter_queue_mode():
            ctx.log.error("Failed to enter queue mode for push sweep")
            ctx.set_active_target(None)
            ctx.set_status("IDLE", "")
            return SkillResult(False, "enter_queue_mode (push sweep) failed")

        theta = PUSH_THETA_MAP.get(target.class_name, 0.0)

        # Push target (T_aim2). Fixed-bin override or dynamic planning, same
        # pattern as throw_skill's T_aim2 selection.
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

        # Chain target for post-push transition (next pick's intercept).
        next_intercept_joint = ctx.scan_next_intercept()

        # Build & dispatch the full trajectory: descent + push stroke + chain.
        self.build_push_trajectory(
            aim_joint, grasp_joint, T_grasp, T_aim2, theta,
            next_intercept_joint=next_intercept_joint,
        )

        # ---- 5. Cleanup ----
        self._log_push_cycle(target)
        ctx.set_active_target(None)
        ctx.set_status("IDLE", "")
        return SkillResult(True, "push complete")

    # ------------------------------------------------------------------
    # Wait for object approach (no suction — push-specific)
    # ------------------------------------------------------------------
    def _wait_for_approach(
        self,
        target: "TrackedObject",
        T_grasp: np.ndarray,
        aim_joint: np.ndarray,
        grasp_joint: np.ndarray,
    ) -> None:
        """Block at T_aim1 until it's time to start the descent.

        Computes the object's ETA at the intercept line, estimates the
        descent duration (aim → grasp via ``trajectory()``), and sleeps
        until ``ETA − descent_time − PUSH_DESCENT_LEAD``. When this method
        returns the caller immediately dispatches the descent+push trajectory
        so the arm arrives at T_grasp1 just as the object reaches the
        intercept.

        No suction is fired — push uses gripper/TCP contact only.
        """
        ctx = self.ctx
        now = time.time()
        v = ctx.conveyor.current
        obj_y = ctx.object_y_now(target, now, v)
        intercept_y = T_grasp[1, 3]
        eta = (obj_y - intercept_y) / (v + 1e-6)
        eta = max(0.0, min(eta, ctx.cfg.AMBUSH_MAX_WAIT))

        # Estimate descent time by building a temporary trajectory. This is
        # the same trapezoidal profile that the actual descent segment will
        # use, so the time estimate is exact (not a rough guess).
        zero = np.zeros_like(ctx.M1)
        _, _, ts_desc = trajectory(
            aim_joint, zero, grasp_joint, zero,
            ctx.M1, ctx.M2, hertz=ctx.cfg.TRAJ_HZ,
        )
        descent_time = float(ts_desc[-1]) if len(ts_desc) > 0 else 0.0

        wait_sec = max(0.0, eta - descent_time - PUSH_DESCENT_LEAD)
        ctx.log.info(
            f"Push ambush: wait {wait_sec:.2f}s, then descend {descent_time:.2f}s "
            f"(ETA {eta:.2f}s, dist {obj_y - intercept_y:.3f} m, "
            f"belt {v:.3f} m/s, lead {PUSH_DESCENT_LEAD:.2f}s)"
        )
        ctx.sleep_until(now + wait_sec)

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
        aim_joint1[-1] = 0.0
        grasp_joint1[-1] = 0.0
        aim_joint2[-1] = 0.0
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

        # Pad to 6-DOF (zero 6th joint), reorient to (6, total).
        traj_push = pad(traj_full_5.T).T
        vel_push = pad(vel_full_5.T).T
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
            q[-1] = 0.0                       # zero 6th joint
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

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
from gp8_control.trajectory.trajectory_primitive import trajectory

if TYPE_CHECKING:
    from gp8_control.skills.context import PickRequest
    from gp8_control.tracking import TrackedObject


# =========================================================================
# Push policy (mirrors throw_skill's THETA_MAP / THROW_BIN_TARGET_MAP)
# =========================================================================

# Classes this skill ACCEPTS (PushSkill.can_handle). The ActionSelector routes
# by app Config.SKILL_BY_CLASS; this set is the skill's own guard so it refuses
# anything it shouldn't handle (selector then falls back to throw). Cans
# ("metal") are pushed; PET bottles ("transparent") are suctioned/thrown, so
# they are intentionally NOT in this set.
PUSH_CLASSES: set[str] = {
    "metal",
}

# NOTE: the old per-class PUSH_THETA_MAP is gone — the push-plane angle
# ``theta`` is now derived geometrically in ``execute`` as the XY heading
# from the aim hover (T_aim1) toward the push target (T_aim2).

# Per-class push bin TARGET (absolute base-frame XYZ, m). When a target's
# class is in this map, T_aim2 is OVERRIDDEN with these coordinates so the
# push aims at a fixed bin/chute location instead of computing from secondary.
PUSH_BIN_TARGET_MAP: dict[str, tuple] = {
    "transparent": (1.2, -0.30, 0.0),  
    "metal":       (1.2,  0.50, 0.0),
}



# ---- Push stroke parameters ------------------------------------------------

# TCP speed during the push stroke (m/s). The arm sweeps at this speed
# parallel to the belt surface. Tune to balance impact force vs. control
# stability; too fast may exceed joint velocity limits.
PUSH_SPEED: float = 2.0

# Push stroke distance (m). How far the TCP travels from T_grasp1 in the
# push direction. Must be long enough to clear the object off the belt but
# short enough to stay within the workspace. This is the DEFAULT used when a
# class is not in PUSH_DISTANCE_MAP below.
PUSH_DISTANCE: float = 0.3

# Per-class push stroke distance (m). Mirrors PUSH_BIN_TARGET_MAP: when a
# target's class is here, its stroke travels this far instead of the default
# PUSH_DISTANCE (e.g. a heavier/larger item may need a longer sweep to clear
# the belt). Classes absent from the map fall back to PUSH_DISTANCE.
PUSH_DISTANCE_MAP: dict[str, float] = {
    "transparent": 0.30,
    "metal":       0.35,
}

# Retreat distance (m) for the high wait pose.  The TCP waits on the line
# from the push target through the intercept point (T_grasp), but offset
# PUSH_RETREAT_DISTANCE behind T_grasp — i.e. in the direction *opposite*
# to the push.  This keeps TCP, object, and target collinear while giving
# the arm room to accelerate into the push stroke.
PUSH_RETREAT_DISTANCE: float = 0.2

# Minimum TCP X position (m) after retreat.  If the full retreat would
# place the TCP at X < PUSH_RETREAT_MIN_X, the retreat distance is scaled
# down proportionally so X stays at exactly this limit.  Prevents the arm
# from over-reaching toward the base.
PUSH_RETREAT_MIN_X: float = 0.25

# 6th joint angle (rad) for all push keyframes.  π/2 ≈ 90° clockwise
# (viewed from above) so the TCP faces the push direction.
PUSH_JOINT6_ANGLE: float = - np.pi / 2.0

# # Empirical wrist offset (rad) ADDED to joint 6 after IK so the pusher face
# # lands at the intended angle on hardware — the _push_orientation twist alone
# # leaves joint 6 ~90° short. Pure flange roll: it does NOT change the approach
# # axis, so the front/back swing is unaffected. Tune if the gripper is remounted.
# PUSH_JOINT6_OFFSET: float = np.pi / 2.0

# Minimum time gap (seconds) between consecutive queued trajectory
# points.  ``queue_traj_point`` round-trip through the URDF→raw
# bridge takes ~60-100 ms; 150 ms gives a comfortable margin so
# ``_push_waypoints`` never falls behind the robot's execution clock.
_MIN_QUEUE_GAP: float = 0.15

FIXED_DELAY_PUSH = 0.8
# Push-stroke depth below grasp height. A lower stroke (larger value) shrinks
# the joint-5 margin to the wrist singularity, so it must be balanced against
# the forward swing end (SWING_BIAS + SWING_ANGLE): at 0.06 the singularity is
# still cleared as long as that forward end stays ≲ +5° (see SWING_BIAS).
# HEIGHT_OFFSET = 0.05
PUSH_HEIGHT = 0.01

# Swing push *half-amplitude* (rad). During the stroke the TCP tilts
# progressively about the horizontal axis perpendicular to the push direction,
# sweeping from ``SWING_BIAS - SWING_ANGLE`` (leaning *back*) at the start to
# ``SWING_BIAS + SWING_ANGLE`` (leaning *forward*) at the end — like a paddle
# swing. The TCP *position* still travels a straight belt-parallel line; only
# the orientation sweeps. Set to 0.0 for a pure perpendicular push.
SWING_ANGLE: float = np.radians(20.0)

# Swing *bias* (rad): centre of the swing sweep. Negative = biased toward
# leaning back. Kept negative so the forward end (``SWING_BIAS + SWING_ANGLE``)
# stays small — a large forward tilt drives joint 5 through 0 (wrist
# singularity), which flips the joint-6 IK branch at the swing end. With
# SWING_ANGLE=20° and SWING_BIAS=-15° the sweep is -35°→+5°: a big (40°) swing
# feel whose forward reach never nears the singularity.
SWING_BIAS: float = np.radians(-5.0)


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
        # aim_joint[-1] += PUSH_JOINT6_ANGLE
        grasp_joint = request.grasp_joint
        T_aim = request.T_aim
        T_grasp = request.T_grasp
        secondary = request.secondary

        # Push does NOT use suction — ensure it's off from any prior cycle.
        ctx.traj_ctrl.suction_off()

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
            # No fixed bin: aim at the secondary object. theta=0 here — the
            # NN-plane rotation is no longer driven by a per-class map; the
            # geometric theta is computed from T_aim2 just below.
            T_aim2 = self.plan_push_target(
                T_grasp, 0.0, T_aim, time.time(), secondary,
            )

        # theta = XY heading from the aim hover (T_aim1) toward the push
        # target (T_aim2). Replaces the old per-class PUSH_THETA_MAP.
        push_heading = self._compute_push_direction(T_aim, T_aim2)
        theta = float(np.arctan2(push_heading[1], push_heading[0]))

        # ---- Compute retreat poses (wait at aim height + grasp at grasp height) ----
        wait_joint, grasp_retreat_joint, T_grasp_retreat = self._compute_retreat_poses(
            T_grasp, T_aim2, T_aim, aim_joint, grasp_joint,
        )

        # Match the aim via-point's wrist to the (push-facing) wait pose so the
        # POSITIONING move keeps joint 6 put. request.aim_joint hard-sets
        # joint 6 to 0 (app.py _select_ambush_target), which would otherwise
        # make move_through swing the wrist ~90° to neutral and back every
        # cycle. Copy first so we don't mutate the request's array.
        aim_joint = aim_joint.copy()
        aim_joint[-1] = wait_joint[-1]

        # Re-enter point queue mode each cycle (same as ThrowSkill).
        if not ctx.traj_ctrl.enter_queue_mode():
            ctx.log.error("Failed to (re)enter queue mode; skipping this pick")
            return SkillResult(False, "enter_queue_mode (pick) failed")

        # ---- 1. POSITIONING: route current → aim hover → low push-start pose
        # (grasp_retreat) and park there. move_through_via PASSES THROUGH the
        # aim hover (vs move_through's direct cut), so the arm rises over before
        # descending — avoiding a belt-dipping path from the far bin-side
        # push_end. Finishing the descent BEFORE the wait means the object's
        # arrival leaves only the stroke to run, so the contact fires
        # immediately instead of waiting out a descent. NOTE: the arm now waits
        # at belt height near the intercept — make sure the incoming object
        # can't graze the parked TCP before the stroke (tune
        # PUSH_RETREAT_DISTANCE / PUSH_HEIGHT if it does). FIXED_DELAY_PUSH
        # should now be retuned DOWN (it no longer covers the descent — only
        # stroke + queue overhead). ----
        ctx.set_status("POSITIONING", target.class_name)
        ctx.move_through_via(current_joint, aim_joint, grasp_retreat_joint)

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

        # Per-class stroke distance (falls back to PUSH_DISTANCE).
        push_distance = PUSH_DISTANCE_MAP.get(target.class_name, PUSH_DISTANCE)

        # Chain: pre-position the NEXT object ONLY when it is a THROW pick.
        # scan_next_intercept returns (grasp_joint, _) and commits the object iff
        # the next routes to throw; a push/none next -> (None, None) (it does its
        # OWN push approach fresh next cycle). When throw-next, the stroke chains
        # straight to that grasp and the committed throw primes suction THERE next
        # cycle — so a throw after a push no longer cold-starts from push_end and
        # fires suction mid-transit ("suction at the floor"). push_time is a rough
        # stroke-duration estimate for the chain's feasibility gate.
        push_time = push_distance / PUSH_SPEED
        next_intercept_joint, _ = ctx.scan_next_intercept(grasp_retreat_joint, push_time)

        # Build & dispatch: STROKE ONLY (append_descent=False) — the arm already
        # descended to grasp_retreat during POSITIONING and waited there. Append a
        # chain to the next throw grasp when there is one (one queued trajectory,
        # same as ThrowSkill); else end at push_end and the next cycle approaches
        # fresh.
        self.build_push_trajectory(
            grasp_retreat_joint, grasp_retreat_joint, T_grasp_retreat, T_aim2, theta,
            next_intercept_joint=next_intercept_joint,
            append_chain=next_intercept_joint is not None,
            append_descent=False, push_distance=push_distance,
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
    # Retreat pose computation
    # ------------------------------------------------------------------
    def _compute_retreat_poses(
        self,
        T_grasp: np.ndarray,
        T_aim2: np.ndarray,
        T_aim: np.ndarray,
        aim_joint: np.ndarray,
        grasp_joint: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute retreat poses behind T_grasp opposite the push direction.

        Both poses share the same retreated XY but differ in Z:

        * **wait_joint** — aim height, for high hover during WAITING.
        * **grasp_retreat_joint** — grasp height minus HEIGHT_OFFSET,
          descent target & push stroke start.

        Returns ``(wait_joint, grasp_retreat_joint, T_grasp_retreat)``.
        Falls back to ``(aim_joint, grasp_joint, T_grasp)`` on IK failure.
        """
        ctx = self.ctx
        push_dir = self._compute_push_direction(T_grasp, T_aim2)
        dx = PUSH_RETREAT_DISTANCE * push_dir[0]
        dy = PUSH_RETREAT_DISTANCE * push_dir[1]

        # Clamp so X doesn't go below PUSH_RETREAT_MIN_X.
        if T_grasp[0, 3] - dx < PUSH_RETREAT_MIN_X and dx > 0:
            available = max(0.0, T_grasp[0, 3] - PUSH_RETREAT_MIN_X)
            if available <= 0:
                return aim_joint.copy(), grasp_joint.copy(), T_grasp.copy()
            scale = available / dx
            dx *= scale
            dy *= scale

        # Bake the push-facing orientation into both poses (replaces the old
        # joint-6 override). The grasp-retreat pose adopts the *stroke-start*
        # swing so the descent ends exactly where the push stroke begins; the
        # high hover stays neutral (swing=0).
        R_wait = self._push_orientation(push_dir, 0.0)
        R_grasp = self._push_orientation(push_dir, self._swing_at(0.0))

        # Wait pose: retreated XY, aim height
        T_wait = T_grasp.copy()
        T_wait[:3, :3] = R_wait
        T_wait[0, 3] -= dx
        T_wait[1, 3] -= dy
        T_wait[2, 3] = T_aim[2, 3]

        # Grasp retreat pose: retreated XY, grasp height - HEIGHT_OFFSET
        T_grasp_retreat = T_grasp.copy()
        T_grasp_retreat[:3, :3] = R_grasp
        T_grasp_retreat[0, 3] -= dx
        T_grasp_retreat[1, 3] -= dy
        # T_grasp_retreat[2, 3] -= HEIGHT_OFFSET
        T_grasp_retreat[2, 3] = PUSH_HEIGHT

        ik_wait = ctx.robot.inverse_kinematics(T_wait)
        ik_grasp = ctx.robot.inverse_kinematics(T_grasp_retreat)
        if ik_wait is None or ik_grasp is None:
            ctx.log.warn("Retreat IK failed; falling back to original poses")
            return aim_joint.copy(), grasp_joint.copy(), T_grasp.copy()

        # Full 6-DOF IK already realises the facing+swing — keep all joints.
        wait_joint = np.asarray(ik_wait, dtype=float)
        grasp_retreat_joint = np.asarray(ik_grasp, dtype=float)
        return wait_joint, grasp_retreat_joint, T_grasp_retreat

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
        # Face the TCP along the push direction (forward == PUSH_JOINT6_ANGLE).
        j6 = self._facing_joint6(self._compute_push_direction(T_grasp1, T_aim2))
        aim_joint1[-1] = j6
        grasp_joint1[-1] = j6
        aim_joint2[-1] = j6
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
        append_chain: bool = True,
        append_descent: bool = True,
        push_distance: float = PUSH_DISTANCE,
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
        zero6 = np.zeros(6)
        seg_traj: list = []
        seg_vel: list = []
        seg_ts: list = []

        # ================================================================
        # Segment 1: Descent  (aim → grasp, 6-DOF time-optimal) — OPTIONAL
        # ================================================================
        # Skipped (append_descent=False) when the arm has ALREADY descended to
        # the push-start pose during POSITIONING and is parked there waiting.
        # Then only the stroke remains, so the object's arrival fires the
        # contact immediately — the descent no longer eats into the
        # arrival-timing budget (FIXED_DELAY_PUSH).
        n_desc = 0
        desc_end_t = 0.0
        if append_descent:
            traj_desc, vel_desc, ts_desc = trajectory(
                aim_joint[:6], zero6,
                grasp_joint[:6], zero6,
                ctx.M1[:6], ctx.M2[:6], hertz=ctx.cfg.TRAJ_HZ,
            )
            seg_traj.append(traj_desc)
            seg_vel.append(vel_desc)
            seg_ts.append(ts_desc)
            n_desc = traj_desc.shape[1]
            desc_end_t = ts_desc[-1]

        # ================================================================
        # Segment 2: Push stroke  (straight line + progressive swing, 6-DOF)
        # ================================================================
        push_dir = self._compute_push_direction(T_grasp, T_aim2)
        traj_stroke, vel_stroke, ts_stroke = self._build_push_stroke(
            T_grasp, push_dir, grasp_joint, push_distance,
        )

        # Clamp segment velocities against robot joint limits (Yaskawa alarm
        # 4414 prevention). Same 2-pass rescale approach as throw_skill.
        traj_stroke, vel_stroke, ts_stroke = self._clamp_stroke_velocity(
            traj_stroke, vel_stroke, ts_stroke,
        )

        if append_descent:
            # Stroke follows the descent: shift its clock to start at the
            # descent end and drop its first sample (duplicate of descent's
            # last = grasp pose).
            ts_stroke = ts_stroke[1:] + desc_end_t
            traj_stroke = traj_stroke[:, 1:]
            vel_stroke = vel_stroke[:, 1:]
        # else: stroke IS the whole motion, starting at t=0 from the already-
        # parked push-start pose. Its first sample == the robot's current
        # position, which satisfies the queue code-204 check after
        # _build_queue_waypoints snaps positions[0] to current_joints.
        seg_traj.append(traj_stroke)
        seg_vel.append(vel_stroke)
        seg_ts.append(ts_stroke)
        n_stroke = traj_stroke.shape[1]

        # ================================================================
        # Segment 3: Chain  (push_end → next intercept or grasp) — OPTIONAL
        # ================================================================
        # With append_chain=False the dispatch ENDS at the hit (push_end): the
        # NEXT cycle's POSITIONING flows from push_end toward the next pick, so
        # we keep inter-pick flow WITHOUT a chain segment. MotoROS2 needs the
        # point queue to drain before queue-mode re-entry, so a chain can't run
        # asynchronously (it would block the next pick's dispatch and trip code
        # 2 'Must call start_point_queue_mode'). Ending at push_end removes that
        # blocking delay.
        push_end_q = traj_stroke[:, -1]         # last waypoint of stroke (6-DOF)
        push_end_dq = vel_stroke[:, -1]         # ~0 (boundary condition)

        if append_chain:
            chain_target = (
                np.asarray(next_intercept_joint, dtype=float)
                if next_intercept_joint is not None
                else np.asarray(aim_joint, dtype=float)
            )
            # Hold the stroke-end wrist angle through the chain (no whip).
            chain_target[5] = push_end_q[5]
            chained_to_next = next_intercept_joint is not None

            traj_chain, vel_chain, ts_chain = trajectory(
                push_end_q, push_end_dq,
                chain_target[:6], zero6,
                ctx.M1[:6], ctx.M2[:6], hertz=ctx.cfg.TRAJ_HZ,
            )
            # Drop chain's first sample (duplicate of stroke's last), shift ts.
            stroke_end_t = seg_ts[-1][-1] if len(seg_ts[-1]) > 0 else desc_end_t
            traj_chain = traj_chain[:, 1:]
            vel_chain = vel_chain[:, 1:]
            ts_chain_shifted = ts_chain[1:] + stroke_end_t

            seg_traj.append(traj_chain)
            seg_vel.append(vel_chain)
            seg_ts.append(ts_chain_shifted)
            final_joint = chain_target
        else:
            chained_to_next = False
            final_joint = push_end_q

        traj_push = np.concatenate(seg_traj, axis=1)
        vel_push = np.concatenate(seg_vel, axis=1)
        ts_full = np.concatenate(seg_ts)

        assert traj_push.shape[1] == vel_push.shape[1] == ts_full.shape[0], (
            f"push traj/vel/ts length mismatch: "
            f"{traj_push.shape[1]}/{vel_push.shape[1]}/{ts_full.shape[0]}"
        )
        ctx.log.info(
            f"Push traj: descent {n_desc} + stroke {n_stroke} steps "
            f"(d={push_distance:.3f}m @ {PUSH_SPEED:.2f}m/s, "
            f"θ={np.degrees(theta):.1f}°, swing "
            f"{np.degrees(SWING_BIAS - SWING_ANGLE):+.0f}°→{np.degrees(SWING_BIAS + SWING_ANGLE):+.0f}°), "
            f"{'chain→next intercept' if chained_to_next else 'end at push_end (no chain)'}"
        )

        # Thin to ≥ _MIN_QUEUE_GAP between points so the synchronous point-push
        # (~60-100 ms/point) keeps up with the robot's consumption. Without
        # this the queue drains mid-push and MotoROS2 exits queue mode →
        # code 2 'Must call start_point_queue_mode'. Essential here because a
        # fast PUSH_SPEED samples the stroke very tightly (e.g. 25 ms apart).
        traj_push, vel_push, ts_full = self._decimate_for_queue(
            traj_push, vel_push, ts_full,
        )

        # Dispatch. No timed release — push is contact-based, no suction.
        ctx.traj_ctrl.send_trajectory_queue(
            traj_push, vel_push, ts_full, final_joint=final_joint,
        )

        self._last_push_meta = {
            "n_descent": n_desc,
            "n_stroke": n_stroke,
            "push_distance": push_distance,
            "push_speed": PUSH_SPEED,
            "theta": theta,
            "swing": SWING_ANGLE,
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

    @staticmethod
    def _facing_joint6(push_dir: np.ndarray) -> float:
        """Joint-6 angle that points the TCP straight along ``push_dir``.

        In the default grasp orientation (``_R_GRASP_DEFAULT`` in app.py) the
        tool's facing axis is base **+X**, and the wrist is "facing forward"
        when joint 6 == ``PUSH_JOINT6_ANGLE`` (NOT 0). Rotating the push
        direction away from +X by ``yaw`` (its angle in the XY belt plane)
        therefore needs the same ``yaw`` added on top of ``PUSH_JOINT6_ANGLE``
        so the TCP keeps looking straight down the push line.

        When ``push_dir`` == +X, ``yaw`` == 0 → joint 6 == ``PUSH_JOINT6_ANGLE``
        (forward), as required.

        NOTE: flip the sign of ``yaw`` here if the wrist turns the *wrong* way
        on hardware — it depends on the joint-6 rotation axis direction.
        """
        yaw = float(np.arctan2(push_dir[1], push_dir[0]))
        return PUSH_JOINT6_ANGLE + yaw

    @staticmethod
    def _swing_at(alpha: float) -> float:
        """Swing tilt (rad) at stroke progress ``alpha`` ∈ [0, 1].

        Linear sweep from ``SWING_BIAS - SWING_ANGLE`` (leaning back) at the
        start to ``SWING_BIAS + SWING_ANGLE`` (leaning forward) at the end. The
        negative ``SWING_BIAS`` keeps the forward end small so the wrist never
        nears the joint-5 singularity.
        """
        return SWING_BIAS + SWING_ANGLE * (2.0 * alpha - 1.0)

    @staticmethod
    def _push_orientation(push_dir: np.ndarray, swing: float) -> np.ndarray:
        """Tool orientation (3x3) facing along ``push_dir``, tilted by ``swing``.

        Generalises ``_R_GRASP_DEFAULT`` (which faces base +X pointing straight
        down) to an arbitrary belt-plane heading plus a forward/back tilt:

          * ``swing`` == 0  → approach axis straight down (-Z), tool faces
            ``push_dir`` (identical to _R_GRASP_DEFAULT when push_dir == +X).
          * ``swing`` > 0   → approach axis leans *forward* (toward push_dir).
          * ``swing`` < 0   → approach axis leans *back* (away from push_dir).

        Base columns (tool axes in base frame) before the wrist twist::

            approach (tool X) = -cos(swing)*up + sin(swing)*f
            side     (tool Y) =  up × f
            facing   (tool Z) =  sin(swing)*up + cos(swing)*f

        where ``f`` is the unit push direction projected onto the belt plane
        and ``up`` is base +Z.

        Then the frame is **twisted by ``PUSH_JOINT6_ANGLE`` about the approach
        axis** (≈ the joint-6/flange axis for a down-pointing tool). The
        gripper's push-facing axis is the tool **Y** axis (not Z), 90° off the
        bare frame, so without this twist the IK solves joint 6 ~90° short of
        its intended baseline. After the twist the tool Y axis aligns with the
        push axis and sweeps back→front as ``swing`` varies — i.e. the swing
        shows up on "the TCP's Y direction" as intended.

        NOTE: flip the sign of ``PUSH_JOINT6_ANGLE`` if joint 6 ends up on the
        wrong side (or the tool Y faces the opposite way) on hardware.
        """
        f = np.array([push_dir[0], push_dir[1], 0.0], dtype=float)
        nf = np.linalg.norm(f)
        f = f / nf if nf > 1e-9 else np.array([1.0, 0.0, 0.0])
        up = np.array([0.0, 0.0, 1.0])
        side = np.cross(up, f)                 # horizontal, ⊥ push_dir
        c, s = np.cos(swing), np.sin(swing)
        approach = -c * up + s * f
        facing = s * up + c * f

        # Twist about the approach axis so joint 6 lands at its
        # PUSH_JOINT6_ANGLE baseline and the swing acts on the tool Y axis.
        cb, sb = np.cos(-PUSH_JOINT6_ANGLE), np.sin(-PUSH_JOINT6_ANGLE)
        side, facing = cb * side + sb * facing, -sb * side + cb * facing
        return np.column_stack((approach, side, facing))

    def _build_push_stroke(
        self,
        T_grasp: np.ndarray,
        direction: np.ndarray,
        grasp_joint: np.ndarray,
        push_distance: float = PUSH_DISTANCE,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Constant-speed straight-line push with a progressive swing.

        The TCP *position* travels a straight belt-parallel line:

        * Start: ``T_grasp`` (at belt height)
        * End:   ``T_grasp + push_distance * direction`` (same Z)

        while the *orientation* sweeps via :meth:`_push_orientation` from
        ``_swing_at(0)`` (leaning back) to ``_swing_at(1)`` (leaning forward).

        Each waypoint is solved with full 6-DOF IK so the swing — and the
        push-facing yaw — are realised by the arm/wrist; **all six joints are
        kept**. If IK fails mid-stroke, the trajectory is truncated.

        Joint velocities are computed via central finite differences with
        boundary velocities forced to zero so the stroke concatenates smoothly
        with the descent (v=0 at end) and chain (v=0 at start) segments.

        Returns ``(traj_6, vel_6, ts)`` with shapes ``(6, n), (6, n), (n,)``.
        """
        ctx = self.ctx
        total_time = push_distance / max(PUSH_SPEED, 1e-6)
        n_steps = max(2, int(total_time * ctx.cfg.TRAJ_HZ))
        dt = total_time / n_steps

        # ---- Cartesian position + swing orientation → joint space via IK ----
        # Seed each IK with the previous waypoint so the analytical solver
        # keeps the SAME wrist branch across the stroke. Without this, when
        # joint 5 crosses 0 (wrist singularity) near the swing end the two
        # Euler solutions cross over and the unseeded min|θ4| pick flips
        # joints 4/6 by ~π — a sudden joint-6 jump that the chain then holds.
        waypoints: list[np.ndarray] = []
        q_seed = np.asarray(grasp_joint, dtype=float)
        for i in range(n_steps + 1):
            alpha = i / n_steps
            T_wp = T_grasp.copy()
            T_wp[:3, :3] = self._push_orientation(direction, self._swing_at(alpha))
            T_wp[0, 3] += alpha * push_distance * direction[0]
            T_wp[1, 3] += alpha * push_distance * direction[1]
            T_wp[2, 3] = PUSH_HEIGHT
            # Z unchanged — belt-parallel motion.
            ik = ctx.robot.inverse_kinematics(T_wp, q_init=q_seed)
            if ik is None:
                ctx.log.warn(
                    f"Push stroke IK failed at step {i}/{n_steps}; "
                    f"truncating stroke to {len(waypoints)} waypoints"
                )
                break
            q_seed = np.asarray(ik, dtype=float)
            waypoints.append(q_seed)   # full 6-DOF

        # Fallback: if fewer than 2 waypoints, return a zero-motion segment
        # so the caller can still concatenate without crashing.
        if len(waypoints) < 2:
            ctx.log.warn("Push stroke degenerate (< 2 IK solutions); no-op segment")
            q0 = np.asarray(grasp_joint, dtype=float)
            traj = np.column_stack([q0, q0])
            vel = np.zeros_like(traj)
            ts = np.array([0.0, dt])
            return traj, vel, ts

        n = len(waypoints)
        traj = np.column_stack(waypoints)              # (6, n)
        ts = np.linspace(0.0, dt * (n - 1), n)        # (n,)

        # ---- Joint velocities via central finite differences ----
        vel = np.zeros_like(traj)                      # (6, n)
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
        # Match the trajectory's DOF (stroke is now full 6-DOF).
        m1 = np.asarray(ctx.M1[: traj.shape[0]], dtype=float)

        for _ in range(2):
            dt_seg = np.maximum(np.diff(ts), 1e-9)
            seg_vel = np.abs(np.diff(traj, axis=1)) / dt_seg[None, :]  # (DOF, n-1)
            ratio = float(np.max(seg_vel / m1[:, None]))
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
            "swing_deg": round(np.degrees(meta.get("swing", 0.0)), 1),
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

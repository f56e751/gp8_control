"""Push skill: position at the intercept, wait for the object, and push it off the belt.

Mirrors the full structure of ``ThrowSkill`` — ambush at the intercept, wait
for the object, then execute a contact push.  Key differences from throw:

  * **No suction**: the gripper/TCP physically pushes the object.
  * **Rule-based push stroke**: a constant-speed Cartesian straight line on the
    belt plane (constant Z), NOT an NN-generated arc.

Dispatched trajectory segments (concatenated into one dispatch):
  1. **Push stroke** (push-start → push_end): Cartesian interpolation at
     ``PUSH_SPEED`` m/s for ``PUSH_DISTANCE`` m, parallel to the belt
     surface.  Direction = T_contact → push-bin target projected onto XY.
  2. **Chain** (push_end → next intercept): time-optimal transition so the
     arm flows to the next pick cycle.
The descent to the push-start pose happens earlier, during POSITIONING
(``move_through_via``), not in this dispatch.

Shared primitives live on ``self.ctx`` (a ``SkillContext``):
  * ``ctx.robot`` (FK/IK), ``ctx.traj_ctrl``, ``ctx.M1``/``ctx.M2``, ``ctx.cfg``
  * ``ctx.move_through`` / ``ctx.wait_for_arrival``
  * ``ctx.set_status(status, detail)``, ``ctx.set_active_target(None)``
"""

from __future__ import annotations

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
# Push policy (mirrors throw_skill's THROW_BIN_TARGET_MAP)
# =========================================================================

# Classes this skill ACCEPTS (PushSkill.can_handle). The ActionSelector routes
# by app Config.SKILL_BY_CLASS; this set is the skill's own guard so it refuses
# anything it shouldn't handle (selector then falls back to throw). Cans
# ("metal") are pushed; PET bottles ("transparent") are suctioned/thrown, so
# they are intentionally NOT in this set.
PUSH_CLASSES: set[str] = {
    "metal",
}

# NOTE: the old per-class PUSH_THETA_MAP is gone — the push direction is derived
# geometrically as ``push_dir`` (XY heading from the contact toward the fixed
# push-bin target, PUSH_BIN_TARGET_MAP). ``theta`` survives only as its logged
# angle (telemetry), computed from ``push_dir`` in ``build_push_trajectory``.

# Per-class push bin TARGET (absolute base-frame XYZ, m). The push aims at this
# fixed bin/chute location for the object's class. A class NOT in this map has
# no push destination and is aborted in ``execute`` — only reachable via
# GP8_FORCE_SKILL on an unmapped class (normal routing gates to PUSH_CLASSES,
# all of which are mapped here).
PUSH_BIN_TARGET_MAP: dict[str, tuple] = {
    "transparent": (1.2, -0.30, 0.0),
    "metal":       (1.2,  0.50, 0.0),
}



# ---- Push stroke parameters ------------------------------------------------

# TCP speed during the push stroke (m/s). The arm sweeps at this speed
# parallel to the belt surface. Tune to balance impact force vs. control
# stability; too fast may exceed joint velocity limits.
PUSH_SPEED: float = 2.0

# Push stroke distance (m). How far the TCP travels from the push-start in the
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

# Retreat distance (m) for the push-start pose. The TCP waits on the line
# from the push target through the intercept point (T_contact), but offset
# PUSH_RETREAT_DISTANCE behind T_contact — i.e. in the direction *opposite*
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

# Push arrival-lead (s): how far BEFORE the object's predicted arrival to end the WAITING
# block, so positioning + the stroke's retreat->contact pre-travel land the stroke ON the
# object (push has no suction to forgive an early/late hit). Was 0.8 on MotoROS2, which
# INCLUDED the ~0.4 s point-queue re-entry per move. On the adv4ncr 250 Hz stream that
# re-entry is GONE, so 0.8 fires the stroke ~0.4 s EARLY -> it misses ahead of the object.
# Default dropped to 0.4; HW-tune via GP8_FIXED_DELAY_PUSH at low speed (RAISE if the stroke
# trails the object, LOWER if it still leads).
FIXED_DELAY_PUSH = float(os.environ.get("GP8_FIXED_DELAY_PUSH", "0.4"))
# ABSOLUTE base-frame Z (m) of the push stroke — assigned directly (not a
# relative offset) to the push-start / stroke waypoints' [2,3]. It MUST sit at
# ~belt surface (GRASP_Z = 0.062), NOT below it: the
# prior 0.01 put the TCP ~5 cm UNDER the belt and drove the arm into it on the first
# real metal push (alarm 4315 / STATE 101 CODE 112). Also balanced against the joint-5
# wrist singularity — 0.06 clears it as long as the forward swing end stays ≲ +5°
# (see SWING_BIAS). HW-calibrate against the measured belt height before fast runs.
# 0.07: raised from 0.06, which sat ~2 mm UNDER belt GRASP_Z=0.062 and grazed the
# belt. Keep the forward swing end ≲ +5° (SWING_BIAS) so it still clears joint-5.
PUSH_HEIGHT = 0.07

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

# Swing mode (env-selectable at launch, like FIXED_DELAY_PUSH). Two contact
# styles for the stroke orientation:
#   * "scoop" — progressive swing: the tool leans back at contact and rolls
#     forward through the stroke (SWING_BIAS ∓ SWING_ANGLE). Scoop-then-flick;
#     contains the object at contact and can shape the release arc, at the cost
#     of an orientation sweep that risks the joint-5 wrist singularity.
#   * "sweep" — fixed tilt (SWEEP_TILT) held for the whole stroke. No orientation
#     sweep, so it is predictable and singularity-safe; loses the scoop.
# Set GP8_PUSH_SWING_MODE=sweep to switch without editing code.
PUSH_SWING_MODE: str = os.environ.get("GP8_PUSH_SWING_MODE", "scoop").strip().lower()

# Fixed tilt (rad) held across the stroke in "sweep" mode — a small forward lean
# (scooping). Kept small so the wrist stays clear of the joint-5 singularity.
# Unused in "scoop" mode.
SWEEP_TILT: float = np.radians(5.0)


class PushSkill(ManipulationSkill):
    """Position at the intercept, wait for the object, and push it off the belt.

    ``execute`` runs the full push cycle — identical flow to ``ThrowSkill``:

      1. **POSITIONING** — ``move_through_via(current, aim, push_start)`` to drive
         to the push-start pose and park there.
      2. **WAITING** — ``wait_for_arrival`` blocks until the object arrives. (Push
         is contact-based, so no suction is fired; any leftover vacuum is cleared.)
      3. **PUSHING** — compute push target, dispatch the push stroke + chain
         trajectory. (adv4ncr 250Hz stream: no queue mode to re-enter.)

    Public planning/build methods mirror ``ThrowSkill`` so external code can
    reuse push logic.
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

    def arrival_lead(self) -> float:
        """End the WAITING block this many seconds before the object arrives.

        Push has NO suction to forgive a late hit (unlike throw, whose object is
        already cup-held), so this lead must equal the FULL post-wait latency —
        scan + dispatch + the stroke's retreat->contact pre-travel — NOT throw's
        small forgiving base (cfg.ACTION_START_LEAD). Empirically ~0.8 s on hardware
        (feature/push, which timed correctly); the merge cut it to throw's 0.2 s and
        push started hitting behind the object. That 0.8 ALREADY includes the retreat
        pre-travel, so it is NOT composed on top of the base (no double-count).
        (adv4ncr stream: the old ~0.4 s point-queue re-entry that 0.8 covered is GONE,
        so the default dropped to 0.4 — env-tune GP8_FIXED_DELAY_PUSH at low speed:
        RAISE if the hit trails the object, LOWER if it still leads.)
        """
        return FIXED_DELAY_PUSH

    def t_to_contact(self, move_time: float) -> float:
        """Honest push timeline (overrides the base legacy heuristic).

        Push contact is an ACTIVE, timed sweep with NO suction forgiveness, so the
        intercept MUST be placed where the object will be at the REAL strike time,
        not where the bare positioning estimate lands. The real budget from "arm
        starts moving" to "stroke contacts the object" is::

            T_setup#1 (dispatch before POSITIONING)
          + T_position (move_through_via: rise to aim hover + descend to retreat)
          + T_setup#2 (dispatch before the stroke)
          + T_contact_offset (stroke travels push_start -> contact line)

        The dispatch overhead and the retreat->contact pre-travel are exactly what
        the old ``move_time * factor`` omitted — why the arm aimed upstream of where
        the can actually was and struck the next object. On the adv4ncr 250Hz stream
        driver the old ~0.4 s point-queue re-entry is gone, so T_setup is now just the
        per-dispatch overhead (``qmode_ms_avg``, ~tens of ms) — HW-calibrate; the 2x
        conservatively budgets the positioning + stroke dispatches. T_position is
        opt_time scaled by OPT_TIME_TO_REAL; T_contact_offset = PUSH_RETREAT_DISTANCE / PUSH_SPEED.
        """
        ctx = self.ctx
        t_setup = ctx.traj_ctrl.qmode_ms_avg / 1000.0          # per-dispatch overhead (stream)
        t_position = move_time * ctx.cfg.OPT_TIME_TO_REAL
        t_pre_travel = PUSH_RETREAT_DISTANCE / max(PUSH_SPEED, 1e-6)
        return 2.0 * t_setup + t_position + t_pre_travel        # dispatch(pos) + dispatch(stroke)

    # ------------------------------------------------------------------
    # Skill entry point (ambush strategy)
    # ------------------------------------------------------------------
    def execute(self, request: "PickRequest") -> SkillResult:
        """Pre-position (with retreat), wait for arrival, then push.

        Follows the same flow as ``ThrowSkill.execute`` (adv4ncr 250Hz stream —
        no point-queue mode to (re)enter):

        1. ``move_through_via(current, aim, push_start)`` (POSITIONING).
           ``push_start`` is the retreat pose — offset behind T_contact opposite
           to the push direction, at belt height — so the arm parks low and ready
           and the object's arrival leaves only the stroke to run.
        2. ``wait_for_arrival`` (WAITING) — blocks until the object reaches the
           intercept line.
        3. Dispatch push stroke + chain trajectory (PUSHING).
        4. Cleanup.
        """
        ctx = self.ctx
        target = request.target
        current_joint = request.current_joint
        aim_joint = request.aim_joint
        contact_joint = request.grasp_joint
        T_contact = request.T_grasp

        # Push does NOT use suction — ensure it's off from any prior cycle.
        ctx.traj_ctrl.suction_off()

        # ---- Compute push target (fixed bin) early ----
        # We need the push direction *before* positioning so we can place the
        # wait pose on the line behind T_contact, opposite to the push. Push
        # sorts to a fixed per-class bin; a class with no bin has nowhere to be
        # pushed, so abort. (Only reachable via GP8_FORCE_SKILL on an unmapped
        # class — normal routing gates to PUSH_CLASSES, all of which are mapped.)
        bin_xyz = PUSH_BIN_TARGET_MAP.get(target.class_name)
        if bin_xyz is None:
            # TODO(push): revisit the handling for an unmapped class. This can be
            # a perception mislabel (unknown/garbage class) rather than a genuine
            # "no bin", so a benign PASS/skip may fit better than this hard abort
            # (self._abort returns SkillResult(False) -> counts as a FAILURE).
            # For now abort cleanly — only reachable via GP8_FORCE_SKILL on an
            # unmapped class, so it doesn't affect normal metal routing.
            ctx.log.warn(f"no push bin for {target.class_name}; abort")
            return self._abort(f"no push bin mapping for {target.class_name}")
        push_target_xyz = np.asarray(bin_xyz, dtype=float)   # (3,) base-frame XYZ
        ctx.log.info(
            f"Push target for {target.class_name}: fixed bin "
            f"({bin_xyz[0]:+.3f}, {bin_xyz[1]:+.3f}, {bin_xyz[2]:+.3f}) m"
        )

        # ---- Compute the push-start pose (retreated, at belt height) ----
        push_start_joint, T_push_start = self._compute_retreat_poses(
            T_contact, push_target_xyz, contact_joint,
        )

        # Match the aim via-point's wrist to the (push-facing) push-start pose so
        # the POSITIONING move keeps joint 6 put. aim_joint's wrist is hard-set to
        # 0 (context.py Intercept build), which would otherwise make the wrist
        # swing ~90° to neutral and back every cycle. Copy first so we don't
        # mutate the request's array.
        aim_joint = aim_joint.copy()
        aim_joint[-1] = push_start_joint[-1]

        # ---- 1. POSITIONING: route current → aim hover → low push-start pose
        # (push_start) and park there. move_through_via PASSES THROUGH the
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
        ctx.move_through_via(current_joint, aim_joint, push_start_joint)

        # ---- 2. WAITING: block until the object arrives ----
        # End the wait arrival_lead() s before arrival. Push has no suction to
        # forgive a late hit, so the lead must equal the FULL post-wait latency
        # (scan + dispatch + stroke retreat->contact pre-travel) — ~0.8 s
        # (FIXED_DELAY_PUSH, HW-recalibrate for stream), NOT throw's small forgiving
        # base. See PushSkill.arrival_lead().
        ctx.set_status("WAITING", target.class_name)
        ctx.wait_for_arrival(target, T_contact[1, 3], offset=self.arrival_lead())

        # ---- 2b. STALE-STROKE GUARD ----
        # If timing slipped and the object has already passed the ENTIRE stroke span,
        # do NOT fire: the stroke would sweep into the NEXT object (the observed
        # "pushed the next PET"). The stroke runs push_start -> push_end; the only
        # belt-Y range it can still contact is between those two ends. The push
        # heading's Y sign is geometry-dependent (for an intercept UPSTREAM of the bin
        # the stroke actually sweeps downstream-in-Y), so compare against the
        # most-DOWNSTREAM (smallest-Y) stroke end, not just push_start. Drop
        # cleanly (same cleanup as the queue-fail path). The timeline fix (skill-aware
        # t_to_contact) should make this rare; this is the hard safety net.
        push_dir_guard = self._compute_push_direction(T_push_start, push_target_xyz)
        push_distance_guard = PUSH_DISTANCE_MAP.get(target.class_name, PUSH_DISTANCE)
        stroke_end_y = T_push_start[1, 3] + push_distance_guard * push_dir_guard[1]
        stroke_min_y = min(float(T_push_start[1, 3]), float(stroke_end_y))
        obj_y_now = ctx.object_y_now(target, time.time(), ctx.conveyor.current)
        if obj_y_now < stroke_min_y:
            ctx.log.warn(
                f"Push abort: id={target.track_id} {target.class_name} already past the "
                f"stroke (y={obj_y_now:+.3f} < stroke_min {stroke_min_y:+.3f}); "
                f"skipping stale stroke"
            )
            return self._abort("object passed stroke span; push aborted")

        # ---- 3. PUSHING: dispatch push traj ----
        # DIAGNOSTIC: object vs intercept at the instant the stroke fires. delta < 0
        # means the object already passed the grasp/contact point and the stroke
        # lands behind it (and can sweep into the next object). The stale-stroke
        # guard above aborts the worst case; this logs every non-aborted stroke so
        # borderline "behind" hits are visible too. See log_action_timing.
        ctx.log_action_timing(target, T_contact[1, 3], "push-stroke")
        ctx.set_status("PUSHING", target.class_name)

        # Per-class stroke distance (falls back to PUSH_DISTANCE).
        push_distance = PUSH_DISTANCE_MAP.get(target.class_name, PUSH_DISTANCE)

        # Build & dispatch: STROKE ONLY — the arm already descended to push-start
        # during POSITIONING and waited there. Chain the follow-through toward the
        # NEXT object's pre-position (push -> its push-start, throw -> its grasp;
        # see _next_chain_park), else a lifted-standby park. Best-effort overlap
        # (pre_delay = the stroke duration the arm is busy first). Symmetric +
        # stateless: the next epoch selects + drives fresh from this closer pose.
        push_time = push_distance / max(PUSH_SPEED, 1e-6)
        next_grasp = self._next_chain_park(push_start_joint, push_time)
        self.build_push_trajectory(
            push_start_joint, T_push_start, push_target_xyz,
            next_grasp=next_grasp,
            push_distance=push_distance,
        )

        # ---- 4. Cleanup ----
        # Safety: turn suction off in case a prior action left it on
        # (push is contact-based, no suction needed).
        ctx.traj_ctrl.suction_off()
        self._log_push_cycle(target)
        ctx.set_active_target(None)
        ctx.set_status("IDLE", "")
        return SkillResult(True, "push complete")

    # ------------------------------------------------------------------
    # Chain park target (push-aware; throw / context untouched)
    # ------------------------------------------------------------------
    def _next_chain_park(self, from_joint: np.ndarray, action_time: float):
        """Pose the follow-through chain should park OVER, matched to the NEXT
        object's skill — a push-aware replacement for ``ctx.next_chain_target``.

        ``ctx.next_chain_target`` always returns the next object's GRASP (contact)
        joints. That is right for a throw-next (it grabs AT the grasp) but WRONG
        for a push-next, which must pre-position at the RETREATED push-start
        (``PUSH_RETREAT_DISTANCE`` behind contact), NOT over the contact. Parking
        over the contact forces the next push cycle's POSITIONING to back up to
        push-start — a mistimed correction (the 2nd-push-onward degradation).

        Replicates ``next_chain_target``'s selection (queue scan +
        ``earliest_reachable_intercept``) READ-ONLY, but when the next object
        routes to PUSH returns its push-start joints (via
        ``_compute_retreat_poses``) instead of its grasp. Throw-next — or a
        push-next with no bin, or retreat IK failure — falls back to the grasp
        joints, byte-identical to ``next_chain_target``. Kept push-local so
        ``throw_skill`` / ``context`` stay untouched.
        """
        ctx = self.ctx
        now = time.time()
        v = ctx.conveyor.current if ctx.conveyor is not None else 0.0
        for cand in list(ctx.queue._objects):
            skill = ctx.skill_obj_for(cand)
            it = ctx.earliest_reachable_intercept(
                cand, from_joint, v, now, pre_delay=action_time,
                t_to_contact_fn=skill.t_to_contact,
            )
            if it is None:
                continue
            bin_xyz = PUSH_BIN_TARGET_MAP.get(cand.class_name)
            if skill.name == self.name and bin_xyz is not None:
                # Next object is a PUSH -> park over its push-START (retreated),
                # not its contact. Reuse this skill's retreat math on the next
                # intercept; _compute_retreat_poses degrades to the grasp joints
                # on IK failure, so the fallback stays safe.
                next_push_start, _ = self._compute_retreat_poses(
                    it.T_grasp, np.asarray(bin_xyz, dtype=float), it.grasp_joint,
                )
                ctx.log.info(
                    f"Chain toward next PUSH: id={cand.track_id} "
                    f"{cand.class_name} @ y={it.intercept_y:+.3f} -> over push-start"
                )
                return next_push_start
            ctx.log.info(
                f"Chain toward next {skill.name}: id={cand.track_id} "
                f"{cand.class_name} @ y={it.intercept_y:+.3f} -> over grasp"
            )
            return it.grasp_joint
        return None

    # ------------------------------------------------------------------
    # Retreat pose computation
    # ------------------------------------------------------------------
    def _compute_retreat_poses(
        self,
        T_contact: np.ndarray,
        push_target_xyz: np.ndarray,
        contact_joint: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the push-start pose behind T_contact, opposite the push direction.

        The arm parks at this retreated, belt-height pose during POSITIONING and
        waits there (see ``execute``): the retreat gives it runway to accelerate
        into the stroke, and the pose already carries the stroke-start swing so
        the descent ends exactly where the push stroke begins.

        Returns ``(push_start_joint, T_push_start)``. Falls back to
        ``(contact_joint, T_contact)`` when the retreat has no room or IK fails.
        """
        ctx = self.ctx
        push_dir = self._compute_push_direction(T_contact, push_target_xyz)
        dx = PUSH_RETREAT_DISTANCE * push_dir[0]
        dy = PUSH_RETREAT_DISTANCE * push_dir[1]

        # Clamp so X doesn't go below PUSH_RETREAT_MIN_X.
        if T_contact[0, 3] - dx < PUSH_RETREAT_MIN_X and dx > 0:
            available = max(0.0, T_contact[0, 3] - PUSH_RETREAT_MIN_X)
            if available <= 0:
                return contact_joint.copy(), T_contact.copy()
            scale = available / dx
            dx *= scale
            dy *= scale

        # Bake the stroke-start swing + push-facing orientation into the pose
        # (replaces the old joint-6 override) so the descent ends exactly where
        # the push stroke begins.
        R_grasp = self._push_orientation(push_dir, self._swing_at(0.0))

        # Push-start pose: retreated XY, absolute PUSH_HEIGHT (≈ belt surface)
        T_push_start = T_contact.copy()
        T_push_start[:3, :3] = R_grasp
        T_push_start[0, 3] -= dx
        T_push_start[1, 3] -= dy
        T_push_start[2, 3] = PUSH_HEIGHT

        ik_grasp = ctx.robot.inverse_kinematics(T_push_start)
        if ik_grasp is None:
            ctx.log.warn("Retreat IK failed; falling back to original pose")
            return contact_joint.copy(), T_contact.copy()

        # Full 6-DOF IK already realises the facing+swing — keep all joints.
        push_start_joint = np.asarray(ik_grasp, dtype=float)
        return push_start_joint, T_push_start

    # ------------------------------------------------------------------
    # Push trajectory build + dispatch
    # ------------------------------------------------------------------
    def build_push_trajectory(
        self,
        push_start_joint: np.ndarray,
        T_push_start: np.ndarray,
        push_target_xyz: np.ndarray,
        next_grasp: "Optional[np.ndarray]" = None,
        push_distance: float = PUSH_DISTANCE,
    ) -> None:
        """Build and dispatch the 2-segment push trajectory.

        The arm has ALREADY descended to the push-start pose during
        POSITIONING and waited there, so this builds only:

        1. **Push stroke** (push-start → push_end): rule-based Cartesian straight
           line at ``PUSH_SPEED`` m/s for ``PUSH_DISTANCE`` m, parallel to
           the belt surface (constant Z). Direction is from the push-start towards
           push_target_xyz, projected onto the XY plane. Each Cartesian waypoint is
           converted to joint space via IK. The stroke STARTS at rest (the arm was
           parked at push-start waiting); its EXIT velocity is carried into the
           chain (continuized — see the build code) so there is no full stop at
           push_end.

        2. **Chain** (push_end → next_intercept or grasp): time-optimal
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
        # Segment 1: Push stroke  (straight line + progressive swing, 6-DOF)
        # ================================================================
        # The arm ALREADY descended to the push-start pose during POSITIONING
        # and is parked there waiting, so the stroke IS the whole motion — the
        # object's arrival fires the contact immediately (the descent no longer
        # eats into the arrival-timing budget, FIXED_DELAY_PUSH).
        push_dir = self._compute_push_direction(T_push_start, push_target_xyz)
        # Push heading angle — telemetry only (log + CSV theta_deg); the stroke
        # uses push_dir directly. Derived from the real push_dir so the logged
        # angle matches the actual stroke.
        theta = float(np.arctan2(push_dir[1], push_dir[0]))
        traj_stroke, vel_stroke, ts_stroke = self._build_push_stroke(
            T_push_start, push_dir, push_start_joint, push_distance,
        )

        # Clamp segment velocities against robot joint limits (Yaskawa alarm
        # 4414 prevention). Same 2-pass rescale approach as throw_skill.
        traj_stroke, vel_stroke, ts_stroke = self._clamp_stroke_velocity(
            traj_stroke, vel_stroke, ts_stroke,
        )

        # Stroke IS the whole motion, starting at t=0 from the already-parked
        # push-start pose (its first sample == the robot's current position).
        # (The old MotoROS2 code-204 "first point == measured position" check is
        # gone on the stream driver.)
        seg_traj.append(traj_stroke)
        seg_vel.append(vel_stroke)
        seg_ts.append(ts_stroke)
        n_stroke = traj_stroke.shape[1]

        # ================================================================
        # Segment 2: Chain  (push_end → next intercept or idle pose)
        # ================================================================
        # Always chains: to the next pick's pre-position when there is one, else
        # to the shared idle/standby pose — so the arm parks HIGH instead of low
        # at push_end (a low bin-side park would make the next cycle's POSITIONING
        # sweep the TCP across the belt).
        push_end_q = traj_stroke[:, -1]         # last waypoint of stroke (6-DOF)
        # CONTINUIZE stroke -> chain: feed the stroke's natural EXIT velocity into the
        # chain's START (push_end_dq) instead of starting the chain from REST. The
        # chain trajectory then begins at full speed, so its POSITIONS flow
        # continuously out of the stroke rather than accelerating from a standstill —
        # which is what removes the visible "push, then stop, rotate + move" pause and
        # its overrun beyond the planned timestamps (throw stays ≈planned because its
        # arc is continuous; push overran ~+0.13s from this stop). Mirrors the throw's
        # release-velocity carry (build_throw_trajectory). Clipped to the joint-vel
        # limits (Yaskawa alarm 4414 safety). With decimation removed, these
        # velocities pass straight to the stream's cubic-Hermite resampler
        # (positions + velocities), so the carried exit velocity now shapes the
        # stroke->chain transition directly.
        if traj_stroke.shape[1] >= 2:
            _dt_end = max(float(ts_stroke[-1] - ts_stroke[-2]), 1e-9)
            push_end_dq = np.clip(
                (traj_stroke[:, -1] - traj_stroke[:, -2]) / _dt_end,
                -ctx.M1[:6], ctx.M1[:6],
            )
            vel_stroke[:, -1] = push_end_dq     # queued stroke now exits at speed
        else:
            push_end_dq = vel_stroke[:, -1]

        # Chain target (3-way): the next pick's pre-position if committed; else, when
        # the queue is EMPTY (no next object detected yet, ①a), the full
        # home/standby pose; else (a next object exists but was NOT committed —
        # different skill or unreachable after the push, ①b/②) the lifted
        # standby — push_end raised to home Z, so the arm clears the belt without
        # the wasted home round-trip. copy() so ctx.idle_joint is never mutated
        # by the wrist write below.
        if next_grasp is not None:
            # Park OVER the next pick's pre-position (raised to home Z), NOT at belt
            # height — a belt-height chain from push_end (bin side) would sweep the
            # TCP low across the belt (alarm 4315 / grazing objects). Raise Z; descend next epoch.
            chain_target = ctx.lifted_standby_joint(next_grasp)
            chain_dest = "over next pre-position"
        elif not ctx.queue:
            chain_target = self.idle_target().copy()
            chain_dest = "home/idle (queue empty)"
        else:
            chain_target = ctx.lifted_standby_joint(push_end_q)
            chain_dest = "lifted standby"
        # Wrist (joint 6) at the chain end = 0 (the standby/idle pose already has
        # joint 6 = 0; parking it elsewhere would flip joint 6 -> Yaskawa alarm 4414).
        chain_target[5] = 0.0

        traj_chain, vel_chain, ts_chain = trajectory(
            push_end_q, push_end_dq,
            chain_target[:6], zero6,
            ctx.M1[:6], ctx.M2[:6], hertz=ctx.cfg.TRAJ_HZ,
        )
        # Drop chain's first sample (duplicate of stroke's last), shift ts.
        stroke_end_t = seg_ts[-1][-1] if len(seg_ts[-1]) > 0 else 0.0
        traj_chain = traj_chain[:, 1:]
        vel_chain = vel_chain[:, 1:]
        ts_chain_shifted = ts_chain[1:] + stroke_end_t

        seg_traj.append(traj_chain)
        seg_vel.append(vel_chain)
        seg_ts.append(ts_chain_shifted)
        final_joint = chain_target

        traj_push = np.concatenate(seg_traj, axis=1)
        vel_push = np.concatenate(seg_vel, axis=1)
        ts_full = np.concatenate(seg_ts)

        assert traj_push.shape[1] == vel_push.shape[1] == ts_full.shape[0], (
            f"push traj/vel/ts length mismatch: "
            f"{traj_push.shape[1]}/{vel_push.shape[1]}/{ts_full.shape[0]}"
        )
        ctx.log.info(
            f"Push traj: stroke {n_stroke} steps "
            f"(d={push_distance:.3f}m @ {PUSH_SPEED:.2f}m/s, "
            f"θ={np.degrees(theta):.1f}°, {PUSH_SWING_MODE} swing "
            f"{np.degrees(self._swing_at(0.0)):+.0f}°→{np.degrees(self._swing_at(1.0)):+.0f}°), "
            f"chain→{chain_dest}"
        )

        # NO decimation: the 250 Hz stream driver resamples to a 4 ms grid, so the
        # full stroke+chain knots are dispatched as-is (the swing knots survive).
        # (Decimation was a MotoROS2 point-queue workaround; there is no queue here,
        # and on the ~0.175 s stroke it collapsed the swing to just its endpoints.)

        # Dispatch. No timed release — push is contact-based, no suction.
        ctx.traj_ctrl.send_trajectory_queue(
            traj_push, vel_push, ts_full, final_joint=final_joint,
        )

        self._last_push_meta = {
            "n_stroke": n_stroke,
            "push_distance": push_distance,
            "push_speed": PUSH_SPEED,
            "theta": theta,
            "swing_mode": PUSH_SWING_MODE,
            "swing": self._swing_at(1.0),   # end-of-stroke tilt (both modes)
            "chain_dest": chain_dest,
        }

    # ------------------------------------------------------------------
    # Push stroke helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_push_direction(
        T_contact: np.ndarray,
        push_target_xyz: np.ndarray,
    ) -> np.ndarray:
        """Unit direction from T_contact to push_target_xyz projected onto the XY belt plane.

        ``push_target_xyz`` is a base-frame (3,) position. Returns a 3D unit
        vector with Z=0. Falls back to +X if T_contact and the target coincide
        in XY.
        """
        delta = np.asarray(push_target_xyz, dtype=float) - T_contact[:3, 3]
        delta[2] = 0.0                         # project onto belt plane
        norm = np.linalg.norm(delta)
        if norm < 1e-6:
            # Degenerate case: push along +X as a safe default.
            return np.array([1.0, 0.0, 0.0])
        return delta / norm

    @staticmethod
    def _swing_at(alpha: float) -> float:
        """Swing tilt (rad) at stroke progress ``alpha`` ∈ [0, 1].

        Behaviour depends on ``PUSH_SWING_MODE``:

        * ``"scoop"`` — progressive sweep from ``SWING_BIAS - SWING_ANGLE``
          (leaning back) at the start to ``SWING_BIAS + SWING_ANGLE`` (leaning
          forward) at the end. The negative ``SWING_BIAS`` keeps the forward end
          small so the wrist never nears the joint-5 singularity.
        * ``"sweep"`` — fixed ``SWEEP_TILT`` held for the whole stroke
          (alpha-independent); no orientation sweep.
        """
        if PUSH_SWING_MODE == "sweep":
            return SWEEP_TILT
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
        T_push_start: np.ndarray,
        direction: np.ndarray,
        push_start_joint: np.ndarray,
        push_distance: float = PUSH_DISTANCE,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Constant-speed straight-line push with a progressive swing.

        The TCP *position* travels a straight belt-parallel line:

        * Start: ``T_push_start`` (at belt height)
        * End:   ``T_push_start + push_distance * direction`` (same Z)

        while the *orientation* sweeps via :meth:`_push_orientation` from
        ``_swing_at(0)`` (leaning back) to ``_swing_at(1)`` (leaning forward).

        Each waypoint is solved with full 6-DOF IK so the swing — and the
        push-facing yaw — are realised by the arm/wrist; **all six joints are
        kept**. If IK fails mid-stroke, the trajectory is truncated.

        Joint velocities are computed via central finite differences with
        boundary velocities forced to zero so the stroke starts at rest (from
        the parked push-start pose) and its end concatenates smoothly with the
        chain (v=0 at the junction).

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
        q_seed = np.asarray(push_start_joint, dtype=float)
        for i in range(n_steps + 1):
            alpha = i / n_steps
            T_wp = T_push_start.copy()
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
            q0 = np.asarray(push_start_joint, dtype=float)
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
        # Boundary: start and end at rest so the stroke starts cleanly from the
        # parked push-start pose and concatenates smoothly with the chain.
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

        meta = self._last_push_meta or {}

        # No io_ms: push is contact-based (no suction IO to time). The old code
        # read ctx.traj_ctrl.last_throw here, which push never writes — so it
        # logged a stale value left by the previous THROW. Dropped.
        row = {
            "iso_time": datetime.datetime.now().isoformat(timespec="milliseconds"),
            "skill": "push",
            "class": target.class_name,
            "belt_mps": round(ctx.conveyor.current, 4),
            "push_speed_mps": meta.get("push_speed", ""),
            "push_distance_m": meta.get("push_distance", ""),
            "theta_deg": round(np.degrees(meta.get("theta", 0.0)), 1),
            "swing_mode": meta.get("swing_mode", ""),
            "swing_deg": round(np.degrees(meta.get("swing", 0.0)), 1),
            "n_stroke": meta.get("n_stroke", ""),
            "chain_dest": meta.get("chain_dest", ""),
        }
        self._append_csv_row(path, row, ctx.log)

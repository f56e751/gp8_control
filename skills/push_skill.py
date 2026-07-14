"""Push skill: position at the intercept, wait for the object, and push it off the belt.

Mirrors the full structure of ``ThrowSkill`` — ambush at the intercept, wait
for the object, then execute a contact push.  Key differences from throw:

  * **No suction**: the gripper/TCP physically pushes the object.
  * **Rule-based push stroke**: a Cartesian straight line on the belt plane —
    run-up acceleration (``PUSH_ACCEL``) from rest, cruising at ``PUSH_SPEED``
    by the contact point, with an impact-synced paddle swing and a post-impact
    lift arc. NOT an NN-generated arc.

Trajectory segments (all concatenated into one dispatch):
  1. **Descent** (aim → grasp): time-optimal joint interpolation via
     ``trajectory()``.
  2. **Push stroke** (grasp_retreat → push_end): 1-D time-optimal run-up +
     cruise along a straight line for ``PUSH_DISTANCE`` m, parallel to the
     belt surface.  Direction = T_grasp1 → T_aim2 projected onto XY.
  3. **Chain** (push_end → next intercept): time-optimal transition so the
     arm flows to the next pick cycle.

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
from gp8_control.trajectory.trajectory_primitive import trajectory, trajectory_3points

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

# NOTE: the old per-class PUSH_THETA_MAP is gone — the push heading is the
# geometric XY direction toward the fixed per-class bin; ``theta`` survives
# only as its logged angle (derived inside ``build_push_trajectory`` from the
# actual ``push_dir``, telemetry-only).

# Per-class push bin TARGET (absolute base-frame XYZ, m). The push aims at
# this fixed bin/chute location; a class with no entry has nowhere to be
# pushed, so ``execute`` aborts cleanly (the old secondary/plan_throw_landing
# fallback was throw-legacy and removed 2026-07-14).
PUSH_BIN_TARGET_MAP: dict[str, tuple] = {
    "transparent": (1.2, -0.30, 0.0),  
    "metal":       (0.95,  0.60, 0.0),
}



# ---- Push stroke parameters ------------------------------------------------

# TCP cruise speed during the push stroke (m/s). The stroke ACCELERATES from
# rest at PUSH_ACCEL over the retreat run-up and cruises at this speed from
# (before) contact to the stroke end. Tune to balance impact force vs. control
# stability; too fast may exceed joint velocity limits. Env-overridable
# (GP8_PUSH_SPEED / GP8_PUSH_ACCEL) for the step-up tracking-limit test —
# see tests/analyze_push_tracking.py.
PUSH_SPEED: float = float(os.environ.get("GP8_PUSH_SPEED", "2.0"))

# TCP acceleration limit (m/s^2) for the stroke run-up (rest → PUSH_SPEED).
# The old constant-speed stroke demanded PUSH_SPEED instantly from rest, which
# was physically impossible — the servo lagged and the actual contact speed was
# uncontrolled. To reach FULL speed by contact this must satisfy
# PUSH_ACCEL >= PUSH_SPEED^2 / (2 * retreat run-up); 2.0/12 needs 0.167 m.
# DO NOT raise casually: 3.0/20 was tried 2026-07-13 and the servo could not
# track it — hits went WEAK (actual speed far below commanded) and the EE
# gouged the belt (per-joint increment clamping bends the chase path BELOW
# the planned scoop). Measure tracking headroom (GP8_MOTION_LOG_DIR) first.
PUSH_ACCEL: float = float(os.environ.get("GP8_PUSH_ACCEL", "12.0"))

# Waypoint sampling rate (Hz) for the push stroke ONLY. Descent/chain stay at
# cfg.TRAJ_HZ (20) — they are plain joint moves the 250 Hz stream resampler
# reconstructs fine from coarse knots. The stroke's straight line + swing +
# lift shape however NEEDS dense knots to survive the joint-space Hermite
# resample (at 20 Hz a 0.2 s stroke is 4 knots and the Cartesian line is lost).
PUSH_STROKE_HZ: float = 100.0

# Post-impact follow-through lift (m). After the contact point the stroke's Z
# rises off PUSH_HEIGHT along a half-cosine, so the paddle floats up off the
# belt instead of skimming it until the chain yanks it up (belt-grazing was
# alarm 4315 territory). 0.0 disables (flat stroke, pre-port behavior).
PUSH_LIFT_HEIGHT: float = 0.02

# Chain-park clearance (m) ABOVE the stroke-start height when a chain parks at
# the NEXT push's backswing pose (chain_park_joint). 0.0 = park EXACTLY at
# the backswing — part of the operator-validated 19:30 2026-07-06 state
# (the later 0.02 hover + other churn degraded hit quality and was rolled
# back to this).
PUSH_CHAIN_PARK_LIFT: float = 0.0

# Approach routing threshold (m, XY distance from the CURRENT TCP to the
# fresh backswing). Below this: descend DIRECTLY (a parked arm with normal
# intercept drift). Above: route via the aim hover (a genuinely far
# reposition — rise over the belt instead of a low cross-belt cut). The old
# joint-delta tolerance (0.25 rad) tripped on ordinary park drift and sent
# the arm up-and-over right after the chain had just descended into the park
# — the observed double up-down bob at push start.
PUSH_APPROACH_VIA_XY: float = 0.30

# Pre-position mode (execute() path). When ON (default) the approach
# (current → [aim hover via] → stroke start) is dispatched as its OWN
# blocking move as soon as the cycle is planned (status POSITIONING) and the
# FIRED trajectory contains only stroke + chain — t_contact becomes
# stroke-only (~0.18 s run-up at the full retreat), a short,
# geometry-independent fire horizon, and ``wait_for_arrival`` re-estimates
# the ETA AFTER positioning so any positioning overrun is absorbed instead
# of shifting the hit. Set GP8_PUSH_PREPOSITION=0 to A/B back to FLOW mode
# (the operator-validated 19:30 2026-07-06 shape: the approach rides inside
# the fired trajectory and the lead must cover its full geometry-dependent
# duration) without editing code.
PUSH_PREPOSITION: bool = os.environ.get("GP8_PUSH_PREPOSITION", "1") != "0"

# Safe transit height (m, absolute TCP Z) for the chain when it parks at the
# NEXT push's backswing (a LOW pose). The chain is a JOINT-space
# interpolation; between two belt-height endpoints its Cartesian Z sags
# BELOW both ends mid-transit (and the paddle edge dips further while the
# wrist swings follow-through→back-lean) — observed belt strikes. The chain
# therefore rises to this height over the park and drops vertically onto it.
PUSH_CHAIN_TRANSIT_Z: float = 0.10

# ---- Safe low-transit (floor-audited) parameters ----------------------------
# Offline floor audit (tests/audit_push_floor.py, 2026-07-14): a plain
# joint-space trajectory() between two backswing-height poses (TCP Z≈0.02)
# sags the COMMANDED TCP down to ~7 mm mid-move and the paddle bottom down to
# −21 mm below the belt (58/90 grid scenarios strike). Joint interpolation
# does not control the EE path, and the wrist re-orientation at low height
# dips the paddle edge further. Low transits are therefore AUDITED at build
# time (``_transit_sag``) and replaced with a Cartesian arc
# (``_build_arc_transit``) when they dip.

# Apex height (m) ABOVE the endpoint-to-endpoint Z line for the Cartesian-arc
# replacement of a LOW transit (backswing→backswing). Operator-chosen start:
# 5 cm. NOTE: the paddle rides ~2 cm below TCP; apex ≈7 cm TCP — marginal vs
# a lying can (~6.5 cm); raise if the arc clips objects. (The chain's long
# cross-belt leg keeps PUSH_CHAIN_TRANSIT_Z=0.10 — do NOT lower it to this.)
PUSH_TRANSIT_ARC_LIFT: float = float(os.environ.get("GP8_PUSH_ARC_LIFT", "0.05"))

# Allowed commanded sag (m) of the paddle bottom BELOW the segment's own
# endpoint clearance before the gate replaces the joint move with the arc.
# Audit data: safe cases sag ≤1 mm, strikes sag 1.4–22 mm → 2 mm separates.
PUSH_TRANSIT_SAG_TOL: float = float(os.environ.get("GP8_PUSH_SAG_TOL", "0.002"))

# Cartesian sampling rate (Hz) for arc/vertical transits (dense, like the
# stroke — coarse joint knots would be flattened back into a joint-space
# Hermite blend); also the floor audit's FK sweep grid (10 ms).
PUSH_TRANSIT_HZ: float = 100.0

# TCP accel/velocity envelope for arc/vertical transits. The arc's cosine
# ease peaks its acceleration (π²L/2T²) AT THE ENDPOINTS — the LOW, near-belt
# points — so inheriting a time-optimal joint move's duration can demand
# accel the servo cannot track (HW-measured, line_stream_test 2026-07-14:
# 5 m/s² → ~1 mm path-dev, 10 → 7 mm, 20 → 70 mm), and the floor audit's
# clearance guarantee would not transfer to the physical path. The transit
# duration is FLOORED so peak accel ≤ ACCEL_MAX and peak vel (πL/2T) ≤
# VEL_MAX (1.4 m/s = the highest speed validated clean at accel 5).
PUSH_TRANSIT_ACCEL_MAX: float = float(os.environ.get("GP8_PUSH_ARC_ACCEL", "5.0"))
PUSH_TRANSIT_VEL_MAX: float = float(os.environ.get("GP8_PUSH_ARC_VEL", "1.4"))

# Paddle bottom geometry in the TOOL frame (tool X = down/approach, Y =
# push-facing, Z = width) — the floor audit's 4-corner FK sweep. GUESSED
# defaults consistent with the code-comment history ("back-lean needed +1 cm
# over neutral") — MEASURE the real paddle and set via env. The gate compares
# sag RELATIVE to the segment's own endpoints, so guessed values shift the
# logged mm but rarely flip the gate decision.
PUSH_PAD_DOWN: float = float(os.environ.get("GP8_PUSH_PAD_DOWN", "0.02"))
PUSH_PAD_FORE: float = float(os.environ.get("GP8_PUSH_PAD_FORE", "0.025"))
PUSH_PAD_HALFW: float = float(os.environ.get("GP8_PUSH_PAD_HALFW", "0.05"))

# ---- Dynamic follow-through (execute() path) --------------------------------
# Follow-through (m) PAST the contact point scales with how far the bin is
# from the contact:  FT = clip(PUSH_FT_GAIN * dist(contact→bin, XY), MIN, MAX)
# and total stroke = actual run-up + FT. A hit close to the bin only needs a
# nudge; a hit far downstream must carry the can much farther. GAIN raised
# 0.25→0.30 (max 0.35→0.40) per operator: downstream pushes needed more
# carry — FT now 0.21 near the bin (0.7 m) up to 0.30 far downstream (1.0 m).
PUSH_FT_GAIN: float = 0.30
PUSH_FT_MIN: float = 0.12
PUSH_FT_MAX: float = 0.40

# Workspace cap for the stroke END point (base-frame XY radius, m). The
# follow-through aims at bins OUTSIDE the arm's reach, so the stroke end must
# be pulled back onto the reachable disc; 0.70 was IK-verified at stroke
# height (mid-stroke IK truncation remains the hard backstop). NOTE this is
# deliberately larger than cfg.MAX_REACH (0.65), which is the conservative
# intercept-planning cap, not the kinematic limit.
PUSH_END_MAX_RADIUS: float = 0.70

# Push stroke distance (m). How far the TCP travels from T_grasp1 in the
# push direction. LEGACY DEFAULT for direct build_push_trajectory callers —
# execute() now computes the per-cycle distance from the run-up + dynamic
# follow-through above instead of using this or the per-class map.
PUSH_DISTANCE: float = 0.3

# (PUSH_DISTANCE_MAP removed 2026-07-06: the per-class fixed stroke length is
# superseded by the dynamic follow-through above — distance now adapts to the
# contact→bin geometry per cycle instead of per class.)

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

# Push arrival-lead (s): how far BEFORE the object's predicted arrival to end
# the WAITING block. COMPUTED PER TRAJECTORY (``dynamic_arrival_lead``) since
# the run-up port — the lead is no longer one opaque constant:
#
#   lead = _stroke_time_to(actual run-up)   # rest→contact time under the profile
#        + qmode_ms_avg                     # measured per-dispatch overhead
#        + PUSH_LEAD_RESIDUAL               # everything not computable (below)
#
# The first term adapts to each cycle's geometry (183 ms at the full 0.2 m
# retreat, less when the min-X clamp shortens the run-up); the residual
# absorbs servo tracking lag + build/IK compute + perception bias and is the
# ONLY knob left to HW-tune: RAISE it if the stroke trails the object, LOWER
# it if the stroke leads. 0.18 = the operator-validated 19:30 2026-07-06
# value; retuned to 0.20 on HW after the 19:30-state restore (each +0.02
# hits ~4 mm earlier at 0.19 m/s belt).
PUSH_LEAD_RESIDUAL = float(os.environ.get("GP8_PUSH_LEAD_RESIDUAL", "0.20"))

# Legacy escape hatch: when GP8_FIXED_DELAY_PUSH is set it OVERRIDES the
# computed lead entirely (old fixed-lead behaviour — useful to A/B the model).
_FIXED_DELAY_PUSH_ENV = os.environ.get("GP8_FIXED_DELAY_PUSH")
# ABSOLUTE base-frame Z (m) of the push stroke AT THE CONTACT POINT (neutral
# paddle). Eye-calibrated 2026-07-06 with tests/push_height_test.py: the old
# 0.07 (derived from the GRASP_Z belt-surface model) struck can TOPS on
# hardware — the suction-grasp Z model does not match the pusher tool's real
# geometry, so trust the visual calibration, not GRASP_Z. Calibration passes:
# 0.03 (paddle centred on a lying can) → 0.01 (operator lowered it further,
# same session). ⚠ 0.01 is ALSO the value that once drove the arm into the
# belt under the old model (alarm 4315 / STATE 101 CODE 112) — that history
# predates the recalibration, but verify with a slow tool stroke ('g' in
# push_height_test) before any full-speed push after changing this.
# (0.015 was tried briefly after a floor strike, then reverted to 0.01 once
# the real cause turned out to be the no-runup fallback's seam cliff — fixed
# by _no_runup_fallback + build's seam bridge, not by height.)
# Env-overridable for RAISED-HEIGHT SHAKEDOWN runs (e.g. GP8_PUSH_HEIGHT=0.05
# to exercise the new transit/gate/preposition code well clear of the belt).
# Everything relative (START_LIFT, LIFT_HEIGHT, arc lift, floor gate) follows
# automatically; only PUSH_CHAIN_TRANSIT_Z is absolute. Default = the
# eye-calibrated contact height.
PUSH_HEIGHT = float(os.environ.get("GP8_PUSH_HEIGHT", "0.01"))

# Extra Z (m) at the STROKE START over PUSH_HEIGHT. The stroke starts in the
# back-lean swing, whose paddle bottom edge dips ~half-width*sin(25°) below
# the neutral pose's lowest point — at PUSH_HEIGHT it gouges the belt
# (observed on HW: neutral cleared at 0.03, back-lean needed 0.04). The
# stroke therefore starts at PUSH_HEIGHT + PUSH_START_LIFT and descends along
# a half-cosine to PUSH_HEIGHT exactly at the contact point (tangent-flat
# there: contact velocity stays purely horizontal). The parked/wait pose uses
# the same lifted height (it IS the stroke-start pose).
PUSH_START_LIFT = 0.01

# Swing push *half-amplitude* (rad). During the stroke the TCP tilts about the
# horizontal axis perpendicular to the push direction — a paddle swing. The
# schedule is IMPACT-SYNCED (ported from the sim's 3-keyframe Slerp): leaning
# back (``SWING_BIAS - SWING_ANGLE``) at the stroke start, exactly NEUTRAL
# (0 = face square to the push) at the contact point, leaning forward
# (``SWING_BIAS + SWING_ANGLE``) at the stroke end. The TCP *position* still
# travels a straight belt-parallel line; only the orientation sweeps. Set to
# 0.0 for a pure perpendicular push.
SWING_ANGLE: float = np.radians(20.0)

# Swing *bias* (rad): shifts BOTH swing endpoints (back-lean start, forward
# follow-through end); the contact-point orientation stays neutral regardless.
# Kept negative so the forward end (``SWING_BIAS + SWING_ANGLE``, currently
# +15°) stays moderate — a large forward tilt drives joint 5 through 0 (wrist
# singularity), which flips the joint-6 IK branch at the swing end.
SWING_BIAS: float = np.radians(-5.0)

# Swing mode (env-selectable at launch, like GP8_FIXED_DELAY_PUSH). Two
# contact styles for the stroke orientation:
#   * "scoop" — impact-synced progressive swing: leaning back
#     (SWING_BIAS - SWING_ANGLE) at the stroke start, exactly NEUTRAL at the
#     contact point, leaning forward (SWING_BIAS + SWING_ANGLE) at the end.
#     Scoop-then-flick; contains the object at contact and can shape the
#     release arc, at the cost of an orientation sweep that risks the joint-5
#     wrist singularity.
#   * "sweep" — fixed tilt (SWEEP_TILT) held for the whole stroke. No
#     orientation sweep, so it is predictable and singularity-safe; loses the
#     scoop.
# Set GP8_PUSH_SWING_MODE=sweep to switch without editing code.
PUSH_SWING_MODE: str = os.environ.get("GP8_PUSH_SWING_MODE", "scoop").strip().lower()

# Fixed tilt (rad) held across the stroke in "sweep" mode — a small forward
# lean (scooping). Kept small so the wrist stays clear of the joint-5
# singularity. Unused in "scoop" mode.
SWEEP_TILT: float = np.radians(5.0)

# ---- Stroke MOTION mode (env-selectable, like PUSH_SWING_MODE) --------------
# Selects the stroke's whole motion SHAPE (distinct from PUSH_SWING_MODE,
# which only picks the ORIENTATION schedule):
#   * "scooping" (default) — the existing shape: half-cosine Z scoop
#     (PUSH_START_LIFT down to PUSH_HEIGHT at contact, PUSH_LIFT_HEIGHT up
#     after), impact-synced swing, run-up at PUSH_ACCEL.
#   * "sweep" — the HW-validated flat straight line (line_stream_test,
#     2026-07-14: path-dev ~1 mm at TCP accel ≤5 m/s²; the scoop's 12 m/s²
#     run-up sits in the measured marginal band): Z flat at PUSH_HEIGHT for
#     the WHOLE stroke, orientation FIXED at SWEEP_TILT (+5° forward lean,
#     same value as swing-mode sweep), run-up at PUSH_SWEEP_ACCEL. The
#     stroke-start poses (backswing / chain park / no-runup fallback) drop
#     PUSH_START_LIFT and carry the same fixed tilt, so park == stroke first
#     knot still holds. Z stays flat for the WHOLE stroke (no post-impact
#     lift: over the reach-capped ~5 cm follow-through a lift spikes the end
#     joint velocity and the uniform 4414 clamp slows the whole stroke —
#     contact force collapse, HW 2026-07-14). The flat exit's chain hook
#     (measured -22 mm) is caught by the direct-chain floor gate instead,
#     which rebuilds a rest-start lifted chain.
# Set GP8_PUSH_STROKE_MODE=sweep to A/B without editing code.
PUSH_STROKE_MODE: str = os.environ.get("GP8_PUSH_STROKE_MODE", "scooping").strip().lower()

# Run-up TCP acceleration (m/s²) for the SWEEP stroke mode. 5.0 is the
# HW-measured clean-tracking ceiling (line_stream_test 2026-07-14: 5 → ~1 mm
# path-dev, 10 → 7 mm, 20 → 70 mm). NOTE: at 5 m/s² the 0.2 m run-up reaches
# only ~1.4 m/s at contact (vs PUSH_SPEED 2.0) — the honest contact speed is
# logged per cycle (contact@...m ≈X m/s); raise the retreat or this accel
# (path-fidelity trade-off) if the hit needs more energy.
PUSH_SWEEP_ACCEL: float = float(os.environ.get("GP8_PUSH_SWEEP_ACCEL", "5.0"))

# Post-contact COMMANDED deceleration cap (m/s²) for the sweep stroke — the
# stroke decelerates to rest inside the follow-through tail (reach-capped to
# ~5 cm, so stopping from ~1.4 m/s needs ~20). 20 exceeds the clean-tracking
# envelope, but the error direction is ALONG the flat line (horizontal
# overshoot, not a floor dive) and a guided ramp beats the 10 ms stop wall it
# replaces (which dove the EE 3 cm below the plane, HW 2026-07-14 19:51).
PUSH_SWEEP_DECEL: float = float(os.environ.get("GP8_PUSH_SWEEP_DECEL", "20.0"))


def _stroke_accel() -> float:
    """Run-up acceleration for the active stroke mode (see PUSH_STROKE_MODE)."""
    return PUSH_SWEEP_ACCEL if PUSH_STROKE_MODE == "sweep" else PUSH_ACCEL


def _stroke_start_lift() -> float:
    """Extra Z at the stroke START over PUSH_HEIGHT for the active mode.

    Scooping starts lifted (back-lean paddle edge clearance, PUSH_START_LIFT);
    sweep is flat — its fixed small forward tilt doesn't dip the edge, and the
    whole point is a constant-Z line.
    """
    return 0.0 if PUSH_STROKE_MODE == "sweep" else PUSH_START_LIFT


class PushSkill(ManipulationSkill):
    """Time one continuous swing so the paddle meets the object on arrival.

    ``execute`` runs the push cycle in one of two shapes, selected by
    ``PUSH_PREPOSITION`` (env ``GP8_PUSH_PREPOSITION``, default ON):

    **PRE-POSITION MODE** (default) — the approach is its own dispatch:

      1. **ARM** — build stroke + chain only (``append_descent=False``);
         ``t_contact`` off the built timestamps is stroke-only (~0.18 s
         run-up) — a short, geometry-independent fire horizon.
      2. **POSITIONING** — send the floor-gated approach (current →
         [aim hover] → the stroke's first knot) NOW, blocking, and park there.
      3. **WAITING** — the ETA is re-estimated AFTER positioning, so a
         positioning overrun is absorbed instead of shifting the hit.
      4. **PUSHING** — fire the pre-built stroke + chain.

    **FLOW MODE** (``GP8_PUSH_PREPOSITION=0``) — the operator-validated
    19:30 2026-07-06 configuration, kept as the A/B escape hatch. No
    backswing parking:

      1. **ARM** — compose the ENTIRE motion up front (approach → backswing →
         run-up stroke → chain) and read the exact contact time off the built
         trajectory timestamps.
      2. **WAITING** — sleep until (arrival − contact time − dispatch −
         residual), holding the current park pose (JGPC zero-order-hold).
      3. **PUSHING** — fire the pre-built trajectory; paddle and object meet
         mid-flow. (adv4ncr 250Hz stream: no queue mode to re-enter.)

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
        """Nominal arrival lead, assuming the full PUSH_RETREAT_DISTANCE run-up.

        Kept for the base-class interface; ``execute`` uses
        :meth:`dynamic_arrival_lead` with the cycle's ACTUAL run-up length
        (min-X clamp aware) instead.
        """
        return self.dynamic_arrival_lead(PUSH_RETREAT_DISTANCE)

    def dynamic_arrival_lead(self, contact_offset: float) -> float:
        """Per-trajectory arrival lead (see the PUSH_LEAD_RESIDUAL comment).

        Push has NO suction to forgive a late hit (unlike throw, whose object
        is already cup-held), so this must cover the FULL post-wait latency:
        the computable parts — the run-up profile's rest→contact time for
        THIS cycle's geometry, plus the measured dispatch overhead — and the
        hand-tuned residual (servo lag, build/IK compute, perception bias).
        ``GP8_FIXED_DELAY_PUSH``, when set, overrides the whole computation
        with the legacy fixed lead.
        """
        if _FIXED_DELAY_PUSH_ENV is not None:
            return float(_FIXED_DELAY_PUSH_ENV)
        t_dispatch = self.ctx.traj_ctrl.qmode_ms_avg / 1000.0
        return self._stroke_time_to(contact_offset) + t_dispatch + PUSH_LEAD_RESIDUAL

    def t_to_contact(self, move_time: float) -> float:
        """Honest push timeline (overrides the base legacy heuristic).

        Push contact is an ACTIVE, timed sweep with NO suction forgiveness, so the
        intercept MUST be placed where the object will be at the REAL strike time,
        not where the bare positioning estimate lands. The real budget from "arm
        starts moving" to "stroke contacts the object" is::

            T_setup#1 (dispatch before POSITIONING)
          + T_position (move_through_via: rise to aim hover + descend to retreat)
          + T_setup#2 (dispatch before the stroke)
          + T_contact_offset (stroke travels grasp_retreat -> contact line)

        The dispatch overhead and the retreat->contact pre-travel are exactly what
        the old ``move_time * factor`` omitted — why the arm aimed upstream of where
        the can actually was and struck the next object. On the adv4ncr 250Hz stream
        driver the old ~0.4 s point-queue re-entry is gone, so T_setup is now just the
        per-dispatch overhead (``qmode_ms_avg``, ~tens of ms) — HW-calibrate; the 2x
        conservatively budgets the positioning + stroke dispatches. T_position is
        opt_time scaled by OPT_TIME_TO_REAL; T_contact_offset = the run-up
        profile's time to cover PUSH_RETREAT_DISTANCE (``_stroke_time_to`` —
        accelerating from rest, NOT the old constant-speed d/v which understated
        the pre-travel by ~80 ms).
        """
        ctx = self.ctx
        t_setup = ctx.traj_ctrl.qmode_ms_avg / 1000.0          # per-dispatch overhead (stream)
        t_position = move_time * ctx.cfg.OPT_TIME_TO_REAL
        t_pre_travel = self._stroke_time_to(PUSH_RETREAT_DISTANCE)
        return 2.0 * t_setup + t_position + t_pre_travel        # dispatch(pos) + dispatch(stroke)

    # ------------------------------------------------------------------
    # Skill entry point (ambush strategy)
    # ------------------------------------------------------------------
    def execute(self, request: "PickRequest") -> SkillResult:
        """Arm the motion, sleep to the fire time, then fire.

        PRE-POSITION MODE (``PUSH_PREPOSITION``, default): build stroke +
        chain (``append_descent=False``, so ``t_contact`` is stroke-only),
        dispatch the floor-gated approach NOW as its own blocking move
        (status POSITIONING) parking at the built stroke's first knot, then
        WAIT and fire.

        FLOW MODE (``GP8_PUSH_PREPOSITION=0``): build approach + stroke +
        chain as ONE trajectory (dispatch=False) with the full approach
        inside ``t_contact``, WAIT, then fire the whole motion.

        Both modes: WAITING — ``wait_for_arrival(offset = t_contact +
        dispatch + residual)``; the stale-stroke guard runs before firing;
        cleanup after.
        """
        ctx = self.ctx
        target = request.target
        current_joint = request.current_joint
        aim_joint = request.aim_joint
        # aim_joint[-1] += PUSH_JOINT6_ANGLE
        grasp_joint = request.grasp_joint
        T_aim = request.T_aim
        T_grasp = request.T_grasp

        # Push does NOT use suction — ensure it's off from any prior cycle.
        ctx.traj_ctrl.suction_off()

        # ---- Compute push target (fixed bin) early ----
        # We need the push direction *before* positioning so we can place
        # the wait pose on the line behind T_grasp, opposite to the push.
        # Push sorts to a fixed per-class bin; a class with no bin has
        # nowhere to be pushed, so abort. (Only reachable via GP8_FORCE_SKILL
        # on an unmapped class — normal routing gates to PUSH_CLASSES, all of
        # which are mapped.)
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
        T_aim2 = np.eye(4)
        T_aim2[:3, :3] = T_aim[:3, :3]
        T_aim2[:3, 3] = np.asarray(bin_xyz, dtype=float)
        ctx.log.info(
            f"Push target for {target.class_name}: fixed bin "
            f"({bin_xyz[0]:+.3f}, {bin_xyz[1]:+.3f}, {bin_xyz[2]:+.3f}) m"
        )

        # ---- Compute the backswing (stroke-start) pose ----
        wait_joint, grasp_retreat_joint, T_grasp_retreat = self._compute_retreat_poses(
            T_grasp, T_aim2, T_aim, aim_joint, grasp_joint,
        )

        # ACTUAL run-up length: the stroke starts at grasp_retreat, so contact
        # happens contact_offset metres in. Measured from the poses (NOT the
        # PUSH_RETREAT_DISTANCE constant) so the min-X clamp's shortened
        # retreat yields the right impact waypoint / swing sync — and the
        # right per-trajectory arrival lead below.
        contact_offset = float(
            np.hypot(*(T_grasp[:2, 3] - T_grasp_retreat[:2, 3]))
        )

        # ---- Dynamic follow-through: scale by the contact→bin distance ----
        # A hit near the bin needs only a nudge; a hit far downstream must
        # carry the can farther. Total stroke = run-up + FT, capped so the
        # stroke END stays on the reachable disc (quadratic for the arc
        # length s where ||retreat_xy + s·dir|| = PUSH_END_MAX_RADIUS).
        push_dir_exec = self._compute_push_direction(T_grasp_retreat, T_aim2)
        dist_bin = float(np.hypot(*(T_aim2[:2, 3] - T_grasp[:2, 3])))
        follow_through = float(
            np.clip(PUSH_FT_GAIN * dist_bin, PUSH_FT_MIN, PUSH_FT_MAX)
        )
        push_distance = contact_offset + follow_through
        rx, ry = float(T_grasp_retreat[0, 3]), float(T_grasp_retreat[1, 3])
        b = rx * push_dir_exec[0] + ry * push_dir_exec[1]
        c = rx * rx + ry * ry - PUSH_END_MAX_RADIUS ** 2
        disc = b * b - c
        if disc >= 0.0:
            s_max = -b + float(np.sqrt(disc))
            if push_distance > s_max:
                ctx.log.info(
                    f"Push FT capped by reach: stroke {push_distance:.3f}→{s_max:.3f}m "
                    f"(end radius {PUSH_END_MAX_RADIUS:.2f}m)"
                )
                # Never shorter than the run-up + a token push (IK truncation
                # is the hard backstop if even this is unreachable).
                push_distance = max(s_max, contact_offset + 0.05)
        ctx.log.info(
            f"Push follow-through {push_distance - contact_offset:.3f}m "
            f"(contact→bin {dist_bin:.2f}m, run-up {contact_offset:.3f}m, "
            f"total {push_distance:.3f}m)"
        )

        # Match the aim via-point's wrist to the push-facing backswing pose so
        # the POSITIONING move keeps joint 6 put. request.aim_joint carries the
        # generic PICK_WRIST_J6 baseline (the belt-wide push-facing MEAN);
        # copying the EXACT wrist removes the remaining few-degree detour
        # during the descent. wait_joint IS the backswing solution now (the
        # separate high-hover IK was dropped — flow mode only ever consumed
        # its wrist). Copy first so we don't mutate the request's array.
        aim_joint = aim_joint.copy()
        aim_joint[-1] = wait_joint[-1]

        # ---- 1. ARM (+ POSITIONING in pre-position mode) ----
        # Both modes pre-build the fired motion NOW: it moves the stroke's
        # ~30 IK solves OFF the post-wait critical path, and t_contact comes
        # from the ACTUAL built trajectory (clamp included). Push has no
        # suction seal to form, so no stationary settle is needed before the
        # stroke — the modes differ only in WHERE the approach rides:
        #   * PRE-POSITION (default): the approach is its OWN blocking
        #     dispatch (status POSITIONING) and the fired motion is stroke +
        #     chain only — t_contact becomes stroke-only (short,
        #     geometry-independent fire horizon).
        #   * FLOW (GP8_PUSH_PREPOSITION=0): the 19:30 2026-07-06
        #     operator-validated shape — approach(current → [aim via] →
        #     backswing) + stroke + chain is ONE dispatch, fired so the
        #     paddle crosses the contact point exactly at the object's
        #     arrival.
        T_cur = ctx.robot.forward_kinematics(
            np.asarray(current_joint, dtype=float)[:6]
        )
        _xy_gap = float(np.hypot(*(T_cur[:2, 3] - T_grasp_retreat[:2, 3])))
        # Genuinely far reposition → rise over via the aim hover; a parked
        # arm with ordinary drift descends directly (no up-and-over bob).
        approach_via = None if _xy_gap < PUSH_APPROACH_VIA_XY else aim_joint

        # Chain pre-plan: the arm frees up ~(object arrival + post-contact
        # motion) from now — estimate for next_chain_target's pre_delay.
        v_belt = ctx.conveyor.current
        y_now = ctx.object_y_now(target, time.time(), v_belt)
        t_wait_est = max(0.0, (y_now - float(T_grasp[1, 3])) / max(v_belt, 1e-6))
        push_time = self._stroke_time_to(push_distance)
        nxt = ctx.next_chain_target(grasp_retreat_joint, t_wait_est + push_time)
        next_grasp, next_cand = nxt if nxt is not None else (None, None)
        chain_park = None
        if next_cand is not None:
            chain_park = ctx.skill_obj_for(next_cand).chain_park_joint(
                next_grasp, next_cand
            )

        if PUSH_PREPOSITION:
            # ARM FIRST (stroke + chain only) so POSITIONING can park at the
            # BUILT stroke's first knot. Parking at grasp_retreat_joint is
            # NOT equivalent: the deep no-runup fallback returns the raw
            # suction-grasp pose while the stroke's first knot is the
            # push-oriented lifted pose (~4 cm + wrist apart). Flow mode's
            # seam bridge catches that cliff INSIDE the dispatch, but a
            # stroke-only dispatch has no bridge — parking anywhere but the
            # built first knot would command a 10 ms cliff at fire time.
            traj_push, vel_push, ts_push, final_joint, t_contact = (
                self.build_push_trajectory(
                    current_joint, grasp_retreat_joint, T_grasp_retreat,
                    T_aim2, next_grasp=next_grasp, append_chain=True,
                    append_descent=False, push_distance=push_distance,
                    contact_offset=contact_offset, chain_park=chain_park,
                    approach_via=None, dispatch=False,
                )
            )
            q_park = traj_push[:, 0].copy()
            # POSITIONING: the same floor-gated approach build that flow mode
            # runs inside build_push_trajectory (shared helper), dispatched
            # standalone and BLOCKING — the arm is parked at the stroke start
            # when this returns and holds it through WAITING (JGPC
            # zero-order-hold). current_joint is still the true pose here:
            # nothing has moved since the epoch read it (the previous cycle's
            # dispatch was blocking, and the arm parks on ZOH between cycles).
            ctx.set_status("POSITIONING", target.class_name)
            traj_pos, vel_pos, ts_pos = self._build_gated_approach(
                current_joint, q_park, via=approach_via,
            )
            ctx.traj_ctrl.send_trajectory_queue(
                traj_pos, vel_pos, ts_pos, final_joint=q_park,
            )
            # SETTLE GUARD (HW finding 2026-07-14): the dispatch returns at
            # the END OF THE COMMAND STREAM, not at physical arrival. On the
            # validated driver config the servo trails by only ~15 ms, but a
            # silently degraded link/limiter — e.g. the bare-relaunch
            # axis_increment_factor=0.1 trap (~10% speed cap + bent EE paths)
            # — leaves the arm FAR from the park with no other symptom, and
            # the stroke would fire from the wrong pose. Verify arrival; a
            # healthy stack settles in well under 100 ms.
            if not ctx.traj_ctrl._wait_for_position(
                q_park, tolerance=0.05, timeout_sec=1.0
            ):
                _q_now = ctx.traj_ctrl.current_joints
                _gap = (
                    float(np.max(np.abs(
                        np.asarray(_q_now, dtype=float)[:6] - q_park[:6]
                    )))
                    if _q_now is not None else float("nan")
                )
                ctx.log.error(
                    f"POSITIONING did not settle within 1 s (max joint gap "
                    f"{_gap:.3f} rad) — the servo is not tracking the stream. "
                    f"Check the driver's axis_increment_factor (bare relaunch "
                    f"= 0.1 commissioning cap) / RT link before pushing."
                )
                return self._abort(
                    "positioning never settled — driver/link degraded?"
                )
        else:
            traj_push, vel_push, ts_push, final_joint, t_contact = (
                self.build_push_trajectory(
                    current_joint, grasp_retreat_joint, T_grasp_retreat,
                    T_aim2, next_grasp=next_grasp, append_chain=True,
                    append_descent=True, push_distance=push_distance,
                    contact_offset=contact_offset, chain_park=chain_park,
                    approach_via=approach_via, dispatch=False,
                )
            )

        # ---- 2. WAITING: sleep until the FIRE time ----
        # offset = the built trajectory's contact time + dispatch overhead +
        # residual → dispatching at (arrival − offset) makes paddle and object
        # meet mid-flow. The arm holds its park pose until then (JGPC ZOH).
        if _FIXED_DELAY_PUSH_ENV is not None:
            lead = float(_FIXED_DELAY_PUSH_ENV)
        else:
            lead = (
                t_contact
                + ctx.traj_ctrl.qmode_ms_avg / 1000.0
                + PUSH_LEAD_RESIDUAL
            )
        _route = "via-hover" if approach_via is not None else "direct"
        _approach_txt = (
            f"stroke-only; pre-positioned {_xy_gap * 100:.0f}cm {_route}"
            if PUSH_PREPOSITION
            else f"approach {_xy_gap * 100:.0f}cm {_route}"
        )
        ctx.log.info(
            f"Push FIRE lead {lead * 1000:.0f}ms = traj contact "
            f"{t_contact * 1000:.0f}ms ({_approach_txt}) + dispatch "
            f"{ctx.traj_ctrl.qmode_ms_avg:.0f}ms + residual "
            f"{PUSH_LEAD_RESIDUAL * 1000:.0f}ms"
            + (" [OVERRIDDEN by GP8_FIXED_DELAY_PUSH]" if _FIXED_DELAY_PUSH_ENV else "")
        )
        ctx.set_status("WAITING", target.class_name)
        ctx.wait_for_arrival(target, T_grasp[1, 3], offset=lead)

        # ---- 2b. STALE-STROKE GUARD ----
        # If timing slipped and the object has already passed the ENTIRE stroke span,
        # do NOT fire: the stroke would sweep into the NEXT object (the observed
        # "pushed the next PET"). The stroke runs grasp_retreat -> push_end; the only
        # belt-Y range it can still contact is between those two ends. The push
        # heading's Y sign is geometry-dependent (for an intercept UPSTREAM of the bin
        # the stroke actually sweeps downstream-in-Y), so compare against the
        # most-DOWNSTREAM (smallest-Y) stroke end, not just grasp_retreat. Drop
        # cleanly (same cleanup as the queue-fail path). The timeline fix (skill-aware
        # t_to_contact) should make this rare; this is the hard safety net.
        stroke_end_y = T_grasp_retreat[1, 3] + push_distance * push_dir_exec[1]
        stroke_min_y = min(float(T_grasp_retreat[1, 3]), float(stroke_end_y))
        obj_y_now = ctx.object_y_now(target, time.time(), ctx.conveyor.current)
        if obj_y_now < stroke_min_y:
            ctx.log.warn(
                f"Push abort: id={target.track_id} {target.class_name} already past the "
                f"stroke (y={obj_y_now:+.3f} < stroke_min {stroke_min_y:+.3f}); "
                f"skipping stale stroke"
            )
            return self._abort("object passed stroke span; push aborted")

        # ---- 3. PUSHING: FIRE the pre-built motion ----
        ctx.log_action_timing(target, T_grasp[1, 3], "push-fire")
        ctx.set_status("PUSHING", target.class_name)
        ctx.traj_ctrl.send_trajectory_queue(
            traj_push, vel_push, ts_push, final_joint=final_joint,
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
        """Compute the backswing pose behind T_grasp opposite the push direction.

        * **grasp_retreat_joint** — stroke-start height (PUSH_HEIGHT +
          PUSH_START_LIFT, stroke-start swing), descent target & push stroke
          start.
        * **wait_joint** — a COPY of the backswing joints. Flow mode only
          ever consumed the old high-hover wait pose's WRIST (``execute``'s
          ``aim_joint[-1]`` donor), so the separate T_wait construction + its
          IK were dropped — the backswing donates its wrist directly.

        ``T_aim`` is no longer used; the parameter is kept for signature
        compatibility with external callers (tests/audit_push_floor.py).

        Returns ``(wait_joint, grasp_retreat_joint, T_grasp_retreat)``.
        Falls back to ``(aim_joint, grasp_joint, T_grasp)`` — no retreat, the
        stroke starts AT the object from standstill — when the backswing is
        unreachable (grasp-retreat IK fails at far-downstream intercepts, or
        the min-X clamp leaves no run-up).
        """
        ctx = self.ctx
        push_dir = self._compute_push_direction(T_grasp, T_aim2)
        dx = PUSH_RETREAT_DISTANCE * push_dir[0]
        dy = PUSH_RETREAT_DISTANCE * push_dir[1]

        # Clamp so X doesn't go below PUSH_RETREAT_MIN_X.
        if T_grasp[0, 3] - dx < PUSH_RETREAT_MIN_X and dx > 0:
            available = max(0.0, T_grasp[0, 3] - PUSH_RETREAT_MIN_X)
            if available <= 0:
                # No run-up possible at all — push from the object itself.
                return self._no_runup_fallback(
                    T_grasp, push_dir, aim_joint, grasp_joint
                )
            scale = available / dx
            dx *= scale
            dy *= scale

        # Bake the push-facing orientation into the pose (replaces the old
        # joint-6 override). The grasp-retreat pose adopts the *stroke-start*
        # swing so the descent ends exactly where the push stroke begins.
        R_grasp = self._push_orientation(push_dir, self._swing_at(0.0))

        # Grasp retreat pose: retreated XY, STROKE-START height. This pose IS
        # the stroke start (back-lean swing), so it carries PUSH_START_LIFT —
        # at bare PUSH_HEIGHT the tilted paddle edge gouges the belt.
        T_grasp_retreat = T_grasp.copy()
        T_grasp_retreat[:3, :3] = R_grasp
        T_grasp_retreat[0, 3] -= dx
        T_grasp_retreat[1, 3] -= dy
        T_grasp_retreat[2, 3] = PUSH_HEIGHT + _stroke_start_lift()

        # Seed the IK with the pose the arm actually moves next-to.
        # Unseeded, the solver picks the min-|θ4| Euler branch regardless of
        # the arm's configuration — when that lands on the OTHER wrist branch
        # the POSITIONING move swings J4/J6 by ~π (a slow, visible wind-up
        # that eats into the arrival budget). Seeding routes through the
        # IK's toward-seed ±2π unwrap + limit check (same job as the sim's
        # unwinding guard, done at the right layer).
        ik_grasp = ctx.robot.inverse_kinematics(
            T_grasp_retreat, q_init=np.asarray(grasp_joint, dtype=float)[:6]
        )
        if ik_grasp is None:
            # Backswing pose out of the workspace (typical far-downstream
            # intercept: the retreat lands beyond reach) — fall back to a
            # no-retreat push from the object itself (standstill start).
            ctx.log.warn(
                f"Backswing IK unreachable at retreat "
                f"({T_grasp_retreat[0, 3]:+.3f}, {T_grasp_retreat[1, 3]:+.3f}); "
                f"pushing without run-up"
            )
            return self._no_runup_fallback(
                T_grasp, push_dir, aim_joint, grasp_joint
            )
        grasp_retreat_joint = np.asarray(ik_grasp, dtype=float)

        # Flow mode only consumes the wait pose's WRIST (execute copies it
        # onto the aim via-point); the backswing solution already carries the
        # push-facing wrist, so it donates it directly — no separate
        # high-hover IK. Full 6-DOF IK realises the facing+swing.
        return grasp_retreat_joint.copy(), grasp_retreat_joint, T_grasp_retreat

    def _no_runup_fallback(
        self,
        T_grasp: np.ndarray,
        push_dir: np.ndarray,
        aim_joint: np.ndarray,
        grasp_joint: np.ndarray,
    ) -> tuple:
        """No-retreat fallback: the stroke-START pose AT the object's XY.

        Back-lean orientation, stroke-start height — i.e. exactly the pose
        the degenerate (contact_offset≈0) stroke begins with, so the approach
        ends where the stroke starts. Returning the raw SUCTION grasp pose
        here (old behaviour) left a ~4 cm Z + orientation CLIFF between the
        approach's last knot and the stroke's second knot (first is sliced as
        a "duplicate") — the observed floor strike. If even this IK fails,
        the raw poses are returned and build's seam bridge smooths the gap.
        """
        T_fb = T_grasp.copy()
        T_fb[:3, :3] = self._push_orientation(push_dir, self._swing_at(0.0))
        T_fb[2, 3] = PUSH_HEIGHT + _stroke_start_lift()
        ik = self.ctx.robot.inverse_kinematics(
            T_fb, q_init=np.asarray(grasp_joint, dtype=float)[:6]
        )
        if ik is None:
            return aim_joint.copy(), grasp_joint.copy(), T_grasp.copy()
        q = np.asarray(ik, dtype=float)
        return q.copy(), q, T_fb

    def chain_park_joint(
        self,
        next_grasp: np.ndarray,
        next_cand: "TrackedObject",
    ) -> "Optional[np.ndarray]":
        """Park pose for a chain whose NEXT object THIS skill will push.

        The next push's BACKSWING pose — retreat XY behind the next grasp
        along its bin heading (same min-X clamp as ``_compute_retreat_poses``),
        back-lean orientation, at the stroke-start height (+
        ``PUSH_CHAIN_PARK_LIFT``, currently 0 → the park IS the backswing).
        The arm waits in the next stroke's start attitude, so firing goes
        straight into the run-up — no suction-point dogleg, no aim-hover
        bounce, no descend blip. Mirrors the sim's ``pushingafter`` chain.

        Returns None (caller parks at the generic lifted standby) when the
        class has no fixed bin — the heading is unknown until that cycle
        plans — or IK fails. Best-effort: the intercept is re-planned next
        epoch; a badly shifted retreat just exceeds the approach-routing
        threshold and takes the via-hover route.
        """
        ctx = self.ctx
        bin_xyz = PUSH_BIN_TARGET_MAP.get(next_cand.class_name)
        if bin_xyz is None:
            return None
        q_next = np.asarray(next_grasp, dtype=float)[:6]
        T_next = ctx.robot.forward_kinematics(q_next)
        d = np.array(
            [bin_xyz[0] - T_next[0, 3], bin_xyz[1] - T_next[1, 3], 0.0]
        )
        nrm = float(np.linalg.norm(d))
        if nrm < 1e-6:
            return None
        push_dir = d / nrm
        dx = PUSH_RETREAT_DISTANCE * push_dir[0]
        dy = PUSH_RETREAT_DISTANCE * push_dir[1]
        if T_next[0, 3] - dx < PUSH_RETREAT_MIN_X and dx > 0:
            available = max(0.0, T_next[0, 3] - PUSH_RETREAT_MIN_X)
            if available <= 0:
                return None
            scale = available / dx
            dx *= scale
            dy *= scale
        T_park = np.eye(4)
        T_park[:3, :3] = self._push_orientation(push_dir, self._swing_at(0.0))
        T_park[0, 3] = T_next[0, 3] - dx
        T_park[1, 3] = T_next[1, 3] - dy
        T_park[2, 3] = PUSH_HEIGHT + _stroke_start_lift() + PUSH_CHAIN_PARK_LIFT
        q = ctx.robot.inverse_kinematics(T_park, q_init=q_next)
        return None if q is None else np.asarray(q, dtype=float)

    # (plan_push_target removed 2026-07-14: the secondary/plan_throw_landing
    # fallback was throw-legacy and semantically wrong for bin-sorting — an
    # unmapped class now aborts cleanly in ``execute`` instead. No external
    # callers.)

    # ------------------------------------------------------------------
    # Push trajectory build + dispatch
    # ------------------------------------------------------------------
    def build_push_trajectory(
        self,
        aim_joint: np.ndarray,
        grasp_joint: np.ndarray,
        T_grasp: np.ndarray,
        T_aim2: np.ndarray,
        next_grasp: "Optional[np.ndarray]" = None,
        append_chain: bool = True,
        append_descent: bool = True,
        push_distance: float = PUSH_DISTANCE,
        contact_offset: "Optional[float]" = None,
        chain_park: "Optional[np.ndarray]" = None,
        approach_via: "Optional[np.ndarray]" = None,
        dispatch: bool = True,
    ) -> "Optional[tuple]":
        """Build (and by default dispatch) the 3-segment push trajectory.

        FLOW MODE (``dispatch=False``): build everything up front and return
        ``(traj, vel, ts, final_joint, t_contact)`` WITHOUT sending —
        ``t_contact`` is the exact trajectory time (post-clamp) at which the
        paddle crosses the contact point, so the caller can fire the whole
        motion timed against the object's arrival instead of parking at the
        backswing and waiting (no suction seal to form — push doesn't need a
        stationary wait). ``approach_via`` (e.g. the aim hover) routes the
        approach segment through a via point when starting far away
        (belt-dip avoidance); None = direct time-optimal approach.

        Segments:

        1. **Descent** (aim_joint → grasp_joint): time-optimal joint
           interpolation via ``trajectory()``. Brings the arm from the high
           hover down to the near-object pose at belt height. FLOOR-GATED:
           the built segment is audited with ``_transit_sag`` and a low
           direct descent that sags below its own endpoints is replaced by a
           Cartesian arc (``_build_arc_transit``); the via-hover route is
           audited warn-only. Built by the shared ``_build_gated_approach``
           helper, which pre-position mode also dispatches standalone as its
           POSITIONING move (with ``append_descent=False`` here).

        2. **Push stroke** (grasp → push_end): rule-based Cartesian straight
           line for ``push_distance`` m parallel to the belt surface, paced by
           the run-up profile (rest → PUSH_ACCEL → PUSH_SPEED cruise) and
           sampled densely at PUSH_STROKE_HZ. Direction is from T_grasp
           towards T_aim2, projected onto the XY plane. ``contact_offset``
           (m from the stroke start to the object — the actual retreat
           run-up) syncs the paddle swing to be NEUTRAL at contact and starts
           the post-impact lift arc there (see ``_build_push_stroke``). Each
           Cartesian waypoint is converted to joint space via IK. The stroke
           STARTS at rest (the arm was parked at grasp_retreat waiting); its
           EXIT velocity is carried into the chain (continuized — see the
           build code) so there is no full stop at push_end.

        3. **Chain** (push_end → next_intercept or grasp): time-optimal
           transition so the arm flows directly to the next pick cycle,
           same pattern as ``ThrowSkill.build_throw_trajectory``. When it
           parks at a LOW next-backswing pose, the final descent from the
           raised transit via is a Cartesian-controlled leg
           (``_build_arc_transit`` with lift=0) instead of a joint move.

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
        # arrival-timing budget (the dynamic arrival lead).
        n_desc = 0
        desc_end_t = 0.0
        if append_descent:
            # Built + floor-gated by the shared helper (also dispatched
            # standalone as pre-position mode's POSITIONING move): the direct
            # route is audited with _transit_sag and replaced by a Cartesian
            # arc when it sags; the via-hover route is audited warn-only.
            traj_desc, vel_desc, ts_desc = self._build_gated_approach(
                aim_joint, grasp_joint, via=approach_via,
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
        # Push heading angle — telemetry only (log + CSV theta_deg); the
        # stroke uses push_dir directly. Derived from the real push_dir so
        # the logged angle matches the actual stroke.
        theta = float(np.arctan2(push_dir[1], push_dir[0]))
        traj_stroke, vel_stroke, ts_stroke = self._build_push_stroke(
            T_grasp, push_dir, grasp_joint, push_distance,
            contact_offset=contact_offset,
        )

        # Clamp segment velocities against robot joint limits (Yaskawa alarm
        # 4414 prevention). Same 2-pass rescale approach as throw_skill.
        # Capture the stretch ratio so t_contact below reflects the ACTUAL
        # (post-clamp) contact time — the fire-time computation depends on it.
        _stroke_T_pre = float(ts_stroke[-1]) if len(ts_stroke) else 1e-9
        traj_stroke, vel_stroke, ts_stroke = self._clamp_stroke_velocity(
            traj_stroke, vel_stroke, ts_stroke,
        )
        _stretch = float(ts_stroke[-1]) / max(_stroke_T_pre, 1e-9)

        # SEAM BRIDGE: the approach must END exactly where the stroke STARTS.
        # Normally it does (grasp_retreat IS the stroke-start pose); the raw
        # no-runup fallback can end at the suction grasp pose instead (~4 cm
        # higher, different wrist). The unconditional "slice the duplicate
        # first stroke sample" then commanded a Z+orientation CLIFF between
        # adjacent knots — the floor strike. Bridge any real gap
        # time-optimally before the stroke.
        if append_descent and seg_traj:
            _q_seam = seg_traj[-1][:, -1]
            _gap = float(np.max(np.abs(_q_seam - traj_stroke[:, 0])))
            # Threshold 0.15 rad: trajectory()'s knot-grid truncation leaves
            # up to ~0.05 rad of residue on EVERY normal approach — bridging
            # that added a stutter before each stroke. Only bridge REAL pose
            # mismatches (the no-runup fallback's ~4 cm + wrist cliff).
            if _gap > 0.15:
                ctx.log.warn(
                    f"Approach/stroke seam gap {_gap:.3f} rad — inserting "
                    f"time-optimal bridge (no-runup fallback path)"
                )
                traj_b, vel_b, ts_b = trajectory(
                    _q_seam, zero6, traj_stroke[:, 0], zero6,
                    ctx.M1[:6], ctx.M2[:6], hertz=ctx.cfg.TRAJ_HZ,
                )
                # ---- FLOOR GATE (bridge) ---- The bridge only runs on the
                # deep no-runup fallback (grasp pose ≈ GRASP_Z → stroke start
                # at ~0.02 m): a low joint-space move WITH wrist
                # re-orientation — exactly the sag mechanism the approach
                # gate protects against. Same audit + arc replacement.
                sag_b, t_sag_b = self._transit_sag(
                    traj_b, vel_b, ts_b, q_end=traj_stroke[:, 0],
                )
                if sag_b < -PUSH_TRANSIT_SAG_TOL:
                    ctx.log.warn(
                        f"Seam bridge sags {sag_b * 1000:.1f}mm below "
                        f"endpoints @t={t_sag_b:.2f}s → Cartesian arc transit"
                    )
                    arc_b = self._build_arc_transit(
                        _q_seam, traj_stroke[:, 0],
                        lift=PUSH_TRANSIT_ARC_LIFT,
                        duration_hint=float(ts_b[-1]),
                    )
                    if arc_b is not None:
                        traj_b, vel_b, ts_b = arc_b
                    else:
                        ctx.log.warn(
                            "Bridge arc IK failed — keeping the sagging "
                            "joint-space bridge (floor-strike risk!)"
                        )
                seg_traj.append(traj_b[:, 1:])
                seg_vel.append(vel_b[:, 1:])
                seg_ts.append(ts_b[1:] + desc_end_t)
                n_desc += traj_b.shape[1] - 1
                desc_end_t += float(ts_b[-1])

        # Trajectory time at which the paddle crosses the contact point:
        # approach duration + (analytic run-up time to contact) × clamp
        # stretch. contact_offset None/degenerate → contact at stroke start.
        _s_hit = float(np.clip(contact_offset, 0.0, push_distance)) \
            if contact_offset is not None else 0.0
        t_contact = desc_end_t + self._stroke_time_to(_s_hit) * _stretch


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
        # Segment 3: Chain  (push_end → next intercept or idle pose) — OPTIONAL
        # ================================================================
        # execute() now ALWAYS chains (append_chain=True): to the next throw's
        # grasp when there is one, else to the shared idle/standby pose — so the
        # arm parks HIGH instead of low at push_end. (append_chain=False is kept
        # as a still-valid option for any caller that wants to END at push_end:
        # then the NEXT cycle's POSITIONING flows from push_end. MotoROS2 needs
        # the point queue to drain before queue-mode re-entry, so a chain can't
        # run asynchronously — it would block the next pick's dispatch and trip
        # code 2 'Must call start_point_queue_mode'.)
        push_end_q = traj_stroke[:, -1]         # last waypoint of stroke (6-DOF)
        # CONTINUIZE stroke -> chain: feed the stroke's natural EXIT velocity into the
        # chain's START (push_end_dq) instead of starting the chain from REST. The
        # chain trajectory then begins at full speed, so its POSITIONS flow
        # continuously out of the stroke rather than accelerating from a standstill —
        # which is what removes the visible "push, then stop, rotate + move" pause and
        # its overrun beyond the planned timestamps (throw stays ≈planned because its
        # arc is continuous; push overran ~+0.13s from this stop). Mirrors the throw's
        # release-velocity carry (build_throw_trajectory). Clipped to the joint-vel
        # limits (Yaskawa alarm 4414 safety). The exit velocity is a least-squares
        # slope over the last ~5 knots (sim's estimate_dq_end) — at 10 ms knot
        # spacing a 2-point difference would hand IK jitter to the chain. These
        # knot velocities ARE consumed now: the stream backend's cubic-Hermite
        # resampler interpolates positions+velocities directly.
        if traj_stroke.shape[1] >= 2:
            push_end_dq = np.clip(
                self._estimate_exit_velocity(traj_stroke, ts_stroke),
                -ctx.M1[:6], ctx.M1[:6],
            )
            vel_stroke[:, -1] = push_end_dq     # queued stroke now exits at speed
        else:
            push_end_dq = vel_stroke[:, -1]

        if append_chain:
            # Chain target (3-way): the next THROW's grasp if committed; else, when
            # the queue is EMPTY (no next object detected yet, ①a), the full
            # home/standby pose; else (a next object exists but was NOT committed —
            # different skill or unreachable after the push, ①b/②) the lifted
            # standby — push_end raised to home Z, so the arm clears the belt without
            # the wasted home round-trip. copy() so ctx.idle_joint is never mutated
            # by the wrist write below.
            chain_via = None
            if chain_park is not None:
                # The NEXT object's skill supplied its own action-start park —
                # for push: the next backswing pose + PUSH_CHAIN_PARK_LIFT.
                # Full 6-DOF IK pose, wrist included — no overwrites.
                chain_target = np.asarray(chain_park, dtype=float).copy()
                chain_dest = "next action-start park"
                # LOW park → route the transit through the park RAISED to
                # PUSH_CHAIN_TRANSIT_Z (same XY/orientation), then descend
                # vertically. A direct low-to-low joint interpolation sags
                # below its endpoints mid-belt (observed floor strikes).
                T_hi = ctx.robot.forward_kinematics(chain_target[:6])
                T_hi[2, 3] = PUSH_CHAIN_TRANSIT_Z
                q_via = ctx.robot.inverse_kinematics(
                    T_hi, q_init=chain_target[:6]
                )
                if q_via is not None:
                    chain_via = np.asarray(q_via, dtype=float)
                else:
                    ctx.log.warn(
                        "Chain transit via IK failed; direct low chain"
                    )
            elif next_grasp is not None:
                # Generic park OVER the next grasp (raised Z, PICK_WRIST_J6
                # baseline from lifted_standby_joint) — a belt-height chain
                # from push_end (bin side) would sweep the TCP low across the
                # belt (alarm 4315 / grazing objects) when we DON'T know the
                # next action-start pose; descend next epoch.
                chain_target = ctx.lifted_standby_joint(next_grasp)
                chain_dest = "over next grasp"
            elif not ctx.queue:
                chain_target = self.idle_target().copy()
                chain_dest = "home/idle (queue empty)"
            else:
                chain_target = ctx.lifted_standby_joint(push_end_q)
                chain_dest = "lifted standby"

            if chain_via is not None:
                # Leg 1 (joint-space, keeps the stroke's exit-velocity
                # carry): push_end → park raised to PUSH_CHAIN_TRANSIT_Z.
                # It travels at 0.10 Z, so it's audited warn-only (no
                # auto-replace in v1 — the warning tells us if it ever dips).
                traj_l1, vel_l1, ts_l1 = trajectory(
                    push_end_q, push_end_dq,
                    chain_via[:6], zero6,
                    ctx.M1[:6], ctx.M2[:6], hertz=ctx.cfg.TRAJ_HZ,
                )
                sag_l1, t_sag_l1 = self._transit_sag(
                    traj_l1, vel_l1, ts_l1, q_end=chain_via[:6],
                )
                if sag_l1 < -PUSH_TRANSIT_SAG_TOL:
                    ctx.log.warn(
                        f"Chain transit leg sags {sag_l1 * 1000:.1f}mm below "
                        f"its endpoints @t={t_sag_l1:.2f}s (audit only — "
                        f"transits at {PUSH_CHAIN_TRANSIT_Z:.2f}m)"
                    )
                # Leg 2 (NEW, Cartesian-controlled final descent onto the
                # LOW park): lift=0 → pure Cartesian blend straight down.
                # The old joint-space final descent (via trajectory_3points)
                # showed −5 mm paddle-bottom sag in one audited geometry.
                leg2 = self._build_arc_transit(
                    chain_via[:6], chain_target[:6], lift=0.0,
                )
                if leg2 is None:
                    ctx.log.warn(
                        "Chain descent arc IK failed; joint-space descent "
                        "(old behavior)"
                    )
                    traj_l2, vel_l2, ts_l2 = trajectory(
                        chain_via[:6], zero6,
                        chain_target[:6], zero6,
                        ctx.M1[:6], ctx.M2[:6], hertz=ctx.cfg.TRAJ_HZ,
                    )
                else:
                    traj_l2, vel_l2, ts_l2 = leg2
                # Combine leg1 + leg2 with the standard seam pattern (drop
                # leg2's duplicate first knot, shift its clock by leg1's
                # end). The combined segment still STARTS with the push_end
                # knot at t=0 — the unconditional drop/shift below expects
                # exactly that.
                _l1_end = float(ts_l1[-1])
                traj_chain = np.concatenate([traj_l1, traj_l2[:, 1:]], axis=1)
                vel_chain = np.concatenate([vel_l1, vel_l2[:, 1:]], axis=1)
                ts_chain = np.concatenate([ts_l1, ts_l2[1:] + _l1_end])
                # Defensive zero-gap dedupe across the seam so the stream's
                # Hermite resampler stays active (a 1-knot leg would leave a
                # duplicate timestamp at the join).
                keep = np.concatenate(([True], np.diff(ts_chain) > 1e-9))
                traj_chain = traj_chain[:, keep]
                vel_chain = vel_chain[:, keep]
                ts_chain = ts_chain[keep]
            else:
                traj_chain, vel_chain, ts_chain = trajectory(
                    push_end_q, push_end_dq,
                    chain_target[:6], zero6,
                    ctx.M1[:6], ctx.M2[:6], hertz=ctx.cfg.TRAJ_HZ,
                )
                # FLOOR GATE (direct chain — the one previously un-audited
                # branch). Decelerating the stroke's exit-velocity carry in
                # joint space can HOOK the EE below the stroke plane
                # (measured -22/-24 mm on HW 2026-07-14, sweep mode, right at
                # the can). If the commanded paddle bottom dips: rebuild
                # REST-START — vertical Cartesian rise over push_end to
                # PUSH_CHAIN_TRANSIT_Z, then a joint move to the target. The
                # carry (and its flow) is dropped only on this violating
                # branch; the stroke then exits at rest.
                sag_c, t_sag_c = self._transit_sag(
                    traj_chain, vel_chain, ts_chain, q_end=chain_target[:6],
                )
                if sag_c < -PUSH_TRANSIT_SAG_TOL:
                    ctx.log.warn(
                        f"Direct chain hooks {sag_c * 1000:.1f}mm below its "
                        f"endpoints @t={t_sag_c:.2f}s → rest-start lifted "
                        f"chain (exit-velocity carry dropped)"
                    )
                    T_up = ctx.robot.forward_kinematics(push_end_q[:6])
                    T_up[2, 3] = max(PUSH_CHAIN_TRANSIT_Z, float(T_up[2, 3]))
                    q_up = ctx.robot.inverse_kinematics(
                        T_up, q_init=push_end_q[:6]
                    )
                    rise = (
                        self._build_arc_transit(
                            push_end_q[:6],
                            np.asarray(q_up, dtype=float)[:6],
                            lift=0.0,
                        )
                        if q_up is not None else None
                    )
                    if rise is None:
                        ctx.log.warn(
                            "Chain rise IK failed — keeping the HOOKING "
                            "direct chain (floor-strike risk!)"
                        )
                    else:
                        vel_stroke[:, -1] = 0.0   # stroke now exits at rest
                        push_end_dq = zero6
                        traj_r, vel_r, ts_r = rise
                        traj_j, vel_j, ts_j = trajectory(
                            np.asarray(q_up, dtype=float)[:6], zero6,
                            chain_target[:6], zero6,
                            ctx.M1[:6], ctx.M2[:6], hertz=ctx.cfg.TRAJ_HZ,
                        )
                        _r_end = float(ts_r[-1])
                        traj_chain = np.concatenate(
                            [traj_r, traj_j[:, 1:]], axis=1
                        )
                        vel_chain = np.concatenate(
                            [vel_r, vel_j[:, 1:]], axis=1
                        )
                        ts_chain = np.concatenate([ts_r, ts_j[1:] + _r_end])
                        keep = np.concatenate(
                            ([True], np.diff(ts_chain) > 1e-9)
                        )
                        traj_chain = traj_chain[:, keep]
                        vel_chain = vel_chain[:, keep]
                        ts_chain = ts_chain[keep]
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
            chain_dest = "push_end (no chain)"
            final_joint = push_end_q

        traj_push = np.concatenate(seg_traj, axis=1)
        vel_push = np.concatenate(seg_vel, axis=1)
        ts_full = np.concatenate(seg_ts)

        assert traj_push.shape[1] == vel_push.shape[1] == ts_full.shape[0], (
            f"push traj/vel/ts length mismatch: "
            f"{traj_push.shape[1]}/{vel_push.shape[1]}/{ts_full.shape[0]}"
        )
        # Estimated Cartesian speed AT CONTACT under the run-up profile: full
        # PUSH_SPEED when the run-up (contact_offset) is long enough, else
        # sqrt(2*a*s). Logged so a min-X-clamped (short) run-up is visible.
        v_contact = None
        contact_note = ""
        if contact_offset is not None:
            v_contact = min(
                PUSH_SPEED,
                float(np.sqrt(2.0 * _stroke_accel() * max(contact_offset, 0.0))),
            )
            contact_note = f", contact@{contact_offset:.3f}m ≈{v_contact:.2f}m/s"
        if PUSH_STROKE_MODE == "sweep":
            swing_txt = (
                f"SWEEP stroke (flat Z, tilt {np.degrees(SWEEP_TILT):+.0f}° "
                f"fixed, a={PUSH_SWEEP_ACCEL:.0f}m/s²)"
            )
        elif PUSH_SWING_MODE == "sweep":
            swing_txt = f"sweep swing {np.degrees(SWEEP_TILT):+.0f}° fixed"
        else:
            swing_txt = (
                f"scoop swing "
                f"{np.degrees(SWING_BIAS - SWING_ANGLE):+.0f}°→0°@hit→"
                f"{np.degrees(SWING_BIAS + SWING_ANGLE):+.0f}°"
            )
        ctx.log.info(
            f"Push traj: descent {n_desc} + stroke {n_stroke} steps "
            f"(d={push_distance:.3f}m, cruise {PUSH_SPEED:.2f}m/s{contact_note}, "
            f"θ={np.degrees(theta):.1f}°, {swing_txt}, "
            f"lift {PUSH_LIFT_HEIGHT * 100:.0f}cm), chain→{chain_dest}"
        )

        # No decimation: the adv4ncr stream backend cubic-Hermite-resamples ALL
        # knots onto its 4 ms grid (and the JTC backend takes the whole
        # trajectory in one action goal) — dense knots are pure fidelity here.
        # The old ≥150 ms thinning existed for MotoROS2's synchronous per-point
        # queue service and was destroying the stroke (a 0.26 s stroke survived
        # as ~2 knots → joint-interpolated blur instead of a straight line).

        self._last_push_meta = {
            "n_descent": n_desc,
            "n_stroke": n_stroke,
            "push_distance": push_distance,
            "push_speed": PUSH_SPEED,
            "contact_offset": contact_offset,
            "v_contact": v_contact,
            "t_contact": t_contact,
            "theta": theta,
            "stroke_mode": PUSH_STROKE_MODE,
            "swing_mode": PUSH_SWING_MODE,
            "swing": self._swing_at(1.0),   # end-of-stroke tilt (both modes)
            "chain_dest": chain_dest,
        }

        # FLOW MODE: hand the composed trajectory (and its exact contact
        # time) back to the caller, which fires it timed against arrival.
        if not dispatch:
            return traj_push, vel_push, ts_full, final_joint, t_contact

        # Dispatch. No timed release — push is contact-based, no suction.
        ctx.traj_ctrl.send_trajectory_queue(
            traj_push, vel_push, ts_full, final_joint=final_joint,
        )
        return None

    # ------------------------------------------------------------------
    # Push stroke helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _stroke_time_to(s: float) -> float:
        """Time (s) for the run-up + cruise profile to cover distance ``s``.

        Rest → PUSH_ACCEL up to PUSH_SPEED, then constant PUSH_SPEED (the
        stroke never decelerates — its exit velocity is carried into the
        chain). Inverse of the profile used by ``_stroke_profile``; also the
        honest pre-travel/stroke-duration model for timing (t_to_contact,
        next_chain_target's pre_delay).
        """
        v = max(PUSH_SPEED, 1e-6)
        a = max(_stroke_accel(), 1e-6)      # mode-dependent (sweep: 5 m/s²)
        s = max(float(s), 0.0)
        s_acc = v * v / (2.0 * a)
        if s <= s_acc:
            return float(np.sqrt(2.0 * s / a))
        return v / a + (s - s_acc) / v

    @classmethod
    def _stroke_profile_decel(
        cls, distance: float, decel_from: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """SWEEP profile: run-up to the contact, then DECELERATE to rest.

        Same accel/cruise phase as :meth:`_stroke_profile` up to ``decel_from``
        (the contact offset), then a constant deceleration sized to reach REST
        exactly at ``distance`` (capped at PUSH_SWEEP_DECEL — a too-short tail
        leaves a small residual exit speed for the chain gate to audit).

        Why: the flat sweep otherwise exits at full speed, and when the
        direct-chain floor gate drops that carry it leaves a 10 ms "stop wall"
        in the command stream — the servo's joint-wise momentum then dives the
        EE ~3 cm below the stroke plane at the reach edge (HW 2026-07-14,
        19:51 session: z 0.031 → 0.003 right at the stroke end). A COMMANDED
        deceleration keeps the reference guiding the arm flat through the
        stop. Contact speed is unaffected (decel starts AT the contact).
        """
        v = max(PUSH_SPEED, 1e-6)
        a = max(_stroke_accel(), 1e-6)
        D = max(float(distance), 1e-9)
        s_hit = float(np.clip(decel_from, 0.0, D))
        tail = D - s_hit
        v_hit = min(v, float(np.sqrt(2.0 * a * s_hit))) if s_hit > 0 else 0.0
        if tail <= 1e-6 or v_hit <= 1e-6:
            return cls._stroke_profile(D)
        a_need = v_hit * v_hit / (2.0 * tail)
        a_dec = min(a_need, PUSH_SWEEP_DECEL)
        v_exit = float(np.sqrt(max(0.0, v_hit * v_hit - 2.0 * a_dec * tail)))
        tA = cls._stroke_time_to(s_hit)
        tB = (v_hit - v_exit) / a_dec
        T = tA + tB
        n = max(2, int(np.ceil(T * PUSH_STROKE_HZ)))
        ts = np.linspace(0.0, T, n + 1)
        t_acc = min(v_hit / a, tA)
        s_acc = 0.5 * a * t_acc * t_acc
        s_arr = np.empty_like(ts)
        for i, t in enumerate(ts):
            if t <= t_acc:
                s_arr[i] = 0.5 * a * t * t
            elif t <= tA:
                s_arr[i] = s_acc + v_hit * (t - t_acc)
            else:
                tau = t - tA
                s_arr[i] = s_hit + v_hit * tau - 0.5 * a_dec * tau * tau
        s_arr = np.clip(s_arr, 0.0, D)
        s_arr[-1] = D
        return ts, s_arr

    @classmethod
    def _stroke_profile(cls, distance: float) -> tuple[np.ndarray, np.ndarray]:
        """Sample the 1-D run-up + cruise profile along the stroke line.

        Port of the sim's ``trajectory_linear_optimal`` idea (time-optimal 1-D
        motion along the Cartesian line) minus the deceleration phase: the old
        constant-speed stroke demanded PUSH_SPEED instantly from rest, so the
        servo lagged and the contact speed was uncontrolled; here the retreat
        run-up is what accelerates the paddle so it CRUISES AT PUSH_SPEED
        through the contact point (provided PUSH_ACCEL reaches it in time —
        see the constant's comment).

        Returns ``(ts, s_arr)`` — uniform time grid at PUSH_STROKE_HZ and the
        distance-along-line at each sample, ending exactly at ``distance``.
        """
        v = max(PUSH_SPEED, 1e-6)
        a = max(_stroke_accel(), 1e-6)      # mode-dependent (sweep: 5 m/s²)
        D = max(float(distance), 1e-9)
        T = cls._stroke_time_to(D)
        n = max(2, int(np.ceil(T * PUSH_STROKE_HZ)))
        ts = np.linspace(0.0, T, n + 1)
        t_acc = v / a
        s_acc = v * v / (2.0 * a)
        s_arr = np.where(
            ts < t_acc,
            0.5 * a * ts ** 2,
            s_acc + v * (ts - t_acc),
        )
        s_arr = np.clip(s_arr, 0.0, D)
        s_arr[-1] = D
        return ts, s_arr

    @staticmethod
    def _estimate_exit_velocity(
        traj: np.ndarray, ts: np.ndarray, k: int = 5
    ) -> np.ndarray:
        """Per-joint least-squares slope of the last ``k`` waypoints.

        Port of the sim's ``estimate_dq_end``: at the dense stroke sampling
        (10 ms knots) a 2-point finite difference amplifies IK jitter straight
        into the chain's start velocity; regressing over ~k knots keeps the
        stroke → chain velocity carry smooth.
        """
        n = traj.shape[1]
        k = int(min(max(k, 2), n))
        t = ts[-k:]
        if float(t[-1] - t[0]) < 1e-9:
            return np.zeros(traj.shape[0])
        return np.array(
            [np.polyfit(t, traj[j, -k:], 1)[0] for j in range(traj.shape[0])]
        )

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
    def _swing_at(alpha: float, alpha_hit: "Optional[float]" = None) -> float:
        """Swing tilt (rad) at stroke progress ``alpha`` ∈ [0, 1].

        Behaviour depends on ``PUSH_SWING_MODE``:

        * ``"sweep"`` — fixed ``SWEEP_TILT`` held for the whole stroke
          (``alpha``/``alpha_hit`` ignored). No orientation sweep, so it is
          predictable and singularity-safe; loses the scoop.

        * ``"scoop"`` (default) — with ``alpha_hit`` (contact point as a
          stroke fraction): IMPACT-SYNCED schedule, ported from the sim's
          3-keyframe Slerp — piecewise-linear through
          (0, ``SWING_BIAS - SWING_ANGLE``) → (``alpha_hit``, **0**) →
          (1, ``SWING_BIAS + SWING_ANGLE``). The paddle face is guaranteed
          square to the push at the instant it meets the object; before this
          port the contact angle was whatever the whole-stroke linear sweep
          happened to pass through at the (geometry-dependent) contact
          fraction. Without ``alpha_hit`` (None, or degenerate — contact
          at/outside the stroke ends): the legacy whole-stroke linear sweep
          back → forward. Since only the tilt angle varies about a FIXED
          horizontal axis, linear angle interpolation IS the geodesic — no
          quaternion Slerp needed.
        """
        if PUSH_STROKE_MODE == "sweep":
            # Sweep MOTION mode: fixed tilt for the whole stroke (and the
            # stroke-start poses that call _swing_at(0.0)), regardless of
            # PUSH_SWING_MODE — the flat line has no orientation schedule.
            return SWEEP_TILT
        if PUSH_SWING_MODE == "sweep":
            return SWEEP_TILT
        back = SWING_BIAS - SWING_ANGLE
        fwd = SWING_BIAS + SWING_ANGLE
        if alpha_hit is None or not (0.02 <= alpha_hit <= 0.98):
            return SWING_BIAS + SWING_ANGLE * (2.0 * alpha - 1.0)
        if alpha <= alpha_hit:
            return back * (1.0 - alpha / alpha_hit)
        return fwd * (alpha - alpha_hit) / (1.0 - alpha_hit)

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
        contact_offset: "Optional[float]" = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Straight-line push: run-up + cruise, impact-synced swing, lift arc.

        The TCP *position* travels a straight belt-parallel line from
        ``T_grasp`` for ``push_distance`` m along ``direction``, paced by the
        1-D run-up profile (``_stroke_profile``: rest → PUSH_ACCEL →
        PUSH_SPEED cruise) and sampled at PUSH_STROKE_HZ.

        ``contact_offset`` (m along the stroke where the object sits — the
        actual retreat run-up length) drives the impact-synced schedules:

        * the swing hits NEUTRAL exactly at contact (:meth:`_swing_at` with
          ``alpha_hit``), and
        * the Z traces a shallow scoop: start at PUSH_HEIGHT +
          PUSH_START_LIFT (back-lean clearance), half-cosine down to
          PUSH_HEIGHT at contact, half-cosine up to PUSH_HEIGHT +
          PUSH_LIFT_HEIGHT at the end.

        When ``contact_offset`` is None/degenerate: legacy whole-stroke
        linear swing, and Z settles from the lifted start height to
        PUSH_HEIGHT over the first 30 % of the stroke (no end lift).

        Each waypoint is solved with full 6-DOF IK; if IK fails mid-stroke,
        the trajectory is truncated. Joint velocities are central finite
        differences; both boundaries are zeroed (the caller re-derives the
        exit velocity for the chain carry).

        Returns ``(traj_6, vel_6, ts)`` with shapes ``(6, n), (6, n), (n,)``.
        """
        ctx = self.ctx
        if PUSH_STROKE_MODE == "sweep" and contact_offset is not None:
            # Sweep: decelerate to rest after the contact (no stop wall for
            # the chain gate to create — see _stroke_profile_decel).
            ts, s_arr = self._stroke_profile_decel(
                push_distance, float(contact_offset)
            )
        else:
            ts, s_arr = self._stroke_profile(push_distance)
        n_samples = len(s_arr)

        # Contact point as a stroke fraction (swing sync + lift start).
        alpha_hit: "Optional[float]" = None
        s_hit: "Optional[float]" = None
        if contact_offset is not None:
            frac = float(np.clip(contact_offset / max(push_distance, 1e-9), 0.0, 1.0))
            if 0.02 <= frac <= 0.98:
                alpha_hit = frac
                s_hit = frac * push_distance

        # ---- Cartesian position + swing orientation → joint space via IK ----
        # Seed each IK with the previous waypoint so the analytical solver
        # keeps the SAME wrist branch across the stroke. Without this, when
        # joint 5 crosses 0 (wrist singularity) near the swing end the two
        # Euler solutions cross over and the unseeded min|θ4| pick flips
        # joints 4/6 by ~π — a sudden joint-6 jump that the chain then holds.
        waypoints: list[np.ndarray] = []
        q_seed = np.asarray(grasp_joint, dtype=float)
        for i in range(n_samples):
            s = float(s_arr[i])
            alpha = s / max(push_distance, 1e-9)
            T_wp = T_grasp.copy()
            T_wp[:3, :3] = self._push_orientation(
                direction, self._swing_at(alpha, alpha_hit)
            )
            T_wp[0, 3] += s * direction[0]
            T_wp[1, 3] += s * direction[1]
            if PUSH_STROKE_MODE == "sweep":
                # Sweep: flat straight line at PUSH_HEIGHT for the WHOLE
                # stroke. Deliberately NO post-impact lift: with the short
                # (reach-capped, ~5 cm) follow-through a 2 cm end-lift spikes
                # the end-of-stroke joint velocity, and the UNIFORM 4414
                # clamp then slows the ENTIRE stroke (HW 2026-07-14: ratio
                # 1.3→1.65, contact speed 1.41→0.82 m/s — "no push force").
                # The flat exit's downward chain hook is handled where it
                # belongs: the direct-chain FLOOR GATE audits the carry and
                # rebuilds a rest-start lifted chain when it would dip.
                T_wp[2, 3] = PUSH_HEIGHT
                ik = ctx.robot.inverse_kinematics(T_wp, q_init=q_seed)
                if ik is None:
                    ctx.log.warn(
                        f"Push stroke IK failed at step {i}/{n_samples - 1}; "
                        f"truncating stroke to {len(waypoints)} waypoints"
                    )
                    break
                q_seed = np.asarray(ik, dtype=float)
                waypoints.append(q_seed)
                continue
            # Z schedule — a shallow scoop, matching the swing schedule:
            #   start  (back-lean): PUSH_HEIGHT + PUSH_START_LIFT (the tilted
            #          paddle edge dips below the neutral pose's lowest point)
            #   contact (neutral) : PUSH_HEIGHT — the true low point; both
            #          cosine ramps are tangent-flat here, so the contact
            #          velocity is purely horizontal at the profile speed
            #   end (follow-thru) : PUSH_HEIGHT + PUSH_LIFT_HEIGHT
            z = PUSH_HEIGHT
            if s_hit is not None:
                if s < s_hit and PUSH_START_LIFT > 0.0:
                    ratio = s / s_hit
                    z += PUSH_START_LIFT * 0.5 * (1.0 + np.cos(np.pi * ratio))
                elif s > s_hit and PUSH_LIFT_HEIGHT > 0.0:
                    # Post-impact follow-through: half-cosine lift off the belt.
                    ratio = (s - s_hit) / max(push_distance - s_hit, 1e-9)
                    z += PUSH_LIFT_HEIGHT * 0.5 * (1.0 - np.cos(np.pi * ratio))
            elif PUSH_START_LIFT > 0.0:
                # Degenerate contact geometry (no run-up — object at the stroke
                # start): still START at the parked pose's lifted height so the
                # first waypoint matches where the arm is parked, and settle to
                # PUSH_HEIGHT over the first 30% of the stroke. No end lift.
                ramp = 0.3 * push_distance
                if s < ramp:
                    z += PUSH_START_LIFT * 0.5 * (1.0 + np.cos(np.pi * s / ramp))
            T_wp[2, 3] = z
            ik = ctx.robot.inverse_kinematics(T_wp, q_init=q_seed)
            if ik is None:
                ctx.log.warn(
                    f"Push stroke IK failed at step {i}/{n_samples - 1}; "
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
            ts = np.array([0.0, 1.0 / PUSH_STROKE_HZ])
            return traj, vel, ts

        n = len(waypoints)
        traj = np.column_stack(waypoints)              # (6, n)
        ts = ts[:n]                                    # truncated on IK failure

        # ---- Joint velocities via central finite differences ----
        # np.gradient handles the (uniform) grid and the truncated tail alike;
        # these knot velocities feed the stream backend's cubic-Hermite
        # resampler directly, so interior values matter.
        vel = np.gradient(traj, ts, axis=1) if n > 2 else np.zeros_like(traj)
        # Boundary: start at rest (the run-up profile really does start from
        # standstill); the end is zeroed here and re-derived by the caller
        # for the stroke → chain velocity carry.
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

            # Re-derive joint velocities at the stretched time scale
            # (np.gradient — the grid is uniform but the joint PROGRESS along
            # it is not, and a truncated stroke keeps this general).
            n = traj.shape[1]
            vel = np.gradient(traj, ts, axis=1) if n > 2 else np.zeros_like(traj)
            vel[:, 0] = 0.0
            vel[:, -1] = 0.0

            ctx.log.warn(
                f"Push stroke clamped: seg vel ratio {ratio:.2f} > 1 "
                f"→ time scale ×{scale:.2f}"
            )
        return traj, vel, ts

    # ------------------------------------------------------------------
    # Floor audit + safe low transit (Cartesian arc)
    # ------------------------------------------------------------------
    def _build_gated_approach(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        via: "Optional[np.ndarray]" = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Approach move (start → [via] → goal) with the FLOOR GATE.

        Shared by both execute modes: flow mode embeds this as the fired
        trajectory's descent segment (``build_push_trajectory`` segment 1);
        pre-position mode dispatches it standalone as the POSITIONING move.

        Route selection is the CALLER's (``PUSH_APPROACH_VIA_XY``):

        * ``via`` given (genuinely far reposition): ``trajectory_3points``
          through the via (aim hover) at rest, so the approach rises over
          the belt instead of cutting a possibly belt-dipping direct path.
          Audited warn-only — it rises high by construction, no auto-replace.
        * ``via`` is None (parked arm with ordinary drift): direct
          time-optimal ``trajectory()``. Joint-space interpolation does not
          control the EE path: between two LOW endpoints the commanded
          paddle bottom sags below both mid-move (audited strikes down to
          −21 mm — the wrist re-orientation dips the paddle edge on top of
          the TCP sag). The built move is audited with ``_transit_sag`` as
          the stream driver would actually command it; a sagging move is
          replaced by a Cartesian arc (``_build_arc_transit``), arc-IK
          failure falls back to a raised-midpoint trapezoid, and if THAT
          fails too the sagging direct move is kept with a loud warning.

        Returns ``(traj, vel, ts)`` — deduped, ts strictly increasing,
        starting at t=0 with both boundary velocities zero.
        """
        ctx = self.ctx
        zero6 = np.zeros(6)
        q_start = np.asarray(q_start, dtype=float)[:6]
        q_goal = np.asarray(q_goal, dtype=float)[:6]
        if via is not None:
            traj_desc, vel_desc, ts_desc = trajectory_3points(
                q_start, zero6,
                np.asarray(via, dtype=float)[:6], zero6,
                q_goal, zero6,
                ctx.M1[:6], ctx.M2[:6], hertz=ctx.cfg.TRAJ_HZ,
            )
        else:
            traj_desc, vel_desc, ts_desc = trajectory(
                q_start, zero6,
                q_goal, zero6,
                ctx.M1[:6], ctx.M2[:6], hertz=ctx.cfg.TRAJ_HZ,
            )
        # Drop any zero-gap duplicate knots (trajectory_3points repeats the
        # via sample) — a non-increasing timestamp would make the stream's
        # Hermite resampler fall back to LINEAR for the WHOLE dispatch,
        # flattening the stroke's velocity profile.
        keep = np.concatenate(([True], np.diff(ts_desc) > 1e-9))
        traj_desc = traj_desc[:, keep]
        vel_desc = vel_desc[:, keep]
        ts_desc = ts_desc[keep]

        # ---- FLOOR GATE (approach) ----
        # q_end=q_goal: audit against the TRUE goal pose — trajectory()'s
        # truncated last knot can sit mid-dip and mask the sag (see
        # _transit_sag docstring).
        sag, t_sag = self._transit_sag(traj_desc, vel_desc, ts_desc,
                                       q_end=q_goal)
        if sag < -PUSH_TRANSIT_SAG_TOL:
            if via is not None:
                # The RISE leg (start → aim hover) is high by construction,
                # but the final DESCENT leg (hover → low park) is still a
                # joint-space interpolation and can dip below the park
                # (observed on HW 2026-07-14: -5.3 mm in SWEEP mode, whose
                # park sits a full START_LIFT lower than scooping's). Rebuild
                # as joint rise + Cartesian vertical descent (the chain's
                # leg1/leg2 pattern); keep the sagging original only if the
                # descent IK fails.
                ctx.log.warn(
                    f"Via-hover approach sags {sag * 1000:.1f}mm below its "
                    f"endpoints @t={t_sag:.2f}s → Cartesian final descent"
                )
                _via6 = np.asarray(via, dtype=float)[:6]
                leg2 = self._build_arc_transit(_via6, q_goal, lift=0.0)
                if leg2 is None:
                    ctx.log.warn(
                        "Via-descent arc IK failed — keeping the SAGGING "
                        "via-hover approach (floor-strike risk!)"
                    )
                else:
                    traj_l1, vel_l1, ts_l1 = trajectory(
                        q_start, zero6, _via6, zero6,
                        ctx.M1[:6], ctx.M2[:6], hertz=ctx.cfg.TRAJ_HZ,
                    )
                    traj_l2, vel_l2, ts_l2 = leg2
                    _l1_end = float(ts_l1[-1])
                    traj_desc = np.concatenate(
                        [traj_l1, traj_l2[:, 1:]], axis=1
                    )
                    vel_desc = np.concatenate([vel_l1, vel_l2[:, 1:]], axis=1)
                    ts_desc = np.concatenate([ts_l1, ts_l2[1:] + _l1_end])
                    keep = np.concatenate(
                        ([True], np.diff(ts_desc) > 1e-9)
                    )
                    traj_desc = traj_desc[:, keep]
                    vel_desc = vel_desc[:, keep]
                    ts_desc = ts_desc[keep]
            else:
                ctx.log.warn(
                    f"Direct descent sags {sag * 1000:.1f}mm below "
                    f"endpoints @t={t_sag:.2f}s → Cartesian arc transit "
                    f"(lift {PUSH_TRANSIT_ARC_LIFT * 100:.0f}cm)"
                )
                arc = self._build_arc_transit(
                    q_start, q_goal,
                    lift=PUSH_TRANSIT_ARC_LIFT,
                    duration_hint=float(ts_desc[-1]),
                )
                if arc is not None:
                    traj_desc, vel_desc, ts_desc = arc
                else:
                    # Arc IK failed: trapezoid fallback — rise over a
                    # raised midpoint (goal orientation), then descend. A
                    # coarse up-and-over, but it breaks the low cut.
                    T_from = ctx.robot.forward_kinematics(q_start)
                    T_to = ctx.robot.forward_kinematics(q_goal)
                    T_mid = T_to.copy()
                    T_mid[:2, 3] = 0.5 * (T_from[:2, 3] + T_to[:2, 3])
                    T_mid[2, 3] = (
                        max(float(T_from[2, 3]), float(T_to[2, 3]))
                        + PUSH_TRANSIT_ARC_LIFT
                    )
                    q_mid = ctx.robot.inverse_kinematics(
                        T_mid, q_init=q_goal
                    )
                    if q_mid is not None:
                        traj_desc, vel_desc, ts_desc = trajectory_3points(
                            q_start, zero6,
                            np.asarray(q_mid, dtype=float)[:6], zero6,
                            q_goal, zero6,
                            ctx.M1[:6], ctx.M2[:6], hertz=ctx.cfg.TRAJ_HZ,
                        )
                        # Mandatory dedupe after every trajectory_3points
                        # (a zero-gap via knot would force the stream's
                        # linear-resample fallback for the whole dispatch).
                        keep = np.concatenate(
                            ([True], np.diff(ts_desc) > 1e-9)
                        )
                        traj_desc = traj_desc[:, keep]
                        vel_desc = vel_desc[:, keep]
                        ts_desc = ts_desc[keep]
                    else:
                        ctx.log.warn(
                            "Arc AND trapezoid-mid IK failed — keeping the "
                            "SAGGING direct descent (floor-strike risk!)"
                        )
        return traj_desc, vel_desc, ts_desc

    def _paddle_bottom_z(self, T: np.ndarray) -> float:
        """Lowest base-frame Z of the paddle's 4 bottom corners at pose ``T``.

        Corners sit PUSH_PAD_DOWN along tool X (approach/down), ±PUSH_PAD_FORE
        along tool Y (push-facing) and ±PUSH_PAD_HALFW along tool Z (width) —
        the same convention as ``tests/audit_push_floor.py::make_corners``.
        The MIN over corners is what strikes the belt first when the wrist
        tilts (the TCP Z alone under-reports the dip by the edge geometry).
        """
        corners = np.array([
            [PUSH_PAD_DOWN,  PUSH_PAD_FORE,  PUSH_PAD_HALFW],
            [PUSH_PAD_DOWN,  PUSH_PAD_FORE, -PUSH_PAD_HALFW],
            [PUSH_PAD_DOWN, -PUSH_PAD_FORE,  PUSH_PAD_HALFW],
            [PUSH_PAD_DOWN, -PUSH_PAD_FORE, -PUSH_PAD_HALFW],
        ])
        pts = (T[:3, :3] @ corners.T).T + T[:3, 3]
        return float(np.min(pts[:, 2]))

    def _transit_sag(self, traj, vel, ts, q_end=None) -> tuple[float, float]:
        """Audit a joint segment for commanded paddle-bottom sag.

        Resamples (traj, vel, ts) exactly like the 250 Hz stream driver
        (cubic Hermite over the knots + per-segment envelope clamp — see
        ``trajectory_controller._stream_trajectory``) but on a coarser
        1/PUSH_TRANSIT_HZ (10 ms) grid, FKs every sample and takes the
        paddle-bottom Z. Returns ``(sag, t_at_min)`` where
        ``sag = min(bottom_z) − min(first, last bottom_z)`` — ≤0 means the
        commanded path dips BELOW its own endpoints. Costs ~10–30 ms for a
        0.2 s transit: runs at ARM/positioning build time, OFF the
        fire-critical path.

        ``q_end`` — the segment's INTENDED goal joints. ``trajectory()``'s
        knot grid TRUNCATES up to ~1/hertz of motion, so the raw last knot
        can itself sit MID-DIP (audited: −5.9 mm at the truncated knot vs
        +1.3 mm at the true backswing) and mask the sag the gate must catch;
        the streamed motion continues to the true goal via ``final_joint`` /
        the next segment's first knot. When given, ``q_end`` is appended
        exactly like the stream driver appends ``final_joint`` (one knot at
        +0.05 s, zero velocity) so both the resample and the endpoint
        reference see the REAL segment goal. Callers that gate/warn on the
        result must pass it. As a bonus this makes 1-knot segments (tiny
        chained drift) auditable; without ``q_end`` they return (0, 0) —
        CubicHermiteSpline needs ≥2 samples.
        """
        ctx = self.ctx
        arr = np.asarray(traj, dtype=float)
        varr = np.asarray(vel, dtype=float)
        times = np.asarray(ts, dtype=float).ravel()
        if q_end is not None:
            arr = np.concatenate(
                [arr, np.asarray(q_end, dtype=float)[:arr.shape[0]].reshape(-1, 1)],
                axis=1,
            )
            varr = np.concatenate([varr, np.zeros((varr.shape[0], 1))], axis=1)
            times = np.append(times, times[-1] + 0.05)
        # Degenerate 1-knot segment: trajectory() emits a SINGLE knot when
        # opt_time < one 20 Hz grid step — the COMMON chained case (the chain
        # parks the arm exactly at the next backswing, so the re-planned
        # descent is a tiny drift). Without a q_end there is nothing to
        # audit — and CubicHermiteSpline needs ≥2 samples (its len==1 diff
        # guard is vacuously true).
        if arr.shape[1] < 2 or float(times[-1]) - float(times[0]) <= 0.0:
            return 0.0, 0.0
        dt = 1.0 / PUSH_TRANSIT_HZ
        grid = np.arange(float(times[0]), float(times[-1]) + dt, dt)
        if varr.shape == arr.shape and np.all(np.diff(times) > 0):
            try:
                from scipy.interpolate import CubicHermiteSpline
                samples = np.column_stack([
                    CubicHermiteSpline(times, arr[j], varr[j])(grid)
                    for j in range(arr.shape[0])
                ])
                # Envelope clamp — same as the stream driver: the servo never
                # receives a sample beyond its two bracketing knots, so the
                # audit must not count cubic over/undershoot either.
                seg = np.clip(
                    np.searchsorted(times, grid, side="right") - 1,
                    0, len(times) - 2,
                )
                lo = np.minimum(arr[:, seg], arr[:, seg + 1]).T
                hi = np.maximum(arr[:, seg], arr[:, seg + 1]).T
                samples = np.clip(samples, lo, hi)
            except Exception:
                samples = np.column_stack(
                    [np.interp(grid, times, arr[j]) for j in range(arr.shape[0])]
                )
        else:
            samples = np.column_stack(
                [np.interp(grid, times, arr[j]) for j in range(arr.shape[0])]
            )
        bz = np.array([
            self._paddle_bottom_z(ctx.robot.forward_kinematics(samples[i, :6]))
            for i in range(samples.shape[0])
        ])
        i_min = int(np.argmin(bz))
        sag = float(bz[i_min] - min(float(bz[0]), float(bz[-1])))
        return sag, float(grid[i_min])

    def _build_arc_transit(
        self,
        q_from: np.ndarray,
        q_to: np.ndarray,
        lift: float,
        duration_hint: "Optional[float]" = None,
    ) -> "Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]":
        """Cartesian-controlled low transit: straight XY, sinusoidal Z bump.

        Replaces a floor-sagging joint-space move (see ``_transit_sag``) with
        an EE-path-controlled arc from FK(q_from) to FK(q_to):

        * Position: XY straight-line lerp; ``Z(u) = lerp(zA, zB, u) +
          lift·sin(πu)`` — a smooth bump whose apex clears the belt
          (``lift=0`` → pure Cartesian blend, used for the chain's final
          vertical descent).
        * Orientation: scipy Slerp evaluated at ``smoothstep(u) = u²(3−2u)``
          so the wrist re-orientation concentrates mid-arc (high point) and
          the LOW endpoints see ~zero angular rate — the paddle-edge dip
          happens where there is clearance, not at belt height.
        * Timing: ``u(t) = ½ − ½·cos(πt/T)`` (rest→rest ease). ``T =
          duration_hint`` (pass the replaced joint move's duration) or the
          SAMPLED arc length / 0.5 m/s as a fallback, floored so a degenerate
          hint (1-knot direct descent → 0.0 s) can't produce a zero-length
          timeline. Sampled densely at PUSH_TRANSIT_HZ — coarse knots would
          be flattened back into a joint-space Hermite blend by the stream.

        Every waypoint is solved with SEEDED IK (previous solution — the same
        wrist-branch-continuity pattern as ``_build_push_stroke``). Any
        mid-path IK failure returns **None**: a transit must reach ``q_to``;
        truncation is not acceptable — callers fall back to the original
        joint move or a raised trapezoid. Endpoints are snapped exactly onto
        ``q_from``/``q_to`` (near the θ5≈0 wrist singularity the J4/J6 Euler
        split is non-unique, so IK(FK(q)) can legitimately land on the other
        branch — log + snap, never assert). The result passes through
        ``_clamp_stroke_velocity`` (alarm 4414 safety), which zeroes both
        boundary velocities — correct for a rest→rest transit.
        """
        ctx = self.ctx
        try:
            from scipy.spatial.transform import Rotation, Slerp
        except Exception:
            ctx.log.warn("scipy Rotation/Slerp unavailable; no arc transit")
            return None

        q_from = np.asarray(q_from, dtype=float)[:6]
        q_to = np.asarray(q_to, dtype=float)[:6]
        T_from = ctx.robot.forward_kinematics(q_from)
        T_to = ctx.robot.forward_kinematics(q_to)
        p_from, p_to = T_from[:3, 3], T_to[:3, 3]

        slerp = Slerp(
            [0.0, 1.0],
            Rotation.from_matrix(np.stack([T_from[:3, :3], T_to[:3, :3]])),
        )

        def _pos_at(u: float) -> np.ndarray:
            p = p_from + (p_to - p_from) * u
            return np.array(
                [p[0], p[1], p[2] + lift * np.sin(np.pi * u)]
            )

        # Duration: prefer the replaced move's own duration; fall back to the
        # SAMPLED path length (the chord alone misses the bump) at 0.5 m/s.
        # Floor at a few samples — an all-zero ts would blow up np.gradient.
        u_probe = np.linspace(0.0, 1.0, 21)
        pts = np.stack([_pos_at(float(u)) for u in u_probe])
        path_len = float(
            np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
        )
        T_dur = float(duration_hint) if duration_hint is not None else 0.0
        if T_dur <= 1e-3:
            T_dur = path_len / 0.5
        # ENVELOPE FLOOR (see PUSH_TRANSIT_ACCEL_MAX): the cosine ease has
        # peak accel π²L/2T² at the LOW endpoints and peak vel πL/2T — floor
        # T so both stay inside the HW-validated tracking envelope, else the
        # commanded arc (and its audited floor clearance) is fiction.
        T_acc = float(np.pi * np.sqrt(
            path_len / (2.0 * max(PUSH_TRANSIT_ACCEL_MAX, 1e-6))
        ))
        T_vel = float(np.pi * path_len / (2.0 * max(PUSH_TRANSIT_VEL_MAX, 1e-6)))
        T_env = max(T_acc, T_vel)
        if T_env > T_dur + 1e-3:
            ctx.log.info(
                f"Arc transit slowed for tracking: {T_dur:.2f}s → {T_env:.2f}s "
                f"(path {path_len:.2f}m, accel ≤{PUSH_TRANSIT_ACCEL_MAX:.0f}m/s², "
                f"vel ≤{PUSH_TRANSIT_VEL_MAX:.1f}m/s)"
            )
            T_dur = T_env
        T_dur = max(T_dur, 4.0 / PUSH_TRANSIT_HZ)

        n = max(2, int(np.ceil(T_dur * PUSH_TRANSIT_HZ)))
        ts = np.linspace(0.0, T_dur, n + 1)
        u_t = 0.5 - 0.5 * np.cos(np.pi * ts / T_dur)     # rest→rest ease
        s_u = u_t * u_t * (3.0 - 2.0 * u_t)              # smoothstep (Slerp)
        R_all = slerp(s_u).as_matrix()

        waypoints: list[np.ndarray] = []
        q_seed = q_from
        for i in range(len(ts)):
            T_wp = np.eye(4)
            T_wp[:3, :3] = R_all[i]
            T_wp[:3, 3] = _pos_at(float(u_t[i]))
            ik = ctx.robot.inverse_kinematics(T_wp, q_init=q_seed)
            if ik is None:
                ctx.log.warn(
                    f"Arc transit IK failed at sample {i}/{len(ts) - 1} "
                    f"(u={u_t[i]:.2f}); falling back to the caller's plan-B"
                )
                return None
            q_seed = np.asarray(ik, dtype=float)
            waypoints.append(q_seed)

        traj = np.column_stack(waypoints)
        # Snap the endpoints EXACTLY onto q_from/q_to so the seams to the
        # neighbouring segments stay knot-perfect (see the docstring on why
        # the IK round-trip may differ — log, never assert).
        for idx, q_ref in ((0, q_from), (-1, q_to)):
            gap = float(np.max(np.abs(traj[:, idx] - q_ref)))
            if gap > 1e-3:
                ctx.log.warn(
                    f"Arc transit endpoint IK round-trip off by {gap:.4f} rad "
                    f"(wrist branch?); snapping to the exact endpoint"
                )
            traj[:, idx] = q_ref
        vel = (
            np.gradient(traj, ts, axis=1)
            if traj.shape[1] > 2 else np.zeros_like(traj)
        )
        vel[:, 0] = 0.0
        vel[:, -1] = 0.0
        return self._clamp_stroke_velocity(traj, vel, ts)

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
            "n_descent": meta.get("n_descent", ""),
            "n_stroke": meta.get("n_stroke", ""),
            "chain_dest": meta.get("chain_dest", ""),
        }
        self._append_csv_row(path, row, ctx.log)

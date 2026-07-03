"""Runtime configuration for the GP8 pick-and-throw app.

``Config`` holds every tunable as a dataclass field (extensive inline comments
explain each). Split out of ``app.py`` so the orchestrator stays about ROS wiring,
target selection, and the main loop, while the tunables live in one place.
Env-overridable fields read the environment at construction (``default_factory``).
Calibration/extrinsics that must match the ``camera_debug`` node live in
``perception/extrinsics.py`` and ``config/*.yaml`` — keep those in sync.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from gp8_control.planning import ThrowDecodingConfig


def _env_default(key: str, default: str) -> str:
    """Look up a config value from the process environment at import time."""
    return os.environ.get(key, default)


@dataclass
class Config:
    # Network
    ROBOT_IP: str = "192.168.255.1"

    # Workspace
    MAX_REACH: float = 0.65
    CONVEYOR_SPEED: float = 0.083
    CONVEYOR_TOPIC: str = "/conveyor/speed"
    CONVEYOR_STALE_SECONDS: float = 2.0
    TARGET_DISTANCE: float = 1.2

    # Fixed lead (s) folded into the throw-landing projection so the aim
    # accounts for belt travel during the swing (planner.plan_throw_landing).
    FIXED_DELAY_THROW: float = 0.2

    # Pick-lead convergence guard: cap on how far ahead (seconds) plan_pick
    # projects the object before aiming. Bounds the fixed-point iteration so
    # it can't diverge to the reach boundary. Set near the real pick time.
    MAX_PICK_LEAD: float = 1.2

    # Test override for the push/throw ActionSelector. Empty = normal routing
    # (per-class via SKILL_BY_CLASS: metal -> push, transparent -> throw). Set to a
    # skill name ("throw" or "push")
    # to pin EVERY object to that one skill, bypassing per-class routing and
    # the skill's can_handle() — handy for driving one skill in isolation
    # (e.g. testing the push path before it's fully wired). Prefer the CLI flag
    # `--skill push|throw` (see main()); GP8_FORCE_SKILL is the env equivalent
    # for launch files. Precedence: CLI flag > env var > "" (normal routing).
    FORCE_SKILL: str = field(
        default_factory=lambda: _env_default("GP8_FORCE_SKILL", "")
    )

    # Per-class skill routing for the ActionSelector (class_name -> skill name).
    # This is the NORMAL routing used when FORCE_SKILL is empty: cans ("metal")
    # are PUSHED off the belt, PET bottles ("transparent") are SUCTIONED and
    # thrown. Any class not listed falls back to the selector's default
    # ("throw"). FORCE_SKILL still overrides this for single-skill testing.
    # Edit the values to re-route a class; keys must match the perception's
    # class_names (metal = can, transparent = PET bottle).
    SKILL_BY_CLASS: dict = field(
        default_factory=lambda: {
            "metal": "push",          # 캔  -> push
            "transparent": "throw",   # 페트병 -> suction throw
        }
    )

    GRASP_INTERCEPT_Y: float = -0.1      # belt-frame Y where the arm waits [m]
    # Grasp height [m]: belt-surface contact Z. Manually verified pose was
    # z=0.067 (terminal_debug: EE x=0.508 y=0.000, suction ON); lowered ~5 mm
    # to 0.062 for firmer contact.
    # Overrides the often-noisy detected Z; the approach (aim) keeps its
    # relative height above this.
    GRASP_Z: float = 0.062
    # SUCTION_LEAD = 1.0 (throw tuning) won the push/throw merge; push had 0.5.
    # If push over-primes, split this per-skill instead of re-globalizing.
    SUCTION_LEAD: float = 1.0           # fire suction this many seconds before arrival [s]
    # SHARED base arrival-lead for every skill: end the WAITING block this many
    # seconds BEFORE the object's predicted arrival so the post-wait trajectory
    # dispatch overlaps the object's final approach and the action lands ON arrival
    # instead of trailing it. On the adv4ncr 250Hz stream driver the old ~0.4 s
    # point-queue re-entry is GONE, so this now only covers the ~10-20 ms command->
    # motion dispatch (HW-measured ~10 ms). A skill needing MORE lead (e.g. push must
    # also cover its stroke's retreat->contact pre-travel) adds its own by overriding
    # ManipulationSkill.arrival_lead() — see base.py. 0 = wait for full predicted
    # arrival. Default 0.02 for the stream dispatch; HW-calibrate + env-override
    # (GP8_ACTION_START_LEAD) if the lift trails/leads arrival.
    ACTION_START_LEAD: float = field(    # [s] — env GP8_ACTION_START_LEAD
        default_factory=lambda: float(os.environ.get("GP8_ACTION_START_LEAD", "0.02"))
    )
    # Fire throw-release suction_off this early to cover the WriteSingleIO
    # service round-trip + pneumatic vent lag (object releases after the
    # command is issued). Tune from the measured "IO call" latency in the log.
    # Positive = release earlier; NEGATIVE = release LATER. lead_steps =
    # round(RELEASE_LEAD * TRAJ_HZ), release_idx = eta_idx - lead_steps, so
    # -0.1 @ 20 Hz shifts release +2 waypoints (~0.1 s of trajectory time later).
    RELEASE_LEAD: float = -0.1          # [s]  (negative -> release ~0.1 s later)

    # Per-cycle timing log (suction-on -> throw start -> release). Empty = off.
    PICK_LOG_CSV: str = field(
        default_factory=lambda: os.path.expanduser("~/gp8_pick_log.csv")
    )
    # Must exceed the camera->pick travel time: belt-Y ~2.48 m at ~0.19 m/s
    # is ~13 s, so 12 s was firing ~1 s before arrival. 25 s covers slower belts.
    AMBUSH_MAX_WAIT: float = 25.0       # give up waiting for arrival after this [s]

    # Trajectory sampling / joint limit scales. Affects the post-throw chain
    # and the pre-pick _move_through (anything via trajectory()/opt_time);
    # NOT the NN-driven throw motion itself (that uses params.T / params.w).
    TRAJ_HZ: float = 20.0
    JOINT_VEL_LIMIT_SCALE: float = 0.9    # 90% of nominal joint velocity (safety margin)
    JOINT_ACCEL_LIMIT_SCALE: float = 6.0  # M2 = M1 × this (aggressive accel/decel)

    # Loop cooldown
    TIME_STEP: float = 1.0 / 25.0
    FRAME_COOLDOWN_DISTANCE: float = 0.8

    # Spatial-dedup threshold for intake. A new detection within this
    # radius of an existing tracked object is treated as the same physical
    # object (so successive camera frames re-detecting it don't enqueue
    # duplicates). 5 cm covers typical position noise.
    OBJECT_MATCH_EPSILON: float = 0.05

    # Pick-feasibility safety factor. _select_ambush_target drops queue heads
    # whose ETA < move_time * factor — i.e. objects that will reach the
    # intercept before the arm can finish positioning. opt_time is known to
    # over-estimate the real move (~2x), so this trusts the real move is only
    # a fraction of the planned one; bump higher (toward 1.0) to be more
    # conservative (drop sooner) or lower to attempt more borderline catches.
    # MERGE NOTE: the old push selection gate (factor + SUCTION_LEAD, tuned to 1.0)
    # is replaced by earliest_reachable_intercept; this factor now gates the intercept
    # feasibility for both skills (the throw return-chain that also used it was removed
    # in Option C — uniform per-object flow). Validate on push.
    PICK_FEASIBILITY_FACTOR: float = 1.05

    # opt_time -> real positioning-time calibration for the SKILL TIMELINE model
    # (skills' t_to_contact, consumed by earliest_reachable_intercept). opt_time is
    # only a proxy for the real positioning move; this scales it. Default 1.0 = use
    # opt_time as-is (conservative: do NOT shrink it — under-estimating positioning
    # is what makes push strike behind the object). Lower it (<1) only if HW logs
    # show opt_time over-estimates the real positioning move. ONLY the push timeline
    # uses it today (throw keeps PICK_FEASIBILITY_FACTOR); env-overridable for
    # re-tuning without a rebuild. See PushSkill.t_to_contact / base.py.
    OPT_TIME_TO_REAL: float = field(    # env GP8_OPT_TIME_TO_REAL
        default_factory=lambda: float(os.environ.get("GP8_OPT_TIME_TO_REAL", "1.0"))
    )

    # Throw NN post-processing (main_sam7)
    THROW_TIME_SCALE: float = 0.85
    RELEASE_EARLY_SHIFT: float = 0.0
    ETA_MIN: float = 0.13
    ETA_MAX: float = 0.95

    # Initial pose
    INITIAL_R: np.ndarray = field(default_factory=lambda: np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ]))
    INITIAL_T: np.ndarray = field(default_factory=lambda: np.array([[0.4], [0.0], [0.1]]))

    def throw_decoding(self) -> ThrowDecodingConfig:
        return ThrowDecodingConfig(
            throw_time_scale=self.THROW_TIME_SCALE,
            release_early_shift=self.RELEASE_EARLY_SHIFT,
            eta_min=self.ETA_MIN,
            eta_max=self.ETA_MAX,
        )

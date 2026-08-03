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
    CONVEYOR_DISTANCE_TOPIC: str = "/conveyor/distance_mm"
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
    GRASP_Z: float = field(             # [m] env GP8_GRASP_Z / launch grasp_z:=
        default_factory=lambda: float(os.environ.get("GP8_GRASP_Z", "0.062"))
    )
    # --- Throw pick: belt-tracking descend (PickWaitMode.TRACK_DESCEND) -------
    # The throw pick no longer waits PARKED at the grasp. It parks at TRACK_Z_START
    # (above the belt), and on the object's arrival runs one cartesian segment that
    # FOLLOWS the object downstream (-Y at belt speed, zero relative velocity) while
    # the TCP Z ramps TRACK_Z_START -> TRACK_Z_END at TRACK_Z_SPEED. The cup therefore
    # settles onto a co-moving object instead of dropping onto one that is sliding
    # underneath it. The throw then starts from wherever that segment ended (further
    # downstream and lower than the nominal grasp).
    # All three are absolute base-frame quantities, env/launch/CLI overridable so they
    # can be swept per run without a rebuild:
    #   GP8_TRACK_Z_START / track_z_start:= / --track-z-start   [m, absolute TCP Z]
    #   GP8_TRACK_Z_END   / track_z_end:=   / --track-z-end     [m, absolute TCP Z]
    #   GP8_TRACK_Z_SPEED / track_z_speed:= / --track-z-speed   [m/s descent rate]
    # NaN (the default) means "derive from GRASP_Z": start = GRASP_Z + TRACK_Z_HOVER,
    # end = GRASP_Z. So an un-flagged run descends the hover height onto the normal
    # grasp plane. TRACK_Z_SPEED <= 0 DISABLES tracking and restores the old parked
    # WAIT_AT_GRASP pick.
    # Defaults below are the operator's current run configuration
    # (start 0.12 / end 0.05 / speed 0.3). The earlier HW-tracked set at belt
    # 0.223 m/s was end 0.03 / speed 0.2 — restore those if the cup starts
    # missing rather than clearing. Pass "nan" to restore the derive-from-
    # GRASP_Z behaviour (start = GRASP_Z + TRACK_Z_HOVER, end = GRASP_Z).
    # Mirrored in gp8_bringup.launch.py's track_z_* args — keep both in sync.
    TRACK_Z_START: float = field(       # [m] env GP8_TRACK_Z_START ("nan" -> GRASP_Z + TRACK_Z_HOVER)
        default_factory=lambda: float(os.environ.get("GP8_TRACK_Z_START", "0.12"))
    )
    TRACK_Z_END: float = field(         # [m] env GP8_TRACK_Z_END ("nan" -> GRASP_Z)
        default_factory=lambda: float(os.environ.get("GP8_TRACK_Z_END", "0.05"))
    )
    TRACK_Z_SPEED: float = field(       # [m/s] env GP8_TRACK_Z_SPEED (<=0 disables tracking)
        default_factory=lambda: float(os.environ.get("GP8_TRACK_Z_SPEED", "0.3"))
    )
    # Hover height above GRASP_Z used when TRACK_Z_START is left at NaN. Also the
    # clearance the parked cup keeps over an approaching object before the descend.
    TRACK_Z_HOVER: float = 0.05         # [m]
    # Extra lead added to the TRACK_DESCEND arrival_lead: start the follow+descend
    # this many seconds EARLIER. The follow is open-loop parallel tracking at belt
    # speed, so any lag in when it PHYSICALLY starts (dispatch/settle latency)
    # persists as a fixed downstream offset — the cup lands that far BEHIND the
    # object. Dialing this up starts the descend earlier and cancels that offset.
    # Empirical knob: observed miss d[m] at belt v[m/s] ~= TRACK_LEAD_T*v, so start
    # with TRACK_LEAD_T ~= d/v (e.g. 5 cm behind at 0.10 m/s -> ~0.5). Positive =
    # earlier; too large lands the cup ahead of a late object. Only applied in
    # TRACK_DESCEND (see ThrowSkill.arrival_lead). env/launch/CLI overridable.
    # Default 0.0: perception capture-to-receipt latency is now compensated
    # live by camera_debug. Keep this only as an explicit empirical override.
    #   GP8_TRACK_LEAD_T / track_lead_t:= / --track-lead-t   [s]
    TRACK_LEAD_T: float = field(        # [s] env GP8_TRACK_LEAD_T
        default_factory=lambda: float(os.environ.get("GP8_TRACK_LEAD_T", "0.0"))
    )
    GRASP_Z: float = 0.062
    # Baseline wrist (joint 6, rad) for EVERY pick/aim/park pose. The suction
    # cup is axially symmetric, and the throw NN drives joints 1-5 only, so J6
    # is a FREE DOF for pick/throw — but it is NOT free for push, whose paddle
    # must face the push heading. Historically this baseline was 0 (neutral),
    # which made every push cycle swing J6 ~-30..-97 deg to the push-facing
    # wrist and back. Set to the MEAN push wait-pose J6 over the belt for the
    # metal-bin geometry (IK sweep 2026-07-06: -30..-97 deg, mean -65 deg), so
    # the paddle parks roughly facing the push direction and both push->push
    # and throw->push transitions rotate J6 by <=~33 deg instead of ~90.
    # Recompute if the push bins move (see PUSH_BIN_TARGET_MAP).
    PICK_WRIST_J6: float = np.radians(-65.0)
    # CAP on how early suction primes, now that priming is POSITION-triggered
    # (position_and_prime fires at max(cup-parked, arrival - SUCTION_LEAD)). The cup
    # is always parked at the grasp before suction fires; this only bounds how far
    # ahead of arrival a SLACK pick (e.g. the first) primes. 0.5 s per request; a
    # backed-up pick primes as soon as the cup parks regardless of this value.
    SUCTION_LEAD: float = 0.5           # max seconds before arrival to prime suction [s]
    # GUARANTEED parked vacuum-forming hold: the throw pick's t_to_contact adds this
    # to the positioning budget, so earliest_reachable_intercept places the grasp far
    # enough DOWNSTREAM that the arm reaches it >= this many seconds before the object
    # arrives — the vacuum then seals during that parked wait (which was ~0 for
    # backed-up 2nd+ objects, the miss cause). An object that can't be caught that far
    # downstream (obj_y_arrival < -y_b) is DROPPED rather than grabbed with no hold.
    # Keep <= SUCTION_LEAD - ACTION_START_LEAD so suction actually fires at park (else
    # the SUCTION_LEAD cap fires it later). Env GP8_MIN_SUCTION_HOLD / launch
    # min_suction_hold:=. 0 = old behaviour (grab at earliest reachable, no hold).
    MIN_SUCTION_HOLD: float = field(    # [s] — env GP8_MIN_SUCTION_HOLD
        default_factory=lambda: float(os.environ.get("GP8_MIN_SUCTION_HOLD", "0.3"))
    )
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
    # round(RELEASE_LEAD * TRAJ_HZ), release_idx = eta_idx - lead_steps, so at
    # the current TRAJ_HZ=50 each 0.02 s shifts the release by one waypoint
    # (-0.1 would shift it +5 waypoints, i.e. 0.1 s LATER).
    # Default 0.0 = release exactly at the NN's eta waypoint, no correction.
    # Env-overridable (GP8_RELEASE_LEAD) so it can be tuned without a rebuild;
    # the bringup launch exposes it as `release_lead:=<value>`. The RViz preview
    # (rviz:=true) logs a recommended value per throw — see ThrowVisualizer.
    # Mirrored in gp8_bringup.launch.py's release_lead arg — keep both in sync.
    RELEASE_LEAD: float = field(        # [s]  env GP8_RELEASE_LEAD / launch release_lead:=
        default_factory=lambda: float(os.environ.get("GP8_RELEASE_LEAD", "0.0"))
    )
    # Z plane used by the runtime RViz point-mass ballistic preview.  This does
    # not affect motion or release control; it only defines where the displayed
    # object trajectory is considered to land.
    THROW_VIZ_IMPACT_Z: float = field(  # [m] env GP8_THROW_VIZ_IMPACT_Z
        default_factory=lambda: float(os.environ.get("GP8_THROW_VIZ_IMPACT_Z", "0.0"))
    )
    # Runtime ballistic evaluation target.  The XY defaults match the throw
    # heading target that was historically hard-coded in throw_skill.py.
    # GOAL_RADIUS is the acceptable horizontal miss distance (roughly the bin
    # opening radius); these values affect planning direction/evaluation only,
    # never the robot's safety limits.
    THROW_GOAL_X: float = field(
        default_factory=lambda: float(os.environ.get("GP8_THROW_GOAL_X", "1.1"))
    )
    THROW_GOAL_Y: float = field(
        default_factory=lambda: float(os.environ.get("GP8_THROW_GOAL_Y", "-0.25"))
    )
    THROW_GOAL_RADIUS: float = field(
        default_factory=lambda: float(os.environ.get("GP8_THROW_GOAL_RADIUS", "0.10"))
    )
    # Optional JSON list of throw bins. Empty keeps the existing single-goal
    # behavior; launch exposes this as ``throw_bins:=...``.
    THROW_BINS: str = field(
        default_factory=lambda: os.environ.get("GP8_THROW_BINS", "")
    )

    # Per-cycle timing log (suction-on -> throw start -> release). Empty = off.
    PICK_LOG_CSV: str = field(
        default_factory=lambda: os.path.expanduser("~/gp8_pick_log.csv")
    )
    # Must exceed the camera->pick travel time: belt-Y ~2.48 m at ~0.19 m/s
    # is ~13 s, so 12 s was firing ~1 s before arrival. 25 s covers slower belts.
    AMBUSH_MAX_WAIT: float = 25.0       # give up waiting for arrival after this [s]

    # Trajectory sampling / joint limit scales. Affects the post-throw chain
    # and the pre-pick _move_through (anything via trajectory()/opt_time), and
    # the RobustThrow NLP arc sampling + release-index granularity.
    # 20 -> 50 Hz (2026-07-24 사용자): throw arc(t_f~0.4-0.6s)가 50ms 간격이면
    # 8~12점뿐이고 release 타이밍 granularity도 50ms라 거칠었다. 50Hz면 arc가
    # 2.5배 촘촘하고 release가 20ms 단위로 정밀. 스트림이 250Hz로 리샘플하므로
    # 실제 모션 부드러움은 이미 250Hz — 여긴 리샘플 전 밀도/타이밍 해상도용.
    # warm DB(연속 B-spline)와는 무관 (샘플링만 바뀜, 재빌드 불필요).
    TRAJ_HZ: float = 50.0
    JOINT_VEL_LIMIT_SCALE: float = 0.9    # 90% of nominal joint velocity (safety margin)
    JOINT_ACCEL_LIMIT_SCALE: float = 6.0  # M2 = M1 × this (aggressive accel/decel)

    # Loop cooldown
    TIME_STEP: float = 1.0 / 25.0
    FRAME_COOLDOWN_DISTANCE: float = 0.85

    # Spatial-dedup threshold for intake. A new detection within this
    # radius of an existing tracked object is treated as the same physical
    # object (so successive camera frames re-detecting it don't enqueue
    # duplicates). 5 cm covers typical position noise.
    OBJECT_MATCH_EPSILON: float = 0.05

    # Belt-direction (Y) dedup tolerance GROWTH per second of dead reckoning,
    # as a fraction of belt speed. A track that hasn't been re-detected for
    # `age` seconds has been extrapolated by v*age, and the belt-speed estimate
    # is only good to a few percent — so its predicted Y is uncertain by
    # ~OBJECT_MATCH_DRIFT_FRAC * v * age. Without this the fixed 5 cm window is
    # exceeded whenever the main loop is busy dispatching a multi-second
    # trajectory (no intake runs during it), and the SAME object re-spawns as a
    # second track at its true position. Across-belt (X) needs no growth term —
    # the object doesn't drift sideways. 0.25 = tolerate a 25% belt-speed error.
    OBJECT_MATCH_DRIFT_FRAC: float = 0.25
    # Hard cap on that grown Y window [m], so a very stale track can't swallow a
    # genuinely different object further down the belt. Generous is safe here:
    # the first detection re-anchors the stale track (age -> 0), so every OTHER
    # detection in the same frame is matched against the tight base window —
    # two objects in one frame stay separate as long as they are > eps apart.
    OBJECT_MATCH_EPS_Y_MAX: float = 0.20
    # Cap for the queue MERGE pass, which is deliberately TIGHTER than the
    # intake cap above. Intake compares a fresh camera detection against a
    # prediction (the detection is ground truth, so a wide window is safe);
    # merging compares two PREDICTIONS with no new evidence, so a wide window
    # there would delete a genuinely separate object. 10 cm still catches a
    # drift-spawned twin (~8 cm after a 4 s loop stall) while keeping objects
    # spaced a normal belt gap apart distinct.
    OBJECT_MERGE_EPS_Y_MAX: float = 0.10

    # 검출↔트랙 연관 방식. "hungarian"(기본): 프레임의 검출 전체×트랙 전체의
    # 비용(창-정규화 거리 합 = "전체 거리")을 scipy 최적 할당으로 한 번에
    # 최소화 — 허용창이 겹칠 만큼 붙어 오는 이웃 물체들의 트랙 교차(스왑)를
    # 방지한다. "greedy": 구 선착순 매칭 (검출마다 창 안 첫 트랙). 두 방식의
    # 게이트(창 밖 = 매칭 불가)는 동일하므로, 물체 간격이 창보다 넓으면 결과도
    # 동일하다. env GP8_TRACK_ASSOC.
    TRACK_ASSOC: str = field(
        default_factory=lambda: _env_default("GP8_TRACK_ASSOC", "hungarian")
    )

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

"""GP8 pick-and-throw main application (ROS 2).

Ported from ROS 1 ``my_gp8_control/src/main_sam7.py`` (류가은 branch)
but running on MotoROS2 + name bridge + ROS 2 Humble. The orchestrator
itself stays small — domain logic lives in ``perception/``,
``tracking/``, ``planning/``, ``controllers/``, and the manipulation
``skills/`` (throw, push). GP8App selects a target each epoch, asks the
``ActionSelector`` which skill should handle it, and lets that skill run.
"""

from __future__ import annotations

import sys
import time
import json
import os
from dataclasses import dataclass, field

from std_msgs.msg import String

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from gp8_control.controllers.trajectory_controller import TrajectoryController
from gp8_control.controllers.moveit_controller import MoveItController
from gp8_control.controllers.pick_delay_tracker import PickDelayTracker
from gp8_control.perception.stream_detection_source import StreamDetectionSource
from gp8_control.perception.conveyor_speed import ConveyorSpeedTracker
from gp8_control.perception.detection_intake import DetectionIntake
from gp8_control.perception import extrinsics as _extrinsics
from gp8_control.trajectory.trajectory_primitive import (
    trajectory,
    trajectory_3points,
    opt_time,
)
from gp8_control.trajectory.predictor import TrajectoryPredictor
from gp8_control.tracking import (
    TrackedObject,
    TrackedObjectQueue,
    FrameGate,
)
from gp8_control.planning import (
    PickThrowPlanner,
    ThrowDecodingConfig,
    TargetStatus,
    lock_or_drop_head,
)
from gp8_control.planning.action_selector import ActionSelector
from gp8_control.robots.gp8 import GP8
from gp8_control.skills import (
    SkillContext,
    PickRequest,
    ThrowSkill,
    PushSkill,
)
from gp8_control.skills.throw_skill import THETA_MAP


# Tool orientation used to assemble grasp/aim 4x4 from the corrected base
# position published by the camera_debug node (which owns the camera→base
# transform, Z offsets, and v*delay back-projection). Must match the value
# camera_debug uses (kept identical to the legacy DetectionIntake default).
_R_GRASP_DEFAULT = np.array(
    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
)


def _make_transform(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float).ravel()
    return T


# =========================================================================
# Configuration
# =========================================================================

def _env_default(key: str, default: str) -> str:
    """Look up a config value from the process environment at import time."""
    import os as _os
    return _os.environ.get(key, default)


@dataclass
class Config:
    # Network
    ROBOT_IP: str = "192.168.255.1"
    # Perception is consumed from the camera PC's HTTP NDJSON stream
    # (see perception/perception_client.py for the wire contract).
    PERCEPTION_URL: str = field(
        default_factory=lambda: _env_default(
            "GP8_PERCEPTION_URL", "http://147.46.175.15:8080/detections/stream"
        )
    )
    PERCEPTION_RECONNECT_DELAY: float = 2.0

    # Workspace
    MAX_REACH: float = 0.65
    CONVEYOR_SPEED: float = 0.083
    CONVEYOR_TOPIC: str = "/conveyor/speed"
    CONVEYOR_STALE_SECONDS: float = 2.0
    TARGET_DISTANCE: float = 1.2

    # Detection (Z offsets shared with camera_debug via perception.extrinsics)
    DETECTION_OFFSET_AIM: float = _extrinsics.DETECTION_OFFSET_AIM
    DETECTION_OFFSET_GRASP: float = _extrinsics.DETECTION_OFFSET_GRASP

    # Pick-cycle delay starts at 0 (first pick uncompensated), then the
    # first observed overhead is adopted as-is, later picks EMA-smooth.
    FIXED_DELAY_THROW: float = 0.2
    DELAY_EMA_ALPHA: float = 0.3

    # Pick-lead convergence guard: cap on how far ahead (seconds) plan_pick
    # projects the object before aiming. Bounds the fixed-point iteration so
    # it can't diverge to the reach boundary. Set near the real pick time.
    MAX_PICK_LEAD: float = 1.2

    # Pick strategy:
    #   "ambush" — park the arm at a fixed intercept line (GRASP_INTERCEPT_Y)
    #              ahead of time and fire suction when the object arrives.
    #              Avoids moving-intercept lead timing entirely.
    #   "moving" — legacy predictive-intercept pick (plan_pick + lock).
    PICK_STRATEGY: str = "ambush"

    # Test override for the push/throw ActionSelector. Empty = normal routing
    # (every object -> throw today). Set to a skill name ("throw" or "push")
    # to pin EVERY object to that one skill, bypassing per-class routing and
    # the skill's can_handle() — handy for driving one skill in isolation
    # (e.g. testing the push path before it's fully wired). Prefer the CLI flag
    # `--skill push|throw` (see main()); GP8_FORCE_SKILL is the env equivalent
    # for launch files. Precedence: CLI flag > env var > "" (normal routing).
    FORCE_SKILL: str = field(
        default_factory=lambda: _env_default("GP8_FORCE_SKILL", "")
    )

    GRASP_INTERCEPT_Y: float = 0.0      # belt-frame Y where the arm waits [m]
    # Grasp height [m]: belt-surface contact Z. Manually verified pose was
    # z=0.067 (terminal_debug: EE x=0.508 y=0.000, suction ON); lowered ~5 mm
    # to 0.062 for firmer contact.
    # Overrides the often-noisy detected Z; the approach (aim) keeps its
    # relative height above this.
    GRASP_Z: float = 0.062
    SUCTION_LEAD: float = 1.0           # fire suction this many seconds before arrival [s]
    # Fire throw-release suction_off this early to cover the WriteSingleIO
    # service round-trip + pneumatic vent lag (object releases after the
    # command is issued). Tune from the measured "IO call" latency in the log.
    RELEASE_LEAD: float = 0.0           # [s]

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
    # over-estimate the real move (~2x), so 0.5 trusts that the real move
    # is roughly half the planned one; bump higher (toward 1.0) to be more
    # conservative (drop sooner) or lower to attempt more catches.
    PICK_FEASIBILITY_FACTOR: float = 0.5

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

    # Fixed extrinsics (shared with camera_debug via perception.extrinsics)
    T_ROBOT2BASE: np.ndarray = field(
        default_factory=lambda: _extrinsics.T_ROBOT2BASE.copy()
    )
    T_BASE2CAM: np.ndarray = field(
        default_factory=lambda: _extrinsics.T_BASE2CAM.copy()
    )

    def throw_decoding(self) -> ThrowDecodingConfig:
        return ThrowDecodingConfig(
            throw_time_scale=self.THROW_TIME_SCALE,
            release_early_shift=self.RELEASE_EARLY_SHIFT,
            eta_min=self.ETA_MIN,
            eta_max=self.ETA_MAX,
        )


# =========================================================================
# Main application
# =========================================================================

class GP8App:
    """Pick-and-throw orchestrator with main_sam7-style tracking.

    Keeps only orchestration: ROS wiring, perception/tracking intake, target
    selection, and the main loop. The manipulation itself lives in
    ``gp8_control.skills`` — each epoch GP8App selects a target, asks the
    ``ActionSelector`` which skill handles it, and runs that skill.
    """

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()
        self.robot = GP8()

        # Belt-state viz publishing (see _publish_belt_state).
        self._status: str = "IDLE"
        self._status_detail: str = ""
        self._belt_state_pub = None
        # The target currently being executed (popped from the queue) — kept
        # here so the viz can keep drawing it while the cycle runs.
        self._active_target: TrackedObject | None = None

        # Latest corrected detection snapshot from the camera_debug node.
        # See /camera_debug/detections — camera_debug owns the perception
        # stream + cam→base transform + v*delay back-projection.
        self._cam_latest: dict | None = None

        self._node: Node | None = None
        self._executor: MultiThreadedExecutor | None = None
        self.traj_ctrl: TrajectoryController | None = None
        self.moveit_ctrl: MoveItController | None = None
        self.detection_source: StreamDetectionSource | None = None
        self.conveyor: ConveyorSpeedTracker | None = None
        self.intake: DetectionIntake | None = None

        self.M1: np.ndarray | None = None
        self.M2: np.ndarray | None = None
        self.predictor: TrajectoryPredictor | None = None
        self.planner: PickThrowPlanner | None = None

        # Manipulation skills + their shared context + the push/throw selector.
        # Built in setup() once the robot resources exist (see _build_skills).
        self.ctx: SkillContext | None = None
        self.throw_skill: ThrowSkill | None = None
        self.push_skill: PushSkill | None = None
        self.selector: ActionSelector | None = None

        # Drop tracked objects whose extrapolated y has fallen past the
        # ambush intercept line — those are already past the robot and
        # uncatchable; the head stays "next-front still in front of the pick".
        self.queue = TrackedObjectQueue(
            self.cfg.MAX_REACH, drop_below_y=self.cfg.GRASP_INTERCEPT_Y,
        )
        self.frame_gate = FrameGate(self.cfg.FRAME_COOLDOWN_DISTANCE)
        self.pick_delay = PickDelayTracker(self.cfg.DELAY_EMA_ALPHA)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup(self) -> None:
        rclpy.init()
        self._node = Node("gp8_manager")
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self._node)

        # Live belt-state stream for the belt_viz TUI (separate node).
        self._belt_state_pub = self._node.create_publisher(
            String, "/gp8_manager/tracked_state", 10
        )

        self.traj_ctrl = TrajectoryController(self._node)
        self.moveit_ctrl = MoveItController(self._node)
        # camera_debug node owns the perception stream + corrections; we just
        # subscribe to its corrected detection list.
        self._node.create_subscription(
            String, "/camera_debug/detections",
            self._on_camera_debug_detections, 10,
        )
        self.conveyor = ConveyorSpeedTracker(
            self._node,
            self.cfg.CONVEYOR_TOPIC,
            self.cfg.CONVEYOR_SPEED,
            self.cfg.CONVEYOR_STALE_SECONDS,
        )

        self.traj_ctrl.wait_for_servers()
        self._load_predictor()
        self._setup_joint_limits()
        self._build_planner()
        self._build_skills()
        self._enable_robot()
        # Always start from a known-off suction state.
        self._node.get_logger().info("Forcing suction OFF at startup.")
        self.traj_ctrl.suction_off()
        self._move_to_initial_pose()
        # main_sam7 trusts the hardcoded extrinsic. Perception now runs on the
        # camera PC (HTTP stream); no local AprilTag/RealSense handshake here.
        self._node.get_logger().info(
            "Using hardcoded T_base2cam (no AprilTag handshake)."
        )

    def _load_predictor(self) -> None:
        self.predictor = TrajectoryPredictor()
        self._node.get_logger().info(
            f"Trajectory NN loaded: {self.predictor.weight_path}"
        )

    def _setup_joint_limits(self) -> None:
        # GP8.velocity_limits is a numpy array of max joint speeds (rad/s)
        self.M1 = np.asarray(self.robot.velocity_limits, dtype=float) * self.cfg.JOINT_VEL_LIMIT_SCALE
        self.M2 = self.M1 * self.cfg.JOINT_ACCEL_LIMIT_SCALE

    def _build_planner(self) -> None:
        self.planner = PickThrowPlanner(
            robot=self.robot,
            predictor=self.predictor,
            M1=self.M1,
            M2=self.M2,
            max_reach=self.cfg.MAX_REACH,
            target_distance=self.cfg.TARGET_DISTANCE,
            decoding=self.cfg.throw_decoding(),
            max_pick_lead=self.cfg.MAX_PICK_LEAD,
        )

    def _build_skills(self) -> None:
        """Wire the shared SkillContext, the manipulation skills, and the selector.

        GP8App holds only orchestration; each manipulation lives in its own
        skill module under ``gp8_control.skills``. Adding e.g. PushSkill is a
        new file plus one entry in the selector list here — no other app.py
        changes — which keeps push/throw work from colliding.
        """
        self.ctx = SkillContext(
            cfg=self.cfg,
            node=self._node,
            robot=self.robot,
            traj_ctrl=self.traj_ctrl,
            planner=self.planner,
            conveyor=self.conveyor,
            queue=self.queue,
            M1=self.M1,
            M2=self.M2,
            intake=self._intake_new_detections,
            publish_state=self._publish_belt_state,
            set_status=self._set_status,
            set_active_target=self._set_active_target,
        )
        self.throw_skill = ThrowSkill(self.ctx)
        self.push_skill = PushSkill(self.ctx)
        # Rule-based for now (always throw); swap this for an RL policy later
        # to choose push vs throw per object. by_class can override per class.
        # FORCE_SKILL (env GP8_FORCE_SKILL) pins every object to one skill for
        # testing; empty -> normal routing.
        force = self.cfg.FORCE_SKILL or None
        if force is not None:
            self._node.get_logger().warn(
                f"ActionSelector FORCED to '{force}' skill for ALL objects "
                f"(GP8_FORCE_SKILL). Disable for normal push/throw routing."
            )
        self.selector = ActionSelector(
            [self.throw_skill, self.push_skill], default="throw", force=force,
        )

    def _build_intake(self) -> None:
        # node=None: the stream source is fed by its own background thread,
        # so poll() reads the latest snapshot without pumping ROS callbacks.
        self.intake = DetectionIntake(
            node=None,
            sam_client=self.detection_source,
            T_robot2base=self.cfg.T_ROBOT2BASE,
            T_base2cam=self.cfg.T_BASE2CAM.copy(),
            offset_aim=self.cfg.DETECTION_OFFSET_AIM,
            offset_grasp=self.cfg.DETECTION_OFFSET_GRASP,
            time_step=self.cfg.TIME_STEP,
            logger=self._node.get_logger(),
            log_raw=False,   # raw cam positions go to belt_viz, not the bringup log
        )

    def _enable_robot(self) -> None:
        """Enable robot in Point Queue Mode (FJT의 INIT_TRAJ_INVALID_STARTING_POS 회피)."""
        self._node.get_logger().info("Enabling robot (point queue mode)...")
        if not self.traj_ctrl.enter_queue_mode():
            raise RuntimeError(
                "Failed to enter point queue mode. Check pendant is in REMOTE "
                "mode with no active alarm and cycle mode AUTO."
            )
        time.sleep(1.0)

    def _move_to_initial_pose(self) -> None:
        # ros2 launch ExecuteProcess does not forward stdin; only prompt
        # when running interactively.
        if sys.stdin.isatty():
            try:
                input("Press Enter to move to initial pose (Ctrl+C to abort).")
            except EOFError:
                pass
        else:
            self._node.get_logger().warn(
                "Auto-proceeding to initial pose in 3s (stdin not a TTY, "
                "likely running under ros2 launch). Ctrl+C to abort."
            )
            time.sleep(3.0)
        initial_T = _make_transform(self.cfg.INITIAL_R, self.cfg.INITIAL_T)
        initial_joint = self.robot.inverse_kinematics(initial_T)
        if initial_joint is None:
            raise RuntimeError("IK failed for initial pose.")

        # `time.sleep` alone does not let ROS 2 callbacks fire — must spin.
        last_log = 0.0
        while (self.traj_ctrl.current_joints is None) and rclpy.ok():
            rclpy.spin_once(self._node, timeout_sec=0.1)
            now = time.time()
            if now - last_log > 1.0:
                self._node.get_logger().info("Waiting for joint states...")
                last_log = now

        current_joint = np.array(self.traj_ctrl.current_joints)
        initial_joint = np.array(initial_joint)
        initial_joint[-1] = 0.0
        zero = np.zeros_like(self.M1)
        traj, vel, timestep = trajectory(
            current_joint, zero,
            initial_joint, zero,
            self.M1, self.M2,
            hertz=self.cfg.TRAJ_HZ,
        )
        self.traj_ctrl.send_trajectory_queue(
            traj, vel, timestep, final_joint=initial_joint
        )
        time.sleep(1.0)

    # ------------------------------------------------------------------
    # Pick execution (legacy "moving" strategy)
    # ------------------------------------------------------------------
    def _execute_pick(self, current_joint, aim_joint, grasp_joint, plan_time) -> None:
        """Send pick trajectory and update PickDelayTracker with measured overhead.

        ``plan_time`` is the epoch's ``now`` — the instant the target position
        was predicted. Logging ``t_start - plan_time`` exposes the planning/IK
        compute latency that is *not* folded into ``fixed_delay`` (which only
        measures from ``t_start`` onward), the prime suspect for a consistent
        downstream pick offset.
        """
        t_start = time.time()
        zero = np.zeros_like(self.M1)
        traj, vel, ts = trajectory_3points(
            current_joint, zero, aim_joint, zero, grasp_joint, zero,
            self.M1, self.M2, hertz=self.cfg.TRAJ_HZ,
        )
        traj_duration = float(np.sum(ts))
        # Pick: ROS1 customcontroller와 동일하게 grasp pose 직전(diff<0.05)에
        # 미리 suction_on 발사 — 공압 지연 보정. 도착 후 별도 attach wait 불필요.
        self.traj_ctrl.send_trajectory_queue_with_attach(
            traj, vel, ts,
            final_joint=grasp_joint,
            attach_target_joint=grasp_joint,
        )
        elapsed = time.time() - t_start

        observed_overhead = elapsed - traj_duration
        compute_latency = t_start - plan_time           # now -> traj send (Δc), uncompensated
        prev = self.pick_delay.value                    # fixed_delay actually used this pick
        self.pick_delay.update(observed_overhead)

        v = self.conveyor.current
        predicted_lead = traj_duration + prev           # what the planner aimed with
        actual_lead = compute_latency + elapsed         # now -> grasp arrival
        shortfall = actual_lead - predicted_lead        # >0 => arm arrives downstream (late)
        self._node.get_logger().info(
            "Pick timing diagnostics:\n"
            f"  belt speed       : {v:7.3f} m/s\n"
            f"  predicted traj   : {traj_duration*1000:7.0f} ms\n"
            f"  actual traj+oh   : {elapsed*1000:7.0f} ms   (overhead {observed_overhead*1000:+.0f} ms)\n"
            f"  compute lag Δc   : {compute_latency*1000:7.0f} ms   (now->send; NOT in fixed_delay)\n"
            f"  fixed_delay used : {prev*1000:7.0f} ms   -> next {self.pick_delay.value*1000:.0f} ms\n"
            f"  predicted lead   : {predicted_lead*1000:7.0f} ms   (traj + fixed_delay)\n"
            f"  actual lead      : {actual_lead*1000:7.0f} ms   (now -> grasp arrival)\n"
            f"  => shortfall     : {shortfall*1000:+7.0f} ms = {v*shortfall*1000:+.1f} mm downstream"
        )

    # ------------------------------------------------------------------
    # Target selection (ambush strategy)
    # ------------------------------------------------------------------
    def _select_ambush_target(
        self, now: float, current_joint: np.ndarray
    ) -> PickRequest | None:
        """Choose the feasible queue head to intercept at GRASP_INTERCEPT_Y.

        Walks the queue from the head, dropping anything we can't actually
        catch (already past the pick line, out of reach, IK fails, or the
        object will pass the intercept before the arm finishes positioning).
        Returns a ``PickRequest`` (target + intercept geometry) for the first
        feasible head, or ``None`` if no head is catchable this epoch. The
        selected head is popped and recorded as the active target.
        """
        v = self.conveyor.current
        intercept_y = self.cfg.GRASP_INTERCEPT_Y
        zero = np.zeros_like(self.M1)
        factor = self.cfg.PICK_FEASIBILITY_FACTOR

        target = None
        target_T_aim = target_T_grasp = None
        target_aim_joint = target_grasp_joint = None
        target_obj_y = target_move_time = target_eta = 0.0

        while self.queue:
            candidate = self.queue.head()
            obj_y = candidate.T_grasp_base[1, 3] - v * (now - candidate.detect_time)

            # already past the pick line → drop
            if obj_y <= intercept_y:
                self.queue.pop_head()
                self._node.get_logger().info(
                    f"Drop {candidate.class_name}: already past intercept "
                    f"(y={obj_y:+.3f} <= {intercept_y:+.3f})"
                )
                continue

            # build intercept pose: detected X, intercept Y, GRASP_Z height
            T_grasp = candidate.T_grasp_base.copy()
            T_aim = candidate.T_aim_base.copy()
            approach_dz = T_aim[2, 3] - T_grasp[2, 3]
            T_grasp[1, 3] = intercept_y
            T_aim[1, 3] = intercept_y
            T_grasp[2, 3] = self.cfg.GRASP_Z
            T_aim[2, 3] = self.cfg.GRASP_Z + approach_dz

            if np.linalg.norm(T_grasp[:2, 3]) > self.cfg.MAX_REACH:
                self.queue.pop_head()
                self._node.get_logger().info(
                    f"Drop {candidate.class_name}: intercept pose out of reach"
                )
                continue

            aim_joint = self.robot.inverse_kinematics(T_aim)
            grasp_joint = self.robot.inverse_kinematics(T_grasp)
            if aim_joint is None or grasp_joint is None:
                self.queue.pop_head()
                self._node.get_logger().warn(
                    f"Drop {candidate.class_name}: IK failed"
                )
                continue
            aim_joint = np.asarray(aim_joint, dtype=float); aim_joint[-1] = 0.0
            grasp_joint = np.asarray(grasp_joint, dtype=float); grasp_joint[-1] = 0.0

            move_time = (
                opt_time(current_joint, zero, aim_joint, zero, self.M1, self.M2)
                + opt_time(aim_joint, zero, grasp_joint, zero, self.M1, self.M2)
            )
            eta = (obj_y - intercept_y) / (v + 1e-6)

            # Feasibility: the arm must be parked at the intercept by the time
            # suction fires (= eta - SUCTION_LEAD), not just by the time the
            # object actually arrives. Drop heads we can't position in time.
            # opt_time over-estimates the real move (~2x), so scale by
            # PICK_FEASIBILITY_FACTOR (default 0.5).
            needed = move_time * factor + self.cfg.SUCTION_LEAD
            if eta < needed:
                self.queue.pop_head()
                self._node.get_logger().info(
                    f"Drop {candidate.class_name}: too late to catch "
                    f"(eta {eta:.2f}s < move {move_time:.2f}s × {factor:.2f} "
                    f"+ suction lead {self.cfg.SUCTION_LEAD:.2f}s = {needed:.2f}s)"
                )
                continue

            # Feasible — keep this as the target and stop scanning.
            target = candidate
            target_T_aim, target_T_grasp = T_aim, T_grasp
            target_aim_joint, target_grasp_joint = aim_joint, grasp_joint
            target_obj_y, target_move_time, target_eta = obj_y, move_time, eta
            break

        if target is None:
            return None  # no feasible head in the queue this epoch

        # Commit to the pick.
        secondary = self.queue.peek_next() if self.queue.has_next() else None
        self.queue.pop_head()
        # Keep the active target visible in belt_viz while we execute the cycle.
        self._active_target = target
        self._node.get_logger().info(
            f"Ambush lock: {target.class_name} @ x={target_T_grasp[0, 3]:+.3f} "
            f"y={intercept_y:.3f} z={target_T_grasp[2, 3]:+.3f} "
            f"(detected y={target_obj_y:+.3f}; eta {target_eta:.2f}s, "
            f"move {target_move_time:.2f}s)"
        )
        return PickRequest(
            target=target,
            current_joint=current_joint,
            T_aim=target_T_aim,
            T_grasp=target_T_grasp,
            aim_joint=target_aim_joint,
            grasp_joint=target_grasp_joint,
            secondary=secondary,
        )

    # ------------------------------------------------------------------
    # Belt-state viz (publishes to /gp8_manager/tracked_state for belt_viz)
    # ------------------------------------------------------------------
    def _set_active_target(self, target: TrackedObject | None) -> None:
        """Set the in-flight target (kept for belt_viz + intake dedup).

        Wired into the SkillContext so a skill can clear it when its cycle
        finishes without reaching back into GP8App.
        """
        self._active_target = target

    def _set_status(self, status: str, detail: str = "") -> None:
        self._status = status
        self._status_detail = detail
        self._publish_belt_state()

    def _publish_belt_state(self) -> None:
        if self._belt_state_pub is None:
            return
        now = time.time()
        v = self.conveyor.current if self.conveyor is not None else 0.0
        objs = []

        def _serialize(obj: TrackedObject, is_target: bool) -> dict:
            y_now = float(obj.T_grasp_base[1, 3] - v * (now - obj.detect_time))
            cam = obj.cam_pos
            return {
                "class": obj.class_name,
                "y_now": y_now,
                "x": float(obj.T_grasp_base[0, 3]),
                "z": float(obj.T_grasp_base[2, 3]),
                "age_s": float(now - obj.detect_time),
                "is_target": is_target,
                "cam": list(cam) if cam is not None else None,
            }

        # Currently-executing target (popped from the queue but still on the belt).
        if self._active_target is not None:
            objs.append(_serialize(self._active_target, True))

        q = getattr(self, "queue", None)
        if q is not None:
            for obj in list(q._objects):
                objs.append(_serialize(obj, False))
        state = {
            "ts": now,
            "belt_mps": float(v),
            "intercept_y": float(self.cfg.GRASP_INTERCEPT_Y),
            "max_reach": float(self.cfg.MAX_REACH),
            "status": self._status,
            "status_detail": self._status_detail,
            "objects": objs,
        }
        try:
            self._belt_state_pub.publish(String(data=json.dumps(state)))
        except Exception:
            pass  # never let viz publishing kill the control loop

    # ------------------------------------------------------------------
    # Epoch stages
    # ------------------------------------------------------------------
    def _intake_new_detections(self, now: float) -> None:
        """Stage 1: poll SAM (subject to frame-gate cooldown).

        TODO(duplicate-detection): SAM does not emit object identity, so
        successive frames re-detect the same physical object as new
        TrackedObjects → robot picks the same item multiple times. The
        current ``FrameGate`` is a coarse time-based workaround that caps
        total throughput (1 poll per cooldown_distance/belt_speed seconds)
        and cannot distinguish "same object" from "new object at similar
        position". Better fixes, in order of preference:
          1. Spatial association at intake — match new detection to
             existing TrackedObject within ε of its conveyor-compensated
             position; merge instead of adding. Cheap (~15 LOC), removes
             cooldown, enables EMA pose refinement as a bonus.
          2. Persistence threshold — require N consecutive frames before
             locking; combine with (1) for noise rejection.
          3. **SAM server change** — add per-track identity (ReID feature
             or tracking ID) to the detection output. May be the cleanest
             long-term fix; needs upstream cooperation since this client
             only consumes positions/class_names.
        """
        snap = self._cam_latest
        if snap is None:
            return  # waiting for the first /camera_debug/detections message
        detections = [d for d in snap.get("detections", []) if d.get("in_workspace")]
        if not detections:
            return

        # camera_debug already applied the camera→base transform, Z offsets,
        # and v*delay back-projection. ``receipt_time`` is the moment for
        # which the corrected positions are valid; the queue extrapolates
        # forward from there.
        detect_time = float(snap.get("receipt_time", time.time()))
        v = self.conveyor.current

        # Spatial dedup: each camera frame re-detects every visible object,
        # so without this the queue fills with duplicates of the same physical
        # object. Project every existing tracked object (the active pick
        # target plus everything in the queue) forward to ``detect_time`` and
        # skip any new detection that lands within OBJECT_MATCH_EPSILON of
        # one. This is the "spatial association" replacement for the old
        # FrameGate cooldown.
        existing: list[TrackedObject] = []
        if self._active_target is not None:
            existing.append(self._active_target)
        existing.extend(self.queue._objects)
        eps = self.cfg.OBJECT_MATCH_EPSILON

        def _matches(obj: TrackedObject, det_x: float, det_y: float) -> bool:
            ox = float(obj.T_grasp_base[0, 3])
            oy = float(obj.T_grasp_base[1, 3] - v * (detect_time - obj.detect_time))
            return abs(ox - det_x) < eps and abs(oy - det_y) < eps

        added = 0
        for d in detections:
            base_aim = d.get("base_aim", [0.0, 0.0, 0.0])
            base_grasp = d.get("base_grasp", [0.0, 0.0, 0.0])
            det_x = float(base_grasp[0])
            det_y = float(base_grasp[1])
            if any(_matches(o, det_x, det_y) for o in existing):
                continue  # already tracking this physical object
            T_aim_base = _make_transform(_R_GRASP_DEFAULT, base_aim)
            T_grasp_base = _make_transform(_R_GRASP_DEFAULT, base_grasp)
            new_obj = TrackedObject(
                T_aim_base=T_aim_base,
                T_grasp_base=T_grasp_base,
                class_name=d.get("class", "?"),
                detect_time=detect_time,
                cam_pos=tuple(d.get("cam", [0.0, 0.0, 0.0])),
            )
            self.queue.add(new_obj)
            existing.append(new_obj)  # dedupe within the same intake too
            added += 1

        if added > 0:
            self.frame_gate.mark(now)  # kept for backward compat (queue-empty reset)
            self._node.get_logger().info(
                f"New frame — {added} new object(s) added (queue size: "
                f"{len(self.queue._objects)}, belt {v:.3f} m/s)"
            )

    def _on_camera_debug_detections(self, msg: String) -> None:
        try:
            self._cam_latest = json.loads(msg.data)
        except (ValueError, TypeError):
            pass

    def _execute_cycle(
        self,
        current_joint: np.ndarray,
        aim_joint1: np.ndarray,
        grasp_joint1: np.ndarray,
        aim_joint2: np.ndarray,
        T_grasp1: np.ndarray,
        T_aim2: np.ndarray,
        theta: float,
        plan_time: float,
    ) -> None:
        """Legacy "moving" strategy: pick (suction mid-trajectory) → throw."""
        # _execute_pick uses send_trajectory_queue_with_attach, which fires
        # suction_on while the arm is still approaching (diff<0.05) — matches
        # ROS1 customcontroller. No post-arrival sleep needed; vacuum has
        # been forming during the final approach.
        self._execute_pick(current_joint, aim_joint1, grasp_joint1, plan_time)

        params = self.planner.compute_throw_params(T_grasp1, T_aim2, theta)
        self.throw_skill.build_throw_trajectory(grasp_joint1, aim_joint2, params)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run_epoch(self, epoch: int) -> None:
        # Pump ROS callbacks so joint_states and conveyor speed stay fresh.
        # Previously DetectionIntake.poll() spun every epoch; with the HTTP
        # stream source it no longer does, so this is now the only spin on
        # idle epochs (trajectory execution still spins during pick/throw).
        rclpy.spin_once(self._node, timeout_sec=0.0)

        current_joint_list = self.traj_ctrl.current_joints
        if current_joint_list is None:
            self._node.get_logger().warn("Joints not available yet.")
            time.sleep(self.cfg.TIME_STEP)
            return
        current_joint = np.array(current_joint_list)
        now = time.time()

        self.conveyor.check_freshness()
        self._publish_belt_state()                # live belt + queue snapshot

        self._intake_new_detections(now)
        self.queue.update(now, self.conveyor.current)
        if not self.queue:
            self.frame_gate.reset()
            time.sleep(self.cfg.TIME_STEP)
            return

        if self.cfg.PICK_STRATEGY == "ambush":
            request = self._select_ambush_target(now, current_joint)
            if request is not None:
                # Decide push vs throw (rule-based today; RL later) and run it.
                skill = self.selector.select(request)
                skill.execute(request)
            return

        # Capture the secondary throw target *before* lock_or_drop_head
        # mutates the queue. Without this, the only way to recover the
        # secondary would be queue.head() *after* the head was popped —
        # an implicit ordering contract that's easy to break by accident.
        secondary = self.queue.peek_next() if self.queue.has_next() else None

        lock = lock_or_drop_head(
            self.queue,
            self.planner,
            current_joint,
            self.conveyor.current,
            now,
            fixed_delay=self.pick_delay.value,
        )
        if lock.status == TargetStatus.DROPPED_IK:
            self._node.get_logger().warn("IK failed in pick adjustment; dropping target")
            return
        if lock.status == TargetStatus.DROPPED_PASSED:
            self._node.get_logger().info("Target passed the reachable arc; dropping")
            return
        if lock.status == TargetStatus.WAIT:
            time.sleep(self.cfg.TIME_STEP)
            return
        # LOCKED
        target_obj = lock.target
        T_aim1, T_grasp1 = lock.T_aim, lock.T_grasp

        self.traj_ctrl.suction_off()
        theta = THETA_MAP.get(target_obj.class_name, 0.0)

        T_aim2 = self.throw_skill.plan_throw_landing(T_grasp1, theta, T_aim1, now, secondary)

        keyframes = self.throw_skill.solve_keyframe_joints(T_aim1, T_grasp1, T_aim2)
        if keyframes is None:
            return
        aim_joint1, grasp_joint1, aim_joint2 = keyframes

        self._node.get_logger().info(f"Target locked: {target_obj.class_name}")
        self._execute_cycle(
            current_joint, aim_joint1, grasp_joint1, aim_joint2,
            T_grasp1, T_aim2, theta, now,
        )

    def run(self) -> None:
        self.setup()
        epoch = 0
        try:
            while rclpy.ok():
                epoch += 1
                try:
                    self.run_epoch(epoch)
                except Exception as e:
                    self._node.get_logger().error(f"Epoch {epoch} failed: {e}")
                    time.sleep(0.1)
        finally:
            try:
                # queue mode를 걸어둔 채 종료하면 다음 실행 시 FJT가 reject됨
                self.traj_ctrl.exit_queue_mode()
            except Exception as e:
                self._node.get_logger().warn(f"exit_queue_mode failed: {e}")
            self._node.destroy_node()
            rclpy.shutdown()


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="gp8_app",
        description="GP8 conveyor pick-and-place orchestrator.",
    )
    parser.add_argument(
        "--skill",
        choices=["throw", "push"],
        default=None,
        help=(
            "Force every object to this manipulation skill (testing aid). "
            "Overrides the GP8_FORCE_SKILL env var. Omit for normal "
            "push/throw routing."
        ),
    )
    # parse_known_args so ROS 2 / ros2 launch-injected args (e.g. --ros-args)
    # pass through harmlessly instead of erroring out.
    args, _ = parser.parse_known_args(argv)

    cfg = Config()
    if args.skill is not None:
        cfg.FORCE_SKILL = args.skill   # CLI flag wins over the env default

    app = GP8App(cfg)
    app.run()


if __name__ == "__main__":
    main()

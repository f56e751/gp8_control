"""GP8 pick-and-throw main application (ROS 2).

Ported from ROS 1 ``my_gp8_control/src/main_sam7.py`` (류가은 branch)
but running on MotoROS2 + name bridge + ROS 2 Humble. The orchestrator
itself stays small — domain logic lives in ``perception/``,
``tracking/``, ``planning/``, and ``controllers/``.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from gp8_control.controllers.trajectory_controller import TrajectoryController
from gp8_control.controllers.moveit_controller import MoveItController
from gp8_control.controllers.pick_delay_tracker import PickDelayTracker
from gp8_control.perception.apriltag_detector import ApriltagDetector
from gp8_control.perception.sam_client import SAMClient
from gp8_control.perception.camera_manager import CameraManager
from gp8_control.perception.conveyor_speed import ConveyorSpeedTracker
from gp8_control.perception.detection_intake import DetectionIntake
from gp8_control.trajectory.trajectory_primitive import (
    trajectory,
    trajectory_3points,
    new_trajectory,
    pad,
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

from gp8_control.robots.gp8 import GP8


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
    SERVER_IP: str = field(
        default_factory=lambda: _env_default("GP8_SAM_SERVER_IP", "127.0.0.1")
    )
    SERVER_PORT: int = field(
        default_factory=lambda: int(_env_default("GP8_SAM_SERVER_PORT", "7150"))
    )

    # Workspace
    MAX_REACH: float = 0.65
    CONVEYOR_SPEED: float = 0.083
    CONVEYOR_TOPIC: str = "/conveyor/speed"
    CONVEYOR_STALE_SECONDS: float = 2.0
    TARGET_DISTANCE: float = 1.2

    # Detection
    DETECTION_OFFSET_AIM: float = 0.07
    DETECTION_OFFSET_GRASP: float = -0.01

    # Pick-cycle delay starts at 0 (first pick uncompensated), then the
    # first observed overhead is adopted as-is, later picks EMA-smooth.
    FIXED_DELAY_THROW: float = 0.2
    DELAY_EMA_ALPHA: float = 0.3

    # Trajectory sampling / joint limit scales
    TRAJ_HZ: float = 20.0
    JOINT_VEL_LIMIT_SCALE: float = 0.8
    JOINT_ACCEL_LIMIT_SCALE: float = 2.5

    # Loop cooldown
    TIME_STEP: float = 1.0 / 25.0
    FRAME_COOLDOWN_DISTANCE: float = 0.8

    # Throw NN post-processing (main_sam7)
    THROW_TIME_SCALE: float = 0.85
    RELEASE_EARLY_SHIFT: float = 0.15
    ETA_MIN: float = 0.13
    ETA_MAX: float = 0.95

    # Initial pose
    INITIAL_R: np.ndarray = field(default_factory=lambda: np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ]))
    INITIAL_T: np.ndarray = field(default_factory=lambda: np.array([[0.4], [0.0], [0.1]]))

    # Fixed extrinsics (main_sam7 trusts these without an AprilTag handshake)
    T_ROBOT2BASE: np.ndarray = field(default_factory=lambda: np.array([
        [0.0, 1.0, 0.0, -0.025],
        [-1.0, 0.0, 0.0, 0.235],
        [0.0, 0.0, 1.0, -0.020],
        [0.0, 0.0, 0.0, 1.0],
    ]))
    T_BASE2CAM: np.ndarray = field(default_factory=lambda: np.array([
        [0.0, -1.0, 0.0, -2.255],
        [-1.0, 0.0, 0.0,  0.450],
        [0.0,  0.0, -1.0, 0.650],
        [0.0,  0.0, 0.0,  1.0],
    ]))

    def throw_decoding(self) -> ThrowDecodingConfig:
        return ThrowDecodingConfig(
            throw_time_scale=self.THROW_TIME_SCALE,
            release_early_shift=self.RELEASE_EARLY_SHIFT,
            eta_min=self.ETA_MIN,
            eta_max=self.ETA_MAX,
        )


# =========================================================================
# App-level policy
# =========================================================================

# Class-specific throw-plane angle (radians, rotation about +Z). Keys are
# SAM class names; classes not listed here throw at theta=0. This is app
# policy connecting perception output to throw geometry — keeping it at
# the app boundary makes that responsibility explicit.
THETA_MAP = {
    "transparent": -np.pi / 12.0,
    "metal":       -np.pi * 25.0 / 180.0,
}


# =========================================================================
# Helpers
# =========================================================================

def _make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.ravel()
    return T


# =========================================================================
# Main application
# =========================================================================

class GP8App:
    """Pick-and-throw orchestrator with main_sam7-style tracking."""

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()
        self.robot = GP8()

        self._node: Node | None = None
        self._executor: MultiThreadedExecutor | None = None
        self.traj_ctrl: TrajectoryController | None = None
        self.moveit_ctrl: MoveItController | None = None
        self.sam_client: SAMClient | None = None
        self.apriltag: ApriltagDetector | None = None
        self.camera: CameraManager | None = None
        self.conveyor: ConveyorSpeedTracker | None = None
        self.intake: DetectionIntake | None = None

        self.M1: np.ndarray | None = None
        self.M2: np.ndarray | None = None
        self.predictor: TrajectoryPredictor | None = None
        self.planner: PickThrowPlanner | None = None

        self.queue = TrackedObjectQueue(self.cfg.MAX_REACH)
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

        self.traj_ctrl = TrajectoryController(self._node)
        self.moveit_ctrl = MoveItController(self._node)
        self.sam_client = SAMClient(self._node, self.cfg.SERVER_IP, self.cfg.SERVER_PORT)
        self.apriltag = ApriltagDetector(self._node)
        self.camera = CameraManager(self._node)
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
        self._build_intake()
        self._enable_robot()
        # Always start from a known-off suction state.
        self._node.get_logger().info("Forcing suction OFF at startup.")
        self.traj_ctrl.suction_off()
        self._move_to_initial_pose()
        # main_sam7 trusts the hardcoded extrinsic; the apriltag node still
        # runs so verification is possible separately.
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
        )

    def _build_intake(self) -> None:
        self.intake = DetectionIntake(
            node=self._node,
            sam_client=self.sam_client,
            T_robot2base=self.cfg.T_ROBOT2BASE,
            T_base2cam=self.cfg.T_BASE2CAM.copy(),
            offset_aim=self.cfg.DETECTION_OFFSET_AIM,
            offset_grasp=self.cfg.DETECTION_OFFSET_GRASP,
            time_step=self.cfg.TIME_STEP,
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
    # Execution
    # ------------------------------------------------------------------
    def _execute_pick(self, current_joint, aim_joint, grasp_joint) -> None:
        """Send pick trajectory and update PickDelayTracker with measured overhead."""
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
        prev = self.pick_delay.value
        self.pick_delay.update(observed_overhead)
        self._node.get_logger().info(
            f"Pick overhead: {observed_overhead*1000:+.0f}ms "
            f"(smoothed {prev*1000:.0f} -> {self.pick_delay.value*1000:.0f}ms, "
            f"predicted traj {traj_duration*1000:.0f}ms, actual {elapsed*1000:.0f}ms)"
        )

    def _execute_transfer(
        self,
        grasp_joint: np.ndarray,
        aim_joint2: np.ndarray,
        params,
    ) -> None:
        """Build and dispatch throw trajectory using already-decoded ThrowParams."""
        n_steps = max(2, int(params.T * self.cfg.TRAJ_HZ))
        s = np.linspace(0.0, 1.0, n_steps + 1)
        s_ext = np.concatenate((s, [params.eta]))

        traj_ext, vel_ext, _, _, ts_ext = new_trajectory(
            s_ext, grasp_joint[:5], aim_joint2[:5], params.w, params.T,
        )

        traj_throw = pad(traj_ext[:-1, :]).T
        vel_throw = pad(vel_ext[:-1, :]).T
        timestep_throw = ts_ext[:-1]
        release_joint = traj_ext[-1, :]

        self._node.get_logger().info(f"Throw T={params.T:.3f}s eta={params.eta:.3f}")

        self.traj_ctrl.send_trajectory_queue_with_release(
            traj_throw, vel_throw, timestep_throw,
            final_joint=aim_joint2,
            release_joint=pad(release_joint.reshape(1, -1)).ravel(),
        )

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
        v_now = self.conveyor.current
        if not self.frame_gate.should_poll(now, v_now):
            time.sleep(self.cfg.TIME_STEP)
            return

        candidates, delay = self.intake.poll(time_to_check=0.3)
        detect_time = time.time()
        if not candidates:
            return

        self.frame_gate.mark(now)
        self._node.get_logger().info(
            f"New frame — {len(candidates)} object(s) detected (belt {v_now:.3f} m/s)"
        )
        for cand in candidates:
            T_aim_base = cand.T_aim.copy()
            T_grasp_base = cand.T_grasp.copy()
            # Compensate for perception-pipeline delay using live speed
            T_aim_base[1, 3] -= v_now * delay
            T_grasp_base[1, 3] -= v_now * delay
            self.queue.add(TrackedObject(
                T_aim_base=T_aim_base,
                T_grasp_base=T_grasp_base,
                class_name=cand.class_name,
                detect_time=detect_time,
            ))

    def _plan_throw_landing(
        self,
        T_grasp1: np.ndarray,
        theta: float,
        T_aim1_fallback: np.ndarray,
        now: float,
        secondary: TrackedObject | None,
    ) -> np.ndarray:
        """Stage 4.2: aim throw at ``secondary`` if feasible; else drop in place.

        ``secondary`` is captured by the caller *before* the lock step so
        this method does not depend on the queue's mutation order.
        """
        if secondary is None:
            return T_aim1_fallback.copy()

        T_aim2, _, _, neg_wait2 = self.planner.plan_throw_landing(
            T_grasp1,
            secondary.T_aim_base.copy(),
            theta,
            secondary.detect_time,
            self.conveyor.current,
            now,
            fixed_delay=self.cfg.FIXED_DELAY_THROW,
        )
        infeasible = (
            neg_wait2 is not None
            or T_aim2[0, 3] < 0.1
            or T_aim2[2, 3] < 0.0
            or T_aim2[2, 3] > self.cfg.MAX_REACH
        )
        if infeasible:
            return T_aim1_fallback.copy()
        return T_aim2

    def _solve_keyframe_joints(
        self,
        T_aim1: np.ndarray,
        T_grasp1: np.ndarray,
        T_aim2: np.ndarray,
    ):
        """Stage 4.3: IK for all three keyframes; zero last joint."""
        aim_joint1 = self.robot.inverse_kinematics(T_aim1)
        grasp_joint1 = self.robot.inverse_kinematics(T_grasp1)
        aim_joint2 = self.robot.inverse_kinematics(T_aim2)
        if aim_joint1 is None or grasp_joint1 is None or aim_joint2 is None:
            self._node.get_logger().warn("IK failed after target lock; aborting")
            return None
        aim_joint1 = np.asarray(aim_joint1, dtype=float)
        grasp_joint1 = np.asarray(grasp_joint1, dtype=float)
        aim_joint2 = np.asarray(aim_joint2, dtype=float)
        aim_joint1[-1] = 0.0
        grasp_joint1[-1] = 0.0
        aim_joint2[-1] = 0.0
        return aim_joint1, grasp_joint1, aim_joint2

    def _execute_cycle(
        self,
        current_joint: np.ndarray,
        aim_joint1: np.ndarray,
        grasp_joint1: np.ndarray,
        aim_joint2: np.ndarray,
        T_grasp1: np.ndarray,
        T_aim2: np.ndarray,
        theta: float,
    ) -> None:
        """Stage 5: pick (suction fires mid-trajectory) → throw."""
        # _execute_pick uses send_trajectory_queue_with_attach, which fires
        # suction_on while the arm is still approaching (diff<0.05) — matches
        # ROS1 customcontroller. No post-arrival sleep needed; vacuum has
        # been forming during the final approach.
        self._execute_pick(current_joint, aim_joint1, grasp_joint1)

        params = self.planner.compute_throw_params(T_grasp1, T_aim2, theta)
        self._execute_transfer(grasp_joint1, aim_joint2, params)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run_epoch(self, epoch: int) -> None:
        current_joint_list = self.traj_ctrl.current_joints
        if current_joint_list is None:
            self._node.get_logger().warn("Joints not available yet.")
            time.sleep(self.cfg.TIME_STEP)
            return
        current_joint = np.array(current_joint_list)
        now = time.time()

        self.conveyor.check_freshness()

        self._intake_new_detections(now)
        self.queue.update(now, self.conveyor.current)
        if not self.queue:
            self.frame_gate.reset()
            time.sleep(self.cfg.TIME_STEP)
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

        T_aim2 = self._plan_throw_landing(T_grasp1, theta, T_aim1, now, secondary)

        keyframes = self._solve_keyframe_joints(T_aim1, T_grasp1, T_aim2)
        if keyframes is None:
            return
        aim_joint1, grasp_joint1, aim_joint2 = keyframes

        self._node.get_logger().info(f"Target locked: {target_obj.class_name}")
        self._execute_cycle(
            current_joint, aim_joint1, grasp_joint1, aim_joint2,
            T_grasp1, T_aim2, theta,
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


def main() -> None:
    app = GP8App()
    app.run()


if __name__ == "__main__":
    main()

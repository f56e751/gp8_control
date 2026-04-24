"""GP8 pick-and-throw main application (ROS 2).

Ported from ROS 1 ``my_gp8_control/src/main_sam7.py`` (류가은 branch)
but running on MotoROS2 + name bridge + ROS 2 Humble.

Key behaviour (preserved from main_sam7):
  - Non-blocking SAM detection polling (``time_to_check=0.3``)
  - Persistent ``tracked_objects`` queue that compensates each frame
    for conveyor motion via ``detect_time``; drops anything past ``-max_reach``
  - Frame cooldown driven by ``time_to_travel = 0.8 / conveyor_speed``
  - Primary target = head of the queue; secondary target (throw landing)
    = the next object on the queue, if present and feasible
  - Iterative 3-pass refinement of both pick trajectory time and throw
    trajectory time (the latter hits the torch NN each iteration)
  - Throw parameter post-processing:
      T   = exp(params[-2]) * THROW_TIME_SCALE
      eta = sigmoid(params[-1])
      eta = clip(eta - RELEASE_EARLY_SHIFT, ETA_MIN, ETA_MAX)

Depends on the name bridge (MotoROS2 raw names -> URDF names) being up, and
on a trajectory NN server at ``tcp://localhost:5555`` (see
``gp8_control.trajectory_server``).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Float64

from gp8_control.controllers.trajectory_controller import TrajectoryController
from gp8_control.controllers.moveit_controller import MoveItController
from gp8_control.perception.apriltag_detector import ApriltagDetector
from gp8_control.perception.sam_client import SAMClient
from gp8_control.perception.camera_manager import CameraManager
from gp8_control.utils.lie_numpy import invSE3
from gp8_control.trajectory.trajectory_primitive import (
    opt_time,
    trajectory,
    trajectory_3points,
    new_trajectory,
    pad,
)
from gp8_control.trajectory.predictor import TrajectoryPredictor

from gp8_control.robots.gp8 import GP8


# =========================================================================
# Configuration
# =========================================================================

def _env_default(key: str, default: str) -> str:
    """Look up a config value from the process environment at import time."""
    import os as _os  # local import keeps module surface clean
    return _os.environ.get(key, default)


@dataclass
class Config:
    # Network — the robot controller uses a private LAN so it's safe to
    # publish the address. The remote SAM inference IP is a *public* address
    # on our GPU cluster, so we don't hardcode it in a public repo; set
    # GP8_SAM_SERVER_IP and GP8_SAM_SERVER_PORT in the launch environment.
    ROBOT_IP: str = "192.168.255.1"
    SERVER_IP: str = field(
        default_factory=lambda: _env_default("GP8_SAM_SERVER_IP", "127.0.0.1")
    )
    SERVER_PORT: int = field(
        default_factory=lambda: int(_env_default("GP8_SAM_SERVER_PORT", "7150"))
    )

    # Workspace
    MAX_REACH: float = 0.65
    # Fallback conveyor speed used only until the first message arrives on
    # /conveyor/speed (std_msgs/Float64, published by esp32_encoder). Matches
    # main_sam7's hardcoded value so behaviour degrades gracefully when the
    # encoder is offline.
    CONVEYOR_SPEED: float = 0.083
    CONVEYOR_TOPIC: str = "/conveyor/speed"
    CONVEYOR_STALE_SECONDS: float = 2.0  # warn once if no update for this long
    TARGET_DISTANCE: float = 1.2

    # Detection
    DETECTION_OFFSET_AIM: float = 0.07
    DETECTION_OFFSET_GRASP: float = -0.01

    # Pick-cycle delay is *not* hardcoded anymore — it starts at 0 (so the
    # very first pick has no compensation and may miss by the belt's travel
    # during the pick), then the first observed overhead is adopted as-is,
    # and every subsequent pick EMA-smooths the measurement. See
    # `_execute_pick` for the update.
    FIXED_DELAY_THROW: float = 0.2
    # How fast the measured pick delay tracks reality after the first sample
    # (0=frozen, 1=always latest sample). 0.3 mixes in ~30% of every new
    # observation.
    DELAY_EMA_ALPHA: float = 0.3

    # Trajectory sampling / joint limit scales (M1 * M2_SCALE = accel cap)
    TRAJ_HZ: float = 20.0
    JOINT_VEL_LIMIT_SCALE: float = 0.8
    JOINT_ACCEL_LIMIT_SCALE: float = 2.5

    # Loop cooldown
    TIME_STEP: float = 1.0 / 25.0
    FRAME_COOLDOWN_DISTANCE: float = 0.8  # meters of belt travel per frame window

    # Throw NN post-processing (main_sam7)
    THROW_TIME_SCALE: float = 0.85
    RELEASE_EARLY_SHIFT: float = 0.15
    ETA_MIN: float = 0.13
    ETA_MAX: float = 0.95

    # Initial pose (tool pointing down, slightly forward)
    INITIAL_R: np.ndarray = field(default_factory=lambda: np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ]))
    INITIAL_T: np.ndarray = field(default_factory=lambda: np.array([[0.4], [0.0], [0.1]]))

    # Fixed extrinsics — both matrices hardcoded exactly as in main_sam7.
    # main_sam7 did NOT use AprilTag at runtime; it just trusted these values.
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

    # Post-pick pause to let the suction cup pull vacuum before throwing
    SUCTION_ATTACH_WAIT: float = 0.1


# =========================================================================
# Data classes
# =========================================================================

@dataclass
class TrackedObject:
    T_aim_base: np.ndarray   # pose at detection time (absolute, conveyor not yet applied)
    T_grasp_base: np.ndarray
    class_name: str
    detect_time: float       # time.time() when this object was observed


# Class-specific throw angle (radians, rotation of throw plane about +Z)
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


def _position_adjustment_for_IK(
    T: np.ndarray, max_reach: float, conveyor_speed: float
):
    """Main_sam7-style reachability projection.

    If the xy-projected position is outside max_reach, push the Y-component
    onto the circle boundary in the direction of belt travel and report the
    corresponding wait / missed-window.

    Returns:
        (T_projected, wait_time_s_or_None, neg_wait_time_s_or_None)
        Exactly one of wait_time / neg_wait_time will be non-None if
        projection had to be applied. Both None means the input is reachable.
    """
    T_tmp = T.copy()
    if np.linalg.norm(T[:2, 3]) <= max_reach:
        return T_tmp, None, None

    denom = max_reach ** 2 - T[0, 3] ** 2
    if denom < 0.0:
        # Even with any Y we can't reach this X; treat as missed
        return T_tmp, None, (T[1, 3] + max_reach) / (conveyor_speed + 1e-6)

    y_boundary = float(np.sqrt(denom))
    if T[1, 3] > 0.0:
        T_tmp[1, 3] = y_boundary
        wait_time = (T[1, 3] - T_tmp[1, 3]) / (conveyor_speed + 1e-6)
        return T_tmp, wait_time, None

    T_tmp[1, 3] = -y_boundary
    neg_wait_time = (T[1, 3] - T_tmp[1, 3]) / (conveyor_speed + 1e-6)
    return T_tmp, None, neg_wait_time


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

        self.T_base2cam: np.ndarray | None = None
        self.M1: np.ndarray | None = None
        self.M2: np.ndarray | None = None
        self.predictor: TrajectoryPredictor | None = None

        # main_sam7 state
        self._tracked_objects: list[TrackedObject] = []
        self._last_frame_time: float | None = None

        # Live conveyor speed (updated from /conveyor/speed callback)
        self._conveyor_speed: float = self.cfg.CONVEYOR_SPEED
        self._last_speed_msg_time: float | None = None
        self._speed_stale_warned: bool = False

        # Purely measurement-driven pick-cycle overhead. Starts at zero; the
        # first successful pick replaces it with the observed value directly
        # (no EMA smoothing yet — nothing to smooth against), and later picks
        # smooth with `DELAY_EMA_ALPHA`.
        self._measured_pick_delay: float = 0.0
        self._pick_delay_initialized: bool = False

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

        self.traj_ctrl.wait_for_servers()
        self._subscribe_conveyor_speed()
        self._load_predictor()
        self._setup_joint_limits()
        self._enable_robot()
        # Always start from a known-off suction state — otherwise, if a
        # previous run ended mid-grasp, the gripper would still be pulling
        # vacuum when we first move, which can fling whatever it's holding.
        self._node.get_logger().info("Forcing suction OFF at startup.")
        self.traj_ctrl.suction_off()
        self._move_to_initial_pose()
        # main_sam7 simply trusts a hardcoded T_base2cam instead of detecting
        # the base AprilTag at runtime. Use that path by default; the apriltag
        # node is still launched in case anyone wants to verify extrinsics.
        self.T_base2cam = self.cfg.T_BASE2CAM.copy()
        self._node.get_logger().info(
            "Using hardcoded T_base2cam (no AprilTag handshake)."
        )

    def _subscribe_conveyor_speed(self) -> None:
        """Subscribe to /conveyor/speed; keep Config default as fallback."""
        self._node.create_subscription(
            Float64,
            self.cfg.CONVEYOR_TOPIC,
            self._on_conveyor_speed,
            1,
        )
        self._node.get_logger().info(
            f"Subscribed to {self.cfg.CONVEYOR_TOPIC} "
            f"(fallback {self.cfg.CONVEYOR_SPEED:.3f} m/s until first message)"
        )

    def _on_conveyor_speed(self, msg: Float64) -> None:
        first = self._last_speed_msg_time is None
        self._conveyor_speed = float(msg.data)
        self._last_speed_msg_time = time.time()
        if first:
            self._node.get_logger().info(
                f"Live conveyor speed acquired: {self._conveyor_speed:.4f} m/s"
            )
        if self._speed_stale_warned:
            self._speed_stale_warned = False

    def _check_speed_freshness(self) -> None:
        """Log once if no speed message received within CONVEYOR_STALE_SECONDS."""
        if self._last_speed_msg_time is None or self._speed_stale_warned:
            return
        if time.time() - self._last_speed_msg_time > self.cfg.CONVEYOR_STALE_SECONDS:
            self._node.get_logger().warn(
                f"No /conveyor/speed update for >{self.cfg.CONVEYOR_STALE_SECONDS:.1f}s; "
                f"using last value {self._conveyor_speed:.4f} m/s"
            )
            self._speed_stale_warned = True

    def _load_predictor(self) -> None:
        self.predictor = TrajectoryPredictor()
        self._node.get_logger().info(
            f"Trajectory NN loaded: {self.predictor.weight_path}"
        )

    def _setup_joint_limits(self) -> None:
        # GP8.velocity_limits is a numpy array of max joint speeds (rad/s),
        # unlike the ROS 1 Yaskawa class which exposed (lo, hi) tuples as
        # `jointvel_bounds`. Treat it as the upper bound directly.
        self.M1 = np.asarray(self.robot.velocity_limits, dtype=float) * self.cfg.JOINT_VEL_LIMIT_SCALE
        self.M2 = self.M1 * self.cfg.JOINT_ACCEL_LIMIT_SCALE

    def _enable_robot(self) -> None:
        """Enable robot in Point Queue Mode (streaming, no start-state check).

        FJT (StartTrajMode)의 INIT_TRAJ_INVALID_STARTING_POS 제약을 피하기 위해
        Point Queue Mode로 진입. 모든 후속 trajectory 전송은 queue 경로를 탄다.
        """
        self._node.get_logger().info("Enabling robot (point queue mode)...")
        if not self.traj_ctrl.enter_queue_mode():
            raise RuntimeError(
                "Failed to enter point queue mode. Check pendant is in REMOTE "
                "mode with no active alarm and cycle mode AUTO."
            )
        time.sleep(1.0)

    def _move_to_initial_pose(self) -> None:
        # ros2 launch ExecuteProcess does not forward stdin to the child, so
        # a blocking ``input()`` would hang forever. When running from a real
        # TTY (e.g. `python -m gp8_control.app` in a terminal) we still give
        # the operator a chance to abort before motion starts.
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

        # Use a direct trajectory rather than MoveIt so we do not depend on
        # planners being configured perfectly for this simple move.
        # `time.sleep` alone does not let ROS 2 subscription callbacks fire —
        # we need to spin the node, otherwise TrajectoryController.current_joints
        # stays None forever even while the bridge is publishing.
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

    def _wait_for_base_tag(self) -> None:
        time.sleep(0.5)
        while rclpy.ok():
            rclpy.spin_once(self._node, timeout_sec=0.05)
            if self.apriltag.T_base is not None:
                self.T_base2cam = invSE3(self.apriltag.T_base)
                self._node.get_logger().info("Base AprilTag detected.")
                return

    # ------------------------------------------------------------------
    # Detection -> world-frame grasp pose
    # ------------------------------------------------------------------

    def _detection_to_grasp(self, position):
        """Camera-frame position -> (T_aim, T_grasp) in robot frame."""
        T_cam = np.eye(4)
        T_cam[:3, 3] = np.asarray(position, dtype=float)
        T_robot = self.cfg.T_ROBOT2BASE @ self.T_base2cam @ T_cam
        grasp_pos = T_robot[:3, 3]

        R_grasp = np.array(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
        )
        T_aim = _make_transform(
            R_grasp,
            grasp_pos + np.array([0.0, 0.0, self.cfg.DETECTION_OFFSET_AIM]),
        )
        T_grasp = _make_transform(
            R_grasp,
            grasp_pos + np.array([0.0, 0.0, self.cfg.DETECTION_OFFSET_GRASP]),
        )
        return T_aim, T_grasp

    def _get_new_detections_non_blocking(
        self, time_to_check: float = 0.3
    ) -> tuple[list, float]:
        """Poll SAM client briefly; return (list of (T_aim, T_grasp, class), delay).

        Mirrors main_sam7.get_new_detections_non_blocking.
        """
        s_time = time.time()
        positions = None
        class_names = None
        delay = 0.0

        while time.time() - s_time < time_to_check:
            rclpy.spin_once(self._node, timeout_sec=0.01)
            p = self.sam_client.positions
            n = self.sam_client.class_names
            if p is not None and n is not None:
                if p and n:
                    positions = p
                    class_names = n
                    delay = self.sam_client.delay or 0.0
                    break
                # Stale / empty publish — wait and retry
                time.sleep(self.cfg.TIME_STEP)
                continue

            # Clear buffers to avoid re-processing stale data.
            self.sam_client.positions = None
            self.sam_client.class_names = None

        if positions is None or class_names is None:
            return [], 0.0

        obj_list = []
        for pos, cls in zip(positions, class_names):
            if not (pos[2] < 0.67 and -0.2 < pos[0] < 0.2):
                continue
            T_aim, T_grasp = self._detection_to_grasp(pos)
            obj_list.append((T_aim, T_grasp, cls))
        return obj_list, delay

    # ------------------------------------------------------------------
    # Queue maintenance
    # ------------------------------------------------------------------

    def _update_tracked_queue(self, current_time: float) -> None:
        """Drop objects that have passed ``-max_reach`` and re-sort by current Y."""
        v = self._conveyor_speed
        keep: list[TrackedObject] = []
        for obj in self._tracked_objects:
            dt = current_time - obj.detect_time
            current_y = obj.T_aim_base[1, 3] - v * dt
            if current_y > -self.cfg.MAX_REACH:
                keep.append(obj)
        keep.sort(
            key=lambda o: o.T_aim_base[1, 3] - v * (current_time - o.detect_time)
        )
        self._tracked_objects = keep

    # ------------------------------------------------------------------
    # Motion planning (detect_time-based compensation)
    # ------------------------------------------------------------------

    def _dynamic_adjustments(
        self,
        T_aim: np.ndarray,
        T_grasp: np.ndarray,
        detect_time: float,
        current_joint: np.ndarray,
        fixed_delay: float = 0.0,
    ):
        """main_sam7.dynamic_adjustments port.

        Returns:
            (T_aim_tmp, T_grasp_tmp, traj_time, wait_time, neg_wait_time)
            or (None, None, None, None, None) on IK failure.
        """
        v = self._conveyor_speed
        # Present-time pose (eliminate perception delay already once)
        T_aim_base = T_aim.copy()
        T_grasp_base = T_grasp.copy()
        T_aim_base[1, 3] -= v * (time.time() - detect_time)
        T_grasp_base[1, 3] -= v * (time.time() - detect_time)

        aim_j = None
        grasp_j = None
        T_aim_tmp = T_aim_base
        T_grasp_tmp = T_grasp_base
        wait_time = None
        neg_wait_time = None

        traj_time = 1.0  # initial estimate
        for _ in range(3):
            offset = traj_time + fixed_delay
            T_aim_pred = T_aim_base.copy()
            T_aim_pred[1, 3] -= v * offset
            T_grasp_pred = T_grasp_base.copy()
            T_grasp_pred[1, 3] -= v * offset

            T_aim_tmp, wait_time, neg_wait_time = _position_adjustment_for_IK(
                T_aim_pred, self.cfg.MAX_REACH, v
            )
            aim_j = self.robot.inverse_kinematics(T_aim_tmp)

            T_grasp_tmp, _, _ = _position_adjustment_for_IK(
                T_grasp_pred, self.cfg.MAX_REACH, v
            )
            grasp_j = self.robot.inverse_kinematics(T_grasp_tmp)

            if aim_j is None or grasp_j is None:
                return None, None, None, None, None

            zero = np.zeros_like(self.M1)
            traj_time1 = opt_time(current_joint, zero, aim_j, zero, self.M1, self.M2)
            traj_time2 = opt_time(aim_j, zero, grasp_j, zero, self.M1, self.M2)
            traj_time = traj_time1 + traj_time2

        return T_aim_tmp, T_grasp_tmp, traj_time, wait_time, neg_wait_time

    def _dynamic_adjustments2(
        self,
        T_grasp1: np.ndarray,
        T_aim2: np.ndarray,
        theta: float,
        detect_time: float,
        target_distance: float,
        fixed_delay: float = 0.1,
    ):
        """main_sam7.dynamic_adjustments2 port — iteratively refines throw
        landing using the torch NN's predicted trajectory time."""
        v = self._conveyor_speed
        T_aim2_base = T_aim2.copy()
        T_aim2_base[1, 3] -= v * (time.time() - detect_time)

        T_aim2_tmp = T_aim2_base
        wait_time = None
        neg_wait_time = None

        traj_time = 1.5
        for _ in range(3):
            offset = traj_time + fixed_delay
            T_aim2_pred = T_aim2_base.copy()
            T_aim2_pred[1, 3] -= v * offset

            T_aim2_tmp, wait_time, neg_wait_time = _position_adjustment_for_IK(
                T_aim2_pred, self.cfg.MAX_REACH, v
            )

            # Ask torch NN (in-process) for predicted throw time
            c, s = np.cos(-theta), np.sin(-theta)
            x1 = c * T_grasp1[0, 3] - s * T_grasp1[1, 3]
            y1 = s * T_grasp1[0, 3] + c * T_grasp1[1, 3]
            x2 = c * T_aim2_tmp[0, 3] - s * T_aim2_tmp[1, 3]
            y2 = s * T_aim2_tmp[0, 3] + c * T_aim2_tmp[1, 3]
            parameters = self.predictor.predict((x1, y1), (x2, y2), target_distance)
            traj_time = float(np.exp(parameters[-2])) * self.cfg.THROW_TIME_SCALE

        return T_aim2_tmp, traj_time, wait_time, neg_wait_time

    # ------------------------------------------------------------------
    # Torch NN query (final / full params)
    # ------------------------------------------------------------------

    def _compute_throw_params(
        self, T_grasp: np.ndarray, T_aim2: np.ndarray, theta: float
    ) -> np.ndarray:
        c, s = np.cos(-theta), np.sin(-theta)
        x1 = c * T_grasp[0, 3] - s * T_grasp[1, 3]
        y1 = s * T_grasp[0, 3] + c * T_grasp[1, 3]
        x2 = c * T_aim2[0, 3] - s * T_aim2[1, 3]
        y2 = s * T_aim2[0, 3] + c * T_aim2[1, 3]
        return self.predictor.predict((x1, y1), (x2, y2), self.cfg.TARGET_DISTANCE)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _execute_pick(self, current_joint, aim_joint, grasp_joint) -> None:
        """Send pick trajectory and measure actual overhead vs prediction.

        Wall-clock between entering this method and the FJT action returning
        is compared to the trajectory's own ``sum(timestep)`` duration; the
        difference is the real fixed_delay, which we EMA-smooth into
        ``self._measured_pick_delay`` for the next pick planning iteration.
        """
        t_start = time.time()
        zero = np.zeros_like(self.M1)
        traj, vel, ts = trajectory_3points(
            current_joint, zero, aim_joint, zero, grasp_joint, zero,
            self.M1, self.M2, hertz=self.cfg.TRAJ_HZ,
        )
        traj_duration = float(np.sum(ts))
        self.traj_ctrl.send_trajectory_queue(traj, vel, ts, final_joint=grasp_joint)
        elapsed = time.time() - t_start

        observed_overhead = elapsed - traj_duration
        prev = self._measured_pick_delay
        # Guard against pathological values (e.g. goal rejected → elapsed
        # much smaller than traj_duration, giving large negative overhead).
        if -0.5 < observed_overhead < 2.0:
            if not self._pick_delay_initialized:
                # First-ever real measurement: adopt as-is so the *second*
                # pick already has a correct fixed_delay, instead of slowly
                # crawling up from 0 through EMA.
                self._measured_pick_delay = observed_overhead
                self._pick_delay_initialized = True
            else:
                alpha = self.cfg.DELAY_EMA_ALPHA
                self._measured_pick_delay = (1.0 - alpha) * prev + alpha * observed_overhead
        self._node.get_logger().info(
            f"Pick overhead: {observed_overhead*1000:+.0f}ms "
            f"(smoothed {prev*1000:.0f} -> {self._measured_pick_delay*1000:.0f}ms, "
            f"predicted traj {traj_duration*1000:.0f}ms, actual {elapsed*1000:.0f}ms)"
        )

    def _execute_transfer(
        self,
        grasp_joint: np.ndarray,
        aim_joint2: np.ndarray,
        parameters: np.ndarray,
    ) -> None:
        """Build and dispatch throw trajectory with post-processed NN params."""
        w = parameters[:-2].reshape(-1, 5)
        T = float(np.exp(parameters[-2])) * self.cfg.THROW_TIME_SCALE
        eta = 1.0 / (1.0 + np.exp(-parameters[-1]))
        eta = float(np.clip(
            eta - self.cfg.RELEASE_EARLY_SHIFT,
            self.cfg.ETA_MIN,
            self.cfg.ETA_MAX,
        ))

        n_steps = max(2, int(T * self.cfg.TRAJ_HZ))
        s = np.linspace(0.0, 1.0, n_steps + 1)
        s_ext = np.concatenate((s, [eta]))

        traj_ext, vel_ext, _, _, ts_ext = new_trajectory(
            s_ext, grasp_joint[:5], aim_joint2[:5], w, T,
        )

        traj_throw = pad(traj_ext[:-1, :]).T
        vel_throw = pad(vel_ext[:-1, :]).T
        timestep_throw = ts_ext[:-1]
        release_joint = traj_ext[-1, :]

        self._node.get_logger().info(f"Throw T={T:.3f}s eta={eta:.3f}")

        self.traj_ctrl.send_trajectory_queue_with_release(
            traj_throw, vel_throw, timestep_throw,
            final_joint=aim_joint2,
            release_joint=pad(release_joint.reshape(1, -1)).ravel(),
        )

    # ------------------------------------------------------------------
    # Main loop — faithful port of main_sam7's while not rospy.is_shutdown()
    # ------------------------------------------------------------------

    def run_epoch(self, epoch: int) -> None:
        current_joint_list = self.traj_ctrl.current_joints
        if current_joint_list is None:
            self._node.get_logger().warn("Joints not available yet.")
            time.sleep(self.cfg.TIME_STEP)
            return
        current_joint = np.array(current_joint_list)
        current_time = time.time()

        self._check_speed_freshness()

        # Frame cooldown scales with live speed; effectively zero when the belt
        # is stationary, so we can re-poll SAM immediately without waiting.
        v_now = self._conveyor_speed
        time_to_travel = (
            self.cfg.FRAME_COOLDOWN_DISTANCE / v_now if v_now > 1e-4 else 0.0
        )

        # --- 1. Recognize new objects (with frame cooldown) --------------
        if (
            self._last_frame_time is not None
            and current_time - self._last_frame_time < time_to_travel
        ):
            time.sleep(self.cfg.TIME_STEP)
        else:
            new_objs, delay = self._get_new_detections_non_blocking(time_to_check=0.3)
            detect_time = time.time()
            if new_objs:
                self._last_frame_time = current_time
                self._node.get_logger().info(
                    f"New frame — {len(new_objs)} object(s) detected "
                    f"(belt {v_now:.3f} m/s)"
                )
                for (T_aim, T_grasp, cls) in new_objs:
                    T_aim_base = T_aim.copy()
                    T_grasp_base = T_grasp.copy()
                    # Compensate for perception-pipeline delay using live speed
                    T_aim_base[1, 3] -= v_now * delay
                    T_grasp_base[1, 3] -= v_now * delay
                    self._tracked_objects.append(TrackedObject(
                        T_aim_base=T_aim_base,
                        T_grasp_base=T_grasp_base,
                        class_name=cls,
                        detect_time=detect_time,
                    ))

        # --- 2. Update queue ---------------------------------------------
        self._update_tracked_queue(current_time)

        if not self._tracked_objects:
            time.sleep(self.cfg.TIME_STEP)
            self._last_frame_time = None
            return

        # --- 3. Time-to-grasp analysis for primary target ----------------
        target_obj = self._tracked_objects[0]
        T_aim1, T_grasp1, _traj_time, wait_time, neg_wait_time = (
            self._dynamic_adjustments(
                target_obj.T_aim_base.copy(),
                target_obj.T_grasp_base.copy(),
                target_obj.detect_time,
                current_joint,
                # Use the live-measured pick delay (seeded with Config value,
                # updated each cycle by `_execute_pick`).
                fixed_delay=self._measured_pick_delay,
            )
        )

        if T_aim1 is None:
            self._node.get_logger().warn("IK failed in pick adjustment; dropping target")
            self._tracked_objects.pop(0)
            return

        if neg_wait_time is not None:
            self._node.get_logger().info("Target passed the reachable arc; dropping")
            self._tracked_objects.pop(0)
            return

        if wait_time is not None and wait_time > 0.001:
            # Not yet in reach — leave target in queue, retry next epoch
            time.sleep(self.cfg.TIME_STEP)
            return

        # --- 4. Target locked — plan & execute ---------------------------
        self._tracked_objects.pop(0)
        self._node.get_logger().info(
            f"Target locked: {target_obj.class_name}"
        )

        self.traj_ctrl.suction_off()

        theta = THETA_MAP.get(target_obj.class_name, 0.0)
        target_distance = self.cfg.TARGET_DISTANCE

        # 4.2 Throw landing (prefer: next queued object's aim pose)
        if self._tracked_objects:
            next_obj = self._tracked_objects[0]
            T_aim2, _traj_time2, _wait2, neg_wait2 = self._dynamic_adjustments2(
                T_grasp1,
                next_obj.T_aim_base.copy(),
                theta,
                next_obj.detect_time,
                target_distance,
                fixed_delay=self.cfg.FIXED_DELAY_THROW,
            )
            infeasible = (
                neg_wait2 is not None
                or T_aim2[0, 3] < 0.1
                or T_aim2[2, 3] < 0.0
                or T_aim2[2, 3] > self.cfg.MAX_REACH
            )
            if infeasible:
                T_aim2 = T_aim1.copy()
        else:
            T_aim2 = T_aim1.copy()

        # 4.3 IK for all three keyframes
        aim_joint1 = self.robot.inverse_kinematics(T_aim1)
        grasp_joint1 = self.robot.inverse_kinematics(T_grasp1)
        aim_joint2 = self.robot.inverse_kinematics(T_aim2)
        if aim_joint1 is None or grasp_joint1 is None or aim_joint2 is None:
            self._node.get_logger().warn("IK failed after target lock; aborting")
            return
        aim_joint1 = np.asarray(aim_joint1, dtype=float)
        grasp_joint1 = np.asarray(grasp_joint1, dtype=float)
        aim_joint2 = np.asarray(aim_joint2, dtype=float)
        aim_joint1[-1] = 0.0
        grasp_joint1[-1] = 0.0
        aim_joint2[-1] = 0.0

        # --- 5. Execute pick -> attach -> throw --------------------------
        self._execute_pick(current_joint, aim_joint1, grasp_joint1)

        # ROS 1 custom controller toggled suction near the grasp pose (diff<0.05)
        # to compensate for pneumatic delay. Our send_trajectory is blocking so
        # we turn on suction immediately after arrival and pause briefly.
        self.traj_ctrl.suction_on()
        time.sleep(self.cfg.SUCTION_ATTACH_WAIT)

        params = self._compute_throw_params(T_grasp1, T_aim2, theta)
        self._execute_transfer(grasp_joint1, aim_joint2, params)

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

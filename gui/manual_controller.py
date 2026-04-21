"""Manual EE controller for web GUI (extracted from real_debug_robot.py)."""
from __future__ import annotations

import threading
import time

import numpy as np
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import qos_profile_sensor_data
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from motoros2_interfaces.srv import WriteSingleIO
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from gp8_control.robots.gp8 import GP8

SUCTION_IO_ADDRESS = 10017
STEP_SIZE = 0.01  # 1cm
MOVE_DURATION = 0.5
JOINT_NAMES = [
    "joint_1_s", "joint_2_l", "joint_3_u",
    "joint_4_r", "joint_5_b", "joint_6_t",
]
WORKSPACE = {
    "x": (0.0, 0.65),
    "y": (-0.65, 0.65),
    "z": (0.0, 0.40),
}

KEY_DELTA = {
    "w": np.array([STEP_SIZE, 0.0, 0.0]),
    "s": np.array([-STEP_SIZE, 0.0, 0.0]),
    "a": np.array([0.0, STEP_SIZE, 0.0]),
    "d": np.array([0.0, -STEP_SIZE, 0.0]),
    "r": np.array([0.0, 0.0, STEP_SIZE]),
    "f": np.array([0.0, 0.0, -STEP_SIZE]),
}

# Fixed transform: link_6_t -> flange -> tool0 -> suction_tip
T_CHAIN = np.array([
    [0., 0., 1., 0.325],
    [0., -1., 0., 0.],
    [1., 0., 0., 0.],
    [0., 0., 0., 1.],
])
T_CHAIN_INV = np.linalg.inv(T_CHAIN)


def _seconds_to_duration(seconds: float) -> Duration:
    sec = int(seconds)
    nanosec = int((seconds - sec) * 1e9)
    return Duration(sec=sec, nanosec=nanosec)


def _wait_future(future, timeout: float = 10.0) -> bool:
    """Poll future.done() without calling spin_until_future_complete."""
    t0 = time.time()
    while not future.done():
        if time.time() - t0 > timeout:
            return False
        time.sleep(0.01)
    return True


class ManualController:
    """EE delta movement + suction control. Injected with a ROS 2 Node."""

    def __init__(self, node: Node, cb_group: ReentrantCallbackGroup) -> None:
        self._node = node
        self._robot = GP8()
        self._lock = threading.Lock()
        self._current_joints: list | None = None
        self._suction_on = False
        self._active = False

        # Joint state subscription — bridge republishes with URDF names
        self._node.create_subscription(
            JointState, "/joint_states_urdf", self._joint_cb, qos_profile_sensor_data,
            callback_group=cb_group,
        )

        # Action client — bridge proxy
        self._fjt_client = ActionClient(
            self._node, FollowJointTrajectory,
            "/motoman_gp8_controller/follow_joint_trajectory",
            callback_group=cb_group,
        )

        # Suction service client
        self._io_client = self._node.create_client(
            WriteSingleIO, "/write_single_io",
            callback_group=cb_group,
        )

    @property
    def active(self) -> bool:
        return self._active

    def set_active(self, value: bool) -> dict:
        self._active = value
        return {"active": self._active}

    def _joint_cb(self, msg: JointState) -> None:
        with self._lock:
            self._current_joints = list(msg.position)

    def get_ee_status(self) -> dict:
        """Return current EE position, link positions, joint limits, and suction state."""
        with self._lock:
            joints = self._current_joints
        if joints is None:
            return {"ee": None, "joints": None, "suction": self._suction_on}

        q = np.array(joints)
        link_frames, ee_frame = self._robot.forward_kinematics_all(q)
        pos = ee_frame[:3, 3]

        # Link positions for visualization (base + 6 links + EE)
        link_positions = [[0.0, 0.0, 0.0]]  # base
        for frame in link_frames:
            p = frame[:3, 3]
            link_positions.append([round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4)])
        link_positions.append([round(float(pos[0]), 4), round(float(pos[1]), 4), round(float(pos[2]), 4)])

        # Joint limits (degrees)
        joint_limits = [
            {"min": round(float(np.degrees(lo)), 1), "max": round(float(np.degrees(hi)), 1)}
            for lo, hi in self._robot.joint_limits
        ]

        # Suction tip pose via fixed chain
        T6 = link_frames[5]
        T_tip = T6 @ T_CHAIN
        tip_pos = T_tip[:3, 3]
        tip_rot = T_tip[:3, :3]
        # RPY from rotation matrix
        tip_rpy = self._rot_to_rpy(tip_rot)

        return {
            "ee": {"x": round(float(pos[0]), 4), "y": round(float(pos[1]), 4), "z": round(float(pos[2]), 4)},
            "tip": {
                "x": round(float(tip_pos[0]), 4),
                "y": round(float(tip_pos[1]), 4),
                "z": round(float(tip_pos[2]), 4),
                "roll": round(float(np.degrees(tip_rpy[0])), 1),
                "pitch": round(float(np.degrees(tip_rpy[1])), 1),
                "yaw": round(float(np.degrees(tip_rpy[2])), 1),
            },
            "joints_deg": [round(float(np.degrees(j)), 1) for j in joints],
            "joint_limits": joint_limits,
            "link_positions": link_positions,
            "suction": self._suction_on,
        }

    @staticmethod
    def _rot_to_rpy(R: np.ndarray) -> np.ndarray:
        """Convert 3x3 rotation matrix to roll-pitch-yaw (XYZ extrinsic)."""
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0.0
        return np.array([roll, pitch, yaw])

    def handle_key(self, key: str) -> dict:
        """Process a key input. Returns result dict. Call from worker thread."""
        if not self._active:
            return {"error": "Controller not active"}

        if key in KEY_DELTA:
            return self._move_ee(KEY_DELTA[key])
        elif key == " ":
            return self._toggle_suction()
        elif key == "p":
            return self.get_ee_status()
        elif key == "g":
            return self._level_suction()
        else:
            return {"error": f"Unknown key: {key}"}

    def _move_ee(self, delta: np.ndarray) -> dict:
        with self._lock:
            joints = self._current_joints
        if joints is None:
            return {"error": "No joint states"}

        current_q = np.array(joints)
        current_T = self._robot.forward_kinematics(current_q)
        new_pos = current_T[:3, 3] + delta

        # Workspace check
        if not (
            WORKSPACE["x"][0] <= new_pos[0] <= WORKSPACE["x"][1]
            and WORKSPACE["y"][0] <= new_pos[1] <= WORKSPACE["y"][1]
            and WORKSPACE["z"][0] <= new_pos[2] <= WORKSPACE["z"][1]
        ):
            return {"error": "Out of workspace", "pos": [round(float(v), 4) for v in new_pos]}

        target_T = current_T.copy()
        target_T[:3, 3] = new_pos

        target_q = self._robot.inverse_kinematics(target_T)
        if target_q is None:
            return {"error": "IK failed"}

        # Build trajectory
        jt = JointTrajectory()
        jt.joint_names = JOINT_NAMES

        pt0 = JointTrajectoryPoint()
        pt0.positions = [float(x) for x in current_q]
        pt0.velocities = [0.0] * 6
        pt0.time_from_start = _seconds_to_duration(0.0)
        jt.points.append(pt0)

        pt1 = JointTrajectoryPoint()
        pt1.positions = [float(x) for x in target_q]
        pt1.velocities = [0.0] * 6
        pt1.time_from_start = _seconds_to_duration(MOVE_DURATION)
        jt.points.append(pt1)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = jt

        # Send goal (polling, not spin_until_future_complete)
        future = self._fjt_client.send_goal_async(goal)
        if not _wait_future(future):
            return {"error": "Goal send timeout"}

        goal_handle = future.result()
        if not goal_handle.accepted:
            return {"error": "Goal rejected"}

        result_future = goal_handle.get_result_async()
        if not _wait_future(result_future, timeout=5.0):
            return {"error": "Execution timeout"}

        # Update joints immediately so next key uses the new state
        with self._lock:
            self._current_joints = [float(x) for x in target_q]

        pos = new_pos
        return {
            "ok": True,
            "ee": {"x": round(float(pos[0]), 4), "y": round(float(pos[1]), 4), "z": round(float(pos[2]), 4)},
        }

    def _level_suction(self) -> dict:
        """Orient suction gripper to point straight down."""
        with self._lock:
            joints = self._current_joints
        if joints is None:
            return {"error": "No joint states"}

        current_q = np.array(joints)
        link_frames, ee_frame = self._robot.forward_kinematics_all(current_q)
        T6 = link_frames[5]
        T_tip = T6 @ T_CHAIN
        tip_pos = T_tip[:3, 3]

        # Check if already vertical
        tip_z = T_tip[:3, 2]
        if np.allclose(tip_z, [0, 0, -1], atol=0.02):
            return {"ok": True, "msg": "Already vertical"}

        # Build desired tip frame: Z = down, keep current X projected to horizontal
        z_desired = np.array([0.0, 0.0, -1.0])
        x_current = T_tip[:3, 0]
        x_proj = x_current - x_current[2] * np.array([0, 0, 1])  # project to XY plane
        x_norm = np.linalg.norm(x_proj)
        if x_norm < 1e-6:
            x_desired = np.array([1.0, 0.0, 0.0])
        else:
            x_desired = x_proj / x_norm
        y_desired = np.cross(z_desired, x_desired)
        y_desired /= np.linalg.norm(y_desired)
        x_desired = np.cross(y_desired, z_desired)  # ensure orthogonal

        T_tip_desired = np.eye(4)
        T_tip_desired[:3, 0] = x_desired
        T_tip_desired[:3, 1] = y_desired
        T_tip_desired[:3, 2] = z_desired
        T_tip_desired[:3, 3] = tip_pos

        # Convert to EE target: T6_desired @ M = T_ee_desired
        T6_desired = T_tip_desired @ T_CHAIN_INV
        T_ee_desired = T6_desired @ self._robot.home_ee

        target_q = self._robot.inverse_kinematics(T_ee_desired)
        if target_q is None:
            return {"error": "IK failed for leveling"}

        # Build and send trajectory (reuse the same pattern as _move_ee)
        jt = JointTrajectory()
        jt.joint_names = JOINT_NAMES

        pt0 = JointTrajectoryPoint()
        pt0.positions = [float(x) for x in current_q]
        pt0.velocities = [0.0] * 6
        pt0.time_from_start = _seconds_to_duration(0.0)
        jt.points.append(pt0)

        pt1 = JointTrajectoryPoint()
        pt1.positions = [float(x) for x in target_q]
        pt1.velocities = [0.0] * 6
        pt1.time_from_start = _seconds_to_duration(MOVE_DURATION)
        jt.points.append(pt1)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = jt

        future = self._fjt_client.send_goal_async(goal)
        if not _wait_future(future):
            return {"error": "Goal send timeout"}

        goal_handle = future.result()
        if not goal_handle.accepted:
            return {"error": "Goal rejected"}

        result_future = goal_handle.get_result_async()
        if not _wait_future(result_future, timeout=5.0):
            return {"error": "Execution timeout"}

        with self._lock:
            self._current_joints = [float(x) for x in target_q]

        return {"ok": True, "msg": "Gripper leveled"}

    def _toggle_suction(self) -> dict:
        self._suction_on = not self._suction_on
        value = 0 if self._suction_on else 1

        req = WriteSingleIO.Request()
        req.address = SUCTION_IO_ADDRESS
        req.value = value

        future = self._io_client.call_async(req)
        if not _wait_future(future, timeout=5.0):
            self._suction_on = not self._suction_on  # revert
            return {"error": "Suction service timeout"}

        result = future.result()
        state = "ON" if self._suction_on else "OFF"
        if result.success:
            return {"ok": True, "suction": self._suction_on, "state": state}
        else:
            self._suction_on = not self._suction_on
            return {"error": f"Suction failed: {result.message}"}

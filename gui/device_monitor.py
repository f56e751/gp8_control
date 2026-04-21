"""디바이스 연결 상태 모니터."""
from __future__ import annotations

import threading
import time

from rclpy.node import Node
from sensor_msgs.msg import JointState, Image


class DeviceMonitor:
    """Tracks topic/service availability by monitoring last-received timestamps."""

    def __init__(self, node: Node) -> None:
        self._node = node
        self._lock = threading.Lock()
        self._last_joint_state: float | None = None
        self._last_camera: float | None = None

        # Subscribe to topics to track liveness (bridge output)
        from rclpy.qos import qos_profile_sensor_data
        self._node.create_subscription(
            JointState, "/joint_states_urdf", self._joint_cb, qos_profile_sensor_data,
        )
        self._node.create_subscription(
            Image, "/camera/color/image_raw", self._camera_cb, 10,
        )

    def _joint_cb(self, msg: JointState) -> None:
        with self._lock:
            self._last_joint_state = time.time()

    def _camera_cb(self, msg: Image) -> None:
        with self._lock:
            self._last_camera = time.time()

    def get_status(self) -> dict:
        """Return device status tree as dict."""
        now = time.time()
        with self._lock:
            robot_ok = (
                self._last_joint_state is not None
                and (now - self._last_joint_state) < 1.0
            )
            camera_ok = (
                self._last_camera is not None
                and (now - self._last_camera) < 2.0
            )

        # Check action server / service availability
        fjt_ok = self._check_topic_exists(
            "/motoman_gp8_controller/follow_joint_trajectory/_action/status"
        )
        suction_ok = self._check_service_exists("/write_single_io")
        moveit_ok = self._check_topic_exists("/move_action/_action/status")

        return {
            "robot": {
                "connected": robot_ok,
                "joint_states": robot_ok,
                "follow_joint_trajectory": fjt_ok,
            },
            "camera": {
                "connected": camera_ok,
            },
            "services": {
                "write_single_io": suction_ok,
                "moveit": moveit_ok,
            },
        }

    def _check_topic_exists(self, topic: str) -> bool:
        """Check if a topic has active publishers."""
        # get_publishers_info_by_topic returns list of TopicEndpointInfo
        return len(self._node.get_publishers_info_by_topic(topic)) > 0

    def _check_service_exists(self, service: str) -> bool:
        """Check if a service exists."""
        # get_service_names_and_types returns list of (name, types)
        names = [name for name, _ in self._node.get_service_names_and_types()]
        return service in names

"""AprilTag-based workspace calibration detector (ROS 2)."""

from __future__ import annotations

import numpy as np
from rclpy.node import Node
from apriltag_msgs.msg import AprilTagDetectionArray

from gp8_control.utils.lie_numpy import quat2SO3


class ApriltagDetector:
    """Detects AprilTag bundles for robot-camera calibration.

    Subscribes to /tag_detections and provides T_base (base tag SE(3) pose).
    """

    BASE_TAG_IDS = (10, 11, 12, 13, 14, 15)

    def __init__(self, node: Node) -> None:
        self._node = node
        self.T_base: np.ndarray | None = None
        self._node.create_subscription(
            AprilTagDetectionArray,
            "/tag_detections",
            self._callback,
            10,
        )

    def _callback(self, msg: AprilTagDetectionArray) -> None:
        detected_ids = {d.id[0] for d in msg.detections}
        all_base_detected = all(tid in detected_ids for tid in self.BASE_TAG_IDS)

        for detection in msg.detections:
            tag_id = detection.id
            if tag_id == tuple(self.BASE_TAG_IDS) and all_base_detected:
                pose = detection.pose.pose.pose
                self.T_base = self._pose_to_SE3(pose)

    @staticmethod
    def _pose_to_SE3(pose) -> np.ndarray:
        """Convert ROS Pose to 4x4 SE(3) matrix."""
        p = pose.position
        o = pose.orientation
        T = np.eye(4)
        T[:3, :3] = quat2SO3(np.array([[o.x, o.y, o.z, o.w]])).reshape(3, 3)
        T[:3, 3] = [p.x, p.y, p.z]
        return T

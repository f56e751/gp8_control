"""RGB-D camera manager (Intel RealSense via ROS 2)."""

from __future__ import annotations

import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class CameraManager:
    """Subscribes to RealSense RGB + Depth topics.

    Provides `rgb_data` and `depth_data` attributes updated by ROS 2 callbacks.
    """

    def __init__(self, node: Node) -> None:
        self._node = node
        self._bridge = CvBridge()
        self.rgb_data: np.ndarray | None = None
        self.depth_data: np.ndarray | None = None

        self._node.create_subscription(
            Image, "/camera/color/image_raw", self._rgb_cb, 10,
        )
        self._node.create_subscription(
            Image, "/camera/aligned_depth_to_color/image_raw", self._depth_cb, 10,
        )

    def _rgb_cb(self, msg: Image) -> None:
        try:
            self.rgb_data = self._bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self._node.get_logger().error(f"RGB conversion error: {e}")

    def _depth_cb(self, msg: Image) -> None:
        try:
            self.depth_data = self._bridge.imgmsg_to_cv2(msg, "passthrough")
        except Exception as e:
            self._node.get_logger().error(f"Depth conversion error: {e}")

    @staticmethod
    def compute_xyz(
        depth_img: np.ndarray,
        intrinsic_matrix: np.ndarray,
    ) -> np.ndarray:
        """Convert depth image to XYZ point cloud using camera intrinsics.

        Returns:
            XYZ image, shape (H, W, 3) in meters.
        """
        h, w = depth_img.shape[:2]
        indices = np.indices((h, w), dtype=np.float32).transpose(1, 2, 0)
        z = depth_img.astype(np.float32)
        x = (indices[..., 1] - intrinsic_matrix[0, 2]) * z / intrinsic_matrix[0, 0]
        y = (indices[..., 0] - intrinsic_matrix[1, 2]) * z / intrinsic_matrix[1, 1]
        return np.stack([x, y, z], axis=-1) / 1000.0

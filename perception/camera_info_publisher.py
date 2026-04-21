"""Calibrated camera info publisher (ROS 2).

Loads camera intrinsics from a YAML calibration file and publishes
CameraInfo messages synced with incoming image timestamps.
"""

from __future__ import annotations

import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


class CameraInfoPublisher(Node):
    """Publishes calibrated CameraInfo synced with image timestamps."""

    def __init__(self) -> None:
        super().__init__("camera_info_publisher")

        self.declare_parameter("camera_info_path", "")
        self.declare_parameter("sync_topic", "image_raw")

        yaml_path = self.get_parameter("camera_info_path").get_parameter_value().string_value
        sync_topic = self.get_parameter("sync_topic").get_parameter_value().string_value

        if not yaml_path:
            self.get_logger().error("camera_info_path parameter is required.")
            return

        self._cam_info = self._load_calibration(yaml_path)
        self._pub = self.create_publisher(CameraInfo, "camera_info_calibrated", 10)
        self.create_subscription(Image, sync_topic, self._image_cb, 10)
        self.get_logger().info(
            f"Publishing camera_info_calibrated synced with: {sync_topic}"
        )

    def _load_calibration(self, yaml_path: str) -> CameraInfo:
        """Load camera calibration from YAML file."""
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        msg = CameraInfo()
        msg.width = data["image_width"]
        msg.height = data["image_height"]
        msg.distortion_model = data["distortion_model"]
        msg.d = [float(x) for x in data["distortion_coefficients"]["data"]]
        msg.k = [float(x) for x in data["camera_matrix"]["data"]]
        msg.r = [float(x) for x in data["rectification_matrix"]["data"]]
        msg.p = [float(x) for x in data["projection_matrix"]["data"]]
        return msg

    def _image_cb(self, msg: Image) -> None:
        """Publish CameraInfo with same timestamp as incoming image."""
        self._cam_info.header.stamp = msg.header.stamp
        self._cam_info.header.frame_id = msg.header.frame_id
        self._pub.publish(self._cam_info)


def main():
    rclpy.init()
    node = CameraInfoPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

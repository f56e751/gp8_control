"""SAM + DINO remote detection client (ROS 2).

Sends RGB + Depth images to a remote GPU server via TCP,
receives 3D object positions and class names.
Publishes overlay image with detection results to /detection_overlay.
"""

from __future__ import annotations

import json
import socket
import struct
import time

import cv2
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image

# Per-class colors (BGR)
CLASS_COLORS = {
    "transparent": (255, 200, 0),
    "metal": (0, 140, 255),
    "plastic": (0, 255, 100),
}
DEFAULT_COLOR = (200, 200, 200)
MARKER_RADIUS = 12
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _recv_all(sock: socket.socket, length: int) -> bytes:
    """Read exactly `length` bytes from socket."""
    data = b""
    while len(data) < length:
        chunk = sock.recv(length - len(data))
        if not chunk:
            raise EOFError("Socket closed prematurely")
        data += chunk
    return data


def _project_3d_to_pixel(
    position_cam: list,
    fx: float, fy: float, cx: float, cy: float,
) -> tuple[int, int] | None:
    """Project 3D camera-frame position to 2D pixel coordinates.

    Args:
        position_cam: [x, y, z] in camera frame (meters).
        fx, fy, cx, cy: camera intrinsics.

    Returns:
        (u, v) pixel coordinates, or None if behind camera.
    """
    x, y, z = position_cam
    if z <= 0.01:
        return None
    u = int(fx * x / z + cx)
    v = int(fy * y / z + cy)
    return u, v


class SAMClient:
    """Client that sends camera data to remote SAM+DINO server.

    On each camera_info callback, sends RGB+Depth to the server and
    updates `positions`, `class_names`, and `delay`.
    Publishes overlay image to /detection_overlay.
    """

    def __init__(self, node: Node, server_ip: str, server_port: int) -> None:
        self._node = node
        self.bridge = CvBridge()
        self._color_image: np.ndarray | None = None
        self._depth_image: np.ndarray | None = None

        self.positions: list | None = None
        self.class_names: list | None = None
        self.delay: float | None = None

        # Camera intrinsics (updated each frame)
        self._fx = self._fy = self._cx = self._cy = 0.0

        self._node.create_subscription(
            Image, "/camera/color/image_raw", self._color_cb, 10,
        )
        self._node.create_subscription(
            Image, "/camera/aligned_depth_to_color/image_raw", self._depth_cb, 10,
        )
        self._node.create_subscription(
            CameraInfo, "/camera/color/camera_info", self._info_cb, 10,
        )

        # Overlay publisher
        self._overlay_pub = self._node.create_publisher(Image, "/detection_overlay", 10)

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((server_ip, server_port))

    def _color_cb(self, msg: Image) -> None:
        self._color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def _depth_cb(self, msg: Image) -> None:
        self._depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def _info_cb(self, msg: CameraInfo) -> None:
        if self._color_image is None or self._depth_image is None:
            return

        # Store intrinsics
        self._fx, self._fy = msg.k[0], msg.k[4]
        self._cx, self._cy = msg.k[2], msg.k[5]

        self.positions, self.class_names, self.delay = None, None, None
        t_start = time.time()

        # Encode and send
        color_bytes = cv2.imencode(".jpg", self._color_image)[1].tobytes()
        depth_bytes = self._depth_image.astype(np.uint16).tobytes()
        intrinsics = struct.pack("!4f", self._fx, self._fy, self._cx, self._cy)

        self._sock.sendall(struct.pack("!I", len(color_bytes)))
        self._sock.sendall(color_bytes)
        self._sock.sendall(struct.pack("!I", len(depth_bytes)))
        self._sock.sendall(depth_bytes)
        self._sock.sendall(intrinsics)

        # Receive results
        n_bytes = struct.unpack("!I", _recv_all(self._sock, 4))[0]
        raw = _recv_all(self._sock, n_bytes)
        results = json.loads(raw.decode("utf-8"))

        self.positions = results["positions"]
        self.class_names = results["class_names"]
        self.delay = time.time() - t_start

        # Publish overlay
        self._publish_overlay()

    def _publish_overlay(self) -> None:
        """Draw detection results on the current image and publish."""
        if self._color_image is None or not self.positions or not self.class_names:
            return

        overlay = self._color_image.copy()

        for pos, cls in zip(self.positions, self.class_names):
            pixel = _project_3d_to_pixel(pos, self._fx, self._fy, self._cx, self._cy)
            if pixel is None:
                continue

            u, v = pixel
            color = CLASS_COLORS.get(cls, DEFAULT_COLOR)

            # Marker circle
            cv2.circle(overlay, (u, v), MARKER_RADIUS, color, 2)
            cv2.circle(overlay, (u, v), 3, color, -1)

            # Label
            label = f"{cls} z={pos[2]:.2f}m"
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
            label_x, label_y = u + MARKER_RADIUS + 4, v - 4
            cv2.rectangle(overlay, (label_x - 2, label_y - th - 4),
                          (label_x + tw + 2, label_y + 4), color, -1)
            cv2.putText(overlay, label, (label_x, label_y),
                        FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Latency info
        if self.delay is not None:
            info = f"SAM latency: {self.delay * 1000:.0f}ms | {len(self.positions)} objects"
            cv2.putText(overlay, info, (10, 25), FONT, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

        # Publish
        overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
        self._overlay_pub.publish(overlay_msg)

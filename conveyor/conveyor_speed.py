"""Live conveyor-belt speed tracker.

Subscribes to a ``std_msgs/Float64`` topic (default ``/conveyor/speed``)
and exposes the latest value via ``current``. Keeps a fallback so callers
get a sensible value before the first message arrives, and warns once if
updates stop flowing for ``stale_seconds``.
"""

from __future__ import annotations

import time

from rclpy.node import Node
from std_msgs.msg import Float64


class ConveyorSpeedTracker:
    def __init__(
        self,
        node: Node,
        topic: str,
        fallback_speed: float,
        stale_seconds: float,
    ) -> None:
        self._node = node
        self._topic = topic
        self._stale_seconds = stale_seconds
        self._speed = fallback_speed
        self._last_msg_time: float | None = None
        self._stale_warned = False

        node.create_subscription(Float64, topic, self._on_msg, 1)
        node.get_logger().info(
            f"Subscribed to {topic} "
            f"(fallback {fallback_speed:.3f} m/s until first message)"
        )

    @property
    def current(self) -> float:
        return self._speed

    def _on_msg(self, msg: Float64) -> None:
        first = self._last_msg_time is None
        self._speed = float(msg.data)
        self._last_msg_time = time.time()
        if first:
            self._node.get_logger().info(
                f"Live conveyor speed acquired: {self._speed:.4f} m/s"
            )
        if self._stale_warned:
            self._stale_warned = False

    def check_freshness(self) -> None:
        """Log once if no message has arrived within the stale window."""
        if self._last_msg_time is None or self._stale_warned:
            return
        if time.time() - self._last_msg_time > self._stale_seconds:
            self._node.get_logger().warn(
                f"No {self._topic} update for >{self._stale_seconds:.1f}s; "
                f"using last value {self._speed:.4f} m/s"
            )
            self._stale_warned = True

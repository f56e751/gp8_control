"""Live conveyor-belt encoder telemetry.

Subscribes to a ``std_msgs/Float64`` topic (default ``/conveyor/speed``)
and the encoder's cumulative ``/conveyor/distance_mm``.  Speed is used for
future ETA projection; cumulative distance is used for past motion so queued
objects do not accumulate ``current_speed * age`` error.
"""

from __future__ import annotations

import time
from collections import deque

from rclpy.node import Node
from std_msgs.msg import Float64


class ConveyorSpeedTracker:
    def __init__(
        self,
        node: Node,
        topic: str,
        fallback_speed: float,
        stale_seconds: float,
        distance_topic: str = "/conveyor/distance_mm",
    ) -> None:
        self._node = node
        self._topic = topic
        self._stale_seconds = stale_seconds
        self._speed = fallback_speed
        self._distance_topic = distance_topic
        self._distance_m: float | None = None
        self._raw_distance_m: float | None = None
        self._last_distance_time: float | None = None
        self._distance_samples = deque(maxlen=400)
        self._last_msg_time: float | None = None
        self._stale_warned = False

        node.create_subscription(Float64, topic, self._on_msg, 1)
        node.create_subscription(Float64, distance_topic, self._on_distance, 1)
        node.get_logger().info(
            f"Subscribed to {topic} and {distance_topic} "
            f"(fallback {fallback_speed:.3f} m/s until first message)"
        )

    @property
    def current(self) -> float:
        return self._speed

    @property
    def distance_m(self) -> float | None:
        """Continuous encoder distance at now, or ``None`` before first data."""
        return self.distance_at(time.time())

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

    def _on_distance(self, msg: Float64) -> None:
        """Unwrap the encoder node's cumulative distance across node resets.

        Normal 20 Hz samples move only a few centimetres.  A jump larger than a
        speed/time-derived plausibility window is treated as the ESP32/node
        counter restarting; it establishes a new raw baseline without moving
        the continuous control coordinate.
        """
        now = time.time()
        raw = float(msg.data) / 1000.0
        if self._raw_distance_m is None:
            self._distance_m = raw
        else:
            delta = raw - self._raw_distance_m
            dt = max(now - float(self._last_distance_time), 0.0)
            plausible_step = max(0.1, abs(self._speed) * dt * 4.0 + 0.02)
            if abs(delta) <= plausible_step:
                self._distance_m = float(self._distance_m) + delta
            else:
                self._node.get_logger().warn(
                    f"Encoder distance reset/jump {delta:+.3f}m; preserving "
                    "continuous tracked-object coordinates"
                )
        self._raw_distance_m = raw
        self._last_distance_time = now
        self._distance_samples.append((now, float(self._distance_m)))

    def distance_at(self, wall_time: float) -> float | None:
        """Interpolate continuous encoder distance at a wall-clock timestamp.

        Camera detections are valid at their ``receipt_time``.  Using the latest
        encoder sample directly would double-count motion between receipt and
        intake, so anchor each track to the distance at that timestamp.
        """
        if not self._distance_samples:
            return None
        samples = self._distance_samples
        if wall_time <= samples[0][0]:
            return samples[0][1]
        if wall_time >= samples[-1][0]:
            # If distance messages pause, preserve the legacy speed integration
            # instead of freezing every tracked object at the last sample.
            dt = float(wall_time - samples[-1][0])
            return samples[-1][1] + self._speed * dt
        for i in range(1, len(samples)):
            t1, d1 = samples[i]
            if t1 >= wall_time:
                t0, d0 = samples[i - 1]
                frac = (wall_time - t0) / max(t1 - t0, 1e-9)
                return d0 + frac * (d1 - d0)
        return samples[-1][1]

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

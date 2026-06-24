"""Fake conveyor + detection publisher for software-in-the-loop simulation.

Mimics the ``camera_debug`` node WITHOUT a camera: it spawns synthetic objects
upstream on the belt, advances them in -Y at a constant speed, and republishes
the *corrected base-frame* detections on ``/camera_debug/detections`` every
frame (re-detecting every object each frame, like the real camera, so the app's
spatial dedup / re-anchor is exercised). It also publishes the belt speed on
``/conveyor/speed``.

This is the perception half of the C1 SIL: pair it with the queue-mode
``mock_robot`` and the app to verify the full pipeline
(detection -> intake/dedup -> selection -> push/throw routing -> skill ->
motion) with no hardware. See ``launch/sim_bringup.launch.py``.

Wall clock (``time.time()``) is used throughout so receipt_time shares the app's
time base (the app extrapolates objects forward from receipt_time).

Params (ros2 run gp8_control fake_belt --ros-args -p belt_speed:=0.1 ...):
  belt_speed      m/s, belt travels -Y                      (default 0.12)
  spawn_interval  s between spawns                           (default 5.0)
  publish_rate    Hz detection frames                        (default 15.0)
  spawn_y         m, upstream start (out of reach is fine)   (default 0.9)
  despawn_y       m, downstream cull                         (default -0.8)
  lane_x          m, forward reach of the lane               (default 0.45)
  grasp_z         m, belt-surface grasp height               (default 0.062)
  aim_dz          m, approach hover above grasp              (default 0.08)
  classes         round-robin class names                    (default
                  ["transparent", "metal"] -> throw / push)

  ros2 run gp8_control fake_belt
"""

from __future__ import annotations

import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, String


class FakeBelt(Node):
    def __init__(self) -> None:
        super().__init__("fake_belt")

        self.declare_parameter("belt_speed", 0.12)
        self.declare_parameter("spawn_interval", 5.0)
        self.declare_parameter("publish_rate", 15.0)
        self.declare_parameter("spawn_y", 0.9)
        self.declare_parameter("despawn_y", -0.8)
        self.declare_parameter("lane_x", 0.45)
        self.declare_parameter("grasp_z", 0.062)
        self.declare_parameter("aim_dz", 0.08)
        self.declare_parameter("classes", ["transparent", "metal"])

        g = self.get_parameter
        self._belt = float(g("belt_speed").value)
        self._spawn_interval = float(g("spawn_interval").value)
        rate = float(g("publish_rate").value)
        self._spawn_y = float(g("spawn_y").value)
        self._despawn_y = float(g("despawn_y").value)
        self._lane_x = float(g("lane_x").value)
        self._grasp_z = float(g("grasp_z").value)
        self._aim_dz = float(g("aim_dz").value)
        self._classes = list(g("classes").value) or ["transparent"]

        self._det_pub = self.create_publisher(String, "/camera_debug/detections", 10)
        self._belt_pub = self.create_publisher(Float64, "/conveyor/speed", 10)

        self._objects: list[dict] = []
        self._spawn_count = 0
        self._last_spawn = 0.0

        self._timer = self.create_timer(1.0 / max(1.0, rate), self._tick)
        self.get_logger().info(
            f"fake_belt: belt {self._belt:.3f} m/s, spawn every "
            f"{self._spawn_interval:.1f}s, classes {self._classes}, "
            f"lane x={self._lane_x:.2f}, y {self._spawn_y:+.2f}->{self._despawn_y:+.2f}"
        )

    def _spawn(self, now: float) -> None:
        cls = self._classes[self._spawn_count % len(self._classes)]
        # Deterministic small lane jitter so successive objects aren't collinear
        # (and dedup can't merge two distinct objects spawned at the same x).
        jitter = 0.06 * ((self._spawn_count % 3) - 1)   # -0.06, 0, +0.06
        self._objects.append({
            "id": self._spawn_count,
            "x": self._lane_x + jitter,
            "y0": self._spawn_y,
            "t0": now,
            "class": cls,
        })
        self.get_logger().info(
            f"spawn #{self._spawn_count} class={cls} x={self._lane_x + jitter:+.2f}"
        )
        self._spawn_count += 1

    def _tick(self) -> None:
        now = time.time()
        self._belt_pub.publish(Float64(data=self._belt))

        if now - self._last_spawn >= self._spawn_interval:
            self._spawn(now)
            self._last_spawn = now

        dets = []
        alive = []
        for o in self._objects:
            y = o["y0"] - self._belt * (now - o["t0"])
            if y < self._despawn_y:
                continue
            alive.append(o)
            x, z = o["x"], self._grasp_z
            dets.append({
                "class": o["class"],
                "base_grasp": [x, y, z],
                "base_aim": [x, y, z + self._aim_dz],
                "cam": [0.0, 0.0, 0.0],
                "in_workspace": True,
            })
        self._objects = alive

        snap = {"receipt_time": now, "belt_mps": self._belt, "detections": dets}
        self._det_pub.publish(String(data=json.dumps(snap)))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FakeBelt()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

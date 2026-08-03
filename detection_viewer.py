#!/usr/bin/env python3
"""Standalone tracked-object viewer for the perception stream (read-only).

Subscribes to ``/camera_debug/detections`` (published by the already-running
``camera_debug`` node) and runs the SAME dedup/tracking used by the app
(``DetectionIntake`` + ``TrackedObjectQueue``) — but with no robot, no MoveIt,
no motion. One row per physical object (stable ``id``), refreshed in place,
plus a scrolling history of NEW/DROP events underneath.

Safe to run any time: it publishes nothing and commands nothing, so the robot
can be off, or the full live stack can be running alongside — it just listens.

Run (no colcon rebuild needed):

    PYTHONPATH=$HOME/ros2_ws/src:$PYTHONPATH python3 -m gp8_control.detection_viewer

or, after the next ``colcon build``:

    ros2 run gp8_control detection_viewer
"""

from __future__ import annotations

import json
import sys
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

from gp8_control.config import Config
from gp8_control.conveyor import ConveyorSpeedTracker
from gp8_control.perception.detection_intake import DetectionIntake
from gp8_control.tracking import TrackedObjectQueue

RENDER_HZ = 10.0
EVENT_HISTORY = 12   # NEW/DROP lines kept on screen


class DetectionViewerNode(Node):
    def __init__(self) -> None:
        super().__init__("gp8_detection_viewer")
        cfg = Config()
        # Same queue/dedup parameters as GP8App so what you see here is what
        # the app would track (coarse drop line at the downstream reach edge).
        self._queue = TrackedObjectQueue(
            cfg.MAX_REACH, drop_below_y=-cfg.MAX_REACH,
        )
        self._intake = DetectionIntake(cfg.OBJECT_MATCH_EPSILON)
        self._conveyor = ConveyorSpeedTracker(
            self, cfg.CONVEYOR_TOPIC, cfg.CONVEYOR_SPEED,
            cfg.CONVEYOR_STALE_SECONDS,
            distance_topic=cfg.CONVEYOR_DISTANCE_TOPIC,
        )
        self._latest: dict | None = None
        self._recv_mono: float = 0.0
        self._events: list[str] = []
        self.create_subscription(
            String, "/camera_debug/detections", self._on_detections, 10,
        )
        self.create_timer(1.0 / RENDER_HZ, self._tick)
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()

    def _on_detections(self, msg: String) -> None:
        try:
            self._latest = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        self._recv_mono = time.monotonic()

    # ------------------------------------------------------------------
    def _event(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self._events.append(f" [{stamp}] {text}")
        del self._events[:-EVENT_HISTORY]

    def _tick(self) -> None:
        now = time.time()
        v = self._conveyor.current
        self._conveyor.check_freshness()

        before = {o.track_id: o for o in self._queue._objects}
        # logger=None: NEW events are rendered into the history pane instead of
        # ROS log lines (which the in-place ANSI redraw would wipe anyway).
        receipt_time = (
            float(self._latest.get("receipt_time", now))
            if self._latest is not None else now
        )
        self._intake.ingest(
            self._latest, self._queue, None, v, logger=None,
            belt_distance_m=self._conveyor.distance_at(receipt_time),
        )
        for obj in self._queue._objects:
            if obj.track_id not in before:
                self._event(
                    f"NEW  id={obj.track_id} {obj.class_name:<12s} "
                    f"x={obj.T_grasp_base[0, 3]:+.3f} "
                    f"y={obj.T_grasp_base[1, 3]:+.3f} conf={obj.conf:.2f}"
                )
        # Consume the snapshot: each message is folded in exactly once, so a
        # stalled stream can't keep re-anchoring tracks with stale positions.
        self._latest = None

        alive_before_drop = {o.track_id for o in self._queue._objects}
        self._queue.update(now, v, belt_distance_m=self._conveyor.distance_m)
        alive = {o.track_id for o in self._queue._objects}
        for tid in sorted(alive_before_drop - alive):
            self._event(f"DROP id={tid} (past reach line)")

        self._render(now, v)

    # ------------------------------------------------------------------
    def _render(self, now: float, v: float) -> None:
        age = (
            f"{time.monotonic() - self._recv_mono:5.1f}s"
            if self._recv_mono else "  n/a"
        )
        out = ["\033[H\033[J"]
        out.append("=== GP8 Detection Viewer — tracked objects (read-only) ===\n")
        out.append(
            f" belt: {v:6.3f} m/s   last frame: {age} ago   "
            f"tracked: {len(self._queue)}\n"
        )
        out.append(" position source: encoder cumulative distance\n\n")
        if not self._queue:
            out.append(" (no tracked objects)\n")
        else:
            out.append(
                "  id  class         conf      x        y_now      z     age\n"
            )
            for obj in self._queue._objects:
                y_now = float(
                    obj.y_at(now, v, self._conveyor.distance_at(now))
                )
                out.append(
                    f" {obj.track_id:3d}  "
                    f"{obj.class_name[:12]:<12s}  "
                    f"{obj.conf:.2f}  "
                    f"{obj.T_grasp_base[0, 3]:+.3f}  "
                    f"{y_now:+.3f}  "
                    f"{obj.T_grasp_base[2, 3]:+.3f}  "
                    f"{now - obj.detect_time:5.1f}s\n"
                )
        out.append(f"\n--- events (last {EVENT_HISTORY}) ---\n")
        if self._events:
            out.extend(f"{line}\n" for line in self._events)
        else:
            out.append(" (none yet)\n")
        sys.stdout.write("".join(out))
        sys.stdout.flush()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DetectionViewerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

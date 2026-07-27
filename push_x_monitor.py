"""Live monitor: print each camera detection's base-frame X and the push
run-up verdict as objects appear on the belt.

Debug helper for the push_skill ``available <= 0`` pass case
(skills/push_skill.py::_compute_backswing_poses): place an object on the
conveyor and watch, per published frame, the raw across-belt ``cx``, the mapped
base X, and the remaining run-up room ``available = x - PUSH_BACKSWING_MIN_X``.

Note the geometry this makes visible: with the current extrinsics
(x = REFERENCE_X_BASE + SIGN*SCALE*cx, workspace |cx| < WORKSPACE_X_ABS) the
in-workspace base X range is an OPEN interval whose lower edge equals
PUSH_BACKSWING_MIN_X — so "x <= 0.25" only occurs AT the belt edge, right where
the in_workspace filter starts dropping the detection entirely.

Run (camera_debug node must be publishing /camera_debug/detections):

    source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
    PYTHONPATH=$HOME/ros2_ws/src \
      ~/ros2_ws/src/gp8_control/.venv/bin/python -m gp8_control.push_x_monitor

Subscribe-only: never touches the robot or the control loop.
"""

from __future__ import annotations

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from gp8_control.perception import extrinsics

try:
    from gp8_control.skills.push_skill import PUSH_BACKSWING_MIN_X
except Exception:      # import chain unavailable outside the app env
    PUSH_BACKSWING_MIN_X = 0.25


class PushXMonitor(Node):

    def __init__(self) -> None:
        super().__init__("push_x_monitor")
        self._frame = 0
        self.create_subscription(
            String, "/camera_debug/detections", self._on_snapshot, 10
        )
        sx = extrinsics.SIGN_CX_TO_BASE_X * extrinsics.SCALE_CX_TO_BASE_X
        lo = extrinsics.REFERENCE_X_BASE - abs(sx) * extrinsics.WORKSPACE_X_ABS
        hi = extrinsics.REFERENCE_X_BASE + abs(sx) * extrinsics.WORKSPACE_X_ABS
        print(
            f"cx→base_X: x = {extrinsics.REFERENCE_X_BASE:+.3f} + ({sx:+.3f})·cx"
            f"   in-workspace x ∈ ({lo:+.3f}, {hi:+.3f})"
        )
        print(
            f"run-up floor PUSH_BACKSWING_MIN_X = {PUSH_BACKSWING_MIN_X:.3f}"
            f"   (x ≤ floor → available=0 → push passes the object)\n"
        )

    def _on_snapshot(self, msg: String) -> None:
        try:
            snap = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        detections = snap.get("detections", [])
        if not detections:
            return
        self._frame += 1
        for d in detections:
            cx, cy = (d.get("cam") or [0.0, 0.0])[:2]
            x, y, _ = (d.get("base_grasp") or [0.0, 0.0, 0.0])[:3]
            available = float(x) - PUSH_BACKSWING_MIN_X
            if not d.get("in_workspace"):
                verdict = "OUT-OF-WS (intake drops this detection)"
            elif available <= 0.0:
                verdict = f"PUSH-PASS  available={available:+.3f} ≤ 0"
            else:
                verdict = f"push-ok    available={available:+.3f}"
            print(
                f"[{self._frame:05d}] {d.get('class', '?'):<12}"
                f"conf={float(d.get('confidence', -1.0)):.2f}  "
                f"cx={float(cx):+.3f} cy={float(cy):+.3f}  "
                f"base x={float(x):+.3f} y={float(y):+.3f}  {verdict}"
            )


def main() -> None:
    rclpy.init()
    node = PushXMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

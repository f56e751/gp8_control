#!/usr/bin/env python3
"""TUI visualization of the GP8 conveyor belt state.

Subscribes to /gp8_manager/tracked_state (JSON in std_msgs/String) published
by gp8_manager and renders a live ASCII strip of the belt — refreshed in
place on the terminal — showing the camera, the pick point, and every
tracked object at its current predicted y position.

Run alongside the bringup in a separate terminal:

    ros2 run gp8_control belt_viz
"""

from __future__ import annotations

import json
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


WIDTH = 78
Y_HIGH = 3.0      # upstream (m) — left edge of the strip
Y_LOW = -0.5      # downstream past pick — right edge


def _y_to_col(y: float) -> int:
    """Map belt-Y in [Y_LOW, Y_HIGH] to a column (high Y on the left)."""
    if y >= Y_HIGH:
        return 0
    if y <= Y_LOW:
        return WIDTH - 1
    frac = (Y_HIGH - y) / (Y_HIGH - Y_LOW)
    return int(round(frac * (WIDTH - 1)))


def _class_marker(cls: str) -> str:
    return (cls[:1] or "?").upper()


class BeltVizNode(Node):
    def __init__(self) -> None:
        super().__init__("gp8_belt_viz")
        self._state: dict | None = None
        self._last_render = 0.0
        self.create_subscription(
            String, "/gp8_manager/tracked_state", self._on_state, 10
        )
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()
        self.get_logger().info(
            "belt_viz subscribed to /gp8_manager/tracked_state"
        )

    def _on_state(self, msg: String) -> None:
        try:
            self._state = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        # Cap the render rate so a chatty publisher doesn't flicker the screen.
        now = time.time()
        if now - self._last_render < 0.08:
            return
        self._last_render = now
        self._render()

    def _render(self) -> None:
        s = self._state or {}
        v = float(s.get("belt_mps", 0.0))
        objs = s.get("objects", []) or []
        intercept_y = float(s.get("intercept_y", 0.0))
        status = s.get("status", "")
        detail = s.get("status_detail", "")

        # Belt row with camera + pick markers.
        row = list("─" * WIDTH)
        row[_y_to_col(Y_HIGH - 0.05)] = "["    # camera at the upstream edge
        row[_y_to_col(intercept_y)] = "R"       # pick point / robot

        # Object markers on a parallel row so they don't overwrite the pick R.
        obj_row = list(" " * WIDTH)
        for o in objs:
            col = _y_to_col(float(o.get("y_now", 0.0)))
            obj_row[col] = _class_marker(o.get("class", "?"))

        # Axis labels along the bottom — one tick per integer-ish y.
        axis = list(" " * WIDTH)
        for y in (Y_HIGH, 2.0, 1.0, 0.0, Y_LOW):
            col = _y_to_col(y)
            label = f"{y:+.1f}"
            start = max(0, min(WIDTH - len(label), col - len(label) // 2))
            for i, ch in enumerate(label):
                axis[start + i] = ch

        out = ["\033[H\033[J"]  # cursor home + clear-to-end
        out.append("=== GP8 Conveyor Belt — live ===\n")
        out.append(
            f" belt: {v:6.3f} m/s   "
            f"status: {status:<12s} {detail}\n"
        )
        out.append(
            f" objects: {len(objs)}   "
            f"([=camera   R=pick   class letter=object)\n\n"
        )
        out.append(
            " belt direction:  upstream  ─────────────────────→  downstream\n\n"
        )
        out.append(f"  {''.join(row)}\n")
        out.append(f"  {''.join(obj_row)}\n")
        out.append(f"  {''.join(axis)}\n\n")
        if objs:
            out.append(" Tracked objects (current predicted positions):\n")
            for o in objs:
                out.append(
                    f"   {str(o.get('class','?')):<12s}"
                    f" y={float(o.get('y_now',0)):+7.3f} m"
                    f"  x={float(o.get('x',0)):+6.3f}"
                    f"  age={float(o.get('age_s',0)):5.2f}s\n"
                )
        else:
            out.append(" (no tracked objects)\n")
        sys.stdout.write("".join(out))
        sys.stdout.flush()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BeltVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\n")
        sys.stdout.flush()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

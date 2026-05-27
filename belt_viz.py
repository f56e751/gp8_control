#!/usr/bin/env python3
"""TUI visualization of the GP8 conveyor belt state.

Subscribes to /gp8_manager/tracked_state (JSON in std_msgs/String) published
by gp8_manager and renders a live ASCII strip of the belt — refreshed in
place — showing the camera, the pick point, and every tracked object as a
moving "●" marker that flows downstream between messages by extrapolating
y_now with the live belt speed.

Run alongside the bringup in a separate terminal:

    ros2 run gp8_control belt_viz
"""

from __future__ import annotations

import json
import shutil
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


Y_HIGH = 3.0       # upstream (m) — left edge of the strip
Y_LOW = -0.5       # downstream past pick — right edge
RENDER_HZ = 20.0   # animation update rate
OBJECT_GLYPH = "●"
TARGET_GLYPH = "◉"     # currently-executing pick target


def _y_to_col(y: float, width: int) -> int:
    """Map belt-Y in [Y_LOW, Y_HIGH] to a column (high Y on the left)."""
    if y >= Y_HIGH:
        return 0
    if y <= Y_LOW:
        return width - 1
    frac = (Y_HIGH - y) / (Y_HIGH - Y_LOW)
    return int(round(frac * (width - 1)))


class BeltVizNode(Node):
    def __init__(self) -> None:
        super().__init__("gp8_belt_viz")
        self._state: dict | None = None
        self._recv_mono: float = 0.0
        self.create_subscription(
            String, "/gp8_manager/tracked_state", self._on_state, 10
        )
        # Render on a timer instead of on-message so objects keep animating
        # between publisher updates (extrapolated by belt speed).
        self.create_timer(1.0 / RENDER_HZ, self._render)
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
        self._recv_mono = time.monotonic()

    def _render(self) -> None:
        # Terminal width may change as the user resizes; re-measure each frame.
        term_cols = shutil.get_terminal_size((80, 24)).columns
        width = max(40, term_cols - 4)

        s = self._state
        out = ["\033[H\033[J"]   # cursor home + clear-to-end

        if s is None:
            out.append("=== GP8 Conveyor Belt — live ===\n")
            out.append(" (waiting for /gp8_manager/tracked_state ...)\n")
            sys.stdout.write("".join(out))
            sys.stdout.flush()
            return

        v = float(s.get("belt_mps", 0.0))
        objs = s.get("objects", []) or []
        intercept_y = float(s.get("intercept_y", 0.0))
        status = str(s.get("status", ""))
        detail = str(s.get("status_detail", ""))

        # Belt-row: dashes with camera "[" and pick "R" overlaid.
        row = list("─" * width)
        row[_y_to_col(Y_HIGH - 0.05, width)] = "["
        row[_y_to_col(intercept_y, width)] = "R"

        # Object row: extrapolate y forward using belt speed since the
        # publisher's snapshot was received. Belt travels -Y, so y decreases
        # over time → moves to the right on screen.
        dt = max(0.0, time.monotonic() - self._recv_mono)
        obj_row = list(" " * width)
        extrapolated = []
        for o in objs:
            y0 = float(o.get("y_now", 0.0))
            y = y0 - v * dt
            col = _y_to_col(y, width)
            is_target = bool(o.get("is_target", False))
            glyph = TARGET_GLYPH if is_target else OBJECT_GLYPH
            # Don't let a queued ● overwrite an active ◉ at the same column.
            if obj_row[col] != TARGET_GLYPH:
                obj_row[col] = glyph
            extrapolated.append((o.get("class", "?"), y,
                                 float(o.get("x", 0.0)),
                                 float(o.get("age_s", 0.0)) + dt,
                                 is_target))

        # Axis labels along the bottom.
        axis = list(" " * width)
        for y in (Y_HIGH, 2.0, 1.0, 0.0, Y_LOW):
            col = _y_to_col(y, width)
            label = f"{y:+.1f}"
            start = max(0, min(width - len(label), col - len(label) // 2))
            for i, ch in enumerate(label):
                axis[start + i] = ch

        out.append("=== GP8 Conveyor Belt — live ===\n")
        out.append(
            f" belt: {v:6.3f} m/s   "
            f"status: {status:<12s} {detail}\n"
        )
        out.append(
            f" objects: {len(objs)}   "
            f"([=camera   R=pick   ●=queued   ◉=active target)\n\n"
        )
        out.append(
            " upstream  ──────  belt direction  ──────→  downstream\n\n"
        )
        out.append(f"  {''.join(row)}\n")
        out.append(f"  {''.join(obj_row)}\n")
        out.append(f"  {''.join(axis)}\n\n")
        if extrapolated:
            out.append(" Tracked objects (live extrapolated positions):\n")
            for cls, y, x, age, is_target in extrapolated:
                tag = " ◉" if is_target else "  "
                out.append(
                    f"  {tag}{str(cls):<12s}"
                    f" y={y:+7.3f} m"
                    f"  x={x:+6.3f}"
                    f"  age={age:5.2f}s\n"
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

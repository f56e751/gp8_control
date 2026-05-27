#!/usr/bin/env python3
"""Camera debug node — owns the perception → base-frame correction pipeline.

Reads the camera's HTTP NDJSON detection stream, applies the camera→base
transform, back-projects each detection's Y by ``belt_speed × perception_delay``,
and publishes the corrected detection list on ``/camera_debug/detections``.
app.py subscribes to that topic instead of re-doing the corrections itself.

A live TUI renders the raw camera positions next to the corrected base
positions so it's obvious when the camera sends garbage (e.g., [0,0,0]
which maps to the constant base x=0.425).

Run alongside the bringup in its own terminal:

    ros2 run gp8_control camera_debug
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, String

from gp8_control.perception import extrinsics
from gp8_control.perception.perception_client import stream_detections


# Perception stream URL — same env-var override as gp8_control.app.Config.
PERCEPTION_URL_DEFAULT = "http://147.46.175.15:8080/detections/stream"
RECONNECT_DELAY = 2.0
CONVEYOR_TOPIC = "/conveyor/speed"
CONVEYOR_FALLBACK_MPS = 0.083

PUBLISH_HZ = 10.0


class CameraDebugNode(Node):
    def __init__(self) -> None:
        super().__init__("camera_debug")

        self._belt_mps = CONVEYOR_FALLBACK_MPS   # until first /conveyor/speed
        self.create_subscription(
            Float64, CONVEYOR_TOPIC, self._on_belt, 1
        )

        self._snap_lock = threading.Lock()
        self._snap: dict | None = None

        self._pub = self.create_publisher(
            String, "/camera_debug/detections", 10
        )
        self.create_timer(1.0 / PUBLISH_HZ, self._publish_and_render)

        self._stream_url = os.environ.get(
            "GP8_PERCEPTION_URL", PERCEPTION_URL_DEFAULT
        )
        self._stream_thread = threading.Thread(
            target=self._run_stream, name="camera-stream", daemon=True
        )
        self._stream_thread.start()

        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()
        self.get_logger().info(
            f"camera_debug reading {self._stream_url} -> /camera_debug/detections"
        )

    # ------------------------------------------------------------------
    # Stream + corrections
    # ------------------------------------------------------------------
    def _on_belt(self, msg: Float64) -> None:
        self._belt_mps = float(msg.data)

    def _run_stream(self) -> None:
        try:
            stream_detections(
                self._stream_url, self._on_record,
                reconnect_delay=RECONNECT_DELAY,
            )
        except Exception as e:
            self.get_logger().error(f"perception stream thread crashed: {e}")

    def _on_record(self, record: dict) -> None:
        """One incoming NDJSON record from the camera PC → corrected snapshot."""
        positions = record.get("positions") or []
        class_names = record.get("class_names") or []
        confidences = record.get("confidences") or [0.0] * len(positions)
        delay_s = float(record.get("elapsed_s") or 0.0)
        stream_ts = float(record.get("timestamp") or 0.0)

        v = float(self._belt_mps)
        receipt = time.time()

        offset_aim = extrinsics.DETECTION_OFFSET_AIM
        offset_grasp = extrinsics.DETECTION_OFFSET_GRASP
        ws_x_abs = extrinsics.WORKSPACE_X_ABS
        ref_x = extrinsics.REFERENCE_X_BASE
        ref_y = extrinsics.REFERENCE_Y_BASE
        ref_z = extrinsics.REFERENCE_Z_BASE
        sx = extrinsics.SIGN_CX_TO_BASE_X * extrinsics.SCALE_CX_TO_BASE_X
        sy = extrinsics.SIGN_CY_TO_BASE_Y * extrinsics.SCALE_CY_TO_BASE_Y

        detections = []
        for pos, cls, conf in zip(positions, class_names, confidences):
            # New format: cx = across-belt offset (m), cy = along-belt offset
            # with + upstream (toward camera). Depth no longer sent — pick
            # height comes from GRASP_Z downstream.
            cx = float(pos[0]) if len(pos) > 0 else 0.0
            cy = float(pos[1]) if len(pos) > 1 else 0.0
            cz = float(pos[2]) if len(pos) > 2 else 0.0   # for display only
            # Workspace = on the belt centerline laterally. Across-belt is now
            # already in belt-frame meters so the same ±0.2 m filter applies.
            in_ws = (-ws_x_abs < cx < ws_x_abs)

            # Camera (belt-frame, image-centre origin) → robot base: simple
            # constant translation. No more 4×4 transform; the image-centre
            # on the belt is at REFERENCE_(X|Y|Z)_BASE.
            x_base = ref_x + sx * cx
            y_base = ref_y + sy * cy
            z_base = ref_z

            # Apply Z offsets for aim / grasp, back-project Y by v*delay so the
            # position is "where the object is at receipt time."
            base_grasp = np.array([x_base, y_base, z_base + offset_grasp])
            base_grasp[1] -= v * delay_s
            base_aim = np.array([x_base, y_base, z_base + offset_aim])
            base_aim[1] -= v * delay_s

            detections.append({
                "class": cls,
                "confidence": float(conf),
                "cam": [cx, cy, cz],
                "base_grasp": [float(base_grasp[0]),
                               float(base_grasp[1]),
                               float(base_grasp[2])],
                "base_aim":   [float(base_aim[0]),
                               float(base_aim[1]),
                               float(base_aim[2])],
                "in_workspace": bool(in_ws),
            })

        snap = {
            "receipt_time": receipt,
            "stream_record_ts": stream_ts,
            "belt_mps": v,
            "perception_delay_s": delay_s,
            "detections": detections,
        }
        with self._snap_lock:
            self._snap = snap

    # ------------------------------------------------------------------
    # Publish + render
    # ------------------------------------------------------------------
    def _publish_and_render(self) -> None:
        with self._snap_lock:
            snap = dict(self._snap) if self._snap is not None else None
        if snap is None:
            self._render_empty()
            return
        snap["publish_ts"] = time.time()
        try:
            self._pub.publish(String(data=json.dumps(snap)))
        except Exception as e:
            self.get_logger().warn(f"publish failed: {e}")
        self._render(snap)

    def _render_empty(self) -> None:
        out = ["\033[H\033[J"]
        out.append("=== GP8 Camera Debug — live ===\n")
        out.append(f" (waiting for stream records from {self._stream_url})\n")
        sys.stdout.write("".join(out))
        sys.stdout.flush()

    def _render(self, snap: dict) -> None:
        v = float(snap.get("belt_mps", 0.0))
        receipt = float(snap.get("receipt_time", time.time()))
        dt = max(0.0, time.time() - receipt)
        delay = float(snap.get("perception_delay_s", 0.0))
        detections = snap.get("detections", [])

        out = ["\033[H\033[J"]
        out.append("=== GP8 Camera Debug — live ===\n")
        out.append(
            f" belt: {v:6.3f} m/s   "
            f"stream delay: {delay:.3f}s   "
            f"snapshot age: {dt:.2f}s   "
            f"detections: {len(detections)}\n\n"
        )
        if not detections:
            out.append(" (no objects in latest record)\n")
        else:
            out.append(
                " class         conf   cam=[cx,cy,cz]                "
                "base_grasp=[x,y,z]               ws\n"
            )
            for d in detections:
                cam = d.get("cam", [0.0, 0.0, 0.0])
                bg = d.get("base_grasp", [0.0, 0.0, 0.0])
                # Extrapolate Y forward for a live feel between records.
                bg_y_live = float(bg[1]) - v * dt
                ws = "IN " if d.get("in_workspace") else "OUT"
                origin = (abs(cam[0]) < 1e-4 and abs(cam[1]) < 1e-4
                          and abs(cam[2]) < 1e-4)
                warn = " ⚠ camera sent origin" if origin else ""
                out.append(
                    f" {str(d.get('class',''))[:12]:<12s}  "
                    f"{d.get('confidence', 0.0):.2f}  "
                    f"[{cam[0]:+.3f},{cam[1]:+.3f},{cam[2]:+.3f}]   "
                    f"[{float(bg[0]):+.3f},{bg_y_live:+.3f},"
                    f"{float(bg[2]):+.3f}]  "
                    f"{ws}{warn}\n"
                )
        sys.stdout.write("".join(out))
        sys.stdout.flush()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CameraDebugNode()
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

#!/usr/bin/env python3
"""Camera debug node — owns the perception → base-frame correction pipeline.

Reads full four-corner boxes from the camera's HTTP NDJSON detection stream,
applies the camera→base transform to every corner, back-projects each Y by
``belt_speed × perception_delay``, and publishes the corrected detection list
on ``/camera_debug/detections``.
app.py subscribes to that topic instead of re-doing the corrections itself.

A live TUI renders the raw camera positions next to the corrected base
positions so it's obvious when the camera sends garbage (e.g., [0,0,0]
which maps to the constant base x=0.425).

Run alongside the bringup in its own terminal:

    ros2 run gp8_control camera_debug
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, String

from gp8_control.perception import extrinsics
from gp8_control.perception.bbox_geometry import as_bbox, bbox_center, bbox_to_base
from gp8_control.perception.clock_sync import (
    ContinuousClockSync,
    latency_url_from_stream,
)
from gp8_control.perception.latency import live_capture_age, select_capture_age
from gp8_control.perception.perception_client import stream_detections


# Perception stream URL — same env-var override as gp8_control.app.Config.
PERCEPTION_URL_DEFAULT = "http://147.46.175.15:8080/detections/stream"
RECONNECT_DELAY = 2.0
CONVEYOR_TOPIC = "/conveyor/speed"
CONVEYOR_FALLBACK_MPS = 0.083

PUBLISH_HZ = 10.0

# --- Perception latency compensation (moving-object downstream bias) ---
# Normal path: continuously synchronize the perception-PC clock through
# /latency, convert each RealSense capture timestamp to the robot timeline, and
# directly measure capture -> receipt age. This automatically includes camera,
# inference, serialization, stream, network, and client parsing for that frame.
#
# Fallback path: if cross-host time or the producer timestamp is unavailable,
# use elapsed_s + estimated frame age + the optional fixed residual below. The
# frame age is estimated as FRAME_AGE_FACTOR × an EMA of record arrival periods.
CAMERA_FPS_FALLBACK = float(os.environ.get("GP8_CAMERA_FPS", "30.0"))
FPS_EMA_ALPHA = float(os.environ.get("GP8_FPS_EMA_ALPHA", "0.2"))     # EMA weight on the newest interval
# Ignore inter-record gaps longer than this (reconnect/stall) so they don't
# poison the frame-period EMA. Raise it if your camera runs slower than ~1 fps.
MAX_FRAME_GAP_S = float(os.environ.get("GP8_MAX_FRAME_GAP_S", "1.0"))
# Fraction of a frame period to count as acquisition age: 1.0 = a full period
# (frame + buffer), 0.5 = mean age of a uniformly-sampled frame.
FRAME_AGE_FACTOR = float(os.environ.get("GP8_FRAME_AGE_FACTOR", "1.0"))
# Residual fixed latency (transport + anything not in elapsed_s), seconds.
# Used only while live cross-host clock synchronization is unavailable.
PERCEPTION_EXTRA_LATENCY_S = float(os.environ.get("GP8_PERCEPTION_LATENCY_S", "0.0"))
CLOCK_SYNC_INTERVAL_S = float(os.environ.get("GP8_CLOCK_SYNC_INTERVAL_S", "5.0"))
CLOCK_SYNC_PROBES = int(os.environ.get("GP8_CLOCK_SYNC_PROBES", "8"))
CLOCK_SYNC_TIMEOUT_S = float(os.environ.get("GP8_CLOCK_SYNC_TIMEOUT_S", "2.0"))
CLOCK_SYNC_MAX_AGE_S = float(os.environ.get("GP8_CLOCK_SYNC_MAX_AGE_S", "15.0"))
MAX_LIVE_CAPTURE_AGE_S = float(os.environ.get("GP8_MAX_CAPTURE_AGE_S", "2.0"))


def _finite_float(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


class CameraDebugNode(Node):
    def __init__(self) -> None:
        super().__init__("camera_debug")

        self._belt_mps = CONVEYOR_FALLBACK_MPS   # until first /conveyor/speed
        self.create_subscription(
            Float64, CONVEYOR_TOPIC, self._on_belt, 1
        )

        self._snap_lock = threading.Lock()
        self._snap: dict | None = None

        self._stream_url = os.environ.get(
            "GP8_PERCEPTION_URL", PERCEPTION_URL_DEFAULT
        )
        self._clock_sync = ContinuousClockSync(
            latency_url_from_stream(self._stream_url),
            interval_s=CLOCK_SYNC_INTERVAL_S,
            probes=CLOCK_SYNC_PROBES,
            timeout_s=CLOCK_SYNC_TIMEOUT_S,
            max_age_s=CLOCK_SYNC_MAX_AGE_S,
        )
        self._clock_sync.start()

        # Live EMA of the inter-record interval -> estimated frame period (real FPS),
        # used for the frame-acquisition-age latency term (see _on_record).
        self._last_record_t: float | None = None
        self._frame_period_ema: float | None = None

        self._pub = self.create_publisher(
            String, "/camera_debug/detections", 10
        )
        self.create_timer(1.0 / PUBLISH_HZ, self._publish_and_render)

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
                skip_initial_record=True,
            )
        except Exception as e:
            self.get_logger().error(f"perception stream thread crashed: {e}")

    def _on_record(self, record: dict) -> None:
        """One incoming NDJSON record from the camera PC → corrected snapshot."""
        bounding_boxes = record.get("bounding_boxes") or []
        class_names = record.get("class_names") or []
        confidences = record.get("confidences") or [0.0] * len(bounding_boxes)
        delay_s = float(record.get("elapsed_s") or 0.0)
        stream_ts = float(record.get("timestamp") or 0.0)

        v = float(self._belt_mps)
        receipt = time.time()
        # EMA the inter-record arrival interval -> estimated frame period (the
        # camera's real effective FPS). Gaps > MAX_FRAME_GAP_S (reconnect/stall)
        # are skipped so they don't poison the estimate.
        if self._last_record_t is not None:
            dt = receipt - self._last_record_t
            if 0.0 < dt < MAX_FRAME_GAP_S:
                if self._frame_period_ema is None:
                    self._frame_period_ema = dt
                else:
                    self._frame_period_ema = (
                        FPS_EMA_ALPHA * dt
                        + (1.0 - FPS_EMA_ALPHA) * self._frame_period_ema
                    )
        self._last_record_t = receipt
        frame_period = (
            self._frame_period_ema
            if self._frame_period_ema is not None
            else 1.0 / max(CAMERA_FPS_FALLBACK, 1e-6)
        )
        # Prefer the producer's per-frame RealSense global timestamp. Older
        # producers (or a camera without usable global time) fall back to the
        # previous inter-arrival-period estimate.
        estimated_frame_age = FRAME_AGE_FACTOR * frame_period
        frame_age, frame_age_source = select_capture_age(
            record.get("capture_age_s"), estimated_frame_age
        )
        clock_estimate = self._clock_sync.estimate()
        capture_timestamp = _finite_float(record.get("capture_timestamp"))
        measured_total_age = None
        if clock_estimate is not None and capture_timestamp is not None:
            measured_total_age = live_capture_age(
                receipt,
                capture_timestamp,
                clock_estimate.offset_s,
                MAX_LIVE_CAPTURE_AGE_S,
            )
            if measured_total_age is None:
                # The stream sends its cached record once on reconnect. Never
                # turn an old cached detection into a fresh one via fallback.
                self.get_logger().warn(
                    "dropping stale/invalid timestamped perception record"
                )
                return

        if measured_total_age is not None:
            # Direct capture -> callback age. This already contains camera/USB,
            # inference, server queue/serialization, network, and JSON parsing.
            # Do not add elapsed_s or the fixed residual again.
            total_delay = measured_total_age
            latency_mode = "live_capture_to_receipt"
            extra_latency = total_delay - delay_s
            fixed_fallback_applied = 0.0
        else:
            # Legacy producer, stale clock estimate, or invalid global timestamp.
            # Preserve the previous estimate so perception does not stop working.
            extra_latency = frame_age + PERCEPTION_EXTRA_LATENCY_S
            total_delay = delay_s + extra_latency
            latency_mode = "fallback_components"
            fixed_fallback_applied = PERCEPTION_EXTRA_LATENCY_S

        server_send_ts = _finite_float(record.get("server_send_timestamp"))
        post_inference_s = None
        network_receive_s = None
        if server_send_ts is not None:
            inference_end_ts = stream_ts + delay_s
            candidate = server_send_ts - inference_end_ts
            if 0.0 <= candidate <= MAX_LIVE_CAPTURE_AGE_S:
                post_inference_s = candidate
            if clock_estimate is not None:
                candidate = receipt - (
                    server_send_ts - clock_estimate.offset_s
                )
                if 0.0 <= candidate <= MAX_LIVE_CAPTURE_AGE_S:
                    network_receive_s = candidate

        offset_aim = extrinsics.DETECTION_OFFSET_AIM
        offset_grasp = extrinsics.DETECTION_OFFSET_GRASP
        ws_x_abs = extrinsics.WORKSPACE_X_ABS
        ref_x = extrinsics.REFERENCE_X_BASE
        ref_y = extrinsics.REFERENCE_Y_BASE
        ref_z = extrinsics.REFERENCE_Z_BASE
        sx = extrinsics.SIGN_CX_TO_BASE_X * extrinsics.SCALE_CX_TO_BASE_X
        sy = extrinsics.SIGN_CY_TO_BASE_Y * extrinsics.SCALE_CY_TO_BASE_Y

        detections = []
        for box, cls, conf in zip(bounding_boxes, class_names, confidences):
            try:
                cam_bbox = as_bbox(box)
            except (TypeError, ValueError) as exc:
                self.get_logger().warn(
                    f"ignoring malformed perception bounding box: {exc}"
                )
                continue

            # The producer sends all four projected corners.  Preserve them
            # through the base-frame conversion, and derive the grasp centre
            # from the complete box so the robot keeps its existing behaviour.
            cx, cy, cz = bbox_center(cam_bbox)
            # Workspace = on the belt centerline laterally. Across-belt is now
            # already in belt-frame meters so the same ±0.2 m filter applies.
            in_ws = (-ws_x_abs < cx < ws_x_abs)

            # Camera (belt-frame, image-centre origin) → robot base: simple
            # constant translation. No more 4×4 transform; the image-centre
            # on the belt is at REFERENCE_(X|Y|Z)_BASE.
            x_base = ref_x + sx * cx
            y_base = ref_y + sy * cy
            z_base = ref_z

            bbox_transform = dict(
                ref_x=ref_x,
                ref_y=ref_y,
                ref_z=ref_z,
                scale_x=sx,
                scale_y=sy,
                y_back_projection=v * total_delay,
            )
            base_bbox_grasp = bbox_to_base(
                cam_bbox, z_offset=offset_grasp, **bbox_transform
            )
            base_bbox_aim = bbox_to_base(
                cam_bbox, z_offset=offset_aim, **bbox_transform
            )

            # Apply Z offsets for aim / grasp, back-project Y by v*delay so the
            # position is "where the object is at receipt time."
            # Back-project Y by v*total_delay (elapsed_s + frame-age + transport)
            # so the position is "where the object is at receipt time," removing the
            # moving-object upstream bias.
            base_grasp = np.array([x_base, y_base, z_base + offset_grasp])
            base_grasp[1] -= v * total_delay
            base_aim = np.array([x_base, y_base, z_base + offset_aim])
            base_aim[1] -= v * total_delay

            detections.append({
                "class": cls,
                "confidence": float(conf),
                "cam": [float(cx), float(cy), float(cz)],
                "cam_bbox": cam_bbox.tolist(),
                "base_grasp": [float(base_grasp[0]),
                               float(base_grasp[1]),
                               float(base_grasp[2])],
                "base_aim":   [float(base_aim[0]),
                               float(base_aim[1]),
                               float(base_aim[2])],
                "base_bbox_grasp": base_bbox_grasp.tolist(),
                "base_bbox_aim": base_bbox_aim.tolist(),
                "in_workspace": bool(in_ws),
            })

        snap = {
            "receipt_time": receipt,
            "stream_record_ts": stream_ts,
            "belt_mps": v,
            "perception_delay_s": delay_s,
            "frame_age_s": frame_age,
            "frame_age_source": frame_age_source,
            "latency_mode": latency_mode,
            "reported_capture_timestamp": record.get("capture_timestamp"),
            "reported_capture_timestamp_domain": record.get(
                "capture_timestamp_domain"
            ),
            "frame_period_s": frame_period,
            "est_fps": (1.0 / frame_period) if frame_period > 0.0 else None,
            "extra_latency_s": extra_latency,
            "applied_delay_s": total_delay,
            "fixed_fallback_latency_s": fixed_fallback_applied,
            "server_post_inference_s": post_inference_s,
            "network_receive_s": network_receive_s,
            "server_send_timestamp": server_send_ts,
            "clock_offset_s": (
                clock_estimate.offset_s if clock_estimate is not None else None
            ),
            "clock_min_rtt_s": (
                clock_estimate.min_rtt_s if clock_estimate is not None else None
            ),
            "clock_sync_error": self._clock_sync.last_error(),
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
            f"detections: {len(detections)}\n"
        )
        applied = float(snap.get("applied_delay_s", delay))
        frame_age = float(snap.get("frame_age_s", 0.0))
        frame_age_source = str(snap.get("frame_age_source", "unknown"))
        latency_mode = str(snap.get("latency_mode", "unknown"))
        est_fps = snap.get("est_fps", None)
        fps_str = f"{est_fps:.1f}" if est_fps is not None else "…"
        offset = snap.get("clock_offset_s")
        rtt = snap.get("clock_min_rtt_s")
        clock_str = (
            f"offset {float(offset) * 1000:+.2f}ms, "
            f"minRTT {float(rtt) * 1000:.2f}ms"
            if offset is not None and rtt is not None
            else "clock sync unavailable"
        )
        if latency_mode == "live_capture_to_receipt":
            post = snap.get("server_post_inference_s")
            network = snap.get("network_receive_s")
            post_str = f"{float(post):.3f}" if post is not None else "n/a"
            network_str = (
                f"{float(network):.3f}" if network is not None else "n/a"
            )
            out.append(
                f" [latency LIVE] frame→infer {frame_age:.3f} "
                f"({frame_age_source}) + infer {delay:.3f} + "
                f"server {post_str} + wire/client {network_str} "
                f"= applied {applied:.3f}s "
                f"(→ {applied * v * 100:+.1f} cm back-proj)\n"
                f"                {clock_str}\n\n"
            )
        else:
            out.append(
                f" [latency FALLBACK] elapsed {delay:.3f} + frame_age "
                f"{frame_age:.3f} ({frame_age_source}, est_fps {fps_str}) + "
                f"fixed {PERCEPTION_EXTRA_LATENCY_S:.3f} = applied "
                f"{applied:.3f}s (→ {applied * v * 100:+.1f} cm back-proj)\n"
                f"                    {clock_str}\n\n"
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
        node._clock_sync.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

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

# --- Vision-vs-encoder belt-speed check (encoder scale audit) -----------------
# Each re-anchor of a track is one (receipt_time, anchor_y) observation of the
# object moving through the camera FOV. The least-squares slope of y(t) over a
# full pass is the belt speed AS THE CAMERA SEES IT. A constant perception
# latency shifts every observation equally and does NOT change the slope, so
#   vision_v / encoder_v  cleanly isolates an encoder SCALE error from a latency
# bias (the two suspects behind the "strikes ~0.3 s late" bias, 2026-07-22).
SPEED_FIT_MIN_POINTS = 5    # need at least this many re-anchors to fit
SPEED_FIT_MIN_SPAN_S = 0.5  # ...spread over at least this much time
SPEED_FIT_EXIT_S = 1.0      # no re-anchor for this long -> object left the FOV


def _linfit_slope(pts: "list[tuple[float, float]]") -> "float | None":
    """Least-squares slope dy/dt of (t, y) points; None if degenerate."""
    n = len(pts)
    t0 = pts[0][0]
    ts = [t - t0 for t, _ in pts]
    ys = [y for _, y in pts]
    st, sy = sum(ts), sum(ys)
    stt = sum(t * t for t in ts)
    sty = sum(t * y for t, y in zip(ts, ys))
    denom = n * stt - st * st
    if denom <= 1e-12:
        return None
    return (n * sty - st * sy) / denom


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
        )
        self._latest: dict | None = None
        self._recv_mono: float = 0.0
        self._events: list[str] = []
        # Speed-fit state: per-track re-anchor observations, encoder samples
        # taken alongside them, finished track ids, and the pass ratios
        # (vision/encoder - 1) accumulated for the aggregate verdict line.
        self._fit_obs: dict[int, list[tuple[float, float]]] = {}
        self._fit_enc: dict[int, list[float]] = {}
        self._fit_done: set[int] = set()
        self._ratios: list[float] = []
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
        self._intake.ingest(self._latest, self._queue, None, v, logger=None)
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

        # Speed fit: a track whose detect_time advanced was re-anchored this
        # tick — record the fresh (receipt_time, anchor_y) pair plus the encoder
        # reading next to it. receipt_time (not viewer arrival) keeps ROS
        # transport jitter off the fit's time axis.
        for obj in self._queue._objects:
            tid = obj.track_id
            if tid in self._fit_done:
                continue
            pts = self._fit_obs.setdefault(tid, [])
            if not pts or obj.detect_time > pts[-1][0] + 1e-6:
                pts.append((obj.detect_time, float(obj.T_grasp_base[1, 3])))
                self._fit_enc.setdefault(tid, []).append(v)

        alive_before_drop = {o.track_id for o in self._queue._objects}
        self._queue.update(now, v)
        alive = {o.track_id for o in self._queue._objects}
        for tid in sorted(alive_before_drop - alive):
            self._event(f"DROP id={tid} (past reach line)")
            self._finalize_speed_fit(tid)

        # A track that hasn't re-anchored for SPEED_FIT_EXIT_S has left the
        # camera FOV — its pass is complete, fit it now (DROP happens much
        # later, at the downstream reach line).
        for obj in self._queue._objects:
            tid = obj.track_id
            pts = self._fit_obs.get(tid)
            if (
                tid not in self._fit_done and pts
                and now - pts[-1][0] > SPEED_FIT_EXIT_S
            ):
                self._finalize_speed_fit(tid)

        self._render(now, v)

    # ------------------------------------------------------------------
    def _finalize_speed_fit(self, tid: int) -> None:
        """Fit one completed FOV pass -> emit a vision-vs-encoder event line."""
        pts = self._fit_obs.pop(tid, None)
        enc = self._fit_enc.pop(tid, [])
        if tid in self._fit_done:
            return
        self._fit_done.add(tid)
        if not pts or len(pts) < SPEED_FIT_MIN_POINTS:
            return   # blip (entered/left frame edge) — not a usable pass
        span = pts[-1][0] - pts[0][0]
        if span < SPEED_FIT_MIN_SPAN_S:
            return
        slope = _linfit_slope(pts)
        if slope is None:
            return
        v_vision = -slope   # belt advances -Y, so a real pass has slope < 0
        v_enc = sum(enc) / len(enc) if enc else 0.0
        if v_vision <= 0.02 or v_enc <= 0.02:
            self._event(
                f"SPEED id={tid} unusable (vision {v_vision:+.3f} / "
                f"enc {v_enc:.3f} m/s, n={len(pts)})"
            )
            return
        ratio = v_vision / v_enc - 1.0
        self._ratios.append(ratio)
        enc_note = (
            "" if self._conveyor._last_msg_time is not None else "  ENC=FALLBACK!"
        )
        self._event(
            f"SPEED id={tid} vision {v_vision:.4f} vs enc {v_enc:.4f} m/s "
            f"-> {ratio:+.1%} (n={len(pts)}, {span:.1f}s){enc_note}"
        )

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
        if self._ratios:
            n = len(self._ratios)
            mean = sum(self._ratios) / n
            var = sum((r - mean) ** 2 for r in self._ratios) / n
            out.append(
                f" scale check: vision/enc {mean:+.1%} ± {var ** 0.5:.1%} "
                f"({n} pass{'es' if n != 1 else ''})"
                f"   [≈0% → encoder OK, bias is latency]\n\n"
            )
        else:
            out.append(
                " scale check: (no completed passes yet — roll an object "
                "through the camera view)\n\n"
            )
        if not self._queue:
            out.append(" (no tracked objects)\n")
        else:
            out.append(
                "  id  class         conf      x        y_now      z     age\n"
            )
            for obj in self._queue._objects:
                y_now = float(
                    obj.T_grasp_base[1, 3] - v * (now - obj.detect_time)
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

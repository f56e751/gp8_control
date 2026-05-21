"""HTTP NDJSON detection stream → SAMClient-compatible read surface.

Bridges the push-based ``perception_client.stream_detections`` (a blocking
forever-loop) to the pull-based interface ``DetectionIntake`` already expects
from the old TCP ``SAMClient``: ``.positions``, ``.class_names``, ``.delay``.

A background daemon thread runs the stream and overwrites a single
last-writer-wins snapshot under a lock; the main loop reads the latest
snapshot each poll. This keeps ``DetectionIntake``'s camera→grasp transform,
workspace filter, and conveyor-delay back-projection untouched.

The robot PC thus consumes perception purely through the documented wire
contract (see ``perception_client.py``) — no local camera / GPU / ROS image
subscriptions.
"""

from __future__ import annotations

import threading
import time

from .perception_client import stream_detections


class StreamDetectionSource:
    """Thread-backed adapter exposing the SAMClient read surface.

    Attributes mirrored from SAMClient so ``DetectionIntake`` is unchanged:
      - ``positions``   : latest ``[[X, Y, Z], ...]`` (camera frame, metres) or None
      - ``class_names`` : latest ``["metal", ...]`` or None
      - ``delay``       : perception latency (s) for conveyor back-projection
    """

    def __init__(self, url: str, reconnect_delay: float = 2.0, logger=None) -> None:
        self._url = url
        self._reconnect_delay = reconnect_delay
        self._logger = logger

        self._lock = threading.Lock()
        self._positions: list | None = None
        self._class_names: list | None = None
        self._elapsed_s: float = 0.0
        self._received_monotonic: float | None = None

        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Spawn the daemon thread streaming detections in the background."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="perception-stream", daemon=True
        )
        self._thread.start()
        if self._logger is not None:
            self._logger.info(f"Perception stream client started: {self._url}")

    def _run(self) -> None:
        # stream_detections loops forever, reconnecting on network drops.
        # Callback exceptions propagate; _on_record is kept trivial so robot
        # logic bugs surface in the main loop, not silently in this thread.
        stream_detections(
            self._url, self._on_record, reconnect_delay=self._reconnect_delay
        )

    def _on_record(self, record: dict) -> None:
        positions = record.get("positions")
        class_names = record.get("class_names")
        elapsed = record.get("elapsed_s")
        with self._lock:
            self._positions = positions
            self._class_names = class_names
            self._elapsed_s = float(elapsed) if elapsed is not None else 0.0
            self._received_monotonic = time.monotonic()

    # ------------------------------------------------------------------
    # SAMClient-compatible read surface
    # ------------------------------------------------------------------
    @property
    def positions(self) -> list | None:
        with self._lock:
            return self._positions

    @positions.setter
    def positions(self, _value) -> None:
        # No-op: DetectionIntake.poll() resets this on the old SAMClient to
        # drop stale frames. The stream is last-writer-wins, so we keep the
        # latest snapshot and let FrameGate handle de-duplication.
        pass

    @property
    def class_names(self) -> list | None:
        with self._lock:
            return self._class_names

    @class_names.setter
    def class_names(self, _value) -> None:
        pass

    @property
    def delay(self) -> float | None:
        """Perception latency (s), clock-independent.

        ``elapsed_s`` (camera-side inference time) + local snapshot staleness
        measured on the robot's monotonic clock. Both terms avoid any
        dependence on cross-machine wall-clock sync (camera PC ↔ robot PC).
        Returns None before the first record (mirrors SAMClient).
        """
        with self._lock:
            if self._received_monotonic is None:
                return None
            return self._elapsed_s + (time.monotonic() - self._received_monotonic)

"""HardwareWorldSource — the live-camera + encoder implementation of WorldSource.

Repackages exactly what ``GP8App.setup()`` used to wire inline (zero behavior
change): the ``/camera_debug/detections`` JSON-String subscription (the
``camera_debug`` node owns the HTTP stream, cam→base transform, Z offsets and
v*delay back-projection — we only store its latest corrected snapshot) and the
existing :class:`ConveyorSpeedTracker` (``/conveyor/speed`` +
``/conveyor/distance_mm`` encoder telemetry) exposed as ``.belt``.
"""

from __future__ import annotations

import json
from typing import Optional

from std_msgs.msg import String

from gp8_control.backends.world_base import WorldSource
from gp8_control.conveyor.conveyor_speed import ConveyorSpeedTracker


class HardwareWorldSource(WorldSource):
    def __init__(self, node, cfg) -> None:
        self._snap: dict | None = None
        # camera_debug node owns the perception stream + corrections; we just
        # subscribe to its corrected detection list.
        node.create_subscription(
            String, "/camera_debug/detections",
            self._on_camera_debug_detections, 10,
        )
        # Encoder telemetry is the single source of truth for belt speed.
        self._belt = ConveyorSpeedTracker(
            node,
            cfg.CONVEYOR_TOPIC,
            cfg.CONVEYOR_SPEED,
            cfg.CONVEYOR_STALE_SECONDS,
            distance_topic=cfg.CONVEYOR_DISTANCE_TOPIC,
        )

    def _on_camera_debug_detections(self, msg: String) -> None:
        try:
            self._snap = json.loads(msg.data)
        except (ValueError, TypeError):
            pass

    def latest_snapshot(self) -> Optional[dict]:
        return self._snap

    @property
    def belt(self) -> ConveyorSpeedTracker:
        return self._belt

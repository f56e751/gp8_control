"""Detection intake: fold camera_debug's corrected detections into the queue.

The ``camera_debug`` node (separate process) owns the sensor→base transform,
Z offsets, ``v×delay`` back-projection, and workspace filter, and publishes
corrected base-frame detections on ``/camera_debug/detections``. This module is
the boundary stage on the CONTROL side: it takes one such snapshot and updates
the ``TrackedObjectQueue``, doing the spatial dedup / re-anchoring the detector
can't (it emits no per-object identity, so each frame re-detects every visible
object).

It lives here — in the control process, NOT in the ``camera_debug`` node —
because the dedup needs robot-side tracking state (the queue and the in-flight
active target), which only exists in the ``gp8_manager`` process.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

import numpy as np

from gp8_control.tracking import TrackedObject

if TYPE_CHECKING:
    from gp8_control.tracking import TrackedObjectQueue


def _make_transform(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float).ravel()
    return T


# Tool orientation: tool pointing down at the belt. Must match the value the
# camera_debug node assumes when it reports base_grasp / base_aim.
_R_GRASP_DEFAULT = np.array(
    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
)


class DetectionIntake:
    """Folds corrected ``camera_debug`` detections into the tracked-object queue.

    Spatial dedup: each camera frame re-detects every visible object (no
    per-object identity), so a new detection within ``eps`` of an existing
    track's conveyor-compensated position is treated as the SAME object and
    RE-ANCHORS that track (resets its belt-extrapolation reference) instead of
    enqueuing a duplicate. Only a genuinely-new position becomes a new object.
    Class-independent: once a spot has an object, any re-detection there is that
    same one.
    """

    def __init__(self, eps: float) -> None:
        #: spatial match radius [m] — OBJECT_MATCH_EPSILON on the app Config.
        self.eps = eps

    def ingest(
        self,
        snapshot: "Optional[dict]",
        queue: "TrackedObjectQueue",
        active_target: "Optional[TrackedObject]",
        v: float,
        logger=None,
    ) -> int:
        """Update ``queue`` from one ``/camera_debug/detections`` snapshot.

        ``snapshot`` is the parsed message dict (``None`` before the first
        message). ``active_target`` is the object currently being manipulated
        (popped from the queue but still on the belt) — included in the dedup so
        an in-flight pick isn't re-enqueued. ``v`` is the current belt speed.

        Mutates ``queue`` (adds new objects / re-anchors existing ones) and
        returns the count of genuinely-new objects added (0 if none). The caller
        owns any frame-gate bookkeeping keyed on that count.
        """
        if snapshot is None:
            return 0
        detections = [
            d for d in snapshot.get("detections", []) if d.get("in_workspace")
        ]
        if not detections:
            return 0

        # camera_debug already applied the camera→base transform, Z offsets, and
        # v*delay back-projection. ``receipt_time`` is the moment for which the
        # corrected positions are valid; the queue extrapolates forward from there.
        detect_time = float(snapshot.get("receipt_time", time.time()))

        # Project every existing tracked object (active target + queue) forward to
        # ``detect_time``; a detection within ``eps`` of one is the SAME object.
        existing: list[TrackedObject] = []
        if active_target is not None:
            existing.append(active_target)
        existing.extend(queue._objects)
        eps = self.eps

        def _matches(obj: TrackedObject, det_x: float, det_y: float) -> bool:
            ox = float(obj.T_grasp_base[0, 3])
            oy = float(obj.T_grasp_base[1, 3] - v * (detect_time - obj.detect_time))
            return abs(ox - det_x) < eps and abs(oy - det_y) < eps

        added = 0
        refreshed = 0
        for d in detections:
            base_aim = d.get("base_aim", [0.0, 0.0, 0.0])
            base_grasp = d.get("base_grasp", [0.0, 0.0, 0.0])
            det_x = float(base_grasp[0])
            det_y = float(base_grasp[1])
            match = next((o for o in existing if _matches(o, det_x, det_y)), None)
            if match is not None:
                # Re-anchor the existing track to this fresh detection instead of
                # adding a duplicate. Resetting the extrapolation reference
                # (detect_time + pose) every frame keeps drift below ``eps`` so a
                # 2nd "object" never spawns at the same spot. Class kept as-is.
                match.T_aim_base = _make_transform(_R_GRASP_DEFAULT, base_aim)
                match.T_grasp_base = _make_transform(_R_GRASP_DEFAULT, base_grasp)
                match.detect_time = detect_time
                match.cam_pos = tuple(d.get("cam", [0.0, 0.0, 0.0]))
                refreshed += 1
                continue
            new_obj = TrackedObject(
                T_aim_base=_make_transform(_R_GRASP_DEFAULT, base_aim),
                T_grasp_base=_make_transform(_R_GRASP_DEFAULT, base_grasp),
                class_name=d.get("class", "?"),
                detect_time=detect_time,
                cam_pos=tuple(d.get("cam", [0.0, 0.0, 0.0])),
            )
            queue.add(new_obj)
            existing.append(new_obj)  # dedupe within the same intake too
            added += 1

        if added > 0 and logger is not None:
            logger.info(
                f"New frame — {added} new object(s) added, {refreshed} re-anchored "
                f"(queue size: {len(queue._objects)}, belt {v:.3f} m/s)"
            )
        return added

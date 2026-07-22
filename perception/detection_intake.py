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


def _fmt_votes(votes: dict) -> str:
    """Compact 'cls:weight' dump (highest first) for the reclass log."""
    return "{" + ", ".join(
        f"{k}:{v:.2f}" for k, v in sorted(votes.items(), key=lambda kv: -kv[1])
    ) + "}"


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

    def __init__(self, eps: float, drift_frac: float = 0.25,
                 eps_y_max: float = 0.20, merge_eps_y_max: float = 0.10,
                 vel_window_s: float = 1.5, vel_min_anchors: int = 4,
                 vel_min_span_s: float = 0.3, vel_max_rms: float = 0.02,
                 vel_clamp_frac: float = 0.25) -> None:
        #: spatial match radius [m] — OBJECT_MATCH_EPSILON on the app Config.
        self.eps = eps
        #: belt-direction tolerance growth per second of dead reckoning, as a
        #: fraction of belt speed (Config.OBJECT_MATCH_DRIFT_FRAC).
        self.drift_frac = drift_frac
        #: cap on the grown Y window [m] (Config.OBJECT_MATCH_EPS_Y_MAX).
        self.eps_y_max = eps_y_max
        #: tighter cap for the merge pass (Config.OBJECT_MERGE_EPS_Y_MAX).
        self.merge_eps_y_max = merge_eps_y_max
        #: per-object velocity-fit policy (Config.OBJECT_VEL_*).
        self.vel_window_s = vel_window_s
        self.vel_min_anchors = vel_min_anchors
        self.vel_min_span_s = vel_min_span_s
        self.vel_max_rms = vel_max_rms
        self.vel_clamp_frac = vel_clamp_frac
        #: track_ids already logged as speed-deviating (one line per object).
        self._vel_logged: set = set()

    def _update_velocity(self, obj, t: float, y: float, v_belt: float,
                         logger=None) -> None:
        """Append one (t, y) anchor and refit ``obj.v_est`` if the fit is trusted.

        Belt travels -Y, so the anchor slope dy/dt is -v; v_est = -slope. The fit
        is accepted only when there are enough anchors spanning enough time with a
        small residual, AND the result sits within ``belt*(1 +/- clamp_frac)`` —
        a fit outside that band is far more likely a mis-association or a bad
        detection than a real >clamp speed, so it is rejected and the previous
        v_est (or the global belt fallback) stands. Frozen when re-detections stop.
        """
        obj.y_anchors.append((float(t), float(y)))
        cutoff = t - self.vel_window_s
        obj.y_anchors = [(ta, ya) for (ta, ya) in obj.y_anchors if ta >= cutoff]
        n = len(obj.y_anchors)
        if n < self.vel_min_anchors:
            return
        ts = np.array([a[0] for a in obj.y_anchors], dtype=float)
        ys = np.array([a[1] for a in obj.y_anchors], dtype=float)
        if ts[-1] - ts[0] < self.vel_min_span_s:
            return
        # LSQ line fit y = slope*(t - t0) + b (t0-shift keeps the matrix well
        # conditioned; wall-clock t values are huge).
        A = np.vstack([ts - ts[0], np.ones(n)]).T
        coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
        v_fit = -float(coef[0])
        rms = float(np.sqrt(np.mean((ys - A @ coef) ** 2)))
        if rms > self.vel_max_rms:
            return                            # noisy/inconsistent — keep fallback
        if v_belt > 0.02:                     # clamp to belt band (skip if belt ~0)
            lo, hi = v_belt * (1 - self.vel_clamp_frac), v_belt * (1 + self.vel_clamp_frac)
            if not (lo <= v_fit <= hi):
                return                        # out of band — reject as mis-association
        obj.v_est = v_fit
        # One diagnostic line per object when its speed deviates notably from the
        # belt — this is the signal to confirm the rolling-can hypothesis on HW.
        if (logger is not None and v_belt > 0.02
                and abs(v_fit - v_belt) / v_belt > 0.10
                and obj.track_id not in self._vel_logged):
            self._vel_logged.add(obj.track_id)
            logger.info(
                f"[track-VEL] id={obj.track_id} {obj.class_name}: v_est "
                f"{v_fit:.3f} m/s vs belt {v_belt:.3f} "
                f"({100 * (v_fit - v_belt) / v_belt:+.0f}%, {n} anchors)"
            )

    def _grown_eps_y(self, age: float, v: float, cap: float) -> float:
        """Belt-direction window for a track last seen ``age`` s ago.

        Grows with the dead-reckoned distance because the belt-speed estimate
        carries a few-percent error: a track un-refreshed for seconds (the main
        loop is blocked while a pick/throw trajectory streams) is predicted at
        v*age downstream with ~drift_frac*v*age of uncertainty. A fixed window
        makes that track fail to match its own re-detection → duplicate spawn.
        """
        return min(self.eps + self.drift_frac * abs(v) * max(age, 0.0), cap)

    def _eps_y(self, age: float, v: float) -> float:
        """Window for matching a fresh DETECTION to a track (generous — the
        detection is ground truth)."""
        return self._grown_eps_y(age, v, self.eps_y_max)

    def _eps_y_merge(self, age: float, v: float) -> float:
        """Window for merging two TRACKS (tight — both sides are predictions
        with no fresh evidence, so a wide window would delete a real object)."""
        return self._grown_eps_y(age, v, self.merge_eps_y_max)

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
            age = detect_time - obj.detect_time
            ox = float(obj.T_grasp_base[0, 3])
            oy = float(obj.T_grasp_base[1, 3] - v * age)
            return (abs(ox - det_x) < eps
                    and abs(oy - det_y) < self._eps_y(age, v))

        added = 0
        refreshed = 0
        for d in detections:
            base_aim = d.get("base_aim", [0.0, 0.0, 0.0])
            base_grasp = d.get("base_grasp", [0.0, 0.0, 0.0])
            det_x = float(base_grasp[0])
            det_y = float(base_grasp[1])
            det_class = d.get("class", "?")
            conf = float(d.get("confidence", -1.0))
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
                match.conf = conf
                # Record this sighting for the per-object speed fit (uses the same
                # match verdict as identity — no extra association needed).
                self._update_velocity(match, detect_time, det_y, v, logger)
                # Class is VOTED, not latched: add this frame's confidence-weighted
                # vote and adopt the running argmax. The spawn frame is often the
                # noisy entry-edge frame (low conf), so a PET that misfired as metal
                # on spawn is corrected here once consistent higher-confidence
                # transparent detections outweigh it — instead of being pushed
                # forever. Log only the flip (no per-frame spam).
                prev_class = match.class_name
                voted = match.vote_class(det_class, conf)
                if voted != prev_class:
                    match.class_name = voted
                    if logger is not None:
                        logger.warn(
                            f"[track-RECLASS] id={match.track_id} {prev_class} -> "
                            f"{voted} (votes {_fmt_votes(match.class_votes)}; "
                            f"det {det_class} conf={conf:.2f})"
                        )
                refreshed += 1
                continue
            new_obj = TrackedObject(
                T_aim_base=_make_transform(_R_GRASP_DEFAULT, base_aim),
                T_grasp_base=_make_transform(_R_GRASP_DEFAULT, base_grasp),
                class_name=det_class,
                detect_time=detect_time,
                cam_pos=tuple(d.get("cam", [0.0, 0.0, 0.0])),
                conf=conf,
            )
            # Seed the class vote with the spawn frame's confidence so a confident
            # spawn class isn't flipped by one stray frame, but a low-confidence one
            # (the usual misfire) is easily outvoted. class_name stays det_class here.
            new_obj.vote_class(det_class, conf)
            self._update_velocity(new_obj, detect_time, det_y, v, logger)  # seed anchor
            queue.add(new_obj)
            existing.append(new_obj)  # dedupe within the same intake too
            if logger is not None:
                logger.info(
                    f"[track-NEW] id={new_obj.track_id} class={new_obj.class_name} "
                    f"x={det_x:+.3f} y={det_y:+.3f} conf={conf:.2f}"
                )
            added += 1

        # Safety net: collapse tracks that are the same physical object. The
        # per-detection dedup above only compares a NEW detection against the
        # existing tracks; it cannot undo a duplicate that already slipped in
        # (e.g. spawned while the arm was mid-pick and the loop wasn't
        # ingesting, or from a detector centroid that jumped along a long
        # object). Without this the stale twin coasts down the belt and the arm
        # picks at empty space.
        merged = queue.merge_duplicates(
            detect_time, v, self.eps, self._eps_y_merge, logger=logger,
        )
        added = max(0, added - merged)

        if added > 0 and logger is not None:
            logger.info(
                f"New frame — {added} new object(s) added, {refreshed} re-anchored "
                f"(queue size: {len(queue._objects)}, belt {v:.3f} m/s)"
            )
        return added

"""Persistent tracked-object queue with conveyor-motion compensation.

Each entry remembers ``detect_time`` so its current Y is recomputed every
epoch from the live conveyor speed. Anything past ``-max_reach`` is
dropped, the rest sorted ascending by current Y so the head is the next
reachable target.

Frame-cooldown timing lives in ``FrameGate`` — this module is purely
about object lifecycle.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np


# Monotonic per-process track-id source. Diagnostic: lets logs follow one
# physical object across dedup re-anchors, and exposes duplicate/ghost tracks
# (a NEW id appearing for an object that should have re-matched an existing one).
_track_id_counter = itertools.count(1)


@dataclass
class TrackedObject:
    T_aim_base: np.ndarray   # pose at detection time (absolute, conveyor not yet applied)
    T_grasp_base: np.ndarray
    class_name: str
    detect_time: float       # time.time() when this object was observed
    # Raw camera-frame position [cx, cy, cz] (m) reported by the perception
    # stream, kept verbatim for belt_viz / diagnostics. None on legacy paths.
    cam_pos: tuple | None = None
    # Latest detection confidence (camera_debug "confidence"); -1.0 until set.
    # Updated on every dedup re-anchor so it reflects the most recent sighting.
    conf: float = -1.0
    # Stable id assigned at creation and KEPT across re-anchors, so logs can
    # follow this track and spot duplicates. Diagnostic only (not used for logic).
    track_id: int = field(default_factory=lambda: next(_track_id_counter))
    # Confidence-weighted class-vote tally {class_name: cumulative_weight}. The
    # class is NOT latched at spawn — the first frame is often the noisy entry-edge
    # frame, so a PET whose spawn misfired as "metal" would otherwise be routed to
    # push forever. Every detection adds its confidence here (see vote_class) and
    # the EFFECTIVE class_name is the running argmax.
    class_votes: dict = field(default_factory=dict)
    # Per-object belt-speed estimation (DetectionIntake fills these; the fit
    # policy lives there, mirroring how dedup lives in intake — this is data).
    #   y_anchors: recent [(detect_time, belt_frame_y)] observed while in the
    #     camera box, trimmed to OBJECT_VEL_WINDOW_S.
    #   v_est: fitted belt-direction speed [m/s, +] for THIS object, or None
    #     until enough valid anchors. Diagnostic only; control dead-reckoning
    #     continues to use the conveyor encoder speed.
    y_anchors: list = field(default_factory=list)
    v_est: float | None = None
    # Diagnostic only: a [<skill>-veto] line has been logged for this track, so
    # selection doesn't repeat it every epoch. The veto itself is re-evaluated
    # live each epoch (a class re-vote can re-route the object to a skill
    # without a veto) — only the LOG is once-per-track.
    veto_logged: bool = False

    def vote_class(self, cls: str, conf: float) -> str:
        """Add a confidence-weighted vote for ``cls``; return the winning class.

        Weight = max(conf, 0) + a tiny floor (so a missing/zero-confidence frame
        still counts once). Cumulative over the track's life, so a low-confidence
        spawn misclassification is quickly overridden by consistent higher-
        confidence detections, while a genuinely-confident class isn't flipped by a
        couple of stray frames. Does NOT mutate class_name — the caller adopts the
        returned winner (so it can log a flip).
        """
        self.class_votes[cls] = (
            self.class_votes.get(cls, 0.0) + max(float(conf), 0.0) + 1e-3
        )
        return max(self.class_votes, key=self.class_votes.get)


class TrackedObjectQueue:
    def __init__(self, max_reach: float, drop_below_y: float = 0.0) -> None:
        self._max_reach = max_reach
        # COARSE drop line: objects whose current y has fallen below this are
        # past the workspace and removed (keeps the head meaningful). Callers pass
        # the worst-case downstream reach edge (-max_reach); the PRECISE per-object
        # "still catchable?" test (lane-specific -y_b, plus arm timing) lives in
        # SkillContext.earliest_reachable_intercept, NOT here — so this must stay
        # coarse and must NOT be the old fixed intercept line, or it would drop
        # downstream-but-still-reachable objects.
        self._drop_below_y = drop_below_y
        self._objects: list[TrackedObject] = []

    def __len__(self) -> int:
        return len(self._objects)

    def __bool__(self) -> bool:
        return bool(self._objects)

    def head(self) -> TrackedObject:
        return self._objects[0]

    def has_next(self) -> bool:
        return len(self._objects) >= 2

    def peek_next(self) -> TrackedObject:
        return self._objects[1]

    def add(self, obj: TrackedObject) -> None:
        self._objects.append(obj)

    def pop_head(self) -> TrackedObject:
        return self._objects.pop(0)

    def merge_duplicates(self, now: float, v: float, eps_x: float,
                         eps_y_fn, logger=None) -> int:
        """Collapse tracks that are the same physical object; return how many
        were removed.

        The detector emits no per-object identity, so intake dedups each new
        detection against the existing tracks. That cannot repair a duplicate
        which already exists — and one appears whenever a track goes
        un-refreshed long enough for its dead-reckoned Y to leave the match
        window (the main loop does not ingest while a pick/throw trajectory is
        streaming), or when the detector's centroid jumps along a long object.
        The stale twin then coasts down the belt and the arm picks empty space.

        Objects are compared at a common time ``now`` (belt travels -Y).
        ``eps_y_fn(age, v)`` supplies the belt-direction window for the older of
        the pair, matching the intake policy. The FRESHEST-anchored track of a
        pair survives (its pose was confirmed most recently) and absorbs the
        other's class votes, so a merge never loses classification evidence.
        """
        if len(self._objects) < 2:
            return 0
        merged = 0
        kept: list[TrackedObject] = []
        # Freshest first: the survivor of each pair is the best-anchored one.
        for obj in sorted(self._objects, key=lambda o: -o.detect_time):
            oy = float(obj.T_grasp_base[1, 3]) - v * (now - obj.detect_time)
            ox = float(obj.T_grasp_base[0, 3])
            twin = None
            for k in kept:
                ky = float(k.T_grasp_base[1, 3]) - v * (now - k.detect_time)
                kx = float(k.T_grasp_base[0, 3])
                age = max(now - obj.detect_time, now - k.detect_time)
                if abs(kx - ox) < eps_x and abs(ky - oy) < eps_y_fn(age, v):
                    twin = k
                    break
            if twin is None:
                kept.append(obj)
                continue
            for cls, w in obj.class_votes.items():
                twin.class_votes[cls] = twin.class_votes.get(cls, 0.0) + w
            if twin.class_votes:
                twin.class_name = max(twin.class_votes, key=twin.class_votes.get)
            merged += 1
            if logger is not None:
                logger.warn(
                    f"[track-MERGE] duplicate id={obj.track_id} folded into "
                    f"id={twin.track_id} (same object at y={oy:+.3f}; "
                    f"class -> {twin.class_name})"
                )
        if merged:
            self._objects = kept
        return merged
    def remove(self, obj: TrackedObject) -> None:
        """Remove ``obj`` wherever it sits (no-op if absent).

        Selection needs this now that it walks PAST veto-skipped entries: the
        committed target / an uncatchable candidate is no longer always the
        head, so ``pop_head`` alone can't take it out.
        """
        try:
            self._objects.remove(obj)
        except ValueError:
            pass

    def update(self, now: float, conveyor_speed: float) -> None:
        """Drop anything past the pick line (drop_below_y), then sort by current Y.

        Belt travels in -Y; the head ends up as the smallest current-y still
        in front of (above) the pick line — i.e. "the next-front object that
        hasn't been passed by the robot yet."
        """
        v = conveyor_speed
        decorated = [
            (obj.T_aim_base[1, 3] - v * (now - obj.detect_time), obj)
            for obj in self._objects
        ]
        decorated = [(y, obj) for y, obj in decorated if y > self._drop_below_y]
        decorated.sort(key=lambda pair: pair[0])
        self._objects = [obj for _, obj in decorated]

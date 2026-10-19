"""Persistent tracked-object queue with conveyor-motion compensation.

Each entry remembers the encoder distance at first detection so its current Y
is recomputed from actual belt travel.  ``speed * age`` remains a legacy
fallback until the distance topic's first sample arrives.

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


# eq=False: a track IS its identity. The generated value __eq__ compares the
# numpy pose arrays, which raises "truth value of an array is ambiguous" —
# so list.remove()/index()/`in` on any non-head entry failed, and
# TrackedObjectQueue.remove() swallowed that ValueError: executed targets
# stayed queued and were manipulated twice (2026-10-19 HW run, ids 14/23).
@dataclass(eq=False)
class TrackedObject:
    T_aim_base: np.ndarray   # pose at detection time (absolute, conveyor not yet applied)
    T_grasp_base: np.ndarray
    class_name: str
    detect_time: float       # time.time() when this object was observed
    # Continuous encoder distance [m] corresponding to detect_time.  The
    # control Y anchor is intentionally frozen at first detection; later camera
    # frames update appearance/X but cannot move a queued object upstream.
    encoder_distance_m: float | None = None
    # Raw camera-frame position [cx, cy, cz] (m) reported by the perception
    # stream, kept verbatim for belt_viz / diagnostics. None on legacy paths.
    cam_pos: tuple | None = None
    # Full projected detection box, retained in both source belt frame and the
    # corrected base frame. Corner order follows the stream contract:
    # top-left, top-right, bottom-right, bottom-left.
    cam_bbox: tuple | None = None
    base_bbox_grasp: tuple | None = None
    base_bbox_aim: tuple | None = None
    # BBox is visualization/output data and may refresh after the frozen control
    # anchor, so retain its own encoder/time reference.
    bbox_encoder_distance_m: float | None = None
    bbox_detect_time: float | None = None
    # Latest detection confidence (camera_debug "confidence"); -1.0 until set.
    # Updated on every dedup re-anchor so it reflects the most recent sighting.
    conf: float = -1.0
    # Simulator-only suction success probability. Real detections do not carry
    # this, so hardware-facing RL mirrors use the deterministic default.
    suction_p: float = 1.0
    # Stable id assigned at creation and KEPT across re-anchors, so logs can
    # follow this track and spot duplicates. Diagnostic only (not used for logic).
    track_id: int = field(default_factory=lambda: next(_track_id_counter))
    # Optional simulator identity. Real camera detections do not provide this;
    # the MuJoCo world source does, so RL reward bookkeeping can attribute a
    # tracked object back to the physical sim box that later resolves.
    sim_object_id: int | None = None
    # Confidence-weighted class-vote tally {class_name: cumulative_weight}. The
    # class is NOT latched at spawn — the first frame is often the noisy entry-edge
    # frame, so a PET whose spawn misfired as "metal" would otherwise be routed to
    # push forever. Every detection adds its confidence here (see vote_class) and
    # the EFFECTIVE class_name is the running argmax.
    class_votes: dict = field(default_factory=dict)
    # Diagnostic only: a [<skill>-veto] line has been logged for this track, so
    # selection doesn't repeat it every epoch. The veto itself is re-evaluated
    # live each epoch (a class re-vote can re-route the object to a skill
    # without a veto) — only the LOG is once-per-track.
    veto_logged: bool = False
    # Producer-side identity, as "<epoch>:<id>" (see DetectionIntake). Tracks
    # with different ids are normally different objects; the exception is a
    # producer that splits one object into two masks / re-ids it at the frame
    # edge -- merge_duplicates folds those when their footprints overlap, and
    # the absorbed id is kept in ``src_aliases`` so its later detections still
    # associate with the survivor. None means the producer offered no identity.
    src_track_id: str | None = None
    src_aliases: set = field(default_factory=set)
    # True once the pose and class below were taken from the frame where the
    # producer saw this object closest to the image centre -- the one view
    # where parallax and lens distortion vanish. Latched: later frames refine
    # display geometry only.
    center_confirmed: bool = False
    # Detections folded into this track (spawn frame = 1). A real object is
    # seen ~30 frames while it crosses the camera view; a one-frame ghost
    # (oversized merge box, stray mask) never gets past a few.
    n_hits: int = 1

    def has_src(self, src: "str | None") -> bool:
        """True iff ``src`` is this track's producer id or one it absorbed."""
        return src is not None and (src == self.src_track_id or src in self.src_aliases)

    def footprint_at(self, now: float, conveyor_speed: float,
                     belt_distance_m: float | None = None):
        """Axis-aligned XY footprint ``(x0, x1, y0, y1)`` of the latest bbox,
        carried along the belt from its own capture to ``now``; None if absent."""
        if not self.base_bbox_grasp:
            return None
        pts = np.asarray(self.base_bbox_grasp, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 2:
            return None
        if self.bbox_encoder_distance_m is not None and belt_distance_m is not None:
            travel = float(belt_distance_m) - float(self.bbox_encoder_distance_m)
        elif self.bbox_detect_time is not None:
            travel = float(conveyor_speed) * (float(now) - float(self.bbox_detect_time))
        else:
            travel = 0.0
        return (float(pts[:, 0].min()), float(pts[:, 0].max()),
                float(pts[:, 1].min()) - travel, float(pts[:, 1].max()) - travel)

    def y_at(self, now: float, conveyor_speed: float,
             belt_distance_m: float | None = None) -> float:
        """Current control Y from encoder travel, with speed×age fallback."""
        anchor_y = float(self.T_grasp_base[1, 3])
        if self.encoder_distance_m is not None and belt_distance_m is not None:
            return anchor_y - (float(belt_distance_m) - self.encoder_distance_m)
        return anchor_y - float(conveyor_speed) * (float(now) - self.detect_time)

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


def footprint_overlap(a, b) -> float:
    """Intersection area over the SMALLER footprint area (0 if either missing).

    Over-the-smaller (not IoU) so a half-object mask inside its full-object
    twin, or a real box inside an oversized glued box, scores ~1.
    """
    if a is None or b is None:
        return 0.0
    ix = min(a[1], b[1]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[2], b[2])
    if ix <= 0.0 or iy <= 0.0:
        return 0.0
    area_a = (a[1] - a[0]) * (a[3] - a[2])
    area_b = (b[1] - b[0]) * (b[3] - b[2])
    smaller = min(area_a, area_b)
    return float(ix * iy / smaller) if smaller > 0.0 else 0.0


def _track_quality(obj: "TrackedObject") -> tuple:
    return (bool(obj.center_confirmed), int(obj.n_hits))


def _absorb(survivor: "TrackedObject", gone: "TrackedObject") -> None:
    """Fold ``gone``'s evidence (class votes, hits, producer ids) into ``survivor``."""
    for cls, w in gone.class_votes.items():
        survivor.class_votes[cls] = survivor.class_votes.get(cls, 0.0) + w
    if survivor.class_votes and not survivor.center_confirmed:
        survivor.class_name = max(survivor.class_votes, key=survivor.class_votes.get)
    survivor.n_hits += gone.n_hits
    for src in (gone.src_track_id, *gone.src_aliases):
        if src is not None and src != survivor.src_track_id:
            survivor.src_aliases.add(src)


def is_established(obj: "TrackedObject", min_hits: int) -> bool:
    """A track the arm may act on: seen in ``min_hits`` frames or pose-confirmed
    at the image centre. Younger tracks stay queued (they keep associating and
    can still merge) but are never selected -- one-frame ghosts die there."""
    return bool(obj.center_confirmed) or int(obj.n_hits) >= int(min_hits)


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
                         eps_y_fn, logger=None,
                         belt_distance_m: float | None = None) -> int:
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
            oy = obj.y_at(now, v, belt_distance_m)
            ox = float(obj.T_grasp_base[0, 3])
            twin = None
            by_overlap = False
            for i, k in enumerate(kept):
                if obj.has_src(k.src_track_id) or k.has_src(obj.src_track_id):
                    # Same producer identity (own id or an absorbed alias):
                    # the same object wherever its centroids sit.
                    twin, by_overlap, twin_i = k, True, i
                    break
                if (obj.src_track_id is not None
                        and k.src_track_id is not None
                        and not (obj.has_src(k.src_track_id) or k.has_src(obj.src_track_id))):
                    # Different producer ids are usually different objects --
                    # unless their footprints genuinely overlap: the producer
                    # split one object into two masks, glued two into one
                    # oversized box, or re-id'd it at the frame edge
                    # (2026-10-19 HW: cam ids 39/42 on one bottle). A centroid
                    # gate can't tell these from touching neighbours; area
                    # overlap can.
                    frac = footprint_overlap(
                        obj.footprint_at(now, v, belt_distance_m),
                        k.footprint_at(now, v, belt_distance_m))
                    if frac >= self.src_merge_overlap:
                        twin, by_overlap = k, True
                        twin_i = i
                        break
                    continue
                ky = k.y_at(now, v, belt_distance_m)
                kx = float(k.T_grasp_base[0, 3])
                age = max(now - obj.detect_time, now - k.detect_time)
                if abs(kx - ox) < eps_x and abs(ky - oy) < eps_y_fn(age, v):
                    twin = k
                    break
            if twin is None:
                kept.append(obj)
                continue
            gone = obj
            if by_overlap and _track_quality(obj) > _track_quality(twin):
                # The established track survives, not whichever was re-anchored
                # last -- a fresh ghost must never replace a real object.
                kept[twin_i], twin, gone = obj, obj, twin
            _absorb(twin, gone)
            merged += 1
            if logger is not None:
                logger.warn(
                    f"[track-MERGE] duplicate id={gone.track_id} folded into "
                    f"id={twin.track_id} (same object at y={oy:+.3f}"
                    f"{', overlapping footprint, producer ids ' + str(sorted(twin.src_aliases | {twin.src_track_id})) if by_overlap else ''}; "
                    f"class -> {twin.class_name})"
                )
        if merged:
            self._objects = kept
        return merged

    #: Minimum footprint overlap (intersection / smaller area) for folding two
    #: tracks that carry DIFFERENT producer ids (see merge_duplicates).
    src_merge_overlap: float = 0.5

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

    def update(self, now: float, conveyor_speed: float,
               belt_distance_m: float | None = None) -> None:
        """Drop anything past the pick line (drop_below_y), then sort by current Y.

        Belt travels in -Y; the head ends up as the smallest current-y still
        in front of (above) the pick line — i.e. "the next-front object that
        hasn't been passed by the robot yet."
        """
        v = conveyor_speed
        decorated = [
            (obj.y_at(now, v, belt_distance_m), obj)
            for obj in self._objects
        ]
        decorated = [(y, obj) for y, obj in decorated if y > self._drop_below_y]
        decorated.sort(key=lambda pair: pair[0])
        self._objects = [obj for _, obj in decorated]

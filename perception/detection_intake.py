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
from collections import deque
from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from gp8_control.tracking import TrackedObject

if TYPE_CHECKING:
    from gp8_control.tracking import TrackedObjectQueue


def _make_transform(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float).ravel()
    return T


def _freeze_bbox(box) -> tuple | None:
    """Store a JSON box as an immutable tuple-of-tuples, preserving ``None``."""
    if box is None:
        return None
    return tuple(tuple(float(v) for v in point) for point in box)


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


def _footprint_extent(bbox) -> float:
    """Longest axis-aligned XY extent of a base-frame bbox [m] (0 if absent)."""
    if not bbox:
        return 0.0
    try:
        pts = np.asarray(bbox, dtype=float)[:, :2]
    except (TypeError, ValueError, IndexError):
        return 0.0
    return float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])))


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

    #: 게이트 밖(매칭 불가) 셀의 비용 — 할당 결과가 이 값이면 미매칭으로 처리.
    _INFEASIBLE = 1e6

    def __init__(self, eps: float, drift_frac: float = 0.25,
                 eps_y_max: float = 0.20, merge_eps_y_max: float = 0.10,
                 assoc: str = "hungarian",
                 max_footprint_m: float | None = None) -> None:
        #: spatial match radius [m] — OBJECT_MATCH_EPSILON on the app Config.
        self.eps = eps
        #: belt-direction tolerance growth per second of dead reckoning, as a
        #: fraction of belt speed (Config.OBJECT_MATCH_DRIFT_FRAC).
        self.drift_frac = drift_frac
        #: cap on the grown Y window [m] (Config.OBJECT_MATCH_EPS_Y_MAX).
        self.eps_y_max = eps_y_max
        #: tighter cap for the merge pass (Config.OBJECT_MERGE_EPS_Y_MAX).
        self.merge_eps_y_max = merge_eps_y_max
        #: 검출↔트랙 연관 방식 (Config.TRACK_ASSOC): "hungarian" | "greedy".
        self.assoc = assoc
        #: Drop detections whose base-frame footprint is longer than this on
        #: either axis [m] (None = off). Real items are <~0.25 m; boxes of
        #: 0.39/0.45 m were the detector gluing an object to its neighbour /
        #: the gap (2026-10-19 HW) and spawned a twin of a real track.
        self.max_footprint_m = max_footprint_m
        self.n_dropped_oversize = 0
        #: camera_debug republishes its latest snapshot on a timer.  Process each
        #: receipt timestamp once so class votes/association are frame-based.
        self._last_receipt_time: float | None = None
        #: Producer ids are unique only within one producer run: a camera
        #: restart begins again at 0 and would otherwise look like the objects
        #: we are already tracking.  Bump an epoch whenever the incoming ids
        #: jump backwards and qualify every stored id with it, so ids from
        #: different runs can never compare equal.
        self._src_epoch = 0
        self._src_max = -1
        self._recent_src: deque = deque()

    def _associate_hungarian(self, tracks, dets, v, detect_time,
                             belt_distance_m=None):
        """프레임 전역 최적 연관 (Hungarian / scipy linear_sum_assignment).

        "전체 거리" = 채택된 (트랙, 검출) 짝들의 창-정규화 거리의 합. 이 합이
        최소가 되는 짝 조합을 한 번에 고르므로, 허용창이 겹치는 이웃 물체에서
        선착순(greedy) 매칭이 일으키는 트랙 교차(스왑)가 생기지 않는다.

        게이트는 greedy(_matches)와 동일 — x는 ±eps, y는 나이 비례 창 밖이면
        매칭 불가. 비용 = hypot(dx/eps, dy/eps_y) + (클래스 불일치 시 +0.25:
        위치가 비슷할 때만 판가름하는 소프트 페널티 — 캔·병이 나란히 올 때 도움).
        리턴: (pairs=[(track, det_dict)], unmatched=[det_dict]).
        """
        if not tracks or not dets:
            return [], list(dets)
        C = np.full((len(tracks), len(dets)), self._INFEASIBLE)
        for i, o in enumerate(tracks):
            age = detect_time - o.detect_time
            ox = float(o.T_grasp_base[0, 3])
            oy = o.y_at(detect_time, v, belt_distance_m)
            eps_y = self._eps_y(age, v)
            for j, dd in enumerate(dets):
                # The producer tracks objects across frames, so when both
                # sides carry an id the answer is already known -- and it is
                # known even where the position gate fails: through a pick
                # trajectory that starved this track of updates for seconds,
                # or a centroid that jumped as a long object cleared the
                # frame edge.  Those two cases are exactly where the gate
                # spawns the duplicate that later has the arm grabbing air.
                src_o, src_d = o.src_track_id, dd["src_track_id"]
                if src_o is not None and src_d is not None:
                    if o.has_src(src_d):   # own id, or one absorbed in a merge
                        C[i, j] = 0.0      # same object: wins the assignment
                    continue               # different: never pair them
                ndx = abs(ox - dd["x"]) / self.eps
                ndy = abs(oy - dd["y"]) / eps_y
                if ndx < 1.0 and ndy < 1.0:
                    cost = float(np.hypot(ndx, ndy))
                    if dd["cls"] != o.class_name:
                        cost += 0.25
                    C[i, j] = cost
        rows, cols = linear_sum_assignment(C)
        ok = [(i, j) for i, j in zip(rows, cols) if C[i, j] < self._INFEASIBLE]
        matched_j = {j for _, j in ok}
        return ([(tracks[i], dets[j]) for i, j in ok],
                [dd for j, dd in enumerate(dets) if j not in matched_j])

    def _src_key(self, raw) -> "str | None":
        """Qualify a producer id with the current epoch, or None if absent."""
        if raw is None:
            return None
        return f"{self._src_epoch}:{int(raw)}"

    def _note_producer_ids(self, detections, now: float) -> None:
        """Detect a producer restart: ids that jump backwards start an epoch.

        A live producer numbers NEW objects upward, but a frame can legitimately
        hold only older objects -- e.g. right after a one-frame ghost took a
        high number (2026-10-19 HW: ghost ids 41/46 made the old "max id
        dropped" rule declare four bogus restarts in 2 minutes, re-keying the
        objects still in view and spawning duplicate tracks). So a restart is
        declared only when EVERY id in the frame is below every id seen in the
        last ``_RESTART_WINDOW_S`` -- a renumbered producer starts again at 0,
        which no still-visible object can match.
        """
        ids = [d.get("src_track_id") for d in detections]
        ids = [int(i) for i in ids if i is not None]
        if not ids:
            return
        recent = self._recent_src
        while recent and now - recent[0][0] > self._RESTART_WINDOW_S:
            recent.popleft()
        hi = max(ids)
        dropped = hi + 2 < self._src_max
        below_all_recent = (not recent) or hi < min(i for _, i in recent)
        if dropped and below_all_recent:
            self._src_epoch += 1
            self._src_max = hi
            recent.clear()
        else:
            self._src_max = max(self._src_max, hi)
        recent.extend((now, i) for i in ids)

    #: Restart test window [s]: longer than an object's ~3.5 s camera transit.
    _RESTART_WINDOW_S = 5.0

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
        belt_distance_m: float | None = None,
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
        detect_time = float(snapshot.get("receipt_time", time.time()))
        if (self._last_receipt_time is not None
                and detect_time <= self._last_receipt_time):
            return 0
        self._last_receipt_time = detect_time
        detections = [
            d for d in snapshot.get("detections", []) if d.get("in_workspace")
        ]
        if self.max_footprint_m is not None:
            kept = [d for d in detections
                    if _footprint_extent(d.get("base_bbox_grasp")) <= self.max_footprint_m]
            dropped = len(detections) - len(kept)
            if dropped:
                self.n_dropped_oversize += dropped
                if logger is not None:
                    logger.warn(
                        f"[track-OVERSIZE] dropped {dropped} detection(s) longer than "
                        f"{self.max_footprint_m:.2f} m (total {self.n_dropped_oversize})"
                    )
            detections = kept
        if not detections:
            return 0
        self._note_producer_ids(detections, detect_time)

        # camera_debug already applied the camera→base transform, Z offsets, and
        # v*delay back-projection. ``receipt_time`` is the moment for which the
        # corrected positions are valid; the queue extrapolates forward from there.
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
            oy = obj.y_at(detect_time, v, belt_distance_m)
            return (abs(ox - det_x) < eps
                    and abs(oy - det_y) < self._eps_y(age, v))

        added = 0
        refreshed = 0

        dets = []
        for d in detections:
            base_grasp = d.get("base_grasp", [0.0, 0.0, 0.0])
            dets.append(dict(
                aim=d.get("base_aim", [0.0, 0.0, 0.0]),
                grasp=base_grasp,
                x=float(base_grasp[0]), y=float(base_grasp[1]),
                cls=d.get("class", "?"),
                conf=float(d.get("confidence", -1.0)),
                suction_p=float(d.get("suction_p", 1.0)),
                cam=d.get("cam", [0.0, 0.0, 0.0]),
                cam_bbox=d.get("cam_bbox"),
                base_bbox_grasp=d.get("base_bbox_grasp"),
                base_bbox_aim=d.get("base_bbox_aim"),
                sim_object_id=d.get("sim_object_id"),
                src_track_id=self._src_key(d.get("src_track_id")),
                center_radius=d.get("center_radius"),
                center_confirmed=bool(d.get("center_confirmed")),
            ))

        def _reanchor(match: TrackedObject, dd: dict) -> None:
            """Fold a re-detection into an existing track.

            Display geometry and confidence refresh every frame.  The pose and
            the class, though, are taken from the frame that saw this object
            CLOSEST TO THE IMAGE CENTRE, because that is where the measurement
            is good: height parallax and lens distortion are zero on the
            optical axis and grow with radius, and an object near the edge may
            still be partly outside the frame with its centroid pulled inward.
            Once the object is past centre the radius only grows, so the
            estimate freezes on its own.

            Before that frame -- and with a producer that never sends the
            flag -- this is the previous rule: refine lane X and Z, leave Y
            on its spawn anchor, and vote the class.
            """
            nonlocal refreshed
            match.cam_pos = tuple(dd["cam"])
            match.cam_bbox = _freeze_bbox(dd["cam_bbox"])
            match.base_bbox_grasp = _freeze_bbox(dd["base_bbox_grasp"])
            match.base_bbox_aim = _freeze_bbox(dd["base_bbox_aim"])
            if dd.get("sim_object_id") is not None:
                match.sim_object_id = int(dd["sim_object_id"])
            match.bbox_encoder_distance_m = belt_distance_m
            match.bbox_detect_time = detect_time
            match.conf = dd["conf"]
            match.suction_p = dd["suction_p"]
            match.n_hits += 1
            src = dd["src_track_id"]
            if src is not None:
                if match.src_track_id is None:
                    match.src_track_id = src
                elif not match.has_src(src):
                    # A second producer id landed on this track by position:
                    # remember it so its next frames associate here too.
                    match.src_aliases.add(src)

            if dd["center_confirmed"]:
                # The producer says this frame is the object's closest
                # approach to the image centre, which it can tell as it
                # happens because it knows the belt velocity.  Take the
                # whole pose from here -- and with it the time/encoder
                # reference the belt extrapolation counts from -- then lock
                # it: every later frame only gets farther from the centre.
                match.center_confirmed = True
                match.T_aim_base[0, 3] = dd["aim"][0]
                match.T_aim_base[1, 3] = dd["aim"][1]
                match.T_aim_base[2, 3] = dd["aim"][2]
                match.T_grasp_base[0, 3] = dd["grasp"][0]
                match.T_grasp_base[1, 3] = dd["grasp"][1]
                match.T_grasp_base[2, 3] = dd["grasp"][2]
                match.detect_time = detect_time
                match.encoder_distance_m = belt_distance_m
                prev_class = match.class_name
                match.class_name = dd["cls"]
                match.class_votes = {dd["cls"]: max(float(dd["conf"]), 0.0) + 1e-3}
                if logger is not None:
                    logger.info(
                        f"[track-CONFIRM] id={match.track_id} {dd['cls']} "
                        f"(was {prev_class}, conf={dd['conf']:.2f}, "
                        f"r={dd['center_radius']}) pose latched at centre"
                    )
            elif not match.center_confirmed:
                # Still on the way in (or a producer that never confirms):
                # the previous rule -- refine lane X and Z, leave Y on its
                # spawn anchor, and keep voting the class.
                match.T_aim_base[0, 3] = dd["aim"][0]
                match.T_aim_base[2, 3] = dd["aim"][2]
                match.T_grasp_base[0, 3] = dd["grasp"][0]
                match.T_grasp_base[2, 3] = dd["grasp"][2]
                prev_class = match.class_name
                voted = match.vote_class(dd["cls"], dd["conf"])
                if voted != prev_class:
                    match.class_name = voted
                    if logger is not None:
                        logger.warn(
                            f"[track-RECLASS] id={match.track_id} {prev_class} "
                            f"-> {voted} (votes {_fmt_votes(match.class_votes)}; "
                            f"det {dd['cls']} conf={dd['conf']:.2f})"
                        )
            # Once confirmed the pose and class are final; later frames only
            # refreshed the display geometry above.
            refreshed += 1

        # ---- 검출 ↔ 트랙 연관 ----
        # hungarian(기본): 프레임 전역 최적 할당 — "전체 거리"(짝별 창-정규화
        # 거리 합) 최소 조합. greedy: 구 선착순 (검출마다 창 안 첫 트랙).
        if self.assoc == "hungarian":
            pairs, unmatched = self._associate_hungarian(
                existing, dets, v, detect_time, belt_distance_m)
        else:
            pairs, unmatched = [], []
            for dd in dets:
                m = next((o for o in existing if _matches(o, dd["x"], dd["y"])), None)
                (pairs.append((m, dd)) if m is not None else unmatched.append(dd))

        for match, dd in pairs:
            _reanchor(match, dd)

        for dd in unmatched:
            # 한 프레임에 같은 물체가 두 박스로 잡히는 중복 검출 흡수: 이미
            # 매칭됐거나 방금 스폰된 트랙과 겹치면 스폰 대신 재앵커한다
            # (hungarian은 1:1 할당이라 두 번째 박스가 여기로 온다 — 구 greedy가
            # 같은 트랙을 두 번 재앵커하던 동작의 보존).
            # A box whose producer id already belongs to a track (e.g. the 2nd
            # mask of a split object, after the 1:1 assignment gave the track
            # its sibling box) folds into that track rather than spawning.
            fb = next((o for o in existing if o.has_src(dd["src_track_id"])), None)
            if fb is None:
                fb = next((o for o in existing if _matches(o, dd["x"], dd["y"])), None)
            if fb is not None:
                _reanchor(fb, dd)
                continue
            new_obj = TrackedObject(
                T_aim_base=_make_transform(_R_GRASP_DEFAULT, dd["aim"]),
                T_grasp_base=_make_transform(_R_GRASP_DEFAULT, dd["grasp"]),
                class_name=dd["cls"],
                detect_time=detect_time,
                encoder_distance_m=belt_distance_m,
                cam_pos=tuple(dd["cam"]),
                cam_bbox=_freeze_bbox(dd["cam_bbox"]),
                base_bbox_grasp=_freeze_bbox(dd["base_bbox_grasp"]),
                base_bbox_aim=_freeze_bbox(dd["base_bbox_aim"]),
                bbox_encoder_distance_m=belt_distance_m,
                bbox_detect_time=detect_time,
                conf=dd["conf"],
                suction_p=dd["suction_p"],
                sim_object_id=(
                    None if dd.get("sim_object_id") is None
                    else int(dd["sim_object_id"])
                ),
                src_track_id=dd["src_track_id"],
                center_confirmed=dd["center_confirmed"],
            )
            # Seed the class vote with the spawn frame's confidence so a confident
            # spawn class isn't flipped by one stray frame, but a low-confidence one
            # (the usual misfire) is easily outvoted. class_name stays det_class here.
            new_obj.vote_class(dd["cls"], dd["conf"])
            queue.add(new_obj)
            existing.append(new_obj)  # dedupe within the same intake too
            if logger is not None:
                logger.info(
                    f"[track-NEW] id={new_obj.track_id} class={new_obj.class_name} "
                    f"x={dd['x']:+.3f} y={dd['y']:+.3f} conf={dd['conf']:.2f}"
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
            belt_distance_m=belt_distance_m,
        )
        added = max(0, added - merged)

        if added > 0 and logger is not None:
            logger.info(
                f"New frame — {added} new object(s) added, {refreshed} re-anchored "
                f"(queue size: {len(queue._objects)}, belt {v:.3f} m/s)"
            )
        return added

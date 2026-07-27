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
from scipy.optimize import linear_sum_assignment

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

    #: 게이트 밖(매칭 불가) 셀의 비용 — 할당 결과가 이 값이면 미매칭으로 처리.
    _INFEASIBLE = 1e6

    def __init__(self, eps: float, drift_frac: float = 0.25,
                 eps_y_max: float = 0.20, merge_eps_y_max: float = 0.10,
                 assoc: str = "hungarian",
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
        #: 검출↔트랙 연관 방식 (Config.TRACK_ASSOC): "hungarian" | "greedy".
        self.assoc = assoc
        #: per-object velocity-fit policy (Config.OBJECT_VEL_*).
        self.vel_window_s = vel_window_s
        self.vel_min_anchors = vel_min_anchors
        self.vel_min_span_s = vel_min_span_s
        self.vel_max_rms = vel_max_rms
        self.vel_clamp_frac = vel_clamp_frac
        #: track_ids already logged as speed-deviating (one line per object).
        self._vel_logged: set = set()
        #: optional callback(v_fit) — CameraSpeedTracker.observe 등, 품질 게이트를
        #: 통과한 물체별 속도 fit의 소비자. 벨트-밴드 클램프 **이전에** 호출한다:
        #: camera 모드에선 클램프 기준(벨트 속도)이 바로 이 소비자의 추정치라,
        #: 클램프 뒤에 보고하면 잘못된 초기 fallback에 갇혀 수렴하지 못한다.
        self.speed_sink = None

    def _associate_hungarian(self, tracks, dets, v, detect_time):
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
            oy = float(o.T_grasp_base[1, 3] - v * age)
            eps_y = self._eps_y(age, v)
            for j, dd in enumerate(dets):
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
        if self.speed_sink is not None:       # 클램프 이전 보고 (docstring 참고)
            self.speed_sink(v_fit, obj.track_id)
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

        dets = []
        for d in detections:
            base_grasp = d.get("base_grasp", [0.0, 0.0, 0.0])
            dets.append(dict(
                aim=d.get("base_aim", [0.0, 0.0, 0.0]),
                grasp=base_grasp,
                x=float(base_grasp[0]), y=float(base_grasp[1]),
                cls=d.get("class", "?"),
                conf=float(d.get("confidence", -1.0)),
                cam=d.get("cam", [0.0, 0.0, 0.0]),
            ))

        def _reanchor(match: TrackedObject, dd: dict) -> None:
            # Re-anchor the existing track to this fresh detection instead of
            # adding a duplicate. Resetting the extrapolation reference
            # (detect_time + pose) every frame keeps drift below the window so a
            # 2nd "object" never spawns at the same spot.
            nonlocal refreshed
            match.T_aim_base = _make_transform(_R_GRASP_DEFAULT, dd["aim"])
            match.T_grasp_base = _make_transform(_R_GRASP_DEFAULT, dd["grasp"])
            match.detect_time = detect_time
            match.cam_pos = tuple(dd["cam"])
            match.conf = dd["conf"]
            # Record this sighting for the per-object speed fit (uses the same
            # match verdict as identity — no extra association needed, so the
            # anchors follow whatever TRACK_ASSOC decided this frame).
            self._update_velocity(match, detect_time, dd["y"], v, logger)
            # Class is VOTED, not latched: add this frame's confidence-weighted
            # vote and adopt the running argmax (spawn frame is often the noisy
            # entry-edge frame). Log only the flip (no per-frame spam).
            prev_class = match.class_name
            voted = match.vote_class(dd["cls"], dd["conf"])
            if voted != prev_class:
                match.class_name = voted
                if logger is not None:
                    logger.warn(
                        f"[track-RECLASS] id={match.track_id} {prev_class} -> "
                        f"{voted} (votes {_fmt_votes(match.class_votes)}; "
                        f"det {dd['cls']} conf={dd['conf']:.2f})"
                    )
            refreshed += 1

        # ---- 검출 ↔ 트랙 연관 ----
        # hungarian(기본): 프레임 전역 최적 할당 — "전체 거리"(짝별 창-정규화
        # 거리 합) 최소 조합. greedy: 구 선착순 (검출마다 창 안 첫 트랙).
        if self.assoc == "hungarian":
            pairs, unmatched = self._associate_hungarian(
                existing, dets, v, detect_time)
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
            fb = next((o for o in existing if _matches(o, dd["x"], dd["y"])), None)
            if fb is not None:
                _reanchor(fb, dd)
                continue
            new_obj = TrackedObject(
                T_aim_base=_make_transform(_R_GRASP_DEFAULT, dd["aim"]),
                T_grasp_base=_make_transform(_R_GRASP_DEFAULT, dd["grasp"]),
                class_name=dd["cls"],
                detect_time=detect_time,
                cam_pos=tuple(dd["cam"]),
                conf=dd["conf"],
            )
            # Seed the class vote with the spawn frame's confidence so a confident
            # spawn class isn't flipped by one stray frame, but a low-confidence one
            # (the usual misfire) is easily outvoted. class_name stays det_class here.
            new_obj.vote_class(dd["cls"], dd["conf"])
            self._update_velocity(new_obj, detect_time, dd["y"], v, logger)  # seed anchor
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
        )
        added = max(0, added - merged)

        if added > 0 and logger is not None:
            logger.info(
                f"New frame — {added} new object(s) added, {refreshed} re-anchored "
                f"(queue size: {len(queue._objects)}, belt {v:.3f} m/s)"
            )
        return added

"""Is the camera -> robot-base mapping oriented right? Watch the objects move.

On the belt every object travels toward the robot, i.e. base-frame Y must
DECREASE at the belt speed. With the producer's per-object ``track_ids`` the
along-belt velocity of each object can be measured from consecutive frames, so
a sign error (camera re-mounted rotated, 2026-10-19: everything picked ~2.7 s
early) or a scale error shows up within a few objects -- before the arm moves.

Only the along-belt (Y) axis is observable this way; the across-belt (X) sign
needs a placed object (nothing moves sideways).
"""

from __future__ import annotations

import statistics
from collections import deque

OK, FLIPPED, SCALE, NO_DATA, BELT_STOPPED = "OK", "FLIPPED", "SCALE?", "NO DATA", "BELT STOPPED"


class BeltDirectionMonitor:
    def __init__(self, *, window_s: float = 30.0, min_span_s: float = 0.4,
                 min_span_m: float = 0.04, ok_band: float = 0.3,
                 min_belt_mps: float = 0.02, min_tracks: int = 2) -> None:
        self.window_s = window_s
        self.min_span_s = min_span_s
        self.min_span_m = min_span_m
        self.ok_band = ok_band
        self.min_belt_mps = min_belt_mps
        self.min_tracks = min_tracks
        self._first: dict = {}            # src id -> (t, y)
        self._estimates: deque = deque()  # (t_last, src id, vy)
        self._belt = 0.0
        self._now = 0.0
        self._last_verdict: dict | None = None   # survives an empty belt

    def update(self, receipt_time: float, detections, belt_mps: float) -> None:
        """Feed one camera_debug frame (base-frame detections)."""
        self._belt = float(belt_mps)
        t = float(receipt_time)
        self._now = t
        live = set()
        for d in detections:
            src = d.get("src_track_id")
            if src is None:
                continue
            y = float(d["base_grasp"][1])
            live.add(src)
            first = self._first.setdefault(src, (t, y))
            span_t = t - first[0]
            if span_t >= self.min_span_s and abs(y - first[1]) >= self.min_span_m:
                vy = (y - first[1]) / span_t
                # one estimate per object, refreshed as it keeps moving
                self._estimates = deque(e for e in self._estimates if e[1] != src)
                self._estimates.append((t, src, vy))
        while self._estimates and t - self._estimates[0][0] > self.window_s:
            self._estimates.popleft()
        if len(self._first) > 200:
            self._first = {k: v for k, v in self._first.items() if k in live}
        st = self._fresh_status()
        if st["state"] in (OK, FLIPPED, SCALE):
            self._last_verdict = {**st, "t": t}

    def status(self) -> dict:
        st = self._fresh_status()
        if st["state"] in (OK, FLIPPED, SCALE):
            return {**st, "age_s": 0.0}
        if self._last_verdict is not None:
            # Nothing measurable right now (belt empty / stopped): keep showing
            # the last real verdict, marked with its age.
            age = self._now - self._last_verdict["t"]
            last = {k: v for k, v in self._last_verdict.items() if k != "t"}
            return {**last, "age_s": age, "stale": True,
                    "message": f"{last['message']} (마지막 판정 {age:.0f}s 전)"}
        return {**st, "age_s": None}

    def _fresh_status(self) -> dict:
        v = self._belt
        vys = [e[2] for e in self._estimates]
        out = {"belt_mps": v, "n_tracks": len(vys)}
        if abs(v) < self.min_belt_mps:
            return {**out, "state": BELT_STOPPED, "ratio": None,
                    "message": "벨트 정지 — 물체가 움직여야 방향을 판정할 수 있음"}
        if len(vys) < self.min_tracks:
            return {**out, "state": NO_DATA, "ratio": None,
                    "message": f"물체 {self.min_tracks}개 이상이 지나가야 판정 (현재 {len(vys)})"}
        vy = statistics.median(vys)
        ratio = -vy / v          # +1: objects move toward the robot at belt speed
        out.update(object_vy=vy, ratio=ratio)
        if abs(ratio - 1.0) <= self.ok_band:
            return {**out, "state": OK,
                    "message": f"정상 — 물체가 로봇 쪽으로 {abs(vy):.3f} m/s (벨트 {v:.3f})"}
        if abs(ratio + 1.0) <= self.ok_band:
            return {**out, "state": FLIPPED,
                    "message": f"반대 방향! 물체 y가 +{vy:.3f} m/s 로 증가 — "
                               "카메라 Y 부호를 바꾸세요 (GP8_CAM_SIGN_CY)"}
        return {**out, "state": SCALE,
                "message": f"속도 불일치 — 물체 {-vy:.3f} m/s vs 벨트 {v:.3f} "
                           f"(비율 {ratio:.2f}): 카메라 스케일/엔코더 확인"}

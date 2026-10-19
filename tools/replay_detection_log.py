"""Replay a recorded camera NDJSON log through the robot-side tracker, offline.

    python -m gp8_control.tools.replay_detection_log CAMERA.jsonl \
        [--steps RUN/steps.jsonl] [--belt 0.22] [--sign-cx +1 --sign-cy -1]

The camera PC writes every streamed record to results_local/detections.jsonl.
This tool converts each record exactly like ``camera_debug`` (cam -> base
transform, v*delay back-projection), then feeds the frames to
``DetectionIntake`` + ``TrackedObjectQueue`` on the control loop's schedule and
reports how many robot tracks one physical object produced.

Two configurations are compared on the identical input:

* ``before`` -- the loop ingests only the LATEST frame at each tick and the
  tracker has no ghost/duplicate guards (the 2026-10-19 behaviour);
* ``after``  -- every frame is drained in order, oversize boxes are dropped,
  footprint-overlapping tracks merge across producer ids, and a track must earn
  ``TRACK_MIN_HITS`` (or centre confirmation) before it is selectable.

With ``--steps`` (an s2r run's steps.jsonl) the loop is "busy" during each
executed skill and ingests nothing, as on hardware; without it the loop ticks
continuously. rclpy-free.
"""

from __future__ import annotations

import argparse
import collections
import json
from types import SimpleNamespace

import numpy as np

from gp8_control.config import Config
from gp8_control.perception import extrinsics
from gp8_control.perception.bbox_geometry import as_bbox, bbox_center, bbox_to_base
from gp8_control.perception.detection_intake import DetectionIntake
from gp8_control.tracking.object_queue import (
    TrackedObjectQueue,
    footprint_overlap,
    is_established,
)

TICK_S = 0.04          # control-loop period while idle (Config.TIME_STEP)
SHORT_ID_FRAMES = 3    # a producer id seen in <= this many frames = ghost


def load_camera_log(path):
    recs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                recs.append(json.loads(line))
            except ValueError:
                continue
    recs.sort(key=lambda r: r.get("timestamp", 0.0))
    return recs


def to_snapshot(rec, v, sign_cx, sign_cy):
    """One camera record -> the camera_debug snapshot the app ingests."""
    receipt = float(rec.get("server_send_timestamp") or rec["timestamp"]) + 0.005
    capture = rec.get("capture_timestamp")
    delay = (receipt - float(capture)) if capture else float(rec.get("elapsed_s") or 0.0)
    sx = sign_cx * extrinsics.SCALE_CX_TO_BASE_X
    sy = sign_cy * extrinsics.SCALE_CY_TO_BASE_Y
    kw = dict(ref_x=extrinsics.REFERENCE_X_BASE, ref_y=extrinsics.REFERENCE_Y_BASE,
              ref_z=extrinsics.REFERENCE_Z_BASE, scale_x=sx, scale_y=sy,
              y_back_projection=v * delay)
    n = len(rec.get("bounding_boxes") or [])
    ids = list(rec.get("track_ids") or []) + [None] * n
    radii = list(rec.get("center_radius_m") or []) + [None] * n
    conf_flags = list(rec.get("center_confirmed") or []) + [False] * n
    dets = []
    for i, box in enumerate(rec.get("bounding_boxes") or []):
        cam_bbox = as_bbox(box)
        cx, cy, cz = bbox_center(cam_bbox)
        xb = extrinsics.REFERENCE_X_BASE + sx * cx
        yb = extrinsics.REFERENCE_Y_BASE + sy * cy - v * delay
        zb = extrinsics.REFERENCE_Z_BASE
        dets.append({
            "class": rec["class_names"][i],
            "confidence": float(rec["confidences"][i]),
            "cam": [float(cx), float(cy), float(cz)],
            "cam_bbox": cam_bbox.tolist(),
            "base_grasp": [xb, yb, zb + extrinsics.DETECTION_OFFSET_GRASP],
            "base_aim": [xb, yb, zb + extrinsics.DETECTION_OFFSET_AIM],
            "base_bbox_grasp": bbox_to_base(
                cam_bbox, z_offset=extrinsics.DETECTION_OFFSET_GRASP, **kw).tolist(),
            "base_bbox_aim": bbox_to_base(
                cam_bbox, z_offset=extrinsics.DETECTION_OFFSET_AIM, **kw).tolist(),
            "in_workspace": bool(-extrinsics.WORKSPACE_X_ABS < cx < extrinsics.WORKSPACE_X_ABS),
            "src_track_id": ids[i],
            "center_radius": radii[i],
            "center_confirmed": bool(conf_flags[i]),
        })
    return {"receipt_time": receipt, "detections": dets}


def busy_windows(steps_path):
    if not steps_path:
        return []
    out = []
    with open(steps_path, encoding="utf-8") as fh:
        for line in fh:
            s = json.loads(line)
            if s.get("executed") and s.get("t_end") is not None:
                out.append((float(s["t_begin"]), float(s["t_end"])))
    return out


def run(snaps, busy, *, v, mode, cfg):
    after = mode == "after"
    intake = DetectionIntake(
        cfg.OBJECT_MATCH_EPSILON, drift_frac=cfg.OBJECT_MATCH_DRIFT_FRAC,
        eps_y_max=cfg.OBJECT_MATCH_EPS_Y_MAX, merge_eps_y_max=cfg.OBJECT_MERGE_EPS_Y_MAX,
        assoc=cfg.TRACK_ASSOC,
        max_footprint_m=(cfg.OBJECT_MAX_FOOTPRINT_M if after else None),
    )
    queue = TrackedObjectQueue(cfg.MAX_REACH, drop_below_y=-cfg.MAX_REACH)
    if not after:
        queue.src_merge_overlap = float("inf")     # old rule: never across ids
    min_hits = cfg.TRACK_MIN_HITS if after else 1
    t0 = snaps[0]["receipt_time"]
    dist = (lambda t: v * (t - t0))
    log = SimpleNamespace(info=lambda m: None, warn=lambda m: None, warning=lambda m: None,
                          error=lambda m: None, debug=lambda m: None)
    all_tracks = {}
    established_ever = set()
    dup_samples = 0
    dup_pairs = set()
    k = 0
    t = t0
    t_end = snaps[-1]["receipt_time"] + 0.5
    last_sample = -1e9
    while t <= t_end:
        if any(a <= t < b for a, b in busy):
            t += TICK_S
            continue
        arrived = []
        while k < len(snaps) and snaps[k]["receipt_time"] <= t:
            arrived.append(snaps[k])
            k += 1
        frames = arrived if after else arrived[-1:]
        for snap in frames:
            intake.ingest(snap, queue, None, v, log, belt_distance_m=dist(snap["receipt_time"]))
        queue.update(t, v, belt_distance_m=dist(t))
        for o in queue._objects:
            all_tracks[o.track_id] = o
            if is_established(o, min_hits):
                established_ever.add(o.track_id)
        if t - last_sample >= 0.1:
            last_sample = t
            est = [o for o in queue._objects if is_established(o, min_hits)]
            for i in range(len(est)):
                for j in range(i + 1, len(est)):
                    a, b = est[i], est[j]
                    fa = a.footprint_at(t, v, dist(t))
                    fb = b.footprint_at(t, v, dist(t))
                    if footprint_overlap(fa, fb) >= 0.5:
                        dup_samples += 1
                        dup_pairs.add((a.track_id, b.track_id))
        t += TICK_S
    return {
        "tracks_created": len(all_tracks),
        "selectable_tracks": len(established_ever),
        "centre_confirmed": sum(1 for o in all_tracks.values() if o.center_confirmed),
        "selectable_ghosts": None,   # filled by caller (needs producer stats)
        "duplicate_pairs": len(dup_pairs),
        "duplicate_samples": dup_samples,
        "oversize_dropped": intake.n_dropped_oversize,
        "_tracks": all_tracks,
        "_established": established_ever,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="replay_detection_log", description=__doc__.split("\n\n")[0])
    ap.add_argument("camera_log")
    ap.add_argument("--steps", default=None, help="s2r steps.jsonl: busy windows of executed skills")
    ap.add_argument("--belt", type=float, default=0.22, help="belt speed [m/s]")
    ap.add_argument("--sign-cx", type=float, default=extrinsics.SIGN_CX_TO_BASE_X)
    ap.add_argument("--sign-cy", type=float, default=extrinsics.SIGN_CY_TO_BASE_Y)
    args = ap.parse_args(argv)

    recs = load_camera_log(args.camera_log)
    frames_per_id = collections.Counter(
        t for r in recs for t in (r.get("track_ids") or []) if t is not None)
    long_ids = {t for t, n in frames_per_id.items() if n > SHORT_ID_FRAMES}
    short_ids = {t for t, n in frames_per_id.items() if n <= SHORT_ID_FRAMES}
    snaps = [to_snapshot(r, args.belt, args.sign_cx, args.sign_cy) for r in recs]
    snaps = [s for s in snaps if s["detections"] or True]
    busy = busy_windows(args.steps)
    cfg = Config()
    print(f"camera: {len(recs)} frames, producer ids {len(frames_per_id)} "
          f"({len(long_ids)} seen >{SHORT_ID_FRAMES} frames, {len(short_ids)} short/ghost)")
    print(f"loop busy windows: {len(busy)} (total {sum(b - a for a, b in busy):.1f}s)")
    rows = {}
    for mode in ("before", "after"):
        r = run(snaps, busy, v=args.belt, mode=mode, cfg=cfg)
        ghosts = 0
        for tid in r["_established"]:
            o = r["_tracks"][tid]
            srcs = {o.src_track_id, *o.src_aliases} - {None}
            raw = {int(s.split(":")[1]) for s in srcs}
            if raw and raw <= short_ids:
                ghosts += 1
        r["selectable_ghosts"] = ghosts
        rows[mode] = r
    keys = ["tracks_created", "selectable_tracks", "centre_confirmed",
            "selectable_ghosts", "duplicate_pairs", "duplicate_samples", "oversize_dropped"]
    print(f"\n{'':22s}{'before':>10s}{'after':>10s}")
    for k in keys:
        print(f"{k:22s}{rows['before'][k]!s:>10s}{rows['after'][k]!s:>10s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

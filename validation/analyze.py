"""Analyze a recorded run (see ``recorder.py``) for sim-to-real fidelity.

    python -m gp8_control.validation.analyze <run_dir> [--sim-steps N] [--no-replay]

Two questions, answered with the SAME metric code on the hardware run and on a
matched sim reference rollout (recorded by ``sim_reference.py`` with the RL
training settings and the hardware's measured belt speed):

A. Action -> motion. For every commanded motion window: command-to-feedback
   lag, tracking error (joint + EE), motion onset delay, settle error, and the
   EE pose/velocity at each suction release. The hardware's own command stream
   is also replayed through the MuJoCo arm (``replay.py``) so HW and sim are
   compared on the identical action. Plus: decision -> first-motion latency
   (planning compute, which costs zero sim time in training), stream jitter,
   suction IO latency, per-skill step durations.

B. State. Joint-feedback freshness, belt speed stability / encoder agreement,
   detection dead-reckoning error, bbox-vs-position consistency and staleness
   inside the observation, track quality, and per-feature distribution
   overlap between HW observations and the sim reference observations.

Writes ``report.md`` (human summary with PASS/WARN/FAIL), ``report.json``
(all numbers) and ``motion_windows.npz`` (cmd/HW/sim traces per window).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gp8_control.validation.obs_schema import ObsSchema

# ----------------------------------------------------------------------------
# Thresholds: (pass_below, warn_below) on the metric's absolute value.
# ----------------------------------------------------------------------------
THRESHOLDS = {
    "motion.lag_gap_ms": (15.0, 40.0),
    "motion.onset_gap_ms": (20.0, 50.0),
    "motion.replay_ee_rms_mm": (5.0, 15.0),
    "motion.replay_ee_max_p95_mm": (15.0, 40.0),
    "motion.release_pos_gap_mm": (10.0, 25.0),
    "motion.release_vel_gap_mps": (0.05, 0.15),
    "motion.release_io_vel_change_mps": (0.05, 0.15),
    "motion.settle_ee_mm": (3.0, 10.0),
    "timing.stream_jitter_p99_ms": (6.0, 10.0),
    "timing.planning_belt_travel_mm": (5.0, 15.0),
    "timing.suction_ack_p95_ms": (30.0, 80.0),
    "timing.step_duration_gap_frac": (0.10, 0.25),
    "state.joint_age_p95_ms": (10.0, 30.0),
    "state.belt_speed_cv": (0.03, 0.08),
    "state.encoder_vs_reported_speed": (0.02, 0.05),
    "state.object_vs_encoder_scale": (0.03, 0.08),
    "state.dead_reckon_median_mm": (10.0, 25.0),
    "state.dead_reckon_p95_mm": (25.0, 50.0),
    "state.obs_bbox_offset_median_mm": (10.0, 50.0),
    "state.obs_stale_bbox_frac": (0.10, 0.30),
    "state.unknown_class_frac": (1e-9, 0.05),
    "state.duplicate_slot_frac": (0.02, 0.10),
    "state.feature_out_of_support_frac": (0.05, 0.20),
}

MOVE_VEL = 0.05          # rad/s: command counts as "moving"
MERGE_GAP_S = 0.30       # merge moving stretches closer than this
STREAM_GAP_S = 0.10      # a command gap this long ends a stream burst
PAD_PRE_S, PAD_POST_S = 0.05, 0.40
MAX_LAG_S, LAG_STEP_S = 0.25, 0.002
ONSET_RAD = 2e-3
STALE_BBOX_S = 0.25
MIN_FEATURE_SAMPLES = 30


def _status(key: str, value) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "N/A"
    ok, warn = THRESHOLDS[key]
    v = abs(float(value))
    return "PASS" if v < ok else ("WARN" if v < warn else "FAIL")


def _pct(a, q) -> float | None:
    a = np.asarray([x for x in a if x is not None and np.isfinite(x)], dtype=float)
    return float(np.percentile(a, q)) if a.size else None


def _stats(a) -> dict:
    a = np.asarray([x for x in a if x is not None and np.isfinite(x)], dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std()),
            "p5": float(np.percentile(a, 5)), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
            "min": float(a.min()), "max": float(a.max())}


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------
@dataclass
class Run:
    path: Path
    meta: dict
    cmd_t: np.ndarray
    cmd_q: np.ndarray
    fb_t: np.ndarray
    fb_q: np.ndarray
    events: list = field(default_factory=list)
    tracks: list = field(default_factory=list)
    steps: list = field(default_factory=list)

    @property
    def schema(self) -> ObsSchema:
        return ObsSchema(**self.meta.get("schema", {}))

    def events_of(self, kind: str) -> list:
        return [e for e in self.events if e.get("type") == kind]


def _read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            pass   # a torn last line from a killed process
    return out


def load_run(run_dir) -> Run:
    run_dir = Path(run_dir).expanduser()
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    parts = {"cmd_t": [], "cmd_q": [], "fb_t": [], "fb_q": []}
    for f in sorted((run_dir / "stream").glob("chunk_*.npz")):
        with np.load(f) as d:
            for k in parts:
                parts[k].append(d[k])

    def cat(k, shape):
        return np.concatenate(parts[k]) if parts[k] else np.zeros(shape)

    cmd_t, cmd_q = cat("cmd_t", (0,)), cat("cmd_q", (0, 6))
    fb_t, fb_q = cat("fb_t", (0,)), cat("fb_q", (0, 6))
    o = np.argsort(cmd_t, kind="stable")
    cmd_t, cmd_q = cmd_t[o], cmd_q[o]
    o = np.argsort(fb_t, kind="stable")
    fb_t, fb_q = fb_t[o], fb_q[o]
    # Drop duplicate feedback stamps (sim batches can re-publish a state).
    if fb_t.size > 1:
        keep = np.concatenate([[True], np.diff(fb_t) > 1e-9])
        fb_t, fb_q = fb_t[keep], fb_q[keep]
    return Run(
        path=run_dir, meta=meta, cmd_t=cmd_t, cmd_q=cmd_q, fb_t=fb_t, fb_q=fb_q,
        events=_read_jsonl(run_dir / "events.jsonl"),
        tracks=_read_jsonl(run_dir / "tracks.jsonl"),
        steps=_read_jsonl(run_dir / "steps.jsonl"),
    )


# ----------------------------------------------------------------------------
# signal helpers
# ----------------------------------------------------------------------------
def _zoh(t_src, q_src, t):
    idx = np.clip(np.searchsorted(t_src, t, side="right") - 1, 0, len(t_src) - 1)
    return q_src[idx]


def _interp(t_src, q_src, t):
    return np.column_stack([np.interp(t, t_src, q_src[:, j]) for j in range(q_src.shape[1])])


class _FK:
    def __init__(self) -> None:
        from gp8_control.robots.gp8 import GP8

        self.robot = GP8()

    def pos(self, q: np.ndarray) -> np.ndarray:
        q = np.atleast_2d(q)
        return np.array([self.robot.forward_kinematics(row)[:3, 3] for row in q])


def motion_windows(run: Run) -> list[dict]:
    t, q = run.cmd_t, run.cmd_q
    if t.size < 3:
        return []
    dt = np.diff(t)
    vel = np.zeros(t.size)
    good = dt > 1e-6
    vel[1:][good] = np.max(np.abs(np.diff(q, axis=0)[good]) / dt[good, None], axis=1)
    moving = vel > MOVE_VEL
    windows = []
    i = 0
    n = t.size
    while i < n:
        if not moving[i]:
            i += 1
            continue
        j = i
        while j + 1 < n:
            if t[j + 1] - t[j] > STREAM_GAP_S:
                break
            if moving[j + 1]:
                j += 1
                continue
            # look ahead for a short pause
            k = j + 1
            while k < n and not moving[k] and t[k] - t[j] <= MERGE_GAP_S \
                    and t[k] - t[k - 1] <= STREAM_GAP_S:
                k += 1
            if k < n and moving[k] and t[k] - t[j] <= MERGE_GAP_S:
                j = k
            else:
                break
        ta, tb = t[max(i - 1, 0)], t[j]
        amp = float(np.max(np.ptp(q[max(i - 1, 0):j + 1], axis=0)))
        if tb - ta >= 0.08 and amp >= 0.01:
            # extend through the trailing hold (same burst) for settle/lag
            k = j
            while k + 1 < n and t[k + 1] - t[k] <= STREAM_GAP_S and t[k + 1] <= tb + PAD_POST_S:
                k += 1
            windows.append({"t0": float(ta - PAD_PRE_S), "t_move_end": float(tb),
                            "t1": float(max(t[k], tb) + 0.05), "amp_rad": amp,
                            "peak_vel": float(vel[i:j + 1].max())})
        i = j + 1
    labels = sorted(((e["t"], e["label"]) for e in run.events_of("motion_op")))
    for w in windows:
        lab = [lb for (tt, lb) in labels if tt <= w["t0"] + PAD_PRE_S]
        w["label"] = lab[-1] if lab else "unlabeled"
    return windows


def tracking_metrics(cmd_t, cmd_q, y_t, y_q, w, fk: _FK | None) -> dict | None:
    """How a measured/simulated joint trace (y) follows the ZOH command."""
    t0, t1 = w["t0"], w["t1"]
    if y_t.size < 4 or y_t[0] > t0 + 0.02 or y_t[-1] < w["t_move_end"]:
        return None
    t1 = min(t1, float(y_t[-1]))
    grid = np.arange(t0, t1, 0.002)
    if grid.size < 20:
        return None
    y = _interp(y_t, y_q, grid)
    best = (None, float("inf"))
    rms0 = None
    for tau in np.arange(0.0, MAX_LAG_S + 1e-9, LAG_STEP_S):
        c = _zoh(cmd_t, cmd_q, grid - tau)
        r = float(np.sqrt(np.mean((y - c) ** 2)))
        if tau == 0.0:
            rms0 = r
        if r < best[1]:
            best = (float(tau), r)
    lag, rms_best = best
    c_best = _zoh(cmd_t, cmd_q, grid - lag)
    c0 = _zoh(cmd_t, cmd_q, grid)
    out = {
        "lag_ms": lag * 1e3,
        "rms_rad_0": rms0,
        "rms_rad_lag": rms_best,
        "max_rad_0": float(np.max(np.abs(y - c0))),
        "max_rad_lag": float(np.max(np.abs(y - c_best))),
    }
    # onset: first departure of command vs of the trace from the window start
    c_dep = np.flatnonzero(np.max(np.abs(c0 - c0[0]), axis=1) > ONSET_RAD)
    y_dep = np.flatnonzero(np.max(np.abs(y - y[0]), axis=1) > ONSET_RAD)
    if c_dep.size and y_dep.size:
        out["onset_ms"] = float((grid[y_dep[0]] - grid[c_dep[0]]) * 1e3)
    settle_t = min(w["t1"], float(y_t[-1]))
    q_settle = _interp(y_t, y_q, np.array([settle_t]))[0]
    q_final = _zoh(cmd_t, cmd_q, np.array([settle_t]))[0]
    out["settle_rad"] = float(np.max(np.abs(q_settle - q_final)))
    if fk is not None:
        sub = slice(None, None, 5)                  # 10 ms
        e_y = fk.pos(y[sub])
        e_c0 = fk.pos(c0[sub])
        e_cl = fk.pos(c_best[sub])
        d0 = np.linalg.norm(e_y - e_c0, axis=1) * 1e3
        dl = np.linalg.norm(e_y - e_cl, axis=1) * 1e3
        out.update(ee_rms_mm_0=float(np.sqrt(np.mean(d0 ** 2))), ee_max_mm_0=float(d0.max()),
                   ee_rms_mm_lag=float(np.sqrt(np.mean(dl ** 2))), ee_max_mm_lag=float(dl.max()))
        out["settle_ee_mm"] = float(np.linalg.norm(
            fk.pos(q_settle)[0] - fk.pos(q_final)[0]) * 1e3)
    return out


def _ee_state(fk: _FK, t_src, q_src, t, *, zoh=False, h=0.01):
    f = _zoh if zoh else _interp
    qs = f(t_src, q_src, np.array([t - h, t, t + h]))
    p = fk.pos(qs)
    return p[1], (p[2] - p[0]) / (2 * h)


# ----------------------------------------------------------------------------
# A. action -> motion
# ----------------------------------------------------------------------------
def analyze_motion(run: Run, *, replay: bool = True) -> dict:
    fk = _FK()
    windows = motion_windows(run)
    res: dict = {"n_windows": len(windows), "windows": []}
    replayer = None
    if replay and windows:
        from gp8_control.validation.replay import ArmReplayer, available

        ok, why = available()
        if ok:
            replayer = ArmReplayer()
            res["replay"] = {"kp": replayer.kp, "kv": replayer.kv, "dt": replayer.dt}
        else:
            res["replay_skipped"] = why
    releases = [e for e in run.events_of("suction") if not e.get("on")]
    acks = {round(e["requested_at"], 6): e for e in run.events_of("suction_ack")}
    traces = {}
    for wi, w in enumerate(windows):
        row = dict(w)
        hw = tracking_metrics(run.cmd_t, run.cmd_q, run.fb_t, run.fb_q, w, fk)
        row["hw"] = hw
        sim_t = sim_q = None
        if replayer is not None and run.fb_t.size >= 2:
            q0 = _interp(run.fb_t, run.fb_q, np.array([w["t0"]]))[0]
            q0p = _interp(run.fb_t, run.fb_q, np.array([w["t0"] - 0.02]))[0]
            qd0 = (q0 - q0p) / 0.02
            hold = _zoh(run.cmd_t, run.cmd_q, np.array([w["t0"]]))[0]
            sim_t, sim_q = replayer.replay(run.cmd_t, run.cmd_q, t0=w["t0"], t1=w["t1"],
                                           q0=q0, qd0=qd0, hold_q=hold)
            row["sim"] = tracking_metrics(run.cmd_t, run.cmd_q, sim_t, sim_q, w, fk)
            # direct HW vs sim on the identical command stream
            t_hi = min(w["t1"], float(run.fb_t[-1]))
            grid = np.arange(w["t0"], t_hi, 0.01)
            if grid.size >= 5:
                qh = _interp(run.fb_t, run.fb_q, grid)
                qs = _interp(sim_t, sim_q, grid)
                d = np.linalg.norm(fk.pos(qh) - fk.pos(qs), axis=1) * 1e3
                row["hw_vs_sim"] = {
                    "joint_rms_rad": float(np.sqrt(np.mean((qh - qs) ** 2))),
                    "joint_max_rad": float(np.max(np.abs(qh - qs))),
                    "ee_rms_mm": float(np.sqrt(np.mean(d ** 2))),
                    "ee_max_mm": float(d.max()),
                }
        # suction releases inside the window (the throw moment)
        rel_rows = []
        for e in releases:
            tr = float(e["t"])
            if not (w["t0"] <= tr <= w["t1"]):
                continue
            rr = {"t": tr}
            p_c, v_c = _ee_state(fk, run.cmd_t, run.cmd_q, tr, zoh=True)
            rr["cmd_pos"], rr["cmd_vel"] = p_c.tolist(), v_c.tolist()
            if run.fb_t.size >= 2 and run.fb_t[0] <= tr - 0.01 and run.fb_t[-1] >= tr + 0.01:
                p_h, v_h = _ee_state(fk, run.fb_t, run.fb_q, tr)
                rr["hw_pos"], rr["hw_vel"] = p_h.tolist(), v_h.tolist()
                rr["hw_speed"] = float(np.linalg.norm(v_h))
                ack = acks.get(round(tr, 6))
                if ack is not None:
                    ta = float(ack["acked_at"])
                    rr["io_ms"] = (ta - tr) * 1e3
                    if run.fb_t[-1] >= ta + 0.01:
                        p_a, v_a = _ee_state(fk, run.fb_t, run.fb_q, ta)
                        rr["io_pos_shift_mm"] = float(np.linalg.norm(p_a - p_h) * 1e3)
                        rr["io_vel_change_mps"] = float(np.linalg.norm(v_a - v_h))
                if sim_t is not None and sim_t[0] <= tr - 0.01 and sim_t[-1] >= tr + 0.01:
                    p_s, v_s = _ee_state(fk, sim_t, sim_q, tr)
                    rr["sim_pos"], rr["sim_vel"] = p_s.tolist(), v_s.tolist()
                    rr["sim_speed"] = float(np.linalg.norm(v_s))
                    rr["pos_gap_mm"] = float(np.linalg.norm(p_h - p_s) * 1e3)
                    rr["vel_gap_mps"] = float(np.linalg.norm(v_h - v_s))
            rel_rows.append(rr)
        row["releases"] = rel_rows
        res["windows"].append(row)
        g = np.arange(w["t0"], w["t1"], 0.004)
        traces[f"w{wi}_t"] = g
        traces[f"w{wi}_cmd"] = _zoh(run.cmd_t, run.cmd_q, g)
        if run.fb_t.size >= 2:
            traces[f"w{wi}_hw"] = _interp(run.fb_t, run.fb_q, g)
        if sim_t is not None:
            traces[f"w{wi}_sim"] = _interp(sim_t, sim_q, g)
    if replayer is not None:
        replayer.close()
    if traces:
        np.savez_compressed(run.path / "motion_windows.npz", **traces)

    # aggregates
    def agg(path):
        vals = []
        for r in res["windows"]:
            cur = r
            for p in path:
                cur = cur.get(p) if isinstance(cur, dict) else None
                if cur is None:
                    break
            if cur is not None:
                vals.append(cur)
        return _stats(vals)

    res["summary"] = {
        "hw_lag_ms": agg(("hw", "lag_ms")),
        "sim_lag_ms": agg(("sim", "lag_ms")),
        "hw_onset_ms": agg(("hw", "onset_ms")),
        "sim_onset_ms": agg(("sim", "onset_ms")),
        "hw_ee_rms_mm_0": agg(("hw", "ee_rms_mm_0")),
        "sim_ee_rms_mm_0": agg(("sim", "ee_rms_mm_0")),
        "hw_settle_ee_mm": agg(("hw", "settle_ee_mm")),
        "sim_settle_ee_mm": agg(("sim", "settle_ee_mm")),
        "hw_vs_sim_ee_rms_mm": agg(("hw_vs_sim", "ee_rms_mm")),
        "hw_vs_sim_ee_max_mm": agg(("hw_vs_sim", "ee_max_mm")),
    }
    per_label = defaultdict(list)
    for r in res["windows"]:
        per_label[r["label"]].append(r)
    res["by_label"] = {}
    for lab, rows in per_label.items():
        res["by_label"][lab] = {
            "n": len(rows),
            "hw_lag_ms_p50": _pct([(r.get("hw") or {}).get("lag_ms") for r in rows], 50),
            "sim_lag_ms_p50": _pct([(r.get("sim") or {}).get("lag_ms") for r in rows], 50),
            "hw_vs_sim_ee_rms_mm_p50": _pct(
                [(r.get("hw_vs_sim") or {}).get("ee_rms_mm") for r in rows], 50),
            "hw_vs_sim_ee_max_mm_p95": _pct(
                [(r.get("hw_vs_sim") or {}).get("ee_max_mm") for r in rows], 95),
            "peak_vel_p50": _pct([r["peak_vel"] for r in rows], 50),
        }
    rel = [rr for r in res["windows"] for rr in r["releases"]]
    res["releases"] = {
        "n": len(rel),
        "pos_gap_mm": _stats([r.get("pos_gap_mm") for r in rel]),
        "vel_gap_mps": _stats([r.get("vel_gap_mps") for r in rel]),
        "hw_speed": _stats([r.get("hw_speed") for r in rel]),
        "sim_speed": _stats([r.get("sim_speed") for r in rel]),
        "io_ms": _stats([r.get("io_ms") for r in rel]),
        "io_vel_change_mps": _stats([r.get("io_vel_change_mps") for r in rel]),
        "io_pos_shift_mm": _stats([r.get("io_pos_shift_mm") for r in rel]),
    }
    return res


def analyze_timing(run: Run) -> dict:
    res: dict = {}
    # stream pacing (inside bursts only)
    if run.cmd_t.size > 2:
        d = np.diff(run.cmd_t)
        d = d[d < STREAM_GAP_S]
        res["stream_dt_ms"] = _stats(d * 1e3)
        res["stream_late_frac"] = float(np.mean(d > 0.006)) if d.size else None
        res["stream_gaps_over_20ms"] = int(np.sum(d > 0.020))
    # suction IO
    acks = run.events_of("suction_ack")
    res["suction_ack_ms"] = _stats([(a["acked_at"] - a["requested_at"]) * 1e3 for a in acks])
    res["suction_off_ack_ms"] = _stats(
        [(a["acked_at"] - a["requested_at"]) * 1e3 for a in acks if not a["on"]])
    # decision -> first new motion command / first measured motion
    plan, plan_fb, durations = [], [], defaultdict(list)
    belt = run.events_of("belt")
    v_med = _pct([b["speed"] for b in belt], 50) or 0.0
    for s in run.steps:
        ex = s.get("executed")
        if not ex:
            continue
        tb, te = s["t_begin"], s.get("t_end")
        if te is not None:
            durations[ex.get("skill")].append(te - tb)
        if run.cmd_t.size == 0:
            continue
        hold = _zoh(run.cmd_t, run.cmd_q, np.array([tb]))[0]
        k0 = int(np.searchsorted(run.cmd_t, tb, side="right"))
        k1 = int(np.searchsorted(run.cmd_t, te if te is not None else tb + 10.0, side="right"))
        seg = run.cmd_q[k0:k1]
        dep = np.flatnonzero(np.max(np.abs(seg - hold), axis=1) > 1e-3) if seg.size else []
        if len(dep):
            t_cmd = float(run.cmd_t[k0 + dep[0]])
            plan.append(t_cmd - tb)
            if run.fb_t.size:
                q_fb0 = _interp(run.fb_t, run.fb_q, np.array([t_cmd]))[0]
                j0 = int(np.searchsorted(run.fb_t, t_cmd, side="right"))
                fseg = run.fb_q[j0:]
                fdep = np.flatnonzero(np.max(np.abs(fseg - q_fb0), axis=1) > ONSET_RAD)
                if fdep.size:
                    plan_fb.append(float(run.fb_t[j0 + fdep[0]]) - tb)
    res["decision_to_motion_cmd_ms"] = _stats(np.asarray(plan) * 1e3)
    res["decision_to_measured_motion_ms"] = _stats(np.asarray(plan_fb) * 1e3)
    p50 = _pct(plan, 50)
    res["belt_speed_median"] = v_med
    res["planning_belt_travel_mm"] = None if p50 is None else p50 * v_med * 1e3
    res["step_duration_s"] = {k: _stats(v) for k, v in durations.items()}
    res["n_executed_steps"] = int(sum(len(v) for v in durations.values()))
    return res


# ----------------------------------------------------------------------------
# B. state
# ----------------------------------------------------------------------------
def _poly_area(pts) -> float:
    p = np.asarray(pts, dtype=float)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _bbox_geom(pts) -> dict | None:
    if not pts or len(pts) != 4:
        return None
    p = np.asarray(pts, dtype=float)[:, :2]
    if not np.all(np.isfinite(p)) or np.allclose(p, 0.0):
        return None
    e01 = np.linalg.norm(p[1] - p[0]) / 2.0
    e12 = np.linalg.norm(p[2] - p[1]) / 2.0
    yaw = math.degrees(math.atan2(p[1, 1] - p[0, 1], p[1, 0] - p[0, 0])) % 90.0
    return {"cx": float(p[:, 0].mean()), "cy": float(p[:, 1].mean()),
            "half_small": float(min(e01, e12)), "half_large": float(max(e01, e12)),
            "yaw_mod90": yaw, "area_sign": float(np.sign(_poly_area(p)))}


def analyze_state(run: Run) -> dict:
    from gp8_control.rl.common import CLASS_IDS

    res: dict = {}
    # joint feedback freshness
    if run.fb_t.size > 2:
        d = np.diff(run.fb_t)
        res["fb_rate_hz"] = float(1.0 / np.median(d))
        res["fb_gap_ms"] = _stats(d * 1e3)
    res["joint_age_ms"] = _stats(
        [s["joint_age"] * 1e3 for s in run.steps if s.get("joint_age") is not None])

    # belt
    belt = run.events_of("belt")
    sp = np.array([b["speed"] for b in belt], dtype=float)
    if sp.size:
        res["belt_speed"] = _stats(sp)
        res["belt_speed_cv"] = float(sp.std() / max(abs(sp.mean()), 1e-9))
        res["belt_msg_age_s"] = _stats([b.get("msg_age") for b in belt])
    bt = np.array([b["t"] for b in belt if b.get("distance_m") is not None], dtype=float)
    bd = np.array([b["distance_m"] for b in belt if b.get("distance_m") is not None], dtype=float)
    if bt.size > 10 and bt[-1] - bt[0] > 2.0:
        enc_v = []
        k = 0
        for i in range(bt.size):
            while k < bt.size and bt[k] < bt[i] + 1.0:
                k += 1
            if k < bt.size:
                enc_v.append((bd[k] - bd[i]) / (bt[k] - bt[i]))
        enc_v = np.asarray(enc_v)
        res["encoder_speed"] = _stats(enc_v)
        if sp.size and enc_v.size:
            res["encoder_vs_reported_speed"] = float(np.median(enc_v) / max(np.median(sp), 1e-9) - 1.0)

    # perception frames
    frames = run.events_of("frame")
    if len(frames) > 2:
        rt = np.array([f["receipt_time"] for f in frames], dtype=float)
        res["frame_rate_hz"] = float(1.0 / np.median(np.diff(rt))) if np.median(np.diff(rt)) > 0 else None
        res["perception_delay_s"] = _stats([f.get("perception_delay_s") for f in frames])
        res["frame_classes"] = sorted({c for f in frames for c in f.get("classes", [])})

    # tracks
    by_track = defaultdict(list)
    gone = {}
    for tr in run.tracks:
        if tr.get("gone"):
            gone[tr["track_id"]] = tr["t_poll"]
            continue
        by_track[tr["track_id"]].append(tr)
    residuals, slopes, x_jit, z_vals, confs = [], [], [], [], []
    halves_s, halves_l, yaws, signs, flips, lifetimes, n_det = [], [], [], [], [], [], []
    unknown = 0
    total_upd = 0
    for tid, ups in by_track.items():
        ups.sort(key=lambda r: r["t_bbox"])
        n_det.append(len(ups))
        if tid in gone:
            lifetimes.append(gone[tid] - ups[0]["t_bbox"])
        geoms = [_bbox_geom(u.get("bbox")) for u in ups]
        first_g = next((g for g in geoms if g is not None), None)
        a0 = ups[0]
        off0 = None if first_g is None else first_g["cy"] - a0["anchor_xyz"][1]
        cls_seq = [u["class_name"] for u in ups]
        flips.append(sum(1 for a, b in zip(cls_seq, cls_seq[1:]) if a != b))
        ys, encs, xs = [], [], []
        for u, g in zip(ups, geoms):
            total_upd += 1
            if u["class_name"] not in CLASS_IDS:
                unknown += 1
            confs.append(u["conf"])
            z_vals.append(u["anchor_xyz"][2])
            if g is None:
                continue
            halves_s.append(g["half_small"])
            halves_l.append(g["half_large"])
            yaws.append(g["yaw_mod90"])
            signs.append(g["area_sign"])
            xs.append(g["cx"])
            if u.get("enc_bbox") is not None:
                ys.append(g["cy"])
                encs.append(u["enc_bbox"])
            if (not u["first"] and off0 is not None and u.get("enc_bbox") is not None
                    and u.get("anchor_enc") is not None):
                pred = u["anchor_xyz"][1] - (u["enc_bbox"] - u["anchor_enc"])
                residuals.append((g["cy"] - off0) - pred)
        if len(xs) >= 3:
            x_jit.append(float(np.std(xs)))
        if len(ys) >= 3 and (max(encs) - min(encs)) > 0.05:
            slopes.append(float(np.polyfit(encs, ys, 1)[0]))
    res["tracks"] = {
        "n_tracks": len(by_track),
        "detections_per_track": _stats(n_det),
        "lifetime_s": _stats(lifetimes),
        "class_flips_per_track": _stats(flips),
        "unknown_class_frac": (unknown / total_upd) if total_upd else None,
        "conf": _stats(confs),
        "z": _stats(z_vals),
        "x_jitter_mm": _stats(np.asarray(x_jit) * 1e3),
        "dead_reckon_residual_mm": _stats(np.asarray(residuals) * 1e3),
        "dead_reckon_abs_p95_mm": _pct(np.abs(np.asarray(residuals)) * 1e3, 95),
        "y_vs_encoder_slope": _stats(slopes),
        "bbox_half_small_m": _stats(halves_s),
        "bbox_half_large_m": _stats(halves_l),
        "bbox_yaw_mod90_deg": _stats(yaws),
        "bbox_axis_aligned_frac": (float(np.mean([min(y, 90 - y) < 3.0 for y in yaws]))
                                   if yaws else None),
        "bbox_ccw_frac": float(np.mean(np.asarray(signs) > 0)) if signs else None,
    }
    if slopes:
        res["tracks"]["object_vs_encoder_scale"] = float(-np.median(slopes) - 1.0)
    res["obs"] = analyze_obs(run)
    return res


def obs_features(run: Run) -> tuple[dict, dict]:
    """Per-feature samples from every recorded observation (+ slot metadata)."""
    schema = run.schema
    feats = defaultdict(list)
    meta = defaultdict(list)
    for s in run.steps:
        obs = s.get("obs")
        if obs is None or len(obs) != schema.width:
            continue
        dec = schema.decode(obs)
        g = dec["globals"]
        for k, v in g.items():
            feats[k].append(v)
        live = [o for o in dec["objects"] if o is not None]
        feats["n_live_slots"].append(len(live))
        slots = s.get("slots") or []
        for si, o in enumerate(dec["objects"]):
            if o is None:
                continue
            for k in ("x", "y", "z", "class_id", "conf"):
                feats[f"obj.{k}"].append(o[k])
            if "c0x" in o:
                pts = [(o[f"c{i}x"], o[f"c{i}y"]) for i in range(4)]
                for i in range(4):
                    feats["obj.corner_x"].append(o[f"c{i}x"])
                    feats["obj.corner_y"].append(o[f"c{i}y"])
                gm = _bbox_geom(pts)
                if gm is not None:
                    feats["obj.bbox_cx_minus_x"].append(gm["cx"] - o["x"])
                    feats["obj.bbox_cy_minus_y"].append(gm["cy"] - o["y"])
                    feats["obj.bbox_half_small"].append(gm["half_small"])
                    feats["obj.bbox_half_large"].append(gm["half_large"])
                    feats["obj.bbox_yaw_mod90"].append(gm["yaw_mod90"])
            if si < len(slots):
                sl = slots[si]
                if sl.get("bbox_detect_time") is not None:
                    meta["bbox_age_s"].append(s["t_begin"] - sl["bbox_detect_time"])
        # duplicate tracks: two live slots on top of each other
        dup = 0
        for i in range(len(live)):
            for j in range(i + 1, len(live)):
                if abs(live[i]["x"] - live[j]["x"]) < 0.04 and abs(live[i]["y"] - live[j]["y"]) < 0.04:
                    dup += 1
        meta["dup_pairs"].append(dup)
    return feats, meta


def analyze_obs(run: Run) -> dict:
    feats, meta = obs_features(run)
    res = {"n_obs": len(feats.get("q1", []))}
    dy = feats.get("obj.bbox_cy_minus_y", [])
    dx = feats.get("obj.bbox_cx_minus_x", [])
    res["bbox_offset_y_mm"] = _stats(np.asarray(dy) * 1e3)
    res["bbox_offset_x_mm"] = _stats(np.asarray(dx) * 1e3)
    res["bbox_offset_median_mm"] = (
        float(np.median(np.hypot(np.asarray(dx), np.asarray(dy))) * 1e3) if dy else None)
    ages = meta.get("bbox_age_s", [])
    res["bbox_age_s"] = _stats(ages)
    res["stale_bbox_frac"] = float(np.mean(np.asarray(ages) > STALE_BBOX_S)) if ages else None
    dups = meta.get("dup_pairs", [])
    res["duplicate_slot_frac"] = float(np.mean(np.asarray(dups) > 0)) if dups else None
    rt = feats.get("remaining_time_frac", [])
    res["remaining_time_frac"] = _stats(rt)
    return res


def _ks(a, b) -> float:
    a, b = np.sort(a), np.sort(b)
    allv = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, allv, side="right") / a.size
    cdf_b = np.searchsorted(b, allv, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def compare_features(hw: Run, sim: Run) -> dict:
    fh, _ = obs_features(hw)
    fs, _ = obs_features(sim)
    out = {}
    for k in sorted(set(fh) | set(fs)):
        a = np.asarray([v for v in fh.get(k, []) if np.isfinite(v)], dtype=float)
        b = np.asarray([v for v in fs.get(k, []) if np.isfinite(v)], dtype=float)
        row = {"hw": _stats(a), "sim": _stats(b)}
        if a.size and b.size:
            lo, hi = np.percentile(b, 0.5), np.percentile(b, 99.5)
            margin = 0.05 * (hi - lo) + 1e-6
            row["out_of_support_frac"] = float(np.mean((a < lo - margin) | (a > hi + margin)))
            row["ks"] = _ks(a, b)
            row["hw_constant"] = bool(np.ptp(a) < 1e-9)
            row["sim_constant"] = bool(np.ptp(b) < 1e-9)
        out[k] = row
    return out


# ----------------------------------------------------------------------------
# verdicts + report
# ----------------------------------------------------------------------------
def _g(d, *path):
    for p in path:
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d


def build_verdicts(r: dict) -> list[dict]:
    hw_m, hw_t, hw_s = r["hw"]["motion"], r["hw"]["timing"], r["hw"]["state"]
    ref = r.get("sim_reference") or {}
    sim_t = ref.get("timing") or {}
    sim_s = ref.get("state") or {}
    rows = []

    def add(group, key, name, value, unit, note=""):
        rows.append({"group": group, "key": key, "name": name, "value": value,
                     "unit": unit, "status": _status(key, value), "note": note})

    def gap(a, b):
        return None if a is None or b is None else a - b

    ms = hw_m.get("summary", {})
    add("action", "motion.lag_gap_ms", "명령→추종 지연 차이 (HW − sim replay, p50)",
        gap(_g(ms, "hw_lag_ms", "p50"), _g(ms, "sim_lag_ms", "p50")), "ms",
        f"HW {_fmt(_g(ms, 'hw_lag_ms', 'p50'))} / sim {_fmt(_g(ms, 'sim_lag_ms', 'p50'))} ms")
    add("action", "motion.onset_gap_ms", "모션 시작 지연 차이 (HW − sim, p50)",
        gap(_g(ms, "hw_onset_ms", "p50"), _g(ms, "sim_onset_ms", "p50")), "ms")
    add("action", "motion.replay_ee_rms_mm", "동일 명령 HW vs sim EE 궤적 RMS (p50)",
        _g(ms, "hw_vs_sim_ee_rms_mm", "p50"), "mm")
    add("action", "motion.replay_ee_max_p95_mm", "동일 명령 HW vs sim EE 최대 오차 (p95)",
        _g(ms, "hw_vs_sim_ee_max_mm", "p95"), "mm")
    rel = hw_m.get("releases", {})
    add("action", "motion.release_pos_gap_mm", "릴리즈 순간 EE 위치 HW vs sim (p50)",
        _g(rel, "pos_gap_mm", "p50"), "mm", f"n={rel.get('n', 0)}")
    add("action", "motion.release_vel_gap_mps", "릴리즈 순간 EE 속도 HW vs sim (p50)",
        _g(rel, "vel_gap_mps", "p50"), "m/s")
    add("action", "motion.release_io_vel_change_mps",
        "석션 IO 지연 동안 EE 속도 변화 (sim은 0, p50)",
        _g(rel, "io_vel_change_mps", "p50"), "m/s",
        f"IO {_fmt(_g(rel, 'io_ms', 'p50'))} ms")
    add("action", "motion.settle_ee_mm", "모션 종료 후 정착 오차 (HW, p50)",
        _g(ms, "hw_settle_ee_mm", "p50"), "mm",
        f"sim {_fmt(_g(ms, 'sim_settle_ee_mm', 'p50'))} mm")
    add("action", "timing.stream_jitter_p99_ms", "250 Hz 스트림 간격 p99 (sim은 4.0)",
        _g(hw_t, "stream_dt_ms", "p99"), "ms",
        f"20 ms 초과 gap {hw_t.get('stream_gaps_over_20ms')}회")
    add("action", "timing.planning_belt_travel_mm",
        "결정→첫 모션 명령 지연 동안 벨트 이동 (sim 학습에선 0)",
        hw_t.get("planning_belt_travel_mm"), "mm",
        f"지연 p50 {_fmt(_g(hw_t, 'decision_to_motion_cmd_ms', 'p50'))} ms")
    add("action", "timing.suction_ack_p95_ms", "석션 IO 요청→ack (p95, sim은 0)",
        _g(hw_t, "suction_ack_ms", "p95"), "ms")
    for skill, st in (hw_t.get("step_duration_s") or {}).items():
        s_ref = _g(sim_t, "step_duration_s", skill, "p50")
        h = st.get("p50")
        val = None if (s_ref in (None, 0) or h is None) else (h - s_ref) / s_ref
        add("action", "timing.step_duration_gap_frac", f"스킬 1회 소요시간 차이 [{skill}]",
            val, "×", f"HW {_fmt(h)} s / sim {_fmt(s_ref)} s")

    add("state", "state.joint_age_p95_ms", "결정 시점 joint state 나이 (p95)",
        _g(hw_s, "joint_age_ms", "p95"), "ms", f"fb {_fmt(hw_s.get('fb_rate_hz'))} Hz")
    add("state", "state.belt_speed_cv", "벨트 속도 변동계수 (sim은 상수)",
        hw_s.get("belt_speed_cv"), "")
    add("state", "state.encoder_vs_reported_speed", "엔코더 적분속도 vs /conveyor/speed",
        hw_s.get("encoder_vs_reported_speed"), "")
    tr = hw_s.get("tracks", {})
    add("state", "state.object_vs_encoder_scale", "물체 실제 이동 vs 엔코더 거리 스케일 오차",
        tr.get("object_vs_encoder_scale"), "", "sim은 정확히 0 (Y를 엔코더로 구동)")
    add("state", "state.dead_reckon_median_mm", "추측항법 y 오차 (중앙값)",
        _g(tr, "dead_reckon_residual_mm", "p50"), "mm")
    add("state", "state.dead_reckon_p95_mm", "추측항법 y |오차| p95",
        tr.get("dead_reckon_abs_p95_mm"), "mm")
    ob = hw_s.get("obs", {})
    add("state", "state.obs_bbox_offset_median_mm", "obs 안 bbox 중심 vs xyz_now 거리 (중앙값)",
        ob.get("bbox_offset_median_mm"), "mm",
        f"sim {_fmt(_g(sim_s, 'obs', 'bbox_offset_median_mm'))} mm")
    add("state", "state.obs_stale_bbox_frac", f"obs bbox가 {STALE_BBOX_S}s 이상 묵은 슬롯 비율",
        ob.get("stale_bbox_frac"), "",
        f"sim {_fmt(_g(sim_s, 'obs', 'stale_bbox_frac'))}")
    add("state", "state.unknown_class_frac", "CLASS_IDS에 없는 클래스 비율 (obs class_id=-1)",
        tr.get("unknown_class_frac"), "")
    add("state", "state.duplicate_slot_frac", "중복 트랙(4 cm 내 두 슬롯)이 있는 스텝 비율",
        ob.get("duplicate_slot_frac"), "", f"sim {_fmt(_g(sim_s, 'obs', 'duplicate_slot_frac'))}")
    for k, row in (r.get("features") or {}).items():
        if row.get("out_of_support_frac") is None:
            continue
        n_hw, n_sim = _g(row, "hw", "n") or 0, _g(row, "sim", "n") or 0
        small = min(n_hw, n_sim) < MIN_FEATURE_SAMPLES
        add("feature", "state.feature_out_of_support_frac", f"obs `{k}` 가 sim 분포 밖",
            None if small else row["out_of_support_frac"], "",
            (f"표본 부족 (HW {n_hw}, sim {n_sim}) · " if small else "") +
            f"HW p50 {_fmt(_g(row, 'hw', 'p50'))} / sim p50 {_fmt(_g(row, 'sim', 'p50'))}, "
            f"KS {row.get('ks', 0):.2f}"
            + (" · HW 상수" if row.get("hw_constant") and not row.get("sim_constant") else ""))
    return rows


def _fmt(v, nd=3) -> str:
    if v is None:
        return "—"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(v):
        return "—"
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.{nd}g}"


def write_report(run_dir: Path, r: dict) -> Path:
    verdicts = r["verdicts"]
    counts = defaultdict(int)
    for v in verdicts:
        counts[v["status"]] += 1
    m = r["hw"]["meta"]
    lines = [
        "# Sim-to-real validation report",
        "",
        f"- run: `{run_dir}`",
        f"- source: `{m.get('source')}` · backend `{m.get('backend')}` · git `{(m.get('git_rev') or '')[:10]}`",
        f"- duration: {_fmt((m.get('t_end') or 0) - (m.get('t_start') or 0))} s · "
        f"executed steps {r['hw']['timing'].get('n_executed_steps')} · "
        f"motion windows {r['hw']['motion'].get('n_windows')}",
        f"- obs schema: `{m.get('schema')}`",
        f"- sim reference: {r.get('sim_reference_path') or '없음 (' + str(r.get('sim_reference_skipped')) + ')'}",
        f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"**PASS {counts['PASS']} · WARN {counts['WARN']} · FAIL {counts['FAIL']} · N/A {counts['N/A']}**",
        "",
    ]
    for group, title in (("action", "## A. Action → 로봇 움직임"),
                         ("state", "## B. State (관측값) 채널"),
                         ("feature", "## C. Observation feature 분포 (HW vs sim reference)")):
        rows = [v for v in verdicts if v["group"] == group]
        if group == "feature":
            rows.sort(key=lambda v: -(v["value"] or 0))
        lines += [title, "", "| 상태 | 항목 | 값 | 비고 |", "|---|---|---|---|"]
        for v in rows:
            val = _fmt(v["value"]) + (f" {v['unit']}" if v["unit"] and v["value"] is not None else "")
            lines.append(f"| {v['status']} | {v['name']} | {val} | {v['note']} |")
        lines.append("")

    hw_m = r["hw"]["motion"]
    if hw_m.get("by_label"):
        lines += ["### 스킬별 모션 (HW vs sim replay)", "",
                  "| label | n | HW lag p50 ms | sim lag p50 ms | EE RMS p50 mm | EE max p95 mm | peak vel rad/s |",
                  "|---|---|---|---|---|---|---|"]
        for lab, b in sorted(hw_m["by_label"].items()):
            lines.append(f"| {lab} | {b['n']} | {_fmt(b['hw_lag_ms_p50'])} | {_fmt(b['sim_lag_ms_p50'])} | "
                         f"{_fmt(b['hw_vs_sim_ee_rms_mm_p50'])} | {_fmt(b['hw_vs_sim_ee_max_mm_p95'])} | "
                         f"{_fmt(b['peak_vel_p50'])} |")
        lines.append("")
    if r.get("notes"):
        lines += ["## 해석 메모", ""] + [f"- {n}" for n in r["notes"]] + [""]
    lines += ["임계값은 `validation/analyze.py::THRESHOLDS`. 전체 수치는 `report.json`, "
              "윈도우별 cmd/HW/sim 궤적은 `motion_windows.npz`.", ""]
    path = run_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def interpret(r: dict) -> list[str]:
    """Plain-language flags for the known structural sim/real differences."""
    notes = []
    hs = r["hw"]["state"]
    ob = hs.get("obs", {})
    ref_ob = _g(r, "sim_reference", "state", "obs") or {}
    off = ob.get("bbox_offset_median_mm")
    if off is not None and off > 50 and (ref_ob.get("bbox_offset_median_mm") or 0) < 10:
        notes.append(
            f"obs의 bbox corner가 xyz_now와 {off:.0f} mm 떨어져 있음 (sim truth track은 ~0). "
            "corner 는 `rl/common.bbox_belt_shift` 로 마지막 검출 이후 벨트 이동만큼 옮겨지므로, "
            "남은 차이는 bbox 프레임과 제어 앵커(첫 검출/v4 중앙 프레임)의 위치 차이 — "
            "카메라 위치 편향이나 물체가 벨트와 다른 속도로 움직이는지 확인.")
    rt = ob.get("remaining_time_frac") or {}
    ref_rt = _g(ref_ob, "remaining_time_frac") or {}
    if rt.get("n") and rt.get("std", 1) < 1e-9 and (ref_rt.get("std") or 0) > 1e-3:
        notes.append(
            "remaining_time_frac가 HW에서 상수(1.0) — 학습에선 240 s 에피소드 동안 1→0. "
            "HW 루프에 에피소드 시계를 넣거나 학습에서 이 feature를 빼야 함.")
    conf = _g(hs, "tracks", "conf") or {}
    if conf.get("n") and conf.get("std", 0) > 0.02:
        notes.append(
            f"HW confidence 분포 p5–p95 = {_fmt(conf.get('p5'))}–{_fmt(conf.get('p95'))} "
            "(sim은 항상 0.9) → 정책은 conf 변화를 본 적이 없음.")
    yaw = _g(hs, "tracks", "bbox_axis_aligned_frac")
    ref_yaw = _g(r, "sim_reference", "state", "tracks", "bbox_axis_aligned_frac")
    if yaw is not None and ref_yaw is not None and yaw > 0.8 and ref_yaw < 0.3:
        notes.append(
            f"HW bbox의 {yaw * 100:.0f}%가 축정렬 (sim은 yaw 랜덤 oriented box) → corner 표현이 다름.")
    plan = _g(r, "hw", "timing", "planning_belt_travel_mm")
    if plan is not None and plan > 5:
        notes.append(
            f"결정 후 첫 모션까지 벨트가 {plan:.1f} mm 이동 — sim(_SimClock)에선 계획 연산이 "
            "시뮬 시간을 소비하지 않아 0. 스킬 타이밍이 이 지연을 흡수하는지 확인 필요.")
    lag = _g(r, "hw", "motion", "summary", "hw_lag_ms", "p50")
    slag = _g(r, "hw", "motion", "summary", "sim_lag_ms", "p50")
    if lag is not None and slag is not None and lag - slag > 15:
        notes.append(
            f"HW 추종 지연 {lag:.0f} ms vs sim {slag:.0f} ms — sim 액추에이터(kp/kv)나 명령에 "
            "순수 지연을 넣어 맞추는 것을 고려.")
    return notes


def analyze(run_dir, *, replay: bool = True, sim_steps: int = 40,
            sim_ref_dir=None, sim_seed: int = 0) -> dict:
    run = load_run(run_dir)
    t0 = time.time()
    result: dict = {"hw": {"meta": run.meta}}
    print(f"[s2r] loaded {run.path}: cmd {run.cmd_t.size}, fb {run.fb_t.size}, "
          f"steps {len(run.steps)}, track updates {len(run.tracks)}", flush=True)
    result["hw"]["motion"] = analyze_motion(run, replay=replay)
    result["hw"]["timing"] = analyze_timing(run)
    result["hw"]["state"] = analyze_state(run)
    print(f"[s2r] HW metrics done ({time.time() - t0:.1f}s)", flush=True)

    ref_run = None
    if sim_ref_dir is None and sim_steps > 0 and run.meta.get("source") != "sim_reference":
        sim_ref_dir = run.path / "sim_reference"
        if not (sim_ref_dir / "meta.json").exists():
            try:
                from gp8_control.validation.sim_reference import run_sim_reference

                v = result["hw"]["timing"].get("belt_speed_median") or None
                print(f"[s2r] recording sim reference ({sim_steps} executed steps, "
                      f"belt {v}) ...", flush=True)
                run_sim_reference(sim_ref_dir, n_steps=sim_steps, belt_speed=v,
                                  schema=run.schema, seed=sim_seed, verbose=False)
            except Exception as exc:
                result["sim_reference_skipped"] = repr(exc)
                sim_ref_dir = None
    if sim_ref_dir is not None and (Path(sim_ref_dir) / "meta.json").exists():
        ref_run = load_run(sim_ref_dir)
        result["sim_reference_path"] = str(sim_ref_dir)
        result["sim_reference"] = {
            "timing": analyze_timing(ref_run),
            "state": analyze_state(ref_run),
        }
        result["features"] = compare_features(run, ref_run)
    elif "sim_reference_skipped" not in result:
        result["sim_reference_skipped"] = "disabled"
    result["verdicts"] = build_verdicts(result)
    result["notes"] = interpret(result)
    out = Path(run.path)
    (out / "report.json").write_text(
        json.dumps(result, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else repr(o)),
        encoding="utf-8")
    path = write_report(out, result)
    print(f"[s2r] report -> {path} ({time.time() - t0:.1f}s)", flush=True)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="gp8_s2r_analyze", description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir")
    parser.add_argument("--no-replay", action="store_true",
                        help="skip replaying the command stream through MuJoCo")
    parser.add_argument("--sim-steps", type=int, default=40,
                        help="executed skills in the sim reference rollout (0 = none)")
    parser.add_argument("--sim-ref", default=None,
                        help="reuse an existing sim reference run dir")
    parser.add_argument("--sim-seed", type=int, default=0)
    args = parser.parse_args(argv)
    res = analyze(args.run_dir, replay=not args.no_replay, sim_steps=args.sim_steps,
                  sim_ref_dir=args.sim_ref, sim_seed=args.sim_seed)
    counts = defaultdict(int)
    for v in res["verdicts"]:
        counts[v["status"]] += 1
    print(f"[s2r] PASS {counts['PASS']} WARN {counts['WARN']} FAIL {counts['FAIL']} "
          f"N/A {counts['N/A']}", flush=True)
    for n in res.get("notes", []):
        print(f"[s2r] note: {n}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

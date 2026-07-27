"""Offline analysis of push tracking headroom from MotionLogger samples.

Turns a step-up test session (several app runs at increasing
GP8_PUSH_SPEED/GP8_PUSH_ACCEL, each with GP8_MOTION_LOG_DIR set) into a
per-step table of what the arm ACTUALLY did — the commanded-vs-actual "knee"
is the tracking limit:

  * actual peak TCP speed during each push stroke  (saturation → stops rising)
  * actual minimum TCP z vs the planned scoop floor (path distortion → dives)
  * peak joint velocity as % of M1                  (headroom)

No ROS needed — reads ``motion_samples_*.csv`` (250 Hz actual joint
positions) and reconstructs TCP motion through the analytical FK.

Usage::

  PYTHONPATH=$HOME/ros2_ws/src \
    ~/ros2_ws/src/gp8_control/.venv/bin/python \
    -m gp8_control.tests.analyze_push_tracking [~/gp8_motion] \
    [--since 2026-07-13] [--steps 2.0,2.2,2.4,2.6]

``--steps`` (optional) labels the session files, in chronological order, with
the commanded speeds you ran — purely cosmetic for the report.

Stroke detection: a burst of TCP speed > 0.6 m/s whose LOW point is near the
belt (median z < 0.06 m) is a push stroke; high-z bursts (chains, transits)
are ignored.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import os
import sys

import numpy as np

from gp8_control.robots.gp8 import GP8

PLAN_Z_MIN = 0.01          # the scoop's commanded low point (PUSH_HEIGHT)
SPEED_GATE = 0.6           # m/s — burst detection threshold
STROKE_Z_GATE = 0.06       # m — bursts whose median z is above this are chains
MIN_BURST_S = 0.10         # s — ignore blips


def analyze_file(path: str, robot: GP8, m1: np.ndarray) -> list[dict]:
    d = np.genfromtxt(path, delimiter=",", skip_header=1)
    if d.ndim != 2 or d.shape[0] < 50:
        return []
    t, q = d[:, 0], d[:, 3:9]
    pos = np.array([robot.forward_kinematics(q[i])[:3, 3] for i in range(len(q))])
    dt = np.gradient(t)
    v = np.linalg.norm(np.gradient(pos, axis=0) / dt[:, None], axis=1)
    v = np.convolve(v, np.ones(5) / 5, "same")          # 5-sample smoothing
    jv = np.abs(np.gradient(q, axis=0) / dt[:, None])   # joint speeds

    strokes = []
    i, n = 0, len(v)
    while i < n:
        if v[i] > SPEED_GATE:
            j = i
            while j < n and (v[j] > SPEED_GATE or (j - i) < 50):
                j += 1
            seg = slice(i, j)
            if (t[j - 1] - t[i]) >= MIN_BURST_S and np.median(pos[seg, 2]) < STROKE_Z_GATE:
                strokes.append({
                    "t0": t[i] - t[0],
                    "dur": t[j - 1] - t[i],
                    "v_peak": float(v[seg].max()),
                    "z_min": float(pos[seg, 2].min()),
                    "jv_pct": float((jv[seg] / m1[None, :]).max() * 100.0),
                })
            i = j
        else:
            i += 1
    return strokes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("log_dir", nargs="?",
                    default=os.path.expanduser("~/gp8_motion"))
    ap.add_argument("--since", default=datetime.date.today().isoformat(),
                    help="only sessions on/after this date (YYYY-MM-DD)")
    ap.add_argument("--steps", default="",
                    help="comma list of commanded speeds, chronological")
    args = ap.parse_args()

    since = datetime.datetime.fromisoformat(args.since).timestamp()
    files = sorted(
        f for f in glob.glob(os.path.join(args.log_dir, "motion_samples_*.csv"))
        if os.path.getmtime(f) >= since
    )
    if not files:
        print(f"no motion_samples_*.csv since {args.since} in {args.log_dir}")
        sys.exit(1)
    labels = [s.strip() for s in args.steps.split(",") if s.strip()]

    robot = GP8()
    m1 = robot.velocity_limits * 0.9

    print(f"{'session':<28}{'cmd':>6} {'strokes':>7} {'v_peak(m/s)':>16} "
          f"{'z_min(mm)':>10} {'jointV%':>8}   verdict")
    print("-" * 88)
    for k, f in enumerate(files):
        strokes = analyze_file(f, robot, m1)
        name = os.path.basename(f).replace("motion_samples_", "").replace(".csv", "")
        cmd = labels[k] if k < len(labels) else "?"
        if not strokes:
            print(f"{name:<28}{cmd:>6} {'0':>7}   (no low-z strokes found)")
            continue
        vp = [s["v_peak"] for s in strokes]
        zmin = min(s["z_min"] for s in strokes)
        jmax = max(s["jv_pct"] for s in strokes)
        dz = (PLAN_Z_MIN - zmin) * 1000.0
        verdict = "OK"
        if dz > 5.0:
            verdict = f"경로왜곡 (계획보다 {dz:.0f}mm 아래)"
        try:
            if cmd != "?" and max(vp) < 0.85 * float(cmd):
                verdict = ("포화 " + verdict) if verdict != "OK" else "속도포화"
        except ValueError:
            pass
        print(f"{name:<28}{cmd:>6} {len(strokes):>7} "
              f"{min(vp):>7.2f}~{max(vp):<7.2f} {zmin * 1000:>+9.1f} "
              f"{jmax:>7.0f}%   {verdict}")

    print("\n해석: cmd를 올렸는데 v_peak가 안 오르면 그 사이가 추종 한계(무릎)."
          "\n      z_min이 계획(+10mm)보다 5mm 이상 낮으면 경로 왜곡 시작 — 그 단계 이상 금지.")


if __name__ == "__main__":
    main()

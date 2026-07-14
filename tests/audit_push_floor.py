"""Offline floor-clearance audit for push motions (step-0 diagnosis — NO robot).

Answers ONE question without touching hardware: **does the COMMANDED path itself
ever dip toward the belt?** The command stream is fully determined by our own
code (skill build -> 20 Hz knots -> stream cubic-Hermite resample -> 4 ms
samples), so it can be reproduced exactly at the desk:

  1. rebuild the same trajectories ``PushSkill`` dispatches (same build code),
  2. resample them exactly like the 250 Hz stream driver
     (``trajectory_controller._stream``: CubicHermiteSpline + knot-envelope clamp),
  3. FK every 4 ms command sample and track the PADDLE-BOTTOM minimum height,
  4. report the worst clearance per scenario over a grid of realistic geometries.

If a scenario's commanded clearance goes negative -> the plan commands a belt
strike (the robot faithfully executing a bad plan); the failing segment and
geometry are printed. If every scenario is clean -> planning is exonerated and a
real strike must come from execution (tracking / limiter) — that is what the
hardware motion log (GP8_MOTION_LOG_DIR) then tests.

Scenario families:
  * ``direct``  — the POSITIONING direct descent: old backswing -> new backswing
                  (intercept re-planned a few cm away, XY gap < PUSH_APPROACH_VIA_XY
                  so no hover via). Suspect #1 for low-to-low joint-space sag.
                  RAW/UNGATED on purpose — diagnosis baseline only, NOT asserted
                  (it proves the floor gate is still needed).
  * ``gated``   — REGRESSION: the SAME direct-descent grid routed through the
                  production gate (``PushSkill._build_gated_approach``: build
                  direct -> ``_transit_sag`` audit -> Cartesian arc replacement
                  -> trapezoid fallback). Reports the chosen route
                  (direct|arc|trapezoid) and direct-vs-chosen duration.
                  ASSERTED: the CHOSEN segment must not sag below
                  -(PUSH_TRANSIT_SAG_TOL + 1mm) under its own endpoints.
                  (Absolute clearance is informational here: the endpoints are
                  the eye-calibrated backswing poses themselves.)
  * ``full``    — REGRESSION: the whole armed motion (approach + run-up stroke
                  + chain with transit-Z via to the next backswing), as built by
                  ``build_push_trajectory(dispatch=False)`` — now internally
                  floor-gated. ASSERTED: no stream sample below -2mm OUTSIDE
                  the stroke segment's time window. The ENTIRE stroke window
                  [descent end .. stroke end] is excluded — not just a band at
                  t_contact — because the back-lean run-up phase commands
                  slightly negative paddle-edge clearance BY DESIGN
                  (eye-calibrated PUSH_HEIGHT = paddle touches belt).

Exit code: 0 iff every ASSERTED (gated/full) scenario passes its rule above;
the raw ``direct`` family never affects the exit code.

Paddle geometry: clearance is measured at the 4 bottom corners of the paddle,
given in TOOL frame as (down, fore, +/-halfw) offsets from the TCP. The defaults
are GUESSES consistent with the code comments — measure the real paddle and pass
``--pad-down/--pad-fore/--pad-halfw``. The belt reference height is defined
self-consistently as "paddle bottom at the neutral contact pose commanded at
PUSH_HEIGHT" (that is what the eye calibration of PUSH_HEIGHT means), so the
which-segment/shape conclusions are robust even with guessed pad numbers; only
the absolute mm margins move.

Run (no ROS needed):
  PYTHONPATH=$HOME/ros2_ws/src ~/ros2_ws/src/gp8_control/.venv/bin/python \
      -m gp8_control.tests.audit_push_floor
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

from gp8_control.config import Config
from gp8_control.robots.gp8 import GP8
from gp8_control.skills.push_skill import (
    PushSkill,
    PUSH_APPROACH_VIA_XY,
    PUSH_BIN_TARGET_MAP,
    PUSH_FT_GAIN,
    PUSH_FT_MIN,
    PUSH_FT_MAX,
    PUSH_HEIGHT,
    PUSH_TRANSIT_ARC_LIFT,
    PUSH_TRANSIT_SAG_TOL,
)
from gp8_control.trajectory.trajectory_primitive import trajectory

STREAM_DT = 0.004  # 250 Hz stream grid (trajectory_controller.STREAM_DT)


# ---------------------------------------------------------------------------
# Minimal offline stand-in for SkillContext (only what the BUILD path touches)
# ---------------------------------------------------------------------------
class _Log:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def info(self, msg: str) -> None:
        print(f"    [info] {msg}")

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"    [WARN] {msg}")

    error = warn


class _FakeCtx:
    def __init__(self) -> None:
        self.robot = GP8()
        self.cfg = Config()
        self.M1 = np.asarray(self.robot.velocity_limits, dtype=float) * self.cfg.JOINT_VEL_LIMIT_SCALE
        self.M2 = self.M1 * self.cfg.JOINT_ACCEL_LIMIT_SCALE
        self.log = _Log()
        self.queue = []  # empty -> "home/idle" chain branch when no park given
        T_idle = np.eye(4)
        T_idle[:3, :3] = np.asarray(self.cfg.INITIAL_R, dtype=float)
        T_idle[:3, 3] = np.asarray(self.cfg.INITIAL_T, dtype=float).ravel()
        q = self.robot.inverse_kinematics(T_idle)
        self.idle_joint = np.asarray(q, dtype=float) if q is not None else np.zeros(6)

    # mirror of SkillContext.lifted_standby_joint (context.py)
    def lifted_standby_joint(self, end_joint: np.ndarray) -> np.ndarray:
        end_joint = np.asarray(end_joint, dtype=float)
        T = self.robot.forward_kinematics(end_joint[:6])
        T[2, 3] = float(self.cfg.INITIAL_T[2, 0])
        q = self.robot.inverse_kinematics(T, q_init=end_joint[:6])
        if q is None:
            return self.idle_joint
        q = np.asarray(q, dtype=float)
        q[-1] = self.cfg.PICK_WRIST_J6
        return q


# ---------------------------------------------------------------------------
# Exact replica of the stream driver's resample (trajectory_controller._stream)
# ---------------------------------------------------------------------------
def stream_resample(traj, vel, ts, final_joint=None):
    """20 Hz knots -> the actual 4 ms command samples the servo would receive."""
    arr = np.asarray(traj, dtype=float)
    varr = np.asarray(vel, dtype=float)
    n_joints = arr.shape[0]
    times = np.asarray(ts, dtype=float).ravel()
    if final_joint is not None:
        arr = np.concatenate([arr, np.asarray(final_joint, dtype=float).reshape(n_joints, 1)], axis=1)
        varr = np.concatenate([varr, np.zeros((n_joints, 1))], axis=1)
        times = np.append(times, times[-1] + 0.05)
    T = float(times[-1])
    grid = np.arange(0.0, T + STREAM_DT, STREAM_DT)
    if varr.shape == arr.shape and np.all(np.diff(times) > 0):
        from scipy.interpolate import CubicHermiteSpline
        samples = np.column_stack(
            [CubicHermiteSpline(times, arr[j], varr[j])(grid) for j in range(n_joints)])
        seg = np.clip(np.searchsorted(times, grid, side="right") - 1, 0, len(times) - 2)
        lo = np.minimum(arr[:, seg], arr[:, seg + 1]).T
        hi = np.maximum(arr[:, seg], arr[:, seg + 1]).T
        samples = np.clip(samples, lo, hi)
    else:
        samples = np.column_stack([np.interp(grid, times, arr[j]) for j in range(n_joints)])
    return grid, samples  # (n_grid,), (n_grid, 6)


# ---------------------------------------------------------------------------
# Paddle-bottom height
# ---------------------------------------------------------------------------
def make_corners(down: float, fore: float, halfw: float) -> np.ndarray:
    """4 bottom corners in TOOL frame. Tool X = approach (down at neutral),
    tool Y = push-facing (paddle normal), tool Z = width (horizontal, ⊥ push)."""
    return np.array([
        [down,  fore,  halfw],
        [down,  fore, -halfw],
        [down, -fore,  halfw],
        [down, -fore, -halfw],
    ])


def bottom_z_of_T(T: np.ndarray, corners: np.ndarray) -> float:
    pts = (T[:3, :3] @ corners.T).T + T[:3, 3]
    return float(np.min(pts[:, 2]))


def sweep_bottom_z(robot, samples: np.ndarray, corners: np.ndarray):
    """FK every command sample -> (bottom_z per sample, tcp_z per sample)."""
    n = samples.shape[0]
    bz = np.empty(n)
    tz = np.empty(n)
    for i in range(n):
        T = robot.forward_kinematics(samples[i, :6])
        bz[i] = bottom_z_of_T(T, corners)
        tz[i] = float(T[2, 3])
    return bz, tz


# ---------------------------------------------------------------------------
# Scenario geometry helpers
# ---------------------------------------------------------------------------
def backswing_for(skill: PushSkill, ctx: _FakeCtx, x: float, y: float, bin_xyz):
    """Backswing (retreat) pose for an intercept at (x, y), aiming at bin_xyz.
    Returns (q_backswing, T_backswing, T_contact) or None if IK failed."""
    T_grasp = np.eye(4)
    T_grasp[:3, 3] = (x, y, ctx.cfg.GRASP_Z)
    T_aim = np.eye(4)
    T_aim[:3, 3] = (x, y, 0.15)
    T_aim2 = np.eye(4)
    T_aim2[:3, 3] = np.asarray(bin_xyz, dtype=float)

    # crude but valid IK seeds (only branch selection depends on them)
    push_dir = skill._compute_push_direction(T_grasp, T_aim2)
    T_seed = T_grasp.copy()
    T_seed[:3, :3] = skill._push_orientation(push_dir, 0.0)
    T_seed[2, 3] = 0.10
    q_seed = ctx.robot.inverse_kinematics(T_seed)
    if q_seed is None:
        return None
    q_seed = np.asarray(q_seed, dtype=float)

    wait_j, q_bs, T_bs = skill._compute_retreat_poses(
        T_grasp, T_aim2, T_aim, q_seed, q_seed)
    # detect the no-runup/raw fallback (backswing == contact pose) — still auditable
    return q_bs, T_bs, T_grasp


def summarize(grid, bz, belt_z):
    """min clearance (m), its time, and index."""
    clear = bz - belt_z
    i = int(np.argmin(clear))
    return float(clear[i]), float(grid[i]), i


def ascii_profile(grid, clear, width=72, height=10):
    """Tiny ASCII plot of clearance vs time (for the worst scenario)."""
    n = len(clear)
    idx = np.linspace(0, n - 1, width).astype(int)
    c = clear[idx]
    lo, hi = float(np.min(c)), float(np.max(c))
    span = max(hi - lo, 1e-6)
    rows = []
    for r in range(height, -1, -1):
        level = lo + span * r / height
        row = "".join("#" if cv >= level else " " for cv in c)
        rows.append(f"  {level * 1000:+7.1f}mm |{row}")
    rows.append(f"           +{'-' * width}")
    rows.append(f"            t=0{' ' * (width - 12)}t={grid[-1]:.2f}s")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pad-down", type=float, default=0.02,
                    help="paddle bottom offset below TCP along tool X [m] (MEASURE!)")
    ap.add_argument("--pad-fore", type=float, default=0.025,
                    help="paddle bottom fore/aft half-extent along tool Y [m] (MEASURE!)")
    ap.add_argument("--pad-halfw", type=float, default=0.05,
                    help="paddle half-width along tool Z [m] (MEASURE!)")
    ap.add_argument("--csv", default=None,
                    help="optional output CSV path for all per-scenario rows")
    args = ap.parse_args()

    corners = make_corners(args.pad_down, args.pad_fore, args.pad_halfw)
    ctx = _FakeCtx()
    skill = PushSkill(ctx)
    robot = ctx.robot
    bin_xyz = PUSH_BIN_TARGET_MAP["metal"]

    # Belt reference: paddle bottom at the NEUTRAL contact pose commanded at
    # PUSH_HEIGHT — by definition of the eye calibration this is the belt surface.
    T_ref = np.eye(4)
    T_ref[:3, :3] = skill._push_orientation(np.array([1.0, 0.0, 0.0]), 0.0)
    T_ref[:3, 3] = (0.5, 0.0, PUSH_HEIGHT)
    belt_z = bottom_z_of_T(T_ref, corners)
    print(f"paddle corners (tool frame): down={args.pad_down} fore=±{args.pad_fore} "
          f"halfw=±{args.pad_halfw}  ->  belt ref z = {belt_z * 1000:+.1f}mm (base frame)")
    print(f"metal bin: {bin_xyz}   PUSH_HEIGHT={PUSH_HEIGHT}   "
          f"via threshold={PUSH_APPROACH_VIA_XY}m\n")

    rows = []
    worst = None  # (clearance, label, grid, clear_curve)

    def record(family, label, grid, bz, tz, extra="", strike=None):
        """Log one scenario row. ``strike=None`` -> flag on cmin<0 (raw
        diagnosis rule); True/False -> the ASSERTED families' own verdict
        (their pass rules differ — sag-relative / outside-stroke-window)."""
        nonlocal worst
        clear = bz - belt_z
        cmin, tmin, _ = summarize(grid, bz, belt_z)
        tcp_min = float(np.min(tz))
        rows.append({
            "family": family, "scenario": label,
            "min_clear_mm": round(cmin * 1000, 1),
            "t_min_s": round(tmin, 3), "dur_s": round(float(grid[-1]), 3),
            "tcp_min_z_mm": round(tcp_min * 1000, 1), "extra": extra,
        })
        if strike is True:
            flag = "  <-- GATED STRIKE (assert fails)"
        elif strike is None and cmin < 0:
            flag = "  <-- STRIKE (commanded)"
        else:
            flag = ""
        note = f"  [{extra}]" if extra else ""
        print(f"  [{family}] {label:55s} min clear {cmin * 1000:+7.1f}mm "
              f"@ t={tmin:5.2f}s / {grid[-1]:4.2f}s{flag}{note}")
        if worst is None or cmin < worst[0]:
            worst = (cmin, f"[{family}] {label}", grid, clear)
        return cmin

    # ================= family 1: direct descent (old bs -> new bs) ============
    # RAW / UNGATED baseline — diagnosis only, never asserted. The same grid is
    # re-run through the production floor gate as family 3 ("gated") below.
    print("== family 1 direct descent (RAW, ungated baseline): old backswing -> "
          "re-planned backswing (no via) ==")
    xs = (0.40, 0.50, 0.60)
    ys = (-0.20, 0.0, 0.20)
    disps = [(0.0, -0.05), (0.0, -0.10), (0.0, -0.15), (0.0, -0.20), (0.0, -0.25),
             (0.05, -0.10), (-0.05, -0.10), (0.10, 0.0), (-0.10, 0.0), (0.0, 0.10)]
    zero6 = np.zeros(6)
    direct_cases = []  # (label, q_old, q_new, traj, vel, ts, raw_cmin) for family 3
    for x in xs:
        for y in ys:
            old = backswing_for(skill, ctx, x, y, bin_xyz)
            if old is None:
                continue
            q_old, T_old, _ = old
            for dx, dy in disps:
                new = backswing_for(skill, ctx, x + dx, y + dy, bin_xyz)
                if new is None:
                    continue
                q_new, T_new, _ = new
                gap = float(np.hypot(*(T_new[:2, 3] - T_old[:2, 3])))
                if gap >= PUSH_APPROACH_VIA_XY:
                    continue  # real code would take the hover via route
                traj, vel, ts = trajectory(q_old[:6], zero6, q_new[:6], zero6,
                                           ctx.M1[:6], ctx.M2[:6], hertz=ctx.cfg.TRAJ_HZ)
                grid, samples = stream_resample(traj, vel, ts, final_joint=q_new[:6])
                bz, tz = sweep_bottom_z(robot, samples, corners)
                label = f"({x:.2f},{y:+.2f}) shift ({dx:+.2f},{dy:+.2f}) gap {gap:.2f}m"
                raw_cmin = record("direct", label, grid, bz, tz)
                direct_cases.append((label, q_old, q_new, traj, vel, ts, raw_cmin))

    # ================= family 3: gated direct descent (regression) ============
    # The SAME grid through the production gate. ASSERT rule (spec): the CHOSEN
    # segment's sag relative to its own endpoints must stay above
    # -(PUSH_TRANSIT_SAG_TOL + 1mm). Absolute clearance is informational here —
    # the endpoints are the eye-calibrated backswing poses, so a direct move
    # kept within the sag tolerance may legitimately graze ~0.
    print(f"\n== family 3 GATED descent: same grid via PushSkill._build_gated_approach "
          f"(tol {PUSH_TRANSIT_SAG_TOL * 1000:.0f}mm, arc lift "
          f"{PUSH_TRANSIT_ARC_LIFT * 100:.0f}cm) ==")
    gated_viol = []          # (family, label, value_mm) assert failures
    n_arc = n_trap = n_kept = 0
    added_durs = []          # chosen - direct duration where the gate replaced
    worst_gated = np.inf     # worst min-clearance over ALL asserted scenarios
    for label, q_old, q_new, traj_d, vel_d, ts_d, raw_cmin in direct_cases:
        dur_direct = float(ts_d[-1])
        # q_end: audit against the TRUE goal — trajectory()'s truncated last
        # knot can sit mid-dip and mask the sag (the push_skill gate bug this
        # regression caught on 2026-07-14).
        sag_direct, _ = skill._transit_sag(traj_d, vel_d, ts_d, q_end=q_new[:6])
        w0 = len(ctx.log.warnings)
        traj_c, vel_c, ts_c = skill._build_gated_approach(
            q_old[:6], q_new[:6], via=None)
        warns = ctx.log.warnings[w0:]
        if any("Arc AND trapezoid-mid IK failed" in w for w in warns):
            route = "kept-direct"
            n_kept += 1
        elif any("Arc transit IK failed" in w or "Slerp unavailable" in w
                 for w in warns):
            route = "trapezoid"
            n_trap += 1
        elif any("Direct descent sags" in w for w in warns):
            route = "arc"
            n_arc += 1
        else:
            route = "direct"
        dur_chosen = float(ts_c[-1])
        if route in ("arc", "trapezoid"):
            added_durs.append(dur_chosen - dur_direct)
        sag_chosen, _ = skill._transit_sag(traj_c, vel_c, ts_c, q_end=q_new[:6])
        viol = sag_chosen < -(PUSH_TRANSIT_SAG_TOL + 0.001)
        grid, samples = stream_resample(traj_c, vel_c, ts_c, final_joint=q_new[:6])
        bz, tz = sweep_bottom_z(robot, samples, corners)
        cmin = record(
            "gated", label, grid, bz, tz,
            extra=(f"route={route} sag {sag_direct * 1000:+.1f}→"
                   f"{sag_chosen * 1000:+.1f}mm dur {dur_direct:.2f}→"
                   f"{dur_chosen:.2f}s raw_clear {raw_cmin * 1000:+.1f}mm"),
            strike=viol,
        )
        worst_gated = min(worst_gated, cmin)
        if viol:
            gated_viol.append(("gated", label, sag_chosen * 1000))

    # ================= family 4: full armed motion (regression) ===============
    # The whole armed motion — now INTERNALLY gated (build_push_trajectory runs
    # the same _build_gated_approach + arc-descent chain leg). ASSERT rule
    # (spec): no stream sample below -2mm OUTSIDE the stroke segment's time
    # window. The ENTIRE stroke window [descent end .. stroke end] is excluded
    # (knot indices [n_desc, n_desc+n_stroke) from the build meta): the
    # back-lean run-up commands slightly negative paddle-edge clearance BY
    # DESIGN (eye-calibrated contact), so a band around t_contact alone would
    # false-positive on a correct motion.
    # NOTE GP8_PUSH_PREPOSITION does not branch this family: the audit calls
    # build_push_trajectory directly, and the flag only routes execute() —
    # running under both env values checks the build path is env-insensitive.
    print("\n== family 4 FULL build (gated): approach + run-up stroke + chain "
          "(transit via); assert outside-stroke > -2mm ==")
    preposition_env = os.environ.get("GP8_PUSH_PREPOSITION", "1")
    print(f"   (GP8_PUSH_PREPOSITION={preposition_env!r} — build path is shared "
          f"by both execute modes)")
    for x, y in ((0.45, 0.15), (0.50, 0.0), (0.55, -0.15), (0.60, 0.10)):
        cur = backswing_for(skill, ctx, x, y + 0.10, bin_xyz)   # parked at stale bs
        bs = backswing_for(skill, ctx, x, y, bin_xyz)
        nxt = backswing_for(skill, ctx, x, y - 0.15, bin_xyz)   # next cycle's park
        if cur is None or bs is None:
            continue
        q_cur, _, _ = cur
        q_bs, T_bs, T_contact = bs
        chain_park = nxt[0] if nxt is not None else None

        contact_offset = float(np.hypot(*(T_contact[:2, 3] - T_bs[:2, 3])))
        dist_bin = float(np.hypot(*(np.asarray(bin_xyz[:2]) - T_contact[:2, 3])))
        ft = float(np.clip(PUSH_FT_GAIN * dist_bin, PUSH_FT_MIN, PUSH_FT_MAX))
        push_distance = contact_offset + ft
        T_aim2 = np.eye(4)
        T_aim2[:3, 3] = np.asarray(bin_xyz, dtype=float)

        # (theta arg removed 2026-07-14: build_push_trajectory now derives the
        # telemetry-only heading internally from push_dir.)
        w0 = len(ctx.log.warnings)
        out = skill.build_push_trajectory(
            q_cur, q_bs, T_bs, T_aim2,
            next_grasp=None, append_chain=True, append_descent=True,
            push_distance=push_distance, contact_offset=contact_offset,
            chain_park=chain_park, approach_via=None, dispatch=False,
        )
        gate_fired = any("Direct descent sags" in w or "Seam bridge sags" in w
                         for w in ctx.log.warnings[w0:])
        traj_p, vel_p, ts_p, final_joint, t_contact = out
        grid, samples = stream_resample(traj_p, vel_p, ts_p, final_joint=final_joint)
        bz, tz = sweep_bottom_z(robot, samples, corners)

        # Stroke time window from the build's own segment bookkeeping
        # (meta n_descent includes any seam-bridge knots; index n_desc is the
        # first stroke knot in the concatenated array).
        meta = skill._last_push_meta
        n_desc = int(meta["n_descent"])
        n_stroke = int(meta["n_stroke"])
        t_lo = float(ts_p[n_desc - 1]) if n_desc > 0 else 0.0
        t_hi = float(ts_p[n_desc + n_stroke - 1])
        clear = bz - belt_z
        outside = (grid < t_lo - 1e-9) | (grid > t_hi + 1e-9)
        if np.any(outside):
            j = int(np.argmin(np.where(outside, clear, np.inf)))
            out_min = float(clear[j])
            out_t = float(grid[j])
        else:
            out_min, out_t = np.inf, 0.0
        viol = out_min < -0.002
        record("full", f"intercept ({x:.2f},{y:+.2f}) d={push_distance:.2f}m",
               grid, bz, tz,
               extra=(f"t_contact={t_contact:.2f}s stroke t∈[{t_lo:.2f},{t_hi:.2f}] "
                      f"excluded; outside min {out_min * 1000:+.1f}mm@t={out_t:.2f}s"
                      f"{', gate fired' if gate_fired else ''}"),
               strike=viol)
        worst_gated = min(worst_gated, out_min)
        if viol:
            gated_viol.append(
                ("full", f"intercept ({x:.2f},{y:+.2f})", out_min * 1000))

    # ================= report =================================================
    print("\n== worst commanded clearance (all families, incl. the raw baseline) ==")
    if worst is not None:
        cmin, label, grid, clear = worst
        print(f"{label}: {cmin * 1000:+.1f}mm")
        print(ascii_profile(grid, clear))

    n_raw = sum(1 for r in rows if r["family"] == "direct")
    n_raw_strike = sum(1 for r in rows
                       if r["family"] == "direct" and r["min_clear_mm"] < 0)
    n_gated = sum(1 for r in rows if r["family"] == "gated")
    n_full = sum(1 for r in rows if r["family"] == "full")
    mean_added = float(np.mean(added_durs)) if added_durs else 0.0
    print(f"\n{len(rows)} scenarios audited "
          f"({n_raw} raw-direct + {n_gated} gated + {n_full} full).")
    print(f"raw ungated baseline: {n_raw_strike}/{n_raw} commanded strikes "
          f"(diagnosis only — shows what the gate prevents).")
    print(f"GATED regression: {n_gated + n_full} scenarios, "
          f"routes arc={n_arc} trapezoid={n_trap} kept-direct={n_kept} "
          f"direct(clean)={n_gated - n_arc - n_trap - n_kept}; "
          f"worst gated clearance {worst_gated * 1000:+.1f}mm; "
          f"mean added duration arc vs direct {mean_added * 1000:+.0f}ms.")
    if gated_viol:
        print(f"=> {len(gated_viol)} GATED assert failure(s):")
        for fam, label, mm in gated_viol:
            print(f"   [{fam}] {label}: {mm:+.1f}mm")
        print("=> The gate/build still commands a floor strike: fix push_skill "
              "planning before running hardware.")
    else:
        print("=> 0 gated strikes: every gate-chosen segment stays within the "
              "sag tolerance and every full build keeps >-2mm clearance outside "
              "the (by-design contact) stroke window.")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"rows written to {args.csv}")
    return 1 if gated_viol else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Offline summary of the push cycle log (``~/gp8_pick_log.push.csv``).

Answers the minimal-timing rewrite's judgement questions from the CSV that
``PushSkill._log_push_cycle`` appends per cycle:

  * ``fire_early_ms`` bias — does the constant-free fire lead run consistently
    early (>0) or late (<0)?  A consistent bias is the ONLY evidence on which a
    timing term may be re-added (see the minimal-timing comment block in
    ``skills/push_skill.py``).
  * ``budget_position_ms`` vs ``approach_ms`` — does the placement model's
    position estimate match what the built trajectory actually planned?
  * abort rate + ``past_stroke_m`` — how often (and by how much) timing slipped
    so far that the stale-stroke guard refused to fire.
  * ``det_age_ms`` — how much of the scoreboard rests on stale dead-reckoning.

stdlib only (csv/statistics) — runs with any ``python3``, no venv needed:

    python3 tools/analyze_pick_log.py                 # default CSV path
    python3 tools/analyze_pick_log.py /path/to.csv    # explicit file
    python3 tools/analyze_pick_log.py --fresh-ms 300  # stricter freshness cut
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys

DEFAULT_CSV = os.path.expanduser("~/gp8_pick_log.push.csv")

# Columns that only the push logger writes. Their presence in the header proves
# the file was written by _log_push_cycle's current schema — the old shared
# ~/gp8_pick_log.csv (throw rows and/or the pre-abort-logging push schema)
# fails this check instead of being silently mis-parsed.
REQUIRED_COLUMNS = (
    "outcome",
    "fire_early_ms",
    "budget_position_ms",
    "approach_ms",
    "det_age_ms",
)


def _f(row: dict, key: str):
    """Float cell or None (blank/missing/garbled cells collapse to None)."""
    v = row.get(key)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _pct(xs: list, q: float) -> float:
    """Nearest-rank percentile of a non-empty pre-sorted list."""
    i = min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))
    return xs[i]


def stats_line(label: str, xs: list, unit: str = "ms") -> str:
    """One aligned summary line: n, mean±σ, median, [p10, p90], min/max."""
    xs = [x for x in xs if x is not None]
    if not xs:
        return f"  {label:<34} (no data)"
    s = sorted(xs)
    mean = statistics.fmean(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return (
        f"  {label:<34} n={len(xs):<4d} mean {mean:+8.1f} ± {sd:6.1f} {unit}"
        f"   med {statistics.median(s):+8.1f}"
        f"   p10/p90 {_pct(s, 0.10):+8.1f}/{_pct(s, 0.90):+8.1f}"
        f"   min/max {s[0]:+8.1f}/{s[-1]:+8.1f}"
    )


def load_rows(path: str) -> tuple[list, int]:
    """Parse the CSV → (rows, n_malformed). Exits with guidance on schema/IO errors."""
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f, restkey="_extra")
            header = reader.fieldnames or []
            missing = [c for c in REQUIRED_COLUMNS if c not in header]
            if missing:
                sys.exit(
                    f"error: {path} lacks column(s) {missing} — this is not a "
                    f"current-schema push log. The old shared ~/gp8_pick_log.csv "
                    f"mixes throw rows / predates abort logging; point me at the "
                    f"push-only file (~/gp8_pick_log.push.csv) written after the "
                    f"minimal-timing logging change."
                )
            rows, malformed = [], 0
            for row in reader:
                # Cell-count drift (a schema change mid-file) shows up as extra
                # cells or a None-valued tail — count, don't trust.
                if "_extra" in row or None in row.values():
                    malformed += 1
                    continue
                rows.append(row)
            return rows, malformed
    except OSError as e:
        sys.exit(f"error: cannot read {path}: {e} (no push cycles logged yet?)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv_path", nargs="?", default=DEFAULT_CSV)
    ap.add_argument(
        "--fresh-ms", type=float, default=500.0,
        help="det_age_ms cut for the 'fresh detections only' fire_early view "
             "(default 500: dead-reckoned longer than this is suspect)",
    )
    args = ap.parse_args()

    rows, malformed = load_rows(args.csv_path)
    if not rows:
        sys.exit(f"{args.csv_path}: no parseable rows")

    ok = [r for r in rows if r["outcome"] == "ok"]
    aborts = [r for r in rows if r["outcome"] == "abort"]
    other = len(rows) - len(ok) - len(aborts)

    print(f"push cycle log: {args.csv_path}")
    print(f"  rows {len(rows)}  (ok {len(ok)}, abort {len(aborts)}"
          + (f", other {other}" if other else "")
          + (f"; skipped {malformed} malformed" if malformed else "") + ")")
    if aborts:
        by_reason: dict = {}
        for r in aborts:
            by_reason[r.get("abort_reason") or "?"] = (
                by_reason.get(r.get("abort_reason") or "?", 0) + 1
            )
        reasons = ", ".join(f"{k}={v}" for k, v in sorted(by_reason.items()))
        print(f"  abort rate {len(aborts) / len(rows):.1%}  ({reasons})")

    # ---- fire timing (the headline judgement) ----
    print("\nfire_early_ms  (>0 fired early, <0 late; residual AFTER applied_lag"
          " when that column is present)")
    print(stats_line("ok cycles", [_f(r, "fire_early_ms") for r in ok]))
    fresh = [
        r for r in ok
        if (_f(r, "det_age_ms") or 0.0) <= args.fresh_ms
    ]
    print(stats_line(f"ok, det_age ≤ {args.fresh_ms:.0f}ms",
                     [_f(r, "fire_early_ms") for r in fresh]))
    for via, name in ((0.0, "route: direct"), (1.0, "route: via-hover")):
        sub = [r for r in ok if _f(r, "budget_via") == via]
        print(stats_line(name, [_f(r, "fire_early_ms") for r in sub]))
    for mode in sorted({r.get("swing_mode") or "?" for r in ok}):
        sub = [r for r in ok if (r.get("swing_mode") or "?") == mode]
        print(stats_line(f"swing: {mode}", [_f(r, "fire_early_ms") for r in sub]))
    if aborts:
        print(stats_line("abort cycles (would-be fire)",
                         [_f(r, "fire_early_ms") for r in aborts]))

    # ---- placement model vs built trajectory ----
    resid = []
    for r in ok:
        b, a = _f(r, "budget_position_ms"), _f(r, "approach_ms")
        if b is not None and a is not None:
            resid.append(b - a)
    print("\nplacement model check")
    print(stats_line("budget_position − approach", resid))
    print(stats_line("budget_runup_ms", [_f(r, "budget_runup_ms") for r in ok]))

    # ---- applied-lag residual (upgrade trigger for the det_age formula) ----
    lag_vals = [x for x in (_f(r, "applied_lag_ms") for r in ok) if x is not None]
    age_fe = []
    for r in ok:
        a, fe = _f(r, "det_age_ms"), _f(r, "fire_early_ms")
        if a is not None and fe is not None:
            age_fe.append((a, fe))
    if lag_vals:
        print("\napplied-lag check")
        if min(lag_vals) == max(lag_vals):
            print(f"  applied_lag_ms                     {lag_vals[0]:.0f}"
                  f" (constant this run)")
        else:
            print(stats_line("applied_lag_ms", lag_vals))
        if len(age_fe) >= 6:
            n = len(age_fe)
            ma = sum(a for a, _ in age_fe) / n
            mf = sum(f for _, f in age_fe) / n
            saa = sum((a - ma) ** 2 for a, _ in age_fe)
            sff = sum((f - mf) ** 2 for _, f in age_fe)
            saf = sum((a - ma) * (f - mf) for a, f in age_fe)
            if saa > 1e-9 and sff > 1e-9:
                print(
                    f"  residual vs det_age                slope "
                    f"{saf / saa * 1000.0:+.1f} ms/s  "
                    f"(r={saf / (saa * sff) ** 0.5:+.2f}, n={n})"
                )
                print(
                    "  [a clear ≈+16 ms/s slope = encoder-scale part is real →"
                    " upgrade the constant to lag + 0.016×det_age]"
                )

    # ---- pipeline + freshness diagnostics ----
    print("\ndiagnostics")
    if any(_f(r, "v_contact_mps") is not None for r in rows):
        print(stats_line("v_contact (planned @contact)",
                         [_f(r, "v_contact_mps") for r in rows], unit="m/s"))
        clamped = [
            r for r in rows if (_f(r, "stroke_clamp_x") or 1.0) > 1.001
        ]
        print(f"  {'stroke clamped (joint limits)':<34} "
              f"{len(clamped)}/{len(rows)} rows"
              + (f", max ×{max(_f(r, 'stroke_clamp_x') for r in clamped):.2f}"
                 if clamped else ""))
    print(stats_line("exec_to_wait_ms (build+plan)",
                     [_f(r, "exec_to_wait_ms") for r in rows]))
    print(stats_line("det_age_ms at fire",
                     [_f(r, "det_age_ms") for r in rows]))
    stale = [r for r in aborts if (r.get("abort_reason") or "") == "stale-stroke"]
    if stale:
        print(stats_line("past_stroke_m (stale aborts)",
                         [(_f(r, "past_stroke_m") or 0.0) * 1000.0 for r in stale],
                         unit="mm"))

    # ---- reading aid ----
    fe = [x for x in (_f(r, "fire_early_ms") for r in fresh) if x is not None]
    if len(fe) >= 5:
        mean = statistics.fmean(fe)
        sd = statistics.stdev(fe)
        if abs(mean) > max(30.0, sd):
            side = "EARLY" if mean > 0 else "LATE"
            print(
                f"\nnote: fresh-detection cycles show a consistent {side} bias "
                f"({mean:+.0f}ms, σ {sd:.0f}) — per the minimal-timing contract, "
                f"re-add ONLY a term this evidence identifies (prefer a runtime "
                f"measurement over a constant)."
            )
        else:
            print(
                f"\nnote: no consistent bias beyond noise "
                f"({statistics.fmean(fe):+.0f}ms, σ {sd:.0f}) — "
                f"keep the constant-free timeline."
            )


if __name__ == "__main__":
    main()

"""Real-to-sim: read run recordings (utils/run_record.py) and compare them.

    python -m gp8_control.tools.real2sim summary RUN
    python -m gp8_control.tools.real2sim track   RUN [--filter box200_80] [--fit]
    python -m gp8_control.tools.real2sim spawn   RUN -o objects.json [--min-hits 3]
    python -m gp8_control.tools.real2sim compare REAL TWIN

RUN is a recording directory (``GP8_RUN_RECORD_DIR``; ``gp8_cycle`` writes
``~/gp8_logs/cycle_demo/<stamp>/record``). rclpy- and mujoco-free.

* ``summary``: what the run did (objects, plans, pushes / throws, buffer
  writes, late / refused pieces, suction, joint-state rate).
* ``track``: the measured joints against the commands the arm was given,
  raw (hold) and through the controller filter model the twin and the planner
  use (``servo_filter_taps``). Prints per-joint RMS / max error, the time
  offset that fits best and the worst moments. ``--fit`` searches the filter
  model (dead time, moving-average widths) that fits this run best: the number
  to give the twin (``GP8_SIM_SERVO_FILTER=D:W1:W2``).
* ``spawn``: the tracked objects of a run as a twin spawn script. Replay it
  with ``GP8_SIM_SPAWN_SCRIPT=objects.json`` (and the run's belt speed,
  ``GP8_SIM_BELT_SPEED``, printed): the twin's boxes reach the same belt
  positions at the same planner times as the real objects did.
* ``compare``: two recordings of the same object stream (a real run and its
  twin replay): per object, what each run did (push / throw / nothing) and the
  planned contact times, plus counts.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

from gp8_control.utils.run_record import load

DT = 0.002                                   # planner / controller sample period
JOINTS = ("S", "L", "U", "R", "B", "T")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _events(rec, kind):
    return [e for e in rec["events"] if e.get("ev") == kind]


def _t0(rec) -> float | None:
    st = _events(rec, "start")
    return float(st[0]["t0"]) if st else None


def _filter_taps(spec: str, dt: float = DT) -> np.ndarray | None:
    """Same construction as backends.mujoco_sim.servo_filter_taps (no mujoco import)."""
    spec = (spec or "").strip().lower()
    if spec in ("", "0", "off", "none"):
        return None
    if spec == "box200_80":
        delay, widths = 0.04, (0.20, 0.08)
    elif spec == "box200":
        delay, widths = 0.08, (0.20,)
    else:
        parts = [float(x) for x in spec.split(":")]
        delay, widths = parts[0], tuple(w for w in parts[1:] if w > 0)
    h = np.zeros(int(round(delay / dt)) + 1)
    h[-1] = 1.0
    for w in widths:
        n = max(1, int(round(w / dt)))
        h = np.convolve(h, np.full(n, 1.0 / n))
    return h


def _apply(h, U: np.ndarray) -> np.ndarray:
    """Causal FIR per joint, history before the first sample = the first sample."""
    if h is None:
        return U.copy()
    pad = np.repeat(U[:1], len(h) - 1, axis=0)
    X = np.vstack([pad, U])
    return np.column_stack([np.convolve(X[:, j], h, mode="valid") for j in range(U.shape[1])])


def _grid(rec, t_start=None, t_end=None, source="joints"):
    """Commands (hold between samples) and measured joints on one 2 ms grid."""
    cmd, jnt = rec["cmd"], rec[source]
    if not len(cmd) or not len(jnt):
        raise SystemExit("recording has no commands or no joint states")
    a = max(cmd[0, 0], jnt[0, 0]) if t_start is None else t_start
    b = min(jnt[-1, 0], cmd[-1, 0] + 2.0) if t_end is None else t_end
    t = np.arange(a, b, DT)
    idx = np.searchsorted(cmd[:, 0], t, side="right") - 1
    U = cmd[np.clip(idx, 0, None), 1:]
    # history before the grid starts: from the earlier commands
    Q = np.column_stack([np.interp(t, jnt[:, 0], jnt[:, 1 + j]) for j in range(6)])
    # only where a joint sample is near (no long dropouts interpolated)
    near = np.abs(jnt[np.clip(np.searchsorted(jnt[:, 0], t), 0, len(jnt) - 1), 0] - t) < 0.02
    return t, U, Q, near


def _moving(U: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Samples where the arm moves (command or measured), +-0.3 s."""
    v = np.max(np.abs(np.diff(np.vstack([Q[:1], Q]), axis=0)), axis=1) / DT
    vu = np.max(np.abs(np.diff(np.vstack([U[:1], U]), axis=0)), axis=1) / DT
    m = (v > 0.02) | (vu > 0.02)
    k = int(0.3 / DT)
    return np.convolve(m.astype(float), np.ones(2 * k + 1), mode="same") > 0


# ----------------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------------
def cmd_summary(args) -> None:
    rec = load(args.run)
    ev = rec["events"]
    kinds = collections.Counter(e.get("ev") for e in ev)
    meta = (_events(rec, "meta") or [{}])[0]
    t0 = _t0(rec)
    print(f"run {args.run}")
    print(f"  mode {meta.get('mode')} backend {meta.get('backend')} "
          f"belt(plan) {meta.get('planner_belt')} bins x {meta.get('bin_x')}")
    if ev:
        print(f"  span {ev[-1]['m'] - ev[0]['m']:.1f} s, planner t0 "
              + ("-" if t0 is None else f"at +{t0 - ev[0]['m']:.1f} s"))
    print(f"  events {dict(kinds)}")
    j = rec["joints"]
    if len(j) > 1:
        dt = np.diff(j[:, 0])
        print(f"  joints {len(j)} samples, {1 / np.median(dt):.0f} Hz median, "
              f"max gap {1e3 * dt.max():.0f} ms")
    c = rec["cmd"]
    if len(c):
        print(f"  commands {len(c)} samples over {c[-1, 0] - c[0, 0]:.1f} s")

    rx = [e["msg"] for e in _events(rec, "rx")]
    tx = [e["msg"] for e in _events(rec, "tx")]
    plans = [m for m in tx if m.get("t") == "plan"]
    replies = [m for m in rx if m.get("t") == "planned"]
    steps = collections.Counter(s.get("kind") for p in plans for s in p.get("steps", []))
    print(f"  plan requests {len(plans)} (steps {dict(steps)}), replies ok "
          f"{sum(bool(r.get('ok')) for r in replies)} / {len(replies)}")
    taken = sum(len(r.get("taken") or []) for r in replies)
    skipped = collections.Counter(re.sub(r"[-\d.]+", "#", str(w))
                                  for r in replies for _, w in (r.get("skipped") or []))
    print(f"  taken {taken}, skipped {sum(skipped.values())} {dict(skipped)}")
    pushes = [_push_log(m["msg"]) for m in rx if m.get("t") == "log"]
    pushes = [p for p in pushes if p]
    if pushes:
        w = np.array([p["early"] for p in pushes])
        vm = np.array([p["vmax"] for p in pushes])
        print(f"  pushes {len(pushes)}: at the backswing early by median {np.median(w):.2f} s "
              f"(max {w.max():.2f}), realized speed up to median {np.median(vm):.2f} of the limit")
    suc = collections.Counter((e.get("src"), e.get("on")) for e in _events(rec, "suc"))
    print(f"  suction {dict(suc)}")
    buf = _events(rec, "buf")
    if buf:
        lead = np.array([e["lead"] for e in buf])
        rtt = np.array([e.get("rtt", np.nan) for e in buf], float)
        print(f"  buffer writes {len(buf)}: lead min {lead.min():.3f} s median "
              f"{np.median(lead):.3f} s; write round trip p95 {1e3 * np.nanpercentile(rtt, 95):.1f} ms")
    for k in ("buf_late", "buf_refused"):
        for e in _events(rec, k):
            print(f"  {k}: {({kk: v for kk, v in e.items() if kk not in ('m', 'w', 'ev')})}")
    lands = _events(rec, "landing")
    if lands:
        def where(e):
            got, cls = e.get("actual_bin_name"), e.get("class_name")
            if not (e.get("manipulated") or e.get("collateral")):
                return "rode past"
            if got is not None and got in (e.get("class_bin_name"), cls, f"push_{cls}"):
                return "in its bin"
            return f"missed ({got})"
        c = collections.Counter(f"{e.get('class_name')} {e.get('manipulated_skill') or '-'}: "
                                f"{where(e)}" for e in lands)
        print(f"  twin landings: {dict(sorted(c.items()))}")
    for e in _events(rec, "stopped"):
        print(f"  stopped: fault {e.get('fault')!r} stats {e.get('stats')}")


_PUSH_RE = re.compile(r"push (\d+): at the backswing ([-\d.]+) s early, contact at ([-\d.]+), "
                      r"realized speed up to ([\d.]+)")


def _push_log(text: str):
    m = _PUSH_RE.search(str(text))
    if not m:
        return None
    return {"id": int(m.group(1)), "early": float(m.group(2)), "tc": float(m.group(3)),
            "vmax": float(m.group(4))}


# ----------------------------------------------------------------------------
# track
# ----------------------------------------------------------------------------
def _err_stats(t, U, Q, near, h, mask=None):
    F = _apply(h, U)
    m = near if mask is None else (near & mask)
    E = (Q - F)[m]
    return F, E, m


def cmd_track(args) -> None:
    rec = load(args.run)
    t0 = _t0(rec) or rec["cmd"][0, 0]
    source = args.source
    if source == "auto":
        source = "joints_fb" if len(rec.get("joints_fb", ())) else "joints"
    hw = (_events(rec, "meta") or [{}])[0].get("backend") == "hw"
    note = ("  (encoder feedback pos_fdb)" if source == "joints_fb" else
            "  (/joint_states = the driver's pos_set, the accepted command; NOT the realized "
            "motion: record joints_fb)" if hw else "  (twin joints)")
    print(f"measured = {source}{note}")
    t, U, Q, near = _grid(rec, source=source)
    mov = _moving(U, Q)
    print(f"{len(t)} samples ({t[-1] - t[0]:.1f} s), moving {mov.mean() * 100:.0f} %, "
          f"joint data present {near.mean() * 100:.0f} %")
    for spec in ("off", args.filter):
        h = _filter_taps(spec)
        F, E, m = _err_stats(t, U, Q, near, h, mov)
        rms, mx = np.sqrt(np.mean(E ** 2, axis=0)), np.max(np.abs(E), axis=0)
        print(f"\n[{spec}] measured - model while moving [deg]")
        print("        " + "  ".join(f"{j:>6}" for j in JOINTS))
        print("  RMS   " + "  ".join(f"{np.degrees(v):6.2f}" for v in rms))
        print("  max   " + "  ".join(f"{np.degrees(v):6.2f}" for v in mx))
        # the extra delay that fits best (transport, clock, model)
        best = None
        for d in range(-25, 41):
            Fs = np.roll(F, d, axis=0)
            if d > 0:
                Fs[:d] = F[0]
            elif d < 0:
                Fs[d:] = F[-1]
            r = float(np.sqrt(np.mean((Q - Fs)[m] ** 2)))
            if best is None or r < best[1]:
                best = (d, r)
        print(f"  all joints RMS {np.degrees(np.sqrt(np.mean(E ** 2))):.3f} deg; best extra delay "
              f"{best[0] * DT * 1e3:+.0f} ms -> {np.degrees(best[1]):.3f} deg")
        if spec == args.filter:
            err = np.max(np.abs(Q - F), axis=1)
            err[~m] = 0
            w = int(0.5 / DT)
            order = np.argsort(-err)
            shown = []
            print("  worst moments (planner time, joint, error):")
            for i in order:
                if len(shown) >= args.worst or err[i] <= 0:
                    break
                if any(abs(i - s) < w for s in shown):
                    continue
                shown.append(i)
                j = int(np.argmax(np.abs(Q[i] - F[i])))
                print(f"    t={t[i] - t0:8.2f}  {JOINTS[j]}  {np.degrees(Q[i, j] - F[i, j]):+.2f} deg"
                      f"  (cmd speed {np.degrees(np.abs(U[min(i + 1, len(U) - 1), j] - U[i, j]) / DT):.0f} deg/s)")
    if args.fit:
        _fit(t, U, Q, near & mov)


def _fit(t, U, Q, m) -> None:
    print("\nfilter fit (dead time D, moving averages W1, W2) on the moving samples:")
    res = []
    for D in np.arange(0.0, 0.1201, 0.004):
        for W1 in np.arange(0.10, 0.3001, 0.02):
            for W2 in (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12):
                h = _filter_taps(f"{D}:{W1}:{W2}")
                F = _apply(h, U)
                res.append((float(np.sqrt(np.mean((Q - F)[m] ** 2))), D, W1, W2))
    res.sort()
    for r, D, W1, W2 in res[:5]:
        print(f"  D {D:.3f}  W1 {W1:.2f}  W2 {W2:.2f}  -> RMS {np.degrees(r):.3f} deg"
              f"   (GP8_SIM_SERVO_FILTER={D:.3f}:{W1:.2f}:{W2:.2f})")
    ref = _filter_taps("box200_80")
    r0 = float(np.sqrt(np.mean((Q - _apply(ref, U))[m] ** 2)))
    print(f"  box200_80 (twin/planner model now): RMS {np.degrees(r0):.3f} deg")


# ----------------------------------------------------------------------------
# spawn
# ----------------------------------------------------------------------------
def _bbox_shape(bb):
    """(half_x, half_y, yaw) of a base-frame 4-corner box, or None."""
    try:
        P = np.asarray(bb, float)[:, :2]
    except Exception:  # noqa: BLE001
        return None
    if P.shape != (4, 2):
        return None
    e0, e1 = P[1] - P[0], P[2] - P[1]
    l0, l1 = float(np.hypot(*e0)), float(np.hypot(*e1))
    if not (0.01 < l0 < 0.4 and 0.01 < l1 < 0.4):
        return None
    yaw = math.atan2(e0[1], e0[0])
    yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2       # a box is symmetric
    return 0.5 * l0, 0.5 * l1, yaw


def track_objects(rec, min_hits: int = 3) -> list[dict]:
    """One row per established object: first established sighting (planner
    time, y), class by majority, median x, median shape."""
    t0 = _t0(rec)
    if t0 is None:
        raise SystemExit("recording has no planner start")
    obs = collections.defaultdict(list)
    belt = []
    for e in _events(rec, "tracks"):
        belt.append(float(e.get("v") or 0.0))
        for o in e["objs"]:
            obs[o["id"]].append((e["m"] - t0, o))
    out = []
    for tid, rows in obs.items():
        est = [(t, o) for t, o in rows if o.get("cc") or int(o.get("hits", 0)) >= min_hits]
        if len(est) < 2:
            continue
        t_obs, o_first = est[0]
        cls = collections.Counter(o["cls"] for _, o in est).most_common(1)[0][0]
        shapes = [s for s in (_bbox_shape(o.get("bbox")) for _, o in est) if s is not None]
        sim_ids = [o.get("sim") for _, o in est if o.get("sim") is not None]
        row = {"id": int(tid), "cls": cls, "x": float(np.median([o["x"] for _, o in est])),
               "sim": sim_ids[0] if sim_ids else None,
               "t_obs": float(t_obs), "y_obs": float(o_first["y"]),
               "z": float(np.median([o["z"] for _, o in est])), "n": len(est)}
        if shapes:
            S = np.asarray(shapes)
            row["half"] = [float(np.median(S[:, 0])), float(np.median(S[:, 1]))]
            row["yaw"] = float(np.median(S[:, 2]))
        out.append(row)
    out.sort(key=lambda r: r["t_obs"] + (r["y_obs"]) / max(1e-3, np.median(belt) if belt else 0.12))
    # A track the tracker split in two (same place, same time) becomes one box.
    v = float(np.median(belt)) if belt else 0.12
    # A track that starts well downstream of where objects first appear is a
    # re-detection after a manipulation (a pushed box still on the belt), not
    # a new object; in the twin the true object id tells the same.
    if out:
        y_in = float(np.median([r["y_obs"] for r in out]))
        late = [r for r in out if r["y_obs"] < y_in - 0.3]
        seen, first = set(), []
        for r in sorted(out, key=lambda r: r["t_obs"]):
            if r in late or (r["sim"] is not None and r["sim"] in seen):
                continue
            seen.add(r["sim"])
            first.append(r)
        dropped = len(out) - len(first)
        if dropped:
            print(f"  ({dropped} re-detected tracks dropped: first seen downstream of "
                  f"y {y_in - 0.3:.2f} or the same twin object)", file=sys.stderr)
        out = sorted(first, key=lambda r: r["t_obs"] + r["y_obs"] / max(1e-3, v))
    kept: list[dict] = []
    for r in out:
        dup = next((k for k in kept if abs(k["x"] - r["x"]) < 0.05
                    and abs((k["y_obs"] - v * (r["t_obs"] - k["t_obs"])) - r["y_obs"]) < 0.06), None)
        if dup is None:
            kept.append(r)
        elif r["n"] > dup["n"]:
            kept[kept.index(dup)] = r
    return kept, v


def cmd_spawn(args) -> None:
    rec = load(args.run)
    objs, v = track_objects(rec, args.min_hits)
    script = {"t_ref": "planner_t0", "belt": v, "source": str(args.run),
              "objects": [{k: r[k] for k in ("cls", "x", "t_obs", "y_obs", "half", "yaw", "id")
                           if k in r} for r in objs]}
    Path(args.out).write_text(json.dumps(script, indent=1))
    c = collections.Counter(r["cls"] for r in objs)
    print(f"{len(objs)} objects {dict(c)} -> {args.out}")
    print(f"twin: GP8_SIM_SPAWN_SCRIPT={args.out} GP8_SIM_BELT_SPEED={v:.3f} "
          f"GP8_SIM_RANDOMIZE=0 (planner model set GT_BELT must match)")
    zs = [r["z"] for r in objs]
    if zs:
        print(f"real object top z (grasp) median {np.median(zs):.3f} m: compare GP8_CYCLE_TOP_Z / "
              f"the twin box top")


# ----------------------------------------------------------------------------
# compare
# ----------------------------------------------------------------------------
def _actions(rec, min_hits: int) -> list[dict]:
    """Per established object: arrival time at y = 0 and what the run did."""
    objs, v = track_objects(rec, min_hits)
    rx = [e["msg"] for e in _events(rec, "rx")]
    tx = [e["msg"] for e in _events(rec, "tx")]
    did: dict[int, dict] = {}
    for p in tx:
        if p.get("t") != "plan":
            continue
        for s in p.get("steps", []):
            if s.get("kind") == "pick":
                did.setdefault(int(s["obj"]["id"]), {})["asked"] = "throw"
            elif s.get("kind") == "push":
                did.setdefault(int(s["id"]), {})["asked"] = "push"
    for r in rx:
        if r.get("t") == "planned":
            for tid in r.get("taken") or []:
                did.setdefault(int(tid), {})["taken"] = True
            arm = r.get("arm") or {}
            if arm.get("kind") == "obj" and r.get("ok"):
                d = did.setdefault(int(arm["id"]), {})
                d["taken"] = True
                d["tc"] = float(arm.get("tc", np.nan))
        elif r.get("t") == "log":
            pl = _push_log(r.get("msg"))
            if pl:
                did.setdefault(pl["id"], {}).update(taken=True, tc=pl["tc"], vmax=pl["vmax"])
    for o in objs:
        o["t_y0"] = o["t_obs"] + o["y_obs"] / max(1e-3, v)        # planner time at y = 0
        d = did.get(o["id"], {})
        o["did"] = d.get("asked", "-") if d.get("taken") else ("refused " + d["asked"]
                                                               if "asked" in d else "-")
        o["tc"] = d.get("tc")
        o["vmax"] = d.get("vmax")
    return objs


def cmd_compare(args) -> None:
    A = _actions(load(args.real), args.min_hits)
    B = _actions(load(args.twin), args.min_hits)
    used = set()
    print("t(y=0): planner time the object crosses y = 0; contact: planned contact "
          "(pick / push) minus t(y=0)")
    print(f"{'real id':>8} {'cls':>12} {'t(y=0)':>7} | {'real':>14} {'contact':>7} | "
          f"{'twin':>14} {'contact':>7} | dt arrival")
    same = 0
    for a in A:
        cand = [(abs(b["t_y0"] - a["t_y0"]), i) for i, b in enumerate(B)
                if i not in used and b["cls"] == a["cls"] and abs(b["t_y0"] - a["t_y0"]) < args.match_s]
        b = None
        if cand:
            _, i = min(cand)
            used.add(i)
            b = B[i]
        def rel(o):
            return f"{o['tc'] - o['t_y0']:7.2f}" if o and o.get("tc") is not None else "      -"
        same += b is not None and b["did"] == a["did"]
        print(f"{a['id']:>8} {a['cls']:>12} {a['t_y0']:7.2f} | {a['did']:>14} {rel(a)} | "
              f"{(b['did'] if b else 'unmatched'):>14} {rel(b)} | "
              + (f"{b['t_y0'] - a['t_y0']:+.2f}" if b else ""))
    print(f"\nsame action {same} / {len(A)}; twin objects unmatched {len(B) - len(used)}")
    for name, X in (("real", A), ("twin", B)):
        c = collections.Counter(o["did"] for o in X)
        print(f"  {name}: {dict(c)}")


# ----------------------------------------------------------------------------
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("summary")
    p.add_argument("run")
    p = sub.add_parser("track")
    p.add_argument("run")
    p.add_argument("--filter", default="box200_80")
    p.add_argument("--fit", action="store_true")
    p.add_argument("--worst", type=int, default=8)
    p.add_argument("--source", default="auto", choices=("auto", "joints", "joints_fb"),
                   help="measured stream (auto: encoder feedback when recorded)")
    p = sub.add_parser("spawn")
    p.add_argument("run")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--min-hits", type=int, default=3)
    p = sub.add_parser("compare")
    p.add_argument("real")
    p.add_argument("twin")
    p.add_argument("--min-hits", type=int, default=3)
    p.add_argument("--match-s", type=float, default=1.0)
    args = ap.parse_args(argv)
    {"summary": cmd_summary, "track": cmd_track, "spawn": cmd_spawn,
     "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

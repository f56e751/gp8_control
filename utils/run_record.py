"""Run recorder for real-to-sim (cycle / piece mode, hardware and twin alike).

Opt-in: ``GP8_RUN_RECORD_DIR=<dir>`` (``scripts/cycle_demo.sh`` sets it to
``<log>/record`` unless ``GP8_RUN_RECORD=0``). One recording =

* ``events.jsonl``  one JSON object per line: ``{"ev": kind, "m": monotonic,
  "w": wall, ...}``. Kinds: ``meta`` (env, config), ``start`` (planner t0),
  ``tx`` / ``rx`` (link messages, no ping/pong), ``buf`` (one controller-buffer
  write: planner k, first tick, length, lead), ``buf_refused``, ``suc``
  (suction event as scheduled / fired), ``det`` (raw camera_debug snapshot),
  ``tracks`` (tracked queue, 10 Hz), ``landing`` (twin only).
* ``joints.f64``    float64 rows ``[monotonic, q1..q6]``: every measured joint
  state the backend received (``/joint_states`` on hardware, the stepper in
  the twin).
* ``cmd.f64``       float64 rows ``[monotonic due, q1..q6]``: every command
  sample the arm was given, at the time it takes effect (buffer: the tick's
  time; 2 ms stream: the send time). Gaps are held by the controller.

Everything is timed on this machine's ``time.monotonic()``; the planner's
time 0 is the ``start`` event's ``t0``. ``tools/real2sim.py`` reads it.
Writing runs on its own thread; the callers only append to a queue.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np

_CURRENT: "RunRecorder | None" = None
_LOCK = threading.Lock()

#: env prefixes copied into the meta event (what the run was configured with)
_ENV_PREFIXES = ("GP8_", "RT_", "GT_")


def current() -> "RunRecorder | None":
    """The process-wide recorder, or None when recording is off."""
    return _CURRENT


def start_from_env(**meta) -> "RunRecorder | None":
    """Start the process-wide recorder if ``GP8_RUN_RECORD_DIR`` is set (once)."""
    global _CURRENT
    with _LOCK:
        if _CURRENT is not None:
            return _CURRENT
        d = os.environ.get("GP8_RUN_RECORD_DIR", "").strip()
        if not d:
            return None
        _CURRENT = RunRecorder(Path(d).expanduser(), meta)
        return _CURRENT


def _default(x):
    """json.dumps fallback (the encoder itself stays in C: no per-float Python)."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, (set, frozenset)):
        return list(x)
    return repr(x)


class RunRecorder:
    FLUSH_S = 0.5
    TRACK_PERIOD_S = 0.1

    def __init__(self, out_dir: Path, meta: dict | None = None) -> None:
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._q: "queue.Queue" = queue.Queue()
        self._closed = False
        self._t_tracks = 0.0
        self._ev = open(self.dir / "events.jsonl", "a", encoding="utf-8")
        self._bin = {"joints": open(self.dir / "joints.f64", "ab"),
                     "cmd": open(self.dir / "cmd.f64", "ab")}
        self._rows: dict[str, list] = {"joints": [], "cmd": []}
        self._rows_lock = threading.Lock()
        env = {k: v for k, v in os.environ.items() if k.startswith(_ENV_PREFIXES)}
        self.event("meta", env=env, **(meta or {}))
        self._thread = threading.Thread(target=self._writer, name="run_record", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    def event(self, kind: str, **fields) -> None:
        if self._closed:
            return
        self._q.put({**fields, "m": time.monotonic(), "w": time.time(), "ev": kind})

    def joints(self, q, t_mono: float | None = None) -> None:
        if self._closed or q is None:
            return
        row = [time.monotonic() if t_mono is None else float(t_mono), *map(float, q)]
        with self._rows_lock:
            self._rows["joints"].append(row)

    def cmd_rows(self, t_first: float, dt: float, U) -> None:
        """Command samples U[i] taking effect at ``t_first + i*dt`` (monotonic)."""
        if self._closed:
            return
        U = np.asarray(U, float)
        if U.ndim != 2 or not len(U):
            return
        t = float(t_first) + dt * np.arange(len(U))
        with self._rows_lock:
            self._rows["cmd"].append(np.column_stack([t, U]))

    def cmd_sample(self, t_mono: float, q) -> None:
        if self._closed:
            return
        with self._rows_lock:
            self._rows["cmd"].append([float(t_mono), *map(float, q)])

    def tracks(self, queue_, conveyor, wall_now: float | None = None) -> None:
        """Snapshot of the tracked-object queue, at most every 0.1 s."""
        mono = time.monotonic()
        if self._closed or mono - self._t_tracks < self.TRACK_PERIOD_S:
            return
        self._t_tracks = mono
        tw = time.time() if wall_now is None else wall_now
        v = float(getattr(conveyor, "current", 0.0) or 0.0)
        dist = getattr(conveyor, "distance_m", None)
        objs = []
        for o in queue_.snapshot():
            try:
                y = float(o.y_at(tw, v, dist))
            except Exception:  # noqa: BLE001
                continue
            row = {"id": int(o.track_id), "cls": o.class_name,
                   "x": float(o.T_grasp_base[0, 3]), "y": y,
                   "z": float(o.T_grasp_base[2, 3]), "hits": int(getattr(o, "n_hits", 0) or 0),
                   "cc": bool(getattr(o, "center_confirmed", False)),
                   "conf": float(getattr(o, "conf", -1.0)),
                   "src": getattr(o, "src_track_id", None),
                   "sim": getattr(o, "sim_object_id", None)}
            bb = getattr(o, "base_bbox_grasp", None)
            if bb is not None:
                row["bbox"] = np.asarray(bb, float).tolist()
            objs.append(row)
        self.event("tracks", v=v, dist=dist, tw=tw, objs=objs)

    # ------------------------------------------------------------------
    def _flush_rows(self) -> None:
        with self._rows_lock:
            rows, self._rows = self._rows, {"joints": [], "cmd": []}
        for name, rr in rows.items():
            if not rr:
                continue
            parts = [np.atleast_2d(np.asarray(r, float)) for r in rr]
            np.vstack(parts).astype("<f8").tofile(self._bin[name])
            self._bin[name].flush()

    def _writer(self) -> None:
        last = time.monotonic()
        while True:
            try:
                item = self._q.get(timeout=self.FLUSH_S)
            except queue.Empty:
                item = None
            if item is not None:
                if item.get("ev") == "__close__":
                    break
                self._ev.write(json.dumps(item, default=_default) + "\n")
            if time.monotonic() - last >= self.FLUSH_S:
                self._ev.flush()
                self._flush_rows()
                last = time.monotonic()
        self._ev.flush()
        self._flush_rows()

    def close(self) -> None:
        if self._closed:
            return
        self.event("end")
        self._closed = True
        self._q.put({"ev": "__close__"})           # the writer drains, flushes, exits
        self._thread.join(timeout=5.0)
        self._ev.close()
        for f in self._bin.values():
            f.close()


def load(run_dir) -> dict:
    """Read a recording: ``{"events": [...], "joints": (N,7), "cmd": (M,7)}``."""
    d = Path(run_dir)
    events = []
    with open(d / "events.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except ValueError:
                    pass                          # a torn last line
    out = {"events": events}
    for name in ("joints", "cmd"):
        p = d / f"{name}.f64"
        a = np.fromfile(p, dtype="<f8") if p.exists() else np.zeros(0)
        a = a[: len(a) // 7 * 7].reshape(-1, 7)
        out[name] = a[np.argsort(a[:, 0], kind="stable")] if len(a) else a
    return out

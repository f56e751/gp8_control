"""Diagnostic CSV logger for PREDICTED-vs-ACTUAL robot motion timing.

Standalone and OPT-IN (env ``GP8_MOTION_LOG_DIR``). It is deliberately isolated
from the control logic so it can never change behaviour: ``TrajectoryController``
calls :meth:`on_command` once per queued trajectory and :meth:`on_sample` from its
joint-state callback — that is the entire contract. Every method is a no-op unless
an output directory is configured, so production runs pay nothing.

Why: the scheduler places intercepts using a PREDICTED move time (the trajectory's
planned duration). If the robot consistently takes LONGER than planned, that error
accumulates cycle over cycle. These CSVs expose it.

Two files are written under ``<out_dir>``:

  * ``motion_commands.csv`` — one row per queued trajectory: when it was queued,
    its PLANNED duration + target, and the ACTUAL wall time the robot took to reach
    that target. ``err_s = actual_dur_s - planned_dur_s`` is the per-command timing
    error; its running sum is the drift to chase. ``reached=0`` means the next
    command started before the robot settled at this one's target (chained / missed).
  * ``motion_samples.csv`` — the actual joint position over time (from
    ``/joint_states_urdf``), tagged with the in-flight command id + euclidean
    distance-to-target, so the real motion can be plotted against the plan.

Thread-safe: ``on_sample`` runs on the executor (joint-state) thread while
``on_command`` runs on the control thread; a lock guards the shared state + writes.
Rows are flushed on every write, so a hard ``Ctrl-C`` loses at most an in-flight
command's completion row (all samples and all completed commands are on disk).
"""

from __future__ import annotations

import csv
import datetime
import math
import os
import threading
import time


_CMD_HEADER = [
    "cmd_id", "skill", "iso_time", "t_queued", "planned_dur_s", "planned_end_t",
    "actual_dur_s", "actual_end_t", "err_s", "reached",
    *[f"start_j{i + 1}" for i in range(6)],
    *[f"target_j{i + 1}" for i in range(6)],
]
_SAMPLE_HEADER = ["t", "cmd_id", "dist_to_target", *[f"j{i + 1}" for i in range(6)]]


def _dist(a, b) -> float:
    n = min(len(a), len(b))
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(n)))


def _iso(t: float) -> str:
    return datetime.datetime.fromtimestamp(t).isoformat(timespec="milliseconds")


class MotionLogger:
    """CSV logger for queued-command timing vs. actual robot motion (see module doc).

    Construct with ``out_dir=None`` (the default when ``GP8_MOTION_LOG_DIR`` is
    unset) for a fully inert no-op instance.
    """

    def __init__(self, out_dir: "str | None", arrival_tol_rad: float = 0.02,
                 logger=None) -> None:
        self._lock = threading.Lock()
        self._logger = logger
        self._tol = float(arrival_tol_rad)
        self._cmd_id = 0
        self._pending: "dict | None" = None
        self._label = ""          # op label (skill name) tagged onto commands
        self._enabled = False
        self._cmd_f = self._sample_f = None
        self._cmd_w = self._sample_w = None
        if not out_dir:
            return
        try:
            out_dir = os.path.expanduser(out_dir)
            os.makedirs(out_dir, exist_ok=True)
            # Stamp filenames per run so successive launches don't overwrite each
            # other (each run -> its own pair of CSVs).
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self._cmd_f = open(
                os.path.join(out_dir, f"motion_commands_{stamp}.csv"), "w", newline="")
            self._sample_f = open(
                os.path.join(out_dir, f"motion_samples_{stamp}.csv"), "w", newline="")
            self._cmd_w = csv.writer(self._cmd_f)
            self._sample_w = csv.writer(self._sample_f)
            self._cmd_w.writerow(_CMD_HEADER)
            self._sample_w.writerow(_SAMPLE_HEADER)
            self._cmd_f.flush()
            self._sample_f.flush()
            self._enabled = True
            if logger is not None:
                logger.info(
                    f"MotionLogger ON -> {out_dir} "
                    f"(motion_commands.csv + motion_samples.csv)"
                )
        except OSError as e:
            self._enabled = False
            if logger is not None:
                logger.warn(f"MotionLogger disabled (could not open files: {e})")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_label(self, label: str) -> None:
        """Tag subsequent on_command() rows with this op label (e.g. skill name)."""
        if not self._enabled:
            return
        with self._lock:
            self._label = label or ""

    # ------------------------------------------------------------------
    def on_command(self, start_joint, target_joint, planned_dur_s: float) -> None:
        """Record that a trajectory was just queued (start, target, planned dur)."""
        if not self._enabled:
            return
        now = time.time()
        try:
            with self._lock:
                # A previous command still open means the next move started before
                # the robot settled at the prior target -> finalize it not-reached.
                if self._pending is not None:
                    self._write_cmd(self._pending, reached=False, actual_end=now)
                self._cmd_id += 1
                self._pending = {
                    "id": self._cmd_id,
                    "skill": self._label,
                    "t_queued": now,
                    "planned_dur": float(planned_dur_s),
                    "planned_end": now + float(planned_dur_s),
                    "start": [float(x) for x in start_joint][:6],
                    "target": [float(x) for x in target_joint][:6],
                    "arrived": False,
                }
        except Exception:
            pass  # diagnostics must never disturb control

    def on_sample(self, joints) -> None:
        """Record the current actual joint position; detect arrival at the target."""
        if not self._enabled or joints is None:
            return
        now = time.time()
        try:
            with self._lock:
                p = self._pending
                cmd_id = p["id"] if p is not None else -1
                dist_str = ""
                if p is not None:
                    dist = _dist(joints, p["target"])
                    dist_str = f"{dist:.5f}"
                    if not p["arrived"] and dist <= self._tol:
                        p["arrived"] = True
                        self._write_cmd(p, reached=True, actual_end=now)
                        self._pending = None
                self._sample_w.writerow(
                    [f"{now:.4f}", cmd_id, dist_str,
                     *[f"{float(j):.5f}" for j in list(joints)[:6]]]
                )
                self._sample_f.flush()
        except Exception:
            pass

    def close(self) -> None:
        """Flush a still-open command and close the files (optional; safe to skip)."""
        if not self._enabled:
            return
        try:
            with self._lock:
                if self._pending is not None:
                    self._write_cmd(self._pending, reached=False,
                                    actual_end=time.time())
                    self._pending = None
                for f in (self._cmd_f, self._sample_f):
                    if f is not None:
                        f.close()
        except Exception:
            pass
        self._enabled = False

    # ------------------------------------------------------------------
    def _write_cmd(self, p: dict, reached: bool, actual_end: float) -> None:
        """Write one finalized command row (called while holding the lock)."""
        actual_dur = actual_end - p["t_queued"]
        err = actual_dur - p["planned_dur"]
        self._cmd_w.writerow([
            p["id"], p.get("skill", ""), _iso(p["t_queued"]), f"{p['t_queued']:.4f}",
            f"{p['planned_dur']:.4f}", f"{p['planned_end']:.4f}",
            f"{actual_dur:.4f}", f"{actual_end:.4f}", f"{err:+.4f}",
            int(bool(reached)),
            *[f"{x:.5f}" for x in p["start"]],
            *[f"{x:.5f}" for x in p["target"]],
        ])
        self._cmd_f.flush()

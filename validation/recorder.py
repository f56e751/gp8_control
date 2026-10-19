"""Sim-to-real run recorder — identical on hardware and in the MuJoCo twin.

Attach once (``Sim2RealRecorder.from_env(app)`` in ``GP8App.setup()``, or
``instrument_runner`` for an RL runner) and it records, on the backend's own
clock (wall on hardware, sim time in a non-realtime twin):

* every 4 ms stream command (wraps ``RobotBackend._emit_sample``);
* every joint-state update (``RobotBackend.joint_state_listener``);
* suction requests / controller acks and the skill motion-op labels;
* belt speed + encoder distance, perception frames, and every per-track
  detection update (polled from the tracked-object queue at 50 Hz);
* one record per executed decision step: the observation in the *training*
  schema (``ObsSchema``) plus per-slot track identity/ages, the executed and
  selected actions, and begin/end times on both the backend and wall clocks.

Run directory layout (analysed by ``gp8_control.validation.analyze``)::

    meta.json            config / schema / env / git revision
    stream/chunk_*.npz   cmd_t, cmd_q (N,6), fb_t, fb_q (M,6)
    events.jsonl         suction, suction_ack, motion_op, belt, frame
    tracks.jsonl         per-track detection updates
    steps.jsonl          per-step observation + timing records

Data is flushed every couple of seconds by a background thread, so a crash or
Ctrl-C loses at most the last chunk. Nothing here imports rclpy.
"""

from __future__ import annotations

import atexit
import json
import os
import socket
import subprocess
import sys
import threading
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gp8_control.rl.common import build_observation
from gp8_control.validation.obs_schema import ObsSchema

ENV_ENABLE = "GP8_S2R_VALIDATE"
ENV_DIR = "GP8_S2R_DIR"
ENV_AUTO_ANALYZE = "GP8_S2R_AUTO_ANALYZE"
DEFAULT_ROOT = Path.home() / "gp8_s2r_runs"

FLUSH_PERIOD_S = 2.0
POLL_PERIOD_S = 0.02
BELT_RECORD_PERIOD_S = 0.05
IDLE_RECORD_PERIOD_S = 0.5


def env_enabled(default: bool = False) -> bool:
    raw = os.environ.get(ENV_ENABLE)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def default_run_dir(source: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return DEFAULT_ROOT / f"{stamp}_{source}"


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _git_revision() -> str | None:
    try:
        here = Path(__file__).resolve().parent
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=here, capture_output=True,
            text=True, timeout=2.0)
        return out.stdout.strip() or None
    except Exception:
        return None


def _bbox_xy(bbox) -> list | None:
    if not bbox:
        return None
    try:
        pts = np.asarray(bbox, dtype=float)
        return pts[:, :2].round(5).tolist()
    except Exception:
        return None


def _view_of(owner):
    """GP8App / SimRlRunner expose queue/traj_ctrl/... directly; RealRlRunner wraps an app."""
    return getattr(owner, "app", owner)


class Sim2RealRecorder:
    def __init__(
        self,
        run_dir,
        *,
        clock,
        source: str,
        schema: ObsSchema | None = None,
        auto_analyze: bool = True,
        analyze_args: tuple[str, ...] = (),
        meta: dict | None = None,
        background_poll: bool = True,
    ) -> None:
        self.run_dir = Path(run_dir).expanduser()
        # Wall-clock world polling is only meaningful when the world runs in
        # wall time. A non-realtime sim outruns it and mutates track fields
        # between the poller's reads (that skewed bbox-vs-encoder slopes to
        # ~0.79 in recorded sim runs) -- there, poll only synchronously at
        # each decision step (begin_step).
        self.background_poll = bool(background_poll)
        (self.run_dir / "stream").mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.source = str(source)
        self.schema = schema or ObsSchema()
        self.auto_analyze = bool(auto_analyze)
        self.analyze_args = tuple(analyze_args)
        self._meta = dict(meta or {})

        self._lock = threading.Lock()
        self._poll_lock = threading.Lock()
        self._cmd_t: list[float] = []
        self._cmd_q: list[tuple] = []
        self._fb_t: list[float] = []
        self._fb_q: list[tuple] = []
        self._event_buf: list[dict] = []
        self._track_buf: list[dict] = []
        self._chunk_idx = 0
        self._n_cmd = 0
        self._n_fb = 0
        self._n_steps = 0
        self._last_fb_t: float | None = None

        self._events_f = (self.run_dir / "events.jsonl").open("a", encoding="utf-8")
        self._tracks_f = (self.run_dir / "tracks.jsonl").open("a", encoding="utf-8")
        self._steps_f = (self.run_dir / "steps.jsonl").open("a", encoding="utf-8")

        self._view = None
        self._track_seen: dict[int, float] = {}
        self._last_frame_receipt: float | None = None
        self._last_belt_t = -1e18

        self._t_start = self.clock.time()
        self._wall_start = time.time()
        self._closed = False
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._background, name="s2r_recorder", daemon=True)
        self._write_meta()
        self._thread.start()
        atexit.register(self.close)

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls, owner, *, source: str, enabled: bool | None = None,
                 run_dir=None) -> "Sim2RealRecorder | None":
        """Build + attach when ``GP8_S2R_VALIDATE`` (or ``enabled``) says so."""
        if not (env_enabled() if enabled is None else bool(enabled)):
            return None
        view = _view_of(owner)
        run_dir = run_dir or os.environ.get(ENV_DIR) or default_run_dir(source)
        auto = os.environ.get(ENV_AUTO_ANALYZE, "1").strip().lower() not in (
            "0", "false", "no", "off")
        rec = cls(
            run_dir,
            clock=view.traj_ctrl.clock,
            source=source,
            schema=ObsSchema.from_env(),
            auto_analyze=auto,
            meta=collect_meta(view),
        )
        rec.attach(view)
        return rec

    def attach(self, view) -> None:
        """Wrap the backend seams and start polling ``view``'s queue/belt/world."""
        self._view = view
        self.attach_backend(view.traj_ctrl)

    def attach_backend(self, backend) -> None:
        orig_emit = backend._emit_sample
        orig_suction = backend._set_suction
        orig_motion_op = backend.set_motion_op
        clock = self.clock
        lock = self._lock

        def _emit_sample(positions):
            t = clock.time()
            orig_emit(positions)
            q = tuple(float(x) for x in positions)
            with lock:
                self._cmd_t.append(t)
                self._cmd_q.append(q)

        def _set_suction(on, requested_at):
            orig_suction(on, requested_at)
            self._event({"type": "suction", "t": float(requested_at), "on": bool(on)})

        def _set_motion_op(label):
            self._event({"type": "motion_op", "t": clock.time(), "label": str(label)})
            return orig_motion_op(label)

        # Instance attributes shadow the class methods the stream engine calls.
        backend._emit_sample = _emit_sample
        backend._set_suction = _set_suction
        backend.set_motion_op = _set_motion_op
        backend.joint_state_listener = self._on_joint_state
        backend.suction_ack_listener = self._on_suction_ack

    # ------------------------------------------------------------------
    # high-rate callbacks (keep these tiny)
    # ------------------------------------------------------------------
    def _on_joint_state(self, q) -> None:
        t = self.clock.time()
        qt = tuple(float(x) for x in q)
        with self._lock:
            self._fb_t.append(t)
            self._fb_q.append(qt)
            self._last_fb_t = t

    def _on_suction_ack(self, on, requested_at, acked_at) -> None:
        self._event({"type": "suction_ack", "on": bool(on),
                     "requested_at": float(requested_at), "acked_at": float(acked_at)})

    def _event(self, rec: dict) -> None:
        with self._lock:
            self._event_buf.append(rec)

    # ------------------------------------------------------------------
    # decision steps
    # ------------------------------------------------------------------
    def begin_step(
        self,
        owner,
        *,
        kind: str,
        executed: dict | None = None,
        selected: dict | None = None,
        pending_indices: tuple[int, int] | None = None,
        remaining_time_frac: float = 1.0,
        extra: dict | None = None,
    ) -> dict:
        view = _view_of(owner)
        t = self.clock.time()
        wall = time.time()
        # Sync poll so every decision has the freshest tracks/belt on record even
        # when a non-realtime sim outruns the 50 Hz wall-clock poller.
        self._poll_world_safe()
        rec: dict[str, Any] = {
            "kind": kind, "t_begin": t, "wall_begin": wall,
            "executed": executed, "selected": selected,
        }
        try:
            refresh = getattr(owner, "_refresh_truth_tracks", None)
            if callable(refresh):
                refresh(t)
            objects = list(view.queue._objects)[: self.schema.max_objects]
            joints = view.traj_ctrl.current_joints
            joints_arr = None if joints is None else np.asarray(joints, dtype=float)
            ee = (np.zeros(3) if joints_arr is None
                  else view.robot.forward_kinematics(joints_arr[:6])[:3, 3])
            v = float(view.conveyor.current)
            obs = build_observation(
                objects=objects,
                max_objects=self.schema.max_objects,
                include_eta=False,
                joints=joints_arr,
                ee_xyz=ee,
                pending_indices=pending_indices or (self.schema.max_objects, 0),
                belt_speed=v,
                remaining_time_frac=remaining_time_frac,
                y_now_for=lambda o: view.ctx.object_y_now(o, t, v),
                bbox_observation=self.schema.bbox_observation,
                include_suction_p=self.schema.include_suction_p,
            )
            rec["obs"] = obs.tolist()
            rec["slots"] = [
                {
                    "track_id": getattr(o, "track_id", None),
                    "class_name": getattr(o, "class_name", None),
                    "conf": float(getattr(o, "conf", -1.0)),
                    "detect_time": getattr(o, "detect_time", None),
                    "bbox_detect_time": getattr(o, "bbox_detect_time", None),
                    "sim_object_id": getattr(o, "sim_object_id", None),
                }
                for o in objects
            ]
            rec["queue_len"] = len(view.queue._objects)
            rec["belt_speed"] = v
            with self._lock:
                last_fb = self._last_fb_t
            rec["joint_age"] = None if last_fb is None else float(t - last_fb)
        except Exception as exc:   # never break the control loop for a log
            rec["obs_error"] = repr(exc)
        if extra:
            rec.update(_jsonable(extra))
        return rec

    def end_step(self, rec: dict, **fields) -> None:
        rec["t_end"] = self.clock.time()
        rec["wall_end"] = time.time()
        rec.update(_jsonable(fields))
        rec["step"] = self._n_steps
        self._n_steps += 1
        try:
            self._steps_f.write(json.dumps(_jsonable(rec), separators=(",", ":")) + "\n")
            self._steps_f.flush()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # background: world polling + periodic flush
    # ------------------------------------------------------------------
    def _background(self) -> None:
        next_flush = time.monotonic() + FLUSH_PERIOD_S
        while not self._stop.is_set():
            if self.background_poll:
                self._poll_world_safe()
            if time.monotonic() >= next_flush:
                self.flush()
                next_flush = time.monotonic() + FLUSH_PERIOD_S
            self._stop.wait(POLL_PERIOD_S)

    def _poll_world_safe(self) -> None:
        try:
            with self._poll_lock:
                self._poll_world()
        except Exception:
            pass

    def _poll_world(self) -> None:
        view = self._view
        if view is None:
            return
        now = self.clock.time()
        belt = getattr(view, "conveyor", None)
        if belt is not None and now - self._last_belt_t >= BELT_RECORD_PERIOD_S:
            last_msg = getattr(belt, "_last_msg_time", None)
            dist = belt.distance_at(now) if hasattr(belt, "distance_at") else None
            self._event({
                "type": "belt", "t": now, "speed": float(belt.current),
                "distance_m": None if dist is None else float(dist),
                "msg_age": None if last_msg is None else float(time.time() - last_msg),
            })
            self._last_belt_t = now
        world = getattr(view, "world", None)
        snap = world.latest_snapshot() if world is not None else None
        if snap:
            receipt = snap.get("receipt_time")
            if receipt is not None and receipt != self._last_frame_receipt:
                self._last_frame_receipt = receipt
                self._event({
                    "type": "frame", "t_seen": now, "receipt_time": float(receipt),
                    "n_det": len(snap.get("detections") or []),
                    "classes": sorted({str(d.get("class")) for d in snap.get("detections") or []}),
                    "belt_mps": snap.get("belt_mps"),
                    "perception_delay_s": snap.get("perception_delay_s"),
                    "applied_delay_s": snap.get("applied_delay_s"),
                })
        queue = getattr(view, "queue", None)
        if queue is None:
            return
        live = set()
        for obj in list(queue._objects):
            tid = getattr(obj, "track_id", None)
            if tid is None:
                continue
            live.add(tid)
            bdt = getattr(obj, "bbox_detect_time", None) or getattr(obj, "detect_time", None)
            if bdt is None or self._track_seen.get(tid) == bdt:
                continue
            first = tid not in self._track_seen
            self._track_seen[tid] = bdt
            T = obj.T_grasp_base
            self._track_buf.append({
                "t_poll": now,
                "track_id": int(tid),
                "first": bool(first),
                "class_name": obj.class_name,
                "class_votes": _jsonable(getattr(obj, "class_votes", {})),
                "conf": float(obj.conf),
                "t_bbox": float(bdt),
                "enc_bbox": getattr(obj, "bbox_encoder_distance_m", None),
                "anchor_t": float(obj.detect_time),
                "anchor_enc": obj.encoder_distance_m,
                "anchor_xyz": [float(T[0, 3]), float(T[1, 3]), float(T[2, 3])],
                "bbox": _bbox_xy(getattr(obj, "base_bbox_grasp", None)),
                "cam": _jsonable(getattr(obj, "cam_pos", None)),
                "sim_object_id": getattr(obj, "sim_object_id", None),
            })
        for tid in [k for k in self._track_seen if k not in live]:
            self._track_buf.append({"t_poll": now, "track_id": int(tid), "gone": True})
            del self._track_seen[tid]

    def flush(self) -> None:
        with self._lock:
            cmd_t, self._cmd_t = self._cmd_t, []
            cmd_q, self._cmd_q = self._cmd_q, []
            fb_t, self._fb_t = self._fb_t, []
            fb_q, self._fb_q = self._fb_q, []
            events, self._event_buf = self._event_buf, []
        with self._poll_lock:
            tracks, self._track_buf = self._track_buf, []
        if cmd_t or fb_t:
            path = self.run_dir / "stream" / f"chunk_{self._chunk_idx:05d}.npz"
            np.savez(
                path,
                cmd_t=np.asarray(cmd_t, dtype=float),
                cmd_q=np.asarray(cmd_q, dtype=float).reshape(-1, 6),
                fb_t=np.asarray(fb_t, dtype=float),
                fb_q=np.asarray(fb_q, dtype=float).reshape(-1, 6),
            )
            self._chunk_idx += 1
            self._n_cmd += len(cmd_t)
            self._n_fb += len(fb_t)
        for ev in events:
            self._events_f.write(json.dumps(_jsonable(ev), separators=(",", ":")) + "\n")
        for tr in tracks:
            self._tracks_f.write(json.dumps(_jsonable(tr), separators=(",", ":")) + "\n")
        self._events_f.flush()
        self._tracks_f.flush()

    # ------------------------------------------------------------------
    # shutdown
    # ------------------------------------------------------------------
    def _write_meta(self, **extra) -> None:
        meta = dict(self._meta)
        meta.update({
            "source": self.source,
            "schema": self.schema.to_dict(),
            "t_start": self._t_start,
            "wall_start": self._wall_start,
            "host": socket.gethostname(),
            "git_rev": meta.get("git_rev") or _git_revision(),
        })
        meta.update(extra)
        self._meta = meta
        tmp = self.run_dir / "meta.json.tmp"
        tmp.write_text(json.dumps(_jsonable(meta), indent=2), encoding="utf-8")
        tmp.replace(self.run_dir / "meta.json")

    def close(self, *, analyze: bool | None = None) -> Path | None:
        """Flush everything; optionally launch the analyzer detached. Idempotent."""
        if self._closed:
            return None
        self._closed = True
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._poll_world_safe()
        self.flush()
        self._write_meta(
            t_end=self.clock.time(), wall_end=time.time(),
            n_cmd=self._n_cmd, n_fb=self._n_fb, n_steps=self._n_steps)
        for f in (self._events_f, self._tracks_f, self._steps_f):
            try:
                f.close()
            except Exception:
                pass
        do_analyze = self.auto_analyze if analyze is None else bool(analyze)
        if do_analyze:
            return self._launch_analyzer()
        return None

    def _launch_analyzer(self) -> Path | None:
        """Run the analyzer in its own session so a launch-wide SIGINT can't kill it."""
        log = self.run_dir / "analyze.log"
        pkg_parent = str(Path(__file__).resolve().parents[2])
        env = dict(os.environ)
        env["PYTHONPATH"] = pkg_parent + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        try:
            with log.open("w", encoding="utf-8") as fh:
                subprocess.Popen(
                    [sys.executable, "-m", "gp8_control.validation.analyze",
                     str(self.run_dir), *self.analyze_args],
                    stdout=fh, stderr=subprocess.STDOUT, env=env,
                    start_new_session=True,
                )
            print(f"[s2r] run saved to {self.run_dir}; analysis running in the "
                  f"background -> {self.run_dir / 'report.md'} (log: {log})", flush=True)
            return self.run_dir / "report.md"
        except Exception as exc:
            print(f"[s2r] could not launch analyzer ({exc}); run it manually: "
                  f"python -m gp8_control.validation.analyze {self.run_dir}", flush=True)
            return None


def collect_meta(view) -> dict:
    cfg = getattr(view, "cfg", None)
    meta: dict[str, Any] = {
        "backend": type(getattr(view, "traj_ctrl", None)).__name__,
        "env": {k: v for k, v in os.environ.items()
                if k.startswith(("GP8_", "ROS_DOMAIN"))},
    }
    if cfg is not None:
        try:
            meta["config"] = _jsonable(cfg)
        except Exception as exc:
            meta["config_error"] = repr(exc)
    core = getattr(getattr(view, "traj_ctrl", None), "_core", None)
    if core is not None:
        try:
            meta["sim_config"] = _jsonable({
                k: v for k, v in vars(core.cfg).items() if k not in ("camera",)
            })
        except Exception:
            pass
    return meta


def instrument_runner(runner, rec: Sim2RealRecorder, *, episode_seconds: float | None = None):
    """Record every ``runner.step`` of a Real/Sim RL runner (instance-level wrap).

    ``episode_seconds`` only feeds the recorded observation's remaining-time
    feature (sim training semantics); None keeps 1.0 like ``RealRlRunner``.
    """
    orig_step = runner.step
    clock = rec.clock
    t0 = clock.time()

    def _describe(action) -> dict | None:
        if action is None or getattr(action, "target", None) is None:
            return None
        tgt = action.target
        return {"skill": action.skill_name, "track_id": getattr(tgt, "track_id", None),
                "class_name": getattr(tgt, "class_name", None)}

    last_idle = [-1e18]

    def step(action):
        pending = runner.pending_action
        if pending is None and action is None:
            # Idle decision: only advances TIME_STEP. Keep one per IDLE_RECORD_PERIOD_S.
            now = clock.time()
            if now - last_idle[0] < IDLE_RECORD_PERIOD_S:
                return orig_step(action)
            last_idle[0] = now
        remaining = 1.0
        if callable(getattr(runner, "remaining_time_frac", None)):
            remaining = runner.remaining_time_frac()   # what the runner feeds its policy
        elif episode_seconds:
            remaining = max(0.0, 1.0 - (clock.time() - t0) / float(episode_seconds))
        rec_step = rec.begin_step(
            runner,
            kind="rl",
            executed=_describe(pending),
            selected=_describe(action),
            pending_indices=runner._pending_indices(),
            remaining_time_frac=remaining,
        )
        reward, info = orig_step(action)
        rec.end_step(
            rec_step,
            reward=float(reward),
            info={k: info.get(k) for k in (
                "executed", "execution_success", "execution_detail",
                "executed_skill", "executed_track_id", "dropped_stale_pending")},
        )
        return reward, info

    runner.step = step
    return runner

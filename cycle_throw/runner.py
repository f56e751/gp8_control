"""Cycle-throw mode: the vendored grasp-and-throw cycle planner drives the arm.

``GP8App`` runs this instead of the per-object skill loop when
``Config.MODE == "cycle"``. The vendored planner (third_party/grasp_and_throw)
is HW-validated, so it runs unchanged in a child process (planner_link.py) and
owns the whole object stream: next object, cycle optimisation (A contact ->
throw A -> B contact), and the 2 ms commands. The app side here:

  * feeds it the tracked objects (perception + belt dead reckoning, as usual),
  * streams its commands sample by sample through ``RobotBackend._emit_sample``
    at k * 2 ms. Gaps hold the last sample. A 2 ms increment window keeps
    catch-up after a late wake under the per-cycle controller limit
    (teammate's 11-03 CODE 112 fix),
  * fires suction on the SAMPLE clock (OFF moved earlier by the valve lead).

Push-routed objects (``skill_for(obj) == "push"``, e.g. metal) are never shown
to the planner. They are handled in PUSH SLOTS between planner segments. A slot
splices only at rest, the one place where it is safe: there the planner's
commands are constant for longer than the controller filter (0.32 s), so its
command history equals what the controller holds. A slot runs:
  1. drain: hide every object, so the planner throws what it holds and ends its
     segment at its wait point; wait until its commands are idle and settled,
  2. pause the 2 ms stream and run the app's PushSkill (no chain move),
  3. return to the exact last planner command (all six joints) and hold it
     >= GP8_CYCLE_SETTLE_S before showing objects again. The planner then
     starts its next segment from the pose it believes it is at.

The planner can run on another machine: GP8_CYCLE_PLANNER_HOST=user@host
starts planner_link.py there over ssh (robot6; the robot PC is too slow to
plan in real time). The two clocks are reconciled by ping/pong inside the
link, so the app's monotonic clock stays the only timebase.

Command output (hardware): samples are written by controller tick into
motoman_controllers/SampleBuffer (driver branch feature/sample-buffer), the
teammate's HW-validated path. The controller puts out sample k at its own 2 ms
tick, so no Python loop paces anything. GP8_CYCLE_OUTPUT=stream (and the MuJoCo
twin, which has no buffer) use the app's own 2 ms stream instead.

It works on both backends through the RobotBackend seam. Hardware needs the
driver at 500 Hz (GP8_STREAM_HZ=500, see utils/stream_rate.py). The MuJoCo twin
runs it in realtime with the identified controller filter (sim setup in app.py).
"""

from __future__ import annotations

import collections
import json
import math
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from gp8_control.tracking.object_queue import is_established
from gp8_control.trajectory.trajectory_primitive import trajectory
from gp8_control.utils.clock import WALL_CLOCK
from gp8_control.utils.stream_rate import STREAM_HZ

# --- the planner's model set (vendored env.sh / README sec. 3 and 7) ---
# GT_BELT / GT_BIN_X / GT_BIN_Z pick the set of learned models; they are
# forwarded to the planner process, which refuses a combination it has no set
# for (env.sh). Defaults = the 11-19 real-robot setting.
PLANNER_DT = 0.002                    # command sample period [s]
PLANNER_BELT_SPEED = float(os.environ.get("GT_BELT", "") or 0.12)   # plan belt speed [m/s]
CYCLE_BIN_X = float(os.environ.get("GT_BIN_X", "") or 1.35)        # bin distance [m]
CYCLE_BIN_Z = float(os.environ.get("GT_BIN_Z", "") or -0.11)       # bin opening height [m]
PLANNER_SET_ENV = ("GT_BELT", "GT_BIN_X", "GT_BIN_Z")
#: class -> bin index (cyc2.BINS order: y = +0.33, 0, -0.33).
CYCLE_BIN_INDEX = {"metal": 0, "paper": 1, "transparent": 2}
#: aim points of the three bins in the base frame [m] (README sec. 3).
CYCLE_BINS_XYZ = {"metal": (CYCLE_BIN_X, 0.33, CYCLE_BIN_Z), "paper": (CYCLE_BIN_X, 0.0, CYCLE_BIN_Z),
                  "transparent": (CYCLE_BIN_X, -0.33, CYCLE_BIN_Z)}
#: rest pose of the planner (robot sign = this package's joint order/sign):
#: TCP (0.40, 0, 0.12). Same value as the vendored rt_dry.py.
CYCLE_HOME = (0.0, 0.28890832, -0.52880029, 0.0, -0.75308772, -1.13446401)
#: planner visibility window on the belt (rt_dry: y <= 1.30 m, >= -0.9 m).
DEFAULT_VIS_Y = (-0.9, 1.30)

_PLANNER_REL = ("third_party", "grasp_and_throw", "planner")
# rt_cont log lines that bracket a segment (the arm leaves / reaches the wait point).
_SEG_START = re.compile(r"^segment \d+: object ")
_SEG_END = re.compile(r"^segment \d+ ends: .* at (-?[\d.]+)\s*$")


def find_planner_dir() -> Path:
    """The vendored planner dir (source tree; it is not colcon-installed)."""
    env = os.environ.get("GP8_CYCLE_PLANNER_DIR")
    here = Path(__file__).resolve()
    candidates = [Path(env)] if env else []
    candidates.append(here.parents[1].joinpath(*_PLANNER_REL))
    for parent in here.parents:
        if parent.name == "install":
            candidates.append(parent.parent.joinpath("src", "gp8_control", *_PLANNER_REL))
            break
    candidates.append(Path.home().joinpath("ros2_ws", "src", "gp8_control", *_PLANNER_REL))
    for c in candidates:
        if (c / "rt_cont.py").is_file():
            return c
    raise FileNotFoundError(
        "cycle planner not found; tried:\n  " + "\n  ".join(map(str, candidates))
        + "\nRun from the source tree, set GP8_CYCLE_PLANNER_DIR, and check that "
        "`git lfs pull` fetched planner/logs/nnws/w2_n2c.npz.")


def cycle_sim_config(cfg=None):
    """MuJoCo twin set up for the planner: its three class bins at x = 1.35 m
    and the identified controller filter on the servos (its commands are
    pre-distorted for it; GP8_SIM_SERVO_FILTER overrides). Belt height stays
    the app's (``cfg.GRASP_Z``), so PushSkill's absolute PUSH_HEIGHT still meets
    the belt; the runner tells the planner the real object top instead
    (``sim_object_top``). Belt speed: the plan's (GT_BELT) unless
    GP8_SIM_BELT_SPEED is set.
    Classes that ``cfg.SKILL_BY_CLASS`` routes to push also get their push bin
    (``push_<class>``, same as bins_from_config)."""
    from gp8_control.backends.mujoco_sim import SimBin, SimConfig

    bins = [SimBin(cls, "throw", x, y, z, 0.15) for cls, (x, y, z) in CYCLE_BINS_XYZ.items()]
    if cfg is not None and not (cfg.FORCE_SKILL and cfg.FORCE_SKILL != "push"):
        from gp8_control.skills.push_skill import PUSH_BIN_TARGET_MAP
        for cls, skill in dict(cfg.SKILL_BY_CLASS).items():
            xyz = PUSH_BIN_TARGET_MAP.get(cls)
            if (cfg.FORCE_SKILL or skill) == "push" and xyz is not None:
                bins.append(SimBin(f"push_{cls}", "push", *map(float, xyz), 0.20))
    sim_cfg = SimConfig(
        bins=tuple(bins),
        servo_filter=os.environ.get("GP8_SIM_SERVO_FILTER", "box200_80"),
    )
    if not os.environ.get("GP8_SIM_BELT_SPEED"):
        sim_cfg.belt_speed = PLANNER_BELT_SPEED
    if cfg is not None:
        sim_cfg.grasp_z = float(cfg.GRASP_Z)
    return sim_cfg


def sim_object_top(sim_cfg) -> float:
    """Top face of a twin box in the base frame (its centre sits at grasp_z)."""
    return float(sim_cfg.grasp_z) + float(sim_cfg.object_half_z)


def _child_env_keys() -> list:
    """Environment the planner process gets from the app: the model-set and
    planner (RT_*) settings env.sh honours, and the piece link's GP8_PIECE_*."""
    fixed = ("GP8_THR_FILTER",) + PLANNER_SET_ENV
    return [k for k in os.environ
            if k in fixed or k.startswith("RT_") or k.startswith("GP8_PIECE_")]


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key, "").strip()
    return float(raw) if raw else float(default)


def _sleep_until(t_mono: float, spin_s: float) -> None:
    """Wall-clock wait that does not wake late: sleep until spin_s before the
    deadline, then yield-spin (time.sleep(0) releases the GIL)."""
    r = t_mono - time.monotonic()
    if r > spin_s:
        time.sleep(r - spin_s)
    while time.monotonic() < t_mono:
        time.sleep(0)


class _BufferOutput:
    """Planner commands into the controller's SampleBuffer by tick (hardware).

    Mirrors the teammate's hw_world bridge: planner sample k = controller tick
    k + shift; pieces only append (an overlap is dropped; a gap is held by the
    controller itself); a late piece is cut to the first tick still writable and
    the controller's own splice/limit check decides; any refusal stops the run.
    Suction events are queued by tick and fired by the BufferClient."""

    CTRL_MIN_LEAD = 3         # the controller's min_lead (sample_buffer.yaml)

    def __init__(self, runner, tc) -> None:
        self.r, self.tc, self.buf = runner, tc, tc.buf
        self.shift = 0
        self.k_written: int | None = None   # last planner index written
        self.last_row: np.ndarray | None = None
        self.pending_ev: list[tuple[int, bool]] = []
        self.min_lead_s: float | None = None
        # Round trips write -> controller ack, measured on every write: a write
        # must start at least the delivery time ahead of the controller's tick.
        self._rtts: collections.deque = collections.deque(maxlen=200)
        self._q: "queue.Queue" = queue.Queue()
        threading.Thread(target=self._writer, name="cycle_buf_writer", daemon=True).start()

    # planner time base -------------------------------------------------
    def start(self) -> float:
        """Planner t = 0 at the next tick; returns that tick's monotonic time."""
        k0 = self.buf.now_index() + 1
        self.shift = k0
        return self.buf.tick_mono(k0)

    def rebase(self, t0: float) -> None:
        """After a re-activation (its tick restarted at 0): map planner time again."""
        k_now = self.buf.now_index() + 1
        self.shift = k_now - int(round((self.buf.tick_mono(k_now) - t0) / PLANNER_DT))
        self.k_written, self.pending_ev = None, []

    @property
    def lead_ticks(self) -> int:
        """Ticks ahead of "now" a write can safely start: the controller's own
        min_lead plus the measured write round trip (95th percentile, x1.5;
        a write has not arrived later than its ack). 40 ticks until measured."""
        if len(self._rtts) < 5:
            return 40
        rtt = float(np.percentile(np.asarray(self._rtts), 95))
        return self.CTRL_MIN_LEAD + int(math.ceil(1.5 * rtt / PLANNER_DT)) + 1

    # commands ----------------------------------------------------------
    def commit(self, k: int, U: np.ndarray) -> None:
        self._q.put((int(k), U))

    def _writer(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            try:
                self._write(*item)
            except Exception as e:  # noqa: BLE001 - a failed write must stop the run
                self.r._fault = self.r._fault or f"buffer write failed: {e!r}"
            finally:
                self._q.task_done()

    def _write(self, k: int, U: np.ndarray) -> None:
        r = self.r
        if r._fault is not None or r._paused.is_set():
            return
        if self.k_written is not None and k <= self.k_written:     # overlap: never rewrite
            U, k = U[self.k_written + 1 - k:], self.k_written + 1
        now = self.buf.now_index()
        # now_index() trusts the last progress message only so far ahead; when
        # progress runs late, the controller is further on than it says.
        now += max(0, int(math.floor(self.buf.index_at(time.monotonic()))) - now)
        kn = now + self.lead_ticks                                 # first tick safely writable
        start = k + self.shift
        if start < kn:
            late = kn - start
            r.stats["late_pieces"] += 1
            r.log.warn(f"[cycle] late command piece: first {late} of {len(U)} samples "
                       f"({late * PLANNER_DT * 1e3:.0f} ms) already past; cut")
            if r.stop_on_late:
                r._fault = r._fault or "late command piece (GP8_CYCLE_STOP_ON_LATE=1)"
                return
            U, k, start = U[late:], k + late, kn
        if not len(U):
            return
        # A gap before the piece is filled with the held command by the
        # controller itself (sample_buffer.cpp: "Gap [e, s) ... hold value"), so
        # the piece goes out at its own start, with its own lead.
        Q, first = U, start
        t_w = time.monotonic()
        res = self.buf.write(first, Q, timeout=0.25)
        if res == 0:
            self._rtts.append(time.monotonic() - t_w)
        if res != 0:
            from gp8_control.controllers.sample_buffer import RES_NAMES
            r._fault = r._fault or (f"controller refused the piece at tick {first}: "
                                    f"{RES_NAMES.get(res, res)}; buffer stopped")
            self.stop()
            return
        lead = (first - now) * PLANNER_DT
        self.min_lead_s = lead if self.min_lead_s is None else min(self.min_lead_s, lead)
        self.k_written = k + len(U) - 1
        self.last_row = np.asarray(U[-1], float).copy()

    # suction -----------------------------------------------------------
    def suction(self, t_plan: float, on: bool) -> None:
        tick = int(round(t_plan / PLANNER_DT)) + self.shift
        now = self.buf.now_index()
        self.pending_ev = [e for e in self.pending_ev if e[0] > now] + [(tick, bool(on))]
        self.buf.set_events(now + 1, self.pending_ev)

    # state -------------------------------------------------------------
    def idle(self, settle_s: float) -> bool:
        """Everything written has played out and the command has been held
        unchanged for ``settle_s``."""
        if self._q.unfinished_tasks or not self.buf.ready():
            return False
        end = self.buf.buffer_end
        if not self.buf.holding or end is None:
            return False
        return time.monotonic() >= self.buf.tick_mono(end) + settle_s

    def last_command(self) -> np.ndarray | None:
        c = self.buf.command
        return np.asarray(c, float) if c else self.last_row

    def stop(self) -> None:
        try:
            self.buf.stop()
        except Exception:  # noqa: BLE001
            pass


class CycleThrowRunner:
    """Owns the arm while the app is in cycle mode (see module docstring)."""

    #: child-process script in this directory (piece mode uses piece_link.py)
    LINK = "planner_link.py"

    def __init__(self, app, *, skill_for=None, push_handler=None,
                 top_z: float | None = None) -> None:
        """``skill_for(obj) -> "throw" | "push" | ...`` routes objects (None =
        every object with a planner bin is thrown). ``push_handler(obj) -> str``
        runs one push with the arm at rest and returns a short result; it is
        required for push-routed objects to be handled at all."""
        self.app = app
        self.skill_for = skill_for
        self.push_handler = push_handler
        self.cfg = app.cfg
        self.backend = app.traj_ctrl
        self.log = app._node.get_logger()
        if getattr(self.backend, "clock", WALL_CLOCK) is not WALL_CLOCK:
            raise RuntimeError("cycle mode needs a wall-clock backend (sim: GP8_SIM_REALTIME=1)")

        self.planner_dir = find_planner_dir()
        self.release_lead = _env_float("GP8_CYCLE_RELEASE_LEAD", self.cfg.RELEASE_LEAD)
        self.vis_y = (_env_float("GP8_CYCLE_VIS_Y_MIN", DEFAULT_VIS_Y[0]),
                      _env_float("GP8_CYCLE_VIS_Y_MAX", DEFAULT_VIS_Y[1]))
        # Object top height sent to the planner: unset = planner default 0.02 m
        # (the HW-validated setting); a number = that height; "track" = the
        # tracked grasp Z.
        self.top_z = os.environ.get("GP8_CYCLE_TOP_Z", "").strip().lower()
        if not self.top_z and top_z is not None:    # e.g. the twin's real box top
            self.top_z = repr(float(top_z))
        self.spin_s = _env_float("GP8_STREAM_SPIN_S", 0.0006)
        # A late command piece stops the run (GP8_CYCLE_STOP_ON_LATE=0: warn only,
        # still never jumps: an over-limit step stops the stream regardless).
        self.stop_on_late = os.environ.get("GP8_CYCLE_STOP_ON_LATE", "1") not in ("0", "false", "no")
        # Per-2 ms-cycle increment limit [rad]: maxIncrement per 2 ms cycle equals
        # the datasheet speed (utils/stream_rate.py). Cycle mode always streams
        # at 2 ms, so this does not depend on GP8_STREAM_HZ.
        self.inc = np.asarray(app.robot.velocity_limits, float) * PLANNER_DT

        self.home = np.asarray(CYCLE_HOME, float)
        self.t0: float | None = None            # planner time 0 (monotonic)
        self._lock = threading.Lock()
        self._cmds: dict[int, np.ndarray] = {}  # sample k -> command (robot sign)
        self._suction: list[tuple[float, bool]] = []   # (fire time [planner s], on)
        self._k_sent = -1
        self._k_last_cmd = -1
        self._halt = threading.Event()
        self._fault: str | None = None
        self._child: subprocess.Popen | None = None
        self._send_lock = threading.Lock()      # rx thread (pong) and main thread write
        self._stderr_tail: collections.deque = collections.deque(maxlen=12)
        # Remote planner (robot6): user@host; empty = run it on this machine.
        self.planner_host = os.environ.get("GP8_CYCLE_PLANNER_HOST", "").strip()
        self.remote_dir = os.environ.get("GP8_CYCLE_REMOTE_DIR",
                                         "/PublicSSD/iitp/demo/gp8_control").rstrip("/")
        self._ready = threading.Event()
        self._ready_msg: dict | None = None
        self._done = threading.Event()
        self.stats = collections.Counter()
        self.max_lag = 0.0

        # --- push slots ---
        # Start draining when a push object is this close (s) to entering reach.
        self.push_lead = _env_float("GP8_CYCLE_PUSH_LEAD", 4.0)
        # Hold an unchanged command this long (s) at every splice: > the 0.32 s
        # controller filter, so the planner's assumed history is the real one.
        self.settle_s = _env_float("GP8_CYCLE_SETTLE_S", 0.4)
        self.drain_timeout = _env_float("GP8_CYCLE_DRAIN_TIMEOUT", 10.0)
        self.hide = False                       # show no object to the planner
        self._seg_active = False                # planner is inside a segment
        self._seg_end_t: float | None = None    # planner time its last segment comes to rest
        # After hiding, wait at least this long (s) before calling the planner
        # idle: a segment chosen just before the hide is logged only after its
        # first reach solve.
        self.min_drain_s = _env_float("GP8_CYCLE_MIN_DRAIN_S", 1.5)
        self.push_max = int(_env_float("GP8_CYCLE_PUSH_MAX", 3))   # pushes per slot
        self._paused = threading.Event()        # emitter stops sending (push owns the arm)
        self._pause_ack = threading.Event()
        self._resume_q: np.ndarray | None = None
        self._last_sent = self.home.copy()      # last command the emitter sent
        self._t_last_change = 0.0               # monotonic time that command last changed
        self._push_tried: set[int] = set()
        self.splices: list[dict] = []           # per push slot: timing + joint check

        # Command output: the controller's SampleBuffer when the backend has it.
        self.out: _BufferOutput | None = None
        if getattr(self.backend, "buf", None) is not None:
            self.out = _BufferOutput(self, self.backend)
        elif self.cfg.BACKEND == "hw" and os.environ.get("GP8_CYCLE_OUTPUT", "buffer") != "stream":
            raise RuntimeError(
                "cycle mode on hardware writes to SampleBufferController, but the backend "
                "has no buffer client: run with GP8_MODE=cycle in the environment (and the "
                "driver built from feature/sample-buffer), or GP8_CYCLE_OUTPUT=stream")
        if self.out is not None:
            # The controller's splice check decides on a slightly late piece (the
            # teammate's bridge does the same); the 2 ms stream would jump instead.
            self.stop_on_late = os.environ.get("GP8_CYCLE_STOP_ON_LATE", "0") not in ("0", "false", "no")

    # ------------------------------------------------------------------
    def now(self) -> float:
        return time.monotonic() - float(self.t0)

    def run(self, intake=None, ok=None) -> None:
        """Blocking: move to the planner's rest pose, start the planner, stream.

        ``intake()`` (the app's perception update) runs every feed period.
        ``ok()`` keeps it running (default: rclpy.ok() and the app's spinner)."""
        if ok is None:
            import rclpy

            def ok():
                return rclpy.ok() and not self.app._spinner_dead

        self._start_planner()
        period = 1.0 / max(1.0, _env_float("GP8_CYCLE_CANDS_HZ", 20.0))
        last_report = time.monotonic()
        try:
            while ok() and self._fault is None and not self._done.is_set():
                if intake is not None:
                    intake()
                if self.push_handler is not None and self._push_due():
                    self._push_slot(intake)
                    continue
                self._send({"t": "cands", "objs": self._candidates()})
                if time.monotonic() - last_report > 10.0:
                    last_report = time.monotonic()
                    self.log.info(f"[cycle] {dict(self.stats)} max lag {1e3 * self.max_lag:.1f} ms")
                time.sleep(period)
        finally:
            self.close()
        if self._fault:
            raise RuntimeError(f"cycle mode stopped: {self._fault}")

    def _start_planner(self) -> None:
        """Home, child process up, command output started, "start" sent."""
        if self.cfg.BACKEND == "hw" and STREAM_HZ != 500.0:
            raise RuntimeError(
                "cycle mode streams 2 ms commands: set GP8_STREAM_HZ=500 AFTER the "
                "adv4ncr driver runs at 500 Hz (controllers.yaml update_rate: 500, "
                "motoman_hardware control period 0.002 s)")
        self._check_belt_speed()
        self._move_home()
        self._spawn_child()
        t_ready = time.monotonic() + _env_float("GP8_CYCLE_READY_TIMEOUT", 180.0)
        while not self._ready.wait(timeout=0.2):
            exited = self._child is not None and self._child.poll() is not None
            if exited or time.monotonic() > t_ready:
                time.sleep(0.3)                       # let the stderr reader catch up
                raise RuntimeError(
                    "cycle planner " + ("exited" if exited else "did not become ready")
                    + "; its last stderr lines:\n  " + "\n  ".join(self._stderr_tail))

        if self.out is not None:
            if not self.backend.buf_activate(5.0):
                raise RuntimeError("SampleBufferController did not start (bringup with "
                                   "sample_buffer:=true, i.e. GP8_SAMPLE_BUFFER=true)")
            self.t0 = self.out.start()
        else:
            self.t0 = time.monotonic()
            emitter = threading.Thread(target=self._emit_loop, name="cycle_emit", daemon=True)
            emitter.start()
        self._send({"t": "start", "t0": self.t0, "home": self.home.tolist(), "run_s": 1e9})
        self.log.info(f"[cycle] model set: belt {PLANNER_BELT_SPEED} m/s, bins x {CYCLE_BIN_X} "
                      f"z {CYCLE_BIN_Z} m")
        self.log.info(f"[cycle] planner started ({self.planner_dir}); release lead "
                      f"{self.release_lead:.3f} s; output "
                      + ("SampleBufferController (controller tick)" if self.out is not None
                         else f"app stream {STREAM_HZ:.0f} Hz"))

    def close(self) -> None:
        """Stop taking objects, let the committed motion finish, then stop."""
        if self._child is None:
            return
        self._send({"t": "stop"})
        # Same rest condition as a push slot: the planner has ended its segment
        # (throwing what it holds), its end piece has arrived and played out,
        # and the command has been still for a moment.
        t_stop = time.monotonic()
        deadline = t_stop + _env_float("GP8_CYCLE_DRAIN_S", 12.0)
        rested = False
        while time.monotonic() < deadline and self._fault is None:
            with self._lock:
                idle = not self._seg_active
                if idle and self._seg_end_t is not None:
                    idle = self._k_last_cmd * PLANNER_DT >= self._seg_end_t + 0.25
            if idle and time.monotonic() - t_stop >= 1.0 and self._out_idle(0.2):
                rested = True
                break
            time.sleep(0.05)
        if not rested and self._fault is None:
            self.log.warn("[cycle] stop: the planner did not come to rest in time; holding where the arm is")
        self._teardown()

    def _teardown(self) -> None:
        """Output stopped (the controller holds the last command), child gone."""
        self._halt.set()
        if self._fault is not None:
            # Stopped on a fault, wherever the arm is: do not leave the vacuum on.
            try:
                self.backend.suction_off()
                self.log.warn("[cycle] fault stop: suction switched off")
            except Exception as e:  # noqa: BLE001
                self.log.error(f"[cycle] fault stop: suction off failed: {e!r}")
        if self.out is not None:
            self.out._q.put(None)
            self.out.stop()                 # nothing more plays; the controller holds the last command
        child, self._child = self._child, None
        try:
            if child.stdin:
                child.stdin.close()
            child.terminate()
            child.wait(5)
        except Exception:  # noqa: BLE001
            child.kill()
        extra = (f"min write lead {self.out.min_lead_s:.3f} s" if self.out is not None
                 and self.out.min_lead_s is not None else f"max lag {1e3 * self.max_lag:.1f} ms")
        self.log.info(f"[cycle] stopped: {dict(self.stats)} {extra}")

    # ------------------------------------------------------------------
    def _check_belt_speed(self) -> None:
        v = float(self.app.conveyor.current)
        if abs(v - PLANNER_BELT_SPEED) > 0.01:
            self.log.warn(
                f"[cycle] belt {v:.3f} m/s but the planner networks are built for "
                f"{PLANNER_BELT_SPEED} m/s: contacts will drift by the speed error")

    def _move_home(self) -> None:
        """Move (with the normal stream) to the planner's rest pose."""
        while self.backend.current_joints is None:
            time.sleep(0.05)
        self._move_to(self.home)
        self.backend.suction_off()
        time.sleep(0.5)

    def _move_to(self, q_goal: np.ndarray) -> None:
        """Rest-to-rest joint move with the normal stream; the last command
        sent is exactly ``q_goal``."""
        q = np.asarray(self.backend.current_joints, float)
        if float(np.max(np.abs(q - np.asarray(q_goal, float)))) < 1e-4:
            self.backend._emit_sample(np.asarray(q_goal, float).tolist())
            return
        zero = np.zeros(6)
        traj, vel, ts = trajectory(q, zero, np.asarray(q_goal, float), zero,
                                   self.app.M1 * 0.5, self.app.M2 * 0.5,
                                   hertz=self.cfg.TRAJ_HZ)
        self.backend.send_trajectory_queue(traj, vel, ts, final_joint=np.asarray(q_goal, float))

    def _routed_to_push(self, obj) -> bool:
        return self.skill_for is not None and self.skill_for(obj) == "push"

    def _candidates(self) -> list[dict]:
        if self.hide:
            return []
        tw = time.time()
        tm = self.now()
        v = float(self.app.conveyor.current)
        dist = self.app.conveyor.distance_m
        out = []
        for obj in self.app.queue.snapshot():
            if not is_established(obj, self.cfg.TRACK_MIN_HITS):
                continue
            if self._routed_to_push(obj):
                continue                    # handled in a push slot, never thrown
            b = CYCLE_BIN_INDEX.get(obj.class_name)
            if b is None:
                continue
            y = float(obj.y_at(tw, v, dist))
            if not self.vis_y[0] <= y <= self.vis_y[1]:
                continue
            o = {"id": int(obj.track_id), "cls": obj.class_name,
                 "x": float(obj.T_grasp_base[0, 3]), "b": b,
                 # The planner extrapolates y(t) = y0 - 0.12 t itself.
                 "y0": y + PLANNER_BELT_SPEED * tm}
            if self.top_z == "track":
                o["z"] = float(obj.T_grasp_base[2, 3])
            elif self.top_z:
                o["z"] = float(self.top_z)
            out.append(o)
        return out

    # ------------------------------------------------------------------
    # push slots (see module docstring)
    # ------------------------------------------------------------------
    # output-independent state used by the push slots and close()
    def _out_idle(self, settle_s: float) -> bool:
        if self.out is not None:
            return self.out.idle(settle_s)
        with self._lock:
            sent = self._k_sent >= self._k_last_cmd
        return sent and time.monotonic() - self._t_last_change >= settle_s

    def _out_last_command(self) -> np.ndarray:
        if self.out is not None:
            c = self.out.last_command()
            return np.asarray(c if c is not None else self.home, float)
        return np.asarray(self._last_sent, float)

    def _out_pause(self) -> bool:
        self._paused.set()
        if self.out is not None:   # JGPC takes the interfaces; it holds the last command
            return self.backend.buf_switch(False, 5.0)
        self._pause_ack.clear()
        return self._pause_ack.wait(0.5)

    def _out_resume(self, q_rest: np.ndarray) -> bool:
        if self.out is not None:
            if not self.backend.buf_activate(5.0):   # holds the current pose, tick restarts
                return False
            self.out.rebase(self.t0)
            self._paused.clear()
            return True
        self._resume_q = q_rest
        self._paused.clear()
        return True

    def _push_due(self) -> list:
        """Push-routed objects within ``push_lead`` s of entering reach, most
        downstream first (each object is tried once)."""
        tw = time.time()
        v = max(float(self.app.conveyor.current), 1e-3)
        dist = self.app.conveyor.distance_m
        r = float(self.cfg.MAX_REACH)
        due = []
        for obj in self.app.queue.snapshot():
            if (obj.track_id in self._push_tried or not self._routed_to_push(obj)
                    or not is_established(obj, self.cfg.TRACK_MIN_HITS)):
                continue
            x = float(obj.T_grasp_base[0, 3])
            y = float(obj.y_at(tw, v, dist))
            y_enter = math.sqrt(max(r * r - x * x, 0.0))
            if (y - y_enter) / v <= self.push_lead:
                due.append((y, obj))
        return [obj for _, obj in sorted(due, key=lambda p: p[0])]

    def _push_slot(self, intake=None) -> None:
        rec = {"t_drain": round(self.now(), 3)}
        self.hide = True
        t_hide = time.monotonic()
        # 1. drain: the planner throws what it holds and ends at its wait point.
        t_limit = time.monotonic() + self.drain_timeout
        while True:
            if self._fault is not None:
                return
            if intake is not None:
                intake()
            self._send({"t": "cands", "objs": []})
            with self._lock:
                idle = not self._seg_active
                # ...and the end piece (rest at t_end + filter settling) has arrived.
                if idle and self._seg_end_t is not None:
                    idle = self._k_last_cmd * PLANNER_DT >= self._seg_end_t + 0.25
            if (idle and time.monotonic() - t_hide >= self.min_drain_s
                    and self._out_idle(self.settle_s)):
                break
            if time.monotonic() > t_limit:
                self._fault = "push slot: the planner did not come to rest"
                return
            time.sleep(0.02)
        q_rest = np.asarray(self._out_last_command(), float).copy()
        rec["t_rest"] = round(self.now(), 3)

        # 2. the planner's output pauses; the app's PushSkill owns the arm.
        if not self._out_pause():
            self._fault = self._fault or "push slot: the command output did not pause"
            return
        pushed = []
        try:
            while len(pushed) < self.push_max:
                due = self._push_due()
                if not due:
                    break
                obj = due[0]
                self._push_tried.add(obj.track_id)
                detail = self.push_handler(obj)
                self.stats["pushes"] += 1
                pushed.append((int(obj.track_id), obj.class_name, str(detail)))
                self.log.info(f"[cycle] push id={obj.track_id} {obj.class_name}: {detail}")
                if intake is not None:
                    intake()
            # 3. hand back at the exact last planner command, then let it settle.
            time.sleep(self.settle_s)
            self._move_to(q_rest)
        except Exception as e:  # noqa: BLE001 - never resume the stream from an unknown pose
            self._fault = f"push slot failed: {e!r}"
            return
        if not self._out_resume(q_rest):
            self._fault = self._fault or "push slot: the command output did not resume"
            return
        time.sleep(self.settle_s)              # the output holds q_rest before planning resumes
        q_now = self.backend.current_joints
        rec.update(t_resume=round(self.now(), 3), pushed=pushed,
                   return_err_rad=(float(np.max(np.abs(np.asarray(q_now, float) - q_rest)))
                                   if q_now is not None else None))
        self.splices.append(rec)
        self.stats["push_slots"] += 1
        self.log.info(f"[cycle] push slot done: {rec}")
        self.hide = False

    # ------------------------------------------------------------------
    # child process
    # ------------------------------------------------------------------
    def _spawn_child(self) -> None:
        if self.planner_host:
            return self._spawn_remote()
        link = Path(__file__).with_name(self.LINK)
        gt_home = self.planner_dir.parent
        py = os.environ.get("GP8_CYCLE_PYTHON", sys.executable)
        env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "USER")
               if k in os.environ}
        env.update(GT_PY=py, GT_PLANNER_DIR=str(self.planner_dir), PYTHONUNBUFFERED="1")
        for k in _child_env_keys():                # honoured by env.sh / the link
            env[k] = os.environ[k]
        cmd = ["bash", "-c", 'source "$1/env.sh" && exec "$GT_PY" -W ignore "$2"',
               "cycle_planner", str(gt_home), str(link)]
        self._child = subprocess.Popen(cmd, cwd=str(self.planner_dir), env=env,
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True, bufsize=1)
        self._start_readers()

    def _spawn_remote(self) -> None:
        """planner_link.py on ``planner_host`` (same branch checked out at
        ``remote_dir``), with a clean env and the vendored env.sh."""
        root = self.remote_dir
        py = os.environ.get("GP8_CYCLE_REMOTE_PYTHON", f"{root}/.venv/bin/python")
        gt_home = f"{root}/third_party/grasp_and_throw"
        fwd = " ".join(f"{k}={shlex.quote(os.environ[k])}" for k in _child_env_keys())
        inner = (f'source {shlex.quote(gt_home)}/env.sh && exec "$GT_PY" -W ignore '
                 f'{shlex.quote(root)}/cycle_throw/{self.LINK}')
        remote = (f"cd {shlex.quote(gt_home)}/planner && exec env -i HOME=\"$HOME\" "
                  f"PATH=/usr/bin:/bin GT_PY={shlex.quote(py)} "
                  f"GT_PLANNER_DIR={shlex.quote(gt_home)}/planner PYTHONUNBUFFERED=1 {fwd} "
                  f"bash -c {shlex.quote(inner)}")
        cmd = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=2",
               "-o", "ServerAliveCountMax=3", self.planner_host, remote]
        self.log.info(f"[cycle] planner on {self.planner_host}:{root}")
        self._child = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True, bufsize=1)
        self._start_readers()

    def _start_readers(self) -> None:
        threading.Thread(target=self._read_child, name="cycle_rx", daemon=True).start()
        threading.Thread(target=self._read_stderr, name="cycle_err", daemon=True).start()

    def _send(self, msg: dict) -> None:
        child = self._child
        if child is None or child.stdin is None:
            return
        try:
            with self._send_lock:
                child.stdin.write(json.dumps(msg) + "\n")
                child.stdin.flush()
        except (BrokenPipeError, ValueError, OSError):
            self._fault = self._fault or "planner process pipe closed"

    def _read_stderr(self) -> None:
        child = self._child
        for line in child.stderr:
            line = line.rstrip()
            if not line:
                continue
            self._stderr_tail.append(line[:300])
            if line.startswith("[env.sh]") or "Traceback" in line or "Error" in line:
                self.log.warn(f"[cycle/planner] {line[:300]}")   # e.g. no model set for GT_BELT x GT_BIN_X
            else:
                debug = getattr(self.log, "debug", None)
                if debug is not None:
                    debug(f"[cycle/planner] {line[:300]}")

    def _read_child(self) -> None:
        child = self._child
        for line in child.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            kind = msg.get("t")
            if kind == "ping":                       # clock sync (planner_link)
                self._send({"t": "pong", "c": msg["c"], "a": time.monotonic()})
            elif kind == "cmd":
                self._on_commit(int(msg["k"]), msg["U"])
            elif kind == "suc" and self.out is not None:
                lead = self.release_lead if not msg["on"] else 0.0
                self.out.suction(float(msg["at"]) - lead, bool(msg["on"]))
                self.stats["suction_on" if msg["on"] else "suction_off"] += 1
            elif kind == "suc":
                lead = self.release_lead if not msg["on"] else 0.0
                with self._lock:
                    self._suction.append((float(msg["at"]) - lead, bool(msg["on"])))
                    self._suction.sort()
            elif kind == "log":
                text = str(msg["msg"])
                if _SEG_START.match(text):
                    self._seg_active = True
                elif _SEG_END.match(text):
                    # The end piece is computed by the planner's command thread
                    # AFTER this line: remember where it must reach.
                    self._seg_end_t = float(_SEG_END.match(text).group(1))
                    self._seg_active = False
                self.log.info(f"[cycle] {text}")
            elif kind == "ready":
                self._ready_msg = msg
                self._ready.set()
            elif kind == "done":
                self.log.info(f"[cycle] planner finished: {msg.get('summary')}")
                self._done.set()
            elif kind == "error":
                self._fault = f"planner error: {msg.get('msg', '')[-600:]}"
                self.log.error(f"[cycle] {self._fault}")
            else:
                self._on_msg(kind, msg)
        if not self._halt.is_set() and not self._done.is_set():   # "done" = a normal exit
            tail = " | ".join(list(self._stderr_tail)[-4:])
            self._fault = self._fault or f"planner process exited ({tail})"

    def _on_msg(self, kind, msg: dict) -> None:
        """Child messages the base protocol does not know (piece mode)."""

    def _on_commit(self, k: int, rows) -> None:
        U = np.asarray(rows, float)
        if U.ndim != 2 or U.shape[1] != 6 or not np.isfinite(U).all():
            self._fault = f"invalid command piece at sample {k}"
            return
        if self._paused.is_set():
            # The planner must be idle while a push owns the arm.
            self._fault = f"planner committed a piece (sample {k}) during a push slot"
            return
        if self.out is not None:
            with self._lock:
                self._k_last_cmd = max(self._k_last_cmd, k + U.shape[0] - 1)
            self.stats["pieces"] += 1
            self.out.commit(k, U)
            return
        with self._lock:
            late = max(0, self._k_sent + 1 - k)
            for i in range(late, U.shape[0]):
                self._cmds[k + i] = U[i]
            self._k_last_cmd = max(self._k_last_cmd, k + U.shape[0] - 1)
        self.stats["pieces"] += 1
        if late:
            # The README's warning sign: a throw can miss after a late piece.
            # Worse, the stream has been holding the last sample, so resuming
            # from this piece would jump the arm. Stop unless told otherwise.
            self.stats["late_pieces"] += 1
            msg = (f"late command piece: first {late} of {U.shape[0]} samples already "
                   f"due ({late * PLANNER_DT * 1e3:.0f} ms); planner too slow for this machine?")
            if self.stop_on_late:
                self._fault = self._fault or msg
            self.log.warn(f"[cycle] {msg}")

    # ------------------------------------------------------------------
    # 2 ms command stream
    # ------------------------------------------------------------------
    def _window_time(self, q: np.ndarray, t: float, hist) -> float | None:
        """Earliest send time >= t at which q stays within one controller
        cycle's increment of the command the controller held one period
        earlier, whatever the cycle phase (vendored robot_base._win_time).
        None if q is over the limit even from the last sent sample."""
        if not hist:
            return t
        lim = self.inc * (1.0 + 1e-6)
        for tc in [t] + [te + PLANNER_DT for te, _ in hist if te + PLANNER_DT > t]:
            ref = hist[0][1]
            for te, qe in hist:
                if te <= tc - PLANNER_DT + 1e-9:
                    ref = qe
                else:
                    break
            if np.all(np.abs(q - ref) <= lim):
                return tc
        return None   # a single step is already over the limit: never send it

    def _emit_loop(self) -> None:
        last = self.home.copy()
        hist = collections.deque(maxlen=8)
        was_paused = False
        k = int(math.floor(self.now() / PLANNER_DT)) + 1
        try:
            while not self._halt.is_set() and self._fault is None:
                if self._paused.is_set():
                    # A push owns the arm: send nothing, keep the sample clock
                    # current, and resume exactly from the hand-back pose.
                    was_paused = True
                    self._pause_ack.set()
                    with self._lock:
                        self._k_sent = max(self._k_sent, int(math.floor(self.now() / PLANNER_DT)))
                    time.sleep(0.001)
                    continue
                if was_paused:
                    was_paused = False
                    last = np.asarray(self._resume_q, float).copy()
                    self._last_sent = last
                    self._t_last_change = time.monotonic()
                    hist.clear()
                    k = int(math.floor(self.now() / PLANNER_DT)) + 1
                    self._pause_ack.clear()
                due = self.t0 + k * PLANNER_DT
                with self._lock:
                    q = self._cmds.pop(k, None)
                    t_k = k * PLANNER_DT
                    fire = []
                    while self._suction and self._suction[0][0] <= t_k:
                        fire.append(self._suction.pop(0)[1])
                if q is not None:
                    last = q
                t_send = self._window_time(last, due, hist)
                if t_send is None:
                    # Hold the previous command instead (the controller keeps it).
                    self.stats["over_increment"] += 1
                    self._fault = self._fault or (
                        f"sample {k} steps more than one controller cycle allows "
                        "(late or discontinuous command); stream stopped")
                    break
                if t_send > due + 1e-9:
                    self.stats["window_delayed"] += 1
                _sleep_until(t_send, self.spin_s)
                t_now = time.monotonic()
                self.max_lag = max(self.max_lag, t_now - due)
                self.backend._emit_sample(last.tolist())
                hist.append((t_now, last))
                if not np.array_equal(last, self._last_sent):
                    self._t_last_change = t_now
                    self._last_sent = last
                with self._lock:
                    self._k_sent = k
                for on in fire:
                    (self.backend.suction_on if on else self.backend.suction_off)()
                    self.stats["suction_on" if on else "suction_off"] += 1
                k += 1
        except Exception as e:  # noqa: BLE001 - a dead stream must stop the run
            self._fault = f"command stream failed: {e!r}"

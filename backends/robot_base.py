"""RobotBackend — the ONE robot-control seam app/skills drive, HW **or** sim.

Why this exists
---------------
Everything the orchestrator and the manipulation skills need from "the robot"
is a small, stable set of calls. The *real* control path
(``controllers/trajectory_controller.py``, adv4ncr 250 Hz stream) and the *sim*
path historically drifted because the old sim mocked the ROS **wire** protocol
(Point Queue Mode services, ``/write_single_io``, ``/joint_states_urdf``) — and
that wire changed under the sim when the backend migrated to ros2_control. The
wire was never the real contract; THIS class is.

Crucially, the **hardware operation logic itself lives here**, moved verbatim
from ``trajectory_controller.py``:

  * the 250 Hz stream engine (:meth:`_stream_trajectory` / :meth:`_stream_hold`):
    cubic-Hermite resample onto the 4 ms grid honoring knot velocities, clamped
    to the bracketing-knot envelope (floor-dive / alarm-4315 fix), grid-time
    suction release, prime-after-release, the ``tick_fn`` early-stop protocol,
    and wall-clock pacing to an absolute monotonic schedule;
  * suction request **coalescing** (a second same-state call is a silent no-op
    *including the telemetry stamp*) and the fire-instant telemetry every
    skill's diagnostics read;
  * the blocking motion surface (``send_trajectory_queue*``, ``pq_*``,
    ``_wait_for_position``) and the queue-mode lifecycle shims.

So the MuJoCo twin executes the *identical* resample/clamp/release/pacing code
as the real arm — a backend only supplies three primitives:

  * :meth:`wait_for_servers` — become ready;
  * :meth:`_emit_sample`     — deliver ONE 4 ms joint-position command
    (hardware publishes ``Float64MultiArray`` to the JGPC; the twin writes the
    position-actuator ctrl targets);
  * :meth:`_set_suction`     — physically toggle the vacuum (hardware enqueues
    a Simple-Message TCP write; the twin welds/releases the grasped box).

plus keeps :attr:`current_joints` fresh (a ``/joint_states`` callback on
hardware; the physics stepper in the twin), and may override :meth:`_ok`
(hardware: ``rclpy.ok()``), :meth:`set_motion_op`, :meth:`close`.

This module imports NO rclpy — backends decide their own transport.

NOTE ON POLICY vs. MECHANISM: *when* to release (``release_index``) and *when*
to prime the next pick (``suction_on_at``) are the skill's policy, passed per
call. This class only provides the mechanism; it bakes in no timing of its own.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np

STREAM_HZ = 250.0
STREAM_DT = 1.0 / STREAM_HZ   # 4 ms


def _max_abs_diff(a, b) -> float:
    n = min(len(a), len(b))
    return max((abs(float(a[i]) - float(b[i])) for i in range(n)), default=0.0)


def _traj_total(timestep) -> float:
    """Total trajectory duration. ``timestep`` from trajectory() is CUMULATIVE time_from_start
    (arange(L+1)/hertz), so the duration is the LAST value — NOT np.sum(timestep) (which is
    ~T*(n+1)/2 and grossly inflates timeouts/deadlines). Review finding #3/#4."""
    t = np.asarray(timestep, dtype=float).ravel()
    return float(t[-1]) if t.size else 0.0


class RobotBackend(ABC):
    """Common robot-control surface skills/app depend on; HW and sim subclass it.

    Subclasses MUST keep :attr:`current_joints` (and optionally
    :attr:`current_jointvels`) fresh from their own state source and implement
    the three abstract methods. Everything else is provided here.
    """

    def __init__(self) -> None:
        # --- live state (the subclass keeps these fresh) ---------------------
        #: Latest measured joint positions [rad], GP8 order [S,L,U,R,B,T];
        #: ``None`` until the first sample arrives.
        self.current_joints: list | None = None
        #: Latest measured joint velocities [rad/s] (same order), or ``None``.
        self.current_jointvels: list | None = None

        # --- pick-cycle timing telemetry (kept for the skills' logs) ---------
        self.last_suction_on_t: float | None = None
        self.last_suction_on_ack_t: float | None = None
        # Joint state snapshotted at the instant suction turned ON, so a skill can
        # FK it to the EE pose the vacuum actually fired at (diagnostic).
        self.last_suction_on_joints: list | None = None
        self.last_suction_off_t: float | None = None
        self.last_suction_off_ack_t: float | None = None
        self.last_throw: dict | None = None
        self._last_throw_ok: bool = True   # #R3: last timed-release dispatch accepted?

        # Kept for API compatibility with the skills' timeline model
        # (push_skill.t_to_contact reads qmode_ms_avg for T_setup). On a modern
        # controller there is NO queue-mode re-entry; the only fixed overhead is
        # dispatch + ~40 ms controller dead-time.
        self.last_qmode_ms: float | None = None
        self.qmode_ms_avg: float = 40.0

        # Persistent-session state (pq_*): just tracks "a pick+hold+throw
        # sequence is in progress" (no queue to keep alive on any backend).
        self._pq_active: bool = False

        # Suction request coalescing: suction_on/off record the REQUESTED state
        # under this lock and no-op on duplicates (telemetry stamp included) —
        # the hardware IO worker also resets `_io_requested_value` to None after
        # a failed write so a later same-state call can retry (it shares these
        # exact attribute names).
        self._io_state_lock = threading.Lock()
        self._io_requested_value: int | None = None

    # ------------------------------------------------------------------
    # Liveness hook — hardware overrides with rclpy.ok() so a ROS shutdown
    # breaks the paced loops; the sim has no external liveness source.
    # ------------------------------------------------------------------
    def _ok(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # 250 Hz stream engine (moved verbatim from trajectory_controller;
    # only rclpy.ok() -> self._ok() and the JGPC publish -> _emit_sample)
    # ------------------------------------------------------------------
    def _stream_trajectory(self, traj, vel, timestep, final_joint=None, *,
                           release_index: int | None = None,
                           suction_on_at: float | None = None,
                           tick_fn=None) -> dict:
        """Stream a trajectory at ~250 Hz through :meth:`_emit_sample`.

        The app's trajectories are sampled at ~20 Hz (``trajectory()``: ``timestep`` is the
        cumulative time_from_start = arange(L+1)/hertz). The 250 Hz control loop wants a
        fresh position every 4 ms, so we RESAMPLE onto a 4 ms grid and emit each sample,
        paced to wall clock.

        ⚠️ SOFT real-time: this is a Python paced loop (GIL / scheduler jitter), NOT hard 250 Hz.
        The controller runs the true 250 Hz; the receiver zero-order-holds our last command
        between emits, so jitter degrades smoothness, not safety.

        ``release_index`` → fire suction_off when the stream reaches that waypoint's time.
        ``suction_on_at`` (wall clock) → after release, fire suction_on once to prime the next pick.
        ``tick_fn`` → called every sample (e.g. to prime suction on a positioning move).
        Returns {'fired': bool, 'primed_next': bool}.
        """
        arr = np.asarray(traj, dtype=float)                       # (n_joints, n_steps)
        varr = np.asarray(vel, dtype=float)                       # velocity profile (same shape)
        n_joints, n_steps = arr.shape
        times = np.asarray(timestep, dtype=float).ravel()         # cumulative time_from_start
        t_release = float(times[int(release_index)]) if release_index is not None else None
        if final_joint is not None:
            arr = np.concatenate(
                [arr, np.asarray(final_joint, dtype=float).reshape(n_joints, 1)], axis=1)
            varr = np.concatenate([varr, np.zeros((n_joints, 1))], axis=1)   # settle at rest
            times = np.append(times, times[-1] + 0.05)
        T = float(times[-1])
        grid = np.arange(0.0, T + STREAM_DT, STREAM_DT)
        # Resample onto the 4ms grid HONORING the knot velocities (#R3: a plain linear resample of
        # the coarse ~20Hz arc yields chord-slope velocity, not the designed NN release speed —
        # matters because throw range ~ v^2). Cubic-Hermite uses positions+velocities; fall back to
        # linear if times aren't strictly increasing (degenerate/segment-join) or scipy is missing.
        if varr.shape == arr.shape and np.all(np.diff(times) > 0):
            try:
                from scipy.interpolate import CubicHermiteSpline
                samples = np.column_stack(
                    [CubicHermiteSpline(times, arr[j], varr[j])(grid) for j in range(n_joints)])
                # SAFETY: cubic-Hermite can OVER/UNDERSHOOT beyond the knot values — a descent
                # decelerating to belt contact dips BELOW the final pose. The receiver is a POSITION
                # controller, so a below-surface command drives the tool INTO the belt -> shock
                # (alarm 4315). Clamp each sample to the envelope of its two bracketing knots:
                # keeps the cubic velocity shape WITHIN bounds but never past a commanded waypoint
                # (linear never overshoots; this makes cubic equally floor-safe).
                seg = np.clip(np.searchsorted(times, grid, side="right") - 1, 0, len(times) - 2)
                lo = np.minimum(arr[:, seg], arr[:, seg + 1]).T   # (n_grid, n_joints)
                hi = np.maximum(arr[:, seg], arr[:, seg + 1]).T
                samples = np.clip(samples, lo, hi)
            except Exception:
                samples = np.column_stack([np.interp(grid, times, arr[j]) for j in range(n_joints)])
        else:
            samples = np.column_stack([np.interp(grid, times, arr[j]) for j in range(n_joints)])

        st = {"fired": False, "primed_next": False}
        start = time.monotonic()      # pacing clock (monotonic)
        wall_start = time.time()      # telemetry clock (wall) — last_throw must stay wall-clock (#3)
        for k in range(samples.shape[0]):
            if not self._ok():
                break
            self._emit_sample(samples[k])

            if (t_release is not None and not st["fired"] and grid[k] >= t_release):
                t_io = time.time()
                self.suction_off()
                st["fired"] = True
                self.last_throw = {
                    "throw_start": wall_start, "release_wall": self.last_suction_off_t,
                    "io_ms": (time.time() - t_io) * 1000.0,
                    "release_index": int(release_index), "n_waypoints": int(n_steps),
                }
            elif (st["fired"] and suction_on_at is not None and not st["primed_next"]
                  and time.time() >= suction_on_at):
                self.suction_on()
                st["primed_next"] = True
            if tick_fn is not None:
                # Existing callbacks return None and continue as before.  An
                # explicit False is reserved for supervised debug motions that
                # must stop at the current streamed command (for example the
                # suction attachment-range measurement).
                if tick_fn() is False:
                    st["stopped_early"] = True
                    st["stop_index"] = int(k)
                    break

            # pace to the next 4 ms tick (drop no samples; sleep the remainder)
            dt_sleep = (start + (k + 1) * STREAM_DT) - time.monotonic()
            if dt_sleep > 0:
                time.sleep(dt_sleep)
        return st

    def _stream_hold(self, hold_joint, deadline_wall: float, tick_fn=None) -> None:
        """Keep emitting ``hold_joint`` at ~250 Hz until wall-clock ``deadline_wall``
        (feeds the servo during an ambush hold). tick_fn runs each cycle (e.g. prime suction)."""
        pose = [float(x) for x in hold_joint]
        nxt = time.monotonic()
        while time.time() < deadline_wall and self._ok():
            self._emit_sample(pose)
            if tick_fn is not None:
                tick_fn()
            nxt += STREAM_DT
            dt_sleep = nxt - time.monotonic()
            if dt_sleep > 0:
                time.sleep(dt_sleep)
            else:
                nxt = time.monotonic()

    # ------------------------------------------------------------------
    # send_trajectory family (stream semantics; the hardware subclass
    # overrides these to branch to its JTC action path when configured)
    # ------------------------------------------------------------------
    def send_trajectory(self, traj, vel, timestep, final_joint=None) -> bool:
        self._stream_trajectory(traj, vel, timestep, final_joint)
        return True

    def send_trajectory_queue(self, traj, vel, timestep, final_joint=None) -> bool:
        """(MotoROS2 name kept.) Blocks for the whole motion."""
        self._stream_trajectory(traj, vel, timestep, final_joint)
        return True

    def send_trajectory_queue_interruptible(
        self, traj, vel, timestep, final_joint, stop_requested,
    ) -> bool:
        """Stream until ``stop_requested()`` becomes true.

        Returns True when the motion stopped early.
        """
        state = self._stream_trajectory(
            traj, vel, timestep, final_joint,
            tick_fn=lambda: not bool(stop_requested()),
        )
        return bool(state.get("stopped_early", False))

    def send_trajectory_queue_with_timed_release(self, traj, vel, timestep, final_joint,
                                                 release_index: int,
                                                 suction_on_at: float | None = None) -> bool:
        """Throw path: fire suction_off when the stream reaches the release waypoint's grid
        time, then (optionally) prime the next pick's vacuum once ``suction_on_at`` passes.
        Returns True iff the next-pick suction was primed (NOT success — call sites discard it).
        """
        arr = np.asarray(traj)
        rel = int(max(0, min(release_index, arr.shape[1] - 1)))
        st = self._stream_trajectory(traj, vel, timestep, final_joint,
                                     release_index=rel, suction_on_at=suction_on_at)
        if not st["fired"]:
            self.suction_off()   # fallback: never carry the object past release
        self._last_throw_ok = True   # stream emit never "rejects" (#R3)
        return st["primed_next"]

    def send_trajectory_queue_timed_suction(self, traj, vel, timestep, final_joint,
                                            suction_on_at: float) -> bool:
        """Positioning move that fires suction_ON once wall-clock ``suction_on_at`` passes
        (priming the vacuum before grasp). Returns True iff suction fired before the move ended."""
        fired = {"v": False}

        def _tick() -> None:
            if not fired["v"] and time.time() >= suction_on_at:
                self.suction_on()
                fired["v"] = True

        self._stream_trajectory(traj, vel, timestep, final_joint, tick_fn=_tick)
        if not fired["v"] and time.time() >= suction_on_at:
            self.suction_on()
            fired["v"] = True
        return fired["v"]

    # ------------------------------------------------------------------
    # Queue-mode shims (no-ops on any modern backend — kept for call-site
    # stability with the MotoROS2-era app/skills)
    # ------------------------------------------------------------------
    def enter_queue_mode(self) -> bool:
        """No-op: the backend is always ready (no Point Queue Mode)."""
        return True

    def exit_queue_mode(self) -> bool:
        """No-op."""
        return True

    # ------------------------------------------------------------------
    # Persistent-session shims (pq_*). No queue keep-alive on any backend,
    # so these are thin: a segment is one stream; a hold feeds the servo.
    # ------------------------------------------------------------------
    @property
    def pq_active(self) -> bool:
        """True while a persistent session is open. **@property** (read as an attribute,
        not called) — a plain method would evaluate truthy always and defeat the
        session-live guard (#A)."""
        return self._pq_active

    def pq_begin(self) -> None:
        self._pq_active = True

    def pq_segment(self, traj, vel, ts, final_joint, *, is_last: bool = False,
                   join_tol: float = 0.05, between_fn=None) -> tuple:
        """Execute one segment; call ``between_fn`` each cycle (used to fire suction_on
        when its wall-clock instant passes). Returns (ok, codes)."""
        self._stream_trajectory(traj, vel, ts, final_joint, tick_fn=between_fn)
        return (True, [])

    def pq_hold_until(self, hold_joint, deadline_wall: float, *,
                      dt: float = 0.12, lead: float = 0.35, tick_fn=None) -> bool:
        """Wait until wall-clock ``deadline_wall`` at the hold pose, feeding the servo;
        ``tick_fn`` runs each cycle (e.g. to prime suction on time)."""
        self._stream_hold(hold_joint, deadline_wall, tick_fn=tick_fn)
        return True

    def pq_throw_segment(self, traj, vel, timestep, final_joint, *,
                         release_index, suction_on_at=None,
                         release_tol: float = 0.05) -> tuple:
        """Throw with a timed suction release (same logic as
        send_trajectory_queue_with_timed_release). Returns ``(ok, primed_next)`` — a 2-tuple to
        match the caller's ``ok_throw, primed_next = ...`` unpack (throw_skill.py) and the sibling
        pq_segment's (ok, ...) contract (#1)."""
        primed_next = self.send_trajectory_queue_with_timed_release(
            traj, vel, timestep, final_joint, release_index, suction_on_at=suction_on_at,
        )
        return (self._last_throw_ok, bool(primed_next))

    def pq_finish(self, *, wait: bool = True, tail_buffer: float = 0.3,
                  settle_tol: float = 0.03) -> None:
        """End the session. (Each segment already blocks, so nothing to drain.)"""
        self._pq_active = False

    # ------------------------------------------------------------------
    # Waiting helpers
    # ------------------------------------------------------------------
    def _wait_for_position(self, target_joint, tolerance: float = 1e-4,
                           timeout_sec: float = 5.0) -> bool:
        """Block until the arm is within ``tolerance`` (max-abs) of ``target_joint``.
        (Private name kept — push_skill2 / push_skill_floor_gated call it.)"""
        target = [float(x) for x in target_joint]
        t_end = time.time() + timeout_sec
        while time.time() < t_end and self._ok():
            if self.current_joints is not None and _max_abs_diff(self.current_joints, target) <= tolerance:
                return True
            time.sleep(0.005)
        return False

    def _wait_trajectory_end(self, total_duration: float, t_start: float | None = None) -> None:
        t0 = t_start if t_start is not None else time.time()
        remaining = total_duration - (time.time() - t0)
        if remaining > 0:
            time.sleep(remaining)

    # ------------------------------------------------------------------
    # Suction — CONCRETE: coalescing + the telemetry every skill's
    # diagnostics read; the physical toggle is the backend's _set_suction.
    # ------------------------------------------------------------------
    def suction_on(self) -> None:
        """Request suction ON once; duplicate requested states are coalesced."""
        requested_at = time.time()
        with self._io_state_lock:
            if self._io_requested_value == 0:
                return
            self._io_requested_value = 0
        self.last_suction_on_t = requested_at
        # Snapshot the joint state AT the fire instant (this may run inside the
        # stream loop mid-move) so callers can FK the true EE pose the vacuum
        # fired at. list() to freeze it against a concurrent state update.
        self.last_suction_on_joints = (
            list(self.current_joints) if self.current_joints is not None else None
        )
        self._set_suction(True, requested_at)

    def suction_off(self) -> None:
        """Request suction OFF once; duplicate requested states are coalesced."""
        requested_at = time.time()
        with self._io_state_lock:
            if self._io_requested_value == 1:
                return
            self._io_requested_value = 1
        self.last_suction_off_t = requested_at
        self._set_suction(False, requested_at)

    # ------------------------------------------------------------------
    # Lifecycle — no-ops by default; hardware overrides.
    # ------------------------------------------------------------------
    def set_motion_op(self, label: str) -> None:
        """Tag the current cycle's motion with a skill label (push/throw) for
        diagnostic logging. No-op by default; a backend with a motion logger
        overrides it."""
        return None

    def close(self, timeout_sec: float = 5.0) -> None:
        """Release backend resources on shutdown. No-op by default."""
        return None

    # ------------------------------------------------------------------
    # Abstract seam — the ONLY things a backend must implement.
    # ------------------------------------------------------------------
    @abstractmethod
    def wait_for_servers(self, timeout_sec: float = 10.0) -> bool:
        """Block until the backend can accept motion (action server up / model
        loaded + stepper running). Return True on ready, False on timeout. The
        app calls this once in ``setup()`` before it starts the control loop."""
        raise NotImplementedError

    @abstractmethod
    def _emit_sample(self, positions: Sequence[float]) -> None:
        """Deliver ONE 4 ms joint-position command [S,L,U,R,B,T] rad.

        Hardware publishes a ``Float64MultiArray`` to the JGPC command topic;
        the twin writes the position-actuator ctrl targets. MUST be fast and
        non-blocking — it runs inside the 250 Hz paced loop.
        """
        raise NotImplementedError

    @abstractmethod
    def _set_suction(self, on: bool, requested_at: float) -> None:
        """Physically toggle the vacuum: True = grip, False = release.

        Called only on requested-STATE CHANGES (coalescing happens in
        suction_on/off above). Hardware enqueues a Simple-Message TCP write on
        its IO worker; the twin welds/releases the nearest box. MUST be
        non-blocking enough not to stall the 250 Hz loop at the release
        instant. Errors are logged, never raised — a killed drive loop is worse
        than a missed toggle (though a failed *release* is a real safety
        concern to catch in bring-up).
        """
        raise NotImplementedError

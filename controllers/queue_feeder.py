"""Single-writer background feeder — keep ONE point-queue session alive ACROSS
cycles so queue mode is entered ONCE at operation start and NEVER re-entered.

The re-entry (~0.4s, a stop+start mode switch) is paid EVERY cycle because the
queue drains to empty between segments and MotoROS2 silently leaves queue mode.
The within-cycle persistent path removed it for one pick→throw; the cross-cycle
reuse removed it for throw→throw chains — but a mixed push/throw workload breaks
those chains, so most cycles still re-enter.

This feeder removes it UNIVERSALLY, for any workload: a dedicated thread is the
SOLE caller of the queue push, and whenever no real segment is pending it emits
zero-velocity keep-alive hold points at the last queued pose (via
``TrajectoryController._pq_keepalive_tick``) so the queue never empties. Skills
(push AND throw) stop calling ``enter_queue_mode`` and instead ``submit_segment``
their trajectory onto the same live session; the feeder splices it onto the
monotonic ``_pq_t`` timeline. Mode is entered once, at startup, and the arm flows
continuously between every pick, throw, push and chain.

THREADING (rclpy): the TrajectoryController's service round-trips spin the node
via ``rclpy.spin_until_future_complete(node, ...)`` on the CALLING thread, routed
through the process-global executor. Two threads driving the same node's wait set
would race. So this feeder is the SINGLE ROS thread for the queue path: it owns
every ``_push_one_point`` / ``pq_segment`` / keep-alive push and their spins, and
also pumps callbacks (``spin_once``) while idle so ``current_joints`` stays fresh.
The caller thread must NOT make ROS calls on the same node while the feeder runs;
it only ``submit_segment``s (a lock-guarded enqueue) and waits on the handle.
Because the feeder is the sole writer of the session state (``_pq_t`` /
``_pq_last_pos`` / ``_pq_motion_start``) and the sole queue client, those need no
lock — only the submit deque + the drained flag are shared.

STATUS: first increment — proven in isolation by ``tests/queue_feeder_spike.py``
before any app/skill wiring.
"""

from __future__ import annotations

import collections
import threading
import time

import rclpy


class SegmentHandle:
    """Handle for a submitted segment. ``done`` is set once the feeder has spliced
    it (pushed all its points, or aborted on reject); ``ok`` is the splice result.
    ``codes`` carries the per-point result codes for diagnostics."""

    __slots__ = ("traj", "vel", "ts", "final_joint", "is_last", "between_fn",
                 "done", "ok", "codes")

    def __init__(self, traj, vel, ts, final_joint, is_last, between_fn):
        self.traj = traj
        self.vel = vel
        self.ts = ts
        self.final_joint = final_joint
        self.is_last = is_last
        self.between_fn = between_fn
        self.done = threading.Event()
        self.ok = None
        self.codes = None

    def wait(self, timeout: "float | None" = None) -> bool:
        """Block until the feeder has spliced this segment; return ``ok``."""
        self.done.wait(timeout=timeout)
        return bool(self.ok)


class QueueFeeder:
    """A background thread that keeps ``ctrl``'s persistent-queue session alive and
    splices submitted segments onto it. Start it once, after ``enter_queue_mode()``
    + ``pq_begin()``; stop it (or it self-stops on a drain) before any thread makes
    ROS calls on the node again.

    ``dt`` / ``lead`` are the keep-alive pacing (buffer ≈ ``lead`` ahead of the
    arm). ``poll`` bounds the idle spin so submits are picked up promptly."""

    def __init__(self, ctrl, *, dt: float = 0.15, lead: float = 0.35,
                 poll: float = 0.01):
        self._ctrl = ctrl
        self._dt = dt
        self._lead = lead
        self._poll = poll
        self._pending = collections.deque()
        self._lock = threading.Lock()
        self._thread: "threading.Thread | None" = None
        self._running = False
        self._drained = False

    # ---- public API (caller thread) --------------------------------------
    @property
    def drained(self) -> bool:
        """True once the queue genuinely emptied (WRONG_MODE) — the session is dead
        and the caller must recover via a fresh enter_queue_mode()/pq_begin()."""
        return self._drained

    @property
    def alive(self) -> bool:
        return self._running and not self._drained

    def start(self) -> None:
        if self._thread is not None:
            return
        self._running = True
        self._drained = False
        self._thread = threading.Thread(
            target=self._run, name="queue_feeder", daemon=True)
        self._thread.start()

    def stop(self, *, timeout: float = 2.0) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def submit_segment(self, traj, vel, ts, final_joint, *, is_last: bool = False,
                       between_fn=None) -> SegmentHandle:
        """Enqueue a real segment for the feeder to splice onto the live session.
        Non-blocking; wait on the returned handle for completion."""
        h = SegmentHandle(traj, vel, ts, final_joint, is_last, between_fn)
        if self._drained:
            # Session already dead — fail fast so the caller recovers, don't enqueue.
            h.ok = False
            h.codes = []
            h.done.set()
            return h
        with self._lock:
            self._pending.append(h)
        return h

    # ---- feeder thread ----------------------------------------------------
    def _run(self) -> None:
        node = self._ctrl._node
        while self._running:
            h = None
            with self._lock:
                if self._pending:
                    h = self._pending.popleft()
            if h is not None:
                # Splice the real segment (blocks the feeder for its push, but the
                # push itself keeps the queue non-empty — no drain meanwhile).
                ok, codes = self._ctrl.pq_segment(
                    h.traj, h.vel, h.ts, h.final_joint,
                    is_last=h.is_last, between_fn=h.between_fn)
                h.ok = ok
                h.codes = codes
                h.done.set()
                if not ok:
                    # A reject on a live session usually means it drained mid-plan.
                    self._drained = True
                    self._running = False
                continue
            # No pending segment: keep the queue alive with one hold tick.
            status = self._ctrl._pq_keepalive_tick(dt=self._dt, lead=self._lead)
            if status == "drained":
                self._drained = True
                self._running = False
                break
            if status == "buffered":
                # Queue deep enough — pump callbacks (refresh current_joints, free
                # BUSY slots) and yield so submits are seen promptly.
                rclpy.spin_once(node, timeout_sec=self._poll)
            # 'pushed': loop immediately (may still be below lead)

    def fail_pending(self) -> None:
        """Signal every outstanding handle as failed (e.g. after a drain) so no
        caller blocks forever on ``handle.wait()``."""
        with self._lock:
            while self._pending:
                h = self._pending.popleft()
                h.ok = False
                h.codes = []
                h.done.set()

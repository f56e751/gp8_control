"""Virtual time and a controller-tick buffer for the MuJoCo twin (piece mode for RL).

Piece mode on hardware writes 2 ms command samples into the controller's
SampleBuffer by tick and the planner link (robot6) plans against the wall
clock. To train on it faster than real time, the twin runs on a VIRTUAL clock:

* ``VirtualTime``: ``mono() = time.monotonic() + skew``. The sim is stepped
  so its time follows ``mono()`` (``drive``). While the planner link computes,
  virtual time runs with the wall clock, so planning delay costs the same sim
  time as on hardware. When nothing is being computed the runner WARPS: the
  link moves its clock by dt first (it commits every command chunk that falls
  due and acks), then ``skew += dt`` and the sim catches up at physics speed.
* ``SimTickBuffer``: the SampleBuffer's rules (``sample_buffer.cpp``: a chunk
  must start ``min_lead`` ticks ahead, per-step velocity / acceleration limits
  at the seam and inside, tail replace, hold the last command) with the
  ``BufferClient`` surface ``cycle_throw.runner._BufferOutput`` uses. Tick k
  is sim time ``k * DT``; ``SimCore`` asks ``at_tick(k)`` for the command and
  the suction events of every 2 ms physics step, so no Python timing loop
  is involved.
"""

from __future__ import annotations

import math
import threading
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

DT = 0.002
#: GP8 datasheet joint speeds [rad/s] (sample_buffer.yaml v_max)
V_MAX = np.array([7.9412480966, 6.7195176202, 9.0757121104, 9.5993108860, 9.5993108860,
                  17.4532925199])
RES_OK, RES_LATE, RES_LIMIT = 0, 1, 2


class VirtualTime:
    """Monotonic clock = wall + skew; the sim follows it (see module docstring)."""

    def __init__(self, core) -> None:
        self.core = core
        self.skew = 0.0
        self._lock = threading.Lock()
        self.warped_s = 0.0
        self.rebase()

    def rebase(self) -> None:
        """Map the sim's current time to the virtual now (after the sim was
        stepped on its own, e.g. a home move)."""
        with self._lock:
            self.c = time.monotonic() + self.skew - float(self.core.sim_time())
            # twin stamps (joint states, truth) on this clock
            self.core._mono_t0 = self.c

    def mono(self) -> float:
        return time.monotonic() + self.skew

    def sim_target(self) -> float:
        return self.mono() - self.c

    def to_mono(self, t_sim: float) -> float:
        return float(t_sim) + self.c

    def drive(self) -> None:
        """Step the sim up to the virtual now (physics speed)."""
        with self._lock:
            target = self.sim_target()
            if target > self.core.sim_time():
                self.core.advance_until(target)

    def sleep(self, seconds: float) -> None:
        """Let ``seconds`` of wall (= virtual) time pass, the sim following."""
        end = time.monotonic() + max(0.0, float(seconds))
        while True:
            self.drive()
            rem = end - time.monotonic()
            if rem <= 0.0:
                return
            time.sleep(min(rem, 0.002))

    def warp(self, dt: float) -> None:
        """Jump forward by dt (the caller has made sure nothing is due inside it)."""
        with self._lock:
            self.skew += float(dt)
            self.warped_s += float(dt)
        self.drive()


class SimTickBuffer:
    """SampleBuffer rules on the sim's ticks (BufferClient-compatible subset)."""

    def __init__(self, vt: VirtualTime, q0, *, min_lead: int = 3, a_factor: float = 20.0,
                 tol: float = 0.02, fire=None) -> None:
        self.vt = vt
        self.dq = V_MAX * DT * (1 + tol)
        self.ddq = a_factor * V_MAX * DT * DT * (1 + tol)
        self.min_lead = int(min_lead)
        self.fire = fire                      # fire(on) for suction, called at the tick
        self.lock = threading.Lock()
        self.ring: dict[int, np.ndarray] = {}
        self.end = 0                          # one past the last buffered sample
        self.k = self._tick_now() + 1         # next tick to output
        self.last = np.asarray(q0, float).copy()
        self.prev = self.last.copy()
        self.events: List[Tuple[int, bool]] = []
        self.ev_log: List[Tuple[int, bool]] = []
        self.n_acc = self.n_rej = 0
        self.rejects: List[str] = []
        self.active = False

    def activate(self, q_hold=None) -> bool:
        """SampleBufferController switched on: hold q_hold, output from the next tick."""
        with self.lock:
            if q_hold is not None:
                self.last = np.asarray(q_hold, float).copy()
                self.prev = self.last.copy()
            self.k = self._tick_now() + 1
            self.end = self.k
            self.ring = {}
            self.events = []
            self.active = True
        return True

    # -- clock ---------------------------------------------------------------
    def _tick_now(self) -> int:
        return int(math.floor(self.vt.core.sim_time() / DT + 1e-6))

    def now_index(self) -> int:
        return max(self.k - 1, self._tick_now())

    def tick_mono(self, k: float) -> float:
        return self.vt.to_mono(float(k) * DT)

    def index_at(self, t_mono: float) -> float:
        return (float(t_mono) - self.vt.c) / DT

    def ready(self, max_age: float = 0.1) -> bool:
        return True

    @property
    def buffer_end(self) -> int:
        return self.end

    @property
    def holding(self) -> bool:
        return self.k >= self.end

    @property
    def command(self):
        return self.last.tolist()

    # -- writes ----------------------------------------------------------------
    def _value_at(self, i: int) -> np.ndarray:
        if i < self.k:
            return self.last if i == self.k - 1 else self.prev
        if i < self.end:
            return self.ring[i]
        return self.ring[self.end - 1] if self.end > self.k else self.last

    def write(self, start: int, Q, timeout: float = 0.1) -> int:
        Q = np.asarray(Q, float).reshape(-1, 6)
        s = int(start)
        with self.lock:
            if s < self.k + self.min_lead:
                self.n_rej += 1
                self.rejects.append(f"too late: start {s}, next tick {self.k}")
                return RES_LATE
            q2, q1 = self._value_at(s - 2), self._value_at(s - 1)
            for r, q in enumerate(Q):
                dq, ddq = q - q1, q - 2 * q1 + q2
                if np.any(np.abs(dq) > self.dq) or np.any(np.abs(ddq) > self.ddq):
                    self.n_rej += 1
                    self.rejects.append(f"limit at tick {s + r}")
                    return RES_LIMIT
                q2, q1 = q1, q
            e = max(self.end, self.k)
            hold = self._value_at(e)
            for i in range(e, s):
                self.ring[i] = hold
            for r, q in enumerate(Q):
                self.ring[s + r] = q
            for i in [i for i in self.ring if i >= s + len(Q)]:
                del self.ring[i]
            self.end = s + len(Q)
            self.n_acc += 1
            return RES_OK

    def stop(self) -> None:
        with self.lock:
            self.end = self.k
            self.events = []

    def set_events(self, from_index: int, events: Sequence[Tuple[int, bool]]) -> None:
        ev = sorted((int(k), bool(on)) for k, on in events)
        with self.lock:
            self.events = [e for e in self.events if e[0] < int(from_index)] + ev

    # -- the sim's side ------------------------------------------------------------
    def at_tick(self, k: int) -> Tuple[Optional[np.ndarray], List[bool]]:
        """Command for physics tick k (None before activation) + suction to fire."""
        with self.lock:
            if not self.active:
                return None, []
            fire = []
            while self.events and self.events[0][0] <= k:
                kk, on = self.events.pop(0)
                fire.append(on)
                self.ev_log.append((kk, on))
            if k < self.k:                # already output (a repeated step): hold
                return self.last, fire
            while self.k <= k:
                cmd = self.ring.pop(self.k, None) if self.k < self.end else None
                self.prev, self.last = self.last, (cmd if cmd is not None else self.last)
                self.k += 1
            return self.last, fire

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
* ``SimTickBuffer``: the SampleBuffer's rules (inherited from
  ``controllers.sample_buffer_rules``, the one Python copy of
  ``sample_buffer.cpp``) with the ``BufferClient`` surface
  ``cycle_throw.runner._BufferOutput`` uses. Tick k is sim time ``k * DT``;
  ``SimCore`` asks ``at_tick(k)`` for the command and the suction events of
  every 2 ms physics step, so no Python timing loop is involved.
"""

from __future__ import annotations

import math
import threading
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

from gp8_control.controllers.sample_buffer_rules import TS as DT
from gp8_control.controllers.sample_buffer_rules import SampleBufferRules


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


class SimTickBuffer(SampleBufferRules):
    """The SampleBuffer on the sim's ticks: the controller's rules (inherited)
    + the ``BufferClient`` surface ``cycle_throw.runner._BufferOutput`` uses.
    Tick k is sim time ``k * DT``; ``SimCore`` calls ``at_tick(k)`` every 2 ms
    physics step for the command and the suction events of that tick."""

    def __init__(self, vt: VirtualTime, q0, **rules) -> None:
        super().__init__(q0, **rules)
        self.vt = vt
        self.lock = threading.Lock()
        self.events: List[Tuple[int, bool]] = []
        self.ev_log: List[Tuple[int, bool]] = []
        self.active = False
        self.reset(q0, k=self._tick_now() + 1)

    def activate(self, q_hold=None) -> bool:
        """SampleBufferController switched on: hold q_hold, output from the next tick."""
        with self.lock:
            self.reset(self.last if q_hold is None else q_hold, k=self._tick_now() + 1)
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
    def command(self):
        return self.last.tolist()

    # -- the app's side (BufferClient surface) ---------------------------------
    def write(self, start: int, Q, timeout: float = 0.1) -> int:
        with self.lock:
            return self.merge(start, Q)[0]

    def stop(self) -> None:
        with self.lock:
            self.stop_now()
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
            while self.k <= k:
                self.output()
            return self.last, fire

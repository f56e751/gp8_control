"""The SampleBuffer controller's rules in Python, in ONE place.

The real rules run in C++ on the robot PC (motoman_controllers/SampleBuffer,
``sample_buffer.cpp``, limits in ``sample_buffer.yaml``). Two Python stand-ins
need the same behaviour and inherit it from here:

* ``backends.sim_tick.SimTickBuffer`` (the twin; ticks = sim steps),
* ``tools.cycle_dry_check._FakeSampleBuffer`` (robot-free check; ticks = wall).

Rules: a chunk ``(start, Q)`` replaces the buffer from ``start`` on (tail
replace); it is refused if ``start < k + min_lead`` (too late) or if any step,
the seam included, exceeds ``|dq| <= v_max*Ts*(1+tol)`` or
``|ddq| <= a_factor*v_max*Ts^2*(1+tol)`` (limit). A gap before ``start`` is
filled with the held value; past the end the last command is held; ``stop``
empties everything after the current tick.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

TS = 0.002
#: sample_buffer.yaml (GP8 datasheet joint speeds 455, 385, 520, 550, 550, 1000 deg/s)
V_MAX = np.array([7.9412480966, 6.7195176202, 9.0757121104, 9.5993108860, 9.5993108860,
                  17.4532925199])
A_FACTOR = 20.0
TOL = 0.02
MIN_LEAD = 3
RES_OK, RES_LATE, RES_LIMIT = 0, 1, 2


class SampleBufferRules:
    """Buffer state + the controller's merge / output rules (no clock, no threads)."""

    def __init__(self, q_hold, *, min_lead: int = MIN_LEAD, a_factor: float = A_FACTOR,
                 tol: float = TOL, v_max=V_MAX) -> None:
        self.dq = np.asarray(v_max, float) * TS * (1 + tol)
        self.ddq = a_factor * np.asarray(v_max, float) * TS * TS * (1 + tol)
        self.min_lead = int(min_lead)
        self.n_acc = self.n_rej = 0
        self.rejects: list = []
        self.reset(q_hold, k=0)

    def reset(self, q_hold, k: int) -> None:
        """Activation: hold q_hold, the next tick to output is k."""
        self.ring: dict[int, np.ndarray] = {}
        self.k = int(k)
        self.end = self.k
        self.last = np.asarray(q_hold, float).copy()
        self.prev = self.last.copy()

    @property
    def holding(self) -> bool:
        return self.k >= self.end

    def value_at(self, i: int) -> np.ndarray:
        if i < self.k:
            return self.last if i == self.k - 1 else self.prev
        if i < self.end:
            return self.ring[i]
        return self.ring[self.end - 1] if self.end > self.k else self.last

    def merge(self, start: int, Q) -> Tuple[int, Optional[str]]:
        """Apply one chunk -> (result code, reason if refused)."""
        Q = np.asarray(Q, float).reshape(-1, 6)
        s = int(start)
        if s < self.k + self.min_lead:
            self.n_rej += 1
            why = f"too late: start {s}, next tick {self.k}"
            self.rejects.append(why)
            return RES_LATE, why
        q2, q1 = self.value_at(s - 2), self.value_at(s - 1)
        for r, q in enumerate(Q):
            dq, ddq = q - q1, q - 2 * q1 + q2
            if np.any(np.abs(dq) > self.dq) or np.any(np.abs(ddq) > self.ddq):
                self.n_rej += 1
                why = f"limit at tick {s + r}"
                self.rejects.append(why)
                return RES_LIMIT, why
            q2, q1 = q1, q
        e = max(self.end, self.k)
        hold = self.value_at(e)
        for i in range(e, s):
            self.ring[i] = hold
        for r, q in enumerate(Q):
            self.ring[s + r] = q
        for i in [i for i in self.ring if i >= s + len(Q)]:
            del self.ring[i]
        self.end = s + len(Q)
        self.n_acc += 1
        return RES_OK, None

    def stop_now(self) -> None:
        self.end = self.k

    def output(self) -> Tuple[int, np.ndarray, bool]:
        """Output tick k: -> (k, command, holding); advances to k + 1."""
        holding = self.holding
        cmd = self.ring.pop(self.k, None) if not holding else None
        cmd = self.last if cmd is None else cmd
        self.prev, self.last = self.last, cmd
        k = self.k
        self.k += 1
        return k, cmd, holding

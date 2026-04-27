"""Frame-cooldown gate for SAM polling.

Independent of the tracked-object queue: this only knows that we should
not re-poll the perception pipeline until the conveyor has carried the
previous frame's contents forward by ``cooldown_distance`` meters. It
exists as its own object so the queue can stay focused on object
lifecycle.
"""

from __future__ import annotations


class FrameGate:
    def __init__(
        self,
        cooldown_distance: float,
        zero_speed_epsilon: float = 1e-4,
    ) -> None:
        """``cooldown_distance``: belt travel between successive SAM polls.

        ``zero_speed_epsilon``: belt speeds at or below this magnitude are
        treated as "stopped" — the cooldown collapses to zero so we can
        re-poll immediately. Negative speeds (which would imply the belt
        is reversing) also bypass the cooldown; they're not a normal
        operating regime so we err on the side of polling.
        """
        self.cooldown_distance = cooldown_distance
        self.zero_speed_epsilon = zero_speed_epsilon
        self._last_frame_time: float | None = None

    def should_poll(self, now: float, conveyor_speed: float) -> bool:
        """True if we have either never polled or the cooldown has elapsed."""
        if self._last_frame_time is None:
            return True
        if conveyor_speed <= self.zero_speed_epsilon:
            return True
        time_to_travel = self.cooldown_distance / conveyor_speed
        return (now - self._last_frame_time) >= time_to_travel

    def mark(self, now: float) -> None:
        self._last_frame_time = now

    def reset(self) -> None:
        """Clear the cooldown timer (typically when the queue empties)."""
        self._last_frame_time = None

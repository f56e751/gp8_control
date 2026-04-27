"""EMA-smoothed pick-cycle overhead tracker.

Each pick we measure ``elapsed - traj_duration`` — the wall-clock cost of
*everything that isn't trajectory execution itself* (action handshake,
serialization, ROS bus latency, etc.). The next pick's planner uses this
as ``fixed_delay`` so its conveyor-motion compensation includes the
overhead.

The first real measurement is adopted directly (no smoothing target to
mix with), and later samples EMA-smooth with ``alpha``. Pathological
samples (goal rejected, clock skew) are clamped out via ``valid_range``
and counted in ``rejected_count`` for diagnostics.
"""

from __future__ import annotations


class PickDelayTracker:
    def __init__(
        self,
        ema_alpha: float,
        valid_range: tuple[float, float] = (-0.5, 2.0),
    ) -> None:
        self.alpha = ema_alpha
        self.valid_range = valid_range
        self._value: float = 0.0
        self._previous: float = 0.0
        self._last_observed: float = 0.0
        self._initialized: bool = False
        self.rejected_count: int = 0

    @property
    def value(self) -> float:
        """The smoothed overhead estimate planners should use as fixed_delay."""
        return self._value

    @property
    def previous(self) -> float:
        """Value before the most recent ``update`` (for logging)."""
        return self._previous

    @property
    def last_observed(self) -> float:
        """Last raw sample, even if it was rejected as out-of-range."""
        return self._last_observed

    @property
    def initialized(self) -> bool:
        return self._initialized

    def update(self, observed: float) -> None:
        """Fold ``observed`` into the EMA. Out-of-range samples are ignored."""
        self._last_observed = observed
        lo, hi = self.valid_range
        if not (lo < observed < hi):
            self.rejected_count += 1
            return
        self._previous = self._value
        if not self._initialized:
            self._value = observed
            self._initialized = True
        else:
            self._value = (1.0 - self.alpha) * self._previous + self.alpha * observed

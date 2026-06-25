from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JointTrajectory:
    joint_names: list[str]
    times: np.ndarray
    positions: np.ndarray

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=np.float64)
        positions = np.asarray(self.positions, dtype=np.float64)

        if times.ndim != 1:
            raise ValueError("times must be a 1D array")
        if positions.ndim != 2:
            raise ValueError("positions must be a 2D array")
        if len(self.joint_names) != positions.shape[1]:
            raise ValueError("joint_names length must match positions column count")
        if len(times) != positions.shape[0]:
            raise ValueError("times length must match positions row count")
        if len(times) < 2:
            raise ValueError("trajectory must contain at least two waypoints")
        if not np.all(np.diff(times) > 0):
            raise ValueError("trajectory times must be strictly increasing")

        object.__setattr__(self, "times", times)
        object.__setattr__(self, "positions", positions)

    @property
    def duration(self) -> float:
        return float(self.times[-1] - self.times[0])

    def sample(self, t: float) -> np.ndarray:
        t = float(t)
        if t <= self.times[0]:
            return self.positions[0].copy()
        if t >= self.times[-1]:
            return self.positions[-1].copy()

        upper_idx = int(np.searchsorted(self.times, t, side="right"))
        lower_idx = upper_idx - 1
        t0 = self.times[lower_idx]
        t1 = self.times[upper_idx]
        q0 = self.positions[lower_idx]
        q1 = self.positions[upper_idx]
        alpha = (t - t0) / (t1 - t0)
        return ((1.0 - alpha) * q0 + alpha * q1).astype(np.float64, copy=True)

    @classmethod
    def from_waypoints(
        cls,
        joint_names: list[str],
        times: list[float] | np.ndarray,
        positions: list[list[float]] | np.ndarray,
    ) -> "JointTrajectory":
        return cls(
            joint_names=list(joint_names),
            times=np.asarray(times, dtype=np.float64),
            positions=np.asarray(positions, dtype=np.float64),
        )

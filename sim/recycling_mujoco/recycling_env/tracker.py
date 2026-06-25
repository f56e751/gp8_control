from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .perception import Detection


@dataclass(frozen=True)
class TrackingState:
    detected: bool
    error_px: np.ndarray
    aligned: bool
    joint_delta: np.ndarray
    q_target: np.ndarray


class BoundingBoxTracker:
    """Simple image-plane tracker that maps bbox center error to joint corrections."""

    def __init__(
        self,
        target_px: tuple[float, float],
        nominal_q: np.ndarray,
        joint_names: list[str],
        control_joint_names: list[str] | None = None,
        tolerance_px: tuple[float, float] = (12.0, 12.0),
        gains: dict[str, float] | None = None,
        max_delta: dict[str, float] | None = None,
    ) -> None:
        self.target_px = np.asarray(target_px, dtype=np.float64)
        self.nominal_q = np.asarray(nominal_q, dtype=np.float64)
        self.joint_names = list(joint_names)
        self.joint_index = {name: idx for idx, name in enumerate(self.joint_names)}
        self.control_joint_names = control_joint_names or ["S_axis", "L_axis"]
        self.tolerance_px = np.asarray(tolerance_px, dtype=np.float64)
        self.gains = gains or {"S_axis": -0.0025, "L_axis": 0.0015}
        self.max_delta = max_delta or {"S_axis": 0.6, "L_axis": 0.5}

    def compute_error(self, detection: Detection) -> np.ndarray:
        if not detection.detected or detection.center_px is None:
            return np.array([np.nan, np.nan], dtype=np.float64)
        center = np.asarray(detection.center_px, dtype=np.float64)
        return center - self.target_px

    def is_aligned(self, detection: Detection) -> bool:
        error = self.compute_error(detection)
        if np.isnan(error).any():
            return False
        return bool(np.all(np.abs(error) <= self.tolerance_px))

    def compute_joint_delta(self, detection: Detection) -> np.ndarray:
        delta = np.zeros_like(self.nominal_q, dtype=np.float64)
        if not detection.detected or detection.center_px is None:
            return delta

        error = self.compute_error(detection)
        dx, dy = float(error[0]), float(error[1])

        if "S_axis" in self.joint_index:
            idx = self.joint_index["S_axis"]
            gain = float(self.gains.get("S_axis", 0.0))
            limit = float(self.max_delta.get("S_axis", np.inf))
            delta[idx] = np.clip(gain * dx, -limit, limit)

        if "L_axis" in self.joint_index:
            idx = self.joint_index["L_axis"]
            gain = float(self.gains.get("L_axis", 0.0))
            limit = float(self.max_delta.get("L_axis", np.inf))
            delta[idx] = np.clip(gain * dy, -limit, limit)

        return delta

    def compute_target(self, detection: Detection) -> TrackingState:
        error = self.compute_error(detection)
        aligned = self.is_aligned(detection)
        delta = self.compute_joint_delta(detection)
        q_target = self.nominal_q + delta
        return TrackingState(
            detected=bool(detection.detected),
            error_px=error,
            aligned=aligned,
            joint_delta=delta,
            q_target=q_target,
        )

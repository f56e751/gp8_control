"""Perception latency selection helpers with legacy-producer fallback."""

from __future__ import annotations

import math


def select_capture_age(reported, estimated: float, max_age_s: float = 2.0):
    """Prefer a valid RealSense frame age; otherwise return the EMA estimate."""
    try:
        value = float(reported)
    except (TypeError, ValueError):
        return float(estimated), "estimated_frame_period"
    if not math.isfinite(value) or value < 0.0 or value > max_age_s:
        return float(estimated), "estimated_frame_period"
    return value, "realsense_global_time"

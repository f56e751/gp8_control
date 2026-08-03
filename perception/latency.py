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


def live_capture_age(
    receipt_time_s,
    capture_timestamp_s,
    server_minus_client_offset_s,
    max_age_s: float = 2.0,
):
    """Return capture-to-client-receipt age, or ``None`` if unusable.

    Producer timestamps use the server clock.  Subtracting the NTP-style
    server-minus-client offset expresses capture time on the client's clock.
    """
    try:
        receipt = float(receipt_time_s)
        capture = float(capture_timestamp_s)
        offset = float(server_minus_client_offset_s)
    except (TypeError, ValueError):
        return None
    age = receipt - (capture - offset)
    if not math.isfinite(age) or age < 0.0 or age > float(max_age_s):
        return None
    return age

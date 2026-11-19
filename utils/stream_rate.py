"""Joint-command stream rate shared by the stream engine and the GP8 model.

The YRC1000micro interpolates every 2 ms. The adv4ncr driver on the robot PC
is configured for 250 Hz (``controllers.yaml`` ``update_rate: 250``,
``motoman_hardware`` ``nominal_dt = 0.004``), so the default stays 250 Hz.

At 250 Hz every other controller cycle receives no new command, which caps
the arm at exactly half its datasheet speed: the "measured" RT ceilings in
``robots/gp8.py`` (2026-07-03) are datasheet x 0.5 to within 0.5 %. The
grasp-and-throw cycle planner (``cycle_throw``) needs the full 2 ms stream.

``GP8_STREAM_HZ=500`` switches the app to 2 ms samples. Set it ONLY after the
driver itself runs at 500 Hz (``update_rate: 500`` and a 0.002 s control
period); a 500 Hz stream into a 250 Hz driver drops every other sample.
"""

from __future__ import annotations

import os

_ALLOWED = (250.0, 500.0)


def _read_stream_hz() -> float:
    raw = os.environ.get("GP8_STREAM_HZ", "250").strip() or "250"
    try:
        hz = float(raw)
    except ValueError as exc:
        raise ValueError(f"GP8_STREAM_HZ={raw!r}: expected 250 or 500") from exc
    if hz not in _ALLOWED:
        raise ValueError(f"GP8_STREAM_HZ={raw!r}: expected 250 or 500")
    return hz


STREAM_HZ: float = _read_stream_hz()
STREAM_DT: float = 1.0 / STREAM_HZ

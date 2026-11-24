"""Piece-mode control settings from ``config/piece_mode.env`` (one place).

The robot gets them from ``scripts/cycle_demo.sh`` (exported before the app
starts). The twin and the RL env call ``apply()``: it sets every value that the
environment does not already set, and also updates the places that read their
value once at import (``skills.push_skill`` module constants) or at Config
creation (``cfg.PUSH_MAX_REACH``), so the order of imports does not matter.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve().parent.parent / "config" / "piece_mode.env"


def values(path: Path | None = None) -> dict:
    p = Path(os.environ.get("GP8_PIECE_SETTINGS", "") or (path or FILE))
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def apply(cfg=None) -> dict:
    """Environment defaults + the already-read copies; returns the values in force."""
    for k, v in values().items():
        os.environ.setdefault(k, v)
    eff = {k: os.environ[k] for k in values()}
    ps = sys.modules.get("gp8_control.skills.push_skill")
    if ps is not None and "GP8_PUSH_ACCEL" in eff:
        ps.PUSH_ACCEL = float(eff["GP8_PUSH_ACCEL"])
    if cfg is not None and "GP8_PUSH_MAX_REACH" in eff:
        cfg.PUSH_MAX_REACH = float(eff["GP8_PUSH_MAX_REACH"])
    return eff

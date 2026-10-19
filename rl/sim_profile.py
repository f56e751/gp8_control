"""Named simulator profiles + the hardware observation/latency models.

``GP8_SIM_PROFILE=hw2026`` makes the RL training sim look like the GP8 cell as
measured on 2026-10-19 (s2r validation runs + the camera PC's detection log):

* belt 0.10-0.30 m/s (HW ran 0.13-0.28; the old training range stopped at 0.20);
* arrivals 0.2-0.7 Hz (HW 0.29 Hz average, 0.69 Hz while feeding);
* objects as the policy SEES them on HW (observation bbox half sizes: short
  1.4-6.2 cm, long 4.5-9.9 cm, belt-aligned), lane x 0.42-0.60 m (HW p5-p95,
  inside the sim's 0.20-0.63 m belt);
* axis-aligned bboxes (the detector never reports an oriented box);
* camera-like observation noise and the control loop's decision latency
  (``ObsNoiseModel`` / ``DecisionLatencyModel`` below).

A profile only fills environment defaults (``os.environ.setdefault``), so any
explicitly exported ``GP8_SIM_*`` / ``GP8_RL_*`` value still wins and the
default (no profile) training setup is unchanged.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

PROFILES: dict[str, dict[str, str]] = {
    "hw2026": {
        "GP8_SIM_RANDOMIZE": "1",
        "GP8_SIM_BELT_SPEED_RANGE": "0.10,0.30",
        "GP8_SIM_SPAWN_RATE_HZ_RANGE": "0.2,0.7",
        "GP8_SIM_OBJECT_HALF_X_RANGE": "0.015,0.05",
        "GP8_SIM_OBJECT_HALF_Y_RANGE": "0.045,0.10",
        "GP8_SIM_RANDOM_YAW": "0",
        "GP8_SIM_SPAWN_X_RANGE": "0.42,0.60",
        "GP8_SIM_BBOX_MODE": "axis",
        "GP8_RL_OBS_NOISE": "hw2026",
        "GP8_RL_DECISION_LATENCY": "hw2026",
    },
}

PROFILE_ENV = "GP8_SIM_PROFILE"


def apply_sim_profile(name: str | None = None) -> dict[str, str]:
    """Fill env defaults for ``name`` (default: ``$GP8_SIM_PROFILE``).

    Returns the keys this call actually set. Unknown names raise, so a typo
    can't silently train on the default profile.
    """
    name = (name if name is not None else os.environ.get(PROFILE_ENV, "")).strip()
    if not name:
        return {}
    if name not in PROFILES:
        raise ValueError(f"unknown {PROFILE_ENV}={name!r}; known: {sorted(PROFILES)}")
    applied = {}
    for key, value in PROFILES[name].items():
        if key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied


# ---------------------------------------------------------------------------
# Observation noise (truth tracks -> camera-like tracks)
# ---------------------------------------------------------------------------
# HW 2026-10-19, 651 detections over two runs.
_HW_CONF_QUANTILES = (
    (0.00, 0.507), (0.05, 0.615), (0.25, 0.823), (0.50, 0.913),
    (0.75, 0.952), (0.95, 0.967), (1.00, 0.974),
)


@dataclass
class ObsNoiseModel:
    """Camera-like corruption of simulator truth tracks.

    Per object (fixed for its lifetime): a position bias -- the control anchor
    error that remains after the centre-frame latch (HW: ~1.7 cm along-belt
    std, tails to ~7 cm). Per refresh: frame jitter on the bbox only (the
    anchor/xyz is latched on HW, the drawn box keeps refreshing) and a fresh
    confidence drawn from the HW distribution (sim used a constant 0.9).
    """

    bias_y_std: float = 0.017
    bias_y_max: float = 0.07
    bias_x_std: float = 0.004
    jitter_y_std: float = 0.0056
    jitter_x_std: float = 0.0013
    # The drawn bbox is not the anchor: on HW the anchor is latched at the
    # centre frame while the bbox is the last (exit-edge) frame, so the box
    # centre sits off xyz by a per-object offset (HW obs: y +24 +- 17 mm,
    # x +11 +- 16 mm).
    bbox_off_y_mean: float = 0.024
    bbox_off_y_std: float = 0.017
    bbox_off_x_mean: float = 0.011
    bbox_off_x_std: float = 0.016
    conf_quantiles: tuple = _HW_CONF_QUANTILES

    @classmethod
    def from_env(cls) -> "ObsNoiseModel | None":
        mode = os.environ.get("GP8_RL_OBS_NOISE", "").strip().lower()
        if mode in ("", "0", "off", "none", "false"):
            return None
        if mode == "hw2026":
            return cls()
        raise ValueError(f"unknown GP8_RL_OBS_NOISE={mode!r}")

    def object_bias(self, rng: np.random.Generator) -> tuple[float, float, float, float]:
        """Per-object ``(anchor_dx, anchor_dy, bbox_dx, bbox_dy)``; bbox offsets
        are relative to the (biased) anchor."""
        by = float(np.clip(rng.normal(0.0, self.bias_y_std), -self.bias_y_max, self.bias_y_max))
        bx = float(rng.normal(0.0, self.bias_x_std))
        oy = float(rng.normal(self.bbox_off_y_mean, self.bbox_off_y_std))
        ox = float(rng.normal(self.bbox_off_x_mean, self.bbox_off_x_std))
        return bx, by, ox, oy

    def frame_jitter(self, rng: np.random.Generator) -> tuple[float, float]:
        return float(rng.normal(0.0, self.jitter_x_std)), float(rng.normal(0.0, self.jitter_y_std))

    def confidence(self, rng: np.random.Generator) -> float:
        qs, vs = zip(*self.conf_quantiles)
        return float(np.interp(rng.uniform(0.0, 1.0), qs, vs))


# ---------------------------------------------------------------------------
# Decision latency (policy decision -> first motion command)
# ---------------------------------------------------------------------------
@dataclass
class DecisionLatencyModel:
    """Wall time the real loop spends between a decision and the first motion
    sample (mask/intercept/trajectory compute). A non-realtime sim spends zero
    sim time there, so the belt never moves while the robot "thinks".

    HW 2026-10-19: median ~40 ms, p95 ~200 ms -> lognormal(median, sigma) with
    sigma = ln(200/40)/1.645, capped.
    """

    median_s: float = 0.040
    p95_s: float = 0.200
    cap_s: float = 0.5

    @classmethod
    def from_env(cls) -> "DecisionLatencyModel | None":
        mode = os.environ.get("GP8_RL_DECISION_LATENCY", "").strip().lower()
        if mode in ("", "0", "off", "none", "false"):
            return None
        if mode == "hw2026":
            return cls()
        try:
            fixed = float(mode)
        except ValueError as exc:
            raise ValueError(f"unknown GP8_RL_DECISION_LATENCY={mode!r}") from exc
        return cls(median_s=fixed, p95_s=fixed, cap_s=max(fixed, 0.0))

    def sample(self, rng: np.random.Generator) -> float:
        if self.p95_s <= self.median_s:
            return float(min(self.median_s, self.cap_s))
        sigma = math.log(self.p95_s / self.median_s) / 1.645
        return float(min(rng.lognormal(math.log(self.median_s), sigma), self.cap_s))

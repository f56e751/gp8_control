"""Single source of truth for throw NN post-processing.

The torch NN returns a flat parameter vector ``params`` whose last two
elements encode trajectory time (log-scale) and release fraction
(logit-scale). We previously decoded these in three places (the planner's
inner refinement loop, the planner's final query, and the executor that
builds the trajectory) — all three must agree, so collapse them into one
``ThrowParams.from_raw`` call.

Encoding (preserved from main_sam7):
    T   = exp(params[-2]) * throw_time_scale
    eta = sigmoid(params[-1])
    eta = clip(eta - release_early_shift, eta_min, eta_max)
    w   = params[:-2].reshape(-1, 5)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThrowDecodingConfig:
    throw_time_scale: float
    release_early_shift: float
    eta_min: float
    eta_max: float


@dataclass(frozen=True)
class ThrowParams:
    w: np.ndarray   # (n_basis, 5)
    T: float        # trajectory duration, seconds
    eta: float      # release fraction in [eta_min, eta_max]

    @classmethod
    def from_raw(cls, raw: np.ndarray, cfg: ThrowDecodingConfig) -> "ThrowParams":
        T = float(np.exp(raw[-2])) * cfg.throw_time_scale
        eta = 1.0 / (1.0 + np.exp(-raw[-1]))
        eta = float(np.clip(eta - cfg.release_early_shift, cfg.eta_min, cfg.eta_max))
        w = raw[:-2].reshape(-1, 5)
        return cls(w=w, T=T, eta=eta)

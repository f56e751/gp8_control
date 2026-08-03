"""Vendored Decision-Transformer thrower from the Thr_DT reproduction.

Source: /PublicSSD/ryugaeun/Thr_DT (re-implementation of Monastirsky, Azulay &
Sintov, "Learning to Throw With a Handful of Samples Using Decision
Transformers", IEEE RA-L 8(2):576-583, 2023 — doi 10.1109/LRA.2022.3229266).

Only the *inference* half is vendored (config + planar env + DT model +
weights); data collection, training and evaluation stay in the Thr_DT repo.
See README.md for the exact provenance and the two-line import diff.
"""

from gp8_control.skills.thr_dt.dt_config import ModelConfig, SimConfig
from gp8_control.skills.thr_dt.decision_transformer import (
    build_model,
    count_parameters,
)
from gp8_control.skills.thr_dt.throw_env import RoboticArm

__all__ = [
    "ModelConfig",
    "SimConfig",
    "RoboticArm",
    "build_model",
    "count_parameters",
]

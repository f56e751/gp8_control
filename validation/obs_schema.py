"""The RL observation schema the validator records and decodes.

Defaults are the training baseline from docs/rl_simulator_status_2026-10-10.md
(6 slots, oriented bbox corners, no suction_p, no ETA -> width 91), NOT the
runner defaults: the point is to see what the *trained* policy would be fed.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import numpy as np

from gp8_control.rl.common import (
    BBOX_OBSERVATION_CORNERS,
    normalize_bbox_observation,
    observation_width,
)

N_GLOBAL = 13
GLOBAL_NAMES = (
    [f"q{i + 1}" for i in range(6)]
    + ["ee_x", "ee_y", "ee_z", "pending_slot", "pending_skill", "belt_speed",
       "remaining_time_frac"]
)


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class ObsSchema:
    max_objects: int = 6
    bbox_observation: str = BBOX_OBSERVATION_CORNERS
    include_suction_p: bool = False
    include_eta: bool = False

    @classmethod
    def from_env(cls) -> "ObsSchema":
        return cls(
            max_objects=int(os.environ.get("GP8_S2R_MAX_OBJECTS", "6")),
            bbox_observation=normalize_bbox_observation(
                os.environ.get("GP8_S2R_BBOX_OBSERVATION", BBOX_OBSERVATION_CORNERS)),
            include_suction_p=_env_bool("GP8_S2R_INCLUDE_SUCTION_P", False),
            include_eta=False,
        )

    @property
    def width(self) -> int:
        return observation_width(
            self.max_objects, self.include_eta, self.bbox_observation,
            self.include_suction_p)

    @property
    def object_fields(self) -> list[str]:
        names = ["x", "y", "z", "class_id", "conf"]
        if self.bbox_observation == BBOX_OBSERVATION_CORNERS:
            names += [f"c{i}{a}" for i in range(4) for a in ("x", "y")]
        else:
            names += ["bbox_w", "bbox_h"]
        if self.include_suction_p:
            names.append("suction_p")
        if self.include_eta:
            names += ["throw_eta", "push_eta"]
        return names

    @property
    def feature_names(self) -> list[str]:
        per = self.object_fields
        names = [f"o{s}.{f}" for s in range(self.max_objects) for f in per]
        return names + GLOBAL_NAMES

    def to_dict(self) -> dict:
        return asdict(self)

    def decode(self, obs) -> dict:
        """Split a flat observation into {'objects': [dict|None], 'globals': dict}."""
        obs = np.asarray(obs, dtype=float).ravel()
        per = self.object_fields
        w = len(per)
        objects = []
        for s in range(self.max_objects):
            row = dict(zip(per, obs[s * w:(s + 1) * w].tolist()))
            # The empty-slot sentinel (rl/common.empty_object_row). A live object
            # whose class is unknown to CLASS_IDS also encodes class_id=-1, so
            # don't key emptiness on the class alone.
            empty = (row["x"] == -1.0 and row["y"] == -1.0
                     and row["class_id"] == -1.0 and row["conf"] == 0.0)
            objects.append(None if empty else row)
        g = obs[self.max_objects * w:]
        return {"objects": objects, "globals": dict(zip(GLOBAL_NAMES, g.tolist()))}

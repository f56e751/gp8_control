"""Shared high-level RL schema and observation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from gp8_control.tracking import TrackedObject


ACTION_THROW = 0
ACTION_PUSH = 1
SKILL_NAMES = ("throw", "push")
BBOX_OBSERVATION_SIZE = "size"
BBOX_OBSERVATION_CORNERS = "corners"
BBOX_OBSERVATION_MODES = (BBOX_OBSERVATION_SIZE, BBOX_OBSERVATION_CORNERS)

CLASS_IDS = {
    "metal": 0,
    "transparent": 1,
    "cardboard": 2,
}


@dataclass(frozen=True)
class HighLevelAction:
    """High-level action stored by object identity, not transient slot order."""

    target: Optional[TrackedObject]
    skill_name: str


def class_id(class_name: str) -> int:
    return CLASS_IDS.get(str(class_name), -1)


def bbox_size(target: TrackedObject) -> tuple[float, float]:
    bbox = target.base_bbox_grasp
    if not bbox:
        return 0.0, 0.0
    points = np.asarray(bbox, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2:
        return 0.0, 0.0
    return (
        float(np.max(points[:, 0]) - np.min(points[:, 0])),
        float(np.max(points[:, 1]) - np.min(points[:, 1])),
    )


def normalize_bbox_observation(mode: str | None) -> str:
    value = (mode or BBOX_OBSERVATION_SIZE).strip().lower()
    if value not in BBOX_OBSERVATION_MODES:
        raise ValueError(
            f"bbox_observation must be one of {BBOX_OBSERVATION_MODES}, got {mode!r}"
        )
    return value


def bbox_corners(target: TrackedObject) -> list[float]:
    bbox = target.base_bbox_grasp
    if not bbox:
        return [0.0] * 8
    points = np.asarray(bbox, dtype=float)
    if points.shape != (4, 3) and points.shape != (4, 2):
        return [0.0] * 8
    return points[:, :2].astype(float).reshape(-1).tolist()


def bbox_features(target: TrackedObject, bbox_observation: str) -> list[float]:
    mode = normalize_bbox_observation(bbox_observation)
    if mode == BBOX_OBSERVATION_CORNERS:
        return bbox_corners(target)
    bbox_w, bbox_h = bbox_size(target)
    return [bbox_w, bbox_h]


def empty_object_row(
    include_eta: bool,
    bbox_observation: str = BBOX_OBSERVATION_SIZE,
    include_suction_p: bool = False,
) -> list[float]:
    bbox_width = (
        8
        if normalize_bbox_observation(bbox_observation) == BBOX_OBSERVATION_CORNERS
        else 2
    )
    row = [-1.0, -1.0, 0.0, -1.0, 0.0] + [0.0] * bbox_width
    if include_suction_p:
        row.append(0.0)
    if include_eta:
        row.extend([-1.0, -1.0])
    return row


def observation_width(
    max_objects: int,
    include_eta: bool,
    bbox_observation: str = BBOX_OBSERVATION_SIZE,
    include_suction_p: bool = False,
) -> int:
    bbox_width = (
        8
        if normalize_bbox_observation(bbox_observation) == BBOX_OBSERVATION_CORNERS
        else 2
    )
    per_object_width = (
        5
        + bbox_width
        + (1 if include_suction_p else 0)
        + (2 if include_eta else 0)
    )
    return int(max_objects) * per_object_width + 6 + 3 + 4


def build_observation(
    *,
    objects: Sequence[TrackedObject],
    max_objects: int,
    include_eta: bool,
    joints,
    ee_xyz,
    pending_indices: tuple[int, int],
    belt_speed: float,
    y_now_for,
    remaining_time_frac: float = 1.0,
    etas_for=None,
    bbox_observation: str = BBOX_OBSERVATION_SIZE,
    include_suction_p: bool = False,
) -> np.ndarray:
    """Encode the project-standard Gym-style observation vector."""
    bbox_observation = normalize_bbox_observation(bbox_observation)
    joint_arr = (
        np.zeros(6, dtype=np.float32)
        if joints is None
        else np.asarray(joints, dtype=np.float32)[:6]
    )
    ee_arr = np.asarray(ee_xyz, dtype=np.float32).reshape(-1)[:3]
    if ee_arr.size < 3:
        ee_arr = np.pad(ee_arr, (0, 3 - ee_arr.size))

    features: list[float] = []
    limited = list(objects)[: int(max_objects)]
    for target in limited:
        row = [
            float(target.T_grasp_base[0, 3]),
            float(y_now_for(target)),
            float(target.T_grasp_base[2, 3]),
            float(class_id(target.class_name)),
            float(target.conf),
            *bbox_features(target, bbox_observation),
        ]
        if include_suction_p:
            row.append(float(np.clip(float(getattr(target, "suction_p", 1.0)), 0.0, 1.0)))
        if include_eta:
            row.extend(etas_for(target) if etas_for is not None else [-1.0, -1.0])
        features.extend(row)

    missing = int(max_objects) - len(limited)
    if missing > 0:
        features.extend(
            empty_object_row(include_eta, bbox_observation, include_suction_p) * missing
        )

    pending_slot, pending_skill = pending_indices
    remaining = float(np.clip(float(remaining_time_frac), 0.0, 1.0))
    globals_ = [
        *joint_arr.tolist(),
        *ee_arr.tolist(),
        float(pending_slot),
        float(pending_skill),
        float(belt_speed),
        remaining,
    ]
    obs = np.asarray(features + globals_, dtype=np.float32)
    expected = observation_width(
        max_objects,
        include_eta,
        bbox_observation,
        include_suction_p,
    )
    if obs.size != expected:
        raise RuntimeError(f"observation width mismatch {obs.size} != {expected}")
    return obs


def skip_action(max_objects: int) -> np.ndarray:
    return np.array([int(max_objects), ACTION_THROW], dtype=int)


def action_summary(action: np.ndarray | Sequence[int]) -> dict:
    slot = int(action[0])
    skill = int(action[1])
    return {
        "object_slot": slot,
        "skill_index": skill,
        "skill_name": SKILL_NAMES[skill] if 0 <= skill < len(SKILL_NAMES) else None,
    }

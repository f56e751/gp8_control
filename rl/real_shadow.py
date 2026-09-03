"""Read-only Gym-style shadow interface for the real robot app."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from gp8_control.rl.common import (
    ACTION_THROW,
    SKILL_NAMES,
    action_summary,
    build_observation,
    skip_action,
)


class RealShadowRl:
    """Gym-shaped read-only view over ``GP8App`` state.

    This helper intentionally does not execute policy actions. It uses the same
    queue, joints, context, selector, and skills that ``app.py`` will use for
    hardware execution, then logs what the high-level RL interface would see.
    """

    def __init__(
        self,
        app,
        *,
        max_objects: int = 6,
        include_eta: bool = False,
        log_path: str | None = None,
    ) -> None:
        self.app = app
        self.max_objects = int(max_objects)
        self.include_eta = bool(include_eta)
        self._log_path = Path(log_path).expanduser() if log_path else None
        self._log_file = None
        if self._log_path is not None:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = self._log_path.open("a", encoding="utf-8")

    @classmethod
    def from_env(cls, app):
        enabled = os.environ.get("GP8_RL_SHADOW", "0").lower() in ("1", "true", "yes")
        if not enabled:
            return None
        max_objects = int(os.environ.get("GP8_RL_MAX_OBJECTS", "6"))
        include_eta = os.environ.get("GP8_RL_INCLUDE_ETA", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        return cls(
            app,
            max_objects=max_objects,
            include_eta=include_eta,
            log_path=os.environ.get("GP8_RL_SHADOW_LOG") or "rl_shadow.jsonl",
        )

    @property
    def log_path(self) -> str | None:
        return None if self._log_path is None else str(self._log_path)

    def close(self) -> None:
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def ordered_objects(self):
        return list(self.app.queue._objects)[: self.max_objects]

    def current_joints(self) -> np.ndarray | None:
        if self.app.traj_ctrl is None or self.app.traj_ctrl.current_joints is None:
            return None
        return np.asarray(self.app.traj_ctrl.current_joints, dtype=float)

    def ee_xyz(self, joints: np.ndarray | None = None) -> np.ndarray:
        if joints is None:
            joints = self.current_joints()
        if joints is None:
            return np.zeros(3, dtype=np.float32)
        transform = self.app.robot.forward_kinematics(np.asarray(joints, dtype=float)[:6])
        return np.asarray(transform[:3, 3], dtype=np.float32)

    def pending_indices(self) -> tuple[int, int]:
        return self.max_objects, ACTION_THROW

    def observation(self, include_eta: bool | None = None) -> np.ndarray:
        include_eta = self.include_eta if include_eta is None else bool(include_eta)
        now = time.time()
        joints = self.current_joints()
        return build_observation(
            objects=self.ordered_objects(),
            max_objects=self.max_objects,
            include_eta=include_eta,
            joints=joints,
            ee_xyz=self.ee_xyz(joints),
            pending_indices=self.pending_indices(),
            belt_speed=self.app.conveyor.current,
            y_now_for=lambda target: self.app.ctx.object_y_now(
                target, now, self.app.conveyor.current
            ),
            etas_for=lambda target: self._etas_for(target, joints, now),
        )

    def action_mask(self) -> np.ndarray:
        now = time.time()
        current_joint = self.current_joints()
        mask = np.zeros((self.max_objects + 1, len(SKILL_NAMES)), dtype=np.uint8)
        mask[self.max_objects, :] = 1
        if current_joint is None:
            return mask
        for slot, target in enumerate(self.ordered_objects()):
            for skill_index, skill_name in enumerate(SKILL_NAMES):
                feasible, _ = self._evaluate_feasibility(
                    target,
                    skill_name,
                    current_joint,
                    now,
                )
                mask[slot, skill_index] = int(feasible)
        return mask

    def app_heuristic_action_indices(self, mask: np.ndarray | None = None) -> np.ndarray:
        if mask is None:
            mask = self.action_mask()
        for slot, target in enumerate(self.ordered_objects()):
            skill_name = self.app.selector.skill_for(target)
            if skill_name not in SKILL_NAMES:
                continue
            skill_index = SKILL_NAMES.index(skill_name)
            if mask[slot, skill_index]:
                return np.array([slot, skill_index], dtype=int)
        return skip_action(self.max_objects)

    def snapshot(self) -> dict[str, Any]:
        obs = self.observation()
        mask = self.action_mask()
        heuristic_action = self.app_heuristic_action_indices(mask)
        return {
            "ts": time.time(),
            "observation": obs.tolist(),
            "action_mask": mask.tolist(),
            "heuristic_action": action_summary(heuristic_action),
            "pending_action": {
                "object_slot": self.max_objects,
                "skill_index": ACTION_THROW,
                "skill_name": SKILL_NAMES[ACTION_THROW],
            },
            "objects": [
                {
                    "slot": slot,
                    "track_id": getattr(obj, "track_id", None),
                    "class_name": obj.class_name,
                    "confidence": float(obj.conf),
                }
                for slot, obj in enumerate(self.ordered_objects())
            ],
        }

    def attach_app_selection(
        self,
        record: dict[str, Any],
        *,
        selected_request=None,
        selected_skill=None,
    ) -> dict[str, Any]:
        if selected_request is not None:
            target = selected_request.target
            record["app_selected"] = {
                "track_id": getattr(target, "track_id", None),
                "class_name": target.class_name,
                "skill_name": None if selected_skill is None else selected_skill.name,
            }
        else:
            record["app_selected"] = None
        return record

    def write_record(self, record: dict[str, Any]) -> None:
        if self._log_file is not None:
            self._log_file.write(json.dumps(record, separators=(",", ":")) + "\n")
            self._log_file.flush()

    def log_epoch(self, *, epoch: int, selected_request=None, selected_skill=None) -> None:
        record = self.snapshot()
        record["epoch"] = int(epoch)
        self.attach_app_selection(
            record,
            selected_request=selected_request,
            selected_skill=selected_skill,
        )
        self.write_record(record)

    def _evaluate_feasibility(
        self,
        target,
        skill_name: str,
        current_joint: np.ndarray,
        now: float,
    ) -> tuple[bool, float]:
        skill = self.app.selector.skills.get(skill_name)
        if skill is None:
            return False, -1.0
        intercept = self.app.ctx.earliest_reachable_intercept(
            target,
            np.asarray(current_joint, dtype=float),
            self.app.conveyor.current,
            now,
            skill=skill,
        )
        if intercept is None:
            return False, -1.0
        if skill.placement_veto(target, intercept) is not None:
            return False, -1.0
        return True, float(intercept.eta)

    def _etas_for(
        self,
        target,
        current_joint: np.ndarray | None,
        now: float,
    ) -> list[float]:
        if current_joint is None:
            return [-1.0, -1.0]
        return [
            self._evaluate_feasibility(target, "throw", current_joint, now)[1],
            self._evaluate_feasibility(target, "push", current_joint, now)[1],
        ]

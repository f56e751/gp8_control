"""Smoke checks for the real-app RL shadow interface."""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from gp8_control.config import Config
from gp8_control.planning.action_selector import ActionSelector
from gp8_control.rl.common import ACTION_PUSH, ACTION_THROW, SKILL_NAMES
from gp8_control.rl.real_shadow import RealShadowRl
from gp8_control.tracking import TrackedObject


class _Skill:
    def __init__(self, name: str, feasible_classes: set[str] | None = None) -> None:
        self.name = name
        self._feasible_classes = feasible_classes

    def can_handle(self, target) -> bool:
        return True

    def placement_veto(self, target, intercept) -> str | None:
        del intercept
        return None if self._feasible_classes is None or target.class_name in self._feasible_classes else "veto"


class _Ctx:
    def object_y_now(self, target, now, belt_speed):
        del now
        return float(target.T_grasp_base[1, 3]) - 0.1 * float(belt_speed)

    def earliest_reachable_intercept(
        self,
        target,
        current_joint,
        belt_speed,
        now,
        *,
        skill,
        pre_delay=0.0,
    ):
        del current_joint, belt_speed, now, pre_delay
        if target.class_name == "uncatchable":
            return None
        pose = target.T_grasp_base.copy()
        return SimpleNamespace(
            eta=1.25,
            T_aim=pose.copy(),
            T_grasp=pose.copy(),
            aim_joint=np.zeros(6),
            grasp_joint=np.zeros(6),
        )


def _object(class_name: str) -> TrackedObject:
    pose = np.eye(4)
    pose[0, 3] = 0.45
    pose[1, 3] = 0.30
    pose[2, 3] = 0.042
    return TrackedObject(
        T_aim_base=pose.copy(),
        T_grasp_base=pose.copy(),
        class_name=class_name,
        detect_time=time.time(),
        conf=0.8,
        base_bbox_grasp=((0.40, 0.20, 0.04), (0.50, 0.35, 0.04)),
    )


def _shadow(objects) -> RealShadowRl:
    skills = {
        "throw": _Skill("throw"),
        "push": _Skill("push", {"metal"}),
    }
    cfg = Config()
    app = SimpleNamespace(
        queue=SimpleNamespace(_objects=list(objects)),
        traj_ctrl=SimpleNamespace(current_joints=[0.0] * 6),
        conveyor=SimpleNamespace(current=0.12),
        ctx=_Ctx(),
        robot=SimpleNamespace(
            forward_kinematics=lambda joints: np.eye(4),
        ),
    )
    app.selector = ActionSelector(
        skills.values(),
        default="throw",
        by_class=cfg.SKILL_BY_CLASS,
        force=None,
    )
    return RealShadowRl(app, max_objects=3, include_eta=True)


def main(argv=None) -> int:
    del argv
    metal = _object("metal")
    transparent = _object("transparent")
    uncatchable = _object("uncatchable")
    shadow = _shadow([metal, transparent, uncatchable])

    mask = shadow.action_mask()
    obs = shadow.observation()
    action = shadow.app_heuristic_action_indices(mask)
    snapshot = shadow.snapshot()

    checks = [
        ("mask shape", mask.shape == (4, len(SKILL_NAMES))),
        ("skip row valid", bool(mask[3, ACTION_THROW]) and bool(mask[3, ACTION_PUSH])),
        ("metal push valid", bool(mask[0, ACTION_PUSH])),
        ("transparent push vetoed", not bool(mask[1, ACTION_PUSH])),
        ("uncatchable masked", not bool(mask[2, ACTION_THROW]) and not bool(mask[2, ACTION_PUSH])),
        ("heuristic selects metal push", action.tolist() == [0, ACTION_PUSH]),
        ("observation width", obs.shape == (3 * 9 + 6 + 3 + 3,)),
        ("snapshot has heuristic action", snapshot["heuristic_action"]["skill_name"] == "push"),
        ("pending is noop in shadow v1", snapshot["pending_action"]["object_slot"] == 3),
    ]
    shadow.close()

    failures = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if failures:
        print("FAILURES:", ", ".join(failures))
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

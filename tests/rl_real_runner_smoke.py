"""Smoke checks for the real-robot Gym-style runner logic without hardware."""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from gp8_control.config import Config
from gp8_control.planning.action_selector import ActionSelector
from gp8_control.rl.common import ACTION_PUSH, ACTION_THROW, HighLevelAction, SKILL_NAMES
from gp8_control.rl.real_runner import RealRlRunner
from gp8_control.skills.base import SkillResult
from gp8_control.tracking import TrackedObject


class _Skill:
    def __init__(self, name: str, allowed_classes: set[str] | None = None) -> None:
        self.name = name
        self._allowed_classes = allowed_classes

    def can_handle(self, target) -> bool:
        return self._allowed_classes is None or target.class_name in self._allowed_classes

    def placement_veto(self, target, intercept) -> str | None:
        del target, intercept
        return None

    def arrival_lead(self) -> float:
        return 0.1


class _PushSkill(_Skill):
    def _compute_retreat_poses(self, T_grasp, T_aim2, T_aim, aim_joint, grasp_joint):
        del T_aim, aim_joint
        T_retreat = T_grasp.copy()
        T_retreat[0, 3] -= 0.2
        return np.asarray(grasp_joint), np.asarray(grasp_joint), T_retreat

    def _compute_push_direction(self, T_grasp_retreat, T_aim2):
        d = T_aim2[:2, 3] - T_grasp_retreat[:2, 3]
        return d / np.linalg.norm(d)

    def _stroke_time_to(self, distance):
        return float(distance) * 2.0


class _Ctx:
    def __init__(self) -> None:
        self.skill_for = None
        self.skill_obj_for = None
        self.chain_target_override = None
        self.eval_calls = []

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
        del current_joint, belt_speed, now
        self.eval_calls.append((target.class_name, skill.name, float(pre_delay)))
        if target.class_name == "uncatchable":
            return None
        pose = target.T_grasp_base.copy()
        return SimpleNamespace(
            eta=1.0 + float(pre_delay),
            T_aim=pose.copy(),
            T_grasp=pose.copy(),
            aim_joint=np.zeros(6),
            grasp_joint=np.zeros(6),
        )


class _Planner:
    def compute_throw_params(self, T_grasp, T_aim, theta, *, target_distance):
        del T_grasp, T_aim, theta, target_distance
        return SimpleNamespace(T=2.0)


class _Queue:
    def __init__(self) -> None:
        self._objects = []

    def update(self, now, speed, *, belt_distance_m=None) -> None:
        del now, speed, belt_distance_m

    def remove(self, target) -> None:
        if target in self._objects:
            self._objects.remove(target)


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
        conf=1.0,
    )


def _runner() -> RealRlRunner:
    cfg = Config()
    cfg.TIME_STEP = 0.0
    ctx = _Ctx()
    skills = {
        "throw": _Skill("throw"),
        "push": _PushSkill("push", {"metal", "transparent"}),
    }
    app = SimpleNamespace(
        cfg=cfg,
        queue=_Queue(),
        traj_ctrl=SimpleNamespace(
            current_joints=[0.0] * 6,
            set_motion_op=lambda label: None,
        ),
        conveyor=SimpleNamespace(
            current=0.12,
            distance_m=0.0,
            check_freshness=lambda: None,
        ),
        ctx=ctx,
        robot=SimpleNamespace(forward_kinematics=lambda joints: np.eye(4)),
        planner=_Planner(),
        selector=ActionSelector(skills.values(), default="throw", by_class=cfg.SKILL_BY_CLASS),
        _active_target=None,
        _publish_belt_state=lambda: None,
        _ingest_detections=lambda now: None,
    )
    runner = RealRlRunner.__new__(RealRlRunner)
    runner.cfg = cfg
    runner.max_objects = 3
    runner.include_eta = True
    runner.app = app
    runner._running = True
    runner._chain_action = None
    runner._pending_action = None
    runner._log_file = None
    runner._install_rl_hooks()
    return runner


def main(argv=None) -> int:
    del argv
    runner = _runner()
    metal = _object("metal")
    transparent = _object("transparent")
    runner.app.queue._objects = [metal, transparent]

    checks = []
    mask0 = runner.action_mask()
    action0 = runner.app_heuristic_action_indices(mask0)
    checks.append(("heuristic selects metal push", action0.tolist() == [0, ACTION_PUSH]))
    reward, info0 = runner.step(runner.action_from_indices(*action0))
    del reward
    checks.append(("step 0 does not execute", not info0["executed"]))
    checks.append(("step 0 stores pending", runner.pending_action is not None))

    executed = []

    def _execute(action):
        executed.append(action)
        runner.app.queue._objects.remove(action.target)
        return SkillResult(True, "ok")

    runner._execute_action = _execute
    mask1 = runner.action_mask()
    checks.append(("pending target masked out", not bool(mask1[0, ACTION_PUSH])))
    action1 = runner.app_heuristic_action_indices(mask1)
    _, info1 = runner.step(runner.action_from_indices(*action1))
    checks.append(("step 1 executes previous pending", bool(info1["executed"])))
    checks.append(("executed metal push", executed[0].target is metal and executed[0].skill_name == "push"))
    checks.append(("current action becomes pending", runner.pending_action.target is transparent))

    stale = _object("metal")
    runner.app.queue._objects = []
    runner._pending_action = HighLevelAction(stale, "push")
    executed.clear()
    _, stale_info = runner.step(None)
    checks.append(("stale pending dropped", stale_info["dropped_stale_pending"]))
    checks.append(("stale pending not executed", not executed))

    runner = _runner()
    metal = _object("metal")
    transparent = _object("transparent")
    runner.app.queue._objects = [metal, transparent]
    runner._pending_action = HighLevelAction(metal, "throw")
    mask = runner.action_mask()
    pre_delays = [call[2] for call in runner.app.ctx.eval_calls if call[0] == "transparent"]
    checks.append(("pending delay applied", pre_delays and all(delay >= 3.0 for delay in pre_delays)))
    checks.append(("pending target excluded", not bool(mask[0, ACTION_THROW]) and not bool(mask[0, ACTION_PUSH])))

    runner._chain_action = HighLevelAction(transparent, "push")
    checks.append(("chain action overrides skill", runner._skill_name_for_object(transparent) == "push"))
    chain = runner._chain_target_override(np.zeros(6), 0.5)
    checks.append(("chain target override returns target", chain is not None and chain[1] is transparent))

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

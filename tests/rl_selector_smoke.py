"""Smoke checks for app-style heuristic routing in the RL simulator runner."""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from gp8_control.config import Config
from gp8_control.planning.action_selector import ActionSelector
from gp8_control.rl.sim_runner import ACTION_PUSH, ACTION_THROW, HighLevelAction, SimRlRunner
from gp8_control.skills.base import SkillResult
from gp8_control.tracking import TrackedObject


class _Skill:
    def __init__(self, name: str, allowed_classes: set[str] | None = None) -> None:
        self.name = name
        self._allowed_classes = allowed_classes

    def can_handle(self, target: TrackedObject) -> bool:
        return self._allowed_classes is None or target.class_name in self._allowed_classes


class _Core:
    def __init__(self, live_ids: set[int] | None = None, events: list[dict] | None = None) -> None:
        self.live_ids = set(live_ids or set())
        self.events = list(events or [])

    def is_object_live(self, sim_object_id: int | None) -> bool:
        return sim_object_id in self.live_ids

    def pop_reward_events(self) -> list[dict]:
        events = list(self.events)
        self.events.clear()
        return events


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


def _runner(cfg: Config | None = None) -> SimRlRunner:
    cfg = cfg or Config()
    runner = SimRlRunner.__new__(SimRlRunner)
    runner.cfg = cfg
    runner.max_objects = 3
    runner._chain_action = None
    runner._pending_action = None
    runner.skills = {
        "throw": _Skill("throw"),
        "push": _Skill("push", {"metal", "transparent"}),
    }
    runner.selector = ActionSelector(
        runner.skills.values(),
        default="throw",
        by_class=cfg.SKILL_BY_CLASS,
        force=cfg.FORCE_SKILL or None,
    )
    runner.queue = SimpleNamespace(_objects=[])
    runner.core = _Core()
    return runner


def main(argv=None) -> int:
    del argv
    checks: list[tuple[str, bool]] = []

    runner = _runner()
    metal = _object("metal")
    transparent = _object("transparent")
    paper = _object("paper")
    checks.extend(
        [
            ("metal routes to push", runner._skill_name_for_object(metal) == "push"),
            (
                "transparent routes to throw",
                runner._skill_name_for_object(transparent) == "throw",
            ),
            ("unknown routes to throw", runner._skill_name_for_object(paper) == "throw"),
        ]
    )

    runner.queue._objects = [metal, transparent, paper]
    mask = np.ones((runner.max_objects + 1, 2), dtype=np.uint8)
    checks.extend(
        [
            (
                "app heuristic selects first object's routed skill",
                runner.app_heuristic_action_indices(mask).tolist() == [0, ACTION_PUSH],
            ),
        ]
    )

    runner._chain_action = HighLevelAction(transparent, "push")
    checks.append(
        (
            "explicit chain action overrides heuristic",
            runner._skill_name_for_object(transparent) == "push",
        )
    )
    runner._chain_action = None
    runner._pending_action = HighLevelAction(paper, "push")
    checks.append(
        (
            "explicit pending action overrides heuristic",
            runner._skill_name_for_object(paper) == "push",
        )
    )
    runner._pending_action = None

    delay_calls = []
    eval_calls = []
    runner.current_joints = lambda: np.zeros(6)
    runner._pending_chain_delay = lambda now, current_joint: delay_calls.append(
        float(np.sum(current_joint)) + 2.5
    ) or 2.5

    def _evaluate(target, skill_name, current_joint, now, *, pre_delay=0.0):
        del target, current_joint, now
        eval_calls.append((skill_name, pre_delay))
        return True, 10.0 + pre_delay

    runner._evaluate_feasibility = _evaluate
    runner.action_mask()
    checks.append(
        (
            "action mask uses pending delay for every skill",
            delay_calls == [2.5]
            and eval_calls == [("throw", 2.5), ("push", 2.5)] * 3,
        )
    )
    eval_calls.clear()
    etas = runner._etas_for(metal, np.zeros(6), time.time())
    checks.append(
        (
            "eta preview uses same pending delay",
            etas == [12.5, 12.5]
            and eval_calls == [("throw", 2.5), ("push", 2.5)],
        )
    )

    stale_runner = _runner()
    stale_runner.cfg.TIME_STEP = 0.0
    stale_target = _object("metal")
    stale_target.sim_object_id = 10
    stale_runner.queue._objects = [stale_target]
    stale_runner.core = _Core(live_ids=set())
    stale_runner.ingest = lambda now=None: None
    executed = []
    stale_runner._execute_action = lambda action: executed.append(action) or SkillResult(True, "ok")
    stale_runner._pending_action = HighLevelAction(stale_target, "push")
    _, stale_info = stale_runner.step(None)
    checks.extend(
        [
            ("stale pending is dropped", bool(stale_info["dropped_stale_pending"])),
            ("stale pending does not execute", not executed),
        ]
    )

    live_runner = _runner()
    live_runner.cfg.TIME_STEP = 0.0
    live_target = _object("metal")
    live_target.sim_object_id = 11
    live_runner.queue._objects = [live_target]
    live_runner.core = _Core(live_ids={11})
    live_runner.ingest = lambda now=None: None
    executed.clear()
    live_runner._execute_action = lambda action: executed.append(action) or SkillResult(True, "ok")
    live_runner._pending_action = HighLevelAction(live_target, "push")
    _, live_info = live_runner.step(None)
    checks.extend(
        [
            ("live pending executes", bool(live_info["executed"]) and len(executed) == 1),
            ("live pending not marked stale", not live_info["dropped_stale_pending"]),
        ]
    )

    prune_runner = _runner()
    resolved = _object("transparent")
    resolved.sim_object_id = 21
    survivor = _object("metal")
    survivor.sim_object_id = 22
    prune_runner.queue._objects = [resolved, survivor]
    prune_runner._active_target = resolved
    prune_runner._pending_action = HighLevelAction(resolved, "throw")
    prune_runner._chain_action = HighLevelAction(resolved, "throw")
    prune_runner._prune_resolved_sim_objects([{"sim_object_id": 21, "reward": 1.0}])
    checks.extend(
        [
            ("resolved sim track pruned", prune_runner.queue._objects == [survivor]),
            ("resolved active target cleared", prune_runner._active_target is None),
            ("resolved pending cleared", prune_runner._pending_action is None),
            ("resolved chain cleared", prune_runner._chain_action is None),
        ]
    )

    cfg = Config()
    cfg.FORCE_SKILL = "throw"
    runner = _runner(cfg)
    checks.append(
        ("force throw overrides metal", runner._skill_name_for_object(_object("metal")) == "throw")
    )

    cfg = Config()
    cfg.FORCE_SKILL = "push"
    runner = _runner(cfg)
    checks.append(
        (
            "force push overrides transparent",
            runner._skill_name_for_object(_object("transparent")) == "push",
        )
    )

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

"""Real-robot Gym-style runner for high-level heuristic/RL actions."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from gp8_control.config import Config
from gp8_control.rl.common import (
    ACTION_THROW,
    SKILL_NAMES,
    HighLevelAction,
    action_summary,
    build_observation,
    skip_action,
)
from gp8_control.skills import PickRequest
from gp8_control.skills.base import SkillResult
from gp8_control.skills.push_skill import (
    PUSH_BIN_TARGET_MAP,
    PUSH_END_MAX_RADIUS,
    PUSH_FT_GAIN,
    PUSH_FT_MAX,
    PUSH_FT_MIN,
)

if TYPE_CHECKING:
    from gp8_control.app import GP8App


class RealRlRunner:
    """Hardware-backed counterpart to ``SimRlRunner``.

    It reuses ``GP8App.setup()`` for ROS/backend initialization, then runs its
    own one-step-ahead high-level control loop.
    """

    def __init__(
        self,
        cfg: Config | None = None,
        *,
        max_objects: int = 6,
        include_eta: bool = False,
        log_path: str | None = None,
        app: "GP8App | None" = None,
    ) -> None:
        self.cfg = cfg or getattr(app, "cfg", None) or Config()
        self.max_objects = int(max_objects)
        self.include_eta = bool(include_eta)
        if app is None:
            from gp8_control.app import GP8App

            app = GP8App(self.cfg)
        self.app = app
        self._running = False
        self._chain_action: HighLevelAction | None = None
        self._pending_action: HighLevelAction | None = None
        self._log_path = Path(log_path).expanduser() if log_path else None
        self._log_file = None
        if self._log_path is not None:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = self._log_path.open("a", encoding="utf-8")

    def start(self) -> None:
        self.app.setup()
        self._install_rl_hooks()
        self._running = True

    def close(self) -> None:
        self._running = False
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        try:
            if self.app.traj_ctrl is not None:
                self.app.traj_ctrl.exit_queue_mode()
        except Exception as exc:
            self._warn(f"exit_queue_mode failed: {exc}")
        try:
            if self.app.traj_ctrl is not None:
                self.app.traj_ctrl.close()
        except Exception as exc:
            self._warn(f"backend close failed: {exc}")
        try:
            if self.app.rl_shadow is not None:
                self.app.rl_shadow.close()
        except Exception as exc:
            self._warn(f"RL shadow close failed: {exc}")
        try:
            if self.app._executor is not None:
                self.app._executor.shutdown()
        except Exception:
            pass
        try:
            if self.app._node is not None:
                self.app._node.destroy_node()
        except Exception:
            pass
        try:
            import rclpy

            rclpy.shutdown()
        except Exception:
            pass

    def is_ok(self) -> bool:
        if not self._running or getattr(self.app, "_spinner_dead", False):
            return False
        try:
            import rclpy

            return bool(rclpy.ok())
        except Exception:
            return True

    def _warn(self, msg: str) -> None:
        node = getattr(self.app, "_node", None)
        if node is not None:
            try:
                node.get_logger().warn(msg)
                return
            except Exception:
                pass
        print(f"[real-rl][warn] {msg}", flush=True)

    @property
    def pending_action(self) -> HighLevelAction | None:
        return self._pending_action

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

    def ingest(self, now: float | None = None) -> None:
        now = time.time() if now is None else float(now)
        self.app.conveyor.check_freshness()
        self.app._publish_belt_state()
        self.app._ingest_detections(now)
        self.app.queue.update(
            now,
            self.app.conveyor.current,
            belt_distance_m=self.app.conveyor.distance_m,
        )

    def action_from_indices(self, object_slot: int, skill_index: int) -> HighLevelAction | None:
        if int(object_slot) >= self.max_objects:
            return None
        objects = self.ordered_objects()
        if int(object_slot) < 0 or int(object_slot) >= len(objects):
            return None
        return HighLevelAction(objects[int(object_slot)], SKILL_NAMES[int(skill_index)])

    def step(self, action: HighLevelAction | None) -> tuple[float, dict]:
        now = time.time()
        self.ingest(now)
        pending = self._pending_action
        self._chain_action = action
        info: dict[str, Any] = {
            "executed": False,
            "execution_success": False,
            "execution_detail": "",
            "dropped_stale_pending": False,
            "reward_events": [],
        }
        try:
            if pending is not None and pending.target is not None and not self._action_is_live(pending):
                info["dropped_stale_pending"] = True
                self._pending_action = None
                time.sleep(self.cfg.TIME_STEP)
            elif pending is not None and pending.target is not None:
                result = self._execute_action(pending)
                info.update(
                    executed=True,
                    execution_success=bool(result.success),
                    execution_detail=result.detail,
                    executed_skill=pending.skill_name,
                    executed_track_id=getattr(pending.target, "track_id", None),
                    executed_class=getattr(pending.target, "class_name", None),
                )
            else:
                time.sleep(self.cfg.TIME_STEP)
        finally:
            self._chain_action = None

        self.ingest()
        self._pending_action = action if self._action_is_live(action) else None
        reward = 0.0
        return reward, info

    def action_mask(self) -> np.ndarray:
        now = time.time()
        current_joint = self.current_joints()
        mask = np.zeros((self.max_objects + 1, len(SKILL_NAMES)), dtype=np.uint8)
        mask[self.max_objects, :] = 1
        if current_joint is None:
            return mask
        pre_delay = self._pending_chain_delay(now, current_joint)
        for slot, target in enumerate(self.ordered_objects()):
            if not self._target_is_live(target):
                continue
            if (
                self._pending_action is not None
                and target is self._pending_action.target
            ):
                continue
            for skill_index, skill_name in enumerate(SKILL_NAMES):
                feasible, _ = self._evaluate_feasibility(
                    target,
                    skill_name,
                    current_joint,
                    now,
                    pre_delay=pre_delay,
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

    def observation(self, include_eta: bool | None = None) -> np.ndarray:
        include_eta = self.include_eta if include_eta is None else bool(include_eta)
        now = time.time()
        current_joint = self.current_joints()
        return build_observation(
            objects=self.ordered_objects(),
            max_objects=self.max_objects,
            include_eta=include_eta,
            joints=current_joint,
            ee_xyz=self.ee_xyz(current_joint),
            pending_indices=self._pending_indices(),
            belt_speed=self.app.conveyor.current,
            y_now_for=lambda target: self.app.ctx.object_y_now(
                target, now, self.app.conveyor.current
            ),
            etas_for=lambda target: self._etas_for(target, current_joint, now),
        )

    def write_step_log(
        self,
        *,
        step_index: int,
        observation: np.ndarray,
        action_mask: np.ndarray,
        selected_action,
        info: dict,
    ) -> None:
        if self._log_file is None:
            return
        record = {
            "ts": time.time(),
            "step": int(step_index),
            "observation": observation.tolist(),
            "action_mask": action_mask.tolist(),
            "selected_action": action_summary(selected_action),
            "pending_action": self._pending_action_summary(),
            "info": info,
        }
        self._log_file.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._log_file.flush()

    def _install_rl_hooks(self) -> None:
        self.app.ctx.skill_for = lambda obj: self._skill_name_for_object(obj)
        self.app.ctx.skill_obj_for = lambda obj: self._skill_for_object(obj)
        self.app.ctx.chain_target_override = self._chain_target_override

    def _object_in_queue(self, target) -> bool:
        return any(obj is target for obj in self.app.queue._objects)

    def _target_is_live(self, target) -> bool:
        return target is not None and self._object_in_queue(target)

    def _action_is_live(self, action: HighLevelAction | None) -> bool:
        return (
            action is not None
            and action.target is not None
            and self._target_is_live(action.target)
        )

    def _execute_action(self, action: HighLevelAction) -> SkillResult:
        target = action.target
        if target is None:
            raise RuntimeError("cannot execute empty action")
        current_joint = self.current_joints()
        if current_joint is None:
            raise RuntimeError("robot joints unavailable")
        skill = self.app.selector.skills[action.skill_name]
        intercept = self.app.ctx.earliest_reachable_intercept(
            target,
            current_joint,
            self.app.conveyor.current,
            time.time(),
            skill=skill,
        )
        if intercept is None:
            self.app.queue.remove(target)
            return SkillResult(False, "intercept infeasible")
        veto = skill.placement_veto(target, intercept)
        if veto is not None:
            return SkillResult(False, veto)
        self.app.queue.remove(target)
        self.app._active_target = target
        request = PickRequest(
            target=target,
            current_joint=current_joint,
            T_aim=intercept.T_aim,
            T_grasp=intercept.T_grasp,
            aim_joint=intercept.aim_joint,
            grasp_joint=intercept.grasp_joint,
            secondary=self._chain_action.target if self._chain_action is not None else None,
        )
        self.app.traj_ctrl.set_motion_op(action.skill_name)
        return skill.execute(request)

    def _evaluate_feasibility(
        self,
        target,
        skill_name: str,
        current_joint: np.ndarray,
        now: float,
        *,
        pre_delay: float = 0.0,
    ) -> tuple[bool, float]:
        skill = self.app.selector.skills[skill_name]
        intercept = self.app.ctx.earliest_reachable_intercept(
            target,
            current_joint,
            self.app.conveyor.current,
            now,
            pre_delay=pre_delay,
            skill=skill,
        )
        if intercept is None:
            return False, -1.0
        if skill.placement_veto(target, intercept) is not None:
            return False, -1.0
        return True, float(intercept.eta)

    def _pending_chain_delay(
        self,
        now: float,
        current_joint: np.ndarray,
    ) -> float:
        action = self._pending_action
        if action is None or action.target is None:
            return 0.0
        if not self._action_is_live(action):
            return 0.0
        try:
            skill = self.app.selector.skills[action.skill_name]
            intercept = self.app.ctx.earliest_reachable_intercept(
                action.target,
                np.asarray(current_joint, dtype=float),
                self.app.conveyor.current,
                now,
                skill=skill,
            )
            if intercept is None:
                return 0.0
            if skill.placement_veto(action.target, intercept) is not None:
                return 0.0
            if action.skill_name == "throw":
                return self._estimate_throw_chain_delay(action.target, intercept)
            if action.skill_name == "push":
                return self._estimate_push_chain_delay(action.target, intercept, now)
        except Exception as exc:
            self._warn(f"pending chain delay preview failed; using immediate mask: {exc}")
        return 0.0

    def _estimate_throw_chain_delay(self, target, intercept) -> float:
        goal_x = float(self.cfg.THROW_GOAL_X)
        goal_y = float(self.cfg.THROW_GOAL_Y)
        theta = float(math.atan2(goal_y, goal_x))
        model_distance = float(math.hypot(goal_x, goal_y))
        params = self.app.planner.compute_throw_params(
            intercept.T_grasp,
            intercept.T_aim,
            theta,
            target_distance=model_distance,
        )
        return max(0.0, float(intercept.eta)) + max(0.0, float(params.T))

    def _estimate_push_chain_delay(self, target, intercept, now: float) -> float:
        skill = self.app.selector.skills["push"]
        bin_xyz = PUSH_BIN_TARGET_MAP.get(target.class_name)
        if bin_xyz is None:
            return max(0.0, float(intercept.eta)) + max(0.0, float(skill.arrival_lead()))

        T_aim2 = np.eye(4)
        T_aim2[:3, :3] = intercept.T_aim[:3, :3]
        T_aim2[:3, 3] = np.asarray(bin_xyz, dtype=float)
        _wait_joint, grasp_retreat_joint, T_grasp_retreat = skill._compute_retreat_poses(
            intercept.T_grasp,
            T_aim2,
            intercept.T_aim,
            intercept.aim_joint,
            intercept.grasp_joint,
        )
        del _wait_joint, grasp_retreat_joint
        contact_offset = float(
            np.hypot(*(intercept.T_grasp[:2, 3] - T_grasp_retreat[:2, 3]))
        )
        push_dir_exec = skill._compute_push_direction(T_grasp_retreat, T_aim2)
        dist_bin = float(np.hypot(*(T_aim2[:2, 3] - intercept.T_grasp[:2, 3])))
        follow_through = float(np.clip(PUSH_FT_GAIN * dist_bin, PUSH_FT_MIN, PUSH_FT_MAX))
        push_distance = contact_offset + follow_through
        rx, ry = float(T_grasp_retreat[0, 3]), float(T_grasp_retreat[1, 3])
        b = rx * float(push_dir_exec[0]) + ry * float(push_dir_exec[1])
        c = rx * rx + ry * ry - PUSH_END_MAX_RADIUS ** 2
        disc = b * b - c
        if disc >= 0.0:
            s_max = -b + float(np.sqrt(disc))
            if push_distance > s_max:
                push_distance = max(s_max, contact_offset + 0.05)

        v_belt = max(float(self.app.conveyor.current), 1e-6)
        y_now = self.app.ctx.object_y_now(target, now, self.app.conveyor.current)
        wait_est = max(0.0, (y_now - float(intercept.T_grasp[1, 3])) / v_belt)
        push_time = skill._stroke_time_to(push_distance)
        return max(0.0, wait_est) + max(0.0, float(push_time))

    def _chain_target_override(self, from_joint: np.ndarray, action_time: float):
        action = self._chain_action
        if action is None or action.target is None:
            return None
        if not self._action_is_live(action):
            return None
        feasible, _ = self._evaluate_feasibility(
            action.target,
            action.skill_name,
            np.asarray(from_joint, dtype=float),
            time.time(),
            pre_delay=float(action_time),
        )
        if not feasible:
            return None
        skill = self.app.selector.skills[action.skill_name]
        intercept = self.app.ctx.earliest_reachable_intercept(
            action.target,
            np.asarray(from_joint, dtype=float),
            self.app.conveyor.current,
            time.time(),
            pre_delay=float(action_time),
            skill=skill,
        )
        return None if intercept is None else (intercept.grasp_joint, action.target)

    def _skill_name_for_object(self, obj) -> str:
        if self._chain_action is not None and self._chain_action.target is obj:
            return self._chain_action.skill_name
        if self._pending_action is not None and self._pending_action.target is obj:
            return self._pending_action.skill_name
        return self.app.selector.skill_for(obj)

    def _skill_for_object(self, obj):
        return self.app.selector.skills[self._skill_name_for_object(obj)]

    def _etas_for(self, target, current_joint: np.ndarray | None, now: float) -> list[float]:
        if current_joint is None:
            return [-1.0, -1.0]
        pre_delay = self._pending_chain_delay(now, current_joint)
        return [
            self._evaluate_feasibility(
                target, "throw", current_joint, now, pre_delay=pre_delay
            )[1],
            self._evaluate_feasibility(
                target, "push", current_joint, now, pre_delay=pre_delay
            )[1],
        ]

    def _pending_indices(self) -> tuple[int, int]:
        if self._pending_action is None or self._pending_action.target is None:
            return self.max_objects, ACTION_THROW
        objects = self.ordered_objects()
        try:
            slot = objects.index(self._pending_action.target)
        except ValueError:
            slot = self.max_objects
        return slot, SKILL_NAMES.index(self._pending_action.skill_name)

    def _pending_action_summary(self) -> dict:
        slot, skill = self._pending_indices()
        return {
            "object_slot": slot,
            "skill_index": skill,
            "skill_name": SKILL_NAMES[skill] if 0 <= skill < len(SKILL_NAMES) else None,
        }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="gp8_real_rl")
    parser.add_argument("--policy", choices=("app",), default="app")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--max-objects", type=int, default=6)
    parser.add_argument("--include-eta", action="store_true")
    parser.add_argument("--log", default="")
    args = parser.parse_args(argv)

    runner = RealRlRunner(
        max_objects=args.max_objects,
        include_eta=args.include_eta,
        log_path=args.log or None,
    )
    try:
        runner.start()
        step_idx = 0
        while runner.is_ok() and (args.steps <= 0 or step_idx < args.steps):
            obs = runner.observation()
            mask = runner.action_mask()
            if args.policy == "app":
                action = runner.app_heuristic_action_indices(mask)
            else:
                action = skip_action(runner.max_objects)
            valid = bool(mask[int(action[0]), int(action[1])])
            high_level_action = (
                runner.action_from_indices(int(action[0]), int(action[1]))
                if valid and int(action[0]) < runner.max_objects
                else None
            )
            reward, info = runner.step(high_level_action)
            del reward
            info["action_valid"] = valid
            runner.write_step_log(
                step_index=step_idx,
                observation=obs,
                action_mask=mask,
                selected_action=action,
                info=info,
            )
            print(
                f"step={step_idx} action={action.tolist()} "
                f"skill={SKILL_NAMES[int(action[1])]} valid={valid} "
                f"pending={runner._pending_indices()} executed={info.get('executed')}",
                flush=True,
            )
            step_idx += 1
    finally:
        runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Non-ROS real-time simulator runner for high-level RL actions."""

from __future__ import annotations

import math
import time

import numpy as np

from gp8_control.backends.mujoco_sim import (
    MujocoRobotBackend,
    MujocoWorldSource,
    SimConfig,
    SimCore,
    bins_from_config,
    class_bin_name_for,
)
from gp8_control.config import Config
from gp8_control.perception.detection_intake import DetectionIntake
from gp8_control.planning import PickThrowPlanner
from gp8_control.planning.action_selector import ActionSelector
from gp8_control.robots.gp8 import GP8
from gp8_control.rl.common import (
    ACTION_PUSH,
    ACTION_THROW,
    SKILL_NAMES,
    HighLevelAction,
    build_observation,
    bbox_size,
    class_id,
    skip_action,
)
from gp8_control.skills.base import SkillResult
from gp8_control.skills import PickRequest, PushSkill, SkillContext, ThrowSkill
from gp8_control.skills.push_skill import (
    PUSH_BIN_TARGET_MAP,
    PUSH_END_MAX_RADIUS,
    PUSH_FT_GAIN,
    PUSH_FT_MAX,
    PUSH_FT_MIN,
)
from gp8_control.tracking import FrameGate, TrackedObject, TrackedObjectQueue
from gp8_control.trajectory.predictor import TrajectoryPredictor
from gp8_control.trajectory.trajectory_primitive import trajectory


class _Logger:
    def info(self, msg: str) -> None:
        print(f"[rl-sim] {msg}", flush=True)

    def warn(self, msg: str) -> None:
        print(f"[rl-sim][warn] {msg}", flush=True)

    warning = warn

    def error(self, msg: str) -> None:
        print(f"[rl-sim][error] {msg}", flush=True)


class _NodeShim:
    def __init__(self) -> None:
        self._logger = _Logger()

    def get_logger(self) -> _Logger:
        return self._logger


def _make_transform(rotation: np.ndarray, translation) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(translation, dtype=float).ravel()
    return transform


class SimRlRunner:
    """App-compatible simulator core used underneath the Gym wrapper.

    The runner keeps full control objects internally and exposes only projections
    to the RL layer. It intentionally does not import or initialize ROS.
    """

    def __init__(
        self,
        cfg: Config | None = None,
        sim_cfg: SimConfig | None = None,
        *,
        max_objects: int = 6,
        include_eta: bool = False,
    ) -> None:
        self.cfg = cfg or Config()
        self.cfg.BACKEND = "mujoco"
        self.max_objects = int(max_objects)
        self.include_eta = bool(include_eta)
        self.node = _NodeShim()
        self.robot = GP8()
        self.core = SimCore(
            sim_cfg
            or SimConfig(
                grasp_z=self.cfg.GRASP_Z,
                bins=bins_from_config(self.cfg),
            )
        )
        self.traj_ctrl = MujocoRobotBackend(self.core)
        self.world = MujocoWorldSource(self.core)
        self.conveyor = self.world.belt
        self.queue = TrackedObjectQueue(
            self.cfg.MAX_REACH,
            drop_below_y=-self.cfg.MAX_REACH,
        )
        self.frame_gate = FrameGate(self.cfg.FRAME_COOLDOWN_DISTANCE)
        self.detection_intake = DetectionIntake(
            self.cfg.OBJECT_MATCH_EPSILON,
            drift_frac=self.cfg.OBJECT_MATCH_DRIFT_FRAC,
            eps_y_max=self.cfg.OBJECT_MATCH_EPS_Y_MAX,
            merge_eps_y_max=self.cfg.OBJECT_MERGE_EPS_Y_MAX,
            assoc=self.cfg.TRACK_ASSOC,
        )
        self.M1 = np.asarray(self.robot.velocity_limits, dtype=float) * self.cfg.JOINT_VEL_LIMIT_SCALE
        self.M2 = self.M1 * self.cfg.JOINT_ACCEL_LIMIT_SCALE
        self.predictor = TrajectoryPredictor()
        self.planner = PickThrowPlanner(
            robot=self.robot,
            predictor=self.predictor,
            M1=self.M1,
            M2=self.M2,
            max_reach=self.cfg.MAX_REACH,
            target_distance=float(np.hypot(self.cfg.THROW_GOAL_X, self.cfg.THROW_GOAL_Y)),
            decoding=self.cfg.throw_decoding(),
            max_pick_lead=self.cfg.MAX_PICK_LEAD,
        )
        self._running = True
        self._active_target: TrackedObject | None = None
        self._status = "IDLE"
        self._status_detail = ""
        self._chain_action: HighLevelAction | None = None
        self._episode_start = time.time()
        self._pending_action: HighLevelAction | None = None
        self._build_skills()

    def _build_skills(self) -> None:
        idle_transform = _make_transform(self.cfg.INITIAL_R, self.cfg.INITIAL_T)
        idle_joint = self.robot.inverse_kinematics(idle_transform)
        if idle_joint is None:
            raise RuntimeError("IK failed for idle/initial pose.")
        idle_joint = np.asarray(idle_joint, dtype=float)
        idle_joint[-1] = self.cfg.PICK_WRIST_J6

        self.ctx = SkillContext(
            cfg=self.cfg,
            node=self.node,
            robot=self.robot,
            traj_ctrl=self.traj_ctrl,
            planner=self.planner,
            conveyor=self.conveyor,
            queue=self.queue,
            M1=self.M1,
            M2=self.M2,
            intake=self.ingest,
            publish_state=lambda: None,
            set_status=self._set_status,
            set_active_target=self._set_active_target,
            skill_for=lambda obj: self._skill_name_for_object(obj),
            skill_obj_for=lambda obj: self._skill_for_object(obj),
            idle_joint=idle_joint,
            ok=lambda: self._running,
            chain_target_override=self._chain_target_override,
        )
        self.skills = {
            "throw": ThrowSkill(self.ctx),
            "push": PushSkill(self.ctx),
        }
        self.selector = ActionSelector(
            self.skills.values(),
            default="throw",
            by_class=self.cfg.SKILL_BY_CLASS,
            force=self.cfg.FORCE_SKILL or None,
        )

    def start(self, timeout_sec: float = 10.0) -> None:
        if not self.traj_ctrl.wait_for_servers(timeout_sec):
            raise RuntimeError("MuJoCo backend did not become ready.")
        self.traj_ctrl.suction_off()
        self._move_to_initial_pose()
        self.core.pop_reward_events()

    def close(self) -> None:
        self._running = False
        self.traj_ctrl.close()

    @property
    def pending_action(self) -> HighLevelAction | None:
        return self._pending_action

    def ordered_objects(self) -> list[TrackedObject]:
        return list(self.queue._objects)[: self.max_objects]

    def _object_in_queue(self, target: TrackedObject) -> bool:
        return any(obj is target for obj in self.queue._objects)

    def _target_is_live(self, target: TrackedObject | None) -> bool:
        if target is None or not self._object_in_queue(target):
            return False
        sim_object_id = getattr(target, "sim_object_id", None)
        if sim_object_id is None:
            return True
        return self.core.is_object_live(sim_object_id)

    def _action_is_live(self, action: HighLevelAction | None) -> bool:
        return (
            action is not None
            and action.target is not None
            and self._target_is_live(action.target)
        )

    def ingest(self, now: float | None = None) -> None:
        now = time.time() if now is None else float(now)
        snapshot = self.world.latest_snapshot()
        receipt_time = float(snapshot.get("receipt_time", now)) if snapshot else now
        belt_distance = self.conveyor.distance_at(receipt_time)
        added = self.detection_intake.ingest(
            snapshot,
            self.queue,
            self._active_target,
            self.conveyor.current,
            self.node.get_logger(),
            belt_distance_m=belt_distance,
        )
        if added:
            self.frame_gate.mark(now)
        self.queue.update(
            now,
            self.conveyor.current,
            belt_distance_m=self.conveyor.distance_m,
        )

    def action_from_indices(self, object_slot: int, skill_index: int) -> HighLevelAction | None:
        if int(object_slot) >= self.max_objects:
            return None
        objects = self.ordered_objects()
        if int(object_slot) < 0 or int(object_slot) >= len(objects):
            return None
        skill_name = SKILL_NAMES[int(skill_index)]
        return HighLevelAction(objects[int(object_slot)], skill_name)

    def step(self, action: HighLevelAction | None) -> tuple[float, dict]:
        now = time.time()
        self.ingest(now)
        pending = self._pending_action
        self._chain_action = action
        reward = 0.0
        info: dict = {
            "executed": False,
            "execution_success": False,
            "execution_detail": "",
            "reward_events": [],
            "dropped_stale_pending": False,
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
                )
                if not result.success:
                    reward -= 0.3
            else:
                time.sleep(self.cfg.TIME_STEP)
        finally:
            self._chain_action = None

        reward_events = self._consume_reward_events()
        self._prune_resolved_sim_objects(reward_events)
        reward += sum(float(event.get("reward", 0.0)) for event in reward_events)
        info["reward_events"] = reward_events
        self.ingest()
        self._pending_action = action if self._action_is_live(action) else None
        return float(reward), info

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
        """Choose the next action using the same class routing as ``app.py``.

        This is a temporary non-learning policy for rollouts. It keeps the RL
        mask broader than the app heuristic, but chooses app-style actions:
        ``SKILL_BY_CLASS`` first, default throw, and ``can_handle`` fallback.
        """
        if mask is None:
            mask = self.action_mask()
        objects = self.ordered_objects()
        for slot, target in enumerate(objects):
            skill_name = self.selector.skill_for(target)
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
            belt_speed=self.conveyor.current,
            y_now_for=lambda target: self.ctx.object_y_now(
                target, now, self.conveyor.current
            ),
            etas_for=lambda target: self._etas_for(target, current_joint, now),
        )

    def current_joints(self) -> np.ndarray | None:
        if self.traj_ctrl.current_joints is None:
            return None
        return np.asarray(self.traj_ctrl.current_joints, dtype=float)

    def ee_xyz(self, joints: np.ndarray | None = None) -> np.ndarray:
        if joints is None:
            joints = self.current_joints()
        if joints is None:
            return np.zeros(3, dtype=np.float32)
        transform = self.robot.forward_kinematics(np.asarray(joints, dtype=float)[:6])
        return np.asarray(transform[:3, 3], dtype=np.float32)

    def _execute_action(self, action: HighLevelAction):
        target = action.target
        if target is None:
            raise RuntimeError("cannot execute empty action")
        current_joint = self.current_joints()
        if current_joint is None:
            raise RuntimeError("robot joints unavailable")
        skill = self.skills[action.skill_name]
        intercept = self.ctx.earliest_reachable_intercept(
            target,
            current_joint,
            self.conveyor.current,
            time.time(),
            skill=skill,
        )
        if intercept is None:
            self.queue.remove(target)
            return SkillResult(False, "intercept infeasible")
        veto = skill.placement_veto(target, intercept)
        if veto is not None:
            return SkillResult(False, veto)
        self.queue.remove(target)
        self._active_target = target
        self.core.mark_manipulated(
            getattr(target, "sim_object_id", None),
            action.skill_name,
            class_bin_name_for(target.class_name),
        )
        request = PickRequest(
            target=target,
            current_joint=current_joint,
            T_aim=intercept.T_aim,
            T_grasp=intercept.T_grasp,
            aim_joint=intercept.aim_joint,
            grasp_joint=intercept.grasp_joint,
            secondary=self._chain_action.target if self._chain_action is not None else None,
        )
        self.traj_ctrl.set_motion_op(action.skill_name)
        return skill.execute(request)

    def _evaluate_feasibility(
        self,
        target: TrackedObject,
        skill_name: str,
        current_joint: np.ndarray,
        now: float,
        *,
        pre_delay: float = 0.0,
    ) -> tuple[bool, float]:
        skill = self.skills[skill_name]
        intercept = self.ctx.earliest_reachable_intercept(
            target,
            current_joint,
            self.conveyor.current,
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
        """Estimated delay before a selected next action can be reached.

        ``action_mask`` is queried before the pending action executes, but a
        selected action is consumed by the skill's post-action chain. This helper
        projects the candidate mask to that chain horizon. The skill execution
        path still performs the final authoritative feasibility check.
        """
        action = self._pending_action
        if action is None or action.target is None:
            return 0.0
        if not self._action_is_live(action):
            return 0.0
        try:
            skill = self.skills[action.skill_name]
            intercept = self.ctx.earliest_reachable_intercept(
                action.target,
                np.asarray(current_joint, dtype=float),
                self.conveyor.current,
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
            self.node.get_logger().warn(
                f"pending chain delay preview failed; using immediate mask: {exc}"
            )
        return 0.0

    def _estimate_throw_chain_delay(self, target: TrackedObject, intercept) -> float:
        goal_x = float(self.cfg.THROW_GOAL_X)
        goal_y = float(self.cfg.THROW_GOAL_Y)
        theta = float(math.atan2(goal_y, goal_x))
        model_distance = float(math.hypot(goal_x, goal_y))
        params = self.planner.compute_throw_params(
            intercept.T_grasp,
            intercept.T_aim,
            theta,
            target_distance=model_distance,
        )
        return max(0.0, float(intercept.eta)) + max(0.0, float(params.T))

    def _estimate_push_chain_delay(
        self,
        target: TrackedObject,
        intercept,
        now: float,
    ) -> float:
        skill = self.skills["push"]
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

        v_belt = max(float(self.conveyor.current), 1e-6)
        y_now = self.ctx.object_y_now(target, now, self.conveyor.current)
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
        skill = self.skills[action.skill_name]
        intercept = self.ctx.earliest_reachable_intercept(
            action.target,
            np.asarray(from_joint, dtype=float),
            self.conveyor.current,
            time.time(),
            pre_delay=float(action_time),
            skill=skill,
        )
        return None if intercept is None else (intercept.grasp_joint, action.target)

    def _skill_name_for_object(self, obj: TrackedObject) -> str:
        if self._chain_action is not None and self._chain_action.target is obj:
            return self._chain_action.skill_name
        if self._pending_action is not None and self._pending_action.target is obj:
            return self._pending_action.skill_name
        return self.selector.skill_for(obj)

    def _skill_for_object(self, obj: TrackedObject):
        return self.skills[self._skill_name_for_object(obj)]

    def _move_to_initial_pose(self) -> None:
        deadline = time.time() + 10.0
        while self.current_joints() is None and time.time() < deadline:
            time.sleep(0.05)
        current = self.current_joints()
        if current is None:
            raise RuntimeError("robot joints unavailable after sim start")
        initial_joint = self.robot.inverse_kinematics(
            _make_transform(self.cfg.INITIAL_R, self.cfg.INITIAL_T)
        )
        if initial_joint is None:
            raise RuntimeError("IK failed for initial pose.")
        initial_joint = np.asarray(initial_joint, dtype=float)
        initial_joint[-1] = self.cfg.PICK_WRIST_J6
        traj, vel, ts = trajectory(
            current,
            np.zeros_like(self.M1),
            initial_joint,
            np.zeros_like(self.M1),
            self.M1,
            self.M2,
            hertz=self.cfg.TRAJ_HZ,
        )
        self.traj_ctrl.send_trajectory_queue(traj, vel, ts, final_joint=initial_joint)

    def _set_status(self, status: str, detail: str = "") -> None:
        self._status = str(status)
        self._status_detail = str(detail)

    def _set_active_target(self, target: TrackedObject | None) -> None:
        self._active_target = target

    def _consume_reward_events(self) -> list[dict]:
        return self.core.pop_reward_events()

    def _prune_resolved_sim_objects(self, reward_events: list[dict]) -> None:
        resolved_ids = {
            int(event["sim_object_id"])
            for event in reward_events
            if event.get("sim_object_id") is not None
        }
        if not resolved_ids:
            return
        self.queue._objects = [
            obj for obj in self.queue._objects
            if getattr(obj, "sim_object_id", None) not in resolved_ids
        ]
        if (
            self._active_target is not None
            and getattr(self._active_target, "sim_object_id", None) in resolved_ids
        ):
            self._active_target = None
        if (
            self._pending_action is not None
            and self._pending_action.target is not None
            and getattr(self._pending_action.target, "sim_object_id", None) in resolved_ids
        ):
            self._pending_action = None
        if (
            self._chain_action is not None
            and self._chain_action.target is not None
            and getattr(self._chain_action.target, "sim_object_id", None) in resolved_ids
        ):
            self._chain_action = None

    def _pending_indices(self) -> tuple[int, int]:
        if self._pending_action is None or self._pending_action.target is None:
            return self.max_objects, ACTION_THROW
        objects = self.ordered_objects()
        try:
            slot = objects.index(self._pending_action.target)
        except ValueError:
            slot = self.max_objects
        skill = SKILL_NAMES.index(self._pending_action.skill_name)
        return slot, skill

    def _etas_for(
        self,
        target: TrackedObject,
        current_joint: np.ndarray | None,
        now: float,
    ) -> list[float]:
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

    def _bbox_size(self, target: TrackedObject, now: float) -> tuple[float, float]:
        del now
        return bbox_size(target)

    @staticmethod
    def _class_id(class_name: str) -> int:
        return class_id(class_name)

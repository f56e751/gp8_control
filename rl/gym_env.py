"""Gymnasium wrapper for the GP8 MuJoCo RL runner."""

from __future__ import annotations

import gc
import os
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    try:
        import gym
        from gym import spaces
    except ImportError:
        gym = None
        spaces = None

from gp8_control.config import Config
from gp8_control.rl.common import (
    ACTION_THROW,
    SKILL_NAMES,
    normalize_bbox_observation,
    observation_width,
)
from gp8_control.rl.sim_runner import SimRlRunner


class GP8RecyclingEnv(gym.Env if gym is not None else object):
    """High-level RL env using one-step-ahead actions."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        max_objects: int = 6,
        include_eta: bool = False,
        max_steps: int = 200,
        max_episode_seconds: float | None = 240.0,
        cfg: Config | None = None,
        realtime: bool | None = None,
        bbox_observation: str | None = None,
        include_suction_p: bool | None = None,
    ) -> None:
        if spaces is None:
            raise ImportError("GP8RecyclingEnv requires gymnasium or gym.")
        self.max_objects = int(max_objects)
        self.include_eta = bool(include_eta)
        self.max_steps = int(max_steps)
        self.max_episode_seconds = (
            None if max_episode_seconds is None else float(max_episode_seconds)
        )
        self.cfg = cfg
        self.realtime = realtime
        self.bbox_observation = normalize_bbox_observation(
            bbox_observation or os.environ.get("GP8_RL_BBOX_OBSERVATION", "size")
        )
        if include_suction_p is None:
            include_suction_p = os.environ.get(
                "GP8_RL_INCLUDE_SUCTION_P", "0"
            ).strip().lower() in ("1", "true", "yes", "on")
        self.include_suction_p = bool(include_suction_p)
        self._runner: SimRlRunner | None = None
        self._step_count = 0
        self._episode_start_time = 0.0
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                observation_width(
                    self.max_objects,
                    self.include_eta,
                    self.bbox_observation,
                    self.include_suction_p,
                ),
            ),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete([self.max_objects + 1, len(SKILL_NAMES)])

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        if gym is not None:
            super_reset = getattr(super(), "reset", None)
            if callable(super_reset):
                super_reset(seed=seed)
        if self._runner is not None:
            self._runner.close()
            self._runner = None
            # The old sim sits in reference cycles; collect now so its MuJoCo
            # buffers don't pile up across episodes.
            gc.collect()
        self._runner = SimRlRunner(
            cfg=self.cfg,
            max_objects=self.max_objects,
            include_eta=self.include_eta,
            bbox_observation=self.bbox_observation,
            include_suction_p=self.include_suction_p,
            sim_seed=seed,
            realtime=self.realtime,
        )
        self._runner.start()
        self._step_count = 0
        self._wait_for_first_objects(float((options or {}).get("startup_timeout", 8.0)))
        self._episode_start_time = self._runner.clock.time()
        obs = self._runner.observation(
            remaining_time_frac=self._remaining_time_frac()
        )
        info = self._info(selected_action=None, selected_skill=None)
        return obs, info

    def step(self, action):
        if self._runner is None:
            raise RuntimeError("reset() must be called before step().")
        object_slot, skill_index = self._parse_action(action)
        mask = self._runner.action_mask()
        valid = bool(mask[object_slot, skill_index])
        next_action = (
            self._runner.action_from_indices(object_slot, skill_index)
            if valid and object_slot < self.max_objects
            else None
        )
        deadline = self._episode_deadline()
        reward, exec_info = self._runner.step(next_action)
        self._step_count += 1
        episode_sim_time = self._episode_sim_time()
        step_limit_hit = self.max_steps > 0 and self._step_count >= self.max_steps
        time_limit_hit = (
            self.max_episode_seconds is not None
            and episode_sim_time >= self.max_episode_seconds
        )
        # Remaining time is observed, so the time budget is a true terminal
        # (no bootstrap); the unobserved step cap stays a truncation.
        terminated = bool(time_limit_hit)
        truncated = bool(step_limit_hit and not time_limit_hit)
        obs = self._runner.observation(
            remaining_time_frac=self._remaining_time_frac()
        )
        info = self._info(selected_action=object_slot, selected_skill=skill_index)
        info.update(exec_info)
        reward, late_reward = self._drop_late_reward_events(
            float(reward),
            info.get("reward_events", []),
            deadline,
        )
        info["action_valid"] = valid
        info["truncated_by_steps"] = bool(step_limit_hit)
        info["truncated_by_time"] = bool(time_limit_hit)
        info["late_reward_dropped"] = float(late_reward)
        if not valid:
            info["invalid_action_treated_as"] = "skip"
        return obs, float(reward), terminated, truncated, info

    def close(self) -> None:
        if self._runner is not None:
            self._runner.close()
            self._runner = None

    def _parse_action(self, action) -> tuple[int, int]:
        arr = np.asarray(action, dtype=int).reshape(-1)
        if arr.size < 2:
            raise ValueError("action must contain object slot and skill index")
        object_slot = int(np.clip(arr[0], 0, self.max_objects))
        skill_index = int(np.clip(arr[1], 0, len(SKILL_NAMES) - 1))
        return object_slot, skill_index

    def _info(self, selected_action, selected_skill) -> dict[str, Any]:
        assert self._runner is not None
        return {
            "action_mask": self._runner.action_mask(),
            "selected_action": selected_action,
            "selected_skill": selected_skill,
            "selected_skill_name": (
                None if selected_skill is None else SKILL_NAMES[int(selected_skill)]
            ),
            "step_count": self._step_count,
            "episode_sim_time": self._episode_sim_time(),
            "max_episode_seconds": self.max_episode_seconds,
            "pending_action": self._runner._pending_indices(),
        }

    def _episode_sim_time(self) -> float:
        if self._runner is None:
            return 0.0
        return max(0.0, float(self._runner.clock.time()) - self._episode_start_time)

    def _episode_deadline(self) -> float | None:
        if self.max_episode_seconds is None:
            return None
        return self._episode_start_time + float(self.max_episode_seconds)

    def _remaining_time_frac(self) -> float:
        if self.max_episode_seconds is None or self.max_episode_seconds <= 0.0:
            return 1.0
        remaining = float(self.max_episode_seconds) - self._episode_sim_time()
        return float(np.clip(remaining / float(self.max_episode_seconds), 0.0, 1.0))

    @staticmethod
    def _drop_late_reward_events(
        reward: float,
        reward_events: list[dict],
        deadline: float | None,
    ) -> tuple[float, float]:
        if deadline is None:
            return float(reward), 0.0
        late = 0.0
        for event in reward_events:
            event_time = event.get("sim_time")
            if event_time is None:
                continue
            if float(event_time) > deadline:
                late += float(event.get("reward", 0.0))
                event["late_for_episode"] = True
            else:
                event["late_for_episode"] = False
        return float(reward) - late, late

    def _wait_for_first_objects(self, timeout_sec: float) -> None:
        assert self._runner is not None

        deadline = self._runner.clock.time() + max(0.0, timeout_sec)
        while self._runner.clock.time() < deadline:
            self._runner.ingest()
            if self._runner.ordered_objects():
                return
            self._runner.clock.sleep(0.05)


def main() -> int:
    env = GP8RecyclingEnv(max_objects=6)
    obs, info = env.reset()
    print("obs", obs.shape, "mask", info["action_mask"].shape)
    action = np.array([env.max_objects, ACTION_THROW], dtype=int)
    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(action)
        print("step", reward, terminated, truncated, info["pending_action"])
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

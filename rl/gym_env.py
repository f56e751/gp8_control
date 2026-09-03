"""Gymnasium wrapper for the real-time GP8 MuJoCo RL runner."""

from __future__ import annotations

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
from gp8_control.rl.common import ACTION_THROW, SKILL_NAMES, observation_width
from gp8_control.rl.sim_runner import SimRlRunner


class GP8RecyclingEnv(gym.Env if gym is not None else object):
    """Real-time high-level RL env using one-step-ahead actions."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        max_objects: int = 6,
        include_eta: bool = False,
        max_steps: int = 200,
        cfg: Config | None = None,
    ) -> None:
        if spaces is None:
            raise ImportError("GP8RecyclingEnv requires gymnasium or gym.")
        self.max_objects = int(max_objects)
        self.include_eta = bool(include_eta)
        self.max_steps = int(max_steps)
        self.cfg = cfg
        self._runner: SimRlRunner | None = None
        self._step_count = 0
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_width(self.max_objects, self.include_eta),),
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
        self._runner = SimRlRunner(
            cfg=self.cfg,
            max_objects=self.max_objects,
            include_eta=self.include_eta,
        )
        self._runner.start()
        self._step_count = 0
        self._wait_for_first_objects(float((options or {}).get("startup_timeout", 8.0)))
        obs = self._runner.observation()
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
        reward, exec_info = self._runner.step(next_action)
        self._step_count += 1
        terminated = False
        truncated = self._step_count >= self.max_steps
        obs = self._runner.observation()
        info = self._info(selected_action=object_slot, selected_skill=skill_index)
        info.update(exec_info)
        info["action_valid"] = valid
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
            "pending_action": self._runner._pending_indices(),
        }

    def _wait_for_first_objects(self, timeout_sec: float) -> None:
        assert self._runner is not None
        import time

        deadline = time.time() + max(0.0, timeout_sec)
        while time.time() < deadline:
            self._runner.ingest()
            if self._runner.ordered_objects():
                return
            time.sleep(0.05)


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

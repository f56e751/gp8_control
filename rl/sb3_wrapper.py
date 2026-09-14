"""Stable-Baselines3 adapters for the GP8 Gym environment."""

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

from gp8_control.rl.common import SKILL_NAMES


class FlatMaskedActionWrapper(gym.ActionWrapper if gym is not None else object):
    """Expose pairwise ``(object_slot, skill)`` actions as one flat Discrete action.

    SB3's standard MultiDiscrete distribution masks dimensions independently,
    while this project needs pairwise validity. A flat action lets
    ``sb3-contrib``'s MaskablePPO consume exactly the mask emitted by
    ``GP8RecyclingEnv``.
    """

    def __init__(self, env) -> None:
        if spaces is None:
            raise ImportError("FlatMaskedActionWrapper requires gymnasium or gym.")
        super().__init__(env)
        self.max_objects = int(env.max_objects)
        self.num_skills = len(SKILL_NAMES)
        self.action_space = spaces.Discrete((self.max_objects + 1) * self.num_skills)

    def action(self, action) -> np.ndarray:
        flat = int(np.asarray(action, dtype=int).item())
        flat = int(np.clip(flat, 0, self.action_space.n - 1))
        slot = flat // self.num_skills
        skill = flat % self.num_skills
        return np.array([slot, skill], dtype=int)

    def reverse_action(self, action) -> int:
        arr = np.asarray(action, dtype=int).reshape(-1)
        if arr.size < 2:
            raise ValueError("action must contain object slot and skill index")
        slot = int(np.clip(arr[0], 0, self.max_objects))
        skill = int(np.clip(arr[1], 0, self.num_skills - 1))
        return slot * self.num_skills + skill

    def action_masks(self) -> np.ndarray:
        mask = None
        runner = getattr(self.unwrapped, "_runner", None)
        if runner is not None:
            mask = runner.action_mask()
        if mask is None:
            return np.ones(self.action_space.n, dtype=bool)
        flat = np.asarray(mask, dtype=bool).reshape(-1)
        if flat.size != self.action_space.n:
            raise RuntimeError(
                f"flat action mask width mismatch {flat.size} != {self.action_space.n}"
            )
        return flat

    def flat_to_pair(self, action: int) -> tuple[int, int]:
        pair = self.action(action)
        return int(pair[0]), int(pair[1])

    def step(self, action):
        return self.env.step(self.action(action))

    def reset(self, **kwargs: Any):
        return self.env.reset(**kwargs)

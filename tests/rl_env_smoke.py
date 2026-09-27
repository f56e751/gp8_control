"""Smoke test for the non-ROS GP8 Gym environment."""

from __future__ import annotations

import sys

import numpy as np

from gp8_control.rl.gym_env import GP8RecyclingEnv
from gp8_control.rl.sim_runner import ACTION_THROW


def main(argv=None) -> int:
    del argv
    env = GP8RecyclingEnv(max_objects=3, include_eta=False, max_steps=3)
    try:
        obs, info = env.reset(options={"startup_timeout": 4.0})
        expected_obs = 3 * 8 + 6 + 3 + 4
        checks = [
            ("obs shape", obs.shape == (expected_obs,)),
            ("mask shape", info["action_mask"].shape == (4, 2)),
            ("skip valid", bool(info["action_mask"][3, ACTION_THROW])),
        ]
        action = np.array([3, ACTION_THROW], dtype=int)
        obs, reward, terminated, truncated, info = env.step(action)
        checks.extend([
            ("step obs shape", obs.shape == (expected_obs,)),
            ("reward finite", np.isfinite(reward)),
            ("not terminated", not terminated),
            ("mask after step", info["action_mask"].shape == (4, 2)),
        ])
        failures = [name for name, ok in checks if not ok]
        for name, ok in checks:
            print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        if failures:
            print("FAILURES:", ", ".join(failures))
            return 1
        print("ALL PASS")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    sys.exit(main())

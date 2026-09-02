"""Record a short one-step-ahead RL environment rollout.

Set ``GP8_SIM_RECORD=/path/to/file.mp4`` and ``MUJOCO_GL=egl`` to write video.
 The default rollout policy emulates ``app.py`` class routing: metal is pushed,
 transparent/other objects are thrown. The action still goes through the RL
 environment's mask and one-step-ahead interface.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from gp8_control.rl.gym_env import GP8RecyclingEnv
from gp8_control.rl.sim_runner import ACTION_THROW, SKILL_NAMES


def _choose_masked_action(mask: np.ndarray, prefer: str) -> np.ndarray:
    prefer_idx = SKILL_NAMES.index(prefer)
    skill_order = [prefer_idx] + [
        idx for idx in range(len(SKILL_NAMES)) if idx != prefer_idx
    ]
    object_rows = mask.shape[0] - 1
    for skill_idx in skill_order:
        for slot in range(object_rows):
            if mask[slot, skill_idx]:
                return np.array([slot, skill_idx], dtype=int)
    return np.array([object_rows, ACTION_THROW], dtype=int)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="rl_env_record")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--max-objects", type=int, default=6)
    parser.add_argument("--startup-timeout", type=float, default=8.0)
    parser.add_argument("--policy", choices=("app", "prefer"), default="app")
    parser.add_argument("--prefer", choices=SKILL_NAMES, default="throw")
    parser.add_argument("--tail-seconds", type=float, default=2.0)
    args = parser.parse_args(argv)

    env = GP8RecyclingEnv(max_objects=args.max_objects, max_steps=args.steps)
    try:
        _, info = env.reset(options={"startup_timeout": args.startup_timeout})
        for step_idx in range(args.steps):
            if args.policy == "app":
                action = env._runner.app_heuristic_action_indices(info["action_mask"])
            else:
                action = _choose_masked_action(info["action_mask"], args.prefer)
            obs, reward, terminated, truncated, info = env.step(action)
            del obs
            print(
                f"step={step_idx} action={action.tolist()} "
                f"skill={SKILL_NAMES[int(action[1])]} reward={reward:+.3f} "
                f"valid={info.get('action_valid')} pending={info.get('pending_action')}",
                flush=True,
            )
            if terminated or truncated:
                break
        time.sleep(max(0.0, args.tail_seconds))
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    sys.exit(main())

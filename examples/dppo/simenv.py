"""Simulator access shared by data collection and evaluation.

Kept free of torch, so the thread limits set in ``worker_env_defaults`` are in
place before any numerical library starts its thread pools.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys

import numpy as np

from gp8_control.rl.common import (
    BBOX_OBSERVATION_CORNERS,
    BBOX_OBSERVATION_MODES,
    SKILL_NAMES,
    skip_action,
)

# Simulator profile of every dataset and every evaluation reported in README.md.
# A variable that is already set in the environment wins, with a warning, so
# another profile can still be tried from the shell.
SIM_PROFILE = {
    "GP8_SIM_RANDOMIZE": "true",
    "GP8_SIM_PHYSICAL_BELT": "true",
    "GP8_SIM_BELT_CONTACT_PARAMS": "old",
    "GP8_SIM_SUCTION_P": "metal:0.5,transparent:0.9",
    "GP8_RL_SIM_TRUTH_TRACKS": "true",
}
_FIXED = {"GP8_SIM_RECORD": "", "GP8_BACKEND": "mujoco", "GP8_SIM_REALTIME": "false"}
_QUIET = {"GP8_RL_SIM_LOG": "0", "GP8_SIM_LANDING_LOG": "0"}


def worker_env_defaults() -> None:
    """Per-process simulator settings; call before the simulator is imported."""
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    os.environ.update(_FIXED)
    for key, value in _QUIET.items():
        os.environ.setdefault(key, value)
    for key, value in SIM_PROFILE.items():
        if os.environ.setdefault(key, value) != value:
            print(f"warning: {key}={os.environ[key]!r} overrides the reported simulator "
                  f"profile ({value!r})", file=sys.stderr)
    known = {*SIM_PROFILE, *_FIXED, *_QUIET}
    extra = sorted(k for k in os.environ if k.startswith("GP8_SIM_") and k not in known)
    if extra:
        print(f"warning: simulator settings outside the reported profile: {', '.join(extra)}",
              file=sys.stderr)


def add_obs_args(parser) -> None:
    """Observation layout; must match between collection, training and evaluation."""
    parser.add_argument("--max-objects", type=int, default=6)
    parser.add_argument("--bbox-observation", choices=BBOX_OBSERVATION_MODES,
                        default=BBOX_OBSERVATION_CORNERS)
    parser.add_argument("--include-suction-p", action="store_true")


def add_env_args(parser) -> None:
    """Episode settings read by ``make_env`` and ``env.reset``, plus the worker count."""
    add_obs_args(parser)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--max-episode-seconds", type=float, default=240.0)
    parser.add_argument("--startup-timeout", type=float, default=8.0)
    parser.add_argument("--workers", type=int, default=1)


def make_env(args):
    from gp8_control.config import Config
    from gp8_control.rl.gym_env import GP8RecyclingEnv

    cfg = Config()
    cfg.PICK_LOG_CSV = ""
    return GP8RecyclingEnv(
        max_objects=args.max_objects,
        max_steps=args.max_steps,
        max_episode_seconds=args.max_episode_seconds,
        cfg=cfg,
        realtime=False,
        bbox_observation=args.bbox_observation,
        include_suction_p=args.include_suction_p,
    )


def run_seeds(worker, seeds: list[int], args) -> list:
    """Call ``worker((seeds, args))`` in ``args.workers`` processes, one env each."""
    workers = max(1, min(args.workers, len(seeds)))
    jobs = [(seeds[i::workers], args) for i in range(workers)]
    if workers == 1:
        return [worker(jobs[0])]
    # spawn: each worker builds its own MuJoCo model and torch predictor.
    with mp.get_context("spawn").Pool(workers) as pool:
        return pool.map(worker, jobs)


def canonical_mask(mask) -> np.ndarray:
    """Flatten a ``(max_objects + 1, n_skills)`` mask and keep ONE skip encoding.

    The env accepts the skip slot with either skill as the same no-op; logging
    and acting through the env's own ``skip_action`` keeps the action likelihood
    from being split across equivalent outputs.
    """
    m = np.asarray(mask, dtype=bool).reshape(-1, len(SKILL_NAMES)).copy()
    slot, skill = skip_action(m.shape[0] - 1)
    m[slot, np.arange(m.shape[1]) != skill] = False
    return m.reshape(-1)


def flat_action(slot: int, skill: int) -> int:
    """``(object_slot, skill)`` -> the flat action index the policy and the data use."""
    return int(slot) * len(SKILL_NAMES) + int(skill)


def split_action(flat: int) -> np.ndarray:
    """Flat action index -> ``[object_slot, skill]`` as ``env.step`` expects it."""
    return np.array(divmod(int(flat), len(SKILL_NAMES)), dtype=int)

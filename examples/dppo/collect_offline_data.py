"""Collect offline rollouts from ``GP8RecyclingEnv`` for DPPO / IQL / BC training.

Behaviour policy: the app's class-routing heuristic
(``SimRlRunner.app_heuristic_action_indices``) with epsilon-uniform exploration
over the valid actions. Epsilon is redrawn every ``--eps-block`` decisions, so
one episode (one belt speed / arrival rate) contains both near-heuristic and
near-random stretches.

One ``ep_<seed>.npz`` per episode (existing files are skipped, so an interrupted
collection resumes):

    obs         (T, obs_dim) float32
    act         (T,)         int64    flat action = object_slot * 2 + skill
    mask        (T, A)       bool     canonical mask (single skip encoding)
    rew         (T,)         float32  reward exactly as returned by env.step
    eps         (T,)         float32  behaviour epsilon at that step
    executed    (T,)         int8     skill executed in that step (-1 = none)
    seed, sim_seconds                 episode seed and simulated duration

Run from the parent directory of the ``gp8_control`` package. The simulator
profile is the one in simenv.py; README.md has the exact commands that
reproduce the reported dataset.

    python -m gp8_control.examples.dppo.collect_offline_data \
        --out-dir gp8_control/examples/dppo/runs/data --episodes 256 --workers 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from gp8_control.examples.dppo.simenv import (
    add_env_args,
    canonical_mask,
    flat_action,
    make_env,
    run_seeds,
    split_action,
    worker_env_defaults,
)
from gp8_control.rl.common import SKILL_NAMES


def collect_episode(env, seed: int, args) -> dict:
    rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed, options={"startup_timeout": args.startup_timeout})
    runner = env._runner
    buf = {k: [] for k in ("obs", "act", "mask", "rew", "eps", "executed")}
    eps = 0.0
    for t in range(args.max_steps):
        if t % args.eps_block == 0:
            eps = float(rng.choice(args.eps_choices))
        mask = canonical_mask(info["action_mask"])
        if rng.random() < eps:
            flat = int(rng.choice(np.flatnonzero(mask)))
        else:
            slot, skill = runner.app_heuristic_action_indices(info["action_mask"])
            flat = flat_action(slot, skill)
            if args.skill_flip is not None and slot < args.max_objects:
                # Skill noise: keep the heuristic's object, swap throw <-> push.
                other = flat_action(slot, 1 - int(skill))
                if mask[other] and rng.random() < args.skill_flip:
                    flat = other
        buf["obs"].append(obs)
        buf["act"].append(flat)
        buf["mask"].append(mask)
        buf["eps"].append(eps)
        obs, reward, terminated, truncated, info = env.step(split_action(flat))
        buf["rew"].append(float(reward))
        buf["executed"].append(
            SKILL_NAMES.index(info["executed_skill"]) if info.get("executed") else -1
        )
        if terminated or truncated:
            break
    return {
        "obs": np.asarray(buf["obs"], dtype=np.float32),
        "act": np.asarray(buf["act"], dtype=np.int64),
        "mask": np.asarray(buf["mask"], dtype=bool),
        "rew": np.asarray(buf["rew"], dtype=np.float32),
        "eps": np.asarray(buf["eps"], dtype=np.float32),
        "executed": np.asarray(buf["executed"], dtype=np.int8),
        "seed": np.int64(seed),
        "sim_seconds": np.float32(info.get("episode_sim_time", 0.0)),
    }


def _worker(job) -> None:
    seeds, args = job
    worker_env_defaults()
    env = make_env(args)
    try:
        for seed in seeds:
            t0 = time.time()
            ep = collect_episode(env, int(seed), args)
            path = Path(args.out_dir) / f"ep_{int(seed):06d}.npz"
            tmp = path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, **ep)
            tmp.replace(path)
            print(f"seed {seed}: steps {len(ep['act'])} return {ep['rew'].sum():+.1f} "
                  f"wall {time.time() - t0:.0f}s", flush=True)
    finally:
        env.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="collect_offline_data")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--seeds-file", default="",
                        help="text file of episode seeds, one per line; replaces "
                             "--seed-start/--episodes")
    parser.add_argument("--eps-choices", type=lambda s: [float(x) for x in s.split(",")],
                        default=[0.0, 0.2, 0.5, 1.0])
    parser.add_argument("--eps-block", type=int, default=20)
    parser.add_argument("--skill-flip", type=float, default=None,
                        help="probability of swapping the heuristic's skill (throw <-> push) "
                             "for the object it chose, when the other skill is valid. Giving "
                             "the option, even as 0, adds one random draw per heuristic "
                             "decision and so changes the episodes a seed produces")
    add_env_args(parser)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    worker_env_defaults()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir = str(out_dir)
    if args.seeds_file:
        wanted = [int(s) for s in Path(args.seeds_file).read_text(encoding="utf-8").split()]
    else:
        wanted = list(range(args.seed_start, args.seed_start + args.episodes))
    seeds = [s for s in wanted if not (out_dir / f"ep_{s:06d}.npz").exists()]
    from gp8_control.tools.train_ppo_smoke import _sim_config_summary

    record = {"args": vars(args), "sim_config": _sim_config_summary(args)}
    record_path = out_dir / "collection.json"
    if record_path.exists():
        # One dataset = one simulator profile: refuse to add episodes under another.
        before = json.loads(record_path.read_text(encoding="utf-8"))["sim_config"]["resolved"]
        now = record["sim_config"]["resolved"]
        if json.dumps(before, sort_keys=True) != json.dumps(now, sort_keys=True):
            raise SystemExit(f"{record_path} was collected under a different simulator profile")
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(f"collecting {len(seeds)} episodes into {out_dir} "
          f"({len(wanted) - len(seeds)} already present)", flush=True)
    if seeds:
        run_seeds(_worker, seeds, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

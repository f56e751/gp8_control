"""Evaluate an offline-trained policy (or a reference policy) in ``GP8RecyclingEnv``.

One full time-budgeted episode per seed, greedy actions. Appends one JSON line
per episode to ``--out``. The simulator profile is the one in simenv.py.

    python -m gp8_control.examples.dppo.eval_offline_policy \
        --ckpt gp8_control/examples/dppo/runs/dppo_s0.pt --tag dppo_s0 \
        --seeds 0-9 --workers 5 --out gp8_control/examples/dppo/runs/test.dppo_s0.jsonl
    ... --policy heuristic --tag heuristic     # the app's class routing
    ... --policy random --tag random           # uniform over valid actions
"""

from __future__ import annotations

import argparse
import json
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


def _parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def run_episode(env, policy, seed: int, args) -> dict:
    rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed, options={"startup_timeout": args.startup_timeout})
    total = 0.0
    selected = {name: 0 for name in (*SKILL_NAMES, "skip")}
    executed = {name: 0 for name in SKILL_NAMES}
    in_bin = missed = 0
    reached_time_limit = False
    steps = 0
    for _ in range(args.max_steps):
        mask = canonical_mask(info["action_mask"])
        if args.policy == "heuristic":
            flat = flat_action(*env._runner.app_heuristic_action_indices(info["action_mask"]))
        elif args.policy == "random":
            flat = int(rng.choice(np.flatnonzero(mask)))
        else:
            flat = policy.act(obs, mask)
        action = split_action(flat)
        slot, skill = action
        selected["skip" if slot >= args.max_objects else SKILL_NAMES[skill]] += 1
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        total += float(reward)
        if info.get("executed"):
            executed[info["executed_skill"]] += 1
        for event in info.get("reward_events", []):
            if event.get("late_for_episode"):
                continue
            in_bin += float(event["reward"]) > 0.0
            missed += float(event["reward"]) < 0.0
        if terminated or truncated:
            reached_time_limit = bool(info.get("truncated_by_time"))
            break
    return {
        "tag": args.tag, "seed": int(seed), "return": total, "steps": steps,
        "selected": selected, "executed": executed,
        "in_bin": int(in_bin), "missed": int(missed),
        "reached_time_limit": reached_time_limit,
        "sim_seconds": float(info.get("episode_sim_time", 0.0)),
    }


def _worker(job) -> list[dict]:
    seeds, args = job
    worker_env_defaults()
    policy = None
    if args.policy == "ckpt":
        import torch as th

        from gp8_control.examples.dppo.offline import OfflinePolicy

        th.set_num_threads(1)
        policy = OfflinePolicy.load(args.ckpt)
    env = make_env(args)
    rows = []
    try:
        for seed in seeds:
            rows.append(run_episode(env, policy, int(seed), args))
            print(json.dumps(rows[-1]), flush=True)
    finally:
        env.close()
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="eval_offline_policy")
    parser.add_argument("--policy", choices=("ckpt", "heuristic", "random"), default="ckpt")
    parser.add_argument("--ckpt", default="",
                        help="checkpoint to evaluate; its observation layout replaces "
                             "--max-objects / --bbox-observation / --include-suction-p")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", type=_parse_seeds, default=list(range(10)))
    add_env_args(parser)
    args = parser.parse_args(argv)
    if args.policy == "ckpt" and not args.ckpt:
        parser.error("--ckpt is required with --policy ckpt")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    worker_env_defaults()
    if args.policy == "ckpt":
        from gp8_control.examples.dppo.offline import OfflinePolicy

        # The env must produce the observation the policy was trained on.
        meta = OfflinePolicy.load_meta(args.ckpt)
        args.max_objects = meta["max_objects"]
        args.bbox_observation = meta["bbox_observation"]
        args.include_suction_p = meta["include_suction_p"]
    results = run_seeds(_worker, args.seeds, args)
    rows = sorted((r for chunk in results for r in chunk), key=lambda r: r["seed"])
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    returns = np.array([r["return"] for r in rows])
    sd = returns.std(ddof=1) if len(returns) > 1 else 0.0
    print(f"{args.tag}: return {returns.mean():.2f} ± {sd:.2f} (n={len(returns)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

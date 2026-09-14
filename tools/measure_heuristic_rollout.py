"""Record a heuristic RL rollout and summarize executed skill durations."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path

from gp8_control.config import Config
from gp8_control.rl.gym_env import GP8RecyclingEnv
from gp8_control.rl.common import SKILL_NAMES


def _stats(values: list[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "mean_s": None,
            "std_s": None,
            "min_s": None,
            "max_s": None,
        }
    return {
        "count": len(values),
        "mean_s": float(statistics.fmean(values)),
        "std_s": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "min_s": float(min(values)),
        "max_s": float(max(values)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="measure_heuristic_rollout")
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Safety cap on Gym decisions. Use <=0 to disable.",
    )
    parser.add_argument("--max-episode-seconds", type=float, default=240.0)
    parser.add_argument("--max-objects", type=int, default=6)
    parser.add_argument("--startup-timeout", type=float, default=8.0)
    parser.add_argument("--tail-seconds", type=float, default=2.0)
    parser.add_argument("--out-dir", default="runs/heuristic_timing")
    parser.add_argument("--csv", default="")
    parser.add_argument("--summary", default="")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv).expanduser().resolve() if args.csv else out_dir / "timings.csv"
    summary_path = (
        Path(args.summary).expanduser().resolve()
        if args.summary
        else out_dir / "summary.json"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    cfg.PICK_LOG_CSV = str(out_dir / "pick_log.csv")
    for suffix in ("", ".push"):
        path = out_dir / f"pick_log{suffix}.csv"
        if path.exists():
            path.unlink()

    max_episode_seconds = (
        None if args.max_episode_seconds <= 0.0 else args.max_episode_seconds
    )
    env = GP8RecyclingEnv(
        max_objects=args.max_objects,
        max_steps=args.steps,
        max_episode_seconds=max_episode_seconds,
        cfg=cfg,
    )
    rows: list[dict] = []
    try:
        _, info = env.reset(seed=args.seed, options={"startup_timeout": args.startup_timeout})
        runner = env._runner
        assert runner is not None
        for step_idx in range(args.steps):
            action = runner.app_heuristic_action_indices(info["action_mask"])
            queued_skill = SKILL_NAMES[int(action[1])]
            t0 = float(runner.clock.time())
            obs, reward, terminated, truncated, info = env.step(action)
            del obs
            t1 = float(runner.clock.time())
            if info.get("executed"):
                rows.append(
                    {
                        "step": int(step_idx),
                        "duration_s": float(t1 - t0),
                        "executed_skill": str(info.get("executed_skill", "")),
                        "execution_success": bool(info.get("execution_success", False)),
                        "execution_detail": str(info.get("execution_detail", "")),
                        "executed_track_id": info.get("executed_track_id"),
                        "reward": float(reward),
                        "queued_next_slot": int(action[0]),
                        "queued_next_skill": queued_skill,
                        "sim_time_start": t0,
                        "sim_time_end": t1,
                    }
                )
            print(
                f"step={step_idx} queued={action.tolist()} next={queued_skill} "
                f"executed={info.get('executed')} "
                f"skill={info.get('executed_skill')} dt={t1 - t0:.3f}s "
                f"t={info.get('episode_sim_time', 0.0):.2f}s "
                f"reward={float(reward):+.3f}",
                flush=True,
            )
            if terminated or truncated:
                break
        if env._runner is not None:
            env._runner.clock.sleep(max(0.0, args.tail_seconds))
    finally:
        env.close()

    fields = [
        "step",
        "duration_s",
        "executed_skill",
        "execution_success",
        "execution_detail",
        "executed_track_id",
        "reward",
        "queued_next_slot",
        "queued_next_skill",
        "sim_time_start",
        "sim_time_end",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    by_skill = {
        skill: _stats(
            [
                float(row["duration_s"])
                for row in rows
                if row["executed_skill"] == skill and row["execution_success"]
            ]
        )
        for skill in SKILL_NAMES
    }
    summary = {
        "steps": int(args.steps),
        "max_episode_seconds": max_episode_seconds,
        "seed": int(args.seed),
        "record_path": os.environ.get("GP8_SIM_RECORD", ""),
        "csv_path": str(csv_path),
        "pick_log_csv": str(out_dir / "pick_log.csv"),
        "push_log_csv": str(out_dir / "pick_log.push.csv"),
        "successful_action_duration_by_skill": by_skill,
        "all_executed_count": len(rows),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

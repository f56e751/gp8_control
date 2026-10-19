"""Preliminary MaskablePPO smoke training for the GP8 recycling env."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from gp8_control.backends.mujoco_sim import SimConfig
from gp8_control.config import Config
from gp8_control.rl.gym_env import GP8RecyclingEnv
from gp8_control.rl.common import SKILL_NAMES
from gp8_control.rl.sb3_wrapper import FlatMaskedActionWrapper


SIM_ENV_KEYS = (
    "GP8_SIM_PHYSICAL_BELT",
    "GP8_SIM_BELT_CONTACT_PARAMS",
    "GP8_SIM_RANDOMIZE",
    "GP8_SIM_BELT_SPEED",
    "GP8_SIM_BELT_SPEED_RANGE",
    "GP8_SIM_SPAWN_INTERVAL",
    "GP8_SIM_SPAWN_RATE_HZ_RANGE",
    "GP8_SIM_SPAWN_X_RANGE",
    "GP8_SIM_RANDOM_CLASS",
    "GP8_SIM_RANDOM_SIZE",
    "GP8_SIM_RANDOM_YAW",
    "GP8_SIM_OBJECT_HALF_X_RANGE",
    "GP8_SIM_OBJECT_HALF_Y_RANGE",
    "GP8_SIM_OBJECT_HALF_Z",
    "GP8_SIM_BOX_COUNT",
    "GP8_SIM_SUCTION_P",
    "GP8_SIM_METAL_SUCTION_BINARY",
    "GP8_SIM_METAL_SUCTION_MODE",
    "GP8_SIM_BBOX_MODE",
    "GP8_SIM_CAM_FOV_GATE",
    "GP8_RL_BBOX_OBSERVATION",
    "GP8_RL_SIM_TRUTH_TRACKS",
    "GP8_RL_INCLUDE_SUCTION_P",
)


def _require_sb3():
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.evaluation import evaluate_policy
        from sb3_contrib.common.maskable.utils import get_action_masks
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    except ImportError as exc:
        raise SystemExit(
            "Missing PPO dependencies. Install with: "
            "python -m pip install stable-baselines3 sb3-contrib tensorboard"
        ) from exc
    return MaskablePPO, evaluate_policy, get_action_masks, Monitor, SubprocVecEnv, VecMonitor


def _thread_env_defaults() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _env_truth_tracks() -> bool | None:
    value = os.environ.get("GP8_RL_SIM_TRUTH_TRACKS")
    if value is None:
        return None
    return value.strip().lower() in ("1", "true", "yes", "on")


def _resolve_truth_tracks(args) -> bool:
    if args.truth_tracks is not None:
        return bool(args.truth_tracks)
    env_value = _env_truth_tracks()
    if env_value is not None:
        return bool(env_value)
    return not bool(args.eval_only)


def _set_truth_tracks_env(args) -> bool:
    enabled = _resolve_truth_tracks(args)
    os.environ["GP8_RL_SIM_TRUTH_TRACKS"] = "true" if enabled else "false"
    return enabled


def _worker_log_path(out_dir: str, rank: int) -> str:
    return str(Path(out_dir).expanduser().resolve() / "worker_logs" / f"pick_log_env{rank}.csv")


def _make_env(args, *, rank: int = 0) -> FlatMaskedActionWrapper:
    cfg = Config()
    cfg.PICK_LOG_CSV = _worker_log_path(args.out_dir, rank)
    env = GP8RecyclingEnv(
        max_objects=args.max_objects,
        include_eta=args.include_eta,
        max_steps=args.max_steps,
        max_episode_seconds=args.max_episode_seconds,
        cfg=cfg,
        realtime=False,
        bbox_observation=args.bbox_observation,
        include_suction_p=args.include_suction_p,
    )
    return FlatMaskedActionWrapper(env)


def _make_monitored_env(args, Monitor, *, rank: int = 0) -> FlatMaskedActionWrapper:
    env = _make_env(args, rank=rank)
    monitor_path = Path(args.out_dir).expanduser().resolve() / "monitor_logs" / f"env{rank}"
    monitor_path.parent.mkdir(parents=True, exist_ok=True)
    return Monitor(env, filename=str(monitor_path))


def _make_subproc_env(rank: int, args_dict: dict[str, Any]):
    def _init():
        _thread_env_defaults()
        os.environ["GP8_SIM_RECORD"] = ""
        os.environ["GP8_BACKEND"] = "mujoco"
        os.environ["GP8_SIM_REALTIME"] = "false"
        if not args_dict.get("sim_logs", False):
            os.environ["GP8_RL_SIM_LOG"] = "0"
            os.environ["GP8_SIM_LANDING_LOG"] = "0"
        args = argparse.Namespace(**args_dict)
        return _make_env(args, rank=rank)

    return _init


def _make_training_env(args, Monitor, SubprocVecEnv, VecMonitor):
    if int(args.n_envs) <= 1:
        return _make_monitored_env(args, Monitor, rank=0)
    args_dict = vars(args).copy()
    env_fns = [_make_subproc_env(rank, args_dict) for rank in range(int(args.n_envs))]
    env = SubprocVecEnv(env_fns, start_method=args.vec_start_method)
    monitor_path = Path(args.out_dir).expanduser().resolve() / "monitor_logs" / "vec_monitor.csv"
    monitor_path.parent.mkdir(parents=True, exist_ok=True)
    return VecMonitor(env, filename=str(monitor_path))


def _write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _sim_config_summary(args) -> dict:
    sim_cfg = SimConfig()
    return {
        "command": sys.argv,
        "env": {key: os.environ.get(key) for key in SIM_ENV_KEYS},
        "resolved": {
            "physical_belt": bool(sim_cfg.physical_belt),
            "belt_contact_params": str(sim_cfg.belt_contact_params),
            "randomize": bool(sim_cfg.randomize),
            "belt_speed": float(sim_cfg.belt_speed),
            "spawn_interval": float(sim_cfg.spawn_interval),
            "belt_speed_range": list(map(float, sim_cfg.belt_speed_range)),
            "spawn_rate_hz_range": list(map(float, sim_cfg.spawn_rate_hz_range)),
            "spawn_x_range": list(map(float, sim_cfg.spawn_x_range)),
            "random_class": bool(sim_cfg.random_class),
            "random_size": bool(sim_cfg.random_size),
            "random_yaw": bool(sim_cfg.random_yaw),
            "object_half_x_range": list(map(float, sim_cfg.object_half_x_range)),
            "object_half_y_range": list(map(float, sim_cfg.object_half_y_range)),
            "object_half_z": float(sim_cfg.object_half_z),
            "box_count": int(sim_cfg.box_count),
            "suction_p": dict(sim_cfg.suction_p),
            "metal_suction_binary": bool(sim_cfg.metal_suction_binary),
            "metal_suction_mode": str(sim_cfg.metal_suction_mode),
            "bbox_mode": str(sim_cfg.bbox_mode),
            "cam_fov_gate": bool(sim_cfg.cam_fov_gate),
            "rl_realtime": False,
            "rl_bbox_observation": str(args.bbox_observation),
            "rl_sim_truth_tracks": bool(_env_truth_tracks()),
            "rl_include_suction_p": bool(args.include_suction_p),
        },
    }


def _object_debug_rows(runner, mask) -> list[dict]:
    objects = runner.ordered_objects()
    now = runner.clock.time()
    joints = runner.current_joints()
    pre_delay = (
        0.0 if joints is None else runner._pending_chain_delay(now, joints)
    )
    rows = []
    for slot, target in enumerate(objects[: runner.max_objects]):
        y_now = runner.ctx.object_y_now(target, now, runner.conveyor.current)
        row = {
            "slot": int(slot),
            "track_id": getattr(target, "track_id", None),
            "sim_object_id": getattr(target, "sim_object_id", None),
            "class_name": str(getattr(target, "class_name", "")),
            "conf": float(getattr(target, "conf", 0.0)),
            "suction_p": float(getattr(target, "suction_p", 1.0)),
            "xyz_now": [
                float(target.T_grasp_base[0, 3]),
                float(y_now),
                float(target.T_grasp_base[2, 3]),
            ],
            "age_s": float(now - float(getattr(target, "last_seen", now))),
            "valid": {
                skill_name: bool(mask[slot, skill_index])
                for skill_index, skill_name in enumerate(SKILL_NAMES)
            },
            "eta": {},
        }
        if joints is not None:
            for skill_name in SKILL_NAMES:
                feasible, eta = runner._evaluate_feasibility(
                    target,
                    skill_name,
                    joints,
                    now,
                    pre_delay=pre_delay,
                )
                row["eta"][skill_name] = float(eta) if feasible else -1.0
        rows.append(row)
    return rows


def _trace_record(
    env,
    step_idx: int,
    decision_sim_time: float,
    after_step_sim_time: float,
    flat_action: int,
    pair,
    masks,
    selected_object: dict | None,
    objects_before_step: list[dict],
    reward,
    info,
) -> dict:
    runner = env.unwrapped._runner
    mask_2d = np.asarray(masks, dtype=bool).reshape(runner.max_objects + 1, len(SKILL_NAMES))
    return {
        "step": int(step_idx),
        "sim_time": float(decision_sim_time),
        "decision_sim_time": float(decision_sim_time),
        "after_step_sim_time": float(after_step_sim_time),
        "flat_action": int(flat_action),
        "object_slot": int(pair[0]),
        "skill_index": int(pair[1]),
        "skill_name": SKILL_NAMES[int(pair[1])],
        "is_skip_slot": bool(int(pair[0]) >= runner.max_objects),
        "selected_object": selected_object,
        "valid_flat_actions": int(np.asarray(masks, dtype=bool).sum()),
        "valid_pairs": np.argwhere(mask_2d).astype(int).tolist(),
        "objects_before_step": objects_before_step,
        "pending_before_step": info.get("pending_before_step"),
        "reward": float(reward),
        "action_valid": bool(info.get("action_valid", False)),
        "episode_sim_time": float(info.get("episode_sim_time", 0.0)),
        "max_episode_seconds": info.get("max_episode_seconds"),
        "truncated_by_time": bool(info.get("truncated_by_time", False)),
        "truncated_by_steps": bool(info.get("truncated_by_steps", False)),
        "executed": bool(info.get("executed", False)),
        "execution_success": bool(info.get("execution_success", False)),
        "execution_detail": str(info.get("execution_detail", "")),
        "executed_skill": info.get("executed_skill"),
        "executed_track_id": info.get("executed_track_id"),
        "dropped_stale_pending": bool(info.get("dropped_stale_pending", False)),
        "reward_events": info.get("reward_events", []),
        "pending_after_step": info.get("pending_action"),
        "truth_tracks": bool(getattr(runner, "truth_tracks", False)),
    }


def train(args) -> int:
    _thread_env_defaults()
    _set_truth_tracks_env(args)
    if not args.sim_logs:
        os.environ["GP8_RL_SIM_LOG"] = "0"
        os.environ["GP8_SIM_LANDING_LOG"] = "0"
    MaskablePPO, evaluate_policy, _get_action_masks, Monitor, SubprocVecEnv, VecMonitor = _require_sb3()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "worker_logs").mkdir(parents=True, exist_ok=True)

    env = _make_training_env(args, Monitor, SubprocVecEnv, VecMonitor)
    t0 = time.time()
    try:
        if args.model:
            model = MaskablePPO.load(
                args.model,
                env=env,
                device=args.device,
                tensorboard_log=str(out_dir / "tb") if args.tensorboard else None,
            )
            model.verbose = args.verbose
            reset_num_timesteps = False
        else:
            policy_kwargs = None
            if args.process_state_input:
                from gp8_control.rl.state_input import ProcessedStateExtractor

                policy_kwargs = dict(
                    features_extractor_class=ProcessedStateExtractor,
                    features_extractor_kwargs=dict(
                        max_objects=args.max_objects,
                        episode_seconds=args.max_episode_seconds or 240.0,
                    ),
                )
            model = MaskablePPO(
                "MlpPolicy",
                env,
                policy_kwargs=policy_kwargs,
                seed=args.seed,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                gamma=args.gamma,
                ent_coef=args.ent_coef,
                device=args.device,
                verbose=args.verbose,
                tensorboard_log=str(out_dir / "tb") if args.tensorboard else None,
            )
            reset_num_timesteps = True
        start_timesteps = int(getattr(model, "num_timesteps", 0))
        from gp8_control.rl.credit import CauseCreditCallback

        model.learn(
            total_timesteps=args.timesteps,
            callback=CauseCreditCallback(),
            progress_bar=args.progress,
            reset_num_timesteps=reset_num_timesteps,
        )
        model_path = out_dir / "ppo_smoke.zip"
        model.save(str(model_path))
        mean_reward, std_reward = evaluate_policy(
            model,
            env,
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            warn=False,
        )
        summary = {
            "total_timesteps": int(args.timesteps),
            "start_timesteps": start_timesteps,
            "final_model_timesteps": int(getattr(model, "num_timesteps", 0)),
            "resume_model_path": str(args.model),
            "seed": int(args.seed),
            "n_envs": int(args.n_envs),
            "n_steps": int(args.n_steps),
            "batch_size": int(args.batch_size),
            "max_objects": int(args.max_objects),
            "max_steps": int(args.max_steps),
            "max_episode_seconds": (
                None
                if args.max_episode_seconds is None
                else float(args.max_episode_seconds)
            ),
            "include_eta": bool(args.include_eta),
            "include_suction_p": bool(args.include_suction_p),
            "bbox_observation": str(args.bbox_observation),
            "process_state_input": bool(args.process_state_input),
            "gamma": float(args.gamma),
            "ent_coef": float(args.ent_coef),
            "device_arg": str(args.device),
            "model_device": str(getattr(model, "device", "")),
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "elapsed_wall_s": float(time.time() - t0),
            "model_path": str(model_path),
            "sim_config": _sim_config_summary(args),
        }
        _write_summary(out_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return 0
    finally:
        env.close()


def evaluate(args) -> int:
    _thread_env_defaults()
    _set_truth_tracks_env(args)
    if args.sim_logs:
        os.environ["GP8_RL_SIM_LOG"] = "1"
        os.environ["GP8_SIM_LANDING_LOG"] = "1"
    MaskablePPO, _evaluate_policy, get_action_masks, _Monitor, _SubprocVecEnv, _VecMonitor = _require_sb3()
    if not args.model:
        raise SystemExit("--model is required with --eval-only")
    env = _make_env(args)
    try:
        model = MaskablePPO.load(args.model, env=env, device=args.device)
        obs, info = env.reset(seed=args.seed, options={"startup_timeout": args.startup_timeout})
        del info
        total_reward = 0.0
        actions = []
        trace_file = None
        if args.trace_jsonl:
            trace_path = Path(args.trace_jsonl).expanduser().resolve()
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_file = trace_path.open("w", encoding="utf-8")
        for step_idx in range(args.eval_steps):
            decision_sim_time = float(env.unwrapped._runner.clock.time())
            masks = get_action_masks(env)
            action, _state = model.predict(
                obs,
                deterministic=True,
                action_masks=masks,
            )
            pair = env.flat_to_pair(int(action))
            pending_before = env.unwrapped._runner._pending_indices()
            mask_2d = np.asarray(masks, dtype=bool).reshape(
                env.unwrapped._runner.max_objects + 1, len(SKILL_NAMES)
            )
            objects_before_step = _object_debug_rows(env.unwrapped._runner, mask_2d)
            selected_object = None
            if int(pair[0]) < env.unwrapped._runner.max_objects:
                objects = env.unwrapped._runner.ordered_objects()
                if int(pair[0]) < len(objects):
                    target = objects[int(pair[0])]
                    selected_object = {
                        "track_id": getattr(target, "track_id", None),
                        "sim_object_id": getattr(target, "sim_object_id", None),
                        "class_name": str(getattr(target, "class_name", "")),
                        "suction_p": float(getattr(target, "suction_p", 1.0)),
                    }
            obs, reward, terminated, truncated, info = env.step(action)
            after_step_sim_time = float(env.unwrapped._runner.clock.time())
            info["pending_before_step"] = [
                int(pending_before[0]),
                int(pending_before[1]),
            ]
            total_reward += float(reward)
            actions.append(
                {
                    "step": int(step_idx),
                    "flat_action": int(action),
                    "object_slot": int(pair[0]),
                    "skill_index": int(pair[1]),
                    "reward": float(reward),
                    "valid": bool(info.get("action_valid", False)),
                    "executed": bool(info.get("executed", False)),
                }
            )
            if trace_file is not None:
                trace_file.write(
                    json.dumps(
                        _trace_record(
                            env,
                            step_idx,
                            decision_sim_time,
                            after_step_sim_time,
                            int(action),
                            pair,
                            masks,
                            selected_object,
                            objects_before_step,
                            reward,
                            info,
                        ),
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                trace_file.flush()
            if terminated or truncated:
                break
        if trace_file is not None:
            trace_file.close()
        tail_seconds = max(0.0, float(args.tail_seconds))
        if env.unwrapped._runner is not None:
            env.unwrapped._runner.clock.sleep(tail_seconds)
        summary = {
            "model_path": str(args.model),
            "steps": len(actions),
            "total_reward": float(total_reward),
            "episode_sim_time": (
                0.0
                if env.unwrapped._runner is None
                else float(env.unwrapped._runner.clock.time())
                - float(env.unwrapped._episode_start_time)
            ),
            "max_episode_seconds": (
                None
                if args.max_episode_seconds is None
                else float(args.max_episode_seconds)
            ),
            "actions": actions,
            "record_path": os.environ.get("GP8_SIM_RECORD", ""),
            "trace_jsonl": str(Path(args.trace_jsonl).expanduser().resolve()) if args.trace_jsonl else "",
            "device_arg": str(args.device),
            "model_device": str(getattr(model, "device", "")),
            "sim_config": _sim_config_summary(args),
        }
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return 0
    finally:
        env.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="train_ppo_smoke")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--model", default="")
    parser.add_argument("--out-dir", default="runs/ppo_smoke")
    parser.add_argument("--timesteps", type=int, default=2000)
    parser.add_argument("--eval-steps", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--max-objects", type=int, default=6)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Safety cap on Gym decisions per episode. Use <=0 to disable.",
    )
    parser.add_argument(
        "--max-episode-seconds",
        type=float,
        default=240.0,
        help="Sim-time episode budget. Use <=0 to disable and rely on --max-steps.",
    )
    parser.add_argument("--include-eta", action="store_true")
    parser.add_argument(
        "--include-suction-p",
        action="store_true",
        default=os.environ.get("GP8_RL_INCLUDE_SUCTION_P", "0").strip().lower()
        in ("1", "true", "yes", "on"),
        help="Append per-object suction_p to the RL observation. Default is off for legacy model compatibility.",
    )
    parser.add_argument(
        "--bbox-observation",
        choices=("size", "corners"),
        default=os.environ.get("GP8_RL_BBOX_OBSERVATION", "size"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--startup-timeout", type=float, default=8.0)
    parser.add_argument("--tail-seconds", type=float, default=2.0)
    parser.add_argument("--trace-jsonl", default="")
    parser.add_argument("--n-steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--vec-start-method", default="fork")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument(
        "--process-state-input",
        action="store_true",
        help="Re-encode joints, pending, class and time inside the policy (rl/state_input.py).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for MaskablePPO policy/training, e.g. auto, cpu, cuda, cuda:0.",
    )
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--sim-logs", action="store_true")
    truth_group = parser.add_mutually_exclusive_group()
    truth_group.add_argument(
        "--truth-tracks",
        dest="truth_tracks",
        action="store_true",
        default=None,
        help="Refresh simulator tracks from MuJoCo truth.",
    )
    truth_group.add_argument(
        "--camera-tracks",
        dest="truth_tracks",
        action="store_false",
        help="Use camera/FOV detections and belt dead reckoning.",
    )
    args = parser.parse_args(argv)
    if args.max_episode_seconds is not None and args.max_episode_seconds <= 0.0:
        args.max_episode_seconds = None
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.eval_only:
        return evaluate(args)
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())

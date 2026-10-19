"""Record a reference rollout of the RL training sim, in the same format as HW.

Runs ``SimRlRunner`` (non-realtime, truth tracks, physical belt, old contact
params — the 2026-10-10 training baseline) with the app's class-routing
heuristic as the policy, and records it with the very same
``Sim2RealRecorder``/``instrument_runner`` the hardware run used. The belt
speed is pinned to the hardware run's measured speed so the two runs are
comparable feature by feature.

Anything here can be overridden from the environment (``GP8_SIM_*`` /
``GP8_RL_*``); explicit settings below are only defaults.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from gp8_control.validation.obs_schema import ObsSchema

TRAINING_ENV_DEFAULTS = {
    "GP8_RL_SIM_TRUTH_TRACKS": "1",
    "GP8_RL_SIM_LOG": "0",
    "GP8_SIM_RANDOMIZE": "1",
    "GP8_SIM_PHYSICAL_BELT": "1",
    "GP8_SIM_BELT_CONTACT_PARAMS": "old",
    "GP8_SIM_SUCTION_P": "1.0",
    "GP8_SIM_LANDING_LOG": "0",
    "GP8_SIM_RECORD": "",
}


def run_sim_reference(
    out_dir,
    *,
    n_steps: int = 30,
    belt_speed: float | None = None,
    schema: ObsSchema | None = None,
    seed: int = 0,
    episode_seconds: float = 240.0,
    max_sim_seconds: float = 240.0,
    source: str = "sim_reference",
    verbose: bool = True,
) -> Path:
    for key, value in TRAINING_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)
    if belt_speed is not None and belt_speed > 0.0:
        os.environ["GP8_SIM_BELT_SPEED_RANGE"] = f"{belt_speed:.4f},{belt_speed:.4f}"

    from gp8_control.config import Config
    from gp8_control.rl.sim_runner import SimRlRunner
    from gp8_control.validation.recorder import (
        Sim2RealRecorder,
        collect_meta,
        instrument_runner,
    )

    schema = schema or ObsSchema()
    runner = SimRlRunner(
        cfg=Config(),
        max_objects=schema.max_objects,
        bbox_observation=schema.bbox_observation,
        include_suction_p=schema.include_suction_p,
        sim_seed=int(seed),
        realtime=False,
    )
    rec = None
    try:
        runner.start()
        meta = collect_meta(runner)
        meta["sim_reference"] = {
            "n_steps": int(n_steps), "seed": int(seed),
            "belt_speed_requested": belt_speed,
            "env": {k: os.environ.get(k) for k in TRAINING_ENV_DEFAULTS},
        }
        rec = Sim2RealRecorder(
            out_dir, clock=runner.clock, source=source, schema=schema,
            auto_analyze=False, meta=meta,
            background_poll=bool(runner.core.cfg.realtime))
        rec.attach(runner)
        instrument_runner(runner, rec, episode_seconds=episode_seconds)
        # n_steps counts EXECUTED skills (idle steps only advance TIME_STEP and
        # the first object needs ~20 s to ride from the camera to the arm).
        executed = 0
        t_limit = runner.clock.time() + float(max_sim_seconds)
        while executed < int(n_steps) and runner.clock.time() < t_limit:
            mask = runner.action_mask()
            action = runner.app_heuristic_action_indices(mask)
            slot, skill = int(action[0]), int(action[1])
            valid = bool(mask[slot, skill])
            hla = (runner.action_from_indices(slot, skill)
                   if valid and slot < runner.max_objects else None)
            _reward, info = runner.step(hla)
            if info.get("executed"):
                executed += 1
                if verbose:
                    print(f"[sim-ref] executed {executed}/{n_steps} "
                          f"t={runner.clock.time():.1f}s skill={info.get('executed_skill')} "
                          f"ok={info.get('execution_success')}", flush=True)
    finally:
        if rec is not None:
            rec.close(analyze=False)
        runner.close()
    return Path(out_dir)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="gp8_s2r_sim_reference")
    parser.add_argument("out_dir")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--belt-speed", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source", default="sim_reference")
    args = parser.parse_args(argv)
    run_sim_reference(args.out_dir, n_steps=args.steps, belt_speed=args.belt_speed,
                      seed=args.seed, source=args.source)
    return 0


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    raise SystemExit(main())

"""Compare commanded conveyor speed against measured belt/object speeds.

Run with the repo parent on PYTHONPATH:

    python -m gp8_control.tests.sim_belt_speed_probe

Use ``--physical`` to test only the experimental physical-belt mode.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

from gp8_control.backends.mujoco_sim import SimConfig, SimCore


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def _run_mode(*, physical: bool, seconds: float, speed: float, spawn_interval: float) -> dict:
    cfg = SimConfig(
        belt_speed=speed,
        spawn_interval=spawn_interval,
        physical_belt=physical,
    )
    core = SimCore(cfg)
    core.start()
    try:
        if not core.wait_ready(10.0):
            raise RuntimeError("sim stepper did not become ready")
        sample_period = 0.05
        sim_start = core.sim_time()
        wall_start = time.perf_counter()
        belt_speeds: list[float] = []
        object_speeds: list[float] = []
        object_base_z: list[float] = []
        object_counts: list[int] = []
        snap = core.speed_probe_snapshot()
        while core.sim_time() - sim_start < seconds:
            snap = core.speed_probe_snapshot()
            belt_speeds.append(float(snap["belt_downstream_speed"]))
            on_belt_objects = [o for o in snap["objects"] if o["state"] == core._ON_BELT]
            on_belt = [float(o["downstream_speed"]) for o in on_belt_objects]
            object_speeds.extend(on_belt)
            object_base_z.extend(float(o["base_z"]) for o in on_belt_objects)
            object_counts.append(len(on_belt))
            if cfg.realtime:
                time.sleep(sample_period)
            else:
                core.advance_seconds(sample_period)
        wall_elapsed = time.perf_counter() - wall_start
        sim_elapsed = core.sim_time() - sim_start
        return {
            "physical": physical,
            "realtime": bool(cfg.realtime),
            "commanded": speed,
            "belt_mean": _mean(belt_speeds),
            "object_mean": _mean(object_speeds),
            "belt_top_base_z": float(snap["belt_top_base_z"]),
            "object_base_z_mean": _mean(object_base_z),
            "object_samples": len(object_speeds),
            "max_objects": max(object_counts) if object_counts else 0,
            "sim_elapsed": sim_elapsed,
            "wall_elapsed": wall_elapsed,
            "realtime_factor": sim_elapsed / wall_elapsed if wall_elapsed > 0.0 else float("inf"),
        }
    finally:
        core.stop()


def _print_result(r: dict) -> None:
    mode = "physical" if r["physical"] else "kinematic"
    clock_mode = "realtime" if r["realtime"] else "fast"
    print(
        f"{mode:9s} {clock_mode:8s} commanded={r['commanded']:.4f} "
        f"belt_mean={r['belt_mean']:.4f} "
        f"object_mean={r['object_mean']:.4f} "
        f"belt_top_base_z={r['belt_top_base_z']:.4f} "
        f"object_base_z_mean={r['object_base_z_mean']:.4f} "
        f"object_samples={r['object_samples']} max_objects={r['max_objects']} "
        f"sim_elapsed={r['sim_elapsed']:.3f} wall_elapsed={r['wall_elapsed']:.3f} "
        f"factor={r['realtime_factor']:.2f}x"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="sim_belt_speed_probe")
    parser.add_argument("--physical", action="store_true", help="test only physical-belt mode")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--speed", type=float, default=0.12)
    parser.add_argument("--spawn-interval", type=float, default=1.0)
    args = parser.parse_args(argv)

    modes = [True] if args.physical else [False, True]
    for physical in modes:
        _print_result(_run_mode(
            physical=physical,
            seconds=args.seconds,
            speed=args.speed,
            spawn_interval=args.spawn_interval,
        ))
    return 0


if __name__ == "__main__":
    sys.exit(main())

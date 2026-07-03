"""Terminal verification of the throw-only chain.

Runs the MuJoCo env headlessly with the default demo policy under throw-only
conditions, prints each grasp/throw event in real time, and ends with a
clean summary you can compare against PyBullet's `_ours_fifo` numbers.

Usage:
    conda activate iitp
    python verify_throw_chain.py
    python verify_throw_chain.py --steps 200 --spawn 1.2 --speed 0.4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from recycling_env.gym_env import RecyclingBBoxGymEnv


GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BLUE = "\033[34m"
GRAY = "\033[90m"
BOLD = "\033[1m"
RESET = "\033[0m"


def colorize(text: str, color: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{color}{text}{RESET}"


def make_env(args):
    env = RecyclingBBoxGymEnv(
        use_camera=False,
        render_mode=None,
        pushing_enabled=False,
        spawn_enabled=True,
        spawn_rate_hz=args.spawn,
        belt_speed=args.speed,
        frame_skip=10,
        max_steps=10_000,
        step_sleep=0.0,
    )

    captures: list[dict] = []
    original_hide = env._hide_object_slot

    def hide_with_capture(slot):
        if slot.spawn_id is not None and slot.active:
            pos, _ = env.env.get_freejoint_pose(slot.freejoint_name)
            # Success = object ACTUALLY inside the PHYSICAL bin (decoupled from the
            # aim target). Capture the bin center (world) + interior half-extents.
            bin_xy = env._bin_center_world_xy_for_slot(slot)
            bin_half = env._bin_interior_half
            captures.append({
                "spawn_id": int(slot.spawn_id),
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z": float(pos[2]),
                "bin_x": float(bin_xy[0]),
                "bin_y": float(bin_xy[1]),
                "bin_hx": float(bin_half[0]),
                "bin_hy": float(bin_half[1]),
                "bin_z_lo": float(env._bin_z_lo),
                "bin_z_hi": float(env._bin_z_hi),
                "manipulated": bool(slot.manipulated),
                "blocked": bool(slot.blocked),
            })
        original_hide(slot)

    env._hide_object_slot = hide_with_capture
    return env, captures


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=200, help="number of gym steps")
    ap.add_argument("--spawn", type=float, default=1.2, help="spawn_rate_hz")
    ap.add_argument("--speed", type=float, default=0.4, help="belt_speed (m/s)")
    ap.add_argument("--quiet", action="store_true", help="suppress per-step output")
    args = ap.parse_args()

    env, captures = make_env(args)
    obs, info = env.reset()

    # Under the 1-step-ahead action design, each throw step grasps + throws in a
    # single step (the grasp is folded in; there is no separate no-motion grasp
    # step). A throw is "chained" when the object it throws is the one the
    # PREVIOUS throw routed its after-trajectory toward (so no route-only
    # re-approach happened in between). A throw is "bootstrapped" when a
    # route-only `initial_approach` (bang-bang) step had to grasp it first.
    counts = {
        "throws_chained": 0,       # threw the obj the prev throw routed toward
        "throws_bootstrapped": 0,  # preceded by a route-only bang-bang grasp
        "route_grasps": 0,         # route-only initial_approach (bang-bang)
        "attach_failed": 0,        # ready_grasp hold-attach timed out
        "skip": 0,
        "throws": 0,
        "other": 0,
    }
    # Chain tracking across steps.
    prev_throw_next_id = None       # next_spawn_id the last throw routed toward
    route_since_last_throw = True   # a route-only step happened before this throw

    print(colorize("=" * 78, BOLD))
    print(colorize(
        f"  Throw-only chain verification  "
        f"(spawn={args.spawn}/s  belt={args.speed} m/s  steps={args.steps})",
        BOLD,
    ))
    print(colorize("=" * 78, BOLD))
    print(f"  {'step':<5}  {'event':<22}  {'detail':<32}  {'cumulative':<12}")
    print("-" * 78)

    t0 = time.time()
    last_print = t0
    for step_i in range(args.steps):
        action = env._sample_demo_action(push_probability=0.0, alternate=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ei = info.get("execution_info", {})
        reason = ei.get("reason", "?")
        handoff = bool(ei.get("handoff_reuse", False))

        event = ""
        detail = ""
        color = GRAY
        if reason == "skip":
            counts["skip"] += 1
            event = "skip"
            detail = ""
        elif reason == "executed":
            counts["throws"] += 1
            current_id = ei.get("current_spawn_id")
            chained = (not route_since_last_throw) and prev_throw_next_id is not None and current_id == prev_throw_next_id
            if chained:
                counts["throws_chained"] += 1
                event = "throw [chain]"
                color = GREEN
            else:
                counts["throws_bootstrapped"] += 1
                event = "throw [from route]"
                color = BLUE
            detail = (f"current={current_id} -> "
                      f"next={ei.get('next_spawn_id')}  "
                      f"T={ei.get('trajectory_duration', 0):.2f}s")
            prev_throw_next_id = ei.get("next_spawn_id")
            route_since_last_throw = False
        elif reason == "initial_approach_executed":
            counts["route_grasps"] += 1
            route_since_last_throw = True
            event = "route [bang-bang grasp]"
            color = YELLOW
            detail = f"obj={ei.get('current_spawn_id')}  route-only approach"
        elif reason in ("push_handoff_routed",):
            route_since_last_throw = True
            event = "route [push handoff]"
            color = YELLOW
            detail = f"obj={ei.get('current_spawn_id')}"
        elif reason == "ready_grasp_attach_failed":
            counts["attach_failed"] += 1
            event = "grasp [attach failed]"
            color = RED
            detail = "held pose, attach timed out (no motion)"
        else:
            counts["other"] += 1
            counts.setdefault("_other_reasons", {})
            counts["_other_reasons"][reason] = counts["_other_reasons"].get(reason, 0) + 1
            event = reason
            color = RED

        cum = (
            f"chained={counts['throws_chained']} "
            f"bootstrap={counts['throws_bootstrapped']} "
            f"route={counts['route_grasps']}"
        )
        if not args.quiet and reason != "skip":
            print(f"  {step_i:<5}  "
                  f"{colorize(event, color):<32}  "
                  f"{detail:<32}  {cum}")

        # Throttled progress for quiet mode
        if args.quiet and time.time() - last_print > 1.0:
            last_print = time.time()
            print(f"  step {step_i}/{args.steps}  {cum}", end="\r", flush=True)

        if truncated:
            break

    elapsed = time.time() - t0
    print()
    print(colorize("=" * 78, BOLD))
    print(colorize("  RESULTS", BOLD))
    print(colorize("=" * 78, BOLD))

    total_throws = counts["throws"]

    # Bin-volume metric. Bin mocap z = 0.05 per gym_env.py target_bin_spawn_z.
    # Floor top = mocap_z - 0.02 = 0.03. Wall top = mocap_z + 0.48 = 0.53.
    # +x wall is intentionally taller (top z=0.63) to catch push overshoots; metric
    # Success = object ACTUALLY inside the PHYSICAL bin volume (2026-06-12),
    # centered on the bin (decoupled from the aim target), using the bin's interior
    # half-extents (between the inner wall faces) and interior z-range -- all read
    # from the bin geoms in the env. No velocity check -- bin is walled.
    in_bin = sum(
        1 for c in captures
        if c["manipulated"]
        and abs(c["x"] - c["bin_x"]) <= c["bin_hx"]
        and abs(c["y"] - c["bin_y"]) <= c["bin_hy"]
        and c["bin_z_lo"] <= c["z"] <= c["bin_z_hi"]
    )
    off_bin = sum(1 for c in captures if c["manipulated"]) - in_bin

    print(f"  steps run:                {step_i + 1}  ({elapsed:.1f}s wall)")
    print(f"  throws executed:          {counts['throws']}")
    print(f"    chained (grasp folded): {colorize(str(counts['throws_chained']), GREEN)}")
    print(f"    bootstrapped (from route): {colorize(str(counts['throws_bootstrapped']), BLUE)}")
    print(f"  route-only grasps:        {counts['route_grasps']}")
    print(f"  attach failed:            {counts['attach_failed']}")
    print(f"  skips:                    {counts['skip']}")
    print(f"  other:                    {counts['other']}")
    if counts.get("_other_reasons"):
        for reason, count in sorted(counts["_other_reasons"].items(), key=lambda x: -x[1]):
            print(f"    - {reason}: {count}")
    if total_throws > 0:
        chain_pct = counts["throws_chained"] / total_throws * 100
        print()
        print(f"  chain rate:               {colorize(f'{chain_pct:5.1f}%', GREEN)}  "
              f"(fraction of throws that chained without a route-only re-approach)")
    print()
    print(f"  thrown objects landed:    {in_bin + off_bin}")
    print(f"    in target bin:          {colorize(str(in_bin), GREEN)}")
    print(f"    off target bin:         {off_bin}")
    if (in_bin + off_bin) > 0:
        print(f"  in-bin accuracy:          "
              f"{colorize(f'{in_bin/(in_bin+off_bin)*100:5.1f}%', GREEN)}")
    print()

    # Pass/fail (computed on the steady-state portion — skip the first
    # ~5 throws where the chain is still warming up).
    warmup = 5
    steady_chain = max(0, counts["throws_chained"] - warmup) if total_throws > warmup else counts["throws_chained"]
    steady_total = max(1, total_throws - warmup) if total_throws > warmup else max(1, total_throws)
    fails = []
    # Chain rate is an efficiency indicator under the 1-step-ahead design (it
    # measures how often a feasible chain target existed AND the policy chained
    # to it without a route-only re-approach). It is NOT directly comparable to
    # the pre-refactor "no-motion grasp" metric, which the old env's FIFO
    # fallback + alternate scan inflated. The 65% floor catches a real chaining
    # collapse; in-bin accuracy below is the hard correctness gate.
    if total_throws > warmup:
        if steady_chain / steady_total < 0.65:
            fails.append(
                f"steady-state chain rate {steady_chain/steady_total:.1%} < 65% "
                f"(after first {warmup} warm-up throws)"
            )
    # In-bin accuracy is the primary correctness gate: every throw that fires
    # must land in the bin.
    if (in_bin + off_bin) > 0 and in_bin / (in_bin + off_bin) < 0.97:
        fails.append(f"in-bin accuracy {in_bin/(in_bin+off_bin):.1%} < 97%")
    # Attach failures cost only a wasted re-grasp step (the next step routes
    # again), not correctness. Tolerate normal grasp-timing variance (~2-3%);
    # flag only a real grasp collapse.
    if counts["attach_failed"] > 0.04 * max(total_throws, 1):
        fails.append(
            f"too many attach failures ({counts['attach_failed']} / {total_throws} throws)"
        )
    if total_throws > warmup:
        print(f"  steady-state chain rate:  "
              f"{colorize(f'{steady_chain/steady_total*100:5.1f}%', GREEN)}  "
              f"(skipping first {warmup} warm-up throws)")

    if not fails:
        print(colorize("  PASS  the throw-only chain works as expected.", GREEN + BOLD))
    else:
        print(colorize("  ISSUES:", YELLOW + BOLD))
        for f in fails:
            print(colorize(f"    - {f}", YELLOW))

    env.close()


if __name__ == "__main__":
    main()

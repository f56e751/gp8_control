"""GUI verification of the throw-only chain.

Opens the MuJoCo passive viewer and runs the throw-only demo policy so you
can watch the grasp->throw->grasp->throw chain. No OpenCV / camera overlay
needed - the MuJoCo viewer is enough.

Usage:
    conda activate iitp
    python verify_throw_chain_gui.py
    python verify_throw_chain_gui.py --spawn 0.8 --speed 0.2 --realtime 1.0
    python verify_throw_chain_gui.py --policy smart           # alt policy
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
import mujoco.viewer

from recycling_env.gym_env import RecyclingBBoxGymEnv
from recycling_env.gui_utils import make_throttled_viewer_hook


GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
GRAY = "\033[90m"
RESET = "\033[0m"


def colorize(text, color):
    if not sys.stdout.isatty():
        return text
    return f"{color}{text}{RESET}"


def smart_policy(env):
    ordered = env._ordered_active_slots()
    if not ordered:
        return np.array([env.max_detections, env.ACTION_THROW])
    # 1-step-ahead semantics: the action specifies the NEXT throw target. Always
    # chain to the first action-addressable (FIFO) slot; the env executes the
    # pending manipulation and routes the EE here.
    return np.array([0, env.ACTION_THROW])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spawn", type=float, default=1.2, help="spawn_rate_hz")
    ap.add_argument("--speed", type=float, default=0.4, help="belt_speed (m/s)")
    ap.add_argument("--policy", choices=["demo", "smart"], default="demo",
                    help="demo = env._sample_demo_action; smart = always target ready_grasp_slot")
    ap.add_argument("--realtime", type=float, default=1.0,
                    help="real-time multiplier (1.0 = real time, 0.5 = half speed for easier viewing)")
    ap.add_argument("--max-steps", type=int, default=10_000, help="bail out after this many gym steps")
    ap.add_argument("--camera", action="store_true",
                    help="enable D435i camera view with bbox detection overlay (needs opencv-python)")
    ap.add_argument("--pushing", action="store_true",
                    help="enable pushing primitive (exercises throw->push and push->{throw,push} chains)")
    ap.add_argument("--debug-chain", action="store_true",
                    help="emit '[chain] ...' log lines for each chain transition/clear event")
    args = ap.parse_args()

    cv2_module = None
    if args.camera:
        try:
            import cv2 as cv2_module
        except ImportError:
            print("warning: opencv-python is not installed; --camera disabled.")
            cv2_module = None

    env = RecyclingBBoxGymEnv(
        use_camera=bool(args.camera and cv2_module is not None),
        render_mode=None,
        pushing_enabled=bool(args.pushing),
        spawn_enabled=True,
        spawn_rate_hz=args.spawn,
        belt_speed=args.speed,
        frame_skip=10,
        max_steps=args.max_steps,
        step_sleep=0.0,
        debug_chain_log=bool(args.debug_chain),
    )

    obs, info = env.reset()

    print(colorize("=" * 70, GRAY))
    print(f"  GUI throw-only verification  "
          f"spawn={args.spawn}/s  belt={args.speed} m/s  policy={args.policy}")
    print(f"  Close the viewer window to stop.  realtime x{args.realtime}")
    print(colorize("=" * 70, GRAY))

    chain_ok = chain_bb = attach_failed = throws = 0

    push_prob = 0.5 if args.pushing else 0.0
    alternate = not args.pushing
    pick_action = (smart_policy if args.policy == "smart"
                   else (lambda env: env._sample_demo_action(push_probability=push_prob, alternate=alternate)))

    sim_dt = env.env.dt * env.frame_skip   # seconds of sim per gym step
    target_realtime_dt = sim_dt / max(args.realtime, 1e-3)

    camera_window = "D435i camera (q to quit)"
    with mujoco.viewer.launch_passive(env.env.model, env.env.data) as viewer:
        env.set_sim_step_hook(make_throttled_viewer_hook(
            viewer, env,
            cv2_module=cv2_module,
            camera_window=camera_window,
        ))

        try:
            for step_i in range(args.max_steps):
                if not viewer.is_running():
                    break
                t_start = time.time()
                action = pick_action(env)
                obs, reward, terminated, truncated, info = env.step(action)
                ei = info.get("execution_info", {})
                reason = ei.get("reason", "?")
                handoff = bool(ei.get("handoff_reuse", False))

                if reason == "executed":
                    throws += 1
                    print(f"  step {step_i:4d}  "
                          f"{colorize('throw', BLUE):<14}  "
                          f"{ei.get('current_spawn_id')} -> {ei.get('next_spawn_id')}  "
                          f"T={ei.get('trajectory_duration', 0):.2f}s")
                elif reason == "initial_approach_executed":
                    if handoff:
                        chain_ok += 1
                        print(f"  step {step_i:4d}  "
                              f"{colorize('grasp [chain]', GREEN):<22}  "
                              f"obj={ei.get('current_spawn_id')}  "
                              f"(no robot motion)  "
                              f"[chain={chain_ok} bb={chain_bb}]")
                    else:
                        chain_bb += 1
                        print(f"  step {step_i:4d}  "
                              f"{colorize('grasp [bang-bang]', YELLOW):<22}  "
                              f"obj={ei.get('current_spawn_id')}  "
                              f"(chain broken; initial approach)  "
                              f"[chain={chain_ok} bb={chain_bb}]")
                elif reason == "ready_grasp_attach_failed":
                    attach_failed += 1
                    print(f"  step {step_i:4d}  "
                          f"{colorize('grasp [attach failed]', YELLOW):<22}  "
                          f"(no motion - hold timed out)")
                elif reason == "skip":
                    pass  # quiet
                else:
                    print(f"  step {step_i:4d}  {reason}")

                # Throttle wall-clock to target real time so the viewer is watchable.
                elapsed = time.time() - t_start
                slack = target_realtime_dt - elapsed
                if slack > 0:
                    time.sleep(slack)

                if truncated:
                    print("  (max_steps reached - resetting)")
                    obs, info = env.reset()
        finally:
            env.set_sim_step_hook(None)
            if cv2_module is not None:
                try:
                    cv2_module.destroyAllWindows()
                except Exception:
                    pass

    print()
    grasps = chain_ok + chain_bb + attach_failed
    print(colorize("=" * 70, GRAY))
    print(f"  throws: {throws}   grasps: {grasps}   "
          f"chain_OK: {colorize(str(chain_ok), GREEN)}   "
          f"bang-bang: {colorize(str(chain_bb), YELLOW)}   "
          f"attach_failed: {attach_failed}")
    if grasps > 0:
        print(f"  chain rate: {colorize(f'{chain_ok/grasps*100:.1f}%', GREEN)}")
    print(colorize("=" * 70, GRAY))
    env.close()


if __name__ == "__main__":
    main()

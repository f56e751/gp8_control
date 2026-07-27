#!/usr/bin/env python3
"""Move via a raised TCP Z, enable suction at target, and hold until release.

The robot first reaches ``--via-z`` (default base Z=0.200 m) at the current
TCP X/Y, then moves to the requested target.  The current TCP orientation is
preserved throughout.  Optional --x/--y override the final planar target.  This
is a supervised real-robot debug command; Enter or Ctrl-C always requests
suction OFF.
"""

from __future__ import annotations

import argparse
import math
import threading
import time

import numpy as np

from gp8_control.robots.gp8 import GP8
from gp8_control.tests.suction_lift_debug import JOINT_ACCEL_RATIO
from gp8_control.trajectory.trajectory_primitive import trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--z", type=float, required=True,
        help="target TCP Z in base frame [m] (required)",
    )
    parser.add_argument(
        "--x", type=float, default=None,
        help="target TCP X [m] (default: current TCP X)",
    )
    parser.add_argument(
        "--y", type=float, default=None,
        help="target TCP Y [m] (default: current TCP Y)",
    )
    parser.add_argument(
        "--via-z", type=float, default=0.200,
        help="intermediate TCP Z in base frame [m] (default 0.200)",
    )
    parser.add_argument(
        "--vel-scale", type=float, default=0.30,
        help="joint velocity scale for the positioning move (default 0.30)",
    )
    parser.add_argument(
        "--plan-only", action="store_true",
        help="connect and solve the target from current state, but do not move or toggle suction",
    )
    args, _ros_unknown = parser.parse_known_args()

    for name in ("z", "x", "y", "via_z"):
        value = getattr(args, name)
        if value is not None and not math.isfinite(value):
            parser.error(f"--{name.replace('_', '-')} must be finite")
    if not 0.0 < args.vel_scale <= 1.0:
        parser.error("--vel-scale must be in (0, 1]")

    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node

    from gp8_control.controllers.trajectory_controller import TrajectoryController

    rclpy.init()
    node = Node("suction_hold_at_z")
    ctrl = TrajectoryController(node)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = None
    suction_io_used = False

    try:
        print("Waiting for servers...")
        if not ctrl.wait_for_servers(timeout_sec=10.0):
            print("  서버 없음 — ~/ros2_ws/debug_bringup.sh 실행 상태를 확인하세요.")
            return

        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()
        deadline = time.time() + 5.0
        while ctrl.current_joints is None and time.time() < deadline:
            time.sleep(0.05)
        if ctrl.current_joints is None:
            print("  joint state 미수신 — joint_state_broadcaster를 확인하세요.")
            return

        gp8 = GP8()
        current_q = np.asarray(ctrl.current_joints, dtype=float)
        current_T = gp8.forward_kinematics(current_q)
        via_T = current_T.copy()
        via_T[2, 3] = float(args.via_z)
        target_T = current_T.copy()
        target_T[0, 3] = current_T[0, 3] if args.x is None else float(args.x)
        target_T[1, 3] = current_T[1, 3] if args.y is None else float(args.y)
        target_T[2, 3] = float(args.z)

        via_q = gp8.inverse_kinematics(via_T, q_init=current_q)
        if via_q is None:
            parser.error("via IK failed; change --via-z")
        via_q = np.asarray(via_q, dtype=float)
        target_q = gp8.inverse_kinematics(target_T, q_init=via_q)
        if target_q is None:
            parser.error(
                "target IK failed; change --z or provide reachable --x/--y"
            )
        target_q = np.asarray(target_q, dtype=float)
        solved_via_T = gp8.forward_kinematics(via_q)
        solved_T = gp8.forward_kinematics(target_q)

        print("\n=== Suction Hold-at-Z Plan ===")
        print(
            "  current TCP : "
            f"({current_T[0,3]:+.4f}, {current_T[1,3]:+.4f}, {current_T[2,3]:+.4f}) m"
        )
        print(
            "  via TCP     : "
            f"({solved_via_T[0,3]:+.4f}, {solved_via_T[1,3]:+.4f}, "
            f"{solved_via_T[2,3]:+.4f}) m"
        )
        print(
            "  target TCP  : "
            f"({solved_T[0,3]:+.4f}, {solved_T[1,3]:+.4f}, {solved_T[2,3]:+.4f}) m"
        )
        print("  orientation : 현재 TCP 자세 유지")
        print(f"  velocity    : joint limit의 {args.vel_scale*100:.1f}%")
        if args.plan_only:
            print("\nplan-only OK — 로봇 이동과 석션 명령을 보내지 않았습니다.")
            return

        print("\n⚠️  실제 로봇이 움직이고 목표 도착 후 석션이 자동으로 켜집니다.")
        print("    이동 경로와 추가 장착판/주변 물체의 충돌 가능성을 확인하세요.")
        if input("실행? (y/N) > ").strip().lower() != "y":
            print("취소.")
            return

        # Start from a known-safe output state, rise to the waypoint, and only
        # then approach the requested suction target.
        suction_io_used = True
        ctrl.suction_off()
        limits = np.asarray(gp8.velocity_limits, dtype=float) * args.vel_scale
        accelerations = limits * JOINT_ACCEL_RATIO
        zero = np.zeros(6, dtype=float)

        def move_segment(q_start: np.ndarray, q_end: np.ndarray, label: str) -> None:
            if np.max(np.abs(q_end - q_start)) <= 1e-6:
                print(f"→ {label}: 이미 해당 위치입니다.")
                return
            q_traj, q_vel, ts = trajectory(
                q_start, zero, q_end, zero,
                limits, accelerations, hertz=100.0,
            )
            print(f"→ {label} ({ts[-1]:.2f}s)...")
            ctrl.send_trajectory_queue(
                q_traj, q_vel, ts, final_joint=q_end,
            )

        move_segment(current_q, via_q, f"경유 Z={args.via_z:.4f}m로 이동")
        move_segment(via_q, target_q, f"목표 Z={args.z:.4f}m로 이동")

        ctrl.suction_on()
        print("→ 석션 ON. 현재 자세에서 유지합니다.")
        input("끄려면 Enter (Ctrl-C도 OFF 후 종료) > ")
        print("→ 석션 OFF.")
    except (KeyboardInterrupt, EOFError):
        print("\n입력 중단 — 석션 OFF 후 종료합니다.")
    finally:
        try:
            if suction_io_used:
                ctrl.suction_off()
            ctrl.close()  # drain the final OFF before ROS/node teardown
        except Exception:
            pass
        try:
            executor.shutdown()
        except Exception:
            pass
        if spin_thread is not None:
            spin_thread.join(timeout=2.0)
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

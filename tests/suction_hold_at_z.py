#!/usr/bin/env python3
"""Move to a requested TCP Z, enable suction, and hold until operator release.

By default only world/base Z changes: the current TCP X/Y and orientation are
preserved.  Optional --x/--y override the planar target.  This is a supervised
real-robot debug command; Enter or Ctrl-C always requests suction OFF.
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
        "--vel-scale", type=float, default=0.10,
        help="joint velocity scale for the positioning move (default 0.10)",
    )
    parser.add_argument(
        "--plan-only", action="store_true",
        help="connect and solve the target from current state, but do not move or toggle suction",
    )
    args, _ros_unknown = parser.parse_known_args()

    for name in ("z", "x", "y"):
        value = getattr(args, name)
        if value is not None and not math.isfinite(value):
            parser.error(f"--{name} must be finite")
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
        target_T = current_T.copy()
        target_T[0, 3] = current_T[0, 3] if args.x is None else float(args.x)
        target_T[1, 3] = current_T[1, 3] if args.y is None else float(args.y)
        target_T[2, 3] = float(args.z)

        target_q = gp8.inverse_kinematics(target_T, q_init=current_q)
        if target_q is None:
            parser.error(
                "target IK failed; change --z or provide reachable --x/--y"
            )
        target_q = np.asarray(target_q, dtype=float)
        solved_T = gp8.forward_kinematics(target_q)

        print("\n=== Suction Hold-at-Z Plan ===")
        print(
            "  current TCP : "
            f"({current_T[0,3]:+.4f}, {current_T[1,3]:+.4f}, {current_T[2,3]:+.4f}) m"
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

        # Start from a known-safe output state, then move at a low joint-speed scale.
        suction_io_used = True
        ctrl.suction_off()
        if np.max(np.abs(target_q - current_q)) > 1e-6:
            limits = np.asarray(gp8.velocity_limits, dtype=float) * args.vel_scale
            accelerations = limits * JOINT_ACCEL_RATIO
            zero = np.zeros(6, dtype=float)
            q_traj, q_vel, ts = trajectory(
                current_q, zero, target_q, zero,
                limits, accelerations, hertz=100.0,
            )
            print(f"→ 목표 위치로 이동 ({ts[-1]:.2f}s)...")
            ctrl.send_trajectory_queue(
                q_traj, q_vel, ts, final_joint=target_q,
            )
        else:
            print("→ 이미 목표 위치입니다.")

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

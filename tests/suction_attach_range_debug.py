#!/usr/bin/env python3
"""Supervised suction attachment-range measurement on the real GP8.

The robot parks above a calibrated contact pose, enables suction, and descends
slowly.  Press Space/Enter/A the moment the object visibly attaches.  The stream
stops and the script converts the joint feedback captured at that keypress to the
``suction_tool`` TCP position.  No throw is generated.

There is currently no vacuum-pressure/attachment input in gp8_control, so the
attachment trigger is manual.  Keep the descent speed low; at the default 5 mm/s,
a 0.2 s human reaction time corresponds to about 1 mm of distance uncertainty.

Run only with the conveyor stopped and ``debug_bringup.sh`` active.
"""

from __future__ import annotations

import argparse
import math
import os
import select
import sys
import threading
import time

import numpy as np

from gp8_control.config import Config
from gp8_control.robots.gp8 import GP8
from gp8_control.tests.suction_lift_debug import (
    JOINT_ACCEL_RATIO,
    TOOL_OFFSET_DEFAULT,
    _ik_tool_down,
)
from gp8_control.trajectory.trajectory_primitive import trajectory


def _tool_point(gp8: GP8, q, tool_offset: float) -> np.ndarray:
    T = gp8.forward_kinematics(np.asarray(q, dtype=float))
    return T[:3, 3] + T[:3, 0] * float(tool_offset)


def _build_descent(
    gp8: GP8,
    x: float,
    y: float,
    start_z: float,
    end_z: float,
    max_speed: float,
    sample_rate: float,
    tool_offset: float,
):
    """Build a zero-end-velocity quintic Cartesian-Z descent."""
    distance = float(start_z - end_z)
    if distance <= 0.0:
        raise ValueError("end Z must be below start Z")
    if max_speed <= 0.0:
        raise ValueError("descent speed must be positive")
    if sample_rate < 20.0:
        raise ValueError("sample rate must be at least 20 Hz")

    # h(u)=10u^3-15u^4+6u^5 has max h'(u)=1.875.  Choose T so the
    # Cartesian Z speed never exceeds max_speed.
    duration = 1.875 * distance / float(max_speed)
    n = max(3, int(math.ceil(duration * sample_rate)) + 1)
    ts = np.linspace(0.0, duration, n)
    u = ts / duration
    h = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    zs = start_z - distance * h

    q_cols = []
    q_seed = None
    for z in zs:
        q = _ik_tool_down(
            gp8, x, y, float(z), q_init=q_seed, tool_offset=tool_offset,
        )
        if q is None:
            raise ValueError(
                f"descent IK failed at ({x:+.3f}, {y:+.3f}, {z:+.3f})"
            )
        q_cols.append(np.asarray(q, dtype=float))
        q_seed = q
    q_traj = np.column_stack(q_cols)
    q_vel = np.gradient(q_traj, ts, axis=1, edge_order=2)
    q_vel[:, 0] = 0.0
    q_vel[:, -1] = 0.0

    tcp = np.vstack([_tool_point(gp8, q_traj[:, i], tool_offset) for i in range(n)])
    tcp_speed = np.linalg.norm(np.gradient(tcp, ts, axis=0, edge_order=2), axis=1)
    return q_traj, q_vel, ts, tcp, float(np.max(tcp_speed))


def _measurement_listener(ctrl, trigger: threading.Event, done: threading.Event, result: dict):
    """Capture one supervised keypress without leaving the terminal in raw mode."""
    if not sys.stdin.isatty():
        result["error"] = "stdin is not a TTY"
        trigger.set()
        return

    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while not done.is_set() and not trigger.is_set():
            ready, _, _ = select.select([fd], [], [], 0.05)
            if not ready:
                continue
            ch = os.read(fd, 1).decode("utf-8", errors="ignore").lower()
            if ch in (" ", "\r", "\n", "a"):
                result["kind"] = "attach"
            elif ch in ("q", "\x03"):
                result["kind"] = "abort"
            else:
                continue
            result["wall_time"] = time.time()
            result["q"] = (
                list(ctrl.current_joints)
                if ctrl.current_joints is not None else None
            )
            trigger.set()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _print_plan(args, contact_z: float, start_z: float, end_z: float,
                ts, tcp, max_tcp_speed: float) -> None:
    print("\n=== Suction Attachment Range Plan ===")
    print(f"  XY             : ({args.x:+.3f}, {args.y:+.3f}) m")
    print(f"  contact TCP Z  : {contact_z:+.3f} m (clearance 0 기준)")
    print(f"  start TCP Z    : {start_z:+.3f} m ({(start_z-contact_z)*1000:+.1f} mm)")
    print(f"  end TCP Z      : {end_z:+.3f} m ({(end_z-contact_z)*1000:+.1f} mm)")
    print(f"  descent        : {(start_z-end_z)*1000:.1f} mm / {ts[-1]:.2f} s")
    print(f"  max TCP speed  : {max_tcp_speed*1000:.2f} mm/s")
    print(f"  path XY drift  : {np.max(np.linalg.norm(tcp[:, :2]-tcp[0, :2], axis=1))*1000:.3f} mm")
    print(f"  reaction error : 약 ±{args.descent_speed*args.reaction_time*1000:.1f} mm "
          f"(@ {args.reaction_time:.2f}s 가정)")
    if end_z < contact_z:
        print(
            f"  ⚠️ WARNING      : end Z가 보정 접촉면보다 "
            f"{(contact_z-end_z)*1000:.1f} mm 낮습니다. 충돌 가능성을 확인하세요."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--x", type=float, default=0.55, help="measurement X [m]")
    parser.add_argument("--y", type=float, default=0.0, help="measurement Y [m]")
    parser.add_argument(
        "--contact-z", type=float, default=None,
        help="calibrated suction_tool contact Z [m] (default Config.GRASP_Z)",
    )
    parser.add_argument(
        "--start-clearance", type=float, default=0.030,
        help="start height above contact Z [m] (default 0.030)",
    )
    parser.add_argument(
        "--max-descent", type=float, default=None,
        help="maximum descent [m] (default: start-clearance, never below contact)",
    )
    parser.add_argument(
        "--end-z", "--final-z", dest="end_z", type=float, default=None,
        help="final suction_tool TCP Z [m] (default: contact-z); "
             "mutually exclusive with --max-descent",
    )
    parser.add_argument(
        "--allow-below-contact", action="store_true",
        help="deprecated compatibility flag; end-z below contact is now allowed",
    )
    parser.add_argument(
        "--descent-speed", type=float, default=0.005,
        help="maximum Cartesian descent speed [m/s] (default 0.005)",
    )
    parser.add_argument(
        "--prime-time", type=float, default=0.30,
        help="suction-on hold before descent [s] (default 0.30)",
    )
    parser.add_argument(
        "--move-vel-scale", type=float, default=0.15,
        help="joint velocity scale for moving to start pose (default 0.15)",
    )
    parser.add_argument("--sample-rate", type=float, default=50.0)
    parser.add_argument("--reaction-time", type=float, default=0.20)
    parser.add_argument("--tool-offset", type=float, default=TOOL_OFFSET_DEFAULT)
    parser.add_argument("--plan-only", action="store_true")
    args, _ros_unknown = parser.parse_known_args()

    cfg = Config()
    contact_z = cfg.GRASP_Z if args.contact_z is None else float(args.contact_z)
    if args.start_clearance <= 0.0:
        parser.error("--start-clearance must be positive")
    if args.end_z is not None and args.max_descent is not None:
        parser.error("use either --end-z or --max-descent, not both")
    start_z = contact_z + float(args.start_clearance)
    if args.end_z is not None:
        end_z = float(args.end_z)
        if end_z >= start_z:
            parser.error("--end-z must be below the start Z")
        max_descent = start_z - end_z
    else:
        max_descent = (
            float(args.start_clearance)
            if args.max_descent is None else float(args.max_descent)
        )
        if max_descent <= 0.0:
            parser.error("--max-descent must be positive")
        end_z = start_z - max_descent
    if args.prime_time < 0.0:
        parser.error("--prime-time must be non-negative")
    if not 0.0 < args.move_vel_scale <= 1.0:
        parser.error("--move-vel-scale must be in (0, 1]")

    gp8 = GP8()
    try:
        q_traj, q_vel, ts, tcp, max_tcp_speed = _build_descent(
            gp8, args.x, args.y, start_z, end_z,
            args.descent_speed, args.sample_rate, args.tool_offset,
        )
    except ValueError as exc:
        parser.error(str(exc))
    _print_plan(args, contact_z, start_z, end_z, ts, tcp, max_tcp_speed)
    if args.plan_only:
        print("\nplan-only OK — robot and suction were not commanded.")
        return

    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from gp8_control.controllers.trajectory_controller import TrajectoryController

    rclpy.init()
    node = Node("suction_attach_range_debug")
    ctrl = TrajectoryController(node)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = None
    try:
        print("\nWaiting for servers...")
        if not ctrl.wait_for_servers(timeout_sec=10.0):
            print("  server unavailable — run ~/ros2_ws/debug_bringup.sh first.")
            return
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()
        deadline = time.time() + 5.0
        while ctrl.current_joints is None and time.time() < deadline:
            time.sleep(0.05)
        if ctrl.current_joints is None:
            print("  no joint state — check joint_state_broadcaster.")
            return

        print("\n⚠️  실제 로봇 저속 하강입니다. 컨베이어 정지/접촉 Z/주변 공간을 확인하세요.")
        if input("시작 자세로 이동? (y/N) > ").strip().lower() != "y":
            print("취소.")
            return

        ctrl.suction_off()
        current_q = np.asarray(ctrl.current_joints, dtype=float)
        start_q = q_traj[:, 0]
        M1 = np.asarray(gp8.velocity_limits, dtype=float) * args.move_vel_scale
        M2 = M1 * JOINT_ACCEL_RATIO
        zero = np.zeros(6, dtype=float)
        move_q, move_v, move_ts = trajectory(
            current_q, zero, start_q, zero, M1, M2, hertz=100.0,
        )
        print(f"→ 대기 자세 Z={start_z:.3f}m로 이동 ({move_ts[-1]:.2f}s)...")
        ctrl.send_trajectory_queue(move_q, move_v, move_ts, final_joint=start_q)

        if input("물체 배치 완료? 석션 ON 후 측정 시작? (y/N) > ").strip().lower() != "y":
            print("취소.")
            return

        trigger = threading.Event()
        listener_done = threading.Event()
        result: dict = {}
        listener = threading.Thread(
            target=_measurement_listener,
            args=(ctrl, trigger, listener_done, result),
            daemon=True,
        )
        listener.start()
        print("\n자동 흡착 감지가 아닙니다.")
        print("Space / Enter / A = 물체가 붙은 순간 키 입력+정지, Q = 중단")
        ctrl.suction_on()
        print(f"→ 석션 ON, 시작 위치에서 {args.prime_time:.2f}s 대기...")
        prime_deadline = time.monotonic() + args.prime_time
        while time.monotonic() < prime_deadline and not trigger.is_set():
            time.sleep(0.005)

        stopped_early = trigger.is_set()
        if not trigger.is_set():
            print(f"→ {args.descent_speed*1000:.1f}mm/s 이하로 하강 시작...")
            stopped_early = ctrl.send_trajectory_queue_interruptible(
                q_traj, q_vel, ts, q_traj[:, -1], trigger.is_set,
            )

        listener_done.set()
        listener.join(timeout=0.5)
        kind = result.get("kind")
        if kind == "attach":
            q_hit = result.get("q")
            if q_hit is None and ctrl.current_joints is not None:
                q_hit = list(ctrl.current_joints)
            if q_hit is None:
                print("  기록 키는 받았지만 joint feedback이 없어 거리를 계산할 수 없습니다.")
            else:
                p_hit = _tool_point(gp8, q_hit, args.tool_offset)
                clearance = float(p_hit[2] - contact_z)
                descended = float(start_z - p_hit[2])
                print("\n=== Attachment Measurement ===")
                print(f"  TCP position       : ({p_hit[0]:+.4f}, {p_hit[1]:+.4f}, {p_hit[2]:+.4f}) m")
                print(f"  start에서 하강      : {descended*1000:.1f} mm")
                print(f"  contact 기준 간격   : {clearance*1000:+.1f} mm")
                print(f"  반응시간 추정오차    : ±{args.descent_speed*args.reaction_time*1000:.1f} mm")
                print(f"  stream early stop  : {stopped_early}")
            input("\n확인 후 Enter → 현재 위치에서 석션 OFF ")
        elif kind == "abort":
            print("\n사용자 중단 — 현재 위치에서 석션 OFF.")
        elif result.get("error"):
            print(f"\n측정 입력 오류: {result['error']}")
        else:
            p_end = _tool_point(gp8, ctrl.current_joints, args.tool_offset)
            print("\n끝 Z까지 흡착 순간 키 입력이 없었습니다.")
            print("  (진공센서 자동 감지가 아니라 작업자가 붙는 순간 키를 누르는 방식입니다.)")
            print(f"  final TCP Z={p_end[2]:+.4f}m, clearance={((p_end[2]-contact_z)*1000):+.1f}mm")
    except KeyboardInterrupt:
        print("\nCtrl-C — 정지 및 석션 OFF.")
    finally:
        try:
            ctrl.suction_off()
            ctrl.close()  # drains the queued OFF command before ROS teardown
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

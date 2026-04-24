"""Queue Mode trajectory sending 전용 테스트 스크립트.

app.py 의존성(SAM/camera/conveyor/NN)을 제외하고 TrajectoryController의
queue 메서드만 직접 호출해서 검증. 실제 MotoROS2 + bridge가 떠 있어야 함.

실행:
  # 먼저 bridge 띄우기
  ros2 launch gp8_control debug_robot.launch.py

  # 그 다음 테스트
  ros2 run gp8_control queue_test

테스트 메뉴:
  1. 단순 2점 이동        — send_trajectory_queue (like _move_to_initial_pose)
  2. 3점 경로 (pick 유사) — send_trajectory_queue (like _execute_pick)
  3. release_joint 테스트 — send_trajectory_queue_with_release (like _execute_transfer)
  q. 종료 (queue → FJT 복귀)
"""

from __future__ import annotations

import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node

from gp8_control.controllers.trajectory_controller import TrajectoryController
from gp8_control.robots.gp8 import GP8
from gp8_control.trajectory.trajectory_primitive import (
    trajectory,
    trajectory_3points,
)


# 테스트 공통 파라미터
JOINT_VEL_SCALE = 0.3    # app.py Config.JOINT_VEL_LIMIT_SCALE 와 동일 수준
JOINT_ACCEL_SCALE = 0.5
TRAJ_HZ = 20.0           # app.py Config.TRAJ_HZ 와 동일. 너무 dense하면 queue push가 trajectory 소비를 못 따라감
TEST_DX = 0.05           # 5cm forward (테스트 이동량)
TEST_DZ = 0.03           # 3cm up (3점 경로 중간점용)


def _build_limits(gp8: GP8) -> tuple[np.ndarray, np.ndarray]:
    M1 = np.asarray(gp8.velocity_limits, dtype=float) * JOINT_VEL_SCALE
    M2 = M1 * JOINT_ACCEL_SCALE
    return M1, M2


def _ik(gp8: GP8, T: np.ndarray) -> np.ndarray | None:
    q = gp8.inverse_kinematics(T)
    if q is None:
        return None
    q = np.asarray(q, dtype=float)
    q[-1] = 0.0  # app.py와 동일 규칙
    return q


def _refresh_joints(ctrl: TrajectoryController, node: Node) -> None:
    """메뉴 대기 중 stale 해진 joint_states를 최신화."""
    for _ in range(5):
        rclpy.spin_once(node, timeout_sec=0.05)


def test_simple_move(ctrl: TrajectoryController, node: Node, gp8: GP8) -> None:
    """2점 이동 — _move_to_initial_pose 와 동일 경로."""
    print("\n[Test 1] simple 2-point move: +5cm X")
    _refresh_joints(ctrl, node)
    current_q = np.array(ctrl.current_joints)
    current_T = gp8.forward_kinematics(current_q)

    target_T = current_T.copy()
    target_T[0, 3] += TEST_DX
    target_q = _ik(gp8, target_T)
    if target_q is None:
        print("  IK failed, aborting.")
        return

    M1, M2 = _build_limits(gp8)
    zero = np.zeros_like(M1)
    traj, vel, ts = trajectory(current_q, zero, target_q, zero, M1, M2, hertz=TRAJ_HZ)
    duration = float(ts[-1])  # cumulative time array — last value = total duration
    print(f"  planned: {traj.shape[1]} pts over {duration:.3f}s")

    t0 = time.time()
    ok = ctrl.send_trajectory_queue(traj, vel, ts, final_joint=target_q)
    elapsed = time.time() - t0
    overhead_ms = (elapsed - duration) * 1000
    print(f"  result : ok={ok}, elapsed={elapsed:.3f}s, "
          f"overhead={overhead_ms:+.0f}ms ({overhead_ms/1000/duration*100:+.1f}%)")


def test_three_point_path(ctrl: TrajectoryController, node: Node, gp8: GP8) -> None:
    """3점 경로 — _execute_pick 과 동일 (current → aim → grasp)."""
    print("\n[Test 2] 3-point path: current → (+5cm X, +3cm Z) → (+5cm X)")
    _refresh_joints(ctrl, node)
    current_q = np.array(ctrl.current_joints)
    current_T = gp8.forward_kinematics(current_q)

    aim_T = current_T.copy()
    aim_T[0, 3] += TEST_DX
    aim_T[2, 3] += TEST_DZ
    aim_q = _ik(gp8, aim_T)

    grasp_T = current_T.copy()
    grasp_T[0, 3] += TEST_DX   # aim과 같은 X, 낮은 Z
    grasp_q = _ik(gp8, grasp_T)

    if aim_q is None or grasp_q is None:
        print("  IK failed, aborting.")
        return

    M1, M2 = _build_limits(gp8)
    zero = np.zeros_like(M1)
    traj, vel, ts = trajectory_3points(
        current_q, zero, aim_q, zero, grasp_q, zero, M1, M2, hertz=TRAJ_HZ,
    )
    duration = float(ts[-1])  # cumulative time array — last value = total duration
    print(f"  planned: {traj.shape[1]} pts over {duration:.3f}s")

    t0 = time.time()
    ok = ctrl.send_trajectory_queue(traj, vel, ts, final_joint=grasp_q)
    elapsed = time.time() - t0
    overhead_ms = (elapsed - duration) * 1000
    print(f"  result : ok={ok}, elapsed={elapsed:.3f}s, "
          f"overhead={overhead_ms:+.0f}ms")


def test_release_trajectory(ctrl: TrajectoryController, node: Node, gp8: GP8) -> None:
    """3점 경로 + 중간에 release_joint 도달 시 suction_off.

    _execute_transfer의 send_trajectory_queue_with_release 경로 검증.
    실제 throw NN은 쓰지 않고 간단한 aim→mid→end 경로로 대체.
    """
    print("\n[Test 3] trajectory with release at mid-point")
    print("  (실제 suction I/O 호출됨 — suction이 ON이면 OFF로 바뀜)")
    _refresh_joints(ctrl, node)
    current_q = np.array(ctrl.current_joints)
    current_T = gp8.forward_kinematics(current_q)

    mid_T = current_T.copy()
    mid_T[0, 3] += TEST_DX * 0.5
    mid_T[2, 3] += TEST_DZ
    mid_q = _ik(gp8, mid_T)

    end_T = current_T.copy()
    end_T[0, 3] += TEST_DX
    end_q = _ik(gp8, end_T)

    if mid_q is None or end_q is None:
        print("  IK failed, aborting.")
        return

    M1, M2 = _build_limits(gp8)
    zero = np.zeros_like(M1)
    traj, vel, ts = trajectory_3points(
        current_q, zero, mid_q, zero, end_q, zero, M1, M2, hertz=TRAJ_HZ,
    )
    duration = float(ts[-1])  # cumulative time array — last value = total duration
    print(f"  planned: {traj.shape[1]} pts over {duration:.3f}s")
    print(f"  release_joint = mid_q (J1={np.degrees(mid_q[0]):.2f}°)")

    t0 = time.time()
    ok = ctrl.send_trajectory_queue_with_release(
        traj, vel, ts, final_joint=end_q, release_joint=mid_q,
    )
    elapsed = time.time() - t0
    overhead_ms = (elapsed - duration) * 1000
    print(f"  result : ok={ok}, elapsed={elapsed:.3f}s, "
          f"overhead={overhead_ms:+.0f}ms")


def main() -> None:
    rclpy.init()
    node = Node("queue_mode_test")
    ctrl = TrajectoryController(node)

    try:
        print("Waiting for servers...")
        if not ctrl.wait_for_servers(timeout_sec=10.0):
            print("  Servers unavailable. Is MotoROS2 + bridge running?")
            return
        print("Waiting for joint state...")
        # 백그라운드 executor 없이 직접 spin — /joint_states_urdf 콜백 발화용
        deadline = time.time() + 5.0
        while ctrl.current_joints is None and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if ctrl.current_joints is None:
            print("  No joint states received.")
            return
        print(f"  current joints (deg): "
              f"{[round(float(np.degrees(j)), 1) for j in ctrl.current_joints]}")

        print("\nEntering queue mode...")
        if not ctrl.enter_queue_mode():
            print("  Failed to enter queue mode.")
            return

        try:
            gp8 = GP8()
            while True:
                print("\n=== Queue Mode Tests ===")
                print("  1: simple 2-point move")
                print("  2: 3-point path (pick-like)")
                print("  3: trajectory with release (throw-like)")
                print("  q: quit (return to FJT mode)")
                choice = input("select > ").strip().lower()
                if choice == "q":
                    break
                elif choice == "1":
                    test_simple_move(ctrl, node, gp8)
                elif choice == "2":
                    test_three_point_path(ctrl, node, gp8)
                elif choice == "3":
                    test_release_trajectory(ctrl, node, gp8)
                else:
                    print(f"  unknown: {choice}")
        finally:
            print("\nExiting queue mode (return to FJT)...")
            ctrl.exit_queue_mode()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

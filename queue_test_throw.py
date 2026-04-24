"""Queue Mode throw trajectory 테스트 — torch NN 사용.

app.py의 _execute_transfer와 동일한 경로:
  1. TrajectoryPredictor (FCN)로 throw 파라미터 추론
  2. new_trajectory로 궤적 합성
  3. TrajectoryController.send_trajectory_queue_with_release로 전송
     - release_joint 도달 시 suction_off

SAM/camera/conveyor/AprilTag 의존 없음. 펜던트에서 로봇이 안전하게
스윙할 수 있는 공간 확보 후 실행할 것.

실행 (torch 필요 → venv python):
  # 먼저 bridge
  ros2 launch gp8_control debug_robot.launch.py

  # 그 다음 throw 테스트
  ~/ros2_ws/src/gp8_control/.venv/bin/python -m gp8_control.queue_test_throw

⚠️  주의: 실제 throw 모션입니다 (스윙). 주변 반경 1m 이상 비우고 실행하세요.
"""

from __future__ import annotations

import time

import numpy as np
import rclpy
from rclpy.node import Node

from gp8_control.controllers.trajectory_controller import TrajectoryController
from gp8_control.robots.gp8 import GP8
from gp8_control.trajectory.predictor import TrajectoryPredictor
from gp8_control.trajectory.trajectory_primitive import new_trajectory, pad, trajectory


# app.py Config와 동일
TRAJ_HZ = 20.0
THROW_TIME_SCALE = 1.0
RELEASE_EARLY_SHIFT = 0.05
ETA_MIN = 0.13
ETA_MAX = 0.95

# Throw 입력 파라미터 (NN에 넣을 값)
GRASP_XY_OFFSET = (0.0, 0.0)     # 현재 EE 기준 (상대 좌표 없음 — 절대 xy 사용)
TARGET_XY = (0.0, 0.0)           # 70cm 전방, 중앙
TARGET_DISTANCE = 1.0            # 50cm throw distance

# Pick→Throw 연속 테스트 파라미터
JOINT_VEL_SCALE = 0.3            # queue_test.py 와 동일
JOINT_ACCEL_SCALE = 0.5
# 현재 EE 위치에서의 오프셋(m) — (dx, dy, dz). 여기로 이동 후 throw 시작.
PICK_APPROACH_OFFSET = (0, 0.0, -0.1)


def _show_throw_plan(
    grasp_xy: tuple[float, float],
    aim_xy: tuple[float, float],
    params: np.ndarray,
    T: float,
    eta: float,
    n_points: int,
    release_joint: np.ndarray,
) -> None:
    print(f"\n=== Throw Plan ===")
    print(f"  NN input : grasp_xy={grasp_xy}, target_xy={TARGET_XY}, "
          f"target_distance={TARGET_DISTANCE}")
    print(f"  NN output: T={T:.3f}s, eta={eta:.3f} (release_ratio)")
    print(f"  aim_xy (after throw): {aim_xy}")
    print(f"  trajectory: {n_points} pts @ {TRAJ_HZ}Hz over {T:.3f}s")
    print(f"  release_joint (deg): "
          f"{[round(float(np.degrees(j)), 1) for j in release_joint]}")


def test_throw(
    ctrl: TrajectoryController,
    node: Node,
    gp8: GP8,
    predictor: TrajectoryPredictor,
) -> None:
    # 최신 joint 확보
    for _ in range(5):
        rclpy.spin_once(node, timeout_sec=0.05)
    if ctrl.current_joints is None:
        print("  No joint states available.")
        return

    current_q = np.array(ctrl.current_joints)
    current_T = gp8.forward_kinematics(current_q)
    grasp_xy = (float(current_T[0, 3]), float(current_T[1, 3]))

    # grasp_joint = 현재 위치. aim_joint2 = grasp 동일 (제자리 복귀).
    # 테스트용: net 변위 0 → NN swing arc 자체만 관찰. throw 방향은
    # 스윙 도중 release 시점의 tip velocity로 결정됨 (grasp→aim 방향이 아님).
    grasp_joint = current_q.copy()
    grasp_joint[-1] = 0.0
    aim_T = current_T.copy()          # aim == grasp
    aim_joint2 = grasp_joint.copy()

    # NN 추론
    params = predictor.predict(grasp_xy, TARGET_XY, TARGET_DISTANCE)
    # app.py _execute_transfer 와 동일한 후처리
    w = params[:-2].reshape(-1, 5)
    T = float(np.exp(params[-2])) * THROW_TIME_SCALE
    eta = 1.0 / (1.0 + np.exp(-params[-1]))
    eta = float(np.clip(eta - RELEASE_EARLY_SHIFT, ETA_MIN, ETA_MAX))

    n_steps = max(2, int(T * TRAJ_HZ))
    s = np.linspace(0.0, 1.0, n_steps + 1)
    s_ext = np.concatenate((s, [eta]))

    traj_ext, vel_ext, _, _, ts_ext = new_trajectory(
        s_ext, grasp_joint[:5], aim_joint2[:5], w, T,
    )
    traj_throw = pad(traj_ext[:-1, :]).T
    vel_throw = pad(vel_ext[:-1, :]).T
    timestep_throw = ts_ext[:-1]
    release_joint = traj_ext[-1, :]

    _show_throw_plan(
        grasp_xy, (aim_T[0, 3], aim_T[1, 3]), params, T, eta,
        n_points=traj_throw.shape[1],
        release_joint=pad(release_joint.reshape(1, -1)).ravel(),
    )

    # 최종 확인
    ans = input("\nExecute this throw trajectory? (y/N) > ").strip().lower()
    if ans != "y":
        print("  Cancelled.")
        return

    duration = float(timestep_throw[-1])
    t0 = time.time()
    ok = ctrl.send_trajectory_queue_with_release(
        traj_throw, vel_throw, timestep_throw,
        final_joint=aim_joint2,
        release_joint=pad(release_joint.reshape(1, -1)).ravel(),
    )
    elapsed = time.time() - t0
    overhead_ms = (elapsed - duration) * 1000
    print(f"\n  result: ok={ok}, elapsed={elapsed:.3f}s, "
          f"theoretical={duration:.3f}s, overhead={overhead_ms:+.0f}ms")


def test_pick_then_throw(
    ctrl: TrajectoryController,
    node: Node,
    gp8: GP8,
    predictor: TrajectoryPredictor,
) -> None:
    """Phase 1: 현재 → grasp point 이동 / Phase 2: throw 스윙. 연속 실행.

    두 궤적을 back-to-back으로 쏴서 가운데 gap 측정.
    gap 이 작으면 Queue Mode 연속성이 잘 동작한다는 증거.
    """
    # 최신 joint
    for _ in range(5):
        rclpy.spin_once(node, timeout_sec=0.05)
    if ctrl.current_joints is None:
        print("  No joint states available.")
        return

    current_q = np.array(ctrl.current_joints)
    current_T = gp8.forward_kinematics(current_q)

    # ---------- Phase 1: current → grasp point ----------
    dx, dy, dz = PICK_APPROACH_OFFSET
    grasp_T = current_T.copy()
    grasp_T[0, 3] += dx
    grasp_T[1, 3] += dy
    grasp_T[2, 3] += dz
    grasp_q_raw = gp8.inverse_kinematics(grasp_T)
    if grasp_q_raw is None:
        print("  IK failed for grasp approach pose.")
        return
    grasp_q = np.asarray(grasp_q_raw, dtype=float)
    grasp_q[-1] = 0.0

    M1 = np.asarray(gp8.velocity_limits, dtype=float) * JOINT_VEL_SCALE
    M2 = M1 * JOINT_ACCEL_SCALE
    zero = np.zeros_like(M1)
    pick_traj, pick_vel, pick_ts = trajectory(
        current_q, zero, grasp_q, zero, M1, M2, hertz=TRAJ_HZ,
    )
    pick_duration = float(pick_ts[-1])

    # ---------- Phase 2: throw (grasp에서 다시 grasp로) ----------
    # NN 입력 grasp_xy 는 phase 1 도착 지점 기준
    grasp_xy = (float(grasp_T[0, 3]), float(grasp_T[1, 3]))
    params = predictor.predict(grasp_xy, TARGET_XY, TARGET_DISTANCE)
    w = params[:-2].reshape(-1, 5)
    T = float(np.exp(params[-2])) * THROW_TIME_SCALE
    eta = 1.0 / (1.0 + np.exp(-params[-1]))
    eta = float(np.clip(eta - RELEASE_EARLY_SHIFT, ETA_MIN, ETA_MAX))

    n_steps = max(2, int(T * TRAJ_HZ))
    s = np.linspace(0.0, 1.0, n_steps + 1)
    s_ext = np.concatenate((s, [eta]))

    traj_ext, vel_ext, _, _, ts_ext = new_trajectory(
        s_ext, grasp_q[:5], grasp_q[:5], w, T,   # start == end (제자리 스윙)
    )
    throw_traj = pad(traj_ext[:-1, :]).T
    throw_vel = pad(vel_ext[:-1, :]).T
    throw_ts = ts_ext[:-1]
    release_joint = traj_ext[-1, :]
    throw_duration = float(throw_ts[-1])

    # ---------- Show plan ----------
    print(f"\n=== Pick → Throw Plan ===")
    print(f"  Phase 1 (pick approach Δ=({dx*100:+.1f}, {dy*100:+.1f}, {dz*100:+.1f})cm): "
          f"{pick_traj.shape[1]} pts over {pick_duration:.3f}s")
    print(f"  Phase 2 (throw swing @ grasp pose)     : "
          f"{throw_traj.shape[1]} pts over {throw_duration:.3f}s")
    print(f"  NN output: T={T:.3f}s, eta={eta:.3f}")
    print(f"  Theoretical total (no gap): {pick_duration + throw_duration:.3f}s")

    ans = input("\nExecute? (y/N) > ").strip().lower()
    if ans != "y":
        print("  Cancelled.")
        return

    # ---------- Execute back-to-back ----------
    t0 = time.time()
    ok1 = ctrl.send_trajectory_queue(
        pick_traj, pick_vel, pick_ts, final_joint=grasp_q,
    )
    t_phase1_end = time.time()
    if not ok1:
        print("  Phase 1 failed.")
        return

    ok2 = ctrl.send_trajectory_queue_with_release(
        throw_traj, throw_vel, throw_ts,
        final_joint=grasp_q,
        release_joint=pad(release_joint.reshape(1, -1)).ravel(),
    )
    t_end = time.time()

    phase1_elapsed = t_phase1_end - t0
    phase2_elapsed = t_end - t_phase1_end
    total_elapsed = t_end - t0
    gap = total_elapsed - pick_duration - throw_duration

    print(f"\n  Phase 1 elapsed: {phase1_elapsed:.3f}s "
          f"(theoretical {pick_duration:.3f}s, overhead {(phase1_elapsed-pick_duration)*1000:+.0f}ms)")
    print(f"  Phase 2 elapsed: {phase2_elapsed:.3f}s "
          f"(theoretical {throw_duration:.3f}s, overhead {(phase2_elapsed-throw_duration)*1000:+.0f}ms)")
    print(f"  Total elapsed  : {total_elapsed:.3f}s")
    print(f"  Gap (total - 두 궤적 이론치): {gap*1000:+.0f}ms "
          f"← pick↔throw 사이 '끊김' 지표 (작을수록 연속적)")


def main() -> None:
    rclpy.init()
    node = Node("queue_throw_test")
    ctrl = TrajectoryController(node)

    try:
        print("Waiting for servers...")
        if not ctrl.wait_for_servers(timeout_sec=10.0):
            print("  Servers unavailable. Is MotoROS2 + bridge running?")
            return

        print("Waiting for joint state...")
        deadline = time.time() + 5.0
        while ctrl.current_joints is None and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if ctrl.current_joints is None:
            print("  No joint states received.")
            return
        print(f"  current joints (deg): "
              f"{[round(float(np.degrees(j)), 1) for j in ctrl.current_joints]}")

        print("\nLoading throw NN...")
        predictor = TrajectoryPredictor()
        print(f"  weights: {predictor.weight_path}")

        print("\nEntering queue mode...")
        if not ctrl.enter_queue_mode():
            print("  Failed to enter queue mode.")
            return

        try:
            gp8 = GP8()
            print("\n⚠️  WARNING: Throw trajectory is a fast swing motion.")
            print("    Ensure >1m clearance around the robot.")
            print("    Ctrl+C to abort at any prompt.\n")
            while True:
                print("\n=== Queue Mode Throw Test (torch NN) ===")
                print("  t: run throw trajectory (제자리 스윙)")
                print("  p: pick → throw 연속 (gap 측정)")
                print("  q: quit (return to FJT mode)")
                choice = input("select > ").strip().lower()
                if choice == "q":
                    break
                elif choice == "t":
                    test_throw(ctrl, node, gp8, predictor)
                elif choice == "p":
                    test_pick_then_throw(ctrl, node, gp8, predictor)
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

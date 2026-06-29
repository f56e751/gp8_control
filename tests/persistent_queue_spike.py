"""Stage C 게이트 — 영속 큐(persistent queue) HW 스파이크.

플랜(`hashed-strolling-diffie.md`) Stage C의 **필수 선행 검증**:
큐모드 재진입(~0.4s ×2/cycle)을 제거하려면 "큐를 안 비우고" 다음 세그먼트
점들을 *실행 중인(비-빈) 큐*에 append할 수 있어야 한다. 이때 MotoROS2가
code 204('first point must match current position')를 **다시 검사하는가?**

가설: 204는 큐 *초기화*(빈 큐 → 첫 점)에만 적용. 비-빈 큐 중간 append는
"첫 점"이 아니므로 통과 → 영속 큐 성립. 이 스크립트가 그걸 실측한다.

검증 절차:
  A. 큐모드 1회 진입.
  B. 세그먼트 A(작고 느린 ~4s 이동)를 푸시 → 로봇이 소비 시작(큐 비-빈).
  C. A가 아직 실행 중일 때, 세그먼트 B(= A의 *계획 끝점*에서 이어지는 추가
     이동)의 점들을 **재진입 없이, 그리고 B[0]을 현재위치로 스냅하지 않고**
     그대로 append. 각 점의 result_code를 기록.
       - 전부 SUCCESS(1)  → mid-stream append OK, 204 없음 → Stage C GREEN
       - 204 / WRONG_MODE → mid-stream 거부 → 영속 큐 추가 작업 필요(RED)
  D. (대조) 큐가 드레인되도록 둔 뒤 한 점 푸시 → WRONG_MODE(2) 기대
     (드레인 → 자동 큐모드 종료 확인).

⚠️ 실제 로봇이 움직인다. 동작은 작고(기본 +6cm X) 느리다(0.2s 간격).
   팔 주변 1m 비우고, REMOTE+AUTO·알람 없음 상태에서 실행.

선행:
  ros2 launch gp8_control debug_robot.launch.py
실행(torch 불필요, 시스템 python OK; -m 권장 — setup.py 미등록):
  PYTHONPATH=$HOME/ros2_ws/src python3 -m gp8_control.tests.persistent_queue_spike
"""

from __future__ import annotations

import time

import numpy as np
import rclpy
from rclpy.node import Node

from gp8_control.controllers.trajectory_controller import (
    JOINT_NAMES,
    TrajectoryController,
    _seconds_to_duration,
)
from gp8_control.robots.gp8 import GP8
from motoros2_interfaces.srv import QueueTrajPoint

# --- safe, slow probe parameters ---
SEG_DX = 0.06        # 6 cm forward for segment A
SEG_B_DX = 0.04      # +4 cm further for segment B (continues from A's end)
DT = 0.2             # 0.2 s between queued points (well above ~40ms push round-trip)
NA = 20              # segment A points  -> ~3.8 s of motion (wide injection window)
NB = 10              # segment B points  -> ~2.0 s

_CODE = {1: "SUCCESS", 2: "WRONG_MODE(204-class)", 3: "INIT_FAILURE",
         4: "BUSY", 5: "INVALID_JOINT_LIST", 6: "UNABLE_TO_PROCESS"}


def _ik(gp8: GP8, T: np.ndarray) -> np.ndarray | None:
    q = gp8.inverse_kinematics(T)
    if q is None:
        return None
    q = np.asarray(q, dtype=float)
    q[-1] = 0.0
    return q


def _lin_segment(q0: np.ndarray, q1: np.ndarray, n: int, t0: float, dt: float):
    """Linear joint interpolation, n points, times t0+dt*[0..n-1].

    Velocities: central differences interior, 0 at the two ends (slow probe —
    a near-zero junction between A and B is acceptable and safe)."""
    s = np.linspace(0.0, 1.0, n)
    pos = np.array([q0 + si * (q1 - q0) for si in s])      # (n, 6)
    times = t0 + dt * np.arange(n)
    vel = np.zeros_like(pos)
    for i in range(1, n - 1):
        vel[i] = (pos[i + 1] - pos[i - 1]) / (times[i + 1] - times[i - 1])
    return pos, vel, times


def _push_segment(ctrl, node, pos, vel, times, *, snap_first_to_current, label):
    """Push points directly, RECORDING every result_code (does not early-return).

    snap_first_to_current=True replicates _build_queue_waypoints (new-queue start);
    False appends raw (mid-stream test — first point is A's planned end, NOT current).
    Returns list of (code:int|None)."""
    pos = [list(p) for p in pos]
    if snap_first_to_current and ctrl.current_joints is not None:
        pos[0] = list(ctrl.current_joints)
    codes = []
    print(f"  [{label}] pushing {len(pos)} pts (snap_first={snap_first_to_current}) ...")
    for i in range(len(pos)):
        req = QueueTrajPoint.Request()
        req.joint_names = JOINT_NAMES
        req.point.positions = [float(x) for x in pos[i]]
        req.point.velocities = [float(x) for x in vel[i]]
        req.point.time_from_start = _seconds_to_duration(float(times[i]))
        code = None
        for _ in range(6):                       # tolerate transient BUSY
            fut = ctrl._queue_point_client.call_async(req)
            rclpy.spin_until_future_complete(node, fut, timeout_sec=2.0)
            res = fut.result()
            code = None if res is None else int(res.result_code.value)
            if code == 4:                         # BUSY -> brief wait, retry
                time.sleep(0.015)
                continue
            break
        codes.append(code)
        if code != 1:
            print(f"    pt {i:2d}: code={code} ({_CODE.get(code, code)}) "
                  f"msg='{res.message if res else 'timeout'}'")
    succ = sum(1 for c in codes if c == 1)
    print(f"  [{label}] {succ}/{len(codes)} SUCCESS; codes={codes}")
    return codes


def main() -> None:
    rclpy.init()
    node = Node("persistent_queue_spike")
    ctrl = TrajectoryController(node)
    try:
        print("Waiting for servers (MotoROS2 + bridge)...")
        if not ctrl.wait_for_servers(timeout_sec=10.0):
            print("  Servers unavailable. Is debug_robot.launch.py running?")
            return
        deadline = time.time() + 5.0
        while ctrl.current_joints is None and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if ctrl.current_joints is None:
            print("  No joint states.")
            return

        gp8 = GP8()
        cur = np.array(ctrl.current_joints)
        T = gp8.forward_kinematics(cur)
        TA = T.copy(); TA[0, 3] += SEG_DX
        TB = T.copy(); TB[0, 3] += SEG_DX + SEG_B_DX
        qA = _ik(gp8, TA)
        qB = _ik(gp8, TB)
        if qA is None or qB is None:
            print("  IK failed for probe targets; aborting.")
            return

        a_pos, a_vel, a_t = _lin_segment(cur, qA, NA, 0.0, DT)
        # B continues from A's PLANNED END (qA), times offset past A's last.
        b_pos, b_vel, b_t = _lin_segment(qA, qB, NB, a_t[-1] + DT, DT)

        print(f"\ncurrent (deg): {[round(float(np.degrees(j)),1) for j in cur]}")
        print(f"segment A: {NA} pts over {a_t[-1]:.1f}s  (+{SEG_DX*100:.0f}cm X)")
        print(f"segment B: {NB} pts over {b_t[-1]-b_t[0]+DT:.1f}s (append while A runs, "
              f"B[0]=A_end NOT current)")
        print("\n⚠️  Robot WILL move (~10cm total, slow). Clear 1m. Enter 'go' to proceed.")
        if input("> ").strip().lower() != "go":
            print("Aborted by user.")
            return

        print("\nEntering queue mode (ONCE) ...")
        if not ctrl.enter_queue_mode():
            print("  Failed to enter queue mode.")
            return
        try:
            # B: push A, then immediately append B WHILE A is still executing.
            _push_segment(ctrl, node, a_pos, a_vel, a_t,
                          snap_first_to_current=True, label="A")
            # A push (~NA*40ms) done; robot now consuming A (≈3.8s). Queue non-empty.
            print("  injecting B immediately (queue should be non-empty / A executing)...")
            b_codes = _push_segment(ctrl, node, b_pos, b_vel, b_t,
                                    snap_first_to_current=False, label="B")

            # wait for the whole A+B stream to finish
            ctrl._wait_trajectory_end(float(b_t[-1]), tail_buffer=0.3)
            reached = max(abs(a - b) for a, b in
                          zip(ctrl.current_joints, list(qB)))
            print(f"\n  final joint max-dev from B target = {reached:.4f} rad")

            ok_midstream = all(c == 1 for c in b_codes)
            print("\n===== STAGE C SPIKE VERDICT =====")
            if ok_midstream:
                print("  GREEN: mid-stream append into a non-empty queue SUCCEEDED "
                      "(no 204 / WRONG_MODE).")
                print("  -> Persistent queue is feasible; re-entry (~0.8s/cycle) can be removed.")
            else:
                bad = [(i, c) for i, c in enumerate(b_codes) if c != 1]
                print(f"  RED: B rejected at {bad}.")
                print("  -> MotoROS2 re-checks the appended point; persistent-queue needs"
                      " a different join (e.g. wait-until-near-end, or floating-start plan).")
            print("=================================")
        finally:
            print("\nReturning to FJT mode ...")
            ctrl.exit_queue_mode()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

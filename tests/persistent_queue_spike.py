"""Stage C 게이트 — 영속 큐(persistent queue) HW 스파이크.

플랜(`hashed-strolling-diffie.md`) Stage C의 **필수 선행 검증**:
큐모드 재진입(측정 ~0.41s/dispatch = dispatch 오버헤드의 61%)을 제거하려면
"큐를 안 비우고" 다음 세그먼트 점들을 *실행 중인(비-빈) 큐*에 append할 수 있어야
한다. 이때 MotoROS2가 "첫 점이 현재 위치와 일치"(code 204류) 검사를 **다시
하는가?**

가설: 그 검사는 큐 *초기화*(빈 큐 → 첫 점)에만 적용. 비-빈 큐 중간 append는
"첫 점"이 아니므로 통과 → 영속 큐 성립. 이 스크립트가 그걸 실측한다.

이 스파이크는 프로덕션 프리미티브
``TrajectoryController.push_segments_persistent`` 를 그대로 호출한다(단, 폴백을
꺼서 append의 raw 코드를 관찰). 세그먼트 A(작고 느린 이동)와 B(= A의 *계획
끝점*에서 이어지는 추가 이동)를 하나의 큐 세션에 연속 푸시하고, B의 result_code를
본다:
  - 전부 SUCCESS(1)  → mid-stream append OK → Stage C GREEN
  - 2/3/6 등 거부    → 재검사 발생 → wait-until-near-end join 필요(RED)

⚠️ 실제 로봇이 움직인다. 동작은 작고(기본 +10cm X 총합) 느리다(0.2s 간격).
   팔 주변 1m 비우고, REMOTE+AUTO·알람 없음 상태에서 실행.

선행:
  ros2 launch gp8_control debug_robot.launch.py
실행(torch 불필요; setup.py 미등록 → -m):
  PYTHONPATH=$HOME/ros2_ws/src python3 -m gp8_control.tests.persistent_queue_spike
"""

from __future__ import annotations

import time

import numpy as np
import rclpy
from rclpy.node import Node

from gp8_control.controllers.trajectory_controller import TrajectoryController
from gp8_control.robots.gp8 import GP8

# --- safe, slow probe parameters ---
SEG_DX = 0.06        # 6 cm forward for segment A
SEG_B_DX = 0.04      # +4 cm further for segment B (continues from A's end)
DT = 0.2             # 0.2 s between queued points (well above ~40ms push round-trip)
NA = 20              # segment A points  -> ~3.8 s of motion (wide injection window)
NB = 10              # segment B points  -> ~2.0 s

_CODE = {1: "SUCCESS", 2: "WRONG_MODE", 3: "INIT_FAILURE",
         4: "BUSY", 5: "INVALID_JOINT_LIST", 6: "UNABLE_TO_PROCESS_POINT"}


def _ik(gp8: GP8, T: np.ndarray) -> "np.ndarray | None":
    q = gp8.inverse_kinematics(T)
    if q is None:
        return None
    q = np.asarray(q, dtype=float)
    q[-1] = 0.0
    return q


def _lin_segment(q0: np.ndarray, q1: np.ndarray, n: int, dt: float):
    """Linear joint interpolation -> (traj DOFxn, vel DOFxn, ts) for the
    controller's queue builder. Times are 0-based (the persistent primitive
    offsets each segment past the previous one). Velocities: central differences
    interior, 0 at the ends (slow probe — a near-zero A→B junction is safe)."""
    s = np.linspace(0.0, 1.0, n)
    pos = np.array([q0 + si * (q1 - q0) for si in s])      # (n, 6)
    times = dt * np.arange(n)
    vel = np.zeros_like(pos)
    for i in range(1, n - 1):
        vel[i] = (pos[i + 1] - pos[i - 1]) / (times[i + 1] - times[i - 1])
    return pos.T, vel.T, times                              # (6,n), (6,n), (n,)


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

        traj_a, vel_a, ts_a = _lin_segment(cur, qA, NA, DT)
        traj_b, vel_b, ts_b = _lin_segment(qA, qB, NB, DT)   # B[0]=qA (A's END, not current)
        segments = [(traj_a, vel_a, ts_a, qA), (traj_b, vel_b, ts_b, qB)]

        print(f"\ncurrent (deg): {[round(float(np.degrees(j)), 1) for j in cur]}")
        print(f"segment A: {NA} pts over {ts_a[-1]:.1f}s  (+{SEG_DX * 100:.0f}cm X)")
        print(f"segment B: {NB} pts over {ts_b[-1]:.1f}s (append while A runs; "
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
            # push_segments_persistent aborts on the first reject, so B's codes
            # are observed RAW — the timeline is monotonic by construction, so a
            # B-reject genuinely means a start-position re-check (not a self-
            # inflicted duplicate timestamp).
            ok, seg_codes = ctrl.push_segments_persistent(
                segments, wait=True, tail_buffer=0.3)
            a_codes, b_codes = seg_codes[0], seg_codes[1]
            print(f"\n  A codes: {a_codes}")
            print(f"  B codes: {b_codes}  "
                  f"({', '.join(_CODE.get(c, str(c)) for c in b_codes)})")
            dev = max(abs(a - b) for a, b in zip(ctrl.current_joints, list(qB)))
            print(f"  final joint max-dev from B target = {dev:.4f} rad")

            print("\n===== STAGE C SPIKE VERDICT =====")
            if not a_codes or not all(c == 1 for c in a_codes):
                print(f"  INCONCLUSIVE: segment A (queue init) did not fully "
                      f"succeed: {a_codes}. Check queue mode / current position.")
            elif not b_codes:
                print("  INCONCLUSIVE: segment B never pushed (A aborted early).")
            elif all(c == 1 for c in b_codes):
                print("  GREEN: mid-stream append into a non-empty queue SUCCEEDED "
                      "(no start-position re-check).")
                print("  -> Persistent queue feasible; re-entry (~0.41s/dispatch) "
                      "can be removed via push_segments_persistent.")
            else:
                bad = [(i, _CODE.get(c, c)) for i, c in enumerate(b_codes) if c != 1]
                print(f"  RED: B rejected at {bad} (see [PERSIST] log for the msg; "
                      "'204'/init text = start-position re-check).")
                print("  -> MotoROS2 re-checks the appended point. Persistent queue "
                      "needs a different join (feed hold points + push throw just "
                      "before drain, or floating-start plan).")
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

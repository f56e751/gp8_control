"""Stage-C HW spike — persistent queue with a dynamic HOLD (ambush miniature).

The A→B append feasibility already tested GREEN (MotoROS2 accepts mid-stream
appends to a non-empty queue). This spike validates the NEXT building block the
real pick→throw needs: keeping ONE queue alive across a multi-second WAIT by
feeding hold points (so no re-entry before the throw), then appending the
continuation — via the controller's session API:

    enter_queue_mode()             # once
    pq_begin()
    pq_segment(A)                  # "pick": drive to grasp
    pq_hold_until(grasp, +HOLD_S)  # "ambush wait": feed holds, queue stays ALIVE
    pq_segment(B, is_last=True)    # "throw": continuation into the SAME queue
    pq_finish()

Verdict:
  - all ok + queue survived the hold + arm reaches B target -> GREEN
    (persistent queue across an ambush wait works; re-entry can be removed)
  - pq_hold_until returns False (WRONG_MODE mid-hold) -> queue drained: the hold
    pacing (lead) is too shallow -> RED / needs a larger lead.

⚠️ Real robot moves ~10cm slowly, then holds at the grasp pose for HOLD_S, then
   continues. Clear 1m; REMOTE+AUTO, no alarms.

  ros2 launch gp8_control debug_robot.launch.py    # terminal 1
  ros2 run gp8_control persistent_queue_spike       # terminal 2
"""

from __future__ import annotations

import time

import numpy as np
import rclpy
from rclpy.node import Node

from gp8_control.controllers.trajectory_controller import TrajectoryController
from gp8_control.robots.gp8 import GP8

SEG_DX = 0.06        # 6 cm forward for segment A ("pick" drive)
SEG_B_DX = 0.04      # +4 cm further for segment B ("throw" continuation)
DT = 0.2             # inter-point spacing for the moving segments
NA = 20              # segment A points
NB = 10              # segment B points
HOLD_S = 3.0         # seconds to hold at grasp (simulated ambush wait)


def _ik(gp8: GP8, T: np.ndarray) -> "np.ndarray | None":
    q = gp8.inverse_kinematics(T)
    if q is None:
        return None
    q = np.asarray(q, dtype=float)
    q[-1] = 0.0
    return q


def _lin_segment(q0: np.ndarray, q1: np.ndarray, n: int, dt: float):
    """Linear joint interp -> (traj 6xn, vel 6xn, ts) 0-based; ends at zero vel."""
    s = np.linspace(0.0, 1.0, n)
    pos = np.array([q0 + si * (q1 - q0) for si in s])
    times = dt * np.arange(n)
    vel = np.zeros_like(pos)
    for i in range(1, n - 1):
        vel[i] = (pos[i + 1] - pos[i - 1]) / (times[i + 1] - times[i - 1])
    return pos.T, vel.T, times


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
        traj_b, vel_b, ts_b = _lin_segment(qA, qB, NB, DT)

        print(f"\ncurrent (deg): {[round(float(np.degrees(j)), 1) for j in cur]}")
        print(f"A: drive +{SEG_DX*100:.0f}cm ({NA} pts) -> HOLD {HOLD_S:.0f}s at grasp "
              f"-> B: continue +{SEG_B_DX*100:.0f}cm ({NB} pts), ONE queue session")
        print("\n⚠️  Robot moves ~10cm, holds, continues. Clear 1m. Enter 'go'.")
        if input("> ").strip().lower() != "go":
            print("Aborted by user.")
            return

        print("\nEntering queue mode (ONCE) ...")
        if not ctrl.enter_queue_mode():
            print("  Failed to enter queue mode.")
            return
        try:
            ctrl.pq_begin()
            t0 = time.time()
            okA, cA = ctrl.pq_segment(traj_a, vel_a, ts_a, qA)
            print(f"  A ({len(cA)} pts): ok={okA}")
            okH = okB = False
            cB = []
            if okA:
                okH = ctrl.pq_hold_until(qA, time.time() + HOLD_S)
                print(f"  HOLD {HOLD_S:.0f}s: queue stayed alive = {okH}")
                if okH:
                    okB, cB = ctrl.pq_segment(traj_b, vel_b, ts_b, qB, is_last=True)
                    print(f"  B ({len(cB)} pts): ok={okB}, codes={cB}")
                else:
                    print("  (hold drained — not appending B)")
            else:
                print("  (A aborted — not holding/appending B)")
            ctrl.pq_finish(wait=True)
            dev = max(abs(a - b) for a, b in zip(ctrl.current_joints, list(qB)))
            print(f"  total {time.time()-t0:.1f}s; final max-dev from B = {dev:.4f} rad")

            print("\n===== STAGE C HOLD-SPIKE VERDICT =====")
            if okA and okH and okB and dev < 0.02:
                print("  GREEN: one queue survived the pick + HOLD + throw with NO "
                      "re-entry; arm reached the continuation target.")
                print("  -> Ready to wire push_segments/hold into throw_skill "
                      "(feed holds during ambush, append throw).")
            elif not okA:
                print(f"  INCONCLUSIVE: segment A (queue init) failed: {cA}. "
                      "Check queue mode / current position.")
            elif not okH:
                print("  RED (hold drained): the queue emptied during the hold — "
                      "increase pq_hold_until lead (more buffered ahead).")
            else:
                print(f"  RED/PARTIAL: okA={okA} okH={okH} okB={okB} dev={dev:.4f}. "
                      f"B codes={cB}. See [PQ] log.")
            print("======================================")
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

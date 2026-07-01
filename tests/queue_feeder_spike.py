"""Stage-1 HW spike for the UNIVERSAL background feeder (cross-cycle, any skill).

The cross-cycle reuse (throw→throw) only keeps the queue alive when consecutive
throws chain; a mixed push/throw workload breaks that and re-enters every cycle.
The QueueFeeder removes the re-entry UNIVERSALLY: a single background thread keeps
the queue alive with hold points while the main thread does other work (perception
/ selection), and segments are SUBMITTED onto the one live session.

This spike proves that mechanism in ISOLATION — no app/skill wiring — under a
SINGLE enter_queue_mode:

    enter_queue_mode(); pq_begin()          # ONCE, main thread
    feeder.start()                          # feeder now owns all queue ROS
    <main thread: simulate run_epoch gaps for ~8s, NO ROS calls>   # keep-alive holds
    submit A (drive +6cm) ; gap ; submit B (+4cm) ; gap ; submit C (back)  # splice
    feeder.stop(); pq_finish(); exit_queue_mode()

Verdict:
  - feeder never drained + all 3 segments ok + arm back at A + ONE enter_queue_mode
    -> GREEN (a single session survives multi-second gaps AND segment splices with
    no re-entry — ready to wire push/throw skills onto submit_segment).
  - feeder.drained True (WRONG_MODE) -> the keep-alive lost the queue (lead too
    shallow / a gap starved it) -> RED.

⚠️ Real robot moves ~10cm forward in steps with multi-second holds between, then
   back. Clear 1m; REMOTE+AUTO, no alarms.

  ros2 launch gp8_control debug_robot.launch.py    # terminal 1
  ros2 run gp8_control queue_feeder_spike           # terminal 2
"""

from __future__ import annotations

import time

import numpy as np
import rclpy
from rclpy.node import Node

from gp8_control.controllers.trajectory_controller import TrajectoryController
from gp8_control.controllers.queue_feeder import QueueFeeder
from gp8_control.robots.gp8 import GP8

SEG_DX = 0.06        # A: 6 cm forward (the "pick" drive)
SEG_B_DX = 0.04      # B: +4 cm further
DT = 0.2             # inter-point spacing for the moving segments
NA = 20
NB = 10
NC = 12
KEEPALIVE_S = 8.0    # simulate run_epoch gaps with NO submit (pure keep-alive)
GAP_S = 1.8          # gap between segment submits (feeder auto-holds meanwhile)


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


def _busy_gap(seconds: float, feeder: QueueFeeder) -> None:
    """Simulate run_epoch work (perception math + sleep) with NO ROS calls, so the
    feeder is the only thread touching the node. Bail early if the feeder drained."""
    t_end = time.time() + seconds
    while time.time() < t_end and not feeder.drained:
        _ = sum(i * i for i in range(2000))     # dummy 'perception/selection' math
        time.sleep(0.05)


def main() -> None:
    rclpy.init()
    node = Node("queue_feeder_spike")
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
        seg_a = _lin_segment(cur, qA, NA, DT)      # drive to A
        seg_b = _lin_segment(qA, qB, NB, DT)       # continue to B
        seg_c = _lin_segment(qB, qA, NC, DT)       # back to A

        print(f"\ncurrent (deg): {[round(float(np.degrees(j)), 1) for j in cur]}")
        print(f"Plan: enter ONCE -> keep-alive {KEEPALIVE_S:.0f}s (no submit) -> "
              f"A(+{SEG_DX*100:.0f}cm) --gap-- B(+{SEG_B_DX*100:.0f}cm) --gap-- C(back), "
              "all one feeder session.")
        print("\n⚠️  Robot moves ~10cm in steps with holds, then back. Clear 1m. Enter 'go'.")
        if input("> ").strip().lower() != "go":
            print("Aborted by user.")
            return

        print("\nEntering queue mode (ONCE) ...")
        if not ctrl.enter_queue_mode():
            print("  Failed to enter queue mode.")
            return
        feeder = QueueFeeder(ctrl)
        results = {}
        try:
            ctrl.pq_begin()
            feeder.start()          # <-- from here, ONLY the feeder touches ROS
            t0 = time.time()

            print(f"  keep-alive for {KEEPALIVE_S:.0f}s (main thread idle, no submit)...")
            _busy_gap(KEEPALIVE_S, feeder)
            results["keepalive_survived"] = not feeder.drained
            print(f"    survived (queue still alive) = {not feeder.drained}")

            for name, seg in (("A", seg_a), ("B", seg_b), ("C", seg_c)):
                if feeder.drained:
                    results[name] = False
                    continue
                is_last = name == "C"
                final = qA if name in ("A", "C") else qB
                h = feeder.submit_segment(seg[0], seg[1], seg[2], final, is_last=is_last)
                ok = h.wait(timeout=15.0)
                results[name] = ok
                print(f"    segment {name}: ok={ok} codes={h.codes}")
                if name != "C":
                    _busy_gap(GAP_S, feeder)      # feeder auto-holds at the segment end

            _busy_gap(2.0, feeder)               # let C finish + settle
        finally:
            feeder.stop()
            # feeder thread is gone -> main thread may spin again
            for _ in range(40):
                rclpy.spin_once(node, timeout_sec=0.02)
            ctrl.pq_finish(wait=True)
            print("\nReturning to FJT mode ...")
            ctrl.exit_queue_mode()

        dev = max(abs(a - b) for a, b in zip(ctrl.current_joints, list(qA)))
        print(f"\n  total {time.time()-t0:.1f}s; final max-dev from A = {dev:.4f} rad; "
              f"drained={feeder.drained}")
        print("\n===== FEEDER SPIKE VERDICT =====")
        all_ok = (results.get("keepalive_survived") and results.get("A")
                  and results.get("B") and results.get("C") and not feeder.drained)
        if all_ok and dev < 0.03:
            print("  GREEN: ONE queue session survived an 8s gap + 3 spliced segments "
                  "with NO re-entry; arm tracked A->B->A.")
            print("  -> feeder mechanism validated; ready to wire push/throw skills "
                  "onto feeder.submit_segment (universal cross-cycle).")
        elif feeder.drained:
            print("  RED (drained): the keep-alive lost the queue (WRONG_MODE). "
                  "Increase lead / shorten the gap; check the feeder pacing.")
        else:
            print(f"  RED/PARTIAL: {results}, dev={dev:.4f}. See [PQ]/[FEEDER] log.")
        print("================================")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

"""Measure the QUEUE-path startup floor (parallel to measure_fjt_floor).

FJT sends one goal (no per-point push) and pays ~0.15s controller startup after
accept. The queue path instead (a) enters queue mode (a stop+start MODE SWITCH
that FJT never pays) and (b) STREAMS points via queue_traj_point (~44ms each),
and the arm begins once enough points are buffered. This measures all three, per
small joint move:

  reenter_ms = enter_queue_mode() cost (start_point_queue_mode) — the re-entry
  push_ms    = time to push the N points
  floor_ms   = push-start -> first joint motion (delivery + controller startup + ramp)

Two profiles (gentle/fast, SAME point count so push time matches) subtract the
0.01-rad detection ramp, exactly like the FJT meter, so the queue's push->motion
floor can be compared to FJT's ~0.15s. Re-enters queue mode each trial (the queue
drains after each move), so reenter_ms is measured every time.

  ros2 launch gp8_control debug_robot.launch.py   # terminal 1
  ros2 run gp8_control measure_queue_floor          # terminal 2
Pendant REMOTE+AUTO, no alarms; clear the arm. Do NOT run with gp8_bringup.
"""

from __future__ import annotations

import statistics as st
import time

import numpy as np
import rclpy
from rclpy.node import Node

from gp8_control.controllers.trajectory_controller import (
    JOINT_NAMES,
    TrajectoryController,
    _seconds_to_duration,
)
from motoros2_interfaces.srv import QueueTrajPoint

MOVE_JOINT = 0          # joint_1_s (base) — small safe move
MOTION_THRESH = 0.01    # rad — first-motion detection
N_PTS = 5               # points per move (FIXED across profiles so push time matches)


def _max_abs(a, b) -> float:
    return max(abs(float(a[i]) - float(b[i])) for i in range(min(len(a), len(b))))


def one_trial(ctrl: TrajectoryController, node: Node, delta: float,
              duration: float, sign: int):
    """Enter queue mode + push a small move; return (reenter_ms, push_ms, floor_ms)."""
    start = np.array(ctrl.current_joints, dtype=float)
    ss = np.linspace(0.0, 1.0, N_PTS)
    pos = [list(start) for _ in range(N_PTS)]
    for i in range(N_PTS):
        pos[i][MOVE_JOINT] = float(start[MOVE_JOINT] + delta * sign * ss[i])
    pos[0] = list(ctrl.current_joints)                 # queue-init (code 204) snap
    times = list(np.linspace(0.0, duration, N_PTS))
    vel = [[0.0] * 6 for _ in range(N_PTS)]            # central-diff interior, 0 at ends
    for i in range(1, N_PTS - 1):
        vel[i][MOVE_JOINT] = (pos[i + 1][MOVE_JOINT] - pos[i - 1][MOVE_JOINT]) / \
            (times[i + 1] - times[i - 1])

    if not ctrl.enter_queue_mode():
        return None
    reenter_ms = ctrl.last_qmode_ms

    t_push = time.time()
    t_motion = None
    for i in range(N_PTS):
        req = QueueTrajPoint.Request()
        req.joint_names = JOINT_NAMES
        req.point.positions = [float(x) for x in pos[i]]
        req.point.velocities = [float(x) for x in vel[i]]
        req.point.time_from_start = _seconds_to_duration(times[i])
        for _ in range(6):
            fut = ctrl._queue_point_client.call_async(req)
            rclpy.spin_until_future_complete(node, fut, timeout_sec=2.0)
            res = fut.result()
            if res is not None and res.result_code.value == 4:   # BUSY
                time.sleep(0.015)
                continue
            break
        if (t_motion is None and ctrl.current_joints is not None
                and _max_abs(ctrl.current_joints, start) > MOTION_THRESH):
            t_motion = time.time()
    push_ms = (time.time() - t_push) * 1000.0

    deadline = time.time() + duration + 0.5
    while t_motion is None and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.005)
        if _max_abs(ctrl.current_joints, start) > MOTION_THRESH:
            t_motion = time.time()
    floor_ms = (t_motion - t_push) * 1000.0 if t_motion is not None else None

    ctrl._wait_trajectory_end(times[-1], t_start=t_push, tail_buffer=0.2)
    return reenter_ms, push_ms, floor_ms


def main() -> None:
    rclpy.init()
    node = Node("measure_queue_floor")
    ctrl = TrajectoryController(node)
    try:
        if not ctrl.wait_for_servers(timeout_sec=10.0):
            print("Servers unavailable — is debug_robot.launch.py up?")
            return
        deadline = time.time() + 5.0
        while ctrl.current_joints is None and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if ctrl.current_joints is None:
            print("No joint states.")
            return
        print(f"current (deg): {[round(float(np.degrees(j)), 1) for j in ctrl.current_joints]}")
        profiles = [("gentle", 0.05, 0.6), ("fast", 0.20, 0.25)]
        n_each = 8
        print(f"\n2 profiles x {n_each} queue moves on joint_{MOVE_JOINT+1} "
              f"({N_PTS} pts each; gentle +/-2.9deg/0.6s, fast +/-11.5deg/0.25s). Clear the arm.")
        if input("Enter 'go' to proceed > ").strip().lower() != "go":
            print("Aborted.")
            return

        try:
            results = {}
            for name, delta, dur in profiles:
                rs, ps, fs = [], [], []
                print(f"\n--- {name} (+/-{np.degrees(delta):.1f} deg, {dur}s) ---")
                print(f"{'trial':>5} {'reenter_ms':>11} {'push_ms':>8} "
                      f"{'floor_ms(push->motion)':>24}")
                for i in range(n_each):
                    out = one_trial(ctrl, node, delta, dur, 1 if i % 2 == 0 else -1)
                    if out is None:
                        print(f"{i:5d}   enter_queue_mode failed")
                        continue
                    r, p, f = out
                    rs.append(r)
                    ps.append(p)
                    if f is not None:
                        fs.append(f)
                    print(f"{i:5d} {r:11.0f} {p:8.0f} "
                          f"{(round(f) if f is not None else 'MISS'):>24}")
                    time.sleep(0.2)
                results[name] = (rs, ps, fs)
        finally:
            ctrl.exit_queue_mode()

        print("\n===== QUEUE STARTUP =====")
        fmeans = {}
        for name, _, _ in profiles:
            rs, ps, fs = results.get(name, ([], [], []))
            if fs:
                fmeans[name] = st.mean(fs)
            print(f"  {name:7s} reenter {st.mean(rs) if rs else 0:5.0f} ms | "
                  f"push {st.mean(ps) if ps else 0:5.0f} ms | "
                  f"push->motion {st.mean(fs) if fs else float('nan'):5.0f} ms "
                  f"(sd {st.pstdev(fs) if len(fs) > 1 else 0:.0f})")
        if "gentle" in fmeans and "fast" in fmeans:
            ramp = fmeans["gentle"] - fmeans["fast"]
            print(f"\n  ramp difference (gentle-fast) ~= {ramp:.0f} ms (detection artifact).")
            print(f"  => queue push->motion minus ramp ~= FAST floor ~= {fmeans['fast']:.0f} ms "
                  f"(= per-point push delivery + controller startup, overlapped).")
            print("  Compare: FJT startup ~150-180ms (no push). Plus queue pays the "
                  "reenter_ms (~400ms) on top, which FJT does not.")
        print("=========================")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

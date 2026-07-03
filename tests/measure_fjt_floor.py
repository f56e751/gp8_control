"""Measure the pure controller startup FLOOR via FJT (no per-point push).

The queue path's "start gap" is dominated by the per-point queue_traj_point
round-trips (~44 ms/point) — NOT a controller floor. FJT sends the whole
trajectory in ONE goal, so it has no per-point push; the time from the goal
being ACCEPTED to the arm first moving is the controller + transport floor that
neither FJT nor the queue can avoid. This measures it directly.

Per trial it reports:
  accept_ms = send -> goal accepted            (action handshake)
  floor_ms  = accepted -> first joint motion   (the FLOOR — no per-point push)
A small base-joint move (+/-3 deg, 0.6 s) alternates sign so the arm oscillates
around its start pose (no drift). Motion is detected on /joint_states_urdf
(~50 Hz -> ~+/-20 ms resolution, the same limit as the queue-path measurement).

  ros2 launch gp8_control debug_robot.launch.py   # terminal 1 (bridge + MotoROS2)
  ros2 run gp8_control measure_fjt_floor           # terminal 2
Pendant REMOTE+AUTO, no alarms; clear the arm.
"""

from __future__ import annotations

import statistics as st
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import qos_profile_sensor_data
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from motoros2_interfaces.srv import StartTrajMode
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

JOINT_NAMES = ["joint_1_s", "joint_2_l", "joint_3_u",
               "joint_4_r", "joint_5_b", "joint_6_t"]
N_TRIALS = 12
MOVE_JOINT = 0          # joint_1_s (base rotation) — safe small move
DELTA = 0.05            # rad (~2.9 deg)
DURATION = 0.6          # s per move
MOTION_THRESH = 0.01    # rad — first-motion detection (above joint_states noise)


def _dur(seconds: float) -> Duration:
    return Duration(sec=int(seconds), nanosec=int((seconds - int(seconds)) * 1e9))


def _max_abs(a, b) -> float:
    return max(abs(float(a[i]) - float(b[i])) for i in range(min(len(a), len(b))))


class FloorMeter(Node):
    def __init__(self) -> None:
        super().__init__("measure_fjt_floor")
        cb = ReentrantCallbackGroup()
        self.cur: "list | None" = None
        self.create_subscription(
            JointState, "/joint_states_urdf", self._js_cb,
            qos_profile_sensor_data, callback_group=cb)
        self._fjt = ActionClient(
            self, FollowJointTrajectory,
            "/motoman_gp8_controller/follow_joint_trajectory", callback_group=cb)
        self._start_traj = self.create_client(
            StartTrajMode, "/start_traj_mode", callback_group=cb)

    def _js_cb(self, msg: JointState) -> None:
        self.cur = list(msg.position)

    def start_traj_mode(self) -> None:
        if not self._start_traj.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn("/start_traj_mode unavailable; assuming already in traj mode.")
            return
        fut = self._start_traj.call_async(StartTrajMode.Request())
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)

    def one_trial(self, sign: int, delta: float, duration: float):
        """Send one FJT goal; return (accept_ms, floor_ms|None)."""
        start = np.array(self.cur, dtype=float)
        target = start.copy()
        target[MOVE_JOINT] += delta * sign

        jt = JointTrajectory()
        jt.joint_names = JOINT_NAMES
        for pos, t in ((start, 0.0), (target, duration)):
            pt = JointTrajectoryPoint()
            pt.positions = [float(x) for x in pos]
            pt.velocities = [0.0] * 6
            pt.time_from_start = _dur(t)
            jt.points.append(pt)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = jt

        t_send = time.time()
        fut = self._fjt.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut)
        gh = fut.result()
        if gh is None or not gh.accepted:
            self.get_logger().warn("goal rejected.")
            return None, None
        t_accept = time.time()

        t_motion = None
        deadline = time.time() + 3.0
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.005)
            if self.cur is not None and _max_abs(self.cur, start) > MOTION_THRESH:
                t_motion = time.time()
                break
        # let the move finish so the arm settles before the next trial
        res = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res)
        accept_ms = (t_accept - t_send) * 1000.0
        floor_ms = (t_motion - t_accept) * 1000.0 if t_motion is not None else None
        return accept_ms, floor_ms


def main() -> None:
    rclpy.init()
    node = FloorMeter()
    try:
        if not node._fjt.wait_for_server(timeout_sec=10.0):
            print("FollowJointTrajectory unavailable — is debug_robot.launch.py up?")
            return
        deadline = time.time() + 5.0
        while node.cur is None and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.cur is None:
            print("No /joint_states_urdf.")
            return
        print(f"current (deg): {[round(float(np.degrees(j)), 1) for j in node.cur]}")
        # Two profiles: GENTLE (slow ramp -> reaching the 0.01 rad detect threshold
        # takes long) vs FAST (steep ramp -> reaches threshold almost immediately).
        # accept->motion = startup + ramp-to-0.01rad; the ramp term differs between
        # profiles, so (gentle - fast) isolates the ramp and FAST ~= pure startup.
        profiles = [("gentle", 0.05, 0.6), ("fast", 0.20, 0.25)]
        n_each = 8
        print(f"\n2 profiles x {n_each} FJT goals on joint_{MOVE_JOINT+1} "
              f"(gentle +/-2.9deg/0.6s, fast +/-11.5deg/0.25s). Clear the arm.")
        if input("Enter 'go' to proceed > ").strip().lower() != "go":
            print("Aborted.")
            return

        node.start_traj_mode()
        time.sleep(0.3)

        results = {}
        for name, delta, dur in profiles:
            accepts, floors = [], []
            print(f"\n--- {name} (+/-{np.degrees(delta):.1f} deg, {dur}s) ---")
            print(f"{'trial':>5} {'accept_ms':>10} {'floor_ms(accept->motion)':>26}")
            for i in range(n_each):
                a, f = node.one_trial((1 if i % 2 == 0 else -1), delta, dur)
                if a is not None:
                    accepts.append(a)
                if f is not None:
                    floors.append(f)
                print(f"{i:5d} {a if a is None else round(a):>10} "
                      f"{(round(f) if f is not None else 'MISS'):>26}")
                time.sleep(0.2)
            results[name] = (accepts, floors)

        print("\n===== FJT STARTUP (no per-point push) =====")
        fmeans = {}
        for name, _, _ in profiles:
            accepts, floors = results[name]
            if floors:
                fmeans[name] = st.mean(floors)
                print(f"  {name:7s} accept->motion: mean {st.mean(floors):5.0f} ms "
                      f"(sd {st.pstdev(floors):.0f}); handshake {st.mean(accepts):.0f} ms")
        if "gentle" in fmeans and "fast" in fmeans:
            ramp = fmeans["gentle"] - fmeans["fast"]
            print(f"\n  ramp difference (gentle-fast) ~= {ramp:.0f} ms (the slow-ramp "
                  f"artifact, NOT startup).")
            print(f"  => pure controller startup ~= FAST floor minus its small ramp "
                  f"~= {fmeans['fast']:.0f} ms (upper bound). The rest of the 'gentle "
                  f"~378ms' was the trajectory ramping to the 0.01 rad threshold.")
        print("===========================================")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

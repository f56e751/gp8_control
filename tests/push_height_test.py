"""Interactive PUSH_HEIGHT calibration — jog the push-stroke height by eye.

The push stroke was observed hitting the TOP of cans, i.e. the real paddle
height differs from what PUSH_HEIGHT models. This tool parks the arm in the
EXACT push-stroke pose (``PushSkill._push_orientation``: push-facing yaw +
J6 twist + swing tilt) over a test point on the belt and lets you nudge the
TCP height in 1/5 mm steps while watching the paddle against a real can.

Prereqs (adv4ncr stream stack, NO app):
  1. Stop ``gp8_bringup.launch.py`` (the app would fight over the JGPC).
  2. Launch the driver stack only::

       ros2 launch motoman_bringup gp8.launch.py robot_ip:=<ip>

     (JointGroupPositionController active, /joint_states publishing,
     pendant REMOTE + no alarms.)
  3. Run this tool with the venv python::

       source /opt/ros/humble/setup.bash   # rclpy
       PYTHONPATH=$HOME/ros2_ws/src:$PYTHONPATH \
         ~/ros2_ws/src/gp8_control/.venv/bin/python \
         -m gp8_control.tests.push_height_test [--x 0.55] [--y 0.0] \
             [--start-lifted | --z 0.02]

Place a can at (--x, --y). Default push direction points at the metal bin
(PUSH_BIN_TARGET_MAP), like a real metal push.

``--start-lifted`` starts at PUSH_HEIGHT + PUSH_START_LIFT (the parked
backswing TCP height) instead of PUSH_HEIGHT — with NEUTRAL swing (rod
vertical), to verify the parked clearance: expect ~START_LIFT of air under
the paddle; pressing ``1`` (back-lean) at the same height should just about
close that gap (the tilt→dip lever, PUSH_PAD_FORE). ``--z`` sets any
explicit start height. The final 5 cm of the descent runs extra slow.

Keys:
  -/+ (or =)  height down/up 5 mm        j/u  height down/up 1 mm
  1/2/3       swing = back(-25°) / NEUTRAL(0° = contact pose) / fwd(+15°)
  s           slow swing sweep back→fwd→neutral at current height
  g           slow 12 cm mini-stroke along the push line and back
  p           print state    h  rise to hover    q  hover + quit

All motion runs at 15 % joint speed. When done, copy the printed value into
``PUSH_HEIGHT`` (skills/push_skill.py).
"""
from __future__ import annotations

import argparse
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from gp8_control.controllers.trajectory_controller import TrajectoryController
from gp8_control.robots.gp8 import GP8
from gp8_control.skills.push_skill import (
    PUSH_BIN_TARGET_MAP,
    PUSH_HEIGHT,
    PUSH_START_LIFT,
    SWING_ANGLE,
    SWING_BIAS,
    PushSkill,
)
from gp8_control.trajectory.trajectory_primitive import trajectory

HOVER_Z = 0.25          # safe approach/exit height (m)
FLOOR_Z = 0.0           # hard floor for the jog (m) — never command below
FINE_ONLY_Z = 0.03      # below this, only 1 mm steps are allowed
STROKE_LEN = 0.12       # mini-stroke length (m)
STROKE_SPEED = 0.15     # mini-stroke TCP speed (m/s) — slow, watch the contact
JOG_HZ = 50.0           # knot rate for jog trajectories (stream resamples to 4 ms)
SPEED_SCALE = 0.15      # fraction of joint velocity limits for ALL moves here
SLOW_FINAL_DZ = 0.05    # last part of the descent (m) runs extra slow ...
SLOW_SCALE = 0.3        # ... at this fraction of the (already slow) M1

SWING_BACK = SWING_BIAS - SWING_ANGLE     # stroke-start lean (parked/wait pose)
SWING_NEUTRAL = 0.0                       # contact-instant pose (impact-synced)
SWING_FWD = SWING_BIAS + SWING_ANGLE      # follow-through lean


def get_key() -> str:
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


class HeightTuner:
    def __init__(self, node: Node, x: float, y: float, bin_xy: tuple,
                 z0: float = PUSH_HEIGHT) -> None:
        self.node = node
        self.robot = GP8()
        self.ctrl = TrajectoryController(node)
        self.M1 = self.robot.velocity_limits * SPEED_SCALE
        self.M2 = self.M1 * 4.0
        self.xy = np.array([x, y], dtype=float)
        d = np.array([bin_xy[0] - x, bin_xy[1] - y, 0.0])
        self.push_dir = d / np.linalg.norm(d)
        self.z = float(z0)
        self.swing = SWING_NEUTRAL

    # ---------------- pose / motion ----------------
    def _pose(self, z: float, swing: float, along: float = 0.0) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = PushSkill._push_orientation(self.push_dir, swing)
        T[0, 3] = self.xy[0] + along * self.push_dir[0]
        T[1, 3] = self.xy[1] + along * self.push_dir[1]
        T[2, 3] = z
        return T

    def _current(self) -> np.ndarray | None:
        cj = self.ctrl.current_joints
        return None if cj is None else np.asarray(cj, dtype=float)

    def _goto(self, T: np.ndarray, m_scale: float = 1.0) -> bool:
        cur = self._current()
        if cur is None:
            print("  !! no /joint_states yet")
            return False
        q = self.robot.inverse_kinematics(T, q_init=cur)
        if q is None:
            print("  !! IK failed for that pose — not moving")
            return False
        zero = np.zeros(6)
        traj, vel, ts = trajectory(cur, zero, np.asarray(q, float), zero,
                                   self.M1 * m_scale, self.M2 * m_scale,
                                   hertz=JOG_HZ)
        self.ctrl.send_trajectory(traj, vel, ts)   # blocks while streaming
        return True

    def _follow(self, poses: list[np.ndarray], dt: float) -> bool:
        """Stream a seeded-IK Cartesian waypoint sequence at fixed dt."""
        cur = self._current()
        if cur is None:
            return False
        qs, seed = [], cur
        for T in poses:
            q = self.robot.inverse_kinematics(T, q_init=seed)
            if q is None:
                print("  !! IK failed mid-sequence — truncated")
                break
            seed = np.asarray(q, float)
            qs.append(seed)
        if len(qs) < 2:
            return False
        traj = np.column_stack(qs)
        ts = np.arange(len(qs)) * dt
        vel = np.gradient(traj, ts, axis=1)
        vel[:, 0] = 0.0
        vel[:, -1] = 0.0
        self.ctrl.send_trajectory(traj, vel, ts)
        return True

    # ---------------- user actions ----------------
    def descend(self) -> None:
        print(f"  hover over ({self.xy[0]:+.3f}, {self.xy[1]:+.3f}) ...")
        if not self._goto(self._pose(HOVER_Z, self.swing)):
            raise RuntimeError("cannot reach hover pose")
        mid = self.z + SLOW_FINAL_DZ
        if HOVER_Z > mid:
            print(f"  descending to z={mid:.3f} ...")
            self._goto(self._pose(mid, self.swing))
        print(f"  slow final approach to z={self.z:.4f} "
              f"({SPEED_SCALE * SLOW_SCALE * 100:.1f}% joint speed) ...")
        self._goto(self._pose(self.z, self.swing), m_scale=SLOW_SCALE)
        self.status()

    def nudge(self, dz: float) -> None:
        nz = self.z + dz
        if nz < FLOOR_Z:
            print(f"  !! floor guard: z stays at {self.z:.4f}")
            return
        if nz < FINE_ONLY_Z and abs(dz) > 0.0011:
            print(f"  !! below {FINE_ONLY_Z:.3f} m only 1 mm steps (u/j)")
            return
        if self._goto(self._pose(nz, self.swing)):
            self.z = nz
            self.status()

    def set_swing(self, swing: float, label: str) -> None:
        if self._goto(self._pose(self.z, swing)):
            self.swing = swing
            print(f"  swing = {label} ({np.degrees(swing):+.0f}°)")

    def sweep(self) -> None:
        print("  swing sweep back→fwd→neutral (watch the lowest paddle point)")
        angles = np.concatenate([
            np.linspace(self.swing, SWING_BACK, 20),
            np.linspace(SWING_BACK, SWING_FWD, 60),
            np.linspace(SWING_FWD, SWING_NEUTRAL, 30),
        ])
        self._follow([self._pose(self.z, a) for a in angles], dt=0.05)
        self.swing = SWING_NEUTRAL
        self.status()

    def stroke(self) -> None:
        print(f"  mini-stroke {STROKE_LEN*100:.0f}cm @ {STROKE_SPEED} m/s at z={self.z:.4f}")
        n = max(2, int(STROKE_LEN / STROKE_SPEED * 20))
        fwd = [self._pose(self.z, self.swing, along=s)
               for s in np.linspace(0.0, STROKE_LEN, n)]
        dt = (STROKE_LEN / STROKE_SPEED) / (n - 1)
        self._follow(fwd, dt)
        time.sleep(0.4)
        print("  ... returning")
        self._follow(list(reversed(fwd)), dt)
        self.status()

    def hover(self) -> None:
        self._goto(self._pose(HOVER_Z, self.swing))
        print("  at hover")

    def status(self) -> None:
        print(
            f"  >>> z = {self.z:.4f} m   swing {np.degrees(self.swing):+.0f}°   "
            f"(PUSH_HEIGHT={PUSH_HEIGHT}, parked="
            f"{PUSH_HEIGHT + PUSH_START_LIFT:.3f})"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--x", type=float, default=0.55, help="test point base X (m)")
    ap.add_argument("--y", type=float, default=0.0, help="test point base Y (m)")
    ap.add_argument("--z", type=float, default=None,
                    help="explicit start TCP height (m); default PUSH_HEIGHT")
    ap.add_argument("--start-lifted", action="store_true",
                    help="start at PUSH_HEIGHT + PUSH_START_LIFT (parked "
                         "backswing height), NEUTRAL swing (rod vertical)")
    args = ap.parse_args()

    z0 = PUSH_HEIGHT + PUSH_START_LIFT if args.start_lifted else PUSH_HEIGHT
    if args.z is not None:
        z0 = args.z

    rclpy.init()
    node = Node("push_height_test")
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()

    tuner = HeightTuner(node, args.x, args.y,
                        PUSH_BIN_TARGET_MAP.get("metal", (1.2, 0.5, 0.0)),
                        z0=z0)

    print(__doc__.split("Keys:")[1].split("All motion")[0])
    print(f"test point ({args.x:+.3f}, {args.y:+.3f}), dir "
          f"({tuner.push_dir[0]:+.2f}, {tuner.push_dir[1]:+.2f}), "
          f"start z={tuner.z:.4f}")

    print("waiting for /joint_states ...")
    t0 = time.time()
    while tuner.ctrl.current_joints is None:
        if time.time() - t0 > 10.0:
            print("no /joint_states after 10 s — is the driver stack up?")
            rclpy.shutdown()
            return
        time.sleep(0.1)

    input("place a can at the test point, clear the area, then ENTER to move "
          "(Ctrl-C to abort) ... ")
    tuner.descend()

    try:
        while True:
            k = get_key()
            if k in ("+", "="):
                tuner.nudge(+0.005)
            elif k == "-":
                tuner.nudge(-0.005)
            elif k == "u":
                tuner.nudge(+0.001)
            elif k == "j":
                tuner.nudge(-0.001)
            elif k == "1":
                tuner.set_swing(SWING_BACK, "back / parked-wait pose")
            elif k == "2":
                tuner.set_swing(SWING_NEUTRAL, "NEUTRAL / contact pose")
            elif k == "3":
                tuner.set_swing(SWING_FWD, "forward / follow-through")
            elif k == "s":
                tuner.sweep()
            elif k == "g":
                tuner.stroke()
            elif k == "p":
                tuner.status()
            elif k == "h":
                tuner.hover()
            elif k in ("q", "\x03"):
                break
    finally:
        print("\nrising to hover and exiting ...")
        try:
            tuner.hover()
        except Exception:
            pass
        print("=" * 56)
        print(f"  RESULT: set PUSH_HEIGHT = {tuner.z:.4f}   "
              f"(skills/push_skill.py)")
        print("=" * 56)
        rclpy.shutdown()


if __name__ == "__main__":
    main()

"""Hardware robot backend — **adv4ncr ros2_control port** of trajectory execution + suction I/O.

This is the HARDWARE implementation of :class:`gp8_control.backends.robot_base.RobotBackend`.
The shared **hardware operation logic** — the 250 Hz stream engine (cubic-Hermite resample
clamped to the knot envelope, grid-time suction release, tick_fn protocol, wall-clock pacing),
suction request coalescing + telemetry, the blocking ``send_trajectory_queue*`` surface, the
``pq_*`` shims and ``_wait_for_position`` — lives in the base class so the MuJoCo twin runs the
*identical* code. This file adds only what is physically hardware:

  * **Joint states** from ``/joint_states`` (adv4ncr ``joint_state_broadcaster``), reordered
    to JOINT_NAMES by name, feeding ``current_joints`` + the MotionLogger.
  * **`_emit_sample`** — publish one 4 ms joint target as a ``Float64MultiArray`` on the
    ``JointGroupPositionController`` command topic (the ``stream`` backend).
  * **Suction / IO** — adv4ncr's ros2_control host exposes NO ``/write_single_io`` service, BUT
    its controller_driver runs the legacy ros-industrial **Simple Message IoServer on TCP 50242**
    (Controller.c OpenTcpSocket(TCP_PORT_IO); IoServer.c handles ROS_MSG_MOTO_WRITE_IO_BIT=2005).
    So ``_call_io`` recovers MotoROS2's suction capability with a small PC-side TCP client — no
    controller-code or wiring change (Option A). ⚠️ UNVERIFIED ON HW: confirm port 50242 is
    reachable, address 10017 is writable, the reply resultCode is success, and the ON/OFF value
    convention (on=0/off=1) matches. On failure it logs loudly and continues (does not raise).
    ``_set_suction`` ENQUEUES the write on a dedicated worker thread so the blocking TCP
    round-trip never stalls the 250 Hz loop at the release instant.
  * **The `jtc` motion backend** (env ``GP8_ADV4NCR_BACKEND=jtc``): hand a whole trajectory to
    ``/joint_trajectory_controller/follow_joint_trajectory`` and let the JTC interpolate at
    250 Hz internally. Methods with a jtc path branch here and delegate to the base (stream)
    otherwise. **The active ros2_control controller must match the backend** (activate
    JointGroupPositionController for ``stream``, JTC for ``jtc`` — they can't both command
    the joints at once).

Public API is preserved 1:1 with the surface skills/app.py/tests actually call — construction
is still ``TrajectoryController(node)`` and every method keeps its name and signature.
"""

from __future__ import annotations

import os
import queue
import socket
import struct
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import qos_profile_sensor_data
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# Shared engine + helpers (re-exported here for existing importers).
from gp8_control.backends.robot_base import (
    STREAM_DT,
    STREAM_HZ,
    RobotBackend,
    _max_abs_diff,
    _traj_total,
)
from gp8_control.utils.motion_logger import MotionLogger

JOINT_NAMES = [
    "joint_1_s", "joint_2_l", "joint_3_u",
    "joint_4_r", "joint_5_b", "joint_6_t",
]
SUCTION_IO_ADDRESS = 10017

# adv4ncr ros2_control names (see motoman_bringup/config/controllers.yaml)
FJT_ACTION = "/joint_trajectory_controller/follow_joint_trajectory"
JOINT_STATES_TOPIC = "/joint_states"
# JointGroupPositionController command topic (position_controllers/JointGroupPositionController)
# — the 250 Hz streaming path: publish a Float64MultiArray of joint positions per cycle.
JGPC_COMMAND_TOPIC = "/JointGroupPositionController/commands"


def _wait_future(fut, timeout_sec=None):
    """Block the calling thread until ``fut`` completes WITHOUT spinning (the
    process-wide MultiThreadedExecutor is the sole spinner). Returns the result,
    or None on timeout / shutdown."""
    ev = threading.Event()
    fut.add_done_callback(lambda _f: ev.set())
    if timeout_sec is None:
        while not ev.wait(0.2):
            if not rclpy.ok():
                break
    else:
        ev.wait(timeout_sec)
    return fut.result() if fut.done() else None


def _seconds_to_duration(seconds: float) -> Duration:
    sec = int(seconds)
    nanosec = int((seconds - sec) * 1e9)
    return Duration(sec=sec, nanosec=nanosec)


class TrajectoryController(RobotBackend):
    """Hardware backend: executes joint trajectories via the adv4ncr stream/JTC;
    suction I/O is a single seam (``_call_io``) over Simple-Message TCP."""

    def __init__(self, node: Node) -> None:
        super().__init__()   # telemetry, coalescing state, qmode shims (robot_base)
        self._node = node

        self._motion_logger = MotionLogger(
            os.environ.get("GP8_MOTION_LOG_DIR"),
            logger=self._node.get_logger(),
        )

        cb_group = ReentrantCallbackGroup()

        # adv4ncr publishes joint states via joint_state_broadcaster on /joint_states.
        self._node.create_subscription(
            JointState, JOINT_STATES_TOPIC,
            self._joint_state_cb, qos_profile_sensor_data,
            callback_group=cb_group,
        )

        self._fjt_client = ActionClient(
            self._node, FollowJointTrajectory, FJT_ACTION,
            callback_group=cb_group,
        )

        # Motion backend: "stream" = gp8_control publishes joint targets to the
        # JointGroupPositionController at ~250 Hz (FCI-style servo streaming); "jtc" =
        # hand a whole trajectory to the JointTrajectoryController action (JTC interpolates
        # at 250 Hz internally). env GP8_ADV4NCR_BACKEND, default "stream".
        self._backend = os.environ.get("GP8_ADV4NCR_BACKEND", "stream").lower()
        self._jgpc_pub = self._node.create_publisher(
            Float64MultiArray, JGPC_COMMAND_TOPIC, 10)
        self._jgpc_msg = Float64MultiArray()   # reused by _emit_sample (250 Hz)

        # Suction IO: Simple Message TCP client to the controller's IoServer
        # (Option A — recovers MotoROS2 /write_single_io without changing the controller).
        # Persistent socket (MAX_IO_CONNECTIONS=1 on the controller), guarded by a lock
        # because suction can be toggled from both the main loop and the feeder thread.
        self._io_ip = os.environ.get("GP8_ROBOT_IP", "192.168.255.1")
        self._io_port = 50242            # TCP_PORT_IO (Controller.h)
        self._io_sock: socket.socket | None = None
        self._io_lock = threading.Lock()
        # Suction I/O runs on a dedicated worker thread (review finding #2): suction_on/off
        # ENQUEUE and return immediately, so a blocking TCP round-trip never stalls the
        # 250 Hz stream/servo loop at the release instant. FIFO preserves off-before-prime order.
        # (The request-coalescing state `_io_state_lock`/`_io_requested_value` lives in the base.)
        self._io_queue: "queue.Queue" = queue.Queue()
        self._io_thread = threading.Thread(target=self._io_worker, daemon=True)
        self._io_thread.start()
        self._io_closed = False

    # ------------------------------------------------------------------
    # RobotBackend primitives
    # ------------------------------------------------------------------
    def _ok(self) -> bool:
        return rclpy.ok()

    def _emit_sample(self, positions) -> None:
        self._jgpc_msg.data = [float(x) for x in positions]
        self._jgpc_pub.publish(self._jgpc_msg)

    def _set_suction(self, on: bool, requested_at: float) -> None:
        # Coalescing + telemetry already happened in the base's suction_on/off;
        # here we only hand the write to the IO worker (never block the loop).
        self._io_queue.put((SUCTION_IO_ADDRESS, 0 if on else 1, requested_at))

    # ------------------------------------------------------------------
    # Setup / state
    # ------------------------------------------------------------------
    def wait_for_servers(self, timeout_sec: float = 10.0) -> bool:
        """Wait for the JTC action server. (No IO service under adv4ncr — see _call_io.)"""
        self._node.get_logger().info(f"Waiting for JTC action server {FJT_ACTION} ...")
        if not self._fjt_client.wait_for_server(timeout_sec=timeout_sec):
            self._node.get_logger().error("JointTrajectoryController action server not available.")
            return False
        self._node.get_logger().info(
            f"[IO] suction via Simple Message TCP {self._io_ip}:{self._io_port} "
            "(UNVERIFIED on HW — see _call_io)."
        )
        self._node.get_logger().info("JTC action server ready.")
        return True

    def set_motion_op(self, label: str) -> None:
        self._motion_logger.set_label(label)

    def _joint_state_cb(self, msg: JointState) -> None:
        # joint_state_broadcaster may publish joints in any order — reorder by name.
        name_to_pos = dict(zip(msg.name, msg.position))
        name_to_vel = dict(zip(msg.name, msg.velocity)) if msg.velocity else {}
        if all(j in name_to_pos for j in JOINT_NAMES):
            self.current_joints = [float(name_to_pos[j]) for j in JOINT_NAMES]
            self.current_jointvels = [float(name_to_vel.get(j, 0.0)) for j in JOINT_NAMES]
            self._motion_logger.on_sample(self.current_joints)

    # ------------------------------------------------------------------
    # Goal building + core send (jtc backend only)
    # ------------------------------------------------------------------
    def _build_goal(self, traj, vel, timestep, final_joint=None, extra_time=0.05):
        """Build a FollowJointTrajectory goal from (n_joints, n_steps) arrays.

        NOTE: unlike the MotoROS2 queue path, we do NOT overwrite positions[0] with
        the measured current joints — the JTC handles the start-state itself (its
        configured start tolerance). If the JTC rejects on start tolerance, widen it
        in controllers.yaml rather than snapping here.
        """
        positions = np.asarray(traj).T.tolist()
        velocities = np.asarray(vel).T.tolist()
        times = list(np.asarray(timestep).tolist())

        final = list(final_joint) if final_joint is not None else positions[-1]
        positions.append(final)
        velocities.append([0.0] * len(JOINT_NAMES))
        times.append(times[-1] + extra_time)

        jt = JointTrajectory()
        jt.joint_names = JOINT_NAMES
        for pos, v, t in zip(positions, velocities, times):
            pt = JointTrajectoryPoint()
            pt.positions = [float(x) for x in pos]
            pt.velocities = [float(x) for x in v]
            pt.time_from_start = _seconds_to_duration(float(t))
            jt.points.append(pt)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = jt
        return goal

    def _send_goal(self, goal_msg):
        """Send a goal; return the accepted goal_handle (or None if rejected)."""
        fut = self._fjt_client.send_goal_async(goal_msg)
        _wait_future(fut)
        gh = fut.result()
        if gh is None or not gh.accepted:
            self._node.get_logger().warn("Trajectory goal rejected.")
            return None
        return gh

    def _send_blocking(self, traj, vel, timestep, final_joint=None) -> bool:
        """Send a trajectory and block until the JTC reports completion."""
        gh = self._send_goal(self._build_goal(traj, vel, timestep, final_joint))
        if gh is None:
            return False
        _wait_future(gh.get_result_async())
        return True

    # ------------------------------------------------------------------
    # send_trajectory family — stream delegates to the base engine; jtc
    # branches keep their original bodies verbatim.
    # ------------------------------------------------------------------
    def send_trajectory(self, traj, vel, timestep, final_joint=None) -> bool:
        if self._backend == "stream":
            return super().send_trajectory(traj, vel, timestep, final_joint)
        return self._send_blocking(traj, vel, timestep, final_joint)

    def send_trajectory_queue(self, traj, vel, timestep, final_joint=None) -> bool:
        """(MotoROS2 name kept.) backend "stream" → 250 Hz JGPC publish; "jtc" → JTC action."""
        if self._backend == "stream":
            return super().send_trajectory_queue(traj, vel, timestep, final_joint)
        return self._send_blocking(traj, vel, timestep, final_joint)

    def send_trajectory_queue_interruptible(
        self, traj, vel, timestep, final_joint, stop_requested,
    ) -> bool:
        """Stream until ``stop_requested()`` becomes true.

        Returns True when the motion stopped early.  This is intentionally
        limited to the default JGPC stream backend: cancelling a JTC action has
        different timing semantics and is not suitable for a millimetre-scale
        supervised attachment measurement.
        """
        if self._backend != "stream":
            raise RuntimeError(
                "interruptible motion requires GP8_ADV4NCR_BACKEND=stream"
            )
        return super().send_trajectory_queue_interruptible(
            traj, vel, timestep, final_joint, stop_requested)

    def send_trajectory_with_release(self, traj, vel, timestep, final_joint, release_joint) -> bool:
        return self._send_with_position_release(traj, vel, timestep, final_joint, release_joint)

    def send_trajectory_queue_with_release(self, traj, vel, timestep, final_joint, release_joint) -> bool:
        return self._send_with_position_release(traj, vel, timestep, final_joint, release_joint)

    def send_trajectory_queue_with_attach(self, traj, vel, timestep, final_joint,
                                          attach_target_joint, attach_tolerance: float = 0.05) -> bool:
        """Fire suction_on just before reaching attach_target_joint (pneumatic lead), during the JTC move."""
        gh = self._send_goal(self._build_goal(traj, vel, timestep, final_joint))
        if gh is None:
            return False
        result_fut = gh.get_result_async()
        total = _traj_total(timestep) + 0.2
        fired = False
        deadline = time.time() + total
        while not result_fut.done() and time.time() < deadline:
            if (not fired and self.current_joints is not None
                    and _max_abs_diff(self.current_joints, attach_target_joint) < attach_tolerance):
                self.suction_on()
                fired = True
            time.sleep(0.01)
        if not fired:
            self.suction_on()
        _wait_future(result_fut, 2.0)
        return True

    def _send_with_position_release(self, traj, vel, timestep, final_joint, release_joint) -> bool:
        """Send a JTC goal; fire suction_off when the arm reaches release_joint during the move."""
        gh = self._send_goal(self._build_goal(traj, vel, timestep, final_joint))
        if gh is None:
            return False
        result_fut = gh.get_result_async()
        timeout = _traj_total(timestep) + 0.1
        reached = self._wait_for_position(release_joint, tolerance=0.05, timeout_sec=timeout)
        self.suction_off()
        if not reached:
            self._node.get_logger().warn("Release joint not detected; suction_off on timeout fallback.")
        _wait_future(result_fut, 2.0)
        return True

    def send_trajectory_queue_with_timed_release(self, traj, vel, timestep, final_joint,
                                                 release_index: int,
                                                 suction_on_at: float | None = None) -> bool:
        """Throw path: fire suction_off when the arm reaches the release WAYPOINT pose during the
        JTC move, then (optionally) prime the next pick's vacuum once ``suction_on_at`` passes.
        Returns True iff the next-pick suction was primed.

        (MotoROS2 used the index during the per-point push; under JTC there is no push, so we
        watch /joint_states for the release POSE = traj[:, release_index] — position-based.
        Under ``stream`` the base engine fires purely on grid time.)
        """
        if self._backend == "stream":
            return super().send_trajectory_queue_with_timed_release(
                traj, vel, timestep, final_joint, release_index, suction_on_at=suction_on_at)
        arr = np.asarray(traj)
        rel = int(max(0, min(release_index, arr.shape[1] - 1)))
        release_pose = [float(x) for x in arr[:, rel]]
        _times = np.asarray(timestep, dtype=float).ravel()
        t_release = float(_times[rel]) if rel < _times.size else float(_times[-1])
        total = _traj_total(timestep)

        state = {"fired": False, "primed_next": False}

        gh = self._send_goal(self._build_goal(traj, vel, timestep, final_joint))
        if gh is None:
            self._last_throw_ok = False   # #R3: propagate rejection to pq_throw_segment's ok
            return False
        # #R2: start the release clock AFTER goal acceptance (~motion start), so the goal-dispatch
        # round-trip latency isn't counted into t_release (which would fire the time backstop early).
        t_start = time.time()
        self._last_throw_ok = True
        self.last_throw = {
            "throw_start": t_start, "release_wall": None, "io_ms": None,
            "release_index": rel, "n_waypoints": int(arr.shape[1]),
        }
        result_fut = gh.get_result_async()
        deadline = t_start + total + 0.3

        while not result_fut.done() and time.time() < deadline:
            # Release on EITHER reaching the release pose OR the release TIME elapsing (#B):
            # a fast throw can sweep through the pose between 4 ms polls; the time backstop
            # guarantees we don't miss it and carry the object to the next intercept.
            if (not state["fired"]
                    and (((self.current_joints is not None)
                          and _max_abs_diff(self.current_joints, release_pose) <= 0.05)
                         or (time.time() - t_start) >= t_release)):
                t_io = time.time()
                self.suction_off()
                state["fired"] = True
                self.last_throw["release_wall"] = self.last_suction_off_t
                self.last_throw["io_ms"] = (time.time() - t_io) * 1000.0
                self._node.get_logger().info(f"Release: suction_off at waypoint {rel}.")
            elif (state["fired"] and suction_on_at is not None and not state["primed_next"]
                  and time.time() >= suction_on_at):
                self.suction_on()
                state["primed_next"] = True
                self._node.get_logger().info("Return-prime: suction_on for next pick.")
            time.sleep(0.004)

        if not state["fired"]:
            self.suction_off()   # fallback: never carry the object past the release
            state["fired"] = True
        _wait_future(result_fut, 2.0)
        self._wait_for_position(final_joint, tolerance=0.03, timeout_sec=1.5)
        return state["primed_next"]

    def send_trajectory_queue_timed_suction(self, traj, vel, timestep, final_joint,
                                            suction_on_at: float) -> bool:
        """Positioning move that fires suction_ON once wall-clock ``suction_on_at`` passes
        (priming the vacuum before grasp). Returns True iff suction fired before the move ended."""
        if self._backend == "stream":
            return super().send_trajectory_queue_timed_suction(
                traj, vel, timestep, final_joint, suction_on_at)
        gh = self._send_goal(self._build_goal(traj, vel, timestep, final_joint))
        if gh is None:
            return False
        result_fut = gh.get_result_async()
        total = _traj_total(timestep) + 0.3
        deadline = time.time() + total
        fired = False
        while not result_fut.done() and time.time() < deadline:
            if not fired and time.time() >= suction_on_at:
                self.suction_on()
                fired = True
            time.sleep(0.01)
        if not fired and time.time() >= suction_on_at:
            self.suction_on()
            fired = True
        _wait_future(result_fut, 2.0)
        return fired

    # ------------------------------------------------------------------
    # Persistent-session shims with a jtc branch (stream lives in the base)
    # ------------------------------------------------------------------
    def pq_segment(self, traj, vel, ts, final_joint, *, is_last: bool = False,
                   join_tol: float = 0.05, between_fn=None) -> tuple:
        """Execute one segment (stream: 250 Hz JGPC publish; jtc: JTC goal); call ``between_fn``
        each cycle (used to fire suction_on when its wall-clock instant passes). Returns (ok, codes)."""
        if self._backend == "stream":
            return super().pq_segment(traj, vel, ts, final_joint,
                                      is_last=is_last, join_tol=join_tol, between_fn=between_fn)
        gh = self._send_goal(self._build_goal(traj, vel, ts, final_joint))
        if gh is None:
            return (False, [])
        result_fut = gh.get_result_async()
        total = _traj_total(ts) + 0.3
        deadline = time.time() + total
        while not result_fut.done() and time.time() < deadline:
            if between_fn is not None:
                between_fn()
            time.sleep(0.01)
        if between_fn is not None:
            between_fn()
        _wait_future(result_fut, 2.0)
        return (True, [])

    def pq_hold_until(self, hold_joint, deadline_wall: float, *,
                      dt: float = 0.12, lead: float = 0.35, tick_fn=None) -> bool:
        """Wait until wall-clock ``deadline_wall`` at the current (hold) pose. Under JTC there
        is no queue to keep alive — the arm simply holds the last commanded pose — so this is a
        responsive sleep that calls ``tick_fn`` each iteration (e.g. to prime suction on time)."""
        if self._backend == "stream":
            return super().pq_hold_until(hold_joint, deadline_wall,
                                         dt=dt, lead=lead, tick_fn=tick_fn)
        while time.time() < deadline_wall and rclpy.ok():
            if tick_fn is not None:
                tick_fn()
            time.sleep(0.01)
        return True

    # ------------------------------------------------------------------
    # Suction IO transport (worker thread + Simple Message TCP client)
    # ------------------------------------------------------------------
    def close(self, timeout_sec: float = 5.0) -> None:
        """Drain pending suction writes and stop the IO worker cleanly.

        The sentinel is queued after all prior writes, so a final ``suction_off``
        is transmitted before the worker exits.  Explicit shutdown also avoids
        leaving a Python thread inside rclpy logging while the ROS node and the
        interpreter are being destroyed.
        """
        if self._io_closed:
            return
        self._io_closed = True
        self._io_queue.put(None)
        self._io_thread.join(timeout=max(0.0, float(timeout_sec)))
        if self._io_thread.is_alive():
            self._node.get_logger().warn(
                "[IO] worker did not stop before shutdown timeout"
            )
            return
        with self._io_lock:
            if self._io_sock is not None:
                try:
                    self._io_sock.close()
                except OSError:
                    pass
                self._io_sock = None

    def _io_worker(self) -> None:
        """Daemon: drain the IO queue and perform the blocking Simple Message TCP writes, so the
        250 Hz stream/servo loop is NEVER blocked by a suction round-trip (review finding #2).
        FIFO order preserves off-before-prime. An IO error is logged, never kills the worker."""
        while True:
            item = self._io_queue.get()
            if item is None:
                break
            addr, val, requested_at = item
            started_at = time.time()
            try:
                applied = self._call_io(addr, val)
                finished_at = time.time()
                if applied:
                    if val == 0:
                        self.last_suction_on_ack_t = finished_at
                        state = "ON"
                    else:
                        self.last_suction_off_ack_t = finished_at
                        state = "OFF"
                    self._node.get_logger().info(
                        f"[IO] suction {state} controller-ack: "
                        f"queue={(started_at - requested_at) * 1000.0:.1f}ms "
                        f"roundtrip={(finished_at - started_at) * 1000.0:.1f}ms "
                        f"total={(finished_at - requested_at) * 1000.0:.1f}ms"
                    )
                else:
                    # Permit a later same-state call to retry after both TCP
                    # attempts failed.  Do not overwrite a newer opposite state.
                    with self._io_state_lock:
                        if self._io_requested_value == val:
                            self._io_requested_value = None
            except Exception as e:
                self._node.get_logger().error(f"[IO] worker write addr={addr} val={val} error: {e}")

    def _call_io(self, address: int, value: int) -> bool:
        """Write a single IO bit via the controller's Simple Message IoServer (TCP 50242).

        Option A: adv4ncr has no /write_single_io ROS service, but its controller_driver
        runs the legacy ros-industrial Simple Message IoServer (Controller.c
        OpenTcpSocket(TCP_PORT_IO); IoServer.c handles ROS_MSG_MOTO_WRITE_IO_BIT=2005), so
        we recover suction with a small TCP client — no controller-code or wiring change.

        Wire format (little-endian, ``__packed__``):
          prefix int32 = len(header+body) = 20
          header: msgType int32 = 2005, commType int32 = 2 (SERVICE_REQUEST), replyType int32 = 0
          body:   ioAddress uint32, ioValue uint32
        Reply: prefix(4) + header(12) + resultCode int32 @ offset 16.

        ⚠️ UNVERIFIED ON HW. Confirm: port reachable; address 10017 writable; the reply
        resultCode's success value (IoResultCodes); ON=0/OFF=1 convention. On failure this
        logs loudly and returns (does NOT raise) so the motion loop isn't killed mid-cycle —
        but a failed suction release IS a safety concern, so it must be caught in low-speed
        validation before any real pick/throw.
        """
        pkt = struct.pack("<iiiiII", 20, 2005, 2, 0, int(address), int(value))
        with self._io_lock:
            for attempt in (0, 1):   # one reconnect retry
                try:
                    if self._io_sock is None:
                        self._io_sock = socket.create_connection(
                            (self._io_ip, self._io_port), timeout=2.0)
                        self._io_sock.settimeout(2.0)
                    self._io_sock.sendall(pkt)
                    reply = self._io_sock.recv(64)
                    if len(reply) >= 20:
                        result_code = struct.unpack_from("<i", reply, 16)[0]
                        # IoResultCodes success value is controller-defined — log for HW verify.
                        self._node.get_logger().debug(
                            f"[IO] write addr={address} val={value} -> resultCode={result_code}")
                    else:
                        self._node.get_logger().warn(
                            f"[IO] short/no reply ({len(reply)} B) for addr={address} val={value}")
                        return False
                    return True
                except OSError as e:
                    self._node.get_logger().warn(
                        f"[IO] TCP write failed (attempt {attempt}) to {self._io_ip}:{self._io_port}: "
                        f"{e}; reconnecting.")
                    try:
                        if self._io_sock is not None:
                            self._io_sock.close()
                    except OSError:
                        pass
                    self._io_sock = None
            self._node.get_logger().error(
                f"[IO] suction write addr={address} val={value} FAILED "
                f"({self._io_ip}:{self._io_port}) — SAFETY: verify release before any real throw.")
            return False

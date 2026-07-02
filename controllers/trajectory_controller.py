"""Low-level trajectory execution + suction I/O — **adv4ncr ros2_control port**.

⚠️ UNTESTED SCAFFOLDING. This is a re-implementation of the MotoROS2 TrajectoryController
for the **adv4ncr/motoman_ROS2** backend (ros2_control + JointTrajectoryController), on
branch ``feat/adv4ncr-ros2control-port``. It has NOT run on hardware; the adv4ncr `.out`
must be loaded + low-speed-validated first (see the safety-review workspace's DEPLOYMENT.md).

What changed vs the MotoROS2 version (and why it is much smaller):
  * **No Point Queue Mode.** ros2_control's JointTrajectoryController (JTC) accepts a whole
    trajectory via a FollowJointTrajectory action and executes it. So the entire queue-mode
    machinery (enter/exit_queue_mode, per-point queue_traj_point push, code-204 first-point
    snapping, pq_* re-entry avoidance) is GONE. ``enter_queue_mode``/``exit_queue_mode`` are
    kept as no-op shims (JTC is always ready) so callers don't change; ``pq_*`` are kept as
    thin JTC-backed shims (persistent-queue existed only to avoid re-entry, which JTC removes).
  * **Joint states** come from ``/joint_states`` (adv4ncr ``joint_state_broadcaster``), not the
    MotoROS2 bridge's ``/joint_states_urdf``. Reordered to JOINT_NAMES by name.
  * **Two motion backends** (env ``GP8_ADV4NCR_BACKEND``, default ``stream``):
      - ``stream`` — gp8_control resamples each trajectory to a 4 ms grid and **publishes joint
        targets to the ``JointGroupPositionController`` at ~250 Hz** (FCI-style servo streaming;
        soft-RT in Python — see _stream_trajectory). Also feeds hold poses during the ambush wait.
      - ``jtc`` — hand a whole trajectory to ``/joint_trajectory_controller/follow_joint_trajectory``
        and let the JTC interpolate at 250 Hz internally.
    Both drive the robot at the controller's 250 Hz; ``stream`` additionally lets the app change the
    target every cycle (needed only for real-time re-targeting). **The active ros2_control controller
    must match the backend** (activate JointGroupPositionController for ``stream``, JTC for ``jtc`` —
    they can't both command the joints at once).
  * **Suction / IO** — adv4ncr's ros2_control host exposes NO ``/write_single_io`` service, BUT
    its controller_driver runs the legacy ros-industrial **Simple Message IoServer on TCP 50242**
    (Controller.c OpenTcpSocket(TCP_PORT_IO); IoServer.c handles ROS_MSG_MOTO_WRITE_IO_BIT=2005).
    So ``_call_io`` recovers MotoROS2's suction capability with a small PC-side TCP client — no
    controller-code or wiring change (Option A). ⚠️ UNVERIFIED ON HW: confirm port 50242 is
    reachable, address 10017 is writable, the reply resultCode is success, and the ON/OFF value
    convention (on=0/off=1) matches. On failure it logs loudly and continues (does not raise).

Public API is preserved 1:1 with the surface skills/app.py actually call, so no skill/app
changes are needed to try this backend.
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
STREAM_HZ = 250.0
STREAM_DT = 1.0 / STREAM_HZ   # 4 ms


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


def _max_abs_diff(a, b) -> float:
    n = min(len(a), len(b))
    return max((abs(float(a[i]) - float(b[i])) for i in range(n)), default=0.0)


def _traj_total(timestep) -> float:
    """Total trajectory duration. ``timestep`` from trajectory() is CUMULATIVE time_from_start
    (arange(L+1)/hertz), so the duration is the LAST value — NOT np.sum(timestep) (which is
    ~T*(n+1)/2 and grossly inflates timeouts/deadlines). Review finding #3/#4."""
    t = np.asarray(timestep, dtype=float).ravel()
    return float(t[-1]) if t.size else 0.0


class TrajectoryController:
    """Executes joint trajectories via the adv4ncr JTC action; suction I/O is a
    single seam (``_call_io``) that is currently a hard blocker (no adv4ncr IO service)."""

    def __init__(self, node: Node) -> None:
        self._node = node
        self.current_joints: list | None = None
        self.current_jointvels: list | None = None

        # Pick-cycle timing telemetry (kept for the skills' logs).
        self.last_suction_on_t: float | None = None
        self.last_suction_off_t: float | None = None
        self.last_throw: dict | None = None
        self._last_throw_ok: bool = True   # #R3: last timed-release dispatch accepted? (pq_throw_segment ok)

        # Kept for API compatibility with the skills' timeline model
        # (push_skill.t_to_contact reads qmode_ms_avg for T_setup). Under JTC there is
        # NO queue-mode re-entry; the only fixed overhead is the action dispatch + the
        # controller's ~40 ms dead-time, so seed it small instead of the old ~400 ms.
        self.last_qmode_ms: float | None = None
        self.qmode_ms_avg: float = 40.0

        # Persistent-session state (pq_*). Under JTC there is no queue to keep alive;
        # these just track "a pick+hold+throw sequence is in progress".
        self._pq_active: bool = False

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
        self._io_queue: "queue.Queue" = queue.Queue()
        self._io_thread = threading.Thread(target=self._io_worker, daemon=True)
        self._io_thread.start()

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
    # Goal building + core send (all trajectory methods route here)
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
    # 250 Hz streaming backend (JointGroupPositionController)
    # ------------------------------------------------------------------
    def _stream_trajectory(self, traj, vel, timestep, final_joint=None, *,
                           release_index: int | None = None,
                           suction_on_at: float | None = None,
                           tick_fn=None) -> dict:
        """Stream a trajectory to the JointGroupPositionController at ~250 Hz.

        The app's trajectories are sampled at ~20 Hz (``trajectory()``: ``timestep`` is the
        cumulative time_from_start = arange(L+1)/hertz). The 250 Hz ros2_control loop wants a
        fresh position every 4 ms, so we RESAMPLE (per-joint linear interpolation) onto a 4 ms
        grid and publish each sample, paced to wall clock.

        ⚠️ SOFT real-time: this is a Python paced loop (GIL / scheduler jitter), NOT hard 250 Hz.
        The controller runs the true 250 Hz; the JGPC zero-order-holds our last command between
        publishes, and the B6 host command_limiter + controller-side clamp bound any per-cycle
        jump — so jitter degrades smoothness, not safety. For hard-RT, a C++ node would be needed.

        ``release_index`` → fire suction_off when the stream reaches that waypoint's time.
        ``suction_on_at`` (wall clock) → after release, fire suction_on once to prime the next pick.
        ``tick_fn`` → called every sample (e.g. to prime suction on a positioning move).
        Returns {'fired': bool, 'primed_next': bool}.
        """
        arr = np.asarray(traj, dtype=float)                       # (n_joints, n_steps)
        varr = np.asarray(vel, dtype=float)                       # velocity profile (same shape)
        n_joints, n_steps = arr.shape
        times = np.asarray(timestep, dtype=float).ravel()         # cumulative time_from_start
        t_release = float(times[int(release_index)]) if release_index is not None else None
        if final_joint is not None:
            arr = np.concatenate(
                [arr, np.asarray(final_joint, dtype=float).reshape(n_joints, 1)], axis=1)
            varr = np.concatenate([varr, np.zeros((n_joints, 1))], axis=1)   # settle at rest
            times = np.append(times, times[-1] + 0.05)
        T = float(times[-1])
        grid = np.arange(0.0, T + STREAM_DT, STREAM_DT)
        # Resample onto the 4ms grid HONORING the knot velocities (#R3: a plain linear resample of
        # the coarse ~20Hz arc yields chord-slope velocity, not the designed NN release speed —
        # matters because throw range ~ v^2). Cubic-Hermite uses positions+velocities; fall back to
        # linear if times aren't strictly increasing (degenerate/segment-join) or scipy is missing.
        if varr.shape == arr.shape and np.all(np.diff(times) > 0):
            try:
                from scipy.interpolate import CubicHermiteSpline
                samples = np.column_stack(
                    [CubicHermiteSpline(times, arr[j], varr[j])(grid) for j in range(n_joints)])
            except Exception:
                samples = np.column_stack([np.interp(grid, times, arr[j]) for j in range(n_joints)])
        else:
            samples = np.column_stack([np.interp(grid, times, arr[j]) for j in range(n_joints)])

        st = {"fired": False, "primed_next": False}
        msg = Float64MultiArray()
        start = time.monotonic()      # pacing clock (monotonic)
        wall_start = time.time()      # telemetry clock (wall) — last_throw must stay wall-clock (#3)
        for k in range(samples.shape[0]):
            if not rclpy.ok():
                break
            msg.data = [float(x) for x in samples[k]]
            self._jgpc_pub.publish(msg)

            if (t_release is not None and not st["fired"] and grid[k] >= t_release):
                t_io = time.time()
                self.suction_off()
                st["fired"] = True
                self.last_throw = {
                    "throw_start": wall_start, "release_wall": self.last_suction_off_t,
                    "io_ms": (time.time() - t_io) * 1000.0,
                    "release_index": int(release_index), "n_waypoints": int(n_steps),
                }
            elif (st["fired"] and suction_on_at is not None and not st["primed_next"]
                  and time.time() >= suction_on_at):
                self.suction_on()
                st["primed_next"] = True
            if tick_fn is not None:
                tick_fn()

            # pace to the next 4 ms tick (drop no samples; sleep the remainder)
            dt_sleep = (start + (k + 1) * STREAM_DT) - time.monotonic()
            if dt_sleep > 0:
                time.sleep(dt_sleep)
        return st

    def _stream_hold(self, hold_joint, deadline_wall: float, tick_fn=None) -> None:
        """Keep publishing ``hold_joint`` at ~250 Hz until wall-clock ``deadline_wall``
        (feeds the servo during an ambush hold). tick_fn runs each cycle (e.g. prime suction)."""
        pose = [float(x) for x in hold_joint]
        msg = Float64MultiArray()
        msg.data = pose
        nxt = time.monotonic()
        while time.time() < deadline_wall and rclpy.ok():
            self._jgpc_pub.publish(msg)
            if tick_fn is not None:
                tick_fn()
            nxt += STREAM_DT
            dt_sleep = nxt - time.monotonic()
            if dt_sleep > 0:
                time.sleep(dt_sleep)
            else:
                nxt = time.monotonic()

    # ------------------------------------------------------------------
    # send_trajectory family (all -> JTC; no queue mode)
    # ------------------------------------------------------------------
    def send_trajectory(self, traj, vel, timestep, final_joint=None) -> bool:
        if self._backend == "stream":
            self._stream_trajectory(traj, vel, timestep, final_joint)
            return True
        return self._send_blocking(traj, vel, timestep, final_joint)

    def send_trajectory_queue(self, traj, vel, timestep, final_joint=None) -> bool:
        """(MotoROS2 name kept.) backend "stream" → 250 Hz JGPC publish; "jtc" → JTC action."""
        if self._backend == "stream":
            self._stream_trajectory(traj, vel, timestep, final_joint)
            return True
        return self._send_blocking(traj, vel, timestep, final_joint)

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
        watch /joint_states for the release POSE = traj[:, release_index] — position-based.)
        """
        if self._backend == "stream":
            arr = np.asarray(traj)
            rel = int(max(0, min(release_index, arr.shape[1] - 1)))
            st = self._stream_trajectory(traj, vel, timestep, final_joint,
                                         release_index=rel, suction_on_at=suction_on_at)
            if not st["fired"]:
                self.suction_off()   # fallback: never carry the object past release
            self._last_throw_ok = True   # stream publish never "rejects" (#R3)
            return st["primed_next"]
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
            fired = {"v": False}

            def _tick() -> None:
                if not fired["v"] and time.time() >= suction_on_at:
                    self.suction_on()
                    fired["v"] = True

            self._stream_trajectory(traj, vel, timestep, final_joint, tick_fn=_tick)
            if not fired["v"] and time.time() >= suction_on_at:
                self.suction_on()
                fired["v"] = True
            return fired["v"]
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
    # Queue-mode shims (no-ops under ros2_control JTC)
    # ------------------------------------------------------------------
    def enter_queue_mode(self) -> bool:
        """No-op under ros2_control: the JTC is always ready (no Point Queue Mode)."""
        return True

    def exit_queue_mode(self) -> bool:
        """No-op under ros2_control."""
        return True

    # ------------------------------------------------------------------
    # Persistent-session shims (pq_*). JTC needs no queue keep-alive, so these
    # are thin: a segment is one JTC goal; a hold is a wall-clock wait.
    # ------------------------------------------------------------------
    @property
    def pq_active(self) -> bool:
        """True while a persistent session is open. **@property** (read as an attribute,
        not called) — callers do `traj_ctrl.pq_active` without parens (app.py, throw_skill.py);
        a plain method would evaluate truthy always and defeat the session-live guard (#A)."""
        return self._pq_active

    def pq_begin(self) -> None:
        self._pq_active = True

    def pq_segment(self, traj, vel, ts, final_joint, *, is_last: bool = False,
                   join_tol: float = 0.05, between_fn=None) -> tuple:
        """Execute one segment (stream: 250 Hz JGPC publish; jtc: JTC goal); call ``between_fn``
        each cycle (used to fire suction_on when its wall-clock instant passes). Returns (ok, codes)."""
        if self._backend == "stream":
            self._stream_trajectory(traj, vel, ts, final_joint, tick_fn=between_fn)
            return (True, [])
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
            self._stream_hold(hold_joint, deadline_wall, tick_fn=tick_fn)
            return True
        while time.time() < deadline_wall and rclpy.ok():
            if tick_fn is not None:
                tick_fn()
            time.sleep(0.01)
        return True

    def pq_throw_segment(self, traj, vel, timestep, final_joint, *,
                         release_index, suction_on_at=None,
                         release_tol: float = 0.05) -> tuple:
        """Throw with a POSITION/time-based suction release (same logic as
        send_trajectory_queue_with_timed_release). Returns ``(ok, primed_next)`` — a 2-tuple to
        match the caller's ``ok_throw, primed_next = ...`` unpack (throw_skill.py) and the sibling
        pq_segment's (ok, ...) contract (#1). ok is True once the segment is dispatched; a jtc-path
        goal rejection is not distinguished here (consistent with pq_segment's optimistic ok)."""
        primed_next = self.send_trajectory_queue_with_timed_release(
            traj, vel, timestep, final_joint, release_index, suction_on_at=suction_on_at,
        )
        return (self._last_throw_ok, bool(primed_next))

    def pq_finish(self, *, wait: bool = True, tail_buffer: float = 0.3,
                  settle_tol: float = 0.03) -> None:
        """End the session. (JTC already blocks per-segment, so nothing to drain.)"""
        self._pq_active = False

    # ------------------------------------------------------------------
    # Waiting helpers
    # ------------------------------------------------------------------
    def _wait_for_position(self, target_joint, tolerance: float = 1e-4,
                           timeout_sec: float = 5.0) -> bool:
        """Block until the arm is within ``tolerance`` (max-abs) of ``target_joint``."""
        target = [float(x) for x in target_joint]
        t_end = time.time() + timeout_sec
        while time.time() < t_end and rclpy.ok():
            if self.current_joints is not None and _max_abs_diff(self.current_joints, target) <= tolerance:
                return True
            time.sleep(0.005)
        return False

    def _wait_trajectory_end(self, total_duration: float, t_start: float | None = None) -> None:
        t0 = t_start if t_start is not None else time.time()
        remaining = total_duration - (time.time() - t0)
        if remaining > 0:
            time.sleep(remaining)

    # ------------------------------------------------------------------
    # Suction / IO  —  SINGLE BLOCKER SEAM
    # ------------------------------------------------------------------
    def suction_on(self) -> None:
        """Enqueue suction ON (non-blocking); the IO worker does the TCP write off the servo loop (#2)."""
        self.last_suction_on_t = time.time()
        self._io_queue.put((SUCTION_IO_ADDRESS, 0))

    def suction_off(self) -> None:
        """Enqueue suction OFF (non-blocking); the IO worker does the TCP write off the servo loop (#2)."""
        self.last_suction_off_t = time.time()
        self._io_queue.put((SUCTION_IO_ADDRESS, 1))

    def _io_worker(self) -> None:
        """Daemon: drain the IO queue and perform the blocking Simple Message TCP writes, so the
        250 Hz stream/servo loop is NEVER blocked by a suction round-trip (review finding #2).
        FIFO order preserves off-before-prime. An IO error is logged, never kills the worker."""
        while True:
            item = self._io_queue.get()
            if item is None:
                break
            addr, val = item
            try:
                self._call_io(addr, val)
            except Exception as e:
                self._node.get_logger().error(f"[IO] worker write addr={addr} val={val} error: {e}")

    def _call_io(self, address: int, value: int) -> None:
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
                    return
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

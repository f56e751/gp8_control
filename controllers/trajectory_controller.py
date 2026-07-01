"""Low-level trajectory execution and suction I/O control (ROS 2).

Uses FollowJointTrajectory action for trajectory execution and
WriteSingleIO service for suction gripper control via MotoROS2.
"""

from __future__ import annotations

import math
import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from builtin_interfaces.msg import Duration
from rclpy.qos import qos_profile_sensor_data
from control_msgs.action import FollowJointTrajectory
from motoros2_interfaces.srv import (
    QueueTrajPoint,
    ResetError,
    StartPointQueueMode,
    StartTrajMode,
    WriteSingleIO,
)
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from gp8_control.utils.motion_logger import MotionLogger

JOINT_NAMES = [
    "joint_1_s", "joint_2_l", "joint_3_u",
    "joint_4_r", "joint_5_b", "joint_6_t",
]
SUCTION_IO_ADDRESS = 10017


def _seconds_to_duration(seconds: float) -> Duration:
    """Convert float seconds to builtin_interfaces Duration."""
    sec = int(seconds)
    nanosec = int((seconds - sec) * 1e9)
    return Duration(sec=sec, nanosec=nanosec)


def _max_abs_diff(a, b) -> float:
    """Max per-element |a_i - b_i| over the shared length (joint-space distance)."""
    n = min(len(a), len(b))
    return max((abs(float(a[i]) - float(b[i])) for i in range(n)), default=0.0)


class TrajectoryController:
    """Executes joint trajectories and controls suction gripper I/O via ROS 2."""

    def __init__(self, node: Node) -> None:
        self._node = node
        self.current_joints: list | None = None
        self.current_jointvels: list | None = None
        self._jmon = None   # [QMODE-DBG] (min,max) joint tracker during a mode switch
        # Measured point-queue-mode re-entry cost (the dominant fixed overhead the
        # timeline model — SkillContext.earliest_reachable_intercept via the skills'
        # t_to_contact — must account for). last = most recent attempt; avg = EMA so
        # a one-off slow switch (e.g. a reset_error retry) doesn't spike the estimate.
        # Seeded to ~0.4 s so cycle 1 has a sane non-zero T_setup before any measurement.
        self.last_qmode_ms: float | None = None
        self.qmode_ms_avg: float = 400.0

        # Persistent-queue session state (Stage C). One queue session spans pick +
        # ambush hold + throw with NO per-segment re-entry (~0.41s each). _pq_t is
        # the running time_from_start of the last queued point (the monotonic
        # session timeline); _pq_last_pos the last queued joint pose; _pq_start the
        # wall clock when the session began (for completion waits). See pq_begin/
        # pq_segment/pq_hold_until/pq_finish.
        self._pq_active: bool = False
        self._pq_t: float = 0.0
        self._pq_last_pos: list | None = None
        self._pq_start: float = 0.0
        # Session start pose + the wall time the arm ACTUALLY first moved (detected
        # during the first pushed segment). pq_hold_until paces against motion-start
        # (not pq_begin) so the hold buffer is ~lead, not lead + startup-dead-time.
        self._pq_start_pose: list | None = None
        self._pq_motion_start: float | None = None

        # Opt-in diagnostic logger: predicted (planned trajectory duration) vs
        # ACTUAL move time + actual joint trace, to chase per-cycle timing drift.
        # No-op unless GP8_MOTION_LOG_DIR is set; never affects control. See
        # gp8_control.utils.motion_logger.
        self._motion_logger = MotionLogger(
            os.environ.get("GP8_MOTION_LOG_DIR"),
            logger=self._node.get_logger(),
        )

        cb_group = ReentrantCallbackGroup()

        # Joint state subscriber — bridge republishes with URDF names
        self._node.create_subscription(
            JointState, "/joint_states_urdf",
            self._joint_state_cb, qos_profile_sensor_data,
            callback_group=cb_group,
        )

        # FollowJointTrajectory action client — bridge proxy
        self._fjt_client = ActionClient(
            self._node,
            FollowJointTrajectory,
            "/motoman_gp8_controller/follow_joint_trajectory",
            callback_group=cb_group,
        )

        # WriteSingleIO service client (MotoROS2)
        self._io_client = self._node.create_client(
            WriteSingleIO, "/write_single_io",
            callback_group=cb_group,
        )

        # Queue Mode clients ---------------------------------------------
        self._stop_traj_client = self._node.create_client(
            Trigger, "/stop_traj_mode",
            callback_group=cb_group,
        )
        self._start_traj_client = self._node.create_client(
            StartTrajMode, "/start_traj_mode",
            callback_group=cb_group,
        )
        self._start_queue_client = self._node.create_client(
            StartPointQueueMode, "/start_point_queue_mode",
            callback_group=cb_group,
        )
        # bridge가 URDF→raw 번역해서 MotoROS2 /queue_traj_point로 포워딩
        self._queue_point_client = self._node.create_client(
            QueueTrajPoint, "/motoman_gp8_controller/queue_traj_point",
            callback_group=cb_group,
        )
        # MotoROS2 alarm/error reset — used to auto-recover from a controller
        # alarm (e.g. 4414 excessive segment velocity) that otherwise blocks
        # all subsequent start_point_queue_mode calls with "active Alarm".
        self._reset_error_client = self._node.create_client(
            ResetError, "/reset_error",
            callback_group=cb_group,
        )

    def wait_for_servers(self, timeout_sec: float = 10.0) -> bool:
        """Wait for action server and I/O service to become available."""
        self._node.get_logger().info("Waiting for FollowJointTrajectory action server...")
        if not self._fjt_client.wait_for_server(timeout_sec=timeout_sec):
            self._node.get_logger().error("FollowJointTrajectory server not available.")
            return False
        self._node.get_logger().info("Waiting for WriteSingleIO service...")
        if not self._io_client.wait_for_service(timeout_sec=timeout_sec):
            self._node.get_logger().error("WriteSingleIO service not available.")
            return False
        self._node.get_logger().info("All servers ready.")
        return True

    # ------------------------------------------------------------------
    # Joint state
    # ------------------------------------------------------------------

    def set_motion_op(self, label: str) -> None:
        """Tag subsequent queued commands with an op label (e.g. skill name) for the
        diagnostic MotionLogger's CSV. No-op unless GP8_MOTION_LOG_DIR is set."""
        self._motion_logger.set_label(label)

    def _joint_state_cb(self, msg: JointState) -> None:
        self.current_joints = list(msg.position)
        self.current_jointvels = list(msg.velocity)
        self._motion_logger.on_sample(self.current_joints)   # diag (no-op if off)
        # [QMODE-DBG] track joint excursion during a mode switch (catches a
        # transient up-down bobble even when the net move is ~0).
        jmon = self._jmon
        if jmon is not None:
            lo, hi = jmon
            for i, p in enumerate(self.current_joints):
                if i < len(lo):
                    if p < lo[i]:
                        lo[i] = p
                    if p > hi[i]:
                        hi[i] = p

    # ------------------------------------------------------------------
    # Trajectory execution
    # ------------------------------------------------------------------

    def send_trajectory(
        self,
        traj: np.ndarray,
        vel: np.ndarray,
        timestep: np.ndarray,
        final_joint: np.ndarray | None = None,
    ) -> bool:
        """Send trajectory via FollowJointTrajectory action and wait for completion."""
        goal_msg = self._build_goal(traj, vel, timestep, final_joint=final_joint)
        future = self._fjt_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self._node, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self._node.get_logger().warn("Trajectory goal rejected.")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self._node, result_future)
        return True

    def send_trajectory_with_release(
        self,
        traj: np.ndarray,
        vel: np.ndarray,
        timestep: np.ndarray,
        final_joint: np.ndarray,
        release_joint: np.ndarray,
    ) -> bool:
        """Send trajectory, release suction at specific joint position, wait for completion.

        Args:
            release_joint: Joint position at which suction should be turned OFF.
        """
        goal_msg = self._build_goal(traj, vel, timestep, final_joint=final_joint)

        # Send goal (non-blocking)
        future = self._fjt_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self._node, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self._node.get_logger().warn("Trajectory goal rejected.")
            return False

        # Throw trajectory assumes the object is already grasped — do not
        # toggle suction on here. Only release it mid-flight at release_joint.
        #
        # A hard timeout (full trajectory duration) guarantees that suction
        # is turned off even if joint-based detection misses the release
        # point (fast throws can zip through the 0.05 rad tolerance zone
        # between two joint_states samples). Better to release a little
        # early than to keep the suction on indefinitely.
        timeout_sec = float(np.sum(timestep)) + 0.1
        reached = self._wait_for_position(
            release_joint, tolerance=0.05, timeout_sec=timeout_sec
        )
        self.suction_off()
        if not reached:
            self._node.get_logger().warn(
                f"Release joint not detected within {timeout_sec:.2f}s; "
                "suction_off fired on timeout fallback."
            )

        # Wait for trajectory completion
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self._node, result_future)
        return True

    # ------------------------------------------------------------------
    # Queue Mode (streaming)
    # ------------------------------------------------------------------

    def enter_queue_mode(self) -> bool:
        """Trajectory mode 해제 후 Point Queue mode 진입.

        컨트롤러 알람/에러(예: 4414 excessive segment velocity)로 진입 실패 시
        ``/reset_error``를 1회 호출하고 재시도한다 — 자동 복구가 없으면 한 번
        알람이 뜬 뒤 이후 모든 진입이 "active Alarm"으로 연쇄 실패한다.
        """
        res = self._try_start_queue_mode()
        if res is not None and res.result_code.value == 1:
            self._node.get_logger().info(f"Queue mode entered: {res.message or 'READY'}")
            return True

        # MotionReadyEnum: 101=Alarm, 102=Error, 112=Inc-move error — all
        # clearable remotely via reset_error. E-Stop/HOLD/TEACH/not-REMOTE need
        # physical action, so don't auto-retry those (it would just loop).
        recoverable = res is not None and res.result_code.value in (101, 102, 112)
        if not recoverable:
            self._node.get_logger().error(
                f"start_point_queue_mode failed: {(res and res.message) or 'timeout'}"
            )
            return False

        self._node.get_logger().warn(
            f"Queue mode blocked ('{res.message}'); calling /reset_error and "
            "retrying once."
        )
        self._reset_error()
        res = self._try_start_queue_mode()
        if res is not None and res.result_code.value == 1:
            self._node.get_logger().info(
                f"Queue mode entered after reset_error: {res.message or 'READY'}"
            )
            return True
        self._node.get_logger().error(
            "start_point_queue_mode still failing after reset_error: "
            f"{(res and res.message) or 'timeout'} (major/hardware alarm? "
            "clear it on the teach pendant)."
        )
        return False

    def _try_start_queue_mode(self):
        """One stop+start attempt; returns the service result (None on timeout)."""
        # [QMODE-DBG] watch whether the stop+start physically moves the arm.
        j0 = list(self.current_joints) if self.current_joints else None
        self._jmon = ([float(x) for x in j0], [float(x) for x in j0]) if j0 else None
        t0 = time.time()
        self._stop_current_mode()
        if not self._start_queue_client.wait_for_service(timeout_sec=5.0):
            self._node.get_logger().error("/start_point_queue_mode unavailable.")
            self._jmon = None
            return None
        fut = self._start_queue_client.call_async(StartPointQueueMode.Request())
        rclpy.spin_until_future_complete(self._node, fut, timeout_sec=10.0)
        res = fut.result()
        # Record the measured switch cost so the timeline model's T_setup tracks the
        # real controller latency. last_qmode_ms keeps the raw value (diagnostic);
        # the EMA folds ONLY a SUCCESSFUL switch (result_code 1) so a 10 s service
        # timeout (res None) or an alarm-blocked attempt can't spike qmode_ms_avg and
        # transiently over-place / drop catchable pushes for several cycles.
        dt_ms = (time.time() - t0) * 1000.0
        self.last_qmode_ms = dt_ms
        if res is not None and res.result_code.value == 1:
            self.qmode_ms_avg = 0.3 * dt_ms + 0.7 * self.qmode_ms_avg   # EMA, alpha=0.3
        if self._jmon is not None and j0 is not None:
            lo, hi = self._jmon
            exc = [round(hi[i] - lo[i], 4) for i in range(len(lo))]
            self._node.get_logger().info(
                f"[QMODE-DBG] mode switch {dt_ms:.0f}ms; "
                f"joint excursion (max-min) = {exc} rad"
            )
        self._jmon = None
        return res

    def _reset_error(self) -> bool:
        """Call MotoROS2 /reset_error to clear an active alarm/error.

        Alarms numbered < 8000 (major / hardware / setting faults) cannot be
        reset remotely; this returns False and the operator must clear them on
        the teach pendant.
        """
        if not self._reset_error_client.wait_for_service(timeout_sec=2.0):
            self._node.get_logger().error("/reset_error unavailable.")
            return False
        fut = self._reset_error_client.call_async(ResetError.Request())
        rclpy.spin_until_future_complete(self._node, fut, timeout_sec=5.0)
        res = fut.result()
        ok = bool(res and res.result_code.value == 1)
        if ok:
            self._node.get_logger().info("reset_error: controller alarm/error cleared.")
        else:
            self._node.get_logger().error(
                f"reset_error failed: {(res and res.message) or 'timeout'} "
                "(major/hardware alarm? clear it on the teach pendant)."
            )
        time.sleep(0.2)   # let the controller settle before the retry
        return ok

    def exit_queue_mode(self) -> bool:
        """Queue mode → Trajectory mode 복귀 (종료 시 사용)."""
        self._stop_current_mode()
        if not self._start_traj_client.wait_for_service(timeout_sec=5.0):
            return False
        fut = self._start_traj_client.call_async(StartTrajMode.Request())
        rclpy.spin_until_future_complete(self._node, fut, timeout_sec=10.0)
        res = fut.result()
        if res is None or res.result_code.value != 1:
            return False
        self._node.get_logger().info("Returned to trajectory mode.")
        return True

    def _stop_current_mode(self) -> bool:
        """현재 활성 traj/queue mode 정지. 모드 전환 전 필수."""
        if not self._stop_traj_client.wait_for_service(timeout_sec=2.0):
            return False
        fut = self._stop_traj_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self._node, fut, timeout_sec=5.0)
        res = fut.result()
        return bool(res and res.success)

    def send_trajectory_queue(
        self,
        traj: np.ndarray,
        vel: np.ndarray,
        timestep: np.ndarray,
        final_joint: np.ndarray | None = None,
    ) -> bool:
        """Queue Mode로 trajectory streaming. FJT의 send_trajectory 대체.

        주의: Queue Mode도 새 큐의 첫 점이 로봇 현재(피드백) 위치와 일치해야
        받아준다(code 204 'first point must match current position').
        _build_queue_waypoints가 positions[0]을 측정 현재 관절로 치환해 처리한다.
        """
        waypoints = self._build_queue_waypoints(traj, vel, timestep, final_joint)
        t_start = time.time()
        if not self._push_waypoints(waypoints):
            return False
        # 마지막 point time_from_start까지 대기 (push_time만큼 이미 진행됨)
        self._wait_trajectory_end(waypoints[-1][2], t_start=t_start)
        return True

    def send_trajectory_queue_with_attach(
        self,
        traj: np.ndarray,
        vel: np.ndarray,
        timestep: np.ndarray,
        final_joint: np.ndarray,
        attach_target_joint: np.ndarray,
        attach_tolerance: float = 0.05,
    ) -> bool:
        """Queue Mode + attach_target_joint 근접 시 suction_on. pick 전용.

        ROS1 customcontroller.publish_trajectory(suction_on=True) 와 동일한
        의도: trajectory 실행 중 grasp pose 직전(diff<tolerance)에 미리
        suction_on을 발사해 공압 지연(밸브 열림 ~ 진공 형성)을 보정.
        보통 attach_target_joint 는 final_joint 와 동일 (grasp_joint).

        Push 는 최대한 빠르게 하고 동시에 joint_states 모니터링.
        """
        waypoints = self._build_queue_waypoints(traj, vel, timestep, final_joint)
        total_duration = waypoints[-1][2]

        t_start = time.time()
        if not self._push_waypoints(waypoints):
            return False

        # 남은 실행 시간 동안 attach_target 근접 감시
        elapsed = time.time() - t_start
        remaining = total_duration - elapsed + 0.1
        if remaining < 0:
            remaining = 0.1
        reached = self._wait_for_position(
            attach_target_joint, tolerance=attach_tolerance, timeout_sec=remaining,
        )
        self.suction_on()
        if not reached:
            self._node.get_logger().warn(
                f"Attach target not detected within {remaining:.2f}s; "
                "suction_on fired on timeout fallback."
            )

        # trajectory 완료까지 대기
        self._wait_trajectory_end(total_duration, t_start=t_start)
        return True

    def send_trajectory_queue_with_release(
        self,
        traj: np.ndarray,
        vel: np.ndarray,
        timestep: np.ndarray,
        final_joint: np.ndarray,
        release_joint: np.ndarray,
    ) -> bool:
        """Queue Mode + release_joint 도달 시 suction_off. throw 전용.

        로봇은 첫 point가 큐에 들어가자마자 실행 시작하므로,
        push 는 최대한 빠르게 하고 동시에 joint_states 모니터링.
        """
        waypoints = self._build_queue_waypoints(traj, vel, timestep, final_joint)
        total_duration = waypoints[-1][2]

        t_start = time.time()
        if not self._push_waypoints(waypoints):
            return False

        # 남은 실행 시간 동안 release_joint 도달 감시
        elapsed = time.time() - t_start
        remaining = total_duration - elapsed + 0.1
        if remaining < 0:
            remaining = 0.1
        reached = self._wait_for_position(
            release_joint, tolerance=0.05, timeout_sec=remaining,
        )
        self.suction_off()
        if not reached:
            self._node.get_logger().warn(
                f"Release joint not detected within {remaining:.2f}s; "
                "suction_off fired on timeout fallback."
            )

        # trajectory 완료까지 대기
        self._wait_trajectory_end(total_duration, t_start=t_start)
        return True

    def send_trajectory_queue_with_timed_release(
        self,
        traj: np.ndarray,
        vel: np.ndarray,
        timestep: np.ndarray,
        final_joint: np.ndarray,
        release_index: int,
        suction_on_at: float | None = None,
    ) -> bool:
        """Queue Mode + suction_off interleaved into the point-push.

        Fires suction_off right after the ``release_index`` waypoint is queued,
        instead of after the whole trajectory. The old "push everything then
        suction_off" path fired late because the BUSY-throttled push takes ~the
        full swing duration, so suction_off landed at the end. The point-queue
        and IO are independent services, so inserting the IO command mid-push
        is safe (no collision/drop).

        ``suction_on_at`` (wall-clock, optional): once the release has fired and
        this instant passes, fire suction_ON ONCE — to PRIME THE NEXT pick's
        vacuum during this throw's return (chain) move, so a back-to-back object
        keeps its full lead. Only fires after the release (never while still
        holding the thrown object). Returns True iff that next-suction fired.
        """
        waypoints = self._build_queue_waypoints(traj, vel, timestep, final_joint)
        total_duration = waypoints[-1][2]

        state = {"fired": False, "primed_next": False}
        t_start = time.time()
        self.last_throw = {
            "throw_start": t_start,
            "release_wall": None,
            "io_ms": None,
            "release_index": release_index,
            "n_waypoints": len(waypoints),
        }

        def _release() -> None:
            t_io = time.time()
            self.suction_off()                   # synchronous WriteSingleIO round-trip
            io_ms = (time.time() - t_io) * 1000.0
            state["fired"] = True
            self.last_throw["release_wall"] = self.last_suction_off_t
            self.last_throw["io_ms"] = io_ms
            self._node.get_logger().info(
                f"Release: suction_off after waypoint {release_index}/{len(waypoints)} "
                f"(IO round-trip {io_ms:.0f} ms)"
            )

        def _prime_next() -> None:
            # Prime the NEXT pick's vacuum, but only AFTER this object's release
            # (don't suck while still holding/releasing the thrown object).
            if (suction_on_at is not None and state["fired"]
                    and not state["primed_next"] and time.time() >= suction_on_at):
                self.suction_on()
                state["primed_next"] = True
                self._node.get_logger().info(
                    "Return-prime: suction_on for next pick during chain move."
                )

        if not self._push_waypoints(
            waypoints, release_index=release_index, release_fn=_release,
            between_fn=_prime_next,
        ):
            return state["primed_next"]
        if not state["fired"]:
            # release_index beyond the pushed points — fire now as a fallback.
            _release()

        # Finish the move, still watching the next-pick suction deadline.
        t_end = t_start + total_duration + 0.1
        while time.time() < t_end:
            rclpy.spin_once(self._node, timeout_sec=0.05)
            _prime_next()
        _prime_next()
        # Settle: wait for the arm to ACTUALLY reach the final pose, not just the
        # time estimate. The BUSY-throttled push delays execution, so the loop
        # above can return while the arm is still finishing the return (chain)
        # move; the NEXT cycle's mode stop would then chop that still-moving arm
        # (the observed bobble). Waiting here means the next cycle starts from a
        # stopped arm.
        self._wait_for_position(final_joint, tolerance=0.03, timeout_sec=1.5)
        return state["primed_next"]

    def send_trajectory_queue_timed_suction(
        self,
        traj: np.ndarray,
        vel: np.ndarray,
        timestep: np.ndarray,
        final_joint: np.ndarray,
        suction_on_at: float,
    ) -> bool:
        """Queue-mode move that fires suction_on ONCE at wall-clock ``suction_on_at``.

        The deadline is checked during BOTH the point-push and the end-wait, so
        the vacuum is primed on time even while the arm is still positioning
        (priming early is harmless for a suction gripper). Used for the ambush
        pre-position: guarantees the full SUCTION_LEAD before object arrival
        regardless of how long positioning takes. Returns True iff suction was
        fired before returning (i.e. ``suction_on_at`` had already passed)."""
        waypoints = self._build_queue_waypoints(traj, vel, timestep, final_joint)
        total_duration = waypoints[-1][2]
        state = {"fired": False}

        def _fire() -> None:
            if not state["fired"] and time.time() >= suction_on_at:
                self.suction_on()
                state["fired"] = True

        t_start = time.time()
        if not self._push_waypoints(waypoints, between_fn=_fire):
            return state["fired"]
        # Finish the move, still watching the suction deadline.
        t_end = t_start + total_duration + 0.1
        while time.time() < t_end:
            rclpy.spin_once(self._node, timeout_sec=0.02)
            _fire()
        _fire()
        return state["fired"]

    def _build_queue_waypoints(
        self,
        traj: np.ndarray,
        vel: np.ndarray,
        timestep: np.ndarray,
        final_joint: np.ndarray | None,
        extra_time: float = 0.05,
    ) -> list[tuple[list, list, float]]:
        """FJT _build_goal과 동일 로직 — positions[0]을 측정 현재 관절로 치환.

        MotoROS2 point-queue는 새 큐의 첫 점이 로봇 현재(피드백) 위치와
        일치해야 받아준다(code 204 'first point must match current position').
        이전엔 큐 모드가 이 체크에서 면제된다고 가정해 치환을 생략했으나,
        실제로는 적용되어, 첫 점이 '계획된 grasp_joint'(측정 위치 아님)인
        던지기 trajectory가 재진입 시 거부됐다. _build_goal과 동일하게
        positions[0]을 현재 관절로 치환해 회피한다.
        """
        positions = traj.T.tolist()
        if self.current_joints is not None:
            positions[0] = list(self.current_joints)
        velocities = vel.T.tolist()
        times = timestep.tolist()

        if final_joint is None:
            final_joint_list = positions[-1]
        else:
            final_joint_list = list(final_joint)

        positions.append(final_joint_list)
        velocities.append([0.0] * 6)
        times.append(times[-1] + extra_time)

        return list(zip(positions, velocities, times))

    def _push_waypoints(
        self,
        waypoints: list[tuple[list, list, float]],
        busy_retry_delay: float = 0.015,
        busy_max_retry: int = 5,
        release_index: int | None = None,
        release_fn=None,
        between_fn=None,
    ) -> bool:
        """waypoint를 /motoman_gp8_controller/queue_traj_point로 순차 push.

        ``release_index``/``release_fn`` 지정 시, 해당 인덱스 waypoint를 큐에
        넣은 직후 ``release_fn``을 1회 호출 — throw 도중 석션 OFF를 포인트
        명령들 사이에 끼워넣는 용도.
        """
        if not self._queue_point_client.wait_for_service(timeout_sec=2.0):
            self._node.get_logger().error("queue_traj_point service unavailable.")
            return False

        # diag (no-op if off): log the queued command — start = snapped current
        # position (waypoints[0]), target = final point, planned dur = its
        # time_from_start. The actual reach time is filled from joint samples.
        if waypoints:
            self._motion_logger.on_command(
                waypoints[0][0], waypoints[-1][0], waypoints[-1][2],
            )

        # [PUSH-DIAG] (only when motion logging is on) measure per-point queue
        # round-trip + WHEN the arm actually starts moving, on ONE clock, to verify
        # whether the startup dead-time is the per-point push round-trips (H1) or
        # MotoROS2-side startup AFTER queueing (H2). current_joints updates during
        # spin_until_future_complete, so we can spot motion-start mid-push.
        _diag = self._motion_logger.enabled
        _t0 = time.time()
        _start_pos = list(waypoints[0][0]) if waypoints else None
        _per_pt_ms = []
        _t_move = None

        busy_total = 0
        for i, (pos, v, t) in enumerate(waypoints):
            req = QueueTrajPoint.Request()
            req.joint_names = JOINT_NAMES
            req.point.positions = [float(x) for x in pos]
            req.point.velocities = [float(x) for x in v]
            req.point.time_from_start = _seconds_to_duration(t)

            ok = False
            _t_pt = time.time()
            for _ in range(busy_max_retry + 1):
                fut = self._queue_point_client.call_async(req)
                rclpy.spin_until_future_complete(self._node, fut, timeout_sec=2.0)
                res = fut.result()
                if res is None:
                    self._node.get_logger().error(f"pt {i}: service timeout")
                    return False
                code = res.result_code.value
                if code == 1:  # SUCCESS
                    ok = True
                    break
                if code == 4:  # BUSY
                    busy_total += 1
                    time.sleep(busy_retry_delay)
                    continue
                self._node.get_logger().error(
                    f"pt {i}: queue code={code} msg='{res.message}'"
                )
                return False
            if not ok:
                self._node.get_logger().warn(
                    f"pt {i}: dropped after {busy_max_retry} BUSY retries"
                )
                return False

            if _diag:
                _per_pt_ms.append((time.time() - _t_pt) * 1000.0)
                if (_t_move is None and _start_pos is not None
                        and self.current_joints is not None):
                    dev = max(abs(a - b) for a, b in
                              zip(self.current_joints, _start_pos))
                    if dev > 0.01:          # arm has physically started moving
                        _t_move = time.time() - _t0

            # Interleave the release: once the release waypoint is queued,
            # fire suction_off (the IO command rides between point commands).
            if release_fn is not None and release_index is not None and i >= release_index:
                release_fn()
                release_fn = None  # fire once

            # Generic per-point hook: e.g. fire an early suction_on the instant
            # its wall-clock deadline passes, mid-push if needed.
            if between_fn is not None:
                between_fn()

        if busy_total > 0:
            # BUSY는 push가 MotoROS2 수신 속도보다 빠를 때 발생하는 정상 신호.
            # 큐가 깊지 않을 때 흔하며, 재시도로 자연스럽게 흡수됨.
            self._node.get_logger().debug(
                f"Queue push: {busy_total} BUSY retries across {len(waypoints)} pts"
            )
        if _diag:
            n = len(_per_pt_ms)
            avg = sum(_per_pt_ms) / n if n else 0.0
            mx = max(_per_pt_ms) if _per_pt_ms else 0.0
            move_s = (f"{_t_move * 1000:.0f}ms (DURING push)"
                      if _t_move is not None else "NOT until push done (-> H2)")
            self._node.get_logger().info(
                f"[PUSH-DIAG] {n} pts queued in {(time.time() - _t0) * 1000:.0f}ms "
                f"(avg {avg:.0f}, max {mx:.0f} ms/pt; busy={busy_total}); "
                f"arm motion start = {move_s}"
            )
            # Persist the same breakdown onto the open command row so err_s can be
            # decomposed (re-entry / push / arm-start) per dispatch in the CSV.
            self._motion_logger.on_dispatch(
                n_pts=len(waypoints),
                push_ms=(time.time() - _t0) * 1000.0,
                perpt_avg=avg, perpt_max=mx, busy=busy_total,
                motion_start_ms=(_t_move * 1000.0) if _t_move is not None else None,
                qmode_ms=self.last_qmode_ms,
            )
        return True

    def _push_one_point(
        self, pos, vel, t, *, busy_retry_delay: float = 0.02,
        busy_max_wait_s: float = 2.0,
    ) -> "int | None":
        """Push ONE queue point; return its result_code.

        BUSY(4) = the bounded MotoROS2 queue is momentarily FULL — normal
        backpressure when streaming faster than the robot consumes (measured
        queue depth ~10 points). We WAIT for a slot, retrying for up to
        ``busy_max_wait_s`` (the robot frees one slot per consumed point,
        ~0.2s), instead of giving up after a few ms — which would abort a long
        continuous stream mid-way. SUCCESS=1, WRONG_MODE=2, INIT_FAILURE=3,
        INVALID_JOINT_LIST=5, UNABLE_TO_PROCESS_POINT=6; ``None`` = service
        timeout. Sustained BUSY past the budget returns 4 (a reject, not a
        silent drop). The message (carries the '204'/init text on a re-checked
        start position) is logged on a hard reject.
        """
        req = QueueTrajPoint.Request()
        req.joint_names = JOINT_NAMES
        req.point.positions = [float(x) for x in pos]
        req.point.velocities = [float(x) for x in vel]
        req.point.time_from_start = _seconds_to_duration(float(t))
        busy_waited = 0.0
        while True:
            fut = self._queue_point_client.call_async(req)
            rclpy.spin_until_future_complete(self._node, fut, timeout_sec=2.0)
            res = fut.result()
            if res is None:
                self._node.get_logger().error("[PERSIST] queue point: service timeout")
                return None
            code = int(res.result_code.value)
            if code == 4:  # queue full -> honor backpressure, wait for a slot
                if busy_waited >= busy_max_wait_s:
                    self._node.get_logger().warn(
                        f"[PERSIST] queue BUSY for {busy_waited:.1f}s; giving up")
                    return 4
                time.sleep(busy_retry_delay)
                busy_waited += busy_retry_delay
                continue
            if code != 1:
                self._node.get_logger().warn(
                    f"[PERSIST] queue reject code={code} msg='{res.message}'")
            return code

    def push_segments_persistent(
        self, segments, *, wait: bool = True, tail_buffer: float = 0.3,
        join_tol: float = 0.05,
    ):
        """Stage-C 영속 큐 프리미티브 — N 세그먼트를 하나의 큐 세션에 연속 스트리밍.

        세그먼트마다 큐모드를 재진입(stop+start, 측정 ~0.41s = dispatch 오버헤드의
        61%)하는 대신, 큐를 비우지 않고 다음 세그먼트 점들을 이어붙여 재진입을 제거.

        segments: ``[(traj, vel, timestep, final_joint), ...]`` (send_trajectory_queue
        와 동일 형식; **속도-연속**이어야 함). 하나의 **순증가** 타임라인으로 병합:
        - 첫 세그먼트의 첫 점만 측정 현재 위치로 스냅(큐 초기화 — MotoROS2는 새 큐의
          첫 점이 현재 위치와 일치해야 함; 거부는 result_code 3/6 + 메시지의 '204').
        - 이후 세그먼트는 직전 세그먼트 끝 시각 뒤로 offset; 공유 경계점(seg[0] ≈ 직전
          끝점, ``join_tol`` 이내)은 **중복 제거**해 시각이 strictly-increasing 유지.
        - 정지(zero-vel) settle 점은 **마지막에 한 번만** 부여(중간 정지/속도 불연속 방지).

        호출 전 ``enter_queue_mode()`` 1회 필요. 반환 ``(ok, seg_codes)`` —
        ``seg_codes[i]`` = 세그먼트 i가 받은 result_code들.

        **거부 시 즉시 중단(abort-on-reject)**: 어떤 점이든 non-SUCCESS면 그 자리에서
        푸시를 멈춰 큐에 **연속 prefix만** 남긴다(구멍 없음). 따라서 큐 드레인
        (WRONG_MODE=2)이든, 비-빈 큐 append 재검사(INIT_FAILURE=3/UNABLE=6)든,
        영리한 재개를 시도하지 않고 안전하게 실패를 알린다 — 팔은 큐된 prefix를 마치고
        정지(물리적으로 안전). 복구(현재 위치 재측정 후 일반 경로)는 호출자 책임.

        NOTE(Stage-C 게이트): 비-빈 큐 append가 시작점 재검사를 받는지 여부는
        ``tests/persistent_queue_spike.py``로 HW 검증. 기본 skill 경로엔 미연결.
        """
        n_seg = len(segments)
        if n_seg == 0:
            return True, []
        if self.current_joints is None:
            self._node.get_logger().error("[PERSIST] no current_joints; aborting.")
            return False, []

        # --- build ONE strictly-increasing, boundary-deduped waypoint stream ---
        stream = []                      # (pos, vel, time, seg_idx)
        running_end = 0.0
        prev_last_pos = None
        for si, (traj, vel, ts, final_joint) in enumerate(segments):
            pos = [list(p) for p in traj.T.tolist()]
            vels = [list(v) for v in vel.T.tolist()]
            times = [float(t) for t in ts]
            if si == 0:
                pos[0] = list(self.current_joints)        # queue-init match
                start_idx, offset = 0, 0.0
            else:
                offset = running_end
                if _max_abs_diff(pos[0], prev_last_pos) <= join_tol:
                    start_idx = 1                          # drop shared boundary point
                else:                                      # discontinuous: connect, keep monotonic
                    start_idx = 0
                    step = (times[1] - times[0]) if len(times) > 1 else 0.05
                    offset = running_end + step
                    self._node.get_logger().warn(
                        f"[PERSIST] seg {si} boundary gap "
                        f"{_max_abs_diff(pos[0], prev_last_pos):.3f} rad; inserting connector.")
            for i in range(start_idx, len(times)):
                stream.append((pos[i], vels[i], times[i] + offset, si))
            running_end = times[-1] + offset
            prev_last_pos = pos[-1]
        # single full-stop settle point at the very end
        last_final = segments[-1][3]
        fj = list(last_final) if last_final is not None else list(prev_last_pos)
        running_end += 0.05
        stream.append((fj, [0.0] * 6, running_end, n_seg - 1))

        # defensive: times must be strictly increasing for the MotoROS2 queue
        for a, b in zip(stream, stream[1:]):
            if b[2] <= a[2]:
                self._node.get_logger().error(
                    f"[PERSIST] non-monotonic time {a[2]:.3f}->{b[2]:.3f}; aborting.")
                return False, [[] for _ in range(n_seg)]

        # --- push with ABORT-ON-REJECT (queue keeps only a continuous prefix) ---
        seg_codes = [[] for _ in range(n_seg)]
        if not self._queue_point_client.wait_for_service(timeout_sec=2.0):
            self._node.get_logger().error("[PERSIST] queue_traj_point unavailable.")
            return False, seg_codes
        t_start = time.time()
        ok = True
        pushed_end = 0.0
        for pos, v, t, si in stream:
            code = self._push_one_point(pos, v, t)
            seg_codes[si].append(code)
            if code != 1:
                ok = False
                self._node.get_logger().warn(
                    f"[PERSIST] seg {si}: rejected (code={code}); stopping push — "
                    "queue holds continuous prefix, caller must recover.")
                break
            pushed_end = t
        if wait:
            # wait only for what was actually queued (the continuous prefix)
            self._wait_trajectory_end(pushed_end, t_start=t_start, tail_buffer=tail_buffer)
        return ok, seg_codes

    # ------------------------------------------------------------------
    # Persistent-queue SESSION (stateful): pick + ambush hold + throw in ONE
    # queue, with NO per-segment re-entry. push_segments_persistent handles the
    # all-known-upfront case; this session adds a dynamic HOLD (feed points at
    # grasp during the ambush wait so the queue never drains) between segments.
    # ------------------------------------------------------------------
    def pq_begin(self) -> None:
        """Open a persistent-queue session on the CURRENT (already-entered) queue.

        Caller must have a fresh ``enter_queue_mode()`` succeed first. Resets the
        monotonic session timeline; the next ``pq_segment`` snaps its first point
        to the measured current position (queue-init / code 204)."""
        if self._pq_active:
            self._node.get_logger().warn(
                "[PQ] pq_begin while a session is active; resetting timeline — "
                "ensure a fresh enter_queue_mode() preceded this.")
        self._pq_active = True
        self._pq_t = 0.0
        self._pq_last_pos = None
        self._pq_start = time.time()
        self._pq_start_pose = list(self.current_joints) if self.current_joints else None
        self._pq_motion_start = None

    def _pq_push_stream(self, wp, *, between_fn=None) -> tuple:
        """Push (pos, vel, time) points with ABORT-ON-REJECT; advance the session
        timeline to the last SUCCESS point. ``between_fn`` (if given) is called
        after each successful push — used to fire a POSITION-BASED suction event
        while the queue drains (current_joints refreshes during each push spin).
        Returns (ok, codes)."""
        codes = []
        ok = True
        for pos, v, t in wp:
            if self._pq_last_pos is not None and t <= self._pq_t:
                self._node.get_logger().error(
                    f"[PQ] non-monotonic time {self._pq_t:.3f}->{t:.3f}; aborting.")
                ok = False
                break
            code = self._push_one_point(pos, v, t)
            codes.append(code)
            if code != 1:
                ok = False
                self._node.get_logger().warn(
                    f"[PQ] point rejected code={code}; stopping (continuous prefix kept).")
                break
            self._pq_t = t
            self._pq_last_pos = list(pos)
            # Record when the arm ACTUALLY first moves (current_joints refreshes in
            # the push spin) — pq_hold_until anchors its pacing to this, not pq_begin.
            if (self._pq_motion_start is None and self._pq_start_pose is not None
                    and self.current_joints is not None
                    and _max_abs_diff(self.current_joints, self._pq_start_pose) > 0.01):
                self._pq_motion_start = time.time()
            if between_fn is not None:
                between_fn()
        return ok, codes

    def pq_segment(self, traj, vel, ts, final_joint, *, is_last: bool = False,
                   join_tol: float = 0.05, between_fn=None) -> tuple:
        """Append one trajectory segment onto the session timeline (no re-entry).

        Session's first segment snaps its first point to current joints; later
        segments append raw and drop the shared boundary point (seg[0] ≈ last
        queued pose within ``join_tol``) so times stay strictly increasing. The
        zero-velocity settle point is added only when ``is_last``. ``between_fn``
        is called after each successful push (position-based IO hook, e.g. throw
        release). (ok, codes)."""
        if not self._pq_active:
            self._node.get_logger().error("[PQ] pq_segment without pq_begin; aborting.")
            return False, []
        pos = [list(p) for p in traj.T.tolist()]
        vels = [list(v) for v in vel.T.tolist()]
        times = [float(t) for t in ts]
        first_seg = self._pq_last_pos is None
        if first_seg:
            if self.current_joints is None:
                self._node.get_logger().error("[PQ] no current_joints; aborting.")
                return False, []
            pos[0] = list(self.current_joints)
            start_idx, offset = 0, 0.0
        else:
            offset = self._pq_t
            if _max_abs_diff(pos[0], self._pq_last_pos) <= join_tol:
                start_idx = 1
            else:
                start_idx = 0
                step = (times[1] - times[0]) if len(times) > 1 else 0.05
                offset = self._pq_t + step
                self._node.get_logger().warn(
                    f"[PQ] segment boundary gap "
                    f"{_max_abs_diff(pos[0], self._pq_last_pos):.3f} rad; connector.")
        wp = [(pos[i], vels[i], times[i] + offset) for i in range(start_idx, len(times))]
        if is_last:
            base = wp[-1][2] if wp else offset
            fj = list(final_joint) if final_joint is not None else list(pos[-1])
            wp.append((fj, [0.0] * 6, base + 0.05))
        if not wp:
            return True, []          # degenerate (single point deduped away)
        if not self._queue_point_client.wait_for_service(timeout_sec=2.0):
            self._node.get_logger().error("[PQ] queue_traj_point unavailable.")
            return False, []
        # Diagnostic (no-op unless GP8_MOTION_LOG_DIR): record this segment as a
        # dispatch so persistent cycles show up in motion_commands.csv comparably
        # to the non-persistent path. qmode_ms = the queue re-entry cost, but ONLY
        # for the session's FIRST segment (which followed enter_queue_mode); later
        # segments (the throw) had NO re-entry -> 0.0, which is exactly the ~0.41s
        # this path removes — so a persistent throw row reads qmode_ms=0 vs the
        # non-persistent throw's ~400ms.
        _log = self._motion_logger.enabled
        if _log:
            self._motion_logger.on_command(wp[0][0], wp[-1][0], wp[-1][2] - wp[0][2])
        _t0 = time.time()
        ok, codes = self._pq_push_stream(wp, between_fn=between_fn)
        if _log:
            self._motion_logger.on_dispatch(
                n_pts=len(wp), push_ms=(time.time() - _t0) * 1000.0,
                qmode_ms=(self.last_qmode_ms if first_seg else 0.0))
        return ok, codes

    def pq_hold_until(self, hold_joint, deadline_wall: float, *,
                      dt: float = 0.12, lead: float = 0.35, tick_fn=None) -> bool:
        """Keep the queue ALIVE at ``hold_joint`` until wall clock ``deadline_wall``.

        Feeds zero-velocity hold points at ``hold_joint`` so queue mode does not
        auto-exit during an ambush wait. Paced: push when the session timeline is
        < ``lead`` ahead of elapsed real MOTION time (anchored to the measured
        motion-start ``_pq_motion_start``, NOT pq_begin), so the buffer ahead of
        the arm is ~``lead`` — no longer ``lead + startup-dead-time``. This is the
        buffer the NEXT segment (the throw) waits behind, so keeping it near the
        keep-alive floor (~1 point) directly cuts the throw's onset delay.

        WARNING: with a small lead the queue is only ~1 hold point deep, so an
        unfed gap AFTER the hold (e.g. a slow NN/IK throw plan) can drain it →
        WRONG_MODE. The caller must PRECOMPUTE the throw during the hold (no unfed
        planning gap after) before shrinking lead, else the throw append aborts.

        Only WRONG_MODE(2) means the queue genuinely drained (fatal → False).
        BUSY(4)/timeout leave the queue ALIVE (full/transient) → skip that point
        and keep going. ``hold_joint`` MUST equal where the arm actually is (the
        last queued pose); a mismatch would be a position step over dt → velocity
        spike / alarm 4414, so it is refused."""
        if not self._pq_active:
            return False
        hold = list(hold_joint)[:6]
        if self._pq_last_pos is not None:
            d = _max_abs_diff(hold, self._pq_last_pos)
            if d > 0.02:
                self._node.get_logger().error(
                    f"[PQ] hold_joint is {d:.3f} rad from the last queued pose; "
                    "refusing (would command a jump). Hold at the segment's end pose.")
                return False
        zero = [0.0] * 6
        # Anchor to when the arm actually started moving (falls back to pq_begin if
        # motion-start was never detected, e.g. a prepositioned pick with no drive).
        anchor = self._pq_motion_start if self._pq_motion_start is not None else self._pq_start
        while time.time() < deadline_wall:
            if self._pq_t < (time.time() - anchor) + lead:
                code = self._push_one_point(hold, zero, self._pq_t + dt)
                if code == 2:  # WRONG_MODE = queue drained/exited -> genuinely lost
                    self._node.get_logger().warn("[PQ] hold: WRONG_MODE — queue drained.")
                    return False
                if code == 1:
                    self._pq_t += dt
                    self._pq_last_pos = hold
                # BUSY(4)/None(timeout): queue still alive (full/transient) -> skip
            else:
                rclpy.spin_once(self._node, timeout_sec=0.01)
            if tick_fn is not None:
                tick_fn()      # e.g. fire suction_on once its wall-clock instant passes
        return True

    def pq_finish(self, *, wait: bool = True, tail_buffer: float = 0.3,
                  settle_tol: float = 0.03) -> None:
        """End the session; optionally wait for the queued motion to complete.

        After the time estimate, CONFIRM arrival at the last queued pose via joint
        feedback — the pure wall-clock estimate can under-shoot by the startup
        dead-time or a BUSY-throttled final push, which would otherwise let a
        caller read a still-moving arm or chop it on the next mode switch."""
        if wait and self._pq_active:
            self._wait_trajectory_end(
                self._pq_t, t_start=self._pq_start, tail_buffer=tail_buffer)
            if self._pq_last_pos is not None:
                self._wait_for_position(
                    np.asarray(self._pq_last_pos, dtype=float),
                    tolerance=settle_tol, timeout_sec=1.5)
        self._pq_active = False

    def pq_throw_segment(self, traj, vel, timestep, final_joint, *,
                         release_index, suction_on_at=None,
                         release_tol: float = 0.05) -> tuple:
        """Append the THROW onto the live session with a POSITION-BASED suction
        release. Returns ``(ok, primed_next)``: ok is False if the append was
        REJECTED (e.g. the queue drained during throw planning) so the caller can
        salvage/abort instead of silently dropping the object — no bogus release
        is fired in that case.

        Release uses a LAYERED detector (fire on entering the release band OR on
        the first receding sample after the closest approach) so a fast swing
        that crosses the band between ~50 ms polls still fires within one sample,
        not chain-late. In the persistent path the throw is queued ~the hold
        buffer AHEAD of execution, so the index/queue-time release of
        ``send_trajectory_queue_with_timed_release`` would fire early — hence
        position-based here.

        WARNING: release timing is throw-critical; verify on HW where the object
        lands and tune release_tol / RELEASE_LEAD before enabling the flag."""
        arr = np.asarray(traj)
        rel = int(max(0, min(release_index, arr.shape[1] - 1)))
        release_pose = [float(x) for x in arr[:, rel]]
        state = {"fired": False, "primed_next": False, "min_diff": float("inf")}
        self.last_throw = {
            "throw_start": time.time(), "release_wall": None, "io_ms": None,
            "release_index": rel, "n_waypoints": int(arr.shape[1]),
        }

        def _do_release(note: str) -> None:
            t_io = time.time()
            self.suction_off()
            state["fired"] = True
            self.last_throw["release_wall"] = getattr(self, "last_suction_off_t", None)
            self.last_throw["io_ms"] = (time.time() - t_io) * 1000.0
            self._node.get_logger().info(f"[PQ] release: suction_off ({note}, idx {rel}).")

        def _tick() -> None:
            if not state["fired"]:
                if self.current_joints is not None:
                    d = _max_abs_diff(self.current_joints, release_pose)
                    receding = (state["min_diff"] <= release_tol * 1.5
                                and d > state["min_diff"] + release_tol / 2)
                    if d <= release_tol or receding:
                        _do_release("at pose" if d <= release_tol else "past closest")
                    else:
                        state["min_diff"] = min(state["min_diff"], d)
            elif (suction_on_at is not None and not state["primed_next"]
                  and time.time() >= suction_on_at):
                self.suction_on()
                state["primed_next"] = True
                self._node.get_logger().info("[PQ] return-prime: suction_on for next pick.")

        ok, _ = self.pq_segment(traj, vel, timestep, final_joint, is_last=True,
                                between_fn=_tick)
        if not ok:
            # Append rejected (queue drained during planning, or a reject). Do NOT
            # fire a release — the throw never queued; the caller salvages/aborts.
            self._node.get_logger().error(
                "[PQ] throw append REJECTED — not queued; caller must recover.")
            return False, state["primed_next"]
        # Post-push: the throw is still draining. Keep catching the release
        # (bounded — never carry the object into the chain) and the prime deadline.
        t_end = time.time() + 1.0
        while (not state["fired"] or not state["primed_next"]) and time.time() < t_end:
            rclpy.spin_once(self._node, timeout_sec=0.02)
            _tick()
        if not state["fired"]:
            _do_release("post-push last resort")
        return True, state["primed_next"]
        return state["primed_next"]

    def _wait_trajectory_end(
        self,
        total_duration: float,
        t_start: float | None = None,
        tail_buffer: float = 0.1,
    ) -> None:
        """time_from_start 기반 trajectory 완료 대기."""
        if t_start is None:
            t_start = time.time()
        remaining = total_duration - (time.time() - t_start) + tail_buffer
        if remaining > 0:
            # spin도 같이 돌려 joint_state 캐시 최신화
            t_end = time.time() + remaining
            while time.time() < t_end:
                rclpy.spin_once(self._node, timeout_sec=0.05)

    def _wait_for_target(self, target_joint, tolerance: float = 1e-4) -> None:
        """Spin until robot reaches target joint position."""
        while rclpy.ok():
            rclpy.spin_once(self._node, timeout_sec=0.01)
            if self.current_joints is not None:
                diff = math.sqrt(sum(
                    (a - b) ** 2 for a, b in zip(self.current_joints, target_joint)
                ))
                if diff < tolerance:
                    break

    def _wait_for_position(
        self,
        target_joint: np.ndarray,
        tolerance: float = 0.05,
        timeout_sec: float | None = None,
    ) -> bool:
        """Spin until robot is near ``target_joint``, or we pass it, or timeout.

        Returns True iff the position was actually reached (one of the
        closeness conditions fired), False iff the hard timeout expired.

        Detection is layered to survive fast throws:
          1. ``diff < tolerance`` — the ideal hit.
          2. ``diff > min_diff + tolerance/2`` — we already grazed the point
             and are now moving away. This saves us when the robot zips
             through the tolerance zone between two joint_states samples.
          3. ``time.time() - start > timeout_sec`` — hard fallback. We'd
             rather release slightly too early than never.
        """
        target_list = target_joint.tolist()
        start = time.time()
        min_diff = float("inf")

        while rclpy.ok():
            rclpy.spin_once(self._node, timeout_sec=0.01)
            if self.current_joints is not None:
                diff = math.sqrt(sum(
                    (a - b) ** 2 for a, b in zip(self.current_joints, target_list)
                ))
                if diff < tolerance:
                    return True
                if min_diff != float("inf") and diff > min_diff + tolerance / 2.0:
                    # Past the closest approach — release now.
                    return True
                if diff < min_diff:
                    min_diff = diff
            if timeout_sec is not None and (time.time() - start) > timeout_sec:
                return False

    # ------------------------------------------------------------------
    # Suction gripper
    # ------------------------------------------------------------------

    def suction_on(self) -> None:
        self.last_suction_on_t = time.time()   # for the pick-cycle timing log
        self._call_io(SUCTION_IO_ADDRESS, 0)

    def suction_off(self) -> None:
        self.last_suction_off_t = time.time()
        self._call_io(SUCTION_IO_ADDRESS, 1)

    def _call_io(self, address: int, value: int) -> None:
        """Call WriteSingleIO service synchronously."""
        req = WriteSingleIO.Request()
        req.address = address
        req.value = value
        future = self._io_client.call_async(req)
        rclpy.spin_until_future_complete(self._node, future)
        result = future.result()
        if not result.success:
            self._node.get_logger().error(f"IO write failed: {result.message}")

    # ------------------------------------------------------------------
    # Goal building
    # ------------------------------------------------------------------

    def _build_goal(
        self,
        traj: np.ndarray,
        vel: np.ndarray,
        timestep: np.ndarray,
        final_joint: np.ndarray | None = None,
        extra_time: float = 0.05,
    ) -> FollowJointTrajectory.Goal:
        """Build FollowJointTrajectory goal from numpy arrays."""
        positions = traj.T.tolist()
        if self.current_joints is not None:
            positions[0] = list(self.current_joints)
        velocities = vel.T.tolist()
        times = timestep.tolist()

        if final_joint is None:
            final_joint_list = positions[-1]
        else:
            final_joint_list = list(final_joint)

        positions.append(final_joint_list)
        velocities.append([0.0] * 6)
        times.append(times[-1] + extra_time)

        jt = JointTrajectory()
        jt.joint_names = JOINT_NAMES
        for pos, v, t in zip(positions, velocities, times):
            pt = JointTrajectoryPoint()
            pt.positions = [float(x) for x in pos]
            pt.velocities = [float(x) for x in v]
            pt.time_from_start = _seconds_to_duration(t)
            jt.points.append(pt)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = jt
        return goal

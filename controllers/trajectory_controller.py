"""Low-level trajectory execution and suction I/O control (ROS 2).

Uses FollowJointTrajectory action for trajectory execution and
WriteSingleIO service for suction gripper control via MotoROS2.
"""

from __future__ import annotations

import math
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
    StartPointQueueMode,
    StartTrajMode,
    WriteSingleIO,
)
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

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


class TrajectoryController:
    """Executes joint trajectories and controls suction gripper I/O via ROS 2."""

    def __init__(self, node: Node) -> None:
        self._node = node
        self.current_joints: list | None = None
        self.current_jointvels: list | None = None

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

    def _joint_state_cb(self, msg: JointState) -> None:
        self.current_joints = list(msg.position)
        self.current_jointvels = list(msg.velocity)

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
        """Trajectory mode 해제 후 Point Queue mode 진입."""
        self._stop_current_mode()
        if not self._start_queue_client.wait_for_service(timeout_sec=5.0):
            self._node.get_logger().error("/start_point_queue_mode unavailable.")
            return False
        fut = self._start_queue_client.call_async(StartPointQueueMode.Request())
        rclpy.spin_until_future_complete(self._node, fut, timeout_sec=10.0)
        res = fut.result()
        if res is None or res.result_code.value != 1:
            self._node.get_logger().error(
                f"start_point_queue_mode failed: "
                f"{res and res.message or 'timeout'}"
            )
            return False
        self._node.get_logger().info(f"Queue mode entered: {res.message or 'READY'}")
        return True

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
    ) -> bool:
        """Queue Mode + suction_off interleaved into the point-push.

        Fires suction_off right after the ``release_index`` waypoint is queued,
        instead of after the whole trajectory. The old "push everything then
        suction_off" path fired late because the BUSY-throttled push takes ~the
        full swing duration, so suction_off landed at the end. The point-queue
        and IO are independent services, so inserting the IO command mid-push
        is safe (no collision/drop).
        """
        waypoints = self._build_queue_waypoints(traj, vel, timestep, final_joint)
        total_duration = waypoints[-1][2]

        state = {"fired": False}
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

        if not self._push_waypoints(
            waypoints, release_index=release_index, release_fn=_release
        ):
            return False
        if not state["fired"]:
            # release_index beyond the pushed points — fire now as a fallback.
            _release()

        self._wait_trajectory_end(total_duration, t_start=t_start)
        return True

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
    ) -> bool:
        """waypoint를 /motoman_gp8_controller/queue_traj_point로 순차 push.

        ``release_index``/``release_fn`` 지정 시, 해당 인덱스 waypoint를 큐에
        넣은 직후 ``release_fn``을 1회 호출 — throw 도중 석션 OFF를 포인트
        명령들 사이에 끼워넣는 용도.
        """
        if not self._queue_point_client.wait_for_service(timeout_sec=2.0):
            self._node.get_logger().error("queue_traj_point service unavailable.")
            return False

        busy_total = 0
        for i, (pos, v, t) in enumerate(waypoints):
            req = QueueTrajPoint.Request()
            req.joint_names = JOINT_NAMES
            req.point.positions = [float(x) for x in pos]
            req.point.velocities = [float(x) for x in v]
            req.point.time_from_start = _seconds_to_duration(t)

            ok = False
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

            # Interleave the release: once the release waypoint is queued,
            # fire suction_off (the IO command rides between point commands).
            if release_fn is not None and release_index is not None and i >= release_index:
                release_fn()
                release_fn = None  # fire once

        if busy_total > 0:
            # BUSY는 push가 MotoROS2 수신 속도보다 빠를 때 발생하는 정상 신호.
            # 큐가 깊지 않을 때 흔하며, 재시도로 자연스럽게 흡수됨.
            self._node.get_logger().debug(
                f"Queue push: {busy_total} BUSY retries across {len(waypoints)} pts"
            )
        return True

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

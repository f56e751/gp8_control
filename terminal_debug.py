"""GP8 로봇 수동 디버그 도구 (ROS 2 / MotoROS2).

키보드로 엔드이펙터 이동 + 석션 그리퍼 제어.
이 스크립트를 실행한 터미널 창에서만 키 입력을 받습니다.
MotoROS2(또는 mock_robot)가 실행 중이어야 합니다.

조작법:
  w/s   : X축 전진/후진 (+/-1cm)
  a/d   : Y축 좌/우 (+/-1cm)
  r/f   : Z축 상/하 (+/-1cm)
  SPACE : 석션 ON/OFF 토글
  g     : 석션 그리퍼 수직 정렬
  p     : 현재 EE 위치 + 관절 상태 출력
  q     : 종료

Usage:
    ros2 run gp8_control terminal_debug
"""

from __future__ import annotations

import sys
import time
from datetime import datetime

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import qos_profile_sensor_data
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from motoros2_interfaces.srv import (
    QueueTrajPoint,
    StartPointQueueMode,
    StartTrajMode,
    WriteSingleIO,
)
from std_srvs.srv import Trigger
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from gp8_control.robots.gp8 import GP8

SUCTION_IO_ADDRESS = 10017
STEP_SIZE = 0.01          # 1cm (일반 키)
BIG_STEP_SIZE = 0.05      # 5cm (Shift + 키)
MOVE_DURATION = 0.5       # 이동 시간 (초)
FAST_MAX_VELOCITY = 1.0   # 최저점/안전높이 이동용 최대 속도 (m/s)
FAST_MIN_DURATION = 0.3   # 최저점/안전높이 이동 최소 시간 (초)
SAFE_HEIGHT = 0.10        # 'x' 키 상승 목표 높이 (m)
ANGLE_STEP = np.radians(5.0)       # 회전 기본 스텝
BIG_ANGLE_STEP = np.radians(15.0)  # 회전 Shift 스텝

# Home pose — app.py의 INITIAL_T/INITIAL_R와 동일 (tool-down, 0.4m 전방, 10cm 위)
HOME_POSITION = np.array([0.4, 0.0, 0.1])
HOME_ROTATION = np.array([
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0],
])
HOME_DURATION = 2.0  # 원상복구는 넓은 움직임이라 넉넉하게 (초)

# Queue Mode sweep 테스트 파라미터
QUEUE_TEST_AMPLITUDE = 0.03   # X축 진폭 (m)
QUEUE_TEST_CYCLES = 2         # sine 사이클 수
QUEUE_TEST_DURATION = 2.0     # 전체 지속 시간 (초)
QUEUE_TEST_RATE_HZ = 20.0     # 점 feed rate
QUEUE_BUSY_RETRY_DELAY = 0.015  # BUSY 응답 시 재시도 간격 (초)
QUEUE_BUSY_MAX_RETRY = 5
JOINT_NAMES = [
    "joint_1_s", "joint_2_l", "joint_3_u",
    "joint_4_r", "joint_5_b", "joint_6_t",
]
WORKSPACE = {
    "x": (0.0, 0.65),
    "y": (-0.65, 0.65),
    "z": (-0.04, 0.40),
}


def timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _seconds_to_duration(seconds: float) -> Duration:
    sec = int(seconds)
    nanosec = int((seconds - sec) * 1e9)
    return Duration(sec=sec, nanosec=nanosec)


if sys.platform == "win32":
    import msvcrt
    def get_key():
        return msvcrt.getch().decode("utf-8", errors="ignore")
else:
    import tty
    import termios
    def get_key():
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


KEY_DELTA = {
    "w": np.array([STEP_SIZE, 0.0, 0.0]),
    "s": np.array([-STEP_SIZE, 0.0, 0.0]),
    "a": np.array([0.0, STEP_SIZE, 0.0]),
    "d": np.array([0.0, -STEP_SIZE, 0.0]),
    "r": np.array([0.0, 0.0, STEP_SIZE]),
    "f": np.array([0.0, 0.0, -STEP_SIZE]),
    "W": np.array([BIG_STEP_SIZE, 0.0, 0.0]),
    "S": np.array([-BIG_STEP_SIZE, 0.0, 0.0]),
    "A": np.array([0.0, BIG_STEP_SIZE, 0.0]),
    "D": np.array([0.0, -BIG_STEP_SIZE, 0.0]),
    "R": np.array([0.0, 0.0, BIG_STEP_SIZE]),
    "F": np.array([0.0, 0.0, -BIG_STEP_SIZE]),
}

# 회전: (axis, angle). world-frame 기준.
KEY_ROT = {
    "u": ("x", ANGLE_STEP),        # roll +
    "j": ("x", -ANGLE_STEP),       # roll -
    "i": ("y", ANGLE_STEP),        # pitch +
    "k": ("y", -ANGLE_STEP),       # pitch -
    "o": ("z", ANGLE_STEP),        # yaw +
    "l": ("z", -ANGLE_STEP),       # yaw -
    "U": ("x", BIG_ANGLE_STEP),
    "J": ("x", -BIG_ANGLE_STEP),
    "I": ("y", BIG_ANGLE_STEP),
    "K": ("y", -BIG_ANGLE_STEP),
    "O": ("z", BIG_ANGLE_STEP),
    "L": ("z", -BIG_ANGLE_STEP),
}


def _axis_rotation_matrix(axis: str, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

# link_6_t -> flange -> tool0 -> suction_tip
T_CHAIN = np.array([
    [0., 0., 1., 0.325],
    [0., -1., 0., 0.],
    [1., 0., 0., 0.],
    [0., 0., 0., 1.],
])
T_CHAIN_INV = np.linalg.inv(T_CHAIN)


class RobotDebugger(Node):
    """키보드로 EE 이동 + 석션 제어하는 디버그 노드."""

    def __init__(self) -> None:
        super().__init__("robot_debugger")
        self._robot = GP8()
        self._suction_on = False
        self._current_joints: list | None = None

        cb_group = ReentrantCallbackGroup()

        self.create_subscription(
            JointState, "/joint_states_urdf",
            self._joint_state_cb, qos_profile_sensor_data,
            callback_group=cb_group,
        )

        self._fjt_client = ActionClient(
            self, FollowJointTrajectory,
            "/motoman_gp8_controller/follow_joint_trajectory",
            callback_group=cb_group,
        )

        self._io_client = self.create_client(
            WriteSingleIO, "/write_single_io",
            callback_group=cb_group,
        )

        self._start_traj_client = self.create_client(
            StartTrajMode, "/start_traj_mode",
            callback_group=cb_group,
        )

        # Queue Mode (streaming) — 테스트용
        self._start_queue_client = self.create_client(
            StartPointQueueMode, "/start_point_queue_mode",
            callback_group=cb_group,
        )
        # 브릿지가 URDF → raw 번역해서 /queue_traj_point로 포워딩
        self._queue_point_client = self.create_client(
            QueueTrajPoint, "/motoman_gp8_controller/queue_traj_point",
            callback_group=cb_group,
        )
        self._stop_traj_client = self.create_client(
            Trigger, "/stop_traj_mode",
            callback_group=cb_group,
        )

    def wait_for_servers(self) -> bool:
        self.get_logger().info("Waiting for FollowJointTrajectory...")
        if not self._fjt_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("FollowJointTrajectory not available.")
            return False
        self.get_logger().info("Waiting for WriteSingleIO...")
        if not self._io_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("WriteSingleIO not available.")
            return False
        self.get_logger().info("All required servers ready.")
        return True

    def start_traj_mode(self) -> bool:
        """MotoROS2 궤적 모드 활성화. mock_robot에는 이 서비스가 없으므로 skip."""
        if not self._start_traj_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(
                "/start_traj_mode not available — assuming mock_robot; skipping."
            )
            return True

        req = StartTrajMode.Request()
        future = self._start_traj_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        result = future.result()
        if result is None:
            self.get_logger().error("StartTrajMode call timed out.")
            return False
        if result.result_code.value != 1:
            self.get_logger().error(
                f"StartTrajMode failed: code={result.result_code.value}, "
                f"msg={result.message}"
            )
            return False
        self.get_logger().info(f"Trajectory mode enabled: {result.message or 'READY'}")
        return True

    def wait_for_joint_state(self) -> bool:
        self.get_logger().info("Waiting for /joint_states_urdf...")
        for _ in range(100):
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._current_joints is not None:
                self.get_logger().info("Joint states received.")
                return True
        self.get_logger().error("No joint states received.")
        return False

    def _joint_state_cb(self, msg: JointState) -> None:
        self._current_joints = list(msg.position)

    def move_ee(self, delta: np.ndarray) -> None:
        if self._current_joints is None:
            self.get_logger().error("Joint states not available.")
            return

        current_q = np.array(self._current_joints)
        current_T = self._robot.forward_kinematics(current_q)
        new_pos = current_T[:3, 3] + delta

        if not self._check_workspace(new_pos):
            self.get_logger().warn(
                f"[{timestamp()}] Out of workspace: "
                f"x={new_pos[0]:.3f} y={new_pos[1]:.3f} z={new_pos[2]:.3f}"
            )
            return

        target_T = current_T.copy()
        target_T[:3, 3] = new_pos

        target_q = self._robot.inverse_kinematics(target_T)
        if target_q is None:
            self.get_logger().warn(f"[{timestamp()}] IK failed for target position.")
            return

        self._send_joint_goal(current_q, target_q, MOVE_DURATION)
        self.get_logger().info(
            f"[{timestamp()}] EE moved to: "
            f"x={new_pos[0]:.3f} y={new_pos[1]:.3f} z={new_pos[2]:.3f}"
        )

    def rotate_ee(self, axis: str, angle: float) -> None:
        """EE 자세를 world frame 기준으로 delta만큼 회전 (위치 유지)."""
        if self._current_joints is None:
            self.get_logger().error("Joint states not available.")
            return

        current_q = np.array(self._current_joints)
        current_T = self._robot.forward_kinematics(current_q)

        R_delta = _axis_rotation_matrix(axis, angle)
        target_T = current_T.copy()
        target_T[:3, :3] = R_delta @ current_T[:3, :3]

        target_q = self._robot.inverse_kinematics(target_T)
        if target_q is None:
            self.get_logger().warn(
                f"[{timestamp()}] IK failed for rotation "
                f"({axis}-axis {np.degrees(angle):+.1f}°)."
            )
            return

        self._send_joint_goal(current_q, target_q, MOVE_DURATION)
        self.get_logger().info(
            f"[{timestamp()}] EE rotated: {axis}-axis {np.degrees(angle):+.1f}°"
        )

    def _check_workspace(self, pos: np.ndarray) -> bool:
        return (
            WORKSPACE["x"][0] <= pos[0] <= WORKSPACE["x"][1]
            and WORKSPACE["y"][0] <= pos[1] <= WORKSPACE["y"][1]
            and WORKSPACE["z"][0] <= pos[2] <= WORKSPACE["z"][1]
        )

    def _send_joint_goal(self, current: np.ndarray, target: np.ndarray, duration: float) -> None:
        jt = JointTrajectory()
        jt.joint_names = JOINT_NAMES

        pt0 = JointTrajectoryPoint()
        pt0.positions = [float(x) for x in current]
        pt0.velocities = [0.0] * 6
        pt0.time_from_start = _seconds_to_duration(0.0)
        jt.points.append(pt0)

        pt1 = JointTrajectoryPoint()
        pt1.positions = [float(x) for x in target]
        pt1.velocities = [0.0] * 6
        pt1.time_from_start = _seconds_to_duration(duration)
        jt.points.append(pt1)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = jt

        future = self._fjt_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Trajectory goal rejected.")
            return

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        wrapped = result_future.result()
        if wrapped is None:
            self.get_logger().warn("FJT result not received.")
            return
        result = wrapped.result
        if result.error_code != 0:
            self.get_logger().warn(
                f"FJT error: status={wrapped.status} "
                f"code={result.error_code} msg='{result.error_string}'"
            )

    def go_home(self) -> None:
        """Home 자세로 복귀 (INITIAL pose: 0.4m 전방, 10cm 높이, tool-down)."""
        if self._current_joints is None:
            self.get_logger().error("Joint states not available.")
            return

        current_q = np.array(self._current_joints)
        target_T = np.eye(4)
        target_T[:3, :3] = HOME_ROTATION
        target_T[:3, 3] = HOME_POSITION

        target_q = self._robot.inverse_kinematics(target_T)
        if target_q is None:
            self.get_logger().warn(f"[{timestamp()}] IK failed for home pose.")
            return

        self._send_joint_goal(current_q, target_q, HOME_DURATION)
        self.get_logger().info(
            f"[{timestamp()}] Homed to pos=({HOME_POSITION[0]:.2f}, "
            f"{HOME_POSITION[1]:.2f}, {HOME_POSITION[2]:.2f}), tool-down "
            f"({HOME_DURATION:.1f}s)"
        )

    # ------------------------------------------------------------------
    # Queue Mode (streaming) — 'pause between motions' 검증용
    # ------------------------------------------------------------------

    def run_queue_sweep_test(self) -> None:
        """Point Queue Mode로 X축 sine sweep을 스트리밍하고 timing 측정.

        Queue Mode가 FJT보다 부드러운지 확인하는 단일 테스트.
        끝나면 자동으로 FJT 모드로 복귀.
        """
        if self._current_joints is None:
            self.get_logger().error("Joint states not available.")
            return

        # 1) waypoint 생성
        current_q = np.array(self._current_joints)
        current_T = self._robot.forward_kinematics(current_q)
        base_pos = current_T[:3, 3].copy()

        n_points = int(QUEUE_TEST_DURATION * QUEUE_TEST_RATE_HZ)
        dt = 1.0 / QUEUE_TEST_RATE_HZ
        self.get_logger().info(
            f"[{timestamp()}] Queue sweep: n={n_points}, "
            f"dt={dt*1000:.0f}ms, amp={QUEUE_TEST_AMPLITUDE*100:.1f}cm X, "
            f"{QUEUE_TEST_CYCLES} cycles in {QUEUE_TEST_DURATION:.1f}s"
        )

        waypoints: list[tuple[np.ndarray, float]] = []
        for i in range(n_points):
            t_i = i * dt
            phase = 2.0 * np.pi * QUEUE_TEST_CYCLES * (t_i / QUEUE_TEST_DURATION)
            dx = QUEUE_TEST_AMPLITUDE * np.sin(phase)
            target_T = current_T.copy()
            target_T[:3, 3] = base_pos + np.array([dx, 0.0, 0.0])
            target_q = self._robot.inverse_kinematics(target_T)
            if target_q is None:
                self.get_logger().warn(
                    f"IK fail at pt {i} (dx={dx*100:.2f}cm); aborting sweep."
                )
                return
            # 첫 점을 현재 위치로 맞춤 → queue 시작 시 0 edge case 방지
            if i == 0:
                target_q = current_q.copy()
            waypoints.append((np.asarray(target_q, dtype=float), t_i))

        # 2) 현재 trajectory mode를 빠져나와야 queue mode 진입 가능
        if not self._stop_trajectory_mode():
            self._return_to_traj_mode()
            return

        # 3) Queue Mode 진입
        if not self._start_queue_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("/start_point_queue_mode unavailable.")
            self._return_to_traj_mode()
            return
        self.get_logger().info(f"[{timestamp()}] Entering queue mode...")
        fut = self._start_queue_client.call_async(StartPointQueueMode.Request())
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        res = fut.result()
        if res is None or res.result_code.value != 1:
            self.get_logger().error(
                f"start_point_queue_mode failed: "
                f"{res and res.message or 'timeout'}"
            )
            self._return_to_traj_mode()
            return
        self.get_logger().info(f"Queue mode entered: {res.message}")

        # 3) Sweep 스트리밍
        t_start = time.time()
        latencies: list[float] = []
        busy_count = 0
        failed = False
        for i, (q, t_i) in enumerate(waypoints):
            req = QueueTrajPoint.Request()
            req.joint_names = JOINT_NAMES
            req.point.positions = q.tolist()
            req.point.velocities = [0.0] * 6
            req.point.time_from_start = _seconds_to_duration(t_i)

            ok = False
            for attempt in range(QUEUE_BUSY_MAX_RETRY + 1):
                t_call = time.time()
                fut = self._queue_point_client.call_async(req)
                rclpy.spin_until_future_complete(self, fut, timeout_sec=2.0)
                res = fut.result()
                call_latency = time.time() - t_call
                if res is None:
                    self.get_logger().error(
                        f"pt {i}: service timeout after {call_latency*1000:.0f}ms"
                    )
                    failed = True
                    break
                code = res.result_code.value
                if code == 1:  # SUCCESS
                    latencies.append(call_latency)
                    ok = True
                    break
                if code == 4:  # BUSY
                    busy_count += 1
                    time.sleep(QUEUE_BUSY_RETRY_DELAY)
                    continue
                self.get_logger().error(
                    f"pt {i}: code={code} msg='{res.message}'"
                )
                failed = True
                break
            if failed:
                break
            if not ok:
                self.get_logger().warn(
                    f"pt {i}: dropped after {QUEUE_BUSY_MAX_RETRY} BUSY retries"
                )

        elapsed = time.time() - t_start

        # 4) 로봇이 마지막 점까지 실행 마치도록 여유 대기
        time.sleep(0.5)

        # 5) 요약
        self.get_logger().info(
            f"[{timestamp()}] Queue sweep DONE. "
            f"elapsed={elapsed:.3f}s (theoretical {QUEUE_TEST_DURATION:.3f}s), "
            f"overhead={(elapsed/QUEUE_TEST_DURATION - 1)*100:+.1f}%"
        )
        if latencies:
            arr = np.array(latencies) * 1000  # ms
            self.get_logger().info(
                f"Service latency: mean={arr.mean():.2f}ms "
                f"max={arr.max():.2f}ms p95={np.percentile(arr,95):.2f}ms"
            )
        self.get_logger().info(
            f"BUSY retries: {busy_count}/{n_points} "
            f"({100.0*busy_count/max(n_points,1):.1f}%)"
        )

        # 6) FJT 모드로 복귀
        self._return_to_traj_mode()

    def run_fjt_mismatch_test(self) -> None:
        """FJT에 traj[0]을 의도적으로 현재 state와 다르게 보내서 MotoROS2 반응 확인.

        목적: '현재 state != traj[0]이면 안 움직인다' 주장 검증.
        J1 에 약 0.5° (~50 pulse) offset → MotoROS2 10 pulse tolerance 충분히 초과.
        예상: INIT_TRAJ_INVALID_STARTING_POS (204) 로 reject.
        """
        if self._current_joints is None:
            self.get_logger().error("Joint states not available.")
            return

        # 테스트 전 state 스냅샷
        before_q = np.array(self._current_joints)

        offset_deg = 0.5
        offset_rad = np.radians(offset_deg)
        bad_start_q = before_q.copy()
        bad_start_q[0] += offset_rad     # J1 에만 offset
        target_q = before_q.copy()        # 움직일 필요 없음

        self.get_logger().info(
            f"=== FJT Mismatch Test ===\n"
            f"  current  J1 = {np.degrees(before_q[0]):+.4f}°\n"
            f"  traj[0]  J1 = {np.degrees(bad_start_q[0]):+.4f}° "
            f"(offset +{offset_deg}° ≈ {offset_rad*1e3:.2f} mrad)\n"
            f"  target   J1 = {np.degrees(target_q[0]):+.4f}°\n"
            f"  Tolerance: MotoROS2 ~10 pulse per axis (= 매우 작은 값)\n"
            f"  Expected : REJECT with code 204 if teammate is correct."
        )

        t0 = time.time()
        self._send_joint_goal(bad_start_q, target_q, 0.5)
        elapsed = time.time() - t0

        # 완료 후 실제 이동 여부 확인 (joint_state 갱신 대기)
        for _ in range(20):
            rclpy.spin_once(self, timeout_sec=0.05)
        after_q = np.array(self._current_joints)
        j1_moved_deg = float(np.degrees(after_q[0] - before_q[0]))

        self.get_logger().info(
            f"=== Result ===\n"
            f"  total elapsed = {elapsed*1000:.1f}ms "
            f"(즉시 reject라면 handshake만 ~30ms; 실행됐다면 ~500ms+)\n"
            f"  J1 displacement = {j1_moved_deg:+.4f}° "
            f"(0에 가까우면 REJECT, ~0.5°이면 ACCEPT/실행)\n"
            f"  → {'REJECTED (팀원 주장 검증됨 ✓)' if abs(j1_moved_deg) < 0.05 else 'EXECUTED (주장 반증)'}"
        )

    def _stop_trajectory_mode(self) -> bool:
        """현재 활성 trajectory/queue mode를 정지. 다음 모드 진입 전 필수."""
        if not self._stop_traj_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("/stop_traj_mode unavailable.")
            return False
        fut = self._stop_traj_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        res = fut.result()
        if res is None:
            self.get_logger().warn("stop_traj_mode timed out.")
            return False
        if not res.success:
            # 이미 정지 상태여도 stop은 보통 성공 반환. 실패는 알람 등 드문 케이스.
            self.get_logger().warn(f"stop_traj_mode: {res.message}")
            return False
        return True

    def _return_to_traj_mode(self) -> None:
        """Queue Mode → FJT(start_traj_mode) 복귀. 실패해도 경고만."""
        # Queue mode에서 벗어나려면 stop을 먼저 불러줘야 함
        self._stop_trajectory_mode()

        if not self._start_traj_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("start_traj_mode unavailable; manual recovery needed.")
            return
        fut = self._start_traj_client.call_async(StartTrajMode.Request())
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        res = fut.result()
        if res is None or res.result_code.value != 1:
            self.get_logger().warn(
                f"Failed to return to FJT mode: {res and res.message or 'timeout'}"
                " — try pressing 'q' and re-running script."
            )
            return
        self.get_logger().info(f"Returned to FJT mode: {res.message or 'READY'}")

    def raise_to_safe_height(self) -> None:
        """EE를 SAFE_HEIGHT까지 현재 자세 유지한 채로 최대 속도로 올림."""
        if self._current_joints is None:
            self.get_logger().error("Joint states not available.")
            return

        current_q = np.array(self._current_joints)
        current_T = self._robot.forward_kinematics(current_q)
        current_pos = current_T[:3, 3]

        if current_pos[2] >= SAFE_HEIGHT - 1e-3:
            self.get_logger().info(
                f"[{timestamp()}] Already at/above safe height "
                f"(z={current_pos[2]:.3f} ≥ {SAFE_HEIGHT:.3f})."
            )
            return

        target_T = current_T.copy()
        target_T[2, 3] = SAFE_HEIGHT

        target_q = self._robot.inverse_kinematics(target_T)
        if target_q is None:
            self.get_logger().warn(f"[{timestamp()}] IK failed for safe height.")
            return

        distance = SAFE_HEIGHT - current_pos[2]
        duration = max(FAST_MIN_DURATION, distance / FAST_MAX_VELOCITY)

        self._send_joint_goal(current_q, target_q, duration)
        self.get_logger().info(
            f"[{timestamp()}] EE raised to safe height: z={SAFE_HEIGHT:.3f} "
            f"(Δ={distance*100:.1f}cm, {duration:.2f}s)"
        )

    def move_to_floor(self) -> None:
        """EE를 workspace z 최저점까지 현재 자세 유지한 채로 최대 속도로 내림."""
        if self._current_joints is None:
            self.get_logger().error("Joint states not available.")
            return

        current_q = np.array(self._current_joints)
        current_T = self._robot.forward_kinematics(current_q)
        current_pos = current_T[:3, 3]
        floor_z = WORKSPACE["z"][0]

        if current_pos[2] <= floor_z + 1e-3:
            self.get_logger().info(
                f"[{timestamp()}] Already at floor (z={current_pos[2]:.3f})."
            )
            return

        target_T = current_T.copy()
        target_T[2, 3] = floor_z

        target_q = self._robot.inverse_kinematics(target_T)
        if target_q is None:
            self.get_logger().warn(f"[{timestamp()}] IK failed for floor position.")
            return

        distance = current_pos[2] - floor_z
        duration = max(FAST_MIN_DURATION, distance / FAST_MAX_VELOCITY)

        self._send_joint_goal(current_q, target_q, duration)
        self.get_logger().info(
            f"[{timestamp()}] EE descended to floor: z={floor_z:.3f} "
            f"(Δ={distance*100:.1f}cm, {duration:.2f}s)"
        )

    def level_suction(self) -> None:
        if self._current_joints is None:
            self.get_logger().error("Joint states not available.")
            return

        current_q = np.array(self._current_joints)
        link_frames, _ = self._robot.forward_kinematics_all(current_q)
        T6 = link_frames[5]
        T_tip = T6 @ T_CHAIN
        tip_pos = T_tip[:3, 3]

        tip_z = T_tip[:3, 2]
        if np.allclose(tip_z, [0, 0, -1], atol=0.02):
            self.get_logger().info(f"[{timestamp()}] Already vertical.")
            return

        z_desired = np.array([0.0, 0.0, -1.0])
        x_current = T_tip[:3, 0]
        x_proj = x_current - x_current[2] * np.array([0, 0, 1])
        x_norm = np.linalg.norm(x_proj)
        if x_norm < 1e-6:
            x_desired = np.array([1.0, 0.0, 0.0])
        else:
            x_desired = x_proj / x_norm
        y_desired = np.cross(z_desired, x_desired)
        y_desired /= np.linalg.norm(y_desired)
        x_desired = np.cross(y_desired, z_desired)

        T_tip_desired = np.eye(4)
        T_tip_desired[:3, 0] = x_desired
        T_tip_desired[:3, 1] = y_desired
        T_tip_desired[:3, 2] = z_desired
        T_tip_desired[:3, 3] = tip_pos

        T6_desired = T_tip_desired @ T_CHAIN_INV
        T_ee_desired = T6_desired @ self._robot.home_ee

        target_q = self._robot.inverse_kinematics(T_ee_desired)
        if target_q is None:
            self.get_logger().warn(f"[{timestamp()}] IK failed for leveling.")
            return

        self._send_joint_goal(current_q, target_q, MOVE_DURATION)
        self.get_logger().info(f"[{timestamp()}] Gripper leveled (vertical).")

    def toggle_suction(self) -> None:
        self._suction_on = not self._suction_on
        value = 0 if self._suction_on else 1
        req = WriteSingleIO.Request()
        req.address = SUCTION_IO_ADDRESS
        req.value = value
        future = self._io_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        result = future.result()
        state = "ON" if self._suction_on else "OFF"
        if result.success:
            self.get_logger().info(f"[{timestamp()}] Suction {state}")
        else:
            self.get_logger().error(f"[{timestamp()}] Suction {state} FAILED: {result.message}")

    def print_status(self) -> None:
        if self._current_joints is None:
            self.get_logger().warn("No joint states.")
            return

        T = self._robot.forward_kinematics(np.array(self._current_joints))
        pos = T[:3, 3]
        deg = np.degrees(self._current_joints)

        self.get_logger().info(
            f"[{timestamp()}] EE: x={pos[0]:.4f} y={pos[1]:.4f} z={pos[2]:.4f}"
        )
        self.get_logger().info(
            f"[{timestamp()}] Joints(deg): [{deg[0]:.1f}, {deg[1]:.1f}, "
            f"{deg[2]:.1f}, {deg[3]:.1f}, {deg[4]:.1f}, {deg[5]:.1f}]"
        )
        self.get_logger().info(
            f"[{timestamp()}] Suction: {'ON' if self._suction_on else 'OFF'}"
        )


def main():
    rclpy.init()
    node = RobotDebugger()

    if not node.wait_for_servers():
        node.destroy_node()
        rclpy.shutdown()
        return

    if not node.wait_for_joint_state():
        node.destroy_node()
        rclpy.shutdown()
        return

    if not node.start_traj_mode():
        node.destroy_node()
        rclpy.shutdown()
        return

    node.print_status()

    print("")
    print("=== GP8 Robot Debug Tool (ROS 2) ===")
    print(f"  w/s   : X +/-{STEP_SIZE*100:.0f}cm   (Shift: +/-{BIG_STEP_SIZE*100:.0f}cm)")
    print(f"  a/d   : Y +/-{STEP_SIZE*100:.0f}cm   (Shift: +/-{BIG_STEP_SIZE*100:.0f}cm)")
    print(f"  r/f   : Z +/-{STEP_SIZE*100:.0f}cm   (Shift: +/-{BIG_STEP_SIZE*100:.0f}cm)")
    print(f"  u/j   : Roll  +/-{np.degrees(ANGLE_STEP):.0f}°   (Shift: +/-{np.degrees(BIG_ANGLE_STEP):.0f}°)")
    print(f"  i/k   : Pitch +/-{np.degrees(ANGLE_STEP):.0f}°   (Shift: +/-{np.degrees(BIG_ANGLE_STEP):.0f}°)")
    print(f"  o/l   : Yaw   +/-{np.degrees(ANGLE_STEP):.0f}°   (Shift: +/-{np.degrees(BIG_ANGLE_STEP):.0f}°)")
    print("  SPACE : Suction toggle")
    print("  g     : Gripper level (수직 정렬)")
    print(f"  z     : Descend to floor z={WORKSPACE['z'][0]:.2f}m (max {FAST_MAX_VELOCITY:.1f} m/s)")
    print(f"  x     : Raise to safe height z={SAFE_HEIGHT:.2f}m (max {FAST_MAX_VELOCITY:.1f} m/s)")
    print(f"  h     : Home pose (0.4m forward, tool-down)")
    print(f"  t     : [TEST] Queue Mode X sweep ±{QUEUE_TEST_AMPLITUDE*100:.0f}cm ({QUEUE_TEST_DURATION:.0f}s)")
    print(f"  b     : [TEST] FJT with bad traj[0] (start-state mismatch test)")
    print("  p     : Print status")
    print("  q     : Quit")
    print("  * This terminal must be focused")
    print("=====================================")
    print("")

    try:
        while rclpy.ok():
            key = get_key()
            if key in KEY_DELTA:
                node.move_ee(KEY_DELTA[key])
                continue
            if key in KEY_ROT:
                axis, angle = KEY_ROT[key]
                node.rotate_ee(axis, angle)
                continue
            if key and key.isalpha():
                key = key.lower()
            if key == " ":
                node.toggle_suction()
            elif key == "g":
                node.level_suction()
            elif key == "z":
                node.move_to_floor()
            elif key == "x":
                node.raise_to_safe_height()
            elif key == "h":
                node.go_home()
            elif key == "t":
                node.run_queue_sweep_test()
            elif key == "b":
                node.run_fjt_mismatch_test()
            elif key == "p":
                node.print_status()
            elif key == "q":
                node.get_logger().info(f"[{timestamp()}] Exiting.")
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

"""GP8 실기 arm — 공식 `robot_env_dt_real.RoboticArm` 의 ROS2 이식.

`collect_data_real_gp8.py` 가 쓰는 실기 인터페이스. **로봇 컴퓨터에서만 실행**
(rclpy + gp8_control 필요). 공식 원본과의 대응은 각 메서드 docstring 에 줄번호로
남긴다. 원본: ThrowBot_official/src/scripts/decision_transformer/real_robot/
robot_env_dt_real.py (+ control_real_gp8.py 의 속도 계산).

공식과 같은 것:
  · 10 Hz 폐루프 — 매 스텝 **실측** /joint_states 기준으로 식 (5.1) 증분
    `q_wp = q_meas + ω·Δt` (:149-154). DT 가 실측 상태를 보고 다음 행동을 낸다.
  · waypoint 속도 = (p − p_old)/Δt  ('segment', control_real_gp8.py:107-115)
  · `proj_on_max_speed` 스케일 (:348-356), `get_state` = 실측 각도 + 1.0 (:298-313)
  · 릴리즈 = 그리퍼 액션 < τ (:382), 시간 초과 = number_steps (:20)

인터페이스 차이 (2026-08-05 사용자 지시 범위):
  · MotoROS point streaming (/joint_command) → 4 ms 위치 스트리밍
    (/JointGroupPositionController/commands). **MotoROS 가 하던 보간을 우리가
    돌린다** — gp8_interp.hermite_coefs 는 저자 포크 motoman_ps 의
    MotionServer.c:1414-1420 이식 (계수 일치 확인).
  · Robotiq 그리퍼 → 흡착. 릴리즈는 릴리즈 스텝 창의 **끝**에서 suction_off
    (env 규약 t_rel = (k_rel+1)·Δt 와 일치).
  · 릴리즈 후 followthrough (coast 1.0/0.5/0.15 스텝) 를 덧붙인다 — 학습 env
    (dt_gp8_env.FOLLOWTHROUGH) 와 같은 규칙. 공식 원본은 마지막 waypoint 에서
    그대로 정지하는데, 4 ms 스트리밍에서 그건 순간 정지 명령이라 위험하다.
    물체는 이미 손을 떠난 뒤라 데이터에는 영향이 없다.
  · 관절 위치를 URDF 한계로 clamp — 공식은 명시하지 않지만 실기에선 하드웨어가
    강제한다 (알람 대신 계획 단계에서 자른다).
"""

from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

# 로봇의 벤더링 트리 우선, 워크스테이션(Thr_DT/gp8) 폴백
try:
    from gp8_control.skills.thr.dt_gp8_env import (
        FOLLOWTHROUGH, GP8Config, PLANAR_IDX, Q_HI, Q_LO, QD_MAX,
        followthrough_nodes)
    from gp8_control.skills.thr.gp8_interp import hermite_coefs, hermite_eval
except ImportError:                                    # 워크스테이션 문법 검사용
    from dt_gp8_env import (FOLLOWTHROUGH, GP8Config, PLANAR_IDX, Q_HI, Q_LO,
                            QD_MAX, followthrough_nodes)
    from sim.gp8_interp import hermite_coefs, hermite_eval

# planner 프레임 → 로봇 프레임 (thr_planners._PLANNER_SIGN 과 동일)
_SIGN = np.array([1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
_STREAM_DT = 0.004                                     # JGPC 4 ms


def load_dt_model(weights):
    """체크포인트 → (model). 로봇의 thr_planners.load_dt 에 위임 —
    ckpt 에 기록된 τ·tau_act·qdd_factor 복원 로직을 재사용한다."""
    from gp8_control.skills import thr_planners as tp
    model, _arm_sim, _cfg, _torch = tp.load_dt(weights)
    model.state_dim = getattr(model, "state_dim", 5)
    model.act_dim = getattr(model, "act_dim", 4)
    return model


class GP8RealArm:
    """공식 RoboticArm 의 필드·메서드 시그니처를 유지한다 (collect 루프가
    그대로 돌도록): update_target / first_step / get_state / step / reset_arm
    / target / target_radius / gripper_thresh."""

    def __init__(self, vel_scale: float = 0.2, node_name="collect_real_gp8"):
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.node import Node
        from std_msgs.msg import Float64MultiArray
        from gp8_control.config import Config
        from gp8_control.controllers.trajectory_controller import (
            JGPC_COMMAND_TOPIC, TrajectoryController)
        from gp8_control.robots.gp8 import GP8
        from gp8_control.trajectory.trajectory_primitive import trajectory

        self._Float64MultiArray = Float64MultiArray
        self._trajectory = trajectory

        rclpy.init()
        self._rclpy = rclpy
        self.node = Node(node_name)
        self._exec = MultiThreadedExecutor()
        self._exec.add_node(self.node)
        self.traj_ctrl = TrajectoryController(self.node)
        threading.Thread(target=self._exec.spin, daemon=True).start()

        if not self.traj_ctrl.wait_for_servers(timeout_sec=10.0):
            raise RuntimeError("컨트롤러 액션 서버 없음 — bringup 확인")
        t_end = time.time() + 5.0
        while self.traj_ctrl.current_joints is None and time.time() < t_end:
            time.sleep(0.05)
        if self.traj_ctrl.current_joints is None:
            raise RuntimeError("/joint_states 미수신")

        self._pub = self.node.create_publisher(
            Float64MultiArray, JGPC_COMMAND_TOPIC, 10)

        self.cfg_sim = GP8Config()                      # 학습 env 상수 공유
        self.cfg_robot, self.robot = Config(), GP8()
        self.dt = 1.0 / self.cfg_sim.update_rate        # 0.1 s   (원본 :18)
        self.number_steps = int(self.cfg_sim.total_time
                                * self.cfg_sim.update_rate)      # :20
        self.target = np.zeros(3)
        self.target_radius = 0.1                        # :33
        self.gripper_thresh = 0.8385                    # ckpt τ 로 덮어쓸 것
        self.max_speed = np.append(np.rad2deg(QD_MAX), 1.0)  # :45 (planar 3+grip)
        self.max_speed_factor = 1.0                     # :52
        self._vel_scale = float(vel_scale)

        # planner 프레임 홈 자세 → 로봇 6축
        self.home_q6 = self._to_robot(np.asarray(self.cfg_sim.home_pose, float))
        self.idle_q6 = self.home_q6.copy()

        # 4 ms 스트리머: (q6_robot, fire_release) 를 순서대로 발행
        self._samples: deque = deque()
        self._stream_lock = threading.Lock()
        self._stream_stop = False
        self._streamer = threading.Thread(target=self._stream_loop, daemon=True)
        self._streamer.start()

        self.curr_step = 0
        self._q_wp = None                               # 직전 waypoint (planner 3축)
        self._v_wp = np.zeros(3)

    # ------------------------------------------------------------------
    # 프레임 변환
    # ------------------------------------------------------------------
    @staticmethod
    def _to_robot(q3_planner, yaw=0.0):
        q6 = np.zeros(6)
        q6[0] = yaw
        q6[PLANAR_IDX] = np.asarray(q3_planner, float)
        return q6 * _SIGN

    def _measured_planner3(self):
        q6 = np.asarray(self.traj_ctrl.current_joints, float) * _SIGN
        return q6[PLANAR_IDX]

    # ------------------------------------------------------------------
    # 4 ms 스트리머 (MotoROS interpolPeriod 발행 루프의 대응물)
    # ------------------------------------------------------------------
    def _stream_loop(self):
        next_t = time.time()
        while not self._stream_stop:
            item = None
            with self._stream_lock:
                if self._samples:
                    item = self._samples.popleft()
            if item is not None:
                q6, fire = item
                msg = self._Float64MultiArray()
                msg.data = [float(v) for v in q6]
                self._pub.publish(msg)
                if fire:
                    self.traj_ctrl.suction_off()        # 릴리즈 창 끝에서 발화
            next_t += _STREAM_DT
            d = next_t - time.time()
            if d > 0:
                time.sleep(d)
            else:
                next_t = time.time()                    # 밀리면 리셋 (soft RT)

    def _enqueue_segment(self, q0, v0, q1, v1, fire_release=False):
        """(q0,v0)→(q1,v1) 한 구간(Δt=0.1 s)을 MotoROS 3차식으로 4 ms 전개."""
        a1, a2 = hermite_coefs(q0, q1, v0, v1, self.dt)
        tau = np.arange(_STREAM_DT, self.dt + _STREAM_DT / 2, _STREAM_DT)
        pos, _vel = hermite_eval(q0, v0, a1, a2, tau)
        pos = np.clip(pos, Q_LO, Q_HI)                  # URDF 한계 (하드웨어 대행)
        with self._stream_lock:
            for i, q3 in enumerate(pos):
                last = (i == len(pos) - 1)
                self._samples.append(
                    (self._to_robot(q3), fire_release and last))

    # ------------------------------------------------------------------
    # 공식 API
    # ------------------------------------------------------------------
    def update_target(self, target):                    # :117
        self.target = np.asarray(target, float)

    def proj_on_max_speed(self, velocity_vector):       # :348-356
        v = np.asarray(velocity_vector, float) * self.max_speed \
            * self.max_speed_factor
        v = np.deg2rad(v)
        v[-1] = velocity_vector[-1]
        return v

    def get_state(self):                                # :298-313
        angles = self._measured_planner3()
        return np.append(angles, 1.0)                   # 원본도 그리퍼 자리 상수 1.0

    def first_step(self, action):                       # collect :59
        """홈 복귀 → 흡착 → 스텝 카운터 초기화 → 스텝 0 실행."""
        self._move_slow(self.home_q6)
        input("  물체를 흡착판에 대고 Enter: ")
        self.traj_ctrl.suction_on()
        time.sleep(1.0)
        self.curr_step = 0
        self._q_wp = self._measured_planner3()
        self._v_wp = np.zeros(3)
        done, _ = self.step(np.asarray(action, float))
        return not done

    def step(self, velocity_vector):                    # :358-...
        """공식과 같은 순서: 스케일 → 식 (5.1) 실측 기준 증분 → waypoint 발행.
        리턴 (done, termination_reason) — 원본 실기 규약."""
        velocity_vector = np.asarray(velocity_vector, float)
        v = self.proj_on_max_speed(velocity_vector)
        w, gripper = v[:3], float(velocity_vector[-1])

        q_meas = self._measured_planner3()              # :149 실측 기준 (폐루프)
        q_new = np.clip(q_meas + w * self.dt, Q_LO, Q_HI)
        v_new = (q_new - self._q_wp) / self.dt          # segment (control_real:107)

        release = gripper < self.gripper_thresh         # :382
        self._enqueue_segment(self._q_wp, self._v_wp, q_new, v_new,
                              fire_release=release)
        self._q_wp, self._v_wp = q_new, v_new

        self.curr_step += 1
        time.sleep(self.dt)                             # :385-388 10 Hz 페이싱

        if release:
            self._followthrough()
            return True, f"Gripper was opened with value: {gripper}"
        if self.curr_step >= self.number_steps:         # :20
            self._followthrough()
            return True, f"Time is up: {self.curr_step * self.dt}"
        return False, None

    def _followthrough(self):
        """릴리즈/종료 후 감속 — 학습 env 와 같은 coast (1.0, 0.5, 0.15)·Δt."""
        nodes = followthrough_nodes(self._q_wp, self._v_wp, self.dt)
        q, v = self._q_wp, self._v_wp
        for f, q_next in zip(FOLLOWTHROUGH, nodes):
            v_next = (q_next - q) / self.dt
            self._enqueue_segment(q, v, np.asarray(q_next, float), v_next)
            q, v = np.asarray(q_next, float), v_next
        # 마지막: 속도 0 으로 한 구간 더 (정지)
        self._enqueue_segment(q, v, q, np.zeros(3))
        # 스트림이 빌 때까지 대기
        while True:
            with self._stream_lock:
                if not self._samples:
                    break
            time.sleep(0.02)

    def reset_arm(self, angle=None):                    # :185
        self._move_slow(self.idle_q6)

    # ------------------------------------------------------------------
    def _move_slow(self, q6_goal):
        """느린 점대점 복귀 (JTC 큐) — 던지기와 무관한 접근/복귀 이동."""
        q_cur = np.asarray(self.traj_ctrl.current_joints, float)
        if np.max(np.abs(q_cur - q6_goal)) < np.deg2rad(0.5):
            return
        zero6 = np.zeros(6)
        M1 = self.robot.M1 * self._vel_scale if hasattr(self.robot, "M1") \
            else np.full(6, 0.5) * self._vel_scale
        M2 = M1 * getattr(self.cfg_robot, "JOINT_ACCEL_LIMIT_SCALE", 3.0)
        mv = self._trajectory(q_cur, zero6, q6_goal, zero6, M1, M2,
                              hertz=self.cfg_robot.TRAJ_HZ)
        self.traj_ctrl.send_trajectory_queue(mv[0], mv[1], mv[2],
                                             final_joint=q6_goal)

    def shutdown(self):
        self._stream_stop = True
        try:
            self.traj_ctrl.suction_off()
            self.traj_ctrl.close()
        finally:
            self._rclpy.shutdown()

"""
THR GP8 rig용 Decision Transformer 학습 환경 (Thr_DT 논문 절차의 env 교체판).

왜 필요한가 — Thr_DT(/PublicSSD/ryugaeun/Thr_DT)의 사전학습 가중치를 THR sim에
그대로 이식하면 0/12다 (2026-07-30 측정). 원인 두 가지를 진단했다:
  ① Thr_DT의 평면 시뮬 env(env/throw_env.py)에는 관절 '위치' 한계가 없다
     (속도 한계만). 학습된 정책은 평면 t₃를 −187°까지 휘두르는데, 이는 GP8 U축
     한계(planner 프레임 +70°)를 릴리즈 2스텝 전부터 위반한다. 한계로 클립하면
     스윙이 망가져 물체가 로봇 뒤(x<0)로 날아간다.
  ② 그 평면 시뮬의 도달 능력 자체가 THR rig에 못 미친다. GP8 위치 한계를 걸고
     무작위 스윙 600회를 굴리면 최대 착지 0.945m (한계 없이도 1.31m)인데,
     THR bin은 1.20~1.70m다 — 학습을 잘 해도 구조적으로 도달 불가.
논문 §5.1은 시뮬 동역학 파라미터가 "arbitrary while in the same scale of the
real one"이라고 명시하므로, Thr_DT의 평면 시뮬은 THR의 GP8 rig와 '다른 로봇'인
셈이다. 따라서 가중치를 이식하는 대신 **논문의 방법론(무작위 OU 데이터 수집 →
HER 목표 재라벨 → DT 오프라인 학습)을 THR rig에 그대로 적용**한다. 이 파일은
그때 쓰는 env이고, Thr_DT의 model/trainer/HER/OU 코드는 수정 없이 재사용한다
(dt_train_gp8.py가 env만 주입).

논문 규약 유지 (Thr_DT/config.py와 동일 출처):
  · 10 Hz 제어, 최대 1초 = 10 스텝                                  [§5.1]
  · 평면 던지기: GP8의 2·3·5번 관절(L/U/B)만 사용, 던지는 방향은
    1번 관절(S)로 해석적 설정 — 이 env는 S=0 평면에서 학습한다      [§5.1]
  · 상태 (θ_L, θ_U, θ_B, θ_gr) + 목표거리 d_g, 행동 (ω_L, ω_U, ω_B, a_gr) [§4.1]
  · 위치 명령 θ_t = θ_{t−1} + ω_t/f (Taylor 근사) + 1차 actuator lag  [식 5.1]
  · 그리퍼는 a_gr ≤ τ에서 열림 (τ = 학습셋 그리퍼 액션 평균)         [§5.1]
  · 희소 보상 ±1 (반경 ρ), 착지점은 첫 낙하 지점                     [식 4.2, §4.1]
  · 비행은 무항력 포물선 — 시뮬(THR)은 quadratic drag이므로 갭이 남지만,
    NLP(PLAN_WITH_DRAG=False)·Physics-only도 같은 무항력 가정이라 공정하다.

THR rig로 교체되는 부분 (모두 실제 GP8 값):
  · FK/Jacobian: throwing.py의 6축 GP8 체인 (S=0, R=T=0 평면 단면)
  · 최대 속도: GP8_QD_MAX[L,U,B] = (385, 520, 550) deg/s
    — 논문이 명시한 값과 정확히 일치하므로 규약 변경이 아니다
  · 위치 한계: GP8_Q_MIN/MAX[L,U,B] = L[−65,145] U[−190,70] B[−135,135]°
    (Thr_DT env에 없던 제약 — 이걸 넣는 것이 ①의 수정)
  · 목표 거리 범위: THR bin 실측 범위를 포함하도록 DG_MIN/DG_MAX
"""

import os

import numpy as np

from .throwing import (G, GP8_Q_MAX, GP8_Q_MIN, GP8_QD_MAX, fk_pos, fk_frames,
                      jacobian)

# planner 프레임 6축 중 평면 던지기에 쓰는 3축 (L, U, B) — 논문 §5.1의 2·3·5번
PLANAR_IDX = [1, 2, 4]

# 위치 한계 두 가지 (2026-08-03 사용자 지시로 기본을 NLP과 동일하게 맞춤).
#   'urdf' : GP8 URDF 원 한계 — 기구학적으로 가능한 전 범위
#   'nlp'  : THR NLP 플래너가 실제로 쓰는 한계 (throw_nlp.Q_LO/Q_HI의 평면 3축).
#            URDF ∩ 실기 한계에 사용자 작업 규칙까지 적용한 값이라 훨씬 좁다.
#            여기에 하드코딩한 이유: throw_nlp는 casadi를 import하는데 DT 실행
#            인터프리터(Thr_Phy/.venv)에는 casadi가 없다. throw_nlp.py를 고치면
#            아래 값도 같이 고쳐야 한다(검증: tests 없이 수동 대조 2026-08-03).
#              L(idx1) [-15, 45] deg   ← throw_nlp Q_LO[1]/Q_HI[1]
#              U(idx2) [-30, 45] deg   ← throw_nlp Q_LO[2]/Q_HI[2] (URDF 기준
#                                        [-45,+30]의 부호 반전 — planner 컨벤션)
#              B(idx4) [ 10,135] deg   ← B_LO_DEG=10, 실기 상한 135
LIMIT_MODE = os.environ.get("DT_GP8_LIMITS", "nlp")
Q_LO_NLP = np.deg2rad([-15.0, -30.0, 10.0])
Q_HI_NLP = np.deg2rad([45.0, 45.0, 135.0])
if LIMIT_MODE == "nlp":
    Q_LO, Q_HI = Q_LO_NLP, Q_HI_NLP
else:
    Q_LO = GP8_Q_MIN[PLANAR_IDX]
    Q_HI = GP8_Q_MAX[PLANAR_IDX]
QD_MAX = GP8_QD_MAX[PLANAR_IDX]          # (385, 520, 550) deg/s — 논문과 일치

# 목표 거리 범위: THR bin 조준점 수평거리는 1.20~1.70 m. 학습 범위는 그보다
# 넓게 잡아 경계 근처의 외삽을 피한다 (논문 [0.5, 2.0] m 정신 유지).
DG_MIN, DG_MAX = 0.6, 2.0


SIM_DT = 1.0 / 240.0        # THR sim 스텝 (sim_env.DT와 같아야 함)

# release 후 팔로우스루 노드 배수 — 궤적이 release에서 끝나면 지터의 '+' 방향이
# 궤적 밖으로 클립되어 무효가 되므로 coast 1스텝 + 감속 2스텝을 덧붙인다.
# dt_cem.traj_from_seq과 **같은 규칙**을 써야 학습(env)과 실행(시뮬)이 일치한다.
FOLLOWTHROUGH = (1.0, 0.5, 0.15)


def followthrough_nodes(q_last, qd_seg, dt, lo=None, hi=None):
    """release 직후 스텝 노드들 (위치 한계로 클램프)."""
    lo = Q_LO if lo is None else lo
    hi = Q_HI if hi is None else hi
    q = np.asarray(q_last, float).copy()
    out = []
    for f in FOLLOWTHROUGH:
        q = np.clip(q + np.asarray(qd_seg, float) * f * dt, lo, hi)
        out.append(q.copy())
    return out


def q6(q3, yaw=0.0):
    """평면 3축 (θ_L, θ_U, θ_B) → GP8 planner 6축 (S=yaw, R=T=0)."""
    q = np.zeros(6)
    q[0] = yaw
    q[PLANAR_IDX] = np.asarray(q3, float)
    return q


class GP8Config:
    """Thr_DT SimConfig과 같은 필드 이름을 갖는 THR rig 설정 (duck typing).
    Thr_DT의 collect_data/train_dt_offline/evaluate_dt가 cfg에서 읽는 값들을
    그대로 제공한다."""

    # --- 제어 루프 [§5.1] ---
    update_rate = 10.0
    total_time = 1.0
    smooth_factor = 0.0              # 속도 상보필터 비활성 [repo]

    # --- 관절 [§5.1] + GP8 실측 한계 ---
    joint_names = ('joint_2_l', 'joint_3_u', 'joint_5_b', 'finger_joint')
    max_speed_deg = tuple(np.rad2deg(QD_MAX))     # (385, 520, 550)
    max_speed_factor = 1.0

    # --- 목표/보상 ---
    dg_min = DG_MIN
    dg_max = DG_MAX
    target_radius = 0.1             # ρ [repo env/robot_env_dt.py:31]
    gripper_thresh = 0.8385         # τ 초기값 (학습셋 평균으로 대체됨) [§5.1]

    # --- 시작 자세 (planner 프레임 L, U, B) ---
    # Thr_DT home (t₂,t₃,t₅)=(0.5,−0.3,−1.5)의 GP8 대응 자세 (부호 규약
    # q_L=t₂, q_U=−t₃, q_B=−t₅ — dt_planner.py의 FK 대조로 검증). GP8에서
    # TCP=(0.271, 0, 0.193)로 물체를 든 시작 높이(P_START_Z=0.20)와 맞고,
    # 세 관절 모두 위치 한계 안이다.
    home_pose = (0.5, 0.3, 1.5)

    # --- 적분/구동 동역학 ---
    substeps = 10
    tau_act = 0.05                  # 1차 actuator lag [§5.1 "arbitrary"]
    gravity = G                      # 9.81 — THR sim과 동일

    # 가속도 한계: |ω_t − ω_{t−1}| ≤ qdd_max·Δt, qdd_max = 5×속도한계.
    # 논문에도 공식 코드에도 없다 (Gazebo에선 링크 관성이 알아서 강제). 실제
    # 드라이브는 100 ms 안에 속도를 불연속으로 못 바꾸므로 명시적으로 건다.
    # 계수: THR NLP 규약은 5.0이지만 NLP은 매끄러움 목적항(W_ACC)이 있어 그
    # 여유를 다 쓰지 않는다. DT는 OU 노이즈 데이터를 모방하므로 허용된 만큼
    # 스텝마다 가속/감속을 뒤집어 눈에 보이는 떨림이 생긴다(가속↔감속 전환
    # 관절당 3~6회/던지기, 2026-08-03 측정). 2.0이면 최대속도 도달에 0.5 s로
    # 산업용 로봇에 현실적이고, 스텝당 Δω 허용이 절반(193→77 deg/s)이 되며,
    # 도달 능력은 3.29 m로 THR bin(1.10~1.62 m)에 2배 여유가 남는다.
    qdd_factor = 2.0
    enforce_acceleration_limit = True

    # 착지 기준면 z [m]. DT는 "어떤 수평면 위의 목표"에 던지도록 학습되는데
    # (§4.1 "a position on some horizontal plane"), THR의 bin은 pit 안에 있어
    # 조준점 12개가 전부 z=−0.08 이다 (sim_env.bin_targets()로 확인). 여기를
    # 0으로 두면 DT가 바닥(z=0) 기준으로 계산해 8 cm 더 낙하하는 동안 수평으로
    # 120~155 mm 더 나가는 계통 오버슛이 생긴다 (2026-08-03 실측: 상수 보정만
    # 걸어도 명중 30.6%→50.0%). 학습 자체를 실제 조준면에서 하는 것이 정답.
    z_land = -0.08

    # --- 물체 ---
    # THR의 던질 물체는 4cm 큐브~17cm 망치. 착지 판정 높이는 논문처럼 물체
    # 반대각(여기서는 4cm 큐브 기준)으로 두고, 실제 bin 판정은 THR sim이 한다.
    object_half_diag = 0.5 * 0.04 * np.sqrt(2)

    @property
    def dt(self):
        return 1.0 / self.update_rate

    @property
    def number_steps(self):
        return int(self.total_time * self.update_rate)     # = 10

    @property
    def max_speed_rad(self):
        return np.asarray(QD_MAX, float)


class GP8ThrowArm:
    """GP8 평면 던지기 arm — Thr_DT env/throw_env.RoboticArm과 같은 공개 API
    (reset / update_target / get_state / step / reward_sparse / 속성들)라서
    Thr_DT의 수집·학습·평가 코드를 그대로 쓸 수 있다."""

    def __init__(self, cfg=None, rng=None, interp=None):
        """interp: None이면 시뮬이 계획 궤적을 그대로 재생한다고 가정.
        dict(traj_hz=…, period=…)를 주면 **시뮬과 똑같이** Yaskawa 컨트롤러
        보간(MotoROS 3차 Hermite)을 거친 궤적에서 release 상태를 계산한다.
        학습 데이터 수집·평가 환경이 실행 환경과 어긋나면(보간 유무 불일치)
        DT가 배운 착탄 라벨이 실제와 달라지므로, 보간 조건마다 데이터를 따로
        수집해야 한다 (2026-07-31 사용자 지시)."""
        self.cfg = cfg if cfg is not None else GP8Config()
        self.rng = rng if rng is not None else np.random.default_rng()
        self.interp = interp
        self.UPDATE_RATE = self.cfg.update_rate
        self.total_time = self.cfg.total_time
        self.number_steps = self.cfg.number_steps
        self.no_rotation = True
        self.smooth_factor = self.cfg.smooth_factor
        self.number_states = 1
        self.her = True
        self.target_radius = self.cfg.target_radius
        self.joints = np.array(self.cfg.joint_names)
        self.max_speed = np.append(np.asarray(self.cfg.max_speed_deg, float), 1.0)
        self.max_speed_factor = self.cfg.max_speed_factor
        self.gripper_thresh = self.cfg.gripper_thresh
        self.target = np.array([1.0, 0.0, 0.0])
        self.reset()
        self.initial_pos = self.object_position.copy()

    # ------------------------------------------------------------------ FK
    def tcp(self, q3=None):
        """평면 단면의 TCP (x, z) — GP8 6축 FK에서 y=0."""
        p = fk_pos(q6(self.q if q3 is None else q3))
        return np.array([p[0], p[2]])

    def tcp_velocity(self, q3, qd3):
        """TCP 선속도 (vx, vz) — GP8 geometric Jacobian (S=R=T=0 평면)."""
        Jv, _ = jacobian(q6(q3))
        qd = np.zeros(6)
        qd[PLANAR_IDX] = np.asarray(qd3, float)
        v = Jv @ qd
        return np.array([v[0], v[2]])

    # --------------------------------------------------------------- reset
    def reset(self):
        self.q = np.array(self.cfg.home_pose, dtype=np.float64)
        self.qdot = np.zeros(3)
        self.qdot_seg = np.zeros(3)     # 스텝 구간 평균 속도 (release용)
        self.qdot_cmd_prev = np.zeros(3)  # 가속도 rate limit용 직전 명령속도
        self.q_hist = [self.q.copy()]   # 스텝 노드 이력 (보간 release 계산용)
        self.gripper_closed = True
        self.velocity = np.zeros(4)
        self.curr_step = 0
        self.curr_time = 0.0
        tip = self.tcp()
        self.object_position = np.array([tip[0], 0.0, tip[1]])
        self.object_height = tip[1]

    # -------------------------------------------------------------- target
    def update_target(self, target):
        self.target = np.asarray(target, dtype=np.float64)

    # --------------------------------------------------------------- state
    def get_state(self):
        """(θ_L, θ_U, θ_B, θ_gr) — θ_gr는 이진 (1=닫힘) [§4.1]."""
        return np.array([self.q[0], self.q[1], self.q[2],
                         1.0 if self.gripper_closed else 0.0])

    # -------------------------------------------------------------- reward
    def reward_sparse(self, obj_pos=None, target=None):
        """희소 보상 [식 4.2] — 착지가 시작 위치보다 앞이어야 성공 [repo]."""
        target = self.target if target is None else target
        obj_pos = self.object_position if obj_pos is None else obj_pos
        d = np.hypot(obj_pos[0] - target[0], obj_pos[1] - target[1])
        if d <= self.target_radius and obj_pos[0] > self.initial_pos[0]:
            return 1.0
        return -1.0

    # ------------------------------------------------------------- helpers
    def smooth_velocity(self, new_velocity):
        old = np.asarray(self.velocity)
        sm = old * self.smooth_factor + new_velocity * (1 - self.smooth_factor)
        sm[-1] = new_velocity[-1]
        return sm

    def proj_on_max_speed(self, velocity_vector):
        """[-1,1] 행동 → rad/s 명령 (그리퍼 성분은 그대로) [repo]."""
        v = velocity_vector * self.max_speed * self.max_speed_factor
        v = v * np.pi / 180.0
        v[-1] = velocity_vector[-1]
        return v

    @property
    def max_speed_rad_arm(self):
        return np.asarray(QD_MAX, float) * self.max_speed_factor

    def _integrate_control_step(self, qdot_cmd):
        """1/f 한 주기 적분: 위치 명령 θ_t = θ_{t−1} + ω_t/f [식 5.1]를 1차
        actuator lag로 추종. Thr_DT env와 같은 적분기에 **GP8 위치 한계**를
        추가한다 (한계에 닿은 관절은 속도 0 — 실기의 리미트 정지).

        self.qdot(순간 속도)과 별도로 self.qdot_seg(이 스텝 구간의 평균 속도
        = (q_k − q_{k−1})/Δt)를 기록한다. release 속도로는 **후자**를 쓴다:
        THR sim은 10 Hz 스텝 궤적을 시뮬 그리드로 보간해 kinematic replay하고
        release 상태를 그 궤적의 수치 미분에서 취하므로(sim_env
        _release_state_from_plan), 구간 평균이 시뮬에서 실제로 발현되는 속도다.
        순간 속도를 쓰면 env가 1.70 m로 예측한 던지기가 시뮬에서 1.04 m로
        떨어지는 계통 편향이 생긴다 (2026-07-30 측정)."""
        c = self.cfg
        h = c.dt / c.substeps
        q_prev = self.q.copy()
        q_cmd = np.clip(self.q + qdot_cmd * c.dt, Q_LO, Q_HI)
        for _ in range(c.substeps):
            qdot_des = (q_cmd - self.q) / c.tau_act
            qdot_des = np.clip(qdot_des, -self.max_speed_rad_arm,
                               self.max_speed_rad_arm)
            q_new = np.clip(self.q + qdot_des * h, Q_LO, Q_HI)
            self.qdot = (q_new - self.q) / h      # 한계에 걸리면 실제 속도도 0
            self.q = q_new
        self.qdot_seg = (self.q - q_prev) / c.dt
        self.q_hist.append(self.q.copy())
        return self.tcp()

    def release_state(self):
        """release 시각의 평면 TCP (위치, 속도) — **시뮬과 같은 파이프라인**.
        interp가 없으면 현재 자세 + 구간 평균 속도(시뮬의 궤적 미분과 동형),
        있으면 스텝 궤적(+팔로우스루)을 시뮬 그리드로 편 뒤 컨트롤러 보간을
        거친 궤적에서 t_rel의 자세·속도를 읽는다."""
        if not self.interp:
            return self.tcp(), self.tcp_velocity(self.q, self.qdot_seg)
        from sim.gp8_interp import controller_track

        nodes = list(self.q_hist) + followthrough_nodes(
            self.q, self.qdot_seg, self.cfg.dt)
        Qp = np.asarray(nodes, float)
        t_nodes = np.arange(len(Qp)) * self.cfg.dt
        # 시뮬(dt_planner.gp8_dt_traj_fn)이 만드는 것과 같은 240 Hz 궤적
        ts = np.arange(0.0, t_nodes[-1] + SIM_DT / 2, SIM_DT)
        Q = np.column_stack([np.interp(ts, t_nodes, Qp[:, j]) for j in range(3)])
        Qd = np.gradient(Q, ts, axis=0)
        t_c, Q_c, V_c = controller_track(
            ts, Q, Qd, traj_hz=self.interp.get("traj_hz"),
            period=self.interp.get("period"), out_dt=SIM_DT)
        t_rel = self.curr_step * self.cfg.dt          # 스텝 실행 직후 시각
        q_r = np.array([np.interp(t_rel, t_c, Q_c[:, j]) for j in range(3)])
        qd_r = np.array([np.interp(t_rel, t_c, V_c[:, j]) for j in range(3)])
        return self.tcp(q_r), self.tcp_velocity(q_r, qd_r)

    def _ballistic_landing(self, pos, vel):
        """release 상태 (pos=(x,z), vel=(vx,vz))의 첫 지면 도달 x [§4.1].
        무항력 포물선 — 논문 env와 동일 가정."""
        g = self.cfg.gravity
        z0 = pos[1] - self.cfg.object_half_diag - getattr(self.cfg, 'z_land', 0.0)
        if z0 <= 0.0:
            return pos[0]
        vz = vel[1]
        t_land = (vz + np.sqrt(vz ** 2 + 2.0 * g * z0)) / g
        return pos[0] + vel[0] * t_land

    # ----------------------------------------------------------------- step
    def step(self, velocity_vector):
        """10 Hz 제어 한 스텝 — Thr_DT env.step과 같은 반환 규약:
        (reward, done, termination_reason, obj_pos, success)."""
        velocity_vector = np.asarray(velocity_vector, dtype=np.float64)
        velocity_vector = self.proj_on_max_speed(velocity_vector)
        velocity_vector = self.smooth_velocity(velocity_vector)
        if getattr(self.cfg, 'enforce_acceleration_limit', False):
            dw = self.cfg.qdd_factor * np.asarray(QD_MAX, float) * self.cfg.dt
            velocity_vector[:3] = np.clip(velocity_vector[:3],
                                          self.qdot_cmd_prev - dw,
                                          self.qdot_cmd_prev + dw)
        self.qdot_cmd_prev = velocity_vector[:3].copy()
        self.velocity = velocity_vector
        qdot_cmd = velocity_vector[:3]
        gripper = velocity_vector[-1]

        tip = self._integrate_control_step(qdot_cmd)
        self.curr_time += self.cfg.dt
        self.curr_step += 1

        if self.gripper_closed:
            self.object_position = np.array([tip[0], 0.0, tip[1]])
            self.object_height = tip[1]

        if self.curr_step >= self.number_steps or gripper < self.gripper_thresh:
            done = True
            if gripper < self.gripper_thresh:
                termination_reason = f"Gripper was opened with value: {gripper}"
                # release 자세·속도는 시뮬과 같은 파이프라인으로 (보간 반영)
                tip, tip_vel = self.release_state()
                x_land = self._ballistic_landing(tip, tip_vel)
                self.gripper_closed = False
                self.object_position = np.array(
                    [x_land, 0.0, self.cfg.object_half_diag])
                self.object_height = self.object_position[2]
                reward = self.reward_sparse()
                success = True
            else:
                termination_reason = f"Time is up: {self.curr_time}"
                reward = -1.0
                success = False
        else:
            if self.object_height <= self.cfg.object_half_diag:
                termination_reason = ("Object is too close to ground: "
                                     f"{self.object_height}")
                done = True
                reward = -1.0
                success = False
            else:
                termination_reason = None
                done = False
                reward = 0.0
                success = False
        return (reward, done, termination_reason, self.object_position.copy(),
                success)

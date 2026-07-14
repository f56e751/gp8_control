"""석션 단독 디버그 — 컨베이어 정지 상태, 지정 지점에서 pick + throw.

기본 사이클: 지정한 베이스 좌표 (x, y) 위 석션 대기 높이(Config.GRASP_Z —
throw_skill 이 ambush 대기에 쓰는 그 그랩 높이)로 내려가 파킹 → 석션 ON(기본
1초 홀드, 진공 형성) → 10 cm 리프트. 이후 선택적으로 **throw**:

  * release XY = pick→bin 방향 10 cm, release Z = pick +33 cm을 우선
    사용한다. 계획이 실패하면 거리 10~25 cm, Z 20~60 cm에서
    제한을 만족하는 가장 가까운 release를 자동 선택한다.
  * release→bin 포물선은 두 지점을 연결하는 최소 속도 탄도를 계산한다.
    release에서 툴 축은 투척 전방 기준 아래 30°다.
  * 스윙은 TCP에 원호를 강제하지 않는다. release pose와 탄도 선속도,
    투척 평면 내 각속도를 spatial Jacobian으로 관절 상태로 바꾸고,
    그 상태를 중간점으로 하는 와인드업→release→팔로스루를 만든다.
    관절 가속도는 제한의 90% 이내이고, 양끝 5% 구간에 jerk ramp를 넣어
    정지점과 release에서 가속도가 0으로 연속이다. 스윙 전체에서 J5 현장
    상한(+60.776°) 및 전 관절 위치/속도 제한을 검사한다. 석션 off는
    send_trajectory_queue_with_timed_release로 release knot에 동기한다.

app.py / 카메라 / 컨베이어 / torch 의존 없음. 로봇 드라이버 bringup
(~/ros2_ws/debug_bringup.sh = adv4ncr ros2_control + inactive JTC + MoveIt)이
떠 있어야 하고, 석션 IO 는 GP8_ROBOT_IP(기본 192.168.255.1):50242 로 나간다.

실행:
  ros2 run gp8_control suction_lift_debug              # 실행 (빌드 필요)
  ros2 run gp8_control suction_lift_debug --plan-only  # 로봇 없이 throw 계획만 검증

옵션:
  --x / --y     그랩 지점 XY [m] (기본 0.55, 0.0)
  --z           그랩 높이 [m] (기본 = Config.GRASP_Z)
  --hold        석션 ON 후 파킹 홀드 시간 [s] (기본 1.0)
  --lift        리프트 높이 [m] (기본 0.10)
  --vel-scale   저속 이동 속도 스케일 (기본 0.3)
  --bin-x/--bin-y  bin XY 위치 [m] (기본 1.5, 0.0)
  --bin-z-offset   bin 목표 높이 = 그랩 높이 + offset [m] (기본 0.10)
  --release-distance pick→bin 방향 release 기준 거리 [m] (기본 0.10)
  --release-z-offset release Z 기준 offset [m] (기본 0.33)
  --tool-offset     gp8.py EE/TCP 위에 추가 적용할 offset [m] (기본 0.0;
                   일반적으로 사용하지 않음)
  --release-lead   석션 off 를 release knot 보다 이만큼 일찍 발화 [s] (기본 0.0;
                   IO/배기 지연 보정용, 음수 = 늦게)
  --rviz-preview    로봇에 명령을 보내지 않고 RViz용 marker/path/joint_states publish

⚠️  throw 는 실제 고속 스윙(기본 ~2.5 m/s)입니다. bin 지점에 통/마커를 두고
    로봇 전방을 완전히 비울 것.
"""

from __future__ import annotations

import argparse
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from gp8_control.config import Config
from gp8_control.robots.gp8 import GP8
from gp8_control.trajectory.trajectory_primitive import trajectory

JOINT_NAMES = [
    "joint_1_s", "joint_2_l", "joint_3_u",
    "joint_4_r", "joint_5_b", "joint_6_t",
]

# perception.detection_intake._R_GRASP_DEFAULT 와 동일 — 툴이 벨트를 향해
# 아래를 보는 자세 (EE x축 = 툴 축 = -z). 실제 pick 의 T_grasp_base 회전과
# 같아야 IK 가 같은 팔 구성(브랜치)으로 풀린다.
R_TOOL_DOWN = np.array([
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0],
])

JOINT_ACCEL_RATIO = 0.5   # 저속 이동: M2 = M1 * 이 값 (queue_test 와 동일)

# ---- acceleration-limited throw 파라미터 ----
G_ACCEL = 9.81
FLOOR_CLEARANCE = 0.005    # 스윙 전체에서 컵 z >= GRASP_Z + 이 값 [m]
JOINT_LIMIT_MARGIN = np.radians(2.0)  # 전 관절 하드 리미트 안쪽 안전 여유
WRIST_BRANCH_JUMP_MAX = np.radians(90.0)  # J4/J6 180° 반대 IK 해 금지
THROW_TOOL_PITCH = np.radians(-30.0)  # 투척 전방 기준 툴 축 pitch (아래가 음수)
THROW_DT = 0.01            # throw 구간 knot 간격 [s] (스트림이 4ms 로 리샘플)
RELEASE_DISTANCE_DEFAULT = 0.10  # release XY = pick XY + distance * unit(pick→bin)
RELEASE_Z_OFFSET_DEFAULT = 0.33  # release Z = pick Z + offset [m]
RELEASE_DISTANCE_MIN = 0.10
RELEASE_DISTANCE_MAX = 0.25
RELEASE_DISTANCE_STEP = 0.05
RELEASE_Z_OFFSET_MIN = 0.20
RELEASE_Z_OFFSET_MAX = 0.60
RELEASE_Z_OFFSET_STEP = 0.02
# MoveIt joint_limits.yaml의 max_acceleration. 10% 안전 여유를 둔다.
THROW_ACCEL_LIMITS = np.array([10.0, 10.0, 10.0, 15.0, 15.0, 20.0])
THROW_ACCEL_SCALE = 0.90
THROW_VELOCITY_SCALE = 0.90
THROW_JERK_RAMP_FRACTION = 0.05  # 가속/감속 구간 양끝 5%를 선형 jerk ramp로 사용
RELEASE_OMEGA_MAX = 8.0          # release 평면내 각속도 탐색 상한 [rad/s]
RELEASE_OMEGA_STEP = 0.01        # 각속도 탐색 간격 [rad/s]
PREMOVE_SPEED = 0.20       # 와인드업 위치로 하강/복귀 카르테시안 이동 속도 [m/s]
PREMOVE_ACCEL = 1.0        # 그 가속도 [m/s^2]
PREMOVE_DT = 0.02          # 그 knot 간격 [s]
PREVIEW_SPEED_DEFAULT = 0.25     # RViz 재생 배속 (실제 시간의 1/4)
PREVIEW_STATIC_RATE = 1.0        # marker/path 재발행 주기 [Hz]
BIN_Z_OFFSET_DEFAULT = 0.10      # bin 목표 높이 = grasp z + 이 값 [m]
TOOL_OFFSET_DEFAULT = 0.0        # gp8.py EE는 MuJoCo grip_site/TCP와 일치하므로 추가 offset 없음.
TOOL_FRAME = "suction_tool"

# GP8 기하 (robots/gp8.py 스크류 정의): 어깨(J2) 위치와 링크 도달 한계.
SHOULDER_XZ = (0.04, 0.330)      # J1 축 기준 어깨 오프셋 (x, z) [m]
WRIST_REACH = 0.6875             # 어깨→손목중심 최대 (상완 0.345 + 전완 0.3425) [m]
TOOL_LEN = 0.325                 # 손목중심→기구학 EE 원점 [m]


def _flange_origin_from_tool(tool_pos, R_flange, tool_offset: float) -> np.ndarray:
    """suction_tool/TCP 목표점 → gp8.py EE 목표점.

    gp8.py 의 FK/IK end-effector 는 MuJoCo `grip_site` 와 일치한다
    (link6 +X 0.325m, ROS URDF flange 기준 +X 0.245m). 따라서 기본값
    tool_offset=0.0 에서는 목표점을 그대로 IK 에 넣는다.

    tool_offset 은 gp8.py EE/TCP 보다 더 앞쪽의 임시 점을 테스트할 때만 쓰는
    추가 offset 이다.

      external_tool = gp8_ee + R_ee[:, 0] * tool_offset
      gp8_ee_target = external_tool_target - R_ee[:, 0] * tool_offset
    """
    return np.asarray(tool_pos, dtype=float) - np.asarray(R_flange, dtype=float)[:, 0] * float(tool_offset)


# =====================================================================
# 기하: 포물선 → release 상태 (ROS 불필요, --plan-only 로 단독 검증 가능)
# =====================================================================
def plan_parabola_throw(
    grasp_xyz, bin_xy, bin_z_offset: float = BIN_Z_OFFSET_DEFAULT,
    release_distance: float = RELEASE_DISTANCE_DEFAULT,
    release_z_offset: float = RELEASE_Z_OFFSET_DEFAULT,
) -> dict:
    """release 지점과 release→bin 최소속도 탄도를 계산한다.

    release XY는 ``pick XY + release_distance * unit(pick→bin)``이고 Z는
    ``pick Z + release_z_offset``이다. release→bin의 수평거리 ``R``과
    높이차 ``dz``에 대해 진공 탄도의 필요 초기속도가 최소가 되는 해를 쓴다.

      v_min² = g (sqrt(R² + dz²) + dz)
      tan(theta) = (sqrt(R² + dz²) + dz) / R

    ``dz``는 bin Z - release Z이다. release pose의 툴 +X축은 투척
    수평방향에서 아래로 30° 기울인다.
    """
    G = np.asarray(grasp_xyz, dtype=float)
    bin_xy = np.asarray(bin_xy, dtype=float)
    z_g = float(G[2])
    z_bin = z_g + float(bin_z_offset)
    pick_to_bin = bin_xy - G[:2]
    D = float(np.linalg.norm(pick_to_bin))
    if D < 0.05:
        raise ValueError(f"bin이 pick과 너무 가깝습니다 ({D:.3f} m)")
    release_distance = float(release_distance)
    if release_distance <= 0.0 or release_distance >= D - 0.05:
        raise ValueError(
            f"release 거리가 pick→bin 거리에 맞지 않습니다 "
            f"(release={release_distance:.3f}, pick→bin={D:.3f} m)"
        )
    pick_to_bin_unit = pick_to_bin / D
    release_pos = np.array([
        G[0] + release_distance * pick_to_bin_unit[0],
        G[1] + release_distance * pick_to_bin_unit[1],
        z_g + float(release_z_offset),
    ])
    d_fly_xy = bin_xy - release_pos[:2]
    R_fly = float(np.linalg.norm(d_fly_xy))
    if R_fly < 0.05:
        raise ValueError(
            f"bin이 release와 너무 가깝습니다 (수평 {R_fly:.3f} m)"
        )
    u_xy = d_fly_xy / R_fly
    dz = z_bin - release_pos[2]
    slant = float(np.hypot(R_fly, dz))
    ballistic_term = slant + dz
    if ballistic_term <= 1e-9:
        raise ValueError("release/bin 높이차로 안정적인 탄도를 만들 수 없습니다")
    speed = float(np.sqrt(G_ACCEL * ballistic_term))
    tan_theta = ballistic_term / R_fly
    vx = speed / float(np.sqrt(1.0 + tan_theta ** 2))
    vz_rel = vx * tan_theta
    v_rel = np.array([vx * u_xy[0], vx * u_xy[1], vz_rel])
    speed = float(np.linalg.norm(v_rel))
    u3 = v_rel / speed

    # 어깨(J2) 3D 위치: J1 이 release 방위각으로 돌았을 때.
    az = np.arctan2(release_pos[1], release_pos[0])
    shoulder = np.array([SHOULDER_XZ[0] * np.cos(az),
                         SHOULDER_XZ[0] * np.sin(az), SHOULDER_XZ[1]])
    r_vec = release_pos - shoulder
    r_dist = float(np.linalg.norm(r_vec))
    reach_limit = WRIST_REACH + TOOL_LEN
    if r_dist > 0.98 * reach_limit:
        raise ValueError(
            f"release 지점이 최대 도달 반경 밖입니다 "
            f"(어깨에서 {r_dist:.3f} m > {0.98 * reach_limit:.3f} m) — "
            f"bin 을 가깝게 하거나 그랩점을 조정하세요"
        )
    # release 기준 툴 축: 전방/아래 30°.
    tool_axis = np.array([
        np.cos(THROW_TOOL_PITCH) * u_xy[0],
        np.cos(THROW_TOOL_PITCH) * u_xy[1],
        np.sin(THROW_TOOL_PITCH),
    ])
    # EE 프레임: x_ee = 툴 축, y_ee = 던짐 수직평면의 법선, z_ee = x×y.
    n_plane = np.array([-u_xy[1], u_xy[0], 0.0])
    x_ee = tool_axis
    y_ee = n_plane - x_ee * float(x_ee @ n_plane)   # 평면법선을 x 에 직교화
    y_ee /= np.linalg.norm(y_ee)
    throw_R = np.column_stack([x_ee, y_ee, np.cross(x_ee, y_ee)])

    # 탄도 낙하점 검산: release 상태에서 bin 높이까지 비행 → bin 이어야 함.
    t_fly = R_fly / vx
    landing_xy = release_pos[:2] + v_rel[:2] * t_fly
    landing_z = release_pos[2] + v_rel[2] * t_fly - 0.5 * G_ACCEL * t_fly ** 2
    landing_err = float(np.linalg.norm(np.array([
        landing_xy[0] - bin_xy[0], landing_xy[1] - bin_xy[1], landing_z - z_bin,
    ])))

    return dict(
        grasp=G, bin_xy=np.asarray(bin_xy, dtype=float),
        bin_xyz=np.array([bin_xy[0], bin_xy[1], z_bin], dtype=float),
        z_g=z_g, z_bin=z_bin, bin_z_offset=float(bin_z_offset),
        D=D, R=R_fly, release_distance=release_distance,
        release_z_offset=float(release_z_offset),
        release_pos=release_pos, v_rel=v_rel, speed=speed, u3=u3, u_xy=u_xy,
        vx=vx, launch_angle=float(np.arctan2(vz_rel, vx)),
        apex=float(vz_rel ** 2 / (2.0 * G_ACCEL)),
        throw_R=throw_R, r_extension=r_dist / reach_limit,
        t_fly=t_fly, landing=np.array([landing_xy[0], landing_xy[1], landing_z]),
        landing_err=landing_err,
    )


def _jerk_ramp_unit_profile(s, ramp_fraction: float):
    """0→1 속도 profile ``h``와 적분 ``H`` 및 미분 ``dh/ds``.

    가속도는 처음/끝 ramp 구간에서 선형으로 증감하고 중간에서
    일정하다. 따라서 h(0)=0, h(1)=1, h'(0)=h'(1)=0,
    H(1)=0.5이다.
    """
    s = np.asarray(s, dtype=float)
    e = float(ramp_fraction)
    if not (0.0 < e < 0.5):
        raise ValueError(f"jerk ramp fraction은 0~0.5 사이여야 합니다 ({e})")
    A = 1.0 / (1.0 - e)
    h = np.empty_like(s)
    H = np.empty_like(s)
    dh = np.empty_like(s)
    first = s < e
    middle = (s >= e) & (s <= 1.0 - e)
    last = s > 1.0 - e
    h[first] = A * s[first] ** 2 / (2.0 * e)
    H[first] = A * s[first] ** 3 / (6.0 * e)
    dh[first] = A * s[first] / e
    h[middle] = A * (s[middle] - 0.5 * e)
    H[middle] = A * (
        0.5 * s[middle] ** 2 - 0.5 * e * s[middle] + e ** 2 / 6.0
    )
    dh[middle] = A
    u = 1.0 - s[last]
    h[last] = 1.0 - A * u ** 2 / (2.0 * e)
    H[last] = s[last] - 0.5 + A * u ** 3 / (6.0 * e)
    dh[last] = A * u / e
    return h, H, dh


def _joint_swing_profile(q_release, qd_release, half_duration: float, dt: float):
    """정지→release→정지 시대칭 관절 스윙을 만든다."""
    q_release = np.asarray(q_release, dtype=float)
    qd_release = np.asarray(qd_release, dtype=float)
    T = float(half_duration)
    n_uniform = max(5, int(np.ceil(T / dt)) + 1)
    e = THROW_JERK_RAMP_FRACTION
    # 운동학 piecewise 경계를 knot에 정확히 포함해 Hermite 리샘플이
    # jerk ramp 끝을 건너뛰지 않게 한다.
    tau = np.unique(np.concatenate([
        np.linspace(0.0, T, n_uniform), np.array([e * T, (1.0 - e) * T]),
    ]))
    s = tau / T
    h, H, _ = _jerk_ramp_unit_profile(s, THROW_JERK_RAMP_FRACTION)
    q_start = q_release - 0.5 * qd_release * T
    q_end = q_release + 0.5 * qd_release * T
    q_pre = q_start[:, None] + qd_release[:, None] * T * H[None, :]
    qd_pre = qd_release[:, None] * h[None, :]
    q_post = (
        q_release[:, None]
        + qd_release[:, None] * T * (s - H)[None, :]
    )
    qd_post = qd_release[:, None] * (1.0 - h)[None, :]
    traj = np.concatenate([q_pre, q_post[:, 1:]], axis=1)
    vel = np.concatenate([qd_pre, qd_post[:, 1:]], axis=1)
    ts = np.concatenate([tau, T + tau[1:]])
    return traj, vel, ts, len(tau) - 1, q_start, q_end


def _line_profile_knots(p0, p1, v_peak, accel, dt):
    """p0→p1 직선을 사다리꼴(v_peak 도달 시 등속) 프로파일로 knot 샘플.
    저속 카르테시안 이동(와인드업 위치로 하강/복귀)용."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    L = float(np.linalg.norm(p1 - p0))
    u = (p1 - p0) / L
    v_tri = float(np.sqrt(accel * L))
    v_pk = min(float(v_peak), v_tri)
    t_a = v_pk / accel
    L_cruise = max(0.0, L - v_pk ** 2 / accel)
    t_c = L_cruise / v_pk

    def _phase(t_end):
        n = max(2, int(round(t_end / dt)) + 1)
        return np.linspace(0.0, t_end, n)

    ta = _phase(t_a)
    sa = 0.5 * accel * ta ** 2
    va = accel * ta
    t_all, s_all, v_all = [ta], [sa], [va]
    t_off, s_off = t_a, sa[-1]
    if t_c > 1e-9:
        tc = _phase(t_c)[1:]
        t_all.append(t_off + tc); s_all.append(s_off + v_pk * tc)
        v_all.append(np.full(tc.shape, v_pk))
        t_off += t_c; s_off += L_cruise
    td = _phase(t_a)[1:]
    t_all.append(t_off + td)
    s_all.append(s_off + v_pk * td - 0.5 * accel * td ** 2)
    v_all.append(v_pk - accel * td)

    t = np.concatenate(t_all)
    s = np.concatenate(s_all)
    ds = np.concatenate(v_all)
    pos = p0[None, :] + s[:, None] * u[None, :]
    vel = ds[:, None] * u[None, :]
    return pos, vel, t


def build_joint_traj(gp8: GP8, pos, vel, t, q_seed, R_fixed=None, R_seq=None,
                     omega_seq=None, tool_offset: float = TOOL_OFFSET_DEFAULT):
    """카르테시안 knot 열 → 관절 궤적 (6,N)/(6,N)/(N,).

    R_fixed: 전 knot 동일 자세(ω=0) → 관절속도는 공간 자코비안 (ω,v) 로
    q̇ = J⁻¹·[0; ṗ] 정확 산출 (상위 3행 = ω, 수치 검증 완료).
    R_seq: knot 별 자세. omega_seq도 주어지면 각 knot의 world-frame 각속도로
    spatial twist를 정확히 풀고, 없으면 저속 slerp용 시간 FD를 사용한다.
    IK 는 이전 knot 해로 warm-start. 실패 시 ValueError.
    """
    n = pos.shape[0]
    traj = np.zeros((6, n))
    velj = np.zeros((6, n))
    q_prev = np.asarray(q_seed, dtype=float)
    for k in range(n):
        T = np.eye(4)
        Rk = R_fixed if R_fixed is not None else R_seq[k]
        T[:3, :3] = Rk
        T[:3, 3] = _flange_origin_from_tool(pos[k], Rk, tool_offset)
        q = gp8.inverse_kinematics(T, q_init=q_prev)
        if q is None:
            raise ValueError(
                f"IK 실패 @ knot {k}/{n - 1} pos=({pos[k][0]:+.3f}, "
                f"{pos[k][1]:+.3f}, {pos[k][2]:+.3f}) — 도달 범위 밖"
            )
        q = np.asarray(q, dtype=float)
        traj[:, k] = q
        if R_fixed is not None or omega_seq is not None:
            J = gp8.jacobian(q)
            omega = (np.zeros(3) if omega_seq is None
                     else np.asarray(omega_seq[k], dtype=float))
            # Space Jacobian의 하위 3행은 원점 선속도가 아니라 spatial v이다.
            # world 점 p에 대해 p_dot=v+ω×p이므로 v=p_dot-ω×p.
            spatial_v = vel[k] - np.cross(omega, pos[k])
            twist = np.concatenate([omega, spatial_v])
            try:
                velj[:, k] = np.linalg.solve(J, twist)
            except np.linalg.LinAlgError:
                velj[:, k] = np.linalg.lstsq(J, twist, rcond=None)[0]
        q_prev = q
    if R_fixed is None and omega_seq is None:
        velj = np.gradient(traj, np.asarray(t, dtype=float), axis=1)
        velj[:, 0] = 0.0
        velj[:, -1] = 0.0
    return traj, velj, np.asarray(t, dtype=float)


def build_throw(gp8: GP8, plan: dict, q_seed, vel_limits,
                tool_offset: float = TOOL_OFFSET_DEFAULT):
    """release constraint를 만족하는 가속도 제한 관절 스윙을 만든다.

    release 평면 법선 축의 각속도를 탐색하여 release TCP 선속도를
    정확히 맞추면서 관절 속도/가속도/위치 제한을 모두 만족하는 가장
    짧은 시대칭 스윙을 선택한다. TCP 경로는 그 관절 스윙의 FK 결과이다.
    """
    q_seed = np.asarray(q_seed, dtype=float)
    vel_limits = np.asarray(vel_limits, dtype=float)
    joint_limits = np.asarray(gp8.joint_limits, dtype=float)
    soft_lower = joint_limits[:, 0] + JOINT_LIMIT_MARGIN
    soft_upper = joint_limits[:, 1] - JOINT_LIMIT_MARGIN
    effective_accel = THROW_ACCEL_LIMITS * THROW_ACCEL_SCALE
    release_pos = np.asarray(plan["release_pos"], dtype=float)

    T_release = np.eye(4)
    T_release[:3, :3] = plan["throw_R"]
    T_release[:3, 3] = _flange_origin_from_tool(
        release_pos, plan["throw_R"], tool_offset,
    )
    q_release = gp8.inverse_kinematics(T_release, q_init=q_seed)
    if q_release is None:
        raise ValueError(
            "release pose IK 실패 — release 오프셋을 조정하세요 "
            f"pos=({release_pos[0]:+.3f}, {release_pos[1]:+.3f}, {release_pos[2]:+.3f})"
        )
    q_release = np.asarray(q_release, dtype=float)
    J_release = gp8.jacobian(q_release)
    n_plane = np.array([-plan["u_xy"][1], plan["u_xy"][0], 0.0])
    kinematic_candidates = []
    reject_counts = {"velocity": 0, "position": 0, "wrist": 0, "floor": 0}

    for omega_mag in np.arange(
        0.0, RELEASE_OMEGA_MAX + 0.5 * RELEASE_OMEGA_STEP,
        RELEASE_OMEGA_STEP,
    ):
        omega_release = -float(omega_mag) * n_plane
        spatial_v = plan["v_rel"] - np.cross(omega_release, release_pos)
        try:
            qd_release = np.linalg.solve(
                J_release, np.concatenate([omega_release, spatial_v]),
            )
        except np.linalg.LinAlgError:
            qd_release = np.linalg.lstsq(
                J_release, np.concatenate([omega_release, spatial_v]), rcond=None,
            )[0]

        vel_ratio = float(np.max(np.abs(qd_release) / vel_limits))
        if vel_ratio > THROW_VELOCITY_SCALE:
            reject_counts["velocity"] += 1
            continue
        # h'(s)의 최댓값은 1/(1-ramp_fraction).
        half_duration = float(np.max(
            np.abs(qd_release)
            / (effective_accel * (1.0 - THROW_JERK_RAMP_FRACTION))
        ))
        if half_duration <= 1e-6:
            continue
        q_start = q_release - 0.5 * qd_release * half_duration
        q_end = q_release + 0.5 * qd_release * half_duration
        q_bounds = np.column_stack([q_start, q_release, q_end])
        if (
            np.any(q_bounds < soft_lower[:, None] - 1e-9)
            or np.any(q_bounds > soft_upper[:, None] + 1e-9)
        ):
            reject_counts["position"] += 1
            continue
        wrist_delta = (
            q_start[[3, 5]] - q_seed[[3, 5]] + np.pi
        ) % (2.0 * np.pi) - np.pi
        wrist_jump = float(np.max(np.abs(wrist_delta)))
        if wrist_jump > WRIST_BRANCH_JUMP_MAX:
            reject_counts["wrist"] += 1
            continue

        accel_peak = np.abs(qd_release) / (
            half_duration * (1.0 - THROW_JERK_RAMP_FRACTION)
        )
        score = (half_duration, -float(np.min(np.minimum(
            q_bounds - joint_limits[:, 0, None],
            joint_limits[:, 1, None] - q_bounds,
        ))))
        kinematic_candidates.append(dict(
            score=score,
            q_start=q_start, q_release=q_release.copy(), q_end=q_end,
            qd_release=qd_release, omega_release=omega_release,
            release_omega=float(omega_mag), vel_ratio=vel_ratio,
            accel_peak=accel_peak,
            accel_ratio=float(np.max(accel_peak / THROW_ACCEL_LIMITS)),
            wrist_branch_jump=wrist_jump,
        ))

    if not kinematic_candidates:
        raise ValueError(
            "release constraint를 만족하는 관절 스윙을 만들 수 없습니다 "
            f"(제외: {reject_counts}) — release Z를 높이거나 bin을 가깝게 하세요"
        )

    best = None
    # 관절 조건으로 정렬한 뒤 상위 후보부터 FK를 계산한다. 기존처럼
    # 모든 omega 후보에 전체 FK를 반복하지 않아 adaptive 탐색 시간을 줄인다.
    for candidate in sorted(kinematic_candidates, key=lambda c: c["score"]):
        traj, velj, ts, i_rel, q_start, q_end = _joint_swing_profile(
            q_release, candidate["qd_release"], candidate["score"][0], THROW_DT,
        )
        swing_pos = np.zeros((traj.shape[1], 3), dtype=float)
        swing_R = np.zeros((traj.shape[1], 3, 3), dtype=float)
        for k in range(traj.shape[1]):
            Tk = gp8.forward_kinematics(traj[:, k])
            swing_R[k] = Tk[:3, :3]
            swing_pos[k] = Tk[:3, 3] + Tk[:3, 0] * float(tool_offset)
        z_min = float(np.min(swing_pos[:, 2]))
        if z_min < plan["z_g"] + FLOOR_CLEARANCE:
            reject_counts["floor"] += 1
            continue
        candidate.update(
            traj=traj, vel=velj, ts=ts, release_idx=i_rel,
            q_start=q_start, q_end=q_end, swing_pos=swing_pos,
            swing_R=swing_R, z_min=z_min,
        )
        best = candidate
        break

    if best is None:
        raise ValueError(
            "관절 제한을 만족하는 후보가 모두 바닥 클리어런스를 위반합니다 "
            f"(제외: {reject_counts})"
        )

    Jr = gp8.jacobian(best["q_release"])
    twist_chk = Jr @ best["qd_release"]
    v_chk = twist_chk[3:] + np.cross(twist_chk[:3], release_pos)
    release_fk = best["swing_pos"][best["release_idx"]]
    orientation_swing = Rotation.from_matrix(
        plan["throw_R"] @ best["swing_R"][0].T
    ).magnitude()
    j5_upper = float(joint_limits[4, 1])
    j5_max = float(np.max(best["traj"][4]))
    plan["start_pos"] = best["swing_pos"][0].copy()
    plan["end_pos"] = best["swing_pos"][-1].copy()
    best.update(
        v_err=float(np.linalg.norm(v_chk - plan["v_rel"])),
        omega_err=float(np.linalg.norm(twist_chk[:3] - best["omega_release"])),
        release_pos_error=float(np.linalg.norm(release_fk - release_pos)),
        orientation_swing=float(orientation_swing),
        j5_max=j5_max, j5_upper=j5_upper, j5_margin=j5_upper - j5_max,
        duration=float(best["ts"][-1]), n_knots=best["traj"].shape[1],
        swing_vel=None,
    )
    best.pop("score", None)
    return best


def _adaptive_release_candidates(preferred_distance: float, preferred_z: float):
    """기준 release에 가까운 순서로 (수평거리, Z offset) 후보를 만든다."""
    distances = list(np.arange(
        RELEASE_DISTANCE_MIN,
        RELEASE_DISTANCE_MAX + 0.5 * RELEASE_DISTANCE_STEP,
        RELEASE_DISTANCE_STEP,
    ))
    z_offsets = list(np.arange(
        RELEASE_Z_OFFSET_MIN,
        RELEASE_Z_OFFSET_MAX + 0.5 * RELEASE_Z_OFFSET_STEP,
        RELEASE_Z_OFFSET_STEP,
    ))
    distances.append(float(preferred_distance))
    z_offsets.append(float(preferred_z))
    distances = sorted({round(float(v), 6) for v in distances})
    z_offsets = sorted({round(float(v), 6) for v in z_offsets})
    candidates = [(d, z) for d in distances for z in z_offsets]
    candidates.sort(key=lambda c: (
        abs(c[0] - preferred_distance) / RELEASE_DISTANCE_STEP
        + abs(c[1] - preferred_z) / RELEASE_Z_OFFSET_STEP,
        abs(c[0] - preferred_distance),
        abs(c[1] - preferred_z),
        c[1],
    ))
    return candidates


def plan_and_build_adaptive_throw(
    gp8: GP8, grasp_xyz, bin_xy, bin_z_offset: float, q_seed, vel_limits,
    preferred_distance: float = RELEASE_DISTANCE_DEFAULT,
    preferred_z_offset: float = RELEASE_Z_OFFSET_DEFAULT,
    tool_offset: float = TOOL_OFFSET_DEFAULT,
):
    """release 거리/Z를 adaptive 탐색해 ``(plan, built)``를 반환한다.

    기준값을 먼저 시도하고, 실패할 때만 pick→bin 방향 10~25 cm,
    pick Z +20~60 cm 격자를 기준값에 가까운 순서로 탐색한다.
    """
    errors = []
    candidates = _adaptive_release_candidates(
        float(preferred_distance), float(preferred_z_offset),
    )
    for index, (distance, z_offset) in enumerate(candidates, start=1):
        try:
            plan = plan_parabola_throw(
                grasp_xyz, bin_xy, bin_z_offset=bin_z_offset,
                release_distance=distance, release_z_offset=z_offset,
            )
            built = build_throw(
                gp8, plan, q_seed=q_seed, vel_limits=vel_limits,
                tool_offset=tool_offset,
            )
        except ValueError as exc:
            errors.append(str(exc))
            continue
        plan.update(
            adaptive_release=(
                abs(distance - preferred_distance) > 1e-9
                or abs(z_offset - preferred_z_offset) > 1e-9
            ),
            preferred_release_distance=float(preferred_distance),
            preferred_release_z_offset=float(preferred_z_offset),
            release_candidates_tested=index,
            release_candidates_total=len(candidates),
        )
        return plan, built

    last_error = errors[-1] if errors else "후보 없음"
    raise ValueError(
        f"adaptive release 계획 실패: {len(candidates)}개 후보가 모두 불가 "
        f"(거리 {RELEASE_DISTANCE_MIN:.2f}~{RELEASE_DISTANCE_MAX:.2f} m, "
        f"Z {RELEASE_Z_OFFSET_MIN:.2f}~{RELEASE_Z_OFFSET_MAX:.2f} m); "
        f"마지막 원인: {last_error}"
    )


def print_throw_plan(plan: dict, built: dict, release_lead: float) -> None:
    p, b = plan, built
    rp, v = p["release_pos"], p["v_rel"]
    ang = float(np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1]))))
    print("\n=== Acceleration-limited Throw Plan ===")
    print(f"  grasp→bin : D={p['D']:.3f} m, "
          f"bin=({p['bin_xyz'][0]:+.3f}, {p['bin_xyz'][1]:+.3f}, {p['bin_xyz'][2]:+.3f}) "
          f"(grasp z +{p['bin_z_offset'] * 100:.1f} cm)")
    mode = "adaptive" if p.get("adaptive_release", False) else "preferred"
    print(f"  release   : pick→bin {p['release_distance'] * 100:.1f} cm, "
          f"pick Z +{p['release_z_offset'] * 100:.1f} cm, "
          f"pos=({rp[0]:+.3f}, {rp[1]:+.3f}, {rp[2]:+.3f}) m "
          f"(그랩 높이 +{(rp[2] - p['z_g']) * 100:.1f} cm, 신전율 {p['r_extension'] * 100:.0f}%)")
    print(f"              selection={mode}, "
          f"candidate {p.get('release_candidates_tested', 1)}/"
          f"{p.get('release_candidates_total', 1)}")
    print(f"              v=({v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f}) m/s, "
          f"|v|={p['speed']:.3f} m/s, {ang:.2f}°  "
          f"(툴 pitch {np.degrees(THROW_TOOL_PITCH):+.0f}°)")
    print(f"  포물선     : release→bin 수평 {p['R']:.3f} m, "
          f"최소속도 해, apex=release +{p['apex'] * 100:.1f} cm")
    print(f"  낙하 검산  : bin 오차 {p['landing_err'] * 1000:.1f} mm (release 후 비행 {p['t_fly']:.3f}s)")
    print(f"  관절 스윙   : 와인드업 {b['duration'] / 2:.3f}s + "
          f"팔로스루 {b['duration'] / 2:.3f}s, "
          f"총 {b['duration']:.3f}s / {b['n_knots']} knots, z_min={b['z_min']:.3f} m")
    print(f"  제한 검사  : max J5={np.degrees(b['j5_max']):.2f}° "
          f"/ limit {np.degrees(b['j5_upper']):.2f}° "
          f"(여유 {np.degrees(b['j5_margin']):.2f}°), "
          f"속도 {b['vel_ratio'] * 100:.1f}% / 가속도 {b['accel_ratio'] * 100:.1f}%")
    print(f"  orientation : 자유 관절 스윙, start→release "
          f"{np.degrees(b['orientation_swing']):.2f}°, "
          f"release |ω|={np.degrees(b['release_omega']):.1f}°/s")
    print(f"  release knot {b['release_idx']} (lead {release_lead:+.3f}s), "
          f"관절속도 max|q̇|/limit = {b['vel_ratio'] * 100:.1f}%, "
          f"release 위치/속도오차 {b['release_pos_error'] * 1000:.3f} mm / "
          f"{b['v_err'] * 1000:.2f} mm/s, "
          f"각속도오차 {np.degrees(b['omega_err']):.3f}°/s")
    if b["vel_ratio"] > 0.8:
        print("  ⚠️ 관절속도가 limit의 80%를 넘습니다")


# =====================================================================
# 저속 이동 헬퍼
# =====================================================================
def _ik_tool_down(gp8: GP8, x: float, y: float, z: float, q_init=None,
                  tool_offset: float = TOOL_OFFSET_DEFAULT):
    """(x, y, z) TCP 위 툴 아래보기 pose 의 joint 해. IK 실패 시 None."""
    T = np.eye(4)
    T[:3, :3] = R_TOOL_DOWN
    T[:3, 3] = _flange_origin_from_tool((x, y, z), R_TOOL_DOWN, tool_offset)
    q = gp8.inverse_kinematics(T, q_init=q_init)
    if q is None:
        return None
    q = np.asarray(q, dtype=float)
    q[-1] = 0.0
    return q


def _move(ctrl, q_from, q_to, M1, M2, hz: float) -> float:
    """q_from → q_to 관절공간 저속 이동 (블록). 소요 시간(계획치) 리턴."""
    zero = np.zeros_like(M1)
    traj, vel, ts = trajectory(q_from, zero, q_to, zero, M1, M2, hertz=hz)
    ctrl.send_trajectory_queue(traj, vel, ts, final_joint=q_to)
    return float(ts[-1])


def _move_cart_slerp(ctrl, gp8: GP8, p0, p1, R0, R1, q_seed,
                     tool_offset: float = TOOL_OFFSET_DEFAULT) -> np.ndarray:
    """p0→p1 카르테시안 직선 + R0→R1 slerp 저속 이동 (블록). 도착 관절값 리턴.
    lift(tool-down) ↔ 와인드업 시작점(기울인 자세) 전환에 사용."""
    pos, vel, t = _line_profile_knots(p0, p1, PREMOVE_SPEED, PREMOVE_ACCEL, PREMOVE_DT)
    key = Rotation.from_matrix(np.stack([R0, R1]))
    slerp = Slerp([0.0, 1.0], key)
    frac = np.linspace(0.0, 1.0, pos.shape[0])
    R_seq = slerp(frac).as_matrix()
    traj, velj, ts = build_joint_traj(
        gp8, pos, vel, t, q_seed, R_seq=R_seq, tool_offset=tool_offset,
    )
    ctrl.send_trajectory_queue(traj, velj, ts, final_joint=traj[:, -1])
    return traj[:, -1].copy()


# =====================================================================
# 사이클
# =====================================================================
def run_throw(ctrl, gp8: GP8, args, grasp_xyz, lift_q, vel_limits) -> None:
    plan, built = plan_and_build_adaptive_throw(
        gp8, grasp_xyz, (args.bin_x, args.bin_y), args.bin_z_offset,
        q_seed=lift_q, vel_limits=vel_limits,
        preferred_distance=args.release_distance,
        preferred_z_offset=args.release_z_offset,
        tool_offset=args.tool_offset,
    )
    print_throw_plan(plan, built, args.release_lead)

    ans = input("\nbin 위치에 통/마커를 뒀습니까? throw 실행? (y/N) > ").strip().lower()
    if ans != "y":
        print("  throw 취소 (석션 유지 중).")
        return

    # release_lead: release knot 시각보다 lead 만큼 이른 knot 로 인덱스 이동.
    ts, i_rel = built["ts"], built["release_idx"]
    t_target = ts[i_rel] - args.release_lead
    rel_idx = int(np.searchsorted(ts, t_target, side="right") - 1)
    rel_idx = max(0, min(rel_idx, len(ts) - 1))

    print("→ 와인드업 시작 관절자세로 이동 (throw TCP offset 적용, 석션 유지)...")
    vel_limits = np.asarray(vel_limits, dtype=float)
    _move(
        ctrl, lift_q, built["q_start"],
        vel_limits * args.vel_scale, vel_limits * args.vel_scale * JOINT_ACCEL_RATIO,
        Config().TRAJ_HZ,
    )
    time.sleep(0.2)

    print(f"→ throw 스윙 ({built['duration']:.2f}s, |v|={plan['speed']:.2f} m/s, "
          f"release @ knot {rel_idx})...")
    ctrl.send_trajectory_queue_with_timed_release(
        built["traj"], built["vel"], built["ts"],
        final_joint=built["q_end"], release_index=rel_idx,
    )
    lt = getattr(ctrl, "last_throw", None) or {}
    io_ms = lt.get("io_ms")
    print(f"  스윙 완료. release IO {io_ms:.1f} ms" if io_ms is not None
          else "  스윙 완료.")

    input("\nEnter → lift 자세로 복귀 ")
    _move(
        ctrl, built["q_end"], lift_q,
        vel_limits * args.vel_scale, vel_limits * args.vel_scale * JOINT_ACCEL_RATIO,
        Config().TRAJ_HZ,
    )
    print("  복귀 완료.")


def run_cycle(ctrl, gp8: GP8, args, z: float, M1, M2, hz: float, vel_limits) -> None:
    x, y = args.x, args.y
    current_q = np.asarray(ctrl.current_joints, dtype=float)

    grasp_q = _ik_tool_down(gp8, x, y, z, tool_offset=args.tool_offset)
    if grasp_q is None:
        print(f"  IK 실패: grasp ({x:+.3f}, {y:+.3f}, {z:+.3f}) — 지점을 바꿔보세요.")
        return
    lift_q = _ik_tool_down(
        gp8, x, y, z + args.lift, q_init=grasp_q, tool_offset=args.tool_offset,
    )
    if lift_q is None:
        print(f"  IK 실패: lift ({x:+.3f}, {y:+.3f}, {z + args.lift:+.3f})")
        return

    print("\n=== Suction Lift Plan ===")
    print(f"  grasp   : ({x:+.3f}, {y:+.3f}, {z:+.3f}) m  ← 석션 대기 높이(GRASP_Z 기준)")
    print(f"  lift    : ({x:+.3f}, {y:+.3f}, {z + args.lift:+.3f}) m  (+{args.lift * 100:.0f} cm)")
    print(f"  hold    : 석션 ON 후 {args.hold:.1f}s 파킹 홀드")

    ans = input("\n물체를 지점에 놓았습니까? 실행? (y/N) > ").strip().lower()
    if ans != "y":
        print("  취소.")
        return

    # 실제 cycle 과 동일: 항상 석션 OFF 로 깨끗하게 시작 (throw_skill.execute).
    ctrl.suction_off()

    print("→ 석션 대기 높이로 이동...")
    t_move = _move(ctrl, current_q, grasp_q, M1, M2, hz)
    print(f"  파킹 완료 ({t_move:.2f}s).")

    # 컵이 파킹된 뒤에만 석션 — position_and_prime 의 position-prime 순서.
    print(f"→ 석션 ON, {args.hold:.1f}s 홀드 (진공 형성)...")
    ctrl.suction_on()
    time.sleep(args.hold)          # JGPC 가 마지막 명령 위치를 유지하므로 파킹 홀드

    print(f"→ +{args.lift * 100:.0f} cm 리프트 (석션 유지)...")
    t_lift = _move(ctrl, grasp_q, lift_q, M1, M2, hz)
    print(f"  리프트 완료 ({t_lift:.2f}s). 물체가 딸려 올라왔는지 확인하세요.")

    ans = input("\nt = 포물선 throw / Enter = 석션 OFF(제자리 릴리즈) > ").strip().lower()
    if ans == "t":
        try:
            run_throw(ctrl, gp8, args, np.array([x, y, z]), lift_q, vel_limits)
        except ValueError as e:
            print(f"  throw 계획 실패: {e}")
            input("\nEnter → 석션 OFF (릴리즈) ")
            ctrl.suction_off()
    else:
        ctrl.suction_off()
        print("  릴리즈 완료.")


def plan_only(gp8: GP8, args, z: float, vel_limits) -> None:
    """로봇/ROS 없이 throw 기하 + IK + 관절궤적을 검증 출력."""
    grasp_xyz = np.array([args.x, args.y, z])
    q_seed = _ik_tool_down(gp8, args.x, args.y, z + args.lift, tool_offset=args.tool_offset)
    if q_seed is None:
        print("lift 자세 IK 실패 — 그랩 지점을 확인하세요.")
        return
    try:
        plan, built = plan_and_build_adaptive_throw(
            gp8, grasp_xyz, (args.bin_x, args.bin_y), args.bin_z_offset,
            q_seed=q_seed, vel_limits=vel_limits,
            preferred_distance=args.release_distance,
            preferred_z_offset=args.release_z_offset,
            tool_offset=args.tool_offset,
        )
    except ValueError as e:
        print(f"throw 계획 실패: {e}")
        return
    print_throw_plan(plan, built, args.release_lead)
    for name, q in (("q_start", built["q_start"]), ("q_end", built["q_end"])):
        print(f"  {name} (deg): {[round(float(np.degrees(j)), 1) for j in q]}")
    # lift↔throw start/end 는 관절공간 저속 연결.
    M1 = np.asarray(vel_limits, dtype=float) * args.vel_scale
    M2 = M1 * JOINT_ACCEL_RATIO
    zero = np.zeros_like(M1)
    _, _, ts_in = trajectory(q_seed, zero, built["q_start"], zero, M1, M2, hertz=100.0)
    _, _, ts_out = trajectory(built["q_end"], zero, q_seed, zero, M1, M2, hertz=100.0)
    print(f"  lift→throw_start 관절 연결 OK ({ts_in[-1]:.2f}s)")
    print(f"  throw_end→lift 관절 연결 OK ({ts_out[-1]:.2f}s)")
    print("\nplan-only OK — 위 값이 타당하면 실제 실행으로 진행하세요.")


def _concat_joint_segments(segments):
    """[(traj(6,N), ts(N)), ...] 를 하나의 시간순 preview trajectory 로 결합."""
    q_parts = []
    t_parts = []
    offset = 0.0
    for i, (traj, ts) in enumerate(segments):
        traj = np.asarray(traj, dtype=float)
        ts = np.asarray(ts, dtype=float)
        start = 1 if i > 0 else 0
        q_parts.append(traj[:, start:])
        t_parts.append(offset + ts[start:])
        offset += float(ts[-1])
    return np.concatenate(q_parts, axis=1), np.concatenate(t_parts)


def _interpolate_joint_state(traj, ts, t: float) -> np.ndarray:
    """불규칙한 trajectory knot 사이를 시간 기준 선형 보간한다.

    이전 구현처럼 바로 앞 knot를 그대로 내보내면 30 Hz RViz가 100 Hz 계획점을
    여러 개씩 건너뛰어 RobotModel이 계단식으로 움직인다.
    """
    traj = np.asarray(traj, dtype=float)
    ts = np.asarray(ts, dtype=float)
    if t <= ts[0]:
        return traj[:, 0].copy()
    if t >= ts[-1]:
        return traj[:, -1].copy()
    i = int(np.searchsorted(ts, t, side="right") - 1)
    dt = float(ts[i + 1] - ts[i])
    if dt <= 0.0:
        return traj[:, i].copy()
    alpha = (float(t) - float(ts[i])) / dt
    return (1.0 - alpha) * traj[:, i] + alpha * traj[:, i + 1]


def _tool_path_from_joint_traj(gp8: GP8, traj, tool_offset: float) -> np.ndarray:
    """관절 궤적을 실제 suction_tool 원점 path(base_link 기준)로 FK 변환."""
    traj = np.asarray(traj, dtype=float)
    pts = np.zeros((traj.shape[1], 3), dtype=float)
    for i in range(traj.shape[1]):
        T = gp8.forward_kinematics(traj[:, i])
        pts[i] = T[:3, 3] + T[:3, 0] * float(tool_offset)
    return pts


def _build_rviz_preview(gp8: GP8, args, z: float, vel_limits):
    """RViz preview 용 전체 joint trajectory + marker geometry 를 생성."""
    grasp_xyz = np.array([args.x, args.y, z], dtype=float)
    grasp_q = _ik_tool_down(gp8, args.x, args.y, z, tool_offset=args.tool_offset)
    if grasp_q is None:
        raise ValueError(f"grasp IK 실패: ({args.x:+.3f}, {args.y:+.3f}, {z:+.3f})")
    lift_q = _ik_tool_down(
        gp8, args.x, args.y, z + args.lift, q_init=grasp_q, tool_offset=args.tool_offset,
    )
    if lift_q is None:
        raise ValueError(f"lift IK 실패: ({args.x:+.3f}, {args.y:+.3f}, {z + args.lift:+.3f})")

    plan, built = plan_and_build_adaptive_throw(
        gp8, grasp_xyz, (args.bin_x, args.bin_y), args.bin_z_offset,
        q_seed=lift_q, vel_limits=vel_limits,
        preferred_distance=args.release_distance,
        preferred_z_offset=args.release_z_offset,
        tool_offset=args.tool_offset,
    )

    M1 = np.asarray(vel_limits, dtype=float) * args.vel_scale
    M2 = M1 * JOINT_ACCEL_RATIO
    zero = np.zeros_like(M1)
    pick_lift_traj, pick_lift_vel, pick_lift_ts = trajectory(
        grasp_q, zero, lift_q, zero, M1, M2, hertz=100.0,
    )

    lift_pos = np.array([args.x, args.y, z + args.lift], dtype=float)
    pre_traj, pre_velj, pre_ts = trajectory(
        lift_q, zero, built["q_start"], zero, M1, M2, hertz=100.0,
    )
    ret_traj, ret_velj, ret_ts = trajectory(
        built["q_end"], zero, lift_q, zero, M1, M2, hertz=100.0,
    )
    # preview가 반복될 때 마지막 lift 자세에서 첫 grasp 자세로 순간이동하지
    # 않도록 한 사이클을 grasp 자세에서 닫는다.
    close_traj, close_velj, close_ts = trajectory(
        lift_q, zero, grasp_q, zero, M1, M2, hertz=100.0,
    )

    preview_traj, preview_ts = _concat_joint_segments([
        (pick_lift_traj, pick_lift_ts),
        (pre_traj, pre_ts),
        (built["traj"], built["ts"]),
        (ret_traj, ret_ts),
        (close_traj, close_ts),
    ])
    preview_traj[:, -1] = preview_traj[:, 0]  # modulo 반복 경계를 수치적으로도 정확히 닫음
    tool_path = _tool_path_from_joint_traj(gp8, preview_traj, args.tool_offset)

    return dict(
        plan=plan, built=built,
        grasp_xyz=grasp_xyz, lift_pos=lift_pos,
        preview_traj=preview_traj, preview_ts=preview_ts,
        tool_path=tool_path,
        ee_path=np.vstack([
            lift_pos[None, :], plan["start_pos"][None, :],
            plan["release_pos"][None, :], plan["end_pos"][None, :],
            lift_pos[None, :],
        ]),
    )


def _publish_rviz_static(pub_markers, pub_path, preview, frame_id: str, stamp):
    from geometry_msgs.msg import Point, PoseStamped
    from nav_msgs.msg import Path
    from visualization_msgs.msg import Marker, MarkerArray

    def point(xyz):
        p = Point()
        p.x, p.y, p.z = map(float, xyz)
        return p

    def color(marker, rgba):
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = rgba

    def marker_base(mid: int, ns: str, mtype: int):
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = stamp
        m.ns = ns
        m.id = mid
        m.type = mtype
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        return m

    plan = preview["plan"]
    grasp = plan["grasp"]
    lift = preview["lift_pos"]
    release = plan["release_pos"]
    bin_xyz = plan["bin_xyz"]

    markers = []
    for mid, ns, xyz, scale, rgba in [
        (0, "pick", grasp, 0.055, (0.1, 0.9, 0.1, 0.95)),
        (1, "lift", lift, 0.045, (0.1, 0.9, 0.9, 0.85)),
        (2, "release", release, 0.065, (1.0, 0.55, 0.0, 0.95)),
        (3, "bin", bin_xyz, 0.075, (0.1, 0.3, 1.0, 0.95)),
        (4, "runup_start", plan["start_pos"], 0.035, (1.0, 1.0, 0.0, 0.85)),
        (5, "follow_end", plan["end_pos"], 0.035, (1.0, 0.0, 1.0, 0.85)),
    ]:
        m = marker_base(mid, ns, Marker.SPHERE)
        m.pose.position = point(xyz)
        m.scale.x = m.scale.y = m.scale.z = scale
        color(m, rgba)
        markers.append(m)

    # 실제 RobotModel preview trajectory 를 FK 한 suction_tool 원점 path.
    stroke = marker_base(10, "ee_preview_path", Marker.LINE_STRIP)
    stroke.scale.x = 0.01
    color(stroke, (1.0, 0.85, 0.05, 0.95))
    stroke.points = [point(p) for p in preview["tool_path"]]
    markers.append(stroke)

    # 고속 throw 구간만 분리한 실제 TCP FK 곡선.
    swing_path = marker_base(13, "throw_swing_path", Marker.LINE_STRIP)
    swing_path.scale.x = 0.016
    color(swing_path, (1.0, 0.15, 0.75, 0.95))
    swing_path.points = [point(p) for p in preview["built"]["swing_pos"]]
    markers.append(swing_path)

    # release 순간 TCP 속도 방향. 스윙 곡선과 탄도 모두 이 화살표에 접한다.
    tangent = marker_base(14, "release_tangent", Marker.ARROW)
    tangent.points = [point(release), point(release + 0.12 * plan["u3"])]
    tangent.scale.x = 0.012
    tangent.scale.y = 0.025
    tangent.scale.z = 0.035
    color(tangent, (1.0, 0.35, 0.05, 0.95))
    markers.append(tangent)

    # ballistic arc after release
    arc = marker_base(11, "ballistic_arc", Marker.LINE_STRIP)
    arc.scale.x = 0.008
    color(arc, (0.0, 0.8, 1.0, 0.95))
    ts = np.linspace(0.0, plan["t_fly"], 50)
    for t in ts:
        xyz = release + plan["v_rel"] * t + np.array([0.0, 0.0, -0.5 * G_ACCEL * t * t])
        arc.points.append(point(xyz))
    markers.append(arc)

    # bin footprint circle as cylinder
    bin_cyl = marker_base(12, "bin_footprint", Marker.CYLINDER)
    bin_cyl.pose.position = point([bin_xyz[0], bin_xyz[1], bin_xyz[2] - 0.01])
    bin_cyl.scale.x = bin_cyl.scale.y = 0.18
    bin_cyl.scale.z = 0.02
    color(bin_cyl, (0.1, 0.3, 1.0, 0.25))
    markers.append(bin_cyl)

    pub_markers.publish(MarkerArray(markers=markers))

    path = Path()
    path.header.frame_id = frame_id
    path.header.stamp = stamp
    for xyz in preview["tool_path"]:
        ps = PoseStamped()
        ps.header = path.header
        ps.pose.position = point(xyz)
        ps.pose.orientation.w = 1.0
        path.poses.append(ps)
    pub_path.publish(path)


def rviz_preview(gp8: GP8, args, z: float, vel_limits) -> None:
    """RViz에서 marker/path와 RobotModel 애니메이션으로 throw 계획을 미리 본다."""
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    preview = _build_rviz_preview(gp8, args, z, vel_limits)
    print_throw_plan(preview["plan"], preview["built"], args.release_lead)

    rclpy.init()
    node = Node("suction_lift_debug_rviz_preview")
    pub_js = node.create_publisher(JointState, "/joint_states", 10)
    pub_js_urdf = node.create_publisher(JointState, "/joint_states_urdf", 10)
    pub_markers = node.create_publisher(
        __import__("visualization_msgs.msg", fromlist=["MarkerArray"]).MarkerArray,
        "/suction_lift_debug/markers", 10,
    )
    pub_path = node.create_publisher(
        __import__("nav_msgs.msg", fromlist=["Path"]).Path,
        "/suction_lift_debug/path", 10,
    )

    q = preview["preview_traj"]
    ts = preview["preview_ts"]
    duration = float(ts[-1])
    preview_speed = float(args.preview_speed)
    preview_rate = float(args.preview_rate)
    if preview_speed <= 0.0:
        raise ValueError(f"preview_speed는 0보다 커야 합니다 ({preview_speed})")
    if preview_rate <= 0.0:
        raise ValueError(f"preview_rate는 0보다 커야 합니다 ({preview_rate})")
    started = time.monotonic()
    frame_id = args.frame_id

    print("\nRViz preview publishing:")
    print("  joint_states : /joint_states and /joint_states_urdf")
    print("  markers      : /suction_lift_debug/markers")
    print("  path         : /suction_lift_debug/path")
    print(f"  tool frame   : URDF {TOOL_FRAME} (MuJoCo grip_site/TCP); extra offset {args.tool_offset:+.3f} m")
    print(f"  fixed frame  : {frame_id}")
    print(f"  playback     : {preview_speed:.2f}x slow-motion "
          f"(계획 {duration:.2f}s → 화면 {duration / preview_speed:.2f}s/cycle, "
          f"publish {preview_rate:.0f}Hz)")
    print("Ctrl-C 로 종료. 실제 로봇 명령은 보내지 않습니다.")

    def tick():
        now = node.get_clock().now().to_msg()
        t = ((time.monotonic() - started) * preview_speed) % duration
        q_now = _interpolate_joint_state(q, ts, t)

        msg = JointState()
        msg.header.stamp = now
        msg.name = JOINT_NAMES
        msg.position = [float(v) for v in q_now]
        pub_js.publish(msg)
        pub_js_urdf.publish(msg)

    def publish_static():
        # marker/path는 움직이지 않는다. 매 animation frame마다 수백 점을
        # 재전송하지 않고 저주기로만 갱신해 RViz callback/render 부하를 줄인다.
        now = node.get_clock().now().to_msg()
        _publish_rviz_static(pub_markers, pub_path, preview, frame_id, now)

    publish_static()
    node.create_timer(1.0 / preview_rate, tick)
    node.create_timer(1.0 / PREVIEW_STATIC_RATE, publish_static)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        # External shutdown can happen when the process is stopped by launch/timeout.
        shutdown_like = (
            e.__class__.__name__ in ("ExternalShutdownException", "RCLError")
            and ("shutdown" in str(e).lower() or "context is invalid" in str(e).lower())
        )
        if not shutdown_like:
            raise
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except KeyboardInterrupt:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--x", type=float, default=0.55, help="grasp X [m] (기본 0.55)")
    parser.add_argument("--y", type=float, default=0.0, help="grasp Y [m] (기본 0.0)")
    parser.add_argument("--z", type=float, default=None,
                        help="grasp Z [m] (기본 Config.GRASP_Z)")
    parser.add_argument("--hold", type=float, default=1.0,
                        help="석션 ON 후 파킹 홀드 [s] (기본 1.0)")
    parser.add_argument("--lift", type=float, default=0.10,
                        help="리프트 높이 [m] (기본 0.10)")
    parser.add_argument("--vel-scale", type=float, default=0.3,
                        help="저속 이동 관절속도 스케일 (기본 0.3)")
    parser.add_argument("--bin-x", type=float, default=1.5,
                        help="throw bin X [m] (기본 1.5)")
    parser.add_argument("--bin-y", type=float, default=0.0,
                        help="throw bin Y [m] (기본 0.0)")
    parser.add_argument("--bin-z-offset", type=float, default=BIN_Z_OFFSET_DEFAULT,
                        help="bin 목표 높이 = grasp Z + offset [m] (기본 0.10)")
    parser.add_argument(
        "--release-distance", "--release-x-offset", dest="release_distance",
        type=float, default=RELEASE_DISTANCE_DEFAULT,
        help="pick→bin 방향 release 기준 거리 [m] (기본 0.10; "
             "--release-x-offset은 하위 호환 별칭)",
    )
    parser.add_argument(
        "--release-z-offset", type=float, default=RELEASE_Z_OFFSET_DEFAULT,
        help="release Z = grasp Z + offset [m] (기본 0.33)",
    )
    parser.add_argument("--tool-offset", type=float, default=TOOL_OFFSET_DEFAULT,
                        help="gp8.py/MuJoCo TCP보다 +X 앞의 추가 offset [m] (기본 0.0)")
    parser.add_argument("--release-lead", type=float, default=0.0,
                        help="석션 off 를 release 보다 이만큼 일찍 [s] (음수=늦게)")
    parser.add_argument("--plan-only", action="store_true",
                        help="로봇 없이 throw 계획 검증만 출력")
    parser.add_argument("--rviz-preview", action="store_true",
                        help="로봇에 명령을 보내지 않고 RViz용 marker/path/joint_states publish")
    parser.add_argument("--preview-rate", type=float, default=60.0,
                        help="RViz preview joint_states publish rate [Hz] (기본 60)")
    parser.add_argument("--preview-speed", type=float, default=PREVIEW_SPEED_DEFAULT,
                        help="RViz preview 재생 배속 (기본 0.25, 1.0=실시간)")
    parser.add_argument("--frame-id", default="base_link",
                        help="RViz marker/path frame id (기본 base_link)")
    args, _ros_unknown = parser.parse_known_args()

    cfg = Config()
    z = cfg.GRASP_Z if args.z is None else args.z
    gp8 = GP8()
    vel_limits = np.asarray(gp8.velocity_limits, dtype=float)

    if args.plan_only:
        plan_only(gp8, args, z, vel_limits)
        return

    if args.rviz_preview:
        rviz_preview(gp8, args, z, vel_limits)
        return

    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from gp8_control.controllers.trajectory_controller import TrajectoryController

    rclpy.init()
    node = Node("suction_lift_debug")
    ctrl = TrajectoryController(node)

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        print("Waiting for servers...")
        if not ctrl.wait_for_servers(timeout_sec=10.0):
            print("  서버 없음 — ~/ros2_ws/debug_bringup.sh 가 떠 있는지 확인하세요.")
            return

        # app.py 와 동일: 엔티티 생성 + 서버 확인 후 백그라운드 MTE 가 유일한
        # 스피너 (joint_states 콜백 + 액션 future 처리).
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()

        print("Waiting for joint state...")
        deadline = time.time() + 5.0
        while ctrl.current_joints is None and time.time() < deadline:
            time.sleep(0.05)
        if ctrl.current_joints is None:
            print("  joint state 미수신 — joint_state_broadcaster 확인.")
            return
        print(f"  current joints (deg): "
              f"{[round(float(np.degrees(j)), 1) for j in ctrl.current_joints]}")

        M1 = vel_limits * args.vel_scale
        M2 = M1 * JOINT_ACCEL_RATIO

        print("\n⚠️  실제 로봇 모션입니다. 컨베이어 정지 + 주변 공간 확보 확인.")
        while True:
            run_cycle(ctrl, gp8, args, z, M1, M2, cfg.TRAJ_HZ, vel_limits)
            ans = input(
                "\n다시 실행? (y = 같은 지점 / 'x y' 새 좌표 입력 / N 종료) > "
            ).strip().lower()
            if ans == "y":
                continue
            parts = ans.split()
            if len(parts) == 2:
                try:
                    args.x, args.y = float(parts[0]), float(parts[1])
                except ValueError:
                    print("  좌표 해석 실패 — 종료합니다.")
                    break
                continue
            break
    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted.")
    finally:
        # 어떤 경로로 끝나든 석션은 끄고 나간다.
        try:
            ctrl.suction_off()
            time.sleep(0.3)   # IO 워커 스레드가 큐를 비울 시간
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

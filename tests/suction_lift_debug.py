"""석션 단독 디버그 — 컨베이어 정지 상태, 지정 지점에서 pick + throw.

기본 사이클: 지정한 베이스 좌표 (x, y) 위 석션 대기 높이(Config.GRASP_Z —
throw_skill 이 ambush 대기에 쓰는 그 그랩 높이)로 내려가 파킹 → 석션 ON(기본
1초 홀드, 진공 형성) → 10 cm 리프트. 이후 선택적으로 **throw**:

  * release XY = pick→bin 방향 10 cm, release Z = pick +33 cm을 우선
    사용한다. 계획이 실패하면 거리 10~35 cm, Z 20~60 cm에서
    제한을 만족하는 가장 가까운 release를 자동 선택한다.
  * release→bin 포물선은 두 지점을 연결하는 최소 속도 탄도를 계산한다.
    release에서 툴 축은 투척 전방 기준 아래 30°다.
  * 스윙은 TCP에 원호를 강제하지 않는다. release pose와 탄도 선속도,
    투척 평면 내 각속도를 spatial Jacobian으로 관절 상태로 바꾼 뒤,
    10 cm lift 종료점→release→팔로스루 전체를 하나의 7차 minimum-jerk
    polynomial로 최적화한다. 별도 runup 이동은 없다. lift 종료점은 정지
    상태, release 위치/속도는 정확한 내부 제약, 끝점은 정지 상태이며 전
    구간의 위치/속도/가속도 제한과 lift→release TCP 경로 길이(직선의
    1.25배 이하)를 검사한다. 석션 off는
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
from functools import lru_cache

import numpy as np

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
RELEASE_DISTANCE_MAX = 0.35
RELEASE_DISTANCE_STEP = 0.05
RELEASE_Z_OFFSET_MIN = 0.20
RELEASE_Z_OFFSET_MAX = 0.60
RELEASE_Z_OFFSET_STEP = 0.02
# gp8_control/gp8_bringup.launch.py의 실제 로봇 기본값. YRC external-
# increment 경로의 축별 한계는 GP8.rt_stream_velocity_limits(factor=1.0)
# 에서 환산하고, 아래 throw scale로 10% 안전 여유를 둔다.
AXIS_INCREMENT_FACTOR_DEFAULT = 1.0
AXIS_ACCELERATION_FACTOR_DEFAULT = GP8.DEFAULT_RT_ACCELERATION_FACTOR
THROW_ACCEL_SCALE = 0.90
# 장거리 throw를 위해 YRC RT-stream 실측 관절속도 상한을 100% 사용한다.
# axis_increment_factor 자체는 하드웨어 허용범위인 1.0을 넘기지 않는다.
THROW_VELOCITY_SCALE = 1.00
THROW_POLY_DEGREE = 7
# release를 전체 polynomial의 어느 시점에 둘지도 함께 탐색한다. lift에서
# release까지가 팔로스루보다 길도록 50% 뒤쪽만 허용한다.
THROW_RELEASE_FRACTIONS = (0.55, 0.60, 0.65, 0.70)
THROW_DURATION_SCALE_MIN = 0.75
THROW_DURATION_SCALE_MAX = 4.0
THROW_DURATION_COARSE_SAMPLES = 21
THROW_DURATION_REFINE_SAMPLES = 13
THROW_POLY_CHECK_SAMPLES = 81
THROW_POLY_FINAL_CHECK_SAMPLES = 401
# lift→release TCP FK 곡선이 두 점의 직선거리보다 25% 넘게 우회하면
# minimum-jerk 값이 작아도 불필요한 backswing으로 보고 버린다.
THROW_TCP_PATH_RATIO_MAX = 1.25
THROW_TCP_PATH_PREFILTER_SAMPLES = 7
THROW_TCP_PATH_CHECK_SAMPLES = 401
RELEASE_OMEGA_MAX = 8.0          # release 평면내 각속도 탐색 상한 [rad/s]
RELEASE_OMEGA_STEP = 0.05        # 경로 제약 후보 탐색 간격 [rad/s]
PREVIEW_SPEED_DEFAULT = 0.25     # RViz 재생 배속 (실제 시간의 1/4)
PREVIEW_STATIC_RATE = 1.0        # marker/path 재발행 주기 [Hz]
RVIZ_TCP_VELOCITY_VECTOR_TIME = 0.10  # 속도 화살표 길이 = v_tcp * 이 시간 [m]
BIN_Z_OFFSET_DEFAULT = 0.10      # bin 목표 높이 = grasp z + 이 값 [m]
TOOL_OFFSET_DEFAULT = 0.0        # gp8.py EE는 MuJoCo grip_site/TCP와 일치하므로 추가 offset 없음.
TOOL_FRAME = "suction_tool"

# GP8 기하 (robots/gp8.py 스크류 정의): 어깨(J2) 위치와 링크 도달 한계.
SHOULDER_XZ = (0.04, 0.330)      # J1 축 기준 어깨 오프셋 (x, z) [m]
WRIST_REACH = 0.6875             # 어깨→손목중심 최대 (상완 0.345 + 전완 0.3425) [m]
TOOL_LEN = 0.325                 # 손목중심→기구학 EE 원점 [m]


def _rt_motion_limits(gp8: GP8, increment_factor: float, acceleration_factor: float):
    """현재 YRC external-increment factor의 축별 속도/가속도 상한.

    ``rt_stream_velocity_limits``는 factor=1.0에서 실측한 속도다. 가속도
    환산은 GP8 모델의 공용 ``rt_stream_acceleration_limits``를 사용한다.
    """
    increment_factor = float(increment_factor)
    acceleration_factor = float(acceleration_factor)
    if not 0.0 < increment_factor <= 1.0:
        raise ValueError("axis increment factor는 (0, 1] 범위여야 합니다")
    if not 0.0 < acceleration_factor <= 1.0:
        raise ValueError("axis acceleration factor는 (0, 1] 범위여야 합니다")
    full_speed = np.asarray(gp8.rt_stream_velocity_limits, dtype=float)
    velocity_limits = full_speed * increment_factor
    acceleration_limits = gp8.rt_stream_acceleration_limits(acceleration_factor)
    return velocity_limits, acceleration_limits


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


def _poly_basis(s, derivative: int = 0, degree: int = THROW_POLY_DEGREE):
    """``s``에서 power-basis polynomial의 미분 기저 행렬을 만든다."""
    s = np.atleast_1d(np.asarray(s, dtype=float))
    basis = np.zeros((s.size, degree + 1), dtype=float)
    for power in range(derivative, degree + 1):
        factor = 1.0
        for k in range(derivative):
            factor *= power - k
        basis[:, power] = factor * s ** (power - derivative)
    return basis


@lru_cache(maxsize=len(THROW_RELEASE_FRACTIONS))
def _minimum_jerk_constraint_map(release_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """7차 minimum-jerk QP의 등식제약→계수 선형 map을 반환한다.

    각 관절에 대해 ``q(s)=sum(a[k] s**k)``이고 ``s=t/T``다. 제약은

      q(0)=q_start, q'(0)=q''(0)=0,
      q(r)=q_release, q'(r)=qd_release*T,
      q'(1)=q''(1)=0

    이다. 남는 1 자유도는 ``integral_0^1 q'''(s)^2 ds``를 최소화한다.
    따라서 release 앞/뒤를 접합한 두 다항식이 아니라 전체 throw에 하나의
    계수 벡터만 사용하고, release에서 모든 차수의 미분이 자동 연속이다.
    """
    r = float(release_fraction)
    if not 0.0 < r < 1.0:
        raise ValueError(f"release fraction은 (0, 1)이어야 합니다 ({r})")

    n_coeff = THROW_POLY_DEGREE + 1
    Q = np.zeros((n_coeff, n_coeff), dtype=float)
    for i in range(3, n_coeff):
        ci = i * (i - 1) * (i - 2)
        for j in range(3, n_coeff):
            cj = j * (j - 1) * (j - 2)
            Q[i, j] = ci * cj / (i + j - 5)

    C = np.vstack([
        _poly_basis([0.0], 0)[0],
        _poly_basis([0.0], 1)[0],
        _poly_basis([0.0], 2)[0],
        _poly_basis([r], 0)[0],
        _poly_basis([r], 1)[0],
        _poly_basis([1.0], 1)[0],
        _poly_basis([1.0], 2)[0],
    ])
    n_constraints = C.shape[0]
    KKT = np.block([
        [Q, C.T],
        [C, np.zeros((n_constraints, n_constraints), dtype=float)],
    ])
    rhs = np.vstack([
        np.zeros((n_coeff, n_constraints), dtype=float),
        np.eye(n_constraints, dtype=float),
    ])
    try:
        constraint_map = np.linalg.solve(KKT, rhs)[:n_coeff]
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"minimum-jerk polynomial QP 해석 실패: {exc}") from exc
    return constraint_map, Q


def _joint_polynomial_profile(
    q_start, q_release, qd_release, total_duration: float,
    release_fraction: float, dt: float, sample_s=None,
):
    """단일 7차 minimum-jerk throw polynomial을 계산한다."""
    q_start = np.asarray(q_start, dtype=float)
    q_release = np.asarray(q_release, dtype=float)
    qd_release = np.asarray(qd_release, dtype=float)
    T = float(total_duration)
    r = float(release_fraction)
    if T <= 0.0:
        raise ValueError(f"throw duration은 양수여야 합니다 ({T})")

    constraint_map, Q = _minimum_jerk_constraint_map(r)
    zeros = np.zeros_like(q_start)
    constraints = np.vstack([
        q_start, zeros, zeros, q_release, qd_release * T, zeros, zeros,
    ])
    coeff = constraint_map @ constraints

    if sample_s is None:
        n_uniform = max(9, int(np.ceil(T / dt)) + 1)
        s = np.unique(np.concatenate([
            np.linspace(0.0, 1.0, n_uniform), np.array([r]),
        ]))
    else:
        s = np.unique(np.concatenate([
            np.asarray(sample_s, dtype=float), np.array([0.0, r, 1.0]),
        ]))
    q = (_poly_basis(s, 0) @ coeff).T
    qd = (_poly_basis(s, 1) @ coeff).T / T
    qdd = (_poly_basis(s, 2) @ coeff).T / T ** 2
    ts = s * T
    i_release = int(np.argmin(np.abs(s - r)))
    # physical jerk 제곱 적분: ds 적분값에 1/T^5가 곱해진다.
    jerk_cost = float(np.sum(coeff * (Q @ coeff)) / T ** 5)
    return q, qd, qdd, ts, i_release, coeff, jerk_cost


def _optimize_joint_polynomial(
    q_start, q_release, qd_release, nominal_release_duration,
    vel_limits, accel_limits, soft_lower, soft_upper,
):
    """release 시점과 전체 시간을 탐색해 가장 짧은 feasible polynomial 선택."""
    q_start = np.asarray(q_start, dtype=float)
    q_release = np.asarray(q_release, dtype=float)
    qd_release = np.asarray(qd_release, dtype=float)
    vel_limits = np.asarray(vel_limits, dtype=float)
    accel_limits = np.asarray(accel_limits, dtype=float)
    soft_lower = np.asarray(soft_lower, dtype=float)
    soft_upper = np.asarray(soft_upper, dtype=float)
    best = None
    check_s = np.linspace(0.0, 1.0, THROW_POLY_CHECK_SAMPLES)
    for release_fraction in THROW_RELEASE_FRACTIONS:
        nominal_total = float(nominal_release_duration) / release_fraction
        duration_min = max(
            4.0 * THROW_DT, THROW_DURATION_SCALE_MIN * nominal_total,
        )
        duration_max = max(
            5.0 * THROW_DT, THROW_DURATION_SCALE_MAX * nominal_total,
        )
        # constraint의 release velocity 행만 T에 비례하므로 polynomial 계수는
        # coeff(T)=coeff_const + T*coeff_slope다. 131개 duration을 Python loop로
        # 다시 만들지 않고 한 번에 broadcast 평가한다.
        constraint_map, Q = _minimum_jerk_constraint_map(release_fraction)
        zeros = np.zeros_like(q_start)
        constraints_const = np.vstack([
            q_start, zeros, zeros, q_release, zeros, zeros, zeros,
        ])
        constraints_slope = np.vstack([
            zeros, zeros, zeros, zeros, qd_release, zeros, zeros,
        ])
        coeff_const = constraint_map @ constraints_const
        coeff_slope = constraint_map @ constraints_slope
        basis_q = _poly_basis(check_s, 0)
        basis_qd = _poly_basis(check_s, 1)
        basis_qdd = _poly_basis(check_s, 2)
        q_const = basis_q @ coeff_const
        q_slope = basis_q @ coeff_slope
        qd_const = basis_qd @ coeff_const
        qd_slope = basis_qd @ coeff_slope
        qdd_const = basis_qdd @ coeff_const
        qdd_slope = basis_qdd @ coeff_slope
        def evaluate_durations(durations):
            durations = np.asarray(durations, dtype=float)
            coeffs = (
                coeff_const[None, :, :]
                + durations[:, None, None] * coeff_slope[None, :, :]
            )
            T = durations[:, None, None]
            positions = q_const[None, :, :] + T * q_slope[None, :, :]
            velocities = qd_const[None, :, :] / T + qd_slope[None, :, :]
            accelerations = (
                qdd_const[None, :, :] / T ** 2
                + qdd_slope[None, :, :] / T
            )
            velocity_ratios = np.max(
                np.abs(velocities) / vel_limits[None, None, :], axis=(1, 2),
            )
            acceleration_ratios = np.max(
                np.abs(accelerations) / accel_limits[None, None, :],
                axis=(1, 2),
            )
            position_ok = np.all(
                (positions >= soft_lower[None, None, :] - 1e-9)
                & (positions <= soft_upper[None, None, :] + 1e-9),
                axis=(1, 2),
            )
            feasible = (
                (velocity_ratios <= THROW_VELOCITY_SCALE + 1e-9)
                & (acceleration_ratios <= THROW_ACCEL_SCALE + 1e-9)
                & position_ok
            )
            return dict(
                durations=durations, coeffs=coeffs,
                accelerations=accelerations,
                velocity_ratios=velocity_ratios,
                acceleration_ratios=acceleration_ratios,
                feasible=feasible,
            )

        # 41점 coarse 탐색으로 feasible 구간을 찾고 그 한 구간만 17점으로
        # 재탐색한다. 기존 131점 전수평가보다 계산량은 작고 시간 해상도는 높다.
        coarse = evaluate_durations(np.linspace(
            duration_min, duration_max, THROW_DURATION_COARSE_SAMPLES,
        ))
        feasible_indices = np.flatnonzero(coarse["feasible"])
        if feasible_indices.size == 0:
            continue
        coarse_idx = int(feasible_indices[0])
        if coarse_idx == 0:
            selected = coarse
            idx = 0
        else:
            selected = evaluate_durations(np.linspace(
                coarse["durations"][coarse_idx - 1],
                coarse["durations"][coarse_idx],
                THROW_DURATION_REFINE_SAMPLES,
            ))
            refined_indices = np.flatnonzero(selected["feasible"])
            if refined_indices.size == 0:
                # coarse endpoint 자체는 feasible이므로 수치 오차 시 그 값을 사용.
                selected = coarse
                idx = coarse_idx
            else:
                idx = int(refined_indices[0])

        total_duration = float(selected["durations"][idx])
        coeff = selected["coeffs"][idx]
        q_coeff = Q @ coeff
        jerk_cost = float(np.sum(coeff * q_coeff) / total_duration ** 5)
        score = (total_duration, jerk_cost)
        result = dict(
            score=score, release_fraction=float(release_fraction),
            duration=total_duration, coeff=coeff,
            vel_ratio=float(selected["velocity_ratios"][idx]),
            accel_ratio=float(selected["acceleration_ratios"][idx]),
            accel_peak=np.max(
                np.abs(selected["accelerations"][idx]), axis=0,
            ),
            jerk_cost=jerk_cost,
        )
        if best is None or score < best["score"]:
            best = result

    if best is None:
        return None

    # 81점 탐색은 후보를 빠르게 고르는 용도다. 최종 선택은 401점으로 다시
    # 검사하여 샘플 사이의 속도/가속도 peak나 관절 리미트 위반을 허용하지 않는다.
    check_traj, check_vel, check_accel, _, _, _, _ = (
        _joint_polynomial_profile(
            q_start, q_release, qd_release, best["duration"],
            best["release_fraction"], THROW_DT,
            sample_s=np.linspace(0.0, 1.0, THROW_POLY_FINAL_CHECK_SAMPLES),
        )
    )
    final_velocity_ratio = float(np.max(
        np.abs(check_vel) / vel_limits[:, None]
    ))
    final_acceleration_ratio = float(np.max(
        np.abs(check_accel) / accel_limits[:, None]
    ))
    if (
        final_velocity_ratio > THROW_VELOCITY_SCALE + 1e-9
        or final_acceleration_ratio > THROW_ACCEL_SCALE + 1e-9
        or np.any(check_traj < soft_lower[:, None] - 1e-9)
        or np.any(check_traj > soft_upper[:, None] + 1e-9)
    ):
        return None
    best.update(
        vel_ratio=final_velocity_ratio,
        accel_ratio=final_acceleration_ratio,
        accel_peak=np.max(np.abs(check_accel), axis=1),
    )

    # 컨트롤러에 보낼 THROW_DT knot를 다시 생성한다. release fraction을
    # 반드시 knot로 삽입하므로 timed suction release와 상태 제약이 정확히 맞는다.
    traj, vel, accel, ts, i_rel, coeff, jerk_cost = _joint_polynomial_profile(
        q_start, q_release, qd_release, best["duration"],
        best["release_fraction"], THROW_DT,
    )
    best.update(
        traj=traj, vel=vel, accel=accel, ts=ts, release_idx=i_rel,
        coeff=coeff, jerk_cost=jerk_cost, q_end=traj[:, -1].copy(),
    )
    return best


def build_throw(gp8: GP8, plan: dict, q_seed, vel_limits, accel_limits,
                tool_offset: float = TOOL_OFFSET_DEFAULT):
    """release constraint를 만족하는 단일 polynomial 관절 스윙을 만든다.

    release 평면 법선 축의 각속도를 탐색하여 release TCP 선속도를
    정확히 맞춘다. 각 후보의 시작 관절점, release 관절점/속도를 제약으로
    걸고 시작→release→팔로스루 전체를 하나의 7차 minimum-jerk polynomial로
    최적화한다. TCP 경로는 그 관절 polynomial의 FK 결과이다.
    """
    q_seed = np.asarray(q_seed, dtype=float)
    vel_limits = np.asarray(vel_limits, dtype=float)
    accel_limits = np.asarray(accel_limits, dtype=float)
    joint_limits = np.asarray(gp8.joint_limits, dtype=float)
    soft_lower = joint_limits[:, 0] + JOINT_LIMIT_MARGIN
    soft_upper = joint_limits[:, 1] - JOINT_LIMIT_MARGIN
    effective_accel = accel_limits * THROW_ACCEL_SCALE
    effective_velocity = vel_limits * THROW_VELOCITY_SCALE
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
    reject_counts = {
        "velocity": 0, "position": 0, "wrist": 0,
        "polynomial": 0, "path_length": 0, "floor": 0,
    }

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
        # throw 시작점은 별도 와인드업 점이 아니라 이미 물체를 10 cm 들어
        # 올린 q_seed 그 자체다. lift→release→follow-through가 한 polynomial
        # 안에 들어가므로 lift 뒤에 방향이 꺾이는 segment 경계가 생기지 않는다.
        q_start = q_seed.copy()
        q_delta = np.abs(q_release - q_start)
        nominal_release_duration = float(max(
            np.max(np.abs(qd_release) / effective_accel),
            np.max(np.sqrt(2.0 * q_delta / effective_accel)),
            np.max(q_delta / effective_velocity),
        ))
        if nominal_release_duration <= 1e-6:
            continue
        q_bounds = np.column_stack([q_start, q_release])
        if (
            np.any(q_bounds < soft_lower[:, None] - 1e-9)
            or np.any(q_bounds > soft_upper[:, None] + 1e-9)
        ):
            reject_counts["position"] += 1
            continue
        wrist_delta = (
            q_release[[3, 5]] - q_seed[[3, 5]] + np.pi
        ) % (2.0 * np.pi) - np.pi
        wrist_jump = float(np.max(np.abs(wrist_delta)))
        if wrist_jump > WRIST_BRANCH_JUMP_MAX:
            reject_counts["wrist"] += 1
            continue

        score = (nominal_release_duration, -float(np.min(np.minimum(
            q_bounds - joint_limits[:, 0, None],
            joint_limits[:, 1, None] - q_bounds,
        ))))
        kinematic_candidates.append(dict(
            score=score,
            q_start=q_start, q_release=q_release.copy(),
            qd_release=qd_release, omega_release=omega_release,
            release_omega=float(omega_mag), vel_ratio=vel_ratio,
            nominal_release_duration=nominal_release_duration,
            wrist_branch_jump=wrist_jump,
        ))

    if not kinematic_candidates:
        raise ValueError(
            "release constraint를 만족하는 관절 스윙을 만들 수 없습니다 "
            f"(제외: {reject_counts}) — release Z를 높이거나 bin을 가깝게 하세요"
        )

    best = None
    # 관절 조건으로 정렬한 뒤 각 후보에 단일 polynomial 최적화를 적용하고,
    # feasible한 후보만 FK로 바닥 간섭을 검사한다.
    for candidate in sorted(kinematic_candidates, key=lambda c: c["score"]):
        optimized = _optimize_joint_polynomial(
            candidate["q_start"], q_release, candidate["qd_release"],
            candidate["nominal_release_duration"], vel_limits, accel_limits,
            soft_lower, soft_upper,
        )
        if optimized is None:
            reject_counts["polynomial"] += 1
            continue
        # 전체 10 ms trajectory를 FK하기 전에 13개 chord로 경로 길이를
        # 하한 평가한다. chord 합이 이미 limit을 넘으면 실제 곡선은 반드시
        # 더 길기 때문에 안전하게 즉시 제외할 수 있다.
        prefilter_s = np.linspace(
            0.0, optimized["release_fraction"],
            THROW_TCP_PATH_PREFILTER_SAMPLES,
        )
        prefilter_traj = (
            _poly_basis(prefilter_s, 0) @ optimized["coeff"]
        ).T
        prefilter_pos = np.zeros((prefilter_traj.shape[1], 3), dtype=float)
        for k in range(prefilter_traj.shape[1]):
            Tk = gp8.forward_kinematics(prefilter_traj[:, k])
            prefilter_pos[k] = Tk[:3, 3] + Tk[:3, 0] * float(tool_offset)
        prefilter_direct = float(np.linalg.norm(
            prefilter_pos[-1] - prefilter_pos[0]
        ))
        if prefilter_direct <= 1e-9:
            reject_counts["path_length"] += 1
            continue
        prefilter_length = float(np.sum(np.linalg.norm(
            np.diff(prefilter_pos, axis=0), axis=1,
        )))
        if (
            prefilter_length / prefilter_direct
            > THROW_TCP_PATH_RATIO_MAX + 1e-9
        ):
            reject_counts["path_length"] += 1
            continue

        traj = optimized["traj"]
        swing_pos = np.zeros((traj.shape[1], 3), dtype=float)
        swing_R = np.zeros((traj.shape[1], 3, 3), dtype=float)
        for k in range(traj.shape[1]):
            Tk = gp8.forward_kinematics(traj[:, k])
            swing_R[k] = Tk[:3, :3]
            swing_pos[k] = Tk[:3, 3] + Tk[:3, 0] * float(tool_offset)
        i_release = optimized["release_idx"]
        direct_runup = float(np.linalg.norm(
            swing_pos[i_release] - swing_pos[0]
        ))
        runup_path_length = float(np.sum(np.linalg.norm(
            np.diff(swing_pos[:i_release + 1], axis=0), axis=1,
        )))
        if direct_runup <= 1e-9:
            reject_counts["path_length"] += 1
            continue
        runup_path_ratio = runup_path_length / direct_runup
        if runup_path_ratio > THROW_TCP_PATH_RATIO_MAX + 1e-9:
            reject_counts["path_length"] += 1
            continue
        # 10 ms controller knot의 chord 합은 곡선 길이를 작게 평가할 수 있다.
        # coarse 검사를 통과한 후보만 dense FK로 재검산해 hard limit을 보장한다.
        dense_s = np.linspace(
            0.0, optimized["release_fraction"],
            THROW_TCP_PATH_CHECK_SAMPLES,
        )
        dense_traj = (_poly_basis(dense_s, 0) @ optimized["coeff"]).T
        dense_runup_pos = np.zeros((dense_traj.shape[1], 3), dtype=float)
        for k in range(dense_traj.shape[1]):
            Tk = gp8.forward_kinematics(dense_traj[:, k])
            dense_runup_pos[k] = (
                Tk[:3, 3] + Tk[:3, 0] * float(tool_offset)
            )
        direct_runup = float(np.linalg.norm(
            dense_runup_pos[-1] - dense_runup_pos[0]
        ))
        runup_path_length = float(np.sum(np.linalg.norm(
            np.diff(dense_runup_pos, axis=0), axis=1,
        )))
        runup_path_ratio = runup_path_length / direct_runup
        if runup_path_ratio > THROW_TCP_PATH_RATIO_MAX + 1e-9:
            reject_counts["path_length"] += 1
            continue
        z_min = float(np.min(swing_pos[:, 2]))
        if z_min < plan["z_g"] + FLOOR_CLEARANCE:
            reject_counts["floor"] += 1
            continue
        candidate.update(
            optimized, swing_pos=swing_pos, swing_R=swing_R, z_min=z_min,
            runup_direct=direct_runup, runup_path_length=runup_path_length,
            runup_path_ratio=runup_path_ratio,
        )
        best = candidate
        break

    if best is None:
        raise ValueError(
            "관절 polynomial 후보가 경로 길이 또는 바닥 제약을 만족하지 못했습니다 "
            f"(제외: {reject_counts})"
        )

    Jr = gp8.jacobian(best["q_release"])
    twist_chk = Jr @ best["qd_release"]
    v_chk = twist_chk[3:] + np.cross(twist_chk[:3], release_pos)
    release_fk = best["swing_pos"][best["release_idx"]]
    relative_R = plan["throw_R"] @ best["swing_R"][0].T
    orientation_swing = float(np.arccos(np.clip(
        0.5 * (np.trace(relative_R) - 1.0), -1.0, 1.0,
    )))
    j5_upper = float(joint_limits[4, 1])
    j5_max = float(np.max(best["traj"][4]))
    i_release = best["release_idx"]
    plan["start_pos"] = best["swing_pos"][0].copy()
    plan["end_pos"] = best["swing_pos"][-1].copy()
    best.update(
        v_err=float(np.linalg.norm(v_chk - plan["v_rel"])),
        omega_err=float(np.linalg.norm(twist_chk[:3] - best["omega_release"])),
        release_pos_error=float(np.linalg.norm(release_fk - release_pos)),
        orientation_swing=float(orientation_swing),
        j5_max=j5_max, j5_upper=j5_upper, j5_margin=j5_upper - j5_max,
        polynomial_release_q_error=float(np.linalg.norm(
            best["traj"][:, i_release] - best["q_release"]
        )),
        polynomial_release_qd_error=float(np.linalg.norm(
            best["vel"][:, i_release] - best["qd_release"]
        )),
        endpoint_velocity_max=float(np.max(np.abs(
            best["vel"][:, [0, -1]]
        ))),
        endpoint_acceleration_max=float(np.max(np.abs(
            best["accel"][:, [0, -1]]
        ))),
        duration=float(best["ts"][-1]), n_knots=best["traj"].shape[1],
        velocity_limits=vel_limits.copy(), acceleration_limits=accel_limits.copy(),
        swing_vel=None,
    )
    best.pop("score", None)
    return best


def _adaptive_release_candidates(preferred_distance: float, preferred_z: float):
    """기준 release 우선, 이후 큰 수평거리/Z 근접 순으로 후보를 만든다.

    경로 길이 제약에서는 pick에 가까운 release가 긴 backswing을 만들기 쉽다.
    사용자 지정 기준값은 항상 첫 번째로 보존하되, 실패 후에는 bin에 더 가까워
    탄도 속도와 우회 경로가 작아지는 큰 release distance부터 검사한다.
    """
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
        0 if (
            abs(c[0] - preferred_distance) <= 1e-9
            and abs(c[1] - preferred_z) <= 1e-9
        ) else 1,
        -c[0],
        abs(c[1] - preferred_z),
        c[1],
    ))
    return candidates


def plan_and_build_adaptive_throw(
    gp8: GP8, grasp_xyz, bin_xy, bin_z_offset: float, q_seed, vel_limits,
    accel_limits,
    preferred_distance: float = RELEASE_DISTANCE_DEFAULT,
    preferred_z_offset: float = RELEASE_Z_OFFSET_DEFAULT,
    tool_offset: float = TOOL_OFFSET_DEFAULT,
):
    """release 거리/Z를 adaptive 탐색해 ``(plan, built)``를 반환한다.

    기준값을 먼저 시도하고, 실패할 때만 pick→bin 방향 10~35 cm,
    pick Z +20~60 cm 격자를 큰 release 거리/Z 기준값 근접 순으로 탐색한다.
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
                accel_limits=accel_limits,
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
    print("\n=== Polynomial-optimized Throw Plan ===")
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
    release_time = b["duration"] * b["release_fraction"]
    print(f"  관절 스윙   : 단일 {THROW_POLY_DEGREE}차 minimum-jerk polynomial, "
          f"release @ {b['release_fraction'] * 100:.0f}%")
    print(f"              lift→release {release_time:.3f}s + "
          f"팔로스루 {b['duration'] - release_time:.3f}s, "
          f"총 {b['duration']:.3f}s / {b['n_knots']} knots, "
          f"z_min={b['z_min']:.3f} m")
    print(f"  제한 검사  : max J5={np.degrees(b['j5_max']):.2f}° "
          f"/ limit {np.degrees(b['j5_upper']):.2f}° "
          f"(여유 {np.degrees(b['j5_margin']):.2f}°), "
          f"속도 {b['vel_ratio'] * 100:.1f}% / 가속도 {b['accel_ratio'] * 100:.1f}%")
    print(f"              lift→release TCP path {b['runup_path_length']:.3f} m "
          f"/ 직선 {b['runup_direct']:.3f} m = {b['runup_path_ratio']:.3f}x "
          f"(limit {THROW_TCP_PATH_RATIO_MAX:.2f}x)")
    print(f"              RT accel limits="
          f"{np.round(b['acceleration_limits'], 1).tolist()} rad/s², "
          f"throw scales: velocity={THROW_VELOCITY_SCALE * 100:.0f}%, "
          f"acceleration={THROW_ACCEL_SCALE * 100:.0f}%")
    print(f"  orientation : 자유 관절 스윙, lift→release "
          f"{np.degrees(b['orientation_swing']):.2f}°, "
          f"release |ω|={np.degrees(b['release_omega']):.1f}°/s")
    print(f"  release knot {b['release_idx']} (lead {release_lead:+.3f}s), "
          f"관절속도 max|q̇|/limit = {b['vel_ratio'] * 100:.1f}%, "
          f"release 위치/속도오차 {b['release_pos_error'] * 1000:.3f} mm / "
          f"{b['v_err'] * 1000:.2f} mm/s, "
          f"각속도오차 {np.degrees(b['omega_err']):.3f}°/s")
    print(f"              polynomial 제약오차 |Δq|="
          f"{b['polynomial_release_q_error']:.2e} rad, |Δq̇|="
          f"{b['polynomial_release_qd_error']:.2e} rad/s, "
          f"양끝 max |q̇|={b['endpoint_velocity_max']:.2e} rad/s, "
          f"|q̈|={b['endpoint_acceleration_max']:.2e} rad/s²")
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


# =====================================================================
# 사이클
# =====================================================================
def run_throw(ctrl, gp8: GP8, args, grasp_xyz, lift_q, move_vel_limits,
              throw_vel_limits, accel_limits) -> None:
    plan, built = plan_and_build_adaptive_throw(
        gp8, grasp_xyz, (args.bin_x, args.bin_y), args.bin_z_offset,
        q_seed=lift_q, vel_limits=throw_vel_limits, accel_limits=accel_limits,
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

    print(f"→ lift 자세에서 단일 polynomial throw 시작 "
          f"({built['duration']:.2f}s, |v|={plan['speed']:.2f} m/s, "
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
    move_vel_limits = np.asarray(move_vel_limits, dtype=float)
    _move(
        ctrl, built["q_end"], lift_q,
        move_vel_limits * args.vel_scale,
        move_vel_limits * args.vel_scale * JOINT_ACCEL_RATIO,
        Config().TRAJ_HZ,
    )
    print("  복귀 완료.")


def run_cycle(ctrl, gp8: GP8, args, z: float, M1, M2, hz: float,
              move_vel_limits, throw_vel_limits, accel_limits) -> None:
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
            run_throw(
                ctrl, gp8, args, np.array([x, y, z]), lift_q,
                move_vel_limits, throw_vel_limits, accel_limits,
            )
        except ValueError as e:
            print(f"  throw 계획 실패: {e}")
            input("\nEnter → 석션 OFF (릴리즈) ")
            ctrl.suction_off()
    else:
        ctrl.suction_off()
        print("  릴리즈 완료.")


def plan_only(gp8: GP8, args, z: float, move_vel_limits,
              throw_vel_limits, accel_limits) -> None:
    """로봇/ROS 없이 throw 기하 + IK + 관절궤적을 검증 출력."""
    grasp_xyz = np.array([args.x, args.y, z])
    q_seed = _ik_tool_down(gp8, args.x, args.y, z + args.lift, tool_offset=args.tool_offset)
    if q_seed is None:
        print("lift 자세 IK 실패 — 그랩 지점을 확인하세요.")
        return
    try:
        plan, built = plan_and_build_adaptive_throw(
            gp8, grasp_xyz, (args.bin_x, args.bin_y), args.bin_z_offset,
            q_seed=q_seed, vel_limits=throw_vel_limits, accel_limits=accel_limits,
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
    # q_start는 lift_q와 동일하다. throw 뒤 복귀 segment만 별도로 검사한다.
    M1 = np.asarray(move_vel_limits, dtype=float) * args.vel_scale
    M2 = M1 * JOINT_ACCEL_RATIO
    zero = np.zeros_like(M1)
    _, _, ts_out = trajectory(built["q_end"], zero, q_seed, zero, M1, M2, hertz=100.0)
    print(f"  throw_start == lift: max |Δq|="
          f"{np.max(np.abs(built['q_start'] - q_seed)):.2e} rad")
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


def _tool_point_velocity(gp8: GP8, q, qd, tool_offset: float):
    """현재 suction_tool 원점의 base_link 기준 위치/선속도를 반환한다.

    ``gp8.jacobian``은 space twist ``[omega, v]``를 반환하므로 world 점
    ``p``의 선속도는 ``v + omega x p``다. ``tool_offset``이 있으면 동일한
    rigid body 위의 offset 점을 사용한다.
    """
    q = np.asarray(q, dtype=float)
    qd = np.asarray(qd, dtype=float)
    T = gp8.forward_kinematics(q)
    point = T[:3, 3] + T[:3, 0] * float(tool_offset)
    twist = gp8.jacobian(q) @ qd
    velocity = twist[3:] + np.cross(twist[:3], point)
    return point, velocity


def _build_rviz_preview(gp8: GP8, args, z: float, move_vel_limits,
                        throw_vel_limits, accel_limits):
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
        q_seed=lift_q, vel_limits=throw_vel_limits, accel_limits=accel_limits,
        preferred_distance=args.release_distance,
        preferred_z_offset=args.release_z_offset,
        tool_offset=args.tool_offset,
    )

    M1 = np.asarray(move_vel_limits, dtype=float) * args.vel_scale
    M2 = M1 * JOINT_ACCEL_RATIO
    zero = np.zeros_like(M1)
    pick_lift_traj, pick_lift_vel, pick_lift_ts = trajectory(
        grasp_q, zero, lift_q, zero, M1, M2, hertz=100.0,
    )

    lift_pos = np.array([args.x, args.y, z + args.lift], dtype=float)
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
        (built["traj"], built["ts"]),
        (ret_traj, ret_ts),
        (close_traj, close_ts),
    ])
    preview_vel, velocity_ts = _concat_joint_segments([
        (pick_lift_vel, pick_lift_ts),
        (built["vel"], built["ts"]),
        (ret_velj, ret_ts),
        (close_velj, close_ts),
    ])
    if not np.allclose(preview_ts, velocity_ts, atol=1e-12, rtol=0.0):
        raise ValueError("RViz preview position/velocity time base가 일치하지 않습니다")
    preview_traj[:, -1] = preview_traj[:, 0]  # modulo 반복 경계를 수치적으로도 정확히 닫음
    preview_vel[:, -1] = preview_vel[:, 0]
    tool_path = _tool_path_from_joint_traj(gp8, preview_traj, args.tool_offset)

    return dict(
        plan=plan, built=built,
        grasp_xyz=grasp_xyz, lift_pos=lift_pos,
        preview_traj=preview_traj, preview_vel=preview_vel,
        preview_ts=preview_ts,
        tool_path=tool_path,
        ee_path=np.vstack([
            lift_pos[None, :], plan["release_pos"][None, :],
            plan["end_pos"][None, :],
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


def rviz_preview(gp8: GP8, args, z: float, move_vel_limits,
                 throw_vel_limits, accel_limits) -> None:
    """RViz에서 marker/path와 RobotModel 애니메이션으로 throw 계획을 미리 본다."""
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Point
    from sensor_msgs.msg import JointState
    from visualization_msgs.msg import Marker, MarkerArray

    preview = _build_rviz_preview(
        gp8, args, z, move_vel_limits, throw_vel_limits, accel_limits,
    )
    print_throw_plan(preview["plan"], preview["built"], args.release_lead)

    rclpy.init()
    node = Node("suction_lift_debug_rviz_preview")
    pub_js = node.create_publisher(JointState, "/joint_states", 10)
    pub_js_urdf = node.create_publisher(JointState, "/joint_states_urdf", 10)
    pub_markers = node.create_publisher(
        MarkerArray, "/suction_lift_debug/markers", 10,
    )
    pub_path = node.create_publisher(
        __import__("nav_msgs.msg", fromlist=["Path"]).Path,
        "/suction_lift_debug/path", 10,
    )

    q = preview["preview_traj"]
    qd = preview["preview_vel"]
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
    print("  TCP velocity : green arrow + live m/s text at suction_tool")
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
        qd_now = _interpolate_joint_state(qd, ts, t)

        msg = JointState()
        msg.header.stamp = now
        msg.name = JOINT_NAMES
        msg.position = [float(v) for v in q_now]
        pub_js.publish(msg)
        pub_js_urdf.publish(msg)

        tool_point, tool_velocity = _tool_point_velocity(
            gp8, q_now, qd_now, args.tool_offset,
        )
        speed = float(np.linalg.norm(tool_velocity))

        def point(xyz):
            p = Point()
            p.x, p.y, p.z = map(float, xyz)
            return p

        arrow = Marker()
        arrow.header.frame_id = frame_id
        arrow.header.stamp = now
        arrow.ns = "tcp_velocity_live"
        arrow.id = 20
        arrow.type = Marker.ARROW
        arrow.pose.orientation.w = 1.0
        if speed > 1e-4:
            arrow.action = Marker.ADD
            arrow.points = [
                point(tool_point),
                point(
                    tool_point
                    + RVIZ_TCP_VELOCITY_VECTOR_TIME * tool_velocity
                ),
            ]
            arrow.scale.x = 0.014
            arrow.scale.y = 0.030
            arrow.scale.z = 0.040
            arrow.color.r = 0.15
            arrow.color.g = 1.0
            arrow.color.b = 0.20
            arrow.color.a = 0.95
        else:
            arrow.action = Marker.DELETE

        label = Marker()
        label.header.frame_id = frame_id
        label.header.stamp = now
        label.ns = "tcp_speed_live"
        label.id = 21
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position = point(tool_point + np.array([0.0, 0.0, 0.075]))
        label.pose.orientation.w = 1.0
        label.scale.z = 0.045
        label.color.r = 0.85
        label.color.g = 1.0
        label.color.b = 0.85
        label.color.a = 1.0
        label.text = f"TCP {speed:.2f} m/s"
        pub_markers.publish(MarkerArray(markers=[arrow, label]))

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
    parser.add_argument(
        "--axis-increment-factor", type=float,
        default=AXIS_INCREMENT_FACTOR_DEFAULT,
        help="YRC external-increment 속도 factor (기본 1.0; bringup과 일치시킬 것)",
    )
    parser.add_argument(
        "--axis-acceleration-factor", type=float,
        default=AXIS_ACCELERATION_FACTOR_DEFAULT,
        help="YRC external-increment 가속도 factor (기본 0.02; bringup과 일치시킬 것)",
    )
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
    try:
        throw_vel_limits, accel_limits = _rt_motion_limits(
            gp8, args.axis_increment_factor, args.axis_acceleration_factor,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(
        "YRC RT limits: "
        f"increment_factor={args.axis_increment_factor:.3f}, "
        f"acceleration_factor={args.axis_acceleration_factor:.3f}\n"
        f"  velocity     ={np.round(throw_vel_limits, 3).tolist()} rad/s\n"
        f"  acceleration ={np.round(accel_limits, 3).tolist()} rad/s²"
    )
    move_vel_limits = np.asarray(gp8.velocity_limits, dtype=float)

    if args.plan_only:
        plan_only(
            gp8, args, z, move_vel_limits, throw_vel_limits, accel_limits,
        )
        return

    if args.rviz_preview:
        rviz_preview(
            gp8, args, z, move_vel_limits, throw_vel_limits, accel_limits,
        )
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

        M1 = move_vel_limits * args.vel_scale
        M2 = M1 * JOINT_ACCEL_RATIO

        print("\n⚠️  실제 로봇 모션입니다. 컨베이어 정지 + 주변 공간 확보 확인.")
        while True:
            run_cycle(
                ctrl, gp8, args, z, M1, M2, cfg.TRAJ_HZ,
                move_vel_limits, throw_vel_limits, accel_limits,
            )
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

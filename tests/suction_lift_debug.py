"""석션 단독 디버그 — 컨베이어 정지 상태, 지정 지점에서 pick + toy 포물선 throw.

기본 사이클: 지정한 베이스 좌표 (x, y) 위 석션 대기 높이(Config.GRASP_Z —
throw_skill 이 ambush 대기에 쓰는 그 그랩 높이)로 내려가 파킹 → 석션 ON(기본
1초 홀드, 진공 형성) → 10 cm 리프트. 이후 선택적으로 **toy 포물선 throw**:

  * 포물선: 그랩점→bin (--bin-x/--bin-y, bin 높이 = 그랩 높이 + --bin-z-offset)
    직선에서
    그랩점 60% 지점 P 를 시작점으로, 양끝 접선 45° 인 대칭 포물선을 bin 까지
    구성 (비행거리 R = 0.4 * 그랩-bin 거리).
  * **release = P 와 bin 의 3:7 내분점** (P→bin 기준 30%, pick→bin 기준 72%):
    release 속도는 g=9.81 탄도 시간매개변수화가 주는 그 지점의 미분값:
    v = (vx·û_xy, vx*(1 - 2s/R)),
    vx = sqrt(gR/2). 이 release 상태(위치+속도)가 유일한 constraint 이고,
    그랩→release 까지의 경로는 자유다. 물체는 나머지 28% 거리를 날아
    bin 에 떨어진다.
  * release 지점은 tool-down 자세로는 도달 범위 밖이므로, 스윙 구간은 툴 축을
    어깨→release 방향으로 기울인 **고정 자세**(ω=0 → 물체 CoM 속도 = EE 속도)
    로 잡는다. lift → 와인드업 시작점 이동에서 자세를 slerp 로 서서히 돌린다
    (컵이 물체를 옆으로 문 채 스윙 — 가벼운 물체 가정).
  * 스윙 = release 접선 위 직선 스트로크: 접선 후방으로 바닥 클리어런스
    (z >= GRASP_Z+5mm, 상한 25cm)가 허용하는 와인드업 → 등가속으로 release
    knot 에서 정확히 v 도달 → 등감속 팔로스루. release knot 에는 자코비안으로
    계산한 정확한 관절속도를 실어 스트림의 Cubic-Hermite 리샘플이 설계 속도를
    재현하게 한다. 석션 off 는 send_trajectory_queue_with_timed_release 로
    release knot 에 동기.

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
  --bin-x/--bin-y  bin XY 위치 [m] (기본 1.0, 0.0)
  --bin-z-offset   bin 목표 높이 = 그랩 높이 + offset [m] (기본 0.10)
  --tool-offset     gp8.py EE/TCP 위에 추가 적용할 offset [m] (기본 0.0;
                   일반적으로 사용하지 않음)
  --release-lead   석션 off 를 release knot 보다 이만큼 일찍 발화 [s] (기본 0.0;
                   IO/배기 지연 보정용, 음수 = 늦게)
  --rviz-preview    로봇에 명령을 보내지 않고 RViz용 marker/path/joint_states publish

⚠️  throw 는 실제 고속 스윙(기본 ~1.5 m/s)입니다. bin 지점에 통/마커를 두고
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

# ---- toy 포물선 throw 파라미터 (사용자 확정 사양) ----
G_ACCEL = 9.81
P_RATIO_FROM_GRASP = 0.6   # 포물선 시작점 P = 그랩→bin 의 60% 지점 (비행거리 R = 0.4*D)
RELEASE_FRACTION_FROM_P = 0.3  # release = P→bin 의 30% 지점 = P:bin 3:7 내분점
FLOOR_CLEARANCE = 0.005    # 스윙 전체에서 컵 z >= GRASP_Z + 이 값 [m]
RUNUP_MAX = 0.25           # 와인드업(런업) 길이 상한 [m]
A_DEC_MIN = 12.0           # 팔로스루 최소 감속도 [m/s^2]
FOLLOWTHROUGH_MAX = 0.03   # release 이후 팔로스루 길이 상한 [m]
THROW_DT = 0.01            # throw 구간 knot 간격 [s] (스트림이 4ms 로 리샘플)
PREMOVE_SPEED = 0.20       # 와인드업 위치로 하강/복귀 카르테시안 이동 속도 [m/s]
PREMOVE_ACCEL = 1.0        # 그 가속도 [m/s^2]
PREMOVE_DT = 0.02          # 그 knot 간격 [s]
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
def plan_parabola_throw(grasp_xyz, bin_xy,
                        bin_z_offset: float = BIN_Z_OFFSET_DEFAULT) -> dict:
    """그랩점/bin 에서 포물선과 release 상태(위치·자세·속도)를 계산.

    포물선의 로컬 높이는 P 에서 45°로 시작하고 bin 높이(z_g+bin_z_offset)를
    통과하도록 z(s)=z_g+s+a*s² 로 둔다 (s: P 로부터 수평거리).
    release 는 P→bin 기준 30% 지점(P:bin = 3:7 내분점)이다.
    탄도 시간매개변수화(g)로 수평속도 vx = sqrt(-g/(2a)) (상수),
    수직속도 vz = vx * (1 - 2s/R).

    release 지점은 tool-down 도달 범위 밖이라 스윙용 툴 자세는 어깨→release
    방향으로 기울인다 (throw_R). 도달 자체가 불가능하면 ValueError.
    """
    G = np.asarray(grasp_xyz, dtype=float)
    z_g = float(G[2])
    d_xy = np.array([bin_xy[0] - G[0], bin_xy[1] - G[1]])
    D = float(np.hypot(*d_xy))
    if D < 0.05:
        raise ValueError(f"bin 이 그랩점과 너무 가깝습니다 (D={D:.3f} m)")
    u_xy = d_xy / D

    if not (0.0 < RELEASE_FRACTION_FROM_P < 1.0):
        raise ValueError(
            "release 비율은 P 보다 뒤, bin 보다 앞이어야 합니다 "
            f"(P→bin fraction={RELEASE_FRACTION_FROM_P:.2f})"
        )

    z_bin = z_g + float(bin_z_offset)
    R_fly = (1.0 - P_RATIO_FROM_GRASP) * D      # 포물선 수평 비행거리
    parabola_a = (z_bin - z_g - R_fly) / (R_fly ** 2)
    if parabola_a >= 0.0:
        raise ValueError(
            f"bin 이 너무 높아 현재 45° 시작 포물선으로는 탄도 계획 불가 "
            f"(bin_z_offset={bin_z_offset:.3f}, R={R_fly:.3f})"
        )
    s_rel = (P_RATIO_FROM_GRASP + RELEASE_FRACTION_FROM_P * (1.0 - P_RATIO_FROM_GRASP)) * D
    s_rel_from_p = s_rel - P_RATIO_FROM_GRASP * D
    f_rel = s_rel_from_p / R_fly
    z_rel = z_g + s_rel_from_p + parabola_a * s_rel_from_p ** 2
    release_pos = np.array([G[0] + u_xy[0] * s_rel, G[1] + u_xy[1] * s_rel, z_rel])

    vx = float(np.sqrt(-G_ACCEL / (2.0 * parabola_a)))  # 수평속도 (상수)
    slope_rel = 1.0 + 2.0 * parabola_a * s_rel_from_p
    vz_rel = vx * slope_rel
    v_rel = np.array([vx * u_xy[0], vx * u_xy[1], vz_rel])
    speed = float(np.linalg.norm(v_rel))
    u3 = v_rel / speed                          # release 접선 단위벡터

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
    # 스윙용 고정 툴 축: 어깨→release 방향 (손목중심을 최대한 어깨쪽으로 당김).
    tool_axis = r_vec / r_dist
    tilt_deg = float(np.degrees(np.arccos(np.clip(-tool_axis[2], -1.0, 1.0))))
    # EE 프레임: x_ee = 툴 축, y_ee = 던짐 수직평면의 법선, z_ee = x×y.
    n_plane = np.array([-u_xy[1], u_xy[0], 0.0])
    x_ee = tool_axis
    y_ee = n_plane - x_ee * float(x_ee @ n_plane)   # 평면법선을 x 에 직교화
    y_ee /= np.linalg.norm(y_ee)
    throw_R = np.column_stack([x_ee, y_ee, np.cross(x_ee, y_ee)])

    # 와인드업: release 접선 후방 직선, 바닥(z_g + clearance) 직전까지 (상한 있음).
    # release 가 포물선 정점이면 접선은 수평(u3[2] ~= 0)이므로 바닥 제약 대신
    # 상한 RUNUP_MAX 를 사용한다.
    if abs(u3[2]) < 1e-9:
        L_run = RUNUP_MAX
    else:
        L_run = min((z_rel - (z_g + FLOOR_CLEARANCE)) / u3[2], RUNUP_MAX)
    if L_run < 0.03:
        raise ValueError(
            f"와인드업 길이가 너무 짧습니다 (L={L_run * 100:.1f} cm)"
        )
    a_acc = speed ** 2 / (2.0 * L_run)
    L_dec_free = speed ** 2 / (2.0 * A_DEC_MIN)
    L_dec = min(L_dec_free, FOLLOWTHROUGH_MAX)
    a_dec = speed ** 2 / (2.0 * L_dec)
    start_pos = release_pos - L_run * u3
    end_pos = release_pos + L_dec * u3

    # 탄도 낙하점 검산: release 상태에서 bin 높이까지 비행 → bin 이어야 함.
    t_fly = (R_fly - s_rel_from_p) / vx
    landing_xy = release_pos[:2] + v_rel[:2] * t_fly
    landing_z = release_pos[2] + v_rel[2] * t_fly - 0.5 * G_ACCEL * t_fly ** 2
    landing_err = float(np.linalg.norm(np.array([
        landing_xy[0] - bin_xy[0], landing_xy[1] - bin_xy[1], landing_z - z_bin,
    ])))

    return dict(
        grasp=G, bin_xy=np.asarray(bin_xy, dtype=float),
        bin_xyz=np.array([bin_xy[0], bin_xy[1], z_bin], dtype=float),
        z_g=z_g, z_bin=z_bin, bin_z_offset=float(bin_z_offset),
        D=D, R=R_fly, P_xy=G[:2] + u_xy * (P_RATIO_FROM_GRASP * D),
        release_ratio=s_rel / D, release_fraction_from_p=f_rel,
        release_pos=release_pos, v_rel=v_rel, speed=speed, u3=u3,
        vx=vx, apex=float(-1.0 / (4.0 * parabola_a)),
        parabola_a=parabola_a, slope_rel=slope_rel,
        throw_R=throw_R, tilt_deg=tilt_deg, r_extension=r_dist / reach_limit,
        L_run=L_run, a_acc=a_acc, L_dec=L_dec, a_dec=a_dec,
        start_pos=start_pos, end_pos=end_pos,
        t_fly=t_fly, landing=np.array([landing_xy[0], landing_xy[1], landing_z]),
        landing_err=landing_err,
    )


def _throw_profile_knots(start, release, end, speed, a_acc, a_dec, dt):
    """start→release 등가속(release 에서 정확히 speed), release→end 등감속의
    비대칭 직선 스트로크를 knot 샘플. Returns (pos, vel, t, i_release)."""
    start = np.asarray(start, dtype=float)
    release = np.asarray(release, dtype=float)
    end = np.asarray(end, dtype=float)
    u = (release - start) / np.linalg.norm(release - start)

    t_a = speed / a_acc
    n_a = max(3, int(round(t_a / dt)) + 1)
    ta = np.linspace(0.0, t_a, n_a)
    sa = 0.5 * a_acc * ta ** 2
    va = a_acc * ta

    t_d = speed / a_dec
    n_d = max(3, int(round(t_d / dt)) + 1)
    td = np.linspace(0.0, t_d, n_d)[1:]
    sd = speed * td - 0.5 * a_dec * td ** 2
    vd = speed - a_dec * td

    L_run = float(np.linalg.norm(release - start))
    t = np.concatenate([ta, t_a + td])
    s = np.concatenate([sa, L_run + sd])
    ds = np.concatenate([va, vd])
    pos = start[None, :] + s[:, None] * u[None, :]
    vel = ds[:, None] * u[None, :]
    return pos, vel, t, n_a - 1


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
                     tool_offset: float = TOOL_OFFSET_DEFAULT):
    """카르테시안 knot 열 → 관절 궤적 (6,N)/(6,N)/(N,).

    R_fixed: 전 knot 동일 자세(ω=0) → 관절속도는 공간 자코비안 (ω,v) 로
    q̇ = J⁻¹·[0; ṗ] 정확 산출 (상위 3행 = ω, 수치 검증 완료).
    R_seq: knot 별 자세 (저속 slerp 이동용) → 관절속도는 시간 FD 근사.
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
        if R_fixed is not None:
            J = gp8.jacobian(q)
            twist = np.concatenate([np.zeros(3), vel[k]])
            try:
                velj[:, k] = np.linalg.solve(J, twist)
            except np.linalg.LinAlgError:
                velj[:, k] = np.linalg.lstsq(J, twist, rcond=None)[0]
        q_prev = q
    if R_fixed is None:
        velj = np.gradient(traj, np.asarray(t, dtype=float), axis=1)
        velj[:, 0] = 0.0
        velj[:, -1] = 0.0
    return traj, velj, np.asarray(t, dtype=float)


def build_throw(gp8: GP8, plan: dict, q_seed, vel_limits,
                tool_offset: float = TOOL_OFFSET_DEFAULT):
    """throw 스트로크(와인드업+release+팔로스루) 관절 궤적 + 안전 마진 리포트."""
    pos, vel, t, i_rel = _throw_profile_knots(
        plan["start_pos"], plan["release_pos"], plan["end_pos"],
        plan["speed"], plan["a_acc"], plan["a_dec"], THROW_DT,
    )
    traj, velj, ts = build_joint_traj(
        gp8, pos, vel, t, q_seed, R_fixed=plan["throw_R"],
        tool_offset=tool_offset,
    )
    vel_ratio = float(np.max(np.abs(velj) / np.asarray(vel_limits)[:, None]))
    if vel_ratio > 1.0:
        raise ValueError(
            f"관절속도 한계 초과 (max|q̇|/limit = {vel_ratio * 100:.0f}%) — "
            f"bin 을 가깝게 해 release 속도를 낮추세요"
        )
    # release knot 의 실제 EE 속도 검산 (자코비안 왕복)
    Jr = gp8.jacobian(traj[:, i_rel])
    v_chk = (Jr @ velj[:, i_rel])[3:]
    v_err = float(np.linalg.norm(v_chk - plan["v_rel"]))
    return dict(
        traj=traj, vel=velj, ts=ts, release_idx=i_rel,
        q_start=traj[:, 0].copy(), q_end=traj[:, -1].copy(),
        vel_ratio=vel_ratio, v_err=v_err,
        z_min=float(pos[:, 2].min()), duration=float(ts[-1]),
        n_knots=pos.shape[0],
    )


def print_throw_plan(plan: dict, built: dict, release_lead: float) -> None:
    p, b = plan, built
    rp, v = p["release_pos"], p["v_rel"]
    ang = float(np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1]))))
    print("\n=== Toy Parabola Throw Plan ===")
    print(f"  grasp→bin : D={p['D']:.3f} m, "
          f"bin=({p['bin_xyz'][0]:+.3f}, {p['bin_xyz'][1]:+.3f}, {p['bin_xyz'][2]:+.3f}) "
          f"(grasp z +{p['bin_z_offset'] * 100:.1f} cm)")
    print(f"  포물선     : P=({p['P_xy'][0]:+.3f}, {p['P_xy'][1]:+.3f}), R={p['R']:.3f} m, "
          f"apex +{p['apex'] * 100:.1f} cm (양끝 45°)")
    print(f"  release   : pick→bin {p['release_ratio'] * 100:.0f}% 지점 "
          f"(P→bin {p['release_fraction_from_p'] * 100:.0f}%) "
          f"pos=({rp[0]:+.3f}, {rp[1]:+.3f}, {rp[2]:+.3f}) m "
          f"(그랩 높이 +{(rp[2] - p['z_g']) * 100:.1f} cm, 신전율 {p['r_extension'] * 100:.0f}%)")
    print(f"              v=({v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f}) m/s, "
          f"|v|={p['speed']:.3f} m/s, {ang:.2f}°  (툴 기울임 {p['tilt_deg']:.0f}°)")
    print(f"  낙하 검산  : bin 오차 {p['landing_err'] * 1000:.1f} mm (release 후 비행 {p['t_fly']:.3f}s)")
    print(f"  스트로크   : 와인드업 {p['L_run'] * 100:.1f} cm (a={p['a_acc']:.1f} m/s²) + "
          f"팔로스루 {p['L_dec'] * 100:.1f} cm (a={p['a_dec']:.1f} m/s²), "
          f"총 {b['duration']:.3f}s / {b['n_knots']} knots, z_min={b['z_min']:.3f} m")
    print(f"  release knot {b['release_idx']} (lead {release_lead:+.3f}s), "
          f"관절속도 마진 max|q̇|/limit = {b['vel_ratio'] * 100:.1f}%, "
          f"release EE 속도오차 {b['v_err'] * 1000:.2f} mm/s")
    if b["vel_ratio"] > 0.8:
        print("  ⚠️ 관절속도 마진이 80% 를 넘습니다 — 저속 팩터 bringup 에선 실패할 수 있음")


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
    plan = plan_parabola_throw(
        grasp_xyz, (args.bin_x, args.bin_y),
        bin_z_offset=args.bin_z_offset,
    )
    built = build_throw(
        gp8, plan, q_seed=lift_q, vel_limits=vel_limits,
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
        plan = plan_parabola_throw(
            grasp_xyz, (args.bin_x, args.bin_y),
            bin_z_offset=args.bin_z_offset,
        )
        built = build_throw(
            gp8, plan, q_seed=q_seed, vel_limits=vel_limits,
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

    plan = plan_parabola_throw(
        grasp_xyz, (args.bin_x, args.bin_y),
        bin_z_offset=args.bin_z_offset,
    )
    built = build_throw(
        gp8, plan, q_seed=lift_q, vel_limits=vel_limits,
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

    preview_traj, preview_ts = _concat_joint_segments([
        (pick_lift_traj, pick_lift_ts),
        (pre_traj, pre_ts),
        (built["traj"], built["ts"]),
        (ret_traj, ret_ts),
    ])
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
    started = time.monotonic()
    frame_id = args.frame_id

    print("\nRViz preview publishing:")
    print("  joint_states : /joint_states and /joint_states_urdf")
    print("  markers      : /suction_lift_debug/markers")
    print("  path         : /suction_lift_debug/path")
    print(f"  tool frame   : URDF {TOOL_FRAME} (MuJoCo grip_site/TCP); extra offset {args.tool_offset:+.3f} m")
    print(f"  fixed frame  : {frame_id}")
    print("Ctrl-C 로 종료. 실제 로봇 명령은 보내지 않습니다.")

    def tick():
        now = node.get_clock().now().to_msg()
        t = (time.monotonic() - started) % duration
        idx = int(np.searchsorted(ts, t, side="right") - 1)
        idx = max(0, min(idx, q.shape[1] - 1))

        msg = JointState()
        msg.header.stamp = now
        msg.name = JOINT_NAMES
        msg.position = [float(v) for v in q[:, idx]]
        pub_js.publish(msg)
        pub_js_urdf.publish(msg)

        # late RViz subscribers를 위해 marker/path도 계속 재발행.
        _publish_rviz_static(pub_markers, pub_path, preview, frame_id, now)

    node.create_timer(1.0 / float(args.preview_rate), tick)
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
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


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
    parser.add_argument("--bin-x", type=float, default=1.0,
                        help="throw bin X [m] (기본 1.0)")
    parser.add_argument("--bin-y", type=float, default=0.0,
                        help="throw bin Y [m] (기본 0.0)")
    parser.add_argument("--bin-z-offset", type=float, default=BIN_Z_OFFSET_DEFAULT,
                        help="bin 목표 높이 = grasp Z + offset [m] (기본 0.10)")
    parser.add_argument("--tool-offset", type=float, default=TOOL_OFFSET_DEFAULT,
                        help="gp8.py/MuJoCo TCP보다 +X 앞의 추가 offset [m] (기본 0.0)")
    parser.add_argument("--release-lead", type=float, default=0.0,
                        help="석션 off 를 release 보다 이만큼 일찍 [s] (음수=늦게)")
    parser.add_argument("--plan-only", action="store_true",
                        help="로봇 없이 throw 계획 검증만 출력")
    parser.add_argument("--rviz-preview", action="store_true",
                        help="로봇에 명령을 보내지 않고 RViz용 marker/path/joint_states publish")
    parser.add_argument("--preview-rate", type=float, default=30.0,
                        help="RViz preview joint_states publish rate [Hz] (기본 30)")
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

    if float(np.hypot(args.x, args.y)) > cfg.MAX_REACH:
        print(f"지점 ({args.x:+.3f}, {args.y:+.3f}) 이 MAX_REACH {cfg.MAX_REACH} m 밖입니다.")
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
                if float(np.hypot(args.x, args.y)) > cfg.MAX_REACH:
                    print(f"  ({args.x:+.3f}, {args.y:+.3f}) 은 MAX_REACH 밖 — 종료합니다.")
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

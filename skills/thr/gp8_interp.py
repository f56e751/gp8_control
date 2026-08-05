"""
Yaskawa 컨트롤러(YRC1000/MotoROS)의 공식 궤적 보간을 시뮬에 적용하는 모듈.

실기 파이프라인 (THR/ros2/robust_throw_skill.py 참고):
  계획 궤적 → **TRAJ_HZ로 샘플한 waypoint(위치+속도)** 를 컨트롤러 큐로 전송
  (`send_trajectory_queue_with_timed_release`) → 컨트롤러가 waypoint 사이를
  자체 보간 주기마다 증분 이동으로 채운다.

컨트롤러 보간식은 MotoROS 드라이버(MotionServer.c,
`Ros_MotionServer_JointTrajDataToIncQueue`)에 있는 그대로다 — 관절별로
두 waypoint (p₀,v₀) → (p₁,v₁), 구간 T에 대해

    accCoef1 = 6(p₁−p₀)/T²  − 2(v₁ + 2v₀)/T
    accCoef2 = −12(p₁−p₀)/T³ + 6(v₁ + v₀)/T²
    pos(τ)   = p₀ + v₀·τ + accCoef1·τ²/2 + accCoef2·τ³/6

즉 양 끝의 위치와 속도를 모두 만족하는 **3차 Hermite**다 (대입하면 pos(0)=p₀,
pos(T)=p₁, pos'(0)=v₀, pos'(T)=v₁이 정확히 성립 — 이 모듈의 테스트가 검증한다).
속도는 그 해석적 미분 pos'(τ) = v₀ + accCoef1·τ + accCoef2·τ²/2 를 쓴다.

왜 이게 결과를 바꾸는가: 계획기는 연속 궤적(NLP은 B-spline, DT는 10 Hz 스텝)을
내지만 실기 로봇이 실제로 그리는 것은 '드문드문한 waypoint를 3차로 이은 곡선'
이다. 던지기처럼 릴리즈 순간의 속도가 곧 착탄을 결정하는 동작에서는 이 차이가
그대로 오차가 된다. 보간 전후를 비교하면 그 크기를 볼 수 있다.

보간 주기(INTERP_PERIOD)는 YRC1000 계열의 표준값 4 ms를 쓴다. waypoint 전송
주기(TRAJ_HZ)는 실기 설정값(gp8_control cfg.TRAJ_HZ)이 이 저장소에 없어
파라미터로 두었다 — 낮출수록 왜곡이 커지므로 감도 스윕의 축이기도 하다.
"""

import numpy as np

INTERP_PERIOD = 0.004     # s — 컨트롤러 보간 주기 (YRC1000 표준 4 ms)
TRAJ_HZ = 50.0            # Hz — 컨트롤러로 보내는 waypoint 샘플 주기


def hermite_coefs(p0, p1, v0, v1, T):
    """MotoROS `Ros_MotionServer_JointTrajDataToIncQueue`의 가속 계수."""
    T = float(T)
    a1 = 6.0 * (p1 - p0) / T ** 2 - 2.0 * (v1 + 2.0 * v0) / T
    a2 = -12.0 * (p1 - p0) / T ** 3 + 6.0 * (v1 + v0) / T ** 2
    return a1, a2


def hermite_eval(p0, v0, a1, a2, tau):
    """위치/속도 (MotoROS 식과 그 해석적 미분). tau는 (m,) 배열 가능."""
    tau = np.asarray(tau, float)[:, None]
    pos = p0 + v0 * tau + a1 * tau ** 2 / 2.0 + a2 * tau ** 3 / 6.0
    vel = v0 + a1 * tau + a2 * tau ** 2 / 2.0
    return pos, vel


def sample_waypoints(ts, Q, Qd, traj_hz=TRAJ_HZ):
    """계획 궤적을 컨트롤러로 보낼 waypoint(위치+속도)로 샘플.
    실기의 `t_nlp = np.arange(0, t_f + dt/2, dt)`와 같은 규약이며, 궤적 끝점이
    격자에 안 걸리면 마지막 점을 덧붙여 종단 상태(정지)를 잃지 않는다."""
    ts = np.asarray(ts, float)
    dt = 1.0 / float(traj_hz)
    t_wp = np.arange(0.0, ts[-1] + dt / 2, dt)
    if t_wp[-1] < ts[-1] - 1e-12:
        t_wp = np.append(t_wp, ts[-1])
    n = Q.shape[1]
    P = np.column_stack([np.interp(t_wp, ts, Q[:, j]) for j in range(n)])
    V = np.column_stack([np.interp(t_wp, ts, Qd[:, j]) for j in range(n)])
    return t_wp, P, V


def controller_track(ts, Q, Qd=None, traj_hz=TRAJ_HZ, period=INTERP_PERIOD,
                     out_dt=None):
    """계획 궤적 → 컨트롤러가 실제로 실행하는 궤적.

    ① traj_hz로 waypoint(위치+속도) 추출 → ② 각 구간을 MotoROS 3차 Hermite로
    period 간격 전개 → ③ out_dt 그리드로 재샘플 (시뮬 스텝과 맞추기 위함).
    리턴: (ts_out, Q_out, Qd_out)."""
    ts = np.asarray(ts, float)
    Q = np.asarray(Q, float)
    Qd = np.gradient(Q, ts, axis=0) if Qd is None else np.asarray(Qd, float)
    t_wp, P, V = sample_waypoints(ts, Q, Qd, traj_hz)

    t_all, q_all, v_all = [], [], []
    for k in range(len(t_wp) - 1):
        T = t_wp[k + 1] - t_wp[k]
        if T <= 1e-12:
            continue
        a1, a2 = hermite_coefs(P[k], P[k + 1], V[k], V[k + 1], T)
        # 구간 내부 격자 (끝점은 다음 구간의 시작이므로 제외 — 중복 방지)
        m = max(1, int(np.ceil(T / period)))
        tau = np.arange(m) * (T / m)
        pos, vel = hermite_eval(P[k], V[k], a1, a2, tau)
        t_all.append(t_wp[k] + tau)
        q_all.append(pos)
        v_all.append(vel)
    # 마지막 waypoint를 종단으로 덧붙임 (정지 상태 보존)
    t_all.append(np.array([t_wp[-1]]))
    q_all.append(P[-1][None, :])
    v_all.append(V[-1][None, :])

    t_ctrl = np.concatenate(t_all)
    q_ctrl = np.vstack(q_all)
    v_ctrl = np.vstack(v_all)

    if out_dt is None:
        return t_ctrl, q_ctrl, v_ctrl
    # 시뮬 그리드로 재샘플: 서보는 4 ms 지령 사이를 매끄럽게 추종하므로 선형 보간
    t_out = np.arange(0.0, t_ctrl[-1] + out_dt / 2, out_dt)
    n = q_ctrl.shape[1]
    Q_out = np.column_stack([np.interp(t_out, t_ctrl, q_ctrl[:, j])
                             for j in range(n)])
    V_out = np.column_stack([np.interp(t_out, t_ctrl, v_ctrl[:, j])
                             for j in range(n)])
    return t_out, Q_out, V_out


def tracking_error(ts, Q, Qd=None, traj_hz=TRAJ_HZ, period=INTERP_PERIOD):
    """계획 궤적 대비 컨트롤러 실행 궤적의 관절 오차 (진단용).
    리턴: dict(max_rad, rms_rad, max_deg, n_waypoints)."""
    ts = np.asarray(ts, float)
    Q = np.asarray(Q, float)
    t_c, Q_c, _ = controller_track(ts, Q, Qd, traj_hz, period, out_dt=None)
    Q_ref = np.column_stack([np.interp(t_c, ts, Q[:, j])
                             for j in range(Q.shape[1])])
    err = np.abs(Q_c - Q_ref)
    dt = 1.0 / float(traj_hz)
    return dict(max_rad=float(err.max()), rms_rad=float(np.sqrt((err ** 2).mean())),
                max_deg=float(np.rad2deg(err.max())),
                n_waypoints=int(np.ceil(ts[-1] / dt)) + 1)

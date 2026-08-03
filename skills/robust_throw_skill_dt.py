"""Throw skill: NLP planner 대신 **Thr_DT Decision Transformer** 가 던지기 궤적을 만든다.

`robust_throw_skill.RobustThrowSkill` (CasADi/IPOPT NLP) 의 `_dt` 변형이다.
픽(hover → suction → press → 벨트추종)·체인·안전 게이트·디스패치는 부모 것을
그대로 쓰고, **계획 단계만** 교체한다:

    RobustThrowSkill : solve_throw_nlp(target)          → B-spline 제어점 P, t*, 릴리즈 윈도우
    DtThrowSkill     : DT rollout(d_g), 10 Hz 자기회귀   → 관절궤적 + 릴리즈 타임스텝

논문/재구현: M. Monastirsky, O. Azulay, A. Sintov, "Learning to Throw With a
Handful of Samples Using Decision Transformers," IEEE RA-L 8(2):576-583, 2023
(doi 10.1109/LRA.2022.3229266). 재구현 = /PublicSSD/ryugaeun/Thr_DT,
추론 코드/가중치는 `skills/thr_dt/` 에 벤더링돼 있다 (거기 README 참고).

──────────────────────────────────────────────────────────────────────────────
DT 가 내놓는 것 (논문 §4.1/§5.1)
──────────────────────────────────────────────────────────────────────────────
* 상태 ŝ_t = (θ₂, θ₃, θ₅, θ_gr ‖ d_g), 행동 a_t = (ω₂, ω₃, ω₅, a_gr), 10 Hz, T ≤ 10 스텝.
* 목표 리턴 R̂ = 1 로 조건화, 목표는 **거리 d_g 하나** — 방향은 J1(S)이 해석적으로 잡는다.
* 위치 명령 θ_t = θ_{t−1} + ω_t/f (식 5.1), 관절은 1차 지연(τ=0.05 s)으로 추종.
* a_gr ≤ τ_gr 인 스텝의 **끝**에서 릴리즈, 그 순간의 tip 속도로 탄도 비행.

──────────────────────────────────────────────────────────────────────────────
각도 규약 — DT (t₂,t₃,t₅) ≡ GP8 **robot(URDF) 규약** (L,U,B)
──────────────────────────────────────────────────────────────────────────────
Thr_DT 평면 시뮬(env/throw_env.py `_fk`)과 GP8 플래너 체인(throwing.py `_CHAIN`)을
직접 비교해 확인했다. planner 규약으로는 (L,U,B) = (t₂, −t₃, −t₅) 이고,
planner→robot 변환 `_PLANNER_SIGN`=[1,1,−1,−1,−1,−1] 를 먹이면

    robot L = t₂,   robot U = t₃,   robot B = t₅      (R = T = 0, S = 조준 yaw)

즉 **부호 변환이 없다**. GP8 의 a1(4 cm)·a3(4 cm) offset 을 0 으로 놓고 tool 길이를
맞추면 두 FK 의 tip 위치가 **0.000000 m** 로 일치한다(검증 완료). 남는 차이는
Thr_DT 시뮬이 생략한 고정 형상뿐이다:

    a1 = 0.040 m, a3 = 0.040 m offset 없음
    l_tool = 0.220 m  (실제 GP8 은 d6 + tool = 0.080 + 0.220 = 0.300 m)

→ Thr_DT 가 예측하는 착지점과 실제 GP8 형상이 만드는 착지점은 다르다. 이 스킬은
**둘 다 계산해서 로그로 남긴다** (`DT-sim` vs `realGP8`). 실기 기준은 realGP8 쪽이다.

──────────────────────────────────────────────────────────────────────────────
⚠ 하드웨어 실현 가능성 — 대부분의 목표에서 DT 궤적은 GP8 에 못 들어간다
──────────────────────────────────────────────────────────────────────────────
Thr_DT 평면 시뮬레이터에는 **관절 위치 한계가 없다** (속도만 clip). 그래서 학습된
정책은 먼 목표에서 U(J3)를 −185°…−260° 까지 크게 뒤로 감는 백스윙을 쓴다. 실제
GP8 의 J3 는 [−70°, +190°] 이고, 그 자세는 TCP 를 x ≈ −0.27 m (로봇 기둥/베이스
안쪽)로 집어넣는다 — 실행하면 충돌이다.

사전 스캔 결과 (d_g 0.50→2.00 m, 0.05 간격 31개, 관절한계 + Cartesian 엔벨로프):

    weights/dt_best.pth       12/31 통과 — d_g 0.50 … 1.05 m
    weights/dt_best_k0.pth     8/31 통과 — d_g 0.50 … 0.85 m  (Thr_DT 자체 지표는 최고)

그래서 기본 가중치는 `dt_best.pth` 다 (k0 가 Thr_DT 30-goal 평가에선 더 좋지만,
GP8 에서 실행 가능한 목표 범위가 더 좁다). 통과 못 하는 목표는 **계획 단계에서
None 을 반환**하므로 `static_pick_throw_dt` 가 그 지점을 아예 집지도 않는다.

──────────────────────────────────────────────────────────────────────────────
NLP 버전과의 구조 차이
──────────────────────────────────────────────────────────────────────────────
* **시작 자세가 고정**이다. NLP 는 grasp 위치 + THROW_LIFT 에서 스윙을 시작하지만,
  DT 는 항상 학습 시의 home 자세(`SimConfig.home_pose`)에서 출발한다 — 즉 물체를
  어디서 집었는지가 던지기에 영향을 주지 않는다. lift 세그먼트는 "픽 자세 →
  DT home 자세" rest-to-rest 이동이 된다.
* **릴리즈 윈도우가 없다.** NLP 해는 t*±25 ms 어디서 놓아도 착지가 보장되지만 DT 는
  한 점이다. 밸브 jitter 가 그대로 착지 오차가 되므로, 스킬이 ±10 ms 당 착지
  변화(민감도)를 계산해 로그에 남긴다.
* **감속 꼬리를 우리가 붙인다.** DT 에피소드는 릴리즈에서 끝나고 팔은 최고속이다
  (NLP 는 v_f=0 구조적 보장). 릴리즈 상태에서 park 까지 사다리꼴 감속 세그먼트를
  만들어 이어 붙인다. 이때 M1 은 `--vel-scale` 로 줄이지 **않는다** — 진입 속도가
  M1 을 넘으면 `trajectory()` 가 그 관절 행을 0 으로 남기기 때문이다(치명적).
  seam 연속성은 dispatch 전에 명시적으로 확인한다.
* 궤적 샘플링은 시뮬의 적분 격자(100 Hz = 10 Hz × substeps 10)를 그대로 쓴다.
  스트리머가 어차피 4 ms 로 리샘플하므로 중간 다운샘플링 손실이 없다.

※ 이 파일은 실기 미검증이다 (2026-08-03 작성). `--plan-only` 로 먼저 확인할 것.
"""

from __future__ import annotations

import datetime
import math
import os
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from gp8_control.skills.robust_throw_skill import (
    MAX_TCP_Z,
    MIN_TCP_X,
    MIN_TCP_Z,
    _PLANNER_SIGN,
    RobustThrowSkill,
)
from gp8_control.trajectory.trajectory_primitive import trajectory

# throwing.py 는 skills/ 에 bare 모듈로 있다 (robust_throw_skill 이 이미 skills/ 를
# sys.path 에 올려둔다 — import 순서상 위의 import 로 보장된다).
_THR_DIR = str(Path(__file__).resolve().parent)
if _THR_DIR not in sys.path:
    sys.path.insert(0, _THR_DIR)
from throwing import G, fk_pos, launch_state  # noqa: E402

if TYPE_CHECKING:
    from gp8_control.tracking import TrackedObject


# =========================================================================
# 설정
# =========================================================================

# DT 가중치. env GP8_DT_WEIGHTS / CLI --weights 로 교체.
#   dt_best.pth        — GP8 실행 가능 목표 d_g 0.50…1.05 m (기본)
#   dt_best_k0.pth     — Thr_DT 30-goal 평가 최고(27.8 cm)지만 GP8 가능 범위 0.50…0.85 m
#   dt_finetuned_real-5.pth — sim2real 파인튜닝 데모 산출물 (교란 시뮬 기준)
DT_WEIGHTS: str = os.environ.get(
    "GP8_DT_WEIGHTS", str(Path(_THR_DIR) / "thr_dt" / "weights" / "dt_best.pth"))

# 목표 리턴 R̂ (논문 §4.3 / repo evaluate_dt.py — 성공한 던지기 = 1).
DT_TARGET_RETURN: float = float(os.environ.get("GP8_DT_TARGET_RETURN", "1.0"))

# DT 는 d_g ∈ [50, 200] cm 로 학습됐다 (논문 §5.1). 밖이면 외삽 — 거부한다.
DG_MIN: float = 0.50
DG_MAX: float = 2.00

# 착탄 게이트 (m): 실제 GP8 형상으로 예측한 착지점이 요청 목표에서 이만큼 넘게
# 벗어나면 계획을 거부. NLP 쪽 LANDING_GATE(30 mm)는 solver 가 착탄을 최적화하기
# 때문에 성립하는 값이고, DT 는 착탄을 명시적으로 풀지 않는 **정책**이라 그렇게
# 조이면 아무것도 통과하지 못한다. 기본값은 "던지기가 목표 방향으로 대충 간다"를
# 확인하는 수준(50 cm)이고, 실제 오차는 항상 로그에 찍힌다. env GP8_DT_LAND_GATE.
DT_LANDING_GATE: float = float(os.environ.get("GP8_DT_LAND_GATE", "0.50"))

# 속도 검사 기준 = 컨트롤러 실측 RT 스트림 상한(robots/gp8._RT_STREAM_VELOCITY_LIMITS,
# L/U/B = 193/259/275 °/s) 의 이 비율. 데이터시트(385/520/550)보다 훨씬 낮고, 이것이
# 실제로 명령이 잘리는 지점이다.
RT_VEL_MARGIN: float = 0.95

# ⚠ DT 아크는 **10 Hz 스텝 경계마다 속도 스파이크**가 있고, 그 값이 RT 상한을 넘는다.
# Thr_DT 시뮬의 관절은 1차 지연(τ_act = 0.05 s)으로 위치 명령을 따라가는데 명령이
# 0.1 s 앞을 가리키므로, 스텝 시작 순간의 속도가
#     q̇ = (q_cmd − q)/τ_act = ω·dt/τ_act = **2ω**
# 로 튀었다가 스텝 끝에서 2ω·e^(−dt/τ_act) ≈ 0.27ω 까지 지수적으로 떨어진다. 즉 아크
# 최대 속도는 **스텝 시작의 과도현상**이고, 실제 던지기를 결정하는 릴리즈 순간 속도는
# 그보다 훨씬 작다 (실측: 아크 peak 520°/s vs 릴리즈 |v_TCP| 0.4 m/s).
#
# 그래서 **raw peak 으로 거부하지 않는다**. 대신 컨트롤러 증분 governor 를 모사해
# (`_rate_limited`) "그래서 릴리즈 상태가 실제로 달라지는가"를 직접 본다. GP8 실측
# 상한(193/259/275 °/s for L/U/B)으로 clamp 해 본 결과, 실행 가능한 목표 전 구간에서
# 릴리즈 시점 추종 오차 **0.00°**, 착지 이동 **0.0 cm** 였다 — 스파이크가 짧아
# (스텝 앞부분 몇 ms) 위치 제어기가 그 스텝 안에서 따라잡는다.
#
# 게이트: 모사한 릴리즈 자세 오차 / 착지 이동으로 판단한다.
DT_MAX_LAG_DEG: float = float(os.environ.get("GP8_DT_MAX_LAG_DEG", "2.0"))
DT_MAX_CLAMP_SHIFT: float = float(os.environ.get("GP8_DT_MAX_CLAMP_SHIFT", "0.10"))
# 그래도 raw peak 초과 자체를 막고 싶으면 (보수적 운용) 1 로 둔다.
DT_VEL_STRICT: bool = os.environ.get("GP8_DT_VEL_STRICT", "0") not in ("0", "", "false")
STREAM_PERIOD: float = 0.004      # controllers/trajectory_controller.STREAM_DT


# =========================================================================
# DT 추론 (프로세스당 1회 로드)
# =========================================================================

_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: dict = {}


def _load_dt(weights: str, log=None):
    """(model, arm_cfg, gripper_thresh) 를 리턴. 프로세스당 가중치별 1회 로드."""
    with _MODEL_LOCK:
        hit = _MODEL_CACHE.get(weights)
        if hit is not None:
            return hit

        import torch

        # NLP 쪽 threadpoolctl(1) 과 같은 이유: BLAS/torch 가 코어 수만큼 스레드를
        # 띄우면 ros2_control 의 4 ms RT UDP 루프가 굶어 RUN_STALL → comm-loss 로
        # 간다 (2026-07-21 실기 재현). 210k 파라미터라 1스레드로도 rollout 은 ms 단위.
        torch.set_num_threads(1)

        from gp8_control.skills.thr_dt import ModelConfig, SimConfig, build_model

        model = build_model(ModelConfig(), dropout=0.0)   # 평가 시 dropout off
        checkpoint = model.load(weights)
        model.eval()

        cfg = SimConfig()
        # 그리퍼 임계 τ = 학습셋 그리퍼 액션 평균 (논문 §5.1) — 체크포인트에 실려 온다.
        tau = float(checkpoint.get("gripper_thresh", cfg.gripper_thresh))
        if log is not None:
            n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
            log.info(f"DT 로드: {weights} (params {n_par}, gripper τ={tau:.4f})")
        _MODEL_CACHE[weights] = (model, cfg, tau)
        return _MODEL_CACHE[weights]


def _rollout(model, cfg, tau, d_g: float, target_return: float):
    """DT 를 d_g 로 자기회귀 rollout 하고 **substep 격자**의 관절궤적을 돌려준다.

    `evaluate_dt.eval_model` 과 동일한 조건화(R̂=target_return, 상태에 d_g 부착,
    직전 (RTG, state, action) 전부를 컨텍스트로)이고, `RoboticArm.step` 의 적분을
    그대로 펼쳐 substep(1/(f·substeps) = 10 ms) 상태를 전부 기록하는 것만 다르다.
    시뮬 상태를 기록하는 이유는 그것이 **DT 가 학습·평가 때 실제로 만들어낸 팔의
    운동**이기 때문이다 (θ_t = θ_{t−1} + ω_t/f 는 명령이고, 관절은 1차 지연으로
    그 뒤를 따른다). 명령열을 그대로 쏘면 실기의 추종 특성이 시뮬과 다른 만큼
    릴리즈 속도가 달라진다.

    리턴 dict:
      q (3,N) qd (3,N)  — DT 규약 (t₂,t₃,t₅), 100 Hz, q[:,0] = home_pose
      t (N,)            — 아크 시작부터의 시간
      i_rel             — 릴리즈 샘플 인덱스 (None = 릴리즈 없이 시간 종료)
      x_land_sim        — Thr_DT 평면 시뮬이 예측한 착지 거리 [m]
      n_steps, actions  — 10 Hz 스텝 수 / 액션 이력 (진단용)
    """
    import torch

    from gp8_control.skills.thr_dt import RoboticArm

    arm = RoboticArm(cfg, rng=np.random.default_rng(0))
    arm.gripper_thresh = tau
    arm.update_target(np.array([float(d_g), 0.0, 0.0]))

    states = torch.zeros((0, model.state_dim), dtype=torch.float32)
    actions = torch.zeros((0, model.act_dim), dtype=torch.float32)
    rewards = torch.zeros(0, dtype=torch.float32)
    rtg = torch.tensor(target_return, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, dtype=torch.long).reshape(1, 1)

    h = cfg.dt / cfg.substeps                       # substep 적분 간격 (10 ms)
    Q, QD, acts = [arm.q.copy()], [np.zeros(3)], []
    i_rel, x_land_sim, ep, done = None, float("nan"), 0, False

    while not done:
        s = np.append(arm.get_state(), arm.target[0])
        states = torch.cat([states, torch.from_numpy(s).float().reshape(1, model.state_dim)])
        actions = torch.cat([actions, torch.zeros((1, model.act_dim))])
        rewards = torch.cat([rewards, torch.zeros(1)])
        with torch.no_grad():
            a_t = model.get_action(states, actions, rewards, rtg, timesteps)
        actions[-1] = a_t
        a = a_t.detach().cpu().numpy()
        acts.append(a.copy())

        # --- RoboticArm.step 의 적분을 substep 기록하며 전개 (env 와 동일 수식) ---
        v = arm.proj_on_max_speed(a.copy())
        v = arm.smooth_velocity(v)
        arm.velocity = v
        q_cmd = arm.q + v[:3] * cfg.dt              # 식 5.1 Taylor 위치 명령
        for _ in range(cfg.substeps):
            qd = np.clip((q_cmd - arm.q) / cfg.tau_act,
                         -arm.max_speed_rad_arm, arm.max_speed_rad_arm)
            arm.qdot = qd
            arm.q = arm.q + qd * h
            Q.append(arm.q.copy())
            QD.append(qd.copy())
        arm.curr_time += cfg.dt
        arm.curr_step += 1
        _, _, tip = arm._fk(arm.q)
        if arm.gripper_closed:
            arm.object_position = np.array([tip[0], 0.0, tip[1]])
            arm.object_height = tip[1]

        # --- 종료/릴리즈 (env.step 과 같은 판정/순서) ---
        if float(v[-1]) < arm.gripper_thresh:
            i_rel = (ep + 1) * cfg.substeps          # 이 스텝의 **끝**에서 릴리즈
            x_land_sim = float(arm._ballistic_landing(
                tip, arm._tip_velocity(arm.q, arm.qdot)))
            done = True
        elif arm.curr_step >= arm.number_steps:
            done = True                              # 시간 종료 — 릴리즈 없음
        elif arm.object_height <= cfg.object_half_diag:
            done = True                              # 물체가 바닥에 닿음 — 릴리즈 없음

        rewards[-1] = 0.0
        rtg = torch.cat([rtg, rtg[0, -1].reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps, torch.ones((1, 1), dtype=torch.long) * (ep + 1)], dim=1)
        ep += 1

    Q = np.asarray(Q).T                              # (3, N)
    QD = np.asarray(QD).T
    return dict(q=Q, qd=QD, t=np.arange(Q.shape[1]) * h, i_rel=i_rel,
                x_land_sim=x_land_sim, n_steps=ep, actions=np.asarray(acts))


# =========================================================================
# 착지 예측 (실제 GP8 형상)
# =========================================================================

def _predict_landing(q_robot: np.ndarray, qd_robot: np.ndarray, land_z: float):
    """robot 규약 (q, q̇) 에서 놓았을 때의 착지점 (base frame XY) + 발사 상태.

    발사점/속도는 NLP 스킬의 착탄 게이트와 **같은** `launch_state` 를 쓴다 —
    TCP 에서 그리퍼 로드축으로 GRIP_OFF(2 cm) 나간 점의 강체속도(ω×r 포함).
    항력은 실물 미배선이라 무항력 포물선 (NLP 게이트와 동일 가정).
    리턴: (p_land (2,) 또는 None, p_eff (3,), v_eff (3,))
    """
    p, v = launch_state(np.asarray(q_robot) * _PLANNER_SIGN,
                        np.asarray(qd_robot) * _PLANNER_SIGN)
    disc = v[2] ** 2 - 2.0 * G * (float(land_z) - p[2])
    if disc < 0.0:
        return None, p, v                            # 그 높이에 도달 못 함
    t_f = (v[2] + math.sqrt(disc)) / G                # 하강 중 통과하는 근
    if t_f <= 0.0:
        return None, p, v
    return p[:2] + v[:2] * t_f, p, v


# =========================================================================
# 실현 가능성 검사 (스킬과 `--dt-scan` 이 공유)
# =========================================================================

def _rate_limited(q_cmd: np.ndarray, t_cmd: np.ndarray, rt: np.ndarray,
                  period: float = STREAM_PERIOD):
    """컨트롤러 증분 governor 를 모사해 실제로 나올 관절 궤적을 추정.

    스트리머는 명령 경로를 4 ms 격자로 리샘플해 JGPC 로 publish 하고, YRC 증분
    motion governor + B6 host command_limiter 가 per-cycle 증분을
    `rt_stream_velocity_limits · period` 로 묶는다. 그 rate limiter 를 그대로 돌린다.

    (실제 리샘플은 cubic Hermite + knot 포락선 clamp 지만, 여기 입력은 이미 10 ms
    간격 조밀 샘플이라 선형 근사와의 차이가 무시할 수준이다 — 이 함수의 목적은
    "governor 가 릴리즈 상태를 바꾸는가" 를 보는 것이다.)

    리턴: (grid, cmd, act, vel) — 전부 4 ms 격자.
    """
    q_cmd = np.asarray(q_cmd, dtype=float)
    t_cmd = np.asarray(t_cmd, dtype=float).ravel()
    grid = np.arange(0.0, float(t_cmd[-1]) + 1e-12, period)
    cmd = np.stack([np.interp(grid, t_cmd, q_cmd[j]) for j in range(q_cmd.shape[0])])
    act = np.zeros_like(cmd)
    act[:, 0] = cmd[:, 0]
    step = np.asarray(rt, dtype=float) * period
    for k in range(1, cmd.shape[1]):
        act[:, k] = act[:, k - 1] + np.clip(cmd[:, k] - act[:, k - 1], -step, step)
    vel = np.zeros_like(act)
    vel[:, 1:] = np.diff(act, axis=1) / period
    return grid, cmd, act, vel


def check_arc(arc_q: np.ndarray, arc_qd: np.ndarray, arc_t: np.ndarray,
              p_target: np.ndarray, robot, *,
              land_gate: float = DT_LANDING_GATE,
              max_lag_deg: float = DT_MAX_LAG_DEG,
              max_clamp_shift: float = DT_MAX_CLAMP_SHIFT,
              vel_strict: bool = DT_VEL_STRICT) -> dict:
    """robot 규약 던지기 아크가 GP8 에서 실행 가능한지 독립 검증.

    NLP 쪽 `_solution_gates` 와 같은 역할이지만 검사 항목이 다르다 — DT 는 한계를
    제약으로 걸고 푸는 solver 가 아니라 **한계를 전혀 모르는 정책**이라, 여기가
    실질적인 첫 방어선이다.

    리턴 dict: reject(str|None), warns(list[str]), 그리고 진단 수치들.
    """
    rt = getattr(robot, "rt_stream_velocity_limits", None)
    rt = (np.asarray(robot.velocity_limits, dtype=float) if rt is None
          else np.asarray(rt, dtype=float))
    jl = np.asarray(robot.joint_limits, dtype=float)
    p_target = np.asarray(p_target, dtype=float).ravel()
    out: dict = {"reject": None, "warns": []}

    # --- 실기 추종 모사: governor 로 clamp 했을 때 실제로 나올 릴리즈 상태 ---
    #     착지 예측은 **이쪽**을 기준으로 한다 (로봇이 실제로 하게 될 동작).
    _, cmd, act, act_v = _rate_limited(arc_q, arc_t, rt)
    out["lag_deg"] = float(np.rad2deg(np.abs(cmd[:, -1] - act[:, -1]).max()))
    p_land, p_eff, v_eff = _predict_landing(act[:, -1], act_v[:, -1], p_target[2])
    p_ideal, _, v_ideal = _predict_landing(arc_q[:, -1], arc_qd[:, -1], p_target[2])
    out["p_land"], out["p_eff"], out["v_eff"] = p_land, p_eff, v_eff
    out["p_land_ideal"], out["v_ideal"] = p_ideal, v_ideal
    out["clamp_shift"] = (float(np.linalg.norm(p_land - p_ideal))
                          if (p_land is not None and p_ideal is not None)
                          else float("nan"))
    out["d_land"] = float(np.linalg.norm(p_land)) if p_land is not None else float("nan")
    out["err"] = (float(np.linalg.norm(p_land - p_target[:2]))
                  if p_land is not None else float("nan"))

    # --- ① 관절 위치 한계 (robot 규약, robots/gp8.joint_limits) ---
    viol = (arc_q < jl[:, :1] - 1e-9) | (arc_q > jl[:, 1:2] + 1e-9)
    out["ok_joint"] = not bool(viol.any())
    if viol.any():
        j, k = (int(v) for v in np.argwhere(viol)[0])
        out["reject"] = (
            f"관절 J{j + 1} 한계 위반 @ 아크 샘플 {k}/{arc_q.shape[1] - 1} "
            f"({np.rad2deg(arc_q[j, k]):+.1f}°, 한계 "
            f"[{np.rad2deg(jl[j, 0]):+.1f}, {np.rad2deg(jl[j, 1]):+.1f}]°) — Thr_DT "
            f"평면 시뮬에는 위치 한계가 없어 학습된 백스윙이 GP8 을 벗어난다")

    # --- ② 속도 / 추종: raw peak 이 아니라 **모사한 릴리즈 상태**로 판단 (위 주석 참고) ---
    ratio = np.abs(arc_qd) / rt[:, None]
    peak = float(ratio.max())
    out["peak_vel_ratio"] = peak
    out["ok_vel_raw"] = peak <= RT_VEL_MARGIN
    out["ok_track"] = (out["lag_deg"] <= max_lag_deg
                       and not (out["clamp_shift"] > max_clamp_shift))
    j_pk = int(np.argmax(ratio.max(axis=1)))
    vel_msg = (f"관절속도가 RT 스트림 상한을 넘는 구간이 있다 — J{j_pk + 1} "
               f"{np.rad2deg(np.abs(arc_qd[j_pk]).max()):.0f}°/s vs 상한 "
               f"{np.rad2deg(rt[j_pk]):.0f}°/s (비율 {peak:.2f})")

    if not out["ok_track"]:
        if out["reject"] is None:
            out["reject"] = (
                f"증분 governor clamp 로 릴리즈 상태가 바뀐다 — 릴리즈 시점 추종 오차 "
                f"{out['lag_deg']:.2f}° (한계 {max_lag_deg:.1f}°), 착지 이동 "
                f"{out['clamp_shift'] * 100:.1f} cm (한계 {max_clamp_shift * 100:.0f} cm). "
                f"{vel_msg}")
    elif peak > RT_VEL_MARGIN:
        msg = (vel_msg + f" — 10 Hz 스텝 시작의 2ω 과도현상이고, governor 를 모사해 보면 "
                         f"릴리즈 시점 추종 오차 {out['lag_deg']:.2f}° / 착지 이동 "
                         f"{out['clamp_shift'] * 100:.1f} cm 로 흡수된다 "
                         f"(스텝 안에서 위치 제어기가 따라잡는다)")
        if vel_strict:
            if out["reject"] is None:
                out["reject"] = msg + " (GP8_DT_VEL_STRICT=1 이라 거부)"
        else:
            out["warns"].append(msg)

    # --- ③ Cartesian 안전 엔벨로프 (아크 전 구간; fk_pos 는 로봇 FK 와 0.000 mm 일치) ---
    P = np.array([fk_pos(arc_q[:, k] * _PLANNER_SIGN) for k in range(arc_q.shape[1])])
    x_min, z_min, z_max = float(P[:, 0].min()), float(P[:, 2].min()), float(P[:, 2].max())
    out["x_min"], out["z_min"], out["z_max"] = x_min, z_min, z_max
    out["ok_cart"] = bool(x_min > MIN_TCP_X and z_min > MIN_TCP_Z and z_max <= MAX_TCP_Z)
    if not out["ok_cart"] and out["reject"] is None:
        out["reject"] = (f"Cartesian 엔벨로프 위반 (x_min={x_min:+.3f}m, "
                         f"z_min={z_min:+.3f}m, z_max={z_max:+.3f}m; 한계 "
                         f"x>{MIN_TCP_X:.2f}, {MIN_TCP_Z:.2f}<z<{MAX_TCP_Z:.2f})")

    # --- ④ 착탄: 실제 GP8 형상 기준 예측이 목표에서 너무 멀면 거부 ---
    out["ok_land"] = p_land is not None and out["err"] <= land_gate
    if out["reject"] is None and not out["ok_land"]:
        if p_land is None:
            out["reject"] = (f"착탄 불가 — 릴리즈 상태가 목표 높이 "
                             f"z={p_target[2]:+.3f}m 에 도달하지 못함 "
                             f"(|v|={np.linalg.norm(v_eff):.2f} m/s)")
        else:
            out["reject"] = (f"착탄 게이트 초과 — realGP8 예측 착지 "
                             f"({p_land[0]:+.3f}, {p_land[1]:+.3f}) m, 목표에서 "
                             f"{out['err'] * 100:.0f} cm > {land_gate * 100:.0f} cm "
                             f"(GP8_DT_LAND_GATE)")
    return out


# =========================================================================
# Skill
# =========================================================================

class DtThrowSkill(RobustThrowSkill):
    """Thr_DT Decision Transformer 로 던지는 `RobustThrowSkill` 변형.

    부모의 `execute` (ambush 픽 사이클) 와 `tests/static_pick_throw_dt.py` 가
    쓰는 공개 스테이지 시그니처를 그대로 유지한다:

        plan_nlp_throw(start_joint, T_grasp, p_target) -> (res, lift) | None
        build_throw_trajectory(grasp_joint, res, lift, next_grasp=, chain_park=)

    이름은 NLP 시절 그대로 두었다 (호출측을 안 고치려고) — 안에서 도는 것만
    NLP → DT 다.
    """

    name = "robust_throw_dt"

    def __init__(self, ctx, weights: str = DT_WEIGHTS,
                 target_return: float = DT_TARGET_RETURN) -> None:
        super().__init__(ctx)
        self.weights = weights
        self.target_return = float(target_return)
        self._dt_cache: dict = {}      # round(d_g, 4) -> rollout dict
        self._last_dt_meta: dict = {}

    # ------------------------------------------------------------------
    # 계획
    # ------------------------------------------------------------------
    def plan_nlp_throw(self, start_joint: np.ndarray, T_grasp: np.ndarray,
                       p_target: np.ndarray):
        """DT rollout → (res, lift). 실현 불가면 None (사유는 로그).

        `T_grasp` 는 로그에만 쓴다 — NLP 와 달리 DT 스윙의 시작 자세는 grasp 위치와
        무관하게 항상 학습 시 home 자세다. `start_joint` 는 lift 의 출발점.
        """
        ctx = self.ctx
        p_target = np.asarray(p_target, dtype=float).ravel()

        # 목표: DT 는 **거리 d_g** 하나로만 조건화된다. 방향은 J1(S) 이 해석적으로
        # 잡는다 (논문 §5.1 "throw direction set analytically by joint 1").
        d_g = float(np.hypot(p_target[0], p_target[1]))
        yaw = float(np.arctan2(p_target[1], p_target[0]))
        if not (DG_MIN - 1e-9 <= d_g <= DG_MAX + 1e-9):
            ctx.log.warn(
                f"DT throw 거부: d_g={d_g:.3f} m 가 학습 범위 "
                f"[{DG_MIN:.2f}, {DG_MAX:.2f}] m 밖 (논문 §5.1)")
            return None
        jl = np.asarray(ctx.robot.joint_limits, dtype=float)
        if not (jl[0, 0] <= yaw <= jl[0, 1]):
            ctx.log.warn(f"DT throw 거부: 조준 yaw {np.rad2deg(yaw):+.1f}° 가 J1 한계 밖")
            return None

        model, cfg, tau = _load_dt(self.weights, ctx.log)
        key = round(d_g, 4)
        roll = self._dt_cache.get(key)
        if roll is None:
            roll = _rollout(model, cfg, tau, d_g, self.target_return)
            self._dt_cache[key] = roll

        if roll["i_rel"] is None:
            ctx.log.warn(
                f"DT throw 거부: d_g={d_g:.3f} m — {roll['n_steps']}스텝 동안 "
                f"그리퍼가 열리지 않음 (a_gr ≤ τ={tau:.3f} 미발생, 시간 종료)")
            return None

        # ---- DT (t₂,t₃,t₅) → robot 규약 6축. R(J4)=T(J6)=0 (평면 던지기) ----
        i_rel = int(roll["i_rel"])
        n = i_rel + 1                                  # 릴리즈까지가 던지기 아크
        arc_q = np.zeros((6, n))
        arc_qd = np.zeros((6, n))
        arc_q[0, :] = yaw                              # S: 조준 (정지 — lift 가 이미 잡음)
        arc_q[1, :] = roll["q"][0, :n]                 # L = t₂
        arc_q[2, :] = roll["q"][1, :n]                 # U = t₃
        arc_q[4, :] = roll["q"][2, :n]                 # B = t₅
        arc_qd[1, :] = roll["qd"][0, :n]
        arc_qd[2, :] = roll["qd"][1, :n]
        arc_qd[4, :] = roll["qd"][2, :n]
        arc_t = roll["t"][:n]

        chk = check_arc(arc_q, arc_qd, arc_t, p_target, ctx.robot)
        for w in chk["warns"]:
            ctx.log.warn(f"DT throw 경고 (d_g={d_g:.3f} m): {w}")
        if chk["reject"] is not None:
            ctx.log.warn(f"DT throw 거부: d_g={d_g:.3f} m — {chk['reject']}")
            return None

        # ---- lift: 픽 자세 → DT home 자세 (rest-to-rest) ----
        q_home = arc_q[:, 0].copy()
        zero6 = np.zeros(6)
        l_traj, l_vel, l_ts = trajectory(
            np.asarray(start_joint, dtype=float), zero6, q_home, zero6,
            ctx.M1[:6], ctx.M2[:6], hertz=ctx.cfg.TRAJ_HZ,
        )
        bad = self._seam_check(l_traj, np.asarray(start_joint, float), q_home, "lift")
        if bad is not None:
            ctx.log.warn(f"DT throw 거부: {bad}")
            return None

        # ---- 진단: 착지 예측(실제 GP8 형상) + 릴리즈 타이밍 민감도 ----
        p_eff, v_eff = chk["p_eff"], chk["v_eff"]
        d_real, err_real = chk["d_land"], chk["err"]
        sens = self._release_sensitivity(roll, arc_q, arc_qd, i_rel, p_target[2], yaw)

        res = dict(
            kind="dt",
            arc=(arc_q, arc_qd, arc_t),
            i_rel=n - 1,                    # 아크 마지막 샘플이 릴리즈 지점
            t_f=float(arc_t[-1]),           # 아크 길이 [s] (호출측 로그 호환)
            J=err_real,                     # 호출측이 "J=" 로 찍는 값 = 예측 착지오차 [m]
            release_time=0.0,               # DT 는 릴리즈 윈도우가 없다 (NLP 는 t*±T/2)
            d_g=d_g, yaw=yaw, n_steps=int(roll["n_steps"]),
            x_land_sim=float(roll["x_land_sim"]), d_land_real=d_real,
            err_real=err_real, p_land=chk["p_land"], p_eff=p_eff, v_eff=v_eff,
            sens_mm_per_10ms=sens, gripper_tau=float(tau),
            a_gr_release=float(roll["actions"][-1][3]),
            peak_vel_ratio=chk["peak_vel_ratio"], lag_deg=chk["lag_deg"],
            clamp_shift=chk["clamp_shift"],
        )
        ctx.log.info(
            f"DT throw [d_g={d_g:.3f} m, yaw={np.rad2deg(yaw):+.1f}°]: "
            f"{roll['n_steps']}스텝 × {1.0 / cfg.update_rate * 1e3:.0f} ms, "
            f"아크 {arc_t[-1]:.2f}s, lift {l_ts[-1]:.2f}s | "
            f"착지 예측 DT-sim {roll['x_land_sim']:.3f} m / realGP8 {d_real:.3f} m "
            f"(목표 오차 {err_real * 100:.1f} cm) | "
            f"|v_release|={np.linalg.norm(v_eff):.2f} m/s, "
            f"릴리즈 민감도 {sens:.0f} mm/10 ms | "
            f"속도 peak {chk['peak_vel_ratio']:.2f}×RT상한 → governor 모사 결과 "
            f"릴리즈 추종오차 {chk['lag_deg']:.2f}°, 착지 이동 "
            f"{chk['clamp_shift'] * 100:.1f} cm"
        )
        return res, (l_traj, l_vel, l_ts)

    # ------------------------------------------------------------------
    def _rt_vel_limits(self) -> np.ndarray:
        """컨트롤러 실측 RT 스트림 속도 상한 [rad/s] (없으면 데이터시트 한계)."""
        rt = getattr(self.ctx.robot, "rt_stream_velocity_limits", None)
        if rt is None:
            return np.asarray(self.ctx.robot.velocity_limits, dtype=float)
        return np.asarray(rt, dtype=float)

    def _release_sensitivity(self, roll: dict, arc_q, arc_qd, i_rel: int,
                             land_z: float, yaw: float) -> float:
        """릴리즈를 ±1 substep(10 ms) 어긋냈을 때 착지점이 움직이는 거리 [mm].

        NLP 해는 t*±25 ms 윈도우 내내 착지가 보장되지만 DT 는 릴리즈가 한 점이다.
        밸브 jitter 가 그대로 착지 오차가 되므로 그 크기를 수치로 남긴다.
        """
        base, _, _ = _predict_landing(arc_q[:, -1], arc_qd[:, -1], land_z)
        if base is None:
            return float("nan")
        j = i_rel - 1
        if j < 0:
            return float("nan")
        q_p = np.zeros(6); qd_p = np.zeros(6)
        q_p[0] = yaw
        q_p[1], q_p[2], q_p[4] = roll["q"][0, j], roll["q"][1, j], roll["q"][2, j]
        qd_p[1], qd_p[2], qd_p[4] = roll["qd"][0, j], roll["qd"][1, j], roll["qd"][2, j]
        prev, _, _ = _predict_landing(q_p, qd_p, land_z)
        if prev is None:
            return float("nan")
        return float(np.linalg.norm(prev - base) * 1000.0)

    @staticmethod
    def _seam_check(traj: np.ndarray, q0: np.ndarray, q1: np.ndarray, tag: str,
                    tol_start: float = 1e-6, tol_end: float = 0.02) -> "Optional[str]":
        """`trajectory()` 결과가 실제로 q0 → q1 을 잇는지 확인.

        `_trajectory_1d` 는 |N| > M1 일 때 **None 을 리턴하고 호출측이 그 관절 행을
        0 으로 남긴다** (trajectory_primitive.py:126 `if result is not None`) — 조용히
        "그 관절을 0 rad 로 보내라"는 명령이 된다. 진입 속도가 있는 감속 세그먼트에서
        특히 위험하므로 항상 확인한다. 세 가지를 본다:
          ① 통째로 0 인 행 (q0/q1 이 둘 다 0 인 관절은 정상이므로 제외)
          ② 시작점 (구성상 정확히 q0 여야 한다)
          ③ 끝점 (L = int(T·hertz) 절단 때문에 미세하게 못 미칠 수 있어 여유 있게)
        """
        q0 = np.asarray(q0, dtype=float).ravel()
        q1 = np.asarray(q1, dtype=float).ravel()
        zero_rows = np.all(traj == 0.0, axis=1) & ~(
            (np.abs(q0) < 1e-12) & (np.abs(q1) < 1e-12))
        if zero_rows.any():
            j = int(np.argmax(zero_rows))
            return (f"{tag} 세그먼트: J{j + 1} 행이 전부 0 — trajectory() 가 이 관절을 "
                    f"만들지 못했다 (진입 속도가 M1 초과). 그대로 보내면 J{j + 1} 을 "
                    f"0°로 끌고 간다 (현재 {np.rad2deg(q0[j]):+.1f}° → "
                    f"{np.rad2deg(q1[j]):+.1f}°)")
        d0 = float(np.max(np.abs(traj[:, 0] - q0)))
        d1 = float(np.max(np.abs(traj[:, -1] - q1)))
        if d0 > tol_start or d1 > tol_end:
            j = int(np.argmax(np.abs(traj[:, -1] - q1)))
            return (f"{tag} 세그먼트 생성 실패 (시작 오차 {np.rad2deg(d0):.3f}°, "
                    f"끝 오차 {np.rad2deg(d1):.2f}° @ J{j + 1}) — trajectory() 가 "
                    f"속도/가속 한계 안에서 이 구간을 만들지 못했다")
        return None

    # ------------------------------------------------------------------
    # 조립 + dispatch
    # ------------------------------------------------------------------
    def build_throw_trajectory(self, grasp_joint: np.ndarray, res: dict, lift,
                               next_grasp: "Optional[np.ndarray]" = None,
                               chain_park: "Optional[np.ndarray]" = None) -> None:
        """lift + DT 아크 + 감속/체인 을 이어 붙여 timed-release 로 디스패치.

        NLP 버전과 안전 게이트(관절 한계 / Cartesian 엔벨로프)는 동일하고, 그 위에
        seam 연속성 확인이 추가된다. 게이트에 걸리면 **디스패치하지 않고** 리턴한다
        (`static_pick_throw_dt` 는 그 사이클을 통째로 제외한다).
        """
        ctx = self.ctx
        l_traj, l_vel, l_ts = lift
        arc_q, arc_qd, arc_t = res["arc"]

        # 아크 첫 샘플은 lift 끝(= DT home, 정지)과 같은 상태라 버리고 시계만 잇는다.
        a_traj = arc_q[:, 1:]
        a_vel = arc_qd[:, 1:]
        a_ts = arc_t[1:] + l_ts[-1]
        i_rel_concat = l_traj.shape[1] + int(res["i_rel"]) - 1

        # ---- 감속 + 체인: 릴리즈 상태(고속)에서 park 까지 사다리꼴 rest 종료 ----
        # M1 은 --vel-scale 로 줄인 ctx.M1 을 쓰지 않는다: 진입 속도가 M1 을 넘으면
        # trajectory() 가 그 관절 행을 0 으로 남긴다(_seam_check 가 잡지만, 애초에
        # 정지 자체가 불가능해진다). 스윙 속도로 감속할 수 있어야 하므로 로봇 한계
        # (실측 RT 상한과 데이터시트 중 작은 쪽)를 쓴다.
        q_rel = a_traj[:, -1]
        qd_rel = a_vel[:, -1]
        M1_stop = np.minimum(
            np.asarray(ctx.robot.velocity_limits, dtype=float)
            * ctx.cfg.JOINT_VEL_LIMIT_SCALE,
            self._rt_vel_limits())
        M2_stop = M1_stop * ctx.cfg.JOINT_ACCEL_LIMIT_SCALE
        if np.any(np.abs(qd_rel) > M1_stop):
            j = int(np.argmax(np.abs(qd_rel) - M1_stop))
            ctx.log.error(
                f"DT throw ABORTED (not dispatched): 릴리즈 속도 J{j + 1} "
                f"{np.rad2deg(qd_rel[j]):+.0f}°/s 가 감속 한계 "
                f"{np.rad2deg(M1_stop[j]):.0f}°/s 를 넘어 정지 세그먼트를 만들 수 없다")
            return

        chain_target, chain_dest = self._chain_target(q_rel, next_grasp, chain_park)
        zero6 = np.zeros(6)
        c_traj, c_vel, c_ts = trajectory(
            q_rel, qd_rel, chain_target[:6], zero6,
            M1_stop, M2_stop, hertz=ctx.cfg.TRAJ_HZ,
        )
        bad = self._seam_check(c_traj, q_rel, chain_target[:6], "감속/체인")
        if bad is not None:
            ctx.log.error(f"DT throw ABORTED (not dispatched): {bad}")
            return
        c_traj, c_vel = c_traj[:, 1:], c_vel[:, 1:]     # 첫 열은 아크 끝과 중복
        c_ts_shift = c_ts[1:] + a_ts[-1]

        traj = np.concatenate((l_traj, a_traj, c_traj), axis=1)
        vel = np.concatenate((l_vel, a_vel, c_vel), axis=1)
        ts = np.concatenate((l_ts, a_ts, c_ts_shift))
        assert traj.shape[1] == vel.shape[1] == ts.shape[0], (
            f"DT throw traj/vel/timestep length mismatch: "
            f"{traj.shape[1]}/{vel.shape[1]}/{ts.shape[0]}")
        if not np.all(np.diff(ts) > 0):
            k = int(np.argmin(np.diff(ts)))
            ctx.log.error(
                f"DT throw ABORTED (not dispatched): timestep 비단조 @ {k} "
                f"({ts[k]:.4f} → {ts[k + 1]:.4f}) — 스트리머의 Hermite 리샘플이 깨진다")
            return

        # ---- 릴리즈 인덱스: 시간 기준 (아크가 100 Hz, lift/체인이 TRAJ_HZ 라 index
        #      기준 lead 는 틀린다). RELEASE_LEAD 부호 규약은 NLP 버전과 동일 —
        #      양수 = 그만큼 일찍 명령(밸브 지연 보정), 음수 = 늦게.
        t_rel = float(ts[i_rel_concat])
        t_cmd = t_rel - float(ctx.cfg.RELEASE_LEAD)
        t_cmd = min(max(t_cmd, float(ts[0])), float(ts[-1]))
        release_idx = int(np.argmin(np.abs(ts - t_cmd)))
        if release_idx > i_rel_concat:
            ctx.log.warn(
                f"릴리즈 명령이 아크 끝을 지나 감속 구간에 떨어진다 "
                f"(RELEASE_LEAD={ctx.cfg.RELEASE_LEAD:+.2f}s) — 물체가 감속 중에 "
                f"떨어져 계획보다 짧게 날아간다. GP8_RELEASE_LEAD 를 0 이상으로.")

        # ---- SAFETY GATE 1: 관절 한계 (연결·변환된 전체 궤적, robot 규약) ----
        jl = np.asarray(ctx.robot.joint_limits, dtype=float)
        viol = (traj < jl[:, :1] - 1e-6) | (traj > jl[:, 1:2] + 1e-6)
        if viol.any():
            j, k = (int(v) for v in np.argwhere(viol)[0])
            ctx.log.error(
                f"DT throw ABORTED (not dispatched): robot-frame joint {j + 1} "
                f"limit violation at sample {k} "
                f"({np.rad2deg(traj[j, k]):+.1f}°, limits "
                f"[{np.rad2deg(jl[j, 0]):+.1f}, {np.rad2deg(jl[j, 1]):+.1f}]°, "
                f"{self._segment_of(k, l_traj.shape[1], a_traj.shape[1])} 구간)")
            return

        # ---- SAFETY GATE 2: Cartesian 엔벨로프 (NLP 버전과 동일 기준/예외) ----
        n_s = traj.shape[1]
        tcp = np.stack([np.asarray(ctx.robot.forward_kinematics(traj[:, k]),
                                   dtype=float)[:3, 3] for k in range(n_s)], axis=1)
        bad_list: list = []
        cleared = np.nonzero(tcp[2] > MIN_TCP_Z)[0]
        if cleared.size == 0:
            bad_list.append((0, f"궤적 전체가 z ≤ {MIN_TCP_Z:.3f}m (lift 가 바닥을 못 벗어남)"))
        else:
            k0 = int(cleared[0])
            dip = np.nonzero(tcp[2, k0:] <= MIN_TCP_Z)[0]
            if dip.size:
                k = k0 + int(dip[0])
                bad_list.append((k, f"TCP z={tcp[2, k]:+.4f}m ≤ {MIN_TCP_Z:.3f}m (바닥/벨트)"))
        near = np.nonzero(tcp[0] <= MIN_TCP_X)[0]
        if near.size:
            k = int(near[0])
            bad_list.append((k, f"TCP x={tcp[0, k]:+.4f}m ≤ {MIN_TCP_X:.3f}m (기둥/베이스)"))
        high = np.nonzero(tcp[2] > MAX_TCP_Z)[0]
        if high.size:
            k = int(high[0])
            bad_list.append((k, f"TCP z={tcp[2, k]:+.4f}m > {MAX_TCP_Z:.3f}m (팔 과다 상승)"))
        if bad_list:
            k, why = min(bad_list)
            n_viol = int(np.count_nonzero(
                (tcp[0] <= MIN_TCP_X) | (tcp[2] <= MIN_TCP_Z) | (tcp[2] > MAX_TCP_Z)))
            ctx.log.error(
                f"DT throw ABORTED (not dispatched): Cartesian 안전 엔벨로프 위반 — "
                f"{why} @ sample {k}/{n_s} (t={ts[k]:.3f}s, "
                f"{self._segment_of(k, l_traj.shape[1], a_traj.shape[1])} 구간, "
                f"위반 샘플 {n_viol}개)")
            ctx.log.error(
                "  위반 지점 관절 (robot frame, deg): "
                + ", ".join(f"J{j + 1}={np.rad2deg(traj[j, k]):+.1f}" for j in range(6)))
            return

        ctx.log.info(
            f"DT throw dispatch: lift {l_traj.shape[1]} + arc {a_traj.shape[1]} "
            f"+ decel/chain {c_traj.shape[1]} samples ({ts[-1]:.2f}s), "
            f"release step {release_idx} (t={ts[release_idx]:.3f}s, 아크 릴리즈 "
            f"t={t_rel:.3f}s, lead {ctx.cfg.RELEASE_LEAD:+.2f}s, "
            f"타이밍 여유 없음 — 민감도 {res['sens_mm_per_10ms']:.0f} mm/10 ms), "
            f"chain->{chain_dest}")

        ctx.traj_ctrl.send_trajectory_queue_with_timed_release(
            traj, vel, ts, final_joint=chain_target, release_index=release_idx)
        self._last_dt_meta = {
            "d_g": res["d_g"], "yaw": res["yaw"], "n_steps": res["n_steps"],
            "arc_T": res["t_f"], "lift_T": float(l_ts[-1]),
            "x_land_sim": res["x_land_sim"], "d_land_real": res["d_land_real"],
            "err_real": res["err_real"],
            "v_release": float(np.linalg.norm(res["v_eff"])),
            "sens_mm_per_10ms": res["sens_mm_per_10ms"],
            "release_idx": release_idx, "n_steps_traj": int(traj.shape[1]),
        }
        # 부모의 _log_throw_cycle 이 참조하는 필드도 채워 둔다 (CSV 호환).
        self._last_throw_meta = {
            "T": 0.0, "lift_T": float(l_ts[-1]), "t_f": res["t_f"],
            "t_star": t_rel - float(l_ts[-1]), "J": res["J"],
            "release_idx": release_idx, "n_steps": int(traj.shape[1]),
        }

    # ------------------------------------------------------------------
    def _chain_target(self, q_end: np.ndarray, next_grasp, chain_park):
        """NLP 버전과 동일한 우선순위: next park > over next grasp > idle > standby."""
        ctx = self.ctx
        if chain_park is not None:
            return np.asarray(chain_park, dtype=float).copy(), "next action-start park"
        if next_grasp is not None:
            return ctx.lifted_standby_joint(next_grasp), "over next grasp"
        if not ctx.queue:
            return self.idle_target().copy(), "home/idle (queue empty)"
        return ctx.lifted_standby_joint(q_end), "lifted standby"

    @staticmethod
    def _segment_of(k: int, n_lift: int, n_arc: int) -> str:
        return "lift" if k < n_lift else ("throw" if k < n_lift + n_arc else "decel/chain")

    # ------------------------------------------------------------------
    def _log_throw_cycle(self, target: "TrackedObject") -> None:
        """사이클 1행을 PICK_LOG_CSV 에 append (DT 진단 컬럼 포함)."""
        ctx = self.ctx
        path = ctx.cfg.PICK_LOG_CSV
        if not path:
            return
        lt = getattr(ctx.traj_ctrl, "last_throw", {}) or {}
        m = self._last_dt_meta or {}
        son = getattr(ctx.traj_ctrl, "last_suction_on_t", None)
        t0 = lt.get("throw_start")
        trel = lt.get("release_wall")

        def _d(a, b):
            return round(a - b, 4) if (a is not None and b is not None) else ""

        row = {
            "iso_time": datetime.datetime.now().isoformat(timespec="milliseconds"),
            "class": target.class_name,
            "skill": self.name,
            "belt_mps": round(ctx.conveyor.current, 4),
            "suction_on_t": round(son, 4) if son else "",
            "throw_start_t": round(t0, 4) if t0 else "",
            "release_t": round(trel, 4) if trel else "",
            "on_to_throwstart_s": _d(t0, son),
            "throwstart_to_release_s": _d(trel, t0),
            "d_g_m": round(m.get("d_g", 0.0), 4),
            "yaw_deg": round(np.rad2deg(m.get("yaw", 0.0)), 2),
            "dt_steps": m.get("n_steps", ""),
            "arc_T_s": round(m.get("arc_T", 0.0), 3),
            "lift_T_s": round(m.get("lift_T", 0.0), 3),
            "x_land_sim_m": round(m.get("x_land_sim", 0.0), 4),
            "d_land_real_m": round(m.get("d_land_real", 0.0), 4),
            "err_real_m": round(m.get("err_real", 0.0), 4),
            "v_release_mps": round(m.get("v_release", 0.0), 3),
            "sens_mm_per_10ms": round(m.get("sens_mm_per_10ms", 0.0), 1),
            "release_idx": m.get("release_idx", ""),
            "n_steps_traj": m.get("n_steps_traj", ""),
            "io_ms": round(lt["io_ms"], 1) if lt.get("io_ms") is not None else "",
        }
        self._append_csv_row(path, row, ctx.log)

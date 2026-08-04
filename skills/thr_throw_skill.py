"""Throw skill: **THR 던지기 계획 모델 3종(nlp / dt / phy)** 을 로봇에 태우는 스킬.

`robust_throw_skill.RobustThrowSkill` (구 NLP 스킬) 의 `_thr` 변형이다. 픽
(hover → suction → press → 벨트추종)·체인·디스패치는 부모 것을 그대로 쓰고,
**계획 단계만** THR 의 traj_fn 으로 교체한다:

    RobustThrowSkill : skills/throw_nlp.solve_throw_nlp        (구 공식화, tool 0.220)
    ThrThrowSkill    : skills/thr_planners.get_traj_fn(model)  (THR 2026-08-03 기준)

모델 3종은 THR 시뮬에서 같은 traj_fn 인터페이스로 비교된 것들이다
(`THR/bench_jitter.py`, 영상 `THR/video/jitter_{nlp,dt,phy}_{ideal,real}.mp4`):

  nlp — CasADi/IPOPT B-spline NLP. release **윈도우** 전 구간의 착탄 정확도를
        목적함수(W_ACC)로 강제 → 밸브 지터에 강하다.
  dt  — Decision Transformer, THR GP8 rig 에서 직접 학습 (RA-L 2023 재현).
        10 Hz 자기회귀, a_gr ≤ τ 스텝에서 릴리즈.
  phy — TossingBot Physics-only 탄도 컨트롤러 (T-RO 2020 재현). 45° 직선 램프.

THR 시뮬 결과 (컨트롤러 보간 ON, 12 던지기 × 3 세션, seed 0 기준):
        ideal(레버0·지터0)          real(무작위 파지·지터±50ms)
  nlp   12/12, 평균 26 mm           12/12, 평균 32 mm
  dt    12/12, 평균 51 mm           11/12, 평균 84 mm
  phy   10/12, 평균 85 mm            7/12, 평균 138 mm

──────────────────────────────────────────────────────────────────────────────
2026-08-03 THR 업데이트가 반영하는 것 (구 skills/throw_nlp.py 대비)
──────────────────────────────────────────────────────────────────────────────
  tool              0.220 → **0.240** (link6→TCP 0.30 → 0.32 m, URDF 기준 통일)
  RELEASE_TIME      0.05  → **0.1 s** (릴리즈 윈도우 2배 — 지터 강건성 우선)
  QDD_LIM           3×qḋ → **5×qḋ**
  W1 (시간 가중치)   0.5   → **5.0**
  CART_CONSTRAINTS  True  → **False** (NLP 에서 기둥 회피가 빠졌다 ⚠ 아래 참고)
  관절 위치 한계     datasheet → **URDF ∩ 실기 ∩ 사용자 규칙**
                     S |q|≤60°, L[−15,45], U[−30,45], B[10,135], |R|≤80° (planner)
⇒ 기존 warm DB (skills/warm_db_*.pkl) 는 이 공식화에서 **전부 무효**다. 그래서
  nlp 모델은 cold multistart 로 돈다 (지점당 수 초~수십 초). static 테스트는
  로봇이 움직이기 전에 전 지점을 미리 계획하므로 실행 중 지연은 없다.

⚠ **CART_CONSTRAINTS=False 의 의미**: THR 은 2026-07-31 사용자 지시로 NLP 에서
  카타시안 제약을 모두 뺐고, 시뮬은 위반을 '보고만' 한다
  (`nlp_planner.CART_CHECK_BLOCKING=False`). 시뮬에서는 그래도 되지만 실기에서는
  기둥/바닥을 실제로 친다. 그래서 이 스킬은 **dispatch 직전 Cartesian 안전
  엔벨로프 게이트를 hard 로 유지한다** (x>0.20, z>0.04). z 상한은
  2026-08-04 사용자 지시로 검사하지 않는다. 위반하는 계획은 실행하지 않고
  그 지점을 건너뛴다.
  GP8_THR_NLP_CART=1 로 두면 NLP 자체의 기둥 회피 제약도 되살린다 (해가 줄지만
  게이트 통과율은 올라간다).

──────────────────────────────────────────────────────────────────────────────
각도 규약 / 기하
──────────────────────────────────────────────────────────────────────────────
THR traj_fn 은 **planner 프레임** 6축을 돌려준다. 로봇(URDF) 규약으로는
`q_robot = q_planner · SIGN`, SIGN = [1, 1, −1, −1, −1, −1] — 부모 스킬의
`_PLANNER_SIGN` 과 동일하다.

기하 정합 (2026-08-04 실물 툴 24 cm 운영자 확인):
  이 경로(THR)          link6 → TCP = 0.320 m  (d6 0.080 + tool 0.240)
  robots/gp8.py         link6 → TCP = 0.320 m  (home_ee x = 0.700)
  skills/throwing.py    link6 → TCP = 0.300 m  (구 NLP 스킬 전용)
픽의 구 22 cm 좌표는 Z를 0.02 m 낮춰 같은 실제 관절 자세가 되도록 변환했다
(press 0.04→0.02, GRASP_Z 0.062→0.042, tracking 0.12/0.05→0.10/0.03).
구 NLP 스킬의 `skills/throwing.py` 는 별도 공식화이므로 이번 THR 수집 경로에서는
쓰지 않는다.

──────────────────────────────────────────────────────────────────────────────
NLP 버전(부모)과의 구조 차이
──────────────────────────────────────────────────────────────────────────────
* 던지기 시작 TCP 는 THR 규약을 따른다 — 물체 xy + z = P_START_Z(0.20 m).
  부모는 grasp + THROW_LIFT(0.10) 였다.
* **dt/phy 는 시작 자세가 계획에 영향을 주지 않는다.** dt 는 항상 학습 시 home
  자세에서 출발하고, phy 는 릴리즈점을 목표로부터 해석적으로 정한다. nlp 만
  p_start 를 실제로 쓴다.
* **감속 꼬리를 우리가 붙인다.** nlp/phy 는 아크가 정지로 끝나지만 dt 는 짧은
  팔로우스루 뒤 속도가 남는다. 릴리즈 상태에서 park 까지 사다리꼴 세그먼트를
  만들어 이어 붙이되, M1 은 `--vel-scale` 로 줄이지 **않는다** — 진입 속도가 M1
  을 넘으면 `trajectory()` 가 그 관절 행을 0 으로 남기기 때문이다(치명적).
  seam 연속성은 dispatch 전에 명시적으로 확인한다.
* 릴리즈: nlp 은 윈도우 중앙(±50 ms 여유), dt/phy 는 점 하나 — 지터 내성 차이가
  여기서 나온다. 스킬이 ±10 ms 당 착지 변화(민감도)를 계산해 로그에 남긴다.

※ 이 파일은 실기 미검증이다 (2026-08-04 작성). `--plan-only` 로 먼저 확인할 것.
"""

from __future__ import annotations

import datetime
import math
import os
from typing import TYPE_CHECKING, Optional

import numpy as np

from gp8_control.skills import thr_planners
from gp8_control.skills.robust_throw_skill import (
    MIN_TCP_X,
    MIN_TCP_Z,
    _PLANNER_SIGN,
    RobustThrowSkill,
)
from gp8_control.skills.thr import throw_nlp
from gp8_control.skills.thr.throwing import G, fk_pos, landing_error, launch_state
from gp8_control.trajectory.trajectory_primitive import trajectory

if TYPE_CHECKING:
    from gp8_control.tracking import TrackedObject


# =========================================================================
# 설정
# =========================================================================

# z 바닥 게이트를 arm 하기 전 요구하는 상승 여유 [m] (아래 GATE 2 주석 참고)
Z_ARM_MARGIN: float = float(os.environ.get("GP8_THR_Z_ARM_MARGIN", "0.02"))

# 속도 게이트 기준 = **URDF/데이터시트 관절 속도 한계** (robots/gp8._VELOCITY_LIMITS,
# S/L/U/R/B/T = 455/385/520/550/550/1000 °/s). 로봇이 물리적으로 낼 수 있는 속도이고
# NLP·DT env 가 계획에 거는 한계(throwing.GP8_QD_MAX)와 같은 값이다.
#
# ⚠ 컨트롤러 외부증분 스트리밍 상한(robots/gp8._RT_STREAM_VELOCITY_LIMITS, 위 값의
#   절반)은 **여기서 고려하지 않는다** (2026-08-04 사용자 지시). 그건 명령을 밀어넣는
#   통로 크기의 문제라 컨트롤러 쪽(axis_increment_factor)에서 다룰 사안이고, 계획이
#   로봇의 물리 한계를 지켰는지와는 별개다. 이전 버전은 이 상한으로 governor clamp 를
#   모사해 거부/경고를 냈는데, 그 때문에 물리적으로 멀쩡한 궤적이 대량 기각됐다.
VEL_GATE_MARGIN: float = float(os.environ.get("GP8_THR_VEL_MARGIN", "1.0"))

# Cartesian 안전 엔벨로프를 **거부 사유로 쓸지, 보고만 할지** (2026-08-04 사용자 지시).
#   런타임(계획/스캔/dispatch) : 기본 False = **보고만** 하고 그대로 진행
#   수집(warm DB 빌드, 실기 데이터 수집) : True = 위반한 해를 **제외**
# 수집 쪽은 tools/build_warm_db_thr.py 가 check_arc(cart_blocking=True) 로,
# tools/collect_real_throws_gp8.py 가 이 모듈 전역을 True 로 세팅해 쓴다.
# ⚠ 런타임 보고 모드에서는 스윙이 기둥/바닥 엔벨로프를 침범해도 실행된다.
#   NLP 은 CART_CONSTRAINTS=1 로 계획 자체가 기둥을 피하지만(바닥 클리어런스는
#   hard 제약이 아님), DT/phy 는 그런 제약이 없다. GP8_THR_CART_BLOCK=1 로
#   종전처럼 차단할 수 있다.
CART_BLOCKING: bool = os.environ.get("GP8_THR_CART_BLOCK", "0") not in ("0", "", "false")

# NLP 자체의 카타시안(기둥 회피) 제약 — **실기 기본은 ON** (2026-08-04 사용자 지시).
# THR 은 시뮬용으로 이걸 껐고(위반을 보고만 함) 그 상태로 계획하면 스윙이 바닥/기둥을
# 파고드는 해가 정상 수렴한다 (실측: 1.60 m 타겟에서 z_min=−0.021 m, x_min=+0.037 m).
# 시뮬은 그래도 되지만 실기는 실제로 친다. GP8_THR_NLP_CART=0 으로 끌 수 있다.
# 주의: 이 값은 warm DB 유효성 키(warm_db_params 의 'col')에 들어가므로, 켜고 끄면
# 서로 다른 config 로 갈린다 — 각각의 DB 가 필요하다.
if os.environ.get("GP8_THR_NLP_CART", "1") not in ("0", "", "false"):
    throw_nlp.CART_CONSTRAINTS = True

# --- NLP 스윙 속도 노브 (실기 전용) ------------------------------------------
# 2026-08-03 업데이트로 W1 0.5→5.0, QDD_LIM 3×→5× 가 되면서 스윙이 짧고 빨라졌다.
# 시뮬에서는 이득이지만 실기에서는 **컨트롤러 증분 governor 의 스트림 속도 상한**
# (robots/gp8._RT_STREAM_VELOCITY_LIMITS, L/U/B = 193/259/275 °/s) 을 넘어선다:
# 실측상 먼 bin(1.35·1.60 m)에서 J3 가 341~372 °/s 까지 올라가고, governor 가
# 잘라내면 릴리즈 자세가 4° 어긋나 착지가 22~47 cm 이동한다 → check_arc 가 거부.
# 이 두 값을 낮추면 스윙이 느려져 상한 안으로 들어온다 (착탄 정확도는 W_ACC 와
# 릴리즈 윈도우가 계속 보장한다). 공식화 상수라 warm DB 를 쓴다면 무효화되지만
# 이 경로는 애초에 warm DB 를 쓰지 않는다.
_W1 = os.environ.get("GP8_THR_NLP_W1")
if _W1:
    throw_nlp.W1 = float(_W1)
_QDD = os.environ.get("GP8_THR_NLP_QDD_SCALE")
if _QDD:
    throw_nlp.QDD_LIM = float(_QDD) * throw_nlp.GP8_QD_MAX


def _predict_landing(q_robot, qd_robot, land_z: float):
    """robot 규약 (q, q̇) 에서 놓았을 때의 착지점 (base frame XY) + 발사 상태.

    발사점/속도는 THR 과 **같은** `launch_state` (TCP 에서 그리퍼 로드축으로
    GRIP_OFF=2 cm 나간 점의 강체속도, ω×r 포함). 항력은 실물 미배선이라 무항력
    포물선 — THR 의 NLP 계획 물리식과 같은 가정이다 (PLAN_WITH_DRAG=False).
    """
    p, v = launch_state(np.asarray(q_robot) * _PLANNER_SIGN,
                        np.asarray(qd_robot) * _PLANNER_SIGN)
    disc = v[2] ** 2 - 2.0 * G * (float(land_z) - p[2])
    if disc < 0.0:
        return None, p, v
    t_f = (v[2] + math.sqrt(disc)) / G          # 하강 중 통과하는 근
    if t_f <= 0.0:
        return None, p, v
    return p[:2] + v[:2] * t_f, p, v


def check_arc(arc_q, arc_qd, arc_t, p_target, robot, model: str,
              cart_blocking: "Optional[bool]" = None) -> dict:
    """dispatch 전 독립 재검증 (스킬과 `--thr-scan` 이 공유).

    THR 은 시뮬 dry-run 으로 기둥/바닥을 검사하지만 로봇에는 그 시뮬이 없다.
    게다가 nlp 은 CART_CONSTRAINTS=False 라 계획 자체에 기둥 회피가 없다 —
    **여기가 실질적인 유일한 방어선**이다.
    리턴 dict: reject(str|None), warns(list), 진단 수치들.
    """
    qd_lim = np.asarray(robot.velocity_limits, float)      # URDF/데이터시트
    jl = np.asarray(robot.joint_limits, float)
    p_target = np.asarray(p_target, float).ravel()
    out: dict = {"reject": None, "warns": []}

    # --- 착지 예측: 계획의 **해석적** 릴리즈 상태 그대로 ---
    #     스트리머(`_stream_trajectory`)가 위치+속도를 cubic Hermite 로 4 ms
    #     리샘플하므로 knot 의 속도가 그대로 재현된다 — 즉 실기 릴리즈 상태는
    #     계획 그대로다. 별도 보정 없이 이 값을 쓴다.
    p_land, p_eff, v_eff = _predict_landing(arc_q[:, -1], arc_qd[:, -1], p_target[2])
    out.update(p_land=p_land, p_eff=p_eff, v_eff=v_eff)
    out["d_land"] = float(np.linalg.norm(p_land)) if p_land is not None else float("nan")
    out["err"] = (float(np.linalg.norm(p_land - p_target[:2]))
                  if p_land is not None else float("nan"))

    # --- ① 관절 위치 한계 (robot 규약, robots/gp8.joint_limits) ---
    viol = (arc_q < jl[:, :1] - 1e-9) | (arc_q > jl[:, 1:2] + 1e-9)
    out["ok_joint"] = not bool(viol.any())
    if viol.any():
        j, k = (int(v) for v in np.argwhere(viol)[0])
        out["reject"] = (f"관절 J{j + 1} 한계 위반 @ 아크 샘플 {k}/{arc_q.shape[1] - 1} "
                         f"({np.rad2deg(arc_q[j, k]):+.1f}°, 한계 "
                         f"[{np.rad2deg(jl[j, 0]):+.1f}, {np.rad2deg(jl[j, 1]):+.1f}]°)")

    # --- ② 관절 속도 한계 (URDF/데이터시트) ---
    ratio = np.abs(arc_qd) / qd_lim[:, None]
    out["peak_vel_ratio"] = float(ratio.max())
    out["ok_vel"] = out["peak_vel_ratio"] <= VEL_GATE_MARGIN
    if not out["ok_vel"] and out["reject"] is None:
        j = int(np.argmax(ratio.max(axis=1)))
        out["reject"] = (f"관절속도 한계 초과 J{j + 1} "
                         f"{np.rad2deg(np.abs(arc_qd[j]).max()):.0f}°/s > "
                         f"{np.rad2deg(qd_lim[j] * VEL_GATE_MARGIN):.0f}°/s (URDF)")

    # --- ③ Cartesian 안전 엔벨로프 (THR fk = 실물 tool 0.240 기준) ---
    P = np.array([fk_pos(arc_q[:, k] * _PLANNER_SIGN) for k in range(arc_q.shape[1])])
    x_min, z_min = float(P[:, 0].min()), float(P[:, 2].min())
    out.update(x_min=x_min, z_min=z_min)
    out["ok_cart"] = bool(x_min > MIN_TCP_X and z_min > MIN_TCP_Z)
    if not out["ok_cart"]:
        msg = (f"Cartesian 엔벨로프 위반 (x_min={x_min:+.3f}m, z_min={z_min:+.3f}m; "
               f"한계 x>{MIN_TCP_X:.2f}, z>{MIN_TCP_Z:.2f})"
               + ("  ※ NLP 의 기둥 회피는 release 창까지만 활성이고 바닥 클리어런스는"
                  " hard 제약이 아니다 (throw_nlp 제약 3b) — 감속 꼬리가 내려앉는"
                  " 해는 여기서만 걸린다" if model == "nlp" else ""))
        # 런타임은 보고만, 수집(warm DB / 실기 데이터)은 제외 — CART_BLOCKING 주석 참고
        block = CART_BLOCKING if cart_blocking is None else bool(cart_blocking)
        if block:
            if out["reject"] is None:
                out["reject"] = msg
        else:
            out["warns"].append(msg + "  — 보고만 하고 진행 "
                                      "(GP8_THR_CART_BLOCK=1 로 차단)")

    # --- ④ 착탄 진단 (blocking 하지 않음) ---
    # 실제 데이터를 폭넓게 모으기 위해 목표 오차가 커도 거부하지 않는다.
    # 단, 탄도 교점 자체를 계산할 수 없는 궤적은 유효한 수집 샘플이 아니므로 거부한다.
    out["ok_land"] = p_land is not None
    if out["reject"] is None and p_land is None:
        out["reject"] = (f"착탄 불가 — 릴리즈 상태가 목표 높이 z={p_target[2]:+.3f}m "
                         f"에 도달하지 못함 (|v|={np.linalg.norm(v_eff):.2f} m/s)")
    return out


# =========================================================================
# Skill
# =========================================================================

class ThrThrowSkill(RobustThrowSkill):
    """THR 계획 모델(nlp/dt/phy)로 던지는 `RobustThrowSkill` 변형.

    부모의 `execute` (ambush 픽 사이클) 와 `tests/static_pick_throw*.py` 가 쓰는
    공개 스테이지 시그니처를 그대로 유지한다:

        plan_nlp_throw(start_joint, T_grasp, p_target) -> (res, lift) | None
        build_throw_trajectory(grasp_joint, res, lift, next_grasp=, chain_park=)

    이름은 NLP 시절 그대로 두었다 (호출측을 안 고치려고) — 안에서 도는 것만
    모델별 traj_fn 이다.
    """

    name = "thr_throw"

    def __init__(self, ctx, model: str = "nlp", weights: "Optional[str]" = None,
                 traj_fn=None) -> None:
        super().__init__(ctx)
        if model not in thr_planners.MODELS:
            raise ValueError(f"알 수 없는 모델: {model} "
                             f"(가능: {', '.join(thr_planners.MODELS)})")
        self.model = model
        self.weights = weights
        # traj_fn 주입: 실기 데이터 수집(tools/collect_real_throws_gp8.py)이 액션
        # 게인 α 를 먹인 rollout 을 쓰기 위해 갈아끼운다. None 이면 모델 기본값.
        # 주입해도 게이트·lift·감속·dispatch 는 전부 그대로 탄다.
        self._traj_fn = (traj_fn if traj_fn is not None
                         else thr_planners.get_traj_fn(model, weights, logger=ctx.log))
        self._last_thr_meta: dict = {}
        self._warned_tool = False

    # ------------------------------------------------------------------
    def _warn_tool_mismatch(self) -> None:
        """robots/gp8.py 의 TCP 길이가 THR 기하와 다르면 한 번 경고."""
        if self._warned_tool:
            return
        self._warned_tool = True
        try:
            home_x = float(np.asarray(self.ctx.robot.home_ee, float)[0, 3])
        except Exception:                                   # noqa: BLE001
            return
        thr_x = 0.38 + thr_planners.GP8_DIMS["d6"] + thr_planners.GP8_DIMS["tool"]
        if abs(home_x - thr_x) > 1e-6:
            self.ctx.log.warn(
                f"기하 불일치: robots/gp8.py home_ee x={home_x:.3f} m vs THR 기준 "
                f"{thr_x:.3f} m (tool {thr_planners.GP8_DIMS['tool']:.3f}). 던지기 "
                f"조준·게이트는 THR 기준, 픽은 gp8.py 기준으로 돈다 — 스킬 docstring 참고")

    # ------------------------------------------------------------------
    # 계획
    # ------------------------------------------------------------------
    def plan_nlp_throw(self, start_joint: np.ndarray, T_grasp: np.ndarray,
                       p_target: np.ndarray):
        """THR traj_fn 호출 → (res, lift). 실현 불가면 None (사유는 로그)."""
        ctx = self.ctx
        self._warn_tool_mismatch()
        p_target = np.asarray(p_target, float).ravel()

        # 던지기 시작 TCP — THR 규약: 흡착점 xy, z = P_START_Z (0.20 m).
        # nlp 만 실제로 쓴다 (dt 는 home 자세 고정, phy 는 릴리즈점을 해석적으로 결정).
        p_start = np.array([T_grasp[0, 3], T_grasp[1, 3], thr_planners.P_START_Z])

        try:
            plan = self._traj_fn(p_target, p_start)
        except (ValueError, RuntimeError, AssertionError) as e:
            ctx.log.warn(f"{self.model} 계획 실패: {type(e).__name__}: {e}")
            return None

        ts = np.asarray(plan["ts"], float).ravel()
        Qp = np.asarray(plan["Q"], float)                     # (N, 6) planner
        Qdp = (np.asarray(plan["Qd"], float) if plan.get("Qd") is not None
               else np.gradient(Qp, ts, axis=0))

        # planner → robot 규약 (부모와 동일한 _PLANNER_SIGN)
        arc_q = (Qp * _PLANNER_SIGN).T                        # (6, N)
        arc_qd = (Qdp * _PLANNER_SIGN).T

        # 릴리즈 시각: 윈도우가 있으면 중앙 (nlp — ±rt/2 여유), 없으면 t_rel (dt/phy)
        win = plan.get("release_window")
        if win is not None:
            t_rel = 0.5 * (float(win[0]) + float(win[1]))
            half = 0.5 * (float(win[1]) - float(win[0]))
        elif plan.get("t_rel") is not None:
            t_rel, half = float(plan["t_rel"]), 0.0
        else:
            ctx.log.warn(f"{self.model} 계획에 release 시각이 없다 — 거부")
            return None
        i_rel = int(np.argmin(np.abs(ts - t_rel)))
        if abs(float(ts[i_rel]) - t_rel) > 1.5 * thr_planners.DT:
            ctx.log.warn(f"{self.model}: release 시각 {t_rel:.4f}s 가 궤적 격자에 "
                         f"없다 (최근접 {ts[i_rel]:.4f}s) — 거부")
            return None

        # 아크는 릴리즈까지 + 그 뒤 꼬리 전부를 그대로 쓴다 (nlp/phy 는 정지로,
        # dt 는 팔로우스루로 끝난다). 게이트는 릴리즈 시점 상태로 판단하므로
        # 검사용 아크는 릴리즈까지로 자른다.
        chk = check_arc(arc_q[:, :i_rel + 1], arc_qd[:, :i_rel + 1],
                        ts[:i_rel + 1], p_target, ctx.robot, self.model)
        for w in chk["warns"]:
            ctx.log.warn(f"{self.model} 경고: {w}")
        if chk["reject"] is not None:
            ctx.log.warn(f"{self.model} 거부: {chk['reject']}")
            return None

        # ---- lift: 픽 자세 → 계획 시작 자세 (rest-to-rest) ----
        q_start_p = np.asarray(plan.get("q_start", Qp[0]), float)
        q_start = q_start_p * _PLANNER_SIGN
        zero6 = np.zeros(6)
        l_traj, l_vel, l_ts = trajectory(
            np.asarray(start_joint, float), zero6, q_start, zero6,
            ctx.M1[:6], ctx.M2[:6], hertz=ctx.cfg.TRAJ_HZ)
        bad = self._seam_check(l_traj, np.asarray(start_joint, float), q_start, "lift")
        if bad is not None:
            ctx.log.warn(f"{self.model} 거부: {bad}")
            return None
        # 계획 아크의 첫 샘플이 정말 q_start 인지 (규약 확인 — 어긋나면 점프한다)
        d0 = float(np.max(np.abs(arc_q[:, 0] - q_start)))
        if d0 > 1e-3:
            ctx.log.warn(f"{self.model} 거부: 계획의 q_start 와 아크 첫 샘플이 "
                         f"{np.rad2deg(d0):.2f}° 어긋난다")
            return None

        sens = self.release_sensitivity(arc_q, arc_qd, i_rel, p_target[2])
        res = dict(
            kind="thr", model=self.model,
            arc=(arc_q, arc_qd, ts), i_rel=i_rel,
            t_f=float(ts[-1]),               # 아크 길이 [s] (호출측 로그 호환)
            J=chk["err"],                    # 호출측이 "J=" 로 찍는 값 = 예측 착지오차 [m]
            release_time=2.0 * half,         # nlp 윈도우 길이 (dt/phy 는 0)
            t_rel=float(ts[i_rel]), half_window=half,
            d_land=chk["d_land"], err=chk["err"], p_land=chk["p_land"],
            v_eff=chk["v_eff"], sens_mm_per_10ms=sens,
            peak_vel_ratio=chk["peak_vel_ratio"], plan_info=plan.get("info", {}))
        ctx.log.info(
            f"{self.model} throw: 아크 {ts[-1]:.3f}s ({len(ts)} 샘플), "
            f"release t={ts[i_rel]:.3f}s"
            + (f" (윈도우 ±{half * 1e3:.0f}ms)" if half > 0 else " (윈도우 없음)")
            + f", lift {l_ts[-1]:.2f}s | 예측 착지 {chk['d_land']:.3f} m "
              f"(목표 오차 {chk['err'] * 100:.1f} cm, 진단만) | "
              f"|v_release|={np.linalg.norm(chk['v_eff']):.2f} m/s, "
              f"민감도 {sens:.0f} mm/10 ms | 속도 peak "
              f"{chk['peak_vel_ratio']:.2f}× (URDF 한계 대비)")
        return res, (l_traj, l_vel, l_ts)

    # ------------------------------------------------------------------
    @staticmethod
    def release_sensitivity(arc_q, arc_qd, i_rel: int, land_z: float) -> float:
        """릴리즈가 한 샘플(≈4 ms) 어긋났을 때 착지가 움직이는 거리 [mm/10 ms]."""
        base, _, _ = _predict_landing(arc_q[:, i_rel], arc_qd[:, i_rel], land_z)
        j = max(0, i_rel - 1)
        if base is None or j == i_rel:
            return float("nan")
        prev, _, _ = _predict_landing(arc_q[:, j], arc_qd[:, j], land_z)
        if prev is None:
            return float("nan")
        dt = max(1e-9, float(thr_planners.DT))
        return float(np.linalg.norm(prev - base) / dt * 0.01 * 1000.0)

    @staticmethod
    def _seam_check(traj: np.ndarray, q0, q1, tag: str,
                    tol_start: float = 1e-6, tol_end: float = 0.02) -> "Optional[str]":
        """`trajectory()` 결과가 실제로 q0 → q1 을 잇는지 확인.

        `_trajectory_1d` 는 |N| > M1 일 때 None 을 리턴하고 호출측이 그 관절 행을
        0 으로 남긴다 (trajectory_primitive.py:126) — 조용히 "그 관절을 0 rad 로
        보내라"는 명령이 된다. 진입 속도가 있는 감속 세그먼트에서 특히 위험하다.
        """
        q0 = np.asarray(q0, float).ravel()
        q1 = np.asarray(q1, float).ravel()
        zero_rows = np.all(traj == 0.0, axis=1) & ~(
            (np.abs(q0) < 1e-12) & (np.abs(q1) < 1e-12))
        if zero_rows.any():
            j = int(np.argmax(zero_rows))
            return (f"{tag} 세그먼트: J{j + 1} 행이 전부 0 — trajectory() 가 이 관절을 "
                    f"만들지 못했다 (진입 속도가 M1 초과). 그대로 보내면 J{j + 1} 을 "
                    f"0°로 끌고 간다 ({np.rad2deg(q0[j]):+.1f}° → {np.rad2deg(q1[j]):+.1f}°)")
        d0 = float(np.max(np.abs(traj[:, 0] - q0)))
        d1 = float(np.max(np.abs(traj[:, -1] - q1)))
        if d0 > tol_start or d1 > tol_end:
            j = int(np.argmax(np.abs(traj[:, -1] - q1)))
            return (f"{tag} 세그먼트 생성 실패 (시작 오차 {np.rad2deg(d0):.3f}°, "
                    f"끝 오차 {np.rad2deg(d1):.2f}° @ J{j + 1})")
        return None

    # ------------------------------------------------------------------
    # 조립 + dispatch
    # ------------------------------------------------------------------
    def build_throw_trajectory(self, grasp_joint: np.ndarray, res: dict, lift,
                               next_grasp: "Optional[np.ndarray]" = None,
                               chain_park: "Optional[np.ndarray]" = None) -> None:
        """lift + 계획 아크 + 감속/체인 을 이어 붙여 timed-release 로 디스패치."""
        ctx = self.ctx
        l_traj, l_vel, l_ts = lift
        arc_q, arc_qd, arc_t = res["arc"]

        # 아크 첫 샘플은 lift 끝과 같은 상태라 버리고 시계만 잇는다.
        a_traj, a_vel = arc_q[:, 1:], arc_qd[:, 1:]
        a_ts = arc_t[1:] + l_ts[-1]
        i_rel_concat = l_traj.shape[1] + int(res["i_rel"]) - 1

        # ---- 감속 + 체인 ----
        # M1 은 --vel-scale 로 줄인 ctx.M1 을 쓰지 않는다: 진입 속도가 M1 을 넘으면
        # trajectory() 가 그 관절 행을 0 으로 남긴다. 스윙 속도로 감속할 수 있어야
        # 하므로 **URDF 관절 속도 한계**를 쓴다 (스트리밍 상한은 고려하지 않는다 —
        # 위 VEL_GATE_MARGIN 주석 참고).
        q_end, qd_end = a_traj[:, -1], a_vel[:, -1]
        M1_stop = (np.asarray(ctx.robot.velocity_limits, float)
                   * ctx.cfg.JOINT_VEL_LIMIT_SCALE)
        M2_stop = M1_stop * ctx.cfg.JOINT_ACCEL_LIMIT_SCALE
        if np.any(np.abs(qd_end) > M1_stop):
            j = int(np.argmax(np.abs(qd_end) - M1_stop))
            ctx.log.error(
                f"{self.model} throw ABORTED (not dispatched): 아크 종료 속도 J{j + 1} "
                f"{np.rad2deg(qd_end[j]):+.0f}°/s 가 감속 한계 "
                f"{np.rad2deg(M1_stop[j]):.0f}°/s 를 넘어 정지 세그먼트를 만들 수 없다")
            return

        chain_target, chain_dest = self._chain_target(q_end, next_grasp, chain_park)
        zero6 = np.zeros(6)
        c_traj, c_vel, c_ts = trajectory(q_end, qd_end, chain_target[:6], zero6,
                                         M1_stop, M2_stop, hertz=ctx.cfg.TRAJ_HZ)
        bad = self._seam_check(c_traj, q_end, chain_target[:6], "감속/체인")
        if bad is not None:
            ctx.log.error(f"{self.model} throw ABORTED (not dispatched): {bad}")
            return
        c_traj, c_vel = c_traj[:, 1:], c_vel[:, 1:]
        c_ts_shift = c_ts[1:] + a_ts[-1]

        traj = np.concatenate((l_traj, a_traj, c_traj), axis=1)
        vel = np.concatenate((l_vel, a_vel, c_vel), axis=1)
        ts = np.concatenate((l_ts, a_ts, c_ts_shift))
        assert traj.shape[1] == vel.shape[1] == ts.shape[0], (
            f"traj/vel/timestep length mismatch: "
            f"{traj.shape[1]}/{vel.shape[1]}/{ts.shape[0]}")
        if not np.all(np.diff(ts) > 0):
            k = int(np.argmin(np.diff(ts)))
            ctx.log.error(
                f"{self.model} throw ABORTED (not dispatched): timestep 비단조 @ {k} "
                f"({ts[k]:.4f} → {ts[k + 1]:.4f}) — 스트리머 Hermite 리샘플이 깨진다")
            return

        # ---- 릴리즈 인덱스: 시간 기준. RELEASE_LEAD 부호 규약은 부모와 동일 —
        #      양수 = 그만큼 일찍 명령(밸브 지연 보정), 음수 = 늦게.
        t_rel = float(ts[i_rel_concat])
        t_cmd = min(max(t_rel - float(ctx.cfg.RELEASE_LEAD), float(ts[0])), float(ts[-1]))
        release_idx = int(np.argmin(np.abs(ts - t_cmd)))
        half = float(res.get("half_window", 0.0))
        if half <= 0.0 and release_idx > i_rel_concat:
            ctx.log.warn(
                f"릴리즈 명령이 계획 시점을 지나 뒤로 밀렸다 "
                f"(RELEASE_LEAD={ctx.cfg.RELEASE_LEAD:+.2f}s) — {self.model} 은 릴리즈 "
                f"윈도우가 없어 그만큼 착지가 어긋난다. GP8_RELEASE_LEAD 확인.")

        # ---- SAFETY GATE 1: 관절 한계 (연결·변환된 전체 궤적) ----
        jl = np.asarray(ctx.robot.joint_limits, float)
        viol = (traj < jl[:, :1] - 1e-6) | (traj > jl[:, 1:2] + 1e-6)
        if viol.any():
            j, k = (int(v) for v in np.argwhere(viol)[0])
            ctx.log.error(
                f"{self.model} throw ABORTED (not dispatched): robot-frame joint "
                f"{j + 1} limit violation at sample {k} "
                f"({np.rad2deg(traj[j, k]):+.1f}°, limits "
                f"[{np.rad2deg(jl[j, 0]):+.1f}, {np.rad2deg(jl[j, 1]):+.1f}]°, "
                f"{self._segment_of(k, l_traj.shape[1], a_traj.shape[1])} 구간)")
            return

        # ---- SAFETY GATE 2: Cartesian 엔벨로프 ----
        # 던지기 아크는 THR fk (tool 0.240) 로, 픽/체인 구간은 로봇 FK 로 본다 —
        # 각 구간을 실제로 만든 기하와 같은 FK 로 검사한다.
        n_s = traj.shape[1]
        n_lift, n_arc = l_traj.shape[1], a_traj.shape[1]
        tcp = np.empty((3, n_s))
        for k in range(n_s):
            if n_lift <= k < n_lift + n_arc:
                tcp[:, k] = fk_pos(traj[:, k] * _PLANNER_SIGN)
            else:
                tcp[:, k] = np.asarray(
                    ctx.robot.forward_kinematics(traj[:, k]), float)[:3, 3]
        bad_list: list = []
        # z 바닥 검사는 팔이 바닥을 '확실히' 벗어난 뒤부터 건다. 출발점(프레스 자세)이
        # 마침 MIN_TCP_Z 근처면(--points z=0.04 == MIN_TCP_Z 가 기본), lift 첫 샘플의
        # 0.2 mm 수준 수치 딥까지 위반으로 잡혀 던지기가 통째로 막힌다 (실측: phy
        # 3건 중 2건). 원래 의도는 '한 번 올라갔다가 다시 내려오는 꼬리 다이브'를
        # 잡는 것이므로, ARM 여유(2 cm)만큼 올라간 뒤부터 검사를 arm 한다.
        cleared = np.nonzero(tcp[2] > MIN_TCP_Z + Z_ARM_MARGIN)[0]
        if cleared.size == 0:
            bad_list.append((0, f"궤적 전체가 z ≤ {MIN_TCP_Z + Z_ARM_MARGIN:.3f}m "
                                f"(lift 가 바닥을 못 벗어남)"))
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
        if bad_list:
            k, why = min(bad_list)
            n_viol = int(np.count_nonzero(
                (tcp[0] <= MIN_TCP_X) | (tcp[2] <= MIN_TCP_Z)))
            head = ("ABORTED (not dispatched)" if CART_BLOCKING
                    else "Cartesian 위반 — **보고만** 하고 dispatch 진행")
            ctx.log.error(
                f"{self.model} throw {head}: {why} @ sample {k}/{n_s} "
                f"(t={ts[k]:.3f}s, {self._segment_of(k, n_lift, n_arc)} 구간, "
                f"위반 샘플 {n_viol}개)")
            ctx.log.error(
                "  위반 지점 관절 (robot frame, deg): "
                + ", ".join(f"J{j + 1}={np.rad2deg(traj[j, k]):+.1f}" for j in range(6)))
            if CART_BLOCKING:
                return

        ctx.log.info(
            f"{self.model} throw dispatch: lift {n_lift} + arc {n_arc} + "
            f"decel/chain {c_traj.shape[1]} samples ({ts[-1]:.2f}s), "
            f"release step {release_idx} (t={ts[release_idx]:.3f}s, 계획 t={t_rel:.3f}s, "
            f"lead {ctx.cfg.RELEASE_LEAD:+.2f}s, "
            + (f"윈도우 ±{half * 1e3:.0f}ms" if half > 0
               else f"윈도우 없음 — 민감도 {res['sens_mm_per_10ms']:.0f} mm/10 ms")
            + f"), chain->{chain_dest}")

        ctx.traj_ctrl.send_trajectory_queue_with_timed_release(
            traj, vel, ts, final_joint=chain_target, release_index=release_idx)
        self._last_thr_meta = {
            "model": self.model, "arc_T": res["t_f"], "lift_T": float(l_ts[-1]),
            "t_rel": t_rel, "half_window": half, "d_land": res["d_land"],
            "err": res["err"], "v_release": float(np.linalg.norm(res["v_eff"])),
            "sens_mm_per_10ms": res["sens_mm_per_10ms"],
            "peak_vel_ratio": res["peak_vel_ratio"], "release_idx": release_idx,
            "n_steps_traj": int(traj.shape[1]),
        }
        # 부모의 CSV 로거가 참조하는 필드도 채워 둔다.
        self._last_throw_meta = {
            "T": 2.0 * half, "lift_T": float(l_ts[-1]), "t_f": res["t_f"],
            "t_star": t_rel - float(l_ts[-1]), "J": res["J"],
            "release_idx": release_idx, "n_steps": int(traj.shape[1]),
        }

    # ------------------------------------------------------------------
    def _chain_target(self, q_end: np.ndarray, next_grasp, chain_park):
        """부모와 동일한 우선순위: next park > over next grasp > idle > standby."""
        ctx = self.ctx
        if chain_park is not None:
            return np.asarray(chain_park, float).copy(), "next action-start park"
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
        """사이클 1행을 PICK_LOG_CSV 에 append (THR 진단 컬럼 포함)."""
        ctx = self.ctx
        path = ctx.cfg.PICK_LOG_CSV
        if not path:
            return
        lt = getattr(ctx.traj_ctrl, "last_throw", {}) or {}
        m = self._last_thr_meta or {}
        son = getattr(ctx.traj_ctrl, "last_suction_on_t", None)
        t0, trel = lt.get("throw_start"), lt.get("release_wall")

        def _d(a, b):
            return round(a - b, 4) if (a is not None and b is not None) else ""

        row = {
            "iso_time": datetime.datetime.now().isoformat(timespec="milliseconds"),
            "class": target.class_name, "skill": self.name,
            "model": m.get("model", self.model),
            "belt_mps": round(ctx.conveyor.current, 4),
            "suction_on_t": round(son, 4) if son else "",
            "throw_start_t": round(t0, 4) if t0 else "",
            "release_t": round(trel, 4) if trel else "",
            "on_to_throwstart_s": _d(t0, son),
            "throwstart_to_release_s": _d(trel, t0),
            "arc_T_s": round(m.get("arc_T", 0.0), 3),
            "lift_T_s": round(m.get("lift_T", 0.0), 3),
            "half_window_s": round(m.get("half_window", 0.0), 4),
            "d_land_m": round(m.get("d_land", 0.0), 4),
            "err_m": round(m.get("err", 0.0), 4),
            "v_release_mps": round(m.get("v_release", 0.0), 3),
            "sens_mm_per_10ms": round(m.get("sens_mm_per_10ms", 0.0), 1),
            "peak_vel_ratio": round(m.get("peak_vel_ratio", 0.0), 3),
            "release_idx": m.get("release_idx", ""),
            "n_steps_traj": m.get("n_steps_traj", ""),
            "io_ms": round(lt["io_ms"], 1) if lt.get("io_ms") is not None else "",
        }
        self._append_csv_row(path, row, ctx.log)

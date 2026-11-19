"""grasping 플래너 — 인식된 물체들의 **처리 순서**를 정하고, 던지기 모듈로 물체 간 궤적을 뽑아 **전체 궤적**을 만든다.

    인식 모듈 (물체 id·클래스·흡착점, 벨트 속도)
        │  objects_from_perception()  → ObjectState(id, position@t0, target=bin 조준점)
        ▼
    plan_full_sequence(robot, objects, belt_speed)
        ├ Part 3  sequence_planner.plan_sequence  — 순서 (전 순열 / greedy), edge 캐시
        ├ Part 2  sequence_planner.solve_moving_transition — 작업영역 창·대기·도착 일관성
        ├ Part 1  ThrowPrimitive  ← 우리 던지기 모듈
        │     빈손 → 첫 물체    : 접근 세그먼트 (trajectory 프리미티브, 도착 시각 고정점)
        │     물체 i → 물체 j   : [픽: 접촉 → 눌림(press) → 상승, 내내 벨트 추종] + [던지기 아크: 종점 = 물체 j **위 hover** 에서 벨트 추종
        │                        상태(움직이는 종점, solve 1회)] + [추종 하강: xy = 물체(벨트 속도), z 만 v_desc 로 내려가 접촉]
        │                        (2026-09-21 사용자: 수직 하강이 아니라 벨트 속도로 물체를 따라가며 내려간다)
        │     물체 j 가 아직 상류 : 아크 종점 = **j 가 처음 잡을 수 있게 되는 자리(p_in) 위 hover, 정지** → 거기서 대기 →
        │                        j 가 오면 벨트 속도로 가속하며 하강해 p_in 에서 접촉 (2026-09-19 사용자 규약)
        │     마지막 물체       : 던지고 **기본 자세(home)로 돌아와 정지** (다음 물체 없음)
        └ 세그먼트를 시간순으로 이어 FullPlan(t, q, qd, suction_events, order)

원본 순서 플래너와 어긋났던 점을 정리한 방식은 sequence_planner.py 헤더 참고. 여기서 추가로 정한 것:
  - 아크는 **물체 표면에서** 하강 속도(+벨트 추종)로 끝난다 — 옆으로 벨트 속도에 맞춰 물체를 따라가며 내려와 닿는다
    (2026-09-19 사용자: settle 같은 멈추는 순간은 없다). 픽 세그먼트는 그 속도를 이어받아 press_depth(1 cm) 만큼 눌리며
    등감속 → 바로 등가속 상승해 START_LIFT 에서 LIFT_SPEED 에 도달한다 (z 속도가 0 을 스치기만 하고 dwell 없음).
    접촉 순간 suction ON. 상승 중 벨트 추종 성분은 선형으로 0 이 되어 던지기 시작 경계(수직 상승)와 정확히 이어진다.
  - 대기(작업영역 진입 전)는 잡은 물체를 든 채 벨트를 따라가는 hold 로 실행한다.
  - 시각은 계획 시각 0 기준. 인식 시각 → 지금까지의 지연과 계획 소요 시간은 `latency_s` 로 넣어 물체를 미리 흘려 둔다.

ROS 무관 (numpy/casadi + robot/gp8 운동학). 단독 실행: python -m gp8_control.grasping.throw_sequence
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from gp8_control.grasping.sequence_planner import (CircularWorkspace, MovingPoint, ObjectState, PlanResult,
                                                   PrimitiveResult, SourceState, plan_sequence)
from gp8_control.throwing.nlp import planner as _nlp_planner
from gp8_control.throwing.throw_planner import (BELT_DIR, DESCEND_SPEED, GRID_DT, LIFT_SPEED, START_LIFT, TRAVEL_Z, UP,
                                                ThrowBoundary, ThrowPlan, belt_joint_velocity, grid_ceil, grid_round,
                                                plan_throw, to_planner)
from gp8_control.throwing.trajectory_primitive import trajectory

# 눌림 (2026-09-21 사용자): 물체 높이와 무관하게 **모든 물체를 절대 z PRESS_Z_FLOOR 까지** 찍어 누른다.
# 깊이 = 윗면 z − PRESS_Z_FLOOR. PRESS_DEPTH 는 상한(인식 z 가 비정상으로 높을 때 보호)일 뿐이다.
# 깊을수록 눌림 감속이 오히려 완만하다 (a = v_d²/2d): 물체 윗면 0.02 기준 하강 1.2 m/s 에서 −0.02 면 깊이 4.0 cm·18 m/s²,
# −0.025 면 깊이 4.5 cm·16 m/s² — 둘 다 관절 가속 한계(20×) 안이다.
PRESS_Z_FLOOR: float = float(os.environ.get("GP8_PRESS_Z_FLOOR", "-0.025"))   # 2026-09-27 사용자: −0.025 까지
PRESS_DEPTH: float = float(os.environ.get("GP8_PRESS_DEPTH", "0.06"))   # 깊이 상한 [m]
PRESS_Z_MIN = PRESS_Z_FLOOR                                             # (호환 이름)
HOLD_S: float = 0.0                    # 눌린 채 벨트 추종 dwell [s] (기본 0 = 멈춤 없음; 석션 형성 시간이 필요하면 지정)
# 세그먼트 샘플 주기 = 로봇 스트림 격자 (500 Hz / 2 ms). 모든 세그먼트 길이를 GRID_DT 배수로 맞춰 전체 궤적의 샘플이 격자에
# 정확히 놓이게 한다 → 스트리머(robot_base._stream_trajectory)가 보간·클램프 없이 그대로 내보낸다 (2026-09-21 사용자: 보간 제거).
HZ_SEG: float = 1.0 / GRID_DT
AIM_UP: float = float(os.environ.get("GP8_THR_AIM_UP", "0.0"))   # 조준면 = bin z + 이 값. 2026-09-22 사용자: 0 — bin z 자체(−0.06)를 조준
HOVER: float = 0.05                    # (구) 상대 hover 높이 — 지금은 TRAVEL_Z 절대 높이를 쓴다 (throw_planner.TRAVEL_Z)
# suction ON 은 접촉 순간이 아니라 **내려가는 중** 이만큼 먼저 (2026-09-21 사용자). 진공이 형성될 시간(MIN_SUCTION_HOLD 0.3 s)을
# 하강 중에 벌어 접촉 즉시 잡힌다. 직전 릴리즈(OFF) 보다 0.05 s 이상 뒤로는 못 당긴다.
SUCTION_LEAD_S: float = float(os.environ.get("GP8_SUCTION_LEAD", "0.3"))   # (미사용, 2026-09-28: ON 은 하강 시작 — _on_at_descend)
# 움직이는 종점의 선형화 기준 시각 t_ref 와 NLP 가 낸 t_f 의 허용 차 [s]. 넘으면 t_f 시각의 자세로 다시 선형화해 재해석
# (아크 끝 자세 = 하강 시작 IK 자세, 이음새 불연속 제거). 0.12 m/s·0.03 s = 3.6 mm 안이면 무시할 수 있다.
RELIN_S: float = float(os.environ.get("GP8_THR_RELIN_S", "0.03"))
# 다음 물체 도달 판정 시각 (2026-10-27 사용자: "그러면 예측한 걸로 해 다시"): 평균 t_ref 가 아니라 이 쌍(던지기 시작 상태 → 다음 물체 끝점)의
# 예측 던지기 시간 t_f — t_ref 시각의 끝점으로 던지기 시간 모델(learning.throw_time_model, 대리 프리미티브와 같은 MLP)이 예측한 시각에서 다시
# 판정하고, 예측이 그 끝점 시각과 RELIN_S 넘게 다르면 새 끝점으로 한 번 더 예측·판정 (최대 JUDGE_ITERS 회). 순서 결정(대리 프리미티브)과 실제 꼬리 계획(NLP)이 같은 판정을 쓴다.
# GP8_JUDGE_TF=0 이면 예전처럼 평균 t_ref 에서만 판정.
JUDGE_TF: bool = os.environ.get("GP8_JUDGE_TF", "1") != "0"
JUDGE_ITERS: int = 2
WARM_PREV_M: float = float(os.environ.get("GP8_THR_WARM_PREV_M", "0.06"))   # 직전 해를 warm start 로 쓰는 시작점 거리 상한 [m]
HOVER_DIRECT_M: float = float(os.environ.get("GP8_HOVER_DIRECT_M", "0.02"))   # 던지기 끝 hover TCP 와 물체 xy 가 이보다 멀면 바로 하강 대신
                                                                                 # 정지→접근 (hold 로 늦어져 exact 종점이 낡은 경우; 2026-09-22 random7 id 9·10)
# 잡기 프로파일 (2026-10-14 사용자: "관절 한계만 고려해서 최고 하강상승 속도를 매번 위치마다 적용"):
#   topp  = 위치(반지름 칸)마다 관절 속도·가속 한계만으로 푼 시간 최적 하강·눌림·상승 (grasping/pick_profile.py). hover 에서 z 속도
#           0 으로 출발하고 z_top 에서 +LIFT_SPEED 로 끝나 warm DB·던지기 경계는 그대로다. 못 푸는 위치(도달 한계 근처)는 const.
#   const = 이전: 등속 하강 DESCEND_SPEED (첫 샘플부터) + 등감속 눌림 + 등가속 상승 LIFT_SPEED.
PICK_PROFILE: str = os.environ.get("GP8_PICK_PROFILE", "topp").strip().lower()
# 분리 방식 (2026-10-30 사용자): 하강·접촉·상승 = 잡기 LP (grasping.sep_pick) 후계산, 던지기 NLP (throwing.sep_throw) 가 상승 끝 수직 속력과
# 다음 물체 하강 시작 수직 속력을 정한다. 던지기는 다음 물체 위 hover 에서 그 하강 속력으로 끝나고, 다음 잡기 LP 가 거기서 이어 받는다.
SEP: bool = os.environ.get("GP8_SEP", "0") not in ("0", "", "false")
SEP_VOUT_DEF: float = float(os.environ.get("GP8_SEP_VOUT", "2.0"))   # 던지기를 아직 모를 때(head 의 대비 잡기) 상승 끝 속력
# 상승 끝 속력 추정의 추가 여유 [m/s] (11-02): 정확 제약 DB(w100ex)의 v_out 은 표 상한에 붙어 있어, 표 상한(VLIM_MARGIN 0.03 뺀 값)
# 그대로면 head 의 잡기 LP(표 추정 길이 3개, 이분법 없음)가 Isaac 10 회에서 136 번 안 풀려 정지했다. 0.03 더 빼면 134/136, 0.05 면 136/136.
# 원인은 LP 한계를 1 cm 칸 아래쪽 끝 반경에서 계산한 것 (106/136 건이 LP 자신의 hover 속도 한계 초과) — 실제 반경 수직선으로 고친 뒤
# 여유 0 으로도 136/136 (11-02 사용자: "여유 0 해"). 풀이 시간은 중앙 0.28 → 0.45 s, 최대 0.74 → 1.34 s.
SEP_VOUT_MARGIN: float = float(os.environ.get("GP8_SEP_VOUT_MARGIN", "0"))
SEP_CORR: bool = os.environ.get("GP8_SEP_CORR", "0") not in ("0", "", "false")   # 분리 잡기 명령 보정 (sep_refine) — 11-02 사용자: 실물에 넣지 않음 (기본 끔, 켜도 CORR_ITERS 0 이면 접촉 오차만 잼)
# 잡기 먼저 (2026-10-30 사용자): 잡기 LP (v_out = 이웃 해) → 던지기 NLP 가 그 명령을 이력으로 바로 이어받아 실제 TCP 높이 제약 (sep_throw "hist")
SEP_PICKFIRST: bool = os.environ.get("GP8_SEP_PICKFIRST", "0") not in ("0", "", "false")
# 대기점 종착도 분리 던지기 NLP 로 (11-03 사용자): 다음 물체가 던지기 끝에 아직 도달 범위 밖이면 종전엔 예전 던지기 NLP (정지 종점) 로 넘어가
# 마감을 다 쓰고도 수렴 못 하고, 대기 시각을 맞추려 한 번 더 풀어 꼬리가 늦었다 (fifo0 11-02: 0.03·0.17 s 늦어 물체를 든 채 정지).
# 분리 NLP 의 정지 종점 판 (끝 = 대기점 hover, 속도 0, v_in = 0) 을 던지기 시간 상한 (대기점 → 하강 시작에 맞음) 과 함께 한 번 푼다.
SEP_STOP_END: bool = os.environ.get("GP8_SEP_STOP_END", "1") not in ("0", "", "false")
# 11-04 사용자 (논문 version2 식 6): 통합 사이클 — 잡아 올린 상태에서 [던지기 i + 잡기 i+1] 를 명령 B-스플라인 하나로 (throwing.cycle_nlp)
CYCLE: bool = os.environ.get("GP8_SEP_CYCLE", "0") not in ("0", "", "false")
# 11-07 사용자 "시뮬 하나 돌려봐 cold로": hover 꼬리 = jcyc1 사이클 (관절식 + A안 연장, 하강 시작 경계). 풀이는 ~/rge_jc/jc_bridge.py (초기값마다 한 프로세스)
JCYC: bool = os.environ.get("GP8_JCYC", "0") not in ("0", "", "false")
# 도달 범위 (jcycle): hover 도착 (사이클 시작) 부터 JC_PW 초 동안 물체 위 수직 툴 IK 가 윗면·hover 두 높이에서 있고 사이클 관절 상자 안.
# JC_PW = 하강·접촉·상승 (실제 t_s ≤ 0.6 s) + 불감 D + 명령 앞섬 (L − D) 여유. 다음 물체는 사이클 끝 (t_c ∈ JC_TC) 부터 연장 JC_EXT + 그 물체의 잡기 창까지.
JC_PW: float = float(os.environ.get("GP8_JCYC_PW", "1.15"))               # 정지 출발: 벨트 추종 0.30 s + 하강·접촉·상승 ≤ 0.6 + 지연 여유
JC_FOLLOW_S: float = float(os.environ.get("GP8_JCYC_FOLLOW_S", "0.30"))   # 정지 출발 head: hover 도착 뒤 벨트 추종 (실제가 hover 에 와서 따라가도록, ≥ W1 + W2)
JC_TC: Tuple[float, float] = (1.10, 1.50)
JC_EXT: float = 0.40
JC_IMPL: str = os.environ.get("GP8_JC_IMPL", "")                                # 11-08 v3: NLP 없는 해석적 사이클 (jc_v3)


def _jc_impl():
    if JC_IMPL == "v3":
        import jc_v3 as JB
    else:
        import jc_bridge as JB
    return JB


if JC_IMPL == "v3":                                                            # 11-08 v3 사이클 (~1.0 s, 연장 없음): 다음 물체 도달 판정 창을 사이클 길이에 맞춘다 (NLP 판 ≈3 s 창이 사이클을 끊음)
    JC_TC = (float(os.environ.get("GP8_V3_TC0", "0.8")), float(os.environ.get("GP8_V3_TC1", "1.2")))
    JC_EXT = 0.0


JC_DB: bool = bool(os.environ.get("GP8_JC_DB"))                               # 11-08: 사이클 DB 범위 = 도달 범위 (jc_bridge.db_cover_i / db_covered)
_JC_CERT: Dict[Any, np.ndarray] = {}                                          # 다음 물체 id → 앞 사이클의 보증 (연장) 명령 (다음 사이클 첫 초기값)
CYCLE_N_INIT: int = int(os.environ.get("GP8_CYCLE_N_INIT", "3"))         # cold 초기값 수 (던지기 INIT_VARIANTS 앞에서부터)
# 사이클 DB (11-05): jsonl 파일들 (os.pathsep — 리눅스 ":" — 로 구분, scratchpad cyc_db.py / cyc_grow.py 출력). 칸 = (실제 시작 xy, 다음 물체 xy, bin) 의 가장 가까운
# 통과 해 CYCLE_DB_K 개를 초기값으로 (첫 제어점만 지금 시작으로), 모두 실패하면 cold. 비우면 cold 만.
CYCLE_DB: str = os.environ.get("GP8_CYC_DB", "")
CYCLE_DB_K: int = int(os.environ.get("GP8_CYC_DB_K", "2"))
CYCLE_DB_OUTER: int = int(os.environ.get("GP8_CYC_DB_OUTER", "0"))     # DB 해 전체 (x·λ) 로 시작하면 바깥 BFGS 횟수 (0 = θ 고정 안쪽 한 번, ~0.5 s)
CYCLE_LAND_TOL: float = float(os.environ.get("GP8_CYC_LAND_TOL", "0.02"))  # 사이클 해의 착지 예측 오차 상한 [m] — 넘으면 실행하지 않음
CYCLE_DB_MAXD: float = float(os.environ.get("GP8_CYC_DB_MAXD", "1.5"))   # 가장 가까운 DB 칸이 이보다 멀면 (격자 단위) 풀지 않는다 — DB 범위 밖
CYCLE_COLD: bool = os.environ.get("GP8_CYC_COLD", "0" if CYCLE_DB else "1") not in ("0", "", "false")   # DB 초기값이 다 실패하면 cold (실시간이면 끔)
# 잡을 수 있는 범위 = 사이클 DB 에서 풀린 접촉 위치 (11-05 사용자 "잡을 수 있는 범위를 풀 수 있었던 DB 기준으로"): x 레인마다 접촉 y 범위
CYCLE_REACH: bool = os.environ.get("GP8_CYC_REACH", "1" if CYCLE_DB else "0") not in ("0", "", "false")
REACH_LEAD_S: float = float(os.environ.get("GP8_CYC_REACH_LEAD", "0.65"))  # hover 도착 → 접촉 (정지 출발: 앞 정렬 0.44 + 하강) — 두 시각 모두 범위 안
# 정지 출발 잡기 DB (11-05, scratchpad stop_db.py): B-스플라인 시작 때 물체 (x, y, 윗면) → 해 (P, T, v_out). v_in 0 잡기의 초기값
SEP_BS_DB: str = os.environ.get("GP8_SEP_BS_DB", "")
_CYC_DB: Optional[dict] = None
_CYC_LANES: Dict[float, list] = {}
_BS_DB: Optional[tuple] = None
_BS_SC = np.array([0.05, 0.05, 0.01])


def _bs_db():
    """GP8_SEP_BS_DB → (키 격자 단위 (n,3), [(P, T, v_out)])."""
    global _BS_DB
    if _BS_DB is None:
        import json
        K, S = [], []
        for f in [f for f in SEP_BS_DB.split(os.pathsep) if f]:
            for line in open(f):
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if d.get("ok"):
                    K.append(np.asarray(d["kb"], float) / _BS_SC); S.append((np.asarray(d["P"], float), float(d["T"]), float(d["v_out"])))
        _BS_DB = (np.array(K).reshape(-1, 3), S)
    return _BS_DB


def _cycle_reach(x: float, y: float) -> bool:
    """사이클 DB 의 통과 해들이 실제로 접촉한 범위 안인가: 접촉 xy = (다음 x, 다음 y − 벨트·t_k). x 레인 (DB 격자) 사이는 양쪽 레인 y 범위의 겹침."""
    _cycle_db(); ln = _CYC_LANES
    if not ln:
        return True
    xs = sorted(ln)
    if x < xs[0] - 0.025 or x > xs[-1] + 0.025:
        return False
    lo = max([v for v in xs if v <= x + 1e-9] or [xs[0]]); hi = min([v for v in xs if v >= x - 1e-9] or [xs[-1]])
    return max(ln[lo][0], ln[hi][0]) <= y <= min(ln[lo][1], ln[hi][1])
_CYC_SC = np.array([0.05, 0.075, 0.05, 0.05])                                # 칸 격자 (x, y, 다음 x, 다음 dy) — 거리는 격자 단위 최대값


def _cycle_db() -> dict:
    """GP8_CYC_DB 를 한 번 읽어 bin 목표별 (격자 단위 키, 해) 로. gate = DB 만들 때와 같은 (실제 x ≥ 0.24, z ≥ h − 3 mm, 접촉 근처 xy ≤ 6 mm)."""
    global _CYC_DB
    if _CYC_DB is None:
        import json
        from gp8_control.grasping import sep_pick as SP
        by: Dict[tuple, list] = {}
        for f in [f for f in CYCLE_DB.split(os.pathsep) if f]:
            for line in open(f):
                try:
                    d = json.loads(line)
                except ValueError:                                            # 아직 쓰는 중인 마지막 줄
                    continue
                g = d.get("gate")
                if not g or g["x_min"] < 0.24 or g["z_min_thr"] < SP.Z_H - 0.003 or g["e_xy"] > 0.006:
                    continue
                k = d["key"]; by.setdefault(tuple(round(v, 3) for v in d["target"]), []).append(
                    (np.array([k[0], k[1], k[2], k[3] - k[1]]) / _CYC_SC, np.asarray(d["P"], float), tuple(d["theta"]),
                     np.asarray(d["x"], float) if "x" in d else None, np.asarray(d["lam"], float) if "x" in d else None))
        _CYC_DB = {b: (np.array([e[0] for e in L]), [e[1:] for e in L]) for b, L in by.items()}
        for b, (Kd, sols) in _CYC_DB.items():                                # 접촉 xy (키 → 다음 x, 다음 y − 0.12·t_k)
            for kk, sol in zip(Kd, sols):
                xn = round(float(kk[2] * _CYC_SC[2]), 3); yc = float(kk[3] * _CYC_SC[3] + kk[1] * _CYC_SC[1]) - 0.12 * float(sol[1][2])
                r = _CYC_LANES.setdefault(xn, [yc, yc]); r[0] = min(r[0], yc); r[1] = max(r[1], yc)
    return _CYC_DB
# 대기점·준비점(정지)에서 잡기도 잡기 LP 로 (11-03 사용자 "고쳐"): 종전엔 정지 hover → 가속 → 추종 하강 → 절대 z PRESS_Z_FLOOR 까지
# 누름이라 물체 높이와 무관했고, 필터 때문에 실제 깊이가 누름 시간에 따라 달라 높이 −0.02~0.05 시뮬에서 윗면 기준 −35~+11 mm 였다.
# 가속(정지 → 물체 위 hover 벨트 추종)까지만 두고 그 hover 에서 잡기 LP (v_in 0) — hover 에서 바로 잡기와 같은 풀이기·같은 윗면 접촉.
SEP_REST_LP: bool = os.environ.get("GP8_SEP_REST_LP", "1") not in ("0", "", "false")
# 기다릴 필요가 없으면 대기점에 서지 않고 정지 → 물체 위 hover (벨트 추종 속도) 사다리꼴 한 번 (11-03 사용자): 종전엔 접근으로 대기점에
# 섰다가 (대기 0 이어도) 0.15 s 가속 — 최고 가속으로 섰다가 곧바로 다시 출발했다. 도착 hover·잡은 뒤 던지기 시작이 한계 밖이면 종전대로.
SEP_DIRECT_HOVER: bool = os.environ.get("GP8_SEP_DIRECT_HOVER", "1") not in ("0", "", "false")
# 사다리꼴(접근·정지·복귀) 가속 = M2 (= 컨트롤러 RT 스트림 가속 한계) × 이 값 (11-03 사용자: 한계에 5 % 여유). 던지기 NLP·잡기 LP 는 그대로.
TRAP_ACC_SCALE: float = float(os.environ.get("GP8_TRAP_ACC_SCALE", "1.0"))   # 11-03: M2 자체가 한계 × GP8_LIM_SCALE (0.9) 라 1.0
DIRECT_WAIT_MAX_S: float = float(os.environ.get("GP8_SEP_DIRECT_WAIT_MAX_S", "0.3"))   # 대기점 경로의 대기가 이 이하면 바로 hover (더 길면 대기점에서 기다림)
# 분리 방식은 모든 던지기를 분리 NLP 로 (11-03 사용자 "모든 경우의 수에서 분리 NLP로 풀게 해"): 준비점으로 던지기(마지막 물체)와
# 정지 종점 실패 뒤도 예전 던지기 NLP 를 쓰지 않는다 — 예전 NLP 의 warm DB(W1 5)가 지금 W1 100 과 달라 초기값으로만 쓰였고, 실물에서
# 마감에 멈춘 값으로 착지 예측 17~122 cm · 두 번 늦어 물체를 든 채 멈췄다. 준비점 = 끝 위치 준비 자세 TCP, 속도 0 (정지 종점 판).
class AheadDone(BaseException):
    """FIG4 look-ahead: 던지기 NLP 를 풀어 _sep_raw 에 남긴 뒤 잡기 LP·조립 없이 빠져나온다 (계획 탐색의 except Exception 에 안 잡힘)."""


SEP_PF_EXTEND_S: float = float(os.environ.get("GP8_SEP_PF_EXTEND_S", "0.20"))   # 잡기 먼저: NLP 마감 연장 (앱 마감 = 잡기 시작 − 0.35 s 중 0.20 s 를 NLP 에, 투영·조립에 ~0.06 s)
# 잡기 프로파일 캐시의 벨트 속도 허용 폭 [m/s] (2026-10-16 실기): 계획 작업마다 엔코더 측정 벨트(0.119, 0.1187 …)가 조금씩 달라
# 캐시를 매번 버리고 칸을 다시 풀어 head 계획이 2.6~5.2 s 걸렸고 전부 늦어 폐기됐다 (로봇이 안 움직임). 기준 = 0.01 m/s 로
# 반올림한 값, 측정이 기준에서 이 폭 안이면 같은 캐시. 프로파일은 기준 + 이 폭의 벨트로 푼다 (추종 몫을 크게 빼 보수적).
PROFILE_BELT_TOL: float = 0.01
# 도달 범위를 지나간 물체 (2026-10-27 사용자: "접근 불가능한 물체는 바로바로 queue에서 빼"): '앞으로도 불가'(retry_after None) 판정은
# 아직 먼 상류(벨트 0.05 에서 8 s 스캔은 40 cm 뿐)·레인 전체 불가도 같은 값이라, **이미 지나갔다**는 것을 따로 확인한다 —
# 그 시각부터 레인 끝(y PASS_Y_BOTTOM)까지 하류 어디서도 hover 도착이 안 되고, PASS_MARGIN_M 상류(조금 이른 시각)에서도 안 되며,
# 그보다 상류에서는 됐던 물체. 여유는 던지기 시간이 평균(t_ref)보다 짧을 몫과 큐 위치 오차 (벨트 0.12 에서 0.17 s).
PASS_MARGIN_M: float = float(os.environ.get("GP8_PASS_MARGIN_M", "0.02"))
PASS_SCAN_M: float = 0.02                                   # 하류·상류 스캔 간격 (도달 구간은 레인마다 25 cm 넘는 한 구간)
PASS_Y_BOTTOM, PASS_Y_TOP = -0.7, 0.9                       # lane_wait_ys 와 같은 레인 끝


# ---------------------------------------------------------------------------
# 입력 변환
# ---------------------------------------------------------------------------
def objects_from_perception(records: Iterable[dict], bins: Sequence[dict], *, belt_speed: float,
                            t_now: Optional[float] = None) -> List[ObjectState]:
    """인식 레코드(how_to_perceive.md 의 objects 항목) → ObjectState 목록 (위치는 t_now 기준으로 흘려 둠).

    records : [{"id": 3, "class": "metal", "grasp": {"robot": [x, y, z]}, "t": 촬영 시각(epoch s)}, …]
    bins    : [{"name", "x", "y", "z", "classes": [...]?}] — classes 가 있으면 그 클래스만, 없으면 가장 가까운 bin.
              조준점 = (x, y, z + AIM_UP)
    """
    v = belt_speed * BELT_DIR
    out = []
    for r in records:
        p = np.asarray(r["grasp"]["robot"], float).reshape(3)
        if t_now is not None and r.get("t") is not None:
            p = p + v * (float(t_now) - float(r["t"]))
        cls = str(r.get("class", ""))
        cand = [b for b in bins if not b.get("classes") or cls in b["classes"]] or list(bins)
        b = min(cand, key=lambda b: (b["x"] - p[0]) ** 2 + (b["y"] - p[1]) ** 2)
        out.append(ObjectState(r["id"], p, np.array([b["x"], b["y"], b["z"] + AIM_UP], float), cls))
    return out


# ---------------------------------------------------------------------------
# 세그먼트
# ---------------------------------------------------------------------------
@dataclass
class Segment:
    name: str
    t: np.ndarray                      # (N,) 세그먼트 시계 (0 = 세그먼트 시작)
    q: np.ndarray                      # (N,6) 로봇 관절
    qd: np.ndarray                     # (N,6)
    suction: List[Tuple[float, bool]] = field(default_factory=list)   # (세그먼트 시계 시각, on?)
    info: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return float(self.t[-1])


def _grid_times(T: float) -> np.ndarray:
    """0 … T (GRID_DT 배수) 를 정확히 k·GRID_DT 로 (linspace 의 부동소수 오차 없이)."""
    return np.arange(int(round(T / GRID_DT)) + 1) * GRID_DT


def _traj_segment(q0, qd0, q1, qd1, M1, M2, hz: float, name: str, kind: str, T: Optional[float] = None) -> Segment:
    """trajectory 프리미티브(사다리꼴) 세그먼트. 최소 시간 T 를 격자(1/hz)로 **올림**한 길이에 맞춘 사다리꼴 — 마지막 샘플이 목표 상태다.
    11-03: 종전엔 trajectory() 가 샘플 수를 내림해 T 보다 최대 한 칸 일찍 끝났고, 그 샘플을 목표로 덮어써 마지막 2 ms 에 위치·속도가
    튀었다 (정지 → 정지 접근의 가속이 컨트롤러 한계 = M2 의 최대 1.12 배, 실물 CODE 112 INC_MOVE_ERROR 정지 직전 동작).
    올림한 T 에서는 같은 가속 M2 로 오르고 순항 속력만 낮아진다. 관절 하나라도 사다리꼴이 안 되면 ValueError (종전엔 0 으로 채웠다).
    T 를 주면 그 길이 (최소 시간보다 짧으면 최소 시간). 가속은 M2 × TRAP_ACC_SCALE (한계 여유)."""
    from gp8_control.throwing.trajectory_primitive import _trajectory_1d, opt_time
    M2 = np.asarray(M2, float) * TRAP_ACC_SCALE
    q0 = np.asarray(q0, float)[:6]; qd0 = np.asarray(qd0, float)[:6]; q1 = np.asarray(q1, float)[:6]; qd1 = np.asarray(qd1, float)[:6]
    n = max(1, int(np.ceil(max(opt_time(q0, qd0, q1, qd1, M1, M2), float(T or 0.0)) * hz - 1e-9)))
    Q, Qd = np.empty((n + 1, 6)), np.empty((n + 1, 6))
    for j in range(6):
        r = _trajectory_1d(q0[j], qd0[j], q1[j], qd1[j], M1[j], M2[j], n / hz + 1e-9, hertz=hz)
        if r is None or len(r[0]) < n + 1:
            raise ValueError(f"{name}: 관절 {j + 1} 사다리꼴 불가 (T {n / hz:.3f}s)")
        Q[:, j], Qd[:, j] = r[0][:n + 1], r[1][:n + 1]
    Q[-1], Qd[-1] = q1, qd1                                            # 잔차는 1e-9 s 몫뿐
    return Segment(name, np.arange(n + 1) / hz, Q, Qd, info=dict(kind=kind))


def _ik_path(robot, pts: np.ndarray, q_init: np.ndarray, j6: float) -> np.ndarray:
    """TCP 점열 → 픽 자세 관절열 (이전 해를 시드로). 실패면 ValueError."""
    Q = np.empty((len(pts), 6))
    q_prev = np.asarray(q_init, float)[:6]
    for k, p in enumerate(pts):
        q = _nlp_planner.pick_joint(robot, p, q_init=q_prev)
        if q is None:
            raise ValueError(f"픽 IK 실패 (TCP {np.round(p, 3).tolist()})")
        q = np.asarray(q, float)[:6]; q[-1] = j6
        Q[k] = q; q_prev = q
    return Q


# ---------------------------------------------------------------------------
# Part 1: 던지기 프리미티브 (우리 throwing 모듈)
# ---------------------------------------------------------------------------
class ThrowPrimitive:
    """sequence_planner 의 primitive_fn — (출발 상태, 다음 물체, 출발 시각) → 세그먼트 + 소요 시간.

    belt_speed     : 벨트 속도 [m/s] (방향은 BELT_DIR = 로봇 −Y)
    robot          : robot/gp8.GP8 (픽 IK·FK). None 이면 생성.
    lift/lift_speed/descend_speed/press_depth/hold_s : 픽·경계 규약 (throw_planner 상수 기본)
    M1, M2         : 접근 세그먼트(trajectory 프리미티브)용 관절 속도·가속 한계. None 이면 로봇 데이터시트 × (1, 10).
    t_ref          : 움직이는 종점의 기준 시각 [s] (대략적인 t_f; 기본 env GP8_THR_CHAIN_T_REF = 0.6). 정확한 종점식(END_MODE=exact)
                     에서는 도착 가능 판정·DB 조회 위치에만 쓰이고 해에는 영향 없음. edge 마다 직전 t_f 로 갱신된다.
    n_random       : NLP cold 폴백의 무작위 재시작 수 (런타임 32). 순서 탐색에서는 불가 edge 가 흔해 빨리 포기하는 게 이득.
    """

    def __init__(self, belt_speed: float, *, robot=None, z_travel: float = TRAVEL_Z, lift_speed: float = LIFT_SPEED,
                 descend_speed: float = DESCEND_SPEED, press_depth: float = PRESS_DEPTH, hold_s: float = HOLD_S,
                 M1=None, M2=None, t_ref: float = None, hz: float = HZ_SEG, n_random: int = 4, logger=None):
        if robot is None:
            from gp8_control.robot.gp8 import GP8
            robot = GP8()
        from gp8_control.config import Config
        cfg = Config()
        self.robot, self.cfg, self.log = robot, cfg, logger
        self.v_belt = float(belt_speed) * BELT_DIR
        self.lift_speed, self.descend_nom = float(lift_speed), float(descend_speed)
        self.hold_s, self.press, self.hz = float(hold_s), float(press_depth), float(hz)
        self.t_ref = float(os.environ.get("GP8_THR_CHAIN_T_REF", "0.6")) if t_ref is None else float(t_ref)
        self._tf_hist: List[float] = []                               # 지금까지의 던지기 시간 → t_ref = 평균 (2026-09-22 사용자)
        self.judge_tf = JUDGE_TF                                      # 도달 판정 = 예측 t_f 시각 (False: 평균 t_ref)
        self._tf_model = None                                         # 던지기 시간 모델 (처음 쓸 때 읽음; 없으면 False)
        self._last_raw: Dict[Any, dict] = {}                          # (종점 종류, 타겟) → 직전 NLP 해 (warm_init)
        self.q_ready = None                                           # 준비 자세 (plan_full_sequence 가 넣음 — 분리 방식 준비점 종착)
        self.throw_only = False                                       # True: held 전이는 [아크, brake] 만 (2026-09-22 사용자: 후속은 따로)
        self.sep = SEP
        self._sep_tab = None; self._sep_line = None; self._sep_raw = {}
        self.end_above = 0.0                                          # 픽 세그먼트 시작(접촉) = 물체 표면
        # 상승 후·하강 전 높이는 **절대값** z_travel (2026-09-21 사용자: 상대값 대신 0.20 고정). hover 점·대기점·던지기 시작점이
        # 모두 이 높이라 물체 높이와 무관하게 DB 격자와 정확히 맞는다. 물체 윗면이 더 높으면 윗면 + TRAVEL_MIN_CLEAR.
        # 하강·상승 높이 h = z_travel − 물체 윗면 z 는 물체마다 달라 t_desc·t_pick 은 물체 z 의 함수다 (아래 *_of).
        self.z_travel = float(z_travel)
        self.t_acc = grid_ceil(float(getattr(cfg, "TRACK_ACCEL_T", 0.15)))   # 정지 hover → 벨트 속도 가속 시간 (상류 대기 후)
        self.j6 = float(cfg.PICK_WRIST_J6)
        self.n_random = int(n_random)
        vl = np.asarray(robot.velocity_limits, float)
        self.M1 = vl * float(cfg.JOINT_VEL_LIMIT_SCALE) if M1 is None else np.asarray(M1, float)
        self.M2 = self.M1 * float(cfg.JOINT_ACCEL_LIMIT_SCALE) if M2 is None else np.asarray(M2, float)
        self.n_nlp = 0
        self._lane_cache: Dict[Tuple[float, float], np.ndarray] = {}   # (x, z) → 대기 가능한 y 목록
        self._dead_lanes: set = set()            # (x, z, 벨트) — 어디서도 hover 도착이 안 되는 레인 (_passed 가 지나감으로 보지 않음)
        self._prof = None                                             # topp 잡기 프로파일 캐시 (벨트 기준이 바뀌면 새로)
        self._prof_nominal = 0.0

    # -- 높이 헬퍼 (물체 윗면 z 의 함수; 시간은 GRID_DT 배수로 올림 → 세그먼트 경계가 2 ms 격자에 놓인다) ------------------
    def z_top(self, z_obj: float) -> float:
        """상승 후·하강 전 절대 높이."""
        from gp8_control.throwing.throw_planner import TRAVEL_MIN_CLEAR
        return max(self.z_travel, float(z_obj) + TRAVEL_MIN_CLEAR)

    def top(self, p) -> np.ndarray:
        """흡착점 p → 그 위 z_top 의 점 (hover / 대기점 / 던지기 시작점)."""
        p = np.asarray(p, float).reshape(3)
        return np.array([p[0], p[1], self.z_top(p[2])])

    def h_of(self, z_obj: float) -> float:
        """하강 = 상승 높이 [m]."""
        return self.z_top(z_obj) - float(z_obj)

    def _profile(self, p_obj):
        """topp: 물체 위치 p_obj 의 (내려감, 올라옴, z_top, z_floor) — 반지름 칸 프로파일 (pick_profile.PickProfiles).
        const 모드이거나 그 위치를 못 풀면 None (이전 등속 규칙)."""
        if PICK_PROFILE not in ("topp", "free") or np.ndim(p_obj) == 0:
            return None
        b = float(np.linalg.norm(self.v_belt))
        if PICK_PROFILE == "free":                                      # 양끝 자유 (2026-10-29): 미리 푼 LP 표 (GP8_PICK_FREE_TABLE)
            if self._prof is None:
                from gp8_control.grasping.pick_profile import FreeProfiles
                self._prof = FreeProfiles(); self._prof_nominal = round(b, 2)
        elif self._prof is None or abs(self._prof_nominal - b) > PROFILE_BELT_TOL + 1e-9:
            from gp8_control.grasping.pick_profile import PickProfiles
            self._prof_nominal = round(b, 2)
            self._prof = PickProfiles(self.robot, self.j6, self._prof_nominal + PROFILE_BELT_TOL, self.lift_speed,
                                      float(self.cfg.JOINT_ACCEL_LIMIT_SCALE))
            self._prof.prewarm(z_top=self.z_top(0.02), z_floor=0.02 - self.press_of(0.02),
                               cache_path=self._profile_cache_path(), load_only=True)   # 파일 캐시가 있으면 읽기만
        p_obj = np.asarray(p_obj, float).reshape(3)
        z_obj = float(p_obj[2])
        z_top, z_floor = self.z_top(z_obj), z_obj - self.press_of(z_obj)
        pr = self._prof.get(p_obj, z_top, z_floor)          # 바닥: 실제 최저점 목표면 칸마다 서보 모델로 찾은 명령 바닥 (pick_profile.ACTUAL_Z)
        return None if pr is None else (pr[0], pr[1], z_top, pr[2])

    def prewarm_profiles(self, cache_dir: Optional[str] = None) -> int:
        """topp 프로파일 칸을 미리 푼다 (앱 시작 때 — 첫 결정에 계산이 붙지 않게). 설정 지문별 파일 캐시. 리턴: 칸 수."""
        if PICK_PROFILE not in ("topp", "free"):
            return 0
        self._profile(np.array([0.45, 0.0, 0.02]))
        return self._prof.prewarm(z_top=self.z_top(0.02), z_floor=0.02 - self.press_of(0.02),
                                  cache_path=self._profile_cache_path(cache_dir))

    def _profile_cache_path(self, cache_dir: Optional[str] = None) -> str:
        import hashlib
        d = cache_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "learning", "models")
        fp = hashlib.sha1(repr((self._prof.fingerprint(), self.z_travel, PRESS_Z_FLOOR)).encode()).hexdigest()[:12]
        return os.path.join(d, f"pick_profile_{fp}.pkl")

    @staticmethod
    def _contact_of(dn, z_top: float, z_floor: float, z_obj: float):
        """내려감 프로파일의 (경로 비율 f, 접촉 경로 길이 s_c, 접촉 시각 T_c, 접촉 속도) — 실제 높이로 비례 (칸 키는 mm 반올림)."""
        f = (z_top - z_floor) / dn.s[-1] if dn.s[-1] > 1e-9 else 1.0
        s_c = (z_top - z_obj) / f
        T_c = dn.time_at(s_c)
        return f, s_c, T_c, f * float(dn.at(T_c)[1])

    def t_desc_of(self, p) -> float:
        """추종 하강 시간 hover → 접촉 (격자로 올림). p = 물체 위치 (topp: 그 위치의 시간 최적) 또는 윗면 z (이전 등속 규칙)."""
        pr = self._profile(p)
        if pr is None:
            z_obj = float(p) if np.ndim(p) == 0 else float(np.asarray(p, float).reshape(3)[2])
            return grid_ceil(self.h_of(z_obj) / self.descend_nom)
        dn, _, z_top, z_floor = pr
        return grid_ceil(self._contact_of(dn, z_top, z_floor, float(np.asarray(p, float).reshape(3)[2]))[2])

    def v_desc_of(self, z_obj: float) -> float:
        """(const) 실효 등속 하강 속도."""
        return self.h_of(z_obj) / grid_ceil(self.h_of(z_obj) / self.descend_nom)

    def press_of(self, z_obj: float) -> float:
        """실제 눌림 깊이: 윗면에서 절대 z PRESS_Z_FLOOR(−0.025) 까지 (상한 press_depth)."""
        return float(min(self.press, max(0.0, float(z_obj) - PRESS_Z_FLOOR)))

    def t_pick_of(self, p) -> float:
        """픽 세그먼트 길이 (접촉 → 눌림 → 상승 끝 = 던지기 시작) — pick_segment 와 같은 격자 올림. p = 물체 위치 또는 윗면 z."""
        pr = self._profile(p)
        if pr is not None:
            dn, up, z_top, z_floor = pr
            T_c = self._contact_of(dn, z_top, z_floor, float(np.asarray(p, float).reshape(3)[2]))[2]
            th = self.hold_s if self.hold_s > 1e-9 else 0.0
            return grid_ceil((dn.duration - T_c) + th + up.duration)
        z_obj = float(p) if np.ndim(p) == 0 else float(np.asarray(p, float).reshape(3)[2])
        vd, d = self.v_desc_of(z_obj), self.press_of(z_obj)
        t1 = grid_ceil(2.0 * d / vd) if (vd > 0 and d > 0) else 0.0
        th = grid_ceil(self.hold_s) if self.hold_s > 1e-9 else 0.0
        t3 = grid_ceil(2.0 * (self.h_of(z_obj) + d) / self.lift_speed)
        return t1 + th + t3

    # -- 자세 헬퍼 ---------------------------------------------------------
    def _pick_ik(self, p, q_init=None):
        q = _nlp_planner.pick_joint(self.robot, p, q_init=q_init)
        if q is None:
            return None
        q = np.asarray(q, float)[:6]; q[-1] = self.j6
        return q

    class NotYetReachable(ValueError):
        """도착 자세가 NLP 관절 한계 밖 — retry_after(절대 시각) 이후면 가능 (None = 앞으로도 불가).
        duration_est: 그때 걸릴 소요 시간 추정 (호출측이 채움). passed: 이미 도달 범위를 지나감 (_passed — 앱이 큐에서 뺀다)."""
        def __init__(self, msg, retry_after, passed: bool = False):
            super().__init__(msg); self.retry_after = retry_after; self.duration_est = None; self.passed = bool(passed)

    def _arrival_ok(self, dest: MovingPoint, t_abs: float):
        """t_abs 에 hover 자세로 도착 가능한가 — hover 자세뿐 아니라 그 뒤 추종 하강·픽이 끝나 **던지기가 시작되는 자세**
        (물체 + START_LIFT, t_abs + t_desc + t_pick) 도 NLP 한계 안이어야 한다. 픽 동안 물체가 하류로 흘러가 시작 자세가
        한계를 넘으면 던지기 계획이 실패한다 (2026-09-21 시뮬: 접근·하강만 하고 못 던짐)."""
        p = self.top(dest.at(t_abs))
        q = self._pick_ik(p)
        if JCYC:                                                              # 11-07 jcycle: 잡기 창 내내 수직 툴 IK·관절 상자 (윗면·hover 높이)
            return (q is not None and self._jc_reach(dest, t_abs, t_abs + JC_PW)), q, p
        if CYCLE and CYCLE_REACH:                                             # 11-05: 사이클 DB 에서 풀린 접촉 범위 (hover 도착·접촉 두 시각) + 잡기 IK
            pc = dest.at(t_abs + REACH_LEAD_S)
            return (q is not None and _cycle_reach(float(p[0]), float(p[1])) and _cycle_reach(float(pc[0]), float(pc[1]))), q, p
        ok = q is not None and _nlp_planner.q_start_in_limits(to_planner(q))
        if ok:
            offs = self._sep_pick_offsets(dest.at(t_abs)) if self.sep else None
            if offs is None:
                po = dest.at(t_abs)                                          # 물체 (윗면)
                t_d = self.t_desc_of(po)
                offs = (t_d + self.t_pick_of(dest.at(t_abs + t_d)),)
            for off in offs:                                                 # 분리 방식: head 의 분리 잡기 LP 가 쓸 수 있는 길이 전부 (짧은·중간·긴)
                q_s = self._pick_ik(self.top(dest.at(t_abs + off)), q_init=q)
                ok = q_s is not None and _nlp_planner.q_start_in_limits(to_planner(q_s))
                if not ok:
                    break
        return ok, q, p

    def _jc_reach(self, mp: MovingPoint, t0: float, t1: float, dt: float = 0.05) -> bool:
        """[t0, t1] (절대 시각) 동안 물체 위 수직 툴 IK 가 윗면 높이와 hover 높이 모두에서 있고, jcycle 명령 관절 상자 안인가.
        GP8_JC_DB 면 대신: t0 의 물체 위치가 사이클 DB 칸 (xi, yi) 반 칸 안인가 (11-08 사용자: DB 범위 = 도달 범위)."""
        if JC_DB:
            import jc_bridge as JB
            return bool(JB.db_cover_i(np.asarray(mp.at(float(t0)), float)))
        from gp8_control.throwing.nlp import throw_nlp as _T
        from gp8_control.throwing import cycle_nlp as _CY
        if not hasattr(self, "_jc_box"):
            self._jc_box = (np.maximum(_T.GP8_Q_MIN, _T.Q_LO_REAL) + _CY.Q_MARGIN, np.minimum(_T.GP8_Q_MAX, _T.Q_HI_REAL) - _CY.Q_MARGIN)
        lo, hi = self._jc_box; q_prev = None
        for t in np.arange(float(t0), float(t1) + 1e-9, dt):
            po = np.asarray(mp.at(t), float)
            for z in (float(po[2]), float(self.top(po)[2])):
                q = self._pick_ik(np.r_[po[:2], z], q_init=q_prev)
                if q is None:
                    return False
                qp = to_planner(q)
                if np.any(qp < lo) or np.any(qp > hi):
                    return False
                q_prev = q
        return True

    def _sep_pick_offsets(self, p_obj) -> Optional[Tuple[float, ...]]:
        """분리 잡기 (hover → 접촉 → 상승 끝) 가 끝나는 시각들 (hover 기준) — head 의 sep_pick_segment 가 고를 수 있는 길이의 최소·중간·최대:
        잡기 시간표 t_pick(r, v_in, v_out) 을 v_in ∈ [0, 하강 상한], v_out ∈ [0.3, 상승 상한] 에서, 길이 N = ⌈t/dt⌉ + 3 (+15: head 의 마지막 시도).
        표에 반경이 없으면 None (종전 하강·픽 시간 모델). 10-31 사용자: 꼬리가 head 와 같은 검사로 다음 물체를 고르게."""
        r = float(np.hypot(p_obj[0], p_obj[1])); z = float(p_obj[2]); k = (int(np.floor(r / 0.01 + 1e-9)), int(round(z / 0.005)))
        cache = self.__dict__.setdefault("_sep_off_cache", {})
        if k in cache:
            return cache[k]
        tb = self._sep_table(); vi = tb.vin_lim(r, z); vo = tb.vout_lim(r, z)
        out = None
        if tb.row(r, z) is not None and vi is not None and vo is not None:
            ts = [tb.t_pick(r, a, b, z) for a in np.linspace(0.0, max(0.0, vi[0]), 3) for b in np.linspace(0.3, max(0.3, vo[0]), 3)]
            n_lo = int(np.ceil(min(ts) / GRID_DT)) + 3; n_hi = int(np.ceil(max(ts) / GRID_DT)) + 3 + 15
            out = (n_lo * GRID_DT, 0.5 * (n_lo + n_hi) * GRID_DT, n_hi * GRID_DT)
        cache[k] = out
        return out

    def tf_window(self, dest: MovingPoint, t_arc0: float, t_j: float, iters: int = 6) -> Tuple[float, float]:
        """던지기 시간 범위 (아크 시계): 끝 hover 와 거기서 잡은 뒤 던지기 시작 자세가 모두 관절 범위 안인 t_f 구간 (t_j 를 포함하는 쪽).
        도착 판정은 예측 t_f(t_j) 한 시각에서만 했는데 NLP 가 더 짧은 t_f 를 고르면 물체가 덜 흘러온 상류에서 끝나 head 가 못 잡았다
        (10-31 실시간 시뮬 물체 62: 잡은 뒤 J3 47.3° > 45°, 정지 후 다른 물체를 예전 방식으로 잡아 5 mm 깊게). 양 끝을 이분법으로."""
        from gp8_control.throwing.nlp import throw_nlp as _T
        lo_b, hi_b = float(_T.T_BOUNDS[0]), float(_T.T_BOUNDS[1])
        ok = lambda t: self._arrival_ok(dest, t_arc0 + t)[0]               # noqa: E731
        t_j = float(np.clip(t_j, lo_b, hi_b))
        lo = lo_b
        if not ok(lo_b):
            a, b = lo_b, t_j                                                 # a 안 됨, b 됨
            for _ in range(iters):
                m = 0.5 * (a + b)
                a, b = (a, m) if ok(m) else (m, b)
            lo = b
        hi = hi_b
        if not ok(hi_b):
            a, b = t_j, hi_b                                                 # a 됨, b 안 됨
            for _ in range(iters):
                m = 0.5 * (a + b)
                a, b = (m, b) if ok(m) else (a, m)
            hi = a
        return lo, hi

    def arrival_boundary(self, dest: MovingPoint, t_abs: float, scan_s: float = 8.0) -> ThrowBoundary:
        """다음 물체 j 에 도착하는 경계 = **물체 위 hover 높이에서 벨트를 따라가는 상태** (시각 t_abs 의 물체 위치 + hover,
        qd = J⁻¹·v_belt, 이동률 = 벨트). 여기서 추종 하강(descend_segment)이 이어져 접촉한다 — 2026-09-21 사용자: 수직이 아니라
        물체를 벨트 속도로 따라가며 내려간다. ThrowBoundary.t 는 호출측이 아크 시계로 다시 찍는다.
        그 시각의 자세가 NLP 한계 밖이면 앞으로 scan_s 초 안에 들어오는 첫 시각을 NotYetReachable(retry_after) 로 알린다."""
        ok, q, p = self._arrival_ok(dest, t_abs)
        if not ok:
            retry = None
            for dt in np.arange(0.05, scan_s + 1e-9, 0.05):
                if self._arrival_ok(dest, t_abs + dt)[0]:
                    retry = t_abs + dt
                    break
            passed = retry is None and self._passed(dest, t_abs, scanned_s=scan_s)
            raise self.NotYetReachable(
                f"다음 물체 위 hover 자세가 NLP 관절 한계 밖 (TCP {np.round(p, 3).tolist()})"
                + (f" — {retry:.2f}s 이후 가능" if retry is not None else
                   " — 앞으로도 불가 (도달 범위 지남)" if passed else " — 앞으로도 불가"), retry, passed=passed)
        qd = belt_joint_velocity(q, self.v_belt)
        # p_tcp/v_tcp_exact/j6: plan_throw 가 정확한 Cartesian 종점식(FK 등식, 툴 수직 + J6 고정)을 쓰기 위한 정보
        return ThrowBoundary(q, qd, None, qd, p_tcp=p, v_tcp_exact=self.v_belt, j6=float(q[5]))

    def _passed(self, dest: MovingPoint, t_abs: float, scanned_s: float = 0.0) -> bool:
        """t_abs 에 hover 도착이 안 되는 물체가 **이미 도달 범위를 지나갔는가** (PASS_MARGIN_M 위 주석). scanned_s: 호출측이
        t_abs 부터 그만큼 하류를 이미 봤다 (arrival_boundary 의 8 s 스캔). 멈춘 벨트·레인 전체 불가·아직 상류면 False."""
        speed = float(np.linalg.norm(self.v_belt))
        if speed <= 1e-9:
            return False
        p0 = dest.at(t_abs)
        lane = (round(float(p0[0]), 3), round(float(p0[2]), 3), round(speed, 3))
        if lane in self._dead_lanes:                                  # 어디서도 도착이 안 되는 레인 (지나간 게 아니다)
            return False
        dt = PASS_SCAN_M / speed
        t = t_abs + max(float(scanned_s), 0.0) + dt                   # 1) 더 하류에서 다시 닿는 자리가 없다
        while float(dest.at(t)[1]) >= PASS_Y_BOTTOM:
            if self._arrival_ok(dest, t)[0]:
                return False
            t += dt
        dm = PASS_MARGIN_M / speed                                    # 2) 여유: 조금 이른 시각(상류)에서도 안 된다
        for k in range(1, int(np.ceil(PASS_MARGIN_M / 0.01 - 1e-9)) + 1):
            if self._arrival_ok(dest, t_abs - min(k * 0.01 / speed, dm))[0]:
                return False
        t = t_abs - dm - dt                                           # 3) 그보다 상류에서는 닿았다 = 지나감
        while float(dest.at(t)[1]) <= PASS_Y_TOP:
            if self._arrival_ok(dest, t)[0]:
                return True
            t -= dt
        if len(self._dead_lanes) < 4096:
            self._dead_lanes.add(lane)
        return False

    def predict_tf(self, start: ThrowBoundary, p_target, end: ThrowBoundary) -> Optional[float]:
        """던지기 시작 상태 → 끝점(end, 물체 위 hover 추종) 던지기의 예측 시간 t_f — 대리 프리미티브(learning.surrogate)의 arc_segment 와 같은
        모델·특징. 모델이 없으면 None (그때는 평균 t_ref 판정)."""
        if self._tf_model is None:
            m = getattr(self, "model", None)                          # 대리 프리미티브는 자기 모델
            try:
                from gp8_control.learning.throw_time_model import ThrowTimeMLP, load_default
                self._tf_model = m if isinstance(m, ThrowTimeMLP) else (load_default() or False)
            except Exception:                                         # noqa: BLE001 — 모델·모듈이 없으면 예전 판정
                self._tf_model = False
        if self._tf_model is False:
            return None
        from gp8_control.learning.throw_time_model import features
        moving = end.q_rate is not None and bool(np.any(end.q_rate))
        X = features(to_planner(start.q), to_planner(start.qd), to_planner(end.q), to_planner(end.qd), np.asarray(p_target, float),
                     1.0 if moving else 0.0)
        return float(np.clip(float(self._tf_model.predict(X)[0][0]), 0.3, 1.5))   # surrogate.T_F_BOUNDS

    def judge_arrival(self, start: ThrowBoundary, p_target, dest: MovingPoint, t_arc0: float) -> Tuple[float, ThrowBoundary]:
        """다음 물체 도달 판정 (JUDGE_TF 위 주석): 리턴 (판정 시각 t_j — 아크 시계, 끝점 boundary). 못 닿으면 arrival_boundary 처럼
        NotYetReachable (retry_after: 기다렸다 잡음 / None: 앞으로도 불가, passed). 지금 잡음·기다렸다 잡음·앞으로도 불가를 가르는 규칙은
        그대로이고 판정하는 시각만 예측 t_f 다. t_ref 에 아직 상류인 물체는 닿기 시작하는 시각 r 의 끝점으로 던지기 시간을 예측해, 예측한
        던지기 끝이 r 이후면 그때 닿는 것으로 본다 (기다리지 않고 바로 이음); 아니면 종전대로 기다렸다 잡음."""
        t_j = self.t_ref
        try:
            end = self.arrival_boundary(dest, t_arc0 + t_j)
        except self.NotYetReachable as e:
            if not self.judge_tf or e.retry_after is None:
                raise
            t_r = float(e.retry_after) - t_arc0
            end = self.arrival_boundary(dest, t_arc0 + t_r)           # r 에는 닿는다 (스캔이 찾은 첫 시각)
            t_p = self.predict_tf(start, p_target, end)
            if t_p is None or t_p < t_r - 1e-9:
                raise                                                 # 던지기가 r 전에 끝난다 → 기다렸다 잡음
            t_j = t_r
        if not self.judge_tf:
            return t_j, end
        for _ in range(JUDGE_ITERS):                                  # 판정은 늘 예측 시각에서; 예측이 RELIN_S 넘게 바뀌었을 때만 한 번 더 예측
            t_p = self.predict_tf(start, p_target, end)
            if t_p is None:
                break
            moved = abs(t_p - t_j) > RELIN_S
            t_j = t_p
            end = self.arrival_boundary(dest, t_arc0 + t_j)           # 예측 시각에 못 닿으면 NotYetReachable (호출측이 종전 규칙으로)
            if not moved:
                break
        return t_j, end

    def passed_at(self, dest: MovingPoint, t_abs: float) -> bool:
        """시도하지 않은 후보(신경망이 더 상류 물체를 골라 건너뛴 하류 물체)가 t_abs 에 이미 지나갔는가."""
        return (not self._arrival_ok(dest, t_abs)[0]) and self._passed(dest, t_abs)

    # -- 정지 hover 대기점 W (2026-09-21 사용자 규칙) ---------------------------------------------------------
    #   빈손(첫 물체 / 앞 던지기 없음): 레인 위 "hover 가능한 구간" 에서 **물체와 가장 가까운 점** 을 대기점으로 한다. 팔이 그 점에
    #       물체보다 늦으면(접근 시간) 팔이 준비되는 시각에 물체가 있을 자리로 내린다.
    #   앞 물체를 던지는 중: 던지기가 끝나고 하강할 시각에 물체가 있을 자리 (= NLP 의 움직이는 종점). 그때 물체가 아직
    #       구간 밖이면 위 규칙으로 가장 가까운 점에 정지 대기 (hover_boundary).
    #   접촉점 p_in 과 대기점의 관계: 가속(t_acc, 이동 v·t_acc/2) + 추종 하강(t_desc) 동안 물체가 흐르므로
    #       W = p_in − v·(t_desc + t_acc/2)   (접촉점보다 상류),  z = hover
    def _wait_point_ok(self, W: np.ndarray):
        """W 가 대기점으로 쓸 수 있는가: W 위 hover 자세와, 거기서 잡고 올라온 던지기 시작 자세(물체 + lift, 접촉 뒤 t_pick 시각의
        물체 위치 = W + v·(lead + t_pick)) 가 모두 NLP hull 안. 벨트 속도가 정해지면 순수 기하라 (x, y) 만으로 정해진다."""
        q = self._pick_ik(self.top(W))
        if q is None or not _nlp_planner.q_start_in_limits(to_planner(q)):
            return None
        lead = self.t_desc_of(W) + 0.5 * self.t_acc
        q_s = self._pick_ik(self.top(W + self.v_belt * (lead + self.t_pick_of(W + self.v_belt * lead))), q_init=q)
        if q_s is None or not _nlp_planner.q_start_in_limits(to_planner(q_s)):
            return None
        return q

    def lane_wait_ys(self, x: float, z: float, y_top: float = 0.9, y_bottom: float = -0.7, step: float = 0.01) -> np.ndarray:
        """레인 x (물체 높이 z) 위에서 대기점으로 쓸 수 있는 y 들 — 상류(+y)→하류 순, step 격자. (x, z) 별 캐시."""
        key = (round(float(x), 3), round(float(z), 3), round(float(np.linalg.norm(self.v_belt)), 3))   # 벨트 속도별 (던지기 시작 자세 검사가 속도에 의존)
        ys = self._lane_cache.get(key)
        if ys is None:
            cand = np.arange(y_top, y_bottom - 1e-9, -step)
            ys = np.array([y for y in cand if self._wait_point_ok(np.array([x, y, z])) is not None])
            self._lane_cache[key] = ys
        return ys

    def hover_boundary(self, dest: MovingPoint, t_in: float) -> Tuple[ThrowBoundary, np.ndarray, float]:
        """접촉 시각 t_in 을 원할 때의 정지 대기점: W = p(t_in) − v·lead 가 가능하면 그대로, 아니면 레인 위 가능 구간에서
        **그 점과 가장 가까운 하류 점**(= 그 시각의 물체와 가장 가까운 가능한 점; 조금 더 늦게 잡는다). 리턴 (boundary, p_in, t_in, t_desc)
        — t_desc 는 W 를 정할 때 쓴 하강 시간 (descend_segment 에 그대로 넘겨야 대기점과 하강 시작이 맞는다)."""
        v, speed = self.v_belt, float(np.linalg.norm(self.v_belt))
        t_desc = self.t_desc_of(dest.at(t_in))
        lead = t_desc + 0.5 * self.t_acc
        t_in = grid_ceil(t_in)                                        # 접촉 시각은 격자 위 (세그먼트 경계)
        W0 = dest.at(t_in) - v * lead
        q = self._wait_point_ok(W0)
        if q is not None:
            return ThrowBoundary(q, np.zeros(6)), W0 + v * lead, t_in, t_desc
        if speed <= 1e-9:
            raise ValueError(f"정지 벨트: 대기점이 한계 밖 (TCP {np.round(self.top(W0), 3).tolist()})")
        ys = self.lane_wait_ys(W0[0], W0[2])
        below = ys[ys <= W0[1] + 1e-9]                                # W0 보다 하류(늦게 잡는 쪽)의 가능한 y
        if len(below) == 0:
            raise ValueError(f"레인 x={W0[0]:.2f} 에서 물체보다 하류에 대기 가능한 점이 없음 (TCP {np.round(self.top(W0), 3).tolist()})")
        # 격자(1 cm) 위 가장 가까운 점 → 접촉 시각을 2 ms 격자로 올려 W 를 그만큼 더 하류로 (≤ 0.6 mm)
        dt = grid_ceil(t_in + (W0[1] - float(below[0])) / speed) - t_in
        W = W0 + v * dt
        q = self._wait_point_ok(W)
        if q is None:                                                 # 격자 반올림으로 경계를 벗어난 드문 경우: 한 칸 더
            for y in below[1:]:
                dt = grid_ceil(t_in + (W0[1] - float(y)) / speed) - t_in
                W = W0 + v * dt
                q = self._wait_point_ok(W)
                if q is not None:
                    break
            else:
                raise ValueError(f"레인 x={W0[0]:.2f} 대기점을 못 찾음")
        return ThrowBoundary(q, np.zeros(6)), W + v * lead, t_in + dt, t_desc

    def wait_point(self, q_from, dest: MovingPoint, t_now: float):
        """빈손 대기점 (사용자 규칙): 레인 위 hover 가능 구간에서 **물체(지금 위치)와 가장 가까운 점**. 팔이 거기에 물체보다
        늦으면(q_from → hover 접근 시간 T_app) 팔이 준비되는 시각에 물체가 있을 자리로 내린다.
        리턴 (hover boundary, p_in, t_in, approach 세그먼트, t_desc)."""
        v, speed = self.v_belt, float(np.linalg.norm(self.v_belt))
        p_now = dest.at(t_now)
        x, y_now, z = float(p_now[0]), float(p_now[1]), float(p_now[2])
        t_desc = self.t_desc_of(p_now)
        lead = t_desc + 0.5 * self.t_acc
        ys_all = self.lane_wait_ys(x, z)
        # 물체가 이미 지나간 접촉점(p_in = W − v·lead 가 물체보다 상류)은 제외 → 남은 것 중 가장 상류 = 물체와 가장 가까운 점
        ys = ys_all[ys_all - speed * lead <= y_now + 1e-9] if speed > 1e-9 else ys_all
        if len(ys) == 0:
            # 레인에 대기점은 있는데 모든 접촉점이 물체보다 PASS_MARGIN_M 넘게 상류 → 이미 지나감 (앞으로도 불가; 앱이 큐에서 뺀다)
            passed = speed > 1e-9 and len(ys_all) > 0 and not np.any(ys_all - speed * lead <= y_now + PASS_MARGIN_M)
            raise self.NotYetReachable(f"레인 x={x:.2f} 에 대기 가능한 점이 없음 (물체 y={y_now:+.2f})"
                                       + (" — 도달 범위 지남" if passed else ""), None, passed=passed)
        y_edge = float(ys[0])                                         # 가능 구간의 상류 끝 (물체와 가장 가까운 점)
        t_in = None
        for y_W in ys:                                                # 상류부터: 팔이 제때 갈 수 있는 첫 점
            W = np.array([x, y_W, z])
            q = self._wait_point_ok(W)
            if q is None:
                continue
            seg = _traj_segment(q_from, np.zeros(6), q, np.zeros(6), self.M1, self.M2, self.hz, "approach", "approach")
            t_arm = t_now + seg.duration                              # 팔이 W 에 서는 시각
            t_obj = t_now + ((y_now - (y_W - speed * lead)) / speed if speed > 1e-9 else 0.0)   # 물체가 p_in 에 오는 시각
            if t_arm + self.t_acc + t_desc <= t_obj + 1e-9:
                # 팔이 먼저다. 1 cm 격자 안쪽으로 더 상류로 당길 여유(slack)가 있으면 당기되, 구간 상류 끝은 못 넘는다
                slack = t_obj - (t_arm + self.t_acc + t_desc)
                y_W = min(y_edge, y_W + speed * slack) if speed > 1e-9 else y_W
                t_in = grid_ceil(t_now + ((y_now - (y_W - speed * lead)) / speed if speed > 1e-9 else self.t_acc + t_desc))
                break
        if t_in is None:                                              # 어디서도 팔이 물체보다 늦다 → 물체가 지나감
            raise ValueError(f"레인 x={x:.2f} 물체 y={y_now:+.2f}: 팔이 준비되기 전에 물체가 가능 구간을 지나감")
        # 격자에 놓인 t_in 으로 W 확정 (격자 올림·slack 때문에 경계를 살짝 벗어나면 hover_boundary 가 하류로 보정)
        hov, p_in, t_in, t_desc = self.hover_boundary(dest, t_in)
        seg = _traj_segment(q_from, np.zeros(6), hov.q, np.zeros(6), self.M1, self.M2, self.hz, "approach", "approach")
        return hov, p_in, t_in, seg, t_desc

    def rest_segment(self, q, duration: float, name="wait") -> Segment:
        """정지 대기 (hover 에서 상류 물체를 기다림). 길이는 GRID_DT 배수로 올림."""
        t = _grid_times(max(GRID_DT, grid_ceil(duration)))
        n = len(t)
        Q = np.repeat(np.asarray(q, float)[None, :6], n, axis=0)
        return Segment(name, t, Q, np.zeros_like(Q), info=dict(kind="wait"))

    def accel_segment(self, dest: MovingPoint, t_h: float, q0) -> Segment:
        """정지 hover (대기점) → 물체 위 hover 를 벨트 속도로 추종: t_acc 동안 xy 등가속, z = hover. t_h = 끝 시각 (물체 바로 위).
        descend_segment 의 가속 구간과 같은 궤적이고, 끝 상태 (hover, 벨트 속도) 가 hover 에서 바로 잡기 LP (v_in 0) 의 시작이다."""
        ta = self.t_acc
        p_h = dest.at(t_h)
        h = self.h_of(float(p_h[2]))
        t = _grid_times(ta)
        p_w = p_h - self.v_belt * (0.5 * ta)                      # 대기점 (가속 동안 v·t_acc/2 이동)
        pts = np.array([p_w + self.v_belt * (0.5 * tk * tk / ta) + UP * h for tk in t])
        Q = _ik_path(self.robot, pts, q0, self.j6)
        Qd = np.gradient(Q, t, axis=0)
        Qd[0] = np.zeros(6)
        Qd[-1] = belt_joint_velocity(Q[-1], self.v_belt)
        return Segment("accel", t, Q, Qd, info=dict(kind="accel", accel=ta, h=h))

    def descend_segment(self, dest: MovingPoint, t_in: float, q0, from_rest: bool, t_desc: Optional[float] = None) -> Segment:
        """물체 위 hover → 표면 접촉, **내내 벨트 추종**: xy = 물체 위치(벨트 속도로 이동), z = hover → 물체 윗면.
        topp: 접촉 위치의 시간 최적 프로파일 (z 속도 0 에서 출발 — 관절 한계만), const: v_desc 등속.
        from_rest=True 면 앞에 가속 구간(t_acc: xy 0 → v_belt 등가속, z 는 hover 유지)을 붙인다 (정지 hover 대기 뒤).
        t_in = 접촉 시각, t_desc = 하강 시간 (호출측이 t_in·대기점을 정할 때 쓴 값 — 주면 그 길이에 맞춰 프로파일 시간을 비례).
        끝 상태 = (벨트 추종 + 하강) 속도 → pick_segment 의 시작과 같다."""
        ta = self.t_acc if from_rest else 0.0
        p_c = dest.at(t_in)
        z_obj = float(p_c[2])
        h = self.h_of(z_obj)
        pr = self._profile(p_c)
        if t_desc is None:
            t_desc = self.t_desc_of(p_c)
        t_desc = grid_ceil(float(t_desc))
        if pr is not None:
            dn, _, z_top, z_floor = pr
            f, s_c, T_c, _ = self._contact_of(dn, z_top, z_floor, z_obj)
            kt = T_c / t_desc                                     # 프로파일 시간 비 (≤ 1: 느리게 — 한계 안; 칸·위치 차이로 1 을 조금 넘을 수 있다)
            if kt > 1.05 and self.log is not None:
                self.log.warn(f"  하강 프로파일 시간 비 {kt:.3f} > 1.05 (t_desc {t_desc:.3f} < 최적 {T_c:.3f})")
            vd = kt * float(dn.at(T_c)[1]) * f                    # 접촉 속도
        else:
            vd = h / t_desc
        T = ta + t_desc
        t = _grid_times(T)
        n = len(t)
        t0 = t_in - T                                             # 세그먼트 시작 절대 시각
        pts = np.empty((n, 3))
        for k, tk in enumerate(t):
            if tk < ta:                                           # 가속: 정지 hover 점에서 xy 등가속, 물체를 따라잡음
                p_hover_xy = dest.at(t0 + ta) - self.v_belt * (0.5 * ta)
                pts[k] = p_hover_xy + self.v_belt * (0.5 * tk * tk / ta) + UP * h
            elif pr is not None:                                  # 추종 하강: xy = 물체, z = 시간 최적 (z_top → 물체 윗면)
                pts[k] = dest.at(t0 + tk) + UP * (h - f * float(dn.at(kt * (tk - ta))[0]))
            else:                                                 # 추종 하강: xy = 물체, z 등속 하강 (z_top → 물체 윗면)
                pts[k] = dest.at(t0 + tk) + UP * (h - vd * (tk - ta))
        pts[-1] = dest.at(t_in)                                   # 마지막 샘플 = 접촉 (부동소수 잔차 제거)
        Q = _ik_path(self.robot, pts, q0, self.j6)
        Qd = np.gradient(Q, t, axis=0)
        Qd[0] = np.zeros(6) if from_rest else belt_joint_velocity(Q[0], self.v_belt)
        Qd[-1] = belt_joint_velocity(Q[-1], self.v_belt - UP * vd)
        return Segment("descend", t, Q, Qd, info=dict(kind="descend", accel=ta, desc=t_desc, h=h, z_top=self.z_top(z_obj),
                                                      v_contact=vd, profile="topp" if pr is not None else "const"))

    # -- 세그먼트 생성 -----------------------------------------------------
    def follow_segment(self, obj: MovingPoint, t_abs0: float, duration: float, q_init, name="wait") -> Segment:
        """잡은 물체를 든 채 벨트를 따라가며 제자리 hold (대기용). 길이는 GRID_DT 배수로 올림."""
        t = _grid_times(max(GRID_DT, grid_ceil(duration)))
        n = len(t)
        pts = np.array([obj.at(t_abs0 + tk) for tk in t])
        Q = _ik_path(self.robot, pts, q_init, self.j6)
        Qd = np.gradient(Q, t, axis=0) if n > 2 else np.zeros_like(Q)
        return Segment(name, t, Q, Qd, info=dict(kind="follow"))

    def pick_segment(self, obj: MovingPoint, t_abs0: float, q_arrive, wait: float = 0.0, qd_arrive=None) -> Tuple[Segment, ThrowBoundary]:
        """물체 표면에 하강 속도로 닿은 상태(아크 끝) → 멈춤 없이 잡아 올린다. 내내 벨트 추종.
        topp: 접촉 위치의 시간 최적 프로파일 — 눌림은 도착 속도(qd_arrive 의 z 성분, 없으면 최적 접촉 속도)에서 바닥 0 까지
        하강 프로파일의 나머지를 같은 시간 비로, 상승은 바닥 0 → z_top +LIFT_SPEED 시간 최적. 격자 올림 몫은 바닥 정지에 더한다.
        const (아래 설명):

        z 프로파일 (물체 표면 = 0):  ① 등감속 눌림: ż −v_d → 0, 깊이 −press  (T1 = 2·press/v_d)
                                  ② (옵션 hold: −press 에서 벨트 추종 dwell)
                                  ③ 등가속 상승: −press → +lift, ż 0 → +v_l    (T3 = 2·(lift+press)/v_l)
        xy: ①② 는 물체를 그대로 추종, ③ 은 추종 성분이 선형으로 0 → 끝 상태 = 던지기 시작 경계 (수직 상승 v_l).
        suction ON = t=0 (접촉). wait 는 hold 에 더한다 (작업영역 진입 대기)."""
        z_obj = float(obj.at(t_abs0)[2])
        pr = self._profile(obj.at(t_abs0))
        if pr is not None:
            return self._pick_segment_topp(obj, t_abs0, q_arrive, wait, qd_arrive, pr)
        vd, vl, d, L = self.v_desc_of(z_obj), self.lift_speed, self.press_of(z_obj), self.h_of(z_obj)   # L: 윗면 → z_top 상승 높이
        # 구간 길이를 GRID_DT 배수로 올리고 등감속/등가속도는 속도 연속(접촉 −vd, 끝 +vl)을 지키도록 그 시간에 맞춘다
        # → 눌림 깊이·상승 높이가 ≤ 0.3 mm 커진다 (d_eff = vd·t1/2, L_eff + d_eff = vl·t3/2)
        t1 = grid_ceil(2.0 * d / vd) if (vd > 0 and d > 0) else 0.0
        th = grid_ceil(self.hold_s + max(0.0, wait)) if (self.hold_s + max(0.0, wait)) > 1e-9 else 0.0
        t3 = grid_ceil(2.0 * (L + d) / vl)
        T = t1 + th + t3
        t = _grid_times(T)
        n = len(t)
        pts = np.empty((n, 3))
        a1 = (vd / t1) if t1 > 0 else 0.0                                 # 눌림 등감속도 (−vd → 0, 깊이 d = vd·t1/2)
        d = 0.5 * vd * t1 if t1 > 0 else 0.0
        a3 = vl / t3                                                      # 상승 등가속도 (0 → vl, 높이 L + d = vl·t3/2)
        for k, tk in enumerate(t):
            if tk <= t1:                                                  # ① 눌림
                pts[k] = obj.at(t_abs0 + tk) + UP * (-vd * tk + 0.5 * a1 * tk * tk)
            elif tk <= t1 + th:                                           # ② hold (벨트 추종)
                pts[k] = obj.at(t_abs0 + tk) + UP * (-d)
            else:                                                         # ③ 상승 — xy 는 끝까지 벨트 추종 (2026-10-29)
                tau = tk - t1 - th
                pts[k] = obj.at(t_abs0 + tk) + UP * (-d + 0.5 * a3 * tau * tau)
        Q = _ik_path(self.robot, pts, q_arrive, self.j6)
        Qd = np.gradient(Q, t, axis=0)
        # 첫 샘플 = 도착 상태 속도(하강+추종), 끝 샘플 = 던지기 시작 경계(벨트 추종 + 수직 상승) 로 정확히 맞춘다
        Qd[0] = belt_joint_velocity(Q[0], self.v_belt - UP * vd)
        start = ThrowBoundary(Q[-1], belt_joint_velocity(Q[-1], self.v_belt + UP * vl))
        Qd[-1] = start.qd
        seg = Segment("pick", t, Q, Qd, suction=[(0.0, True)],
                      info=dict(kind="pick", press=t1, hold=th, lift=t3, wait=wait, press_depth=d, lift_h=L, z_top=self.z_top(z_obj)))
        return seg, start

    def _pick_segment_topp(self, obj: MovingPoint, t_abs0: float, q_arrive, wait: float, qd_arrive, pr) -> Tuple[Segment, ThrowBoundary]:
        """pick_segment 의 topp 판: ① 눌림 = 하강 프로파일의 접촉 뒤 나머지 (도착 속도에 맞춘 시간 비) ② 바닥 정지 (hold + wait +
        격자 올림 몫, z 속도 0 이라 이음이 매끄럽다) ③ 상승 = 바닥 0 → z_top +LIFT_SPEED 시간 최적 (xy 추종 성분은 선형으로 0 →
        끝 상태 = 던지기 시작 경계, 수직 상승)."""
        from gp8_control.throwing.throw_planner import tcp_of
        dn, up, z_top, z_floor = pr
        z_obj = float(obj.at(t_abs0)[2])
        L, d = z_top - z_obj, z_obj - z_floor                             # 윗면 → z_top, 윗면 → 바닥
        f, s_c, T_c, v_opt = self._contact_of(dn, z_top, z_floor, z_obj)
        vd = v_opt
        if qd_arrive is not None and np.any(np.asarray(qd_arrive, float)[:6]):
            q_a, qd_a = np.asarray(q_arrive, float)[:6], np.asarray(qd_arrive, float)[:6]
            eps = 1e-4                                                    # 도착 관절 속도의 TCP z 성분 = 하강 세그먼트의 접촉 속도
            vd = max(0.0, -float((tcp_of(q_a + eps * qd_a)[2] - tcp_of(q_a - eps * qd_a)[2]) / (2 * eps)))
        kt = vd / v_opt if v_opt > 1e-9 else 1.0
        kt = max(kt, 0.25)                                                # 도착 하강 속력이 거의 0 (앞 던지기가 v_in≈0 으로 끝남) 이면 눌림 구간이
                                                                          # 끝없이 길어져 2 ms 격자 반복이 멈춘다 (10-30 FIG4 head 순서 결정) → 기준의 4 배까지
        T_p = (dn.duration - T_c) / kt if (kt > 1e-6 and d > 1e-9) else 0.0
        f_up = (z_top - z_floor) / up.s[-1] if up.s[-1] > 1e-9 else 1.0
        T_u = up.duration
        th_req = (self.hold_s if self.hold_s > 1e-9 else 0.0) + max(0.0, wait)
        T = grid_ceil(T_p + th_req + T_u)
        th = T - T_p - T_u                                                # ≥ th_req (격자 올림 몫 포함)
        t = _grid_times(T)
        pts = np.empty((len(t), 3))
        base = obj.at(t_abs0 + T_p + th)
        for k, tk in enumerate(t):
            if tk <= T_p:                                                 # ① 눌림: 윗면 아래로 (하강 프로파일의 나머지)
                pts[k] = obj.at(t_abs0 + tk) - UP * (f * float(dn.at(T_c + kt * tk)[0]) - L)
            elif tk <= T_p + th:                                          # ② 바닥 정지 (벨트 추종)
                pts[k] = obj.at(t_abs0 + tk) - UP * d
            else:                                                         # ③ 상승 — xy 는 끝까지 벨트 추종 (2026-10-29 사용자: "하강과 상승 할 때 항상
                tau = tk - T_p - th                                       #    수평 속도 벨트 속도로 제약"; 종전엔 추종 성분을 선형으로 0 까지 줄였다)
                pts[k] = obj.at(t_abs0 + tk) + UP * (-d + f_up * float(up.at(tau)[0]))
        pts[0] = obj.at(t_abs0)
        Q = _ik_path(self.robot, pts, q_arrive, self.j6)
        Qd = np.gradient(Q, t, axis=0)
        Qd[0] = belt_joint_velocity(Q[0], self.v_belt - UP * vd)
        v_end = float(getattr(up, "v_end", self.lift_speed))           # 양끝 자유: 그 칸의 LP 탈출 속도 (던지기 시작 경계)
        start = ThrowBoundary(Q[-1], belt_joint_velocity(Q[-1], self.v_belt + UP * v_end))   # 던지기 시작 = 벨트 추종 + 수직 상승
        Qd[-1] = start.qd
        seg = Segment("pick", t, Q, Qd, suction=[(0.0, True)],
                      info=dict(kind="pick", press=T_p, hold=th, lift=T_u, wait=wait, press_depth=d, lift_h=L, z_top=z_top,
                                v_contact=vd, profile=PICK_PROFILE, v_end=v_end))
        return seg, start

    # -- 분리 방식 (GP8_SEP) ---------------------------------------------------
    def _sep_table(self):
        if self._sep_tab is None:
            from gp8_control.grasping.sep_pick import Table
            self._sep_tab = Table()
        return self._sep_tab

    def _sep_lines(self):
        if self._sep_line is None:
            from gp8_control.grasping.sep_pick import LineCache
            self._sep_line = LineCache(float(np.linalg.norm(self.v_belt)), robot=self.robot, j6=self.j6)
        return self._sep_line

    @staticmethod
    def _vz(q, qd) -> float:
        """관절 상태의 TCP 수직 속도 [m/s]."""
        from gp8_control.throwing.throw_planner import tcp_of
        q = np.asarray(q, float)[:6]; qd = np.asarray(qd, float)[:6]; eps = 1e-4
        return float((tcp_of(q + eps * qd)[2] - tcp_of(q - eps * qd)[2]) / (2 * eps))

    def sep_prewarm(self) -> None:
        """분리 방식 첫 사용 비용을 시작 때 치른다 (10-30 실시간 시뮬: 첫 꼬리가 NLP 조립 12 s 로 마감을 넘겨 0 반복으로 끝났다)."""
        from gp8_control.grasping import sep_pick as SP
        from gp8_control.throwing import sep_throw as ST
        t0 = time.time()
        tw = {"ramp": ST.prewarm("ramp")}                                  # 조립 + IPOPT 풀이기 생성까지 (첫 꼬리가 6 s 를 치르지 않게, 10-30)
        ST.sep_entry(np.array([0.4, 0.2]), np.array([0.4, 0.0]), np.array([1.35, 0.0, -0.06]))
        ST.warm_entry(np.array([0.4, 0.2]), np.array([0.4, 0.0]), np.array([1.35, 0.0, -0.06]))   # 던지기 warm DB 적재 (첫 풀이에 2.7 s, 10-30)
        if SEP_PICKFIRST:
            if ST.ALT:                                                      # 번갈아 풀기 판 (sep_alt): 조립 + IPOPT 첫 호출까지
                from gp8_control.throwing import sep_alt as SA
                tw["alt"] = SA.prewarm()
            else:
                tw["hist"] = ST.prewarm("hist")
        if CYCLE:                                                           # 사이클 NLP 조립 + 사이클 DB 적재 (첫 사이클이 치르지 않게)
            from gp8_control.throwing import cycle_nlp as CY
            tw["cycle"] = CY.prewarm()
            if CYCLE_DB:
                t1 = time.time(); n_db = sum(len(v[1]) for v in _cycle_db().values()); tw[f"cycle_db {n_db}"] = time.time() - t1
        if SEP_BS_DB:
            t1 = time.time(); tw[f"stop_db {len(_bs_db()[1])}"] = time.time() - t1
        if self.log is not None:
            self.log.info("분리 NLP 풀이기 예열 (조립 + 풀이기 생성): " + ", ".join(f"{k} {v:.1f}s" for k, v in tw.items()))
        tb = self._sep_table()
        tw["lines"] = self._sep_lines().prewarm(min(tb.rows) * 0.01, (max(tb.rows) + 1) * 0.01)
        SP._qp(np.eye(2), np.zeros(2), np.ones((1, 2)), np.array([-1.0]), np.array([1.0]))
        if self.log is not None:
            self.log.info(f"분리 방식 예열 {time.time() - t0:.1f}s (NLP 조립·DB·잡기 LP 수직선 {len(self._sep_lines().data)} 개 {tw['lines']:.1f}s)")

    def sep_vguess(self, r: float, z: Optional[float] = None) -> float:
        lim = self._sep_table().vout_lim(r, z)                                   # 잡기 LP 가 풀리는 상승 끝 속력 상한
        return float(np.clip(SEP_VOUT_DEF, 0.3, lim[0] if lim is not None else SEP_VOUT_DEF))

    def path_vel_basis(self, q):
        """잡기 IK 경로(툴 수직, J6 고정)를 따라 TCP 가 벨트 속도·위쪽 1 m/s 로 움직일 때의 관절 속도 (중앙 차분) — 잡기 LP 명령과 같은 규약."""
        from gp8_control.throwing.throw_planner import tcp_of
        q = np.asarray(q, float)[:6]; p = np.asarray(tcp_of(q), float); eps = 1e-4
        out = []
        for d in (self.v_belt, UP):
            qa, qb = self._pick_ik(p + d * eps, q), self._pick_ik(p - d * eps, q)
            if qa is None or qb is None:
                return None
            out.append((np.asarray(qa, float)[:6] - np.asarray(qb, float)[:6]) / (2 * eps))
        return tuple(out)

    def sep_pick_segment(self, obj: MovingPoint, t_abs0: float, q_hover, v_in: float, v_out: float, N: Optional[int] = None,
                         bisect: bool = True, hist=None, v_out_free: bool = True):
        """hover(하강 속력 v_in 으로 지남) → 실제 윗면 접촉 → z_top(상승 속력 v_out 으로 지남): 잡기 LP 명령. xy 는 내내 물체(벨트 속도).
        N 을 주면 그 길이 (던지기 시작 시각을 먼저 정한 경우). 리턴 (Segment 'pick', 던지기 시작 경계)."""
        from gp8_control.grasping import sep_pick as SP
        from gp8_control.throwing import sep_throw as ST
        p0 = obj.at(t_abs0); z_obj = float(p0[2]); r = float(np.hypot(p0[0], p0[1]))
        bnd = self._sep_lines().get(r)                                         # 실제 반경 (1 mm) 수직선의 한계 (11-02, 종전 1 cm 칸 아래쪽 끝)
        from gp8_control.throwing.throw_planner import tcp_of
        q_hover = np.asarray(q_hover, float)[:6]
        p_h = np.asarray(tcp_of(q_hover), float)
        z0 = float(p_h[2]) if abs(float(p_h[2]) - SP.Z_H) < 0.01 else SP.Z_H      # 앞 던지기의 마지막 샘플 높이 (격자 올림 몫)
        if SP.RT:                                                             # 11-04 논문 version2 식 6: B-스플라인 잡기, 실제 시간축
            pre_Q, hist_b, q_b, t_b, v_b = self._rt_lead(obj, t_abs0, q_hover, hist, v_in)
            tpk = self._sep_table().t_pick(r, max(0.0, float(v_b)), float(v_out), z_obj)
            bs = self._sep_bs(obj, t_b, q_b, max(0.0, float(v_b)), float(v_out), float(tpk) if tpk else 0.30, z_obj, hist_b, v_out_free)
            if bs.get("ok"):
                n_pre = 0 if pre_Q is None else len(pre_Q) - 1
                Q = bs["Q"] if pre_Q is None else np.vstack([pre_Q[:-1], bs["Q"]])
                N = len(Q) - 1; t = np.arange(N + 1) * GRID_DT
                Qd = np.gradient(Q, t, axis=0)
                start = ThrowBoundary(Q[-1], (Q[-1] - Q[-2]) / GRID_DT)
                start.p_safe = np.r_[obj.at(t_abs0 + t[-1])[:2], SP.Z_H + 0.10]   # RT 정지 대체의 목표 (잡은 물체 위)
                Qd[-1] = start.qd
                seg = Segment("pick", t, Q, Qd, suction=[(n_pre * GRID_DT, True)],
                              info=dict(kind="pick", sep=True, v_in=float(v_b), v_out=float(bs["v_out"]), N=int(N), k_touch=int(bs["k_t"]) + n_pre,
                                        lead_pre_s=n_pre * GRID_DT,
                                        z_top=float(SP.Z_H), z_obj=z_obj, profile="sep_bs", P_cmd=None, z_lp=None,
                                        v3={k: bs.get(k) for k in ("ok", "why", "T", "e_w", "e_rot", "z_gap", "z_min", "vr", "ar", "it", "s")}))
                return seg, start
            msg = (f"잡기 B-스플라인 불가 ({bs.get('why')}, T {bs.get('T', float('nan')):.3f}, e_w {1e3 * bs.get('e_w', float('nan')):.1f} mm, "
                   f"z_min {1e3 * bs.get('z_min', float('nan')):.1f} mm, 물체 ({p0[0]:.3f}, {p0[1]:.3f}, {z_obj:.3f}), v_in {v_b:.2f}, "
                   f"앞 정렬 {'있음' if pre_Q is not None else '없음'}, {bs.get('s', 0):.1f}s)")
            if self.log is not None:
                self.log.warn("  " + msg)
            raise ValueError(msg)                                             # RT: 종전 잡기로 넘어가지 않는다 (실제 시간축과 안 맞는 상태를 남김)
        res = None
        if N is None:                                                          # 길이를 표의 잡기 시간 추정으로 먼저 (이분법은 LP 수십 번 — head 6 s)
            tpk = self._sep_table().t_pick(r, max(0.0, float(v_in)), float(v_out), z_obj)
            if tpk is not None:
                N0 = int(np.ceil(tpk / GRID_DT)) + 3
                for N_try in (N0, N0 + 6, N0 + 15):
                    res = ST.call(SP.solve, r, max(0.0, float(v_in)), float(v_out), z_o=z_obj, N=N_try, bounds=bnd, z0=z0,
                                  near=True)
                    if res is not None:
                        break
        if res is None and (bisect or N is not None):                    # 길이 이분법 (큰 LP 수십 번, 안 풀리는 조합이면 수 s — 10-30 head 6.7 s)
            res = ST.call(SP.solve, r, max(0.0, float(v_in)), float(v_out), z_o=z_obj, N=N, bounds=bnd, z0=z0)
        if res is None:
            raise ValueError(f"분리 잡기 LP 불가 (r {r:.3f}, v_하강 {v_in:.2f}, v_상승 {v_out:.2f}, N {N})")
        N, k_t, zc = res
        t = np.arange(N + 1) * GRID_DT
        pts = np.array([obj.at(t_abs0 + tk) for tk in t]); pts[:, 2] = zc
        # 첫 샘플 = 앞 계획의 마지막 자세 (이음 연속). hover 와 물체 추정이 어긋나 있으면 (앞 꼬리 때의 추정 → 지금 추정) 그 차를 잡기 동안
        # 5차 곡선으로 0 까지 줄인다 — 종전엔 1 cm 안이면 첫 샘플만 hover 로 두어 2 ms 사이에 그만큼 뛰었다.
        off = np.r_[p_h[:2] - pts[0, :2], 0.0]
        if np.linalg.norm(off) < HOVER_DIRECT_M:
            s_ = t / t[-1]; pts = pts + off[None] * (1 - (10 * s_ ** 3 - 15 * s_ ** 4 + 6 * s_ ** 5))[:, None]
        Q = _ik_path(self.robot, pts, q_hover, self.j6)
        if np.linalg.norm(p_h - pts[0]) < 1e-6:
            Q[0] = q_hover
        v3 = None
        if SP.V3:                                                             # 잡기 v3 (11-04): 6축 명령 + v_out, 조건은 실제 궤적에
            v3 = self._sep_v3(obj, t_abs0, q_hover, max(0.0, float(v_in)), float(v_out), int(N), r, z_obj, z0, bnd, hist, v_out_free)
            if v3.get("ok"):
                N, Q, k_t, v_out = int(v3["N"]), v3["Q"], int(v3["k_t"]), float(v3["v_out"])
                t = np.arange(N + 1) * GRID_DT
            elif self.log is not None:
                self.log.warn(f"  잡기 v3 불가 → 종전 잡기 LP 명령 ({v3.get('tried')})")
        Qd = np.gradient(Q, t, axis=0)
        Qd[0] = belt_joint_velocity(Q[0], self.v_belt - UP * v_in)
        if v3 is not None and v3.get("ok"):
            qd_end = (Q[-1] - Q[-2]) / GRID_DT                                # 잡기 끝 명령 속도 그대로 (던지기 시작 = 이음)
        else:
            vb = self.path_vel_basis(Q[-1])
            qd_end = (vb[0] + vb[1] * v_out) if vb is not None else belt_joint_velocity(Q[-1], self.v_belt + UP * v_out)
        start = ThrowBoundary(Q[-1], qd_end)
        Qd[-1] = start.qd
        A_r, b_r = SP.realized_affine(int(len(zc) - 1), max(0.0, float(v_in)), z0=z0)
        seg = Segment("pick", t, Q, Qd, suction=[(0.0, True)],
                      info=dict(kind="pick", sep=True, v_in=float(v_in), v_out=float(v_out), N=int(N), k_touch=int(k_t), z_top=float(SP.Z_H),
                                z_obj=z_obj, profile="sep", P_cmd=pts, z_lp=A_r @ zc + b_r,
                                v3=None if v3 is None else {k: v3.get(k) for k in ("ok", "tried", "e_c", "e_w", "e_rot", "z_gap", "vr", "ar", "it", "s")}))
        return seg, start

    def _rt_lead(self, obj: MovingPoint, t_abs0: float, q_hover, hist, v_in: float):
        """잡기 시작 실제 상태 (실제 시각 D — 이전 명령만으로 정해짐) 가 물체 위 h 가 아니면 (xy > 2 mm 또는 z > 5 mm 어긋남), 명령이 물체를
        L 만큼 앞서 벨트 속도로 따라가는 구간 (0.12 s 이어 붙임 + 필터 기억 등속) 을 잡기 앞에 둔다. 리턴 (앞 구간 명령 또는 None, 잡기 이력,
        잡기 첫 명령, 잡기 시작 시각, 잡기 v_in)."""
        from gp8_control.grasping import sep_pick as SP
        from gp8_control.grasping.workspace_gate import tcp_batch
        hk = SP._h(); pad = len(hk); L = float(np.arange(pad) @ hk) * GRID_DT; ND = int(np.argmax(np.abs(hk) > 1e-15))
        vb = np.asarray(self.v_belt, float)
        if hist is None or len(hist) < 2:                                     # 이력 없음: hover 를 지나 벨트 속도로 따라오던 명령으로 가정
            from gp8_control.throwing.throw_planner import tcp_of
            p_h = np.asarray(tcp_of(q_hover), float); th_ = -np.arange(pad - 1, 0, -1) * GRID_DT
            hist = _ik_path(self.robot, np.array([p_h + vb * tk for tk in th_])[::-1], q_hover, self.j6)[::-1]
        hist = np.asarray(hist, float)[:, :6]
        if len(hist) < pad - 1:                                               # 기록이 필터 길이보다 짧으면 첫 명령으로 채운다
            hist = np.vstack([np.repeat(hist[:1], pad - 1 - len(hist), 0), hist])
        U = np.vstack([hist[-(pad - 1):], q_hover[None]])
        q_D = np.array([hk[ND:] @ U[::-1][:len(hk) - ND, j] for j in range(6)])      # 실제 시각 D 의 실제 관절 (명령 0 까지만)
        q_D1 = np.array([hk[ND - 1:] @ U[::-1][:len(hk) - ND + 1, j] for j in range(6)])  # 한 샘플 앞 (실제 속도용)
        p_D, p_D1 = tcp_batch(np.stack([q_D, q_D1])); o_D = obj.at(t_abs0 + ND * GRID_DT)
        if np.abs(p_D[:2] - o_D[:2]).max() <= 0.002 and abs(p_D[2] - SP.Z_H) <= 0.005:
            v_real = max(0.0, -(p_D[2] - p_D1[2]) / GRID_DT)                  # 실제 하강 속력 (RT 에선 명령 속도와 다르다 — 초기값용)
            return None, hist, q_hover, t_abs0, v_real
        from gp8_control.throwing.throw_planner import tcp_of
        p_h = np.asarray(tcp_of(q_hover), float)
        n = int(round((0.12 + (pad - 1) * GRID_DT) / GRID_DT)); tt = np.arange(n + 1) * GRID_DT
        u_ = np.clip(tt / 0.12, 0, 1); b = 10 * u_ ** 3 - 15 * u_ ** 4 + 6 * u_ ** 5
        tgt = np.array([np.r_[obj.at(t_abs0 + tk + L)[:2], SP.Z_H] for tk in tt])
        cur = np.array([p_h + vb * tk for tk in tt])
        pre = _ik_path(self.robot, (1 - b)[:, None] * cur + b[:, None] * tgt, q_hover, self.j6); pre[0] = q_hover
        if self.log is not None:
            self.log.info(f"  잡기 앞 실제 정렬: 실제 툴이 물체와 xy {1e3 * np.abs(p_D[:2] - o_D[:2]).max():.1f} mm, z {1e3 * (p_D[2] - SP.Z_H):+.1f} mm "
                          f"어긋남 → 명령을 L 만큼 앞서 {tt[-1]:.2f}s 따라감")
        return pre, np.vstack([hist, pre[:-1]]), pre[-1], t_abs0 + tt[-1], 0.0

    def _sep_bs(self, obj: MovingPoint, t_abs0: float, q_hover, v_in: float, v_out: float, T0: float, z_obj: float, hist,
                v_out_free: bool) -> dict:
        """잡기 B-스플라인 (논문 식 6). 초기 명령: 실제 높이가 h 에서 (−v_in) 윗면을 지나 h (+v_out) 로 가는 3 차 Hermite 를
        등속 지연 (L − D) 만큼 앞당긴 명령, xy 는 물체를 L 만큼 앞선 점, 앞 명령 속도에서 0.12 s 에 걸쳐 이어 붙임. 끝나면 T 를 격자에 맞춰 한 번 더."""
        from gp8_control.grasping import sep_pick as SP
        from gp8_control.throwing import sep_throw as ST
        t_c = time.time()
        hk = SP._h(); pad = len(hk); L = float(np.arange(pad) @ hk) * GRID_DT; ND = int(np.argmax(np.abs(hk) > 1e-15))
        lead = int(round(L / GRID_DT)) - ND
        if hist is None or len(hist) < 2:
            from gp8_control.throwing.throw_planner import tcp_of
            p_h = np.asarray(tcp_of(q_hover), float); th_ = -np.arange(pad - 1, 0, -1) * GRID_DT
            ph = np.array([np.r_[p_h[:2] + np.asarray(self.v_belt, float)[:2] * tk, min(p_h[2] - v_in * tk, p_h[2] + 0.2)] for tk in th_])
            hist = _ik_path(self.robot, ph[::-1], q_hover, self.j6)[::-1]
        hist = np.asarray(hist, float)[:, :6]
        vo0 = float(np.clip(v_out, 0.3, SP.BS_VOUT_MAX)) if v_out > 0.05 else 1.0
        P_db = None
        if SEP_BS_DB and v_in < 1e-6:                                         # 정지 출발: DB 의 가장 가까운 해 (첫 제어점만 지금 시작) — SLP 3~10 반복
            Kb, Sb = _bs_db(); po = obj.at(t_abs0)
            if len(Kb):
                dist = np.abs(Kb - np.r_[po[:2], z_obj] / _BS_SC).max(1); j = int(np.argmin(dist))
                if dist[j] > 1.5:
                    return dict(ok=False, why=f"정지 잡기 DB 범위 밖 (가장 가까운 칸 {dist[j]:.1f} 격자)", s=time.time() - t_c)
                P_db = Sb[j][0].copy(); P_db[0] = q_hover; T0 = Sb[j][1]; vo0 = Sb[j][2]
        N0 = max(40, int(round(T0 / GRID_DT))); jc = min(max(int(N0 * v_in / (v_in + vo0 + 1e-9)), 10), N0 - 10)

        def herm(j, j0, z0_, v0_, j1, z1, v1):
            s = (j - j0) / (j1 - j0); TT = (j1 - j0) * GRID_DT
            return (2 * s ** 3 - 3 * s ** 2 + 1) * z0_ + (s ** 3 - 2 * s ** 2 + s) * TT * v0_ + (-2 * s ** 3 + 3 * s ** 2) * z1 + (s ** 3 - s ** 2) * TT * v1
        zr = lambda j: (herm(j, 0, SP.Z_H, -v_in, jc, z_obj, 0.0) if j <= jc else
                        (herm(j, jc, z_obj, 0.0, N0, SP.Z_H, vo0) if j <= N0 else SP.Z_H + vo0 * (j - N0) * GRID_DT))
        tt = np.arange(N0 + 1) * GRID_DT
        if P_db is None:
            try:
                Q0 = _ik_path(self.robot, np.array([np.r_[obj.at(t_abs0 + tk + L)[:2], zr(k + lead)] for k, tk in enumerate(tt)]), q_hover, self.j6)
            except ValueError as e:
                return dict(ok=False, why=f"초기 IK ({e})", s=time.time() - t_c)
            qd_h = (q_hover - hist[-1]) / GRID_DT; u_ = np.clip(tt / 0.12, 0, 1); w_ = 1 - (10 * u_ ** 3 - 15 * u_ ** 4 + 6 * u_ ** 5)
            Q0 = Q0 + ((q_hover[None] + qd_h[None] * tt[:, None]) - Q0) * w_[:, None]; Q0[0] = q_hover
            P = SP.bs_fit(Q0, tt[-1], q_hover)
        else:
            P = P_db; tt = np.array([0.0, T0])                                # solve_bs 는 tt[-1] (= T) 만 쓴다
        obj_at = lambda tk: obj.at(t_abs0 + tk)
        from gp8_control.throwing.nlp import throw_nlp as _T
        from gp8_control.throwing.nlp.planner import PLANNER_SIGN as _SG
        if CYCLE:                                                             # 사이클: 다음은 사이클 NLP (실제 관절 범위) — 좁은 던지기 상자는 상승을 막는다
            _lo = np.maximum(_T.GP8_Q_MIN, _T.Q_LO_REAL) + np.deg2rad(0.2); _hi = np.minimum(_T.GP8_Q_MAX, _T.Q_HI_REAL) - np.deg2rad(0.2)
        else:
            _lo, _hi = _T.Q_LO, _T.Q_HI
        qlim = (np.minimum(_lo * _SG, _hi * _SG), np.maximum(_lo * _SG, _hi * _SG))   # 관절 상자 (로봇 규약)
        out = ST.call(SP.solve_bs, hist, P, float(tt[-1]), obj_at, z_obj, SP.Z_H, vo0, self.robot.velocity_limits, self.j6, v_out_free=v_out_free,
                      qlim=qlim)
        if out.get("ok"):                                                     # T 를 2 ms 격자에 (던지기가 격자 샘플에서 이어 받는다)
            Tg = float(np.ceil(out["T"] / GRID_DT - 1e-9) * GRID_DT)
            o2 = ST.call(SP.solve_bs, hist, out["P"], Tg, obj_at, z_obj, SP.Z_H, out["v_out"], self.robot.velocity_limits, self.j6,
                         iters=8, fix_T=True, v_out_free=v_out_free, qlim=qlim)
            if o2.get("ok"):
                out = o2
            out["Q"] = SP.bs_samples(out["P"], float(np.floor(out["T"] / GRID_DT + 1e-9) * GRID_DT) if not o2.get("ok") else Tg)
            out["Q"][0] = q_hover
        out["s"] = time.time() - t_c
        return out

    def _sep_v3(self, obj: MovingPoint, t_abs0: float, q_hover, v_in: float, v_out: float, N0: int, r: float, z_obj: float, z0: float, bnd,
                hist, v_out_free: bool) -> dict:
        """잡기 v3 (논문 식 7): 길이 사다리 N0, +5, +15 — 각 길이의 초기 명령 = 종전 높이 LP 의 z + 물체를 필터 평균 지연만큼 앞선 xy
        (초기값만; 조건은 실제 시각의 물체), 앞 명령 속도에서 0.12 s 에 걸쳐 이어 붙임. 이력이 없으면 같은 앞선 수직선으로 가정."""
        from gp8_control.grasping import sep_pick as SP
        from gp8_control.throwing import sep_throw as ST
        t_c = time.time()
        hk = SP._h(); pad = len(hk); lag = float(np.arange(pad) @ hk) * GRID_DT    # 필터 평균 지연 L
        if hist is None or len(hist) < 2:                                     # 가정 이력: 시작 자세를 지나 벨트 속도로 따라가며 v_in 으로 내려오던 명령
            from gp8_control.throwing.throw_planner import tcp_of
            p_h = np.asarray(tcp_of(q_hover), float); th_ = -np.arange(pad - 1, 0, -1) * GRID_DT
            ph = np.array([np.r_[p_h[:2] + np.asarray(self.v_belt, float)[:2] * tk, min(p_h[2] - v_in * tk, p_h[2] + 0.2)] for tk in th_])
            hist = _ik_path(self.robot, ph[::-1], q_hover, self.j6)[::-1]
        hist = np.asarray(hist, float)[:, :6]
        qd_h = (q_hover - hist[-1]) / GRID_DT
        tried = []
        # 11-04 사용자 v_in 상한 없앰 (V2): 빠르게 내려와도 명령은 0.05 s 안에 멈추므로, 길이를 풀릴 때까지 늘린다
        ladder = (N0, N0 + 5, N0 + 15) if not ST.V2 else (N0, N0 + 5, N0 + 15, N0 + 30, N0 + 50, N0 + 80)
        for N_try in ladder:
            res = ST.call(SP.solve, r, v_in, v_out, z_o=z_obj, N=N_try, bounds=bnd, z0=z0, near=True)
            if res is None and ST.V2:                                         # 종전 높이 LP 는 초기값용 — 그 v_in 에서 안 풀리면 정지 출발 높이를 초기값으로
                res = ST.call(SP.solve, r, 0.0, v_out, z_o=z_obj, N=N_try, bounds=bnd, z0=z0, near=True)
            if res is None:
                tried.append((N_try, "높이 LP 불가")); continue
            tt = np.arange(N_try + 1) * GRID_DT
            pts0 = np.array([obj.at(t_abs0 + tk + lag) for tk in tt]); pts0[:, 2] = res[2]
            Q0 = _ik_path(self.robot, pts0, q_hover, self.j6)
            u_ = np.clip(tt / 0.12, 0.0, 1.0); w_ = 1 - (10 * u_ ** 3 - 15 * u_ ** 4 + 6 * u_ ** 5)
            Q0 = Q0 + ((q_hover[None] + qd_h[None] * tt[:, None]) - Q0) * w_[:, None]
            Q0[0] = q_hover
            K = N_try + 1 + SP.M_CONT
            obj_xy = np.array([obj.at(t_abs0 + k * GRID_DT)[:2] for k in range(K)])
            out = ST.call(SP.solve_v3, hist, Q0, obj_xy, z_obj, SP.Z_H, v_out, self.robot.velocity_limits, self.j6, v_out_free=v_out_free)
            tried.append((N_try, "ok" if out.get("ok") else out.get("why"), round(1e3 * float(out.get("e_c", np.nan) or np.nan), 2),
                          round(1e3 * float(out.get("e_w", np.nan) or np.nan), 2), round(float(out.get("ar", np.nan) or np.nan), 2)))
            if out.get("ok"):
                out.update(N=N_try, tried=tried, s=time.time() - t_c)
                return out
        return dict(ok=False, tried=tried, s=time.time() - t_c)

    def sep_refine(self, pick: Segment, obj: MovingPoint, t_abs0: float, nxt: Optional[Segment], hist=None) -> Segment:
        """분리 잡기 명령 보정 (sep_pick.correct): 앞 명령(hist — 앱이 준 실제 이력, 없으면 정지 가정)과 뒤 세그먼트(nxt)를 그대로 두고
        잡기 명령만 옮겨, 필터를 지난 실제 TCP 가 물체 xy (스트림 선행만큼 이른 시각) 와 LP 의 실제 z 를 따르게. 양끝 자세·속도는 그대로."""
        from gp8_control.grasping import sep_pick as SP
        from gp8_control.grasping.workspace_gate import tcp_batch
        from gp8_control.throwing import sep_throw as ST
        if not SEP_CORR or not pick.info.get("sep") or pick.info.get("P_cmd") is None:
            return pick
        t0c = time.time()
        if SP.CORR_STOP is not None and t0c > SP.CORR_STOP - 0.3:          # 실시간 꼬리: 잡기 시작 전에 끝낼 시간이 없으면 보정 없이
            pick.info = dict(pick.info, corr=dict(ok=False, fail=f"시간 부족 ({SP.CORR_STOP - t0c:+.2f}s)"))
            return pick
        z_lp = np.asarray(pick.info["z_lp"], float); K = len(z_lp)
        lead = SP.lead_s()
        xy = np.array([obj.at(t_abs0 + k * GRID_DT - lead)[:2] for k in range(K)])
        fut = None if nxt is None else np.asarray(nxt.q, float)[1:]
        if os.environ.get("GP8_SEP_CORR_DUMP"):                               # 진단: 보정 입력 저장
            import pickle
            with open(os.path.join(os.environ["GP8_SEP_CORR_DUMP"], f"corr_{time.time():.3f}.pkl"), "wb") as f:
                pickle.dump(dict(P=pick.info["P_cmd"], Q=pick.q, hist=hist, fut=fut, xy=xy, z_lp=z_lp, z_top=float(pick.info["z_obj"]),
                                 v_lim=np.asarray(self.robot.velocity_limits, float), j6=self.j6), f)
        try:
            P_new, Q_new, inf = ST.call(SP.correct, pick.info["P_cmd"], pick.q, self._pick_ik, tcp_batch, hist, fut, xy, z_lp,
                                           float(pick.info["z_obj"]), self.robot.velocity_limits)
        except Exception as e:                                                # noqa: BLE001 — 보정 실패는 원래 명령으로
            if self.log is not None:
                self.log.warn(f"분리 잡기 보정 실패 ({type(e).__name__}: {e}) — 보정 없이")
            return pick
        b = inf.get("before", {}); a = inf.get("after", b)
        info = dict(pick.info, corr=dict(ok=bool(inf.get("ok")), e_before=b.get("e_contact"), e_after=a.get("e_contact"), z_min=a.get("z_min"),
                                         dP_max=inf.get("dP_max", 0.0), v_ratio=inf.get("v_ratio"), a_ratio=inf.get("a_ratio"), lim0=inf.get("lim_before"),
                                         hist=hist is not None, s=time.time() - t0c, fail=inf.get("fail")))
        if not inf.get("ok"):
            pick.info = info
            return pick
        Qd = np.gradient(Q_new, pick.t, axis=0); Qd[0], Qd[-1] = pick.qd[0], pick.qd[-1]
        info["P_cmd"] = P_new
        return Segment("pick", pick.t, Q_new, Qd, suction=list(pick.suction), info=info)

    def sep_vout_guess(self, src: MovingPoint, p_target, dest: MovingPoint, start_time: float, r_i: float, v_in: float,
                       z_i: Optional[float] = None) -> float:
        """잡기 먼저 판의 상승 끝 속력: 같은 타겟·가까운 시작의 직전 해 → 분리 DB 최근접 해 → 기본값. 잡기 LP 가 풀리는 상한 안으로."""
        from gp8_control.throwing import sep_throw as ST
        tb = self._sep_table()
        if z_i is None:
            z_i = float(src.at(start_time)[2])
        lim = tb.vout_lim(r_i, z_i); vmax = lim[0] - SEP_VOUT_MARGIN if lim is not None else SEP_VOUT_DEF
        tpk = tb.t_pick(r_i, v_in, self.sep_vguess(r_i, z_i), z_i) or 0.3
        p_s = src.at(start_time + tpk)
        prev = self._sep_raw.get(tuple(np.round(np.asarray(p_target, float), 2)))
        if prev is not None and np.linalg.norm(prev["p_start"][:2] - p_s[:2]) < WARM_PREV_M:
            v = float(prev["raw"]["v_out"])
        else:
            se = ST.sep_entry(p_s[:2], dest.at(start_time + tpk + self.t_ref)[:2], p_target)
            v = float(se["v_out"]) if se is not None else self.sep_vguess(r_i, z_i)
        return float(np.clip(v, 0.3, vmax))

    def _sep_pickfirst(self, src: MovingPoint, obj, dest: MovingPoint, start_time: float, q_h, qd_h, v_in: float, z_obj: float, r_i: float,
                       hist, cls: str):
        """잡기 먼저: 잡기 LP (v_out = 이웃 해) → 던지기가 그 명령을 이력으로 바로 이어받는다 (실제 TCP 높이 하한·접촉 상한).
        리턴 (pick, start, t_arc0, arc, plan). 잡기 LP·NLP 실패는 ValueError (호출부가 종전 순서로), 도착 판단은 NotYetReachable."""
        from gp8_control.grasping import sep_pick as SP
        from gp8_control.throwing import sep_throw as ST
        v_g = self.sep_vout_guess(src, obj.target, dest, start_time, r_i, v_in, z_obj)
        err = None
        for v_o in dict.fromkeys([v_g, max(0.3, 0.7 * v_g), min(v_g, SEP_VOUT_DEF), 1.0, 0.3]):   # 속력 상한 표는 v_in·v_out 을 따로 구해
            try:                                                              # 둘 다 크면 LP 가 안 풀린다 (10-30 sep13: v_in 1.65 + v_out 2.59)
                pick, start = self.sep_pick_segment(src, start_time, q_h, v_in, v_o, hist=es.get("hist"))
                v_o = float(pick.info.get("v_out", v_o))
                break
            except ValueError as e:
                err = e
        else:
            raise err
        t_arc0 = start_time + pick.duration
        t_j, end = self.judge_arrival(start, obj.target, dest, t_arc0)
        end.t = t_j
        p_n = dest.at(t_arc0 + t_j); r_n = float(np.hypot(p_n[0], p_n[1]))
        if hist is not None and len(hist):                                   # 잡기 앞 명령 = 앱이 준 실제 이력 (없으면 hover 속도로 직선)
            Uh = np.vstack([np.asarray(hist, float)[:, :6], pick.q])
        else:
            k = np.arange(len(ST.G_H), 0, -1)[:, None] * GRID_DT
            Uh = np.vstack([np.asarray(q_h, float)[None, :6] - np.asarray(qd_h, float)[None, :6] * k, pick.q])
        tc_ = np.arange(int(round(ST.H_S / GRID_DT)) + 1) * GRID_DT          # 잡기 LP 가 가정한 이어짐 (6c) — 하한·상한의 기준 (늘 가능한 해)
        pts = np.array([src.at(t_arc0 + t) for t in tc_]); pts[:, 2] = float(start.tcp[2]) + v_o * np.minimum(tc_, SP.M_RISE * GRID_DT)
        Uc = _ik_path(self.robot, pts, start.q, self.j6); Uc[0] = start.q
        hd = ST.hist_data(Uh, Uc, z_obj, (int(pick.info["k_touch"]) - int(pick.info["N"])) * GRID_DT)
        from gp8_control.throwing.nlp import throw_nlp as _T
        keep_dl = _T.SOLVE_DEADLINE                                          # 잡기 LP·보정은 이미 끝났다 → NLP 는 앱 마감(잡기 시작 − SEP_POST_S) 보다
        if keep_dl is not None:                                              # SEP_PF_EXTEND_S 늦게까지 (궤적 조립·전달만 남음, 10-30)
            _T.SOLVE_DEADLINE = keep_dl + SEP_PF_EXTEND_S
        try:
            arc, plan, _vo, _vi = self.sep_arc_segment(start, obj.target, end, z_obj, r_i, r_n, v_o, cls, hist=hd, dest=dest, t_arc0=t_arc0)
        finally:
            _T.SOLVE_DEADLINE = keep_dl
        return pick, start, t_arc0, arc, plan

    def _fig4_hist(self, es: dict, start: ThrowBoundary, v_fix: float, z_obj: float) -> Optional[dict]:
        """FIG4 꼬리의 실제 이력 판 입력: 이력 = 교체 지점(상승 끝) 앞 명령 (head 의 잡기 LP 포함) + 던지기 시작 자세,
        하한·상한 기준 = 잡기 LP 가 가정한 이어짐 (상승 끝 높이에서 v_out 으로 M_RISE 샘플 더 오름, xy 는 벨트를 따름), 접촉 = k_touch."""
        from gp8_control.grasping import sep_pick as SP
        from gp8_control.throwing import sep_throw as ST
        try:
            pi = es["pick_info"]
            v_o = float(pi.get("v_out") if pi.get("v_out") is not None else v_fix)
            Uh = np.vstack([np.asarray(es["hist"], float)[:, :6], np.asarray(start.q, float)[None, :6]])
            tc_ = np.arange(int(round(ST.H_S / GRID_DT)) + 1) * GRID_DT
            pts = np.repeat(np.asarray(start.tcp, float)[None, :], len(tc_), 0)
            pts[:, :2] += np.asarray(self.v_belt, float)[None, :2] * tc_[:, None]
            pts[:, 2] = float(start.tcp[2]) + v_o * np.minimum(tc_, SP.M_RISE * GRID_DT)
            Uc = _ik_path(self.robot, pts, start.q, self.j6); Uc[0] = start.q
            t_c = (int(pi["k_touch"]) - int(pi["N"])) * GRID_DT
            return ST.hist_data(Uh, Uc, z_obj, t_c)
        except Exception as e:                                               # noqa: BLE001 — 이력 판 입력을 못 만들면 종전(ramp) 던지기
            if self.log is not None:
                self.log.warn(f"FIG4 이력 입력 실패 ({type(e).__name__}: {e}) — 종전 분리 던지기로")
            return None

    def sep_arc_segment(self, start: ThrowBoundary, p_target, end: ThrowBoundary, z_obj: float, r_i: float, r_n: float,
                        v_out_fixed: Optional[float], cls: str = "", hist: Optional[dict] = None,
                        dest: Optional[MovingPoint] = None, t_arc0: Optional[float] = None):
        """분리 던지기 NLP (상승 끝 v_out — 고정 또는 변수 — 와 다음 물체 하강 시작 v_in 변수). 리턴 (Segment, ThrowPlan, v_out, v_in).
        hist: 실제 이력 판 입력 (sep_throw.hist_data). dest·t_arc0: 다음 물체와 아크 시작 시각 — 주면 던지기 시간을 끝 hover·잡은 뒤 자세가
        관절 범위 안인 구간으로 묶고 (tf_window), 풀이 뒤 실제 t_f 로 다시 검사해 안 되면 NotYetReachable (그 물체는 건너뜀)."""
        from gp8_control.throwing import sep_throw as ST
        if not end.exact:
            raise ValueError("분리 던지기는 정확한 움직이는 종점만")
        tf_win = None
        if dest is not None and t_arc0 is not None:
            tf_win = self.tf_window(dest, float(t_arc0), float(end.t))
        key = tuple(np.round(np.asarray(p_target, float), 2))
        prev = self._sep_raw.get(key)
        warm = prev["raw"] if (prev is not None and np.linalg.norm(prev["p_start"] - start.tcp) < WARM_PREV_M) else None
        z_n = float(dest.at(float(t_arc0) + float(end.t))[2]) if (dest is not None and t_arc0 is not None) else None   # 다음 물체 윗면
        tp, vo, vi, raw = ST.call(ST.plan, start.q, p_target, end, self.v_belt, z_obj, r_i, r_n, self._sep_table(), z_n=z_n,
                                  v_out_fixed=v_out_fixed, start_qd=(start.qd if v_out_fixed is not None else None), warm=warm, logger=self.log,
                                  vel_basis=(None if v_out_fixed is not None else self.path_vel_basis(start.q)), hist=hist, tf_win=tf_win)
        if tf_win is not None and not self._arrival_ok(dest, float(t_arc0) + float(tp.t_f))[0]:
            raise self.NotYetReachable(f"던지기 끝(t_f {tp.t_f:.3f}s, 범위 {tf_win[0]:.3f}–{tf_win[1]:.3f}s)에서 잡은 뒤 던지기 시작 자세가 "
                                       f"NLP 관절 한계 밖", None, passed=False)
        self._sep_raw[key] = dict(raw=raw, p_start=np.asarray(start.tcp, float))
        self.n_nlp += 1
        seg = Segment("throw", tp.t, tp.q, tp.qd, suction=[(tp.t_release, False)],
                      info=dict(kind="throw", t_f=tp.t_f, J=tp.info.get("J"), land_err=tp.land_err, cls=cls, sep=True, v_out=vo, v_in=vi,
                                sep_mode=tp.info.get("mode"), lift=tp.info.get("lift")))
        return seg, tp, vo, vi

    # -- 통합 사이클 (11-04 사용자, 논문 version2 식 6, GP8_SEP_CYCLE=1) ------------------------------------------------------------
    def _cycle_inits(self, q0p, dest: MovingPoint, t_abs0: float, t_j: float):
        """cold 초기값: 던지기 INIT_VARIANTS 의 꺾은선 (시작 → 다음 물체 위 hover) + 수직 잡기 (명령을 물체보다 L 앞서, 잡은 뒤 0.2 s 더 상승)."""
        from scipy.interpolate import BSpline as _BS
        from gp8_control.grasping import sep_pick as SP
        from gp8_control.throwing import cycle_nlp as CY
        from gp8_control.throwing.nlp import throw_nlp as _T
        from gp8_control.throwing.nlp.planner import INIT_VARIANTS, PLANNER_SIGN as SG
        hk = SP._h(); L = float(np.arange(len(hk)) @ hk) * GRID_DT; ND = int(np.argmax(np.abs(hk) > 1e-15)); D = ND * GRID_DT
        lead = int(round((L - D) / GRID_DT)); h = SP.Z_H; z_o = float(dest.at(t_abs0)[2])
        out = []; self._cycle_init_err = []
        for k, v in enumerate([v for v in INIT_VARIANTS if v is not None][:CYCLE_N_INIT]):
          for L_xy in (L, 0.0):                                              # 물체를 L 앞선 점이 IK 밖이면 앞서지 않은 점으로
            tf0 = float(np.clip(t_j if t_j else v["T0"], 0.45, 0.9)); chi0 = float(v["chi0"])
            q_hov = self._pick_ik(np.r_[dest.at(t_abs0 + tf0 + L_xy)[:2], h])
            if q_hov is None:
                self._cycle_init_err.append(f"cold{k} hover IK"); continue
            Pt = _T.init_ctrl_points(q0p, np.asarray(v["dq_swing"], float), chi0, p_f=q_hov * SG)
            sp = _BS(_T.KNOTS, Pt.T, _T.DEGREE); n1 = int(round(tf0 / GRID_DT)); Q1 = sp(np.arange(n1 + 1) / n1)
            n2 = int(round(0.36 / GRID_DT)); jc = n2 // 2; n2e = n2 + 60          # 잡은 뒤 0.12 s 더 상승 (먼 물체에서 IK 밖으로 안 나가게)

            def zr(j):
                if j <= jc:
                    s_ = j / jc; return h + (z_o + 0.02 - h) * (3 * s_ ** 2 - 2 * s_ ** 3)
                if j <= n2:
                    s_ = (j - jc) / (n2 - jc); return z_o + 0.02 + (h - z_o - 0.02) * (3 * s_ ** 2 - 2 * s_ ** 3)
                return min(h + 0.6 * (j - n2) * GRID_DT, h + 0.15)
            try:
                pts = np.array([np.r_[dest.at(t_abs0 + tf0 + (j + 1) * GRID_DT + L_xy)[:2], zr(j + 1 + lead) - 0.02] for j in range(n2e)])
                Q2 = _ik_path(self.robot, pts, q_hov, self.j6) * SG
            except ValueError as e:
                self._cycle_init_err.append(f"cold{k} 잡기 IK ({str(e)[:40]})"); continue
            Qc = np.vstack([Q1, Q2]); t_c0 = (len(Qc) - 1) * GRID_DT
            out.append((f"cold{k}", dict(P=CY.fit(Qc, t_c0, q0p), theta=(t_c0, min(chi0 * tf0 + D, tf0 - 0.1), tf0 + jc * GRID_DT + L))))
            break
        return out

    def _cycle_fixed(self, obj, dest: MovingPoint, start_time: float, es: dict, cls: str) -> "PrimitiveResult":
        """잡아 올린 상태 (throw_start) 에서 한 사이클: 던지기 obj + dest 잡기 + (안전) 정지. 끝 = dest 를 잡고 상승 중 (다음 꼬리의 throw_start)."""
        from gp8_control.grasping import sep_pick as SP
        from gp8_control.grasping.workspace_gate import tcp_batch
        from gp8_control.throwing import cycle_nlp as CY
        from gp8_control.throwing import sep_throw as ST
        from gp8_control.throwing.nlp.planner import PLANNER_SIGN as SG
        t_c0w = time.time()
        start = ThrowBoundary(np.asarray(es["joint"], float)[:6], np.asarray(es.get("qd", np.zeros(6)), float)[:6])
        t_j, _end = self.judge_arrival(start, obj.target, dest, start_time)        # 다음 물체가 잡을 수 있는 자리인지 (못 잡으면 NotYetReachable)
        hk = SP._h(); ND = int(np.argmax(np.abs(hk) > 1e-15)); D = ND * GRID_DT; RT = CY._consts()["RT"]
        hist = es.get("hist")
        if hist is not None and len(hist) >= 2:
            Uh = np.vstack([np.asarray(hist, float)[:, :6], start.q[None]])
        else:                                                                 # 이력 없음: 지금 명령 속도로 등속이던 것으로
            kk = np.arange(len(ST.G_H), 0, -1)[:, None] * GRID_DT
            Uh = np.vstack([start.q[None] - start.qd[None] * kk, start.q[None]])
        hd = ST.hist_data(Uh, Uh[-1:], 0.0, 0.1)
        q0p = start.q * SG; v0p = start.qd * SG
        z_d = float(dest.at(start_time)[2]); vb = np.asarray(self.v_belt, float)
        fixed = dict(q0=q0p, v0=v0p, tgt=np.asarray(obj.target, float), j6=float(self.j6) * SG[5], ztop=z_d, zh=SP.Z_H, vbc=np.r_[vb[:2], 0.0])
        obj_at = lambda tk: dest.at(start_time + tk)
        best = None; errs = []; db_note = ""
        inits = []
        if CYCLE_DB:                                                          # DB 칸 키: 실제 시작 xy (이력의 필터 출력), 다음 물체 xy (지금)
            db = _cycle_db(); tg = np.asarray(obj.target, float)
            bk = min(db, key=lambda b: float(np.hypot(b[0] - tg[0], b[1] - tg[1]))) if db else None
            if bk is not None and np.hypot(bk[0] - tg[0], bk[1] - tg[1]) < 0.02:
                m = min(len(hk), len(Uh)); x0 = tcp_batch((hk[:m, None] * Uh[::-1][:m]).sum(0)[None] / hk[:m].sum())[0]
                o_n = dest.at(start_time)
                key = np.array([x0[0], x0[1], o_n[0], o_n[1] - x0[1]]) / _CYC_SC
                Kd, sols = db[bk]; dist = np.abs(Kd - key).max(1)
                if dist.min() > CYCLE_DB_MAXD:                                # DB 범위 밖 — 실시간에 cold 는 못 기다린다
                    raise ValueError(f"사이클 DB 범위 밖 (가장 가까운 칸 {dist.min():.1f} 격자, 시작 {np.round(x0[:2], 3).tolist()}, "
                                     f"다음 {np.round(o_n[:2], 3).tolist()})")
                for j in np.argsort(dist)[:CYCLE_DB_K]:
                    Pj, thj, xj, lj = sols[j]
                    if xj is not None:                                        # 해 전체 (원·쌍대) — 안쪽 30~50 반복
                        inits.append((f"dbx{dist[j]:.1f}", dict(x=xj, lam=lj, theta=thj)))
                    else:
                        Pn = Pj.copy(); Pn[:, 0] = q0p
                        inits.append((f"db{dist[j]:.1f}", dict(P=Pn, theta=thj)))
                db_note = f", DB 칸 거리 {dist.min():.1f}"
        for phase in ("db", "cold"):
            if phase == "cold":
                if best is not None or not CYCLE_COLD:
                    break
                inits = self._cycle_inits(q0p, dest, start_time, t_j)
            for ini in inits:
                try:
                    r = ST.call(CY.run, fixed, hd, obj_at, [ini], logger=None, max_outer=CYCLE_DB_OUTER if "x" in ini[1] else None)
                except ValueError as e:
                    errs.append(str(e)[:80]); continue
                if best is None or r["J"] < best["J"]:
                    best = r
            inits = []
        if best is None:
            raise ValueError("사이클 NLP 실패 — " + "; ".join(errs + getattr(self, "_cycle_init_err", []))[:300])
        t_c, t_s, t_k = (float(v) for v in best["theta"])
        Q = CY.sample(best["P"], t_c) * SG; n = len(Q) - 1; t = np.arange(n + 1) * GRID_DT
        Qd = np.gradient(Q, t, axis=0)
        # 실제 (필터) — 방출 상태·하강 시작 시각
        pad = len(hk); Uf = np.vstack([Uh[:-1], Q, np.repeat(Q[-1:], pad, 0)])
        qr = np.stack([np.convolve(Uf[:, j], hk)[:len(Uf)] for j in range(6)], -1)[len(Uh) - 1:len(Uh) - 1 + n + 1 + ND]
        pr = tcp_batch(qr)
        from gp8_control.throwing.throw_planner import _predict_landing as _pl
        i_s = int(round(t_s / GRID_DT))                                       # 방출 실제 상태 → 착지 (앱 던지기와 같은 식: 그립 오프셋, 무항력)
        p_l, v_l, p_land, land_err = _pl(qr[i_s] * SG, (qr[i_s + 1] - qr[i_s - 1]) / (2 * GRID_DT) * SG, np.asarray(obj.target, float))
        if land_err > CYCLE_LAND_TOL:                                         # 11-05 실시간: θ 고정 (DB t*) 해가 착지 9.2 cm 로 실행돼 bin 밖
            raise ValueError(f"사이클 착지 예측 오차 {land_err * 100:.1f} cm > {CYCLE_LAND_TOL * 100:.0f} cm ({best['tag']})")
        ks = min(n - 2, int(np.ceil((t_s + RT / 2) / GRID_DT)) + 1)                   # 던지기 | 잡기 나누는 곳 = 방출 창 끝
        i_k = int(round(t_k / GRID_DT))
        dn = np.flatnonzero((np.arange(len(pr)) < i_k) & (pr[:, 2] >= SP.Z_H)); i_dn = int(dn[-1]) if len(dn) else ks
        t_on = max(0.0, (i_dn - ks) * GRID_DT)                                       # 석션 ON = 실제 툴이 h 아래로 내려가기 시작할 때
        thr = Segment("throw", t[:ks + 1], Q[:ks + 1], Qd[:ks + 1], suction=[(t_s, False)],
                      info=dict(kind="throw", t_f=float(t[ks]), J=best["J"], land_err=land_err, cls=cls, sep=True, cycle=True, v_out=0.0, v_in=0.0,
                                end_kind="object", dest_id=getattr(dest, "object_id", None), sep_mode="cycle"))
        pk = Segment("pick", t[ks:] - t[ks], Q[ks:], Qd[ks:], suction=[(t_on, True)],
                     info=dict(kind="pick", sep=True, cycle=True, v_in=0.0, v_out=float((pr[-1, 2] - pr[-2, 2]) / GRID_DT), N=int(n - ks),
                               k_touch=int(i_k - ks), z_top=float(SP.Z_H), z_obj=z_d, profile="cycle"))
        nxt = ThrowBoundary(Q[-1], (Q[-1] - Q[-2]) / GRID_DT)
        nxt.p_safe = np.r_[dest.at(start_time + t_c)[:2], SP.Z_H + 0.10]
        br = self.brake_segment(nxt)
        tp = ThrowPlan(t=t[:ks + 1], q=Q[:ks + 1], qd=Qd[:ks + 1], qdd=np.zeros_like(Q[:ks + 1]), t_f=float(t[ks]), t_release=t_s,
                       release_window=(t_s - RT / 2, t_s + RT / 2), i_release=i_s, t_suction_on=None, start=start, end=None,
                       p_target=np.asarray(obj.target, float), p_launch=p_l, v_launch=v_l, p_land=p_land, land_err=land_err,
                       info=dict(model="cycle", t_f=float(t[ks]), t_star=t_s, t_k=t_k, t_c=t_c, J=best["J"], tag=f"cycle-{best['tag']}",
                                 solve_s=time.time() - t_c0w, iters=best["it"]),
                       raw=dict(P=best["P"], theta=best["theta"]))
        if self.log is not None:
            self.log.warn(f"  ℹ 사이클 ({best['tag']}, {time.time() - t_c0w:.1f}s{db_note}): t_c {t_c:.3f} t* {t_s:.3f} t_k {t_k:.3f}, 착지 예측 오차 "
                          f"{land_err * 100:.1f} cm, 접촉 실제 xy {1e3 * np.abs(pr[i_k, :2] - dest.at(start_time + t_k)[:2]).max():.1f} mm, "
                          f"끝 명령 TCP {np.round(tcp_batch(Q[-1:])[0], 3).tolist()} → 정지 목표 {np.round(nxt.p_safe, 3).tolist()}, "
                          f"정지 끝 TCP {np.round(tcp_batch(br.q[-1:])[0], 3).tolist()}")
        self.n_nlp += 1
        return PrimitiveResult(True, t_k, dest.at(start_time + t_k),
                               end_state=dict(joint=Q[-1], qd=nxt.qd, moving=True, throw_start=True, z_obj=z_d,
                                              pick_info=dict(k_touch=pk.info["k_touch"], N=pk.info["N"], v_out=pk.info["v_out"])),
                               payload=dict(segments=[thr, pk, br], plan=tp), effort_cost=float(best["J"]))

    def _jcycle(self, obj, src: MovingPoint, dest, start_time: float, es: dict, cls: str, ready_home=None):
        """hover (물체 i 위, 하강 시작) 에서 jcyc1 사이클: [잡기 i, 던지기 i, (꼬리 없을 때) 보증 명령 + 정지]. 끝 = 물체 i+1 위 hover (sep_hover).
        ready_home (관절) 이 있으면 다음 물체 없음: 끝 = (home xy, h) 위 실제 정지 — dict(segments, plan, duration) 를 돌려준다 (final_throw 용)."""
        JB = _jc_impl()
        from gp8_control.grasping import sep_pick as SP
        from gp8_control.throwing import sep_throw as ST
        from gp8_control.throwing.throw_planner import _predict_landing as _pl
        from gp8_control.throwing.nlp.planner import PLANNER_SIGN as SG
        try:
            from gp8_control.sim import pause_clock as _pc
        except ImportError:
            _pc = None
        t0w = time.time()
        hk = SP._h(); ND = int(np.argmax(np.abs(hk) > 1e-15)); D = ND * GRID_DT; RT = 0.2
        q_s = np.asarray(es["joint"], float)[:6]; qd_s = np.asarray(es.get("qd", np.zeros(6)), float)[:6]
        hist = es.get("hist"); jh = es.get("jc_hist")
        if jh is not None and (hist is None or len(hist) < len(jh)):          # 자리표시 (연장 명령) 가 실어 온 앞 사이클 명령 이력
            hist = jh
        if hist is not None and len(hist) >= 2:
            Uh = np.vstack([np.asarray(hist, float)[:, :6], q_s[None]])
            if np.max(np.abs(Uh[-2] - q_s)) < 1e-12:                           # 이력 끝 = 지금 명령 (중복) 이면 하나 뺀다
                Uh = Uh[:-1]
        else:
            kk = np.arange(len(ST.G_H) + 12, 0, -1)[:, None] * GRID_DT
            Uh = np.vstack([q_s[None] - qd_s[None] * kk, q_s[None]])
        p_i = np.asarray(src.at(start_time + D), float)                        # 실제 r = 0 = 코드 D
        tgt = np.asarray(obj.target, float); oid = getattr(obj, "object_id", None); ready = ready_home is not None
        if ready:
            from gp8_control.throwing.throw_planner import tcp_of
            p_n = np.asarray(tcp_of(np.asarray(ready_home, float)[:6]), float)[:3]   # 정지점 = home TCP 그대로 (다음 head 는 home 에서 출발한다고 본다)
        else:
            p_n = np.asarray(dest.at(start_time + D), float)
            if getattr(dest, "object_id", None) == oid:
                raise ValueError("다음 물체가 든 물체 자신")
            if JC_DB:                                                         # DB 범위 = 도달 범위: (물체 i, 다음 물체, 빈) 칸이 있어야
                if not JB.db_covered(p_i, p_n, tgt):
                    raise ValueError(f"(물체 i, 다음 물체, 빈) 이 사이클 DB 범위 밖 (i {np.round(p_i[:2], 3).tolist()}, n {np.round(p_n[:2], 3).tolist()})")
            elif not self._jc_reach(dest, start_time + JC_TC[0], start_time + JC_TC[1] + JC_EXT + JC_PW):
                raise ValueError(f"다음 물체가 사이클 끝·연장·다음 잡기 창 동안 도달 범위 밖 (지금 {np.round(dest.at(start_time)[:2], 3).tolist()})")
        if not self._jc_reach(src, start_time, start_time + JC_PW):           # 지금 물체: head 가 이미 걸렀어야 한다 (진단)
            raise ValueError(f"잡을 물체가 사이클 잡기 창 동안 도달 범위 밖 (지금 {np.round(src.at(start_time)[:2], 3).tolist()})")
        cert = es.get("jc_cert")
        if cert is None:
            cert = _JC_CERT.pop(oid, None)
        if cert is not None and np.max(np.abs(np.asarray(cert, float)[0] - q_s)) > 1e-6:   # 보증 명령은 지금 명령에서 시작해야 초기값이 된다
            cert = None
        lg = (lambda m: self.log.warn("  " + m[:600])) if self.log is not None else None
        res = (_pc.call("jcycle", JB.solve, Uh, p_i, p_n, tgt, cert, lg, ready) if _pc is not None else JB.solve(Uh, p_i, p_n, tgt, cert, lg, ready))
        if not res.get("ok"):
            raise ValueError(f"jcycle 풀이 실패 ({len(res.get('tried', []))} 시도, {res.get('s', 0):.0f}s)")
        U = np.asarray(res["U"], float); n = len(U) - 1; t = np.arange(n + 1) * GRID_DT; Qd = np.gradient(U, t, axis=0)
        t_c, t_st, t_s, t_k = (float(v) for v in res["theta"])
        ks = int(np.clip(int(round((t_s + D) / GRID_DT)), 2, n - 2)); i_rel = int(round((t_st + D) / GRID_DT))
        pad = len(hk); Uf = np.vstack([Uh[:-1], U, np.repeat(U[-1:], pad, 0)])
        qr = np.stack([np.convolve(Uf[:, j], hk)[:len(Uf)] for j in range(6)], -1)[len(Uh) - 1:]
        p_l, v_l, p_land, land_err = _pl(qr[i_rel] * SG, (qr[i_rel + 1] - qr[i_rel - 1]) / (2 * GRID_DT) * SG, tgt)
        pk = Segment("pick", t[:ks + 1], U[:ks + 1], Qd[:ks + 1], suction=[(D, True)],
                     info=dict(kind="pick", sep=True, cycle=True, jcycle=True, v_in=0.0, v_out=0.0, N=int(ks), k_touch=int(round((t_k + D) / GRID_DT)),
                               z_top=float(SP.Z_H), z_obj=float(p_i[2]), profile="jcycle"))
        thr = Segment("throw", t[ks:] - t[ks], U[ks:], Qd[ks:], suction=[(float(t[i_rel] - t[ks]), False)],
                      info=dict(kind="throw", t_f=float(t[n] - t[ks]), J=res["J"], land_err=land_err, cls=cls, sep=True, cycle=True, v_out=0.0, v_in=0.0,
                                end_kind=("ready" if ready else "object"), dest_id=(None if ready else getattr(dest, "object_id", None)), sep_mode="jcycle",
                                jc_ready=ready))
        tp = ThrowPlan(t=thr.t, q=thr.q, qd=thr.qd, qdd=np.zeros_like(thr.q), t_f=float(thr.duration), t_release=float(t[i_rel] - t[ks]),
                       release_window=(float(t[i_rel] - t[ks]) - RT / 2, float(t[i_rel] - t[ks]) + RT / 2), i_release=int(i_rel - ks), t_suction_on=None,
                       start=ThrowBoundary(U[ks], Qd[ks]), end=None, p_target=tgt, p_launch=p_l, v_launch=v_l, p_land=p_land, land_err=land_err,
                       info=dict(model="jcycle", t_f=float(thr.duration), t_star=t_st, t_k=t_k, t_c=t_c, J=res["J"], tag=f"jcycle-{res['tag']}",
                                 solve_s=time.time() - t0w, iters=res.get("it")),
                       raw=dict(P=res["P"], theta=res["theta"]))
        segs = [pk, thr]; duration = float(t[n])
        Qc = np.asarray(res["cert"], float)
        jc_hist = np.vstack([Uh[:-1], U])[-200:]                               # 사이클 끝까지의 명령 (다음 사이클·게이트의 이력)
        if ready:                                                             # 정지 명령 (연장) + 0.3 s 유지 → 실제가 (home xy, h) 에 선다. brake 가 아니다 (교체 지점 아님)
            n_h = int(round(0.30 / GRID_DT)); Qs = np.vstack([Qc, np.repeat(Qc[-1:], n_h, 0)]); ts = np.arange(len(Qs)) * GRID_DT
            st = Segment("settle", ts, Qs, np.gradient(Qs, ts, axis=0), info=dict(kind="settle", ws_exclude=True, jc_ready=True))
            segs.append(st); duration += st.duration
            if self.log is not None:
                g = res.get("gate", {})
                self.log.warn(f"  ℹ jcycle→준비점 ({res['tag']}, 풀이 {res.get('s', 0):.1f}s, 시도 {len(res.get('tried', []))}): t_c {t_c:.3f} t* {t_st:.3f} "
                              f"J {res['J']:.2f}, 착지 예측 {land_err * 100:.1f} cm, 정지 오차 {1e3 * g.get('ext_contact_p', 0):.2f} mm, 유지 z {1e3 * g.get('hold_z_min', 0):.2f} mm")
            self.n_nlp += 1
            return dict(segments=segs, plan=tp, duration=duration)
        _JC_CERT[getattr(dest, "object_id", None)] = Qc
        if self.throw_only:                                                   # 꼬리가 안 오면: 보증 명령 (다음 물체 위 하강 → 접촉 → 정지 명령 = h 위) 뒤 정지
            seg_b = self.jc_cert_segment(Qc, jc_hist, float(p_n[2]))
            segs.append(seg_b); duration += seg_b.duration
        if self.log is not None:
            g = res.get("gate", {})
            self.log.warn(f"  ℹ jcycle ({res['tag']}, 풀이 {res.get('s', 0):.1f}s, 시도 {len(res.get('tried', []))}, r_free {res.get('r_free')}): t_c {t_c:.3f} "
                          f"t* {t_st:.3f} t_s {t_s:.3f} t_k {t_k:.3f}, J {res['J']:.2f}, 착지 예측 {land_err * 100:.1f} cm, 집기 xy {1e3 * g.get('e_xy_path', 0):.2f} mm, "
                          f"연장 xy {1e3 * g.get('ext_exy', 0):.2f} mm")
        self.n_nlp += 1
        return PrimitiveResult(True, duration, dest.at(start_time + t[n]),
                               end_state=dict(joint=U[-1], qd=Qd[-1], moving=True, sep_hover=True, z_obj=float(p_n[2])),
                               payload=dict(segments=segs, plan=tp), effort_cost=float(res["J"]))

    def jc_cert_segment(self, Qc: np.ndarray, jc_hist, z_obj: float) -> Segment:
        """보증 (연장) 명령 + 정지 = 통합 사이클의 자리표시 brake. 끝 명령은 정지 (NLP 의 EXT_REST) 라 그대로 두면 실제가 다음 물체 위 h 에 와 선다.
        info: jc_hover (교체 지점 = 이 자리), jc_cert_q (다음 사이클 초기값), jc_hist (이력), ws_exclude (하강은 게이트 z 검사 제외)."""
        Qc = np.asarray(Qc, float); tq = np.arange(len(Qc)) * GRID_DT
        Qdc = np.gradient(Qc, tq, axis=0) if len(Qc) >= 2 else np.zeros_like(Qc)
        br = self.brake_segment(ThrowBoundary(Qc[-1], Qdc[-1]))
        return Segment("brake", np.r_[tq, tq[-1] + br.t[1:]], np.vstack([Qc, br.q[1:]]), np.vstack([Qdc, br.qd[1:]]),
                       info=dict(kind="brake", ws_exclude=True, jc_cert=True, jc_hover=True, jc_cert_q=Qc,
                                 jc_hist=None if jc_hist is None else np.asarray(jc_hist, float)[-200:], z_obj=float(z_obj)))

    def _call_held_sep(self, source: SourceState, dest: MovingPoint, start_time: float) -> PrimitiveResult:
        """분리 방식의 잡은 물체 전이. 시작이
          sep_hover  : 물체 위 hover 를 v_in 으로 지나는 상태 → [잡기 LP(v_in, v_out*), 던지기] (v_out 은 던지기 NLP 가 정함)
          throw_start: 이미 잡아 올린 던지기 시작 상태 → [던지기] (v_out 은 그 상태의 상승 속력으로 고정)
          그 밖     : 기존 하강 뒤 접촉 상태 → [기존 잡기, 던지기] (v_out 고정)
        던지기는 다음 물체 위 hover 에서 하강 속력 v_in' 으로 끝난다 (end_state sep_hover). 대기점·준비점 종착은 기존 경로."""
        obj = source.obj
        src = obj.moving(self.v_belt)
        es = source.end_state if isinstance(source.end_state, dict) else {}
        cls = getattr(obj, "class_name", "")
        p0 = src.at(start_time); z_obj = float(p0[2]); r_i = float(np.hypot(p0[0], p0[1]))
        mode = "fixed" if es.get("throw_start") else ("hover" if es.get("sep_hover") else "legacy")
        if JCYC and mode == "hover":                                          # 11-07 jcyc1 사이클 (잡기 i + 던지기 i, hover → 다음 hover)
            if getattr(self, "ahead_only", False):
                self._ahead_dest = getattr(dest, "object_id", None)
                raise AheadDone()
            try:
                return self._jcycle(obj, src, dest, start_time, es, cls)
            except (ValueError, RuntimeError) as e:
                if self.log is not None:
                    self.log.warn(f"  jcycle 실패 ({str(e)[:200]})")
                return PrimitiveResult.infeasible(f"jcycle: {str(e)[:100]}")
        head_segs: List[Segment] = []
        if mode == "fixed":
            start = ThrowBoundary(np.asarray(es["joint"], float)[:6], np.asarray(es.get("qd", np.zeros(6)), float)[:6])
            t_arc0 = start_time; v_fix = max(0.0, self._vz(start.q, start.qd)); r_i = float(np.hypot(*start.tcp[:2]))
        elif mode == "legacy":
            q_arrive = es.get("joint")
            if q_arrive is None:
                q_arrive = self._pick_ik(src.at(start_time) + UP * self.end_above)
                if q_arrive is None:
                    return PrimitiveResult.infeasible("현재 물체 자세 IK 실패")
            pick, start = self.pick_segment(src, start_time, q_arrive, qd_arrive=es.get("qd"))
            head_segs = [pick]; t_arc0 = start_time + pick.duration; v_fix = max(0.0, self._vz(start.q, start.qd))
        else:
            q_h = np.asarray(es["joint"], float)[:6]; qd_h = np.asarray(es.get("qd", np.zeros(6)), float)[:6]
            v_in_i = max(0.0, -self._vz(q_h, qd_h)); v_g = self.sep_vguess(r_i, z_obj)
            tpk = self._sep_table().t_pick(r_i, v_in_i, v_g, z_obj)
            if tpk is None:
                return PrimitiveResult.infeasible(f"분리 잡기 표에 반지름 {r_i:.3f} 없음")
            N_est = int(np.ceil(tpk / GRID_DT)) + 3
            t_arc0 = start_time + N_est * GRID_DT                          # 잡기 먼저에서 도착 불가가 바로 나와도 아래 처리부가 쓴다 (sep14)
        if CYCLE and mode == "fixed":                                         # 통합 사이클 (도착 불가는 아래 종전 정지 종점 처리로)
            if getattr(self, "ahead_only", False):
                self._ahead_dest = getattr(dest, "object_id", None)
                raise AheadDone()
            try:
                return self._cycle_fixed(obj, dest, start_time, es, cls)
            except self.NotYetReachable:
                pass
            except (ValueError, RuntimeError) as e:
                if self.log is not None:
                    self.log.warn(f"  사이클 실패 ({str(e)[:160]})")
                return PrimitiveResult.infeasible(f"사이클: {str(e)[:100]}")
        try:
            pf = None
            if mode == "hover" and SEP_PICKFIRST and not getattr(self, "no_pickfirst", False):
                try:
                    pf = self._sep_pickfirst(src, obj, dest, start_time, q_h, qd_h, v_in_i, z_obj, r_i, es.get("hist"), cls)
                except self.NotYetReachable:                                # 다음 물체 도착 불가는 순서와 무관 — 종전처럼 처리 (sep13: 23번 헛돎)
                    raise
                except ValueError as e:                                     # 잡기 LP·실제 이력 NLP 실패 → 종전 순서 (던지기 먼저)
                    if self.log is not None:
                        self.log.warn(f"잡기 먼저 실패 ({e}) — 던지기 먼저로")
            for _attempt in range(3):
                if pf is not None:
                    pick, start, t_arc0, arc, plan = pf
                    head_segs = [pick]
                    break
                if mode == "hover":
                    t_arc0 = start_time + N_est * GRID_DT
                    p_s = src.at(t_arc0).copy(); p_s[2] = self.z_travel
                    q_s = self._pick_ik(p_s)
                    if q_s is None:
                        raise ValueError("던지기 시작(상승 끝) 자세 IK 실패")
                    vb = self.path_vel_basis(q_s)
                    start = ThrowBoundary(q_s, (vb[0] + vb[1] * v_g) if vb is not None else belt_joint_velocity(q_s, self.v_belt + UP * v_g))
                t_j, end = self.judge_arrival(start, obj.target, dest, t_arc0)
                end.t = t_j
                p_n = dest.at(t_arc0 + t_j); r_n = float(np.hypot(p_n[0], p_n[1]))
                hd_fix = None
                if mode == "fixed" and es.get("pick_info") and es.get("hist") is not None:
                    hd_fix = self._fig4_hist(es, start, v_fix, float(es.get("z_obj", z_obj)))   # FIG4 꼬리: head 의 잡기 명령을 이력으로 (꼬리 잡의 held 위치는 0)
                arc, plan, vo, vi = self.sep_arc_segment(start, obj.target, end, z_obj, r_i, r_n, None if mode == "hover" else v_fix, cls,
                                                         hist=hd_fix, dest=dest, t_arc0=t_arc0)
                if getattr(self, "ahead_only", False):                       # FIG4 look-ahead: 해만 남기고 나간다
                    self._ahead_dest = getattr(dest, "object_id", None)
                    raise AheadDone()
                if mode == "hover":
                    try:
                        pick, start = self.sep_pick_segment(src, start_time, q_h, v_in_i, vo, N=N_est, hist=es.get("hist"), v_out_free=False)
                    except ValueError:
                        N_est += 6
                        continue
                    head_segs = [pick]
                break
            else:
                raise ValueError("분리 잡기 LP 가 던지기 시작 시각에 맞지 않음 (3회)")
            if mode == "hover":
                head_segs = [self.sep_refine(head_segs[0], src, start_time, arc, es.get("hist"))]
            self._note_tf(plan.t_f)
            t_hover = t_arc0 + arc.duration
            arc.info["end_kind"] = "object"; arc.info["dest_id"] = getattr(dest, "object_id", None)
            end_state = dict(joint=arc.q[-1], qd=arc.qd[-1], moving=True, sep_hover=True)
            segs = head_segs + [arc]
            duration = t_hover - start_time
            if self.throw_only:
                br = self.brake_segment(self._rest_bnd(arc))
                br.info["ws_exclude"] = True                               # 하강 속력으로 끝난 던지기 뒤 정지 = 하강 시작 (작업공간 게이트 밖)
                segs.append(br); duration += br.duration
            return PrimitiveResult(True, duration, dest.at(t_hover), end_state=end_state, payload=dict(segments=segs, plan=plan),
                                   effort_cost=float(plan.info.get("J", 0.0)))
        except self.NotYetReachable as e:
            if e.retry_after is None:                                       # 앞으로도 못 잡음 — 잡기 LP 를 풀 필요 없다 (후보가 많으면 수십 초였다)
                e.duration_est = (t_arc0 - start_time) + self.t_ref
                return PrimitiveResult.infeasible(str(e), retry_after=e.retry_after, duration_est=e.duration_est, passed=e.passed)
            if mode == "hover":                                             # 대기점 종착: 대비 속력으로 잡고 기존 던지기
                pick, start = self.sep_pick_segment(src, start_time, q_h, v_in_i, v_g, hist=es.get("hist"), v_out_free=False)
                head_segs = [pick]; t_arc0 = start_time + pick.duration
            arc = None
            if SEP_STOP_END:
                v_o = v_g if mode == "hover" else v_fix
                hd = None
                if mode == "fixed" and es.get("pick_info") and es.get("hist") is not None:
                    hd = self._fig4_hist(es, start, v_fix, float(es.get("z_obj", z_obj)))
                err2 = None
                for h in ([hd, None] if hd is not None else [None]):
                    try:
                        arc, plan, hov, p_in, t_in, t_desc = self.sep_stop_arc(start, obj.target, dest, t_arc0, float(e.retry_after), z_obj,
                                                                               r_i, v_o, cls, hist=h)
                        break
                    except (ValueError, RuntimeError) as e2:
                        err2 = e2
                if arc is None and self.q_ready is not None:
                    # 대기점에 못 서면 준비점에 선다 (분리 NLP 준비점 종점) — 다음 물체는 다음 head 가 준비점에서 접근해 잡는다
                    if self.log is not None:
                        self.log.warn(f"분리 정지 종점 실패 ({err2}) — 준비점 종점 (분리 NLP) 으로")
                    arc, plan = self.sep_ready_arc(start, obj.target, self.q_ready, z_obj, r_i, v_o, cls, hist=hd)
                    t_hover = t_arc0 + arc.duration
                    if mode == "hover":
                        head_segs = [self.sep_refine(head_segs[0], src, start_time, arc, es.get("hist"))]
                    self._note_tf(plan.t_f)
                    br = self.brake_segment(self._rest_bnd(arc))
                    if self.throw_only:
                        segs = head_segs + [arc, br]
                        return PrimitiveResult(True, t_hover - start_time + br.duration, dest.at(t_hover),
                                               end_state=dict(joint=br.q[-1], qd=np.zeros(6), moving=False),
                                               payload=dict(segments=segs, plan=plan), effort_cost=float(plan.info.get("J", 0.0)))
                    acc_segs, t_h, end_b = self.approach_to_hover(br.q[-1], dest, t_hover + br.duration, lp=True)
                    segs = head_segs + [arc, br] + acc_segs
                    return PrimitiveResult(True, t_h - start_time, dest.at(t_h),
                                           end_state=dict(joint=end_b.q, qd=end_b.qd, sep_hover=True, z_obj=float(dest.at(t_h)[2])),
                                           payload=dict(segments=segs, plan=plan), effort_cost=float(plan.info.get("J", 0.0)))
                if arc is None:
                    raise ValueError(f"분리 정지 종점 불가 ({err2})")
            if arc is None:
                hov, p_in, t_in, t_desc = self.hover_boundary(dest, e.retry_after)
                arc, plan = self.arc_segment(start, obj.target, hov, cls=cls)
                t_hover = t_arc0 + arc.duration
                if t_hover > t_in - t_desc - self.t_acc + 1e-6:
                    t_in = t_hover + self.t_acc + t_desc
                    hov, p_in, t_in, t_desc = self.hover_boundary(dest, t_in)
                    arc, plan = self.arc_segment(start, obj.target, hov, cls=cls)
            t_hover = t_arc0 + arc.duration
            if mode == "hover":
                head_segs = [self.sep_refine(head_segs[0], src, start_time, arc, es.get("hist"))]
            wait = max(0.0, grid_round(t_in - self.t_acc - t_desc - t_hover))
            self._note_tf(plan.t_f)
            if self.throw_only:
                arc.info["end_kind"] = "wait"; arc.info["dest_id"] = getattr(dest, "object_id", None)
                segs = head_segs + [arc, self.brake_segment(self._rest_bnd(arc))]
                duration = t_hover - start_time + segs[-1].duration
                return PrimitiveResult(True, duration, p_in, end_state=dict(joint=segs[-1].q[-1], qd=np.zeros(6), moving=False),
                                       payload=dict(segments=segs, plan=plan), effort_cost=float(plan.info.get("J", 0.0)))
            desc = self.descend_segment(dest, t_in, plan.q[-1], from_rest=True, t_desc=t_desc)
            segs = head_segs + [arc] + ([self.rest_segment(plan.q[-1], wait)] if wait > 1e-3 else []) + [desc]
            return PrimitiveResult(True, t_in - start_time, dest.at(t_in), end_state=dict(joint=desc.q[-1], qd=desc.qd[-1]),
                                   payload=dict(segments=segs, plan=plan), effort_cost=float(plan.info.get("J", 0.0)))

    def sep_stop_arc(self, start: ThrowBoundary, p_target, dest: MovingPoint, t_arc0: float, t_reach: float, z_obj: float, r_i: float,
                     v_out: float, cls: str = "", hist: Optional[dict] = None):
        """분리 던지기 정지 종점 판: 다음 물체가 t_reach 에야 도달 범위에 들어올 때, 던지기를 그 물체의 대기점 hover 에 정지로 끝낸다.
        접촉 시각 t_in 은 먼저 정한다 — 도달 시각과 (아크 시작 + 최근 평균 던지기 시간 + 가속·하강) 중 늦은 쪽 → 대기점·하강 시간이 정해지고,
        던지기 시간 상한 = 대기점에서 하강을 시작해야 하는 시각 − 아크 시작 (NLP 제약) 이라 한 번 풀면 대기 시각과 맞는다.
        리턴 (Segment, ThrowPlan, hov, p_in, t_in, t_desc). 못 풀면 ValueError (호출부가 예전 던지기 NLP 로)."""
        from gp8_control.throwing import sep_throw as ST
        from gp8_control.throwing.nlp import throw_nlp as _T
        from gp8_control.throwing.throw_planner import tcp_of
        t_desc0 = self.t_desc_of(dest.at(t_reach))
        t_in0 = max(float(t_reach), float(t_arc0) + max(self.t_ref, _T.T_BOUNDS[0]) + self.t_acc + t_desc0)
        hov, p_in, t_in, t_desc = self.hover_boundary(dest, t_in0)
        tf_hi = t_in - t_desc - self.t_acc - float(t_arc0) - GRID_DT               # 2 ms 격자 올림 몫
        if tf_hi < _T.T_BOUNDS[0]:
            raise ValueError(f"대기점 하강 시작까지 {tf_hi:.3f}s — 던지기 시간 하한보다 짧음")
        q_w = np.asarray(hov.q, float)[:6]
        end = ThrowBoundary(q_w, np.zeros(6), 0.0, None, p_tcp=np.asarray(tcp_of(q_w), float), v_tcp_exact=np.zeros(3), j6=float(q_w[5]))
        key = ("stop",) + tuple(np.round(np.asarray(p_target, float), 2))
        prev = self._sep_raw.get(key)
        warm = prev["raw"] if (prev is not None and np.linalg.norm(prev["p_start"] - start.tcp) < WARM_PREV_M) else None
        r_w = float(np.hypot(*end.p_tcp[:2]))
        tp, vo, vi, raw = ST.call(ST.plan, start.q, p_target, end, self.v_belt, z_obj, r_i, r_w, self._sep_table(),
                                  v_out_fixed=float(v_out), start_qd=start.qd, warm=warm, logger=self.log, hist=hist,
                                  tf_win=(_T.T_BOUNDS[0], tf_hi), stop_end=True)
        if t_arc0 + grid_ceil(tp.t_f) > t_in - t_desc - self.t_acc + 1e-6:
            raise ValueError(f"정지 종점 던지기 t_f {tp.t_f:.3f}s 가 대기 시각 {tf_hi:.3f}s 를 넘음")
        self._sep_raw[key] = dict(raw=raw, p_start=np.asarray(start.tcp, float))
        self.n_nlp += 1
        seg = Segment("throw", tp.t, tp.q, tp.qd, suction=[(tp.t_release, False)],
                      info=dict(kind="throw", t_f=tp.t_f, J=tp.info.get("J"), land_err=tp.land_err, cls=cls, sep=True, v_out=vo, v_in=0.0, p_rest=np.asarray(end.p_tcp, float),
                                sep_mode=tp.info.get("mode"), lift=tp.info.get("lift"), stop_end=True))
        return seg, tp, hov, p_in, t_in, t_desc

    def _note_tf(self, t_f: float) -> None:
        """던지기 시간 기록 → t_ref(다음 물체가 종착 시각에 구역 안인지 판단하는 기준) = 최근 평균 (2026-09-22 사용자)."""
        self._tf_hist.append(float(t_f))
        self.t_ref = float(np.mean(self._tf_hist[-30:]))

    def _rest_bnd(self, arc: Segment) -> ThrowBoundary:
        """던지기 끝 명령 상태 → 정지 대체의 시작. RT 정지 종점 던지기 (실제가 p_rest 에 서도록 푼 것) 면 정지 목표 = 그 실제 정지 점
        (명령은 실제와 다른 곳에 있다 — 그 자리에 세우면 실제가 따라가 내려간다, 11-04 RT 시뮬 z 0.071 m)."""
        b = ThrowBoundary(arc.q[-1], np.asarray(arc.qd[-1], float))
        p_rest = (arc.info or {}).get("p_rest")
        from gp8_control.grasping import sep_pick as SP
        if SP.RT and p_rest is not None:
            b.p_safe = np.asarray(p_rest, float)
        return b

    def brake_segment(self, start: ThrowBoundary) -> Segment:
        """상승 끝(던지기 시작 상태)에서 정지 — 던지기 꼬리(NLP)가 제때 안 왔을 때의 안전 대체 (2026-09-22). trajectory 프리미티브로
        (q, q̇) → (q + q̇·0.03, 0). 꼬리가 제때 오면 이 세그먼트는 통째로 던지기로 교체된다."""
        qd0 = np.asarray(start.qd, float)[:6]
        if np.max(np.abs(qd0)) < 1e-6:                                # 이미 정지 → 1 샘플 (trajectory() 는 0/0 → NaN)
            q0 = np.asarray(start.q, float)[:6]
            return Segment("brake", np.array([0.0]), q0[None, :].copy(), np.zeros((1, 6)), info=dict(kind="brake"))
        q1 = np.asarray(start.q, float)[:6] + qd0 * 0.03
        from gp8_control.grasping import sep_pick as SP
        if SP.RT:                                                       # 11-04 RT: 잡기 끝 명령은 내려오는 중일 수 있다 (실제만 h 를 v_out 으로 지남) —
            p_up = np.asarray(start.tcp, float).copy()                  # 그 자리에 세우면 실제가 낮은 명령 쪽으로 따라 내려간다 → h + 0.10 m 이상으로 올려 세운다
            p_up[2] = max(float(p_up[2]), SP.Z_H + 0.10)
            if getattr(start, "p_safe", None) is not None:              # 잡기 끝 명령 xy 는 실제에 거의 안 보여 물체에서 멀 수 있다 → 잡은 물체 위로
                p_up = np.asarray(start.p_safe, float)
            q_up = self._pick_ik(p_up, q_init=np.asarray(start.q, float)[:6])
            if q_up is not None:
                q1 = q_up
        seg = _traj_segment(start.q, start.qd, q1, np.zeros(6), self.M1, self.M2, self.hz, "brake", "brake")
        if not np.all(np.isfinite(seg.q)):
            raise ValueError("brake 세그먼트에 NaN")
        return seg

    def arc_segment(self, start: ThrowBoundary, p_target, end: Optional[ThrowBoundary], cls: str = "") -> Tuple[Segment, ThrowPlan]:
        saved = _nlp_planner.NLP_MAX_RANDOM
        _nlp_planner.NLP_MAX_RANDOM = self.n_random
        # 직전에 푼 같은 종류(정지 종점 / 움직이는 종점)의 해로 먼저 polish (2026-09-22): 준비점 복귀 던지기는 매번 거의 같은 문제라
        # DB 최근접(10 cm 격자)보다 직전 해가 더 가깝다 → 1.5~3.5 s → 0.5~1 s
        kind = "chain" if (end is not None and end.t is not None) else "rest"
        key = (kind, tuple(np.round(np.asarray(p_target, float), 2)))
        p_s = start.tcp
        prev = self._last_raw.get(key)
        # 직전 해는 시작 TCP 가 WARM_PREV_M 안일 때만 (멀면 다른 basin 근처라 polish 가 DB 최근접보다 오래 걸린다)
        warm = prev["raw"] if (prev is not None and np.linalg.norm(prev["p_start"] - p_s) < WARM_PREV_M) else None
        try:
            plan = plan_throw(start, np.asarray(p_target, float), end, warm_init=warm, logger=self.log)
        finally:
            _nlp_planner.NLP_MAX_RANDOM = saved
        if plan.raw is not None:
            self._last_raw[key] = dict(raw=plan.raw, p_start=np.asarray(p_s, float))
        self.n_nlp += 1
        # cls: 물체 클래스. 실행기가 클래스별 릴리즈 선행을 고르는 데 쓴다 (sequence_app._abs_events).
        seg = Segment("throw", plan.t, plan.q, plan.qd, suction=[(plan.t_release, False)],
                      info=dict(kind="throw", t_f=plan.t_f, J=plan.info.get("J"), land_err=plan.land_err, cls=cls))
        return seg, plan

    def approach_to_hover(self, q0, dest: MovingPoint, t_abs0: float, lp: bool = False):
        """빈손 정지 자세 q0 → 대기점(wait_point: 가능 구간에서 물체와 가장 가까운 점) 위 정지 hover (trajectory 프리미티브)
        → (대기) → 가속+추종 하강 → 접촉. NLP 없음. 리턴 (segments, t_in, end_boundary).
        lp=True (분리 방식, SEP_REST_LP): 하강 없이 가속까지 — 끝 = 물체 위 hover 를 벨트 속도로 추종 (그 시각 t_in − t_desc 를 t_in 으로
        리턴), 잡기는 호출측이 잡기 LP 로."""
        q0 = np.asarray(q0, float)[:6]
        hov, p_in, t_in, seg, t_desc = self.wait_point(q0, dest, t_abs0)
        wait = max(0.0, grid_round(t_in - self.t_acc - t_desc - (t_abs0 + seg.duration)))
        if lp:
            t_h = t_in - t_desc
            if SEP_DIRECT_HOVER and wait <= DIRECT_WAIT_MAX_S:
                d = self.direct_hover(q0, dest, t_abs0, t_h)
                if d is not None:
                    return d
            acc = self.accel_segment(dest, t_h, hov.q)
            segs = [seg] + ([self.rest_segment(hov.q, wait)] if wait > 1e-3 else []) + [acc]
            return segs, t_h, ThrowBoundary(acc.q[-1], acc.qd[-1])
        desc = self.descend_segment(dest, t_in, hov.q, from_rest=True, t_desc=t_desc)
        segs = [seg] + ([self.rest_segment(hov.q, wait)] if wait > 1e-3 else []) + [desc]
        return segs, t_in, ThrowBoundary(desc.q[-1], desc.qd[-1])

    def direct_hover(self, q0, dest: MovingPoint, t_abs0: float, t_h_wait: float):
        """정지 q0 → 물체 위 hover 를 벨트 속도로 추종하는 상태 (도착 시각의 물체 위치) — 사다리꼴 한 번. 도착 시각 T: 그 시각 hover 로 가는
        최소 시간과 맞을 때까지 고정점 반복 (물체가 0.12 m/s 라 2~3 번). 도착 hover·잡은 뒤 던지기 시작이 NLP 한계 밖이면 대기점 경로의
        도착 시각(t_h_wait)으로 한 번 더, 그래도 안 되면 None (호출측이 대기점 경로로). 리턴 (segments, t_h, end_boundary)."""
        from gp8_control.throwing.trajectory_primitive import opt_time
        q0 = np.asarray(q0, float)[:6]; z6 = np.zeros(6)

        def target(T):
            ok, q, _p = self._arrival_ok(dest, t_abs0 + T)
            if q is None:
                return False, None, None
            q = np.asarray(q, float)[:6]
            return ok, q, belt_joint_velocity(q, self.v_belt)

        T = GRID_DT
        for _ in range(8):
            ok, q, qd = target(T)
            if q is None:
                break
            T_n = grid_ceil(opt_time(q0, z6, q, qd, self.M1, self.M2 * TRAP_ACC_SCALE))
            if T_n <= T + 1e-9:
                break
            T = T_n
        cands = [T, grid_ceil(t_h_wait - t_abs0)]
        for T in cands:
            ok, q, qd = target(T)
            if not ok or opt_time(q0, z6, q, qd, self.M1, self.M2 * TRAP_ACC_SCALE) > T + 1e-9:
                continue
            try:
                seg = _traj_segment(q0, z6, q, qd, self.M1, self.M2, self.hz, "approach", "approach", T=T)
            except ValueError:
                continue
            seg.info["direct_hover"] = True
            return [seg], t_abs0 + seg.duration, ThrowBoundary(seg.q[-1], seg.qd[-1])
        return None

    # -- primitive_fn ----------------------------------------------------------
    def __call__(self, source: SourceState, dest: MovingPoint, start_time: float) -> PrimitiveResult:
        try:
            if not source.held:
                if source.joint is None:
                    return PrimitiveResult.infeasible("빈손 출발인데 관절 자세가 없다")
                # 빈손: 가능 구간에서 물체와 가장 가까운 대기점으로 접근(정지) → 필요하면 대기 → 가속·추종 하강 → 접촉
                # (분리 방식: 가속까지 → 물체 위 hover 추종 상태 sep_hover — 다음 전이가 잡기 LP 로 잡는다)
                lp = bool(self.sep and SEP_REST_LP)
                segs, t_in, end = self.approach_to_hover(source.joint, dest, start_time, lp=lp)
                duration, arrive_p = t_in - start_time, dest.at(t_in)
                es = dict(joint=end.q, qd=end.qd)
                if lp:
                    es.update(sep_hover=True, z_obj=float(arrive_p[2]))
                # 석션 ON 은 조립 단계(_on_at_descend)가 하강 시작으로 옮긴다 (잡기 LP 는 그 시작 = hover)
                return PrimitiveResult(True, duration, arrive_p,
                                       end_state=es,
                                       payload=dict(segments=segs, plan=None), effort_cost=0.0)

            obj = source.obj
            if obj is None or obj.target is None:
                return PrimitiveResult.infeasible("잡은 물체의 타겟이 없다")
            if self.sep:
                return self._call_held_sep(source, dest, start_time)
            src = obj.moving(self.v_belt)
            q_arrive = (source.end_state or {}).get("joint") if isinstance(source.end_state, dict) else None
            if q_arrive is None:                                        # 첫 물체를 이미 들고 있는 경우: 그 자리 픽 자세
                q_arrive = self._pick_ik(src.at(start_time) + UP * self.end_above)
                if q_arrive is None:
                    return PrimitiveResult.infeasible("현재 물체 자세 IK 실패")
            es = source.end_state if isinstance(source.end_state, dict) else {}
            if es.get("throw_start"):
                # 이미 잡아 올린 상태(던지기 시작 경계)에서 출발 — 실행 중인 계획의 마지막 던지기를 "다음 물체로 끝나는 던지기" 로
                # 바꿀 때 (2026-09-21 사용자: 다음 물체가 도달 범위 안이면 종착점은 그 물체). 픽 세그먼트 없음.
                start = ThrowBoundary(np.asarray(es["joint"], float)[:6], np.asarray(es.get("qd", np.zeros(6)), float)[:6])
                head_segs: List[Segment] = []
                t_pick = 0.0
            else:
                pick, start = self.pick_segment(src, start_time, q_arrive, qd_arrive=es.get("qd"))
                head_segs = [pick]
                t_pick = pick.duration
            t_arc0 = start_time + t_pick
            try:
                # 다음 물체가 (아크 시계로 예측 던지기 시간 t_f 시점에 — judge_arrival, 2026-10-27) 잡을 수 있는 자리면: 아크가 그 물체 위
                # hover 에서 벨트 추종 상태로 끝나고 (움직이는 종점), 바로 추종 하강 → 접촉
                t_j, end = self.judge_arrival(start, obj.target, dest, t_arc0)
                end.t = t_j
                arc, plan = self.arc_segment(start, obj.target, end, cls=getattr(obj, "class_name", ""))
                # 종점의 1차 선형화 기준 t_ref 가 실제 t_f 와 멀면(|Δ| > RELIN_S) 그 시각의 자세로 다시 선형화해 한 번 더 푼다 —
                # 벨트 0.3 m/s 에서 0.5 s 어긋나면 종점 자세가 16 cm 외삽되어 IK 자세와 J6 20°·U 5.6° 차이가 났다 (2026-09-21).
                for _ in range(2):
                    if plan.info.get("end_exact") or abs(plan.t_f - float(end.t)) <= RELIN_S:
                        break                                            # 정확한 종점식이면 재선형화 불필요
                    end = self.arrival_boundary(dest, t_arc0 + plan.t_f)
                    end.t = plan.t_f
                    arc, plan = self.arc_segment(start, obj.target, end, cls=getattr(obj, "class_name", ""))
                self._note_tf(plan.t_f)                                  # t_ref = 지금까지 던지기 시간 평균
                t_hover = t_arc0 + arc.duration                          # = t_f 를 격자로 올린 시각 (plan_throw 가 정렬)
                t_desc = self.t_desc_of(dest.at(t_hover))                  # topp: 위치마다 다르다 → 접촉 위치로 한 번 더
                t_desc = self.t_desc_of(dest.at(t_hover + t_desc))
                t_in = t_hover + t_desc
                if self.throw_only:                                       # 아크 + brake(안전 정지). 하강·픽은 던지기 끝 직전에 최신 위치로 따로
                    arc.info["end_kind"] = "object"; arc.info["dest_id"] = getattr(dest, "object_id", None)
                    segs = head_segs + [arc, self.brake_segment(self._rest_bnd(arc))]
                    duration = t_hover - start_time + segs[-1].duration
                    return PrimitiveResult(True, duration, dest.at(t_hover), end_state=dict(joint=arc.q[-1], qd=arc.qd[-1], moving=True),
                                           payload=dict(segments=segs, plan=plan), effort_cost=float(plan.info.get("J", 0.0)))
                desc = self.descend_segment(dest, t_in, plan.q[-1], from_rest=False, t_desc=t_desc)
                segs = head_segs + [arc, desc]
            except self.NotYetReachable as e:
                if e.retry_after is None:
                    e.duration_est = t_pick + self.t_ref
                    raise
                # 아직 상류: 처음 잡을 수 있게 되는 자리(p_in, t_in) 위 정지 hover 로 던지고 대기 → 가속·추종 하강
                hov, p_in, t_in, t_desc = self.hover_boundary(dest, e.retry_after)
                arc, plan = self.arc_segment(start, obj.target, hov, cls=getattr(obj, "class_name", ""))     # 고정 종점, vf = 0
                t_hover = t_arc0 + arc.duration
                if t_hover > t_in - t_desc - self.t_acc + 1e-6:          # 던지기가 물체보다 늦다 → 그때 자리로 다시
                    t_in = t_hover + self.t_acc + t_desc
                    hov, p_in, t_in, t_desc = self.hover_boundary(dest, t_in)
                    arc, plan = self.arc_segment(start, obj.target, hov, cls=getattr(obj, "class_name", ""))
                    t_hover = t_arc0 + arc.duration
                wait = max(0.0, grid_round(t_in - self.t_acc - t_desc - t_hover))
                self._note_tf(plan.t_f)
                if self.throw_only:                                       # 아크(대기점에서 정지) + brake. 대기·하강은 따로
                    arc.info["end_kind"] = "wait"; arc.info["dest_id"] = getattr(dest, "object_id", None)
                    # 정지(brake)는 아크의 실제 끝 속도에서 — 평소엔 0 이라 1 샘플이지만, 마감에 멈춘 NLP 값을 속도 한계 안으로 옮기며 끝을
                    # 풀었으면(throw_nlp._project_limits) 0 이 아닐 수 있다 (0 에서 시작하면 이음에서 속도가 튄다)
                    segs = head_segs + [arc, self.brake_segment(self._rest_bnd(arc))]
                    duration = t_hover - start_time + segs[-1].duration
                    return PrimitiveResult(True, duration, p_in, end_state=dict(joint=segs[-1].q[-1], qd=np.zeros(6), moving=False),
                                           payload=dict(segments=segs, plan=plan), effort_cost=float(plan.info.get("J", 0.0)))
                desc = self.descend_segment(dest, t_in, plan.q[-1], from_rest=True, t_desc=t_desc)
                segs = head_segs + [arc] + ([self.rest_segment(plan.q[-1], wait)] if wait > 1e-3 else []) + [desc]
            duration = t_in - start_time
            arrive_p = dest.at(t_in)
            return PrimitiveResult(True, duration, arrive_p,
                                   end_state=dict(joint=desc.q[-1], qd=desc.qd[-1]),
                                   payload=dict(segments=segs, plan=plan),
                                   effort_cost=float(plan.info.get("J", 0.0)))
        except self.NotYetReachable as e:
            return PrimitiveResult.infeasible(str(e), retry_after=e.retry_after, duration_est=e.duration_est, passed=e.passed)
        except (ValueError, RuntimeError, AssertionError) as e:
            return PrimitiveResult.infeasible(f"{type(e).__name__}: {str(e).splitlines()[0][:100]}")

    _READY_DB = None

    @classmethod
    def _ready_entry(cls, p_start, p_target) -> Optional[dict]:
        """준비점 종점 DB (GP8_SEP_READY_DB, 기본 throwing/nlp/warm_db_sep_ready_b012.pkl — 시작 TCP 격자 × bin 의 준비점 정지 종점 해, 11-03)
        에서 같은 타겟·시작 TCP xy 가 가장 가까운 해 (warm 형식). 없으면 None."""
        if cls._READY_DB is None:
            import pickle
            path = os.environ.get("GP8_SEP_READY_DB") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "throwing", "nlp",
                                                                       "warm_db_sep_ready_b012.pkl")
            es = []
            if os.path.exists(path):
                with open(path, "rb") as f:
                    es = pickle.load(f)
            cls._READY_DB = (es, np.array([e["target"] for e in es], float).reshape(-1, 3), np.array([e["p_start"] for e in es], float).reshape(-1, 3))
        es, tg, ps = cls._READY_DB
        if not es:
            return None
        sel = np.flatnonzero(np.linalg.norm(tg - np.asarray(p_target, float)[None, :3], axis=1) < 0.02)
        if len(sel) == 0:
            return None
        k = int(sel[np.argmin(np.linalg.norm(ps[sel, :2] - np.asarray(p_start, float)[None, :2], axis=1))])
        return es[k]

    def sep_ready_arc(self, start: ThrowBoundary, p_target, q_home, z_obj: float, r_i: float, v_out: float, cls: str = "",
                      hist: Optional[dict] = None):
        """분리 던지기 준비점 종점 판 (11-03): 끝 = 준비 자세 TCP 에 속도 0 (v_in 0). sep_stop_arc 와 같은 정지 종점 NLP (같은 풀이기·분리 DB),
        던지기 시간 범위는 NLP 기본. 실제 이력(hist)으로 못 풀면 이력 없이 한 번 더. 리턴 (Segment, ThrowPlan). 못 풀면 ValueError."""
        from gp8_control.throwing import sep_throw as ST
        from gp8_control.throwing.nlp import throw_nlp as _T
        from gp8_control.throwing.throw_planner import tcp_of
        q_w = np.asarray(q_home, float)[:6]
        end = ThrowBoundary(q_w, np.zeros(6), 0.0, None, p_tcp=np.asarray(tcp_of(q_w), float), v_tcp_exact=np.zeros(3), j6=float(q_w[5]))
        key = ("ready",) + tuple(np.round(np.asarray(p_target, float), 2))
        prev = self._sep_raw.get(key)
        warm = prev["raw"] if (prev is not None and np.linalg.norm(prev["p_start"] - start.tcp) < WARM_PREV_M) else None
        if warm is None:                                                   # 직전 해가 멀면 준비점 DB 의 가장 가까운 시작 (분리 DB 는 움직이는 종점 해뿐)
            ent = self._ready_entry(start.tcp, p_target)
            if ent is not None:
                warm = dict(P=np.asarray(ent["P"], float), t_f=float(ent["t_f"]), t_star=float(ent["t_star"]), v_out=float(ent["v_out"]),
                            v_in=0.0)
        r_w = float(np.hypot(*end.p_tcp[:2]))
        err = None
        for h in ([hist, None] if hist is not None else [None]):
            try:
                tp, vo, vi, raw = ST.call(ST.plan, start.q, p_target, end, self.v_belt, z_obj, r_i, r_w, self._sep_table(),
                                          v_out_fixed=float(v_out), start_qd=start.qd, warm=warm, logger=self.log, hist=h,
                                          tf_win=(_T.T_BOUNDS[0], _T.T_BOUNDS[1]), stop_end=True)
                break
            except (ValueError, RuntimeError) as e:
                err = e
        else:
            raise ValueError(f"분리 준비점 종점 불가 ({err})")
        self._sep_raw[key] = dict(raw=raw, p_start=np.asarray(start.tcp, float))
        self.n_nlp += 1
        seg = Segment("throw", tp.t, tp.q, tp.qd, suction=[(tp.t_release, False)],
                      info=dict(kind="throw", t_f=tp.t_f, J=tp.info.get("J"), land_err=tp.land_err, cls=cls, sep=True, v_out=vo, v_in=0.0, p_rest=np.asarray(end.p_tcp, float),
                                sep_mode=tp.info.get("mode"), lift=tp.info.get("lift"), stop_end=True, end_kind="ready"))
        return seg, tp

    def final_throw(self, source: SourceState, start_time: float, home_joint=None) -> Optional[Dict[str, Any]]:
        """마지막 물체: 다음 물체가 없으니 던지고 **기본 자세(home_joint)로 돌아와 정지**한다.
        home 이 NLP 한계 안이면 아크 종점을 home 으로 고정, 아니면 자유 종점 후 trajectory 로 복귀. 실패면 None."""
        obj = source.obj
        if obj is None or obj.target is None:
            return None
        src = obj.moving(self.v_belt)
        q_arrive = (source.end_state or {}).get("joint") if isinstance(source.end_state, dict) else None
        if q_arrive is None:
            q_arrive = self._pick_ik(src.at(start_time) + UP * self.end_above)
            if q_arrive is None:
                return None
        try:
            es = source.end_state if isinstance(source.end_state, dict) else {}
            if JCYC and self.sep and es.get("sep_hover") and home_joint is not None and (es.get("jc_hist") is not None or es.get("hist") is not None):
                # 11-07 통합 사이클: 잡기 + 던지기 + home 위 정지 (이력 없는 호출 = 예열 → 예전 경로)
                return self._jcycle(obj, src, None, start_time, es, getattr(obj, "class_name", ""), ready_home=home_joint)
            if es.get("throw_start"):
                start = ThrowBoundary(np.asarray(es["joint"], float)[:6], np.asarray(es.get("qd", np.zeros(6)), float)[:6])
                segs = []
            elif self.sep and es.get("sep_hover"):
                q_h = np.asarray(es["joint"], float)[:6]; p0 = src.at(start_time)
                v_in = max(0.0, -self._vz(q_h, es.get("qd", np.zeros(6))))
                pick, start = self.sep_pick_segment(src, start_time, q_h, v_in, self.sep_vguess(float(np.hypot(p0[0], p0[1])), float(p0[2])),
                                                    hist=es.get("hist"))
                segs = [pick]
            else:
                pick, start = self.pick_segment(src, start_time, q_arrive, qd_arrive=es.get("qd"))
                segs = [pick]
            q_home = None if home_joint is None else np.asarray(home_joint, float)[:6]
            if self.sep:                                                 # 분리 방식: 준비점 종점 분리 NLP 만 (예전 던지기 NLP 안 씀)
                if q_home is None:
                    raise ValueError("분리 방식 마지막 던지기: 준비 자세가 없다")
                v_o = max(0.0, self._vz(start.q, start.qd)); r_i = float(np.hypot(*start.tcp[:2]))
                z_obj = float(es.get("z_obj", src.at(start_time)[2]))
                hd = None
                if es.get("throw_start") and es.get("pick_info") and es.get("hist") is not None:
                    hd = self._fig4_hist(es, start, v_o, z_obj)
                arc, plan = self.sep_ready_arc(start, obj.target, q_home, z_obj, r_i, v_o, getattr(obj, "class_name", ""), hist=hd)
                segs.append(arc)
                if self.throw_only:
                    segs.append(self.brake_segment(self._rest_bnd(arc)))
            elif q_home is not None and _nlp_planner.q_start_in_limits(to_planner(q_home)):
                arc, plan = self.arc_segment(start, obj.target, ThrowBoundary(q_home, np.zeros(6)), cls=getattr(obj, "class_name", ""))
                arc.info["end_kind"] = "ready"
                segs.append(arc)
                if self.throw_only:
                    segs.append(self.brake_segment(self._rest_bnd(arc)))   # 실제 끝 속도에서 (위와 같은 이유)
            else:
                arc, plan = self.arc_segment(start, obj.target, None, cls=getattr(obj, "class_name", ""))
                segs.append(arc)
                if q_home is not None:                                   # 자유 종점 → home 복귀 (정지 → 정지)
                    segs.append(_traj_segment(plan.q[-1], np.asarray(plan.qd[-1], float), q_home, np.zeros(6), self.M1, self.M2, self.hz,
                                              "home", "home"))
        except (ValueError, RuntimeError, AssertionError) as e:
            if self.log is not None:
                self.log.warn(f"마지막 던지기 실패: {e}")
            return None
        if segs and segs[0].info.get("sep") and len(segs) >= 2:
            segs[0] = self.sep_refine(segs[0], src, start_time, segs[1], es.get("hist"))
        self._note_tf(plan.t_f)
        return dict(segments=segs, plan=plan, duration=sum(sg.duration for sg in segs))


# ---------------------------------------------------------------------------
# 전체 궤적
# ---------------------------------------------------------------------------
@dataclass
class FullPlan:
    order: Tuple[Hashable, ...]              # 처리 순서 (첫 항 "robot" = 빈손 출발)
    t: np.ndarray                            # (N,) 계획 시각 0 기준 [s]
    q: np.ndarray                            # (N,6) 로봇 관절
    qd: np.ndarray                           # (N,6)
    suction_events: List[Tuple[float, bool]] # (절대 시각, on?)
    segments: List[Tuple[float, Segment]]    # (시작 절대 시각, 세그먼트)
    sequence: PlanResult
    throws: List[ThrowPlan]
    unprocessed: Tuple[Hashable, ...]
    passed: Tuple[Hashable, ...] = ()         # 이미 도달 범위를 지나간 물체 (앞으로도 불가) — 앱이 바로 큐에서 뺀다 (2026-10-27)

    @property
    def duration(self) -> float:
        return float(self.t[-1]) if len(self.t) else 0.0

    def summary(self) -> str:
        s = [f"순서 {' → '.join(str(o) for o in self.order)}  (미처리 {list(self.unprocessed)})",
             f"전체 {self.duration:.2f}s, 세그먼트 {len(self.segments)}, 던지기 {len(self.throws)}, "
             f"NLP 호출 {self.sequence.n_evaluated} edge"]
        for t0, seg in self.segments:
            extra = ""
            if seg.info.get("kind") == "throw":
                extra = f"  t_f {seg.info['t_f']:.3f}s J={seg.info['J']:.2f} 착지오차 {seg.info['land_err'] * 100:.1f}cm"
            elif seg.info.get("kind") == "pick" and seg.info.get("sep"):
                extra = (f"  잡기 LP (하강·접촉·상승 한 명령) v↓{seg.info.get('v_in', 0):.2f} v↑{seg.info.get('v_out', 0):.2f} m/s, "
                         f"실제 접촉 ≈ +{seg.info.get('k_touch', 0) * GRID_DT:.3f}s")
            elif seg.info.get("kind") == "pick":
                extra = (f"  press {seg.info.get('press', 0):.2f}s({seg.info.get('press_depth', 0) * 1e3:.0f}mm)"
                         f" hold {seg.info.get('hold', 0):.2f} lift {seg.info.get('lift', 0):.2f}")
            elif seg.info.get("kind") == "wait":
                extra = "  (hover 정지 — 상류 물체 대기)"
            elif seg.info.get("kind") == "accel":
                extra = "  (정지 hover → 벨트 추종 가속, 끝 = 물체 위 hover → 잡기 LP)"
            elif seg.info.get("kind") == "descend":
                extra = (f"  (벨트 추종 하강: xy = 물체, z {seg.info.get('z_top', 0):.2f} → 윗면, {seg.info.get('h', 0) * 100:.0f} cm"
                         + (f", 앞 {seg.info['accel']:.2f}s 가속" if seg.info.get("accel") else "") + ")")
            elif seg.info.get("kind") == "home":
                extra = "  (기본 자세 복귀)"
            s.append(f"  {t0:6.2f}s  {seg.name:9s} {seg.duration:5.2f}s" + extra)
        s.append("suction: " + ", ".join(f"{t:.2f}s {'ON' if on else 'OFF'}" for t, on in self.suction_events))
        return "\n".join(s)


def default_home_joint(robot=None) -> np.ndarray:
    """기본 자세 = Config.INITIAL_R/T (픽 방향, (0.4, 0, 0.1)), J6 = PICK_WRIST_J6."""
    from gp8_control.config import Config
    if robot is None:
        from gp8_control.robot.gp8 import GP8
        robot = GP8()
    cfg = Config()
    T0 = np.eye(4); T0[:3, :3] = cfg.INITIAL_R; T0[:3, 3] = np.asarray(cfg.INITIAL_T, float).reshape(3)
    q = np.asarray(robot.inverse_kinematics(T0), float)[:6]; q[-1] = float(cfg.PICK_WRIST_J6)
    return q


def _first_then_y(objs: Sequence[ObjectState], first_id=None) -> List[ObjectState]:
    """y 순(먼저 지나갈 물체부터), 단 first_id 가 있으면 그 물체를 맨 앞으로 — 순서 정책(learning.order_policy)이 고른 물체를
    먼저 시도하고, 못 잡으면 나머지를 fifo 로 (2026-09-30)."""
    order = sorted(objs, key=lambda o: float(np.asarray(o.position)[1]))
    if first_id is not None:
        order = [o for o in order if o.object_id == first_id] + [o for o in order if o.object_id != first_id]
    return order


def _arc_end_est(prim: ThrowPrimitive, source: SourceState, t0: float = 0.0) -> float:
    """잡은 물체의 던지기가 끝날(다음 물체 위 hover 에 닿을) 시각 추정 = 프리미티브의 arrival_boundary 기준 시각 (t_arc0 + t_ref)."""
    es = source.end_state if isinstance(source.end_state, dict) else {}
    if es.get("throw_start"):
        return t0 + prim.t_ref
    return t0 + prim.t_pick_of(source.obj.moving(prim.v_belt).at(t0)) + prim.t_ref


def _passed_below(order: Sequence[ObjectState], first_id, prim: ThrowPrimitive, t_abs: float, logger=None, tag: str = "",
                  start: Optional[ThrowBoundary] = None, p_target=None) -> List[Hashable]:
    """순서 정책이 first_id 를 골라 시도하지 않는 **그보다 하류(y 작은)** 물체 중 t_abs 에 이미 도달 범위를 지난 것 (2026-10-27 사용자:
    "접근 불가능한 물체는 바로바로 queue에서 빼"). fifo 는 하류부터 시도하므로 프리미티브가 같은 판정을 낸다 (PrimitiveResult.passed).
    first_id 보다 상류인데 지난 물체(바깥 레인은 도달 구간이 짧다)는 다음 결정에서 하류 쪽이 되어 걸러진다.
    start(던지기 시작 상태)·p_target(잡은 물체의 조준점)을 주면 꼬리 계획과 같은 판정(judge_arrival: 예측 t_f 시각)으로 보고, t_abs 는 아크 시작
    시각이다. 없으면 t_abs 시각의 도착으로 본다 (head)."""
    fo = next((o for o in order if o.object_id == first_id), None)
    if fo is None:
        return []
    y1 = float(np.asarray(fo.position)[1])
    out: List[Hashable] = []
    for o in order:
        if o.object_id == first_id or float(np.asarray(o.position)[1]) >= y1:
            continue
        mp = o.moving(prim.v_belt)
        if start is not None:
            try:
                prim.judge_arrival(start, p_target, mp, t_abs)
                passed = False
            except prim.NotYetReachable as e:
                passed = e.passed
        else:
            passed = prim.passed_at(mp, t_abs)
        if passed:
            out.append(o.object_id)
            if logger is not None:
                logger.info(f"  {tag}: {o.object_id} 도달 범위 지남 (순서 정책이 {first_id} 를 골라 건너뛴 하류 물체)")
    return out


def _fifo_sequence(source: SourceState, objs: Sequence[ObjectState], prim: ThrowPrimitive, logger=None,
                   first_id=None, passed: Optional[List[Hashable]] = None) -> PlanResult:
    """순서 탐색 보류 (2026-09-21 사용자): **y 가 작은(먼저 지나갈) 물체부터** 그대로 처리한다. 앞 던지기가 끝나는 시점에
    hover 가 닿는 범위를 이미 지난 물체는 놓친 것으로 건너뛴다 (프리미티브가 NotYetReachable(앞으로도 불가) 로 알림);
    아직 상류면 프리미티브가 대기점에서 기다린다. NLP 는 채택된 edge 만 푼다 (물체 n개 → n−1 회 + 마지막 던지기).
    passed: 주면 이미 도달 범위를 지난 물체 id 를 담는다 (꼬리에서 순서 정책이 고른 물체보다 하류인 미시도 물체 포함)."""
    from gp8_control.grasping.sequence_planner import PlannedTransition, TransitionResult
    order = _first_then_y(objs, first_id)
    seq_ids: List[Hashable] = [source.obj.object_id if source.held else "robot"]
    transitions: List[PlannedTransition] = []
    t_cursor, cost, n_eval = 0.0, 0.0, 0
    cur = source
    skip: set = set()
    if passed is not None and first_id is not None and source.held and prim.throw_only:
        es = source.end_state if isinstance(source.end_state, dict) else {}
        if es.get("throw_start"):                                     # 꼬리: 던지기 시작 상태에서 같은 판정 (예측 t_f 시각)
            st = ThrowBoundary(np.asarray(es["joint"], float)[:6], np.asarray(es.get("qd", np.zeros(6)), float)[:6])
            skip = set(_passed_below(order, first_id, prim, 0.0, logger, "fifo", start=st, p_target=source.obj.target))
        else:
            skip = set(_passed_below(order, first_id, prim, _arc_end_est(prim, source), logger, "fifo"))
        passed.extend(o.object_id for o in order if o.object_id in skip)
    for o in order:
        n_eval += 1
        if o.object_id in skip:                                          # 순서 정책이 고른 물체가 안 돼 하류로 돌아옴 — 이미 판정함
            if logger is not None:
                logger.info(f"  fifo: {seq_ids[-1]}→{o.object_id} @t={t_cursor:.2f}s 건너뜀 — hover 관절 한계 밖 — 앞으로도 불가 (도달 범위 지남)")
            continue
        mp = o.moving(prim.v_belt); mp.object_id = o.object_id            # 던지기 종점 정보(dest_id)용
        res = prim(cur, mp, t_cursor)
        if not res.feasible:
            if res.passed and passed is not None and o.object_id not in passed:
                passed.append(o.object_id)
            if logger is not None:
                logger.info(f"  fifo: {seq_ids[-1]}→{o.object_id} @t={t_cursor:.2f}s 건너뜀 — {res.reason}")
            continue
        tr = TransitionResult(True, res.duration, 0.0, res.duration, np.asarray(res.arrival_position, float), res, None, 0.0)
        transitions.append(PlannedTransition(seq_ids[-1], o.object_id, t_cursor, tr))
        seq_ids.append(o.object_id)
        t_cursor += res.duration; cost += float(res.effort_cost or 0.0)
        cur = SourceState(True, o, end_state=res.end_state)
        if logger is not None:
            logger.info(f"  fifo: {transitions[-1].source_id}→{o.object_id} OK {res.duration:.2f}s  [누적 {t_cursor:.2f}s]")
        if prim.throw_only:
            # 꼬리(throw_only)는 던지기 하나 [아크, brake] 로 끝난다 — 다음 물체의 하강·픽은 head 가 따로 붙인다.
            # 여기서 두 번째 edge 를 이으면 그 물체의 픽이 하강 없이 hover 자세에서 시작해 한 샘플에 13.6° 점프했다
            # (2026-09-28 16:20 실기 1217→1219→1220: brake → pick, L 18.9° → 32.6°, 명령 제한기가 잘라 팔이 어긋남).
            break
    return PlanResult(tuple(seq_ids), tuple(transitions), len(seq_ids), t_cursor, cost, n_evaluated=n_eval)


def _head_only_plan(source: SourceState, objs: Sequence[ObjectState], prim: ThrowPrimitive, logger=None,
                    start_state: Optional[dict] = None, q_home=None, first_id=None) -> FullPlan:
    """빈손 → y 순으로 처음 잡을 수 있는 물체 하나: 접근·대기·하강(프리미티브) + 픽 + brake. 던지기 없음 (throws=[]).
    start_state(던지기 끝 상태, 2026-09-22): moving=True 면 물체 위 hover 를 벨트 추종 중 → 그 물체(dest_id)가 아직 잡을 수 있으면
    바로 하강, 지나갔으면 brake 로 정지 후 다음 물체로 접근. 정지 상태면 그 자세에서 접근."""
    from gp8_control.grasping.sequence_planner import PlannedTransition, TransitionResult
    order = _first_then_y(objs, first_id)
    pre: List[Segment] = []
    if start_state is not None and start_state.get("moving"):
        q0, qd0 = np.asarray(start_state["joint"], float)[:6], np.asarray(start_state.get("qd", np.zeros(6)), float)[:6]
        dest_id = start_state.get("dest_id")
        o = next((x for x in order if x.object_id == dest_id), None)
        if o is not None:
            dest = o.moving(prim.v_belt); z_obj = float(dest.at(0.0)[2])
            # exact 종점은 t_ref 시각의 물체 위치로 풀린 것 — hold·지연으로 실제 시각이 밀리면 물체는 벨트를 따라 더 가 있다
            # (1 s → 12 cm). hover 와 물체 xy 가 HOVER_DIRECT_M 넘게 어긋나면 바로 하강(첫 샘플부터 xy=물체 → 위치 점프)하지 않고
            # 정지 뒤 접근·하강으로 간다.
            from gp8_control.throwing.throw_planner import tcp_of
            off = float(np.linalg.norm(np.asarray(tcp_of(q0), float)[:2] - np.asarray(dest.at(0.0), float)[:2]))
            from gp8_control.grasping import sep_pick as _SPr
            if JCYC:                                                        # 11-07 통합 사이클: 잡기는 꼬리 (jcycle) 가 — head 는 자리표시만
                cert = start_state.get("jc_cert")
                if cert is not None and np.max(np.abs(np.asarray(cert, float)[0] - q0)) < 1e-6:
                    br = prim.jc_cert_segment(cert, start_state.get("jc_hist"), z_obj); note = "hover (통합 사이클 자리 = 보증 명령)"
                else:
                    br = prim.brake_segment(ThrowBoundary(q0, qd0))
                    br.info.update(jc_hover=True, z_obj=z_obj, jc_hist=start_state.get("jc_hist")); note = "hover (통합 사이클 자리 = 정지)"
                return _assemble_head([br], o, objs, res_dur=0.3, logger=logger, note=note, skipped=())
            if off > HOVER_DIRECT_M and not _SPr.RT:                       # RT: 명령은 물체를 L 만큼 앞서는 게 정상 — 실제 상태는 잡기 앞 정렬이 본다
                if logger is not None:
                    logger.info(f"  head: {dest_id} 위 hover 가 물체와 {off * 100:.1f} cm 어긋남 (> {HOVER_DIRECT_M * 100:.0f} cm) → 정지 후 접근")
            else:
                try:
                    if prim.sep:                                            # 분리 방식: hover 를 v_in 으로 지나며 바로 잡기 LP (대비 상승 속력)
                        p0 = dest.at(0.0)
                        v_in = max(0.0, -prim._vz(q0, qd0))
                        r0 = float(np.hypot(p0[0], p0[1])); v_g = prim.sep_vguess(r0, float(p0[2]))
                        nid = start_state.get("next_id")                    # FIG4: head 가 먼저 정한 다음 물체 → 그 던지기에 맞는 상승 끝 속력
                        nx = next((x for x in objs if x.object_id == nid), None) if nid is not None else None
                        if nx is not None:
                            try:
                                v_g = prim.sep_vout_guess(dest, o.target, nx.moving(prim.v_belt), 0.0, r0, v_in, float(p0[2]))
                            except Exception:                               # noqa: BLE001 — 기본값으로
                                pass
                        _t0 = time.time()
                        # 늦은 head 방지 (10-30): 고른 상승 끝 속력으로 한 번, 표 추정 길이로만 (길이 이분법 없음 — 안 풀리는 조합이면 수 s 를 써서
                        # head 가 마감을 넘겼다, fig4g 6.7 s). 사다리는 사용자 지시로 뺐다. 안 풀리면 아래 except 의 정지 후 접근으로.
                        pick, start = prim.sep_pick_segment(dest, 0.0, q0, v_in, float(v_g), bisect=False, hist=start_state.get("hist"))
                        _t1 = time.time()
                        if not CYCLE and not _nlp_planner.q_start_in_limits(to_planner(start.q)):   # 기존 head 와 같은 검사 (10-30 sep12: U 45.1° 로 든 채 28번 실패)
                            raise ValueError(f"잡은 뒤 던지기 시작 자세가 NLP 관절 한계 밖 ({np.round(np.degrees(start.q), 1).tolist()}°)")
                        br = prim.brake_segment(start)
                        _t2 = time.time()
                        segs = [prim.sep_refine(pick, dest, 0.0, br, start_state.get("hist")), br]
                        _t3 = time.time()
                        return _assemble_head(segs, o, objs, res_dur=float(pick.info["k_touch"]) * GRID_DT, logger=logger,
                                              note=(f"hover 에서 바로 잡기 (분리) [잡기LP {(_t1 - _t0) * 1e3:.0f} · 검사·정지 {(_t2 - _t1) * 1e3:.0f}"
                                                    f" · 보정/측정 {(_t3 - _t2) * 1e3:.0f} ms]"), skipped=())
                    t_in = prim.t_desc_of(dest.at(0.0))
                    t_in = prim.t_desc_of(dest.at(t_in))                    # topp: 접촉 위치로 한 번 더
                    desc = prim.descend_segment(dest, t_in, q0, from_rest=False, t_desc=t_in)
                    pick, start = prim.pick_segment(dest, t_in, desc.q[-1], qd_arrive=desc.qd[-1])
                    # 던지기 시작 자세(잡기 끝)가 NLP 관절 상자 밖이면 이 물체는 던질 수 없다 — 잡지 않는다 (2026-10-29 Isaac id 167:
                    # 던지기 종착 hover 는 NLP 변수라 상자 안이지만 잡는 동안 벨트를 따라 2~3 cm 내려가 J2 가 45° 를 넘었고, 꼬리가
                    # 'q_start 한계 밖'으로 22번 연속 실패하며 팔이 물체를 든 채 멈춰 있었다)
                    if not CYCLE and not _nlp_planner.q_start_in_limits(to_planner(start.q)):
                        raise ValueError(f"잡은 뒤 던지기 시작 자세가 NLP 관절 한계 밖 ({np.round(np.degrees(start.q), 1).tolist()}°)")
                    segs = [desc, pick, prim.brake_segment(start)]
                    return _assemble_head(segs, o, objs, res_dur=t_in, logger=logger, note="hover 에서 바로 하강", skipped=())
                except (ValueError, RuntimeError) as e:
                    if logger is not None:
                        logger.warn(f"  head: {dest_id} 위 hover 에서 하강 실패 ({e}) → 정지 후 다음 물체")
        # 물체가 지나갔거나 하강 불가 → 정지(brake) 뒤 정지 출발
        brake = prim.brake_segment(ThrowBoundary(q0, qd0))
        pre = [brake]
        source = SourceState(False, None, joint=np.asarray(brake.q[-1], float))
    elif start_state is not None and np.max(np.abs(np.asarray(start_state.get("qd", np.zeros(6)), float)[:6])) > 1e-6:
        # 정지 종점(대기점·준비점) 던지기인데 끝 속도가 남은 경우 — 마감에 멈춘 NLP 값을 속도 한계 안으로 옮기며 끝을 풀었을 때
        # (throw_nlp._project_limits). 이 후속 계획이 던지기 뒤 정지 구간을 대신하므로 먼저 감속한다 (안 하면 이음에서 속도가 튄다).
        q0 = np.asarray(start_state["joint"], float)[:6]
        brake = prim.brake_segment(ThrowBoundary(q0, np.asarray(start_state["qd"], float)[:6]))
        pre = [brake]
        source = SourceState(False, None, joint=np.asarray(brake.q[-1], float))
    skipped: List[Hashable] = []
    passed: List[Hashable] = []                                         # 이미 도달 범위를 지남 → 앱이 큐에서 뺀다 (2026-10-27)
    t_pre = sum(sg.duration for sg in pre)
    skip = set(_passed_below(order, first_id, prim, t_pre, logger, "head")) if first_id is not None else set()
    passed += [o.object_id for o in order if o.object_id in skip]
    for o in order:
        if o.object_id in skip:
            continue
        res = prim(source, o.moving(prim.v_belt), t_pre)
        if not res.feasible:
            if logger is not None:
                logger.info(f"  head: robot→{o.object_id} 건너뜀 — {res.reason}")
            (passed if res.passed else skipped).append(o.object_id)
            continue
        segs = pre + list(res.payload["segments"])
        t_cursor = sum(sg.duration for sg in segs)
        q_arrive = res.end_state["joint"]
        dest = o.moving(prim.v_belt)
        if JCYC:                                                            # 11-07 통합 사이클: 접근 (hover 추종) + 벨트 추종 0.30 s + 자리표시 정지 (명령 = 다음 사이클 이력)
            from gp8_control.grasping import sep_pick as _SPj
            hk_ = _SPj._h(); L_ = float(np.arange(len(hk_)) @ hk_) * GRID_DT                  # 필터 지연 (등속에서 명령이 앞서야 할 시간)
            n_f = int(round(JC_FOLLOW_S / GRID_DT)); qd_a = np.asarray(res.end_state.get("qd", np.zeros(6)), float)[:6]
            try:
                pts = np.array([np.r_[dest.at(t_cursor + k * GRID_DT + L_ * min(1.0, k / n_f))[:2], _SPj.Z_H] for k in range(1, n_f + 1)])
                Qf = _ik_path(prim.robot, pts, np.asarray(q_arrive, float)[:6], prim.j6)
                Qf = np.vstack([np.asarray(q_arrive, float)[None, :6], Qf]); tf_ = np.arange(n_f + 1) * GRID_DT
                fol = Segment("hover", tf_, Qf, np.gradient(Qf, tf_, axis=0), info=dict(kind="hover", jc_follow=True))
                segs += [fol]; t_cursor += fol.duration; q_arrive = Qf[-1]; qd_a = fol.qd[-1]
            except ValueError as e:
                if logger is not None:
                    logger.warn(f"  head: {o.object_id} hover 추종 구간 IK 실패 ({e}) → 추종 없이")
            br = prim.brake_segment(ThrowBoundary(q_arrive, qd_a))
            hq = np.vstack([sg.q if i == 0 else sg.q[1:] for i, sg in enumerate(segs)]) if segs else np.zeros((0, 6))
            if (start_state or {}).get("hist") is not None:
                hq = np.vstack([np.asarray(start_state["hist"], float)[:, :6], hq])
            br.info.update(jc_hover=True, z_obj=float(dest.at(t_cursor)[2]), jc_hist=hq[-200:] if len(hq) else None)
            segs += [br]
            return _assemble_head(segs, o, objs, res_dur=t_cursor + 0.3, logger=logger, note="hover + 벨트 추종 (통합 사이클 자리)", skipped=skipped, passed=passed)
        if res.end_state.get("sep_hover"):
            # 대기점 → 가속 → hover 에서 잡기 LP (v_in 0, 11-03). 안 풀리면 종전 하강·누름으로 다시 접근 계획
            p0 = dest.at(t_cursor); r0 = float(np.hypot(p0[0], p0[1])); v_g = prim.sep_vguess(r0, float(p0[2]))
            nid = (start_state or {}).get("next_id")
            nx = next((x for x in objs if x.object_id == nid), None) if nid is not None else None
            if nx is not None:
                try:
                    v_g = prim.sep_vout_guess(dest, o.target, nx.moving(prim.v_belt), t_cursor, r0, 0.0, float(p0[2]))
                except Exception:                                       # noqa: BLE001 — 기본값으로
                    pass
            try:
                h_pre = None                                                  # 잡기 앞 명령 = 시작 이력 + 이 계획의 접근·대기 (이음 샘플 제외)
                if segs:
                    h_pre = np.vstack([sg.q if i == 0 else sg.q[1:] for i, sg in enumerate(segs)])[:-1]
                    if (start_state or {}).get("hist") is not None:
                        h_pre = np.vstack([np.asarray(start_state["hist"], float), h_pre])
                pick, start = prim.sep_pick_segment(dest, t_cursor, q_arrive, 0.0, float(v_g), bisect=False, hist=h_pre)
                if not CYCLE and not _nlp_planner.q_start_in_limits(to_planner(start.q)):
                    raise ValueError(f"잡은 뒤 던지기 시작 자세가 NLP 관절 한계 밖 ({np.round(np.degrees(start.q), 1).tolist()}°)")
                brake = prim.brake_segment(start)
                segs += [pick, brake]
                return _assemble_head(segs, o, objs, res_dur=t_cursor + float(pick.info["k_touch"]) * GRID_DT, logger=logger,
                                      note="대기점에서 잡기 LP (분리)", skipped=skipped, passed=passed)
            except (ValueError, RuntimeError) as e:
                from gp8_control.grasping import sep_pick as _SPr
                if _SPr.RT:                                                  # RT: 종전 하강·누름은 실제 시간축과 안 맞는 상태를 남긴다 → 이 물체 건너뜀
                    if logger is not None:
                        logger.warn(f"  head: {o.object_id} 대기점 잡기 실패 ({e}) → 건너뜀")
                    skipped.append(o.object_id)
                    continue
                if logger is not None:
                    logger.warn(f"  head: {o.object_id} 대기점 잡기 LP 실패 ({e}) → 종전 하강·누름")
                segs_l, t_in_l, end_l = prim.approach_to_hover(source.joint, dest, t_pre, lp=False)
                segs = pre + segs_l
                t_cursor = sum(sg.duration for sg in segs)
                q_arrive = end_l.q
                res = PrimitiveResult(True, t_in_l - t_pre, dest.at(t_in_l), end_state=dict(joint=end_l.q, qd=end_l.qd))
        pick, start = prim.pick_segment(dest, t_cursor, q_arrive, qd_arrive=res.end_state.get("qd"))
        brake = prim.brake_segment(start)
        segs += [pick, brake]
        return _assemble_head(segs, o, objs, res_dur=res.duration + t_pre, logger=logger, skipped=skipped, passed=passed)
    # 폴백 (2026-09-23 사용자): 앞 던지기가 이 물체 위 hover 로 끝났는데 그 물체도, 다음 물체들도 못 잡으면 팔이 벨트 위 허공에
    # 그대로 멈춘다. 그러면 준비점(home)으로 돌려놓는다 — 다음 물체는 거기서 새로 접근하면 된다.
    seq = PlanResult(("robot",), (), 1, 0.0, 0.0, n_evaluated=len(order))
    if q_home is not None and start_state is not None:
        q0 = np.asarray(start_state["joint"], float)[:6]
        qd0 = np.asarray(start_state.get("qd", np.zeros(6)), float)[:6]
        segs = list(pre)
        if not segs and np.max(np.abs(qd0)) > 1e-6:                    # 아직 움직이는 중이면 먼저 정지
            segs = [prim.brake_segment(ThrowBoundary(q0, qd0))]
        q_from = np.asarray(segs[-1].q[-1], float) if segs else q0
        qh = np.asarray(q_home, float)[:6]
        if np.max(np.abs(qh - q_from)) > 1e-4:
            segs.append(_traj_segment(q_from, np.zeros(6), qh, np.zeros(6), prim.M1, prim.M2, prim.hz, "home", "home"))
        if segs:
            segments, t0 = [], 0.0
            for sg in segs:
                segments.append((t0, sg)); t0 += sg.duration
            ts, qs, qds = [], [], []
            for t_seg, sg in segments:
                k0 = 1 if ts and len(sg.t) > 1 else 0
                ts.append(sg.t[k0:] + t_seg); qs.append(sg.q[k0:]); qds.append(sg.qd[k0:])
            t = np.concatenate(ts); q = np.concatenate(qs); qd = np.concatenate(qds)
            if len(t) > 1 and not np.all(np.diff(t) > 0.5 * GRID_DT):
                keep = np.concatenate([[True], np.diff(t) > 0.5 * GRID_DT])
                t, q, qd = t[keep], q[keep], qd[keep]
            kk = t / GRID_DT
            if len(t) and np.abs(kk - np.round(kk)).max() < 1e-3:
                t = np.round(kk) * GRID_DT
            if logger is not None:
                logger.info(f"  head: 잡을 수 있는 물체가 없음 ({[x.object_id for x in order]}) → 준비점 복귀 {t0:.2f}s")
            return FullPlan(("robot",), t, q, qd, [], segments, seq, [],
                            tuple(x.object_id for x in objs if x.object_id not in passed), tuple(passed))
    return FullPlan(("robot",), np.zeros(0), np.zeros((0, 6)), np.zeros((0, 6)), [], [], seq, [],
                    tuple(x.object_id for x in objs if x.object_id not in passed), tuple(passed))


def transitions_done(seq: PlanResult) -> bool:
    return len(seq.transitions) > 0


def _on_at_descend(events, segments):
    """흡착 ON 을 **그 픽 직전 하강(descend)이 시작하는 순간**으로 (2026-09-28 사용자: ON 은 미리 켤 필요 없다, 하강 시작할 때
    켜면 된다). 종전의 '접촉 −SUCTION_LEAD_S' 는 travel z 0.12·1.2 m/s 에서 앞 던지기 안(릴리즈 50 ms 뒤)으로 들어가
    놓자마자 다시 빨아들였다. 하강 없이 바로 픽이면 접촉 순간 그대로. OFF(릴리즈)는 건드리지 않는다."""
    starts = [(float(t0), sg.name) for t0, sg in segments]
    out = []
    for t_ev, on in events:
        if on:
            k = next((i for i, (t0, n) in enumerate(starts) if n == "pick" and abs(t0 - t_ev) < 1e-6), None)
            if k is not None and k >= 1 and starts[k - 1][1] == "descend":
                t_ev = starts[k - 1][0]
        out.append((t_ev, on))
    return sorted(out, key=lambda e: e[0])


def _assemble_head(segs: List[Segment], o: ObjectState, objs: Sequence[ObjectState], res_dur: float, logger=None, note: str = "",
                   skipped=None, passed=()) -> FullPlan:
    """skipped: 시도했지만 못 잡은 물체 (FullPlan.unprocessed → 앱이 실패로 센다). None 이면 종전대로 o 밖의 전부.
    후보를 넓혀 넘기면(앱 _submit, 2026-09-30) 시도조차 안 한 상류 물체가 매번 실패로 세여 3번 만에 포기됐다.
    passed: 이미 도달 범위를 지난 물체 (FullPlan.passed → 앱이 바로 큐에서 뺀다)."""
    from gp8_control.grasping.sequence_planner import PlannedTransition, TransitionResult
    segments, t0 = [], 0.0
    for sg in segs:
        segments.append((t0, sg)); t0 += sg.duration
    ts, qs, qds, events = [], [], [], []
    for t_seg, sg in segments:
        k0 = 1 if ts and len(sg.t) > 1 else 0
        ts.append(sg.t[k0:] + t_seg); qs.append(sg.q[k0:]); qds.append(sg.qd[k0:])
        events += [(t_seg + tk, on) for tk, on in sg.suction]
    t = np.concatenate(ts); q = np.concatenate(qs); qd = np.concatenate(qds)
    if len(t) > 1 and not np.all(np.diff(t) > 0.5 * GRID_DT):        # 1 샘플 brake 등 시각 겹침 제거 (np.gradient 0 나눗셈 방지)
        keep = np.concatenate([[True], np.diff(t) > 0.5 * GRID_DT])
        t, q, qd = t[keep], q[keep], qd[keep]
    kk = t / GRID_DT
    if np.abs(kk - np.round(kk)).max() < 1e-3:
        t = np.round(kk) * GRID_DT
    events = _on_at_descend(events, segments)
    tr = TransitionResult(True, res_dur, 0.0, res_dur, None, None, None, 0.0)
    seq = PlanResult(("robot", o.object_id), (PlannedTransition("robot", o.object_id, 0.0, tr),), 2, t0, 0.0, n_evaluated=1)
    if logger is not None:
        logger.info(f"  head: →{o.object_id} 접촉 +{res_dur:.2f}s, 픽 끝 +{t0 - segs[-1].duration:.2f}s (던지기는 꼬리로){' — ' + note if note else ''}")
    unproc = (tuple(x.object_id for x in objs if x.object_id != o.object_id and x.object_id not in passed) if skipped is None
              else tuple(skipped))
    return FullPlan(("robot", o.object_id), t, q, qd, sorted(events, key=lambda e: e[0]), segments, seq, [], unproc, tuple(passed))


def plan_full_sequence(robot_joint, objects: Sequence[ObjectState], belt_speed: float, *,
                       held: Optional[ObjectState] = None, workspace_radius: float = 0.65,
                       primitive: Optional[ThrowPrimitive] = None, search: str = "greedy", horizon: int = 8,
                       home_joint=None, latency_s: float = 0.0, start_state=None, head_only: bool = False,
                       throw_only: bool = False, logger=None, first_id=None) -> FullPlan:
    """현재 로봇 상태 + 인식된 물체들 → 처리 순서 + 전체 관절 궤적 + suction 타이밍.

    robot_joint : 현재 관절 (6,). held 가 None 이면 빈손 정지 상태에서 출발 (첫 세그먼트 = 접근).
    held        : 지금 잡고 있는 물체 (있으면 그 물체부터 던진다; robot_joint 는 무시).
    objects     : 후보 물체들 (계획 시각 0 의 위치). held 가 목록에 있으면 제외한다.
    search      : "fifo"(y 순, 순서 탐색 없음 — 2026-09-21 사용자) | "greedy" | "exhaustive"(전 순열, rollouts 기록)
    home_joint  : 마지막 물체를 던진 뒤 돌아와 정지할 기본 자세 (None = Config.INITIAL_R/T)
    latency_s   : 이 계획을 세우는 데 걸릴 시간 추정 — 모든 물체를 그만큼 미리 흘려 둔다 (실행 시각 = 계획 시각 0).
    start_state : held 와 함께: dict(joint, qd) = 이미 잡아 올린 **던지기 시작 상태**. 픽 없이 첫 던지기부터 계획한다
                  (실행 중인 계획의 마지막 던지기를 새로 들어온 물체로 끝나게 바꾸는 "꼬리 재계획", 2026-09-21).
    head_only   : 빈손 출발일 때 **첫 물체의 접근·대기·하강·픽 + brake(정지)** 만 (NLP 없음, ms) — 바로 실행을 시작하고 던지기는
                  실행 중에 꼬리(start_state)로 풀어 끼운다 (2026-09-22: 계획 시간이 물체를 놓치게 하지 않도록).
    """
    prim = primitive or ThrowPrimitive(belt_speed, logger=logger)
    prim.throw_only = bool(throw_only)
    q_home = default_home_joint(prim.robot) if home_joint is None else np.asarray(home_joint, float)[:6]
    prim.q_ready = q_home                                               # 분리 정지 종점 실패 뒤 준비점 종착 (11-03)
    v = belt_speed * BELT_DIR
    shift = v * float(latency_s)
    objs = [ObjectState(o.object_id, np.asarray(o.position, float) + shift, o.target, o.class_name)
            for o in objects if held is None or o.object_id != held.object_id]
    if held is not None:
        if start_state and start_state.get("sep_hover"):
            # 분리 방식 꼬리: 잡기 LP 시작(hover) 상태에서 — 물체는 그 순간 hover 바로 아래 (명령 기준, 이미 미리 흘린 위치)
            from gp8_control.throwing.throw_planner import tcp_of
            # 앱이 지금 추정(obj_xy, 이미 t_exec 로 흘린 위치)을 주면 그것 — hover 는 앞 꼬리 때의 추정이라 그 뒤 갱신·시간축 밀림만큼 어긋난다
            # (Isaac sep4: 명령이 물체를 ~60 ms 덜 앞서 실제 접촉이 벨트 상류로 ~7 mm). 둘의 차는 잡기 동안 5차 곡선으로 메운다 (sep_pick_segment).
            p_h = np.asarray(tcp_of(np.asarray(start_state["joint"], float)[:6]), float)
            xy = start_state.get("obj_xy")
            if xy is not None:
                xy = np.asarray(xy, float)[:2] + shift[:2]                    # 다른 물체와 같은 물체 선행 (latency_s)
            xy = p_h[:2] if xy is None or (not JCYC and np.linalg.norm(xy - p_h[:2]) > HOVER_DIRECT_M) else xy   # JCYC: 명령 TCP 는 물체와 무관 (필터 앞섬)
            z_o = float(start_state.get("z_obj", 0.02))
            held = ObjectState(held.object_id, np.array([xy[0], xy[1], z_o]), held.target, held.class_name)
            source = SourceState(True, held, end_state=dict(joint=np.asarray(start_state["joint"], float)[:6],
                                                            qd=np.asarray(start_state.get("qd", np.zeros(6)), float)[:6], sep_hover=True,
                                                            hist=start_state.get("hist"), jc_cert=start_state.get("jc_cert"), jc_hist=start_state.get("jc_hist")))
        else:
            held = ObjectState(held.object_id, np.asarray(held.position, float) + shift, held.target, held.class_name)
            es0 = None
            if start_state:
                es0 = dict(joint=np.asarray(start_state["joint"], float)[:6], qd=np.asarray(start_state.get("qd", np.zeros(6)), float)[:6],
                           throw_start=True)
                for k in ("hist", "pick_info", "z_obj"):                      # FIG4 꼬리: 실제 이력 판 입력 (10-30 fig4_dl60: 여기서 빠져 종전 분리 판으로 풀렸다)
                    if start_state.get(k) is not None:
                        es0[k] = start_state[k]
            source = SourceState(True, held, end_state=es0)
    else:
        q0 = np.asarray(robot_joint, float)[:6]
        if start_state is not None and not start_state.get("moving"):
            q0 = np.asarray(start_state["joint"], float)[:6]
        source = SourceState(False, None, joint=q0)

    ws = CircularWorkspace(workspace_radius)
    if head_only and not source.held:
        return _head_only_plan(source, objs, prim, logger, start_state=start_state, q_home=q_home, first_id=first_id)
    passed: List[Hashable] = []                                     # 이미 도달 범위를 지난 물체 (앱이 큐에서 뺀다, 2026-10-27)
    if search == "fifo":
        seq = _fifo_sequence(source, objs, prim, logger, first_id=first_id, passed=passed)
    else:
        seq = plan_sequence(source, objs, prim, v, ws, search=search, horizon=horizon, logger=logger)

    # ---- 세그먼트 이어붙이기 ----
    segments: List[Tuple[float, Segment]] = []
    throws: List[ThrowPlan] = []
    t_cursor = 0.0
    last_source = source
    by_id = {o.object_id: o for o in objs}
    tr_marks: List[Tuple[int, float, int]] = []            # transition 별 (segments 시작 idx, t_cursor, throws 수) — 되돌리기용
    for tr in seq.transitions:
        tr_marks.append((len(segments), t_cursor, len(throws)))
        r = tr.result
        prim_res = r.primitive
        assert prim_res is not None and prim_res.payload is not None
        if r.wait_time > 1e-6 and last_source.held:          # 진입 대기: 잡은 채 벨트 추종 hold
            q_init = (last_source.end_state or {}).get("joint") if isinstance(last_source.end_state, dict) else None
            if q_init is None:
                q_init = prim._pick_ik(last_source.obj.moving(v).at(t_cursor) + UP * prim.end_above)
            fol = prim.follow_segment(last_source.obj.moving(v), t_cursor, r.wait_time, q_init)
            segments.append((t_cursor, fol))
            t_cursor += fol.duration                             # (wait_time 을 격자로 올린 길이)
        for seg in prim_res.payload["segments"]:
            segments.append((t_cursor, seg))
            t_cursor += seg.duration
        if prim_res.payload.get("plan") is not None:
            throws.append(prim_res.payload["plan"])
        last_source = SourceState(True, by_id[tr.destination_id], end_state=prim_res.end_state)

    # ---- 마지막 물체: 던지고 기본 자세로 복귀·정지 ----
    order = list(seq.sequence)
    dropped: List[Hashable] = []
    if prim.throw_only and transitions_done(seq):
        last_source = SourceState(False, None)                          # 던지기 하나로 끝 (2026-09-22: 후속은 따로 붙임)
    while last_source.held:
        fin = prim.final_throw(last_source, t_cursor, home_joint=q_home)
        if fin is not None:
            for seg in fin["segments"]:
                segments.append((t_cursor, seg)); t_cursor += seg.duration
            throws.append(fin["plan"])
            break
        # 던질 수 없으면 그 물체의 transition(접근·하강·픽)을 통째로 빼고 이전 물체를 마지막으로 삼는다 —
        # 안 그러면 팔이 물체 위에서 끝나 다음 계획(home 출발 가정)과 어긋난다 (2026-09-21 시뮬)
        if logger is not None:
            logger.warn(f"마지막 물체 {last_source.obj.object_id} 던지기 계획 실패 — 이 물체를 계획에서 제외")
        dropped.append(last_source.obj.object_id)
        if not tr_marks:
            break
        i0, t0, n_thr = tr_marks.pop()
        del segments[i0:]; del throws[n_thr:]
        t_cursor = t0
        order.pop()
        prev = order[-1] if order else None
        if prev is None or prev == "robot":
            last_source = source if not source.held else SourceState(True, source.obj)
            if not source.held:
                break
        elif held is not None and prev == held.object_id:
            last_source = source                                        # 출발(held) 물체로 되돌아감 — objs 에는 없다 (KeyError 방지)
        else:
            prev_tr = seq.transitions[len(order) - 2] if len(order) >= 2 else None
            end_state = prev_tr.result.primitive.end_state if prev_tr is not None and prev_tr.result.primitive else None
            last_source = SourceState(True, by_id[prev], end_state=end_state)

    # ---- 배열 결합 (첫 샘플 중복 제거, 시각 단조) ----
    ts, qs, qds, events = [], [], [], []
    for t0, seg in segments:
        k0 = 1 if ts and len(seg.t) > 1 else 0
        ts.append(seg.t[k0:] + t0); qs.append(seg.q[k0:]); qds.append(seg.qd[k0:])
        events += [(t0 + tk, on) for tk, on in seg.suction]
    t = np.concatenate(ts) if ts else np.zeros(0)
    q = np.concatenate(qs) if qs else np.zeros((0, 6))
    qd = np.concatenate(qds) if qds else np.zeros((0, 6))
    if len(t) > 1 and not np.all(np.diff(t) > 0):
        keep = np.concatenate([[True], np.diff(t) > 0])
        t, q, qd = t[keep], q[keep], qd[keep]
    # 격자 스냅: 모든 세그먼트 길이가 GRID_DT 배수이므로 t 는 k·GRID_DT 여야 한다 — 부동소수 누적 오차만 지우고,
    # 어긋나면(격자 이탈) 경고만 하고 그대로 둔다 (스트리머가 Hermite 로 리샘플해 여전히 실행은 된다)
    if len(t):
        kk = t / GRID_DT
        off = np.abs(kk - np.round(kk)).max()
        if off < 1e-3 and np.all(np.diff(np.round(kk)) == 1):
            t = np.round(kk) * GRID_DT
        elif logger is not None:
            logger.warn(f"궤적 샘플이 {GRID_DT * 1e3:.0f} ms 격자를 벗어남 (최대 {off * GRID_DT * 1e3:.2f} ms, "
                        f"간격 {np.unique(np.round(np.diff(t) * 1e3, 2))[:6]} ms) — 스트리머가 리샘플한다")
    events.sort(key=lambda e: e[0])
    seq = PlanResult(tuple(order), seq.transitions[:max(0, len(order) - 1)], len(order), t_cursor, seq.total_cost,
                     n_evaluated=seq.n_evaluated, rollouts=seq.rollouts) if dropped else seq
    events = _on_at_descend(events, segments)                 # ON = 하강 시작 (사용자 2026-09-28)
    done = set(seq.sequence)
    if prim.throw_only and search == "fifo":
        # 꼬리(던지기 하나)는 첫 성공에서 멈춘다 — 그 뒤 후보는 시도하지 않았으니 미처리(= 앱이 실패로 셈)에 넣지 않는다.
        # 넣으면 연쇄할 때마다 나머지 후보가 실패로 세여 잡을 수 있는 물체를 3번 만에 포기했다 (2026-09-30 시뮬 nn_dense id 5).
        tried = [o.object_id for o in _first_then_y(objs, first_id)[:seq.n_evaluated]]
        unproc = tuple(i for i in tried if i not in done and i not in passed)
    else:
        unproc = tuple(o.object_id for o in objs if o.object_id not in done and o.object_id not in passed)
    return FullPlan(seq.sequence, t, q, qd, events, segments, seq, throws, unproc, tuple(i for i in passed if i not in done))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("seq"); log.warn = log.warning
    from gp8_control.config import Config
    from gp8_control.robot.gp8 import GP8
    cfg, robot = Config(), GP8()
    belt = 0.12
    bins = [dict(name="binA", x=1.35, y=0.075, z=-0.06), dict(name="binB", x=1.35, y=-0.075, z=-0.06)]  # 조준면 = z+0.05
    recs = [dict(id=1, **{"class": "metal"}, grasp=dict(robot=[0.45, 0.10, 0.02])),
            dict(id=2, **{"class": "transparent"}, grasp=dict(robot=[0.40, 0.45, 0.02])),
            dict(id=3, **{"class": "metal"}, grasp=dict(robot=[0.50, 0.80, 0.02]))]
    objs = objects_from_perception(recs, bins, belt_speed=belt)
    q_home = default_home_joint(robot)
    t0 = time.time()
    full = plan_full_sequence(q_home, objs, belt, latency_s=2.0, logger=log)
    print(f"\n계획 {time.time() - t0:.1f}s\n" + full.summary())
    print(f"샘플 {len(full.t)}, q 범위(deg) {np.round(np.rad2deg(full.q.min(0)), 0)} ~ {np.round(np.rad2deg(full.q.max(0)), 0)}")

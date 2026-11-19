"""컨베이어 물체 **처리 순서** 플래너 (Part 2: 이동 물체 보정, Part 3: 순서 선택) — 프리미티브·로봇 무관.

원본(conveyor_sequence_planner.py, 노트북 이식판)을 2026-09-18 에 gp8_control 의 던지기 프리미티브
(`throwing/throw_planner`) 에 맞춰 손질했다. 원본과 달라진 점과 이유:

  1. 프리미티브가 **움직이는 목적지를 스스로 처리**한다.
     원본은 "정지 시작점·정지 종점" 프리미티브를 밖에서 고정점 반복(최대 50회)으로 돌려 도착 위치를 맞췄다.
     우리 NLP 는 종점을 t_f 의 1차식으로 안에 넣어 한 번에 풀므로(throw_planner '순환 해소'), 프리미티브에
     목적지의 **위치+속도**(`MovingPoint`)와 출발 시각을 넘기고, 프리미티브가 자기 도착 위치·시각을 돌려준다.
     Part 2 는 반복 대신 (a) 작업영역 진입 전이면 그만큼 기다렸다 출발(1회 재계획), (b) 도착 위치 일관성 검사만 한다.
     → NLP 호출이 edge 당 1~2회.
  2. 프리미티브 입력에 **속도**가 있다 (원본은 위치만). 시작 상승 속도·종점 하강+벨트 추종 속도 경계조건에 필요.
  3. 프리미티브가 **실패를 반환**할 수 있다 (`PrimitiveResult.infeasible(reason)`). NLP 도달 불가는 원형 작업영역보다
     좁은 관절 한계에서 생기므로 예외가 아니라 infeasible edge 로 취급한다.
  4. 시작 상태가 `SourceState` 다: 잡은 물체(위치·속도·타겟) 또는 빈손 로봇(관절 자세). 첫 transition 이 "접근"이 되도록.
  5. 탐색: 기본 **greedy**(매 단계 가장 빨리 처리 가능한 다음 물체) — 2026-09-19 사용자: 당분간 greedy 로 채우고,
     나중에 전 순열(`search="exhaustive"`)로 매번 풀어 결과를 저장한 뒤 신경망이 순서를 뽑도록 바꿀 예정.
     전 순열 모드는 모든 rollout 을 `PlanResult.rollouts` 에 남긴다 (학습 데이터). edge 결과는
     (source, dest, 출발 시각 bin) 로 캐시해 재사용 — NLP 가 edge 당 0.5~1 s 라 필수.
  6. 마지막 물체의 던지기(다음 물체 없음, 자유 종점)는 순서 선택 뒤 `throw_sequence.plan_full_sequence` 가 붙인다.

좌표: 전부 로봇 base [m], 시각은 계획 시각 0 기준 [s]. 작업영역은 xy 평면 원(`CircularWorkspace`) — 거친 선별용이고
실제 도달 가능성은 프리미티브(NLP 관절 한계)가 최종 판정한다. numpy 외 의존 없음.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from itertools import permutations
from typing import Any, Callable, Dict, Hashable, Iterable, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 데이터
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CircularWorkspace:
    """xy 평면 원형 도달 영역 (거친 선별)."""
    radius: float
    center: Tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if not math.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError("workspace radius must be a positive finite number")


@dataclass
class MovingPoint:
    """계획 시각 0 의 위치 + 등속 속도 (로봇 base, m / m/s)."""
    position: np.ndarray
    velocity: np.ndarray

    def __post_init__(self):
        self.position = np.asarray(self.position, float).reshape(3)
        self.velocity = np.asarray(self.velocity, float).reshape(3)

    def at(self, t: float) -> np.ndarray:
        return self.position + self.velocity * float(t)


@dataclass(frozen=True)
class ObjectState:
    """계획 시각 0 의 컨베이어 물체."""
    object_id: Hashable
    position: Sequence[float]          # 흡착점 (로봇 base, m)
    target: Any = None                 # 이 물체를 던질 조준점 (프리미티브에 그대로 전달)
    class_name: str = ""

    def moving(self, velocity) -> MovingPoint:
        return MovingPoint(self.position, velocity)


@dataclass
class SourceState:
    """transition 의 출발 상태.

    held=True : 물체 `obj` 를 잡고 있다 (팔은 그 물체의 흡착점에서 벨트를 따라 움직임). 던지고 다음으로 간다.
    held=False: 빈손 — `joint` 자세에서 정지. 다음 물체로 접근만 한다 (던지기 없음).
    """
    held: bool
    obj: Optional[ObjectState] = None
    joint: Optional[np.ndarray] = None            # 빈손일 때 현재 관절 (6,)
    end_state: Any = None                         # 직전 프리미티브가 넘긴 도착 상태 (관절 자세 등, 불투명)


@dataclass
class PrimitiveResult:
    """프리미티브 결과. 움직이는 목적지를 프리미티브가 처리했으므로 도착 위치·시각을 함께 돌려준다."""
    feasible: bool
    duration: float = math.inf                    # 출발 → 목적지 도착 [s] (픽 세그먼트 포함)
    arrival_position: Optional[np.ndarray] = None # 도착 시각의 TCP(=물체) 위치 (검증용)
    end_state: Any = None                         # 다음 transition 의 SourceState.end_state 로 전달
    payload: Any = None                           # 실행에 필요한 것 (세그먼트·ThrowPlan …) — 여기서는 해석 안 함
    effort_cost: float = 0.0
    reason: Optional[str] = None
    retry_after: Optional[float] = None           # 불가 사유가 "아직 상류" 면: 도착이 가능해지는 절대 시각 (Part 2 가 대기 후 재시도)
    passed: bool = False                          # 이미 도달 범위를 지나간 물체 (앞으로도 불가) — 앱이 바로 큐에서 뺀다 (2026-10-27)

    @staticmethod
    def infeasible(reason: str, retry_after: Optional[float] = None,
                   duration_est: Optional[float] = None, passed: bool = False) -> "PrimitiveResult":
        """duration_est: 가능해졋을 때 걸릴 소요 시간 추정 (Part 2 가 대기 시간 = retry_after − duration_est − 출발 로 계산)."""
        return PrimitiveResult(False, duration=(math.inf if duration_est is None else float(duration_est)),
                               reason=reason, retry_after=retry_after, passed=bool(passed))


# primitive_fn(source, dest, start_time) -> PrimitiveResult
#   source     : SourceState (잡은 물체 or 빈손)
#   dest       : MovingPoint (다음 물체, 계획 시각 0 위치 + 벨트 속도) — 프리미티브가 도착 시각 위치를 스스로 맞춘다
#   start_time : 출발 시각 [s, 계획 시각 0 기준]
PrimitiveFunction = Callable[[SourceState, MovingPoint, float], PrimitiveResult]


@dataclass
class TransitionResult:
    feasible: bool
    motion_time: float = math.inf
    wait_time: float = 0.0
    total_time: float = math.inf
    grasp_position: Optional[np.ndarray] = None
    primitive: Optional[PrimitiveResult] = None
    reason: Optional[str] = None
    arrival_error_m: float = math.inf        # 프리미티브 도착 위치 vs 물체 예측 위치


@dataclass
class PlannedTransition:
    source_id: Hashable
    destination_id: Hashable
    start_time: float
    result: TransitionResult


@dataclass
class PlanResult:
    sequence: Tuple[Hashable, ...]           # 처리 순서 (첫 항 = 현재 물체 또는 "robot")
    transitions: Tuple[PlannedTransition, ...]
    processed_count: int
    total_time: float
    total_cost: float
    n_evaluated: int = 0                      # 프리미티브 호출 수 (캐시 미스)
    rollouts: Tuple[Tuple[Tuple[Hashable, ...], float, float], ...] = ()   # exhaustive: (순서, 시간, 비용) 전부 (학습용)


# ---------------------------------------------------------------------------
# Part 2: 이동 물체 보정 (프리미티브가 이동을 처리 → 여기서는 대기·작업영역·일관성만)
# ---------------------------------------------------------------------------
def reachable_time_interval(point: MovingPoint, workspace: CircularWorkspace) -> Optional[Tuple[float, float]]:
    """등속 이동점이 원형 작업영역 안에 있는 시각 구간 [t_in, t_out] (없으면 None)."""
    x = point.position[0] - workspace.center[0]
    y = point.position[1] - workspace.center[1]
    vx, vy = point.velocity[0], point.velocity[1]
    a = vx * vx + vy * vy
    c = x * x + y * y - workspace.radius ** 2
    if a <= 1e-15:
        return (-math.inf, math.inf) if c <= 0.0 else None
    b = 2.0 * (x * vx + y * vy)
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    r = math.sqrt(disc)
    return ((-b - r) / (2.0 * a), (-b + r) / (2.0 * a))


def solve_moving_transition(source: SourceState, dest: MovingPoint, primitive_fn: PrimitiveFunction,
                            workspace: CircularWorkspace, *, start_time: float = 0.0,
                            max_refine: int = 3, arrival_tol_m: float = 0.01,
                            duration_guess: float = 1.5) -> TransitionResult:
    """출발 상태에서 움직이는 다음 물체로의 transition 하나.

    프리미티브가 도착 시각·위치를 스스로 맞추므로 반복은 없다. 대신
      - 도착이 작업영역 **진입 전**이면 그 차이만큼 기다렸다 출발 (wait_time), 재계획
      - 프리미티브가 "아직 상류(관절 한계 밖), t 이후 가능"(`retry_after`) 을 돌려주면 그 시각에 도착하도록 기다렸다 재계획
        (원형 작업영역보다 NLP 관절 한계가 좁아 여기서 대기가 결정되는 경우가 많다)
      - 도착이 작업영역 **진출 후**면 infeasible
      - 프리미티브 도착 위치와 물체 예측 위치의 차이가 arrival_tol_m 를 넘으면 재계획 (최대 max_refine 회)
    """
    interval = reachable_time_interval(dest, workspace)
    if interval is None:
        return TransitionResult(False, reason="object never enters workspace")
    t_in, t_out = interval
    if t_out < start_time:
        return TransitionResult(False, reason="object has left workspace")

    wait = 0.0
    prim: Optional[PrimitiveResult] = None
    err = math.inf
    dur_est = duration_guess
    for _ in range(max(1, max_refine)):
        t0 = start_time + wait
        prim = primitive_fn(source, dest, t0)
        if not prim.feasible:
            ra = prim.retry_after
            if math.isfinite(prim.duration):
                dur_est = prim.duration
            if ra is not None and ra <= t_out + 1e-9 and ra - dur_est > t0 + 1e-6:
                wait = ra - dur_est - start_time          # 그 시각에 도착하도록 출발을 늦춘다
                continue
            return TransitionResult(False, reason=prim.reason or "primitive infeasible", primitive=prim)
        dur_est = prim.duration
        if not math.isfinite(prim.duration) or prim.duration < 0.0:
            raise ValueError("primitive duration must be a non-negative finite number")
        arrival = t0 + prim.duration
        if arrival > t_out + 1e-9:
            return TransitionResult(False, reason="object leaves workspace before interception", primitive=prim)
        if arrival < t_in - 1e-9:                  # 물체가 아직 밖 — 그만큼 더 기다렸다 출발
            wait += t_in - arrival
            continue
        p_pred = dest.at(arrival)
        err = (float(np.linalg.norm(np.asarray(prim.arrival_position, float) - p_pred))
               if prim.arrival_position is not None else 0.0)
        if err <= arrival_tol_m:
            break
    assert prim is not None
    arrival = start_time + wait + prim.duration
    if arrival < t_in - 1e-9 or arrival > t_out + 1e-9:
        return TransitionResult(False, reason="could not place arrival inside workspace window", primitive=prim)
    if err > arrival_tol_m:
        return TransitionResult(False, reason=f"arrival inconsistency {err * 1e3:.1f} mm", primitive=prim,
                                arrival_error_m=err)
    return TransitionResult(True, motion_time=prim.duration, wait_time=wait, total_time=wait + prim.duration,
                            grasp_position=dest.at(arrival), primitive=prim, arrival_error_m=err)


# ---------------------------------------------------------------------------
# Part 3: 순서 선택
# ---------------------------------------------------------------------------
def plan_sequence(source: SourceState, candidates: Iterable[ObjectState], primitive_fn: PrimitiveFunction,
                  conveyor_velocity, workspace: CircularWorkspace, *, search: str = "greedy", horizon: int = 8,
                  effort_weight: float = 0.001, time_bin_s: float = 0.05, logger=None) -> PlanResult:
    """처리량 우선 순서: 처리 가능 개수 최대 → Σ(total_time + effort_weight·effort) 최소.

    search = "greedy"     : 매 단계 total_time 최소인 다음 물체 (기본, 당분간)
             "exhaustive" : 후보 ≤ horizon−1 개의 전 순열 rollout, 전부 PlanResult.rollouts 에 기록 (학습 데이터용)
    edge 결과는 (source_id, dest_id, 출발 시각 bin) 캐시 — 순열마다 같은 edge 를 다시 풀지 않는다.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if search not in ("greedy", "exhaustive"):
        raise ValueError("search must be 'greedy' or 'exhaustive'")
    v = np.asarray(conveyor_velocity, float).reshape(3)
    objects = list(candidates)
    src_id: Hashable = source.obj.object_id if (source.held and source.obj is not None) else "robot"
    ids = [src_id] + [o.object_id for o in objects]
    if len(set(ids)) != len(ids):
        raise ValueError("object_id values must be unique")
    if search == "exhaustive":
        objects = objects[:max(0, horizon - 1)]

    cache: Dict[Tuple[Hashable, Hashable, int], TransitionResult] = {}
    n_eval = [0]

    def edge(src: SourceState, sid: Hashable, dst: ObjectState, t0: float) -> TransitionResult:
        key = (sid, dst.object_id, int(round(t0 / time_bin_s)))
        hit = cache.get(key)
        if hit is None:
            n_eval[0] += 1
            tic = time.time()
            hit = solve_moving_transition(src, dst.moving(v), primitive_fn, workspace, start_time=t0)
            cache[key] = hit
            if logger is not None:
                logger.info(f"  seq: {sid}→{dst.object_id} @t0={t0:.2f}s → "
                            + (f"OK {hit.total_time:.2f}s (wait {hit.wait_time:.2f})" if hit.feasible
                               else f"X {hit.reason}") + f"  [{time.time() - tic:.1f}s]")
        return hit

    def rollout(order: Sequence[ObjectState]):
        src, sid, t, cost, trans, seq = source, src_id, 0.0, 0.0, [], [src_id]
        for dst in order:
            r = edge(src, sid, dst, t)
            if not r.feasible:
                break
            trans.append(PlannedTransition(sid, dst.object_id, t, r))
            t += r.total_time
            eff = r.primitive.effort_cost if r.primitive is not None else 0.0
            cost += r.total_time + effort_weight * float(eff)
            src = SourceState(True, dst, end_state=r.primitive.end_state if r.primitive else None)
            sid = dst.object_id
            seq.append(sid)
        return tuple(seq), tuple(trans), t, cost

    best = ((src_id,), (), 0.0, 0.0)
    rollouts = []

    def better(cand, cur):
        return len(cand[0]) > len(cur[0]) or (len(cand[0]) == len(cur[0]) and cand[3] < cur[3])

    if search == "exhaustive":
        for order in permutations(objects):
            cand = rollout(order)
            rollouts.append((cand[0], cand[2], cand[3]))
            if better(cand, best):
                best = cand
    else:
        remaining, order = list(objects), []
        src, sid, t = source, src_id, 0.0
        while remaining:
            scored = [(edge(src, sid, d, t), d) for d in remaining]
            feas = [(r, d) for r, d in scored if r.feasible]
            if not feas:
                break
            r, d = min(feas, key=lambda x: x[0].total_time)
            order.append(d); remaining.remove(d)
            t += r.total_time
            src = SourceState(True, d, end_state=r.primitive.end_state if r.primitive else None); sid = d.object_id
        best = rollout(order)

    seq, trans, t_total, cost = best
    return PlanResult(seq, trans, len(seq), t_total, cost, n_evaluated=n_eval[0], rollouts=tuple(rollouts))

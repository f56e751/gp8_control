"""처리 순서 신경망 정책 (학습 3단계, 2026-09-30).

rh4(하류 4개의 모든 순서를 평가해 첫 선택만 실행, 결정 0.3~0.7 s)를 흉내 내 **다음에 잡을 물체**를 ms 단위로 고른다.

결정 한 번:
  1) 하류부터 잡을 수 있는 K(=4) 개 후보마다 (2026-10-27: 종전 하류 K 개) 지금 상태 → 그 물체까지의 한 단계 이동을 프리미티브로
     실제 계산 (가능 여부·걸리는 시간·대기)
  2) 후보 특징(위치·순위·남은 여유 시간·그 물체를 먼저 잡은 뒤 다른 물체들의 여유 등) + 팔 상태 → MLP 점수
  3) 가능한 후보 중 점수 최고. 보이는 물체 중 가능한 것이 없으면 None, 모델이 없으면 fifo.

추론은 numpy 만 (throw_time_model 과 같은 이유 — 계획 프로세스에 torch 를 올리지 않는다).
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

K = 4                     # 후보 수 (rh4 와 같게)
M_CTX = 8                 # 여유 시간 특징에 쓰는 하류 물체 수
N_CAND = 19
N_ARM = 5
N_FEAT = N_CAND + N_ARM
STD_FLOOR = 1e-3        # 정규화 표준편차 하한 (아래 OrderPolicyMLP)
DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "models", "order_policy.npz")


def _arm_xy(src) -> np.ndarray:
    if src.held and src.obj is not None:
        return np.asarray(src.obj.position, float)[:2]
    from gp8_control.throwing.throw_planner import tcp_of
    return np.asarray(tcp_of(np.asarray(src.joint, float)[:6]), float)[:2]


def candidate_features(src, vis, prim, v, ws, reach_y_min: float):
    """리턴 (F (K,N_FEAT) float32, mask (K,) bool — 가능한 후보, edges: 후보별 (물체, TransitionResult|None)).
    후보 (2026-10-27 사용자: "접근 가능한 것 중 4개를 해야지 queue가 아니더라도"): 하류부터 잡을 수 있는 K 개 — 가능한 후보가 앞에
    모이고 모자라면 본 불가 물체로 채운다 (mask False). 맥락·y_min·보이는 수는 본 불가 물체를 뺀 것 (plan_timing.candidate_features_timed 와 같다)."""
    from gp8_control.grasping.sequence_planner import solve_moving_transition
    speed = float(np.linalg.norm(v)) or 1e-6
    ok_e, bad = [], []
    for o in vis:
        if len(ok_e) >= K:
            break
        r = solve_moving_transition(src, o.moving(v), prim, ws, start_time=0.0)
        (ok_e.append((o, r)) if r.feasible else bad.append(o))
    edges = ok_e + [(o, None) for o in bad[:K - len(ok_e)]]
    bad_ids = {o.object_id for o in bad}
    pool = [o for o in vis if o.object_id not in bad_ids]
    ctx = list(pool[:M_CTX])
    feas_T = [r.total_time for _o, r in edges if r is not None]
    T_min = min(feas_T) if feas_T else 0.0
    arm = _arm_xy(src)
    from gp8_control.learning.order_search import bin_code, same_bin
    held_bin = bin_code(src.obj.target) if (src.held and src.obj is not None and src.obj.target is not None) else 0.0
    y_min = float(ok_e[0][0].position[1]) if ok_e else 0.0
    arm_f = [1.0 if src.held else 0.0, arm[0], arm[1], held_bin, len(feas_T) / K]
    F = np.zeros((K, N_FEAT), np.float32)
    mask = np.zeros(K, bool)
    for i, (o, r) in enumerate(edges):
        p = np.asarray(o.position, float)
        ok = r is not None
        T1 = float(r.total_time) if ok else 0.0
        W = float(r.wait_time) if ok else 0.0
        slack_c = (p[1] - reach_y_min) / speed
        others = [q for q in ctx if q.object_id != o.object_id]
        sl_after = [((float(q.position[1]) - speed * T1) - reach_y_min) / speed for q in others]
        near = min((float(np.linalg.norm(np.asarray(q.position, float)[:2] - p[:2])) for q in others), default=1.0)
        cbin = bin_code(o.target)
        F[i, :N_CAND] = [1.0 if ok else 0.0, T1, W, p[0], p[1], i / (K - 1), p[1] - y_min, slack_c,
                         p[0] - arm[0], p[1] - arm[1], 1.0 if (src.held and src.obj is not None and same_bin(o.target, src.obj.target)) else 0.0, cbin,
                         min(len(pool), 12) / 10.0, min(sl_after, default=10.0), sum(s < 1.0 for s in sl_after),
                         sum(s < 2.5 for s in sl_after), near, (T1 - T_min) if ok else 0.0, slack_c - T1]
        F[i, N_CAND:] = arm_f
        mask[i] = ok
    return F, mask, edges


class OrderPolicyMLP:
    """후보별 점수 MLP (공유 가중치). 입력·정규화는 numpy, 층: 선형 → tanh … → 선형(1)."""

    def __init__(self, weights, biases, x_mean, x_std):
        self.W = [np.asarray(w, np.float64) for w in weights]
        self.b = [np.asarray(b, np.float64) for b in biases]
        self.x_mean = np.asarray(x_mean, float)
        # 학습 때 상수였던 특징(std ≈ 1e-6)은 런타임 값이 조금만 달라도 수만 σ 로 튀어 점수가 폭주한다 → 하한 (2026-10-15)
        self.x_std = np.where(np.asarray(x_std, float) < STD_FLOOR, 1.0, np.asarray(x_std, float))

    def scores(self, F: np.ndarray) -> np.ndarray:
        h = (np.asarray(F, float) - self.x_mean) / self.x_std          # x_std 는 load/생성 때 하한 (STD_FLOOR)
        for k, (w, b) in enumerate(zip(self.W, self.b)):
            h = h @ w + b
            if k < len(self.W) - 1:
                h = np.tanh(h)
        return h[..., 0]

    def save(self, path: str = DEFAULT_PATH, **meta) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        arrs = {f"W{k}": w for k, w in enumerate(self.W)} | {f"b{k}": b for k, b in enumerate(self.b)}
        np.savez(path, n_layers=len(self.W), x_mean=self.x_mean, x_std=self.x_std, meta=np.asarray(repr(meta)), **arrs)

    @classmethod
    def load(cls, path: str = DEFAULT_PATH) -> "OrderPolicyMLP":
        z = np.load(path, allow_pickle=False)
        n = int(z["n_layers"])
        return cls([z[f"W{k}"] for k in range(n)], [z[f"b{k}"] for k in range(n)], z["x_mean"], z["x_std"])


CYCLE_S = 1.0            # 물체 하나 처리에 걸리는 대략적 시간 [s] (픽 0.32 + 던지기 0.57 + 하강·대기)


def congested(vis, v, reach_y_min: float, cycle_s: float = CYCLE_S, horizon: int = 12) -> bool:
    """fifo 순서로 한 개에 cycle_s 씩 처리한다고 볼 때 도달 구역을 벗어나기 전에 못 잡을 물체가 있는가.
    없으면 fifo 가 이미 최선에 가깝다 (보통 간격 흐름에서 신경망이 fifo 와 다르게 골라 −1.5 % 손해 — 2026-09-30)."""
    speed = float(np.linalg.norm(v)) or 1e-6
    for j, o in enumerate(vis[:horizon]):
        if (float(o.position[1]) - reach_y_min) / speed < (j + 1) * cycle_s:
            return True
    return False


def make_chooser(model: Optional[OrderPolicyMLP], reach_y_min: float, gate: bool = False, cycle_s: float = CYCLE_S,
                 margin: float = 0.5):
    """run_stream(choose_fn=…) 용 정책 함수. model 이 None 이면 fifo.
    gate=True  : 혼잡할 때(congested)만 신경망.
    margin > 0 : 신경망이 fifo 후보(가능한 첫 후보)와 다른 물체를 고를 때, 그 확률이 fifo 후보 확률보다 margin 이상 높을
                 때만 따른다 — 확신 없을 때는 fifo (보통 간격 흐름의 손해를 줄이려고)."""
    from gp8_control.grasping.sequence_planner import solve_moving_transition

    def fifo(src, vis, prim, v, ws):
        for o in vis:
            r = solve_moving_transition(src, o.moving(v), prim, ws, start_time=0.0)
            if r.feasible:
                return o, r
        return None

    def choose(src, vis, prim, v, ws):
        if model is None or not vis or (gate and not congested(vis, v, reach_y_min, cycle_s)):
            return fifo(src, vis, prim, v, ws)
        F, mask, edges = candidate_features(src, vis, prim, v, ws, reach_y_min)
        if not mask.any():                                                # 보이는 물체 중 잡을 수 있는 것이 없다 (vis 를 다 봤음)
            return None
        s = model.scores(F)
        s[~mask] = -np.inf
        i = int(np.argmax(s))
        j = int(np.argmax(mask))                                          # fifo 후보 = 가능한 첫 후보
        if margin > 0 and i != j:
            p = np.exp(s - s[i]); p /= p.sum()
            if p[i] - p[j] < margin:
                i = j
        return edges[i]
    return choose

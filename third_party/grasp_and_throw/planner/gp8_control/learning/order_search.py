"""처리 순서 정답 데이터 (학습 2단계, 2026-09-29).

무작위 벨트 장면을 만들고, 대리 프리미티브(잡기 = 실제 계산, 던지기 = 신경망)로 **모든 순서**를 평가해
(처리 개수 최대 → 총시간 최소) 가장 좋은 순서를 정답으로 남긴다. 같은 장면에서 fifo(y 순)도 평가해 비교한다.

    python -m gp8_control.learning.order_search --n 2000 --out order_data.jsonl [--max-obj 6] [--seed 0]

한 줄 = 장면 하나: 물체 목록(위치·클래스·타겟), fifo/best 순서와 개수·시간, 모든 순열의 (순서, 개수, 시간) 요약.
exhaustive 는 물체 n 개에 n! 순열이라 --max-obj 6(720) 이하로 둔다. edge 결과는 sequence_planner 가 캐시한다.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import time
from contextlib import redirect_stdout
from typing import List, Optional, Sequence

import os

import numpy as np

from gp8_control.grasping.sequence_planner import CircularWorkspace, ObjectState, SourceState, plan_sequence
from gp8_control.grasping.throw_sequence import BELT_DIR, _fifo_sequence, default_home_joint

# bin 배치 (run_sequence_app.sh 와 같은 좌표; 클래스 → 조준점). 2026-10-15 재학습: 순서 정책·던지기 시간 모델이 2 bin (±0.165) 만 보고
# 학습돼 3 bin (+0.33 / 0 / −0.33) 과 iitp 의 x 0.9 를 몰랐다 → 학습 흐름이 아래 배치를 섞는다 (GP8_LEARN_LAYOUTS).
LAYOUTS = {
    "3bin_x135": {"metal": (1.35, 0.33, -0.06), "paper": (1.35, 0.0, -0.06), "transparent": (1.35, -0.33, -0.06)},
    "3bin_x110": {"metal": (1.10, 0.33, -0.06), "paper": (1.10, 0.0, -0.06), "transparent": (1.10, -0.33, -0.06)},
    "3bin_x160": {"metal": (1.60, 0.33, -0.06), "paper": (1.60, 0.0, -0.06), "transparent": (1.60, -0.33, -0.06)},
    "2bin_x135": {"metal": (1.35, 0.165, -0.06), "transparent": (1.35, -0.165, -0.06)},
    "2bin_x090": {"metal": (0.90, 0.165, -0.06), "transparent": (0.90, -0.165, -0.06)},       # iitp 브랜치 GP8_BIN_X=0.9
}
BINS = dict(LAYOUTS["3bin_x135"])        # 기본 배치 (시나리오 읽기 등 배치를 따로 안 주는 곳)
BIN_Y_SCALE = 0.33                       # bin 특징 = 조준 y / 0.33 (3 bin +1/0/−1, 2 bin ±0.5)


def layout_mix():
    """GP8_LEARN_LAYOUTS = "3bin_x135:0.5,2bin_x090:0.3,..." (가중치) → [(이름, 가중치)]. 비우면 3bin_x135 만."""
    raw = os.environ.get("GP8_LEARN_LAYOUTS", "").strip()
    if not raw:
        return [("3bin_x135", 1.0)]
    out = []
    for item in raw.split(","):
        name, _, w = item.partition(":")
        if name.strip() not in LAYOUTS:
            raise ValueError(f"GP8_LEARN_LAYOUTS: 모르는 배치 {name!r} ({sorted(LAYOUTS)})")
        out.append((name.strip(), float(w or 1.0)))
    return out


def pick_layout(rng) -> dict:
    mix = layout_mix()
    w = np.array([m[1] for m in mix], float)
    return LAYOUTS[mix[int(rng.choice(len(mix), p=w / w.sum()))][0]]


def bin_code(target) -> float:
    """조준점 → bin 특징 (y / 0.33, [-1, 1]). 예전 sign(y) 는 종이 bin(y=0)이 '안 듦' 과 같았다."""
    return float(np.clip(float(target[1]) / BIN_Y_SCALE, -1.0, 1.0)) if target is not None else 0.0


def same_bin(a, b) -> bool:
    return a is not None and b is not None and bool(np.allclose(np.asarray(a, float), np.asarray(b, float), atol=1e-3))


OBJ_Z = 0.02
# 2026-10-15 사용자: 레인 가운데 못 잡는 구간이 없는 x 0.32 ~ 잡을 수 있는 끝 0.595 (앱 판정).
# 2026-10-29 사용자 ("0.32 말고 접근 가능한 x범위 전부해"): 앱 레인 판정(lane_wait_ys, 벨트 0.12, z 0.02)으로 대기 가능한 y 가
# 하나라도 있는 x 전부 — 0.18 (y +0.31 한 점) ~ 0.58, 0.60 부터 없음. x < 0.32 는 레인 가운데(y≈0 근처)가 끊겨 상류·하류 두 구간.
X_RANGE = (0.18, 0.595)
Y_FIRST = (0.25, 0.90)           # 첫 물체 y (큐 진입선 ≈ 0.9 에서 도달 구역까지)
GAP = (0.03, 0.35)               # 이웃 물체 y 간격 [m] — 0.14 m 보다 촘촘하면 팔이 못 따라가 순서가 중요해진다
WORKSPACE_R = 0.65


def random_scene(rng: np.random.Generator, n: int, gap=GAP) -> List[ObjectState]:
    y = float(rng.uniform(*Y_FIRST))
    objs = []
    for k in range(n):
        if k:
            y += float(rng.uniform(*gap))
        cls = str(rng.choice(list(BINS)))
        objs.append(ObjectState(k, np.array([float(rng.uniform(*X_RANGE)), y, OBJ_Z]), np.array(BINS[cls]), cls))
    return objs


def fifo_order(source: SourceState, objs: Sequence[ObjectState], prim):
    """런타임 fifo 와 같은 규칙 (y 순, 못 잡는 물체는 건너뜀)."""
    with redirect_stdout(io.StringIO()):
        r = _fifo_sequence(source, objs, prim, logger=None)
    return r


def best_order(source: SourceState, objs: Sequence[ObjectState], prim, belt: float, horizon: int):
    with redirect_stdout(io.StringIO()):
        return plan_sequence(source, objs, prim, belt * BELT_DIR, CircularWorkspace(WORKSPACE_R),
                             search="exhaustive", horizon=horizon, logger=None)


def summarize(r) -> dict:
    """PlanResult → 개수(처리한 물체 수, 'robot' 제외)·시간."""
    seq = [s for s in r.sequence if s != "robot"]
    return dict(order=[int(s) for s in seq], n=len(seq), t=round(float(r.total_time), 4))


def scene_record(objs, fifo, best, dt_s: float, n_pred: int) -> dict:
    return dict(objects=[dict(id=int(o.object_id), p=[round(float(v), 4) for v in o.position], cls=o.class_name,
                              target=[float(v) for v in o.target]) for o in objs],
                fifo=summarize(fifo), best=summarize(best),
                rollouts=[dict(order=[int(s) for s in seq if s != "robot"], t=round(float(t), 4)) for seq, t, _c in best.rollouts],
                dt_s=round(dt_s, 3), n_pred=int(n_pred))


def main() -> None:
    from gp8_control.learning.surrogate import SurrogateThrowPrimitive
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="장면 수")
    ap.add_argument("--min-obj", type=int, default=3)
    ap.add_argument("--max-obj", type=int, default=5)
    ap.add_argument("--belt", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gap", type=float, nargs=2, default=list(GAP), help="이웃 물체 y 간격 범위 [m]")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    prim = SurrogateThrowPrimitive(a.belt)
    q_home = default_home_joint(prim.robot)
    t0 = time.time()
    with open(a.out, "w") as f:
        for k in range(a.n):
            n = int(rng.integers(a.min_obj, a.max_obj + 1))
            objs = random_scene(rng, n, gap=tuple(a.gap))
            src = SourceState(False, None, joint=q_home)
            prim.n_pred = 0
            ts = time.time()
            fifo = fifo_order(src, objs, prim)
            best = best_order(src, objs, prim, a.belt, horizon=n + 1)
            rec = scene_record(objs, fifo, best, time.time() - ts, prim.n_pred)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (k + 1) % 20 == 0 or k == 0:
                print(f"  {k + 1}/{a.n}  ({time.time() - t0:.0f}s)  마지막: n={n} fifo {rec['fifo']['n']}개 {rec['fifo']['t']:.2f}s"
                      f" / best {rec['best']['n']}개 {rec['best']['t']:.2f}s  [{rec['dt_s']:.1f}s, 예측 {rec['n_pred']}회]", flush=True)
    print(f"완료 {a.n} 장면 → {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

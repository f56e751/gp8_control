#!/usr/bin/env python3
"""저장된 warm DB entry 별 스윙 아크의 Cartesian 엔벨로프 보고.

빌드는 GP8_MAX_TCP_Z=99 (사실상 OFF) 로 돌렸으므로, 실제 해가 얼마나 높이
뜨는지는 저장 후에 확인해야 한다. 각 entry 의 아크를 300 샘플로 훑어
z_max / z_min / x_min 을 뽑는다 (빌더 _gates 와 동일한 fk_pos·샘플 수).

사용: .venv/bin/python arc_envelope.py [db.pkl] [--limit 0.85]
"""
import argparse
import pickle
import sys
from pathlib import Path

REPO = Path("/home/robotics/ros2_ws/src/gp8_control")
sys.path.insert(0, str(REPO / "skills"))

ap = argparse.ArgumentParser()
ap.add_argument("db", nargs="?", default=str(REPO / "skills" / "warm_db_128_12pairs.pkl"))
ap.add_argument("--limit", type=float, default=0.85,
                help="이 z 를 넘는 entry 를 표시 (기본 = 스킬 MAX_TCP_Z 기본값)")
a = ap.parse_args()

import numpy as np
from throw_nlp import _spline_eval
from throwing import fk_pos

with open(a.db, "rb") as f:
    db = pickle.load(f)
entries = db["entries"]
p = db["params"]
print(f"{a.db}\n  entries={len(entries)}  rt={p['rt']}  flight={p['flight']}\n")
print(f"{'#':>2}  {'target':>22}  {'start':>20}  {'z_max':>7} {'z_min':>7} {'x_min':>7}  {'J':>6}")

rows = []
for i, e in enumerate(entries):
    q_of, _, _ = _spline_eval(e["P"], e["t_f"])
    T = np.array([fk_pos(q_of(t)) for t in np.linspace(0.0, e["t_f"], 300)])
    z_max, z_min, x_min = T[:, 2].max(), T[:, 2].min(), T[:, 0].min()
    tgt = "(" + ",".join(f"{v:6.3f}" for v in e["target"]) + ")"
    st = "(" + ",".join(f"{v:5.2f}" for v in e["p_start"]) + ")"
    flag = "  <-- limit 초과" if z_max > a.limit else ""
    print(f"{i:>2}  {tgt:>22}  {st:>20}  {z_max:7.3f} {z_min:7.3f} {x_min:7.3f}  "
          f"{e['J']:6.2f}{flag}")
    rows.append(z_max)

if rows:
    over = sum(1 for z in rows if z > a.limit)
    print(f"\n아크 최고 z: max={max(rows):.3f} m, min={min(rows):.3f} m")
    print(f"{a.limit} m 초과 entry: {over}/{len(rows)}")
    if over:
        print(f"  → 이 entry 들은 GP8_MAX_TCP_Z 를 기본 0.85 로 두면 런타임 게이트에")
        print(f"    기각된다. 스크립트가 99 로 export 하므로 현재 설정에서는 통과.")

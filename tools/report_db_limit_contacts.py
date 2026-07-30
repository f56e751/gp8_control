#!/usr/bin/env python3
"""warm DB entry 중 관절 한계 / Cartesian 엔벨로프에 '닿는' 궤적을 진단 리포트.

제외(필터)가 아니라 **보고만** 한다 — 어떤 (start,target) 조합이 한계에 붙어
푸는지 파악용. 한계값은 DB params(q_lo/q_hi, 빌드 당시 formulation)에서 읽으므로
빌드 env 와 자동 일치한다.

  관절 접촉: 제어점 P[j,k] 가 Q_LO[j]/Q_HI[j] 의 JOINT_EPS(°) 이내 → 그 축 active.
             (한계는 제어점 convex hull 이라 여기서 닿으면 궤적도 그 knot 에서 닿음)
  Cartesian: 스윙 아크 TCP(fk_pos) 가 nominal 엔벨로프(x≤0.20, z≤0.04, z≥0.85)에
             들어가면 보고. (이 빌드는 Cartesian off 라 실제로 들어갈 수 있음)

사용: PYTHONPATH=$HOME/ros2_ws/src .venv/bin/python \
        tools/report_db_limit_contacts.py skills/<db>.pkl
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skills"))
import throw_nlp  # noqa: E402
from throwing import fk_pos  # noqa: E402

JOINT_EPS = np.deg2rad(0.5)          # 한계 접촉 판정 (°)
NOM_MIN_X, NOM_MIN_Z, NOM_MAX_Z = 0.20, 0.04, 0.85   # nominal Cartesian 엔벨로프
_SIGN = np.array([1, 1, -1, -1, -1, -1])             # planner -> robot
NAMES = ["J1(S)", "J2(L)", "J3(U)", "J4(R)", "J5(B)", "J6(T)"]


def _arc_xyz(P, tf, n=200):
    sp = [BSpline(throw_nlp.KNOTS, P[j], throw_nlp.DEGREE) for j in range(6)]
    q = lambda t: np.array([sp[j](min(max(t / tf, 0.0), 1.0)) for j in range(6)])
    return np.array([fk_pos(q(t)) for t in np.linspace(0.0, tf, n)])


def main(path):
    d = pickle.load(open(path, "rb"))
    e = d["entries"] if isinstance(d, dict) and "entries" in d else d
    p = d.get("params", {}) if isinstance(d, dict) else {}
    Q_LO = np.asarray(p.get("q_lo", throw_nlp.Q_LO), float)
    Q_HI = np.asarray(p.get("q_hi", throw_nlp.Q_HI), float)
    print(f"DB: {path}  ({len(e)} entries)  rt={p.get('rt')} flight={p.get('flight')}")
    print(f"관절 한계(robot°): " + ", ".join(
        f"{NAMES[j]}[{np.rad2deg(v0):+.0f},{np.rad2deg(v1):+.0f}]"
        for j, (v0, v1) in enumerate(
            (Q_LO[j], Q_HI[j]) if _SIGN[j] > 0 else (-Q_HI[j], -Q_LO[j]) for j in range(6))))
    print(f"Cartesian nominal: x>{NOM_MIN_X}, {NOM_MIN_Z}<z<{NOM_MAX_Z}\n")

    jcount = np.zeros(6, int)
    ccount = 0
    for i, en in enumerate(e):
        P = np.asarray(en["P"], float)
        tf = float(en["t_f"])
        tags = []
        # 관절 접촉 (제어점 기준). 로봇 규약으로 어느 끝인지 표기.
        for j in range(6):
            at_lo = np.any(P[j] <= Q_LO[j] + JOINT_EPS)
            at_hi = np.any(P[j] >= Q_HI[j] - JOINT_EPS)
            if at_lo or at_hi:
                jcount[j] += 1
                # robot 규약 끝단 라벨
                if _SIGN[j] > 0:
                    end = ("lo" if at_lo else "") + ("hi" if at_hi else "")
                else:  # 부호반전 → planner lo=robot hi
                    end = ("hi" if at_lo else "") + ("lo" if at_hi else "")
                tags.append(f"{NAMES[j]}@{end}")
        # Cartesian 접촉
        T = _arc_xyz(P, tf)
        x0, z0, z1 = T[:, 0].min(), T[:, 2].min(), T[:, 2].max()
        cart = []
        if x0 <= NOM_MIN_X:
            cart.append(f"x_min={x0:+.3f}")
        if z0 <= NOM_MIN_Z:
            cart.append(f"z_min={z0:+.3f}")
        if z1 >= NOM_MAX_Z:
            cart.append(f"z_max={z1:+.3f}")
        if cart:
            ccount += 1
        if tags or cart:
            tg = np.round(en["target"], 3)
            st = np.round(en["p_start"], 2)
            print(f"  [{i:3d}] st{st}→tg{tg} | 관절:{','.join(tags) or '-'} | "
                  f"Cart:{' '.join(cart) or '-'}")

    print(f"\n요약: 관절 접촉 {int((jcount>0).any()) and '' or ''}")
    for j in range(6):
        if jcount[j]:
            print(f"   {NAMES[j]}: {jcount[j]}/{len(e)} entry 가 한계 접촉")
    print(f"   Cartesian nominal 엔벨로프 진입: {ccount}/{len(e)} entry")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "skills/warm_db_144_newlim_128.pkl")

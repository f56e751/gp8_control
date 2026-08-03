#!/usr/bin/env python3
"""정적(멈춘 벨트) 연속 pick-and-throw 테스트 — **THR 계획 모델 3종(nlp/dt/phy)**.

`tests/static_pick_throw.py` (구 NLP 스킬) 의 `_thr` 변형이다. 픽/실행/사전계산
흐름은 그 파일을 그대로 재사용하고, 스킬만 `RobustThrowSkill` →
`ThrThrowSkill`(`skills/thr_throw_skill.py`) 로 바꿔 끼운다. 사이클 구조는 동일:

  hover → suction ON → press(지점 z까지 하강) → PICK_TIME 유지
  → **계획 시작 자세로 lift → 계획 아크 스윙 → 릴리즈 → 감속** → 다음 지점 위로 chain

모델 (THR/bench_jitter.py 와 같은 3종, 같은 traj_fn 인터페이스):
  nlp  CasADi/IPOPT B-spline. 릴리즈 **윈도우**(±50 ms) 전 구간 착탄 정확도 강제
  dt   Decision Transformer, THR GP8 rig 학습 (RA-L 2023 재현)
  phy  TossingBot Physics-only 탄도 컨트롤러 (T-RO 2020 재현)

실행 (driver 스택이 떠 있어야 하고 gp8_manager 앱은 꺼져 있어야 한다):

  PYTHONPATH=$HOME/ros2_ws/src ~/ros2_ws/src/gp8_control/.venv/bin/python \\
    -m gp8_control.tests.static_pick_throw_thr --model nlp \\
    --points "0.45,0.20,0.04" --target "1.35,0,-0.08" --plan-only

보통은 tests/run_static_pick_throw_{nlp,dt,phy}.sh 로 실행한다.

──────────────────────────────────────────────────────────────────────────────
NLP 판(static_pick_throw.py)과 달라지는 점
──────────────────────────────────────────────────────────────────────────────
* **기하가 다르다**: THR 경로는 tool=0.240 (link6→TCP 0.32 m, 2026-08-03 URDF
  기준 통일). 구 스킬/`robots/gp8.py` 는 0.220 (0.30 m) 그대로다. 던지기 조준·
  게이트는 THR 기준, 픽(집기)은 gp8.py 기준으로 돈다 — 자세한 것은
  `skills/thr_throw_skill.py` docstring.
* **warm DB 를 쓰지 않는다.** 2026-08-03 공식화 변경(tool/rt/W1/QDD/한계)으로
  기존 skills/warm_db_*.pkl 은 전부 무효다. nlp 은 cold multistart 로 돈다 —
  지점당 수 초~수십 초. 로봇이 움직이기 전에 전 지점을 미리 계획하므로 실행 중
  지연은 없다 (사전계산 시간은 길어진다).
* **`--target` 은 3D 착지 목표 그대로** 다 (구 NLP 판과 동일). dt 는 그중 수평거리
  d_g = hypot(x,y) 와 방향만 쓰고 z 는 착지 예측면으로만 쓴다; phy 는 3D 목표를
  그대로 쓴다; nlp 은 3D 목표를 목적함수에 넣는다.
* 사이클 로그의 `J=` 는 NLP 목적함수가 아니라 **예측 착지 오차 [m]** 다.

THR 전용 옵션 (나머지는 전부 static_pick_throw 와 동일):
  --model {nlp,dt,phy}  계획 모델 (기본 nlp, env GP8_THR_MODEL)
  --weights PATH        dt 체크포인트 (기본 skills/thr/weights/gp8_dt_best.pth)
  --thr-scan            지점을 돌기 전에 target 별 실현 가능성만 표로 출력하고
                        종료 (로봇 무명령, --plan-only 보다 가볍다)
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from gp8_control.tests import static_pick_throw as base
from gp8_control.skills import thr_planners
from gp8_control.skills.thr_throw_skill import ThrThrowSkill


def _scan(model: str, weights, targets, points) -> None:
    """로봇/ROS 없이 각 target 의 계획 실현 가능성만 표로 출력.

    실기 계획과 **같은** 판정을 쓴다 (`thr_throw_skill.check_arc`) — 여기서
    통과한 target 은 계획 단계도 통과한다. p_start 는 첫 지점 기준으로 잡는다
    (nlp 만 p_start 에 민감하다 — dt/phy 는 무관).
    """
    from gp8_control.robots.gp8 import GP8
    from gp8_control.skills.robust_throw_skill import _PLANNER_SIGN
    from gp8_control.skills.thr_throw_skill import check_arc

    robot = GP8()
    traj_fn = thr_planners.get_traj_fn(model, weights)
    x0, y0, _ = points[0]
    p_start = np.array([x0, y0, thr_planners.P_START_Z])
    print(f"\n=== THR 실현 가능성 스캔 [{model}] ===")
    print(f"    기하/공식화: {thr_planners.geometry_summary()}")
    if model == "dt":
        print(f"    weights: {os.path.basename(weights or thr_planners.DT_WEIGHTS)}")
    if model == "nlp":
        print(f"    p_start: {np.round(p_start, 3)} (첫 지점 기준 — nlp 만 민감)")
    print(f"\n{'target (x,y,z)':>22} {'d':>6} | {'관절':>4} {'추종':>4} {'TCP':>4} "
          f"{'착탄':>4} | {'예측착지':>9} {'오차':>7} | {'|v|':>5} {'민감도':>8} "
          f"{'윈도우':>7} | {'계획':>6}")
    n_ok, notes = 0, []
    import time as _t
    for tgt in targets:
        d = float(np.hypot(tgt[0], tgt[1]))
        head = f"{str(np.round(tgt, 3)):>22} {d:6.3f}"
        t0 = _t.time()
        try:
            plan = traj_fn(tgt, p_start)
        except Exception as e:                                  # noqa: BLE001
            print(f"{head} | 계획 실패: {type(e).__name__}: {str(e).splitlines()[0]}")
            continue
        dt_plan = _t.time() - t0
        ts = np.asarray(plan["ts"], float).ravel()
        Q = (np.asarray(plan["Q"], float) * _PLANNER_SIGN).T
        Qd = ((np.asarray(plan["Qd"], float) if plan.get("Qd") is not None
               else np.gradient(np.asarray(plan["Q"], float), ts, axis=0))
              * _PLANNER_SIGN).T
        win = plan.get("release_window")
        if win is not None:
            t_rel, half = 0.5 * (win[0] + win[1]), 0.5 * (win[1] - win[0])
        else:
            t_rel, half = float(plan["t_rel"]), 0.0
        i = int(np.argmin(np.abs(ts - t_rel)))
        c = check_arc(Q[:, :i + 1], Qd[:, :i + 1], ts[:i + 1], tgt, robot, model)

        # 민감도: 릴리즈가 한 샘플 어긋났을 때 착지 이동 (mm/10 ms).
        # ⚠ 반드시 **계획끼리** 비교한다 — 한쪽을 clamp 후 착지로 두면 governor
        #   효과가 민감도로 새어 들어온다 (실측: NLP 1.35 m 에서 13 → 519 mm 로 왜곡).
        sens = ThrThrowSkill.release_sensitivity(Q, Qd, i, float(tgt[2]))
        ok = c["reject"] is None
        n_ok += int(ok)
        print(f"{head} | {'ok' if c['ok_joint'] else 'LIM!':>4} "
              f"{'ok' if c['ok_track'] else 'TRK!':>4} "
              f"{'ok' if c['ok_cart'] else 'TCP!':>4} "
              f"{'ok' if c['ok_land'] else 'LAND':>4} | {c['d_land']:9.3f} "
              f"{c['err'] * 100:6.1f}c | {np.linalg.norm(c['v_eff']):5.2f} "
              f"{sens:6.0f}mm {half * 2e3:5.0f}ms | "
              f"{c['clamp_shift'] * 100:5.1f}c {c['resample_shift'] * 100:5.1f}c | "
              f"{dt_plan:5.1f}s")
        if c["reject"] is not None:
            notes.append(f"  거부 {np.round(tgt, 3)}: {c['reject']}")
        for w in c["warns"]:
            notes.append(f"  경고 {np.round(tgt, 3)}: {w}")

    print(f"\n실행 가능: {n_ok}/{len(targets)} target")
    for line in notes:
        print(line)
    print("\n관절=GP8 위치 한계 / 추종=증분 governor 모사 후 릴리즈 상태 보존 / "
          "TCP=Cartesian 안전 엔벨로프 / 착탄=예측 오차 게이트")
    print("예측착지=THR 기하(tool 0.240) + 무항력 포물선 기준 착지 거리, "
          "민감도=릴리즈 10 ms 어긋남당 착지 변화")
    print("윈도우=릴리즈 허용 폭 (nlp 만 >0 — 그 안 어디서 놓아도 착탄이 보장된다)")


def main() -> None:
    if "-h" in sys.argv or "--help" in sys.argv:
        print(__doc__)
        print("=" * 76)
        print("아래는 static_pick_throw 에서 그대로 물려받은 옵션이다:\n")

    # allow_abbrev=False 필수: 켜져 있으면 argparse 가 `--target` 을 THR 옵션의
    # 축약으로 삼켜 static_pick_throw 의 --target 이 사라진다.
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--model", default=os.environ.get("GP8_THR_MODEL", "nlp"),
                    choices=thr_planners.MODELS)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--thr-scan", action="store_true")
    thr_args, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + rest

    if thr_args.thr_scan:
        from gp8_control.config import Config
        ap2 = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        ap2.add_argument("--points", default="0.45,0.20,0.062")
        ap2.add_argument("--target", default="1.35,0,-0.08")
        a2, _ = ap2.parse_known_args()
        pts = base._parse_points(a2.points, default_z=Config().GRASP_Z)
        # 스캔은 target 만 보므로 지점 수와 개수를 맞추라고 요구하지 않는다.
        n_tok = len([t for t in a2.target.split(";") if t.strip()])
        tgts = base._parse_targets(a2.target, max(n_tok, 1))
        _scan(thr_args.model, thr_args.weights, tgts, pts)
        return

    # 스킬 주입: static_pick_throw._build_ctx_and_skill 은 모듈 전역
    # `RobustThrowSkill` 을 인스턴스화한다. 이름만 바꿔 끼우면 사전계산/실행
    # 경로 전체가 THR 스킬을 쓴다 (원본 파일은 건드리지 않는다).
    model, weights = thr_args.model, thr_args.weights

    class _Injected(ThrThrowSkill):
        def __init__(self, ctx):
            super().__init__(ctx, model=model, weights=weights)

    base.RobustThrowSkill = _Injected
    print(f"[THR] model = {model}, skill = ThrThrowSkill")
    print(f"[THR] {thr_planners.geometry_summary()}")
    if model == "dt":
        print(f"[THR] weights = {weights or thr_planners.DT_WEIGHTS}")
    if model == "nlp":
        print("[THR] warm DB 미사용 — cold multistart (지점당 수 초~수십 초, "
              "전부 로봇 이동 전에 끝난다)")
    base.main()


if __name__ == "__main__":
    main()

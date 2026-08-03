#!/usr/bin/env python3
"""정적(멈춘 벨트) 연속 pick-and-throw 테스트 — **Thr_DT Decision Transformer** 판.

`tests/static_pick_throw.py` (NLP planner) 의 `_dt` 변형이다. 픽/실행/사전계산
흐름은 그 파일을 그대로 재사용하고, 스킬만 `RobustThrowSkill`(CasADi NLP) →
`DtThrowSkill`(Thr_DT DT, `skills/robust_throw_skill_dt.py`) 로 바꿔 끼운다.
사이클 구조는 동일하다:

  hover → suction ON → press(지점 z까지 하강) → PICK_TIME 유지
  → **DT home 자세로 lift → DT 궤적 스윙 → 릴리즈 → 감속** → 다음 지점 위로 chain

실행 (driver 스택이 떠 있어야 하고 gp8_manager 앱은 꺼져 있어야 한다):

  PYTHONPATH=$HOME/ros2_ws/src ~/ros2_ws/src/gp8_control/.venv/bin/python \\
    -m gp8_control.tests.static_pick_throw_dt \\
    --points "0.45,0.20,0.04" --target "0.9,0,0" --plan-only

──────────────────────────────────────────────────────────────────────────────
NLP 판과 달라지는 점 (`--target` 해석이 특히 다르다)
──────────────────────────────────────────────────────────────────────────────
* **`--target` 의 x,y 는 방향과 거리로만 쓰인다.** DT 는 목표 **거리**
  d_g = hypot(x, y) 하나로 조건화되고, 방향은 J1(S)이 atan2(y, x) 로 잡는다
  (논문 §5.1). z 는 DT 에 들어가지 않고 **착지 예측 평면**으로만 쓴다 — Thr_DT
  자체는 물체가 base frame z ≈ 0 에 떨어진다고 가정하고 학습됐다. bin 바닥이
  z=−0.08 이면 z 를 그렇게 주면 되지만, DT 가 그걸 알고 던지는 것은 아니다.
* **d_g ∈ [0.50, 2.00] m** 밖은 학습 범위 밖이라 거부된다 (논문 §5.1).
* **던지기 시작 자세가 고정**이다. NLP 는 grasp + THROW_LIFT 에서 스윙을 시작하지만
  DT 는 항상 학습 시 home 자세에서 출발한다 — 어디서 집었는지가 던지기에 영향을
  주지 않는다. 그래서 `--points` 는 "무엇을 집을지"만 정하고, 같은 target 이면
  모든 지점의 던지기 궤적이 동일하다.
* **계획이 빠르다.** NLP 는 지점당 1~14 s(warm DB 미스 시)지만 DT rollout 은 수십 ms
  다. warm DB 가 없고 `GP8_THROW_WARM_DB` 같은 env 도 안 쓴다.
* **릴리즈 윈도우가 없다.** NLP 해는 t*±25 ms 어디서 놓아도 착지가 보장되지만 DT 는
  한 점이다 — 로그의 `민감도 N mm/10 ms` 가 밸브 jitter 1 스텝당 착지 변화량이다.
* 사이클 로그에 `J=` 로 찍히는 값은 NLP 의 목적함수가 아니라 **실제 GP8 형상 기준
  예측 착지 오차 [m]** 다 (DtThrowSkill 이 그 자리에 넣는다).

⚠ 대부분의 목표에서 DT 궤적은 GP8 관절 한계를 벗어난다 (Thr_DT 평면 시뮬에는
위치 한계가 없어 학습된 백스윙이 U(J3)를 −185°…−260° 까지 감는다). 계획 단계가
그런 지점을 거부하므로 로봇은 그 지점을 집지도 않는다. 사전 스캔 결과 실행 가능한
d_g 는 `weights/dt_best.pth` 기준 0.50…1.05 m — 자세한 것은
`skills/robust_throw_skill_dt.py` 의 docstring 참고.

DT 전용 옵션 (나머지 옵션은 전부 static_pick_throw 와 동일):
  --weights PATH        DT 가중치 (기본 skills/thr_dt/weights/dt_best.pth,
                        env GP8_DT_WEIGHTS)
  --dt-return R         조건화 리턴 R̂ (기본 1.0 = 성공한 던지기)
  --dt-scan             지점을 돌기 전에 요청된 target 들의 실현 가능성만 표로
                        출력하고 종료 (로봇 무명령, --plan-only 보다 가볍다)
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from gp8_control.tests import static_pick_throw as base
from gp8_control.skills.robust_throw_skill_dt import (
    DG_MAX,
    DG_MIN,
    DT_TARGET_RETURN,
    DT_WEIGHTS,
    DtThrowSkill,
)


def _scan(weights: str, target_return: float, targets, points) -> None:
    """로봇/ROS 없이 각 target 의 DT 궤적 실현 가능성만 표로 출력.

    실기 계획과 **같은** 판정을 쓴다 (`robust_throw_skill_dt.check_arc`) — 여기서
    통과한 target 은 계획 단계도 통과한다.
    """
    from gp8_control.robots.gp8 import GP8
    from gp8_control.skills.robust_throw_skill_dt import (
        _load_dt, _predict_landing, _rollout, check_arc,
    )

    robot = GP8()
    model, sim_cfg, tau = _load_dt(weights)
    print(f"\n=== DT 실현 가능성 스캔 ({weights}, τ={tau:.4f}, R̂={target_return}) ===")
    print(f"{'target (x,y,z)':>21} {'d_g':>6} {'yaw':>6} | {'관절':>4} {'속도':>4} "
          f"{'TCP':>4} {'착탄':>4} | {'DT-sim':>7} {'realGP8':>8} {'오차':>7} | "
          f"{'|v|':>5} {'민감도':>7} | {'v peak':>6} {'lag°':>5} {'clamp':>6}")
    seen, n_ok, notes = {}, 0, []
    for t in targets:
        d_g = float(np.hypot(t[0], t[1]))
        yaw = float(np.arctan2(t[1], t[0]))
        head = f"{str(np.round(t, 3)):>21} {d_g:6.3f} {np.rad2deg(yaw):+6.1f}"
        key = (round(d_g, 4), round(float(t[2]), 4))
        if key in seen:
            print(f"{head} | (위 target 과 동일)")
            n_ok += seen[key]
            continue
        if not (DG_MIN - 1e-9 <= d_g <= DG_MAX + 1e-9):
            print(f"{head} | d_g 가 학습 범위 [{DG_MIN}, {DG_MAX}] m 밖 — 거부")
            seen[key] = 0
            continue
        roll = _rollout(model, sim_cfg, tau, d_g, target_return)
        if roll["i_rel"] is None:
            print(f"{head} | 릴리즈 없음 ({roll['n_steps']}스텝 시간종료) — 거부")
            seen[key] = 0
            continue
        n = roll["i_rel"] + 1
        q = np.zeros((6, n)); qd = np.zeros((6, n))
        q[0] = yaw
        q[1], q[2], q[4] = roll["q"][0, :n], roll["q"][1, :n], roll["q"][2, :n]
        qd[1], qd[2], qd[4] = roll["qd"][0, :n], roll["qd"][1, :n], roll["qd"][2, :n]
        c = check_arc(q, qd, roll["t"][:n], t, robot)

        prev, _, _ = _predict_landing(q[:, -2], qd[:, -2], float(t[2]))
        sens = (np.linalg.norm(prev - c["p_land"]) * 1000.0
                if (prev is not None and c["p_land"] is not None) else float("nan"))
        ok = c["reject"] is None
        seen[key] = 1 if ok else 0
        n_ok += seen[key]
        print(f"{head} | {'ok' if c['ok_joint'] else 'LIM!':>4} "
              f"{'ok' if c['ok_track'] else 'TRK!':>4} "
              f"{'ok' if c['ok_cart'] else 'TCP!':>4} "
              f"{'ok' if c['ok_land'] else 'LAND':>4} | {roll['x_land_sim']:7.3f} "
              f"{c['d_land']:8.3f} {c['err'] * 100:6.1f}c | "
              f"{np.linalg.norm(c['v_eff']):5.2f} {sens:5.0f}mm | "
              f"{c['peak_vel_ratio']:6.2f} {c['lag_deg']:5.2f} "
              f"{c['clamp_shift'] * 100:5.1f}c")
        if c["reject"] is not None:
            notes.append(f"  거부 d_g={d_g:.3f}: {c['reject']}")
        for w in c["warns"]:
            notes.append(f"  경고 d_g={d_g:.3f}: {w}")

    print(f"\n실행 가능: {n_ok}/{len(targets)} target "
          f"(지점 {len(points)}개는 무엇을 집을지만 정한다 — DT 던지기는 "
          f"grasp 위치와 무관)")
    for line in notes:
        print(line)
    print("\n관절=GP8 위치 한계 / 속도=증분 governor 모사 후 릴리즈 상태 보존 / "
          "TCP=Cartesian 안전 엔벨로프 / 착탄=realGP8 예측 오차 게이트")
    print("DT-sim=Thr_DT 평면 시뮬 착지, realGP8=실제 GP8 형상 기준 착지 예측 "
          "(a1·a3 offset + tool 8 cm 차이 반영)")
    print("민감도=릴리즈가 10 ms 어긋날 때 착지가 움직이는 거리 "
          "(NLP 과 달리 DT 는 릴리즈 윈도우가 없다)")
    print("v peak=RT 스트림 상한 대비 아크 최대 속도 (10 Hz 스텝 시작의 2ω "
          "과도현상 — 그 자체로는 거부 사유가 아니다)")
    print("lag°/clamp=컨트롤러 증분 governor 를 모사했을 때 릴리즈 시점 추종 오차와 "
          "그로 인한 착지 이동 — 이게 hard 게이트다")


def main() -> None:
    if "-h" in sys.argv or "--help" in sys.argv:
        print(__doc__)
        print("=" * 76)
        print("아래는 static_pick_throw 에서 그대로 물려받은 옵션이다:\n")

    # allow_abbrev=False 필수: 켜져 있으면 argparse 가 `--target` 을
    # `--dt-return` 이전 이름의 축약으로 삼키거나, 앞으로 추가될 `--target*`
    # 옵션과 충돌한다 (실제로 `--target-return` 이었을 때 `--target` 이
    # 그쪽으로 매칭돼 static_pick_throw 의 --target 이 사라졌다).
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--weights", default=DT_WEIGHTS)
    ap.add_argument("--dt-return", type=float, default=DT_TARGET_RETURN)
    ap.add_argument("--dt-scan", action="store_true")
    dt_args, rest = ap.parse_known_args()

    # DT 옵션은 여기서 소비하고, 나머지는 static_pick_throw.main() 의 파서로 넘긴다.
    sys.argv = [sys.argv[0]] + rest

    if dt_args.dt_scan:
        # target/points 만 파싱해서 실현 가능성 표를 찍고 끝낸다 (ROS 불필요).
        from gp8_control.config import Config
        ap2 = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        ap2.add_argument("--points", default="0.45,0.20,0.062;0.45,-0.10,0.062")
        ap2.add_argument("--target", default="1.2,0,0")
        a2, _ = ap2.parse_known_args()
        pts = base._parse_points(a2.points, default_z=Config().GRASP_Z)
        # 스캔은 target 만 보는 것이므로 지점 수와 개수를 맞추라고 요구하지 않는다
        # (실행 모드에서는 static_pick_throw 가 그대로 1:1 을 강제한다).
        n_tok = len([t for t in a2.target.split(";") if t.strip()])
        tgts = base._parse_targets(a2.target, max(n_tok, 1))
        _scan(dt_args.weights, dt_args.dt_return, tgts, pts)
        return

    # 스킬 주입: static_pick_throw._build_ctx_and_skill 은 모듈 전역
    # `RobustThrowSkill` 을 인스턴스화한다. 이름만 바꿔 끼우면 사전계산/실행
    # 경로 전체가 DT 스킬을 쓴다 (원본 파일은 건드리지 않는다).
    weights, target_return = dt_args.weights, dt_args.dt_return

    class _Injected(DtThrowSkill):
        def __init__(self, ctx):
            super().__init__(ctx, weights=weights, target_return=target_return)

    base.RobustThrowSkill = _Injected
    print(f"[DT] skill = DtThrowSkill, weights = {weights}, R̂ = {target_return}")
    print("[DT] --target 의 x,y = 던지기 거리/방향, z = 착지 예측 평면 "
          "(DT 조건화에는 거리 d_g 만 들어간다)")
    base.main()


if __name__ == "__main__":
    main()

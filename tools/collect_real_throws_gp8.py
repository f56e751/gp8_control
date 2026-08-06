#!/usr/bin/env python3
"""DT sim2real — **실기 던지기 데이터 수집** (착지 거리는 조작자가 실측 입력).

논문 §4.4/§5.4 의 절차를 실물 GP8 에 옮긴 것:

    "a dataset of real throws was collected by generating a set of random goals
     and attempting to throw to them using the pre-trained model and when
     alpha_max = 3.0. After applying HER, the prior DT model is refined."

THR 의 `dt_finetune_gp8.collect_real_throws` 는 이 '실기' 를 **PyBullet 시뮬**로
대신했다 (지터·무작위 파지를 켠 프록시). 이 스크립트는 그 자리에 **진짜 로봇**을
넣는다. 파이프라인 5단계 중 바뀌는 것은 ②실행 ③측정 둘뿐이고, ①rollout ④HER
⑤파인튜닝은 THR 코드를 그대로 쓴다 (④⑤는 tools/finetune_dt_gp8.py).

    ① DT rollout + 게인 α   thr_planners.dt_rollout_with_gain   (그대로)
    ② 궤적 실행             ← **로봇** (이 파일)
    ③ 착지 측정             ← **조작자 실측 입력** (이 파일)
    ④ HER 재라벨 K=0        skills/thr/dt_model/her.py          (그대로)
    ⑤ 파인튜닝              skills/thr/dt_model/trainer.py      (그대로)

──────────────────────────────────────────────────────────────────────────────
목표 거리 커버리지가 개수보다 중요하다
──────────────────────────────────────────────────────────────────────────────
기존 `data_gp8/real_gp8_n-12_herK-0.pkl` (시뮬 프록시)은 11궤적인데 HER 목표
(=실제 착지거리)가 0.87~1.80 m 중 7개가 1.42 m 이상으로 몰려 있다. α~U(1,3) 가
액션을 최대 3배로 키워 대부분 멀리 날아갔고, HER K=0 은 달성한 착지를 무조건
성공(+1)으로 만든다. 그 결과 파인튜닝된 `gp8_dt_ft2_real-11.pth` 는 가까운
목표에도 세게 던진다 (실측: 1.10 m 목표에 1.67 m).

그래서 이 스크립트는 **목표 거리를 균등 격자로 순회**하고(`--goals`), 각 거리마다
`--reps` 회씩 던진다. 수집 중 실제 착지 히스토그램을 계속 보여주므로 한쪽으로
쏠리면 바로 알 수 있다.

──────────────────────────────────────────────────────────────────────────────
안전 (2026-08-04 사용자 지시 "안전")
──────────────────────────────────────────────────────────────────────────────
* α 기본 범위는 **U(1.0, 1.5)** — 논문의 3.0 이 아니다. α 는 모터 액션을 그대로
  배수하므로 실기에서는 스윙 속도가 그만큼 올라가 컨트롤러 증분 governor 상한과
  관절 한계에 걸린다. 탐색 다양성을 일부 포기하고 안전을 택한 값이다.
* 뽑은 α 로 만든 궤적이 런타임과 **같은 게이트**(`thr_throw_skill.check_arc` +
  dispatch 직전 관절/Cartesian 게이트)를 통과하지 못하면 α 를 `--alpha-step`
  배씩 낮춰 재시도하고, α=1.0 에서도 실패하면 그 던지기를 건너뛴다.
  → 게이트를 통과한 궤적만 로봇에 나간다.
* 매 사이클 조작자 확인(Enter)을 받는다. `--no-confirm` 은 두지 않았다.
* Ctrl-C / 예외 / 정상 종료 어디서든 suction OFF 를 보낸다.
* 수집물은 던지기 **한 번마다 즉시** 파일에 append 된다 (중단해도 유실 없음).

──────────────────────────────────────────────────────────────────────────────
실행
──────────────────────────────────────────────────────────────────────────────
    cd ~/ros2_ws/src/gp8_control
    ./tools/run_collect_real_throws_gp8.sh

기본값은 --goals 1.1,1.3,1.5,1.7 --reps 5 이다. 다른 값을 쓰려면:
    ./tools/run_collect_real_throws_gp8.sh --goals 1.1,1.3 --reps 3

스크립트가 ROS/venv 환경을 로드하고 driver 스택을 확인해 없으면 자동 시작한다.
gp8_manager 앱이 켜져 있으면 충돌 방지를 위해 실행하지 않는다.

**착지 거리 입력 규약**: 로봇 **베이스 회전축(J1 축) 중심**에서 물체가 처음
떨어진 지점까지의 **수평 거리 [m]**. 이것이 DT 의 목표 d_g 와 같은 정의다.
가로 편차까지 남기고 싶으면 "x,y" 로 입력해도 된다 (거리는 hypot 으로 계산).
빈 입력 = 그 던지기 버림 (측정 실패/물체 이탈 등).
"""

from __future__ import annotations

import argparse
import datetime
import os
import pickle
import sys
import threading
import time

import numpy as np


# tests/run_static_pick_throw_dt.sh 의 24 cm TCP 픽 격자와 동일하다.
# 수집 루프는 이 순서를 idx % len(points)로 순환한다.
DEFAULT_POINTS = ";".join((
    "0.40,0.30,0.02", "0.40,0.20,0.02", "0.40,0.10,0.02",
    "0.40,0.0,0.02", "0.40,-0.10,0.02", "0.40,-0.20,0.02",
    "0.50,0.30,0.02", "0.50,0.20,0.02", "0.50,0.10,0.02",
    "0.50,0.0,0.02", "0.50,-0.10,0.02", "0.50,-0.20,0.02",
))


def _parse_goals(spec: str) -> list:
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok:
            out.append(float(tok))
    if not out:
        sys.exit("--goals 비어 있음")
    return out


def _ask_landing(prompt: str):
    """조작자에게 착지 거리를 받는다. 리턴: (d, xy|None) 또는 None(버림)."""
    while True:
        raw = input(prompt).strip()
        if raw == "":
            return None
        if raw.lower() in ("q", "quit"):
            raise KeyboardInterrupt
        try:
            vals = [float(v) for v in raw.replace(" ", ",").split(",") if v]
        except ValueError:
            print("    숫자로 입력하라 (예: 1.23  또는  1.20,0.10). 빈 줄 = 버림")
            continue
        if len(vals) == 1:
            return float(vals[0]), None
        if len(vals) == 2:
            return float(np.hypot(vals[0], vals[1])), (vals[0], vals[1])
        print("    'd' 또는 'x,y' 형식이어야 한다")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="DT 실기 던지기 데이터 수집 (착지 실측 입력)")
    ap.add_argument("--goals", default="0.9,1.1,1.3,1.5,1.7",
                    help="목표 거리 격자 [m] (균등 커버리지가 중요 — 모듈 docstring)")
    ap.add_argument("--reps", type=int, default=1, help="목표당 던지기 수")
    ap.add_argument("--points", default=DEFAULT_POINTS,
                    help='물체 픽 지점 "x,y,z;..." — 기본 12개를 순환하며 사용')
    ap.add_argument("--yaw-deg", type=float, default=0.0,
                    help="던지는 방향 (base frame). 목표는 (d·cos, d·sin, --land-z)")
    ap.add_argument("--land-z", type=float, default=0.0,
                    help="착지면 높이 [m] — 바닥이면 0, bin 바닥이면 -0.08")
    ap.add_argument("--weights", default=None, help="DT 체크포인트 (기본 v9)")
    ap.add_argument("--alpha-max", type=float, default=1.5,
                    help="액션 게인 α~U(1,이 값). 논문 §5.4 는 3.0 — 실기 안전상 "
                         "1.5 로 운용 (2026-08-06 사용자 지시). --alpha-max 3.0 "
                         "으로 언제든 논문값 실행 가능")
    ap.add_argument("--alpha-step", type=float, default=0.9,
                    help="게이트 실패 시 α 를 이 배수로 낮춰 재시도")
    ap.add_argument("--vel-scale", type=float, default=0.2,
                    help="던지기 외 이동 속도 (계획 아크·감속은 영향 없음)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="기본 data_real/real_throws_<날짜>.pkl (이어붙임)")
    args, _ros = ap.parse_known_args()

    if not 0.0 < args.vel_scale <= 1.0:
        sys.exit("--vel-scale 은 (0, 1]")
    if args.alpha_max < 1.0:
        sys.exit("--alpha-max 는 1.0 이상")

    # NLP/torch 가 스레드를 코어 수만큼 띄우면 ros2_control 의 4ms RT 루프가 굶어
    # RUN_STALL → comm-loss 로 간다 (2026-07-21 실기 재현).
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(1)
    except ImportError:
        pass

    # 수집물에는 **카타시안 위반 궤적을 넣지 않는다** (2026-08-04 사용자 지시).
    # 런타임은 보고만 하지만, DB/데이터셋에 들어가면 그 해가 계속 재사용되므로
    # 수집 경로에서는 제외한다 — warm DB 빌더도 같은 규칙(cart_blocking=True).
    from gp8_control.skills import thr_throw_skill as _tts
    _tts.CART_BLOCKING = True

    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node

    from gp8_control.config import Config
    from gp8_control.controllers.trajectory_controller import TrajectoryController
    from gp8_control.robots.gp8 import GP8
    from gp8_control.skills import thr_planners
    from gp8_control.skills.thr_throw_skill import ThrThrowSkill
    from gp8_control.tests import static_pick_throw as base
    from gp8_control.trajectory.trajectory_primitive import trajectory

    weights = args.weights or thr_planners.DT_WEIGHTS
    goals = _parse_goals(args.goals)
    points = base._parse_points(args.points, default_z=Config().GRASP_Z)
    out_path = args.out or os.path.join(
        os.path.expanduser("~/ros2_ws/src/gp8_control/data_real"),
        f"real_throws_{datetime.date.today().isoformat()}.pkl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rng = np.random.default_rng(args.seed)
    yaw = np.deg2rad(args.yaw_deg)

    # 기존 수집물 이어붙이기
    records = []
    if os.path.exists(out_path):
        with open(out_path, "rb") as f:
            records = pickle.load(f)
        print(f"기존 수집물 {len(records)}건 이어붙임: {out_path}")

    print(f"\n=== DT 실기 데이터 수집 ===")
    print(f"  weights : {os.path.basename(weights)}")
    print(f"  목표    : {goals} m × {args.reps}회 = {len(goals) * args.reps} 던지기")
    print(f"  방향    : yaw {args.yaw_deg:+.1f}°, 착지면 z={args.land_z:+.3f} m")
    print(f"  게인    : α ~ U(1.0, {args.alpha_max}) — 게이트 실패 시 "
          f"×{args.alpha_step} 씩 낮춰 재시도 (안전)")
    print(f"  출력    : {out_path}")

    rclpy.init()
    node = Node("collect_real_throws_gp8")
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    traj_ctrl = TrajectoryController(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    suction_used = False
    try:
        if not traj_ctrl.wait_for_servers(timeout_sec=10.0):
            sys.exit("컨트롤러 액션 서버 없음 — driver 스택(bringup) 확인")
        t_end = time.time() + 5.0
        while traj_ctrl.current_joints is None and time.time() < t_end:
            time.sleep(0.05)
        if traj_ctrl.current_joints is None:
            sys.exit("/joint_states 미수신 — driver 스택 확인")

        cfg, robot = Config(), GP8()

        # 게인 rollout 을 쓰는 traj_fn 을 스킬에 주입한다. 게이트/lift/감속/
        # dispatch 는 전부 ThrThrowSkill 것을 그대로 탄다.
        box: dict = {}

        def _gain_traj_fn(target, p_start):
            from gp8_control.skills.thr.dt_gp8_env import DG_MAX, DG_MIN
            d = float(np.hypot(target[0], target[1]))
            d_clip = min(max(d, DG_MIN), DG_MAX)
            mem, k_rel, x_land = thr_planners.dt_rollout_with_gain(
                d_clip, box["alpha"], weights=weights, logger=node.get_logger())
            if k_rel is None:
                raise ValueError(f"DT 미-release (d_g={d_clip:.2f}m, α={box['alpha']:.2f})")
            box["mem"], box["k_rel"], box["x_land_env"] = mem, k_rel, x_land
            box["d_clip"] = d_clip
            return thr_planners.dt_plan_from_mem(
                mem, k_rel, x_land, target, d_clip, weights,
                logger=node.get_logger(), alpha=box["alpha"])

        base.RobustThrowSkill = lambda ctx: ThrThrowSkill(
            ctx, model="dt", weights=weights, traj_fn=_gain_traj_fn)
        ctx, skill, M1, M2, idle_joint = base._build_ctx_and_skill(
            cfg, robot, node, traj_ctrl)
        ctx.M1 = ctx.M1 * args.vel_scale
        ctx.M2 = ctx.M1 * cfg.JOINT_ACCEL_LIMIT_SCALE
        pos_M1, pos_M2 = ctx.M1, ctx.M2
        zero6 = np.zeros(6)

        # 시작: idle 자세로
        q_cur = np.asarray(traj_ctrl.current_joints, float)
        mv = trajectory(q_cur, zero6, idle_joint, zero6, pos_M1, pos_M2,
                        hertz=cfg.TRAJ_HZ)
        print(f"\n→ init(idle) 자세로 이동 ({mv[2][-1]:.2f}s)")
        traj_ctrl.send_trajectory_queue(mv[0], mv[1], mv[2], final_joint=idle_joint)

        print("\n물체를 픽 지점에 놓고 시작하라. 던지기는 실제 스윙 (반경 1 m 확보).")
        print("착지 거리는 매 던지기 후 직접 재서 입력한다 "
              "(베이스 축 → 착지점 수평거리 [m], 빈 줄 = 그 던지기 버림).")
        if input("시작 Enter (q 취소): ").strip().lower() == "q":
            return

        suction_used = True
        traj_ctrl.suction_off()

        plan_list = [(g, r) for g in goals for r in range(args.reps)]
        n_done = n_skip = 0
        for idx, (d_goal, rep) in enumerate(plan_list):
            x, y, z = points[idx % len(points)]
            target = np.array([d_goal * np.cos(yaw), d_goal * np.sin(yaw),
                               args.land_z])
            print(f"\n===== [{idx + 1}/{len(plan_list)}] 목표 {d_goal:.2f} m "
                  f"(rep {rep + 1}/{args.reps}) | 픽 ({x:+.3f}, {y:+.3f}, {z:+.3f}) =====")

            # ---- α 를 낮춰가며 게이트를 통과하는 계획을 찾는다 (안전) ----
            T_grasp, q_grasp, q_hover, q_press = base._solve_point(
                skill, robot, cfg, x, y, z)
            if q_grasp is None or q_hover is None or q_press is None:
                print("  픽 지점 IK 실패 — 건너뜀")
                n_skip += 1
                continue
            alpha = float(rng.uniform(1.0, args.alpha_max))
            planned = None
            while True:
                box["alpha"] = alpha
                planned = skill.plan_nlp_throw(q_press, T_grasp, target)
                if planned is not None:
                    break
                if alpha <= 1.0 + 1e-9:
                    break
                alpha = max(1.0, alpha * args.alpha_step)
                print(f"  게이트 실패 → α 를 {alpha:.3f} 으로 낮춰 재시도")
            if planned is None:
                print("  α=1.0 에서도 게이트 통과 실패 — 이 던지기 건너뜀")
                n_skip += 1
                continue
            res, lift = planned

            # ---- 궤적 조립 (dispatch 는 캡처해서 확인 후 재생) ----
            cap = base._CaptureThrowCtrl()
            real_ctrl, ctx.traj_ctrl = ctx.traj_ctrl, cap
            try:
                skill.build_throw_trajectory(q_press, res, lift, next_grasp=None)
            finally:
                ctx.traj_ctrl = real_ctrl
            if cap.throw is None:
                print("  던지기 궤적이 안전 게이트에 걸림 — 건너뜀")
                n_skip += 1
                continue

            print(f"  계획 OK: α={alpha:.3f}, env 예측 착지 "
                  f"{box['x_land_env']:.3f} m, THR 기하 예측 {res['d_land']:.3f} m")
            ans = input("  이 사이클 실행 Enter (s 건너뜀, q 종료): ").strip().lower()
            if ans == "q":
                break
            if ans == "s":
                n_skip += 1
                continue

            # ---- 픽: 접근 → hover → suction → press → 유지 ----
            q_cur = np.asarray(traj_ctrl.current_joints, float)
            a_traj, a_vel, a_ts = trajectory(q_cur, zero6, q_hover, zero6,
                                             pos_M1, pos_M2, hertz=cfg.TRAJ_HZ)
            print(f"  → hover 이동 ({a_ts[-1]:.2f}s)")
            traj_ctrl.send_trajectory_queue(a_traj, a_vel, a_ts,
                                            final_joint=q_hover)
            traj_ctrl.suction_on()
            d_traj, d_vel, d_ts = trajectory(q_hover, zero6, q_press, zero6,
                                             pos_M1, pos_M2, hertz=cfg.TRAJ_HZ)
            from gp8_control.skills.robust_throw_skill import PICK_TIME
            print(f"  → press 하강 z={z:+.3f}m ({d_ts[-1]:.2f}s) + {PICK_TIME:.1f}s 유지")
            traj_ctrl.send_trajectory_queue(d_traj, d_vel, d_ts,
                                            final_joint=q_press)
            time.sleep(PICK_TIME)

            ans = input("  물체 grasp 완료 — 던지려면 Enter (q 던지기 생략): ").strip().lower()
            if ans == "q":
                print("  던지기 생략 — suction OFF")
                traj_ctrl.suction_off()
                n_skip += 1
                continue

            # ---- 던지기 ----
            th = cap.throw
            print(f"  → THROW (아크 {res['t_f']:.3f}s, release idx "
                  f"{th['release_index']}/{th['traj'].shape[1] - 1})")
            traj_ctrl.send_trajectory_queue_with_timed_release(
                th["traj"], th["vel"], th["ts"],
                final_joint=th["final_joint"],
                release_index=th["release_index"])
            traj_ctrl.suction_off()

            # ---- 착지 실측 입력 ----
            got = _ask_landing("  ▶ 착지 거리 입력 [m] (베이스 축 기준 수평거리, "
                               "'x,y' 도 가능, 빈 줄 = 버림): ")
            if got is None:
                print("  측정 없음 — 이 던지기는 데이터에서 제외")
                n_skip += 1
            else:
                d_actual, xy = got
                records.append(dict(
                    iso_time=datetime.datetime.now().isoformat(timespec="seconds"),
                    weights=os.path.basename(weights),
                    d_goal=float(box["d_clip"]), alpha=float(alpha),
                    yaw=float(yaw), land_z=float(args.land_z),
                    pick_point=(float(x), float(y), float(z)),
                    mem=box["mem"], k_rel=int(box["k_rel"]),
                    x_land_env=float(box["x_land_env"]),
                    d_land_pred=float(res["d_land"]),
                    d_actual=float(d_actual), xy_actual=xy,
                ))
                with open(out_path + ".tmp", "wb") as f:
                    pickle.dump(records, f)
                os.replace(out_path + ".tmp", out_path)   # 원자적 — 중단에 안전
                n_done += 1
                print(f"  기록 ✓ 실측 {d_actual:.3f} m "
                      f"(env 예측 {box['x_land_env']:.3f} → 편차 "
                      f"{1e3 * (d_actual - box['x_land_env']):+.0f} mm, "
                      f"THR 기하 예측 {res['d_land']:.3f} → "
                      f"{1e3 * (d_actual - res['d_land']):+.0f} mm)")
                _print_coverage(records, goals)

        # 종료: idle 복귀
        q_cur = np.asarray(traj_ctrl.current_joints, float)
        mv = trajectory(q_cur, zero6, idle_joint, zero6, pos_M1, pos_M2,
                        hertz=cfg.TRAJ_HZ)
        print(f"\n→ idle 복귀 ({mv[2][-1]:.2f}s)")
        traj_ctrl.send_trajectory_queue(mv[0], mv[1], mv[2], final_joint=idle_joint)

        print(f"\n===== 수집 완료: 기록 {n_done}건, 건너뜀 {n_skip}건 "
              f"(누적 {len(records)}건) =====")
        print(f"  → {out_path}")
        _print_coverage(records, goals)
        print("\n다음 단계: 파인튜닝")
        print(f"  python -m gp8_control.tools.finetune_dt_gp8 --data {out_path}")
    except KeyboardInterrupt:
        print("\n중단됨 (Ctrl-C)")
    finally:
        if suction_used:
            try:
                traj_ctrl.suction_off()
                print("suction OFF 전송됨")
            except Exception:                                   # noqa: BLE001
                pass
        rclpy.shutdown()


def _print_coverage(records, goals):
    """실제 착지 거리 분포 — 한쪽으로 쏠리면 바로 보이게 (ft2 편향 재발 방지)."""
    if not records:
        return
    d = np.array([r["d_actual"] for r in records])
    print(f"    누적 착지 분포 (n={len(d)}): "
          f"min {d.min():.2f} / 중앙 {np.median(d):.2f} / max {d.max():.2f} m")
    edges = np.array(goals, float)
    mid = np.concatenate(([edges[0] - 0.1], (edges[:-1] + edges[1:]) / 2,
                          [edges[-1] + 0.1]))
    cnt, _ = np.histogram(d, bins=mid)
    print("    목표별 실착지 개수: "
          + "  ".join(f"{g:.1f}m:{c}" for g, c in zip(edges, cnt)))


if __name__ == "__main__":
    main()

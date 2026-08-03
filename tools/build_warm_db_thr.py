#!/usr/bin/env python3
"""THR NLP 모델용 warm start DB 빌더 — **grasp 12 × target 12 = 144쌍, 초기값 128개**.

각 (시작점, 타겟) 쌍을 128개 초기해로 전부 풀고 **하나만** 저장한다. 저장 파일은
`skills/thr/warm_db_thr.pkl` (THR nlp_planner 와 같은 v2 축적 포맷) 이고,
런타임(`skills/thr_planners.nlp_traj_fn`)이 target 최근접 entry 로 full warm start
polish 를 걸어 cold multistart 수십 초 → 1~2 초로 줄인다.

──────────────────────────────────────────────────────────────────────────────
저장 정책: **'J 최소' 가 아니라 '실기 게이트를 통과하는 것 중 J 최소'**
──────────────────────────────────────────────────────────────────────────────
THR 빌더(`nlp_planner.build_warm_db`)는 J 최소 해 하나를 저장한다. 시뮬에서는 그게
맞지만 실기에서는 J 최소 해가 곧잘 **컨트롤러 증분 governor 의 스트림 속도 상한**
(robots/gp8._RT_STREAM_VELOCITY_LIMITS, L/U/B = 193/259/275 °/s) 을 넘는다 —
2026-08-03 업데이트로 W1 0.5→5.0, QDD 3×→5× 가 되면서 스윙이 짧고 빨라졌기 때문이다
(실측: 먼 bin 에서 J3 341~372 °/s → governor clamp 로 착지 22~47 cm 이동 → 런타임
`check_arc` 가 거부). 그런 해를 DB 에 넣으면 warm polish 도 같은 basin 으로 수렴해
결국 거부되므로 DB 가 무용지물이 된다.

그래서 이 빌더는 128개 해를 전부 `thr_throw_skill.check_arc` (런타임과 **같은**
판정: 관절 한계 / governor clamp / Cartesian 엔벨로프 / 착탄) 로 검사하고
  ① 통과한 해 중 J 최소  → 저장 (gate_ok=True)
  ② 하나도 없으면 J 최소 → 저장하되 gate_ok=False 로 표시
한다. ②는 그 쌍이 현재 공식화·하드웨어 조합에서 실행 불가라는 뜻이고, 리포트가
따로 집계한다. 128개나 푸는 이유가 바로 이것 — 느리지만 상한 안에 드는 국소해를
찾을 기회를 넓히는 것이다.

──────────────────────────────────────────────────────────────────────────────
실행
──────────────────────────────────────────────────────────────────────────────
    cd ~/ros2_ws && source install/setup.bash
    PYTHONPATH=$HOME/ros2_ws/src ~/ros2_ws/src/gp8_control/.venv/bin/python \
        -m gp8_control.tools.build_warm_db_thr --workers 7

    # 다른 공식화로 한 벌 더 (같은 파일에 config 로 공존, 런타임이 자동 선택)
    GP8_THR_NLP_W1=0.5 ... -m gp8_control.tools.build_warm_db_thr --workers 7

옵션: --workers N, --n-init 128, --resume(기본 on, 이미 있는 쌍 건너뜀),
      --pairs "pi,ti;..."(부분 재구축), --out PATH

⚠ 오래 걸린다. 144쌍 × 128초기해 = 18,432 solve, 8코어/7워커에서 **약 3시간**
  (구 DB 실측 11,636 s). nohup 으로 띄우고 로그를 보는 것을 권한다.
  워커는 fork Pool 이다 — 이 빌더는 ROS2 노드가 아닌 **오프라인 프로세스**라
  fork 가 안전하다 (런타임 스킬이 순차인 이유와 구분할 것).
  중단해도 쌍 단위로 파일에 반영되므로 --resume 으로 이어서 돌린다.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np

from gp8_control.robots.gp8 import GP8
from gp8_control.skills import thr_planners
from gp8_control.skills.robust_throw_skill import _PLANNER_SIGN
from gp8_control.skills.thr import throw_nlp
from gp8_control.skills.thr_throw_skill import check_arc

# --- 12 grasp 시작점: THR sim_env.OBJ_SLOTS 의 물체 배치 xy, z = P_START_Z ------
OBJ_SLOTS_XY = [(0.40, 0.30), (0.40, 0.20), (0.40, 0.10),
                (0.40, 0.00), (0.40, -0.10), (0.40, -0.20),
                (0.50, 0.30), (0.50, 0.20), (0.50, 0.10),
                (0.50, 0.00), (0.50, -0.10), (0.50, -0.20)]

# --- 12 타겟: THR sim_env.bin_targets() (tilt 0 이라 세 열의 z 가 같다) ---------
#     x ∈ {1.10, 1.35, 1.60} × y ∈ {+0.225, +0.075, −0.075, −0.225}, z = −0.08
BIN_X = (1.10, 1.35, 1.60)
BIN_Y = (0.225, 0.075, -0.075, -0.225)
BIN_Z = -0.08


def points():
    return [np.array([x, y, thr_planners.P_START_Z]) for x, y in OBJ_SLOTS_XY]


def targets():
    return [np.array([x, y, BIN_Z]) for x in BIN_X for y in BIN_Y]


# ---------------------------------------------------------------------------
# 초기해 128개
# ---------------------------------------------------------------------------
def make_inits(p_target, p_start, n_init):
    """고정 INIT_VARIANTS 6개 + 무작위 섭동 (n_init−6)개.

    섭동 seed 는 base 하나가 아니라 INIT_VARIANTS 전체를 순환한다 — 한 계열
    주변만 맴돌지 않고 여러 스윙 스타일의 이웃을 고루 훑는다 (THR plan_best_wide
    와 같은 규칙, 개수만 128 로 확장). seed 는 (target, p_start) 로 고정이라
    재실행해도 같은 후보가 나온다.
    """
    inits = [(f"init{k}", v) for k, v in enumerate(thr_planners.INIT_VARIANTS)]
    base = np.array([0.0, 0.6, 0.9, 0.0, 0.7, 0.0])
    seeds = [np.asarray(v["dq_swing"], float) if v else base
             for v in thr_planners.INIT_VARIANTS]
    rng = np.random.default_rng(
        int(abs(p_target[0] * 1e4 + p_target[1] * 1e3
                + p_start[0] * 1e2 + p_start[1] * 1e1)) % 2**31)
    for k in range(max(0, n_init - len(inits))):
        dq = seeds[k % len(seeds)] + rng.uniform(-0.6, 0.6, 6) * [1, 1, 1, 0, 1, 0]
        inits.append((f"rand{k}", dict(dq_swing=dq.tolist(),
                                       T0=float(rng.uniform(0.35, 1.45)),
                                       chi0=float(rng.uniform(0.25, 0.75)))))
    return inits


# ---------------------------------------------------------------------------
# 워커
# ---------------------------------------------------------------------------
def _init_worker():
    """BLAS/OpenMP 스레드를 1개로 제한. 안 하면 워커 7개 × OpenBLAS 8스레드가
    8코어에서 spin-wait 경합해 후보당 solve 시간이 폭증한다 (THR 실측 8s→85s)."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(1)
    except ImportError:
        pass


def _solve_one(args):
    """후보 하나 solve (순수 NLP). 리턴: (tag, res|None, 실패사유|None)."""
    p_start, p_target, q_start, tag, init = args
    try:
        res = throw_nlp.solve_throw_nlp(p_start, p_target, init=init,
                                        q_start=q_start)
    except (RuntimeError, AssertionError, ValueError) as e:
        return tag, None, f"{type(e).__name__}"
    if res.get("pos_viol_dense", 0.0) > 1e-3:
        return tag, None, f"위치한계 잔존 {res['pos_viol_dense']:.1e}"
    return tag, res, None


# ---------------------------------------------------------------------------
# 게이트 (런타임과 동일 판정)
# ---------------------------------------------------------------------------
def gate_of(res, p_target, robot):
    """해 하나를 런타임 check_arc 로 검사 → (ok, dict 진단)."""
    q_of, qd_of, _ = throw_nlp._spline_eval(res["P"], res["t_f"])
    ts = np.arange(0.0, res["t_f"] + thr_planners.DT / 2, thr_planners.DT)
    Q = (np.array([q_of(t) for t in ts]) * _PLANNER_SIGN).T
    Qd = (np.array([qd_of(t) for t in ts]) * _PLANNER_SIGN).T
    t_rel = float(res["t_star"])            # 윈도우 중앙 = 런타임 릴리즈 시점
    i = int(np.argmin(np.abs(ts - t_rel)))
    c = check_arc(Q[:, :i + 1], Qd[:, :i + 1], ts[:i + 1], p_target, robot, "nlp")
    return c["reject"] is None, c


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="THR NLP warm DB 빌더 (12×12쌍, 초기값 128개)")
    ap.add_argument("--n-init", type=int, default=128)
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 8) - 1))
    ap.add_argument("--out", default=thr_planners.WARM_DB_PATH)
    ap.add_argument("--no-resume", action="store_true",
                    help="이미 저장된 쌍도 다시 푼다 (기본은 건너뜀)")
    ap.add_argument("--pairs", default=None,
                    help='부분 재구축 "pi,ti;pi,ti;..." (0-based)')
    args = ap.parse_args()

    _init_worker()          # 부모도 1스레드 (게이트 계산이 BLAS 를 쓴다)
    thr_planners.WARM_DB_PATH = args.out
    robot = GP8()
    P, T = points(), targets()
    params = thr_planners.warm_db_params()

    print(f"=== THR NLP warm DB 빌드 ===")
    print(f"  출력   : {args.out}")
    print(f"  공식화 : {thr_planners.geometry_summary()}")
    print(f"  쌍     : {len(P)} grasp × {len(T)} target = {len(P) * len(T)}")
    print(f"  초기해 : {args.n_init}/쌍  (총 {len(P) * len(T) * args.n_init} solve)")
    print(f"  워커   : {args.workers} (fork — 오프라인 프로세스라 안전)")

    # 이미 있는 쌍 (같은 공식화 config 안에서만)
    db = thr_planners.db_read(args.out)
    cfg = thr_planners.db_select(db, params)
    have = set()
    if cfg and not args.no_resume:
        for e in cfg["entries"]:
            have.add((tuple(np.round(np.asarray(e["p_start"], float), 3)),
                      tuple(np.round(np.asarray(e["target"], float), 3))))
        print(f"  resume : 기존 entry {len(have)}개 건너뜀")

    todo = [(pi, ti) for pi in range(len(P)) for ti in range(len(T))]
    if args.pairs:
        want = {tuple(int(v) for v in tok.split(","))
                for tok in args.pairs.split(";") if tok.strip()}
        todo = [x for x in todo if x in want]

    t0 = time.time()
    n_done = n_gate_ok = n_skip = n_fail = 0
    for n, (pi, ti) in enumerate(todo, 1):
        p_start, p_target = P[pi], T[ti]
        key = (tuple(np.round(p_start, 3)), tuple(np.round(p_target, 3)))
        tag_pair = (f"[{n:3d}/{len(todo)}] p({p_start[0]:.2f},{p_start[1]:+.2f}) → "
                    f"t({p_target[0]:.2f},{p_target[1]:+.3f})")
        if key in have:
            n_skip += 1
            print(f"{tag_pair}: skip (이미 있음)", flush=True)
            continue

        # 시작자세를 먼저 확정 — 한계 안 해가 없으면 이 쌍은 애초에 불가
        try:
            q_start = throw_nlp.ik_start_pose(p_start)
        except ValueError as e:
            n_fail += 1
            print(f"{tag_pair}: 시작자세 IK 불가 — {e}", flush=True)
            continue

        inits = make_inits(p_target, p_start, args.n_init)
        cands = [(p_start, p_target, q_start, tag, init) for tag, init in inits]
        ts_pair = time.time()
        with mp.get_context("fork").Pool(args.workers,
                                         initializer=_init_worker) as pool:
            out = pool.map(_solve_one, cands)
        feas = [(r["J"], tag, r) for tag, r, err in out if r is not None]
        if not feas:
            n_fail += 1
            why = {}
            for tag, r, err in out:
                if err:
                    why[err] = why.get(err, 0) + 1
            print(f"{tag_pair}: 전멸 ({len(cands)}개 후보) — "
                  f"{', '.join(f'{k}×{v}' for k, v in sorted(why.items()))}",
                  flush=True)
            continue

        # 게이트 통과 해 중 J 최소 (없으면 J 최소 + gate_ok=False)
        # J 오름차순이라 '첫 게이트 통과' 가 곧 '통과 해 중 J 최소' 다.
        feas.sort(key=lambda x: x[0])
        pick, pick_gate, pick_ok = None, None, False
        for J, tag, r in feas:
            ok, c = gate_of(r, p_target, robot)
            if ok:
                pick, pick_gate, pick_ok = (J, tag, r), c, True
                break
        if pick is None:                       # 통과 해 없음 → J 최소를 표시만 하고 저장
            J, tag, r = feas[0]
            _, pick_gate = gate_of(r, p_target, robot)
            pick, pick_ok = (J, tag, r), False

        J, tag, res = pick
        entry = dict(target=p_target, p_start=p_start, q_start=res["q_start"],
                     P=res["P"], t_f=res["t_f"], t_star=res["t_star"],
                     lam_g=res["lam_g"], u_pos=res["u_pos"], J=res["J"],
                     gate_ok=bool(pick_ok), tag=tag,
                     err_pred=float(pick_gate["err"]),
                     d_land=float(pick_gate["d_land"]),
                     peak_vel_ratio=float(pick_gate["peak_vel_ratio"]),
                     clamp_shift=float(pick_gate["clamp_shift"]),
                     reject=pick_gate["reject"])
        thr_planners.db_merge_save([entry], args.out)   # 쌍 단위로 즉시 반영
        n_done += 1
        n_gate_ok += int(pick_ok)
        el = time.time() - ts_pair
        eta = (time.time() - t0) / max(n_done, 1) * (len(todo) - n)
        print(f"{tag_pair}: {'OK  ' if pick_ok else 'GATE'} J={J:.3f} "
              f"t_f={res['t_f']:.2f}s 수렴 {len(feas)}/{len(cands)} "
              f"착지오차 {pick_gate['err'] * 100:.1f}cm "
              f"vpeak {pick_gate['peak_vel_ratio']:.2f}× "
              f"[{el:.0f}s, ETA {eta / 60:.0f}분]"
              + ("" if pick_ok else f"\n      거부: {pick_gate['reject']}"),
              flush=True)

    dt = time.time() - t0
    print(f"\n=== 완료: {n_done}쌍 저장 (게이트 통과 {n_gate_ok}, "
          f"미통과 {n_done - n_gate_ok}), skip {n_skip}, 실패 {n_fail} "
          f"— {dt / 60:.1f}분 ===")
    print(f"  → {args.out}")
    if n_done and n_gate_ok < n_done:
        print("  ⚠ 게이트 미통과 entry 는 런타임 check_arc 가 그대로 거부한다 "
              "(그 지점은 던지지 않고 건너뜀). GP8_THR_NLP_W1 을 낮춰 다시 빌드하면 "
              "스윙이 느려져 통과율이 올라간다.")


if __name__ == "__main__":
    sys.exit(main())

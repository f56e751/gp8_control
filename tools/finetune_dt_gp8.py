#!/usr/bin/env python3
"""DT sim2real — 수집한 **실기 던지기**로 HER 재라벨 + 파인튜닝.

`tools/collect_real_throws_gp8.py` 가 모은 기록(mem + 실측 착지거리)을 받아
논문 §4.4/§5.4 의 나머지 두 단계를 수행한다:

    ④ HER 재라벨 K=0  — 목표 := **실제 착지 거리** (skills/thr/dt_model/her.py)
    ⑤ 파인튜닝        — 사전학습 가중치에서 이어서 (dt_model/trainer.py)

둘 다 THR/Thr_DT 코드를 그대로 호출한다 (벤더링본, skills/thr/dt_model/).
학습 하이퍼파라미터도 THR `dt_finetune_gp8.finetune` 과 동일:
AdamW lr 1e-4 / weight_decay TrainConfig / 선형 워밍업 5000 / 10 iter × 100 step.

──────────────────────────────────────────────────────────────────────────────
HER K=0 이 무슨 일을 하는가 (해석 주의)
──────────────────────────────────────────────────────────────────────────────
"목표 := 실제 착지점" 이므로 **모든 궤적이 성공(+1)** 이 된다. 즉 이 데이터는
'어디를 맞히는 법' 이 아니라 **'이 액션열은 실제로 이만큼 날아간다'** 를 가르친다
(env 가 모델링하지 않는 레버·지터·공기저항·기하 오차가 여기로 흡수된다).

그래서 **데이터의 착지 거리 분포가 곧 모델이 배우는 사거리 분포**다. 한쪽에
몰리면 모델도 그쪽으로 쏠린다 — 실제로 시뮬 프록시로 만든
`gp8_dt_ft2_real-11.pth` 는 11샘플 중 7개가 1.42 m 이상이라 1.10 m 목표에도
1.67 m 를 던진다. 이 스크립트는 학습 전에 분포를 출력하고, 심하게 치우쳤으면
경고한다 (`--min-bins`).

──────────────────────────────────────────────────────────────────────────────
실행
──────────────────────────────────────────────────────────────────────────────
    cd ~/ros2_ws && source install/setup.bash
    PYTHONPATH=$HOME/ros2_ws/src ~/ros2_ws/src/gp8_control/.venv/bin/python \\
        -m gp8_control.tools.finetune_dt_gp8 \\
        --data ~/ros2_ws/src/gp8_control/data_real/real_throws_2026-08-04.pkl

결과: skills/thr/weights/gp8_dt_real-<N>.pth
검증: tests/run_static_pick_throw_dt.sh --thr-scan --weights <그 경로>
      (파인튜닝 전/후를 같은 표로 비교할 수 있다)
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np


def load_records(paths):
    recs = []
    for p in paths:
        with open(p, "rb") as f:
            r = pickle.load(f)
        recs.extend(r)
        print(f"  {p}: {len(r)}건")
    return recs


def coverage_report(recs, n_bins=5, min_bins=3):
    """착지 분포 리포트 + 치우침 경고. 리턴: 채워진 bin 수."""
    d = np.array([r["d_actual"] for r in recs], float)
    lo, hi = d.min(), d.max()
    print(f"\n실측 착지 거리: n={len(d)}, {lo:.2f} ~ {hi:.2f} m "
          f"(중앙 {np.median(d):.2f}, 평균 {d.mean():.2f})")
    if hi - lo < 1e-6:
        print("  ⚠ 전부 같은 거리 — 사거리를 배울 수 없다")
        return 0
    cnt, edges = np.histogram(d, bins=n_bins)
    for c, a, b in zip(cnt, edges[:-1], edges[1:]):
        bar = "#" * int(c)
        print(f"  {a:.2f}~{b:.2f} m | {c:2d} {bar}")
    filled = int((cnt > 0).sum())
    if filled < min_bins:
        print(f"  ⚠ {n_bins}구간 중 {filled}개만 채워졌다 — 모델이 그 사거리로 "
              f"쏠린다 (gp8_dt_ft2_real-11 이 이 문제로 과투했다). "
              f"목표 거리를 넓혀 더 모으는 것을 권한다.")
    # α 분포도 같이
    a = np.array([r.get("alpha", np.nan) for r in recs], float)
    if np.isfinite(a).any():
        print(f"  게인 α: {np.nanmin(a):.2f} ~ {np.nanmax(a):.2f} "
              f"(평균 {np.nanmean(a):.2f})")
    # env 예측 대비 실제 편차 = reality gap
    e = np.array([r["d_actual"] - r["x_land_env"] for r in recs], float)
    print(f"  env 예측 대비 실제 편차: 평균 {1e3 * e.mean():+.0f} mm "
          f"(표준편차 {1e3 * e.std():.0f} mm) ← 이게 파인튜닝이 흡수할 gap")
    g = np.array([r["d_actual"] - r["d_land_pred"] for r in recs], float)
    print(f"  THR 기하 예측 대비 편차: 평균 {1e3 * g.mean():+.0f} mm "
          f"(표준편차 {1e3 * g.std():.0f} mm)")
    return filled


def main() -> None:
    ap = argparse.ArgumentParser(description="실기 데이터로 DT 파인튜닝 (HER K=0)")
    ap.add_argument("--data", nargs="+", required=True,
                    help="collect_real_throws_gp8.py 산출 pkl (여러 개 가능)")
    ap.add_argument("--weights", default=None,
                    help="사전학습 가중치 (기본: 수집에 쓴 것 = skills/thr/weights/v9)")
    ap.add_argument("--k-her", type=int, default=0,
                    help="HER K (논문 최적 0 = 목표를 실제 착지로 재라벨)")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-bins", type=int, default=3,
                    help="5구간 중 최소 몇 개가 채워져야 하는지 (미만이면 경고)")
    ap.add_argument("--force", action="store_true",
                    help="분포 경고를 무시하고 진행")
    ap.add_argument("--out-dir", default=None,
                    help="기본 skills/thr/weights/")
    args = ap.parse_args()

    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(1)
    except ImportError:
        pass

    import torch

    from gp8_control.skills import thr_planners
    from gp8_control.skills.thr.dt_model import ModelConfig, build_model
    from gp8_control.skills.thr.dt_model.batch import dataset_stats, make_get_batch
    from gp8_control.skills.thr.dt_model.dt_config import TrainConfig
    from gp8_control.skills.thr.dt_model.her import generate_her_memory
    from gp8_control.skills.thr.dt_model.trainer import Trainer

    print("=== 실기 데이터 로드 ===")
    recs = load_records(args.data)
    if not recs:
        sys.exit("데이터 없음")
    filled = coverage_report(recs, min_bins=args.min_bins)
    if filled < args.min_bins and not args.force:
        sys.exit("\n분포가 너무 치우쳤다 — 더 모으거나 --force 로 진행하라.")

    weights = args.weights or recs[0].get("weights")
    if weights and not os.path.isabs(weights):
        weights = os.path.join(os.path.dirname(thr_planners.DT_WEIGHTS), weights)
    weights = weights or thr_planners.DT_WEIGHTS
    out_dir = args.out_dir or os.path.dirname(thr_planners.DT_WEIGHTS)
    print(f"\n사전학습 가중치: {weights}")

    # --- ④ HER 재라벨 (목표 := 실제 착지 거리) ---
    #     reward_sparse 재계산을 위해 학습 env 인스턴스가 필요하다 (arm 만 쓴다).
    model, arm, cfg, torch_mod = thr_planners.load_dt(weights)
    rng = np.random.default_rng(args.seed)
    buf = []
    for r in recs:
        d = float(r["d_actual"])
        tgt = np.array([d, 0.0, 0.0])
        arm.update_target(tgt)
        buf.extend(generate_her_memory(arm, r["mem"], target=tgt,
                                       obj_final_pos=tgt, k=args.k_her, rng=rng))
    print(f"\nHER K={args.k_her} 재라벨 → 학습 궤적 {len(buf)}개")

    # --- ⑤ 파인튜닝 (THR dt_finetune_gp8.finetune 과 동일 절차) ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mc, tc = ModelConfig(), TrainConfig()
    torch.manual_seed(args.seed)

    ft = build_model(mc)                      # 학습이므로 dropout 은 기본값 유지
    ckpt = ft.load(weights)
    ft = ft.to(device)
    tau = ckpt.get("gripper_thresh")

    successes, fails, oc = dataset_stats(buf)
    # oc(그리퍼 닫힘비)는 액션이 정확히 ±1 일 때만 정의된다. DT rollout 의 a_gr 은
    # sigmoid 실수(0~1)라 항상 None 이다 — THR dt_finetune_gp8 도 같다. 참고 통계일 뿐,
    # 학습에는 영향이 없다 (Trainer 의 BCE 는 soft label 을 그대로 받는다).
    oc_s = "n/a (a_gr 이 sigmoid 실수)" if oc is None else f"{oc:.2f}"
    print(f"파인튜닝 데이터: {len(buf)}궤적 (+1 {successes} / -1 {fails}, "
          f"그리퍼 닫힘비 {oc_s}), device={device}")

    get_batch = make_get_batch(buf, mc, tc, device, np.random.default_rng(args.seed))
    opt = torch.optim.AdamW(ft.parameters(), lr=args.lr,
                            weight_decay=tc.weight_decay)
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / args.warmup, 1))
    tr = Trainer(model=ft, optimizer=opt, batch_size=tc.batch_size,
                 get_batch=get_batch, device=device, scheduler=sch,
                 bce_weight=tc.bce_weight, grad_clip=tc.grad_clip)
    for it in range(args.iters):
        loss = tr.train_iteration(args.steps, verbose=False)
        print(f"  iter {it + 1}/{args.iters} | loss {loss:.5f}")

    os.makedirs(out_dir, exist_ok=True)
    name = f"gp8_dt_real-{len(recs)}.pth"
    ft.save(name, folder=out_dir, extra={"gripper_thresh": tau})
    path = os.path.join(out_dir, name)
    print(f"\n→ {path}")
    print("\n검증 (파인튜닝 전/후 같은 표로 비교):")
    print(f"  ./tests/run_static_pick_throw_dt.sh --thr-scan            # 전 (v9)")
    print(f"  ./tests/run_static_pick_throw_dt.sh --thr-scan --weights {path}")
    print("  ※ 실기 적용 전 --plan-only 로 게이트 통과 여부를 반드시 확인할 것.")


if __name__ == "__main__":
    main()

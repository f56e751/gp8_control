#!/usr/bin/env python3
"""Build skills/warm_db.pkl — offline warm-start DB for RobustThrowSkill.

순수 오프라인 수치계산: ROS 노드를 띄우지 않고, 로봇/컨트롤러로 아무 것도
보내지 않는다. skills/throw_nlp.py(스킬이 import하는 바로 그 복사본)로 각
(target × p_start) 조합을 cold multistart로 풀고, 스킬의 _solution_gates와
동일한 게이트를 통과한 최적(J 최소) 해만 entry로 담는다.

파일 포맷 (소비자 robust_throw_skill._load_warm_db 와 일치):
    pickle: dict(params=<_formulation_params와 같은 키/값>, entries=[entry...])
    entry:  dict(target, p_start, P, t_f, t_star, lam_g, u_pos, J, ...)

사용:
    .venv/bin/python tools/build_warm_db.py            # 빌드 + 검증 + 리포트 데이터
    .venv/bin/python tools/build_warm_db.py --dry-run  # 그리드만 출력
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from multiprocessing import get_context
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SKILLS_DIR = REPO / "skills"
# robust_throw_skill.THROW_WARM_DB = skills/warm_db.pkl (skills/ 가 _THR_DIR).
# 스킬 import는 ROS 환경을 요구할 수 있어 경로만 동일 규칙으로 재현한다.
DB_PATH = SKILLS_DIR / "warm_db.pkl"

# ============================================================================
# 입력 좌표 (실측 출처는 BUILD_REPORT.md 참고 — 임의 추정값 금지)
# ============================================================================
# base-frame 착지 목표 [x, y, z] (m). 출처는 BUILD_REPORT.md의 provenance 표 참고:
#  - bin XY (1.10, -0.25): skills/*_skill.py THROW_BIN_X/Y (설정값 — 실측 확인 필요 플래그)
#  - bin z 0.162 = GRASP_Z(0.062, 실측) + 0.10 (bin 조준 컨벤션: tests/suction_lift_debug.py
#    BIN_Z_OFFSET_DEFAULT)
#  - 벨트 fallback z 0.132 = GRASP_Z + DETECTION_OFFSET_AIM(0.07, perception/extrinsics.py)
#  - 벨트 밴드 x = REFERENCE_X_BASE(0.45) ± WORKSPACE_X_ABS(0.2) (캘리브레이션)
TARGETS: list[tuple[float, float, float]] = [
    (1.60, 0.00, 0.0),      # metal bin (운영자 실측 2026-07-20, 바닥 높이)
    (1.20, 0.00, 0.0),      # transparent(페트병) bin (운영자 실측 2026-07-20)
    (0.45, -0.40, 0.132),   # 벨트 중앙선 downstream fallback 조준점
    (0.55, -0.30, 0.132),   # 벨트 밴드 가장자리 downstream fallback 조준점
]
# base-frame 던지기 시작 TCP [x, y, z] (m) — grasp(z=GRASP_Z) + THROW_LIFT 상승.
# intercept 존: x는 벨트 밴드 [0.25, 0.65], y는 reach 원판 [-y_b, +y_b] 대표점.
P_STARTS: list[tuple[float, float, float]] = [
    (0.45, 0.30, 0.162),    # 중앙선, entry-edge 대기 (y_b(0.45)=0.469)
    (0.45, 0.00, 0.162),    # 중앙선 정중앙
    (0.45, -0.30, 0.162),   # 중앙선 downstream intercept
    (0.30, 0.00, 0.162),    # 밴드 근측
    (0.60, 0.00, 0.162),    # 밴드 원측 (y_b(0.60)=0.25)
]

# 스킬 상수 미러 (robust_throw_skill.py 와 동일해야 함)
THROW_LIFT = 0.10          # grasp TCP + 이만큼 상승 = 던지기 시작
LANDING_GATE = 0.03        # 윈도우 dense 착탄오차 상한 (m)
INIT_VARIANTS = (          # robust_throw_skill.INIT_VARIANTS_ROS 미러
    None,
    dict(dq_swing=[0.0, 0.9, 0.3, 0.0, 0.7, 0.0], T0=0.6),
    dict(dq_swing=[0.0, 0.5, 0.9, 0.0, 0.2, 0.0], T0=1.1, chi0=0.7),
    dict(dq_swing=[-0.13, 0.7, 0.6, 0.0, 0.4, 0.0]),
)


# ============================================================================
# Worker (spawn: BLAS 스레드 제한을 import 전에 걸기 위해)
# ============================================================================
def _worker_init() -> None:
    # BLAS 1-thread — 워커간 스레드 경합으로 solve가 ~10× 느려지는 함정 방지.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    sys.path.insert(0, str(SKILLS_DIR))
    import numpy  # noqa: F401  (env 적용된 상태로 첫 import)
    import threadpoolctl
    threadpoolctl.threadpool_limits(1)


def _gates(res: dict, p_target) -> "str | None":
    """robust_throw_skill._solution_gates 와 동일 기준 (독립 재검증)."""
    import numpy as np
    from throw_nlp import _spline_eval
    from throwing import GP8_QD_MAX, fk_pos, jacobian, landing_error

    if res.get("pos_viol_dense", 0.0) > 1e-3:
        return f"pos_viol_dense {res['pos_viol_dense']:.1e}"
    q_of, qd_of, _ = _spline_eval(res["P"], res["t_f"])
    qd_ratio = max(np.max(np.abs(qd_of(t)) / GP8_QD_MAX)
                   for t in np.linspace(0.0, res["t_f"], 400))
    if qd_ratio > 1.0:
        return f"qd ratio {qd_ratio:.2f} > 1"
    rt = res["release_time"]
    errs = []
    for t in res["t_star"] + np.linspace(-rt / 2, rt / 2, 11):
        q, qd = q_of(t), qd_of(t)
        errs.append(landing_error(fk_pos(q), jacobian(q)[0] @ qd, np.asarray(p_target)))
    e_max = float(np.max(errs))
    if not np.isfinite(e_max) or e_max > LANDING_GATE:
        return f"landing window max {e_max * 1e3:.0f}mm > {LANDING_GATE * 1e3:.0f}mm"
    return None


def _solve_one(job: tuple) -> dict:
    """(ti, si, vi, target, p_start) 한 조합 cold solve + 게이트."""
    ti, si, vi, target, p_start = job
    import numpy as np
    import throw_nlp

    t0 = time.time()
    base = dict(ti=ti, si=si, vi=vi, target=list(target), p_start=list(p_start))
    try:
        res = throw_nlp.solve_throw_nlp(
            np.asarray(p_start, float), np.asarray(target, float),
            release_time=throw_nlp.RELEASE_TIME, init=INIT_VARIANTS[vi],
        )
    except (RuntimeError, ValueError, AssertionError) as e:  # 스킬과 동일한 실패 집합
        return dict(**base, ok=False, why=f"solve {type(e).__name__}: {e}",
                    solve_s=time.time() - t0)
    why = _gates(res, target)
    if why is not None:
        return dict(**base, ok=False, why=f"gate: {why}", solve_s=time.time() - t0,
                    J=float(res["J"]), t_f=float(res["t_f"]))
    entry = dict(
        target=np.asarray(target, float), p_start=np.asarray(p_start, float),
        P=res["P"], t_f=res["t_f"], t_star=res["t_star"],
        lam_g=res["lam_g"], u_pos=res["u_pos"], J=res["J"],
        q_start=res["q_start"], variant=vi,
    )
    return dict(**base, ok=True, entry=entry, solve_s=time.time() - t0,
                J=float(res["J"]), t_f=float(res["t_f"]))


def _polish_check(job: tuple) -> dict:
    """entry 하나로 근처 target' warm polish — 시간/게이트 실증 (오프라인)."""
    entry, dxyz = job
    import numpy as np
    import throw_nlp

    target2 = np.asarray(entry["target"], float) + np.asarray(dxyz, float)
    t0 = time.time()
    try:
        res = throw_nlp.solve_throw_nlp(
            np.asarray(entry["p_start"], float), target2,
            release_time=throw_nlp.RELEASE_TIME,
            warm_data={k: entry[k] for k in ("P", "t_f", "t_star", "lam_g", "u_pos")},
        )
    except Exception as e:
        return dict(ok=False, why=f"{type(e).__name__}: {e}",
                    warm_s=time.time() - t0, dxyz=list(dxyz),
                    target=list(entry["target"]))
    why = _gates(res, target2)
    return dict(ok=why is None, why=why, warm_s=time.time() - t0,
                dxyz=list(dxyz), target=list(entry["target"]), J=float(res["J"]))


# ============================================================================
# Main (parent 프로세스는 throw_nlp 를 import하지 않는다 — spawn 워커만)
# ============================================================================
def _formulation_params_standalone() -> dict:
    """robust_throw_skill._formulation_params 복제 (같은 throw_nlp 복사본 기준)."""
    sys.path.insert(0, str(SKILLS_DIR))
    import throw_nlp
    from throwing import GP8_QD_MAX
    return dict(rt=throw_nlp.RELEASE_TIME, w_acc=throw_nlp.W_ACC,
                n_ctrl=throw_nlp.N_CTRL, n_win=throw_nlp.N_WIN,
                q_lo=throw_nlp.Q_LO.tolist(), q_hi=throw_nlp.Q_HI.tolist(),
                qd_max=GP8_QD_MAX.tolist(),
                col=(throw_nlp.COL_R, throw_nlp.COL_H),
                pos_mode=getattr(throw_nlp, "POS_LIMIT_MODE", "colloc"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(DB_PATH))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json-log", default=str(REPO / "tools" / "warm_db_build_log.json"))
    args = ap.parse_args()

    if not TARGETS or not P_STARTS:
        sys.exit("TARGETS/P_STARTS 가 비어 있음 — 실측 좌표를 채운 뒤 실행하라.")

    jobs = [(ti, si, vi, t, p)
            for ti, t in enumerate(TARGETS)
            for si, p in enumerate(P_STARTS)
            for vi in range(len(INIT_VARIANTS))]
    print(f"grid: {len(TARGETS)} targets x {len(P_STARTS)} starts x "
          f"{len(INIT_VARIANTS)} variants = {len(jobs)} solves, "
          f"{args.workers} workers")
    for ti, t in enumerate(TARGETS):
        print(f"  target[{ti}] = {t}")
    for si, p in enumerate(P_STARTS):
        print(f"  p_start[{si}] = {p}")
    if args.dry_run:
        return

    t_all = time.time()
    ctx = get_context("spawn")
    with ctx.Pool(args.workers, initializer=_worker_init) as pool:
        results = pool.map(_solve_one, jobs)

    # (target, p_start)별 게이트 통과 최소-J 해만 entry로
    best: dict[tuple, dict] = {}
    rejected = []
    for r in results:
        if r["ok"]:
            key = (r["ti"], r["si"])
            if key not in best or r["J"] < best[key]["J"]:
                best[key] = r
        else:
            rejected.append(r)
    entries = [best[k]["entry"] for k in sorted(best)]

    params = _formulation_params_standalone()
    out_path = Path(args.out)
    with open(out_path, "wb") as f:
        pickle.dump(dict(params=params, entries=entries), f)
    print(f"\nwrote {out_path}: {len(entries)} entries "
          f"({len(rejected)} attempts rejected) in {time.time() - t_all:.0f}s")

    # ---- 검증 1: 로드 라운드트립 (_load_warm_db 로직 복제: params subset 비교) ----
    with open(out_path, "rb") as f:
        db = pickle.load(f)
    db_params = db.get("params", {})
    assert all(db_params.get(k) == v for k, v in params.items()), \
        "roundtrip params mismatch — consumer would discard this DB"
    assert len(db["entries"]) == len(entries)
    print(f"roundtrip OK: params subset-match, {len(db['entries'])} entries")

    # ---- 검증 2: polish 실증 (entry 몇 개, target 2~5cm 이동, warm vs cold) ----
    polish_jobs = [(e, d) for e in entries[:3]
                   for d in ([0.03, -0.02, 0.0], [-0.02, 0.04, 0.01])]
    polish = []
    if polish_jobs:
        with ctx.Pool(min(args.workers, len(polish_jobs)),
                      initializer=_worker_init) as pool:
            polish = pool.map(_polish_check, polish_jobs)
        for p in polish:
            tag = "OK" if p["ok"] else f"FAIL({p['why']})"
            print(f"polish {tag}: target={p['target']} dxyz={p['dxyz']} "
                  f"warm={p['warm_s']:.1f}s")

    # ---- 리포트 재료 JSON ----
    def _san(r):
        return {k: v for k, v in r.items() if k != "entry"}
    with open(args.json_log, "w") as f:
        json.dump(dict(
            targets=[list(t) for t in TARGETS],
            p_starts=[list(p) for p in P_STARTS],
            params={k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in params.items()},
            solves=[_san(r) for r in results],
            polish=polish,
            n_entries=len(entries), total_s=round(time.time() - t_all, 1),
        ), f, indent=1, default=str)
    print(f"log -> {args.json_log}")


if __name__ == "__main__":
    main()

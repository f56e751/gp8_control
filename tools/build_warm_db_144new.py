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
    .venv/bin/python tools/build_warm_db.py            # 축적(append) 빌드 + 검증
    .venv/bin/python tools/build_warm_db.py --rebuild  # 전 조합 강제 재계산
    .venv/bin/python tools/build_warm_db.py --dry-run  # 그리드/생략 조합만 출력

기본은 **축적 모드**: 기존 warm_db.pkl의 공식화가 현재와 같으면 entry를 유지하고
TARGETS/P_STARTS에 새로 추가된 (target, p_start) 조합만 풀어 병합한다. 공식화
(W_ACC/윈도우/한계/치수)가 바뀐 경우에만 전체 재계산이 필요하며 자동 감지된다.
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
# PAIRED 모드 (2026-07-24 사용자 "shuffle 안 할 테니 이 쌍 순서대로 최적화").
# run_static_pick_throw.sh 가 --shuffle-points 없이 돌면 point[i]→TARGETS[i] 가
# 고정 쌍이라, 각 target을 '그 target에 실제로 쓰이는 던지기 시작 자세'로 최적화할
# 수 있다. True면 TARGETS[i]와 P_STARTS[i]를 1:1로 짝지어 len(TARGETS)개 조합만
# 푼다(교차곱 아님). 이러면 런타임 lift 자세와 DB start가 일치해 polish가 즉시
# 수렴하고(먼 start = 8~14s·나쁜 basin 문제 해소) Cartesian 게이트 기각도 사라진다.
PAIRED = False
TARGETS: list[tuple[float, float, float]] = [
    # 새 타겟 (run_static_pick_throw.sh, 2026-07-28 사용자, x=1.10/1.35/1.6, z=-0.08).
    (1.10, 0.225, -0.08), (1.10, 0.075, -0.08),
    (1.10, -0.075, -0.08), (1.10, -0.225, -0.08),
    (1.35, 0.225, -0.08), (1.35, 0.075, -0.08),
    (1.35, -0.075, -0.08), (1.35, -0.225, -0.08),
    (1.60, 0.225, -0.08), (1.60, 0.075, -0.08),
    (1.60, -0.075, -0.08), (1.60, -0.225, -0.08),
]
# base-frame 던지기 시작 TCP [x, y, z] (m). PAIRED 모드에서는 TARGETS와 1:1 대응.
# 값 = run_static_pick_throw.sh POINTS[i] 의 (x, y) + 던지기 시작 z. 시작 z는
# static_pick_throw 가 쓰는 것과 동일하게 계산: place(=press_z + PLACE_ABOVE_PRESS
# 0.05) + THROW_LIFT 0.10. POINTS 의 press_z = 0.02 → 시작 z = 0.02+0.05+0.10 = 0.17.
# 이 좌표가 plan_nlp_throw 의 p_lift 와 같아야 polish 가 그 자세에서 바로 출발한다.
P_STARTS: list[tuple[float, float, float]] = [
    # press z=0.04 → lift z=0.19, TARGETS 와 1:1 (POINTS 순서 그대로)
    (0.40, 0.30, 0.19), (0.40, 0.20, 0.19),
    (0.40, 0.10, 0.19), (0.40, 0.00, 0.19),
    (0.40, -0.10, 0.19), (0.40, -0.20, 0.19),
    (0.50, 0.30, 0.19), (0.50, 0.20, 0.19),
    (0.50, 0.10, 0.19), (0.50, 0.00, 0.19),
    (0.50, -0.10, 0.19), (0.50, -0.20, 0.19),
]

# 스킬 상수 미러 (robust_throw_skill.py 와 동일해야 함)
THROW_LIFT = 0.10          # grasp TCP + 이만큼 상승 = 던지기 시작
LANDING_GATE = 0.03        # 윈도우 dense 착탄오차 상한 (m)
# Cartesian 안전 엔벨로프 — robust_throw_skill.MIN_TCP_X / MIN_TCP_Z 미러.
# 스윙 아크의 TCP가 기둥/베이스(x ≤ 0.20)나 바닥/벨트(z ≤ 0.04)로 들어가는 해는
# 실기 dispatch 게이트에서 어차피 기각되므로 DB에 넣지 않는다. 이 검사가 없으면
# 바닥을 파는 해가 entry로 저장되고, 런타임이 그걸 warm start로 써서 다시 바닥을
# 파는 해로 수렴 → dispatch 기각, 이 악순환이 생긴다 (2026-07-23 실측: 12건 중 8건).
MIN_TCP_X = -99.0
MIN_TCP_Z = -99.0
# TCP z 상한 (2026-07-25 사용자): 스윙 아크의 TCP z 가 이 높이를 넘는 해는 제외하고
# 통과분 중 min-J(top-1)를 저장. 팔을 너무 높이 드는 궤적 배제.
MAX_TCP_Z = 99.0
# 오프라인 multistart 초기해 — 런타임(robust_throw_skill.INIT_VARIANTS_ROS)보다
# 의도적으로 넓게. 빌더는 (target,p_start)별로 게이트 통과한 것 중 min-J 해만
# 저장하므로(main의 best dict) 변형이 많을수록 더 나은 basin을 고를 확률이 커진다
# (2026-07-23 사용자 "최대한 다양한 초기값으로 최적화 뒤 best 저장"). dq_swing =
# [S,L,U,R,B,T] 오프셋(플래너 규약); 지배축은 L(어깨)·U(elbow)·B(wrist pitch),
# S(yaw)는 조준. 축 강조/스윙 진폭/시간(T0)/release비(chi0)/yaw부호를 교차.
def _make_init_variants(n: int = 24, seed: int = 0) -> tuple:
    """넓은 multistart 초기해 집합 (격자에서 시드 고정 샘플링).

    2026-07-23 사용자 "초기값 많이 해보고 최적값 고르는 넓은 탐색". 빌더는
    (target, p_start)별로 게이트를 통과한 것 중 min-J 해만 저장하므로(main의 best
    dict), 변형이 많을수록 더 나은 basin을 고를 확률이 커진다.

    **오프셋 범위는 새 관절 한계에 맞춰 재설계했다.** q_L ≤ +45°인데 lift 자세의
    q_L이 이미 22~34°라 L 여유가 0.19~0.40 rad뿐 — 구 변형들의 L 오프셋(0.4~1.1)은
    전부 한계를 넘어 무의미하다. 그래서 스윙을 **U(elbow)·B(wrist) 주도**로 짠다
    (여유: U ~0.73~1.05 rad, B는 상한 +135° 유지라 ~1.5 rad로 넉넉 — wrist snap을
    살리는 쪽이 정확도에 유리하다). dq_swing = [S,L,U,R,B,T] 오프셋(플래너 규약).

    시드 고정이라 spawn 워커들이 같은 목록을 재현한다 (vi 인덱스 일관성 필수).
    """
    import random
    grid = [dict(dq_swing=[sy, l, u, 0.0, b, 0.0], T0=t0, chi0=c0)
            for sy in (-0.13, 0.0, 0.13)      # 조준 yaw (±/중립)
            for l in (0.05, 0.15, 0.28)       # 어깨: 한계 여유 안에서만
            for u in (0.35, 0.60, 0.90)       # elbow: 주 구동축
            for b in (0.20, 0.55, 0.90)       # wrist pitch: 상한 여유가 커 넓게
            for t0 in (0.6, 0.9, 1.2)         # 스윙 소요시간 초기값
            for c0 in (0.45, 0.60)]           # release 시점 비율
    random.Random(seed).shuffle(grid)
    return (None,) + tuple(grid[:max(0, n - 1)])


# 54 = 넓은 탐색 (2026-07-23 사용자 지정). 486점 격자에서 시드 고정 샘플 53개 + None.
# 총시간 ≈ 조합수 × 변형수 × (solve 1건 시간) ÷ 워커 — 변형 수가 유일한 실질 레버다
# (워커는 이미 최대, expand:True로 이미 2× 가속, GPU 백엔드 없음). 조합당 54개를
# 모두 풀고 그중 게이트 통과 min-J 해 하나만 저장하므로, 넓힐수록 해 품질은 오르고
# 시간은 선형으로 는다. 이력: 4 -> 9 -> 24 -> 12 -> 54.
INIT_VARIANTS = _make_init_variants(128)


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
    from throwing import GP8_QD_MAX, fk_pos, landing_error, launch_state

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
        # 발사점은 NLP와 동일하게 launch_state (TCP + 로드축 GRIP_OFF, ω×r 포함)
        # — 스킬의 _solution_gates 와 동일 기준. bare TCP로 재검증하면 2cm/ω×r
        # 만큼 어긋나 정상 해가 착탄 게이트에 잘못 걸린다.
        p_eff, v_eff = launch_state(q_of(t), qd_of(t))
        errs.append(landing_error(p_eff, v_eff, np.asarray(p_target)))
    e_max = float(np.max(errs))
    if not np.isfinite(e_max) or e_max > LANDING_GATE:
        return f"landing window max {e_max * 1e3:.0f}mm > {LANDING_GATE * 1e3:.0f}mm"
    # Cartesian 엔벨로프: 스윙 아크 전 구간의 TCP (플래너 규약 fk_pos — 로봇 FK와
    # 0.000mm 일치 검증됨). 아크는 lift 자세(z≈0.162)에서 출발하므로 예외 구간 없이
    # 전 샘플을 본다. 위반 사유 문자열은 'cartesian' 접두어로 main()이 집계한다.
    T = np.array([fk_pos(q_of(t)) for t in np.linspace(0.0, res["t_f"], 300)])
    x_min, z_min, z_max = float(T[:, 0].min()), float(T[:, 2].min()), float(T[:, 2].max())
    if x_min <= MIN_TCP_X or z_min <= MIN_TCP_Z or z_max > MAX_TCP_Z:
        return (f"cartesian x_min={x_min:+.4f} z_min={z_min:+.4f} z_max={z_max:+.4f} "
                f"(한계 x>{MIN_TCP_X:.2f}, {MIN_TCP_Z:.2f}<z<{MAX_TCP_Z:.2f})")
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
    """robust_throw_skill._formulation_params 복제 (같은 throw_nlp 복사본 기준).

    키를 바꾸면 로더(robust_throw_skill._formulation_params)와 반드시 함께 바꿀 것.
    """
    sys.path.insert(0, str(SKILLS_DIR))
    import throw_nlp
    from throwing import GP8_DIMS, GP8_QD_MAX
    return dict(rt=throw_nlp.RELEASE_TIME, w_acc=throw_nlp.W_ACC,
                n_ctrl=throw_nlp.N_CTRL, n_win=throw_nlp.N_WIN,
                q_lo=throw_nlp.Q_LO.tolist(), q_hi=throw_nlp.Q_HI.tolist(),
                qd_max=GP8_QD_MAX.tolist(),
                col=(throw_nlp.COL_R, throw_nlp.COL_H),
                pos_mode=getattr(throw_nlp, "POS_LIMIT_MODE", "colloc"),
                # 2026-07-21 세분화 — 로더와 동일 (목적함수 가중치/가속도/t_f
                # 범위/차수/치수까지 전부 유효성 키).
                w1=throw_nlp.W1, w2=throw_nlp.W2, w_sens=throw_nlp.W_SENS,
                qdd_lim=throw_nlp.QDD_LIM.tolist(),
                t_bounds=tuple(throw_nlp.T_BOUNDS),
                degree=throw_nlp.DEGREE,
                dims=dict(GP8_DIMS),
                # flight 모델 마커 — 로더(robust_throw_skill._formulation_params)와
                # 동일 키. 항력/RK4·발사점·τ 변수가 공식화를 바꾸면 entry 무효.
                flight=getattr(throw_nlp, "FLIGHT_MODEL", "parabola"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(DB_PATH))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rebuild", action="store_true",
                    help="기존 DB를 무시하고 전 조합 재계산 (기본은 축적/append)")
    ap.add_argument("--refresh", action="store_true",
                    help="축적된 DB의 모든 (target,p_start) 쌍을 현재 init/formulation "
                         "으로 재최적화 (TARGETS/P_STARTS 무시, 축적 집합 보존·궤적 갱신)")
    ap.add_argument("--json-log", default=str(REPO / "tools" / "warm_db_build_log.json"))
    args = ap.parse_args()

    if not TARGETS or not P_STARTS:
        sys.exit("TARGETS/P_STARTS 가 비어 있음 — 실측 좌표를 채운 뒤 실행하라.")

    params = _formulation_params_standalone()
    out_path = Path(args.out)

    # ---- 축적(append) 모드 (기본): 같은 공식화면 기존 entry에 이어붙인다 ----
    # entry는 공식화 파라미터(W_ACC/윈도우/관절한계/기둥/치수 반영 해)가 같을
    # 때만 warm start로 유효하다. 공식화가 다르면 기존 entry는 폐기하고 전체
    # 재계산; 같으면 이미 커버된 (target, p_start) 조합은 건너뛰고 새 조합만
    # 풀어 병합한다 — TARGETS/P_STARTS에 좌표를 추가하고 재실행하면 그만큼만 돈다.
    def _key(t, p):
        return (tuple(round(float(v), 3) for v in t),
                tuple(round(float(v), 3) for v in p))

    existing: list = []
    if args.refresh:
        # --- refresh: 축적된 DB 의 '모든 쌍'을 현재 54 init/formulation 으로 재최적화 ---
        # (2026-07-24 사용자 "축적 다하면 54 초기값으로 축적된 쌍들의 궤적 값을
        # 업데이트"). TARGETS/P_STARTS 리스트는 무시하고, DB 에 이미 쌓인 (target,
        # p_start) 쌍 전부를 새로 푼다 — 기존 entry 는 폐기하고 결과로 교체. 일반
        # --rebuild 는 하드코딩된 현재 리스트만 다시 풀어 다른 config 의 축적분을
        # 잃지만, 이 모드는 축적 집합을 통째로 보존하며 궤적만 갱신한다.
        if not out_path.exists():
            sys.exit("--refresh: 기존 DB 없음 — 먼저 축적(append)하라")
        with open(out_path, "rb") as f:
            old = pickle.load(f)
        oe = old.get("entries", [])
        combos = [(i, i, e["target"], e["p_start"]) for i, e in enumerate(oe)]
        have = set()                       # 전부 재계산
        n_total = len(combos)
        print(f"refresh: 축적 DB {len(oe)} 쌍을 {len(INIT_VARIANTS)} init best-of 로 "
              f"재최적화 (TARGETS/P_STARTS 무시, 공식화 params 현재값으로 기록)")
    else:
        if not args.rebuild and out_path.exists():
            try:
                with open(out_path, "rb") as f:
                    old = pickle.load(f)
                if all(old.get("params", {}).get(k) == v for k, v in params.items()):
                    existing = old.get("entries", [])
                    print(f"append: 기존 DB {len(existing)} entries 유지 (공식화 일치)")
                else:
                    print("append 불가: 기존 DB 공식화 불일치 — 전체 재계산 (기존 폐기)")
            except Exception as e:  # 손상 파일 → 새로 만든다
                print(f"기존 DB 로드 실패({type(e).__name__}) — 전체 재계산")

        have = {_key(e["target"], e["p_start"]) for e in existing}
        if PAIRED:
            # TARGETS[i] ↔ P_STARTS[i] 1:1 (교차곱 아님). ti/si는 로그·best 키 용도로
            # 동일 인덱스를 쓴다 (한 쌍당 entry 1개).
            if len(TARGETS) != len(P_STARTS):
                sys.exit(f"PAIRED 모드: len(TARGETS)={len(TARGETS)} != "
                         f"len(P_STARTS)={len(P_STARTS)} — 1:1 이어야 함")
            combos = [(i, i, t, p) for i, (t, p) in enumerate(zip(TARGETS, P_STARTS))]
            n_total = len(TARGETS)
        else:
            combos = [(ti, si, t, p)
                      for ti, t in enumerate(TARGETS)
                      for si, p in enumerate(P_STARTS)]
            n_total = len(TARGETS) * len(P_STARTS)
    new_combos = [(ti, si, t, p) for (ti, si, t, p) in combos
                  if _key(t, p) not in have]
    jobs = [(ti, si, vi, t, p)
            for (ti, si, t, p) in new_combos
            for vi in range(len(INIT_VARIANTS))]
    print(f"grid: {'PAIRED ' if PAIRED else ''}{n_total}조합 — "
          f"신규 {len(new_combos)}조합 x {len(INIT_VARIANTS)} variants = "
          f"{len(jobs)} solves (기존 커버 {n_total - len(new_combos)}조합 생략), "
          f"{args.workers} workers")
    for ti, t in enumerate(TARGETS):
        print(f"  target[{ti}] = {t}")
    for si, p in enumerate(P_STARTS):
        print(f"  p_start[{si}] = {p}")
    if args.dry_run:
        return
    if not jobs:
        print("신규 조합 없음 — DB 변경 없이 종료")
        return

    t_all = time.time()
    ctx = get_context("spawn")
    from collections import defaultdict

    def _write(entries_now):
        # 원자적 쓰기: 쓰는 도중 죽어도(OOM 등) 기존 DB가 깨지지 않게 tmp→rename.
        tmp_path = out_path.with_suffix(".pkl.tmp")
        with open(tmp_path, "wb") as f:
            pickle.dump(dict(params=params, entries=entries_now), f)
        os.replace(tmp_path, out_path)

    # 증분 저장 (2026-07-24 사용자 "1개 궤적 축적할 때마다 바로바로 db에 저장"):
    # imap_unordered 로 결과를 스트리밍하고, 한 쌍(=(ti,si))의 전 variant 가
    # 끝나는 즉시 게이트 통과 최소-J 해를 골라 entry 에 붙이고 DB 를 원자적으로
    # 다시 쓴다. 중간에 끊겨도(끊김/OOM) 그때까지 확정된 쌍의 궤적은 보존된다.
    n_per = len(INIT_VARIANTS)
    pending: dict = defaultdict(list)
    best: dict[tuple, dict] = {}
    rejected: list = []
    results: list = []
    entries = list(existing)          # 기존(append 유지분) 위에 하나씩 이어붙임
    done = 0
    with ctx.Pool(args.workers, initializer=_worker_init) as pool:
        for r in pool.imap_unordered(_solve_one, jobs):
            results.append(r)
            key = (r["ti"], r["si"])
            if r["ok"]:
                if key not in best or r["J"] < best[key]["J"]:
                    best[key] = r
            else:
                rejected.append(r)
            pending[key].append(r)
            if len(pending[key]) == n_per:        # 이 쌍의 전 variant 완료
                done += 1
                if key in best:
                    entries.append(best[key]["entry"])
                    _write(entries)               # ★ 한 쌍 확정 즉시 원자적 저장
                    tag = f"J={best[key]['J']:.2f}"
                else:
                    tag = "entry 없음 (전 변형 위반/실패)"
                del pending[key]
                print(f"  [{done}/{len(new_combos)}] combo{key} → entry {len(entries)}개 ({tag})")
    _write(entries)                   # 최종 일관성 보장 (전멸/마지막 실패 케이스 포함)
    print(f"\nwrote {out_path}: {len(entries)} entries "
          f"(신규 {len(best)}, 기존 유지 {len(existing)}, "
          f"{len(rejected)} attempts rejected) in {time.time() - t_all:.0f}s")

    # ---- Cartesian 엔벨로프 위반 보고 (2026-07-23 사용자 요청) ----
    # 바닥/기둥을 침범한 해가 몇 건이나 나왔는지, 그리고 '전 변형이 위반해서 결국
    # entry를 못 만든' 조합이 어디인지 보고한다 — 후자가 실기에서 던지지 못하는 조합.
    cart = [r for r in rejected if str(r.get("why", "")).startswith("gate: cartesian")]
    if cart:
        import re as _re
        zs, xs = [], []
        for r in cart:
            m = _re.search(r"x_min=([-+0-9.]+) z_min=([-+0-9.]+)", r["why"])
            if m:
                xs.append(float(m.group(1))); zs.append(float(m.group(2)))
        per_combo: dict = {}
        for r in cart:
            per_combo.setdefault((r["ti"], r["si"]), 0)
            per_combo[(r["ti"], r["si"])] += 1
        no_entry = [k for k in per_combo if k not in best]
        print(f"\n[Cartesian 엔벨로프 위반] {len(cart)}/{len(jobs)} solves "
              f"(x≤{MIN_TCP_X} 또는 z≤{MIN_TCP_Z}) — DB에서 제외됨")
        if zs:
            print(f"  최저 TCP z = {min(zs):+.4f} m,  최소 TCP x = {min(xs):+.4f} m")
        print(f"  위반이 나온 (target,start) 조합: {len(per_combo)}/{len(new_combos)}")
        if no_entry:
            # refresh 모드에서는 ti 가 DB 쌍 인덱스라 TARGETS[ti] 가 안 맞으므로,
            # 실제 조합 좌표를 combos 에서 찾아 보고한다.
            combo_xy = {(ti, si): (t, p) for (ti, si, t, p) in combos}
            print(f"  ** 전 변형이 위반해 entry 생성 실패: {len(no_entry)}조합 **")
            for ti, si in sorted(no_entry):
                t, p = combo_xy[(ti, si)]
                tt = tuple(round(float(v), 3) for v in t)
                pp = tuple(round(float(v), 3) for v in p)
                print(f"     target={tt}  start={pp}")
        else:
            print("  (모든 조합이 위반하지 않는 대안 해를 찾아 entry 확보)")
    else:
        print("\n[Cartesian 엔벨로프] 위반 solve 없음")

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

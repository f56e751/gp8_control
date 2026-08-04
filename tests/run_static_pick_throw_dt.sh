#!/usr/bin/env bash
# 정적(멈춘 벨트) 연속 pick-and-throw — **DT (Decision Transformer, THR GP8 rig 학습)**
#
# THR dt_planner 의 로봇판 — Monastirsky/Azulay/Sintov RA-L 2023 재현을
# GP8 rig 에서 직접 학습한 가중치. 10 Hz 자기회귀로 (ω_L, ω_U, ω_B, a_gr) 를
# 생성하고 a_gr ≤ τ 스텝에서 릴리즈한다. 목표는 거리 d_g 하나로 조건화되고
# 방향은 J1(S) 이 해석적으로 잡는다 [논문 §5.1].
# THR 시뮬(보간 ON, 12던지기): ideal 12/12 평균 51 mm / real 11/12 평균 84 mm.
#
# ⚠ 릴리즈 윈도우가 없다 (nlp 은 ±50 ms). 밸브 지터가 그대로 착지 오차가
#   되므로 로그의 `민감도 N mm/10 ms` 를 보고 GP8_RELEASE_LEAD 를 맞출 것.
# ⚠ 학습 범위는 d_g ∈ [0.6, 2.0] m (dt_gp8_env.DG_MIN/MAX). 밖이면 클립된다.
#
# 사용:
#   ./run_static_pick_throw_dt.sh --thr-scan    # ★ 먼저 이걸로 실현 가능성 확인 (로봇 무명령)
#   ./run_static_pick_throw_dt.sh --plan-only   # 계획+궤적조립 검증 (driver 불필요)
#   ./run_static_pick_throw_dt.sh               # 실기 (driver 자동 기동)
#   ./run_static_pick_throw_dt.sh --target "1.35,0,-0.08"   # CLI 가 아래 배열을 덮어씀
#
# 공통 기계(빌드/driver/env)는 run_static_pick_throw_thr.sh 에 있다. 여기서는
# 지점·목표·모델별 옵션만 정한다.
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# 물체 지점 목록 (base frame "x,y,z" [m])
#   z = 프레스가 내려가는 최종 TCP 높이. 물체(place)는 z + 0.05 m 위에 놓인 것으로
#   본다 (static_pick_throw.py 의 PLACE_ABOVE_PRESS). "x,y" 로 z 생략 시 GRASP_Z.
# dt 는 집는 위치가 던지기에 영향을 주지 않는다 — 스윙이 항상 학습 시
#   home 자세에서 출발하므로 지점은 '무엇을 집을지'만 정한다.
# ============================================================================
POINTS=(
    "0.40,0.30,0.04"
    "0.40,0.20,0.04"
    "0.40,0.10,0.04"
    "0.40,0.0,0.04"
    "0.40,-0.10,0.04"
    "0.40,-0.20,0.04"
    
    "0.50,0.30,0.04"
    "0.50,0.20,0.04"
    "0.50,0.10,0.04"
    "0.50,0.0,0.04"
    "0.50,-0.10,0.04"
    "0.50,-0.20,0.04"
)

# ============================================================================
# 던지기 착지 목표 ("x,y,z" [m]) — 1개면 전 지점 공통, 여러 개면 지점 수와 일치.
# 기본값은 실기 bin 좌표 (구 run_static_pick_throw.sh 와 동일: 1.10/1.35/1.60,
# z=−0.08 = bin 바닥). dt 는 그중 수평거리 d_g=hypot(x,y) 와 방향만 쓰고 z 는 착지 예측면으로만 쓴다.
# ============================================================================
TARGETS=(
    "1.10,0.225,-0.08"
    "1.10,0.075,-0.08"
    "1.10,-0.075,-0.08"
    "1.10,-0.225,-0.08"

    "1.35,0.225,-0.08"
    "1.35,0.075,-0.08"
    "1.35,-0.075,-0.08"
    "1.35,-0.225,-0.08"

    "1.60,0.225,-0.08"
    "1.60,0.075,-0.08"
    "1.60,-0.075,-0.08"
    "1.60,-0.225,-0.08"
)

# ── 실측 bin 좌표 (2026-08-04 사용자 측정) — warm DB 재빌드 후 이걸로 교체 ──
# 위 좌표는 skills/thr/warm_db_thr.pkl 의 144 entry 와 **정확히 일치**해서 nlp 이
# exact warm hit 로 0.4~1초에 계획된다. 아래 실측 좌표는 그 DB 와 83~100 mm
# 어긋나 polish 가 2~3초로 느려지고 게이트 통과율도 떨어진다 (1.20 m 거부 실측).
# tools/build_warm_db_thr.py --from-sh <이 파일> 로 아래 좌표용 DB 를 채운 뒤
# 위/아래 블록을 맞바꾸면 된다 (DB 는 두 좌표계를 같은 config 에 함께 담는다).
# TARGETS=(
#     "1.20,0.225,0.02"
#     "1.20,0.075,0.02"
#     "1.20,-0.075,0.02"
#     "1.20,-0.225,0.02"
#
#     "1.441,0.225,0.085"
#     "1.441,0.075,0.085"
#     "1.441,-0.075,0.085"
#     "1.441,-0.225,0.085"
#
#     "1.683,0.225,0.15"
#     "1.683,0.075,0.15"
#     "1.683,-0.075,0.15"
#     "1.683,-0.225,0.15"
# )

# --confirm-throw : 흡착+유지 후 Enter 를 눌러야 던진다 (q 입력 시 그 사이클 생략)
# --vel-scale     : 던지기 외 이동 속도 (계획 아크와 감속 꼬리는 영향 없음)
# --shuffle-points: 지점 방문 순서 무작위 (TARGETS 순서는 고정)
# --no-confirm    : 사이클마다 Enter 확인 생략
EXTRA_ARGS_STR="--confirm-throw --vel-scale 0.2"

# DT 체크포인트 (skills/thr/weights/):
#   gp8_dt_best_v9.pth     2026-08-03 17:04  ← 기본값. bin 1.10/1.35/1.60 m 에서
#                          예측 착지 오차 6.2 / 7.9 / 1.2 cm, 전부 게이트 통과.
#   gp8_dt_best.pth        2026-07-30 (THR dt_planner 의 기본값) — **쓰지 말 것**.
#                          1.10 은 0.21 m 밖에 못 던지고 1.35/1.60 은 미-release.
#                          tool 0.240 으로 기하가 바뀌기 전 학습분이다.
#   gp8_dt_ft2_real-11.pth 2026-08-03 16:17 (실기 11회 파인튜닝) — 너무 세게 던진다
#                          (1.10 목표에 1.67 m). 파인튜닝 데이터 재확인 필요.
export GP8_THR_DT_WEIGHTS="${GP8_THR_DT_WEIGHTS:-$HOME/ros2_ws/src/gp8_control/skills/thr/weights/gp8_dt_best_v9.pth}"
# 조건화 목표 리턴 R̂ (성공한 던지기 = 1) [논문 §4.3]
export GP8_THR_DT_RETURN="${GP8_THR_DT_RETURN:-1.0}"

join_semi() { local IFS=";"; echo "$*"; }
export THR_MODEL="dt"
export POINTS_ARG="$(join_semi "${POINTS[@]}")"
export TARGET_ARG="$(join_semi "${TARGETS[@]}")"
export EXTRA_ARGS_STR
exec "$HERE/run_static_pick_throw_thr.sh" "$@"

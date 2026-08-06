#!/usr/bin/env bash
# 정적(멈춘 벨트) 연속 pick-and-throw — **NLP (CasADi/IPOPT B-spline 최적화)**
#
# THR nlp_planner 의 로봇판. 릴리즈 **윈도우** ±50 ms 전 구간에서 착탄
# 정확도를 목적함수(W_ACC)로 강제하므로 밸브 지터에 가장 강하다.
# THR 시뮬(보간 ON, 12던지기): ideal 12/12 평균 26 mm / real 12/12 평균 32 mm.
#
# warm DB: skills/thr/warm_db_thr.pkl (144 entry, affb25e 재빌드) 이 현재 공식화와
#   일치해 아래 기본 POINTS/TARGETS 는 **정확 warm hit** 로 지점당 0.4~1초에
#   계획된다. DB 에 없는 지점/타겟은 cold multistart (수 초~수십 초) — 어느 쪽이든
#   로봇이 움직이기 전에 끝난다. (구 2026-08-03 "DB 전부 무효" 경고는 재빌드로 해소)
#
# 사용:
#   ./run_static_pick_throw_nlp.sh --thr-scan    # ★ 먼저 이걸로 실현 가능성 확인 (로봇 무명령)
#   ./run_static_pick_throw_nlp.sh --plan-only   # 계획+궤적조립 검증 (driver 불필요)
#   ./run_static_pick_throw_nlp.sh               # 실기 (driver 자동 기동)
#   ./run_static_pick_throw_nlp.sh --target "1.35,0,-0.08"   # CLI 가 아래 배열을 덮어씀
#
# 공통 기계(빌드/driver/env)는 run_static_pick_throw_thr.sh 에 있다. 여기서는
# 지점·목표·모델별 옵션만 정한다.
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# 물체 지점 목록 (base frame "x,y,z" [m])
#   z = 프레스가 내려가는 최종 TCP 높이. 물체(place)는 z + 0.05 m 위에 놓인 것으로
#   본다 (static_pick_throw.py 의 PLACE_ABOVE_PRESS). "x,y" 로 z 생략 시 GRASP_Z.
#   24 cm TCP의 0.02 m는 구 22 cm TCP의 0.04 m와 같은 실제 관절 자세다.
# nlp 은 집는 위치(p_start)가 궤적에 실제로 영향을 준다.
# ============================================================================
POINTS=(
    "0.40,0.30,0.02"
    "0.40,0.20,0.02"
    "0.40,0.10,0.02"
    "0.40,0.0,0.02"
    "0.40,-0.10,0.02"
    "0.40,-0.20,0.02"

    "0.50,0.30,0.02"
    "0.50,0.20,0.02"
    "0.50,0.10,0.02"
    "0.50,0.0,0.02"
    "0.50,-0.10,0.02"
    "0.50,-0.20,0.02"
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

# NLP 자체의 카타시안(기둥 회피) 제약 — 실기 기본 ON.
# 끄면(THR 시뮬 기본값) 스윙이 바닥/기둥을 파고드는 해가 나와 dispatch 게이트에서
# 걸린다 (실측: 1.60 m 타겟 z_min=−0.021 m). warm DB 유효성 키라 on/off 가 서로 다른
# config 로 갈리므로, 바꾸면 그쪽 DB 를 따로 빌드해야 한다.
export GP8_THR_NLP_CART="${GP8_THR_NLP_CART:-1}"
# 순차 multistart 예산 (고정 초기해 6개가 전멸했을 때만 쓴다).
export GP8_THR_NLP_MAX_RANDOM="${GP8_THR_NLP_MAX_RANDOM:-32}"

join_semi() { local IFS=";"; echo "$*"; }
export THR_MODEL="nlp"
export POINTS_ARG="$(join_semi "${POINTS[@]}")"
export TARGET_ARG="$(join_semi "${TARGETS[@]}")"
export EXTRA_ARGS_STR
exec "$HERE/run_static_pick_throw_thr.sh" "$@"

#!/usr/bin/env bash
# 정적(멈춘 벨트) 연속 pick-and-throw — **NLP (CasADi/IPOPT B-spline 최적화)**
#
# THR nlp_planner 의 로봇판. 릴리즈 **윈도우** ±50 ms 전 구간에서 착탄
# 정확도를 목적함수(W_ACC)로 강제하므로 밸브 지터에 가장 강하다.
# THR 시뮬(보간 ON, 12던지기): ideal 12/12 평균 26 mm / real 12/12 평균 32 mm.
#
# ⚠ warm DB 를 쓰지 않는다 — 2026-08-03 공식화 변경(tool 0.240, rt 0.1, W1 5.0,
#   QDD 5×, 관절한계)으로 기존 skills/warm_db_*.pkl 이 전부 무효다. cold
#   multistart 라 지점당 수 초~수십 초 걸리지만 전부 로봇이 움직이기 전에 끝난다.
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
# nlp 은 집는 위치(p_start)가 궤적에 실제로 영향을 준다.
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
# 던지기 목표 ("x,y,z" [m]) — 1개만 쓰면 전 지점 공통,
# 여러 개면 지점 수와 같아야 하고 순서대로 1:1 매칭된다.
# ============================================================================
TARGETS=(
    "1.20,0.225,0.02"
    "1.20,0.075,0.02"
    "1.20,-0.075,0.02"
    "1.20,-0.225,0.02"

    "1.441,0.225,0.085"
    "1.441,0.075,0.085"
    "1.441,-0.075,0.085"
    "1.441,-0.225,0.085"

    "1.683,0.225,0.15"
    "1.683,0.075,0.15"
    "1.683,-0.075,0.15"
    "1.683,-0.225,0.15"
)


# --confirm-throw : 흡착+유지 후 Enter 를 눌러야 던진다 (q 입력 시 그 사이클 생략)
# --vel-scale     : 던지기 외 이동 속도 (계획 아크와 감속 꼬리는 영향 없음)
# --shuffle-points: 지점 방문 순서 무작위 (TARGETS 순서는 고정)
# --no-confirm    : 사이클마다 Enter 확인 생략
EXTRA_ARGS_STR="--confirm-throw --vel-scale 0.2"

# NLP 자체의 기둥 회피 제약 (THR 기본 off). 1 로 켜면 해가 줄지만 실기
# Cartesian 게이트 통과율이 올라간다.
export GP8_THR_NLP_CART="${GP8_THR_NLP_CART:-0}"
# 순차 multistart 예산 (고정 초기해 6개가 전멸했을 때만 쓴다).
export GP8_THR_NLP_MAX_RANDOM="${GP8_THR_NLP_MAX_RANDOM:-32}"

join_semi() { local IFS=";"; echo "$*"; }
export THR_MODEL="nlp"
export POINTS_ARG="$(join_semi "${POINTS[@]}")"
export TARGET_ARG="$(join_semi "${TARGETS[@]}")"
export EXTRA_ARGS_STR
exec "$HERE/run_static_pick_throw_thr.sh" "$@"

#!/usr/bin/env bash
# 정적(멈춘 벨트) 연속 pick-and-throw — **PHY (TossingBot Physics-only 탄도 컨트롤러)**
#
# Thr_Phy run_thr_sim 의 로봇판 — Zeng et al. T-RO 2020 의 physics-only
# 베이스라인. 릴리즈점 r 을 목표로부터 해석적으로 정하고(반경 c_d=0.60 m,
# 높이 c_h=0.30 m 원 위, 목표와 xy 공선 [Eq.1]) 45° 직선 램프로 가속한다.
# THR 시뮬(보간 ON, 12던지기): ideal 10/12 평균 85 mm / real 7/12 평균 138 mm.
#
# ⚠ 릴리즈 윈도우가 없고, 궤적이 직선 램프라 관절 여유가 가장 빡빡하다
#   (IK 실패·Cartesian 게이트 위반이 세 모델 중 가장 잦다). --thr-scan 필수.
#
# 사용:
#   ./run_static_pick_throw_phy.sh --thr-scan    # ★ 먼저 이걸로 실현 가능성 확인 (로봇 무명령)
#   ./run_static_pick_throw_phy.sh --plan-only   # 계획+궤적조립 검증 (driver 불필요)
#   ./run_static_pick_throw_phy.sh               # 실기 (driver 자동 기동)
#   ./run_static_pick_throw_phy.sh --target "1.35,0,-0.08"   # CLI 가 아래 배열을 덮어씀
#
# 공통 기계(빌드/driver/env)는 run_static_pick_throw_thr.sh 에 있다. 여기서는
# 지점·목표·모델별 옵션만 정한다.
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# 물체 지점 목록 (base frame "x,y,z" [m])
#   z = 프레스가 내려가는 최종 TCP 높이. 물체(place)는 z + 0.05 m 위에 놓인 것으로
#   본다 (static_pick_throw.py 의 PLACE_ABOVE_PRESS). "x,y" 로 z 생략 시 GRASP_Z.
# phy 는 집는 위치가 던지기에 영향을 주지 않는다 — 릴리즈점을 목표로부터
#   해석적으로 정한다.
# ============================================================================
POINTS=(
    "0.40,0.10,0.04"
    "0.40,0.0,0.04"
    "0.40,-0.10,0.04"

    "0.50,0.10,0.04"
    "0.50,0.0,0.04"
    "0.50,-0.10,0.04"
)

# ============================================================================
# 던지기 착지 목표 ("x,y,z" [m]) — 1개면 전 지점 공통, 여러 개면 지점 수와 일치.
# 기본값은 실기 bin 좌표 (구 run_static_pick_throw.sh 와 동일: 1.10/1.35/1.60,
# z=−0.08 = bin 바닥). phy 는 3D 목표를 그대로 쓴다.
# ============================================================================
TARGETS=(
    "1.10,0.075,-0.08"
    "1.10,-0.075,-0.08"
    "1.35,0.075,-0.08"
    "1.35,-0.075,-0.08"
    "1.60,0.075,-0.08"
    "1.60,-0.075,-0.08"
)

# --confirm-throw : 흡착+유지 후 Enter 를 눌러야 던진다 (q 입력 시 그 사이클 생략)
# --vel-scale     : 던지기 외 이동 속도 (계획 아크와 감속 꼬리는 영향 없음)
# --shuffle-points: 지점 방문 순서 무작위 (TARGETS 순서는 고정)
# --no-confirm    : 사이클마다 Enter 확인 생략
EXTRA_ARGS_STR="--confirm-throw --vel-scale 0.2"


join_semi() { local IFS=";"; echo "$*"; }
export THR_MODEL="phy"
export POINTS_ARG="$(join_semi "${POINTS[@]}")"
export TARGET_ARG="$(join_semi "${TARGETS[@]}")"
export EXTRA_ARGS_STR
exec "$HERE/run_static_pick_throw_thr.sh" "$@"

#!/usr/bin/env bash
# 정적(멈춘 벨트) 연속 pick-and-throw 테스트 — 올인원 래퍼.
# ROS 소싱 → colcon build → driver 스택 bringup(필요 시) → 테스트 실행.
#
# ── 설정은 아래 POINTS / TARGETS / EXTRA_ARGS 배열을 편집한다 ──
# 사용:
#   ./run_static_pick_throw.sh               # 배열값으로 실기 모드 (driver 자동 기동)
#   ./run_static_pick_throw.sh --plan-only   # 계획 검증만 (driver 불필요 — 기동 생략)
#   ./run_static_pick_throw.sh --vel-scale 0.5
#   ./run_static_pick_throw.sh --target "1.6,0,0"   # CLI가 배열값을 덮어씀(같은 옵션 뒤가 우선)
#
# driver 스택: 이미 떠 있으면 재사용, 없으면 이 스크립트가
#   ros2 launch gp8_control gp8_bringup.launch.py app:=false moveit:=false
# 를 백그라운드로 띄운다 (로그: $DRIVER_LOG). 기본적으로 스크립트가 끝나도
# driver는 계속 떠 있어 다음 실행에서 재사용된다 (KEEP_DRIVER=0이면 종료 시 정리).
# 전제: 펜던트 E-stop 해제 + 알람 리셋 + REMOTE (아니면 activation 거부 — CODE 10x).
set -eo pipefail   # -u는 ROS setup.bash가 미정의 변수를 참조해서 쓰지 않는다

# ============================================================================
# 물체 지점 목록 (base frame "x,y,z" [m]) — 한 줄에 하나씩 추가/수정.
#   z = 프레스가 내려가는 최종 TCP 높이 (오프셋 없이 그대로 적용).
#   물체(place)는 z + 0.05m 위에 놓인 것으로 간주 (static_pick_throw.py의
#   PLACE_ABOVE_PRESS) — hover/잡기/던지기 기준은 place, 하강만 z까지.
#   "x,y"로 z를 생략하면 Config.GRASP_Z.
# ============================================================================
POINTS=(
    # "0.40,0.30,0.04"
    "0.40,0.20,0.04"
    # "0.40,0.10,0.04"
    "0.40,0.02,0.04"
    # "0.40,-0.10,0.04"
    "0.40,-0.21,0.04"
    
    # "0.50,0.30,0.04"
    "0.50,0.20,0.04"
    # "0.50,0.10,0.04"
    "0.50,0.0,0.04"
    # "0.50,-0.10,0.04"
    "0.50,-0.20,0.04"
)

# ============================================================================
# 던지기 목표 ("x,y,z" [m]) — 1개만 쓰면 전 지점 공통,
# 여러 개면 지점 수와 같아야 하고 순서대로 1:1 매칭된다.
# ============================================================================
TARGETS=(
    "1.09, 0.16, -0.11"
    "1.09, 0.16, -0.11"
    "1.09, -0.16, -0.11"
    "1.09, -0.16, -0.11"
    "1.40, 0.0, -0.11"
    "1.40, 0.0, -0.11"
)

# 항상 붙는 추가 옵션 (예: "--vel-scale" "0.3", 셔플 재현은 "--seed" "42", "--shuffle-points" )
# --shuffle-points: 지점 방문 순서를 무작위로 섞음. TARGETS 순서는 고정이라
# "i번째 던지기 → TARGETS[i]"는 그대로, 어느 지점을 i번째로 집는지만 랜덤.
# "--no-confirm" "--confirm-throw"
EXTRA_ARGS=("--confirm-throw" "--vel-scale" "0.2")

# driver 스택: 1 = 스크립트 종료 후에도 유지(재실행 빠름), 0 = 종료 시 함께 정리
KEEP_DRIVER=1

# ============================================================================
WS="$HOME/ros2_ws"
PKG_DIR="$WS/src/gp8_control"
VENV_PY="$PKG_DIR/.venv/bin/python"
DRIVER_LOG="/tmp/gp8_driver_bringup.log"
JTC_ACTION="/joint_trajectory_controller/follow_joint_trajectory"

# ── 6쌍(저bin, 4cm/grip40mm) warm DB 연결 ──────────────────────────────────
# 이 테스트를 skills/warm_db_6pairs_x109_140.pkl (target x=1.09/1.40, GRIP_OFF 4cm 로 빌드) 로 돌린다.
#   • GP8_GRIP_OFF=0.04 → 플래너 GRIP_OFF/launch_state 발사점이 4cm 가 되어
#     FLIGHT_MODEL 이 grip40mm 이 됨 → 6쌍 DB 의 formulation 해시와 매칭
#     (안 맞으면 로더가 entry 를 전부 무시하고 cold 로 재풀이 → 3~14s/쌍).
#   • GP8_THROW_WARM_DB → warm DB 파일을 6쌍 DB 로 지정.
# 둘 다 이 스크립트 실행에만 export (기본값 2cm + warm_db.pkl 은 앱/다른 실행 그대로).
# 연결 해제하려면 아래 두 줄을 주석 처리(그러면 기본 2cm + warm_db.pkl).
WARM_DB_6PAIRS="$PKG_DIR/skills/warm_db_6pairs_x109_140.pkl"
export GP8_GRIP_OFF="${GP8_GRIP_OFF:-0.04}"
export GP8_THROW_WARM_DB="${GP8_THROW_WARM_DB:-$WARM_DB_6PAIRS}"

join_semi() { local IFS=";"; echo "$*"; }   # 배열 → "a;b;c"

driver_up() {
    timeout 5 ros2 action list 2>/dev/null | grep -q "$JTC_ACTION"
}

if [ ! -x "$VENV_PY" ]; then
    echo "venv python 없음: $VENV_PY  (cd $PKG_DIR && uv sync)" >&2
    exit 1
fi

POINTS_ARG="$(join_semi "${POINTS[@]}")"
TARGET_ARG="$(join_semi "${TARGETS[@]}")"
ARGS=(--points "$POINTS_ARG" --target "$TARGET_ARG" "${EXTRA_ARGS[@]}" "$@")

echo "=== [1/4] source ROS + build gp8_control ==="
source /opt/ros/humble/setup.bash
# gp8_bringup.launch.py 는 motoman_bringup(별도 워크스페이스)에 의존한다.
# 이 오버레이가 없으면 launch 가 "package 'motoman_bringup' not found" 로 죽고,
# driver 사망 핸들러가 (무관한) 펜던트 CODE 10x 안내를 출력한다. 오버레이 선(先)source.
MOTOMAN_OVERLAY="${GP8_MOTOMAN_OVERLAY:-/home/robotics/Desktop/motoman_ros2_safety_review/motoman_ROS2/install/setup.bash}"
if [ -f "$MOTOMAN_OVERLAY" ]; then
    source "$MOTOMAN_OVERLAY"
else
    echo "경고: motoman 오버레이 없음($MOTOMAN_OVERLAY) — GP8_MOTOMAN_OVERLAY 로 지정 필요" >&2
fi
cd "$WS"
colcon build --packages-select gp8_control
source "$WS/install/setup.bash"

echo "=== [2/4] driver 스택 확인/기동 ==="
DRIVER_PID=""
if [[ " ${ARGS[*]} " == *" --plan-only "* ]]; then
    echo "plan-only — driver 불필요, 기동 생략"
elif driver_up; then
    echo "driver 스택 이미 실행 중 — 재사용"
else
    echo "driver 스택 시작 (app:=false moveit:=false, log: $DRIVER_LOG)"
    ros2 launch gp8_control gp8_bringup.launch.py app:=false moveit:=false \
        >"$DRIVER_LOG" 2>&1 &
    DRIVER_PID=$!
    deadline=$((SECONDS + 40))
    ready=0
    while [ $SECONDS -lt $deadline ]; do
        if ! kill -0 "$DRIVER_PID" 2>/dev/null; then
            echo "driver가 기동 중 죽음 — 로그 마지막:" >&2
            tail -n 12 "$DRIVER_LOG" >&2
            grep -m1 "ROBOT ERROR" "$DRIVER_LOG" >&2 || true
            echo "(CODE: 101=ALARM 103=ESTOP 104=NOT_PLAY 105=NOT_REMOTE 106=SERVO_OFF 107=HOLD" >&2
            echo " → 펜던트에서 E-stop 해제/알람 리셋/REMOTE 확인 후 재실행)" >&2
            exit 1
        fi
        if driver_up; then ready=1; break; fi
        sleep 1
    done
    if [ "$ready" != "1" ]; then
        echo "driver가 40s 내에 준비되지 않음 — 로그 마지막:" >&2
        tail -n 12 "$DRIVER_LOG" >&2
        kill "$DRIVER_PID" 2>/dev/null || true
        exit 1
    fi
    if [ "$KEEP_DRIVER" = "1" ]; then
        echo "driver 준비 완료 (PID $DRIVER_PID — 종료 후에도 유지. 중지: kill $DRIVER_PID)"
    else
        echo "driver 준비 완료 (PID $DRIVER_PID — 테스트 종료 시 함께 정리)"
        trap 'echo "driver 정리 (PID $DRIVER_PID)"; kill $DRIVER_PID 2>/dev/null || true' EXIT
    fi
fi

echo "=== [3/4] 실행 인자 ==="
echo "points: $POINTS_ARG"
echo "target: $TARGET_ARG"
[ "$#" -gt 0 ] && echo "cli   : $*"

echo "=== [4/4] run static_pick_throw ==="
echo "warm DB : $GP8_THROW_WARM_DB (GP8_GRIP_OFF=$GP8_GRIP_OFF → grip$(awk "BEGIN{printf \"%.0f\", $GP8_GRIP_OFF*1000}")mm)"
[ -f "$GP8_THROW_WARM_DB" ] || echo "  경고: warm DB 파일 없음 — cold 로 진행됨" >&2
# 배열값 먼저 + CLI 뒤 — argparse는 같은 옵션 중복 시 마지막 것을 쓰므로
# CLI --points/--target이 배열값을 덮어쓰고, 플래그는 그대로 추가된다.
# GP8_GRIP_OFF / GP8_THROW_WARM_DB 는 위에서 export 됨 (6쌍 DB 연결).
env PYTHONPATH="$WS/src:${PYTHONPATH:-}" \
    "$VENV_PY" -m gp8_control.tests.static_pick_throw "${ARGS[@]}"

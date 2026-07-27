#!/usr/bin/env bash
# 동적(카메라 인식 + 컨베이어 벨트 이동) 자율 pick-and-throw 실행 래퍼.
#
# run_static_pick_throw.sh 는 멈춘 벨트에서 정해진 POINTS/TARGETS 로 계획을
# 검증하는 "테스트"다. 이 스크립트는 그게 아니라 **실제 운영 경로**를 띄운다:
#   camera_debug(카메라 PC HTTP 스트림 → /camera_debug/detections)
#   + esp32 컨베이어 엔코더(/conveyor/speed)
#   + gp8_bringup(드라이버 + MoveIt + gp8_manager 앱)
# gp8_manager 가 인식된 물체를 벨트 위에서 ambush 로 잡고, 선택된 스킬(기본
# robust_throw = 우리가 튜닝한 CasADi/IPOPT NLP thrower + warm_db.pkl)로 던진다.
#
# ┌───────────────────────────────────────────────────────────────────────┐
# │  경고: 이 스크립트는 실제 로봇을 '자율'로 움직여 물체를 던진다.          │
# │  • 팔 반경 1 m 를 완전히 비울 것 (throw = 실제 스윙).                    │
# │  • 펜던트: E-stop 해제 + 알람 리셋 + REMOTE + AUTO.                      │
# │  • E-stop 을 손에 둘 것. 앱 기동 직전 ENTER 확인 게이트가 있다           │
# │    (CONFIRM_START=1). 게이트 이후에는 카메라가 잡는 대로 계속 던진다.    │
# └───────────────────────────────────────────────────────────────────────┘
#
# 사용:
#   ./run_dynamic_pick_throw.sh                 # 기본값으로 전체 스택 기동
#   SKILL="" ./run_dynamic_pick_throw.sh        # 강제 스킬 해제 → 클래스별 라우팅
#   SKILL=throw ./run_dynamic_pick_throw.sh     # NN thrower 로
#   START_CONVEYOR=0 ./run_dynamic_pick_throw.sh  # 컨베이어 노드는 안 띄움(이미 HW 발행 중)
#   GP8_PERCEPTION_URL=http://호스트:8080/detections/stream ./run_dynamic_pick_throw.sh
#   BRINGUP_EXTRA="rviz:=true axis_increment_factor:=0.5" ./run_dynamic_pick_throw.sh
#
# 이미 떠 있는 구성요소는 재사용한다(메모리: 로봇 PC 는 기본 도메인에 스택이
# 떠 있는 경우가 많음). 드라이버(controller_manager)만 떠 있고 앱이 없으면
# 전체 bringup 대신 move_group + 앱만 따로 붙여 드라이버를 재사용한다
# (gp8_bringup 은 controller_manager 를 무조건 다시 띄워 중복 충돌하므로).
set -eo pipefail   # -u 는 ROS setup.bash 가 미정의 변수를 참조해서 쓰지 않는다

# ============================================================================
# 설정
# ============================================================================
# 강제 스킬: robust_throw(NLP·튜닝 대상) | throw(NN) | push | "" (클래스별 라우팅)
SKILL="${SKILL:-robust_throw}"
# 카메라 PC HTTP 검출 스트림. 미설정이면 .env → camera_debug 기본값 순으로 사용.
GP8_PERCEPTION_URL="${GP8_PERCEPTION_URL:-}"
MOVEIT="${MOVEIT:-1}"            # move_group 기동(앱 IK/플래닝에 필수). 0 = 이미 떠 있음/생략
START_CAMERA="${START_CAMERA:-1}"     # camera_debug 없으면 기동
START_CONVEYOR="${START_CONVEYOR:-1}" # /conveyor/speed 발행자 없으면 esp32 엔코더 기동
CONFIRM_START="${CONFIRM_START:-1}"   # 로봇 구동(앱) 시작 전 ENTER 확인(안전). 0 = 즉시
KEEP_STACK="${KEEP_STACK:-1}"         # 1 = 스크립트 종료 후에도 유지, 0 = 종료 시 정리
# gp8_bringup 추가 인자(공백 구분). 예: "rviz:=true grasp_z:=0.06 axis_increment_factor:=0.5"
BRINGUP_EXTRA="${BRINGUP_EXTRA:-}"

# ============================================================================
WS="$HOME/ros2_ws"
PKG_DIR="$WS/src/gp8_control"
VENV_PY="$PKG_DIR/.venv/bin/python"
LOGDIR="/tmp/gp8_dynamic"
mkdir -p "$LOGDIR"
BRINGUP_LOG="$LOGDIR/bringup.log"
APP_LOG="$LOGDIR/app.log"
MOVEIT_LOG="$LOGDIR/move_group.log"
CAMERA_LOG="$LOGDIR/camera_debug.log"
CONVEYOR_LOG="$LOGDIR/conveyor.log"
FJT_ACTION="/joint_trajectory_controller/follow_joint_trajectory"
PIDS=()   # KEEP_STACK=0 일 때 정리할, 이 스크립트가 띄운 PID 들

# --- helpers ----------------------------------------------------------------
node_up()  { timeout 6 ros2 node list  2>/dev/null | grep -qx "/$1"; }
action_up(){ timeout 6 ros2 action list 2>/dev/null | grep -q  "$1"; }
topic_has_pub() {   # 발행자 ≥1 인가
    timeout 6 ros2 topic info "$1" 2>/dev/null | grep -qE "Publisher count: [1-9]"
}
wait_for() {        # wait_for "설명" 타임아웃 검사함수 [인자...]
    local desc="$1" deadline=$(( SECONDS + $2 )); shift 2
    while [ $SECONDS -lt "$deadline" ]; do
        if "$@"; then return 0; fi
        sleep 1
    done
    echo "  ✗ $desc 준비 안 됨 (${*})" >&2; return 1
}
start_bg() {        # start_bg 로그파일 명령...  → PID 기록
    local log="$1"; shift
    ( "$@" ) >"$log" 2>&1 &
    local pid=$!
    PIDS+=("$pid")
    echo "$pid"
}

if [ ! -x "$VENV_PY" ]; then
    echo "venv python 없음: $VENV_PY  (cd $PKG_DIR && uv sync)" >&2
    exit 1
fi

# ============================================================================
echo "=== [1/6] source ROS + motoman 오버레이 + build gp8_control ==="
source /opt/ros/humble/setup.bash
# gp8_bringup 은 motoman_bringup(별도 워크스페이스)에 의존 — 오버레이 선(先)source.
MOTOMAN_OVERLAY="${GP8_MOTOMAN_OVERLAY:-/home/robotics/Desktop/motoman_ros2_safety_review/motoman_ROS2/install/setup.bash}"
if [ -f "$MOTOMAN_OVERLAY" ]; then
    source "$MOTOMAN_OVERLAY"
else
    echo "경고: motoman 오버레이 없음($MOTOMAN_OVERLAY) — GP8_MOTOMAN_OVERLAY 로 지정 필요" >&2
fi
cd "$WS"
colcon build --packages-select gp8_control
source "$WS/install/setup.bash"

# --- GP8_PERCEPTION_URL: 셸 → .env → camera_debug 기본값 --------------------
if [ -z "$GP8_PERCEPTION_URL" ] && [ -f "$PKG_DIR/.env" ]; then
    GP8_PERCEPTION_URL="$(grep -E '^GP8_PERCEPTION_URL=' "$PKG_DIR/.env" | tail -1 | cut -d= -f2- | tr -d '"'"'"'' || true)"
fi
if [ -n "$GP8_PERCEPTION_URL" ]; then
    export GP8_PERCEPTION_URL
    echo "  GP8_PERCEPTION_URL=$GP8_PERCEPTION_URL"
else
    echo "  GP8_PERCEPTION_URL 미설정 — camera_debug 기본값 사용(147.46.175.15:8080)"
fi

# ============================================================================
echo "=== [2/6] 컨베이어 엔코더 (/conveyor/speed) ==="
if topic_has_pub /conveyor/speed; then
    echo "  /conveyor/speed 이미 발행 중 — 재사용"
elif [ "$START_CONVEYOR" = "1" ]; then
    echo "  esp32_encoder conveyor_node 기동 (log: $CONVEYOR_LOG)"
    start_bg "$CONVEYOR_LOG" ros2 run esp32_encoder conveyor_node >/dev/null
    if wait_for "/conveyor/speed 발행" 15 topic_has_pub /conveyor/speed; then
        echo "  ✓ /conveyor/speed 발행 시작"
    else
        echo "  ! 컨베이어 엔코더가 안 뜸 — ESP32(/dev/ttyUSB0)·dialout 권한 확인." >&2
        echo "    (앱은 없어도 CONVEYOR_FALLBACK 로 동작하지만 벨트 이동은 실제 인식 안 됨)" >&2
    fi
else
    echo "  START_CONVEYOR=0 — 컨베이어 노드 생략 (외부 HW 발행 가정)"
fi
# 라이브 속도 한 값 보고(0 이면 엔코더 죽었을 수 있음 — 메모리: dead esp32 → weak/late throws)
SPD="$(timeout 3 ros2 topic echo --once /conveyor/speed 2>/dev/null | grep -m1 'data:' | awk '{print $2}' || true)"
[ -n "$SPD" ] && echo "  현재 벨트 속도 ≈ ${SPD} m/s $( [ "$SPD" = "0.0" ] && echo '(⚠ 0 — 엔코더/24V 확인)')"

# ============================================================================
echo "=== [3/6] 카메라 인식 (camera_debug → /camera_debug/detections) ==="
if node_up camera_debug; then
    echo "  camera_debug 이미 실행 중 — 재사용"
elif [ "$START_CAMERA" = "1" ]; then
    echo "  camera_debug 기동 (log: $CAMERA_LOG)"
    start_bg "$CAMERA_LOG" ros2 run gp8_control camera_debug >/dev/null
    if wait_for "camera_debug /camera_debug/detections 발행" 20 topic_has_pub /camera_debug/detections; then
        echo "  ✓ /camera_debug/detections 발행 시작"
    else
        echo "  ! camera_debug 가 안 뜸 — 로그 마지막:" >&2; tail -n 8 "$CAMERA_LOG" >&2
        echo "    스트림 도달 확인: curl -N \"\$GP8_PERCEPTION_URL\" | head" >&2
    fi
else
    echo "  START_CAMERA=0 — camera_debug 생략"
fi

# ============================================================================
echo "=== [4/6] 드라이버 + MoveIt + 앱 스택 ==="
if node_up gp8_manager; then
    # 앱이 이미 떠 있음 → 전체 스택 재사용. 강제 스킬은 기동 시점에 고정되므로
    # 다른 스킬을 원하면 기존 gp8_manager 를 먼저 종료해야 한다.
    echo "  gp8_manager 이미 실행 중 — 전체 스택 재사용 (SKILL 변경하려면 기존 앱 종료 후 재실행)"
    STACK_MODE="reuse-app"
elif action_up "$FJT_ACTION" || node_up controller_manager; then
    # 드라이버만 떠 있음 → gp8_bringup 은 controller_manager 를 중복 기동해 충돌한다.
    # 그래서 move_group + 앱만 따로 붙여 라이브 드라이버를 재사용한다.
    echo "  드라이버(controller_manager) 이미 실행 중 — bringup 전체 대신 재사용 경로"
    STACK_MODE="reuse-driver"
    if [ "$MOVEIT" = "1" ] && ! node_up move_group; then
        echo "  move_group 기동 (log: $MOVEIT_LOG)"
        start_bg "$MOVEIT_LOG" ros2 launch motoman_gp8_moveit_config move_group.launch.py >/dev/null
        wait_for "move_group /move_action" 40 action_up /move_action \
            && echo "  ✓ move_group 준비" \
            || { echo "  ✗ move_group 실패 — 앱 IK/플래닝이 30s 대기 후 실패한다. 로그:" >&2; tail -n 8 "$MOVEIT_LOG" >&2; }
    elif node_up move_group; then
        echo "  move_group 이미 실행 중 — 재사용"
    fi
else
    # 아무것도 없음 → 정석: gp8_bringup 전체(드라이버+MoveIt+앱)
    echo "  전체 gp8_bringup 기동 (skill:=${SKILL:-<클래스라우팅>} moveit:=$( [ "$MOVEIT" = 1 ] && echo true || echo false ))"
    echo "  log: $BRINGUP_LOG"
    STACK_MODE="full-bringup"
fi

# ============================================================================
echo "=== [5/6] 로봇 구동(앱) 시작 확인 ==="
if [ "$STACK_MODE" != "reuse-app" ] && [ "$CONFIRM_START" = "1" ]; then
    echo "──────────────────────────────────────────────────────────────"
    echo " 카메라/컨베이어 준비 완료. 지금부터 앱이 인식된 물체를 자율로"
    echo " 잡고 던진다. 팔 반경 1 m 를 비웠는지, 펜던트 REMOTE+AUTO·알람"
    echo " 해제인지, E-stop 이 손에 있는지 확인하라."
    echo "──────────────────────────────────────────────────────────────"
    read -r -p " 로봇 자율 pick-throw 를 시작하려면 ENTER (중단: Ctrl-C): " _
fi

case "$STACK_MODE" in
  full-bringup)
    # gp8_bringup: 드라이버 + (선택)MoveIt + gp8_manager 앱. skill:= 로 강제 스킬 전달.
    # shellcheck disable=SC2086
    start_bg "$BRINGUP_LOG" ros2 launch gp8_control gp8_bringup.launch.py \
        app:=true moveit:=$( [ "$MOVEIT" = 1 ] && echo true || echo false ) \
        skill:="$SKILL" $BRINGUP_EXTRA >/dev/null
    ;;
  reuse-driver)
    # 라이브 드라이버 재사용: 앱만 venv python 으로 직접 기동(--skill 이 env 를 이김).
    # torch 때문에 venv 필요. PYTHONPATH 로 소스 트리 지정.
    APP_CMD=("$VENV_PY" -m gp8_control.app)
    [ -n "$SKILL" ] && APP_CMD+=(--skill "$SKILL")
    echo "  앱 기동: ${APP_CMD[*]}  (log: $APP_LOG)"
    ( export PYTHONPATH="$WS/src:${PYTHONPATH:-}"; exec "${APP_CMD[@]}" ) >"$APP_LOG" 2>&1 &
    PIDS+=("$!")
    ;;
  reuse-app)
    : ;;   # 이미 떠 있음
esac

if [ "$STACK_MODE" != "reuse-app" ]; then
    APP_READY_LOG="$( [ "$STACK_MODE" = full-bringup ] && echo "$BRINGUP_LOG" || echo "$APP_LOG" )"
    if wait_for "gp8_manager 앱" 60 node_up gp8_manager; then
        echo "  ✓ gp8_manager 실행 중"
    else
        echo "  ✗ 앱이 60s 내 안 뜸 — 로그 마지막:" >&2; tail -n 15 "$APP_READY_LOG" >&2
        echo "    (드라이버 activation 거부면 CODE 10x: 펜던트 REMOTE/AUTO/알람 확인)" >&2
        exit 1
    fi
fi

# ============================================================================
echo "=== [6/6] 실행 상태 ==="
echo "  스택 모드 : $STACK_MODE"
echo "  스킬      : ${SKILL:-<클래스별 라우팅>}"
echo "  로그      : bringup=$BRINGUP_LOG  app=$APP_LOG  camera=$CAMERA_LOG  conveyor=$CONVEYOR_LOG"
echo "  노드 확인 : ros2 node list | grep -E 'gp8_manager|camera_debug|move_group'"
echo "  검출 확인 : ros2 topic echo /camera_debug/detections --once"
echo

if [ "$KEEP_STACK" != "1" ]; then
    # shellcheck disable=SC2064
    trap "echo; echo '스택 정리...'; kill ${PIDS[*]} 2>/dev/null || true" EXIT
    echo "  KEEP_STACK=0 — Ctrl-C(또는 스크립트 종료) 시 이 스크립트가 띄운 프로세스를 정리한다."
else
    echo "  KEEP_STACK=1 — 종료 후에도 스택 유지. 정리: kill ${PIDS[*]:-<PID>}"
fi

# 앱 로그를 foreground 로 tail — 운영자가 pick/throw 동작을 실시간 확인.
# Ctrl-C 는 tail 만 끊는다(백그라운드 앱은 KEEP_STACK 설정을 따름).
TAIL_LOG="$( [ "$STACK_MODE" = full-bringup ] && echo "$BRINGUP_LOG" || echo "$APP_LOG" )"
if [ "$STACK_MODE" = "reuse-app" ]; then
    echo "  (앱을 이 스크립트가 안 띄웠으므로 tail 생략 — 기존 터미널 로그 확인)"
else
    echo "  --- app 로그 tail (Ctrl-C 로 빠져나옴) ---"
    tail -n +1 -f "$TAIL_LOG"
fi

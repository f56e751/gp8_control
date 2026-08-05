#!/usr/bin/env bash
# 정적(멈춘 벨트) 연속 pick-and-throw — **THR 계획 모델 공통 실행기**.
#
# 직접 실행하지 말고 모델별 래퍼를 쓴다 (POINTS/TARGETS 는 거기서 편집):
#     ./run_static_pick_throw_nlp.sh    # CasADi/IPOPT B-spline NLP
#     ./run_static_pick_throw_dt.sh     # Decision Transformer (THR GP8 rig 학습)
#     ./run_static_pick_throw_phy.sh    # TossingBot Physics-only 탄도
#
# 래퍼가 export 하는 것: THR_MODEL, POINTS_ARG, TARGET_ARG, EXTRA_ARGS_STR
# 이 스크립트: ROS 소싱 → colcon build → driver 스택 bringup(필요 시) → 테스트 실행.
# 전제: 펜던트 E-stop 해제 + 알람 리셋 + REMOTE (아니면 activation 거부 — CODE 10x).
set -eo pipefail   # -u는 ROS setup.bash가 미정의 변수를 참조해서 쓰지 않는다

: "${THR_MODEL:?모델별 래퍼(run_static_pick_throw_{nlp,dt,phy}.sh)로 실행하라}"

WS="$HOME/ros2_ws"
PKG_DIR="$WS/src/gp8_control"
VENV_PY="$PKG_DIR/.venv/bin/python"
DRIVER_LOG="/tmp/gp8_driver_bringup.log"
JTC_ACTION="/joint_trajectory_controller/follow_joint_trajectory"
KEEP_DRIVER="${KEEP_DRIVER:-1}"   # 1 = 종료 후에도 driver 유지(재실행 빠름)

# ============================================================================
# THR 공통 설정 — 세 모델이 공유한다
# ============================================================================
# 던지기 시작 TCP 높이 [m] (THR sim_env.P_START_Z). 물체를 집은 뒤 이 높이로
# 들어올린 자세에서 스윙이 시작된다. nlp 만 실제로 이 값에 민감하다
# (dt 는 학습 시 home 자세 고정, phy 는 릴리즈점을 목표로부터 해석적으로 결정).
export GP8_THR_P_START_Z="${GP8_THR_P_START_Z:-0.20}"
#
# Cartesian 안전 엔벨로프 (dispatch 직전 hard 게이트):
#   x > 0.20, z > 0.04 (z 상한은 검사하지 않음)
# ⚠ THR 은 2026-07-31 사용자 지시로 NLP 에서 카타시안 제약을 모두 뺐고
#   (throw_nlp.CART_CONSTRAINTS=False) 시뮬은 위반을 보고만 한다. 시뮬은 그래도
#   되지만 실기는 기둥/바닥을 실제로 친다 — 그래서 하한 게이트는 hard 로 유지한다.
#   위반하는 계획은 실행하지 않고 그 지점을 통째로 건너뛴다.
# 1 로 두면 NLP 자체의 기둥 회피 제약도 되살린다 (해가 줄지만 게이트 통과율↑).
export GP8_THR_NLP_CART="${GP8_THR_NLP_CART:-0}"
#
# 착탄 목표 오차는 진단값으로만 출력하며 계획을 거부하지 않는다.
#
# release 명령 선행 시간 [s]. 양수 = 그만큼 일찍 명령(밸브 지연 보정), 음수 = 늦게.
# nlp 은 ±50 ms 릴리즈 윈도우가 있어 지터를 흡수하지만 **dt/phy 는 릴리즈가 한
# 점**이라 어긋난 만큼 그대로 착지 오차가 된다 (스킬이 mm/10 ms 민감도를 찍는다).
# 밸브 지연을 실측했다면 그 값을 양수로 넣고, 아니면 0 으로 둔다.
export GP8_RELEASE_LEAD="${GP8_RELEASE_LEAD:-0.0}"
#
# ⚠ 구 NLP 스킬용 env (GP8_THROW_WARM_DB / GP8_GRIP_OFF / GP8_L_LO_DEG /
#   GP8_U_LO_DEG) 는 여기서 쓰지 않는다. THR 경로는 벤더링된 skills/thr/ 의
#   공식화(관절 한계·GRIP_OFF 2 cm)를 그대로 쓰고, 2026-08-03 공식화 변경으로
#   기존 warm DB 는 전부 무효다.

driver_up() { timeout 5 ros2 action list 2>/dev/null | grep -q "$JTC_ACTION"; }

if [ ! -x "$VENV_PY" ]; then
    echo "venv python 없음: $VENV_PY  (cd $PKG_DIR && uv sync)" >&2
    exit 1
fi

# shellcheck disable=SC2206
EXTRA=( ${EXTRA_ARGS_STR:-} )
ARGS=(--model "$THR_MODEL" --points "$POINTS_ARG" --target "$TARGET_ARG"
      "${EXTRA[@]}" "$@")

NO_ROBOT=0
[[ " ${ARGS[*]} " == *" --plan-only "* ]] && NO_ROBOT=1
[[ " ${ARGS[*]} " == *" --thr-scan "* ]] && NO_ROBOT=1

echo "=== [1/4] source ROS + build gp8_control ==="
source /opt/ros/humble/setup.bash
# gp8_bringup.launch.py 는 motoman_bringup(별도 워크스페이스)에 의존한다.
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
if [ "$NO_ROBOT" = "1" ]; then
    echo "plan-only / thr-scan — driver 불필요, 기동 생략"
elif driver_up; then
    echo "driver 스택 이미 실행 중 — 재사용"
else
    echo "driver 스택 시작 (app:=false moveit:=false, log: $DRIVER_LOG)"
    ros2 launch gp8_control gp8_bringup.launch.py app:=false moveit:=false \
        >"$DRIVER_LOG" 2>&1 &
    DRIVER_PID=$!
    deadline=$((SECONDS + 40)); ready=0
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
echo "model : $THR_MODEL"
echo "points: $POINTS_ARG"
echo "target: $TARGET_ARG"
[ -n "${EXTRA_ARGS_STR:-}" ] && echo "extra : $EXTRA_ARGS_STR"
[ "$#" -gt 0 ] && echo "cli   : $*"

echo "=== [4/4] run static_pick_throw_thr (model=$THR_MODEL) ==="
echo "P_START_Z  : $GP8_THR_P_START_Z m (던지기 시작 TCP 높이)"
echo "Cartesian  : x>0.20 m, z>0.04 m (z 상한 없음)"
echo "NLP cart   : GP8_THR_NLP_CART=$GP8_THR_NLP_CART (1=NLP 기둥 회피 제약 부활)"
echo "release    : GP8_RELEASE_LEAD=$GP8_RELEASE_LEAD s (+일찍 −늦게)"
[ -n "${GP8_THR_DT_WEIGHTS:-}" ] && echo "DT weights : $GP8_THR_DT_WEIGHTS"
env PYTHONPATH="$WS/src:${PYTHONPATH:-}" \
    "$VENV_PY" -m gp8_control.tests.static_pick_throw_thr "${ARGS[@]}"

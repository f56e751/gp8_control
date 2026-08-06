#!/usr/bin/env bash
# DT sim-to-real throw-data collection: environment setup, validation, and launch.
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
WS="$(cd -- "$PKG_DIR/../.." && pwd)"
VENV_PY="$PKG_DIR/.venv/bin/python"

ROS_SETUP="${GP8_ROS_SETUP:-/opt/ros/humble/setup.bash}"
MOTOMAN_SETUP="${GP8_MOTOMAN_OVERLAY:-/home/robotics/Desktop/motoman_ros2_safety_review/motoman_ROS2/install/setup.bash}"
WS_SETUP="$WS/install/setup.bash"
DRIVER_LOG="${GP8_DRIVER_LOG:-/tmp/gp8_collect_driver.log}"
JTC_ACTION="/joint_trajectory_controller/follow_joint_trajectory"

DEFAULT_GOALS="${GP8_COLLECT_GOALS:-1.1,1.3,1.5,1.7}"
DEFAULT_REPS="${GP8_COLLECT_REPS:-3}"
# 착지면 높이 [m] — bin 바닥 기준. 맨바닥에 던지면 0 으로 덮어쓸 것.
DEFAULT_LAND_Z="${GP8_COLLECT_LAND_Z:--0.08}"

die() {
    echo "오류: $*" >&2
    exit 1
}

has_option() {
    local wanted="$1"
    shift
    local arg
    for arg in "$@"; do
        if [[ "$arg" == "$wanted" || "$arg" == "$wanted="* ]]; then
            return 0
        fi
    done
    return 1
}

driver_up() {
    timeout 5 ros2 action list 2>/dev/null | grep -Fxq "$JTC_ACTION"
}

manager_up() {
    timeout 5 ros2 node list 2>/dev/null | grep -Fxq "/gp8_manager"
}

[[ -f "$ROS_SETUP" ]] || die "ROS 2 환경을 찾지 못했습니다: $ROS_SETUP"
[[ -x "$VENV_PY" ]] || die "프로젝트 Python을 찾지 못했습니다: $VENV_PY"

echo "[1/5] ROS 2 및 작업공간 환경 로드"
# shellcheck disable=SC1090
source "$ROS_SETUP"

if [[ -f "$MOTOMAN_SETUP" ]]; then
    # shellcheck disable=SC1090
    source "$MOTOMAN_SETUP"
else
    echo "경고: Motoman overlay가 없습니다: $MOTOMAN_SETUP" >&2
fi

[[ -f "$WS_SETUP" ]] || die "작업공간이 빌드되지 않았습니다. 먼저: cd $WS && colcon build --packages-select gp8_control"
# shellcheck disable=SC1090
source "$WS_SETUP"

# 기존 ROS Python 경로를 보존해야 venv에서도 rclpy를 가져올 수 있다.
export PYTHONPATH="$WS/src:${PYTHONPATH:-}"
# 2026-08-06 B 전환: effort_pid(공식 물리) 학습 조건 재현 — 셋 다 필수.
# 빠지면 계획 rollout 이 학습 env 와 어긋난다 (특히 LIMITS 기본값은 nlp 협소).
export DT_GP8_DYN="${DT_GP8_DYN:-effort_pid}"
export DT_GP8_LIMITS="${DT_GP8_LIMITS:-urdf}"
export DT_GP8_WP_VEL="${DT_GP8_WP_VEL:-segment}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "[2/5] Python 의존성 확인"
"$VENV_PY" -c '
import casadi
import rclpy
import torch
import gp8_control.tools.collect_real_throws_gp8
print(f"  rclpy: {rclpy.__file__}")
print(f"  torch: {torch.__version__}")
print(f"  casadi: {casadi.__version__}")
' || die "Python 의존성 확인 실패. ROS setup과 $VENV_PY 상태를 확인하세요."

# 도움말은 로봇/터미널 검사 없이 볼 수 있어야 한다.
if has_option "--help" "$@" || has_option "-h" "$@"; then
    exec "$VENV_PY" -m gp8_control.tools.collect_real_throws_gp8 "$@"
fi

[[ -t 0 ]] || die "이 수집기는 착지 거리를 직접 입력하므로 대화형 터미널에서 실행해야 합니다."

echo "[3/5] gp8_manager 충돌 확인"
if manager_up; then
    die "/gp8_manager가 실행 중입니다. manager를 종료한 뒤 다시 실행하세요."
fi

echo "[4/5] GP8 드라이버 확인"
if driver_up; then
    echo "  기존 GP8 드라이버를 사용합니다."
else
    [[ -f "$MOTOMAN_SETUP" ]] || die "드라이버를 자동 시작하려면 Motoman overlay가 필요합니다: $MOTOMAN_SETUP"

    echo "  드라이버를 시작합니다. 로그: $DRIVER_LOG"
    ros2 launch gp8_control gp8_bringup.launch.py \
        app:=false moveit:=false rviz:=false >"$DRIVER_LOG" 2>&1 &
    DRIVER_PID=$!

    for _ in $(seq 1 40); do
        if ! kill -0 "$DRIVER_PID" 2>/dev/null; then
            tail -n 30 "$DRIVER_LOG" >&2 || true
            die "GP8 드라이버가 시작 중 종료되었습니다. 펜던트 Remote/Play, Servo ON, safety 상태를 확인하세요."
        fi
        if driver_up; then
            echo "  드라이버 준비 완료(PID $DRIVER_PID). 수집 종료 후에도 계속 실행됩니다."
            break
        fi
        sleep 1
    done

    if ! driver_up; then
        tail -n 30 "$DRIVER_LOG" >&2 || true
        die "40초 안에 trajectory action이 준비되지 않았습니다: $JTC_ACTION"
    fi
fi

ARGS=("$@")
if ! has_option "--goals" "$@"; then
    ARGS=(--goals "$DEFAULT_GOALS" "${ARGS[@]}")
fi
if ! has_option "--reps" "$@"; then
    ARGS=(--reps "$DEFAULT_REPS" "${ARGS[@]}")
fi
if ! has_option "--land-z" "$@"; then
    ARGS=(--land-z "$DEFAULT_LAND_Z" "${ARGS[@]}")
fi

echo "[5/5] 실험 수집기 실행"
echo "  안전 펜스 내부에 사람이 없는지 확인하세요. 로봇이 실제로 움직입니다."
printf "  명령:"
printf " %q" "$VENV_PY" -m gp8_control.tools.collect_real_throws_gp8 "${ARGS[@]}"
printf "\n"

exec "$VENV_PY" -m gp8_control.tools.collect_real_throws_gp8 "${ARGS[@]}"

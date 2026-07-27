#!/usr/bin/env bash
# 움직이는 컨베이어에서 robust_throw(NLP) 연속 pick-and-throw 실행 래퍼.
# 카메라(camera_debug)를 백그라운드로 켜고, gp8_manager 앱을 robust_throw로 돌린다.
#   - move_group(MoveIt)은 쓰지 않는다: 앱이 MoveItController를 호출하지 않으므로
#     GP8_USE_MOVEIT 미설정이면 생성 자체를 건너뛴다 (moveit:=false).
#   - 벨트 속도(/conveyor/speed)는 엔코더가 이미 떠 있으면 쓰고, 없으면 앱이
#     하드코딩 속도로 폴백한다. 엔코더를 이 스크립트로 같이 켜려면 START_ENCODER=1.
#
# 사용:
#   ./run_conveyor_robust_throw.sh                       # 카메라 + 앱(robust_throw)
#   ./run_conveyor_robust_throw.sh axis_increment_factor:=0.3   # 저속 첫 시험
#   START_ENCODER=1 ./run_conveyor_robust_throw.sh       # 엔코더도 함께 기동
#   GP8_PERCEPTION_URL=http://<ip>:8080/detections/stream ./run_conveyor_robust_throw.sh
#
# ⚠️ 실제 풀 스윙 던지기 — 로봇 반경 1 m 확보. 펜던트 E-stop 해제/알람 리셋/REMOTE.
#    카메라 PC의 인지 스트림이 송출 중이어야 물체를 집는다.
set -eo pipefail   # -u는 ROS setup.bash가 미정의 변수를 참조해서 쓰지 않는다

SKILL="${SKILL:-robust_throw}"
START_ENCODER="${START_ENCODER:-0}"   # 1이면 esp32_encoder도 백그라운드로 기동
WS="$HOME/ros2_ws"

BG_PIDS=()
cleanup() { for pid in "${BG_PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT

echo "=== [1/3] source ROS + build gp8_control ==="
source /opt/ros/humble/setup.bash
cd "$WS"
colcon build --packages-select gp8_control
source "$WS/install/setup.bash"

echo "=== [2/3] camera_debug (+ 엔코더) 백그라운드 기동 ==="
if [ "$START_ENCODER" = "1" ]; then
    echo "엔코더: esp32_encoder conveyor_node (log: /tmp/gp8_encoder.log)"
    ros2 run esp32_encoder conveyor_node >/tmp/gp8_encoder.log 2>&1 &
    BG_PIDS+=($!)
fi
echo "카메라: ros2 run gp8_control camera_debug (log: /tmp/gp8_camera_debug.log)"
ros2 run gp8_control camera_debug >/tmp/gp8_camera_debug.log 2>&1 &
BG_PIDS+=($!)
sleep 2   # 토픽이 뜰 시간

echo "=== [3/3] bringup + app (skill:=$SKILL, moveit 미사용) ==="
echo "물체를 벨트에 올리면 순서대로 집어 던진다. 중지: Ctrl-C (카메라도 함께 정리)."
# 앱+드라이버 foreground. moveit:=false + GP8_USE_MOVEIT 미설정 → move_group 없이
# 30초 대기도 없이 부팅. CLI 인자("$@")는 launch로 전달(예: axis_increment_factor:=0.3).
ros2 launch gp8_control gp8_bringup.launch.py \
    app:=true skill:="$SKILL" moveit:=false "$@"

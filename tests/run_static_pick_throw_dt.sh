#!/usr/bin/env bash
# 정적(멈춘 벨트) 연속 pick-and-throw 테스트 — **Thr_DT Decision Transformer** 판.
# run_static_pick_throw.sh(NLP planner) 의 _dt 변형. ROS 소싱 → colcon build →
# driver 스택 bringup(필요 시) → 테스트 실행 까지 구조는 동일하다.
#
# ── 설정은 아래 POINTS / TARGETS / EXTRA_ARGS 배열을 편집한다 ──
# 사용:
#   ./run_static_pick_throw_dt.sh --dt-scan    # ★ 먼저 이걸로 실현 가능성 확인 (로봇 무명령)
#   ./run_static_pick_throw_dt.sh --plan-only  # 계획+궤적조립 검증 (driver 불필요)
#   ./run_static_pick_throw_dt.sh              # 배열값으로 실기 모드 (driver 자동 기동)
#   ./run_static_pick_throw_dt.sh --weights .../dt_best_k0.pth
#   ./run_static_pick_throw_dt.sh --target "0.9,0,0"   # CLI 가 배열값을 덮어씀
#
# driver 스택: 이미 떠 있으면 재사용, 없으면 이 스크립트가
#   ros2 launch gp8_control gp8_bringup.launch.py app:=false moveit:=false
# 를 백그라운드로 띄운다 (로그: $DRIVER_LOG). KEEP_DRIVER=0 이면 종료 시 정리.
# 전제: 펜던트 E-stop 해제 + 알람 리셋 + REMOTE (아니면 activation 거부 — CODE 10x).
set -eo pipefail   # -u는 ROS setup.bash가 미정의 변수를 참조해서 쓰지 않는다

# ============================================================================
# ⚠ NLP 판과 --target 의 의미가 다르다 — 반드시 읽을 것
# ============================================================================
# DT 는 목표를 **거리 하나**로만 받는다 (논문 §5.1: 상태에 d_g 를 붙이고, 던지는
# 방향은 J1(S)이 해석적으로 잡는다). 그래서:
#
#   TARGETS 의 x,y  →  d_g = hypot(x,y) 와 조준 yaw = atan2(y,x) 로만 쓰인다
#   TARGETS 의 z    →  DT 에는 안 들어간다. **착지 예측 평면**으로만 쓴다
#                      (Thr_DT 는 물체가 base frame z≈0 에 떨어진다고 보고 학습됐다)
#
# 그리고 **DT 궤적은 대부분의 목표에서 GP8 관절 한계를 벗어난다**: Thr_DT 평면
# 시뮬레이터에는 관절 위치 한계가 없어서(속도만 clip) 학습된 정책이 먼 목표에서
# U(J3)를 −185°…−260° 까지 뒤로 감는 백스윙을 쓴다. 실제 GP8 J3 는 [−70°,+190°]
# 이고 그 자세는 TCP 를 x≈−0.27 m (기둥/베이스 안쪽)로 집어넣는다.
#
#   사전 스캔 (d_g 0.50→2.00 m, 0.05 간격 31개, 관절한계+Cartesian 엔벨로프):
#     weights/dt_best.pth      12/31 통과 — d_g 0.50 … 1.05 m   ← 기본값
#     weights/dt_best_k0.pth    8/31 통과 — d_g 0.50 … 0.85 m
#       (k0 는 Thr_DT 자체 30-goal 평가에서는 최고(27.8 cm)지만, 그 성능을 내는
#        큰 백스윙이 GP8 에 안 들어간다 — 실행 가능 범위가 더 좁다)
#
# 계획 단계가 불가능한 target 을 **거부**하므로 로봇은 그 지점을 집지도 않는다
# (static_pick_throw 의 사이클 단위 SKIP 과 동일). 그래도 실기 전에 `--dt-scan`
# 으로 먼저 확인하는 것이 빠르다.
# ============================================================================

# ============================================================================
# 물체 지점 목록 (base frame "x,y,z" [m]) — 한 줄에 하나씩.
#   z = 프레스가 내려가는 최종 TCP 높이 (오프셋 없이 그대로 적용).
#   물체(place)는 z + 0.05m 위에 놓인 것으로 간주 (static_pick_throw.py 의
#   PLACE_ABOVE_PRESS). "x,y"로 z 를 생략하면 Config.GRASP_Z.
#
#   ※ NLP 판과 달리 **어디서 집었는지는 던지기에 영향이 없다** — DT 스윙은 항상
#     학습 시 home 자세에서 출발한다. 지점은 "무엇을 집을지"만 정한다.
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
# 던지기 목표 ("x,y,z" [m]) — 1개만 쓰면 전 지점 공통, 여러 개면 지점 수와 같아야
# 하고 순서대로 1:1 매칭된다.
#
# 기본값은 dt_best.pth 로 실현 가능한 구간(d_g 0.50…1.05 m) 안에서 고른 6개다.
# z=0 은 Thr_DT 가 가정하는 착지면(로봇 base 평면)이다 — bin 바닥이 z=−0.08 이면
# 그 값을 주면 착지 예측이 그 높이 기준으로 바뀐다 (DT 의 조준 자체는 안 바뀐다).
# ============================================================================
TARGETS=(
    "0.60,0.10,0.0"
    "0.75,0.0,0.0"
    "0.90,-0.10,0.0"

    "0.95,0.10,0.0"
    "1.00,0.0,0.0"
    "1.05,-0.10,0.0"
)

# 항상 붙는 추가 옵션.
# --confirm-throw : 흡착+유지 후 Enter 를 눌러야 던진다 (q 입력 시 그 사이클 생략)
# --vel-scale     : 던지기 외 이동 속도 (DT 스윙 아크와 감속 꼬리는 영향 없음 —
#                   아크는 10 Hz 정책이 정하고, 감속은 로봇 한계로 만든다)
# --shuffle-points: 지점 방문 순서 무작위 (TARGETS 순서는 고정)
# --no-confirm    : 사이클마다 Enter 확인 생략
EXTRA_ARGS=("--confirm-throw" "--vel-scale" "0.2")

# driver 스택: 1 = 스크립트 종료 후에도 유지(재실행 빠름), 0 = 종료 시 함께 정리
KEEP_DRIVER=1

# ============================================================================
WS="$HOME/ros2_ws"
PKG_DIR="$WS/src/gp8_control"
VENV_PY="$PKG_DIR/.venv/bin/python"
DRIVER_LOG="/tmp/gp8_driver_bringup.log"
JTC_ACTION="/joint_trajectory_controller/follow_joint_trajectory"

# ── DT 설정 ──────────────────────────────────────────────────────────────────
# 가중치. skills/thr_dt/weights/ 에 3개가 벤더링돼 있다 (거기 README 참고):
#   dt_best.pth             기본 — GP8 실행 가능 d_g 0.50…1.05 m
#   dt_best_k0.pth          Thr_DT 30-goal 최고지만 GP8 가능 범위 0.50…0.85 m
#   dt_finetuned_real-5.pth sim2real 파인튜닝 데모 산출물 (교란 시뮬 기준)
export GP8_DT_WEIGHTS="${GP8_DT_WEIGHTS:-$PKG_DIR/skills/thr_dt/weights/dt_best.pth}"
# 조건화 리턴 R̂ (논문 §4.3 / repo evaluate_dt.py: 성공한 던지기 = 1)
export GP8_DT_TARGET_RETURN="${GP8_DT_TARGET_RETURN:-1.0}"
# 착탄 게이트 [m]: 실제 GP8 형상 기준 예측 착지가 목표에서 이만큼 넘게 벗어나면
# 그 target 을 거부. DT 는 착탄을 명시적으로 푸는 solver 가 아니라 정책이므로
# NLP 의 30 mm 같은 값으로 조이면 아무것도 통과하지 못한다. 실제 오차는 항상
# 로그(`J=`)에 찍히므로, 여기서는 "대충 목표 방향으로 간다"만 거른다.
export GP8_DT_LAND_GATE="${GP8_DT_LAND_GATE:-0.50}"
#
# Cartesian 안전 엔벨로프는 NLP 판과 같은 상수를 공유한다 (robust_throw_skill 의
# MIN_TCP_X 0.20 / MIN_TCP_Z 0.04 / MAX_TCP_Z). max-z 만 env 로 조절 가능.
export GP8_MAX_TCP_Z="${GP8_MAX_TCP_Z:-0.85}"
# 그리퍼 로드 오프셋 [m] — 발사점(흡착 물체 CoM 근사). throwing.launch_state 가
# 이 값으로 착지를 예측한다. NLP 판과 동일하게 2 cm.
export GP8_GRIP_OFF="${GP8_GRIP_OFF:-0.02}"
# release 명령 선행 시간 [s]. 양수 = 그만큼 일찍 명령(밸브 지연 보정), 음수 = 늦게.
# ⚠ NLP 판에는 t*±25 ms 릴리즈 윈도우가 있어 jitter 가 흡수되지만 **DT 는 릴리즈가
#   한 점**이다. 게다가 DT 아크는 릴리즈에서 끝나므로, 음수 lead 는 명령을 감속
#   구간으로 밀어 물체가 계획보다 짧게 날아간다 (스킬이 경고한다). 밸브 지연을
#   실측했다면 그 값을 양수로 넣고, 아니면 0 으로 둔다.
export GP8_RELEASE_LEAD="${GP8_RELEASE_LEAD:-0.0}"
#
# ⚠ NLP 판의 warm DB(GP8_THROW_WARM_DB) / 관절한계 조임(GP8_L_LO_DEG,
#   GP8_U_LO_DEG)은 **쓰지 않는다** — 그건 throw_nlp 의 공식화 파라미터이고 DT
#   경로에는 solver 가 없다. 여기서 export 하지 않는 이유가 그것이다.

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

# driver 가 필요 없는 모드 (계획/스캔만)
NO_ROBOT=0
[[ " ${ARGS[*]} " == *" --plan-only "* ]] && NO_ROBOT=1
[[ " ${ARGS[*]} " == *" --dt-scan "* ]] && NO_ROBOT=1

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
    echo "plan-only / dt-scan — driver 불필요, 기동 생략"
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
echo "target: $TARGET_ARG   (x,y → 거리 d_g + 조준 yaw / z → 착지 예측 평면)"
[ "$#" -gt 0 ] && echo "cli   : $*"

echo "=== [4/4] run static_pick_throw_dt (Decision Transformer) ==="
echo "weights : $GP8_DT_WEIGHTS"
[ -f "$GP8_DT_WEIGHTS" ] || { echo "  ERROR: DT 가중치 파일 없음" >&2; exit 1; }
echo "R̂       : $GP8_DT_TARGET_RETURN   (조건화 목표 리턴)"
echo "착탄게이트: $GP8_DT_LAND_GATE m (realGP8 예측 착지 오차 상한)"
echo "max-z   : GP8_MAX_TCP_Z=$GP8_MAX_TCP_Z"
echo "grip off: GP8_GRIP_OFF=$GP8_GRIP_OFF m (발사점 = TCP + 로드축 이만큼)"
echo "release : GP8_RELEASE_LEAD=$GP8_RELEASE_LEAD s (+일찍 −늦게; DT 는 릴리즈 윈도우 없음)"
# 배열값 먼저 + CLI 뒤 — argparse 는 같은 옵션 중복 시 마지막 것을 쓰므로
# CLI --points/--target 이 배열값을 덮어쓰고, 플래그는 그대로 추가된다.
env PYTHONPATH="$WS/src:${PYTHONPATH:-}" \
    "$VENV_PY" -m gp8_control.tests.static_pick_throw_dt "${ARGS[@]}"

# gp8_control

ROS 2 (Humble) runtime control package for the Yaskawa **GP8** 6-DOF arm
running MotoROS2 firmware. Originally part of
[iitp_robot_control](https://github.com/f56e751/iitp_robot_control); split out
into its own repository so the production control stack is decoupled from
the RL training stack.

## What's inside

| Component | Purpose |
|---|---|
| `app.py` | Main pick-and-throw orchestrator (`gp8_manager` node) — tracked-objects queue, conveyor-compensated IK, throw trajectory, torch NN release-param predictor. Ported from `main_sam7.py`. |
| `bridge.py` | MotoROS2 ↔ URDF joint-name bridge. Renames `joint_1..6` ↔ `joint_1_s..joint_6_t` on `/joint_states` and proxies `/follow_joint_trajectory`. |
| `robots/gp8.py` | GP8 kinematics (PoE screw axes, FK/IK, Jacobian). |
| `trajectory/predictor.py` | In-process torch FCN loader for throw trajectory parameters. |
| `trajectory/trajectory_primitive.py` | `opt_time`, `trajectory_3points`, `new_trajectory`, etc. |
| `controllers/trajectory_controller.py` | FJT action client with suction release-on-pass logic. |
| `controllers/moveit_controller.py` | MoveIt 2 wrapper (used only for initial-pose planning). |
| `perception/` | SAM ZMQ client, AprilTag subscriber, RealSense camera manager, calibrated CameraInfo publisher. |
| `mock/mock_robot.py` | Fake MotoROS2 for dev work without the physical robot. |
| `gui/` | Flask-based web GUI for manual EE jogging and status. |
| `launch/gp8_bringup.launch.py` | Full bringup — bridge, robot_state_publisher, MoveIt, RealSense, AprilTag, `gp8_manager`. |
| `launch/debug_robot.launch.py` | Minimal bringup (bridge + TF + MoveIt) for interactive scripts. |
| `terminal_debug.py` | 키보드 기반 EE jog / 회전 / home / 석션 / Queue Mode sweep / FJT mismatch 테스트 도구. |
| `tests/queue_test.py` | TrajectoryController Queue 메서드 3가지 시나리오 분리 검증. |
| `tests/queue_test_throw.py` | torch NN throw trajectory + pick→throw 연속 테스트. |
| `model/NN_newprimitive2_…pt` | Bundled throw-trajectory FCN weight (1.1 MB). |
| `config/` | RealSense calibration, AprilTag settings, robot.yaml. |

## Prerequisites

- ROS 2 Humble on Ubuntu 22.04
- MotoROS2 running on the YRC1000micro controller
- micro-ROS Agent reachable at `192.168.255.5:8888` (or wherever MotoROS2
  is configured)
- `esp32_encoder` ROS 2 node publishing `/conveyor/speed` (optional, falls
  back to hardcoded speed)
- Remote SAM inference server reachable — set its address via env vars
  before launching:
  ```bash
  export GP8_SAM_SERVER_IP=<your-sam-server-ip>
  export GP8_SAM_SERVER_PORT=7150   # default
  ```
  (The public GitHub copy of this repo intentionally ships with
  `127.0.0.1` as a placeholder so no internal infra IPs leak.)
- [`uv`](https://astral.sh/uv) for Python venv management
- ESP32 conveyor encoder on `/dev/ttyUSB0` (user in `dialout` group)

## Setup

```bash
# Inside this package
cd ~/ros2_ws/src/gp8_control
uv sync                              # creates .venv with torch, scipy, etc.

# Build the ROS 2 package
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select gp8_control
```

## Launch

Needs **two terminals** (plus whatever is already running MotoROS2 /
micro-ROS Agent / Docker).

### Terminal 1 — bringup (bridge + cameras + MoveIt + gp8_manager)

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch gp8_control gp8_bringup.launch.py
```

`gp8_manager` is launched via an `ExecuteProcess` wrapper that uses this
package's `.venv/bin/python` (found by walking up from the launch file) so
that torch is available. Override the venv location with
`GP8_VENV_PYTHON=/path/to/python` if needed.

### Terminal 2 — conveyor encoder (publishes `/conveyor/speed`)

Lives in a sibling package:
<https://github.com/f56e751/esp32_encoder>

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 run esp32_encoder conveyor_node
```

First time only — add the user to `dialout` so the ESP32 USB-UART
(`/dev/ttyUSB0`) is accessible, then **log out and back in** (group
changes don't apply to already-running shells):

```bash
sudo usermod -aG dialout $USER
```

The encoder node prints a live TUI with belt speed (m/s) and cumulative
distance, and publishes `/conveyor/speed` (`std_msgs/Float64`) which
`gp8_manager` subscribes to for conveyor-compensated picking.

If the encoder node is *not* running, `gp8_manager` falls back to
`Config.CONVEYOR_SPEED` (hardcoded) and logs a warning — picking still
works but is less accurate when the belt speed drifts.

### micro-ROS Agent (once per boot)

Needed to bridge MotoROS2 ↔ ROS 2. Without it, `/write_single_io` /
`/start_traj_mode` never appear and nodes hang on *"Waiting for …"*.

```bash
# 1. Already running?
sudo docker ps | grep microros_agent

# 2. Start (keep on one line — trailing spaces after \ will break it)
sudo docker run -d --rm --net=host --name microros_agent microros/micro-ros-agent:humble udp4 --port 8888 -v6

# 3. Verify
sudo docker logs microros_agent 2>&1 | grep -i session   # "session established"
ros2 service list | grep write_single_io
```

Pendant must be in **REMOTE + AUTO** with no alarms.

## Debug / Test 도구

SAM/camera/conveyor 전체 스택 없이 **로봇 동작과 Queue Mode 경로**만 검증하고 싶을 때.

### 공통 사전 작업

**터미널 A — 경량 bringup** (bridge + TF + MoveIt):

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch gp8_control debug_robot.launch.py
```

### 1. `terminal_debug` — 키보드 jog & Queue sweep

가장 상위 수준의 대화형 디버그 툴. 실제 로봇을 키로 조작 + 여러 검증 기능.

```bash
ros2 run gp8_control terminal_debug
```

주요 키:

| 키 | 동작 |
|---|---|
| `w/a/s/d/r/f` | EE ±1cm X/Y/Z jog (Shift = ±5cm) |
| `u/j/i/k/o/l` | Roll/Pitch/Yaw ±5° (Shift = ±15°) |
| `g` | 석션 그리퍼 수직 정렬 |
| `Space` | 석션 ON/OFF 토글 |
| `z` | 최저점으로 최대속도 하강 |
| `x` | SAFE_HEIGHT (10cm) 로 상승 |
| `h` | Home pose 복귀 |
| `t` | [TEST] Queue Mode X sweep ±3cm (2s) |
| `b` | [TEST] FJT start-state mismatch 재현 (204 reject 확인) |
| `p` | 현재 상태 출력 |
| `q` | 종료 |

### 2. `queue_test` — Queue 메서드 단위 검증

`TrajectoryController`의 Queue 메서드를 3가지 시나리오로 분리 검증. torch 불필요.

```bash
ros2 run gp8_control queue_test
```

메뉴:
- `1` 단순 2점 이동 — `send_trajectory_queue` (like `_move_to_initial_pose`)
- `2` 3점 경로 (pick 유사) — `send_trajectory_queue` (like `_execute_pick`)
- `3` release_joint 테스트 — `send_trajectory_queue_with_release` (like `_execute_transfer`)
- `q` 종료 (자동으로 FJT 모드 복귀)

### 3. `queue_test_throw` — torch NN throw 테스트

`_execute_transfer` 와 동일 경로 재현: FCN 추론 → `new_trajectory` 합성 → Queue 전송. torch가 필요하므로 **venv python** 으로 실행.

```bash
PYTHONPATH=$HOME/ros2_ws/src:$PYTHONPATH ~/ros2_ws/src/gp8_control/.venv/bin/python -m gp8_control.tests.queue_test_throw
```

메뉴:
- `t` 제자리 throw 스윙 (grasp == aim, NN arc만 관찰)
- `p` **pick → throw 연속** (두 trajectory back-to-back, gap 측정)
- `q` 종료

스크립트 상단 상수로 조정 가능:
```python
TARGET_XY = (0.7, 0.0)              # NN 의 aim 위치 입력
TARGET_DISTANCE = 0.5               # throw 거리
PICK_APPROACH_OFFSET = (-0.2, 0, 0) # pick approach (dx, dy, dz) in meters
```

⚠️ **throw 는 스윙 모션**. 주변 1m 이상 비우고 실행. y/N 확인 단계에서 중단 가능.

### 도구 선택 가이드

| 목적 | 도구 |
|---|---|
| 로봇 상태 확인, 수동 jog | `terminal_debug` |
| FJT vs Queue Mode 동작 비교 | `terminal_debug` → `t`/`b` 키 |
| `TrajectoryController` Queue 메서드 버그 재현 | `queue_test` |
| throw NN 추론 결과 확인 | `queue_test_throw` → `t` |
| pick↔throw 연속 끊김 측정 | `queue_test_throw` → `p` |
| 전체 pipeline 통합 | `ros2 launch gp8_control gp8_bringup.launch.py` |

## Topology

```
MotoROS2 (on YRC1000micro)  ─ /joint_states (joint_1..6, BEST_EFFORT)
                            ─ /follow_joint_trajectory  (joint_1..6)
                            ─ /write_single_io, /read_single_io, /start_traj_mode
          │
          │  (ROS 2 DDS over 192.168.255.x)
          ▼
  motoros2_name_bridge  ─ /joint_states_urdf  (joint_1_s..6_t)
                       ─ /motoman_gp8_controller/follow_joint_trajectory
          │
          ▼
  gp8_manager (app.py)
   ├─ /joint_states_urdf      (subscribe)
   ├─ /conveyor/speed          (subscribe — esp32_encoder)
   ├─ SAM ZMQ client → remote GPU
   └─ FJT action → bridge → MotoROS2
```

## Status

Production-tested on YRC1000micro + MotoROS2 0.2.1 + Humble as of
2026-04. Pick timing (`fixed_delay`) is learned adaptively per run.

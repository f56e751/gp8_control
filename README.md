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
| `camera_debug.py` | Perception node (`camera_debug`) — reads the camera PC's HTTP NDJSON stream, applies the camera→base transform + `v×delay` back-projection, and publishes corrected detections on `/camera_debug/detections`. **Must run for `app.py` to pick.** |
| `perception/` | Supports `camera_debug`: HTTP stream client (`perception_client`), camera/base extrinsics (`extrinsics`), and the control-side detection intake/dedup (`detection_intake`, consumed by `app.py`). |
| `conveyor/` | `ConveyorSpeedTracker` — subscribes `/conveyor/speed` (encoder node) and exposes the live belt speed to the app + skills. |
| `mock/mock_robot.py` | Fake MotoROS2 (incl. Point Queue Mode + real-time playback) for dev/sim without the physical robot. |
| `mock/fake_belt.py` | Fake `camera_debug` for simulation — spawns objects on the belt, publishes `/camera_debug/detections` + `/conveyor/speed`. |
| `gui/` | Flask-based web GUI for manual EE jogging and status. |
| `launch/gp8_bringup.launch.py` | Full bringup — bridge, robot_state_publisher, MoveIt, `gp8_manager`. **Does NOT start `camera_debug`** — run that separately. |
| `launch/sim_bringup.launch.py` | Software-in-the-loop sim — `mock_robot` + `fake_belt` + RSP + MoveIt + RViz + the app (no hardware). |
| `launch/debug_robot.launch.py` | Minimal bringup (bridge + TF + MoveIt) for interactive scripts. |
| `belt_viz.py` | TUI rendering the live belt — every tracked object (`●`), the active target (`◉`), and app status — from `/gp8_manager/tracked_state` (published by `app.py`). Works in real or sim. |
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
- Camera PC's HTTP detection stream reachable. The **`camera_debug` node**
  (not `app.py`) reads it — set the URL in the terminal that runs
  `camera_debug`:
  ```bash
  export GP8_PERCEPTION_URL=http://<camera-pc-ip>:8080/detections/stream
  ```
  Perception (RealSense + SAM/DINO) runs on the camera PC; `camera_debug`
  consumes the NDJSON stream, applies the camera→base transform, and
  republishes `/camera_debug/detections`, which `app.py` subscribes to. The
  public GitHub copy ships with a placeholder URL so no internal infra IPs leak.

  > **`camera_debug` 가 떠 있어야 picking이 동작합니다.** `app.py` 는
  > `/camera_debug/detections` 만 구독하므로, `camera_debug` 가 없으면 노드는
  > 떠도 탐지가 비어 **물체를 못 집습니다.** (앱 단독 기동/디버깅은 카메라
  > 없이도 됨.) `camera_debug` 는 송출이 꺼져 있어도 죽지 않고 2초마다 재접속.
  >
  > 스트림 도달 여부 확인 (camera_debug 띄우는 PC에서):
  > ```bash
  > curl -N "$GP8_PERCEPTION_URL" | head
  > ```
  > JSON 라인이 흐르면 OK. 송출 서버 코드는 이 repo가 아니라 카메라 PC에 있는
  > 별도 코드이며, 스트림 와이어 규약은 `perception/perception_client.py` 참고.
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

### Quick Start (순서 중요)

1. **micro-ROS Agent (docker)** — 부팅당 1회. 없으면 `/write_single_io` /
   `/start_traj_mode` 가 안 뜨고 노드가 *"Waiting for …"* 에서 멈춤.
2. **Terminal 1 — bringup** (bridge + robot_state_publisher + MoveIt + gp8_manager)
3. **Terminal 2 — `camera_debug`** (`/camera_debug/detections` 발행) —
   **picking 하려면 필수.** bringup 에 포함되지 않으니 따로 띄움.
4. **Terminal 3 — conveyor encoder** (`/conveyor/speed` 발행; camera_debug 의
   back-projection + 앱 속도보정에 쓰임, 선택이지만 권장)

펜던트는 **REMOTE + AUTO**, 알람 없는 상태여야 함. 자세한 명령은 아래 각
섹션 참고.

### Terminal 1 — bringup (bridge + robot_state_publisher + MoveIt + gp8_manager)

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch gp8_control gp8_bringup.launch.py
```

`gp8_manager` is launched via an `ExecuteProcess` wrapper that uses this
package's `.venv/bin/python` (found by walking up from the launch file) so
that torch is available. Override the venv location with
`GP8_VENV_PYTHON=/path/to/python` if needed.

### Terminal 2 — `camera_debug` (publishes `/camera_debug/detections`)

`app.py` subscribes to `/camera_debug/detections`; that topic is produced by
the **`camera_debug` node**, which `gp8_bringup` does **not** start. Run it in
its own terminal (on the robot PC) or picking never happens:

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
export GP8_PERCEPTION_URL=http://<camera-pc-ip>:8080/detections/stream
ros2 run gp8_control camera_debug
```

It reads the camera PC's HTTP stream (`GP8_PERCEPTION_URL`), applies the
camera→base transform + `v×delay` back-projection, and publishes the corrected
detections. It also subscribes to `/conveyor/speed` for the back-projection.
A live TUI shows raw vs corrected positions.

### Skill 선택 — throw만 / push만 실행 (디버그)

`gp8_manager` 는 매 객체를 `ActionSelector` 가 push/throw 스킬로 라우팅합니다.
**기본 라우팅은 클래스별**(`Config.SKILL_BY_CLASS`): `metal`(캔) → **push**,
`transparent`(페트병) → **throw**, 그 외 → throw. 디버그·테스트용으로 **모든
객체를 한 스킬로 고정**할 수 있습니다.

| 모드 | 실행 |
|---|---|
| 정상 라우팅 (기본, 클래스별) | `ros2 launch gp8_control gp8_bringup.launch.py` |
| throw 만 | `GP8_FORCE_SKILL=throw ros2 launch gp8_control gp8_bringup.launch.py` |
| push 만 | `GP8_FORCE_SKILL=push ros2 launch gp8_control gp8_bringup.launch.py` |

- 우선순위: CLI `--skill {throw,push}` > 환경변수 `GP8_FORCE_SKILL` > 기본(클래스별 라우팅).
- 기동 로그에 `ActionSelector FORCED to '<skill>' skill for ALL objects` 가 뜨면 강제 모드.
- `push` 는 **실제 접촉 스윕**입니다 (`skills/push_skill.py`) — intercept 에서 대기 후
  벨트면과 평행하게 등속 직선으로 밀어내며 스윙을 줍니다. `metal` 클래스에 한해
  `can_handle()` 이 `True` (정상 라우팅에서 캔이 push 로 감).

bringup 없이 앱만 단독으로 띄울 땐 venv python 으로 플래그를 직접 줄 수 있습니다
(torch 때문에 venv 필요; move_group/bridge 가 없어 실제 picking 은 안 됨):

```bash
PYTHONPATH=$HOME/ros2_ws/src:$PYTHONPATH \
  ~/ros2_ws/src/gp8_control/.venv/bin/python -m gp8_control.app --skill push
```

### Terminal 3 — conveyor encoder (publishes `/conveyor/speed`)

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

**확인 / 가짜 발행 (디버그):**

```bash
ros2 topic echo /conveyor/speed      # 값 흐름 확인
ros2 topic hz   /conveyor/speed      # 발행 주기 확인
# 인코더 없이 가짜 속도로 테스트 (camera_debug/app 단독 점검):
ros2 topic pub /conveyor/speed std_msgs/msg/Float64 "{data: 0.12}" -r 10
```

> 시뮬(`sim_bringup`)에서는 `fake_belt` 가 `/conveyor/speed` 를 직접 발행하므로
> 인코더도 위 가짜 pub 도 필요 없습니다.

### `belt_viz` — 벨트 상태 시각화 (실행 중인 gp8_manager 모니터)

`gp8_manager` 가 발행하는 `/gp8_manager/tracked_state` 를 구독해 벨트를 ASCII
스트립으로 실시간 렌더링합니다 — **무엇이 잡혔는지** 한눈에 봅니다: 추적 물체
(`●`), 현재 잡는 타깃(`◉`), max reach / pick 지점, 상태
(`IDLE`/`POSITIONING`/`WAITING`/`THROWING`/`PUSHING`), 벨트 속도. 메시지 사이에도
belt 속도로 외삽돼 부드럽게 흐릅니다.

```bash
ros2 run gp8_control belt_viz
```

- **`app.py`(gp8_manager)가 떠 있어야** 보입니다 (그 노드가 토픽을 발행). 실로봇·
  시뮬(`sim_bringup`) **둘 다** 동작.
- TUI 라 **SSH 로 그대로** 보입니다 (GUI/RViz 불필요).

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
- `2` 3점 경로 (pick 유사) — `send_trajectory_queue`
- `3` release_joint 테스트 — `send_trajectory_queue_with_release`
- `q` 종료 (자동으로 FJT 모드 복귀)

### 3. `queue_test_throw` — torch NN throw 테스트

throw 경로 재현: FCN 추론 → `new_trajectory` 합성 → Queue 전송 (`ThrowSkill` 과 동일 primitive). torch가 필요하므로 **venv python** 으로 실행.

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
| 전체 pipeline 통합 (실로봇) | `gp8_bringup.launch.py` + `camera_debug` (+ encoder) |
| 전체 pipeline 시뮬 (무하드웨어) | `sim_bringup.launch.py` + `belt_viz` |
| 벨트 위 물체/타깃/상태 보기 | `belt_viz` (실로봇·시뮬 공통) |

## Simulation (SIL — no hardware)

Run the **full pipeline** (detection → intake/dedup → selection → push/throw
routing → skill → motion) with no robot and no camera, by swapping in two fakes:

- `mock_robot` — fakes MotoROS2 incl. **Point Queue Mode** (the path the app
  actually uses), playing queued points back in real time so the app's
  wall-clock timing (eta, suction lead) stays meaningful.
- `fake_belt` — fakes `camera_debug`: spawns objects on the belt and publishes
  `/camera_debug/detections` + `/conveyor/speed`.

```bash
# one terminal — mock_robot + fake_belt + RSP + MoveIt + RViz + app
ros2 launch gp8_control sim_bringup.launch.py
#   tune:  belt_speed:=0.08 spawn_interval:=4.0   |   headless (SSH): rviz:=false
#   one skill:  GP8_FORCE_SKILL=throw ros2 launch gp8_control sim_bringup.launch.py

# another terminal — belt strip TUI (objects, queue, target, status)
ros2 run gp8_control belt_viz
```

Watch the arm intercept belt objects in RViz (3D) and the belt state in
`belt_viz` (TUI — works over SSH; use `rviz:=false`). The real robot can be
powered off; `mock_robot` replaces it. **Don't run `gp8_bringup` at the same
time** — the `/joint_states_urdf` topics would collide.

> **Caveats** — kinematic only (no grasp/throw physics); a "picked" object is
> not removed from the belt (`fake_belt` keeps flowing it until it passes), so
> use this to verify **motion path / interception timing**, not grasp success.
> RViz Fixed Frame defaults to `base_link` — change it if your URDF root differs.

## Topology

```
camera PC (RealSense + SAM/DINO)  ─ HTTP NDJSON detection stream
          │  (GP8_PERCEPTION_URL)
          ▼
  camera_debug  ─ cam→base transform + v×delay back-projection
               ─ /camera_debug/detections   (corrected base-frame poses)
          │                                  (+ subscribes /conveyor/speed)
          ▼
MotoROS2 (on YRC1000micro)  ─ /joint_states (joint_1..6, BEST_EFFORT)
                            ─ /follow_joint_trajectory  (joint_1..6)
                            ─ /write_single_io, /read_single_io, /start_traj_mode
          │
          │  (ROS 2 DDS over 192.168.255.x)
          ▼
  motoros2_name_bridge  ─ /joint_states_urdf  (joint_1_s..6_t)
                       ─ /motoman_gp8_controller/{follow_joint_trajectory,
                          queue_traj_point, start_point_queue_mode, ...}
          │
          ▼
  gp8_manager (app.py)
   ├─ /joint_states_urdf       (subscribe)
   ├─ /conveyor/speed           (subscribe — esp32_encoder)
   ├─ /camera_debug/detections  (subscribe — camera_debug)
   └─ Point Queue Mode → bridge → MotoROS2
```

## Status

Production-tested on YRC1000micro + MotoROS2 0.2.1 + Humble as of
2026-04. Picks via the **ambush** strategy (park at the reachable intercept,
fire suction on arrival); per-class push/throw routing (`metal`→push,
`transparent`→throw).

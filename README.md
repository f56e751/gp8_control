# gp8_control

Yaskawa **GP8** 컨베이어 분류 데모 (ROS 2 Humble). 카메라 PC가 벨트 위 물체를
검출해 보내면, 로봇이 움직이는 벨트에서 물체를 잡아 **던지거나(throw)**
**밀어서(push)** 통에 넣는다. MuJoCo 시뮬레이터와 RL 학습 환경, 실기와 sim의
차이를 재는 검증 도구도 함께 들어 있다.

---

## 실행 (매번)

### 0. 실행 전 확인

- [ ] 펜던트 **REMOTE + AUTO**, 알람 없음
- [ ] 엔코더 24V 전원 ON, ESP32 USB 연결 (`ls /dev/ttyUSB0`)
- [ ] 카메라 PC(robot6) 검출 스트림이 떠 있음 → [카메라 PC](#카메라-pc-robot6)
- [ ] 로봇 주변 1 m 비움 (실제로 집고 던진다)
- [ ] 이전에 띄운 launch / 노드는 모두 Ctrl-C로 종료

### A. 기본 데모 (rule-based app) — 터미널 2개

| 터미널 | 명령 | 하는 일 |
|---|---|---|
| 1 | `gp8_app` | 드라이버 + MoveIt + 벨트 엔코더 + 앱. **3초 뒤 초기 자세로 이동** |
| 2 | `gp8_camera` | 카메라 검출 → 로봇 좌표 (`/camera_debug/detections`) |

### B. RL runner + sim-to-real 검증 기록 — 터미널 3개

| 터미널 | 명령 | 하는 일 |
|---|---|---|
| 1 | `gp8_robot` | 드라이버 + MoveIt + 벨트 엔코더 (로봇은 아직 안 움직임) |
| 2 | `gp8_camera` | 카메라 검출 |
| 3 | `gp8_s2r` | RL runner (지금은 rule-based 정책). **바로 초기 자세로 이동** |

끝나면 터미널 3에서 **Ctrl-C** → 1~2분 뒤 리포트가 생긴다:

```bash
RUN=$(ls -td ~/gp8_s2r_runs/*/ | head -1); cat $RUN/report.md
```

> 명령이 "찾을 수 없음"이면 [처음 한 번 설정](#처음-한-번-설정)의 alias 등록을 먼저.
> alias 없이 직접: `<ws>/src/gp8_control/scripts/{app,robot,camera,s2r_run}.sh`

### 자주 바꾸는 값

명령 앞에 붙이면 그 실행에만 적용된다 (예: `GP8_TRACK_LEAD_T=0.14 gp8_s2r`).

| 변수 | 기본 | 의미 |
|---|---|---|
| `GP8_TRACK_LEAD_T` | `0.12` | throw 집기를 이만큼(초) 일찍 시작. 물체 **뒤**를 잡으면 올리고 **앞**을 잡으면 내린다 |
| `GP8_CAM_SIGN_CX` / `GP8_CAM_SIGN_CY` | `+1` / `-1` | 카메라 축 부호. 카메라가 5월 보정 대비 180° 돌아가 있어 둘 다 반전 (2026-10-19) |
| `GP8_FORCE_SKILL` | (없음) | `throw` / `push` 로 고정 (디버그) |
| `GP8_TRACK_MIN_HITS` | `3` | 이만큼 검출되거나 화면 중앙 확정을 받은 트랙만 집는다 (가짜 검출 차단) |
| `GP8_OBJECT_MAX_FOOTPRINT_M` | `0.30` | 이보다 긴 검출 박스는 무시 (두 물체를 이어 붙인 박스) |
| `GP8_S2R_VALIDATE` | app `0` / runner `1` | 검증 기록 켜기 (`gp8_app s2r_validate:=true` 도 가능) |

---

## 처음 한 번 설정

```bash
# 1) 워크스페이스 (데모용은 ~/iitp_demo_ws)
mkdir -p ~/iitp_demo_ws/src && cd ~/iitp_demo_ws/src
git clone git@github-gp8:f56e751/gp8_control.git
git clone https://github.com/f56e751/esp32_encoder.git
git clone https://github.com/Yaskawa-Global/motoros2_interfaces.git
git clone https://github.com/Yaskawa-Global/motoman_ros2_support_packages.git
#   + motoman_gp8_moveit_config (패키지 폴더 복사)
#   + motoman_gp8_support/urdf/gp8_macro.xacro 의 B축 상한 1.0607 rad (손목판, 셀 전용)

# 2) Python venv (torch / mujoco / scipy — rclpy 는 시스템 ROS 것을 씀)
cd ~/iitp_demo_ws/src/gp8_control && uv sync --extra sim

# 3) 빌드 — 드라이버(adv4ncr motoman_ROS2) 워크스페이스를 underlay 로
scripts/build.sh --symlink-install --packages-skip gp8_control   # 다른 패키지
scripts/build.sh                                                  # gp8_control

# 4) 짧은 명령 등록 (~/.bashrc): gp8_env gp8_build gp8_robot gp8_app gp8_camera gp8_s2r
scripts/install_aliases.sh && source ~/.bashrc
```

- 드라이버 워크스페이스 기본 경로: `~/Desktop/motoman_ros2_safety_review/motoman_ROS2/install`
  (`GP8_DRIVER_WS` 로 변경). 이 드라이버는 컨트롤러와 UDP/TCP로 직접 통신하며
  250 Hz로 관절 위치를 받는다 — micro-ROS Agent는 쓰지 않는다.
- **코드를 고친 뒤에는 `gp8_build`.** gp8_control 은 copy 빌드라 빌드해야 반영된다
  (`--symlink-install` 은 이 패키지 구조에서 실패).
- 다른 워크스페이스가 source 된 터미널에서 colcon 을 직접 돌리지 말 것 — 그
  워크스페이스가 underlay 로 박힌다. `gp8_build` 는 깨끗한 환경에서 빌드한다.

---

## 상태 확인 / 문제 해결

```bash
gp8_env                                          # 이 터미널을 워크스페이스로 (어느 gp8_control 인지 출력)
ros2 topic echo --once /conveyor/speed           # 벨트 속도 (안 나오면 엔코더 24V·USB)
ros2 topic hz /camera_debug/detections           # 카메라 검출
ros2 topic hz /joint_states                      # 로봇 상태 (~250 Hz)
ros2 control list_hardware_components            # MotomanHardware 가 active 인지
```

| 증상 | 원인 / 해결 |
|---|---|
| `gp8_robot` 띄워도 로봇이 안 움직임 | 정상. `gp8_s2r`(또는 `gp8_app`) 이 초기 자세로 보낸다 |
| 로봇이 잠긴 듯 반응 없음 | 이전 드라이버가 남음 → 전부 Ctrl-C, `pkill -f ros2_control_node`, 다시 |
| `fallback 0.083 m/s until first message` 가 계속 | `/conveyor/speed` 없음 → 엔코더 전원·USB, 터미널 1 의 `conveyor_node` 에러 |
| 물체를 안 집음 | 카메라 스트림 끊김 → `ros2 topic hz /camera_debug/detections`, [카메라 PC](#카메라-pc-robot6) |
| **모든 물체를 한참(~2 s) 일찍** 처리 | 카메라 축 방향이 반대 → `gp8_camera` 대시보드 맨 윗줄 **방향:** 이 `FLIPPED`(빨강)인지, 카메라 영상(`http://147.46.175.15:8080/`) 오른쪽 화살표가 `REVERSED` 인지 확인 → `GP8_CAM_SIGN_CY` 부호 |
| 물체 **뒤/앞**을 잡음 (속도에 비례) | `GP8_TRACK_LEAD_T` 조정 (0.02 단위) |
| 처음 몇 cm 일찍 처리 | 카메라 v4(`center_confirmed`) 송출인지 확인 — 로그에 `[track-CONFIRM]` 이 물체마다 떠야 함 |
| 같은 물체를 두 번 / 빈 곳을 집음 | 로그의 `[track-MERGE]`·`[track-OVERSIZE]`, 트랙 수 확인. 카메라 로그를 [replay](#결과--도구) 로 재현 |

---

## 카메라 PC (robot6)

- 접속: `ssh iitp@147.46.175.15`, 코드: `/PublicSSD/iitp/demo/iitp_perception`
  (`master` — schema v4: `track_ids`, `center_radius_m`, `center_confirmed`)
- 실행/재시작: `docker stop iitp_local; cd /PublicSSD/iitp/demo/iitp_perception && ./docker_local.sh`
- 확인: `curl -s http://147.46.175.15:8080/detections/stream | head -c 400` → `"schema_version": 4`
- 모든 검출 기록: `results_local/detections.jsonl` (replay 도구 입력)
- 로봇 쪽 v4(중앙 프레임에서 위치 재고정)는 카메라가 v4 를 보낼 때만 동작한다.

### 카메라 방향 확인 (축 부호)

영상(`http://147.46.175.15:8080/`) 오른쪽에 추적기가 학습한 **벨트 흐름 화살표**가
뜬다. 재시작 직후나 물체가 아직 안 지나갔으면 `FLOW: learning`(학습 중), 벨트가
멈추면 `FLOW: stopped`. 물체가 한두 개 지나가면 화살표로 바뀐다.

| 영상 화살표 | 의미 | 실행 |
|---|---|---|
| **DOWN (+cy)**, 주황 `REVERSED` (2026-10-19 현재 설치) | 카메라가 180° 돌아가 있음 | 그냥 `gp8_camera` (기본 `CX=+1`, `CY=-1`) |
| **UP (−cy)**, 초록 `OK` (원래 방향으로 다시 단 경우) | 부호 반전 불필요 | `GP8_CAM_SIGN_CX=-1 GP8_CAM_SIGN_CY=+1 gp8_camera` |

- **최종 확인**: `gp8_camera` 대시보드 맨 윗줄 `방향:` 이 초록 `OK` 면 맞다.
  빨강 `FLIPPED` 면 `GP8_CAM_SIGN_CY` 부호를 반대로.
- **X축은 판정 불가** (물체가 옆으로 안 움직임): 기본값은 180° 회전을 가정해 X·Y
  둘 다 반전한다. 카메라를 다시 달았다면 벨트 한쪽 끝에 물체를 올려 로봇이 같은
  쪽으로 가는지 확인하고, 반대면 `GP8_CAM_SIGN_CX` 부호를 바꾼다.

---

## 결과 / 도구

| 도구 | 명령 (`gp8_env` 후, 저장소 폴더에서) |
|---|---|
| sim-to-real 리포트 다시 만들기 | `.venv/bin/python -m gp8_control.validation.analyze <run_dir> [--sim-steps 40]` |
| 카메라 로그로 트래커 재현 (중복·가짜 트랙 수) | `.venv/bin/python -m gp8_control.tools.replay_detection_log detections.jsonl --sign-cx 1 --sign-cy -1 [--steps <run>/steps.jsonl]` |
| 로봇 없이 검증 파이프라인 확인 | `.venv/bin/python -m gp8_control.validation.sim_reference /tmp/x --steps 10 --source sim_as_hw` → `analyze /tmp/x` |
| MuJoCo 스모크 테스트 | `.venv/bin/python -m gp8_control.tests.sim_smoke [--viewer]` |
| MuJoCo 시뮬레이터로 앱 실행 | `ros2 launch gp8_control sim.launch.py [viewer:=true]` |

### RL 학습 — 실기형 sim 설정 (`GP8_SIM_PROFILE=hw2026`)

2026-10-19 실기 측정값에 맞춘 학습 환경: 벨트 0.10–0.30 m/s, 유입 0.2–0.7 Hz,
실기 관측 크기·레인·축정렬 bbox, 카메라형 관측 노이즈(위치 치우침·흔들림, 실기
confidence 분포), 결정→모션 지연(중앙 40 ms). 관측 차원(91)은 그대로.
기존 기본 설정은 바뀌지 않으며, 명시적으로 export 한 `GP8_SIM_*` 값이 우선한다.
정의: `rl/sim_profile.py`. 실기 runner 는 학습과 같은 240 s 반복 에피소드 시계로
`remaining_time_frac` 을 넣는다 (`gp8_s2r --episode-seconds 0` 이면 예전처럼 1.0).

```bash
uv sync --extra sim --extra rl        # 한 번
cd <ws>/src && GP8_SIM_PROFILE=hw2026 GP8_SIM_PHYSICAL_BELT=true GP8_SIM_BELT_CONTACT_PARAMS=old \
  gp8_control/.venv/bin/python -m gp8_control.tools.train_ppo_smoke \
  --out-dir gp8_control/runs/hw2026_100k --timesteps 100000 --n-envs 16 --n-steps 16 \
  --batch-size 128 --max-episode-seconds 240 --bbox-observation corners --truth-tracks --device auto
```

검증 리포트가 보는 것 (`validation/`): 250 Hz 명령 대비 실제 추종(같은 명령을
MuJoCo 팔에 재생해 HW vs sim 비교), 결정→첫 모션 지연, 석션 IO, 그리고 RL
관측값(91차원)의 채널별 정확도와 sim 분포 비교. 자세한 건 `validation/README.md`.

---

## 코드 구조

| 경로 | 내용 |
|---|---|
| `app.py` | 앱 (`gp8_manager`): 검출 수신 → 집을 물체 선택 → 스킬 실행 |
| `skills/` | `ThrowSkill`, `PushSkill` 와 공용 `SkillContext` (조작 로직은 전부 여기) |
| `planning/action_selector.py` | 물체 → 스킬 라우팅 (`metal`→push, `transparent`→throw). RL 정책이 대체할 자리 |
| `perception/` | `camera_debug` 보조: 스트림 클라이언트, 외부 보정(`extrinsics`), 검출↔트랙 연관(`detection_intake`) |
| `camera_debug.py` | 카메라 스트림 → 로봇 좌표 변환 + 지연 보정 → `/camera_debug/detections` |
| `tracking/object_queue.py` | 트랙 큐, 중복 병합, 선택 가능 판정(`is_established`) |
| `backends/` | 로봇/월드 seam. `robot_base.py`(250 Hz 스트림 엔진, HW·sim 공용), `ros_world.py`(HW), `mujoco_sim.py`(MuJoCo twin) |
| `controllers/trajectory_controller.py` | HW 백엔드: JointGroupPositionController 250 Hz 스트림 + 석션 TCP IO |
| `rl/` | RL 환경(`gym_env`, `sim_runner`), 실기 runner(`real_runner`), 관측 스키마(`common`) |
| `validation/` | sim-to-real 기록기·분석기 |
| `tools/replay_detection_log.py` | 카메라 로그 → 트래커 오프라인 재현 |
| `scripts/` | 실행 스크립트 (위 `gp8_*` 명령) |
| `launch/` | `gp8_bringup`(앱 포함), `debug_robot`(앱 없음), `sim`(MuJoCo) |
| `model/` | throw 궤적 NN 가중치 |

`CLAUDE.md` 에 파일 간 계약과 설계 의도가 더 자세히 있다.

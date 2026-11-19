# cycle 모드 실기 실행 (robot6 계획 + 로봇 PC 컨트롤러 버퍼 송신)

플래너: 팀원 `grasp_and_throw_20261119` (11-19 실물 설정 — 통 입구 −0.11 m, 벨트·통 거리별 모델 묶음 8 개,
릴리즈 창 명령 맞춤). 기본 묶음 `zm011_z` 는 물체 윗면 높이를 입력으로 쓰므로 `GP8_CYCLE_TOP_Z` 가 맞아야 한다.

구성:
- **로봇 PC** (`~/iitp_demo_ws`, 브랜치 `feature/cycle-throw`): 드라이버, 카메라 입력, 앱, 석션.
  명령은 **SampleBufferController** (드라이버 `motoman_controllers/SampleBuffer`) 버퍼에 **틱 번호로** 미리 쓴다.
  컨트롤러가 자기 2 ms 틱마다 하나씩 내보낸다 (팀원 11-04 실기 검증 경로와 같다).
- **robot6** (`/PublicSSD/iitp/demo/gp8_control`, 같은 브랜치): 사이클 플래너. 로봇 PC 앱이 ssh 로 띄운다.
- 금속(push 대상)은 플래너에 넘기지 않는다. 플래너가 대기 자세에서 쉬는 구간에 앱의 PushSkill 로 민다 (push slot).
  그동안 컨트롤러를 JointGroupPositionController 로 바꿨다가, 대기 자세로 정확히 돌아온 뒤 버퍼로 다시 켠다.

## 0. 한 번만 설정 (2026-11-19 모두 완료)

- 로봇 PC 드라이버 `~/Desktop/motoman_ros2_safety_review/motoman_ROS2`: 팀원 드라이버와 **같게** 맞추고 빌드했다
  (브랜치 `feature/sample-buffer` 1cd66ee + 팀원 작업본 수정 = 500 Hz, nominal_dt 0.002, fault 10 / decel 50).
  이전 상태 백업: `~/iitp_demo_ws/cycle_deploy_20261118/driver_backup/` (패치 + 250 Hz 파일).
- robot6: `/PublicSSD/iitp/demo/gp8_control` (같은 브랜치) + `.venv` (numpy 1.26.4, scipy 1.15.3, casadi 3.8.1)
- 로봇 PC → robot6 키: `~/.ssh/id_ed25519_robot6`, `~/.ssh/config` 의 `robot6-planner` (→ iitp@147.46.175.15).
  robot6 `~iitp/.ssh/authorized_keys` 에 `from="147.46.146.94"` 로 등록됨. 확인: `ssh -o BatchMode=yes robot6-planner hostname`

## 1. 실행 전 확인 (매번)

- [ ] 펜던트 REMOTE + AUTO(PLAY)·서보 ON, 알람 없음. 로봇 주변 1.5 m 비움 (x 1.35 m 까지 휘두른다)
- [ ] 벨트 속도 = `GT_BELT` (기본 **0.12 m/s**). 묶음이 있는 값: 0.10 · 0.12 · 0.14 · 0.16 · 0.18 · 0.20 (실물 확인은 0.10 · 0.12 · 0.14)
- [ ] throw 통: x = `GT_BIN_X` (기본 **1.35 m**; 1.10 · 1.60 도 묶음 있음, 벨트 0.12 일 때만), y **+0.33 금속 / 0 종이 / −0.33 투명**, 입구 높이 **−0.11 m**. push 통 (금속): (0.85, 0.60)
- [ ] robot6 이 한가한지: `ssh robot6-planner uptime` 의 load 가 **2 미만** (팀원 DB 빌드·Isaac 이 돌면 계획이 잘린다)
- [ ] 로봇 없이 점검 (아래 2) 이 PASS

## 2. 로봇 없이 점검 (로봇 PC)

```bash
source ~/iitp_demo_ws/src/gp8_control/scripts/env.sh
PY="env PYTHONPATH=$HOME/iitp_demo_ws/src:$PYTHONPATH $HOME/iitp_demo_ws/src/gp8_control/.venv/bin/python"

# (a) 30 초: robot6 계획 + 가짜 버퍼 컨트롤러 (C++ 과 같은 규칙)
$PY -m gp8_control.tools.cycle_dry_check --host robot6-planner --seconds 30

# (b) 선택: 진짜 ROS + 진짜 SampleBufferController, 모의 하드웨어 (도메인 77 — 실기와 섞이지 않는다)
ROS_DOMAIN_ID=77 ros2 launch motoman_bringup gp8.launch.py use_mock_hardware:=true sample_buffer:=true &   # 다른 터미널 권장
ROS_DOMAIN_ID=77 GP8_ROBOT_IP=127.0.0.1 GP8_STREAM_HZ=500 \
  $PY -m gp8_control.tools.cycle_dry_check --ros --host robot6-planner --seconds 35 --slot-test 12
```
`SUMMARY PASS` 이어야 한다: **late pieces 0, refused 0, ticks vs planned max |diff| 0**, 마감에 잘린 풀이 20 % 이하
(한가한 robot6 에서 약 10 % 가 정상). 2026-11-19 결과: (a) 조각 36 개 수락·거부 0·11,118 틱 일치, (b) 같은 결과 +
push slot (컨트롤러 전환 4 회, 재활성화 뒤에도 틱 일치), 왕복 0.47~0.63 ms.

## 3. 실행

```bash
# 터미널 1: 카메라 (평소와 같다)
gp8_camera

# 터미널 2: 드라이버 + 앱 (cycle 모드)
GP8_MODE=cycle GP8_STREAM_HZ=500 GP8_SAMPLE_BUFFER=true \
GP8_CYCLE_PLANNER_HOST=robot6-planner \
GP8_CYCLE_RELEASE_LEAD=0.2 \
GP8_CYCLE_TOP_Z=0.042 \
gp8_app
```

| 변수 | 값 | 뜻 |
|---|---|---|
| `GP8_MODE=cycle` | | cycle 플래너가 팔을 맡는다 |
| `GP8_STREAM_HZ=500` | | 앱의 일반 스트림 (홈 이동·push) 도 2 ms (드라이버가 500 Hz) |
| `GP8_SAMPLE_BUFFER=true` | | bringup 이 SampleBufferController 를 꺼진 채로 로드 (cycle 모드가 켜고 끈다). 없으면 앱이 시작을 거부한다 |
| `GP8_CYCLE_PLANNER_HOST` | `robot6-planner` | 계획은 robot6 에서. 비우면 로봇 PC 에서 (느려서 안 된다) |
| `GP8_CYCLE_RELEASE_LEAD` | `0.2` | 석션 OFF 를 계획 릴리즈보다 0.2 s 먼저 (팀원 실기 값) |
| `GP8_CYCLE_TOP_Z` | `0.042` | 물체 윗면 = 컵 접촉 높이 [m]. 플래너 기본은 0.02 인데, 우리 throw 가 실기에서 접촉하던 높이는 `GRASP_Z` 0.042 다. **첫 실행에서 확인할 것** (확인된 범위 −0.02 ~ 0.05) |
| `GP8_FORCE_SKILL=throw` | (선택) | push 없이 전부 throw (금속은 +0.33 통). 처음에는 이것부터 |
| `RT_ORDER=fifo` | (선택) | 순서 신경망 대신 가장 하류 물체부터 |
| `GT_BELT=0.16` | (선택) | 다른 벨트 속도용 묶음 (0.10 ~ 0.20). 컨베이어를 먼저 그 속도로 맞춘다. **레포에는 기본 묶음 (`zm011_z`) 만 있고 나머지 7 개는 robot6 체크아웃에만 있다** (계획은 robot6 에서 돌므로 그대로 고를 수 있다; 다른 곳에서 고르면 `[env.sh] no set` 으로 멈춘다) |
| `GT_BIN_X=1.10` | (선택) | 다른 통 거리용 묶음 (1.10 · 1.60, 벨트 0.12 만). 통을 그 거리에 놓는다. 없는 조합이면 앱이 `[env.sh] no set of models ...` 를 띄우고 멈춘다 |
| `GP8_CYCLE_OUTPUT=stream` | (비상용) | 버퍼 대신 앱이 2 ms 마다 직접 송신 (예전 방식) |

기동 로그에서 볼 것:
```
[cycle] planner on robot6-planner:/PublicSSD/iitp/demo/gp8_control
[cycle] clock sync: offset ... s, best round trip <1 ms 안팎>
[buf] controller switch -> SampleBufferController
[cycle] planner started (...); release lead 0.200 s; output SampleBufferController (controller tick)
[cycle] plan s1 cycle 1: 0.4 s (deadline in 1.8 s) ... Solve_Succeeded
```
- `controller refused the piece ... too late / limit` → 버퍼가 멈추고 팔은 그 자리에 선다 (설계상). robot6 부하를 본다.
- `late command piece ... cut` (경고) → 조각 앞부분이 늦어 잘렸다. 컨트롤러가 이음새를 받아들이면 계속 간다. 자주 나오면 robot6 부하.
- 앱은 시작할 때 팔을 홈 (TCP 0.40, 0, 0.12) 으로 천천히 옮긴 뒤 버퍼로 바꾼다. 끝나면 JointGroupPositionController 로 되돌린다.
- **종료는 Ctrl+C 한 번** — 앱이 먼저 정리한다 (새 물체 안 받음 → 든 물체 던지고 대기 자세 → 버퍼 정지 → JointGroupPositionController 로 복귀), 그다음 드라이버가 꺼진다 (수 초 ~ 25 s). **두 번** 누르면 그 자리에서 모두 끈다. 백그라운드로 띄운 경우 `kill -TERM <gp8_cycle pid>` 가 같은 정리 종료다.

## 3b. 조각 모드 (`GP8_DEMO_MODE=pieces`)

push 시간 없이 던지기와 push 를 한 명령 경로로 잇는다. 설계·변수·트윈 결과: `cycle_throw/PIECES.md`.
```bash
GP8_DEMO_MODE=pieces GP8_DEMO_CHECK_ONLY=1 gp8_cycle   # 로봇 없이 점검
GP8_DEMO_MODE=pieces gp8_cycle                         # 실기
```

## 4. 처음 돌릴 때 순서

1. 빈 벨트, `GP8_FORCE_SKILL=throw`: 홈 이동·버퍼 전환·대기만 (20 초)
2. 투명 물체 1 개씩: 접촉 높이 (`GP8_CYCLE_TOP_Z`) 와 착지 확인. 짧으면/길면 `GP8_CYCLE_RELEASE_LEAD` 조정
3. 투명 여러 개 연속
4. 금속 섞기 (push slot). push 는 250 Hz 기준으로 튜닝돼 있어 500 Hz 에서 다시 맞춰야 할 수 있다

## 남은 일 / 한계

- 처리량: push slot 하나가 5~10 s 를 쓴다 (twin). 금속이 많으면 throw 가 줄어든다.
- 순서 정책·시작값은 받은 폴더의 것 (`ord_a6`, NN). 팀원 최신 실기는 `ord_a6_z` + 라이브러리.
- 팀원 다리의 벨트 정지 처리·추종 감시는 넣지 않았다 (하드웨어 쪽 보호에 맡김).

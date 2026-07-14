# suction_lift_debug / RViz preview 사용 가이드

이 문서는 `gp8_control`의 석션 pick + lift + throw 디버그 스크립트 사용법을 정리한다.

대상 파일:

- `tests/suction_lift_debug.py`
- `launch/suction_lift_preview.launch.py`
- `rviz/suction_lift_preview.rviz`

## 1. 목적

`suction_lift_debug`는 카메라, 컨베이어, `app.py` 전체 스택 없이 고정된 좌표에서 아래 동작만 검증하기 위한 스크립트다.

1. 지정한 pick 위치로 이동
2. 석션 ON
3. lift
4. throw 궤적으로 이동
5. release 시점에 석션 OFF

RViz preview 모드는 실제 로봇, 컨트롤러, 석션 IO를 건드리지 않고 `/joint_states`, marker, path만 publish해서 로봇이 어떻게 움직일지 먼저 확인하는 용도다.

## 2. 빌드 / 소스

코드를 수정했으면 먼저 빌드한다.

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select gp8_control
source install/setup.bash
```

launch 파일은 source tree의 Python 코드를 우선 보도록 되어 있지만, 설치된 entry point와 의존성 반영을 위해 빌드 후 `source install/setup.bash`까지 하는 것을 기준으로 한다.

## 3. RViz preview 실행

실제 로봇을 움직이기 전에 먼저 이걸 실행한다.

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch gp8_control suction_lift_preview.launch.py
```

기본값:

- pick: `(x=0.55, y=0.0, z=Config.GRASP_Z)`
- lift: `0.10 m`
- bin: `(x=1.0, y=0.0, z=pick_z + 0.10)`
- extra tool offset: `0.0 m`
- preview rate: `30 Hz`

파라미터를 바꿔서 실행하는 예:

```bash
ros2 launch gp8_control suction_lift_preview.launch.py \
  x:=0.55 y:=0.0 \
  bin_x:=0.95 bin_y:=0.0 bin_z_offset:=0.10 \
  tool_offset:=0.0 \
  preview_rate:=30.0
```

RViz 없이 topic만 보고 싶으면:

```bash
ros2 launch gp8_control suction_lift_preview.launch.py rviz:=false
```

## 4. RViz에서 보이는 것

RViz fixed frame은 `base_link` 기준이다.

주요 topic:

- `/joint_states`: preview 로봇 자세
- `/joint_states_urdf`: URDF 표시용 joint states
- `/suction_lift_debug/markers`: pick/bin/release/tool/path marker
- `/suction_lift_debug/path`: tool frame 기준 preview path
- `/tf`: URDF의 `flange -> suction_tool` fixed transform 포함

마커 의미:

| 색 / 형태 | 의미 |
|---|---|
| 초록 sphere | pick / grasp 위치 |
| 하늘색 sphere | lift 후 위치 |
| 주황 sphere | release 위치 |
| 파란 sphere | bin 최종 목표 위치 |
| 노란 sphere | throw runup 시작점 |
| 자홍 sphere | follow-through 끝점 |
| 노란 line | preview joint trajectory를 FK해서 얻은 실제 `suction_tool` 원점 path |
| 하늘색 line | release 후 물체가 날아가는 탄도 궤적 |
| 빨간 막대 | MuJoCo `grip_site`와 맞춘 TCP 시각화. `flange` 원점에서 `suction_tool` 원점까지 |
| 빨간 작은 sphere | `suction_tool` frame 원점 |

빨간 막대는 marker가 아니라 RViz RobotModel에 포함된 URDF visual이다.

## 5. Tool frame 정의

현재 모델은 MuJoCo `combined_test.xml`의 `grip_site`를 기준으로 한다.

```text
MuJoCo link6 -> grip_site = link6 local +X * 0.325 m
ROS URDF link_6_t -> flange = link_6_t local +X * 0.080 m
따라서 URDF flange -> suction_tool = flange local +X * 0.245 m
suction_tool 자세 = flange 자세와 동일
```

`gp8_control.robots.gp8.forward_kinematics()`의 EE는 이미 MuJoCo `grip_site`/TCP와 일치한다. 따라서 `suction_lift_debug.py`의 기본 `--tool-offset`은 `0.0`이고, IK/FK에는 추가 24 cm를 더하지 않는다.

시작 pick 자세의 축 방향은 다음과 같다.

```text
flange +X = world -Z  # 아래쪽, 실제 툴 축
flange +Y = world +Y
flange +Z = world +X
```

따라서 시작 자세에서 URDF의 빨간 TCP 막대는 flange에서 아래 방향으로 내려간다. throw 궤적의 위치 constraint는 `gp8.py` EE, 즉 MuJoCo `grip_site`/`suction_tool` 원점 기준이다.

```text
gp8_ee_target = suction_tool_target
tool_offset 기본값 0.0
```

## 6. Throw geometry 기본값

기본 bin은 pick보다 x 방향으로 앞쪽이고 z는 10 cm 높다.

```text
pick = (0.55, 0.0, Config.GRASP_Z)
bin  = (1.00, 0.0, Config.GRASP_Z + 0.10)
```

포물선 기준:

1. `P`는 pick→bin 직선의 60% 지점
2. release는 `P→bin` 구간의 30% 지점
3. 따라서 release는 전체 pick→bin 기준 72% 지점
4. release 이후 물체가 나머지 구간을 탄도로 날아 bin에 떨어지도록 release 속도를 계산

코드 상수:

```text
P_RATIO_FROM_GRASP = 0.6
RELEASE_FRACTION_FROM_P = 0.3
BIN_Z_OFFSET_DEFAULT = 0.10
TOOL_OFFSET_DEFAULT = 0.0
```

기본 조건에서 plan-only 검증 시 대표적으로 아래 값이 나온다.

```text
bin=(+1.000,+0.000,+0.162)
release = pick→bin 72%
v=(+1.409,+0.000,+1.034) m/s
|v|=1.748 m/s
angle=36.25 deg
```

## 7. 로봇 없이 계획만 검증

로봇, RViz 없이 계산만 확인하려면:

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run gp8_control suction_lift_debug --plan-only
```

파라미터 예:

```bash
ros2 run gp8_control suction_lift_debug --plan-only \
  --x 0.55 --y 0.0 \
  --bin-x 0.95 --bin-y 0.0 \
  --bin-z-offset 0.10 \
  --tool-offset 0.0
```

여기서 IK 실패, 속도 초과, reach 초과가 뜨면 실제 로봇 실행 전에 먼저 파라미터를 조정해야 한다.

## 8. 실제 로봇 실행

실제 실행 전 조건:

- 로봇 주변 clear
- bin 위치 실제로 표시
- 펜던트 `REMOTE + AUTO`
- MotoROS2 / micro-ROS agent 정상
- debug bringup 실행
- 석션 IO가 현재 잘 동작하는 상태

bringup:

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch gp8_control debug_robot.launch.py
```

별도 터미널에서 실행:

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run gp8_control suction_lift_debug
```

파라미터를 명시하는 예:

```bash
ros2 run gp8_control suction_lift_debug \
  --x 0.55 --y 0.0 \
  --lift 0.10 \
  --bin-x 0.95 --bin-y 0.0 \
  --bin-z-offset 0.10 \
  --tool-offset 0.0 \
  --vel-scale 0.3
```

실행 중 프롬프트 흐름:

1. object가 pick 위치에 있는지 확인
2. Enter로 pick/lift 진행
3. throw 준비 후 `t` 입력
4. bin 위치가 맞는지 확인
5. Enter로 throw/release 실행

주의: throw는 저속 move가 아니라 실제 고속 swing이다. RViz preview와 `--plan-only`를 먼저 통과시키고, 처음에는 bin을 가깝게 둔 조건으로 시작한다.

## 9. 자주 쓰는 옵션

```text
--x, --y              pick XY [m]
--z                  pick Z [m], 기본 Config.GRASP_Z
--hold               석션 ON 후 hold 시간 [s]
--lift               lift 높이 [m]
--vel-scale          pick/lift 등 저속 이동 scale
--bin-x, --bin-y     bin XY [m]
--bin-z-offset       bin z = pick z + offset [m]
--tool-offset        flange +X 방향 tool offset [m]
--release-lead       release knot 대비 석션 OFF timing 보정 [s]
--plan-only          로봇 없이 계획만 검증
--rviz-preview       RViz preview용 publish만 수행
--preview-rate       RViz preview joint state publish rate [Hz]
```

## 10. Troubleshooting

### RViz에서 로봇이 가만히 있음

확인:

```bash
ros2 topic hz /joint_states
ros2 topic echo /joint_states --once
ros2 topic list | grep suction_lift_debug
```

`/joint_states`가 안 나오면 preview node가 죽은 것이다. launch 로그에서 `suction_lift_debug` 에러를 먼저 본다.

### `unrecognized arguments: --ros-args ...` 에러

이 문제는 ROS launch가 붙이는 `--ros-args`를 argparse가 처리하지 못해서 생긴다. 현재 코드는 처리되도록 수정되어 있다. 그래도 뜨면 빌드/소스가 예전 install을 보고 있는 것이다.

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select gp8_control
source install/setup.bash
```

### 마커가 안 보임

RViz display를 확인한다.

- MarkerArray topic: `/suction_lift_debug/markers`
- Path topic: `/suction_lift_debug/path`
- Fixed Frame: `base_link`

### TF / 빨간 막대가 이상함

기준은 아래 하나다.

```text
flange -> suction_tool
translation = (0.245, 0, 0) in flange frame
rotation = identity
```

즉 RViz의 `suction_tool`은 MuJoCo `grip_site`와 맞춘 frame이고, flange frame 기준 `+X` 방향 24.5 cm다. 자세는 flange와 같다.

### plan-only에서 reach / IK 실패

우선 조정 순서:

1. `bin_x`를 줄여 bin을 가깝게 둔다.
2. `bin_z_offset`을 너무 크게 두지 않는다.
3. `tool_offset`은 기본 `0.0`으로 둔다. 이 값은 MuJoCo/gp8.py TCP보다 더 앞의 임시 점을 테스트할 때만 바꾼다.

실제 TCP 길이 모델은 URDF의 `flange -> suction_tool = +X 0.245 m`와 `gp8.py`의 EE convention에 들어있다.

## 11. 권장 작업 순서

1. `--plan-only`로 계산 검증
2. `suction_lift_preview.launch.py`로 RViz preview 확인
3. 빨간 막대가 실제 툴 방향/길이와 맞는지 확인
4. bin을 가까운 위치로 두고 실제 로봇 저위험 조건에서 테스트
5. release timing이 늦거나 빠르면 `--release-lead`로 IO 지연만 보정

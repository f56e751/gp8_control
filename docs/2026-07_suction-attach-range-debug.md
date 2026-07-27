# Suction attachment-range debug

`suction_attach_range_debug`는 물체를 던지지 않는다. 정지한 컨베이어 위의
지정 XY에서 석션을 켜고, 보정된 접촉 높이 위에서 천천히 하강하며 물체가
붙는 순간의 실제 `suction_tool` TCP 높이를 기록한다.

현재 `gp8_control`에는 진공 압력/흡착 완료 입력이 없으므로 자동 감지가
아니라 작업자가 실제로 물체가 붙는 것을 보고 그 순간 키를 누르는 방식이다.
따라서 화면의 `흡착 순간 키 입력 없음`은 흡착 실패를 자동 판정한 결과가
아니라, 종료 높이에 도달할 때까지 기록 키가 입력되지 않았다는 뜻이다.

## 사용 순서

먼저 로봇을 움직이지 않고 경로를 확인한다.

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run gp8_control suction_attach_range_debug \
  --x 0.30 --y -0.30 --plan-only
```

실제 측정 전 별도 터미널에서 bringup을 실행한다.

```bash
~/ros2_ws/debug_bringup.sh
```

측정 실행:

```bash
ros2 run gp8_control suction_attach_range_debug \
  --x 0.30 --y -0.30
```

기본 동작:

- 접촉 기준 TCP Z: `Config.GRASP_Z = 0.062 m`
- 시작 위치: 접촉 기준 `+30 mm`
- 종료 위치: 접촉 기준 `0 mm`이며 그 아래로는 명령하지 않음
- 최대 하강 속도: `5 mm/s`
- 석션 ON 후 시작 위치 대기: `0.30 s`
- `Space`, `Enter`, `A`: 붙은 순간 기록하고 즉시 stream 정지
- `Q`: 측정 중단

붙는 순간 키를 누르면 다음을 출력한다.

```text
TCP position
start에서 하강한 거리
contact 기준 간격
반응시간에 따른 추정 오차
```

기본 5 mm/s에서 반응시간을 0.2초로 보면 거리 오차는 약 `±1 mm`다.

## 주요 옵션

```text
--contact-z        suction_tool이 물체와 접촉하는 보정 Z [m]
--start-clearance  접촉 Z 위 시작 간격 [m], 기본 0.030
--max-descent      최대 하강거리 [m], 기본 start-clearance
--end-z/--final-z  최종 suction_tool TCP Z [m], 기본 contact-z
--descent-speed    최대 TCP 하강속도 [m/s], 기본 0.005
--prime-time       석션 ON 후 하강 전 대기 [s], 기본 0.30
--move-vel-scale   시작 자세 이동 관절속도 배율, 기본 0.15
--reaction-time    출력 오차 계산용 작업자 반응시간 [s], 기본 0.20
--plan-only        실제 로봇과 석션을 건드리지 않고 경로만 검증
```

`--end-z`와 `--max-descent`는 동시에 쓸 수 없다. 예를 들어 접촉 기준보다
8 mm 위에서 끝내려면 다음처럼 실행한다.

```bash
ros2 run gp8_control suction_attach_range_debug \
  --x 0.30 --y -0.30 --end-z 0.070
```

`--end-z`는 보정된 접촉 Z보다 낮은 값도 직접 지정할 수 있다. 이 경우
스크립트는 실행을 막지 않고 충돌 위험 경고를 출력한다. 물체 형상이 달라
접촉 높이가 다르면 `--contact-z`를 먼저 정확하게 설정하고, 실제 로봇을
구동하기 전에 반드시 `--plan-only`로 IK와 최종 높이를 확인한다.
기존 `--allow-below-contact`는 명령 호환성을 위해 남아 있지만 더 이상
필요하지 않다.

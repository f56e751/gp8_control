# Suction hold at Z

`suction_hold_at_z`는 현재 TCP의 X/Y와 자세를 유지한 채 먼저 base 기준
`Z=0.200 m`를 경유하고 지정한 최종 Z로 이동한다. 최종 목표 도착 후 석션을
켜고, 작업자가 Enter를 누를 때까지 현재 자세와 석션을 유지한다. Enter,
Ctrl-C, 입력 종료 중 어느 경로로 끝나도 석션 OFF 명령을 전송하고 종료한다.

```bash
ros2 run gp8_control suction_hold_at_z --z 0.070
```

기본 경유 높이 `0.200 m`를 바꾸려면 `--via-z`를 사용한다.

```bash
ros2 run gp8_control suction_hold_at_z --z 0.061 --via-z 0.250
```

다른 XY로 함께 이동하려면 다음처럼 지정한다.

```bash
ros2 run gp8_control suction_hold_at_z \
  --x 0.30 --y -0.30 --z 0.070 --vel-scale 0.30
```

현재 상태에서 IK만 확인하고 로봇과 석션에 명령을 보내지 않으려면:

```bash
ros2 run gp8_control suction_hold_at_z --z 0.070 --plan-only
```

`--z`, `--via-z`, `--x`, `--y`는 모두 base frame 기준 TCP 좌표이며 단위는 m다.
`--x/--y`를 생략하면 실행 시점의 현재 TCP X/Y를 사용한다. 자세도 현재 TCP
자세를 그대로 유지한다. 경유점은 항상 현재 TCP X/Y에서 생성되고, 최종점만
선택적으로 `--x/--y`를 사용한다. 이 스크립트는 MoveIt 충돌 회피를 사용하지 않으므로
실제 실행 전에 주변 공간과 추가 장착판의 이동 경로를 직접 확인해야 한다.
기본 이동 속도는 관절 속도 제한의 30%이며 `--vel-scale`로 조정할 수 있다.

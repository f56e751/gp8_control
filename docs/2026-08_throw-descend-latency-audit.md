# Throw descend 잔여 지연 진단

작성일: 2026-08-03  
대상 브랜치: `feature/full-bbox-projection`

## 현상

엔코더 누적거리로 큐의 물체 위치를 추적하도록 바꾼 뒤, 첫 물체만 맞고 두
번째 이후 물체가 늦어지는 누적 오차는 해소됐다. 그러나 모든 물체에서 석션
패드가 바운딩박스 중앙보다 진행 방향 뒤쪽에 닿는 고정 시간 지연 형태의 오차가
남아 있다. 컨베이어가 빠를수록 오차 거리가 커지고, 빠른 조건에서는 물체가
통과한 뒤 패드가 내려온다.

바운딩박스 앞쪽 20%를 목표로 바꾸는 실험은 증상을 공간 오프셋으로 가릴 뿐
원인을 보정하지 못하므로 되돌렸다. 현재 throw는 다시 네 꼭짓점의 중심을
사용한다.

## 현재 반영된 보정

### Perception 구간

`camera_debug.py`의 정상 경로는 다음 전체 시간을 프레임마다 직접 계산한다.

1. RealSense 프레임 캡처
2. 카메라/USB 전달 및 프레임 대기
3. perception 추론
4. 서버 직렬화 및 송신 대기
5. 네트워크 전송
6. 제어 컴퓨터 수신 및 JSON 파싱

Perception 서버의 `/latency` 응답으로 서버-클라이언트 시계를 동기화하고,
`capture_timestamp`를 제어 컴퓨터 시간축으로 변환해 capture-to-receipt age를
구한다. 이 값만큼 `belt_speed * delay`를 base Y에서 역투영한다.

2026-08-03 확인 당시 서버는 다음 필드를 정상 제공했다.

- `protocol=gp8-latency-v1`
- `capture_timestamp_domain=timestamp_domain.global_time`
- `capture_age_s`: 약 35~41 ms
- `elapsed_s`(추론): 약 80~83 ms
- `server_send_timestamp`

따라서 `latency_mode=live_capture_to_receipt`인 동안 카메라·추론·네트워크의
큰 지연이 통째로 빠져 있을 가능성은 낮다. 단, clock sync 실패로
`fallback_components`가 되면 post-inference/network 잔여 시간이 완전히 포함되지
않을 수 있으므로 실행 중 latency mode를 반드시 기록해야 한다.

### Detection 이후 물체 이동

`TrackedObject`는 최초 검출 시점의 엔코더 누적거리와 Y를 고정 앵커로 저장한다.
그 뒤 물체 Y는 속도 적분이 아니라 현재 누적거리와 앵커 누적거리의 차이로
계산한다. 재검출은 lane X, bbox, confidence, class만 갱신하고 제어 Y 앵커를
앞뒤로 옮기지 않는다.

이 변경으로 이전의 "첫 물체는 성공하지만 두 번째 이후는 지나간 자리를 잡는"
누적 추적 문제를 분리했다. 카메라 bbox 이동으로 추정한 속도는 제어에 사용하지
않는다.

## 아직 직접 측정·보정되지 않는 구간

### 1. Descend 명령에서 실제 TCP 운동까지

현재 공통 `ACTION_START_LEAD`는 20 ms이고, track-descend는 100 ms ramp의 절반인
50 ms를 추가한다. `TRACK_LEAD_T=0`이면 프로그램은 물체 중심 도착 70 ms 전에
descend 전송을 시작한다.

하지만 20 ms는 실시간 측정값이 아니라 기존 HW 관측에 기반한 상수다. 아래
항목은 매 사이클 측정되지 않는다.

- Python publisher에서 ros2_control controller까지의 전달 시간
- controller command가 실제 GP8 관절 운동으로 나타날 때까지의 시간
- 실제 관절이 명령 관절을 따라가는 tracking error
- `/joint_states` 자체의 관측 지연

따라서 실제 command-to-motion 지연이 20 ms보다 크면 물체가 패드보다 앞서며,
오차 거리는 대략 `belt_speed * residual_delay`로 속도에 비례한다.

### 2. 가속도 제한에 의한 추종 지연

`_build_track_descend()`는 생성한 경로의 관절 **속도** 제한만 검사한다. 관절
가속도는 검사하지 않는다. 실제 bringup은 `axis_acceleration_factor=0.02`의
driver-side limiter를 사용하므로 100 ms 안에 정지 상태에서 벨트 속도로
올라가라는 명령이 제한될 수 있다.

이 경우 명령 TCP는 벨트 속도를 따라가더라도 실제 TCP는 ramp 초반부터 뒤처진다.
속도가 빠를수록 요구 가속도와 누적 위치 오차가 모두 커진다. 현재
`track-descend done` 로그는 최종 **명령 joint**를 FK한 값이지 실제 joint의
접촉 시점 FK가 아니므로 이 오차를 드러내지 못한다.

### 3. 접촉 후 즉시 흡착된다는 가정

Z가 `TRACK_Z_END`에 도달한 뒤 코드는 100 ms 동안 벨트 방향 속도를 감속한다.
주석과 로직은 Z 접촉 순간 물체가 이미 패드에 완전히 붙어 벨트가 아래에서
미끄러진다고 가정한다.

실제로 seal 형성이 늦으면 감속 구간에서 물체가 계속 벨트 속도로 움직이는 동안
패드는 평균적으로 절반 속도만 움직인다. 이 상대 이동은 시간으로 환산하면
`0.5 * TRACK_DECEL_T = 50 ms`다.

또한 streaming backend는 `final_joint`가 전달되면 같은 최종 자세를 50 ms 뒤에
한 번 더 붙여 settle한다. 물체가 아직 붙지 않았다면 여기서도 50 ms만큼 더
앞서간다. 두 항을 합치면 최대 약 100 ms 상당의 상대 이동이다.

| 벨트 속도 | 100 ms 상당 상대 이동 |
|---:|---:|
| 0.10 m/s | 10 mm |
| 0.20 m/s | 20 mm |
| 0.40 m/s | 40 mm |

이 크기와 속도 비례 형태는 현재 관찰과 일치한다. 실행 로그에서도 계획
track-descend가 0.34 s인 조건에서 blocking 호출은 대략 0.40~0.44 s 뒤에
반환됐다. 약 50 ms의 settle과 Python/ROS scheduling 시간이 계획시간 표시에는
포함되지 않기 때문이다.

### 4. 대기 중 속도 변화

`position_and_prime()`은 호출 시점의 속도와 물체 위치로 도착 시각을 한 번
계산한 뒤 절대시각까지 기다린다. track-descend 경로의 벨트 속도 역시 wait pose를
계획할 때 한 번 샘플링한다. 긴 대기 중 컨베이어 속도가 변해도 active target의
도착 deadline과 이미 생성된 descend 경로는 다시 계산하지 않는다.

속도가 충분히 안정적이면 주원인은 아니지만, 가감속 중 투입하거나 속도 토픽이
흔들리는 조건에서는 고정 시간 오차가 추가될 수 있다.

## 현재 결론

1. 바운딩박스 중심 선택 자체가 주원인은 아니다.
2. live timestamp 모드에서는 카메라·추론·네트워크 지연은 위치 보정에 포함된다.
3. 가장 가능성이 큰 잔여 항은 실제 로봇의 command tracking lag와 접촉 후 seal
   형성 시간이다.
4. driver 가속도 limiter, 100 ms track 감속, 50 ms final settle이 이 잔여 지연을
   만들거나 확대할 수 있다.
5. 현재 로그만으로 command lag와 seal lag를 정확히 분리할 수는 없다.
6. 과거 `TRACK_LEAD_T=0.3s`가 실기에서 잘 맞았던 것은 잔여 시간 지연을 경험적으로
   상쇄했다는 증거이지만, 여러 지연을 한 상수로 합친 값이므로 근본 측정값은 아니다.

## 다음 측정 순서

### A. 지연이 시간 상수인지 확인

같은 물체와 자세에서 벨트 속도만 바꾸고 실제 중심 대비 접촉 오차 `miss_m`를
측정한다. 각 실행에서 `miss_m / belt_mps`를 계산한다. 이 값이 속도에 관계없이
비슷하면 공간 보정 문제가 아니라 잔여 시간 지연이다.

### B. 명령 TCP와 실제 TCP 동시 기록

descend 전송 직전부터 종료까지 다음을 같은 monotonic clock으로 250 Hz에 가깝게
기록한다.

- command joint/TCP
- `/joint_states` actual joint/TCP
- encoder 누적거리로 계산한 object Y
- 예정된 Z-contact 시각과 actual TCP가 `TRACK_Z_END`를 통과한 시각

`actual_tcp_y - predicted_object_y`를 actual Z-contact 시점에 비교하면
command-to-motion/driver tracking lag를 직접 얻을 수 있다. 현재의
`fire-timing`은 descend 시작 전 값이고 `track-descend done`은 명령 FK라서 이
측정을 대신할 수 없다.

### C. Seal 형성 시각 측정

가능하면 vacuum pressure switch/센서의 상승 시각을 기록한다. 센서가 없다면
고속 영상으로 패드 접촉, 물체가 벨트에서 미끄러지기 시작한 시점, 물체가 들리는
시점을 같은 프레임에서 읽는다. actual Z-contact와 seal 사이 시간이 확인되면
감속/settle 구간의 상대 이동을 별도로 계산할 수 있다.

### D. Perception 모드 확인

각 실기 실행에서 `camera_debug`의 다음 값을 파일 로그로 남긴다.

- `latency_mode`
- `applied_delay_s`
- `capture_age_s`
- `server_post_inference_s`
- `network_receive_s`
- `clock_min_rtt_s`
- `clock_sync_error`

`fallback_components` 실행은 live 실행과 분리해 비교한다.

측정 전에는 bbox 기준점을 옮기거나 새로운 고정 오프셋을 기본값으로 넣지 않는다.
먼저 actual contact 시점의 시간 오차를 분해한 뒤, command lead와 경로/driver
제약을 각각 맞추는 것이 다음 수정의 기준이다.

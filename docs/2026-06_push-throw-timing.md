# GP8 push/throw 타이밍·라우팅 조사 및 수정 (2026-06)

branch: `merge/push-throw`

이 세션에서 conveyor pick-and-throw/push의 타이밍·라우팅 문제 5건을 잡고,
"생성한 궤적이 실제로 몇 초 걸리는가"를 측정·분석하는 진단 도구를 추가했다.
아래는 무엇을 왜 고쳤는지 + 타이밍 조사에서 알아낸 사실(다음 작업의 근거)을 남긴다.

---

## 1. 해결한 문제 (요약 + 커밋)

| # | 증상 | 원인 | 수정 | 커밋 |
|---|---|---|---|---|
| 1 | push 치는 타이밍이 안 맞음 | push가 throw용 도착 lead(0.2s)를 재사용 — push는 흡착 용서가 없어 전체 후속지연만큼 lead 필요 | `arrival_lead()` 훅 분리 + push는 `FIXED_DELAY_PUSH=0.8` | `c5f52e5`, `427e2d8` |
| 2 | throw 후 home 복귀했다 다시 push 대기자세 (낭비) | 커밋된 다음 픽이 없으면 무조건 home(idle)로 chain | 큐 비었을 때(①a)만 home, 그 외엔 **lifted standby**(끝자세 XY 유지, Z만 home) | `ed96f0c` |
| 3 | 페트병을 push로 캔 위치에 넣음 | 트랙 class가 **최초 1프레임에 latch**(저신뢰 진입 프레임이 metal로 오분류되면 평생 metal) | **신뢰도 가중 class 투표**(argmax) | `36283f0` |
| 4 | 캔 미스 / 다음 PET 밀침 | intercept 배치용 `move_time`이 **큐 재진입(~0.4s×2)+push 선이동을 누락** → 상류 stale 지점 조준 | **skill 소유 `t_to_contact` 타임라인** + 측정된 큐스위치 EMA + stale-stroke abort + scan 게이트 순서 | `7524769` |
| 5 | push 이동이 planned보다 ~0.13s 길게 걸림 | 스트로크 끝에서 **완전 정지(v=0)** 후 체인 재가속(+손목 회전) | 스트로크 출구속도를 체인 시작으로 **연속화**(throw의 릴리스속도 carry와 동일) | `24de0f7` |

진단 도구: `9e714db`, `67be957`.

---

## 2. 진단 도구 (`utils/motion_logger.py`)

**목적**: 스케줄러의 "예측 이동시간(planned 궤적 길이)" vs "실제 이동시간"의 차이를
측정해 사이클 누적 드리프트를 추적.

**켜는 법**: `GP8_MOTION_LOG_DIR=$HOME/gp8_motion ros2 launch gp8_control gp8_bringup.launch.py`
(미설정 시 전부 no-op — 프로덕션 영향 0). 파일명에 실행 타임스탬프가 붙어 덮어쓰지 않음.

**산출 (실행별 2파일)**:
- `motion_commands_<stamp>.csv` — dispatch 1건/행:
  `cmd_id, skill(push/throw), t_queued, planned_dur_s, actual_dur_s, err_s(=actual-planned), reached, start_j*, target_j*`.
  `err_s` 누적합 = 드리프트. `reached=0` = 도달 전에 다음 dispatch 시작(체이닝/놓침).
- `motion_samples_<stamp>.csv` — 실제 관절 trace `t, cmd_id, dist_to_target, j1..6`.

**배선(기존 코드 무오염)**: `TrajectoryController._push_waypoints`→`on_command`,
`_joint_state_cb`→`on_sample`, `run_epoch`이 `set_motion_op(skill.name)`으로 skill 라벨 전달.

**`[PUSH-DIAG]` 로그**(motion logging 켜질 때만): dispatch마다
`N pts queued in Xms (avg/max ms/pt; busy=B); arm motion start = ...`.
`DURING push` = 포인트 미는 중 출발(H1), `NOT until push done` = 적재 후에야 출발(H2).

---

## 3. 타이밍 조사 결과 (핵심 지식)

### 3.1 예측 vs 실제 = **상수 오프셋 (~+0.29s/ dispatch)**
`motion_commands.csv` 분석: `err = actual - planned`가 planned 길이와 **무상관**
(`corr≈0`), 즉 비례형이 아니라 **dispatch당 거의 일정한 +0.29s**. 사이클당 dispatch가
여러 번이라 ~0.29×N 누적 = 관측된 드리프트. (push +0.293, throw +0.291)

### 3.2 분해 — **시작지연 + 이동초과**
관절 Δ 기반(타깃 무관, throw 스윙에도 유효) span 측정:
- **THROW: 실제 이동구간 ≈ planned** (큰 스윙 span/plan = 0.95~1.00).
- **PUSH: planned보다 ~1.3배(+~0.13s) 김** — 정지 때문 (3.4).
- 두 skill 공통 **시작 dead-time ~0.25s**: 명령을 큐에 넣고 로봇이 움직이기 시작할
  때까지의 지연 (3.3).

즉 `actual = 시작지연(~0.25s, 공통) + 이동(throw≈planned / push는 +0.13s 초과)`.

### 3.3 시작지연의 원인 = **동기식 per-point 큐 적재 (주로 H1) + 고정 바닥(H2)**
- `_push_waypoints`가 포인트를 **하나씩 `call_async`+`spin_until_future_complete`**로
  전송(직렬). 브리지 `_on_queue_traj_point`가 다시 상위 MotoROS2로 블로킹 포워드 →
  **포인트당 ~35-45ms 왕복**(`[PUSH-DIAG]` avg).
- 로봇은 **포인트 ~7-8개 버퍼되면 출발**(`[PUSH-DIAG]`: 큰 dispatch는 280ms에 출발,
  전체 적재 안 기다림) → 지배적 = H1(동기 push). 작은 dispatch는 적재 후 ~0.15s 더
  기다림 = 고정 H2 바닥.
- 변동: BUSY 재시도 + 포인트 수 + agent 지터 + 첫-이동 워밍업(cmd1 +1.1s 등) → **단일
  상수 하드코딩 부적절**. 측정(EMA)이 정답.

### 3.4 이동초과(push) = **스트로크→체인 정지**
push 궤적엔 throw엔 없는 내부 정지가 있었다:
- 스트로크가 push_end에서 **v=0 완전정지** 후 체인이 **정지 상태에서 재가속**(+손목
  j6를 push-facing→0으로 회전). 육안의 "밀고 → 정지 → 돌리고 → 이동".
- 각 v=0 junction에서 컨트롤러가 planned 타임스탬프보다 더 dwell/ramp → push만 초과.
  throw는 릴리스 속도를 체인으로 이어받아 **연속**(정지 0개) → planned 일치.

**수정(`24de0f7`)**: 체인을 스트로크 **출구속도(`push_end_dq`)에서 생성** → 체인 위치가
연속으로 흐름. M1 clip(alarm 4414 안전). (주의: 큐 속도는 `_decimate_for_queue`가
위치에서 재계산하므로, 핵심은 "체인 위치 연속"이지 vel 배열 값이 아님.)

**결과(연속화 후)**: throw를 런-조건 기준선으로 삼으면 push 고유 초과
`span/plan(push−throw)`가 **+0.22 → +0.04**로 붕괴, push err **0.293→0.259**(throw는
~0.29 유지). **정지 제거는 작동했으나 효과는 modest** — 정지는 소수 기여자였고,
지배적 비용은 공통 시작지연(~0.25s)이다.
(주의: 런 간 변동이 커서[throw도 1.08→1.17] 절대값 비교는 불가, 상대(gap)만 유효.)

### 3.5 근본 한계 — **"몇 초 걸릴지" 사전 정확 예측은 불가능**
MotoROS2 point-queue 인터페이스 조사 결과:
- `QueueTrajPoint` 응답 = `SUCCESS/BUSY`뿐. **큐 깊이·완료·소요시간 신호 없음.**
  `time_from_start`도 보장이 아니라 요청. 별도 status 토픽 없음.
- 시작지연은 컨트롤러 내부 + 가변 → push 전에 계산 불가.

따라서:
- **사전(a-priori)**: planned 이동시간만 앎. 총 벽시계(시작+이동)는 모름.
- **실행 중/후**: `/joint_states`(~50Hz)로 **측정**하면 실제 시간을 앎(ground truth).
  `_wait_for_position`/MotionLogger가 이미 측정.
- 연속화는 push *이동*을 planned에 더 가깝게 만들었을 뿐, "사전에 안다"로 바꾸지 못함.

---

## 4. 핵심 결론

1. **드리프트는 dispatch당 상수 ~0.29s** (비례 아님) → 상수항으로 모델링하되 **측정**해야 함.
2. **이동시간 ≈ planned** (throw 정확, push는 연속화로 근접). 미지의 핵심은 **시작지연**.
3. **시작지연은 사전 예측 불가** (MotoROS2 한계) → 유일한 정답은 `/joint_states` 피드백 측정.
4. push의 정지 오버런은 연속화로 제거됨(modest). 남은 큰 덩어리는 **공통 시작지연**.

---

## 5. 남은 작업 / 다음 단계 (우선순위)

1. **시작지연(~0.25s) 공략 — 가장 큰 레버**: `_push_waypoints`의 동기 per-point push를
   **비동기/파이프라인**으로 (현재는 한 점 보내고 응답 대기·반복). 버퍼가 빨리 차서 출발이
   빨라지고 변동도 줄어듦. MotoROS2 큐/BUSY·순서 처리 주의.
2. **측정 기반 self-calibration**: dispatch 오버헤드(또는 `qmode_ms_avg`처럼)를 EMA로
   측정해 스케줄러 타임라인(`t_to_contact`)에 반영. 사전 예측은 못 해도 평균 보정됨.
   (로거는 진단 전용으로 유지; 컨트롤러가 자체 측정.)
3. **positioning via 정지 연속화**: `move_through_via`의 aim via(v=0)도 남아 있음(효과는 작음).
4. **손목 회전**: 체인의 j6→0 리셋은 다음 동작이 요구 → 줄이려면 다음 동작이 push-facing
   wrist를 받게 하는 별도 작업.
5. **검증**: 런 간 노이즈가 크므로 같은 세션 controlled A/B(또는 더 많은 사이클)로 재확인.

---

## 6. 튜닝 노브 / 관련 파일

- `Config.OPT_TIME_TO_REAL` (env `GP8_OPT_TIME_TO_REAL`, 기본 1.0): push positioning
  추정 보정(곱). 축소 금지(과소추정=뒤를 침). HW 로그로 튜닝.
- `FIXED_DELAY_PUSH=0.8` (`push_skill.py`): push 도착 lead. feature/push 실측값.
- `Config.ACTION_START_LEAD` (env `GP8_ACTION_START_LEAD`, 기본 0.2): throw 공유 base lead.
- `controllers/trajectory_controller.py`: `qmode_ms_avg`(큐 재진입 측정 EMA), `_push_waypoints`(동기 push + PUSH-DIAG).
- `skills/context.py`: `earliest_reachable_intercept`(intercept 배치), `t_to_contact_fn`, `scan_next_intercept`.
- `skills/base.py` / `push_skill.py` / `throw_skill.py`: `t_to_contact`, `arrival_lead`, build_*_trajectory.
- `perception/detection_intake.py` + `tracking/object_queue.py`: class 투표(`vote_class`), track_id/conf.

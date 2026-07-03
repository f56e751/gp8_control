# GP8 큐/FJT 시작 딜레이 실측 및 극복 방법 (2026-07)

branch: `merge/push-throw`

"새 궤적을 입력하면 팔이 움직이기까지 얼마나 걸리는가"를 **하드웨어로 실측**하고,
그 딜레이의 구성·불가피한 바닥·극복 방법을 정리한다. 앞선
`2026-06_push-throw-timing.md`(큐 디스패치 오버헤드 조사)의 후속으로, "예측"에서
"**딜레이 제거**"로 방향을 튼 뒤의 결과다.

---

## 0. TL;DR

- **명령 → 팔 첫 움직임**: FJT **~0.2s**, 큐(재진입 포함) **~0.6s**.
- 차이 ~0.4s는 **전적으로 "큐 모드 재진입"**(매 디스패치마다 냄). FJT는 이걸 안 냄.
- **큐를 안 비우면(persistent) 큐도 ~0.2s** — 재진입이 사라짐.
- **불가피한 바닥 = 컨트롤러 startup ~0.15s** (FJT·큐 공통, 정지→첫움직임). 소프트웨어로 못 줄임.
- **극복법 = 필요한 순간보다 먼저 밀어넣기**(pre-load) + **안 멈추기**(연속 스트리밍) + **선행 출발**(lead).
  예측 가능한 부분은 딜레이를 100% 숨길 수 있고, 반응형(잡은 뒤 던지기) 부분의 최소 반응지연만 남는다.

---

## 1. 딜레이 지도 (실측 기반)

| 성분 | 크기 | 성격 | 줄이기 |
|---|---|---|---|
| **큐 모드 재진입** | **~0.4s / 매 디스패치** | stop_traj + start_point_queue_mode 모드 스위치. 큐가 드레인되면 자동 종료돼 매번 다시 진입 | ✅ **persistent queue**(큐를 안 비움) |
| **per-point 푸시** | ~26–44ms/점 (버퍼필 ~6–7점 ≈ 0.2–0.3s) | queue_traj_point 서비스 왕복(브리지+MotoROS2). startup과 겹침 | ✅ async/파이프라인 푸시 |
| **컨트롤러 startup** | **~0.15s (150–180ms)** | 정지→첫움직임. YRC1000/MotoROS2가 궤적 받아 실제 모션 시작하기까지 | ❌ **불가피**(단 정지 출발 1회분) |
| 액션 핸드셰이크(FJT) | ~6ms | goal accept | 무시 가능 |

정리: **명령→첫움직임 = FJT ~0.2s(startup) vs 큐 ~0.6s(재진입+푸시+startup)**.

---

## 2. 실측 방법 & 원본 수치

측정 도구(둘 다 `ros2 run gp8_control <name>`, `debug_robot.launch.py` 필요):
- `measure_fjt_floor` — FJT single goal, `accept → 첫 관절 움직임`.
- `measure_queue_floor` — 큐, `재진입 / per-point 푸시 / push → 첫 움직임`.

**램프 아티팩트 분리(핵심 방법)**: 첫 움직임을 `0.01 rad` 임계로 검출하는데, 부드러운
궤적은 천천히 가속해 임계 도달까지 시간이 걸린다(= 램프, startup 아님). 그래서
**gentle(0.05rad/0.6s) vs fast(0.2rad/0.25s)** 두 프로파일을 재고 `gentle − fast`로 램프를
빼면 순수 startup이 남는다.

### 2.1 FJT (2026-07-01)
- 핸드셰이크 ~6ms.
- accept→motion: gentle **378ms**, fast **192ms** → 램프차 **185ms**.
- **순수 컨트롤러 startup ≈ ~150–180ms**.

### 2.2 큐 (2026-07-01)
- 재진입 `reenter_ms` **~402ms (매 move)**.
- per-point 푸시 ~130ms(5점, ~26ms/점, startup과 겹침).
- push→motion: gentle **357ms**, fast(클린) **~208ms** → 램프차 ~149ms.
- 큐 push→motion(클린) ~208ms ≈ FJT accept→motion ~192ms → **startup ~0.15s는 공통**.
- ⚠️ 아티팩트: fast 절반이 ~23ms로 튐 = 재진입(~0.4s) 도중 팔이 이미 ~0.025rad 움직여
  (이전 move 잔여/모드스위치) `start` 대비 임계 조기 통과. 클린값은 ~208ms(스크립트 평균 116은 왜곡).

### 2.3 bringup 로그 교차확인
`motion_commands_*.csv` 분해(qmode_ms/push_ms/motion_start_ms): 재진입 ~400ms, per-point
~44ms/점, 버퍼필 ~6–7점(~290ms)에 팔 출발. 작은 디스패치는 `motion_start ≈ push_ms`
(startup이 푸시에 겹쳐 가려짐).

---

## 3. 무엇이 불가피하고 무엇이 아닌가

- **불가피(진짜 바닥)**: 컨트롤러 startup **~0.15s** — 정지→첫움직임. FJT·큐 둘 다 냄. 표준
  인터페이스(FJT·큐)로 명령하는 한 못 피함(없애려면 MotoPlus/RT 저수준 제어 — 범위 밖).
  단 **정지 상태에서 출발할 때만** 낸다.
- **제거 가능**: 큐 재진입(~0.4s, persistent), per-point 푸시(~수십 ms/점, async).

정정 이력(투명성): 처음엔 "~0.15–0.25s 컨트롤러 바닥"이라 추정 → 중간에 로그 오독으로
"바닥≈0"이라 정정(오판, per-point 푸시+램프에 가려진 것) → FJT 2-프로파일 실측으로
**~0.15s가 맞다고 최종 확정**.

---

## 4. 딜레이를 극복하는 방법 (핵심)

딜레이는 "명령→움직임"의 **latency**다. 필요한 순간보다 **딜레이만큼 먼저** 넣으면 그
순간엔 이미 움직이거나 버퍼에 들어있어 지연이 **빈 시간에 흡수**된다(제거가 아니라 hide).

1. **pre-load / look-ahead** — 다음 궤적을 필요 전에 큐에 밀어넣음. = persistent queue의 핵심.
   컨베이어라 물체 도착 시점 예측 가능 → 미리 넣을 수 있음.
2. **연속 스트리밍(안 멈추기)** — startup은 정지→첫움직임에만. pick→hold→throw→다음pick을
   한 큐로 안 멈추고 흘리면 startup은 맨 처음 1회, 재진입 0.
3. **선행 보정(lead)** — 언제 필요한지 알면 딜레이만큼 먼저 출발(앱의 `start_lead`/`arrival_lead`).

**한계 = 진짜 남는 최소치**: 미리 아는 것만 밀어넣을 수 있다. throw는 물체 잡은 순간에야
확정(반응형)이라 그 계산시간(planning-gap)은 못 미리 넣음 → **hold 중 precompute-at-lock**으로
최소화. + 큐 안 비우려면 **최소 몇 점(~0.1–0.15s) 버퍼** 필요. 결국 최소 ≈ "마지막 순간
정보 반응 + 최소 버퍼" ~0.1–0.2s.

---

## 5. 구현 상태 (persistent queue, 커밋됨, 플래그 OFF)

`Config.PERSISTENT_QUEUE`(env `GP8_PERSISTENT_QUEUE`, 기본 OFF, non-prepositioned만). ON이면
pick+ambush-hold+throw를 **하나의 큐 세션**(`pq_begin/pq_segment/pq_hold_until/pq_finish/
pq_throw_segment`)으로 스트리밍 → 재진입 #2 제거.

- HW 검증됨: mid-stream append GREEN, hold-across-wait(3s) GREEN(스파이크).
- 실 로그(155311, row 14): 재진입 제거 확인(**qmode_ms=0**) — **하지만 err_s가 0.24→0.64로 늘어
  순 절감 0(net-zero)**. 이유: throw가 **hold 버퍼(≈lead 0.35 + startup) 뒤에 큐잉**돼 그만큼
  지연. 즉 재진입을 hold 버퍼가 1:1 대체.
- 순 절감을 내려면: **hold 페이싱을 측정 motion-start에 앵커링 + lead 축소**(버퍼 lead+D→lead,
  lead ~0.12–0.15s) → throw가 hold 후 ~0.15s만에 시작 → 순 ~0.25–0.4s/cycle.
- 안전: 적대적 리뷰(3렌즈)로 CRITICAL 다수 잡고 fail-safe 수정(pq_* 반환코드 전파 + clean-abort +
  committed_next 롤백; 릴리스 레이어드 검출). **미해결(플래그 OFF 유지)**: planning-gap 드레인은
  현재 clean-abort(성공시키려면 precompute/salvage), 릴리스 타이밍 위치폴링(첫 HW서 착지점 튜닝),
  prepositioned 사이클 미적용(전체 절감엔 cross-cycle 필요).

관련 커밋: `ac8712a`(로깅) `9ddfcdd`(프리미티브) `a18cfcf`(BUSY 백프레셔) `2c932e0`(세션)
`e3ffa0f`(throw 연결+fail-safe) `04c17fa`(persistent 로깅) / 측정도구 `763d952` `4d415f2`.

---

## 6. 다음 단계 (우선순위)

1. **hold 페이싱 앵커링 + lead 축소** — persistent throw의 net-zero를 실제 ~0.25–0.4s/cycle 절감으로.
2. **planning-gap 처리** — precompute-at-lock(hold 중 throw 미리 계산) 또는 non-persistent salvage.
3. **릴리스 결정론화** — 위치폴링 대신 세션시간 기반(측정 motion-start 오프셋) time-anchored 릴리스.
4. **cross-cycle 세션** — prepositioned 사이클까지 재진입 제거(현재 non-prepositioned만) = 전체 절감.
5. **async 푸시(Stage A)** — per-point 푸시 축소(startup과 겹치지만 여지 있음).

---

## 7. 한 줄 결론

**"큐 지연 0"은 불가**(컨트롤러 startup ~0.15s는 정지 출발마다의 불가피 바닥). 그러나 큐의 큰
비효율(**재진입 ~0.4s/디스패치**)은 persistent로 제거 가능하고, **예측 가능한 부분은 미리
밀어넣어(pre-load) 딜레이를 사이클타임에서 안 보이게** 만들 수 있다. 남는 건 반응형 최소치뿐.

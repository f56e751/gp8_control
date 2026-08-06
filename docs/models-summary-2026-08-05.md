# 던지기 모델 3종 — 확정 사항 정리 (2026-08-05, **08-06 대폭 갱신**)

세 모델 모두 같은 `traj_fn(target, p_start, v_start, ctx) → dict(ts, Q, Qd,
t_rel|release_window, q_start, info)` 인터페이스로 Yaskawa GP8 에 붙는다.
이 문서는 2026-08-06 까지 **실측으로 확정된** 사실만 적는다.

## 0. 공통 (rig / 게이트)

| 항목 | 값 | 비고 |
|---|---|---|
| 툴 길이 | 0.240 m (link6→TCP 0.320) | URDF/THR 기준 |
| 스트리밍 | `/JointGroupPositionController/commands`, 4 ms 위치 | 스트리밍 자체 속도상한은 **고려하지 않음** (사용자 지시) |
| 프레임 | `q_robot = q_planner · SIGN`, SIGN=[1,1,−1,−1,−1,−1] | FK 대조 0.000000 m 검증 |
| 관절 한계 | URDF: L −65~+145°, U −70~+190°, B ±135° | |
| 속도 한계 | URDF/데이터시트 455/385/520/550/550/1000 °/s | |
| 가속도 한계 | **NLP: 속도×3** = 1155/1560/1650 °/s². **DT: 명시 한계 없음** (2026-08-06 B 전환 — 공식 Gazebo 사슬로 창발, 순간 5,600 °/s²+) | DT 의 구 box(3×) 규약은 `DT_GP8_DYN=box` 로 보존 |
| Cartesian | TCP **x > 0.20, z > 0.02** 반평면 2개 — 단일 출처 `throwing.py TCP_X_MIN/Z` | NLP 제약·DT 필터·실기 게이트 세 곳 동일값 (실측으로 어긋남 확인 후 통일) |
| 실기 게이트 | ① 관절 ② 속도(URDF) ③ Cartesian(보고, `GP8_THR_CART_BLOCK=1` 차단) ④ 착탄 예측 | `thr_throw_skill.check_arc` |

로봇 스캔 실측 (p_start 0.45/0.20/0.02 → 1.35/0.075/−0.08, 2026-08-05 저녁):

| 모델 | 예측 착지오차 | 릴리즈 민감도 (per 10 ms) | 윈도우 | 계획 시간 |
|---|---|---|---|---|
| nlp | **0.1 cm** | **0 mm** | **±50 ms** | 3~7 s (DB 적중 시) |
| dt | 0.7 cm | 2 mm | 없음 | 1.3 s |
| phy | 5.0 cm | 73 mm | 없음 | 0.1~0.3 s |

※ dt 행은 **box(3×) 규약 시절**(`official_k0`) 값. 2026-08-06 effort_pid 전환 후의 dt 는 §2-B — env 예측 오차 16~21 cm·민감도 86~192 mm 로 보이지만 전자는 실기 보정이 반영된 정상 수치다 (오독 주의).

## 1. NLP — CasADi/IPOPT B-spline (THR 자체)

**소스**: `THR/throw_nlp.py`, `THR/nlp_planner.py` (워크스테이션) →
로봇 `skills/thr/throw_nlp.py` 벤더링.

- **공식화**: B-spline 위치궤적 최적화. 관절·속도·가속도 한계를 **볼록껍질로
  연속시간 강제** — 실행 곡선 최대 가속도가 정확히 한계 1.0×에 얹힘 (실측
  1124/1548/1627 vs 한계 1155/1560/1650).
- **릴리즈 윈도우 ±50 ms**: 윈도우 전 구간의 착탄 정확도를 목적함수(W_ACC)로
  강제 → 지터에 구조적으로 강함. 세 모델 중 유일하게 윈도우를 가짐.
- **Cartesian 엔벨로프를 hard constraint 로 내장** (제약 3b, 릴리즈+버퍼까지
  sigmoid 게이트).
- **p_start 민감**: 물체를 집은 위치에서 스윙 시작 (DT 와 다름).
- **warm DB**: v2 축적 포맷 `{version:2, configs:[{params, entries}]}`.
  공식화 파라미터(qdd_lim, col=('env',0.20,0.02), …)가 키 — 공식화가 바뀌면
  entry 가 자동 격리된다. **2026-08-05 빌드: 12×12 = 144쌍 × 초기값 128,
  게이트 전 쌍 통과, 201.6분.** 로봇 `skills/thr/warm_db_thr.pkl` 연결 완료
  (polish 적중 시 계획 3~7 s, 미적중 시 multistart 수십 s).
- 제어: 위치 스트리밍 (240 Hz 궤적 → 4 ms 재샘플).

## 2. DT — Decision Transformer (공식 ThrowBot 충실 재현)

**소스**: 저자 본인 공식 저장소 **MaxorPaxor/ThrowBot** (RA-L 2023, IEEE
9984828, 커밋 43e8c1b — `/PublicSSD/ryugaeun/ThrowBot_official/` 보존).
재현 코드 `Thr_DT/gp8/` → 로봇 `skills/thr/` 벤더링.
**2026-08-05 사용자 지시로 공식 코드 외 추가·삭제 없음** (CEM prior·topup·
Cartesian 수집필터 제거됨). **2026-08-06 B 전환**: 가속도 한계(3×)와 feasible
box 를 버리고 공식 Gazebo 물리 사슬을 재현하는 effort_pid 동역학이 기본이
됐다 — §2-B 참조.

### 확정된 공식 규약

| 항목 | 값 | 공식 근거 |
|---|---|---|
| 모델 | state 5 (관절3+그리퍼+목표), act 4, K=20, embed 128, n_layer 1, **210,058 파라미터** | `train_dt_offline.py` 활성 라인 |
| 적분 | 식 (5.1) 직사각 `θ_t = θ_{t−1} + ω_t/f`, 10 Hz | §5.1 |
| **waypoint 속도** | **segment**: `(p_k−p_{k−1})/Δt` = 그 노드의 DT 출력 ω_k | `control_real_gp8.py:107-115` (재생 경로) |
| 관절 위치한계 | **URDF 전범위** (코드에 clip 없음 — Gazebo 하드스톱의 대응) | `gp8_macro_gazebo.xacro` |
| 속도한계 | 385/520/550 °/s, 행동 스케일링에 내장 (구조적으로 초과 불가) | `robot_env_dt.py:40` |
| 가속도한계 | 공식엔 없음 → **2026-08-06 부터 공식 그대로 재현** (`dyn_effort_pid.py`: URDF effort 176.4/107.6/32.7 N·m + 공식 PID 게인 + 1 kHz 관성 동역학, COM=관절원점의 조악한 공식 xacro 그대로) | `gp8_gazebo_controller.yaml`, xacro |
| 시뮬 수집 | 무작위 1000 에피소드, OU(σ0.6, θ0, dt0.2), 목표 U(0.5,2.0) 재추첨, 릴리즈 randint(0,9) | `collect_data_dt.py` 마지막 줄 |
| HER | k = −1/0/1/3/5 **각각 별도 버퍼** (합치지 않음) — **학습은 k=0 버퍼**(에피소드당 1궤적) | `agent_dt.remember`, 활성 호출 k_her=0 |
| 학습 | lr 3e-4, warmup 5000, 100 epoch × 100 step, eval-best 저장 | `train_dt_offline.py:237-253` |
| 파인튜닝 | 사전학습 로드 → 워밍업 2000 step → **10 iter × 100 step**, lr 3e-4, iter별 스냅숏 | `:189-232` |
| 추론 | 조건부 생성 R̂=1.0, τ(그리퍼 임계)=학습셋 통계 | `evaluate_dt.py` |

### 환경 대역 (Gazebo 없음 → 명시 구현)

- 보간: MotoROS 3차 Hermite — 저자 포크 `motoman_ps` `MotionServer.c:1414-1420`
  이식(`gp8_interp.py`), 계수 일치 대조 완료.
- **feasible box** (`dt_gp8_env.feasible_qdot_box`) — **box 모드 한정** (effort_pid 기본에선 미사용): 속도·가속도·위치(+제동
  여유) 교집합으로 명령을 **적분 전에** 투영, 실현값(`action_exec`)을 데모에
  기록. 없으면 라벨의 60.5%가 실행 불가값 (|라벨−실현| 평균 0.222 실측).
  segment 에서는 보간 가속도 증폭 4배(`ACC_RECON_GAIN`)를 나눠 실행 곡선이
  한계 1.0×에 정확히 얹힘 (실측 0.94~1.00×).
- 릴리즈 속도 = 구간 평균 `(q_k−q_{k−1})/Δt` (논문 미명시 — 순간속도를 쓰면
  1.70 m 예측이 1.04 m 착지, 실측으로 배제).

### 특성

- **p_start 를 쓰지 않는다** — 항상 home 자세에서 스윙 (물체 위치 무관).
- 조준: yaw = atan2(t_y, t_x) 를 S 관절로, d_g = 수평거리 (§5.1).
- 릴리즈 = 노드 시각 `(k_rel+1)·0.1 s`. 윈도우 없음.
- 학습 조건 재현에 **환경변수 3개 필수**: `DT_GP8_DYN=effort_pid`,
  `DT_GP8_LIMITS=urdf`, `DT_GP8_WP_VEL=segment` (dt sh·collect sh 가 export).
  waypoint 규약 전환(`zero`/`trapezoid`)은 **해당 규약으로 재학습한 가중치와
  짝**이어야 함 — 실행만 바꾸면 학습·실행이 어긋난다.

### 가중치 (로봇 `skills/thr/weights/`)

| 파일 | 내용 | bin 1.10/1.35/1.60 (env 예측) |
|---|---|---|
| `gp8_dt_dyn_ft_real12.pth` | **기본 (08-06)**. effort_pid 시뮬(k0) → **실기 12던지기 파인튜닝** (논문 절차 완주, τ=0.8295) | 0.94/1.15/1.41 — 목표보다 낮은 건 **실기 보정의 증거** (실기가 env 보다 +0.15~0.4 m 멀리 나감) |
| `gp8_dt_dyn_k0.pth` | effort_pid 시뮬 전용 베이스 (파인튜닝 시작점) | 12.8/23.6/35.8 cm |
| `gp8_dt_official_k0.pth` | 구 box(3×) 규약 학습분. **이름과 달리 저자 배포본 아님** — 저자 DT 가중치는 미공개(저장소엔 DDPG 4개뿐), 전부 우리 학습 | 14.5/0.7/13.1 cm |
| `gp8_dt_ft_realproxy-36.pth` | box + 시뮬 real-proxy 36 파인튜닝 (참고) | — |
| 구 v9/nolag 계열 | 비공식 규약 학습분 — **쓰지 말 것** | — |

### 성능 실측 (PyBullet ThrowSim, 12물체→12bin×3세션)

| 조건 | 명중 | 통과오차 |
|---|---|---|
| `_ideal` (지터 0, CoM 파지, 보간 ON) | **22/36 (61.1%)** | 평균 100 mm |
| `_real` (지터 ±50 ms, 무작위 파지, 보간 ON) — 파인튜닝 전 | 22/36 (61.1%) | 평균 111 mm |
| `_real` — **real-proxy 36던지기 파인튜닝 후** | **27/36 (75.0%)** | 평균 99 mm |

→ 소량 실데이터 파인튜닝으로 sim-to-real 격차가 흡수되는 논문 구조가 proxy
에서 재현됨 (+13.9%p). 실기 순서: `tools/collect_data_real_gp8.py`(공식 실기
루프의 ROS2+수동측정 이식) → `dt_finetune_gp8.py`(공식 레시피) → 가중치 교체.

#### 8시드 재측정 (2026-08-06, n=96) — 위 3세션 값의 정밀판

3세션은 1σ≈8 %p라 파인튜닝 효과의 크기를 가르지 못한다. 같은 규약
(`DT_GP8_LIMITS=urdf`, `DT_GP8_WP_VEL=segment`)으로 8시드를 다시 쟀다
(`THR/run_spec_bench.sh` → `summarize_3model.py`).

| 조건 | 명중 | 통과오차 중앙 |
|---|---|---|
| `_ideal` — 파인튜닝 전 | 61/96 (63.5%) | 95 mm |
| `_real` — 파인튜닝 전 | 59/96 (61.5%) | 79 mm |
| `_ideal` — 파인튜닝 후 | 63/96 (65.6%) | 69 mm |
| `_real` — **파인튜닝 후** | **65/96 (67.7%)** | 81 mm |

- 3세션 값(61.1 / 61.1%)과 8시드 값(63.5 / 61.5%)은 정합한다.
- **파인튜닝 효과는 +6.2 %p** (현실 61.5 → 67.7%). 위 표의 +13.9 %p는 n=36
  값이고 1σ≈8 %p라 두 값은 통계적으로 구분되지 않는다 — **개선 방향은
  재현되나 크기는 아직 확정할 수 없다.**
- **지터 하락이 −2.1 %p (파인튜닝 후 +2.1 %p)로 사실상 무반응이다.** DT는
  릴리즈를 노드 시각에 두어 그 근방 각속도가 완만하기 때문이며, §3의 PHY
  (73 mm/10 ms)와 대조된다. 세 모델 중 타이밍 지터에 무너지는 것은 PHY 뿐이다.

3모델 비교(같은 조건, n=96): NLP 96.9 / 90.6%, DT 63.5 / 61.5%,
PHY 76.0 / 43.8%. 상세는 `docs/paper_experiments_2026-08-04.md` §4-3.

※ 위 성능 표들(3세션·8시드)은 전부 **box 규약 `official_k0`** 기준이다.
effort_pid 전환 후의 ThrowSim 벤치는 아직 없다.

### 2-B. effort_pid 전환 + 실기 파인튜닝 완주 (2026-08-06)

사용자 지시("가속도 한계를 없애고 공식 Gazebo 방식 반영, 논문 충실 최우선")로
DT 만 3× 규칙을 폐기하고 공식 물리 사슬을 재현했다 (`Thr_DT/gp8/dyn_effort_pid.py`
→ 로봇 `skills/thr/` 벤더링, `DT_GP8_DYN={effort_pid|box}` 스위치):

  10 Hz 위치명령(식 5.1) → JTC 선형램프 설정점 → PID 토크(공식 게인)
  → URDF effort 클램프 → 평면 3링크 관성 동역학(공식 xacro, COM=관절원점) → 1 kHz 적분

검증: 질점모델 TCP vs THR FK **0.000 mm**(50자세) · 정적 유지 드리프트 0.2° ·
명령 −550°/s 첫 스텝 실현(순간가속 ≥5,600°/s²) · 연산 0.1 s/시뮬초.

**box 규약의 포화 진단이 전환 동기**: box(3×÷보간증폭4)의 무작위 데이터는
착지 ≥1.6 m 가 0%라 원거리 조건화가 붕괴(k_rel=3 고정, α 게인도 계획 rollout
에서 잘림 — 실기 15던지기로 확인: 목표 1.3/1.5 에도 착지 1.15~1.20 고정).
늦은 릴리즈는 무작위 데이터에서 오히려 손해(k=9 중앙값 음수 = 뒤로 던짐)라
데이터만으로는 못 배우는 구조였다.

**논문 절차 완주** (①→②→③, 전부 2026-08-06):
| 단계 | 결과 |
|---|---|
| ① 시뮬 학습 `dyn_k0` (무작위 1000, k=0, 처음부터·시드 결정론 재현 확인) | best eval 25.4 cm. 무작위 착지 최대 2.081 m — **꼬리 열림** (box 는 1.405) |
| ② 실기 수집 — **dyn_k0 자신이** 12던지기 (α≤1.5, `real_throws_2026-08-06.pkl`) | 착지 1.21~2.42 m. **실기가 env 예측보다 +0.15~0.4 m 멀리** = reality gap |
| ③ 공식 레시피 파인튜닝 (워밍업 2000+10×100, lr 3e-4, HER K=0) → `dyn_ft_real12` **배포** | env 예측이 목표보다 낮아짐 = gap 보정 학습됨. 실기 검증 던지기는 아직 |

부수 확정 사항:
- **착지면 정합**: 논문은 z=0(그들 바닥), 우리는 `z_land=−0.08`(bin 조준면) —
  시뮬 학습·게이트·실기 측정·HER 이 전부 −0.08 로 일치해 **페널티 없음**
  (0 으로 뒀던 시절 오버슛 120~155 mm 실측 후 08-03 에 고정된 값).
- **sim-real 정합 실측**: box 궤적은 예측-실측 ~3 cm (실기 15던지기) — 3× 규약
  궤적은 서보가 정확히 추종. effort_pid 는 +0.15~0.4 m 오버슛 — 파인튜닝이 흡수.
- 릴리즈 민감도 86~192 mm/10 ms (box 의 ~30배, 가속 중 릴리즈) — 실기 산포가
  크게 보여도 예상 동작.
- 반복 도구: `finetune_real_gp8.py` (기본 = dyn_k0 베이스에서 누적 풀 재파인튜닝,
  공식 traject_cut 스윕 의미론 / `--filter-weights`·`--traject-cut` 지원).
  수집은 `--reps 1`(목표당 1회)·`--alpha-max 1.5`(논문 3.0, 안전 하향)가 기본.
- 릴리즈 타이밍: 실기 수집에서는 **모델의 그리퍼 채널 < τ 가 결정** (외부 고정/
  랜덤 아님 — 랜덤 releasestep 은 시뮬 무작위 수집 전용).

## 3. PHY — TossingBot Physics-only (Thr_Phy 재현)

**소스**: `Thr_Phy/run_thr_sim.py` + `tossingbot/{config,physics}.py` (T-RO
2020 §III-C) → 로봇 `skills/thr/tossingbot/` 벤더링 (상수까지 동기 확인).

- **해석적 탄도**: 릴리즈점 r = f(목표; c_h=0.30, c_d=0.60), 속도 방향 45°
  고정 계열, g=9.81. 학습·DB 없음, 결정론.
- **궤적**: 직선 가속 0.28 m → 릴리즈 → 팔로우스루 0.10 m → 감속 0.14 m.
  릴리즈가 정확히 격자점에 오도록 스텝 수 정수화.
- 계획 0.1~0.3 s — 세 모델 중 최속. 정확도 5.0 cm (예측 기준).
- **릴리즈 타이밍 민감도 73 mm/10 ms — 세 모델 중 최악** (직선 가속 중
  릴리즈라 속도가 계속 변하는 구간에서 놓음).
- 미해결: B축 실행 가속도 1754 °/s² = 한계 1.06배 초과 (`PHY_ACCEL_DIST=0.28`
  기인). 거리 상수 조정으로 해결 가능하나 아직 반영 안 됨.
- 알려진 재현 이슈: 논문 Eq.2 인쇄오류·pedestal 기하 주의 (Thr_Phy 문서).

## 4. 위치 (2026-08-05 저녁 기준)

| | 워크스테이션 | 로봇 (`gp8_control`) |
|---|---|---|
| NLP | `THR/` (nlp 전용으로 정리됨) | `skills/thr/throw_nlp.py` + `warm_db_thr.pkl` |
| DT | `Thr_DT/gp8/` (dyn_effort_pid·finetune_real_gp8·collect_data_gp8 포함) | `skills/thr/dt_*.py`, `dyn_effort_pid.py`, `weights/` |
| PHY | `Thr_Phy/` | `skills/thr/tossingbot/` |
| 공식 원본 | `ThrowBot_official/` (43e8c1b) | — |
| 비NLP 잔재 | `THR_archive_2026-08-05/` (bench_jitter, 영상, 결과 json) | — |

실행: `tests/run_static_pick_throw_{nlp,dt,phy}.sh` (공통 기계 `_thr.sh`,
`--thr-scan` 으로 무명령 점검). 커밋: main `affb25e` = feature/throwing
`eeb5bce` (08-05 저녁 푸시) — **08-06 의 B 전환분(dyn_effort_pid·가중치·sh)은
아직 미커밋.**

**실기 던지기 2026-08-06 실행됨**: box 모델 15던지기(오전, 참고용) + dyn_k0
12던지기(공식 절차 ②, 파인튜닝에 사용). `dyn_ft_real12` 배포 후의 검증
던지기는 아직 — 다음 실기 세션의 첫 일감.

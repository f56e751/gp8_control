# THR 던지기 모델 3종 + DT sim2real — 실행 순서 (runbook)

작성 2026-08-04. 대상: `skills/thr_throw_skill.py` (nlp / dt / phy) 와
`tools/collect_real_throws_gp8.py` → `tools/finetune_dt_gp8.py`.

> **현재 상태: 실기 던지기는 아직 한 번도 실행하지 않았다.**
> `--thr-scan`(계획 검증)과 `--plan-only`(궤적 조립 + 안전 게이트)까지만 통과했다.
> 아래 순서는 그 전제에서 시작한다. 0단계를 건너뛰지 말 것.

전체 배경·설계 근거는 각 파일 docstring에 있다:
- `skills/thr_throw_skill.py` — 모델 3종, 2026-08-03 업데이트 반영 내역, 기하 불일치
- `skills/thr_planners.py` — 로봇 이식에서 달라진 3가지, warm DB
- `tools/build_warm_db_thr.py` — warm DB 저장 정책
- `tools/collect_real_throws_gp8.py` — sim2real 수집 절차와 안전 장치

---

## 0. 매 세션 전제 (체크리스트)

- [ ] 펜던트: **E-stop 해제 / 알람 리셋 / REMOTE** (아니면 activation 거부 — CODE 10x)
- [ ] `gp8_manager` 앱이 **꺼져** 있을 것 (앱과 테스트가 둘 다 JGPC에 쓰면 충돌)
- [ ] 로봇 반경 **1 m 확보** — 실제 풀 스윙이다
- [ ] 착지 구역에 사람/장비 없음, bin 위치 확인
- [ ] 물체를 픽 지점에 배치 (기본 `0.40/0.50 × y` — sh의 `POINTS`)

driver 스택은 `run_static_pick_throw_*.sh` 가 알아서 띄운다 (이미 떠 있으면 재사용).

---

## 1. 실기 첫 던지기 — 모델별 동작 확인

**목적**: 계획→실행 경로가 실제로 도는지, 스윙/릴리즈 타이밍이 눈으로 보기에 정상인지.
아직 데이터를 모으는 단계가 아니다.

### 1-1. 로봇 없이 먼저 (필수)

```bash
cd ~/ros2_ws/src/gp8_control/tests
./run_static_pick_throw_phy.sh --thr-scan     # 가장 빠름
./run_static_pick_throw_dt.sh  --thr-scan
./run_static_pick_throw_nlp.sh --thr-scan     # warm DB 적중이면 쌍당 1~4초
```

- [ ] 세 모델 모두 목표 6/6 통과 확인
- [ ] `nlp` 로그에 `warm DB: ... entry 144개 (현재 공식화 일치)` 가 뜨는지
      (안 뜨면 공식화가 바뀐 것 — 5-A 참고)

기대값 (예측 착지오차, 2026-08-04 실측):

| 모델 | 1.10 m | 1.35 m | 1.60 m |
|---|---|---|---|
| phy | 2.2 cm | 2.4 cm | 4.9 cm |
| dt (v9) | 6.2 cm | 7.9 cm | 1.2 cm |
| nlp (warm) | 0.6 cm | 7.8 cm | 4.5 cm |

### 1-2. 궤적 조립까지 (로봇 무명령)

```bash
./run_static_pick_throw_phy.sh --plan-only
```

- [ ] `ABORTED` 없이 `throw dispatch` 가 뜨는지
- [ ] `계획 성공 N/N`

### 1-3. 실기 (여기서 처음 움직인다)

```bash
./run_static_pick_throw_phy.sh      # --confirm-throw 가 기본
```

- [ ] init(idle) 이동이 정상인지 먼저 보고, 이상하면 즉시 중단
- [ ] 사이클마다 Enter 확인 — **첫 사이클은 `s`로 건너뛰지 말고** hover/press까지
      보고 나서 던지기 Enter
- [ ] 물체가 목표 방향으로 날아가는지, 착지점이 로그의 `예측 착지`와 대략 맞는지
- [ ] `dt`, `nlp` 도 같은 방식으로 1회씩

⚠ 중단이 필요하면 Ctrl-C — 어느 경로로 끝나도 suction OFF가 나간다.

**여기까지 통과해야 2단계로 간다.** 예측과 실제가 크게 다르면(>30 cm) 먼저
5-B(기하 불일치)를 의심할 것.

---

## 2. DT 실기 데이터 수집

**목적**: `mem`(상태·액션열) + **실측 착지거리** 쌍을 모은다. 이것이 HER의 재료다.

```bash
cd ~/ros2_ws && source install/setup.bash
PYTHONPATH=$HOME/ros2_ws/src ~/ros2_ws/src/gp8_control/.venv/bin/python \
    -m gp8_control.tools.collect_real_throws_gp8 \
    --goals 1.1,1.3,1.5,1.7 --reps 5
```

- [ ] 목표 격자를 `1.1,1.3,1.5,1.7` 로 시작 (드라이런상 **0.9 m는 α=1.0까지
      낮춰도 게이트를 못 넘어 자동 skip** 된다 — 넣어도 데이터가 안 쌓인다)
- [ ] 4구간 × 5회 = **20 던지기**. 소요 1~1.5시간 (측정 시간 포함)
- [ ] 매 던지기 후 **착지 거리 입력**
      - 규약: **베이스 회전축(J1) 중심 → 물체가 처음 떨어진 지점**의 수평거리 [m]
      - `1.23` 또는 `1.20,0.10`(x,y) 둘 다 가능
      - 빈 줄 = 그 던지기 버림 (물체가 튀어 나가 측정 불가 등)
- [ ] 누적 히스토그램이 한쪽으로 쏠리지 않는지 매번 확인 (자동 출력됨)

안전 동작 (자동, 개입 불필요):
- α ~ U(1.0, **1.5**) — 논문 3.0 대신 (α는 모터 액션 배수라 실기에선 속도가 그만큼 오른다)
- 게이트 실패 시 α를 ×0.9씩 낮춰 재시도 → α=1.0에서도 실패하면 그 던지기 skip
- 던지기 한 번마다 즉시 저장 → **중단해도 유실 없음. 재실행하면 이어붙는다**

출력: `data_real/real_throws_<날짜>.pkl`

> 20회로 부족하면 같은 명령을 다시 돌리면 된다 (같은 파일에 누적).
> `--goals` 를 바꿔 범위를 넓혀도 된다.

---

## 3. HER 재라벨 + 파인튜닝

```bash
PYTHONPATH=$HOME/ros2_ws/src ~/ros2_ws/src/gp8_control/.venv/bin/python \
    -m gp8_control.tools.finetune_dt_gp8 \
    --data ~/ros2_ws/src/gp8_control/data_real/real_throws_<날짜>.pkl
```

- [ ] 학습 전 출력되는 **착지 분포**를 볼 것. 5구간 중 3개 미만이면 학습이 막힌다
      (`--force` 로 무시 가능하지만 권하지 않는다 — 아래 이유)
- [ ] `env 예측 대비 실제 편차` 확인 = 파인튜닝이 흡수할 reality gap
- [ ] loss가 내려가는지 (드라이런: 0.145 → 0.106)

출력: `skills/thr/weights/gp8_dt_real-<N>.pth`

> **왜 분포가 중요한가**: HER K=0은 "목표 := 실제 착지점"이라 **모든 궤적이
> 성공(+1)** 이 된다. 즉 데이터가 가르치는 것은 '어디를 맞히는 법'이 아니라
> '이 액션열은 실제로 이만큼 날아간다'이다. 따라서 **데이터의 착지 분포가 곧
> 모델이 배우는 사거리 분포**다. 실제로 기존 시뮬 프록시 데이터
> (`data_gp8/real_gp8_n-12_herK-0.pkl`, 11샘플 중 7개가 1.42 m 이상)로 만든
> `gp8_dt_ft2_real-11.pth` 는 1.10 m 목표에도 1.67 m 를 던진다.

---

## 4. 전/후 비교 → 채택 여부 결정

```bash
cd ~/ros2_ws/src/gp8_control/tests
./run_static_pick_throw_dt.sh --thr-scan                                    # 전 (v9)
./run_static_pick_throw_dt.sh --thr-scan --weights <위에서 나온 .pth>        # 후
```

- [ ] 예측 착지오차가 목표 전 구간에서 줄었는지 (v9 기준: 6.2 / 7.9 / 1.2 cm)
- [ ] `실행 가능 N/N` 이 줄지 않았는지 (줄었으면 파인튜닝이 궤적을 게이트 밖으로 밀어낸 것)
- [ ] `--plan-only` 로 게이트 통과 재확인
- [ ] 실기 1회 던져서 실제 착지 확인

채택하려면 기본 가중치를 바꾼다:

```bash
# tests/run_static_pick_throw_dt.sh 의 GP8_THR_DT_WEIGHTS 기본값을 수정하거나,
export GP8_THR_DT_WEIGHTS=$HOME/ros2_ws/src/gp8_control/skills/thr/weights/gp8_dt_real-20.pth
```

- [ ] 좋으면 커밋 (`.pth` + `data_real/*.pkl`)

**나빠지면** 2단계로 돌아가 목표 범위를 넓히거나 횟수를 늘린다. 파인튜닝은
사전학습 가중치에서 매번 새로 시작하므로 여러 번 시도해도 누적 열화는 없다.

---

## 5. 알려진 이슈 / 참고

### 5-A. NLP warm DB가 안 잡힐 때

`nlp` 로그에 `현재 공식화와 일치하는 config 없음 — cold 진행` 이 뜨면 공식화
파라미터(tool, rt, W1, W2, W_ACC, QDD, 관절한계, GRIP_OFF …)가 DB 빌드 시점과
달라진 것이다. cold는 지점당 12~30초 걸린다 (동작은 한다).

재빌드 (약 2.6시간, 7워커):
```bash
nohup setsid bash -c 'source /opt/ros/humble/setup.bash && \
  source $HOME/ros2_ws/install/setup.bash && \
  PYTHONPATH=$HOME/ros2_ws/src $HOME/ros2_ws/src/gp8_control/.venv/bin/python \
  -m gp8_control.tools.build_warm_db_thr --n-init 128 --workers 7' \
  > /tmp/warm_db_thr_build.log 2>&1 < /dev/null &
tail -f /tmp/warm_db_thr_build.log
```
DB는 공식화별로 config가 공존하므로 예전 것을 덮어쓰지 않는다.

### 5-B. 기하 불일치 (미해결, 실기 확인 필요)

| 경로 | link6 → TCP |
|---|---|
| THR (던지기 조준·게이트) | **0.320 m** (d6 0.080 + tool 0.240) |
| `robots/gp8.py` (픽 IK) | 0.300 m (home_ee x = 0.680) |

사용자 확인(2026-08-04)상 실물 로드는 24 cm 이므로 THR 쪽이 맞다. 그런데
`robots/gp8.py` 는 **일부러 두었다** — 그 FK로 픽 높이(`PRESS_Z` 등)가 실기에서
경험적으로 튜닝돼 있어 지금 고치면 그 튜닝이 통째로 어긋난다.

맞추려면: `gp8.py` 의 `home_ee` x를 0.700으로 올리고 → 픽 높이 전부 재튜닝 →
구 NLP 스킬의 warm DB(`skills/warm_db_*.pkl`) 재빌드. **별건으로 다룰 것.**

실기 착지가 예측과 계통적으로 어긋나면(항상 2~3 cm 길거나 짧음) 여기를 의심한다.

### 5-C. NLP 먼 bin의 속도 상한

2026-08-03 업데이트(W1 0.5→5.0, QDD 3×→5×)로 스윙이 빨라져, 먼 bin에서 J3가
341~372 °/s까지 올라간다 (컨트롤러 스트림 상한 259 °/s). warm DB가 **게이트를
통과하는 해**를 저장하므로 지금은 144/144 통과하지만, DB 없이 cold로 풀면
1.35/1.60 m가 거부된다. 그때는:

```bash
GP8_THR_NLP_W1=0.5 ./run_static_pick_throw_nlp.sh    # 스윙을 느리게
```

### 5-D. 릴리즈 타이밍

- `nlp` 만 릴리즈 **윈도우 ±50 ms** 가 있다 (그 안 어디서 놓아도 착탄 보장)
- `dt` / `phy` 는 릴리즈가 **한 점** — 로그의 `민감도 N mm/10 ms` 가 밸브 지터
  1스텝당 착지 변화량이다 (실측 dt 8~175, phy 40~114 mm/10 ms)
- 밸브 지연을 실측했다면 `GP8_RELEASE_LEAD` 에 **양수**로 넣는다 (그만큼 일찍 명령)

### 5-E. 브랜치

작업 트리는 `main` 에 있고, 커밋은 `feature/throwing` 에도 cherry-pick 되어
푸시된다 (내용 동일). `main` 은 origin 대비 앞서 있고 아직 푸시하지 않았다.

---

## 부록: 모델 3종 요약

| | nlp | dt | phy |
|---|---|---|---|
| 방식 | CasADi/IPOPT B-spline | Decision Transformer | TossingBot 탄도 |
| 목표 입력 | 3D 목표 | 거리 d_g만 (방향은 J1) | 3D 목표 |
| 릴리즈 | 윈도우 ±50 ms | 점 | 점 |
| 시작 자세 영향 | **있음** (p_start) | 없음 (학습 home 고정) | 없음 (해석적 릴리즈점) |
| 계획 시간 | warm 1~4 s / cold 12~30 s | ~1 s | ~0.2 s |
| THR 시뮬 (real 조건) | 12/12, 32 mm | 11/12, 84 mm | 7/12, 138 mm |

`--thr-scan` 은 로봇에 아무 명령도 보내지 않으므로 언제든 돌려도 안전하다.

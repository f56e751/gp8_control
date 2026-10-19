# 오프라인 DPPO — GP8 재활용 시뮬레이터

시뮬레이터에서 미리 모아 둔 로그만으로 "벨트 위의 어떤 물체를 어떤 기술(던지기 / 밀기)로
처리할지"를 정하는 정책을 학습합니다. 학습 방법은 DPPO(Direct Preference-based Policy
Optimization, [snu-mllab/DPPO](https://github.com/snu-mllab/DPPO))를 이 환경의 이산 행동에
맞게 옮긴 것이고, 같은 데이터로 학습한 IQL과 BC를 비교 기준으로 함께 제공합니다.

이 폴더는 `gp8_control`의 시뮬레이터 환경(`rl/gym_env.py`)을 가져다 쓰기만 하며, 폴더 밖의
코드는 수정하지 않습니다.

## 결과

240초 에피소드의 반환(통에 넣으면 +1, 놓치거나 실패하면 −0.3)입니다. 각 숫자는 학습 seed
10개의 평균이고, 괄호는 학습 seed별 평균의 최저–최고입니다. 학습된 정책 하나는 시뮬레이터
seed 0~9의 10개 에피소드로 평가했습니다.

| 학습 데이터 | 학습 스텝 | DPPO | IQL | BC |
|---|---|---|---|---|
| 100 에피소드 | 5,000 | **75.01** (72.22–77.82) | 71.59 (69.65–72.70) | 72.08 (68.52–75.20) |
| 200 에피소드 | 4,000 | **81.11** (79.02–83.36) | 76.58 (71.25–80.10) | 74.28 (72.01–75.59) |
| 500 에피소드 | 10,000 | **81.96** (80.66–83.29) | 75.14 (72.35–78.51) | 74.27 (72.40–76.96) |
| 1,000 에피소드 | 20,000 | **82.12** (80.25–83.60) | 78.02 (76.05–79.95) | 73.99 (72.81–76.02) |

기준선(시뮬레이터 seed 0~9): 휴리스틱 정책 74.46, 무작위 정책 50.08. 학습 데이터 자체의
에피소드 평균 반환은 60.31입니다.

- 세 방법 모두 같은 데이터, 같은 정책 네트워크, 같은 학습 스텝 수, 같은 학습률 감소(코사인)를
  씁니다.
- IQL은 학습률을 고정하면(`--lr-schedule constant`) 73.03 / 72.33 / 77.94 / 79.72(위 표와 같은 순서,
  학습 seed 10개)가 나옵니다. 크기마다 두 조건 중 높은 쪽을 골라도 DPPO가 2.0~4.5점 높습니다.
- 100 에피소드에서는 DPPO가 휴리스틱(74.46)과 비슷한 수준입니다.

### DPPO 구성 요소별 비교

학습률 감소 조건, 학습 seed 10개.

| 구간 점수 | 구간 길이 | 100 | 200 | 500 | 1,000 |
|---|---|---|---|---|---|
| 선호 예측 모델 (기본값) | 3 | 75.01 | 81.11 | 81.96 | 82.12 |
| 반환 − baseline | 2 | 77.05 | 79.94 | 80.93 | 81.37 |
| 반환 − baseline | 3 | 75.60 | 79.20 | 79.78 | 79.52 |

데이터가 200 에피소드 이상이면 선호 예측 모델이 가장 좋고, 100 에피소드에서는 선호 예측
모델을 학습시킬 데이터가 부족해 `--score baseline --seg-len 2`가 더 좋습니다.

1,000 에피소드에서 따로 확인한 것. 기준 설정은 `--score baseline --seg-len 2 --lr-schedule constant`입니다.

| 바꾼 것 | 반환 | 학습 seed |
|---|---|---|
| 기준 설정 | 80.06 | 10 |
| 선호 margin 없음 (`--pref-margin 0`) | 78.85 | 10 |
| 결정–보상 정렬 없음 (`--decision-lag 0`), margin 없음 | 75.39 | 5 |
| 학습률 감소 추가 | 81.37 | 10 |
| 학습 4만 스텝 + 학습률 감소 | 80.84 | 10 |
| BC anchor 0.02 | 79.74 | 10 |
| 원본 DPPO에 가까운 설정(구간 20~50, 배치 64~256, anchor 없음, 정렬 없음, 상한 있는 거리 `1 − π(a\|s)`) | 66.3–69.0 | 3 |

listwise(InfoNCE) 형태의 loss, 덜 선호된 구간의 행동 확률만 낮추는 loss, 덜 선호된 구간에서
정책을 균등 분포로 보내는 loss도 시험했지만 76.4~79.0으로 위 기준 설정을 넘지 못했습니다. 이
loss들과 위 표 마지막 줄의 상한 있는 거리는 코드에서 뺐으므로, 지금 코드로는 재현할 수 없습니다.

### 평가 방법에 대해 알아 둘 점

- DPPO의 설정(구간 길이, margin, 점수 방식, 학습 길이 등)을 고를 때와 위 표를 만들 때 같은
  시뮬레이터 seed 0~9를 사용했습니다. 설정 선택에 쓰지 않은 seed로 다시 평가한 숫자는
  없습니다.
- IQL의 expectile 0.9 / temperature 10은 256 에피소드 데이터에서 고른 값이고, 데이터 크기별로
  다시 고르지는 않았습니다. IQL 논문 기본값(0.7 / 3)으로는 1,000 에피소드에서 74.14(학습 seed
  5개, 학습률 고정)입니다.
- 학습 스텝 수는 데이터가 작을수록 줄였습니다(200 에피소드 이상은 에피소드 수 × 20 스텝,
  100 에피소드는 5,000 스텝). 스텝 수를 2만으로 고정하면 100 에피소드에서 DPPO 67.90,
  IQL 69.18로 둘 다 과적합합니다(학습 seed 5개, 학습률 고정, 반환 − baseline, 구간 2).
  세 방법에 같은 스텝 수를 주지만, 한 스텝에 보는 표본 수는 DPPO 96개(구간 32개 × 3),
  BC 64개, IQL 256개입니다.

## 방법

1. **데이터** (`collect_offline_data.py`). 앱의 휴리스틱 정책(금속은 밀기, 투명은 던지기)에
   무작위 행동을 섞어 수집합니다. 무작위 행동을 고를 확률은 결정 20개마다 0 / 0.2 / 0.5 / 1.0
   중 하나로 다시 뽑습니다. 에피소드 하나는 240초이고 결정이 평균 129개 들어 있습니다.
2. **구간과 정렬** (`offline.py::load_segments`). 로그를 연속된 결정 3개짜리 구간으로 자르고,
   구간의 반환은 `env.step`이 돌려준 보상을 그대로 더합니다. 이 환경은 t에 고른 행동을 t+1에
   실행하므로, 보상 구간 [i, i+3)에는 그 안에서 **실행된** 결정, 즉 [i−1, i+2)에 고른 결정을
   짝짓습니다(`--decision-lag 1`). 보상 값은 바꾸지 않으며, 환경이 `info["credit_moves"]`로
   알려 주는 보상 재배치도 쓰지 않습니다.
3. **선호 예측 모델** (`offline.py::fit_preference_scores`). 구간 쌍에 "반환이 높은 쪽이
   선호된다"는 라벨을 붙이고(반환 차이가 표준편차의 0.5배보다 작은 쌍은 제외), Bradley–Terry
   loss로 `score(구간) = V(첫 관측) + Σ A(s, a)`를 학습합니다. 구간의 순위에는 행동이 기여한
   부분 `Σ A(s, a)`만 씁니다. 좋은 선택을 했는데 흡착이 운 나쁘게 실패한 구간도, 비슷한 구간들의
   결과가 평균되어 제 순위를 받습니다. 에피소드를 5묶음으로 나눠, 각 구간은 자기 에피소드를
   보지 않은 모델이 채점합니다.
4. **정책 학습** (`offline.py::dppo_loss`). 배치 안에서 구간 i가 j보다 선호될 때마다
   `−log[e^{−d_i} / (e^{−d_i} + e^{−λ d_j})]`를 줄입니다. `d`는 구간에 기록된 행동의 평균
   masked NLL입니다. 점수 차이가 점수 표준편차의 1배(`--pref-margin`)보다 작은 쌍은 선호
   관계로 치지 않습니다. NLL은 상한이 없으므로 `--bc-anchor × 평균 NLL`을 더합니다.
5. **네트워크** (`offline.py::OfflinePolicy`). 물체별 토큰과 전역 토큰 하나를 3층 Transformer에
   넣고, 물체 토큰에서 그 물체에 대한 (던지기, 밀기) logit을, 전역 토큰에서 건너뛰기 logit을
   읽습니다. IQL의 Q, V와 선호 예측 모델도 같은 몸통을 씁니다.

### 원본 DPPO와 다른 점

| | 원본 DPPO | 이 구현 |
|---|---|---|
| 행동 | 연속 | 이산 (물체 슬롯 × 기술) |
| 구간 거리 | 정책 행동과 기록된 행동의 L2 | 기록된 행동의 평균 masked NLL |
| BC anchor | 없음 | 0.1 (NLL에 상한이 없어서 추가) |
| 구간 길이 / 배치 | 100 / 256 | 3 / 32 |
| 선호 쌍 | 모든 쌍 | 점수 차이가 margin보다 큰 쌍만 |
| 결정–보상 정렬 | 없음 | 한 스텝 (행동이 한 스텝 뒤에 실행되는 환경) |
| 선호 라벨 | 사람 또는 스크립트 | 스크립트 (구간 반환 비교) |

## 실행

파이썬 환경은 레포 루트에서 `uv sync --extra sim --extra rl`로 만듭니다. 모든 명령은
`gp8_control`의 **상위 폴더**에서 실행합니다.

시뮬레이터 설정(벨트·물체 무작위화, 물리 벨트, 흡착 성공률 금속 0.5 / 투명 0.9 등)은
`simenv.py`의 `SIM_PROFILE`에 있고, 수집과 평가 스크립트가 자동으로 적용합니다. 같은 이름의
환경 변수가 이미 설정되어 있으면 그 값이 우선하며 경고가 출력됩니다.

```bash
PY=gp8_control/.venv/bin/python
D=gp8_control/examples/dppo
RUNS=$D/runs                                  # 산출물 폴더 (git에서 제외됨)

# 1. 데이터 수집: 결과에 쓴 1,000 에피소드. 에피소드 하나에 CPU 5~8분.
head -n 256  $D/data_seeds_1000.txt > seeds_a.txt       # seed 1000~1255
tail -n +257 $D/data_seeds_1000.txt > seeds_b.txt       # 나머지 744개
$PY -m gp8_control.examples.dppo.collect_offline_data --out-dir $RUNS/data_1000 \
    --seeds-file seeds_a.txt --workers 10
$PY -m gp8_control.examples.dppo.collect_offline_data --out-dir $RUNS/data_1000 \
    --seeds-file seeds_b.txt --skill-flip 0 --workers 10
$D/scripts/make_subsets.sh                              # data_100 / data_200 / data_500

# 2. 학습 (기본값이 위 표의 설정)
$PY -m gp8_control.examples.dppo.train_dppo --data "$RUNS/data_1000/ep_*.npz" --out $RUNS/dppo.pt
$PY -m gp8_control.examples.dppo.train_iql  --data "$RUNS/data_1000/ep_*.npz" --out $RUNS/iql.pt
$PY -m gp8_control.examples.dppo.train_dppo --data "$RUNS/data_1000/ep_*.npz" --out $RUNS/bc.pt \
    --mode bc

# 3. 평가: 에피소드별 결과가 JSON 한 줄씩 --out 파일에 추가됨
$PY -m gp8_control.examples.dppo.eval_offline_policy --ckpt $RUNS/dppo.pt --tag dppo \
    --seeds 0-9 --workers 5 --out $RUNS/test.dppo.jsonl
$PY -m gp8_control.examples.dppo.eval_offline_policy --policy heuristic --tag heuristic \
    --seeds 0-9 --workers 5 --out $RUNS/test.heuristic.jsonl
```

- 데이터 크기가 다르면 `--steps`를 위 표의 값으로 바꿉니다(100: 5000, 200: 4000, 500: 10000).
- 수집을 둘로 나누는 이유: 처음 256개 에피소드는 `--skill-flip`(휴리스틱이 고른 물체의 기술을
  바꾸는 탐색 옵션)이 생기기 전에 모았고, 나머지 744개는 그 옵션이 확률 0으로 켜진 상태에서
  모았습니다. 확률이 0이어도 난수를 하나씩 쓰기 때문에, 위처럼 나눠야 같은 seed에서 같은
  에피소드가 나옵니다. 수집 방식 자체는 두 묶음이 같습니다.

표 전체(방법 3개 × 크기 4개 × 학습 seed 10개)는 SLURM으로 다시 만들 수 있습니다.

```bash
gp8_control/examples/dppo/scripts/make_table_tasks.sh > tasks.txt
sbatch --array=1-$(wc -l < tasks.txt) gp8_control/examples/dppo/scripts/slurm_task.sh tasks.txt
gp8_control/examples/dppo/scripts/summarize.sh gp8_control/examples/dppo/runs/table \
    dppo_n100 iql_n100 bc_n100 dppo_n1000 iql_n1000 bc_n1000
```

`slurm_task.sh`는 작업 하나에 GPU 1개 / CPU 10개 / 메모리 60GB를 요청합니다(연구실 클러스터
규칙). 로그인 노드에서는 학습이나 시뮬레이션을 돌리지 않습니다.

### 재현성

- 같은 종류의 노드에서는 같은 데이터와 같은 seed로 학습, 평가, 수집이 모두 비트 단위로
  재현됩니다. 이 폴더의 코드로 DPPO(두 점수 방식), IQL, BC를 다시 학습해, 평가 에피소드
  10개의 반환이 정리 전 코드의 실행과 하나씩 일치하는 것을 확인했고, 수집도 seed 4개에서 저장된
  에피소드와 일치했습니다.
- 노드 종류(CPU / GPU)가 다르면 부동소수점 연산의 마지막 자릿수가 달라지고, 그 차이가 학습
  중에 커집니다. 200 에피소드, 학습 seed 0의 DPPO는 한 노드 종류에서 80.53, 다른 노드
  종류에서 82.33이 나왔습니다. 위 표는 여러 노드에서 돌린 학습 seed 10개의 평균입니다.
- 학습은 CPU 연산 스레드를 1개로 고정하고 돌렸습니다(`slurm_task.sh`). 스레드 수가 바뀌면
  선호 margin을 정하는 표준편차 계산의 마지막 자릿수가 달라질 수 있습니다.

## 파일

| 파일 | 내용 |
|---|---|
| `offline.py` | 정책 네트워크, 구간 생성과 정렬, 선호 예측 모델, baseline, DPPO loss |
| `train_dppo.py` | DPPO 학습. `--mode bc`로 BC 학습 |
| `train_iql.py` | IQL 학습 (비교 기준) |
| `eval_offline_policy.py` | 시뮬레이터 평가 (체크포인트 / 휴리스틱 / 무작위) |
| `collect_offline_data.py` | 오프라인 데이터 수집 |
| `simenv.py` | 수집과 평가가 함께 쓰는 시뮬레이터 설정, 환경 생성, 행동 인코딩 |
| `data_seeds_1000.txt` | 결과에 쓴 1,000 에피소드의 seed. 앞에서부터 100 / 200 / 500개가 작은 데이터셋 |
| `scripts/make_subsets.sh` | 100 / 200 / 500 에피소드 데이터셋 만들기 |
| `scripts/make_table_tasks.sh`, `slurm_task.sh`, `summarize.sh` | 표 전체 재현과 집계 |

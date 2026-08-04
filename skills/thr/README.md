# `skills/thr` — THR 던지기 모델 3종 벤더링 (2026-08-04)

`thr_planners.py` · `thr_throw_skill.py` 가 쓰는 **계획/추론 전용** 코드와 가중치.
`skills/throwing.py` · `throw_nlp.py` (구 NLP 스킬 전용, tool 0.220) 와 이름이
겹치므로 **절대 sys.path 로 노출하지 않는다** — 전부 패키지 상대 import 다.

## 파일 대응

| 여기 | 원본 | 변경 |
|---|---|---|
| `throwing.py` | `THR/throwing.py` | 없음 (byte-identical) |
| `throw_nlp.py` | `THR/throw_nlp.py` | import 1줄 |
| `dt_gp8_env.py` | `THR/dt_gp8_env.py` | import 1줄 |
| `dt_cem.py` | `THR/dt_cem.py` | import 3줄 |
| `dt_model/dt_config.py` | `Thr_DT/config.py` | 없음 |
| `dt_model/trajectory_gpt2.py` | `Thr_DT/model/trajectory_gpt2.py` | 없음 |
| `dt_model/decision_transformer.py` | `Thr_DT/model/decision_transformer.py` | import 1줄 |
| `dt_model/her.py` | `Thr_DT/agent/her.py` | 없음 |
| `dt_model/trainer.py` | `Thr_DT/agent/trainer.py` | 없음 |
| `dt_model/batch.py` | `Thr_DT/train_dt_offline.py` 의 `dataset_stats`·`make_get_batch` | 함수 2개 발췌 |
| `tossingbot/{__init__,config,physics}.py` | `Thr_Phy/tossingbot/` | 없음 |
| `weights/gp8_dt_best.pth` | `THR/weights_gp8/dt_best.pth` | 없음 |
| `weights/gp8_dt_best_v9.pth` | `THR/weights_gp8_v9/dt_best.pth` | 없음 |
| `weights/gp8_dt_ft2_real-11.pth` | `THR/weights_gp8_ft2/dt_finetuned_real-11.pth` | 없음 |

**import 변경 전부** (원본은 bare import + `sys.path.insert` 를 쓴다):

```
throw_nlp.py                : from throwing import (…)          → from .throwing import (…)   [2곳]
dt_gp8_env.py               : from throwing import (…)          → from .throwing import (…)
dt_cem.py                   : from dt_gp8_env import …          → from .dt_gp8_env import …   [2곳]
                              from throwing import fk_pos, …    → from .throwing import …
dt_model/decision_transformer.py
                            : sys.path.insert + from model.trajectory_gpt2 import …
                                                                 → from .trajectory_gpt2 import …
```

로직·상수·수식은 한 글자도 건드리지 않았다. 시뮬(`nlp_planner.py`, `sim/`,
`bench_jitter.py`)과 오프라인 대규모 학습(`dt_train_gp8.py`)은 가져오지 않았다.

`dt_model/batch.py` 만 파일 전체가 아니라 함수 2개 발췌인데, 원본
`train_dt_offline.py` 가 `env.throw_env`(평면 시뮬)와 `evaluate_dt` 를 import 하기
때문이다 — 로봇 rig 는 `dt_gp8_env`(GP8) 라서 평면 env 를 끌어올 이유가 없다.
이 3개(her/trainer/batch)는 **실기 sim2real 파인튜닝**
(`tools/collect_real_throws_gp8.py` → `tools/finetune_dt_gp8.py`)에만 쓰인다. `nlp_planner.py` 를 안 가져온 이유는 pybullet 의존(시뮬 dry-run
충돌 검사)과 fork 기반 병렬 multistart 때문이고, 그 두 가지를 뺀 순차판이
`skills/thr_planners.nlp_traj_fn` 이다 (거기 docstring 에 차이 3가지 명시).

## 이 벤더링이 담고 있는 2026-08-03 업데이트

```
GP8_DIMS tool     0.220 → 0.240   (link6→TCP 0.30 → 0.32 m, URDF 기준 통일)
RELEASE_TIME      0.05  → 0.1 s   (릴리즈 윈도우 2배)
QDD_LIM           3×    → 5×qḋ
W1                0.5   → 5.0
CART_CONSTRAINTS  True  → False   (NLP 에서 기둥 회피 제거)
관절 위치 한계     datasheet → URDF ∩ 실기 ∩ 사용자 규칙
                  planner: S |q|≤60°, L[−15,45], U[−30,45], B[10,135], |R|≤80°
```

⇒ `skills/warm_db_*.pkl` (구 NLP 스킬용) 은 이 공식화에서 전부 무효다. 이 경로는
warm DB 를 아예 쓰지 않는다.

## 재동기화

THR 을 다시 학습/수정했으면 위 표대로 복사하고 import 줄만 다시 고친다. 가중치만
바꿀 때는 `.pth` 만 덮어쓰거나 `GP8_THR_DT_WEIGHTS` 로 경로를 직접 준다.

## 의존성

`numpy` + `casadi`(nlp) + `torch`(dt) + `scipy` — 전부 `.venv` 에 있다.
pybullet 은 **필요 없다** (시뮬 코드를 안 가져왔다).

# `skills/thr_dt` — vendored Decision-Transformer thrower (Thr_DT)

`robust_throw_skill_dt.py` 가 쓰는 **추론 전용** 코드/가중치. 원본은

    /PublicSSD/ryugaeun/Thr_DT

(논문: M. Monastirsky, O. Azulay, A. Sintov, *"Learning to Throw With a Handful
of Samples Using Decision Transformers"*, IEEE RA-L 8(2):576-583, 2023,
doi 10.1109/LRA.2022.3229266 — 공식 코드 `MaxorPaxor/ThrowBot` 의 재구현).

`skills/throwing.py` · `throw_nlp.py` 가 THR 레포에서 복사돼 온 것과 같은 방식이다.

## 파일 대응

| 여기 | Thr_DT 원본 | 변경 |
|---|---|---|
| `dt_config.py` | `config.py` | 없음 (byte-identical) |
| `trajectory_gpt2.py` | `model/trajectory_gpt2.py` | 없음 |
| `throw_env.py` | `env/throw_env.py` | import 1줄 |
| `decision_transformer.py` | `model/decision_transformer.py` | import 1줄 |
| `weights/dt_best.pth` | `weights/dt_best.pth` | 없음 |
| `weights/dt_best_k0.pth` | `weights/sweep/k0/dt_best.pth` | 없음 |
| `weights/dt_finetuned_real-5.pth` | `weights/dt_finetuned_real-5.pth` | 없음 |

**import 변경 전부** (원본은 `sys.path.insert(0, <repo root>)` 로 bare import 를
쓴다 — 여기서는 `config` / `model` / `env` 같은 일반적인 이름이 프로세스 전역
sys.path[0] 에 올라가는 것을 피하려고 상대 import 로 바꿨다. `gp8_control/model/`
패키지와 이름이 겹친다):

```
throw_env.py               : from config import SimConfig
                           → from .dt_config import SimConfig
decision_transformer.py    : from model.trajectory_gpt2 import GPT2Config, GPT2Model
                           → from .trajectory_gpt2 import GPT2Config, GPT2Model
```

그 외 로직·상수·수식은 한 글자도 건드리지 않았다. 학습/데이터수집/평가
(`train_dt_offline.py`, `collect_data.py`, `evaluate_dt.py`, `finetune_dt.py`,
`agent/`)는 가져오지 않았다 — 로봇에서는 추론만 한다.

## 재동기화

Thr_DT 를 다시 학습했거나 코드를 고쳤으면:

```bash
# Thr_DT 가 있는 워크스테이션에서
scp Thr_DT/config.py                     robotics@<host>:~/ros2_ws/src/gp8_control/skills/thr_dt/dt_config.py
scp Thr_DT/model/trajectory_gpt2.py      robotics@<host>:~/ros2_ws/src/gp8_control/skills/thr_dt/trajectory_gpt2.py
scp Thr_DT/weights/dt_best.pth           robotics@<host>:~/ros2_ws/src/gp8_control/skills/thr_dt/weights/
# throw_env.py / decision_transformer.py 는 위 import 2줄을 다시 바꿔줘야 한다.
```

가중치만 바꿀 때는 `.pth` 만 덮어쓰면 된다 (`--weights` / `GP8_DT_WEIGHTS`
로 경로를 직접 줘도 된다).

## 의존성

`numpy` + `torch` 만 필요하다 (GPT-2 백본이 자체 구현이라 `transformers` 불필요).
`.venv` 에 이미 torch 2.13.0+cu130 이 있다. 추론은 CPU 1스레드로 강제된다
(`torch.set_num_threads(1)`) — ros2_control 4 ms RT 루프를 굶기지 않기 위해서다
(NLP 쪽 `threadpoolctl.threadpool_limits(1)` 과 같은 이유).

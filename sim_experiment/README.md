# sim_experiment — 공유 서버에서 하는 시뮬레이터 실험

`sim_experiment` 브랜치의 실험 안내서. 대상 업무(Simulator 1, 2 / 성호·아청·정호·동환):

1. RL에서 pushing이 선택되지 않는 이유 분석 — 벽(가드) 제약을 없애면 충분한지, 접촉(contact dynamics) 때문에 물체가 통에 안 들어가는지 등
2. (pushing 자체가 현실과 맞는지 확인)
3. (실기에서 바뀐 제약을 시뮬레이터에 반영)

진행 상황과 결정 사항은 [`TODO.md`](TODO.md)에 기록한다. **작업 전에 꼭 읽을 것.**

> **실기에 넣을 정책은 조각 모드 환경으로 학습한다** (2026-11-24): [`docs/RL_PIECE_TRAINING.md`](../docs/RL_PIECE_TRAINING.md).
> 이 문서의 skill 모드 환경에는 실기 컨트롤러 지연이 없어서, 지연을 넣으면 던지기가 무너진다.

---

## 1. 서버와 폴더 구조 (robot17)

코드와 결과는 한 곳에 두고 같이 쓰고, Python 환경만 각자 홈에 설치한다.

| 무엇 | 위치 | 비고 |
|---|---|---|
| 코드 (이 레포) | `/PublicSSD/dhkang/IITP/gp8_control` | `sim_experiment` 브랜치. 모두 같은 작업 트리를 쓴다 |
| 실험 결과 (모델·로그·영상) | `/PublicHDD/dhkang/IITP/runs/` | SSD에는 큰 파일을 두지 않는다 |
| Python 환경 | 각자 `~/envs/gp8_sim` | 각자 설치 (아래 2절) |

- 두 폴더는 `robotics` 그룹(연구실 전원)에 읽기·쓰기가 열려 있다. 새로 만든 파일도 그룹 쓰기 권한이 자동으로 붙는다(기본 ACL).
- GPU: RTX 5090 1장. CPU 32코어, 메모리 60 GB. **올리기 전에 `uptime`, `nvidia-smi`, `who`로 다른 사람 작업을 확인**하고, 남의 프로세스는 건드리지 않는다.

### git 공유 시 주의

- 같은 작업 트리를 여러 명이 쓰므로 **수정 전에 `git status`로 다른 사람이 고치던 파일이 있는지 확인**한다. 큰 수정은 TODO.md에 누가 무엇을 고치는지 적고 시작한다.
- 다른 계정으로 처음 git을 쓰면 "dubious ownership" 오류가 난다. 한 번만 실행:
  ```bash
  git config --global --add safe.directory /PublicSSD/dhkang/IITP/gp8_control
  ```
- 레포에 `core.sharedRepository=group`이 설정돼 있어 커밋 객체도 그룹 쓰기로 생성된다.

---

## 2. 환경 설치 (계정마다 한 번)

버전은 `pyproject.toml` + `uv.lock`에 고정돼 있다(torch 2.13, mujoco 3.10, stable-baselines3 / sb3-contrib 2.9, gymnasium 1.4, numpy 1.26).
레포 README의 설치 명령(`uv sync --extra sim`)에는 **학습용 패키지(`--extra rl`)가 빠져 있으니** 아래 명령을 쓴다.

```bash
# uv 없으면: curl -LsSf https://astral.sh/uv/install.sh | sh
cd /PublicSSD/dhkang/IITP/gp8_control
UV_PROJECT_ENVIRONMENT=$HOME/envs/gp8_sim uv sync --frozen --extra sim --extra rl
```

- **git LFS**: sim의 메시(STL) 파일은 git LFS로 관리된다. robot17에는 git-lfs가 시스템에 없다. 각자 홈에 한 번 설치해 둬야 git 명령(checkout, pull)이 LFS 훅에서 실패하지 않는다. 공유 레포의 메시 파일은 이미 받아 두었다.
  ```bash
  cd /tmp && curl -sSL -o lfs.tgz https://github.com/git-lfs/git-lfs/releases/download/v3.6.1/git-lfs-linux-amd64-v3.6.1.tar.gz \
    && tar xzf lfs.tgz && cp git-lfs-3.6.1/git-lfs ~/.local/bin/ && rm -r lfs.tgz git-lfs-3.6.1 && git lfs version
  ```
- `UV_PROJECT_ENVIRONMENT`를 꼭 붙인다. 안 붙이면 공유 코드 폴더 안에 `.venv`가 생겨 다른 사람과 섞인다.
- ROS는 필요 없다(학습 경로는 rclpy를 import하지 않는다).
- 확인:
  ```bash
  ~/envs/gp8_sim/bin/python -c "import mujoco, torch, sb3_contrib; print(mujoco.__version__, torch.__version__, torch.cuda.is_available())"
  ```

---

## 3. 실행

학습 스크립트는 **레포의 부모 폴더에서** `python -m gp8_control....`로 실행한다(패키지 이름으로 import하기 때문).

```bash
cd /PublicSSD/dhkang/IITP
PY=~/envs/gp8_sim/bin/python
RUNS=/PublicHDD/dhkang/IITP/runs
```

### 사전 점검 (본 학습 전에 반드시)

몇 분짜리 짧은 실행으로 설정 적용·결과 기록·오류 없음을 확인한다.

```bash
OUT=$RUNS/$(date +%m%d_%H%M)_$USER_smoke
GP8_SIM_PROFILE=hw2026 GP8_SIM_PHYSICAL_BELT=true GP8_SIM_BELT_CONTACT_PARAMS=old \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
$PY -m gp8_control.tools.train_ppo_smoke --out-dir $OUT \
  --timesteps 512 --n-envs 4 --n-steps 16 --batch-size 64 --seed 0 \
  --max-steps 1000 --max-episode-seconds 240 \
  --bbox-observation corners --truth-tracks --device cpu --verbose 1
```

- 결과 폴더 이름은 **날짜_시간_계정_설명**으로, 기존 폴더와 겹치지 않게 한다.
- 학습 명령·평가·렌더링 전체 절차는 [`docs/rl_simulator_status_2026-10-10.md`](../docs/rl_simulator_status_2026-10-10.md)를 따른다.
- `summary.json`의 `mean_reward`는 쓰지 말 것(문서에 적힌 대로 신뢰할 수 없음). 비교는 seed 0~9 고정 평가로 한다.
- `GP8_SIM_PROFILE=hw2026`: 2026-10-19 실기 측정값(벨트 속도, 도착 빈도, 물체 크기·위치, 인식 노이즈, 결정 지연)으로 맞춘 설정. README에 "10-19 이전 학습 모델은 다시 학습해야 한다"고 되어 있다.

### push 진단과 영상 (`push_diag.py`, `replay_view.py`)

학습 없이 정해진 규칙(push 우선 / throw 우선 / 현행 규칙)이나 학습된 정책으로 에피소드 하나를 돌린다. push 허용 비율, push된 물체의 물리 이상(관통, 벨트 파고듦, 튕김), 결과, 문제 장면 영상을 남긴다.

```bash
OUT=$RUNS/$(date +%m%d_%H%M)_${USER}_diag
MUJOCO_GL=egl GP8_SIM_PROFILE=hw2026 GP8_SIM_PHYSICAL_BELT=true GP8_SIM_BELT_CONTACT_PARAMS=old \
GP8_PUSH_GUARD=0 OMP_NUM_THREADS=1 \
$PY -m gp8_control.sim_experiment.push_diag --out $OUT --seed 0 --policy push_first
# 학습된 정책: --policy model --model $RUNS/<run>/ppo_smoke.zip
```

결과 폴더:
- `*_summary.json`: 요약
- `*_push_objects.jsonl`: push된 물체별 기록
- `*_steps.jsonl`: 결정별 기록
- `clips/*.mp4`: 문제 장면. 파일 이름에 이유가 붙는다. penetration=관통, sink=벨트에 파고듦, fast=비정상 속도, launch_up=위로 튕김, spin=비정상 회전, missed=바닥, wrong_bin=다른 통, exec_fail=실행 실패, ok=정상 참고용
- `*_state.npz` + `model.mjb`: 전체 상태 기록. 다른 시점·구간은 아래처럼 직접 본다.

```bash
# 서버에서 mp4로 (sim 시각 120~126초)
MUJOCO_GL=egl $PY -m gp8_control.sim_experiment.replay_view $OUT/model.mjb $OUT/push_first_seed0_state.npz --t0 120 --t1 126 --mp4 /tmp/x.mp4
# 화면 있는 PC에서 뷰어로: 폴더를 복사하고 pip install mujoco numpy opencv-python 후
python replay_view.py model.mjb push_first_seed0_state.npz --t0 120 --t1 126 --view --speed 0.25
```

### 병렬 설정 (이 서버 기준)

자세한 근거는 TODO.md의 "속도 측정" 항목.

- 시뮬레이터(MuJoCo)는 CPU에서만 돈다. 학습 속도는 **병렬 환경 수(`--n-envs`)**가 좌우한다. 환경 하나가 CPU 코어 하나를 쓴다.
- 정책 신경망이 작아서(입력 91, 은닉 64×2) GPU를 써도 빨라지지 않는다. `--device cpu`가 기본.
- 여러 명이 동시에 돌리면 코어를 나눠 쓴다. 한 사람이 32코어를 다 쓰지 않는다.

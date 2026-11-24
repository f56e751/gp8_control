# 조각 모드로 RL 학습하기 (시뮬레이터 → 실기)

시뮬레이터에서 학습한 정책을 **실제 로봇에 그대로 넣어 쓰기 위한** 안내서다.
정책은 "다음에 어떤 물체를 어떤 스킬(던지기 / 치기)로 처리할지"만 정하고, 로봇을 실제로 움직이는 것은
**조각 모드**(팀원 grasp-and-throw 플래너, 실기와 같은 코드)가 맡는다.

> 2026-11-24 확인: 같은 물체 흐름에서 실기와 시뮬레이터가 같은 행동을 했다
> (규칙 / 시험 정책 2 개 / 학습된 모델, 각 16/16, 계획 접촉 시각 0.15~0.2 s 이내). 자세한 기록: `cycle_throw/PIECES.md`.

---

## 1. 왜 이 환경인가 (기존 RL 환경과의 차이)

| | 기존 `GP8RecyclingEnv` (`rl/gym_env.py`) | **조각 모드 `GP8PieceEnv` (`rl/piece_env.py`)** |
|---|---|---|
| 로봇 움직임 | 스킬이 만든 궤적을 그대로 보냄 (skill 모드) | 팀원 플래너가 계획 (실기 조각 모드와 같은 코드) |
| 컨트롤러 지연 (40 ms + 200/80 ms 이동평균, 실기 측정으로 확인) | 없음 | 있음 |
| 실기 대응 | 지연을 켜면 던지기 성공률 0.83 → 0.14~0.24 | 실기와 같은 행동 (위 확인) |
| 행동·관찰 형식 | `[칸, 스킬]`, 관찰 55 차원 | **같음** |

실제 로봇에도 같은 지연이 있으므로, 실기에 넣을 정책은 조각 모드 환경에서 학습한다.

## 2. 준비 (robot17, 계정마다 한 번)

환경 설치·git LFS 는 `sim_experiment/README.md` 2 절과 같다 (`uv sync --frozen --extra sim --extra rl`, numpy 1.26).
조각 모드는 추가로 **플래너 모델 파일**(git LFS, 약 100 MB)이 필요하다.

```bash
cd /PublicSSD/dhkang/IITP/gp8_control
git lfs pull --include "third_party/grasp_and_throw/planner/logs/**"
PY=~/envs/gp8_sim/bin/python
cd .. && $PY -c "import mujoco, casadi, sb3_contrib; print('ok')"
```

- ROS 는 필요 없다. 플래너 링크는 환경마다 로컬 자식 프로세스로 뜬다 (같은 파이썬, `GP8_CYCLE_PYTHON` 으로 바꿀 수 있음).
- 벨트 속도는 **플래너 모델 세트의 속도로 고정**된다 (기본 `GT_BELT=0.12`, git 에 있는 세트는 이것 하나).
  `hw2026` 프로필의 벨트 속도 무작위(0.10~0.30)는 조각 모드에서 무시된다. 다른 속도를 쓰려면 그 속도의 모델 세트가
  필요하다 (robot6 체크아웃에만 있음, `CLAUDE.md` cycle mode 절).

## 3. 먼저 돌려 보기 (몇 분)

레포의 **부모 폴더에서** 실행한다.

```bash
cd /PublicSSD/dhkang/IITP
# 규칙 정책으로 환경 한 판 (결정 수, 보상, 물체 결과가 나온다)
GP8_SIM_PROFILE=hw2026 GP8_SIM_PHYSICAL_BELT=true GP8_SIM_BELT_CONTACT_PARAMS=old \
$PY -m gp8_control.rl.piece_env --seconds 120 --policy heuristic
```

`EPISODE {...}` 한 줄이 나오고 `error: null` 이면 된다. 60 s 짜리가 실제 약 20~30 s 걸린다 (아래 6 절).

## 4. 학습

기존 학습 스크립트에 `--env piece` 만 붙인다 (MaskablePPO, 행동 마스크 포함).

```bash
OUT=/PublicHDD/dhkang/IITP/runs/$(date +%m%d_%H%M)_${USER}_piece
GP8_SIM_PROFILE=hw2026 GP8_SIM_PHYSICAL_BELT=true GP8_SIM_BELT_CONTACT_PARAMS=old \
GP8_PIECE_MOVE_WORKERS=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
$PY -m gp8_control.tools.train_ppo_smoke --env piece --out-dir $OUT \
    --timesteps 300000 --n-envs 8 --n-steps 256 --batch-size 256 \
    --max-episode-seconds 240 --eval-episodes 0
```

- **CPU**: 환경 하나마다 플래너 링크 1 개 + 이동 계산 프로세스 `GP8_PIECE_MOVE_WORKERS` 개(기본 6)가 뜬다.
  32 코어에서 `--n-envs 8` 이면 `GP8_PIECE_MOVE_WORKERS=2` 정도로 둔다. 다른 사람 작업을 먼저 확인할 것.
- **속도**: 플래너가 계산하는 동안은 실제 시간과 같이 흐르고 (계산이 늦어 놓치는 것까지 실기와 같게), 계산이 없으면
  시간을 건너뛴다. 물체가 촘촘하면 실시간의 약 2 배, 드문드문하면 약 5 배.
- 리셋마다 트윈과 링크가 새로 뜬다 (몇 초). 에피소드는 길게 (240 s) 잡는 편이 낫다.
- `--eval-only`, `--trace-jsonl` 은 기존 환경 전용이다. 평가는 5 절로 한다.

## 5. 환경 정의 (학습할 때 알아야 할 것)

- **결정 시점**: 조각 모드가 다음 단계를 정할 때 묻는다. 들고 있는 물체를 던지기 약 3 s 전 (`RT_AHEAD`),
  또는 팔이 비었을 때. 정책이 push 를 고르면 같은 계획 안에서 한 번 더 묻는다 (push 최대 2 개 + 집기 1 개를 한 번에 계획,
  지금 규칙과 같음).
- **관찰** (`rl/piece_policy.py` `PieceDecider.observation`, 실기와 같은 코드): 조각 모드가 **아직 맡지 않은** 추적 물체를
  하류부터 최대 6 칸 + 관절 + 끝점 위치 + 벨트 속도 + 남은 시간 비율. 형식은 기존 환경과 같다 (`rl/common.build_observation`).
- **행동**: `[칸, 스킬]` (칸 6 = 건너뛰기, 스킬 0 = 던지기, 1 = 치기). 학습 스크립트는 14 개짜리 하나로 펴서 쓴다
  (`칸 × 2 + 스킬`).
- **가능한 행동 (마스크)**: 그 순간 조각 모드가 제안하는 후보 물체 중, 던지기는 플래너 통이 있는 클래스,
  치기는 `PushSkill` 이 받는 클래스. 건너뛰기는 항상 가능.
- **고른 행동이 그대로 안 되는 경우** (정해진 대체, 결정 감사에 남는다): push 를 스킬이 거절하면 던지기로,
  지금 push 가 불가능하면 다시 묻기, 한 계획에 push 가 이미 2 개면 다음 계획으로.
- **보상**: 트윈의 물체 결과. 던진 물체는 클래스 통, 친 물체는 `push_<클래스>` 통에 들어가면 +1, 다른 곳 -0.3,
  아무도 안 건드리고 지나가면 0. (보고서의 "놓친 물체 감점 -0.3" 옵션 `GP8_RL_MISS_PENALTY` 는 이 코드에 아직 없다.
  넣는다면 `backends/mujoco_sim.py` `resolved_object_reward` 한 곳에.)

## 6. 평가: 시뮬레이터에서 모델 돌려 보기

```bash
# 학습된 모델이 결정 (결정 감사: 고른 대로 계획됐는지, 링크가 받았는지)
$PY -m gp8_control.tools.piece_twin --seconds 240 --policy $OUT/ppo_smoke.zip
# 비교 기준: 지금 규칙
$PY -m gp8_control.tools.piece_twin --seconds 240 --policy heuristic
```

`RESULT` 의 `outcomes` (물체별 결과) 와 `decision_audit` 를 비교한다. 같은 물체 흐름으로 비교하려면 둘 다
`--spawn-script objects.json` 을 준다 (7 절).

## 7. 실기에 넣기 (로봇 PC)

1. 모델 파일을 로봇 PC 로: `scp $OUT/ppo_smoke.zip robotics@<로봇PC>:gp8_models/` (로봇 PC `.venv` 에 sb3-contrib 2.9 있음;
   numpy 버전이 달라도 읽힌다).
2. 먼저 **가상 물체로** (실제 물체 없이, 로봇은 움직인다):
   ```bash
   GP8_DEMO_MODE=pieces GP8_DEMO_NO_CAMERA=1 GP8_PIECE_POLICY=~/gp8_models/ppo_smoke.zip gp8_cycle
   # 다른 터미널, 로그에 "arm rest" 가 보이면:
   source scripts/env.sh && python3 -m gp8_control.tools.fake_detections transparent,metal 16 1.4
   ```
3. 실제 물체로: `GP8_DEMO_MODE=pieces GP8_PIECE_POLICY=~/gp8_models/ppo_smoke.zip gp8_cycle`
4. 로그 확인: `[policy] decision n: ...` (고른 것), `[policy] result: ... -> planned ..., link took it` (실행된 것).
   `GP8_PIECE_POLICY=heuristic` 은 지금 규칙을 같은 통로로 돌린다 (비교 기준).

## 8. 실기와 시뮬레이터가 같은지 확인 (real-to-sim)

`gp8_cycle` 은 실행마다 `~/gp8_logs/cycle_demo/<시각>/record/` 에 기록한다. 그 물체 흐름을 트윈에 같은 정책으로 넣어 비교한다.

```bash
python -m gp8_control.tools.real2sim spawn RECORD -o objects.json          # 실기 물체 흐름 -> 트윈 스크립트
python -m gp8_control.tools.piece_twin --seconds <마지막 물체 + 20> --spawn-script objects.json \
       --policy <같은 정책> --record TWIN
python -m gp8_control.tools.real2sim compare RECORD TWIN                   # 물체별 행동·접촉 시각
python -m gp8_control.tools.real2sim track RECORD                          # 실제 팔 vs 컨트롤러 필터 모델
```

- `--seconds` 는 **실기 기록의 마지막 물체 시각 + 약 20 s** 로 잡는다. `compare` 표의 `t(y=0)` (플래너 시간) 가 가장 큰 값을
  보면 된다. 짧으면 마지막 물체들이 트윈 실행 종료와 겹쳐 다르게 나온다 (예: 마지막 물체 79 s 인 실행을 80 s 로 재생하면
  16 개 중 14 개만 같고, 100 s 로 하면 16/16).
- 정책은 실기와 같은 것을 준다 (`heuristic`, 같은 `.zip` 등). 정책 없이 재생하면 지금 규칙으로 결정한다.

## 9. 바꿀 때: 한 규칙은 한 곳

| 바꾸려는 것 | 고칠 곳 (여기만) |
|---|---|
| 관찰, 마스크, 정책 불러오기 | `rl/piece_policy.py` (`PieceDecider`, `load_policy`) |
| 조각 모드 조립 (스킬, 스킬별 클래스, 결정기 연결) | `cycle_throw/piece_stack.py` (`PieceStack`; 실기 `AppPieceStack`, 트윈 `TwinPieceStack`) |
| 결정 시점, push 묶음, 대체 동작 | `cycle_throw/piece_runner.py` (`_decide`, `_steps_decided`) |
| push 도달 범위·가속 (조각 모드) | `config/piece_mode.env` (실기 `gp8_cycle` 과 트윈이 같이 읽음) |
| 석션 밸브 지연 | `cycle_throw/runner.py` `VALVE_S` |
| 컨트롤러 버퍼 규칙 (파이썬) | `controllers/sample_buffer_rules.py` |
| 보상 | `backends/mujoco_sim.py` (`resolved_object_reward`), 정답 통은 `TwinPieceStack` |

실기 현장 보정값 (물체 윗면 높이, 실제 벨트 위 push 높이, 카메라 부호) 은 로봇 PC `.env` 에만 있다. 트윈은 자기 기하로 정한다.

## 10. 알려진 한계 (학습 전에 볼 것)

- **캔 push 가 통에 잘 안 들어갈 수 있다**: metal 통이 상류 쪽이라 흐름을 거슬러 밀어야 하는 경우 (보고서 실험 7).
  물체 위치 분포에 따라 다르다 (`hw2026` 에서는 잘 들어감). 정책이 push 를 피하도록 배우면 이것부터 의심한다.
- **벨트 높이**: 로봇 좌표에서 실제 벨트가 트윈보다 약 5 cm 낮다. 트윈은 자기 벨트에 맞춘 높이를 쓴다.
- **트윈 카메라 시야 제한을 끄지 말 것** (`GP8_SIM_CAM_FOV_GATE=0`): 빗나간 push 뒤 남은 캔이 로봇 앞에서 다시 잡혀,
  실기에는 없는 결정이 생긴다.
- **실기 결과 데이터는 아직 적다**: 실제 물체로 통에 들어간 비율은 기록되지 않는다 (영상 대조 필요).

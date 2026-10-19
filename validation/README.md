# Sim-to-real 검증

RL 시뮬레이터 env가 실기와 얼마나 같은지 잰다. 실기 루프를 돌리면 자동으로
기록되고, 종료(Ctrl-C) 시 분석이 백그라운드로 돌아 `report.md`가 생긴다.

## 실행

저장소 README의 **실행 B** 그대로 (터미널 3개):

```bash
gp8_robot     # 1: 드라이버 + 벨트 엔코더
gp8_camera    # 2: 카메라 검출
gp8_s2r       # 3: RL runner + 기록 (바로 초기 자세로 이동, 끝나면 Ctrl-C)
```

- 터미널 3 로그에 `[s2r] sim-to-real validation recording -> ~/gp8_s2r_runs/...` 가 보여야 한다.
- throw / push 가 각각 10~20회 이상 실행되게 물체를 흘린다 (표본이 적으면 분포 비교가 N/A).
- 기존 app 으로 기록: `gp8_app s2r_validate:=true`.
- `gp8_s2r` 옵션: `--no-validate`(기록 끄기), `--validate-dir DIR`, `--confirm-initial`(Enter 확인).

## 결과

```bash
RUN=$(ls -td ~/gp8_s2r_runs/*/ | head -1)
tail -3 $RUN/analyze.log        # "[s2r] report -> ..." 가 나오면 완료
cat $RUN/report.md
```

- 분석만 다시: `gp8_env && .venv/bin/python -m gp8_control.validation.analyze $RUN --sim-steps 40`
- 로봇 없이 파이프라인 확인 (1분, sim 끼리 비교라 action 항목은 PASS 가 정상):
  ```bash
  gp8_env; PY=$GP8_WS/src/gp8_control/.venv/bin/python
  $PY -m gp8_control.validation.sim_reference /tmp/s2r_selftest --steps 10 --seed 1 --source sim_as_hw
  $PY -m gp8_control.validation.analyze /tmp/s2r_selftest --sim-steps 10
  ```

## 무엇을 검증하나

- **A. Action → 움직임**: 250 Hz 명령 대비 실제 joint 추종 지연·오차, HW 명령 스트림을
  MuJoCo 팔에 그대로 재생해 HW vs sim 직접 비교(EE 궤적, 릴리즈 순간 위치·속도),
  결정→첫 모션 지연, 스트림 jitter, 석션 IO 지연, 스킬 소요시간.
- **B. State**: joint state 나이, 벨트 속도/엔코더 일치, 추측항법 오차, obs 내 bbox
  위치 불일치, 클래스/중복 트랙, 학습 스키마(91차원) feature별 분포를
  같은 벨트 속도의 sim reference rollout과 비교.

## 파일

| 파일 | 내용 |
|---|---|
| `recorder.py` | 명령 스트림·joint feedback·석션·벨트·트랙·스텝별 obs 기록 (HW/sim 공용) |
| `analyze.py` | 리포트 생성. 임계값은 `THRESHOLDS` |
| `replay.py` | HW 명령 스트림을 MuJoCo 팔에 재생 |
| `sim_reference.py` | 같은 벨트 속도로 학습 설정 sim rollout 기록 |
| `obs_schema.py` | 학습 관측 스키마(기본 91차원) 정의·디코딩 |

# warm_db.pkl BUILD REPORT (2026-07-20)

`skills/warm_db.pkl` — RobustThrowSkill의 offline warm-start DB. 순수 오프라인
수치계산으로 생성 (ROS 노드/로봇 IO 없음). 빌드 스크립트: `tools/build_warm_db.py`,
원시 로그: `tools/warm_db_build_log.json`.

## 1. 사용 좌표와 출처 (임의 추정 없음 — 전부 리포 내 설정/캘리브레이션 값)

**주의: 좌표는 전부 플래너 질의(p_target/p_start, Cartesian)이며 base frame.
DB entry 내부의 관절값(P, q_start)은 THR 플래너 관절 규약이다 — 소비자
(`robust_throw_skill.py`)가 경계에서 `_PLANNER_SIGN`으로 변환한다.**

### Targets

| target [m] | 출처 |
|---|---|
| (1.10, −0.25, 0.162) | XY: `skills/throw_skill.py:32-33` / `robust_throw_skill.py` `THROW_BIN_X/Y` — 라이브 스킬이 던지기 방위각에 실제 사용하는 설정값. 단 주석이 "Tune to the measured bin centre"라 실측 확인 필요(→ §5). z = GRASP_Z(0.062, 실측 `config.py:79-81`) + 0.10 (bin 조준 컨벤션 `tests/suction_lift_debug.py:122` `BIN_Z_OFFSET_DEFAULT`, `docs/2026-07_suction-lift-throw-debug-guide.md`) |
| (0.45, −0.40, 0.132) | 벨트 fallback 조준점(THROW_BIN_TARGET_MAP 비어 있을 때 runtime은 T_aim 기반 좌표를 질의). x=벨트 중앙선 `REFERENCE_X_BASE=0.45` (`perception/extrinsics.py:53`), z = GRASP_Z + `DETECTION_OFFSET_AIM=0.07` (`perception/extrinsics.py:32`) |
| (0.55, −0.30, 0.132) | 위와 동일 컨벤션, 벨트 밴드(0.45±0.20, `WORKSPACE_X_ABS`) 내 가장자리 대표점 |

### p_start (던지기 시작 TCP = grasp + THROW_LIFT 0.10 상승, z = 0.062+0.10 = 0.162)

| p_start [m] | 의미 |
|---|---|
| (0.45, +0.30, 0.162) | 중앙선, entry-edge 대기 (reach 원판 y_b(0.45)=0.469 내) |
| (0.45, 0.00, 0.162) | 중앙선 정중앙 |
| (0.45, −0.30, 0.162) | 중앙선 downstream intercept |
| (0.30, 0.00, 0.162) | 벨트 밴드 근측 |
| (0.60, 0.00, 0.162) | 벨트 밴드 원측 (y_b(0.60)=0.25 내) |

intercept 존 근거: 벨트 밴드 x∈[0.25, 0.65] (`extrinsics.py` 캘리브레이션),
동적 intercept y∈[−y_b, +y_b] (`skills/context.py:218-273`, `MAX_REACH=0.65`).

## 2. 빌드 결과

- 그리드: 3 targets × 5 starts × 4 init variants(`INIT_VARIANTS_ROS` 미러) = **60 cold solves**, 6 workers (spawn + BLAS 1-thread), 총 345 s
- 시도 성공 20 / 기각 40 → (target, start) 조합별 최소-J 해만 채택 = **9 entries** (15조합 중 9)
- cold 성공 solve 시간: min 5.1 / med 7.9 / max 88.5 s. 채택 해 t_f ≈ 0.30 s, J: bin ≈ 0.84, 벨트 fallback ≈ 0.39

### 조합별

| target \ start | s0 (+0.30) | s1 (0.0) | s2 (−0.30) | s3 (근측) | s4 (원측) |
|---|---|---|---|---|---|
| t0 bin | ✗ 착탄게이트(inf) | ✓ J=0.841 | ✓ J=0.836 | ✓ J=0.854 | ✗ 착탄게이트(inf) |
| t1 벨트 중앙 | ✗ IPOPT 실패 | ✓ J=0.393 | ✓ J=0.391 | ✓ J=0.383 | ✗ IPOPT 실패 |
| t2 벨트 가장자리 | ✗ IPOPT 실패 | ✓ J=0.400 | ✓ J=0.397 | ✗ IPOPT 실패 | ✓ J=0.398 |

기각 사유는 두 종류뿐: ① 윈도우 dense 착탄오차 inf(나쁜 basin — 게이트가 의도대로
차단), ② IPOPT infeasible(4개 init variant 전부). 모두 entry로 넣지 않았다.
runtime엔 XY 최근접 2개 entry로 polish하므로 실패 조합 근방 질의도 인접 entry가
커버하고, polish 실패 시 cold multistart로 자동 fallback한다.

## 3. 파일 포맷 / 라운드트립 검증

- `pickle: dict(params=..., entries=[9])`, params는 `_formulation_params()`와 동일 키
  (rt=0.05, w_acc=1e4, n_ctrl=12, n_win=7, q_lo/q_hi, qd_max, col=(0.18, 0.55),
  **pos_mode="hull"**) — 스킬이 import하는 바로 그 `skills/throw_nlp.py` 복사본에서 추출
- `_load_warm_db` 로직 복제로 재로드: params subset 일치, 9 entries ✓
- entry 키: target, p_start, P, t_f, t_star, lam_g, u_pos, J (+q_start, variant 부가)

## 4. polish 실증 (오프라인)

- 빌드 직후 6질의(entry target ±2–5 cm): **6/6 게이트 통과**, 4.4–5.6 s (동시 6개 실행 = CPU 경합 포함)
- 무경합 단독 1질의: **warm 2.6 s vs cold 4.3 s (×1.7)** — cold median 7.9 s / max 88.5 s 대비 ×3–34
- 품질 효과: 같은 질의에서 cold는 J=6.05 나쁜 국소해로 수렴, warm은 J=0.883 유지
  — warm DB는 속도뿐 아니라 **basin(해 품질) 고정** 효과가 실측됨
- 문서상 기대치 "~1–2 s"보다는 느림(이 PC 기준). 필요하면 `expand`/JIT 튜닝 여지

## 5. 질문 목록 (보류/확인 필요)

1. **bin 실측 좌표**: THROW_BIN_X/Y=(1.10, −0.25)는 코드 주석상 "실측 중심으로 튜닝
   필요"인 설정값이다. 실측 bin 중심 XY와 **개구부 높이 z**(현재 0.162는 debug 툴
   컨벤션 grasp+0.10)를 재면 그 좌표로 DB 재빌드 권장 (`tools/build_warm_db.py`의
   TARGETS만 바꿔 재실행).
2. 벨트 fallback 조준 z(0.132 = GRASP_Z+0.07)는 카메라 aim 오프셋 컨벤션에서 유도 —
   물체 높이별 실제 T_aim z 분포 확인 시 대표 z 보강 가능.
3. entry-edge(+y) 시작점에서의 던지기는 현 공식화로 전부 실패(§2) — 실전에서 그
   자세로 던질 일이 있는지(있다면 init variant 보강 필요) 확인.

## 6. 재빌드 방법

```bash
cd ~/ros2_ws/src/gp8_control
.venv/bin/python tools/build_warm_db.py            # 빌드+검증 일체
.venv/bin/python tools/build_warm_db.py --dry-run  # 그리드 확인만
```

공식화 파라미터(throw_nlp.py 상수, 특히 POS_LIMIT_MODE)가 바뀌면 DB는 로더에서
자동 무시되므로 반드시 재빌드할 것.

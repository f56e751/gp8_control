"""THR 던지기 계획 모델 3종의 **추론/계획 코드**를 로봇으로 벤더링한 패키지.

원본:
  /PublicSSD/ryugaeun/THR       — throwing.py, throw_nlp.py, dt_gp8_env.py, dt_cem.py,
                                   weights_gp8*/ (GP8 rig 학습 DT 체크포인트)
  /PublicSSD/ryugaeun/Thr_Phy   — tossingbot/ (TossingBot Physics-only 컨트롤러)
  /PublicSSD/ryugaeun/Thr_DT    — DT 모델 정의 (GPT-2 백본 + DecisionTransformer)

import 줄만 상대 import 로 바꿨고 나머지는 byte-identical 이다 (README.md 참고).
`skills/throwing.py` · `skills/throw_nlp.py` (기존 NLP 스킬이 쓰는 구버전)와
**이름이 겹치므로** 절대 sys.path 로 노출하지 않는다 — 전부 패키지 상대 import.
"""

__all__ = ["throwing", "throw_nlp", "dt_gp8_env", "dt_cem", "dt_model", "tossingbot"]

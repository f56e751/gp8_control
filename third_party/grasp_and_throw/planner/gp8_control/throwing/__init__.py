"""[throwing] 던지기 계획·실행 — NLP 플래너 (Ryu et al.) 만 남김 (2026-09-18, NN 던지기 삭제).

throw_planner.py         공개 API (사용법: how_to_throw.md): ThrowBoundary(관절 자세·속도) → plan_throw → ThrowPlan(궤적 + suction 타이밍)
                         object_boundary / plan_throw_between_objects: 물체 위치·속도 → 경계 상태 → 체인 계획
nlp/                     NLP 본체 (throw_nlp.py CasADi/IPOPT B-spline, throwing.py FK, planner.py 어댑터·warm DB)
nlp_throw_skill.py       로봇 스킬: 픽(track-descend) → plan_throw (시작 = 잡은 자세+lift, 종점 = 다음 물체) → timed release
trajectory_primitive.py  구간 보간 프리미티브 (lift / 감속·체인 세그먼트, grasping 도 사용)
"""
from gp8_control.throwing.throw_planner import (BELT_DIR, DESCEND_SPEED, LIFT_SPEED, START_LIFT, ThrowBoundary, ThrowPlan,
                                                object_boundary, plan_throw, plan_throw_between_objects)

__all__ = ["BELT_DIR", "DESCEND_SPEED", "LIFT_SPEED", "START_LIFT", "ThrowBoundary", "ThrowPlan", "object_boundary", "plan_throw",
           "plan_throw_between_objects"]

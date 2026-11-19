"""[grasping 플래너] 어떤 물체를 어떤 순서로·어디서·언제 잡을지 + 잡는 동작.

sequence_planner.py  처리 순서 플래너 (Part 2 이동 물체 보정 · Part 3 순서 선택) — 프리미티브·로봇 무관
throw_sequence.py    ThrowPrimitive(우리 던지기 모듈을 Part 1 로) + plan_full_sequence: 물체들 → 순서 + 전체 궤적 + suction
                     → how_to_sequence.md
context.py        earliest_reachable_intercept (동적 인터셉트 고정점 해), position_and_prime,
                  wait/sleep/chain 등 모든 스킬이 쓰는 모션 프리미티브 (ROS)
track_descend.py  벨트 속도로 물체를 따라가며 하강하는 픽 세그먼트 (TRACK_DESCEND)
base.py           ManipulationSkill 계약 (execute / placement_veto / t_to_contact ...)
action_selector.py 클래스 → 스킬 라우팅
"""
from gp8_control.grasping.base import ManipulationSkill, SkillResult
from gp8_control.grasping.action_selector import ActionSelector
from gp8_control.grasping.track_descend import (
    PickWaitMode, PICK_WAIT_MODE, DEFAULT_PICK_WAIT_MODE, TrackDescendPickMixin,
)
from gp8_control.grasping.sequence_planner import (CircularWorkspace, MovingPoint, ObjectState, PlanResult,
                                                   PrimitiveResult, SourceState, plan_sequence)
try:                                   # rclpy 가 없는 환경(순서 플래너 단독 실행)에서도 패키지는 import 되게
    from gp8_control.grasping.context import SkillContext, PickRequest, Intercept
except ImportError:                    # noqa: F401
    SkillContext = PickRequest = Intercept = None   # type: ignore[assignment]

__all__ = ["ManipulationSkill", "SkillResult", "SkillContext", "PickRequest", "Intercept",
           "ActionSelector", "PickWaitMode", "PICK_WAIT_MODE", "DEFAULT_PICK_WAIT_MODE",
           "TrackDescendPickMixin", "CircularWorkspace", "MovingPoint", "ObjectState", "PlanResult",
           "PrimitiveResult", "SourceState", "plan_sequence"]

"""Throw with the teammate's cycle planner, as a piece (piece mode only).

The motion itself (suction pick from the belt, the optimized throw cycle, the
filter-compensated commands) is computed by the vendored planner in the piece
link process. This skill only says WHAT to throw and into which bin. The piece
runner turns that into the planner's ``pick`` / ``throw`` requests, because
how a throw starts and ends depends on the arm's state and on the next piece.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gp8_control.skills.base import ManipulationSkill, MotionPiece, SkillResult

if TYPE_CHECKING:
    from gp8_control.skills.context import PickRequest
    from gp8_control.tracking import TrackedObject

#: class -> planner bin index (cyc2.BINS order: y = +0.33, 0, -0.33)
PLANNER_BIN_INDEX = {"metal": 0, "paper": 1, "transparent": 2}


class PlannerThrowSkill(ManipulationSkill):
    name = "throw"

    def can_handle(self, target: "TrackedObject") -> bool:
        return target.class_name in PLANNER_BIN_INDEX

    def plan_piece(self, request: "PickRequest", t_start: float):
        target = request.target
        b = PLANNER_BIN_INDEX.get(target.class_name)
        if b is None:
            return SkillResult(False, f"no planner bin for {target.class_name}")
        return MotionPiece(kind="planner", request={"bin": b},
                           detail=f"throw {target.class_name} id={target.track_id} -> bin {b}")

    def execute(self, request: "PickRequest") -> SkillResult:
        return SkillResult(False, "the planner throw runs in piece mode only (GP8_MODE=pieces)")

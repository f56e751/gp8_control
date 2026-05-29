"""Push skill (stub).

Placeholder for the conveyor-push manipulation. It implements the
:class:`~gp8_control.skills.base.ManipulationSkill` interface so the rest of
the system (ActionSelector, app loop) is already wired for it — a teammate
fills in :meth:`execute` here without touching ``app.py``.

Implementation notes for whoever picks this up:
  * Shared primitives are on ``self.ctx`` (a ``SkillContext``):
      - ``ctx.move_through(current, aim, grasp)`` — queue-mode move to a pose
      - ``ctx.wait_for_arrival_and_suction(target, intercept_y)`` — park + wait
        (drop/replace the suction part if push doesn't grasp)
      - ``ctx.sleep_until(deadline)`` — block while pumping ROS/intake/viz
      - ``ctx.scan_next_intercept()`` — next reachable object's joint config
      - ``ctx.robot`` (IK/FK), ``ctx.traj_ctrl``, ``ctx.planner``, ``ctx.M1/M2``,
        ``ctx.cfg``, ``ctx.queue``, ``ctx.conveyor``
      - ``ctx.set_status(status, detail)``, ``ctx.set_active_target(None)``
  * Re-enter queue mode (``ctx.traj_ctrl.enter_queue_mode()``) before each
    dispatch, like ThrowSkill does — MotoROS2 drops queue mode when a queue
    drains.
  * Return a ``SkillResult(success, detail)``; clear the active target and set
    status IDLE on the way out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gp8_control.skills.base import ManipulationSkill, SkillResult

if TYPE_CHECKING:
    from gp8_control.skills.context import PickRequest
    from gp8_control.tracking import TrackedObject


class PushSkill(ManipulationSkill):
    """Sweep/push an object off the belt instead of throwing it. NOT YET BUILT."""

    name = "push"

    def can_handle(self, target: "TrackedObject") -> bool:
        # Not implemented yet — refuse so the ActionSelector falls back to the
        # default (throw) skill until execute() is filled in.
        return False

    def execute(self, request: "PickRequest") -> SkillResult:
        ctx = self.ctx
        ctx.log.warn(
            "PushSkill.execute is not implemented yet — object not handled. "
            "Fill in gp8_control/skills/push_skill.py."
        )
        # Leave the cell in a clean idle state so the main loop keeps running.
        ctx.traj_ctrl.suction_off()
        ctx.set_active_target(None)
        ctx.set_status("IDLE", "")
        return SkillResult(False, "push not implemented")

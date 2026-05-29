"""Push skill (debug stub).

Not a real conveyor push yet. ``execute`` performs a throwaway motion — lift
the end-effector 5 cm straight up, then lower it 5 cm back to the start — so
the push *dispatch path* can be exercised on real hardware before the actual
sweep is written. Force it with ``GP8_FORCE_SKILL=push`` (or ``--skill push``);
under normal routing ``can_handle`` refuses so every object still goes to throw.

Shared primitives live on ``self.ctx`` (a ``SkillContext``):
  * ``ctx.robot`` (FK/IK), ``ctx.traj_ctrl``, ``ctx.M1``/``ctx.M2``, ``ctx.cfg``
  * ``ctx.move_through`` / ``ctx.wait_for_arrival_and_suction`` / ``ctx.sleep_until``
  * ``ctx.set_status(status, detail)``, ``ctx.set_active_target(None)``
When the real push lands here, replace ``execute`` with the sweep and flip
``can_handle`` to accept the classes push should own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gp8_control.skills.base import ManipulationSkill, SkillResult
from gp8_control.trajectory.trajectory_primitive import trajectory

if TYPE_CHECKING:
    from gp8_control.skills.context import PickRequest
    from gp8_control.tracking import TrackedObject

#: Debug motion amplitude — EE goes up this far (m) then back down.
_DEBUG_LIFT_M = 0.05


class PushSkill(ManipulationSkill):
    """Debug push: lift the EE 5 cm and lower it back. NOT a real sweep yet."""

    name = "push"

    def can_handle(self, target: "TrackedObject") -> bool:
        # Real push routing isn't built — refuse so the ActionSelector falls
        # back to the default (throw) skill under normal routing. The debug
        # up/down motion only runs when push is FORCED (force= bypasses this).
        return False

    def execute(self, request: "PickRequest") -> SkillResult:
        """DEBUG: lift the EE +5 cm (base Z) then lower it 5 cm back — one cycle."""
        ctx = self.ctx
        ctx.set_status("PUSHING", request.target.class_name)
        ctx.traj_ctrl.suction_off()

        current_joint = np.asarray(request.current_joint, dtype=float)

        # EE pose now -> raised 5 cm in base Z. The lower target is exactly the
        # start config, so the arm returns precisely to where it began.
        T_up = ctx.robot.forward_kinematics(current_joint).copy()
        T_up[2, 3] += _DEBUG_LIFT_M

        up_ik = ctx.robot.inverse_kinematics(T_up)
        if up_ik is None:
            ctx.log.warn(
                f"push debug: IK unreachable for +{_DEBUG_LIFT_M * 100:.0f}cm "
                "pose; skipping cycle"
            )
            ctx.set_active_target(None)
            ctx.set_status("IDLE", "")
            return SkillResult(False, "push debug IK failed")
        up_joint = np.asarray(up_ik, dtype=float)

        zero = np.zeros_like(ctx.M1)
        # Up then down. Re-enter queue mode before EACH segment: MotoROS2 drops
        # queue mode once a trajectory's queue drains (send_trajectory_queue
        # blocks until then), so the next push would be rejected otherwise —
        # the same per-segment re-entry ThrowSkill does.
        for label, start, goal in (
            ("up", current_joint, up_joint),
            ("down", up_joint, current_joint),
        ):
            if not ctx.traj_ctrl.enter_queue_mode():
                ctx.log.error(
                    f"push debug: enter_queue_mode ({label}) failed; aborting"
                )
                ctx.set_active_target(None)
                ctx.set_status("IDLE", "")
                return SkillResult(False, f"enter_queue_mode ({label}) failed")
            traj, vel, ts = trajectory(
                start, zero, goal, zero, ctx.M1, ctx.M2, hertz=ctx.cfg.TRAJ_HZ,
            )
            ctx.traj_ctrl.send_trajectory_queue(traj, vel, ts, final_joint=goal)

        ctx.set_active_target(None)
        ctx.set_status("IDLE", "")
        return SkillResult(True, "push debug up/down cycle complete")

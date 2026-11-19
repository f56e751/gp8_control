"""One push inside a cycle-mode push slot, with the app's own PushSkill.

Shared by the app (GP8App, cycle mode) and the rclpy-free sim runner, so the
slot runs the exact same push code as the per-object skill loop. The only
difference from a normal push is that the chain move toward the next object is
switched off: the slot hands the arm back to the cycle planner at its wait
point, not to the next object.
"""

from __future__ import annotations

import numpy as np

from gp8_control.skills import PickRequest


def make_push_handler(ctx, push_skill, queue, *, on_execute=None):
    """``handler(obj) -> str`` for ``CycleThrowRunner(push_handler=...)``.

    ``ctx`` is the app's SkillContext. ``on_execute(obj)`` runs right before the
    push is dispatched (the sim marks the object as manipulated there)."""

    def handle(obj) -> str:
        current = ctx.traj_ctrl.current_joints
        if current is None:
            return "skipped: no joint state"
        current = np.asarray(current, float)
        now = ctx.clock.time()
        it = ctx.earliest_reachable_intercept(obj, current, ctx.conveyor.current, now,
                                              skill=push_skill)
        if it is None:
            queue.remove(obj)
            return "dropped: uncatchable"
        veto = push_skill.placement_veto(obj, it)
        if veto is not None:
            return f"skipped: {veto}"
        queue.remove(obj)
        if on_execute is not None:
            on_execute(obj)
        saved = ctx.chain_target_override
        ctx.chain_target_override = lambda from_joint, action_time: None   # park, no chain
        try:
            ctx.set_active_target(obj)
            result = push_skill.execute(PickRequest(
                target=obj, current_joint=current, T_aim=it.T_aim, T_grasp=it.T_grasp,
                aim_joint=it.aim_joint, grasp_joint=it.grasp_joint, secondary=None))
        finally:
            ctx.chain_target_override = saved
            ctx.set_active_target(None)
        return f"{'ok' if result.success else 'failed'}: {result.detail}"

    return handle

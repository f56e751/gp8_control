"""Base class for manipulation skills."""

from __future__ import annotations

import csv
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from gp8_control.skills.context import SkillContext, PickRequest
    from gp8_control.tracking import TrackedObject


@dataclass
class SkillResult:
    """Outcome of a skill execution.

    ``success`` is the minimal signal the orchestrator needs; ``detail`` is a
    short human/debug string. Kept as a return value (rather than None) so a
    future RL ``ActionSelector`` can use it as a reward / bookkeeping signal.
    """

    success: bool
    detail: str = ""


class ManipulationSkill(ABC):
    """An action that handles one object (throw, push, ...).

    Subclasses implement :meth:`execute`. All shared robot resources and
    motion/timing helpers are reached through ``self.ctx`` (a
    :class:`~gp8_control.skills.context.SkillContext`), injected at
    construction — skills never reach back into ``GP8App``. Adding a new skill
    is therefore a single new file plus one line in the ``ActionSelector``
    wiring.
    """

    #: Stable identifier used by ActionSelector wiring/routing.
    name: str = "skill"

    def __init__(self, ctx: "SkillContext") -> None:
        self.ctx = ctx

    def can_handle(self, target: "TrackedObject") -> bool:
        """Whether this skill is applicable to ``target``.

        Default: yes. Override to refuse objects this skill cannot handle
        (e.g. a class out of the throw envelope) so the selector falls back.
        """
        return True

    def idle_target(self) -> "np.ndarray":
        """Joint pose the post-action chain returns to when there is NO next
        object to pre-position.

        Default: the shared standby pose on the context
        (:attr:`SkillContext.idle_joint`, from ``cfg.INITIAL_R/T``, wrist zeroed),
        computed once at setup. Override to give a skill its own idle pose —
        mirrors :meth:`can_handle` (base default + per-skill override). Returns a
        TARGET, not a motion, so the return stays folded into the skill's single
        chained trajectory (no extra dispatch / queue-mode re-entry).
        """
        return self.ctx.idle_joint

    def arrival_lead(self) -> float:
        """Seconds before the object's predicted arrival to END the WAITING block.

        The wait helpers (``ctx.position_and_prime`` / ``ctx.wait_for_arrival``)
        stop ``arrival_lead()`` seconds early so the post-wait trajectory dispatch
        overlaps the object's final approach and the action lands ON arrival.
        Default: the shared
        :attr:`~gp8_control.config.Config.ACTION_START_LEAD` dispatch budget common
        to all skills (adv4ncr 250Hz stream: ~10-20 ms; the old ~0.4 s point-queue
        re-entry is gone). Override to ADD a skill-specific lead —
        mirrors :meth:`can_handle` / :meth:`idle_target` (base default + per-skill
        override). E.g. push waits BEHIND the contact line, so it overrides this to
        add its stroke's retreat->contact pre-travel.
        """
        return self.ctx.cfg.ACTION_START_LEAD

    def t_to_contact(self, move_time: float) -> float:
        """Wall-clock from "the arm starts moving toward the target" to "contact
        lands ON the object" — the budget ``earliest_reachable_intercept`` uses to
        place the grasp intercept (and judge feasibility): the object is aimed at
        where it will be ``t_to_contact`` from now.

        ``move_time`` is the solver's per-iteration ``opt_time`` positioning
        estimate (passed in so this stays a cheap scalar inside the fixed point —
        no re-planning). The full decomposition is
        ``T_setup (queue re-entry) + T_position + T_contact_offset``; a skill whose
        contact is timing-critical overrides this to add its real terms.

        Base default = the LEGACY estimate ``move_time * PICK_FEASIBILITY_FACTOR``,
        so a skill that doesn't override (THROW) keeps byte-identical behaviour —
        throw parks at the grasp and grabs PASSIVELY on the object's arrival
        (suction primed early), so it is timing-forgiving and its working heuristic
        is left untouched. Mirrors :meth:`arrival_lead` / :meth:`idle_target`
        (base default + per-skill override). PUSH overrides this — its contact is an
        active, timed sweep with no forgiveness, so it needs the honest timeline.
        """
        return move_time * self.ctx.cfg.PICK_FEASIBILITY_FACTOR

    # ------------------------------------------------------------------
    # Shared helpers (used by concrete skills)
    # ------------------------------------------------------------------
    def _ik_keyframes(self, transforms, wrist: float = 0.0):
        """Solve IK for each keyframe ``transforms`` and set joint 6 (wrist) to
        ``wrist``. Returns a tuple of joint arrays (one per transform), or ``None``
        (after a warning) if any IK fails. Shared by ThrowSkill/PushSkill
        ``solve_keyframe_joints`` — throw passes ``wrist=0``, push a push-facing angle."""
        joints = []
        for T in transforms:
            q = self.ctx.robot.inverse_kinematics(T)
            if q is None:
                self.ctx.log.warn("IK failed for a keyframe; aborting")
                return None
            q = np.asarray(q, dtype=float)
            q[-1] = wrist
            joints.append(q)
        return tuple(joints)

    def _abort(self, detail: str) -> "SkillResult":
        """Shared execute() failure cleanup: release suction, clear the active target,
        go IDLE, and return a failed ``SkillResult(detail)``. The caller logs the
        specific error first."""
        ctx = self.ctx
        ctx.traj_ctrl.suction_off()
        ctx.set_active_target(None)
        ctx.set_status("IDLE", "")
        return SkillResult(False, detail)

    @staticmethod
    def _append_csv_row(path: str, row: dict, logger=None) -> None:
        """Append one dict ``row`` to the CSV at ``path`` (writing the header when the
        file is new/empty). No-op if ``path`` is falsy. OSError is swallowed (logged
        via ``logger`` if given). Shared by the skills' pick-cycle timing logs."""
        if not path:
            return
        try:
            new_file = not os.path.exists(path) or os.path.getsize(path) == 0
            with open(path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if new_file:
                    w.writeheader()
                w.writerow(row)
        except OSError as e:
            if logger is not None:
                logger.warn(f"csv-log write failed: {e}")

    @abstractmethod
    def execute(self, request: "PickRequest") -> "SkillResult":
        """Perform the full manipulation for the selected target.

        ``request`` carries the target plus the intercept geometry produced by
        the app's target-selection step.
        """
        raise NotImplementedError

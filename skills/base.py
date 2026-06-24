"""Base class for manipulation skills."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

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

    @abstractmethod
    def execute(self, request: "PickRequest") -> "SkillResult":
        """Perform the full manipulation for the selected target.

        ``request`` carries the target plus the intercept geometry produced by
        the app's target-selection step.
        """
        raise NotImplementedError

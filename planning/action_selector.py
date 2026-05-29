"""Decide which manipulation skill handles a given object."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional

if TYPE_CHECKING:
    from gp8_control.skills.base import ManipulationSkill
    from gp8_control.skills.context import PickRequest


class ActionSelector:
    """Route each object to a manipulation skill (push vs throw vs ...).

    This is the decision point the project will eventually replace with a
    reinforcement-learning policy. Today it is rule-based:

      * an optional per-class override map (``by_class``: class_name -> skill
        name), else
      * the ``default`` skill.

    If the chosen skill reports it cannot handle the target
    (:meth:`ManipulationSkill.can_handle`), the selector falls back to the
    default. ``select`` returns a skill whose ``execute`` the app then calls —
    swapping in an RL policy means replacing only this class.
    """

    def __init__(
        self,
        skills: "Iterable[ManipulationSkill]",
        default: str = "throw",
        by_class: "Optional[dict[str, str]]" = None,
    ) -> None:
        self.skills = {s.name: s for s in skills}
        if default not in self.skills:
            raise ValueError(
                f"default skill {default!r} not among {list(self.skills)}"
            )
        self.default = default
        self.by_class = dict(by_class or {})

    def select(self, request: "PickRequest") -> "ManipulationSkill":
        """Return the skill that should handle ``request.target``."""
        name = self.by_class.get(request.target.class_name, self.default)
        skill = self.skills.get(name)
        if skill is None or not skill.can_handle(request.target):
            skill = self.skills[self.default]
        return skill

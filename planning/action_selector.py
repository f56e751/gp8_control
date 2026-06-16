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

    Testing override: ``force`` (skill name) pins every object to one skill,
    bypassing ``by_class``, ``default``, AND ``can_handle``. Use it to drive a
    single skill in isolation — e.g. ``force="throw"`` to test throwing only,
    or ``force="push"`` to test the push path even while PushSkill is still a
    stub (its ``can_handle`` would otherwise refuse and fall back to throw).
    Leave it ``None`` for normal routing.
    """

    def __init__(
        self,
        skills: "Iterable[ManipulationSkill]",
        default: str = "throw",
        by_class: "Optional[dict[str, str]]" = None,
        force: "Optional[str]" = None,
    ) -> None:
        self.skills = {s.name: s for s in skills}
        if default not in self.skills:
            raise ValueError(
                f"default skill {default!r} not among {list(self.skills)}"
            )
        if force is not None and force not in self.skills:
            raise ValueError(
                f"force skill {force!r} not among {list(self.skills)}"
            )
        self.default = default
        self.by_class = dict(by_class or {})
        self.force = force

    def skill_for(self, target) -> str:
        """Skill NAME that will handle ``target`` (force > by_class > default,
        with a can_handle fallback to default).

        Exposed so the chaining logic can pre-position the NEXT object with the
        SAME skill that will actually run it — keeping "approach + manipulate"
        one set per skill.
        """
        # Testing override: pin to one skill, ignoring routing and can_handle.
        if self.force is not None:
            return self.force
        name = self.by_class.get(target.class_name, self.default)
        skill = self.skills.get(name)
        if skill is None or not skill.can_handle(target):
            name = self.default
        return name

    def select(self, request: "PickRequest") -> "ManipulationSkill":
        """Return the skill that should handle ``request.target``."""
        return self.skills[self.skill_for(request.target)]

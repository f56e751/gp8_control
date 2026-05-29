"""Manipulation skills (throw, push, ...).

A *skill* is the executor half of the pick-and-place system: an
``ActionSelector`` (see ``gp8_control.planning.action_selector``) decides
which skill handles a given object, then calls ``execute`` on the chosen
skill. All shared robot resources and motion helpers live on the injected
``SkillContext`` so a new skill can be added as a single new file without
touching ``app.py``.
"""

from gp8_control.skills.base import ManipulationSkill, SkillResult
from gp8_control.skills.context import SkillContext, PickRequest
from gp8_control.skills.throw_skill import ThrowSkill
from gp8_control.skills.push_skill import PushSkill

__all__ = [
    "ManipulationSkill",
    "SkillResult",
    "SkillContext",
    "PickRequest",
    "ThrowSkill",
    "PushSkill",
]

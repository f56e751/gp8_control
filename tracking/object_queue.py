"""Persistent tracked-object queue with conveyor-motion compensation.

Each entry remembers ``detect_time`` so its current Y is recomputed every
epoch from the live conveyor speed. Anything past ``-max_reach`` is
dropped, the rest sorted ascending by current Y so the head is the next
reachable target.

Frame-cooldown timing lives in ``FrameGate`` — this module is purely
about object lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrackedObject:
    T_aim_base: np.ndarray   # pose at detection time (absolute, conveyor not yet applied)
    T_grasp_base: np.ndarray
    class_name: str
    detect_time: float       # time.time() when this object was observed


class TrackedObjectQueue:
    def __init__(self, max_reach: float) -> None:
        self._max_reach = max_reach
        self._objects: list[TrackedObject] = []

    def __len__(self) -> int:
        return len(self._objects)

    def __bool__(self) -> bool:
        return bool(self._objects)

    def head(self) -> TrackedObject:
        return self._objects[0]

    def has_next(self) -> bool:
        return len(self._objects) >= 2

    def peek_next(self) -> TrackedObject:
        return self._objects[1]

    def add(self, obj: TrackedObject) -> None:
        self._objects.append(obj)

    def pop_head(self) -> TrackedObject:
        return self._objects.pop(0)

    def update(self, now: float, conveyor_speed: float) -> None:
        """Drop anything past -max_reach, then sort by current Y.

        Single decorate-sort-undecorate pass — the original two-step
        version recomputed the conveyor offset for every object twice.
        """
        v = conveyor_speed
        decorated = [
            (obj.T_aim_base[1, 3] - v * (now - obj.detect_time), obj)
            for obj in self._objects
        ]
        decorated = [(y, obj) for y, obj in decorated if y > -self._max_reach]
        decorated.sort(key=lambda pair: pair[0])
        self._objects = [obj for _, obj in decorated]

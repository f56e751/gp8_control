"""Clock abstractions for realtime and simulator-time execution."""

from __future__ import annotations

import time
from typing import Protocol


class Clock(Protocol):
    def time(self) -> float:
        """Return the execution timeline time in seconds."""

    def monotonic(self) -> float:
        """Return a monotonic clock on the same timeline as ``time``."""

    def sleep(self, seconds: float) -> None:
        """Block or advance the execution timeline by ``seconds``."""


class WallClock:
    def time(self) -> float:
        return time.time()

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(max(0.0, float(seconds)))


WALL_CLOCK = WallClock()

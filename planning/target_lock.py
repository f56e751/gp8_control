"""Outcome of trying to lock the queue head as the next pick target.

Replaces the previous ``return None`` for three different reasons (drop
on IK failure, drop because the target passed, retry because we need to
wait) with an explicit, exhaustive enum + payload. The caller in
``run_epoch`` reads ``status`` and decides whether to sleep, drop, or
proceed — sleeping is no longer hidden inside the lock attempt.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from gp8_control.tracking.object_queue import TrackedObject, TrackedObjectQueue


class TargetStatus(Enum):
    LOCKED = "locked"           # caller may execute pick/throw
    WAIT = "wait"               # not yet in reach; head left in queue
    DROPPED_IK = "dropped_ik"   # IK failed; head removed
    DROPPED_PASSED = "dropped_passed"  # target moved past the arm; head removed


@dataclass
class TargetLock:
    status: TargetStatus
    target: TrackedObject | None = None
    T_aim: np.ndarray | None = None
    T_grasp: np.ndarray | None = None


def lock_or_drop_head(
    queue: TrackedObjectQueue,
    planner,
    current_joint: np.ndarray,
    conveyor_speed: float,
    now: float,
    fixed_delay: float,
    wait_threshold: float = 0.001,
) -> TargetLock:
    """Try to lock the queue head; mutate the queue based on outcome.

    Mutation contract — read this carefully, callers depend on it:
        LOCKED          → head is popped and returned in ``target``
        DROPPED_IK      → head is popped (IK failure, will not retry)
        DROPPED_PASSED  → head is popped (already past the reach arc)
        WAIT            → head stays in the queue (caller retries next epoch)

    In other words: LOCKED and DROPPED_* both consume the head; only
    WAIT leaves the queue unchanged. The function name reflects this —
    the head is either locked (and consumed) or dropped (and consumed).
    """
    target = queue.head()
    T_aim, T_grasp, _, wait_time, neg_wait_time = planner.plan_pick(
        target.T_aim_base.copy(),
        target.T_grasp_base.copy(),
        target.detect_time,
        current_joint,
        conveyor_speed,
        now,
        fixed_delay=fixed_delay,
    )

    if T_aim is None:
        queue.pop_head()
        return TargetLock(TargetStatus.DROPPED_IK)

    if neg_wait_time is not None:
        queue.pop_head()
        return TargetLock(TargetStatus.DROPPED_PASSED)

    if wait_time is not None and wait_time > wait_threshold:
        return TargetLock(TargetStatus.WAIT)

    queue.pop_head()
    return TargetLock(
        TargetStatus.LOCKED, target=target, T_aim=T_aim, T_grasp=T_grasp,
    )

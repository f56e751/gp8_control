"""WorldSource — the perception + belt seam app/skills consume, HW **or** sim.

Companion to :mod:`gp8_control.backends.robot_base`. Where ``RobotBackend``
abstracts "the arm", this abstracts "the belt world": the stream of corrected,
base-frame object detections and the conveyor's speed/encoder-distance state.
Two implementations:

  * **hardware** (:mod:`gp8_control.backends.ros_world`): subscribes to the
    ``camera_debug`` node's corrected snapshot on ``/camera_debug/detections``
    and wraps the existing :class:`ConveyorSpeedTracker`
    (``/conveyor/speed`` + ``/conveyor/distance_mm``).
  * **simulator** (:mod:`gp8_control.backends.mujoco_sim`): synthesizes the
    SAME snapshot dict from the physics boxes' base-frame poses and integrates
    an encoder-equivalent belt distance from the stepped physics.

Design notes
------------
* The snapshot is deliberately the **raw schema-v2 dict** that ``camera_debug``
  already publishes (JSON-decoded) — NOT a dataclass. ``DetectionIntake``
  consumes it via ``.get()`` with defaults, so keeping the dict shape means the
  intake, dedup, association, and bbox paths run UNCHANGED on both worlds.
  Schema reference: envelope ``{receipt_time, belt_mps, detections, ...}``;
  per-detection ``{class, confidence, cam, cam_bbox, base_grasp, base_aim,
  base_bbox_grasp, base_bbox_aim, in_workspace}`` (see camera_debug.py).
  ``receipt_time`` MUST share the app's ``time.time()`` base and be strictly
  increasing per new frame — the intake's duplicate-frame guard keys on it.
* Belt state is exposed as a **ConveyorSpeedTracker-compatible object**
  (``.current``, ``.distance_m``, ``.distance_at(wall_t)``,
  ``.check_freshness()``) so ``app.conveyor = world.belt`` leaves every
  existing call site and the ``SkillContext.conveyor`` field untouched.
  Encoder distance is the authoritative record of past belt motion; speed is
  for future projection/ETA.
* Belt-frame convention (unchanged): the belt travels in ``-Y`` (Y DECREASES
  as an object advances).

This module imports NO rclpy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class WorldSource(ABC):
    """Perception + belt-state source; HW and sim subclass it.

    Minimal by design: the app only ever needs the latest corrected snapshot
    and the belt tracker. HOW those are produced (a live camera vs. a physics
    twin) is entirely the subclass's business.
    """

    @abstractmethod
    def latest_snapshot(self) -> Optional[dict]:
        """Most recent perception snapshot (schema-v2 dict), or ``None`` before
        the first frame.

        Must be cheap/non-blocking (the control loop calls it every epoch):
        return the last received/computed frame, do not fetch synchronously.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def belt(self):
        """The belt-state tracker (ConveyorSpeedTracker-compatible duck type):
        ``.current`` [m/s], ``.distance_m`` [m or None], ``.distance_at(wall_t)``,
        ``.check_freshness()``. The app assigns this to ``self.conveyor``."""
        raise NotImplementedError

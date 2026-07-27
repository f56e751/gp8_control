"""Conveyor-belt telemetry.

Not perception (no camera/detection): ``ConveyorSpeedTracker`` simply
subscribes to the belt-speed topic (``/conveyor/speed``, published by the
encoder node) and exposes the latest value. Kept in its own package so the
belt-state concern is separate from camera/detection code.
"""

from gp8_control.conveyor.conveyor_speed import ConveyorSpeedTracker
from gp8_control.conveyor.camera_speed import CameraSpeedTracker

__all__ = ["ConveyorSpeedTracker", "CameraSpeedTracker"]

"""Backend seams: RobotBackend (the arm) + WorldSource (the belt world).

The hardware implementations are :class:`TrajectoryController`
(``controllers/trajectory_controller.py``) and :class:`HardwareWorldSource`
(``ros_world.py``); the MuJoCo twin lives in ``mujoco_sim.py``. The heavy /
optional imports (rclpy, mujoco) stay in their own modules — import those
directly where needed.
"""

from gp8_control.backends.robot_base import RobotBackend
from gp8_control.backends.world_base import WorldSource

__all__ = ["RobotBackend", "WorldSource"]

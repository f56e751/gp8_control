"""Sim-to-real validation of the RL simulator env (rclpy-free).

* :mod:`.recorder` — attaches to any :class:`RobotBackend` + app/runner and
  records the 250 Hz command stream, joint feedback, suction IO, per-step
  observations (in the training schema) and per-track detection updates.
* :mod:`.analyze` — turns a recorded run into ``report.md``/``report.json``:
  command->motion fidelity (incl. replaying the HW command stream through the
  MuJoCo twin) and state-channel fidelity vs. a matched sim reference rollout.
"""

from gp8_control.validation.obs_schema import ObsSchema

__all__ = ["ObsSchema"]

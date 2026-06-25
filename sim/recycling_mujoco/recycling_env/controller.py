from __future__ import annotations

import numpy as np

from .sim_env import RecyclingSimEnv
from .trajectory import JointTrajectory


class JointTrajectoryController:
    def __init__(self, env: RecyclingSimEnv, joint_names: list[str]) -> None:
        self.env = env
        self.joint_names = list(joint_names)
        self.actuator_names = [f"{joint_name}_act" for joint_name in self.joint_names]
        self.trajectory: JointTrajectory | None = None
        self.start_time: float | None = None
        self.last_target = self.env.get_joint_positions(self.joint_names)

    def load_trajectory(self, trajectory: JointTrajectory, start_time: float | None = None) -> None:
        if trajectory.joint_names != self.joint_names:
            raise ValueError(
                f"Trajectory joints {trajectory.joint_names} do not match controller joints {self.joint_names}"
            )
        self.trajectory = trajectory
        self.start_time = self.env.sim_time if start_time is None else float(start_time)
        self.last_target = trajectory.sample(trajectory.times[0])

    def clear(self) -> None:
        self.trajectory = None
        self.start_time = None

    def hold_position(self, q_target: np.ndarray) -> None:
        q_target = np.asarray(q_target, dtype=np.float64)
        if q_target.shape != (len(self.joint_names),):
            raise ValueError("q_target shape does not match number of joints")
        for actuator_name, value in zip(self.actuator_names, q_target):
            self.env.set_actuator_ctrl(actuator_name, float(value))
        self.last_target = q_target.copy()

    def update(self, current_time: float | None = None) -> np.ndarray:
        if self.trajectory is None:
            self.hold_position(self.last_target)
            return self.last_target.copy()

        sim_time = self.env.sim_time if current_time is None else float(current_time)
        local_time = self.trajectory.times[0] + (sim_time - float(self.start_time))
        target = self.trajectory.sample(local_time)
        self.hold_position(target)
        return target

    def is_finished(self, current_time: float | None = None) -> bool:
        if self.trajectory is None or self.start_time is None:
            return True
        sim_time = self.env.sim_time if current_time is None else float(current_time)
        elapsed = sim_time - self.start_time
        return elapsed >= self.trajectory.duration

    def get_tracking_state(self) -> tuple[np.ndarray, np.ndarray]:
        actual = self.env.get_joint_positions(self.joint_names)
        desired = self.last_target.copy()
        return desired, actual

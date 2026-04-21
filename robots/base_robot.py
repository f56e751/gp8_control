from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np


class BaseRobot(ABC):
    """Abstract base class for robot models."""

    @abstractmethod
    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """Compute forward kinematics. (n,) -> SE(3) 4x4"""
        ...

    @abstractmethod
    def inverse_kinematics(
        self, T: np.ndarray, q_init: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """Compute inverse kinematics. SE(3) 4x4 -> (n,) or None if failed"""
        ...

    @abstractmethod
    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """Compute Jacobian. (n,) -> 6xn"""
        ...

    @property
    @abstractmethod
    def n_joints(self) -> int:
        """Number of joints."""
        ...

    @property
    @abstractmethod
    def joint_limits(self) -> List[Tuple[float, float]]:
        """Joint position limits as list of (lower, upper) tuples in radians."""
        ...

    @property
    @abstractmethod
    def velocity_limits(self) -> np.ndarray:
        """Joint velocity limits in rad/s."""
        ...

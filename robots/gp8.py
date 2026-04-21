"""Yaskawa GP8 kinematic model using Product-of-Exponentials (PoE) formulation.

Ported and cleaned up from legacy/src/robots/yaskawa_gp8.py.  Uses Lie-algebra
utilities from gp8_control.utils.lie_algebra for all screw-axis computations.

The GP8 is a 6-DOF industrial articulated robot with the following kinematic
structure (screw axes defined in the space frame):

    J1  z-axis rotation at (0, 0, 0.330)
    J2  y-axis rotation at (0.04, 0, 0.330)
    J3 -y-axis rotation at (0.04, 0, 0.675)
    J4 -x-axis rotation at (0.38, 0, 0.715)
    J5 -y-axis rotation at (0.38, 0, 0.715)
    J6 -x-axis rotation at (0.38, 0, 0.715)

End-effector home position: (0.705, 0, 0.715).
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from gp8_control.robots.base_robot import BaseRobot
from gp8_control.utils.lie_algebra import (
    adjoint_se3,
    exp_se3,
    screw_to_se3,
)


class GP8(BaseRobot):
    """Yaskawa GP8 kinematics via Product of Exponentials."""

    # Number of degrees of freedom
    _N_JOINTS: int = 6

    # Joint position limits (radians) -- from the GP8 datasheet / URDF
    _JOINT_LIMITS: List[Tuple[float, float]] = [
        (np.radians(-170.0), np.radians(170.0)),   # J1  S
        (np.radians(-65.0),  np.radians(145.0)),    # J2  L
        (np.radians(-70.0),  np.radians(190.0)),    # J3  U
        (np.radians(-190.0), np.radians(190.0)),    # J4  R
        (np.radians(-135.0), np.radians(135.0)),    # J5  B
        (np.radians(-360.0), np.radians(360.0)),    # J6  T
    ]

    # Maximum joint velocities (rad/s) -- from the GP8 datasheet
    _VELOCITY_LIMITS: np.ndarray = np.radians(
        np.array([455.0, 385.0, 520.0, 550.0, 550.0, 1000.0])
    )

    def __init__(self) -> None:
        self._screws, self._joint_positions = self._build_screws()
        self._M = self._build_home_ee()

    # ------------------------------------------------------------------
    # BaseRobot abstract property implementations
    # ------------------------------------------------------------------

    @property
    def n_joints(self) -> int:
        return self._N_JOINTS

    @property
    def joint_limits(self) -> List[Tuple[float, float]]:
        return list(self._JOINT_LIMITS)

    @property
    def velocity_limits(self) -> np.ndarray:
        return self._VELOCITY_LIMITS.copy()

    # ------------------------------------------------------------------
    # Kinematic parameter construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_screws() -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Build the space-frame screw axes and joint-axis positions."""
        # Joint rotation axes (space frame)
        w = [
            np.array([0.0,  0.0,  1.0]),   # J1
            np.array([0.0,  1.0,  0.0]),   # J2
            np.array([0.0, -1.0,  0.0]),   # J3
            np.array([-1.0, 0.0,  0.0]),   # J4
            np.array([0.0, -1.0,  0.0]),   # J5
            np.array([-1.0, 0.0,  0.0]),   # J6
        ]

        # Points on each joint axis (metres)
        q = [
            np.array([0.0,  0.0,  0.330]),
            np.array([0.04, 0.0,  0.330]),
            np.array([0.04, 0.0,  0.675]),
            np.array([0.38, 0.0,  0.715]),
            np.array([0.38, 0.0,  0.715]),
            np.array([0.38, 0.0,  0.715]),
        ]

        # Space-frame screw axes: S = [w; -w x q]
        screws = [
            np.concatenate([wi, -np.cross(wi, qi)]) for wi, qi in zip(w, q)
        ]
        return screws, q

    @staticmethod
    def _build_home_ee() -> np.ndarray:
        """Home configuration SE(3) of the end-effector (all joints zero)."""
        M = np.eye(4)
        M[0, 3] = 0.705
        M[2, 3] = 0.715
        return M

    # ------------------------------------------------------------------
    # Forward kinematics (PoE)
    # ------------------------------------------------------------------

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """Compute end-effector pose via the Product of Exponentials.

        T_ee = exp(S1*q1) * exp(S2*q2) * ... * exp(S6*q6) * M

        Args:
            q: Joint angles of shape (6,) in radians.

        Returns:
            4x4 homogeneous transformation matrix of the end-effector.
        """
        q = np.asarray(q, dtype=np.float64).ravel()
        T = np.eye(4)
        for Si, qi in zip(self._screws, q):
            T = T @ screw_to_se3(Si, qi)
        return T @ self._M

    def forward_kinematics_all(
        self, q: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """FK returning both per-link frames and the EE frame.

        Args:
            q: Joint angles of shape (6,).

        Returns:
            (link_frames, ee_frame) where link_frames has shape (6, 4, 4)
            and ee_frame has shape (4, 4).
        """
        q = np.asarray(q, dtype=np.float64).ravel()
        link_frames = np.zeros((self._N_JOINTS, 4, 4))
        T = np.eye(4)
        for i, (Si, qi) in enumerate(zip(self._screws, q)):
            T = T @ screw_to_se3(Si, qi)
            link_frames[i] = T
        ee_frame = T @ self._M
        return link_frames, ee_frame

    # ------------------------------------------------------------------
    # Jacobian (space Jacobian)
    # ------------------------------------------------------------------

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """Compute the 6xN space Jacobian via PoE.

        J_s(:, i) = Ad_{T_0...T_{i-1}} * S_i

        Args:
            q: Joint angles of shape (6,).

        Returns:
            6x6 space Jacobian matrix.
        """
        q = np.asarray(q, dtype=np.float64).ravel()
        link_frames, _ = self.forward_kinematics_all(q)

        Js = np.zeros((6, self._N_JOINTS))
        Js[:, 0] = self._screws[0]
        for i in range(1, self._N_JOINTS):
            Js[:, i] = adjoint_se3(link_frames[i - 1]) @ self._screws[i]
        return Js

    # ------------------------------------------------------------------
    # Inverse kinematics (analytical, ported from legacy)
    # ------------------------------------------------------------------

    def inverse_kinematics(
        self, T: np.ndarray, q_init: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """Analytical inverse kinematics for the GP8.

        Uses a geometric approach for joints 1-3 (cosine law) and Euler-angle
        decomposition for joints 4-6.

        Args:
            T: Desired 4x4 end-effector pose.
            q_init: Ignored for analytical IK (kept for API compatibility).

        Returns:
            6-vector of joint angles in radians, or None if no valid solution.
        """
        T = np.asarray(T, dtype=np.float64)

        # Distance from joint-5 centre to the EE origin along the EE x-axis
        l_rod = self._M[0, 3] - self._joint_positions[4][0]

        # Wrist centre position (joint 4/5/6 coincide on GP8)
        wrist_pos = T[:3, 3] - l_rod * T[:3, 0]

        # --- Joints 1-3 via geometry/cosine law --------------------------
        theta_1 = np.arctan2(wrist_pos[1], wrist_pos[0])

        # Project wrist into the vertical plane through J1
        dist_xy = np.sqrt(wrist_pos[0] ** 2 + wrist_pos[1] ** 2)
        wrist_planar = np.array([dist_xy, 0.0, wrist_pos[2]])

        q2_pos = self._joint_positions[1]  # J2 zero-config position
        q3_pos = self._joint_positions[2]  # J3 zero-config position
        q5_pos = self._joint_positions[4]  # J5 zero-config position

        L1 = np.linalg.norm(q3_pos - q2_pos)
        L2 = np.linalg.norm(q5_pos - q3_pos)
        alpha_0 = np.arctan2(
            q5_pos[2] - q3_pos[2], q5_pos[0] - q3_pos[0]
        )

        L = np.linalg.norm(wrist_planar - q2_pos)

        # Reachability check
        if L > L1 + L2 or L < np.abs(L1 - L2):
            return None

        cos_alpha1 = np.clip(
            (L1**2 + L**2 - L2**2) / (2.0 * L1 * L), -1.0, 1.0
        )
        cos_alpha2 = np.clip(
            (L1**2 + L2**2 - L**2) / (2.0 * L1 * L2), -1.0, 1.0
        )

        alpha_1 = np.arccos(cos_alpha1)
        alpha_2 = np.arccos(cos_alpha2)
        alpha_3 = np.arctan2(
            wrist_planar[2] - q2_pos[2], wrist_planar[0] - q2_pos[0]
        )

        theta_2 = np.pi / 2.0 - (alpha_1 + alpha_3)
        theta_3 = (np.pi / 2.0 - alpha_0) - (np.pi - alpha_2)

        # --- Joints 4-6 via Euler-angle decomposition ---------------------
        R_03 = R.from_euler("yz", [theta_2 - theta_3, theta_1])
        R_euler = R_03.as_matrix().T @ T[:3, :3]
        theta_6, theta_5, theta_4 = R.from_matrix(R_euler).as_euler("xyx")
        theta_4, theta_5, theta_6 = -theta_4, -theta_5, -theta_6

        first_soln = np.array([
            theta_1, theta_2, theta_3, theta_4, theta_5, theta_6
        ])

        # There is a second Euler-angle solution
        second_soln = first_soln.copy()
        second_soln[3] += np.pi if first_soln[3] <= 0 else -np.pi
        second_soln[4] = -first_soln[4]
        second_soln[5] += np.pi if first_soln[5] <= 0 else -np.pi

        valid1 = self._is_within_limits(first_soln)
        valid2 = self._is_within_limits(second_soln)

        if valid1 and valid2:
            # Prefer the solution with smaller wrist rotation
            return (
                first_soln
                if np.abs(first_soln[3]) <= np.abs(second_soln[3])
                else second_soln
            )
        if valid1:
            return first_soln
        if valid2:
            return second_soln

        # Neither solution is within joint limits
        return None

    def inverse_kinematics_numerical(
        self,
        T: np.ndarray,
        q_init: Optional[np.ndarray] = None,
        tol: float = 1e-6,
        max_iter: int = 100,
        damping: float = 1e-3,
    ) -> Optional[np.ndarray]:
        """Numerical (damped least-squares) inverse kinematics fallback.

        Uses the Levenberg-Marquardt style update:
            dq = J^T (J J^T + lambda I)^{-1} e

        Args:
            T: Desired 4x4 end-effector pose.
            q_init: Initial joint guess (defaults to zeros).
            tol: Convergence tolerance on the twist error norm.
            max_iter: Maximum iterations.
            damping: Damping factor lambda.

        Returns:
            Joint angles (6,) or None if not converged.
        """
        from gp8_control.utils.lie_algebra import log_se3, inv_se3

        q = (
            np.asarray(q_init, dtype=np.float64).copy()
            if q_init is not None
            else np.zeros(self._N_JOINTS)
        )

        for _ in range(max_iter):
            T_curr = self.forward_kinematics(q)
            err_T = T @ inv_se3(T_curr)
            xi = log_se3(err_T)
            if np.linalg.norm(xi) < tol:
                return q

            J = self.jacobian(q)
            JJt = J @ J.T + damping * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, xi)
            q = q + dq

            # Clamp to joint limits
            for i, (lo, hi) in enumerate(self._JOINT_LIMITS):
                q[i] = np.clip(q[i], lo, hi)

        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_within_limits(self, q: np.ndarray) -> bool:
        """Check whether all joint values are within position limits."""
        for i, (lo, hi) in enumerate(self._JOINT_LIMITS):
            if q[i] < lo or q[i] > hi:
                return False
        return True

    @property
    def home_ee(self) -> np.ndarray:
        """Home end-effector frame M (all joints at zero)."""
        return self._M.copy()

    @property
    def screws(self) -> List[np.ndarray]:
        """List of 6-vector space-frame screw axes."""
        return [s.copy() for s in self._screws]

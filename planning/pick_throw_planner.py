"""Pick & throw motion planner — pure compute.

No ROS, no wall-clock; ``now`` is passed in by the caller. Every piece
of mutable runtime state (conveyor speed, detect_time, current joint)
crosses the boundary as a method argument. This makes the planner cheap
to unit-test against synthetic poses.

Behaviour preserved from main_sam7:
  - 3-pass refinement of trajectory time vs predicted-position offset
  - Reachability projection onto the max_reach circle along the belt axis
  - Throw NN query inside the 2nd planner so the time estimate updates
    every iteration (using ``ThrowParams.from_raw`` for decoding)
"""

from __future__ import annotations

import numpy as np

from gp8_control.trajectory.trajectory_primitive import opt_time
from gp8_control.planning.throw_params import ThrowParams, ThrowDecodingConfig


def _position_adjustment_for_IK(
    T: np.ndarray, max_reach: float, conveyor_speed: float
):
    """main_sam7-style reachability projection.

    If the xy-projected position is outside max_reach, push the Y-component
    onto the circle boundary in the direction of belt travel and report the
    corresponding wait / missed-window.

    Returns:
        (T_projected, wait_time_s_or_None, neg_wait_time_s_or_None).
        Exactly one of wait_time / neg_wait_time will be non-None if
        projection had to be applied. Both None means input is reachable.
    """
    T_tmp = T.copy()
    if np.linalg.norm(T[:2, 3]) <= max_reach:
        return T_tmp, None, None

    denom = max_reach ** 2 - T[0, 3] ** 2
    if denom < 0.0:
        # Even with any Y we can't reach this X; treat as missed
        return T_tmp, None, (T[1, 3] + max_reach) / (conveyor_speed + 1e-6)

    y_boundary = float(np.sqrt(denom))
    if T[1, 3] > 0.0:
        T_tmp[1, 3] = y_boundary
        wait_time = (T[1, 3] - T_tmp[1, 3]) / (conveyor_speed + 1e-6)
        return T_tmp, wait_time, None

    T_tmp[1, 3] = -y_boundary
    neg_wait_time = (T[1, 3] - T_tmp[1, 3]) / (conveyor_speed + 1e-6)
    return T_tmp, None, neg_wait_time


def _rotate_xy(x: float, y: float, theta: float) -> tuple[float, float]:
    c, s = np.cos(-theta), np.sin(-theta)
    return c * x - s * y, s * x + c * y


class PickThrowPlanner:
    def __init__(
        self,
        robot,
        predictor,
        M1: np.ndarray,
        M2: np.ndarray,
        max_reach: float,
        target_distance: float,
        decoding: ThrowDecodingConfig,
    ) -> None:
        self.robot = robot
        self.predictor = predictor
        self.M1 = M1
        self.M2 = M2
        self.max_reach = max_reach
        self.target_distance = target_distance
        self.decoding = decoding

    # ------------------------------------------------------------------
    # Pick (main_sam7.dynamic_adjustments)
    # ------------------------------------------------------------------
    def plan_pick(
        self,
        T_aim: np.ndarray,
        T_grasp: np.ndarray,
        detect_time: float,
        current_joint: np.ndarray,
        conveyor_speed: float,
        now: float,
        fixed_delay: float = 0.0,
    ):
        """Iteratively converge on (pick aim, pick grasp) at execution time.

        Returns:
            (T_aim_tmp, T_grasp_tmp, traj_time, wait_time, neg_wait_time)
            or (None, None, None, None, None) on IK failure.
        """
        v = conveyor_speed
        T_aim_base = T_aim.copy()
        T_grasp_base = T_grasp.copy()
        T_aim_base[1, 3] -= v * (now - detect_time)
        T_grasp_base[1, 3] -= v * (now - detect_time)

        T_aim_tmp = T_aim_base
        T_grasp_tmp = T_grasp_base
        wait_time = None
        neg_wait_time = None

        traj_time = 1.0  # initial estimate
        for _ in range(3):
            offset = traj_time + fixed_delay
            T_aim_pred = T_aim_base.copy()
            T_aim_pred[1, 3] -= v * offset
            T_grasp_pred = T_grasp_base.copy()
            T_grasp_pred[1, 3] -= v * offset

            T_aim_tmp, wait_time, neg_wait_time = _position_adjustment_for_IK(
                T_aim_pred, self.max_reach, v
            )
            aim_j = self.robot.inverse_kinematics(T_aim_tmp)

            T_grasp_tmp, _, _ = _position_adjustment_for_IK(
                T_grasp_pred, self.max_reach, v
            )
            grasp_j = self.robot.inverse_kinematics(T_grasp_tmp)

            if aim_j is None or grasp_j is None:
                return None, None, None, None, None

            zero = np.zeros_like(self.M1)
            traj_time1 = opt_time(current_joint, zero, aim_j, zero, self.M1, self.M2)
            traj_time2 = opt_time(aim_j, zero, grasp_j, zero, self.M1, self.M2)
            traj_time = traj_time1 + traj_time2

        return T_aim_tmp, T_grasp_tmp, traj_time, wait_time, neg_wait_time

    # ------------------------------------------------------------------
    # Throw landing (main_sam7.dynamic_adjustments2)
    # ------------------------------------------------------------------
    def plan_throw_landing(
        self,
        T_grasp1: np.ndarray,
        T_aim2: np.ndarray,
        theta: float,
        detect_time: float,
        conveyor_speed: float,
        now: float,
        fixed_delay: float = 0.1,
    ):
        """Iteratively refine throw landing, asking torch NN for throw time.

        ``wait_time`` (positive) is reported but main_sam7 deliberately
        ignores it for throws — a throw landing on the reach-circle
        boundary is acceptable. Only ``neg_wait_time`` (target already
        gone) is treated as infeasible by the caller.
        """
        v = conveyor_speed
        T_aim2_base = T_aim2.copy()
        T_aim2_base[1, 3] -= v * (now - detect_time)

        T_aim2_tmp = T_aim2_base
        wait_time = None
        neg_wait_time = None

        traj_time = 1.5
        for _ in range(3):
            offset = traj_time + fixed_delay
            T_aim2_pred = T_aim2_base.copy()
            T_aim2_pred[1, 3] -= v * offset

            T_aim2_tmp, wait_time, neg_wait_time = _position_adjustment_for_IK(
                T_aim2_pred, self.max_reach, v
            )

            x1, y1 = _rotate_xy(T_grasp1[0, 3], T_grasp1[1, 3], theta)
            x2, y2 = _rotate_xy(T_aim2_tmp[0, 3], T_aim2_tmp[1, 3], theta)
            raw = self.predictor.predict((x1, y1), (x2, y2), self.target_distance)
            traj_time = ThrowParams.from_raw(raw, self.decoding).T

        return T_aim2_tmp, traj_time, wait_time, neg_wait_time

    # ------------------------------------------------------------------
    # Final NN query (full decoded params)
    # ------------------------------------------------------------------
    def compute_throw_params(
        self, T_grasp: np.ndarray, T_aim2: np.ndarray, theta: float
    ) -> ThrowParams:
        x1, y1 = _rotate_xy(T_grasp[0, 3], T_grasp[1, 3], theta)
        x2, y2 = _rotate_xy(T_aim2[0, 3], T_aim2[1, 3], theta)
        raw = self.predictor.predict((x1, y1), (x2, y2), self.target_distance)
        return ThrowParams.from_raw(raw, self.decoding)

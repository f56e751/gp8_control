"""MoveIt 2 based robot controller for initial pose and planning.

Used only for setup (moving to initial pose). Runtime trajectory
execution goes through TrajectoryController instead.

Uses MoveGroup action interface (move_group node must be running).
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    PlanningOptions,
    RobotState,
)
from sensor_msgs.msg import JointState

JOINT_NAMES = [
    "joint_1_s", "joint_2_l", "joint_3_u",
    "joint_4_r", "joint_5_b", "joint_6_t",
]


class MoveItController:
    """High-level MoveIt 2 interface for GP8 via MoveGroup action."""

    def __init__(self, node: Node, group_name: str = "motoman_gp8") -> None:
        self._node = node
        self._group_name = group_name
        cb_group = ReentrantCallbackGroup()

        self._move_group_client = ActionClient(
            self._node, MoveGroup, "move_action",
            callback_group=cb_group,
        )
        self._node.get_logger().info("Waiting for MoveGroup action server...")
        self._move_group_client.wait_for_server(timeout_sec=30.0)
        self._node.get_logger().info("MoveGroup action server ready.")

    def set_joint_state(self, goal_joint: list, wait: bool = True) -> bool:
        """Move robot to target joint state via MoveIt planning."""
        # Build joint constraints
        constraints = Constraints()
        for name, value in zip(JOINT_NAMES, goal_joint):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(value)
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)

        # Build motion plan request
        request = MotionPlanRequest()
        request.group_name = self._group_name
        request.goal_constraints.append(constraints)
        request.num_planning_attempts = 5
        request.allowed_planning_time = 5.0

        # Build MoveGroup goal
        goal = MoveGroup.Goal()
        goal.request = request
        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = False  # plan and execute

        # Send goal
        future = self._move_group_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self._node, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self._node.get_logger().error("MoveGroup goal rejected.")
            return False

        if wait:
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self._node, result_future)
            result = result_future.result().result
            if result.error_code.val == result.error_code.SUCCESS:
                self._node.get_logger().info("MoveIt planning + execution succeeded.")
                return True
            self._node.get_logger().error(
                f"MoveIt failed with error code: {result.error_code.val}"
            )
            return False

        return True

    def shutdown(self) -> None:
        pass

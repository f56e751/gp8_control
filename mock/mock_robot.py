"""Mock robot node for development without real hardware.

Exposes the *bridge-facing* interface that every downstream component
already targets, so switching between real robot (MotoROS2 + name bridge)
and mock requires no client-side changes:

  - /joint_states_urdf publisher (URDF joint names)
  - /motoman_gp8_controller/follow_joint_trajectory action server
  - /write_single_io service (suction on/off)
  - /robot_enable service (robot enable trigger)

Usage:
  ros2 run gp8_control mock_robot
"""

from __future__ import annotations

import copy
import math
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from motoros2_interfaces.srv import WriteSingleIO
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger


JOINT_NAMES = [
    "joint_1_s", "joint_2_l", "joint_3_u",
    "joint_4_r", "joint_5_b", "joint_6_t",
]


class MockRobot(Node):
    """Simulates MotoROS2 interfaces for development testing."""

    def __init__(self) -> None:
        super().__init__("mock_robot")
        self._lock = threading.Lock()
        # GP8 neutral posture (same as PyBullet ThrowEnv HOME_JOINTS)
        import math
        self._joint_positions = [0.0, 0.0, 0.0, 0.0, -math.pi / 2, 0.0]
        self._joint_velocities = [0.0] * 6
        self._suction_on = False

        cb_group = ReentrantCallbackGroup()

        # Publisher: /joint_states_urdf (matches bridge output)
        self._js_pub = self.create_publisher(
            JointState, "/joint_states_urdf", qos_profile_sensor_data,
        )
        self._js_timer = self.create_timer(0.02, self._publish_joint_states)  # 50 Hz

        # Action server: /motoman_gp8_controller/follow_joint_trajectory (matches bridge)
        self._fjt_server = ActionServer(
            self,
            FollowJointTrajectory,
            "/motoman_gp8_controller/follow_joint_trajectory",
            execute_callback=self._execute_trajectory,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=cb_group,
        )

        # Service: /write_single_io (suction)
        self.create_service(
            WriteSingleIO, "/write_single_io", self._write_io_cb,
            callback_group=cb_group,
        )

        # Service: /robot_enable
        self.create_service(
            Trigger, "/robot_enable", self._robot_enable_cb,
            callback_group=cb_group,
        )

        self.get_logger().info(
            "Mock robot ready: /joint_states_urdf, "
            "/motoman_gp8_controller/follow_joint_trajectory, "
            "/write_single_io, /robot_enable"
        )

    # ------------------------------------------------------------------
    # Joint state publisher
    # ------------------------------------------------------------------

    def _publish_joint_states(self) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        with self._lock:
            msg.position = list(self._joint_positions)
            msg.velocity = list(self._joint_velocities)
        msg.effort = [0.0] * 6
        self._js_pub.publish(msg)

    # ------------------------------------------------------------------
    # FollowJointTrajectory action
    # ------------------------------------------------------------------

    def _goal_callback(self, goal_request):
        self.get_logger().info(
            f"Trajectory goal received: {len(goal_request.trajectory.points)} points"
        )
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle):
        self.get_logger().info("Trajectory cancel requested")
        return CancelResponse.ACCEPT

    async def _execute_trajectory(self, goal_handle):
        """Simulate trajectory execution by interpolating through waypoints."""
        trajectory = goal_handle.request.trajectory
        feedback_msg = FollowJointTrajectory.Feedback()

        for i, point in enumerate(trajectory.points):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return FollowJointTrajectory.Result()

            # Update joint positions to this waypoint
            with self._lock:
                self._joint_positions = list(point.positions)
                if point.velocities:
                    self._joint_velocities = list(point.velocities)

            # Feedback
            feedback_msg.actual.positions = list(point.positions)
            feedback_msg.actual.time_from_start = point.time_from_start
            goal_handle.publish_feedback(feedback_msg)

            # Wait proportional to time_from_start difference
            if i < len(trajectory.points) - 1:
                dt = self._duration_to_sec(trajectory.points[i + 1].time_from_start) - \
                     self._duration_to_sec(point.time_from_start)
                # Speed up simulation: use 10% of real time
                time.sleep(max(0.01, dt * 0.1))

        # Final position
        if trajectory.points:
            with self._lock:
                self._joint_positions = list(trajectory.points[-1].positions)
                self._joint_velocities = [0.0] * 6

        goal_handle.succeed()
        result = FollowJointTrajectory.Result()
        self.get_logger().info("Trajectory execution complete")
        return result

    @staticmethod
    def _duration_to_sec(d: Duration) -> float:
        return d.sec + d.nanosec * 1e-9

    # ------------------------------------------------------------------
    # WriteSingleIO service (suction)
    # ------------------------------------------------------------------

    def _write_io_cb(self, request, response):
        self._suction_on = request.value == 0  # 0=ON, 1=OFF in the real system
        state = "ON" if self._suction_on else "OFF"
        self.get_logger().info(f"Suction {state} (address={request.address}, value={request.value})")
        response.success = True
        response.message = f"IO {request.address} set to {request.value}"
        return response

    # ------------------------------------------------------------------
    # Robot enable service
    # ------------------------------------------------------------------

    def _robot_enable_cb(self, request, response):
        self.get_logger().info("Robot enabled (mock)")
        response.success = True
        response.message = "Mock robot enabled"
        return response


def main():
    rclpy.init()
    node = MockRobot()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

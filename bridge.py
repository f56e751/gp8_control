"""Joint-name bridge between MotoROS2 and URDF-based consumers.

MotoROS2 on this controller publishes /joint_states and exposes the
/follow_joint_trajectory action with joint names joint_1..joint_6
(its default single-group naming). Every other piece of the codebase —
URDF, SRDF, MoveIt controllers config, GUI, Mock, RL kinematics — uses
the Motoman S/L/U/R/B/T convention: joint_1_s..joint_6_t.

To avoid a controller-side config reload (which requires pendant access),
this node translates names in both directions:

  /joint_states  (raw, BEST_EFFORT)
        │   [rename joint_1..6 -> joint_1_s..6_t]
        ▼
  /joint_states_urdf  (sensor_data QoS)                        ← consumers

  consumers ──► /motoman_gp8_controller/follow_joint_trajectory
                  │   [rename URDF -> raw, forward]
                  ▼
                /follow_joint_trajectory  (MotoROS2)
                  │   [rename raw -> URDF in feedback/result]
                  ▼
                caller

The action server path matches motoman_gp8_moveit_config's
moveit_controllers.yaml so MoveIt routes through the bridge with no
config change.

Usage:
    ros2 run gp8_control name_bridge
"""

from __future__ import annotations

import threading

import rclpy
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from control_msgs.action import FollowJointTrajectory
from motoros2_interfaces.srv import QueueTrajPoint
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


URDF_NAMES = [
    "joint_1_s", "joint_2_l", "joint_3_u",
    "joint_4_r", "joint_5_b", "joint_6_t",
]
RAW_NAMES = [
    "joint_1", "joint_2", "joint_3",
    "joint_4", "joint_5", "joint_6",
]

RAW_TO_URDF = dict(zip(RAW_NAMES, URDF_NAMES))
URDF_TO_RAW = dict(zip(URDF_NAMES, RAW_NAMES))


def _rename_joint_state(msg: JointState) -> JointState:
    """Return a new JointState with joint names translated raw → URDF."""
    out = JointState()
    out.header = msg.header
    out.name = [RAW_TO_URDF.get(n, n) for n in msg.name]
    out.position = list(msg.position)
    out.velocity = list(msg.velocity)
    out.effort = list(msg.effort)
    return out


def _translate_trajectory(traj: JointTrajectory, table: dict) -> JointTrajectory:
    """Return a new JointTrajectory with joint_names remapped via table.

    Point positions/velocities are reordered to match the new name order.
    If any name is missing from `table`, the name is passed through
    unchanged (and its column is kept as-is).
    """
    new_names = [table.get(n, n) for n in traj.joint_names]

    out = JointTrajectory()
    out.header = traj.header
    out.joint_names = new_names
    for pt in traj.points:
        new_pt = JointTrajectoryPoint()
        new_pt.positions = list(pt.positions)
        new_pt.velocities = list(pt.velocities)
        new_pt.accelerations = list(pt.accelerations)
        new_pt.effort = list(pt.effort)
        new_pt.time_from_start = pt.time_from_start
        out.points.append(new_pt)
    return out


class NameBridge(Node):
    """Bridge node: joint_state republisher + FollowJointTrajectory proxy."""

    def __init__(self) -> None:
        super().__init__("motoros2_name_bridge")

        cb_group = ReentrantCallbackGroup()

        # Joint state relay ------------------------------------------------
        self._js_pub = self.create_publisher(
            JointState, "/joint_states_urdf", qos_profile_sensor_data,
        )
        self.create_subscription(
            JointState, "/joint_states",
            self._on_joint_states, qos_profile_sensor_data,
            callback_group=cb_group,
        )

        # FJT proxy --------------------------------------------------------
        self._upstream = ActionClient(
            self, FollowJointTrajectory, "/follow_joint_trajectory",
            callback_group=cb_group,
        )
        self._server = ActionServer(
            self,
            FollowJointTrajectory,
            "/motoman_gp8_controller/follow_joint_trajectory",
            execute_callback=self._execute,
            goal_callback=self._on_goal,
            cancel_callback=self._on_cancel,
            callback_group=cb_group,
        )

        # Queue point proxy -----------------------------------------------
        # /motoman_gp8_controller/queue_traj_point (URDF names)
        #   → translate joint_names URDF→raw
        #   → /queue_traj_point (MotoROS2, raw names)
        self._queue_upstream = self.create_client(
            QueueTrajPoint, "/queue_traj_point",
            callback_group=cb_group,
        )
        self._queue_server = self.create_service(
            QueueTrajPoint,
            "/motoman_gp8_controller/queue_traj_point",
            self._on_queue_traj_point,
            callback_group=cb_group,
        )

        self.get_logger().info(
            "Name bridge ready: /joint_states_urdf, "
            "/motoman_gp8_controller/follow_joint_trajectory, "
            "/motoman_gp8_controller/queue_traj_point"
        )

    # ------------------------------------------------------------------
    # Joint state relay
    # ------------------------------------------------------------------

    def _on_joint_states(self, msg: JointState) -> None:
        self._js_pub.publish(_rename_joint_state(msg))

    # ------------------------------------------------------------------
    # FJT proxy
    # ------------------------------------------------------------------

    def _on_goal(self, goal_request):
        if not self._upstream.server_is_ready():
            self.get_logger().warn("Upstream FJT not ready; rejecting goal.")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _on_cancel(self, goal_handle):
        return CancelResponse.ACCEPT

    def _execute(self, goal_handle):
        """Forward the trajectory with names translated URDF → raw."""
        incoming = goal_handle.request
        translated_goal = FollowJointTrajectory.Goal()
        translated_goal.trajectory = _translate_trajectory(
            incoming.trajectory, URDF_TO_RAW,
        )
        translated_goal.multi_dof_trajectory = incoming.multi_dof_trajectory
        translated_goal.path_tolerance = incoming.path_tolerance
        translated_goal.component_path_tolerance = incoming.component_path_tolerance
        translated_goal.goal_tolerance = incoming.goal_tolerance
        translated_goal.component_goal_tolerance = incoming.component_goal_tolerance
        translated_goal.goal_time_tolerance = incoming.goal_time_tolerance

        done = threading.Event()
        state: dict = {"upstream_handle": None, "result": None, "status": None}

        def on_feedback(fb_msg):
            fb = fb_msg.feedback
            out = FollowJointTrajectory.Feedback()
            out.header = fb.header
            out.joint_names = [RAW_TO_URDF.get(n, n) for n in fb.joint_names]
            out.desired = fb.desired
            out.actual = fb.actual
            out.error = fb.error
            goal_handle.publish_feedback(out)

        send_future = self._upstream.send_goal_async(
            translated_goal, feedback_callback=on_feedback,
        )

        def on_goal_response(fut):
            handle = fut.result()
            if handle is None or not handle.accepted:
                state["result"] = FollowJointTrajectory.Result()
                state["status"] = 6  # STATUS_ABORTED
                done.set()
                return
            state["upstream_handle"] = handle
            result_future = handle.get_result_async()

            def on_result(rf):
                wrapped = rf.result()
                if wrapped is None:
                    state["result"] = FollowJointTrajectory.Result()
                    state["status"] = 6
                else:
                    state["result"] = wrapped.result
                    state["status"] = wrapped.status
                done.set()

            result_future.add_done_callback(on_result)

        send_future.add_done_callback(on_goal_response)

        # Wait for completion, polling for client-side cancel request.
        while not done.wait(timeout=0.1):
            if goal_handle.is_cancel_requested and state["upstream_handle"] is not None:
                cancel_future = state["upstream_handle"].cancel_goal_async()
                cancel_future.add_done_callback(lambda _f: None)

        status = state["status"]
        if goal_handle.is_cancel_requested or status == 5:  # STATUS_CANCELED
            goal_handle.canceled()
        elif status == 4:  # STATUS_SUCCEEDED
            goal_handle.succeed()
        else:
            goal_handle.abort()

        return state["result"] or FollowJointTrajectory.Result()

    # ------------------------------------------------------------------
    # Queue point proxy
    # ------------------------------------------------------------------

    def _on_queue_traj_point(self, request, response):
        """Translate URDF joint_names → raw and forward to MotoROS2."""
        if not self._queue_upstream.service_is_ready():
            # queue_traj_point 응답 타입 규약: result_code=UNSPECIFIED fallback
            # (MotoROS2 QueueResultEnum has no such code; use 3=INIT_FAILURE)
            response.result_code.value = 3
            response.message = "Upstream /queue_traj_point not ready."
            return response

        forwarded = QueueTrajPoint.Request()
        forwarded.joint_names = [URDF_TO_RAW.get(n, n) for n in request.joint_names]
        forwarded.point = request.point

        fut = self._queue_upstream.call_async(forwarded)
        done = threading.Event()
        fut.add_done_callback(lambda _f: done.set())
        if not done.wait(timeout=5.0):
            response.result_code.value = 3
            response.message = "Upstream /queue_traj_point timed out."
            return response

        upstream_res = fut.result()
        if upstream_res is None:
            response.result_code.value = 3
            response.message = "Upstream returned None."
            return response

        response.result_code = upstream_res.result_code
        response.message = upstream_res.message
        return response


def main() -> None:
    rclpy.init()
    node = NameBridge()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

"""Minimal stream-backend sim: mock robot + TF + RViz (no app, no MoveIt).

For RViz-testing code that drives the adv4ncr 250 Hz stream directly —
e.g. ``tests/line_stream_test.py`` (EE-line IK streaming) — without any
hardware. The (extended) ``mock_robot`` echoes
``/JointGroupPositionController/commands`` into ``/joint_states`` (ideal
servo), robot_state_publisher turns that into TF, and RViz shows the arm
moving in real time (the stream loop is wall-clock paced).

  ros2 launch gp8_control stream_sim.launch.py
  ros2 launch gp8_control stream_sim.launch.py rviz:=false   # headless

then, in another shell (ROS + workspace sourced):

  ros2 run gp8_control line_stream_test --sweep 0.5,1.0,1.5,2.0,2.5

(or the venv python with PYTHONPATH=$HOME/ros2_ws/src:$PYTHONPATH — the
``:$PYTHONPATH`` matters, a bare assignment drops rclpy.)

Needs motoman_gp8_support (URDF) like debug_robot.launch.py. The full-app
sim remains sim_bringup.launch.py; this one launches NOTHING that moves the
arm on its own.

⚠️ NEVER run this while the REAL robot stack is up on the same ROS domain:
the mock publishes fake /joint_states + /joint_states_urdf that interleave
with the real joint_state_broadcaster's, corrupting the live app's notion of
where the arm is. On the robot PC, isolate first:

  export ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1   # both terminals (sim + test)
"""

import os
import subprocess

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    use_rviz = LaunchConfiguration("rviz")
    args = [
        DeclareLaunchArgument("rviz", default_value="true",
                              description="Launch RViz"),
    ]

    # ---- robot model (URDF -> TF) -------------------------------------
    xacro_path = os.path.join(
        get_package_share_directory("motoman_gp8_support"), "urdf", "gp8.xacro",
    )
    robot_description = subprocess.check_output(["xacro", xacro_path], text=True)

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        parameters=[{"robot_description": robot_description}],
        remappings=[("joint_states", "joint_states_urdf")],
        output="screen",
    )

    # ---- mock robot (echoes the 250 Hz JGPC stream into joint states) --
    mock_robot = Node(
        package="gp8_control", executable="mock_robot", name="mock_robot",
        output="screen",
    )

    # ---- RViz ---------------------------------------------------------
    rviz = Node(
        package="rviz2", executable="rviz2", name="rviz2",
        arguments=["-d", PathJoinSubstitution([
            FindPackageShare("gp8_control"), "rviz", "sim.rviz",
        ])],
        condition=IfCondition(use_rviz),
        output="screen",
    )

    return LaunchDescription([
        *args,
        robot_state_publisher,
        mock_robot,
        rviz,
    ])

"""GP8 debug launch file (ROS 2).

Minimal launch for robot debugging — no camera, no SAM server, no perception.
Only launches:
  1. Name bridge (MotoROS2 raw names <-> URDF S/L/U/R/B/T names)
  2. Robot state publisher (URDF -> TF), consuming bridge's /joint_states_urdf
  3. MoveIt 2 (move_group)

MotoROS2 runs on the robot controller firmware — no node needed here.
It provides: /joint_states (raw names), /follow_joint_trajectory, /write_single_io.
The bridge translates those to the URDF convention used by everything else.

Usage:
  ros2 launch gp8_control debug_robot.launch.py
"""

import os
import subprocess
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node, SetRemap
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    xacro_path = os.path.join(
        get_package_share_directory("motoman_gp8_support"),
        "urdf", "gp8.xacro",
    )
    robot_description = subprocess.check_output(["xacro", xacro_path], text=True)

    # =====================================================================
    # 1. Name bridge (MotoROS2 raw names <-> URDF names)
    # =====================================================================
    name_bridge = Node(
        package="gp8_control",
        executable="name_bridge",
        name="motoros2_name_bridge",
        output="screen",
    )

    # =====================================================================
    # 2. Robot state publisher (URDF -> TF)
    #    Remap /joint_states -> /joint_states_urdf so TF uses bridge output.
    # =====================================================================
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        parameters=[{"robot_description": robot_description}],
        remappings=[("joint_states", "joint_states_urdf")],
        output="screen",
    )

    # =====================================================================
    # 3. MoveIt 2 — scoped remap so move_group reads bridge output.
    # =====================================================================
    moveit_launch = GroupAction([
        SetRemap(src="/joint_states", dst="/joint_states_urdf"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare("motoman_gp8_moveit_config"),
                    "launch", "move_group.launch.py",
                ])
            ]),
        ),
    ])

    # =====================================================================
    # Assemble
    # =====================================================================
    return LaunchDescription([
        name_bridge,
        robot_state_publisher,
        moveit_launch,
    ])

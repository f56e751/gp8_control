"""RViz-only preview for gp8_control suction_lift_debug.

This launch file does not start the real robot driver, controllers, MoveIt, or
any suction IO path.  It starts:

  1. robot_state_publisher with the GP8 URDF
  2. suction_lift_debug --rviz-preview, which publishes /joint_states plus
     marker/path topics
  3. RViz with the preview displays preconfigured

Use this before running the hardware debug script.
"""

import os
import subprocess

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    x = LaunchConfiguration("x")
    y = LaunchConfiguration("y")
    lift = LaunchConfiguration("lift")
    bin_x = LaunchConfiguration("bin_x")
    bin_y = LaunchConfiguration("bin_y")
    bin_z_offset = LaunchConfiguration("bin_z_offset")
    tool_offset = LaunchConfiguration("tool_offset")
    vel_scale = LaunchConfiguration("vel_scale")
    preview_rate = LaunchConfiguration("preview_rate")
    rviz_enabled = LaunchConfiguration("rviz")

    args = [
        DeclareLaunchArgument("x", default_value="0.55",
                              description="pick/grasp X [m]"),
        DeclareLaunchArgument("y", default_value="0.0",
                              description="pick/grasp Y [m]"),
        DeclareLaunchArgument("lift", default_value="0.10",
                              description="lift height after suction [m]"),
        DeclareLaunchArgument("bin_x", default_value="1.0",
                              description="target bin X [m]"),
        DeclareLaunchArgument("bin_y", default_value="0.0",
                              description="target bin Y [m]"),
        DeclareLaunchArgument("bin_z_offset", default_value="0.10",
                              description="target bin height offset from grasp z [m]"),
        DeclareLaunchArgument("tool_offset", default_value="0.0",
                              description="extra offset beyond gp8.py/MuJoCo TCP [m]"),
        DeclareLaunchArgument("vel_scale", default_value="0.3",
                              description="low-speed move velocity scale"),
        DeclareLaunchArgument("preview_rate", default_value="30.0",
                              description="preview /joint_states rate [Hz]"),
        DeclareLaunchArgument("rviz", default_value="true",
                              description="start RViz"),
    ]

    # Let the source tree version run before a colcon rebuild during iteration.
    ros2_ws_src = os.path.realpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
    )
    set_pythonpath = SetEnvironmentVariable(
        "PYTHONPATH", ros2_ws_src + ":" + os.environ.get("PYTHONPATH", ""),
    )

    xacro_path = os.path.join(
        get_package_share_directory("gp8_control"),
        "urdf",
        "gp8_mujoco_suction_tool.xacro",
    )
    robot_description = subprocess.check_output(["xacro", xacro_path], text=True)

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        parameters=[{"robot_description": robot_description}],
        output="screen",
    )

    preview_args = [
        "--rviz-preview",
        "--x", x,
        "--y", y,
        "--lift", lift,
        "--bin-x", bin_x,
        "--bin-y", bin_y,
        "--bin-z-offset", bin_z_offset,
        "--tool-offset", tool_offset,
        "--vel-scale", vel_scale,
        "--preview-rate", preview_rate,
    ]
    preview_node = Node(
        package="gp8_control",
        executable="suction_lift_debug",
        name="suction_lift_debug_rviz_preview",
        arguments=preview_args,
        output="screen",
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", PathJoinSubstitution([
            FindPackageShare("gp8_control"), "rviz", "suction_lift_preview.rviz",
        ])],
        condition=IfCondition(rviz_enabled),
        output="screen",
    )

    return LaunchDescription([
        *args,
        set_pythonpath,
        robot_state_publisher,
        preview_node,
        rviz,
    ])

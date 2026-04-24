"""GP8 debug bringup launch file (ROS 2).

Parallel to ``gp8_bringup.launch.py`` but for interactive keyboard debugging
(``terminal_debug``) instead of the autonomous ``gp8_app``. Launches:

  1. Name bridge (MotoROS2 raw names <-> URDF S/L/U/R/B/T names)
  2. Robot state publisher (URDF -> TF)
  3. MoveIt 2 (move_group)
  4. terminal_debug in a separate gnome-terminal window (optional)

MotoROS2 runs on the robot controller firmware — no node needed here.

Usage:
  # Auto-spawn terminal_debug in a new window (default)
  ros2 launch gp8_control debug_bringup.launch.py

  # Skip auto-spawn; run `ros2 run gp8_control terminal_debug` yourself
  ros2 launch gp8_control debug_bringup.launch.py spawn_terminal:=false
"""

import os
import subprocess

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    GroupAction,
    IncludeLaunchDescription,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetRemap
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    spawn_terminal_arg = DeclareLaunchArgument(
        "spawn_terminal", default_value="true",
        description="Auto-open terminal_debug in a new gnome-terminal window",
    )

    xacro_path = os.path.join(
        get_package_share_directory("motoman_gp8_support"),
        "urdf", "gp8.xacro",
    )
    robot_description = subprocess.check_output(["xacro", xacro_path], text=True)

    name_bridge = Node(
        package="gp8_control",
        executable="name_bridge",
        name="motoros2_name_bridge",
        output="screen",
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        parameters=[{"robot_description": robot_description}],
        remappings=[("joint_states", "joint_states_urdf")],
        output="screen",
    )

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

    # terminal_debug는 tty.setraw()로 raw 키보드 입력을 받으므로 별도 TTY 필요.
    # ros2 launch가 stdin을 자식 프로세스에 포워딩하지 않아서, gnome-terminal
    # 창을 새로 띄워서 거기서 실행.
    ros2_ws = os.path.expanduser("~/ros2_ws")
    terminal_cmd = (
        f"source /opt/ros/humble/setup.bash && "
        f"source {ros2_ws}/install/setup.bash && "
        f"ros2 run gp8_control terminal_debug; "
        f"echo; echo '[terminal_debug exited — press Enter to close]'; read"
    )
    terminal_debug = ExecuteProcess(
        cmd=[
            "gnome-terminal", "--title=GP8 Terminal Debug",
            "--", "bash", "-c", terminal_cmd,
        ],
        output="log",
        condition=IfCondition(LaunchConfiguration("spawn_terminal")),
    )

    return LaunchDescription([
        spawn_terminal_arg,
        name_bridge,
        robot_state_publisher,
        moveit_launch,
        terminal_debug,
    ])

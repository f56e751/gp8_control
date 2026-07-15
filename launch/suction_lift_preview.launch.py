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
    release_distance = LaunchConfiguration("release_distance")
    release_z_offset = LaunchConfiguration("release_z_offset")
    tool_offset = LaunchConfiguration("tool_offset")
    vel_scale = LaunchConfiguration("vel_scale")
    axis_increment_factor = LaunchConfiguration("axis_increment_factor")
    axis_acceleration_factor = LaunchConfiguration("axis_acceleration_factor")
    preview_rate = LaunchConfiguration("preview_rate")
    preview_speed = LaunchConfiguration("preview_speed")
    preview_domain_id = LaunchConfiguration("preview_domain_id")
    rviz_enabled = LaunchConfiguration("rviz")

    args = [
        DeclareLaunchArgument("x", default_value="0.55",
                              description="pick/grasp X [m]"),
        DeclareLaunchArgument("y", default_value="0.0",
                              description="pick/grasp Y [m]"),
        DeclareLaunchArgument("lift", default_value="0.10",
                              description="lift height after suction [m]"),
        DeclareLaunchArgument("bin_x", default_value="1.5",
                              description="target bin X [m]"),
        DeclareLaunchArgument("bin_y", default_value="0.0",
                              description="target bin Y [m]"),
        DeclareLaunchArgument("bin_z_offset", default_value="0.10",
                              description="target bin height offset from grasp z [m]"),
        DeclareLaunchArgument("release_distance", default_value="0.10",
                              description="preferred release distance along pick-to-bin [m]"),
        DeclareLaunchArgument("release_z_offset", default_value="0.33",
                              description="release Z offset from pick Z [m]"),
        DeclareLaunchArgument("tool_offset", default_value="0.0",
                              description="extra offset beyond gp8.py/MuJoCo TCP [m]"),
        DeclareLaunchArgument("vel_scale", default_value="0.3",
                              description="low-speed move velocity scale"),
        DeclareLaunchArgument(
            "axis_increment_factor", default_value="1.0",
            description="YRC external-increment velocity factor used for planning",
        ),
        DeclareLaunchArgument(
            "axis_acceleration_factor", default_value="0.02",
            description="YRC external-increment acceleration factor used for planning",
        ),
        DeclareLaunchArgument("preview_rate", default_value="60.0",
                              description="preview /joint_states rate [Hz]"),
        DeclareLaunchArgument("preview_speed", default_value="0.25",
                              description="preview playback speed (1.0=real time)"),
        DeclareLaunchArgument(
            "preview_domain_id", default_value="42",
            description="isolated ROS_DOMAIN_ID for preview nodes",
        ),
        DeclareLaunchArgument("rviz", default_value="true",
                              description="start RViz"),
    ]

    # 실제 bringup의 joint_state_broadcaster/robot_state_publisher가 같은
    # /joint_states와 /tf를 발행 중이어도 preview 화면을 덮어쓰지 못하게 한다.
    # 이 launch에서 시작하는 preview, RSP, RViz 세 프로세스만 domain 42에서
    # 통신한다.
    set_preview_domain = SetEnvironmentVariable(
        "ROS_DOMAIN_ID", preview_domain_id,
    )

    # Let the source tree version run before a colcon rebuild during iteration.
    ros2_ws_src = os.path.realpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
    )
    set_pythonpath = SetEnvironmentVariable(
        "PYTHONPATH", ros2_ws_src + ":" + os.environ.get("PYTHONPATH", ""),
    )
    set_python_unbuffered = SetEnvironmentVariable("PYTHONUNBUFFERED", "1")

    xacro_path = os.path.join(
        get_package_share_directory("gp8_control"),
        "urdf",
        "gp8_mujoco_suction_tool.xacro",
    )
    robot_description = subprocess.check_output(["xacro", xacro_path], text=True)

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="suction_lift_preview_state_publisher",
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
        "--release-distance", release_distance,
        "--release-z-offset", release_z_offset,
        "--tool-offset", tool_offset,
        "--vel-scale", vel_scale,
        "--axis-increment-factor", axis_increment_factor,
        "--axis-acceleration-factor", axis_acceleration_factor,
        "--preview-rate", preview_rate,
        "--preview-speed", preview_speed,
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
        name="suction_lift_preview_rviz",
        arguments=["-d", PathJoinSubstitution([
            FindPackageShare("gp8_control"), "rviz", "suction_lift_preview.rviz",
        ])],
        condition=IfCondition(rviz_enabled),
        output="screen",
    )

    return LaunchDescription([
        *args,
        set_preview_domain,
        set_pythonpath,
        set_python_unbuffered,
        robot_state_publisher,
        preview_node,
        rviz,
    ])

"""GP8 debug launch file (ROS 2) — adv4ncr hardened RT driver.

`gp8_bringup.launch.py` MINUS the app: brings up the IDENTICAL robot stack the
app runs on, so interactive debug scripts (suction_lift_debug, terminal_debug,
queue_test, ...) run against exactly the same environment — then run the debug
script in a second terminal instead of `gp8_manager`.

Launches (same pieces, same defaults as gp8_bringup):
  1. adv4ncr ros2_control stack (ros2_control_node + motoman_hardware +
     joint_state_broadcaster + JointGroupPositionController ACTIVE) +
     robot_state_publisher — via motoman_bringup/gp8.launch.py.
  2. joint_trajectory_controller spawned INACTIVE — its FollowJointTrajectory
     action server must exist for TrajectoryController.wait_for_servers()
     even in the default "stream" backend; inactive so it never claims the
     command interface from the active JGPC.
  3. MoveIt 2 (move_group).

No camera, no perception, no app. Requires the motoman_ROS2 workspace overlaid
(its install/setup.bash sourced) so motoman_bringup resolves — same requirement
as gp8_bringup. Pendant in REMOTE with no alarms; suction IO goes over TCP
50242 directly.

Usage:
  ros2 launch gp8_control debug_robot.launch.py
  # low commissioning speed:
  ros2 launch gp8_control debug_robot.launch.py \\
      axis_increment_factor:=0.1 axis_acceleration_factor:=0.01
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    # Same launch arguments + defaults as gp8_bringup.launch.py (keep in sync).
    robot_ip_arg = DeclareLaunchArgument(
        "robot_ip", default_value="192.168.255.1",
        description="Yaskawa controller IP address",
    )
    inc_factor_arg = DeclareLaunchArgument(
        "axis_increment_factor", default_value="1.0",
        description="Per-cycle increment (velocity) factor [0..1]. Same default "
                    "as gp8_bringup; pass 0.1 for LOW commissioning speed.",
    )
    acc_factor_arg = DeclareLaunchArgument(
        "axis_acceleration_factor", default_value="0.02",
        description="Per-cycle acceleration factor [0..1] (hardware-validated max).",
    )

    # =====================================================================
    # 1. adv4ncr driver stack (ros2_control) + robot_state_publisher —
    #    the VALIDATED motoman_bringup/gp8.launch.py, exactly as gp8_bringup
    #    includes it.
    # =====================================================================
    adv4ncr_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("motoman_bringup"), "launch", "gp8.launch.py",
            ])
        ]),
        launch_arguments={
            "robot_ip": LaunchConfiguration("robot_ip"),
            "axis_increment_factor": LaunchConfiguration("axis_increment_factor"),
            "axis_acceleration_factor": LaunchConfiguration("axis_acceleration_factor"),
        }.items(),
    )

    # =====================================================================
    # 2. joint_trajectory_controller INACTIVE (see gp8_bringup §2b: action
    #    server up for wait_for_servers, command interface unclaimed).
    # =====================================================================
    jtc_spawner_inactive = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_trajectory_controller", "--inactive",
                   "--controller-manager", "/controller_manager"],
        output="screen",
    )

    # =====================================================================
    # 3. MoveIt 2 — /joint_states straight from joint_state_broadcaster.
    # =====================================================================
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("motoman_gp8_moveit_config"),
                "launch", "move_group.launch.py",
            ])
        ]),
    )

    return LaunchDescription([
        robot_ip_arg,
        inc_factor_arg,
        acc_factor_arg,
        adv4ncr_stack,
        jtc_spawner_inactive,
        moveit_launch,
    ])

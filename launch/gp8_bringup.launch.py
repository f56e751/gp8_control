"""GP8 full system bringup launch file (ROS 2).

Launches all nodes needed for GP8 pick-and-throw operation:
  1. Robot state publisher (URDF -> TF)
  2. MoveIt 2
  3. RealSense camera
  4. Camera calibration publisher
  5. Image rectification
  6. AprilTag detector
  7. Main control app

MotoROS2 runs on the robot controller firmware — no node needed here.

Usage:
  ros2 launch gp8_control gp8_bringup.launch.py
"""

import os
import subprocess
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    GroupAction,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetRemap
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # Package path for config files
    # After colcon build: configs are in share/gp8_control/config/
    # During development with symlink: fall back to source tree
    try:
        pkg_share = get_package_share_directory("gp8_control")
        config_dir = os.path.join(pkg_share, "config")
    except Exception:
        pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_dir = os.path.join(pkg_dir, "config")

    # =====================================================================
    # Environment
    # =====================================================================
    # Since gp8_control is now standalone under ros2_ws/src and no longer
    # imports gp8_sorting, the only extra PYTHONPATH entry we need is
    # ros2_ws/src itself (so `.venv/bin/python -m gp8_control.app` can find
    # the source tree without colcon-install first). Resolved from this
    # launch file's own location for robustness.
    _ros2_ws_src = os.path.realpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
    )
    set_pythonpath = SetEnvironmentVariable(
        "PYTHONPATH",
        _ros2_ws_src + ":" + os.environ.get("PYTHONPATH", ""),
    )

    # =====================================================================
    # Launch arguments
    # =====================================================================
    robot_ip_arg = DeclareLaunchArgument(
        "robot_ip", default_value="192.168.255.1",
        description="Yaskawa controller IP address",
    )
    camera_info_path_arg = DeclareLaunchArgument(
        "camera_info_path",
        default_value=os.path.join(config_dir, "realsense_camera_info.yaml"),
        description="Path to camera calibration YAML",
    )

    camera_info_path = LaunchConfiguration("camera_info_path")

    # =====================================================================
    # 1. Robot model (URDF -> TF)
    # =====================================================================
    xacro_path = os.path.join(
        get_package_share_directory("motoman_gp8_support"),
        "urdf", "gp8.xacro",
    )
    robot_description = subprocess.check_output(
        ["xacro", xacro_path], text=True,
    )

    # Load SRDF for MoveItPy (used by gp8_app's MoveItController)
    moveit_config_share = get_package_share_directory("motoman_gp8_moveit_config")
    with open(os.path.join(moveit_config_share, "config", "gp8.srdf"), "r") as f:
        robot_description_semantic = f.read()

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        parameters=[{
            "robot_description": robot_description,
        }],
        remappings=[("joint_states", "joint_states_urdf")],
        output="screen",
    )

    # =====================================================================
    # 2a. MotoROS2 (robot driver)
    #     Runs ON the robot controller, not as a ROS 2 node.
    #     Provides: /joint_states, /follow_joint_trajectory,
    #               /write_single_io, /read_single_io
    # =====================================================================

    # =====================================================================
    # 2b. Name bridge — translates MotoROS2 raw joint names (joint_1..6) to
    #     URDF S/L/U/R/B/T convention (joint_1_s..6_t) on /joint_states_urdf,
    #     and proxies /motoman_gp8_controller/follow_joint_trajectory.
    # =====================================================================
    name_bridge = Node(
        package="gp8_control",
        executable="name_bridge",
        name="motoros2_name_bridge",
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
    # 4. RealSense camera
    # =====================================================================
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("realsense2_camera"),
                "launch", "rs_launch.py",
            ])
        ]),
        launch_arguments={
            "align_depth.enable": "true",
            "pointcloud.enable": "true",
            # Default ROS 2 realsense driver publishes under
            # {camera_namespace}/{camera_name}/... which gives /camera/camera/...
            # All downstream consumers (apriltag, camera_info_publisher,
            # sam_client) expect /camera/color/... so collapse the namespace.
            "camera_namespace": "",
        }.items(),
    )

    # =====================================================================
    # 5. Camera calibration publisher
    # =====================================================================
    camera_info_publisher = Node(
        package="gp8_control",
        executable="camera_info_publisher",
        name="camera_info_publisher",
        namespace="camera/color",
        parameters=[{
            "camera_info_path": camera_info_path,
            "sync_topic": "image_raw",
        }],
        output="screen",
    )

    # =====================================================================
    # 6. Image rectification — not wired up.
    # ---------------------------------------------------------------------
    # The main_sam7 pipeline does all perception on the remote SAM GPU
    # server using raw images (/camera/color/image_raw) and AprilTag is not
    # used at runtime, so local rectification would just produce an unused
    # topic and spam "topics not synchronized" warnings every second. Keep
    # this note as a reminder in case someone re-enables it; if you do,
    # remember to also remap `image` → `image_raw` for image_proc.
    # =====================================================================

    # =====================================================================
    # 7. AprilTag detector
    # =====================================================================
    apriltag_node = Node(
        package="apriltag_ros",
        executable="apriltag_node",
        name="apriltag_node",
        remappings=[
            ("image_rect", "/camera/color/image_raw"),
            ("camera_info", "/camera/color/camera_info"),
        ],
        parameters=[
            os.path.join(config_dir, "apriltag_settings.yaml"),
        ],
        output="screen",
    )

    # =====================================================================
    # 8. Main control app
    # ---------------------------------------------------------------------
    # The app imports `torch` (via gp8_control.trajectory.predictor). System
    # python doesn't have torch, so launch it with the repo's uv-managed
    # venv python rather than the installed script (whose shebang points
    # at /usr/bin/python3).
    #
    # Search strategy (first match wins):
    #   1. GP8_VENV_PYTHON environment variable (explicit override)
    #   2. Walk parents of this launch file until we find a `.venv/bin/python`
    #      (works when the package is built with --symlink-install)
    #   3. Standard developer path `~/Documents/Github/iitp_robot_control/.venv`
    # =====================================================================
    def _resolve_venv_python() -> str:
        override = os.environ.get("GP8_VENV_PYTHON")
        if override:
            return override

        launch_dir = os.path.dirname(os.path.realpath(os.path.abspath(__file__)))
        d = launch_dir
        for _ in range(10):
            candidate = os.path.join(d, ".venv", "bin", "python")
            if os.path.isfile(candidate):
                return candidate
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent

        fallback = os.path.expanduser(
            "~/Documents/Github/iitp_robot_control/.venv/bin/python"
        )
        if os.path.isfile(fallback):
            return fallback

        raise RuntimeError(
            "Could not locate .venv/bin/python. Run `uv sync` in the repo "
            "root or set GP8_VENV_PYTHON to your venv's python."
        )

    _venv_python = _resolve_venv_python()

    # Point PYTHONPATH at ros2_ws/src so `gp8_control` resolves to the live
    # source tree (no dependency on `iitp_robot_control` anymore).
    gp8_app = ExecuteProcess(
        cmd=[_venv_python, "-m", "gp8_control.app"],
        output="screen",
        additional_env={
            "PYTHONPATH": _ros2_ws_src + ":" + os.environ.get("PYTHONPATH", ""),
        },
    )

    # =====================================================================
    # Assemble
    # =====================================================================
    return LaunchDescription([
        set_pythonpath,
        robot_ip_arg,
        camera_info_path_arg,
        name_bridge,
        robot_state_publisher,
        moveit_launch,
        realsense_launch,
        camera_info_publisher,
        apriltag_node,
        gp8_app,
    ])

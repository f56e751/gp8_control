"""GP8 full system bringup launch file (ROS 2).

Launches all nodes needed for GP8 pick-and-throw operation:
  1. Robot state publisher (URDF -> TF)
  2. MoveIt 2
  3. Main control app

Perception runs entirely on the camera PC, which exposes an HTTP NDJSON
detection stream. The robot PC consumes it via the app's
StreamDetectionSource (set GP8_PERCEPTION_URL), so no local RealSense /
camera-calibration / AprilTag nodes are launched here.

MotoROS2 runs on the robot controller firmware — no node needed here.

Usage:
  ros2 launch gp8_control gp8_bringup.launch.py
"""

import os
import subprocess
from typing import Dict

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    GroupAction,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node, SetRemap
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def _load_dotenv(path: str) -> Dict[str, str]:
    """Minimal `.env` parser: KEY=VALUE lines, `#` comments, blank lines.

    No interpolation, no export syntax, no multi-line values. Quotes around
    the value are stripped. Missing file is silently OK — returns {}.
    """
    out: Dict[str, str] = {}
    if not os.path.isfile(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def generate_launch_description():

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
    # Perception (RealSense camera, camera-info publisher, AprilTag) runs on
    # the camera PC, which serves detections over HTTP. Nothing to launch on
    # the robot PC; the app connects to GP8_PERCEPTION_URL on its own.
    # =====================================================================

    # =====================================================================
    # 3. Main control app
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
    #   3. Known source-tree locations (this package and the legacy
    #      iitp_robot_control layout, just in case).
    # =====================================================================
    def _resolve_venv_python() -> str:
        override = os.environ.get("GP8_VENV_PYTHON")
        if override:
            return override

        # 1. Walk up from the real location of this launch file. When colcon
        #    is built with --symlink-install the install tree points back at
        #    the package source, so this reaches
        #    ros2_ws/src/gp8_control/.venv directly.
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

        # 2. Known standalone package source tree (when colcon copy-build
        #    puts __file__ inside install/share and the walk-up above can't
        #    escape back to the repo).
        for fallback in (
            os.path.expanduser("~/ros2_ws/src/gp8_control/.venv/bin/python"),
            os.path.expanduser("~/Documents/Github/iitp_robot_control/.venv/bin/python"),
        ):
            if os.path.isfile(fallback):
                return fallback

        raise RuntimeError(
            "Could not locate .venv/bin/python. Run `uv sync` inside "
            "ros2_ws/src/gp8_control or set GP8_VENV_PYTHON to a python "
            "with torch installed."
        )

    _venv_python = _resolve_venv_python()

    # Load .env from the package source tree (gitignored) so the perception
    # stream URL and other infra-specific overrides stay out of the public
    # repo. Existing process env wins — set GP8_PERCEPTION_URL before
    # `ros2 launch` to override whatever is in the file.
    dotenv_vars = _load_dotenv(os.path.join(_ros2_ws_src, "gp8_control", ".env"))
    if dotenv_vars:
        print(
            f"[gp8_bringup] loaded {len(dotenv_vars)} var(s) from .env: "
            f"{list(dotenv_vars.keys())}"
        )

    app_env = {
        # Point PYTHONPATH at ros2_ws/src so `gp8_control` resolves to the
        # live source tree (no dependency on `iitp_robot_control` anymore).
        "PYTHONPATH": _ros2_ws_src + ":" + os.environ.get("PYTHONPATH", ""),
    }
    for k, v in dotenv_vars.items():
        # Don't clobber an already-set value from the launching shell — that
        # way `GP8_PERCEPTION_URL=... ros2 launch ...` still takes priority.
        if k not in os.environ:
            app_env[k] = v

    gp8_app = ExecuteProcess(
        cmd=[_venv_python, "-m", "gp8_control.app"],
        output="screen",
        additional_env={
            **app_env,
        },
    )

    # =====================================================================
    # Assemble
    # =====================================================================
    return LaunchDescription([
        set_pythonpath,
        robot_ip_arg,
        name_bridge,
        robot_state_publisher,
        moveit_launch,
        gp8_app,
    ])

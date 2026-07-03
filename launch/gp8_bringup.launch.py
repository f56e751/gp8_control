"""GP8 full system bringup launch file (ROS 2) — adv4ncr hardened RT driver.

Launches all nodes needed for GP8 pick-and-throw operation:
  1. adv4ncr ros2_control stack (ros2_control_node + motoman_hardware +
     joint_state_broadcaster + JointGroupPositionController) + robot_state
     publisher — via motoman_bringup/gp8.launch.py (the validated bringup).
  2. MoveIt 2
  3. Main control app

Perception runs entirely on the camera PC, which exposes an HTTP NDJSON
detection stream. The robot PC consumes it via the app's
StreamDetectionSource (set GP8_PERCEPTION_URL), so no local RealSense /
camera-calibration / AprilTag nodes are launched here.

**Driver: the hardened adv4ncr RT driver (NOT MotoROS2).** The app's
TrajectoryController already targets it (default backend "stream" ->
/JointGroupPositionController/commands at 250 Hz; /joint_states from
joint_state_broadcaster; suction via the Simple Message IoServer on TCP
50242). This launch therefore brings up that ros2_control stack instead of
the old MotoROS2 name_bridge. Requires the motoman_ROS2 workspace overlaid
(both install/setup.bash sourced) so motoman_bringup/motoman_hardware/
motoman_description resolve.

Speed factors default to the LOW commissioning values (0.1 / 0.01); raise
only through the DEPLOYMENT.md §4.4 ramp after the drills pass:
  ros2 launch gp8_control gp8_bringup.launch.py \\
      axis_increment_factor:=0.35 axis_acceleration_factor:=0.02

Usage:
  ros2 launch gp8_control gp8_bringup.launch.py
"""

import os
from typing import Dict

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


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
    # NOTE: app default set to FULL speed (factor 1.0) per operator request. The
    # 1.0 velocity ceiling + comm-loss stopping (3.4 cm EE @ ~1 m/s) + throw/push
    # motion paths were validated this session, BUT the graduated §4.3 drill set
    # (E-stop mid-move + comm-loss) was NOT run at every step 0.5/0.75/1.0, and
    # suction (TCP 50242) + throw release-timing calibration are still pending.
    # Override to go slow:  axis_increment_factor:=0.1 axis_acceleration_factor:=0.01
    inc_factor_arg = DeclareLaunchArgument(
        "axis_increment_factor", default_value="1.0",
        description="Per-cycle increment (velocity) factor [0..1]. Default FULL speed (1.0) "
                    "per operator request; pass 0.1 to fall back to LOW commissioning speed.",
    )
    acc_factor_arg = DeclareLaunchArgument(
        "axis_acceleration_factor", default_value="0.02",
        description="Per-cycle acceleration factor [0..1]. 0.02 = hardware-validated max "
                    "(sync-fix holds); throw needs only ~0.005, so this is ample headroom.",
    )

    # Robot model (URDF -> TF), robot_description, and SRDF are now provided by
    # the included adv4ncr stack's robot_state_publisher and by move_group
    # respectively — no standalone robot_state_publisher / xacro call here.

    # =====================================================================
    # 2. adv4ncr driver stack (ros2_control) + robot_state_publisher
    #    Brings up: ros2_control_node (motoman_hardware, RT UDP path),
    #    joint_state_broadcaster (-> /joint_states with S/L/U/R/B/T names, so
    #    NO name_bridge needed), JointGroupPositionController (active, the app's
    #    default "stream" backend), and robot_state_publisher (/joint_states ->
    #    TF). The controller must be in REMOTE with no alarm; suction goes over
    #    TCP 50242 directly (Simple Message IoServer), not a ROS service.
    #
    #    Reuses the VALIDATED motoman_bringup/gp8.launch.py so this file never
    #    re-implements the ros2_control wiring. Speed factors are forwarded.
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
    # 2b. joint_trajectory_controller spawned INACTIVE.
    #    The app's TrajectoryController.wait_for_servers() waits for the JTC
    #    FollowJointTrajectory action server even in the default "stream"
    #    backend (which actually drives via the ACTIVE JointGroupPositionController
    #    at 250 Hz). A configured-but-inactive JTC creates its action server
    #    (on_configure) so wait_for_servers passes, WITHOUT claiming the position
    #    command interface — so it never conflicts with the active JGPC. It stays
    #    inactive because the stream backend never sends it a goal.
    #    (For GP8_ADV4NCR_BACKEND=jtc, activate it instead of JGPC — they can't
    #    both be active.)
    # =====================================================================
    jtc_spawner_inactive = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_trajectory_controller", "--inactive",
                   "--controller-manager", "/controller_manager"],
        output="screen",
    )

    # =====================================================================
    # 3. MoveIt 2 — reads /joint_states directly from joint_state_broadcaster
    #    (the adv4ncr broadcaster already publishes S/L/U/R/B/T names, so the
    #    old /joint_states_urdf name_bridge remap is gone).
    # =====================================================================
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("motoman_gp8_moveit_config"),
                "launch", "move_group.launch.py",
            ])
        ]),
    )

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
        inc_factor_arg,
        acc_factor_arg,
        adv4ncr_stack,
        jtc_spawner_inactive,
        moveit_launch,
        gp8_app,
    ])

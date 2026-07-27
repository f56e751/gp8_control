"""GP8 software-in-the-loop (SIL) simulation bringup (ROS 2).

Runs the FULL app pipeline (detection -> intake/dedup -> selection ->
push/throw routing -> skill -> motion) with NO hardware, by swapping in two
fakes:

  1. mock_robot   — fakes MotoROS2 incl. Point Queue Mode; plays queued points
                    back in real time so the app's wall-clock timing holds.
  2. fake_belt    — fakes camera_debug: spawns objects on the belt and
                    publishes /camera_debug/detections + /conveyor/speed.

Plus the real robot_state_publisher + move_group + RViz for visualization, and
the real app (unchanged). Watch the arm intercept belt objects in RViz.

  ros2 launch gp8_control sim_bringup.launch.py
  ros2 launch gp8_control sim_bringup.launch.py belt_speed:=0.08 spawn_interval:=4.0
  GP8_FORCE_SKILL=throw ros2 launch gp8_control sim_bringup.launch.py   # one skill

Requires the same packages as debug_robot.launch.py (motoman_gp8_support,
motoman_gp8_moveit_config) and the torch venv (see gp8_bringup).
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
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetRemap
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def _load_dotenv(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not os.path.isfile(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            out[key.strip()] = value.strip().strip('"').strip("'")
    return out


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
    for fallback in (
        os.path.expanduser("~/ros2_ws/src/gp8_control/.venv/bin/python"),
        os.path.expanduser("~/Documents/Github/iitp_robot_control/.venv/bin/python"),
    ):
        if os.path.isfile(fallback):
            return fallback
    raise RuntimeError(
        "Could not locate .venv/bin/python. Run `uv sync` inside "
        "ros2_ws/src/gp8_control or set GP8_VENV_PYTHON."
    )


def generate_launch_description():
    # ---- launch args --------------------------------------------------
    belt_speed = LaunchConfiguration("belt_speed")
    spawn_interval = LaunchConfiguration("spawn_interval")
    lane_x = LaunchConfiguration("lane_x")
    use_rviz = LaunchConfiguration("rviz")
    args = [
        DeclareLaunchArgument("belt_speed", default_value="0.12",
                              description="Belt speed (m/s)"),
        DeclareLaunchArgument("spawn_interval", default_value="5.0",
                              description="Seconds between spawned objects"),
        DeclareLaunchArgument("lane_x", default_value="0.45",
                              description="Belt lane X (m); low values "
                                          "exercise the near-base push veto"),
        DeclareLaunchArgument("rviz", default_value="true",
                              description="Launch RViz"),
    ]

    # ---- environment / PYTHONPATH (so `-m gp8_control.app` resolves) --
    _ros2_ws_src = os.path.realpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
    )
    set_pythonpath = SetEnvironmentVariable(
        "PYTHONPATH", _ros2_ws_src + ":" + os.environ.get("PYTHONPATH", ""),
    )

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

    # ---- fakes --------------------------------------------------------
    mock_robot = Node(
        package="gp8_control", executable="mock_robot", name="mock_robot",
        output="screen",
    )
    fake_belt = Node(
        package="gp8_control", executable="fake_belt", name="fake_belt",
        parameters=[{"belt_speed": belt_speed, "spawn_interval": spawn_interval,
                     "lane_x": lane_x}],
        output="screen",
    )

    # ---- MoveIt move_group (only so MoveItController's wait_for_server
    #      returns immediately; the app never plans through it) ----------
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

    # ---- RViz ---------------------------------------------------------
    rviz = Node(
        package="rviz2", executable="rviz2", name="rviz2",
        arguments=["-d", PathJoinSubstitution([
            FindPackageShare("gp8_control"), "rviz", "sim.rviz",
        ])],
        condition=IfCondition(use_rviz),
        output="screen",
    )

    # ---- app (real, unchanged) — venv python for torch ----------------
    _venv_python = _resolve_venv_python()
    dotenv_vars = _load_dotenv(os.path.join(_ros2_ws_src, "gp8_control", ".env"))
    app_env = {"PYTHONPATH": _ros2_ws_src + ":" + os.environ.get("PYTHONPATH", "")}
    for k, v in dotenv_vars.items():
        if k not in os.environ:
            app_env[k] = v

    gp8_app = ExecuteProcess(
        cmd=[_venv_python, "-m", "gp8_control.app"],
        output="screen",
        additional_env=app_env,
    )

    return LaunchDescription([
        *args,
        set_pythonpath,
        robot_state_publisher,
        mock_robot,
        fake_belt,
        moveit_launch,
        rviz,
        gp8_app,
    ])

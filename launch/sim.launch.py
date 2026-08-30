"""In-process MuJoCo sim launch — the app with GP8_BACKEND=mujoco.

The simulator is NOT a separate node any more: ``backends/mujoco_sim.py``
runs inside the app process behind the same RobotBackend/WorldSource seams
the hardware uses, so this launch only starts the app (with the uv venv
python — it needs torch AND mujoco: ``uv sync --extra sim``) plus the
belt_viz TUI. No bridge, no ros2_control stack, no camera_debug.

Usage:
  ros2 launch gp8_control sim.launch.py                # headless twin
  ros2 launch gp8_control sim.launch.py viewer:=true   # MuJoCo viewer window
  ros2 launch gp8_control sim.launch.py skill:=throw   # pin one skill
  ros2 launch gp8_control sim.launch.py spawn_y:=0.9   # shorter belt (default:
                                                       #  camera ref 2.47 m)
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _resolve_venv_python() -> str:
    """Same strategy as gp8_bringup.launch.py: env override, walk-up, fallbacks."""
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
    fallback = os.path.expanduser("~/ros2_ws/src/gp8_control/.venv/bin/python")
    if os.path.isfile(fallback):
        return fallback
    raise RuntimeError(
        "Could not locate .venv/bin/python. Run `uv sync --extra sim` inside "
        "ros2_ws/src/gp8_control or set GP8_VENV_PYTHON."
    )


def generate_launch_description() -> LaunchDescription:
    venv_python = _resolve_venv_python()
    # The SOURCE tree must win over the colcon-installed copy on PYTHONPATH:
    # the vendored MJCF (sim/recycling_mujoco/) is deliberately NOT installed
    # (setup.py), so `gp8_control` resolved from install/ can't find scene.xml.
    # The repo root is wherever the venv walk found .venv (its parent dir is
    # ros2_ws/src, the entry PYTHONPATH needs).
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(venv_python)))
    src_dir = os.path.dirname(repo_root)
    pythonpath = src_dir + ":" + os.environ.get("PYTHONPATH", "")

    skill_arg = DeclareLaunchArgument(
        "skill", default_value=os.environ.get("GP8_FORCE_SKILL", ""),
        description="Force-skill for this run ('' = normal routing).",
    )
    viewer_arg = DeclareLaunchArgument(
        "viewer", default_value="false",
        description="Open the MuJoCo passive viewer window (needs a DISPLAY).",
    )
    belt_speed_arg = DeclareLaunchArgument(
        "belt_speed", default_value=os.environ.get("GP8_SIM_BELT_SPEED", "0.12"),
        description="Sim conveyor speed [m/s].",
    )
    spawn_interval_arg = DeclareLaunchArgument(
        "spawn_interval", default_value=os.environ.get("GP8_SIM_SPAWN_INTERVAL", "5.0"),
        description="Sim box spawn interval [s].",
    )
    throw_bins_arg = DeclareLaunchArgument(
        "throw_bins", default_value=os.environ.get("GP8_THROW_BINS", ""),
        description=(
            "Optional JSON throw-bin list (same format as gp8_bringup): "
            "[{\"name\":\"left\",\"x\":1.2,\"y\":0.3,\"z\":0.08,\"radius\":0.1}]. "
            "The twin places a bin at each entry; '' = one bin at throw_goal_x/y."
        ),
    )
    spawn_y_arg = DeclareLaunchArgument(
        "spawn_y", default_value=os.environ.get("GP8_SIM_SPAWN_Y", ""),
        description=(
            "Base-frame Y where sim boxes enter the belt [m]. '' = the real "
            "camera reference point (extrinsics.REFERENCE_Y_BASE, 2.47 m), so "
            "the detection->pick lead time matches hardware."
        ),
    )

    gp8_app = ExecuteProcess(
        cmd=[venv_python, "-m", "gp8_control.app"],
        output="screen",
        additional_env={
            "PYTHONPATH": pythonpath,
            "GP8_BACKEND": "mujoco",
            "GP8_FORCE_SKILL": LaunchConfiguration("skill"),
            "GP8_SIM_VIEWER": LaunchConfiguration("viewer"),
            "GP8_SIM_BELT_SPEED": LaunchConfiguration("belt_speed"),
            "GP8_SIM_SPAWN_INTERVAL": LaunchConfiguration("spawn_interval"),
            "GP8_SIM_SPAWN_Y": LaunchConfiguration("spawn_y"),
            "GP8_THROW_BINS": LaunchConfiguration("throw_bins"),
        },
    )

    belt_viz = Node(
        package="gp8_control",
        executable="belt_viz",
        name="belt_viz",
        output="screen",
    )

    return LaunchDescription([
        skill_arg, viewer_arg, belt_speed_arg, spawn_interval_arg, spawn_y_arg,
        throw_bins_arg, gp8_app, belt_viz,
    ])

"""GP8 Level B simulation bringup — MuJoCo digital twin (ROS 2).

Same full app pipeline as ``sim_bringup.launch.py`` (detection -> intake ->
selection -> push/throw routing -> skill -> motion) with no hardware, but the
kinematic ``mock_robot`` is swapped for ``mujoco_robot``: a MuJoCo-backed twin
that speaks the *identical* ROS contract (Point Queue Mode, FJT,
/joint_states_urdf, /write_single_io, ...) and renders the GP8 executing the
app's commands live, on the real robot meshes, over the belt, intercepting the
objects perception reports. You watch the robot in the MuJoCo window instead of
(or alongside) RViz.

  ros2 launch gp8_control sim_mujoco.launch.py
  ros2 launch gp8_control sim_mujoco.launch.py headless:=true      # no window; /mujoco/image
  ros2 launch gp8_control sim_mujoco.launch.py physics:=true       # real dynamics + contact
  ros2 launch gp8_control sim_mujoco.launch.py belt_speed:=0.08 spawn_interval:=4.0
  GP8_FORCE_SKILL=throw ros2 launch gp8_control sim_mujoco.launch.py

The twin needs `mujoco` in the gp8_control uv venv (same venv that runs the app
for torch):  uv pip install --python <venv> 'mujoco>=3.1'. It is launched with
that venv python (system python has no mujoco), exactly like the app.

Requires the same support/moveit packages as debug_robot.launch.py.
"""

import os
import subprocess
import sys
from typing import Dict, List

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
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


_TRUE_TOKENS = ("1", "true", "yes", "on")
_FALSE_TOKENS = ("0", "false", "no", "off", "")


def _resolve_bool(context, name: str) -> bool:
    """Resolve a boolean launch arg, but LOUDLY warn on an unrecognized value
    (e.g. a `physics:=ture` typo) instead of silently treating it as false and
    running the wrong mode."""
    raw = LaunchConfiguration(name).perform(context)
    low = raw.strip().lower()
    if low in _TRUE_TOKENS:
        return True
    if low in _FALSE_TOKENS:
        return False
    sys.stderr.write(
        f"\n*** sim_mujoco.launch: arg '{name}:={raw}' is not a recognized "
        f"boolean (true/false) — treating as FALSE. Did you mean "
        f"'{name}:=true'? ***\n\n"
    )
    return False


def _setup(context, *_args, **_kwargs) -> List:
    belt_speed = LaunchConfiguration("belt_speed")
    spawn_interval = LaunchConfiguration("spawn_interval")
    use_rviz = LaunchConfiguration("rviz")

    headless = _resolve_bool(context, "headless")
    physics = _resolve_bool(context, "physics")
    objects = _resolve_bool(context, "objects")

    ros2_ws_src = os.path.realpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
    )
    pythonpath = ros2_ws_src + ":" + os.environ.get("PYTHONPATH", "")

    # robot model (URDF -> TF) so RViz/move_group have a description
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

    # --- the MuJoCo twin: run under the venv python (system python has no mujoco),
    #     same way the app is launched. Build the CLI flags from the args. ---
    venv_python = _resolve_venv_python()
    # The twin needs the *source* tree on PYTHONPATH because sim/recycling_mujoco/
    # (scene.xml + meshes) is deliberately NOT colcon-installed (see VENDOR.md), so
    # `-m gp8_control` must resolve to source, not the install copy. This launch
    # file, once installed, sits under install/.../share — so derive the source
    # root from the venv instead (it lives at <src>/gp8_control/.venv/bin/python;
    # do NOT realpath — bin/python is a symlink to the system interpreter).
    twin_src = os.path.abspath(os.path.join(venv_python, *([os.pardir] * 4)))
    if not os.path.isfile(os.path.join(
        twin_src, "gp8_control", "sim", "recycling_mujoco", "scene.xml")
    ):
        twin_src = os.path.expanduser("~/ros2_ws/src")
    twin_cmd = [venv_python, "-m", "gp8_control.mock.mujoco_robot"]
    if physics:
        twin_cmd.append("--physics")
    if headless:
        twin_cmd.append("--headless")
    if not objects:
        twin_cmd.append("--no-objects")
    if physics:
        # the twin owns the conveyor in physics mode — give it the belt params.
        # --ros-args must come LAST (it consumes every following token).
        twin_cmd += [
            "--ros-args",
            "-p", "belt_speed:=" + belt_speed.perform(context),
            "-p", "spawn_interval:=" + spawn_interval.perform(context),
        ]
    twin_env = {"PYTHONPATH": twin_src + ":" + os.environ.get("PYTHONPATH", "")}
    if headless:
        twin_env.setdefault("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))
    mujoco_robot = ExecuteProcess(
        cmd=twin_cmd, output="screen", additional_env=twin_env,
    )

    # In physics mode the twin OWNS the belt and publishes /camera_debug/detections
    # + /conveyor/speed itself (camera bridge), so fake_belt must NOT also run —
    # two publishers on those topics would fight. In kinematic mode fake_belt is
    # the perception source and the twin mirrors it.
    fake_belt = None
    if not physics:
        fake_belt = Node(
            package="gp8_control", executable="fake_belt", name="fake_belt",
            parameters=[{"belt_speed": belt_speed, "spawn_interval": spawn_interval}],
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

    rviz = Node(
        package="rviz2", executable="rviz2", name="rviz2",
        arguments=["-d", PathJoinSubstitution([
            FindPackageShare("gp8_control"), "rviz", "sim.rviz",
        ])],
        condition=IfCondition(use_rviz),
        output="screen",
    )

    # app (real, unchanged) — venv python for torch
    dotenv_vars = _load_dotenv(os.path.join(ros2_ws_src, "gp8_control", ".env"))
    app_env = {"PYTHONPATH": pythonpath}
    for k, v in dotenv_vars.items():
        if k not in os.environ:
            app_env[k] = v
    gp8_app = ExecuteProcess(
        cmd=[venv_python, "-m", "gp8_control.app"],
        output="screen",
        additional_env=app_env,
    )

    actions = [robot_state_publisher, mujoco_robot, moveit_launch, rviz, gp8_app]
    if fake_belt is not None:
        actions.append(fake_belt)
    return actions


def generate_launch_description():
    args = [
        DeclareLaunchArgument("belt_speed", default_value="0.12",
                              description="Belt speed (m/s)"),
        DeclareLaunchArgument("spawn_interval", default_value="5.0",
                              description="Seconds between spawned objects"),
        DeclareLaunchArgument("rviz", default_value="false",
                              description="Also launch RViz (MuJoCo is the main view)"),
        DeclareLaunchArgument("headless", default_value="false",
                              description="No MuJoCo window; render to /mujoco/image (SSH)"),
        DeclareLaunchArgument("physics", default_value="false",
                              description="Full dynamics+contact (default: kinematic mirror)"),
        DeclareLaunchArgument("objects", default_value="true",
                              description="Mirror /camera_debug/detections as belt boxes"),
    ]
    ros2_ws_src = os.path.realpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
    )
    set_pythonpath = SetEnvironmentVariable(
        "PYTHONPATH", ros2_ws_src + ":" + os.environ.get("PYTHONPATH", ""),
    )
    return LaunchDescription([*args, set_pythonpath, OpaqueFunction(function=_setup)])

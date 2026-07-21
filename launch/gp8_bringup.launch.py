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
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
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
    # Throw suction-RELEASE timing (config.py RELEASE_LEAD). Shifts the suction_off
    # waypoint vs the NN release point: POSITIVE = release EARLIER, NEGATIVE = LATER
    # (covers WriteSingleIO + vent lag). At TRAJ_HZ=20, ±0.1 s = ±2 waypoints. Pass
    # `release_lead:=-0.05` to tune WITHOUT a rebuild; when the arg is omitted a
    # shell-set GP8_RELEASE_LEAD is honored, else -0.1.
    release_lead_arg = DeclareLaunchArgument(
        "release_lead",
        default_value=EnvironmentVariable("GP8_RELEASE_LEAD", default_value="-0.1"),
        description="Throw suction-release lead [s]: +earlier / -later (RELEASE_LEAD).",
    )
    # Guaranteed parked vacuum-forming hold before the throw lift (config.py
    # MIN_SUCTION_HOLD). Places the grasp far enough downstream that the arm parks
    # >= this many seconds before the object arrives; objects that can't be caught
    # that far downstream are dropped. Pass `min_suction_hold:=0.3` to tune without a
    # rebuild; omitted -> shell GP8_MIN_SUCTION_HOLD, else 0.3.
    min_suction_hold_arg = DeclareLaunchArgument(
        "min_suction_hold",
        default_value=EnvironmentVariable("GP8_MIN_SUCTION_HOLD", default_value="0.3"),
        description="Guaranteed parked suction hold before throw lift [s] (MIN_SUCTION_HOLD).",
    )
    # Absolute base-frame TCP Z where throw/pick parks and primes suction.
    # `grasp_z:=0.0615` allows millimetre-level contact calibration without a
    # rebuild; omitted -> shell GP8_GRASP_Z, else Config default 0.062 m.
    grasp_z_arg = DeclareLaunchArgument(
        "grasp_z",
        default_value=EnvironmentVariable("GP8_GRASP_Z", default_value="0.062"),
        description="Throw/pick suction wait TCP Z in base frame [m] (GRASP_Z).",
    )
    # Throw pick belt-tracking descend (skills/throw_skill.py): the cup follows the
    # object downstream at belt speed while lowering from track_z_start to
    # track_z_end at track_z_speed. "nan" (default) derives the heights from grasp_z
    # (start = grasp_z + 0.05, end = grasp_z); track_z_speed:=0 disables tracking and
    # restores the old parked wait-at-grasp pick.
    # track_z_* / track_lead_t defaults below are the values that tracked best on
    # HW at belt 0.223 m/s. Pass "nan" to restore the derive-from-grasp_z heights.
    track_z_start_arg = DeclareLaunchArgument(
        "track_z_start",
        default_value=EnvironmentVariable("GP8_TRACK_Z_START", default_value="0.12"),
        description="Throw pick: TCP Z the descend starts from [m] (nan -> grasp_z + 0.05).",
    )
    track_z_end_arg = DeclareLaunchArgument(
        "track_z_end",
        default_value=EnvironmentVariable("GP8_TRACK_Z_END", default_value="0.03"),
        description="Throw pick: TCP Z the descend ends at [m] (nan -> grasp_z).",
    )
    track_z_speed_arg = DeclareLaunchArgument(
        "track_z_speed",
        default_value=EnvironmentVariable("GP8_TRACK_Z_SPEED", default_value="0.2"),
        description="Throw pick: descend rate while tracking the belt [m/s] (<=0 disables).",
    )
    track_lead_t_arg = DeclareLaunchArgument(
        "track_lead_t",
        default_value=EnvironmentVariable("GP8_TRACK_LEAD_T", default_value="0.3"),
        description="Throw pick: start the tracking descend this many s EARLIER to "
                    "cancel a downstream landing offset (~= miss[m]/belt[m/s]).",
    )
    # Pin every object to ONE manipulation skill for this run:
    # `skill:=throw|robust_throw|push`. Empty (default) = normal per-class
    # routing. robust_throw is the NLP (CasADi/IPOPT) thrower and needs casadi
    # in .venv; omitted -> shell GP8_FORCE_SKILL, else normal routing.
    skill_arg = DeclareLaunchArgument(
        "skill",
        default_value=EnvironmentVariable("GP8_FORCE_SKILL", default_value=""),
        description="Force ALL objects to one skill: throw|robust_throw|push (empty = class routing).",
    )
    rviz_arg = DeclareLaunchArgument(
        "rviz", default_value="false",
        description="Start RViz with live throw trajectory/release/ballistic markers.",
    )
    throw_viz_impact_z_arg = DeclareLaunchArgument(
        "throw_viz_impact_z",
        default_value=EnvironmentVariable("GP8_THROW_VIZ_IMPACT_Z", default_value="0.0"),
        description="Base-frame Z plane where the RViz ballistic preview lands [m].",
    )
    throw_goal_x_arg = DeclareLaunchArgument(
        "throw_goal_x",
        default_value=EnvironmentVariable("GP8_THROW_GOAL_X", default_value="1.1"),
        description="Throw evaluation goal/bin center X in base_link [m].",
    )
    throw_goal_y_arg = DeclareLaunchArgument(
        "throw_goal_y",
        default_value=EnvironmentVariable("GP8_THROW_GOAL_Y", default_value="-0.25"),
        description="Throw evaluation goal/bin center Y in base_link [m].",
    )
    throw_goal_radius_arg = DeclareLaunchArgument(
        "throw_goal_radius",
        default_value=EnvironmentVariable("GP8_THROW_GOAL_RADIUS", default_value="0.10"),
        description="Horizontal acceptance radius for predicted throw landing [m].",
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
        condition=IfCondition(LaunchConfiguration("moveit")),
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

    # `app:=false` — 드라이버 스택(adv4ncr + JTC inactive + MoveIt)만 띄우고
    # gp8_manager 앱은 생략. 정적 테스트(tests/static_pick_throw.py)나 디버그
    # 도구처럼 JointGroupPositionController에 직접 명령을 쓰는 프로세스는 앱과
    # 동시에 돌 수 없으므로 이 모드로 bringup 한다.
    app_arg = DeclareLaunchArgument(
        "app",
        default_value="true",
        description="Run the gp8_manager app (false = driver stack only, for tests).",
    )
    # MoveIt(move_group)도 선택화: 정적 테스트/디버그 도구는 JTC 액션 + 250Hz
    # 스트림 + 자체 IK만 쓰므로 move_group이 필요 없다 (그리고 move_group이
    # 죽어도 테스트에는 지장이 없다).
    moveit_arg = DeclareLaunchArgument(
        "moveit",
        default_value="true",
        description="Run MoveIt move_group (false = skip; tests don't need it).",
    )

    gp8_app = ExecuteProcess(
        cmd=[_venv_python, "-m", "gp8_control.app"],
        output="screen",
        condition=IfCondition(LaunchConfiguration("app")),
        additional_env={
            **app_env,
            # Throw suction-release lead: `release_lead:=` launch arg (falling back
            # to shell GP8_RELEASE_LEAD, then -0.1) -> GP8_RELEASE_LEAD for Config.
            "GP8_RELEASE_LEAD": LaunchConfiguration("release_lead"),
            # Guaranteed parked suction hold: `min_suction_hold:=` -> GP8_MIN_SUCTION_HOLD.
            "GP8_MIN_SUCTION_HOLD": LaunchConfiguration("min_suction_hold"),
            # Throw/pick suction wait height: `grasp_z:=` -> GP8_GRASP_Z.
            "GP8_GRASP_Z": LaunchConfiguration("grasp_z"),
            # Throw pick belt-tracking descend: `track_z_start/end/speed:=`.
            "GP8_TRACK_Z_START": LaunchConfiguration("track_z_start"),
            "GP8_TRACK_Z_END": LaunchConfiguration("track_z_end"),
            "GP8_TRACK_Z_SPEED": LaunchConfiguration("track_z_speed"),
            # Throw pick descend timing lead: `track_lead_t:=` -> GP8_TRACK_LEAD_T.
            "GP8_TRACK_LEAD_T": LaunchConfiguration("track_lead_t"),
            # Force-skill for this run: `skill:=` -> GP8_FORCE_SKILL ("" = routing).
            "GP8_FORCE_SKILL": LaunchConfiguration("skill"),
            # Visualization-only impact plane; never changes robot motion.
            "GP8_THROW_VIZ_IMPACT_Z": LaunchConfiguration("throw_viz_impact_z"),
            "GP8_THROW_GOAL_X": LaunchConfiguration("throw_goal_x"),
            "GP8_THROW_GOAL_Y": LaunchConfiguration("throw_goal_y"),
            "GP8_THROW_GOAL_RADIUS": LaunchConfiguration("throw_goal_radius"),
        },
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="gp8_throw_rviz",
        arguments=["-d", PathJoinSubstitution([
            FindPackageShare("gp8_control"), "rviz", "throw_runtime.rviz",
        ])],
        condition=IfCondition(LaunchConfiguration("rviz")),
        output="screen",
    )

    # =====================================================================
    # Assemble
    # =====================================================================
    return LaunchDescription([
        set_pythonpath,
        robot_ip_arg,
        inc_factor_arg,
        acc_factor_arg,
        release_lead_arg,
        min_suction_hold_arg,
        grasp_z_arg,
        track_z_start_arg,
        track_z_end_arg,
        track_z_speed_arg,
        track_lead_t_arg,
        skill_arg,
        app_arg,
        moveit_arg,
        rviz_arg,
        throw_viz_impact_z_arg,
        throw_goal_x_arg,
        throw_goal_y_arg,
        throw_goal_radius_arg,
        adv4ncr_stack,
        jtc_spawner_inactive,
        moveit_launch,
        gp8_app,
        rviz,
    ])

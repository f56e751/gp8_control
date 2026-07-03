# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`README.md` is the authoritative operator guide (launch order, pendant state,
debug tools, topology) and is largely in Korean. Read it for *how to run*; this
file covers *how the code is shaped* and the non-obvious cross-file contracts.

## What this is

ROS 2 (Humble) runtime control for a Yaskawa **GP8** 6-DOF arm on MotoROS2
firmware, doing conveyor **pick-and-throw**: a camera PC streams object
detections, the arm intercepts objects on a moving belt, suctions them, and
flings them to a target via a learned (torch FCN) throw trajectory.

## Build & run

```bash
# 1. Python venv with torch/scipy/numpy<2 (rclpy etc. come from system ROS, NOT here)
cd ~/ros2_ws/src/gp8_control && uv sync

# 2. Build the ROS 2 package
cd ~/ros2_ws && source /opt/ros/humble/setup.bash
colcon build --packages-select gp8_control
source ~/ros2_ws/install/setup.bash

# 3. Full bringup (bridge + robot_state_publisher + MoveIt + gp8_manager)
ros2 launch gp8_control gp8_bringup.launch.py

# Lightweight bringup for the debug tools (bridge + TF + MoveIt, no app)
ros2 launch gp8_control debug_robot.launch.py
```

Prereqs that must already be up: micro-ROS Agent (docker, once per boot),
MotoROS2 on the controller, pendant in **REMOTE + AUTO** with no alarms, and
`GP8_PERCEPTION_URL` pointing at the camera PC's NDJSON stream. Without the
camera stream the node still boots but picks nothing.

### torch lives in `.venv`, not system ROS

The app imports `torch` (the throw predictor), which the system Python doesn't
have. `gp8_bringup.launch.py` therefore runs the app as
`ExecuteProcess([<.venv>/bin/python, -m, gp8_control.app])` — **not** the
`gp8_app` console entry point (whose shebang is system Python). The launch file
locates the venv by walking up from its own path; override with
`GP8_VENV_PYTHON`. Any standalone app/test invocation must likewise use
`.venv/bin/python` with `PYTHONPATH=$HOME/ros2_ws/src`.

### Force a single skill (debug)

`GP8_FORCE_SKILL=throw|push` (env, for launch) or `--skill throw|push` (CLI flag,
wins over env) pins every object to one skill, bypassing routing *and*
`can_handle`. Empty = normal routing (everything → throw today).

## Tests

There is **no pytest suite** and `colcon test` is not meaningful here. The
`tests/` dir holds *interactive hardware* scripts that move the real robot:

```bash
ros2 run gp8_control queue_test          # Queue-mode controller methods, no torch
ros2 run gp8_control terminal_debug      # keyboard EE jog / suction / queue sweep
# queue_test_throw is NOT a registered entry point — run via venv python:
PYTHONPATH=$HOME/ros2_ws/src ~/ros2_ws/src/gp8_control/.venv/bin/python \
  -m gp8_control.tests.queue_test_throw  # torch NN throw / pick→throw chaining
```

All of these require `debug_robot.launch.py` running first and a clear 1 m
radius around the arm (throw is a real swing).

## Architecture

### Orchestrator + skills (the central design)

`app.py` (`GP8App`, node name `gp8_manager`) is deliberately thin — it does ROS
wiring, perception intake, target selection, and the main loop. **Manipulation
logic lives in `skills/`, not the app.** Each epoch:

1. `_select_ambush_target` walks the tracked-object queue, drops un-catchable
   heads, and returns a `PickRequest` (target + intercept geometry).
2. `ActionSelector.select(request)` (`planning/action_selector.py`) routes the
   object to a `ManipulationSkill` — rule-based today (always `throw`); this is
   the seam meant to be replaced by an RL policy later.
3. `skill.execute(request)` runs the whole manipulation.

Skills (`skills/base.py::ManipulationSkill`) never reach back into `GP8App`.
Everything they need — robot, controllers, planner, conveyor, queue, and
loop-owned callbacks (`intake`, `publish_state`, `set_status`,
`set_active_target`) plus shared motion primitives (`move_through`,
`wait_for_arrival_and_suction`, `sleep_until`, `scan_next_intercept`) — is on
the injected `SkillContext` (`skills/context.py`). **Adding a skill = one new
file + one entry in the `ActionSelector` list in `GP8App._build_skills`.** Keep
it that way; this decoupling is intentional (see memory: skill-architecture
direction). `ThrowSkill` is the real one; `PushSkill` is a debug stub
(`can_handle` returns `False`, only runs when forced).

### Pick strategies

`Config.PICK_STRATEGY` selects between `"ambush"` (default — park at a fixed
intercept line `GRASP_INTERCEPT_Y`, fire suction on arrival; skill-based path)
and `"moving"` (legacy predictive intercept via `lock_or_drop_head` +
`_execute_cycle`, still in `run_epoch`). New work goes through ambush + skills;
the moving path reuses `ThrowSkill`'s public planning methods.

### Two-process bridge — `bridge.py` is mandatory

MotoROS2 names joints `joint_1..6`; everything else (URDF/SRDF/MoveIt/GUI/
kinematics) uses `joint_1_s..joint_6_t`. `NameBridge` translates **both
directions** and proxies the FJT action and `queue_traj_point` service under
`/motoman_gp8_controller/...`. Consumers read `/joint_states_urdf` and call the
proxied action — they never touch raw MotoROS2 topics directly. The bridge also
**re-stamps** relayed `/joint_states` with the local clock: MotoROS2 can emit a
negative-seconds stamp before its clock syncs, which makes `rclcpp::Time` throw
and SIGABRT-kills every C++ consumer (move_group, robot_state_publisher) — do
not remove that re-stamp.

### MotoROS2 Point Queue Mode gotchas (`controllers/trajectory_controller.py`)

The robot runs in **Point Queue Mode**, not plain FJT, to avoid
`INIT_TRAJ_INVALID_STARTING_POS`. Two hard rules, both already handled but easy
to break:

- **Re-enter queue mode before *every* trajectory.** MotoROS2 silently leaves
  queue mode once a trajectory's queue drains; the next push is rejected with
  "Must call start_point_queue_mode". This is why `ThrowSkill.execute` and
  `PushSkill.execute` call `enter_queue_mode()` before each segment (pick, then
  throw). Skip it and only the first object works.
- **First queued point must equal the measured current position** (code 204).
  `_build_queue_waypoints` overwrites `positions[0]` with `current_joints` to
  satisfy this — even though it looks like it sends a "planned" start.

`enter_queue_mode()` auto-calls `/reset_error` once on recoverable alarms
(101/102/112, e.g. 4414 excessive segment velocity) and retries; hardware/major
alarms still need the pendant. Always `exit_queue_mode()` on shutdown or the
next run's FJT goals get rejected.

### Throw trajectory (`skills/throw_skill.py` + `trajectory/`)

The throw is NN-driven: `TrajectoryPredictor` (torch FCN, weights bundled in
`model/*.pt`) → `compute_throw_params` → `new_trajectory` synthesizes the arc.
The NN motion is time-parameterised with **no joint-speed bound**, so
`build_throw_trajectory` rescales `params.T` (up to 2 passes) to keep every
segment under `M1` joint-velocity limits (else Yaskawa alarm 4414). Suction
release is a **timed interleave**: `send_trajectory_queue_with_timed_release`
fires `suction_off` right after the `release_index` waypoint is *queued* (point
queue and IO are independent services), not after the whole push — the old
"push all, then release" path released too late. The full NN arc (including
follow-through to `aim_joint2`) is kept, then a chain segment to the next pick's
intercept (or current grasp) is appended so the arm flows between picks instead
of parking high. Per-class throw angles live in `THETA_MAP`.

### Perception / tracking data path

Live detections arrive on the `/camera_debug/detections` ROS topic, published by
a separate `camera_debug` node that owns the HTTP NDJSON stream, the cam→base
transform, Z offsets, and `v*delay` back-projection. `GP8App` only subscribes
and reads pre-corrected base-frame poses. (`StreamDetectionSource` /
`DetectionIntake` / `_build_intake` also exist for an in-process intake path but
are **not wired into `setup()`** currently — don't assume they're live.)
`_intake_new_detections` does spatial dedup against `OBJECT_MATCH_EPSILON`
because the detector emits no per-object identity (re-detects the same item each
frame). Belt-frame Y *decreases* as the belt advances; objects are extrapolated
forward by `conveyor.current` (from `/conveyor/speed`, with a hardcoded fallback).

### Package layout note

This is a flat ROS 2 Python package: source dirs sit at the repo root but are
mapped into the `gp8_control.*` namespace via `package_dir` in `setup.py`. When
you add a new subpackage directory you must register it in **both** `packages`
and `package_dir` there, or the colcon install won't import it. Always import as
`from gp8_control.<subpkg> import ...`.

## Configuration

Most tunables are dataclass fields on `Config` in `config.py` (extensive inline
comments explain each). Calibration/extrinsics that must match the `camera_debug`
node live in `perception/extrinsics.py` and `config/*.yaml` — keep them in sync.
Real infra IPs/URLs go in a gitignored `.env` (template: `.env.example`); the
launch file loads it without clobbering shell-set vars.

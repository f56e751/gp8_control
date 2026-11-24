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

`GP8_FORCE_SKILL=throw|robust_throw|push` (env), launch arg
`skill:=throw|robust_throw|push`, or `--skill ...` (CLI flag, wins over env) pins
every object to one skill, bypassing routing *and* `can_handle`. Empty = normal
routing. `robust_throw` is the NLP (CasADi/IPOPT) thrower in
`skills/robust_throw_skill.py` — loaded lazily in `_build_skills`, needs `casadi`
in `.venv` plus `skills/throw_nlp.py`/`skills/throwing.py`; without them normal
throw/push runs still boot (selecting it then is a hard error).

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

The exception is `tests/sim_smoke.py` — an rclpy/torch-free end-to-end check of
the MuJoCo backends (stepper realtime lock → schema-v2 detections → encoder
distance → timed weld pick → lift → timed-release throw). It runs anywhere
numpy/scipy/mujoco import (including Windows):
`python -m gp8_control.tests.sim_smoke [--viewer]`.

## Architecture

### Orchestrator + skills (the central design)

`app.py` (`GP8App`, node name `gp8_manager`) is deliberately thin — it does ROS
wiring, perception intake, target selection, and the main loop. **Manipulation
logic lives in `skills/`, not the app.** Each epoch:

1. `_select_ambush_target` walks the tracked-object queue, drops un-catchable
   heads, and returns a `PickRequest` (target + intercept geometry).
2. `ActionSelector.select(request)` (`planning/action_selector.py`) routes the
   object to a `ManipulationSkill` — rule-based today (per-class via
   `Config.SKILL_BY_CLASS`: `metal`→push, `transparent`→throw); this is the seam
   meant to be replaced by an RL policy later.
3. `skill.execute(request)` runs the whole manipulation.

Skills (`skills/base.py::ManipulationSkill`) never reach back into `GP8App`.
Everything they need — robot, controllers, planner, conveyor, queue, and
loop-owned callbacks (`intake`, `publish_state`, `set_status`,
`set_active_target`) plus shared motion primitives (`move_through`,
`wait_for_arrival_and_suction`, `sleep_until`, `scan_next_intercept`) — is on
the injected `SkillContext` (`skills/context.py`). **Adding a skill = one new
file + one entry in the `ActionSelector` list in `GP8App._build_skills`.** Keep
it that way; this decoupling is intentional (see memory: skill-architecture
direction). Both `ThrowSkill` and `PushSkill` are active — routing sends
`metal`→push and `transparent`→throw (`Config.SKILL_BY_CLASS`), and each skill's
`can_handle` gates the classes it accepts (`PushSkill` accepts `PUSH_CLASSES`).

### Pick strategies

Every object goes through the **ambush** path: `_select_ambush_target` picks a
**dynamic** grasp intercept (`Intercept.intercept_y`, NOT the old fixed
`GRASP_INTERCEPT_Y` line), the arm waits there and grabs on arrival, and the
selected `ManipulationSkill` runs the manipulation. (An older predictive "moving"
strategy and its support code — `lock_or_drop_head`, `_execute_cycle`,
`PICK_STRATEGY` — have been removed.)

How the arm *meets* the object at that intercept is the skill's `PickWaitMode`.
`ThrowSkill` defaults to **`TRACK_DESCEND`**: it parks at `TRACK_Z_START` above
the intercept, and on arrival dispatches one cartesian segment
(`_build_track_descend`) that follows the object downstream at belt speed
(ramp → cruise → ramp, zero relative velocity through the cruise) while the TCP Z
ramps to `TRACK_Z_END` at `TRACK_Z_SPEED`. The throw's start pose is then
**re-bound to where that follow ended** (downstream and lower than the nominal
grasp) before `compute_throw_params`. The three heights/rate are per-run flags —
`track_z_start:=` / `track_z_end:=` / `track_z_speed:=` (launch),
`--track-z-start/--track-z-end/--track-z-speed` (CLI),
`GP8_TRACK_Z_START/END/SPEED` (env); `nan` derives the heights from `GRASP_Z`,
and `track_z_speed:=0` restores the old parked `WAIT_AT_GRASP` pick.
`RobustThrowSkill` has its own, different pick (`HOVER_DESCEND`: vertical press
to `PRESS_Z` on arrival, *then* belt-follow for `PICK_TIME`) — the two are
deliberately separate.

### Cycle-throw mode (`GP8_MODE=cycle`, `cycle_throw/`)

A second run mode beside the skill loop. In this mode the vendored,
HW-validated grasp-and-throw cycle planner (`third_party/grasp_and_throw/`)
owns the arm and the whole object stream. **Keep the vendored tree unchanged.**
Integration lives only in `cycle_throw/`:

* `planner_link.py` runs in a **child process** (cwd = vendored `planner/`,
  its `env.sh` sourced). It calls `rt_cont.run` unchanged, the same way the
  vendored `rt_dry.py` does. It must not import our `gp8_control`, because
  the planner ships its own `gp8_control` fork that has to win the import. It
  talks JSON lines to the app (`now`/`cands`/`commit`/`suction`, README sec. 6
  of the vendored folder). Its stdin reader starts only **after**
  `rt_inv.pool()` forks: a fork while a thread blocks in `stdin.readline`
  deadlocks every pool worker.
* `runner.py::CycleThrowRunner` (app process) feeds tracked objects
  (`y0 = y + 0.12·t`) and streams the committed 2 ms samples through
  `RobotBackend._emit_sample` at `t0 + k·2 ms`, holding the last sample in gaps.
  It applies the vendored 2 ms increment-window guard (CODE 112) and fires
  suction on the sample clock. The planner's "robot sign" equals our joint
  convention (its home pose FKs to TCP (0.40, 0, 0.12) in `robots/gp8.py`).
* The planner pre-distorts its commands for the identified controller filter
  (40 ms dead time + 200/80 ms moving averages). The twin therefore runs cycle
  mode with `SimConfig.servo_filter="box200_80"` (`cycle_sim_config()`).
  Without the filter every throw misses.
* Push-routed objects (`ActionSelector`, metal by default) are never shown to
  the planner. They run in **push slots** (`runner._push_slot`,
  `push_slot.make_push_handler`). The runner hides all objects so the planner
  ends its segment at its wait point. It waits until the end piece has arrived
  (the "segment ends" log line comes before the command thread commits it),
  pauses the 2 ms stream, and runs the normal `PushSkill` with no chain move.
  It then returns to the exact last planner command and holds it
  ≥ `GP8_CYCLE_SETTLE_S` (> the 0.32 s filter) before resuming. Splicing
  anywhere else breaks the planner's command-history assumption.
* Object top height: the planner defaults to 0.02 m, but our stack's belt sits
  higher (`GRASP_Z` 0.042, `PUSH_HEIGHT` 0.03 = paddle on belt). The twin keeps
  our belt and passes the real top (`sim_object_top`). On hardware, check the
  top height (`GP8_CYCLE_TOP_Z`) before running.
* Command output on hardware goes through the **controller buffer**
  (`motoman_controllers/SampleBuffer`, driver branch `feature/sample-buffer`,
  loaded inactive with `sample_buffer:=true` / `GP8_SAMPLE_BUFFER=true`).
  `runner._BufferOutput` writes whole pieces by controller tick via
  `TrajectoryController.buf` (the teammate's `controllers/sample_buffer.py`
  `BufferClient`; the topics are created at construction when `GP8_MODE=cycle`).
  The controller puts out sample k at its tick k, so no Python loop paces the
  stream. Mapping: planner k = tick + `shift`, re-mapped after each
  activation (the tick restarts at 0). The MuJoCo twin and
  `GP8_CYCLE_OUTPUT=stream` keep the 2 ms Python stream.
  `tools/cycle_dry_check` checks both paths without the robot (fake buffer
  with the C++ rules, or `--ros` on mock hardware, domain 77).
* The planner normally runs on robot6 (`GP8_CYCLE_PLANNER_HOST=robot6-planner`)
  over ssh; `planner_link` maps its clock onto the app's by ping/pong.
* Hardware needs a 500 Hz driver plus `GP8_STREAM_HZ=500`
  (`utils/stream_rate.py`; default 250; the robot PC driver runs 500 Hz since 11-19). The NN
  comes in model sets per belt speed / bin distance (`GT_BELT` 0.10–0.20,
  `GT_BIN_X` 1.10 / 1.35 / 1.60; bins at z −0.11). The runner reads the same
  variables (y0 extrapolation, sim bins) and forwards them to the planner,
  whose `env.sh` refuses a combination without a set. Git holds only the
  default set (`zm011_z`, 100 MB via LFS); the other seven live untracked in
  the robot6 checkout (`/PublicSSD/iitp/demo/gp8_control`).

### Piece mode (`GP8_MODE=pieces`, `cycle_throw/piece_runner.py`)

A third run mode: skills return a `MotionPiece` (`skills/base.py`) from
`plan_piece()` instead of moving the arm. `PieceRunner` sends the steps due
next (`throw` held object, pushes, `pick` of the next throw object) as ONE
`plan` request to `cycle_throw/piece_link.py` (robot6, vendored planner dir),
which plans them as one planned REALIZED motion and fits the commands once
with the vendored `rt_inv` (split only at contact keys). Throw = `rn2` cycle
(`skills/planner_throw_skill.py` says what/which bin); push =
`PushSkill.plan_piece`: the stroke commands from the backswing + fire time,
planned as F(U) = the identified controller filter applied to them (what the
real arm does with that push today). Moves between steps are vendored `p2p`
from the current moving state, so there is no rest hand-off, settle hold,
controller switch or command smoothing. Buffer write lead comes from the
measured write round trip (`runner._BufferOutput.lead_ticks`). Routing stays
`ctx.skill_for` (the RL seam); the vendored tree is unchanged. Details, flags,
twin results: `cycle_throw/PIECES.md`. `gp8_cycle` runs it with
`GP8_DEMO_MODE=pieces`.

Planning rules that keep the arm moving (do not regress them): the safe
default (rt_cont's end at the fixed wait point) is queued as lazy chunks right
after `rn2`, and a better plan replaces it only while its first chunk is
unsent; a plan that ends at rest keeps its end open, and the next request
carries on from it at `b + L`; the app never fills buffer gaps (the controller
holds). Env for the link must start with `GP8_PIECE_`/`RT_` to be forwarded.

**Real-to-sim recording**: `GP8_RUN_RECORD_DIR` (on by default in
`gp8_cycle`, `<log>/record`) makes the cycle/piece runner record joints,
commands as played, link messages, detections and tracks
(`utils/run_record.py`, recorder thread; encode with C json, never per-float
Python, or the link's planning deadlines slip). `tools/real2sim.py`
summarizes, compares measured joints with the filter model (`--fit`), turns a
run's object stream into a twin spawn script (`GP8_SIM_SPAWN_SCRIPT`, started
at planner t0) and compares two runs object by object.

**One place per piece-mode rule** (keep it so): `cycle_throw/piece_stack.py`
`PieceStack` assembles piece mode (skills, which classes each skill takes, the
policy decider, the runner); `AppPieceStack` (app, robot) and `TwinPieceStack`
(MuJoCo twin without ROS: RL env, `tools/piece_twin.py`) only add what
differs. The decision (observation, mask, policy) is `rl/piece_policy.py`
`PieceDecider`. The SampleBuffer rules in Python are
`controllers/sample_buffer_rules.py` `SampleBufferRules` (the twin's
`SimTickBuffer` and the dry check's fake inherit them; C++ on the robot).
Piece-mode control settings live in `config/piece_mode.env` (read by
`gp8_cycle` and by `cycle_throw/piece_settings.py` for the twin); robot-site
calibration (object top, push height on the real belt, camera signs) stays in
the robot PC's `.env`. The suction valve delay is `runner.VALVE_S` (0.2 s): the
robot fires OFF that early, the cycle/piece twin's valve releases that late
(`GP8_SIM_SUCTION_OFF_DELAY`).

**RL on piece mode** (guide for trainers: `docs/RL_PIECE_TRAINING.md`;
`rl/piece_env.py`, `train_ppo_smoke --env piece`): the
policy picks the next object/skill whenever piece mode asks
(`PieceRunner(decide_fn=...)`); the twin runs on virtual time
(`backends/sim_tick.py`: wall speed while the link plans, warped otherwise; the
link moves its clock on `warp` after committing what falls due and acks
`warped`; commands go through `SimTickBuffer`, the SampleBuffer rules on sim
ticks). Runners take `vtime=`; without it every clock is `time.monotonic`.

**Running with robot6** (operator steps in README section C): the robot PC
runs driver + app + camera_debug, robot6 (`iitp@147.46.175.15`, checkout
`/PublicSSD/iitp/demo/gp8_control` with its own `.venv`) runs the planner
link over ssh (`GP8_CYCLE_PLANNER_HOST=robot6-planner`) and the vision docker.
Deploy = `git pull` on BOTH machines; rebuild on the robot PC with the motoman
driver workspace sourced first. One command: `GP8_DEMO_MODE=pieces gp8_cycle`
(`GP8_DEMO_CHECK_ONLY=1` = robot-free check).

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

### Backend seams (`backends/`) — HW and sim behind two base classes

The app drives the robot and the belt world through two Python seams, selected
by `Config.BACKEND` (`GP8_BACKEND=hw|mujoco`, CLI `--backend`, launch
`sim.launch.py`):

* `backends/robot_base.py::RobotBackend` — **the hardware operation logic
  lives HERE**, shared by both implementations: the 250 Hz stream engine
  (cubic-Hermite resample clamped to the bracketing-knot envelope, grid-time
  suction release, `tick_fn` early-stop protocol, wall-clock pacing), suction
  request coalescing + telemetry, the `send_trajectory_queue*`/`pq_*` surface,
  `_wait_for_position`. A backend implements only `wait_for_servers`,
  `_emit_sample` (one 4 ms joint sample), and `_set_suction`.
* `backends/world_base.py::WorldSource` — `latest_snapshot()` (the raw
  schema-v2 dict `DetectionIntake` consumes, unchanged) + `.belt` (a
  `ConveyorSpeedTracker`-compatible object; the app assigns it to
  `self.conveyor`).

Implementations: hardware = `controllers/trajectory_controller.py`
(`TrajectoryController(RobotBackend)`) + `backends/ros_world.py`
(`HardwareWorldSource`); sim = `backends/mujoco_sim.py` (`SimCore` +
`MujocoRobotBackend` + `MujocoWorldSource`, all rclpy-free). Do NOT re-mock the
ROS wire for simulation — the class contract is the contract (the old
wire-mocking `mock/` stack drifted and was deleted for exactly that reason).

### Control path (hardware: `controllers/trajectory_controller.py`)

This branch runs the **adv4ncr 250 Hz stream** driver: the base-class engine
resamples each trajectory onto a 4 ms grid and `TrajectoryController` publishes
each sample to `/JointGroupPositionController/commands` (Float64MultiArray),
driving suction via a Simple-Message TCP call (port 50242) on a dedicated IO
worker thread. The rate is `GP8_STREAM_HZ` (`utils/stream_rate.py`, 250 or 500).
The YRC1000micro interpolates every 2 ms, so at 250 Hz the RT speed ceiling in
`robots/gp8.py` is half the datasheet. Use 500 only with a 500 Hz driver. A `jtc` fallback backend (`GP8_ADV4NCR_BACKEND=jtc`, FJT action)
stays hardware-only inside this file. There is **no MotoROS2 Point Queue Mode**:
`enter_queue_mode`/`exit_queue_mode`/`pq_*` remain only as no-op lifecycle
shims (in the base class). Skills just dispatch trajectories. (The prior
MotoROS2 queue contract lives on the pre-migration `main` branch, not here.)

### MuJoCo twin (`backends/mujoco_sim.py`)

`SimCore` loads the vendored scene (+ a reachable conveyor injected via MjSpec)
and runs a **wall-clock-servoed stepping thread** (2 ms physics; the arm's
position actuators ZOH the last streamed 4 ms command exactly like the real
JGPC). Stream samples and suction toggles are NOT applied on arrival: they go
into one lock-free time-stamped queue (`_pending_ctrl`) that the stepper
replays per substep on the physics clock, and `SimCore.lock` is held per
substep, never across a catch-up batch — so a render stall (`viewer.sync()`)
can't starve the 250 Hz stream thread or skew the release timing. Unused boxes
park below the floor with collisions off + gravity compensation (compile-time
`gravcomp` on the box bodies — runtime-only values are ignored when
`ngravcomp == 0`). Suction ON only ARMS the vacuum; it seals on contact
(`_vacuum_tick`, per substep): with the cup face over a box's top face and within
`grab_gap`, the box is pressed — its geom thickness follows the cup down to
`crush_min_frac`, its bottom is clamped to the belt — and the weld to link6
engages when the cup stops descending, at the crushed pose. The activation-time
relative pose must be written into `model.eq_data` (the XML weld's zero relpose
is baked at compile time); suction OFF releases the weld and the box flies with
its true dragged velocity. Never weld at a distance: a box hanging 56 mm below
the cup adds ω×r to its release velocity (+0.3 m range) and gets dragged
through the belt during the press. On-belt boxes' Y is kinematically driven to match
the integrated encoder distance exactly (contact friction would otherwise brake
them ~15% under the commanded belt speed); X/Z stay dynamic. Perception is
synthesized schema-v2 through the real `perception/bbox_geometry.py` transform,
**gated to the real camera's image**: `SimCamera` (pose =
`extrinsics.T_ROBOT2BASE @ T_BASE2CAM`, pinhole = `config/realsense_camera_info.yaml`)
re-poses the vendored `d435i` body/`<camera>` and only boxes it can see are
reported (belt Y ≈ 2.15–2.79 m), so the app's belt dead reckoning carries every
object from the camera to the pick exactly as on hardware. Boxes spawn just
past the view's upstream edge (`spawn_y` nan → derived).

Perception lag in the twin: `GP8_SIM_PERCEPTION_LAG` [s] (default 0) makes
`SimCore` report every belt object `belt_speed x lag` upstream, in the camera
snapshot AND `truth_detection_for` (the RL path), stamped fresh. The hardware
push fires `PUSH_PERCEPTION_LAG` (0.28 s, `GP8_PUSH_PERCEPTION_LAG`) early to
cover that lag, so a zero-lag twin with the HW correction pushes 3-8 cm ahead
of the object (`sim_experiment/`, push analysis 2026-11-18). Either run the
twin with lag 0.28 or set the correction to 0; never mix. `GP8_PUSH_GUARD=0`
drops the far-side guard limits (sim experiments only; the guard is real).
`sim_experiment/` holds that analysis's tools (README/TODO inside; robot17).

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
of parking high. The throw heading is computed per-object as the bearing from the
grasp to the fixed bin (`THROW_BIN_X`/`THROW_BIN_Y`).

### Perception / tracking data path

Live detections arrive on the `/camera_debug/detections` ROS topic, published by
a separate `camera_debug` node that owns the schema-v2 HTTP NDJSON stream,
transforms all four bounding-box corners into the base frame, applies Z offsets
and `v*delay` back-projection, and derives the legacy grasp target from the box
centre. `GP8App` only subscribes and reads pre-corrected base-frame poses and
the retained full boxes. (`StreamDetectionSource` /
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

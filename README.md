# gp8_control

ROS 2 (Humble) runtime control package for the Yaskawa **GP8** 6-DOF arm
running MotoROS2 firmware. Originally part of
[iitp_robot_control](https://github.com/f56e751/iitp_robot_control); split out
into its own repository so the production control stack is decoupled from
the RL training stack.

## What's inside

| Component | Purpose |
|---|---|
| `app.py` | Main pick-and-throw orchestrator (`gp8_manager` node) — tracked-objects queue, conveyor-compensated IK, throw trajectory, torch NN release-param predictor. Ported from `main_sam7.py`. |
| `bridge.py` | MotoROS2 ↔ URDF joint-name bridge. Renames `joint_1..6` ↔ `joint_1_s..joint_6_t` on `/joint_states` and proxies `/follow_joint_trajectory`. |
| `robots/gp8.py` | GP8 kinematics (PoE screw axes, FK/IK, Jacobian). |
| `trajectory/predictor.py` | In-process torch FCN loader for throw trajectory parameters. |
| `trajectory/trajectory_primitive.py` | `opt_time`, `trajectory_3points`, `new_trajectory`, etc. |
| `controllers/trajectory_controller.py` | FJT action client with suction release-on-pass logic. |
| `controllers/moveit_controller.py` | MoveIt 2 wrapper (used only for initial-pose planning). |
| `perception/` | SAM ZMQ client, AprilTag subscriber, RealSense camera manager, calibrated CameraInfo publisher. |
| `mock/mock_robot.py` | Fake MotoROS2 for dev work without the physical robot. |
| `gui/` | Flask-based web GUI for manual EE jogging and status. |
| `launch/gp8_bringup.launch.py` | Full bringup — bridge, robot_state_publisher, MoveIt, RealSense, AprilTag, `gp8_manager`. |
| `launch/debug_robot.launch.py` | Minimal bringup (bridge + TF + MoveIt) for interactive scripts. |
| `model/NN_newprimitive2_…pt` | Bundled throw-trajectory FCN weight (1.1 MB). |
| `config/` | RealSense calibration, AprilTag settings, robot.yaml. |

## Prerequisites

- ROS 2 Humble on Ubuntu 22.04
- MotoROS2 running on the YRC1000micro controller
- micro-ROS Agent reachable at `192.168.255.5:8888` (or wherever MotoROS2
  is configured)
- `esp32_encoder` ROS 2 node publishing `/conveyor/speed` (optional, falls
  back to hardcoded speed)
- Remote SAM inference server reachable — set its address via env vars
  before launching:
  ```bash
  export GP8_SAM_SERVER_IP=<your-sam-server-ip>
  export GP8_SAM_SERVER_PORT=7150   # default
  ```
  (The public GitHub copy of this repo intentionally ships with
  `127.0.0.1` as a placeholder so no internal infra IPs leak.)
- [`uv`](https://astral.sh/uv) for Python venv management
- ESP32 conveyor encoder on `/dev/ttyUSB0` (user in `dialout` group)

## Setup

```bash
# Inside this package
cd ~/ros2_ws/src/gp8_control
uv sync                              # creates .venv with torch, scipy, etc.

# Build the ROS 2 package
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select gp8_control
```

## Launch

Needs **two terminals** (plus whatever is already running MotoROS2 /
micro-ROS Agent / Docker).

### Terminal 1 — bringup (bridge + cameras + MoveIt + gp8_manager)

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch gp8_control gp8_bringup.launch.py
```

`gp8_manager` is launched via an `ExecuteProcess` wrapper that uses this
package's `.venv/bin/python` (found by walking up from the launch file) so
that torch is available. Override the venv location with
`GP8_VENV_PYTHON=/path/to/python` if needed.

### Terminal 2 — conveyor encoder (publishes `/conveyor/speed`)

Lives in a sibling package:
<https://github.com/f56e751/esp32_encoder>

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 run esp32_encoder conveyor_node
```

First time only — add the user to `dialout` so the ESP32 USB-UART
(`/dev/ttyUSB0`) is accessible, then **log out and back in** (group
changes don't apply to already-running shells):

```bash
sudo usermod -aG dialout $USER
```

The encoder node prints a live TUI with belt speed (m/s) and cumulative
distance, and publishes `/conveyor/speed` (`std_msgs/Float64`) which
`gp8_manager` subscribes to for conveyor-compensated picking.

If the encoder node is *not* running, `gp8_manager` falls back to
`Config.CONVEYOR_SPEED` (hardcoded) and logs a warning — picking still
works but is less accurate when the belt speed drifts.

### micro-ROS Agent (once per boot)

Needed to bridge MotoROS2 ↔ ROS 2. Without it, `/write_single_io` /
`/start_traj_mode` never appear and nodes hang on *"Waiting for …"*.

```bash
# 1. Already running?
sudo docker ps | grep microros_agent

# 2. Start (keep on one line — trailing spaces after \ will break it)
sudo docker run -d --rm --net=host --name microros_agent microros/micro-ros-agent:humble udp4 --port 8888 -v6

# 3. Verify
sudo docker logs microros_agent 2>&1 | grep -i session   # "session established"
ros2 service list | grep write_single_io
```

Pendant must be in **REMOTE + AUTO** with no alarms.

## Topology

```
MotoROS2 (on YRC1000micro)  ─ /joint_states (joint_1..6, BEST_EFFORT)
                            ─ /follow_joint_trajectory  (joint_1..6)
                            ─ /write_single_io, /read_single_io, /start_traj_mode
          │
          │  (ROS 2 DDS over 192.168.255.x)
          ▼
  motoros2_name_bridge  ─ /joint_states_urdf  (joint_1_s..6_t)
                       ─ /motoman_gp8_controller/follow_joint_trajectory
          │
          ▼
  gp8_manager (app.py)
   ├─ /joint_states_urdf      (subscribe)
   ├─ /conveyor/speed          (subscribe — esp32_encoder)
   ├─ SAM ZMQ client → remote GPU
   └─ FJT action → bridge → MotoROS2
```

## Status

Production-tested on YRC1000micro + MotoROS2 0.2.1 + Humble as of
2026-04. Pick timing (`fixed_delay`) is learned adaptively per run.

#!/bin/bash
# Rebuild gp8_control in a CLEAN environment (no other workspace leaking into
# the underlay chain). gp8_control is copy-built: run this after every edit.
# Extra args go to colcon, e.g.  build.sh --packages-select gp8_control esp32_encoder
WS="$(cd "$(dirname "$0")/../../.." && pwd)"
DRIVER="${GP8_DRIVER_WS:-$HOME/Desktop/motoman_ros2_safety_review/motoman_ROS2/install}"
cd "$WS" && exec env -i HOME="$HOME" PATH=/usr/bin:/bin TERM="${TERM:-xterm}" bash -c "
  source /opt/ros/humble/setup.bash && source '$DRIVER/setup.bash' &&
  colcon build --parallel-workers 3 ${*:---packages-select gp8_control}"

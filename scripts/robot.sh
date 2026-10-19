#!/bin/bash
# Robot driver + TF + MoveIt + belt encoder. The arm does NOT move here.
# Pair with s2r_run.sh (RL runner) — for the plain app use app.sh instead.
source "$(dirname "$0")/env.sh"
exec ros2 launch gp8_control debug_robot.launch.py "$@"

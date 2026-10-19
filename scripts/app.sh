#!/bin/bash
# Full demo: driver + MoveIt + belt encoder + gp8_manager (rule-based app).
# Moves to the initial pose ~3 s after start.
source "$(dirname "$0")/env.sh"
exec ros2 launch gp8_control gp8_bringup.launch.py \
    track_lead_t:="${GP8_TRACK_LEAD_T:-0.12}" "$@"

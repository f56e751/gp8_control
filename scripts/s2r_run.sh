#!/bin/bash
# RL real runner (rule-based policy) + sim-to-real validation recording.
# Needs robot.sh + camera.sh running. Moves to the initial pose immediately.
# Ctrl-C ends the run; ~/gp8_s2r_runs/<stamp>_rl_hw/report.md follows in 1-2 min.
source "$(dirname "$0")/env.sh"
export GP8_TRACK_LEAD_T="${GP8_TRACK_LEAD_T:-0.12}"
exec "$GP8_WS/src/gp8_control/.venv/bin/python" -m gp8_control.rl.real_runner \
    --bbox-observation corners --steps 0 "$@"

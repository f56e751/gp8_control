#!/bin/bash
# camera_debug: camera PC stream -> /camera_debug/detections (robot base frame).
# The camera is mounted rotated 180 deg vs. its May calibration, so both axis
# signs are flipped by default (2026-10-19). Override GP8_CAM_SIGN_CX/CY if
# the camera is re-mounted / re-calibrated.
source "$(dirname "$0")/env.sh"
export GP8_CAM_SIGN_CX="${GP8_CAM_SIGN_CX:-+1}"
export GP8_CAM_SIGN_CY="${GP8_CAM_SIGN_CY:--1}"
exec ros2 run gp8_control camera_debug "$@"

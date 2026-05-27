"""Shared perception extrinsics, Z offsets, and workspace-filter constants.

Both ``gp8_control.app.Config`` and ``gp8_control.camera_debug`` import these
so the correction pipeline has a single source of truth. Lives in a module
that does NOT pull torch / rclpy so it's safe to import from any node.
"""

from __future__ import annotations

import numpy as np


# Fixed extrinsics (main_sam7 trusts these without an AprilTag handshake).
T_ROBOT2BASE: np.ndarray = np.array([
    [0.0, 1.0, 0.0, -0.025],
    [-1.0, 0.0, 0.0, 0.235],
    [0.0, 0.0, 1.0, -0.020],
    [0.0, 0.0, 0.0, 1.0],
])

# X = -2.235 so a centered detection maps to belt-Y = 2.47 m (measured
# camera→pick belt-direction length): 2.235 + 0.235 (T_ROBOT2BASE Y) = 2.47.
T_BASE2CAM: np.ndarray = np.array([
    [0.0, -1.0, 0.0, -2.235],
    [-1.0, 0.0, 0.0,  0.450],
    [0.0,  0.0, -1.0, 0.650],
    [0.0,  0.0, 0.0,  1.0],
])

# Z offsets applied to the cam→base position to get aim (above) and grasp
# (at object surface) poses.
DETECTION_OFFSET_AIM: float = 0.07
DETECTION_OFFSET_GRASP: float = -0.01

# Workspace filter — camera-frame Z<0.67 keeps belt-surface detections,
# |X|<0.2 rejects anything off the belt centerline.
WORKSPACE_Z_MAX: float = 0.67
WORKSPACE_X_ABS: float = 0.2


# ---------------------------------------------------------------------------
# Belt-frame camera format (current)
# ---------------------------------------------------------------------------
# The camera now publishes object positions in belt-frame metres relative to
# the conveyor image-centre, instead of raw camera-frame coords. cx is the
# across-belt offset; cy is the along-belt offset with **+ pointing upstream**
# (toward the camera, before the object arrives at the robot). Depth is no
# longer sent — pick height is anchored to GRASP_Z downstream.
#
# Mapping → robot base frame is a simple constant translation: the image
# centre on the belt is at this fixed point in base coords (derived from the
# old extrinsic transform: image centre cx=cy=0 mapped to base (0.425, 2.470)).
REFERENCE_X_BASE: float = 0.425
REFERENCE_Y_BASE: float = 2.470
REFERENCE_Z_BASE: float = 0.630   # only used for display; ambush picks at GRASP_Z

# Per-axis sign: flip these if a test shows the arm goes to the opposite side.
# cy is upstream-positive → same as base +Y (belt flows toward base −Y).
SIGN_CX_TO_BASE_X: float = +1.0
SIGN_CY_TO_BASE_Y: float = +1.0

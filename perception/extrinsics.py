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

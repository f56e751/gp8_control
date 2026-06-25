#!/usr/bin/env python3
"""Level A, step 1 — kinematics agreement check: gp8_control FK vs the MuJoCo model.

Confirms that gp8_control's `robots/gp8.py` (PoE FK) and the vendored MuJoCo
model (`sim/recycling_mujoco/`) describe the SAME robot in the SAME convention,
so a trajectory gp8_control computes will move the MuJoCo arm to the same place.
If this passes, the Level A replay preview (joint-trajectory -> MuJoCo actuators)
is meaningful; if not, the deltas below tell us where the conventions diverge
(joint sign/zero, base mount, TCP offset).

This is the GLUE — it lives OUTSIDE the vendored tree (see VENDOR.md) and only
reads the model; it does NOT modify recycling_mujoco/.

No rendering (raw MjModel/MjData + mj_forward), so it runs headless anywhere
MuJoCo is importable — no GL/EGL needed.

Run (on a machine with numpy+scipy+mujoco, e.g. conda `iitp` or the gp8 venv +
`pip install mujoco`), with gp8_control importable:

    PYTHONPATH=$HOME/ros2_ws/src python sim/preview_gp8_check.py

Deps: numpy, scipy (gp8 FK), mujoco. (No torch / rclpy / opencv needed here.)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# gp8_control must be importable as a package: add the dir that CONTAINS it
# (e.g. ~/ros2_ws/src) to sys.path. This file is <repo>/sim/preview_gp8_check.py,
# so <repo> = parents[1] and its parent (parents[2]) holds the `gp8_control` pkg.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO.parent))

import mujoco  # noqa: E402
from gp8_control.robots.gp8 import GP8  # noqa: E402

SCENE = Path(__file__).resolve().parent / "recycling_mujoco" / "scene.xml"

# gp8 joint order [J1..J6] == MuJoCo joint names, in order.
MJ_JOINTS = ["S_axis", "L_axis", "U_axis", "R_axis", "B_axis", "T_axis"]
MJ_BASE_BODY = "yaskawa_robot"   # robot mount; gp8 base frame == this body frame
MJ_EE_SITE = "grip_site"         # MuJoCo TCP; expected == gp8 EE (home M)

POS_TOL_M = 0.002    # 2 mm
ROT_TOL_DEG = 1.0

# Test configurations (rad), [S, L, U, R, B, T]. Kept inside joint limits.
TEST_Q = [
    np.zeros(6),
    np.array([0.0,  0.3, -0.3,  0.0,  0.3,  0.0]),
    np.array([0.5, -0.2,  0.4,  0.3, -0.4,  0.6]),
    np.array([-0.8, 0.6,  0.2, -0.5,  0.5, -1.0]),
    np.array([1.2,  0.1, -0.1,  1.0,  0.8,  0.2]),
]


def rot_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Geodesic angle (deg) between two rotation matrices."""
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def main() -> int:
    if not SCENE.is_file():
        print(f"scene not found: {SCENE}")
        return 2

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    gp8 = GP8()

    print(f"model: {SCENE}")
    print(f"joints(check): {MJ_JOINTS}")
    print(f"base body: {MJ_BASE_BODY}   EE site: {MJ_EE_SITE}\n")

    worst_pos = 0.0
    worst_rot = 0.0
    for q in TEST_Q:
        mujoco.mj_resetData(model, data)
        for name, val in zip(MJ_JOINTS, q):
            data.joint(name).qpos[0] = float(val)
        mujoco.mj_forward(model, data)

        # MuJoCo EE in base frame: R_base^T @ (p_site_world - p_base_world)
        p_base_w = data.body(MJ_BASE_BODY).xpos.copy()
        R_base_w = data.body(MJ_BASE_BODY).xmat.reshape(3, 3).copy()
        p_site_w = data.site(MJ_EE_SITE).xpos.copy()
        R_site_w = data.site(MJ_EE_SITE).xmat.reshape(3, 3).copy()
        p_mj = R_base_w.T @ (p_site_w - p_base_w)
        R_mj = R_base_w.T @ R_site_w

        # gp8 FK in its base frame
        T = gp8.forward_kinematics(q)
        p_gp8, R_gp8 = T[:3, 3], T[:3, :3]

        dp = float(np.linalg.norm(p_mj - p_gp8))
        dr = rot_angle_deg(R_mj, R_gp8)
        worst_pos = max(worst_pos, dp)
        worst_rot = max(worst_rot, dr)

        print(f"q = [{', '.join(f'{v:+.2f}' for v in q)}]")
        print(f"  gp8   EE  : [{p_gp8[0]:+.4f} {p_gp8[1]:+.4f} {p_gp8[2]:+.4f}]")
        print(f"  mujoco EE : [{p_mj[0]:+.4f} {p_mj[1]:+.4f} {p_mj[2]:+.4f}]")
        print(f"  Δpos = {dp*1000:7.2f} mm    Δrot = {dr:6.2f}°\n")

    ok = worst_pos <= POS_TOL_M and worst_rot <= ROT_TOL_DEG
    print(f"worst Δpos = {worst_pos*1000:.2f} mm (tol {POS_TOL_M*1000:.0f} mm), "
          f"worst Δrot = {worst_rot:.2f}° (tol {ROT_TOL_DEG:.0f}°)")
    print("RESULT:", "✓ KINEMATICS MATCH — replay preview is faithful"
          if ok else "✗ MISMATCH — see deltas (joint sign/zero, base, or TCP offset)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""MuJoCo backends — the physics twin behind the RobotBackend/WorldSource seams.

The twin runs the vendored scene (``sim/recycling_mujoco/scene.xml``, plus a
reachable conveyor injected via MjSpec — the vendored belt sits out of the
GP8's reach; glue only, never edits the vendored tree, per VENDOR.md) and
plugs into the app through the SAME Python seams the hardware uses:

  * :class:`MujocoRobotBackend` — the base class's 250 Hz stream engine (the
    REAL hardware operation logic: clamped-Hermite resample, grid-time release,
    wall-clock pacing) runs unchanged; its ``_emit_sample`` writes the position
    actuators' ctrl targets instead of publishing to the JGPC, and
    ``_set_suction`` welds/releases the nearest belt box instead of a TCP IO
    write. Because the weld physically drags the box through the throw swing,
    the box's free-joint velocity at release IS the throw velocity — no fling
    synthesis needed.
  * :class:`MujocoWorldSource` — synthesizes the schema-v2 detection snapshot
    from the physics boxes through the *actual* perception transform code
    (:mod:`gp8_control.perception.bbox_geometry` + ``extrinsics``), and an
    encoder-equivalent belt distance integrated from the stepped physics
    (finally exercising the app's encoder-distance tracking path in sim).

Both share one :class:`SimCore`, which owns MjModel/MjData and a free-running
**wall-clock-servoed stepping thread**: physics advances so ``data.time``
tracks real time (the app's ``time.time()`` ETAs stay truthful), the belt and
boxes keep moving while the arm idles, and the position servos zero-order-hold
the last streamed command between 4 ms samples exactly like the real JGPC.

This module imports NO rclpy. It needs ``mujoco>=3.1`` (``uv sync --extra sim``).

Headless by default; set ``GP8_SIM_VIEWER=1`` for a passive viewer window.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from gp8_control.backends.robot_base import RobotBackend
from gp8_control.backends.world_base import WorldSource
from gp8_control.perception import extrinsics
from gp8_control.perception.bbox_geometry import bbox_to_base

try:
    import mujoco
except ImportError as _exc:   # surfaced on SimCore construction, not import
    mujoco = None
    _MUJOCO_IMPORT_ERROR: ImportError | None = _exc
else:
    _MUJOCO_IMPORT_ERROR = None


# scene.xml: <pkg>/backends/mujoco_sim.py -> <pkg>/sim/recycling_mujoco/.
# The vendored sim/ tree is deliberately NOT colcon-installed (setup.py), so
# when this module is imported from the install-tree copy the __file__-relative
# path misses — fall back to the source tree (or the GP8_SIM_SCENE override).
def _find_scene() -> Path:
    rel = ("sim", "recycling_mujoco", "scene.xml")
    candidates = []
    env = os.environ.get("GP8_SIM_SCENE")
    if env:
        candidates.append(Path(env))
    candidates.append(Path(__file__).resolve().parents[1].joinpath(*rel))
    candidates.append(Path.home().joinpath("ros2_ws", "src", "gp8_control", *rel))
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        "MuJoCo scene not found; tried:\n  "
        + "\n  ".join(str(c) for c in candidates)
        + "\nThe vendored sim/ tree is not colcon-installed — run from the "
        "source tree (PYTHONPATH=~/ros2_ws/src), set GP8_SIM_SCENE to the "
        "scene.xml path, and check `git lfs pull` hydrated the meshes."
    )

def _find_camera_info() -> Path | None:
    """config/realsense_camera_info.yaml (same source-tree fallbacks as the scene)."""
    rel = ("config", "realsense_camera_info.yaml")
    candidates = []
    env = os.environ.get("GP8_SIM_CAMERA_INFO")
    if env:
        candidates.append(Path(env))
    candidates.append(Path(__file__).resolve().parents[1].joinpath(*rel))
    candidates.append(Path.home().joinpath("ros2_ws", "src", "gp8_control", *rel))
    return next((c for c in candidates if c.is_file()), None)


@dataclass(frozen=True)
class SimCamera:
    """The real camera, in the gp8 BASE frame: pose from
    ``extrinsics.T_ROBOT2BASE @ T_BASE2CAM`` (origin ~(0.425, 2.47, 0.63) m,
    optical axis straight down) and the pinhole from
    ``config/realsense_camera_info.yaml``. Used to (a) place the vendored
    ``d435i`` body + ``<camera>`` in the scene and (b) gate the synthesized
    detections to what that camera can actually see."""
    pos: tuple            # optical centre, base frame [m]
    R: tuple              # 3x3 rows: base <- optical (columns = optical axes in base)
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def fovy_deg(self) -> float:
        return math.degrees(2.0 * math.atan(0.5 * self.height / self.fy))

    def project(self, p_base) -> tuple | None:
        """Pixel (u, v) of a base-frame point, or None if behind the camera."""
        R = np.asarray(self.R); d = R.T @ (np.asarray(p_base, dtype=float) - np.asarray(self.pos))
        if d[2] <= 1e-6:
            return None
        return (self.fx * d[0] / d[2] + self.cx, self.fy * d[1] / d[2] + self.cy)

    def sees(self, p_base) -> bool:
        uv = self.project(p_base)
        return uv is not None and 0.0 <= uv[0] < self.width and 0.0 <= uv[1] < self.height

    def view_y_span(self, z_base: float) -> tuple:
        """Base-frame Y interval visible on the horizontal plane z_base."""
        R = np.asarray(self.R); ys = []
        depth = float(self.pos[2]) - z_base
        for v in (0.0, float(self.height)):
            # optical-frame ray through pixel row v at the image's x-centre
            d = np.array([0.0, (v - self.cy) / self.fy, 1.0]) * depth
            ys.append(float((R @ d)[1] + self.pos[1]))
        return (min(ys), max(ys))


def _default_camera() -> SimCamera:
    T = extrinsics.T_ROBOT2BASE @ extrinsics.T_BASE2CAM
    fx, fy, cx, cy, w, h = 638.663, 638.205, 642.904, 361.377, 1280, 720   # yaml mirror
    info = _find_camera_info()
    if info is not None:
        txt = info.read_text()
        m_w = re.search(r"image_width:\s*(\d+)", txt); m_h = re.search(r"image_height:\s*(\d+)", txt)
        m_k = re.search(r"camera_matrix:.*?data:\s*\[([^\]]*)\]", txt, re.S)
        if m_w and m_h and m_k:
            k = [float(x) for x in m_k.group(1).replace("\n", " ").split(",")]
            fx, fy, cx, cy = k[0], k[4], k[2], k[5]
            w, h = int(m_w.group(1)), int(m_h.group(1))
    return SimCamera(pos=tuple(float(x) for x in T[:3, 3]),
                     R=tuple(tuple(float(x) for x in row) for row in T[:3, :3]),
                     fx=fx, fy=fy, cx=cx, cy=cy, width=w, height=h)


MJ_CAMERA_BODY = "d435i"        # vendored RealSense model, re-posed onto the real camera
MJ_CAMERA = "d435i_view"        # its <camera> child (euler pi about x w.r.t. the body)

# gp8 joint order [S, L, U, R, B, T] == these MuJoCo joints / position actuators
# (confirmed numerically by sim/preview_gp8_check.py).
MJ_JOINTS = ["S_axis", "L_axis", "U_axis", "R_axis", "B_axis", "T_axis"]
MJ_ACTS = ["S_axis_act", "L_axis_act", "U_axis_act", "R_axis_act", "B_axis_act", "T_axis_act"]
MJ_BASE_BODY = "yaskawa_robot"   # gp8 base frame == this body; detections are base-frame
MJ_GRIP_SITE = "grip_site"
MJ_LINK6 = "link6"               # weld body1 (the vendored suction welds anchor here)

# Free-joint boxes already in the vendored model, reused as the object pool,
# and their per-box suction welds (body1=link6, body2=red_box_N, active=false).
BOX_BODIES = ["red_box"] + [f"red_box_{i}" for i in range(2, 13)]
BOX_JOINTS = ["box_free"] + [f"box_free_{i}" for i in range(2, 13)]
BOX_GEOMS = ["red_box_geom"] + [f"red_box_geom_{i}" for i in range(2, 13)]
BOX_WELDS = ["suction_weld_red_box"] + [f"suction_weld_red_box_{i}" for i in range(2, 13)]
# red_box_geom half-extents (size in combined_test.xml): x, y (footprint), z (height)
BOX_HALF_X, BOX_HALF_Y, BOX_HALF_Z = 0.05, 0.075, 0.015
# Parked (unused) boxes sit below the ground plane, each at its own spot, with
# collisions OFF and gravity compensated. (Stacking all 12 free bodies on ONE
# point under an infinite plane made the solver eject them up through the
# floor into a pile beside the robot base — ~70 permanent contacts that
# doubled mj_step and starved the command stream when the viewer was on.)
def _park_pos(i: int) -> np.ndarray:
    return np.array([-2.0 - 0.3 * (i % 4), -2.0 - 0.3 * (i // 4), -1.0])
_HOME = [0.0, 0.0, 0.0, 0.0, -math.pi / 2, 0.0]   # B down — matches app startup

# Per-class box tint so throw (transparent) vs push (metal) reads at a glance.
_CLASS_RGBA = {
    "transparent": (0.30, 0.65, 1.00, 0.55),
    "metal": (0.62, 0.62, 0.68, 1.00),
}
_METAL_SUCTION_OK_RGBA = (0.95, 0.78, 0.18, 1.00)
_METAL_SUCTION_FAIL_RGBA = (0.85, 0.18, 0.12, 1.00)
_DEFAULT_RGBA = (0.85, 0.20, 0.20, 1.00)

# The reachable conveyor we add for the twin (the vendored belt is out of the
# GP8's reach). Geometry in the gp8 BASE frame; placed into the world via the
# yaskawa_robot body pose at build time.
BELT_BODY = "gp8_belt"
BELT_GEOM = "gp8_belt_surface"
BELT_JOINT = "gp8_belt_slide"
BELT_ACT = "gp8_belt_act"
_PHYSICAL_BELT_BASE_X = 0.415
_PHYSICAL_BELT_CENTER_Y = 497.0
_PHYSICAL_BELT_HALF_X = 0.215
_PHYSICAL_BELT_HALF_Y = 500.0
_PHYSICAL_BELT_HALF_Z = 0.035
_BELT_CONTACT_OLD = "old"


@dataclass(frozen=True)
class SimBin:
    """An open-top bin placed in the twin, in the gp8 BASE frame.

    ``x, y`` is the bin centre (== the app's throw goal / push bin target),
    ``rim_z`` the height of the opening; the bin's floor sits on the sim
    ground so its walls run from the ground up to the rim.
    """
    name: str
    kind: str          # "throw" | "push" (tint + telemetry label only)
    x: float
    y: float
    rim_z: float = 0.0
    half_w: float = 0.20   # inner half-width [m]


# Mirror of the app defaults (Config.THROW_GOAL_X/Y, THROW_VIZ_IMPACT_Z and
# push_skill.PUSH_BIN_TARGET_MAP["metal"]) for environments where the app
# config can't be imported (no ROS: Windows smoke). bins_from_config() is
# the real source; keep these in sync if those defaults move.
_FALLBACK_BINS = (
    SimBin("throw", "throw", 0.85, 0.00, 0.0),
    SimBin("push_metal", "push", 0.85, 0.60, 0.0),
)

CLASS_BIN_BY_OBJECT_CLASS = {
    "transparent": "throw",
    "metal": "push_metal",
}


def class_bin_name_for(class_name: str) -> str | None:
    """Correct sorting bin for an object class in the v1 RL objective."""
    return CLASS_BIN_BY_OBJECT_CLASS.get(str(class_name))


def resolved_object_reward(
    *,
    manipulated: bool,
    actual_bin_name: str | None,
    class_bin_name: str | None,
    collateral: bool,
) -> float:
    """Outcome-based: +1 in its class bin (picked or knocked in), -0.3 if picked
    or knocked off anywhere else, 0 if it just rides past."""
    if actual_bin_name is not None and actual_bin_name == class_bin_name:
        return 1.0
    return -0.3 if (manipulated or collateral) else 0.0


def bins_from_config(cfg=None) -> tuple:
    """Bins matching the REAL setup, derived from the app's own targets:

      * throw — ``Config.THROW_BINS`` (JSON list, same format the launch
        ``throw_bins:=`` takes) if set, else the single ``THROW_GOAL_X/Y``
        goal with its rim on the ``THROW_VIZ_IMPACT_Z`` landing plane;
      * push  — ``push_skill.PUSH_BIN_TARGET_MAP`` for every class that
        ``Config.SKILL_BY_CLASS`` routes to push.

    ``cfg`` is an app ``Config`` (pass the live one so env/launch overrides
    apply); ``None`` builds a default ``Config``. Falls back to
    :data:`_FALLBACK_BINS` when the app config is not importable.
    """
    try:
        if cfg is None:
            from gp8_control.config import Config
            cfg = Config()
        from gp8_control.skills.push_skill import PUSH_BIN_TARGET_MAP
    except Exception as exc:   # ImportError (no rclpy) or a config failure
        print(f"[sim] app config unavailable ({exc!r}); using fallback bins",
              flush=True)
        return _FALLBACK_BINS
    half_w = 0.5 * _env_float("GP8_SIM_BIN_W", 0.40)
    bins: list[SimBin] = []
    raw = (getattr(cfg, "THROW_BINS", "") or "").strip()
    if raw:
        for i, b in enumerate(json.loads(raw)):
            bins.append(SimBin(str(b.get("name", f"throw{i}")), "throw",
                               float(b["x"]), float(b["y"]),
                               float(b.get("z", cfg.THROW_VIZ_IMPACT_Z)), half_w))
    else:
        bins.append(SimBin("throw", "throw", float(cfg.THROW_GOAL_X),
                           float(cfg.THROW_GOAL_Y),
                           float(cfg.THROW_VIZ_IMPACT_Z), half_w))
    for cls, skill in dict(cfg.SKILL_BY_CLASS).items():
        xyz = PUSH_BIN_TARGET_MAP.get(cls)
        if skill == "push" and xyz is not None:
            bins.append(SimBin(f"push_{cls}", "push", float(xyz[0]),
                               float(xyz[1]), float(xyz[2]), half_w))
    return tuple(bins)


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


def _env_int_or_none(key: str) -> int | None:
    val = os.environ.get(key)
    if val is None or val == "":
        return None
    try:
        return int(val)
    except ValueError:
        return None


def _env_float_range(key: str, default: tuple[float, float]) -> tuple[float, float]:
    raw = os.environ.get(key)
    if not raw:
        return default
    parts = [p.strip() for p in raw.split(",", 1)]
    if len(parts) != 2:
        return default
    try:
        lo, hi = float(parts[0]), float(parts[1])
    except ValueError:
        return default
    return (lo, hi) if lo <= hi else default


def _env_suction_p(key: str, default: str = "") -> dict[str, float]:
    """``"0.9"`` -> {"*": 0.9} (every class); ``"metal:0.5,transparent:0.9"`` -> per class."""
    raw = os.environ.get(key, default).strip()
    if raw and ":" not in raw:
        return {"*": _clamp01(float(raw))}
    out = {}
    for part in raw.split(","):
        if ":" in part:
            name, p = part.split(":", 1)
            out[name.strip()] = _clamp01(float(p))
    return out


def _clamp01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


@dataclass
class SimConfig:
    """Belt/world parameters (env-overridable via GP8_SIM_*)."""

    belt_speed: float = field(default_factory=lambda: _env_float("GP8_SIM_BELT_SPEED", 0.12))
    spawn_interval: float = field(default_factory=lambda: _env_float("GP8_SIM_SPAWN_INTERVAL", 5.0))
    physical_belt: bool = field(default_factory=lambda: _env_bool("GP8_SIM_PHYSICAL_BELT", False))
    belt_actuator_speed_scale: float = field(
        default_factory=lambda: _env_float("GP8_SIM_BELT_ACTUATOR_SPEED_SCALE", 1.0))
    belt_contact_params: str = field(
        default_factory=lambda: os.environ.get("GP8_SIM_BELT_CONTACT_PARAMS", "current").strip().lower())
    realtime: bool = field(default_factory=lambda: _env_bool("GP8_SIM_REALTIME", True))
    randomize: bool = field(default_factory=lambda: _env_bool("GP8_SIM_RANDOMIZE", False))
    seed: int | None = field(default_factory=lambda: _env_int_or_none("GP8_SIM_SEED"))
    belt_speed_range: tuple[float, float] = field(
        default_factory=lambda: _env_float_range("GP8_SIM_BELT_SPEED_RANGE", (0.05, 0.20)))
    # Set rate; the 12-box pool and spawn clearance keep the real inflow well below it.
    spawn_rate_hz_range: tuple[float, float] = field(
        default_factory=lambda: _env_float_range("GP8_SIM_SPAWN_RATE_HZ_RANGE", (2.0, 4.0)))
    spawn_x_range: tuple[float, float] = field(
        default_factory=lambda: _env_float_range("GP8_SIM_SPAWN_X_RANGE", (0.30, 0.58)))
    random_class: bool = field(default_factory=lambda: _env_bool("GP8_SIM_RANDOM_CLASS", True))
    random_size: bool = field(default_factory=lambda: _env_bool("GP8_SIM_RANDOM_SIZE", True))
    random_yaw: bool = field(default_factory=lambda: _env_bool("GP8_SIM_RANDOM_YAW", True))
    object_half_x_range: tuple[float, float] = field(
        default_factory=lambda: _env_float_range("GP8_SIM_OBJECT_HALF_X_RANGE", (0.0375, 0.05)))
    object_half_y_range: tuple[float, float] = field(
        default_factory=lambda: _env_float_range("GP8_SIM_OBJECT_HALF_Y_RANGE", (0.05, 0.11)))
    object_half_z: float = field(default_factory=lambda: _env_float("GP8_SIM_OBJECT_HALF_Z", BOX_HALF_Z))
    object_mass: float = field(default_factory=lambda: _env_float("GP8_SIM_OBJECT_MASS", 0.2))
    spawn_clearance_margin: float = field(default_factory=lambda: _env_float("GP8_SIM_SPAWN_CLEARANCE_MARGIN", 0.03))
    spawn_max_tries: int = field(default_factory=lambda: int(_env_float("GP8_SIM_SPAWN_MAX_TRIES", 10)))
    bbox_mode: str = field(default_factory=lambda: os.environ.get("GP8_SIM_BBOX_MODE", "oriented").strip().lower())
    # Boxes enter the belt just UPSTREAM of the real camera's field of view
    # (nan = derived in SimCore from `camera`: view edge + a box + 5 cm), so
    # each box is first detected as it enters the image exactly like on
    # hardware and the detection->intercept lead time matches (~2.8 m /
    # belt_speed from the 2.47 m camera). GP8_SIM_SPAWN_Y overrides.
    spawn_y: float = field(default_factory=lambda: _env_float(
        "GP8_SIM_SPAWN_Y", float("nan")))
    # The real camera (pose from extrinsics, pinhole from config yaml).
    camera: SimCamera = field(default_factory=_default_camera)
    # Only boxes inside the camera image are reported (GP8_SIM_CAM_FOV_GATE=0
    # reports every on-belt box, the pre-camera behaviour).
    cam_fov_gate: bool = field(default_factory=lambda: os.environ.get(
        "GP8_SIM_CAM_FOV_GATE", "1").lower() not in ("0", "false", "no"))
    despawn_y: float = -0.8       # base-frame Y past which boxes are recycled
    lane_x: float = field(default_factory=lambda: _env_float("GP8_SIM_LANE_X", 0.45))
    grasp_z: float = 0.042        # box CENTRE height in base frame (pass cfg.GRASP_Z)
    aim_dz: float = 0.08
    classes: tuple = field(default_factory=lambda: tuple(
        c.strip() for c in os.environ.get("GP8_SIM_CLASSES", "transparent,metal").split(",")
        if c.strip()) or ("transparent",))
    # Suction seals on CONTACT, like the real cup: the cup face must be over
    # the box's top face (footprint + grab_xy_margin) and within grab_gap above
    # it. Pressing further crushes the box (deformable object + cup bellows):
    # its thickness follows the cup face down to crush_min_frac of the full
    # height, its bottom stays on the belt, and the weld engages only when the
    # cup stops descending — so the box never goes through the belt.
    grab_gap: float = field(default_factory=lambda: _env_float("GP8_SIM_GRAB_GAP", 0.008))
    grab_xy_margin: float = field(default_factory=lambda: _env_float("GP8_SIM_GRAB_XY_MARGIN", 0.01))
    crush_min_frac: float = field(default_factory=lambda: _env_float("GP8_SIM_CRUSH_MIN", 0.4))
    # Suction success probability per class (unlisted classes: "*" or 1.0); randomized
    # (RL) runs default to metal 0.5 / transparent 0.9.
    # GP8_SIM_METAL_SUCTION_BINARY overrides it for metal.
    suction_p: dict = field(default_factory=lambda: _env_suction_p(
        "GP8_SIM_SUCTION_P", "metal:0.5,transparent:0.9" if _env_bool("GP8_SIM_RANDOMIZE") else ""))
    metal_suction_binary: bool = field(default_factory=lambda: _env_bool("GP8_SIM_METAL_SUCTION_BINARY", False))
    metal_suction_mode: str = field(default_factory=lambda: os.environ.get(
        "GP8_SIM_METAL_SUCTION_MODE", "binary").strip().lower())
    # Stiffen the vendored position servos (kp=5000/kv=100 lags a fast throw).
    kp: float = field(default_factory=lambda: _env_float("GP8_SIM_KP", 12000.0))
    kv: float = field(default_factory=lambda: _env_float("GP8_SIM_KV", 220.0))
    det_hz: float = 15.0          # synthesized perception frame rate
    viewer: bool = field(default_factory=lambda: os.environ.get(
        "GP8_SIM_VIEWER", "0").lower() in ("1", "true", "yes"))
    # Headless video capture: set GP8_SIM_RECORD=/path/out.mp4 to have the
    # stepper render offscreen frames (MUJOCO_GL=egl/osmesa) into an mp4 —
    # lets a remote/SSH run produce a demo video with no display.
    record_path: str = field(default_factory=lambda: os.environ.get("GP8_SIM_RECORD", ""))
    record_fps: float = field(default_factory=lambda: _env_float("GP8_SIM_RECORD_FPS", 24.0))
    landing_log: bool = field(default_factory=lambda: _env_bool("GP8_SIM_LANDING_LOG", True))
    # Bins at the app's real throw goal / push targets (see bins_from_config;
    # the app passes its live Config). () = no bins. Width: GP8_SIM_BIN_W.
    bins: tuple = field(default_factory=bins_from_config)

    def __post_init__(self) -> None:
        if self.metal_suction_mode not in ("binary", "zero", "one"):
            self.metal_suction_mode = "binary"


_BIN_RGBA = {
    "throw": (0.30, 0.65, 1.00, 0.30),
    "push": (0.62, 0.62, 0.68, 0.30),
}
_BIN_WALL_T = 0.01          # wall / floor-plate thickness [m]
_BIN_FLOOR_Z = 0.005        # bin floor-plate centre above the sim ground [m]


def _add_bin(spec, base_pos, b: SimBin) -> None:
    """Open-top bin: floor plate on the ground + 4 walls up to the rim."""
    bx, by, bz = float(base_pos[0]), float(base_pos[1]), float(base_pos[2])
    rim = bz + b.rim_z                   # world z of the opening
    if rim <= 2 * _BIN_FLOOR_Z + 0.02:
        raise ValueError(f"sim bin {b.name!r}: rim_z {b.rim_z} is below the ground")
    body = spec.worldbody.add_body(name=f"sim_bin_{b.name}", pos=[bx + b.x, by + b.y, 0.0])
    rgba = list(_BIN_RGBA.get(b.kind, (0.8, 0.8, 0.2, 0.3)))
    hw, t = b.half_w, _BIN_WALL_T
    body.add_geom(name=f"sim_bin_{b.name}_floor", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[hw + t, hw + t, _BIN_FLOOR_Z], pos=[0.0, 0.0, _BIN_FLOOR_Z],
                  rgba=rgba)
    hh = 0.5 * rim
    for tag, pos, size in (
        ("xp", [hw + t / 2, 0.0, hh], [t / 2, hw + t, hh]),
        ("xn", [-hw - t / 2, 0.0, hh], [t / 2, hw + t, hh]),
        ("yp", [0.0, hw + t / 2, hh], [hw + t, t / 2, hh]),
        ("yn", [0.0, -hw - t / 2, hh], [hw + t, t / 2, hh]),
    ):
        body.add_geom(name=f"sim_bin_{b.name}_{tag}", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=size, pos=pos, rgba=rgba)


def _add_old_belt_box_contact_pairs(spec) -> None:
    for box_geom in BOX_GEOMS:
        spec.add_pair(
            geomname1=BELT_GEOM,
            geomname2=box_geom,
            friction=[0.7, 0.3, 0.05, 0.0001, 0.0001],
            solref=[0.002, 1.0],
            solimp=[0.99, 0.999, 0.001, 0.5, 2.0],
        )


def _pose_camera(spec, base_pos, cam: SimCamera) -> None:
    """Move the vendored d435i body so its <camera> child IS the real camera.

    MuJoCo cameras look along their -z with +y up; the optical frame looks
    along +z with +y down, so R_mj = R_opt @ diag(1, -1, -1). The vendored
    child camera is rotated pi about x w.r.t. the body (keeps the mesh's
    lens-forward relation), hence R_body = R_mj @ Rx(pi).
    """
    R_opt = np.asarray(cam.R)
    R_mj = R_opt @ np.diag([1.0, -1.0, -1.0])
    R_body = R_mj @ np.diag([1.0, -1.0, -1.0])
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, R_body.reshape(9))
    body = spec.body(MJ_CAMERA_BODY)
    body.pos = [float(base_pos[i]) + float(cam.pos[i]) for i in range(3)]
    body.quat = quat
    spec.camera(MJ_CAMERA).fovy = cam.fovy_deg


def _build_twin_model(scene_path: str, base_pos, lane_x: float, center_y: float,
                      grasp_z: float, half_len: float, bins=(), camera: SimCamera | None = None,
                      physical_belt: bool = False, object_half_z: float = BOX_HALF_Z,
                      belt_contact_params: str = "current"):
    """Load the vendored scene and add a reachable conveyor (+ bins) via MjSpec.

    The surface top sits one box-half below grasp_z so a box rests with its
    centre at grasp_z (== where the app's grasp pose and our detections put it).
    The surface runs along base Y centred at ``center_y`` with half-length
    ``half_len`` (both in the base frame). ``bins`` are :class:`SimBin`
    targets placed at their base-frame XY.
    Returns a compiled MjModel. Glue: never edits the vendored XML on disk.
    """
    spec = mujoco.MjSpec.from_file(scene_path)
    bx, by, bz = float(base_pos[0]), float(base_pos[1]), float(base_pos[2])
    top = bz + grasp_z - float(object_half_z)  # belt surface top (world z)
    thick = _PHYSICAL_BELT_HALF_Z if physical_belt else 0.02
    belt_x = _PHYSICAL_BELT_BASE_X if physical_belt else lane_x
    belt_y = _PHYSICAL_BELT_CENTER_Y if physical_belt else center_y
    belt = spec.worldbody.add_body(
        name=BELT_BODY, pos=[bx + belt_x, by + belt_y, top - thick],
    )
    if physical_belt:
        belt.explicitinertial = True
        belt.mass = 1.20269
        belt.inertia = [0.04236, 0.027888, 0.014472]
        belt.add_joint(
            name=BELT_JOINT,
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[0.0, -1.0, 0.0],
            range=[-1000.0, 1000.0],
            actfrcrange=[-5000.0, 5000.0],
        )
    belt.add_geom(
        name=BELT_GEOM, type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[
            _PHYSICAL_BELT_HALF_X if physical_belt else 0.16,
            _PHYSICAL_BELT_HALF_Y if physical_belt else half_len,
            thick,
        ],
        pos=[0.0, 0.0, 0.0],
        rgba=[0.12, 0.12, 0.14, 1.0], friction=[0.3, 0.02, 0.002],
    )
    if physical_belt:
        act = spec.add_actuator(
            name=BELT_ACT,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            target=BELT_JOINT,
            ctrlrange=[0.0, 10.0],
            forcerange=[-5000.0, 5000.0],
        )
        act.set_to_velocity(80000.0)
        if str(belt_contact_params).lower() == _BELT_CONTACT_OLD:
            _add_old_belt_box_contact_pairs(spec)
    for b in bins:
        _add_bin(spec, base_pos, b)
    if camera is not None:
        _pose_camera(spec, base_pos, camera)
    # Parked boxes are held by gravity compensation (see _park_box). The
    # compiler counts bodies with nonzero gravcomp (mjModel.ngravcomp) and the
    # passive-force pass skips the feature entirely when that count is 0, so
    # the flag must be non-zero at COMPILE time; _activate_box zeroes it on
    # spawn and _park_box restores it.
    for name in BOX_BODIES:
        spec.body(name).gravcomp = 1.0
    # Allow HD offscreen rendering (the recorder); default framebuffer is 640x480.
    spec.visual.global_.offwidth = max(int(spec.visual.global_.offwidth), 1280)
    spec.visual.global_.offheight = max(int(spec.visual.global_.offheight), 720)
    return spec.compile()


class SimBeltTracker:
    """ConveyorSpeedTracker-compatible belt state fed by the physics stepper.

    ``distance_at`` mirrors the hardware tracker's semantics: interpolate the
    (wall_t, cumulative distance) samples; past the last sample fall back to
    ``last + speed*dt`` so a pause never freezes tracked objects.
    """

    def __init__(self, speed: float, now_fn=None) -> None:
        self._speed = float(speed)
        self._now_fn = now_fn or time.time
        self._samples: deque = deque(maxlen=400)
        self._samples.append((self._now_fn(), 0.0))

    # -- fed by the stepper -------------------------------------------------
    def _push(self, wall_t: float, distance_m: float) -> None:
        self._samples.append((float(wall_t), float(distance_m)))

    def reset(self, wall_t: float = 0.0, distance_m: float = 0.0) -> None:
        self._samples.clear()
        self._samples.append((float(wall_t), float(distance_m)))

    # -- ConveyorSpeedTracker surface ---------------------------------------
    @property
    def current(self) -> float:
        return self._speed

    @property
    def distance_m(self) -> float | None:
        return self.distance_at(self._now_fn())

    def distance_at(self, wall_time: float) -> float | None:
        samples = list(self._samples)
        if not samples:
            return None
        if wall_time <= samples[0][0]:
            return samples[0][1]
        if wall_time >= samples[-1][0]:
            return samples[-1][1] + self._speed * float(wall_time - samples[-1][0])
        for i in range(1, len(samples)):
            t1, d1 = samples[i]
            if t1 >= wall_time:
                t0, d0 = samples[i - 1]
                frac = (wall_time - t0) / max(t1 - t0, 1e-9)
                return d0 + frac * (d1 - d0)
        return samples[-1][1]

    def check_freshness(self) -> None:
        return None   # the stepper feeds continuously; nothing to go stale


class _Recorder:
    """Offscreen mp4 capture driven from the stepper thread (headless demo).

    Lazily creates the ``mujoco.Renderer`` on the FIRST capture so the GL
    context lives on the stepper thread; a GL failure (no egl/osmesa) disables
    recording with one message instead of killing the sim.
    """

    _W, _H = 960, 540

    def __init__(self, core: "SimCore", path: str, fps: float) -> None:
        self._core = core
        self._path = path
        self._period = 1.0 / max(1.0, fps)
        self._fps = max(1.0, fps)
        self._next_t = 0.0
        self._renderer = None
        self._writer = None
        self._dead = False
        self._cam = mujoco.MjvCamera()
        self._cam.azimuth, self._cam.elevation, self._cam.distance = 135.0, -20.0, 2.8
        self._cam.lookat[:] = (0.45, 0.10, 0.55)
        self._frames = 0

    def maybe_capture(self) -> None:
        self._capture(lock_scene=True)

    def maybe_capture_locked(self) -> None:
        self._capture(lock_scene=False)

    def _capture(self, *, lock_scene: bool) -> None:
        if self._dead:
            return
        now = self._core.sim_time() if not self._core.cfg.realtime else time.monotonic()
        if now < self._next_t:
            return
        self._next_t = now + self._period
        try:
            if self._renderer is None:
                import cv2  # opencv-python (base dep) writes the mp4
                self._cv2 = cv2
                self._renderer = mujoco.Renderer(
                    self._core.model, height=self._H, width=self._W)
                self._writer = cv2.VideoWriter(
                    self._path, cv2.VideoWriter_fourcc(*"mp4v"),
                    self._fps, (self._W, self._H))
            if lock_scene:
                with self._core.lock:
                    self._renderer.update_scene(self._core.data, camera=self._cam)
            else:
                self._renderer.update_scene(self._core.data, camera=self._cam)
            frame = self._renderer.render()          # RGB, outside the lock
            self._writer.write(frame[:, :, ::-1])    # cv2 wants BGR
            self._frames += 1
        except Exception as exc:
            self._dead = True
            print(f"[sim-record] disabled ({exc!r}) — set MUJOCO_GL=egl or osmesa "
                  "for headless rendering", flush=True)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            print(f"[sim-record] wrote {self._frames} frames -> {self._path}", flush=True)
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass

    def next_time(self) -> float:
        if self._dead:
            return float("inf")
        return float(self._next_t)


class SimCore:
    """Owns MjModel/MjData + the wall-clock-servoed stepping thread.

    Concurrency: `lock` guards ALL MjData/MjModel mutation and is held per
    2 ms SUBSTEP (never across a catch-up batch), so ``set_suction`` waits at
    most one ``mj_step``. The stream engine's ``_emit_sample`` never takes it:
    samples go into a lock-free time-stamped queue that the stepper replays
    (see ``_pending_ctrl``). Joint snapshots are handed to the bound backend
    as fresh list objects (GIL-atomic swap, same pattern as the ROS
    joint_states callback).
    """

    _FREE, _ON_BELT, _GRABBED, _LOOSE = "free", "on_belt", "grabbed", "loose"

    def __init__(self, cfg: SimConfig | None = None) -> None:
        if mujoco is None:
            raise ImportError(
                "the MuJoCo sim backend needs the 'mujoco' package "
                "(uv sync --extra sim): " + repr(_MUJOCO_IMPORT_ERROR))
        scene = _find_scene()
        self.cfg = cfg or SimConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self._suction_rng = np.random.default_rng(None if self.cfg.seed is None else int(self.cfg.seed) + 1_000_003)
        self._random_spawn_rate_hz = 0.0
        self._next_random_spawn_time = float("inf")
        if self.cfg.randomize:
            self.cfg.belt_speed = float(self._rng.uniform(*self.cfg.belt_speed_range))
            self._random_spawn_rate_hz = float(self._rng.uniform(*self.cfg.spawn_rate_hz_range))
            if self._random_spawn_rate_hz > 0.0:
                self._next_random_spawn_time = float(
                    self._rng.exponential(1.0 / self._random_spawn_rate_hz)
                )

        # base pose is fixed in the XML (yaskawa_robot @ (-0.05,0,0.6), identity
        # rot); read it from a throwaway load to place the reachable belt.
        base = mujoco.MjModel.from_xml_path(str(scene)).body(MJ_BASE_BODY).pos
        if math.isnan(self.cfg.spawn_y):
            # just past the upstream edge of the camera image at box-top height
            top_z = self.cfg.grasp_z + BOX_HALF_Z
            self.cfg.spawn_y = self.cfg.camera.view_y_span(top_z)[1] + BOX_HALF_Y + 0.05
        # The surface spans despawn_y..spawn_y (+0.1 m margin each end) and
        # is centred BETWEEN them — not on the base origin, which would leave
        # the 2.47 m camera-reference spawn point hanging past the belt end.
        center_y = 0.5 * (self.cfg.spawn_y + self.cfg.despawn_y)
        half_len = 0.5 * (self.cfg.spawn_y - self.cfg.despawn_y) + 0.1
        self.bins: tuple = tuple(self.cfg.bins)
        self.model = _build_twin_model(
            str(scene), base, self.cfg.lane_x, center_y, self.cfg.grasp_z, half_len,
            self.bins, self.cfg.camera, self.cfg.physical_belt, self.cfg.object_half_z,
            self.cfg.belt_contact_params)
        # Landing telemetry: loose (thrown / pushed) boxes are classified when
        # they come down — inside a bin footprint below its rim, or a miss.
        self.bin_hits: dict[str, int] = {b.name: 0 for b in self.bins}
        self.bin_misses = 0
        self.data = mujoco.MjData(self.model)
        self.lock = threading.Lock()
        self._dt = float(self.model.opt.timestep)

        self._act_ids = [int(self.model.actuator(n).id) for n in MJ_ACTS]
        self._belt_act_id = (
            int(self.model.actuator(BELT_ACT).id) if self.cfg.physical_belt else None
        )
        self._belt_joint = (
            self.data.joint(BELT_JOINT) if self.cfg.physical_belt else None
        )
        # Stiffen the position servos so the arm tracks the streamed 4 ms
        # commands closely — otherwise it arrives at the grasp pose late and the
        # timing-sensitive ambush grasp misses. (position actuator:
        # gainprm[0]=kp, biasprm=[0,-kp,-kv].)
        for aid in self._act_ids:
            self.model.actuator_gainprm[aid][0] = self.cfg.kp
            self.model.actuator_biasprm[aid][1] = -self.cfg.kp
            self.model.actuator_biasprm[aid][2] = -self.cfg.kv
        self._grip_id = int(self.model.site(MJ_GRIP_SITE).id)
        self._link6_id = int(self.model.body(MJ_LINK6).id)

        # Seat the model at the app's startup posture (B at -pi/2) and hold it.
        for name, val in zip(MJ_JOINTS, _HOME):
            self.data.joint(name).qpos[0] = float(val)
        for aid, val in zip(self._act_ids, _HOME):
            self.data.ctrl[aid] = float(val)
        mujoco.mj_forward(self.model, self.data)
        self._base_p = self.data.body(MJ_BASE_BODY).xpos.copy()
        self._base_R = np.eye(3)   # yaskawa_robot has identity orientation
        self._belt_top_w = float(self._base_p[2]) + self.cfg.grasp_z - float(self.cfg.object_half_z)

        # --- streamed command timeline (written by _emit_sample at 250 Hz) --
        # Each sample is queued with its ARRIVAL wall time; per substep the
        # stepper applies the latest sample whose arrival time <= that
        # substep's wall time — the real JGPC's 4 ms zero-order hold, replayed
        # on the physics clock. When the stepper stalls (viewer sync, GC) and
        # catches up in a burst, the arm therefore traces the exact command
        # history instead of jumping to the newest sample. Lock-free: deque
        # append/popleft are GIL-atomic; _ctrl_target is swapped whole.
        # Suction toggles ride the SAME timeline (kind "suction"), so a release
        # fired while the stepper is stalled lands at its true place in the
        # command history rather than being applied early to a lagging arm.
        self._pending_ctrl: deque = deque(maxlen=4000)   # (t, kind, payload); ~16 s
        self._ctrl_target = np.array(_HOME, dtype=float)   # currently applied

        # --- object pool -----------------------------------------------------
        self._box_joints = [self.data.joint(n) for n in BOX_JOINTS]
        self._box_geom_ids = [int(self.model.geom(n).id) for n in BOX_GEOMS]
        self._box_body_ids = [int(self.model.body(n).id) for n in BOX_BODIES]
        self._weld_ids = [int(self.model.equality(n).id) for n in BOX_WELDS]
        self._box_state = [self._FREE] * len(BOX_JOINTS)
        self._box_class = [""] * len(BOX_JOINTS)
        self._box_suction_p = [1.0] * len(BOX_JOINTS)
        self._box_serial = [0] * len(BOX_JOINTS)   # spawn number, for the landing log
        self._box_manipulated = [False] * len(BOX_JOINTS)
        self._box_manipulated_skill: list[str | None] = [None] * len(BOX_JOINTS)
        self._box_class_bin_name: list[str | None] = [None] * len(BOX_JOINTS)
        self._box_sealed = [False] * len(BOX_JOINTS)        # welded at least once
        self._box_reward_done = [False] * len(BOX_JOINTS)   # reward already emitted
        # RL step that decided each box's fate (picked, or knocked off the belt);
        # the env sets step_index before each step.
        self.step_index = 0
        self._box_cause_step: list[int | None] = [None] * len(BOX_JOINTS)
        self._reward_events: list[dict] = []
        self._box_half_size = [
            np.array([BOX_HALF_X, BOX_HALF_Y, BOX_HALF_Z], dtype=float)
            for _ in BOX_JOINTS
        ]
        # Kinematic along-belt coordinate per ON_BELT box (world Y): contact
        # friction during a step brakes a purely velocity-asserted box (~0.10
        # realized vs 0.12 commanded), which would desync the boxes from the
        # encoder-distance integral and every ETA. Y is therefore DRIVEN;
        # X/Z stay dynamic (gravity, pushes, contacts).
        self._box_belt_y = [0.0] * len(BOX_JOINTS)
        self._grabbed: int | None = None
        self._vacuum_on = False            # suction armed (seals on contact)
        self._suction_attempt_ok: bool | None = None
        self._pressing: int | None = None  # box under the cup, being crushed
        self._grip_z_prev: float | None = None
        self._grip_vz = 0.0                # cup face vertical speed [m/s], world
        self._spawn_count = 0
        self._last_spawn = 0.0
        # Vendored collision masks, restored on spawn (parked boxes get 0/0).
        self._box_contype = [int(self.model.geom_contype[g]) for g in self._box_geom_ids]
        self._box_conaff = [int(self.model.geom_conaffinity[g]) for g in self._box_geom_ids]
        for i in range(len(BOX_JOINTS)):
            self._park_box(i)

        # --- world-source outputs -------------------------------------------
        self.belt = SimBeltTracker(
            self.cfg.belt_speed,
            now_fn=(self.sim_time if not self.cfg.realtime else time.time),
        )
        if not self.cfg.realtime:
            self.belt.reset(0.0, 0.0)
        self.latest_snapshot: dict | None = None   # atomic ref swap by stepper
        self._encoder_distance = 0.0
        self._last_det_pub = 0.0

        # --- stepping thread -------------------------------------------------
        self._backend: "MujocoRobotBackend" | None = None
        self._running = False
        self._first_step = threading.Event()
        self._thread: threading.Thread | None = None
        self._viewer = None
        self._recorder = (
            _Recorder(self, self.cfg.record_path, self.cfg.record_fps)
            if self.cfg.record_path else None
        )

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        if not self.cfg.realtime:
            self._last_spawn = float(self.data.time)
            self.advance_steps(1)
            return
        self._thread = threading.Thread(
            target=self._stepper, name="gp8_sim_stepper", daemon=True)
        self._thread.start()

    def stop(self, timeout_sec: float = 2.0) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=timeout_sec)
            self._thread = None
        elif self._recorder is not None:
            self._recorder.close()
            self._recorder = None

    def wait_ready(self, timeout_sec: float = 10.0) -> bool:
        return self._first_step.wait(timeout=timeout_sec)

    def sim_time(self) -> float:
        return float(self.data.time)

    def _next_physical_step_target(self, deadline: float) -> float:
        target = float(deadline)
        if self.cfg.physical_belt:
            det_period = 1.0 / max(1.0, self.cfg.det_hz)
            if self._last_det_pub > 0.0:
                target = min(target, self._last_det_pub + det_period)
            if not self.cfg.randomize and self.cfg.spawn_interval > 0.0:
                target = min(target, self._last_spawn + self.cfg.spawn_interval)
        if self._recorder is not None:
            target = min(target, self._recorder.next_time())
        return max(self.sim_time() + self._dt, target)

    def advance_until(self, deadline: float) -> None:
        deadline = float(deadline)
        while self._running and self.sim_time() < deadline:
            target = self._next_physical_step_target(deadline)
            remaining = max(0.0, target - self.sim_time())
            n = max(1, int(math.ceil(remaining / self._dt)))
            self.advance_steps(min(n, 50 if self.cfg.physical_belt else 250))

    def advance_seconds(self, duration: float) -> None:
        duration = max(0.0, float(duration))
        if duration <= 0.0:
            return
        self.advance_until(self.sim_time() + duration)

    def advance_steps(self, n: int = 1) -> None:
        n = max(0, int(n))
        if n <= 0:
            return
        det_period = 1.0 / max(1.0, self.cfg.det_hz)
        pend = self._pending_ctrl
        for _ in range(n):
            suction_events = []
            with self.lock:
                t_sub = float(self.data.time)
                while pend and pend[0][0] <= t_sub:
                    _, kind, payload = pend.popleft()
                    if kind == "ctrl":
                        self._ctrl_target = payload
                    else:
                        suction_events.append(payload)
                for on in suction_events:
                    self._apply_suction(on)
                self._step_once_locked()
        self._post_step_batch(n, float(self.data.time), det_period)
        self._first_step.set()

    def bind_backend(self, backend: "MujocoRobotBackend") -> None:
        self._backend = backend

    def mark_manipulated(
        self,
        sim_object_id: int | None,
        skill_name: str,
        class_bin_name: str | None,
    ) -> bool:
        """Mark the sim box corresponding to a committed control target."""
        if sim_object_id is None:
            return False
        with self.lock:
            for i, serial in enumerate(self._box_serial):
                if int(serial) != int(sim_object_id):
                    continue
                if self._box_state[i] == self._FREE:
                    return False
                self._box_manipulated[i] = True
                self._box_cause_step[i] = self.step_index
                self._box_manipulated_skill[i] = str(skill_name)
                self._box_class_bin_name[i] = class_bin_name
                return True
        return False

    def resolve_unsealed_pick(self, sim_object_id: int | None) -> bool:
        """After a suction pick: if the box never sealed, emit its manipulated-miss
        reward now instead of when it later rides off the belt end (no repeat)."""
        if sim_object_id is None:
            return False
        with self.lock:
            for i, serial in enumerate(self._box_serial):
                if int(serial) != int(sim_object_id) or self._box_state[i] == self._FREE:
                    continue
                if self._box_sealed[i] or self._box_reward_done[i]:
                    return False
                base = self._to_base(self._box_joints[i].qpos[0:3])
                self._emit_reward_event(i, base, None, collateral=False)
                self._box_reward_done[i] = True
                return True
        return False

    def is_object_live(self, sim_object_id: int | None) -> bool:
        """Whether a sim object id still refers to an active physical box."""
        if sim_object_id is None:
            return False
        with self.lock:
            for i, serial in enumerate(self._box_serial):
                if int(serial) == int(sim_object_id):
                    return self._box_state[i] != self._FREE
        return False

    def pop_reward_events(self) -> list[dict]:
        """Return and clear resolved-object reward events."""
        with self.lock:
            events = list(self._reward_events)
            self._reward_events.clear()
        return events

    # ------------------------------------------------------------------
    # RobotBackend hooks (called from the app/skill thread)
    # ------------------------------------------------------------------
    def set_ctrl_target(self, positions) -> None:
        """Queue one 4 ms command sample (stream thread). Never blocks."""
        stamp = self.sim_time() if not self.cfg.realtime else time.monotonic()
        self._pending_ctrl.append((stamp, "ctrl", np.array(positions, dtype=float)))

    def set_suction(self, on: bool) -> None:
        """Queue a suction toggle (skill/IO thread). Applied by the stepper at
        its wall-time slot in the command timeline — see ``_apply_suction``."""
        stamp = self.sim_time() if not self.cfg.realtime else time.monotonic()
        self._pending_ctrl.append((stamp, "suction", bool(on)))

    def _apply_suction(self, on: bool) -> None:
        """Arm the vacuum on suction ON (it seals on contact, see
        ``_vacuum_tick``); release the weld on OFF. Stepper thread, under lock.

        The weld physically drags the box through the swing, so at release its
        free-joint qvel already carries the true throw velocity — it flies
        ballistically with no synthesized fling.
        """
        self._vacuum_on = bool(on)
        if on:
            self._suction_attempt_ok = None
            self._vacuum_tick()
        else:
            self._suction_attempt_ok = None
            self._pressing = None
            self._release_grabbed()

    def _box_under_cup(self) -> int | None:
        """The on-belt box whose top face the cup is over and touching/inside."""
        grip = self.data.site(self._grip_id).xpos
        m = self.cfg.grab_xy_margin
        for i, jnt in enumerate(self._box_joints):
            if self._box_state[i] != self._ON_BELT:
                continue
            body = self.data.body(self._box_body_ids[i])
            d = body.xmat.reshape(3, 3).T @ (grip - body.xpos)     # cup in box frame
            hx, hy, hz = [float(v) for v in self._box_half_size[i]]
            if abs(d[0]) > hx + m or abs(d[1]) > hy + m:
                continue
            half_z = float(self.model.geom_size[self._box_geom_ids[i]][2])
            gap = float(grip[2] - (body.xpos[2] + half_z))          # cup face above top
            if -2.0 * hz <= gap <= self.cfg.grab_gap:
                return i
        return None

    def _vacuum_tick(self) -> None:
        """Per substep while the vacuum is armed and nothing is welded yet.

        Contact -> the box is being PRESSED: its thickness follows the cup face
        down (to crush_min_frac), its bottom is held on the belt, and the cup
        may sink further into it (bellows). The weld engages the moment the cup
        stops descending (press over, or a parked cup the box slid under), at
        the crushed pose — so the carried box's centre sits ~at the cup face
        rather than hanging below it, and it never penetrates the belt.
        """
        if not self._vacuum_on or self._grabbed is not None:
            return
        i = self._box_under_cup()
        self._pressing = i
        if i is None:
            return                       # vacuum sucking air — silent miss
        jnt = self._box_joints[i]
        gid = self._box_geom_ids[i]
        cup_z = float(self.data.site(self._grip_id).xpos[2])
        orig_half_z = float(self._box_half_size[i][2])
        min_half = orig_half_z * self.cfg.crush_min_frac
        half = float(np.clip(0.5 * (cup_z - self._belt_top_w), min_half, orig_half_z))
        self.model.geom_size[gid][2] = half
        jnt.qpos[2] = self._belt_top_w + half          # bottom stays on the belt
        jnt.qvel[2] = 0.0
        if self._grip_vz >= -0.01:                      # cup no longer descending
            if self._suction_attempt_ok is None:
                p = float(np.clip(self._box_suction_p[i], 0.0, 1.0))
                self._suction_attempt_ok = bool(float(self._suction_rng.random()) < p)
            if not self._suction_attempt_ok:
                return
            self._weld_box(i)
            self._pressing = None

    def _weld_box(self, best: int) -> None:
        # Write the ACTIVATION-TIME relative pose into eq_data before enabling:
        # the XML weld's zero relpose was resolved at compile time from qpos0,
        # which would snap the box to its spawn pose relative to link6.
        eqid = self._weld_ids[best]
        b1 = self.data.body(self._link6_id)
        b2 = self.data.body(self._box_body_ids[best])
        r1 = b1.xmat.reshape(3, 3)
        relpos = r1.T @ (b2.xpos - b1.xpos)
        q1_inv = np.empty(4)
        mujoco.mju_negQuat(q1_inv, b1.xquat)
        relquat = np.empty(4)
        mujoco.mju_mulQuat(relquat, q1_inv, b2.xquat)
        self.model.eq_data[eqid][:] = 0.0
        self.model.eq_data[eqid][3:6] = relpos
        self.model.eq_data[eqid][6:10] = relquat
        self.model.eq_data[eqid][10] = 1.0   # torquescale
        self.data.eq_active[eqid] = 1
        self._box_state[best] = self._GRABBED
        self._box_sealed[best] = True
        self._grabbed = best

    def _release_grabbed(self) -> None:
        if self._grabbed is None:
            return
        self.data.eq_active[self._weld_ids[self._grabbed]] = 0
        self._box_state[self._grabbed] = self._LOOSE
        self._grabbed = None
        if self._pressing is not None and self._box_state[self._pressing] != self._ON_BELT:
            self._pressing = None

    # ------------------------------------------------------------------
    # belt / object lifecycle (stepper-owned, under self.lock)
    # ------------------------------------------------------------------
    def _to_base(self, world_xyz) -> np.ndarray:
        return self._base_R.T @ (np.asarray(world_xyz) - self._base_p)

    def _to_world(self, base_xyz) -> np.ndarray:
        return self._base_p + self._base_R @ np.asarray(base_xyz)

    @staticmethod
    def _yaw_quat(theta: float) -> tuple[float, float, float, float]:
        return (math.cos(0.5 * theta), 0.0, 0.0, math.sin(0.5 * theta))

    @staticmethod
    def _yaw_from_xmat(xmat) -> float:
        mat = np.asarray(xmat, dtype=float).reshape(3, 3)
        return float(math.atan2(mat[1, 0], mat[0, 0]))

    @staticmethod
    def _boxes_overlap_2d(c1, h1, th1: float, c2, h2, th2: float) -> bool:
        d = np.asarray(c2, dtype=float)[:2] - np.asarray(c1, dtype=float)[:2]
        u1 = np.array([math.cos(th1), math.sin(th1)])
        v1 = np.array([-math.sin(th1), math.cos(th1)])
        u2 = np.array([math.cos(th2), math.sin(th2)])
        v2 = np.array([-math.sin(th2), math.cos(th2)])
        for axis in (u1, v1, u2, v2):
            r1 = float(h1[0]) * abs(float(u1 @ axis)) + float(h1[1]) * abs(float(v1 @ axis))
            r2 = float(h2[0]) * abs(float(u2 @ axis)) + float(h2[1]) * abs(float(v2 @ axis))
            if abs(float(d @ axis)) > r1 + r2:
                return False
        return True

    def _spawn_pose_is_clear(self, base_x: float, base_y: float, half_size, yaw: float) -> bool:
        margin = max(0.0, float(self.cfg.spawn_clearance_margin))
        cand_h = (float(half_size[0]) + margin, float(half_size[1]) + margin)
        cand_c = np.array([base_x, base_y], dtype=float)
        for i, state in enumerate(self._box_state):
            if state == self._FREE:
                continue
            body = self.data.body(self._box_body_ids[i])
            base = self._to_base(body.xpos)
            h = self._box_half_size[i]
            ex_h = (float(h[0]) + margin, float(h[1]) + margin)
            if self._boxes_overlap_2d(cand_c, cand_h, yaw, base[:2], ex_h, self._yaw_from_xmat(body.xmat)):
                return False
        return True

    def _sample_box_half_size(self) -> np.ndarray:
        if not (self.cfg.randomize and self.cfg.random_size):
            return np.array([BOX_HALF_X, BOX_HALF_Y, BOX_HALF_Z], dtype=float)
        hx = float(self._rng.uniform(*self.cfg.object_half_x_range))
        hy = float(self._rng.uniform(*self.cfg.object_half_y_range))
        hz = float(self.cfg.object_half_z)
        return np.array([hx, hy, hz], dtype=float)

    def _apply_box_size(self, i: int, half_size) -> None:
        hx, hy, hz = [float(v) for v in half_size]
        gid = self._box_geom_ids[i]
        bid = self._box_body_ids[i]
        self.model.geom_size[gid] = (hx, hy, hz)
        self.model.geom_rbound[gid] = float(math.sqrt(hx * hx + hy * hy + hz * hz))
        mass = float(self.cfg.object_mass)
        self.model.body_mass[bid] = mass
        self.model.body_inertia[bid] = (
            mass / 3.0 * (hy * hy + hz * hz),
            mass / 3.0 * (hx * hx + hz * hz),
            mass / 3.0 * (hx * hx + hy * hy),
        )
        self._box_half_size[i] = np.array([hx, hy, hz], dtype=float)

    def _sample_spawn_base_x(self, half_size, yaw: float) -> float | None:
        if not self.cfg.randomize:
            return float(self.cfg.lane_x + 0.06 * ((self._spawn_count % 3) - 1))
        tries = max(1, int(self.cfg.spawn_max_tries))
        for _ in range(tries):
            x = float(self._rng.uniform(*self.cfg.spawn_x_range))
            if self._spawn_pose_is_clear(x, self.cfg.spawn_y, half_size, yaw):
                return x
        return None

    def _sample_spawn_class(self) -> str:
        classes = tuple(self.cfg.classes) or ("transparent",)
        if self.cfg.randomize and self.cfg.random_class:
            return str(classes[int(self._rng.integers(0, len(classes)))])
        return str(classes[self._spawn_count % len(classes)])

    def _sample_suction_p(self, class_name: str) -> float:
        if self.cfg.metal_suction_binary and str(class_name) == "metal":
            if self.cfg.metal_suction_mode == "zero":
                return 0.0
            if self.cfg.metal_suction_mode == "one":
                return 1.0
            return float(self._suction_rng.integers(0, 2))
        p = self.cfg.suction_p
        return float(p.get(str(class_name), p.get("*", 1.0)))

    def _box_rgba(self, class_name: str, suction_p: float) -> tuple:
        if self.cfg.metal_suction_binary and str(class_name) == "metal":
            return (
                _METAL_SUCTION_OK_RGBA
                if float(suction_p) >= 0.5
                else _METAL_SUCTION_FAIL_RGBA
            )
        return _CLASS_RGBA.get(str(class_name), _DEFAULT_RGBA)

    def _schedule_next_random_spawn(self, now: float) -> None:
        if self.cfg.randomize and self._random_spawn_rate_hz > 0.0:
            self._next_random_spawn_time = float(now) + float(
                self._rng.exponential(1.0 / self._random_spawn_rate_hz)
            )
        else:
            self._next_random_spawn_time = float("inf")

    def _spawn_box(self) -> None:
        idx = next((i for i, s in enumerate(self._box_state) if s == self._FREE), None)
        if idx is None:
            return
        cls = self._sample_spawn_class()
        half_size = self._sample_box_half_size()
        yaw = float(self._rng.uniform(-math.pi, math.pi)) if (self.cfg.randomize and self.cfg.random_yaw) else 0.0
        spawn_x = self._sample_spawn_base_x(half_size, yaw)
        if spawn_x is None:
            return
        suction_p = self._sample_suction_p(cls)
        world = self._to_world([spawn_x, self.cfg.spawn_y, self._belt_top_w - self._base_p[2] + float(half_size[2])])
        self._activate_box(idx)
        self._apply_box_size(idx, half_size)
        jnt = self._box_joints[idx]
        jnt.qpos[0:3] = world
        jnt.qpos[3:7] = self._yaw_quat(yaw)
        jnt.qvel[:] = 0.0
        self._box_state[idx] = self._ON_BELT
        self._box_class[idx] = cls
        self._box_suction_p[idx] = suction_p
        self._box_belt_y[idx] = float(world[1])
        self.model.geom_rgba[self._box_geom_ids[idx]] = self._box_rgba(cls, suction_p)
        self._spawn_count += 1
        self._box_serial[idx] = self._spawn_count
        self._box_manipulated[idx] = False
        self._box_manipulated_skill[idx] = None
        self._box_class_bin_name[idx] = None
        self._box_sealed[idx] = False
        self._box_reward_done[idx] = False
        self._box_cause_step[idx] = None

    def _park_box(self, i: int) -> None:
        """Retire box i: collisions off, gravity compensated, stashed below the floor."""
        jnt = self._box_joints[i]
        jnt.qpos[0:3] = _park_pos(i)
        jnt.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        jnt.qvel[:] = 0.0
        gid = self._box_geom_ids[i]
        self._apply_box_size(i, (BOX_HALF_X, BOX_HALF_Y, BOX_HALF_Z))
        self.model.geom_contype[gid] = 0
        self.model.geom_conaffinity[gid] = 0
        self.model.body_gravcomp[self._box_body_ids[i]] = 1.0
        self._box_state[i] = self._FREE
        self._box_class[i] = ""
        self._box_suction_p[i] = 1.0
        self._box_manipulated[i] = False
        self._box_manipulated_skill[i] = None
        self._box_class_bin_name[i] = None
        self._box_sealed[i] = False
        self._box_reward_done[i] = False
        self._box_cause_step[i] = None

    def _activate_box(self, i: int) -> None:
        """Re-enable a parked box's collisions/gravity before placing it."""
        gid = self._box_geom_ids[i]
        self.model.geom_contype[gid] = self._box_contype[i]
        self.model.geom_conaffinity[gid] = self._box_conaff[i]
        self.model.body_gravcomp[self._box_body_ids[i]] = 0.0

    def _apply_belt_velocity(self) -> None:
        """Pre-step: assert on-belt boxes' world-Y velocity (solver hint so
        contacts see the conveyor motion). The authoritative advance happens in
        :meth:`_enforce_belt_kinematics` after the step."""
        if self.cfg.physical_belt:
            if self._belt_act_id is not None:
                self.data.ctrl[self._belt_act_id] = (
                    self.cfg.belt_speed * self.cfg.belt_actuator_speed_scale
                )
            if self._belt_joint is not None:
                self._belt_joint.qvel[0] = self.cfg.belt_speed * self.cfg.belt_actuator_speed_scale
            return
        floor = self._base_p[2] + self.cfg.grasp_z - 0.03
        for i, jnt in enumerate(self._box_joints):
            if self._box_state[i] != self._ON_BELT:
                continue
            if jnt.qpos[2] >= floor:
                jnt.qvel[1] = -self.cfg.belt_speed   # conveyor motion (world -Y)

    def _enforce_belt_kinematics(self, dt: float) -> None:
        """Post-step: DRIVE each on-belt box's world-Y so it advances at exactly
        the belt speed — contact friction inside mj_step brakes a merely
        velocity-asserted box (~0.10 realized vs 0.12 commanded), which would
        desync the boxes from the encoder-distance integral and every ETA.
        X/Z stay fully dynamic (gravity, pushes, contacts)."""
        if self.cfg.physical_belt:
            return
        floor = self._base_p[2] + self.cfg.grasp_z - 0.03
        for i, jnt in enumerate(self._box_joints):
            if self._box_state[i] != self._ON_BELT:
                continue
            if jnt.qpos[2] >= floor:
                self._box_belt_y[i] -= self.cfg.belt_speed * dt
                jnt.qpos[1] = self._box_belt_y[i]
                jnt.qvel[1] = -self.cfg.belt_speed

    def _bin_at(self, base_xyz) -> SimBin | None:
        """The bin whose footprint contains base_xyz below its rim, if any."""
        for b in self.bins:
            if (abs(base_xyz[0] - b.x) <= b.half_w and abs(base_xyz[1] - b.y) <= b.half_w
                    and base_xyz[2] < b.rim_z):
                return b
        return None

    def _log_landing(self, i: int, base) -> str | None:
        tag = f"{self._box_class[i] or 'box'} #{self._box_serial[i]}"
        hit = self._bin_at(base)
        if hit is not None:
            self.bin_hits[hit.name] += 1
            if self.cfg.landing_log:
                print(f"[sim] {tag} landed IN bin '{hit.name}' "
                      f"({base[0]:+.2f}, {base[1]:+.2f})  hits={self.bin_hits}", flush=True)
            return hit.name
        self.bin_misses += 1
        near = ""
        if self.bins:
            b = min(self.bins, key=lambda b: math.hypot(base[0] - b.x, base[1] - b.y))
            near = f"  nearest '{b.name}' d={math.hypot(base[0] - b.x, base[1] - b.y):.2f}m"
        if self.cfg.landing_log:
            print(f"[sim] {tag} MISSED ({base[0]:+.2f}, {base[1]:+.2f}){near}  "
                  f"misses={self.bin_misses}", flush=True)
        return None

    def _emit_reward_event(self, i: int, base, actual_bin_name: str | None,
                           collateral: bool) -> None:
        if self._box_reward_done[i]:
            return
        class_name = self._box_class[i]
        manipulated = bool(self._box_manipulated[i])
        class_bin_name = (
            self._box_class_bin_name[i]
            if self._box_class_bin_name[i] is not None
            else class_bin_name_for(class_name)
        )
        reward = resolved_object_reward(
            manipulated=manipulated,
            actual_bin_name=actual_bin_name,
            class_bin_name=class_bin_name,
            collateral=collateral,
        )
        self._reward_events.append({
            "sim_time": float(self.sim_time()),
            "sim_object_id": int(self._box_serial[i]),
            "class_name": class_name,
            "manipulated": manipulated,
            "manipulated_skill": self._box_manipulated_skill[i],
            "class_bin_name": class_bin_name,
            "actual_bin_name": actual_bin_name,
            "reward": reward,
            "base_xyz": [float(base[0]), float(base[1]), float(base[2])],
            "collateral": bool(collateral),
            "cause_step": self._box_cause_step[i],
        })

    def _recycle_boxes(self) -> None:
        """Retire boxes off the belt end / on the floor; flag ones knocked off.

        Loose boxes (thrown / pushed) are recycled once they come down — on
        the floor (miss) or at the bottom of a bin (hit) — and logged.
        """
        off_belt = self._base_p[2] + self.cfg.grasp_z - 0.05
        fallen = self._base_p[2] - 0.4   # ~floor level, world z
        for i, jnt in enumerate(self._box_joints):
            if self._box_state[i] in (self._FREE, self._GRABBED):
                continue
            world = jnt.qpos[0:3]
            base = self._to_base(world)
            loose = self._box_state[i] == self._LOOSE
            in_bin = loose and self._bin_at(base) is not None
            bin_bottom = fallen
            gone = (base[1] < self.cfg.despawn_y or abs(base[0]) > 1.6
                    or (world[2] < bin_bottom if in_bin else world[2] < fallen))
            if gone:
                actual_bin_name = None
                if loose:
                    actual_bin_name = self._log_landing(i, base)
                collateral = loose and not self._box_manipulated[i]
                self._emit_reward_event(i, base, actual_bin_name, collateral)
                self._park_box(i)
            elif self._box_state[i] == self._ON_BELT and world[2] < off_belt:
                self._box_state[i] = self._LOOSE   # pushed / knocked off the belt
                if not self._box_manipulated[i]:
                    self._box_cause_step[i] = self.step_index

    # ------------------------------------------------------------------
    # synthesized perception (schema-v2, through the real transform code)
    # ------------------------------------------------------------------
    def _build_snapshot(self, now: float) -> dict:
        ref_x, ref_y, ref_z = (extrinsics.REFERENCE_X_BASE,
                               extrinsics.REFERENCE_Y_BASE,
                               extrinsics.REFERENCE_Z_BASE)
        sx = extrinsics.SIGN_CX_TO_BASE_X * extrinsics.SCALE_CX_TO_BASE_X
        sy = extrinsics.SIGN_CY_TO_BASE_Y * extrinsics.SCALE_CY_TO_BASE_Y
        offset_grasp = extrinsics.DETECTION_OFFSET_GRASP
        offset_aim = extrinsics.DETECTION_OFFSET_AIM
        ws_x_abs = extrinsics.WORKSPACE_X_ABS

        dets = []
        cam = self.cfg.camera if self.cfg.cam_fov_gate else None
        for i, jnt in enumerate(self._box_joints):
            if self._box_state[i] != self._ON_BELT:
                continue
            b = self._to_base(jnt.qpos[0:3])
            # The real detector only reports what is inside the image: gate on
            # the box's top-face centre. Boxes past the view are then tracked
            # by the app's belt dead reckoning, exactly as on hardware.
            hx, hy, hz = [float(v) for v in self._box_half_size[i]]
            if cam is not None and not cam.sees((b[0], b[1], b[2] + hz)):
                continue
            # Base -> belt-frame camera coords (the inverse of camera_debug's
            # constant-translation mapping), then run the corners through the
            # SAME bbox_to_base the real pipeline uses (zero latency in sim).
            cx = (float(b[0]) - ref_x) / sx
            cy = (float(b[1]) - ref_y) / sy
            body = self.data.body(self._box_body_ids[i])
            axes = body.xmat.reshape(3, 3)
            footprint = []
            for sx_box, sy_box in ((-1.0, -1.0), (-1.0, 1.0), (1.0, 1.0), (1.0, -1.0)):
                corner_world = body.xpos + sx_box * hx * axes[:, 0] + sy_box * hy * axes[:, 1]
                footprint.append(self._to_base(corner_world))
            if self.cfg.bbox_mode in ("axis", "axis_aligned", "aabb"):
                pts = np.asarray(footprint, dtype=float)
                x0, y0 = np.min(pts[:, :2], axis=0)
                x1, y1 = np.max(pts[:, :2], axis=0)
                footprint = [
                    np.array([x0, y0, b[2]], dtype=float),
                    np.array([x0, y1, b[2]], dtype=float),
                    np.array([x1, y1, b[2]], dtype=float),
                    np.array([x1, y0, b[2]], dtype=float),
                ]
            cam_corners = []
            for corner_base in footprint:
                cam_corners.append([
                    (float(corner_base[0]) - ref_x) / sx,
                    (float(corner_base[1]) - ref_y) / sy,
                    0.0,
                ])
            cam_bbox = np.array(cam_corners, dtype=float)
            kw = dict(ref_x=ref_x, ref_y=ref_y, ref_z=ref_z,
                      scale_x=sx, scale_y=sy, y_back_projection=0.0)
            x_base = ref_x + sx * cx
            y_base = ref_y + sy * cy
            dets.append({
                "class": self._box_class[i],
                "confidence": 0.9,
                "suction_p": float(self._box_suction_p[i]),
                "sim_object_id": int(self._box_serial[i]),
                "cam": [float(cx), float(cy), 0.0],
                "cam_bbox": cam_bbox.tolist(),
                "base_grasp": [x_base, y_base, ref_z + offset_grasp],
                "base_aim": [x_base, y_base, ref_z + offset_aim],
                "base_bbox_grasp": bbox_to_base(cam_bbox, z_offset=offset_grasp, **kw).tolist(),
                "base_bbox_aim": bbox_to_base(cam_bbox, z_offset=offset_aim, **kw).tolist(),
                "in_workspace": bool(-ws_x_abs < cx < ws_x_abs),
            })
        return {
            "receipt_time": now,               # strictly increasing per frame
            "belt_mps": self.cfg.belt_speed,
            "spawn_rate_hz": (
                self._random_spawn_rate_hz
                if self.cfg.randomize
                else (1.0 / self.cfg.spawn_interval if self.cfg.spawn_interval > 0.0 else 0.0)
            ),
            "perception_delay_s": 0.0,
            "applied_delay_s": 0.0,
            "latency_mode": "sim",
            "physical_belt": bool(self.cfg.physical_belt),
            "detections": dets,
        }

    def truth_detection_for(self, sim_object_id: int | None) -> dict | None:
        """Return a schema-v2 detection for a live sim object, without FOV gating.

        This is a simulator-only RL hook. It does not create tracks; callers use
        it to refresh already-tracked objects by simulator identity.
        """
        with self.lock:
            return self._truth_detection_for_locked(sim_object_id)

    def _truth_detection_for_locked(self, sim_object_id: int | None) -> dict | None:
        if sim_object_id is None:
            return None
        ref_x, ref_y, ref_z = (
            extrinsics.REFERENCE_X_BASE,
            extrinsics.REFERENCE_Y_BASE,
            extrinsics.REFERENCE_Z_BASE,
        )
        sx = extrinsics.SIGN_CX_TO_BASE_X * extrinsics.SCALE_CX_TO_BASE_X
        sy = extrinsics.SIGN_CY_TO_BASE_Y * extrinsics.SCALE_CY_TO_BASE_Y
        offset_grasp = extrinsics.DETECTION_OFFSET_GRASP
        offset_aim = extrinsics.DETECTION_OFFSET_AIM
        ws_x_abs = extrinsics.WORKSPACE_X_ABS

        for i, jnt in enumerate(self._box_joints):
            if int(self._box_serial[i]) != int(sim_object_id):
                continue
            if self._box_state[i] != self._ON_BELT:
                return None

            b = self._to_base(jnt.qpos[0:3])
            hx, hy, _hz = [float(v) for v in self._box_half_size[i]]
            cx = (float(b[0]) - ref_x) / sx
            cy = (float(b[1]) - ref_y) / sy
            body = self.data.body(self._box_body_ids[i])
            axes = body.xmat.reshape(3, 3)
            footprint = []
            for sx_box, sy_box in (
                (-1.0, -1.0),
                (-1.0, 1.0),
                (1.0, 1.0),
                (1.0, -1.0),
            ):
                corner_world = body.xpos + sx_box * hx * axes[:, 0] + sy_box * hy * axes[:, 1]
                footprint.append(self._to_base(corner_world))
            if self.cfg.bbox_mode in ("axis", "axis_aligned", "aabb"):
                pts = np.asarray(footprint, dtype=float)
                x0, y0 = np.min(pts[:, :2], axis=0)
                x1, y1 = np.max(pts[:, :2], axis=0)
                footprint = [
                    np.array([x0, y0, b[2]], dtype=float),
                    np.array([x0, y1, b[2]], dtype=float),
                    np.array([x1, y1, b[2]], dtype=float),
                    np.array([x1, y0, b[2]], dtype=float),
                ]
            cam_bbox = np.array(
                [
                    [
                        (float(corner_base[0]) - ref_x) / sx,
                        (float(corner_base[1]) - ref_y) / sy,
                        0.0,
                    ]
                    for corner_base in footprint
                ],
                dtype=float,
            )
            kw = dict(
                ref_x=ref_x,
                ref_y=ref_y,
                ref_z=ref_z,
                scale_x=sx,
                scale_y=sy,
                y_back_projection=0.0,
            )
            x_base = ref_x + sx * cx
            y_base = ref_y + sy * cy
            return {
                "class": self._box_class[i],
                "confidence": 0.9,
                "suction_p": float(self._box_suction_p[i]),
                "sim_object_id": int(self._box_serial[i]),
                "cam": [float(cx), float(cy), 0.0],
                "cam_bbox": cam_bbox.tolist(),
                "base_grasp": [x_base, y_base, ref_z + offset_grasp],
                "base_aim": [x_base, y_base, ref_z + offset_aim],
                "base_bbox_grasp": bbox_to_base(cam_bbox, z_offset=offset_grasp, **kw).tolist(),
                "base_bbox_aim": bbox_to_base(cam_bbox, z_offset=offset_aim, **kw).tolist(),
                "in_workspace": bool(-ws_x_abs < cx < ws_x_abs),
            }
        return None

    def speed_probe_snapshot(self) -> dict:
        """Diagnostic transport speeds.

        Belt travel is world -Y. Speeds below are positive when moving
        downstream, matching ``cfg.belt_speed``.
        """
        with self.lock:
            belt_qvel = 0.0
            if self.cfg.physical_belt and self._belt_joint is not None:
                belt_qvel = float(self._belt_joint.qvel[0])
            objects = []
            for i, state in enumerate(self._box_state):
                if state == self._FREE:
                    continue
                jnt = self._box_joints[i]
                objects.append({
                    "slot": i,
                    "state": state,
                    "class": self._box_class[i],
                    "serial": int(self._box_serial[i]),
                    "x": float(jnt.qpos[0]),
                    "y": float(jnt.qpos[1]),
                    "z": float(jnt.qpos[2]),
                    "base_z": float(self._to_base(jnt.qpos[0:3])[2]),
                    "vy": float(jnt.qvel[1]),
                    "downstream_speed": float(-jnt.qvel[1]),
                })
            return {
                "time": float(self.data.time),
                "commanded_speed": float(self.cfg.belt_speed),
                "physical_belt": bool(self.cfg.physical_belt),
                "belt_top_world_z": float(self._belt_top_w),
                "belt_top_base_z": float(self._belt_top_w - self._base_p[2]),
                "belt_actuator_command": (
                    float(self.cfg.belt_speed * self.cfg.belt_actuator_speed_scale)
                    if self.cfg.physical_belt else 0.0
                ),
                "belt_slide_qvel": belt_qvel,
                "belt_downstream_speed": belt_qvel,
                "objects": objects,
            }

    def _step_once_locked(self) -> None:
        self.data.ctrl[self._act_ids] = self._ctrl_target
        self._apply_belt_velocity()
        mujoco.mj_step(self.model, self.data)
        self._enforce_belt_kinematics(self._dt)
        gz = float(self.data.site(self._grip_id).xpos[2])
        if self._grip_z_prev is not None:
            self._grip_vz = (gz - self._grip_z_prev) / self._dt
        self._grip_z_prev = gz
        self._vacuum_tick()
        if self.cfg.randomize and self._random_spawn_rate_hz > 0.0:
            spawn_p = min(1.0, self._random_spawn_rate_hz * self._dt)
            if float(self._rng.random()) < spawn_p:
                self._spawn_box()
                self._last_spawn = float(self.data.time)
                self._schedule_next_random_spawn(float(self.data.time))

    def _post_step_batch(self, n: int, now: float, det_period: float) -> None:
        with self.lock:
            self._encoder_distance += self.cfg.belt_speed * self._dt * int(n)
            if not self.cfg.randomize and now - self._last_spawn >= self.cfg.spawn_interval:
                self._spawn_box()
                self._last_spawn = now
            self._recycle_boxes()
            pos = [float(self.data.joint(n_).qpos[0]) for n_ in MJ_JOINTS]
            vel = [float(self.data.joint(n_).qvel[0]) for n_ in MJ_JOINTS]
            if now - self._last_det_pub >= det_period:
                self.latest_snapshot = self._build_snapshot(now)
                self._last_det_pub = now
        self.belt._push(now, self._encoder_distance)
        b = self._backend
        if b is not None:
            b.current_joints = pos
            b.current_jointvels = vel
        if self._recorder is not None:
            self._recorder.maybe_capture()

    def warp_seconds(self, duration: float) -> None:
        """Fast-forward idle kinematic-belt waits without integrating every step."""
        duration = max(0.0, float(duration))
        if duration <= 0.0:
            return
        if self.cfg.physical_belt:
            self.advance_seconds(duration)
            return
        det_period = 1.0 / max(1.0, self.cfg.det_hz)
        with self.lock:
            end = float(self.data.time) + duration
            while self.data.time + 1e-9 < end:
                next_spawn = float("inf")
                if self.cfg.randomize and self._random_spawn_rate_hz > 0.0:
                    next_spawn = self._next_random_spawn_time
                elif not self.cfg.randomize and self.cfg.spawn_interval > 0.0:
                    next_spawn = self._last_spawn + self.cfg.spawn_interval
                next_record = (
                    self._recorder.next_time()
                    if self._recorder is not None
                    else float("inf")
                )
                seg_end = min(end, next_spawn, next_record)
                dt = max(0.0, float(seg_end - self.data.time))
                if dt > 0.0:
                    for i, jnt in enumerate(self._box_joints):
                        if self._box_state[i] != self._ON_BELT:
                            continue
                        self._box_belt_y[i] -= self.cfg.belt_speed * dt
                        jnt.qpos[1] = self._box_belt_y[i]
                        jnt.qvel[1] = -self.cfg.belt_speed
                    self._encoder_distance += self.cfg.belt_speed * dt
                    self.data.time = seg_end
                    self._recycle_boxes()
                if next_spawn <= self.data.time + 1e-9:
                    self._spawn_box()
                    self._last_spawn = float(self.data.time)
                    self._schedule_next_random_spawn(float(self.data.time))
                if self._recorder is not None:
                    self._recorder.maybe_capture_locked()
            mujoco.mj_forward(self.model, self.data)
            self._vacuum_tick()
            pos = [float(self.data.joint(n_).qpos[0]) for n_ in MJ_JOINTS]
            vel = [float(self.data.joint(n_).qvel[0]) for n_ in MJ_JOINTS]
            now = float(self.data.time)
            if now - self._last_det_pub >= det_period:
                self.latest_snapshot = self._build_snapshot(now)
                self._last_det_pub = now
            encoder_distance = float(self._encoder_distance)
        self.belt._push(now, encoder_distance)
        b = self._backend
        if b is not None:
            b.current_joints = pos
            b.current_jointvels = vel

    # ------------------------------------------------------------------
    # the stepping thread
    # ------------------------------------------------------------------
    def _stepper(self) -> None:
        if self.cfg.viewer:
            try:
                import mujoco.viewer as mj_viewer
                self._viewer = mj_viewer.launch_passive(self.model, self.data)
                self._viewer.cam.azimuth = 135.0
                self._viewer.cam.elevation = -20.0
                self._viewer.cam.distance = 2.2
                self._viewer.cam.lookat[:] = (0.25, 0.0, 0.75)
            except Exception:
                self._viewer = None   # no GL — run headless

        t0 = time.monotonic()
        last_sync = 0.0
        det_period = 1.0 / max(1.0, self.cfg.det_hz)
        self._last_spawn = time.time()
        try:
            while self._running:
                # Servo sim time to the wall clock: step exactly the substeps
                # needed to catch up (bounded so a scheduler hiccup can't
                # trigger a spiral of death; 50 * 2 ms = 100 ms absorbed).
                lag = (time.monotonic() - t0) - self.data.time
                n = int(np.clip(round(lag / self._dt), 0, 50))
                if n > 0:
                    pend = self._pending_ctrl
                    for _ in range(n):
                        # ZOH replay: the command in force at this substep's
                        # wall time (sim time is servoed to wall time from t0).
                        t_sub = t0 + self.data.time
                        suction_events = []
                        while pend and pend[0][0] <= t_sub:
                            _, kind, payload = pend.popleft()
                            if kind == "ctrl":
                                self._ctrl_target = payload
                            else:
                                suction_events.append(payload)
                        # Lock per substep (NOT per batch): other lock users
                        # (snapshot readers, tests) wait at most one mj_step.
                        with self.lock:
                            for on in suction_events:
                                self._apply_suction(on)
                            self._step_once_locked()
                    self._post_step_batch(n, time.time(), det_period)
                    self._first_step.set()
                if self._viewer is not None and time.monotonic() - last_sync >= 1.0 / 60.0:
                    self._viewer.sync()
                    last_sync = time.monotonic()
                if self._recorder is not None:
                    self._recorder.maybe_capture()
                time.sleep(self._dt)
        finally:
            if self._recorder is not None:
                self._recorder.close()
            if self._viewer is not None:
                try:
                    self._viewer.close()
                except Exception:
                    pass


class MujocoRobotBackend(RobotBackend):
    """RobotBackend over the MuJoCo twin: the base's 250 Hz stream engine runs
    the identical hardware logic; only the sample sink and suction differ."""

    def __init__(self, core: SimCore, *, clock=None) -> None:
        super().__init__(clock=clock)
        self._core = core
        core.bind_backend(self)

    def wait_for_servers(self, timeout_sec: float = 10.0) -> bool:
        self._core.start()
        return self._core.wait_ready(timeout_sec)

    def _emit_sample(self, positions) -> None:
        self._core.set_ctrl_target(positions)

    def fast_wait_poll_interval(self) -> float | None:
        if self._core.cfg.realtime:
            return None
        if self._core.cfg.physical_belt:
            return 1.0 / max(1.0, self._core.cfg.det_hz)
        return 0.25

    def fast_wait(self, duration: float) -> bool:
        if self._core.cfg.realtime:
            return False
        self._core.warp_seconds(duration)
        return True

    def _set_suction(self, on: bool, requested_at: float) -> None:
        self._core.set_suction(on)
        # In-process toggle == immediate controller ack.
        if on:
            self.last_suction_on_ack_t = self.clock.time()
        else:
            self.last_suction_off_ack_t = self.clock.time()

    def close(self, timeout_sec: float = 5.0) -> None:
        self._core.stop(timeout_sec)


class MujocoWorldSource(WorldSource):
    """WorldSource over the same twin: schema-v2 snapshots synthesized by the
    stepper + the encoder-equivalent belt tracker."""

    def __init__(self, core: SimCore) -> None:
        self._core = core

    def latest_snapshot(self) -> Optional[dict]:
        return self._core.latest_snapshot

    @property
    def belt(self) -> SimBeltTracker:
        return self._core.belt

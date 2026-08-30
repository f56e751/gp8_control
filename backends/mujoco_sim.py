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
_PARK = np.array([0.0, 0.0, -2.0])   # stash unused boxes below the floor
_HOME = [0.0, 0.0, 0.0, 0.0, -math.pi / 2, 0.0]   # B down — matches app startup

# Per-class box tint so throw (transparent) vs push (metal) reads at a glance.
_CLASS_RGBA = {
    "transparent": (0.30, 0.65, 1.00, 0.55),
    "metal": (0.62, 0.62, 0.68, 1.00),
}
_DEFAULT_RGBA = (0.85, 0.20, 0.20, 1.00)

# The reachable conveyor we add for the twin (the vendored belt is out of the
# GP8's reach). Geometry in the gp8 BASE frame; placed into the world via the
# yaskawa_robot body pose at build time.
BELT_BODY = "gp8_belt"
BELT_GEOM = "gp8_belt_surface"


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
    SimBin("throw", "throw", 1.1, -0.25, 0.0),
    SimBin("push_metal", "push", 0.80, 0.60, 0.0),
)


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


@dataclass
class SimConfig:
    """Belt/world parameters (env-overridable via GP8_SIM_*)."""

    belt_speed: float = field(default_factory=lambda: _env_float("GP8_SIM_BELT_SPEED", 0.12))
    spawn_interval: float = field(default_factory=lambda: _env_float("GP8_SIM_SPAWN_INTERVAL", 5.0))
    # Boxes enter the belt where the real camera first sees them: the image
    # centre sits REFERENCE_Y_BASE (2.47 m) upstream of the robot, so the
    # detection->intercept lead time matches hardware (~2.5 m / belt_speed).
    spawn_y: float = field(default_factory=lambda: _env_float(
        "GP8_SIM_SPAWN_Y", extrinsics.REFERENCE_Y_BASE))
    despawn_y: float = -0.8       # base-frame Y past which boxes are recycled
    lane_x: float = field(default_factory=lambda: _env_float("GP8_SIM_LANE_X", 0.45))
    grasp_z: float = 0.042        # box CENTRE height in base frame (pass cfg.GRASP_Z)
    aim_dz: float = 0.08
    classes: tuple = field(default_factory=lambda: tuple(
        c.strip() for c in os.environ.get("GP8_SIM_CLASSES", "transparent,metal").split(",")
        if c.strip()) or ("transparent",))
    grab_radius: float = field(default_factory=lambda: _env_float("GP8_SIM_GRAB_RADIUS", 0.10))
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
    # Bins at the app's real throw goal / push targets (see bins_from_config;
    # the app passes its live Config). () = no bins. Width: GP8_SIM_BIN_W.
    bins: tuple = field(default_factory=bins_from_config)


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


def _build_twin_model(scene_path: str, base_pos, lane_x: float, center_y: float,
                      grasp_z: float, half_len: float, bins=()):
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
    top = bz + grasp_z - BOX_HALF_Z            # belt surface top (world z)
    thick = 0.02
    belt = spec.worldbody.add_body(
        name=BELT_BODY, pos=[bx + lane_x, by + center_y, top - thick],
    )
    belt.add_geom(
        name=BELT_GEOM, type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.16, half_len, thick], pos=[0.0, 0.0, 0.0],
        rgba=[0.12, 0.12, 0.14, 1.0], friction=[0.3, 0.02, 0.002],
    )
    for b in bins:
        _add_bin(spec, base_pos, b)
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

    def __init__(self, speed: float) -> None:
        self._speed = float(speed)
        self._samples: deque = deque(maxlen=400)
        self._samples.append((time.time(), 0.0))

    # -- fed by the stepper -------------------------------------------------
    def _push(self, wall_t: float, distance_m: float) -> None:
        self._samples.append((float(wall_t), float(distance_m)))

    # -- ConveyorSpeedTracker surface ---------------------------------------
    @property
    def current(self) -> float:
        return self._speed

    @property
    def distance_m(self) -> float | None:
        return self.distance_at(time.time())

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
        if self._dead:
            return
        now = time.monotonic()
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
            with self._core.lock:
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


class SimCore:
    """Owns MjModel/MjData + the wall-clock-servoed stepping thread.

    Concurrency: `lock` guards ALL MjData/MjModel mutation. The stream engine's
    ``_emit_sample`` only writes the 6-float ``_ctrl_target`` under it
    (microseconds); the stepper holds it per substep batch. Joint snapshots are
    handed to the bound backend as fresh list objects (GIL-atomic swap, same
    pattern as the ROS joint_states callback).
    """

    _FREE, _ON_BELT, _GRABBED, _LOOSE = "free", "on_belt", "grabbed", "loose"

    def __init__(self, cfg: SimConfig | None = None) -> None:
        if mujoco is None:
            raise ImportError(
                "the MuJoCo sim backend needs the 'mujoco' package "
                "(uv sync --extra sim): " + repr(_MUJOCO_IMPORT_ERROR))
        scene = _find_scene()
        self.cfg = cfg or SimConfig()

        # base pose is fixed in the XML (yaskawa_robot @ (-0.05,0,0.6), identity
        # rot); read it from a throwaway load to place the reachable belt.
        base = mujoco.MjModel.from_xml_path(str(scene)).body(MJ_BASE_BODY).pos
        # The surface spans despawn_y..spawn_y (+0.1 m margin each end) and
        # is centred BETWEEN them — not on the base origin, which would leave
        # the 2.47 m camera-reference spawn point hanging past the belt end.
        center_y = 0.5 * (self.cfg.spawn_y + self.cfg.despawn_y)
        half_len = 0.5 * (self.cfg.spawn_y - self.cfg.despawn_y) + 0.1
        self.bins: tuple = tuple(self.cfg.bins)
        self.model = _build_twin_model(
            str(scene), base, self.cfg.lane_x, center_y, self.cfg.grasp_z, half_len,
            self.bins)
        # Landing telemetry: loose (thrown / pushed) boxes are classified when
        # they come down — inside a bin footprint below its rim, or a miss.
        self.bin_hits: dict[str, int] = {b.name: 0 for b in self.bins}
        self.bin_misses = 0
        self.data = mujoco.MjData(self.model)
        self.lock = threading.Lock()
        self._dt = float(self.model.opt.timestep)

        self._act_ids = [int(self.model.actuator(n).id) for n in MJ_ACTS]
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

        # --- streamed command target (written by _emit_sample at 250 Hz) -----
        self._ctrl_target = np.array(_HOME, dtype=float)

        # --- object pool -----------------------------------------------------
        self._box_joints = [self.data.joint(n) for n in BOX_JOINTS]
        self._box_geom_ids = [int(self.model.geom(n).id) for n in BOX_GEOMS]
        self._box_body_ids = [int(self.model.body(n).id) for n in BOX_BODIES]
        self._weld_ids = [int(self.model.equality(n).id) for n in BOX_WELDS]
        self._box_state = [self._FREE] * len(BOX_JOINTS)
        self._box_class = [""] * len(BOX_JOINTS)
        self._box_serial = [0] * len(BOX_JOINTS)   # spawn number, for the landing log
        # Kinematic along-belt coordinate per ON_BELT box (world Y): contact
        # friction during a step brakes a purely velocity-asserted box (~0.10
        # realized vs 0.12 commanded), which would desync the boxes from the
        # encoder-distance integral and every ETA. Y is therefore DRIVEN;
        # X/Z stay dynamic (gravity, pushes, contacts).
        self._box_belt_y = [0.0] * len(BOX_JOINTS)
        self._grabbed: int | None = None
        self._spawn_count = 0
        self._last_spawn = 0.0
        for jnt in self._box_joints:   # park the pool below the floor
            jnt.qpos[0:3] = _PARK
            jnt.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
            jnt.qvel[:] = 0.0

        # --- world-source outputs -------------------------------------------
        self.belt = SimBeltTracker(self.cfg.belt_speed)
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
        self._thread = threading.Thread(
            target=self._stepper, name="gp8_sim_stepper", daemon=True)
        self._thread.start()

    def stop(self, timeout_sec: float = 2.0) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=timeout_sec)

    def wait_ready(self, timeout_sec: float = 10.0) -> bool:
        return self._first_step.wait(timeout=timeout_sec)

    def bind_backend(self, backend: "MujocoRobotBackend") -> None:
        self._backend = backend

    # ------------------------------------------------------------------
    # RobotBackend hooks (called from the app/skill thread)
    # ------------------------------------------------------------------
    def set_ctrl_target(self, positions) -> None:
        with self.lock:
            self._ctrl_target[:] = [float(x) for x in positions]

    def set_suction(self, on: bool) -> None:
        """Weld the nearest on-belt box on suction ON; release it on OFF.

        The weld physically drags the box through the swing, so at release its
        free-joint qvel already carries the true throw velocity — it flies
        ballistically with no synthesized fling.
        """
        with self.lock:
            if on:
                self._grab_nearest()
            else:
                self._release_grabbed()

    def _grab_nearest(self) -> None:
        if self._grabbed is not None:
            return
        grip = self.data.site(self._grip_id).xpos
        best, bestd = None, self.cfg.grab_radius
        for i, jnt in enumerate(self._box_joints):
            if self._box_state[i] != self._ON_BELT:
                continue
            d = float(np.linalg.norm(jnt.qpos[0:3] - grip))
            if d < bestd:
                best, bestd = i, d
        if best is None:
            return   # vacuum sucking air — matches hardware's silent miss
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
        self._grabbed = best

    def _release_grabbed(self) -> None:
        if self._grabbed is None:
            return
        self.data.eq_active[self._weld_ids[self._grabbed]] = 0
        self._box_state[self._grabbed] = self._LOOSE
        self._grabbed = None

    # ------------------------------------------------------------------
    # belt / object lifecycle (stepper-owned, under self.lock)
    # ------------------------------------------------------------------
    def _to_base(self, world_xyz) -> np.ndarray:
        return self._base_R.T @ (np.asarray(world_xyz) - self._base_p)

    def _to_world(self, base_xyz) -> np.ndarray:
        return self._base_p + self._base_R @ np.asarray(base_xyz)

    def _spawn_box(self) -> None:
        idx = next((i for i, s in enumerate(self._box_state) if s == self._FREE), None)
        if idx is None:
            return
        cls = self.cfg.classes[self._spawn_count % len(self.cfg.classes)]
        jitter = 0.06 * ((self._spawn_count % 3) - 1)   # -0.06, 0, +0.06
        world = self._to_world([self.cfg.lane_x + jitter, self.cfg.spawn_y, self.cfg.grasp_z])
        jnt = self._box_joints[idx]
        jnt.qpos[0:3] = world
        jnt.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        jnt.qvel[:] = 0.0
        self._box_state[idx] = self._ON_BELT
        self._box_class[idx] = cls
        self._box_belt_y[idx] = float(world[1])
        self.model.geom_rgba[self._box_geom_ids[idx]] = _CLASS_RGBA.get(cls, _DEFAULT_RGBA)
        self._spawn_count += 1
        self._box_serial[idx] = self._spawn_count

    def _apply_belt_velocity(self) -> None:
        """Pre-step: assert on-belt boxes' world-Y velocity (solver hint so
        contacts see the conveyor motion). The authoritative advance happens in
        :meth:`_enforce_belt_kinematics` after the step."""
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

    def _log_landing(self, i: int, base) -> None:
        tag = f"{self._box_class[i] or 'box'} #{self._box_serial[i]}"
        hit = self._bin_at(base)
        if hit is not None:
            self.bin_hits[hit.name] += 1
            print(f"[sim] {tag} landed IN bin '{hit.name}' "
                  f"({base[0]:+.2f}, {base[1]:+.2f})  hits={self.bin_hits}", flush=True)
            return
        self.bin_misses += 1
        near = ""
        if self.bins:
            b = min(self.bins, key=lambda b: math.hypot(base[0] - b.x, base[1] - b.y))
            near = f"  nearest '{b.name}' d={math.hypot(base[0] - b.x, base[1] - b.y):.2f}m"
        print(f"[sim] {tag} MISSED ({base[0]:+.2f}, {base[1]:+.2f}){near}  "
              f"misses={self.bin_misses}", flush=True)

    def _recycle_boxes(self) -> None:
        """Retire boxes off the belt end / on the floor; flag ones knocked off.

        Loose boxes (thrown / pushed) are recycled once they come down — on
        the floor (miss) or at the bottom of a bin (hit) — and logged.
        """
        off_belt = self._base_p[2] + self.cfg.grasp_z - 0.05
        fallen = self._base_p[2] - 0.4   # ~floor level, world z
        bin_bottom = 2 * _BIN_FLOOR_Z + BOX_HALF_Z + 0.03   # resting on a bin floor
        for i, jnt in enumerate(self._box_joints):
            if self._box_state[i] in (self._FREE, self._GRABBED):
                continue
            world = jnt.qpos[0:3]
            base = self._to_base(world)
            loose = self._box_state[i] == self._LOOSE
            in_bin = loose and self._bin_at(base) is not None
            gone = (base[1] < self.cfg.despawn_y or abs(base[0]) > 1.6
                    or (world[2] < bin_bottom if in_bin else world[2] < fallen))
            if gone:
                if loose:
                    self._log_landing(i, base)
                jnt.qpos[0:3] = _PARK
                jnt.qvel[:] = 0.0
                self._box_state[i] = self._FREE
                self._box_class[i] = ""
            elif self._box_state[i] == self._ON_BELT and world[2] < off_belt:
                self._box_state[i] = self._LOOSE   # pushed / knocked off the belt

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
        for i, jnt in enumerate(self._box_joints):
            if self._box_state[i] != self._ON_BELT:
                continue
            b = self._to_base(jnt.qpos[0:3])
            # Base -> belt-frame camera coords (the inverse of camera_debug's
            # constant-translation mapping), then run the corners through the
            # SAME bbox_to_base the real pipeline uses (zero latency in sim).
            cx = (float(b[0]) - ref_x) / sx
            cy = (float(b[1]) - ref_y) / sy
            cam_bbox = np.array([
                [cx - BOX_HALF_X, cy - BOX_HALF_Y, 0.0],
                [cx - BOX_HALF_X, cy + BOX_HALF_Y, 0.0],
                [cx + BOX_HALF_X, cy + BOX_HALF_Y, 0.0],
                [cx + BOX_HALF_X, cy - BOX_HALF_Y, 0.0],
            ])
            kw = dict(ref_x=ref_x, ref_y=ref_y, ref_z=ref_z,
                      scale_x=sx, scale_y=sy, y_back_projection=0.0)
            x_base = ref_x + sx * cx
            y_base = ref_y + sy * cy
            dets.append({
                "class": self._box_class[i],
                "confidence": 0.9,
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
            "perception_delay_s": 0.0,
            "applied_delay_s": 0.0,
            "latency_mode": "sim",
            "detections": dets,
        }

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
                    with self.lock:
                        for _ in range(n):
                            self.data.ctrl[self._act_ids] = self._ctrl_target
                            self._apply_belt_velocity()
                            mujoco.mj_step(self.model, self.data)
                            self._enforce_belt_kinematics(self._dt)
                        self._encoder_distance += self.cfg.belt_speed * self._dt * n
                        now = time.time()
                        if now - self._last_spawn >= self.cfg.spawn_interval:
                            self._spawn_box()
                            self._last_spawn = now
                        self._recycle_boxes()
                        # joint snapshot for the backend (fresh lists: readers
                        # see either the old or the new object, never a tear)
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

    def __init__(self, core: SimCore) -> None:
        super().__init__()
        self._core = core
        core.bind_backend(self)

    def wait_for_servers(self, timeout_sec: float = 10.0) -> bool:
        self._core.start()
        return self._core.wait_ready(timeout_sec)

    def _emit_sample(self, positions) -> None:
        self._core.set_ctrl_target(positions)

    def _set_suction(self, on: bool, requested_at: float) -> None:
        self._core.set_suction(on)
        # In-process toggle == immediate controller ack.
        if on:
            self.last_suction_on_ack_t = time.time()
        else:
            self.last_suction_off_ack_t = time.time()

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

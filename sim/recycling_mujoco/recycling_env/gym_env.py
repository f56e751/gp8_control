from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import torch
except ImportError:
    torch = None

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    try:
        import gym
        from gym import spaces
    except ImportError:
        gym = None
        spaces = None

try:
    from .action_control import RecyclingActionMixin
    from .controller import JointTrajectoryController
    from .demo_policy import DemoSelector
    from .gripper import SuctionGripper
    from .gui_utils import make_throttled_viewer_hook
    from .perception import ColorSegmentationDetector, Detection
    from .sim_env import RecyclingSimEnv
    from .tracker import BoundingBoxTracker, TrackingState
except ImportError:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from recycling_env.action_control import RecyclingActionMixin
    from recycling_env.controller import JointTrajectoryController
    from recycling_env.demo_policy import DemoSelector
    from recycling_env.gripper import SuctionGripper
    from recycling_env.gui_utils import make_throttled_viewer_hook
    from recycling_env.perception import ColorSegmentationDetector, Detection
    from recycling_env.sim_env import RecyclingSimEnv
    from recycling_env.tracker import BoundingBoxTracker, TrackingState

try:
    from models.model import FCN
    from utils.pushing_trajectory_generator import PushingTrajectoryGenerator
    from utils.yaskawa_gp8 import Yaskawa
except ImportError:
    FCN = None
    PushingTrajectoryGenerator = None
    Yaskawa = None

try:
    from mujoco.viewer import launch_passive
except ImportError:
    launch_passive = None


if gym is None:
    _EnvBase = object
else:
    _EnvBase = gym.Env


@dataclass(frozen=True)
class WrapperStepState:
    detection: Detection
    tracking: TrackingState
    q_target: np.ndarray


@dataclass
class ObjectSlot:
    freejoint_name: str
    site_name: str
    body_name: str
    geom_name: str
    equality_name: str
    home_quat: np.ndarray
    target_xy: np.ndarray | None = None
    active: bool = False
    manipulated: bool = False
    manipulated_time: float | None = None
    blocked: bool = False
    spawn_id: int | None = None
    is_obstacle: bool = False
    suction_p: float = 1.0           # per-object throw-grasp success rate (in the obs)
    suction_ok: bool | None = None   # latched Bernoulli(suction_p) roll; None = unrolled


class RecyclingBBoxGymEnv(RecyclingActionMixin, _EnvBase):
    """Preliminary gym-style wrapper that exposes world-frame box coordinates as the state."""

    metadata = {"render_modes": ["rgb_array", "bgr_array"], "render_fps": 20}

    def __init__(
        self,
        model_path: str = "scene.xml",
        cam_w: int = 320,
        cam_h: int = 240,
        camera_name: str = "d435i_view",
        belt_speed: float = 0.5,
        # Re-calibrated 2026-06-11 from 1.25 after stiffening slider_act kv
        # (2000->80000): with the stiff carrier no longer sagging, 1.25 over-drove
        # objects to ~0.44 (> belt_speed). 1.10 = 1.25 * belt_speed / measured_ride
        # (0.453) lands the object ride speed at ~belt_speed (consistent idle/load).
        belt_actuator_speed_scale: float = 1.10,
        use_physical_belt: bool = True,
        startup_duration: float = 2.0,
        frame_skip: int = 10,
        max_steps: int = 5000,
        render_mode: str | None = None,
        use_camera: bool = True,
        flip_code: int | None = -1,
        lower_hsv: np.ndarray | None = None,
        upper_hsv: np.ndarray | None = None,
        min_area: float = 150.0,
        target_px: tuple[float, float] | None = None,
        max_detections: int = 6,
        spawn_enabled: bool = True,
        spawn_rate_hz: float = 1.0,
        spawn_x: float = -1.5,
        spawn_y_range: tuple[float, float] = (0.25, 0.65),
        spawn_x_range: tuple[float, float] | None = None,
        spawn_y: float | None = None,
        spawn_z: float = 0.605,
        despawn_x: float = 1.2,
        despawn_y: float | None = None,
        stale_object_despawn_y: float = 0.0,
        randomize_object_size: bool = True,
        randomize_object_yaw: bool = True,
        spawn_clearance_margin: float = 0.03,
        spawn_max_placement_tries: int = 10,
        obstacle_prob: float = 0.0,
        suction_success_rate_range: tuple[float, float] = (1.0, 1.0),
        manipulated_object_lifetime: float = 2.0,
        object_elimination_z: float = 0.05,
        throwing_model_path: str = "models/ckpt/NN_newprimitive2_40000datapoint_500epochs_v2_B10.pt",
        target_xy: tuple[float, float] = (1.0, 0.0),
        target_xy_candidates: tuple[tuple[float, float], ...] | None = None,
        action_lateral_center: float = 0.4,
        action_reachable_window: tuple[float, float] = (0.2, 0.6),
        action_reachable_radius: float = 0.69,
        action_prediction_horizon: float = 0.1,
        action_max_intercept_time: float = 1.8,
        action_grasp_height: float = 0.03,
        action_single_object_standby_xy: tuple[float, float] = (0.37, 0.5),
        action_single_object_standby_z: float = 0.05,
        action_initial_approach_duration: float = 1.0,
        action_initial_velocity_scale: float = 0.4,
        action_wait_timeout: float = 6.0,
        throw_feasibility_projection_seconds: float = 0.0,
        pushing_enabled: bool = True,
        pushing_hover_height: float = 0.07,
        pushing_velocity_limit_scale: float = 0.6,
        pushing_params: dict[str, float] | None = None,
        push_cone_offset: float = 0.0,
        step_sleep: float = 0.02,
        hidden_position: tuple[float, float, float] = (-10.0, -10.0, 1.5),
        tolerance_px: tuple[float, float] = (15.0, 15.0),
        gains: dict[str, float] | None = None,
        max_delta: dict[str, float] | None = None,
        q_home: np.ndarray | None = None,
        q_ready: np.ndarray | None = None,
        debug_chain_log: bool = False,
    ) -> None:
        if spaces is None:
            raise ImportError(
                "RecyclingBBoxGymEnv requires gymnasium or gym to be installed."
            )

        self.render_mode = render_mode
        self.use_camera = bool(use_camera)
        self.flip_code = flip_code
        self.frame_skip = int(frame_skip)
        self.max_steps = int(max_steps)
        self.belt_speed = float(belt_speed)
        self.belt_actuator_speed_scale = float(belt_actuator_speed_scale)
        # When True, the belt geom physically slides and carries objects via
        # friction (the slider_act actuator runs at belt_speed*scale; the qvel
        # override in _apply_conveyor_object_motion is skipped). When False,
        # the original kinematic behavior runs: belt is frozen, every active
        # object's qvel is forced to (-belt_speed) per tick. Available as an
        # A/B flag during the physical-belt rollout.
        self.use_physical_belt = bool(use_physical_belt)
        self.startup_duration = float(startup_duration)
        self.max_detections = int(max_detections)
        self.spawn_enabled = bool(spawn_enabled)
        self.spawn_rate_hz = float(spawn_rate_hz)
        legacy_spawn_path = (
            spawn_x != -1.5
            or spawn_y_range != (0.25, 0.65)
            or despawn_x != 1.2
        )
        if spawn_x_range is None:
            spawn_x_range = spawn_y_range if legacy_spawn_path else (0.20, 0.53)
        if spawn_y is None:
            spawn_y = spawn_x if legacy_spawn_path else 1.0
        if despawn_y is None:
            despawn_y = despawn_x if legacy_spawn_path else -1.4

        self.spawn_x_range = (float(spawn_x_range[0]), float(spawn_x_range[1]))
        self.spawn_y = float(spawn_y)
        self.spawn_z = float(spawn_z)
        self.despawn_y = float(despawn_y)
        self.stale_object_despawn_y = float(stale_object_despawn_y)
        self.manipulated_object_lifetime = float(manipulated_object_lifetime)
        self.object_elimination_z = float(object_elimination_z)
        self.throwing_model_path = str((Path(__file__).resolve().parent.parent / throwing_model_path).resolve())
        self.target_xy = np.asarray(target_xy, dtype=np.float64)
        # AIM target -- the point the throw/push MOTION is directed at. DECOUPLED
        # from the physical bin (2026-06-12): a fixed point at robot-x 0.8, NOT the
        # bin centroid. The physical bin is positioned separately and
        # size-dependently (see the `_bin_*` helpers); only the bin's x is
        # decoupled (aim 0.8 vs bin center 0.7 + bin_half_x), the y is shared.
        if target_xy_candidates is None:
            target_xy_candidates = ((0.8, 0.1), (0.8, -0.1))
        self.target_xy_candidates = tuple(
            np.asarray(candidate, dtype=np.float64) for candidate in target_xy_candidates
        )
        if len(self.target_xy_candidates) == 0:
            raise ValueError("target_xy_candidates must not be empty")
        # Robot-frame x of the bin's CLOSER (-x) face -- just past the table's far
        # edge (~0.678) so the bin doesn't overlap the table. The bin CENTER is
        # this + the bin's x half-extent (size-dependent; see `_bin_center_*`).
        self._bin_near_edge_x = 0.7
        self.action_lateral_center = float(action_lateral_center)
        self.action_reachable_window = (
            float(action_reachable_window[0]),
            float(action_reachable_window[1]),
        )
        self.action_reachable_radius = float(action_reachable_radius)
        # DEPRECATED (2026-06-11): the push cone was removed in favour of
        # screening the push against the same reachable circle as the throw, so
        # RL learns the bin-opening geometry from reward (see
        # `_push_cone_target_y`). These two are no longer read; kept only so the
        # ctor signature / external callers don't break.
        self.push_cone_offset = float(push_cone_offset)
        self._push_cone_min_reach = 0.03
        # Throw action-mask projection horizon. The throw screen tests
        # feasibility at `obj_y - belt_speed * horizon` (0 by default; the
        # throw builder absorbs drift via its own wait_time). PUSH uses no
        # constant horizon -- `_predict_push_hover` (the single push-hover
        # predictor) computes the iterated reach-time (mirrors the throw).
        self.throw_feasibility_projection_seconds = float(throw_feasibility_projection_seconds)
        self.action_prediction_horizon = float(action_prediction_horizon)
        self.action_max_intercept_time = float(action_max_intercept_time)
        self.action_single_object_standby_xyz = np.array(
            [
                float(action_single_object_standby_xy[0]),
                float(action_single_object_standby_xy[1]),
                float(action_single_object_standby_z),
            ],
            dtype=np.float64,
        )
        self.action_initial_approach_duration = float(action_initial_approach_duration)
        self.action_initial_velocity_scale = float(action_initial_velocity_scale)
        self.action_wait_timeout = float(action_wait_timeout)
        self.pushing_enabled = bool(pushing_enabled)
        self.pushing_hover_height = float(pushing_hover_height)
        self.pushing_velocity_limit_scale = float(pushing_velocity_limit_scale)
        self.pushing_params = {
            "velocity": 1.4,
            "acc_limit": 10.0,
            "start_margin": 0.17,
            "distance": 0.15,
            # Push-target z (= grip_site z in robot frame) = action_grasp_height +
            # height_offset. Setting -0.05 puts the push grip_site exactly one
            # object-height (5 cm) below the grasping z, so the pusher engages
            # the box from the side at the bottom of its body rather than from
            # above the top face.
            "height_offset": -0.05,
            # Tilt at start (-swing) and end (+swing) of the push gives a
            # windup/follow-through motion that shrinks the after-push
            # trajectory.
            "swing_angle": 15.0,
            "swing_radius": 0.10,
        }
        if pushing_params is not None:
            self.pushing_params.update({str(key): float(value) for key, value in pushing_params.items()})
        self.step_sleep = float(step_sleep)
        self.hidden_position = np.asarray(hidden_position, dtype=np.float64)
        self._rng = np.random.default_rng()
        self._spawn_counter = 0
        self._target_sample_counter = 0
        self._reward_accumulator = 0.0

        self.joint_names = ["S_axis", "L_axis", "U_axis", "R_axis", "B_axis", "T_axis"]
        self.q_home = np.zeros(6, dtype=np.float64) if q_home is None else np.asarray(q_home, dtype=np.float64)
        self.q_ready = (
            np.array(
                [0.933726, 0.93985696, 0.34133495, 0.0, -0.97227431, 0.0],
                dtype=np.float64,
            )
            if q_ready is None
            else np.asarray(q_ready, dtype=np.float64)
        )

        if self.q_home.shape != (6,):
            raise ValueError("q_home must have shape (6,)")
        if self.q_ready.shape != (6,):
            raise ValueError("q_ready must have shape (6,)")
        if self.frame_skip < 1:
            raise ValueError("frame_skip must be at least 1")
        if self.belt_actuator_speed_scale <= 0.0:
            raise ValueError("belt_actuator_speed_scale must be positive")
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if self.max_detections < 1:
            raise ValueError("max_detections must be at least 1")
        if self.spawn_rate_hz < 0.0:
            raise ValueError("spawn_rate_hz must be non-negative")
        if self.spawn_x_range[0] > self.spawn_x_range[1]:
            raise ValueError("spawn_x_range must be ordered as (min_x, max_x)")
        if self.object_elimination_z < 0.0:
            raise ValueError("object_elimination_z must be non-negative")
        if self.stale_object_despawn_y > 0.0:
            raise ValueError("stale_object_despawn_y must be non-positive")
        if self.manipulated_object_lifetime <= 0.0:
            raise ValueError("manipulated_object_lifetime must be positive")
        if self.action_reachable_window[0] >= self.action_reachable_window[1]:
            raise ValueError("action_reachable_window must be ordered as (min_y, max_y)")
        if self.action_reachable_radius <= 0.0:
            raise ValueError("action_reachable_radius must be positive")
        if self.action_max_intercept_time <= 0.0:
            raise ValueError("action_max_intercept_time must be positive")
        self.action_grasp_height = float(action_grasp_height)
        if self.action_grasp_height <= 0.0:
            raise ValueError("action_grasp_height must be positive")
        if self.pushing_hover_height <= 0.0:
            raise ValueError("pushing_hover_height must be positive")
        if self.pushing_velocity_limit_scale <= 0.0:
            raise ValueError("pushing_velocity_limit_scale must be positive")
        if self.step_sleep < 0.0:
            raise ValueError("step_sleep must be non-negative")
        if self.hidden_position.shape != (3,):
            raise ValueError("hidden_position must have shape (3,)")

        self.env = RecyclingSimEnv(
            model_path=model_path,
            cam_w=cam_w,
            cam_h=cam_h,
            camera_name=camera_name,
        )
        self.controller = JointTrajectoryController(self.env, self.joint_names)

        lower = np.array([0, 150, 50], dtype=np.uint8) if lower_hsv is None else np.asarray(lower_hsv, dtype=np.uint8)
        upper = np.array([10, 255, 255], dtype=np.uint8) if upper_hsv is None else np.asarray(upper_hsv, dtype=np.uint8)
        self.detector = ColorSegmentationDetector(lower_hsv=lower, upper_hsv=upper, kernel_size=5, min_area=min_area)
        self.object_slots = self._build_object_slots()
        self.object_geom_sizes_xy = self._build_object_geom_sizes_xy()
        first_slot = self.object_slots[0]
        self.gripper = SuctionGripper(
            self.env,
            target_body=first_slot.body_name,
            target_geom=first_slot.geom_name,
            target_site=first_slot.site_name,
            target_freejoint=first_slot.freejoint_name,
            equality_name=first_slot.equality_name,
        )
        self.object_half_height = float(self.env.model.geom("red_box_geom").size[2])
        self.object_height = float(2.0 * self.object_half_height)
        # ---- Per-spawn object size (real-world package shapes) -------------------
        # Footprint is randomized per spawn within these HALF-extent ranges (full
        # side lengths: x in [7.5,15] cm, y in [10,30] cm); height is fixed at 3 cm
        # (-> object_half_height = 0.015 from the XML default). When
        # `randomize_object_size` is False, each slot gets a fixed, reproducible
        # size spread across the ranges (deterministic mode for debugging).
        self.randomize_object_size = bool(randomize_object_size)
        # Randomize yaw (rotation about world z) per spawn so boxes arrive at
        # arbitrary in-plane angles (real-world packages are not axis-aligned).
        self.randomize_object_yaw = bool(randomize_object_yaw)
        self._obj_size_x_half_range = (0.0375, 0.075)   # 7.5..15 cm full
        self._obj_size_y_half_range = (0.05, 0.15)      # 10..30 cm full
        self._obj_half_height = float(self.object_half_height)  # 0.015 (3 cm), constant
        self._object_mass = 0.2  # kg, kept constant; inertia recomputed per size
        # Spawn-time clearance: a new box must not overlap any active object's
        # footprint (inflated by this margin); otherwise the spawn is deferred.
        self.spawn_clearance_margin = float(spawn_clearance_margin)
        self.spawn_max_placement_tries = max(1, int(spawn_max_placement_tries))
        # Fraction of spawns that are OBSTACLES: objects that belong to no bin
        # (the robot should leave them on the belt). Marked per-slot via
        # `slot.is_obstacle` and exposed in the observation's is_obstacle field.
        self.obstacle_prob = float(np.clip(obstacle_prob, 0.0, 1.0))
        # Per-object throw-grasp suction success rate is drawn from this range at
        # spawn (uniform). Default (1.0, 1.0) = suction always succeeds (current
        # behavior). The rate is exposed per-object in the obs; the realized
        # Bernoulli outcome is rolled at the grasp instant (see `_suction_takes`).
        _sl, _sh = float(suction_success_rate_range[0]), float(suction_success_rate_range[1])
        if not (0.0 <= _sl <= _sh <= 1.0):
            raise ValueError("suction_success_rate_range must satisfy 0 <= low <= high <= 1")
        self.suction_success_rate_range = (_sl, _sh)
        # World z of the belt surface (the spawn_z default was tuned for the old
        # 5 cm cube: surface = spawn_z - cube_half_height(0.025)).
        self._belt_surface_z = float(self.spawn_z) - 0.025
        self.object_site_names = [slot.site_name for slot in self.object_slots]
        self.robot_base_xyz = self.env.get_body_position("yaskawa_robot").astype(np.float64)
        self.robot_base_xy = self.robot_base_xyz[:2].copy()
        self.pusher_collision_name = "pusher_collision"
        try:
            pusher_geom_id = self.env.require_geom(self.pusher_collision_name)
            self._pusher_collision_contype = int(self.env.model.geom_contype[pusher_geom_id])
            self._pusher_collision_conaffinity = int(self.env.model.geom_conaffinity[pusher_geom_id])
            self.env.set_geom_collision_enabled(self.pusher_collision_name, False)
        except KeyError:
            self.pusher_collision_name = None
            self._pusher_collision_contype = 1
            self._pusher_collision_conaffinity = 1
        self.target_bin_home_quat = self.env.get_body_quaternion("target_bin").astype(np.float64)
        self.target_bin_spawn_z = 0.05
        # Cache the (fixed) physical-bin extents -- size-dependent, read once from
        # the bin geoms. Used by placement, the in-bin metric, and the obs.
        self._bin_center_x_robot = float(self._bin_near_edge_x + self._bin_near_half_extent_x())
        self._bin_outer_full = self._bin_outer_full_xy()
        self._bin_interior_half = self._bin_interior_half_xy()
        self._bin_z_lo, self._bin_z_hi = self._bin_z_bounds_world()
        self._target_bin_active = False
        self.robot_kinematics = Yaskawa() if Yaskawa is not None else None
        self.throwing_model = self._load_throwing_model()
        self.pushing_trajectory_generator = self._build_pushing_trajectory_generator()

        self.target_px = target_px if target_px is not None else (self.env.cam_w // 2 - 60, self.env.cam_h // 2)
        self.tracker = BoundingBoxTracker(
            target_px=self.target_px,
            nominal_q=self.q_ready,
            joint_names=self.joint_names,
            tolerance_px=tolerance_px,
            gains=gains or {"S_axis": 0.0025, "L_axis": -0.0015},
            max_delta=max_delta or {"S_axis": 0.5, "L_axis": 0.35},
        )

        # Per object (13 fields):
        #   [center_x, center_y, bbox_w, bbox_h, target_x, target_y, is_obstacle,
        #    v_measured, suction_p, bin_x, bin_y, bin_w, bin_h]
        #   ALL position fields are ROBOT frame (robot_x = world_x - robot_base_x).
        #   center/bbox: object centroid + AABB (ROBOT frame, full extents).
        #   target_x/y: the AIM point the motion is directed at (ROBOT frame).
        #   is_obstacle in {0,1}: 1 = object belongs to no bin.
        #   v_measured: the object's measured belt-direction forward speed (>=0).
        #   suction_p: per-object throw-grasp success RATE in [0,1] (the realized
        #     grasp outcome is a hidden Bernoulli(suction_p)); 0 for obstacles.
        #   bin_x/y, bin_w/h: the PHYSICAL bin's center (ROBOT frame) + outer AABB
        #     full extents -- DECOUPLED from the aim target; 0 for obstacles.
        # + globals [belt_speed (nominal), spawn_rate]. Both speeds are present so a
        # policy can use the per-object measured speed or the global nominal one.
        self.observation_space = spaces.Box(
            low=np.concatenate(
                (
                    np.tile(np.array([-20.0, -20.0, 0.0, 0.0, -20.0, -20.0, 0.0, 0.0, 0.0, -20.0, -20.0, 0.0, 0.0], dtype=np.float32), self.max_detections),
                    np.array([-20.0, 0.0], dtype=np.float32),
                )
            ),
            high=np.concatenate(
                (
                    np.tile(np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 1.0, 20.0, 1.0, 20.0, 20.0, 20.0, 20.0], dtype=np.float32), self.max_detections),
                    np.array([20.0, 20.0], dtype=np.float32),
                )
            ),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete([self.max_detections + 1, 2])

        self._step_count = 0
        self._seen_detection = False
        self._last_detection: Detection | None = None
        self._last_detections: list[Detection] = []
        self._last_tracking: TrackingState | None = None
        self._last_frame_bgr: np.ndarray | None = None
        self._sim_step_hook: Callable[[], bool] | None = None
        # 1-step-ahead semantics: action[N] = (next_slot, next_primitive). At
        # step N the env executes the manipulation specified by action[N-1]
        # (the "pending" below), then routes the EE toward action[N]'s slot.
        # Step 0 has no pending, so it's route-only.
        self._pending_target_slot: ObjectSlot | None = None
        self._pending_action_primitive: int | None = None
        # Per-step per-(slot, primitive) action feasibility, computed once at the
        # end of every step()/reset() by `_compute_slot_feasibility_flags()`.
        # Used by the demo policy and exposed to RL as info["action_mask"].
        self._slot_feasibility_flags: dict[int, dict[str, bool]] = {}
        # When the previous step routed the EE to a push hover above an object's
        # cone push_start, this records that object's spawn_id. The next push
        # step uses the cone plan (clean vertical strike + wait) when the slot it
        # pushes matches this id; otherwise (un-routed/external push) it falls
        # back to the interception solver. See action_control.py
        # `_route_to_push_handoff` / `_execute_push_transition_with_next`.
        self._push_routed_spawn_id: int | None = None
        # The predicted contact object position (robot frame) the routed hover was
        # placed above. The next push strikes a FIXED push to THIS target (no
        # strike-time re-planning). Paired with `_push_routed_spawn_id`.
        self._push_routed_target_xyz: np.ndarray | None = None
        # Demo-policy state (only consumed by `demo_policy.py`). Persists alt
        # bookkeeping (`_demo_next_primitive`) between calls; ignored by RL.
        self._demo_next_primitive = self.ACTION_THROW
        # Slot currently being pushed. While set, `_apply_conveyor_object_motion`
        # skips the velocity override for this slot so the pusher's contact
        # force can actually accelerate the box laterally.
        self._pushing_slot: ObjectSlot | None = None
        self._demo_selector = DemoSelector(self)
        self._debug_chain_log = bool(debug_chain_log)

    def _log_chain(self, event: str, **fields: Any) -> None:
        """Emit one grep-friendly `[chain]` line when --debug-chain is on.
        Used to attribute "EE ends up at hover(B) but next push goes to C"
        events to one of the three known clearing paths (env pre-amble,
        demo-policy planner reject, push failure)."""
        if not self._debug_chain_log:
            return
        try:
            sim_time = float(getattr(self.env, "sim_time", 0.0))
        except Exception:
            sim_time = 0.0
        try:
            ee = np.asarray(self.env.get_site_position("grip_site"), dtype=np.float64)
            ee_str = f"({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f})"
        except Exception:
            ee_str = "(?,?,?)"
        parts = [
            f"[chain] T={sim_time:.2f}",
            f"step={self._step_count}",
            f"event={event}",
            f"EE={ee_str}",
        ]
        for key, value in fields.items():
            parts.append(f"{key}={value}")
        print(" ".join(parts), flush=True)

    def _slot_id(self, slot: Any | None) -> str:
        if slot is None:
            return "None"
        spawn_id = getattr(slot, "spawn_id", None)
        return "None" if spawn_id is None else str(spawn_id)

    def _name_sort_key(self, name: str, prefix: str) -> tuple[int, str]:
        match = re.fullmatch(rf"{re.escape(prefix)}(?:_(\d+))?", name)
        if match is None:
            return (10**9, name)
        suffix = match.group(1)
        return (1 if suffix is None else int(suffix), name)

    def _build_object_slots(self) -> list[ObjectSlot]:
        freejoint_names = sorted(
            [name for name in self.env.joint_ids if name.startswith("box_free")],
            key=lambda name: self._name_sort_key(name, "box_free"),
        )
        site_names = sorted(
            [name for name in self.env.site_ids if name.startswith("box_site")],
            key=lambda name: self._name_sort_key(name, "box_site"),
        )
        slot_count = min(len(freejoint_names), len(site_names))
        if slot_count == 0:
            raise RuntimeError("No reusable object slots found in the MuJoCo model.")

        slots: list[ObjectSlot] = []
        self.env.reset()
        for index, (freejoint_name, site_name) in enumerate(zip(freejoint_names[:slot_count], site_names[:slot_count])):
            _, quat = self.env.get_freejoint_pose(freejoint_name)
            suffix = "" if index == 0 else f"_{index + 1}"
            slots.append(
                ObjectSlot(
                    freejoint_name=freejoint_name,
                    site_name=site_name,
                    body_name=f"red_box{suffix}",
                    geom_name=f"red_box_geom{suffix}",
                    equality_name=f"suction_weld_red_box{suffix}",
                    home_quat=quat.copy(),
                )
            )
        return slots

    def _build_object_geom_sizes_xy(self) -> list[np.ndarray]:
        sizes_xy: list[np.ndarray] = []
        for index in range(len(self.object_slots)):
            geom_name = "red_box_geom" if index == 0 else f"red_box_geom_{index + 1}"
            geom_size = np.asarray(self.env.model.geom(geom_name).size[:2], dtype=np.float32)
            sizes_xy.append((2.0 * geom_size).astype(np.float32))
        return sizes_xy

    def _load_throwing_model(self) -> Any:
        if torch is None or FCN is None:
            return None
        model = FCN([5, 256, 512, 256, 52])
        state_dict = torch.load(self.throwing_model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _build_pushing_trajectory_generator(self) -> Any:
        if not self.pushing_enabled or PushingTrajectoryGenerator is None or self.robot_kinematics is None:
            return None
        velocity_limits = np.array([bounds[1] for bounds in self.robot_kinematics.jointvel_bounds], dtype=np.float64)
        m1 = velocity_limits * self.pushing_velocity_limit_scale
        m2 = m1 * 2.0
        m2h = m1 * 70.0
        return PushingTrajectoryGenerator(
            self.robot_kinematics,
            m1,
            m2,
            m2h,
            hertz=1.0 / self.env.dt,
            hover_height=self.pushing_hover_height,
        )

    def _neutral_tracking(self) -> TrackingState:
        q_actual = self.env.get_joint_positions(self.joint_names).astype(np.float64)
        return TrackingState(
            detected=False,
            error_px=np.array([np.nan, np.nan], dtype=np.float64),
            aligned=False,
            joint_delta=np.zeros_like(self.q_ready, dtype=np.float64),
            q_target=q_actual,
        )

    def _tracking_from_detection(self, detection: Detection) -> TrackingState:
        tracking = self._neutral_tracking()
        return TrackingState(
            detected=bool(detection.detected),
            error_px=tracking.error_px,
            aligned=False,
            joint_delta=tracking.joint_delta,
            q_target=tracking.q_target.copy(),
        )

    def _select_primary_detection(self, detections: list[Detection]) -> Detection:
        if not detections:
            return Detection(False, None, None, 0.0, None)
        target = np.asarray(self.target_px, dtype=np.float64)
        return min(
            detections,
            key=lambda detection: float(np.linalg.norm(np.asarray(detection.center_px, dtype=np.float64) - target))
            if detection.center_px is not None
            else float("inf"),
        )

    def _capture_detection(self) -> tuple[np.ndarray, Detection, list[Detection]]:
        if not self.use_camera:
            frame_bgr = np.zeros((self.env.cam_h, self.env.cam_w, 3), dtype=np.uint8)
            detection = Detection(False, None, None, 0.0, None)
            detections: list[Detection] = []
            self._last_frame_bgr = frame_bgr.copy()
            return frame_bgr, detection, detections
        frame_bgr = self.env.render_bgr(flip_code=self.flip_code)
        detections = self.detector.detect_all(frame_bgr)
        detection = self._select_primary_detection(detections)
        self._last_frame_bgr = frame_bgr.copy()
        return frame_bgr, detection, detections

    def _hide_object_slot(self, slot: ObjectSlot) -> None:
        if self.gripper.is_attached() and self.gripper.target_freejoint == slot.freejoint_name:
            self.gripper.set_enabled(False)
        self.env.set_freejoint_pose(slot.freejoint_name, self.hidden_position, slot.home_quat)
        joint_id = self.env.require_joint(slot.freejoint_name)
        qvel_adr = self.env.model.jnt_dofadr[joint_id]
        self.env.data.qvel[qvel_adr : qvel_adr + 6] = 0.0
        slot.active = False
        slot.manipulated = False
        slot.manipulated_time = None
        slot.blocked = False
        slot.spawn_id = None
        slot.target_xy = None
        slot.is_obstacle = False

    def _sample_target_xy(self) -> np.ndarray:
        candidate = self.target_xy_candidates[self._target_sample_counter % len(self.target_xy_candidates)]
        self._target_sample_counter += 1
        return np.asarray(candidate, dtype=np.float64).copy()

    def _object_size_for_spawn(self, slot_index: int) -> tuple[float, float, float]:
        """Half-extents (hx, hy, hz) for the box spawning into `slot_index`.
        Footprint is randomized per spawn (drawn from self._rng, so it is
        reproducible for a given env seed); height is fixed at `_obj_half_height`.
        With `randomize_object_size=False`, each slot instead gets a fixed size
        spread deterministically across the ranges (debugging mode)."""
        xr = self._obj_size_x_half_range
        yr = self._obj_size_y_half_range
        if self.randomize_object_size:
            hx = float(self._rng.uniform(xr[0], xr[1]))
            hy = float(self._rng.uniform(yr[0], yr[1]))
        else:
            n = max(len(self.object_slots) - 1, 1)
            frac = float(slot_index) / float(n)
            hx = xr[0] + frac * (xr[1] - xr[0])
            hy = yr[0] + frac * (yr[1] - yr[0])
        return hx, hy, self._obj_half_height

    def _apply_object_size(self, slot_index: int, hx: float, hy: float, hz: float) -> None:
        """Resize the slot's box geom + body inertia LIVE so the per-spawn size
        takes effect without recompiling the model (geom_size/rbound and
        body_mass/inertia are read by MuJoCo each step). Also refreshes the
        observation size cache so the obs reports the true current footprint."""
        geom_name = "red_box_geom" if slot_index == 0 else f"red_box_geom_{slot_index + 1}"
        body_name = "red_box" if slot_index == 0 else f"red_box_{slot_index + 1}"
        gid = self.env.require_geom(geom_name)
        bid = self.env.require_body(body_name)
        self.env.model.geom_size[gid] = (hx, hy, hz)
        self.env.model.geom_rbound[gid] = float(np.sqrt(hx * hx + hy * hy + hz * hz))
        m = self._object_mass
        self.env.model.body_mass[bid] = m
        # Solid-box inertia (full extents 2h): Ixx = m/12*((2hy)^2+(2hz)^2) = m/3*(hy^2+hz^2)
        self.env.model.body_inertia[bid] = (
            m / 3.0 * (hy * hy + hz * hz),
            m / 3.0 * (hx * hx + hz * hz),
            m / 3.0 * (hx * hx + hy * hy),
        )
        self.object_geom_sizes_xy[slot_index] = np.array([2.0 * hx, 2.0 * hy], dtype=np.float32)

    def set_object_object_collisions(self, enabled: bool) -> None:
        """DEBUG ONLY: enable/disable box<->box contacts at runtime to isolate the
        chaining algorithm from object-object collision noise.

        belt<->box and pusher<->box are explicit <pair>s (always active regardless
        of bitmasks); box<->bin and box<->robot use default contacts. box<->box has
        no explicit pair, so it is governed by contype/conaffinity. Setting every
        box geom to contype=2, conaffinity=1 disables box<->box only:
          box<->box: (2&1)|(2&1)=0  -> OFF
          box<->bin: (bin.contype 1 & box.conaff 1)=1 -> ON (objects still land)
        Default (enabled=True) restores contype=conaffinity=1. No XML edit."""
        contype = 1 if enabled else 2
        conaffinity = 1
        for slot in self.object_slots:
            gid = self.env.require_geom(slot.geom_name)
            self.env.model.geom_contype[gid] = contype
            self.env.model.geom_conaffinity[gid] = conaffinity

    def _sample_spawn_yaw(self) -> float:
        """In-plane spawn yaw (rad). Random in [-pi, pi] when enabled, else 0."""
        if not self.randomize_object_yaw:
            return 0.0
        return float(self._rng.uniform(-np.pi, np.pi))

    def _compose_yaw_quat(self, home: np.ndarray, theta: float) -> np.ndarray:
        """Apply a world-z yaw of `theta` to `home`: q = q_yaw (x) home. Yaw only
        keeps the box flat on the belt. MuJoCo quat order is (w, x, y, z)."""
        home = np.asarray(home, dtype=np.float64)
        if theta == 0.0:
            return home
        c, s = float(np.cos(0.5 * theta)), float(np.sin(0.5 * theta))
        w2, x2, y2, z2 = float(home[0]), float(home[1]), float(home[2]), float(home[3])
        return np.array(
            [c * w2 - s * z2, c * x2 - s * y2, c * y2 + s * x2, c * z2 + s * w2],
            dtype=np.float64,
        )

    @staticmethod
    def _boxes_overlap_2d(
        c1: np.ndarray, h1: tuple[float, float], th1: float,
        c2: np.ndarray, h2: tuple[float, float], th2: float,
    ) -> bool:
        """2D oriented-bounding-box overlap via the Separating Axis Theorem.
        c* = centers (x,y); h* = (half_x, half_y); th* = yaw (rad). Boxes overlap
        iff no separating axis exists among the 4 box-edge normals."""
        d = np.asarray(c2, dtype=np.float64)[:2] - np.asarray(c1, dtype=np.float64)[:2]
        u1 = np.array([np.cos(th1), np.sin(th1)]); v1 = np.array([-np.sin(th1), np.cos(th1)])
        u2 = np.array([np.cos(th2), np.sin(th2)]); v2 = np.array([-np.sin(th2), np.cos(th2)])
        for axis in (u1, v1, u2, v2):
            r1 = h1[0] * abs(float(u1 @ axis)) + h1[1] * abs(float(v1 @ axis))
            r2 = h2[0] * abs(float(u2 @ axis)) + h2[1] * abs(float(v2 @ axis))
            if abs(float(d @ axis)) > r1 + r2:
                return False  # separated on this axis -> no overlap
        return True

    def _spawn_pose_is_clear(self, x: float, y: float, hx: float, hy: float, theta: float) -> bool:
        """True if a box of half-extents (hx,hy) at (x,y) with yaw `theta` does not
        overlap any ACTIVE object's footprint, inflated by `spawn_clearance_margin`."""
        m = self.spawn_clearance_margin
        cand_c = np.array([x, y], dtype=np.float64)
        cand_h = (hx + m, hy + m)  # inflate candidate -> keep a margin gap
        for slot in self.object_slots:
            if not slot.active:
                continue
            pos, quat = self.env.get_freejoint_pose(slot.freejoint_name)
            gid = self.env.require_geom(slot.geom_name)
            gsize = self.env.model.geom_size[gid]
            ex_h = (float(gsize[0]), float(gsize[1]))
            w, qx, qy, qz = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
            ex_yaw = float(np.arctan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))
            if self._boxes_overlap_2d(cand_c, cand_h, theta, pos[:2], ex_h, ex_yaw):
                return False
        return True

    def _spawn_object_slot(self, slot: ObjectSlot) -> bool:
        """Spawn `slot` with a randomized footprint + yaw at a CLEAR pose on the
        spawn line. Tries up to `spawn_max_placement_tries` random x lanes; the
        first that does not overlap an active object's footprint (OBB clearance)
        is used. Returns False (deferring the spawn, model untouched) if no clear
        pose is found this tick -- so the belt self-throttles under congestion
        instead of piling objects up. Because all objects share the belt speed
        (no relative motion), a clear pose at spawn stays clear down the belt."""
        slot_index = self.object_slots.index(slot)
        hx, hy, hz = self._object_size_for_spawn(slot_index)
        theta = self._sample_spawn_yaw()
        chosen_x: float | None = None
        for _ in range(self.spawn_max_placement_tries):
            x = float(self._rng.uniform(self.spawn_x_range[0], self.spawn_x_range[1]))
            if self._spawn_pose_is_clear(x, self.spawn_y, hx, hy, theta):
                chosen_x = x
                break
        if chosen_x is None:
            return False  # no clear lane -> defer spawn (no model mutation)

        self._apply_object_size(slot_index, hx, hy, hz)
        # Rest the (flat) box on the belt surface: center z = surface + half-height.
        pos = np.array([chosen_x, self.spawn_y, self._belt_surface_z + hz], dtype=np.float64)
        self.env.set_freejoint_pose(slot.freejoint_name, pos, self._compose_yaw_quat(slot.home_quat, theta))
        joint_id = self.env.require_joint(slot.freejoint_name)
        qvel_adr = self.env.model.jnt_dofadr[joint_id]
        # Give fresh objects the conveyor's nominal downstream motion once at spawn.
        self.env.data.qvel[qvel_adr : qvel_adr + 3] = np.array([0.0, -self.belt_speed, 0.0], dtype=np.float64)
        # Obstacle objects belong to NO bin: no target, flagged in the obs.
        slot.is_obstacle = bool(float(self._rng.random()) < self.obstacle_prob)
        slot.target_xy = None if slot.is_obstacle else self._sample_target_xy()
        # Per-object suction success RATE (drawn from the configured range); the
        # realized Bernoulli outcome is rolled lazily at the grasp instant.
        _sl, _sh = self.suction_success_rate_range
        slot.suction_p = 1.0 if (_sl == 1.0 and _sh == 1.0) else float(self._rng.uniform(_sl, _sh))
        slot.suction_ok = None  # unrolled
        slot.active = True
        slot.manipulated = False
        slot.manipulated_time = None
        slot.blocked = False
        slot.spawn_id = self._spawn_counter
        self._spawn_counter += 1
        return True

    def _object_in_its_bin(self, slot: ObjectSlot, pos: np.ndarray) -> bool:
        """True if the object is ACTUALLY inside its own PHYSICAL bin volume --
        i.e. within the bin's interior footprint (between the inner wall faces)
        and below the wall rim. Centered on the bin (decoupled from the aim
        target), with size-dependent extents. The eval uses the same test."""
        bin_c = self._bin_center_world_xy_for_slot(slot)
        half = self._bin_interior_half
        return bool(
            abs(float(pos[0]) - float(bin_c[0])) <= float(half[0])
            and abs(float(pos[1]) - float(bin_c[1])) <= float(half[1])
            and self._bin_z_lo <= float(pos[2]) <= self._bin_z_hi
        )

    def _resolved_object_reward(self, slot: ObjectSlot, pos: np.ndarray) -> float:
        """Outcome reward credited when an object resolves (despawns):
          +1.0  intended (manipulated) object landed in its bin,
          -0.3  intended object left the belt NOT in its bin (missed),
          -0.3  non-intended object knocked off the conveyor (fell below the belt),
           0.0  non-intended object rode off the end untouched (natural).
        A non-target object only counts as 'knocked off' if it fell below the belt
        surface (z-drop); riding to the normal downstream end is neutral."""
        if getattr(slot, "manipulated", False):
            return 1.0 if self._object_in_its_bin(slot, pos) else -0.3
        if float(pos[2]) < self.object_elimination_z:
            return -0.3
        return 0.0

    def _despawn_finished_objects(self) -> None:
        for slot in self.object_slots:
            if not slot.active:
                continue
            if self._slot_is_attached(slot):
                continue
            pos, _ = self.env.get_freejoint_pose(slot.freejoint_name)
            # Bin-entry resolution: the moment a manipulated object has SETTLED
            # inside its bin, credit the +1 and despawn it right away. This latches
            # the success (a later fall-through can't undo it) and stops the object
            # from outliving the single bin mocap once the bin moves to the next
            # target. "Settled" = in the bin volume AND at rest (|v| < 0.1 m/s),
            # which avoids crediting objects that merely fly/fall through the bin's
            # airspace (belt/flight speeds are >= belt_speed = 0.4).
            if getattr(slot, "manipulated", False) and self._object_in_its_bin(slot, pos):
                if float(np.linalg.norm(self._slot_linear_velocity_xyz(slot))) < 0.1:
                    self._reward_accumulator += 1.0
                    self._hide_object_slot(slot)
                    continue
            if slot.manipulated:
                age = None if slot.manipulated_time is None else float(self.env.sim_time - slot.manipulated_time)
                should_hide = (
                    float(pos[1]) <= self.despawn_y
                    or float(pos[2]) < self.object_elimination_z
                    or (age is not None and age >= self.manipulated_object_lifetime)
                )
            elif getattr(slot, "blocked", False):
                should_hide = (
                    float(pos[1]) <= self.stale_object_despawn_y
                    or float(pos[1]) <= self.despawn_y
                    or float(pos[2]) < self.object_elimination_z
                )
            else:
                should_hide = (
                    float(pos[1]) <= self.despawn_y
                    or float(pos[2]) < self.object_elimination_z
                )
            if should_hide:
                # Credit the object's resolved outcome to this step's reward.
                self._reward_accumulator += self._resolved_object_reward(slot, pos)
                self._hide_object_slot(slot)

    def _mark_missed_objects(self) -> None:
        for slot in self.object_slots:
            if not slot.active or getattr(slot, "manipulated", False) or getattr(slot, "blocked", False):
                continue
            if self._slot_is_attached(slot):
                continue
            obj_xyz = self._slot_grasp_throw_xyz(slot)
            radial_distance = float(np.linalg.norm(obj_xyz[:2]))
            if radial_distance > self.action_reachable_radius and float(obj_xyz[1]) < 0.0:
                slot.blocked = True

    def _bin_geom_ids(self) -> list[tuple[str, int]]:
        """(name, id) of every physical-bin geom (name-prefixed 'target_bin').
        The bin is assumed axis-aligned (identity home orientation), as in
        combined_test.xml, so geom local x/y/z map to world x/y/z."""
        import mujoco
        model = self.env.model
        out = []
        for g in range(model.ngeom):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
            if nm and nm.startswith("target_bin"):
                out.append((nm, g))
        return out

    def _bin_near_half_extent_x(self) -> float:
        """Distance from the bin body center to its CLOSEST (-x) face (so bin
        placement is size-dependent)."""
        model = self.env.model
        near = 0.0
        for _nm, g in self._bin_geom_ids():
            near = min(near, float(model.geom_pos[g][0] - model.geom_size[g][0]))
        return -near

    def _bin_outer_full_xy(self) -> np.ndarray:
        """Bin OUTER footprint full extents (x, y) -- the bin's world-axis-aligned
        bounding box, analogous to an object's aabb_w/aabb_h. Size-dependent."""
        model = self.env.model
        hx = hy = 0.0
        for _nm, g in self._bin_geom_ids():
            hx = max(hx, abs(float(model.geom_pos[g][0])) + float(model.geom_size[g][0]))
            hy = max(hy, abs(float(model.geom_pos[g][1])) + float(model.geom_size[g][1]))
        return np.array([2.0 * hx, 2.0 * hy], dtype=np.float64)

    def _bin_interior_half_xy(self) -> np.ndarray:
        """Bin INTERIOR half-extents (x, y) -- the usable footprint between the
        inner wall faces. Used by the 'object actually inside the bin' metric.
        Read from the wall geoms; falls back to the outer footprint if absent."""
        model = self.env.model
        ix = iy = float("inf")
        for nm, g in self._bin_geom_ids():
            if "wall_pos_x" in nm or "wall_neg_x" in nm:
                ix = min(ix, abs(float(model.geom_pos[g][0])) - float(model.geom_size[g][0]))
            if "wall_pos_y" in nm or "wall_neg_y" in nm:
                iy = min(iy, abs(float(model.geom_pos[g][1])) - float(model.geom_size[g][1]))
        half = 0.5 * self._bin_outer_full_xy()
        return np.array([ix if np.isfinite(ix) else half[0],
                         iy if np.isfinite(iy) else half[1]], dtype=np.float64)

    def _bin_z_bounds_world(self) -> tuple[float, float]:
        """World z-range of the bin interior: floor (base) top to the lowest wall
        rim. An object 'inside' must lie within this height. Size-dependent."""
        model = self.env.model
        floor_top = -float("inf")
        rim = float("inf")
        for nm, g in self._bin_geom_ids():
            top = float(model.geom_pos[g][2]) + float(model.geom_size[g][2])
            if "base" in nm:
                floor_top = max(floor_top, top)
            elif "wall" in nm:
                rim = min(rim, top)
        z0 = self.target_bin_spawn_z + (floor_top if np.isfinite(floor_top) else -0.02)
        z1 = self.target_bin_spawn_z + (rim if np.isfinite(rim) else 0.48)
        return float(z0), float(z1)

    def _bin_center_robot_xy_for_slot(self, slot: ObjectSlot | None = None) -> np.ndarray:
        """Physical bin CENTER (robot frame), DECOUPLED from the aim target: x is
        size-dependent (near -x face at `_bin_near_edge_x`); y is SHARED with the
        slot's aim target (only x is decoupled)."""
        if slot is not None and slot.target_xy is not None:
            aim_y = float(slot.target_xy[1])
        else:
            aim_y = float(self.target_xy[1])
        return np.array([self._bin_center_x_robot, aim_y], dtype=np.float64)

    def _bin_center_world_xy_for_slot(self, slot: ObjectSlot | None = None) -> np.ndarray:
        return (self.robot_base_xy + self._bin_center_robot_xy_for_slot(slot)).astype(np.float64)

    def _target_world_xy_for_slot(self, slot: ObjectSlot | None = None) -> np.ndarray:
        if slot is not None and slot.target_xy is not None:
            return (self.robot_base_xy + np.asarray(slot.target_xy, dtype=np.float64)).astype(np.float64)
        return (self.robot_base_xy + self.target_xy).astype(np.float64)

    def _show_target_bin_for_slot(self, slot: ObjectSlot | None = None) -> None:
        # Place the PHYSICAL bin at its own (size-dependent) center -- decoupled
        # from the aim target (`_target_world_xy_for_slot`).
        bin_world_xy = self._bin_center_world_xy_for_slot(slot)
        target_bin_pos = np.array(
            [bin_world_xy[0], bin_world_xy[1], self.target_bin_spawn_z],
            dtype=np.float64,
        )
        self.env.set_mocap_body_pose("target_bin", target_bin_pos, self.target_bin_home_quat)
        self._target_bin_active = True

    def _hide_target_bin(self) -> None:
        self.env.set_mocap_body_pose("target_bin", self.hidden_position, self.target_bin_home_quat)
        self._target_bin_active = False

    def _maybe_spawn_object(self) -> None:
        if not self.spawn_enabled or self.spawn_rate_hz <= 0.0:
            return
        # NOTE: _maybe_spawn_object is called PER SIM TICK from
        # `_advance_simulation` (action_control.py:1446), so the probability
        # must be per-tick (= spawn_rate_hz * dt), NOT per gym step. Previous
        # version multiplied by frame_skip too, which produced an effective
        # spawn rate 10x higher than requested.
        spawn_probability = min(1.0, self.spawn_rate_hz * self.env.dt)
        if float(self._rng.random()) >= spawn_probability:
            return

        for slot in self.object_slots:
            if not slot.active:
                self._spawn_object_slot(slot)
                break

    def _ordered_active_world_boxes(self) -> list[np.ndarray]:
        active_boxes: list[np.ndarray] = []
        for slot in self._ordered_active_slots():
            slot_index = self.object_slots.index(slot)
            size_xy = self.object_geom_sizes_xy[slot_index]
            # Report the object CENTROID (body center) as the detection position,
            # matching what a bbox detector returns -- not the grasp `box_site`
            # (which is offset 0.04 m in the box's local -y and now rotates with
            # yaw). Control reads the centroid directly from sim and is unaffected.
            # ROBOT frame (world centroid minus the robot base) so EVERY obs
            # position field shares ONE frame (centroid, aim target, bin center).
            center = self._slot_center_world_xyz(slot).astype(np.float64)
            center[:2] -= self.robot_base_xy
            center = center.astype(np.float32)
            # Report the AXIS-ALIGNED BOUNDING BOX that contains the (rotated)
            # object -- what a top-down bbox detector sees -- instead of the true
            # local size. AABB-of-OBB: project the box half-extents onto world
            # x/y via the body rotation matrix. Handles yaw and any tilt. For pure
            # yaw t this is w = sx|cos t| + sy|sin t|, h = sx|sin t| + sy|cos t|.
            R = self.env.get_body_rotation(slot.body_name)
            hx, hy, hz = 0.5 * float(size_xy[0]), 0.5 * float(size_xy[1]), float(self._obj_half_height)
            aabb_w = 2.0 * (abs(R[0, 0]) * hx + abs(R[0, 1]) * hy + abs(R[0, 2]) * hz)
            aabb_h = 2.0 * (abs(R[1, 0]) * hx + abs(R[1, 1]) * hy + abs(R[1, 2]) * hz)
            # is_obstacle: 1 = object belongs to no bin (leave it on the belt).
            # Obstacles carry a don't-care target sentinel (0,0); the flag is the
            # signal. Binnable objects report their (continuous) bin coordinates.
            if slot.is_obstacle:
                is_obstacle = 1.0
                target_xy = np.zeros(2, dtype=np.float32)
                # Obstacles belong to no bin -> bin fields are a don't-care (0,0,0,0).
                bin_xy = np.zeros(2, dtype=np.float32)
                bin_wh = np.zeros(2, dtype=np.float32)
                suction_p = 0.0  # obstacles are never grasped
            else:
                is_obstacle = 0.0
                target_xy = (
                    self.target_xy.astype(np.float32)
                    if slot.target_xy is None
                    else np.asarray(slot.target_xy, dtype=np.float32)
                )
                # Physical bin: center (ROBOT frame) + outer AABB full extents,
                # decoupled from the aim target above. Mirrors the object's
                # center/aabb fields; same frame as the centroid and the target.
                bin_xy = self._bin_center_robot_xy_for_slot(slot).astype(np.float32)
                bin_wh = self._bin_outer_full.astype(np.float32)
                suction_p = float(slot.suction_p)  # throw-grasp success RATE
            # Measured per-object speed (belt-direction forward speed, >=0) -- the
            # measured-velocity channel alongside the global nominal belt_speed.
            v_measured = float(self._slot_forward_speed(slot))
            active_boxes.append(
                np.array(
                    [center[0], center[1], aabb_w, aabb_h,
                     target_xy[0], target_xy[1], is_obstacle, v_measured,
                     suction_p, bin_xy[0], bin_xy[1], bin_wh[0], bin_wh[1]],
                    dtype=np.float32,
                )
            )
        return active_boxes[: self.max_detections]

    def _observation_from_detections(self, detections: list[Detection]) -> np.ndarray:
        obs = np.concatenate(
            (
                np.tile(
                    np.array([-1.0, -1.0, 0.0, 0.0, self.target_xy[0], self.target_xy[1], 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    self.max_detections,
                ),
                np.array([self.belt_speed, self.spawn_rate_hz], dtype=np.float32),
            )
        )
        del detections
        for idx, world_box in enumerate(self._ordered_active_world_boxes()):
            obs[idx * 13 : (idx + 1) * 13] = world_box
        return obs

    def _initialize_belt_motion(self) -> None:
        self.env.set_actuator_ctrl("slider_act", self._belt_actuator_command())
        joint_id = self.env.require_joint("slider_joint")
        qvel_adr = self.env.model.jnt_dofadr[joint_id]
        self.env.data.qvel[qvel_adr] = self._belt_actuator_command()

    def _build_info(self, detection: Detection, detections: list[Detection], tracking: TrackingState) -> dict[str, Any]:
        bbox = None if detection.bbox is None else np.asarray(detection.bbox, dtype=np.int32)
        center_px = None if detection.center_px is None else np.asarray(detection.center_px, dtype=np.float32)
        all_bboxes = [
            None if current_detection.bbox is None else np.asarray(current_detection.bbox, dtype=np.int32)
            for current_detection in detections[: self.max_detections]
        ]
        all_centers = [
            None if current_detection.center_px is None else np.asarray(current_detection.center_px, dtype=np.float32)
            for current_detection in detections[: self.max_detections]
        ]
        box_world_xyz = []
        box_world_xy = []
        ordered_spawn_ids = []
        for slot in self._ordered_active_slots():
            site_pos = self.env.get_site_position(slot.site_name).astype(np.float32)
            box_world_xyz.append(site_pos)
            box_world_xy.append(site_pos[:2].copy())
            ordered_spawn_ids.append(int(slot.spawn_id))
        active_object_count = len(box_world_xyz)
        detected_box_world_xy = [box[:2].copy() for box in self._ordered_active_world_boxes()]
        detected_box_world_boxes = [box.copy() for box in self._ordered_active_world_boxes()]
        # RL action mask. Shape (max_detections+1, 2) uint8. Row = slot index
        # (same ordering as the obs bboxes), col 0 = throw, col 1 = push. Row
        # `max_detections` is the skip sentinel and is ALWAYS 1 (always
        # available). For each addressable slot, 1 = (slot, primitive) is
        # guaranteed executable by the env (single source of truth via
        # `_compute_slot_feasibility_flags`), 0 = infeasible.
        action_mask = np.zeros((self.max_detections + 1, 2), dtype=np.uint8)
        action_mask[self.max_detections, :] = 1  # skip is always feasible
        for idx, prims in self._slot_feasibility_flags.items():
            action_mask[idx, 0] = int(bool(prims.get("throw", False)))
            action_mask[idx, 1] = int(bool(prims.get("push", False)))
        return {
            "detected": bool(detection.detected),
            "bbox": bbox,
            "center_px": center_px,
            "action_mask": action_mask,
            "detections": all_bboxes,
            "centers_px": all_centers,
            "detected_box_world_xy": detected_box_world_xy,
            "detected_box_world_boxes": detected_box_world_boxes,
            "num_detections": int(len(detections)),
            "active_object_count": int(active_object_count),
            "ordered_spawn_ids": ordered_spawn_ids,
            "conveyor_speed": float(self.belt_speed),
            "spawn_rate_hz": float(self.spawn_rate_hz),
            "target_bin_active": bool(self._target_bin_active),
            "target_bin_world_xy": self._bin_center_world_xy_for_slot().astype(np.float32),
            "pending_target_spawn_id": (
                None
                if self._pending_target_slot is None or self._pending_target_slot.spawn_id is None
                else int(self._pending_target_slot.spawn_id)
            ),
            "area": float(detection.area),
            "target_px": np.asarray(self.target_px, dtype=np.float32),
            "error_px": tracking.error_px.astype(np.float32),
            "aligned": bool(tracking.aligned),
            "q_target": tracking.q_target.astype(np.float32),
            "q_actual": self.env.get_joint_positions(self.joint_names).astype(np.float32),
            "box_world_xy": box_world_xy,
            "box_world_xyz": box_world_xyz,
            "sim_time": float(self.env.sim_time),
            "step_count": int(self._step_count),
        }

    def render_tracking_overlay(
        self,
        frame_bgr: np.ndarray | None = None,
        detection: Detection | None = None,
        detections: list[Detection] | None = None,
        tracking: TrackingState | None = None,
    ) -> np.ndarray:
        if cv2 is None:
            raise ImportError("render_tracking_overlay requires opencv-python to be installed.")

        if frame_bgr is None:
            if self._last_frame_bgr is None:
                frame_bgr, detection_from_frame = self._capture_detection()
                detection = detection or detection_from_frame
            else:
                frame_bgr = self._last_frame_bgr.copy()
        else:
            frame_bgr = frame_bgr.copy()

        if detections is None:
            detections = self._last_detections if self._last_detections else self.detector.detect_all(frame_bgr)
        if detection is None:
            detection = self._select_primary_detection(detections)

        if tracking is None:
            tracking = self._last_tracking if self._last_tracking is not None else self._neutral_tracking()

        primary_index = None
        if detection.detected and detection.bbox is not None:
            for idx, current_detection in enumerate(detections):
                if current_detection.bbox == detection.bbox:
                    primary_index = idx
                    break

        overlay_bgr = self.detector.draw_detections(
            frame_bgr,
            detections=detections[: self.max_detections],
            target_px=self.target_px,
            primary_index=primary_index,
        )
        text_color = (0, 255, 0) if tracking.aligned else (0, 255, 255)
        dx_val = float(tracking.error_px[0]) if not np.isnan(tracking.error_px[0]) else 0.0
        dy_val = float(tracking.error_px[1]) if not np.isnan(tracking.error_px[1]) else 0.0
        h, w = overlay_bgr.shape[:2]
        text_x = max(10, w - 190)
        cv2.putText(
            overlay_bgr,
            f"dx={dx_val:.1f} dy={dy_val:.1f}",
            (text_x, h - 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            text_color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay_bgr,
            f"aligned={tracking.aligned}",
            (text_x, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            text_color,
            1,
            cv2.LINE_AA,
        )
        return overlay_bgr

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if gym is not None:
            super_reset = getattr(super(), "reset", None)
            if callable(super_reset):
                super_reset(seed=seed)
                if hasattr(self, "np_random") and self.np_random is not None:
                    self._rng = self.np_random

        self.env.reset()
        self.env.set_joint_positions(self.joint_names, self.q_ready)
        self.gripper.release()
        self.gripper.set_enabled(False)
        self._spawn_counter = 0
        self._target_sample_counter = 0
        self._reward_accumulator = 0.0
        self.controller.hold_position(self.q_ready.copy())
        self._step_count = 0
        self._seen_detection = False
        self._last_frame_bgr = None
        self._last_detections = []
        self._pending_target_slot = None
        self._pending_action_primitive = None
        self._push_routed_spawn_id = None
        self._push_routed_target_xyz = None
        self._demo_next_primitive = self.ACTION_THROW
        self._pushing_slot = None
        for slot in self.object_slots:
            self._hide_object_slot(slot)
        if self.pusher_collision_name is not None:
            self.env.set_geom_collision_enabled(self.pusher_collision_name, False)
        self._hide_target_bin()
        self._initialize_belt_motion()

        _, detection, detections = self._capture_detection()
        tracking = self._tracking_from_detection(detection)
        self._last_detection = detection
        self._last_detections = detections
        self._last_tracking = tracking

        obs = self._observation_from_detections(detections)
        self._slot_feasibility_flags = self._compute_slot_feasibility_flags()
        info = self._build_info(detection, detections, tracking)
        return obs, info

    def step(self, action: int | np.integer | np.ndarray | tuple[int, int] | list[int] | dict[str, Any] | None = None) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # 1-step-ahead semantics: action[N] = (next_slot, next_primitive). This
        # step executes the manipulation set up by action[N-1] (the pending
        # slot/primitive below), then routes the EE toward action[N]'s entry
        # pose so step N+1's manipulation can start there. Step 0 (and any step
        # whose pending was cleared) has no pending → it is route-only.
        pending_slot = self._pending_target_slot
        pending_primitive = self._pending_action_primitive
        if pending_slot is not None and not self._slot_in_action_window(pending_slot):
            self._log_chain(
                "pending_dropped",
                dropped_slot=self._slot_id(pending_slot),
                active=bool(pending_slot.active),
                blocked=bool(getattr(pending_slot, "blocked", False)),
            )
            pending_slot = None
            pending_primitive = None
            self._pending_target_slot = None
            self._pending_action_primitive = None

        execution_success = False
        execution_info: dict[str, Any] = {"reason": "step_uninitialized"}
        # True on route-only / skip steps (no manipulation attempted).
        route_only = False
        # Outcome-based reward: accumulated as objects RESOLVE (despawn) during this
        # step's sim ticks (see `_resolved_object_reward`). Reset here, read at the
        # end of step(). Reward is the object's fate, not the act of manipulating.
        self._reward_accumulator = 0.0

        action_index, next_primitive = self._parse_action(action)
        if self._last_detection is not None and (self._last_frame_bgr is not None or not self.use_camera):
            detection = self._last_detection
        else:
            _, detection, detections = self._capture_detection()
            self._last_detections = detections

        # action[N]'s chain target (the NEXT object to manipulate). The skip
        # sentinel (action_index == max_detections) means "no next" → the EE is
        # routed to standby by the manipulation's after-trajectory.
        next_slot: ObjectSlot | None = None
        if action_index != self.max_detections:
            next_slot = self._selected_slot_from_action(action_index)

        # Show the bin for the object actually engaged THIS step (the pending
        # one being manipulated), or — at a route-only step — the slot being
        # approached.
        bin_slot = pending_slot if (pending_slot is not None and pending_slot.active) else next_slot
        if bin_slot is not None:
            self._show_target_bin_for_slot(bin_slot)

        try:
            if pending_slot is None or pending_primitive is None:
                # ── Route-only step (bootstrap / cleared pending) ──
                route_only = True
                if next_slot is not None:
                    execution_success, execution_info = self._route_to_action_entry(
                        next_slot, next_primitive
                    )
                else:
                    # No pending, no next → idle. Hold the controller's last
                    # commanded target (not the sagged pose) to avoid the idle
                    # ratchet-down drift (see tmp_idle_drift_long.py, 2026-05-22).
                    self.controller.hold_position(self.controller.last_target)
                    self._advance_simulation(self.frame_skip)
                    execution_success = False
                    execution_info = {"reason": "skip"}
            else:
                # ── Manipulation step: finish action[N-1], route toward action[N] ──
                execution_success, execution_info = self._execute_pending_manipulation(
                    pending_slot, pending_primitive, next_slot, next_primitive
                )
                self._log_chain(
                    "manipulation_outcome",
                    primitive=("push" if int(pending_primitive) == self.ACTION_PUSH else "throw"),
                    pending_slot=self._slot_id(pending_slot),
                    next_slot=self._slot_id(next_slot),
                    outcome=("success" if execution_success else "failure"),
                    reason=str(execution_info.get("reason", "")),
                )
                if not execution_success:
                    self._recover_after_action_failure(execution_info)
        finally:
            self._hide_target_bin()

        # Queue the pending manipulation for the next step from where the EE was
        # ACTUALLY routed this step. The agent's action[N] drives that routing
        # (it is the executor's next_slot), but the executor may instead land at
        # standby when action[N]'s slot is infeasible. Pending must track the
        # real EE pose so the next step grasps/pushes from the right place.
        if route_only:
            # A route-only step routed the EE toward action[N]'s entry pose.
            if execution_success and next_slot is not None and next_slot.active:
                self._pending_target_slot = next_slot
                self._pending_action_primitive = next_primitive
            else:
                self._pending_target_slot = None
                self._pending_action_primitive = None
        else:
            # A manipulation's after-trajectory reports the slot/primitive it
            # routed toward (None when it fell back to standby). This is read
            # REGARDLESS of whether the manipulation itself succeeded: chain
            # continuation is independent of whether the push connected (jhsong)
            # -- the after-traj routes the EE to the next object's hover on both
            # the success and the rdd-miss paths, so the next pending must track
            # it either way. `_pending_from_execution_info` already returns
            # (None, None) when there is no valid next target (standby fallback),
            # so this is safe on failures too.
            pend_slot, pend_primitive = self._pending_from_execution_info(execution_info)
            self._pending_target_slot = pend_slot
            self._pending_action_primitive = pend_primitive

        # The routed-push marker (set by the executor when it routed the EE to a
        # cone hover for a next push) is only valid if the next manipulation IS
        # that push. Clear it otherwise so the cone plan (clean vertical strike)
        # is only used when the EE is actually pre-positioned for that slot.
        if not (
            self._pending_action_primitive == self.ACTION_PUSH
            and self._pending_target_slot is not None
            and self._push_routed_spawn_id is not None
            and self._pending_target_slot.spawn_id is not None
            and int(self._pending_target_slot.spawn_id) == self._push_routed_spawn_id
        ):
            self._push_routed_spawn_id = None
            self._push_routed_target_xyz = None

        self._step_count += 1
        _, next_detection, next_detections = self._capture_detection()
        if next_detection.detected:
            self._seen_detection = True

        self._last_detection = next_detection
        self._last_detections = next_detections
        next_tracking = self._tracking_from_detection(next_detection)
        self._last_tracking = next_tracking

        obs = self._observation_from_detections(next_detections)
        # Outcome-based reward: sum of resolved-object outcomes during this step
        # (+1 in-bin, -0.3 miss, -0.3 collateral knock-off, 0 natural ride-off).
        # Deferred by design -- a throw/push is credited when its object settles.
        reward = float(self._reward_accumulator)
        terminated = False
        truncated = self._step_count >= self.max_steps
        self._slot_feasibility_flags = self._compute_slot_feasibility_flags()
        info = self._build_info(next_detection, next_detections, next_tracking)
        info["execution_success"] = bool(execution_success)
        info["execution_info"] = execution_info
        info["route_only"] = bool(route_only)
        info["selected_action"] = action_index
        info["selected_primitive"] = next_primitive
        info["selected_primitive_name"] = self._action_primitive_name(next_primitive)
        info["pending_target_spawn_id"] = (
            None
            if self._pending_target_slot is None or self._pending_target_slot.spawn_id is None
            else int(self._pending_target_slot.spawn_id)
        )
        return obs, reward, terminated, truncated, info

    def _pending_from_execution_info(
        self, execution_info: dict[str, Any]
    ) -> tuple[ObjectSlot | None, int | None]:
        """Resolve the (slot, primitive) the just-executed manipulation's
        after-trajectory actually routed the EE toward, to queue as the next
        step's pending manipulation. Returns (None, None) when the after-traj
        fell back to standby (no chain)."""
        next_spawn_id = execution_info.get("next_spawn_id")
        next_primitive_name = execution_info.get("next_primitive_name")
        if next_spawn_id is None or next_primitive_name is None:
            return None, None
        slot = self._slot_from_spawn_id(next_spawn_id)
        if slot is None or not slot.active:
            return None, None
        if next_primitive_name == self._action_primitive_name(self.ACTION_PUSH):
            return slot, self.ACTION_PUSH
        return slot, self.ACTION_THROW

    def _execute_pending_manipulation(
        self,
        pending_slot: ObjectSlot,
        pending_primitive: int,
        next_slot: ObjectSlot | None,
        next_primitive: int,
    ) -> tuple[bool, dict[str, Any]]:
        """Execute the manipulation of `pending_slot` (set up by the previous
        action) with `pending_primitive`, ending the trajectory routed toward
        `next_slot`'s entry pose for `next_primitive`. The EE is assumed to be
        at `pending_slot`'s entry pose already (routed by the prior step)."""
        if int(pending_primitive) == self.ACTION_THROW:
            # The EE was routed to the grasp pose; make ONE open-loop grasp
            # attempt (no success check), then fire the throw REGARDLESS of
            # whether it caught the object -- matches the real robot, which has no
            # grasp-success sensor. A miss => empty throw, scored 0 by the outcome
            # reward. (Never abort here.)
            if not self._slot_is_attached(pending_slot):
                self._execute_ready_grasp(pending_slot)
            return self._execute_throw_transition(
                pending_slot, next_slot, next_primitive=next_primitive
            )
        return self._execute_push_transition_with_next(
            pending_slot, next_slot, next_primitive=next_primitive
        )

    def _route_to_action_entry(
        self, slot: ObjectSlot, primitive: int
    ) -> tuple[bool, dict[str, Any]]:
        """Route the EE toward `slot`'s entry pose for `primitive` without
        completing a manipulation. Used at route-only steps (step 0 / cleared
        pending). For throw the entry pose is the vertical grasp pose (and the
        object is grasped in passing); for push it is the hover pose above
        push_start."""
        if not slot.active:
            return False, {"reason": "inactive_slot"}
        if int(primitive) == self.ACTION_PUSH:
            return self._route_to_push_handoff(slot)
        return self._execute_initial_approach(slot)

    def _slot_in_action_window(self, slot: ObjectSlot | None) -> bool:
        if slot is None or not slot.active or getattr(slot, "manipulated", False) or getattr(slot, "blocked", False):
            return False
        if self._slot_is_attached(slot):
            return True
        obj_xyz = self._slot_grasp_throw_xyz(slot)
        return bool(float(obj_xyz[1]) > 0.05)

    def _slot_from_spawn_id(self, spawn_id: int | None) -> ObjectSlot | None:
        if spawn_id is None:
            return None
        for slot in self.object_slots:
            if slot.spawn_id == int(spawn_id):
                return slot
        return None

    def _block_slot_for_failure(self, spawn_id: int | None) -> None:
        slot = self._slot_from_spawn_id(spawn_id)
        if slot is None or self._slot_is_attached(slot):
            return
        slot.blocked = True

    def _sample_demo_action(self, push_probability: float = 0.5, alternate: bool = True) -> dict[str, int | None]:
        # Thin shim. Selection logic lives in `recycling_env/demo_policy.py`.
        # When RL training replaces this layer, callers stop using this method.
        return self._demo_selector.sample_action(push_probability=push_probability, alternate=alternate)

    def _recover_after_action_failure(self, execution_info: dict[str, Any]) -> None:
        reason = execution_info.get("reason")
        if reason not in {
            "suction_attach_failed",
            "robot_driven_push_displacement_too_small",
            "not_reachable",
            "object_past_push_window",
            "object_downstream_of_cone",
            "throw_intercept_too_slow",
        }:
            return

        self._block_slot_for_failure(execution_info.get("current_spawn_id"))

        if reason == "suction_attach_failed":
            self.gripper.set_enabled(False)
            self.gripper.release()
            self._pending_target_slot = None
            self._pending_action_primitive = None
            self._demo_next_primitive = self.ACTION_PUSH
            recovery_target = self.q_ready.copy()
            for joint_index, joint_name in enumerate(self.joint_names):
                actuator_id = self.env.require_actuator(f"{joint_name}_act")
                ctrl_min, ctrl_max = self.env.model.actuator_ctrlrange[actuator_id]
                recovery_target[joint_index] = np.clip(recovery_target[joint_index], ctrl_min, ctrl_max)
            self._execute_two_point_move(recovery_target, duration=0.35)

    def render(self) -> np.ndarray:
        mode = self.render_mode or "rgb_array"
        if mode == "rgb_array":
            return self.env.render_rgb()
        if mode == "bgr_array":
            return self.env.render_bgr(flip_code=self.flip_code)
        raise ValueError(f"Unsupported render_mode: {mode}")

    def close(self) -> None:
        self.env.close()

    def run_live_demo(
        self,
        num_steps: int | None = None,
        window_name: str = "visual tracking live",
        sleep: bool = True,
    ) -> None:
        if cv2 is None:
            raise ImportError("run_live_demo requires opencv-python to be installed.")
        if launch_passive is None:
            raise ImportError("run_live_demo requires mujoco viewer support to be installed.")

        max_steps = self.max_steps if num_steps is None else int(num_steps)
        obs, info = self.reset()
        print("reset obs:", obs)
        print("reset detected:", info["detected"])

        try:
            with launch_passive(self.env.model, self.env.data) as viewer:
                stop_requested = False

                def _live_step_hook() -> bool:
                    nonlocal stop_requested
                    viewer.sync()
                    try:
                        frame_bgr, detection, detections = self._capture_detection()
                    except Exception:
                        frame_bgr = self._last_frame_bgr
                        detection = self._last_detection
                        detections = self._last_detections
                    overlay_bgr = self.render_tracking_overlay(
                        frame_bgr=frame_bgr,
                        detection=detection,
                        detections=detections,
                        tracking=self._last_tracking,
                    )
                    cv2.imshow(window_name, overlay_bgr)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        stop_requested = True
                        return False
                    return True

                self.set_sim_step_hook(_live_step_hook)
                for _ in range(max_steps):
                    if stop_requested:
                        break
                    action = self._sample_demo_action(push_probability=0.5, alternate=True)
                    _, _, terminated, truncated, step_info = self.step(action)
                    execution_info = step_info.get("execution_info", {})
                    if execution_info.get("reason") not in {"skip", "bootstrap_invalid_action"}:
                        print(
                            "demo action:",
                            step_info.get("selected_action"),
                            step_info.get("selected_primitive_name"),
                            execution_info.get("reason"),
                        )
                    viewer.sync()

                    overlay_bgr = self.render_tracking_overlay(
                        frame_bgr=self._last_frame_bgr,
                        detection=self._last_detection,
                        detections=self._last_detections,
                        tracking=self._last_tracking,
                    )
                    cv2.imshow(window_name, overlay_bgr)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    if stop_requested:
                        break
                    if terminated or truncated:
                        break
                    if sleep:
                        time.sleep(self.env.dt * self.frame_skip)
        finally:
            self.set_sim_step_hook(None)
            if cv2 is not None:
                cv2.destroyAllWindows()

def debug_world_state(env: RecyclingBBoxGymEnv, num_steps: int = 50) -> None:
    obs, info = env.reset()

    print("reset obs:", obs)
    print("reset detected_box_world_boxes:", info.get("detected_box_world_boxes"))
    print("reset box_world_xy:", info.get("box_world_xy"))
    print("-" * 80)

    for step in range(int(num_steps)):
        action = env._sample_demo_action(push_probability=0.5, alternate=True)
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"step {step:03d}")
        print("obs shape:", obs.shape)
        print("obs:", obs)
        print("detected:", info["detected"])
        print("num_detections:", info["num_detections"])
        print("detected_box_world_boxes:", info.get("detected_box_world_boxes"))
        print("conveyor_speed:", info.get("conveyor_speed"))
        print("spawn_rate_hz:", info.get("spawn_rate_hz"))
        print("box_world_xy:", info.get("box_world_xy"))
        print("box_world_xyz:", info.get("box_world_xyz"))
        print("-" * 80)

        if info.get("detected_box_world_boxes"):
            first_box = np.asarray(info["detected_box_world_boxes"][0], dtype=np.float32)
            if not np.allclose(obs[:6], first_box[:6], atol=1e-5):
                print("WARNING: obs[:6] does not match first detected world box:", obs[:6], first_box[:6])
        if not np.isclose(obs[-2], info.get("conveyor_speed", np.nan), atol=1e-5):
            print("WARNING: obs[-2] does not match conveyor_speed:", obs[-2], info.get("conveyor_speed"))
        if not np.isclose(obs[-1], info.get("spawn_rate_hz", np.nan), atol=1e-5):
            print("WARNING: obs[-1] does not match spawn_rate_hz:", obs[-1], info.get("spawn_rate_hz"))

        if terminated or truncated:
            break


if __name__ == "__main__":
    # Single-command GUI verification of the throw-only chain.
    # Run from the repo root:
    #     python recycling_env/gym_env.py
    #     python recycling_env/gym_env.py --spawn 0.8 --speed 0.2 --realtime 0.5
    import argparse
    import time as _time
    import mujoco.viewer

    parser = argparse.ArgumentParser(
        description="GUI run of the recycling env. Default: both throws and pushes. "
                    "Use --throwing for throw-only, --pushing for push-only."
    )
    parser.add_argument("--spawn", type=float, default=1.2, help="spawn_rate_hz")
    parser.add_argument("--speed", type=float, default=0.4, help="belt_speed (m/s)")
    parser.add_argument("--realtime", type=float, default=1.0,
                        help="real-time multiplier (1.0 = real time, 0.5 = half speed)")
    parser.add_argument("--camera", action="store_true",
                        help="enable D435i camera view with bbox detection overlay (needs opencv-python)")
    parser.add_argument("--suction", type=float, default=1.0,
                        help="per-object throw-grasp success rate p in [0,1] (1.0 = always succeeds)")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--throwing", dest="mode", action="store_const", const="throw",
                            help="throw-only (no pushes)")
    mode_group.add_argument("--pushing", dest="mode", action="store_const", const="push",
                            help="push-only (no throws)")
    parser.set_defaults(mode="both")
    parser.add_argument("--debug-chain", action="store_true",
                        help="emit '[chain] ...' log lines for each chain transition/clear event")
    args = parser.parse_args()
    if args.mode == "throw":
        pushing_enabled = False
        push_prob = 0.0
        alternate = True
        mode_label = "throw-only"
    elif args.mode == "push":
        pushing_enabled = True
        push_prob = 1.0
        alternate = False
        mode_label = "push-only"
    else:  # both (default)
        pushing_enabled = True
        push_prob = 0.5
        alternate = False
        mode_label = "throw + push"

    cv2_module = None
    if args.camera:
        try:
            import cv2 as cv2_module
        except ImportError:
            print("warning: opencv-python is not installed; --camera disabled.")
            cv2_module = None

    env = RecyclingBBoxGymEnv(
        pushing_enabled=pushing_enabled,
        spawn_rate_hz=args.spawn,
        belt_speed=args.speed,
        use_camera=bool(args.camera and cv2_module is not None),
        step_sleep=0.0,
        debug_chain_log=bool(args.debug_chain),
        suction_success_rate_range=(args.suction, args.suction),
    )
    try:
        obs, info = env.reset()
        throws = throws_chained = throws_bootstrapped = route_grasps = attach_failed = 0
        # Under 1-step-ahead semantics the grasp is folded into the throw step;
        # a throw is "chained" when it manipulates the obj the previous throw
        # routed toward (no route-only re-approach in between).
        prev_throw_next_id = None
        route_since_last_throw = True
        sim_dt = env.env.dt * env.frame_skip
        target_dt = sim_dt / max(args.realtime, 1e-3)

        camera_window = "D435i camera (q to quit)"
        with mujoco.viewer.launch_passive(env.env.model, env.env.data) as viewer:
            env.set_sim_step_hook(make_throttled_viewer_hook(
                viewer, env,
                cv2_module=cv2_module,
                camera_window=camera_window,
            ))
            extra = "  (camera view enabled)" if (cv2_module is not None and env.use_camera) else ""
            print(f"Recycling env GUI  mode={mode_label}  spawn={args.spawn}/s  "
                  f"belt={args.speed} m/s  realtime x{args.realtime}.{extra}  "
                  f"Close the viewer window to stop.")
            try:
                while viewer.is_running():
                    t_start = _time.time()
                    action = env._sample_demo_action(push_probability=push_prob, alternate=alternate)
                    obs, _, terminated, truncated, info = env.step(action)
                    ei = info.get("execution_info", {})
                    reason = ei.get("reason", "?")
                    if reason == "executed":
                        throws += 1
                        current_id = ei.get("current_spawn_id")
                        chained = (not route_since_last_throw) and prev_throw_next_id is not None and current_id == prev_throw_next_id
                        if chained:
                            throws_chained += 1
                            tag = "throw [chain]    "
                        else:
                            throws_bootstrapped += 1
                            tag = "throw [from route]"
                        print(f"{tag}  cur={current_id} -> next={ei.get('next_spawn_id')}  "
                              f"T={ei.get('trajectory_duration', 0):.2f}s")
                        prev_throw_next_id = ei.get("next_spawn_id")
                        route_since_last_throw = False
                    elif reason == "initial_approach_executed":
                        route_grasps += 1
                        route_since_last_throw = True
                        print(f"route [bang-bang grasp]  obj={ei.get('current_spawn_id')}  "
                              f"[chained={throws_chained}  route={route_grasps}]")
                    elif reason == "push_handoff_routed":
                        route_since_last_throw = True
                        print(f"route [push handoff]     obj={ei.get('current_spawn_id')}")
                    elif reason == "ready_grasp_attach_failed":
                        attach_failed += 1
                        print(f"grasp [attach failed]  no motion - hold timed out")
                    elapsed = _time.time() - t_start
                    if elapsed < target_dt:
                        _time.sleep(target_dt - elapsed)
                    if truncated:
                        obs, info = env.reset()
            finally:
                env.set_sim_step_hook(None)
                if cv2_module is not None:
                    try:
                        cv2_module.destroyAllWindows()
                    except Exception:
                        pass

        print()
        print(f"throws={throws}  chained={throws_chained}  from_route={throws_bootstrapped}  "
              f"route_grasps={route_grasps}  attach_failed={attach_failed}")
        if throws > 0:
            print(f"chain rate: {throws_chained / throws * 100:.1f}%  "
                  f"(throws that chained without a route-only re-approach)")
    finally:
        env.close()

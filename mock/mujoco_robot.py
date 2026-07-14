"""MuJoCo-backed mock robot — the Level B physics/visual digital twin.

A drop-in for ``mock_robot``: it exposes the *identical* bridge-facing ROS
contract (Point Queue Mode, FollowJointTrajectory, ``/joint_states_urdf``,
``/write_single_io``, ``/robot_enable``, ``reset_error``, ...) by subclassing
``MockRobot`` — so the app drives it with exactly the same topics/services the
real MotoROS2 robot receives — but instead of echoing the commanded joints
blind, it feeds those same commands into the vendored MuJoCo model
(``sim/recycling_mujoco/scene.xml``) and renders the GP8 moving live.

Two modes (``--physics`` to switch):

  kinematic (default)  ``qpos := commanded joints`` then ``mj_forward`` — a
      perfectly stable mirror of the commanded trajectory. Belt boxes mirror
      ``/camera_debug/detections`` so what you see == what the app perceives,
      but there is no contact/grasp: it verifies motion path / interception
      timing, not grasp success. ``/joint_states_urdf`` == commanded joints.

  physics  (``--physics``)  the **coherent digital twin** + a *camera bridge*:
      the twin OWNS the belt — it spawns physics boxes, rides them down a real
      conveyor surface, and PUBLISHES ``/camera_debug/detections`` +
      ``/conveyor/speed`` FROM the MuJoCo box positions (so it replaces
      ``fake_belt``; the launch must not also start it). The app perceives the
      real boxes, reaches for them, and ``/write_single_io`` ON welds the
      nearest box to the gripper (it rides the swing and flies on release);
      the pusher geom shoves boxes for the push skill. ``/joint_states_urdf``
      reports the actual tracked qpos. So the objects truly react.

The reachable conveyor is added to the model at load time with ``MjSpec`` (the
vendored belt sits in the env's own frame, out of the GP8's reach) — glue that
does NOT edit the vendored tree (VENDOR.md). The model is otherwise driven
through the raw ``mujoco`` API; ``recycling_env`` is never imported (its
top-level ``utils``/``models`` would shadow ``gp8_control``'s).

Run (needs ``mujoco>=3.1`` in the venv; a DISPLAY for the window else
``--headless``):

    PYTHONPATH=$HOME/ros2_ws/src \\
      ~/ros2_ws/src/gp8_control/.venv/bin/python -m gp8_control.mock.mujoco_robot --physics

or, with the full pipeline, ``ros2 launch gp8_control sim_mujoco.launch.py physics:=true``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64, String

from gp8_control.mock.mock_robot import JOINT_NAMES, MockRobot

try:
    import mujoco
except ImportError as _exc:   # not in this interpreter (e.g. plain `ros2 run`)
    mujoco = None
    _MUJOCO_IMPORT_ERROR: ImportError | None = _exc
    mj_viewer = None
else:
    _MUJOCO_IMPORT_ERROR = None
    try:  # the GUI viewer (glfw); headless boxes fall back to --headless
        import mujoco.viewer as mj_viewer
    except Exception:  # pragma: no cover - depends on GL availability
        mj_viewer = None


# scene.xml is two dirs up: <pkg>/mock/mujoco_robot.py -> <pkg>/sim/recycling_mujoco/
_SCENE = Path(__file__).resolve().parents[1] / "sim" / "recycling_mujoco" / "scene.xml"

# gp8 joint order [S, L, U, R, B, T] == these MuJoCo joints / position actuators
# (confirmed numerically by sim/preview_gp8_check.py).
MJ_JOINTS = ["S_axis", "L_axis", "U_axis", "R_axis", "B_axis", "T_axis"]
MJ_ACTS = ["S_axis_act", "L_axis_act", "U_axis_act", "R_axis_act", "B_axis_act", "T_axis_act"]
MJ_BASE_BODY = "yaskawa_robot"   # gp8 base frame == this body; detections are base-frame
MJ_GRIP_SITE = "grip_site"

# Free-joint boxes already in the vendored model, reused as the object pool.
BOX_JOINTS = ["box_free"] + [f"box_free_{i}" for i in range(2, 13)]
BOX_GEOMS = ["red_box_geom"] + [f"red_box_geom_{i}" for i in range(2, 13)]
BOX_BODIES = ["red_box"] + [f"red_box_{i}" for i in range(2, 13)]
BOX_HALF_Z = 0.015          # red_box_geom half-height (size z in combined_test.xml)
_PARK = np.array([0.0, 0.0, -2.0])   # stash unused boxes below the floor

# Per-class box tint so throw (transparent) vs push (metal) reads at a glance.
_CLASS_RGBA = {
    "transparent": (0.30, 0.65, 1.00, 0.55),
    "metal": (0.62, 0.62, 0.68, 1.00),
}
_DEFAULT_RGBA = (0.85, 0.20, 0.20, 1.00)

# The reachable conveyor we add for the physics twin (the vendored belt is out
# of the GP8's reach). Geometry in the gp8 BASE frame; placed into the world via
# the yaskawa_robot body pose at build time. lane_x/grasp_z mirror fake_belt.
BELT_BODY = "gp8_belt"
BELT_GEOM = "gp8_belt_surface"


def _build_twin_model(scene_path: str, base_pos, lane_x: float, grasp_z: float,
                      half_len: float):
    """Load the vendored scene and add a reachable conveyor via MjSpec.

    The surface top sits one box-half below grasp_z so a box rests with its
    centre at grasp_z (== where the app's grasp pose and our detections put it).
    Returns a compiled MjModel. Glue: never edits the vendored XML on disk.
    """
    spec = mujoco.MjSpec.from_file(scene_path)
    bx, by, bz = float(base_pos[0]), float(base_pos[1]), float(base_pos[2])
    top = bz + grasp_z - BOX_HALF_Z            # belt surface top (world z)
    thick = 0.02
    belt = spec.worldbody.add_body(
        name=BELT_BODY, pos=[bx + lane_x, by, top - thick],
    )
    belt.add_geom(
        name=BELT_GEOM, type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.16, half_len, thick], pos=[0.0, 0.0, 0.0],
        rgba=[0.12, 0.12, 0.14, 1.0], friction=[0.3, 0.02, 0.002],
    )
    return spec.compile()


class MujocoRobot(MockRobot):
    """``mock_robot`` whose joint state is produced by a live MuJoCo model."""

    # box lifecycle states
    _FREE, _ON_BELT, _GRABBED, _LOOSE = "free", "on_belt", "grabbed", "loose"

    def __init__(
        self,
        *,
        physics: bool = False,
        headless: bool = False,
        show_objects: bool = True,
        render_hz: float = 60.0,
    ) -> None:
        super().__init__()   # all the services / queue / FJT / 50 Hz js timer
        self._physics = physics
        self._headless = headless
        self._show_objects = show_objects
        self._render_dt = 1.0 / max(1.0, render_hz)
        self._running = True

        if not _SCENE.is_file():
            raise FileNotFoundError(
                f"MuJoCo scene not found: {_SCENE}\n"
                "Did `git lfs pull` hydrate sim/recycling_mujoco/ meshes?"
            )

        # --- belt / spawn params (match fake_belt so the app behaves the same)
        self._belt_speed = self._declare("belt_speed", 0.12)
        self._spawn_interval = self._declare("spawn_interval", 5.0)
        self._spawn_y = self._declare("spawn_y", 0.9)
        self._despawn_y = self._declare("despawn_y", -0.8)
        self._lane_x = self._declare("lane_x", 0.45)
        self._grasp_z = self._declare("grasp_z", 0.062)
        self._aim_dz = self._declare("aim_dz", 0.08)
        cls = self._declare("classes", ["transparent", "metal"])
        self._classes = list(cls) or ["transparent"]

        # --- load the model (vendored scene; + a reachable belt in physics mode)
        if physics:
            # base pose is fixed in the XML (yaskawa_robot @ (-0.05,0,0.6),
            # identity rot); read it from a throwaway load to place the belt.
            base = mujoco.MjModel.from_xml_path(str(_SCENE)).body(MJ_BASE_BODY).pos
            half_len = 0.5 * (self._spawn_y - self._despawn_y) + 0.1
            self._model = _build_twin_model(
                str(_SCENE), base, self._lane_x, self._grasp_z, half_len)
        else:
            self._model = mujoco.MjModel.from_xml_path(str(_SCENE))
        self._data = mujoco.MjData(self._model)
        self._mj_lock = threading.Lock()
        self._dt = float(self._model.opt.timestep)

        self._act_ids = [int(self._model.actuator(n).id) for n in MJ_ACTS]
        if physics:
            # Stiffen the position servos so the arm tracks the commanded
            # trajectory closely — otherwise it arrives at the grasp pose late
            # and the timing-sensitive ambush grasp misses. (position actuator:
            # gainprm[0]=kp, biasprm=[0,-kp,-kv].)
            kp, kv = 12000.0, 220.0
            for aid in self._act_ids:
                self._model.actuator_gainprm[aid][0] = kp
                self._model.actuator_biasprm[aid][1] = -kp
                self._model.actuator_biasprm[aid][2] = -kv
        self._grip_id = int(self._model.site(MJ_GRIP_SITE).id)
        self._base_p = self._data.body(MJ_BASE_BODY).xpos.copy()
        self._base_R = np.eye(3)   # yaskawa_robot has identity orientation

        # Seat the model at the mock's home posture (B at -pi/2) and hold it.
        home = list(self._joint_positions)
        for name, val in zip(MJ_JOINTS, home):
            self._data.joint(name).qpos[0] = float(val)
        for aid, val in zip(self._act_ids, home):
            self._data.ctrl[aid] = float(val)
        mujoco.mj_forward(self._model, self._data)
        self._base_p = self._data.body(MJ_BASE_BODY).xpos.copy()

        # --- joint-state snapshot the ROS publish thread reads (so the executor
        #     never touches MjData concurrently with the sim loop).
        self._measured_lock = threading.Lock()
        self._measured_pos = list(home)
        self._measured_vel = [0.0] * 6

        # --- object pool
        self._box_joints = [self._opt_joint(n) for n in BOX_JOINTS]
        self._box_geom_ids = [self._opt_geom(n) for n in BOX_GEOMS]
        self._box_state = [self._FREE] * len(BOX_JOINTS)
        self._box_class = [""] * len(BOX_JOINTS)
        self._park_all_boxes()

        # --- mode-specific wiring
        self._det_lock = threading.Lock()
        self._latest_dets: list[tuple[float, float, float, str]] = []
        if physics:
            # the twin IS the perception + conveyor source (replaces fake_belt)
            self._det_pub = self.create_publisher(String, "/camera_debug/detections", 10)
            self._belt_pub = self.create_publisher(Float64, "/conveyor/speed", 10)
            self._spawn_count = 0
            self._last_spawn = 0.0
            self._last_pub = 0.0
            self._grabbed: int | None = None
            # Release velocity is taken from the COMMANDED trajectory (gp8 FK
            # Jacobian x commanded joint velocity), not the actual arm: the
            # position actuators lag a fast throw, so the physical EE is nearly
            # still at release — the box would just drop. The commanded EE
            # velocity is the intended throw speed, and is render-rate-independent.
            from gp8_control.robots.gp8 import GP8
            self._gp8 = GP8()
            self._cmd_joint_vel = np.zeros(6)   # filled by _drive_queue override
            self._prev_suction = False
        elif show_objects:
            # kinematic: mirror whatever the app perceives onto belt boxes
            self.create_subscription(
                String, "/camera_debug/detections", self._on_detections, 10)

        self._img_pub = None
        if headless:
            self._img_pub = self.create_publisher(Image, "/mujoco/image", 5)

        self.get_logger().info(
            "MuJoCo robot ready "
            f"(mode={'physics twin + camera bridge' if physics else 'kinematic mirror'}, "
            f"render={'headless /mujoco/image' if headless else 'viewer window'}, "
            f"objects={'on' if show_objects else 'off'}). "
            "Same ROS contract as mock_robot."
            + ("  Publishing /camera_debug/detections — do NOT also run fake_belt."
               if physics else "")
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _declare(self, name, default):
        self.declare_parameter(name, default)
        return self.get_parameter(name).value

    def _opt_joint(self, name: str):
        try:
            return self._data.joint(name)
        except Exception:
            return None

    def _opt_geom(self, name: str) -> int:
        try:
            return int(self._model.geom(name).id)
        except Exception:
            return -1

    def _grip_world(self) -> np.ndarray:
        return self._data.site(self._grip_id).xpos.copy()

    def _to_base(self, world_xyz) -> np.ndarray:
        return self._base_R.T @ (np.asarray(world_xyz) - self._base_p)

    def _to_world(self, base_xyz) -> np.ndarray:
        return self._base_p + self._base_R @ np.asarray(base_xyz)

    # ------------------------------------------------------------------
    # Joint-state publisher — report what MuJoCo actually shows
    # ------------------------------------------------------------------

    def _publish_joint_states(self) -> None:
        if not hasattr(self, "_measured_lock"):   # super().__init__ still running
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        with self._measured_lock:
            msg.position = list(self._measured_pos)
            msg.velocity = list(self._measured_vel)
        msg.effort = [0.0] * 6
        self._js_pub.publish(msg)
        self._js_pub_raw.publish(msg)   # adv4ncr stream stack reads /joint_states

    def _drive_queue(self) -> None:
        """Same real-time queue playback as MockRobot, but also exposes the
        commanded joint VELOCITY (the active segment's slope) — the throw needs
        it for an accurate release fling. (Base discards it as zeros.)"""
        now = time.time()
        vel = [0.0] * 6
        with self._q_lock:
            q = self._q
            if not q:
                return
            if now <= q[0][0]:
                pos = list(q[0][1])
            elif now >= q[-1][0]:
                pos = list(q[-1][1])
                # Hold the terminal segment's velocity briefly after the queue
                # drains, so a throw release (suction-off) that lands a tick late
                # still flings at the trajectory's end speed instead of zero.
                if len(q) >= 2 and (now - q[-1][0]) < 0.15:
                    t0, p0, _ = q[-2]
                    t1, p1, _ = q[-1]
                    seg = max(1e-6, t1 - t0)
                    vel = [(p1[j] - p0[j]) / seg for j in range(len(p0))]
            else:
                pos = list(q[-1][1])
                for i in range(len(q) - 1):
                    t0, p0, _ = q[i]
                    t1, p1, _ = q[i + 1]
                    if t0 <= now <= t1:
                        dt = max(1e-6, t1 - t0)
                        a = (now - t0) / dt
                        pos = [p0[j] + a * (p1[j] - p0[j]) for j in range(len(p0))]
                        vel = [(p1[j] - p0[j]) / dt for j in range(len(p0))]
                        break
        with self._lock:
            self._joint_positions = list(pos)
            self._joint_velocities = list(vel)
        self._cmd_joint_vel = np.asarray(vel)

    # ==================================================================
    # KINEMATIC mode — mirror perceived detections onto belt boxes
    # ==================================================================

    def _on_detections(self, msg: String) -> None:
        try:
            snap = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        out: list[tuple[float, float, float, str]] = []
        for det in snap.get("detections", []):
            g = det.get("base_grasp")
            if not g or len(g) < 3:
                continue
            out.append((float(g[0]), float(g[1]), float(g[2]), str(det.get("class", ""))))
        with self._det_lock:
            self._latest_dets = out

    def _park_all_boxes(self) -> None:
        for jnt in self._box_joints:
            if jnt is None:
                continue
            jnt.qpos[0:3] = _PARK
            jnt.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
            jnt.qvel[:] = 0.0

    def _set_box_color(self, i: int, cls: str) -> None:
        gid = self._box_geom_ids[i]
        if gid >= 0:
            self._model.geom_rgba[gid] = _CLASS_RGBA.get(cls, _DEFAULT_RGBA)

    def _mirror_boxes(self) -> None:
        with self._det_lock:
            dets = list(self._latest_dets)
        n = min(len(dets), len(self._box_joints))
        for i, jnt in enumerate(self._box_joints):
            if jnt is None:
                continue
            if i < n:
                x, y, z, cls = dets[i]
                jnt.qpos[0:3] = self._to_world([x, y, z])
                jnt.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
                jnt.qvel[:] = 0.0
                self._set_box_color(i, cls)
            else:
                jnt.qpos[0:3] = _PARK
                jnt.qvel[:] = 0.0

    # ==================================================================
    # PHYSICS mode — twin owns the belt + objects; camera bridge perception
    # ==================================================================

    def _free_box(self) -> int | None:
        for i, s in enumerate(self._box_state):
            if s == self._FREE and self._box_joints[i] is not None:
                return i
        return None

    def _spawn_box(self, now: float) -> None:
        i = self._free_box()
        if i is None:
            return
        cls = self._classes[self._spawn_count % len(self._classes)]
        jitter = 0.06 * ((self._spawn_count % 3) - 1)   # -0.06, 0, +0.06
        world = self._to_world([self._lane_x + jitter, self._spawn_y, self._grasp_z])
        jnt = self._box_joints[i]
        jnt.qpos[0:3] = world
        jnt.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        jnt.qvel[:] = 0.0
        self._box_state[i] = self._ON_BELT
        self._box_class[i] = cls
        self._set_box_color(i, cls)
        self.get_logger().info(
            f"spawn #{self._spawn_count} ({cls}) on belt at base y={self._spawn_y:+.2f}")
        self._spawn_count += 1

    def _apply_belt_velocity(self) -> None:
        """Force on-belt boxes' world-Y velocity to the belt speed. Called every
        substep so a static surface's friction can't brake them (X/Z stay free,
        so the pusher and gravity still act)."""
        floor = self._base_p[2] + self._grasp_z - 0.03
        for i, jnt in enumerate(self._box_joints):
            if jnt is None or self._box_state[i] != self._ON_BELT:
                continue
            if jnt.qpos[2] >= floor:
                jnt.qvel[1] = -self._belt_speed   # conveyor motion (world -Y)

    def _recycle_boxes(self) -> None:
        """Retire boxes off the belt end / on the floor; flag ones knocked off."""
        off_belt = self._base_p[2] + self._grasp_z - 0.05
        for i, jnt in enumerate(self._box_joints):
            if jnt is None or self._box_state[i] in (self._FREE, self._GRABBED):
                continue
            world = jnt.qpos[0:3]
            base = self._to_base(world)
            fallen = self._base_p[2] - 0.4   # ~floor level, base-relative
            if base[1] < self._despawn_y or world[2] < fallen or abs(base[0]) > 1.6:
                jnt.qpos[0:3] = _PARK
                jnt.qvel[:] = 0.0
                self._box_state[i] = self._FREE
                self._box_class[i] = ""
            elif self._box_state[i] == self._ON_BELT and world[2] < off_belt:
                self._box_state[i] = self._LOOSE   # pushed / knocked off the belt
                self.get_logger().info(
                    f"box {i} ({self._box_class[i]}) knocked off the belt (push)")

    def _commanded_ee_velocity(self, cmd) -> np.ndarray:
        """World EE velocity from the commanded joint trajectory: position
        Jacobian (finite-diff gp8 FK) x the commanded joint velocity (the queue
        segment slope). Independent of render rate, so a throw flings correctly."""
        q = np.asarray(cmd, dtype=float)
        ee0 = self._gp8.forward_kinematics(q)[:3, 3]
        jp = np.empty((3, 6))
        eps = 1e-5
        for k in range(6):
            dq = q.copy()
            dq[k] += eps
            jp[:, k] = (self._gp8.forward_kinematics(dq)[:3, 3] - ee0) / eps
        return self._base_R @ (jp @ self._cmd_joint_vel)   # base R == identity

    def _update_grasp(self, elapsed: float, cmd) -> None:
        """Weld the nearest box on suction ON; fling it on suction OFF."""
        suction = bool(self._suction_on)

        if suction and self._grabbed is None:
            # Grab the box nearest the COMMANDED gripper (== gp8 FK EE, validated
            # to match grip_site): that's where the app aimed and where the box
            # is, so the grasp is robust to the actual arm lagging behind.
            cmd_grip = self._to_world(self._gp8.forward_kinematics(cmd)[:3, 3])
            best, bestd = None, 0.10   # grab radius (m)
            for i, jnt in enumerate(self._box_joints):
                if jnt is None or self._box_state[i] != self._ON_BELT:
                    continue
                d = float(np.linalg.norm(jnt.qpos[0:3] - cmd_grip))
                if d < bestd:
                    best, bestd = i, d
            if best is not None:
                self._grabbed = best
                self._box_state[best] = self._GRABBED
                self.get_logger().info(
                    f"suction grasp: box {best} ({self._box_class[best]})")
        elif not suction and self._grabbed is not None:
            i = self._grabbed
            jnt = self._box_joints[i]
            fling = self._commanded_ee_velocity(cmd)
            jnt.qvel[0:3] = fling                # leave the hand at throw speed
            jnt.qvel[3:6] = 0.0
            self._box_state[i] = self._LOOSE     # now free physics -> flies
            self._grabbed = None
            self.get_logger().info(
                f"release: box {i} flung at {np.linalg.norm(fling):.2f} m/s")
        self._prev_suction = suction

    def _pin_grabbed(self) -> None:
        if self._grabbed is None:
            return
        jnt = self._box_joints[self._grabbed]
        jnt.qpos[0:3] = self._grip_world()       # held at the suction cup
        jnt.qvel[:] = 0.0

    def _publish_belt_and_dets(self, now: float) -> None:
        self._belt_pub.publish(Float64(data=float(self._belt_speed)))
        dets = []
        for i, jnt in enumerate(self._box_joints):
            if jnt is None or self._box_state[i] != self._ON_BELT:
                continue
            b = self._to_base(jnt.qpos[0:3])
            x, y = float(b[0]), float(b[1])
            dets.append({
                "class": self._box_class[i],
                "base_grasp": [x, y, self._grasp_z],
                "base_aim": [x, y, self._grasp_z + self._aim_dz],
                "cam": [0.0, 0.0, 0.0],
                "in_workspace": True,
            })
        snap = {"receipt_time": now, "belt_mps": self._belt_speed, "detections": dets}
        self._det_pub.publish(String(data=json.dumps(snap)))

    def _physics_update(self, elapsed: float, cmd) -> None:
        now = time.time()
        # spawn on schedule
        if now - self._last_spawn >= self._spawn_interval:
            self._spawn_box(now)
            self._last_spawn = now
        # command the arm
        for aid, val in zip(self._act_ids, cmd):
            self._data.ctrl[aid] = float(val)
        # decide grab/release (fling speed from the commanded EE), then integrate
        self._update_grasp(elapsed, cmd)
        nsteps = int(np.clip(round(elapsed / self._dt), 1, 40))
        for _ in range(nsteps):
            self._apply_belt_velocity()          # re-assert belt motion each step
            mujoco.mj_step(self._model, self._data)
            self._pin_grabbed()                  # keep the held box on the cup
        self._recycle_boxes()                    # retire boxes off the belt
        # camera bridge: publish perception at ~15 Hz
        if now - self._last_pub >= 1.0 / 15.0:
            self._publish_belt_and_dets(now)
            self._last_pub = now
        pos = [float(self._data.joint(n).qpos[0]) for n in MJ_JOINTS]
        vel = [float(self._data.joint(n).qvel[0]) for n in MJ_JOINTS]
        with self._measured_lock:
            self._measured_pos = pos
            self._measured_vel = vel

    # ------------------------------------------------------------------
    # one tick: dispatch to the active mode
    # ------------------------------------------------------------------

    def _update_sim(self, elapsed: float) -> None:
        with self._lock:
            cmd = list(self._joint_positions)
        with self._mj_lock:
            if self._physics:
                self._physics_update(elapsed, cmd)
            else:
                if self._show_objects:
                    self._mirror_boxes()
                for name, val in zip(MJ_JOINTS, cmd):
                    self._data.joint(name).qpos[0] = float(val)
                    self._data.joint(name).qvel[0] = 0.0
                mujoco.mj_forward(self._model, self._data)
                with self._measured_lock:
                    self._measured_pos = list(cmd)
                    self._measured_vel = [0.0] * 6

    # ------------------------------------------------------------------
    # Render loops (run in the MAIN thread; ROS spins in a side thread)
    # ------------------------------------------------------------------

    def run_render(self) -> None:
        if not self._headless and mj_viewer is not None:
            try:
                self._run_viewer()
                return
            except Exception as exc:   # no DISPLAY over SSH, GL init failure, ...
                self.get_logger().warning(
                    f"GUI viewer unavailable ({exc}); falling back to headless "
                    "/mujoco/image (view in rqt_image_view / RViz)."
                )
        if self._img_pub is None:
            self._img_pub = self.create_publisher(Image, "/mujoco/image", 5)
        os.environ.setdefault("MUJOCO_GL", "egl")
        self._run_headless()

    def _run_viewer(self) -> None:
        with mj_viewer.launch_passive(self._model, self._data) as viewer:
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -20.0
            viewer.cam.distance = 2.2
            viewer.cam.lookat[:] = (0.25, 0.0, 0.75)
            last = time.time()
            while self._running and viewer.is_running() and rclpy.ok():
                t0 = time.time()
                # the passive viewer renders MjData/MjModel from its own daemon
                # thread; hold its lock while we mutate qpos/ctrl/geom_rgba.
                with viewer.lock():
                    self._update_sim(t0 - last)
                last = t0
                viewer.sync()
                time.sleep(max(0.0, self._render_dt - (time.time() - t0)))
        self._running = False

    def _run_headless(self) -> None:
        try:
            renderer = mujoco.Renderer(self._model, height=480, width=640)
        except Exception as exc:  # pragma: no cover - GL backend dependent
            self.get_logger().error(
                f"Could not create an offscreen renderer ({exc}). "
                "Set MUJOCO_GL=egl (headless GPU) or osmesa (CPU) and retry, "
                "or run with a DISPLAY for the viewer window. The ROS contract "
                "still works without rendering."
            )
            renderer = None
        last = time.time()
        period = max(self._render_dt, 1.0 / 15.0)   # cap headless to <=15 Hz
        try:
            while self._running and rclpy.ok():
                t0 = time.time()
                self._update_sim(t0 - last)
                last = t0
                if renderer is not None and self._img_pub is not None:
                    with self._mj_lock:
                        renderer.update_scene(self._data, camera="d435i_view")
                        frame = renderer.render()
                    self._publish_image(frame)
                time.sleep(max(0.0, period - (time.time() - t0)))
        finally:
            if renderer is not None:
                # free the GL context now, while EGL is still alive. Under an
                # abrupt shutdown the EGL display may already be gone, so the
                # destroy can still raise EGL_BAD_CONTEXT — purely teardown
                # noise (the process is exiting); swallow it.
                try:
                    renderer.close()
                except Exception:
                    pass

    def _publish_image(self, frame: np.ndarray) -> None:
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "d435i_view"
        msg.height, msg.width = int(frame.shape[0]), int(frame.shape[1])
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = 3 * msg.width
        msg.data = frame.tobytes()
        self._img_pub.publish(msg)


def _parse_flags(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mujoco_robot",
        description="MuJoCo-backed mock robot (Level B digital twin).",
    )
    parser.add_argument("--physics", action="store_true",
                        help="coherent physics twin + camera bridge "
                             "(default: stable kinematic mirror)")
    parser.add_argument("--headless", action="store_true",
                        help="no window; render to /mujoco/image (SSH-friendly)")
    parser.add_argument("--no-objects", action="store_true",
                        help="kinematic: do not mirror detections onto belt boxes")
    known, _ = parser.parse_known_args(argv)   # ignore ROS args (--ros-args ...)
    return known


def main(args=None) -> None:
    flags = _parse_flags((args if args is not None else sys.argv)[1:])
    if mujoco is None:
        sys.stderr.write(
            "mujoco_robot needs the 'mujoco' package, not importable under this "
            f"Python ({sys.executable}).\nIt lives in the gp8_control uv venv — run "
            "it with the venv python:\n"
            "  PYTHONPATH=$HOME/ros2_ws/src \\\n"
            "    ~/ros2_ws/src/gp8_control/.venv/bin/python -m gp8_control.mock.mujoco_robot\n"
            "or:  ros2 launch gp8_control sim_mujoco.launch.py\n"
            "(install it once with:  uv pip install --python "
            "~/ros2_ws/src/gp8_control/.venv/bin/python 'mujoco>=3.1')\n"
            f"import error: {_MUJOCO_IMPORT_ERROR}\n"
        )
        raise SystemExit(2)
    if flags.headless:
        # Offscreen rendering needs an explicit GL backend; egl = headless GPU.
        # Override at the shell (MUJOCO_GL=osmesa) for a CPU-only box.
        os.environ.setdefault("MUJOCO_GL", "egl")
    rclpy.init(args=args)
    node = executor = spin_thread = None
    try:
        node = MujocoRobot(
            physics=flags.physics,
            headless=flags.headless,
            show_objects=not flags.no_objects,
        )
        executor = MultiThreadedExecutor()
        executor.add_node(node)

        def _spin() -> None:
            try:
                executor.spin()
            except (ExternalShutdownException, KeyboardInterrupt):
                pass
            except Exception:
                # context torn down on shutdown raises a bare RCLError in the
                # wait set; only surface it if we are NOT already shutting down.
                if rclpy.ok():
                    raise

        spin_thread = threading.Thread(target=_spin, daemon=True)
        spin_thread.start()
        node.run_render()           # blocks in the main thread
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        # A SIGTERM during the (~4 s) model load tears the context down while
        # we're mid-construct; that surfaces as an RCLError. Swallow it only if
        # we're already shutting down — otherwise it's a real fault, re-raise.
        if rclpy.ok():
            raise
    finally:
        try:   # a Ctrl-C (SIGINT) can land mid-cleanup — don't let it traceback
            if node is not None:
                node._running = False
            if executor is not None:
                executor.shutdown()
            if spin_thread is not None:
                spin_thread.join(timeout=2.0)   # let in-flight callbacks finish
            if node is not None:
                node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()

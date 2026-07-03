from __future__ import annotations

import os
from pathlib import Path

# os.environ.setdefault("MUJOCO_GL", "egl")
import platform
if platform.system() == "Linux":
    os.environ.setdefault("MUJOCO_GL", "egl")
else:
    os.environ.setdefault("MUJOCO_GL", "glfw")

import mujoco
import numpy as np


class RecyclingSimEnv:
    """Thin MuJoCo environment wrapper for the recycling demo scene."""

    def __init__(
        self,
        model_path: str = "scene.xml",
        cam_w: int = 320,
        cam_h: int = 240,
        camera_name: str = "d435i_view",
        gl_backend: str | None = None,
    ) -> None:
        if gl_backend:
            current_backend = os.environ.get("MUJOCO_GL")
            if current_backend != gl_backend:
                raise ValueError(
                    f"MUJOCO_GL is already set to '{current_backend}'. "
                    f"Requested backend '{gl_backend}' must be configured before importing recycling_env."
                )

        self.project_root = Path(__file__).resolve().parent.parent
        self.model_path = self._resolve_model_path(model_path)
        self.camera_name = camera_name
        self.cam_w = cam_w
        self.cam_h = cam_h

        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, self.cam_h, self.cam_w)

        self.actuator_ids = self._collect_names(mujoco.mjtObj.mjOBJ_ACTUATOR)
        self.joint_ids = self._collect_names(mujoco.mjtObj.mjOBJ_JOINT)
        self.body_ids = self._collect_names(mujoco.mjtObj.mjOBJ_BODY)
        self.geom_ids = self._collect_names(mujoco.mjtObj.mjOBJ_GEOM)
        self.site_ids = self._collect_names(mujoco.mjtObj.mjOBJ_SITE)
        self.sensor_ids = self._collect_names(mujoco.mjtObj.mjOBJ_SENSOR)
        self.camera_ids = self._collect_names(mujoco.mjtObj.mjOBJ_CAMERA)
        self.equality_ids = self._collect_names(mujoco.mjtObj.mjOBJ_EQUALITY)

    def _resolve_model_path(self, model_path: str) -> Path:
        path = Path(model_path)
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()

    def _collect_names(self, obj_type: mujoco.mjtObj) -> dict[str, int]:
        count_map = {
            mujoco.mjtObj.mjOBJ_ACTUATOR: self.model.nu,
            mujoco.mjtObj.mjOBJ_JOINT: self.model.njnt,
            mujoco.mjtObj.mjOBJ_BODY: self.model.nbody,
            mujoco.mjtObj.mjOBJ_GEOM: self.model.ngeom,
            mujoco.mjtObj.mjOBJ_SITE: self.model.nsite,
            mujoco.mjtObj.mjOBJ_SENSOR: self.model.nsensor,
            mujoco.mjtObj.mjOBJ_CAMERA: self.model.ncam,
            mujoco.mjtObj.mjOBJ_EQUALITY: self.model.neq,
        }
        count = count_map[obj_type]
        names: dict[str, int] = {}
        for idx in range(count):
            name = mujoco.mj_id2name(self.model, obj_type, idx)
            if name:
                names[name] = idx
        return names

    @property
    def dt(self) -> float:
        return float(self.model.opt.timestep)

    @property
    def sim_time(self) -> float:
        return float(self.data.time)

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def close(self) -> None:
        renderer = getattr(self, "renderer", None)
        if renderer is None:
            return
        try:
            renderer.close()
        except Exception:
            pass
        finally:
            self.renderer = None

    def step(self, nstep: int = 1) -> None:
        for _ in range(nstep):
            mujoco.mj_step(self.model, self.data)

    def render_rgb(self, camera_name: str | None = None) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera_name or self.camera_name)
        return self.renderer.render().copy()

    def render_bgr(self, camera_name: str | None = None, flip_code: int | None = -1) -> np.ndarray:
        rgb = self.render_rgb(camera_name=camera_name)
        bgr = rgb[..., ::-1].copy()
        if flip_code is not None:
            if flip_code == 0:
                bgr = np.flipud(bgr).copy()
            elif flip_code == 1:
                bgr = np.fliplr(bgr).copy()
            elif flip_code == -1:
                bgr = np.flipud(np.fliplr(bgr)).copy()
            else:
                raise ValueError(f"Unsupported flip_code: {flip_code}")
        return bgr

    def set_actuator_ctrl(self, actuator_name: str, value: float) -> None:
        self.data.ctrl[self.require_actuator(actuator_name)] = value

    def set_actuator_ctrls(self, actuator_values: dict[str, float]) -> None:
        for name, value in actuator_values.items():
            self.set_actuator_ctrl(name, value)

    def require_actuator(self, actuator_name: str) -> int:
        return self._require_name(self.actuator_ids, actuator_name, "actuator")

    def require_joint(self, joint_name: str) -> int:
        return self._require_name(self.joint_ids, joint_name, "joint")

    def require_body(self, body_name: str) -> int:
        return self._require_name(self.body_ids, body_name, "body")

    def require_geom(self, geom_name: str) -> int:
        return self._require_name(self.geom_ids, geom_name, "geom")

    def require_site(self, site_name: str) -> int:
        return self._require_name(self.site_ids, site_name, "site")

    def require_sensor(self, sensor_name: str) -> int:
        return self._require_name(self.sensor_ids, sensor_name, "sensor")

    def require_equality(self, equality_name: str) -> int:
        return self._require_name(self.equality_ids, equality_name, "equality")

    def _require_name(self, name_to_id: dict[str, int], name: str, obj_kind: str) -> int:
        try:
            return name_to_id[name]
        except KeyError as exc:
            available = ", ".join(sorted(name_to_id))
            raise KeyError(f"Unknown {obj_kind} '{name}'. Available: {available}") from exc

    def get_joint_positions(self, joint_names: list[str]) -> np.ndarray:
        return np.asarray([self.data.joint(name).qpos[0] for name in joint_names], dtype=np.float64)

    def set_joint_positions(self, joint_names: list[str], qpos: np.ndarray, zero_velocity: bool = True) -> None:
        qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if qpos.shape != (len(joint_names),):
            raise ValueError(f"qpos must have shape ({len(joint_names)},)")
        for joint_name, joint_value in zip(joint_names, qpos):
            joint_id = self.require_joint(joint_name)
            qpos_adr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_adr] = float(joint_value)
            if zero_velocity:
                qvel_adr = self.model.jnt_dofadr[joint_id]
                self.data.qvel[qvel_adr] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def get_joint_velocities(self, joint_names: list[str]) -> np.ndarray:
        return np.asarray([self.data.joint(name).qvel[0] for name in joint_names], dtype=np.float64)

    def get_body_position(self, body_name: str) -> np.ndarray:
        body_id = self.require_body(body_name)
        return self.data.xpos[body_id].copy()

    def get_body_quaternion(self, body_name: str) -> np.ndarray:
        body_id = self.require_body(body_name)
        return self.data.xquat[body_id].copy()

    def get_body_rotation(self, body_name: str) -> np.ndarray:
        body_id = self.require_body(body_name)
        return self.data.xmat[body_id].reshape(3, 3).copy()

    def set_mocap_body_pose(self, body_name: str, pos: np.ndarray, quat: np.ndarray) -> None:
        body_id = self.require_body(body_name)
        mocap_id = self.model.body_mocapid[body_id]
        if mocap_id < 0:
            raise ValueError(f"Body '{body_name}' is not a mocap body.")
        pos = np.asarray(pos, dtype=np.float64)
        quat = np.asarray(quat, dtype=np.float64)
        if pos.shape != (3,):
            raise ValueError("pos must have shape (3,)")
        if quat.shape != (4,):
            raise ValueError("quat must have shape (4,)")
        self.data.mocap_pos[mocap_id] = pos
        self.data.mocap_quat[mocap_id] = quat / np.linalg.norm(quat)
        mujoco.mj_forward(self.model, self.data)

    def set_geom_collision_enabled(
        self,
        geom_name: str,
        enabled: bool,
        contype: int | None = None,
        conaffinity: int | None = None,
    ) -> None:
        geom_id = self.require_geom(geom_name)
        if enabled:
            self.model.geom_contype[geom_id] = int(1 if contype is None else contype)
            self.model.geom_conaffinity[geom_id] = int(1 if conaffinity is None else conaffinity)
        else:
            self.model.geom_contype[geom_id] = 0
            self.model.geom_conaffinity[geom_id] = 0
        mujoco.mj_forward(self.model, self.data)

    def get_site_position(self, site_name: str) -> np.ndarray:
        site_id = self.require_site(site_name)
        return self.data.site_xpos[site_id].copy()

    def get_site_rotation(self, site_name: str) -> np.ndarray:
        site_id = self.require_site(site_name)
        return self.data.site_xmat[site_id].reshape(3, 3).copy()

    def get_sensor_data(self, sensor_name: str) -> np.ndarray:
        sensor_id = self.require_sensor(sensor_name)
        start = self.model.sensor_adr[sensor_id]
        dim = self.model.sensor_dim[sensor_id]
        return self.data.sensordata[start : start + dim].copy()

    def get_freejoint_pose(self, joint_name: str) -> tuple[np.ndarray, np.ndarray]:
        joint_id = self.require_joint(joint_name)
        qpos_adr = self.model.jnt_qposadr[joint_id]
        qpos = self.data.qpos[qpos_adr : qpos_adr + 7].copy()
        return qpos[:3], qpos[3:]

    def set_freejoint_pose(self, joint_name: str, pos: np.ndarray, quat: np.ndarray) -> None:
        joint_id = self.require_joint(joint_name)
        qpos_adr = self.model.jnt_qposadr[joint_id]
        qvel_adr = self.model.jnt_dofadr[joint_id]
        pos = np.asarray(pos, dtype=np.float64)
        quat = np.asarray(quat, dtype=np.float64)
        if pos.shape != (3,):
            raise ValueError("pos must have shape (3,)")
        if quat.shape != (4,):
            raise ValueError("quat must have shape (4,)")
        self.data.qpos[qpos_adr : qpos_adr + 3] = pos
        self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat / np.linalg.norm(quat)
        self.data.qvel[qvel_adr : qvel_adr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def list_named_objects(self) -> dict[str, list[str]]:
        return {
            "actuators": sorted(self.actuator_ids),
            "joints": sorted(self.joint_ids),
            "bodies": sorted(self.body_ids),
            "sites": sorted(self.site_ids),
            "sensors": sorted(self.sensor_ids),
            "cameras": sorted(self.camera_ids),
            "equalities": sorted(self.equality_ids),
        }

    def __del__(self) -> None:
        self.close()

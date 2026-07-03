from __future__ import annotations

import numpy as np

from .sim_env import RecyclingSimEnv


class SuctionGripper:
    def __init__(
        self,
        env: RecyclingSimEnv,
        gripper_site: str = "grip_site",
        target_body: str = "red_box",
        target_geom: str = "red_box_geom",
        target_site: str = "box_site",
        target_freejoint: str = "box_free",
        parent_body: str = "link6",
        equality_name: str = "suction_weld_red_box",
        attach_distance: float = 0.08,
        attach_forward_distance: float = 0.06,
        attach_forward_margin: float = 0.025,
        attach_lateral_distance: float = 0.08,
        use_weld: bool = True,
    ) -> None:
        self.env = env
        self.gripper_site = gripper_site
        self.target_body = target_body
        self.target_geom = target_geom
        self.target_site = target_site
        self.target_freejoint = target_freejoint
        self.parent_body = parent_body
        self.equality_name = equality_name
        self.attach_distance = float(attach_distance)
        self.attach_forward_distance = float(attach_forward_distance)
        self.attach_forward_margin = float(attach_forward_margin)
        self.attach_lateral_distance = float(attach_lateral_distance)
        self.use_weld = bool(use_weld)

        self.enabled = False
        self.attached = False
        self._body_offset = np.zeros(3, dtype=np.float64)
        self._quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._equality_id: int | None = None
        self._using_weld = False
        self._refresh_equality_id()

    def _refresh_equality_id(self) -> None:
        self._equality_id = None
        if not self.use_weld:
            return
        try:
            self._equality_id = self.env.require_equality(self.equality_name)
            self.env.data.eq_active[self._equality_id] = 0
        except KeyError:
            self._equality_id = None

    def set_target(
        self,
        *,
        target_body: str,
        target_geom: str,
        target_site: str,
        target_freejoint: str,
        equality_name: str,
    ) -> None:
        if self.attached:
            self.release()
        self.target_body = target_body
        self.target_geom = target_geom
        self.target_site = target_site
        self.target_freejoint = target_freejoint
        self.equality_name = equality_name
        self._refresh_equality_id()

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        if not self.enabled and self.attached:
            self.release()

    def is_attached(self) -> bool:
        return self.attached

    def get_distance(self) -> float:
        grip_pos = self.env.get_site_position(self.gripper_site)
        target_pos = self.env.get_site_position(self.target_site)
        return float(np.linalg.norm(grip_pos - target_pos))

    def get_target_pose_in_gripper_frame(self) -> np.ndarray:
        grip_pos = self.env.get_site_position(self.gripper_site)
        grip_rot = self.env.get_site_rotation(self.gripper_site)
        body_pos = self.env.get_body_position(self.target_body)
        return grip_rot.T @ (body_pos - grip_pos)

    def _passes_pybullet_like_check(self) -> bool:
        rel_pos = self.get_target_pose_in_gripper_frame()
        approach_axis = 0
        lateral_distance = float(np.linalg.norm(rel_pos[1:3]))

        # The approach axis is vertical for the top-down suction grasp, so the
        # forward limit should scale with the object's HEIGHT (geom_size[2], the
        # z half-extent), not its footprint. (It used geom_size[approach_axis] =
        # the x half-extent -- harmless for the old equal-sided cubes, but the
        # wrong dimension for the flat/elongated objects.)
        geom_size = np.asarray(self.env.model.geom(self.target_geom).size[:3], dtype=np.float64)
        forward_limit = max(
            self.attach_forward_distance,
            float(geom_size[2] + self.attach_forward_margin),
        )
        return bool(abs(rel_pos[approach_axis]) <= forward_limit and lateral_distance <= self.attach_lateral_distance)

    def passes_geometry(self) -> bool:
        """Geometric feasibility of the suction grasp (object inside the suction
        window) WITHOUT attaching -- the grasp 'instant' is the first tick this is
        True. Mirrors the check inside `try_attach`, exposed so callers can decide
        (e.g. a stochastic suction roll) before committing to the weld."""
        return self._passes_pybullet_like_check()

    def _quat_conjugate(self, quat: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat, dtype=np.float64)
        return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)

    def _quat_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = np.asarray(q1, dtype=np.float64)
        w2, x2, y2, z2 = np.asarray(q2, dtype=np.float64)
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float64,
        )

    def _set_weld_relpose(self) -> None:
        if self._equality_id is None:
            return

        parent_pos = self.env.get_body_position(self.parent_body)
        parent_rot = self.env.get_body_rotation(self.parent_body)
        parent_quat = self.env.get_body_quaternion(self.parent_body)
        body_pos = self.env.get_body_position(self.target_body)
        body_quat = self.env.get_body_quaternion(self.target_body)

        rel_pos = parent_rot.T @ (body_pos - parent_pos)
        rel_quat = self._quat_multiply(self._quat_conjugate(parent_quat), body_quat)
        rel_quat = rel_quat / np.linalg.norm(rel_quat)

        self.env.model.eq_data[self._equality_id, 3:6] = rel_pos
        self.env.model.eq_data[self._equality_id, 6:10] = rel_quat

    def try_attach(self) -> bool:
        if not self.enabled:
            return False
        if self.get_distance() > self.attach_distance and not self._passes_pybullet_like_check():
            return False
        if not self._passes_pybullet_like_check():
            return False

        grip_pos = self.env.get_site_position(self.gripper_site)
        body_pos, body_quat = self.env.get_freejoint_pose(self.target_freejoint)
        self._body_offset = body_pos - grip_pos
        self._quat = body_quat
        self.attached = True
        self._using_weld = self._equality_id is not None
        if self._using_weld:
            self._set_weld_relpose()
            self.env.data.eq_active[self._equality_id] = 1
        self.update()
        return True

    def update(self) -> None:
        if not (self.enabled and self.attached):
            return
        if self._using_weld:
            return
        grip_pos = self.env.get_site_position(self.gripper_site)
        body_pos = grip_pos + self._body_offset
        self.env.set_freejoint_pose(self.target_freejoint, body_pos, self._quat)

    def release(self) -> None:
        if self._using_weld and self._equality_id is not None:
            self.env.data.eq_active[self._equality_id] = 0
        self._using_weld = False
        self.attached = False


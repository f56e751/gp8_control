"""Detection polling + camera-to-robot pose conversion.

Wraps the non-blocking poll of a perception source and the per-detection
camera→robot transform that used to live as private methods on
``GP8App``. The output is a list of ``GraspCandidate`` ready to be added
to the tracked-object queue.

The source is duck-typed: it only needs ``.positions`` / ``.class_names`` /
``.delay`` attributes. Both the old ROS-callback-driven ``SAMClient`` and the
HTTP-stream ``StreamDetectionSource`` satisfy this. When ``node`` is None
(the stream source needs no ROS spinning), ``poll`` just reads the latest
snapshot instead of pumping ROS callbacks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import rclpy
from rclpy.node import Node


def _make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.ravel()
    return T


# Default tool orientation: tool pointing down at the belt.
_R_GRASP_DEFAULT = np.array(
    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
)


@dataclass
class GraspCandidate:
    T_aim: np.ndarray
    T_grasp: np.ndarray
    class_name: str
    cam_pos: tuple   # raw camera-frame position [cx, cy, cz] from the stream


class DetectionIntake:
    def __init__(
        self,
        node: Node | None,
        sam_client,
        T_robot2base: np.ndarray,
        T_base2cam: np.ndarray,
        offset_aim: float,
        offset_grasp: float,
        time_step: float,
        # Workspace filter — Z<0.67 keeps belt-surface detections; X∈(-0.2,0.2)
        # rejects anything off the belt centerline.
        workspace_z_max: float = 0.67,
        workspace_x_abs: float = 0.2,
        logger=None,
        log_raw: bool = False,
    ) -> None:
        self._node = node
        self._sam = sam_client
        self._T_robot2base = T_robot2base
        self._T_base2cam = T_base2cam
        self._offset_aim = offset_aim
        self._offset_grasp = offset_grasp
        self._time_step = time_step
        self._z_max = workspace_z_max
        self._x_abs = workspace_x_abs
        self._R_grasp = _R_GRASP_DEFAULT
        # Optional raw-detection logging: when set, every detection seen on
        # the stream is logged with its camera-frame position, transformed
        # base-frame grasp position, and the filter decision. Used to debug
        # cases like base-X stuck at the constant 0.425 (camera sent [0,0,0]).
        self._logger = logger
        self._log_raw = log_raw

    def _camera_to_grasp(self, position) -> tuple[np.ndarray, np.ndarray]:
        T_cam = np.eye(4)
        T_cam[:3, 3] = np.asarray(position, dtype=float)
        T_robot = self._T_robot2base @ self._T_base2cam @ T_cam
        grasp_pos = T_robot[:3, 3]
        T_aim = _make_transform(
            self._R_grasp, grasp_pos + np.array([0.0, 0.0, self._offset_aim])
        )
        T_grasp = _make_transform(
            self._R_grasp, grasp_pos + np.array([0.0, 0.0, self._offset_grasp])
        )
        return T_aim, T_grasp

    def _in_workspace(self, position) -> bool:
        return position[2] < self._z_max and -self._x_abs < position[0] < self._x_abs

    # ------------------------------------------------------------------
    def poll(
        self, time_to_check: float = 0.3
    ) -> tuple[list[GraspCandidate], float]:
        """Poll SAM client for ``time_to_check`` seconds.

        Returns ``(candidates, perception_delay)`` — the delay is the
        SAM-side timestamp the caller uses to back-project the conveyor
        motion that occurred between detection and now.
        """
        s_time = time.time()
        positions = None
        class_names = None
        delay = 0.0

        while time.time() - s_time < time_to_check:
            if self._node is not None:
                # Old SAMClient path: pump ROS callbacks so the camera_info
                # subscription can populate the buffers.
                rclpy.spin_once(self._node, timeout_sec=0.01)
            else:
                # Stream-source path: no ROS callbacks to pump. Pace the loop
                # so a cold start (no record yet) doesn't busy-wait a core.
                time.sleep(self._time_step)
            p = self._sam.positions
            n = self._sam.class_names
            if p is not None and n is not None:
                if p and n:
                    positions = p
                    class_names = n
                    delay = self._sam.delay or 0.0
                    break
                # Stale / empty publish — wait and retry
                time.sleep(self._time_step)
                continue
            self._sam.positions = None
            self._sam.class_names = None

        if positions is None or class_names is None:
            return [], 0.0

        candidates: list[GraspCandidate] = []
        for pos, cls in zip(positions, class_names):
            in_ws = self._in_workspace(pos)
            T_aim, T_grasp = (None, None)
            if in_ws:
                T_aim, T_grasp = self._camera_to_grasp(pos)
                candidates.append(
                    GraspCandidate(
                        T_aim=T_aim, T_grasp=T_grasp, class_name=cls,
                        cam_pos=(float(pos[0]), float(pos[1]), float(pos[2])),
                    )
                )

            if self._log_raw and self._logger is not None:
                cx, cy, cz = (float(pos[0]), float(pos[1]), float(pos[2]))
                if in_ws and T_grasp is not None:
                    bx, by, bz = (
                        float(T_grasp[0, 3]),
                        float(T_grasp[1, 3]),
                        float(T_grasp[2, 3]),
                    )
                    self._logger.info(
                        f"raw cam=[{cx:+.4f},{cy:+.4f},{cz:+.4f}] cls={cls}"
                        f" -> base x={bx:+.3f} y={by:+.3f} z={bz:+.3f}"
                        f" (delay {delay:.3f}s)"
                    )
                else:
                    self._logger.info(
                        f"raw cam=[{cx:+.4f},{cy:+.4f},{cz:+.4f}] cls={cls}"
                        f" FILTERED (workspace) (delay {delay:.3f}s)"
                    )
        return candidates, delay

"""Shared resources and motion helpers handed to every skill."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import rclpy

from gp8_control.trajectory.trajectory_primitive import trajectory

if TYPE_CHECKING:
    from rclpy.node import Node
    from gp8_control.app import Config
    from gp8_control.robots.gp8 import GP8
    from gp8_control.controllers.trajectory_controller import TrajectoryController
    from gp8_control.planning import PickThrowPlanner
    from gp8_control.perception.conveyor_speed import ConveyorSpeedTracker
    from gp8_control.tracking import TrackedObject, TrackedObjectQueue


@dataclass
class PickRequest:
    """Selected target plus the intercept geometry needed to act on it.

    Produced by the app's target-selection step (``_select_ambush_target``)
    and consumed by whichever skill the ``ActionSelector`` picks, so a skill
    can approach and manipulate the object without re-running selection.
    """

    target: "TrackedObject"
    current_joint: np.ndarray
    T_aim: np.ndarray
    T_grasp: np.ndarray
    aim_joint: np.ndarray
    grasp_joint: np.ndarray
    secondary: "Optional[TrackedObject]" = None


@dataclass
class SkillContext:
    """Robot resources + cross-cutting helpers shared by all skills.

    Bundles the hardware/algorithm handles and the loop-owned callbacks
    (perception intake, viz publish, status, active-target) so skills don't
    reach back into ``GP8App``. The app builds one of these in ``setup()`` and
    injects it into each skill. The motion/timing helpers below are the
    primitives every skill (throw, push, ...) reuses to approach and meet an
    object at the intercept line.
    """

    cfg: "Config"
    node: "Node"
    robot: "GP8"
    traj_ctrl: "TrajectoryController"
    planner: "PickThrowPlanner"
    conveyor: "ConveyorSpeedTracker"
    queue: "TrackedObjectQueue"
    M1: np.ndarray
    M2: np.ndarray
    # Cross-cutting operations owned by the app loop, wired in by GP8App.
    intake: Callable[[float], None]
    publish_state: Callable[[], None]
    set_status: Callable[[str, str], None]
    set_active_target: Callable[["Optional[TrackedObject]"], None]

    @property
    def log(self):
        return self.node.get_logger()

    # ------------------------------------------------------------------
    # Shared motion / timing primitives
    # ------------------------------------------------------------------
    def object_y_now(self, target: "TrackedObject", now: float, v: float) -> float:
        """Object's belt-frame Y at ``now`` (belt travels -Y, so Y decreases)."""
        return target.T_grasp_base[1, 3] - v * (now - target.detect_time)

    def move_through(
        self,
        current_joint: np.ndarray,
        aim_joint: np.ndarray,
        grasp_joint: np.ndarray,
    ) -> None:
        """Queue-mode move directly to the grasp pose, no aim hover.

        The legacy 3-point motion went current -> aim (hover ~8 cm above
        grasp) -> grasp. After the throw's chain leaves the arm at the
        previous intercept's GRASP_Z, that hover meant every cycle bounced
        UP to aim and back DOWN to grasp — which is what made the suction
        fire at "a position much higher than the object." Go direct to
        grasp so the arm stays at GRASP_Z when transitioning between
        picks. ``aim_joint`` is kept in the signature for back-compat but
        unused.
        """
        zero = np.zeros_like(self.M1)
        traj, vel, ts = trajectory(
            current_joint, zero, grasp_joint, zero,
            self.M1, self.M2, hertz=self.cfg.TRAJ_HZ,
        )
        self.traj_ctrl.send_trajectory_queue(traj, vel, ts, final_joint=grasp_joint)

    def sleep_until(self, deadline: float) -> None:
        """Block until ``deadline`` (wall clock), staying responsive to shutdown.

        During the long ambush wait we still pump ROS callbacks (so belt
        speed and detection snapshots stay fresh), publish viz state, and
        — importantly — keep ingesting new detections into the queue and
        updating its order. Otherwise objects that arrive on the belt
        during the wait are invisible to app.py until the current cycle's
        throw completes (~10 s later), often too late to catch.
        """
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.0)
            self.publish_state()
            now = time.time()
            self.intake(now)
            if self.queue is not None:
                self.queue.update(now, self.conveyor.current if self.conveyor else 0.0)
            # Clamp non-negative: the loop body (spin + intake + queue.update)
            # can overrun the remaining time-to-deadline, making the delta
            # negative -> time.sleep() would raise "sleep length must be
            # non-negative" and kill the epoch mid-wait (arm parked, no
            # suction/throw). max(0.0, ...) lets the while-guard exit instead.
            time.sleep(max(0.0, min(0.05, deadline - time.time())))

    def wait_for_arrival_and_suction(
        self, target: "TrackedObject", intercept_y: float
    ) -> None:
        """One-shot wait, parked at the grasp pose. Suction fires SUCTION_LEAD
        seconds before the object arrives so the vacuum is already pulling, but
        this method only returns at the predicted arrival — so the caller's
        lift/throw motion starts on time (at eta), not early.

        eta is computed once here from remaining distance / belt speed sampled
        now (no per-tick recompute). Assumes a roughly steady belt.
        """
        now = time.time()
        v = self.conveyor.current
        obj_y = self.object_y_now(target, now, v)
        eta = (obj_y - intercept_y) / (v + 1e-6)
        eta = max(0.0, min(eta, self.cfg.AMBUSH_MAX_WAIT))
        lead = min(self.cfg.SUCTION_LEAD, eta)   # can't fire before now
        self.log.info(
            f"Ambush: suction in {eta - lead:.2f}s, arrival/lift in {eta:.2f}s "
            f"(dist {obj_y - intercept_y:.3f} m / belt {v:.3f} m/s, lead {lead:.2f}s)"
        )

        # 1) park until SUCTION_LEAD before arrival, then suction ON (still parked)
        self.sleep_until(now + eta - lead)
        self.traj_ctrl.suction_on()
        # 2) keep sucking, parked, until the object actually arrives — then return
        #    so the lift/throw motion begins at eta.
        self.sleep_until(now + eta)

    def wait_for_arrival(
        self, target: "TrackedObject", intercept_y: float, offset = 0.18
    ) -> None:
        """One-shot wait, parked at the grasp pose. Suction fires SUCTION_LEAD
        seconds before the object arrives so the vacuum is already pulling, but
        this method only returns at the predicted arrival — so the caller's
        lift/throw motion starts on time (at eta), not early.

        eta is computed once here from remaining distance / belt speed sampled
        now (no per-tick recompute). Assumes a roughly steady belt.
        """
        now = time.time()
        v = self.conveyor.current
        obj_y = self.object_y_now(target, now, v)
        eta = (obj_y - intercept_y) / (v + 1e-6)
        eta = max(0.0, min(eta, self.cfg.AMBUSH_MAX_WAIT))
        lead = min(self.cfg.SUCTION_LEAD, eta)   # can't fire before now
        self.log.info(
            f"Ambush: suction in {eta - lead:.2f}s, arrival/lift in {eta:.2f}s "
            f"(dist {obj_y - intercept_y:.3f} m / belt {v:.3f} m/s, lead {lead:.2f}s)"
        )
        self.sleep_until(now + eta - offset)

    def scan_next_intercept(self) -> "Optional[np.ndarray]":
        """Re-poll the live queue for the next reachable object's intercept joint.

        Used as the post-release chain target so the arm flows straight to the
        next pick instead of parking. Re-polls here (NOT a value captured at
        lock time) because new objects may have entered the queue during the
        ~10 s ambush wait. Walks from the head and returns the first reachable
        candidate's intercept joint config, or ``None`` if none is reachable.
        """
        self.queue.update(
            time.time(),
            self.conveyor.current if self.conveyor is not None else 0.0,
        )
        for cand in list(self.queue._objects):
            T_next_grasp = cand.T_grasp_base.copy()
            T_next_grasp[1, 3] = self.cfg.GRASP_INTERCEPT_Y
            T_next_grasp[2, 3] = self.cfg.GRASP_Z
            if np.linalg.norm(T_next_grasp[:2, 3]) > self.cfg.MAX_REACH:
                continue
            ik = self.robot.inverse_kinematics(T_next_grasp)
            if ik is None:
                continue
            next_intercept_joint = np.asarray(ik, dtype=float)
            next_intercept_joint[-1] = 0.0
            self.log.info(
                f"Throw chain target: {cand.class_name} at "
                f"x={float(T_next_grasp[0, 3]):+.3f}"
            )
            return next_intercept_joint
        return None

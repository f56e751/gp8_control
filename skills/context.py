"""Shared resources and motion helpers handed to every skill."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import rclpy

from gp8_control.trajectory.trajectory_primitive import trajectory, opt_time

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
    # Set True by the prior throw's return (chain) when it already primed the
    # NEXT pick's suction (vacuum ON). The upcoming pick then must NOT re-prime
    # or clear it; it is reset to False once that pick consumes it.
    suction_primed_for_pick: bool = False

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

    def position_and_prime(
        self,
        current_joint: np.ndarray,
        aim_joint: np.ndarray,
        grasp_joint: np.ndarray,
        target: "TrackedObject",
        intercept_y: float,
        skip_move: bool = False,
    ) -> None:
        """Drive to the grasp pose, priming suction SUCTION_LEAD before arrival.

        Unlike the old "position (blocking) THEN wait+suction" split (which fired
        suction only after positioning finished, so a slow positioning ate into
        the lead), the suction-on here is keyed to an ABSOLUTE wall-clock instant
        ``t_suction = arrival - SUCTION_LEAD``. If that instant falls while the
        arm is still positioning, suction fires mid-move (priming the vacuum
        early is harmless). This guarantees the full SUCTION_LEAD regardless of
        how long positioning takes, so a borderline pick keeps its lead. Returns
        once the object has reached the intercept (caller then lifts/throws).
        """
        # If the prior throw's return (chain) already primed THIS object's
        # suction (vacuum ON), don't re-prime — just drive to grasp and wait.
        already_primed = self.suction_primed_for_pick
        self.suction_primed_for_pick = False

        now = time.time()
        v = self.conveyor.current
        obj_y = self.object_y_now(target, now, v)
        eta = max(0.0, min((obj_y - intercept_y) / (v + 1e-6), self.cfg.AMBUSH_MAX_WAIT))
        t_arrival = now + eta
        t_suction = t_arrival - self.cfg.SUCTION_LEAD     # absolute; may be <= now
        self.log.info(
            f"Ambush: {'suction PRE-primed on return; ' if already_primed else ''}"
            f"prime {self.cfg.SUCTION_LEAD:.2f}s before arrival, arrival/lift in "
            f"{eta:.2f}s (dist {obj_y - intercept_y:.3f} m / belt {v:.3f} m/s)"
        )

        zero = np.zeros_like(self.M1)
        if skip_move:
            # Arm already AT the grasp pose (the return swing parked it here) — no
            # drive, no mode switch. Just prime (if not already on) and wait, so
            # the return swing flows straight into the grab without a bobble.
            if not already_primed:
                self.sleep_until(t_suction)
                self.traj_ctrl.suction_on()
        else:
            traj, vel, ts = trajectory(
                current_joint, zero, grasp_joint, zero,
                self.M1, self.M2, hertz=self.cfg.TRAJ_HZ,
            )
            if already_primed:
                # Vacuum already on (primed during the return chain) — just drive in.
                self.traj_ctrl.send_trajectory_queue(traj, vel, ts, final_joint=grasp_joint)
            else:
                # Fire suction the instant t_suction passes — mid-move when the
                # object is already within SUCTION_LEAD by the time we get there.
                fired = self.traj_ctrl.send_trajectory_queue_timed_suction(
                    traj, vel, ts, final_joint=grasp_joint, suction_on_at=t_suction,
                )
                if not fired:
                    # t_suction still ahead -> park until it, then prime.
                    self.sleep_until(t_suction)
                    self.traj_ctrl.suction_on()
        self.set_status("WAITING", getattr(target, "class_name", ""))
        # Wait out the rest until arrival, then return so the throw begins right
        # as the object reaches the intercept.
        self.sleep_until(t_arrival)

    def scan_next_intercept(
        self, from_joint: np.ndarray, throw_time: float,
    ) -> "tuple[Optional[np.ndarray], Optional[float]]":
        """Re-poll the queue for the next object the pick will ACTUALLY complete,
        and the absolute wall-clock instant to prime its suction.

        Uses the SAME feasibility gate as ``_select_ambush_target`` so the throw's
        return (chain) only commits to — and primes — an object the next pick can
        finish: the arm must reach that intercept before the object arrives. The
        arm gets there only after it finishes the current throw, so the gate adds
        ``throw_time``; ``opt_time`` over-estimates the move ~2x so it is scaled
        by PICK_FEASIBILITY_FACTOR:

            feasible iff  eta >= throw_time + move_time * factor

        This is what makes "went there => will pick it" hold: without it the chain
        flies to an intercept the pick gate then rejects (= "went but never
        picked"). ``from_joint`` is the move-estimate reference (the current
        pick's grasp, where the swing lifts from). Returns
        ``(next_intercept_joint, suction_on_at)`` for the first feasible object,
        else ``(None, None)`` — then the throw just returns to the current
        intercept and the next pick is chosen fresh. ``suction_on_at`` =
        (that object's arrival - SUCTION_LEAD), primed during the chain.
        """
        now = time.time()
        v = self.conveyor.current if self.conveyor is not None else 0.0
        self.queue.update(now, v)
        zero = np.zeros_like(self.M1)
        factor = self.cfg.PICK_FEASIBILITY_FACTOR
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
            cand_y = self.object_y_now(cand, now, v)
            eta = max(0.0, (cand_y - self.cfg.GRASP_INTERCEPT_Y) / (v + 1e-6))
            # SAME gate as _select_ambush_target, plus the throw the arm must
            # finish first: only chain to / prime an object the next pick can
            # actually complete (else it would fly to an intercept the pick gate
            # then rejects).
            move_time = opt_time(
                from_joint, zero, next_intercept_joint, zero, self.M1, self.M2,
            )
            needed = throw_time + move_time * factor
            if eta < needed:
                self.log.info(
                    f"Chain skip {cand.class_name}: can't complete in time "
                    f"(eta {eta:.2f}s < throw {throw_time:.2f}s + move "
                    f"{move_time:.2f}s × {factor:.2f} = {needed:.2f}s)"
                )
                continue
            suction_on_at = now + eta - self.cfg.SUCTION_LEAD
            self.log.info(
                f"Chain target: {cand.class_name} at "
                f"x={float(T_next_grasp[0, 3]):+.3f} (prime in "
                f"{max(0.0, suction_on_at - now):.2f}s, arrival {eta:.2f}s)"
            )
            return next_intercept_joint, suction_on_at
        return None, None

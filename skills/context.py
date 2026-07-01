"""Shared resources and motion helpers handed to every skill."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import rclpy

from gp8_control.trajectory.trajectory_primitive import (
    trajectory,
    trajectory_3points,
    decimate_for_queue,
    opt_time,
)

if TYPE_CHECKING:
    from rclpy.node import Node
    from gp8_control.app import Config
    from gp8_control.robots.gp8 import GP8
    from gp8_control.controllers.trajectory_controller import TrajectoryController
    from gp8_control.planning import PickThrowPlanner
    from gp8_control.conveyor import ConveyorSpeedTracker
    from gp8_control.tracking import TrackedObject, TrackedObjectQueue
    from gp8_control.skills.base import ManipulationSkill


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
    # True when this object IS the previous throw's committed return target — the
    # swing already parked the arm at its grasp pose, so the pick must not re-enter
    # queue mode (stop would chop the swing) or re-drive; just wait + grab.
    prepositioned: bool = False


# Fixed-point convergence for earliest_reachable_intercept (intercept-Y <-> move-time
# depend on each other because the object keeps moving while the arm moves). The loop
# is cheap (closed-form IK + arithmetic), so we converge tightly.
_INTERCEPT_TOL = 1e-4          # m
_INTERCEPT_RELAX = 0.5         # under-relaxation for stability (plan_pick-style)
_INTERCEPT_MAX_ITERS = 30      # generous cap; a 1-D contraction converges in a few


@dataclass
class Intercept:
    """Earliest reachable intercept for one object: where + how to grab it.

    ``intercept_y`` is the dynamic belt-frame Y to grab at (NOT the old fixed
    ``GRASP_INTERCEPT_Y`` line); ``eta`` is the time from now until the object
    reaches that Y; ``move_time`` is the arm's estimated travel to the grasp pose.
    Produced by :meth:`SkillContext.earliest_reachable_intercept`.
    """

    intercept_y: float
    T_aim: np.ndarray
    T_grasp: np.ndarray
    aim_joint: np.ndarray
    grasp_joint: np.ndarray
    eta: float
    move_time: float


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
    # Skill NAME that will handle a given object (wired to ActionSelector.skill_for).
    # Lets the chain pre-position the NEXT object with the skill that will run it.
    skill_for: Callable[["TrackedObject"], str]
    # The SKILL OBJECT (not just its name) that will handle a given object. Lets the
    # intercept solver query that skill's t_to_contact() timeline — push and throw
    # have different setup/positioning/contact costs, so the grasp must be placed
    # using the skill that will actually run. Wired in GP8App._build_skills.
    skill_obj_for: Callable[["TrackedObject"], "ManipulationSkill"]
    # Default standby pose (6-DOF joint vector, wrist/j6 = 0) the post-action
    # chain returns to when there is NO next object to pre-position. Computed once
    # in GP8App._build_skills from cfg.INITIAL_R/T (the boot pose). Skills reach it
    # via ManipulationSkill.idle_target() — see base.py. Returning a TARGET (not a
    # motion) keeps the return folded into each skill's single chained trajectory,
    # so no extra dispatch / queue-mode re-entry (which would chop the swing).
    idle_joint: np.ndarray
    # Set True by the prior throw's return (chain) when it already primed the
    # NEXT pick's suction (vacuum ON). The upcoming pick then must NOT re-prime
    # or clear it; it is reset to False once that pick consumes it.
    suction_primed_for_pick: bool = False
    # The object the throw's return swing was committed to (sent to its intercept).
    # The next selection marks its request ``prepositioned`` iff it picks THIS same
    # object — so "already at the grasp" is decided by object identity, not a
    # distance guess. Consumed (cleared) by the next selection.
    committed_next: "Optional[TrackedObject]" = None
    # The dynamic intercept (Y + grasp/aim poses + joints) chosen for ``committed_next``
    # at commit time. The catchable judgment is made ONCE here; the next selection
    # REUSES this instead of recomputing, so the parked pose and the pick pose agree.
    committed_intercept: "Optional[Intercept]" = None

    @property
    def log(self):
        return self.node.get_logger()

    # ------------------------------------------------------------------
    # Shared motion / timing primitives
    # ------------------------------------------------------------------
    def object_y_now(self, target: "TrackedObject", now: float, v: float) -> float:
        """Object's belt-frame Y at ``now`` (belt travels -Y, so Y decreases)."""
        return target.T_grasp_base[1, 3] - v * (now - target.detect_time)

    def earliest_reachable_intercept(
        self,
        target: "TrackedObject",
        current_joint: np.ndarray,
        v: float,
        now: float,
        pre_delay: float = 0.0,
        t_to_contact_fn: "Optional[Callable[[float], float]]" = None,
    ) -> "Optional[Intercept]":
        """Earliest belt-Y at which the arm can grab ``target``, or ``None`` if it
        can't be caught anywhere in the workspace before passing downstream.

        The object travels in -Y; for its lane X the reach window is
        ``Y in [-y_b, +y_b]`` with ``y_b = sqrt(MAX_REACH^2 - x^2)``. We grab as
        EARLY as possible: if the object is still upstream of the entry edge
        ``+y_b`` when the arm can be ready, wait at ``+y_b``; otherwise meet it
        where it will be when the arm arrives (which may be downstream of the old
        ``y=0`` line — that is the point). We give up ONLY when the object would
        pass the downstream edge ``-y_b`` before the arm — after any ``pre_delay``
        (e.g. the current throw the arm must finish first) — can reach it. Being
        upstream (not yet arrived) is never a reason to give up.

        intercept-Y and the arm's ``move_time`` depend on each other (the object
        keeps moving while the arm moves), so we fixed-point iterate to
        convergence. The loop is cheap (closed-form IK + arithmetic). The
        ``PICK_FEASIBILITY_FACTOR`` margin is baked into the arm-travel estimate
        here, so callers just use the result (or drop on ``None``).

        ``current_joint`` is the move-estimate reference (the pose the arm lifts
        from). ``pre_delay`` is dead time before the arm starts moving (0 for the
        immediate pick; the throw duration for the post-throw chain).
        """
        cfg = self.cfg
        x = float(target.T_grasp_base[0, 3])
        obj_y = self.object_y_now(target, now, v)
        denom = cfg.MAX_REACH ** 2 - x ** 2
        if denom <= 0.0:
            return None                          # lane laterally out of reach (degenerate)
        y_b = float(np.sqrt(denom))
        if obj_y < -y_b:
            return None                          # already past the downstream reach edge

        zero = np.zeros_like(self.M1)
        factor = cfg.PICK_FEASIBILITY_FACTOR
        approach_dz = float(target.T_aim_base[2, 3] - target.T_grasp_base[2, 3])
        T_grasp = target.T_grasp_base.copy()
        T_aim = target.T_aim_base.copy()
        T_grasp[0, 3] = x
        T_aim[0, 3] = x

        y_guess = float(min(obj_y, y_b))         # seed at the object, capped at the entry edge
        y_eval = y_guess
        aj = gj = None
        move_time = 0.0
        for _ in range(_INTERCEPT_MAX_ITERS):
            y_eval = y_guess
            T_grasp[1, 3] = y_eval
            T_grasp[2, 3] = cfg.GRASP_Z
            T_aim[1, 3] = y_eval
            T_aim[2, 3] = cfg.GRASP_Z + approach_dz
            gj = self.robot.inverse_kinematics(T_grasp)
            aj = self.robot.inverse_kinematics(T_aim)
            if gj is None or aj is None:
                return None
            gj = np.asarray(gj, dtype=float); gj[-1] = 0.0
            aj = np.asarray(aj, dtype=float); aj[-1] = 0.0
            move_time = (
                opt_time(current_joint, zero, aj, zero, self.M1, self.M2)
                + opt_time(aj, zero, gj, zero, self.M1, self.M2)
            )
            # Where the object will be once the arm reaches CONTACT (after pre_delay).
            # The contact budget is the skill's real timeline (t_to_contact_fn:
            # setup + positioning + contact offset) when supplied, else the legacy
            # opt_time*factor proxy. This is the placement+feasibility fix: the old
            # proxy omitted queue-re-entry + push pre-travel, so the grasp was aimed
            # upstream of where the object actually was at strike.
            budget = (
                t_to_contact_fn(move_time) if t_to_contact_fn is not None
                else move_time * factor
            )
            obj_y_arrival = obj_y - v * (pre_delay + budget)
            if obj_y_arrival < -y_b:
                return None                      # exits downstream before the arm arrives
            target_y = min(obj_y_arrival, y_b)   # wait at the entry edge if still upstream
            if abs(target_y - y_eval) < _INTERCEPT_TOL:
                break
            y_guess = (1.0 - _INTERCEPT_RELAX) * y_eval + _INTERCEPT_RELAX * target_y

        eta = (obj_y - y_eval) / (v + 1e-6)
        return Intercept(
            intercept_y=y_eval,
            T_aim=T_aim.copy(),
            T_grasp=T_grasp.copy(),
            aim_joint=aj,
            grasp_joint=gj,
            eta=eta,
            move_time=move_time,
        )

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
        # Thin to >= queue-gap so the synchronous point push keeps up with the
        # robot's consumption — otherwise a long positioning move (e.g. from the
        # bin-side push_end back to the next intercept) drains the queue and
        # MotoROS2 exits queue mode (code 2 'Must call start_point_queue_mode').
        traj, vel, ts = decimate_for_queue(traj, vel, ts)
        self.traj_ctrl.send_trajectory_queue(traj, vel, ts, final_joint=grasp_joint)

    def move_through_via(
        self,
        current_joint: np.ndarray,
        aim_joint: np.ndarray,
        grasp_joint: np.ndarray,
    ) -> None:
        """Queue-mode move current -> aim -> grasp, ACTUALLY passing through aim.

        Unlike :meth:`move_through` (which ignores ``aim_joint`` and cuts a
        direct path to ``grasp_joint``), this routes through the ``aim_joint``
        via-point (zero velocity there) using ``trajectory_3points``. The push
        skill uses it so the arm rises to the high aim hover before descending
        to the low push-start pose, instead of cutting a direct (possibly
        belt-dipping) path from a far parked pose like the bin-side push_end.
        """
        zero = np.zeros_like(self.M1)
        traj, vel, ts = trajectory_3points(
            current_joint, zero,
            aim_joint, zero,
            grasp_joint, zero,
            self.M1, self.M2, hertz=self.cfg.TRAJ_HZ,
        )
        # Thin to >= queue-gap so the synchronous point push keeps up with the
        # robot's consumption (see move_through) — a long via-routed positioning
        # has even more points, so this matters more here.
        traj, vel, ts = decimate_for_queue(traj, vel, ts)
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

    def position_and_prime(
        self,
        current_joint: np.ndarray,
        aim_joint: np.ndarray,
        grasp_joint: np.ndarray,
        target: "TrackedObject",
        intercept_y: float,
        skip_move: bool = False,
        start_lead: "Optional[float]" = None,
        persistent: bool = False,
    ) -> bool:
        """Drive to the grasp pose, priming suction SUCTION_LEAD before arrival.

        Returns True on success; False (persistent path only) if a queue push was
        rejected / the queue drained during the wait, so the caller aborts cleanly.

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
        if persistent:
            # Stage-C persistent path: ONE queue session stays alive from the pick
            # drive through the ambush hold so the throw needs NO re-entry.
            # ``skip_move`` = a CROSS-CYCLE prepositioned reuse: the prior throw's
            # chain already flew the arm to this grasp on the SAME live session, so
            # there is no drive to push — go straight to the keep-alive hold (which
            # continues the still-streaming chain into holds at grasp).
            st = {"fired": already_primed}

            def _prime() -> None:
                if not st["fired"] and time.time() >= t_suction:
                    self.traj_ctrl.suction_on()
                    st["fired"] = True

            if not skip_move:
                traj, vel, ts = trajectory(
                    current_joint, zero, grasp_joint, zero,
                    self.M1, self.M2, hertz=self.cfg.TRAJ_HZ,
                )
                ok_drive, _ = self.traj_ctrl.pq_segment(
                    traj, vel, ts, grasp_joint, between_fn=_prime)
                if not ok_drive:
                    self.log.error("[PQ] pick drive push rejected; aborting pick.")
                    return False
            self.set_status("WAITING", getattr(target, "class_name", ""))
            if start_lead is None:
                start_lead = self.cfg.ACTION_START_LEAD
            # pq_hold_until returns False iff the queue drained during the wait
            # (cross-cycle: iff the session died before this cycle resumed it).
            ok_hold = self.traj_ctrl.pq_hold_until(
                grasp_joint, t_arrival - start_lead, tick_fn=_prime)
            # GUARANTEE the suction prime: on a reuse cycle (skip_move) the hold is
            # the ONLY priming hook, and if the object is already within start_lead
            # of arrival the hold loop never iterates -> _prime never fires -> empty
            # gripper. Mirror the non-persistent skip_move path's unconditional prime.
            if not st["fired"] and time.time() >= t_suction:
                self.traj_ctrl.suction_on()
                st["fired"] = True
            return ok_hold
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
        # End the wait `start_lead` s before predicted arrival so the post-wait
        # queue-mode re-entry (~0.4 s) overlaps the object's final approach and the
        # lift lands on arrival instead of trailing it. start_lead defaults to the
        # shared cfg.ACTION_START_LEAD; the skill passes its own arrival_lead().
        if start_lead is None:
            start_lead = self.cfg.ACTION_START_LEAD
        self.sleep_until(t_arrival - start_lead)
        return True

    def lifted_standby_joint(self, end_joint: np.ndarray) -> np.ndarray:
        """Park pose for when a next object EXISTS but wasn't committed for
        pre-position (it routes to a different skill, or isn't reachable after this
        action). Keeps the action's end XY but RAISES Z to the home/idle height, so
        the arm lifts straight up off the belt — avoiding the low-to-low belt sweep
        that motivated parking at home — WITHOUT the wasteful full trip to the home
        pose. Only a truly empty queue (no next object detected yet) returns all the
        way home (via idle_joint); the callers choose between the two. Falls back to
        idle_joint on FK/IK failure.
        """
        end_joint = np.asarray(end_joint, dtype=float)
        T = self.robot.forward_kinematics(end_joint[:6])
        T[2, 3] = float(self.cfg.INITIAL_T[2, 0])      # home/idle Z (cfg.INITIAL_T)
        q = self.robot.inverse_kinematics(T, q_init=end_joint[:6])
        if q is None:
            self.log.warn("lifted-standby IK failed; parking at home/idle pose")
            return self.idle_joint
        q = np.asarray(q, dtype=float)
        q[-1] = 0.0
        return q

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
        # Walk from the head (most downstream = most urgent) and commit to the
        # FIRST object the next pick can actually complete. earliest_reachable_intercept
        # does the same dynamic-Y feasibility as _select_ambush_target, but with
        # pre_delay=throw_time (the arm only starts moving after this throw finishes),
        # so "catchable" already accounts for the throw + move. The chosen intercept
        # is judged ONCE here and stored (committed_intercept) for the next selection
        # to reuse — no re-judgment.
        for cand in list(self.queue._objects):
            # GATE BEFORE judging catchability: if the most-downstream remaining
            # candidate routes to a DIFFERENT skill (e.g. a can -> push), STOP the
            # chain here. This must come BEFORE the catchability test: judging the can
            # with its (larger) push budget + pre_delay=throw_time can mark it
            # uncatchable-after-throw and `continue` PAST it, committing an upstream
            # PET — which then bypasses the can next epoch via the committed path and
            # STARVES it, even though the IMMEDIATE pick (pre_delay=0) might still
            # catch it. Stopping first guarantees the can is re-evaluated fresh next
            # epoch. (This chain only ever pre-positions a THROW grasp; a non-throw
            # next does its OWN skill's approach fresh — one set per skill.)
            if self.skill_for(cand) != "throw":
                self.log.info(
                    f"Chain stop: next id={cand.track_id} {cand.class_name} routes to "
                    f"'{self.skill_for(cand)}', not throw — no throw pre-position"
                )
                return None, None
            # cand routes to throw here; judge catchability with its timeline (== the
            # legacy budget, since throw doesn't override t_to_contact).
            cand_skill = self.skill_obj_for(cand)
            it = self.earliest_reachable_intercept(
                cand, from_joint, v, now, pre_delay=throw_time,
                t_to_contact_fn=cand_skill.t_to_contact,
            )
            if it is None:
                self.log.info(
                    f"Chain skip id={cand.track_id} {cand.class_name}: not catchable "
                    f"after throw ({throw_time:.2f}s) + move"
                )
                continue
            # Prime the NEXT object's suction during the return CHAIN, NEVER during
            # this throw. Catching the next object upstream (+y_b) makes its eta
            # small, which would otherwise fire the prime mid-throw and RE-GRAB the
            # object we just released (it'd be carried to the next pick). Clamp to
            # >= throw end so the prime always lands after the release, on the chain.
            suction_on_at = max(
                now + it.eta - self.cfg.SUCTION_LEAD, now + throw_time,
            )
            self.committed_next = cand
            self.committed_intercept = it
            self.log.info(
                f"Chain target: id={cand.track_id} {cand.class_name} at "
                f"x={float(it.T_grasp[0, 3]):+.3f} y={it.intercept_y:+.3f} "
                f"(prime in {max(0.0, suction_on_at - now):.2f}s, arrival {it.eta:.2f}s)"
            )
            return it.grasp_joint, suction_on_at
        return None, None

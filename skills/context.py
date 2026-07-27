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
    # intercept solver query that skill's intercept_time_budget() timeline — push and
    # throw have different positioning/contact costs, so the grasp must be placed
    # using the skill that will actually run. Wired in GP8App._build_skills.
    skill_obj_for: Callable[["TrackedObject"], "ManipulationSkill"]
    # Default standby pose (6-DOF joint vector, wrist/j6 = 0) the post-action
    # chain returns to when there is NO next object to pre-position. Computed once
    # in GP8App._build_skills from cfg.INITIAL_R/T (the boot pose). Skills reach it
    # via ManipulationSkill.idle_target() — see base.py. Returning a TARGET (not a
    # motion) keeps the return folded into each skill's single chained trajectory,
    # so no extra dispatch (which would chop the swing).
    idle_joint: np.ndarray

    @property
    def log(self):
        return self.node.get_logger()

    # ------------------------------------------------------------------
    # Shared motion / timing primitives
    # ------------------------------------------------------------------
    def object_y_now(self, target: "TrackedObject", now: float, v: float) -> float:
        """Object's belt-frame Y at ``now`` (belt travels -Y, so Y decreases).

        Use the conveyor encoder speed consistently for dead reckoning, intercept
        planning, and arrival waits.  A per-object fitted speed was introduced to
        compensate for encoder scale error / objects drifting on the belt, but the
        HW logs measured transparent objects at the acceptance-floor (~25% slower)
        and accumulated 40--50 cm of false position error after camera visibility
        ended.  Keep the old selection below as a commented record of that intent.
        """
        # Previous intent: why use this object's camera-fitted speed instead of the
        # encoder?  It was meant to follow rolling/slipping objects independently.
        # v_obj = target.v_est if getattr(target, "v_est", None) is not None else v
        # Current policy: one belt speed for the complete timing chain, matching the
        # last committed behaviour and avoiding mixed v_est/belt ETA calculations.
        v_obj = v
        return target.T_grasp_base[1, 3] - v_obj * (now - target.detect_time)

    def log_action_timing(self, target: "TrackedObject", intercept_y: float, tag: str) -> None:
        """DIAGNOSTIC: object position vs the intercept at the instant the action fires.

        ``delta = obj_y - intercept_y``. The belt travels -Y, so ``delta > 0`` means the
        object is still UPSTREAM of the intercept (approaching — good, we act as it
        arrives); ``delta < 0`` means it has ALREADY PASSED the intercept and the action
        lands BEHIND it — the multi-object "석션/push가 물체 지나간 자리에서 일어난다"
        symptom. ``age`` is how long the object has been dead-reckoned since its last
        detection (large for 2nd+ objects that coasted through the previous cycle).
        Called by each skill right before it commits (throw lift / push stroke)."""
        now = time.time()
        v = self.conveyor.current if self.conveyor is not None else 0.0
        obj_y = self.object_y_now(target, now, v)
        delta = obj_y - intercept_y
        state = "approaching(+)" if delta >= 0.0 else "PASSED-BEHIND(-)"
        self.log.info(
            f"[fire-timing] {tag} id={target.track_id} {target.class_name}: "
            f"obj_y={obj_y:+.3f} intercept_y={intercept_y:+.3f} delta={delta:+.3f}m "
            f"{state} age={now - target.detect_time:.2f}s belt={v:.3f}m/s"
        )

    def log_suction_on(self, target: "TrackedObject") -> None:
        """DIAGNOSTIC: at the instant suction turned ON, log the object's CALCULATED
        (belt-extrapolated) position vs the end-effector's ACTUAL position.

        The object position is ``object_y_now`` evaluated at ``last_suction_on_t``
        (the exact fire instant), with its lane X/Z; the EE position is the forward
        kinematics of ``last_suction_on_joints`` (the joint state snapshotted the
        moment the vacuum fired, possibly mid-move). ``dY = obj_y - ee_y`` is how far
        along the belt the object is from the cup when suction fires: ``>0`` the
        object is still upstream of the cup, ``<0`` it has already passed it."""
        tc = self.traj_ctrl
        t_son = getattr(tc, "last_suction_on_t", None)
        q_son = getattr(tc, "last_suction_on_joints", None)
        if t_son is None or q_son is None:
            self.log.warn("[suction-on] no suction-on snapshot available (joints/time None)")
            return
        v = self.conveyor.current if self.conveyor is not None else 0.0
        obj_x = float(target.T_grasp_base[0, 3])
        obj_y = self.object_y_now(target, t_son, v)
        obj_z = float(target.T_grasp_base[2, 3])
        try:
            T_ee = self.robot.forward_kinematics(np.asarray(q_son, dtype=float)[:6])
            ee_x, ee_y, ee_z = float(T_ee[0, 3]), float(T_ee[1, 3]), float(T_ee[2, 3])
        except Exception as e:  # noqa: BLE001 — diagnostic must never kill the cycle
            self.log.warn(f"[suction-on] EE FK failed: {e}")
            return
        self.log.info(
            f"[suction-on] id={getattr(target, 'track_id', '?')} "
            f"obj=({obj_x:+.3f},{obj_y:+.3f},{obj_z:+.3f}) "
            f"ee=({ee_x:+.3f},{ee_y:+.3f},{ee_z:+.3f}) "
            f"dX={obj_x - ee_x:+.3f} dY={obj_y - ee_y:+.3f} dZ={obj_z - ee_z:+.3f} "
            f"belt={v:.3f} age={t_son - target.detect_time:.2f}s"
        )

    def earliest_reachable_intercept(
        self,
        target: "TrackedObject",
        current_joint: np.ndarray,
        v: float,
        now: float,
        pre_delay: float = 0.0,
        skill: "Optional[ManipulationSkill]" = None,
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
        # Previous intent: use a camera-fitted per-object speed for position,
        # projected contact, and ETA so a rolling/slipping object can differ from
        # the conveyor.  HW logs showed that estimate causing large stale-track
        # errors, so retain the original line only as documentation.
        # v_obj = target.v_est if getattr(target, "v_est", None) is not None else v
        # Current policy: use the conveyor speed consistently, as object_y_now()
        # and position_and_prime() do.
        v_obj = v
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
            # Wrist baseline: J6 is a free DOF for pick/throw (symmetric cup,
            # 5-DOF throw NN), so park it where PUSH will want it — the
            # push-facing mean (cfg.PICK_WRIST_J6, was 0) — to kill the
            # ~90 deg J6 round-trips between cycles.
            gj = np.asarray(gj, dtype=float); gj[-1] = cfg.PICK_WRIST_J6
            aj = np.asarray(aj, dtype=float); aj[-1] = cfg.PICK_WRIST_J6
            move_time = (
                opt_time(current_joint, zero, aj, zero, self.M1, self.M2)
                + opt_time(aj, zero, gj, zero, self.M1, self.M2)
            )
            # Where the object will be once the arm reaches CONTACT (after pre_delay).
            # The contact budget comes from the skill that will RUN this object,
            # queried with the candidate's concrete geometry (intercept_time_budget:
            # push prices its real backswing route + run-up; throw's base default
            # keeps the legacy opt_time*factor proxy). No skill → legacy proxy.
            budget = (
                skill.intercept_time_budget(
                    target, current_joint, T_aim, T_grasp, aj, gj, move_time,
                ) if skill is not None
                else move_time * factor
            )
            obj_y_arrival = obj_y - v_obj * (pre_delay + budget)
            if obj_y_arrival < -y_b:
                return None                      # exits downstream before the arm arrives
            target_y = min(obj_y_arrival, y_b)   # wait at the entry edge if still upstream
            if abs(target_y - y_eval) < _INTERCEPT_TOL:
                break
            y_guess = (1.0 - _INTERCEPT_RELAX) * y_eval + _INTERCEPT_RELAX * target_y

        eta = (obj_y - y_eval) / (v_obj + 1e-6)
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
        # DIAGNOSTIC: wall-clock positioning duration (this call blocks for the
        # whole move). Push's wait_for_arrival re-checks arrival AFTER this, so a
        # long positioning here is what makes the stroke land behind the object.
        _t_pos0 = time.time()
        self.traj_ctrl.send_trajectory_queue(traj, vel, ts, final_joint=grasp_joint)
        self.log.info(f"[positioning] move_through_via {time.time() - _t_pos0:.2f}s")

    def sleep_until(self, deadline: float) -> None:
        """Block until ``deadline`` (wall clock), staying responsive to shutdown.

        The background MTE keeps belt speed and detection snapshots fresh; this
        loop publishes viz state and — importantly — keeps ingesting new
        detections into the queue and updating its order. Otherwise objects that
        arrive on the belt during the wait are invisible to app.py until the
        current cycle's throw completes (~10 s later), often too late to catch.
        """
        while rclpy.ok() and time.time() < deadline:
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
        # DIAGNOSTIC: the object is already at/below the intercept when the WAIT
        # begins -> positioning (move_through_via) overran and the object reached
        # the intercept before the arm was ready; the action will land BEHIND it.
        if (obj_y - intercept_y) <= 0.0:
            self.log.warn(
                f"[late] id={getattr(target, 'track_id', '?')} object already "
                f"at/below intercept at wait entry: obj_y={obj_y:+.3f} "
                f"intercept_y={intercept_y:+.3f} dist={obj_y - intercept_y:+.3f}m "
                f"age={now - target.detect_time:.2f}s — arm not ready in time"
            )
        self.sleep_until(now + eta - offset)

    def position_and_prime(
        self,
        current_joint: np.ndarray,
        aim_joint: np.ndarray,
        grasp_joint: np.ndarray,
        target: "TrackedObject",
        intercept_y: float,
        start_lead: "Optional[float]" = None,
        prime_suction: bool = True,
    ) -> bool:
        """Drive to the grasp pose, then POSITION-PRIME suction: fire it only once
        the cup is PARKED at the grasp, capped at SUCTION_LEAD before arrival.

        Always returns True (kept as bool for the caller's signature).

        Restores the fix-throwing behaviour that the wall-clock timer regressed:
        suction is keyed to the arm actually REACHING the grasp (cup down at the
        object), never to a bare predicted-arrival clock. The vacuum fires at
        ``max(cup-parked, arrival - SUCTION_LEAD)``:
          * backed-up object (eta < SUCTION_LEAD): fires the instant the cup parks
            at the grasp — as early as physically possible, but NEVER while the cup
            is still descending mid-move (the old bug that fired with the cup high);
          * object with slack (eta >= SUCTION_LEAD, e.g. the first pick): fires
            SUCTION_LEAD before arrival, cup already parked.
        The adv4ncr 250 Hz stream (<10 ms command->motion) makes "fire when parked"
        land within a stream tick of the grasp. Returns once the object has reached
        the intercept (caller then lifts/throws).

        ``prime_suction=False`` skips the suction entirely (still parks + waits +
        returns at ``arrival - start_lead``). TRACK_DESCEND passes this because its
        wait pose is a HOVER above the object: priming at the hover would run the
        vacuum in air for ~SUCTION_LEAD before the descend starts (a stationary
        suction-on gap). That caller fires suction as the descend begins instead.
        """
        now = time.time()
        v = self.conveyor.current
        obj_y = self.object_y_now(target, now, v)
        eta = max(0.0, min((obj_y - intercept_y) / (v + 1e-6), self.cfg.AMBUSH_MAX_WAIT))
        t_arrival = now + eta
        self.log.info(
            f"Ambush: cap prime {self.cfg.SUCTION_LEAD:.2f}s before arrival, "
            f"arrival/lift in {eta:.2f}s (dist {obj_y - intercept_y:.3f} m / "
            f"belt {v:.3f} m/s)"
        )

        # 1) Drive to the grasp pose — NO suction during the move. send_trajectory_queue
        #    blocks for the whole move (stream paces it in real time), so the cup is
        #    parked AT the grasp when it returns.
        zero = np.zeros_like(self.M1)
        traj, vel, ts = trajectory(
            current_joint, zero, grasp_joint, zero,
            self.M1, self.M2, hertz=self.cfg.TRAJ_HZ,
        )
        _t_pos0 = time.time()
        self.traj_ctrl.send_trajectory_queue(traj, vel, ts, final_joint=grasp_joint)
        _pos_dur = time.time() - _t_pos0
        # DIAGNOSTIC: did positioning finish before the object reaches the intercept?
        # If it exceeds eta, the object arrives before the cup parks and the pick
        # lands behind it (the throughput-limited case).
        if _pos_dur > eta:
            self.log.warn(
                f"[positioning-overrun] id={getattr(target, 'track_id', '?')} "
                f"positioning {_pos_dur:.2f}s > eta {eta:.2f}s: object reaches "
                f"intercept before the cup parks; action will land behind"
            )
        else:
            self.log.info(
                f"[positioning] {_pos_dur:.2f}s (eta {eta:.2f}s, "
                f"margin {eta - _pos_dur:.2f}s)"
            )

        # 2) Cup is parked at the grasp. Prime at max(now, arrival - SUCTION_LEAD):
        #    never before the cup is down (now = just parked), never more than
        #    SUCTION_LEAD early. Backed-up object -> ~now; slack pick -> waits until
        #    SUCTION_LEAD before arrival.
        #    prime_suction=False (TRACK_DESCEND): the wait pose is a HOVER above the
        #    object, not the grasp, so priming here would run the vacuum in air for
        #    ~SUCTION_LEAD before the descend even starts (the stationary suction-on
        #    gap). That caller instead fires suction AS the descend begins.
        self.set_status("WAITING", getattr(target, "class_name", ""))
        if prime_suction:
            t_suction = max(time.time(), t_arrival - self.cfg.SUCTION_LEAD)
            self.sleep_until(t_suction)
            self.traj_ctrl.suction_on()
            # DIAGNOSTIC: object's calculated position vs the EE's actual position at
            # the instant suction fired (see log_suction_on).
            self.log_suction_on(target)

        # 3) End the wait `start_lead` s before predicted arrival so the post-wait
        #    trajectory dispatch overlaps the object's final approach and the lift
        #    lands ON arrival. start_lead defaults to cfg.ACTION_START_LEAD; the
        #    skill passes its own arrival_lead().
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
        q[-1] = self.cfg.PICK_WRIST_J6   # shared wrist baseline (see config)
        return q

    def next_chain_target(self, from_joint: np.ndarray, action_time: float):
        """Grasp joints of the next object the arm should head toward AFTER the current
        action, so the follow-through chain OVERLAPS the next approach (recovers the
        throughput the plain-standby park lost). Reuses ``earliest_reachable_intercept``
        with ``pre_delay=action_time`` (the arm frees up only after this action) — its
        fixed-point loop resolves the move-time <-> object-position circularity. Returns
        ``(grasp_joint, candidate)`` for the first feasible object (wrist at
        cfg.PICK_WRIST_J6), or ``None`` (no next / none catchable) so the caller parks at
        lifted_standby. The candidate lets the caller pick a chain WRIST for the next
        object's skill (push keeps its push-facing wrist). Works for ANY next skill
        (symmetric). NO commitment: the next epoch still SELECTS + DRIVES fresh from this
        closer pose, so the handoff stays stateless (no committed/prepositioned/skip_move).
        Candidates the skill's ``placement_veto`` refuses are skipped, mirroring the
        selection walk — the chain never parks at a backswing that won't be swung."""
        now = time.time()
        v = self.conveyor.current if self.conveyor is not None else 0.0
        # No queue.update() here — earliest_reachable_intercept computes each object's
        # position from object_y_now itself, and mutating the queue mid-chain (pruning)
        # is a side effect the next epoch's own update should own.
        for cand in list(self.queue._objects):
            skill = self.skill_obj_for(cand)
            it = self.earliest_reachable_intercept(
                cand, from_joint, v, now, pre_delay=action_time,
                skill=skill,
            )
            if it is None:
                continue
            # Same gate the selection walk applies (placement_veto): don't
            # pre-position the chain at a backswing selection is already known
            # to refuse — without this the arm parks at a soon-to-be-vetoed
            # near-base metal and waits there without ever swinging. Silent
            # skip: selection owns the once-per-track [<skill>-veto] log line,
            # and the veto stays re-evaluated (class re-votes can un-veto).
            if skill.placement_veto(cand, it) is not None:
                continue
            self.log.info(
                f"Chain toward next: id={cand.track_id} {cand.class_name} @ "
                f"y={it.intercept_y:+.3f} (arrival {it.eta:.2f}s, pre_delay {action_time:.2f}s)"
            )
            return it.grasp_joint, cand
        return None

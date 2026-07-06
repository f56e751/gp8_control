"""GP8 pick-and-throw main application (ROS 2).

Ported from ROS 1 ``my_gp8_control/src/main_sam7.py`` (류가은 branch)
but running on MotoROS2 + name bridge + ROS 2 Humble. The orchestrator
itself stays small — domain logic lives in ``perception/``,
``tracking/``, ``planning/``, ``controllers/``, and the manipulation
``skills/`` (throw, push). GP8App selects a target each epoch, asks the
``ActionSelector`` which skill should handle it, and lets that skill run.
"""

from __future__ import annotations

import sys
import threading
import time
import json

from std_msgs.msg import String

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from gp8_control.controllers.trajectory_controller import TrajectoryController
from gp8_control.controllers.moveit_controller import MoveItController
from gp8_control.conveyor import ConveyorSpeedTracker
from gp8_control.perception.detection_intake import DetectionIntake
from gp8_control.trajectory.trajectory_primitive import trajectory
from gp8_control.trajectory.predictor import TrajectoryPredictor
from gp8_control.tracking import (
    TrackedObject,
    TrackedObjectQueue,
    FrameGate,
)
from gp8_control.config import Config
from gp8_control.planning import (
    PickThrowPlanner,
)
from gp8_control.planning.action_selector import ActionSelector
from gp8_control.robots.gp8 import GP8
from gp8_control.skills import (
    SkillContext,
    PickRequest,
    ThrowSkill,
    PushSkill,
)


def _make_transform(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float).ravel()
    return T


# =========================================================================
# Main application
# =========================================================================

class GP8App:
    """Pick-and-throw orchestrator with main_sam7-style tracking.

    Keeps only orchestration: ROS wiring, perception/tracking intake, target
    selection, and the main loop. The manipulation itself lives in
    ``gp8_control.skills`` — each epoch GP8App selects a target, asks the
    ``ActionSelector`` which skill handles it, and runs that skill.
    """

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()
        self.robot = GP8()

        # Belt-state viz publishing (see _publish_belt_state).
        self._status: str = "IDLE"
        self._status_detail: str = ""
        self._belt_state_pub = None
        # The target currently being executed (popped from the queue) — kept
        # here so the viz can keep drawing it while the cycle runs.
        self._active_target: TrackedObject | None = None

        # Latest corrected detection snapshot from the camera_debug node.
        # See /camera_debug/detections — camera_debug owns the perception
        # stream + cam→base transform + v*delay back-projection.
        self._cam_latest: dict | None = None

        self._node: Node | None = None
        self._executor: MultiThreadedExecutor | None = None
        self._spin_thread: threading.Thread | None = None
        self._spinner_dead = False
        self.traj_ctrl: TrajectoryController | None = None
        self.moveit_ctrl: MoveItController | None = None
        self.conveyor: ConveyorSpeedTracker | None = None

        self.M1: np.ndarray | None = None
        self.M2: np.ndarray | None = None
        self.predictor: TrajectoryPredictor | None = None
        self.planner: PickThrowPlanner | None = None

        # Manipulation skills + their shared context + the push/throw selector.
        # Built in setup() once the robot resources exist (see _build_skills).
        self.ctx: SkillContext | None = None
        self.throw_skill: ThrowSkill | None = None
        self.push_skill: PushSkill | None = None
        self.selector: ActionSelector | None = None

        # Coarse drop: only when an object's extrapolated y has fallen below the
        # WORST-CASE downstream reach edge (-MAX_REACH, the centerline lane). The
        # precise per-object "still catchable?" test is done by
        # SkillContext.earliest_reachable_intercept (using the lane-specific -y_b);
        # this queue prefilter just keeps the head meaningful without prematurely
        # dropping downstream-but-reachable objects.
        self.queue = TrackedObjectQueue(
            self.cfg.MAX_REACH, drop_below_y=-self.cfg.MAX_REACH,
        )
        self.frame_gate = FrameGate(self.cfg.FRAME_COOLDOWN_DISTANCE)
        self.detection_intake = DetectionIntake(self.cfg.OBJECT_MATCH_EPSILON)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup(self) -> None:
        rclpy.init()
        self._node = Node("gp8_manager")
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self._node)

        # Live belt-state stream for the belt_viz TUI (separate node).
        self._belt_state_pub = self._node.create_publisher(
            String, "/gp8_manager/tracked_state", 10
        )

        self.traj_ctrl = TrajectoryController(self._node)
        self.moveit_ctrl = MoveItController(self._node)
        # camera_debug node owns the perception stream + corrections; we just
        # subscribe to its corrected detection list.
        self._node.create_subscription(
            String, "/camera_debug/detections",
            self._on_camera_debug_detections, 10,
        )
        self.conveyor = ConveyorSpeedTracker(
            self._node,
            self.cfg.CONVEYOR_TOPIC,
            self.cfg.CONVEYOR_SPEED,
            self.cfg.CONVEYOR_STALE_SECONDS,
        )

        self.traj_ctrl.wait_for_servers()
        # Now that every subscription + service/action client exists and servers
        # are up, background-spin the executor: from here it is the SOLE spinner.
        # It pumps every subscription (joint_states, detections, conveyor) AND
        # completes every service/action future, so the main loop (and later the
        # feeder thread) only ISSUE call_async + block on _wait_future — no thread
        # calls rclpy.spin_* directly, eliminating the wait-set race. Started AFTER
        # entity creation (cf. gui/server.py) so it never races wait-set builds; the
        # ROS work below (_enable_robot / suction / initial pose) needs it live.
        self._spin_thread = threading.Thread(
            target=self._spin_executor, name="gp8_mte_spin", daemon=True)
        self._spin_thread.start()
        self._load_predictor()
        self._setup_joint_limits()
        self._build_planner()
        self._build_skills()
        self._enable_robot()
        # Always start from a known-off suction state.
        self._node.get_logger().info("Forcing suction OFF at startup.")
        self.traj_ctrl.suction_off()
        self._move_to_initial_pose()
        # main_sam7 trusts the hardcoded extrinsic. Perception now runs on the
        # camera PC (HTTP stream); no local AprilTag/RealSense handshake here.
        self._node.get_logger().info(
            "Using hardcoded T_base2cam (no AprilTag handshake)."
        )

    def _spin_executor(self) -> None:
        """Background MTE spin target, GUARDED: it is the SOLE spinner, so if an
        unhandled callback exception killed it silently, every _wait_future would
        hang. On death, flag it and shut the context down so no-timeout waits
        (which poll rclpy.ok()) unblock and the main loop exits into its finally."""
        try:
            self._executor.spin()
        except Exception as e:  # noqa: BLE001 — sole spinner must not die silently
            self._spinner_dead = True
            try:
                self._node.get_logger().fatal(f"MTE spinner died: {e}; shutting down.")
            except Exception:
                pass
            try:
                rclpy.try_shutdown()
            except Exception:
                pass

    def _load_predictor(self) -> None:
        self.predictor = TrajectoryPredictor()
        self._node.get_logger().info(
            f"Trajectory NN loaded: {self.predictor.weight_path}"
        )

    def _setup_joint_limits(self) -> None:
        # GP8.velocity_limits is a numpy array of max joint speeds (rad/s)
        self.M1 = np.asarray(self.robot.velocity_limits, dtype=float) * self.cfg.JOINT_VEL_LIMIT_SCALE
        self.M2 = self.M1 * self.cfg.JOINT_ACCEL_LIMIT_SCALE

    def _build_planner(self) -> None:
        self.planner = PickThrowPlanner(
            robot=self.robot,
            predictor=self.predictor,
            M1=self.M1,
            M2=self.M2,
            max_reach=self.cfg.MAX_REACH,
            target_distance=self.cfg.TARGET_DISTANCE,
            decoding=self.cfg.throw_decoding(),
            max_pick_lead=self.cfg.MAX_PICK_LEAD,
        )

    def _build_skills(self) -> None:
        """Wire the shared SkillContext, the manipulation skills, and the selector.

        GP8App holds only orchestration; each manipulation lives in its own
        skill module under ``gp8_control.skills``. Adding e.g. PushSkill is a
        new file plus one entry in the selector list here — no other app.py
        changes — which keeps push/throw work from colliding.
        """
        # Default standby pose for the idle chain: the same boot/initial pose
        # _move_to_initial_pose uses (cfg.INITIAL_R/T), wrist at PICK_WRIST_J6. Computed
        # once here so a skill's post-action chain can return to it when no next
        # object is queued (ManipulationSkill.idle_target -> ctx.idle_joint),
        # instead of parking low at the belt (e.g. a push's push_end).
        idle_T = _make_transform(self.cfg.INITIAL_R, self.cfg.INITIAL_T)
        idle_joint = self.robot.inverse_kinematics(idle_T)
        if idle_joint is None:
            raise RuntimeError("IK failed for idle/initial pose (cfg.INITIAL_R/T).")
        idle_joint = np.asarray(idle_joint, dtype=float)
        idle_joint[-1] = self.cfg.PICK_WRIST_J6   # shared wrist baseline

        self.ctx = SkillContext(
            cfg=self.cfg,
            node=self._node,
            robot=self.robot,
            traj_ctrl=self.traj_ctrl,
            planner=self.planner,
            conveyor=self.conveyor,
            queue=self.queue,
            M1=self.M1,
            M2=self.M2,
            intake=self._ingest_detections,
            publish_state=self._publish_belt_state,
            set_status=self._set_status,
            set_active_target=self._set_active_target,
            # Deferred: self.selector is built just below; the lambda is only
            # called at run time (after setup), by which point it is set.
            skill_for=lambda obj: self.selector.skill_for(obj),
            # Same deferred lookup, resolved to the skill OBJECT (name -> object via
            # the selector's skills map) so the intercept solver can use its timeline.
            skill_obj_for=lambda obj: self.selector.skills[self.selector.skill_for(obj)],
            idle_joint=idle_joint,
        )
        self.throw_skill = ThrowSkill(self.ctx)
        self.push_skill = PushSkill(self.ctx)
        # Rule-based class routing (SKILL_BY_CLASS): cans -> push, PET bottles ->
        # throw; anything else -> default "throw". Swap this for an RL policy
        # later by replacing ActionSelector. FORCE_SKILL (env GP8_FORCE_SKILL /
        # CLI --skill) pins every object to one skill for testing, overriding
        # the class routing; empty -> normal per-class routing.
        force = self.cfg.FORCE_SKILL or None
        if force is not None:
            self._node.get_logger().warn(
                f"ActionSelector FORCED to '{force}' skill for ALL objects "
                f"(GP8_FORCE_SKILL). Disable for normal push/throw routing."
            )
        else:
            self._node.get_logger().info(
                f"ActionSelector per-class routing: {self.cfg.SKILL_BY_CLASS} "
                f"(default 'throw')"
            )
        self.selector = ActionSelector(
            [self.throw_skill, self.push_skill],
            default="throw",
            by_class=self.cfg.SKILL_BY_CLASS,
            force=force,
        )

    def _enable_robot(self) -> None:
        """Ready the robot for motion (one-time; a no-op on the adv4ncr stream backend)."""
        self._node.get_logger().info("Enabling robot for motion...")
        if not self.traj_ctrl.enter_queue_mode():
            raise RuntimeError(
                "Failed to ready the robot. Check pendant is in REMOTE "
                "mode with no active alarm and cycle mode AUTO."
            )
        time.sleep(1.0)

    def _move_to_initial_pose(self) -> None:
        # ros2 launch ExecuteProcess does not forward stdin; only prompt
        # when running interactively.
        if sys.stdin.isatty():
            try:
                input("Press Enter to move to initial pose (Ctrl+C to abort).")
            except EOFError:
                pass
        else:
            self._node.get_logger().warn(
                "Auto-proceeding to initial pose in 3s (stdin not a TTY, "
                "likely running under ros2 launch). Ctrl+C to abort."
            )
            time.sleep(3.0)
        initial_T = _make_transform(self.cfg.INITIAL_R, self.cfg.INITIAL_T)
        initial_joint = self.robot.inverse_kinematics(initial_T)
        if initial_joint is None:
            raise RuntimeError("IK failed for initial pose.")

        # The background MTE fills current_joints; just poll it (no spin here).
        last_log = 0.0
        while (self.traj_ctrl.current_joints is None) and rclpy.ok():
            time.sleep(0.1)
            now = time.time()
            if now - last_log > 1.0:
                self._node.get_logger().info("Waiting for joint states...")
                last_log = now

        current_joint = np.array(self.traj_ctrl.current_joints)
        initial_joint = np.array(initial_joint)
        initial_joint[-1] = self.cfg.PICK_WRIST_J6   # shared wrist baseline
        zero = np.zeros_like(self.M1)
        traj, vel, timestep = trajectory(
            current_joint, zero,
            initial_joint, zero,
            self.M1, self.M2,
            hertz=self.cfg.TRAJ_HZ,
        )
        self.traj_ctrl.send_trajectory_queue(
            traj, vel, timestep, final_joint=initial_joint
        )
        time.sleep(1.0)

    # ------------------------------------------------------------------
    # Target selection (ambush strategy)
    # ------------------------------------------------------------------
    def _select_ambush_target(
        self, now: float, current_joint: np.ndarray
    ) -> PickRequest | None:
        """Choose the most-downstream catchable object and its EARLIEST intercept.

        Walks the queue from the head (most downstream = most urgent). For each,
        ``ctx.earliest_reachable_intercept`` returns the dynamic intercept — the
        earliest belt-Y the arm can grab at (it moves along the belt to meet the
        object soonest, and will catch it downstream of the old y=0 line if that
        is where it can still reach it) — or ``None`` when the object can't be
        caught anywhere in the workspace before it passes the downstream reach
        edge. ``None`` is the only drop reason. Returns a ``PickRequest`` for the
        first catchable head, or ``None`` if none is catchable this epoch. The
        selected head is popped and recorded as the active target.
        """
        v = self.conveyor.current

        # Walk from the head (most downstream = most urgent). Take the FIRST object
        # the arm can still catch in its workspace: earliest_reachable_intercept
        # returns the dynamic intercept (grab at the EARLIEST reachable belt-Y, even
        # downstream of y=0) or None when the object can't be caught before it passes
        # the downstream reach edge. None is the ONLY drop reason now.
        target = None
        target_it = None
        while self.queue:
            candidate = self.queue.head()
            # Place/judge the intercept with the timeline of the skill that will
            # ACTUALLY run this object (push has a much larger time-to-contact than
            # throw). Routing is geometry-independent, so resolving it here matches
            # the skill selected later at run_epoch.
            skill = self.ctx.skill_obj_for(candidate)
            it = self.ctx.earliest_reachable_intercept(
                candidate, current_joint, v, now,
                t_to_contact_fn=skill.t_to_contact,
            )
            if it is None:
                self.queue.pop_head()
                self._node.get_logger().info(
                    f"Drop id={candidate.track_id} {candidate.class_name} "
                    f"(conf {candidate.conf:.2f}): uncatchable in workspace "
                    f"(out of reach, or passes downstream before the arm arrives)"
                )
                continue
            target, target_it = candidate, it
            break

        if target is None:
            return None  # no catchable head in the queue this epoch

        # Commit to the pick.
        secondary = self.queue.peek_next() if self.queue.has_next() else None
        self.queue.pop_head()
        # Keep the active target visible in belt_viz while we execute the cycle.
        self._active_target = target
        # DIAGNOSTIC: where the object actually is NOW (obj_y) vs the entry-edge
        # reach limit (y_b) and how long it has been dead-reckoned (age). A 2nd+
        # object that coasted through the previous cycle shows obj_y already BELOW
        # y_b (past the entry edge) with a large age and a small eta — i.e. it is
        # engaged late. The intercept (y=) is where the arm will wait.
        obj_y_sel = self.ctx.object_y_now(target, now, v)
        x_sel = float(target.T_grasp_base[0, 3])
        _denom = self.cfg.MAX_REACH ** 2 - x_sel ** 2
        y_b_sel = float(np.sqrt(_denom)) if _denom > 0.0 else float("nan")
        self._node.get_logger().info(
            f"Ambush lock: id={target.track_id} {target.class_name} "
            f"(conf {target.conf:.2f}) @ x={target_it.T_grasp[0, 3]:+.3f} "
            f"y={target_it.intercept_y:+.3f} z={target_it.T_grasp[2, 3]:+.3f} "
            f"(eta {target_it.eta:.2f}s, move {target_it.move_time:.2f}s) "
            f"obj_y={obj_y_sel:+.3f} y_b={y_b_sel:+.3f} "
            f"age={now - target.detect_time:.2f}s"
        )
        # DIAGNOSTIC: full queue snapshot at lock (active target + everything still
        # queued), so a GHOST track (a spurious [track-NEW] for an already-thrown
        # object) or a mis-sorted head is obvious across a multi-object run.
        _q_dump = ", ".join(
            f"#{o.track_id}:{o.class_name}:y{self.ctx.object_y_now(o, now, v):+.2f}"
            for o in list(self.queue._objects)
        ) or "(empty)"
        self._node.get_logger().info(
            f"[queue] locked #{target.track_id}; remaining=[{_q_dump}]"
        )
        return PickRequest(
            target=target,
            current_joint=current_joint,
            T_aim=target_it.T_aim,
            T_grasp=target_it.T_grasp,
            aim_joint=target_it.aim_joint,
            grasp_joint=target_it.grasp_joint,
            secondary=secondary,
        )

    # ------------------------------------------------------------------
    # Belt-state viz (publishes to /gp8_manager/tracked_state for belt_viz)
    # ------------------------------------------------------------------
    def _set_active_target(self, target: TrackedObject | None) -> None:
        """Set the in-flight target (kept for belt_viz + intake dedup).

        Wired into the SkillContext so a skill can clear it when its cycle
        finishes without reaching back into GP8App.
        """
        self._active_target = target

    def _set_status(self, status: str, detail: str = "") -> None:
        self._status = status
        self._status_detail = detail
        self._publish_belt_state()

    def _publish_belt_state(self) -> None:
        if self._belt_state_pub is None:
            return
        now = time.time()
        v = self.conveyor.current if self.conveyor is not None else 0.0
        objs = []

        def _serialize(obj: TrackedObject, is_target: bool) -> dict:
            y_now = float(obj.T_grasp_base[1, 3] - v * (now - obj.detect_time))
            cam = obj.cam_pos
            return {
                "class": obj.class_name,
                "y_now": y_now,
                "x": float(obj.T_grasp_base[0, 3]),
                "z": float(obj.T_grasp_base[2, 3]),
                "age_s": float(now - obj.detect_time),
                "is_target": is_target,
                "cam": list(cam) if cam is not None else None,
            }

        # Currently-executing target (popped from the queue but still on the belt).
        if self._active_target is not None:
            objs.append(_serialize(self._active_target, True))

        q = getattr(self, "queue", None)
        if q is not None:
            for obj in list(q._objects):
                objs.append(_serialize(obj, False))
        state = {
            "ts": now,
            "belt_mps": float(v),
            "intercept_y": float(self.cfg.GRASP_INTERCEPT_Y),
            "max_reach": float(self.cfg.MAX_REACH),
            "status": self._status,
            "status_detail": self._status_detail,
            "objects": objs,
        }
        try:
            self._belt_state_pub.publish(String(data=json.dumps(state)))
        except Exception:
            pass  # never let viz publishing kill the control loop

    # ------------------------------------------------------------------
    # Epoch stages
    # ------------------------------------------------------------------
    def _ingest_detections(self, now: float) -> None:
        """Fold the latest camera_debug snapshot into the queue (spatial dedup).

        Thin wrapper around ``DetectionIntake.ingest`` (perception/): the
        detection→queue association lives there; the app keeps only the
        frame-gate bookkeeping keyed on whether anything new was added.
        """
        added = self.detection_intake.ingest(
            self._cam_latest, self.queue, self._active_target,
            self.conveyor.current, self._node.get_logger(),
        )
        if added:
            self.frame_gate.mark(now)  # kept for backward compat (queue-empty reset)

    def _on_camera_debug_detections(self, msg: String) -> None:
        try:
            self._cam_latest = json.loads(msg.data)
        except (ValueError, TypeError):
            pass

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run_epoch(self, epoch: int) -> None:
        # The background MTE keeps joint_states / detections / conveyor fresh
        # continuously now (no per-epoch spin). Yield briefly so idle epochs
        # don't hot-loop the main thread.
        time.sleep(0.001)

        current_joint_list = self.traj_ctrl.current_joints
        if current_joint_list is None:
            self._node.get_logger().warn("Joints not available yet.")
            time.sleep(self.cfg.TIME_STEP)
            return
        current_joint = np.array(current_joint_list)
        now = time.time()

        self.conveyor.check_freshness()
        self._publish_belt_state()                # live belt + queue snapshot

        self._ingest_detections(now)
        self.queue.update(now, self.conveyor.current)
        if not self.queue:
            self.frame_gate.reset()
            time.sleep(self.cfg.TIME_STEP)
            return

        request = self._select_ambush_target(now, current_joint)
        if request is not None:
            # Decide push vs throw (rule-based today; RL later) and run it.
            skill = self.selector.select(request)
            # DIAGNOSTIC: id + class -> skill in one line. A PET (transparent)
            # showing "-> push" here means its TRACK is labelled metal (latched or
            # ghost) — cross-check against [track-MATCH-MISMATCH] / [track-NEW].
            self._node.get_logger().info(
                f"Route id={request.target.track_id} {request.target.class_name} "
                f"(conf {request.target.conf:.2f}) -> {skill.name}"
            )
            # Tag this cycle's queued commands with the skill (push/throw) for the
            # diagnostic motion CSV (no-op unless GP8_MOTION_LOG_DIR is set).
            self.traj_ctrl.set_motion_op(skill.name)
            skill.execute(request)

    def run(self) -> None:
        self.setup()
        epoch = 0
        try:
            while rclpy.ok() and not self._spinner_dead:
                epoch += 1
                try:
                    self.run_epoch(epoch)
                except Exception as e:
                    self._node.get_logger().error(f"Epoch {epoch} failed: {e}")
                    time.sleep(0.1)
        finally:
            try:
                # queue mode를 걸어둔 채 종료하면 다음 실행 시 FJT가 reject됨
                self.traj_ctrl.exit_queue_mode()
            except Exception as e:
                self._node.get_logger().warn(f"exit_queue_mode failed: {e}")
            # Stop the background spinner before tearing down the node so spin()
            # returns and its daemon thread exits cleanly.
            if self._executor is not None:
                self._executor.shutdown()
            self._node.destroy_node()
            rclpy.shutdown()


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="gp8_app",
        description="GP8 conveyor pick-and-place orchestrator.",
    )
    parser.add_argument(
        "--skill",
        choices=["throw", "push"],
        default=None,
        help=(
            "Force every object to this manipulation skill (testing aid). "
            "Overrides the GP8_FORCE_SKILL env var. Omit for normal "
            "push/throw routing."
        ),
    )
    # parse_known_args so ROS 2 / ros2 launch-injected args (e.g. --ros-args)
    # pass through harmlessly instead of erroring out.
    args, _ = parser.parse_known_args(argv)

    cfg = Config()
    if args.skill is not None:
        cfg.FORCE_SKILL = args.skill   # CLI flag wins over the env default

    app = GP8App(cfg)
    app.run()


if __name__ == "__main__":
    main()

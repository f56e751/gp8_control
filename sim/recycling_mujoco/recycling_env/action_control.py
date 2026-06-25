from __future__ import annotations

import contextlib
import io
import time
from typing import Any, Callable

import numpy as np

from .trajectory import JointTrajectory

try:
    import torch
except ImportError:
    torch = None

try:
    from scipy.spatial.transform import Slerp
except ImportError:
    Slerp = None


class RecyclingActionMixin:
    """Robot action, trajectory, and physical conveyor behavior for the Gym wrapper."""

    ACTION_THROW = 0
    ACTION_PUSH = 1

    def _format_demo_action(self, object_index: int | None, primitive: int) -> dict[str, int | None]:
        return {"object_index": None if object_index is None else int(object_index), "primitive": int(primitive)}

    def _parse_action(
        self,
        action: int | np.integer | np.ndarray | tuple[int, int] | list[int] | dict[str, Any] | None,
    ) -> tuple[int | None, int]:
        """Decode legacy discrete actions or [object_index, primitive] Gym actions."""
        if action is None:
            return None, self.ACTION_THROW

        if isinstance(action, dict):
            action_index = action.get("object_index", action.get("index", action.get("action", None)))
            primitive = action.get("primitive", action.get("action_type", self.ACTION_THROW))
        else:
            action_array = np.asarray(action).reshape(-1)
            if action_array.size == 0:
                return None, self.ACTION_THROW
            action_index = action_array[0]
            primitive = action_array[1] if action_array.size > 1 else self.ACTION_THROW

        if isinstance(primitive, str):
            primitive_value = self.ACTION_PUSH if primitive.lower() == "push" else self.ACTION_THROW
        else:
            primitive_value = int(primitive)
        primitive_value = self.ACTION_PUSH if primitive_value == self.ACTION_PUSH else self.ACTION_THROW

        if action_index is None:
            return None, primitive_value
        return int(action_index), primitive_value

    def _action_primitive_name(self, primitive: int) -> str:
        return "push" if int(primitive) == self.ACTION_PUSH else "throw"

    def _ordered_active_slots(self) -> list[Any]:
        active_slots = [
            slot for slot in self.object_slots
            if slot.active and not getattr(slot, "manipulated", False) and not getattr(slot, "blocked", False) and slot.spawn_id is not None
        ]

        def _slot_priority(slot: Any) -> tuple[int, int, float, int]:
            try:
                obj_xyz = self._slot_grasp_throw_xyz(slot)
                obj_x = float(obj_xyz[0])
                obj_y = float(obj_xyz[1])
            except Exception:
                obj_x = float("inf")
                obj_y = float("-inf")
            actionable = int(obj_y > 0.05)
            can_enter_reach = int(abs(obj_x) <= (self.action_reachable_radius + 1e-6))
            return (-actionable, -can_enter_reach, obj_y if actionable else float("inf"), int(slot.spawn_id))

        return sorted(active_slots, key=_slot_priority)

    def _projected_obj_xyz_for_feasibility(self, slot: Any, horizon: float) -> np.ndarray | None:
        """Slot's projected obj position at expected manipulation-firing time.
        For a 1-step-ahead action, info["action_mask"] is consumed at the end of
        step N and the chosen action's manipulation fires at the START of step
        N+2 (step N+1 routes toward it). The intervening step's duration is the
        drift gap and depends on which primitive is being chained, so each
        primitive screens with its own horizon -- the THROW already absorbs
        the wait via `wait_time` in `_build_initial_approach_plan`, so its
        screening horizon is small/zero; the PUSH has no comparable in-builder
        absorption past the immediate strike, so its horizon is larger."""
        try:
            obj_xyz = self._slot_grasp_throw_xyz(slot).copy()
        except Exception:
            return None
        speed = float(getattr(self, "belt_speed", 0.0))
        if float(horizon) > 0.0 and speed > 0.0:
            obj_xyz[1] -= speed * float(horizon)
        return obj_xyz

    def _slot_cone_push_plan(self, slot: Any):
        """Hover-seeded cone push plan for `slot` -- seeded from its cone hover
        so the feasibility/duration estimate matches the pre-positioned clean
        strike the executor will run (not a chase from the current EE). Returns
        the cone-plan 4-tuple `(planned_obj_xyz, trajectory, push_motion,
        contact_time)` or None. SINGLE source of truth for the pending push's
        `last_trajectory`, used by both screening (`_slot_throw_feasible`
        push-pending branch) and demo selection (`_current_slot_cone_plan`
        delegates here)."""
        if self.pushing_trajectory_generator is None:
            return None
        obj_xyz = self._slot_grasp_throw_xyz(slot)
        if float(obj_xyz[1]) <= 0.05:
            return None
        target_xy = self._slot_target_xy(slot)
        cone = self._push_cone_target_y(obj_xyz, target_xy)
        if cone is None:
            return None
        cone_obj_xyz = obj_xyz.copy()
        cone_obj_xyz[1] = cone[0]
        hover_q = self._push_handoff_q_from_obj(cone_obj_xyz, target_xy=target_xy)
        if hover_q is None:
            return None
        start_world_z = self._slot_center_world_xyz(slot)[2]
        return self._build_cone_pushing_plan(
            hover_q,
            obj_xyz,
            start_world_z,
            target_xy=target_xy,
            object_speed=self._slot_forward_speed(slot),
        )

    def _slot_throw_feasible(
        self, slot: Any, base_horizon: float = 0.0, pending_push_traj: Any = None
    ) -> bool:
        """Single-slot throw feasibility (env contract): True means the executor
        will throw to EXACTLY this slot (no standby substitution). SCREENING ==
        EXECUTION per pending-context, mirroring the push consolidation:

          - Pending A is a THROW (chained throw->throw): A's throw trajectory
            ends at this slot's grasp, so screen with the SAME predictor the
            executor (`_execute_throw_transition`) and demo scoring
            (`_throw_transition_margin`) use -- `_predict_next_grasp_xyz(obj_A,
            slot_grasp, ...)`. It projects the candidate forward by A's throw
            duration internally (its iterated `trajectory_duration`), so the
            `base_horizon` projection is subsumed -- no double-count. Feasible
            iff it returns a prediction with no `neg_wait_time` (index 4), which
            is exactly the executor's "predicted_next_object_missed -> standby"
            trigger.
          - Pending A is a PUSH (push->throw): A's push after-trajectory routes
            the EE to this slot's grasp, so screen with the SAME placement
            predictor the executor + demo selection use -- `_build_pushing_after_trajectory`'s
            THROW branch on A's hover-seeded cone trajectory. Feasible iff it
            returns a NON-standby handoff (its internal y-window [0.08,0.45]
            reject == the chain breaking to standby). `pending_push_traj` (A's
            cone plan, computed once per mask step) is reused to avoid a rebuild
            per candidate; falls back to building it here.
          - No pending A (bootstrap/route-only): the EE reaches this slot's grasp
            via a bang-bang approach (`_execute_initial_approach`), so screen with
            `_build_initial_approach_plan` (projected by `base_horizon`)."""
        if not slot.active or getattr(slot, "manipulated", False):
            return False
        if getattr(slot, "is_obstacle", False):
            return False  # obstacle: belongs to no bin -> never a valid action

        pending_slot = getattr(self, "_pending_target_slot", None)
        pending_prim = getattr(self, "_pending_action_primitive", None)
        pending_active = (
            pending_slot is not None
            and pending_prim is not None
            and getattr(pending_slot, "active", False)
            and pending_slot is not slot
        )
        chained_from_throw = pending_active and int(pending_prim) == self.ACTION_THROW
        chained_from_push = pending_active and int(pending_prim) == self.ACTION_PUSH

        if chained_from_throw:
            try:
                obj_a = self._slot_grasp_throw_xyz(pending_slot)
                slot_grasp = self._slot_grasp_throw_xyz(slot)
                spd = self._slot_planning_speed(slot)
                pred = self._predict_next_grasp_xyz(
                    obj_a,
                    slot_grasp,
                    target_xy=self._slot_target_xy(pending_slot),
                    next_obj_speed=spd,
                    arrival_y_offset=self._throw_handoff_y_offset(spd),
                )
            except Exception:
                return False
            # pred[4] is neg_wait_time -- executor falls to standby when it is
            # not None (object arrives too early to catch).
            return pred is not None and pred[4] is None

        if chained_from_push:
            # Screen with the EXACT placement predictor the push->throw executor
            # + demo selection use: A's after-traj THROW branch on A's cone traj.
            try:
                push_plan = pending_push_traj
                if push_plan is None:
                    push_plan = self._slot_cone_push_plan(pending_slot)
                if push_plan is not None:
                    push_traj = push_plan[1]
                    after = self._build_pushing_after_trajectory(
                        push_traj,
                        slot,
                        current_slot=pending_slot,
                        next_primitive=self.ACTION_THROW,
                        # selection-time convention: A has not run yet, so project
                        # the candidate forward by A's push duration.
                        extra_projection_seconds=float(push_traj.duration),
                    )
                    return (
                        after is not None
                        and after[1] is not None
                        and not np.allclose(
                            after[1], self.action_single_object_standby_xyz, atol=1e-6
                        )
                    )
            except Exception:
                return False
            # Could not model A's push (cone plan failed) -> fall through to the
            # bang-bang approach proxy below.

        # Bootstrap / route-only (or push-pending fallback): bang-bang approach.
        obj_xyz = self._projected_obj_xyz_for_feasibility(slot, float(max(base_horizon, 0.0)))
        if obj_xyz is None:
            return False
        if float(obj_xyz[1]) <= 0.05:
            return False
        try:
            plan = self._build_initial_approach_plan(
                obj_xyz, object_speed=self._slot_execution_speed(slot)
            )
        except Exception:
            return False
        return plan is not None

    def _pending_manip_duration(self) -> float:
        """Duration of the currently-pending manipulation (object A) -- the time
        the robot is busy before it can manipulate a NEWLY chosen candidate.

        Mirrors the PyBullet pairwise scheme: the edge cost A->B projects B
        forward by A's manipulation `traj_time`. Here A is `_pending_target_slot`
        / `_pending_action_primitive` (the manipulation that fires next step;
        any candidate the mask admits fires the step AFTER, so it must wait for
        A to finish). A's after-trajectory already routes the EE to the
        candidate's hover, so this duration is the WHOLE 'go get B ready' term --
        there is no separate route leg. Returns 0.0 at route-only/idle steps
        (no pending A -> robot is ready now)."""
        slot = getattr(self, "_pending_target_slot", None)
        prim = getattr(self, "_pending_action_primitive", None)
        if slot is None or prim is None or not getattr(slot, "active", False):
            return 0.0
        try:
            obj_a = self._slot_grasp_throw_xyz(slot)
            target_a = self._slot_target_xy(slot)
            if int(prim) == self.ACTION_THROW:
                mr = self._throwing_model_parameters(
                    obj_a, self.action_single_object_standby_xyz, target_xy=target_a
                )
                return float(mr[1]) if mr is not None else 1.0
            # push: cone plan trajectory duration from current EE pose.
            q = self.env.get_joint_positions(self.joint_names).astype(np.float64)
            z = self._slot_center_world_xyz(slot)[2]
            plan = self._build_cone_pushing_plan(
                q, obj_a, z, target_xy=target_a,
                object_speed=self._slot_planning_speed(slot),
            )
            return float(plan[1].duration) if plan is not None else 1.0
        except Exception:
            return 1.0

    def _push_strike_from_hover_time(
        self, cone_obj_xyz: np.ndarray, target_xy: np.ndarray, hover_q: np.ndarray,
    ) -> float | None:
        """Time from the cone hover to first contact: bang-bang descent
        hover->push_start PLUS the start_margin line traverse. The push-specific
        extra on top of A's duration (the EE is already AT the hover when A's
        after-traj finishes; only the strike itself still has to happen)."""
        try:
            push_obj = np.asarray(cone_obj_xyz, dtype=np.float64).copy()
            push_obj[2] += float(self.pushing_params.get("height_offset", 0.0))
            push_dir, push_start, push_end, _hp = self.pushing_trajectory_generator.calculate_pushing_geometry(
                push_obj, target_xy, self.pushing_params,
            )
            t_start, _, _, _ = self.pushing_trajectory_generator.get_pushing_SE3_poses(
                push_start, push_end, push_dir=push_dir,
                swing_angle=float(self.pushing_params.get("swing_angle", 0.0)),
            )
            if t_start is None:
                return None
            with self._quiet_planner_output():
                q_push_start = self.robot_kinematics.inverse_kinematics_np_push(
                    t_start, seed=np.asarray(hover_q, dtype=np.float64),
                )
            if q_push_start is None:
                return None
            descend = self._opt_time_bangbang(hover_q, np.asarray(q_push_start, dtype=np.float64))
            margin = float(self.pushing_params.get("start_margin", 0.10)) / max(
                float(self.pushing_params.get("velocity", 1.4)), 1e-3
            )
            return float(descend + margin)
        except Exception:
            return None

    def _predict_push_hover(
        self,
        obj_now_xyz: np.ndarray,
        target_xy: np.ndarray,
        ref_q: np.ndarray,
        base_travel: float = 0.0,
        iterations: int = 3,
        skip_route: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Place the push HOVER (the previous trajectory's endpoint) above the
        push_start of where the object WILL BE at contact time -- the push analog
        of the throw's `_predict_next_grasp_xyz` (which ends the previous
        trajectory at the predicted grasp position).

        The object keeps drifting until the pusher actually contacts it, by:
          base_travel : extra time BEFORE the EE starts routing to the hover
                        (chained push, selection-time: the current push's
                         duration; route-only / execution-time: 0).
          t_route     : time for the EE to ROUTE from `ref_q` to the hover
                        (always measured: bang-bang ref_q -> hover_q).
          t_strike    : descent (hover->push_start) + start_margin traverse, i.e.
                        `_push_strike_from_hover_time` -- the one push-specific
                        term beyond the throw (the EE sits at the hover, then
                        still has to drop and sweep before it touches the object).
        We iterate the fixed point predicted_y = obj_y - speed*(base_travel +
        t_route + t_strike) (t_route/t_strike depend on the hover pose, which
        depends on predicted_y), clamp into the cone INTERSECT workspace band,
        and return the predicted cone object xyz + the hover IK. With the hover
        here, the strike is a clean vertical drop (no lateral lead -> no chase)
        that lands on the object's actual contact position (no under-projection
        -> no miss). Returns `(predicted_cone_obj_xyz, hover_q)` or None if
        un-pushable (object drifts below the reachable cone before contact, or
        IK fails)."""
        if self.pushing_trajectory_generator is None:
            return None
        obj_now_xyz = np.asarray(obj_now_xyz, dtype=np.float64)
        target_xy = np.asarray(target_xy, dtype=np.float64)
        ref_q = np.asarray(ref_q, dtype=np.float64)
        speed = float(max(self.belt_speed, 1e-6))
        base_travel = float(max(base_travel, 0.0))

        cone_obj = None
        hover_q = None
        t_route = 0.0
        t_strike = 0.0
        for _ in range(int(iterations)):
            horizon = base_travel + t_route + t_strike
            proj = obj_now_xyz.copy()
            proj[1] = float(obj_now_xyz[1]) - speed * horizon
            cone = self._push_cone_target_y(proj, target_xy)
            if cone is None:
                return None  # object will have drifted below the reachable cone
            cone_target_y, _eff_lower, _eff_upper = cone
            cone_obj = proj.copy()
            cone_obj[1] = cone_target_y
            try:
                hover_q = self._push_handoff_q_from_obj(cone_obj, target_xy=target_xy, seed=ref_q)
            except Exception:
                return None
            if hover_q is None:
                return None  # hover IK failed
            # skip_route: the caller's PREVIOUS trajectory already routes the EE
            # to this hover (throw->push: the throw trajectory IS the route), so
            # the horizon must be base_travel (that trajectory's duration) +
            # t_strike, with NO separate ref_q->hover_q leg. Default path keeps
            # measuring the route (route-only / push->push).
            t_route = 0.0 if skip_route else self._opt_time_bangbang(ref_q, hover_q)
            st = self._push_strike_from_hover_time(cone_obj, target_xy, hover_q)
            if st is None:
                return None  # strike push_start IK failed
            t_strike = st
        if cone_obj is None or hover_q is None:
            return None
        return cone_obj, hover_q

    def _slot_push_feasible(self, slot: Any, base_horizon: float = 0.0) -> bool:
        """Single-slot push feasibility (env contract). SCREENING IS the hover
        algorithm: the action is feasible iff `_predict_push_hover` can produce a
        valid (in-cone, reachable, IK-solvable) hover for the object's projected-
        at-contact position. There is NO separate feasibility predicate -- this
        delegates to the SAME `_predict_push_hover` that places the hover and
        whose result the strike consumes, so screening, placement, and execution
        agree by construction. `base_horizon` = the pending manipulation's
        duration (the candidate fires after A finishes); ref_q = current EE."""
        if not slot.active or getattr(slot, "manipulated", False):
            return False
        if getattr(slot, "is_obstacle", False):
            return False  # obstacle: belongs to no bin -> never a valid action
        if not getattr(self, "pushing_enabled", False) or self.pushing_trajectory_generator is None:
            return False
        try:
            obj_now = self._slot_grasp_throw_xyz(slot)
        except Exception:
            return False
        if float(obj_now[1]) <= 0.05:
            return False
        target_xy = self._slot_target_xy(slot)
        try:
            current_q = self.env.get_joint_positions(self.joint_names).astype(np.float64)
        except Exception:
            return False
        return self._predict_push_hover(
            obj_now, target_xy, current_q, base_travel=float(max(base_horizon, 0.0))
        ) is not None

    def _compute_slot_feasibility_flags(self) -> dict[int, dict[str, bool]]:
        """Per-slot per-primitive ACTION feasibility for the action-addressable
        prefix of `_ordered_active_slots()` (idx < max_detections). Returns
        `{idx: {"throw": bool, "push": bool}}`. The skip sentinel
        (idx == max_detections) is handled by the action-mask emitter.

        This is the SINGLE SOURCE OF TRUTH for "(slot, primitive) is feasible
        right now": demo policy and the env's `info["action_mask"]` both consult
        it, and the executors use the same builders -- so True is the env's
        contract that the action will run."""
        flags: dict[int, dict[str, bool]] = {}
        ordered = self._ordered_active_slots()
        addressable = min(len(ordered), int(self.max_detections))
        # A candidate chosen now fires only AFTER the pending manipulation (A)
        # finishes, so every candidate is projected forward by A's duration
        # (mirrors the PyBullet pairwise edge cost A->B). 0 at route-only/idle.
        base_horizon = self._pending_manip_duration()
        # push->throw screening reuses the pending push's hover-seeded cone
        # trajectory (identical for every candidate), so build it ONCE here
        # rather than per candidate inside `_slot_throw_feasible`.
        pending_push_traj = None
        pending_slot = getattr(self, "_pending_target_slot", None)
        pending_prim = getattr(self, "_pending_action_primitive", None)
        if (
            pending_slot is not None
            and pending_prim is not None
            and int(pending_prim) == self.ACTION_PUSH
            and getattr(pending_slot, "active", False)
        ):
            pending_push_traj = self._slot_cone_push_plan(pending_slot)
        for idx in range(addressable):
            slot = ordered[idx]
            flags[idx] = {
                "throw": self._slot_throw_feasible(
                    slot, base_horizon=base_horizon, pending_push_traj=pending_push_traj
                ),
                "push": self._slot_push_feasible(slot, base_horizon=base_horizon),
            }
        return flags

    def _world_to_throw_xyz(self, world_xyz: np.ndarray) -> np.ndarray:
        world_xyz = np.asarray(world_xyz, dtype=np.float64)
        return np.array(
            [
                world_xyz[0] - self.robot_base_xyz[0],
                world_xyz[1] - self.robot_base_xyz[1],
                world_xyz[2] - self.robot_base_xyz[2],
            ],
            dtype=np.float64,
        )

    def _slot_center_world_xyz(self, slot: Any) -> np.ndarray:
        pos, _ = self.env.get_freejoint_pose(slot.freejoint_name)
        return pos.astype(np.float64)

    def _slot_linear_velocity_xyz(self, slot: Any) -> np.ndarray:
        joint_id = self.env.require_joint(slot.freejoint_name)
        qvel_adr = self.env.model.jnt_dofadr[joint_id]
        return self.env.data.qvel[qvel_adr : qvel_adr + 3].copy().astype(np.float64)

    def _slot_forward_speed(self, slot: Any) -> float:
        velocity = self._slot_linear_velocity_xyz(slot)
        speed = float(-velocity[1])
        if not np.isfinite(speed):
            return 0.0
        return float(max(speed, 0.0))

    def _slot_planning_speed(self, slot: Any) -> float:
        measured_speed = self._slot_forward_speed(slot)
        nominal_speed = float(max(self.belt_speed, 1e-6))
        min_tracking_speed = 0.75 * nominal_speed
        speed = max(measured_speed, min_tracking_speed)
        # Cap at nominal belt_speed: fresh spawns have transient impulse
        # velocities (up to ~1.3x belt_speed) before settling. Using those in
        # the throw predictor over-projects where the obj will be and lands
        # the gripper too far downstream of where the obj actually ends up.
        return float(min(speed, nominal_speed))

    def _slot_has_close_lead_object(self, slot: Any, y_gap_threshold: float = 0.22, x_gap_threshold: float = 0.08) -> bool:
        slot_xyz = self._slot_grasp_throw_xyz(slot)
        slot_x = float(slot_xyz[0])
        slot_y = float(slot_xyz[1])
        for other in self.object_slots:
            if other is slot or not other.active or getattr(other, "manipulated", False):
                continue
            other_xyz = self._slot_grasp_throw_xyz(other)
            other_x = float(other_xyz[0])
            other_y = float(other_xyz[1])
            if other_y >= slot_y:
                continue
            if (slot_y - other_y) <= y_gap_threshold and abs(slot_x - other_x) <= x_gap_threshold:
                return True
        return False

    def _slot_execution_speed(self, slot: Any) -> float:
        measured_speed = self._slot_forward_speed(slot)
        nominal_speed = float(max(self.belt_speed, 1e-6))
        obj_y = float(self._slot_grasp_throw_xyz(slot)[1])
        if self._slot_has_close_lead_object(slot):
            min_execution_speed = 0.45 * nominal_speed if obj_y >= 0.8 else 0.05 * nominal_speed
        else:
            min_execution_speed = 0.5 * nominal_speed
        speed = max(measured_speed, min_execution_speed)
        # Cap at nominal belt_speed (see _slot_planning_speed for rationale).
        return float(min(speed, nominal_speed))

    def _quiet_planner_output(self) -> contextlib.AbstractContextManager[Any]:
        return contextlib.redirect_stdout(io.StringIO())

    def _slot_target_xy(self, slot: Any | None) -> np.ndarray:
        if slot is not None and getattr(slot, "target_xy", None) is not None:
            return np.asarray(slot.target_xy, dtype=np.float64)
        return np.asarray(self.target_xy, dtype=np.float64)

    def _set_pusher_collision_enabled(self, enabled: bool) -> None:
        geom_name = getattr(self, "pusher_collision_name", None)
        if geom_name is None:
            return
        self.env.set_geom_collision_enabled(
            geom_name,
            bool(enabled),
            contype=getattr(self, "_pusher_collision_contype", 1),
            conaffinity=getattr(self, "_pusher_collision_conaffinity", 1),
        )

    def _slot_grasp_throw_xyz(self, slot: Any) -> np.ndarray:
        grasp_xyz = self._world_to_throw_xyz(self._slot_center_world_xyz(slot))
        grasp_xyz[2] = self.action_grasp_height
        return grasp_xyz

    def _set_gripper_target_slot(self, slot: Any) -> None:
        self.gripper.set_target(
            target_body=slot.body_name,
            target_geom=slot.geom_name,
            target_site=slot.site_name,
            target_freejoint=slot.freejoint_name,
            equality_name=slot.equality_name,
        )

    def _slot_is_attached(self, slot: Any) -> bool:
        return bool(
            self.gripper.is_attached()
            and self.gripper.target_freejoint == slot.freejoint_name
        )

    def _suction_takes(self, slot: Any) -> bool:
        """Whether the suction GRIPS this object at the grasp instant: Bernoulli(
        slot.suction_p), rolled ONCE per object (latched in slot.suction_ok) with
        the env's seeded RNG, so the realized success rate is exactly p regardless
        of how many per-tick attach attempts occur. p>=1 / p<=0 short-circuit (no
        draw), keeping the default (p=1) byte-identical."""
        ok = getattr(slot, "suction_ok", None)
        if ok is None:
            p = float(getattr(slot, "suction_p", 1.0))
            if p >= 1.0:
                ok = True
            elif p <= 0.0:
                ok = False
            else:
                ok = bool(self._rng.random() < p)
            slot.suction_ok = ok
        return ok

    def _attach_slot_with_suction(self, slot: Any) -> bool:
        self._set_gripper_target_slot(slot)
        self.gripper.set_enabled(True)
        if self.gripper.is_attached():
            return True
        # Stochastic suction: only grip if the object is geometrically in the
        # suction window AND the (latched, once-per-object) Bernoulli roll takes.
        if self.gripper.passes_geometry() and self._suction_takes(slot):
            if self.gripper.try_attach():
                self._advance_simulation(1)
                return True
        return False

    def _rotate_xy(self, xy: np.ndarray, theta: float) -> np.ndarray:
        return np.array(
            [
                np.cos(theta) * xy[0] - np.sin(theta) * xy[1],
                np.sin(theta) * xy[0] + np.cos(theta) * xy[1],
            ],
            dtype=np.float64,
        )

    def _inverse_kinematics_vertical_grasp_xyz(self, throw_xyz: np.ndarray) -> np.ndarray | None:
        if self.robot_kinematics is None:
            return None
        pos = np.asarray(throw_xyz, dtype=np.float64)
        if pos.shape != (3,) or not np.all(np.isfinite(pos)):
            return None

        theta_1 = np.arctan2(pos[1], pos[0])
        dist = np.sqrt(pos[0] * pos[0] + pos[1] * pos[1])

        joint_positions = self.robot_kinematics.joint_positions
        joint2_zeropos = np.array([joint_positions[1][0], joint_positions[1][2]], dtype=np.float64)
        joint3_zeropos = np.array([joint_positions[2][0], joint_positions[2][2]], dtype=np.float64)
        joint5_zeropos = np.array([joint_positions[4][0], joint_positions[4][2]], dtype=np.float64)
        ee_zeropos = np.array([self.robot_kinematics.M[0, 3], self.robot_kinematics.M[2, 3]], dtype=np.float64)

        l1 = np.linalg.norm(joint3_zeropos - joint2_zeropos)
        l2 = np.linalg.norm(joint5_zeropos - joint3_zeropos)
        alpha_0 = np.arctan2(
            joint5_zeropos[1] - joint3_zeropos[1],
            joint5_zeropos[0] - joint3_zeropos[0],
        )

        joint5_pos = np.array([dist, pos[2] + (ee_zeropos[0] - joint5_zeropos[0])], dtype=np.float64)
        length = np.linalg.norm(joint5_pos - joint2_zeropos)
        if length > l1 + l2 or length < abs(l1 - l2):
            return None

        cos_alpha_1 = np.clip((l1 * l1 + length * length - l2 * l2) / (2.0 * l1 * length), -1.0, 1.0)
        cos_alpha_2 = np.clip((l1 * l1 + l2 * l2 - length * length) / (2.0 * l1 * l2), -1.0, 1.0)
        alpha_1 = np.arccos(cos_alpha_1)
        alpha_2 = np.arccos(cos_alpha_2)
        alpha_3 = np.arctan2(joint5_pos[1] - joint2_zeropos[1], joint5_pos[0] - joint2_zeropos[0])

        theta_2 = np.pi / 2.0 - (alpha_1 + alpha_3)
        theta_3 = (np.pi / 2.0 - alpha_0) - (np.pi - alpha_2)
        theta_5 = theta_2 - theta_3 - np.pi / 2.0
        q = np.array([theta_1, theta_2, theta_3, 0.0, theta_5, 0.0], dtype=np.float64)

        joint_bounds = np.asarray(self.robot_kinematics.joint_bounds, dtype=np.float64)
        if np.any(q < joint_bounds[:, 0]) or np.any(q > joint_bounds[:, 1]):
            return None
        return q

    def _forward_kinematics_xyz(self, q: np.ndarray) -> np.ndarray | None:
        if self.robot_kinematics is None:
            return None
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (6,):
            return None
        return np.asarray(self.robot_kinematics.forward_kinematics(q)[:3, 3], dtype=np.float64)

    def _predict_next_grasp_xyz(
        self,
        obj_xyz: np.ndarray,
        next_obj_xyz: np.ndarray,
        target_xy: np.ndarray | None = None,
        next_obj_speed: float | None = None,
        arrival_y_offset: float = 0.0,
        iterations: int = 3,
    ) -> tuple[np.ndarray, np.ndarray, float, float | None, float | None] | None:
        predicted_next_xyz = np.asarray(next_obj_xyz, dtype=np.float64).copy()
        trajectory_parameters: np.ndarray | None = None
        trajectory_duration = 1.0
        wait_time: float | None = None
        neg_wait_time: float | None = None
        object_speed = float(self.belt_speed if next_obj_speed is None else max(next_obj_speed, 1e-6))

        for _ in range(int(iterations)):
            proj = self._project_grasp_with_wait(
                next_obj_xyz, trajectory_duration, object_speed, arrival_y_offset
            )
            if proj is None:
                return None
            predicted_next_xyz, wait_time, neg_wait_time = proj

            model_result = self._throwing_model_parameters(obj_xyz, predicted_next_xyz, target_xy=target_xy)
            if model_result is None:
                return None
            trajectory_parameters, trajectory_duration = model_result

        if trajectory_parameters is None:
            return None
        return predicted_next_xyz, trajectory_parameters, trajectory_duration, wait_time, neg_wait_time

    def _project_grasp_with_wait(
        self,
        next_obj_xyz: np.ndarray,
        horizon: float,
        object_speed: float,
        arrival_y_offset: float = 0.0,
    ) -> tuple[np.ndarray, float | None, float | None] | None:
        """Shared grasp-handoff projection used by BOTH throw->throw
        (`_predict_next_grasp_xyz`) and push->throw (`_build_pushing_after_trajectory`
        THROW branch): forward-project the next object by `horizon` at `object_speed`,
        add the arrival lead, reach-radius clamp, and derive (wait_time, neg_wait_time).
          wait_time != None     -> object still upstream of the reach edge: hold this long.
          neg_wait_time != None -> object will be past the reach edge by arrival: ABANDON.
        Returns (predicted_xyz, wait_time, neg_wait_time) or None if x is outside the
        reachable disk. One rule => push->throw acquires exactly like throw->throw."""
        speed = max(float(object_speed), 1e-6)
        p = np.asarray(next_obj_xyz, dtype=np.float64).copy()
        p[1] -= speed * float(horizon)
        p[1] += float(arrival_y_offset)
        wait_time: float | None = None
        neg_wait_time: float | None = None
        if float(np.linalg.norm(p[:2])) > self.action_reachable_radius:
            reach_y_sq = self.action_reachable_radius**2 - p[0] ** 2
            if reach_y_sq < 0.0:
                return None
            reach_y = float(np.sqrt(reach_y_sq))
            if p[1] > 0.0:
                wait_time = (p[1] - reach_y) / speed
                p[1] = reach_y
            else:
                neg_wait_time = (p[1] + reach_y) / speed
                p[1] = -reach_y
        return p, wait_time, neg_wait_time

    def _throw_handoff_y_offset(self, object_speed: float) -> float:
        return float(max(object_speed, 0.0) * 0.04)

    def _next_action_xyz(self, next_slot: Any | None, next_primitive: int | None = None) -> np.ndarray:
        if next_slot is None:
            return self.action_single_object_standby_xyz.copy()

        primitive = self.ACTION_THROW if next_primitive is None else int(next_primitive)
        if primitive == self.ACTION_PUSH and self.pushing_trajectory_generator is not None:
            return self._push_handoff_xyz_from_obj(
                self._slot_grasp_throw_xyz(next_slot),
                target_xy=self._slot_target_xy(next_slot),
            )
        return self._slot_grasp_throw_xyz(next_slot)

    def _push_handoff_xyz_from_obj(self, obj_xyz: np.ndarray, target_xy: np.ndarray | None = None) -> np.ndarray:
        if self.pushing_trajectory_generator is None:
            return np.asarray(obj_xyz, dtype=np.float64).copy()
        push_obj_xyz = np.asarray(obj_xyz, dtype=np.float64).copy()
        push_obj_xyz[2] += float(self.pushing_params.get("height_offset", 0.0))
        target_xy = np.asarray(self.target_xy if target_xy is None else target_xy, dtype=np.float64)
        try:
            _, _, _, hover_pos = self.pushing_trajectory_generator.calculate_pushing_geometry(
                push_obj_xyz,
                target_xy,
                self.pushing_params,
            )
        except Exception:
            return np.asarray(obj_xyz, dtype=np.float64).copy()
        return np.asarray(hover_pos, dtype=np.float64)

    def _push_start_xyz_from_obj(self, obj_xyz: np.ndarray, target_xy: np.ndarray | None = None) -> np.ndarray | None:
        """Return the gripper position at the push START — behind the object at
        object-height, matching `start_pos` from `calculate_pushing_geometry`.
        This is where the gripper should arrive at the end of a chained
        previous trajectory so that the next push can begin without an
        approach phase."""
        if self.pushing_trajectory_generator is None:
            return None
        push_obj_xyz = np.asarray(obj_xyz, dtype=np.float64).copy()
        push_obj_xyz[2] += float(self.pushing_params.get("height_offset", 0.0))
        target_xy = np.asarray(self.target_xy if target_xy is None else target_xy, dtype=np.float64)
        try:
            _, push_start, _, _ = self.pushing_trajectory_generator.calculate_pushing_geometry(
                push_obj_xyz,
                target_xy,
                self.pushing_params,
            )
        except Exception:
            return None
        return np.asarray(push_start, dtype=np.float64)

    def _push_handoff_q_from_obj(
        self,
        obj_xyz: np.ndarray,
        target_xy: np.ndarray | None = None,
        seed: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Return the 6-joint state at the push HANDOFF pose -- directly above
        push_start at world z = pushing_after_height, with push orientation.

        This is used as the after-trajectory goal for push->push chains so the
        EE clears obj height during the inter-push traverse; the next push's
        bang-bang then descends straight down to push_start without sweeping
        the pusher through any obj on the belt.
        """
        if self.pushing_trajectory_generator is None or self.robot_kinematics is None:
            return None
        push_obj_xyz = np.asarray(obj_xyz, dtype=np.float64).copy()
        push_obj_xyz[2] += float(self.pushing_params.get("height_offset", 0.0))
        target_xy = np.asarray(self.target_xy if target_xy is None else target_xy, dtype=np.float64)
        try:
            push_dir, push_start, push_end, hover_pos = self.pushing_trajectory_generator.calculate_pushing_geometry(
                push_obj_xyz,
                target_xy,
                self.pushing_params,
            )
            t_start, _, _, _ = self.pushing_trajectory_generator.get_pushing_SE3_poses(
                push_start,
                push_end,
                push_dir=push_dir,
                swing_angle=float(self.pushing_params.get("swing_angle", 0.0)),
            )
        except Exception:
            return None
        if t_start is None or hover_pos is None:
            return None
        # Re-use push orientation; only swap the translation to hover xyz.
        t_handoff = np.array(t_start, dtype=np.float64, copy=True)
        t_handoff[:3, 3] = np.asarray(hover_pos, dtype=np.float64)
        if seed is None:
            seed = self.env.get_joint_positions(self.joint_names).astype(np.float64)
        else:
            seed = np.asarray(seed, dtype=np.float64)
        try:
            with self._quiet_planner_output():
                q_handoff = self.robot_kinematics.inverse_kinematics_np_push(t_handoff, seed=seed)
        except Exception:
            return None
        if q_handoff is None:
            return None
        q = np.asarray(q_handoff, dtype=np.float64)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            return None
        return q

    def _route_to_push_handoff(self, slot: Any, iterations: int = 3) -> tuple[bool, dict[str, Any]]:
        """Route the EE to the push HANDOFF (hover) pose above `slot`'s push_start,
        without pushing. Used at route-only steps so the subsequent push
        manipulation strikes straight down (no chase) and hits (no miss).

        The hover is placed above the push_start of where the object WILL BE at
        CONTACT time -- a forward projection by (route time to the hover) +
        (strike descent + start-margin traverse), iterated and clamped into the
        cone. This mirrors the THROW, whose previous trajectory ends at the
        predicted grasp position. (The earlier wait-at-cone-edge redesign aimed
        the hover at a FIXED cone_target_y and dropped this projection; that left
        the strike with only two bad choices -- lead the object laterally =>
        chase, or drop straight down => the object out-drifts the pusher => miss.
        Restoring the projection here fixes both: the strike becomes vertical AND
        lands on the object.)"""
        if not slot.active:
            return False, {"reason": "inactive_slot"}
        target_xy = self._slot_target_xy(slot)
        self._set_pusher_collision_enabled(False)
        obj_xyz = self._slot_grasp_throw_xyz(slot)
        if float(obj_xyz[1]) <= 0.05:
            return False, {"reason": "object_past_push_window", "obj_xyz": obj_xyz}
        # Place the hover via the SINGLE push-hover predictor (same function the
        # screening and strike use), from the CURRENT EE pose (route-only is the
        # cold-start case; base_travel=0). `_predict_push_hover` returns the hover
        # IK and the predicted contact target the strike will aim at.
        ref_q = self.env.get_joint_positions(self.joint_names).astype(np.float64)
        predicted = self._predict_push_hover(obj_xyz, target_xy, ref_q, base_travel=0.0)
        if predicted is None:
            return False, {"reason": "object_downstream_of_cone", "obj_xyz": obj_xyz}
        cone_obj_xyz, q_goal = predicted
        if q_goal is None:
            return False, {"reason": "push_handoff_ik_failed", "obj_xyz": cone_obj_xyz}
        trajectory = self._build_bangbang_move(q_goal)
        if trajectory is None:
            return False, {"reason": "push_handoff_trajectory_failed", "obj_xyz": cone_obj_xyz}
        if not self._trajectory_is_safe(trajectory):
            return False, {"reason": "unsafe_push_handoff_trajectory", "obj_xyz": cone_obj_xyz}
        self._execute_trajectory(trajectory)
        self._hold_current_robot_pose()
        # Mark this slot as pre-positioned AND stash the predicted contact target
        # (`cone_obj_xyz`, robot frame) the hover was placed above. The next push
        # step strikes a FIXED push to THIS target -- no strike-time re-planning
        # -- so the strike is a clean vertical drop that lands where the hover
        # already anticipated (no chase, no miss). Single source of truth.
        self._push_routed_spawn_id = (
            int(slot.spawn_id) if slot.spawn_id is not None else None
        )
        self._push_routed_target_xyz = np.asarray(cone_obj_xyz, dtype=np.float64).copy()
        return True, {
            "reason": "push_handoff_routed",
            "current_spawn_id": slot.spawn_id,
            "obj_xyz": cone_obj_xyz,
        }

    def _push_start_q_from_obj(
        self,
        obj_xyz: np.ndarray,
        target_xy: np.ndarray | None = None,
        seed: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Return the 6-joint state at the push start (behind obj at obj height,
        with push orientation). Used as the terminal joint state for chained
        throw->push and as the after-trajectory goal for push->push."""
        if self.pushing_trajectory_generator is None or self.robot_kinematics is None:
            return None
        push_obj_xyz = np.asarray(obj_xyz, dtype=np.float64).copy()
        push_obj_xyz[2] += float(self.pushing_params.get("height_offset", 0.0))
        target_xy = np.asarray(self.target_xy if target_xy is None else target_xy, dtype=np.float64)
        try:
            push_dir, push_start, push_end, _ = self.pushing_trajectory_generator.calculate_pushing_geometry(
                push_obj_xyz,
                target_xy,
                self.pushing_params,
            )
            t_start, _, _, _ = self.pushing_trajectory_generator.get_pushing_SE3_poses(
                push_start,
                push_end,
                push_dir=push_dir,
                swing_angle=float(self.pushing_params.get("swing_angle", 0.0)),
            )
        except Exception:
            return None
        if t_start is None:
            return None
        if seed is None:
            seed = self.env.get_joint_positions(self.joint_names).astype(np.float64)
        else:
            seed = np.asarray(seed, dtype=np.float64)
        try:
            with self._quiet_planner_output():
                q_push_start = self.robot_kinematics.inverse_kinematics_np_push(t_start, seed=seed)
        except Exception:
            return None
        if q_push_start is None:
            return None
        q = np.asarray(q_push_start, dtype=np.float64)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            return None
        return q

    def _predict_next_push_hover_handoff(
        self,
        obj_xyz: np.ndarray,
        next_obj_xyz: np.ndarray,
        current_target_xy: np.ndarray | None = None,
        next_target_xy: np.ndarray | None = None,
        next_obj_speed: float | None = None,
        seed_q: np.ndarray | None = None,
        iterations: int = 3,
    ) -> tuple[np.ndarray, np.ndarray, float, float | None, float | None, np.ndarray] | None:
        """Throw->push handoff via the SAME `_predict_push_hover` the route/
        push->push paths use, so screening, the throw endpoint, and the strike
        all agree (no cone-unaware push_start handoff, no `used_cone=False`
        old-solver fallback). The throw trajectory ROUTES the EE to the push
        HOVER (elevated above push_start of where the object will be at
        contact), so this returns the hover gripper xyz as the throw endpoint
        AND the predicted cone object xyz -- the strike target the executor
        stashes into `_push_routed_target_xyz`.

        Returns the same 6-tuple shape as the legacy `_predict_next_push_handoff_xyz`
        it replaces: (handoff_gripper_xyz, trajectory_parameters,
        trajectory_duration, wait_time, neg_wait_time, planned_cone_obj_xyz),
        or None when un-pushable. wait_time/neg_wait_time are always None: the
        cone clamp keeps the hover reachable and the strike executor handles any
        residual upstream wait at strike time; the too-early / drifted-past case
        is `_predict_push_hover` returning None here (== the old neg_wait/
        downstream rejection -> the caller's standby fallback)."""
        if self.pushing_trajectory_generator is None:
            return None
        obj_xyz = np.asarray(obj_xyz, dtype=np.float64)
        next_obj_xyz = np.asarray(next_obj_xyz, dtype=np.float64)
        next_target_xy = np.asarray(
            self.target_xy if next_target_xy is None else next_target_xy, dtype=np.float64
        )
        if seed_q is not None:
            ref_q = np.asarray(seed_q, dtype=np.float64)
        else:
            ref_q = self.env.get_joint_positions(self.joint_names).astype(np.float64)

        trajectory_parameters: np.ndarray | None = None
        trajectory_duration = 1.0
        handoff_xyz: np.ndarray | None = None
        cone_obj_xyz: np.ndarray | None = None
        for _ in range(int(iterations)):
            # The throw trajectory IS the route to the hover, so skip_route=True
            # -> _predict_push_hover's horizon = trajectory_duration + t_strike
            # (the throw duration plus the strike's descent+margin), NOT an extra
            # bogus ref_q->hover leg. Fixed-point on trajectory_duration, which
            # depends on the hover endpoint, which depends on the projection.
            predicted = self._predict_push_hover(
                next_obj_xyz, next_target_xy, ref_q,
                base_travel=trajectory_duration, skip_route=True,
            )
            if predicted is None:
                return None  # object un-pushable at contact -> standby fallback
            cone_obj_xyz, _hover_q = predicted
            handoff_xyz = self._push_handoff_xyz_from_obj(cone_obj_xyz, target_xy=next_target_xy)
            model_result = self._throwing_model_parameters(
                obj_xyz, handoff_xyz, target_xy=current_target_xy
            )
            if model_result is None:
                return None
            trajectory_parameters, trajectory_duration = model_result

        if trajectory_parameters is None or handoff_xyz is None or cone_obj_xyz is None:
            return None
        return handoff_xyz, trajectory_parameters, trajectory_duration, None, None, cone_obj_xyz

    def _throwing_trajectory_primitive(
        self,
        s: np.ndarray,
        q0: np.ndarray,
        qT: np.ndarray,
        w: np.ndarray,
        T: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build the throw trajectory waypoints. The NN-corrected interpolation
        spans joints 0..4 (the 5 joints affecting the end-effector position).
        Joint 5 (T-axis / wrist roll) is interpolated as a plain quintic
        s-curve from q0[5] to qT[5] when 6-joint endpoints are provided so we
        can chain into push, which needs a non-zero terminal wrist orientation.
        For 5-joint endpoints (legacy), joint 5 stays at zero."""
        s_col = s[:, None]
        f = 10 * (s_col**3) - 15 * (s_col**4) + 6 * (s_col**5)
        g = (s_col**3) * ((s_col - 1) ** 3)

        basis_count = w.shape[0]
        exp_center = np.arange(basis_count) / float(basis_count - 1)
        phi = np.exp(-(basis_count**2) * ((s_col - exp_center) ** 2))

        q0 = np.asarray(q0, dtype=np.float64)
        qT = np.asarray(qT, dtype=np.float64)
        q5d = q0[:5] + f * (qT[:5] - q0[:5]) + g * np.dot(phi, w)

        if q0.shape[-1] >= 6 and qT.shape[-1] >= 6:
            joint5_blend = float(q0[5]) + (f.squeeze(-1) * (float(qT[5]) - float(q0[5])))
            joint5_col = joint5_blend.reshape(-1, 1)
        else:
            joint5_col = np.zeros((q5d.shape[0], 1), dtype=np.float64)

        t = (s_col * T).squeeze()
        padded_q = np.concatenate((q5d, joint5_col), axis=1)
        return padded_q, t, q5d

    def _throwing_model_parameters(
        self,
        obj_xyz: np.ndarray,
        next_obj_xyz: np.ndarray,
        target_xy: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float] | None:
        if torch is None or self.throwing_model is None:
            return None

        target_xy = np.asarray(self.target_xy if target_xy is None else target_xy, dtype=np.float64)
        theta = np.arctan2(target_xy[1], target_xy[0])
        target_dist = float(np.linalg.norm(target_xy))
        obj_xy = np.asarray(obj_xyz[:2], dtype=np.float64)
        next_obj_xy = np.asarray(next_obj_xyz[:2], dtype=np.float64)
        obj_xy_rot = self._rotate_xy(obj_xy, -theta)
        next_obj_xy_rot = self._rotate_xy(next_obj_xy, -theta)
        nn_input = torch.tensor(
            [obj_xy_rot[0], obj_xy_rot[1], next_obj_xy_rot[0], next_obj_xy_rot[1], target_dist],
            dtype=torch.float32,
        )
        with torch.no_grad():
            trajectory_parameters = self.throwing_model(nn_input).detach().cpu().numpy()

        duration = float(np.exp(trajectory_parameters[-2]))
        return trajectory_parameters, duration

    def _build_throwing_trajectory(
        self,
        q_start: np.ndarray,
        obj_xyz: np.ndarray,
        next_obj_xyz: np.ndarray,
        target_xy: np.ndarray | None = None,
        trajectory_parameters: np.ndarray | None = None,
        next_primitive: int | None = None,
        next_target_xy: np.ndarray | None = None,
        next_obj_actual_xyz: np.ndarray | None = None,
    ) -> JointTrajectory | None:
        if trajectory_parameters is None:
            model_result = self._throwing_model_parameters(obj_xyz, next_obj_xyz, target_xy=target_xy)
            if model_result is None:
                return None
            trajectory_parameters, duration = model_result
        else:
            trajectory_parameters = np.asarray(trajectory_parameters, dtype=np.float64)
            duration = float(np.exp(trajectory_parameters[-2]))

        q_start = np.asarray(q_start, dtype=np.float64)
        if q_start.shape != (6,):
            return None

        weights = trajectory_parameters[:-2].reshape(-1, 5)
        num_steps = max(2, int(duration / self.env.dt))
        s = np.linspace(0.0, 1.0, num_steps + 1)

        # When the next action is push, end the throw at the push HOVER (elevated
        # above push_start by `pushing_after_height`, with the angled push
        # orientation), so the next push begins from the SAME pose the route->push
        # / push->push paths use and the strike is a clean vertical drop. The
        # executor stashes the cone strike target (`_predict_next_push_hover_handoff`)
        # so this push runs the matured `used_cone=True` fixed strike -- one
        # predictor everywhere. For all other cases (or fallback when push IK is
        # not solvable), end at vertical-grasp orientation at next_obj_xyz.
        next_obj_q = None
        if next_primitive is not None and int(next_primitive) == self.ACTION_PUSH:
            # IK_push needs the OBJECT's predicted (cone-clamped) position so the
            # hover geometry matches what the next push step will strike at.
            # `next_obj_xyz` is the gripper's hover target; `next_obj_actual_xyz`
            # is the predicted cone object xyz from the handoff predictor.
            obj_for_push = next_obj_actual_xyz if next_obj_actual_xyz is not None else next_obj_xyz
            next_obj_q = self._push_handoff_q_from_obj(
                obj_for_push,
                target_xy=next_target_xy,
                seed=q_start,
            )
        if next_obj_q is None:
            next_obj_q = self._inverse_kinematics_vertical_grasp_xyz(next_obj_xyz)
        if next_obj_q is None:
            return None

        # When chaining into push, pass the full 6-joint endpoints so the
        # primitive interpolates joint 5 (T-axis / wrist roll) toward the push
        # orientation. For throw->throw chains we keep the legacy 5-joint call
        # so the trajectory primitive's joint-5 column stays at zero (the NN
        # was trained that way).
        if next_primitive is not None and int(next_primitive) == self.ACTION_PUSH:
            traj, timestep, _ = self._throwing_trajectory_primitive(
                s, q_start[:6], next_obj_q[:6], weights, duration,
            )
        else:
            traj, timestep, _ = self._throwing_trajectory_primitive(
                s, q_start[:5], next_obj_q[:5], weights, duration,
            )
        return JointTrajectory.from_waypoints(
            joint_names=self.joint_names,
            times=timestep.tolist(),
            positions=traj.tolist(),
        )

    def _build_pushing_trajectory(
        self,
        q_start: np.ndarray,
        obj_xyz: np.ndarray,
        target_xy: np.ndarray | None = None,
    ) -> JointTrajectory | None:
        if self.pushing_trajectory_generator is None or self.robot_kinematics is None:
            return None

        push_obj_xyz = np.asarray(obj_xyz, dtype=np.float64).copy()
        push_obj_xyz[2] += float(self.pushing_params.get("height_offset", 0.0))
        target_xy = np.asarray(self.target_xy if target_xy is None else target_xy, dtype=np.float64)
        try:
            with self._quiet_planner_output():
                push_dir, push_start, push_end, _ = self.pushing_trajectory_generator.calculate_pushing_geometry(
                    push_obj_xyz,
                    target_xy,
                    self.pushing_params,
                )
                t_start, t_end, _, _ = self.pushing_trajectory_generator.get_pushing_SE3_poses(
                    push_start,
                    push_end,
                    push_dir=push_dir,
                    swing_angle=float(self.pushing_params.get("swing_angle", 0.0)),
                )
        except Exception:
            return None
        if t_start is None or t_end is None:
            return None

        q_start = np.asarray(q_start, dtype=np.float64)
        with self._quiet_planner_output():
            q_push_start = self.robot_kinematics.inverse_kinematics_np_push(t_start, seed=q_start)
        if q_push_start is None:
            return None

        # Pass the caller's q_start through so screening callers see a SHORT
        # bang-bang from the routed cone hover, not a long lateral move from
        # the env's current EE pose. Execution-time callers still pass the
        # env's current joints, which matches the env state, so behavior there
        # is unchanged.
        approach = self._build_bangbang_move(
            np.asarray(q_push_start, dtype=np.float64),
            q_start_override=q_start,
        )
        if approach is None:
            return None

        line_length = float(np.linalg.norm(np.asarray(push_end, dtype=np.float64) - np.asarray(push_start, dtype=np.float64)))
        push_velocity = max(float(self.pushing_params.get("velocity", 1.4)), 1e-3)
        push_duration = max(line_length / push_velocity, self.env.dt)
        num_push_steps = max(2, int(np.ceil(push_duration / self.env.dt)))
        push_times = np.linspace(0.0, push_duration, num_push_steps + 1)

        if Slerp is not None:
            from scipy.spatial.transform import Rotation as _Rotation

            rotations = _Rotation.from_matrix([t_start[:3, :3], t_end[:3, :3]])
            slerp = Slerp([0.0, 1.0], rotations)
        else:
            slerp = None

        push_positions = np.zeros((num_push_steps + 1, len(self.joint_names)), dtype=np.float64)
        previous_q = np.asarray(q_push_start, dtype=np.float64)
        for index, alpha in enumerate(np.linspace(0.0, 1.0, num_push_steps + 1)):
            transform = np.eye(4, dtype=np.float64)
            transform[:3, 3] = (1.0 - alpha) * np.asarray(push_start, dtype=np.float64) + alpha * np.asarray(push_end, dtype=np.float64)
            if slerp is None:
                transform[:3, :3] = t_start[:3, :3]
            else:
                transform[:3, :3] = slerp([float(alpha)]).as_matrix()[0]
            with self._quiet_planner_output():
                q = self.robot_kinematics.inverse_kinematics_np_push(transform, seed=previous_q)
            if q is None:
                return None
            push_positions[index, :] = np.asarray(q, dtype=np.float64)
            previous_q = push_positions[index, :]

        times = np.concatenate((approach.times, approach.duration + push_times[1:]))
        positions = np.vstack((approach.positions, push_positions[1:, :]))
        if not np.all(np.isfinite(positions)):
            return None
        if times.size != positions.shape[0] or times.size < 2 or not np.all(np.diff(times) > 0.0):
            return None

        if float(np.max(np.abs(positions - positions[0]))) < 1e-5:
            return None

        return JointTrajectory.from_waypoints(
            joint_names=self.joint_names,
            times=times.tolist(),
            positions=positions.tolist(),
        )

    def _push_motion_from_obj_xyz(
        self,
        obj_xyz: np.ndarray,
        start_world_z: float,
        target_xy: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if self.pushing_trajectory_generator is None:
            return None
        obj_xyz = np.asarray(obj_xyz, dtype=np.float64)
        target_xy = np.asarray(self.target_xy if target_xy is None else target_xy, dtype=np.float64)
        try:
            push_dir, _, push_end, _ = self.pushing_trajectory_generator.calculate_pushing_geometry(
                obj_xyz,
                target_xy,
                self.pushing_params,
            )
        except Exception:
            return None
        push_dir = np.asarray(push_dir, dtype=np.float64)
        if push_dir.shape != (3,) or not np.all(np.isfinite(push_dir)):
            return None
        start_world = np.array(
            [
                obj_xyz[0] + self.robot_base_xyz[0],
                obj_xyz[1] + self.robot_base_xyz[1],
                start_world_z,
            ],
            dtype=np.float64,
        )
        target_robot_xyz = np.asarray(push_end, dtype=np.float64)
        target_world = np.array(
            [
                target_robot_xyz[0] + self.robot_base_xyz[0],
                target_robot_xyz[1] + self.robot_base_xyz[1],
                start_world_z,
            ],
            dtype=np.float64,
        )
        return start_world, target_world, push_dir

    def _push_target_world_xyz(self, slot: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        return self._push_motion_from_obj_xyz(
            self._slot_grasp_throw_xyz(slot),
            self._slot_center_world_xyz(slot)[2],
            target_xy=self._slot_target_xy(slot),
        )

    def _push_cone_target_y(
        self, obj_xyz: np.ndarray, target_xy: np.ndarray
    ) -> tuple[float, float, float] | None:
        """Where (robot-frame y) to aim the push, screened against the REACHABLE
        CIRCLE -- identical region to the throw (2026-06-11).

        NAME IS LEGACY: this used to clamp the push target into a ~45-deg cone
        `|obj_y - bin_y| <= (bin_x - obj_x)` (the wedge that geometrically
        guarantees the push ray enters the bin's -x opening), intersected with
        the workspace disk. That hand-designed cone prior is REMOVED: it baked
        "where a push can succeed" into the action mask, so an RL agent could
        never attempt -- and therefore never learn from -- a push outside the
        wedge. Both primitives now screen against the same reachable disk and the
        policy learns the bin-opening geometry from the (outcome-based) reward.
        `push_cone_offset` / `_push_cone_min_reach` are no longer read.

        The robust predict -> WAIT -> EXECUTE -> ABANDON tracking is unchanged;
        only the boundary between the three zones moves from cone-disk to disk:
            reach_y_max = sqrt(r^2 - obj_x^2)   (r = action_reachable_radius)
          - obj upstream of +reach_y_max -> aim at the edge (WAIT; the object
            drifts in and the executor's live-position wait catches it),
          - obj inside the disk            -> aim at the object's own y (EXECUTE),
          - obj past -reach_y_max          -> None (ABANDON, past the far edge).
        This mirrors the throw's `_build_initial_approach_plan` / the disk clamp
        in `_project_grasp_with_wait`. The strike direction is unaffected -- it
        is computed toward the bin in `_push_motion_from_obj_xyz` regardless of
        where along the disk the object sits.

        `target_xy` is retained for signature compatibility (callers still pass
        the bin pose) but is no longer used here. Returns (target_y, -reach_y_max,
        +reach_y_max) in robot frame, or None when the object's x is outside the
        reachable x-disk or the object is already past the far reachable edge.
        """
        obj_x = float(obj_xyz[0])
        obj_y = float(obj_xyz[1])
        r = float(self.action_reachable_radius)
        reach_y_sq = r * r - obj_x * obj_x
        if reach_y_sq <= 0.0:
            return None  # object's x is outside the 2D reachable disk
        reach_y_max = float(np.sqrt(reach_y_sq))
        if obj_y < -reach_y_max:
            return None  # ABANDON: object already past the far reachable edge
        # WAIT (obj upstream of +edge) -> aim at the edge; EXECUTE (in-disk) ->
        # aim at the object's own y. Symmetric with the throw's reach clamp.
        target_y = float(np.clip(obj_y, -reach_y_max, reach_y_max))
        return target_y, -reach_y_max, reach_y_max

    def _build_cone_pushing_plan(
        self,
        q_start: np.ndarray,
        obj_xyz: np.ndarray,
        start_world_z: float,
        target_xy: np.ndarray | None = None,
        object_speed: float | None = None,
    ) -> tuple[np.ndarray, JointTrajectory, tuple[np.ndarray, np.ndarray, np.ndarray], float] | None:
        """Wait-at-the-cone-edge push plan (no chase). Aims the push at the cone
        push point (see `_push_cone_target_y`) with only a SMALL contact lead for
        the short vertical strike (object drift during the bang-bang descent +
        start-margin traverse). Returns the same 4-tuple as
        `_build_predicted_pushing_plan` so call sites are interchangeable.

        Assumes the EE has been pre-positioned above the cone push_start (the
        previous step routed it there), so the strike bang-bang is short. From a
        far q_start the strike would be long and the contact would overshoot the
        cone -> returns None (the caller uses the interception solver instead)."""
        if self.pushing_trajectory_generator is None:
            return None
        obj_xyz = np.asarray(obj_xyz, dtype=np.float64).copy()
        target_xy = np.asarray(self.target_xy if target_xy is None else target_xy, dtype=np.float64)
        if float(obj_xyz[1]) <= 0.05:
            return None
        cone = self._push_cone_target_y(obj_xyz, target_xy)
        if cone is None:
            return None
        cone_target_y, _cone_lower, _cone_upper = cone

        push_distance = max(float(self.pushing_params.get("distance", 0.15)), 0.0)
        push_start_margin = max(float(self.pushing_params.get("start_margin", 0.10)), 0.0)
        push_velocity = max(float(self.pushing_params.get("velocity", 1.4)), 1e-3)
        line_duration = (push_start_margin + push_distance) / push_velocity
        start_margin_time = push_start_margin / push_velocity
        speed = float(self.belt_speed if object_speed is None else max(object_speed, 1e-6))

        # VERTICAL strike (2026-06-01 fix): aim the strike AT the cone push point
        # (where the EE already hovers), drifting downstream by ONLY the
        # start-margin line traverse the pusher travels before contact
        # (~speed*start_margin/velocity ~= 5 cm). The previous ct-based damped
        # fixed-point led the contact point DOWNSTREAM of the hover by
        # speed*(bang_bang_dur + margin); since planned_y was downstream of the
        # hover, the bang-bang hover->push_start(planned_y) was LATERAL, so ct
        # blew up (~0.7 s) and self-reinforced into a ~34-42 cm chase (proven:
        # 130 builds, mean lead 0.42 m, ct 0.71 s). Aiming at cone_target_y makes
        # the descent vertical; any residual in-strike timing error is absorbed
        # by the executor's wait-then-rebuild loop in
        # _execute_push_transition_with_next.
        planned_obj_xyz = obj_xyz.copy()
        planned_obj_xyz[1] = float(cone_target_y) - speed * start_margin_time
        # Push-window guard. The cone-vs-cone_lower check that used to live here
        # has been DROPPED -- with the workspace clamp now baked into
        # `_push_cone_target_y`, the cone bounds we get back are already the
        # intersection of cone and workspace, so a slight overshoot from the
        # contact lead would otherwise reject a perfectly engageable push and
        # let the object drift past (the throw-only maturity gap). The push
        # primitive's own `obj_y <= 0.05` gate catches anything past the window.
        if float(planned_obj_xyz[1]) <= 0.05:
            return None
        # Reach clamp (mirror the throw, _build_initial_approach_plan). After
        # clamping we do NOT re-reject on cone_lower: the clamped point is on
        # the workspace edge and the push ray still aims into the bin opening,
        # so refusing it would just send the object past untouched.
        radial = float(np.linalg.norm(planned_obj_xyz[:2]))
        if radial > self.action_reachable_radius:
            reach_y_sq = self.action_reachable_radius**2 - float(planned_obj_xyz[0]) ** 2
            if reach_y_sq < 0.0:
                return None
            planned_obj_xyz[1] = float(np.sign(planned_obj_xyz[1])) * float(np.sqrt(reach_y_sq))

        trajectory = self._build_pushing_trajectory(q_start, planned_obj_xyz, target_xy=target_xy)
        if trajectory is None:
            return None
        push_motion = self._push_motion_from_obj_xyz(planned_obj_xyz, start_world_z, target_xy=target_xy)
        if push_motion is None:
            return None
        contact_time = max(0.0, float(trajectory.duration) - line_duration)
        return planned_obj_xyz, trajectory, push_motion, contact_time

    def _build_predicted_pushing_plan(
        self,
        q_start: np.ndarray,
        obj_xyz: np.ndarray,
        start_world_z: float,
        target_xy: np.ndarray | None = None,
        object_speed: float | None = None,
        iterations: int = 8,
    ) -> tuple[np.ndarray, JointTrajectory, tuple[np.ndarray, np.ndarray, np.ndarray], float] | None:
        # Iteratively predict where obj will be at contact time (= when pusher
        # reaches push_start). Fixed-point iteration:
        #   planned_obj = obj_initial - belt_speed * bang_bang_dur(planned_obj)
        # Converges iff belt_speed/EE_bang_bang_velocity < 1. If gripper's bang-bang
        # EE velocity is below belt speed (e.g., because action_initial_velocity_scale
        # is small or the chain alignment was already small), the iteration is
        # divergent: each round inflates the shift further. We detect this by
        # checking whether contact_time grew vs the previous iter, and bail to
        # the previous (smaller) planned_obj_xyz. This way the loop converges
        # toward the right meeting point when possible and clamps at the best
        # estimate when not.
        if float(np.asarray(obj_xyz, dtype=np.float64)[1]) <= 0.05:
            return None
        planned_obj_xyz = np.asarray(obj_xyz, dtype=np.float64).copy()
        contact_time = 0.0
        push_distance = max(float(self.pushing_params.get("distance", 0.15)), 0.0)
        push_start_margin = max(float(self.pushing_params.get("start_margin", 0.10)), 0.0)
        push_velocity = max(float(self.pushing_params.get("velocity", 1.4)), 1e-3)
        # The push line runs from push_start (= obj - start_margin * push_dir)
        # to push_end (= obj + push_distance * push_dir), so the line is
        # `start_margin + push_distance` long, not just `push_distance`.
        # contact_time = trajectory.duration - line_duration = bang-bang duration.
        line_duration = (push_start_margin + push_distance) / push_velocity
        target_xy = np.asarray(self.target_xy if target_xy is None else target_xy, dtype=np.float64)
        speed = float(self.belt_speed if object_speed is None else max(object_speed, 1e-6))

        for _ in range(int(iterations)):
            trajectory = self._build_pushing_trajectory(q_start, planned_obj_xyz, target_xy=target_xy)
            if trajectory is None:
                return None
            contact_time = max(0.0, trajectory.duration - line_duration)
            predicted_obj_xyz = np.asarray(obj_xyz, dtype=np.float64).copy()
            # Under physical belt (use_physical_belt=True, post Phase 1 belt
            # rollout): the belt geometry physically slides and carries the obj
            # via friction. The obj continues moving downstream at ~belt_speed
            # during the entire push primitive: (a) bang-bang approach until
            # EE reaches push_start, then (b) line traverse of start_margin
            # until pusher's leading face contacts the obj. Total time-to-
            # first-contact = contact_time + start_margin / push_velocity.
            # Without the start_margin term, the predictor under-projects by
            # ~60 mm at belt_speed=0.5 / start_margin=0.17 / push_vel=1.4,
            # leaving the pusher arriving 60 mm upstream of where obj actually
            # is and producing the dominant NT_CONTACT_FLOOR failure mode.
            # Under kinematic belt (use_physical_belt=False, legacy): the obj
            # is qvel-frozen during the push window (per the _pushing_slot
            # skip + zero-qvel-at-push-start trick) and effectively stationary.
            # Skip the projection in that case to match the legacy behavior.
            if getattr(self, "use_physical_belt", False):
                start_margin_traverse_time = push_start_margin / push_velocity
                total_time_to_contact = contact_time + start_margin_traverse_time
                predicted_obj_xyz[1] -= speed * total_time_to_contact
            predicted_obj_xyz[2] = obj_xyz[2]
            # DAMPED fixed-point update (alpha < 1). predicted_obj_xyz is the full
            # projection target (object at contact). The UNDAMPED full step
            # diverges under the physical belt -- the belt is fast vs the EE's
            # bang-bang response from a short move, so each round inflates the
            # lead until it settles at a far ~42 cm / 0.67 s interception (the EE
            # "chases" the object downstream before pushing). Damping keeps the
            # iterate in the basin of the NEAR root (short descent from the
            # pre-positioned hover) -> the EE descends straight onto push_start
            # with a small, physically-correct lead. This is NOT a clamp (which
            # under-leads -> pusher arrives upstream -> NT_CONTACT_FLOOR); the lead
            # stays correct, just near. (Diagnosed 2026-05-29; see plan file.)
            new_planned_obj_xyz = planned_obj_xyz + 0.5 * (predicted_obj_xyz - planned_obj_xyz)
            if float(np.linalg.norm(new_planned_obj_xyz[:2] - planned_obj_xyz[:2])) < 1e-4:
                planned_obj_xyz = new_planned_obj_xyz
                break
            planned_obj_xyz = new_planned_obj_xyz

        # Rebuild for the converged interception and recompute contact_time so the
        # returned trajectory and contact_time are consistent with planned_obj.
        trajectory = self._build_pushing_trajectory(q_start, planned_obj_xyz, target_xy=target_xy)
        if trajectory is None:
            return None
        contact_time = max(0.0, trajectory.duration - line_duration)
        push_motion = self._push_motion_from_obj_xyz(planned_obj_xyz, start_world_z, target_xy=target_xy)
        if push_motion is None:
            return None
        return planned_obj_xyz, trajectory, push_motion, contact_time

    def _trajectory_end_velocity(self, trajectory: JointTrajectory) -> np.ndarray:
        positions = np.asarray(trajectory.positions, dtype=np.float64)
        times = np.asarray(trajectory.times, dtype=np.float64)
        if positions.shape[0] < 2:
            return np.zeros(positions.shape[1], dtype=np.float64)
        dt = float(times[-1] - times[-2])
        if dt <= 1e-9:
            return np.zeros(positions.shape[1], dtype=np.float64)
        return (positions[-1] - positions[-2]) / dt

    def _build_pushing_after_trajectory(
        self,
        last_trajectory: JointTrajectory,
        next_slot: Any | None,
        current_slot: Any | None = None,
        next_primitive: int | None = None,
        iterations: int = 3,
        extra_projection_seconds: float = 0.0,
    ) -> tuple[JointTrajectory, np.ndarray | None] | None:
        # `extra_projection_seconds` is added to the projection of the next obj
        # so callers can model time that will elapse BEFORE the after-traj runs.
        # Selection-time callers (demo policy scoring `_push_transition_margin`,
        # `_throw_transition_supports_next`) pass the predicted current-push
        # trajectory's duration so the y-window gate sees where obj B will be
        # at the moment of the next push, not where B is now. The execution-time
        # caller in `_execute_push_transition_with_next` keeps the default 0
        # because B has already drifted physically during the just-executed
        # push.
        if self.pushing_trajectory_generator is None:
            return None

        # Cleared here; set only on the push->throw THROW success path so the
        # executor dwells the right amount (None on standby/push/None paths).
        self._after_traj_grasp_wait_time = None

        last_q_end = np.asarray(last_trajectory.positions[-1], dtype=np.float64)
        last_dq_end = self._trajectory_end_velocity(last_trajectory)

        if next_slot is None:
            q_goal = self._inverse_kinematics_vertical_grasp_xyz(self.action_single_object_standby_xyz)
            if q_goal is None:
                return None
            trajectory = self._build_bangbang_move(q_goal)
            if trajectory is None:
                return None
            return trajectory, self.action_single_object_standby_xyz.copy()

        primitive = self.ACTION_THROW if next_primitive is None else int(next_primitive)
        # CAPPED tracking speed for the THROW-branch parking projection (used at
        # :1522). The post-push route to the next grasp is long, so projection
        # error = speed_error * duration is amplified; using the raw uncapped
        # _slot_forward_speed over-projected the parked pose (transient spawn
        # spikes / the +-10% steady spread) and drove the push->throw grasp
        # failure. _slot_planning_speed clamps to [0.75*belt, belt], matching the
        # reliable throw->throw predictor _predict_next_grasp_xyz. (PUSH branch
        # below does not use next_speed -- it uses _predict_push_hover.)
        next_speed = self._slot_planning_speed(next_slot)

        def _standby_fallback() -> tuple[JointTrajectory, np.ndarray] | None:
            q_goal = self._inverse_kinematics_vertical_grasp_xyz(self.action_single_object_standby_xyz)
            if q_goal is None:
                return None
            trajectory = self._build_bangbang_move(q_goal)
            if trajectory is None:
                return None
            return trajectory, self.action_single_object_standby_xyz.copy()

        if primitive == self.ACTION_THROW:
            # Acquire the next object the SAME way throw->throw does: over-project
            # via the shared `_project_grasp_with_wait` so the (slow, under-load)
            # object lands UPSTREAM of the parked pose, propagate wait_time so the
            # executor dwells, and ABANDON on neg_wait when the object would be past
            # by arrival (mirrors _predict_next_grasp_xyz, replacing the old short-
            # horizon projection + magic [0.08,0.45] window).
            next_grasp_xyz = self._slot_grasp_throw_xyz(next_slot)
            arrival_lead = self._throw_handoff_y_offset(next_speed)
            # Horizon = the bang-bang route the EE will take + the time already
            # modeled by the caller (base/push duration) + the EE-arrival->capture
            # pad (the SAME derived pad `_build_initial_approach_plan` uses). The
            # route duration depends on the parked pose, so iterate to a fixed point.
            prediction_pad = max(float(self.action_prediction_horizon), 0.08)
            planned_next_obj_xyz = next_grasp_xyz.copy()
            after_traj: JointTrajectory | None = None
            grasp_wait_time: float | None = None
            for _ in range(int(iterations)):
                q_goal = self._inverse_kinematics_vertical_grasp_xyz(planned_next_obj_xyz)
                if q_goal is None:
                    return _standby_fallback()
                after_traj = self._build_bangbang_move(q_goal)
                if after_traj is None:
                    return _standby_fallback()
                horizon = float(after_traj.duration) + float(extra_projection_seconds) + prediction_pad
                proj = self._project_grasp_with_wait(
                    next_grasp_xyz, horizon, next_speed, arrival_lead
                )
                if proj is None:
                    return _standby_fallback()
                predicted_next, grasp_wait_time, neg_wait_time = proj
                if neg_wait_time is not None:
                    # Object will be past the reachable edge by arrival -> doomed.
                    return _standby_fallback()
                if float(np.linalg.norm(predicted_next[:2] - planned_next_obj_xyz[:2])) < 1e-4:
                    planned_next_obj_xyz = predicted_next
                    break
                planned_next_obj_xyz = predicted_next

            if after_traj is None:
                return None
            # Stash the wait so the executor can dwell before the grasp window
            # opens (mirrors throw->throw's wait dwell), pre-positioning the object
            # upstream so `_open_loop_grasp_window` can catch it. Consumed once.
            self._after_traj_grasp_wait_time = grasp_wait_time
            return after_traj, planned_next_obj_xyz

        # next_primitive == ACTION_PUSH: end the after-trajectory at the next
        # object's hover via the SINGLE push-hover predictor (same function the
        # screening and the strike use). ref_q = last_q_end (where this push
        # leaves the EE -- the route to the next hover starts there);
        # base_travel = extra_projection_seconds (time before this after-traj
        # runs, i.e. the current push's remaining duration at selection time, 0
        # at execution). `_predict_push_hover` returns the predicted-at-contact
        # target + the hover IK; placing the hover there makes the next strike a
        # clean vertical drop that lands on the object. Falls back to standby if
        # no valid hover exists (object will have drifted past the cone).
        target_xy = self._slot_target_xy(next_slot)
        next_obj_xyz = self._slot_grasp_throw_xyz(next_slot)
        predicted = self._predict_push_hover(
            next_obj_xyz, target_xy, last_q_end,
            base_travel=float(max(extra_projection_seconds, 0.0)),
        )
        if predicted is None:
            return _standby_fallback()
        cone_obj_xyz, push_q = predicted
        push_xyz = self._push_handoff_xyz_from_obj(cone_obj_xyz, target_xy=target_xy)
        if push_q is None or push_xyz is None:
            return _standby_fallback()
        after_traj = self._build_bangbang_move(push_q)
        if after_traj is None:
            return _standby_fallback()
        # Stash the predicted contact target so the executor can commit it to
        # `_push_routed_target_xyz` IFF it actually runs this after-traj (this
        # function is also called at selection time by the demo policy, so we do
        # NOT write the committed marker here -- only a transient the executor
        # consumes). Single source of truth for the next fixed strike.
        self._after_traj_push_target_xyz = np.asarray(cone_obj_xyz, dtype=np.float64).copy()
        return after_traj, np.asarray(push_xyz, dtype=np.float64)

    def _trajectory_is_safe(self, trajectory: JointTrajectory) -> bool:
        positions = np.asarray(trajectory.positions, dtype=np.float64)
        if not np.all(np.isfinite(positions)):
            return False
        if self.robot_kinematics is None:
            return True

        joint_bounds = np.asarray(self.robot_kinematics.joint_bounds, dtype=np.float64)
        tolerance = 1e-6
        return bool(
            np.all(positions >= (joint_bounds[:, 0][None, :] - tolerance))
            and np.all(positions <= (joint_bounds[:, 1][None, :] + tolerance))
        )

    def _initial_motion_limits(self) -> tuple[np.ndarray, np.ndarray]:
        if self.robot_kinematics is None:
            max_velocity = np.array((455, 385, 520, 550, 550, 1000), dtype=np.float64) * (np.pi / 180.0)
        else:
            velocity_bounds = np.asarray(self.robot_kinematics.jointvel_bounds, dtype=np.float64)
            max_velocity = velocity_bounds[:, 1]
        max_velocity = max_velocity * self.action_initial_velocity_scale
        max_acceleration = max_velocity * 2.0
        return max_velocity, max_acceleration

    def _opt_time_1d(
        self,
        xi: float,
        vi: float,
        xf: float,
        vf: float,
        max_velocity: float,
        max_acceleration: float,
    ) -> float:
        xi = float(xi)
        vi = float(vi)
        xf = float(xf)
        vf = float(vf)
        max_velocity = float(max_velocity)
        max_acceleration = float(max_acceleration)

        if vf >= vi:
            if xf - xi < 0.5 / max_acceleration * (vf * vf - vi * vi):
                xi, xf, vi, vf = -xi, -xf, -vi, -vf
        elif xf - xi < 0.5 / max_acceleration * (vi * vi - vf * vf):
            xi, xf, vi, vf = -xi, -xf, -vi, -vf

        threshold = -0.5 / max_acceleration * (vi * vi + vf * vf) + max_velocity * max_velocity / max_acceleration
        if xf - xi <= threshold:
            root = np.sqrt(max(0.0, 0.5 * (vi * vi + vf * vf) + max_acceleration * (xf - xi)))
            return float(max(0.0, (-vi + root) / max_acceleration + (-vf + root) / max_acceleration))

        t1 = (max_velocity - vi) / max_acceleration
        t12 = ((xf - xi) - threshold) / max_velocity
        t2 = (max_velocity - vf) / max_acceleration
        return float(max(0.0, t1 + t12 + t2))

    def _opt_time_bangbang(self, q_start: np.ndarray, q_goal: np.ndarray) -> float:
        max_velocity, max_acceleration = self._initial_motion_limits()
        q_start = np.asarray(q_start, dtype=np.float64)
        q_goal = np.asarray(q_goal, dtype=np.float64)
        times = [
            self._opt_time_1d(q_start[i], 0.0, q_goal[i], 0.0, max_velocity[i], max_acceleration[i])
            for i in range(len(self.joint_names))
        ]
        return float(max(max(times), self.env.dt))

    def _trajectory_1d(
        self,
        xi: float,
        vi: float,
        xf: float,
        vf: float,
        max_velocity: float,
        max_acceleration: float,
        duration: float,
        times: np.ndarray,
    ) -> np.ndarray | None:
        if vf >= vi:
            peak_velocity = (max_acceleration * (xf - xi) + (vi * vi - vf * vf) / 2.0) / (
                vi - vf + duration * max_acceleration
            )
        else:
            peak_velocity = (max_acceleration * (xf - xi) + (vf * vf - vi * vi) / 2.0) / (
                vf - vi + duration * max_acceleration
            )

        trajectory = np.zeros_like(times, dtype=np.float64)
        if (peak_velocity - vi) * (peak_velocity - vf) < 0.0:
            t1 = abs(peak_velocity - vi) / max_acceleration
            t2 = abs(vf - peak_velocity) / max_acceleration
            t12 = duration - (t1 + t2)
            accel_sign = 1.0 if peak_velocity > vi else -1.0
        else:
            if peak_velocity > vi:
                b = vi + vf + duration * max_acceleration
                c = (vi * vi + vf * vf) / 2.0 + max_acceleration * (xf - xi)
                peak_velocity = (b - np.sqrt(max(0.0, b * b - 4.0 * c + 1e-8))) / 2.0
                accel_sign = 1.0
            else:
                b = vi + vf - duration * max_acceleration
                c = (vi * vi + vf * vf) / 2.0 - max_acceleration * (xf - xi)
                peak_velocity = (b + np.sqrt(max(0.0, b * b - 4.0 * c + 1e-8))) / 2.0
                accel_sign = -1.0
            if abs(peak_velocity) > max_velocity:
                return None
            t1 = abs(peak_velocity - vi) / max_acceleration
            t2 = abs(vf - peak_velocity) / max_acceleration
            t12 = duration - (t1 + t2)

        t12 = max(0.0, t12)
        p1 = xi + vi * t1 + 0.5 * accel_sign * max_acceleration * t1 * t1
        p2 = p1 + peak_velocity * t12
        decel_sign = -accel_sign

        first_mask = times < t1
        second_mask = (times >= t1) & (times < t1 + t12)
        third_mask = ~(first_mask | second_mask)

        t_first = times[first_mask]
        trajectory[first_mask] = xi + vi * t_first + 0.5 * accel_sign * max_acceleration * t_first * t_first
        trajectory[second_mask] = p1 + peak_velocity * (times[second_mask] - t1)

        t_third = times[third_mask] - (t1 + t12)
        trajectory[third_mask] = p2 + peak_velocity * t_third + 0.5 * decel_sign * max_acceleration * t_third * t_third
        trajectory[-1] = xf
        return trajectory

    def _build_bangbang_move(
        self,
        q_goal: np.ndarray,
        duration_override: float | None = None,
        q_start_override: np.ndarray | None = None,
    ) -> JointTrajectory | None:
        # Default to the env's CURRENT joints. Screening/feasibility callers
        # (e.g. `_slot_push_feasible` -> `_build_cone_pushing_plan` ->
        # `_build_pushing_trajectory`) pass `q_start_override` so the bang-bang
        # is measured from the pose the EE WILL be in when the action fires
        # (the routed cone hover), not from wherever the EE is RIGHT NOW. Without
        # this override the screening secretly bang-bangs from the current EE
        # pose -> inflated strike_time -> the cone plan's planned_y is dragged
        # below the push window -> mask falsely says "infeasible."
        if q_start_override is not None:
            q_start = np.asarray(q_start_override, dtype=np.float64)
        else:
            q_start = self.env.get_joint_positions(self.joint_names).astype(np.float64)
        q_goal = np.asarray(q_goal, dtype=np.float64)
        max_velocity, max_acceleration = self._initial_motion_limits()
        duration = self._opt_time_bangbang(q_start, q_goal)
        if duration_override is not None:
            duration = max(float(duration), float(duration_override))
        # Defensive cap: if the planner's time estimate is absurd (e.g., target
        # is unreachable in a normal action window because an obj has flown
        # off-belt and the planner is trying to track it), abort cleanly
        # instead of allocating gigabytes for the trajectory array.
        if not np.isfinite(duration) or duration > 10.0:
            return None
        num_steps = max(2, int(np.ceil(duration / self.env.dt)))
        times = np.linspace(0.0, duration, num_steps + 1)
        positions = np.zeros((num_steps + 1, len(self.joint_names)), dtype=np.float64)
        for i in range(len(self.joint_names)):
            joint_traj = self._trajectory_1d(
                q_start[i],
                0.0,
                q_goal[i],
                0.0,
                max_velocity[i],
                max_acceleration[i],
                duration,
                times,
            )
            if joint_traj is None:
                return None
            positions[:, i] = joint_traj
        positions[-1, :] = q_goal
        return JointTrajectory.from_waypoints(
            joint_names=self.joint_names,
            times=times.tolist(),
            positions=positions.tolist(),
        )

    def _execute_trajectory(
        self,
        trajectory: JointTrajectory,
        release_time: float | None = None,
        release_callback: Callable[[], None] | None = None,
    ) -> None:
        self.controller.load_trajectory(trajectory, start_time=self.env.sim_time)
        start_time = self.env.sim_time
        released = False
        while not self.controller.is_finished(self.env.sim_time):
            if (
                not released
                and release_time is not None
                and (self.env.sim_time - start_time) >= release_time
            ):
                if release_callback is not None:
                    release_callback()
                released = True
            self.controller.update(self.env.sim_time)
            self._advance_simulation(1)
        self.controller.hold_position(trajectory.positions[-1])
        if not released and release_time is not None:
            if release_callback is not None:
                release_callback()

    def _execute_two_point_move(self, q_goal: np.ndarray, duration: float = 1.0) -> None:
        q_start = self.env.get_joint_positions(self.joint_names)
        trajectory = JointTrajectory.from_waypoints(
            joint_names=self.joint_names,
            times=[0.0, float(duration)],
            positions=[q_start.tolist(), np.asarray(q_goal, dtype=np.float64).tolist()],
        )
        self._execute_trajectory(trajectory)
        self.controller.hold_position(np.asarray(q_goal, dtype=np.float64))

    def _release_fraction_from_parameters(self, trajectory_parameters: np.ndarray) -> float:
        trajectory_parameters = np.asarray(trajectory_parameters, dtype=np.float64)
        return float(1.0 / (1.0 + np.exp(-trajectory_parameters[-1])))

    def _build_initial_approach_plan(
        self,
        obj_xyz: np.ndarray,
        object_speed: float | None = None,
        iterations: int = 4,
    ) -> tuple[np.ndarray, JointTrajectory, float] | None:
        obj_xyz = np.asarray(obj_xyz, dtype=np.float64)
        target_xyz = obj_xyz.copy()
        target_xyz[2] = self.action_grasp_height
        wait_time = 0.0
        prediction_pad = max(float(self.action_prediction_horizon), 0.08)
        speed = float(self.belt_speed if object_speed is None else max(object_speed, 1e-6))

        for _ in range(int(iterations)):
            radial_distance = float(np.linalg.norm(target_xyz[:2]))
            if radial_distance > self.action_reachable_radius:
                reach_y_sq = self.action_reachable_radius**2 - target_xyz[0] ** 2
                if reach_y_sq < 0.0:
                    return None
                target_xyz[1] = np.sign(target_xyz[1]) * float(np.sqrt(reach_y_sq))

            q_goal = self._inverse_kinematics_vertical_grasp_xyz(target_xyz)
            if q_goal is None:
                return None

            trajectory = self._build_bangbang_move(q_goal)
            if trajectory is None:
                return None

            predicted_y = obj_xyz[1] - speed * (trajectory.duration + prediction_pad)
            predicted_xyz = obj_xyz.copy()
            predicted_xyz[1] = predicted_y
            predicted_xyz[2] = self.action_grasp_height

            predicted_radial = float(np.linalg.norm(predicted_xyz[:2]))
            if predicted_radial <= self.action_reachable_radius:
                wait_time = 0.0
                target_xyz = predicted_xyz
            else:
                reach_y_sq = self.action_reachable_radius**2 - predicted_xyz[0] ** 2
                if reach_y_sq < 0.0:
                    return None
                reach_y = np.sign(predicted_xyz[1]) * float(np.sqrt(reach_y_sq))
                target_xyz = predicted_xyz.copy()
                target_xyz[1] = reach_y
                wait_time = max(0.0, (obj_xyz[1] - reach_y) / speed - trajectory.duration)

        q_goal = self._inverse_kinematics_vertical_grasp_xyz(target_xyz)
        if q_goal is None:
            return None
        trajectory = self._build_bangbang_move(q_goal)
        if trajectory is None:
            return None

        predicted_y = obj_xyz[1] - speed * (trajectory.duration + prediction_pad)
        # Projected-position feasibility, mirroring the push chain's
        # `_push_cone_target_y` (`obj_y < eff_lower -> None`): if the object will
        # be at/below the graspable forward window by the time the EE arrives, the
        # grasp is doomed. Reject HERE (return None) instead of clamping to the
        # back of the reach disk and returning a plan that aims behind the base.
        # Because this builder is the single source for both the feasibility
        # screen (`_slot_throw_feasible` bootstrap) and the executor
        # (`_execute_initial_approach`), the mask and execution agree on the
        # PROJECTED position -- no current-position band-aid needed. (0.05 = the
        # same threshold the old current-y gate used.)
        if predicted_y <= 0.05:
            return None
        if target_xyz[1] < predicted_y and speed > 0.0:
            wait_time = max(wait_time, (predicted_y - target_xyz[1]) / speed)
        if wait_time > 0.0:
            stretched_trajectory = self._build_bangbang_move(
                q_goal,
                duration_override=float(trajectory.duration + wait_time),
            )
            if stretched_trajectory is not None:
                trajectory = stretched_trajectory
                wait_time = 0.0
        return target_xyz, trajectory, wait_time

    def _open_loop_grasp_window(self, slot: Any) -> bool:
        """Terminal grasp acquisition, mirroring the push STRIKE's execution-time
        wait (`_execute_push_transition_with_next`). The EE is already parked at
        the predicted grasp pose; enable suction and -- from the object's LIVE
        position -- compute how long until it reaches the grip site along the belt
        (`wait_time`, the grasp analog of push's `release_y`/`wait_time`, with no
        contact/strike term since the EE is AT the grasp point, not hovering).
        Hold (suction on) through that arrival plus a short capture window,
        attaching the instant the object enters the suction window. No corrective
        re-aim, no retry, no closed-loop tracking. Returns whether an object is
        held at the end."""
        if not slot.active:
            return False
        self._set_gripper_target_slot(slot)
        self._hold_current_robot_pose()
        self.gripper.set_enabled(True)

        speed = max(float(self._slot_forward_speed(slot)), 1e-6)
        wait_time = 0.0
        grip_xyz = self._forward_kinematics_xyz(
            self.env.get_joint_positions(self.joint_names).astype(np.float64)
        )
        if grip_xyz is not None:
            obj_y = float(self._slot_grasp_throw_xyz(slot)[1])
            grip_y = float(grip_xyz[1])
            wait_time = min(max(0.0, (obj_y - grip_y) / speed), float(self.action_wait_timeout))

        # Cross-belt (x) correction. The object's x is CONSTANT after spawn, but
        # the long bang-bang route from the awkward push-end can leave the EE a
        # few cm off the object's lane in x (and "hold current pose" would freeze
        # it there). So command the grasp pose at the object's KNOWN x (keeping the
        # current grasp y / grasp height) and HOLD that during the wait. One-time
        # command to the known-correct target -- not closed-loop tracking; the
        # y-wait still brings the object in. Fixes the cross-belt jams that kept
        # push->throw objects ~3-5 cm off-center, just outside the suction window.
        hold_q = self.env.get_joint_positions(self.joint_names).astype(np.float64)
        if grip_xyz is not None:
            corrected_xyz = np.array(
                [float(self._slot_grasp_throw_xyz(slot)[0]),
                 float(grip_xyz[1]),
                 float(self.action_grasp_height)],
                dtype=np.float64,
            )
            q_corr = self._inverse_kinematics_vertical_grasp_xyz(corrected_xyz)
            if q_corr is not None:
                hold_q = q_corr

        # ---- diagnostic (observation-only): how far did the route's endpoint
        # (intended parked grasp pose, stashed by the executor) miss, and how
        # close did the object center get to the grip site during the dwell?
        intended = getattr(self, "_next_grasp_intended", None)
        kind = "bootstrap"
        exec_err = None
        horizon_after = None   # after_traj.duration used in the projection
        horizon_true = None    # actual sim time from projection-read to here
        horizon_gap = None     # true - after  (the unmodeled term)
        proj_speed = None      # capped next_speed the projection used
        if (intended is not None and slot.spawn_id is not None
                and int(intended[0]) == int(slot.spawn_id) and grip_xyz is not None):
            kind = str(intended[2])
            exec_err = (np.asarray(grip_xyz, dtype=np.float64)
                        - np.asarray(intended[1], dtype=np.float64)).tolist()
            if len(intended) >= 5:
                horizon_after = float(intended[4])
                horizon_true = float(self.env.sim_time) - float(intended[3])
                horizon_gap = horizon_true - horizon_after
            proj_speed = float(intended[5]) if len(intended) >= 6 else None
        elif intended is not None:
            kind = str(intended[2]) + "?"
        self._next_grasp_intended = None  # consume
        start_obj_off = None  # signed (obj - ee) at window start, throw frame
        if grip_xyz is not None:
            try:
                start_obj_off = (self._world_to_throw_xyz(self._slot_center_world_xyz(slot))
                                 - np.asarray(grip_xyz, dtype=np.float64)).tolist()
            except Exception:
                start_obj_off = None
        min_horiz = float("inf")
        min_obj_off = None
        # ----------------------------------------------------------------------

        capture_window = 0.2  # ~time the object is within the suction window
        max_steps = max(1, int((wait_time + capture_window) / self.env.dt))
        for _ in range(max_steps):
            if self.gripper.is_attached():
                break
            if self.gripper.passes_geometry():
                # Grasp instant: the object has entered the suction window. Roll
                # ONCE (latched) whether the suction grips -- Bernoulli(suction_p).
                # On success attach; on failure stop trying (-> empty throw), never
                # re-grip on a later tick. Statistically exactly p among graspable.
                if self._suction_takes(slot):
                    self.gripper.try_attach()
                break
            try:
                ee = self._forward_kinematics_xyz(
                    self.env.get_joint_positions(self.joint_names).astype(np.float64))
                if ee is not None and slot.active:
                    off = (self._world_to_throw_xyz(self._slot_center_world_xyz(slot))
                           - np.asarray(ee, dtype=np.float64))
                    horiz = float(np.linalg.norm(off[:2]))
                    if horiz < min_horiz:
                        min_horiz = horiz
                        min_obj_off = off.tolist()
            except Exception:
                pass
            self.controller.hold_position(hold_q)
            self._advance_simulation(1)
            if not slot.active:
                break
        attached = self._slot_is_attached(slot)
        self._last_grasp_diag = {
            "kind": kind,
            "exec_err": exec_err,
            "obj_off_closest": min_obj_off,
            "start_obj_off": start_obj_off,
            "wait_time": float(wait_time),
            "min_horiz": None if min_horiz == float("inf") else min_horiz,
            "horizon_after": horizon_after,
            "horizon_true": horizon_true,
            "horizon_gap": horizon_gap,
            "obj_speed": float(speed),  # measured forward speed at grasp-open
            "proj_speed": proj_speed,   # capped speed the projection used
            "attached": bool(attached),
        }
        return attached

    def _dwell_for_grasp_handoff(self, next_primitive: int | None) -> None:
        """If a push->throw after-trajectory stashed a grasp wait, dwell it now
        (mirrors throw->throw's wait dwell in `_execute_throw_transition`) so the
        next object is pre-positioned upstream before `_open_loop_grasp_window`
        opens. Consumes the stash."""
        gw = getattr(self, "_after_traj_grasp_wait_time", None)
        self._after_traj_grasp_wait_time = None
        if (next_primitive is not None
                and int(next_primitive) == self.ACTION_THROW
                and gw is not None and gw > self.env.dt):
            self._hold_current_robot_pose()
            self._advance_simulation(max(1, int(gw / self.env.dt)))

    def _execute_initial_approach(self, slot: Any) -> tuple[bool, dict[str, Any]]:
        if not slot.active:
            return False, {"reason": "inactive_slot"}

        self._set_pusher_collision_enabled(False)
        obj_xyz = self._slot_grasp_throw_xyz(slot)
        object_speed = self._slot_execution_speed(slot)
        # Feasibility is now decided on the PROJECTED position inside
        # `_build_initial_approach_plan` (it returns None when the object will be
        # past the graspable window by arrival), so the old current-position gate
        # (`obj_xyz[1] <= 0.05`) is redundant -- a doomed object surfaces as a
        # None plan -> `initial_approach_plan_failed` below.
        plan = self._build_initial_approach_plan(obj_xyz, object_speed=object_speed)
        if plan is None:
            return False, {"reason": "initial_approach_plan_failed", "obj_xyz": obj_xyz}
        target_xyz, trajectory, idle_time = plan

        self._execute_trajectory(trajectory)
        if idle_time > 0.0:
            self._hold_current_robot_pose()
            self._advance_simulation(max(1, int(idle_time / self.env.dt)))

        # Terminal grasp during the route: push-style execution-time wait (see
        # `_open_loop_grasp_window`). The route SUCCEEDS regardless of attach --
        # the next step fires the throw either way (always-throw; no abort here).
        attached = self._open_loop_grasp_window(slot)
        return True, {
            "reason": "initial_approach_executed",
            "current_spawn_id": slot.spawn_id,
            "obj_xyz": obj_xyz,
            "target_xyz": target_xyz,
            "object_speed": object_speed,
            "wait_time": float(trajectory.duration + max(0.0, idle_time)),
            "idle_time": max(0.0, idle_time),
            "approach_duration": trajectory.duration,
            "object_attached": bool(attached),
        }

    def _execute_ready_grasp(self, slot: Any) -> tuple[bool, dict[str, Any]]:
        if not slot.active:
            return False, {"reason": "inactive_slot"}

        self._set_pusher_collision_enabled(False)
        # Terminal grasp: push-style execution-time wait (see
        # `_open_loop_grasp_window`) -- recompute the wait until the object
        # reaches the grip site from its LIVE position, hold through arrival, then
        # capture. The caller fires the throw regardless of the result (always-
        # throw), so a miss is simply an empty throw.
        attached = self._open_loop_grasp_window(slot)
        return attached, {
            "reason": "ready_grasp" if attached else "ready_grasp_empty",
            "current_spawn_id": slot.spawn_id if slot.active else None,
            "object_attached": bool(attached),
        }

    def _hold_current_robot_pose(self) -> np.ndarray:
        q_current = self.env.get_joint_positions(self.joint_names)
        self.controller.hold_position(q_current)
        return q_current

    def _belt_actuator_command(self) -> float:
        return self.belt_speed * self.belt_actuator_speed_scale

    def _apply_conveyor_object_motion(self) -> None:
        # When the belt is physically simulated, friction carries the objects
        # and the qvel override below would fight the physics. Return early so
        # the belt actuator + contact force drive the motion.
        if getattr(self, "use_physical_belt", False):
            return
        # Reset BOTH linear (qvel[0:3]) and angular (qvel[3:6]) components every
        # tick to mirror PyBullet's `p.resetBaseVelocity(linearVelocity=...)`,
        # which defaults angular to zero. Without zeroing angular, friction at
        # the contact patch torques the box about the x-axis and the cube
        # rolls forward.
        target_velocity_6d = np.array(
            [0.0, -self.belt_speed, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        for slot in self.object_slots:
            if not slot.active or getattr(slot, "manipulated", False):
                continue
            if self._slot_is_attached(slot):
                continue
            # Skip the slot currently being pushed; otherwise the qvel reset
            # would erase the pusher's lateral contact force every tick and
            # the box could never actually move sideways.
            if getattr(self, "_pushing_slot", None) is slot:
                continue
            joint_id = self.env.require_joint(slot.freejoint_name)
            qvel_adr = self.env.model.jnt_dofadr[joint_id]
            self.env.data.qvel[qvel_adr : qvel_adr + 6] = target_velocity_6d

    def _advance_simulation(self, nstep: int = 1) -> None:
        for _ in range(int(nstep)):
            self.env.set_actuator_ctrl("slider_act", self._belt_actuator_command())
            self._apply_conveyor_object_motion()
            self.env.step()
            self._mark_missed_objects()
            self._despawn_finished_objects()
            self._maybe_spawn_object()
            if self._sim_step_hook is not None:
                should_continue = self._sim_step_hook()
                if should_continue is False:
                    return
            if self.step_sleep > 0.0:
                time.sleep(self.step_sleep)

    def set_sim_step_hook(self, hook: Callable[[], bool] | None) -> None:
        self._sim_step_hook = hook

    def _selected_slot_from_action(self, action: int | np.integer | None) -> Any | None:
        ordered_slots = self._ordered_active_slots()
        if not ordered_slots:
            return None
        action_idx = 0 if action is None else int(action)
        if action_idx < 0 or action_idx >= len(ordered_slots):
            return None
        return ordered_slots[action_idx]

    def _execute_throw_transition(
        self,
        current_slot: Any,
        next_slot: Any | None,
        next_primitive: int | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        slot = current_slot
        if not slot.active:
            return False, {"reason": "inactive_slot"}
        self._set_pusher_collision_enabled(False)
        if next_slot is not None and not next_slot.active:
            return False, {"reason": "inactive_next_slot"}
        # Always throw -- matches the real robot, which has no grasp-success check.
        # Make one last best-effort attach, then fire the throw regardless of
        # whether an object is held (a miss => empty throw, scored 0 by the
        # outcome reward). `threw_loaded` records whether an object was actually
        # on the gripper at fire time (observation-only, for measurement).
        if not self._slot_is_attached(slot):
            self._attach_slot_with_suction(slot)
        threw_loaded = bool(self._slot_is_attached(slot))

        current_spawn_id = slot.spawn_id
        q_start = self.env.get_joint_positions(self.joint_names).astype(np.float64)
        obj_xyz = self._forward_kinematics_xyz(q_start)
        if obj_xyz is None:
            return False, {"reason": "forward_kinematics_failed"}

        def _standby_throw_plan() -> tuple[np.ndarray, np.ndarray, float] | None:
            next_obj_xyz = self.action_single_object_standby_xyz.copy()
            model_result = self._throwing_model_parameters(
                obj_xyz,
                next_obj_xyz,
                target_xy=self._slot_target_xy(slot),
            )
            if model_result is None:
                return None
            trajectory_parameters, trajectory_duration = model_result
            return next_obj_xyz, trajectory_parameters, trajectory_duration

        # (`_find_alternate_throw_prediction` -- the executor-side rescue that
        #  silently substituted a different slot when the agent's choice was
        #  infeasible -- was removed in the 2026-05-31 RL screening refactor.
        #  Action feasibility is now an env-level CONTRACT exposed via
        #  `info["action_mask"]`, so the agent never picks an infeasible
        #  (slot, primitive). The fallback paths below collapse to the standby
        #  throw, which is the true safety net for contract bugs.)

        # Planned obj position at handoff time. Only set when chaining into push.
        planned_next_obj_xyz: np.ndarray | None = None
        if next_slot is None:
            # Agent chose skip (or chain target was dropped). Throw to standby.
            standby_plan = _standby_throw_plan()
            if standby_plan is None:
                return False, {"reason": "trajectory_build_failed", "obj_xyz": obj_xyz}
            next_obj_xyz, trajectory_parameters, trajectory_duration = standby_plan
            wait_time = None
            next_spawn_id = None
            next_primitive_name = None
        else:
            next_obj_speed = self._slot_planning_speed(next_slot)
            if next_primitive is not None and int(next_primitive) == self.ACTION_PUSH:
                next_obj_xyz = self._slot_grasp_throw_xyz(next_slot).copy()
                prediction = self._predict_next_push_hover_handoff(
                    obj_xyz,
                    next_obj_xyz,
                    current_target_xy=self._slot_target_xy(slot),
                    next_target_xy=self._slot_target_xy(next_slot),
                    next_obj_speed=next_obj_speed,
                    seed_q=q_start,
                )
            else:
                next_obj_xyz = self._next_action_xyz(next_slot, next_primitive=next_primitive)
                prediction = self._predict_next_grasp_xyz(
                    obj_xyz,
                    next_obj_xyz,
                    target_xy=self._slot_target_xy(slot),
                    next_obj_speed=next_obj_speed,
                    arrival_y_offset=self._throw_handoff_y_offset(next_obj_speed),
                )
            if prediction is None:
                # Contract violation: the action mask said next_slot was feasible
                # but the prediction now fails. Fall to standby (safety net).
                standby_plan = _standby_throw_plan()
                if standby_plan is None:
                    return False, {"reason": "trajectory_prediction_failed", "obj_xyz": obj_xyz, "next_obj_xyz": next_obj_xyz}
                next_obj_xyz, trajectory_parameters, trajectory_duration = standby_plan
                wait_time = None
                neg_wait_time = None
                next_spawn_id = None
                next_primitive_name = None
            else:
                # Push prediction returns 6-tuple (incl. planned_obj_xyz);
                # grasp prediction returns 5-tuple. Handle both.
                if next_primitive is not None and int(next_primitive) == self.ACTION_PUSH:
                    next_obj_xyz, trajectory_parameters, trajectory_duration, wait_time, neg_wait_time, planned_next_obj_xyz = prediction
                else:
                    next_obj_xyz, trajectory_parameters, trajectory_duration, wait_time, neg_wait_time = prediction
                if neg_wait_time is not None:
                    # Contract violation: agent's choice reports neg_wait_time
                    # (object will arrive too early). Fall to standby.
                    standby_plan = _standby_throw_plan()
                    if standby_plan is None:
                        return False, {
                            "reason": "predicted_next_object_missed",
                            "obj_xyz": obj_xyz,
                            "next_obj_xyz": next_obj_xyz,
                            "neg_wait_time": neg_wait_time,
                        }
                    next_obj_xyz, trajectory_parameters, trajectory_duration = standby_plan
                    wait_time = None
                    next_spawn_id = None
                    next_primitive_name = None
                    planned_next_obj_xyz = None
                else:
                    next_spawn_id = next_slot.spawn_id
                    next_primitive_name = self._action_primitive_name(
                        self.ACTION_THROW if next_primitive is None else next_primitive
                    )

        # Determine the chained-into primitive so the throw trajectory can end
        # in the right pose (vertical-grasp for next=throw, behind-obj at obj
        # height with push orientation for next=push).
        next_primitive_for_traj: int | None = None
        next_target_xy_for_traj: np.ndarray | None = None
        if next_slot is not None:
            if next_primitive_name == self._action_primitive_name(self.ACTION_PUSH):
                next_primitive_for_traj = self.ACTION_PUSH
                next_target_xy_for_traj = self._slot_target_xy(next_slot)
            else:
                next_primitive_for_traj = self.ACTION_THROW

        trajectory = self._build_throwing_trajectory(
            q_start,
            obj_xyz,
            next_obj_xyz,
            target_xy=self._slot_target_xy(slot),
            trajectory_parameters=trajectory_parameters,
            next_primitive=next_primitive_for_traj,
            next_target_xy=next_target_xy_for_traj,
            next_obj_actual_xyz=planned_next_obj_xyz,
        )
        if trajectory is None:
            return False, {"reason": "trajectory_build_failed", "obj_xyz": obj_xyz}
        if not self._trajectory_is_safe(trajectory):
            return False, {"reason": "unsafe_trajectory", "obj_xyz": obj_xyz, "next_obj_xyz": next_obj_xyz}

        release_fraction = self._release_fraction_from_parameters(trajectory_parameters)
        release_time = float(np.clip(release_fraction, 0.0, 1.0) * trajectory.duration)

        def _release_current_object() -> None:
            self.gripper.set_enabled(False)
            # Once thrown, the object should be governed by physics rather than conveyor kinematics.
            slot.manipulated = True
            slot.manipulated_time = float(self.env.sim_time)

        self._execute_trajectory(
            trajectory,
            release_time=release_time,
            release_callback=_release_current_object,
        )
        if wait_time is not None and wait_time > 0.0:
            self._hold_current_robot_pose()
            self._advance_simulation(max(1, int(wait_time / self.env.dt)))

        self._hold_current_robot_pose()
        # A throw->push chain now ends the EE at the cone HOVER (Edit B) with the
        # strike target predicted by `_predict_next_push_hover_handoff`, so we
        # STASH it exactly like route->push / push->push: the next push runs the
        # matured `used_cone=True` fixed vertical strike (ONE predictor
        # everywhere), not the old interception solver. For throw->throw and the
        # standby/fallback paths (planned_next_obj_xyz / next_spawn_id None), we
        # clear so `used_cone` correctly disarms.
        if (
            next_primitive_name == self._action_primitive_name(self.ACTION_PUSH)
            and planned_next_obj_xyz is not None
            and next_spawn_id is not None
        ):
            self._push_routed_spawn_id = int(next_spawn_id)
            self._push_routed_target_xyz = np.asarray(planned_next_obj_xyz, dtype=np.float64).copy()
        else:
            self._push_routed_spawn_id = None
            self._push_routed_target_xyz = None
        # Diagnostic stash: intended parked grasp target for a throw->throw chain,
        # consumed once by the next step's _open_loop_grasp_window (observation-only).
        if (
            next_primitive_name == self._action_primitive_name(self.ACTION_THROW)
            and next_obj_xyz is not None
            and next_spawn_id is not None
        ):
            self._next_grasp_intended = (
                int(next_spawn_id),
                np.asarray(next_obj_xyz, dtype=np.float64).copy(),
                "throw->throw",
                float(self.env.sim_time),       # t_proj (~projection read time)
                float(trajectory_duration),     # after_dur used in the projection
                float(self._slot_planning_speed(next_slot)),  # capped speed used
            )
        return True, {
            "reason": "executed",
            "current_spawn_id": current_spawn_id,
            "next_spawn_id": next_spawn_id,
            "obj_xyz": obj_xyz,
            "next_obj_xyz": next_obj_xyz,
            "next_primitive_name": next_primitive_name,
            "trajectory_duration": trajectory_duration,
            "release_fraction": release_fraction,
            "release_time": release_time,
            "wait_time": wait_time,
            "threw_loaded": threw_loaded,
            "grasp_diag": getattr(self, "_last_grasp_diag", None),
        }

    def _execute_push_transition(self, current_slot: Any, next_slot: Any | None) -> tuple[bool, dict[str, Any]]:
        return self._execute_push_transition_with_next(current_slot, next_slot, next_primitive=None)

    def _execute_push_transition_with_next(
        self,
        current_slot: Any,
        next_slot: Any | None,
        next_primitive: int | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        slot = current_slot
        if not self.pushing_enabled or self.pushing_trajectory_generator is None:
            return False, {"reason": "pushing_disabled"}
        if not slot.active:
            return False, {"reason": "inactive_slot"}
        # Capture EE pose at primitive entry, before any motion is commanded.
        # Used by tmp_push_feasibility_analysis.py to classify what state the EE
        # came in at (standby vs chained vs mid-action).
        try:
            ee_xyz_at_push_primitive_start = np.asarray(
                self.env.get_site_position("grip_site"), dtype=np.float64
            ).copy()
        except Exception:
            ee_xyz_at_push_primitive_start = None
        self._log_chain(
            "push_entry",
            current_slot=self._slot_id(current_slot),
            next_slot=self._slot_id(next_slot),
            next_primitive=("push" if next_primitive == self.ACTION_PUSH else
                            ("throw" if next_primitive == self.ACTION_THROW else "none")),
        )
        # NOTE: pusher collision is enabled LATER, just before the strike -- not
        # here -- so the elevated pusher does not brush the object while it drifts
        # under the hover during the cone wait (NT_PUSHER_BRUSHED / NEIGHBOR).

        current_spawn_id = slot.spawn_id
        obj_xyz = self._slot_grasp_throw_xyz(slot)
        object_speed = self._slot_forward_speed(slot)
        if float(obj_xyz[1]) <= 0.05:
            self._set_pusher_collision_enabled(False)
            return False, {"reason": "object_past_push_window", "obj_xyz": obj_xyz}
        initial_world_pos = self._slot_center_world_xyz(slot)
        q_start = self.env.get_joint_positions(self.joint_names).astype(np.float64)
        target_xy = self._slot_target_xy(slot)
        # Push parameters for the wait / contact-lead timing.
        _pd = max(float(self.pushing_params.get("distance", 0.15)), 0.0)
        _sm = max(float(self.pushing_params.get("start_margin", 0.10)), 0.0)
        _pv = max(float(self.pushing_params.get("velocity", 1.4)), 1e-3)
        line_duration = (_sm + _pd) / _pv
        start_margin_time = _sm / _pv

        # Plan source.
        #   used_cone: the previous trajectory routed the EE to THIS slot's
        #     cone-edge hover. At execution (the previous trajectory has now
        #     ended) we recompute the cone from the object's CURRENT position and
        #     decide per jhsong's spec:
        #       - past the downstream cone edge  -> screen out (skip; un-pushable)
        #       - within the cone                -> strike NOW, aiming at where the
        #                                           object will be after the strike's
        #                                           own contact-lead (descent + margin)
        #       - upstream of the cone           -> WAIT until it drifts to the
        #                                           strike point, then strike.
        #     The strike aims at the object's projected-at-CONTACT position
        #     computed HERE (at strike time), not the stale route-time edge --
        #     that staleness was the miss (object had drifted ~0.3 m past the
        #     route-time edge by the time the strike fired).
        #   else: un-routed (external/RL policy pushed without a route step) ->
        #     fall back to the interception solver from the current EE pose.
        used_cone = (
            self._push_routed_spawn_id is not None
            and slot.spawn_id is not None
            and int(slot.spawn_id) == int(self._push_routed_spawn_id)
            and getattr(self, "_push_routed_target_xyz", None) is not None
        )
        routed_target = getattr(self, "_push_routed_target_xyz", None)
        self._push_routed_spawn_id = None  # consume
        self._push_routed_target_xyz = None
        if used_cone:
            # FIXED strike to the STASHED predicted contact target the hover was
            # placed above (by `_predict_push_hover`). The hover already encodes
            # the full drift (route + descent + margin), so the strike just aims
            # there -- no strike-time cone recompute, no re-aim. A short WAIT is
            # the ONLY adjustment: if the object hasn't yet drifted to the strike
            # point (still upstream), hold until it arrives; never re-aim.
            strike_obj_xyz = np.asarray(routed_target, dtype=np.float64).copy()
            _ws = max(float(object_speed), 1e-6)
            _probe = self._build_pushing_trajectory(q_start, strike_obj_xyz, target_xy=target_xy)
            if _probe is None:
                self._set_pusher_collision_enabled(False)
                return False, {
                    "reason": "object_downstream_of_cone",
                    "current_spawn_id": current_spawn_id,
                    "obj_xyz": obj_xyz,
                }
            _ct = max(0.0, float(_probe.duration) - line_duration)
            # Object arrives at the strike point after drifting (ct + margin); if
            # it is still upstream of that release point, wait the difference.
            release_y = float(strike_obj_xyz[1]) + _ws * (_ct + start_margin_time)
            wait_time = min(max(0.0, (float(obj_xyz[1]) - release_y) / _ws),
                            float(self.action_wait_timeout))
            if wait_time > self.env.dt:
                self._hold_current_robot_pose()
                self._advance_simulation(max(1, int(wait_time / self.env.dt)))
                if not slot.active:
                    self._set_pusher_collision_enabled(False)
                    return False, {"reason": "inactive_slot"}
                object_speed = self._slot_forward_speed(slot)
                initial_world_pos = self._slot_center_world_xyz(slot)
                q_start = self.env.get_joint_positions(self.joint_names).astype(np.float64)
            trajectory = self._build_pushing_trajectory(
                q_start, strike_obj_xyz, target_xy=target_xy
            )
            push_motion = self._push_motion_from_obj_xyz(
                strike_obj_xyz, initial_world_pos[2], target_xy=target_xy
            )
            if trajectory is None or push_motion is None:
                self._set_pusher_collision_enabled(False)
                return False, {
                    "reason": "object_downstream_of_cone",
                    "current_spawn_id": current_spawn_id,
                    "obj_xyz": obj_xyz,
                }
            planned_obj_xyz = strike_obj_xyz
            contact_time = max(0.0, float(trajectory.duration) - line_duration)
        else:
            plan = self._build_predicted_pushing_plan(
                q_start, obj_xyz, initial_world_pos[2],
                target_xy=target_xy, object_speed=object_speed,
            )
            if plan is None:
                self._set_pusher_collision_enabled(False)
                return False, {
                    "reason": "pushing_trajectory_build_failed",
                    "current_spawn_id": current_spawn_id,
                    "obj_xyz": obj_xyz,
                }
            planned_obj_xyz, trajectory, push_motion, contact_time = plan
        if float(planned_obj_xyz[1]) <= -0.16:
            self._set_pusher_collision_enabled(False)
            return False, {
                "reason": "object_past_push_window",
                "current_spawn_id": current_spawn_id,
                "obj_xyz": obj_xyz,
                "planned_obj_xyz": planned_obj_xyz,
                "predicted_contact_time": contact_time,
            }
        if not self._trajectory_is_safe(trajectory):
            self._set_pusher_collision_enabled(False)
            return False, {
                "reason": "unsafe_pushing_trajectory",
                "obj_xyz": obj_xyz,
                "planned_obj_xyz": planned_obj_xyz,
            }
        push_start_world, push_target_world, push_dir = push_motion

        if self._slot_is_attached(slot):
            self.gripper.set_enabled(False)
            self._advance_simulation(max(1, int(0.05 / self.env.dt)))

        # Enable the pusher collision now -- after any cone wait -- so the strike
        # engages the object but the elevated hover did not brush it while it
        # drifted in during the wait.
        self._set_pusher_collision_enabled(True)
        # Mark this slot as the active push target so the per-tick conveyor
        # qvel override skips it and the pusher's lateral contact force
        # actually moves the box.
        previous_pushing_slot = getattr(self, "_pushing_slot", None)
        self._pushing_slot = slot
        # Under kinematic belt (use_physical_belt=False, legacy): zero the
        # box's velocity at the start of the push so the residual belt-speed
        # momentum doesn't carry it away from the pusher's leading face during
        # the brief push window. After this, qvel is left to physics so the
        # pusher's contact force can actually accelerate it.
        # Under physical belt: the obj is being friction-driven by the belt;
        # zeroing qvel would jerk it to a stop unrealistically. Leave it alone
        # so the pusher engages a moving obj (which the factor=1 predictor in
        # _build_predicted_pushing_plan now aims at correctly).
        if slot.active and not getattr(self, "use_physical_belt", False):
            joint_id = self.env.require_joint(slot.freejoint_name)
            qvel_adr = self.env.model.jnt_dofadr[joint_id]
            self.env.data.qvel[qvel_adr : qvel_adr + 6] = 0.0
        # Diagnostic capture for predictor-mistime analysis. T_bang = end of
        # bang-bang (EE at push_start). T_meet = T_bang + start_margin/push_velocity
        # is the instant EE first reaches planned_obj_xyz along the push line.
        push_start_sim_time = float(self.env.sim_time)
        _push_start_margin_param = float(self.pushing_params.get("start_margin", 0.17))
        _push_velocity_param = max(float(self.pushing_params.get("velocity", 1.4)), 1e-6)
        line_start_margin_dur = _push_start_margin_param / _push_velocity_param
        t_bang_target = push_start_sim_time + float(contact_time)
        t_meet_target = t_bang_target + line_start_margin_dur
        captured_obj_at_bang_end: np.ndarray | None = None
        captured_obj_at_meet: np.ndarray | None = None
        captured_ee_at_meet: np.ndarray | None = None
        # Bang-bang incidental-contact detector: count pusher_collision <->
        # slot.geom contacts that occur before T_bang. Hypothesis: long
        # bang-bang trajectories swing the pusher tool past the obj on the
        # approach, brushing it and imparting unintended momentum.
        bang_bang_contact_count = 0           # pusher vs target slot (legacy)
        bang_bang_neighbor_contact_count = 0  # pusher vs non-target box
        bang_bang_target_neighbor_collision_count = 0  # target box vs non-target box
        pusher_geom_id = None
        slot_geom_id = None
        neighbor_geom_ids: set[int] = set()  # all red_box geoms except target
        pusher_name = getattr(self, "pusher_collision_name", None)
        slot_geom_name = getattr(slot, "geom_name", None)
        if pusher_name is not None and slot_geom_name is not None:
            try:
                pusher_geom_id = int(self.env.require_geom(pusher_name))
                slot_geom_id = int(self.env.require_geom(slot_geom_name))
                # Enumerate all red_box geoms (naming: red_box_geom, red_box_geom_2..12)
                for candidate in ["red_box_geom"] + [f"red_box_geom_{i}" for i in range(2, 13)]:
                    if candidate == slot_geom_name:
                        continue
                    try:
                        gid = int(self.env.require_geom(candidate))
                        neighbor_geom_ids.add(gid)
                    except Exception:
                        pass
            except Exception:
                pusher_geom_id = None
                slot_geom_id = None
                neighbor_geom_ids = set()
        try:
            self.controller.load_trajectory(trajectory, start_time=self.env.sim_time)
            while not self.controller.is_finished(self.env.sim_time):
                self.controller.update(self.env.sim_time)
                self._advance_simulation(1)
                # Scan contacts; only count during bang-bang phase.
                if (pusher_geom_id is not None and slot_geom_id is not None
                        and self.env.sim_time < t_bang_target):
                    ncon = int(self.env.data.ncon)
                    target_hit_this_tick = False
                    neighbor_hit_this_tick = False
                    target_neighbor_hit_this_tick = False
                    for ci in range(ncon):
                        c = self.env.data.contact[ci]
                        g1 = int(c.geom1)
                        g2 = int(c.geom2)
                        # pusher vs target
                        if ((g1 == pusher_geom_id and g2 == slot_geom_id)
                                or (g2 == pusher_geom_id and g1 == slot_geom_id)):
                            target_hit_this_tick = True
                        # pusher vs neighbor
                        elif ((g1 == pusher_geom_id and g2 in neighbor_geom_ids)
                                or (g2 == pusher_geom_id and g1 in neighbor_geom_ids)):
                            neighbor_hit_this_tick = True
                        # target vs neighbor (cascading collision)
                        elif ((g1 == slot_geom_id and g2 in neighbor_geom_ids)
                                or (g2 == slot_geom_id and g1 in neighbor_geom_ids)):
                            target_neighbor_hit_this_tick = True
                    if target_hit_this_tick:
                        bang_bang_contact_count += 1
                    if neighbor_hit_this_tick:
                        bang_bang_neighbor_contact_count += 1
                    if target_neighbor_hit_this_tick:
                        bang_bang_target_neighbor_collision_count += 1
                if captured_obj_at_bang_end is None and self.env.sim_time >= t_bang_target:
                    captured_obj_at_bang_end = self._slot_center_world_xyz(slot).copy()
                if captured_obj_at_meet is None and self.env.sim_time >= t_meet_target:
                    captured_obj_at_meet = self._slot_center_world_xyz(slot).copy()
                    captured_ee_at_meet = np.asarray(
                        self.env.get_site_position("grip_site"), dtype=np.float64
                    ).copy()
            self.controller.hold_position(trajectory.positions[-1])
        finally:
            self._pushing_slot = previous_pushing_slot

        # base_travel = 0 at EXECUTION time: the current push has already run, so
        # the after-traj (the route to the next hover) starts immediately -- there
        # is no extra upstream wait before it. `_predict_push_hover` inside the
        # after-traj already projects the next object forward by t_route (the
        # after-traj bang-bang) + t_strike (descent + start-margin), which is the
        # full time-to-contact. The old hand-tuned 0.15 s constant is removed --
        # it was compensating for the previous non-projecting after-traj and is
        # subsumed by the single predictor now.
        after_plan = self._build_pushing_after_trajectory(
            trajectory,
            next_slot,
            current_slot=slot,
            next_primitive=next_primitive,
            extra_projection_seconds=0.0,
        )
        # Note: under the 1-step-ahead action design, `next_slot` is the agent's
        # explicit chain choice (no selector hint / FIFO fallback). If it can't
        # be chained the after-traj falls back to standby and the next step is
        # route-only -- the agent's choice is the source of truth, so there is
        # no executor-side alternate-slot rescue scan here.
        if after_plan is None:
            self._set_pusher_collision_enabled(False)
            return False, {
                "reason": "pushing_after_trajectory_build_failed",
                "current_spawn_id": current_spawn_id,
                "obj_xyz": obj_xyz,
                "planned_obj_xyz": planned_obj_xyz,
            }
        after_trajectory, next_obj_xyz = after_plan
        # If the after-plan (still) fell back to standby (no alternate found),
        # drop the chain. The selector is responsible for avoiding this case
        # via _push_transition_margin / _push_chain_score; if it still happens
        # and no alternate exists, the next step starts fresh from standby.
        if (
            next_slot is not None
            and next_primitive is not None
            and int(next_primitive) == self.ACTION_PUSH
            and next_obj_xyz is not None
            and np.allclose(next_obj_xyz, self.action_single_object_standby_xyz, atol=1e-6)
        ):
            effective_next_slot = None
            effective_next_primitive = None
        else:
            effective_next_slot = next_slot
            effective_next_primitive = next_primitive

        # Diagnostic stash: intended parked grasp target for a push->throw chain
        # (next_obj_xyz from the THROW branch = planned grasp pose). Consumed once
        # by the next step's _open_loop_grasp_window (observation-only).
        if (
            effective_next_slot is not None
            and effective_next_primitive is not None
            and int(effective_next_primitive) == self.ACTION_THROW
            and next_obj_xyz is not None
            and effective_next_slot.spawn_id is not None
        ):
            self._next_grasp_intended = (
                int(effective_next_slot.spawn_id),
                np.asarray(next_obj_xyz, dtype=np.float64).copy(),
                "push->throw",
                float(self.env.sim_time),          # t_proj (~projection read time)
                float(after_trajectory.duration),  # after_dur used in the projection
                float(self._slot_planning_speed(effective_next_slot)),  # capped speed used
            )

        final_pos = self._slot_center_world_xyz(slot)
        joint_id = self.env.require_joint(slot.freejoint_name)
        qvel_adr = self.env.model.jnt_dofadr[joint_id]
        final_linear_velocity = self.env.data.qvel[qvel_adr : qvel_adr + 3].copy()
        displacement_xy = final_pos[:2] - push_start_world[:2]
        displacement = float(np.linalg.norm(displacement_xy))
        push_dir_xy = np.asarray(push_dir[:2], dtype=np.float64)
        belt_dir_xy = np.array([0.0, -1.0], dtype=np.float64)
        push_lateral_xy = push_dir_xy - belt_dir_xy * float(np.dot(push_dir_xy, belt_dir_xy))
        push_lateral_norm = float(np.linalg.norm(push_lateral_xy))
        if push_lateral_norm > 1e-6:
            push_lateral_unit = push_lateral_xy / push_lateral_norm
            robot_driven_displacement = float(np.dot(displacement_xy, push_lateral_unit))
            min_robot_driven_displacement = max(
                0.004,
                0.03 * float(self.pushing_params.get("distance", 0.15)) * push_lateral_norm,
            )
        else:
            robot_driven_displacement = float(np.dot(displacement_xy, push_dir_xy / (np.linalg.norm(push_dir_xy) + 1e-12)))
            min_robot_driven_displacement = max(0.004, 0.03 * float(self.pushing_params.get("distance", 0.15)))
        if robot_driven_displacement < min_robot_driven_displacement:
            self._set_pusher_collision_enabled(False)
            # Route the EE through the after-trajectory's hover_pos target so
            # the next push starts from a clean elevated pose. Without this,
            # the EE is left at push_end (low z) and the next push's bang-bang
            # sweeps across the belt at obj height, brushing whatever obj it
            # crosses. Multi-trial eval (2026-05-21, n=3 each) showed this
            # lifts conditional push success from 18.3%±4.1 to 24.7%±1.8
            # (+1.58σ) and tightens NO_TRANSFER variance from σ=21 to σ=3.
            if after_trajectory is not None:
                try:
                    self._execute_trajectory(after_trajectory)
                    self._dwell_for_grasp_handoff(effective_next_primitive)
                except Exception:
                    pass
            self._hold_current_robot_pose()
            # CHAIN CONTINUATION IS INDEPENDENT OF WHETHER THIS PUSH CONNECTED
            # (jhsong). The after-traj has already routed the EE toward the next
            # object's hover exactly as on the success path, so a missed push
            # must hand off the SAME chain-continuation bookkeeping -- otherwise
            # the chain silently breaks on every miss and the EE cold-starts from
            # standby next step (the vicious cycle: miss -> no chain -> standby ->
            # next push 1.5 s away -> miss). The ONLY thing a miss does NOT do is
            # mark the slot manipulated (it wasn't pushed).
            if (
                effective_next_slot is not None
                and effective_next_primitive is not None
                and int(effective_next_primitive) == self.ACTION_PUSH
                and effective_next_slot.spawn_id is not None
            ):
                self._push_routed_spawn_id = int(effective_next_slot.spawn_id)
                self._push_routed_target_xyz = getattr(self, "_after_traj_push_target_xyz", None)
            else:
                self._push_routed_spawn_id = None
                self._push_routed_target_xyz = None
            self._log_chain(
                "rdd_too_small_exit",
                current_slot=self._slot_id(current_slot),
                next_slot=self._slot_id(next_slot),
                rdd=f"{robot_driven_displacement:.4f}",
                min_rdd=f"{min_robot_driven_displacement:.4f}",
                ran_after_traj=("yes" if after_trajectory is not None else "no"),
            )
            return False, {
                "reason": "robot_driven_push_displacement_too_small",
                "current_spawn_id": current_spawn_id,
                "next_spawn_id": None if effective_next_slot is None else effective_next_slot.spawn_id,
                "next_primitive_name": None if effective_next_slot is None or effective_next_primitive is None else self._action_primitive_name(
                    effective_next_primitive
                ),
                "next_obj_xyz": next_obj_xyz,
                "obj_xyz": obj_xyz,
                "planned_obj_xyz": planned_obj_xyz,
                "object_speed": object_speed,
                "initial_world_xyz": initial_world_pos,
                "start_world_xyz": push_start_world,
                "final_world_xyz": final_pos,
                "target_world_xyz": push_target_world,
                "predicted_contact_time": contact_time,
                "push_dir": push_dir,
                "final_linear_velocity": final_linear_velocity,
                "displacement": displacement,
                "robot_driven_displacement": robot_driven_displacement,
                "min_robot_driven_displacement": min_robot_driven_displacement,
                "obj_xyz_at_bang_end": captured_obj_at_bang_end,
                "obj_xyz_at_meet": captured_obj_at_meet,
                "ee_xyz_at_meet": captured_ee_at_meet,
                "bang_bang_contact_count": bang_bang_contact_count,
                "bang_bang_neighbor_contact_count": bang_bang_neighbor_contact_count,
                "bang_bang_target_neighbor_collision_count": bang_bang_target_neighbor_collision_count,
                "ee_xyz_at_push_primitive_start": ee_xyz_at_push_primitive_start,
            }
        slot.manipulated = True
        slot.manipulated_time = float(self.env.sim_time)
        self._execute_trajectory(after_trajectory)
        self._dwell_for_grasp_handoff(effective_next_primitive)
        self._set_pusher_collision_enabled(False)
        self._hold_current_robot_pose()
        # If the after-traj routed the EE toward a NEXT push, the EE is now at
        # that slot's cone hover -> mark it so the next push step uses the cone
        # plan (clean vertical strike). Otherwise clear the marker.
        if (
            effective_next_slot is not None
            and effective_next_primitive is not None
            and int(effective_next_primitive) == self.ACTION_PUSH
            and effective_next_slot.spawn_id is not None
        ):
            self._push_routed_spawn_id = int(effective_next_slot.spawn_id)
            # Commit the predicted contact target the after-traj's _predict_push_hover
            # computed (transient stashed in _build_pushing_after_trajectory), so
            # the next push strikes a FIXED push to it (no strike-time re-plan).
            self._push_routed_target_xyz = getattr(self, "_after_traj_push_target_xyz", None)
        else:
            self._push_routed_spawn_id = None
            self._push_routed_target_xyz = None
        return True, {
            "reason": "pushed",
            "current_spawn_id": current_spawn_id,
            "next_spawn_id": None if effective_next_slot is None else effective_next_slot.spawn_id,
            "obj_xyz": obj_xyz,
            "planned_obj_xyz": planned_obj_xyz,
            "object_speed": object_speed,
            "next_obj_xyz": next_obj_xyz,
            "next_primitive_name": None if effective_next_slot is None or effective_next_primitive is None else self._action_primitive_name(
                effective_next_primitive
            ),
            "initial_world_xyz": initial_world_pos,
            "start_world_xyz": push_start_world,
            "final_world_xyz": final_pos,
            "target_world_xyz": push_target_world,
            "predicted_contact_time": contact_time,
            "push_dir": push_dir,
            "final_linear_velocity": final_linear_velocity,
            "displacement": displacement,
            "robot_driven_displacement": robot_driven_displacement,
            "min_robot_driven_displacement": min_robot_driven_displacement,
            "trajectory_duration": trajectory.duration,
            "after_trajectory_duration": after_trajectory.duration,
            "pushing_params": dict(self.pushing_params),
            "obj_xyz_at_bang_end": captured_obj_at_bang_end,
            "obj_xyz_at_meet": captured_obj_at_meet,
            "ee_xyz_at_meet": captured_ee_at_meet,
            "bang_bang_contact_count": bang_bang_contact_count,
            "ee_xyz_at_push_primitive_start": ee_xyz_at_push_primitive_start,
        }

    def _execute_action_transition(
        self,
        current_slot: Any,
        next_slot: Any | None,
        primitive: int,
        next_primitive: int | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        if int(primitive) == self.ACTION_PUSH:
            return self._execute_push_transition_with_next(current_slot, next_slot, next_primitive=next_primitive)
        return self._execute_throw_transition(current_slot, next_slot, next_primitive=next_primitive)

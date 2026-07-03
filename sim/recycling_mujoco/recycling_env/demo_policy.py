"""Demo action selector for `RecyclingBBoxGymEnv`.

This module is policy-only — it picks WHICH action to issue. It has no role
in executing actions; that's `RecyclingBBoxGymEnv.step()` and
`action_control.py`. When RL training replaces this layer, callers stop
importing this file.

Under the 1-step-ahead action design, `action[N] = (next_slot, next_primitive)`
specifies the object to manipulate NEXT. The env executes the pending
manipulation (set up by `action[N-1]`) and routes the EE toward `action[N]`'s
entry pose. So this selector:
  - At a route-only / bootstrap step (`env._pending_target_slot is None`):
    picks the FIRST object to manipulate.
  - Otherwise: given the pending object (`env._pending_target_slot`, manipulated
    THIS step with `env._pending_action_primitive`), scores candidate NEXT
    objects with `_select_{push,throw}_handoff_candidate` and emits the best.

The only demo-specific field written back is `_demo_next_primitive`, used to
alternate throw↔push. The chain target is now the action itself — there is no
side-channel hint and no FIFO fallback. Candidate indices are restricted to
`< env.max_detections` so every emitted action is valid in the Gym action
space (an RL policy sees the same constraint).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .gym_env import ObjectSlot, RecyclingBBoxGymEnv


class DemoSelector:
    def __init__(self, env: "RecyclingBBoxGymEnv"):
        self._env = env

    # ─── public entry point ──────────────────────────────────────────────

    def sample_action(self, push_probability: float = 0.5, alternate: bool = True) -> dict[str, int | None]:
        env = self._env
        # Primitive-fallback gate: only allow the selectors to try the OTHER
        # primitive when the demo really is mixed (0 < push_prob < 1). At the
        # pure endpoints (push_prob = 0 throw-only, push_prob = 1 push-only) we
        # want to honor the mode strictly -- no silent fallback to the other.
        env._demo_allow_primitive_fallback = bool(0.0 < float(push_probability) < 1.0)
        ordered_slots = env._ordered_active_slots()
        if not ordered_slots:
            return env._format_demo_action(env.max_detections, env.ACTION_THROW)

        pending_slot = env._pending_target_slot
        pending_primitive = env._pending_action_primitive
        if pending_slot is not None and not env._slot_in_action_window(pending_slot):
            pending_slot = None
            pending_primitive = None

        if pending_slot is None or pending_primitive is None:
            # ── Route-only / bootstrap step: choose the FIRST object to act on. ──
            return self._sample_bootstrap_action(ordered_slots, push_probability, alternate)

        # ── Chain step: the pending object is manipulated THIS step. Choose the
        # NEXT object to chain to and emit it as the action. ──
        desired_next_primitive = self._desired_next_primitive(
            pending_primitive, push_probability, alternate
        )
        if int(pending_primitive) == env.ACTION_PUSH:
            chosen_index, chosen_primitive = self._select_push_handoff_candidate(
                pending_slot, ordered_slots, desired_next_primitive,
            )
        else:
            chosen_index, chosen_primitive = self._select_throw_handoff_candidate(
                pending_slot, ordered_slots, desired_next_primitive,
            )

        if chosen_index is None:
            # No feasible next object. Emit skip: after the pending manipulation
            # the EE routes to standby and the next step is route-only.
            return env._format_demo_action(env.max_detections, env.ACTION_THROW)

        env._demo_next_primitive = chosen_primitive
        return env._format_demo_action(chosen_index, chosen_primitive)

    # ─── shared helpers ──────────────────────────────────────────────────

    def _desired_next_primitive(
        self, current_primitive: int, push_probability: float, alternate: bool
    ) -> int:
        """Primitive the demo policy WANTS for the next manipulation. With
        `alternate`, flip relative to the primitive being executed this step;
        otherwise sample by `push_probability`. Throw-only when pushing is
        disabled."""
        env = self._env
        if not env.pushing_enabled or env.pushing_trajectory_generator is None:
            return env.ACTION_THROW
        if alternate:
            return env.ACTION_THROW if int(current_primitive) == env.ACTION_PUSH else env.ACTION_PUSH
        return env.ACTION_PUSH if float(env._rng.random()) < float(push_probability) else env.ACTION_THROW

    def _sample_bootstrap_action(
        self, ordered_slots: list["ObjectSlot"], push_probability: float, alternate: bool
    ) -> dict[str, int | None]:
        """No pending manipulation (step 0 or cleared pending). Pick the first
        feasible object/primitive among the action-addressable slots
        (indices < max_detections)."""
        env = self._env
        if not env.pushing_enabled or env.pushing_trajectory_generator is None:
            primitive = env.ACTION_THROW
        elif alternate:
            primitive = int(env._demo_next_primitive)
        else:
            primitive = env.ACTION_PUSH if float(env._rng.random()) < float(push_probability) else env.ACTION_THROW

        addressable = min(len(ordered_slots), env.max_detections)
        primitives_to_try: tuple[int, ...] = (int(primitive),)
        if (
            env.pushing_enabled
            and env.pushing_trajectory_generator is not None
            and getattr(env, "_demo_allow_primitive_fallback", True)
        ):
            fallback = env.ACTION_THROW if int(primitive) == env.ACTION_PUSH else env.ACTION_PUSH
            primitives_to_try = (int(primitive), int(fallback))

        # Read the env's cached action mask (single source of truth, computed
        # once per step). Falls back to the live predicate if the cache is empty
        # (e.g. very first call before reset has populated it).
        flags = getattr(env, "_slot_feasibility_flags", None) or {}
        for prim in primitives_to_try:
            prim_name = "push" if int(prim) == env.ACTION_PUSH else "throw"
            for idx in range(addressable):
                # Obstacles belong to no bin -- never select one, even if the
                # cached mask flags are stale w.r.t. the current slot ordering.
                if getattr(ordered_slots[idx], "is_obstacle", False):
                    continue
                if flags:
                    feasible = bool(flags.get(idx, {}).get(prim_name, False))
                else:
                    feasible = self._demo_slot_supports_primitive(ordered_slots[idx], prim)
                if feasible:
                    env._demo_next_primitive = prim
                    return env._format_demo_action(idx, prim)
        # Nothing feasible right now → idle skip (route-only).
        return env._format_demo_action(env.max_detections, env.ACTION_THROW)

    def _demo_slot_supports_primitive(
        self,
        slot: "ObjectSlot",
        primitive: int,
        obj_xyz_override: np.ndarray | None = None,
    ) -> bool:
        # `obj_xyz_override` lets selection-time callers test feasibility at a
        # *projected* obj position instead of the slot's current position.
        # Selection (in _push_transition_margin) projects the next obj forward
        # by trajectory.duration + after_traj.duration; without using the same
        # predicate as the step-N+1 check, selection accepts chain candidates
        # whose projected position would fail the step-N+1 gate (predicted_y
        # passes the loose 0.05 window but _build_predicted_pushing_plan
        # rejects with plan[0][1] <= -0.16). That asymmetry is one of the
        # root causes of the 91% chain phenomenon rate.
        env = self._env
        if not slot.active or getattr(slot, "manipulated", False):
            return False
        if int(primitive) == env.ACTION_PUSH:
            obj_xyz = (
                np.asarray(obj_xyz_override, dtype=np.float64)
                if obj_xyz_override is not None
                else env._slot_grasp_throw_xyz(slot)
            )
            if float(obj_xyz[1]) <= 0.05:
                return False
            if not env.pushing_enabled or env.pushing_trajectory_generator is None:
                return False
            target_xy = env._slot_target_xy(slot)
            # Pushable only if the object is upstream-of-or-inside its bin cone
            # (|obj_y - bin_y| <= bin_x - obj_x). Seed the cone plan from the cone
            # HOVER pose (where the EE will be pre-positioned at execution), so
            # the feasibility test matches the clean-strike the push will do.
            cone = env._push_cone_target_y(obj_xyz, target_xy)
            if cone is None:
                return False
            cone_obj_xyz = obj_xyz.copy()
            cone_obj_xyz[1] = cone[0]
            hover_q = env._push_handoff_q_from_obj(cone_obj_xyz, target_xy=target_xy)
            if hover_q is None:
                return False
            start_world_z = env._slot_center_world_xyz(slot)[2]
            object_speed = env._slot_planning_speed(slot)
            plan = env._build_cone_pushing_plan(
                hover_q,
                obj_xyz,
                start_world_z,
                target_xy=target_xy,
                object_speed=object_speed,
            )
            return bool(plan is not None and env._trajectory_is_safe(plan[1]))
        obj_xyz = env._slot_grasp_throw_xyz(slot)
        if float(obj_xyz[1]) <= 0.05:
            return False
        plan = env._build_initial_approach_plan(obj_xyz, object_speed=env._slot_execution_speed(slot))
        if plan is None:
            return False
        return True

    # ─── throw selection (relocated, behavior unchanged) ─────────────────

    def _throw_transition_supports_next(
        self, current_slot: "ObjectSlot", next_slot: "ObjectSlot | None", next_primitive: int,
    ) -> bool:
        env = self._env
        if current_slot is None or not current_slot.active:
            return False
        if next_slot is None:
            return True
        q_start = env.env.get_joint_positions(env.joint_names).astype(np.float64)
        obj_xyz = env._forward_kinematics_xyz(q_start)
        if obj_xyz is None:
            return False
        next_speed = env._slot_planning_speed(next_slot)
        if int(next_primitive) == env.ACTION_PUSH:
            prediction = env._predict_next_push_hover_handoff(
                obj_xyz,
                env._slot_grasp_throw_xyz(next_slot).copy(),
                current_target_xy=env._slot_target_xy(current_slot),
                next_target_xy=env._slot_target_xy(next_slot),
                next_obj_speed=next_speed,
                seed_q=q_start,
            )
        else:
            prediction = env._predict_next_grasp_xyz(
                obj_xyz,
                env._slot_grasp_throw_xyz(next_slot).copy(),
                target_xy=env._slot_target_xy(current_slot),
                next_obj_speed=next_speed,
                arrival_y_offset=env._throw_handoff_y_offset(next_speed),
            )
        if prediction is None:
            return False
        predicted_next_xyz = np.asarray(prediction[0], dtype=np.float64)
        # neg_wait_time is at index 4 (push prediction returns 6-tuple incl.
        # planned_obj_xyz at index 5; grasp prediction returns 5-tuple).
        neg_wait_time = prediction[4]
        if neg_wait_time is not None:
            return False
        if int(next_primitive) == env.ACTION_THROW and (
            float(predicted_next_xyz[1]) <= 0.08 or float(predicted_next_xyz[1]) >= 0.45
        ):
            return False
        if int(next_primitive) == env.ACTION_PUSH:
            trajectory_duration = float(prediction[2])
            wait_time_val = float(prediction[3]) if prediction[3] is not None else 0.0
            # See `_throw_transition_margin` for the rationale: must account for
            # the push action's own sim time (~1 sec) before contact.
            push_action_dt_est = 1.0
            obj_y_at_push_time = (
                float(env._slot_grasp_throw_xyz(next_slot)[1])
                - next_speed * (trajectory_duration + wait_time_val)
            )
            obj_y_at_push_contact = obj_y_at_push_time - next_speed * push_action_dt_est
            if obj_y_at_push_contact <= -0.10:
                return False
            if float(predicted_next_xyz[1]) <= 0.05 or float(predicted_next_xyz[1]) >= 0.55:
                return False
        return True

    def _throw_transition_margin(
        self, current_slot: "ObjectSlot", next_slot: "ObjectSlot | None", next_primitive: int,
    ) -> float | None:
        env = self._env
        if current_slot is None or not current_slot.active or next_slot is None:
            return None
        q_start = env.env.get_joint_positions(env.joint_names).astype(np.float64)
        obj_xyz = env._forward_kinematics_xyz(q_start)
        if obj_xyz is None:
            return None
        next_speed = env._slot_planning_speed(next_slot)
        if int(next_primitive) == env.ACTION_PUSH:
            prediction = env._predict_next_push_hover_handoff(
                obj_xyz,
                env._slot_grasp_throw_xyz(next_slot).copy(),
                current_target_xy=env._slot_target_xy(current_slot),
                next_target_xy=env._slot_target_xy(next_slot),
                next_obj_speed=next_speed,
                seed_q=q_start,
            )
        else:
            prediction = env._predict_next_grasp_xyz(
                obj_xyz,
                env._slot_grasp_throw_xyz(next_slot).copy(),
                target_xy=env._slot_target_xy(current_slot),
                next_obj_speed=next_speed,
                arrival_y_offset=env._throw_handoff_y_offset(next_speed),
            )
        if prediction is None:
            return None
        predicted_next_xyz = np.asarray(prediction[0], dtype=np.float64)
        # neg_wait_time at index 4 (see _throw_transition_supports_next note).
        neg_wait_time = prediction[4]
        if neg_wait_time is not None:
            return None
        if int(next_primitive) == env.ACTION_THROW and (
            float(predicted_next_xyz[1]) <= 0.08 or float(predicted_next_xyz[1]) >= 0.45
        ):
            return None
        if int(next_primitive) == env.ACTION_PUSH:
            trajectory_duration = float(prediction[2])
            wait_time_val = float(prediction[3]) if prediction[3] is not None else 0.0
            # `obj_y_at_push_time` = obj's y when the push action FIRES (just
            # after throw + wait). The push itself takes another `push_action_dt`
            # sec of sim before it contacts the obj. The executor rejects pushes
            # whose `planned_obj_xyz[1]` lands past -0.16. So the selector must
            # leave enough margin for the push's own trajectory duration.
            # Use a conservative ~1 sec push estimate (≈ bang-bang + line +
            # contact line + safety) which translates to belt_speed * 1.0 ≈
            # 0.4 m of additional obj motion before contact.
            push_action_dt_est = 1.0
            obj_y_at_push_time = (
                float(env._slot_grasp_throw_xyz(next_slot)[1])
                - next_speed * (trajectory_duration + wait_time_val)
            )
            obj_y_at_push_contact = obj_y_at_push_time - next_speed * push_action_dt_est
            if obj_y_at_push_contact <= -0.10:
                return None
            if float(predicted_next_xyz[1]) <= 0.05 or float(predicted_next_xyz[1]) >= 0.55:
                return None
        predicted_y = float(predicted_next_xyz[1])
        preferred_y = 0.23 if int(next_primitive) == env.ACTION_THROW else 0.18
        return float(-abs(predicted_y - preferred_y))

    def _throw_chain_score(self, current_slot: "ObjectSlot", next_slot: "ObjectSlot | None") -> float | None:
        env = self._env
        margin = self._throw_transition_margin(current_slot, next_slot, env.ACTION_THROW)
        if margin is None or next_slot is None:
            return None
        followup_margin = -np.inf
        for candidate_slot in env._ordered_active_slots():
            if candidate_slot is current_slot or candidate_slot is next_slot:
                continue
            candidate_margin = self._throw_transition_margin(next_slot, candidate_slot, env.ACTION_THROW)
            if candidate_margin is None:
                continue
            followup_margin = max(followup_margin, float(candidate_margin))
        if not np.isfinite(followup_margin):
            return None
        return float(margin + 2.0 + 0.25 * followup_margin)

    def _throw_candidate_score(
        self, current_slot: "ObjectSlot", candidate_slot: "ObjectSlot", next_primitive: int,
    ) -> float | None:
        env = self._env
        if int(next_primitive) == env.ACTION_THROW:
            score = self._throw_chain_score(current_slot, candidate_slot)
            if score is not None:
                return score
            # No visible 2-ahead followup: fall back to the immediate transition
            # margin so we still chain to a feasible next obj instead of dropping
            # to a route-only re-approach. Candidates that DO enable a followup
            # still outrank these (chain score adds a +2.0 bonus).
            return self._throw_transition_margin(current_slot, candidate_slot, env.ACTION_THROW)
        return self._throw_transition_margin(current_slot, candidate_slot, next_primitive)

    def _select_throw_handoff_candidate(
        self,
        current_slot: "ObjectSlot",
        ordered_slots: list["ObjectSlot"],
        desired_next_primitive: int,
    ) -> tuple[int | None, int]:
        env = self._env
        current_index = ordered_slots.index(current_slot) if current_slot in ordered_slots else -1
        # Restrict to action-addressable slots: the Gym action can only select
        # indices < max_detections, so the emitted chain target must be one too.
        addressable = min(len(ordered_slots), env.max_detections)
        candidate_indices = [
            idx for idx in range(addressable)
            if idx != current_index and not getattr(ordered_slots[idx], "is_obstacle", False)
        ]
        best_index: int | None = None
        best_primitive = desired_next_primitive
        best_score = -np.inf

        # Only consider the alternate primitive if pushing is actually enabled
        # AND the caller asked for a mixed mode (fallback gate). Otherwise the
        # throw planner can pick a "next=push" candidate that later fails with
        # `pushing_disabled`, and in pure throw/push mode the alternate would
        # break the requested mode.
        primitives_to_try: tuple[int, ...] = (int(desired_next_primitive),)
        if (
            env.pushing_enabled
            and env.pushing_trajectory_generator is not None
            and getattr(env, "_demo_allow_primitive_fallback", True)
        ):
            fallback = env.ACTION_THROW if int(desired_next_primitive) == env.ACTION_PUSH else env.ACTION_PUSH
            primitives_to_try = (int(desired_next_primitive), int(fallback))

        for primitive in primitives_to_try:
            for candidate_index in candidate_indices:
                candidate_slot = ordered_slots[candidate_index]
                score = self._throw_candidate_score(current_slot, candidate_slot, primitive)
                if score is not None and score > best_score:
                    best_index = candidate_index
                    best_primitive = primitive
                    best_score = score
            if best_index is not None:
                break

        return best_index, best_primitive

    # ─── push selection (mirror of throw) ────────────────────────────────

    def _current_slot_cone_plan(self, current_slot: "ObjectSlot"):
        """Cone push plan for `current_slot`, seeded from its cone hover so the
        feasibility/duration estimate matches the pre-positioned clean strike the
        executor will run (not a chase from the current EE). Returns the 4-tuple
        or None. Used by the push transition scorers for `last_trajectory`.

        Delegates to the env's `_slot_cone_push_plan` -- the SINGLE source of
        truth shared with screening (`_slot_throw_feasible` push-pending), so
        selection and the action mask use the byte-identical A-trajectory."""
        return self._env._slot_cone_push_plan(current_slot)

    def _push_transition_supports_next(
        self, current_slot: "ObjectSlot", next_slot: "ObjectSlot | None", next_primitive: int,
    ) -> bool:
        env = self._env
        if current_slot is None or not current_slot.active:
            return False
        if not env.pushing_enabled or env.pushing_trajectory_generator is None:
            return False
        if next_slot is None:
            return True
        plan = self._current_slot_cone_plan(current_slot)
        if plan is None:
            return False
        _, trajectory, _, _ = plan
        # See note in `_push_transition_margin`: at selection time the push of
        # `current_slot` hasn't run yet, so project obj B forward by the push
        # duration in addition to the after-traj duration so the y-window gate
        # matches what the demo selector will see at step N+1.
        after_plan = env._build_pushing_after_trajectory(
            trajectory,
            next_slot,
            current_slot=current_slot,
            next_primitive=next_primitive,
            extra_projection_seconds=float(trajectory.duration),
        )
        if after_plan is None:
            return False
        _, next_obj_xyz = after_plan
        if next_obj_xyz is None:
            return False
        if int(next_primitive) == env.ACTION_PUSH:
            return bool(not np.allclose(next_obj_xyz, env.action_single_object_standby_xyz, atol=1e-6))
        return True

    def _push_transition_margin(
        self, current_slot: "ObjectSlot", next_slot: "ObjectSlot | None", next_primitive: int,
    ) -> float | None:
        env = self._env
        if current_slot is None or not current_slot.active or next_slot is None:
            return None
        if not env.pushing_enabled or env.pushing_trajectory_generator is None:
            return None
        plan = self._current_slot_cone_plan(current_slot)
        if plan is None:
            return None
        _, trajectory, _, _ = plan
        # Selection-time scoring: the predicted push of `current_slot` will run
        # before the after-traj, so obj B's projected y at the moment of the
        # next push must account for trajectory.duration of belt drift, not
        # just the after-traj duration. Without this, the gate accepts chains
        # that B will have drifted out of by step N+1 (see plan).
        after_plan = env._build_pushing_after_trajectory(
            trajectory,
            next_slot,
            current_slot=current_slot,
            next_primitive=next_primitive,
            extra_projection_seconds=float(trajectory.duration),
        )
        if after_plan is None:
            return None
        _, next_obj_xyz = after_plan
        if next_obj_xyz is None:
            return None
        # For next=PUSH a standby fallback (unreachable next pre-push pose)
        # disqualifies the candidate — it would commit to a chain that breaks.
        if int(next_primitive) == env.ACTION_PUSH:
            if np.allclose(next_obj_xyz, env.action_single_object_standby_xyz, atol=1e-6):
                return None
        # Same y-window gates as throw: keep the next obj inside a graspable
        # / pushable window so the chained step doesn't immediately fail.
        predicted_y = float(next_obj_xyz[1])
        if int(next_primitive) == env.ACTION_THROW and (predicted_y <= 0.08 or predicted_y >= 0.45):
            return None
        if int(next_primitive) == env.ACTION_PUSH and (predicted_y <= 0.05 or predicted_y >= 0.55):
            return None
        # Selection-vs-execution alignment: also run the SAME feasibility
        # predicate that step N+1's chained-action gate uses, but with the
        # *projected* obj position. This catches candidates whose projected
        # position passes the loose y-window above but fails the stricter
        # _build_predicted_pushing_plan / planned_obj.y > -0.16 check.
        # Without this, selection accepts chains that step N+1 will reject,
        # producing the ~91% chain phenomenon rate.
        if int(next_primitive) == env.ACTION_PUSH:
            if not self._demo_slot_supports_primitive(
                next_slot, env.ACTION_PUSH, obj_xyz_override=next_obj_xyz,
            ):
                return None
        preferred_y = 0.23 if int(next_primitive) == env.ACTION_THROW else 0.18
        return float(-abs(predicted_y - preferred_y))

    def _push_chain_score(self, current_slot: "ObjectSlot", next_slot: "ObjectSlot | None") -> float | None:
        env = self._env
        margin = self._push_transition_margin(current_slot, next_slot, env.ACTION_PUSH)
        if margin is None or next_slot is None:
            return None
        followup_margin = -np.inf
        for candidate_slot in env._ordered_active_slots():
            if candidate_slot is current_slot or candidate_slot is next_slot:
                continue
            candidate_margin = self._push_transition_margin(next_slot, candidate_slot, env.ACTION_PUSH)
            if candidate_margin is None:
                continue
            followup_margin = max(followup_margin, float(candidate_margin))
        if not np.isfinite(followup_margin):
            return None
        return float(margin + 2.0 + 0.25 * followup_margin)

    def _push_candidate_score(
        self, current_slot: "ObjectSlot", candidate_slot: "ObjectSlot", next_primitive: int,
    ) -> float | None:
        env = self._env
        if int(next_primitive) == env.ACTION_PUSH:
            score = self._push_chain_score(current_slot, candidate_slot)
            if score is not None:
                return score
            # No visible 2-ahead followup: fall back to the immediate transition
            # margin so we still chain instead of dropping to a route-only step.
            return self._push_transition_margin(current_slot, candidate_slot, env.ACTION_PUSH)
        return self._push_transition_margin(current_slot, candidate_slot, next_primitive)

    def _select_push_handoff_candidate(
        self,
        current_slot: "ObjectSlot",
        ordered_slots: list["ObjectSlot"],
        desired_next_primitive: int,
    ) -> tuple[int | None, int]:
        env = self._env
        current_index = ordered_slots.index(current_slot) if current_slot in ordered_slots else -1
        # Restrict to action-addressable slots: the Gym action can only select
        # indices < max_detections, so the emitted chain target must be one too.
        addressable = min(len(ordered_slots), env.max_detections)
        candidate_indices = [
            idx for idx in range(addressable)
            if idx != current_index and not getattr(ordered_slots[idx], "is_obstacle", False)
        ]
        best_index: int | None = None
        best_primitive = desired_next_primitive
        best_score = -np.inf

        primitives_to_try: tuple[int, ...] = (int(desired_next_primitive),)
        if (
            env.pushing_enabled
            and env.pushing_trajectory_generator is not None
            and getattr(env, "_demo_allow_primitive_fallback", True)
        ):
            fallback = env.ACTION_THROW if int(desired_next_primitive) == env.ACTION_PUSH else env.ACTION_PUSH
            primitives_to_try = (int(desired_next_primitive), int(fallback))

        for primitive in primitives_to_try:
            for candidate_index in candidate_indices:
                candidate_slot = ordered_slots[candidate_index]
                score = self._push_candidate_score(current_slot, candidate_slot, primitive)
                if score is not None and score > best_score:
                    best_index = candidate_index
                    best_primitive = primitive
                    best_score = score
            if best_index is not None:
                break

        return best_index, best_primitive

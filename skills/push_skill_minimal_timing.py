"""Push-only timing experiment with no empirical delay constants.

Enable with ``GP8_PUSH_TIMING_MODE=minimal``.  The normal :class:`PushSkill`
remains the default, and throw timing is not changed.

This experiment deliberately separates positioning from the timed stroke:

1. Build the stroke and chain.
2. Move to the exact first stroke waypoint and verify actual joint arrival.
3. Re-estimate the object's ETA after positioning.
4. Dispatch exactly ``t_contact`` before predicted object arrival.

Only kinematic/planned durations remain in the timing equations.  There is no
build, dispatch, tail, settle, residual, fixed-delay, or ``OPT_TIME_TO_REAL``
correction.  The settle check itself remains as a state-based safety gate; its
timeout is not added to either timing equation.
"""

from __future__ import annotations

import numpy as np

from gp8_control.skills.push_skill import (
    PUSH_RETREAT_DISTANCE,
    PushSkill,
)


class MinimalTimingPushSkill(PushSkill):
    """Pre-positioned push using only motion-derived timing terms."""

    name = "push"

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._intercept_budgets_by_track: dict[int, dict] = {}
        self._executing_intercept_budget: dict = {}

    def preposition_enabled(self) -> bool:
        """Always isolate the stroke from bin-dependent approach motion."""
        return True

    def dynamic_arrival_lead(self, contact_offset: float) -> float:
        """Nominal interface lead: physical run-up time only."""
        return self._stroke_time_to(contact_offset)

    def t_to_contact(self, move_time: float) -> float:
        """Legacy scalar fallback when candidate geometry is unavailable.

        The live app uses :meth:`intercept_time_budget`, which replaces this
        generic grasp-pose proxy with the candidate's real push-start pose.
        This fallback is retained for out-of-tree callers still using the old
        scalar callback.
        """
        t_position = max(0.0, float(move_time))
        t_runup = self._stroke_time_to(PUSH_RETREAT_DISTANCE)
        return t_position + t_runup

    def intercept_time_budget(
        self,
        target,
        current_joint: np.ndarray,
        T_aim: np.ndarray,
        T_grasp: np.ndarray,
        aim_joint: np.ndarray,
        grasp_joint: np.ndarray,
        move_time: float,
    ) -> float:
        """Candidate-specific ``current→q_park→contact`` horizon.

        ``move_time`` is intentionally ignored: it ends at the suction grasp
        pose and does not know the bin-dependent push heading.  For every Y in
        the intercept fixed-point loop this method instead:

        * derives the fixed-bin push direction;
        * solves the actual backswing/stroke-start joint ``q_park``;
        * estimates the same direct/via positioning route execute() will use;
        * adds run-up time for the actual retreat after the min-X clamp.

        All terms come from geometry and the motion profiles.  No empirical
        timing constant or multiplier is added.
        """
        del move_time
        budget_parts = self._candidate_qpark_timing(
            target,
            current_joint,
            T_aim,
            T_grasp,
            aim_joint,
            grasp_joint,
        )
        if budget_parts is None:
            return float("inf")
        self._last_intercept_budget = budget_parts
        track_id = getattr(target, "track_id", None)
        if track_id is not None:
            self._intercept_budgets_by_track[int(track_id)] = budget_parts.copy()
        return float(budget_parts["position_s"] + budget_parts["runup_s"])

    def execute(self, request):
        """Capture this target's converged placement budget before chain planning."""
        track_id = getattr(request.target, "track_id", None)
        self._executing_intercept_budget = self._intercept_budgets_by_track.get(
            int(track_id) if track_id is not None else -1,
            {},
        ).copy()
        return super().execute(request)

    def fire_lead(self, t_contact: float) -> float:
        """Use the exact built stroke timestamp as the complete fire lead."""
        return max(0.0, float(t_contact))

    def log_fire_timing(
        self, lead: float, t_contact: float, approach_text: str
    ) -> None:
        self.ctx.log.info(
            f"[push-minimal] FIRE lead {lead * 1000:.0f}ms = exact built "
            f"contact timestamp only ({approach_text}); corrections 0ms"
        )

    def log_pipeline_timing(
        self,
        elapsed: float,
        *,
        preposition: bool,
        positioning_planned: float | None,
    ) -> None:
        planned = float(positioning_planned or 0.0)
        measured_extra = elapsed - planned
        predicted_position = self._executing_intercept_budget.get("position_s")
        comparison = ""
        if predicted_position is not None:
            delta = planned - float(predicted_position)
            comparison = (
                f"; intercept position prediction "
                f"{float(predicted_position) * 1000:.0f}ms, "
                f"gated-plan delta {delta * 1000:+.0f}ms"
            )
        self.ctx.log.info(
            f"[push-minimal] exec→wait measured {elapsed * 1000:.0f}ms = "
            f"positioning plan {planned * 1000:.0f}ms + unmodelled/measured "
            f"{measured_extra * 1000:.0f}ms{comparison}; no value is added "
            f"to fire timing"
        )

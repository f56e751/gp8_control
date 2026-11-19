"""Belt-tracking descend pick — the GRASP half of the pick-and-throw cycle.

Split out of ``throw_skill.py`` so the grasping planner is one module: how the
cup meets a moving object (``PickWaitMode``), when the wait must end so the
follow lands ON arrival (``arrival_lead``), and the cartesian segment that
follows the object downstream at belt speed while descending onto it
(``_build_track_descend``). ``ThrowSkill`` mixes this in; any other skill that
wants the same pick can too.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from gp8_control.grasping.context import SkillContext


class PickWaitMode(Enum):
    """How the arm meets the object at the ambush intercept.

    Extension point: map object classes to a wait mode in ``PICK_WAIT_MODE``
    so e.g. fragile classes can track-and-descend while flat ones park at
    grasp height.
    """

    WAIT_AT_GRASP = "wait_at_grasp"   # cup parked at grasp height; suction on arrival
    TRACK_DESCEND = "track_descend"   # park high, then follow the object at belt
                                      # speed while descending onto it


# Per-class wait mode (class_name -> PickWaitMode). Classes not listed use
# DEFAULT_PICK_WAIT_MODE.
PICK_WAIT_MODE: dict[str, PickWaitMode] = {}
DEFAULT_PICK_WAIT_MODE = PickWaitMode.TRACK_DESCEND

# Belt-tracking descend shaping (see ThrowSkill._build_track_descend). Both are
# small compared to the descend itself and are NOT operator flags — the three
# tunables the operator passes per run are cfg.TRACK_Z_START/END/SPEED.
# Ramp-up to belt speed [s]. The arm is at REST at the hover when the object
# arrives, so it cannot start at belt speed; it accelerates over this window.
# ``arrival_lead`` starts the segment TRACK_ACCEL_T/2 early, which makes the
# cup's along-belt position match the object's EXACTLY from t = TRACK_ACCEL_T
# onward (the ramp's half-window lag is pre-paid) — so the descend lands on a
# co-moving object with zero relative velocity.
TRACK_ACCEL_T: float = 0.10
# Ramp-down to rest [s] after Z reaches TRACK_Z_END. The NN throw arc starts
# from rest (new_trajectory's dq(0)=0 boundary condition), so the tracking
# segment must stop before it. The object is already sealed to the cup by then,
# so the belt just slips underneath — the lag here is harmless.
TRACK_DECEL_T: float = 0.10


class TrackDescendPickMixin:
    """Pick-side methods shared with :class:`~gp8_control.grasping.base.ManipulationSkill`
    subclasses. Expects ``self.ctx`` (a :class:`SkillContext`) and cooperates with
    ``ManipulationSkill.arrival_lead`` via ``super()`` — list this mixin FIRST in the
    class bases: ``class ThrowSkill(TrackDescendPickMixin, ManipulationSkill)``.
    """

    ctx: "SkillContext"

    def arrival_lead(self) -> float:
        """End the wait half a ramp-up window early in TRACK_DESCEND mode.

        The tracking segment starts from rest and reaches belt speed only after
        ``TRACK_ACCEL_T``; over that ramp the cup covers ``v*TRACK_ACCEL_T/2``
        while the object covers ``v*TRACK_ACCEL_T``. Dispatching the segment
        ``TRACK_ACCEL_T/2`` before predicted arrival pre-pays exactly that
        difference, so cup and object are aligned from the end of the ramp on
        (and stay aligned through the descend — verified: relative offset 0 mm
        from t=TRACK_ACCEL_T until Z lands).

        This is keyed off the CONFIG only, not the per-object wait mode, because
        the lead is consumed by ``position_and_prime`` before the mode's segment
        is known to be buildable. A parked (WAIT_AT_GRASP / fallback) pick
        therefore ends its wait 50 ms early — well inside the SUCTION_LEAD cap
        that actually gates the parked prime, so it changes nothing there.
        """
        lead = super().arrival_lead()
        if self._track_z_params() is not None:
            lead += 0.5 * TRACK_ACCEL_T
            # Empirical timing knob: start the follow+descend TRACK_LEAD_T earlier
            # to cancel a fixed downstream landing offset (the object leading the
            # cup at touchdown). See Config.TRACK_LEAD_T.
            lead += float(self.ctx.cfg.TRACK_LEAD_T)
        return lead

    # ------------------------------------------------------------------
    # Belt-tracking descend (TRACK_DESCEND wait mode)
    # ------------------------------------------------------------------
    def _track_z_params(self) -> "Optional[tuple[float, float, float]]":
        """Resolved ``(z_start, z_end, z_speed)`` for the tracking descend, or
        ``None`` when it is disabled / degenerate (caller falls back to the
        parked WAIT_AT_GRASP pick).

        NaN start/end mean "derive from GRASP_Z" (see ``Config.TRACK_Z_START``):
        start = GRASP_Z + TRACK_Z_HOVER, end = GRASP_Z. Disabled when the speed
        is non-positive or the two heights don't leave a downward travel.
        """
        cfg = self.ctx.cfg
        v_desc = float(cfg.TRACK_Z_SPEED)
        if not np.isfinite(v_desc) or v_desc <= 0.0:
            return None
        z_start = float(cfg.TRACK_Z_START)
        if not np.isfinite(z_start):
            z_start = float(cfg.GRASP_Z) + float(cfg.TRACK_Z_HOVER)
        z_end = float(cfg.TRACK_Z_END)
        if not np.isfinite(z_end):
            z_end = float(cfg.GRASP_Z)
        if z_start - z_end <= 1e-4:
            return None
        return z_start, z_end, v_desc

    def _tcp_z_joint(self, T_grasp: np.ndarray, z_abs: float, seed_joint: np.ndarray):
        """IK for the grasp TCP's XY/orientation held at absolute height ``z_abs``,
        seeded with ``seed_joint`` so the solution stays on the same wrist branch.
        ``None`` on failure (caller falls back)."""
        T_h = np.asarray(T_grasp, dtype=float).copy()
        T_h[2, 3] = float(z_abs)
        q = self.ctx.robot.inverse_kinematics(
            T_h, q_init=np.asarray(seed_joint, dtype=float)[:6]
        )
        return None if q is None else np.asarray(q, dtype=float)

    def _build_track_descend(
        self,
        T_grasp: np.ndarray,
        wait_joint: np.ndarray,
        v_belt: float,
        z_start: float,
        z_end: float,
        v_desc: float,
    ):
        """Cartesian segment that FOLLOWS the object downstream while descending.

        Starting at the intercept (``T_grasp`` XY, height ``z_start``) the TCP
        moves along -Y — the belt direction — with the profile
        ``ramp up over TRACK_ACCEL_T -> cruise at v_belt -> ramp down over
        TRACK_DECEL_T``, while Z falls linearly from ``z_start`` to ``z_end`` at
        ``v_desc``. Cruise means ZERO relative velocity to the object, so the cup
        settles onto an object that is stationary in its frame instead of sliding
        underneath it. Both ends are at rest: the arm is parked when the segment
        is dispatched, and the NN throw arc that follows needs a rest start.

        Each waypoint's IK is seeded with the previous solution to hold one wrist
        branch (same contract as the push stroke). If IK fails part-way — the
        follow ran out of reach downstream — the segment is TRUNCATED there and
        the throw simply starts from the shorter follow.

        Returns ``(traj (6,n), vel (6,n), ts (n,))`` or ``None`` when the belt is
        stopped or too few waypoints solved.
        """
        ctx = self.ctx
        if v_belt < 1e-3:
            return None                        # belt stopped -> nothing to track
        dt = 1.0 / ctx.cfg.TRAJ_HZ
        t_desc = (z_start - z_end) / v_desc
        t_acc = TRACK_ACCEL_T
        t_dec = TRACK_DECEL_T
        t_cruise_end = max(t_desc, t_acc)      # hold belt speed until Z has landed
        t_total = t_cruise_end + t_dec
        n_steps = max(2, int(round(t_total / dt)))

        def _along(t: float) -> float:
            """Downstream distance travelled at segment time ``t``."""
            if t <= t_acc:
                return v_belt * t * t / (2.0 * t_acc)
            s = v_belt * (t_acc * 0.5 + (min(t, t_cruise_end) - t_acc))
            if t > t_cruise_end:
                tau = min(t - t_cruise_end, t_dec)
                s += v_belt * (tau - 0.5 * tau * tau / t_dec)
            return s

        waypoints = []
        q_seed = np.asarray(wait_joint, dtype=float)
        for i in range(n_steps + 1):
            t = min(i * dt, t_total)
            T_wp = np.asarray(T_grasp, dtype=float).copy()
            T_wp[1, 3] -= _along(t)            # the belt advances toward -Y
            T_wp[2, 3] = z_start - v_desc * min(t, t_desc)
            ik = ctx.robot.inverse_kinematics(T_wp, q_init=q_seed)
            if ik is None:
                ctx.log.warn(
                    f"track-descend IK failed @ {i}/{n_steps} "
                    f"(y-{_along(t) * 1000:.0f}mm, z={T_wp[2, 3]:.3f}) — "
                    f"truncating the follow to {len(waypoints)} waypoints"
                )
                break
            q_seed = np.asarray(ik, dtype=float)
            waypoints.append(q_seed)

        if len(waypoints) < 3:
            ctx.log.warn("track-descend segment unbuildable — parking at the grasp")
            return None

        n = len(waypoints)
        traj = np.column_stack(waypoints)                  # (6, n)
        ts = np.linspace(0.0, dt * (n - 1), n)
        vel = np.zeros_like(traj)
        vel[:, 1:-1] = (traj[:, 2:] - traj[:, :-2]) / (2.0 * dt)
        # Both ends at rest (parked start, rest start for the throw arc). On a
        # truncated follow the tail velocity is forced to 0 as well — the arm is
        # at its reach limit there and must stop regardless.
        vel[:, 0] = 0.0
        vel[:, -1] = 0.0

        # Joint-velocity check. Unlike the throw arc we must NOT stretch time to
        # fix an overrun — stretching breaks the belt sync that is the whole
        # point — so an over-limit follow is abandoned for the parked pick.
        seg = np.abs(np.diff(traj, axis=1)) / dt
        ratio = float(np.max(seg / np.asarray(ctx.M1[:6], dtype=float)[:, None]))
        if ratio > 1.0:
            ctx.log.warn(
                f"track-descend exceeds joint velocity limits (max {ratio:.2f}x) — "
                f"parking at the grasp instead"
            )
            return None
        return traj, vel, ts

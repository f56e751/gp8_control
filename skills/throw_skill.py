"""Throw skill: grasp an object at the intercept, then fling it (NN throw)."""

from __future__ import annotations

import datetime
import json
import time
from enum import Enum
from typing import TYPE_CHECKING, Optional

import numpy as np

from gp8_control.skills.base import ManipulationSkill, SkillResult
from gp8_control.trajectory.trajectory_primitive import (
    new_trajectory,
    trajectory,
    pad,
)
from gp8_control.skills.throw_visualizer import ThrowVisualizer

if TYPE_CHECKING:
    from gp8_control.skills.context import PickRequest
    from gp8_control.tracking import TrackedObject


# =========================================================================
# Throw policy (was app-level policy in app.py)
# =========================================================================

# Per-class throw bin TARGET (absolute base-frame XYZ, m). When a target's
# class is in this map, T_aim2 is OVERRIDDEN with these coordinates so the
# NN throw aims at a fixed bin location instead of secondary/T_aim hover.
# Empty default — fill in with measured bin coords (e.g., from terminal_debug).
THROW_BIN_TARGET_MAP: dict[str, tuple] = {}


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


class ThrowSkill(ManipulationSkill):
    """Pick (suction) at the intercept and throw the object via the NN trajectory.

    ``execute`` runs the full ambush cycle (position → wait → suction → throw →
    chain to next intercept). The individual stages —
    :meth:`plan_throw_landing`, :meth:`build_throw_trajectory` — are public so
    external code can reuse the throw planning.
    """

    name = "throw"

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._last_throw_meta: dict = {}
        self._throw_bins = self._parse_throw_bins(ctx.cfg.THROW_BINS)
        self._visualizer = ThrowVisualizer(
            ctx.node,
            ctx.robot,
            impact_z=ctx.cfg.THROW_VIZ_IMPACT_Z,
            goal_xy=(ctx.cfg.THROW_GOAL_X, ctx.cfg.THROW_GOAL_Y),
            goal_radius=ctx.cfg.THROW_GOAL_RADIUS,
        )

    @staticmethod
    def _parse_throw_bins(raw: str) -> list[tuple[str, np.ndarray, float]]:
        if not raw.strip():
            return []
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("throw_bins must be a JSON list")
        bins = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"throw_bins[{i}] must be an object")
            name = str(item.get("name", f"bin{i + 1}"))
            xyz = np.asarray([item["x"], item["y"], item["z"]], dtype=float)
            radius = float(item.get("radius", 0.10))
            if not np.all(np.isfinite(xyz)) or not np.isfinite(radius) or radius <= 0.0:
                raise ValueError(f"throw_bins[{i}] has invalid coordinates/radius")
            bins.append((name, xyz, radius))
        return bins

    def _select_throw_bin(self, T_grasp: np.ndarray):
        """Choose once from the planned final grasp; the throw flow stays unchanged."""
        if not self._throw_bins:
            return None
        grasp_xy = np.asarray(T_grasp[:2, 3], dtype=float)
        distances = [float(np.linalg.norm(xyz[:2] - grasp_xy))
                     for _, xyz, _ in self._throw_bins]
        index = int(np.argmin(distances))
        selected = self._throw_bins[index]
        self.ctx.log.info(
            f"[bin-select] grasp=({grasp_xy[0]:+.3f},{grasp_xy[1]:+.3f}) "
            f"candidates={[f'{b[0]}:{d:.3f}m' for b, d in zip(self._throw_bins, distances)]} "
            f"selected={selected[0]}"
        )
        return selected

    def t_to_contact(self, move_time: float) -> float:
        """Throw pick budget = positioning estimate + a GUARANTEED parked
        vacuum-forming hold (``cfg.MIN_SUCTION_HOLD``).

        Adding the hold to the base ``move_time * PICK_FEASIBILITY_FACTOR`` makes
        ``earliest_reachable_intercept`` place the grasp far enough DOWNSTREAM that
        the arm reaches it ~``MIN_SUCTION_HOLD`` before the object arrives, so the
        vacuum seals during the parked wait instead of firing with ~0 lead (the
        backed-up 2nd+ object miss). An object that can't be caught that far
        downstream is dropped by the same solver rather than grabbed with no hold.
        """
        base = move_time * self.ctx.cfg.PICK_FEASIBILITY_FACTOR
        return base + self.ctx.cfg.MIN_SUCTION_HOLD

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

    # ------------------------------------------------------------------
    # Skill entry point (ambush strategy)
    # ------------------------------------------------------------------
    def execute(self, request: "PickRequest") -> SkillResult:
        """Pre-position (mode-dependent), wait for arrival + suction, then throw."""
        ctx = self.ctx
        target = request.target
        current_joint = request.current_joint
        aim_joint = request.aim_joint
        grasp_joint = request.grasp_joint
        T_aim = request.T_aim
        T_grasp = request.T_grasp
        secondary = request.secondary

        mode = PICK_WAIT_MODE.get(target.class_name, DEFAULT_PICK_WAIT_MODE)

        # Start clean (uniform per-object flow: no cross-cycle suction hand-off).
        ctx.traj_ctrl.suction_off()

        # TRACK_DESCEND (default): park at TRACK_Z_START above the intercept, and on
        # the object's arrival run one belt-tracking segment that follows it at belt
        # speed while the cup descends to TRACK_Z_END. The throw then starts from the
        # END of that follow — further downstream and lower than the nominal grasp —
        # so T_grasp/grasp_joint are re-bound below.
        # WAIT_AT_GRASP (per-class, or the fallback when tracking is disabled/
        # unbuildable): park AT the grasp height and take the object passively.
        wait_joint = np.asarray(grasp_joint, dtype=float)
        track = None                              # (traj, vel, ts) | None
        z_params = self._track_z_params() if mode == PickWaitMode.TRACK_DESCEND else None
        if z_params is not None:
            z_start, z_end, v_desc = z_params
            q_wait = self._tcp_z_joint(T_grasp, z_start, grasp_joint)
            if q_wait is None:
                ctx.log.warn(
                    f"track-descend hover IK failed at z={z_start:.3f} — "
                    f"falling back to the parked grasp wait"
                )
            else:
                track = self._build_track_descend(
                    T_grasp, q_wait, float(ctx.conveyor.current),
                    z_start, z_end, v_desc,
                )
                if track is not None:
                    wait_joint = q_wait
                    ctx.log.info(
                        f"track-descend: hover z={z_start:.3f} -> z={z_end:.3f} "
                        f"@ {v_desc:.3f} m/s ({(z_start - z_end) / v_desc:.2f}s), "
                        f"following belt {float(ctx.conveyor.current):.3f} m/s for "
                        f"{float(track[2][-1]):.2f}s"
                    )

        # The tracking trajectory's final joint is already known before pickup,
        # so choose the nearest bin once from its predicted final grasp pose.
        bin_grasp = T_grasp
        if track is not None:
            bin_grasp = ctx.robot.forward_kinematics(track[0][:, -1][:6])
        selected_bin = self._select_throw_bin(bin_grasp)

        # Drive to the wait pose and (for the parked pick) prime suction SUCTION_LEAD
        # before the object's arrival. Returns arrival_lead() before the object
        # reaches the intercept.
        #
        # TRACK_DESCEND (track is not None): do NOT prime at the hover. The wait pose
        # is TRACK_Z_START above the object, so an early prime there runs the vacuum
        # in air for SUCTION_LEAD-arrival_lead (~0.43 s) while the arm sits STILL,
        # before the descend even starts — the stationary suction-on gap. Instead
        # park silently and fire suction AS the descend begins (below), so the vacuum
        # forms DURING the descent motion and there is no still period. The descend's
        # Z-landing takes (z_start-z_end)/v_desc (~0.5 s at defaults), which is the
        # vacuum-formation window SUCTION_LEAD used to buy while parked. WAIT_AT_GRASP
        # / fallback (track is None) keeps the original parked prime.
        ctx.set_status("POSITIONING", target.class_name)
        ctx.position_and_prime(
            current_joint, aim_joint, wait_joint, target, T_grasp[1, 3],
            start_lead=self.arrival_lead(),
            prime_suction=(track is None),
        )

        # DIAGNOSTIC: object vs intercept at the instant the lift/throw fires.
        # delta < 0 -> the object already passed the grasp point and we suction
        # empty belt / lift behind it (the multi-object symptom). See
        # SkillContext.log_action_timing.
        ctx.log_action_timing(target, T_grasp[1, 3], "throw-lift")

        if track is not None:
            # The object is arriving now: run the follow+descend. send_trajectory_queue
            # blocks for the whole segment (the 250 Hz stream paces it in real time),
            # so the cup is down on the object, at rest, when it returns.
            t_traj, t_vel, t_ts = track
            q_end = t_traj[:, -1]
            # Fire suction AS the descend starts — NOT parked high above it. suction_on()
            # only enqueues on the IO worker and returns immediately, so the descend
            # dispatch follows with no stationary gap: the vacuum forms while the cup
            # is already moving down onto the object and is fully pulled by the time it
            # settles at z_end (see the prime_suction=False rationale above).
            ctx.traj_ctrl.suction_on()
            ctx.log_suction_on(target)
            ctx.traj_ctrl.send_trajectory_queue(
                t_traj, t_vel, t_ts, final_joint=q_end,
            )
            # Re-bind the throw's start pose to where the follow actually ended.
            # compute_throw_params and the NN arc both key off these, so leaving
            # them at the nominal intercept would plan a throw from a pose the arm
            # is no longer in (a jump at the swing start, and a mis-aimed arc).
            grasp_joint = np.asarray(q_end, dtype=float)
            T_grasp = ctx.robot.forward_kinematics(grasp_joint[:6])
            ctx.log.info(
                f"track-descend done: throw starts at "
                f"({T_grasp[0, 3]:+.3f}, {T_grasp[1, 3]:+.3f}, {T_grasp[2, 3]:+.3f}) m"
            )

        # Lift + throw.
        ctx.set_status("THROWING", target.class_name)

        if selected_bin is None:
            goal_x = float(ctx.cfg.THROW_GOAL_X)
            goal_y = float(ctx.cfg.THROW_GOAL_Y)
            goal_radius = float(ctx.cfg.THROW_GOAL_RADIUS)
        else:
            _, selected_xyz, goal_radius = selected_bin
            goal_x, goal_y = map(float, selected_xyz[:2])
        model_distance = float(np.hypot(goal_x, goal_y))
        self._visualizer.set_goal((goal_x, goal_y), goal_radius)

        # Match the coordinate convention used to train the FCN: theta is the
        # base-origin azimuth of the throw target, and the grasp/aim XY inputs
        # are rotated by -theta inside PickThrowPlanner.  Do NOT use the
        # point-to-point bearing (goal - grasp) here; that rotates coordinates
        # about the wrong reference and produced a ~44 deg release-direction
        # error on hardware. The NN distance is the selected bin's base-frame
        # XY radius, matching the existing single-goal convention.
        theta = float(np.arctan2(
            goal_y,
            goal_x,
        ))
        ctx.log.info(
            f"Throw frame: theta={np.degrees(theta):+.2f}deg (goal azimuth), "
            f"grasp=({T_grasp[0, 3]:+.3f},{T_grasp[1, 3]:+.3f}), "
            f"goal=({goal_x:+.3f},{goal_y:+.3f}), "
            f"model_distance={model_distance:.3f}m"
        )

        # The first object is commonly locked while the queue still contains
        # only that object, so request.secondary is None. More objects can be
        # detected during the long ambush wait; bind the first one here, at
        # actual throw time, just as next_chain_target() re-polls the live
        # queue below. Without this late bind the NN arc falls back over the
        # current grasp while the appended chain heads to the newly arrived
        # next object, producing the observed first-cycle dogleg.
        if secondary is None and ctx.queue:
            secondary = next(iter(ctx.queue._objects), None)
            if secondary is not None:
                ctx.log.info(
                    "Late-bound throw secondary from live queue: "
                    f"id={secondary.track_id} {secondary.class_name} "
                    f"x={secondary.T_aim_base[0, 3]:+.3f}"
                )

        # A selected runtime bin behaves exactly like the existing fixed bin;
        # without throw_bins, preserve the legacy per-class/fallback flow.
        if selected_bin is not None:
            bin_name, bin_xyz, _ = selected_bin
            T_aim2 = np.eye(4)
            T_aim2[:3, :3] = T_aim[:3, :3]
            T_aim2[:3, 3] = bin_xyz
            ctx.log.info(
                f"Throw target: selected bin {bin_name} "
                f"({bin_xyz[0]:+.3f}, {bin_xyz[1]:+.3f}, {bin_xyz[2]:+.3f}) m"
            )
        else:
            bin_xyz = THROW_BIN_TARGET_MAP.get(target.class_name)
            if bin_xyz is not None:
                T_aim2 = np.eye(4)
                T_aim2[:3, :3] = T_aim[:3, :3]
                T_aim2[:3, 3] = np.asarray(bin_xyz, dtype=float)
                ctx.log.info(
                    f"Throw target for {target.class_name}: fixed bin "
                    f"({bin_xyz[0]:+.3f}, {bin_xyz[1]:+.3f}, {bin_xyz[2]:+.3f}) m"
                )
            else:
                T_aim2 = self.plan_throw_landing(
                    T_grasp, theta, T_aim, time.time(), secondary,
                )

        aim_joint2 = ctx.robot.inverse_kinematics(T_aim2)
        if aim_joint2 is None:
            ctx.log.warn("Throw IK failed after grab; lifting in place")
            aim_joint2, T_aim2 = aim_joint, T_aim
        # Hold the pick wrist through the throw — the NN drives joints 1-5
        # only, so J6 just parks wherever the grasp left it (PICK_WRIST_J6
        # baseline) instead of snapping to 0 and back around every throw.
        aim_joint2 = np.asarray(aim_joint2, dtype=float)
        aim_joint2[-1] = float(grasp_joint[-1])

        params = ctx.planner.compute_throw_params(
            T_grasp, T_aim2, theta, target_distance=model_distance,
        )
        # Chain the follow-through toward the NEXT object's grasp (best-effort) so the arm
        # OVERLAPS the next approach with this throw instead of parking far and re-driving
        # serially. Symmetric + stateless: the next epoch still SELECTS + DRIVES fresh from
        # this closer pose (no commit/preposition). None -> lifted-standby park.
        nxt = ctx.next_chain_target(grasp_joint, float(params.T))
        next_grasp, next_cand = nxt if nxt is not None else (None, None)
        # Chain PARK: ask the NEXT object's skill where to park (push → its
        # backswing pose; default None → lifted standby over the grasp).
        chain_park = None
        if next_cand is not None:
            chain_park = ctx.skill_obj_for(next_cand).chain_park_joint(
                next_grasp, next_cand
            )
        self.build_throw_trajectory(
            grasp_joint, aim_joint2, params,
            next_grasp=next_grasp, chain_park=chain_park,
        )
        ctx.traj_ctrl.suction_off()   # release the object after the throw
        self._log_throw_cycle(target)
        ctx.set_active_target(None)
        ctx.set_status("IDLE", "")
        return SkillResult(True, "throw complete")

    # ------------------------------------------------------------------
    # Throw planning
    # ------------------------------------------------------------------
    def plan_throw_landing(
        self,
        T_grasp1: np.ndarray,
        theta: float,
        T_aim1_fallback: np.ndarray,
        now: float,
        secondary: "Optional[TrackedObject]",
    ) -> np.ndarray:
        """Aim throw at ``secondary`` if feasible; else drop in place.

        ``secondary`` is captured by the caller *before* the lock step so
        this method does not depend on the queue's mutation order.
        """
        ctx = self.ctx
        if secondary is None:
            return T_aim1_fallback.copy()

        # ``secondary.T_aim_base.z`` is camera_debug's display reference
        # (REFERENCE_Z_BASE + DETECTION_OFFSET_AIM = 0.700 m), not an
        # operational robot hover. The post-throw chain parks at INITIAL_T.z;
        # use that same safe, reachable height for the parametric arc endpoint
        # so the arc and chain share one Z convention.
        T_aim2_seed = secondary.T_aim_base.copy()
        # Convert the frozen detection anchor to NOW using encoder distance.
        # The pure planner can then project only the future throw lead from a
        # zero-age pose instead of re-integrating current_speed over track age.
        T_aim2_seed[1, 3] = ctx.object_y_now(
            secondary, now, ctx.conveyor.current
        )
        T_aim2_seed[2, 3] = float(ctx.cfg.INITIAL_T[2, 0])
        T_aim2, _, _, neg_wait2 = ctx.planner.plan_throw_landing(
            T_grasp1,
            T_aim2_seed,
            theta,
            now,
            ctx.conveyor.current,
            now,
            fixed_delay=ctx.cfg.FIXED_DELAY_THROW,
        )
        infeasible = (
            neg_wait2 is not None
            or T_aim2[0, 3] < 0.1
            or T_aim2[2, 3] < 0.0
        )
        if infeasible:
            ctx.log.warn(
                "Next-object throw aim infeasible; using current-object hover: "
                f"secondary aim=({T_aim2[0, 3]:+.3f}, "
                f"{T_aim2[1, 3]:+.3f}, {T_aim2[2, 3]:+.3f})m, "
                f"past_reach={neg_wait2 is not None}"
            )
            return T_aim1_fallback.copy()
        ctx.log.info(
            "Next-object throw arc endpoint: "
            f"({T_aim2[0, 3]:+.3f}, {T_aim2[1, 3]:+.3f}, "
            f"{T_aim2[2, 3]:+.3f})m"
        )
        return T_aim2

    # ------------------------------------------------------------------
    # Throw trajectory build + dispatch
    # ------------------------------------------------------------------
    def build_throw_trajectory(
        self,
        grasp_joint: np.ndarray,
        aim_joint2: np.ndarray,
        params,
        next_grasp: "Optional[np.ndarray]" = None,
        chain_park: "Optional[np.ndarray]" = None,
    ) -> None:
        """Build and dispatch throw trajectory using already-decoded ThrowParams.

        The NN throw arc (grasp → release → aim_joint2) is generated and
        velocity-clamped exactly as the trained swing, so the motion UP TO the
        release sample is identical to the original throw — same path, same
        velocity, same release timing. After the release the full arc is
        kept up to aim_joint2 (8 cm hover, dq(T)=0) and chained from rest to a
        safe park — a lifted-standby, or the shared idle pose when the queue is
        empty — so the arm parks high; the next pick is then selected and driven
        fresh next epoch (no cross-object pre-position).

        The timed suction release still fires at ``release_idx``
        (= eta_idx - lead_steps); in the cut case that is the last arc
        waypoint, so release timing/position/velocity are unchanged.
        """
        ctx = self.ctx
        throw_T = float(params.T)
        n_steps = max(2, int(throw_T * ctx.cfg.TRAJ_HZ))
        s = np.linspace(0.0, 1.0, n_steps + 1)

        traj_ext, vel_ext, _, _, ts_ext = new_trajectory(
            s, grasp_joint[:5], aim_joint2[:5], params.w, throw_T,
        )
        # traj_ext, vel_ext: (n_steps+1, 5); ts_ext: (n_steps+1,)

        # Clamp the throw to the robot's joint velocity limits. The NN throw is
        # time-parameterised (params.T) with NO joint-speed bound, so a large
        # joint sweep in a short T can command a queued segment faster than the
        # controller allows -> Yaskawa alarm 4414 "excessive segment velocity"
        # (seen on S/L). new_trajectory's path depends only on s; velocity ∝
        # 1/T, so stretching T by the over-limit ratio brings every segment
        # under the limit in one rescale (2nd pass guards the n_steps re-sample).
        m1_5 = np.asarray(ctx.M1[:5], dtype=float)
        for _ in range(2):
            dt_seg = np.maximum(np.diff(ts_ext), 1e-9)[:, None]
            seg_vel = np.abs(np.diff(traj_ext, axis=0)) / dt_seg   # (n,5) rad/s
            ratio = float(np.max(seg_vel / m1_5[None, :]))
            if ratio <= 1.0:
                break
            throw_T = throw_T * ratio * 1.05    # +5% margin
            n_steps = max(2, int(throw_T * ctx.cfg.TRAJ_HZ))
            s = np.linspace(0.0, 1.0, n_steps + 1)
            traj_ext, vel_ext, _, _, ts_ext = new_trajectory(
                s, grasp_joint[:5], aim_joint2[:5], params.w, throw_T,
            )
            ctx.log.warn(
                f"Throw clamped: seg vel ratio {ratio:.2f} > 1 "
                f"-> T {params.T:.3f}->{throw_T:.3f}s (n_steps {n_steps})"
            )

        eta_idx = int(round(params.eta * n_steps))
        eta_idx = max(0, min(eta_idx, n_steps))
        lead_steps = int(round(ctx.cfg.RELEASE_LEAD * ctx.cfg.TRAJ_HZ))
        release_idx = max(0, min(eta_idx - lead_steps, n_steps))

        # The arc + velocity clamp above are UNCHANGED, so the swing up to the
        # release sample is byte-identical to the original throw. Only the
        # post-release tail differs (see docstring):
        #   * next pick known -> CUT at the release sample and fly straight to
        #     it from the actual release state (large velocity); drop the
        #     wasted release→aim_joint2 follow-through.
        #   * no next pick    -> keep the full arc to aim_joint2 (8 cm hover,
        #     at rest) and chain to the idle/standby pose (idle_target()).
        # Either branch then appends one fresh time-optimal trajectory() from
        # (start_q5, start_dq5) to chain_target — only the start state / target
        # differ, so the concat/dispatch code below stays shared.
        # Uniform flow: always keep the FULL arc to aim_joint2 (8 cm hover, at rest) and
        # chain the follow-through to a safe park — home/idle when the queue is EMPTY, else
        # a lifted-standby near the last pose so the next pick approaches fresh from near the
        # belt (no wasted home round-trip). No cut-at-release / pre-position of a next object
        # (that throw-only chaining is what tangled push<->throw alternation). copy() so the
        # shared ctx.idle_joint is never mutated.
        traj_pre_5 = traj_ext.T                    # (5, n_steps+1) full arc
        vel_pre_5 = vel_ext.T
        ts_pre = ts_ext
        start_q5 = traj_ext[-1]                     # aim_joint2 at rest
        start_dq5 = vel_ext[-1]                     # ~0 (NN boundary condition)
        if chain_park is not None:
            # The NEXT object's skill supplied its own action-start park (push:
            # its backswing pose + clearance). Full 6-DOF pose, wrist included.
            chain_target = np.asarray(chain_park, dtype=float).copy()
            chain_dest = "next action-start park"
        elif next_grasp is not None:
            # Park OVER the next grasp (its XY raised to home Z), NOT at belt height. A
            # belt-height chain endpoint would leave the arm low, and the next epoch's
            # drive to a different-lane grasp would sweep the TCP low across the belt
            # (floor-dip / grazing). lifted_standby_joint keeps the XY, raises Z, and
            # parks the wrist at the shared PICK_WRIST_J6 baseline.
            chain_target = ctx.lifted_standby_joint(next_grasp)
            chain_dest = "over next grasp"
        elif not ctx.queue:
            chain_target = self.idle_target().copy()
            chain_dest = "home/idle (queue empty)"
        else:
            chain_target = ctx.lifted_standby_joint(aim_joint2)
            chain_dest = "lifted standby"

        # Make the two post-release destinations explicit in the HW log. This
        # distinguishes the NN arc endpoint (aim_joint2) from the appended
        # chain endpoint and makes any remaining XY mismatch directly visible.
        try:
            T_arc_end = ctx.robot.forward_kinematics(
                np.append(start_q5, float(grasp_joint[-1]))
            )
            T_chain_end = ctx.robot.forward_kinematics(chain_target[:6])
            ctx.log.info(
                "Throw endpoint TCP: "
                f"arc=({T_arc_end[0, 3]:+.3f}, {T_arc_end[1, 3]:+.3f}, "
                f"{T_arc_end[2, 3]:+.3f})m -> "
                f"chain=({T_chain_end[0, 3]:+.3f}, {T_chain_end[1, 3]:+.3f}, "
                f"{T_chain_end[2, 3]:+.3f})m [{chain_dest}]"
            )
        except Exception as exc:  # diagnostic only; never block a throw
            ctx.log.warn(f"Throw endpoint TCP logging failed: {exc}")

        # 6-DOF chain: the ARC is 5-DOF (the NN drives joints 1-5; J6 stays
        # parked at the pick wrist), but the chain must be able to ROTATE the
        # wrist toward the park target (e.g. the next push's backswing J6) —
        # so it runs on all six joints, starting from the arc end with J6 at
        # the held wrist and zero wrist velocity.
        held_wrist = float(grasp_joint[-1])
        start_q6 = np.append(start_q5, held_wrist)
        start_dq6 = np.append(start_dq5, 0.0)
        zero6 = np.zeros(6)
        traj_chain, vel_chain, ts_chain = trajectory(
            start_q6, start_dq6,
            chain_target[:6], zero6,
            ctx.M1[:6], ctx.M2[:6], hertz=ctx.cfg.TRAJ_HZ,
        )
        # Drop the chain's first column — it's start_q6 (the last sample of the
        # kept arc: aim_joint2 + held wrist), so it would be a duplicate. Slice
        # traj/vel/ts the SAME way so the three stay equal length: if the chain
        # degenerates to a single sample (chain_target ≈ start_q6 -> opt_time ≈
        # 0) all three become empty and only the arc remains. (Slicing ts alone
        # would leave traj/vel one column longer, and zip() in
        # _build_queue_waypoints would then silently drop a waypoint and bind
        # final_joint to the wrong timestamp.)
        traj_chain = traj_chain[:, 1:]
        vel_chain = vel_chain[:, 1:]
        ts_chain_shifted = ts_chain[1:] + ts_pre[-1]

        # Pad the ARC to 6-DOF (J6 held at the pick wrist, zero velocity —
        # zero-POSITION padding would snap the wrist to 0 at the throw start),
        # then append the 6-DOF chain.
        traj_pre_6 = pad(traj_pre_5.T, fill=held_wrist).T
        vel_pre_6 = pad(vel_pre_5.T).T
        traj_throw = np.concatenate((traj_pre_6, traj_chain), axis=1)   # (6, total)
        vel_throw = np.concatenate((vel_pre_6, vel_chain), axis=1)
        timestep_throw = np.concatenate((ts_pre, ts_chain_shifted))
        assert traj_throw.shape[1] == vel_throw.shape[1] == timestep_throw.shape[0], (
            f"throw traj/vel/timestep length mismatch: "
            f"{traj_throw.shape[1]}/{vel_throw.shape[1]}/{timestep_throw.shape[0]}"
        )
        final_joint = chain_target

        ctx.log.info(
            f"Throw T={params.T:.3f}s eta={params.eta:.3f} -> release step "
            f"{release_idx}/{traj_throw.shape[1] - 1} (eta step {eta_idx}, "
            f"lead {ctx.cfg.RELEASE_LEAD:.2f}s, full arc->{chain_dest})"
        )

        # Publish the exact path/release waypoint that is about to be sent to
        # the robot.  RViz visualization is best-effort and never alters the
        # control trajectory.
        evaluation = self._visualizer.publish(
            traj_throw, vel_throw, timestep_throw,
            release_index=release_idx,
            eta_index=eta_idx,
            release_lead=ctx.cfg.RELEASE_LEAD,
            throw_last_index=n_steps,
            eta_min=ctx.cfg.ETA_MIN,
            eta_max=ctx.cfg.ETA_MAX,
            trajectory_hz=ctx.cfg.TRAJ_HZ,
        )

        ctx.traj_ctrl.send_trajectory_queue_with_timed_release(
            traj_throw, vel_throw, timestep_throw,
            final_joint=final_joint,
            release_index=release_idx,
        )
        # The controller call blocks until the complete throw/park trajectory
        # finishes, so this is deliberately a post-throw assessment.  It is a
        # model-based prediction, not an observation of the physical object.
        self._visualizer.log_post_throw(evaluation)
        self._last_throw_meta = {
            "T": params.T, "eta": params.eta,
            "release_idx": release_idx, "n_steps": n_steps,
        }

    # ------------------------------------------------------------------
    # Per-cycle timing log
    # ------------------------------------------------------------------
    def _log_throw_cycle(self, target: "TrackedObject") -> None:
        """Append one pick-cycle timing row to PICK_LOG_CSV: suction-on ->
        throw-start -> release, for offline analysis of the release timing."""
        ctx = self.ctx
        path = ctx.cfg.PICK_LOG_CSV
        if not path:
            return
        lt = getattr(ctx.traj_ctrl, "last_throw", {}) or {}
        meta = self._last_throw_meta or {}
        son = getattr(ctx.traj_ctrl, "last_suction_on_t", None)
        t0 = lt.get("throw_start")
        trel = lt.get("release_wall")

        def _d(a, b):
            return round(a - b, 4) if (a is not None and b is not None) else ""

        row = {
            "iso_time": datetime.datetime.now().isoformat(timespec="milliseconds"),
            "class": target.class_name,
            "belt_mps": round(ctx.conveyor.current, 4),
            "suction_on_t": round(son, 4) if son else "",
            "throw_start_t": round(t0, 4) if t0 else "",
            "release_t": round(trel, 4) if trel else "",
            "on_to_throwstart_s": _d(t0, son),
            "throwstart_to_release_s": _d(trel, t0),
            "throw_T_s": round(meta.get("T", 0.0), 3),
            "eta": round(meta.get("eta", 0.0), 3),
            "release_idx": meta.get("release_idx", ""),
            "n_steps": meta.get("n_steps", ""),
            "io_ms": round(lt["io_ms"], 1) if lt.get("io_ms") is not None else "",
        }
        self._append_csv_row(path, row, ctx.log)

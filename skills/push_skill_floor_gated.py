from __future__ import annotations

import datetime
import os
import time
from typing import TYPE_CHECKING, Optional

import numpy as np

from gp8_control.skills.base import ManipulationSkill, SkillResult
from gp8_control.trajectory.trajectory_primitive import trajectory, trajectory_3points

if TYPE_CHECKING:
    from gp8_control.skills.context import PickRequest
    from gp8_control.tracking import TrackedObject

PUSH_CLASSES: set[str] = { "metal",}

PUSH_BIN_TARGET_MAP: dict[str, tuple] = {
    "transparent": (1.2, -0.30, 0.0),
    "metal":       (0.95,  0.60, 0.0),
}

PUSH_SPEED: float = float(os.environ.get("GP8_PUSH_SPEED", "2.0"))
PUSH_ACCEL: float = float(os.environ.get("GP8_PUSH_ACCEL", "12.0"))

PUSH_TRANSIT_ACCEL_MAX: float = float(os.environ.get("GP8_PUSH_ARC_ACCEL", "5.0"))
PUSH_TRANSIT_VEL_MAX: float = float(os.environ.get("GP8_PUSH_ARC_VEL", "1.4"))

PUSH_STROKE_HZ: float = 100.0

PUSH_LIFT_HEIGHT: float = 0.02

PUSH_CHAIN_PARK_LIFT: float = 0.0

PUSH_APPROACH_VIA_XY: float = 0.30

PUSH_PREPOSITION: bool = os.environ.get("GP8_PUSH_PREPOSITION", "1") != "0"

PUSH_TRANSIT_ARC_LIFT: float = float(os.environ.get("GP8_PUSH_ARC_LIFT", "0.05"))

# Uniform sine-lift amplitude for the unified single-arc chain transit
# (push_end -> next action start). Raises the mid-arc above the belt by
# construction so a low->low transit never skims the surface.
PUSH_TRANSIT_LIFT: float = float(os.environ.get("GP8_PUSH_TRANSIT_LIFT", "0.08"))

PUSH_TRANSIT_SAG_TOL: float = float(os.environ.get("GP8_PUSH_SAG_TOL", "0.002"))

PUSH_TRANSIT_HZ: float = 100.0

PUSH_PAD_DOWN: float = float(os.environ.get("GP8_PUSH_PAD_DOWN", "0.02"))
PUSH_PAD_FORE: float = float(os.environ.get("GP8_PUSH_PAD_FORE", "0.025"))
PUSH_PAD_HALFW: float = float(os.environ.get("GP8_PUSH_PAD_HALFW", "0.05"))

PUSH_FT_GAIN: float = 0.30
PUSH_FT_MIN: float = 0.12
PUSH_FT_MAX: float = 0.40

PUSH_END_MAX_RADIUS: float = 0.70

PUSH_DISTANCE: float = 0.3

PUSH_BACKSWING_DISTANCE: float = 0.2

PUSH_BACKSWING_MIN_X: float = 0.25

PUSH_JOINT6_ANGLE: float = - np.pi / 2.0

PUSH_LEAD_RESIDUAL = float(os.environ.get("GP8_PUSH_LEAD_RESIDUAL", "0.20"))

_FIXED_DELAY_PUSH_ENV = os.environ.get("GP8_FIXED_DELAY_PUSH")
PUSH_HEIGHT = float(os.environ.get("GP8_PUSH_HEIGHT", "0.01"))

PUSH_START_LIFT = 0.01

SWING_ANGLE: float = np.radians(20.0)

SWING_BIAS: float = np.radians(-5.0)

class PushSkill(ManipulationSkill):

    name = "push"

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._last_push_meta: dict = {}

    def can_handle(self, target: "TrackedObject") -> bool:
        return target.class_name in PUSH_CLASSES

    def arrival_lead(self) -> float:
        return self.dynamic_arrival_lead(PUSH_BACKSWING_DISTANCE)

    def dynamic_arrival_lead(self, contact_offset: float) -> float:
        if _FIXED_DELAY_PUSH_ENV is not None:
            return float(_FIXED_DELAY_PUSH_ENV)
        t_dispatch = self.ctx.traj_ctrl.qmode_ms_avg / 1000.0
        return self._stroke_time_to(contact_offset) + t_dispatch + PUSH_LEAD_RESIDUAL

    def t_to_contact(self, move_time: float) -> float:
        ctx = self.ctx
        t_setup = ctx.traj_ctrl.qmode_ms_avg / 1000.0
        t_position = move_time * ctx.cfg.OPT_TIME_TO_REAL
        t_pre_travel = self._stroke_time_to(PUSH_BACKSWING_DISTANCE)
        return 2.0 * t_setup + t_position + t_pre_travel

    def execute(self, request: "PickRequest") -> SkillResult:
        ctx = self.ctx
        target = request.target
        current_joint = request.current_joint
        aim_joint = request.aim_joint
        impact_joint = request.grasp_joint
        T_impact = request.T_grasp

        ctx.traj_ctrl.suction_off()

        bin_xyz = PUSH_BIN_TARGET_MAP.get(target.class_name)
        if bin_xyz is None:
            ctx.log.warn(f"no push bin for {target.class_name}; abort")
            return self._abort(f"no push bin mapping for {target.class_name}")
        bin_pos = np.asarray(bin_xyz, dtype=float)
        ctx.log.info(
            f"Push target for {target.class_name}: fixed bin "
            f"({bin_xyz[0]:+.3f}, {bin_xyz[1]:+.3f}, {bin_xyz[2]:+.3f}) m"
        )

        poses = self._compute_backswing_poses(T_impact, bin_pos, impact_joint)
        if poses is None:
            ctx.log.warn(
                f"Push pass: id={target.track_id} {target.class_name} — no run-up "
                f"can be secured (grasp too close to base or backswing unreachable); "
                f"skipping push"
            )
            return self._abort("no run-up available; object passed")
        backswing_joint, T_backswing = poses

        contact_offset = float( np.hypot(*(T_impact[:2, 3] - T_backswing[:2, 3])) )

        push_dir_exec = self._compute_push_direction(T_backswing, bin_pos)
        dist_bin = float(np.hypot(*(bin_pos[:2] - T_impact[:2, 3])))
        # TODO: can be tuned. clip values.
        follow_through = float( np.clip(PUSH_FT_GAIN * dist_bin, PUSH_FT_MIN, PUSH_FT_MAX) )
        push_distance = contact_offset + follow_through
        
        rx, ry = float(T_backswing[0, 3]), float(T_backswing[1, 3])
        b = rx * push_dir_exec[0] + ry * push_dir_exec[1]
        c = rx * rx + ry * ry - PUSH_END_MAX_RADIUS ** 2
        disc = b * b - c
        if disc >= 0.0:
            s_max = -b + float(np.sqrt(disc))
            if push_distance > s_max:
                ctx.log.info(
                    f"Push FT capped by reach: stroke {push_distance:.3f}→{s_max:.3f}m "
                    f"(end radius {PUSH_END_MAX_RADIUS:.2f}m)"
                )
                push_distance = max(s_max, contact_offset + 0.05)
        ctx.log.info(
            f"Push follow-through {push_distance - contact_offset:.3f}m "
            f"(contact→bin {dist_bin:.2f}m, run-up {contact_offset:.3f}m, "
            f"total {push_distance:.3f}m)"
        )

        # May be deleted
        aim_joint = aim_joint.copy()
        aim_joint[-1] = backswing_joint[-1]

        T_cur = ctx.robot.forward_kinematics(
            np.asarray(current_joint, dtype=float)[:6]
        )
        _xy_gap = float(np.hypot(*(T_cur[:2, 3] - T_backswing[:2, 3])))
        
        # May be deleted
        approach_via = None if _xy_gap < PUSH_APPROACH_VIA_XY else aim_joint

        v_belt = ctx.conveyor.current
        y_now = ctx.object_y_now(target, time.time(), v_belt)
        t_wait_est = max(0.0, (y_now - float(T_impact[1, 3])) / max(v_belt, 1e-6))
        push_time = self._stroke_time_to(push_distance)
        nxt = ctx.next_chain_target(backswing_joint, t_wait_est + push_time)
        next_grasp, next_cand = nxt if nxt is not None else (None, None)
        chain_park = None
        if next_cand is not None:
            chain_park = ctx.skill_obj_for(next_cand).chain_park_joint(
                next_grasp, next_cand
            )

        if PUSH_PREPOSITION:
            traj_push, vel_push, ts_push, final_joint, t_contact = (
                self.build_push_trajectory(
                    current_joint, backswing_joint, T_backswing,
                    bin_pos, next_grasp=next_grasp, append_chain=True,
                    append_descent=False, push_distance=push_distance,
                    contact_offset=contact_offset, chain_park=chain_park,
                    approach_via=None, dispatch=False,
                )
            )
            q_park = traj_push[:, 0].copy()
            ctx.set_status("POSITIONING", target.class_name)
            traj_pos, vel_pos, ts_pos = self._build_gated_approach(
                current_joint, q_park, via=approach_via,
            )
            ctx.traj_ctrl.send_trajectory_queue(
                traj_pos, vel_pos, ts_pos, final_joint=q_park,
            )
            if not ctx.traj_ctrl._wait_for_position(
                q_park, tolerance=0.05, timeout_sec=1.0
            ):
                _q_now = ctx.traj_ctrl.current_joints
                _gap = (
                    float(np.max(np.abs(
                        np.asarray(_q_now, dtype=float)[:6] - q_park[:6]
                    )))
                    if _q_now is not None else float("nan")
                )
                ctx.log.error(
                    f"POSITIONING did not settle within 1 s (max joint gap "
                    f"{_gap:.3f} rad) — the servo is not tracking the stream. "
                    f"Check the driver's axis_increment_factor (bare relaunch "
                    f"= 0.1 commissioning cap) / RT link before pushing."
                )
                return self._abort(
                    "positioning never settled — driver/link degraded?"
                )
        else:
            traj_push, vel_push, ts_push, final_joint, t_contact = (
                self.build_push_trajectory(
                    current_joint, backswing_joint, T_backswing,
                    bin_pos, next_grasp=next_grasp, append_chain=True,
                    append_descent=True, push_distance=push_distance,
                    contact_offset=contact_offset, chain_park=chain_park,
                    approach_via=approach_via, dispatch=False,
                )
            )

        if _FIXED_DELAY_PUSH_ENV is not None:
            lead = float(_FIXED_DELAY_PUSH_ENV)
        else:
            lead = (
                t_contact
                + ctx.traj_ctrl.qmode_ms_avg / 1000.0
                + PUSH_LEAD_RESIDUAL
            )
        _route = "via-hover" if approach_via is not None else "direct"
        _approach_txt = (
            f"stroke-only; pre-positioned {_xy_gap * 100:.0f}cm {_route}"
            if PUSH_PREPOSITION
            else f"approach {_xy_gap * 100:.0f}cm {_route}"
        )
        ctx.log.info(
            f"Push FIRE lead {lead * 1000:.0f}ms = traj contact "
            f"{t_contact * 1000:.0f}ms ({_approach_txt}) + dispatch "
            f"{ctx.traj_ctrl.qmode_ms_avg:.0f}ms + residual "
            f"{PUSH_LEAD_RESIDUAL * 1000:.0f}ms"
            + (" [OVERRIDDEN by GP8_FIXED_DELAY_PUSH]" if _FIXED_DELAY_PUSH_ENV else "")
        )
        ctx.set_status("WAITING", target.class_name)
        ctx.wait_for_arrival(target, T_impact[1, 3], offset=lead)

        stroke_end_y = T_backswing[1, 3] + push_distance * push_dir_exec[1]
        stroke_min_y = min(float(T_backswing[1, 3]), float(stroke_end_y))
        obj_y_now = ctx.object_y_now(target, time.time(), ctx.conveyor.current)
        if obj_y_now < stroke_min_y:
            ctx.log.warn(
                f"Push abort: id={target.track_id} {target.class_name} already past the "
                f"stroke (y={obj_y_now:+.3f} < stroke_min {stroke_min_y:+.3f}); "
                f"skipping stale stroke"
            )
            return self._abort("object passed stroke span; push aborted")

        ctx.log_action_timing(target, T_impact[1, 3], "push-fire")
        ctx.set_status("PUSHING", target.class_name)
        ctx.traj_ctrl.send_trajectory_queue(
            traj_push, vel_push, ts_push, final_joint=final_joint,
        )

        ctx.traj_ctrl.suction_off()
        self._log_push_cycle(target)
        ctx.set_active_target(None)
        ctx.set_status("IDLE", "")
        return SkillResult(True, "push complete")

    def _compute_backswing_poses(
        self,
        T_impact: np.ndarray,
        bin_pos: np.ndarray,
        impact_joint: np.ndarray,
    ) -> "Optional[tuple[np.ndarray, np.ndarray]]":
        """Backswing (stroke-start) pose retreated behind ``T_impact`` opposite
        the push direction, or ``None`` when no run-up can be secured — the
        caller then passes on the object (no push attempt).

        The retreat is clamped so the backswing X never drops below
        ``PUSH_BACKSWING_MIN_X`` (a reduced but still valid run-up). It gives up
        — returning ``None`` — only when there is NO room to retreat at all (the
        object was grasped at X <= ``PUSH_BACKSWING_MIN_X``) or the backswing
        pose is unreachable.

        Returns ``(backswing_joint, T_backswing)``.
        """
        ctx = self.ctx
        push_dir = self._compute_push_direction(T_impact, bin_pos)
        dx = PUSH_BACKSWING_DISTANCE * push_dir[0]
        dy = PUSH_BACKSWING_DISTANCE * push_dir[1]

        if T_impact[0, 3] - dx < PUSH_BACKSWING_MIN_X and dx > 0:
            available = max(0.0, T_impact[0, 3] - PUSH_BACKSWING_MIN_X)
            ctx.log.warn(
                f"[push-runup] grasp x={T_impact[0, 3]:+.3f}, "
                f"MIN_X={PUSH_BACKSWING_MIN_X:.3f} -> available={available:+.3f} "
                f"(push_dir=({push_dir[0]:+.2f},{push_dir[1]:+.2f}), "
                f"retreat dx={dx:+.3f})"
            )
            if available <= 0:
                # No room to retreat: the object was grasped at
                # X <= PUSH_BACKSWING_MIN_X, so there is no space for a run-up.
                # Pass the object (no push attempt).
                #
                # TODO(no-runup): a zero-run-up push (paddle placed AT the
                # object's XY ~2cm, follow-through sweeps it from standstill —
                # the old no-runup fallback, case A) IS geometrically possible
                # and MAY work for light / low-friction objects. Removed for now:
                # no pre-contact momentum, and the paddle would sit on the
                # object's arrival point so the object slides into a stationary
                # paddle (premature / side contact). Revisit from experiment and
                # record the verdict here.
                return None
            scale = available / dx
            dx *= scale
            dy *= scale

        R_backswing = self._push_orientation(push_dir, self._swing_at(0.0))

        T_backswing = T_impact.copy()
        T_backswing[:3, :3] = R_backswing
        T_backswing[0, 3] -= dx
        T_backswing[1, 3] -= dy
        T_backswing[2, 3] = PUSH_HEIGHT + PUSH_START_LIFT

        ik_backswing = ctx.robot.inverse_kinematics(
            T_backswing, q_init=np.asarray(impact_joint, dtype=float)[:6]
        )
        if ik_backswing is None:
            # Backswing pose out of the workspace (typical far-downstream
            # intercept). No run-up available → pass the object (same decision
            # as the no-runup TODO above).
            ctx.log.warn(
                f"Backswing IK unreachable at "
                f"({T_backswing[0, 3]:+.3f}, {T_backswing[1, 3]:+.3f}); "
                f"passing object (no run-up)"
            )
            return None
        backswing_joint = np.asarray(ik_backswing, dtype=float)

        return backswing_joint, T_backswing

    def chain_park_joint(
        self,
        next_grasp: np.ndarray,
        next_cand: "TrackedObject",
    ) -> "Optional[np.ndarray]":
        ctx = self.ctx
        bin_xyz = PUSH_BIN_TARGET_MAP.get(next_cand.class_name)
        if bin_xyz is None:
            return None
        q_next = np.asarray(next_grasp, dtype=float)[:6]
        T_next = ctx.robot.forward_kinematics(q_next)
        d = np.array(
            [bin_xyz[0] - T_next[0, 3], bin_xyz[1] - T_next[1, 3], 0.0]
        )
        nrm = float(np.linalg.norm(d))
        if nrm < 1e-6:
            return None
        push_dir = d / nrm
        dx = PUSH_BACKSWING_DISTANCE * push_dir[0]
        dy = PUSH_BACKSWING_DISTANCE * push_dir[1]
        if T_next[0, 3] - dx < PUSH_BACKSWING_MIN_X and dx > 0:
            available = max(0.0, T_next[0, 3] - PUSH_BACKSWING_MIN_X)
            if available <= 0:
                return None
            scale = available / dx
            dx *= scale
            dy *= scale
        T_park = np.eye(4)
        T_park[:3, :3] = self._push_orientation(push_dir, self._swing_at(0.0))
        T_park[0, 3] = T_next[0, 3] - dx
        T_park[1, 3] = T_next[1, 3] - dy
        T_park[2, 3] = PUSH_HEIGHT + PUSH_START_LIFT + PUSH_CHAIN_PARK_LIFT
        q = ctx.robot.inverse_kinematics(T_park, q_init=q_next)
        return None if q is None else np.asarray(q, dtype=float)

    def build_push_trajectory(
        self,
        aim_joint: np.ndarray,
        backswing_joint: np.ndarray,
        T_backswing: np.ndarray,
        bin_pos: np.ndarray,
        next_grasp: "Optional[np.ndarray]" = None,
        append_chain: bool = True,
        append_descent: bool = True,
        push_distance: float = PUSH_DISTANCE,
        contact_offset: "Optional[float]" = None,
        chain_park: "Optional[np.ndarray]" = None,
        approach_via: "Optional[np.ndarray]" = None,
        dispatch: bool = True,
    ) -> "Optional[tuple]":
        ctx = self.ctx
        zero6 = np.zeros(6)
        seg_traj: list = []
        seg_vel: list = []
        seg_ts: list = []

        n_desc = 0
        desc_end_t = 0.0
        if append_descent:
            traj_desc, vel_desc, ts_desc = self._build_gated_approach(
                aim_joint, backswing_joint, via=approach_via,
            )
            seg_traj.append(traj_desc)
            seg_vel.append(vel_desc)
            seg_ts.append(ts_desc)
            n_desc = traj_desc.shape[1]
            desc_end_t = ts_desc[-1]

        push_dir = self._compute_push_direction(T_backswing, bin_pos)
        theta = float(np.arctan2(push_dir[1], push_dir[0]))
        traj_stroke, vel_stroke, ts_stroke = self._build_push_stroke(
            T_backswing, push_dir, backswing_joint, push_distance,
            contact_offset=contact_offset,
        )

        _stroke_T_pre = float(ts_stroke[-1]) if len(ts_stroke) else 1e-9
        traj_stroke, vel_stroke, ts_stroke = self._clamp_stroke_velocity(
            traj_stroke, vel_stroke, ts_stroke,
        )
        _stretch = float(ts_stroke[-1]) / max(_stroke_T_pre, 1e-9)

        if append_descent and seg_traj:
            _q_seam = seg_traj[-1][:, -1]
            _gap = float(np.max(np.abs(_q_seam - traj_stroke[:, 0])))
            if _gap > 0.15:
                ctx.log.warn(
                    f"Approach/stroke seam gap {_gap:.3f} rad — inserting "
                    f"time-optimal bridge (no-runup fallback path)"
                )
                traj_b, vel_b, ts_b = trajectory(
                    _q_seam, zero6, traj_stroke[:, 0], zero6,
                    ctx.M1[:6], ctx.M2[:6], hertz=ctx.cfg.TRAJ_HZ,
                )
                sag_b, t_sag_b = self._transit_sag(
                    traj_b, vel_b, ts_b, q_end=traj_stroke[:, 0],
                )
                if sag_b < -PUSH_TRANSIT_SAG_TOL:
                    ctx.log.warn(
                        f"Seam bridge sags {sag_b * 1000:.1f}mm below "
                        f"endpoints @t={t_sag_b:.2f}s → Cartesian arc transit"
                    )
                    arc_b = self._build_arc_transit(
                        _q_seam, traj_stroke[:, 0],
                        lift=PUSH_TRANSIT_ARC_LIFT,
                        duration_hint=float(ts_b[-1]),
                    )
                    if arc_b is not None:
                        traj_b, vel_b, ts_b = arc_b
                    else:
                        ctx.log.warn(
                            "Bridge arc IK failed — keeping the sagging "
                            "joint-space bridge (floor-strike risk!)"
                        )
                seg_traj.append(traj_b[:, 1:])
                seg_vel.append(vel_b[:, 1:])
                seg_ts.append(ts_b[1:] + desc_end_t)
                n_desc += traj_b.shape[1] - 1
                desc_end_t += float(ts_b[-1])

        _s_hit = float(np.clip(contact_offset, 0.0, push_distance)) \
            if contact_offset is not None else 0.0
        t_contact = desc_end_t + self._stroke_time_to(_s_hit) * _stretch

        if append_descent:
            ts_stroke = ts_stroke[1:] + desc_end_t
            traj_stroke = traj_stroke[:, 1:]
            vel_stroke = vel_stroke[:, 1:]
        seg_traj.append(traj_stroke)
        seg_vel.append(vel_stroke)
        seg_ts.append(ts_stroke)
        n_stroke = traj_stroke.shape[1]

        # The stroke already ends at rest (_build_push_stroke zeros vel[:, -1]).
        # The unified chain arc is rest-to-rest, so no belt-parallel exit-velocity
        # carry is needed here (the old hover-via chain carried push_end_dq into
        # leg1; the single arc does not).
        push_end_q = traj_stroke[:, -1]

        if append_chain:
            if chain_park is not None:
                chain_target = np.asarray(chain_park, dtype=float).copy()
                chain_dest = "next action-start park"
            elif next_grasp is not None:
                chain_target = ctx.lifted_standby_joint(next_grasp)
                chain_dest = "over next grasp"
            elif not ctx.queue:
                chain_target = self.idle_target().copy()
                chain_dest = "home/idle (queue empty)"
            else:
                chain_target = ctx.lifted_standby_joint(push_end_q)
                chain_dest = "lifted standby"

            # Single EE-controlled arc: push_end -> chain_target with a uniform
            # sine lift (PUSH_TRANSIT_LIFT) that clears the belt by construction.
            # Replaces the old hover-via two-segment chain (joint leg1 up to a
            # fixed hover height + arc leg2 down) and its full stop at the
            # hover. The arc is rest-to-rest; the stroke already ends at
            # rest, so the stroke->arc seam is continuous. IK failure (e.g. a
            # long/awkward traverse) -> hold at push_end; the next action's
            # descent repositions from there.
            chain_arc = self._build_arc_transit(
                push_end_q[:6],
                np.asarray(chain_target, dtype=float)[:6],
                lift=PUSH_TRANSIT_LIFT,
            )
            if chain_arc is None:
                ctx.log.warn(
                    "Chain arc IK failed; skipping chain — holding at "
                    "push_end (next action's descent repositions)"
                )
                chain_dest = "push_end (chain arc IK failed)"
                final_joint = push_end_q
            else:
                traj_chain, vel_chain, ts_chain = chain_arc
                stroke_end_t = (
                    seg_ts[-1][-1] if len(seg_ts[-1]) > 0 else desc_end_t
                )
                traj_chain = traj_chain[:, 1:]
                vel_chain = vel_chain[:, 1:]
                ts_chain_shifted = ts_chain[1:] + stroke_end_t

                seg_traj.append(traj_chain)
                seg_vel.append(vel_chain)
                seg_ts.append(ts_chain_shifted)
                final_joint = chain_target
        else:
            chain_dest = "push_end (no chain)"
            final_joint = push_end_q

        traj_push = np.concatenate(seg_traj, axis=1)
        vel_push = np.concatenate(seg_vel, axis=1)
        ts_full = np.concatenate(seg_ts)

        assert traj_push.shape[1] == vel_push.shape[1] == ts_full.shape[0], (
            f"push traj/vel/ts length mismatch: "
            f"{traj_push.shape[1]}/{vel_push.shape[1]}/{ts_full.shape[0]}"
        )
        v_contact = None
        contact_note = ""
        if contact_offset is not None:
            v_contact = min(
                PUSH_SPEED,
                float(np.sqrt(2.0 * PUSH_ACCEL * max(contact_offset, 0.0))),
            )
            contact_note = f", contact@{contact_offset:.3f}m ≈{v_contact:.2f}m/s"
        swing_txt = (
            f"scoop swing "
            f"{np.degrees(SWING_BIAS - SWING_ANGLE):+.0f}°→0°@hit→"
            f"{np.degrees(SWING_BIAS + SWING_ANGLE):+.0f}°"
        )
        ctx.log.info(
            f"Push traj: descent {n_desc} + stroke {n_stroke} steps "
            f"(d={push_distance:.3f}m, cruise {PUSH_SPEED:.2f}m/s{contact_note}, "
            f"θ={np.degrees(theta):.1f}°, {swing_txt}, "
            f"lift {PUSH_LIFT_HEIGHT * 100:.0f}cm), chain→{chain_dest}"
        )

        self._last_push_meta = {
            "n_descent": n_desc,
            "n_stroke": n_stroke,
            "push_distance": push_distance,
            "push_speed": PUSH_SPEED,
            "contact_offset": contact_offset,
            "v_contact": v_contact,
            "t_contact": t_contact,
            "theta": theta,
            "swing": self._swing_at(1.0),
            "chain_dest": chain_dest,
        }

        if not dispatch:
            return traj_push, vel_push, ts_full, final_joint, t_contact

        ctx.traj_ctrl.send_trajectory_queue(
            traj_push, vel_push, ts_full, final_joint=final_joint,
        )
        return None

    @staticmethod
    def _stroke_time_to(s: float) -> float:
        v = max(PUSH_SPEED, 1e-6)
        a = max(PUSH_ACCEL, 1e-6)
        s = max(float(s), 0.0)
        s_acc = v * v / (2.0 * a)
        if s <= s_acc:
            return float(np.sqrt(2.0 * s / a))
        return v / a + (s - s_acc) / v

    @classmethod
    def _stroke_profile(cls, distance: float) -> tuple[np.ndarray, np.ndarray]:
        v = max(PUSH_SPEED, 1e-6)
        a = max(PUSH_ACCEL, 1e-6)
        D = max(float(distance), 1e-9)
        T = cls._stroke_time_to(D)
        n = max(2, int(np.ceil(T * PUSH_STROKE_HZ)))
        ts = np.linspace(0.0, T, n + 1)
        t_acc = v / a
        s_acc = v * v / (2.0 * a)
        s_arr = np.where(
            ts < t_acc,
            0.5 * a * ts ** 2,
            s_acc + v * (ts - t_acc),
        )
        s_arr = np.clip(s_arr, 0.0, D)
        s_arr[-1] = D
        return ts, s_arr

    @staticmethod
    def _compute_push_direction(
        T_from: np.ndarray,
        bin_pos: np.ndarray,
    ) -> np.ndarray:
        delta = np.asarray(bin_pos, dtype=float) - T_from[:3, 3]
        delta[2] = 0.0
        norm = np.linalg.norm(delta)
        if norm < 1e-6:
            return np.array([1.0, 0.0, 0.0])
        return delta / norm

    @staticmethod
    def _swing_at(alpha: float, alpha_hit: "Optional[float]" = None) -> float:
        back = SWING_BIAS - SWING_ANGLE
        fwd = SWING_BIAS + SWING_ANGLE
        if alpha_hit is None or not (0.02 <= alpha_hit <= 0.98):
            return SWING_BIAS + SWING_ANGLE * (2.0 * alpha - 1.0)
        if alpha <= alpha_hit:
            return back * (1.0 - alpha / alpha_hit)
        return fwd * (alpha - alpha_hit) / (1.0 - alpha_hit)

    @staticmethod
    def _push_orientation(push_dir: np.ndarray, swing: float) -> np.ndarray:
        f = np.array([push_dir[0], push_dir[1], 0.0], dtype=float)
        nf = np.linalg.norm(f)
        f = f / nf if nf > 1e-9 else np.array([1.0, 0.0, 0.0])
        up = np.array([0.0, 0.0, 1.0])
        side = np.cross(up, f)
        c, s = np.cos(swing), np.sin(swing)
        approach = -c * up + s * f
        facing = s * up + c * f

        cb, sb = np.cos(-PUSH_JOINT6_ANGLE), np.sin(-PUSH_JOINT6_ANGLE)
        side, facing = cb * side + sb * facing, -sb * side + cb * facing
        return np.column_stack((approach, side, facing))

    def _build_push_stroke(
        self,
        T_backswing: np.ndarray,
        direction: np.ndarray,
        backswing_joint: np.ndarray,
        push_distance: float = PUSH_DISTANCE,
        contact_offset: "Optional[float]" = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ctx = self.ctx
        ts, s_arr = self._stroke_profile(push_distance)
        n_samples = len(s_arr)

        alpha_hit: "Optional[float]" = None
        s_hit: "Optional[float]" = None
        if contact_offset is not None:
            frac = float(np.clip(contact_offset / max(push_distance, 1e-9), 0.0, 1.0))
            if 0.02 <= frac <= 0.98:
                alpha_hit = frac
                s_hit = frac * push_distance

        waypoints: list[np.ndarray] = []
        q_seed = np.asarray(backswing_joint, dtype=float)
        for i in range(n_samples):
            s = float(s_arr[i])
            alpha = s / max(push_distance, 1e-9)
            T_wp = T_backswing.copy()
            T_wp[:3, :3] = self._push_orientation(
                direction, self._swing_at(alpha, alpha_hit)
            )
            T_wp[0, 3] += s * direction[0]
            T_wp[1, 3] += s * direction[1]
            z = PUSH_HEIGHT
            if s_hit is not None:
                if s < s_hit and PUSH_START_LIFT > 0.0:
                    ratio = s / s_hit
                    z += PUSH_START_LIFT * 0.5 * (1.0 + np.cos(np.pi * ratio))
                elif s > s_hit and PUSH_LIFT_HEIGHT > 0.0:
                    ratio = (s - s_hit) / max(push_distance - s_hit, 1e-9)
                    z += PUSH_LIFT_HEIGHT * 0.5 * (1.0 - np.cos(np.pi * ratio))
            elif PUSH_START_LIFT > 0.0:
                ramp = 0.3 * push_distance
                if s < ramp:
                    z += PUSH_START_LIFT * 0.5 * (1.0 + np.cos(np.pi * s / ramp))
            T_wp[2, 3] = z
            ik = ctx.robot.inverse_kinematics(T_wp, q_init=q_seed)
            if ik is None:
                ctx.log.warn(
                    f"Push stroke IK failed at step {i}/{n_samples - 1}; "
                    f"truncating stroke to {len(waypoints)} waypoints"
                )
                break
            q_seed = np.asarray(ik, dtype=float)
            waypoints.append(q_seed)

        if len(waypoints) < 2:
            ctx.log.warn("Push stroke degenerate (< 2 IK solutions); no-op segment")
            q0 = np.asarray(backswing_joint, dtype=float)
            traj = np.column_stack([q0, q0])
            vel = np.zeros_like(traj)
            ts = np.array([0.0, 1.0 / PUSH_STROKE_HZ])
            return traj, vel, ts

        n = len(waypoints)
        traj = np.column_stack(waypoints)
        ts = ts[:n]

        vel = np.gradient(traj, ts, axis=1) if n > 2 else np.zeros_like(traj)
        vel[:, 0] = 0.0
        vel[:, -1] = 0.0

        return traj, vel, ts

    def _clamp_stroke_velocity(
        self,
        traj: np.ndarray,
        vel: np.ndarray,
        ts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ctx = self.ctx
        m1 = np.asarray(ctx.M1[: traj.shape[0]], dtype=float)

        for _ in range(2):
            dt_seg = np.maximum(np.diff(ts), 1e-9)
            seg_vel = np.abs(np.diff(traj, axis=1)) / dt_seg[None, :]
            ratio = float(np.max(seg_vel / m1[:, None]))
            if ratio <= 1.0:
                break
            scale = ratio * 1.05
            ts = ts * scale

            n = traj.shape[1]
            vel = np.gradient(traj, ts, axis=1) if n > 2 else np.zeros_like(traj)
            vel[:, 0] = 0.0
            vel[:, -1] = 0.0

            ctx.log.warn(
                f"Push stroke clamped: seg vel ratio {ratio:.2f} > 1 "
                f"→ time scale ×{scale:.2f}"
            )
        return traj, vel, ts

    def _build_gated_approach(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        via: "Optional[np.ndarray]" = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ctx = self.ctx
        zero6 = np.zeros(6)
        q_start = np.asarray(q_start, dtype=float)[:6]
        q_goal = np.asarray(q_goal, dtype=float)[:6]
        if via is not None:
            traj_desc, vel_desc, ts_desc = trajectory_3points(
                q_start, zero6,
                np.asarray(via, dtype=float)[:6], zero6,
                q_goal, zero6,
                ctx.M1[:6], ctx.M2[:6], hertz=ctx.cfg.TRAJ_HZ,
            )
        else:
            traj_desc, vel_desc, ts_desc = trajectory(
                q_start, zero6,
                q_goal, zero6,
                ctx.M1[:6], ctx.M2[:6], hertz=ctx.cfg.TRAJ_HZ,
            )
        keep = np.concatenate(([True], np.diff(ts_desc) > 1e-9))
        traj_desc = traj_desc[:, keep]
        vel_desc = vel_desc[:, keep]
        ts_desc = ts_desc[keep]

        sag, t_sag = self._transit_sag(traj_desc, vel_desc, ts_desc,
                                       q_end=q_goal)
        if sag < -PUSH_TRANSIT_SAG_TOL:
            if via is not None:
                ctx.log.warn(
                    f"Via-hover approach sags {sag * 1000:.1f}mm below its "
                    f"endpoints @t={t_sag:.2f}s → Cartesian final descent"
                )
                _via6 = np.asarray(via, dtype=float)[:6]
                leg2 = self._build_arc_transit(_via6, q_goal, lift=0.0)
                if leg2 is None:
                    ctx.log.warn(
                        "Via-descent arc IK failed — keeping the SAGGING "
                        "via-hover approach (floor-strike risk!)"
                    )
                else:
                    traj_l1, vel_l1, ts_l1 = trajectory(
                        q_start, zero6, _via6, zero6,
                        ctx.M1[:6], ctx.M2[:6], hertz=ctx.cfg.TRAJ_HZ,
                    )
                    traj_l2, vel_l2, ts_l2 = leg2
                    _l1_end = float(ts_l1[-1])
                    traj_desc = np.concatenate(
                        [traj_l1, traj_l2[:, 1:]], axis=1
                    )
                    vel_desc = np.concatenate([vel_l1, vel_l2[:, 1:]], axis=1)
                    ts_desc = np.concatenate([ts_l1, ts_l2[1:] + _l1_end])
                    keep = np.concatenate(
                        ([True], np.diff(ts_desc) > 1e-9)
                    )
                    traj_desc = traj_desc[:, keep]
                    vel_desc = vel_desc[:, keep]
                    ts_desc = ts_desc[keep]
            else:
                ctx.log.warn(
                    f"Direct descent sags {sag * 1000:.1f}mm below "
                    f"endpoints @t={t_sag:.2f}s → Cartesian arc transit "
                    f"(lift {PUSH_TRANSIT_ARC_LIFT * 100:.0f}cm)"
                )
                arc = self._build_arc_transit(
                    q_start, q_goal,
                    lift=PUSH_TRANSIT_ARC_LIFT,
                    duration_hint=float(ts_desc[-1]),
                )
                if arc is not None:
                    traj_desc, vel_desc, ts_desc = arc
                else:
                    T_from = ctx.robot.forward_kinematics(q_start)
                    T_to = ctx.robot.forward_kinematics(q_goal)
                    T_mid = T_to.copy()
                    T_mid[:2, 3] = 0.5 * (T_from[:2, 3] + T_to[:2, 3])
                    T_mid[2, 3] = (
                        max(float(T_from[2, 3]), float(T_to[2, 3]))
                        + PUSH_TRANSIT_ARC_LIFT
                    )
                    q_mid = ctx.robot.inverse_kinematics(
                        T_mid, q_init=q_goal
                    )
                    if q_mid is not None:
                        traj_desc, vel_desc, ts_desc = trajectory_3points(
                            q_start, zero6,
                            np.asarray(q_mid, dtype=float)[:6], zero6,
                            q_goal, zero6,
                            ctx.M1[:6], ctx.M2[:6], hertz=ctx.cfg.TRAJ_HZ,
                        )
                        keep = np.concatenate(
                            ([True], np.diff(ts_desc) > 1e-9)
                        )
                        traj_desc = traj_desc[:, keep]
                        vel_desc = vel_desc[:, keep]
                        ts_desc = ts_desc[keep]
                    else:
                        ctx.log.warn(
                            "Arc AND trapezoid-mid IK failed — keeping the "
                            "SAGGING direct descent (floor-strike risk!)"
                        )
        return traj_desc, vel_desc, ts_desc

    def _paddle_bottom_z(self, T: np.ndarray) -> float:
        corners = np.array([
            [PUSH_PAD_DOWN,  PUSH_PAD_FORE,  PUSH_PAD_HALFW],
            [PUSH_PAD_DOWN,  PUSH_PAD_FORE, -PUSH_PAD_HALFW],
            [PUSH_PAD_DOWN, -PUSH_PAD_FORE,  PUSH_PAD_HALFW],
            [PUSH_PAD_DOWN, -PUSH_PAD_FORE, -PUSH_PAD_HALFW],
        ])
        pts = (T[:3, :3] @ corners.T).T + T[:3, 3]
        return float(np.min(pts[:, 2]))

    def _transit_sag(self, traj, vel, ts, q_end=None) -> tuple[float, float]:
        ctx = self.ctx
        arr = np.asarray(traj, dtype=float)
        varr = np.asarray(vel, dtype=float)
        times = np.asarray(ts, dtype=float).ravel()
        if q_end is not None:
            arr = np.concatenate(
                [arr, np.asarray(q_end, dtype=float)[:arr.shape[0]].reshape(-1, 1)],
                axis=1,
            )
            varr = np.concatenate([varr, np.zeros((varr.shape[0], 1))], axis=1)
            times = np.append(times, times[-1] + 0.05)
        if arr.shape[1] < 2 or float(times[-1]) - float(times[0]) <= 0.0:
            return 0.0, 0.0
        dt = 1.0 / PUSH_TRANSIT_HZ
        grid = np.arange(float(times[0]), float(times[-1]) + dt, dt)
        if varr.shape == arr.shape and np.all(np.diff(times) > 0):
            try:
                from scipy.interpolate import CubicHermiteSpline
                samples = np.column_stack([
                    CubicHermiteSpline(times, arr[j], varr[j])(grid)
                    for j in range(arr.shape[0])
                ])
                seg = np.clip(
                    np.searchsorted(times, grid, side="right") - 1,
                    0, len(times) - 2,
                )
                lo = np.minimum(arr[:, seg], arr[:, seg + 1]).T
                hi = np.maximum(arr[:, seg], arr[:, seg + 1]).T
                samples = np.clip(samples, lo, hi)
            except Exception:
                samples = np.column_stack(
                    [np.interp(grid, times, arr[j]) for j in range(arr.shape[0])]
                )
        else:
            samples = np.column_stack(
                [np.interp(grid, times, arr[j]) for j in range(arr.shape[0])]
            )
        bz = np.array([
            self._paddle_bottom_z(ctx.robot.forward_kinematics(samples[i, :6]))
            for i in range(samples.shape[0])
        ])
        i_min = int(np.argmin(bz))
        sag = float(bz[i_min] - min(float(bz[0]), float(bz[-1])))
        return sag, float(grid[i_min])

    def _build_arc_transit(
        self,
        q_from: np.ndarray,
        q_to: np.ndarray,
        lift: float,
        duration_hint: "Optional[float]" = None,
    ) -> "Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]":
        ctx = self.ctx
        try:
            from scipy.spatial.transform import Rotation, Slerp
        except Exception:
            ctx.log.warn("scipy Rotation/Slerp unavailable; no arc transit")
            return None

        q_from = np.asarray(q_from, dtype=float)[:6]
        q_to = np.asarray(q_to, dtype=float)[:6]
        T_from = ctx.robot.forward_kinematics(q_from)
        T_to = ctx.robot.forward_kinematics(q_to)
        p_from, p_to = T_from[:3, 3], T_to[:3, 3]

        slerp = Slerp(
            [0.0, 1.0],
            Rotation.from_matrix(np.stack([T_from[:3, :3], T_to[:3, :3]])),
        )

        def _pos_at(u: float) -> np.ndarray:
            p = p_from + (p_to - p_from) * u
            return np.array(
                [p[0], p[1], p[2] + lift * np.sin(np.pi * u)]
            )

        u_probe = np.linspace(0.0, 1.0, 21)
        pts = np.stack([_pos_at(float(u)) for u in u_probe])
        path_len = float(
            np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
        )
        T_dur = float(duration_hint) if duration_hint is not None else 0.0
        if T_dur <= 1e-3:
            T_dur = path_len / 0.5
        T_acc = float(np.pi * np.sqrt(
            path_len / (2.0 * max(PUSH_TRANSIT_ACCEL_MAX, 1e-6))
        ))
        T_vel = float(np.pi * path_len / (2.0 * max(PUSH_TRANSIT_VEL_MAX, 1e-6)))
        T_env = max(T_acc, T_vel)
        if T_env > T_dur + 1e-3:
            ctx.log.info(
                f"Arc transit slowed for tracking: {T_dur:.2f}s → {T_env:.2f}s "
                f"(path {path_len:.2f}m, accel ≤{PUSH_TRANSIT_ACCEL_MAX:.0f}m/s², "
                f"vel ≤{PUSH_TRANSIT_VEL_MAX:.1f}m/s)"
            )
            T_dur = T_env
        T_dur = max(T_dur, 4.0 / PUSH_TRANSIT_HZ)

        n = max(2, int(np.ceil(T_dur * PUSH_TRANSIT_HZ)))
        ts = np.linspace(0.0, T_dur, n + 1)
        u_t = 0.5 - 0.5 * np.cos(np.pi * ts / T_dur)
        s_u = u_t * u_t * (3.0 - 2.0 * u_t)
        R_all = slerp(s_u).as_matrix()

        waypoints: list[np.ndarray] = []
        q_seed = q_from
        for i in range(len(ts)):
            T_wp = np.eye(4)
            T_wp[:3, :3] = R_all[i]
            T_wp[:3, 3] = _pos_at(float(u_t[i]))
            ik = ctx.robot.inverse_kinematics(T_wp, q_init=q_seed)
            if ik is None:
                ctx.log.warn(
                    f"Arc transit IK failed at sample {i}/{len(ts) - 1} "
                    f"(u={u_t[i]:.2f}); falling back to the caller's plan-B"
                )
                return None
            q_seed = np.asarray(ik, dtype=float)
            waypoints.append(q_seed)

        traj = np.column_stack(waypoints)
        for idx, q_ref in ((0, q_from), (-1, q_to)):
            gap = float(np.max(np.abs(traj[:, idx] - q_ref)))
            if gap > 1e-3:
                ctx.log.warn(
                    f"Arc transit endpoint IK round-trip off by {gap:.4f} rad "
                    f"(wrist branch?); snapping to the exact endpoint"
                )
            traj[:, idx] = q_ref
        vel = (
            np.gradient(traj, ts, axis=1)
            if traj.shape[1] > 2 else np.zeros_like(traj)
        )
        vel[:, 0] = 0.0
        vel[:, -1] = 0.0
        return self._clamp_stroke_velocity(traj, vel, ts)

    def _log_push_cycle(self, target: "TrackedObject") -> None:
        ctx = self.ctx
        path = ctx.cfg.PICK_LOG_CSV
        if not path:
            return

        meta = self._last_push_meta or {}

        row = {
            "iso_time": datetime.datetime.now().isoformat(timespec="milliseconds"),
            "skill": "push",
            "class": target.class_name,
            "belt_mps": round(ctx.conveyor.current, 4),
            "push_speed_mps": meta.get("push_speed", ""),
            "push_distance_m": meta.get("push_distance", ""),
            "theta_deg": round(np.degrees(meta.get("theta", 0.0)), 1),
            "swing_mode": "scoop",
            "swing_deg": round(np.degrees(meta.get("swing", 0.0)), 1),
            "n_descent": meta.get("n_descent", ""),
            "n_stroke": meta.get("n_stroke", ""),
            "chain_dest": meta.get("chain_dest", ""),
        }
        self._append_csv_row(path, row, ctx.log)

"""Safe PPO residuals at the app simulator's existing trajectory seam.

This is deliberately independent of the retired Gym simulator.  It wraps one
MuJoCo backend instance, leaving target selection, skills, the 250 Hz stream
engine, suction commands and the simulator's physics unchanged.  A zero action
passes the exact original arrays to that stream engine.
"""
from __future__ import annotations

import hashlib
from typing import Callable

import numpy as np


OBSERVATION_SIZE = 97
ACTION_SIZE = 6
MAX_RESIDUAL_SCALE = .08
PHASES = ("throw", "push_route", "push_prepare")
GUARD_POLICY = ("strict_joint_bounds; strict_M1_for_changed_250Hz_velocities; "
                "unchanged_native_velocity_exceedances_preserved_identically")


def _bump(times, start, end):
    """Position and analytic velocity of a compact, endpoint-flat bump.

    Quantize the zero boundaries *inward* to actual trajectory knots. This
    keeps the Hermite interpolation outside the editable interval identical,
    including coarse 20 Hz knots adjacent to a protected release window.
    """
    times = np.asarray(times, dtype=float)
    value, derivative = np.zeros_like(times), np.zeros_like(times)
    inside = times[(times >= start - 1e-12) & (times <= end + 1e-12)]
    if inside.size < 3 or inside[-1] - inside[0] <= .03:
        return value, derivative
    start, end = float(inside[0]), float(inside[-1])
    u = (times - start) / (end - start)
    mask = (u > 1e-12) & (u < 1. - 1e-12)
    x = u[mask]
    value[mask] = 16. * x**2 * (1. - x)**2
    derivative[mask] = 32. * x * (1. - x) * (1. - 2. * x) / (end - start)
    return value, derivative


def correct_trajectory(traj, vel, times, action, edit_end, *, release_time=None,
                       transfer_start=None, scale=MAX_RESIDUAL_SCALE):
    """Return corrected (6, N) knots and velocities, or the original objects.

    Throw: the first three coefficients edit before release and the final
    three edit the return after release. A composed Push uses the first three
    before its stroke and the final three after ``transfer_start``. Standalone
    approach routing retains symmetric/skewed bumps before ``edit_end``.
    Only joints 1--3 are editable.
    """
    action = np.asarray(action, dtype=float)
    if action.shape != (ACTION_SIZE,) or not np.isfinite(action).all():
        raise ValueError("Expected six finite residual coefficients")
    if not np.isfinite(scale) or not 0. <= scale <= MAX_RESIDUAL_SCALE:
        raise ValueError("Residual scale must be between zero and 0.08 rad")
    if not np.any(action) or scale == 0.:
        return traj, vel
    arr, varr = np.asarray(traj, dtype=float), np.asarray(vel, dtype=float)
    times = np.asarray(times, dtype=float).ravel()
    if (arr.ndim != 2 or arr.shape[0] != 6 or varr.shape != arr.shape
            or arr.shape[1] != times.size or not np.all(np.diff(times) > 0.)):
        raise ValueError("Expected matching six-joint knots and increasing times")
    start, end = float(times[0]) + .02, min(float(edit_end), float(times[-1]) - .02)
    first, d_first = _bump(times, start, end)
    if release_time is not None and transfer_start is not None:
        raise ValueError("Use either a Throw release or a Push transfer boundary")
    post_start = (float(release_time) + .04 if release_time is not None
                  else transfer_start)
    if post_start is not None:
        second, d_second = _bump(times, float(post_start),
                                 float(times[-1]) - .02)
    else:
        # The skewed basis adds an independent early/late route choice.
        knots = times[(times >= start - 1e-12) & (times <= end + 1e-12)]
        if knots.size >= 3 and knots[-1] - knots[0] > .03:
            skew = 2. * (times - knots[0]) / (knots[-1] - knots[0]) - 1.
            second = first * skew
            d_second = d_first * skew + first * 2. / (knots[-1] - knots[0])
        else:
            second, d_second = np.zeros_like(times), np.zeros_like(times)
    coefficients = np.clip(action, -1., 1.).reshape(2, 3)
    delta = scale * (np.stack((first, second), axis=1) @ coefficients).T
    d_delta = scale * (np.stack((d_first, d_second), axis=1) @ coefficients).T
    maximum = float(np.max(np.abs(delta)))
    if maximum <= 0.:
        return traj, vel
    if maximum > scale:
        delta *= scale / maximum
        d_delta *= scale / maximum
    corrected, corrected_vel = arr.copy(), varr.copy()
    corrected[:3] += delta
    corrected_vel[:3] += d_delta
    return corrected, corrected_vel


def stream_samples(traj, vel, times, final_joint=None):
    """Reproduce RobotBackend's clamped-Hermite 250 Hz sample preparation."""
    from gp8_control.backends.robot_base import STREAM_DT

    arr, varr = np.asarray(traj, float), np.asarray(vel, float)
    times = np.asarray(times, float).ravel()
    if final_joint is not None:
        arr = np.column_stack((arr, np.asarray(final_joint, float)))
        varr = np.column_stack((varr, np.zeros(arr.shape[0])))
        times = np.append(times, times[-1] + .05)
    grid = np.arange(0., float(times[-1]) + STREAM_DT, STREAM_DT)
    if varr.shape == arr.shape and np.all(np.diff(times) > 0.):
        try:
            from scipy.interpolate import CubicHermiteSpline
            samples = np.column_stack([
                CubicHermiteSpline(times, arr[j], varr[j])(grid)
                for j in range(arr.shape[0])])
            seg = np.clip(np.searchsorted(times, grid, side="right") - 1, 0,
                          len(times) - 2)
            samples = np.clip(samples, np.minimum(arr[:, seg], arr[:, seg + 1]).T,
                              np.maximum(arr[:, seg], arr[:, seg + 1]).T)
        except Exception:
            samples = np.column_stack([np.interp(grid, times, row) for row in arr])
    else:
        samples = np.column_stack([np.interp(grid, times, row) for row in arr])
    return grid, samples


def _digest(array):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(array, float)).tobytes()).hexdigest()


class SafeResidualAdapter:
    """Bind a shared two-head PPO policy to one ``SimRlRunner`` instance.

    The callback contract matches the stashed PPO collector:
    ``callback(core, observation97, head, phase)`` returns action, latent,
    log probability, reward/cost values, normalized observation.  Head zero
    controls Throw; head one controls Push.  ``current_context`` is available
    to physical-step safety instrumentation throughout the blocking stream.
    """
    def __init__(self, policy=None, *, deterministic=False, decision_callback=None):
        self.policy = policy
        self.deterministic = bool(deterministic)
        self.decision_callback: Callable | None = decision_callback
        self.records: list[dict] = []
        self.current_context: dict | None = None
        self.runner = None
        self._original_stream = None

    def bind(self, runner):
        if self.runner is not None:
            raise RuntimeError("Adapter already bound")
        self.runner = runner
        self._original_stream = runner.traj_ctrl._stream_trajectory

        def stream(traj, vel, timestep, final_joint=None, *, release_index=None,
                   suction_on_at=None, tick_fn=None):
            corrected, corrected_vel, record = self.transform(
                traj, vel, timestep, final_joint, release_index=release_index)
            self.current_context = None if record is None else {
                key: record[key] for key in ("task", "phase", "target_id", "start_sim_time",
                    "release_time", "prepare_end", "stroke_end", "duration",
                    "prepare_goal_q6", "prepare_goal_xyz", "transfer_start")}
            if self.current_context is not None:
                # The monitor may attach measured boundary errors to this same
                # serializable record; the context itself is never stored in it.
                self.current_context["record"] = record
            runner.core.safe_residual_context = self.current_context
            try:
                return self._original_stream(corrected, corrected_vel, timestep,
                    final_joint, release_index=release_index,
                    suction_on_at=suction_on_at, tick_fn=tick_fn)
            finally:
                self.current_context = None
                runner.core.safe_residual_context = None

        runner.traj_ctrl._stream_trajectory = stream
        return self

    def unbind(self):
        if self.runner is not None:
            self.runner.traj_ctrl._stream_trajectory = self._original_stream
            self.runner.core.safe_residual_context = None
        self.runner, self._original_stream = None, None

    def _objects(self, target_id):
        core = self.runner.core
        objects = []
        with core.lock:
            for index, state in enumerate(core._box_state):
                if state == core._FREE:
                    continue
                joint = core._box_joints[index]
                objects.append(dict(id=int(core._box_serial[index]),
                    position=np.asarray(core._to_base(joint.qpos[:3]), float).copy(),
                    size=np.asarray(core._box_half_size[index], float).copy(),
                    velocity=(np.asarray(core._base_R, float).T @
                              np.asarray(joint.qvel[:3], float)).copy()))
        target = next((item for item in objects if item["id"] == target_id), None)
        if target is None:
            return None, [], objects
        neighbors = sorted((item for item in objects if item is not target), key=lambda item:
            (float(np.sum((item["position"] - target["position"])**2)), item["id"]))[:5]
        return target, neighbors, objects

    def observation(self, traj, times, edit_end, release_time, phase, target_id):
        runner = self.runner
        q = np.asarray(runner.traj_ctrl.current_joints, float)
        qdot = np.asarray(runner.traj_ctrl.current_jointvels, float)
        target, neighbors, objects = self._objects(target_id)
        if target is None:
            return None, []
        nominal = np.asarray(traj, float)
        edit_goal = np.array([np.interp(edit_end, times, row) for row in nominal])
        final_goal = nominal[:, -1]
        next_action = runner._chain_action
        next_target = next_action.target if next_action is not None else None
        next_id = getattr(next_target, "sim_object_id", None)
        next_object = next((item for item in objects if item["id"] == next_id), None)
        next_primitive = (1 if next_action is not None and next_action.skill_name == "throw"
                          else 2 if next_action is not None and next_action.skill_name == "push" else 0)
        relative = (next_object["position"] - target["position"]
                    if next_object is not None else np.zeros(3))
        from gp8_control.skills.throw_skill import THROW_BIN_TARGET_MAP
        from gp8_control.skills.push_skill import PUSH_BIN_TARGET_MAP
        active_target = runner._active_target
        destinations = THROW_BIN_TARGET_MAP if phase == "throw" else PUSH_BIN_TARGET_MAP
        destination = destinations.get(getattr(active_target, "class_name", ""), (.85, 0., 0.))
        neighborhood = []
        for item in neighbors:
            neighborhood.extend([*(item["position"] - target["position"]),
                                 *item["size"], *item["velocity"]])
        neighborhood.extend([0.] * (9 * (5 - len(neighbors))))
        values = np.concatenate((q / np.pi, qdot / 10., target["position"], target["size"],
            neighborhood, (edit_goal - q) / np.pi, (final_goal - q) / np.pi,
            [float(times[-1]) / 4., float(edit_end) / 4., (release_time or 0.) / 4.],
            np.eye(3)[PHASES.index(phase)], np.eye(3)[next_primitive], relative,
            [runner.core.cfg.belt_speed, destination[0], destination[1],
             len(neighbors) / 5., len(objects) / 12.],
            [float(index < len(neighbors)) for index in range(5)]))
        observation = np.asarray(values, np.float32)
        if observation.shape != (OBSERVATION_SIZE,) or not np.isfinite(observation).all():
            raise ValueError(f"Invalid app-simulator observation: {observation.shape}")
        return observation, [item["id"] for item in neighbors]

    def _safe(self, traj, vel, times, final_joint, nominal_samples=None):
        runner = self.runner
        _, samples = stream_samples(traj, vel, times, final_joint)
        from gp8_control.backends.robot_base import STREAM_DT
        bounds = np.asarray(runner.robot.joint_limits, float)
        velocities = np.diff(samples, axis=0) / STREAM_DT
        ceiling = np.asarray(runner.M1, float)
        # The original stream can exceed its design M1 slightly after
        # clamped-Hermite resampling, even in joints/segments the actor cannot
        # edit. Preserve those native commands, while requiring every changed
        # sample velocity to satisfy M1 and never worsening any exceedance.
        nominal_velocities = (np.diff(nominal_samples, axis=0) / STREAM_DT
                              if nominal_samples is not None else np.zeros_like(velocities))
        changed = np.abs(velocities - nominal_velocities) > 1e-10
        hard_ceiling = np.broadcast_to(ceiling, velocities.shape)
        return bool(np.isfinite(samples).all() and np.isfinite(velocities).all()
            and np.all(samples >= bounds[None, :, 0] - 1e-9)
            and np.all(samples <= bounds[None, :, 1] + 1e-9)
            and np.all(np.abs(velocities) <= np.maximum(hard_ceiling,
                       np.abs(nominal_velocities)) + 1e-9)
            and np.all(np.abs(velocities[changed]) <= hard_ceiling[changed] + 1e-9))

    def transform(self, traj, vel, timestep, final_joint=None, *, release_index=None):
        runner = self.runner
        target = runner._active_target
        target_id = getattr(target, "sim_object_id", None)
        if target_id is None:
            return traj, vel, None
        times = np.asarray(timestep, float).ravel()
        if times.size < 3 or not np.all(np.diff(times) > 0.):
            return traj, vel, None
        task = runner._skill_name_for_object(target)
        release_time, prepare_end, stroke_end, transfer_start = None, None, None, None
        if release_index is not None:
            task, phase = "throw", "throw"
            # NumPy indexing preserves both the backend's positive and negative index semantics.
            release_time = float(times[int(release_index)])
            edit_end = release_time - .04
            subphase = "throw_with_return"
        elif task == "push" and runner._status == "PUSHING":
            phase, subphase = "push_prepare", "prepare_and_transfer_with_protected_stroke"
            meta = runner.skills["push"]._last_push_meta
            n_descent, n_stroke = int(meta.get("n_descent", 0)), int(meta.get("n_stroke", 0))
            if n_descent < 0 or n_stroke <= 0 or n_descent + n_stroke > times.size:
                return traj, vel, None
            prepare_end = float(times[n_descent - 1] if n_descent else times[0])
            stroke_end = float(times[n_descent + n_stroke - 1])
            transfer_start = stroke_end + .04
            edit_end = prepare_end - .02
        elif task == "push" and runner._status == "POSITIONING":
            phase = "push_route"
            subphase, edit_end = "positioning", float(times[-1])
        else:
            # In particular, never edit Throw's tracking descent / suction sealing.
            return traj, vel, None
        first, _ = _bump(times, float(times[0]) + .02,
                         min(edit_end, float(times[-1]) - .02))
        post_start = release_time + .04 if release_time is not None else transfer_start
        second, _ = (_bump(times, post_start, float(times[-1]) - .02)
                     if post_start is not None else (np.zeros_like(times), None))
        if transfer_start is not None and not np.any(first) and np.any(second):
            phase = "push_route"
        active = bool(np.any(first) or np.any(second))
        prepare_goal_q6, prepare_goal_xyz = None, None
        if prepare_end is not None:
            prepare_goal_q6 = [float(np.interp(prepare_end, times, row)) for row in np.asarray(traj)]
            prepare_goal_xyz = np.asarray(runner.robot.forward_kinematics(
                np.asarray(prepare_goal_q6))[:3, 3], float).tolist()
        boundary_record = dict(task=task, phase=phase, subphase=subphase,
            head=0 if task == "throw" else 1, target_id=int(target_id),
            start_sim_time=float(runner.core.sim_time()), prepare_end=prepare_end,
            stroke_end=stroke_end, release_time=release_time, duration=float(times[-1]),
            transfer_start=transfer_start,
            prepare_goal_q6=prepare_goal_q6, prepare_goal_xyz=prepare_goal_xyz,
            active=False, inactive_reason=None)
        observation, neighbor_ids = self.observation(traj, times, edit_end,
            release_time, phase, int(target_id))
        if observation is None:
            boundary_record["inactive_reason"] = "target_resolved_before_control"
            return traj, vel, boundary_record
        if not active:
            boundary_record["inactive_reason"] = "insufficient_residual_window"
            return traj, vel, boundary_record
        head = 0 if task == "throw" else 1
        if self.decision_callback is not None:
            result = self.decision_callback(runner.core, observation, head, phase)
        elif self.policy is not None:
            result = self.policy.act(observation, head, deterministic=self.deterministic)
        else:
            result = np.zeros(6), np.zeros(6), 0., np.zeros(2), observation.copy()
        action, latent, logp, value, normalized = result
        action, latent = np.asarray(action, float), np.asarray(latent, float)
        value, normalized = np.asarray(value, float), np.asarray(normalized, float)
        if (action.shape != (6,) or latent.shape != (6,) or value.shape != (2,)
                or normalized.shape not in ((OBSERVATION_SIZE,), (OBSERVATION_SIZE + 3,))
                or not all(np.isfinite(item).all() for item in (action, latent, value, normalized))
                or not np.isfinite(logp)):
            raise ValueError("Actor returned invalid residual/training metadata")
        proposed, proposed_vel = correct_trajectory(traj, vel, times, action,
            edit_end, release_time=release_time, transfer_start=transfer_start)
        delta = np.asarray(proposed, float) - np.asarray(traj, float)
        d_delta = np.asarray(proposed_vel, float) - np.asarray(vel, float)
        maximum = float(np.max(np.abs(delta)))
        _, nominal_samples = stream_samples(traj, vel, times, final_joint)
        executed, executed_vel, projection = traj, vel, 1.
        if maximum > 0.:
            projection = 0.
            for exponent in range(11):
                alpha = .5**exponent
                candidate = np.asarray(traj, float) + alpha * delta
                candidate_vel = np.asarray(vel, float) + alpha * d_delta
                if self._safe(candidate, candidate_vel, times, final_joint, nominal_samples):
                    executed, executed_vel, projection = candidate, candidate_vel, alpha
                    break
        protected = ((times <= times[0] + .02 + 1e-12)
                     | (times >= times[-1] - .02 - 1e-12))
        if release_time is not None:
            protected |= np.abs(times - release_time) <= .04 + 1e-12
        if prepare_end is not None:
            protected |= ((times >= prepare_end - .02 - 1e-12)
                          & (times <= stroke_end + .04 + 1e-12))
        protected_position_equal = np.array_equal(np.asarray(executed)[:, protected],
                                                   np.asarray(traj)[:, protected])
        protected_velocity_equal = np.array_equal(np.asarray(executed_vel)[:, protected],
                                                   np.asarray(vel)[:, protected])
        if not protected_position_equal or not protected_velocity_equal:
            raise RuntimeError("Residual changed a protected boundary")
        from gp8_control.backends.robot_base import STREAM_DT
        nominal_sample_velocities = np.diff(nominal_samples, axis=0) / STREAM_DT
        if executed is traj and executed_vel is vel:
            sample_velocities = nominal_sample_velocities
        else:
            _, executed_samples = stream_samples(executed, executed_vel, times, final_joint)
            sample_velocities = np.diff(executed_samples, axis=0) / STREAM_DT
        changed = np.abs(sample_velocities - nominal_sample_velocities) > 1e-10
        ceiling = np.broadcast_to(np.asarray(runner.M1, float), sample_velocities.shape)
        record = dict(task=task, phase=phase, subphase=subphase, head=head,
            target_id=int(target_id), neighbor_ids=neighbor_ids,
            start_sim_time=float(runner.core.sim_time()),
            observation=observation.tolist(), action=action.tolist(), latent=latent.tolist(),
            logp=float(logp), value=value.tolist(), normalized=normalized.tolist(),
            active=True, inactive_reason=None, rejected=bool(projection == 0.),
            projection=float(projection), projection_factor=float(projection),
            projected=bool(0. < projection < 1.), residual_scale_rad=MAX_RESIDUAL_SCALE,
            proposed_max_joint_delta=maximum,
            max_joint_delta=float(np.max(np.abs(np.asarray(executed) - np.asarray(traj)))),
            edit_end=float(edit_end), prepare_end=prepare_end, stroke_end=stroke_end,
            transfer_start=transfer_start,
            prepare_window_active=bool(np.any(first)),
            transfer_window_active=bool(transfer_start is not None and np.any(second)),
            prepare_goal_q6=prepare_goal_q6, prepare_goal_xyz=prepare_goal_xyz,
            release_time=release_time, release_index=release_index,
            duration=float(times[-1]), nominal_position_sha256=_digest(traj),
            nominal_velocity_sha256=_digest(vel), executed_position_sha256=_digest(executed),
            executed_velocity_sha256=_digest(executed_vel),
            exact_identity=bool(executed is traj and executed_vel is vel),
            protected_position_equal=bool(protected_position_equal),
            protected_velocity_equal=bool(protected_velocity_equal),
            protected_knot_count=int(np.count_nonzero(protected)),
            protected_stroke_knots=(int(n_stroke) if prepare_end is not None else 0),
            guard_policy=GUARD_POLICY,
            baseline_existing_velocity_ratio=float(np.max(np.abs(nominal_sample_velocities) / ceiling)),
            candidate_changed_max_ratio=(float(np.max(np.abs(sample_velocities[changed]) /
                ceiling[changed])) if np.any(changed) else 0.),
            candidate_changed_velocity_samples=int(np.count_nonzero(changed)))
        self.records.append(record)
        return executed, executed_vel, record

"""Phase-time residual control over the unchanged native 250 Hz stream.

Policy ownership can move to a declared next Push while the physical operation
is still the preceding Throw/Push. Decisions are sampled at the actual native
sample boundary, never prospectively at the beginning of the previous skill.
"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np

from gp8_control.backends.robot_base import STREAM_DT
from gp8_control.rl.safe_residual_adapter import SafeResidualAdapter as SceneAdapter
from gp8_control.rl.safe_residual_adapter import stream_samples


OBSERVATION_SIZE = 97
ACTION_SIZE = 6
NORMALIZED_SIZE = 107
MAX_RESIDUAL_SCALE = .25
ROLES = ('throw_loaded', 'throw_return', 'push_approach', 'push_return')
GUARD_POLICY = ('strict_joint_bounds; changed250Hz_velocity_strict_M1; '
                'native_velocity_and_acceleration_exceedances_not_worsened; '
                'independent_joint_pair_projection_then_common_static_projection')


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(array, float)).tobytes()).hexdigest()


def route_delta(grid, action, start, end, scale=MAX_RESIDUAL_SCALE):
    """Two independent C2 endpoint-flat bases for each of joints 1--3."""
    grid, action = np.asarray(grid, float), np.asarray(action, float)
    if action.shape != (6,) or not np.isfinite(action).all():
        raise ValueError('Expected six finite route coefficients')
    if not np.isfinite(scale) or not 0 <= scale <= MAX_RESIDUAL_SCALE:
        raise ValueError('Route amplitude must be between zero and 0.25 radians')
    delta = np.zeros((len(grid), 6), float)
    inside = grid[(grid >= start - 1e-12) & (grid <= end + 1e-12)]
    if len(inside) < 5 or inside[-1] - inside[0] < .032 - 1e-12:
        return delta
    start, end = float(inside[0]), float(inside[-1])
    u = (grid - start) / (end - start)
    mask = (u > 1e-12) & (u < 1 - 1e-12)
    # Cubic powers make both velocity and acceleration vanish at the seam.
    bump = np.zeros_like(grid)
    bump[mask] = 64 * u[mask]**3 * (1 - u[mask])**3
    bases = np.column_stack((bump, bump * (2 * u - 1)))
    delta[:, :3] = float(scale) * bases @ np.clip(action, -1, 1).reshape(2, 3)
    maximum = float(np.max(np.abs(delta)))
    if maximum > scale and maximum > 0:
        delta *= scale / maximum
    return delta


def dynamic_guard(runner, candidate, nominal):
    """Check the actual outgoing grid, including finite-difference seams."""
    candidate, nominal = np.asarray(candidate, float), np.asarray(nominal, float)
    limits = np.asarray(runner.robot.joint_limits, float)
    velocity = np.diff(candidate, axis=0) / STREAM_DT
    native_velocity = np.diff(nominal, axis=0) / STREAM_DT
    ceiling = np.broadcast_to(np.asarray(runner.M1, float), velocity.shape)
    changed = np.abs(velocity - native_velocity) > 1e-10
    acceleration = np.diff(velocity, axis=0) / STREAM_DT
    native_acceleration = np.diff(native_velocity, axis=0) / STREAM_DT
    accel_limits = np.broadcast_to(np.asarray(runner.M2, float), acceleration.shape)
    checks = dict(finite=bool(np.isfinite(candidate).all()),
        joint_bounds=bool(np.all(candidate >= limits[None, :, 0] - 1e-9)
                          and np.all(candidate <= limits[None, :, 1] + 1e-9)),
        native_velocity_nonworsening=bool(np.all(np.abs(velocity) <= np.maximum(ceiling, np.abs(native_velocity)) + 1e-8)),
        changed_velocity_M1=bool(np.all(np.abs(velocity[changed]) <= ceiling[changed] + 1e-8)),
        acceleration_nonworsening=bool(np.all(np.abs(acceleration) <= np.maximum(accel_limits, np.abs(native_acceleration)) + 1e-7)))
    accepted = all(checks.values())
    return dict(accepted=accepted, reason='accepted' if accepted else next(key for key,passed in checks.items() if not passed),
        candidate_changed_max_ratio=(float(np.max(np.abs(velocity[changed]) / ceiling[changed]))
                                     if np.any(changed) else 0.),
        baseline_existing_velocity_ratio=float(np.max(np.abs(native_velocity) / ceiling)),
        acceleration_limit_ratio=float(np.max(np.abs(acceleration) /
            np.maximum(accel_limits, np.abs(native_acceleration)))) if acceleration.size else 0.,
        candidate_changed_max_ratio_per_joint=[float(np.max(np.abs(velocity[:,j][changed[:,j]]) /
            ceiling[:,j][changed[:,j]])) if np.any(changed[:,j]) else 0. for j in range(6)],
        acceleration_limit_ratio_per_joint=(np.max(np.abs(acceleration) /
            np.maximum(accel_limits, np.abs(native_acceleration)), axis=0).tolist()
            if acceleration.size else [0.]*6),
        changed_velocity_samples=int(np.count_nonzero(changed)))


class _Scene(SceneAdapter):
    """Reuse the audited 97D schema, with a finite resolved-owner fallback."""
    def __init__(self, adapter):
        super().__init__()
        self.adapter = adapter

    def _objects(self, target_id):
        target, neighbors, objects = super()._objects(target_id)
        if target is not None:
            return target, neighbors, objects
        # A released/processed physical owner can disappear before its return.
        # The return still has controllable commands and surrounding obstacles.
        position = np.asarray(self.runner.ee_xyz(), float)
        target = dict(id=target_id, position=position, size=np.zeros(3), velocity=np.zeros(3))
        neighbors = sorted(objects, key=lambda obj:
            (float(np.sum((obj['position'] - position)**2)), obj['id']))[:5]
        return target, neighbors, objects


class SafeV2Adapter:
    def __init__(self, policy=None, *, deterministic=False, decision_callback=None,
                 scale=MAX_RESIDUAL_SCALE, static_guard=None):
        if not np.isfinite(scale) or not 0 <= scale <= MAX_RESIDUAL_SCALE:
            raise ValueError('Invalid residual amplitude')
        self.policy, self.deterministic = policy, bool(deterministic)
        self.decision_callback, self.scale, self.static_guard = decision_callback, float(scale), static_guard
        self.records = []
        self.current_owner = None
        self.window_duration = 0.
        self.current_context = None
        self.runner = None
        self._frame = None
        self._accepted_next = None
        self._throw_boundary = None
        self._last_decision_time = None
        self._scene = _Scene(self)

    @staticmethod
    def _id(target):
        value = getattr(target, 'sim_object_id', None)
        return None if value is None else int(value)

    def bind(self, runner):
        if self.runner is not None:
            raise RuntimeError('Adapter is already bound')
        self.runner = runner
        self._original_stream = runner.traj_ctrl._stream_trajectory
        self._original_emit = runner.traj_ctrl._emit_sample
        self._original_next = runner.ctx.next_chain_target
        self._visualizer = runner.skills['throw']._visualizer
        self._original_publish = self._visualizer.publish

        def next_target(*args, **kwargs):
            result = self._original_next(*args, **kwargs)
            source = self._id(runner._active_target)
            self._accepted_next = None
            if result is not None:
                _, target = result
                self._accepted_next = dict(source=source, target=target,
                    id=self._id(target), task=runner._skill_name_for_object(target))
            return result

        def publish(traj, vel, times, *args, **kwargs):
            self._throw_boundary = dict(source=self._id(runner._active_target),
                count=np.asarray(traj).shape[1], times_sha256=digest(times),
                last_index=int(kwargs['throw_last_index']))
            return self._original_publish(traj, vel, times, *args, **kwargs)

        def stream(traj, vel, times, final_joint=None, *, release_index=None,
                   suction_on_at=None, tick_fn=None):
            if self._frame is not None:
                raise RuntimeError('Nested native trajectory streams are unsupported')
            frame = self._make_frame(traj, vel, times, final_joint, release_index)
            self._frame = frame
            self.current_context = None if frame is None else frame['context']
            runner.core.safe_residual_context = self.current_context
            try:
                return self._original_stream(traj, vel, times, final_joint,
                    release_index=release_index, suction_on_at=suction_on_at, tick_fn=tick_fn)
            finally:
                if frame is not None:
                    self._measure_endpoints(frame, float(runner.core.sim_time()))
                self._frame = None
                self.current_context = None
                self.current_owner = None
                self.window_duration = 0.
                runner.core.safe_residual_context = None
                runner.core.safe_v2_actor_context = None

        def emit(positions):
            frame = self._frame
            if frame is None:
                return self._original_emit(positions)
            k = frame['cursor']
            if k >= len(frame['grid']) or not np.array_equal(np.asarray(positions), frame['nominal'][k]):
                raise RuntimeError('Native stream differs from the audited 250 Hz grid')
            self._measure_endpoints(frame, float(runner.core.sim_time()))
            for window in frame['windows']:
                if window['index'] == k:
                    self._decide(frame, window)
            frame['cursor'] += 1
            outgoing = frame['executed'][k]
            # The identity branch forwards the original object, not a re-sample.
            if np.array_equal(outgoing, np.asarray(positions)):
                return self._original_emit(positions)
            return self._original_emit(outgoing.copy())

        runner.ctx.next_chain_target = next_target
        self._visualizer.publish = publish
        runner.traj_ctrl._stream_trajectory = stream
        runner.traj_ctrl._emit_sample = emit
        return self

    def unbind(self):
        if self.runner is None:
            return
        runner = self.runner
        runner.ctx.next_chain_target = self._original_next
        self._visualizer.publish = self._original_publish
        runner.traj_ctrl._stream_trajectory = self._original_stream
        runner.traj_ctrl._emit_sample = self._original_emit
        runner.core.safe_residual_context = None
        runner.core.safe_v2_actor_context = None
        self.runner = None

    def _make_frame(self, traj, vel, times, final_joint, release_index):
        runner = self.runner
        target = runner._active_target
        physical_id = self._id(target)
        if physical_id is None:
            return None
        times = np.asarray(times, float).ravel()
        if len(times) < 2:
            return None
        grid, nominal = stream_samples(traj, vel, times, final_joint)
        physical_task = runner._skill_name_for_object(target)
        begin, end = float(grid[0]), float(times[-1])
        phase_end = float(grid[-1])
        release, prepare_end, stroke_end = None, None, None
        accepted = self._accepted_next
        if accepted is None or accepted['source'] != physical_id:
            accepted = None
        windows = []

        def add(role, owner, start, finish, edit_start=None, edit_end=None, source='current'):
            if finish <= start:
                return
            index = int(np.searchsorted(grid, start - 1e-12))
            if index >= len(grid):
                return
            actual_start = float(grid[index])
            a = actual_start + .02 if edit_start is None else max(float(edit_start), actual_start + .02)
            b = float(finish) - .02 if edit_end is None else min(float(edit_end), float(finish) - .02)
            if not np.any(route_delta(grid, np.ones(6), a, b, scale=1e-3)):
                return
            owner_task = 'throw' if role.startswith('throw') else 'push'
            owner_id = self._id(owner)
            windows.append(dict(role=role, head=0 if owner_task == 'throw' else 1,
                owner_id=owner_id, owner_task=owner_task, target=owner,
                source=source, index=index, trigger_time=actual_start,
                phase_end=float(finish), edit_start=a, edit_end=b,
                physical_task=physical_task, physical_target_id=physical_id))

        if release_index is not None:
            physical_task = 'throw'
            release = float(times[int(release_index)])
            add('throw_loaded', target, begin, release - .04)
            boundary = self._throw_boundary
            if (boundary is not None and boundary['source'] == physical_id
                    and boundary['count'] == np.asarray(traj).shape[1]
                    and boundary['times_sha256'] == digest(times)):
                arc_end = float(times[boundary['last_index']])
            else:
                # Boundary ownership must never rely on a guessed velocity seam.
                arc_end, accepted = end, None
            if accepted is not None:
                add('throw_return', target, release + .04, arc_end)
                add('push_approach' if accepted['task'] == 'push' else 'throw_return',
                    accepted['target'], arc_end, phase_end, edit_end=end - .02,
                    source='accepted_next')
            else:
                add('throw_return', target, release + .04, phase_end, edit_end=end - .02)
        elif physical_task == 'push' and runner._status == 'PUSHING':
            meta = runner.skills['push']._last_push_meta
            nd, ns = int(meta.get('n_descent', 0)), int(meta.get('n_stroke', 0))
            if nd < 0 or ns <= 0 or nd + ns > len(times):
                raise RuntimeError('Missing native Push stroke boundaries')
            prepare_end = float(times[nd - 1] if nd else times[0])
            stroke_end = float(times[nd + ns - 1])
            add('push_approach', target, begin, prepare_end)
            owner = accepted['target'] if accepted is not None else target
            role = ('push_approach' if accepted is not None and accepted['task'] == 'push'
                    else 'throw_return' if accepted is not None else 'push_return')
            add(role, owner, stroke_end + .04, phase_end, edit_end=end - .02,
                source='accepted_next' if accepted is not None else 'current')
        elif physical_task == 'push' and runner._status == 'POSITIONING':
            add('push_approach', target, begin, phase_end, edit_end=end - .02)
        else:
            return None
        windows.sort(key=lambda w: w['index'])
        if len({w['index'] for w in windows}) != len(windows):
            raise RuntimeError('Two policy decisions occupy the same native sample')
        prepared_goal = (self._sample(nominal, grid, prepare_end) if prepare_end is not None else None)
        boundary_record = dict(active=False, task=physical_task, target_id=physical_id)
        context = dict(task=physical_task, phase='throw' if physical_task == 'throw' else 'push_prepare',
            target_id=physical_id, start_sim_time=float(runner.core.sim_time()),
            release_time=release, prepare_end=prepare_end, stroke_end=stroke_end,
            duration=end, transfer_start=None if stroke_end is None else stroke_end + .04,
            prepare_goal_q6=None if prepared_goal is None else prepared_goal.tolist(),
            prepare_goal_xyz=None if prepared_goal is None else self._fk(prepared_goal), record=boundary_record)
        return dict(grid=grid, nominal=nominal, executed=nominal.copy(), windows=windows,
            context=context, cursor=0, endpoint_records=[], started=float(runner.core.sim_time()),
            physical_task=physical_task, physical_target_id=physical_id)

    @staticmethod
    def _sample(samples, grid, time):
        return np.asarray([np.interp(time, grid, row) for row in np.asarray(samples).T], float)

    def _fk(self, q):
        return np.asarray(self.runner.robot.forward_kinematics(np.asarray(q, float))[:3, 3], float).tolist()

    def _measure_endpoints(self, frame, now):
        for record in frame['endpoint_records']:
            if record.get('endpoint_reached') or now < record['phase_end_time'] - 1e-9:
                continue
            if 'endpoint_position_error_m' in record and 'endpoint_q_error_rad' in record:
                # A physical-step core can measure directly from MjData before
                # the next control tick; retain its more precise measurement.
                record['endpoint_reached'] = True
                continue
            q = np.asarray(self.runner.traj_ctrl.current_joints, float)
            goal = np.asarray(record['goal_q6'], float)
            record.update(endpoint_reached=True, endpoint_measured_time=now,
                endpoint_q_error_rad=float(np.max(np.abs(q - goal))),
                endpoint_position_error_m=float(np.linalg.norm(
                    np.asarray(self._fk(q)) - np.asarray(record['goal_xyz']))))

    def observation(self, frame, window):
        runner = self.runner
        target = window['target']
        if target is None:
            target = SimpleNamespace(class_name='transparent' if window['head'] == 0 else 'metal')
        # Read-only proxy changes only which object's scene is encoded, never
        # the executor/monitor's active physical target.
        class Proxy:
            def __getattr__(self, key):
                return getattr(runner, key)
        proxy = Proxy()
        proxy._active_target = target
        self._scene.runner = proxy
        old_phase = ('throw' if window['head'] == 0 else
                     'push_prepare' if window['source'] == 'current' else 'push_route')
        return self._scene.observation(frame['nominal'].T, frame['grid'],
            window['phase_end'], frame['context']['release_time'], old_phase, window['owner_id'])

    def _decide(self, frame, window):
        runner = self.runner
        now = float(runner.core.sim_time())
        if self._last_decision_time is not None and now <= self._last_decision_time + 1e-10:
            raise RuntimeError('Policy decision has no positive elapsed physical time')
        self.window_duration = float(window['phase_end'] - window['trigger_time'])
        goal = self._sample(frame['nominal'], frame['grid'], window['phase_end'])
        phase_end_time = frame['started'] + window['phase_end']
        self.current_owner = dict(id=window['owner_id'], task=window['owner_task'], phase=window['role'],
            target=window['target'], source=window['source'], goal_q6=goal.tolist(), goal_xyz=self._fk(goal),
            phase_end_time=phase_end_time, actual_boundary_time=now,
            physical_task=frame['physical_task'], physical_target_id=frame['physical_target_id'])
        observation, neighbor_ids = self.observation(frame, window)
        if self.decision_callback is not None:
            result = self.decision_callback(runner.core, observation, window['head'], window['role'])
        elif self.policy is not None:
            result = self.policy.act(observation, window['head'], deterministic=self.deterministic)
        else:
            result = np.zeros(6), np.zeros(6), 0., np.zeros(2), np.zeros(NORMALIZED_SIZE)
        action, latent, logp, value, normalized = result
        action, latent, value, normalized = (np.asarray(x, float) for x in (action, latent, value, normalized))
        if (action.shape != (6,) or latent.shape != (6,) or value.shape != (2,)
                or normalized.shape != (NORMALIZED_SIZE,) or not np.isfinite(logp)
                or not all(np.isfinite(x).all() for x in (action, latent, value, normalized))):
            raise ValueError('Invalid v2 policy output or 107D normalized observation')
        delta = route_delta(frame['grid'], action, window['edit_start'], window['edit_end'], self.scale)
        proposed_max = float(np.max(np.abs(delta)))
        alpha, selected_guard, static_metrics = 1., None, {}
        joint_factors = np.ones(3)
        guard_rejections = []
        executed = frame['executed']
        if proposed_max > 0:
            # Joint constraints are independent. A time-optimal saturated
            # joint must not freeze the other joints' feasible route changes.
            for joint in range(3):
                if not np.any(delta[:,joint]):
                    continue
                joint_factors[joint] = 0.
                isolated = np.zeros_like(delta)
                isolated[:,joint] = delta[:,joint]
                for exponent in range(13):
                    fraction = .5**exponent
                    guard = dynamic_guard(runner, frame['executed'] + fraction * isolated, frame['nominal'])
                    if guard['accepted']:
                        joint_factors[joint] = fraction
                        break
                    guard_rejections.append(dict(joint=joint+1, alpha=fraction, reason=guard['reason']))
            joint_delta = delta.copy()
            joint_delta[:,:3] *= joint_factors[None,:]
            alpha = 0.
            for exponent in (range(13) if np.any(joint_delta) else ()):
                fraction = .5**exponent
                candidate = frame['executed'] + fraction * joint_delta
                guard = dynamic_guard(runner, candidate, frame['nominal'])
                if not guard['accepted']:
                    guard_rejections.append(dict(alpha=fraction, reason=guard['reason']))
                    continue
                if self.static_guard is not None:
                    protected = np.max(np.abs(delta), axis=1) == 0
                    details = dict(window, phase=window['role'], owner_task=window['owner_task'],
                        target_id=window['owner_id'], protected_mask=protected,
                        release_time=frame['context']['release_time'],
                        prepare_end=frame['context']['prepare_end'], stroke_end=frame['context']['stroke_end'])
                    static = self.static_guard(runner, candidate, frame['nominal'], frame['grid'], details)
                    if isinstance(static, bool):
                        static = dict(accepted=static)
                    if not static['accepted']:
                        guard_rejections.append(dict(alpha=fraction, reason=static.get('reason','static_guard')))
                        continue
                    static_metrics = dict(static)
                alpha, executed, selected_guard = fraction, candidate, guard
                break
        if selected_guard is None:
            selected_guard = dynamic_guard(runner, executed, frame['nominal'])
        protected = np.max(np.abs(delta), axis=1) == 0
        if not np.array_equal(executed[protected], frame['executed'][protected]):
            raise RuntimeError('Residual modified an immutable physical sample')
        actual_delta = executed - frame['executed']
        projection_ratio = (float(np.clip(np.linalg.norm(actual_delta) / np.linalg.norm(delta),0.,1.))
                            if proposed_max > 0 else 1.)
        record = dict(task=window['owner_task'], phase=window['role'], role=window['role'],
            head=window['head'], target_id=window['owner_id'], owner_id=window['owner_id'],
            owner_source=window['source'], physical_task=frame['physical_task'],
            physical_target_id=frame['physical_target_id'], neighbor_ids=neighbor_ids,
            start_sim_time=now, actual_boundary_time=now, native_boundary_offset=window['trigger_time'],
            phase_end_time=phase_end_time, phase_end_offset=window['phase_end'],
            duration=self.window_duration, window_duration=self.window_duration,
            edit_start=window['edit_start'], edit_end=window['edit_end'],
            goal_q6=goal.tolist(), goal_xyz=self._fk(goal),
            owner_goal_q6=goal.tolist(), owner_goal_xyz=self._fk(goal),
            observation=observation.tolist(), action=action.tolist(), latent=latent.tolist(),
            logp=float(logp), value=value.tolist(), normalized=normalized.tolist(),
            active=True, rejected=bool(proposed_max > 0 and not np.any(actual_delta)),
            projected=bool(0 < projection_ratio < 1 - 1e-12),
            projection=projection_ratio, projection_factor=projection_ratio,
            projection_definition='L2 norm(executed_delta) / L2 norm(proposed_delta)',
            joint_projection=joint_factors.tolist(), static_projection=float(alpha),
            effective_joint_projection=(alpha * joint_factors).tolist(), residual_scale_rad=self.scale,
            proposed_max_joint_delta=proposed_max, max_joint_delta=float(np.max(np.abs(actual_delta))),
            nominal_position_sha256=digest(frame['nominal']), executed_position_sha256=digest(executed),
            exact_identity=bool(not np.any(actual_delta)),
            stream_exact_identity=bool(np.array_equal(executed, frame['nominal'])),
            protected_position_equal=True, protected_velocity_equal=True,
            prepare_window_active=bool(window['role'] == 'push_approach' and window['source'] == 'current'),
            transfer_window_active=bool(window['source'] == 'accepted_next' or window['role'] in ('throw_return', 'push_return')),
            prepare_end=frame['context']['prepare_end'], stroke_end=frame['context']['stroke_end'],
            release_time=frame['context']['release_time'], guard_policy=GUARD_POLICY,
            endpoint_reached=False,
            static_guard=static_metrics, **{k:v for k,v in selected_guard.items() if k != 'accepted'})
        record['guard_rejections'] = guard_rejections
        frame['executed'] = executed
        self.records.append(record)
        frame['endpoint_records'].append(record)
        self.current_owner['record'] = record
        runner.core.safe_v2_actor_context = self.current_owner
        if (window['role'] == 'push_approach' and window['source'] == 'current'
                and window['owner_id'] == frame['physical_target_id']):
            record['prepare_goal_q6'] = frame['context']['prepare_goal_q6']
            record['prepare_goal_xyz'] = frame['context']['prepare_goal_xyz']
            frame['context']['record'] = record
        self._last_decision_time = now


SafeResidualAdapter = SafeV2Adapter

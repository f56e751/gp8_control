"""Finite offered workloads for residual Safe PPO on the native GP8 twin.

The selector, skills, controller, camera and physics are shared by both arms.
Only the trajectory adapter's six coefficients differ. All arrivals are fixed
before execution; unfinished outcomes are counted at the observed task deadline.
"""
from __future__ import annotations

from contextlib import redirect_stdout
from collections import Counter
import hashlib
import io
import json
import time

import mujoco
import numpy as np

from gp8_control.backends.mujoco_sim import SimCore, MJ_JOINTS
from gp8_control.config import Config
from gp8_control.rl.sim_runner import SimRlRunner
from gp8_control.rl.safe_residual_adapter import SafeResidualAdapter
from gp8_control.rl.safe_contact_monitor import SafeContactMonitor


OBSERVATION_SIZE = 100


def default_config():
    return dict(offer_count=20, rate_range=[.6, .9], belt_speed=.12,
                pool_size=24, time_limit=120., physical_belt=True,
                belt_contact_params='old', suction_probability=1.,
                truth_tracks=True, max_objects=6, residual_scale=.08,
                termination='fixed_offers_then_all_physical_objects_resolved',
                success_target_auxiliary=10, quiet=True)


def make_tape(scene_seed, config):
    rng = np.random.default_rng(int(scene_seed))
    rate = float(rng.uniform(*config['rate_range']))
    offered_time = .5
    rows = []
    # A bounded inter-arrival jitter avoids changing moderate traffic into
    # accidental Poisson bursts. Positions/classes/sizes/yaws remain randomized.
    for index in range(int(config['offer_count'])):
        if index:
            offered_time += float(rng.uniform(.8, 1.2)) / rate
        rows.append(dict(id=index + 1, time=offered_time,
                         x=float(rng.uniform(.33, .55)),
                         size=[float(rng.uniform(.03, .045)),
                               float(rng.uniform(.05, .075)), .015],
                         yaw=float(rng.uniform(-np.pi, np.pi)),
                         class_name='transparent' if index % 2 == 0 else 'metal',
                         suction_p=float(config['suction_probability'])))
    return rows, rate


class EpisodeStopped(Exception):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


class OfferedTapeCore(SimCore):
    def __init__(self, cfg, tape, config):
        self.safe_tape, self.safe_config = tape, config
        self.safe_active = False
        self.safe_started = 0.
        self.safe_arrival_index = 0
        self.safe_admissions, self.safe_outcomes = [], []
        self.safe_resolved = {}
        self.safe_active_counts = []
        self.safe_collector = None
        self.safe_runner = None
        self.safe_residual_context = None
        cfg.randomize = False
        cfg.spawn_interval = float('inf')
        cfg.belt_speed = float(config['belt_speed'])
        cfg.box_count = int(config['pool_size'])
        cfg.realtime = False
        cfg.viewer = False
        cfg.record_path = ''
        cfg.physical_belt = bool(config['physical_belt'])
        cfg.belt_contact_params = str(config['belt_contact_params'])
        cfg.suction_p = {'*': float(config['suction_probability'])}
        cfg.metal_suction_binary = False
        super().__init__(cfg)
        self.safe_monitor = SafeContactMonitor(self)
        self._safe_ticks = 0

    def begin_episode(self):
        self.safe_started = self.sim_time()
        self.safe_active = True

    def safe_budget_features(self):
        total = max(len(self.safe_tape), 1)
        unresolved = sum(state != self._FREE for state in self._box_state)
        remaining = max(0., self.safe_config['time_limit'] - (self.sim_time() - self.safe_started))
        return np.asarray([(total - self.safe_arrival_index) / total,
                           unresolved / total,
                           remaining / self.safe_config['time_limit']], np.float32)

    def _spawn_box(self):
        # Native random/default spawn requests cannot modify a frozen workload.
        return None

    def _admit_offers(self):
        elapsed = self.sim_time() - self.safe_started
        while self.safe_arrival_index < len(self.safe_tape):
            row = self.safe_tape[self.safe_arrival_index]
            if row['time'] > elapsed + 1e-9:
                return
            slot = next((i for i, state in enumerate(self._box_state) if state == self._FREE), None)
            if slot is None or not self._spawn_pose_is_clear(row['x'], self.cfg.spawn_y,
                                                           row['size'], row['yaw']):
                return  # Common FIFO admission; no offered object is dropped.
            self._activate_box(slot)
            self._apply_box_size(slot, row['size'])
            joint = self._box_joints[slot]
            joint.qpos[:3] = self._to_world([row['x'], self.cfg.spawn_y,
                                           self._belt_top_w - self._base_p[2] + row['size'][2]])
            joint.qpos[3:7] = self._yaw_quat(row['yaw'])
            joint.qvel[:] = 0.
            self._box_state[slot] = self._ON_BELT
            self._box_class[slot] = row['class_name']
            self._box_suction_p[slot] = row['suction_p']
            self._box_belt_y[slot] = float(joint.qpos[1])
            self.model.geom_rgba[self._box_geom_ids[slot]] = self._box_rgba(row['class_name'], row['suction_p'])
            self._spawn_count += 1
            self._box_serial[slot] = row['id']
            self._box_manipulated[slot] = False
            self._box_manipulated_skill[slot] = None
            self._box_class_bin_name[slot] = None
            self._box_sealed[slot] = False
            self._box_reward_done[slot] = False
            self._box_cause_step[slot] = None
            self.safe_admissions.append(dict(**row, actual_time=elapsed))
            self.safe_arrival_index += 1
            # Native clearance reads body xpos/xmat, not free-joint qpos.
            # Refresh after every admission before checking a FIFO backlog.
            mujoco.mj_forward(self.model, self.data)

    def _park_box(self, slot):
        serials = getattr(self, '_box_serial', [])
        if slot < len(serials) and int(serials[slot]) > 0 and self.safe_active:
            serial = int(serials[slot])
            self.safe_resolved.setdefault(serial, dict(id=serial,
                time=self.sim_time() - self.safe_started,
                position=self._box_joints[slot].qpos[:3].copy().tolist()))
        return super()._park_box(slot)

    def _emit_reward_event(self, *args, **kwargs):
        before = len(self._reward_events)
        super()._emit_reward_event(*args, **kwargs)
        for row in self._reward_events[before:]:
            self.safe_outcomes.append(dict(row))

    def _step_once_locked(self):
        if self.safe_active and self.sim_time() - self.safe_started >= self.safe_config['time_limit'] - 1e-9:
            raise EpisodeStopped('time_limit')
        super()._step_once_locked()
        if not self.safe_active:
            return
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise EpisodeStopped('physics_nonfinite')
        self.safe_monitor.observe()
        self._admit_offers()
        self._safe_ticks += 1
        if self._safe_ticks % 10 == 0:
            self.safe_active_counts.append(sum(s != self._FREE for s in self._box_state))
            if self.safe_collector is not None:
                self.safe_collector.track_neighbors()
        context = self.safe_residual_context
        if context and context.get('prepare_end') is not None:
            record = context.get('record')
            if (record is not None and 'prepare_position_error_m' not in record
                    and self.sim_time() - context['start_sim_time'] >= context['prepare_end']):
                goal = context.get('prepare_goal_q6')
                if goal is not None and self.safe_runner is not None:
                    # Backend telemetry is refreshed at the end of its batch;
                    # this boundary is sampled inside the current substep.
                    q = np.asarray([self.data.joint(name).qpos[0] for name in MJ_JOINTS], float)
                    goal = np.asarray(goal, float)
                    record['prepare_q_error_rad'] = float(np.max(np.abs(q - goal)))
                    record['prepare_position_error_m'] = float(np.linalg.norm(
                        self.safe_runner.ee_xyz(q) - self.safe_runner.ee_xyz(goal)))

    def warp_seconds(self, duration):
        # Dense safety monitoring must not skip contacts during an idle wait.
        self.advance_seconds(duration)


class NativeResidualRunner(SimRlRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.safe_executions = []

    def _execute_action(self, action):
        began = self.clock.time()
        record = dict(task=action.skill_name,
            target_id=getattr(action.target, 'sim_object_id', None),
            start=began, end=None, success=False, detail='interrupted_at_task_deadline')
        self.safe_executions.append(record)
        try:
            result = super()._execute_action(action)
            record.update(success=bool(result.success), detail=result.detail)
            return result
        finally:
            record['end'] = self.clock.time()


def _huber(values):
    values = np.maximum(np.asarray(values, float), 0.)
    return np.where(values < 1., .5 * values**2, values - .5)


class ResidualTransitions:
    def __init__(self, runner, policy, episode_id, deterministic):
        self.runner, self.core, self.policy = runner, runner.core, policy
        self.episode_id, self.deterministic = episode_id, deterministic
        self.rows, self.pending = [], None

    def snapshot(self):
        core = self.core
        s = core.safe_monitor.snapshot()
        return dict(time=core.sim_time(), outcomes=len(core.safe_outcomes),
                    contacts=s['contact_events'], impulse=s['impulse'], falls=s['falls'])

    def neighbors(self):
        c = self.core
        return {int(c._box_serial[i]): (float(c._box_joints[i].qpos[0]), bool(c._box_manipulated[i]))
                for i, state in enumerate(c._box_state) if state != c._FREE}

    def track_neighbors(self):
        if self.pending is None:
            return
        current = self.neighbors()
        for serial, (position, manipulated) in current.items():
            if serial in self.pending['_ignored'] or manipulated:
                continue
            origin = self.pending['_origins'].setdefault(serial, position)
            self.pending['_shifts'][serial] = max(self.pending['_shifts'].get(serial, 0.), abs(position - origin))

    def decide(self, core, observation, head, phase):
        self.finalize()
        raw = np.concatenate([observation, core.safe_budget_features()]).astype(np.float32)
        if self.policy is None:
            result = np.zeros(6), np.zeros(6), 0., np.zeros(2), raw.copy()
        else:
            result = self.policy.act(raw, head, deterministic=self.deterministic)
        action, latent, logp, values, normalized = result
        target = self.runner._active_target
        ignored = {getattr(target, 'sim_object_id', None)}
        next_action = self.runner._chain_action
        if next_action is not None:
            ignored.add(getattr(next_action.target, 'sim_object_id', None))
        self.pending = dict(raw_obs=raw, obs=np.asarray(normalized).copy(), head=int(head), phase=phase,
            action=np.asarray(action).copy(), latent=np.asarray(latent).copy(), logp=float(logp),
            value=np.asarray(values).copy(), episode_id=self.episode_id,
            _before=self.snapshot(), _ignored=ignored,
            _origins={k:v[0] for k,v in self.neighbors().items() if not v[1]}, _shifts={},
            _record_start=len(self.runner.safe_adapter.records))
        return result

    def finalize(self, done=False, reason=None):
        if self.pending is None:
            return
        self.track_neighbors()
        row, self.pending = self.pending, None
        before = row.pop('_before')
        now = self.snapshot()
        outcomes = self.core.safe_outcomes[before['outcomes']:]
        successful = lambda o: o['actual_bin_name'] is not None and o['actual_bin_name'] == o['class_bin_name']
        successes = sum(successful(o) for o in outcomes)
        failures = sum(not successful(o) and (o['manipulated'] or o['collateral']) for o in outcomes)
        misses = len(outcomes) - successes - failures
        contacts = max(0, now['contacts'] - before['contacts'])
        impulse = max(0., now['impulse'] - before['impulse'])
        falls = max(0, now['falls'] - before['falls'])
        records = self.runner.safe_adapter.records[row.pop('_record_start'):]
        rejected = sum(bool(r['rejected']) for r in records)
        projection = sum(1. - r['projection'] for r in records if r.get('active', True))
        shift = float(_huber([d / .04 for d in row.pop('_shifts').values()]).sum())
        row.pop('_ignored'); row.pop('_origins')
        duration = max(0., now['time'] - before['time'])
        pose_bonus = 0.
        if row['phase'] == 'push_prepare':
            matching = next((r for r in records if 'prepare_position_error_m' in r), None)
            if matching:
                pose_bonus = .5 * float(np.exp(-(matching['prepare_position_error_m'] / .02)**2
                                                -(matching['prepare_q_error_rad'] / .06)**2))
        components = dict(bin_success=4. * successes, attempted_failure=-2. * failures,
            unprocessed=-.5 * misses, time=-.02 * duration, contact=-.4 * contacts,
            impulse=-.1 * impulse, neighbor_shift=-.2 * shift,
            push_pose=pose_bonus, projection=-.1 * projection,
            action=-.01 * float(np.mean(np.square(row['action']))))
        if done:
            resolved_events = {int(o['sim_object_id']) for o in self.core.safe_outcomes}
            # Already issued unsealed-pick failures must not be charged twice.
            components['unfinished_workload'] = -sum(
                2. if (any(int(self.core._box_serial[i]) == offered['id'] and self.core._box_manipulated[i]
                           for i in range(len(self.core._box_state)))) else .5
                for offered in self.core.safe_tape if offered['id'] not in resolved_events)
            components['physics_failure'] = -10. if reason == 'physics_nonfinite' else 0.
        row.update(reward=float(sum(components.values())),
                   cost=float(.15 * contacts + .05 * impulse + 5. * falls + 2. * rejected
                              + (10. if reason == 'physics_nonfinite' else 0.)),
                   duration=duration, done=bool(done), terminated=bool(done), truncated=False,
                   contacts=int(contacts), impulse_ns=float(impulse), collateral_falls=int(falls),
                   rejected=int(rejected), reward_components=components)
        self.rows.append(row)


def _collect_episode(scene_seed, policy, config, deterministic=False):
    begun = time.monotonic()
    config = dict(default_config(), **config)
    if config['termination'] != 'fixed_offers_then_all_physical_objects_resolved':
        raise ValueError('This protocol uses fixed offered workloads, not success-quota early termination')
    if config['offer_count'] < 1 or config['pool_size'] < 12 or config['time_limit'] <= 0:
        raise ValueError('Invalid task budget')
    tape, rate = make_tape(scene_seed, config)
    cfg = Config()
    cfg.PICK_LOG_CSV = ''
    factory = lambda sim_cfg: OfferedTapeCore(sim_cfg, tape, config)
    runner = NativeResidualRunner(cfg=cfg, sim_seed=int(scene_seed), realtime=False,
                                  max_objects=int(config['max_objects']), core_factory=factory)
    runner.truth_tracks = bool(config['truth_tracks'])
    core = runner.core
    core.safe_runner = runner
    collector = ResidualTransitions(runner, policy, f'moderate_{scene_seed}', deterministic)
    adapter = SafeResidualAdapter(policy=None, deterministic=deterministic,
                                  decision_callback=collector.decide)
    runner.safe_adapter = adapter
    adapter.bind(runner)
    core.safe_collector = collector
    reason, steps = 'workload_resolved', 0
    try:
        # Initial positioning has no target and cannot invoke the learned policy.
        runner.start()
        core.begin_episode()
        while True:
            if (core.safe_arrival_index == len(tape)
                    and all(state == core._FREE for state in core._box_state)):
                break
            core.step_index = steps
            runner.ingest()
            mask = runner.action_mask()
            selected = runner.app_heuristic_action_indices(mask)
            action = runner.action_from_indices(*selected)
            runner.step(action)
            steps += 1
    except EpisodeStopped as stopped:
        reason = stopped.reason
    finally:
        # A deadline can interrupt before the native end-of-batch recycle.
        # Resolve the already reached final state without advancing any time.
        with core.lock:
            core._recycle_boxes()
        core.safe_active = False
        collector.finalize(done=True, reason=reason)
        runner.close()
    outcomes = core.safe_outcomes
    correct = [o for o in outcomes if o['actual_bin_name'] is not None and o['actual_bin_name'] == o['class_bin_name']]
    execution_count = len(runner.safe_executions)
    elapsed = core.sim_time() - core.safe_started
    outcome_ids = {int(o['sim_object_id']) for o in outcomes}
    s = core.safe_monitor.snapshot()
    successes_by_time = sorted(o['sim_time'] - core.safe_started for o in correct)
    counts = np.asarray(core.safe_active_counts)
    summary = dict(scene_seed=int(scene_seed), offered=len(tape), admitted=len(core.safe_admissions),
        correct_bin=len(correct), task_success_percent=100. * len(correct) / len(tape),
        not_admitted=len(tape)-len(core.safe_admissions), unresolved=len(tape)-len(outcome_ids),
        physical_unresolved=sum(state != core._FREE for state in core._box_state),
        physical_resolved=len(core.safe_resolved), termination_reason=reason,
        terminated=True, truncated=False, task_time_limit_observed=True,
        sim_seconds=elapsed, wall_seconds=time.monotonic()-begun,
        actions=execution_count, policy_decisions=len(collector.rows), contacts=s['contact_events'],
        contacts_per_100_actions=100. * s['contact_events'] / max(execution_count, 1),
        impulse_ns=s['impulse'], impulse_per_action_ns=s['impulse'] / max(execution_count, 1),
        collateral_falls=s['falls'], falls_per_100_offered=100. * s['falls'] / len(tape),
        throughput_per_minute=60. * len(correct) / max(elapsed, 1e-9),
        target_success_time=(successes_by_time[config['success_target_auxiliary']-1]
                             if len(successes_by_time) >= config['success_target_auxiliary'] else None),
        requested_rate_hz=rate, active_count_mean=float(counts.mean()) if len(counts) else 0.,
        active_count_max=int(counts.max()) if len(counts) else 0,
        admission_max_delay=max((r['actual_time']-r['time'] for r in core.safe_admissions), default=0.),
        tape_sha256=hashlib.sha256(json.dumps(tape, sort_keys=True).encode()).hexdigest(),
        nonzero_decisions=sum(r['max_joint_delta'] > 1e-10 for r in adapter.records),
        max_joint_delta_rad=max((r['max_joint_delta'] for r in adapter.records), default=0.),
        rejected_decisions=sum(bool(r['rejected']) for r in adapter.records),
        decision_phases=dict(Counter(r['phase'] for r in collector.rows)),
        per_skill={task:dict(attempts=sum(r['task']==task for r in runner.safe_executions),
            correct=sum(o['manipulated_skill']==task for o in correct)) for task in ('throw','push')})
    trace = dict(tape=tape, admissions=core.safe_admissions, outcomes=outcomes,
                 physical_resolutions=list(core.safe_resolved.values()),
                 contacts=core.safe_monitor.events, falls=core.safe_monitor.falls,
                 executions=runner.safe_executions, residuals=adapter.records)
    return dict(rows=collector.rows, summary=summary, trace=trace)


def collect_episode(scene_seed, policy, config, deterministic=False):
    if config.get('quiet', True):
        with redirect_stdout(io.StringIO()):
            return _collect_episode(scene_seed, policy, config, deterministic)
    return _collect_episode(scene_seed, policy, config, deterministic)

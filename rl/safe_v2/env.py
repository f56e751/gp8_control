"""Phase-owned route learning on the preserved native simulator workload.

The full offered workload defines the cost budget. Neural decisions may split
one native motion, but cannot dilute that budget by adding more decisions.
"""
from __future__ import annotations

from collections import Counter
from contextlib import redirect_stdout
import hashlib
import io
import json
import time

import numpy as np

from gp8_control.config import Config
from gp8_control.backends.mujoco_sim import MJ_JOINTS
from gp8_control.rl.safe_simulator_env import (
    OfferedTapeCore, NativeResidualRunner, EpisodeStopped, make_tape, _huber,
    default_config as native_default_config,
)
from gp8_control.rl.safe_residual_adapter import SafeResidualAdapter as OriginalPhaseAdapter


ROLES = ('throw_loaded', 'throw_return', 'push_approach', 'push_return')
OBSERVATION_SIZE = 107
REWARD = dict(correct_bin=4., attempted_failure=-2., unprocessed=-.5, second=-.02,
              contact_event=-1., impulse_ns=-.1, proximity_second=-2.,
              neighbor_shift=-.2, push_pose_max=.5, projection_loss=-.1,
              action_mean_square=-.01)
COST = dict(contact_event=1., impulse_ns=.1, collateral_fall=10.,
            rejected_residual=2., physics_nonfinite=10.)


def default_config():
    config = native_default_config()
    config.update(residual_scale=.25, version='phase_route_v2')
    return config


class RouteCore(OfferedTapeCore):
    def __init__(self, cfg, tape, config):
        super().__init__(cfg, tape, config)
        from gp8_control.rl.safe_v2.monitor import RouteContactMonitor
        self.safe_monitor = RouteContactMonitor(self)
        self.safe_v2_actor_context = None

    def _step_once_locked(self):
        super()._step_once_locked()
        context = self.safe_v2_actor_context or {}
        record = context.get('record')
        if (self.safe_active and record is not None and 'endpoint_position_error_m' not in record
                and self.sim_time() >= float(context['phase_end_time'])-1e-9):
            q = np.asarray([self.data.joint(name).qpos[0] for name in MJ_JOINTS], float)
            goal = np.asarray(context['goal_q6'], float)
            record['endpoint_q_error_rad'] = float(np.max(np.abs(q-goal)))
            record['endpoint_position_error_m'] = float(np.linalg.norm(
                self.safe_runner.ee_xyz(q)-np.asarray(context['goal_xyz'], float)))
            record['endpoint_measured_time'] = self.sim_time()


class RouteTransitions:
    def __init__(self, runner, policy, episode_id, deterministic):
        self.runner, self.core, self.policy = runner, runner.core, policy
        self.episode_id, self.deterministic = episode_id, bool(deterministic)
        self.rows, self.pending = [], None

    def snapshot(self):
        c = self.core
        s = c.safe_monitor.snapshot()
        return dict(time=c.sim_time(), outcomes=len(c.safe_outcomes),
                    contacts=int(s['contact_events']), impulse=float(s['impulse']),
                    falls=int(s['falls']), proximity=float(s['proximity_integral']))

    def neighbors(self):
        c = self.core
        return {int(c._box_serial[i]):(float(c._box_joints[i].qpos[0]), bool(c._box_manipulated[i]))
                for i,state in enumerate(c._box_state) if state != c._FREE}

    def track_neighbors(self):
        if self.pending is None:
            return
        for serial,(position,manipulated) in self.neighbors().items():
            if serial in self.pending['_ignored'] or manipulated:
                continue
            origin = self.pending['_origins'].setdefault(serial, position)
            self.pending['_shifts'][serial] = max(self.pending['_shifts'].get(serial, 0.), abs(position-origin))

    def decide(self, core, observation, head, role):
        self.finalize()
        adapter = self.runner.safe_adapter
        monitor = core.safe_monitor.snapshot()
        role_index = ROLES.index(role)
        geometry = np.asarray([float(adapter.window_duration)/4.,
            np.clip(float(monitor['current_clearance_m']), -.05, .15)/.05,
            float(getattr(core, '_grabbed', None) is not None)], np.float32)
        raw = np.concatenate([observation, core.safe_budget_features(),
                              np.eye(4, dtype=np.float32)[role_index], geometry]).astype(np.float32)
        if raw.shape != (OBSERVATION_SIZE,) or not np.isfinite(raw).all():
            raise ValueError('Expected a finite107-feature route observation')
        if self.policy is None:
            result = np.zeros(6), np.zeros(6), 0., np.zeros(2), raw.copy()
        else:
            result = self.policy.act(raw, head, deterministic=self.deterministic)
        action, latent, logp, values, normalized = result
        owner = dict(adapter.current_owner)
        ignored = {owner.get('id')}
        chain = self.runner._chain_action
        if chain is not None:
            ignored.add(getattr(chain.target, 'sim_object_id', None))
        self.pending = dict(raw_obs=raw, obs=np.asarray(normalized).copy(), head=int(head),
            phase=role, role=role, owner_target_id=owner.get('id'),
            action=np.asarray(action).copy(), latent=np.asarray(latent).copy(),
            logp=float(logp), value=np.asarray(values).copy(), episode_id=self.episode_id,
            _before=self.snapshot(), _ignored=ignored,
            _origins={k:v[0] for k,v in self.neighbors().items() if not v[1]},
            _shifts={}, _record_start=len(adapter.records))
        return result

    def finalize(self, done=False, reason=None):
        if self.pending is None:
            return
        self.track_neighbors()
        row, self.pending = self.pending, None
        before, now = row.pop('_before'), self.snapshot()
        outcomes = self.core.safe_outcomes[before['outcomes']:]
        successes = sum(o['actual_bin_name'] is not None and o['actual_bin_name']==o['class_bin_name'] for o in outcomes)
        failures = sum(not(o['actual_bin_name'] is not None and o['actual_bin_name']==o['class_bin_name'])
                       and (o['manipulated'] or o['collateral']) for o in outcomes)
        misses = len(outcomes)-successes-failures
        contacts = max(0, now['contacts']-before['contacts'])
        impulse = max(0., now['impulse']-before['impulse'])
        falls = max(0, now['falls']-before['falls'])
        proximity = max(0., now['proximity']-before['proximity'])
        records = self.runner.safe_adapter.records[row.pop('_record_start'):]
        rejected = sum(bool(r.get('rejected', False)) for r in records)
        projection = sum(1.-float(r.get('projection', 1.)) for r in records)
        shift = float(_huber([d/.04 for d in row.pop('_shifts').values()]).sum())
        row.pop('_ignored'); row.pop('_origins')
        duration = max(0., now['time']-before['time'])
        pose = 0.
        if row['role'] == 'push_approach':
            match = next((r for r in records if r.get('endpoint_position_error_m') is not None
                          and r.get('endpoint_q_error_rad') is not None), None)
            if match is not None:
                pose = REWARD['push_pose_max']*float(np.exp(
                    -(match['endpoint_position_error_m']/.02)**2-(match['endpoint_q_error_rad']/.06)**2))
        components = dict(bin_success=REWARD['correct_bin']*successes,
            attempted_failure=REWARD['attempted_failure']*failures,
            unprocessed=REWARD['unprocessed']*misses, time=REWARD['second']*duration,
            contact=REWARD['contact_event']*contacts, impulse=REWARD['impulse_ns']*impulse,
            proximity=REWARD['proximity_second']*proximity,
            neighbor_shift=REWARD['neighbor_shift']*shift, push_pose=pose,
            projection=REWARD['projection_loss']*projection,
            action=REWARD['action_mean_square']*float(np.square(row['action']).mean()))
        if done:
            resolved = {int(o['sim_object_id']) for o in self.core.safe_outcomes}
            components['unfinished_workload'] = -sum(
                2. if any(int(self.core._box_serial[i])==offer['id'] and self.core._box_manipulated[i]
                          for i in range(len(self.core._box_state))) else .5
                for offer in self.core.safe_tape if offer['id'] not in resolved)
            components['physics_failure'] = -10. if reason=='physics_nonfinite' else 0.
        row.update(reward=float(sum(components.values())),
            cost=float(COST['contact_event']*contacts+COST['impulse_ns']*impulse
                       +COST['collateral_fall']*falls+COST['rejected_residual']*rejected
                       +(COST['physics_nonfinite'] if reason=='physics_nonfinite' else 0.)),
            duration=duration, terminated=bool(done), done=bool(done), truncated=False,
            contacts=contacts, impulse_ns=impulse, collateral_falls=falls,
            proximity_integral=proximity, rejected=rejected, reward_components=components)
        self.rows.append(row)


def _collect(scene_seed, policy, config, deterministic, original_baseline):
    from gp8_control.rl.safe_v2.adapter import SafeV2Adapter
    from gp8_control.rl.safe_v2.monitor import command_static_guard

    begun = time.monotonic()
    config = dict(default_config(), **config)
    if config['termination'] != 'fixed_offers_then_all_physical_objects_resolved':
        raise ValueError('Route learning uses a fixed offered workload and an observed finite deadline')
    tape, rate = make_tape(scene_seed, config)
    cfg = Config()
    cfg.PICK_LOG_CSV = ''
    runner = NativeResidualRunner(cfg=cfg, sim_seed=int(scene_seed), realtime=False,
        max_objects=int(config['max_objects']), core_factory=lambda c:RouteCore(c,tape,config))
    runner.truth_tracks = bool(config['truth_tracks'])
    core = runner.core
    core.safe_runner = runner
    collector = None if original_baseline else RouteTransitions(runner, policy, f'route_{scene_seed}', deterministic)
    if original_baseline:
        adapter = OriginalPhaseAdapter(policy=None, deterministic=True)
    else:
        adapter = SafeV2Adapter(policy=None, deterministic=deterministic, decision_callback=collector.decide,
            scale=float(config['residual_scale']),
            static_guard=command_static_guard)
    runner.safe_adapter = adapter
    adapter.bind(runner)
    core.safe_collector = collector
    reason, steps = 'workload_resolved', 0
    try:
        runner.start()
        core.begin_episode()
        while True:
            if core.safe_arrival_index==len(tape) and all(state==core._FREE for state in core._box_state):
                break
            core.step_index = steps
            runner.ingest()
            action = runner.action_from_indices(*runner.app_heuristic_action_indices(runner.action_mask()))
            runner.step(action)
            steps += 1
    except EpisodeStopped as stopped:
        reason = stopped.reason
    finally:
        with core.lock:
            core._recycle_boxes()
        core.safe_active = False
        if collector is not None:
            collector.finalize(done=True, reason=reason)
        runner.close()

    outcomes = core.safe_outcomes
    correct = [o for o in outcomes if o['actual_bin_name'] is not None and o['actual_bin_name']==o['class_bin_name']]
    outcome_ids = {int(o['sim_object_id']) for o in outcomes}
    execution_count = len(runner.safe_executions)
    elapsed = core.sim_time()-core.safe_started
    s = core.safe_monitor.snapshot()
    decisions = [] if collector is None else collector.rows
    records = adapter.records
    rejected = sum(bool(r.get('rejected',False)) for r in records)
    total_cost = float(COST['contact_event']*s['contact_events']+COST['impulse_ns']*s['impulse']
        +COST['collateral_fall']*s['falls']+COST['rejected_residual']*rejected
        +(COST['physics_nonfinite'] if reason=='physics_nonfinite' else 0.))
    times = sorted(o['sim_time']-core.safe_started for o in correct)
    counts = np.asarray(core.safe_active_counts)
    push_ids = {int(r['target_id']) for r in runner.safe_executions if r['task']=='push' and r['target_id'] is not None}
    approach_ids = {int(r['target_id']) for r in records if r.get('phase')=='push_approach' and r.get('active',True)}
    nonzero = [r for r in records if float(r.get('max_joint_delta',0.))>1e-10]
    assigned_contacts = sum(r['contacts'] for r in decisions)
    assigned_impulse = sum(r['impulse_ns'] for r in decisions)
    assigned_cost = sum(r['cost'] for r in decisions)
    summary = dict(scene_seed=int(scene_seed), offered=len(tape), admitted=len(core.safe_admissions),
        correct_bin=len(correct), task_success_percent=100.*len(correct)/len(tape),
        not_admitted=len(tape)-len(core.safe_admissions), unresolved=len(tape)-len(outcome_ids),
        physical_unresolved=sum(state!=core._FREE for state in core._box_state),
        physical_resolved=len(core.safe_resolved), termination_reason=reason,
        terminated=True,truncated=False,task_time_limit_observed=True,
        sim_seconds=elapsed,wall_seconds=time.monotonic()-begun,actions=execution_count,
        policy_decisions=len(decisions),contacts=int(s['contact_events']),
        contacts_per_100_actions=100.*s['contact_events']/max(execution_count,1),
        object_contacts=int(s['object_contact_events']),environment_contacts=int(s['environment_contact_events']),
        impulse_ns=float(s['impulse']),impulse_per_action_ns=float(s['impulse'])/max(execution_count,1),
        collateral_falls=int(s['falls']),falls_per_100_offered=100.*s['falls']/len(tape),
        proximity_integral=float(s['proximity_integral']),
        minimum_clearance_m=float(s['minimum_clearance_m']),
        total_cost=total_cost,cost_per_offered=total_cost/len(tape),
        transition_assigned_cost=float(assigned_cost),initial_unassigned_cost=float(total_cost-assigned_cost),
        assigned_contact_events=int(assigned_contacts),initial_unassigned_contact_events=int(s['contact_events']-assigned_contacts),
        assigned_impulse_ns=float(assigned_impulse),
        throughput_per_minute=60.*len(correct)/max(elapsed,1e-9),
        target_success_time=(times[config['success_target_auxiliary']-1]
                             if len(times)>=config['success_target_auxiliary'] else None),
        requested_rate_hz=rate,active_count_mean=float(counts.mean()) if counts.size else 0.,
        active_count_max=int(counts.max()) if counts.size else 0,
        admission_max_delay=max((r['actual_time']-r['time'] for r in core.safe_admissions),default=0.),
        tape_sha256=hashlib.sha256(json.dumps(tape,sort_keys=True).encode()).hexdigest(),
        nonzero_decisions=len(nonzero),max_joint_delta_rad=max((float(r.get('max_joint_delta',0.)) for r in records),default=0.),
        rejected_decisions=rejected,decision_phases=dict(Counter(r['phase'] for r in decisions)),
        push_unique_attempted=len(push_ids),push_with_owned_approach=len(push_ids & approach_ids),
        per_skill={task:dict(attempts=sum(r['task']==task for r in runner.safe_executions),
            correct=sum(o['manipulated_skill']==task for o in correct)) for task in ('throw','push')})
    unresolved_objects = [dict(id=int(core._box_serial[i]),state=state,class_name=core._box_class[i],
        qpos=core._box_joints[i].qpos.copy().tolist(),qvel=core._box_joints[i].qvel.copy().tolist(),
        manipulated=bool(core._box_manipulated[i]),reward_done=bool(core._box_reward_done[i]))
        for i,state in enumerate(core._box_state) if state!=core._FREE]
    trace = dict(tape=tape,admissions=core.safe_admissions,outcomes=outcomes,
        physical_resolutions=list(core.safe_resolved.values()),contacts=core.safe_monitor.events,
        falls=core.safe_monitor.falls,executions=runner.safe_executions,residuals=records,
        final_unresolved_objects=unresolved_objects)
    adapter.unbind()
    return dict(rows=decisions,summary=summary,trace=trace)


def collect_episode(scene_seed, policy, config=None, deterministic=False):
    config = dict(default_config(), **(config or {}))
    with redirect_stdout(io.StringIO()) if config.get('quiet',True) else _passthrough():
        return _collect(scene_seed,policy,config,deterministic,False)


def collect_baseline_episode(scene_seed, config=None):
    config = dict(default_config(), **(config or {}))
    with redirect_stdout(io.StringIO()) if config.get('quiet',True) else _passthrough():
        return _collect(scene_seed,None,config,True,True)


def _passthrough():
    from contextlib import nullcontext
    return nullcontext()

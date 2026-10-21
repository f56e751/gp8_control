"""Independently recompute provenance, on-policy behavior and v2 measurements."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gp8_control.rl.safe_ppo import calculate_gae
from gp8_control.rl.safe_v2.env import ROLES, COST, make_tape
from gp8_control.rl.safe_v2.ppo import FixedScaler, SafePPO, warmup_return_scales
from gp8_control.tools.audit_safe_simulator import require, same_state, load_state, json_hash
from gp8_control.tools.train_safe_simulator_v2 import ROOT, sha, source_hashes, save_json


def arrays_from(path):
    with np.load(path) as archive:
        return {key: archive[key].copy() for key in archive.files}


def rows_from(arrays):
    return [{key: value[index].item() if value[index].ndim == 0 else value[index]
             for key, value in arrays.items()} for index in range(len(arrays['head']))]


def near(actual, expected, message, tolerance=1e-6):
    require(abs(float(actual)-float(expected)) <= tolerance, message)


def verify_preservation(out, protocol):
    path = ROOT/'runs'/f"{protocol['run']}_preservation.json"
    preservation = json.loads(path.read_text())
    files = preservation['previous_files_sha256']
    for relative, digest in files.items():
        require(sha(ROOT/relative) == digest, f'Preserved v1 artifact changed: {relative}')
    budget = preservation['training_budget']
    require(budget['updates_per_model'] == protocol['training']['updates']
            and budget['episodes_per_update'] == protocol['training']['episodes_per_update']
            and budget['warmup_episodes'] == protocol['training']['warmup_episodes'],
            'Training budget changed from preregistration')
    require(budget['initialization_seeds'] == protocol['model_initialization_seeds'], 'Initialization seeds changed')
    require(preservation['regression_scene_seeds'] == protocol['evaluation']['regression_scene_seeds']
            and preservation['fresh_scene_seeds'] == protocol['evaluation']['fresh_scene_seeds'],
            'Heldout suites differ from preregistration')
    return dict(checked=len(files), all_unchanged=True, manifest_sha256=sha(path))


def verify_initialization(out, checkpoint, protocol):
    raw_path = out/'warmup_observations.npz'
    raw, warmup = arrays_from(raw_path), json.loads((out/'warmup.json').read_text())
    width = int(protocol['observation_size'])
    observations = raw['raw_obs']
    require(observations.ndim == 2 and observations.shape[1] == width == 107
            and len(observations) > 0 and np.isfinite(observations).all(), 'Invalid warmup observations')
    require(warmup['observations_sha256'] == sha(raw_path), 'Warmup observation digest differs')
    require(set(map(int, raw['scene_seed'])) <= set(protocol['training']['warmup_scene_seeds']), 'Foreign warmup scenes')
    fitted = FixedScaler(width).fit(observations)
    saved = arrays_from(out/'scaler.npz')
    np.testing.assert_array_equal(fitted.mean, saved['mean'])
    np.testing.assert_array_equal(fitted.std, saved['std'])
    require(fitted.count == int(saved['count']) == warmup['observations'], 'Warmup count differs')
    require(warmup['rows_sha256'] == sha(out/'warmup_rows.npz'), 'Warmup rows changed')
    return_scales = warmup_return_scales(rows_from(arrays_from(out/'warmup_rows.npz')))
    calibration_path = out/'calibration.json'
    calibration = json.loads(calibration_path.read_text())
    np.testing.assert_array_equal(return_scales, calibration['return_scales'])
    original = warmup['original_baseline_summaries']
    require(len(original) == protocol['training']['warmup_episodes'], 'Missing original baseline warmup episodes')
    baseline_cost = float(np.mean([s['total_cost']/s['offered'] for s in original]))
    near(calibration['baseline_mean_cost_per_offered'], baseline_cost, 'Original baseline calibration differs', 1e-10)
    budget = protocol['algorithm']['cost_budget_fraction'] * baseline_cost
    near(calibration['cost_budget_per_offered'], budget, 'Calibrated budget differs', 1e-10)
    near(protocol['algorithm']['cost_budget_per_offered'], budget, 'Protocol budget differs', 1e-10)
    require(protocol['calibration_sha256'] == sha(calibration_path), 'Frozen calibration changed')
    np.testing.assert_array_equal(return_scales, protocol['algorithm']['frozen_return_scales'])
    manifest = json.loads((out/'initialization.json').read_text())
    require(manifest['from_scratch'] and not manifest['previous_checkpoint_loaded'], 'Run was not initialized independently')
    checks = []
    for label, seed in enumerate(protocol['model_initialization_seeds']):
        path = checkpoint/f'safe_seed{label}_update000.pt'
        state = load_state(path)
        policy = SafePPO(seed, fitted, obs_size=width, return_scales=return_scales)
        require(same_state(state['network'], policy.network.state_dict()), 'Initial network does not reconstruct from seed')
        require(state['total_updates'] == state['total_actor_optimizer_steps'] == state['total_critic_optimizer_steps'] == 0,
                'Initial checkpoint contains optimizer steps')
        require(not state['actor_optimizer']['state'] and not state['critic_optimizer']['state'], 'Initial Adam is nonempty')
        require(same_state(state['scaler'], fitted.state()), 'Initial frozen scaler differs')
        np.testing.assert_array_equal(state['return_scales'], return_scales)
        require(state['metadata']['initialization_seed'] == seed
                and state['metadata']['protocol_sha256'] == sha(out/'protocol.json'), 'Initial checkpoint metadata differs')
        require(manifest['models'][label]['initial_checkpoint_sha256'] == sha(path), 'Initial checkpoint was modified')
        actor = {id(v) for group in policy.actor_optimizer.param_groups for v in group['params']}
        critic = {id(v) for group in policy.critic_optimizer.param_groups for v in group['params']}
        require(not actor.intersection(critic), 'Actor and critics share optimizer parameters')
        for head in (0, 1):
            np.testing.assert_array_equal(policy.act(observations[0], head, True)[0], np.zeros(6))
        checks.append(dict(model=label, initialization_seed=seed, zero_actor=True, both_adam_empty=True,
                           actor_critic_parameters_disjoint=True, checkpoint_sha256=sha(path)))
    return dict(raw_observations=len(observations), scaler_recomputed_exactly=True,
                return_scales_recomputed_exactly=True, cost_budget_per_offered=budget, models=checks)


def _verify_feature_layout(raw, roles, heads):
    require(raw.ndim == 2 and raw.shape[1] == 107 and np.isfinite(raw).all(), 'Invalid 107D observations')
    require(np.all(raw[:, 97:100] >= -1e-6) and np.all(raw[:, 97:100] <= 1.+1e-6), 'Invalid offered/time budget features')
    indices = [ROLES.index(str(role)) for role in roles]
    np.testing.assert_array_equal(raw[:, 100:104], np.eye(4, dtype=np.float32)[indices])
    require(np.array_equal(heads, [0 if str(role).startswith('throw') else 1 for role in roles]), 'Role/head mismatch')
    require(np.all(raw[:, 104] > 0) and np.all(raw[:, 105] >= -1.-1e-6)
            and np.all(raw[:, 105] <= 3.+1e-6), 'Invalid route duration/clearance features')
    require(np.isin(raw[:, 106], [0., 1.]).all(), 'Invalid held-object feature')


def verify_batches(out, checkpoint, protocol):
    updates, episodes = (int(protocol['training'][k]) for k in ('updates', 'episodes_per_update'))
    wanted = {f'seed{label}_update{update:03d}.npz' for label in (0, 1) for update in range(1, updates+1)}
    require({p.name for p in (out/'batches').glob('*.npz')} == wanted, 'Complete batch archive set differs')
    require(len(list((out/'batch_stats').glob('*.json'))) == 2*updates, 'Incomplete model update statistics')
    all_scenes, models, max_logp, max_value, total_rows = [], [], 0., 0., 0
    budget = float(protocol['algorithm']['cost_budget_per_offered'])
    for label, seed in enumerate(protocol['model_initialization_seeds']):
        cumulative = 0
        policy = SafePPO(seed, obs_size=107)
        for update in range(1, updates+1):
            previous_path = checkpoint/f'safe_seed{label}_update{update-1:03d}.pt'
            policy.load(previous_path)
            previous_multiplier = policy.multiplier
            require(policy.total_updates == update-1, 'Behavior checkpoint is not the previous update')
            stats = json.loads((out/'batch_stats'/f'update{update:03d}_seed{label}.json').read_text())
            summaries = json.loads((out/'episode_summaries'/f'seed{label}_update{update:03d}.json').read_text())
            expected = protocol['training']['training_scene_seeds'][str(label)][(update-1)*episodes:update*episodes]
            require(stats['scene_seeds'] == expected == [s['scene_seed'] for s in summaries], 'Full episode seed order differs')
            require(len(summaries) == episodes, 'A full workload was dropped')
            arrays = arrays_from(out/'batches'/f'seed{label}_update{update:03d}.npz')
            count = len(arrays['head'])
            require(count == stats['decisions'] == sum(s['policy_decisions'] for s in summaries) and count > 0,
                    'Decision counts do not close')
            for key, value in arrays.items():
                if np.issubdtype(value.dtype, np.number):
                    require(np.isfinite(value).all(), f'Nonfinite batch {key}')
            _verify_feature_layout(arrays['raw_obs'], arrays['role'], arrays['head'])
            np.testing.assert_allclose(policy.scaler.normalize(arrays['raw_obs']), arrays['obs'], rtol=0, atol=1e-6)
            rows = rows_from(arrays)
            calculate_gae(rows)
            actual_ids = list(dict.fromkeys(map(str, arrays['episode_id'])))
            require(actual_ids == [f"route_{s['scene_seed']}" for s in summaries if s['policy_decisions']], 'Episode chronology differs')
            for summary in summaries:
                require(summary['offered'] == protocol['episode_config']['offer_count'], 'Training changed offered quota')
                require(summary['terminated'] and not summary['truncated'], 'Finite task did not terminate')
                indices = np.flatnonzero(arrays['episode_id'] == f"route_{summary['scene_seed']}")
                near(arrays['cost'][indices].sum(), summary['transition_assigned_cost'], 'Assigned episode cost differs')
                near(summary['total_cost'], summary['transition_assigned_cost'] + summary['initial_unassigned_cost'],
                     'Full episode cost was lost or duplicated')
                require(summary['initial_unassigned_cost'] >= -1e-6, 'Assigned cost exceeds full solver cost')
                if not len(indices):
                    continue
                require(np.array_equal(indices, np.arange(indices[0], indices[-1]+1)), 'Interleaved episode decisions')
                require(arrays['terminated'][indices].sum() == 1 and arrays['terminated'][indices[-1]]
                        and not arrays['truncated'][indices].any(), 'Incorrect terminal/GAE boundary')
                require(np.all(np.diff(arrays['raw_obs'][indices, 99]) <= 1e-6), 'Remaining task time increased')
                computed = (COST['contact_event']*arrays['contacts'][indices]
                    + COST['impulse_ns']*arrays['impulse_ns'][indices]
                    + COST['collateral_fall']*arrays['collateral_falls'][indices]
                    + COST['rejected_residual']*arrays['rejected'][indices])
                if summary['termination_reason'] == 'physics_nonfinite':
                    computed[-1] += COST['physics_nonfinite']
                np.testing.assert_allclose(computed, arrays['cost'][indices], rtol=0, atol=1e-5)
            with torch.no_grad():
                distribution, values = policy.network(torch.as_tensor(arrays['obs'], dtype=torch.float32),
                                                     torch.as_tensor(arrays['head'], dtype=torch.long))
                latent = torch.as_tensor(arrays['latent'], dtype=torch.float32)
                actions = torch.tanh(latent)
                logp = (distribution.log_prob(latent)-torch.log(1-actions.square()+1e-6)).sum(-1).numpy()
            np.testing.assert_allclose(actions.numpy(), arrays['action'], rtol=0, atol=1e-6)
            errors = [float(np.max(np.abs(logp-arrays['logp']))), float(np.max(np.abs(values.numpy()-arrays['value'])))]
            require(max(errors) <= 1e-4, 'Behavior log probability or original-unit values differ')
            max_logp, max_value = max(max_logp, errors[0]), max(max_value, errors[1])
            measured = float(np.mean([s['total_cost']/s['offered'] for s in summaries]))
            near(stats['cost_per_offered'], measured, 'Constraint used decision count or missed a workload')
            near(stats['cost_budget_per_offered'], budget, 'Constraint budget changed')
            near(stats['multiplier_before'], previous_multiplier, 'Wrong Lagrange starting state')
            expected_multiplier = float(np.clip(previous_multiplier+.5*(measured-budget)/max(budget, .1), 0., 30.))
            near(stats['multiplier'], expected_multiplier, 'Lagrange workload update differs')
            near(stats['mean_reward'], arrays['reward'].mean(), 'Mean reward differs')
            near(stats['mean_cost_per_decision_diagnostic'], arrays['cost'].mean(), 'Diagnostic mean cost differs')
            cumulative += count
            state = load_state(checkpoint/f'safe_seed{label}_update{update:03d}.pt')
            require(state['total_updates'] == update and state['metadata']['decisions'] == cumulative,
                    'Checkpoint update/decision counters differ')
            require(state['metadata']['protocol_sha256'] == sha(out/'protocol.json'), 'Checkpoint protocol changed')
            for kind in ('actor', 'critic'):
                require(state[f'total_{kind}_optimizer_steps'] == stats[f'total_{kind}_optimizer_steps'], 'Adam step counter differs')
                require(bool(state[f'{kind}_optimizer']['state']), f'{kind} Adam state is absent')
            np.testing.assert_array_equal(state['return_scales'], protocol['algorithm']['frozen_return_scales'])
            require(stats['total_optimizer_steps'] == state['total_actor_optimizer_steps']+state['total_critic_optimizer_steps'],
                    'Combined optimizer count differs')
            total_rows += count
            all_scenes.extend(expected)
        final, archived = checkpoint/f'safe_seed{label}.pt', checkpoint/f'safe_seed{label}_update{updates:03d}.pt'
        require(same_state(load_state(final), load_state(archived)), 'Latest policy is not the full fixed last checkpoint')
        models.append(dict(model=label, updates=updates, episodes=updates*episodes, decisions=cumulative,
                           final_checkpoint_sha256=sha(final), both_adam_present=True))
    require(len(all_scenes) == len(set(all_scenes)), 'Training scenes overlap')
    require(not set(all_scenes).intersection(protocol['training']['warmup_scene_seeds']), 'Warmup/train overlap')
    summary = json.loads((out/'training_summary.json').read_text())
    require(summary['training_complete'] and summary['updates_per_model'] == updates
            and json.loads((out/'committed_update.json').read_text())['update'] == updates, 'Final pair did not commit')
    return dict(batches=2*updates, full_episodes=2*updates*episodes, decisions=total_rows,
                max_behavior_logp_error=max_logp, max_behavior_value_error=max_value,
                phase_cost_recomputed=True, full_workload_budget_recomputed=True, models=models)


PHYSICAL_TRACE_KEYS = ('tape', 'admissions', 'outcomes', 'physical_resolutions', 'contacts',
                       'falls', 'executions', 'final_unresolved_objects')


def verify_rollout(result, config, policy=None):
    summary, trace = result['summary'], result['trace']
    tape, offered = trace['tape'], int(config['offer_count'])
    ids = list(range(1, offered+1))
    require([int(r['id']) for r in tape] == ids and summary['offered'] == offered, 'Wrong offered workload')
    require(summary['tape_sha256'] == json_hash(tape), 'Offered tape digest differs')
    admissions, outcomes, resolutions = (trace[k] for k in ('admissions', 'outcomes', 'physical_resolutions'))
    admitted = [int(r['id']) for r in admissions]
    require(admitted == ids[:len(admitted)], 'Admission dropped or reordered an offer')
    for index, row in enumerate(admissions):
        require({k:v for k,v in row.items() if k != 'actual_time'} == tape[index], 'Admitted object differs from tape')
        require(row['actual_time'] >= row['time']-1e-7, 'An object arrived before its offer')
    near(summary['admitted'], len(admitted), 'Admission count differs')
    near(summary['not_admitted'], offered-len(admitted), 'Unadmitted offers were lost')
    outcome_ids, resolution_ids = [int(r['sim_object_id']) for r in outcomes], [int(r['id']) for r in resolutions]
    require(len(outcome_ids) == len(set(outcome_ids)) and set(outcome_ids) <= set(admitted), 'Outcome duplicated or nonexistent')
    require(len(resolution_ids) == len(set(resolution_ids)) and set(resolution_ids) <= set(outcome_ids), 'Physical resolution duplicated or lost')
    require(summary['physical_resolved'] == len(resolutions)
            and summary['physical_unresolved']+len(resolutions) == len(admitted), 'Physical workload closure differs')
    require(summary['unresolved'] == offered-len(outcome_ids), 'Unresolved outcome count differs')
    require({int(r['id']) for r in trace['final_unresolved_objects']} == set(admitted)-set(resolution_ids), 'Final physical objects differ')
    correct = [r for r in outcomes if r['actual_bin_name'] is not None and r['actual_bin_name'] == r['class_bin_name']]
    require(summary['correct_bin'] == len(correct), 'Correct-bin count differs')
    near(summary['task_success_percent'], 100.*len(correct)/offered, 'Success denominator differs')
    contacts, falls, records = trace['contacts'], trace['falls'], trace['residuals']
    require(summary['actions'] == len(trace['executions']) and summary['contacts'] == len(contacts), 'Action/contact count differs')
    require(summary['collateral_falls'] == len(falls) == len({f['object_id'] for f in falls}), 'Falls duplicated')
    near(summary['impulse_ns'], sum(r['impulse'] for r in contacts), 'Impulse differs')
    environment = [r for r in contacts if r['kind'] == 'robot_environment']
    require(summary['environment_contacts'] == len(environment)
            and summary['object_contacts'] == len(contacts)-len(environment), 'Environment/object contact split differs')
    rejected = sum(bool(r.get('rejected')) for r in records)
    total_cost = COST['contact_event']*len(contacts)+COST['impulse_ns']*summary['impulse_ns']
    total_cost += COST['collateral_fall']*len(falls)+COST['rejected_residual']*rejected
    if summary['termination_reason'] == 'physics_nonfinite':
        total_cost += COST['physics_nonfinite']
    near(summary['total_cost'], total_cost, 'Full solver cost differs')
    near(summary['cost_per_offered'], total_cost/offered, 'Full workload cost denominator differs')
    elapsed = summary['sim_seconds']
    require(summary['terminated'] and not summary['truncated'] and elapsed <= config['time_limit']+.01, 'Finite deadline semantics differ')
    if summary['termination_reason'] == 'workload_resolved':
        require(not summary['physical_unresolved'] and len(admitted) == offered, 'Natural termination left physical work')
    elif summary['termination_reason'] == 'time_limit':
        near(elapsed, config['time_limit'], 'Time-limit terminal is early/late', .01)
    else:
        raise AssertionError(f"Evaluation ended in a physics failure: {summary['termination_reason']}")
    target = int(config['success_target_auxiliary'])
    require((summary['target_success_time'] is not None) == (len(correct) >= target), 'Success-target auxiliary differs')
    active = [r for r in records if r.get('active')]
    decisions = result['decisions']
    if result['policy'] == 'baseline':
        require(not decisions and summary['policy_decisions'] == 0, 'Original baseline has learned decisions')
        require(all(r['exact_identity'] and not np.any(r['action']) for r in active), 'Original baseline was edited')
        return dict(decisions=0, max_final_actor_error=0., offered=offered, contacts=len(contacts))
    require(len(decisions) == summary['policy_decisions'] == len(active), 'Decision records do not close')
    require(all(b['actual_boundary_time'] > a['actual_boundary_time'] for a,b in zip(active, active[1:])), 'Decisions are prospective or simultaneous')
    maximum_error = 0.
    for row, record in zip(decisions, active):
        raw = np.asarray(row['raw_obs'], np.float32)
        _verify_feature_layout(raw[None], [row['role']], [row['head']])
        np.testing.assert_allclose(raw[:97], record['observation'], rtol=0, atol=1e-6)
        np.testing.assert_allclose(row['action'], record['action'], rtol=0, atol=1e-7)
        require(record['role'] == row['role'] and record['head'] == row['head'], 'Physical phase owner differs')
        if policy is not None:
            action, latent, logp, value, normalized = policy.act(raw, row['head'], deterministic=True)
            error = max(float(np.max(np.abs(action-row['action']))), float(np.max(np.abs(value-row['value']))))
            maximum_error = max(maximum_error, error)
            require(error <= 1e-6, 'Evaluation actor/critic is not the fixed deterministic final model')
            np.testing.assert_allclose(latent, record['latent'], rtol=0, atol=1e-6)
            np.testing.assert_allclose(normalized, record['normalized'], rtol=0, atol=1e-6)
            near(logp, record['logp'], 'Evaluation log probability differs', 1e-5)
        else:
            require(not np.any(row['action']) and record['exact_identity'], 'v2 zero ablation changed native commands')
        require(record['protected_position_equal'] and record['protected_velocity_equal'], 'Protected seam changed')
        require(record['candidate_changed_max_ratio'] <= 1.+1e-7
                and record['acceleration_limit_ratio'] <= 1.+1e-7, 'Physical command dynamics guard failed')
        require(record['max_joint_delta'] <= config['residual_scale']+1e-8, 'Residual exceeded its action range')
        near(row['reward'], sum(row['reward_components'].values()), 'Recorded reward components do not add up')
    near(sum(r['cost'] for r in decisions), summary['transition_assigned_cost'], 'Evaluation assigned cost differs')
    near(summary['transition_assigned_cost']+summary['initial_unassigned_cost'], total_cost, 'Evaluation full cost lost at a phase boundary')
    return dict(decisions=len(decisions), max_final_actor_error=maximum_error, offered=offered, contacts=len(contacts))


def verify_identity(path):
    path = Path(path)
    proof = json.loads(path.read_text())
    require(proof['passed'] and all(proof['identity_checks'].values()), 'Preflight zero/native identity failed')
    require(proof['baseline_control'] == proof['zero_control'], 'Preflight commands/IO/final states differ')
    for relative, digest in proof['sources_sha256'].items():
        require(sha(ROOT/relative) == digest, f'Preflight source changed: {relative}')
    for label in ('baseline', 'zero'):
        result = json.loads((path.parent/f"{label}_scene{proof['scene_seed']}.json").read_text())
        for key in PHYSICAL_TRACE_KEYS:
            require(json_hash(result['trace'][key]) == proof['trace_sha256'][label][key], 'Preflight trace changed')
        require(result['control'] == proof[f'{label}_control'], 'Preflight control artifact changed')
    return dict(passed=True, scene_seed=proof['scene_seed'], offered=proof['offer_count'],
                commands_io_final_state_exact=True, proof_sha256=sha(path))


def verify_evaluation(out, checkpoint, protocol):
    path = out/'evaluation_protocol.json'
    definition = json.loads(path.read_text())
    require(definition['training_protocol_sha256'] == sha(out/'protocol.json'), 'Evaluation protocol changed')
    require(definition['source_sha256'] == protocol['source_sha256'], 'Evaluation sources differ from training')
    require(definition['episode_config'] == protocol['episode_config'] and definition['deterministic'], 'Evaluation conditions differ')
    arms = ['baseline', 'v2_zero', 'safe_seed0', 'safe_seed1']
    require(definition['policies'] == arms, 'Expected original baseline, v2-zero and two fixed models')
    scenes = definition['scene_seeds']
    expected = protocol['evaluation']['regression_scene_seeds']+protocol['evaluation']['fresh_scene_seeds']
    require(scenes == expected and len(scenes) == len(set(scenes)), 'Heldout scene suites changed')
    training = set(protocol['training']['warmup_scene_seeds'])
    for values in protocol['training']['training_scene_seeds'].values():
        training.update(values)
    require(not training.intersection(scenes), 'Evaluation leaked into training')
    policies = {}
    for label, seed in enumerate(protocol['model_initialization_seeds']):
        p = checkpoint/f'safe_seed{label}.pt'
        require(definition['checkpoint_sha256'][str(p.relative_to(ROOT))] == sha(p), 'Evaluated final checkpoint changed')
        policy = SafePPO(seed, obs_size=107)
        policy.load(p)
        policies[f'safe_seed{label}'] = policy
    require({p.name for p in (out/'evaluation').glob('*.json')} ==
            {f'{arm}_scene{scene}.json' for arm in arms for scene in scenes}, 'Evaluation set is incomplete or contains extra rollouts')
    checks, identical = [], 0
    for scene in scenes:
        tape, _ = make_tape(scene, definition['episode_config'])
        results = {}
        for arm in arms:
            result = json.loads((out/'evaluation'/f'{arm}_scene{scene}.json').read_text())
            require(result['scene_seed'] == scene and result['policy'] == arm
                    and result['evaluation_protocol_sha256'] == sha(path), 'Evaluation rollout identity differs')
            require(result['trace']['tape'] == tape, 'Policies received different external offers')
            checks.append(dict(scene_seed=scene, policy=arm,
                               **verify_rollout(result, definition['episode_config'], policies.get(arm))))
            results[arm] = result
        for key in PHYSICAL_TRACE_KEYS:
            require(results['baseline']['trace'][key] == results['v2_zero']['trace'][key],
                    f'Zero ablation changed physical trace {key} in scene {scene}')
        near(results['baseline']['summary']['sim_seconds'], results['v2_zero']['summary']['sim_seconds'], 'Zero actor changed physical episode time', 1e-10)
        near(results['baseline']['summary']['total_cost'], results['v2_zero']['summary']['total_cost'], 'Zero actor changed solver cost', 1e-8)
        identical += 1
    completion = json.loads((out/'evaluation_summary.json').read_text())
    require(completion['complete'] and completion['result_count'] == len(checks), 'Evaluation did not complete')
    return dict(scene_count=len(scenes), rollout_count=len(checks), fixed_tapes_equal=True,
                zero_baseline_physical_trace_identity_scenes=identical,
                max_final_actor_error=max(r['max_final_actor_error'] for r in checks), checks=checks)


def audit(run='feature_safe_route_20261021_v2', *, training_only=False, identity_proof=None):
    out, checkpoint = ROOT/'runs'/run, ROOT/'checkpoints'/run
    path = out/'protocol.json'
    protocol = json.loads(path.read_text())
    result = dict(run=run, passed=False, training_only=bool(training_only),
                  protocol_sha256=sha(path), evaluation_complete=False)
    output = out/('audit_training.json' if training_only else 'audit.json')
    try:
        require(protocol['version'] == 2 and not protocol['previous_checkpoint_loaded'], 'Unexpected experiment version/initialization')
        require(protocol['source_sha256'] == source_hashes(), 'Frozen runtime/trainer sources changed')
        result['sources_frozen'] = True
        result['previous_artifacts'] = verify_preservation(out, protocol)
        result['initialization'] = verify_initialization(out, checkpoint, protocol)
        result['training'] = verify_batches(out, checkpoint, protocol)
        proof = identity_proof or ROOT/'runs'/'feature_route_adapter_axis_pilot_20261021'/'adapter_verification.json'
        result['native_identity'] = verify_identity(proof)
        if not training_only:
            result['evaluation'] = verify_evaluation(out, checkpoint, protocol)
            result['evaluation_complete'] = True
        result['passed'] = True
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        save_json(output, result)
        raise
    save_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='feature_safe_route_20261021_v2')
    parser.add_argument('--training-only', action='store_true')
    parser.add_argument('--identity-proof')
    args = parser.parse_args()
    result = audit(args.run, training_only=args.training_only, identity_proof=args.identity_proof)
    print(json.dumps({k:result[k] for k in ('run', 'passed', 'training_only', 'evaluation_complete')}))


if __name__ == '__main__':
    main()

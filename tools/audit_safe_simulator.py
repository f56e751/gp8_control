"""Recompute the provenance and measurements of a native-simulator Safe PPO run."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from gp8_control.rl.safe_ppo import FixedScaler, SafePPO, calculate_gae
from gp8_control.tools.train_safe_simulator import ROOT, save_json, sha, source_hashes


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def same_state(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same_state(a[key], b[key]) for key in a)
    if isinstance(a, (tuple, list)):
        return isinstance(b, type(a)) and len(a) == len(b) and all(same_state(x, y) for x, y in zip(a, b))
    return a == b


def load_state(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def verify_initialization(out, checkpoint, protocol):
    with np.load(out/'warmup_observations.npz') as raw:
        observations = raw['raw_obs'].copy()
        scenes = raw['scene_seed'].copy()
    width = protocol['observation_size']
    require(observations.ndim == 2 and observations.shape[1] == width and len(observations) > 0,
            'Warmup must contain actual finite decision observations')
    require(np.isfinite(observations).all(), 'Warmup contains nonfinite observations')
    warmup = json.loads((out/'warmup.json').read_text())
    require(warmup['observations_sha256'] == sha(out/'warmup_observations.npz'), 'Warmup raw archive hash differs')
    require(set(map(int, scenes)) <= set(protocol['training']['warmup_scene_seeds']), 'Unexpected warmup seed')
    fitted = FixedScaler(width).fit(observations)
    with np.load(out/'scaler.npz') as saved:
        np.testing.assert_allclose(fitted.mean, saved['mean'], rtol=0, atol=0)
        np.testing.assert_allclose(fitted.std, saved['std'], rtol=0, atol=0)
        require(fitted.count == int(saved['count']) == warmup['observations'], 'Warmup observation count differs')
    manifest = json.loads((out/'initialization.json').read_text())
    require(manifest['from_scratch'] and not manifest['previous_checkpoint_loaded'], 'Initialization is not independent')
    records = []
    for label, seed in enumerate(protocol['model_initialization_seeds']):
        path = checkpoint/f'safe_seed{label}_update000.pt'
        state = load_state(path)
        policy = SafePPO(seed, fitted, obs_size=width)
        require(same_state(state['network'], policy.network.state_dict()), f'Model{label} does not reconstruct from declared seed{seed}')
        require(state['total_updates'] == state['total_optimizer_steps'] == 0, 'Initial checkpoint contains training steps')
        require(not state['optimizer']['state'], 'Initial Adam state is nonempty')
        require(same_state(state['scaler'], fitted.state()), 'Initial checkpoint scaler differs')
        require(state['metadata']['initialization_seed'] == seed, 'Initial seed metadata differs')
        saved_manifest = manifest['models'][label]
        require(saved_manifest['initial_checkpoint_sha256'] == sha(path), 'Initial checkpoint was modified')
        for head in (0, 1):
            require(np.array_equal(policy.act(observations[0], head, True)[0], np.zeros(6)), 'Initial actor is nonzero')
        records.append(dict(model=label, initialization_seed=seed, empty_adam=True,
                            zero_actor=True, reconstructed_exactly=True, checkpoint_sha256=sha(path)))
    return dict(raw_observations=len(observations), scaler_recomputed_exactly=True,
                warmup_scene_count=len(set(map(int, scenes))), models=records)


def verify_batches(out, checkpoint, protocol):
    updates, episodes = (protocol['training'][key] for key in ('updates', 'episodes_per_update'))
    stats_paths = sorted((out/'batch_stats').glob('update*_seed*.json'))
    require(len(stats_paths) == 2*updates, 'Training statistics do not cover both complete model runs')
    expected_npz = {f'seed{label}_update{update:03d}.npz' for label in (0, 1) for update in range(1, updates+1)}
    require({p.name for p in (out/'batches').glob('*.npz')} == expected_npz, 'Training batch archive set differs')
    total_rows, total_episodes = 0, 0
    maximum_logp = maximum_value = 0.
    all_training_scenes = []
    model_results = []
    for label in (0, 1):
        cumulative = 0
        for update in range(1, updates+1):
            previous_path = checkpoint/f'safe_seed{label}_update{update-1:03d}.pt'
            previous = load_state(previous_path)
            policy = SafePPO(protocol['model_initialization_seeds'][label], obs_size=protocol['observation_size'])
            policy.load(previous_path)
            require(policy.total_updates == update-1, 'Behavior checkpoint is not the preceding PPO update')
            stats = json.loads((out/'batch_stats'/f'update{update:03d}_seed{label}.json').read_text())
            expected_scenes = protocol['training']['training_scene_seeds'][str(label)][(update-1)*episodes:update*episodes]
            require(stats['scene_seeds'] == expected_scenes, 'Training batch scene order differs from frozen protocol')
            summaries = json.loads((out/'episode_summaries'/f'seed{label}_update{update:03d}.json').read_text())
            require([s['scene_seed'] for s in summaries] == expected_scenes, 'Episode summary order differs')
            require(len(summaries) == episodes, 'A full offered-workload episode was dropped')
            require(all(s['terminated'] and not s['truncated'] for s in summaries), 'Expected finite-task terminals')
            with np.load(out/'batches'/f'seed{label}_update{update:03d}.npz') as data:
                arrays = {key:data[key].copy() for key in data.files}
            count = len(arrays['head'])
            require(count == sum(s['policy_decisions'] for s in summaries) == stats['decisions'], 'Decision counts do not close')
            require(count > 0, 'Empty PPO batch')
            for key, arr in arrays.items():
                if np.issubdtype(arr.dtype, np.number):
                    require(np.isfinite(arr).all(), f'Nonfinite {key} in batch')
            np.testing.assert_allclose(policy.scaler.normalize(arrays['raw_obs']), arrays['obs'], rtol=0, atol=1e-6)
            require(arrays['obs'].shape == (count, protocol['observation_size']), 'Incorrect observation width')
            require(np.all(arrays['raw_obs'][:, -3:] >= -1e-7)
                    and np.all(arrays['raw_obs'][:, -3:] <= 1.+1e-7), 'Invalid observed workload/time budget feature')
            rows = [{key:arrays[key][i].item() if arrays[key][i].ndim == 0 else arrays[key][i]
                     for key in arrays if key != 'phase'} for i in range(count)]
            calculate_gae(rows)
            ordered_ids = list(dict.fromkeys(map(str, arrays['episode_id'])))
            expected_ids = [f"moderate_{s['scene_seed']}" for s in summaries if s['policy_decisions']]
            require(ordered_ids == expected_ids, 'Chronological decision episodes differ from submitted scenes')
            for identity in ordered_ids:
                indices = np.where(arrays['episode_id'] == identity)[0]
                require(np.array_equal(indices, np.arange(indices[0], indices[-1]+1)), 'Interleaved episode decisions')
                require(arrays['terminated'][indices].sum() == 1 and arrays['terminated'][indices[-1]], 'Missing unique final terminal')
                require(not arrays['truncated'][indices].any(), 'Unexpected artificial truncation')
                require(np.all(np.diff(arrays['raw_obs'][indices, -1]) <= 1e-7), 'Observed remaining time increases within an episode')
            with torch.no_grad():
                obs = torch.as_tensor(arrays['obs'], dtype=torch.float32)
                heads = torch.as_tensor(arrays['head'], dtype=torch.long)
                latent = torch.as_tensor(arrays['latent'], dtype=torch.float32)
                distribution, values = policy.network(obs, heads)
                action = torch.tanh(latent)
                logp = (distribution.log_prob(latent)-torch.log(1-action.square()+1e-6)).sum(-1).numpy()
            np.testing.assert_allclose(action.numpy(), arrays['action'], rtol=0, atol=1e-6)
            logp_error = float(np.max(np.abs(logp-arrays['logp'])))
            value_error = float(np.max(np.abs(values.numpy()-arrays['value'])))
            require(max(logp_error, value_error) <= 1e-4, 'Stored behavior is not the actual previous policy')
            maximum_logp, maximum_value = max(maximum_logp, logp_error), max(maximum_value, value_error)
            require(abs(float(arrays['cost'].mean())-stats['mean_cost']) < 1e-6, 'Mean cost differs')
            require(abs(float(arrays['reward'].mean())-stats['mean_reward']) < 1e-6, 'Mean reward differs')
            cumulative += count
            require(stats['cumulative_decisions'] == cumulative, 'Cumulative decision counter differs')
            current = load_state(checkpoint/f'safe_seed{label}_update{update:03d}.pt')
            require(current['total_updates'] == update and current['metadata']['decisions'] == cumulative, 'Update checkpoint counter differs')
            require(current['metadata']['protocol_sha256'] == sha(out/'protocol.json'), 'Checkpoint frozen protocol differs')
            require(current['total_optimizer_steps'] == stats['total_optimizer_steps'], 'Optimizer step count differs')
            total_rows += count
            total_episodes += len(summaries)
            all_training_scenes.extend(expected_scenes)
        final_path = checkpoint/f'safe_seed{label}.pt'
        archive_path = checkpoint/f'safe_seed{label}_update{updates:03d}.pt'
        require(same_state(load_state(final_path), load_state(archive_path)), 'Latest checkpoint is not the full fixed final update')
        model_results.append(dict(model=label, updates=updates, episodes=updates*episodes,
            decisions=cumulative, final_checkpoint_sha256=sha(final_path), final_archive_sha256=sha(archive_path)))
    require(len(all_training_scenes) == len(set(all_training_scenes)), 'Training scene seeds overlap across models/updates')
    require(not set(all_training_scenes).intersection(protocol['training']['warmup_scene_seeds']), 'Training and warmup overlap')
    summary = json.loads((out/'training_summary.json').read_text())
    require(summary['training_complete'] and summary['updates_per_model'] == updates, 'Training did not complete its frozen budget')
    require(json.loads((out/'committed_update.json').read_text())['update'] == updates, 'Final pair was not committed')
    return dict(batches=2*updates, full_episodes=total_episodes, decisions=total_rows,
        max_behavior_logp_error=maximum_logp, max_behavior_value_error=maximum_value,
        all_episodes_chronological=True, models=model_results)


def verify_identity(path):
    path = Path(path)
    proof = json.loads(path.read_text())
    require(proof.get('passed'), 'Native/zero actor identity proof failed')
    require(all(proof['identity_checks'].values()), 'A native/zero actor identity check failed')
    require(all(proof['live_collector_checks'].values()), 'A live collector/guard check failed')
    for relative, expected in proof['source_sha256'].items():
        require(sha(ROOT/relative) == expected, f'Identity proof has stale source {relative}')
    for name, expected in proof['artifacts'].items():
        require(sha(path.parent/name) == expected, f'Identity proof artifact changed: {name}')
    require(proof['control']['baseline'] == proof['control']['zero_actor'], 'Native and zero actor physical controls differ')
    return dict(passed=True, scope=proof['verification_scope'], proof_sha256=sha(path),
        source_sha256=proof['source_sha256'], physical_control_samples=proof['control']['baseline']['count'])


def verify_rollout(result, config, policy=None):
    summary, trace = result['summary'], result['trace']
    require(trace is not None, 'Evaluation needs raw trace for closure checks')
    tape = trace['tape']
    offered = config['offer_count']
    require(len(tape) == offered and summary['offered'] == offered, 'Wrong fixed workload size')
    ids = [int(r['id']) for r in tape]
    require(ids == list(range(1, offered+1)), 'Offered IDs are not the fixed workload')
    require(summary['tape_sha256'] == json_hash(tape), 'Tape digest differs')
    admissions, outcomes, resolutions = (trace[key] for key in ('admissions', 'outcomes', 'physical_resolutions'))
    admission_ids = [r['id'] for r in admissions]
    require(admission_ids == ids[:len(admissions)], 'Admission is not FIFO or silently dropped an offered object')
    require(all(r['actual_time'] + 1e-7 >= r['time'] for r in admissions), 'Object admitted before offered')
    require(all(a['actual_time'] <= b['actual_time'] for a,b in zip(admissions, admissions[1:])), 'Admission time order differs')
    require(summary['admitted'] == len(admissions) and summary['not_admitted'] == offered-len(admissions), 'Admission counters differ')
    outcome_ids = [int(o['sim_object_id']) for o in outcomes]
    resolution_ids = [int(o['id']) for o in resolutions]
    require(len(outcome_ids) == len(set(outcome_ids)) and set(outcome_ids) <= set(admission_ids), 'Duplicated or nonexistent outcome')
    require(len(resolution_ids) == len(set(resolution_ids)) and set(resolution_ids) <= set(admission_ids), 'Duplicated or nonexistent physical resolution')
    require(summary['physical_resolved'] == len(resolutions), 'Physical resolution count differs')
    require(summary['physical_unresolved'] + len(resolutions) == len(admissions), 'Physical outcome closure differs')
    require(summary['unresolved'] == offered-len(outcome_ids), 'Unresolved offered outcome count differs')
    correct = [o for o in outcomes if o['actual_bin_name'] is not None and o['actual_bin_name'] == o['class_bin_name']]
    require(summary['correct_bin'] == len(correct), 'Correct-bin count differs')
    require(abs(summary['task_success_percent']-100.*len(correct)/offered) < 1e-8, 'Task success denominator differs')
    require(summary['actions'] == len(trace['executions']), 'Physical action count differs')
    require(summary['contacts'] == len(trace['contacts']), 'Contact event count differs')
    require(summary['collateral_falls'] == len(trace['falls']), 'Collateral fall count differs')
    require(abs(summary['impulse_ns']-sum(c['impulse'] for c in trace['contacts'])) < 1e-7, 'Contact impulse sum differs')
    elapsed = summary['sim_seconds']
    require(elapsed <= config['time_limit']+.01, 'Observed task budget was exceeded')
    require(summary['terminated'] and not summary['truncated'], 'Finite-budget evaluation must be a true terminal')
    if summary['termination_reason'] == 'workload_resolved':
        require(not summary['physical_unresolved'] and len(admissions) == offered, 'Natural completion still has unresolved physical work')
    elif summary['termination_reason'] == 'time_limit':
        require(abs(elapsed-config['time_limit']) < .01, 'Time-limit terminal occurred at the wrong time')
    else:
        raise AssertionError(f"Evaluation physics/runtime failure: {summary['termination_reason']}")
    target = int(config['success_target_auxiliary'])
    reached = len(correct) >= target
    require((summary['target_success_time'] is not None) == reached, 'Auxiliary target reach indicator differs')
    if reached:
        require(0 <= summary['target_success_time'] <= elapsed+.01, 'Auxiliary success time outside episode')
        # Correct-bin events are emitted at the same native physics step that
        # parks the object. Their absolute and episode-relative timestamps
        # independently recover the positioning offset without changing trace.
        resolution_times = {int(item['id']):item['time'] for item in resolutions}
        offsets = np.asarray([o['sim_time']-resolution_times[int(o['sim_object_id'])] for o in correct])
        require(np.ptp(offsets) < 1e-7, 'Correct-bin outcome/resolution time offsets differ')
        target_time = sorted(o['sim_time']-float(offsets[0]) for o in correct)[target-1]
        require(abs(summary['target_success_time']-target_time) < 1e-7, 'Auxiliary target time differs from raw outcomes')
    residuals = trace['residuals']
    active = [r for r in residuals if r.get('active')]
    decisions = result['decisions']
    require(len(decisions) == summary['policy_decisions'] == len(active), 'Evaluation decision count differs')
    maximum_error = 0.
    for decision, record in zip(decisions, active):
        raw = np.asarray(decision['raw_obs'], np.float32)
        require(raw.shape == (100,) and np.isfinite(raw).all(), 'Invalid evaluation observation')
        require(record['head'] == decision['head'], 'Actor head differs from physical residual record')
        np.testing.assert_allclose(raw[:97], record['observation'], rtol=0, atol=1e-6)
        np.testing.assert_allclose(decision['action'], record['action'], rtol=0, atol=1e-7)
        if policy is not None:
            action, latent, _, value, normalized = policy.act(raw, decision['head'], deterministic=True)
            error = max(float(np.max(np.abs(action-np.asarray(decision['action'])))),
                        float(np.max(np.abs(value-np.asarray(decision['value'])))))
            maximum_error = max(maximum_error, error)
            require(error <= 1e-6, 'Evaluation did not use the deterministic fixed final policy')
            np.testing.assert_allclose(normalized, record['normalized'], rtol=0, atol=1e-6)
            np.testing.assert_allclose(latent, record['latent'], rtol=0, atol=1e-6)
        else:
            require(not np.any(decision['action']) and record['exact_identity'], 'Baseline commands contain a residual')
            require(record['nominal_position_sha256'] == record['executed_position_sha256']
                    and record['nominal_velocity_sha256'] == record['executed_velocity_sha256'], 'Baseline trajectory changed')
        require(record['protected_position_equal'] and record['protected_velocity_equal'], 'Protected physical command boundary changed')
        require(record['candidate_changed_max_ratio'] <= 1.+1e-8, 'Changed physical velocity violates M1')
        require(record['max_joint_delta'] <= config['residual_scale']+1e-8, 'Residual exceeds frozen geometry scale')
    return dict(decisions=len(decisions), max_final_actor_error=maximum_error,
                offered=offered, contacts=len(trace['contacts']), physical_actions=len(trace['executions']))


def verify_evaluation(out, checkpoint, protocol):
    from gp8_control.rl.safe_simulator_env import make_tape
    evaluation_path = out/'evaluation_protocol.json'
    definition = json.loads(evaluation_path.read_text())
    require(definition['training_protocol_sha256'] == sha(out/'protocol.json'), 'Evaluation training definition changed')
    require(definition['episode_config'] == protocol['episode_config'], 'Evaluation environment differs from training')
    require(definition['policies'] == ['baseline', 'safe_seed0', 'safe_seed1'], 'Expected original baseline and two fixed models')
    require(definition['deterministic'], 'Evaluation must be deterministic')
    train_scenes = set(protocol['training']['warmup_scene_seeds'])
    for values in protocol['training']['training_scene_seeds'].values():
        train_scenes.update(values)
    scenes = definition['scene_seeds']
    preservation_path = ROOT/'runs'/f"{protocol['run']}_preservation.json"
    if preservation_path.exists():
        preservation = json.loads(preservation_path.read_text())
        require(scenes == preservation['heldout_scene_seeds'], 'Evaluation seeds differ from the preregistered heldout suite')
    require(not train_scenes.intersection(scenes), 'Evaluation leaked into training/warmup')
    require(len(scenes) == len(set(scenes)), 'Duplicated evaluation scene')
    policies = {}
    for label in (0, 1):
        path = checkpoint/f'safe_seed{label}.pt'
        relative = str(path.relative_to(ROOT))
        require(definition['checkpoint_sha256'][relative] == sha(path), 'Evaluated checkpoint digest differs')
        policy = SafePPO(protocol['model_initialization_seeds'][label], obs_size=protocol['observation_size'])
        policy.load(path)
        policies[f'safe_seed{label}'] = policy
    wanted = {f'{name}_scene{scene}.json' for name in definition['policies'] for scene in scenes}
    require({path.name for path in (out/'evaluation').glob('*.json')} == wanted, 'Evaluation result suite is incomplete or includes extra scenes')
    checks = []
    for scene in scenes:
        expected_tape, _ = make_tape(scene, definition['episode_config'])
        common_hash = None
        for name in definition['policies']:
            path = out/'evaluation'/f'{name}_scene{scene}.json'
            result = json.loads(path.read_text())
            require(result['evaluation_protocol_sha256'] == sha(evaluation_path), 'Rollout used a different evaluation protocol')
            require(result['scene_seed'] == scene and result['policy'] == name, 'Evaluation result identity differs')
            require(result['trace']['tape'] == expected_tape, 'Evaluation external workload was not the seeded fixed tape')
            tape_hash = result['summary']['tape_sha256']
            if common_hash is None:
                common_hash = tape_hash
            require(tape_hash == common_hash, 'Policies were compared on different offered workloads')
            checks.append(dict(scene_seed=scene, policy=name,
                **verify_rollout(result, definition['episode_config'], policies.get(name))))
    completion = json.loads((out/'evaluation_summary.json').read_text())
    require(completion['complete'] and completion['result_count'] == len(checks), 'Evaluation is not complete')
    return dict(scene_count=len(scenes), rollout_count=len(checks), fixed_tapes_equal=True,
        deterministic_final_actor_recomputed=True,
        max_final_actor_error=max(c['max_final_actor_error'] for c in checks),
        offered_objects_per_policy=len(scenes)*protocol['episode_config']['offer_count'], checks=checks)


def audit(run='feature_safe_20261020_v1', *, training_only=False, identity_proof=None):
    out, checkpoint = ROOT/'runs'/run, ROOT/'checkpoints'/run
    protocol_path = out/'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    result = dict(run=run, passed=False, training_only=bool(training_only),
                  protocol_sha256=sha(protocol_path), evaluation_complete=False)
    try:
        require(protocol['source_sha256'] == source_hashes(), 'Frozen simulator/trainer source or asset hashes changed')
        result['sources_frozen'] = True
        preservation_path = ROOT/'runs'/f'{run}_preservation.json'
        if preservation_path.exists():
            preservation = json.loads(preservation_path.read_text())
            for relative, expected in preservation['previous_artifacts_sha256'].items():
                require(sha(ROOT/relative) == expected, f'Previous experiment artifact changed: {relative}')
            budget = preservation['training_budget']
            require(budget['updates_per_model'] == protocol['training']['updates']
                    and budget['episodes_per_update'] == protocol['training']['episodes_per_update'],
                    'Training budget differs from preregistration')
            result['previous_artifacts'] = dict(checked=len(preservation['previous_artifacts_sha256']),
                all_unchanged=True, preservation_manifest_sha256=sha(preservation_path))
        result['initialization'] = verify_initialization(out, checkpoint, protocol)
        result['training'] = verify_batches(out, checkpoint, protocol)
        result['native_identity'] = verify_identity(identity_proof or out/'implementation_verification.json')
        if not training_only:
            result['evaluation'] = verify_evaluation(out, checkpoint, protocol)
            result['evaluation_complete'] = True
        result['passed'] = True
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        save_json(out/('audit_training.json' if training_only else 'audit.json'), result)
        raise
    save_json(out/('audit_training.json' if training_only else 'audit.json'), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='feature_safe_20261020_v1')
    parser.add_argument('--training-only', action='store_true')
    parser.add_argument('--identity-proof')
    args = parser.parse_args()
    result = audit(args.run, training_only=args.training_only, identity_proof=args.identity_proof)
    print(json.dumps({key:result[key] for key in ('run', 'passed', 'training_only', 'evaluation_complete')}, ensure_ascii=False))


if __name__ == '__main__':
    main()

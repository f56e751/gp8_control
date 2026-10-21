"""Evaluate frozen stream-trained policies on reset single-skill scenes.

This is a transfer test, with no model fitting, reward tuning, or scene filtering.
Run from the repository parent with ``python -m gp8_control.tools.evaluate_safe_fixed_scene``.
The original stream experiments and their checkpoints are read-only inputs.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import gc
import hashlib
import io
import json
import multiprocessing as mp
from pathlib import Path
import re
import resource
import time
import traceback

import numpy as np

from gp8_control.tools.train_safe_simulator_v2 import ROOT, sha, save_json, source_hashes as v2_source_hashes

ARMS = ('baseline', 'v2_zero', 'safe_seed0', 'safe_seed1')
SKILLS = ('throw', 'push')
NEIGHBORS = (2, 5)
MODEL_RUN = 'feature_safe_route_20261021_v2'
METRICS = ('target_success_percent', 'contacts_per100', 'impulse_ns_per_attempt',
           'falls_per100', 'collision_free_success_percent', 'object_contacts_per100',
           'push_prepare_contacts_per100', 'attempted_percent')
IDENTITY_KEYS = ('command_sha256', 'command_timeline_sha256', 'io_sha256', 'final_state_sha256')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def payload_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def run_path(run):
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}', run), 'Use a plain new run name')
    require(run not in (MODEL_RUN, 'feature_safe_20261020_v1'), 'Original runs are read-only')
    return ROOT/'runs'/run


def fixed_sources():
    files = (ROOT/'rl/safe_fixed_scene_env.py', Path(__file__).resolve())
    require(all(path.is_file() for path in files), 'The new fixed-scene environment is not ready')
    return dict(v2_source_hashes(), **{str(path.relative_to(ROOT)): sha(path) for path in files})


def preservation_snapshot():
    """Freeze every existing original artifact, including reports and all updates."""
    files = set()
    for run in ('feature_safe_20261020_v1', MODEL_RUN):
        for folder in ('runs', 'checkpoints'):
            files.update(path for path in (ROOT/folder/run).rglob('*') if path.is_file())
        for path in (ROOT/'runs').glob(run+'_preservation.json'):
            files.add(path)
    return {str(path.relative_to(ROOT)): sha(path) for path in sorted(files)}


def enumerate_scenes(count, pilot=False):
    require(count >= (1 if pilot else 32), 'Use at least32 scenes per stratum outside a named pilot')
    scenes = []
    for index, (skill, neighbors) in enumerate((s, n) for s in SKILLS for n in NEIGHBORS):
        start = (5900000+index*1000) if pilot else (5100000+index*100000)
        for seed in range(start, start+count):
            scenes.append(dict(target_skill=skill, neighbor_count=neighbors, scene_seed=seed,
                               scene_id=f'{skill}_n{neighbors}_scene{seed}'))
    require(len({s['scene_seed'] for s in scenes}) == len(scenes), 'Repeated scene seeds')
    return scenes


def load_inputs():
    """Accept only the audited final128-update v2 checkpoints, never model selection."""
    from gp8_control.rl.safe_v2.ppo import SafePPO
    model_out = ROOT/'runs'/MODEL_RUN
    protocol_path = model_out/'protocol.json'
    trained = json.loads(protocol_path.read_text())
    complete = json.loads((model_out/'training_summary.json').read_text())
    audit = json.loads((model_out/'audit.json').read_text())
    require(complete.get('training_complete') and complete['updates_per_model'] == 128,
            'The original fixed128-update training must be complete')
    require(audit.get('passed') and audit.get('evaluation_complete'), 'The original full audit must pass')
    require(trained['source_sha256'] == v2_source_hashes(), 'Frozen native/v2 runtime source changed')
    states, hashes = {}, {}
    for label in (0, 1):
        policy = SafePPO(40+label)
        path = ROOT/'checkpoints'/MODEL_RUN/f'safe_seed{label}.pt'
        meta = policy.load(path)
        require(meta.get('run') == MODEL_RUN and meta.get('model') == label
                and meta.get('update') == policy.total_updates == 128
                and meta.get('initialization_seed') == 40+label
                and meta.get('protocol_sha256') == sha(protocol_path), 'Foreign or nonfinal checkpoint')
        states[f'safe_seed{label}'] = policy.rollout_state()
        hashes[str(path.relative_to(ROOT))] = sha(path)
    used = set(trained['training']['warmup_scene_seeds'])
    for values in trained['training']['training_scene_seeds'].values():
        used.update(values)
    return states, hashes, trained, used


def read_config(path=None):
    from gp8_control.rl.safe_fixed_scene_env import default_config
    config = default_config()
    if path:
        updates = json.loads(Path(path).read_text())
        require(set(updates) <= set(config), 'Unknown fixed-scene config key')
        config.update(updates)
    require(config.get('belt_speed') == .12 and config.get('physical_belt') is True,
            'The transfer test uses the frozen physical belt at0.12m/s')
    require(config.get('action_horizon') == 15., 'All arms require a fixed15-second action-start horizon')
    require(config.get('time_limit') == 120. and config.get('selection_target_y') == .58,
            'Keep the trained120s budget denominator and preregistered0.58m selection gate')
    require(float(config.get('time_limit', 0.)) >= float(config.get('prelude_limit', 0.))+15.,
            'The total deadline must accommodate every full observation horizon')
    return config


def prepare(args):
    out = run_path(args.run)
    if args.fresh:
        require(not out.exists() or not any(path.name != 'old_evidence.json' for path in out.iterdir()),
                'Fresh run already contains protocol or result files')
    else:
        require((out/'protocol.json').is_file(), 'Resume requires an existing frozen protocol')
    states, checkpoints, trained, used = load_inputs()
    scenes = enumerate_scenes(args.seeds_per_stratum, args.pilot)
    require(not used.intersection(s['scene_seed'] for s in scenes), 'Evaluation overlaps training/warmup')
    config = read_config(args.episode_config)
    definition = dict(format_version=1, run=args.run, experiment='fixed-reset single-skill transfer test',
        date='2026-10-21', pilot=bool(args.pilot), model_run=MODEL_RUN,
        model_initialization_seeds=[40, 41], training=False, model_selection='fixed final update128',
        training_protocol_sha256=sha(ROOT/'runs'/MODEL_RUN/'protocol.json'),
        checkpoint_sha256=checkpoints, source_sha256=fixed_sources(), policies=list(ARMS),
        scenes=scenes, seeds_per_stratum=args.seeds_per_stratum, episode_config=config,
        observation_size=107, action_size=6, deterministic=True,
        scene_semantics=dict(initial_placement='one target and2or5 neighbors at reset; no additional offers',
            layout_version='native_belt_supported_push_backswing_v2',
            target_x_m=dict(throw=[.36,.44], push=[.47,.49]), target_y_m=[2.20,2.24],
            layout_acceptance='only initial AABB nonoverlap and whole-box belt support with8mm margin; no simulated outcome filtering',
            prelude='native camera/intake and physical belt to the fixed target-y<=0.58m geometric gate, then the first feasible unchanged native selection; no outcome-dependent filtering',
            reset='same robot integration state, object placement, geometry, friction and native tracking/queue',
            action_start='state and native selection hashed before the first neural callback',
            observation='all arms continue to action_start+15s; unresolved is retained as failure',
            failures='every scheduled scene remains in the denominator, including no selection or wrong outcome'),
        statistics=dict(primary_denominator='all scheduled single-skill opportunities; one per scene',
            learned='mean of the two fixed trained models within each paired scene',
            difference='learned minus original baseline', bootstrap_resamples=10000, bootstrap_seed=20261021,
            bootstrap_unit='scene seed, stratified within neighbor counts for combined skill rows',
            uncertainty='environment-seed uncertainty conditional on these two trained models',
            success='the intended complete native skill finishes and the target resolves into its correct bin during the common window; a horizon-interrupted skill is a failure',
            contacts='all unintended solver contact events starting from action_start to action_start+15s; object-only and Push preparation are secondary',
            collision_free_success='correct target bin plus zero action-window event starts and action-window impulse<=1e-9Ns; prelude-started continuing events therefore prevent collision-free success',
            prelude='Policy-free prelude contacts are retained separately and excluded from the primary action-window metric.',
            attempt='Per attempt means per scheduled opportunity, including unsuccessful selection; actual attempted percentage is also shown.',
            sensor='simulator solver state and handled-object heuristic; not a measured real sensor'),
        limitations=['Transfer of frozen stream-trained107D policies; no fixed-scene-specific learning.',
            'No additional offers; native single-skill selection and dynamics remain in use.',
            'The same120s time-budget denominator is retained. Remaining offers=0, unresolved is normalized by3or6 initial objects instead of20, and next-action chaining is disabled; the input distribution differs from training.',
            'Baseline/zero identity is checked on complete command/IO/final-state traces.',
            'The fixed horizon may leave targets unresolved; those scenes count as failures.',
            'Two trained seeds are shown separately; the bootstrap does not estimate a population of trained models.'])
    if (out/'old_evidence.json').is_file():
        definition['old_evidence_sha256'] = sha(out/'old_evidence.json')
    if args.resume:
        previous = json.loads((out/'protocol.json').read_text())
        # Preservation is bound once before the very first episode.
        definition['preservation_manifest_sha256'] = previous['preservation_manifest_sha256']
        require(previous == definition, 'Resume arguments, native sources or frozen models differ')
    else:
        out.mkdir(parents=True, exist_ok=True)
        save_json(out/'preservation.json', dict(existing_files_sha256=preservation_snapshot()))
        definition['preservation_manifest_sha256'] = sha(out/'preservation.json')
        save_json(out/'protocol.json', definition)
    verify_freeze(out, definition)
    return out, definition, states


def verify_freeze(out, protocol):
    require(protocol['source_sha256'] == fixed_sources(), 'Runtime/native/asset source changed during evaluation')
    for relative, digest in protocol['checkpoint_sha256'].items():
        require(sha(ROOT/relative) == digest, f'Original final checkpoint changed: {relative}')
    manifest_path = out/'preservation.json'
    require(sha(manifest_path) == protocol['preservation_manifest_sha256'], 'Preservation manifest changed')
    if protocol.get('old_evidence_sha256'):
        require(sha(out/'old_evidence.json') == protocol['old_evidence_sha256'], 'Original fixed-scene evidence changed')
    previous = json.loads(manifest_path.read_text())['existing_files_sha256']
    for relative, digest in previous.items():
        require((ROOT/relative).is_file() and sha(ROOT/relative) == digest,
                f'Original experiment artifact changed: {relative}')
    return dict(source_count=len(protocol['source_sha256']), preserved_artifact_count=len(previous),
                all_original_bytes_unchanged=True)


def worker(job):
    started = time.monotonic()
    initial = resource.getrusage(resource.RUSAGE_SELF)
    capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            import torch
            from gp8_control.rl.safe_v2.ppo import SafePPO
            from gp8_control.rl.safe_fixed_scene_env import collect_episode
            torch.set_num_threads(1)
            policy = None
            if job['state'] is not None:
                policy = SafePPO(0)
                policy.restore_rollout_state(job['state'])
            result = collect_episode(job['scene_seed'], job['target_skill'], job['neighbor_count'],
                policy, config=job['config'], original_baseline=job['policy'] == 'baseline', deterministic=True)
        require(isinstance(result, dict) and all(k in result for k in
            ('summary', 'trace', 'rows', 'initial_state_sha256', 'selection_sha256')), 'Invalid collector result')
    except Exception:
        result = dict(execution_error=traceback.format_exc())
    gc.collect()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = None
    try:
        rss = next(float(line.split()[1])/1024. for line in Path('/proc/self/status').read_text().splitlines()
                   if line.startswith('VmRSS:'))
    except (OSError, StopIteration):
        pass
    result['worker_resources'] = dict(pid=os.getpid(), wall_seconds=time.monotonic()-started,
        user_cpu_seconds=usage.ru_utime-initial.ru_utime, system_cpu_seconds=usage.ru_stime-initial.ru_stime,
        peak_rss_mib=usage.ru_maxrss/1024., rss_after_gc_mib=rss)
    if capture.getvalue():
        result['simulator_log'] = capture.getvalue()
    return result


def evaluate(args):
    out, protocol, states = prepare(args)
    if args.prepare_only:
        print(json.dumps(dict(prepared=True, protocol=str(out/'protocol.json'), scenes=len(protocol['scenes']))), flush=True)
        return
    result_dir = out/'evaluation'
    result_dir.mkdir(exist_ok=True)
    protocol_sha = sha(out/'protocol.json')
    jobs = []
    for scene in protocol['scenes']:
        for arm in ARMS:
            path = result_dir/f"{scene['scene_id']}_{arm}.json"
            if path.exists():
                old = json.loads(path.read_text())
                require(old.get('evaluation_protocol_sha256') == protocol_sha and not old.get('execution_error'),
                        f'Existing episode is foreign or has an execution error: {path.name}')
                continue
            jobs.append(dict(**scene, policy=arm, state=states.get(arm), config=protocol['episode_config'], path=path))
    expected = len(protocol['scenes'])*len(ARMS)
    completed, started = expected-len(jobs), time.monotonic()
    print(json.dumps(dict(stage='evaluation_start', completed=completed, expected=expected,
                          workers=args.workers, pilot=protocol['pilot'])), flush=True)
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn')) as executor:
        futures = {executor.submit(worker, {k:v for k,v in job.items() if k != 'path'}):job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            result = future.result()
            save_json(job['path'], dict(policy=job['policy'], scene_seed=job['scene_seed'], scene_id=job['scene_id'],
                target_skill=job['target_skill'], neighbor_count=job['neighbor_count'],
                evaluation_protocol_sha256=protocol_sha, **result))
            require(not result.get('execution_error'), f"Collector execution failed; preserved in{job['path']}: {result.get('execution_error')}")
            completed += 1
            if completed % max(4, args.progress_every) == 0 or completed == expected:
                elapsed = time.monotonic()-started
                save_json(out/'progress.json', dict(completed=completed, expected=expected, elapsed_seconds=elapsed))
                print(json.dumps(dict(completed=completed, expected=expected, elapsed_seconds=round(elapsed, 2))), flush=True)
    save_json(out/'evaluation_summary.json', dict(complete=True, scene_count=len(protocol['scenes']), policy_count=4,
        result_count=expected, evaluation_protocol_sha256=protocol_sha,
        wall_seconds_this_invocation=time.monotonic()-started))
    audit(out)
    summarize(out)
    print('FIXED_SCENE_EVALUATION complete', flush=True)


def read_results(out, protocol):
    expected = {f"{scene['scene_id']}_{arm}.json" for scene in protocol['scenes'] for arm in ARMS}
    require({p.name for p in (out/'evaluation').glob('*.json')} == expected, 'Incomplete or additional evaluation episodes')
    raw = {}
    for scene in protocol['scenes']:
        group = {}
        for arm in ARMS:
            result = json.loads((out/'evaluation'/f"{scene['scene_id']}_{arm}.json").read_text())
            require(not result.get('execution_error') and result['evaluation_protocol_sha256'] == sha(out/'protocol.json'),
                    'Foreign or errored result')
            require(all(result[key] == scene[key] for key in ('scene_seed', 'scene_id', 'target_skill', 'neighbor_count'))
                    and result['policy'] == arm, 'Episode identity differs from its preregistered scene')
            group[arm] = result
        raw[scene['scene_id']] = group
    return raw


def episode_metrics(summary):
    required = ('target_success', 'action_contacts', 'action_object_contacts', 'action_impulse_ns', 'action_collateral_falls',
                'preparation_contacts', 'attempted_skill')
    require(all(key in summary for key in required), f'Missing fixed-scene metric fields: {set(required)-set(summary)}')
    success = float(bool(summary['target_success']))
    contact = float(summary['action_contacts'])
    require(all(np.isfinite(float(summary[k])) and float(summary[k]) >= 0 for k in required),
            'Invalid physical measurement')
    return dict(target_success_percent=100.*success, contacts_per100=100.*contact,
        impulse_ns_per_attempt=float(summary['action_impulse_ns']), falls_per100=100.*float(summary['action_collateral_falls']),
        collision_free_success_percent=100.*success*float(contact == 0. and float(summary['action_impulse_ns']) <= 1e-9),
        object_contacts_per100=100.*float(summary['action_object_contacts']),
        push_prepare_contacts_per100=100.*float(summary['preparation_contacts']),
        attempted_percent=100.*float(bool(summary['attempted_skill'])))


def verify_trace(result):
    """Derive every digest and reported risk count from the saved raw trace."""
    trace, summary = result['trace'], result['summary']
    payloads = dict(reset_state_sha256=trace['reset_state'],
        action_start_state_sha256=trace['action_start_state'], initial_state_sha256=trace['action_start_state'],
        selection_sha256=trace['selection'], command_sha256=trace['commands'],
        command_timeline_sha256=dict(times=trace['command_times'], commands=trace['commands']),
        io_sha256=trace['io'], final_state_sha256=trace['final_state'])
    for key, value in payloads.items():
        computed = payload_sha(value)
        require(trace.get(key) == computed, f'Saved trace digest does not reconstruct: {key}')
        for source in (summary, result):
            if key in source:
                require(source[key] == computed, f'Episode/summary digest differs: {key}')
    require(summary['scene_sha256'] == payload_sha(trace['scene']), 'Seeded layout digest differs')
    require(summary['physical_end_objects_sha256'] == payload_sha(trace['end_objects']), 'Object terminal-state digest differs')
    start = float(trace['action_start'])
    events = [e for e in trace['contacts'] if e['start_time'] >= start-1e-9]
    falls = [e for e in trace['falls'] if e['sim_time'] >= start-1e-9]
    counts = dict(contacts=len(trace['contacts']), action_contacts=len(events), collateral_falls=len(trace['falls']),
        action_collateral_falls=len(falls), action_object_contacts=sum(e['kind']!='robot_environment' for e in events),
        preparation_contacts=sum(e['phase']=='push_prepare' for e in events), actions=len(trace['executions']))
    for key, value in counts.items():
        require(summary[key] == value, f'Raw trace count differs: {key}')
    impulse = sum(e['impulse'] for e in trace['contacts'])
    require(abs(summary['impulse_ns']-impulse) < 1e-6, 'Accumulated solver impulse does not close')
    require(abs(summary['action_impulse_ns']-(impulse-trace['prelude_monitor']['impulse'])) < 1e-6,
            'Action-window solver impulse does not close')
    require(abs(summary['observation_duration']-(trace['observation_end']-start)) < 1e-9,
            'Observation duration differs from trace')
    selected = trace['selection']
    intended = selected['target_id'] == selected['intended_target_id'] and selected['skill'] == selected['requested_skill']
    target_outcome = next((o for o in trace['outcomes'] if o['sim_object_id']==selected['intended_target_id']), None)
    success = bool(intended and trace['skill_end'] is not None and summary['termination_reason']!='physics_nonfinite'
                   and target_outcome and target_outcome['actual_bin_name'] is not None
                   and target_outcome['actual_bin_name']==target_outcome['class_bin_name'])
    require(summary['target_success'] == success and summary['target_outcome'] == target_outcome,
            'Target outcome/success does not reconstruct from all outcomes')
    require(summary['attempted_skill'] == bool(trace['executions']), 'Native attempt flag differs')
    require(not summary['initially_grasped'] and trace['reset_state']['grabbed'] is None,
            'The target must start ungrasped')
    objects = trace['action_start_objects']
    require(summary['action_start_object_count'] == sum(o['state']!='free' for o in objects),
            'Action-start live-object count differs')
    require(summary['action_start_neighbor_count'] == sum(o['state']!='free' and o['id']!=selected['intended_target_id'] for o in objects),
            'Action-start neighbor count differs')
    require(summary['action_start_neighbors_on_belt'] == sum(o['state']=='on_belt' and o['id']!=selected['intended_target_id'] for o in objects),
            'Action-start belt-supported neighbor count differs')
    return dict(intended_target_selected=intended, protocol_issue=summary.get('protocol_issue'))


def audit(out):
    """Recompute identities and final-network outputs without changing any model."""
    protocol = json.loads((out/'protocol.json').read_text())
    preserved = verify_freeze(out, protocol)
    states, checkpoints, _, _ = load_inputs()
    require(checkpoints == protocol['checkpoint_sha256'], 'Frozen model input changed')
    raw = read_results(out, protocol)
    from gp8_control.rl.safe_v2.ppo import SafePPO
    from gp8_control.rl.safe_fixed_scene_env import make_scene
    policies = {}
    for arm, state in states.items():
        policies[arm] = SafePPO(0)
        policies[arm].restore_rollout_state(state)
    count, rows_checked, error = 0, 0, 0.
    for scene in protocol['scenes']:
        group = raw[scene['scene_id']]
        for key in ('initial_state_sha256', 'selection_sha256'):
            require(len({result[key] for result in group.values()}) == 1,
                    f"Initial physical state/native selection differs: {scene['scene_id']} {key}")
        for key in ('reset_state_sha256', 'action_start_state_sha256', 'selection_sha256'):
            require(all(result['trace'].get(key) for result in group.values()), f'Missing state identity: {key}')
            require(len({result['trace'][key] for result in group.values()}) == 1,
                    f"Initial native/core metadata differs: {scene['scene_id']} {key}")
        for key in IDENTITY_KEYS:
            require(group['baseline']['trace'].get(key) and group['baseline']['trace'][key] == group['v2_zero']['trace'].get(key),
                    f"Original/zero physical identity differs: {scene['scene_id']} {key}")
        for key in ('commands', 'command_times', 'io', 'contacts', 'falls', 'outcomes',
                    'physical_resolutions', 'executions', 'end_objects'):
            require(group['baseline']['trace'][key] == group['v2_zero']['trace'][key],
                    f"Original/zero physical or monitor telemetry differs: {scene['scene_id']} {key}")
        count += 1
        for arm, result in group.items():
            summary = result['summary']
            require(result['trace']['scene'] == make_scene(scene['scene_seed'], scene['target_skill'],
                    scene['neighbor_count'], protocol['episode_config']), 'Seeded initial geometry does not reconstruct')
            verify_trace(result)
            episode_metrics(summary)
            require(summary['neighbor_count'] == scene['neighbor_count'] and summary['requested_skill'] == scene['target_skill'],
                    'Target stratum differs')
            require('target_outcome' in summary, 'Unresolved target outcome must be retained explicitly')
            if summary['attempted_skill'] and summary['termination_reason'] not in ('physics_nonfinite', 'deadline'):
                require(abs(float(summary['observation_duration'])-protocol['episode_config']['action_horizon']) < .02,
                        'Unequal observation horizon')
            if arm == 'baseline':
                require(not result['rows'], 'Original baseline unexpectedly invoked a neural policy')
            for row in result['rows']:
                require(np.shape(row['raw_obs']) == (107,) and np.isfinite(row['raw_obs']).all(), 'Invalid transfer observation')
                require(np.shape(row['action']) == (6,) and np.isfinite(row['action']).all(), 'Invalid residual action')
                if arm == 'v2_zero':
                    require(np.array_equal(row['action'], np.zeros(6)), 'Zero policy produced a nonzero residual')
                    continue
                expected = policies[arm].act(row['raw_obs'], row['head'], deterministic=True)
                for key, value in zip(('action', 'latent', 'logp', 'value', 'obs'), expected):
                    delta = float(np.max(np.abs(np.asarray(row[key])-np.asarray(value))))
                    error = max(error, delta)
                    require(delta < 1e-5, f'Final frozen policy output differs: {key}')
                rows_checked += 1
    result = dict(passed=True, evaluation_complete=True, protocol_sha256=sha(out/'protocol.json'),
        scene_count=count, result_count=count*4, paired_initial_state_and_selection_exact=True,
        original_zero_full_physical_identity_exact=True, frozen_policy_decisions_recomputed=rows_checked,
        raw_payload_hashes_recomputed=True, original_zero_solver_contact_outcome_telemetry_exact=True,
        maximum_policy_output_error=error, **preserved,
        note='Only evaluation/inference ran; no optimizer update, tuning or scene exclusion.')
    save_json(out/'audit.json', result)
    return result


def paired_group(scenes, raw, seed=20261021, resamples=10000):
    """Resample scene clusters inside each density; never duplicate training seeds as N."""
    require(bool(scenes), 'Empty group')
    data = {arm:np.array([[episode_metrics(raw[s['scene_id']][arm]['summary'])[key] for key in METRICS]
                          for s in scenes]) for arm in ARMS}
    learned = .5*(data['safe_seed0']+data['safe_seed1'])
    difference = learned-data['baseline']
    rng = np.random.default_rng(seed)
    bootstrap = np.zeros((resamples, len(METRICS)))
    for neighbors in sorted({s['neighbor_count'] for s in scenes}):
        indices = np.array([i for i,s in enumerate(scenes) if s['neighbor_count'] == neighbors])
        selections = rng.integers(0, len(indices), size=(resamples, len(indices)))
        bootstrap += difference[indices][selections].sum(axis=1)/len(scenes)
    confidence = np.quantile(bootstrap, [.025, .975], axis=0)
    def means(values):
        return dict(zip(METRICS, map(float, values.mean(axis=0))))
    per_policy = {arm:means(data[arm]) for arm in ARMS}
    def outcome_label(summary):
        outcome = summary['target_outcome']
        if outcome is None:
            return 'unresolved'
        if summary['target_success']:
            return 'correct_bin'
        return 'failed_or_wrong_bin'
    outcomes = {arm:dict(Counter(outcome_label(raw[s['scene_id']][arm]['summary']) for s in scenes)) for arm in ARMS}
    stage_info = {}
    for arm in ARMS:
        summaries = [raw[s['scene_id']][arm]['summary'] for s in scenes]
        phases = sorted({phase for row in summaries for phase in row.get('action_event_start_phases', {})})
        stage_info[arm] = dict(event_start_phases={phase:dict(
            contacts_per100=100.*sum(row.get('action_event_start_phases',{}).get(phase,{}).get('count',0)
                                     for row in summaries)/len(scenes),
            event_accumulated_impulse_ns_per_scene=sum(row.get('action_event_start_phases',{}).get(phase,{}).get('impulse_ns',0.)
                                     for row in summaries)/len(scenes)) for phase in phases},
            intended_target_selected_count=sum(bool(row.get('intended_target_selected')) for row in summaries),
            protocol_issue_counts=dict(Counter(str(row['protocol_issue']) for row in summaries if row.get('protocol_issue'))),
            prelude_contacts_per_scene=float(np.mean([row.get('prelude_contacts',0) for row in summaries])),
            prelude_impulse_ns_per_scene=float(np.mean([row.get('prelude_impulse_ns',0.) for row in summaries])),
            action_start_neighbors_mean=float(np.mean([row.get('action_start_neighbor_count',0) for row in summaries])),
            action_start_neighbors_on_belt_mean=float(np.mean([row.get('action_start_neighbors_on_belt',0) for row in summaries])))
    return dict(n_scenes=len(scenes), n_rollouts=4*len(scenes), scene_ids=[s['scene_id'] for s in scenes],
        baseline=per_policy['baseline'], learned=means(learned), per_policy=per_policy,
        paired_differences={key:dict(mean=float(difference[:,i].mean()),
            ci95=list(map(float, confidence[:,i]))) for i,key in enumerate(METRICS)}, target_outcomes=outcomes, stage_info=stage_info,
        termination_counts={arm:dict(Counter(raw[s['scene_id']][arm]['summary']['termination_reason'] for s in scenes)) for arm in ARMS})


def summarize(out):
    protocol = json.loads((out/'protocol.json').read_text())
    passed = json.loads((out/'audit.json').read_text())
    require(passed.get('passed') and passed['protocol_sha256'] == sha(out/'protocol.json'), 'A complete audit is required')
    raw = read_results(out, protocol)
    groups = []
    for skill in SKILLS:
        for neighbors in (*NEIGHBORS, 'all'):
            scenes = [s for s in protocol['scenes'] if s['target_skill'] == skill
                      and (neighbors == 'all' or s['neighbor_count'] == neighbors)]
            groups.append(dict(target_skill=skill, neighbor_count=neighbors,
                **paired_group(scenes, raw, resamples=protocol['statistics']['bootstrap_resamples'],
                               seed=protocol['statistics']['bootstrap_seed'])))
    resources = [r['worker_resources'] for group in raw.values() for r in group.values()]
    result = dict(complete=True, experiment=protocol['experiment'], transfer_test=True, training=False,
        protocol_sha256=sha(out/'protocol.json'), model_run=MODEL_RUN, audit_passed=True,
        scene_count=len(protocol['scenes']), result_count=len(protocol['scenes'])*4,
        groups=groups, statistics=protocol['statistics'], limitations=protocol['limitations'],
        worker_resources=dict(episode_wall_seconds_mean=float(np.mean([r['wall_seconds'] for r in resources])),
            episode_user_cpu_seconds_mean=float(np.mean([r['user_cpu_seconds'] for r in resources])),
            maximum_peak_rss_mib=max(r['peak_rss_mib'] for r in resources),
            maximum_rss_after_gc_mib=max(r['rss_after_gc_mib'] or 0. for r in resources)))
    save_json(out/'comparison_summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    evaluate_parser = sub.add_parser('eval')
    evaluate_parser.add_argument('--run', required=True)
    evaluate_parser.add_argument('--workers', type=int, default=8)
    evaluate_parser.add_argument('--seeds-per-stratum', type=int, default=64)
    evaluate_parser.add_argument('--pilot', action='store_true')
    evaluate_parser.add_argument('--episode-config')
    evaluate_parser.add_argument('--prepare-only', action='store_true')
    evaluate_parser.add_argument('--progress-every', type=int, default=32)
    startup = evaluate_parser.add_mutually_exclusive_group(required=True)
    startup.add_argument('--fresh', action='store_true')
    startup.add_argument('--resume', action='store_true')
    for command in ('audit', 'summarize'):
        sub.add_parser(command).add_argument('--run', required=True)
    args = parser.parse_args()
    if args.mode == 'eval':
        require(args.workers > 0 and args.seeds_per_stratum > 0, 'Counts must be positive')
        evaluate(args)
    elif args.mode == 'audit':
        audit(run_path(args.run))
    else:
        summarize(run_path(args.run))


if __name__ == '__main__':
    main()

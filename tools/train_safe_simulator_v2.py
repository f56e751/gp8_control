"""Train/evaluate residual Safe PPO inside the native feature/simulator with v2 physical boundary control.

Run from the repository parent with ``python -m gp8_control.tools.train_safe_simulator_v2``.
Existing tracked sources, stashed models, and previous experiments are not edited.
"""
from __future__ import annotations

import os
for _thread_key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_thread_key] = '1'

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import copy
import gc
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import re
import resource
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INITIAL_SEEDS = (40, 41)
WARMUP_START = 4000000
TRAINING_START = 4100000
TRAINING_STRIDE = 100000
OBS_SIZE = 107


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f'Unsupported JSON value {type(item).__name__}')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=convert))
    temporary.replace(path)


def save_npz(path, **arrays):
    """Publish only complete archives, including the once-fitted scaler."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def run_paths(run):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}', run):
        raise ValueError('Use one plain run name, without path components')
    return ROOT/'runs'/run, ROOT/'checkpoints'/run


def read_config(args):
    from gp8_control.rl.safe_v2.env import default_config
    config = default_config()
    if args.episode_config:
        config.update(json.loads(Path(args.episode_config).read_text()))
    return config


def source_hashes():
    from gp8_control.tools.train_safe_simulator import source_hashes as frozen_v1_sources
    files = [ROOT/'tools/train_safe_simulator_v2.py', ROOT/'tools/supervise_safe_simulator_v2.py',
             *sorted((ROOT/'rl/safe_v2').rglob('*.py'))]
    return dict(frozen_v1_sources(), **{str(path.relative_to(ROOT)):sha(path) for path in files if path.is_file()})


def worker(job):
    worker_started=time.monotonic()
    initial_usage=resource.getrusage(resource.RUSAGE_SELF)
    import torch
    from gp8_control.rl.safe_v2.ppo import SafePPO
    from gp8_control.rl.safe_v2.env import collect_episode, collect_baseline_episode
    torch.set_num_threads(1)
    state = job.get('state')
    policy = None
    if state is not None:
        policy = SafePPO(0, obs_size=int(state['obs_size']))
        policy.restore_rollout_state(state)
    # Independent reproducible exploration for every full episode, independent
    # of worker scheduling or the number of parallel workers.
    torch.manual_seed((17*int(job['scene_seed']) + 987654 + 100003*int(job.get('model', 0))) % (2**63))
    if job.get('name') == 'baseline':
        result = collect_baseline_episode(scene_seed=int(job['scene_seed']), config=job['config'])
    else:
        result = collect_episode(scene_seed=int(job['scene_seed']), policy=policy,
                                 config=job['config'], deterministic=bool(job.get('deterministic', False)))
    if not isinstance(result, dict) or 'rows' not in result or 'summary' not in result:
        raise TypeError('collect_episode must return rows and summary')
    # Native runner/core/backend cycles retain large MuJoCo buffers unless
    # collected after the env function and its local references have returned.
    gc.collect()
    usage=resource.getrusage(resource.RUSAGE_SELF)
    rss_mib=None
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                rss_mib=float(line.split()[1])/1024.
                break
    except OSError:
        pass
    result['worker_resources']=dict(pid=os.getpid(),wall_seconds=time.monotonic()-worker_started,
        user_cpu_seconds=usage.ru_utime-initial_usage.ru_utime,
        system_cpu_seconds=usage.ru_stime-initial_usage.ru_stime,
        peak_rss_mib=usage.ru_maxrss/1024.,rss_after_gc_mib=rss_mib)
    return result


def collect(executor, jobs):
    futures = {executor.submit(worker, job): index for index, job in enumerate(jobs)}
    results = {}
    for future in as_completed(futures):
        index = futures[future]
        try:
            results[index] = future.result()
        except BaseException as exc:
            raise RuntimeError(f"Episode scene_seed={jobs[index]['scene_seed']} failed") from exc
    return [results[index] for index in range(len(jobs))]


def parameter_digest(policy):
    digest = hashlib.sha256()
    for key, tensor in policy.network.state_dict().items():
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def metadata(args, label, update, decisions, protocol_path):
    return dict(run=args.run, model=label, initialization_seed=INITIAL_SEEDS[label],
                update=update, decisions=decisions, full_episodes=update*args.episodes,
                observation_size=OBS_SIZE, protocol_sha256=sha(protocol_path),
                initialization='fresh body, zero residual heads, fresh optimizer')


def check_metadata(meta, args, label, protocol_path):
    if (meta.get('run') != args.run or meta.get('model') != label
            or meta.get('initialization_seed') != INITIAL_SEEDS[label]
            or meta.get('protocol_sha256') != sha(protocol_path)):
        raise RuntimeError('Checkpoint belongs to a different run, model, or frozen protocol')
    return meta


def assert_on_policy(policy, rows):
    import torch
    with torch.no_grad():
        obs = torch.as_tensor(np.asarray([row['obs'] for row in rows]), dtype=torch.float32)
        raw = np.asarray([row['raw_obs'] for row in rows])
        np.testing.assert_allclose(policy.scaler.normalize(raw), obs.numpy(), rtol=0, atol=1e-6)
        heads = torch.as_tensor([row['head'] for row in rows], dtype=torch.long)
        latent = torch.as_tensor(np.asarray([row['latent'] for row in rows]), dtype=torch.float32)
        distribution, values = policy.network(obs, heads)
        action = torch.tanh(latent)
        logp = (distribution.log_prob(latent)-torch.log(1-action.square()+1e-6)).sum(-1).numpy()
    logp_error = float(np.max(np.abs(logp-np.asarray([row['logp'] for row in rows]))))
    value_error = float(np.max(np.abs(values.numpy()-np.asarray([row['value'] for row in rows]))))
    if max(logp_error, value_error) > 1e-4:
        raise RuntimeError(f'Collector behavior mismatch: logp={logp_error}, value={value_error}')
    return dict(max_logp_error=logp_error, max_value_error=value_error)


def save_batch(path, rows):
    keys = ('obs', 'raw_obs', 'action', 'latent', 'head', 'logp', 'value',
            'reward', 'cost', 'duration', 'episode_id')
    arrays = {key: np.asarray([row[key] for row in rows]) for key in keys}
    arrays['terminated'] = np.asarray([row.get('terminated', row.get('done', False)) for row in rows], bool)
    arrays['truncated'] = np.asarray([row.get('truncated', False) for row in rows], bool)
    arrays['next_value'] = np.asarray([row.get('next_value', [0., 0.]) for row in rows], np.float32)
    arrays['phase'] = np.asarray([str(row.get('phase', row.get('role',row['head']))) for row in rows])
    arrays['role'] = np.asarray([str(row.get('role',row.get('phase','unknown'))) for row in rows])
    for key in ('contacts','impulse_ns','impulse','collateral_falls','rejected','proximity_integral'):
        if all(key in row for row in rows):
            arrays[key]=np.asarray([row[key] for row in rows])
    save_npz(path, **arrays)


def write_history(out, history):
    save_json(out/'updates.json', history)
    if history:
        fields = list(history[0])
        with (out/'updates.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(history)


def calibrated_definition(plain, calibration_path):
    definition=copy.deepcopy(plain)
    if Path(calibration_path).is_file():
        calibration=json.loads(Path(calibration_path).read_text())
        definition['algorithm'].update(cost_budget_per_offered=calibration['cost_budget_per_offered'],
            original_baseline_warmup_mean_cost_per_offered=calibration['baseline_mean_cost_per_offered'],
            frozen_return_scales=calibration['return_scales'])
        definition['calibration_sha256']=sha(calibration_path)
    return definition


def train(args):
    from gp8_control.rl.safe_v2.ppo import FixedScaler, SafePPO, warmup_return_scales
    from gp8_control.rl.safe_v2.env import REWARD, COST
    out, ckpt = run_paths(args.run)
    if args.fresh and any(path.exists() and any(path.iterdir()) for path in (out, ckpt)):
        raise RuntimeError('Fresh run already contains artifacts; choose a new name or explicitly resume')
    config = read_config(args)
    warmup_start=int(getattr(args,'warmup_seed_start',WARMUP_START))
    training_start=int(getattr(args,'training_seed_start',TRAINING_START))
    training_stride=int(getattr(args,'training_seed_stride',TRAINING_STRIDE))
    if training_stride < args.updates*args.episodes:
        raise ValueError('Training model seed ranges must not overlap')
    plain_definition = dict(run=args.run, version=2, backend='native feature/simulator physical boundaries',
        previous_checkpoint_loaded=False, model_initialization_seeds=list(INITIAL_SEEDS),
        observation_size=OBS_SIZE, action_size=6, episode_config=config,
        roles=['throw_loaded','throw_return','push_approach','push_return'],
        training=dict(updates=args.updates, episodes_per_update=args.episodes,
            warmup_episodes=args.warmup_episodes,
            warmup_scene_seeds=list(range(warmup_start, warmup_start+args.warmup_episodes)),
            training_scene_seeds={str(label):list(range(training_start+label*training_stride,
                training_start+label*training_stride+args.updates*args.episodes)) for label in (0,1)},
            final_checkpoint='fixed final update, no evaluation-based selection'),
        evaluation=dict(regression_scene_seeds=list(range(3400000,3400020)),
                        fresh_scene_seeds=list(range(3500000,3500020))),
        algorithm=dict(name='boundary PPO-Lagrangian, separate actor/reward-critic/cost-critic bodies',
            actor_heads=['throw','push'], gamma_per_second=.99, gae_lambda=.95,
            epochs_max=6, minibatch_size=128, actor_learning_rate=.0002, critic_learning_rate=.0002,
            constraint='mean full episode total_cost/offered, including pre-policy intervals and zero-decision episodes',
            cost_budget_rule='fraction of original baseline warmup mean total_cost/offered',
            cost_budget_fraction=args.budget_fraction, multiplier_initial=1., multiplier_rate=.5,
            multiplier_denominator_floor=.1, multiplier_bounds=[0.,30.],
            critic_loss='mean squared original-unit prediction error divided by frozen per-head return scale squared',
            critic_scale_rule='max(1,std(v2-zero warmup GAE targets)); no later refitting',
            actor_gradient_clip=.5, critic_gradient_clip=.5,
            actor_optimizer_separate=True, normalization='actual107D v2-zero warmup; std floor0.1, frozen',
            termination='finite offered workload resolves or observed time budget; unfinished outcomes penalized',
            truncation='actual continuation value required for artificial boundaries'),
        reward=dict(REWARD,
            unfinished_unmanipulated=-.5, unfinished_manipulated=-2.,
            unfinished_rule='only offered IDs without existing outcome, no duplicate charge',physics_nonfinite=-10.),
        cost=dict(COST,proximity=0.,includes_environment_contacts=True),
        measurement=dict(source='MuJoCo solver states/forces at every physics step',
            contact_force_threshold_n=.1, same_pair_event_gap_seconds=.03,
            denominator='all fixed offered objects; policy decision count is not the constraint denominator'),
        residual_geometry=dict(max_joint_residual_rad=float(config['residual_scale']), editable_joints=[1,2,3],
            action='symmetric/skew basis coefficients,6 active dimensions per physical boundary',
            velocity_guard='changed250Hz samples satisfyM1, unchanged native overspeed preserved',
            timing_and_handoff='record exact protection/retiming from frozen v2 adapter sources'),
        source_sha256=source_hashes())
    calibration_path = out/'calibration.json'
    protocol_path = out/'protocol.json'
    definition = calibrated_definition(plain_definition, calibration_path)
    if args.resume:
        if not protocol_path.is_file():
            raise RuntimeError('Resume needs an existing same-run protocol')
        saved = json.loads(protocol_path.read_text())
        if saved != definition:
            if not (out/'committed_update.json').exists() and saved == plain_definition:
                save_json(protocol_path, definition)
            else:
                raise RuntimeError('Resume requires identical frozen sources, budget and protocol')
    out.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)
    if not protocol_path.exists():
        save_json(protocol_path, definition)
    for folder in ('batches', 'batch_stats', 'episode_summaries'):
        (out/folder).mkdir(exist_ok=True)
    begun = time.monotonic()
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn')) as executor:
        scaler_path = out/'scaler.npz'
        if scaler_path.exists():
            with np.load(scaler_path) as data:
                scaler = FixedScaler(OBS_SIZE)
                scaler.restore(dict(mean=data['mean'],std=data['std'],count=int(data['count']),minimum_std=.1,frozen=True))
            calibration = json.loads(calibration_path.read_text())
        else:
            zero_jobs=[dict(scene_seed=warmup_start+i,config=config,state=None,name='v2_zero')
                       for i in range(args.warmup_episodes)]
            baseline_jobs=[dict(scene_seed=warmup_start+i,config=config,state=None,name='baseline')
                           for i in range(args.warmup_episodes)]
            collected_warmup=collect(executor,zero_jobs+baseline_jobs)
            results=collected_warmup[:args.warmup_episodes]
            baseline_results=collected_warmup[args.warmup_episodes:]
            warmup_rows=[row for result in results for row in result['rows']]
            observations=[row['raw_obs'] for row in warmup_rows]
            if not observations:
                raise RuntimeError('Warmup produced no actual physical boundary decisions')
            scaler=FixedScaler(OBS_SIZE).fit(observations)
            return_scales=warmup_return_scales(warmup_rows)
            raw_path=out/'warmup_observations.npz'
            save_npz(raw_path,raw_obs=np.asarray(observations,np.float32),
                episode_id=np.asarray([str(row['episode_id']) for row in warmup_rows]),
                scene_seed=np.asarray([job['scene_seed'] for job,result in zip(zero_jobs,results) for row in result['rows']],np.int64),
                head=np.asarray([row['head'] for row in warmup_rows],np.int8))
            save_batch(out/'warmup_rows.npz',warmup_rows)
            baseline_costs=[float(r['summary']['total_cost'])/int(r['summary']['offered']) for r in baseline_results]
            calibration=dict(return_scales=return_scales.tolist(),
                baseline_mean_cost_per_offered=float(np.mean(baseline_costs)),
                cost_budget_per_offered=args.budget_fraction*float(np.mean(baseline_costs)),
                cost_budget_fraction=args.budget_fraction,
                source='original baseline full-episode solver-measured cost, including intervals before first policy decision',
                units='weighted safety cost per offered object; no division by decision count')
            save_json(out/'warmup.json',dict(observations=scaler.count,
                scene_seeds=[job['scene_seed'] for job in zero_jobs],observations_sha256=sha(raw_path),
                rows_sha256=sha(out/'warmup_rows.npz'),summaries=[r['summary'] for r in results],
                original_baseline_summaries=[r['summary'] for r in baseline_results],
                worker_resources=[r.get('worker_resources') for r in collected_warmup]))
            save_json(calibration_path,calibration)
            save_npz(scaler_path,mean=scaler.mean,std=scaler.std,count=scaler.count)
            print(f'WARMUP complete: {scaler.count} observations, budget{calibration["cost_budget_per_offered"]:.6f}/offered',flush=True)
        final_definition=calibrated_definition(plain_definition,calibration_path)
        if json.loads(protocol_path.read_text()) != final_definition:
            if (out/'committed_update.json').exists():
                raise RuntimeError('Cannot change calibrated protocol after initial checkpoints')
            save_json(protocol_path,final_definition)
        definition=final_definition
        budget=float(calibration['cost_budget_per_offered'])
        policies=[SafePPO(seed,scaler,obs_size=OBS_SIZE,return_scales=calibration['return_scales']) for seed in INITIAL_SEEDS]
        committed = out/'committed_update.json'
        initialize = args.fresh or not committed.is_file()
        if initialize:
            if args.resume and (any((out/'batch_stats').glob('*.json'))
                    or any(ckpt.glob('safe_seed?.pt'))
                    or any(path.name[-6:-3] != '000' for path in ckpt.glob('safe_seed?_update*.pt'))):
                raise RuntimeError('Uncommitted initialization contains trained artifacts; refuse recovery')
            for label, policy in enumerate(policies):
                policy.save(ckpt/f'safe_seed{label}_update000.pt', metadata(args, label, 0, 0, protocol_path))
            save_json(out/'initialization.json', dict(from_scratch=True, previous_checkpoint_loaded=False,
                models=[dict(model=label, initialization_seed=INITIAL_SEEDS[label], parameter_sha256=parameter_digest(policy),
                    initial_checkpoint_sha256=sha(ckpt/f'safe_seed{label}_update000.pt'),
                    updates=0, optimizer_steps=0, actor_adam_state_entries=len(policy.actor_optimizer.state),
                    critic_adam_state_entries=len(policy.critic_optimizer.state)) for label, policy in enumerate(policies)]))
            common, decisions, history = 0, [0, 0], []
            save_json(out/'committed_update.json', dict(update=0))
        else:
            common = int(json.loads(committed.read_text())['update'])
            decisions = []
            for label, policy in enumerate(policies):
                meta = check_metadata(policy.load(ckpt/f'safe_seed{label}_update{common:03d}.pt'), args, label, protocol_path)
                if int(meta['update']) != common:
                    raise RuntimeError('Checkpoint update does not match commit marker')
                decisions.append(int(meta['decisions']))
            history = [json.loads(path.read_text()) for path in sorted((out/'batch_stats').glob('*.json'))
                       if int(json.loads(path.read_text())['update']) <= common]
        for update in range(common+1, args.updates+1):
            jobs = []
            for label, policy in enumerate(policies):
                state = policy.rollout_state()
                for offset in range(args.episodes):
                    index = (update-1)*args.episodes+offset
                    jobs.append(dict(scene_seed=training_start+label*training_stride+index,
                                     state=state, model=label, config=config))
            collection_started=time.monotonic()
            collected = collect(executor, jobs)
            collection_seconds=time.monotonic()-collection_started
            for label, policy in enumerate(policies):
                episodes = collected[label*args.episodes:(label+1)*args.episodes]
                rows = [row for episode in episodes for row in episode['rows']]
                if not rows:
                    raise RuntimeError('No actual trajectory correction decisions in training batch')
                behavior = assert_on_policy(policy, rows)
                # Computing GAE in update checks complete chronology and required bootstraps.
                stats = policy.update(rows, [episode['summary'] for episode in episodes], budget)
                decisions[label] += len(rows)
                stats.update(model=label, initialization_seed=INITIAL_SEEDS[label], update=update,
                    cumulative_decisions=decisions[label], full_episodes=update*args.episodes,
                    heads=dict(Counter(str(row['head']) for row in rows)),
                    phases=dict(Counter(str(row.get('phase', row['head'])) for row in rows)),
                    scene_seeds=[job['scene_seed'] for job in jobs[label*args.episodes:(label+1)*args.episodes]],
                    on_policy_verification=behavior, elapsed_seconds=time.monotonic()-begun,
                    collection_wall_seconds=collection_seconds,
                    worker_resources=[episode.get('worker_resources') for episode in episodes])
                save_batch(out/'batches'/f'seed{label}_update{update:03d}.npz', rows)
                save_json(out/'episode_summaries'/f'seed{label}_update{update:03d}.json', [ep['summary'] for ep in episodes])
                save_json(out/'batch_stats'/f'update{update:03d}_seed{label}.json', stats)
                meta = metadata(args, label, update, decisions[label], protocol_path)
                policy.save(ckpt/f'safe_seed{label}_update{update:03d}.pt', meta)
                latest = ckpt/f'safe_seed{label}.pt'
                temporary = latest.with_suffix('.pt.tmp')
                policy.save(temporary, meta)
                temporary.replace(latest)
                history.append(stats)
                print(json.dumps({key:stats[key] for key in ('model', 'update', 'cumulative_decisions',
                    'mean_reward', 'cost_per_offered', 'cost_budget_per_offered', 'multiplier', 'heads', 'elapsed_seconds')}), flush=True)
            write_history(out, history)
            save_json(out/'committed_update.json', dict(update=update))
    save_json(out/'training_summary.json', dict(training_complete=True, from_scratch=True,
        previous_checkpoint_loaded=False, initialization_seeds=list(INITIAL_SEEDS),
        updates_per_model=args.updates, episodes_per_model=args.updates*args.episodes,
        cost_budget_per_offered=budget, frozen_return_scales=calibration['return_scales'],
        seed_results=[row for row in history if row['update'] == args.updates],
        checkpoint_paths=[str(ckpt/f'safe_seed{label}.pt') for label in (0, 1)], wall_seconds=time.monotonic()-begun))
    print('TRAINING complete', flush=True)


def scene_seed_list(text):
    values=[]
    for token in text.split(','):
        if ':' in token:
            start,stop=map(int,token.split(':'))
            values.extend(range(start,stop))
        else:
            values.append(int(token))
    if not values or len(values)!=len(set(values)):
        raise argparse.ArgumentTypeError('Use unique scene seeds or comma-separated half-open ranges')
    return values


def evaluate(args):
    from gp8_control.rl.safe_v2.ppo import SafePPO
    out, _ = run_paths(args.run)
    model_out, ckpt = run_paths(args.model_run or args.run)
    protocol_path = model_out/'protocol.json'
    trained = json.loads(protocol_path.read_text())
    if not (model_out/'training_summary.json').is_file():
        raise RuntimeError('Evaluate the fixed completed training run only')
    if trained['source_sha256'] != source_hashes():
        raise RuntimeError('Training source changed before evaluation; investigate instead of silently changing the problem')
    config = trained['episode_config']
    if args.episode_config and config != read_config(args):
        raise RuntimeError('Evaluation config differs from frozen training environment')
    states, checkpoint_hashes = {}, {}
    if args.policy in ('safe', 'compare'):
        for label in (0, 1):
            policy = SafePPO(INITIAL_SEEDS[label], obs_size=OBS_SIZE)
            path = ckpt/f'safe_seed{label}.pt'
            meta = policy.load(path)
            if (meta['run'] != trained['run'] or meta['update'] != trained['training']['updates']
                    or meta['protocol_sha256'] != sha(protocol_path)):
                raise RuntimeError('Evaluation checkpoint is not the fixed last trained model')
            states[f'safe_seed{label}'] = policy.rollout_state()
            checkpoint_hashes[str(path.relative_to(ROOT))] = sha(path)
    if args.policy in ('zero','compare'):
        states=dict(v2_zero=None,**states)
    if args.policy in ('baseline','compare'):
        states=dict(baseline=None,**states)
    used_training = set(trained['training']['warmup_scene_seeds'])
    for seeds in trained['training']['training_scene_seeds'].values():
        used_training.update(seeds)
    if used_training.intersection(args.scene_seeds):
        raise RuntimeError('Evaluation scene seeds overlap training or scaler warmup')
    definition = dict(run=args.run, model_run=trained['run'], scene_seeds=args.scene_seeds,
        episode_config=config, checkpoint_sha256=checkpoint_hashes, training_protocol_sha256=sha(protocol_path),
        policies=list(states), deterministic=True, source_sha256=source_hashes(),
        scene_suites={str(scene):('fresh' if scene in trained['evaluation']['fresh_scene_seeds']
                                 else 'regression' if scene in trained['evaluation']['regression_scene_seeds'] else 'custom')
                      for scene in args.scene_seeds},
        selection='fixed final model; no evaluation-based tuning')
    target = out/'evaluation_protocol.json'
    if target.exists() and json.loads(target.read_text()) != definition:
        raise RuntimeError('Evaluation protocol already exists with different settings')
    out.mkdir(parents=True, exist_ok=True)
    save_json(target, definition)
    result_dir = out/'evaluation'
    result_dir.mkdir(exist_ok=True)
    jobs, paths = [], []
    for name, state in states.items():
        for scene in args.scene_seeds:
            path = result_dir/f'{name}_scene{scene}.json'
            if path.exists():
                previous = json.loads(path.read_text())
                if previous.get('evaluation_protocol_sha256') != sha(target):
                    raise RuntimeError('Existing result belongs to another evaluation protocol')
                continue
            jobs.append(dict(scene_seed=scene, state=state, model=int(name[-1]) if name.startswith('safe_seed') else 0,
                             config=config, deterministic=True, name=name))
            paths.append(path)
    started = time.monotonic()
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn')) as executor:
        futures = {executor.submit(worker, job): (job, path) for job, path in zip(jobs, paths)}
        for future in as_completed(futures):
            job, path = futures[future]
            result = future.result()
            save_json(path, dict(policy=job['name'], scene_seed=job['scene_seed'],
                evaluation_protocol_sha256=sha(target), summary=result['summary'],
                trace=result.get('trace'),
                decisions=[{key:row[key] for key in ('raw_obs','obs','action','latent','logp','head','value',
                    'reward','cost','duration','contacts','impulse_ns','impulse','collateral_falls',
                    'role','phase','episode_id','terminated','truncated','reward_components','cost_components') if key in row}
                           for row in result['rows']]))
            print(json.dumps(dict(policy=job['name'], scene_seed=job['scene_seed'],
                                  elapsed_seconds=time.monotonic()-started)), flush=True)
    save_json(out/'evaluation_summary.json', dict(complete=True, scene_count=len(args.scene_seeds),
        policy_count=len(states), result_count=len(args.scene_seeds)*len(states),
        evaluation_protocol_sha256=sha(target), wall_seconds=time.monotonic()-started))
    print('EVALUATION complete', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    training = sub.add_parser('train')
    training.add_argument('--run', required=True)
    training.add_argument('--workers', type=int, default=8)
    training.add_argument('--updates', type=int, default=128)
    training.add_argument('--episodes', type=int, default=12)
    training.add_argument('--warmup-episodes', type=int, default=16)
    training.add_argument('--budget-fraction', type=float, default=.8)
    training.add_argument('--warmup-seed-start',type=int,default=WARMUP_START)
    training.add_argument('--training-seed-start',type=int,default=TRAINING_START)
    training.add_argument('--training-seed-stride',type=int,default=TRAINING_STRIDE)
    training.add_argument('--episode-config')
    startup = training.add_mutually_exclusive_group(required=True)
    startup.add_argument('--fresh', action='store_true')
    startup.add_argument('--resume', action='store_true')
    evaluation = sub.add_parser('eval')
    evaluation.add_argument('--run', required=True)
    evaluation.add_argument('--model-run')
    evaluation.add_argument('--workers', type=int, default=8)
    evaluation.add_argument('--scene-seeds', type=scene_seed_list, default=scene_seed_list('3400000:3400020,3500000:3500020'))
    evaluation.add_argument('--policy', choices=('baseline', 'zero', 'safe', 'compare'), default='compare')
    evaluation.add_argument('--episode-config')
    args = parser.parse_args()
    if args.workers < 1 or (args.mode == 'train' and (args.updates < 1 or args.episodes < 1 or args.warmup_episodes < 1 or not 0 < args.budget_fraction <= 1)):
        parser.error('Counts and workers must be positive; budget fraction must be in(0,1]')
    if args.mode == 'train':
        train(args)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()

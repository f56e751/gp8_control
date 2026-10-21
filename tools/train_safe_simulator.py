"""Train/evaluate residual Safe PPO inside the native feature/simulator backend.

Run from the repository parent with ``python -m gp8_control.tools.train_safe_simulator``.
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
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import re
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INITIAL_SEEDS = (30, 31)
WARMUP_START = 3000000
TRAINING_START = 3100000
TRAINING_STRIDE = 100000
OBS_SIZE = 100


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
    from gp8_control.rl.safe_simulator_env import default_config
    config = default_config()
    if args.episode_config:
        config.update(json.loads(Path(args.episode_config).read_text()))
    return config


def source_hashes():
    files = [ROOT/'tools/train_safe_simulator.py', ROOT/'rl/safe_ppo.py',
             ROOT/'rl/safe_simulator_env.py', ROOT/'rl/safe_residual_adapter.py',
             ROOT/'rl/safe_contact_monitor.py',
             ROOT/'rl/sim_runner.py', ROOT/'rl/common.py', ROOT/'config.py']
    for folder in ('skills', 'planning', 'backends', 'robots', 'trajectory',
                   'perception', 'tracking', 'utils'):
        files.extend(sorted((ROOT/folder).rglob('*.py')))
        files.extend(sorted((ROOT/folder).rglob('*.xml')))
    for folder, patterns in (('sim/recycling_mujoco', ('*.xml', '*.stl')),
                             ('model', ('*.py', '*.pt')),
                             ('config', ('*.yaml', '*.json'))):
        for pattern in patterns:
            files.extend(sorted((ROOT/folder).rglob(pattern)))
    return {str(path.relative_to(ROOT)): sha(path) for path in files if path.is_file()}


def worker(job):
    import torch
    from gp8_control.rl.safe_ppo import SafePPO
    from gp8_control.rl.safe_simulator_env import collect_episode
    torch.set_num_threads(1)
    state = job.get('state')
    policy = None
    if state is not None:
        policy = SafePPO(0, obs_size=int(state['obs_size']))
        policy.restore_rollout_state(state)
    # Independent reproducible exploration for every full episode, independent
    # of worker scheduling or the number of parallel workers.
    torch.manual_seed((17*int(job['scene_seed']) + 987654 + 100003*int(job.get('model', 0))) % (2**63))
    result = collect_episode(scene_seed=int(job['scene_seed']), policy=policy,
                             config=job['config'], deterministic=bool(job.get('deterministic', False)))
    if not isinstance(result, dict) or 'rows' not in result or 'summary' not in result:
        raise TypeError('collect_episode must return rows and summary')
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
    arrays['phase'] = np.asarray([str(row.get('phase', row['head'])) for row in rows])
    save_npz(path, **arrays)


def write_history(out, history):
    save_json(out/'updates.json', history)
    if history:
        fields = list(history[0])
        with (out/'updates.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(history)


def train(args):
    from gp8_control.rl.safe_ppo import FixedScaler, SafePPO
    out, ckpt = run_paths(args.run)
    if args.fresh and any(path.exists() and any(path.iterdir()) for path in (out, ckpt)):
        raise RuntimeError('Fresh run already contains artifacts; choose a new name or explicitly resume')
    config = read_config(args)
    definition = dict(run=args.run, backend='feature/simulator native SimCore',
        previous_checkpoint_loaded=False, model_initialization_seeds=list(INITIAL_SEEDS),
        observation_size=OBS_SIZE, episode_config=config,
        training=dict(updates=args.updates, episodes_per_update=args.episodes,
            warmup_episodes=args.warmup_episodes, warmup_scene_seeds=list(range(WARMUP_START, WARMUP_START+args.warmup_episodes)),
            training_scene_seeds={str(label):list(range(TRAINING_START+label*TRAINING_STRIDE,
                TRAINING_START+label*TRAINING_STRIDE+args.updates*args.episodes)) for label in (0, 1)},
            final_checkpoint='fixed final update, no evaluation-based selection'),
        algorithm=dict(name='chronological PPO-Lagrangian', actor_heads=['throw', 'push'],
            gamma_per_second=.99, gae_lambda=.95, epochs_max=6, minibatch_size=128,
            learning_rate=.0002, cost_limit_per_decision=args.cost_limit,
            normalization='new actual baseline observations, std floor0.1, frozen',
            termination='finite offered workload resolves or observed time budget expires; unfinished outcomes penalized',
            truncation='only artificial collection interruption, actual continuation value required'),
        reward=dict(correct_class_bin=4., attempted_failure=-2., unprocessed=-.5,
            seconds=-.02, unintended_contact_event=-.4, impulse_ns=-.1,
            neighbor_lateral_huber_4cm=-.2, push_pose_bonus_max=.5,
            projection_loss=-.1, action_mean_square=-.01,
            unfinished_unmanipulated=-.5, unfinished_manipulated=-2.,
            unfinished_rule='only offered IDs without an existing outcome; no duplicate charge',
            physics_nonfinite=-10.),
        cost=dict(unintended_contact_event=.15, impulse_ns=.05,
            collateral_fall=5., rejected_residual=2., physics_nonfinite=10.),
        measurement=dict(source='MuJoCo solver state and mj_contactForce at every physics step',
            contact_force_threshold_n=.1, same_pair_event_gap_seconds=.03,
            collateral_fall_depth_m=.08,
            intended_contacts='normal tool/active-target suction pickup, loaded Throw, and protected Push stroke exempt',
            next_throw_grasp_exemption='tool/declared next Throw target in final0.12s of return/transfer only',
            success='native correct class-bin outcome per fixed offered workload',
            denominator='all offered objects, including delayed, unadmitted, and unfinished objects'),
        residual_geometry=dict(max_joint_residual_rad=.08, editable_joints=[1, 2, 3],
            endpoint_guard_seconds=.02, throw_release_guard_seconds=.04,
            throw='independent pre-release and return bumps; pickup descent unchanged',
            push='route plus pre-stroke preparation and post-stroke transfer; actual stroke unchanged',
            push_stroke_guard_seconds=.04,
            velocity_guard='changed250Hz samples satisfyM1; unchanged native overspeed preserved, no exceedance worsened',
            handoff='protected position/velocity knots and final state retained'),
        source_sha256=source_hashes())
    protocol_path = out/'protocol.json'
    if args.resume:
        if not protocol_path.is_file() or json.loads(protocol_path.read_text()) != definition:
            raise RuntimeError('Resume requires an identical frozen protocol and source files')
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
                scaler.restore(dict(mean=data['mean'], std=data['std'], count=int(data['count']), minimum_std=.1, frozen=True))
        else:
            jobs = [dict(scene_seed=WARMUP_START+index, config=config, state=None) for index in range(args.warmup_episodes)]
            results = collect(executor, jobs)
            observations = [row['raw_obs'] for result in results for row in result['rows']]
            if not observations:
                raise RuntimeError('Warmup produced no actual policy decisions')
            scaler = FixedScaler(OBS_SIZE).fit(observations)
            raw_path = out/'warmup_observations.npz'
            save_npz(raw_path, raw_obs=np.asarray(observations, np.float32),
                episode_id=np.asarray([str(row.get('episode_id', job['scene_seed']))
                    for job, result in zip(jobs, results) for row in result['rows']]),
                scene_seed=np.asarray([job['scene_seed']
                    for job, result in zip(jobs, results) for row in result['rows']], np.int64),
                head=np.asarray([int(row.get('head', -1)) for result in results for row in result['rows']], np.int8))
            save_json(out/'warmup.json', dict(observations=scaler.count, scene_seeds=[job['scene_seed'] for job in jobs],
                observations_sha256=sha(raw_path),
                summaries=[result['summary'] for result in results]))
            # The scaler is the warmup commit marker: publish it last so a
            # startup interruption can repeat warmup without losing provenance.
            save_npz(scaler_path, mean=scaler.mean, std=scaler.std, count=scaler.count)
            print(f'WARMUP complete: {scaler.count} native simulator decision observations', flush=True)
        policies = [SafePPO(seed, scaler, obs_size=OBS_SIZE) for seed in INITIAL_SEEDS]
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
                    updates=0, optimizer_steps=0, adam_state_entries=len(policy.optimizer.state)) for label, policy in enumerate(policies)]))
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
                    jobs.append(dict(scene_seed=TRAINING_START+label*TRAINING_STRIDE+index,
                                     state=state, model=label, config=config))
            collected = collect(executor, jobs)
            for label, policy in enumerate(policies):
                episodes = collected[label*args.episodes:(label+1)*args.episodes]
                rows = [row for episode in episodes for row in episode['rows']]
                if not rows:
                    raise RuntimeError('No actual trajectory correction decisions in training batch')
                behavior = assert_on_policy(policy, rows)
                # Computing GAE in update checks complete chronology and required bootstraps.
                stats = policy.update(rows, args.cost_limit)
                decisions[label] += len(rows)
                stats.update(model=label, initialization_seed=INITIAL_SEEDS[label], update=update,
                    cumulative_decisions=decisions[label], full_episodes=update*args.episodes,
                    heads=dict(Counter(str(row['head']) for row in rows)),
                    phases=dict(Counter(str(row.get('phase', row['head'])) for row in rows)),
                    scene_seeds=[job['scene_seed'] for job in jobs[label*args.episodes:(label+1)*args.episodes]],
                    on_policy_verification=behavior, elapsed_seconds=time.monotonic()-begun)
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
                    'mean_reward', 'mean_cost', 'multiplier', 'heads', 'elapsed_seconds')}), flush=True)
            write_history(out, history)
            save_json(out/'committed_update.json', dict(update=update))
    save_json(out/'training_summary.json', dict(training_complete=True, from_scratch=True,
        previous_checkpoint_loaded=False, initialization_seeds=list(INITIAL_SEEDS),
        updates_per_model=args.updates, episodes_per_model=args.updates*args.episodes,
        seed_results=[row for row in history if row['update'] == args.updates],
        checkpoint_paths=[str(ckpt/f'safe_seed{label}.pt') for label in (0, 1)], wall_seconds=time.monotonic()-begun))
    print('TRAINING complete', flush=True)


def scene_seed_list(text):
    if ':' in text:
        first, stop = map(int, text.split(':'))
        values = list(range(first, stop))
    else:
        values = [int(value) for value in text.split(',')]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError('Use unique scene seeds, e.g. 3400000:3400020 or 1,2,3')
    return values


def evaluate(args):
    from gp8_control.rl.safe_ppo import SafePPO
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
    if args.policy in ('baseline', 'compare'):
        states = dict(baseline=None, **states)
    used_training = set(trained['training']['warmup_scene_seeds'])
    for seeds in trained['training']['training_scene_seeds'].values():
        used_training.update(seeds)
    if used_training.intersection(args.scene_seeds):
        raise RuntimeError('Evaluation scene seeds overlap training or scaler warmup')
    definition = dict(run=args.run, model_run=trained['run'], scene_seeds=args.scene_seeds,
        episode_config=config, checkpoint_sha256=checkpoint_hashes, training_protocol_sha256=sha(protocol_path),
        policies=list(states), deterministic=True, source_sha256=source_hashes(),
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
            jobs.append(dict(scene_seed=scene, state=state, model=0 if name == 'baseline' else int(name[-1]),
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
                decisions=[{key:row[key] for key in ('raw_obs', 'action', 'head', 'value') if key in row}
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
    training.add_argument('--updates', type=int, default=32)
    training.add_argument('--episodes', type=int, default=12)
    training.add_argument('--warmup-episodes', type=int, default=12)
    training.add_argument('--cost-limit', type=float, default=.10)
    training.add_argument('--episode-config')
    startup = training.add_mutually_exclusive_group(required=True)
    startup.add_argument('--fresh', action='store_true')
    startup.add_argument('--resume', action='store_true')
    evaluation = sub.add_parser('eval')
    evaluation.add_argument('--run', required=True)
    evaluation.add_argument('--model-run')
    evaluation.add_argument('--workers', type=int, default=8)
    evaluation.add_argument('--scene-seeds', type=scene_seed_list, default=scene_seed_list('3400000:3400020'))
    evaluation.add_argument('--policy', choices=('baseline', 'safe', 'compare'), default='compare')
    evaluation.add_argument('--episode-config')
    args = parser.parse_args()
    if args.workers < 1 or (args.mode == 'train' and (args.updates < 1 or args.episodes < 1 or args.warmup_episodes < 1 or args.cost_limit < 0)):
        parser.error('Counts and workers must be positive, cost limit nonnegative')
    if args.mode == 'train':
        train(args)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()

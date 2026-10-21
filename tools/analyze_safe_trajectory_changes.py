"""Measure same-state nominal versus learned250Hz command trajectories.

Replays fixed final policies; does not train or modify recorded evaluations.
Cartesian values are GP8 forward kinematics of commanded joint positions,
not physical tracking measurements or distances between independent rollouts.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp

import numpy as np

from gp8_control.tools.train_safe_simulator import ROOT, run_paths, save_json, save_npz, sha, source_hashes


def positions(robot, joints):
    """Batch the existing revolute PoE model; validated against GP8 FK below."""
    joints = np.asarray(joints, float)
    transforms = np.broadcast_to(np.eye(4), (len(joints), 4, 4)).copy()
    identity = np.eye(3)
    for index, screw in enumerate(robot._screws):
        w, v = screw[:3], screw[3:]
        skew = np.array([[0., -w[2], w[1]], [w[2], 0., -w[0]], [-w[1], w[0], 0.]])
        angle = joints[:, index, None, None]
        rotation = identity + np.sin(angle)*skew + (1.-np.cos(angle))*(skew@skew)
        exponential = np.broadcast_to(np.eye(4), transforms.shape).copy()
        exponential[:, :3, :3] = rotation
        exponential[:, :3, 3] = (identity-rotation) @ np.cross(w, v)
        transforms = transforms @ exponential
    return (transforms @ robot._M)[:, :3, 3]


def measure(job):
    from gp8_control.rl.safe_ppo import SafePPO
    from gp8_control.rl.safe_simulator_env import collect_episode
    from gp8_control.rl.safe_residual_adapter import SafeResidualAdapter, stream_samples
    from gp8_control.robots.gp8 import GP8

    robot = GP8()
    validation = np.random.default_rng(20261020).uniform(-1.5, 1.5, (32, 6))
    np.testing.assert_allclose(positions(robot, validation),
        np.asarray([robot.forward_kinematics(q)[:3, 3] for q in validation]), rtol=0, atol=2e-14)
    policy = SafePPO(job['initialization_seed'], obs_size=100)
    policy.load(job['checkpoint'])
    original = SafeResidualAdapter.transform
    records = []

    def transform(adapter, traj, vel, timestep, final_joint=None, *, release_index=None):
        corrected, corrected_vel, record = original(adapter, traj, vel, timestep,
                                                   final_joint, release_index=release_index)
        if record is not None and record.get('active'):
            grid, nominal = stream_samples(traj, vel, timestep, final_joint)
            _, executed = stream_samples(corrected, corrected_vel, timestep, final_joint)
            xyz_nominal = positions(robot, nominal)
            xyz_executed = positions(robot, executed)
            distance = np.linalg.norm(xyz_executed-xyz_nominal, axis=1)
            joint_difference = np.abs(executed-nominal)
            entry = dict(model=job['model'], scene_seed=job['scene_seed'], index=len(records),
                task=record['task'], phase=record['phase'], target_id=record['target_id'],
                active_samples=len(grid), duration=float(grid[-1]),
                knot_max_joint_degrees=float(np.rad2deg(record['max_joint_delta'])),
                stream_max_joint_degrees=float(np.rad2deg(joint_difference.max())),
                end_effector_max_cm=float(100.*distance.max()),
                end_effector_rms_cm=float(100.*np.sqrt(np.mean(distance**2))),
                end_effector_mean_cm=float(100.*distance.mean()),
                projection=record['projection'], rejected=record['rejected'],
                protected_position_equal=record['protected_position_equal'],
                protected_velocity_equal=record['protected_velocity_equal'],
                sample_nominal_sha256=None)
            records.append(entry)
            # Save all small reference curves for a reproducible plot later.
            curve_path = job['output']/f"model{job['model']}_scene{job['scene_seed']}_decision{entry['index']:02d}.npz"
            save_npz(curve_path, time=grid, nominal_q=nominal, executed_q=executed,
                     nominal_xyz=xyz_nominal, executed_xyz=xyz_executed)
            entry['curve'] = str(curve_path.relative_to(ROOT))
            entry['curve_sha256'] = sha(curve_path)
        return corrected, corrected_vel, record

    SafeResidualAdapter.transform = transform
    try:
        result = collect_episode(job['scene_seed'], policy, job['config'], deterministic=True)
    finally:
        SafeResidualAdapter.transform = original
    reference = json.loads(job['reference'].read_text())
    actual = {k:v for k,v in result['summary'].items() if k != 'wall_seconds'}
    expected = {k:v for k,v in reference['summary'].items() if k != 'wall_seconds'}
    if actual != expected or json.loads(json.dumps(result['trace'])) != reference['trace']:
        raise AssertionError('Instrumentation changed the saved physical rollout')
    return dict(model=job['model'], scene_seed=job['scene_seed'], records=records, physical_trace_exact=True)


def describe(records, field):
    values = np.asarray([r[field] for r in records], float)
    return dict(count=len(values), mean=float(values.mean()), median=float(np.median(values)),
                percentile95=float(np.percentile(values, 95)), maximum=float(values.max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='feature_safe_20261020_v1')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    out, checkpoints = run_paths(args.run)
    protocol = json.loads((out/'protocol.json').read_text())
    evaluation = json.loads((out/'evaluation_protocol.json').read_text())
    if protocol['source_sha256'] != source_hashes():
        raise RuntimeError('Frozen simulator sources changed')
    output = out/'trajectory_changes'
    curves = output/'curves'
    curves.mkdir(parents=True, exist_ok=True)
    jobs = [dict(model=label, initialization_seed=protocol['model_initialization_seeds'][label],
                 scene_seed=scene, checkpoint=checkpoints/f'safe_seed{label}.pt',
                 config=protocol['episode_config'], output=curves,
                 reference=out/'evaluation'/f'safe_seed{label}_scene{scene}.json')
            for label in (0, 1) for scene in evaluation['scene_seeds']]
    results = []
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn')) as executor:
        futures = [executor.submit(measure, job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
            print(f'REPLAY VERIFIED {len(results)}/{len(jobs)}', flush=True)
    results.sort(key=lambda r:(r['model'], r['scene_seed']))
    records = [record for result in results for record in result['records']]
    groups = dict(all=records, throw=[r for r in records if r['task']=='throw'],
                  push=[r for r in records if r['task']=='push'],
                  model0=[r for r in records if r['model']==0], model1=[r for r in records if r['model']==1])
    fields = ('knot_max_joint_degrees', 'stream_max_joint_degrees', 'end_effector_max_cm',
              'end_effector_rms_cm', 'end_effector_mean_cm')
    summary = dict(run=args.run, physical_rollouts_match=True, replay_count=len(results),
        definitions=dict(comparison='same-state nominal command versus projected learned command for each actual residual decision',
            cartesian='GP8 forward kinematics of250Hz commanded joints; not measured tracking and not ballistic object flight',
            statistics='equal weight per NN decision; maximum then average/median across decisions',
            fk_validation='32 random joint vectors compared to native GP8.forward_kinematics at2e-14m absolute tolerance'),
        groups={name:{field:describe(group, field) for field in fields} for name,group in groups.items()},
        records=records, protocol_sha256=sha(out/'protocol.json'),
        evaluation_protocol_sha256=sha(out/'evaluation_protocol.json'))
    save_json(output/'summary.json', summary)
    print(json.dumps({name:values['end_effector_max_cm'] for name,values in summary['groups'].items()}), flush=True)


if __name__ == '__main__':
    main()

"""Verify v2 replays and measure exact250Hz same-state commanded TCP changes.

Final mode requires the frozen final checkpoints and completed recorded
evaluations. Pilot mode uses a separate constant actor and is explicitly
excluded from final policy statistics. Neither mode changes runtime sources.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import gc
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from gp8_control.tools.analyze_safe_trajectory_changes import positions
from gp8_control.tools.train_safe_simulator_v2 import ROOT, run_paths, save_json, save_npz, sha, source_hashes


ROLES = ('throw_loaded', 'throw_return', 'push_approach', 'push_return')
DECISION_KEYS = ('raw_obs','obs','action','latent','logp','head','value', 'reward','cost',
    'duration','contacts','impulse_ns','impulse','collateral_falls','role','phase',
    'episode_id','terminated','truncated','reward_components','cost_components')


class ConstantActor:
    def __init__(self, amplitude):
        self.amplitude = float(amplitude)

    def act(self, observation, head, deterministic=False):
        action = np.full(6, self.amplitude, np.float32)
        return action, np.arctanh(action), 0., np.zeros(2), np.asarray(observation).copy()


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, default=json_value)


def decision_projection(rows):
    return [{key:row[key] for key in DECISION_KEYS if key in row} for row in rows]


def describe(records, field):
    values = np.asarray([row[field] for row in records], float)
    if not len(values):
        return dict(count=0, mean=None, median=None, percentile95=None, maximum=None)
    return dict(count=len(values), mean=float(values.mean()), median=float(np.median(values)),
                percentile95=float(np.percentile(values,95)), maximum=float(values.max()))


def curve_rank(curve):
    return (curve['metadata']['end_effector_max_cm'], -curve['metadata']['model'],
            -curve['metadata']['scene_seed'], -curve['metadata']['index'])


def artifact_path(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def draw_overlays(curves,output,pilot):
    """Four illustrative role maxima; these panels are not average paths."""
    os.environ.setdefault('MPLCONFIGDIR','/tmp/gp8_safe_v2_matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig=plt.figure(figsize=(13,10),facecolor='white')
    names=dict(throw_loaded='Throw: loaded motion',throw_return='Throw: return / next Throw approach',
               push_approach='Push: owned approach',push_return='Push: return')
    for panel,role in enumerate(ROLES,1):
        ax=fig.add_subplot(2,2,panel,projection='3d')
        curve=curves.get(role)
        if curve is None:
            ax.set_title(names[role]+'\nNo eligible decision recorded',fontsize=10)
            ax.set_axis_off()
            continue
        arrays,meta=curve['arrays'],curve['metadata']
        nominal,executed=arrays['nominal_xyz'],arrays['executed_xyz']
        ax.plot(*nominal.T,color='#697386',linestyle='--',linewidth=1.8,label='Native nominal command')
        ax.plot(*executed.T,color='#00896c',linewidth=2.2,label='Executed policy command')
        ax.scatter(*executed[0],color='#273343',marker='o',s=23)
        ax.scatter(*executed[-1],color='#273343',marker='s',s=23)
        index=int(np.argmax(arrays['distance_m']))
        connector=np.vstack((nominal[index],executed[index]))
        ax.plot(*connector.T,color='#e78c27',linewidth=2.1)
        ax.scatter(*executed[index],color='#e78c27',s=22)
        midpoint=connector.mean(axis=0)
        ax.text(*midpoint,f"  {meta['end_effector_max_cm']:.2f} cm",fontsize=8,color='#9b4e00',
                bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=.3))
        ax.set_title(names[role]+f"\nmodel {meta['model']} | scene {meta['scene_seed']} | decision {meta['index']}",
                     fontsize=10,pad=10)
        points=np.vstack((nominal,executed))
        low,high=points.min(axis=0),points.max(axis=0)
        spread=np.maximum(high-low,.05)
        margin=np.maximum(.08*spread,.01)
        ax.set_xlim(low[0]-margin[0],high[0]+margin[0])
        ax.set_ylim(low[1]-margin[1],high[1]+margin[1])
        ax.set_zlim(low[2]-margin[2],high[2]+margin[2])
        ax.set_box_aspect(high-low+2*margin)
        ax.set_xlabel('Base X (m)',fontsize=8,labelpad=5)
        ax.set_ylabel('Base Y (m)',fontsize=8,labelpad=5)
        ax.set_zlabel('Base Z (m)',fontsize=8,labelpad=5)
        ax.tick_params(labelsize=7,pad=1)
        for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
            axis.set_major_locator(MaxNLocator(nbins=3))
        ax.view_init(elev=25,azim=-58)
        if panel==1:
            ax.legend(loc='upper left',fontsize=7,frameon=True)
    title=('PILOT constant actor: commanded trajectory examples' if pilot else
           'Final learned policies: commanded trajectory examples')
    fig.suptitle(title+'\nEach panel selects the largest displacement within that role.',fontsize=14,y=.98)
    fig.text(.5,.025,'Illustrative maxima, not average paths. Overall statistics are reported separately in '
             +('pilot_summary.json.' if pilot else 'summary.json.'),ha='center',fontsize=9,color='#444b55')
    fig.subplots_adjust(left=.02,right=.98,bottom=.08,top=.88,wspace=.04,hspace=.26)
    path=output/'trajectory_overlays.png'
    fig.savefig(path,dpi=180,facecolor='white')
    plt.close(fig)
    return dict(path=artifact_path(path),sha256=sha(path),
                selection='Within-role maximum displacement examples; all-decision averages are separate.')


def policy_for(job):
    if job.get('constant_action') is not None:
        return ConstantActor(job['constant_action'])
    from gp8_control.rl.safe_v2.ppo import SafePPO
    policy = SafePPO(job['initialization_seed'], obs_size=107)
    metadata = policy.load(job['checkpoint'])
    if (metadata.get('run') != job['model_run'] or metadata.get('model') != job['model']
            or metadata.get('update') != job['final_update']
            or metadata.get('protocol_sha256') != job['protocol_sha256']):
        raise RuntimeError('Analysis requires the recorded fixed final checkpoint')
    return policy


def measure(job):
    """One replay; the interception reads arrays and never mutates records."""
    from gp8_control.rl.safe_v2.adapter import SafeV2Adapter
    from gp8_control.rl.safe_v2.env import collect_episode
    from gp8_control.robots.gp8 import GP8

    robot = GP8()
    validation = np.random.default_rng(20261021).uniform(-1.5,1.5,(32,6))
    np.testing.assert_allclose(positions(robot,validation),
        np.asarray([robot.forward_kinematics(q)[:3,3] for q in validation]),rtol=0,atol=2e-14)
    policy = policy_for(job)
    original = SafeV2Adapter._decide
    records, best_curves = [], {}

    def decide(adapter, frame, window):
        before = frame['executed'].copy()
        original(adapter,frame,window)
        record = adapter.records[-1]
        mask = ((frame['grid'] >= window['trigger_time'] - 1e-12)
                & (frame['grid'] <= window['phase_end'] + 1e-12))
        if not np.any(mask):
            raise AssertionError('A recorded decision has no physical phase samples')
        grid = frame['grid'][mask].copy()
        nominal = frame['nominal'][mask].copy()
        prior = before[mask].copy()
        executed = frame['executed'][mask].copy()
        # Earlier decisions must not contaminate this phase's reference path.
        if not np.array_equal(prior,nominal):
            raise AssertionError('Policy ownership windows overlap in physical commands')
        nominal_xyz, executed_xyz = positions(robot,nominal), positions(robot,executed)
        distance = np.linalg.norm(executed_xyz-nominal_xyz,axis=1)
        joint = np.abs(executed-nominal)
        entry = dict(model=job['model'],scene_seed=job['scene_seed'],index=len(records),
            scene_suite=job.get('scene_suite','pilot' if job.get('constant_action') is not None else 'custom'),
            role=record['phase'],task=record['task'],head=int(record['head']),
            target_id=record['target_id'],physical_task=record['physical_task'],
            physical_target_id=record['physical_target_id'],owner_source=record['owner_source'],
            actual_boundary_time=float(record['actual_boundary_time']),
            phase_end_time=float(record['phase_end_time']),duration=float(record['duration']),
            samples=len(grid),nonzero=bool(record['max_joint_delta']>1e-10),
            stream_max_joint_radians=float(joint.max()),stream_max_joint_degrees=float(np.degrees(joint.max())),
            end_effector_max_cm=float(100.*distance.max()),
            end_effector_mean_cm=float(100.*distance.mean()),
            end_effector_rms_cm=float(100.*np.sqrt(np.mean(distance**2))),
            projection=float(record['projection']),rejected=bool(record['rejected']),
            joint_projection=record['joint_projection'],static_projection=float(record['static_projection']))
        records.append(entry)
        role = entry['role']
        if role not in best_curves or curve_rank({'metadata':entry}) > curve_rank(best_curves[role]):
            raw = np.asarray(record['observation'],float)
            target_position, target_size = raw[12:15],raw[15:18]
            neighbors = raw[18:63].reshape(5,9)
            valid = raw[-5:] > .5
            arrays = dict(time=grid,time_from_decision=grid-window['trigger_time'],
                simulation_time=grid+frame['started'],nominal_q=nominal,before_q=prior,executed_q=executed,
                nominal_xyz=nominal_xyz,executed_xyz=executed_xyz,distance_m=distance,
                owner_position=target_position,owner_half_size=target_size,
                neighbor_positions=(neighbors[valid,:3]+target_position),
                neighbor_half_sizes=neighbors[valid,3:6],
                goal_q6=np.asarray(record['goal_q6'],float),goal_xyz=np.asarray(record['goal_xyz'],float))
            best_curves[role] = dict(metadata=entry,arrays=arrays)

    SafeV2Adapter._decide = decide
    try:
        result = collect_episode(job['scene_seed'],policy,job['config'],deterministic=True)
    finally:
        SafeV2Adapter._decide = original
    reference = json.loads(Path(job['reference']).read_text())
    actual_summary = {k:v for k,v in result['summary'].items() if k!='wall_seconds'}
    expected_summary = {k:v for k,v in reference['summary'].items() if k!='wall_seconds'}
    if canonical(actual_summary) != canonical(expected_summary):
        raise AssertionError(f"Replay summary differs from recorded evaluation: {job['reference']}")
    if canonical(result['trace']) != canonical(reference['trace']):
        raise AssertionError(f"Instrumentation changed the physical/actor trace: {job['reference']}")
    if canonical(decision_projection(result['rows'])) != canonical(reference['decisions']):
        raise AssertionError('Replay on-policy decision/reward/cost records differ')
    attempted = {r['target_id'] for r in result['trace']['executions'] if r['task']=='push' and r['target_id'] is not None}
    owned = {r['target_id'] for r in records if r['role']=='push_approach'}
    nonzero_owned = {r['target_id'] for r in records if r['role']=='push_approach' and r['nonzero']}
    answer = dict(model=job['model'],scene_seed=job['scene_seed'],records=records,best_curves=best_curves,
        physical_trace_exact=True,decision_records_exact=True,reference_sha256=sha(job['reference']),
        push_unique_attempted=len(attempted),push_unique_owned=len(attempted & owned),
        push_unique_nonzero_owned=len(attempted & nonzero_owned))
    del result
    gc.collect()
    return answer


def summarize(run,results,curves,output,provenance,pilot=False):
    results.sort(key=lambda r:(r['model'],r['scene_seed']))
    records = [record for result in results for record in result['records']]
    groups = dict(all=records)
    groups.update({role:[r for r in records if r['role']==role] for role in ROLES})
    groups.update({task:[r for r in records if r['task']==task] for task in ('throw','push')})
    groups.update({f'model{model}':[r for r in records if r['model']==model]
                   for model in sorted({r['model'] for r in records})})
    groups.update({f'model{model}_{role}':[r for r in records if r['model']==model and r['role']==role]
                   for model in sorted({r['model'] for r in records}) for role in ROLES})
    for suite in sorted({r['scene_suite'] for r in records}):
        groups[f'suite_{suite}']=[r for r in records if r['scene_suite']==suite]
        for role in ROLES:
            groups[f'suite_{suite}_{role}']=[r for r in records if r['scene_suite']==suite and r['role']==role]
    fields = ('stream_max_joint_degrees','end_effector_max_cm','end_effector_mean_cm','end_effector_rms_cm','projection')
    group_stats = {}
    for name,group in groups.items():
        group_stats[name] = {field:describe(group,field) for field in fields}
        group_stats[name].update(nonzero_decisions=sum(r['nonzero'] for r in group),
            nonzero_percent=(100.*sum(r['nonzero'] for r in group)/len(group) if group else None))
    overlays = {}
    for role,curve in curves.items():
        path = output/f'overlay_{role}.npz'
        save_npz(path,**curve['arrays'],metadata_json=np.asarray(canonical(curve['metadata'])))
        overlays[role] = dict(path=artifact_path(path),sha256=sha(path),**curve['metadata'],
            selection='largest executed TCP displacement within role; illustration only, not checkpoint selection')
    attempted = sum(r['push_unique_attempted'] for r in results)
    owned = sum(r['push_unique_owned'] for r in results)
    nonzero = sum(r['push_unique_nonzero_owned'] for r in results)
    summary = dict(run=run,kind='constant_actor_pilot' if pilot else 'fixed_final_policy_analysis',
        pilot_excluded_from_final_statistics=bool(pilot),physical_rollouts_match=True,
        replay_count=len(results),groups=group_stats,
        push_approach_control=dict(unique_push_targets_attempted=attempted,
            unique_targets_with_owned_approach=owned,owned_approach_percent=100.*owned/max(attempted,1),
            unique_targets_with_nonzero_owned_approach=nonzero,
            nonzero_owned_target_percent=100.*nonzero/max(attempted,1),
            decision_nonzero_percent=group_stats['push_approach']['nonzero_percent']),
        definitions=dict(comparison='Same physical state and native nominal250Hz commands versus projected commands of this one actual decision phase.',
            cartesian='Native GP8 FK at every250Hz commanded point; not measured tracking, independent rollout difference, or object flight.',
            decision_statistics='Equal weight per actual NN decision, including zero/rejected decisions; mean of per-decision maxima is not a time-weighted mean.',
            end_effector_mean_cm='Average distance over all native samples within each decision physical phase; then averaged equally across decisions.',
            reference='Every replay summary except wall time, full physical/actor trace and stored decision/reward/cost rows match exactly.',
            fk_validation='32 random q vectors compared with native GP8.forward_kinematics at2e-14m absolute tolerance.',
            overlays='Four role examples selected transparently by maximum within-role displacement for illustration; no learned checkpoint selection.'),
        records=records,replays=[{k:v for k,v in result.items() if k!='records'} for result in results],
        overlays=overlays,provenance=provenance)
    summary['overlay_figure']=draw_overlays(curves,output,pilot)
    save_json(output/('pilot_summary.json' if pilot else 'summary.json'),summary)
    csv_fields = [key for key in records[0] if key not in ('joint_projection',)] if records else []
    if records:
        with (output/'records.csv').open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=csv_fields)
            writer.writeheader();writer.writerows({key:r[key] for key in csv_fields} for r in records)
    return summary


def pilot(args):
    from gp8_control.rl.safe_v2.env import collect_episode,default_config
    output = ROOT/'runs'/'feature_route_trajectory_analysis_pilot_20261021'
    if args.output:
        output=Path(args.output).expanduser().resolve()
    if (output/'summary.json').exists():
        raise RuntimeError('Keep constant-actor pilot outputs separate from final policy analysis')
    output.mkdir(parents=True,exist_ok=True)
    config=dict(default_config(),offer_count=args.pilot_offers,time_limit=80.)
    reference=collect_episode(args.pilot_scene,ConstantActor(args.pilot_action),config,deterministic=True)
    reference_path=output/'constant_reference.json'
    save_json(reference_path,dict(summary=reference['summary'],trace=reference['trace'],
                                decisions=decision_projection(reference['rows'])))
    del reference
    gc.collect()
    result=measure(dict(model=0,scene_seed=args.pilot_scene,constant_action=args.pilot_action,
                        config=config,reference=reference_path))
    curves=result.pop('best_curves')
    summary=summarize('constant_actor_pilot',[result],curves,output,
        dict(constant_action=args.pilot_action,reference_sha256=sha(reference_path),
             analyzer_sha256=sha(Path(__file__))),pilot=True)
    print(json.dumps(dict(output=str(output),physical_trace_exact=True,
        groups={role:summary['groups'][role]['end_effector_max_cm'] for role in ROLES},
        push_approach_control=summary['push_approach_control'])),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='feature_safe_route_20261021_v2')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--output',help='Separate derived-output directory; evaluation files are never changed.')
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--pilot-action',type=float,default=.4)
    parser.add_argument('--pilot-scene',type=int,default=4300002)
    parser.add_argument('--pilot-offers',type=int,default=6)
    args=parser.parse_args()
    if args.workers<1 or args.pilot_offers<1 or not np.isfinite(args.pilot_action) or abs(args.pilot_action)>=1:
        parser.error('Positive counts and a finite constant action strictly between-1and1 are required')
    if args.pilot:
        return pilot(args)
    out,checkpoints=run_paths(args.run)
    protocol_path,evaluation_path=out/'protocol.json',out/'evaluation_protocol.json'
    if not (out/'training_summary.json').is_file() or not (out/'evaluation_summary.json').is_file():
        raise RuntimeError('Complete final training and recorded evaluation before trajectory analysis')
    protocol,evaluation=json.loads(protocol_path.read_text()),json.loads(evaluation_path.read_text())
    if protocol['source_sha256']!=source_hashes():
        raise RuntimeError('Frozen simulator/training sources changed')
    model_out,checkpoints=run_paths(evaluation.get('model_run',args.run))
    if model_out!=out:
        raise RuntimeError('Run analysis from the completed model/evaluation run itself')
    output=Path(args.output).expanduser().resolve() if args.output else out/'trajectory_changes_v2'
    if output==out or output==out/'evaluation':
        raise RuntimeError('Use a separate derived-output folder')
    if (output/'pilot_summary.json').exists():
        raise RuntimeError('Keep final policy analysis separate from constant-actor pilot outputs')
    output.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for label in (0,1):
        checkpoint=checkpoints/f'safe_seed{label}.pt'
        expected=evaluation['checkpoint_sha256'].get(str(checkpoint.relative_to(ROOT)))
        if expected!=sha(checkpoint):
            raise RuntimeError('Checkpoint differs from the recorded final evaluation')
        for scene in evaluation['scene_seeds']:
            reference=out/'evaluation'/f'safe_seed{label}_scene{scene}.json'
            saved=json.loads(reference.read_text())
            if saved['evaluation_protocol_sha256']!=sha(evaluation_path):
                raise RuntimeError('Recorded scene belongs to another evaluation protocol')
            jobs.append(dict(model=label,initialization_seed=protocol['model_initialization_seeds'][label],
                model_run=protocol['run'],final_update=protocol['training']['updates'],
                protocol_sha256=sha(protocol_path),checkpoint=checkpoint,
                scene_seed=scene,scene_suite=evaluation.get('scene_suites',{}).get(str(scene),'custom'),
                config=protocol['episode_config'],reference=reference))
    results,curves=[],{}
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn')) as executor:
        futures=[executor.submit(measure,job) for job in jobs]
        for future in as_completed(futures):
            result=future.result()
            best=result.pop('best_curves')
            for role,curve in best.items():
                if role not in curves or curve_rank(curve)>curve_rank(curves[role]):
                    curves[role]=curve
            results.append(result)
            print(f'REPLAY VERIFIED {len(results)}/{len(jobs)}',flush=True)
    if protocol['source_sha256']!=source_hashes():
        raise RuntimeError('Frozen sources changed during replay analysis')
    summary=summarize(args.run,results,curves,output,dict(protocol_sha256=sha(protocol_path),
        evaluation_protocol_sha256=sha(evaluation_path),analyzer_sha256=sha(Path(__file__)),
        checkpoint_sha256=evaluation['checkpoint_sha256']))
    print(json.dumps(dict(output=str(output),replays=len(results),
        overall=summary['groups']['all']['end_effector_max_cm'],
        push_approach_control=summary['push_approach_control'])),flush=True)


if __name__=='__main__':main()

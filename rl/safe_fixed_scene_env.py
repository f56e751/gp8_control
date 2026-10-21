"""Frozen-v2 transfer evaluation on one fixed, initially ungrasped scene.

The belt, perception, selector, skills and guards remain native. Objects are
placed once, upstream inside the native camera view, then move on the belt.
No arrivals occur. A common, policy-free perception prelude precedes exactly
one native selected skill. Safety is observed to action_start + 15 seconds,
including after target resolution. This is not the legacy held-Throw reset.
"""
from __future__ import annotations

from collections import Counter
from contextlib import redirect_stdout, nullcontext
import hashlib
import io
import json
import time

import mujoco
import numpy as np

from gp8_control.config import Config
from gp8_control.backends.mujoco_sim import _PHYSICAL_BELT_BASE_X, _PHYSICAL_BELT_HALF_X
from gp8_control.rl.safe_simulator_env import EpisodeStopped, NativeResidualRunner
from gp8_control.rl.safe_residual_adapter import SafeResidualAdapter as OriginalPhaseAdapter
from gp8_control.rl.safe_v2.adapter import SafeV2Adapter
from gp8_control.rl.safe_v2.env import RouteCore, RouteTransitions, default_config as route_config
from gp8_control.rl.safe_v2.monitor import command_static_guard


def default_config():
    config = route_config()
    config.update(time_limit=120., prelude_limit=40., action_horizon=15., selection_target_y=.58,
                  version='fixed_reset_native_full_skill_v2', quiet=True)
    return config


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, default=_json)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _overlap(first, second, gap=.008):
    # Scene boxes have zero yaw. Physical separation, not outcomes, is the only
    # acceptance criterion during construction.
    return all(abs(first[axis]-second[axis]) < first['size'][k]+second['size'][k]+gap
               for k,axis in enumerate(('x','y')))


def _on_belt(row,margin=.008):
    return (_PHYSICAL_BELT_BASE_X-_PHYSICAL_BELT_HALF_X+margin <= row['x']-row['size'][0]
            and row['x']+row['size'][0] <= _PHYSICAL_BELT_BASE_X+_PHYSICAL_BELT_HALF_X-margin)


def make_scene(scene_seed, skill, neighbor_count, config=None):
    if skill not in ('throw','push') or neighbor_count not in (2,5):
        raise ValueError('Use throw/push and exactly 2 or 5 neighbors')
    rng = np.random.default_rng(int(scene_seed))
    # Push needs the native full .2m backswing with x>=.24. This declared
    # geometry follows that existing gate; execution/veto are never bypassed.
    target_x=(.47,.49) if skill=='push' else (.36,.44)
    target = dict(id=1, time=0., x=float(rng.uniform(*target_x)),
        y=float(rng.uniform(2.20,2.24)), size=[float(rng.uniform(.030,.043)),
        float(rng.uniform(.045,.060)), .015], yaw=0.,
        class_name='transparent' if skill=='throw' else 'metal', suction_p=1.)
    # Every neighbor is upstream of the designated target. The first row
    # includes both lateral sides; the second row adds crowding for n=5.
    if skill=='push':
        anchors=([(-.105,.13),(.075,.13)] if neighbor_count==2 else
                 [(-.115,.13),(.075,.13),(0.,.26),(-.115,.39),(.075,.39)])
    else:
        anchors = [(-.105,.13),(.105,.13)] if neighbor_count==2 else [
            (-.115,.13),(0.,.16),(.115,.13),(-.075,.285),(.075,.285)]
    rows=[target]
    for index,(dx,dy) in enumerate(anchors,2):
        for _ in range(1000):
            row=dict(id=index,time=0.,x=float(target['x']+dx+rng.uniform(-.012,.012)),
                y=float(target['y']+dy+rng.uniform(-.012,.012)),
                size=[float(rng.uniform(.025,.040)),float(rng.uniform(.028,.043)),.015],
                yaw=0.,class_name='metal' if index%2==0 else 'transparent',suction_p=1.)
            if _on_belt(row) and all(not _overlap(row,other) for other in rows):
                rows.append(row)
                break
        else:
            raise RuntimeError('Could not construct the predeclared nonoverlapping layout')
    return dict(scene_seed=int(scene_seed),requested_skill=skill,neighbor_count=neighbor_count,
                target_id=1,coordinate_frame='robot base',rows=rows,
                layout_version='native_belt_supported_push_backswing_v2',
                construction='Seeded dimensions/jitter; only initial AABB separation and full belt support with8mm margin can reject a sample; no outcome filtering.')


def _queue_state(runner):
    return [dict(id=getattr(o,'sim_object_id',None),class_name=o.class_name,
        grasp=np.asarray(o.T_grasp_base).tolist(),aim=np.asarray(o.T_aim_base).tolist(),
        detect_time=float(o.detect_time),encoder_distance_m=getattr(o,'encoder_distance_m',None),
        suction_p=float(getattr(o,'suction_p',1.))) for o in runner.queue._objects]


def state_payload(runner):
    core=runner.core
    mask=mujoco.mjtState.mjSTATE_INTEGRATION
    integration=np.empty(mujoco.mj_stateSize(core.model,mask))
    mujoco.mj_getState(core.model,core.data,integration,mask)
    model_fields=('geom_size','geom_rbound','geom_friction','geom_contype','geom_conaffinity',
                  'body_mass','body_inertia','body_gravcomp','eq_data','actuator_gainprm','actuator_biasprm')
    return dict(integration=integration.tolist(),model={key:np.asarray(getattr(core.model,key)).tolist()
        for key in model_fields},box_state=list(core._box_state),box_serial=list(core._box_serial),
        box_class=list(core._box_class),box_suction_p=list(core._box_suction_p),
        box_half_size=[np.asarray(x).tolist() for x in core._box_half_size],
        box_belt_y=list(core._box_belt_y),manipulated=list(core._box_manipulated),
        manipulated_skill=list(core._box_manipulated_skill),box_sealed=list(core._box_sealed),
        reward_done=list(core._box_reward_done),cause_step=list(core._box_cause_step),
        grabbed=core._grabbed,vacuum_on=core._vacuum_on,pressing=core._pressing,
        suction_attempt_ok=core._suction_attempt_ok,ctrl_target=core._ctrl_target.tolist(),
        pending_ctrl=list(core._pending_ctrl),encoder_distance=float(core._encoder_distance),
        belt_distance=float(runner.conveyor.distance_m),belt_speed=float(runner.conveyor.current),
        rng=core._rng.bit_generator.state,suction_rng=core._suction_rng.bit_generator.state,
        queue=_queue_state(runner),latest_snapshot=core.latest_snapshot)


class FixedSceneCore(RouteCore):
    def __init__(self,cfg,scene,config):
        self.fixed_scene=scene
        self.fixed_action_deadline=None
        super().__init__(cfg,scene['rows'],config)

    def _admit_offers(self):
        return None

    def place_scene(self):
        if self.safe_active or any(state!=self._FREE for state in self._box_state):
            raise RuntimeError('Fixed layouts require a fresh inactive core after robot startup')
        for slot,row in enumerate(self.fixed_scene['rows']):
            self._activate_box(slot)
            self._apply_box_size(slot,row['size'])
            joint=self._box_joints[slot]
            joint.qpos[:3]=self._to_world([row['x'],row['y'],
                self._belt_top_w-self._base_p[2]+row['size'][2]])
            joint.qpos[3:7]=self._yaw_quat(row['yaw'])
            joint.qvel[:]=0.
            self._box_state[slot]=self._ON_BELT
            self._box_class[slot]=row['class_name']
            self._box_suction_p[slot]=row['suction_p']
            self._box_belt_y[slot]=float(joint.qpos[1])
            self.model.geom_rgba[self._box_geom_ids[slot]]=self._box_rgba(row['class_name'],row['suction_p'])
            self._box_serial[slot]=row['id']
            self.safe_admissions.append(dict(**row,actual_time=0.))
        self._spawn_count=len(self.fixed_scene['rows'])
        self.safe_arrival_index=len(self.safe_tape)
        mujoco.mj_forward(self.model,self.data)
        self.latest_snapshot=self._build_snapshot(self.sim_time())
        self._last_det_pub=self.sim_time()

    def _step_once_locked(self):
        if (self.safe_active and self.fixed_action_deadline is not None
                and self.sim_time()>=self.fixed_action_deadline-1e-9):
            raise EpisodeStopped('fixed_action_horizon')
        super()._step_once_locked()


class FixedTransitions(RouteTransitions):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.decision_contexts=[]

    def decide(self,core,observation,head,role):
        grabbed=getattr(core,'_grabbed',None)
        held_id=None if grabbed is None else int(core._box_serial[int(grabbed)])
        result=super().decide(core,observation,head,role)
        self.decision_contexts.append(dict(time=core.sim_time(),role=role,head=int(head),
            holding=grabbed is not None,held_object_id=held_id,
            observation_holding_feature=float(self.pending['raw_obs'][-1])))
        return result


def _end_objects(core):
    states=[]
    for row in core.safe_tape:
        slot=core._box_serial.index(row['id'])
        states.append(dict(id=row['id'],class_name=row['class_name'],state=core._box_state[slot],
            qpos=core._box_joints[slot].qpos.copy().tolist(),
            qvel=core._box_joints[slot].qvel.copy().tolist(),
            manipulated=bool(core._box_manipulated[slot]),
            outcome=next((o for o in core.safe_outcomes if o['sim_object_id']==row['id']),None),
            physical_resolution=core.safe_resolved.get(row['id'])))
    return states


def _phase_stats(events):
    return {phase:dict(count=len(group),impulse_ns=float(sum(e['impulse'] for e in group)),
        peak_force_n=max((float(e['peak_force']) for e in group),default=0.))
        for phase in sorted({e['phase'] for e in events})
        for group in [[e for e in events if e['phase']==phase]]}


def _collect(scene_seed,skill,neighbor_count,policy,config,original_baseline,deterministic):
    begun=time.monotonic()
    config=dict(default_config(),**config)
    for key in ('time_limit','prelude_limit','action_horizon'):
        if not np.isfinite(config[key]) or config[key]<=0:
            raise ValueError('Finite positive scene deadlines are required')
    if config['prelude_limit']+config['action_horizon']>config['time_limit']:
        raise ValueError('Overall deadline must cover the full prelude and action horizon')
    if float(config['belt_speed'])!=.12 or not config['physical_belt']:
        raise ValueError('Fixed-scene transfer keeps the native physical belt at .12m/s')
    if int(config['pool_size'])<neighbor_count+1:
        raise ValueError('Object pool cannot hold the full fixed layout')
    scene=make_scene(scene_seed,skill,neighbor_count,config)
    cfg=Config();cfg.PICK_LOG_CSV=''
    runner=NativeResidualRunner(cfg=cfg,sim_seed=int(scene_seed),realtime=False,
        max_objects=int(config['max_objects']),core_factory=lambda c:FixedSceneCore(c,scene,config))
    runner.truth_tracks=bool(config['truth_tracks'])
    core=runner.core;core.safe_runner=runner
    collector=None if original_baseline else FixedTransitions(runner,policy,f'fixed_{skill}_{neighbor_count}_{scene_seed}',deterministic)
    adapter=(OriginalPhaseAdapter(policy=None,deterministic=True) if original_baseline else
        SafeV2Adapter(policy=None,deterministic=deterministic,decision_callback=collector.decide,
            scale=float(config['residual_scale']),static_guard=command_static_guard))
    runner.safe_adapter=adapter
    core.safe_collector=collector
    commands=[];command_times=[];io_events=[]
    original_emit=runner.traj_ctrl._emit_sample
    original_io=runner.traj_ctrl._set_suction
    def emit(q):
        command_times.append(core.sim_time());commands.append(np.asarray(q,float).copy().tolist())
        return original_emit(q)
    def suction(on,requested_at):
        io_events.append(dict(on=bool(on),requested_at=float(requested_at),actual_time=core.sim_time()))
        return original_io(on,requested_at)
    reset_state=action_state=action_start_objects=None
    selection=None;action_start=None;skill_end=None;prelude_monitor=None
    reason='fixed_action_horizon';steps=0
    adapter_bound=False
    try:
        runner.start()
        runner.traj_ctrl._emit_sample=emit
        runner.traj_ctrl._set_suction=suction
        core.place_scene()
        core.begin_episode()
        reset_state=state_payload(runner)
        # No policy or skill executes during this common camera/tracking prelude.
        while core.sim_time()-core.safe_started < config['prelude_limit']-1e-9:
            runner.ingest()
            # Native intercepts can be scheduled many seconds ahead. Define a
            # common geometric prelude endpoint before asking the unchanged
            # selector, so the 15-second measurement contains the execution.
            # This gate is fixed before evaluation and never uses outcomes.
            target_slot=core._box_serial.index(scene['target_id'])
            target_y=float(core._to_base(core._box_joints[target_slot].qpos[:3])[1])
            if core._box_state[target_slot]!=core._FREE and target_y<=float(config['selection_target_y']):
                action=runner.action_from_indices(*runner.app_heuristic_action_indices(runner.action_mask()))
                if action is not None:
                    break
            core.advance_seconds(float(runner.cfg.TIME_STEP));steps+=1
        else:
            action=None
        action_start=core.sim_time()
        prelude_monitor=core.safe_monitor.snapshot()
        action_state=state_payload(runner)
        action_start_objects=_end_objects(core)
        selection=dict(target_id=None if action is None else getattr(action.target,'sim_object_id',None),
            skill=None if action is None else action.skill_name,
            intended_target_id=scene['target_id'],requested_skill=skill,
            sim_time=action_start,queue_order=[getattr(o,'sim_object_id',None) for o in runner.ordered_objects()])
        core.fixed_action_deadline=action_start+float(config['action_horizon'])
        adapter.bind(runner);adapter_bound=True
        if action is not None:
            # Native step executes the pending action, uses no next-chain
            # target, and retains its own intercept/veto/reward/prune handling.
            runner._pending_action=action
            core.step_index=0
            runner.step(None)
        skill_end=core.sim_time()
        # Resolution is telemetry only; it must never shorten risk observation.
        while core.sim_time()<core.fixed_action_deadline-1e-9:
            core.advance_seconds(min(.04,core.fixed_action_deadline-core.sim_time()))
    except EpisodeStopped as stopped:
        reason=stopped.reason
    finally:
        with core.lock:
            core._recycle_boxes()
        core.safe_active=False
        if collector is not None:
            collector.finalize(done=True,reason=reason)
        final_state=state_payload(runner)
        end_objects=_end_objects(core)
        if adapter_bound:
            adapter.unbind()
        runner.traj_ctrl._emit_sample=original_emit
        runner.traj_ctrl._set_suction=original_io
        runner.close()
    if reset_state is None or action_state is None or selection is None:
        raise RuntimeError('Scene stopped before its comparable action-start state')
    target_outcome=next((o for o in core.safe_outcomes if o['sim_object_id']==scene['target_id']),None)
    intended=selection['target_id']==scene['target_id'] and selection['skill']==skill
    correct=bool(intended and skill_end is not None and reason!='physics_nonfinite'
                 and target_outcome is not None and target_outcome['actual_bin_name'] is not None
                 and target_outcome['actual_bin_name']==target_outcome['class_bin_name'])
    monitor=core.safe_monitor.snapshot()
    prelude_events=[e for e in core.safe_monitor.events if e['start_time']<action_start-1e-9]
    main_events=[e for e in core.safe_monitor.events if e['start_time']>=action_start-1e-9]
    main_falls=[e for e in core.safe_monitor.falls if e['sim_time']>=action_start-1e-9]
    records=adapter.records
    contexts=[] if collector is None else collector.decision_contexts
    loaded=[c for c in contexts if c['role']=='throw_loaded']
    initial_hash=digest(action_state);selection_hash=digest(selection)
    command_hash=digest(commands);timeline_hash=digest(dict(times=command_times,commands=commands))
    summary=dict(scene_seed=int(scene_seed),requested_skill=skill,neighbor_count=neighbor_count,
        scene_sha256=digest(scene),offered=neighbor_count+1,admitted=len(core.safe_admissions),
        additional_arrivals=0,intended_target_id=scene['target_id'],selected_target_id=selection['target_id'],
        action_start_object_count=sum(o['state']!=core._FREE for o in action_start_objects),
        action_start_neighbor_count=sum(o['state']!=core._FREE and o['id']!=scene['target_id'] for o in action_start_objects),
        action_start_neighbors_on_belt=sum(o['state']==core._ON_BELT and o['id']!=scene['target_id'] for o in action_start_objects),
        selected_skill=selection['skill'],intended_target_selected=intended,
        protocol_issue=None if intended else 'intended_target_not_selected',
        correct_bin=int(correct),target_bin_success=correct,target_success=correct,task_success_percent=100.*correct,
        target_outcome=target_outcome,target_physical_resolved=scene['target_id'] in core.safe_resolved,
        target_unresolved=target_outcome is None,actions=len(runner.safe_executions),
        attempted_skill=bool(runner.safe_executions),
        skill_completed=skill_end is not None,initially_grasped=reset_state['grabbed'] is not None,
        prelude_seconds=action_start-core.safe_started,action_horizon=float(config['action_horizon']),
        observed_action_seconds=core.sim_time()-action_start,observation_duration=core.sim_time()-action_start,
        action_horizon_completed=abs(core.sim_time()-action_start-float(config['action_horizon']))<core._dt+1e-8,
        skill_seconds=None if skill_end is None else skill_end-action_start,
        sim_seconds=core.sim_time()-core.safe_started,wall_seconds=time.monotonic()-begun,
        termination_reason=reason,contacts=int(monitor['contact_events']),impulse_ns=float(monitor['impulse']),
        object_contacts=sum(e['kind']!='robot_environment' for e in main_events),
        environment_contacts=sum(e['kind']=='robot_environment' for e in main_events),
        action_object_contacts=sum(e['kind']!='robot_environment' for e in main_events),
        action_environment_contacts=sum(e['kind']=='robot_environment' for e in main_events),
        total_object_contacts=int(monitor['object_contact_events']),
        total_environment_contacts=int(monitor['environment_contact_events']),
        preparation_contacts=sum(e['phase']=='push_prepare' for e in main_events),
        collateral_falls=int(monitor['falls']),proximity_integral=float(monitor['proximity_integral']),
        minimum_clearance_m=float(monitor['minimum_clearance_m']),
        prelude_contacts=len(prelude_events),prelude_impulse_ns=float(prelude_monitor['impulse']),
        prelude_collateral_falls=int(prelude_monitor['falls']),prelude_proximity_integral=float(prelude_monitor['proximity_integral']),
        action_contacts=len(main_events),action_impulse_ns=float(monitor['impulse']-prelude_monitor['impulse']),
        action_collateral_falls=len(main_falls),
        action_proximity_integral=float(monitor['proximity_integral']-prelude_monitor['proximity_integral']),
        event_start_phases=_phase_stats(core.safe_monitor.events),action_event_start_phases=_phase_stats(main_events),
        throw_loaded_and_return_contact_start_count=sum(e['phase'] in ('throw_loaded','throw_return') for e in main_events),
        throw_loaded_decisions=len(loaded),throw_loaded_decisions_holding=sum(c['holding'] for c in loaded),
        policy_decisions=len(contexts),decision_phases=dict(Counter(c['role'] for c in contexts)),
        nonzero_decisions=sum(float(r.get('max_joint_delta',0.))>1e-10 for r in records),
        rejected_decisions=sum(bool(r.get('rejected',False)) for r in records),
        initial_state_sha256=initial_hash,selection_sha256=selection_hash,
        command_sha256=command_hash,command_timeline_sha256=timeline_hash,
        io_sha256=digest(io_events),final_state_sha256=digest(final_state),
        physical_end_objects_sha256=digest(end_objects))
    trace=dict(scene=scene,admissions=core.safe_admissions,selection=selection,
        reset_state_sha256=digest(reset_state),action_start_state_sha256=initial_hash,
        initial_state_sha256=initial_hash,selection_sha256=selection_hash,
        reset_state=reset_state,action_start_state=action_state,action_start_objects=action_start_objects,
        command_times=command_times,commands=commands,io=io_events,
        command_sha256=command_hash,command_timeline_sha256=timeline_hash,io_sha256=digest(io_events),
        final_state_sha256=digest(final_state),final_state=final_state,
        outcomes=core.safe_outcomes,physical_resolutions=list(core.safe_resolved.values()),
        contacts=core.safe_monitor.events,falls=core.safe_monitor.falls,
        executions=runner.safe_executions,residuals=records,decision_contexts=contexts,
        end_objects=end_objects,action_start=action_start,skill_end=skill_end,
        observation_end=core.sim_time(),prelude_monitor=prelude_monitor,
        definitions=dict(reset='Fresh robot startup, then one initial upstream placement; the physical belt continues moving.',
            initial_state='Exact MuJoCo integration state, mutable model geometry/friction/collision, object metadata, RNG, native encoder/control/queue at first native selection after the predeclared target-y geometry gate.',
            horizon='All arms integrate to action_start+action_horizon; target resolution never ends observation early.',
            event_start_phases='Count and full accumulated impulse attributed to the phase where an event starts; an event can continue across later phases.',
            impulse_windows='Prelude/action impulse uses actual cumulative solver-impulse snapshot differences, not event-start attribution.',
            pickup='Target initially ungrasped; complete native pickup is measured. Loaded/return diagnostic is not a held-reset reproduction.',
            row_rewards='Frozen-v2 full-workload collector rewards are retained only as diagnostics; single-target success is the designated target outcome.'))
    return dict(summary=summary,trace=trace,rows=[] if collector is None else collector.rows,
                initial_state_sha256=initial_hash,selection_sha256=selection_hash)


def collect_episode(scene_seed,skill,neighbor_count,policy,config=None,original_baseline=False,deterministic=True):
    config=dict(default_config(),**(config or {}))
    with redirect_stdout(io.StringIO()) if config.get('quiet',True) else nullcontext():
        return _collect(scene_seed,skill,neighbor_count,policy,config,bool(original_baseline),bool(deterministic))


def collect_baseline_episode(scene_seed,skill,neighbor_count,config=None):
    return collect_episode(scene_seed,skill,neighbor_count,None,config,original_baseline=True)

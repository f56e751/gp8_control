"""The original stream engine must retain its timing and IO under v2 control."""
import hashlib
import threading
import unittest
from types import SimpleNamespace

import numpy as np

from gp8_control.backends.robot_base import RobotBackend
from gp8_control.rl.safe_v2.adapter import SafeV2Adapter, dynamic_guard, route_delta


class Clock:
    def __init__(self): self.now = 0.
    def time(self): return self.now
    def monotonic(self): return self.now
    def sleep(self, dt): self.now += max(0., float(dt))


class Backend(RobotBackend):
    def __init__(self, clock):
        super().__init__(clock=clock)
        self.commands, self.io = [], []
        self.current_joints = [0.] * 6
        self.current_jointvels = [0.] * 6
    def wait_for_servers(self, timeout_sec=1.): return True
    def _emit_sample(self, positions):
        self.commands.append(np.asarray(positions, float).copy())
        self.current_joints = np.asarray(positions, float).tolist()
    def _set_suction(self, on, requested_at): self.io.append((on, requested_at))


def fixture(task='throw', next_task='push'):
    clock = Clock()
    backend = Backend(clock)
    target = SimpleNamespace(sim_object_id=1, class_name='transparent' if task == 'throw' else 'metal')
    upcoming = SimpleNamespace(sim_object_id=2, class_name='metal' if next_task == 'push' else 'transparent')
    next_result = None if next_task is None else (np.zeros(6), upcoming)
    core = SimpleNamespace(sim_time=clock.time, safe_residual_context=None,
        _box_state=[], _FREE=0, lock=threading.RLock(), cfg=SimpleNamespace(belt_speed=.12))
    robot = SimpleNamespace(joint_limits=[(-10., 10.)] * 6,
        forward_kinematics=lambda q: np.block([[np.eye(3), np.asarray(q[:3]).reshape(3, 1)],
                                             [np.zeros((1, 3)), np.ones((1, 1))]]))
    runner = SimpleNamespace(core=core, traj_ctrl=backend, robot=robot,
        M1=np.full(6, 100.), M2=np.full(6, 10000.),
        _active_target=target, _status='THROWING' if task == 'throw' else 'PUSHING',
        _chain_action=None, _skill_name_for_object=lambda t: 'push' if t.class_name == 'metal' else 'throw',
        ctx=SimpleNamespace(next_chain_target=lambda *a, **kw: next_result),
        skills={'throw':SimpleNamespace(_visualizer=SimpleNamespace(publish=lambda *a, **kw: None)),
                'push':SimpleNamespace(_last_push_meta={'n_descent':1,'n_stroke':25})},
        ee_xyz=lambda: np.asarray(backend.current_joints)[:3])
    return runner, clock


class V2AdapterTest(unittest.TestCase):
    def setUp(self):
        self.times = np.arange(101) * .02
        self.q = np.zeros((6, len(self.times)))
        self.dq = np.zeros_like(self.q)

    def attach(self, runner, action):
        calls = []
        adapter = SafeV2Adapter()
        def callback(core, obs, head, role):
            calls.append(dict(time=core.sim_time(),head=head,role=role,
                owner=adapter.current_owner.copy(),physical=core.safe_residual_context.copy()))
            return np.asarray(action), np.zeros(6), 0., np.zeros(2), np.zeros(107)
        adapter.decision_callback = callback
        adapter.bind(runner)
        adapter.observation = lambda *a: (np.zeros(97, np.float32), [])
        return adapter, calls

    def arm(self, runner):
        runner.ctx.next_chain_target(np.zeros(6), .5)
        runner.skills['throw']._visualizer.publish(self.q,self.dq,self.times,throw_last_index=50)

    def test_zero_actor_preserves_all_native_commands_io_time_and_tick(self):
        native, native_clock = fixture()
        native_ticks = []
        native.traj_ctrl._stream_trajectory(self.q,self.dq,self.times, np.zeros(6),
            release_index=25, tick_fn=lambda: native_ticks.append(native_clock.time()))
        runner, clock = fixture()
        adapter, calls = self.attach(runner, np.zeros(6));self.arm(runner)
        ticks=[]
        runner.traj_ctrl._stream_trajectory(self.q,self.dq,self.times,np.zeros(6),
            release_index=25,tick_fn=lambda:ticks.append(clock.time()))
        np.testing.assert_array_equal(runner.traj_ctrl.commands,native.traj_ctrl.commands)
        self.assertEqual(runner.traj_ctrl.io,native.traj_ctrl.io)
        self.assertEqual(ticks,native_ticks)
        self.assertEqual(clock.time(),native_clock.time())
        self.assertEqual([c['role']for c in calls],['throw_loaded','throw_return','push_approach'])
        self.assertTrue(all(b['time']>a['time']for a,b in zip(calls,calls[1:])))
        self.assertEqual(calls[-1]['owner']['id'],2)
        self.assertEqual(calls[-1]['physical']['target_id'],1)
        self.assertEqual(calls[-1]['head'],1)
        self.assertTrue(all(r['exact_identity']for r in adapter.records))
        self.assertIsNone(runner.core.safe_residual_context)

    def test_nonzero_tail_decision_happens_after_chain_boundary(self):
        runner,clock=fixture();adapter,calls=self.attach(runner,np.full(6,.5));self.arm(runner)
        runner.traj_ctrl._stream_trajectory(self.q,self.dq,self.times,release_index=25)
        last=adapter.records[-1]
        self.assertEqual(last['phase'],'push_approach')
        self.assertAlmostEqual(last['actual_boundary_time'],1.)
        self.assertGreater(last['max_joint_delta'],.05)
        commands=np.asarray(runner.traj_ctrl.commands)
        grid=np.arange(len(commands))*.004
        np.testing.assert_array_equal(commands[(grid>=.46-1e-12)&(grid<=.54+1e-12)],0.)
        np.testing.assert_array_equal(commands[:,3:],0.)

    def test_short_push_preparation_is_controlled_during_previous_transfer(self):
        runner,clock=fixture(task='push');adapter,calls=self.attach(runner,np.full(6,.5))
        runner.ctx.next_chain_target(np.zeros(6),.5)
        runner.traj_ctrl._stream_trajectory(self.q,self.dq,self.times)
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0]['role'],'push_approach')
        self.assertEqual(calls[0]['owner']['id'],2)
        self.assertAlmostEqual(calls[0]['time'],.54)
        commands=np.asarray(runner.traj_ctrl.commands)
        np.testing.assert_array_equal(commands[:131],0.)
        self.assertGreater(adapter.records[0]['max_joint_delta'],.05)

    def test_unaccepted_next_action_does_not_receive_ownership(self):
        runner,_=fixture(next_task=None);adapter,calls=self.attach(runner,np.zeros(6));self.arm(runner)
        runner._chain_action=SimpleNamespace(target=SimpleNamespace(sim_object_id=99),skill_name='push')
        runner.traj_ctrl._stream_trajectory(self.q,self.dq,self.times,release_index=25)
        self.assertEqual([c['role']for c in calls],['throw_loaded','throw_return'])
        self.assertTrue(all(c['owner']['id']==1 for c in calls))

    def test_next_throw_approach_is_owned_by_throw_head(self):
        runner,_=fixture(task='push',next_task='throw');adapter,calls=self.attach(runner,np.zeros(6))
        runner.ctx.next_chain_target(np.zeros(6),.5)
        runner.traj_ctrl._stream_trajectory(self.q,self.dq,self.times)
        self.assertEqual(calls[0]['head'],0)
        self.assertEqual(calls[0]['role'],'throw_return')
        self.assertEqual(calls[0]['owner']['id'],2)

    def test_negative_release_index_and_unbind_preserve_contract(self):
        runner,_=fixture();original=runner.traj_ctrl._stream_trajectory
        adapter,_=self.attach(runner,np.zeros(6));self.arm(runner)
        runner.traj_ctrl._stream_trajectory(self.q,self.dq,self.times,release_index=-76)
        self.assertEqual(adapter.records[0]['release_time'],.5)
        adapter.unbind()
        self.assertEqual(runner.traj_ctrl._stream_trajectory,original)

    def test_two_basis_coefficients_and_c2_seams_are_nontrivial(self):
        grid=np.arange(501)*.004
        symmetric=route_delta(grid,[1,0,0,0,0,0],.02,1.98)
        skew=route_delta(grid,[0,0,0,1,0,0],.02,1.98)
        self.assertGreater(np.max(symmetric[:,0]),.2)
        self.assertLess(np.min(skew[:,0]),-.03)
        self.assertGreater(np.max(skew[:,0]),.03)
        np.testing.assert_array_equal(symmetric[grid<=.02],0.)
        np.testing.assert_array_equal(symmetric[grid>=1.98],0.)

    def test_native_immutable_overspeed_allowed_but_changed_overspeed_rejected(self):
        runner,_=fixture();runner.M1[:]=1.;runner.M2[:]=10000.
        nominal=np.zeros((100,6));nominal[:,4]=np.arange(100)*.008
        candidate=nominal.copy();candidate[:,0]=np.arange(100)*.002
        self.assertTrue(dynamic_guard(runner,candidate,nominal)['accepted'])
        candidate[:,0]=np.arange(100)*.008
        self.assertFalse(dynamic_guard(runner,candidate,nominal)['accepted'])

    def test_resolved_owner_produces_finite_scene_observation(self):
        runner,_=fixture();adapter=SafeV2Adapter().bind(runner);self.arm(runner)
        frame=adapter._make_frame(self.q,self.dq,self.times,None,25)
        obs,neighbors=adapter.observation(frame,frame['windows'][-1])
        self.assertEqual(obs.shape,(97,))
        self.assertTrue(np.isfinite(obs).all())
        self.assertEqual(neighbors,[])

    def test_physics_endpoint_measurement_is_not_overwritten_by_control_cache(self):
        runner,_=fixture();adapter=SafeV2Adapter().bind(runner)
        record=dict(phase_end_time=1.,endpoint_reached=False,
            endpoint_position_error_m=.123,endpoint_q_error_rad=.456,endpoint_measured_time=1.)
        adapter._measure_endpoints({'endpoint_records':[record]},1.1)
        self.assertEqual(record['endpoint_position_error_m'],.123)
        self.assertEqual(record['endpoint_q_error_rad'],.456)
        self.assertTrue(record['endpoint_reached'])

    def test_static_guard_receives_five_args_and_preserves_projection_diagnostics(self):
        runner,_=fixture();adapter,_=self.attach(runner,np.full(6,.5));self.arm(runner)
        checked=[]
        def guard(actual_runner,candidate,nominal,grid,window):
            self.assertIs(actual_runner,runner)
            self.assertIn('release_time',window)
            self.assertEqual(candidate.shape,nominal.shape)
            accepted=np.max(np.abs(candidate-nominal))<=.02
            checked.append(accepted)
            return dict(accepted=bool(accepted),reason='test_floor')
        adapter.static_guard=guard
        runner.traj_ctrl._stream_trajectory(self.q,self.dq,self.times,release_index=25)
        self.assertTrue(checked)
        self.assertTrue(any(r['guard_rejections']for r in adapter.records))
        self.assertTrue(all(r['projection']<1 for r in adapter.records))

    def test_one_saturated_joint_does_not_freeze_other_two_joint_pairs(self):
        runner,_=fixture();runner.M1[0]=1e-9;runner.M2[0]=1e-9
        adapter,_=self.attach(runner,np.full(6,.5));self.arm(runner)
        runner.traj_ctrl._stream_trajectory(self.q,self.dq,self.times,release_index=25)
        for record in adapter.records:
            self.assertEqual(record['joint_projection'][0],0.)
            self.assertGreater(record['joint_projection'][1],0.)
            self.assertGreater(record['joint_projection'][2],0.)
            self.assertGreater(record['max_joint_delta'],.05)
            self.assertLess(record['projection'],1.)
            self.assertTrue(all(ratio<=1.+1e-8 for ratio in record['candidate_changed_max_ratio_per_joint']))
            self.assertTrue(all(ratio<=1.+1e-8 for ratio in record['acceleration_limit_ratio_per_joint']))
        commands=np.asarray(runner.traj_ctrl.commands)
        np.testing.assert_array_equal(commands[:,0],0.)
        self.assertGreater(np.max(np.abs(commands[:,1:3])),.05)


if __name__=='__main__':unittest.main()

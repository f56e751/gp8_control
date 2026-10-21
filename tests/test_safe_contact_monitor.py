"""Physical contact semantics exercised with a deterministic solver fixture."""
from types import SimpleNamespace as NS
import json
import unittest

import numpy as np

from gp8_control.rl.safe_contact_monitor import SafeContactMonitor


class _Model:
    ngeom = 6
    geom_bodyid = np.array([1, 2, 3, 4, 5, 0])
    body_parentid = np.array([0, 0, 1, 0, 0, 0])

    def body(self, name):
        if name != 'yaskawa_robot':
            raise KeyError(name)
        return NS(id=1)

    def geom(self, gid):
        return NS(name=['arm', 'pusher_collision', 'box1', 'box2', 'box3', 'belt'][gid])


class _Core:
    _FREE = 'free'

    def __init__(self):
        self.model = _Model()
        self._box_geom_ids = [2, 3, 4]
        self._box_serial = [1, 2, 3]
        self._box_state = ['on_belt'] * 3
        self._box_manipulated = [True, False, False]
        self._box_joints = [NS(qpos=np.array([x, 0., .642, 1., 0., 0., 0.]))
                            for x in (.4, .5, .6)]
        self._base_p = np.array([0., 0., .6])
        self._belt_top_w, self._dt = .627, .002
        self._grabbed, self._vacuum_on = None, False
        self.cfg = NS(despawn_y=-.8)
        self.safe_residual_context, self.time = None, .002
        self.safe_runner = NS(_status='POSITIONING', _active_target=NS(sim_object_id=1),
                             _skill_name_for_object=lambda _: 'throw',
                             _chain_action=None, cfg=NS(MAX_REACH=.65))
        self.set_contacts((1, 3))

    def set_contacts(self, *pairs, force=5.):
        self.data = NS(ncon=len(pairs), contact=[NS(geom1=g1, geom2=g2) for g1, g2 in pairs],
                       force=force)

    def sim_time(self):
        return self.time

    def _to_base(self, xyz):
        return xyz - self._base_p


def _force_reader(model, data, index, output):
    output[0] = data.force


class SafeContactTests(unittest.TestCase):
    def setUp(self):
        self.core = _Core()
        self.monitor = SafeContactMonitor(self.core, force_reader=_force_reader)

    def _throw_return(self, now=.9, next_skill='throw'):
        self.core.time = now
        self.core.safe_runner._chain_action = NS(skill_name=next_skill, target=NS(sim_object_id=2))
        self.core.safe_residual_context = dict(task='throw', phase='throw', target_id=1,
            start_sim_time=0., release_time=.3, duration=1.)

    def _push(self, now=.9, adapter_phase='push_prepare'):
        self.core.time = now
        self.core.safe_residual_context = dict(task='push', phase=adapter_phase, target_id=1,
            start_sim_time=0., prepare_end=.2, stroke_end=.6, duration=1.)

    def test_throw_approach_is_not_normal_suction_pick(self):
        self.assertEqual(self.monitor.classify_pair(1, 2)[0], 'robot_target')
        self.core._vacuum_on = True
        self.assertEqual(self.monitor.classify_pair(1, 2)[0], 'normal_target')
        self.assertEqual(self.monitor.classify_pair(0, 2)[0], 'robot_target')

    def test_full_push_boundaries_work_for_both_adapter_head_phases(self):
        for adapter_phase in ('push_prepare', 'push_route'):
            for time, phase, normal in ((.1, 'push_prepare', False),
                                        (.4, 'push_stroke', True), (.8, 'push_transfer', False)):
                with self.subTest(adapter_phase=adapter_phase, time=time):
                    self._push(time, adapter_phase)
                    self.assertEqual(self.monitor.phase(), phase)
                    self.assertEqual(self.monitor.classify_pair(1, 2)[0] == 'normal_target', normal)

    def test_chained_throw_tool_grasp_in_last_120ms_of_throw_return(self):
        self._throw_return()
        self.assertEqual(self.monitor.classify_pair(1, 3)[0], 'normal_target')
        self.monitor.observe()
        self.assertEqual(self.monitor.snapshot()['contacts'], 0)
        self.assertEqual(self.monitor.snapshot()['normal_contact_samples'], 1)

    def test_chained_throw_grasp_in_push_transfer(self):
        self._push()
        self.core.safe_runner._chain_action = NS(skill_name='throw', target=NS(sim_object_id=2))
        self.assertEqual(self.monitor.classify_pair(1, 3)[0], 'normal_target')

    def test_next_target_contact_before_window_is_unintended(self):
        self._throw_return(.879)
        self.assertEqual(self.monitor.classify_pair(1, 3)[0], 'robot_other')
        self._push(.879)
        self.assertEqual(self.monitor.classify_pair(1, 3)[0], 'robot_other')

    def test_queued_push_target_is_not_a_chained_suction_grasp(self):
        self._throw_return(next_skill='push')
        self.assertEqual(self.monitor.classify_pair(1, 3)[0], 'robot_other')

    def test_arm_link_touching_next_throw_target_is_unintended(self):
        self._throw_return()
        self.assertEqual(self.monitor.classify_pair(0, 3)[0], 'robot_other')

    def test_window_does_not_allow_next_target_during_push_stroke(self):
        self._push(.4)
        self.core.safe_residual_context['duration'] = .5
        self.core.safe_runner._chain_action = NS(skill_name='throw', target=NS(sim_object_id=2))
        self.assertEqual(self.monitor.phase(), 'push_stroke')
        self.assertEqual(self.monitor.classify_pair(1, 3)[0], 'robot_other')

    def test_impulse_uses_step_duration_and_ignores_duplicate_observe(self):
        self.monitor.observe()
        self.monitor.observe()
        self.assertEqual(self.monitor.snapshot()['contacts'], 1)
        self.assertAlmostEqual(self.monitor.snapshot()['impulse'], .01)
        self.core.time += .002
        self.monitor.observe()
        self.assertEqual(self.monitor.snapshot()['contacts'], 1)
        self.assertAlmostEqual(self.monitor.snapshot()['impulse'], .02)

    def test_force_at_threshold_is_excluded(self):
        self.core.set_contacts((1, 3), force=.1)
        self.monitor.observe()
        self.assertEqual(self.monitor.snapshot()['contacts'], 0)

    def test_gap_and_reused_slot_start_distinct_events(self):
        self.monitor.observe()
        self.core.time += .04
        self.monitor.observe()
        self.assertEqual(self.monitor.snapshot()['contacts'], 2)
        self.core._box_serial[1] = 4
        self.core.time += .002
        self.monitor.observe()
        self.assertEqual(self.monitor.snapshot()['contacts'], 3)
        self.assertEqual(self.monitor.events[-1]['object_ids'], [4])

    def test_collateral_fall_is_unique_and_excludes_intended_objects(self):
        self.monitor.observe()
        self.core._box_joints[1].qpos[2] = .5
        self.core.time += .002
        self.monitor.observe()
        self.core.time += .002
        self.monitor.observe()
        self.assertEqual(self.monitor.snapshot()['falls'], 1)
        self.assertEqual(self.monitor.falls[0]['cause_task'], 'throw')
        self.core._box_joints[0].qpos[2] = .5
        self.core.set_contacts((0, 2))
        self.core.time += .002
        self.monitor.observe()
        self.assertEqual(self.monitor.snapshot()['falls'], 1)

    def test_simultaneous_object_chain_is_independent_of_contact_order(self):
        self.core.set_contacts((3, 4), (2, 3))
        self.monitor.observe()
        self.assertEqual(self.monitor.snapshot()['contacts'], 2)
        self.assertAlmostEqual(self.monitor.snapshot()['impulse'], .02)
        self.assertEqual(self.monitor._collateral_ids, {2, 3})

    def test_passive_object_traffic_is_not_robot_cost(self):
        self.core._box_manipulated = [False] * 3
        self.core.safe_runner._active_target = None
        self.core.set_contacts((3, 4))
        self.monitor.observe()
        self.assertEqual(self.monitor.snapshot()['contacts'], 0)

    def test_metrics_and_events_are_json_serializable(self):
        self.monitor.observe()
        json.dumps(dict(snapshot=self.monitor.snapshot(), events=self.monitor.events,
                        falls=self.monitor.falls))


if __name__ == '__main__':
    unittest.main()

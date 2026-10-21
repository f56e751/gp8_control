"""Finite-workload invariants without constructing a MuJoCo model."""

from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

import numpy as np

from gp8_control.backends.mujoco_sim import SimCore
from gp8_control.rl.safe_simulator_env import (
    EpisodeStopped, OfferedTapeCore, ResidualTransitions, _collect_episode,
    default_config, make_tape,
)


class FiniteWorkloadTests(unittest.TestCase):
    def test_tape_is_fixed_moderate_and_has_both_skills(self):
        config = default_config()
        tape, rate = make_tape(920001, config)
        again, again_rate = make_tape(920001, config)
        other, _ = make_tape(920002, config)
        self.assertEqual(tape, again)
        self.assertEqual(rate, again_rate)
        self.assertNotEqual(tape, other)
        self.assertEqual([row['id'] for row in tape], list(range(1, 21)))
        self.assertEqual(sum(row['class_name'] == 'metal' for row in tape), 10)
        self.assertEqual(sum(row['class_name'] == 'transparent' for row in tape), 10)
        self.assertTrue(.6 <= rate <= .9)
        intervals = np.diff([row['time'] for row in tape]) * rate
        self.assertTrue(np.all(intervals >= .8))
        self.assertTrue(np.all(intervals <= 1.2))

    def test_budget_counts_live_failed_pick_until_physical_resolution(self):
        core = object.__new__(OfferedTapeCore)
        core.safe_tape = [{'id': index} for index in range(1, 4)]
        core.safe_arrival_index = 1
        core._box_state = [core._ON_BELT, core._FREE]
        core.safe_outcomes = [{'sim_object_id': 1, 'reward': -.3}]
        core.safe_started = 3.
        core.data = SimpleNamespace(time=13.)
        core.safe_config = {'time_limit': 120.}
        np.testing.assert_allclose(core.safe_budget_features(), [2/3, 1/3, 110/120])
        core._box_state[0] = core._FREE
        np.testing.assert_allclose(core.safe_budget_features(), [2/3, 0., 110/120])

    def test_physical_resolution_is_recorded_before_native_slot_reset(self):
        core = object.__new__(OfferedTapeCore)
        core._box_serial = [7]
        core._box_state = [core._ON_BELT]
        core._box_joints = [SimpleNamespace(qpos=np.array([.4, -.81, .5, 1., 0., 0., 0.]))]
        core.safe_active, core.safe_started = True, 2.
        core.safe_resolved = {}
        core.data = SimpleNamespace(time=12.)
        with patch.object(SimCore, '_park_box') as native_park:
            core._park_box(0)
        native_park.assert_called_once_with(0)
        self.assertEqual(core.safe_resolved[7]['time'], 10.)
        self.assertEqual(core.safe_resolved[7]['position'], [.4, -.81, .5])
        core._box_joints[0].qpos[:3] = 0.
        self.assertEqual(core.safe_resolved[7]['position'], [.4, -.81, .5])

    def test_outcomes_survive_consuming_native_reward_queue(self):
        core = object.__new__(OfferedTapeCore)
        core._reward_events, core.safe_outcomes = [], []
        outcome = {'sim_object_id': 9, 'base_xyz': [.4, .2, -.41], 'reward': -.3}
        with patch.object(SimCore, '_emit_reward_event',
                          side_effect=lambda *a, **kw: core._reward_events.append(outcome)):
            core._emit_reward_event(0, [.4, .2, -.41], None, False)
        self.assertEqual(core.safe_outcomes, [outcome])
        self.assertIsNot(core.safe_outcomes[0], outcome)
        core._reward_events.clear()
        outcome['reward'] = 99.
        self.assertEqual(core.safe_outcomes[0]['reward'], -.3)

    def test_fifo_backlog_refreshes_pose_before_checking_next_offer(self):
        core = object.__new__(OfferedTapeCore)
        core.safe_tape, _ = make_tape(720001, dict(default_config(), offer_count=2))
        core.safe_arrival_index, core.safe_started = 0, 0.
        core.safe_admissions = []
        core.cfg = SimpleNamespace(spawn_y=2.4)
        core.data = SimpleNamespace(time=10.)
        core._box_state = [core._FREE, core._FREE]
        core._box_joints = [SimpleNamespace(qpos=np.zeros(7), qvel=np.zeros(6)) for _ in range(2)]
        core._box_geom_ids = [0, 1]
        core._base_p, core._belt_top_w = np.zeros(3), .6
        core._spawn_count = 0
        core.model = SimpleNamespace(geom_rgba=np.zeros((2, 4)))
        for field in ('_box_class', '_box_suction_p', '_box_belt_y', '_box_serial',
                      '_box_manipulated', '_box_manipulated_skill', '_box_class_bin_name',
                      '_box_sealed', '_box_reward_done', '_box_cause_step'):
            setattr(core, field, [None, None])
        core._activate_box = lambda slot: None
        core._apply_box_size = lambda slot, size: None
        core._to_world = lambda pos: np.asarray(pos)
        core._box_rgba = lambda cls, probability: [1., 1., 1., 1.]
        calls = []
        core._spawn_pose_is_clear = lambda *args: calls.append('clear') or True
        with patch('gp8_control.rl.safe_simulator_env.mujoco.mj_forward',
                   side_effect=lambda *args: calls.append('forward')):
            core._admit_offers()
        self.assertEqual(calls, ['clear', 'forward', 'clear', 'forward'])
        self.assertEqual(core.safe_arrival_index, 2)
        self.assertEqual(core._box_serial, [1, 2])

    def test_observed_deadline_stops_before_another_physics_step(self):
        core = object.__new__(OfferedTapeCore)
        core.safe_active, core.safe_started = True, 0.
        core.safe_config = {'time_limit': 120.}
        core.data = SimpleNamespace(time=120.)
        with patch.object(SimCore, '_step_once_locked') as native_step:
            with self.assertRaises(EpisodeStopped) as raised:
                core._step_once_locked()
        self.assertEqual(raised.exception.reason, 'time_limit')
        native_step.assert_not_called()

    def test_pre_episode_start_does_not_consume_deadline(self):
        core = object.__new__(OfferedTapeCore)
        core.safe_active, core.safe_started = False, 0.
        core.safe_config = {'time_limit': 120.}
        core.data = SimpleNamespace(time=150.)
        with patch.object(SimCore, '_step_once_locked') as native_step:
            core._step_once_locked()
        native_step.assert_called_once_with()

    def test_deadline_flushes_final_native_outcome_before_terminal_penalty(self):
        config = dict(default_config(), offer_count=1)
        tape, _ = make_tape(730001, config)
        calls = []
        monitor = SimpleNamespace(events=[], falls=[], snapshot=lambda: dict(
            contact_events=0, impulse=0., falls=0))
        core = SimpleNamespace(_FREE='free', _box_state=['loose'], safe_started=0.,
            safe_arrival_index=1, safe_admissions=[dict(tape[0], actual_time=.5)],
            safe_outcomes=[], safe_resolved={}, safe_active_counts=[1], safe_active=False,
            safe_monitor=monitor, sim_time=lambda: 120., lock=threading.Lock())

        def begin():
            core.safe_active = True

        def recycle():
            self.assertTrue(core.safe_active)
            calls.append('recycle')
            core.safe_outcomes.append(dict(sim_object_id=1, actual_bin_name='throw',
                class_bin_name='throw', sim_time=120., manipulated_skill='throw'))
            core._box_state[0] = 'free'
            core.safe_resolved[1] = dict(id=1, time=120.)

        def step(action):
            raise EpisodeStopped('time_limit')

        def finalize(**kwargs):
            calls.append('finalize')
            self.assertEqual(len(core.safe_outcomes), 1)
            self.assertEqual(kwargs, dict(done=True, reason='time_limit'))

        core.begin_episode, core._recycle_boxes = begin, recycle
        runner = SimpleNamespace(core=core, safe_executions=[], start=lambda: None,
            ingest=lambda: None, action_mask=lambda: None,
            app_heuristic_action_indices=lambda mask: [0, 0],
            action_from_indices=lambda *args: None, step=step,
            close=lambda: calls.append('close'))
        collector = SimpleNamespace(rows=[], finalize=finalize, decide=lambda *args: None)
        adapter = SimpleNamespace(records=[], bind=lambda r: None)
        with patch('gp8_control.rl.safe_simulator_env.NativeResidualRunner', return_value=runner), \
             patch('gp8_control.rl.safe_simulator_env.ResidualTransitions', return_value=collector), \
             patch('gp8_control.rl.safe_simulator_env.SafeResidualAdapter', return_value=adapter):
            result = _collect_episode(730001, None, config)
        self.assertEqual(calls, ['recycle', 'finalize', 'close'])
        self.assertEqual(result['summary']['correct_bin'], 1)
        self.assertEqual(result['summary']['unresolved'], 0)

    def _collector_with_unfinished_workload(self):
        outcome = dict(sim_object_id=1, manipulated=True, collateral=False,
                       actual_bin_name=None, class_bin_name='throw')
        core = SimpleNamespace(
            safe_outcomes=[outcome], safe_tape=[{'id': i} for i in (1, 2, 3)],
            _box_serial=[1, 2, 3], _box_state=['on_belt']*3,
            _box_manipulated=[True, True, False], sim_time=lambda: 5.,
            safe_monitor=SimpleNamespace(snapshot=lambda: dict(contact_events=0, impulse=0., falls=0)))
        runner = SimpleNamespace(core=core, safe_adapter=SimpleNamespace(records=[]))
        collector = ResidualTransitions(runner, None, 'episode', True)
        collector.track_neighbors = lambda: None
        collector.pending = dict(phase='throw', action=np.zeros(6),
            _before=dict(time=0., outcomes=0, contacts=0, impulse=0., falls=0),
            _ignored=set(), _origins={}, _shifts={}, _record_start=0)
        return collector

    def test_terminal_penalty_does_not_charge_failed_pick_twice(self):
        collector = self._collector_with_unfinished_workload()
        collector.finalize(done=True, reason='time_limit')
        row = collector.rows[0]
        self.assertEqual(row['reward_components']['attempted_failure'], -2.)
        # ID 1 has already failed; only manipulated ID 2 and untouched ID 3 remain.
        self.assertEqual(row['reward_components']['unfinished_workload'], -2.5)
        self.assertAlmostEqual(row['reward'], -4.6)
        self.assertTrue(row['terminated'])
        self.assertFalse(row['truncated'])
        self.assertEqual(row['cost'], 0.)

    def test_nonfinite_failure_is_an_absorbing_failure_cost(self):
        collector = self._collector_with_unfinished_workload()
        collector.finalize(done=True, reason='physics_nonfinite')
        row = collector.rows[0]
        self.assertEqual(row['reward_components']['physics_failure'], -10.)
        self.assertEqual(row['cost'], 10.)
        self.assertTrue(row['terminated'])


if __name__ == '__main__':
    unittest.main()

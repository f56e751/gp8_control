"""Phase ownership, delayed outcomes and fixed-workload constraint tests."""
from types import SimpleNamespace as NS
import unittest

import numpy as np

from gp8_control.rl.safe_v2.env import RouteTransitions
from gp8_control.rl.safe_v2.ppo import SafePPO
from gp8_control.rl.safe_ppo import FixedScaler


class _Core:
    _FREE = 'free'

    def __init__(self):
        self.now = 0.
        self.metrics = dict(contact_events=0, impulse=0., falls=0,
                            proximity_integral=0., current_clearance_m=-.01)
        self.safe_monitor = NS(snapshot=lambda: self.metrics.copy())
        self.safe_outcomes = []
        self.safe_tape = [dict(id=i) for i in (1, 2, 3, 4)]
        self._box_serial = [1, 2, 3, 4]
        self._box_state = ['on_belt'] * 4
        self._box_manipulated = [True, False, False, False]
        self._box_joints = [NS(qpos=np.array([.1*i, 0., .1, 1., 0., 0., 0.]))
                            for i in range(4)]
        self._grabbed = None

    def sim_time(self):
        return self.now

    def safe_budget_features(self):
        return np.asarray([.8, .1, .7], np.float32)


def _fixture():
    core = _Core()
    adapter = NS(window_duration=1.2, current_owner=dict(id=1, task='throw'), records=[])
    runner = NS(core=core, safe_adapter=adapter, _chain_action=NS(target=NS(sim_object_id=2)),
                _active_target=NS(sim_object_id=1))
    return core, runner, RouteTransitions(runner, None, 'episode', True)


def _outcome(serial, correct=False, manipulated=True):
    return dict(sim_object_id=serial, actual_bin_name='throw' if correct else None,
                class_bin_name='throw', manipulated=manipulated, collateral=False)


class PhaseCollectorTests(unittest.TestCase):
    def test_next_push_owner_and_role_are_encoded_in_107_features(self):
        core, runner, collector = _fixture()
        runner.safe_adapter.current_owner = dict(id=2, task='push', physical_target_id=1)
        observation = np.arange(97, dtype=np.float32)/1000.
        result = collector.decide(core, observation, 1, 'push_approach')
        row = collector.pending
        self.assertEqual(row['raw_obs'].shape, (107,))
        np.testing.assert_array_equal(row['raw_obs'][:97], observation)
        np.testing.assert_allclose(row['raw_obs'][97:100], [.8, .1, .7])
        np.testing.assert_array_equal(row['raw_obs'][100:104], [0., 0., 1., 0.])
        np.testing.assert_allclose(row['raw_obs'][104:], [.3, -.2, 0.])
        np.testing.assert_array_equal(result[-1], row['raw_obs'])
        self.assertEqual(row['owner_target_id'], 2)
        self.assertEqual(row['head'], 1)
        self.assertEqual(runner._active_target.sim_object_id, 1)
        self.assertEqual(core._box_manipulated, [True, False, False, False])

    def test_contacts_after_actual_handoff_belong_to_push_approach_row(self):
        core, runner, collector = _fixture()
        observation = np.zeros(97, np.float32)
        collector.decide(core, observation, 0, 'throw_loaded')
        runner.safe_adapter.records.append(dict(rejected=False, projection=1.))
        core.now = 1.
        runner.safe_adapter.current_owner = dict(id=2, task='push')
        collector.decide(core, observation, 1, 'push_approach')
        runner.safe_adapter.records.append(dict(rejected=False, projection=1.))
        core.now = 1.4
        core.metrics.update(contact_events=2, impulse=3., proximity_integral=.2)
        collector.finalize(done=True, reason='time_limit')
        first, approach = collector.rows
        self.assertEqual(first['contacts'], 0)
        self.assertEqual(first['cost'], 0.)
        self.assertEqual(first['reward_components']['proximity'], 0.)
        self.assertEqual(approach['role'], 'push_approach')
        self.assertEqual(approach['owner_target_id'], 2)
        self.assertEqual(approach['contacts'], 2)
        self.assertAlmostEqual(approach['impulse_ns'], 3.)
        self.assertAlmostEqual(approach['proximity_integral'], .2)
        self.assertAlmostEqual(approach['cost'], 2.3)
        self.assertEqual(approach['reward_components']['contact'], -2.)
        self.assertAlmostEqual(approach['reward_components']['impulse'], -.3)
        self.assertAlmostEqual(approach['reward_components']['proximity'], -.4)
        self.assertLess(approach['reward'], 0.)

    def test_delayed_outcome_waits_and_failed_pick_is_not_charged_twice(self):
        core, runner, collector = _fixture()
        core.safe_outcomes.append(_outcome(1))  # Earlier unsealed failure, still physically live.
        core._box_manipulated[1] = True
        observation = np.zeros(97, np.float32)
        collector.decide(core, observation, 0, 'throw_loaded')
        runner.safe_adapter.records.append(dict(rejected=False, projection=1.))
        core.now = .5
        runner.safe_adapter.current_owner = dict(id=2, task='push')
        collector.decide(core, observation, 1, 'push_approach')
        self.assertEqual(collector.rows[0]['reward_components']['bin_success'], 0.)
        core.safe_outcomes.append(_outcome(4, correct=True))
        core._box_state[3] = core._FREE
        core.now = 1.
        collector.finalize(done=True, reason='time_limit')
        final = collector.rows[-1]
        self.assertEqual(final['reward_components']['bin_success'], 4.)
        self.assertEqual(final['reward_components']['attempted_failure'], 0.)
        # ID 1 already has an outcome, ID 4 just landed: only IDs 2 and 3 remain.
        self.assertEqual(final['reward_components']['unfinished_workload'], -2.5)
        self.assertTrue(final['terminated'])
        self.assertFalse(final['truncated'])
        collector.finalize(done=True, reason='time_limit')
        self.assertEqual(len(collector.rows), 2)

    def test_push_pose_bonus_requires_a_measured_approach_endpoint(self):
        core, runner, collector = _fixture()
        runner.safe_adapter.current_owner = dict(id=2, task='push')
        collector.decide(core, np.zeros(97, np.float32), 1, 'push_approach')
        runner.safe_adapter.records.append(dict(rejected=False, projection=1.,
            endpoint_position_error_m=0., endpoint_q_error_rad=0.))
        core.now = .2
        collector.finalize()
        self.assertEqual(collector.rows[0]['reward_components']['push_pose'], .5)


class WorkloadConstraintTests(unittest.TestCase):
    def _update_with_splits(self, count):
        scaler = FixedScaler(107).fit(np.zeros((2, 107), np.float32))
        policy = SafePPO(39, scaler)
        rows = []
        raw = np.zeros(107, np.float32)
        for index in range(count):
            action, latent, logp, value, normalized = policy.act(raw, index % 2)
            rows.append(dict(obs=normalized, action=action, latent=latent, logp=logp,
                value=value, head=index % 2, reward=0., cost=4./count,
                duration=1./count, episode_id='with_decisions',
                terminated=index == count-1, truncated=False))
        # The second full workload has no NN decisions and still spends budget.
        summaries = [dict(offered=20, total_cost=4.), dict(offered=20, total_cost=6.)]
        return policy.update(rows, summaries, budget=.3)

    def test_phase_splitting_cannot_dilute_offered_workload_cost(self):
        two = self._update_with_splits(2)
        five = self._update_with_splits(5)
        self.assertEqual(two['full_episodes'], 2)
        self.assertEqual(five['full_episodes'], 2)
        self.assertAlmostEqual(two['cost_per_offered'], .25)
        self.assertAlmostEqual(five['cost_per_offered'], .25)
        self.assertAlmostEqual(two['multiplier'], five['multiplier'])
        self.assertNotEqual(two['mean_cost_per_decision_diagnostic'],
                            five['mean_cost_per_decision_diagnostic'])


if __name__ == '__main__':
    unittest.main()

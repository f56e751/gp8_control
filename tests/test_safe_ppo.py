"""Algorithm checks independent of the MuJoCo integration."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from gp8_control.rl.safe_ppo import FixedScaler, SafePPO, calculate_gae


def row(episode=0, reward=1., value=(.5, .2), **kw):
    return dict(episode_id=episode, reward=reward, cost=.1, value=value,
                duration=1., terminated=False, truncated=False, **kw)


class SafePPOTest(unittest.TestCase):
    def test_terminal_has_no_bootstrap(self):
        r = row()
        r['terminated'] = True
        result = calculate_gae([r], gamma=.9)
        np.testing.assert_allclose(result['targets'], [[1., .1]])

    def test_truncation_bootstraps_without_leaking_reset(self):
        first = row(next_value=(2., 3.))
        first['truncated'] = True
        second = row(episode=1, reward=100.)
        second['terminated'] = True
        result = calculate_gae([first, second], gamma=.9)
        np.testing.assert_allclose(result['targets'][0], [2.8, 2.8], rtol=1e-6)
        np.testing.assert_allclose(result['targets'][1], [100., .1])
        with self.assertRaisesRegex(ValueError, 'requires'):
            first.pop('next_value')
            calculate_gae([first])

    def test_incomplete_sequence_rejected(self):
        with self.assertRaises(ValueError):
            calculate_gae([row()])
        first = row()
        first['terminated'] = True
        with self.assertRaises(ValueError):
            calculate_gae([first, row()])

    def test_full_checkpoint_restores_rng_and_general_width(self):
        scaler = FixedScaler(102).fit(np.zeros((2, 102)))
        source = SafePPO(30, scaler=scaler, obs_size=102)
        obs = np.zeros(102)
        self.assertEqual(source.act(obs, 0, True)[0].tolist(), [0.] * 6)
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / 'policy.pt'
            source.save(checkpoint, {'seed': 30})
            first = source.act(obs, 1)
            target = SafePPO(31, obs_size=102)
            self.assertEqual(target.load(checkpoint), {'seed': 30})
            second = target.act(obs, 1)
            for a, b in zip(first, second):
                np.testing.assert_array_equal(a, b)
            with self.assertRaisesRegex(ValueError, 'width'):
                SafePPO(30).load(checkpoint)
        with self.assertRaisesRegex(RuntimeError, 'frozen'):
            scaler.fit(np.ones((2, 102)))

    def test_two_head_update_uses_safety_cost(self):
        scaler = FixedScaler(100).fit(np.zeros((2, 100)))
        policy = SafePPO(30, scaler=scaler, obs_size=100)
        rows = []
        for index in range(12):
            action, latent, logp, value, obs = policy.act(np.zeros(100), index % 2)
            rows.append(dict(episode_id=0, reward=float(index % 3), cost=.5,
                duration=.2, value=value, obs=obs, latent=latent, logp=logp,
                head=index % 2, terminated=index == 11, truncated=False))
        stats = policy.update(rows, cost_limit=.1)
        self.assertEqual(stats['total_updates'], 1)
        self.assertGreater(stats['total_optimizer_steps'], 0)
        self.assertAlmostEqual(stats['multiplier'], 1.08)
        self.assertTrue(np.isfinite(stats['loss']))


if __name__ == '__main__':
    unittest.main()

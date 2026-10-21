"""Recovery tests for real PPO with a tiny deterministic mock collector."""
from argparse import Namespace
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from gp8_control.rl.safe_v2.ppo import SafePPO
from gp8_control.tools import train_safe_simulator_v2 as trainer


class _Executor:
    def __init__(self, *args, **kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False


def _collect(_executor, jobs):
    results = []
    for job in jobs:
        rows = []
        state = job.get('state')
        policy = None
        if state is not None:
            policy = SafePPO(0, obs_size=107)
            policy.restore_rollout_state(state)
            torch.manual_seed(int(job['scene_seed']))
        for index in range(0 if job.get('name')=='baseline' else 4):
            raw = np.linspace(0., 1., 107) + index*.01
            row = dict(raw_obs=raw)
            if policy is not None:
                action, latent, logp, value, obs = policy.act(raw, index % 2)
                row.update(obs=obs, action=action, latent=latent, logp=logp,
                    value=value, head=index % 2, reward=float(action.sum()),
                    cost=.2, duration=.5, episode_id=job['scene_seed'],
                    terminated=index == 3, truncated=False)
            if policy is None:
                row.update(obs=raw.copy(),action=np.zeros(6),latent=np.zeros(6),logp=0.,
                    value=np.zeros(2),head=index%2,reward=1.,cost=.2,duration=.5,
                    episode_id=job['scene_seed'],terminated=index==3,truncated=False)
            rows.append(row)
        results.append(dict(rows=rows, summary=dict(scene_seed=job['scene_seed'],offered=20,total_cost=4.)))
    return results


class SafeV2TrainerTest(unittest.TestCase):
    def test_interrupt_after_warmup_can_resume_initialization(self):
        args = Namespace(run='startup', fresh=True, resume=False, updates=1,
            episodes=1, warmup_episodes=1, workers=1, episode_config=None,
            budget_fraction=.8)
        def interrupt(*args, **kwargs):
            raise RuntimeError('startup interrupt')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(trainer, 'ROOT', root), patch.object(trainer, 'read_config', return_value={'residual_scale':.25}), \
                 patch.object(trainer, 'ProcessPoolExecutor', _Executor), patch.object(trainer, 'collect', _collect), patch.object(trainer,'source_hashes',return_value={}), \
                 contextlib.redirect_stdout(io.StringIO()):
                with patch.object(SafePPO, 'save', interrupt):
                    with self.assertRaisesRegex(RuntimeError, 'startup interrupt'):
                        trainer.train(args)
                self.assertTrue((root/'runs/startup/scaler.npz').is_file())
                self.assertTrue((root/'runs/startup/warmup_observations.npz').is_file())
                self.assertFalse((root/'runs/startup/committed_update.json').exists())
                args.fresh, args.resume = False, True
                trainer.train(args)
                self.assertTrue((root/'runs/startup/training_summary.json').is_file())
                initial = torch.load(root/'checkpoints/startup/safe_seed0_update000.pt', weights_only=False)
                self.assertEqual(initial['total_updates'], 0)
                self.assertEqual(initial['actor_optimizer']['state'], {})

    def test_common_commit_resume_reproduces_uninterrupted_training(self):
        args = Namespace(run='tiny', fresh=True, resume=False, updates=2,
            episodes=2, warmup_episodes=2, workers=1, episode_config=None,
            budget_fraction=.8)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(trainer, 'ROOT', root), patch.object(trainer, 'read_config', return_value={'residual_scale':.25}), \
                 patch.object(trainer, 'ProcessPoolExecutor', _Executor), patch.object(trainer, 'collect', _collect), patch.object(trainer,'source_hashes',return_value={}), \
                 contextlib.redirect_stdout(io.StringIO()):
                trainer.train(args)
                checkpoint = root/'checkpoints/tiny'
                final = [torch.load(checkpoint/f'safe_seed{i}.pt', weights_only=False) for i in (0, 1)]
                # Simulate a process exiting after an uncommitted pair was written.
                trainer.save_json(root/'runs/tiny/committed_update.json', dict(update=1))
                args.fresh, args.resume = False, True
                trainer.train(args)
                for i, expected in enumerate(final):
                    actual = torch.load(checkpoint/f'safe_seed{i}.pt', weights_only=False)
                    self.assertEqual(actual['total_updates'], 2)
                    for key, tensor in expected['network'].items():
                        torch.testing.assert_close(actual['network'][key], tensor, rtol=0, atol=0)
                    self.assertEqual(actual['actor_optimizer']['param_groups'], expected['actor_optimizer']['param_groups'])
                self.assertEqual(json.loads((root/'runs/tiny/committed_update.json').read_text())['update'], 2)
                self.assertEqual(len(json.loads((root/'runs/tiny/updates.json').read_text())), 4)
                with self.assertRaisesRegex(RuntimeError, 'artifacts'):
                    args.fresh, args.resume = True, False
                    trainer.train(args)


if __name__ == '__main__':
    unittest.main()

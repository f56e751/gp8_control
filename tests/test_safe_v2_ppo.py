"""Verify real cost learning and isolation from large critic gradients."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from gp8_control.rl.safe_v2.ppo import FixedScaler, SafePPO


def policy(seed=40):
    return SafePPO(seed,FixedScaler(107).fit(np.zeros((2,107))),return_scales=(2.,1.))


def collision_rows(p, scale=1.):
    rows=[]
    raw=np.zeros(107,np.float32)
    normalized=p.scaler.normalize(raw)
    for index in range(32):
        costly=index%2
        latent=np.zeros(6,np.float32);latent[0]=.2 if costly else -.2
        with torch.no_grad():
            dist,value=p.network(torch.from_numpy(normalized[None]),torch.tensor([0]))
            z=torch.from_numpy(latent[None]);action=torch.tanh(z)
            logp=(dist.log_prob(z)-torch.log(1.-action.square()+1e-6)).sum(-1)
        rows.append(dict(obs=normalized,raw_obs=raw,latent=latent,action=action.numpy()[0],
            head=0,logp=float(logp),value=value.numpy()[0],reward=0.,cost=float(costly)*scale,
            duration=1.,episode_id=index,terminated=True,truncated=False))
    return rows


def costly_probability(p):
    with torch.no_grad():
        dist=p.network.actor(torch.zeros((1,107)),torch.tensor([0]))
        return float((1.-dist.cdf(torch.zeros((1,6))))[0,0])


class SafeV2PPOTest(unittest.TestCase):
    def test_bodies_and_optimizer_parameters_are_disjoint(self):
        p=policy()
        groups=[{id(v) for v in module.parameters()} for module in
                (p.network.actor,p.network.reward_critic,p.network.cost_critic)]
        for i in range(3):
            for j in range(i+1,3):self.assertFalse(groups[i]&groups[j])
        actor={id(v) for group in p.actor_optimizer.param_groups for v in group['params']}
        critic={id(v) for group in p.critic_optimizer.param_groups for v in group['params']}
        self.assertFalse(actor&critic)
        self.assertEqual(actor,groups[0]);self.assertEqual(critic,groups[1]|groups[2])

    def test_high_collision_action_probability_decreases(self):
        p=policy();before=costly_probability(p)
        stats=p.update(collision_rows(p),[dict(offered=20,total_cost=16.)],budget=.2)
        self.assertLess(costly_probability(p),before)
        self.assertGreater(stats['actor_grad_norm'],0.)
        self.assertGreater(stats['cost_advantage_std'],0.)

    def test_million_times_larger_critic_loss_does_not_suppress_actor_update(self):
        normal,large=policy(),policy()
        initial={k:v.clone() for k,v in normal.network.actor.state_dict().items()}
        a=normal.update(collision_rows(normal),[dict(offered=20,total_cost=16.)],budget=.2)
        b=large.update(collision_rows(large,1e6),[dict(offered=20,total_cost=16e6)],budget=.2)
        self.assertGreater(b['cost_value_mse'],a['cost_value_mse']*1e10)
        self.assertGreater(b['actor_grad_norm'],a['actor_grad_norm']*.99)
        self.assertGreater(sum(float((large.network.actor.state_dict()[k]-v).abs().sum()) for k,v in initial.items()),1e-4)
        for k,v in normal.network.actor.state_dict().items():
            torch.testing.assert_close(v,large.network.actor.state_dict()[k],atol=1e-6,rtol=1e-5)

    def test_constraint_counts_full_workloads_including_no_decision_episode(self):
        p=policy()
        stats=p.update(collision_rows(p),[dict(offered=20,total_cost=20.),dict(offered=20,total_cost=0.)],budget=.2)
        self.assertEqual(stats['cost_per_offered'],.5)
        self.assertAlmostEqual(stats['multiplier'],1.75)
        self.assertEqual(stats['full_episodes'],2)

    def test_values_stay_in_original_units(self):
        p=policy();p.return_scales=np.array([100.,20.])
        with torch.no_grad():
            p.network.reward_critic.output.bias.fill_(7.)
            p.network.cost_critic.output.bias.fill_(3.)
        np.testing.assert_array_equal(p.value(np.zeros(107),0),[7.,3.])

    def test_checkpoint_restores_both_adam_states_and_sampling_rng(self):
        p=policy();p.update(collision_rows(p),[dict(offered=20,total_cost=16.)],budget=.2)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'v2.pt';p.save(path,dict(seed=40))
            first=p.act(np.zeros(107),0)
            expected_permutation=p.rng.permutation(10)
            restored=policy(41);self.assertEqual(restored.load(path),dict(seed=40))
            second=restored.act(np.zeros(107),0)
            for x,y in zip(first,second):np.testing.assert_array_equal(x,y)
            np.testing.assert_array_equal(restored.rng.permutation(10),expected_permutation)
            self.assertTrue(restored.actor_optimizer.state);self.assertTrue(restored.critic_optimizer.state)
            self.assertEqual(restored.total_optimizer_steps,p.total_optimizer_steps)


if __name__=='__main__':unittest.main()

"""Boundary Safe PPO with independent actor and reward/cost value networks.

Critics predict original reward/cost units. Only their regression errors are
scaled by the once-fitted warmup return scales, and no critic gradient enters
the actor optimizer or its gradient clipping. The constraint uses full offered
workloads, including episodes without a policy decision.
"""
from __future__ import annotations

from itertools import chain
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from gp8_control.rl.safe_ppo import FixedScaler, calculate_gae

OBS_SIZE = 107
ACTION_SIZE = 6


def _body(width):
    body = nn.Sequential(nn.Linear(width, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
    for layer in body:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, np.sqrt(2.))
            nn.init.zeros_(layer.bias)
    return body


class Actor(nn.Module):
    def __init__(self, obs_size):
        super().__init__()
        self.body = _body(obs_size)
        self.output = nn.Linear(64, 2*ACTION_SIZE)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.log_std = nn.Parameter(torch.full((2, ACTION_SIZE), -1.5))

    def forward(self, observations, heads):
        hidden = self.body(observations)
        indices = torch.arange(len(hidden), device=hidden.device)
        means = self.output(hidden).reshape(-1, 2, ACTION_SIZE)[indices, heads]
        return Normal(means, self.log_std.clamp(-3., .2).exp()[heads])


class ValueNetwork(nn.Module):
    def __init__(self, obs_size):
        super().__init__()
        self.body = _body(obs_size)
        self.output = nn.Linear(64, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, observations):
        return self.output(self.body(observations))


class BoundaryNetwork(nn.Module):
    def __init__(self, obs_size=OBS_SIZE):
        super().__init__()
        self.actor = Actor(obs_size)
        self.reward_critic = ValueNetwork(obs_size)
        self.cost_critic = ValueNetwork(obs_size)

    @property
    def log_std(self):
        return self.actor.log_std

    def values(self, observations):
        return torch.cat((self.reward_critic(observations), self.cost_critic(observations)), dim=1)

    def forward(self, observations, heads):
        return self.actor(observations, heads), self.values(observations)


def warmup_return_scales(rows):
    targets = calculate_gae(rows)['targets'].astype(np.float64)
    return np.maximum(1., targets.std(axis=0))


class SafePPO:
    def __init__(self, seed=0, scaler=None, obs_size=OBS_SIZE, return_scales=(1., 1.)):
        torch.set_num_threads(1)
        torch.manual_seed(int(seed))
        self.obs_size = int(obs_size)
        self.rng = np.random.default_rng(seed)
        self.network = BoundaryNetwork(self.obs_size)
        self.scaler = FixedScaler(self.obs_size)
        if isinstance(scaler, FixedScaler):
            self.scaler.restore(scaler.state())
        elif scaler is not None:
            self.scaler.restore(scaler)
        self.return_scales = self._validate_scales(return_scales)
        self.actor_optimizer = torch.optim.Adam(self.network.actor.parameters(), lr=2e-4)
        self.critic_optimizer = torch.optim.Adam(chain(self.network.reward_critic.parameters(),
                                                       self.network.cost_critic.parameters()), lr=2e-4)
        self.multiplier = 1.
        self.total_updates = 0
        self.total_actor_optimizer_steps = 0
        self.total_critic_optimizer_steps = 0

    @staticmethod
    def _validate_scales(scales):
        scales = np.asarray(scales, np.float64)
        if scales.shape != (2,) or not np.isfinite(scales).all() or np.any(scales < 1.):
            raise ValueError('Use two frozen finite warmup return scales, each at least one')
        return scales.copy()

    @property
    def total_optimizer_steps(self):
        return self.total_actor_optimizer_steps+self.total_critic_optimizer_steps

    def act(self, observation, head, deterministic=False):
        if int(head) not in (0, 1):
            raise ValueError('Throw head is0; Push head is1')
        normalized = self.scaler.normalize(observation)
        if normalized.shape != (self.obs_size,):
            raise ValueError(f'Expected one{self.obs_size}-feature boundary observation')
        with torch.no_grad():
            distribution, values = self.network(torch.from_numpy(normalized[None]), torch.tensor([int(head)]))
            latent = distribution.mean if deterministic else distribution.sample()
            action = torch.tanh(latent)
            logp = (distribution.log_prob(latent)-torch.log(1.-action.square()+1e-6)).sum(-1)
        return action.numpy()[0], latent.numpy()[0], float(logp.item()), values.numpy()[0], normalized

    def value(self, observation, head):
        return self.act(observation, head, deterministic=True)[3]

    def update(self, rows, episode_summaries, budget):
        if not np.isfinite(budget) or budget < 0 or not episode_summaries:
            raise ValueError('A finite nonnegative workload budget and full episode summaries are required')
        episode_costs = []
        for summary in episode_summaries:
            offered, cost = int(summary['offered']), float(summary['total_cost'])
            if offered <= 0 or not np.isfinite(cost) or cost < 0:
                raise ValueError('Every full episode needs a positive offered count and nonnegative total cost')
            episode_costs.append(cost/offered)
        gae = calculate_gae(rows)
        n = len(rows)
        obs = torch.as_tensor(np.asarray([row['obs'] for row in rows]), dtype=torch.float32)
        latent = torch.as_tensor(np.asarray([row['latent'] for row in rows]), dtype=torch.float32)
        heads = torch.as_tensor([row['head'] for row in rows], dtype=torch.long)
        old_logp = torch.as_tensor([row['logp'] for row in rows], dtype=torch.float32)
        if (obs.shape != (n, self.obs_size) or latent.shape != (n, ACTION_SIZE)
                or torch.any((heads < 0) | (heads > 1))
                or not all(torch.isfinite(value).all() for value in (obs, latent, old_logp))):
            raise ValueError('Invalid on-policy boundary batch')
        action = torch.tanh(latent)
        combined = gae['advantages'][:, 0]-self.multiplier*gae['advantages'][:, 1]
        standardized = (combined-combined.mean())/(combined.std()+1e-8)
        advantage = torch.as_tensor(standardized, dtype=torch.float32)
        targets = torch.from_numpy(gae['targets'])
        scale = torch.as_tensor(self.return_scales, dtype=torch.float32)
        logs = {key:[] for key in ('policy_loss', 'critic_loss', 'entropy', 'kl',
            'actor_grad_norm', 'actor_grad_norm_after_clip', 'critic_grad_norm',
            'critic_grad_norm_after_clip', 'reward_value_mse', 'cost_value_mse')}
        steps = 0
        critic_parameters = list(chain(self.network.reward_critic.parameters(), self.network.cost_critic.parameters()))
        for epoch in range(6):
            epoch_kls = []
            for start in range(0, n, 128):
                if start == 0:
                    permutation = self.rng.permutation(n)
                indices = permutation[start:start+128]
                distribution = self.network.actor(obs[indices], heads[indices])
                logp = (distribution.log_prob(latent[indices])-torch.log(1.-action[indices].square()+1e-6)).sum(-1)
                log_ratio = logp-old_logp[indices]
                ratio = log_ratio.exp()
                policy_loss = torch.maximum(-advantage[indices]*ratio,
                    -advantage[indices]*ratio.clamp(.8, 1.2)).mean()
                entropy = distribution.entropy().sum(-1).mean()
                actor_loss = policy_loss-.001*entropy
                self.actor_optimizer.zero_grad()
                if not torch.isfinite(actor_loss):
                    raise FloatingPointError('Nonfinite actor loss')
                actor_loss.backward()
                actor_norm = float(nn.utils.clip_grad_norm_(self.network.actor.parameters(), .5))
                self.actor_optimizer.step()
                values = self.network.values(obs[indices])
                error = values-targets[indices]
                raw_mse = error.square().mean(dim=0)
                critic_loss = (error/scale).square().mean()
                self.critic_optimizer.zero_grad()
                if not torch.isfinite(critic_loss):
                    raise FloatingPointError('Nonfinite critic loss')
                critic_loss.backward()
                critic_norm = float(nn.utils.clip_grad_norm_(critic_parameters, .5))
                self.critic_optimizer.step()
                kl = float(((ratio-1.)-log_ratio).mean().detach())
                epoch_kls.append(kl)
                for key, value in dict(policy_loss=policy_loss, critic_loss=critic_loss,
                    entropy=entropy, kl=kl, actor_grad_norm=actor_norm,
                    actor_grad_norm_after_clip=min(actor_norm, .5), critic_grad_norm=critic_norm,
                    critic_grad_norm_after_clip=min(critic_norm, .5),
                    reward_value_mse=raw_mse[0], cost_value_mse=raw_mse[1]).items():
                    logs[key].append(float(value.detach()) if isinstance(value, torch.Tensor) else float(value))
                steps += 1
            if float(np.mean(epoch_kls)) > .02:
                break
        cost_per_offered = float(np.mean(episode_costs))
        multiplier_before = self.multiplier
        self.multiplier = float(np.clip(self.multiplier+.5*(cost_per_offered-budget)/max(budget, .1), 0., 30.))
        self.total_updates += 1
        self.total_actor_optimizer_steps += steps
        self.total_critic_optimizer_steps += steps
        return dict(**{key:float(np.mean(value)) for key, value in logs.items()},
            actor_grad_norm_max=max(logs['actor_grad_norm']), critic_grad_norm_max=max(logs['critic_grad_norm']),
            epochs=epoch+1, optimizer_steps=2*steps, actor_optimizer_steps=steps, critic_optimizer_steps=steps,
            total_optimizer_steps=self.total_optimizer_steps,
            total_actor_optimizer_steps=self.total_actor_optimizer_steps,
            total_critic_optimizer_steps=self.total_critic_optimizer_steps,
            total_updates=self.total_updates, decisions=n, full_episodes=len(episode_summaries),
            cost_per_offered=cost_per_offered, cost_budget_per_offered=float(budget),
            mean_cost_per_decision_diagnostic=float(np.mean([row['cost'] for row in rows])),
            mean_reward=float(np.mean([row['reward'] for row in rows])),
            multiplier_before=multiplier_before, multiplier=self.multiplier,
            reward_advantage_std=float(gae['advantages'][:,0].std()),
            cost_advantage_std=float(gae['advantages'][:,1].std()),
            combined_advantage_std=float(combined.std()), normalized_actor_advantage_std=float(standardized.std()),
            raw_return_mean=gae['targets'].mean(axis=0).tolist(),
            raw_return_std=gae['targets'].std(axis=0).tolist(), frozen_return_scales=self.return_scales.tolist(),
            std=self.network.log_std.detach().clamp(-3., .2).exp().numpy().tolist())

    def save(self, path, metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(format_version=2, obs_size=self.obs_size, action_size=ACTION_SIZE,
            network=self.network.state_dict(), scaler=self.scaler.state(), return_scales=self.return_scales,
            actor_optimizer=self.actor_optimizer.state_dict(), critic_optimizer=self.critic_optimizer.state_dict(),
            numpy_rng=self.rng.bit_generator.state, torch_rng=torch.get_rng_state(), multiplier=self.multiplier,
            total_updates=self.total_updates, total_actor_optimizer_steps=self.total_actor_optimizer_steps,
            total_critic_optimizer_steps=self.total_critic_optimizer_steps, metadata=metadata), path)

    def load(self, path):
        state = torch.load(path, map_location='cpu', weights_only=False)
        if (state.get('format_version') != 2 or int(state['obs_size']) != self.obs_size
                or int(state['action_size']) != ACTION_SIZE):
            raise ValueError('Checkpoint does not belong to this v2 policy architecture')
        self.network.load_state_dict(state['network'])
        self.scaler.restore(state['scaler'])
        self.return_scales = self._validate_scales(state['return_scales'])
        self.actor_optimizer.load_state_dict(state['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state['critic_optimizer'])
        self.rng.bit_generator.state = state['numpy_rng']
        torch.set_rng_state(state['torch_rng'])
        self.multiplier = float(state['multiplier'])
        self.total_updates = int(state['total_updates'])
        self.total_actor_optimizer_steps = int(state['total_actor_optimizer_steps'])
        self.total_critic_optimizer_steps = int(state['total_critic_optimizer_steps'])
        return state.get('metadata', {})

    def rollout_state(self):
        return dict(format_version=2, obs_size=self.obs_size, action_size=ACTION_SIZE,
            network={key:value.detach().numpy().copy() for key,value in self.network.state_dict().items()},
            scaler=self.scaler.state(), return_scales=self.return_scales.copy(), multiplier=self.multiplier,
            total_updates=self.total_updates, total_actor_optimizer_steps=self.total_actor_optimizer_steps,
            total_critic_optimizer_steps=self.total_critic_optimizer_steps)

    def restore_rollout_state(self, state):
        if state.get('format_version') != 2 or int(state['obs_size']) != self.obs_size:
            raise ValueError('Rollout state is not this v2 policy architecture')
        self.network.load_state_dict({key:torch.as_tensor(value) for key,value in state['network'].items()})
        self.scaler.restore(state['scaler'])
        self.return_scales = self._validate_scales(state['return_scales'])
        self.multiplier = float(state['multiplier'])
        self.total_updates = int(state['total_updates'])
        self.total_actor_optimizer_steps = int(state['total_actor_optimizer_steps'])
        self.total_critic_optimizer_steps = int(state['total_critic_optimizer_steps'])

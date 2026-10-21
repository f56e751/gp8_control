"""PPO-Lagrangian port for native feature/simulator residual decisions.

Unlike the fixed-target contextual bandit, the two primitive heads share one
reward/cost value function. Costs caused after one primitive therefore influence
the advantages of preceding decisions, including decisions from the other head.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


OBS_SIZE = 97
ACTION_SIZE = 6


class FixedScaler:
    """Fit actual stream warmup observations once, then keep their scale fixed."""

    def __init__(self, obs_size=OBS_SIZE, minimum_std=.1):
        if obs_size <= 0 or not np.isfinite(minimum_std) or minimum_std <= 0:
            raise ValueError('A positive feature count and standard-deviation floor are required')
        self.minimum_std = float(minimum_std)
        self.mean = np.zeros(int(obs_size), np.float64)
        self.std = np.ones(int(obs_size), np.float64)
        self.count = 0
        self.frozen = False

    def fit(self, observations):
        if self.frozen:
            raise RuntimeError('Warmup scaler is frozen; refitting would invalidate on-policy log probabilities')
        values = np.asarray(observations, np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.mean) or not len(values) or not np.isfinite(values).all():
            raise ValueError('Expected a nonempty finite matrix of warmup observations')
        self.mean = values.mean(axis=0)
        self.std = np.maximum(values.std(axis=0), self.minimum_std)
        self.count = len(values)
        self.frozen = True
        return self

    def normalize(self, observation):
        value = np.asarray(observation, np.float64)
        if value.shape[-1:] != self.mean.shape or not np.isfinite(value).all():
            raise ValueError('Observation has the wrong size or contains nonfinite values')
        return np.clip((value-self.mean)/self.std, -10., 10.).astype(np.float32)

    def state(self):
        return dict(mean=self.mean.copy(), std=self.std.copy(), count=self.count,
            minimum_std=self.minimum_std, frozen=self.frozen)

    def restore(self, state):
        mean, std = np.asarray(state['mean'], np.float64), np.asarray(state['std'], np.float64)
        minimum = float(state['minimum_std'])
        if (mean.ndim != 1 or std.shape != mean.shape or not np.isfinite(mean).all()
                or not np.isfinite(std).all() or not np.isfinite(minimum) or minimum <= 0
                or np.any(std < minimum) or int(state['count']) < 0):
            raise ValueError('Invalid frozen-scaler state')
        self.mean, self.std = mean.copy(), std.copy()
        self.minimum_std, self.count, self.frozen = minimum, int(state['count']), bool(state['frozen'])


class StreamNetwork(nn.Module):
    def __init__(self, obs_size=OBS_SIZE):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(int(obs_size), 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.actor = nn.Linear(64, 2*ACTION_SIZE)
        self.critic = nn.Linear(64, 2)
        self.log_std = nn.Parameter(torch.full((2, ACTION_SIZE), -1.5))
        for layer in self.body:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, np.sqrt(2.))
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.actor.weight)
        nn.init.zeros_(self.actor.bias)
        nn.init.zeros_(self.critic.weight)
        nn.init.zeros_(self.critic.bias)

    def forward(self, observations, heads):
        hidden = self.body(observations)
        indices = torch.arange(len(hidden), device=hidden.device)
        means = self.actor(hidden).reshape(-1, 2, ACTION_SIZE)[indices, heads]
        std = self.log_std.clamp(-3., .2).exp()[heads]
        return Normal(means, std), self.critic(hidden)


def calculate_gae(rows, gamma=.99, lam=.95):
    """Time-discounted reward/cost GAE with correct episode boundary handling.

    A true terminal has zero bootstrap. A rollout/safety cap marked truncated
    requires the value of its actual continuation state in ``next_value``.
    Recurrence never crosses an episode reset, even for truncation. Legacy
    ``done`` is accepted only as a true terminal for archived data.
    """
    if not rows:
        raise ValueError('GAE requires at least one decision')
    if not 0 < gamma <= 1 or not 0 <= lam <= 1:
        raise ValueError('Invalid GAE discount or lambda')
    values = np.asarray([row['value'] for row in rows], np.float64)
    signals = np.asarray([[row['reward'], row['cost']] for row in rows], np.float64)
    durations = np.asarray([row['duration'] for row in rows], np.float64)
    if (values.shape != (len(rows), 2) or not np.isfinite(values).all()
            or not np.isfinite(signals).all() or not np.isfinite(durations).all() or np.any(durations < 0)):
        raise ValueError('GAE requires finite two-component values and nonnegative durations')
    completed = set()
    boundaries, bootstraps = [], []
    for i, row in enumerate(rows):
        episode = row['episode_id']
        if episode in completed:
            raise ValueError('Episode decisions are not chronological/contiguous')
        terminated = bool(row.get('terminated', row.get('done', False)))
        truncated = bool(row.get('truncated', False))
        if terminated and truncated:
            raise ValueError('An episode cannot be both terminated and truncated')
        boundary = terminated or truncated
        same_next = i+1 < len(rows) and rows[i+1]['episode_id'] == episode
        if boundary:
            completed.add(episode)
            if same_next:
                raise ValueError('An ended episode continues with the same episode ID')
        elif not same_next:
            raise ValueError('Collect complete episodes or mark the collection boundary truncated')
        if terminated:
            next_value = np.zeros(2, np.float64)
        elif same_next:
            next_value = values[i+1]
        else:
            if 'next_value' not in row:
                raise ValueError('Truncation requires its actual next-state value')
            next_value = np.asarray(row['next_value'], np.float64)
            if next_value.shape != (2,) or not np.isfinite(next_value).all():
                raise ValueError('Invalid truncation bootstrap value')
        boundaries.append(boundary)
        bootstraps.append(next_value)
    discounts = np.power(float(gamma), durations)
    advantages = np.zeros_like(values)
    following = np.zeros(2, np.float64)
    for i in range(len(rows)-1, -1, -1):
        delta = signals[i] + discounts[i]*bootstraps[i] - values[i]
        following = delta + (0. if boundaries[i] else discounts[i]*float(lam)*following)
        advantages[i] = following
    return dict(advantages=advantages.astype(np.float32),
        targets=(advantages+values).astype(np.float32), discounts=discounts)


class SafePPO:
    def __init__(self, seed=0, scaler=None, obs_size=OBS_SIZE):
        torch.set_num_threads(1)
        torch.manual_seed(int(seed))
        self.rng = np.random.default_rng(seed)
        self.obs_size = int(obs_size)
        self.network = StreamNetwork(self.obs_size)
        self.scaler = FixedScaler(self.obs_size)
        if isinstance(scaler, FixedScaler):
            self.scaler.restore(scaler.state())
        elif scaler is not None:
            self.scaler.restore(scaler)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=2e-4)
        self.multiplier = 1.
        self.total_optimizer_steps = 0
        self.total_updates = 0

    def act(self, observation, head, deterministic=False):
        if int(head) not in (0, 1):
            raise ValueError('Throw head is 0 and Push/route head is 1')
        normalized = self.scaler.normalize(observation)
        if normalized.shape != (self.obs_size,):
            raise ValueError(f'act expects one {self.obs_size}-feature observation')
        with torch.no_grad():
            distribution, value = self.network(torch.from_numpy(normalized[None]), torch.tensor([int(head)]))
            latent = distribution.mean if deterministic else distribution.sample()
            action = torch.tanh(latent)
            logp = (distribution.log_prob(latent)-torch.log(1-action.square()+1e-6)).sum(-1)
        return action.numpy()[0], latent.numpy()[0], float(logp.item()), value.numpy()[0], normalized

    def value(self, observation, head):
        """Evaluate an actual continuation state without sampling an action."""
        return self.act(observation, head, deterministic=True)[3]

    def update(self, rows, cost_limit=.10):
        if not np.isfinite(cost_limit) or cost_limit < 0:
            raise ValueError('Cost limit must be finite and nonnegative')
        gae = calculate_gae(rows)
        observations = torch.as_tensor(np.asarray([r['obs'] for r in rows]), dtype=torch.float32)
        latent = torch.as_tensor(np.asarray([r['latent'] for r in rows]), dtype=torch.float32)
        heads = torch.as_tensor([r['head'] for r in rows], dtype=torch.long)
        old_logp = torch.as_tensor([r['logp'] for r in rows], dtype=torch.float32)
        if (observations.shape != (len(rows), self.obs_size) or latent.shape != (len(rows), ACTION_SIZE)
                or torch.any((heads < 0) | (heads > 1)) or not torch.isfinite(observations).all()
                or not torch.isfinite(latent).all() or not torch.isfinite(old_logp).all()):
            raise ValueError('Invalid on-policy rollout batch')
        action = torch.tanh(latent)
        combined = gae['advantages'][:, 0] - self.multiplier*gae['advantages'][:, 1]
        advantage = torch.as_tensor((combined-combined.mean())/(combined.std()+1e-8), dtype=torch.float32)
        targets = torch.from_numpy(gae['targets'])
        losses, policy_losses, value_losses, entropies, kls = [], [], [], [], []
        epochs = steps = 0
        for epoch in range(6):
            epoch_kls = []
            permutation = self.rng.permutation(len(rows))
            for start in range(0, len(rows), 128):
                indices = permutation[start:start+128]
                distribution, values = self.network(observations[indices], heads[indices])
                logp = (distribution.log_prob(latent[indices])-torch.log(1-action[indices].square()+1e-6)).sum(-1)
                delta = logp-old_logp[indices]
                ratio = delta.exp()
                policy_loss = torch.maximum(-advantage[indices]*ratio,
                    -advantage[indices]*ratio.clamp(.8, 1.2)).mean()
                value_loss = (values-targets[indices]).square().mean()
                entropy = distribution.entropy().sum(-1).mean()
                loss = policy_loss+.25*value_loss-.001*entropy
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite StreamPPO loss')
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), .5)
                self.optimizer.step()
                steps += 1
                losses.append(float(loss.detach()))
                policy_losses.append(float(policy_loss.detach()))
                value_losses.append(float(value_loss.detach()))
                entropies.append(float(entropy.detach()))
                kl = float(((ratio-1)-delta).mean().detach())
                kls.append(kl)
                epoch_kls.append(kl)
            epochs = epoch+1
            if float(np.mean(epoch_kls)) > .02:
                break
        mean_cost = float(np.mean([row['cost'] for row in rows]))
        self.multiplier = float(np.clip(self.multiplier+.2*(mean_cost-cost_limit), 0., 30.))
        self.total_optimizer_steps += steps
        self.total_updates += 1
        return dict(loss=float(np.mean(losses)), policy_loss=float(np.mean(policy_losses)),
            value_loss=float(np.mean(value_losses)), entropy=float(np.mean(entropies)),
            kl=float(np.mean(kls)), final_epoch_kl=float(np.mean(epoch_kls)),
            epochs=epochs, optimizer_steps=steps, total_optimizer_steps=self.total_optimizer_steps,
            total_updates=self.total_updates, decisions=len(rows), mean_cost=mean_cost,
            mean_reward=float(np.mean([row['reward'] for row in rows])), multiplier=self.multiplier,
            std=self.network.log_std.detach().clamp(-3., .2).exp().numpy().tolist())

    def save(self, path, metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(obs_size=self.obs_size, network=self.network.state_dict(), scaler=self.scaler.state(),
            optimizer=self.optimizer.state_dict(), numpy_rng=self.rng.bit_generator.state,
            torch_rng=torch.get_rng_state(), multiplier=self.multiplier,
            total_optimizer_steps=self.total_optimizer_steps, total_updates=self.total_updates,
            metadata=metadata), path)

    def load(self, path):
        state = torch.load(path, map_location='cpu', weights_only=False)
        if int(state.get('obs_size', len(state['scaler']['mean']))) != self.obs_size:
            raise ValueError('Checkpoint observation width differs from this policy')
        self.network.load_state_dict(state['network'])
        self.scaler.restore(state['scaler'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.rng.bit_generator.state = state['numpy_rng']
        torch.set_rng_state(state['torch_rng'])
        self.multiplier = float(state['multiplier'])
        self.total_optimizer_steps = int(state['total_optimizer_steps'])
        self.total_updates = int(state['total_updates'])
        return state.get('metadata', {})

    def rollout_state(self):
        return dict(obs_size=self.obs_size, network={key: value.detach().numpy().copy() for key, value in self.network.state_dict().items()},
            scaler=self.scaler.state(), multiplier=self.multiplier,
            total_optimizer_steps=self.total_optimizer_steps, total_updates=self.total_updates)

    def restore_rollout_state(self, state):
        if int(state.get('obs_size', len(state['scaler']['mean']))) != self.obs_size:
            raise ValueError('Rollout observation width differs from this policy')
        self.network.load_state_dict({key: torch.as_tensor(value) for key, value in state['network'].items()})
        self.scaler.restore(state['scaler'])
        self.multiplier = float(state.get('multiplier', 1.))
        self.total_optimizer_steps = int(state.get('total_optimizer_steps', 0))
        self.total_updates = int(state.get('total_updates', 0))


# Name retained for the original trajectory adapter interface.
StreamPPO = SafePPO

"""Re-encode the flat RL observation inside the policy network (MlpPolicy).

Per object: class_id -> one-hot, plus an is_pending flag. Global: joints ->
(sin, cos), pending_skill -> 2-dim one-hot (none = 0), remaining time ->
sinusoidal embedding of the seconds. Other entries pass through unchanged.
"""

from __future__ import annotations

import math

import torch as th
from torch import nn
from torch.nn import functional as F

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

N_GLOBAL = 13  # joints 6, ee 3, pending_slot, pending_skill, belt_speed, remaining_time
N_CLASSES = 3  # metal, transparent, cardboard (rl/common.py CLASS_IDS)
TIME_EMB_DIM = 16


class SinusoidalPosEmb(nn.Module):
    """1D sinusoidal positional embeddings as in Attention is All You Need."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: th.Tensor) -> th.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = th.exp(th.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = th.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ProcessedStateExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, max_objects: int = 6, episode_seconds: float = 240.0) -> None:
        obs_dim = observation_space.shape[0]
        obj_dim = (obs_dim - N_GLOBAL) // max_objects
        # Per object: class_id (1 column) becomes a one-hot, plus is_pending.
        obj_out = obj_dim - 1 + N_CLASSES + 1
        glob_out = 12 + 3 + 2 + 1 + TIME_EMB_DIM
        super().__init__(observation_space, max_objects * obj_out + glob_out)
        self.n, self.obj_dim = int(max_objects), obj_dim
        self.episode_seconds = float(episode_seconds)
        self.time_emb = SinusoidalPosEmb(TIME_EMB_DIM)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        b, n = obs.shape[0], self.n
        objs = obs[:, : n * self.obj_dim].view(b, n, self.obj_dim)
        g = obs[:, n * self.obj_dim :]
        cls = objs[..., 3].round().long()  # -1 = empty slot -> all-zero one-hot
        cls_onehot = F.one_hot((cls + 1).clamp(min=0), N_CLASSES + 1)[..., 1:].float()
        slot, skill = g[:, 9].round().long(), g[:, 10]
        is_pending = (th.arange(n, device=obs.device) == slot[:, None]).float()[..., None]
        obj_feat = th.cat([objs[..., :3], cls_onehot, objs[..., 4:], is_pending], dim=-1)
        has = (slot < n).float()
        pending = th.stack([has * (skill == 0).float(), has * (skill == 1).float()], dim=1)
        q = g[:, :6]
        glob = th.cat(
            [q.sin(), q.cos(), g[:, 6:9], pending, g[:, 11:12],
             self.time_emb(g[:, 12] * self.episode_seconds)],
            dim=1,
        )
        return th.cat([obj_feat.flatten(1), glob], dim=1)

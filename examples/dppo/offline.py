"""Offline preference-based policy learning (DPPO) for the GP8 recycling env.

Discrete-action port of DPPO (Direct Preference-based Policy Optimization,
snu-mllab/DPPO). Logged rollouts are cut into short segments, every segment gets
a preference score, and the policy is trained to stay close to preferred
segments and away from less preferred ones:

  1. ``load_segments`` cuts the rollouts. A decision made at step t is executed
     at step t+1, so a segment pairs the rewards of steps [i, i+H) with the
     decisions made at [i-1, i+H-1) — the decisions executed in that window.
     Reward values are never changed.
  2. ``fit_preference_scores`` trains a preference predictor on return-labelled
     segment pairs and scores every segment with it (cross-fitted by episode).
     ``fit_score_baseline`` is the simpler alternative: segment return minus what
     the segment's first observation predicts.
  3. ``dppo_loss`` is the ranked pairwise loss of the original, with the mean
     masked NLL of the logged actions as the segment distance.

The policy consumes exactly the ``GP8RecyclingEnv`` observation and the flat
``(object_slot, skill)`` action of ``FlatMaskedActionWrapper``.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F

from gp8_control.rl.common import SKILL_NAMES, empty_object_row, observation_width

POLICY_KEYS = ("max_objects", "bbox_observation", "include_suction_p", "hidden")
CLASS_COL = 3   # per-object column holding the class id (rl/common.py::build_observation)


def policy_kwargs(args) -> dict:
    """``OfflinePolicy`` constructor arguments from parsed command-line arguments."""
    return {key: getattr(args, key) for key in POLICY_KEYS}


def check_obs_width(obs: th.Tensor, args) -> None:
    """Fail early if the data was collected with other observation flags."""
    expected = observation_width(args.max_objects, False, args.bbox_observation,
                                 args.include_suction_p)
    if obs.shape[-1] != expected:
        raise SystemExit(
            f"data obs width {obs.shape[-1]} != {expected}; pass the observation flags "
            "(--max-objects / --bbox-observation / --include-suction-p) the data was "
            "collected with"
        )


def rollout_files(pattern: str) -> list[str]:
    """Episode files matching ``pattern``, in the order that defines episode indices."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no rollout files match {pattern!r}")
    return files


class OfflinePolicy(nn.Module):
    """Masked categorical policy over the flat ``(object_slot, skill)`` actions.

    A shared per-object embedding and one global token go through a 3-layer
    pre-LN Transformer without positional encoding; the per-slot logits are read
    from the object tokens and the skip logits from the global token.

    ``critic`` keeps the same body but reads all action values from the global
    token; ``scalar`` reads a single value from it (state value / baseline).
    """

    def __init__(
        self,
        *,
        max_objects: int = 6,
        bbox_observation: str = "corners",
        include_suction_p: bool = False,
        hidden: int = 256,
        scalar: bool = False,
        critic: bool = False,
    ) -> None:
        super().__init__()
        self.meta = dict(
            max_objects=int(max_objects),
            bbox_observation=str(bbox_observation),
            include_suction_p=bool(include_suction_p),
            hidden=int(hidden),
        )
        self.scalar = bool(scalar)
        self.critic = bool(critic)
        self.max_objects = int(max_objects)
        self.obs_dim = observation_width(
            max_objects, False, bbox_observation, include_suction_p
        )
        self.n_actions = (int(max_objects) + 1) * len(SKILL_NAMES)
        # Observation = max_objects object rows, then the global features.
        self.obj_dim = len(empty_object_row(False, bbox_observation, include_suction_p))
        n_global = self.obs_dim - self.max_objects * self.obj_dim
        d = hidden
        self.obj_embed = nn.Sequential(
            nn.Linear(self.obj_dim, d), nn.GELU(), nn.Linear(d, d))
        self.global_embed = nn.Sequential(
            nn.Linear(n_global, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=8, dim_feedforward=4 * d, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=3, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(d)
        if scalar or critic:
            self.scene_mlp = nn.Sequential(
                nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1 if scalar else self.n_actions))
        else:
            self.slot_head = nn.Linear(d, len(SKILL_NAMES))
            self.skip_head = nn.Linear(d, len(SKILL_NAMES))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=2.0 ** 0.5)
                nn.init.zeros_(module.bias)

    def outputs(self, obs: th.Tensor) -> th.Tensor:
        """Unmasked per-action outputs ``(B, n_actions)`` (``(B, 1)`` if scalar)."""
        b, n = obs.shape[0], self.max_objects
        objs = obs[:, : n * self.obj_dim].view(b, n, self.obj_dim)
        present = objs[..., CLASS_COL] >= 0             # class id -1 = empty slot
        tokens = th.cat(
            [self.global_embed(obs[:, n * self.obj_dim:])[:, None], self.obj_embed(objs)],
            dim=1,
        )
        keep = th.ones(b, 1, dtype=th.bool, device=obs.device)
        h = self.out_norm(
            self.encoder(tokens, src_key_padding_mask=~th.cat([keep, present], dim=1))
        )
        scene, obj_tokens = h[:, 0], h[:, 1:]
        if self.scalar or self.critic:
            return self.scene_mlp(scene)
        return th.cat([self.slot_head(obj_tokens).flatten(1), self.skip_head(scene)], dim=1)

    def logits(self, obs: th.Tensor, mask: th.Tensor) -> th.Tensor:
        return self.outputs(obs).masked_fill(~mask.bool(), -1e9)

    @th.no_grad()
    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        """Greedy flat action for one observation and its canonical mask."""
        obs_t = th.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        mask_t = th.as_tensor(np.asarray(mask, dtype=bool)).unsqueeze(0)
        # argmax of the log-probabilities, not of the raw logits: the same action
        # except on exact ties, and what the reported evaluations used.
        logp = F.log_softmax(self.logits(obs_t, mask_t), dim=-1)
        return int(logp.argmax(dim=-1).item())

    def save(self, path: str | Path, **extra) -> None:
        th.save({"model": self.state_dict(), "meta": self.meta, **extra}, str(path))

    @staticmethod
    def load_meta(path: str | Path) -> dict:
        """Constructor arguments stored in a checkpoint, without building the network."""
        meta = th.load(str(path), map_location="cpu", weights_only=False)["meta"]
        return {key: meta[key] for key in POLICY_KEYS}

    @classmethod
    def load(cls, path: str | Path) -> "OfflinePolicy":
        ckpt = th.load(str(path), map_location="cpu", weights_only=False)
        policy = cls(**{key: ckpt["meta"][key] for key in POLICY_KEYS})
        policy.load_state_dict(ckpt["model"])
        policy.eval()
        return policy


def segment_nll(policy: OfflinePolicy, obs, act, mask) -> th.Tensor:
    """Mean per-step NLL of the logged actions. ``(B, H, ...)`` -> ``(B,)``."""
    b, h = act.shape
    logp = F.log_softmax(policy.logits(obs.flatten(0, 1), mask.flatten(0, 1)), dim=-1)
    nll = -logp.gather(1, act.reshape(-1, 1)).squeeze(1).view(b, h)
    # Divide by a tensor, not by the Python number: on CUDA a scalar divisor is
    # applied as a multiplication by its reciprocal, which is one bit off for h = 3
    # and would not reproduce the reported runs.
    return nll.sum(dim=1) / nll.new_full((b,), float(h))


def dppo_loss(dists: th.Tensor, scores: th.Tensor, lambd: float, bc_anchor: float,
              margin: float = 0.0) -> th.Tensor:
    """Ranked pairwise DPPO loss over one batch of segments.

    For every pair with segment i preferred over j:
        -log[ e^{-d_i} / (e^{-d_i} + e^{-lambd * d_j}) ]
    evaluated in log space. ``bc_anchor * mean(d)`` bounds the NLL distance,
    which (unlike the original's bounded L2 distance) can otherwise drift
    without limit because the ranking term only sees distance differences.
    """
    # The pair mask below does not need the sort; it is kept so the terms are summed
    # in the same order as in the runs behind the reported numbers.
    order = th.argsort(scores, descending=True)
    ds, ss = dists[order], scores[order]
    di = ds.unsqueeze(1).expand(-1, ds.shape[0])
    dj = ds.unsqueeze(0).expand(ds.shape[0], -1)
    pair_logp = -di - th.logsumexp(th.stack([-di, -lambd * dj]), dim=0)
    # Pairs whose scores differ by no more than ``margin`` carry no preference
    # (the "equally preferable" answer of a PbRL teacher); margin 0 = strict order.
    preferred = (ss.unsqueeze(1) - ss.unsqueeze(0)) > margin
    if not bool(preferred.any()):
        return bc_anchor * dists.mean()
    return -pair_logp[preferred].mean() + bc_anchor * dists.mean()


def load_segments(pattern: str, *, seg_len: int = 3, stride: int = 1,
                  decision_lag: int = 1) -> dict:
    """Cut per-episode rollout files (collect_offline_data.py) into segments.

    A segment is ``seg_len`` consecutive steps of one episode. Its return is the
    sum of the env rewards of steps [i, i + seg_len), exactly as ``env.step``
    returned them. Its observations and actions are the decisions made
    ``decision_lag`` steps earlier: the env executes a chosen action one step
    after it is chosen, so with ``decision_lag=1`` a segment holds the decisions
    that were EXECUTED inside its reward window.
    """
    obs_l, act_l, mask_l, ret_l, ep_l = [], [], [], [], []
    files = rollout_files(pattern)
    steps = 0
    k = decision_lag
    for ep_index, path in enumerate(files):
        d = np.load(path)
        obs, act, mask, rew = d["obs"], d["act"], d["mask"], d["rew"]
        steps += len(act)
        starts = [s for s in range(0, len(act) - seg_len + 1, stride) if s >= k]
        if not starts:
            continue
        for i in starts:
            obs_l.append(obs[i - k:i - k + seg_len])
            act_l.append(act[i - k:i - k + seg_len])
            mask_l.append(mask[i - k:i - k + seg_len])
        ret_l.append(np.array([rew[i:i + seg_len].sum() for i in starts], dtype=np.float32))
        ep_l.append(np.full(len(starts), ep_index))
    return {
        "obs": th.as_tensor(np.stack(obs_l).astype(np.float32)),
        "act": th.as_tensor(np.stack(act_l).astype(np.int64)),
        "mask": th.as_tensor(np.stack(mask_l).astype(bool)),
        "episode": np.concatenate(ep_l),
        "ret": th.as_tensor(np.concatenate(ret_l)),
        "episodes": len(files),
        "steps": steps,
    }


def fit_score_baseline(
    data: dict,
    policy_kwargs: dict,
    *,
    folds: int = 5,
    steps: int = 3000,
    batch_size: int = 256,
    lr: float = 3e-4,
    seed: int = 0,
    device: str = "cpu",
) -> th.Tensor:
    """Expected segment return given the segment's FIRST observation, cross-fitted.

    A segment's return depends as much on what was on the belt when it started
    (and on the action already pending then) as on the decisions made inside it.
    Subtracting this baseline leaves the part the segment's own decisions are
    responsible for, so the ranking compares segments that started alike. Each
    fold's predictor is fitted on the other folds' episodes, so a segment never
    sees its own return.
    """
    gen = th.Generator().manual_seed(seed)
    first_obs, ret = data["obs"][:, 0], data["ret"]
    fold_of = th.as_tensor(data["episode"] % folds)
    baseline = th.zeros_like(ret)
    for fold in range(folds):
        train = (fold_of != fold).nonzero().squeeze(1)
        net = OfflinePolicy(**policy_kwargs, scalar=True)
        net.to(device)
        opt = th.optim.Adam(net.parameters(), lr=lr)
        for _ in range(steps):
            idx = train[th.randint(0, len(train), (batch_size,), generator=gen)]
            pred = net.outputs(first_obs[idx].to(device)).squeeze(1)
            loss = F.mse_loss(pred, ret[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        held = (fold_of == fold).nonzero().squeeze(1)
        with th.no_grad():
            baseline[held] = net.outputs(first_obs[held].to(device)).squeeze(1).cpu()
    return baseline


def fit_preference_scores(
    data: dict,
    policy_kwargs: dict,
    *,
    folds: int = 5,
    steps: int = 6000,
    batch_size: int = 256,
    lr: float = 3e-4,
    label_margin: float = 0.5,
    seed: int = 0,
    device: str = "cpu",
) -> th.Tensor:
    """Segment scores from a learned preference predictor, cross-fitted by episode.

    The original DPPO does not rank segments by their raw return: it first fits a
    preference predictor on labelled segment pairs and ranks by that model's score.
    Here the labels come from the scripted teacher (the segment with the higher
    return is preferred; pairs closer than ``label_margin`` return standard
    deviations are skipped) and the predictor is
        score(segment) = V(first observation) + sum_t A(s_t, a_t),
    with A centred over the legal actions of each state. It is trained with the
    Bradley-Terry loss, -log sigmoid(score_i - score_j) for i preferred over j.
    The returned score is only the action part sum_t A(s_t, a_t): what the starting
    situation explains goes to V, and the luck of one segment's outcome (a failed
    suction on a good choice) is averaged over all similar segments instead of
    deciding the ranking. Each fold is scored by a model that never saw its episodes.
    """
    gen = th.Generator().manual_seed(seed)
    obs, act, mask, ret = data["obs"], data["act"], data["mask"], data["ret"]
    fold_of = th.as_tensor(data["episode"] % folds)
    gap = label_margin * ret.std().item()
    scores = th.zeros_like(ret)

    def action_part(net, idx):
        o, a, m = obs[idx].to(device), act[idx].to(device), mask[idx].to(device).float()
        b, h = a.shape
        out = net.outputs(o.flatten(0, 1))
        legal = m.flatten(0, 1)
        centre = (out * legal).sum(dim=1) / legal.sum(dim=1).clamp(min=1.0)
        adv = out.gather(1, a.reshape(-1, 1)).squeeze(1) - centre
        return adv.view(b, h).sum(dim=1)

    for fold in range(folds):
        train = (fold_of != fold).nonzero().squeeze(1)
        adv_net = OfflinePolicy(**policy_kwargs).to(device)
        val_net = OfflinePolicy(**policy_kwargs, scalar=True).to(device)
        params = [*adv_net.parameters(), *val_net.parameters()]
        opt = th.optim.Adam(params, lr=lr)
        for _ in range(steps):
            idx = train[th.randint(0, len(train), (batch_size,), generator=gen)]
            s = action_part(adv_net, idx) + val_net.outputs(obs[idx, 0].to(device)).squeeze(1)
            r = ret[idx].to(device)
            diff_r = r - r.roll(1)
            labelled = diff_r.abs() > gap
            if not bool(labelled.any()):
                continue
            logit = (s - s.roll(1))[labelled]
            loss = F.binary_cross_entropy_with_logits(logit, (diff_r[labelled] > 0).float())
            opt.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        held = (fold_of == fold).nonzero().squeeze(1)
        adv_net.eval()
        with th.no_grad():
            for chunk in held.split(4096):
                scores[chunk] = action_part(adv_net, chunk).cpu()
    return scores

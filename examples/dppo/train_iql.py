"""Train an IQL (Implicit Q-Learning) policy on collected rollouts.

The reward-supervised offline-RL baseline for train_dppo.py: same data, same
policy network (``OfflinePolicy``), so eval_offline_policy.py evaluates both
identically.

Discrete IQL: twin Q(s, .) fitted by TD against V(s'), V(s) by expectile
regression toward min Q(s, a), and the policy by advantage-weighted masked NLL
with weights exp(temperature * (Q - V)). The reward is the one ``env.step``
returned, unchanged.

    python -m gp8_control.examples.dppo.train_iql \
        --data 'gp8_control/examples/dppo/runs/data_1000/ep_*.npz' \
        --out gp8_control/examples/dppo/runs/iql_s0.pt

The defaults (expectile 0.9, temperature 10) are the values tuned for this env;
the IQL paper's defaults are 0.7 and 3.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch as th
from torch.nn import functional as F

from gp8_control.examples.dppo.offline import (
    OfflinePolicy,
    check_obs_width,
    policy_kwargs,
    rollout_files,
)
from gp8_control.examples.dppo.simenv import add_obs_args


def load_transitions(pattern: str) -> dict:
    """Flatten per-episode rollout files into (s, a, r, s', done) tensors."""
    files = rollout_files(pattern)
    obs, nxt, act, mask, rew, done = [], [], [], [], [], []
    for path in files:
        d = np.load(path)
        ep_obs, ep_act = d["obs"], d["act"]
        n = len(ep_act)
        obs.append(ep_obs)
        # The observation after the last step is not stored; the episode ends at
        # the observed time budget, so that step is a true terminal anyway.
        nxt.append(np.concatenate([ep_obs[1:], ep_obs[-1:]]))
        act.append(ep_act)
        mask.append(d["mask"])
        rew.append(d["rew"])
        done.append(np.arange(n) == n - 1)
    return {
        "obs": th.as_tensor(np.concatenate(obs).astype(np.float32)),
        "next_obs": th.as_tensor(np.concatenate(nxt).astype(np.float32)),
        "act": th.as_tensor(np.concatenate(act).astype(np.int64)),
        "mask": th.as_tensor(np.concatenate(mask).astype(bool)),
        "rew": th.as_tensor(np.concatenate(rew).astype(np.float32)),
        "done": th.as_tensor(np.concatenate(done).astype(np.float32)),
        "episodes": len(files),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="train_iql")
    parser.add_argument("--data", required=True, help="glob of ep_*.npz rollout files")
    parser.add_argument("--out", required=True)
    parser.add_argument("--expectile", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=10.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005, help="target-Q soft update")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-schedule", choices=("cosine", "constant"), default="cosine",
                        help="cosine anneals the policy learning rate to zero over --steps "
                             "(the schedule of the official IQL actor); Q and V stay constant")
    parser.add_argument("--hidden", type=int, default=256)
    add_obs_args(parser)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if th.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    th.manual_seed(args.seed)
    data = load_transitions(args.data)
    check_obs_width(data["obs"], args)
    n = len(data["act"])
    kwargs = policy_kwargs(args)
    device = th.device(args.device)
    policy = OfflinePolicy(**kwargs).to(device)
    # Q and V reuse the policy body: Q reads all action values from the global
    # token, V a single value.
    q1 = OfflinePolicy(**kwargs, critic=True).to(device)
    q2 = OfflinePolicy(**kwargs, critic=True).to(device)
    value = OfflinePolicy(**kwargs, scalar=True).to(device)
    q1_target, q2_target = copy.deepcopy(q1), copy.deepcopy(q2)

    opt_q = th.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=args.lr)
    opt_v = th.optim.Adam(value.parameters(), lr=args.lr)
    opt_pi = th.optim.Adam(policy.parameters(), lr=args.lr)
    sched_pi = (th.optim.lr_scheduler.CosineAnnealingLR(opt_pi, T_max=args.steps)
                if args.lr_schedule == "cosine" else None)
    gen = th.Generator().manual_seed(args.seed)
    print(f"IQL expectile {args.expectile} temperature {args.temperature} | episodes "
          f"{data['episodes']} transitions {n} | device {device}",
          flush=True)

    for step in range(1, args.steps + 1):
        idx = th.randint(0, n, (args.batch_size,), generator=gen)
        obs, nxt = data["obs"][idx].to(device), data["next_obs"][idx].to(device)
        act, mask = data["act"][idx].to(device), data["mask"][idx].to(device)
        rew, done = data["rew"][idx].to(device), data["done"][idx].to(device)

        with th.no_grad():
            q_min = th.min(q1_target.outputs(obs), q2_target.outputs(obs))
            q_min = q_min.gather(1, act[:, None]).squeeze(1)
            target = rew + args.gamma * (1.0 - done) * value.outputs(nxt).squeeze(1)
        v = value.outputs(obs).squeeze(1)
        diff = q_min - v
        v_loss = (th.abs(args.expectile - (diff < 0).float()) * diff.pow(2)).mean()
        opt_v.zero_grad()
        v_loss.backward()
        opt_v.step()

        q_loss = sum(
            F.mse_loss(net.outputs(obs).gather(1, act[:, None]).squeeze(1), target)
            for net in (q1, q2)
        )
        opt_q.zero_grad()
        q_loss.backward()
        opt_q.step()

        weight = th.exp(args.temperature * diff.detach()).clamp(max=100.0)
        logp = F.log_softmax(policy.logits(obs, mask), dim=-1)
        pi_loss = -(weight * logp.gather(1, act[:, None]).squeeze(1)).mean()
        opt_pi.zero_grad()
        pi_loss.backward()
        opt_pi.step()
        if sched_pi is not None:
            sched_pi.step()

        with th.no_grad():
            for net, tgt in ((q1, q1_target), (q2, q2_target)):
                for p, tp in zip(net.parameters(), tgt.parameters()):
                    tp.lerp_(p, args.tau)
        if step % 1000 == 0:
            print(f"step {step}: q {q_loss.item():.4f} v {v_loss.item():.4f} "
                  f"pi {pi_loss.item():.4f} adv {diff.mean().item():+.3f} "
                  f"V {v.mean().item():.2f}", flush=True)

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    policy.cpu().save(out, train_args=vars(args))
    print(f"saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

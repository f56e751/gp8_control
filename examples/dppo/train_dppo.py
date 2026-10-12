"""Train an offline policy on collected rollouts: DPPO, or a BC baseline.

Run from the parent directory of the ``gp8_control`` package:

    python -m gp8_control.examples.dppo.train_dppo \
        --data 'gp8_control/examples/dppo/runs/data_1000/ep_*.npz' \
        --out gp8_control/examples/dppo/runs/dppo_s0.pt          # DPPO
    ... --mode bc                                                # behaviour cloning

The defaults are the reported DPPO recipe: segments of 3 decisions aligned with
the rewards of the steps they were executed in, ranked by a learned preference
predictor, pairs closer than one score standard deviation skipped, cosine
learning-rate decay. See offline.py for each piece.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch as th

from gp8_control.examples.dppo.offline import (
    OfflinePolicy,
    check_obs_width,
    dppo_loss,
    fit_preference_scores,
    fit_score_baseline,
    load_segments,
    policy_kwargs,
    segment_nll,
)
from gp8_control.examples.dppo.simenv import add_obs_args

# (--seg-len, --decision-lag) of each mode when not given. BC only needs
# (observation, action) pairs, so it uses the reported BC batches: 2 steps, no lag.
SEGMENT_DEFAULTS = {"dppo": (3, 1), "bc": (2, 0)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="train_dppo")
    parser.add_argument("--data", required=True, help="glob of ep_*.npz rollout files")
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=("dppo", "bc"), default="dppo",
                        help="bc = plain masked NLL on the same segments; the ranking "
                             "options (--score, --pref-margin, --lambd, --bc-anchor) are unused")
    parser.add_argument("--score", choices=("preference", "baseline", "return"),
                        default="preference",
                        help="what ranks the segments: a cross-fitted preference predictor "
                             "(offline.py::fit_preference_scores), the segment return minus a "
                             "cross-fitted prediction from its first observation "
                             "(fit_score_baseline), or the raw segment return")
    parser.add_argument("--seg-len", type=int, default=None,
                        help="decisions per segment (default: 3 for dppo, 2 for bc)")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--decision-lag", type=int, default=None,
                        help="pair each reward window with the decisions made this many steps "
                             "earlier (default: 1 for dppo = the decisions executed inside the "
                             "window, since actions run one step after they are chosen; 0 for "
                             "bc); rewards are unchanged")
    parser.add_argument("--pref-margin", type=float, default=1.0,
                        help="pairs whose scores differ by less than this many score standard "
                             "deviations carry no preference (0 = every strictly ordered pair)")
    parser.add_argument("--lambd", type=float, default=0.5,
                        help="weight of the less preferred segment's distance in the pair loss")
    parser.add_argument("--bc-anchor", type=float, default=0.1,
                        help="weight of the mean NLL added to the ranking loss")
    parser.add_argument("--batch-segs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-schedule", choices=("cosine", "constant"), default="cosine",
                        help="cosine anneals the learning rate to zero over --steps")
    parser.add_argument("--hidden", type=int, default=256)
    add_obs_args(parser)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if th.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    seg_len, decision_lag = SEGMENT_DEFAULTS[args.mode]
    if args.seg_len is None:
        args.seg_len = seg_len
    if args.decision_lag is None:
        args.decision_lag = decision_lag
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    th.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    data = load_segments(args.data, seg_len=args.seg_len, stride=args.stride,
                         decision_lag=args.decision_lag)
    check_obs_width(data["obs"], args)
    n_segments = len(data["act"])
    kwargs = policy_kwargs(args)
    scores = data["ret"]
    if args.mode == "dppo" and args.score == "baseline":
        baseline = fit_score_baseline(data, kwargs, seed=args.seed, device=args.device)
        scores = data["ret"] - baseline
        explained = 1.0 - (scores.var() / data["ret"].var()).item()
        print(f"score baseline: explains {explained:.2f} of segment-return variance", flush=True)
    elif args.mode == "dppo" and args.score == "preference":
        scores = fit_preference_scores(data, kwargs, seed=args.seed, device=args.device)
        corr = th.corrcoef(th.stack([scores, data["ret"]]))[0, 1].item()
        print(f"score model: correlation with segment return {corr:.2f}", flush=True)
    margin = args.pref_margin * scores.std().item()

    # Built after the score models: they draw their initial weights from the same
    # random stream, and the reported runs created the policy last.
    device = th.device(args.device)
    policy = OfflinePolicy(**kwargs).to(device).train()
    opt = th.optim.Adam(policy.parameters(), lr=args.lr)
    sched = (th.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
             if args.lr_schedule == "cosine" else None)
    print(
        f"mode {args.mode} | episodes {data['episodes']} steps {data['steps']} "
        f"segments {n_segments} | seg return mean "
        f"{data['ret'].mean():.2f} sd {data['ret'].std():.2f} | device {device}",
        flush=True,
    )

    for step in range(1, args.steps + 1):
        idx = rng.choice(n_segments, size=args.batch_segs, replace=False)
        dists = segment_nll(policy, data["obs"][idx].to(device), data["act"][idx].to(device),
                            data["mask"][idx].to(device))
        if args.mode == "dppo":
            loss = dppo_loss(dists, scores[idx].to(device), args.lambd, args.bc_anchor, margin)
        else:
            loss = dists.mean()
        opt.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        if sched is not None:
            sched.step()
        if step % 1000 == 0:
            print(f"step {step}: loss {loss.item():.4f} nll {dists.mean().item():.3f}",
                  flush=True)

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    policy.cpu().save(out, train_args=vars(args))
    print(f"saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

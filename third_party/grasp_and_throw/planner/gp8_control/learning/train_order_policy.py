"""순서 정책 MLP 학습 — 후보별 점수 → (가능 후보만) softmax → 전문가 선택의 교차 엔트로피 (학습 3단계).

    python -m gp8_control.learning.train_order_policy --data pol_r0.npz [pol_r1.npz …] --out learning/models/order_policy.npz
    (torch 가 없으면 같은 학습을 numpy 로 — --backend numpy)

검증은 **흐름 단위**로 뗀다 (같은 흐름의 연속 결정은 비슷해 무작위로 나누면 과대평가). 지표: 전문가와 같은 선택 비율.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from gp8_control.learning.order_policy import DEFAULT_PATH, OrderPolicyMLP


def _train_numpy(a, F, Mk, Y, G, A=None) -> None:
    """torch 가 없는 서버(robot19 빌드 venv)용: 같은 모델(후보별 공유 tanh MLP)·손실(가능 후보 softmax CE)·AdamW·cosine·
    흐름 단위 검증·최고 검증 일치율 epoch 선택을 numpy 로."""
    rng = np.random.default_rng(0)
    groups = np.unique(G)
    val_g = set(rng.choice(groups, size=max(1, int(len(groups) * a.val)), replace=False).tolist())
    va = np.isin(G, list(val_g)); tr = ~va
    flat = F[tr][Mk[tr]]
    xm, xs = flat.mean(0), flat.std(0)
    xs = np.where(xs < 1e-3, 1.0, xs)                # 상수 특징은 정규화하지 않는다 (OrderPolicyMLP 와 같은 하한)
    X = ((F - xm) / xs).astype(np.float64)
    dims = [F.shape[2]] + list(a.hidden) + [1]
    Ws = [rng.uniform(-1, 1, (dims[i], dims[i + 1])) / np.sqrt(dims[i]) for i in range(len(dims) - 1)]
    bs = [rng.uniform(-1, 1, dims[i + 1]) / np.sqrt(dims[i]) for i in range(len(dims) - 1)]
    params = Ws + bs
    m1 = [np.zeros_like(p) for p in params]; m2 = [np.zeros_like(p) for p in params]
    b1, b2, eps = 0.9, 0.999, 1e-8

    def forward(idx):
        hs = [X[idx]]
        for k, (w, b) in enumerate(zip(Ws, bs)):
            z = hs[-1] @ w + b
            hs.append(np.tanh(z) if k < len(Ws) - 1 else z)
        s = hs[-1][..., 0]
        return hs, np.where(Mk[idx], s, -1e9)

    tr_i, va_i = np.flatnonzero(tr), np.flatnonzero(va)
    fifo_like = (Y[va] == np.argmax(Mk[va], axis=1)).mean()
    At = None if A is None else np.nan_to_num(A.astype(np.float64))
    print(f"결정: 학습 {len(tr_i)} / 검증 {len(va_i)} (흐름 {len(groups) - len(val_g)} / {len(val_g)}), 층 {dims} [numpy]. "
          f"검증에서 fifo 선택 = 전문가 비율 {fifo_like * 100:.1f}%", flush=True)
    if A is not None:
        oracle = float(np.nanmax(np.where(Mk[va], A[va], np.nan), axis=1).mean())
        print(f"  목표 = rollout 이득 회귀. 검증 이득: fifo 0, 전문가(최대) {oracle:.4f} 개/결정 (문턱 {a.adv_margin})", flush=True)
    best = (-1.0, None, -1); step = 0; t0 = time.time()
    for ep in range(a.epochs):
        lr = a.lr * 0.5 * (1 + np.cos(np.pi * ep / a.epochs))
        perm = np.random.default_rng(ep).permutation(tr_i)
        for k0 in range(0, len(perm), a.batch):
            idx = perm[k0:k0 + a.batch]; B = len(idx)
            hs, s = forward(idx)
            if A is None:                                                   # 전문가 선택 softmax CE
                p = np.exp(s - s.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
                g = p.copy(); g[np.arange(B), Y[idx]] -= 1.0; g /= B        # dL/ds
            else:                                                           # rollout 이득 회귀 (가능 후보만, Huber δ=0.5)
                m = Mk[idx]; r = np.where(m, hs[-1][..., 0] - At[idx], 0.0)
                g = np.clip(r, -0.5, 0.5) / max(1, int(m.sum()))
            grad_W, grad_b = [None] * len(Ws), [None] * len(Ws)
            d = g[..., None]                                                # (B,K,1)
            for k in range(len(Ws) - 1, -1, -1):
                grad_W[k] = np.einsum("bki,bkj->ij", hs[k], d); grad_b[k] = d.sum((0, 1))
                if k > 0:
                    d = (d @ Ws[k].T) * (1.0 - hs[k] ** 2)
            step += 1
            for j, (prm, gr) in enumerate(zip(params, grad_W + grad_b)):
                m1[j] = b1 * m1[j] + (1 - b1) * gr; m2[j] = b2 * m2[j] + (1 - b2) * gr * gr
                mh = m1[j] / (1 - b1 ** step); vh = m2[j] / (1 - b2 ** step)
                prm *= (1 - lr * a.wd)
                prm -= lr * mh / (np.sqrt(vh) + eps)
        _hs, sv = forward(va_i)
        if A is None:
            acc = float((sv.argmax(1) == Y[va_i]).mean())
        else:                                                               # 검증 지표 = 고른 후보의 실제 이득 평균 (fifo 0, 문턱 적용)
            acc = _adv_score(sv, Mk[va_i], At[va_i], a.adv_margin)
        if acc > best[0]:
            best = (acc, ([w.copy() for w in Ws], [b.copy() for b in bs]), ep)
        if ep % 10 == 0 or ep == a.epochs - 1:
            print(f"  ep {ep:3d}  검증 일치율 {acc * 100:.1f}%  (최고 {best[0] * 100:.1f}% @ {best[2]})  [{time.time() - t0:.0f}s]", flush=True)
    model = OrderPolicyMLP(best[1][0], best[1][1], xm, xs)
    if A is None:
        model.save(a.out, hidden=a.hidden, best_epoch=best[2], val_acc=best[0], data=a.data, backend="numpy", target="ce")
        print(f"저장 {a.out} (검증 일치율 {best[0] * 100:.1f}%, fifo 기준 {fifo_like * 100:.1f}%)")
    else:
        model.save(a.out, hidden=a.hidden, best_epoch=best[2], val_adv=best[0], data=a.data, backend="numpy", target="adv",
                   adv_margin=a.adv_margin)
        print(f"저장 {a.out} (검증 이득 {best[0]:.4f} 개/결정, 문턱 {a.adv_margin})")


def _adv_score(s, mask, adv, margin: float) -> float:
    """예측 점수 s 로 고른 후보(최고 후보가 fifo 후보보다 margin 이상일 때만, 아니면 fifo)의 실제 이득 평균."""
    s = np.where(mask, s, -np.inf)
    i = s.argmax(1); j = mask.argmax(1)
    r = np.arange(len(s))
    pick = np.where(s[r, i] - s[r, j] >= margin, i, j)
    return float(adv[r, pick].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", default=DEFAULT_PATH)
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--backend", default="auto", choices=["auto", "torch", "numpy"])
    ap.add_argument("--target", default="ce", choices=["ce", "adv"], help="ce: 전문가 선택 / adv: 후보별 rollout 이득 회귀 (numpy)")
    ap.add_argument("--adv-margin", type=float, default=0.1)
    a = ap.parse_args()
    F, Mk, Y, G, A = [], [], [], [], []
    for k, p in enumerate(a.data):
        z = np.load(p)
        F.append(z["F"]); Mk.append(z["mask"]); Y.append(z["y"]); G.append(z["group"] + 100000 * k)
        if a.target == "adv":
            A.append(z["adv"])
    F, Mk, Y, G = np.concatenate(F), np.concatenate(Mk), np.concatenate(Y), np.concatenate(G)
    if a.target == "adv":
        return _train_numpy(a, F, Mk, Y, G, np.concatenate(A))
    try:
        if a.backend == "numpy":
            raise ImportError
        import torch
    except ImportError:
        if a.backend == "torch":
            raise
        return _train_numpy(a, F, Mk, Y, G)
    torch.set_num_threads(a.threads); torch.manual_seed(0)
    rng = np.random.default_rng(0)
    groups = np.unique(G)
    val_g = set(rng.choice(groups, size=max(1, int(len(groups) * a.val)), replace=False).tolist())
    va = np.isin(G, list(val_g)); tr = ~va
    flat = F[tr][Mk[tr]]
    xm, xs = flat.mean(0), flat.std(0)
    xs = np.where(xs < 1e-3, 1.0, xs)                # 상수 특징은 정규화하지 않는다 (OrderPolicyMLP 와 같은 하한)
    Ft = torch.from_numpy(((F - xm) / xs).astype(np.float32))
    Mt = torch.from_numpy(Mk); Yt = torch.from_numpy(Y)
    dims = [F.shape[2]] + list(a.hidden) + [1]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(torch.nn.Tanh())
    net = torch.nn.Sequential(*layers)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    def logits(idx):
        s = net(Ft[idx]).squeeze(-1)
        return s.masked_fill(~Mt[idx], -1e9)

    tr_i, va_i = np.flatnonzero(tr), torch.from_numpy(np.flatnonzero(va))
    fifo_like = (Y[va] == np.argmax(Mk[va], axis=1)).mean()        # "가능한 첫 후보" (= fifo) 가 전문가와 같은 비율
    print(f"결정: 학습 {len(tr_i)} / 검증 {len(va_i)} (흐름 {len(groups) - len(val_g)} / {len(val_g)}), 층 {dims}. "
          f"검증에서 fifo 선택 = 전문가 비율 {fifo_like * 100:.1f}%", flush=True)
    best = (-1.0, None, -1)
    t0 = time.time()
    for ep in range(a.epochs):
        net.train()
        perm = torch.from_numpy(np.random.default_rng(ep).permutation(tr_i))
        for k in range(0, len(perm), a.batch):
            b = perm[k:k + a.batch]
            loss = torch.nn.functional.cross_entropy(logits(b), Yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            acc = (logits(va_i).argmax(1) == Yt[va_i]).float().mean().item()
        if acc > best[0]:
            best = (acc, {k: v.clone() for k, v in net.state_dict().items()}, ep)
        if ep % 10 == 0 or ep == a.epochs - 1:
            print(f"  ep {ep:3d}  검증 일치율 {acc * 100:.1f}%  (최고 {best[0] * 100:.1f}% @ {best[2]})  [{time.time() - t0:.0f}s]", flush=True)
    net.load_state_dict(best[1])
    lin = [m for m in net if isinstance(m, torch.nn.Linear)]
    model = OrderPolicyMLP([m.weight.detach().numpy().T for m in lin], [m.bias.detach().numpy() for m in lin], xm, xs)
    with torch.no_grad():
        ref = net(Ft[va_i[:64]]).squeeze(-1).numpy()
    assert np.allclose(model.scores(F[va_i[:64].numpy()]), ref, atol=1e-4), "numpy 추론 불일치"
    model.save(a.out, hidden=a.hidden, best_epoch=best[2], val_acc=best[0], data=a.data)
    print(f"저장 {a.out} (검증 일치율 {best[0] * 100:.1f}%, fifo 기준 {fifo_like * 100:.1f}%)")


if __name__ == "__main__":
    main()

"""DT 파인튜닝용 배치 샘플러/통계 — Thr_DT `train_dt_offline.py` 발췌.

원본 파일 전체를 벤더링하지 않은 이유: 그쪽은 `env.throw_env`(평면 시뮬)와
`evaluate_dt` 를 import 하는데, 로봇에서 쓰는 rig 는 `dt_gp8_env`(GP8) 라서
평면 env 를 끌어올 이유가 없다. 아래 두 함수만 **문자 그대로** 옮겼다:

    dataset_stats   (train_dt_offline.py 의 동명 함수)
    make_get_batch  (동상)

로직은 한 글자도 바꾸지 않았다. 원본이 바뀌면 여기도 같이 갱신할 것.
"""

import numpy as np
import torch


def dataset_stats(trajectories):
    """Success/fail and gripper open/close statistics
    [repo train_dt_offline.py:36-89]."""
    fails = successes = open_g = closed_g = 0
    for trajectory in trajectories:
        for t in trajectory:
            if t[2] == -1:
                fails += 1
            if t[2] == 1:
                successes += 1
            if t[1][-1] == -1:
                open_g += 1
            if t[1][-1] == 1:
                closed_g += 1
    oc_ratio = (closed_g / (open_g + closed_g)
                if (open_g and closed_g) else None)
    return successes, fails, oc_ratio


def make_get_batch(trajectories, model_cfg, train_cfg, device, rng):
    """Batch sampler, identical to [repo train_dt_offline.py:91-140]."""
    state_dim, act_dim, K = model_cfg.state_dim, model_cfg.act_dim, model_cfg.K

    def get_batch(batch_size=256, max_len=K):
        batch_inds = rng.choice(np.arange(len(trajectories)),
                                size=batch_size, replace=True)
        s, a, r, d, rtg, ts, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[batch_inds[i]]

            state = np.array([t[0][np.array([0, 1, 2, 3, -1])] for t in traj])
            s.append(state.reshape(1, -1, state_dim))
            a.append(np.array([t[1] for t in traj]).reshape(1, -1, act_dim))
            r.append(np.array([t[2] for t in traj]).reshape(1, -1, 1))
            # rewards are 0 until release, so RTG_t = final reward for all t
            rtg.append(np.array([traj[-1][2]] * len(traj)).reshape(1, -1, 1))
            d.append(np.array([t[3] for t in traj]).reshape(1, -1))

            noise_int = rng.integers(0, train_cfg.timestep_noise)
            ts.append(np.array([t + noise_int
                                for t in range(len(traj))]).reshape(1, -1))

            tlen = s[-1].shape[1]
            s[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            a[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen, act_dim)), a[-1]], axis=1)
            r[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate(
                [np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1)
            ts[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen)), ts[-1]], axis=1)
            mask.append(np.concatenate(
                [np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(
            dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(
            dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(
            dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(
            dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(
            dtype=torch.float32, device=device)
        ts = torch.from_numpy(np.concatenate(ts, axis=0)).to(
            dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
        return s, a, r, d, rtg, ts, mask

    return get_batch

"""Hindsight Experience Replay goal relabeling (paper S4.2).

Port of Agent.generate_her_memory of the official code
(ThrowBot/src/scripts/decision_transformer/agent/agent_dt.py:210-260):

* k = -1 : keep the original (random) goal -- "raw" buffer
* k = 0  : relabel the goal with the landing spot x_land
* k > 0  : landing spot + (k-1) goals sampled inside a circle of radius
           2*rho around x_land (her_radius_factor = 2)

Rewards of the terminal transition are recomputed with reward_sparse against
the new goal; a released throw that landed backwards therefore still gets -1
(reward_sparse also requires x_land > x_initial).
"""

import numpy as np


def generate_her_memory(arm, trajectory, target, obj_final_pos, k,
                        radius_factor=2.0, rng=None):
    """Returns a list of relabeled trajectories.

    `trajectory` is a list of (state4, action, reward, done, success) tuples,
    where state4 does NOT yet contain the goal; the goal (x-coordinate) is
    appended here, exactly like the official implementation.
    """
    rng = rng if rng is not None else np.random.default_rng()

    if k == -1:
        target_list = [np.asarray(target, dtype=np.float64)]
    elif k == 0:
        target_list = [np.asarray(obj_final_pos, dtype=np.float64)]
    else:
        target_list = [np.asarray(obj_final_pos, dtype=np.float64)]
        for _ in range(k - 1):
            rand = rng.random() * 2 - 1  # U[-1, 1]
            x = obj_final_pos[0] + arm.target_radius * rand * radius_factor
            if x > 0:
                target_list.append(np.array([x, 0.0, 0.0]))

    out = []
    for trg in target_list:
        new_trajectory = []
        for old_tuple in trajectory:
            state = np.append(old_tuple[0], trg[0])
            action = old_tuple[1]
            done = old_tuple[3]
            success = old_tuple[4]
            if success:
                reward = arm.reward_sparse(obj_pos=obj_final_pos, target=trg)
            else:
                reward = old_tuple[2]
            new_trajectory.append([state, action, reward, done])
        out.append(new_trajectory)
    return out


def generate_target(rng=None, dg_min=0.5, dg_max=2.0):
    """Random goal x in [dg_min, dg_max] m
    [repo agent/agent_dt.py:262-270]."""
    rng = rng if rng is not None else np.random.default_rng()
    x = rng.random() * (dg_max - dg_min) + dg_min
    return np.array([x, 0.0, 0.0])

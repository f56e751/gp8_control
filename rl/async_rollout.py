"""MaskablePPO rollouts without lockstep: every env gets its action as soon as it asks.

    python -m gp8_control.tools.train_ppo_smoke --env piece --async-rollout ...

Why: SB3's ``SubprocVecEnv`` steps all envs together, so each step waits for
the slowest env. A piece-mode env decides every 0.1 s .. several s of wall
time (the planner computes at wall speed), and while it waits for its action
its clocks stand still (``PieceRunner._ask``), so in lockstep most envs sit
idle: 12 envs ran at ~0.7x real time each instead of ~2.4x alone.

How it runs:

* each env lives in its own worker process (``_worker``), resets itself at the
  end of an episode and reports (obs, mask, reward, done, info) after a step;
* ``AsyncMaskablePPO.collect_rollouts`` hands actions to whichever envs are
  waiting (one batched policy call), and stops handing them out once the
  rollout has its ``n_steps`` transitions. The envs then wait for their next
  action with their clocks stopped, through the update, so every transition
  comes from the current policy (on-policy as in SB3);
* each env's transitions form one segment of the rollout; GAE runs per
  segment (``segment_gae``, the same formula as ``RolloutBuffer``),
  bootstrapped with the value of the observation the env is waiting on.

The algorithm sees ONE env with ``n_steps`` = the whole rollout
(``AsyncEnvPool.vec_env``), so minibatches and epochs are as with lockstep.
Piece env only: ``CauseCreditCallback`` moves rewards along buffer rows of
one env, which the segment layout does not keep (the piece env sends no
``credit_moves``).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from multiprocessing.connection import wait
from pathlib import Path

import numpy as np
import torch as th
from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper


def _worker(remote, parent_remote, env_fn_wrapper, seed) -> None:
    parent_remote.close()
    env = env_fn_wrapper.var()
    remote.send((env.observation_space, env.action_space))
    obs, _ = env.reset(seed=seed)
    remote.send((obs, env.action_masks()))
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, reward, term, trunc, info = env.step(data)
                if term or trunc:
                    info["terminal_observation"] = obs
                    obs, _ = env.reset()
                remote.send((obs, env.action_masks(), float(reward), bool(term), bool(trunc), info))
            elif cmd == "close":
                break
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        remote.close()


class _SpacesVecEnv(VecEnv):
    """What the algorithm sees: one env with the pool's spaces. Stepping goes
    through the pool (``AsyncMaskablePPO.collect_rollouts``)."""

    def __init__(self, pool: "AsyncEnvPool", observation_space, action_space) -> None:
        super().__init__(1, observation_space, action_space)
        self.pool = pool

    def reset(self):
        return np.zeros((1,) + self.observation_space.shape, dtype=self.observation_space.dtype)

    def step_async(self, actions) -> None:
        raise RuntimeError("async rollout: envs are stepped by AsyncEnvPool")

    def step_wait(self):
        raise RuntimeError("async rollout: envs are stepped by AsyncEnvPool")

    def close(self) -> None:
        self.pool.close()

    def has_attr(self, attr_name: str) -> bool:
        return attr_name == "action_masks"

    def get_attr(self, attr_name, indices=None):
        return [None]

    def set_attr(self, attr_name, value, indices=None) -> None:
        pass

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise RuntimeError("async rollout: envs are stepped by AsyncEnvPool")

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False]


class AsyncEnvPool:
    """Envs in worker processes, each stepped on its own (see module docstring).
    Writes the episode log ``monitor_path`` in VecMonitor's format (r, l, t)."""

    def __init__(self, env_fns, *, start_method: str, seed: int, monitor_path: Path) -> None:
        ctx = mp.get_context(start_method)
        self.n = len(env_fns)
        self.remotes, self.procs = [], []
        for rank, fn in enumerate(env_fns):
            remote, work_remote = ctx.Pipe()
            p = ctx.Process(target=_worker, daemon=True,
                            args=(work_remote, remote, CloudpickleWrapper(fn), seed + rank))
            p.start()
            work_remote.close()
            self.remotes.append(remote)
            self.procs.append(p)
        obs_space, act_space = self.remotes[0].recv()
        for r in self.remotes[1:]:
            r.recv()
        self.vec_env = _SpacesVecEnv(self, obs_space, act_space)
        self.obs, self.mask = [None] * self.n, [None] * self.n
        self.start = [True] * self.n                 # obs is the first of an episode
        self.waiting = set(range(self.n))            # obs given, no action yet
        for i, r in enumerate(self.remotes):
            self.obs[i], self.mask[i] = r.recv()
        self._ep_r = [0.0] * self.n
        self._ep_l = [0] * self.n
        self.t0 = time.time()
        self._mon = open(monitor_path, "w", encoding="utf-8")
        self._mon.write("#" + json.dumps({"t_start": self.t0, "env_id": "None"}) + "\n")
        self._mon.write("r,l,t\n")
        self._mon.flush()
        self._closed = False

    def send(self, i: int, action: int) -> None:
        self.waiting.discard(i)
        self.remotes[i].send(("step", int(action)))

    def replies(self, busy):
        """Block until at least one of the envs ``busy`` has stepped; yields
        (i, reward, term, trunc, info, ep_info or None) and stores its next obs."""
        for r in wait([self.remotes[i] for i in busy]):
            i = self.remotes.index(r)
            obs, mask, reward, term, trunc, info = r.recv()
            self._ep_r[i] += reward
            self._ep_l[i] += 1
            ep = None
            if term or trunc:
                ep = {"r": round(self._ep_r[i], 6), "l": self._ep_l[i],
                      "t": round(time.time() - self.t0, 6)}
                self._mon.write(f"{ep['r']},{ep['l']},{ep['t']}\n")
                self._mon.flush()
                self._ep_r[i], self._ep_l[i] = 0.0, 0
            self.obs[i], self.mask[i], self.start[i] = obs, mask, bool(term or trunc)
            self.waiting.add(i)
            yield i, reward, term, trunc, info, ep

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for r in self.remotes:
            try:
                r.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for p in self.procs:
            p.join(timeout=60.0)
        self._mon.close()


def segment_gae(rewards, values, dones, last_value: float, gamma: float, lam: float):
    """GAE of one env's consecutive transitions (``RolloutBuffer``'s formula):
    dones[t] = transition t ended its episode; last_value = V(obs after the last one)."""
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(n)):
        next_value = last_value if t == n - 1 else values[t + 1]
        non_terminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        gae = delta + gamma * lam * non_terminal * gae
        adv[t] = gae
    return adv, adv + np.asarray(values, dtype=np.float32)


class AsyncMaskablePPO(MaskablePPO):
    """MaskablePPO whose rollouts come from ``AsyncEnvPool`` (module docstring)."""

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps, use_masking=True) -> bool:
        pool = env.pool
        self.policy.set_training_mode(False)
        rollout_buffer.reset()
        callback.on_rollout_start()
        segs = {i: [] for i in range(pool.n)}
        busy: dict = {}                               # env -> transition in flight
        given = got = 0
        while got < n_rollout_steps:
            ids = sorted(pool.waiting)[: n_rollout_steps - given]
            if ids:
                obs = np.stack([pool.obs[i] for i in ids])
                masks = np.stack([pool.mask[i] for i in ids])
                with th.no_grad():
                    actions, values, log_probs = self.policy(
                        obs_as_tensor(obs, self.device), action_masks=masks)
                for j, i in enumerate(ids):
                    busy[i] = dict(obs=obs[j], mask=masks[j], start=pool.start[i],
                                   action=int(actions[j]), value=float(values[j]),
                                   log_prob=float(log_probs[j]))
                    pool.send(i, busy[i]["action"])
                given += len(ids)
            for i, reward, term, trunc, info, ep in pool.replies(list(busy)):
                tr = busy.pop(i)
                if trunc and info.get("terminal_observation") is not None:
                    with th.no_grad():     # timeout: bootstrap as SB3 does
                        v = self.policy.predict_values(
                            obs_as_tensor(info["terminal_observation"][None], self.device))
                    reward += self.gamma * float(v)
                tr["reward"], tr["done"] = reward, bool(term or trunc)
                segs[i].append(tr)
                got += 1
                self.num_timesteps += 1
                if ep is not None:
                    self.ep_info_buffer.extend([ep])
                callback.update_locals({"infos": [info], "dones": np.array([tr["done"]])})
                if not callback.on_step():
                    return False
        # every env now waits for its next action (clocks stopped) through the update
        ids = [i for i in range(pool.n) if segs[i]]
        with th.no_grad():
            last_values = self.policy.predict_values(
                obs_as_tensor(np.stack([pool.obs[i] for i in ids]), self.device)).cpu().numpy().ravel()
        advantages, returns = [], []
        for i, lv in zip(ids, last_values):
            seg = segs[i]
            adv, ret = segment_gae([t["reward"] for t in seg], [t["value"] for t in seg],
                                   [t["done"] for t in seg], float(lv), self.gamma, self.gae_lambda)
            advantages.append(adv)
            returns.append(ret)
            for t in seg:
                rollout_buffer.add(
                    t["obs"][None], np.array([[t["action"]]]), np.array([t["reward"]]),
                    np.array([t["start"]]), th.tensor([[t["value"]]]),
                    th.tensor([t["log_prob"]]), action_masks=t["mask"][None])
        rollout_buffer.advantages[:, 0] = np.concatenate(advantages)
        rollout_buffer.returns[:, 0] = np.concatenate(returns)
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

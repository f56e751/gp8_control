"""RL env on piece mode: the agent picks the next object and skill; piece mode
(the teammate's planner, the same code the real robot runs) moves the arm.

    python -m gp8_control.rl.piece_env --episodes 1 --seconds 120 --policy heuristic

Why: the skill-mode RL sim (``gym_env.GP8RecyclingEnv``) moves the arm with
the skills' own trajectories and no controller filter. With the real filter
(40 ms + 200/80 ms averages, verified on the robot 2026-11-24) its throws
collapse, while piece mode plans the realized motion through the filter and
matched the real robot (34 real objects replayed: 33 same actions).

How it runs:

* the MuJoCo twin with the controller filter (``cycle_sim_config``), stepped
  on VIRTUAL time (``backends/sim_tick.py``): while the planner link computes,
  time runs at wall speed (planning delay costs the same as on the robot);
  otherwise it is warped over. Commands go through a SampleBuffer on the
  sim's ticks.
* ``PieceRunner`` with ``decide_fn``: whenever piece mode needs its next step
  (about RT_AHEAD = 3 s before the held object's throw, or when the arm is
  free) the runner thread asks the agent and blocks until ``step()`` gives
  the action. Actions and observations are those of ``GP8RecyclingEnv``
  (``[slot, skill]``, slot ``max_objects`` = skip; skill 0 throw, 1 push).
* reward: the twin's object outcomes (+1 in its class bin, -0.3 wrong / floor;
  ``GP8_RL_MISS_PENALTY`` for untouched objects as in the skill env), summed
  over the time until the next decision.

The planner link runs as a local child process per env (GP8_CYCLE_PLANNER_HOST
is cleared); each reset starts a new twin and link (a few seconds).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import queue
import threading
import time
import types
from typing import Any

import numpy as np

try:  # gymnasium preferred, gym accepted (as gym_env)
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    try:
        import gym
        from gym import spaces
    except ImportError:
        gym = None
        spaces = None

from gp8_control.rl.common import SKILL_NAMES, normalize_bbox_observation, observation_width
from gp8_control.rl.piece_policy import heuristic_action

_DONE = object()


class GP8PieceEnv(gym.Env if gym is not None else object):
    """High-level RL env whose actions run through piece mode (see module docstring)."""

    metadata = {"render_modes": []}

    def __init__(self, *, max_objects: int = 6, max_episode_seconds: float = 240.0,
                 cfg=None, bbox_observation: str | None = None,
                 decision_timeout_s: float = 600.0) -> None:
        if spaces is None:
            raise ImportError("GP8PieceEnv requires gymnasium or gym.")
        self.max_objects = int(max_objects)
        self.max_episode_seconds = float(max_episode_seconds)
        self.cfg = cfg
        self.bbox_observation = normalize_bbox_observation(
            bbox_observation or os.environ.get("GP8_RL_BBOX_OBSERVATION", "size"))
        self.decision_timeout_s = float(decision_timeout_s)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, dtype=np.float32,
            shape=(observation_width(self.max_objects, False, self.bbox_observation, False),))
        self.action_space = spaces.MultiDiscrete([self.max_objects + 1, len(SKILL_NAMES)])
        self._sim = None
        self._runner = None
        self._thread = None
        self._to_agent: queue.Queue = queue.Queue()
        self._from_agent: queue.Queue = queue.Queue()
        self._stop = False
        self._step_count = 0
        self._t_start = 0.0
        self._pending_obs = None
        self._last_mask = None
        self.last_result: dict = {}

    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if gym is not None:
            super_reset = getattr(super(), "reset", None)
            if callable(super_reset):
                super_reset(seed=seed)
        self._shutdown()
        self._build(seed)
        self._step_count = 0
        item = self._to_agent.get(timeout=self.decision_timeout_s)
        if item is _DONE:
            raise RuntimeError(f"piece env ended before its first decision: {self.last_result}")
        obs, info = item
        return obs, info

    def step(self, action):
        if self._runner is None:
            raise RuntimeError("reset() must be called before step().")
        self._from_agent.put(np.asarray(action, dtype=int).reshape(-1))
        item = self._to_agent.get(timeout=self.decision_timeout_s)
        self._step_count += 1
        events = self._sim.core.pop_reward_events()
        self._sim._prune_resolved_sim_objects(events)
        reward = float(sum(float(e.get("reward", 0.0)) for e in events))
        if item is _DONE:
            obs = self._pending_obs
            info = dict(self.last_result, reward_events=events, episode_end=True)
            return obs, reward, True, False, info
        obs, info = item
        info = dict(info, reward_events=events)
        return obs, reward, False, False, info

    def close(self) -> None:
        self._shutdown()

    def action_mask(self) -> np.ndarray:
        """(max_objects + 1, n_skills) mask of the decision being asked (skip always valid)."""
        if self._last_mask is None:
            m = np.zeros((self.max_objects + 1, len(SKILL_NAMES)), dtype=np.uint8)
            m[self.max_objects, :] = 1
            return m
        return self._last_mask

    # ------------------------------------------------------------------
    def _build(self, seed) -> None:
        os.environ["GP8_SIM_REALTIME"] = "0"
        os.environ["GP8_PIECE_VIRTUAL"] = "1"
        os.environ["GP8_CYCLE_PLANNER_HOST"] = ""
        # The robot's piece-mode push settings (gp8_cycle: reach 0.70 in piece
        # mode; .env: accel 24). Push height stays the twin's (its belt is
        # ~5 cm higher than the real one in the robot frame).
        os.environ.setdefault("GP8_PUSH_MAX_REACH", "0.70")
        os.environ.setdefault("GP8_PUSH_ACCEL", "24")
        from gp8_control.backends.mujoco_sim import class_bin_name_for
        from gp8_control.backends.sim_tick import SimTickBuffer, VirtualTime
        from gp8_control.config import Config
        from gp8_control.cycle_throw import runner as R
        from gp8_control.cycle_throw.piece_runner import PieceRunner
        from gp8_control.rl.sim_runner import SimRlRunner
        from gp8_control.skills.planner_throw_skill import PLANNER_BIN_INDEX, PlannerThrowSkill

        from gp8_control.rl.sim_profile import apply_sim_profile

        apply_sim_profile()                   # a GP8_SIM_PROFILE fills env defaults first
        cfg = self.cfg or Config()
        cfg.BACKEND = "mujoco"
        sim_cfg = R.cycle_sim_config(cfg)
        if seed is not None:
            sim_cfg.seed = int(seed)
        sim = SimRlRunner(cfg, sim_cfg=sim_cfg, realtime=False, max_objects=self.max_objects,
                          bbox_observation=self.bbox_observation, sim_seed=seed)
        sim.start()
        vt = VirtualTime(sim.core)
        backend = sim.traj_ctrl
        buf = SimTickBuffer(vt, sim.core._ctrl_target)
        sim.core.set_tick_source(buf)
        backend.buf = buf
        # the controller holds the last COMMAND when the buffer takes over
        backend.buf_activate = lambda timeout=5.0: buf.activate(sim.core._ctrl_target)
        app = types.SimpleNamespace(cfg=sim.cfg, robot=sim.robot, traj_ctrl=backend,
                                    conveyor=sim.conveyor, queue=sim.queue, _node=sim.node,
                                    _spinner_dead=False, M1=sim.M1, M2=sim.M2,
                                    world=types.SimpleNamespace(_core=sim.core))
        throw = PlannerThrowSkill(sim.ctx)
        push = sim.skills["push"]
        self._can = {"throw": lambda o: o.class_name in PLANNER_BIN_INDEX,
                     "push": lambda o: push.can_handle(o)}

        bin_names = {b.name for b in sim.core.bins}

        def on_exec(o, skill):
            # The piece twin's bins: the planner's class bins (thrown) and the
            # push bins "push_<class>" (cycle_sim_config); the right one depends
            # on the skill that handled the object.
            want = f"push_{o.class_name}" if skill == "push" else o.class_name
            if want not in bin_names:
                want = class_bin_name_for(o.class_name)
            sim.core.mark_manipulated(o.sim_object_id, skill, want)

        from gp8_control.rl.piece_policy import PieceDecider

        # The SAME observation / mask code the robot uses (GP8_PIECE_POLICY).
        self._decider = PieceDecider(
            queue=sim.queue, ctx=sim.ctx, robot=sim.robot, joints_fn=sim.current_joints,
            conveyor=sim.conveyor, can=self._can, clock=sim.clock, max_objects=self.max_objects,
            bbox_observation=self.bbox_observation, remaining_fn=self._remaining)
        runner = PieceRunner(app, ctx=sim.ctx, skills={"throw": throw, "push": push},
                             skill_for=sim.ctx.skill_for, top_z=R.sim_object_top(sim.core.cfg),
                             on_execute=on_exec, vtime=vt, decide_fn=self._decide)
        self._sim, self._runner, self._vt, self._buf = sim, runner, vt, buf
        self._stop = False
        self._to_agent = queue.Queue()
        self._from_agent = queue.Queue()
        self._t_start = None

        def ok():
            if self._stop:
                return False
            if runner.t0 is None:
                return True
            if self._t_start is None:
                self._t_start = runner.now()
            return runner.now() - self._t_start < self.max_episode_seconds

        def run():
            w0 = time.monotonic()
            err = None
            try:
                runner.run(intake=sim.ingest, ok=ok)
            except Exception as e:  # noqa: BLE001
                err = repr(e)
            self._pending_obs = self._observe()
            self.last_result = dict(error=err, stats=dict(runner.stats),
                                    sim_s=round(sim.core.sim_time(), 1),
                                    wall_s=round(time.monotonic() - w0, 1),
                                    warped_s=round(vt.warped_s, 1), buf_rejects=buf.rejects[:5])
            self._to_agent.put(_DONE)

        self._thread = threading.Thread(target=run, name="piece_env", daemon=True)
        self._thread.start()

    def _shutdown(self) -> None:
        if self._thread is not None:
            self._stop = True
            self._from_agent.put(None)            # unblock a waiting decision
            self._thread.join(timeout=60.0)
            self._thread = None
        if self._sim is not None:
            try:
                self._sim.close()
            except Exception:  # noqa: BLE001
                pass
            self._sim = None
            self._runner = None
            gc.collect()

    # ------------------------------------------------------------------
    # runner thread side
    # ------------------------------------------------------------------
    def _remaining(self) -> float:
        if self._t_start is None or self._runner is None or self._runner.t0 is None:
            return 1.0
        left = self.max_episode_seconds - (self._runner.now() - self._t_start)
        return float(np.clip(left / self.max_episode_seconds, 0.0, 1.0))

    def _observe(self):
        # sim-only observation models (truth tracks, noise) first, then the shared encoder
        self._sim._refresh_truth_tracks(self._sim.clock.time())
        return self._decider.observation()

    def _decide(self, runner, cands, t_from):
        """decide_fn: hand the decision to the agent and wait for step()."""
        if self._stop:
            return None
        obs = self._observe()
        mask = self._decider.mask(cands)
        self._last_mask = mask
        slots = [int(o.track_id) for o in self._decider.objects()]
        info = {"action_mask": mask, "t_from": float(t_from),
                "sim_time": float(self._sim.core.sim_time()), "slots": slots}
        self._sim.core.step_index = self._step_count
        self._to_agent.put((obs, info))
        a = self._from_agent.get()
        if a is None or self._stop:
            return None
        return self._decider.to_choice(a, cands, mask, slots)


# ----------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--policy", choices=("heuristic", "random"), default="heuristic")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    env = GP8PieceEnv(max_episode_seconds=args.seconds)
    rng = np.random.default_rng(args.seed)
    for ep in range(args.episodes):
        w0 = time.monotonic()
        obs, info = env.reset(seed=args.seed + ep)
        total, n = 0.0, 0
        outcomes: dict = {}
        while True:
            mask = info["action_mask"]
            if args.policy == "heuristic":
                a = heuristic_action(mask, env.max_objects)
            else:
                valid = np.argwhere(mask)
                a = valid[rng.integers(len(valid))]
            obs, r, term, trunc, info = env.step(a)
            total += r
            n += 1
            for e in info.get("reward_events", []):
                key = (f"{e.get('class_name')} {e.get('manipulated_skill') or '-'}: "
                       f"{'in its bin' if e.get('actual_bin_name') and e.get('actual_bin_name') == e.get('class_bin_name') else 'missed ' + str(e.get('actual_bin_name'))}"
                       f" r={e.get('reward')}")
                outcomes[key] = outcomes.get(key, 0) + 1
            if term or trunc:
                break
        print("EPISODE", json.dumps(dict(env.last_result, ep=ep, decisions=n, reward=round(total, 2),
                                         episode_wall_s=round(time.monotonic() - w0, 1),
                                         outcomes=dict(sorted(outcomes.items()))),
                                    default=str), flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

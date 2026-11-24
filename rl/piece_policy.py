"""The piece-mode decision: observation, action mask and policy, shared by the
RL twin (``rl/piece_env.py``) and the real robot (``GP8_PIECE_POLICY``).

``PieceRunner`` calls ``decide_fn(runner, candidates, t_from)`` whenever piece
mode needs its next step. ``PieceDecider`` builds the SAME observation and mask
in both places (``rl.common.build_observation`` over the tracked queue, the
measured joints, the belt speed) so a policy trained in the twin sees on the
robot what it saw in training, and maps ``[slot, skill]`` back to the object:

* slot ``i`` < max_objects: the i-th object of the tracked queue (most
  downstream first); valid if piece mode offers it now (``candidates``) and
  the skill takes its class (throw: a planner bin; push: ``PushSkill``);
* slot ``max_objects``: skip (nothing now; asked again shortly).

Policies (``GP8_PIECE_POLICY``): ``heuristic`` (the app's rule: first valid
slot, push if allowed, else throw) or the path of a MaskablePPO ``.zip``
(deterministic). ``remaining_time_frac`` (an episode feature in training)
runs over ``GP8_PIECE_POLICY_EPISODE_S`` (240 s) from the run start and
wraps around on the robot.
"""

from __future__ import annotations

import math
import os
import time

import numpy as np

from gp8_control.rl.common import SKILL_NAMES, build_observation


def heuristic_action(mask: np.ndarray, max_objects: int) -> np.ndarray:
    """The app's rule over a mask: first (most downstream) valid slot, push if allowed."""
    for slot in range(max_objects):
        if mask[slot].any():
            return np.array([slot, 1 if mask[slot, 1] else 0])
    return np.array([max_objects, 0])


class PieceDecider:
    def __init__(self, *, queue, ctx, robot, joints_fn, conveyor, can: dict, clock,
                 max_objects: int = 6, bbox_observation: str = "size", policy=None,
                 remaining_fn=None, log=None) -> None:
        self.queue, self.ctx, self.robot = queue, ctx, robot
        self.joints_fn, self.conveyor, self.can, self.clock = joints_fn, conveyor, can, clock
        self.max_objects = int(max_objects)
        self.bbox_observation = bbox_observation
        self.policy = policy                  # (obs, mask) -> [slot, skill]; None = external
        self.remaining_fn = remaining_fn or (lambda: 1.0)
        self.log = log
        self.n = 0
        self.last = None                      # (obs, mask, slots, action)

    # ------------------------------------------------------------------
    def objects(self) -> list:
        return list(self.queue._objects)[: self.max_objects]

    def observation(self) -> np.ndarray:
        joints = self.joints_fn()
        ee = np.zeros(3, np.float32)
        if joints is not None:
            ee = np.asarray(self.robot.forward_kinematics(np.asarray(joints, float)[:6])[:3, 3],
                            np.float32)
        now = self.clock.time()
        v = float(self.conveyor.current)
        return build_observation(
            objects=self.objects(), max_objects=self.max_objects, include_eta=False,
            joints=joints, ee_xyz=ee, pending_indices=(self.max_objects, 0), belt_speed=v,
            remaining_time_frac=float(self.remaining_fn()),
            y_now_for=lambda o: self.ctx.object_y_now(o, now, v),
            bbox_observation=self.bbox_observation).astype(np.float32)

    def mask(self, cands) -> np.ndarray:
        ids = {it["o"]["id"] for it in cands}
        m = np.zeros((self.max_objects + 1, len(SKILL_NAMES)), dtype=np.uint8)
        m[self.max_objects, :] = 1
        for slot, obj in enumerate(self.objects()):
            if int(obj.track_id) in ids:
                for k, name in enumerate(SKILL_NAMES):
                    m[slot, k] = int(bool(self.can[name](obj)))
        return m

    def to_choice(self, action, cands, mask, slots):
        a = np.asarray(action, dtype=int).reshape(-1)
        slot, skill = int(a[0]), int(a[1])
        if slot >= len(slots) or slot >= self.max_objects or not mask[slot, skill]:
            return None                         # skip / invalid = nothing now
        it = next((c for c in cands if c["o"]["id"] == slots[slot]), None)
        return None if it is None else (it, SKILL_NAMES[skill])

    # ------------------------------------------------------------------
    def __call__(self, runner, cands, t_from):
        """decide_fn for PieceRunner with a built-in policy (the robot)."""
        obs, mask = self.observation(), self.mask(cands)
        slots = [int(o.track_id) for o in self.objects()]
        w0 = time.perf_counter()
        action = self.policy(obs, mask)
        choice = self.to_choice(action, cands, mask, slots)
        self.n += 1
        self.last = (obs, mask, slots, np.asarray(action).tolist())
        if self.log is not None:
            what = "skip" if choice is None else f"{choice[1]} {choice[0]['o']['id']}"
            self.log(f"[policy] decision {self.n}: {what} (action {np.asarray(action).tolist()}, "
                     f"{1e3 * (time.perf_counter() - w0):.1f} ms, valid "
                     f"{int(mask[:self.max_objects].sum())})")
        return choice


def load_policy(spec: str, max_objects: int):
    """``heuristic`` or a MaskablePPO .zip -> (obs, mask) -> [slot, skill]."""
    spec = (spec or "").strip()
    if spec in ("", "heuristic", "rule"):
        return lambda obs, mask: heuristic_action(mask, max_objects)
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(spec, device="cpu")
    n_sk = len(SKILL_NAMES)

    def act(obs, mask):
        flat, _ = model.predict(obs, action_masks=np.asarray(mask, bool).reshape(-1),
                                deterministic=True)
        f = int(np.asarray(flat).item())
        return np.array([f // n_sk, f % n_sk])

    return act


def episode_remaining(now_fn, t_start_fn, episode_s: float | None = None):
    """remaining_time_frac on the robot: an episode clock that wraps around."""
    E = float(episode_s or float(os.environ.get("GP8_PIECE_POLICY_EPISODE_S", "240")))

    def f():
        t0 = t_start_fn()
        if t0 is None:
            return 1.0
        el = max(0.0, now_fn() - t0)
        return 1.0 - math.fmod(el, E) / E

    return f

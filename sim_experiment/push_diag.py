"""Push diagnostics for the RL simulator (sim_experiment branch).

Two questions, one rollout:

1. **Mask** — how often is push even *offered* to the policy? Every decision
   records, per live object, whether throw / push is feasible in the action
   mask. Run once with GP8_PUSH_GUARD=1 and once with =0 to see how much the
   far-side guard limits (PUSH_EXEC_MIN_BACKSWING_X veto) remove.
2. **Physics** — when push IS executed, does the sim behave? A per-substep
   (2 ms) hook on ``SimCore._step_once_locked`` follows every box touched by
   push until it resolves and records: deepest contact penetration, lowest
   box bottom vs the belt top, peak linear / vertical / angular speed, and
   the outcome (class bin, wrong bin, missed). Thresholds only *flag*; the
   raw peaks are written so the cut-offs can be revisited.

Policy (``--policy``):
  push_first — push the first push-feasible slot, else skip (isolates push);
  throw_first — same with throw (the comparison arm for push vs throw);
  heuristic  — the app's rule-based choice (what the robot does today);
  model      — a trained MaskablePPO checkpoint (``--model``), deterministic.

Visual check (always on):
  * ``<tag>_state.npz`` + ``model.mjb`` — the full sim state every 20 ms
    (qpos, box sizes/colours). Replay it in the MuJoCo viewer or render any
    window with ``sim_experiment/replay_view.py``.
  * ``clips/`` — mp4 clips (rendered from that state log, so video time ==
    sim time) of the problem cases: physics flags, missed / wrong bin,
    failed executions, plus a few successful pushes for reference.

Run from the repository's parent directory:
  python -m gp8_control.sim_experiment.push_diag --out DIR --seed 0
Simulator settings come from the usual GP8_SIM_* env (e.g. GP8_SIM_PROFILE).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

# Same per-process env the training workers set (tools/train_ppo_smoke.py).
os.environ.setdefault("GP8_BACKEND", "mujoco")
os.environ["GP8_SIM_REALTIME"] = "false"
os.environ.setdefault("GP8_RL_SIM_LOG", "0")
os.environ.setdefault("GP8_SIM_LANDING_LOG", "0")

from gp8_control.rl.sim_profile import apply_sim_profile  # noqa: E402

apply_sim_profile()

import mujoco  # noqa: E402

from gp8_control.backends import mujoco_sim  # noqa: E402
from gp8_control.config import Config  # noqa: E402
from gp8_control.rl.common import SKILL_NAMES  # noqa: E402
from gp8_control.rl.gym_env import GP8RecyclingEnv  # noqa: E402

PUSH = SKILL_NAMES.index("push")
THROW = SKILL_NAMES.index("throw")

# Flag thresholds (indicative only; raw peaks are always saved).
PEN_FLAG_M = 0.005        # contact penetration deeper than 5 mm
SINK_FLAG_M = 0.005       # box bottom more than 5 mm below the belt top (on the belt)
SPEED_FLAG = 3.0          # m/s; the push stroke itself cruises at 2.0 m/s
VZ_UP_FLAG = 1.0          # m/s upward
SPIN_FLAG = 50.0          # rad/s


class PushPhysicsMonitor:
    """Per-substep peaks for every box whose manipulating skill is push."""

    def __init__(self, core) -> None:
        self.core = core
        m = core.model
        self.geom_to_box = {g: i for i, g in enumerate(core._box_geom_ids)}
        try:
            self.pusher_gid = int(m.geom(mujoco_sim._PUSHER_GEOM).id)
        except Exception:
            self.pusher_gid = -1
        self.belt_gid = int(m.geom(mujoco_sim.BELT_GEOM).id)
        self.active: dict[int, dict] = {}   # box index -> running record
        self.done: dict[int, dict] = {}     # sim serial -> final record
        self.belt_half_x = mujoco_sim._PHYSICAL_BELT_HALF_X
        # Belt footprint in base-frame X (physical belt: centre 0.415, half 0.215).
        self.belt_x0 = mujoco_sim._PHYSICAL_BELT_BASE_X - self.belt_half_x
        self.belt_x1 = mujoco_sim._PHYSICAL_BELT_BASE_X + self.belt_half_x
        # State log for replay / clip rendering (every LOG_EVERY substeps).
        self.log_t: list[float] = []
        self.log_qpos: list[np.ndarray] = []
        self.log_size: list[np.ndarray] = []
        self.log_rgba: list[np.ndarray] = []
        self._substep = 0
        self._gids = np.asarray(core._box_geom_ids, dtype=int)

    LOG_EVERY = 10   # 10 x 2 ms = 50 Hz

    def _log_state(self) -> None:
        core = self.core
        self.log_t.append(float(core.data.time))
        self.log_qpos.append(core.data.qpos.astype(np.float32).copy())
        self.log_size.append(core.model.geom_size[self._gids].astype(np.float32).copy())
        self.log_rgba.append(core.model.geom_rgba[self._gids].astype(np.float32).copy())

    def save_state(self, path: Path) -> None:
        np.savez_compressed(
            path, t=np.asarray(self.log_t), qpos=np.asarray(self.log_qpos),
            box_geom_ids=self._gids, box_size=np.asarray(self.log_size),
            box_rgba=np.asarray(self.log_rgba))

    def _record_for(self, i: int) -> dict:
        serial = int(self.core._box_serial[i])
        rec = self.active.get(i)
        if rec is None or rec["serial"] != serial:
            rec = {
                "serial": serial,
                "class_name": self.core._box_class[i],
                "t_start": float(self.core.data.time),
                # base-frame box position when push committed to it (the box
                # keeps riding the belt until the paddle arrives) and at first
                # paddle contact — for failure-vs-geometry analysis.
                "start_xy": [round(float(v), 4) for v in
                             self.core._to_base(np.asarray(self.core._box_joints[i].qpos[:3]))[:2]],
                "contact_xy": None, "t_contact": None,
                "max_pen_m": 0.0, "max_pen_with": "",
                "max_sink_m": 0.0,
                "max_speed": 0.0, "max_vz_up": 0.0, "max_spin": 0.0,
                "pusher_contact_steps": 0,
                "samples": 0,
            }
            self.active[i] = rec
        return rec

    def after_step(self) -> None:
        core = self.core
        d = core.data
        self._substep += 1
        if self._substep % self.LOG_EVERY == 0:
            self._log_state()
        watched = [
            i for i, s in enumerate(core._box_manipulated_skill)
            if s == "push" and core._box_state[i] != core._FREE
        ]
        if not watched:
            return
        belt_top = float(core._belt_top_w)
        for i in watched:
            rec = self._record_for(i)
            jnt = core._box_joints[i]
            q = jnt.qpos
            v = jnt.qvel
            half_z = float(core._box_half_size[i][2])
            bottom = float(q[2]) - half_z
            # Only count "sink" while the box centre is over the belt footprint
            # (base-frame X inside the belt, 2 cm margin); a box pushed past
            # the edge drops below the belt top legitimately while it falls.
            bx = float(core._to_base(np.asarray(q[:3]))[0])
            if (core._box_state[i] == core._ON_BELT
                    and self.belt_x0 + 0.02 < bx < self.belt_x1 - 0.02):
                rec["max_sink_m"] = max(rec["max_sink_m"], belt_top - bottom)
            speed = float(math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2))
            rec["max_speed"] = max(rec["max_speed"], speed)
            rec["max_vz_up"] = max(rec["max_vz_up"], float(v[2]))
            rec["max_spin"] = max(rec["max_spin"], float(math.sqrt(v[3] ** 2 + v[4] ** 2 + v[5] ** 2)))
            rec["samples"] += 1
        wset = set(watched)
        for k in range(d.ncon):
            c = d.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            for g, other in ((g1, g2), (g2, g1)):
                i = self.geom_to_box.get(g)
                if i is None or i not in wset:
                    continue
                rec = self.active[i]
                pen = -float(c.dist)
                if other == self.pusher_gid:
                    rec["pusher_contact_steps"] += 1
                    if rec["contact_xy"] is None:
                        rec["t_contact"] = float(self.core.data.time)
                        rec["contact_xy"] = [round(float(v), 4) for v in self.core._to_base(
                            np.asarray(self.core._box_joints[i].qpos[:3]))[:2]]
                if pen > rec["max_pen_m"]:
                    rec["max_pen_m"] = pen
                    if other == self.pusher_gid:
                        rec["max_pen_with"] = "pusher"
                    elif other == self.belt_gid:
                        rec["max_pen_with"] = "belt"
                    elif other in self.geom_to_box:
                        rec["max_pen_with"] = "box"
                    else:
                        rec["max_pen_with"] = mujoco.mj_id2name(
                            core.model, mujoco.mjtObj.mjOBJ_GEOM, other) or f"geom{other}"

    def resolve(self, events: list[dict]) -> list[dict]:
        out = []
        for ev in events:
            if ev.get("manipulated_skill") != "push":
                continue
            serial = int(ev["sim_object_id"])
            rec = None
            for i, r in list(self.active.items()):
                if r["serial"] == serial:
                    rec = self.active.pop(i)
                    break
            rec = dict(rec or {"serial": serial, "samples": 0})
            rec.update(
                t_end=float(ev.get("sim_time", 0.0)),
                class_name=ev.get("class_name"),
                class_bin=ev.get("class_bin_name"),
                actual_bin=ev.get("actual_bin_name"),
                reward=float(ev.get("reward", 0.0)),
                land_xyz=ev.get("base_xyz"),
                duration_s=float(ev.get("sim_time", 0.0)) - float(rec.get("t_start", ev.get("sim_time", 0.0))),
            )
            rec["outcome"] = (
                "class_bin" if rec["actual_bin"] is not None and rec["actual_bin"] == rec["class_bin"]
                else ("wrong_bin" if rec["actual_bin"] is not None else "missed")
            )
            flags = []
            if rec.get("max_pen_m", 0.0) > PEN_FLAG_M:
                flags.append("penetration")
            if rec.get("max_sink_m", 0.0) > SINK_FLAG_M:
                flags.append("sink")
            if rec.get("max_speed", 0.0) > SPEED_FLAG:
                flags.append("fast")
            if rec.get("max_vz_up", 0.0) > VZ_UP_FLAG:
                flags.append("launch_up")
            if rec.get("max_spin", 0.0) > SPIN_FLAG:
                flags.append("spin")
            rec["flags"] = flags
            out.append(rec)
        return out


def _pick_clips(push_records: list[dict], exec_failures: list[dict], limit: int) -> list[dict]:
    """Round-robin over problem categories so the clips are varied, then add
    up to two clean successes as a reference."""
    buckets: dict[str, list[dict]] = {}
    for r in push_records:
        t0 = float(r.get("t_start", r.get("t_end", 0.0))) - 1.0
        t1 = float(r.get("t_end", t0 + 3.0)) + 1.5
        t1 = min(t1, t0 + 12.0)
        cats = list(r["flags"]) + ([r["outcome"]] if r["outcome"] != "class_bin" else [])
        base = {"serial": r.get("serial"), "t0": t0, "t1": t1, "outcome": r["outcome"],
                "flags": r["flags"], "class_name": r.get("class_name")}
        if not cats:
            buckets.setdefault("ok", []).append(dict(base, why="ok"))
        for cat in cats:
            buckets.setdefault(cat, []).append(dict(base, why=cat))
    for e in exec_failures:
        buckets.setdefault("exec_fail", []).append(
            {"t0": e["t_sim"] - 0.5, "t1": e["t_sim"] + 4.0, "why": "exec_fail",
             "skill": e["skill"], "detail": e["detail"]})
    order = [k for k in ("penetration", "sink", "fast", "launch_up", "spin",
                         "missed", "wrong_bin", "exec_fail") if k in buckets]
    picked, seen = [], set()
    while len(picked) < limit and any(buckets.get(k) for k in order):
        for k in order:
            if buckets.get(k) and len(picked) < limit:
                c = buckets[k].pop(0)
                key = (c.get("serial"), round(c["t0"], 1))
                if key in seen:
                    continue
                seen.add(key)
                picked.append(c)
    for c in buckets.get("ok", [])[:2]:
        picked.append(c)
    for n, c in enumerate(picked):
        c["name"] = f"{n:02d}_{c['why']}" + (f"_obj{c['serial']}" if c.get("serial") is not None else "")
    return picked


def _install_hook():
    orig = mujoco_sim.SimCore._step_once_locked

    def _step(self):
        orig(self)
        mon = getattr(self, "_push_diag_monitor", None)
        if mon is not None:
            mon.after_step()

    mujoco_sim.SimCore._step_once_locked = _step


_LAST_PROBS: dict = {}


def _choose(policy: str, mask: np.ndarray, env: GP8RecyclingEnv, obs=None, model=None) -> np.ndarray:
    n = mask.shape[0] - 1
    if policy == "model":
        flat_mask = np.asarray(mask, dtype=bool).reshape(-1)
        try:  # policy probabilities over (throw, push, skip) for the analysis
            import torch as th
            obs_t, _ = model.policy.obs_to_tensor(obs)
            with th.no_grad():
                dist = model.policy.get_distribution(obs_t, action_masks=flat_mask[None])
                probs = dist.distribution.probs[0].cpu().numpy()
            k = len(SKILL_NAMES)
            _LAST_PROBS.update(
                p_throw=float(probs[THROW:n * k:k].sum()),
                p_push=float(probs[PUSH:n * k:k].sum()),
                p_skip=float(probs[n * k:].sum()))
        except Exception as exc:
            _LAST_PROBS.update(error=repr(exc))
        a, _ = model.predict(obs, action_masks=flat_mask, deterministic=True)
        a = int(np.asarray(a).reshape(-1)[0])
        return np.array([a // len(SKILL_NAMES), a % len(SKILL_NAMES)])
    if policy in ("push_first", "throw_first"):
        want = PUSH if policy == "push_first" else THROW
        for slot in range(n):
            if mask[slot, want]:
                return np.array([slot, want])
        return np.array([n, 0])
    idx = env._runner.app_heuristic_action_indices(mask)
    return np.asarray(idx, dtype=int).reshape(-1)[:2]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="push_diag")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", choices=("push_first", "throw_first", "heuristic", "model"), default="push_first")
    ap.add_argument("--episode-seconds", type=float, default=240.0)
    ap.add_argument("--max-steps", type=int, default=8000)  # skip 0.04 s; 1000이면 skip 많은 정책이 240 s 전에 잘림
    ap.add_argument("--bbox-observation", default="corners")
    ap.add_argument("--model", default="", help="MaskablePPO .zip for --policy model")
    ap.add_argument("--max-clips", type=int, default=16)
    ap.add_argument("--no-clips", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.policy}_seed{args.seed}"
    _install_hook()
    os.environ.setdefault("GP8_RL_SIM_TRUTH_TRACKS", "1")

    env = GP8RecyclingEnv(
        max_objects=6, max_steps=args.max_steps,
        max_episode_seconds=args.episode_seconds, cfg=Config(),
        realtime=False, bbox_observation=args.bbox_observation,
    )
    model = None
    if args.policy == "model":
        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(args.model, device="cpu")
    t0 = time.time()
    obs, info = env.reset(seed=args.seed)
    mon = PushPhysicsMonitor(env._runner.core)
    env._runner.core._push_diag_monitor = mon

    mask_stats = Counter()
    exec_stats = Counter()
    push_records: list[dict] = []
    exec_failures: list[dict] = []
    steps = 0
    with open(out / f"{tag}_steps.jsonl", "w") as fstep:
        while True:
            mask = np.asarray(info["action_mask"])
            live = mask[:-1].any(axis=1)  # a slot with any feasible skill
            n_slots = int(sum(1 for _ in env._runner.ordered_objects()))
            mask_stats["decisions"] += 1
            mask_stats["slots_live"] += n_slots
            mask_stats["slots_throw_ok"] += int(mask[:-1, THROW].sum())
            mask_stats["slots_push_ok"] += int(mask[:-1, PUSH].sum())
            mask_stats["slots_any_ok"] += int(live.sum())
            mask_stats["decisions_with_push_ok"] += int(mask[:-1, PUSH].any())
            mask_stats["decisions_with_throw_ok"] += int(mask[:-1, THROW].any())
            _LAST_PROBS.clear()
            action = _choose(args.policy, mask, env, obs, model)
            probs_now = dict(_LAST_PROBS)
            t_dec = float(env._runner.core.sim_time())
            obs, reward, term, trunc, info = env.step(action)
            steps += 1
            if info.get("executed") and not info.get("execution_success"):
                exec_failures.append({"t_sim": t_dec, "skill": info.get("executed_skill"),
                                      "detail": info.get("execution_detail")})
            if info.get("executed"):
                exec_stats[f"{info.get('executed_skill')}_{'ok' if info.get('execution_success') else 'fail'}"] += 1
                if not info.get("execution_success"):
                    exec_stats[f"fail_detail:{str(info.get('execution_detail'))[:60]}"] += 1
            recs = mon.resolve(info.get("reward_events", []))
            push_records.extend(recs)
            fstep.write(json.dumps({
                "step": steps, "t": info.get("episode_sim_time"), "t_sim": t_dec,
                "action": [int(a) for a in action], "reward": float(reward),
                "executed": info.get("executed"), "executed_skill": info.get("executed_skill"),
                "success": info.get("execution_success"), "detail": info.get("execution_detail"),
                "push_ok_slots": int(mask[:-1, PUSH].sum()), "throw_ok_slots": int(mask[:-1, THROW].sum()),
                "policy_probs": probs_now or None,
                "events": [{k: e.get(k) for k in ("class_name", "manipulated_skill", "actual_bin_name",
                                                   "class_bin_name", "reward", "collateral")}
                           for e in info.get("reward_events", [])],
            }) + "\n")
            if term or trunc:
                break
    core = env._runner.core
    belt_speed = float(core.cfg.belt_speed)
    model_path = out / "model.mjb"
    if not model_path.exists():   # parallel runs share one folder: write + atomic rename
        tmp = out / f".model.{os.getpid()}.mjb"
        mujoco.mj_saveModel(core.model, str(tmp), None)
        os.replace(tmp, model_path)
    state_path = out / f"{tag}_state.npz"
    mon.save_state(state_path)
    env.close()

    clips = [] if args.no_clips else _pick_clips(push_records, exec_failures, args.max_clips)
    if clips:
        from gp8_control.sim_experiment.replay_view import render_clip
        cdir = out / "clips"
        cdir.mkdir(exist_ok=True)
        for c in clips:
            name = f"{tag}_{c['name']}.mp4"
            try:
                render_clip(model_path, state_path, cdir / name, c["t0"], c["t1"],
                            focus_serial=c.get("serial"))
                c["file"] = f"clips/{name}"
            except Exception as exc:  # rendering must never lose the numbers
                c["error"] = repr(exc)
        with open(out / f"{tag}_clips.json", "w") as f:
            json.dump(clips, f, indent=1)

    with open(out / f"{tag}_push_objects.jsonl", "w") as f:
        for r in push_records:
            f.write(json.dumps(r) + "\n")
    outcome = Counter(r["outcome"] for r in push_records)
    flags = Counter(fl for r in push_records for fl in r["flags"])
    summary = {
        "policy": args.policy, "seed": args.seed,
        "push_guard": os.environ.get("GP8_PUSH_GUARD", "1"),
        "steps": steps, "wall_s": time.time() - t0, "belt_speed": belt_speed,
        "mask": dict(mask_stats),
        "push_offered_frac_of_live_slots": (
            mask_stats["slots_push_ok"] / mask_stats["slots_live"] if mask_stats["slots_live"] else None),
        "throw_offered_frac_of_live_slots": (
            mask_stats["slots_throw_ok"] / mask_stats["slots_live"] if mask_stats["slots_live"] else None),
        "executed": dict(exec_stats),
        "push_objects_resolved": len(push_records),
        "push_outcome": dict(outcome),
        "push_flags": dict(flags),
        "push_flagged_objects": sum(1 for r in push_records if r["flags"]),
        "exec_failures": len(exec_failures),
        "clips": len(clips),
        "thresholds": {"pen_m": PEN_FLAG_M, "sink_m": SINK_FLAG_M, "speed": SPEED_FLAG,
                       "vz_up": VZ_UP_FLAG, "spin": SPIN_FLAG},
        "env": {k: v for k, v in os.environ.items() if k.startswith(("GP8_SIM_", "GP8_RL_", "GP8_PUSH_", "GP8_TRACK_"))},
    }
    (out / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in (
        "policy", "seed", "push_guard", "steps", "wall_s", "push_offered_frac_of_live_slots",
        "throw_offered_frac_of_live_slots", "executed", "push_outcome", "push_flags")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

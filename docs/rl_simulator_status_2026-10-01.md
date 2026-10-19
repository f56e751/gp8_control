# RL Simulator Status

This note summarizes the current Gym-style RL simulator work for the GP8
recycling task. It is intended for teammate review and handoff, not as a log of
every experiment we ran.

## Goal

The RL environment trains a high-level policy on top of the same MuJoCo
simulator and skill code used by the robot control stack. The policy does not
output joint trajectories. It chooses which visible object should be handled
next and whether that object should be thrown or pushed. The existing throw and
push skill implementations then generate and execute the lower-level motion.

The current recommended simulator setting is:

- MuJoCo backend.
- Physical conveyor belt.
- Old-repo belt contact parameters.
- Randomized object generation for training and evaluation.
- Ground-truth simulator tracks for RL observations.
- Corner bounding-box observations.
- Time-budgeted episodes.
- No suction-probability observation by default.

## MDP Definition

### State

The observation is a fixed-width vector built by `rl/common.py::build_observation`.
For the default setting, the environment keeps up to `max_objects=6` object
slots. Objects are ordered by the same queue used by the simulator runner.

Each object slot contains:

| Field | Meaning |
|---|---|
| `x_now, y_now, z_now` | Current object position in the robot base frame. |
| `class_id` | `metal=0`, `transparent=1`, `cardboard=2`, unknown/empty `-1`. |
| `confidence` | Detection confidence carried by the tracking object. |
| `bbox_corners` | Four base-frame XY box corners, flattened to 8 numbers. |

The global part contains:

| Field | Meaning |
|---|---|
| `robot_joints[6]` | Current GP8 joint positions. |
| `ee_xyz[3]` | Current end-effector position in the robot base frame. |
| `pending_object_slot` | Slot of the action that was selected on the previous Gym step and is pending execution. |
| `pending_skill` | Pending skill index, where `0=throw`, `1=push`. |
| `belt_speed` | Current conveyor speed. |
| `remaining_time_frac` | Remaining episode time divided by the time budget. |

Default observation width:

```text
6 objects * (3 xyz + 1 class + 1 confidence + 8 bbox corners)
+ 6 robot joints
+ 3 ee xyz
+ 1 pending object slot
+ 1 pending skill
+ 1 belt speed
+ 1 remaining time fraction
= 91
```

Optional observation features:

- `include_eta`: appends `throw_eta, push_eta` per object. This is currently
  off by default.
- `include_suction_p`: appends one suction-success probability per object.
  This is implemented for suction experiments, but off by default so the
  baseline observation remains 91-dimensional.

Empty object slots are padded with:

```text
x=-1, y=-1, z=0, class_id=-1, confidence=0, bbox=0
```

### Action

The logical action is:

```text
(object_slot, skill)
```

where:

- `object_slot` is one of `0..max_objects-1`, or `max_objects` for skip.
- `skill` is `0=throw` or `1=push`.

For MaskablePPO, this pair is flattened by `rl/sb3_wrapper.py` into one
discrete action:

```text
flat_action = object_slot * num_skills + skill
```

With `max_objects=6` and two skills, the flat action space has 14 actions:
12 object/skill choices plus two skip actions.

### Action Mask

The action mask is separate from the observation. It is computed by
`SimRlRunner.action_mask()`.

An object/skill pair is valid only if the corresponding skill can be executed
at the one-step-ahead horizon. The feasibility check uses the same skill-level
logic as execution:

- reachable intercept exists,
- IK succeeds through the skill planner,
- skill placement veto does not reject the action,
- the target object is still live,
- the target is not already the pending action.

The skip row is always valid. Invalid selected actions are treated as skip by
the Gym environment, but MaskablePPO should normally avoid them through the
mask.

### Transition

The environment uses a one-step-ahead action convention.

At Gym step `t`:

1. The action selected on step `t-1` is executed, if it is still live.
2. The newly selected action from step `t` becomes the pending action.
3. The next observation reflects the simulator state after the executed action
   and after pending-action bookkeeping.

This mirrors the fact that the robot often needs to plan toward the next object
while the current skill is finishing.

### Reward

Rewards are sparse resolved-object rewards from the simulator.

The intended bin is class-dependent:

| Class | Intended bin |
|---|---|
| `transparent` | throw/transparent bin |
| `metal` | push/metal bin |

The intended bin does not depend on which skill was selected. For example, a
metal object is still judged by whether it reaches the metal bin, even if the
policy selected throw.

Reward logic:

| Event | Reward |
|---|---|
| Manipulated object resolves inside its class bin | `+1.0` |
| Manipulated object resolves outside its class bin | `-0.3` |
| Unmanipulated object is knocked off as collateral | `-0.3` |
| Unmanipulated object naturally passes without being resolved by the skill | `0.0` |
| Skill execution failure | additional `-0.3` |

Reward events are dropped if they occur after the episode time deadline.

### Episode Termination

Episodes are time-budgeted:

```text
max_episode_seconds = 240
```

There is also a safety cap:

```text
max_steps = 1000
```

The environment currently uses truncation rather than terminal success/failure.
The remaining time fraction is part of the observation because the value
function depends strongly on how much episode time remains.

## Simulator Defaults

Recommended environment variables for the current baseline:

```bash
export GP8_SIM_RANDOMIZE=true
export GP8_SIM_PHYSICAL_BELT=true
export GP8_SIM_BELT_CONTACT_PARAMS=old
export GP8_RL_SIM_TRUTH_TRACKS=true
export GP8_RL_BBOX_OBSERVATION=corners
export GP8_RL_INCLUDE_SUCTION_P=false
```

Randomized simulator ranges:

| Quantity | Range |
|---|---|
| Belt speed | uniform `[0.05, 0.20]` m/s |
| Spawn rate | sampled from `[0.5, 1.0]` Hz |
| Spawn x | uniform `[0.30, 0.58]` m |
| Class | randomized over configured classes, currently transparent and metal |
| Size | half x `[0.0375, 0.075]`, half y `[0.05, 0.15]`, half z `0.015` |
| Yaw | uniform over `[-pi, pi]` |

The current default uses simulator truth tracks for RL. Camera/FOV-style tracks
are still useful for realism studies, but truth tracks are the recommended
training default because they remove perception dead-reckoning artifacts from
the first RL policy work.

## Training Entry Point

The PPO training entry point is:

```bash
python -m gp8_control.tools.train_ppo_smoke
```

The script supports:

- `--n-envs` for parallel environments,
- `--n-steps` and `--batch-size` for PPO rollout/minibatch settings,
- `--device cpu|cuda|auto`,
- `--tensorboard`,
- `--eval-only`,
- `--trace-jsonl`,
- `--truth-tracks` / `--camera-tracks`,
- `--include-suction-p`.

Example baseline command:

```bash
cd /PublicSSD/jhsong

GP8_SIM_RECORD= \
GP8_SIM_RANDOMIZE=true \
GP8_SIM_PHYSICAL_BELT=true \
GP8_SIM_BELT_CONTACT_PARAMS=old \
GP8_RL_SIM_TRUTH_TRACKS=true \
GP8_RL_BBOX_OBSERVATION=corners \
GP8_RL_INCLUDE_SUCTION_P=false \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
gp8_control/.venv/bin/python -m gp8_control.tools.train_ppo_smoke \
  --out-dir gp8_control/runs/example_truth_randomized_100k \
  --timesteps 100000 \
  --n-envs 16 \
  --n-steps 16 \
  --batch-size 128 \
  --seed 0 \
  --max-steps 1000 \
  --max-episode-seconds 240 \
  --bbox-observation corners \
  --truth-tracks \
  --device auto \
  --tensorboard \
  --verbose 1
```

For final comparison, do not rely only on the short train-time evaluation in
`summary.json`. Use the fixed multi-seed evaluation protocol and inspect the
trace/action counts.

## Representative Results

The table below lists only representative results that are useful context for
the next developer. These are 10-seed evaluation sweeps, not the short
train-time `summary.json` metric.

| Condition | Notes | Mean reward | Push behavior |
|---|---|---:|---|
| Truth tracks, randomized, despawn fixed, `n_envs=16`, `n_steps=16`, batch `128` | Strong current no-suction baseline, but not perfectly repeatable across reruns. | About `80-85` in the better runs | Usually throw-only |
| Truth tracks, randomized, despawn fixed, `n_envs=8`, `n_steps=32`, batch `128` | Earlier strong result. | About `84.75` | Some runs showed pushes, but push selection was not stable |
| Camera-style observation, randomized, despawn fixed | Lower and less stable than truth-track observation in our sweeps. | Often lower than truth-track runs | Push selection inconsistent |
| Suction binary, metal half graspable/half ungraspable | Optional observation includes `suction_p`; the policy sometimes learns push, but rewards remained lower than no-suction baseline. | About `49-60` in tested runs | Pushes appeared, especially in the better `n_steps=32` run |
| Metal always ungraspable | Stress test for whether pushing can be learned. | About `39-44` in tested runs | Pushes appeared frequently |

Interpretation so far:

- The simulator and MDP are usable for PPO smoke training.
- The best policies can obtain reasonable sorting reward, but training is
  still high-variance.
- Push is executable and can be selected, but PPO often converges to throw-only
  policies unless the task strongly forces pushing.
- Ground-truth object observations currently produce more reliable results than
  camera-style observations.
- Longer or repeated training is not guaranteed to recover a poor 100k run.
  Model selection should use fixed multi-seed evaluation, not the train-time
  rollout reward alone.

## Optional Suction Experiments

Suction probability support is implemented but disabled by default.

Relevant settings:

```bash
export GP8_RL_INCLUDE_SUCTION_P=true
export GP8_SIM_METAL_SUCTION_BINARY=true
export GP8_SIM_METAL_SUCTION_MODE=binary  # binary, zero, or one
```

Modes:

- `binary`: each metal object gets `suction_p` sampled as `0` or `1`; transparent
  objects remain suctionable.
- `zero`: all metal objects are ungraspable.
- `one`: all metal objects are graspable.

When `include_suction_p=true`, the observation width increases from 91 to 97
for `max_objects=6`.

## Current Limitations

- PPO results are sensitive to seeds and rollout settings.
- Push reward/selection is not yet robust in the general randomized task.
- The current recommended policy observation uses simulator truth tracks; this
  is intentionally optimistic compared with real camera observations.
- GPU training is supported through the `--device` argument, but each machine's
  PyTorch/CUDA environment must be set up separately.
- The real-robot RL path is currently best treated as shadow/evaluation support
  until a trained policy is selected and reviewed.

## Files To Review

Core RL/simulator files:

- `rl/common.py`: shared observation/action schema.
- `rl/gym_env.py`: Gym environment and episode handling.
- `rl/sim_runner.py`: non-ROS simulator runner and one-step-ahead execution.
- `rl/sb3_wrapper.py`: flat masked action wrapper for MaskablePPO.
- `tools/train_ppo_smoke.py`: PPO train/eval entry point.
- `backends/mujoco_sim.py`: MuJoCo simulator, randomization, suction, bins,
  reward events, and despawn/recycle logic.

Real/shadow integration files:

- `rl/real_runner.py`
- `rl/real_shadow.py`

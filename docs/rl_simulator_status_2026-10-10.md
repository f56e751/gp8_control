# RL Simulator Status - 2026-10-10

This document defines the current Gym-style reinforcement-learning problem for
the GP8 recycling simulator and records the simulator configuration that should
be used for subsequent training.

## Objective and Scope

The learned policy is a high-level scheduler. It does not generate joint
trajectories. At each decision, it chooses an object and one of the existing
robot skills (`throw` or `push`). The simulator then uses the same skill
planning and trajectory-generation code used by the robot-control stack.

The current training baseline uses:

- MuJoCo with the physical conveyor and old-repository contact parameters;
- randomized belt speed, arrivals, object class, size, position, and yaw;
- simulator ground-truth object tracks;
- oriented bounding-box corners in the observation;
- a 240-second simulated-time episode budget;
- class-dependent sorting rewards;
- MaskablePPO with one-step-ahead actions;
- suction probability in the simulator, but not in the baseline observation.

## MDP Definition

The environment is implemented by `rl/gym_env.py::GP8RecyclingEnv`. The MDP is
event-driven at the level of completed robot skills: one Gym step can advance
simulation time by a different amount depending on the pending skill.

### Observation

The observation is a flat `float32` vector built by
`rl/common.py::build_observation`. Up to `max_objects=6` live objects are
represented. Objects use the simulator runner's queue order, and the policy
selects slots in that ordered list.

With the recommended `corners` representation, each object slot contains 13
values:

| Field | Width | Meaning |
|---|---:|---|
| `xyz_now` | 3 | Current object center/grasp position in the robot base frame. |
| `class_id` | 1 | `metal=0`, `transparent=1`, `cardboard=2`, unknown/empty `-1`. |
| `confidence` | 1 | Confidence stored by the object track. Truth tracks normally use a stable high confidence. |
| `bbox_corners` | 8 | Four ordered XY corners of the object footprint in the robot base frame. |

The 13 global values are:

| Field | Width | Meaning |
|---|---:|---|
| `robot_joints` | 6 | Current GP8 joint positions. |
| `ee_xyz` | 3 | Current end-effector position in the robot base frame. |
| `pending_object_slot` | 1 | Queue slot of the action selected on the previous Gym step. The skip/none sentinel is `max_objects`. |
| `pending_skill` | 1 | Pending skill index: `0=throw`, `1=push`. |
| `belt_speed` | 1 | Current conveyor speed in m/s. |
| `remaining_time_frac` | 1 | Remaining simulated episode time divided by the episode budget, clipped to `[0,1]`. |

Therefore, the default observation width is:

```text
6 objects * 13 values + 13 global values = 91
```

Empty slots use `xyz=(-1,-1,0)`, `class_id=-1`, `confidence=0`, and zero bbox
features.

Optional observation fields are:

- `include_suction_p`: adds one scalar in `[0,1]` per object. The width becomes
  97 for six slots. This is implemented and tested, but remains off in the
  baseline so suction settings can change without silently changing the policy
  input schema.
- `include_eta`: adds `throw_eta` and `push_eta` per object. These are the
  estimated one-step-ahead execution times and are off by default.
- `bbox_observation=size`: replaces eight corner coordinates with width and
  height. This legacy form is not the recommended baseline.

The default uses simulator truth tracks. Camera/FOV tracks remain available for
sim-to-real studies, but they introduce perception visibility and dead-reckoning
effects and therefore define a different observation process.

### Action

The logical action is:

```text
(object_slot, skill)
```

`object_slot` is `0..5` for a live queue slot or `6` for skip. `skill` is
`0=throw` or `1=push`.

`FlatMaskedActionWrapper` flattens this pair for MaskablePPO:

```text
flat_action = object_slot * 2 + skill
```

This produces 14 discrete outputs: 12 object/skill combinations and two
equivalent skip encodings. The skill component of a skip action has no physical
effect.

### Action Mask

The action mask is supplied separately from the observation. A non-skip action
is enabled only when:

- the slot contains a live object;
- the object is not already the pending target;
- an intercept exists at the one-step-ahead chain horizon;
- the skill planner accepts the placement;
- the required inverse-kinematics/trajectory feasibility checks pass.

Both skip encodings are always valid. If an invalid action reaches the Gym
environment, it is treated as skip. MaskablePPO should ordinarily prevent that
case.

Suction probability is deliberately not part of the mask. An object with low
or zero suction probability can still be selected for throw; learning whether
that is useful is left to the policy when `suction_p` is observed.

### Transition and One-Step-Ahead Convention

The selected action does not execute immediately. At Gym step `t`:

1. The action selected at step `t-1` is checked again and executed if still
   live.
2. The action selected at step `t` is stored as the next pending action.
3. The simulator advances through the executed skill, reward events are
   collected, resolved objects are removed from the queue, and the next
   observation is built.

The pending action is stored by object identity rather than only by slot index,
so queue reordering does not silently retarget it. Feasibility masking projects
candidate actions through the estimated duration of the pending skill's chain
movement; execution performs the authoritative final check.

When no pending action exists, or when a stale pending target is dropped, the
simulator advances by the runner's idle `TIME_STEP`.

### Reward

The objective is sorting by object class, independent of the selected skill:

| Object class | Correct bin |
|---|---|
| `transparent` | transparent/throw bin at nominal `(0.85, 0.00)` |
| `metal` | metal/push bin at nominal `(0.85, 0.60)` |

The resolved-object reward is:

| Outcome | Reward |
|---|---:|
| Object resolves in its class bin | `+1.0` |
| Manipulated object resolves anywhere else | `-0.3` |
| Unmanipulated object is knocked off as collateral | `-0.3` |
| Unmanipulated object naturally rides past | `0.0` |
| Skill execution fails | additional `-0.3` |

An object is rewarded once and then parked/recycled. The corrected despawn rule
uses the common floor-level `fallen` threshold for both bin hits and misses;
this prevents visually resolved bin objects from remaining in the policy queue.

Physical resolution can occur several decisions after the action that caused
it. `rl/credit.py::CauseCreditCallback` uses each event's `cause_step` to move
that reward back to the causative rollout-buffer step before PPO computes
advantages. With the current `gamma=1.0`, moving the reward preserves episode
return while correcting action credit. This move is possible only when the
causative step is still in the current PPO rollout buffer; older causes retain
the reward at its resolution step.

Reward events occurring after the episode deadline are removed from the
episode return.

### Episode Boundary

The primary horizon is simulated time:

```text
max_episode_seconds = 240
```

Reaching this observed time limit sets `terminated=True`; it is a true terminal
because `remaining_time_frac` is included in the observation. A separate
`max_steps=1000` safety cap sets `truncated=True` when reached before the time
limit. This distinction prevents value bootstrapping beyond the time-budgeted
task while retaining a guard against pathological decision loops.

## Current Simulator Profiles

### Dense randomized profile (current code default)

When `GP8_SIM_RANDOMIZE=true`, the current branch defaults to:

| Quantity | Setting |
|---|---|
| Object pool | 24 bodies |
| Belt speed | uniform `[0.05, 0.20]` m/s per episode |
| Requested spawn rate | uniform `[0.5, 2.0]` Hz per episode |
| Inter-arrival process | exponential intervals at the sampled episode rate |
| Spawn X | uniform `[0.30, 0.58]` m, subject to clearance |
| Object half-size X | uniform `[0.03, 0.06]` m |
| Object half-size Y | uniform `[0.05, 0.10]` m |
| Object half-size Z | `0.015` m |
| Spawn clearance margin | `0.01` m added to both objects in overlap checks |
| Class | randomized between transparent and metal |
| Yaw | uniform `[-pi, pi]` |
| Suction probability | metal `0.5`, transparent `0.9` unless overridden |

The requested arrival rate is not guaranteed to equal the accepted spawn rate.
A spawn is skipped when all object bodies are active or when no collision-free
X candidate is found within ten attempts.

Objects 13-24 are generated at model-build time. Their explicit inertial frame,
mass, box inertia, geometry, contact pairs, and collision parameters now match
the original 12-object pool.

### Original-density comparison profile (optional)

The controlled non-dense comparison explicitly sets:

| Quantity | Setting |
|---|---|
| Object pool | 12 bodies |
| Requested spawn rate | `[0.5, 1.0]` Hz |
| Object half-size X | `[0.0375, 0.075]` m |
| Object half-size Y | `[0.05, 0.15]` m |
| Spawn clearance margin | `0.03` m |

All other MDP and training settings are held equal when comparing this profile
with the dense profile.

## Running the Three Reference Experiments

The three latest runs share `100k` timesteps, seed 0, 16 environments, 16 PPO
steps per environment, minibatches of 128, truth tracks, corner observations,
the physical belt, and old belt-contact parameters. They differ only in object
density and suction environment:

| Run | Object profile | Suction probability | `suction_p` observed? |
|---|---|---|---|
| Dense, suction always | Current 24-object dense profile | all classes `1.0` | No |
| Dense, default suction | Current 24-object dense profile | metal `0.5`, transparent `0.9` | No |
| Original density, suction always | Legacy 12-object profile | all classes `1.0` | No |

Activate the project's Python environment, then run from the parent directory
of the `gp8_control` package. In the commands below, `python` must resolve to
that environment. Define the common arguments once:

```bash
COMMON_ARGS=(--timesteps 100000 --n-envs 16 --n-steps 16 --batch-size 128 \
  --seed 0 --max-steps 1000 --max-episode-seconds 240 \
  --bbox-observation corners --truth-tracks --device auto --tensorboard --verbose 1)
```

Run the dense, suction-always experiment:

```bash
GP8_SIM_RANDOMIZE=true GP8_SIM_PHYSICAL_BELT=true \
GP8_SIM_BELT_CONTACT_PARAMS=old GP8_RL_INCLUDE_SUCTION_P=false \
GP8_SIM_SUCTION_P=1.0 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m gp8_control.tools.train_ppo_smoke \
  --out-dir gp8_control/runs/dense_suction_always_100k_seed0 "${COMMON_ARGS[@]}"
```

Run the dense experiment with the current default suction probabilities:

```bash
GP8_SIM_RANDOMIZE=true GP8_SIM_PHYSICAL_BELT=true \
GP8_SIM_BELT_CONTACT_PARAMS=old GP8_RL_INCLUDE_SUCTION_P=false \
GP8_SIM_SUCTION_P='metal:0.5,transparent:0.9' \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m gp8_control.tools.train_ppo_smoke \
  --out-dir gp8_control/runs/dense_suction_default_100k_seed0 "${COMMON_ARGS[@]}"
```

Run the original-density, suction-always comparison (optional):

```bash
GP8_SIM_RANDOMIZE=true GP8_SIM_PHYSICAL_BELT=true \
GP8_SIM_BELT_CONTACT_PARAMS=old GP8_RL_INCLUDE_SUCTION_P=false \
GP8_SIM_SUCTION_P=1.0 GP8_SIM_BOX_COUNT=12 \
GP8_SIM_SPAWN_RATE_HZ_RANGE=0.5,1.0 \
GP8_SIM_OBJECT_HALF_X_RANGE=0.0375,0.075 \
GP8_SIM_OBJECT_HALF_Y_RANGE=0.05,0.15 \
GP8_SIM_SPAWN_CLEARANCE_MARGIN=0.03 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m gp8_control.tools.train_ppo_smoke \
  --out-dir gp8_control/runs/original_density_suction_always_100k_seed0 \
  "${COMMON_ARGS[@]}"
```

Each run writes its command, resolved simulator configuration, device, model,
TensorBoard events, and summary to its output directory. Check this metadata
before comparing runs.

Do **not** use `summary.json::mean_reward` or `summary.json::std_reward` as
validation results. The current training script computes them immediately after
learning on the training vector environment, with a very small evaluation
episode count. In our experiments these values did not reliably predict the
independent rollout performance and were sometimes misleading. This validation
path has not yet been corrected.

Use the separate, fixed multi-seed evaluation protocol for model comparison and
selection. Record per-seed episode return and throw/push/skip counts, then report
their aggregate statistics. TensorBoard training-rollout reward remains useful
for monitoring optimization progress, but it is not a replacement for that
independent evaluation.

## Fixed Multi-Seed Evaluation

Evaluate seeds `0..9` using exactly the same simulator profile and observation
schema as training. Do not rely on environment defaults: export the profile
explicitly before evaluating.

Common settings:

```bash
export GP8_SIM_RANDOMIZE=true
export GP8_SIM_PHYSICAL_BELT=true
export GP8_SIM_BELT_CONTACT_PARAMS=old
export GP8_RL_INCLUDE_SUCTION_P=false
unset GP8_SIM_METAL_SUCTION_BINARY GP8_SIM_METAL_SUCTION_MODE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

Choose one matching profile. Dense with suction always possible:

```bash
unset GP8_SIM_BOX_COUNT GP8_SIM_SPAWN_RATE_HZ_RANGE
unset GP8_SIM_OBJECT_HALF_X_RANGE GP8_SIM_OBJECT_HALF_Y_RANGE
unset GP8_SIM_SPAWN_CLEARANCE_MARGIN
export GP8_SIM_SUCTION_P=1.0
```

Dense with the current default suction probabilities:

```bash
unset GP8_SIM_BOX_COUNT GP8_SIM_SPAWN_RATE_HZ_RANGE
unset GP8_SIM_OBJECT_HALF_X_RANGE GP8_SIM_OBJECT_HALF_Y_RANGE
unset GP8_SIM_SPAWN_CLEARANCE_MARGIN
export GP8_SIM_SUCTION_P='metal:0.5,transparent:0.9'
```

Original density with suction always possible:

```bash
export GP8_SIM_SUCTION_P=1.0
export GP8_SIM_BOX_COUNT=12
export GP8_SIM_SPAWN_RATE_HZ_RANGE=0.5,1.0
export GP8_SIM_OBJECT_HALF_X_RANGE=0.0375,0.075
export GP8_SIM_OBJECT_HALF_Y_RANGE=0.05,0.15
export GP8_SIM_SPAWN_CLEARANCE_MARGIN=0.03
```

Set the checkpoint and output directory, then run all ten seeds:

```bash
MODEL=gp8_control/runs/REPLACE_WITH_RUN/ppo_smoke.zip
EVAL_DIR=gp8_control/runs/REPLACE_WITH_RUN/eval_10seed
mkdir -p "$EVAL_DIR"
set -e

for SEED in $(seq 0 9); do
  GP8_SIM_RECORD= python -m gp8_control.tools.train_ppo_smoke \
    --eval-only --model "$MODEL" --device auto \
    --seed "$SEED" --eval-steps 1000 \
    --max-steps 1000 --max-episode-seconds 240 \
    --bbox-observation corners --truth-tracks \
    --trace-jsonl "$EVAL_DIR/seed_${SEED}.jsonl" \
    > "$EVAL_DIR/seed_${SEED}.log" 2>&1
done
```

Aggregate episode return and selected/executed action counts from the traces:

```bash
python - "$EVAL_DIR" <<'PY'
import collections
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
rows = []
selected_total = collections.Counter()
executed_total = collections.Counter()

for path in sorted(root.glob("seed_*.jsonl"), key=lambda p: int(p.stem.split("_")[1])):
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not records or not records[-1].get("truncated_by_time"):
        raise RuntimeError(f"{path} did not reach the time terminal")
    selected = collections.Counter(
        "skip" if row["is_skip_slot"] else row["skill_name"] for row in records
    )
    executed = collections.Counter(
        row["executed_skill"] for row in records
        if row.get("executed") and row.get("executed_skill")
    )
    episode_return = sum(float(row["reward"]) for row in records)
    seed = int(path.stem.split("_")[1])
    rows.append((seed, episode_return))
    selected_total.update(selected)
    executed_total.update(executed)
    print(seed, "reward", episode_return, "selected", dict(selected),
          "executed", dict(executed))

returns = [value for _, value in rows]
if len(rows) != 10:
    raise RuntimeError(f"expected 10 completed seeds, found {len(rows)}")
print("reward mean/min/max", statistics.mean(returns), min(returns), max(returns))
print("selected totals", dict(selected_total))
print("executed totals", dict(executed_total))
PY
```

All ten trace files must exist and reach the 240-second simulated-time terminal
before accepting the aggregate. A model trained with `include_suction_p=true`
must also be evaluated with that flag; its observation width is incompatible
with the baseline model.

## Rendering Policy Rollouts

Rendering uses the same deterministic evaluation path. First export the common
settings and the matching simulator profile from the evaluation section above.
Then render one seed on a headless server with EGL:

```bash
MODEL=gp8_control/runs/REPLACE_WITH_RUN/ppo_smoke.zip
RENDER_DIR=gp8_control/runs/REPLACE_WITH_RUN/render
SEED=0
mkdir -p "$RENDER_DIR"

MUJOCO_GL=egl \
GP8_SIM_RECORD="$RENDER_DIR/seed_${SEED}.mp4" \
GP8_SIM_RECORD_FPS=24 \
python -m gp8_control.tools.train_ppo_smoke \
  --eval-only --model "$MODEL" --device auto \
  --seed "$SEED" --eval-steps 1000 \
  --max-steps 1000 --max-episode-seconds 240 \
  --bbox-observation corners --truth-tracks --tail-seconds 3 \
  --trace-jsonl "$RENDER_DIR/seed_${SEED}.jsonl" \
  > "$RENDER_DIR/seed_${SEED}.log" 2>&1
```

The rendered run must use the same object-density, suction, observation, and
contact settings as the checkpoint's numerical evaluation. Rendering is much
slower than evaluation without video.

Keep the JSONL trace beside each video. It provides the selected slot, selected
skill, executed skill, reward events, and object state needed to explain what is
visible in the rollout.

## Optional Suction Observation

Suction behavior and suction observation are separate controls:

```bash
export GP8_SIM_SUCTION_P='metal:0.5,transparent:0.9'
export GP8_RL_INCLUDE_SUCTION_P=true
```

With `include_suction_p=false`, the simulator still samples suction outcomes,
but the policy cannot condition its action on the object's probability. With
it enabled, each object's probability is appended to its slot and the model
input width changes from 91 to 97. Models trained with one width cannot be
loaded into the other schema without adaptation.

The metal-only experiment controls remain available:

- `GP8_SIM_METAL_SUCTION_BINARY=true` with mode `binary`: each metal object is
  assigned `0` or `1`;
- mode `zero`: every metal object is ungraspable;
- mode `one`: every metal object is graspable.

Suction probability does not affect push execution directly. It controls
whether suction attachment succeeds during grasp-based behavior.

## Known Limitations

- PPO results remain seed-sensitive and can converge to a throw-only policy.
- Push is executable and sometimes learned, but is not consistently selected
  under the general randomized objective.
- Ground-truth tracks are intentionally optimistic relative to real camera
  observations.
- The current suction-default task is partially observable unless
  `include_suction_p` is enabled.
- More objects and a higher requested spawn range do not guarantee a matching
  accepted arrival rate because spawn clearance and pool availability still
  apply.
- GPU acceleration applies to policy optimization; MuJoCo environments remain
  CPU processes, so simulator stepping is still the main throughput cost.
- Real-robot policy execution should remain in shadow/evaluation mode until a
  selected checkpoint and its exact observation schema are reviewed.

## Main Files

- `rl/common.py`: observation fields, dimensions, class IDs, and action schema.
- `rl/gym_env.py`: Gym transition, episode boundary, and late-reward filtering.
- `rl/sim_runner.py`: one-step-ahead execution, feasibility masks, and queue
  maintenance.
- `rl/credit.py`: delayed reward-credit reassignment for PPO rollouts.
- `rl/sb3_wrapper.py`: flat pairwise action space and masks.
- `tools/train_ppo_smoke.py`: training/evaluation entry point and run metadata.
- `backends/mujoco_sim.py`: physical belt, object generation, suction, bin
  resolution, reward events, and recycling.
- `rl/real_runner.py` and `rl/real_shadow.py`: real-robot mirror and shadow
  integration.

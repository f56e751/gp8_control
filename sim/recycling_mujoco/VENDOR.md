# Vendored: robot-recycling-mujoco-env (MuJoCo throw simulator)

Physics (MuJoCo) digital twin of the GP8 conveyor pick-and-throw setup, vendored
into gp8_control to drive a **preview** of throws/pushes with real contact +
camera rendering (more accurate than the kinematic `mock_robot` + `fake_belt`
SIL under `launch/sim_bringup.launch.py`).

## Provenance

- **Source**: https://github.com/tmjeong1103/robot-recycling-mujoco-env.git
- **Branch**: `wip-throwing-fixes`  (the throwing simulator — NOT `master`)
- **Commit**: `63e0922c52fcda043521a17754435594667b7da1`
  ("Finalized for distribution", 2026-06-12)
- **Vendored on**: 2026-06-25

Vendored (copied) rather than submodule/pip because upstream is versioned via a
WIP branch and is effectively frozen; a self-contained snapshot is simpler and
more stable. Record the commit above so a future update is a bounded diff.

## What was copied

- `recycling_env/` — the env package. Entry: `gym_env.RecyclingBBoxGymEnv`
  (the throw env), wrapping `sim_env.RecyclingSimEnv` (the MuJoCo model/data/
  renderer wrapper). Pulls in action_control, controller, demo_policy, gripper,
  gui_utils, perception, tracker, trajectory.
- `utils/` — Lie_numpy, yaskawa_gp8 (GP8 kinematics), trajectory_calculation,
  pushing_trajectory_generator.
- `models/` — model.py (`FCN`) + `ckpt/NN_..._B10.pt` (throw-param NN weight,
  same family as gp8_control's `model/*.pt`).
- `meshes/` — robot + scene STL/OBJ assets (~48 MB; required to load the model).
- `scene.xml` (top-level) → includes `combined_test.xml` (robot + conveyor +
  objects); `floor_isaac_style.xml`.
- `verify_throw_chain.py` / `verify_throw_chain_gui.py` — runnable throw-sim
  demos (good reference for driving the env).
- `environment.yml` — upstream conda deps.

## Omitted

`README.md`, `CLAUDE.md`, `ENVIRONMENT.md`, `.git`, `notebooks/` (the WIP branch
has none). Docs only — not needed to run.

## Dependencies (NOT in gp8_control's uv venv by default)

`mujoco>=3.1`, `numpy`, `scipy`, `gymnasium>=0.29`, `torch`, `opencv-python>=4.8`,
`matplotlib` (see `environment.yml`; upstream uses conda env `iitp`, python 3.10).

## How to run (standalone)

This tree uses **top-level** packages (`recycling_env`, `utils`, `models`), so
run with this directory as the import root:

```bash
cd sim/recycling_mujoco
python verify_throw_chain.py            # headless
python verify_throw_chain_gui.py        # mujoco.viewer GUI
# or:  from recycling_env.gym_env import RecyclingBBoxGymEnv
```

## Integration notes

- **NOT part of the colcon/ROS build** — `sim/` is not registered in `setup.py`,
  so it isn't installed with the gp8_control ROS package. Run it from source.
- **Import-path isolation**: this tree's `utils` / `models` are top-level names
  and would shadow other packages. Do NOT add this dir to the same PYTHONPATH as
  the ROS workspace (gp8_control has its own `gp8_control.utils`). Run the sim
  from its own dir / process.
- **Sync**: to pull a future upstream change, `git diff <commit-above>..<new>`
  on the source repo and apply the delta here. Keep any gp8_control-side glue
  OUTSIDE this dir (don't edit vendored files in place) so re-vendoring stays a
  clean replace.
- **Kinematics agreement**: before trusting the preview, verify gp8_control's
  `robots/gp8.py` FK and this model's geometry agree for the same joint angles
  (axis names S/L/U/R/B/T match; link/frame calibration may not).

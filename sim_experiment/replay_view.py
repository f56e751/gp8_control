"""Replay a push_diag state log: render an mp4 window, or play it in the viewer.

The log (``<tag>_state.npz``) holds the full sim state every 20 ms; together
with the compiled model (``model.mjb``) any moment can be re-watched from any
angle. Only kinematics are replayed (qpos + box sizes/colours), no physics.

  # mp4 of sim time 120.0-126.0 s (headless server: MUJOCO_GL=egl)
  python -m gp8_control.sim_experiment.replay_view RUN/model.mjb RUN/push_first_seed0_state.npz \
      --t0 120 --t1 126 --mp4 out.mp4
  # interactive (a machine with a display; space = pause)
  python -m gp8_control.sim_experiment.replay_view RUN/model.mjb RUN/push_first_seed0_state.npz \
      --t0 120 --t1 126 --view --speed 0.25

Clip times are listed in ``<tag>_clips.json`` / ``<tag>_push_objects.jsonl``.
Copying the run folder (model.mjb + *_state.npz) to a laptop with
``pip install mujoco numpy`` is enough for ``--view``; this file has no other
project imports.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

_W, _H, _FPS = 960, 540, 30


def _load(model_path, state_path):
    m = mujoco.MjModel.from_binary_path(str(model_path))
    z = np.load(str(state_path))
    return m, mujoco.MjData(m), z


def _apply(m, d, z, k: int) -> None:
    d.qpos[:] = z["qpos"][k]
    gids = z["box_geom_ids"]
    m.geom_size[gids] = z["box_size"][k]
    m.geom_rgba[gids] = z["box_rgba"][k]
    mujoco.mj_forward(m, d)


def _frames(t: np.ndarray, t0: float, t1: float, fps: float):
    for tt in np.arange(max(t0, t[0]), min(t1, t[-1]), 1.0 / fps):
        yield int(np.searchsorted(t, tt).clip(0, len(t) - 1))


def _camera(m, d, z, focus_serial=None):
    """Look at the belt / push-bin area in the robot base frame (body
    ``yaskawa_robot``); base-frame (0.6, 0.3) sits between the belt lane and
    the metal push bin at (0.85, 0.60)."""
    mujoco.mj_forward(m, d)
    base = d.body("yaskawa_robot").xpos.copy()
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 200.0, -35.0, 1.8
    cam.lookat[:] = base + np.array([0.6, 0.3, 0.0])
    return cam


def render_clip(model_path, state_path, out_path, t0: float, t1: float,
                focus_serial=None, fps: float = _FPS) -> int:
    """Write an mp4 of sim time [t0, t1]. Returns the frame count."""
    import cv2
    m, d, z = _load(model_path, state_path)
    t = z["t"]
    renderer = mujoco.Renderer(m, height=_H, width=_W)
    cam = _camera(m, d, z, focus_serial)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (_W, _H))
    n = 0
    try:
        for k in _frames(t, t0, t1, fps):
            _apply(m, d, z, k)
            renderer.update_scene(d, camera=cam)
            img = renderer.render()
            cv2.putText(img, f"t={t[k]:.2f}s", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2)
            writer.write(img[:, :, ::-1])
            n += 1
    finally:
        writer.release()
        renderer.close()
    return n


def view(model_path, state_path, t0: float, t1: float, speed: float = 1.0) -> None:
    import mujoco.viewer
    m, d, z = _load(model_path, state_path)
    t = z["t"]
    ks = list(_frames(t, t0, t1, _FPS))
    with mujoco.viewer.launch_passive(m, d) as v:
        while v.is_running():
            for k in ks:
                if not v.is_running():
                    break
                _apply(m, d, z, k)
                v.sync()
                time.sleep(1.0 / _FPS / max(speed, 1e-3))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="replay_view")
    ap.add_argument("model")
    ap.add_argument("state")
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=1e9)
    ap.add_argument("--mp4", default="")
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0)
    a = ap.parse_args(argv)
    if a.mp4:
        n = render_clip(a.model, a.state, Path(a.mp4), a.t0, a.t1)
        print(f"wrote {n} frames -> {a.mp4}")
    if a.view:
        view(a.model, a.state, a.t0, a.t1, a.speed)
    if not (a.mp4 or a.view):
        z = np.load(a.state)
        print(f"state log: {len(z['t'])} frames, t={z['t'][0]:.2f}..{z['t'][-1]:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

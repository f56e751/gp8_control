"""Watch a final Safe PPO policy or baseline on the frozen native workload.

The viewer receives a separate model/data copy. Rendering, mouse interaction,
pausing and wall-clock pacing do not modify the evaluated physical state.
Only this entry point substitutes a display subclass; frozen training sources
and saved evaluation results are left intact.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import copy
import json
import math
import time

import mujoco

from gp8_control.rl import safe_simulator_env as environment
from gp8_control.rl.safe_ppo import SafePPO
from gp8_control.tools.train_safe_simulator import run_paths, sha, source_hashes


class Display:
    def __init__(self, core, *, speed, paused):
        import mujoco.viewer
        self.core, self.speed, self.paused = core, speed, paused
        self.model = copy.copy(core.model)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_copyData(self.data, self.model, core.data)
        self.handle = mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.key)
        self.handle.cam.azimuth = 135.
        self.handle.cam.elevation = -20.
        self.handle.cam.distance = 2.2
        self.handle.cam.lookat[:] = (.25, 0., .75)
        self.reset_clock()
        self.last_frame = -math.inf
        self.sync()

    def key(self, keycode):
        if keycode == 32:
            self.paused = not self.paused

    def reset_clock(self):
        self.wall_anchor = time.monotonic()
        self.sim_anchor = self.core.sim_time()

    def check_open(self):
        if not self.handle.is_running():
            raise environment.EpisodeStopped('viewer_closed')

    def sync(self):
        self.check_open()
        with self.handle.lock():
            self.model.geom_size[:] = self.core.model.geom_size
            self.model.geom_rgba[:] = self.core.model.geom_rgba
            mujoco.mj_copyData(self.data, self.model, self.core.data)
            mujoco.mj_forward(self.model, self.data)
        self.handle.sync()

    def before_step(self):
        self.check_open()
        waited = False
        while self.paused:
            waited = True
            self.sync()
            time.sleep(.02)
        if waited:
            self.reset_clock()

    def after_step(self):
        now = self.core.sim_time()
        if now - self.last_frame < 1./30.:
            return
        self.sync()
        self.last_frame = now
        deadline = self.wall_anchor + (now-self.sim_anchor)/self.speed
        while time.monotonic() < deadline and not self.paused:
            self.check_open()
            time.sleep(min(.02, max(0., deadline-time.monotonic())))

    def close(self):
        self.handle.close()


def replay(args):
    out, checkpoint = run_paths(args.run)
    protocol_path = out/'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    if protocol['source_sha256'] != source_hashes():
        raise RuntimeError('Frozen simulator/training source changed; the recorded experiment cannot be replayed faithfully.')
    policy = None
    if args.policy != 'baseline':
        label = int(args.policy[-1])
        policy = SafePPO(protocol['model_initialization_seeds'][label], obs_size=protocol['observation_size'])
        metadata = policy.load(checkpoint/f'safe_seed{label}.pt')
        if (metadata['run'] != args.run or metadata['update'] != protocol['training']['updates']
                or metadata['protocol_sha256'] != sha(protocol_path)):
            raise RuntimeError('Checkpoint is not the fixed final policy for this experiment.')
        policy.network.eval()
    if not args.headless and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        raise RuntimeError('No desktop display found. Run this command in a desktop terminal, or use --headless.')

    original_core = environment.OfferedTapeCore

    class PreviewCore(original_core):
        def __init__(self, *values, **keywords):
            self.display = None
            super().__init__(*values, **keywords)

        def begin_episode(self):
            super().begin_episode()
            if not args.headless:
                self.display = Display(self, speed=args.speed, paused=args.paused)

        def advance_steps(self, n=1):
            if self.display is not None:
                self.display.before_step()
            super().advance_steps(n)
            if self.display is not None:
                self.display.after_step()

        def stop(self, timeout_sec=2.):
            try:
                if self.display is not None:
                    self.display.close()
                    self.display = None
            finally:
                super().stop(timeout_sec)

    print(f"Policy={args.policy}, scene_seed={args.scene_seed}, objects={protocol['episode_config']['offer_count']}, "
          f"speed={args.speed}x. Space: pause/resume; close window or Ctrl+C: stop.", flush=True)
    environment.OfferedTapeCore = PreviewCore
    try:
        result = environment.collect_episode(args.scene_seed, policy, protocol['episode_config'], deterministic=True)
    finally:
        environment.OfferedTapeCore = original_core
    summary = result['summary']
    print(json.dumps({key:summary[key] for key in ('scene_seed', 'offered', 'correct_bin',
        'task_success_percent', 'contacts', 'collateral_falls', 'physical_unresolved',
        'termination_reason', 'sim_seconds')}, ensure_ascii=False), flush=True)
    if args.verify_recorded:
        name = 'baseline' if args.policy == 'baseline' else f'safe_seed{args.policy[-1]}'
        recorded = json.loads((out/'evaluation'/f'{name}_scene{args.scene_seed}.json').read_text())
        # Wall time is intentionally different. All physical traces and other
        # measurements must be bit-identical in a headless replay.
        expected = {key:value for key,value in recorded['summary'].items() if key != 'wall_seconds'}
        actual = {key:value for key,value in summary.items() if key != 'wall_seconds'}
        if expected != actual or recorded['trace'] != json.loads(json.dumps(result['trace'])):
            raise RuntimeError('Replay differs from the stored physical evaluation.')
        print('Recorded summary and full physical trace match exactly.', flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='feature_safe_20261020_v1')
    parser.add_argument('--policy', choices=('baseline', 'safe0', 'safe1'), default='safe0')
    parser.add_argument('--scene-seed', type=int, default=3400000)
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--paused', action='store_true', help='Start paused; press Space to begin')
    parser.add_argument('--headless', action='store_true', help='Replay without opening a viewer')
    parser.add_argument('--verify-recorded', action='store_true', help='Require exact equality with the saved rollout')
    args = parser.parse_args(argv)
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error('--speed must be positive and finite')
    try:
        return replay(args)
    except KeyboardInterrupt:
        return 130
    except (RuntimeError, FileNotFoundError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())

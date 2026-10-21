"""Replay a recorded fixed-reset scene with a separate display-only model."""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import json
import math

from gp8_control.rl import safe_fixed_scene_env as environment
from gp8_control.rl.safe_v2.ppo import SafePPO
from gp8_control.tools.replay_safe_simulator import Display
from gp8_control.tools.train_safe_simulator_v2 import ROOT, sha, source_hashes


def replay(args):
    out = ROOT/'runs'/args.run
    protocol = json.loads((out/'protocol.json').read_text())
    source_out = ROOT/'runs'/args.source_run
    source_protocol = json.loads((source_out/'protocol.json').read_text())
    if (protocol['model_run'] != args.source_run
            or protocol['training_protocol_sha256'] != sha(source_out/'protocol.json')):
        raise RuntimeError('The fixed-scene protocol used a different model experiment')
    if source_protocol['source_sha256'] != source_hashes():
        raise RuntimeError('The frozen native/v2 runtime has changed')
    fixed_source = protocol.get('source_sha256', {})
    for name, expected in fixed_source.items():
        if sha(ROOT/name) != expected:
            raise RuntimeError(f'The recorded fixed-scene source has changed: {name}')
    config = protocol.get('episode_config', protocol.get('config', environment.default_config()))
    policy = None
    if args.policy.startswith('safe'):
        label = int(args.policy[-1])
        policy = SafePPO(source_protocol['model_initialization_seeds'][label], obs_size=107)
        path = ROOT/'checkpoints'/args.source_run/f'safe_seed{label}.pt'
        if sha(path) != protocol['checkpoint_sha256'][str(path.relative_to(ROOT))]:
            raise RuntimeError('The recorded final checkpoint has changed')
        meta = policy.load(path)
        if (meta.get('run') != args.source_run or meta.get('update') != source_protocol['training']['updates']
                or meta.get('protocol_sha256') != sha(source_out/'protocol.json')):
            raise RuntimeError('Expected the frozen final model')
        policy.network.eval()
    if not args.headless and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        raise RuntimeError('Run the viewer from a desktop terminal, or use --headless')
    original = environment.FixedSceneCore

    class PreviewCore(original):
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

    environment.FixedSceneCore = PreviewCore
    print(f'Policy={args.policy}; {args.skill}; neighbors={args.neighbors}; scene={args.scene_seed}. '
          'Space: pause/resume. The initial belt/perception prelude runs before the one skill.', flush=True)
    try:
        if args.policy == 'baseline':
            result = environment.collect_baseline_episode(args.scene_seed, args.skill, args.neighbors, config)
        else:
            result = environment.collect_episode(args.scene_seed, args.skill, args.neighbors,
                                                  policy, config, deterministic=True)
    finally:
        environment.FixedSceneCore = original
    print(json.dumps({k: result['summary'][k] for k in (
        'scene_seed', 'requested_skill', 'neighbor_count', 'intended_target_selected',
        'task_success_percent', 'action_contacts', 'action_impulse_ns',
        'action_collateral_falls', 'observed_action_seconds')}, ensure_ascii=False))
    if args.verify_recorded:
        arm = 'safe_seed'+args.policy[-1] if args.policy.startswith('safe') else args.policy
        path = out/'evaluation'/f'{args.skill}_n{args.neighbors}_scene{args.scene_seed}_{arm}.json'
        recorded = json.loads(path.read_text())
        exclude = {'wall_seconds'}
        expected = {k:v for k,v in recorded['summary'].items() if k not in exclude}
        actual = {k:v for k,v in result['summary'].items() if k not in exclude}
        trace = json.loads(json.dumps(result['trace'], default=environment._json))
        if expected != actual or recorded['trace'] != trace:
            raise RuntimeError('The replay differs from the saved physical evaluation')
        print('Recorded summary and full physical trace match exactly.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='feature_safe_fixed_20261021_v1')
    parser.add_argument('--source-run', default='feature_safe_route_20261021_v2')
    parser.add_argument('--policy', choices=('baseline', 'v2_zero', 'safe0', 'safe1'), default='safe0')
    parser.add_argument('--skill', choices=('throw', 'push'), default='throw')
    parser.add_argument('--neighbors', type=int, choices=(2,5), default=5)
    parser.add_argument('--scene-seed', type=int, default=5200000)
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--paused', action='store_true')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--verify-recorded', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error('--speed must be finite and positive')
    try:
        replay(args)
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

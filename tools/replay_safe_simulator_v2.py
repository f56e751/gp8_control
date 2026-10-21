"""Replay the final v2 policy with a render-only copy of the native simulator."""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import json
import math

from gp8_control.rl.safe_v2 import env as environment
from gp8_control.rl.safe_v2.ppo import SafePPO
from gp8_control.tools.replay_safe_simulator import Display
from gp8_control.tools.train_safe_simulator_v2 import run_paths, sha, source_hashes


def replay(args):
    out, checkpoint = run_paths(args.run)
    protocol_path = out/'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    if protocol['source_sha256'] != source_hashes():
        raise RuntimeError('The frozen v2 runtime differs from the recorded experiment')
    if not (out/'training_summary.json').is_file():
        raise RuntimeError('Wait for the fixed final training checkpoint')
    policy = None
    if args.policy.startswith('safe'):
        label = int(args.policy[-1])
        policy = SafePPO(protocol['model_initialization_seeds'][label], obs_size=107)
        metadata = policy.load(checkpoint/f'safe_seed{label}.pt')
        if (metadata['run'] != args.run or metadata['update'] != protocol['training']['updates']
                or metadata['protocol_sha256'] != sha(protocol_path)):
            raise RuntimeError('Not the fixed final checkpoint')
        policy.network.eval()
    if not args.headless and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        raise RuntimeError('Use a desktop terminal for the viewer, or add --headless')
    original = environment.RouteCore

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

    print(f'Policy={args.policy}, scene={args.scene_seed}, speed={args.speed}x; Space: pause/resume.', flush=True)
    environment.RouteCore = PreviewCore
    try:
        if args.policy == 'baseline':
            result = environment.collect_baseline_episode(args.scene_seed, protocol['episode_config'])
        else:
            result = environment.collect_episode(args.scene_seed, policy, protocol['episode_config'], deterministic=True)
    finally:
        environment.RouteCore = original
    summary = result['summary']
    print(json.dumps({key:summary[key] for key in ('scene_seed','offered','correct_bin','task_success_percent',
        'contacts','collateral_falls','physical_unresolved','termination_reason','sim_seconds')}, ensure_ascii=False))
    if args.verify_recorded:
        name = 'safe_seed'+args.policy[-1] if args.policy.startswith('safe') else args.policy
        recorded = json.loads((out/'evaluation'/f'{name}_scene{args.scene_seed}.json').read_text())
        exclude = ('wall_seconds','worker_cpu_seconds','worker_peak_rss_mb','worker_post_gc_rss_mb')
        expected = {k:v for k,v in recorded['summary'].items() if k not in exclude}
        actual = {k:v for k,v in summary.items() if k not in exclude}
        if expected != actual or recorded['trace'] != json.loads(json.dumps(result['trace'])):
            raise RuntimeError('Replay differs from the saved physical evaluation')
        print('Full physical trace matches the saved evaluation exactly.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='feature_safe_route_20261021_v2')
    parser.add_argument('--policy', choices=('baseline','v2_zero','safe0','safe1'), default='safe0')
    parser.add_argument('--scene-seed', type=int, default=3500000)
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--paused', action='store_true')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--verify-recorded', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error('--speed must be positive and finite')
    try:
        replay(args)
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

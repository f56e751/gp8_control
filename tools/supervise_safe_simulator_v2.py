"""Bound worker lifetime around the unchanged native Safe PPO trainer.

Run from the repository parent.  Each child receives the complete, unchanged
training budget; only the first child uses --fresh.  Later pools resume the
last atomic committed_update marker and its full optimizer/RNG checkpoints.
Log/history sidecars are siblings of the training run, so they cannot trip the
frozen trainer's fresh-directory guard.  This module never imports the trainer.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
TRAINER = 'gp8_control.tools.train_safe_simulator_v2'


def _save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def _committed(marker, budget):
    if not marker.is_file():
        return 0
    update = json.loads(marker.read_text()).get('update')
    if isinstance(update, bool) or not isinstance(update, int) or not 0 <= update <= budget:
        raise RuntimeError(f'Invalid committed update in {marker}: {update!r}')
    return update


def _command(args, fresh):
    command = [sys.executable, '-m', TRAINER, 'train',
               '--fresh' if fresh else '--resume', '--run', args.run,
               '--workers', str(args.workers), '--updates', str(args.updates),
               '--episodes', str(args.episodes),
               '--warmup-episodes', str(args.warmup_episodes),
               '--budget-fraction', str(args.budget_fraction),
               '--warmup-seed-start',str(args.warmup_seed_start),
               '--training-seed-start',str(args.training_seed_start),
               '--training-seed-stride',str(args.training_seed_stride)]
    if args.episode_config:
        command.extend(('--episode-config', str(Path(args.episode_config).expanduser().resolve())))
    return command


def _stop_process_group(process, timeout=10.):
    """Stop only this newly created child session, including pool descendants."""
    signaled = False
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            signaled = True
        except ProcessLookupError:
            pass
    try:
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    finally:
        # A parent may exit before its children. The isolated group remains
        # ours. This also runs if a second interrupt arrives during cleanup.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return signaled


def supervise(args, *, popen=None, stop_group=None, sleep=None):
    """Run training pools; unexpected child exits always stop supervision.

    Optional callables are test seams. No existing process is inspected or
    controlled: cleanup is limited to sessions started by this invocation.
    """
    popen = subprocess.Popen if popen is None else popen
    stop_group = _stop_process_group if stop_group is None else stop_group
    sleep = time.sleep if sleep is None else sleep
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}', args.run):
        raise ValueError('Use one plain run name, without path components')
    out, checkpoints = ROOT/'runs'/args.run, ROOT/'checkpoints'/args.run
    marker = out/'committed_update.json'
    if args.fresh and any(path.exists() and any(path.iterdir()) for path in (out, checkpoints)):
        raise RuntimeError('Fresh run already contains artifacts; choose a new name or explicitly resume')
    if args.resume and not (out/'protocol.json').is_file():
        raise RuntimeError('Resume requires an existing frozen training protocol')
    sidecar = ROOT/'runs'/f'{args.run}_supervisor'
    if sidecar.is_dir() and any(path.name not in ('execution_restarts.json',
                                                'execution_restarts.json.tmp', 'execution_log.txt')
                               for path in sidecar.iterdir()):
        raise RuntimeError(f'Supervisor sidecar contains unrelated artifacts: {sidecar}')
    sidecar.mkdir(parents=True, exist_ok=True)
    history_path, log_path = sidecar/'execution_restarts.json', sidecar/'execution_log.txt'
    if history_path.is_file():
        history = json.loads(history_path.read_text())
        if history.get('run') != args.run:
            raise RuntimeError('Supervisor sidecar belongs to another run')
    else:
        history = dict(run=args.run, training_run=str(out), segments=[])
    history.update(policy='resume identical committed full state; algorithm, seeds and full budget unchanged',
                   requested_updates=args.updates, workers=args.workers,
                   maximum_committed_updates_per_pool=args.updates_per_pool,
                   status='running', last_started_unix=time.time())
    _save_json(history_path, history)
    first_child = True
    while True:
        first = _committed(marker, args.updates)
        if first >= args.updates and (out/'training_summary.json').is_file():
            history.update(status='complete', ended_unix=time.time())
            _save_json(history_path, history)
            print('ALL TRAINING COMPLETE', flush=True)
            return 0
        goal = min(args.updates, first + args.updates_per_pool)
        command = _command(args, fresh=bool(first_child and args.fresh))
        segment = dict(start_update=first, stop_after_update=goal,
                       workers=args.workers, command=command,
                       began_unix=time.time(), planned_restart=False)
        history['segments'].append(segment)
        process, interruption = None, None
        with log_path.open('a', buffering=1) as log:
            try:
                process = popen(command, cwd=ROOT.parent, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
                segment['pid'] = process.pid
                _save_json(history_path, history)
                print(f'POOL update {first}/{args.updates}; workers={args.workers}; '
                      f'restart after committed update {goal}; log={log_path}', flush=True)
                last = first
                while process.poll() is None:
                    current = _committed(marker, args.updates)
                    if current != last:
                        print(f'COMMITTED {current}/{args.updates}', flush=True)
                        last = current
                    if current >= goal and current < args.updates:
                        # Recheck liveness so an unrelated child failure at the
                        # boundary is not mistaken for our planned restart.
                        if process.poll() is None:
                            segment['planned_restart'] = bool(stop_group(process))
                        break
                    sleep(.25)
                result = process.wait()
            except BaseException as exc:
                interruption = type(exc).__name__
                history.update(status='interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed',
                               ended_unix=time.time(), error=str(exc))
                raise
            finally:
                if process is not None:
                    stop_group(process)
                    segment['exit_code'] = process.returncode
                segment.update(end_update=_committed(marker, args.updates), ended_unix=time.time())
                if interruption:
                    segment['interruption'] = interruption
                _save_json(history_path, history)
        current = segment['end_update']
        if segment['planned_restart']:
            if result not in (0, -signal.SIGTERM, -signal.SIGKILL):
                history.update(status='failed', ended_unix=time.time(),
                               error=f'Trainer failed with exit {result} at a planned restart boundary')
                _save_json(history_path, history)
                raise RuntimeError(f"{history['error']}; see {log_path}")
            if current < goal or current <= first:
                history.update(status='failed', error='Planned restart made no complete committed progress')
                _save_json(history_path, history)
                raise RuntimeError(history['error'])
            first_child = False
            continue
        if result != 0 or current < args.updates:
            history.update(status='failed', ended_unix=time.time(),
                           error=f'Trainer exited {result} at update {current}/{args.updates}')
            _save_json(history_path, history)
            raise RuntimeError(f"{history['error']}; see {log_path}")
        history.update(status='complete', ended_unix=time.time())
        _save_json(history_path, history)
        print('ALL TRAINING COMPLETE', flush=True)
        return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--updates', type=int, default=128)
    parser.add_argument('--episodes', type=int, default=12)
    parser.add_argument('--warmup-episodes', type=int, default=16)
    parser.add_argument('--budget-fraction', type=float, default=.8)
    parser.add_argument('--warmup-seed-start',type=int,default=4000000)
    parser.add_argument('--training-seed-start',type=int,default=4100000)
    parser.add_argument('--training-seed-stride',type=int,default=100000)
    parser.add_argument('--episode-config')
    parser.add_argument('--updates-per-pool', '--restart-every', dest='updates_per_pool', type=int, default=6)
    startup = parser.add_mutually_exclusive_group(required=True)
    startup.add_argument('--fresh', action='store_true')
    startup.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if (min(args.workers, args.updates, args.episodes, args.warmup_episodes, args.updates_per_pool) < 1
            or not math.isfinite(args.budget_fraction) or not 0 < args.budget_fraction <= 1):
        parser.error('Counts must be positive and the budget fraction in(0,1]')
    return args


def main(argv=None):
    args = parse_args(argv)
    previous_handler = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    try:
        return supervise(args)
    except KeyboardInterrupt:
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == '__main__':
    raise SystemExit(main())

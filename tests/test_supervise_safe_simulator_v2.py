"""Exercise pool restart/failure semantics without starting any subprocess."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gp8_control.tools import supervise_safe_simulator_v2 as supervisor


class _Child:
    def __init__(self, scenario, pid):
        self.scenario, self.pid, self.returncode = scenario, pid, None

    def poll(self):
        if self.returncode is None and self.scenario.committed >= self.scenario.budget:
            self.returncode = 0
        return self.returncode

    def wait(self, timeout=None):
        if self.poll() is None:
            raise AssertionError('The fake child must be finished or explicitly stopped')
        return self.returncode


class _Scenario:
    def __init__(self, root, budget, failure=False, interrupt=False):
        self.root, self.budget = root, budget
        self.failure, self.interrupt = failure, interrupt
        self.commands, self.children, self.stops = [], [], []
        self.committed, self.ticks = 0, 0

    def popen(self, command, *, cwd, stdout, stderr, start_new_session):
        self.assertions = (cwd, stderr, start_new_session)
        self.commands.append(command)
        out = self.root/'runs'/'demo'
        if '--fresh' in command:
            # Supervision sidecars must not defeat the trainer's fresh guard.
            if out.exists() and any(out.iterdir()):
                raise AssertionError('Fresh run was polluted before trainer start')
            out.mkdir(parents=True, exist_ok=True)
            (out/'protocol.json').write_text('{}')
        child = _Child(self, 10000 + len(self.commands))
        self.children.append(child)
        self.active = child
        return child

    def sleep(self, seconds):
        self.ticks += 1
        if self.interrupt:
            raise KeyboardInterrupt()
        if self.failure and self.ticks == 2:
            self.active.returncode = 7
            return
        self.committed += 1
        marker = self.root/'runs'/'demo'/'committed_update.json'
        marker.write_text(json.dumps({'update': self.committed}))

    def stop(self, child):
        self.stops.append(child.pid)
        if child.poll() is None:
            child.returncode = -15
            return True
        return False


class SupervisorV2Tests(unittest.TestCase):
    def _args(self):
        return supervisor.parse_args(['--fresh', '--run', 'demo', '--updates', '5',
            '--episodes', '12', '--warmup-episodes', '16', '--workers', '8',
            '--updates-per-pool', '2'])

    def test_complete_commits_restart_as_full_budget_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'gp8_control'
            scenario = _Scenario(root, budget=5)
            with patch.object(supervisor, 'ROOT', root), redirect_stdout(io.StringIO()):
                result = supervisor.supervise(self._args(), popen=scenario.popen,
                    stop_group=scenario.stop, sleep=scenario.sleep)
            self.assertEqual(result, 0)
            self.assertEqual(len(scenario.commands), 3)
            self.assertIn('--fresh', scenario.commands[0])
            self.assertTrue(all('--resume' in command for command in scenario.commands[1:]))
            for command in scenario.commands:
                self.assertIn('gp8_control.tools.train_safe_simulator_v2',command)
                self.assertEqual(command[command.index('--budget-fraction')+1],'0.8')
                self.assertEqual(command[command.index('--updates')+1], '5')
                self.assertEqual(command[command.index('--episodes')+1], '12')
                self.assertEqual(command[command.index('--warmup-episodes')+1], '16')
            history = json.loads((root/'runs'/'demo_supervisor'/'execution_restarts.json').read_text())
            self.assertEqual(history['status'], 'complete')
            self.assertEqual([s['start_update'] for s in history['segments']], [0, 2, 4])
            self.assertEqual([s['end_update'] for s in history['segments']], [2, 4, 5])
            self.assertEqual([s['planned_restart'] for s in history['segments']], [True, True, False])
            self.assertTrue(scenario.assertions[2])
            self.assertTrue(all(child.pid in scenario.stops for child in scenario.children))

    def test_unexpected_child_failure_stops_without_another_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'gp8_control'
            scenario = _Scenario(root, budget=5, failure=True)
            with patch.object(supervisor, 'ROOT', root), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'exited 7'):
                    supervisor.supervise(self._args(), popen=scenario.popen,
                        stop_group=scenario.stop, sleep=scenario.sleep)
            self.assertEqual(len(scenario.commands), 1)
            history = json.loads((root/'runs'/'demo_supervisor'/'execution_restarts.json').read_text())
            self.assertEqual(history['status'], 'failed')
            self.assertEqual(history['segments'][0]['end_update'], 1)
            self.assertFalse(history['segments'][0]['planned_restart'])
            self.assertIn(scenario.children[0].pid, scenario.stops)

    def test_keyboard_interrupt_cleans_up_owned_child_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'gp8_control'
            scenario = _Scenario(root, budget=5, interrupt=True)
            with patch.object(supervisor, 'ROOT', root), redirect_stdout(io.StringIO()):
                with self.assertRaises(KeyboardInterrupt):
                    supervisor.supervise(self._args(), popen=scenario.popen,
                        stop_group=scenario.stop, sleep=scenario.sleep)
            self.assertEqual(len(scenario.commands), 1)
            self.assertEqual(scenario.children[0].returncode, -15)
            history = json.loads((root/'runs'/'demo_supervisor'/'execution_restarts.json').read_text())
            self.assertEqual(history['status'], 'interrupted')


if __name__ == '__main__':
    unittest.main()

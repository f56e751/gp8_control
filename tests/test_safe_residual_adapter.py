"""Boundary and stream-contract checks for the app-simulator residual seam."""
import unittest
from types import SimpleNamespace

import numpy as np

from gp8_control.rl.safe_residual_adapter import (
    SafeResidualAdapter, correct_trajectory, stream_samples,
)


class ResidualContractTest(unittest.TestCase):
    def setUp(self):
        self.times = np.arange(41, dtype=float) * .05
        self.q = np.zeros((6, self.times.size))
        self.dq = np.zeros_like(self.q)

    def test_zero_action_returns_original_objects(self):
        q, dq = correct_trajectory(self.q, self.dq, self.times, np.zeros(6), .96,
                                   release_time=1.)
        self.assertIs(q, self.q)
        self.assertIs(dq, self.dq)

    def test_throw_preserves_release_window_on_actual_stream_grid(self):
        q, dq = correct_trajectory(self.q, self.dq, self.times, np.ones(6), .96,
                                   release_time=1.)
        grid, actual = stream_samples(q, dq, self.times)
        _, nominal = stream_samples(self.q, self.dq, self.times)
        protected = (grid >= .96 - 1e-12) & (grid <= 1.04 + 1e-12)
        np.testing.assert_array_equal(actual[protected], nominal[protected])
        np.testing.assert_array_equal(q[3:], self.q[3:])
        np.testing.assert_array_equal(dq[3:], self.dq[3:])
        self.assertGreater(np.max(np.abs(q)), .01)
        self.assertLessEqual(np.max(np.abs(q)), .08)

    def test_push_preserves_entire_stroke_and_return(self):
        q, dq = correct_trajectory(self.q, self.dq, self.times, np.ones(6), .78)
        grid, actual = stream_samples(q, dq, self.times)
        _, nominal = stream_samples(self.q, self.dq, self.times)
        np.testing.assert_array_equal(actual[grid >= .8], nominal[grid >= .8])
        np.testing.assert_array_equal(q[:, self.times >= .8], self.q[:, self.times >= .8])
        self.assertGreater(np.max(np.abs(q)), .01)

    def _adapter(self, action):
        calls = []
        core = SimpleNamespace(sim_time=lambda: 3.)
        backend = SimpleNamespace(_stream_trajectory=lambda *args, **kwargs:
            calls.append((args, kwargs, core.safe_residual_context)))
        runner = SimpleNamespace(core=core, traj_ctrl=backend,
            _active_target=SimpleNamespace(sim_object_id=12),
            _skill_name_for_object=lambda target: "throw", _status="THROWING",
            robot=SimpleNamespace(joint_limits=[(-10., 10.)] * 6), M1=np.full(6, 20.))
        adapter = SafeResidualAdapter(decision_callback=lambda *args:
            (action, np.zeros(6), 0., np.zeros(2), np.zeros(100))).bind(runner)
        adapter.observation = lambda *args: (np.zeros(97), [])
        return adapter, runner, calls

    def test_negative_release_index_keeps_backend_semantics(self):
        adapter, runner, calls = self._adapter(np.full(6, .5))
        runner.traj_ctrl._stream_trajectory(self.q, self.dq, self.times,
                                           release_index=-21)
        record = adapter.records[0]
        self.assertEqual(record["release_time"], self.times[-21])
        self.assertEqual(calls[0][1]["release_index"], -21)
        self.assertEqual(len(record["normalized"]), 100)
        self.assertTrue(record["protected_position_equal"])
        self.assertTrue(record["protected_velocity_equal"])
        self.assertIsNone(runner.core.safe_residual_context)

    def test_bound_zero_action_is_exact_stream_identity(self):
        adapter, runner, calls = self._adapter(np.zeros(6))
        runner.traj_ctrl._stream_trajectory(self.q, self.dq, self.times,
                                           release_index=20)
        self.assertIs(calls[0][0][0], self.q)
        self.assertIs(calls[0][0][1], self.dq)
        self.assertEqual(calls[0][2]["release_time"], 1.)
        self.assertTrue(adapter.records[0]["exact_identity"])

    def test_strict_sample_velocity_guard_projects_action(self):
        adapter, runner, _ = self._adapter(np.ones(6))
        runner.M1[:] = .01
        adapter.transform(self.q, self.dq, self.times, release_index=20)
        record = adapter.records[0]
        self.assertGreater(record["projection"], 0.)
        self.assertLess(record["projection"], 1.)
        self.assertLess(record["max_joint_delta"], record["proposed_max_joint_delta"])

    def test_native_immutable_joint_overspeed_does_not_block_legal_residual(self):
        adapter, runner, _ = self._adapter(np.full(6, .5))
        runner.M1[:] = .6
        self.q[4] = 2. * self.times
        self.dq[4] = 2.
        q, dq, record = adapter.transform(self.q, self.dq, self.times, release_index=20)
        self.assertGreater(record["baseline_existing_velocity_ratio"], 3.)
        self.assertGreater(record["max_joint_delta"], 0.)
        self.assertLessEqual(record["candidate_changed_max_ratio"], 1. + 1e-9)
        np.testing.assert_array_equal(q[4], self.q[4])
        np.testing.assert_array_equal(dq[4], self.dq[4])

    def test_no_window_push_exposes_protected_stroke_context_without_actor(self):
        adapter, runner, calls = self._adapter(np.zeros(6))
        runner._skill_name_for_object = lambda target: "push"
        runner._status = "PUSHING"
        runner.skills = {"push": SimpleNamespace(_last_push_meta={"n_descent": 2, "n_stroke": 39})}
        runner.robot.forward_kinematics = lambda q: np.eye(4)
        runner.traj_ctrl._stream_trajectory(self.q, self.dq, self.times)
        self.assertEqual(len(adapter.records), 0)
        context = calls[0][2]
        self.assertEqual(context["prepare_end"], .05)
        self.assertEqual(context["stroke_end"], 2.)
        self.assertFalse(context["record"]["active"])

    def test_composed_push_edits_both_routes_and_preserves_whole_stroke(self):
        q, dq = correct_trajectory(self.q, self.dq, self.times, np.ones(6), .78,
                                   transfer_start=1.24)
        grid, actual = stream_samples(q, dq, self.times)
        _, nominal = stream_samples(self.q, self.dq, self.times)
        protected = (grid >= .8 - 1e-12) & (grid <= 1.2 + 1e-12)
        np.testing.assert_array_equal(actual[protected], nominal[protected])
        knots = (self.times >= .8) & (self.times <= 1.2 + 1e-12)
        np.testing.assert_array_equal(q[:, knots], self.q[:, knots])
        np.testing.assert_array_equal(dq[:, knots], self.dq[:, knots])
        self.assertGreater(np.max(np.abs(q[:, self.times < .8])), .01)
        self.assertGreater(np.max(np.abs(q[:, self.times > 1.24])), .01)

    def test_short_prepare_with_transfer_window_is_push_route_decision(self):
        adapter, runner, calls = self._adapter(np.full(6, .5))
        runner._skill_name_for_object = lambda target: "push"
        runner._status = "PUSHING"
        runner.skills = {"push": SimpleNamespace(_last_push_meta={"n_descent": 2, "n_stroke": 10})}
        runner.robot.forward_kinematics = lambda q: np.eye(4)
        runner.traj_ctrl._stream_trajectory(self.q, self.dq, self.times)
        self.assertEqual(len(adapter.records), 1)
        record = adapter.records[0]
        self.assertEqual(record["phase"], "push_route")
        self.assertFalse(record["prepare_window_active"])
        self.assertTrue(record["transfer_window_active"])
        self.assertGreater(record["max_joint_delta"], 0.)
        q, dq = calls[0][0][:2]
        np.testing.assert_array_equal(q[:, :12], self.q[:, :12])
        np.testing.assert_array_equal(dq[:, :12], self.dq[:, :12])
        self.assertEqual(calls[0][2]["prepare_end"], .05)
        self.assertEqual(calls[0][2]["stroke_end"], .55)


if __name__ == "__main__":
    unittest.main()

"""Paired uncertainty, censoring, and workload-closure checks."""
import copy
import unittest

import numpy as np

from gp8_control.tools.report_safe_simulator import paired_statistics, values_from_result
from gp8_control.tools.audit_safe_simulator import verify_rollout, json_hash


class SafeReportingTest(unittest.TestCase):
    def test_model_initializations_are_averaged_inside_scene_cluster(self):
        result = paired_statistics([1., 10., 100.], [[2., 11., 101.], [0., 9., 99.]])
        self.assertEqual(result['difference'], 0.)
        self.assertEqual(result['difference_ci95'], [0., 0.])
        self.assertEqual(result['learned_model_means'], [38., 36.])

    def test_paired_scene_noise_cancels(self):
        result = paired_statistics([1., 10., 100.], [[3., 12., 102.], [3., 12., 102.]])
        self.assertEqual(result['difference'], 2.)
        self.assertEqual(result['difference_ci95'], [2., 2.])

    def test_unreached_target_uses_full_budget_score(self):
        result = dict(summary=dict(target_success_time=None, contacts=3, offered=20))
        metrics = values_from_result(result, dict(time_limit=120.))
        self.assertEqual(metrics['target_capped_seconds'], 120.)
        self.assertEqual(metrics['target_reach_percent'], 0.)
        self.assertEqual(metrics['contacts_per_20_offered'], 3.)

    def test_closure_rejects_silently_dropped_offer(self):
        tape = [dict(id=1,time=.5), dict(id=2,time=1.)]
        summary = dict(offered=2, tape_sha256=json_hash(tape), admitted=2, not_admitted=0,
            physical_resolved=0, physical_unresolved=2, unresolved=2, correct_bin=0,
            task_success_percent=0., actions=0, contacts=0, collateral_falls=0,
            impulse_ns=0., sim_seconds=120., terminated=True, truncated=False,
            termination_reason='time_limit', target_success_time=None, policy_decisions=0)
        trace = dict(tape=tape, admissions=[dict(**r,actual_time=r['time']) for r in tape],
            outcomes=[], physical_resolutions=[], executions=[], contacts=[], falls=[], residuals=[])
        result = dict(summary=summary, trace=trace, decisions=[])
        config = dict(offer_count=2, time_limit=120., success_target_auxiliary=10, residual_scale=.08)
        self.assertEqual(verify_rollout(result, config)['offered'], 2)
        corrupted = copy.deepcopy(result)
        corrupted['trace']['admissions'].pop(0)
        with self.assertRaisesRegex(AssertionError, 'FIFO'):
            verify_rollout(corrupted, config)


if __name__ == '__main__':
    unittest.main()

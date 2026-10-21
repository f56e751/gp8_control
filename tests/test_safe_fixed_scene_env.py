"""Fixed scenes retain native selection, observation horizon and zero identity."""
import unittest

import numpy as np

from gp8_control.rl.safe_fixed_scene_env import collect_episode, digest, make_scene, _overlap, _on_belt


class ConstantPolicy:
    def act(self, observation, head, deterministic=False):
        assert np.asarray(observation).shape==(107,)
        action=np.full(6,.35,np.float32)
        return action,np.arctanh(action),0.,np.zeros(2),np.asarray(observation).copy()


class FixedSceneTest(unittest.TestCase):
    def test_seeded_layouts_only_reject_geometric_overlap(self):
        for skill in ('throw','push'):
            for neighbors in (2,5):
                for seed in range(50):
                    scene=make_scene(seed,skill,neighbors)
                    self.assertEqual(digest(scene),digest(make_scene(seed,skill,neighbors)))
                    self.assertEqual(len(scene['rows']),neighbors+1)
                    target=scene['rows'][0]
                    self.assertEqual(target['id'],1)
                    self.assertTrue(all(r['y']>target['y'] for r in scene['rows'][1:]))
                    self.assertTrue(any(r['x']<target['x'] for r in scene['rows'][1:]))
                    self.assertTrue(any(r['x']>target['x'] for r in scene['rows'][1:]))
                    for index,row in enumerate(scene['rows']):
                        self.assertTrue(_on_belt(row))
                        self.assertTrue(all(not _overlap(row,other) for other in scene['rows'][:index]))
        self.assertNotEqual(digest(make_scene(1,'throw',2)),digest(make_scene(2,'throw',2)))

    def test_native_zero_identity_and_common_horizon_for_both_skills(self):
        for skill,neighbors in (('throw',2),('push',5)):
            baseline=collect_episode(4600000,skill,neighbors,None,original_baseline=True)
            zero=collect_episode(4600000,skill,neighbors,None)
            for key in ('initial_state_sha256','selection_sha256'):
                self.assertEqual(baseline[key],zero[key])
            for key in ('reset_state_sha256','command_sha256','command_timeline_sha256',
                        'io_sha256','final_state_sha256'):
                self.assertEqual(baseline['trace'][key],zero['trace'][key],key)
            for key in ('contacts','falls','outcomes','physical_resolutions','executions','end_objects'):
                self.assertEqual(baseline['trace'][key],zero['trace'][key],key)
            for result in (baseline,zero):
                s=result['summary']
                self.assertTrue(s['intended_target_selected'])
                self.assertEqual(s['actions'],1)
                self.assertEqual(s['additional_arrivals'],0)
                self.assertFalse(s['initially_grasped'])
                self.assertTrue(s['action_horizon_completed'])
                self.assertAlmostEqual(s['observed_action_seconds'],15.,places=6)
                self.assertGreater(s['prelude_seconds'],10.)
                self.assertTrue(s['skill_completed'])
                self.assertEqual(s['action_start_neighbor_count'],neighbors)
                self.assertEqual(s['action_start_neighbors_on_belt'],neighbors)
            if skill=='throw':
                self.assertGreater(zero['summary']['throw_loaded_decisions'],0)
                self.assertEqual(zero['summary']['throw_loaded_decisions'],zero['summary']['throw_loaded_decisions_holding'])

    def test_push_native_selection_regression_from_development_pilot(self):
        # The v1 pilot selected neighbors because target x<.44 failed the
        # native .24m backswing gate. New geometry retains the native selector.
        for seed,neighbors in ((5902000,2),(5903000,5)):
            result=collect_episode(seed,'push',neighbors,None)
            summary=result['summary']
            self.assertEqual(summary['selected_target_id'],1)
            self.assertEqual(summary['selected_skill'],'push')
            self.assertTrue(summary['intended_target_selected'])
            self.assertEqual(summary['action_start_neighbors_on_belt'],neighbors)
            self.assertEqual(summary['actions'],1)

    def test_nonzero_policy_reaches_native_guarded_decisions(self):
        result=collect_episode(4600000,'push',2,ConstantPolicy())
        self.assertTrue(result['summary']['action_horizon_completed'])
        self.assertGreater(result['summary']['nonzero_decisions'],0)
        self.assertTrue(any(r['phase']=='push_approach' for r in result['trace']['residuals']))
        for row in result['trace']['residuals']:
            self.assertLessEqual(max(row['candidate_changed_max_ratio_per_joint']),1.+1e-7)
            self.assertLessEqual(max(row['acceleration_limit_ratio_per_joint']),1.+1e-7)
        self.assertTrue(all(np.asarray(row['raw_obs']).shape==(107,) for row in result['rows']))

    def test_partial_skill_at_horizon_is_not_a_task_success(self):
        result=collect_episode(4600000,'throw',2,None,config={'action_horizon':.05})
        self.assertFalse(result['summary']['skill_completed'])
        self.assertFalse(result['summary']['target_success'])
        self.assertEqual(result['summary']['actions'],1)
        self.assertTrue(result['summary']['action_horizon_completed'])


if __name__=='__main__':
    unittest.main()

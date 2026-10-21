"""Fixed-scene evaluation denominators, paired uncertainty and preservation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from gp8_control.tools import evaluate_safe_fixed_scene as evaluator


def summary(success=True, contacts=0, **updates):
    return dict(target_success=success, action_contacts=contacts, action_object_contacts=contacts,
        action_impulse_ns=.25*contacts, action_collateral_falls=0,
        preparation_contacts=0, attempted_skill=True,
        target_outcome={'actual_bin_name':'correct'} if success else None,
        termination_reason='fixed_action_horizon', **updates)


class FixedEvaluationTest(unittest.TestCase):
    def test_all_four_strata_are_preregistered_and_seed_disjoint(self):
        scenes = evaluator.enumerate_scenes(32)
        self.assertEqual(len(scenes), 128)
        self.assertEqual(len({s['scene_seed'] for s in scenes}), 128)
        for skill in ('throw','push'):
            for count in (2,5):
                self.assertEqual(sum(s['target_skill']==skill and s['neighbor_count']==count for s in scenes),32)
        self.assertTrue(set(s['scene_seed'] for s in scenes).isdisjoint(
            s['scene_seed'] for s in evaluator.enumerate_scenes(1,pilot=True)))
        with self.assertRaisesRegex(RuntimeError,'at least32'):
            evaluator.enumerate_scenes(31)

    def test_failed_selection_and_unresolved_target_remain_in_denominator(self):
        scenes = evaluator.enumerate_scenes(1,pilot=True)[:2]
        raw = {}
        for index, scene in enumerate(scenes):
            episode = summary(success=index==0)
            episode['attempted_skill'] = index==0
            raw[scene['scene_id']] = {arm:{'summary':episode} for arm in evaluator.ARMS}
        group = evaluator.paired_group(scenes,raw,resamples=100)
        self.assertEqual(group['n_scenes'],2)
        self.assertEqual(group['baseline']['target_success_percent'],50.)
        self.assertEqual(group['baseline']['attempted_percent'],50.)
        self.assertEqual(group['target_outcomes']['baseline'],{'correct_bin':1,'unresolved':1})

    def test_two_models_are_averaged_inside_each_scene_before_bootstrap(self):
        scenes = evaluator.enumerate_scenes(1,pilot=True)[:2]
        raw = {scene['scene_id']:{arm:{'summary':summary(success=arm!='safe_seed0',contacts=0 if arm=='safe_seed0' else 2)}
                                for arm in evaluator.ARMS} for scene in scenes}
        group = evaluator.paired_group(scenes,raw,resamples=100)
        self.assertEqual(group['n_scenes'],2)
        self.assertEqual(group['learned']['target_success_percent'],50.)
        self.assertEqual(group['per_policy']['safe_seed0']['target_success_percent'],0.)
        self.assertEqual(group['per_policy']['safe_seed1']['target_success_percent'],100.)
        self.assertEqual(group['paired_differences']['target_success_percent'],{'mean':-50.,'ci95':[-50.,-50.]})
        self.assertEqual(group['paired_differences']['contacts_per100']['mean'],-100.)

    def test_action_window_is_primary_and_collision_free_success_needs_both(self):
        episode = summary(contacts=0)
        episode['contacts'] = 50  # policy-free prelude must not contaminate primary risk
        metrics = evaluator.episode_metrics(episode)
        self.assertEqual(metrics['contacts_per100'],0.)
        self.assertEqual(metrics['collision_free_success_percent'],100.)
        self.assertEqual(evaluator.episode_metrics(summary(contacts=1))['collision_free_success_percent'],0.)
        self.assertEqual(evaluator.episode_metrics(summary(success=False))['collision_free_success_percent'],0.)
        crossing=summary(contacts=0)
        crossing['action_impulse_ns']=.01
        self.assertEqual(evaluator.episode_metrics(crossing)['collision_free_success_percent'],0.)
        with self.assertRaisesRegex(RuntimeError,'Missing fixed-scene'):
            evaluator.episode_metrics({'target_success':True})

    def test_preservation_detects_modified_original_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            original=root/'runs/original.txt'
            original.parent.mkdir()
            original.write_text('original bytes')
            out=root/'runs/new';out.mkdir()
            evaluator.save_json(out/'preservation.json',{'existing_files_sha256':{'runs/original.txt':evaluator.sha(original)}})
            protocol={'source_sha256':{'runtime.py':'fixed'},'checkpoint_sha256':{},
                'preservation_manifest_sha256':evaluator.sha(out/'preservation.json')}
            with patch.object(evaluator,'ROOT',root),patch.object(evaluator,'fixed_sources',return_value={'runtime.py':'fixed'}):
                self.assertTrue(evaluator.verify_freeze(out,protocol)['all_original_bytes_unchanged'])
                original.write_text('changed')
                with self.assertRaisesRegex(RuntimeError,'artifact changed'):
                    evaluator.verify_freeze(out,protocol)

    def test_no_partial_extra_or_foreign_results_are_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);(out/'evaluation').mkdir()
            scenes=evaluator.enumerate_scenes(1,pilot=True)[:1]
            evaluator.save_json(out/'protocol.json',{'scenes':scenes})
            with self.assertRaisesRegex(RuntimeError,'Incomplete or additional'):
                evaluator.read_results(out,{'scenes':scenes})
            for arm in evaluator.ARMS:
                evaluator.save_json(out/'evaluation'/f"{scenes[0]['scene_id']}_{arm}.json",dict(
                    **scenes[0],policy=arm,evaluation_protocol_sha256=evaluator.sha(out/'protocol.json')))
            self.assertEqual(len(evaluator.read_results(out,{'scenes':scenes})),1)
            evaluator.save_json(out/'evaluation'/'unexpected.json',{})
            with self.assertRaisesRegex(RuntimeError,'Incomplete or additional'):
                evaluator.read_results(out,{'scenes':scenes})


if __name__=='__main__':
    unittest.main()

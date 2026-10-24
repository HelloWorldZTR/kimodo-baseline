"""Regression tests for relative-command anchors, without model inference."""

import copy
from pathlib import Path
import unittest

import numpy as np

from kimodo_benchmark.anchored_conditions import scoring_window
from kimodo_benchmark.conditions import condition_spec, expected_mask
from kimodo_benchmark.protocol import (ANCHORED_ADAPTER, KEYFRAMES, ROOT, f_to_k,
                                      load_protocol, read_json, samples)


class AnchoredConditionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol(ROOT/'benchmark_sources/command9_main_20261024/protocol.json')
        cls.geometry = read_json(ROOT/'tests/fixtures/soma_geometry.json')

    def test_unchanged_tasks_have_identical_conditions(self):
        for task in self.protocol['tasks']:
            if task['name'] in ('strike','kick','jump','squat'):
                continue
            for command, frames in zip(task['commands'],task['lengths']):
                before=condition_spec(task,command,frames,self.geometry)
                after=condition_spec(task,command,frames,self.geometry,ANCHORED_ADAPTER)
                self.assertEqual(before,after)

    def test_every_command_uses_same_joint_origin_and_unclipped_delta(self):
        for task in self.protocol['tasks']:
            name=task['name']
            if name not in ('strike','kick','jump','squat'):
                continue
            for command,frames in zip(task['commands'],task['lengths']):
                spec=condition_spec(task,command,frames,self.geometry,ANCHORED_ADAPTER)
                start,target,end=spec['constraints']
                mask=expected_mask(spec,self.geometry)
                self.assertEqual(np.flatnonzero(mask.any(1)).tolist(),spec['frame_indices'])
                self.assertEqual(int(mask.sum()),len(spec['frame_indices'])*(14 if name in ('jump','squat') else 17))
                np.testing.assert_array_equal(start['primary_target_k'],end['primary_target_k'])
                difference=np.asarray(target['primary_target_k'])-start['primary_target_k']
                component=2 if name in ('strike','kick') else 1
                self.assertAlmostEqual(difference[component],float(np.float32(command[0]))*(-1 if name=='squat' else 1),places=7)
                self.assertIn(0,start['frame_indices'])
                self.assertIn(KEYFRAMES[name],target['frame_indices'])
                if name=='kick':
                    self.assertEqual(target['primary_joint'],'RightFoot')
                    joints=np.asarray(target['global_joints_positions'])[0]
                    self.assertAlmostEqual(joints[self.geometry['names30'].index('RightToeBase'),1],1.,places=6)
                if name=='strike':
                    np.testing.assert_allclose(start['primary_target_k'],f_to_k(task['reference']['positions_f'][21]))
                    self.assertIn(15,start['frame_indices'])
                    self.assertIn(45,end['frame_indices'])
                if name=='squat':
                    self.assertEqual(start['frame_indices'],list(range(7)))
                    self.assertEqual(target['frame_indices'],list(range(58,66)))

    def test_squat_windows_match_five_score_frames(self):
        task=next(task for task in self.protocol['tasks'] if task['name']=='squat')
        for command in task['commands']:
            spec=condition_spec(task,command,82,self.geometry,ANCHORED_ADAPTER)
            height=np.full(spec['native_frames'],np.nan)
            for part in spec['constraints']:
                height[part['frame_indices']]=part['primary_target_k'][1]
            points=np.arange(82)*1.5
            lo=np.floor(points).astype(int); hi=np.ceil(points).astype(int)
            # Check only windows fully covered by the requested anchor plateaux.
            sampled=np.where(lo==hi,height[lo],(height[lo]+height[hi])/2)
            self.assertTrue(np.isfinite(sampled[:5]).all())
            self.assertTrue(np.isfinite(sampled[39:44]).all())
            self.assertTrue(np.isfinite(sampled[-5:]).all())
            self.assertAlmostEqual(float(sampled[:5].mean()-sampled[39:44].mean()),float(command[0]),places=6)
            np.testing.assert_allclose(sampled[:5],sampled[-5:])

    def test_overlapping_writes_are_rejected(self):
        task=next(task for task in self.protocol['tasks'] if task['name']=='jump')
        spec=condition_spec(task,task['commands'][0],66,self.geometry,ANCHORED_ADAPTER)
        spec['constraints'].append(copy.deepcopy(spec['constraints'][0]))
        with self.assertRaisesRegex(ValueError,'Overlapping'):
            expected_mask(spec,self.geometry)


if __name__=='__main__':
    unittest.main()

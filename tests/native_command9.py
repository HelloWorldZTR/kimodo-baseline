"""Native constraint integration checks, CPU only; never load a model.

Run separately in the Kimodo environment:
CUDA_VISIBLE_DEVICES='' python tests/native_command9.py
"""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise RuntimeError('Set CUDA_VISIBLE_DEVICES to an empty string for these CPU tests')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

torch.set_num_threads(1)
from kimodo_benchmark.conditions import condition_spec, describe_skeleton, expected_mask, native_constraints
from kimodo_benchmark.generate import CachedEncoder, NativeEngine
from kimodo_benchmark.protocol import ANCHORED_ADAPTER, KEYFRAMES, load_protocol, read_json, samples, task_by_name


class NativeConditionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard = patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('CUDA is forbidden'))
        cls.guard.start()
        from kimodo.skeleton import SOMASkeleton30
        from kimodo.motion_rep.reps.kimodo_motionrep import KimodoMotionRep
        cls.skeleton = SOMASkeleton30().cpu()
        cls.geometry = describe_skeleton(cls.skeleton, cls.skeleton.somaskel77)
        cls.rep = KimodoMotionRep(cls.skeleton, fps=30).cpu()
        cls.protocol = load_protocol(Path(os.environ.get('KIMODO_BENCHMARK_BUNDLE', ROOT / 'benchmark_sources/command9')) / 'protocol.json')
        from kimodo.motion_rep.stats import Stats
        cls.rep.stats = Stats()
        # Synthetic nontrivial stats expose normalization/range errors without
        # loading any checkpoint. Actual asset readiness is a separate file check.
        cls.rep.stats.register_from_tensors(torch.linspace(-.4, .4, 369), torch.linspace(.5, 1.5, 369))

    @classmethod
    def tearDownClass(cls):
        cls.guard.stop()

    def test_native_geometry_matches_cpu_fixture(self):
        self.assertEqual(self.geometry, read_json(ROOT / 'tests/fixtures/soma_geometry.json'))

    def test_anchored_conditions_match_native_masks_and_targets(self):
        for task in self.protocol['tasks']:
            if task['name'] not in ('strike', 'kick', 'jump', 'squat'):
                continue
            for command, frames in zip(task['commands'], task['lengths']):
                spec = condition_spec(task, command, frames, self.geometry, ANCHORED_ADAPTER)
                constraints = native_constraints(spec, self.skeleton, 'cpu')
                observed, mask = self.rep.create_conditions_from_constraints(constraints, spec['native_frames'], False, 'cpu')
                np.testing.assert_array_equal(mask.numpy(), expected_mask(spec, self.geometry))
                for part in spec['constraints']:
                    indices = part['frame_indices']
                    primary_index = self.geometry['names30'].index(part['primary_joint'])
                    np.testing.assert_allclose(observed[indices, 5+3*primary_index:8+3*primary_index],
                                               np.broadcast_to(part['primary_target_k'], (len(indices), 3)), atol=1e-6)
                    np.testing.assert_allclose(observed[indices, 1], np.asarray(part['global_joints_positions'])[:, 0, 1], atol=1e-6)
                normalized, normalized_mask = self.rep.create_conditions_from_constraints_batched(
                    constraints, torch.tensor([spec['native_frames']]), True, 'cpu')
                np.testing.assert_array_equal(normalized[0], self.rep.normalize(observed))
                np.testing.assert_array_equal(normalized_mask[0], mask)
        self.assertFalse(torch.cuda.is_initialized())

    def test_all_command_conditions_against_native_api(self):
        from kimodo.geometry import matrix_to_cont6d
        tested = 0
        for task in self.protocol['tasks']:
            for command, frames in zip(task['commands'], task['lengths']):
                spec = condition_spec(task, command, frames, self.geometry)
                constraints = native_constraints(spec, self.skeleton, 'cpu')
                observed, mask = self.rep.create_conditions_from_constraints(constraints, spec['native_frames'], False, 'cpu')
                np.testing.assert_array_equal(mask.numpy(), expected_mask(spec, self.geometry))
                self.assertTrue(torch.isfinite(observed).all())
                indices = spec['frame_indices']
                np.testing.assert_allclose(observed[indices][:, [0, 2]].numpy(), spec['smooth_root_2d'], atol=1e-6)
                if task['name'] in KEYFRAMES:
                    frame = indices[0]
                    positions = np.asarray(spec['global_joints_positions'])[0]
                    np.testing.assert_allclose(observed[frame, 1], positions[0, 1], atol=1e-6)
                    np.testing.assert_array_equal(observed[frame, 3:5], [1, 0])
                    anchor = spec['joint_names'][0]
                    selected = {'Hips': ['Hips'], 'RightHand': ['RightHand', 'RightHandMiddleEnd'],
                                'RightFoot': ['RightFoot', 'RightToeBase']}[anchor]
                    for name in selected:
                        j = self.geometry['names30'].index(name)
                        np.testing.assert_allclose(observed[frame, 5 + 3*j:8 + 3*j], positions[j], atol=1e-6)
                    j = self.geometry['names30'].index(anchor)
                    rotation = torch.tensor(spec['global_joints_rots'][0][j], dtype=torch.float32)
                    np.testing.assert_allclose(observed[frame, 95 + 6*j:101 + 6*j], matrix_to_cont6d(rotation), atol=1e-6)
                if tested % 100 == 0 or task['name'] != 'reach':
                    normalized, mask2 = self.rep.create_conditions_from_constraints(constraints, spec['native_frames'], True, 'cpu')
                    np.testing.assert_array_equal(mask2, mask)
                    # Native Stats uses max(std, eps); our positive std is far from eps.
                    expected = self.rep.normalize(observed)
                    np.testing.assert_array_equal(normalized, expected)
                    batched, mask3 = self.rep.create_conditions_from_constraints_batched(constraints, torch.tensor([spec['native_frames']]), True, 'cpu')
                    np.testing.assert_array_equal(batched[0], normalized)
                    np.testing.assert_array_equal(mask3[0], mask)
                tested += 1
        self.assertEqual(tested, sum(len(task['commands']) for task in self.protocol['tasks']))
        self.assertFalse(torch.cuda.is_initialized())

    def test_engine_wiring_seeds_and_cache_without_model(self):
        from kimodo.sanitize import sanitize_text
        import random
        calls, noises = [], []
        skeleton, rep = self.skeleton, self.rep
        def encoder(texts):
            calls.extend(texts)
            # Even an encoder that consumes RNG must not alter diffusion noise.
            torch.randn(17)
            return torch.ones((1, 1, 4096)), [1]
        class DummyModel:
            fps = 30
            motion_rep = rep
            text_encoder = CachedEncoder(encoder)
            def __call__(model, prompt, n, **kwargs):
                self.assertEqual(kwargs['num_denoising_steps'], 100)
                self.assertEqual(kwargs['cfg_type'], 'separated')
                self.assertEqual(kwargs['cfg_weight'], [2., 2.])
                self.assertFalse(kwargs['post_processing'])
                self.assertFalse(kwargs['multi_prompt'])
                self.assertEqual(kwargs['first_heading_angle'], 0.)
                self.assertEqual(len(kwargs['constraint_lst']), 1)
                model.text_encoder([sanitize_text(prompt)])
                noises.append((torch.randn(n, 369), np.random.randn(3), random.random()))
                return dict(posed_joints=np.zeros((n, 77, 3), np.float32),
                            local_rot_mats=np.broadcast_to(np.eye(3), (n, 77, 3, 3)),
                            root_positions=np.zeros((n, 3)), smooth_root_pos=np.zeros((n, 3)))
        dummy = DummyModel()
        dummy.skeleton = skeleton
        engine = NativeEngine.__new__(NativeEngine)
        engine.model, engine.geometry, engine.device, engine.sanitize = dummy, self.geometry, 'cpu', sanitize_text
        records = list(samples(self.protocol, ['strike'], maximum_commands=3))[:3]
        for record in records + records[:1]:
            task = task_by_name(self.protocol, record['task'])
            spec = condition_spec(task, record['command'], record['frames'], self.geometry)
            arrays = engine.infer(record, spec)
            self.assertEqual(str(arrays['actual_prompt']), sanitize_text(record['prompt']))
            self.assertTrue(np.isfinite(arrays['observed_motion']).all())
        self.assertEqual(len(calls), 1)
        for noise in noises[1:]:
            for a, b in zip(noises[0], noise):
                np.testing.assert_array_equal(a, b)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main(verbosity=2)

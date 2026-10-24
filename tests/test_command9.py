"""CPU-only adapter tests. Synthetic motion only: no model/checkpoint loading.

Requires a previously exported bundle (default benchmark_sources/command9).
Run: CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p 'test_command9.py' -v
"""

import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from kimodo_benchmark.conditions import condition_spec, expected_mask, shortest_rotation
from kimodo_benchmark.export import build_protocol
from kimodo_benchmark.generate import (CachedEncoder, generate_samples, output_lock,
                                       prepare_run)
from kimodo_benchmark.protocol import (JOINTS_22, KEYFRAMES, LENGTHS, ROOT, SEEDS,
                                      SETTINGS, TASKS, archive_path, canonicalize,
                                      f_to_k, k_to_f,
                                      load_protocol, native_frames, read_json,
                                      read_sample, samples,
                                      scoring_positions, task_by_name, write_npz)
from kimodo_benchmark.score import (connect_scoring_source, measure_sample, save_reports,
                                   score_run, skating_ratio, summarize)

BUNDLE = Path(os.environ.get('KIMODO_BENCHMARK_BUNDLE', ROOT / 'benchmark_sources/command9'))
GEOMETRY = read_json(ROOT / 'tests/fixtures/soma_geometry.json')


class SyntheticEngine:
    """Deterministic static fixtures exercise I/O, not Kimodo generation quality."""
    geometry = GEOMETRY

    def __init__(self, protocol):
        self.protocol, self.calls = protocol, 0

    def infer(self, record, spec):
        self.calls += 1
        pose = np.asarray(task_by_name(self.protocol, record['task'])['reference']['positions_f'])
        n = spec['native_frames']
        native = np.zeros((n, 77, 3), np.float32)
        for j, name in enumerate(JOINTS_22):
            native[:, self.geometry['names77'].index(name)] = f_to_k(pose[j])
        return dict(native_positions_k=native,
                    local_rot_mats=np.broadcast_to(np.eye(3, dtype=np.float32), (n, 77, 3, 3)).copy(),
                    root_positions=np.repeat(f_to_k(pose[:1]).astype(np.float32), n, axis=0),
                    smooth_root_pos=np.repeat(f_to_k(pose[:1]).astype(np.float32), n, axis=0),
                    observed_motion=np.zeros((n, 369), np.float32),
                    motion_mask=expected_mask(spec, self.geometry), actual_prompt=np.asarray(record['prompt']))


class Command9Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.guard = patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('GPU use forbidden in CPU tests'))
        cls.guard.start()
        cls.protocol = load_protocol(BUNDLE / 'protocol.json')
        connect_scoring_source(BUNDLE / 'scoring_source', cls.protocol)

    @classmethod
    def tearDownClass(cls):
        cls.guard.stop()
        # Separate frozen releases may be tested in the same discovery process.
        for name in tuple(sys.modules):
            if name.split('.')[0] in ('shared_motion', 'data_processing', 'src'):
                del sys.modules[name]
        sys.path[:] = [entry for entry in sys.path if Path(entry).resolve() != (BUNDLE / 'scoring_source').resolve()]

    def task(self, name):
        return task_by_name(self.protocol, name)

    def pose(self, name):
        task = self.task(name)
        return np.repeat(np.asarray(task['reference']['positions_f'], np.float32)[None], LENGTHS[name], axis=0)

    def test_frozen_composite_panel(self):
        records = list(samples(self.protocol))
        self.assertEqual(len(records), 23400)
        self.assertEqual(len({r['sample_id'] for r in records}), 23400)
        self.assertEqual(len(list(samples(self.protocol, maximum_commands=3))), 243)
        base, limb, ev = (read_json(BUNDLE / n) for n in ('base_protocol.json', 'limb_protocol.json', 'limb_evaluation.json'))
        rebuilt = build_protocol(base, limb, ev)
        self.assertEqual(rebuilt['tasks'], self.protocol['tasks'])
        # Preserve the original reach workspace exactly, despite later upstream changes.
        self.assertEqual(self.task('reach'), next(t for t in base['tasks'] if t['name'] == 'reach'))
        self.assertEqual(self.task('kick')['definition'], ev['definitions']['kick'])
        broken = copy.deepcopy(base)
        broken['tasks'][0]['commands'][0][0] += .01
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            build_protocol(broken, limb, ev)

    def test_native_duration_and_interpolation(self):
        self.assertEqual([native_frames(LENGTHS[t]) for t in TASKS], [90, 90, 45, 75, 73, 99, 123, 60, 90])
        for frames in LENGTHS.values():
            n = native_frames(frames)
            # A constant-velocity native trajectory must retain exact 20 Hz times.
            native = np.zeros((n, 77, 3), np.float32)
            native[:, :, 2] = np.arange(n)[:, None]
            scored = scoring_positions(native, GEOMETRY['names77'], frames)
            np.testing.assert_array_equal(scored[:, 0, 0], np.arange(frames) * 1.5)
            self.assertLessEqual((frames - 1) / 20, (n - 1) / 30)
            self.assertLess((n - 1) / 30 - (frames - 1) / 20, 1 / 30)
        np.testing.assert_array_equal(k_to_f(f_to_k(np.arange(6).reshape(2, 3))), np.arange(6).reshape(2, 3))

    def test_anatomical_joint_mapping(self):
        native = np.broadcast_to(np.arange(77)[None, :, None], (90, 77, 3))
        output = scoring_positions(native, GEOMETRY['names77'], 60)
        np.testing.assert_array_equal(output[0, :, 0], [GEOMETRY['names77'].index(n) for n in JOINTS_22])
        self.assertEqual(JOINTS_22[8], 'RightFoot')
        self.assertEqual(JOINTS_22[11], 'RightToeBase')
        self.assertEqual(JOINTS_22[17], 'RightArm')

    def test_all_commands_have_only_agreed_mask(self):
        for task in self.protocol['tasks']:
            for command, frames in zip(task['commands'], task['lengths']):
                spec = condition_spec(task, command, frames, GEOMETRY)
                mask = expected_mask(spec, GEOMETRY)
                if task['name'] in KEYFRAMES:
                    self.assertEqual(np.flatnonzero(mask.any(1)).tolist(), [KEYFRAMES[task['name']]])
                    self.assertEqual(mask.sum(), 14 if task['name'] in ('jump', 'squat') else 17)
                    self.assertFalse(mask[:, 275:].any())  # no velocities or contacts
                else:
                    self.assertEqual(np.flatnonzero(mask.any(0)).tolist(), [0, 2])
                    self.assertTrue(mask[:, [0, 2]].all())

    def test_event_targets_and_fixed_distal_offsets(self):
        names = GEOMETRY['names30']
        neutral = np.asarray(GEOMETRY['neutral30'])
        for name in KEYFRAMES:
            task = self.task(name)
            pose = np.asarray(task['reference']['positions_f'])
            command = np.asarray(task['commands'][-1], np.float32)
            spec = condition_spec(task, command, LENGTHS[name], GEOMETRY)
            if name == 'strike':
                target = pose[17] + [command[0], 0, 0]
            elif name == 'kick':
                target = np.array([pose[11, 0] + command[0], pose[11, 1], 1.0])
            elif name == 'reach':
                target = command + [0, 0, pose[0, 2]]
            else:
                target = np.array([0, 0, pose[0, 2] + command[0] * (1 if name == 'jump' else -1)])
            np.testing.assert_allclose(k_to_f(spec['primary_target_k']), target, atol=1e-7)
            np.testing.assert_array_equal(spec['global_root_heading'], [[1, 0]])
            if name in ('kick', 'strike', 'reach'):
                a, d = ('RightFoot', 'RightToeBase') if name == 'kick' else ('RightHand', 'RightHandMiddleEnd')
                ai, di = names.index(a), names.index(d)
                positions = np.asarray(spec['global_joints_positions'])[0]
                rotation = np.asarray(spec['global_joints_rots'])[0, ai]
                np.testing.assert_allclose(positions[di] - positions[ai], rotation @ (neutral[di] - neutral[ai]), atol=1e-7)
        # Extreme targets are retained, not projected back into native limb reach.
        spec = condition_spec(self.task('strike'), [3, 0, 0], 40, GEOMETRY)
        self.assertGreater(spec['primary_target_k'][2], 2.5)

    def test_root_path_directions_and_sidestep_endpoint(self):
        for name, sign in [('walk', 1), ('jog', 1), ('back_walk', -1)]:
            spec = condition_spec(self.task(name), [1.2, 0, 0], LENGTHS[name], GEOMETRY)
            path = np.asarray(spec['smooth_root_2d'])
            np.testing.assert_allclose(path[:, 1], sign * np.float32(1.2) * np.arange(len(path)) / 30)
            self.assertFalse(path[:, 0].any())
        for distance in (-.7, .7):
            spec = condition_spec(self.task('sidestep'), [distance, 0, 0], 30, GEOMETRY)
            path = np.asarray(spec['smooth_root_2d'])
            np.testing.assert_allclose(path[-2:, 0], -distance)
            self.assertEqual(path[0, 0], 0)
            self.assertFalse(path[:, 1].any())

    def test_shortest_rotation_parallel_antiparallel_and_general(self):
        for a, b in [([0, 0, 2], [0, 0, 4]), ([0, 1, 0], [0, -2, 0]), ([1, 2, 3], [-1, 3, .2])]:
            rotation = shortest_rotation(a, b)
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(rotation), 1)
            np.testing.assert_allclose(rotation @ (a / np.linalg.norm(a)), b / np.linalg.norm(b), atol=1e-12)
        with self.assertRaises(ValueError):
            shortest_rotation([0, 0, 0], [1, 0, 0])

    def test_frozen_measurements_on_analytic_fixtures(self):
        for name, sign in [('walk', 1), ('back_walk', -1), ('jog', 1)]:
            positions = self.pose(name)
            positions[:, :, 0] += sign * 1.1 * np.arange(len(positions))[:, None] / 20
            np.testing.assert_allclose(measure_sample(self.task(name), positions)[0], [1.1], atol=1e-5)
        positions = self.pose('jump')
        positions[32, :, 2] += .4
        np.testing.assert_allclose(measure_sample(self.task('jump'), positions)[0], [.4], atol=1e-6)
        positions = self.pose('squat')
        positions[30:40, :, 2] -= .35
        np.testing.assert_allclose(measure_sample(self.task('squat'), positions)[0], [.35], atol=1e-6)
        positions = self.pose('reach')
        target = np.array([.3, -.2, .4])
        positions[-5:, 21] = positions[0, 0] + target
        np.testing.assert_allclose(measure_sample(self.task('reach'), positions)[0], target, atol=1e-6)
        positions = self.pose('sidestep')
        positions[:, :, 1] -= np.linspace(0, .7, len(positions))[:, None]
        np.testing.assert_allclose(measure_sample(self.task('sidestep'), positions)[0], [.7], atol=1e-6)

    def test_limb_metric_retains_root_travel_and_uses_ankle(self):
        positions = self.pose('kick')
        positions[:, :, 0] += np.linspace(0, .8, len(positions))[:, None]
        positions[:, 8, 0] += np.linspace(0, .5, len(positions))
        positions[:, 11, 0] += 100  # Toe proxy must not become the ankle metric.
        measured = measure_sample(self.task('kick'), positions)[0]
        np.testing.assert_allclose(measured, [1.3], atol=1e-6)
        positions = self.pose('strike')
        pulse = np.interp(np.arange(40), [0, 10, 20, 30, 39], [0, 0, .8, 0, 0])
        positions[:, 21, 0] += pulse
        baseline = measure_sample(self.task('strike'), positions)[0]
        positions[:, :, 0] += np.arange(40)[:, None] * .02
        self.assertGreater(float(measure_sample(self.task('strike'), positions)[0][0]), float(baseline[0]) + .05)

    def test_fixed_frame_translation_yaw_invariance(self):
        angle = .8
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        for name in TASKS:
            positions = self.pose(name)
            positions[:, :, 0] += np.arange(len(positions))[:, None] * .01
            if name == 'strike':
                # A unique punch event avoids the upstream argmax's undefined
                # tie between numerical noise in an entirely static arm.
                positions[:, 21, 0] += .8 * np.exp(-((np.arange(40) - 20) / 5) ** 2)
            changed = positions.copy()
            changed[..., :2] = positions[..., :2] @ rotation.T + [4, -3]
            before = measure_sample(self.task(name), positions)[0]
            after = measure_sample(self.task(name), changed)[0]
            np.testing.assert_allclose(before, after, atol=2e-5, err_msg=name)
        degenerate = self.pose('walk')
        degenerate[0, 1] = degenerate[0, 2]
        with self.assertRaisesRegex(ValueError, 'Degenerate'):
            canonicalize(degenerate)

    def test_summary_balancing_and_errors_before_averaging(self):
        task = self.task('walk')
        entries = []
        for record in samples(self.protocol, ['walk'], 1):
            error = .2 if record['seed'] != SEEDS[1] else -.4
            entries.append(dict(record, measured=[record['command'][0] + error], quality={},
                                native_primary_error_m=.1, skating_ratio=0.))
        result, _ = summarize(task, entries, SEEDS)
        np.testing.assert_allclose(result['measured'][0], [entries[0]['command'][0]], atol=1e-6)
        self.assertAlmostEqual(result['mae'][0], .8 / 3, places=6)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            summarize(task, entries[:-1], SEEDS)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            summarize(task, entries + entries[:1], SEEDS)

    def test_skating_uses_toes_ground_and_20hz(self):
        positions = self.pose('walk')
        positions[:, [10, 11], 2] = .02
        self.assertEqual(skating_ratio(positions), 0)
        positions[:, :, 0] += np.arange(len(positions))[:, None] / 20
        self.assertGreater(skating_ratio(positions), .9)
        positions[:, [10, 11], 2] = .1
        self.assertEqual(skating_ratio(positions), 0)

    def test_cache_does_not_mutate_native_embeddings(self):
        import torch
        calls = []
        def encoder(texts):
            calls.extend(texts)
            return torch.ones((1, 1, 4)), [1]
        cache = CachedEncoder(encoder)
        first, lengths = cache(['hello'])
        first.zero_()
        second, _ = cache(['hello'])
        self.assertEqual(calls, ['hello'])
        self.assertEqual(lengths, [1])
        self.assertTrue((second == 1).all())

    def test_synthetic_roundtrip_resume_reports_and_corruption(self):
        with tempfile.TemporaryDirectory(prefix='kimodo-unit-') as tmp:
            output, reports = Path(tmp) / 'samples', Path(tmp) / 'reports'
            unsigned = dict(settings=SETTINGS, geometry=GEOMETRY, runtime={'fixture': True}, files={}, assets={})
            with output_lock(output):
                run = prepare_run(output, BUNDLE / 'protocol.json', self.protocol, unsigned)
                engine = SyntheticEngine(self.protocol)
                self.assertEqual(generate_samples(self.protocol, output, engine, run, maximum_commands=1), {'generated': 81, 'reused': 0})
                self.assertEqual(generate_samples(self.protocol, output, engine, run, maximum_commands=1), {'generated': 0, 'reused': 81})
                self.assertEqual(engine.calls, 81)
                with self.assertRaisesRegex(ValueError, 'Run identity changed'):
                    prepare_run(output, BUNDLE / 'protocol.json', self.protocol, dict(unsigned, runtime={'fixture': False}))
            with self.assertRaisesRegex(ValueError, 'Missing'):
                score_run(self.protocol, run, output)
            result, entries, panels = score_run(self.protocol, run, output, allow_partial=True)
            self.assertEqual(result['scored_samples'], 81)
            self.assertEqual(len(result['tasks']), 9)
            self.assertIsNone(result['macro_normalized_mae'])
            self.assertFalse(result['complete'])
            reach = next(t for t in result['tasks'] if t['task'] == 'reach')
            self.assertIn('valid', reach['quality'])
            self.assertIn('all', reach['position_metrics'])
            save_reports(reports, self.protocol, run, result, entries, panels, output)
            for name in TASKS:
                self.assertTrue((reports / (name + '_response.png')).stat().st_size > 1000)
                html = (reports / (name + '_motions.html')).read_text()
                self.assertIn('Native 30 Hz', html)
                self.assertEqual(html.count('"prompt":'), 9)
            record = next(samples(self.protocol))
            path = archive_path(output, record)
            arrays = read_sample(path, record, run['run_id'], GEOMETRY['names77'])
            arrays['observed_motion'][0, 0] += 1
            write_npz(path, **arrays)
            with self.assertRaisesRegex(ValueError, 'content hash'):
                generate_samples(self.protocol, output, engine, run, maximum_commands=1)
            path.unlink()
            with self.assertRaisesRegex(ValueError, 'Incomplete prompt/seed'):
                score_run(self.protocol, run, output, allow_partial=True)

    def test_preflight_imports_no_torch_or_kimodo(self):
        # Fresh interpreter proves that CLI help and readiness imports cannot initialize CUDA.
        program = "import sys; import kimodo_benchmark.generate; assert 'torch' not in sys.modules; assert 'kimodo' not in sys.modules"
        subprocess.run([sys.executable, '-c', program], cwd=ROOT, check=True)


if __name__ == '__main__':
    unittest.main()

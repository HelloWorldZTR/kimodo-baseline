"""Current-main protocol parity tests; CPU only and no model loading."""

import copy
import os
from pathlib import Path
import sys
import unittest

import numpy as np
from omegaconf import OmegaConf

from kimodo_benchmark.protocol import ROOT, TASKS, identity, load_protocol, samples, validate_protocol
from kimodo_benchmark.score import connect_scoring_source

BUNDLE = Path(os.environ.get('KIMODO_BENCHMARK_MAIN_BUNDLE', ROOT / 'benchmark_sources/command9_main_20261024'))


class MainProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol(BUNDLE / 'protocol.json')
        cls.source = BUNDLE / 'scoring_source'
        connect_scoring_source(cls.source, cls.protocol)

    @classmethod
    def tearDownClass(cls):
        for name in tuple(sys.modules):
            if name.split('.')[0] in ('shared_motion', 'data_processing', 'src'):
                del sys.modules[name]
        sys.path[:] = [entry for entry in sys.path if Path(entry).resolve() != cls.source.resolve()]

    def test_all_plans_match_current_main(self):
        from shared_motion.tasks.registry import get_task
        from shared_motion.tasks.evaluation import references_from_manifest
        real = OmegaConf.load(self.source / 'config/data/real.yaml')
        reach = OmegaConf.load(self.source / 'config/task/parameters/reach.yaml')
        reach.workspace = str(self.source / reach.workspace)
        grid = OmegaConf.load(self.source / 'config/benchmark/dense.yaml')
        evaluate = OmegaConf.load(self.source / 'config/evaluate.yaml')
        for task in self.protocol['tasks']:
            references = references_from_manifest(self.source / real.episodes, task['name'])
            plan = get_task(task['name']).evaluation_plans(
                'benchmark', evaluate.points, reach if task['name'] == 'reach' else None,
                references, command_grid=grid)[0]
            np.testing.assert_array_equal(plan.commands.numpy(), np.asarray(task['commands'], np.float32))
            self.assertEqual(plan.lengths, task['lengths'])
            self.assertEqual(plan.metadata, task['metadata'])
            self.assertIn(task['reference']['episode_id'], {reference['id'] for reference in references})
            self.assertEqual(task['reference']['source_split'], 'train')

    def test_main_identity_and_new_bounds(self):
        self.assertEqual(self.protocol['sources']['main_revision'], '7a23d470f7b234a46bb5b2d7660e268faebf9728')
        self.assertEqual(self.protocol['sources']['release'], 'real_kick_v2_20261023')
        tasks = {task['name']: task for task in self.protocol['tasks']}
        self.assertEqual(tasks['strike']['definition']['bounds'], [[.33, .88]])
        self.assertEqual(tasks['kick']['definition']['bounds'], [[.46, 1.14]])
        self.assertEqual(tasks['reach']['definition']['bounds'], [[-.12, .64], [-.70, .33], [-.19, .94]])
        self.assertEqual(len(tasks['reach']['commands']), 1704)
        self.assertEqual(self.protocol['expected_samples'], 17496)
        records = list(samples(self.protocol, maximum_commands=3))
        self.assertEqual(len(records), 243)
        for name in TASKS:
            self.assertEqual(sum(record['task'] == name for record in records), 27)
        self.assertEqual(tasks['reach']['metadata']['candidate_status_counts']['reference_verified'], 1664)
        # Out-of-domain frozen labels are reported, never clamped into valid bins.
        self.assertEqual(tasks['kick']['metadata']['command_grid']['outside_reference_count'], 9)
        self.assertEqual(tasks['strike']['metadata']['command_grid']['outside_reference_count'], 1)

    def test_reject_out_of_range_commands_even_if_resigned(self):
        protocol = copy.deepcopy(self.protocol)
        task = next(task for task in protocol['tasks'] if task['name'] == 'kick')
        task['commands'][0][0] = 2.0
        del protocol['protocol_id']
        protocol['protocol_id'] = identity(protocol)
        with self.assertRaisesRegex(ValueError, 'outside current task bounds'):
            validate_protocol(protocol)


if __name__ == '__main__':
    unittest.main(verbosity=2)

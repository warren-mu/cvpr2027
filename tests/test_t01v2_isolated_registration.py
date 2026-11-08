"""CPU provenance wiring fixtures; no Unity or research measurements."""
import copy
from pathlib import Path
import unittest

from src.envs import pair_v2_isolated_shard as isolated
from src.tools import merge_pair_v2 as merge
from src.tools.t01v2_live_progress import MODULES


class IsolatedRegistrationTests(unittest.TestCase):
    def configuration(self):
        return isolated.enable_config({'name': 'fixture', 'scene': 'FloorPlan1', 'seed': 0})

    def test_exact_new_producer_and_policy_are_required_together(self):
        config = self.configuration()
        command = ['python3', '-u', '-m', 'src.envs.pair_v2_isolated_shard', '--name', 'fixture']
        self.assertEqual(merge.shard_producer(command, config), (['--name', 'fixture'], True))
        for key, value in (('attempt_setup_version', 'unreviewed'), ('attempt_setup_policy', {}),
                           ('producer_module', 'src.envs.pair_v2_solvable_shard'),
                           ('solvability_version', None)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                merge.shard_producer(command, {**config, key: value})

    def test_old_source_is_not_relabelled_as_isolated(self):
        config = self.configuration()
        old = ['python3', '-m', 'src.envs.pair_v2_solvable_shard']
        with self.assertRaises(ValueError):
            merge.shard_producer(old, config)
        for key in ('attempt_setup_version', 'attempt_setup_policy'):
            config.pop(key)
        config['producer_module'] = 'src.envs.pair_v2_solvable_shard'
        self.assertEqual(merge.shard_producer(old, config), ([], True))
        with self.assertRaises(ValueError):
            merge.shard_producer(['python3', '-m', 'src.envs.pair_v2_isolated_shard'], config)

    def test_policy_only_and_version_only_do_not_authorize_a_source(self):
        for missing in ('attempt_setup_version', 'attempt_setup_policy'):
            config = self.configuration()
            config.pop(missing)
            with self.assertRaises(ValueError):
                merge.shard_producer(['python3', '-m', 'src.envs.pair_v2_isolated_shard'], config)

    def test_unversioned_pair_cannot_claim_new_setup_evidence(self):
        for field in ('attempt_setup_version', 'attempt_setup_policy', 'attempt_setup',
                      'attempt_setup_reference', 'attempt_setup_native_prefix_reference'):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Historical source cannot silently claim'):
                merge.check_pair({field: {'claimed': True}}, Path('/unused/pair.json'))

    def test_source_and_pair_setup_policies_must_agree_before_reading_artifacts(self):
        config = self.configuration()
        keys = ('preflight_version', 'evidence_version', 'search_version', 'search_policy',
                'solvability_version', 'solvability_policy', 'generation_version', 'generation_policy',
                'attempt_setup_version', 'attempt_setup_policy')
        arguments = {key: config[key] for key in keys}
        row = copy.deepcopy(arguments)
        row['attempt_setup_policy'] = {}
        with self.assertRaisesRegex(ValueError, 'Pair attempt setup differs'):
            merge.check_pair(row, Path('/unused/pair.json'), clone_enabled=True, **arguments)

    def test_live_progress_recognizes_both_distinct_producers(self):
        self.assertIn('src.envs.pair_v2_isolated_shard', MODULES)
        self.assertIn('src.envs.pair_v2_solvable_shard', MODULES)


if __name__ == '__main__':
    unittest.main()

"""Legacy controller entry must retain every original science/provenance gate.

These CPU dispatch fixtures mock the already separately tested native boundary;
they assert no retrospective fixed-label claim or source-record mutation.
"""
import copy
import unittest
from unittest.mock import patch

from test_pair_table_ranked_source import fixture, boundaries
from src.envs import pair_table_source as source
from src.envs import pair_v2_look_unit_policy as policy
from src.envs import pair_v2_fixed_labels as labels
from src.envs import pair_v2_stay_order as stay
from src.tools import t01v2_controller_review_checkpoint as controller_checkpoint


def legacy_fixture():
    f = fixture(kind='look_stay')
    f['container']['_meta']['command'][2] = source.CONTROLLER_CHECKPOINT_MODULE
    common = {k: copy.deepcopy(f['data']['config'][k]) for k in
              ('scene', 'raw_evidence_dir', 'enable_candidate_clone', 'solvability_version',
               'candidate_policy', 'frozen_render_version', 'frozen_render_policy')}
    cfg = {**common, **policy.camera_fields(60, 'FloorPlan10'), **stay.fields(),
           'producer_module': policy.PRODUCER}
    cfg['shard'] = copy.deepcopy(cfg)
    preflight = copy.deepcopy(f['data']['metrics']['scene_preflight_reference'])
    f['data']['config'] = cfg
    f['data']['metrics'] = {**{k: copy.deepcopy(v) for k, v in cfg.items() if k != 'shard'},
                            'scene_preflight_reference': preflight}
    for key in labels.PAIR_FIELDS:
        f['row'].pop(key, None)
    return f


class LegacyControllerSourceTests(unittest.TestCase):
    def test_exact_controller_producer_only(self):
        f = legacy_fixture()
        with patch.object(source, 'audited_result', return_value=(f['container'], f['container_ref'])):
            self.assertEqual(source._container('unused')[2], [f['entry']])
            f['container']['_meta']['command'] = ['python3', '-m', 'src.tools.untrusted',
                                                 '--hint', source.CONTROLLER_CHECKPOINT_MODULE]
            with self.assertRaises(ValueError):
                source._container('unused')

    def test_legacy_loader_binds_own_controller_without_rewriting_old_labels(self):
        f = legacy_fixture()
        before = copy.deepcopy(f)
        with boundaries(self, f, expected_policy=policy, support_module=controller_checkpoint) as calls:
            package = source.load_reviewed_pair('unused', f['pair_id'], pair_file=f['pair_ref']['file'])
        self.assertEqual(calls['load'], [True])
        self.assertEqual(calls['association'], [True])
        self.assertEqual(len(calls['science']), 1)
        self.assertEqual(package['controller_preflight_association']['controller_id'], 'marked_controller_000')
        self.assertIsNone(package['fixed_label_verification']['fixed_label_version'])
        self.assertNotIn('t01_t02_assignment_equal', package['fixed_label_verification'])
        self.assertFalse(package['execution_authorized'])
        self.assertIsNone(package['full_t02_acceptance'])
        self.assertEqual(f, before)

    def test_original_scientific_or_controller_failure_still_rejects(self):
        for gate in ('science', 'association'):
            f = legacy_fixture()
            with self.subTest(gate=gate), boundaries(self, f, expected_policy=policy,
                    support_module=controller_checkpoint, fail_science=gate == 'science',
                    fail_association=gate == 'association'):
                with self.assertRaisesRegex(ValueError, 'original'):
                    source.load_reviewed_pair('unused', f['pair_id'])

    def test_foreign_published_controller_is_not_accepted(self):
        f = legacy_fixture()
        f['entry']['controller_preflight_association']['controller_id'] = 'marked_controller_999'
        with boundaries(self, f, expected_policy=policy, support_module=controller_checkpoint):
            with self.assertRaisesRegex(ValueError, 'Published Controller'):
                source.load_reviewed_pair('unused', f['pair_id'])


if __name__ == '__main__':
    unittest.main()

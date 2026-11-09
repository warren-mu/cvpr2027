"""CPU G01-to-T02 dispatch tests; no rendered data or review authorization.

The real G01 source context, policy, passed-pair matching, label context and
table package checks run here. Original byte-backed Controller/lease, geometry,
collection/admission and scientific validators have their own fixtures; their
explicit boundaries below verify routing and failure propagation, not physics.
"""
import copy
from contextlib import ExitStack, contextmanager
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

from src.envs import pair_table_source as source
from src.envs import pair_v2_fov as fov
from src.envs import pair_v2_g01_policy as policy
from src.envs import pair_v2_g01_search as search
from src.envs import pair_v2_fixed_labels as labels
from src.envs import pair_v2_stay_order as stay
from src.envs import pair_v2_role_policy as roles
from src.tools import t01v2_g01_review_checkpoint as g01
from src.tools import t01v2_g01_association as association
from src.tools import t01v2_controller_association as controllers
from src.tools import t01v2_flip_ranked_review_checkpoint as ranked

_spec = importlib.util.spec_from_file_location('g01_table_existing_fixture',
    Path(__file__).with_name('test_pair_table_ranked_source.py'))
_existing = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_existing)


def fixture(*, kind='look_flip', pool='look'):
    f = _existing.fixture(kind=kind)
    name = 't01v2_checkpoint_g01_table_fixture'
    f['container_ref']['file'] = 'results/' + name + '.json'
    f['container']['_meta'].update(name=name,
        command=['python3', '-m', source.G01_CHECKPOINT_MODULE, '--name', name])
    f['container']['config']['name'] = name
    cfg = f['data']['config']
    # A new synthetic G01 fixture, never an upgrade of a historical source.
    for value in (cfg, cfg['shard'], f['data']['metrics']):
        for key in tuple(value):
            if key.startswith(('flip_unit_', 'flip_search_', 'look_unit_')):
                value.pop(key)
        value.update(policy.declared('FloorPlan203'), **stay.fields(),
            producer_module=policy.PRODUCER, enable_candidate_clone=False, scene_pairs_per_class=8)
    row = f['row']
    mode = 'flip' if kind.endswith('flip') else 'stay'
    row.update(policy.declared('FloorPlan203'), producer_module=policy.PRODUCER,
        mode=mode, proposed_pool=pool, proposal={'mode': mode, 'projection_pool': {'pool': pool}},
        projection_pool_reference={'file': '/synthetic/raw/projection_pool.json', 'sha256': 'pool'})
    f['association'].update(version=association.VERSION,
        baseline_reference={'file': '/synthetic/raw/baseline.json', 'sha256': 'baseline'},
        witness_resource_admission={'owner': {'pid': 11}, 'worlds': [
            {'world_index': 0, 'lease_token': 'independent-w0'},
            {'world_index': 1, 'lease_token': 'independent-w1'}]})
    f['entry']['controller_preflight_association'] = copy.deepcopy(f['association'])
    return f


@contextmanager
def boundaries(test, f, *, fail=None):
    """Enter the actual G01 context; mock only independently tested boundaries."""
    before = (source.merge.shard_producer, source.merge.load_source, source.merge.check_pair,
              roles.verify_proposal_role, fov.declared_fields, controllers.policy)
    calls = []
    def event(label):
        calls.append(label)
        test.assertIs(fov.declared_fields, policy.declared_fields)
        test.assertIsNot(source.merge.load_source, before[1])
        if fail == label:
            raise ValueError('original ' + label + ' failure')
    def load(path):
        event('load')
        test.assertEqual(path, source.merge.ROOT / f['source_ref']['file'])
        return f['data'], f['source_ref']
    def collection(data):
        event('collection'); test.assertIs(data, f['data'])
    def admission(data):
        event('admission'); test.assertIs(data, f['data'])
    def controller(*args, **kwargs):
        event('association')
        test.assertEqual(args, (f['data'], f['source_ref'], f['compact'], f['row'], f['pair_path']))
        test.assertIn(kwargs, ({}, {'root': source.merge.ROOT.resolve()}))
        return copy.deepcopy(f['association'])
    def science(row, path, **kwargs):
        event('science')
        test.assertEqual(row, f['row']); test.assertEqual(path, f['pair_path'])
        test.assertEqual(kwargs['raw_root'], f['data']['config']['raw_evidence_dir'])
        test.assertEqual(kwargs['field_of_view'], 60)
        test.assertEqual(kwargs['frozen_render_policy'], f['data']['config']['frozen_render_policy'])
        test.assertEqual(kwargs['candidate_policy'], f['data']['config']['candidate_policy'])
        test.assertEqual(labels._CONTEXT.get()[0], f['pair_id'])
        return {'proposal_identity': f['entry']['proposal_identity'], 'full_scientific_fixture': True}
    def geometry(*args, **kwargs):
        event('geometry')
        test.assertEqual(args[0], f['row'])
        return {'passed': True, 'scope': 'geometry boundary fixture'}
    passed = ({'eligible': True, 'partial_source': True}, f['journal_ref'],
              [(f['compact'], f['pair_ref'], f['pair_path'])])
    with ExitStack() as stack:
        for target, name, options in (
            (source, 'audited_result', {'return_value': (f['container'], f['container_ref'])}),
            (g01, '_BASE_LOAD', {'side_effect': load}),
            (g01, 'verify_collection', {'side_effect': collection}),
            (g01, 'verify_admission', {'side_effect': admission}),
            (source, 'verified_passed_rows', {'return_value': passed}),
            (g01, 'verified_passed_rows', {'return_value': passed}),
            (source.merge, 'read_reference', {'side_effect': lambda ref, **kw: copy.deepcopy(f['files'][ref['file']])}),
            (source.merge, 'reference_path', {'side_effect': lambda ref, **kw: Path(ref['file'])}),
            (source.merge, 'final_pair_eligible', {'return_value': True}),
            (association, 'verify_pair_controller', {'side_effect': controller}),
            (controllers, 'verify_pair_controller', {'side_effect': AssertionError('G01 used ordinary Controller verifier')}),
            (ranked, 'source_support', {'side_effect': AssertionError('G01 used ranked source context')}),
            (source.merge, 'check_pair', {'side_effect': science}),
            (g01.geometry_proof, 'IndexedReader', {'return_value': lambda ref: {**search.fields(), 'fixture': ref['file']}}),
            (g01.geometry_proof, 'verify_geometry_evidence', {'side_effect': geometry}),
            (labels, 'verify_witness_labels', {'side_effect': lambda row, replay, read: {'fixture': replay}}),
            (source, 'native_witness', {'side_effect': lambda row, index, replay, read: {'fixture_world': index}}),
        ):
            stack.enter_context(patch.object(target, name, **options))
        # The context must restore to the patched scientific boundary while it
        # is active; after ExitStack closes every original function is restored.
        patched = (source.merge.shard_producer, source.merge.load_source, source.merge.check_pair,
                   roles.verify_proposal_role, fov.declared_fields, controllers.policy)
        try:
            yield calls
        finally:
            test.assertEqual((source.merge.shard_producer, source.merge.load_source, source.merge.check_pair,
                roles.verify_proposal_role, fov.declared_fields, controllers.policy), patched)
            test.assertIsNone(labels._CONTEXT.get())
    test.assertEqual((source.merge.shard_producer, source.merge.load_source, source.merge.check_pair,
        roles.verify_proposal_role, fov.declared_fields, controllers.policy), before)


class G01TableSourceTests(unittest.TestCase):
    def test_exact_g01_checkpoint_dispatch_rejects_argument_hint_and_wrong_name(self):
        f = fixture()
        with patch.object(source, 'audited_result', return_value=(f['container'], f['container_ref'])):
            self.assertEqual(source._container('unused')[2], [f['entry']])
            f['container']['_meta']['command'] = ['python3', '-m', 'src.tools.unreviewed',
                '--hint', source.G01_CHECKPOINT_MODULE]
            with self.assertRaises(ValueError): source._container('unused')
            f['container']['_meta']['command'] = ['python3', '-m', source.G01_CHECKPOINT_MODULE, '--name', 'another']
            with self.assertRaises(ValueError): source._container('unused')

    def test_actual_g01_context_keeps_all_classes_pool_crossovers_and_fixed_labels(self):
        for kind in search.KINDS:
            for pool in ('look', 'direct'):
                with self.subTest(kind=kind, pool=pool):
                    f = fixture(kind=kind, pool=pool); before = copy.deepcopy(f)
                    with boundaries(self, f) as calls:
                        package = source.load_reviewed_pair('unused', f['pair_id'], pair_file=f['pair_ref']['file'])
                    self.assertEqual(calls[:3], ['load', 'collection', 'admission'])
                    self.assertEqual(calls.count('association'), 2)
                    self.assertEqual(calls.count('science'), 1)
                    needs_geometry = kind.endswith('flip') and pool == 'look'
                    self.assertEqual(calls.count('geometry'), int(needs_geometry))
                    checked = package['independent_verification']
                    self.assertEqual(checked['g01_proposal_verification']['proposed_pool'], pool)
                    self.assertEqual(checked['g01_proposal_verification']['actual_kind'], kind)
                    self.assertEqual(checked['g01_proposal_verification']['class_cross_over'], kind.split('_', 1)[0] != pool)
                    self.assertEqual(package['controller_preflight_association'], f['association'])
                    self.assertEqual(package['source_config']['producer_module'], policy.PRODUCER)
                    self.assertEqual(package['source_provenance']['journal'], f['journal_ref'])
                    self.assertTrue(package['fixed_label_verification']['t01_t02_assignment_equal'])
                    self.assertEqual(package['kind'], kind)
                    self.assertFalse(package['execution_authorized'])
                    self.assertIsNone(package['human_review_approved'])
                    self.assertIsNone(package['full_t02_acceptance'])
                    self.assertEqual(f, before)

    def test_original_collection_admission_association_geometry_and_science_fail_closed(self):
        for failure in ('collection', 'admission', 'association', 'geometry', 'science'):
            with self.subTest(failure=failure):
                f = fixture()
                with boundaries(self, f, fail=failure) as calls:
                    with self.assertRaisesRegex(ValueError, 'original ' + failure + ' failure'):
                        source.load_reviewed_pair('unused', f['pair_id'])
                if failure != 'science': self.assertNotIn('science', calls)

    def test_published_association_and_label_tampering_cannot_grant_table_input(self):
        for change in ('controller', 'lease', 'source', 'labels'):
            with self.subTest(change=change):
                f = fixture()
                published = f['entry']['controller_preflight_association']
                if change == 'controller': published['controller_id'] = 'marked_controller_001'
                elif change == 'lease': published['witness_resource_admission']['worlds'][0]['lease_token'] = 'another'
                elif change == 'source': published['source_reference']['sha256'] = 'another'
                else: f['row']['candidate_label_assignment'] = {'A': 1, 'B': 1}
                with boundaries(self, f), self.assertRaises(ValueError):
                    source.load_reviewed_pair('unused', f['pair_id'])

    def test_g01_container_cannot_load_ranked_or_old_g01_scene(self):
        for change in ('ranked', 'old_g01', 'ranked_row'):
            with self.subTest(change=change):
                f = fixture()
                if change == 'ranked':
                    f['data'] = _existing.fixture()['data']
                elif change == 'old_g01':
                    f['data']['config']['g01_version'] = 'g01_mixed_train80_selected60_v1'
                else:
                    f['row']['flip_unit_version'] = ranked.policy.VERSION
                with boundaries(self, f) as calls, self.assertRaises(ValueError):
                    source.load_reviewed_pair('unused', f['pair_id'])
                self.assertNotIn('science', calls)

    def test_existing_dispatchers_do_not_enter_g01_context(self):
        from src.envs import pair_v2_stay_priority_policy as stay_policy
        from src.tools import t01v2_stay_priority_review_checkpoint as stay_checker
        for mode in ('legacy', 'ranked', 'stay'):
            with self.subTest(mode=mode):
                f = _existing.fixture(legacy=mode == 'legacy')
                options = {}
                if mode == 'stay':
                    f['container']['_meta']['command'][2] = source.STAY_CHECKPOINT_MODULE
                    fields = stay_policy.camera_fields(60, 'FloorPlan203')
                    for value in (f['data']['config'], f['data']['config']['shard'], f['data']['metrics']):
                        value.update(fields, producer_module=stay_policy.PRODUCER)
                    options.update(expected_policy=stay_policy, support_module=stay_checker)
                with patch.object(g01, 'source_support', side_effect=AssertionError('Old source entered G01 context')):
                    with _existing.boundaries(self, f, **options) as calls:
                        package = source.load_reviewed_pair('unused', f['pair_id'])
                self.assertEqual(calls['load'], [mode != 'legacy'])
                self.assertFalse(package['execution_authorized'])


if __name__ == '__main__':
    unittest.main()

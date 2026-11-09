"""CPU source-dispatch fixtures; full scientific validators are boundary mocks.

Controller association, complete geometry, native witness and table validation
retain their own byte-backed tests. These tests exercise the new loader wiring
and fail-closed package binding; they do not claim any simulator result.
"""
import copy
from contextlib import contextmanager
from pathlib import Path
import unittest
from unittest.mock import patch

from src.envs import pair_table_source as source
from src.envs import pair_v2_flip_ranked_policy as policy
from src.envs import pair_v2_fixed_labels as labels
from src.envs import pair_v2_stay_order as stay
from src.tools import t01v2_controller_association as controllers
from src.tools import t01v2_flip_ranked_review_checkpoint as ranked


def xyz(x=0.):
    return {'x': x, 'y': 0., 'z': 0.}


def fixture(*, legacy=False, kind='look_flip'):
    pair_id = 'floorplan203_00000'
    name = 't01v2_checkpoint_ranked_fixture'
    container_ref = {'file': 'results/' + name + '.json', 'audit': {'warn': []}}
    source_ref = {'file': 'results/synthetic_scene.json', 'sha256': 'source-sha',
                  'metrics_sha256': 'metrics-sha', 'git_commit': 'source-commit'}
    pair_ref = {'file': '/synthetic/pairs/pair.json', 'sha256': 'pair-sha'}
    journal_ref = {'file': '/synthetic/journal.jsonl', 'sha256': 'journal-sha'}
    cfg = {**policy.camera_fields(60, 'FloorPlan203'), **stay.fields(),
           'scene': 'FloorPlan203', 'producer_module': policy.PRODUCER,
           'raw_evidence_dir': '/synthetic/raw', 'enable_candidate_clone': False,
           'solvability_version': 'fixture-solvability', 'candidate_policy': 'fixture-marked-policy',
           'frozen_render_version': 'fixture-frozen', 'frozen_render_policy': {'fixture': 'unchanged'}}
    cfg['shard'] = copy.deepcopy(cfg)
    measured = {**{key: value for key, value in cfg.items() if key != 'shard'},
                'scene_preflight_reference': {'file': '/synthetic/raw/controller001/scene_preflight.json',
                                              'sha256': 'last-controller-sha'}}
    command = {'position': xyz(), 'rotation': xyz(), 'horizon': 0., 'standing': True, 'forceAction': False}
    agent = {'position': xyz(), 'rotation': xyz(), 'horizon': 0., 'standing': True}
    files = {}; portable = []
    for world in ('w0', 'w1'):
        for basename, content in ((f'snapshot_{world}.json', {
                'agent': copy.deepcopy(agent), 'agent_setup_command': copy.deepcopy(command),
                'clone_context': None, 'reference': {'file': '/synthetic/raw/' + world + '.json', 'sha256': world},
                'capture_reference': {'files': {key: {'file': '/synthetic/raw/' + world + '_' + key,
                                                     'sha256': world + key} for key in ('rgb', 'metadata', 'masks')}}}),
                (f'witness_{world}.json', {'fixture': world})):
            ref = {'file': '/synthetic/pairs/' + basename, 'sha256': basename}
            portable.append(ref); files[ref['file']] = content
    domain = {'reachable_positions': [xyz(), xyz(.25)],
              'reachable_reference': {'file': '/synthetic/raw/reachable.json', 'sha256': 'reachable'}}
    row = {**labels.pair_fields(pair_id, ['A', 'B']),
        'pair_id': pair_id, 'scene': 'FloorPlan203', 'kind': kind, 'status': 'passed', 'discard_reasons': [],
        'target_w0': 'A', 'target_w1': 'B' if kind.endswith('flip') else 'A',
        'identities': {'candidate_names': ['A', 'B'], 'anchor_name': 'X'},
        'view': {'position': xyz(), 'yaw': 0., 'horizon': 0.}, 'instruction': 'Select the marked object closest to X.',
        'scene_preflight_reference': {'file': '/synthetic/pairs/scene_preflight.json', 'sha256': 'pair-controller-sha'},
        'portable_files': portable, 'search_controller_id': 'marked_controller_000',
        'observation_search': {'domains': [domain, copy.deepcopy(domain)], 'selected_action': {'kind': 'turn_left'},
            'anchor_visible_actions_w0': [{'kind': 'turn_left'}], 'anchor_visible_actions_w1': [{'kind': 'turn_right'}]}}
    files[pair_ref['file']] = row
    compact = {key: row[key] for key in ('pair_id', 'scene', 'kind', 'status', 'discard_reasons')}
    compact['raw_attempt'] = {'file': '/synthetic/raw/attempt.json', 'sha256': 'attempt'}
    association = {'version': controllers.VERSION, 'controller_id': 'marked_controller_000',
        'source_final_controller_id': 'marked_controller_001', 'source_reference': copy.deepcopy(source_ref),
        'pair_preflight_reference': copy.deepcopy(row['scene_preflight_reference']),
        'controller_preflight_reference': {'file': '/synthetic/raw/controller000/scene_preflight.json',
                                          'sha256': 'pair-controller-sha'},
        'original_source_final_preflight_reference': copy.deepcopy(measured['scene_preflight_reference']),
        'source_record_mutated': False}
    entry = {key: row[key] for key in ('pair_id', 'scene', 'kind', 'target_w0', 'target_w1')}
    entry.update(pair_reference=pair_ref, source_file=source_ref['file'], source_sha256=source_ref['sha256'],
        source_metrics_sha256=source_ref['metrics_sha256'], proposal_identity='authenticated-proposal',
        controller_preflight_association=copy.deepcopy(association))
    container = {'_meta': {'name': name, 'end': 'recorded', 'exit_code': 0,
        'command': ['python3', '-m', 'src.tools.t01v2_review_checkpoint' if legacy else source.RANKED_CHECKPOINT_MODULE,
                    '--name', name]}, 'config': {'name': name}, 'sources': [copy.deepcopy(source_ref)],
        'metrics': {'kind': 't01v2_review_checkpoint', 'status': 'checkpoint_ready_for_human_review',
            'errors': [], 'final_pilot_acceptance': False, 'candidate_pairs': [entry]}}
    if legacy:
        cfg.clear(); cfg.update(raw_evidence_dir='/synthetic/raw', solvability_version='fixture-solvability')
        measured['scene_preflight_reference'] = copy.deepcopy(row['scene_preflight_reference'])
        for key in labels.PAIR_FIELDS:
            row.pop(key, None)
    return {'container': container, 'container_ref': container_ref, 'data': {'config': cfg, 'metrics': measured},
        'source_ref': source_ref, 'entry': entry, 'row': row, 'pair_ref': pair_ref,
        'pair_path': Path(pair_ref['file']), 'compact': compact, 'journal_ref': journal_ref,
        'association': association, 'files': files, 'pair_id': pair_id}


@contextmanager
def boundaries(test, f, *, fail_science=False, fail_association=False, expected_policy=policy, support_module=ranked):
    active = {'entered': False}
    calls = {'load': [], 'science': [], 'association': []}
    original_policy = controllers.policy
    @contextmanager
    def support():
        test.assertFalse(active['entered']); active['entered'] = True
        try:
            yield
        finally:
            active['entered'] = False
    def load(path):
        calls['load'].append(active['entered'])
        return f['data'], f['source_ref']
    def association(*args):
        test.assertTrue(active['entered'])
        test.assertIs(controllers.policy, expected_policy)
        test.assertEqual(args, (f['data'], f['source_ref'], f['compact'], f['row'], f['pair_path']))
        calls['association'].append(True)
        if fail_association:
            raise ValueError('original Controller association failed')
        return copy.deepcopy(f['association'])
    def science(row, pair_path, **kwargs):
        calls['science'].append((active['entered'], copy.deepcopy(kwargs)))
        test.assertEqual(row, f['row']); test.assertEqual(pair_path, f['pair_path'])
        if fail_science:
            raise ValueError('original scientific gate failed')
        return {'proposal_identity': 'authenticated-proposal', 'fixture': 'full-checker boundary'}
    with patch.object(source, 'audited_result', return_value=(f['container'], f['container_ref'])), \
            patch.object(support_module, 'source_support', side_effect=support), \
            patch.object(source.merge, 'load_source', side_effect=load), \
            patch.object(source, 'verified_passed_rows', return_value=(
                {'eligible': True, 'partial_source': True}, f['journal_ref'],
                [(f['compact'], f['pair_ref'], f['pair_path'])])), \
            patch.object(source.merge, 'read_reference', side_effect=lambda ref, **kw: copy.deepcopy(f['files'][ref['file']])), \
            patch.object(source.merge, 'reference_path', side_effect=lambda ref, **kw: Path(ref['file'])), \
            patch.object(source.merge, 'final_pair_eligible', return_value=True), \
            patch.object(controllers, 'verify_pair_controller', side_effect=association), \
            patch.object(source.merge, 'check_pair', side_effect=science), \
            patch.object(source, 'native_witness', side_effect=lambda row, index, replay, read: {'fixture_world': index}):
        try:
            yield calls
        finally:
            test.assertFalse(active['entered'])
            test.assertIs(controllers.policy, original_policy)


class RankedTableSourceTests(unittest.TestCase):
    def test_stay_priority_source_uses_exact_policy_and_full_scientific_chain(self):
        from src.envs import pair_v2_stay_priority_policy as stay_policy
        from src.tools import t01v2_stay_priority_review_checkpoint as stay_checker
        f = fixture()
        f['container']['_meta']['command'][2] = source.STAY_CHECKPOINT_MODULE
        fields = stay_policy.camera_fields(60, 'FloorPlan203')
        cfg = f['data']['config']
        cfg.update(fields, producer_module=stay_policy.PRODUCER)
        cfg['shard'].update(fields, producer_module=stay_policy.PRODUCER)
        f['data']['metrics'].update(fields, producer_module=stay_policy.PRODUCER)
        for failure in (False, True):
            with self.subTest(scientific_failure=failure):
                with boundaries(self, f, expected_policy=stay_policy, support_module=stay_checker,
                                fail_science=failure) as calls:
                    if failure:
                        with self.assertRaisesRegex(ValueError, 'original scientific gate failed'):
                            source.load_reviewed_pair('unused', f['pair_id'])
                    else:
                        package = source.load_reviewed_pair('unused', f['pair_id'])
                        self.assertTrue(package['fixed_label_verification']['t01_t02_assignment_equal'])
                        self.assertEqual(package['source_config']['producer_module'], stay_policy.PRODUCER)
                        self.assertFalse(package['execution_authorized'])
                self.assertEqual(calls['load'], [True])
                self.assertEqual(calls['association'], [True])
        f['container']['_meta']['command'] = ['python3', '-m', 'src.tools.unknown', '--hint', source.STAY_CHECKPOINT_MODULE]
        with patch.object(source, 'audited_result', return_value=(f['container'], f['container_ref'])):
            with self.assertRaises(ValueError):
                source._container('unused')

    def test_new_checkpoint_exact_command_whitelist(self):
        f = fixture()
        with patch.object(source, 'audited_result', return_value=(f['container'], f['container_ref'])):
            self.assertEqual(source._container('unused')[2], [f['entry']])
            f['container']['_meta']['command'] = ['python3', '-m', 'src.tools.unreviewed', '--hint', source.RANKED_CHECKPOINT_MODULE]
            with self.assertRaises(ValueError): source._container('unused')

    def test_pair_controller_not_last_controller_and_fixed_labels_bind_both_look_modes(self):
        for kind in ('look_flip', 'look_stay'):
            with self.subTest(kind=kind):
                f = fixture(kind=kind); before = copy.deepcopy(f)
                with boundaries(self, f) as calls:
                    package = source.load_reviewed_pair('unused', f['pair_id'])
                self.assertEqual(calls['load'], [True]); self.assertEqual(calls['association'], [True])
                self.assertTrue(calls['science'][0][0])
                kwargs = calls['science'][0][1]
                self.assertEqual(kwargs['field_of_view'], 60)
                self.assertEqual(kwargs['frozen_render_version'], f['data']['config']['frozen_render_version'])
                self.assertEqual(kwargs['candidate_policy'], f['data']['config']['candidate_policy'])
                self.assertEqual(package['controller_preflight_association']['controller_id'], 'marked_controller_000')
                self.assertEqual(package['controller_preflight_association']['source_final_controller_id'], 'marked_controller_001')
                self.assertTrue(package['fixed_label_verification']['t01_t02_assignment_equal'])
                self.assertEqual(package['source_config']['field_of_view'], 60)
                self.assertFalse(package['execution_authorized']); self.assertIsNone(package['human_review_approved'])
                self.assertEqual(package['source_provenance']['journal'], f['journal_ref'])
                self.assertEqual(f, before)

    def test_old_source_does_not_enter_new_context_or_gain_fixed_label_claim(self):
        f = fixture(legacy=True, kind='direct_flip')
        with boundaries(self, f) as calls:
            package = source.load_reviewed_pair('unused', f['pair_id'])
        self.assertEqual(calls['load'], [False]); self.assertEqual(calls['association'], [])
        self.assertFalse(calls['science'][0][0])
        self.assertNotIn('fixed_label_verification', package)
        self.assertNotIn('controller_preflight_association', package)
        f['data']['metrics']['scene_preflight_reference']['sha256'] = 'different'
        with boundaries(self, f), self.assertRaisesRegex(ValueError, 'Pair preflight differs'):
            source.load_reviewed_pair('unused', f['pair_id'])

    def test_wrong_source_version_association_and_fixed_assignment_fail_closed(self):
        for change in ('version', 'controller', 'source', 'assignment'):
            with self.subTest(change=change):
                f = fixture()
                if change == 'version': f['data']['config']['flip_unit_version'] = 'ranked-v1'
                elif change == 'controller': f['entry']['controller_preflight_association']['controller_id'] = 'marked_controller_001'
                elif change == 'source': f['entry']['controller_preflight_association']['source_reference']['sha256'] = 'another-source'
                else: f['row']['candidate_label_assignment'] = {'A': 1, 'B': 1}
                with boundaries(self, f), self.assertRaises(ValueError):
                    source.load_reviewed_pair('unused', f['pair_id'])

    def test_full_scientific_and_controller_failures_propagate_with_context_restored(self):
        for flag, error in (('fail_science', 'original scientific gate failed'),
                            ('fail_association', 'original Controller association failed')):
            with self.subTest(flag=flag):
                f = fixture()
                with boundaries(self, f, **{flag: True}), self.assertRaisesRegex(ValueError, error):
                    source.load_reviewed_pair('unused', f['pair_id'])


if __name__ == '__main__':
    unittest.main()

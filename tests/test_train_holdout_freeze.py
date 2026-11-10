"""Synthetic metadata boundaries, not substitute native/cache qualifications."""
from contextlib import ExitStack
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.tools import train_holdout_freeze as holdout
from src.tools import smoke_e2e_pilot_dataset as dataset


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()
    path.write_bytes(body)
    return {'file': str(path), 'sha256': hashlib.sha256(body).hexdigest()}


def synthetic_qualification(root):
    entries = []
    scenes = [f'FloorPlan{base + n}' for base in (0, 200, 300, 400) for n in range(1, 6)]
    for i in range(83):
        entries.append({'dataset_key': f'fixture:{i:03}', 'proposal_identity': hashlib.sha256(str(i).encode()).hexdigest(),
            'scene': scenes[(i // 4) % len(scenes)], 'kind': holdout.KINDS[i % 4], 'world_order': [0, 1]})
    refs, proofs = [], []
    for index in range(4):
        name = 'fixture_cache_' + str(index) if index < 3 else Path(holdout.CACHE31_FILE).stem
        fields = {field: {'file': f'/scratch/fixture/{index}/{field}.json', 'sha256': str(index + 1) * 64}
                  for field in ('authorization_manifest', 'validation_report', 'sampling_plan')}
        metrics = dict(fields, producer_exit_code=0, errors=[], cleanup_errors=[], cache_matches_live=True,
                       sample_complete=True, downstream_allowed=True, stop_downstream=False)
        data = {'config': {}, 'metrics': metrics, '_meta': {'name': name, 'end': 'ended', 'exit_code': 0,
                    'metrics_sha256': holdout.canonical_hash(metrics)}}
        written = write(root / 'results' / (name + '.json'), data)
        reference = dict(written, file='results/' + name + '.json')
        refs.append(reference)
        proofs.append(dict(fields, source=dict(reference, metrics_sha256=data['_meta']['metrics_sha256']),
                           dataset_keys=[r['dataset_key'] for r in entries[index::4]]))
    policy = write(root / 'configs/fixture_policy.json', {'cache_sources': refs[:3]})
    split = dataset.split_entries(entries, dataset.SCENE_SPLIT)
    manifest = {'version': dataset.VERSION, 'partition': 'train', 'field_of_view': 60, 'seed': 0,
                'cache_sources': refs, 'entries': entries, 'counts_by_kind': holdout.counts(entries), 'split': split}
    manifest['common_pool_hash'] = holdout.canonical_hash(manifest)
    manifest_ref = write(root / holdout.MANIFEST_FILE, manifest)
    proof = {'version': dataset.VERSION, 'cache_authorized': True, 'counts_by_kind': holdout.counts(entries),
        'split_policy': dataset.SCENE_SPLIT, 'split': split, 'train_keys': split['train_keys'], 'eval_keys': split['eval_keys'],
        'world_orders': {row['dataset_key']: row['world_order'] for row in entries},
        'sources': [p['source'] for p in proofs], 'formal_cache_batches': proofs,
        'common_pool_hash': manifest['common_pool_hash']}
    argv = ['python3', '-u', '-m', 'src.tools.smoke_e2e_pilot_dataset', '--name', holdout.QUALIFICATION_NAME]
    for reference in refs: argv += ['--cache-source', json.dumps(reference)]
    argv += ['--out', holdout.MANIFEST_FILE, '--split-policy', holdout.POLICY, '--timeout-seconds', '5160']
    data = {'_meta': {'command': argv, 'name': holdout.QUALIFICATION_NAME, 'end': 'ended', 'exit_code': 0},
        'config': {'name': holdout.QUALIFICATION_NAME, 'version': dataset.VERSION, 'cache_source': refs,
            'deadline_fallback_52': False, 'out': holdout.MANIFEST_FILE, 'split_policy': holdout.POLICY},
        'metrics': {'kind': 'smoke_e2e_pilot_dataset_preflight', 'passed': True, 'errors': [],
            'producer_exit_code': 0, 'manifest': manifest_ref, 'dataset_proof': proof}}
    data['_meta']['metrics_sha256'] = holdout.canonical_hash(data['metrics'])
    result_ref = write(root / holdout.QUALIFICATION_FILE, data)
    result_ref['file'] = holdout.QUALIFICATION_FILE
    return data, result_ref, manifest, policy


class HoldoutFreezeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.data, self.reference, self.manifest, policy = synthetic_qualification(self.root)
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        # Original auditor implementation is an external boundary: fixtures are
        # byte-backed metadata but never claimed to be real run_logged runs.
        self.stack.enter_context(patch.object(holdout, 'audited_result', side_effect=self.audit_fixture))
        self.stack.enter_context(patch.object(dataset, 'FALLBACK52_POLICY', policy))

    def audit_fixture(self, path):
        path = Path(path).resolve()
        body = path.read_bytes(); data = json.loads(body)
        return data, {'file': path.relative_to(self.root).as_posix(), 'sha256': hashlib.sha256(body).hexdigest(),
                      'metrics_sha256': holdout.canonical_hash(data['metrics']), 'exit_code': data['_meta']['exit_code'],
                      'audit': {'fail': []}}

    def change_qualification(self, mutator):
        data = copy.deepcopy(self.data); mutator(data)
        written = write(self.root / holdout.QUALIFICATION_FILE, data)
        return dict(written, file=holdout.QUALIFICATION_FILE)

    def freeze(self):
        return holdout.freeze(self.reference, root=self.root)

    def receipt(self, frozen, reference):
        command = ['python3', '-m', 'src.tools.train_holdout_freeze', '--name', holdout.NAME,
                   '--qualification-result', json.dumps(self.reference)]
        data = {'_meta': {'name': holdout.NAME, 'command': command, 'end': 'ended', 'exit_code': 0},
            'config': {'name': holdout.NAME, 'version': holdout.VERSION, 'qualification_result': self.reference},
            'metrics': {'kind': 'train_holdout_freeze', 'passed': True, 'producer_exit_code': 0, 'errors': [],
                'frozen_holdout': reference, 'counts_by_partition': frozen['counts_by_partition']},
            'sources': [frozen['population']['qualification_result']]}
        return write(self.root / 'results' / (holdout.NAME + '.json'), data)

    def test_actual_completed_metadata_freezes_exact_original_split_once(self):
        frozen, reference, source = self.freeze()
        self.assertEqual(frozen['split'], dataset.split_entries(self.manifest['entries'], holdout.POLICY))
        self.assertEqual(frozen['population']['dataset_keys_sha256'],
                         holdout.canonical_hash(sorted(row['dataset_key'] for row in self.manifest['entries'])))
        self.assertEqual(frozen['missing_room_types'], [])
        self.assertEqual(len(frozen['selected_scenes']), 4)
        self.assertEqual(sum(sum(x.values()) for x in frozen['counts_by_partition'].values()), 83)
        self.assertEqual((self.root / holdout.DESTINATION).stat().st_mode & 0o222, 0)
        before = (self.root / holdout.DESTINATION).read_bytes()
        with self.assertRaisesRegex(ValueError, 'never recompute'):
            self.freeze()
        self.assertEqual((self.root / holdout.DESTINATION).read_bytes(), before)
        self.assertEqual(source['file'], self.reference['file'])
        self.receipt(frozen, reference)
        self.assertEqual(holdout.load_frozen_holdout(reference, root=self.root)[0], frozen)

    def test_failed_pending_or_fallback_cannot_freeze_even_if_manifest_exists(self):
        mutations = [lambda d: d['_meta'].update(exit_code=1), lambda d: d['_meta'].pop('end'),
            lambda d: d['metrics'].update(passed=False), lambda d: d['metrics'].update(errors=['failure']),
            lambda d: d['config'].update(deadline_fallback_52=True),
            lambda d: d['_meta']['command'].extend(['--deadline-fallback-52']),
            lambda d: d['_meta']['command'].__setitem__(3, 'src.tools.record_command')]
        for change in mutations:
            with self.subTest(change=change):
                ref = self.change_qualification(change)
                with self.assertRaises((ValueError, AssertionError)):
                    holdout.freeze(ref, root=self.root)
                self.assertFalse((self.root / holdout.DESTINATION).exists())

    def test_manifest_source_population_and_original_split_tampering_rejected(self):
        changes = [lambda d: d['metrics']['manifest'].update(sha256='0' * 64),
            lambda d: d['metrics']['dataset_proof'].update(cache_authorized=False),
            lambda d: d['metrics']['dataset_proof']['formal_cache_batches'][0]['dataset_keys'].pop(),
            lambda d: d['metrics']['dataset_proof']['formal_cache_batches'][0]['source'].update(metrics_sha256='0' * 64),
            lambda d: d['metrics']['dataset_proof']['split']['selected_scenes'].pop(),
            lambda d: d['config']['cache_source'][3].update(file='results/other_cache.json')]
        for change in changes:
            with self.subTest(change=change):
                ref = self.change_qualification(change)
                with self.assertRaises(ValueError): holdout.freeze(ref, root=self.root)
                self.assertFalse((self.root / holdout.DESTINATION).exists())

    def test_hash_and_original_audit_failures_propagate(self):
        with self.assertRaisesRegex(ValueError, 'SHA changed'):
            holdout.freeze(dict(self.reference, sha256='0' * 64), root=self.root)
        with patch.object(holdout, 'audited_result', side_effect=ValueError('Original audit failure')):
            with self.assertRaisesRegex(ValueError, 'Original audit failure'):
                self.freeze()

    def test_self_consistent_file_without_successful_freeze_receipt_is_not_authorized(self):
        frozen, reference, _ = self.freeze()
        with self.assertRaises(FileNotFoundError): holdout.load_frozen_holdout(reference, root=self.root)
        receipt_ref = self.receipt(frozen, reference)
        for change in (lambda d: d['_meta'].update(exit_code=1),
                       lambda d: d['metrics'].update(passed=False),
                       lambda d: d['metrics']['frozen_holdout'].update(sha256='0' * 64),
                       lambda d: d['_meta']['command'].__setitem__(2, 'src.tools.record_command')):
            original = json.loads(Path(receipt_ref['file']).read_text())
            bad = copy.deepcopy(original); change(bad); write(Path(receipt_ref['file']), bad)
            with self.assertRaises(ValueError): holdout.load_frozen_holdout(reference, root=self.root)
            write(Path(receipt_ref['file']), original)

    def test_pilot_requires_original_population_and_every_scene_assignment(self):
        frozen, _, _ = self.freeze()
        holdout.assert_pilot_split(list(reversed(self.manifest['entries'])), self.manifest['split'], frozen)
        changed = copy.deepcopy(self.manifest['entries']); changed[0]['scene'] = 'FloorPlan20'
        with self.assertRaisesRegex(ValueError, 'Pilot population'):
            holdout.assert_pilot_split(changed, self.manifest['split'], frozen)
        split = copy.deepcopy(self.manifest['split'])
        split['train_keys'].append(split['eval_keys'].pop())
        with self.assertRaisesRegex(ValueError, 'scene split'):
            holdout.assert_pilot_split(self.manifest['entries'], split, frozen)

    def test_expansion_never_reselects_scenes_or_calls_original_selector(self):
        frozen, _, _ = self.freeze()
        held = frozen['selected_scenes'][0]
        expanded = copy.deepcopy(self.manifest['entries'])
        for i, scene in enumerate((held, 'FloorPlan20', 'FloorPlan420')):
            expanded.append({'dataset_key': 'new:' + str(i), 'proposal_identity': hashlib.sha256(('new' + str(i)).encode()).hexdigest(),
                             'scene': scene, 'kind': 'look_stay'})
        before = copy.deepcopy(frozen)
        with patch.object(dataset, 'split_entries', side_effect=AssertionError('Must not reselect')):
            train, evaluation = holdout.partition_entries(expanded, frozen)
            self.assertIn('new:0', {r['dataset_key'] for r in evaluation})
            self.assertTrue({'new:1', 'new:2'} <= {r['dataset_key'] for r in train})
            self.assertTrue(holdout.is_holdout_scene(held, frozen))
        self.assertEqual(frozen, before)

    def test_absent_room_is_reported_without_supplementary_selection(self):
        manifest = copy.deepcopy(self.manifest)
        manifest['entries'] = [row for row in manifest['entries'] if holdout.room_type(row['scene']) != 'bathroom']
        manifest['split'] = dataset.split_entries(manifest['entries'], holdout.POLICY)
        frozen = holdout.frozen_document(manifest, self.reference, {'file': holdout.MANIFEST_FILE, 'sha256': 'a' * 64})
        holdout.validate_frozen(frozen)
        self.assertEqual(frozen['missing_room_types'], ['bathroom'])
        self.assertEqual(len(frozen['selected_scenes']), 3)
        entry = {'dataset_key': 'futurebath', 'proposal_identity': 'f' * 64, 'scene': 'FloorPlan401', 'kind': 'look_stay'}
        train, evaluation = holdout.partition_entries([entry], frozen)
        self.assertEqual(train, [entry]); self.assertEqual(evaluation, [])

    def test_duplicate_test_scene_and_edited_population_hash_rejected(self):
        frozen, _, _ = self.freeze()
        for change in (lambda d: d['population']['entries'].append(d['population']['entries'][0]),
                       lambda d: d['population']['entries'][0].update(scene='FloorPlan221'),
                       lambda d: d['population'].update(dataset_keys_sha256='0' * 64),
                       lambda d: d.update(seed=False)):
            bad = copy.deepcopy(frozen); change(bad)
            with self.assertRaises(ValueError): holdout.validate_frozen(bad)

    def test_only_fixed_repository_destination_is_loadable(self):
        frozen, reference, _ = self.freeze(); self.receipt(frozen, reference)
        elsewhere = write(self.root / 'data/world_pairs/not_frozen.json', frozen)
        with self.assertRaisesRegex(ValueError, 'fixed artifact path'):
            holdout.load_frozen_holdout(elsewhere, root=self.root)


if __name__ == '__main__':
    unittest.main()

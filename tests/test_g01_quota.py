"""Static-ticket CPU fixtures; native source science is an explicit mock boundary.

Use unittest discover -s tests. These fixtures do not certify a real inventory,
T02 pass, Stay filter or balance report. Test publication gates are real; the
full source loader is an explicit tested call boundary rather than fake science.
"""
import copy
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_pair_v2_g01 import record as mixed_record
from src.envs import pair_v2_g01_policy as policy
from src.envs import pair_v2_g01_search as search
from src.envs import pair_v2_g01_shard as shard
from src.envs import pair_v2_stay_order as stay
from src.tools import g01_quota_plan as quota
from src.tools import g01_collect_parallel as collect
from src.tools import g01_parallel as parent
from src.tools import t01v2_g01_review_checkpoint as review


def counts(**kwargs):
    return {kind: kwargs.get(kind, 0) for kind in quota.KINDS}


def ref(path):
    return {'file': str(path), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n')
    return ref(path)


def plan(partition='train', scenes=None, caps=None):
    scenes = scenes or (['FloorPlan1', 'FloorPlan7'] if partition == 'train' else ['FloorPlan21', 'FloorPlan22'])
    caps = caps or counts(look_flip=1, look_stay=2)
    return {'version': quota.VERSION, 'partition': partition, 'parent_run_name': f'gen_{partition}_quota_fixture',
        'count_stage': quota.STAGES[partition],
        'targets': {kind: 60 if partition == 'test' else 0 if kind.startswith('direct_') else 40 for kind in quota.KINDS},
        'confirmed_by_kind': counts(), 'confirmed_by_scene': {},
        'count_basis_sources': [{'file': 'logs/review/fixture/count_cutoff.json', 'sha256': 'a' * 64}],
        'outstanding_reservations': [],
        'assignments': [{'scene': scene, 'actual_output_caps': dict(caps),
                         'allowed_kinds': [kind for kind in quota.KINDS if caps[kind] > 0]} for scene in scenes]}


def configuration(root, *, partition='train', caps=None):
    value = plan(partition, caps=caps)
    reference = write(root / 'configs/quota.json', value)
    scene = value['assignments'][0]['scene']
    name = value['parent_run_name'] + '_c6_' + scene.lower()
    ticket = quota.assignment(value, reference, scene, name)
    fields = {**policy.declared(scene, partition, ticket), **stay.fields(), 'producer_module': policy.PRODUCER}
    cfg = {**fields, 'name': name, 'scene': scene, 'raw_evidence_dir': str(root / 'raw'),
           'deadline': 10000000000, 'enable_candidate_clone': False,
           'scene_pairs_per_class': policy.class_cap(partition)}
    cfg['shard'] = {**copy.deepcopy(fields), 'name': name, 'scene': scene}
    return value, reference, cfg


def row(config, *, pool='look', mode='stay', actual='look_stay', status='passed'):
    return {**shard.tags(config), 'counts_as_pair_attempt': True, 'proposed_pool': pool,
            'mode': mode, 'kind': actual, 'status': status, 'discard_reasons': []}


class PlanTests(unittest.TestCase):
    def test_train_stage_and_pending_reservations_are_not_confirmed_passes(self):
        value = plan()
        value['confirmed_by_scene'] = {'FloorPlan203': counts(look_flip=8)}
        value['confirmed_by_kind'] = counts(look_flip=8)
        pending = {'reservation_id': 'already_running_legacy',
            'source': {'file': 'logs/pbs/legacy/payload.json', 'sha256': 'b' * 64},
            'scope': 'submitted_job_upper_bound',
            'by_scene': {'FloorPlan204': counts(look_flip=8)}, 'by_kind': counts(look_flip=8)}
        value['outstanding_reservations'] = [pending]
        measured = quota.validate_structure(value)
        self.assertEqual(measured['confirmed_by_kind']['look_flip'], 8)
        self.assertEqual(measured['reserved_upper_bound_by_kind']['look_flip'], 8)
        self.assertEqual(measured['remaining_before_new_tickets']['look_flip'], 24)
        self.assertFalse(measured['count_basis_authenticated'])
        wrong = copy.deepcopy(value); wrong['count_stage'] = quota.STAGES['test']
        with self.assertRaises(ValueError): quota.validate_structure(wrong)
        wrong = copy.deepcopy(value); wrong['confirmed_by_kind']['look_flip'] = 41
        with self.assertRaises(ValueError): quota.validate_structure(wrong)

    def test_global_and_aggregate_scene_caps_cannot_be_granted_to_each_worker(self):
        value = plan('test', caps=counts(look_flip=3))
        value['confirmed_by_scene'] = {f'FloorPlan{number}': counts(look_flip=6) for number in range(23, 32)
                                       if number <= 30}
        value['confirmed_by_scene']['FloorPlan221'] = counts(look_flip=6)
        value['confirmed_by_kind'] = counts(look_flip=54)
        measured = quota.validate_structure(value)
        self.assertEqual(measured['new_tickets_by_kind']['look_flip'], 6)
        value['assignments'][1]['actual_output_caps']['look_flip'] = 4
        with self.assertRaisesRegex(ValueError, 'global'): quota.validate_structure(value)
        value = plan('test', caps=counts(look_flip=3))
        value['confirmed_by_scene'] = {'FloorPlan21': counts(look_flip=4)}
        value['confirmed_by_kind'] = counts(look_flip=4)
        with self.assertRaisesRegex(ValueError, 'scene ticket'): quota.validate_structure(value)

    def test_legacy_overreservations_do_not_become_confirmed_and_block_new_tickets(self):
        value = plan('test', caps=counts(look_flip=1))
        pending = {f'FloorPlan{i}': counts(look_flip=6) for i in range(21, 31)}
        pending['FloorPlan221'] = counts(look_flip=6)
        value['outstanding_reservations'] = [{'reservation_id': 'legacy_queued_and_running',
            'source': {'file': 'logs/review/old_reservations.json', 'sha256': 'c' * 64},
            'scope': 'submitted_job_upper_bound', 'by_scene': pending,
            'by_kind': quota.summed(pending.values())}]
        with self.assertRaises(ValueError): quota.validate_structure(value)
        self.assertEqual(value['confirmed_by_kind'], counts())

    def test_no_train_direct_wrong_split_duplicate_or_partial_allowed_domain(self):
        for edit in ('direct', 'split', 'duplicate', 'allowed', 'boolean', 'stage'):
            value = plan()
            if edit == 'direct':
                value['assignments'][0]['actual_output_caps']['direct_stay'] = 1
                value['assignments'][0]['allowed_kinds'].insert(0, 'direct_stay')
            elif edit == 'split': value['assignments'][0]['scene'] = 'FloorPlan21'
            elif edit == 'duplicate': value['assignments'][1]['scene'] = value['assignments'][0]['scene']
            elif edit == 'allowed': value['assignments'][0]['allowed_kinds'] = ['look_flip']
            elif edit == 'boolean': value['assignments'][0]['actual_output_caps']['look_flip'] = True
            else: value['count_stage'] = 'producer_raw_pass_count'
            with self.subTest(edit=edit), self.assertRaises(ValueError): quota.validate_structure(value)

    def test_real_dispatch_fails_closed_without_actual_count_basis_adapter(self):
        with tempfile.TemporaryDirectory() as text:
            root = Path(text); value, reference, config = configuration(root)
            with patch.object(quota, 'ROOT', root):
                with self.assertRaisesRegex(ValueError, 'count-basis adapter is not implemented'):
                    quota.load_plan(reference)
                with self.assertRaisesRegex(ValueError, 'count-basis adapter is not implemented'):
                    quota.validate_assignment(config, authenticate=True)
                changed = {**reference, 'sha256': '0' * 64}
                with self.assertRaisesRegex(ValueError, 'bytes changed'): quota.load_plan(changed)


class TestCountBasisTests(unittest.TestCase):
    def fixture(self, *, repeated=False):
        value = plan('test', scenes=['FloorPlan22'], caps=counts(look_stay=1))
        value['confirmed_by_kind'] = counts(look_flip=1)
        value['confirmed_by_scene'] = {'FloorPlan21': counts(look_flip=1)}
        containers, packages = {}, {}
        for index in range(2 if repeated else 1):
            name = f't01v2_checkpoint_test_fixture_{index}'
            path = 'results/' + name + '.json'
            published = {'file': path, 'sha256': str(index + 1) * 64, 'audit': {'warn': []}}
            pair = {'file': f'data/world_pairs/v2/source{index}/floorplan21_00001/pair.json', 'sha256': 'b' * 64}
            entry = {'pair_id': 'floorplan21_00001', 'scene': 'FloorPlan21', 'kind': 'look_flip',
                     'proposal_identity': 'c' * 64, 'pair_reference': pair}
            container = {'_meta': {'name': name, 'exit_code': 0, 'end': 'fixture ended',
                'command': ['python3', '-m', 'src.tools.t01v2_g01_review_checkpoint', '--name', name]},
                'config': {'name': name}, 'metrics': {'kind': 't01v2_review_checkpoint',
                    'status': 'checkpoint_ready_for_human_review', 'errors': [], 'final_pilot_acceptance': False,
                    'candidate_pairs': [entry], 'published_pair_count': 1, 'published_by_class': counts(look_flip=1)}}
            containers[path] = container, published
            packages[pair['file']] = {'kind': 'look_flip', 'scene': 'FloorPlan21',
                'target_names': ['A', 'B'], 'source_config': {'g01_partition': 'test'},
                'independent_verification': {'proposal_identity': entry['proposal_identity']},
                'source_provenance': {'pair': pair, 'scene': {'file': 'results/scene.json', 'sha256': 'd' * 64}}}
        value['count_basis_sources'] = [{key: ref[key] for key in ('file', 'sha256')}
                                        for _, ref in containers.values()]
        return value, containers, packages

    def test_successful_publication_rechecks_every_entry_and_deduplicates_proposals(self):
        from src.envs import pair_table_source as source
        value, containers, packages = self.fixture(repeated=True)
        def audited(path): return containers['results/' + Path(path).name]
        def loaded(path, pair_id, *, pair_file):
            self.assertEqual(pair_id, 'floorplan21_00001')
            return copy.deepcopy(packages[pair_file])
        # The real _container publication/name/producer gate runs. Byte audit and
        # full scientific loader remain explicit boundaries, never fake physics.
        with patch.object(source, 'audited_result', side_effect=audited), \
             patch.object(source, 'load_reviewed_pair', side_effect=loaded) as loader:
            proof = quota.authenticate_count_basis(value)
        self.assertEqual(loader.call_count, 2)
        self.assertEqual(proof['confirmed_by_kind'], counts(look_flip=1))
        self.assertEqual(len(proof['unique_proposals']), 1)
        self.assertEqual(len(proof['duplicate_occurrences']), 1)
        self.assertNotIn('training_authorized', proof)

    def test_failed_or_partial_publication_and_changed_original_source_are_not_counts(self):
        from src.envs import pair_table_source as source
        for failure in ('exit1', 'unpublished', 'wrong_producer', 'changed_sha', 'raw_count',
                        'train_pair', 'pair_reference', 'loader_failure', 'reservation'):
            value, containers, packages = self.fixture()
            container, published = next(iter(containers.values()))
            package = next(iter(packages.values()))
            if failure == 'exit1': container['_meta']['exit_code'] = 1
            elif failure == 'unpublished': container['metrics']['status'] = 'invalid_evidence'
            elif failure == 'wrong_producer': container['_meta']['command'][2] = 'src.tools.t01v2_review_checkpoint'
            elif failure == 'changed_sha': published['sha256'] = 'e' * 64
            elif failure == 'raw_count': value['confirmed_by_kind'] = counts(look_flip=17)
            elif failure == 'train_pair': package['source_config']['g01_partition'] = 'train'
            elif failure == 'pair_reference': package['source_provenance']['pair'] = {'file': 'other', 'sha256': 'f' * 64}
            elif failure == 'reservation': value['outstanding_reservations'] = [{'source': 'unimplemented'}]
            error = ValueError('original independent science failure') if failure == 'loader_failure' else None
            with self.subTest(failure=failure), \
                 patch.object(source, 'audited_result', return_value=(container, published)), \
                 patch.object(source, 'load_reviewed_pair', return_value=package, side_effect=error):
                with self.assertRaises(ValueError): quota.authenticate_count_basis(value)

    def test_conflicting_duplicate_measurement_is_rejected_not_selected(self):
        from src.envs import pair_table_source as source
        value, containers, packages = self.fixture(repeated=True)
        list(packages.values())[1]['target_names'] = ['B', 'A']
        with patch.object(source, 'audited_result', side_effect=lambda p: containers['results/' + Path(p).name]), \
             patch.object(source, 'load_reviewed_pair', side_effect=lambda p, i, pair_file: packages[pair_file]):
            with self.assertRaisesRegex(ValueError, 'conflicting'): quota.authenticate_count_basis(value)

    def test_runtime_assignment_does_not_repeat_historical_science_but_checker_does(self):
        with tempfile.TemporaryDirectory() as text:
            root = Path(text); _, reference, config = configuration(root)
            with patch.object(quota, 'ROOT', root), \
                 patch.object(quota, 'authenticate_count_basis', side_effect=ValueError('count authentication required')) as auth:
                quota.validate_assignment(config)
                auth.assert_not_called()
                with self.assertRaisesRegex(ValueError, 'count authentication required'):
                    quota.validate_assignment(config, authenticate=True)
                auth.assert_called_once()



class RuntimeTests(unittest.TestCase):
    def test_stay_only_never_spends_disabled_flip_or_direct_proposals(self):
        q = search.SceneQuota(8, actual_output_caps=counts(look_stay=2), allowed_kinds=['look_stay'])
        catalog = mixed_record(); catalog.update(search.fields(True))
        iterator = search.MixedIterator([catalog], q)
        for index in range(2):
            proposal, _, _ = next(iterator)
            self.assertEqual((proposal['projection_pool']['pool'], proposal['mode']), ('look', 'stay'))
            q.record(str(index), {'counts_as_pair_attempt': True, 'proposed_pool': 'look', 'mode': 'stay',
                                 'kind': 'look_stay', 'status': 'passed'})
        self.assertTrue(q.complete())
        with self.assertRaises(StopIteration): next(iterator)
        self.assertEqual(q.attempts['look_flip'], 0)
        self.assertEqual(q.passed['direct_stay'], 0)

    def test_filled_flip_switches_to_stay_and_class_crossover_keeps_its_real_verdict(self):
        q = search.SceneQuota(8, actual_output_caps=counts(look_flip=1, look_stay=2),
                              allowed_kinds=['look_flip', 'look_stay'])
        q.record('flip', {'counts_as_pair_attempt': True, 'proposed_pool': 'look', 'mode': 'flip',
                         'kind': 'look_flip', 'status': 'passed'})
        self.assertEqual(q.desired_modes('look', 'flip'), ['stay'])
        with self.assertRaises(search.CollectionQuotaFull): q.before_export({'kind': 'direct_stay'})
        with tempfile.TemporaryDirectory() as text:
            _, _, config = configuration(Path(text), caps=counts(look_stay=2))
            measured = {'kind': 'direct_stay', 'validity': {'acceptance_passed': True}}
            def original(*args, **kwargs):
                kwargs['attempt_row'].update(measured)
                raise search.CollectionQuotaFull('direct_stay')
            # Physical/render scientific implementation is not mocked as a pass:
            # this fixture isolates the existing export-quota exception boundary.
            with patch.object(shard, '_ORIGINAL_BUILD', side_effect=original), \
                 patch.object(shard.labels, 'pair_context'):
                out = shard.measured_build(None, config, {}, {'candidate_names': ['A', 'B']},
                    {'mode': 'stay', 'projection_pool': {'pool': 'look'}, '_private_pool_record': {
                     'points': {}, 'stay_points_by_pool': {'look': {}}}}, {}, 'floorplan1_00001', None, None)
            self.assertEqual(out['kind'], 'direct_stay')
            self.assertTrue(out['validity']['acceptance_passed'])
            self.assertEqual(out['discard_reasons'], ['collection_quota_full'])
            self.assertEqual(out['collection_quota_skip']['actual_output_cap'], 0)
            self.assertNotIn('pair_file', out)

    def test_policy_and_spawn_context_propagate_exact_plan_and_restore_hooks(self):
        with tempfile.TemporaryDirectory() as text:
            root = Path(text); value, reference, config = configuration(root)
            with patch.object(quota, 'ROOT', root), patch.object(quota, 'authenticate_count_basis', return_value={'fixture_boundary': True}):
                policy.validate_source(config, config)
                before = shard.marked.tags
                with shard.installed(config) as q:
                    self.assertEqual(shard.marked.tags()['g01_quota_assignment'], config['g01_quota_assignment'])
                    self.assertEqual(q.limits, config['actual_output_caps'])
                    self.assertEqual(q.allowed, config['allowed_kinds'])
                    raw = row(config)
                    compact = shard.compact_attempt(raw)
                    self.assertEqual(compact['g01_quota_assignment'], config['g01_quota_assignment'])
                self.assertIs(shard.marked.tags, before)
                broken = copy.deepcopy(config); broken['shard']['actual_output_caps']['look_stay'] = 8
                with self.assertRaises(ValueError): policy.validate_source(broken)
                broken = copy.deepcopy(config); broken['g01_version'] = policy.VERSION
                with self.assertRaises(ValueError): policy.validate_source(broken)

    def test_real_generator_and_replay_entry_restore_ticket_hooks_and_labels(self):
        from src.tools import g01_admission
        from contextlib import contextmanager
        class Bridge:
            def report(self): return {'fixture_resource_boundary': True}
        @contextmanager
        def bridge(config):
            yield Bridge()
        with tempfile.TemporaryDirectory() as text:
            root = Path(text); _, _, config = configuration(root)
            before = shard.marked.tags, shard.marked.generator_worker, shard.labels._CONTEXT.get()
            def generation(cfg, connection):
                self.assertEqual(shard.marked.tags()['actual_output_caps'], cfg['actual_output_caps'])
                self.assertEqual(shard.marked.tags()['g01_search_version'], search.QUOTA_VERSION)
                return {'original_generator_boundary': True}
            with patch.object(quota, 'ROOT', root), patch.object(g01_admission, 'witness_bridge', side_effect=bridge), \
                 patch.object(shard.stay, '_ORIGINAL_GENERATOR', side_effect=generation):
                self.assertTrue(shard.generator_worker(config, None)['original_generator_boundary'])
            self.assertEqual((shard.marked.tags, shard.marked.generator_worker, shard.labels._CONTEXT.get()), before)
            config.update(candidate_policy=shard.marked.candidates.CANDIDATE_POLICY,
                snapshot={'clone_context': None}, candidate_names=['A', 'B'], enable_candidate_clone=True,
                **shard.labels.pair_fields('floorplan1_00001', ['A', 'B']))
            def replay(cfg, connection):
                self.assertFalse(cfg['enable_candidate_clone'])
                self.assertEqual(cfg['g01_quota_assignment'], config['g01_quota_assignment'])
                self.assertEqual(shard.labels._CONTEXT.get()[0], 'floorplan1_00001')
                return {'original_replay_boundary': True}
            with patch.object(quota, 'ROOT', root), patch.object(shard.look, '_ORIGINAL_REPLAY', side_effect=replay):
                actual = shard.replay_worker(config, None)
                self.assertEqual(actual['actual_output_caps'], config['actual_output_caps'])
            self.assertEqual(shard.labels._CONTEXT.get(), before[2])
            with patch.object(quota, 'ROOT', root), \
                 patch.object(shard.look, '_ORIGINAL_REPLAY', side_effect=ValueError('original witness failure')):
                with self.assertRaisesRegex(ValueError, 'original witness failure'):
                    shard.replay_worker(config, None)
            self.assertEqual(shard.labels._CONTEXT.get(), before[2])


class IndependentJournalTests(unittest.TestCase):
    def fixture(self, root, *, bad=None):
        _, _, cfg = configuration(root, caps=counts(look_stay=2))
        raw = row(cfg)
        if bad == 'disabled_proposal': raw.update(mode='flip')
        if bad == 'raw_assignment': raw['actual_output_caps']['look_stay'] = 3
        if bad == 'forbidden_output': raw['kind'] = 'direct_stay'
        if bad == 'valid_skip':
            raw.update(kind='direct_stay', status='rejected', validity={'acceptance_passed': True},
                       discard_reasons=['collection_quota_full'], collection_quota_skip=shard.quota_skip(cfg, 'direct_stay'))
        raw_ref = write(root / 'raw/attempt.json', raw)
        journal_path = root / 'data/world_pairs/v2/fixture/journal.jsonl'
        journal_path.parent.mkdir(parents=True)
        journal_path.write_text(json.dumps({'raw_attempt': raw_ref, 'proposed_pool': 'look'}) + '\n')
        q = shard.make_quota(cfg)
        if bad not in ('disabled_proposal', 'forbidden_output'):
            q.record(root / 'raw/attempt.json', raw)
        metrics = {'g01_collection': q.summary(), 'passed_by_class': q.passed}
        if bad == 'reported_count': metrics['passed_by_class'] = counts(look_stay=2)
        return {'config': cfg, 'metrics': metrics}, ref(journal_path)

    def test_original_journal_is_replayed_and_actual_quotas_not_saved_flags_are_checked(self):
        for bad in (None, 'valid_skip', 'disabled_proposal', 'forbidden_output', 'raw_assignment', 'reported_count'):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as text:
                root = Path(text); data, journal = self.fixture(root, bad=bad)
                with patch.object(quota, 'ROOT', root), patch.object(review.merge, 'ROOT', root), \
                     patch.object(quota, 'authenticate_count_basis', return_value={'fixture_boundary': True}), \
                     patch.object(review, 'verified_passed_rows', return_value=({'eligible': True}, journal, [])):
                    if bad in (None, 'valid_skip'):
                        self.assertEqual(review.verify_collection(data), data['metrics']['g01_collection'])
                    else:
                        with self.assertRaises((ValueError, search.CollectionQuotaFull)):
                            review.verify_collection(data)


class QuotaSourceContextTests(unittest.TestCase):
    def test_new_policy_reaches_original_geometry_and_science_without_old_version_spoof(self):
        from test_g01_geometry_checkpoint import evidence_fixture, source_fixture
        with tempfile.TemporaryDirectory() as text:
            f = evidence_fixture(text)
            value = plan(scenes=[f.scene], caps=counts(look_flip=1))
            reference = write(f.root / 'configs/quota.json', value)
            name = value['parent_run_name'] + '_c6_' + f.scene.lower()
            fields = policy.declared(f.scene, 'train', quota.assignment(value, reference, f.scene, name))
            cfg = f.data['config']; cfg.update(fields, name=name)
            cfg['shard'].update(fields, name=name)
            f.data['metrics'].update(fields, name=name)
            f.row.update(fields)
            catalog = copy.deepcopy(f.read(f.pool_ref)); catalog.update(search.fields(True))
            f.pool_ref = f.write(Path(f.pool_ref['file']), catalog)
            f.row['projection_pool_reference'] = f.pool_ref
            f.refresh_index(); f.association['controller_index_reference'] = f.index
            positions = {obj['name']: obj['position'] for obj in f.baseline['objects']}
            def science(row, path, **kwargs):
                self.assertEqual(review.roles.verify_proposal_role(row, positions)['g01_search_version'], search.QUOTA_VERSION)
                return {'original_scientific_boundary': True}
            with patch.object(quota, 'ROOT', f.root):
                with source_fixture(f, science):
                    checked = review.merge.check_pair(f.row, f.pair_path, raw_root=cfg['raw_evidence_dir'])
                    self.assertTrue(checked['original_scientific_boundary'])
                    f.row['actual_output_caps'] = counts(look_flip=2)
                    with self.assertRaisesRegex(ValueError, 'static quota assignment'):
                        review.merge.check_pair(f.row, f.pair_path, raw_root=cfg['raw_evidence_dir'])


class ParentWiringTests(unittest.TestCase):
    def test_exact_submission_command_binds_quota_manifest_and_old_default_is_unchanged(self):
        with tempfile.TemporaryDirectory() as text:
            root = Path(text); value = plan(); reference = write(root / 'configs/quota.json', value)
            name, scenes, concurrency = value['parent_run_name'], [r['scene'] for r in value['assignments']], 6
            timeout = collect.required_timeout(scenes, concurrency)
            command = ['python3', '-m', collect.MODULE, '--name', name, '--scenes', *scenes,
                '--concurrency', '6', '--benchmark-source', 'results/benchmark.json', '--timeout-seconds', str(timeout),
                '--partition', 'train', '--quota-plan', reference['file'], '--quota-plan-sha256', reference['sha256']]
            base = root / 'logs/pbs/submission'; base.mkdir(parents=True)
            payload = {'name': name, 'command': command, 'budget_group': 'G01', 'task_id': 'T01',
                'budget_batch': 't00b_t01_20261108', 'queue': 'autox', 'requested_host': None,
                'requested_gpu_count': 1, 'requested_wall_seconds': timeout + 30}
            write(base / 'payload.json', payload)
            write(base / 'submission.json', {'stdout': '700001.hopper-m-02\n'})
            kwargs = dict(scenes=scenes, concurrency=concurrency, benchmark='results/benchmark.json',
                          partition='train', explicit_partition=True, quota_reference=reference)
            collect.submission_identity(root, name, '700001.hopper-m-02', timeout, **kwargs)
            for changed in ({**reference, 'sha256': 'f' * 64}, None):
                with self.assertRaises(ValueError):
                    collect.submission_identity(root, name, '700001.hopper-m-02', timeout,
                                                **{**kwargs, 'quota_reference': changed})

    def test_parent_injects_only_its_scene_tickets_and_restores_after_exception(self):
        with tempfile.TemporaryDirectory() as text:
            root = Path(text); value = plan(); reference = write(root / 'configs/quota.json', value)
            scenes = [r['scene'] for r in value['assignments']]
            supplied = {'plan': value, 'reference': reference, 'arithmetic': quota.validate_structure(value),
                        'count_basis_proof': {'fixture_boundary': True}}
            before = parent.make_shard
            def base(name, scene, *args, **kwargs):
                return {'name': name, 'scene': scene, 'original_resource_proof': 'unchanged',
                        **policy.declared(scene)}
            with patch.object(parent, 'make_shard', side_effect=base) as original:
                with self.assertRaisesRegex(RuntimeError, 'fixture stop'):
                    with collect.installed(scenes=scenes, concurrency=6, benchmark='results/benchmark.json',
                            proof={'source': {'file': 'results/benchmark.json'}}, quota=supplied):
                        actual = parent.make_shard(value['parent_run_name'] + '_c6_floorplan1', 'FloorPlan1')
                        self.assertEqual(actual['allowed_kinds'], ['look_flip', 'look_stay'])
                        self.assertEqual(actual['actual_output_caps'], counts(look_flip=1, look_stay=2))
                        self.assertEqual(actual['original_resource_proof'], 'unchanged')
                        self.assertEqual(actual['g01_quota_assignment']['plan_reference'], reference)
                        raise RuntimeError('fixture stop')
                self.assertIs(parent.make_shard, original)
            self.assertIs(parent.make_shard, before)


if __name__ == '__main__':
    unittest.main()

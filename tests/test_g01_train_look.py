"""CPU policy/metadata tests; original science remains an explicit fixture boundary."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_g01_quota import plan as legacy_plan, counts, write, row
from test_pair_v2_g01 import record as mixed_record
from src.envs import pair_v2_g01_policy as policy
from src.envs import pair_v2_g01_search as search
from src.envs import pair_v2_g01_shard as shard
from src.envs.pair_table_identity import identity_fields
from src.tools import g01_quota_plan as quota
from src.tools import g01_train_inventory as inventory
from src.tools import g01_ticket_bank as bank
from src.tools import g01_shared_unit as shared_g01
from src.tools import gpu_shared_claim as shared
from src.tools import g01_collect_parallel as collect
from src.tools import train_holdout_freeze as holdout


def plan(scenes=None, caps=None):
    value = legacy_plan(scenes=scenes, caps=caps)
    value.update(version=quota.TRAIN_VERSION, count_basis_authority=inventory.AUTHORITY,
        holdout_reference={'file': holdout.DESTINATION, 'sha256': 'f' * 64}, construction_occupancy_by_scene={})
    return value


def rows(flip=2, stay_a=2, stay_b=2, *, scene='FloorPlan1', displacement=.15):
    result = []
    for kind, side, n in [('look_flip', 0, flip), ('look_stay', 0, stay_a), ('look_stay', 1, stay_b)]:
        for i in range(n):
            name = f'{scene}_{kind}_{side}_{i}'
            reference = {'file': 'data/fixture/' + name + '.json', 'sha256': hashlib.sha256(name.encode()).hexdigest()}
            identity = identity_fields(name, reference)
            result.append({'pair_id': name, 'original_pair_id': name, 'dataset_key': identity['dataset_key'],
                'dataset_identity': identity, 'scene': scene, 'kind': kind,
                'candidate_names': ['A', 'B'], 'target_names': ['A', 'B'] if kind.endswith('flip') else [(['A', 'B'][side])] * 2,
                's0_target_left': [True, False], 'stay_displacement': {'distance_m': displacement}, 'synthetic': False})
    return result


class TrainingLookCounts(unittest.TestCase):
    def test_holdout_then_exact_stay_cutoff_then_original_balance(self):
        original = rows() + rows(scene='FloorPlan2')
        short = rows(flip=0, stay_a=1, stay_b=1, scene='FloorPlan3', displacement=.149999)
        observed = inventory.postfilter(original + short, {'FloorPlan2'})
        self.assertEqual(observed['confirmed_by_kind'], counts(look_flip=2, look_stay=4))
        self.assertEqual(observed['confirmed_by_scene'], {'FloorPlan1': counts(look_flip=2, look_stay=4)})
        self.assertEqual(len(observed['filter_omissions']), 8)
        self.assertFalse(observed['ratio_ready'])
        self.assertEqual(observed['balance']['seed'], 0)
        self.assertEqual(inventory.postfilter(rows(flip=0, stay_a=3, stay_b=1), set())['confirmed_by_kind']['look_stay'], 2)

    def test_ratio_is_not_a_relabeling_or_downsampling_algorithm(self):
        x = inventory.postfilter(rows(flip=50, stay_a=20, stay_b=20), set())
        self.assertTrue(x['ratio_ready']); self.assertTrue(x['global_ready'])
        y = inventory.postfilter(rows(flip=51, stay_a=20, stay_b=20), set())
        self.assertEqual(y['confirmed_by_kind']['look_flip'], 51)
        self.assertFalse(y['ratio_ready']); self.assertTrue(y['needs_human_ratio_decision'])
        self.assertTrue(all(y['class_full'].values())); self.assertFalse(y['global_ready'])

    def test_missing_stage_holdout_or_source_cannot_grant_tickets(self):
        for edit in ('holdout', 'stage', 'authority', 'basis'):
            value = plan()
            if edit == 'holdout': value.pop('holdout_reference')
            elif edit == 'stage': value['count_stage'] = 'cache_only'
            elif edit == 'authority': value['count_basis_authority'] = quota.RECEIPT_AUTHORITY
            else: value['count_basis_sources'] = []
            with self.subTest(edit=edit), self.assertRaises(ValueError): quota.validate_structure(value)

    def test_original_scene_occupancy_cannot_be_released_by_postfiltering(self):
        value = plan(scenes=['FloorPlan1'], caps=counts(look_stay=1))
        value['construction_occupancy_by_scene'] = {'FloorPlan1': counts(look_stay=8)}
        with self.assertRaisesRegex(ValueError, 'scene/class capacity'): quota.validate_structure(value)

    def test_unqualified_original_inventory_is_reserved_globally_not_double_counted_per_scene(self):
        value = plan(scenes=['FloorPlan1'], caps=counts(look_stay=5))
        value['construction_occupancy_by_scene'] = {'FloorPlan1': counts(look_stay=3)}
        value['outstanding_reservations'] = [{'reservation_id': 'old', 'source': value['count_basis_sources'][0],
            'scope': 'unverified_inventory_upper_bound', 'by_kind': counts(look_stay=3),
            'by_scene': {'FloorPlan1': counts(look_stay=3)}}]
        proof = quota.validate_structure(value)
        self.assertEqual(proof['remaining_before_new_tickets']['look_stay'], 37)
        self.assertEqual(proof['new_tickets_by_kind']['look_stay'], 5)

    def test_class40_stops_that_class_even_when_other_class_is_missing(self):
        value = plan(scenes=['FloorPlan2'], caps=counts(look_stay=1))
        value['confirmed_by_kind'] = counts(look_flip=40)
        value['confirmed_by_scene'] = {'FloorPlan1': counts(look_flip=40)}
        value['construction_occupancy_by_scene'] = copy.deepcopy(value['confirmed_by_scene'])
        self.assertEqual(quota.validate_structure(value)['new_tickets_by_kind'], counts(look_stay=1))
        value['assignments'][0]['actual_output_caps']['look_flip'] = 1
        value['assignments'][0]['allowed_kinds'] = ['look_flip', 'look_stay']
        with self.assertRaisesRegex(ValueError, 'global remaining'): quota.validate_structure(value)

    def test_parent_auth_reuses_count_receipt_and_rejects_holdout_and_unknown_inventory_loss(self):
        value = plan(scenes=['FloorPlan2'])
        proof = {key: copy.deepcopy(value[key]) for key in ('holdout_reference', 'confirmed_by_kind',
            'confirmed_by_scene', 'construction_occupancy_by_scene')}
        proof.update(reserved_upper_bound_by_kind=counts(), reserved_upper_bound_by_scene={}, needs_human_ratio_decision=False)
        with patch.object(inventory, 'load_receipt', return_value=(proof, {'file': 'receipt'})), \
                patch.object(holdout, 'load_frozen_holdout', return_value=({'selected_scenes': ['FloorPlan1']}, {})):
            self.assertFalse(quota.authenticate_count_basis(value)['parent_native_revalidation_performed'])
            value['assignments'][0]['scene'] = 'FloorPlan1'
            with self.assertRaisesRegex(ValueError, 'held-out'): quota.authenticate_count_basis(value)
            value['assignments'][0]['scene'] = 'FloorPlan2'
            proof['reserved_upper_bound_by_kind'] = counts(look_flip=2)
            proof['reserved_upper_bound_by_scene'] = {'FloorPlan3': counts(look_flip=2)}
            with self.assertRaisesRegex(ValueError, 'conservative reservation'): quota.authenticate_count_basis(value)


class TrainingLookRuntime(unittest.TestCase):
    def config(self, root):
        value = plan(scenes=['FloorPlan1']); ref = write(root / 'configs/train.json', value)
        name = value['parent_run_name'] + '_c6_floorplan1'
        ticket = quota.assignment(value, ref, 'FloorPlan1', name)
        fields = {**policy.declared('FloorPlan1', 'train', ticket, nearby=True), **policy.stay.fields()}
        cfg = {**fields, 'name': name, 'scene': 'FloorPlan1', 'deadline': 10000000000,
            'enable_candidate_clone': False, 'scene_pairs_per_class': 8, 'producer_module': policy.PRODUCER}
        cfg['shard'] = copy.deepcopy(cfg)
        return value, ref, cfg

    def test_actual_new_policy_tags_and_quota_stay_priority_restore_old_hooks(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); value, ref, cfg = self.config(root)
            with patch.object(quota, 'ROOT', root):
                policy.validate_source(cfg)
                q = shard.make_quota(cfg)
                self.assertEqual(q.desired_modes('look', 'flip'), ['stay', 'flip'])
                self.assertEqual(q.desired_modes('direct', 'flip'), [])
                with shard.installed(cfg) as actual:
                    self.assertTrue(actual.training)
                    tags = shard.tags(cfg)
                    search.validate_fields(tags)
                    self.assertEqual(tags['g01_search_version'], search.TRAIN_VERSION)
                    self.assertNotIn('g01_engineering_smoke', tags)
                    self.assertEqual(tags['g01_policy']['scene_timeout_seconds'], 1200)
                self.assertFalse(shard.make_quota(None).training)
                q.record('one', row(cfg, actual='look_stay'))
                q.record('two', row(cfg, actual='look_stay'))
                self.assertEqual(q.desired_modes('look', 'stay'), ['flip'])

    def test_finite_round_prioritizes_available_stay_without_new_discovery(self):
        from types import SimpleNamespace
        q = SimpleNamespace(configuration_enabled=False, training=True)
        flip = {'stream': SimpleNamespace(quota=q, priority_choice=lambda: ('look', 'flip'))}
        stay = {'stream': SimpleNamespace(quota=q, priority_choice=lambda: ('look', 'stay'))}
        active = [flip, stay]; scheduled = search.ConfigurationPriorityRound(active)
        self.assertIs(scheduled.pop('direct'), stay)
        self.assertIs(scheduled.pop('look'), flip)
        self.assertFalse(scheduled); self.assertEqual(active, [flip, stay])

    def test_new_training_cannot_enable_direct_or_test_configuration(self):
        for kwargs in [dict(actual_output_caps=counts(direct_flip=1), allowed_kinds=['direct_flip']),
                       dict(actual_output_caps=counts(look_stay=1), allowed_kinds=['look_stay'], seen_configurations=[])]:
            with self.assertRaises(ValueError): search.SceneQuota(8, training=True, **kwargs)
        with self.assertRaises(ValueError): search.fields(False, training=True)
        with self.assertRaises(ValueError): policy.fields('test', True, training=True)

    def test_old_policy_bytes_and_count_authority_are_not_upgraded(self):
        from src.envs.pair_v2_g01_search import POLICY
        self.assertEqual(search.fields()['g01_search_policy'], POLICY)
        self.assertNotEqual(search.fields(True)['g01_search_version'], search.TRAIN_VERSION)
        with self.assertRaisesRegex(ValueError, 'legacy plans'): quota.authenticate_count_basis(legacy_plan())

    def test_one_scene_formal_smoke_reuses_complete_original_grant(self):
        self.assertEqual(collect.required_timeout(['FloorPlan1'], 6, 'train'), 1600)
        self.assertEqual(collect.runner_timeout(['FloorPlan1'], 6, 'train'), 1480)

    def test_train_bank_real_shared_identity_priority_and_disjoint_other_tickets(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            value = plan(scenes=['FloorPlan1', 'FloorPlan2'])
            base_ref = write(root / 'configs/base.json', value)
            b = {'version': bank.TRAIN_VERSION, 'base_plan_reference': base_ref, 'units': [
                {'slice_id': 's01', 'parent_run_name': 'gen_train_look_smoke', 'scenes': ['FloorPlan1']},
                {'slice_id': 's02', 'parent_run_name': 'gen_train_look_next', 'scenes': ['FloorPlan2']}]}
            bref = write(root / 'configs/bank.json', b)
            member = bank.expected_member(b, bref, 's01', root=root)
            mref = write(root / 'configs/member.json', member)
            benchmark = write(root / 'results/benchmark.json', {})
            command = ['python3', '-u', '-m', collect.MODULE, '--name', member['parent_run_name'],
                '--scenes', 'FloorPlan1', '--concurrency', '6', '--benchmark-source', benchmark['file'],
                '--timeout-seconds', '1600', '--partition', 'train', '--quota-plan', mref['file'],
                '--quota-plan-sha256', mref['sha256'], '--nearby-physics']
            unit = {'name': member['parent_run_name'], 'producer_module': collect.MODULE,
                'budget_group': 'G01', 'gpus': 1, 'cpu_cores': 12, 'priority': 250,
                'timeout_seconds': 1600, 'command': command,
                'references': {'quota_plan': mref, 'benchmark_source': benchmark,
                    'g01_ticket_bank': bref, 'g01_ticket_slice': 's01'}}
            identity, keys = shared_g01.validate_unit(unit, root)
            self.assertEqual(identity['partition'], 'train')
            self.assertEqual(member['outstanding_reservations'][0]['by_scene'],
                {'FloorPlan2': value['assignments'][1]['actual_output_caps']})
            unit.update(identity=identity, claim_keys=keys, unit_id=shared.canonical_hash(identity))
            self.assertEqual(shared.validate_unit(unit, root), (identity, keys))
            unit['priority'] = 300
            with self.assertRaisesRegex(ValueError, 'priority'): shared.validate_unit(unit, root)
            b['units'][1]['scenes'] = ['FloorPlan1']
            with self.assertRaisesRegex(ValueError, 'every original scene'): bank.validate_bank(b, root=root)


class TrainingMetadataReceiptTests(unittest.TestCase):
    def fixture(self, root, diagnostic='present', input_rows=None):
        from src.envs.pair_v2_marked_policy import stay_horizontal_displacement
        manifest = {'entries': [], 'balance': {'selected': []}, 'split': {}, 'cache_sources': []}
        published = []
        for index, item in enumerate(input_rows if input_rows is not None else rows(flip=2, stay_a=1, stay_b=1)):
            directory = root / 'data/world_pairs/v2/fixture' / str(index)
            portable = []
            worlds = [{'objects': [{'name': 'X', 'objectType': 'Book', 'position': {'x': x, 'y': 0, 'z': 0}}]}
                      for x in (0, item['stay_displacement']['distance_m'])]
            for i, world in enumerate(worlds): portable.append(write(directory / f'snapshot_w{i}.json', world))
            identity = hashlib.sha256(str(index).encode()).hexdigest()
            pair = {'pair_id': item['pair_id'], 'scene': item['scene'], 'kind': item['kind'], 'status': 'passed',
                'identities': {'candidate_names': ['A', 'B'], 'anchor_name': 'X'},
                'target_w0': item['target_names'][0], 'target_w1': item['target_names'][1],
                'portable_files': portable, 'proposal_hash': identity,
                'stay_horizontal_displacement': stay_horizontal_displacement(worlds, 'X')}
            if diagnostic == 'absent': pair.pop('stay_horizontal_displacement')
            elif diagnostic == 'mismatched': pair['stay_horizontal_displacement']['distance_m'] = .25
            reference = write(directory / 'pair.json', pair)
            dataset_identity = identity_fields(item['pair_id'], reference); key = dataset_identity['dataset_key']
            manifest['entries'].append({'dataset_key': key, 'original_pair_id': item['pair_id'],
                'proposal_identity': identity, 'scene': item['scene'], 'kind': item['kind'], 'pair_reference': reference})
            manifest['balance']['selected'].append({'dataset_key': key, 'dataset_identity': dataset_identity,
                'candidate_names': ['A', 'B'], 'original_target_names': item['target_names'],
                'original_s0_target_left': item['s0_target_left']})
            published.append({'pair_id': item['pair_id'], 'scene': item['scene'], 'kind': item['kind'],
                'pair_reference': reference, 'proposal_identity': identity,
                'independent_verification': {'proposal_identity': identity, 'validity': {'acceptance_passed': True}}})
        checkpoint_ref = {'file': 'results/independent.json', 'sha256': '1' * 64}
        container = {'_meta': {'end': 'actual-ended', 'exit_code': 0}, 'metrics': {'errors': []}}
        return manifest, published, checkpoint_ref, container

    def bind_inventory(self, stack, root, manifest, entries, cref, container):
        from src.envs import pair_table_source as sources
        from src.tools import merge_pair_v2 as merge
        qualification = {'file': 'results/qualified.json', 'sha256': '2' * 64}
        href = {'file': holdout.DESTINATION, 'sha256': '3' * 64}
        stack.enter_context(patch.object(holdout, 'authenticate_qualification', return_value=(
            manifest, {'formal_cache_batches': []}, qualification, {'file': 'manifest', 'sha256': '4' * 64})))
        stack.enter_context(patch.object(holdout, 'load_frozen_holdout', return_value=({'selected_scenes': []}, href)))
        stack.enter_context(patch.object(holdout, 'assert_pilot_split'))
        stack.enter_context(patch.object(sources, '_container', return_value=(container, cref, entries)))
        stack.enter_context(patch.object(merge, 'proposal_identity', side_effect=lambda r: r['proposal_hash']))
        return qualification, href

    def rejected_source(self, root, published, checkpoint):
        from src.tools import t02_export_parallel as parallel
        pair = json.loads(Path(published['pair_reference']['file']).read_text())
        name = 't02_export_fixture'; job = '123.hopper-m-02'; scratch = root / 'scratch'
        folder = scratch / 't02_export' / name / job / 'pair_0000'
        package = {'pair_id': pair['pair_id'], 'scene': pair['scene'], 'source_config': {'field_of_view': 60},
            'source_provenance': {'container': checkpoint, 'pair': published['pair_reference']},
            'independent_verification': {'proposal_identity': published['proposal_identity']}}
        package_ref = write(folder / 'authenticated_source.json', package)
        argv = ['python3', '-m', 'src.tools.t02_export', '--name', name, '--source', checkpoint['file'],
            '--pair-id', pair['pair_id'], '--pair-file', published['pair_reference']['file']]
        cfg = {'name': name, 'version': parallel.export.VERSION, 'pbs_jobid': job,
            'cuda_visible_devices': 'GPU-fixture', 'cpu_affinity': list(range(12)), 'node': 'fixture',
            'scratch': str(scratch), 'scratch_directory': str(folder.parent), 'gpus': 1, 'depth': 4,
            'pair_id': [pair['pair_id']], 'source': checkpoint['file'], 'pair_file': published['pair_reference']['file']}
        row = {'pair_id': pair['pair_id'], 'status': 'rejected', 'cleanup_errors': [], 'table_complete': False,
            'rejection_reasons': ['source_world_restore_failed'], 'scratch_directory': str(folder), 'source_package': package_ref,
            'error': {'type': 'ValueError', 'message': 'source_world_restore_failed'}}
        metrics = {'producer_exit_code': 0, 'kind': 't02_table_export', 'export_checks_complete': True,
            'errors': [], 'cleanup_errors': [], 'unstarted_pair_ids': [], 'black_frame_detected': False, 'pairs': [row]}
        data = {'config': cfg, '_meta': {'command': argv, 'name': name, 'end': 'ended', 'exit_code': 0}, 'metrics': metrics}
        return data, {'file': 'results/rejected.json', 'sha256': '9' * 64}

    def test_real_pair_snapshot_shas_recompute_stay_and_bind_original_population(self):
        from contextlib import ExitStack
        from src.envs import pair_table_source as sources
        from src.tools import merge_pair_v2 as merge
        from src.tools import smoke_e2e_pilot_dataset as dataset
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            root = Path(d).resolve(); manifest, entries, cref, container = self.fixture(root)
            qualification = {'file': 'results/qualified.json', 'sha256': '2' * 64}
            href = {'file': holdout.DESTINATION, 'sha256': '3' * 64}
            stack.enter_context(patch.object(holdout, 'authenticate_qualification', return_value=(
                manifest, {'formal_cache_batches': []}, qualification, {'file': 'manifest', 'sha256': '4' * 64})))
            stack.enter_context(patch.object(holdout, 'load_frozen_holdout', return_value=({'selected_scenes': ['FloorPlan2']}, href)))
            stack.enter_context(patch.object(holdout, 'assert_pilot_split'))
            stack.enter_context(patch.object(sources, '_container', return_value=(container, cref, entries)))
            stack.enter_context(patch.object(merge, 'proposal_identity', side_effect=lambda r: r['proposal_hash']))
            native = stack.enter_context(patch.object(dataset, 'authenticate_cache_batch', side_effect=AssertionError('Must not reauthenticate native here')))
            result = inventory.build_inventory(qualification, href, [cref], root=root)
            self.assertEqual(result['confirmed_by_kind'], counts(look_flip=2, look_stay=2))
            self.assertEqual(result['construction_occupancy_by_kind'], result['confirmed_by_kind'])
            self.assertFalse(result['native_scientific_revalidation_performed']); native.assert_not_called()
            self.assertTrue(result['ratio_ready'])
            # Original raw bytes are never repaired if a referenced snapshot changes.
            ref = next(r for r in result['input_metadata_references'] if r['file'].endswith('snapshot_w1.json'))
            Path(ref['file']).write_text('{}')
            with self.assertRaises(ValueError): inventory.build_inventory(qualification, href, [cref], root=root)

    def test_legacy_absent_diagnostic_uses_original_snapshot_but_conflicting_diagnostic_fails(self):
        from contextlib import ExitStack
        from src.envs import pair_table_source as sources
        from src.tools import merge_pair_v2 as merge
        for diagnostic in ('absent', 'mismatched'):
            with self.subTest(diagnostic=diagnostic), tempfile.TemporaryDirectory() as d, ExitStack() as stack:
                root = Path(d).resolve(); manifest, entries, cref, container = self.fixture(root, diagnostic)
                qualification = {'file': 'results/qualified.json', 'sha256': '2' * 64}
                href = {'file': holdout.DESTINATION, 'sha256': '3' * 64}
                stack.enter_context(patch.object(holdout, 'authenticate_qualification', return_value=(
                    manifest, {'formal_cache_batches': []}, qualification, {'file': 'manifest', 'sha256': '4' * 64})))
                stack.enter_context(patch.object(holdout, 'load_frozen_holdout', return_value=({'selected_scenes': []}, href)))
                stack.enter_context(patch.object(holdout, 'assert_pilot_split'))
                stack.enter_context(patch.object(sources, '_container', return_value=(container, cref, entries)))
                stack.enter_context(patch.object(merge, 'proposal_identity', side_effect=lambda r: r['proposal_hash']))
                if diagnostic == 'mismatched':
                    with self.assertRaisesRegex(ValueError, 'Stay displacement changed'):
                        inventory.build_inventory(qualification, href, [cref], root=root)
                else:
                    before = {e['pair_reference']['file']: Path(e['pair_reference']['file']).read_bytes() for e in entries}
                    result = inventory.build_inventory(qualification, href, [cref], root=root)
                    stays = [r for r in result['qualified_rows'] if r['kind'].endswith('stay')]
                    self.assertTrue(all(r['stay_displacement']['distance_m'] == .15 for r in stays))
                    self.assertTrue(all(r['original_stay_diagnostic_present'] is False for r in stays))
                    self.assertEqual(result['confirmed_by_kind'], counts(look_flip=2, look_stay=2))
                    self.assertEqual(before, {path: Path(path).read_bytes() for path in before})

    def test_receipt_requires_exact_command_inputs_and_audited_source_ledger(self):
        ref = {'file': 'results/count.json', 'sha256': '1' * 64}
        qualified = {'file': 'results/qualified.json', 'sha256': '2' * 64}
        freeze = {'file': 'results/train_holdout_freeze.json', 'sha256': '3' * 64}
        heldout = {'file': holdout.DESTINATION, 'sha256': '4' * 64}
        construction = {'file': 'results/construction.json', 'sha256': '5' * 64}
        argv = ['--name', 'count', '--qualification', json.dumps(qualified), '--holdout', json.dumps(heldout),
            '--construction-source', json.dumps(construction), '--timeout-seconds', '120']
        proof = {'qualification_reference': qualified, 'construction_publications': [construction], 'rejection_publications': []}
        data = {'config': {'name': 'count', 'version': inventory.VERSION, 'arguments': argv,
                    'qualification_reference': qualified, 'holdout_reference': heldout, 'construction_sources': [construction],
                    'known_rejection_sources': []},
                '_meta': {'name': 'count', 'end': 'ended', 'exit_code': 0,
                    'command': ['python3', '-m', inventory.MODULE, *argv]},
                'metrics': {'producer_exit_code': 0, 'passed': True, 'errors': [], 'kind': inventory.VERSION, 'inventory': proof},
                'sources': [qualified, construction, freeze]}
        def audited(path):
            return ({}, freeze) if Path(path).name == 'train_holdout_freeze.json' else (data, ref)
        with patch.object(inventory, 'audited_result', side_effect=audited), patch.object(inventory, 'build_inventory', return_value=proof):
            self.assertEqual(inventory.load_receipt(ref)[0], proof)
            data['sources'].pop()
            with self.assertRaisesRegex(ValueError, 'source ledger'): inventory.load_receipt(ref)
            data['sources'].append(freeze)
            data['config']['construction_sources'] = [qualified]
            with self.assertRaisesRegex(ValueError, 'complete input cutoff'): inventory.load_receipt(ref)

    def test_forty_balance_dropped_stays_do_not_deadlock_opposite_side_repair(self):
        from contextlib import ExitStack
        from src.envs import pair_table_source as sources
        from src.tools import merge_pair_v2 as merge
        # Ten scenes have four Stay-A constructions each; their unused scene
        # capacity permits four opposite-side pairs despite global raw count40.
        original_rows = [item for scene in range(1, 11) for item in rows(
            flip=0, stay_a=4, stay_b=0, scene=f'FloorPlan{scene}')]
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            root = Path(d).resolve(); manifest, entries, cref, container = self.fixture(root, input_rows=original_rows)
            qualification = {'file': 'results/qualified.json', 'sha256': '2' * 64}
            href = {'file': holdout.DESTINATION, 'sha256': '3' * 64}
            stack.enter_context(patch.object(holdout, 'authenticate_qualification', return_value=(
                manifest, {'formal_cache_batches': []}, qualification, {'file': 'manifest', 'sha256': '4' * 64})))
            stack.enter_context(patch.object(holdout, 'load_frozen_holdout', return_value=({'selected_scenes': []}, href)))
            stack.enter_context(patch.object(holdout, 'assert_pilot_split'))
            stack.enter_context(patch.object(sources, '_container', return_value=(container, cref, entries)))
            stack.enter_context(patch.object(merge, 'proposal_identity', side_effect=lambda r: r['proposal_hash']))
            observed = inventory.build_inventory(qualification, href, [cref], root=root)
            self.assertEqual(observed['cached_by_kind']['look_stay'], 40)
            self.assertEqual(observed['construction_occupancy_by_kind']['look_stay'], 40)
            self.assertEqual(observed['confirmed_by_kind']['look_stay'], 0)
            self.assertEqual(len(observed['balance']['dropped']), 40)
            self.assertEqual(observed['reserved_upper_bound_by_kind']['look_stay'], 0)
            candidate = plan(scenes=['FloorPlan1'], caps=counts(look_stay=4))
            for field in ('holdout_reference', 'confirmed_by_kind', 'confirmed_by_scene', 'construction_occupancy_by_scene'):
                candidate[field] = observed[field]
            self.assertEqual(quota.validate_structure(candidate)['new_tickets_by_kind']['look_stay'], 4)
            with patch.object(inventory, 'load_receipt', return_value=(observed, {})):
                self.assertTrue(quota.authenticate_count_basis(candidate)['count_basis_authenticated'])
            candidate['assignments'][0]['actual_output_caps']['look_stay'] = 5
            with self.assertRaisesRegex(ValueError, 'scene/class capacity'): quota.validate_structure(candidate)

    def test_complete_original_rejection_releases_pending_but_unknown_errors_do_not(self):
        from contextlib import ExitStack
        from src.tools import t02_export_parallel as parallel
        from src.tools import merge_pair_v2 as merge
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            root = Path(d).resolve(); _, published, _, _ = self.fixture(root)
            entry = published[0]; pair = json.loads(Path(entry['pair_reference']['file']).read_text())
            checkpoint = {'file': 'results/checkpoint.json', 'sha256': '8' * 64}
            data, source = self.rejected_source(root, entry, checkpoint)
            metrics = data['metrics']; row = metrics['pairs'][0]; package_ref = row['source_package']
            constructed = [{'proposal_identity': entry['proposal_identity'], 'scene': pair['scene'],
                'kind': pair['kind'], 'target_names': [pair['target_w0'], pair['target_w1']],
                'published_bindings': [{'pair_reference': entry['pair_reference'], 'checkpoint_reference': checkpoint}]}]
            stack.enter_context(patch.object(inventory, 'audited_result', return_value=(data, source)))
            stack.enter_context(patch.object(merge, 'proposal_identity', side_effect=lambda r: r['proposal_hash']))
            result, actual = inventory.known_rejections([source], constructed, root=root)
            self.assertEqual(result[0]['proposal_identity'], entry['proposal_identity']); self.assertEqual(actual, [source])
            for change in ('timeout', 'black', 'cleanup', 'unfinished', 'omitted_reason', 'native_timeout',
                           'unknown_valueerror', 'raw_error', 'cache_group_failure', 'child_command'):
                original = copy.deepcopy(data)
                if change == 'timeout': data['_meta']['exit_code'] = 1
                elif change == 'black': metrics['black_frame_detected'] = True
                elif change == 'cleanup': row['cleanup_errors'] = ['unknown owned process']
                elif change == 'unfinished': data['_meta']['end'] = None
                elif change == 'omitted_reason': row['rejection_reasons'] = []
                elif change == 'native_timeout': row['error']['type'] = 'TimeoutError'
                elif change == 'unknown_valueerror': row['error']['message'] = 'unexpected failure'
                elif change == 'raw_error': row['raw_index_error'] = 'disk quota exceeded'
                elif change == 'cache_group_failure': metrics['kind'] = 't02_formal_cache_check'
                else: data['_meta']['command'][-1] += '.different'
                with self.subTest(change=change), self.assertRaises(ValueError):
                    inventory.known_rejections([source], constructed, root=root)
                data.clear(); data.update(original); metrics = data['metrics']; row = metrics['pairs'][0]
            Path(package_ref['file']).write_text('{}')
            with self.assertRaises(ValueError): inventory.known_rejections([source], constructed, root=root)

    def test_actual_inventory_releases_only_known_rejections_and_short_stays_not_scene_occupancy(self):
        from contextlib import ExitStack
        originals = rows(flip=2, stay_a=1, stay_b=1)
        originals[2]['stay_displacement']['distance_m'] = .149999
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory).resolve()
            manifest, entries, cref, container = self.fixture(root, input_rows=originals)
            # Only the two Flip pairs have completed qualification. The Stay
            # measurements remain authentic construction metadata, not cache PASS.
            manifest['entries'] = manifest['entries'][:2]
            manifest['balance']['selected'] = manifest['balance']['selected'][:2]
            qualification, href = self.bind_inventory(stack, root, manifest, entries, cref, container)
            initial = inventory.build_inventory(qualification, href, [cref], root=root)
            self.assertEqual(initial['reserved_upper_bound_by_kind'], counts(look_stay=1))
            self.assertEqual(initial['construction_occupancy_by_kind'], counts(look_flip=2, look_stay=2))
            self.assertEqual(len(initial['permanent_short_stay_entries']), 1)
            rejected, ref = self.rejected_source(root, entries[3], cref)
            stack.enter_context(patch.object(inventory, 'audited_result', return_value=(rejected, ref)))
            final = inventory.build_inventory(qualification, href, [cref], known_rejection_sources=[ref], root=root)
            self.assertEqual(final['reserved_upper_bound_by_kind'], counts())
            self.assertEqual(final['construction_occupancy_by_kind'], initial['construction_occupancy_by_kind'])
            self.assertEqual(final['confirmed_by_kind'], counts(look_flip=2))
            candidate = plan(scenes=['FloorPlan1'], caps=counts(look_stay=6))
            for field in ('holdout_reference', 'confirmed_by_kind', 'confirmed_by_scene', 'construction_occupancy_by_scene'):
                candidate[field] = final[field]
            self.assertEqual(quota.validate_structure(candidate)['new_tickets_by_kind'], counts(look_stay=6))
            with patch.object(inventory, 'load_receipt', return_value=(final, {})):
                self.assertTrue(quota.authenticate_count_basis(candidate)['count_basis_authenticated'])
            candidate['assignments'][0]['actual_output_caps']['look_stay'] = 7
            with self.assertRaisesRegex(ValueError, 'scene/class capacity'): quota.validate_structure(candidate)
            rejected['metrics']['pairs'][0]['error']['type'] = 'TimeoutError'
            with self.assertRaisesRegex(ValueError, 'original explicit native scientific rejection'):
                inventory.build_inventory(qualification, href, [cref], known_rejection_sources=[ref], root=root)

    def test_duplicate_uncached_stay_cannot_pick_only_the_short_variant(self):
        from contextlib import ExitStack
        originals = rows(flip=1, stay_a=2, stay_b=0)
        originals[1]['stay_displacement']['distance_m'] = .149999
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory).resolve()
            manifest, entries, cref, container = self.fixture(root, input_rows=originals)
            changed = json.loads(Path(entries[2]['pair_reference']['file']).read_text())
            changed['proposal_hash'] = entries[1]['proposal_identity']
            entries[2]['pair_reference'] = write(Path(entries[2]['pair_reference']['file']), changed)
            entries[2]['proposal_identity'] = entries[1]['proposal_identity']
            entries[2]['independent_verification']['proposal_identity'] = entries[1]['proposal_identity']
            manifest['entries'] = manifest['entries'][:1]
            manifest['balance']['selected'] = manifest['balance']['selected'][:1]
            qualification, href = self.bind_inventory(stack, root, manifest, entries, cref, container)
            observed = inventory.build_inventory(qualification, href, [cref], root=root)
            self.assertEqual(observed['reserved_upper_bound_by_kind'], counts(look_stay=1))
            self.assertEqual(observed['permanent_short_stay_entries'], [])
            self.assertEqual(observed['construction_occupancy_by_kind'], counts(look_flip=1, look_stay=1))

    def test_rejection_container_binding_and_complete_paircheck_receipt_are_required(self):
        from contextlib import ExitStack
        from src.tools import merge_pair_v2 as merge
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory).resolve(); _, entries, cref, _ = self.fixture(root)
            data, ref = self.rejected_source(root, entries[0], cref)
            row = data['metrics']['pairs'][0]
            pair = json.loads(Path(entries[0]['pair_reference']['file']).read_text())
            constructed = [{'proposal_identity': entries[0]['proposal_identity'], 'scene': pair['scene'], 'kind': pair['kind'],
                'target_names': [pair['target_w0'], pair['target_w1']],
                'published_bindings': [{'pair_reference': entries[0]['pair_reference'], 'checkpoint_reference': cref}]}]
            stack.enter_context(patch.object(inventory, 'audited_result', return_value=(data, ref)))
            stack.enter_context(patch.object(merge, 'proposal_identity', side_effect=lambda value: value['proposal_hash']))
            row.update(error=None, table_complete=True, native_evidence_verified=True, pair_checks_passed=False,
                rejection_reasons=['fixture_original_pair_check_failure'],
                table_manifest={'file': 'fixture/table.json', 'sha256': 'a' * 64})
            detail = {'table_manifest': row['table_manifest'], 'native_checks': {'passed': True},
                'pair_checks': {'passed': False, 'synthetic_only': False, 'reasons': row['rejection_reasons']}}
            row['detail'] = write(Path(row['scratch_directory']) / 'checks.json', detail)
            self.assertEqual(len(inventory.known_rejections([ref], constructed, root=root)[0]), 1)
            other = copy.deepcopy(constructed); other[0]['published_bindings'][0]['checkpoint_reference']['sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'published binding'):
                inventory.known_rejections([ref], other, root=root)
            Path(row['detail']['file']).write_text('{}')
            with self.assertRaises(ValueError): inventory.known_rejections([ref], constructed, root=root)

    def accepted_cache(self, root, published, checkpoint):
        """Byte-backed s0 masks; original scientific/cache audit is a mock boundary."""
        from src.envs.pair_table_export import TABLE_VERSION, FROZEN_RENDER_VERSION
        from src.tools import t02_export as export
        data, result_ref = self.rejected_source(root, published, checkpoint)
        row = data['metrics']['pairs'][0]; directory = Path(row['scratch_directory'])
        pair = json.loads(Path(published['pair_reference']['file']).read_text())
        package = json.loads(Path(row['source_package']['file']).read_text())
        package.update(kind=pair['kind'], candidate_names=pair['identities']['candidate_names'],
                       target_names=[pair['target_w0'], pair['target_w1']])
        package['independent_verification']['validity'] = {'acceptance_passed': True}
        row['source_package'] = write(directory / 'authenticated_source.json', package)
        table_dir = directory / 'table'; table_dir.mkdir()
        mask_refs = []
        for i, column in enumerate((0, 254)):
            body = bytearray(255 * 255); body[column] = 1
            path = table_dir / f'w{i}.mask'; path.write_bytes(body)
            mask_refs.append({'file': path.name, 'sha256': hashlib.sha256(body).hexdigest(), 'bytes': len(body)})
        identity = identity_fields(pair['pair_id'], published['pair_reference'])
        table = {'version': TABLE_VERSION, 'pair_id': pair['pair_id'], 'scene': pair['scene'], 'kind': pair['kind'],
            'candidate_names': package['candidate_names'], 'target_names': package['target_names'],
            'depth': 4, 'initial_state': 's0', 'dataset_identity': identity,
            'source': {'t01': package['source_provenance'], 'export_version': export.VERSION,
                       'frozen_render_version': FROZEN_RENDER_VERSION, 'field_of_view': 60},
            'states': {'s0': {'captures': [{'masks': {target: mask_refs[i]}, 'synthetic': False}
                        for i, target in enumerate(package['target_names'])]}}, 'files': mask_refs}
        table_ref = write(table_dir / 'table.json', table)
        table_ref['bytes'] = (table_dir / 'table.json').stat().st_size
        row.update(status='exported', table_complete=True, native_evidence_verified=True, pair_checks_passed=True,
                   error=None, rejection_reasons=[], table_manifest=table_ref)
        detail = {'table_manifest': table_ref, 'native_checks': {'passed': True},
                  'pair_checks': {'passed': True, 'synthetic_only': False}}
        row['detail'] = write(directory / 'checks.json', detail)
        cache_ref = {'file': 'results/cache.json', 'sha256': 'c' * 64}
        evidence = {'source': cache_ref, 'certificate': {'tables': {identity['dataset_key']: table_ref}},
            'complete': [{'dataset_key': identity['dataset_key'], 'pair_id': pair['pair_id'],
                          'source': result_ref, 'table_manifest': table_ref}],
            'data': {'metrics': {key: {'file': key, 'sha256': 'a' * 64} for key in
                                 ('validation_report', 'authorization_manifest', 'sampling_plan')}}}
        return data, result_ref, cache_ref, evidence, table

    def test_additional_cache_uses_original_s0_masks_and_complete_receipts_only(self):
        from contextlib import ExitStack
        from src.tools import smoke_e2e_pilot_certificates as certificates
        from src.tools import merge_pair_v2 as merge
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            root = Path(d).resolve(); _, entries, cref, _ = self.fixture(root)
            data, result_ref, cache_ref, evidence, table = self.accepted_cache(root, entries[0], cref)
            stack.enter_context(patch.object(certificates, 'certificate_metadata', return_value=evidence))
            stack.enter_context(patch.object(inventory, 'audited_result', return_value=(data, result_ref)))
            stack.enter_context(patch.object(merge, 'proposal_identity', side_effect=lambda r: r['proposal_hash']))
            native = stack.enter_context(patch.object(certificates, 'load_certificate_batch', side_effect=AssertionError('No native/full-table reload')))
            output, sources, proofs, refs = inventory.additional_cache_rows([cache_ref], root=root)
            self.assertEqual(output[0]['s0_target_left'], [True, False]); native.assert_not_called()
            self.assertEqual(output[0]['s0_left_measurements'][0]['visible_pixels'], 1)
            self.assertEqual(sources, [cache_ref]); self.assertFalse(proofs[0]['native_scientific_revalidation_performed'])
            for field, bad in [('exit_code', 1), ('end', None)]:
                old = data['_meta'][field]; data['_meta'][field] = bad
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'incomplete or failed'):
                    inventory.additional_cache_rows([cache_ref], root=root)
                data['_meta'][field] = old
            original_command = list(data['_meta']['command'])
            data['_meta']['command'][-1] += '.different'
            with self.assertRaisesRegex(ValueError, 'command changed source identity'):
                inventory.additional_cache_rows([cache_ref], root=root)
            data['_meta']['command'] = original_command
            row = data['metrics']['pairs'][0]; row['pair_checks_passed'] = False
            with self.assertRaisesRegex(ValueError, 'not accepted'): inventory.additional_cache_rows([cache_ref], root=root)
            row['pair_checks_passed'] = True
            mask_ref = next(ref for ref in refs if ref['file'].endswith('w0.mask'))
            Path(mask_ref['file']).write_bytes(bytes(255 * 255))
            with self.assertRaisesRegex(ValueError, 'mask bytes changed'): inventory.additional_cache_rows([cache_ref], root=root)

    def test_growing_certified_count_population_preserves_frozen_split_and_stops_class40(self):
        from contextlib import ExitStack
        from src.tools import smoke_e2e_pilot_certificates as certificates
        originals = [item for scene in range(1, 6) for item in rows(flip=8, stay_a=0, stay_b=0, scene=f'FloorPlan{scene}')]
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            root = Path(d).resolve(); manifest, entries, cref, container = self.fixture(root, input_rows=originals)
            # Initial immutable population has39 in this fixture; the real loader
            # requires83. The appended40th pair is not inserted into that split.
            manifest['entries'] = manifest['entries'][:-1]
            manifest['balance']['selected'] = manifest['balance']['selected'][:-1]
            qualification, href = self.bind_inventory(stack, root, manifest, entries, cref, container)
            split = copy.deepcopy(manifest['split'])
            authority = {'mode': certificates.MODE, 'extra_source_reauthentication': copy.deepcopy(certificates.EXTRA)}
            manifest['qualification'] = authority
            stack.enter_context(patch.object(holdout, 'authenticate_population', return_value=(
                manifest, {'formal_cache_batches': [], 'qualification': authority}, qualification, {'file': 'manifest', 'sha256': '4' * 64})))
            data, result_ref, cache_ref, evidence, _ = self.accepted_cache(root, entries[-1], cref)
            stack.enter_context(patch.object(certificates, 'certificate_metadata', return_value=evidence))
            stack.enter_context(patch.object(inventory, 'audited_result', return_value=(data, result_ref)))
            result = inventory.build_inventory(qualification, href, [cref], certificate_count_basis=True,
                                               additional_cache_sources=[cache_ref], root=root)
            self.assertEqual(result['confirmed_by_kind']['look_flip'], 40)
            self.assertTrue(result['class_full']['look_flip']); self.assertFalse(result['formal_pool_authorized'])
            self.assertEqual(result['extra_source_reauthentication'], certificates.EXTRA)
            self.assertEqual(result['additional_cache_publications'], [cache_ref])
            self.assertEqual(manifest['split'], split); self.assertEqual(len(manifest['entries']), 39)
            value = plan(scenes=['FloorPlan6'], caps=counts(look_stay=1))
            for key in ('holdout_reference', 'confirmed_by_kind', 'confirmed_by_scene', 'construction_occupancy_by_scene'):
                value[key] = result[key]
            self.assertEqual(quota.validate_structure(value)['new_tickets_by_kind'], counts(look_stay=1))
            value['assignments'][0]['actual_output_caps']['look_flip'] = 1
            value['assignments'][0]['allowed_kinds'] = ['look_flip', 'look_stay']
            with self.assertRaisesRegex(ValueError, 'global remaining'): quota.validate_structure(value)

    def test_new_dataset_key_cannot_count_the_same_original_proposal_twice(self):
        from contextlib import ExitStack
        from src.tools import smoke_e2e_pilot_certificates as certificates
        originals = [item for scene in range(1, 6) for item in rows(flip=8, stay_a=0, stay_b=0, scene=f'FloorPlan{scene}')]
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            root = Path(d).resolve(); manifest, entries, cref, container = self.fixture(root, input_rows=originals)
            manifest['entries'] = manifest['entries'][:-1]
            manifest['balance']['selected'] = manifest['balance']['selected'][:-1]
            duplicate = json.loads(Path(entries[-1]['pair_reference']['file']).read_text())
            duplicate['proposal_hash'] = entries[-2]['proposal_identity']
            entries[-1]['pair_reference'] = write(Path(entries[-1]['pair_reference']['file']), duplicate)
            entries[-1]['proposal_identity'] = entries[-2]['proposal_identity']
            entries[-1]['independent_verification']['proposal_identity'] = entries[-2]['proposal_identity']
            qualification, href = self.bind_inventory(stack, root, manifest, entries, cref, container)
            data, result_ref, cache_ref, evidence, _ = self.accepted_cache(root, entries[-1], cref)
            stack.enter_context(patch.object(certificates, 'certificate_metadata', return_value=evidence))
            stack.enter_context(patch.object(inventory, 'audited_result', return_value=(data, result_ref)))
            with self.assertRaisesRegex(ValueError, 'repeat an original proposal'):
                inventory.build_inventory(qualification, href, [cref], additional_cache_sources=[cache_ref], root=root)

    def test_explicit_pair_scientific_rejection_removes_only_that_pair_not_holdout(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            root = Path(d).resolve(); manifest, entries, cref, container = self.fixture(root)
            qualification, href = self.bind_inventory(stack, root, manifest, entries, cref, container)
            rejected, rejection_ref = self.rejected_source(root, entries[0], cref)
            stack.enter_context(patch.object(inventory, 'audited_result', return_value=(rejected, rejection_ref)))
            before = inventory.build_inventory(qualification, href, [cref], root=root)
            after = inventory.build_inventory(qualification, href, [cref], known_rejection_sources=[rejection_ref], root=root)
            self.assertEqual(before['confirmed_by_kind']['look_flip'], 2)
            self.assertEqual(after['confirmed_by_kind']['look_flip'], 1)
            self.assertEqual(after['holdout_reference'], before['holdout_reference'])
            self.assertEqual(len(after['later_scientific_exclusions']), 1)
            self.assertEqual(after['construction_occupancy_by_kind'], before['construction_occupancy_by_kind'])
            rejected['_meta']['exit_code'] = 1
            with self.assertRaisesRegex(ValueError, 'complete original scientific rejection'):
                inventory.build_inventory(qualification, href, [cref], known_rejection_sources=[rejection_ref], root=root)

    def test_unknown_left_is_never_balance_or_global_ready(self):
        population = rows(flip=40, stay_a=20, stay_b=20)
        population[0]['s0_target_left'][0] = None
        result = inventory.postfilter(population, set())
        self.assertTrue(all(result['class_full'].values()))
        self.assertIsNone(result['global_balance_within_0_4_to_0_6'])
        self.assertFalse(result['global_ready']); self.assertIsNone(result['formal_T04_ready'])

    def test_pilot_certificate_exception_never_becomes_formal_count_authority(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory).resolve(); manifest, entries, cref, container = self.fixture(root)
            manifest['qualification'] = {'mode': 'approved_original_certificates83_after02_v1',
                'extra_source_reauthentication': {'status': 'not_awaited', 'passed': None}}
            qualification, href = self.bind_inventory(stack, root, manifest, entries, cref, container)
            with self.assertRaisesRegex(ValueError, 'explicit opt-in'):
                inventory.build_inventory(qualification, href, [cref], root=root)


if __name__ == '__main__': unittest.main()

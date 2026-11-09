"""Configuration inventory/proposal fixtures; no invented scientific results."""
import copy
import random
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from src.envs import pair_v2_configuration as configuration
from src.envs import pair_v2_g01_policy as policy
from src.envs import pair_v2_g01_search as search
from src.envs import pair_v2_g01_shard as shard
from src.envs import pair_v2_stay_order as stay
from src.tools import g01_quota_plan as quota
from src.tools import t01v2_g01_review_checkpoint as review
from test_g01_quota import counts, plan, write
import test_g01_quota as quota_fixtures
from test_pair_v2_g01 import record as mixed_record


def original_row(kind='look_flip', *, x=0):
    return {'scene': 'FloorPlan21', 'view': {'position': {'x': x, 'y': 0.9, 'z': 1}, 'yaw': 270, 'horizon': 30},
        'identities': {'anchor_name': 'Book_X', 'candidate_names': ['Bowl_A', 'Vase_B']},
        'kind': kind, 'mode': kind.split('_')[1], 'proposed_pool': kind.split('_')[0],
        'status': 'passed', 'discard_reasons': [], 'counts_as_pair_attempt': True}


def measured(kind='look_flip', **kwargs):
    row = original_row(kind, **kwargs)
    row['g01_configuration'] = configuration.from_pair(row)
    return row


def cfg(root, *, seen=None):
    seen = [] if seen is None else seen
    value = plan('test', scenes=['FloorPlan21'], caps=counts(look_flip=2, look_stay=2, direct_flip=2, direct_stay=2))
    value.update(version=quota.CONFIG_VERSION, confirmed_configurations=seen)
    if seen:
        value['confirmed_by_scene'] = {'FloorPlan21': counts()}
        for item in seen:
            value['confirmed_by_scene']['FloorPlan21'][item['kind']] += 1
        value['confirmed_by_kind'] = dict(value['confirmed_by_scene']['FloorPlan21'])
    ref = write(root / 'configs/quota.json', value)
    name = value['parent_run_name'] + '_c6_floorplan21'
    ticket = quota.assignment(value, ref, 'FloorPlan21', name)
    fields = {**policy.declared('FloorPlan21', 'test', ticket), **stay.fields(), 'producer_module': policy.PRODUCER}
    config = {**fields, 'name': name, 'scene': 'FloorPlan21', 'deadline': 10000000000,
        'enable_candidate_clone': False, 'scene_pairs_per_class': 6, 'raw_evidence_dir': str(root / 'raw')}
    config['shard'] = {**copy.deepcopy(fields), 'name': name, 'scene': 'FloorPlan21'}
    return value, ref, config


class ConfigurationIdentityTests(unittest.TestCase):
    def test_world_order_unordered_candidates_and_anchor_placement_do_not_change_configuration(self):
        row = original_row()
        identity = configuration.from_pair(row)
        worlds = [{'agent_setup_command': identity['key']['start_state'], 'anchor_pose': i} for i in (0, 1)]
        row.update(target_w0='Bowl_A', target_w1='Vase_B', pair_id='different')
        row['identities']['candidate_names'].reverse()
        self.assertEqual(configuration.from_pair(row, worlds[::-1]), identity)

    def test_only_exact_command_numeric_equivalence_is_normalized(self):
        row = original_row(x=-0.0)
        first = configuration.from_pair(row)
        row['view'].update(yaw=270.0, horizon=30.0)
        row['view']['position']['x'] = 0
        self.assertEqual(configuration.from_pair(row), first)
        worlds = [{'agent_setup_command': first['key']['start_state'],
                   'agent': {'horizon': 30.000003814697266, 'position': {'x': 1e-9}}} for _ in range(2)]
        self.assertEqual(configuration.from_pair(row, worlds), first)
        row['view']['position']['x'] = 1e-12
        self.assertNotEqual(configuration.from_pair(row), first)
        with self.assertRaises(ValueError): configuration.from_pair(row, worlds)

    def test_each_semantic_input_changes_key_and_class_does_not(self):
        row = original_row(); base = configuration.from_pair(row)
        row['kind'] = 'look_stay'
        self.assertEqual(configuration.from_pair(row), base)
        for field in ('scene', 'view', 'anchor', 'candidate'):
            changed = copy.deepcopy(row)
            if field == 'scene': changed['scene'] = 'FloorPlan22'
            elif field == 'view': changed['view']['yaw'] = 225
            elif field == 'anchor': changed['identities']['anchor_name'] = 'Other_X'
            else: changed['identities']['candidate_names'][0] = 'Other_A'
            self.assertNotEqual(configuration.from_pair(changed), base)

    def test_seeded_selection_is_input_order_independent_and_per_class(self):
        identity = configuration.from_pair(original_row())
        rows = [{'configuration': identity, 'proposal_identity': str(i), 'kind': kind}
                for kind in ('look_flip', 'look_stay') for i in range(5)]
        # Proposal identity is global; different actual classes need distinct proposals.
        for row in rows: row['proposal_identity'] = row['kind'] + row['proposal_identity']
        expected = configuration.select(rows)
        shuffled = list(rows); random.Random(123).shuffle(shuffled)
        self.assertEqual(configuration.select(shuffled), expected)
        self.assertEqual([r['kind'] for r in expected['selected']], ['look_flip', 'look_stay'])
        self.assertEqual(len(expected['duplicates']), 8)
        broken = copy.deepcopy(identity); broken['key']['start_state']['forceAction'] = True
        with self.assertRaises(ValueError): configuration.validate(broken)


class ConfigurationRuntimeTests(unittest.TestCase):
    def quota(self, seen=()):
        return search.SceneQuota(6, actual_output_caps=counts(look_flip=2, look_stay=2, direct_flip=2, direct_stay=2),
            allowed_kinds=list(search.KINDS), scene='FloorPlan21', seen_configurations=list(seen))

    def catalog(self, *, x=0):
        value = mixed_record(); row = original_row(x=x)
        value.update(search.fields(True, True), view=row['view'], identity=row['identities'])
        return value

    def test_controller_restart_and_new_iterator_share_actual_class_occupancy(self):
        quota = self.quota()
        first = measured('look_flip'); quota.record('attempt1', first)
        # A new Controller produces a new catalog/iterator, but the same scene quota.
        iterator = search.MixedIterator([self.catalog()], quota)
        proposal, _, _ = next(iterator)
        self.assertEqual((proposal['projection_pool']['pool'], proposal['mode']), ('look', 'stay'))
        self.assertEqual(quota.desired_modes('look', 'flip', measured(x=1)['g01_configuration']), ['stay', 'flip'])
        with self.assertRaisesRegex(ValueError, 'already represented'):
            quota.record('attempt2', first)

    def test_historical_occupancy_only_suppresses_its_actual_class_and_configuration(self):
        value = measured()['g01_configuration']
        quota = self.quota([{'configuration': value, 'kind': 'look_flip'}])
        self.assertEqual(quota.desired_modes('look', 'flip', value), ['stay'])
        self.assertEqual(quota.desired_modes('direct', 'flip', value), ['flip', 'stay'])
        self.assertEqual(quota.desired_modes('look', 'flip', measured(x=1)['g01_configuration']), ['flip', 'stay'])
        self.assertEqual(quota.passed, counts())

    def test_actual_crossover_duplicate_is_capacity_skip_after_science(self):
        value = measured('look_stay')['g01_configuration']
        quota = self.quota([{'configuration': value, 'kind': 'look_stay'}])
        row = measured('look_stay'); row['proposed_pool'] = 'direct'
        with self.assertRaises(search.ConfigurationAlreadyCollected): quota.before_export(row)
        row.update(status='rejected', validity={'acceptance_passed': True},
                   discard_reasons=['collection_configuration_already_represented'])
        quota.record('cross', row)
        self.assertEqual(quota.passed, counts())
        self.assertEqual(quota.attempts['direct_stay'], 1)

    def test_old_default_and_v1_schema_are_unchanged(self):
        old = search.SceneQuota(6)
        self.assertNotIn('seen_configurations', old.summary())
        self.assertEqual(old.summary()['version'], search.VERSION)
        old = search.SceneQuota(6, actual_output_caps=counts(look_stay=1), allowed_kinds=['look_stay'])
        self.assertEqual(old.summary()['version'], search.QUOTA_VERSION)
        self.assertNotIn('configuration_version', old.summary())

    def test_new_plan_assignment_and_spawn_scope_bind_exact_configuration_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seen = [{'configuration': measured()['g01_configuration'], 'kind': 'look_flip'}]
            plan_value, ref, config = cfg(root, seen=seen)
            with patch.object(quota, 'ROOT', root):
                policy.validate_source(config, config)
                with shard.installed(config) as state:
                    self.assertEqual(state.configuration_seen[(seen[0]['configuration']['configuration_id'], 'look_flip')], seen[0]['configuration'])
                    self.assertEqual(shard.marked.tags()['g01_search_version'], search.CONFIG_VERSION)
                changed = copy.deepcopy(config); changed['g01_seen_configurations'] = []
                with self.assertRaises(ValueError): policy.validate_source(changed)
            wrong = copy.deepcopy(plan_value); wrong['confirmed_by_kind'] = counts()
            with self.assertRaises(ValueError): quota.validate_structure(wrong)


class ConfigurationInventoryTests(unittest.TestCase):
    def test_authenticated_complete_pool_deduplicates_before_counting_and_retains_raw(self):
        from src.envs import pair_table_source as sources
        fixture = quota_fixtures.TestCountBasisTests()
        value, containers, packages = fixture.fixture(repeated=True)
        rows = []
        for index, ((path, (container, ref)), package) in enumerate(zip(containers.items(), packages.values())):
            entry = container['metrics']['candidate_pairs'][0]
            entry['proposal_identity'] = str(index + 4) * 64
            package['independent_verification']['proposal_identity'] = entry['proposal_identity']
            package['row'] = original_row()
            identity = configuration.from_pair(package['row'])
            package['worlds'] = [{'agent_setup_command': identity['key']['start_state']} for _ in (0, 1)]
            rows.append({'configuration': identity, 'kind': 'look_flip', 'proposal_identity': entry['proposal_identity']})
        value.update(version=quota.CONFIG_VERSION,
            confirmed_configurations=configuration.select(rows)['seen_configurations'])
        def audited(path): return containers['results/' + Path(path).name]
        def loaded(path, pair_id, *, pair_file): return copy.deepcopy(packages[pair_file])
        with patch.object(sources, 'audited_result', side_effect=audited), \
             patch.object(sources, 'load_reviewed_pair', side_effect=loaded) as original_loader:
            proof = quota.authenticate_count_basis(value)
            self.assertEqual(original_loader.call_count, 2)
            self.assertEqual(proof['raw_unique_proposals_by_kind']['look_flip'], 2)
            self.assertEqual(proof['confirmed_by_kind']['look_flip'], 1)
            self.assertEqual(len(proof['configuration_selection']['duplicates']), 1)
            broken = copy.deepcopy(value); broken['confirmed_configurations'] = []
            with self.assertRaisesRegex(ValueError, 'configuration occupancy'):
                quota.authenticate_count_basis(broken)

    def test_new_source_replays_configuration_duplicates_in_original_journal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); _, _, config = cfg(root)
            raw = measured(); raw.update(shard.tags(config))
            raw_ref = write(root / 'raw/attempt.json', raw)
            import json
            journal = root / 'data/world_pairs/v2/source/journal.jsonl'
            journal.parent.mkdir(parents=True)
            journal.write_text(json.dumps({'raw_attempt': raw_ref, 'proposed_pool': 'look'}) + '\n')
            from test_g01_quota import ref
            state = shard.make_quota(config); state.record(root / 'raw/attempt.json', raw)
            data = {'config': config, 'metrics': {'g01_collection': state.summary(), 'passed_by_class': state.passed}}
            with patch.object(quota, 'ROOT', root), patch.object(review.merge, 'ROOT', root), \
                 patch.object(quota, 'authenticate_count_basis', return_value={'fixture_source_boundary': True}), \
                 patch.object(review, 'verified_passed_rows', return_value=({'eligible': True}, ref(journal), [])):
                self.assertEqual(review.verify_collection(data), state.summary())
                raw['g01_configuration']['key']['anchor_name'] = 'tampered'
                raw_ref = write(root / 'raw/attempt.json', raw)
                journal.write_text(json.dumps({'raw_attempt': raw_ref, 'proposed_pool': 'look'}) + '\n')
                with patch.object(review, 'verified_passed_rows', return_value=({'eligible': True}, ref(journal), [])):
                    with self.assertRaises(ValueError): review.verify_collection(data)


if __name__ == '__main__':
    unittest.main()

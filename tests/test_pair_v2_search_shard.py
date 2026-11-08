"""CPU-only synthetic control-flow fixtures; no THOR, GPU, or runtime claims."""
import copy
from contextlib import contextmanager, ExitStack
import itertools
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.envs import pair_v2_search_shard as shard


def identity(anchor='X', origin='natural'):
    return {'candidate_names': ['A', 'B'], 'anchor_name': anchor,
            'target_type': 'Bowl', 'anchor_type': anchor, 'search_candidate_origin': origin}


class Channel:
    def __init__(self):
        self.messages = []

    def send(self, value):
        self.messages.append(copy.deepcopy(value))


class BodyFixture:
    """Use the real search control flow with explicit simulated backend returns."""
    def __init__(self, root, outcomes, *, natural=True, empty_clone_union=False,
                 initial_preflight=True, later_preflights=None, orphan=False):
        self.root = Path(root).resolve()
        self.config = shard.enable_config({'name': 'fixture', 'scene': 'FloorPlan1',
            'runtime': {'scratch': str(self.root)}, 'deadline': time.time() + 100,
            'raw_evidence_dir': str(self.root / 'raw'), 'attempt_journal': 'attempts.jsonl'})
        Path(self.config['raw_evidence_dir']).mkdir()
        self.session = shard.session_factory(self.root)
        self.channel, self.alive, self.controllers = Channel(), {}, []
        self.sources = [{'name': f'S{i}', 'objectType': f'T{i}'} for i in range(len(outcomes))]
        self.outcomes, self.searches = outcomes, []
        self.natural, self.empty_clone_union = natural, empty_clone_union
        self.preflights = [initial_preflight, *(later_preflights or [])]
        self.orphan = orphan

    @contextmanager
    def controller(self, config, role, **kwargs):
        number = len(self.controllers)
        pid = 100 + number
        ident = {'pid': pid, 'start_ticks': 1000 + number, 'pgid': 9, 'sid': 9}
        self.controllers.append({'role': role, 'deadline': config['deadline'], 'pid': pid})
        if self.alive:
            raise AssertionError('Construction Controllers overlapped')
        self.alive[pid] = ident
        controller = SimpleNamespace(unity_pid=pid, number=number,
                                     _build=SimpleNamespace(commit_id=shard.engine.BUILD))
        try:
            yield controller
        finally:
            if not self.orphan:
                self.alive.pop(pid)

    def simulator(self, controller, config, directory):
        fixture = self

        class Simulator:
            def __init__(self):
                self.store = fixture.session.store(directory, shard.RAW_BUDGET_BYTES)
                self.controller, self.config, self.scene_preflight = controller, config, None

            def preflight_scene(self):
                passed = fixture.preflights[controller.number] if controller.number < len(fixture.preflights) else True
                snapshot = {'objects': copy.deepcopy(fixture.sources), 'source': controller.number}
                self.scene_preflight = {'passed': passed, 'snapshot': snapshot, 'reasons': []}
                self.store.save_json('scene_preflight.json', self.scene_preflight)
                return copy.deepcopy(self.scene_preflight)

            def finish(self):
                return self.store.index()
        return Simulator()

    def attempt(self, sim, preflight, source, visit, source_index):
        passed = bool(self.outcomes[source_index] and preflight['passed'])
        record = {'source_name': source['name'], 'source_type': source['objectType'],
            'fixed_source_index': source_index, 'controller_id': visit['controller_id'],
            'status': 'clone_ready' if passed else 'clone_setup_failed', 'passed': passed,
            'clone_attempted': bool(preflight['passed']), 'error': None}
        record['record_reference'] = sim.store.save_json('clone_selection.json', record)
        visit['clone_record'] = record
        prepared = {'snapshot': {'objects': copy.deepcopy(self.sources), 'expanded': True},
                    'lineage': {'candidate_names': ['C', 'D']}}
        return record, prepared if passed else None

    def groups(self, objects, lineage=None):
        if lineage is not None:
            return [] if self.empty_clone_union else [identity(), identity('Z', 'cloned')]
        return [identity()] if self.natural else []

    def search(self, sim, config, baseline, groups, state, handle, connection):
        self.searches.append({'role': state['selected_controller_id'], 'groups': copy.deepcopy(groups),
                              'attempts_published': len(state['clone_attempts']),
                              'deadline': config['deadline']})

    def run(self):
        with ExitStack() as stack:
            for target, name, value in (
                (shard.base, 'ROOT', self.root), (shard.stable, '_SESSION', self.session),
                (shard.clone, '_PREPARATION', None), (shard.base, 'controller_session', self.controller),
                (shard, 'StableCloneSimulator', self.simulator), (shard, 'clone_attempt', self.attempt),
                (shard, 'eligible_clone_sources', lambda objects: copy.deepcopy(self.sources)),
                (shard, 'union_identity_sets', self.groups), (shard, 'run_round_robin', self.search),
                (shard, 'process_identity', lambda pid: self.alive.get(pid))):
                stack.enter_context(patch.object(target, name, value))
            return shard.search_body(self.config, self.channel)


class SearchControlTests(unittest.TestCase):
    def test_six_failures_then_new_natural_controller_with_one_absolute_deadline(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = BodyFixture(root, [False] * 6)
            result = fixture.run()
            self.assertEqual(len(result['clone_attempts']), 6)
            self.assertEqual([row['fixed_source_index'] for row in result['clone_attempts']], list(range(6)))
            self.assertEqual(len(fixture.controllers), 7)
            self.assertEqual(fixture.searches[0]['role'], 'search_controller_06')
            self.assertEqual({row['deadline'] for row in fixture.controllers}, {fixture.config['deadline']})
            self.assertTrue(all(row['preflight_reference']['sha256'] for row in result['clone_attempts']))
            self.assertEqual(result['cleanup_errors'], [])

    def test_first_success_stops_and_keeps_natural_union_published_before_search(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = BodyFixture(root, [False, True, True])
            result = fixture.run()
            self.assertEqual(len(fixture.controllers), 2)
            self.assertEqual([row['passed'] for row in result['clone_attempts']], [False, True])
            self.assertEqual(fixture.searches[0]['attempts_published'], 2)
            self.assertEqual([row['search_candidate_origin'] for row in fixture.searches[0]['groups']], ['natural', 'cloned'])

    def test_physical_clone_success_empty_union_preserves_fresh_natural_opportunity(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = BodyFixture(root, [True, True], empty_clone_union=True)
            result = fixture.run()
            self.assertEqual(len(result['clone_attempts']), 1)
            self.assertIs(result['clone_attempts'][0]['passed'], True)
            self.assertEqual(len(fixture.controllers), 2)
            self.assertEqual(fixture.searches[0]['role'], 'search_controller_01')
            self.assertEqual(result['controller_visits'][0]['clone_group_outcome'], 'clone_ready_no_eligible_identity')

    def test_initial_failed_preflight_is_not_rerolled(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = BodyFixture(root, [False, True], initial_preflight=False)
            result = fixture.run()
            self.assertEqual(len(fixture.controllers), 1)
            self.assertEqual(result['clone_attempts'], [])
            self.assertEqual(result['scene_excluded']['reason'], 'settled_baseline_window_preflight_failed')
            self.assertTrue(result['scene_preflight_reference']['sha256'])

    def test_no_clone_sources_searches_existing_natural_groups(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = BodyFixture(root, [])
            result = fixture.run()
            self.assertEqual(len(fixture.controllers), 1)
            self.assertEqual(result['clone_preparation']['status'], 'natural_fallback')
            self.assertEqual(len(fixture.searches), 1)

    def test_uncertain_cleanup_never_starts_next_controller(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = BodyFixture(root, [False, True], orphan=True)
            result = fixture.run()
            self.assertEqual(len(fixture.controllers), 1)
            self.assertEqual(result['status'], 'worker_failed')
            self.assertTrue(result['cleanup_errors'])

    def test_measured_w0_denominator_and_physical_streak_are_not_invented(self):
        counts = {}
        for value in (True, False, None, 1):
            shard.update_w0(counts, 'Mug', {'physical_w0': {'valid_physical': value}})
        self.assertEqual(counts['Mug'], {'measured_attempts': 2, 'passed': 1,
                                       'rejected': 1, 'measurement_unavailable': 2})
        self.assertFalse(shard.physical_rejected({'discard_reasons': ['evidence_incomplete_storage_cap']}))
        self.assertTrue(shard.physical_rejected({'physical_w1': {'valid_physical': False}}))
        self.assertFalse(shard.physical_rejected({'physical_w0': {'valid_physical': 0}}))

    def run_scheduler(self, root, build, *, limit=30):
        root = Path(root).resolve()
        config = {'scene': 'FloorPlan1', 'max_attempts': limit, 'scene_pairs_per_class': 5,
                  'raw_evidence_dir': str(root), 'attempt_journal': 'journal.jsonl'}
        state = {'rows': [], 'excluded': None, 'status': 'completed_search', 'w0_by_anchor': {},
                 'selected_controller_id': 'fixture', 'cleanup_errors': []}
        sim = SimpleNamespace(check_deadline=lambda: None, reachable_positions=lambda: {'positions': []})
        seen = []

        def instrumented(*args, **kwargs):
            seen.append(args[3]['anchor_name'])
            return build(*args, **kwargs)

        with (patch.object(shard.base, 'build_pair', instrumented),
              patch.object(shard, 'proposal_iterator', lambda *args: itertools.repeat(({}, {}))),
              patch.object(shard, 'summary', lambda state, cfg: {'status': state['status']})):
            with (root / 'journal.jsonl').open('x') as handle:
                shard.run_round_robin(sim, config, {}, [identity('X'), identity('Y')], state, handle, Channel())
        return state, seen

    def test_round_robin_skips_each_group_after_ten_measured_physical_rejections(self):
        def build(*args, **kwargs):
            row = kwargs['attempt_row']
            row.update(physical_w0={'valid_physical': False}, discard_reasons=['w0_physical'])
            return row
        with tempfile.TemporaryDirectory() as root:
            state, seen = self.run_scheduler(root, build)
            self.assertEqual(seen, ['X', 'Y'] * 10)
            self.assertEqual(len(state['rows']), 20)
            self.assertTrue(all(row['skipped'] for row in state['scheduler']))

    def test_evidence_capacity_is_expected_rejection_and_search_continues(self):
        def build(*args, **kwargs):
            kwargs['attempt_row']['physical_w0'] = {'valid_physical': True}
            raise shard.EvidenceIncomplete('synthetic cap')
        with tempfile.TemporaryDirectory() as root, patch.object(shard.traceback, 'print_exc') as trace:
            state, seen = self.run_scheduler(root, build, limit=4)
            self.assertEqual(seen, ['X', 'Y', 'X', 'Y'])
            self.assertTrue(all(row['discard_reasons'] == ['evidence_incomplete_storage_cap'] for row in state['rows']))
            self.assertEqual(state['w0_by_anchor']['X']['passed'], 2)
            self.assertTrue(all(row['physical_rejections'] == 0 for row in state['scheduler']))
            trace.assert_not_called()

    def test_private_bindings_restore_and_export_refuses_cap_before_archive(self):
        config = shard.enable_config({'name': 'fixture'})
        before = (shard.stable.EvidenceSession, shard.stable.RetentionPolicy,
                  shard.stable._CLONE_GENERATOR, shard.stable.replay_worker,
                  shard.base.wait_replay, shard.base.export_pair)
        row = {'identities': identity()}

        def stable_generator(cfg, connection):
            with (patch.object(shard.stable, '_SESSION', SimpleNamespace(retention_plan=lambda: {'can_materialize_complete': False})),
                  patch.object(shard, 'union_identity_sets', return_value=[identity()]),
                  patch.object(shard.clone, 'JsonProofArchive') as archive):
                with self.assertRaises(shard.EvidenceIncomplete):
                    shard.base.export_pair(cfg, row, [{'objects': []}], [], [], [], [])
                archive.assert_not_called()
            raise RuntimeError('synthetic worker failure')

        with patch.object(shard, '_STABLE_GENERATOR', stable_generator):
            with self.assertRaisesRegex(RuntimeError, 'synthetic worker failure'):
                shard.generator_worker(config, Channel())
        self.assertEqual(before, (shard.stable.EvidenceSession, shard.stable.RetentionPolicy,
            shard.stable._CLONE_GENERATOR, shard.stable.replay_worker,
            shard.base.wait_replay, shard.base.export_pair))
        self.assertEqual(row['search_policy'], shard.SEARCH_POLICY)

    def test_main_writer_preserves_versions_without_starting_worker(self):
        captured = []

        def fake_main():
            cfg, metrics = {}, {}
            shard.base.write_result('fixture', metrics, cfg)
            return 7

        original = shard.stable.generator_worker
        with (patch.object(shard.stable, 'main', fake_main),
              patch.object(shard.base, 'write_result', lambda n, m, c: captured.append((m, c)))):
            self.assertEqual(shard.main(), 7)
        self.assertIs(shard.stable.generator_worker, original)
        metrics, config = captured[0]
        self.assertEqual(metrics['search_version'], shard.SEARCH_VERSION)
        self.assertEqual(config['search_policy'], shard.SEARCH_POLICY)
        self.assertEqual(config['preflight_version'], 'settled_baseline_window_v1')
        self.assertEqual(config['evidence_version'], 'anchor_union_look_v1')
        self.assertTrue(config['enable_candidate_clone'])

    def test_fixed_source_stays_eligible_when_current_fresh_order_changes(self):
        # The approved six-source cutoff belongs to the FIRST Controller only.
        objects = [{'name': f'N{i}', 'objectType': f'T{i}', 'pickupable': True} for i in range(7)]
        source = objects[-1]
        with tempfile.TemporaryDirectory() as temporary:
            session = shard.session_factory(temporary)
            sim = SimpleNamespace(store=session.store(Path(temporary) / 'raw', 100000))
            preflight = {'passed': True, 'snapshot': {'objects': objects}}
            visit = {'controller_id': 'fresh', 'preflight_reference': {'file': 'fixture', 'sha256': 'x'}}
            prepared = {'passed': True, 'lineage': {},
                        'record_reference': sim.store.save_json('validated.json', {'passed': True}), 'record': {}}
            with (patch.object(shard.clone, 'select_clone_token', return_value={'native': True}),
                  patch.object(shard, 'prepare_clone_baseline', return_value=prepared) as prepare):
                record, result = shard.clone_attempt(sim, preflight, source, visit, 5)
            self.assertTrue(record['passed'])
            self.assertIs(result, prepared)
            prepare.assert_called_once()

    def test_fresh_source_missing_or_no_longer_singleton_does_not_create_clone(self):
        source = {'name': 'M1', 'objectType': 'Mug', 'pickupable': True}
        objects = [source, {'name': 'M2', 'objectType': 'Mug', 'pickupable': True}]
        with tempfile.TemporaryDirectory() as temporary:
            session = shard.session_factory(temporary)
            sim = SimpleNamespace(store=session.store(Path(temporary) / 'raw', 100000))
            visit = {'controller_id': 'fresh', 'preflight_reference': {}}
            with patch.object(shard, 'prepare_clone_baseline') as prepare:
                record, result = shard.clone_attempt(sim, {'passed': True, 'snapshot': {'objects': objects}}, source, visit, 0)
            self.assertEqual(record['status'], 'source_unavailable')
            self.assertIsNone(result)
            prepare.assert_not_called()

    def test_replay_black_preserves_bounded_proof_without_claiming_complete_raw(self):
        with tempfile.TemporaryDirectory() as temporary:
            original_store = shard.engine.EvidenceStore
            config = {'runtime': {'scratch': temporary}, 'snapshot': {},
                      '_private_source_image_bytes': {}}

            def actual_replay(cfg, connection):
                store = shard.engine.EvidenceStore(Path(temporary) / 'replay', 100000)
                image = store.save('rgb.png', b'synthetic payload; PNG validation belongs to black guard')
                ref = store.save_json('black.json', {'files': {'rgb': image}, 'black': True})
                return {'exit_code': 86, 'black_frame_reference': ref, 'cleanup_errors': []}

            with patch.object(shard.engine, 'replay_worker', actual_replay):
                result = shard.replay_worker(config, Channel())
            self.assertIs(shard.engine.EvidenceStore, original_store)
            self.assertFalse(result['buffered_raw_complete'])
            self.assertTrue(result['terminal_evidence']['preserved'])
            self.assertEqual(result['terminal_evidence']['capacity_plan']['payload_limit_bytes'], shard.TERMINAL_RESERVE_BYTES)
            self.assertTrue(Path(result['black_frame_reference']['file']).is_file())
            self.assertNotIn('_private_source_image_bytes', config['snapshot'])

    def test_black_child_cleanup_is_not_lost_before_build_pair_raises(self):
        result = {'exit_code': 86, 'black_frame_reference': {'file': 'actual', 'sha256': 'actual'},
                  'cleanup_errors': ['actual child evidence finalization failed']}
        with (patch.object(shard.stable, 'wait_replay', return_value=result),
              patch.object(shard.clone, '_REPLAY_CLEANUP', [])):
            with self.assertRaises(shard.engine.BlackFrame):
                shard.wait_replay()
            self.assertEqual(shard.clone._REPLAY_CLEANUP, result['cleanup_errors'])

    def test_uncertain_replay_cleanup_stops_before_a_second_witness(self):
        result = {'cleanup_errors': ['owned child may remain']}
        with (patch.object(shard.stable, 'wait_replay', return_value=result),
              patch.object(shard.clone, '_REPLAY_CLEANUP', [])):
            with self.assertRaisesRegex(SystemExit, 'cleanup uncertain'):
                shard.wait_replay()

    def test_native_proposal_query_restores_common_baseline_before_reading_moved_receptacle(self):
        objects = [{'name': name, 'position': {'x': index, 'y': 0, 'z': 0},
                    'receptacle': name == 'R'} for index, name in enumerate(('A', 'B', 'X', 'R'))]
        baseline, events = {'objects': objects, 'reference': {'file': 'baseline', 'sha256': 'fixture'}}, []

        def restored(value):
            self.assertIs(value, baseline)
            events.append('restore_common')
            return {}, {'restore_deterministic': True}

        def points(name):
            self.assertEqual(events, ['restore_common', 'setup_record'])
            events.append('native_query')
            return [{'selected_position': {'x': 0, 'y': 0, 'z': 0}, 'receptacle_name': name}]

        sim = SimpleNamespace(restore_snapshot=restored, action_count=9, spawn_points=points,
            store=SimpleNamespace(save_json=lambda *args: events.append('setup_record')))
        config = {'max_receptacles': 1, 'max_points_per_side': 1, 'max_view_positions': 1}
        with (patch.object(shard.base, 'placement_groups', return_value=([], [])),
              patch.object(shard.base, 'view_proposals', return_value=[{'view': 'actual'}]),
              patch.object(shard.base, 'construction_pairs', return_value=[{'proposal': 'actual'}])):
            result = next(shard.proposal_iterator(sim, baseline, identity(), [], config))
        self.assertEqual(events, ['restore_common', 'setup_record', 'native_query'])
        self.assertEqual(result, ({'proposal': 'actual'}, {'view': 'actual'}))

    def test_failed_common_restore_blocks_native_query(self):
        sim = SimpleNamespace(restore_snapshot=lambda value: ({}, {'restore_deterministic': False}),
            action_count=1, store=SimpleNamespace(save_json=lambda *args: None))
        with self.assertRaisesRegex(RuntimeError, 'Common baseline state restoration failed'):
            next(shard.proposal_iterator(sim, {}, identity(), [], {}))

    def test_cap_memory_only_reference_is_explicit_even_inside_virtual_read_view(self):
        with tempfile.TemporaryDirectory() as temporary:
            session = shard.SearchEvidenceSession(temporary, raw_budget_bytes=1)
            store = session.store(Path(temporary) / 'raw', 1000)
            ref = store.save_json('preflight.json', {'passed': False, 'reasons': ['measured rejection']})
            with session.reading():
                self.assertTrue(Path(ref['file']).is_file())  # Runtime access remains possible.
                declared = shard.raw_reference(ref)
            self.assertFalse(declared['materialized'])
            self.assertIn('not_retained_raw', declared['availability'])
            self.assertEqual(declared['sha256'], ref['sha256'])

    def test_compact_preflight_retains_measured_values_without_filling_unknown(self):
        record = {'passed': False, 'reasons': ['actual window displacement'],
            'diagnostics': {'initial_settling': {'settled': True, 'criterion': 'native_rest',
                'observations': [{'measurements': {'objects': {'X': {'baseline_displacement_m': .02}}}}]},
                'stability_window': {'completed': True,
                    'observations': [{'measurements': {'objects': {'X': {'baseline_displacement_m': .03}}}}]}}}
        compact = shard.compact_preflight(record)
        self.assertEqual(compact['initial_max_displacement_m'], .02)
        self.assertEqual(compact['window_max_displacement_m'], .03)
        self.assertFalse(compact['passed'])
        missing = shard.compact_preflight({})
        self.assertIsNone(missing['observed_initial_steps'])
        self.assertIsNone(missing['window_max_displacement_m'])


if __name__ == '__main__':
    unittest.main()

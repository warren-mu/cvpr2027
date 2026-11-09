"""CPU-only plan/provenance and real synthetic Look table integration fixtures."""
from contextlib import ExitStack, redirect_stdout
from dataclasses import replace
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.envs.paired_env import RGBFrame
from src.envs.pair_table import RenderedView
from src.envs.pair_table_export import build_table, state_id
from src.tools import smoke_e2e_look_plan as look
from src.tools import smoke_e2e_views as views
from src.tools import smoke_e2e_scope as scope
from src.tools import smoke_e2e_replay as replay
from src.tools import smoke_e2e_eval as evaluation

ROOT = Path(__file__).resolve().parents[1]


class PlanFixture:
    def __init__(self):
        self.plan = json.loads((ROOT / 'configs/smoke_e2e_look_five.json').read_text())
        self.reference = {'file': look.CHECKPOINT, 'sha256': 'checkpoint-bytes',
                          'metrics_sha256': look.CHECKPOINT_METRICS_SHA256, 'git_commit': 'fixture-only'}
        self.entries, self.reviews, self.packages = [], [], {}
        for index, row in enumerate(self.plan['pairs']):
            pair_ref = {'file': f'data/world_pairs/v2/fixture/{row["pair_id"]}/pair.json',
                        'sha256': f'pair-bytes-{index}'}
            source_image = {'file': f'data/world_pairs/v2/fixture/{row["pair_id"]}/review.png',
                            'sha256': row['review_image_sha256']}
            entry = {'pair_id': row['pair_id'], 'kind': row['kind'], 'scene': 'FloorPlan203',
                     'journal_index': index, 'source_file': look.SCENE_SOURCE,
                     'proposal_identity': f'unique-proposal-{index}', 'pair_reference': pair_ref,
                     'review_image': source_image}
            self.entries.append(entry)
            self.reviews.append({'file': row['review_image'], 'sha256': row['review_image_sha256'],
                                 'pair_id': row['pair_id'], 'source_file': look.SCENE_SOURCE,
                                 'kind': row['kind'], 'source_image': source_image})
            self.packages[row['pair_id']] = {'pair_id': row['pair_id'], 'kind': row['kind'],
                'source_provenance': {'container': self.reference, 'pair': pair_ref,
                                      'scene': {'file': look.SCENE_SOURCE}},
                'independent_verification': {'proposal_identity': entry['proposal_identity']},
                'sources': [self.reference]}
        self.data = {'_meta': {'exit_code': 0, 'end': 'fixture-ended'}, 'metrics': {
            'kind': 't01v2_review_checkpoint', 'status': 'checkpoint_ready_for_human_review',
            'errors': [], 'final_pilot_acceptance': False,
            'candidate_pairs': self.entries, 'review_files': self.reviews}}
        self.loaded = []

    def load(self, path, pair_id, *, pair_file):
        self.loaded.append((path, pair_id, pair_file))
        return self.packages[pair_id]

    def image(self, path, root, limit):
        row = next(row for row in self.plan['pairs'] if Path(row['review_image']).name == Path(path).name)
        return {'file': str(path), 'sha256': row['review_image_sha256'], 'bytes': 1}

    def run(self, plan=None):
        with patch.object(look, 'audited_result', return_value=(self.data, self.reference)):
            return look.review_packages(self.plan if plan is None else plan, {}, root=ROOT,
                                        file_ref=self.image, load_pair=self.load)


def mask(count, offset=0):
    return bytes(offset) + bytes([1]) * count + bytes(255 * 255 - offset - count)


class LookRenderer:
    def domain(self, world, state):
        return {'complete': True, 'moves': [], 'evidence': {'synthetic': True}}

    def render_pair(self, state):
        visible = state.heading_index != 0
        result = []
        for world in (0, 1):
            source = {'pair_id': 'synthetic_look', 'state_id': state_id(state), 'world_id': f'w{world}',
                      'snapshot_sha256': str(world) * 64, 'fixture': True}
            result.append(RenderedView(
                RGBFrame(bytes([40 + (world * 20 if visible else 0), 50, 60]) * (255 * 255), 255, 255),
                {'X': mask(300 if visible else 0, 1000), 'A': mask(300), 'B': mask(300, 500)},
                {'A': 1, 'B': 1, 'X': 2},
                {'position': {'x': 0, 'y': 0, 'z': 0}, 'heading': state.heading_index},
                'synthetic_cached_worker', f'cached:{state_id(state)}:{world}', source, True))
        return tuple(result)


def look_table(renderer=None):
    return build_table(renderer or LookRenderer(), pair_id='synthetic_look', scene='FloorPlan203', kind='look_flip',
        instruction='Select the marked object closest to the Box.', anchor_name='X',
        candidate_names=('A', 'B'), target_names=('A', 'B'),
        initial_position={'x': 0, 'y': 0, 'z': 0}, initial_heading=0, source={'synthetic': True})


class LookPlanTests(unittest.TestCase):
    def test_shadow_fork_precedes_anchor_seen_without_filtering_original_pool(self):
        class ShadowRenderer(LookRenderer):
            def render_pair(self, state):
                views = super().render_pair(state)
                if state.heading_index == 1:
                    return tuple(replace(view, masks={**view.masks, 'X': mask(0)}) for view in views)
                return views
        table = look_table(ShadowRenderer())
        certificate = {'tables': {table.pair_id: table.manifest_reference}}
        with patch.object(scope, 'authorize_smoke', return_value=certificate):
            backend = scope.SmokeTableBackend({table.pair_id: table}, certificate=certificate,
                pair_checks={}, report={}, authenticated_ids=set())
        try:
            observations = []
            for world in ('w0', 'w1'):
                env = scope.SmokeTableEnvironment(backend)
                env.reset(table.pair_id, world)
                observation, _, _, info = env.step('turn_right')
                self.assertEqual(info['tau'], 1)
                self.assertIsNone(info['x_seen_step'])
                observations.append(observation)
                _, _, _, info = env.step('turn_right')
                self.assertEqual((info['tau'], info['x_seen_step']), (1, 2))
                env.close()
            self.assertNotEqual(observations[0].image.data, observations[1].image.data)
            coverage = scope.coverage([table])
            self.assertGreater(coverage['shadow_divergent_states_by_pair'][table.pair_id], 0)
            self.assertEqual(coverage['in_clean_subset_by_pair'], {table.pair_id: False})
            clean = look_table()
            self.assertEqual(scope.shadow_summary([clean])['in_clean_subset_by_pair'], {clean.pair_id: True})
        finally:
            backend.close()

    def test_frozen_plan_loads_exact_first_five_without_claiming_review(self):
        fixture = PlanFixture()
        plan, _, images, packages, sources = fixture.run()
        self.assertEqual([row['pair_id'] for row in packages], list(look.PAIR_IDS))
        self.assertEqual([call[2] for call in fixture.loaded], [row['pair_reference']['file'] for row in fixture.entries])
        self.assertEqual([row['sha256'] for row in images], list(look.IMAGE_HASHES))
        self.assertEqual(plan['authorization']['Claude_review_of_these_images'], 'not_confirmed')
        self.assertFalse(plan['planning']['old_Direct_T0_is_same_pool_baseline'])
        self.assertEqual(sources, [fixture.reference])

    def test_scope_order_review_and_sample_size_cannot_be_silently_changed(self):
        fixture = PlanFixture()
        for mutate in (
            lambda p: p['pairs'].reverse(),
            lambda p: p['authorization'].update(Claude_review_of_these_images='approved'),
            lambda p: p['planning'].update(cache_distinct_pair_states=200),
            lambda p: p['planning'].update(old_Direct_T0_is_same_pool_baseline=True),
            lambda p: p['pairs'][0].update(pair_id='replacement'),
        ):
            changed = copy.deepcopy(fixture.plan)
            mutate(changed)
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                fixture.run(changed)

    def test_duplicate_changed_checkpoint_or_wrong_image_source_blocks(self):
        for mutate in (
            lambda f: f.entries[1].update(proposal_identity=f.entries[0]['proposal_identity']),
            lambda f: f.reference.update(metrics_sha256='changed'),
            lambda f: f.entries.reverse(),
            lambda f: f.reviews[0].update(source_image={'file': 'other'}),
            lambda f: f.packages[look.PAIR_IDS[0]]['source_provenance'].update(scene={'file': 'other'}),
        ):
            fixture = PlanFixture()
            mutate(fixture)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                fixture.run()

    def test_dispatcher_preserves_original_direct_plan_and_only_opts_in_exact_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'configs').mkdir()
            direct = json.loads((ROOT / 'configs/smoke_e2e_reviewed_five.json').read_text())
            path = root / 'configs/plan.json'
            path.write_text(json.dumps(direct))
            def load(_, pair_id):
                row = next(row for row in direct['pairs'] if row['pair_id'] == pair_id)
                return {'pair_id': pair_id, 'kind': row['kind'], 'sources': [],
                        'source_provenance': {'scene': {'file': views.APPROVED_SCENE}}}
            real_ref = views.file_ref
            def reference(path, parent, limit):
                return real_ref(path, parent, limit) if Path(parent).name == 'configs' else {'file': str(path)}
            with patch.object(views, 'ROOT', root), patch.object(views, 'file_ref', reference), \
                    patch.object(views, 'load_reviewed_pair', load), patch.object(look, 'review_packages') as dispatch:
                result = views.review_packages(path)
                self.assertEqual([p['pair_id'] for p in result[3]], list(views.APPROVED_PAIRS))
                dispatch.assert_not_called()
                path.write_text(json.dumps(PlanFixture().plan))
                views.review_packages(path)
                dispatch.assert_called_once()
                wrong = PlanFixture().plan
                wrong['version'] = 'unrecognized_look_plan'
                path.write_text(json.dumps(wrong))
                with self.assertRaises(ValueError):
                    views.review_packages(path)

    def test_black_gate_precedes_source_loading_and_starts_no_export(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            packages = [{'pair_id': pair} for pair in look.PAIR_IDS]
            identity = {'pbs_jobid': '123.hopper-m-02', 'scratch': str(root / 'scratch'),
                        'cpu_affinity': list(range(12)), 'cuda_visible_devices': 'fixture-only'}
            for name, value in (('ROOT', root), ('execution_environment', lambda *_: identity),
                ('install_signals', lambda: None)):
                stack.enter_context(patch.object(views, name, value))
            loader = stack.enter_context(patch.object(views, 'review_packages',
                return_value=({}, {}, [], packages, [])))
            stack.enter_context(patch.object(views.os, 'sched_getaffinity', return_value=set(range(12)), create=True))
            stack.enter_context(patch.object(views.signal, 'setitimer'))
            stack.enter_context(patch.object(views.sys, 'argv', ['fixture', '--name', 'smoke_e2e_look_fixture',
                '--review-plan', 'configs/smoke_e2e_look_five.json', '--timeout-seconds', '600',
                '--pair-timeout-seconds', '120', '--gate-timeout-seconds', '30']))
            gate = {'_meta': {'exit_code': 86}, 'metrics': {'cleanup_errors': []}}
            stack.enter_context(patch.object(views, 'logged_gate', return_value=(gate, {}, [])))
            worker = stack.enter_context(patch.object(views, 'worker_result'))
            self.assertEqual(views.main(), 86)
            worker.assert_not_called()
            loader.assert_not_called()
            saved = json.loads((root / 'results/smoke_e2e_look_fixture.json').read_text())
            self.assertEqual(saved['config']['approved_pair_ids'], [])
            self.assertFalse(saved['metrics']['step1_passed'])

    def test_real_synthetic_look_table_replay_records_late_tau_without_formal_certificate(self):
        table = look_table()
        self.assertTrue(table.states[table.initial_state].shared)
        tables = {table.pair_id: table}
        certificate = {'tables': {table.pair_id: table.manifest_reference}}
        packages = {table.pair_id: {'t01': {'witnesses': [
            {'actions': [{'kind': 'turn_right'}, {'kind': 'select', 'target_name': target}]}
            for target in ('A', 'B')]}}}
        with patch.object(scope, 'authorize_smoke', return_value=certificate):
            result = replay.replay(tables, packages, {}, {}, set(), certificate)
        self.assertTrue(result['witness_replay_passed'])
        self.assertFalse(result['coverage']['all_selected_pairs_tau0'])
        self.assertIsNone(result['coverage']['T3_correctness_claim'])
        self.assertIsNone(result['formal_T02_acceptance'])
        for row in result['world_replays']:
            self.assertEqual((row['kind'], row['tau'], row['x_seen_step'], row['steps']), ('look_flip', 1, 1, 2))
            self.assertIsNone(row['trace'][0]['tau'])
        self.assertTrue(table.synthetic)
        self.assertFalse(table.cache_verified)

    def test_three_evaluation_strategies_keep_actual_look_temporal_logs(self):
        table = look_table()
        certificate = {'tables': {table.pair_id: table.manifest_reference}}
        backends = []
        try:
            for strategy in evaluation.STRATEGIES:
                environments = []
                for world in ('w0', 'w1'):
                    with patch.object(scope, 'authorize_smoke', return_value=certificate):
                        backend = scope.SmokeTableBackend({table.pair_id: table}, certificate=certificate,
                                                         pair_checks={}, report={}, authenticated_ids=set())
                    backends.append(backend)
                    environments.append((table.pair_id, world, scope.SmokeTableEnvironment(backend)))
                def policy(contexts):
                    result = []
                    for index, context in enumerate(contexts):
                        action = ('turn_right' if strategy == 'diag_base' and not context['responses'] else 'select 1')
                        result.append({'response': action, 'action': action, 'forced_observation': False,
                            'response_token_ids': [1], 'response_token_count': 1,
                            'sample_id': f'fixture-{strategy}-{len(context["responses"])}-{index}'})
                    return result
                rows = evaluation.collect_strategy(strategy, environments, policy, lambda _: None)
                self.assertEqual(len(rows), 2)
                for row in rows:
                    self.assertEqual(row['pair_kind'], 'look_flip')
                    self.assertIsNone(row['trace'][0]['tau_before'])
                    self.assertEqual(row['tau'], None if strategy == 'obs_never' else 1)
                    self.assertEqual(row['x_seen_step'], None if strategy == 'obs_never' else 1)
                    self.assertTrue(all('selected_object' in step and 'success' in step for step in row['trace']))
        finally:
            for backend in backends:
                backend.close()


if __name__ == '__main__':
    unittest.main()

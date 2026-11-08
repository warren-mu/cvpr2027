"""Synthetic CPU control-flow fixtures; no Unity, GPU, or research measurements."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.envs import pair_v2_isolated_shard as isolated


def position(x=0):
    return {'x': x, 'y': 0.0, 'z': 0.0}


class Store:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.fail_setup = None

    def save_json(self, name, value):
        if name.startswith('attempt_setup_') and self.fail_setup is not None:
            raise self.fail_setup
        raw = isolated.engine.encoded(value)
        path = self.directory / name
        with path.open('xb') as stream:
            stream.write(raw)
        return {'file': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}


class Sim:
    def __init__(self, directory):
        self.store = Store(directory)
        self.query_log, self.calls = [], []
        self.objects = [{'name': name, 'pickupable': True, 'moveable': False,
                         'position': position(index), 'rotation': position()}
                        for index, name in enumerate(('A', 'B', 'X', 'Y'))]
        self.agent = {'position': position(), 'rotation': position(), 'horizon': 0.0, 'standing': True}
        self.count, self.error, self.restore_flag, self.actual_delta = 0, None, True, 0

    def baseline(self):
        value = {'scene': 'FloorPlan1', 'key': 'common', 'objects': copy.deepcopy(self.objects),
                 'agent': copy.deepcopy(self.agent)}
        return {**value, 'reference': self.store.save_json('baseline.json', value)}

    def restore_snapshot(self, baseline):
        self.calls.append('restore_common')
        self.objects, self.agent = copy.deepcopy(baseline['objects']), copy.deepcopy(baseline['agent'])
        self.objects[2]['position']['x'] += self.actual_delta
        self.count += 1
        ref = self.store.save_json(f'action_{self.count:06d}.json', {'actual_objects': self.objects})
        self.query_log.append({'action': 'synthetic_restore', 'reference': ref})
        if self.error is not None:
            raise self.error
        return None, {'restore_deterministic': self.restore_flag}

    def snapshot(self, key):
        self.calls.append('actual_snapshot')
        value = {'scene': 'FloorPlan1', 'objects': copy.deepcopy(self.objects), 'agent': copy.deepcopy(self.agent)}
        return {**value, 'reference': self.store.save_json(key + '.json', value)}

    def action(self, action, **arguments):
        self.count += 1
        ref = self.store.save_json(f'action_{self.count:06d}.json', {'action': action, 'arguments': arguments})
        self.query_log.append({'action': action, 'reference': ref, 'exception': None})
        return None


def invoke(sim, baseline, original, number=0, anchor='X'):
    return isolated.isolated_build(original, sim, {'scene': 'FloorPlan1'}, baseline,
        {'candidate_names': ['A', 'B'], 'anchor_name': anchor}, {'mode': 'flip'},
        {'position': position(), 'yaw': 0, 'horizon': 0}, f'floorplan1_{number:05d}', None, [])


class AttemptIsolationTests(unittest.TestCase):
    def test_every_resumed_or_alternating_attempt_resets_common_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            sim = Sim(directory)
            baseline = sim.baseline()
            # A single lazy iterator is resumed, interleaved with another identity.
            first, second = iter(('X', 'X')), iter(('Y',))
            anchors = [next(first), next(second), next(first)]
            def original(sim, cfg, common, identities, *args, attempt_row):
                self.assertEqual(sim.objects, baseline['objects'])
                sim.calls.append('position_agent')
                sim.agent['rotation']['y'] = 45
                sim.calls.append('native_w0')
                sim.objects[2]['position']['x'] = 99
                actual_w0 = copy.deepcopy(sim.objects)
                camera = copy.deepcopy(sim.agent)
                sim.calls.append('complete_w0')
                # No outer setup is allowed to run inside the frozen builder.
                self.assertEqual(sim.objects, actual_w0)
                self.assertEqual(sim.agent, camera)
                sim.calls.append('native_w1')
                sim.objects[3]['position']['x'] = -99
                attempt_row.update(status='rejected', discard_reasons=['synthetic_native_outcome'])
                return attempt_row
            for number, anchor in enumerate(anchors):
                result = invoke(sim, baseline, original, number, anchor)
                self.assertTrue(result['attempt_setup']['setup_passed'])
                self.assertTrue(Path(result['attempt_setup_reference']['file']).exists())
            self.assertEqual(sim.calls, ['restore_common', 'actual_snapshot', 'position_agent',
                                         'native_w0', 'complete_w0', 'native_w1'] * 3)

    def test_mutating_native_exception_does_not_leak_to_next_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            sim = Sim(directory)
            baseline = sim.baseline()
            def failing(sim, *args, attempt_row):
                sim.objects[3]['position']['x'] = 80
                raise RuntimeError('synthetic native obstruction')
            with self.assertRaisesRegex(RuntimeError, 'obstruction'):
                invoke(sim, baseline, failing)
            seen = []
            def following(sim, *args, attempt_row):
                seen.append(copy.deepcopy(sim.objects))
                return attempt_row
            invoke(sim, baseline, following, 1)
            self.assertEqual(seen, [baseline['objects']])

    def test_false_or_unknown_restore_is_a_rejected_attempt_not_scene_exception(self):
        for flag in (False, None):
            with tempfile.TemporaryDirectory() as directory:
                sim = Sim(directory)
                baseline = sim.baseline()
                sim.restore_flag = flag
                original = unittest.mock.Mock()
                row = invoke(sim, baseline, original)
                original.assert_not_called()
                self.assertEqual(row['discard_reasons'], ['attempt_baseline_restore_mismatch'])
                self.assertNotIn('physical_w0', row)
                self.assertFalse(row['attempt_setup']['setup_passed'])

    def test_actual_state_is_rechecked_even_when_restore_claims_true(self):
        with tempfile.TemporaryDirectory() as directory:
            sim = Sim(directory)
            baseline = sim.baseline()
            sim.actual_delta = .02
            original = unittest.mock.Mock()
            row = invoke(sim, baseline, original)
            original.assert_not_called()
            self.assertFalse(row['attempt_setup']['independent_state']['restore_deterministic'])

    def test_strict_signed_zero_agent_rule_is_not_relaxed(self):
        with tempfile.TemporaryDirectory() as directory:
            sim = Sim(directory)
            baseline = sim.baseline()
            old = sim.restore_snapshot
            def restore(value):
                result = old(value)
                sim.agent['horizon'] = -0.0
                return result
            sim.restore_snapshot = restore
            row = invoke(sim, baseline, unittest.mock.Mock())
            self.assertEqual(row['discard_reasons'], ['attempt_baseline_restore_mismatch'])
            self.assertFalse(row['attempt_setup']['independent_state']['agent_pose_exact'])

    def test_ordinary_restore_error_retains_native_action_reference_and_rejects(self):
        with tempfile.TemporaryDirectory() as directory:
            sim = Sim(directory)
            baseline = sim.baseline()
            sim.error = RuntimeError('synthetic camera restore mismatch')
            row = invoke(sim, baseline, unittest.mock.Mock())
            self.assertEqual(row['discard_reasons'], ['attempt_baseline_restore_error'])
            record = json.loads(Path(row['attempt_setup_reference']['file']).read_text())
            self.assertIn('camera restore mismatch', record['error'])
            self.assertEqual(len(record['action_references']), 1)

    def test_black_and_deadline_propagate_without_becoming_ordinary_rejection(self):
        for error in (isolated.engine.BlackFrame({'file': 'synthetic_black'}), TimeoutError('synthetic deadline')):
            with tempfile.TemporaryDirectory() as directory:
                sim = Sim(directory)
                baseline = sim.baseline()
                sim.error = error
                with self.assertRaises(type(error)) as caught:
                    invoke(sim, baseline, unittest.mock.Mock())
                self.assertIs(caught.exception, error)
                record = json.loads((Path(directory) / 'attempt_setup_floorplan1_00000.json').read_text())
                self.assertEqual(record['status'], 'interrupted')
                self.assertEqual(len(record['action_references']), 1)

    def test_proof_write_failure_blocks_construction_and_preserves_original_black(self):
        with tempfile.TemporaryDirectory() as directory:
            sim = Sim(directory)
            baseline = sim.baseline()
            sim.store.fail_setup = OSError('synthetic cap')
            original = unittest.mock.Mock()
            row = invoke(sim, baseline, original)
            original.assert_not_called()
            self.assertEqual(row['discard_reasons'], ['attempt_baseline_evidence_unavailable'])
            self.assertIn('proof_save_error', row['attempt_setup'])
            black = isolated.engine.BlackFrame({'file': 'synthetic_black'})
            sim.error = black
            with self.assertRaises(isolated.engine.BlackFrame) as caught:
                invoke(sim, baseline, original, 1)
            self.assertIs(caught.exception, black)

    def test_changed_baseline_bytes_fail_before_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            sim = Sim(directory)
            baseline = sim.baseline()
            baseline['objects'][0]['position']['x'] = 5
            row = invoke(sim, baseline, unittest.mock.Mock())
            self.assertEqual(row['discard_reasons'], ['attempt_baseline_restore_error'])
            self.assertEqual(sim.calls, [])

    def test_hooks_and_old_policies_restored_after_worker_exception(self):
        config = isolated.enable_config({'name': 'fixture', 'scene': 'FloorPlan1', 'seed': 0})
        old = isolated.base.build_pair, isolated.stable.compact_attempt, isolated.search.summary, isolated.base.export_pair
        original_factory = isolated.previous.roles.view_expanded_proposals
        def worker(cfg, connection):
            self.assertEqual(cfg['solvability_version'], isolated.previous.policy.SOLVABILITY_VERSION)
            self.assertEqual(cfg['generation_version'], isolated.previous.roles.ROLE_VERSION)
            self.assertIsNot(isolated.base.build_pair, old[0])
            self.assertIs(isolated.previous.roles.view_expanded_proposals,
                          isolated.coverage.coverage_view_expanded_proposals)
            raise RuntimeError('synthetic downstream failure')
        with patch.object(isolated, '_GENERATOR', worker), self.assertRaisesRegex(RuntimeError, 'downstream'):
            isolated.generator_worker(config, None)
        self.assertEqual(old, (isolated.base.build_pair, isolated.stable.compact_attempt, isolated.search.summary,
                              isolated.base.export_pair))
        self.assertIs(isolated.previous.roles.view_expanded_proposals, original_factory)

    def test_coverage_config_summary_and_result_are_explicit_only_inside_worker(self):
        config = isolated.enable_config({'name': 'fixture', 'scene': 'FloorPlan1', 'seed': 0})
        self.assertEqual(config['proposal_order_policy'], isolated.coverage.ORDER_POLICY)
        original_factory = isolated.previous.roles.view_expanded_proposals
        def worker(cfg, channel):
            self.assertIs(isolated.previous.roles.view_expanded_proposals,
                          isolated.coverage.coverage_view_expanded_proposals)
            result = isolated.search.summary({}, cfg)
            self.assertEqual(result['proposal_order_version'], isolated.coverage.ORDER_VERSION)
            return {}
        with patch.object(isolated.search, 'summary', lambda state, cfg: {}), \
                patch.object(isolated, '_GENERATOR', worker):
            result = isolated.generator_worker(config, None)
        self.assertEqual(result['proposal_order_policy'], config['proposal_order_policy'])
        self.assertIs(isolated.previous.roles.view_expanded_proposals, original_factory)

    def test_native_prefix_saved_before_original_export_and_only_once(self):
        with tempfile.TemporaryDirectory() as directory:
            sim = Sim(directory)
            baseline = sim.baseline()
            original_method = sim.action
            def original(sim, *args, attempt_row):
                sim.action('TeleportFull', position=position())
                sim.action('PausePhysicsAutoSim')
                sim.action('PlaceObjectAtPoint', objectId='X')
                first = copy.deepcopy(attempt_row['attempt_setup_native_prefix_reference'])
                proof = json.loads(Path(first['file']).read_text())
                self.assertEqual([a['action'] for a in proof['action_references']],
                    ['synthetic_restore', 'TeleportFull', 'PausePhysicsAutoSim', 'PlaceObjectAtPoint'])
                sim.action('SetObjectPoses')
                sim.action('PlaceObjectAtPoint', objectId='X')
                self.assertEqual(attempt_row['attempt_setup_native_prefix_reference'], first)
                return attempt_row
            invoke(sim, baseline, original)
            self.assertEqual(sim.action, original_method)
            self.assertNotIn('action', vars(sim))

    def test_main_writer_keeps_explicit_new_identity_even_before_worker(self):
        captured = []
        def main():
            isolated.base.write_result('fixture', {}, {})
            return 1
        old = isolated.previous.generator_worker, isolated.previous.enable_config
        with patch.object(isolated.previous, 'main', main), \
                patch.object(isolated.base, 'write_result', lambda n, m, c: captured.append((m, c))):
            self.assertEqual(isolated.main(), 1)
        self.assertEqual(old, (isolated.previous.generator_worker, isolated.previous.enable_config))
        metrics, config = captured[0]
        self.assertEqual(config['producer_module'], 'src.envs.pair_v2_isolated_shard')
        self.assertEqual(metrics['attempt_setup_policy'], isolated.ATTEMPT_SETUP_POLICY)
        self.assertEqual(config['attempt_setup_version'], metrics['attempt_setup_version'])


if __name__ == '__main__':
    unittest.main()

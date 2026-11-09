"""Synthetic approved-scope boundaries. No simulator or scientific outcomes."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.envs import pair_v2_policy as old
from src.envs import pair_v2_stable as stable
from src.envs import pair_v2_nearby_physics as nearby
from src.envs import pair_v2_marked_shard as marked
from src.envs import pair_v2_g01_policy as policy
from src.envs import pair_v2_simulator as engine


def point(x):
    return {'x': x, 'y': 0., 'z': 0.}


def obj(name, x, receptacle=False):
    return {'name': name, 'objectType': 'Table' if receptacle else 'Book',
            'objectId': name + '_id', 'position': point(x), 'rotation': point(0),
            'pickupable': not receptacle, 'moveable': False, 'receptacle': receptacle,
            'parentReceptacles': ['table_id']}


def baseline():
    return [obj('X', 0), obj('near', .5), obj('far', 5), obj('future', 2.3),
            obj('table', 50, True), obj('table2', 60, True)]


REF = {'file': 'original/baseline.json', 'sha256': 'a'*64}


def construction(base, before, position, *, shifts=None, support='table', error=0, rest=True, agent=True):
    realized = copy.deepcopy(before)
    by_name = {o['name']: o for o in realized}
    by_name['X']['position'] = point(position + error)
    by_name['X']['parentReceptacles'] = [support + '_id']
    for name, amount in (shifts or {}).items():
        by_name[name]['position']['x'] += amount
    requested = [{'objectName': o['name'], 'position': point(position) if o['name']=='X' else copy.deepcopy(o['position']),
                  'rotation': copy.deepcopy(o['rotation'])} for o in before if o['pickupable']]
    evidence = {'X': {'action': 'GetSpawnCoordinatesAboveReceptacle', 'success': True,
                'selected_position': point(position), 'coordinates': [point(position)], 'receptacle_name': support}}
    checked = old.physical_check(before, requested, realized, ['X'], native_is_scene_at_rest=rest,
                                 placement_evidence_by_name=evidence)
    common = stable.common_baseline_check(base, realized, ['X'], REF)
    raw = {'before_objects': copy.deepcopy(before), 'requested_poses': requested, 'realized_objects': realized,
           'edited_names': ['X'], 'native_is_scene_at_rest': rest, 'placement_evidence_by_name': evidence,
           'pose_measurements': engine.pose_measurements(before, realized),
           'settling': {'settled': rest, 'observations': [{'native_is_scene_at_rest': rest}]},
           'agent_unchanged': agent, 'agent_before': {'position': point(0)},
           'agent_after': {'position': point(0 if agent else 1)},
           'edited': {'X': {'position_error_m': engine.distance(point(position), by_name['X']['position']), 'parent_membership': True}},
           'common_baseline_check': common, 'common_baseline_check_reference': {'file': 'check', 'sha256': 'b'*64}}
    raw['valid_physical'] = bool(checked['valid_physical'] and common['passed'] and agent)
    return raw


def pair(*, first=None, second=None, p0=0, p1=1, support1='table'):
    base = baseline()
    w0 = construction(base, base, p0, **(first or {}))
    w1 = construction(base, w0['realized_objects'], p1, support=support1, **(second or {}))
    return base, [w0, w1]


class NearbyPhysicsTests(unittest.TestCase):
    def test_far_drift_is_recorded_original_rule_false_without_mutating_evidence(self):
        base, worlds = pair(first={'shifts': {'far': .08}})
        saved = copy.deepcopy(worlds)
        result = nearby.measure_pair(worlds, base, REF)
        self.assertTrue(all(m['valid_physical'] for m in result['worlds']))
        self.assertFalse(result['pair']['passes_original_physics_rule'])
        self.assertAlmostEqual(result['pair']['far_object_drift_max_m'], .08)
        self.assertEqual(worlds, saved)
        far = next(r for r in result['pair']['far_drift_objects'] if r['name']=='far')
        self.assertEqual(far['distance_to_requested_X_m_by_world'], [5., 4.])
        self.assertEqual(far['object_type'], 'Book')

    def test_radius_boundary_baseline_not_after_and_support_both_worlds(self):
        for name, p1, support in [('near', 1, 'table'), ('future', 2, 'table'), ('table2', 1, 'table2')]:
            with self.subTest(name=name):
                base, worlds = pair(first={'shifts': {name: .011}}, p1=p1, support1=support)
                result = nearby.measure_pair(worlds, base, REF)
                self.assertIn(name, result['pair']['nearby_physics_context']['nearby_object_names'])
                self.assertFalse(result['worlds'][0]['valid_physical'])
        base, worlds = pair(first={'shifts': {'near': -.02}})
        # baseline .5 is near even though realized .48 would differ from a future reference.
        self.assertFalse(nearby.measure_pair(worlds, base, REF)['worlds'][0]['valid_physical'])
        base[1]['position']['x'] = .5000001
        worlds = [construction(base, base, 0, shifts={'near': .011})]
        worlds.append(construction(base, worlds[0]['realized_objects'], -1))
        self.assertTrue(nearby.measure_pair(worlds, base, REF)['worlds'][0]['valid_physical'])

    def test_saved_original_constructor_bool_cannot_fabricate_old_rule_failure(self):
        base, worlds = pair()
        worlds[0]['valid_physical'] = False
        with self.assertRaisesRegex(ValueError, 'constructor verdict'):
            nearby.measure_pair(worlds, base, REF)

    def test_two_world_final_requested_pivots_not_surface_or_preferred_points(self):
        base, worlds = pair(first={'shifts': {'future': .03}}, p1=2)
        result = nearby.measure_pair(worlds, base, REF)
        self.assertFalse(result['worlds'][0]['valid_physical'])
        self.assertEqual(result['pair']['nearby_physics_context']['requested_anchor_positions'], [point(0), point(2)])
        # Authenticated original physics still controls legal placement: no invented success.
        worlds[1]['placement_evidence_by_name']['X']['selected_position'] = point(100)
        self.assertFalse(nearby.measure_pair(worlds, base, REF)['worlds'][1]['valid_physical'])

    def test_anchor_rest_parent_agent_and_original_common_proof_are_not_relaxed(self):
        for kwargs in ({'error': .02001}, {'rest': False}, {'agent': False}):
            base, worlds = pair(first=kwargs)
            self.assertFalse(nearby.measure_pair(worlds, base, REF)['worlds'][0]['valid_physical'])
        base, worlds = pair()
        worlds[0]['placement_evidence_by_name']['X']['success'] = False
        self.assertFalse(nearby.measure_pair(worlds, base, REF)['worlds'][0]['valid_physical'])
        worlds[0]['common_baseline_check']['passed'] = False
        with self.assertRaises(ValueError): nearby.measure_pair(worlds, base, REF)

    def test_runtime_rechecks_w0_after_actual_w1_fallback_and_keeps_same_measurement_object(self):
        base, worlds = pair(first={'shifts': {'future': .03}}, p1=2)
        original_worlds = copy.deepcopy(worlds)
        class Sim:
            clone_context = None
            scene_preflight = {'snapshot': {'objects': base, 'reference': REF}}
            def construct_world(self, start, key, placements, require_agent=None):
                index = 0 if key.endswith('_w0') else 1
                raw = copy.deepcopy(worlds[index])
                return {'passed': raw['valid_physical'], 'physical': raw, 'snapshot': {'objects': raw['realized_objects']}}
        sim = Sim(); row = {}; old_aug = marked.augment_physical_result
        with nearby.pair_context(sim, row, nearby.fields()):
            first = sim.construct_world({}, 'pair_w0', [])
            measurement = marked.augment_physical_result({}, first['physical'])
            self.assertTrue(first['passed']); self.assertTrue(measurement['nonanchor_gate_pending'])
            second = sim.construct_world({}, 'pair_w1', [])
            self.assertFalse(second['passed']); self.assertFalse(first['passed'])
            self.assertFalse(measurement['valid_physical']); self.assertFalse(measurement['nonanchor_gate_pending'])
            self.assertIs(measurement, first['physical']['nearby_physics_measurement'])
            self.assertTrue(row['nearby_physics_context']['finalized'])
            self.assertFalse(row['passes_original_physics_rule'])
        self.assertIs(marked.augment_physical_result, old_aug)
        self.assertNotIn('construct_world', vars(sim))
        self.assertEqual(worlds, original_worlds)

    def test_native_failure_or_incomplete_failure_is_not_promoted_or_rewritten(self):
        base = baseline()
        for outcome in ('raise', {'passed': False, 'physical': {}}, {'passed': False, 'physical': {'placement_evidence_by_name': {}}}):
            class Sim:
                clone_context = None
                scene_preflight = {'snapshot': {'objects': base, 'reference': REF}}
                def construct_world(self, *args, **kwargs):
                    if outcome == 'raise': raise RuntimeError('original native exhaustion')
                    return outcome
            sim = Sim(); row = {}; old_aug = marked.augment_physical_result
            with nearby.pair_context(sim, row, nearby.fields()):
                if outcome == 'raise':
                    with self.assertRaisesRegex(RuntimeError, 'original native exhaustion'):
                        sim.construct_world({}, 'pair_w0', [])
                else:
                    self.assertIs(sim.construct_world({}, 'pair_w0', []), outcome)
                    self.assertIsNone(row['passes_original_physics_rule'])
            self.assertIs(marked.augment_physical_result, old_aug)

    def test_w1_native_failure_does_not_publish_pending_w0_as_pass(self):
        base, worlds = pair(first={'shifts': {'far': .03}})
        class Sim:
            clone_context = None
            scene_preflight = {'snapshot': {'objects': base, 'reference': REF}}
            def construct_world(self, start, key, placements, require_agent=None):
                if key.endswith('_w1'): raise RuntimeError('original native failure')
                return {'passed': False, 'physical': copy.deepcopy(worlds[0])}
        sim = Sim(); row = {}
        with self.assertRaisesRegex(RuntimeError, 'original native failure'):
            with nearby.pair_context(sim, row, nearby.fields()):
                first = sim.construct_world({}, 'pair_w0', [])
                row['physical_w0'] = marked.augment_physical_result({}, first['physical'])
                self.assertTrue(row['physical_w0']['valid_physical'])
                sim.construct_world({}, 'pair_w1', [])
        self.assertIsNone(row['physical_w0']['valid_physical'])
        self.assertIsNone(row['passes_original_physics_rule'])
        self.assertIsNone(row['far_object_drift_max_m'])

    def test_old_sources_unchanged_explicit_new_version_and_smoke_authorization(self):
        for partition in ('train', 'test'):
            previous = policy.fields(partition)
            self.assertNotIn('nearby_physics_version', previous)
            new = policy.fields(partition, nearby=True)
            self.assertEqual(new['g01_version'], previous['g01_version'] + '_nearby_v1')
            nearby.validate_fields(new)
        smoke = policy.fields('test', True, True, True, True)
        self.assertEqual(smoke['g01_policy']['scene_timeout_seconds'], 600)
        self.assertFalse(smoke['g01_engineering_smoke']['formal_collection_eligible'])
        with self.assertRaises(ValueError): policy.fields('train', True, False, True, True)
        with self.assertRaises(ValueError): nearby.validate_fields({})

    def test_original_flag_true_and_summary_does_not_call_physics_pass_acceptance(self):
        base, worlds = pair()
        value = nearby.measure_pair(worlds, base, REF)
        self.assertTrue(value['pair']['passes_original_physics_rule'])
        counts = dict.fromkeys(('counted_attempts', 'finalized_two_worlds', 'pending_or_unmeasured',
            'both_worlds_pass_new_physics', 'pass_new_but_fail_original_physics',
            'exported_original_rule_true', 'exported_original_rule_false'), 0)
        row = {**value['pair'], 'counts_as_pair_attempt': True, 'status': 'rejected',
               'physical_w0': value['worlds'][0], 'physical_w1': value['worlds'][1]}
        from src.envs import pair_v2_g01_shard as shard
        row = shard.compact_attempt(row)
        nearby.update_counts(counts, row)
        self.assertEqual(counts['both_worlds_pass_new_physics'], 1)
        self.assertEqual(counts['exported_original_rule_true'], 0)
        nearby.update_counts(counts, {'counts_as_pair_attempt': True})
        self.assertEqual(counts['pending_or_unmeasured'], 1)


if __name__ == '__main__': unittest.main()

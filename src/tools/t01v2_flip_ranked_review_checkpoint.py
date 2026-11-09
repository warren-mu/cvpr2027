"""Independent used-point geometry proof for the explicit supplement15 source.

The original source/journal/Controller checks and full scientific pair checker
remain mandatory. Unused proposal diagnostics are not independent acceptance
measurements. Only the new Flip role predicate replaces the historical native
surface-point ratio; Stay and all measured physics/image/target gates are kept.
"""
import copy
import json
from pathlib import Path
import re
from contextlib import contextmanager
from . import t01v2_flip_review_checkpoint as original
from .t01v2_posthoc_solvability import verified_passed_rows
from src.envs import pair_v2_flip_ranked_policy as policy
from src.envs import pair_v2_spawn_sharing as sharing
from src.envs import pair_v2_bbox_rank as rank
from src.envs import pair_v2_flip_geometry as geometry
from src.envs import pair_v2_projection_pool as pools
from src.envs import pair_v2_simulator as engine
from src.envs import pair_v2_native_retry as retry
from src.envs.pair_simulator import path_complete, point
from src.envs.pair_v2_construct import adjacent_moves

merge, roles = original.merge, original.roles
same = original.association.same


class IndexedReader:
    """Only bytes belonging to the already authenticated pair Controller."""
    def __init__(self, proof, *, root):
        self.root = Path(root).resolve()
        index_path = merge.reference_path(proof['controller_index_reference'], root=self.root)
        self.directory = index_path.parent
        index = merge.read_reference(proof['controller_index_reference'], root=self.root)
        self.files, self.cache = {}, {}
        for ref in index['files']:
            path = (self.root / ref['file']).resolve()
            if not path.is_relative_to(self.directory) or path in self.files:
                raise ValueError('Duplicate/foreign Controller index entry')
            self.files[path] = ref

    def action(self, index):
        path = self.directory / f'action_{index:06d}.json'
        if path not in self.files:
            raise ValueError('Missing original action in Controller index')
        return self(self.files[path])

    def __call__(self, ref):
        path = merge.reference_path(ref, root=self.root, within=self.directory)
        indexed = self.files.get(path)
        if indexed is None or indexed['sha256'] != ref['sha256']:
            raise ValueError('Geometry reference absent from authenticated Controller index')
        if 'bytes' in indexed and indexed['bytes'] != path.stat().st_size:
            raise ValueError('Geometry reference differs from indexed byte length')
        key = (path, ref['sha256'])
        if key not in self.cache:
            self.cache[key] = json.loads(path.read_text())
        return self.cache[key]


def _require_same(actual, expected, message):
    if not same(actual, expected):
        raise ValueError(message)


def _native_token(value, read, scene, camera, view):
    """Reconstruct compact references from the literal authenticated query."""
    if 'spawn_reference' in value:
        compact = value['spawn_reference']
        query = read(compact['query_reference'])
        spawn = {**copy.deepcopy(compact), 'coordinates': query['after_metadata'].get('actionReturn')}
        token = {'position': copy.deepcopy(value['position']),
                 'receptacle_name': value['receptacle_name'], 'spawn': spawn}
    else:
        token = geometry.as_token(value)
        spawn = token['spawn']
        query = read(spawn['query_reference'])
    engine.validate_spawn({'spawn': spawn}, scene)
    origin = query
    while origin.get('action') == 'GetSpawnCoordinatesAboveReceptacle':
        origin = read.action(origin['index'] - 1)
    expected_view = {'position': view['position'], 'rotation': {'x': 0, 'y': view['yaw'], 'z': 0},
                     'horizon': view['horizon'], 'standing': True, 'forceAction': False}
    if (origin.get('action') != 'TeleportFull' or origin.get('exception') is not None
            or origin.get('arguments') != expected_view
            or origin['after_metadata'].get('lastActionSuccess') is not True
            or (origin.get('capture') or {}).get('files', {}).get('metadata') != camera['source_reference']):
        raise ValueError('Native pool camera lacks the original requested-view capture')
    _require_same(origin['after_metadata'], read(camera['source_reference']),
                  'Pool camera metadata differs from original view action')
    receptor = engine.objects_by_name(query['before_metadata']['objects'])[token['receptacle_name']]
    expected = {'action': 'GetSpawnCoordinatesAboveReceptacle', 'success': True,
        'scene': scene, 'receptacle_name': receptor['name'], 'receptacle_id_at_query': receptor['objectId'],
        'receptacle_position_at_query': receptor['position'], 'selected_position': token['position'],
        'coordinates': query['after_metadata'].get('actionReturn'), 'query_reference': spawn['query_reference'],
        'source': engine.SOURCE + 'PhysicsRemoteFPSAgentController.cs#L3423',
        'scope': 'native receptacle samples; placement requires subsequent settling/membership checks'}
    _require_same(spawn, expected, 'Native token differs from its literal original query')
    if (query.get('action') != 'GetSpawnCoordinatesAboveReceptacle' or query.get('exception') is not None
            or query['after_metadata'].get('lastActionSuccess') is not True or receptor.get('receptacle') is not True
            or query.get('arguments') != {'objectId': receptor['objectId'], 'anywhere': True}):
        raise ValueError('Missing successful original native coordinate query')
    for metadata in (query['before_metadata'], query['after_metadata']):
        _require_same(pools.camera_from_metadata(metadata, camera['source_reference']), camera,
                      'Native query used a different actual camera')
    return token


def _observation_cameras(saved, camera, read, config):
    actions = [entry['action'] for entry in saved]
    if actions[:2] != [{'kind': 'turn_left'}, {'kind': 'turn_right'}]:
        raise ValueError('Original one-action turn inventory changed')
    indices = set()
    for action in actions[2:]:
        path = action['path']
        query = read(path['query_reference'])
        tolerance = float(config.get('path_endpoint_tolerance_m', .05))
        target, start = action['position'], camera['agent_position']
        if (action.get('kind') != 'move' or type(action.get('index')) is not int
                or action['index'] not in range(1, 5) or action['index'] in indices
                or adjacent_moves([target], start) != [target]):
            raise ValueError('Predicted one-step move is outside original adjacent action domain')
        indices.add(action['index'])
        if (query.get('action') != 'GetShortestPathToPoint' or query.get('exception') is not None
                or query['after_metadata'].get('lastActionSuccess') is not True
                or query.get('arguments') != {'position': start, 'target': target, 'allowedError': tolerance}):
            raise ValueError('One-step move lacks its original native path query')
        raw = query['after_metadata'].get('actionReturn')
        expected = {'source': start, 'target': target, 'raw': raw,
            'complete': path_complete(raw, point(start), point(target), tolerance),
            'query_reference': path['query_reference'], 'endpoint_tolerance_m': tolerance}
        _require_same(path, expected, 'One-step path declaration differs from original native response')
        if expected['complete'] is not True:
            raise ValueError('One-step path was not natively complete')
    rebuilt = [pools.camera_after_action(camera, action) for action in actions]
    _require_same(saved, rebuilt, 'Observation cameras differ from original action transformations')
    return rebuilt


def _portable(row, pair_path, basename, root):
    matches = [ref for ref in row['portable_files'] if Path(ref['file']).name == basename]
    if len(matches) != 1:
        raise ValueError('Missing/duplicate portable construction: ' + basename)
    return merge.read_reference(matches[0], root=root, within=Path(pair_path).resolve().parent)


def verify_used_geometry(row, pair_path, baseline, read, config, *, root):
    """Recompute only used points, binding originals rather than trusting tags."""
    for module in (sharing, rank, geometry):
        module.validate_fields(row)
    if (row.get('producer_module') != policy.PRODUCER or row.get('flip_unit_version') != policy.VERSION
            or row.get('flip_unit_policy') != policy.POLICY or row.get('kind') != 'look_flip'):
        raise ValueError('Complete geometry verifier requires the explicit new v2 Look Flip source')
    return verify_geometry_evidence(row, pair_path, baseline, read, config, root=root)


def verify_geometry_evidence(row, pair_path, baseline, read, config, *, root):
    """Shared proposal proof, after the caller authenticates its own producer.

    This verifies used native inputs and complete tuple/retry evidence only.
    It does not select a producer, authorize a source, or accept an actual class.
    Both callers must still execute their full original scientific checker.
    """
    for module in (sharing, rank, geometry):
        module.validate_fields(row)
    pool = read(row['projection_pool_reference'])
    for module in (sharing, rank, geometry):
        module.validate_fields(pool)
    objects = engine.objects_by_name(baseline['objects'])
    names, anchor_name = row['identities']['candidate_names'], row['identities']['anchor_name']
    if names != sorted(set(names)) or len(names) != 2:
        raise ValueError('Geometry needs unchanged stable A/B names')
    a, b, anchor = objects[names[0]], objects[names[1]], objects[anchor_name]
    context = {'candidate_a': {'name': a['name'], 'position': a['position']},
        'candidate_b': {'name': b['name'], 'position': b['position']},
        'anchor': {'name': anchor['name'], 'axisAlignedBoundingBox': anchor['axisAlignedBoundingBox']},
        'coordinate_evidence': geometry.POLICY['storage']}
    _require_same(pool['flip_geometry_context'], context, 'Pool geometry differs from exact Controller baseline')
    camera_ref = pool['camera']['source_reference']
    camera = pools.camera_from_metadata(read(camera_ref), camera_ref)
    _require_same(pool['camera'], camera, 'Pool camera differs from original indexed native metadata')
    _require_same(row['view'], pool['view'], 'Pair and proposal-pool views differ')
    if (camera['vertical_fov_degrees'] != config['field_of_view']
            or camera['width'] != 255 or camera['height'] != 255):
        raise ValueError('Pool camera does not use declared FOV and image dimensions')
    cameras = _observation_cameras(pool['observation_cameras'], camera, read, config)
    # Expand just used entries. Unused catalog totals remain source diagnostics.
    checked = {**pool, 'camera': camera, 'observation_cameras': cameras,
               'flip_geometry_context': context, 'points': {'look': {'A': [], 'B': []}}}
    token_cache = {}
    def certify(value, side):
        token = _native_token(value, read, row['scene'], camera, pool['view'])
        reference = geometry.native_reference(token)
        matches = [entry for entry in pool['points']['look'][side]
                   if same(entry['token'], reference)]
        if len(matches) != 1:
            raise ValueError('Used native point absent/duplicated in declared role pool')
        prediction = geometry.predict(token, a, b, anchor, camera, cameras)
        if not prediction['proposal_geometry_passed'] or prediction['distance_role'] != side:
            raise ValueError('Used point fails complete predicted-centre geometry')
        key = json.dumps(reference, sort_keys=True)
        if key not in token_cache:
            checked['points']['look'][side].append({'token': token, 'native_key': prediction['native_key']})
            token_cache[key] = token
        return token
    declared = row['proposal_role']
    sides = [declared[world + '_source_side'] for world in ('w0', 'w1')]
    initial = [certify(row['flip_geometry_initial_tuple']['worlds'][world]['native_reference'], side)
               for world, side in zip(('w0', 'w1'), sides)]
    final = [certify(row['proposal'][world], side) for world, side in zip(('w0', 'w1'), sides)]
    initial_proof, final_proof = (geometry.complete_pair(checked, *tokens, source_sides=sides)
                                for tokens in (initial, final))
    _require_same(row['flip_geometry_initial_tuple'], initial_proof, 'Initial complete-tuple prediction was altered')
    _require_same(row['flip_geometry_final_tuple'], final_proof, 'Final complete-tuple prediction was altered')
    counts = {}
    for world_index, world in enumerate(('w0', 'w1')):
        construction = _portable(row, pair_path, 'construction_' + world + '.json', root)
        request = read(construction['request_reference'])
        records = request['native_retry_records']
        if set(records) != {anchor_name}:
            raise ValueError('Construction retry inventory differs from sole edited anchor')
        _require_same(row['native_retry_records'][world], records, 'Pair retry summary differs from construction')
        record = records[anchor_name]
        ledger = read(record['ledger_reference'])
        _require_same(ledger, {key: value for key, value in record.items() if key != 'ledger_reference'},
                      'Construction retries differ from original indexed ledger')
        attempts = record['attempts']
        if (record.get('version') != retry.NATIVE_RETRY_VERSION or record.get('policy') != retry.NATIVE_RETRY_POLICY
                or record.get('scene') != row['scene'] or record.get('object_name') != anchor_name
                or record.get('record_key') != row['pair_id'] + '_' + world + '_0'
                or record.get('status') != 'placed' or record.get('fresh_controller_required') is not False
                or record.get('pair_attempt_counter_modified') is not False
                or not 1 <= len(attempts) <= 40 or record.get('attempted_point_count') != len(attempts)
                or record.get('failed_point_count') != len(attempts) - 1
                or record.get('max_points') != 40
                or not len(attempts) <= record.get('supplied_point_count', 0)):
            raise ValueError('Accepted world lacks complete bounded successful native retry ledger')
        source_anchor = engine.objects_by_name(construction['before_objects'])[anchor_name]
        _require_same(record['rotation'], source_anchor['rotation'], 'Native retry rotation differs from source world anchor')
        expected_guards, previous_action = [], None
        opposite = initial[1] if world == 'w0' else final[0]
        for index, attempt in enumerate(attempts):
            token = certify(attempt['spawn_token'], sides[world_index])
            geometry.validate_placement(checked, token, sides[world_index], opposite=opposite,
                                        opposite_role=sides[1 - world_index])
            action = read(attempt['action_reference'])
            before, after = action['before_metadata'], action['after_metadata']
            target = engine.objects_by_name(before['objects'])[anchor_name]
            arguments = {'objectId': target['objectId'], 'position': token['position'],
                         'rotation': record['rotation'], 'forceKinematic': True}
            expected_success = index == len(attempts) - 1
            if (attempt.get('token_index') != index or attempt.get('native_success') is not expected_success
                    or action.get('action') != 'PlaceObjectAtPoint' or action.get('index') != attempt['action_index']
                    or after.get('lastActionSuccess') is not expected_success
                    or action.get('exception') != attempt.get('error')
                    or (expected_success and action.get('exception') is not None)
                    or (action.get('capture') or {}).get('valid') is not True
                    or (action.get('capture') or {}).get('black') is not False
                    or (action.get('capture') or {}).get('source_action') != 'PlaceObjectAtPoint'
                    or (action.get('capture') or {}).get('action_index') != action['index']):
                raise ValueError('Retry entry differs from its actual native placement outcome')
            for actual in (action['arguments'], attempt['arguments']):
                _require_same(actual, arguments, 'Retry dispatched a different native point/anchor/rotation')
            for key, metadata in (('before_object_poses', before), ('after_object_poses', after)):
                _require_same(attempt[key], retry.object_pose_state(metadata), 'Retry pose evidence differs from native action')
            query = read(token['spawn']['query_reference'])
            if query['index'] >= action['index']:
                raise ValueError('Placement token query did not precede actual use')
            if previous_action is not None:
                if action['index'] != previous_action['index'] + 1:
                    raise ValueError('Unrecorded native action between placement retries')
                _require_same(before, previous_action['after_metadata'], 'Retry action chain is discontinuous')
            if not expected_success:
                allowed_error = 'RuntimeError: PlaceObjectAtPoint: ' + str(after.get('errorMessage'))
                if action.get('exception') not in (None, allowed_error):
                    raise ValueError('Retry failure is not the original explicit native placement error')
                unchanged = retry.unchanged_objects(before, after)
                _require_same(attempt['unchanged_check'], unchanged, 'Failed native retry lost exact unchanged-state proof')
                if unchanged['unchanged'] is not True:
                    raise ValueError('Failed native retry changed objects')
            expected_guards.append({'native_reference': geometry.native_reference(token),
                'opposite_reference': geometry.native_reference(opposite),
                'source_side': sides[world_index], 'opposite_side': sides[1 - world_index]})
            previous_action = action
        _require_same(attempts[0]['spawn_token'], initial[world_index]['spawn'], 'First native attempt differs from initial tuple')
        for actual in (record['successful_token'], attempts[-1]['spawn_token']):
            _require_same(actual, final[world_index]['spawn'], 'Final tuple differs from actual successful native token')
        placement = construction['placement_evidence_by_name'][anchor_name]
        _require_same({key: value for key, value in placement.items() if key not in ('requested_position', 'geometry')},
                      final[world_index]['spawn'], 'Physical construction used a different final native token')
        _require_same(placement['geometry']['placement_reference'], record['action_reference'],
                      'Physical construction does not bind final retry action')
        _require_same(record['action_reference'], attempts[-1]['action_reference'], 'Successful native action reference differs')
        _require_same(record['arguments'], attempts[-1]['arguments'], 'Successful native arguments differ')
        _require_same(row['flip_geometry_native_guards'][world], expected_guards,
                      'Actual retry geometry guards missing, reordered or bound to wrong opposite point')
        counts[world] = len(attempts)
    return {'passed': True, 'version': geometry.VERSION, 'initial_tuple': initial_proof,
        'final_tuple': final_proof, 'verified_native_attempts_by_world': counts,
        'candidate_positions': {name: objects[name]['position'] for name in names},
        'scope': 'independent used-point proposal geometry and native retry provenance; full scientific checks remain mandatory',
        'unused_pool_diagnostics_revalidated': False}


def verify_role(row, positions, proof):
    """Original seed orientation; only predicted-centre proposal roles change."""
    names = row['identities']['candidate_names']
    _require_same({name: positions[name] for name in names}, proof['candidate_positions'],
                  'Scientific checker baseline differs from geometry baseline')
    match = re.fullmatch(re.escape(row['scene'].lower()) + r'_([0-9]{5})', row['pair_id'])
    if match is None or row.get('mode') != 'flip' or row['proposal'].get('mode') != 'flip':
        raise ValueError('Geometry role proof needs explicit Flip attempt identity')
    canonical = copy.deepcopy(row['proposal'])
    if roles.flip_orientation(row['scene'], int(match.group(1)), 0)['flip_swapped']:
        canonical['w0'], canonical['w1'] = canonical['w1'], canonical['w0']
    expected_proposal, expected = roles.orient_proposal(canonical, row['scene'], int(match.group(1)), 0)
    expected.update(candidate_a_name=names[0], candidate_b_name=names[1])
    if (row.get('generation_version') != roles.ROLE_VERSION or row.get('generation_policy') != roles.ROLE_POLICY
            or expected != row['proposal_role'] or expected_proposal != row['proposal']):
        raise ValueError('Recorded Flip assignment differs from unchanged seed0 rule')
    sides = proof['final_tuple']['source_sides']
    if sides != [expected['w0_source_side'], expected['w1_source_side']]:
        raise ValueError('Predicted-centre role differs from unchanged seed assignment')
    targets = [row.get('target_w0'), row.get('target_w1')]
    if any(target not in names for target in targets):
        raise ValueError('Accepted row lacks actual target identities')
    actual = ['A' if target == names[0] else 'B' for target in targets]
    return {'passed': True, 'intended_candidate_roles': sides, 'actual_candidate_roles': actual,
        'actual_targets_match_intended': actual == sides, 'assignment_digest': expected['assignment_digest'],
        'scope': geometry.POLICY['scope'], 'flip_search_version': original.search.VERSION,
        'flip_geometry_version': geometry.VERSION}


@contextmanager
def source_support():
    previous_policy = original.policy
    try:
        original.policy = policy
        with original.source_support():
            previous_load, previous_check = merge.load_source, merge.check_pair
            loaded = {}
            def load(path):
                data, reference = previous_load(path)
                if original.is_flip_source(data['config']):
                    loaded[data['config']['raw_evidence_dir']] = (data, reference)
                return data, reference
            def check(row, path, **kwargs):
                if row.get('flip_search_version') is None and row.get('producer_module') != policy.PRODUCER:
                    return previous_check(row, path, **kwargs)
                if (row.get('producer_module') != policy.PRODUCER or row.get('flip_unit_version') != policy.VERSION
                        or row.get('flip_unit_policy') != policy.POLICY):
                    raise ValueError('Historical Flip source cannot claim new v2 complete geometry')
                for module in (sharing, rank, geometry):
                    module.validate_fields(row)
                if row.get('mode') != 'flip':
                    return previous_check(row, path, **kwargs)  # Stay unchanged.
                data, reference = loaded[kwargs['raw_root']]
                eligible, _, passed = verified_passed_rows(data)
                if not eligible['eligible']:
                    raise ValueError('Geometry pair comes from ineligible ended source')
                matches = [entry for entry in passed if Path(entry[2]).resolve() == Path(path).resolve()]
                if len(matches) != 1:
                    raise ValueError('Geometry pair absent/duplicated in authenticated passed journal')
                journal, pair_ref, _ = matches[0]
                _require_same(row, merge.read_reference(pair_ref, within=Path(path).resolve().parent),
                              'Geometry row differs from authenticated portable pair')
                prior_association = original.association.policy
                try:
                    original.association.policy = policy
                    association = original.association.verify_pair_controller(data, reference, journal, row, path)
                finally:
                    original.association.policy = prior_association
                read = IndexedReader(association, root=merge.ROOT)
                proof = verify_used_geometry(row, path, read(association['baseline_reference']), read,
                                             data['config'], root=merge.ROOT)
                previous_role = roles.verify_proposal_role
                def role(actual, positions):
                    if actual is row:
                        return verify_role(actual, positions, proof)
                    return previous_role(actual, positions)
                try:
                    roles.verify_proposal_role = role
                    result = previous_check(row, path, **kwargs)
                finally:
                    roles.verify_proposal_role = previous_role
                return {**result, 'complete_flip_geometry_verification': proof}
            merge.load_source, merge.check_pair = load, check
            try:
                yield
            finally:
                merge.load_source, merge.check_pair = previous_load, previous_check
    finally:
        original.policy = previous_policy


def main(argv=None):
    prior = original.original.collect_sources, original.controller.policy, original.association.policy
    with source_support():
        try:
            original.original.collect_sources = original.controller.collect_sources
            original.controller.policy = original.association.policy = policy
            return original.original.main(argv)
        finally:
            original.original.collect_sources, original.controller.policy, original.association.policy = prior


if __name__ == '__main__':
    raise SystemExit(main())

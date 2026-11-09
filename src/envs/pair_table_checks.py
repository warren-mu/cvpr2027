"""CPU checks over explicit tables; no renderer or inferred missing outcomes."""
from collections import deque
from dataclasses import replace
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import time

from .pair_table import encoded, eligible_names, classify_views


def minimum_world_turns(table, state_id, world, *, already_seen=False, max_turns=5):
    """Observe until own X was seen, then reach a marked target and select."""
    seen = already_seen or table.states[state_id].anchor_pixels[world] >= 250
    queue, visited = deque([(state_id, seen, 0)]), {(state_id, seen)}
    while queue:
        current, saw, distance = queue.popleft()
        if distance + 1 <= max_turns and saw and table.target_names[world] in table.selectable_names(current, world):
            return distance + 1
        if distance + 2 > max_turns:
            continue
        for following in table.states[current].transitions.values():
            following_seen = saw or table.states[following].anchor_pixels[world] >= 250
            key = (following, following_seen)
            if key not in visited:
                visited.add(key)
                queue.append((following, following_seen, distance + 1))
    return None


def min_solution_turns(table):
    """Belief-prefix BFS: shared actions until first fork, separate continuations.

    Counts select as one turn and minimizes the worse world. Returning to a
    shared state after forking does not merge histories or reset 'seen'.
    """
    initial = table.initial_state
    if table.kind.startswith('direct_'):
        values = [minimum_world_turns(table, initial, w, already_seen=True) for w in (0, 1)]
        return max(values) if None not in values else None
    queue, visited, candidates = deque([(initial, 0)]), {initial}, []
    while queue:
        state_id, depth = queue.popleft()
        state = table.states[state_id]
        if not state.shared:
            costs = [minimum_world_turns(table, state_id, w, max_turns=5 - depth) for w in (0, 1)]
            if None not in costs:
                candidates.append(depth + max(costs))
            continue
        if depth >= 4:
            continue
        for following in state.transitions.values():
            if following not in visited:
                visited.add(following)
                queue.append((following, depth + 1))
    return min(candidates) if candidates else None


def visible_first_actions(table, world):
    start = table.states[table.initial_state]
    return sorted(action for action, destination in start.transitions.items()
                  if table.states[destination].anchor_pixels[world] >= 250)


def replay_witness(table, world, witness):
    """Replay names and literal destination positions; never legacy move indices."""
    actions = witness.get('actions')
    if not isinstance(actions, list):
        raise ValueError('Witness actions unavailable')
    if len(actions) > 5:
        return {'applicable': False, 'passed': None, 'reason': 'original_witness_exceeds_D_plus_1'}
    state_id, trace, selected = table.initial_state, [], None
    for index, action in enumerate(actions):
        state = table.states[state_id]
        kind = action.get('kind')
        if kind == 'select':
            name = action.get('target_name') or action.get('selected_object_name')
            selected = name
            success = name == table.target_names[world] and name in table.selectable_names(state_id, world)
            trace.append({'turn': index + 1, 'state_id': state_id, 'action':
                f'select {table.labels[name]}' if name in table.labels else None, 'object_name': name})
            return {'applicable': True, 'passed': success and index == len(actions) - 1, 'trace': trace}
        if kind == 'move':
            destination = action.get('position')
            matches = [text for text, target in state.transitions.items() if text.startswith('move ')
                       and table.positions[table.states[target].view.viewpoint_id] == destination]
            if len(matches) != 1:
                return {'applicable': True, 'passed': False, 'trace': trace, 'reason': 'witness_destination_not_available'}
            text = matches[0]
        elif kind in ('turn_left', 'turn_right'):
            text = kind
        else:
            return {'applicable': True, 'passed': False, 'trace': trace, 'reason': 'unknown_witness_action'}
        if text not in state.transitions or index >= 4:
            return {'applicable': True, 'passed': False, 'trace': trace, 'reason': 'witness_observation_not_available'}
        state_id = state.transitions[text]
        trace.append({'turn': index + 1, 'action': text, 'state_id': state_id})
    return {'applicable': True, 'passed': False, 'trace': trace, 'reason': 'witness_did_not_select'}


def check_pair(table, t01):
    """Checks depend on explicit authenticated T01 records, not saved pass labels."""
    if t01.get('pair_id') != table.pair_id:
        raise ValueError('T01/table pair identity mismatch')
    start = table.states[table.initial_state]
    reasons, observations = [], {}
    if table.kind.startswith('look_'):
        if not start.shared:
            reasons.append('look_start_not_shared')
        first = t01.get('selected_action')
        destination = start.transitions.get(first)
        if destination is None or table.states[destination].shared:
            reasons.append('look_selected_first_action_not_fork')
        for world in (0, 1):
            measured = visible_first_actions(table, world)
            original = t01.get(f'anchor_visible_actions_w{world}')
            observations[f'w{world}'] = measured
            if not measured or not isinstance(original, list) or sorted(original) != measured:
                reasons.append(f't01_first_step_visibility_mismatch_w{world}')
    elif start.shared or any(pixels < 250 for pixels in start.anchor_pixels):
        reasons.append('direct_start_anchor_not_seen_both_worlds')
    minimum = min_solution_turns(table)
    if minimum is None or minimum > 5:
        reasons.append('not_solvable_within_budget')
    witnesses = t01.get('witnesses')
    if not isinstance(witnesses, list) or len(witnesses) != 2:
        raise ValueError('Both T01 witnesses required')
    replays = [replay_witness(table, world, witnesses[world]) for world in (0, 1)]
    if any(row['passed'] is False for row in replays):
        reasons.append('witness_replay_failed')
    return {'passed': not reasons, 'reasons': reasons, 'min_solution_turns': minimum,
            'non_anchor_divergent_states': sum(state.anchor_pixels == (0, 0) and state.frame_max_abs_diff > 3
                                              for state in table.states.values()),
            'first_step_visible_actions': observations, 'witness_replay': replays,
            'prefix_verified_identical': table.export_verified and all(
                state.frame_max_abs_diff <= 3 and state.anchor_pixels == (0, 0)
                for state in table.states.values() if state.shared),
            'synthetic_only': table.synthetic, 'cache_matches_live': None if table.synthetic else table.cache_verified,
            'task_acceptance': None}


def verified_read(reference, roots):
    path = Path(reference['file']).resolve()
    if not any(path.is_relative_to(Path(root).resolve()) for root in roots):
        raise ValueError('Raw table proof escaped explicit evidence roots')
    if path.stat().st_size > 48 * 1024 * 1024:
        raise ValueError('Oversized raw table proof')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != reference['sha256']:
        raise ValueError('Raw table proof changed')
    return data


def pose_restore_measurements(metadata, snapshot):
    """Read-only all-object pose comparison; numeric signed zeros are equal."""
    objects = {obj['name']: obj for obj in metadata['objects']}
    expected = {obj['name']: obj for obj in snapshot['objects']}
    if (len(objects) != len(metadata['objects']) or len(expected) != len(snapshot['objects'])
            or set(objects) != set(expected)):
        raise ValueError('Native metadata inventory differs from the declared snapshot')
    positions, rotations, position_mismatches, rotation_mismatches = [], [], [], []
    for name, target in expected.items():
        actual = objects[name]
        for key in ('position', 'rotation'):
            if any(type(obj[key].get(axis)) not in (int, float) or not math.isfinite(obj[key][axis])
                   for obj in (actual, target) for axis in 'xyz'):
                raise ValueError('Missing/nonfinite reported object pose')
        positions.append(math.sqrt(sum((actual['position'][axis] - target['position'][axis]) ** 2 for axis in 'xyz')))
        rotations.extend(abs(actual['rotation'][axis] - target['rotation'][axis]) for axis in 'xyz')
        if actual['position'] != target['position']:
            position_mismatches.append(name)
        if actual['rotation'] != target['rotation']:
            rotation_mismatches.append(name)
    return {'all_object_positions_exact': not position_mismatches,
            'all_object_rotations_exact': not rotation_mismatches,
            'max_position_delta_m': max(positions, default=0.),
            'max_raw_rotation_component_delta_degrees': max(rotations, default=0.),
            'position_mismatch_names': sorted(position_mismatches),
            'rotation_mismatch_names': sorted(rotation_mismatches),
            'comparison': 'exact finite numeric xyz; no epsilon; raw Euler components; signed zeros equal'}


def frozen_non_anchor_measurements(left, right, anchor_name):
    """Report actual readback differences; approved acceptance thresholds stay."""
    names = [{obj['name'] for obj in metadata['objects']} for metadata in (left, right)]
    if anchor_name not in names[0] or names[0] != names[1]:
        raise ValueError('Frozen render inventory changed')
    filtered = [{'objects': [obj for obj in metadata['objects'] if obj['name'] != anchor_name]}
                for metadata in (left, right)]
    return pose_restore_measurements(filtered[1], filtered[0])


def world_metadata_matches(metadata, snapshot):
    objects = {obj['name']: obj for obj in metadata['objects']}
    expected = {obj['name']: obj for obj in snapshot['objects']}
    if len(objects) != len(metadata['objects']) or set(objects) != set(expected):
        raise ValueError('Native metadata inventory differs from the bound world snapshot')
    for name in objects:
        if any(objects[name].get(key) != expected[name].get(key) for key in ('objectType', 'moveable', 'pickupable')):
            raise ValueError('Native object identity/role changed')
    for requested in snapshot['poses']:
        actual = objects[requested['objectName']]['position']
        if math.sqrt(sum((actual[k] - requested['position'][k]) ** 2 for k in ('x', 'y', 'z'))) > .01:
            raise ValueError('Native query/capture is not in its declared world pose state')
    return pose_restore_measurements(metadata, snapshot)


def verify_agent_setup(action, state, table):
    arguments = action.get('arguments', {})
    if (action.get('action') != 'TeleportFull' or action.get('exception') is not None
            or action['after_metadata'].get('lastActionSuccess') is not True
            or arguments.get('position') != table.positions[state.view.viewpoint_id]
            or arguments.get('rotation', {}).get('y') != state.view.heading_index * 45):
        raise ValueError('Native camera setup does not execute the declared discrete viewpoint/heading')
    return action['after_metadata']['agent']


def verify_cached_binding(table, row):
    """Bind a cache report's reference pixels, masks and labels to this table."""
    if row['pair_id'] != table.pair_id or row.get('kind') != table.kind:
        raise ValueError('Cache report names another table or pair kind')
    state = table.states[row['state_id']]
    from .pair_table import label_assignment
    assignments = label_assignment(table.pair_id, table.candidate_names)
    for world in (0, 1):
        evidence = row['worlds'][f'w{world}']
        source, actual = evidence['cached_source'], state.captures[world]
        if (source['rgb_sha256'] != actual.rgb.sha256 or source['render_id'] != actual.render_id
                or source['snapshot_sha256'] != actual.source['snapshot_sha256']):
            raise ValueError('Cache report was computed from different cached capture bytes')
        expected_labels = {name: assignments[name] for name in eligible_names(actual, table.candidate_names)}
        if evidence['cached_labels'] != expected_labels or any(
                evidence['masks'][name]['cached_mask_sha256'] != hashlib.sha256(actual.masks[name]).hexdigest()
                for name in (table.anchor_name, *table.candidate_names)):
            raise ValueError('Cache report used different masks or candidate labels')


def verify_paused_setup(evidence, read):
    sequence = evidence.get('paused_setup_actions')
    expected = ['PausePhysicsAutoSim', 'SetObjectPoses', 'PhysicsSyncTransforms', 'TeleportFull']
    if not isinstance(sequence, list) or [row.get('action') for row in sequence] != expected:
        raise ValueError('Missing complete paused native setup chain')
    native = [json.loads(read(row['reference'])) for row in sequence]
    for index, (row, action) in enumerate(zip(sequence, native)):
        if (action.get('action') != row['action'] or action.get('exception') is not None
                or action.get('after_metadata', {}).get('lastActionSuccess') is not True):
            raise ValueError('Failed paused setup action')
        if index and (action.get('index') != native[index - 1].get('index') + 1
                      or action['before_metadata'] != native[index - 1]['after_metadata']):
            raise ValueError('Paused setup chain is interrupted')
    if (sequence[0]['reference'] != evidence.get('physics_pause_action')
            or sequence[-1]['reference'] != evidence.get('agent_setup_action')):
        raise ValueError('Pause/camera source is not the declared setup chain')
    return native


def verify_render_sources(table, *, evidence_roots):
    """Decode original captures and native actions, independent of table pixels.

    This certifies only the raw export chain. Cache validation and pair-level
    solvability remain separate requirements before any production rollout.
    """
    if table.synthetic:
        return {'passed': False, 'reason': 'synthetic_not_real_render_evidence'}
    import numpy as np
    from PIL import Image
    read = lambda ref: verified_read(ref, evidence_roots)
    from .pair_table_export import FROZEN_RENDER_VERSION, verify_frozen_world_request, frozen_target_measurement
    if table.source.get('frozen_render_version') != FROZEN_RENDER_VERSION:
        raise ValueError('Real table lacks its approved frozen-render policy')
    seen = set()
    for key, state in table.states.items():
        measured = classify_views(state.captures, table.anchor_name, kind=table.kind)
        if (state.shared is not measured['shared'] or state.anchor_pixels != measured['anchor_pixels']
                or state.frame_max_abs_diff != measured['frame_max_abs_diff']):
            raise ValueError('Native verification state classification differs from its pair kind')
        for world, view in enumerate(state.captures):
            src, capture = view.source, view.source['capture']
            if (src.get('pair_id') != table.pair_id or src.get('world_id') != f'w{world}'
                    or src.get('state_id') != key or view.render_id in seen):
                raise ValueError('Capture source identity mismatch')
            seen.add(view.render_id)
            identity = json.loads(read(src['process_reference']))
            if (src['process_reference']['sha256'] != view.process_id
                    or not identity.get('worker') or not identity.get('unity')):
                raise ValueError('Originating exporter/Unity process identity proof is missing')
            metadata = json.loads(read(capture['files']['metadata']))
            with Image.open(io.BytesIO(read(capture['files']['rgb']))) as image:
                rgb = np.asarray(image).copy()
            if (rgb.dtype != np.uint8 or rgb.shape != (255, 255, 3) or not np.any(rgb)
                    or rgb.tobytes() != view.rgb.data or capture.get('black') is not False
                    or capture.get('valid') is not True or capture.get('rgb_sha256') != view.rgb.sha256):
                raise ValueError('Raw render RGB not equal to stored table evidence')
            with np.load(io.BytesIO(read(capture['files']['masks'])), allow_pickle=False) as archive:
                names, masks = archive['names'].tolist(), archive['masks']
                if len(set(names)) != len(names) or masks.dtype != np.bool_ or masks.shape != (len(names), 255, 255):
                    raise ValueError('Invalid raw instance masks')
                actual_masks = {name: mask.astype('uint8').tobytes() for name, mask in zip(names, masks)}
            objects = {obj['name']: obj for obj in metadata['objects']}
            if len(objects) != len(metadata['objects']):
                raise ValueError('Duplicate raw object identities')
            for name, mask in view.masks.items():
                if name not in objects or actual_masks.get(name, bytes(255 * 255)) != mask:
                    raise ValueError('Raw mask does not match table object')
            agent = metadata['agent']
            pose = {'position': agent['position'], 'rotation': agent['rotation'],
                    'horizon': agent['cameraHorizon'], 'standing': agent['isStanding']}
            camera = {k: metadata.get(k) for k in ('cameraPosition', 'fov', 'screenWidth', 'screenHeight')}
            if encoded(view.agent) != encoded({'agent': pose, 'camera': camera}):
                raise ValueError('Camera proof differs from raw metadata')
            for name, distance in view.distances.items():
                actual = sum((pose['position'][k] - objects[name]['position'][k]) ** 2 for k in ('x', 'y', 'z')) ** .5
                if distance != actual:
                    raise ValueError('Candidate selectability distance differs from metadata')
            action = json.loads(read(src['render_action']))
            if (action.get('action') != 'Pass' or action.get('exception') is not None
                    or action.get('after_metadata') != metadata or action.get('capture') != capture):
                raise ValueError('Capture is not the actual successful render action result')
            snapshot = json.loads(read(src['snapshot_reference']))
            placement = json.loads(read(src['world_pose_action']))
            if (src.get('snapshot_sha256') != src['snapshot_reference']['sha256']
                    or placement.get('action') != 'SetObjectPoses' or placement.get('exception') is not None
                    or placement.get('arguments', {}).get('objectPoses') != snapshot.get('poses')
                    or placement.get('arguments', {}).get('placeStationary') is not True
                    or set(objects) != {o['name'] for o in snapshot['objects']}):
                raise ValueError('Capture not bound to its complete authenticated world snapshot')
            measured_restore = world_metadata_matches(metadata, snapshot)
            target_measurement = frozen_target_measurement(metadata['objects'], table.candidate_names,
                table.anchor_name, table.target_names[world], table.source.get('candidate_policy'))
            if target_measurement != src.get('frozen_target_measurement'):
                raise ValueError('Frozen target readback not reproduced from native metadata')
            if measured_restore != src.get('restore_readback'):
                raise ValueError('Saved restore readback differs from native metadata')
            setup_actions = verify_paused_setup(src, read)
            if world == 0 and (src['world_pose_action'] != src['paused_setup_actions'][1]['reference']
                    or action['index'] != setup_actions[-1]['index'] + 1
                    or action['before_metadata'] != setup_actions[-1]['after_metadata']):
                raise ValueError('w0 capture is not directly after its paused setup')
            if world == 1:
                original_refs = verify_frozen_world_request(snapshot, read)
                if (original_refs != table.source.get('render_source_worlds')
                        or original_refs[0] != state.captures[0].source['snapshot_reference']
                        or snapshot['frozen_render_recipe']['anchor_name'] != table.anchor_name):
                    raise ValueError('Frozen w1 recipe names another pair/anchor source')
            realized_agent = verify_agent_setup(json.loads(read(src['agent_setup_action'])), state, table)
            if encoded(metadata['agent']) != encoded(realized_agent):
                raise ValueError('Actual capture camera changed after its requested state setup')
            if world == 1:
                sequence = src.get('between_renders')
                if not isinstance(sequence, list) or [a['action'] for a in sequence] != ['SetObjectPoses', 'PhysicsSyncTransforms', 'Pass']:
                    raise ValueError('Paired render action chain missing or moved the agent')
                for row in sequence:
                    actual = json.loads(read(row['reference']))
                    if actual.get('action') != row['action'] or actual.get('exception') is not None:
                        raise ValueError('Unsuccessful intervening paired render action')
                left_src = state.captures[0].source
                previous = json.loads(read(left_src['render_action']))
                for row in sequence:
                    actual = json.loads(read(row['reference']))
                    if (actual['index'] != previous['index'] + 1
                            or actual['before_metadata'] != previous['after_metadata']):
                        raise ValueError('Paired paused capture sequence is interrupted')
                    previous = actual
                if (sequence[0]['reference'] != src['world_pose_action']
                        or sequence[-1]['reference'] != src['render_action']
                        or src['physics_pause_action'] != left_src['physics_pause_action']):
                    raise ValueError('w1 capture does not retain the same paused render chain')
                left_metadata = json.loads(read(left_src['capture']['files']['metadata']))
                comparison = frozen_non_anchor_measurements(left_metadata, metadata, table.anchor_name)
                if comparison != src.get('non_anchor_pose_comparison'):
                    raise ValueError('Actual non-anchor pose comparison is not reproduced')
        # Domain records must be byte-identical in their public implications;
        # build_table already refuses world-specific neighbor differences.
        if len(state.graph_evidence) != 2 or any(d.get('complete') is not True for d in state.graph_evidence):
            raise ValueError('Incomplete graph proof')
        from .pair_v2_construct import adjacent_moves
        from .pair_simulator import path_complete, point
        from .pair_v2_solvable_policy import native_path_known
        public_domains = []
        for world, domain in enumerate(state.graph_evidence):
            evidence = domain['evidence']
            setup = json.loads(read(evidence['setup']))
            setup_actions = verify_paused_setup(evidence, read)
            expected_snapshot_ref = state.captures[world].source['snapshot_reference']
            graph_snapshot_ref = evidence.get('world_snapshot_reference')
            if (evidence.get('world') != world or not isinstance(graph_snapshot_ref, dict)
                    or graph_snapshot_ref.get('sha256') != expected_snapshot_ref.get('sha256')
                    or read(graph_snapshot_ref) != read(expected_snapshot_ref)):
                raise ValueError('Native graph proof belongs to the other world')
            snapshot = json.loads(read(expected_snapshot_ref))
            world_metadata_matches({'objects': setup['objects']}, snapshot)
            placement = json.loads(read(evidence['world_pose_action']))
            if (placement.get('action') != 'SetObjectPoses' or placement.get('exception') is not None
                    or placement['arguments'].get('objectPoses') != snapshot['poses']
                    or evidence['world_pose_action'] != evidence['paused_setup_actions'][1]['reference']):
                raise ValueError('Native graph setup did not apply its authenticated world')
            if world_metadata_matches(setup_actions[2]['after_metadata'], snapshot) != evidence.get('restore_readback'):
                raise ValueError('Graph restore readback differs from its actual native sync result')
            realized_agent = verify_agent_setup(json.loads(read(evidence['agent_setup_action'])), state, table)
            actual_position = setup['agent']['position']
            if sum((actual_position[k] - table.positions[state.view.viewpoint_id][k]) ** 2
                   for k in ('x', 'y', 'z')) ** .5 > .01:
                raise ValueError('Actual graph query is not at its declared viewpoint')
            reachable_action = json.loads(read(evidence['reachable_reference']))
            if (reachable_action.get('action') != 'GetReachablePositions'
                    or reachable_action.get('exception') is not None
                    or reachable_action['after_metadata'].get('lastActionSuccess') is not True
                    or reachable_action['after_metadata'].get('actionReturn') != evidence['reachable_positions']):
                raise ValueError('Graph reachable positions lack their native action proof')
            queries = [json.loads(read(q['reference'])) for q in evidence['queries']]
            expected_points = adjacent_moves(evidence['reachable_positions'], actual_position)
            if len(queries) != len(expected_points):
                raise ValueError('Graph omitted an adjacent native query')
            available = []
            for index, (query, destination) in enumerate(zip(queries, expected_points), 1):
                args = query.get('arguments', {})
                if (query.get('action') != 'GetShortestPathToPoint' or query.get('exception') is not None
                        or args.get('position') != actual_position
                        or args.get('target') != destination
                        or query['after_metadata'].get('lastActionSuccess') is not True):
                    raise ValueError('Graph path is unknown or belongs to another endpoint')
                world_metadata_matches(query['before_metadata'], snapshot)
                if encoded(query['before_metadata']['agent']) != encoded(realized_agent):
                    raise ValueError('Native path query camera differs from its authenticated state setup')
                tolerance = args.get('allowedError')
                if type(tolerance) not in (float, int) or tolerance != .05:
                    raise ValueError('Graph changed the existing native path tolerance')
                raw_path = query['after_metadata'].get('actionReturn')
                if not native_path_known(raw_path):
                    raise ValueError('Malformed native path is unknown, not an unavailable move')
                if path_complete(raw_path, point(args['position']), point(destination), tolerance):
                    available.append((index, destination))
            declared = [(move['index'], move['position']) for move in domain['moves']]
            if available != declared:
                raise ValueError('Graph availability differs from actual native paths')
            public_domains.append(available)
        if public_domains[0] != public_domains[1]:
            raise ValueError('graph_mismatch')
    # Independently rebuild the full bounded state graph from verified native
    # domains; omitted states/edges must not make a hard example seem solvable.
    from .pair_core import ViewState
    from .pair_table_export import point_id, state_id, directions
    pending, reached = deque([(table.initial_state, 0)]), {table.initial_state}
    while pending:
        key, depth = pending.popleft()
        state = table.states[key]
        moves = state.graph_evidence[0]['moves']
        expected_neighbors = directions(table.positions[state.view.viewpoint_id], state.view.heading_index, moves)
        if state.neighbors != expected_neighbors:
            raise ValueError('Public neighbor text differs from actual native graph')
        destinations = {'turn_left': state_id(ViewState(state.view.viewpoint_id, (state.view.heading_index-1) % 8)),
                        'turn_right': state_id(ViewState(state.view.viewpoint_id, (state.view.heading_index+1) % 8))}
        destinations.update({f'move {m["index"]}': state_id(ViewState(point_id(m['position']), state.view.heading_index)) for m in moves})
        required = {a: target for a, target in destinations.items() if depth < 4 or target in table.states}
        if state.transitions != required:
            raise ValueError('Table omitted or changed an in-budget native graph transition')
        if depth < 4:
            for target in destinations.values():
                if target not in table.states:
                    raise ValueError('D4 table is missing an actual reachable state')
                if target not in reached:
                    reached.add(target)
                    pending.append((target, depth+1))
    if reached != set(table.states):
        raise ValueError('Table includes states outside the fixed D4 graph')
    table.export_verified = True
    return {'passed': True, 'render_count': len(seen), 'prefix_verified_identical': True,
            'scope': 'raw export only; cache and solvability checks remain required'}


def authorize_tables(tables, *, pair_checks, cache_report, authenticated_cache_source_ids=None):
    """Dataset-wide stop: never silently filter away a failed cache validation."""
    from .pair_table_cache import require_cache_validation
    try:
        require_cache_validation(cache_report)
    except ValueError as error:
        raise ValueError('cache_validation_failed_or_pending: stop all downstream use') from error
    from .pair_table_cache import sample_state_keys
    tables = list(tables)
    population = [(table.pair_id, key) for table in tables for key in table.states]
    plan = cache_report['sampling_plan']
    if sample_state_keys(population, seed=plan['seed'], sample_size=plan['sample_size']) != plan:
        raise ValueError('Cache validation belongs to another table population')
    # The pure cache helper intentionally does not read raw files. Its caller
    # must first authenticate all fresh captures; IDs must cover the exact set.
    required_sources = {(row['pair_id'], row['state_id'], world,
                         row['worlds'][world]['fresh_source']['render_id'])
                        for row in cache_report['states'] for world in ('w0', 'w1')}
    if authenticated_cache_source_ids is None or set(authenticated_cache_source_ids) != required_sources:
        raise ValueError('Fresh cache capture source authentication is missing')
    indexed = {table.pair_id: table for table in tables}
    for row in cache_report['states']:
        table = indexed[row['pair_id']]
        verify_cached_binding(table, row)
    for table in tables:
        check = pair_checks.get(table.pair_id)
        if (table.synthetic or not table.export_verified or not isinstance(check, dict)
                or check.get('passed') is not True or check.get('synthetic_only') is not False):
            raise ValueError('Table remains unverified for production rollout')
    for table in tables:
        table.cache_verified = True


def authenticate_cache_sources(tables, samples, *, evidence_roots):
    """Check actual fresh PNG/NPZ/native source chains before accepting a sample.

    Existing native graph evidence is retained: this is a fresh capture check,
    not a claim that a new-process navigation graph was re-enumerated.
    """
    indexed = {table.pair_id: table for table in tables}
    grouped, identities = {}, set()
    for sample in samples:
        table = indexed[sample.pair_id]
        state = table.states[sample.state_id]
        if tuple(sample.cached) != tuple(state.captures) or sample.kind != table.kind:
            raise ValueError('Sample cached captures or pair kind differ from the explicit table')
        key = (sample.pair_id, sample.state_id)
        if key in grouped:
            raise ValueError('Duplicate fresh paired-state source')
        grouped[key] = sample
    for pair_id in sorted({key[0] for key in grouped}):
        checked = copy.copy(indexed[pair_id])
        checked.states = dict(checked.states)
        for (source_pair, state_id), sample in grouped.items():
            if source_pair != pair_id:
                continue
            measured = classify_views(sample.fresh, checked.anchor_name, kind=checked.kind)
            checked.states[state_id] = replace(checked.states[state_id], captures=tuple(sample.fresh),
                shared=measured['shared'], frame_max_abs_diff=measured['frame_max_abs_diff'],
                anchor_pixels=measured['anchor_pixels'])
        result = verify_render_sources(checked, evidence_roots=evidence_roots)
        if result.get('passed') is not True:
            raise ValueError('Fresh paired-state captures are not authenticated real evidence')
        for (source_pair, state_id), sample in grouped.items():
            if source_pair == pair_id:
                for world, view in enumerate(sample.fresh):
                    identities.add((pair_id, state_id, f'w{world}', view.render_id))
    return identities


def measure_lookup_throughput(environment, pair_id, world, *, steps):
    """Only call inside run_logged. CPU lookup timing, not model/render timing."""
    if type(steps) is not int or steps <= 0:
        raise ValueError('Explicit positive step count required')
    environment.reset(pair_id, world)
    started = time.perf_counter()
    episodes = 0
    for _ in range(steps):
        if environment.done:
            environment.reset(pair_id, world)
            episodes += 1
        actions = environment.frame.observation_actions
        # Fixed action selection avoids any model or policy inference here.
        action = actions[0].text() if actions and environment.steps < 4 else 'select 1'
        environment.step(action)
    elapsed = time.perf_counter() - started
    return {'steps': steps, 'elapsed_seconds': elapsed, 'environment_steps_per_second': steps / elapsed,
            'resets_during_timing': episodes, 'includes': 'lookup, public observation, reward, reset after terminal',
            'excludes': 'initial table load, initial reset, model inference, rendering',
            'synthetic_only': environment.backend.table.synthetic,
            'model_steps_per_second': None, 'twenty_times_model_requirement_passed': None}


def export_cost_estimate(records, *, target_pairs=1400):
    """Conditional extrapolation from explicitly measured complete export units.

    These are allocation GPU-hours from measured export-unit time, not PBS billed
    accounting. Callers must separately retain initialization and failed-job cost.
    """
    if type(target_pairs) is not int or target_pairs != 1400:
        raise ValueError('This task requests a fixed 1400-pair conditional estimate')
    rows = list(records)
    seen = set()
    for row in rows:
        if row.get('pair_id') in seen or not isinstance(row.get('pair_id'), str):
            raise ValueError('Explicit distinct exported pair IDs required')
        seen.add(row['pair_id'])
        for key in ('export_elapsed_seconds', 'allocated_gpus', 'stored_bytes', 'state_count'):
            value = row.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('Missing measured export cost: ' + key)
        if row.get('export_completed') is not True or row.get('synthetic_only') is not False:
            return {'estimate': None, 'reason': 'incomplete_or_synthetic_export_units', 'records': rows}
    if not rows:
        return {'estimate': None, 'reason': 'no_observed_export_units', 'records': []}
    mean_gpu_hours = sum(row['export_elapsed_seconds'] * row['allocated_gpus'] / 3600 for row in rows) / len(rows)
    mean_bytes = sum(row['stored_bytes'] for row in rows) / len(rows)
    return {'records': rows, 'target_pairs': target_pairs,
            'estimate': {'allocation_gpu_hours': mean_gpu_hours * target_pairs,
                         'storage_bytes': mean_bytes * target_pairs},
            'formula': '1400 times arithmetic mean of observed per-pair export allocation GPU-hours/bytes',
            'scope': 'conditional on measured scene/state distribution; excludes unobserved startup, queue, failure and retry costs',
            'billing_estimate': None, 'over_twenty_allocation_gpu_hours_requires_approval': mean_gpu_hours * target_pairs > 20}

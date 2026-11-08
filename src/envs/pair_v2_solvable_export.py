"""Supplement-4 portable observations/review; earlier exporters stay unchanged.

All extra observations have already been measured before the search adapter's
capacity decision. This wrapper copies authenticated bytes, adds one portable
proof index, and writes the final pair once. It never starts a simulator.
"""
import copy
import hashlib
import io
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def references(value):
    if isinstance(value, dict):
        if isinstance(value.get('file'), str) and isinstance(value.get('sha256'), str):
            yield value
        for item in value.values():
            yield from references(item)
    elif isinstance(value, list):
        for item in value:
            yield from references(item)


def read_bytes(reference, raw_root):
    path = Path(reference['file']).resolve()
    if not path.is_relative_to(Path(raw_root).resolve()):
        raise ValueError('Solvability evidence escaped its own scene raw directory')
    content = path.read_bytes()
    if len(content) > 32 * 1024 * 1024 or digest(content) != reference['sha256']:
        raise ValueError('Missing, oversized, or changed solvability evidence')
    return content


def capture_observation(capture, read, identities, instruction):
    """Read actual RGB/masks and deterministic public marks, never missing=zeros."""
    import numpy as np
    from PIL import Image
    from .pair_v2_policy import mark_observation
    files = capture['files']
    metadata = json.loads(read(files['metadata']))
    with Image.open(io.BytesIO(read(files['rgb']))) as image:
        rgb = np.asarray(image).copy()
    if rgb.dtype != np.uint8 or rgb.shape != (255, 255, 3) or not np.any(rgb):
        raise ValueError('Solvability requires an actual valid nonblack 255px RGB capture')
    with np.load(io.BytesIO(read(files['masks'])), allow_pickle=False) as archive:
        names, values = archive['names'].tolist(), archive['masks'].copy()
    if (len(set(names)) != len(names) or values.dtype != np.bool_
            or values.shape != (len(names), 255, 255)
            or not set(names) <= {obj['name'] for obj in metadata['objects']}):
        raise ValueError('Solvability masks lack unique measured object identities')
    masks = dict(zip(names, values))
    if (capture.get('valid') is not True or capture.get('black') is not False
            or capture.get('rgb_shape') != [255, 255, 3] or capture.get('rgb_sha256') != digest(rgb.tobytes())
            or capture.get('sequence_id') != metadata.get('sequenceId')
            or capture.get('mask_pixels_by_name') != {name: int(mask.sum()) for name, mask in masks.items()}):
        raise ValueError('Captured image/mask bytes disagree with their native capture record')
    return mark_observation({'rgb': rgb, 'masks_by_name': masks, 'metadata': metadata},
        identities['candidate_names'], instruction,
        public_context='Actions: move <index>; turn_left; turn_right; select <mark>.')


def prepare_portable(config, row, directory, root):
    import numpy as np
    from PIL import Image
    report = copy.deepcopy(row['observation_search'])
    files, json_files, captures, loaded = [], [], [], {}
    total = 0

    def save(content, filename):
        nonlocal total
        if total + len(content) > 64 * 1024 * 1024:
            raise ValueError('Portable solvability proof exceeded its declared byte limit')
        path = directory / filename
        with path.open('xb') as stream:
            stream.write(content)
        total += len(content)
        ref = {'file': str(path.relative_to(root)), 'sha256': digest(content)}
        files.append(ref)
        return ref

    raw = lambda ref: read_bytes(ref, config['raw_evidence_dir'])
    seen = {}
    for reference in references(report):
        if Path(reference['file']).suffix != '.json':
            continue
        previous = seen.get(reference['file'])
        if previous is not None:
            if previous != reference['sha256']:
                raise ValueError('Conflicting solvability raw JSON hashes')
            continue
        seen[reference['file']] = reference['sha256']
        ref = save(raw(reference), f'solvability_json_{len(json_files):03d}.json')
        json_files.append({'source_file': reference['file'], **ref})
    anchor = row['identities']['anchor_name']
    for action in report['actions']:
        for slot in action['worlds']:
            if slot['status'] != 'observed':
                continue
            index, world_index = action['action_index'], slot['world_index']
            obs = capture_observation(slot['capture_reference'], raw, row['identities'], row['instruction'])
            loaded[(index, world_index)] = obs
            prefix = f'solvability_a{index:02d}_w{world_index}'
            image_files = {kind: save(raw(slot['capture_reference']['files'][kind]),
                prefix + ('_rgb.png' if kind == 'rgb' else '_masks.npz')) for kind in ('rgb', 'masks')}
            buffer = io.BytesIO()
            Image.fromarray(obs['marked']).save(buffer, format='PNG')
            image_files['marked'] = save(buffer.getvalue(), prefix + '_marked.png')
            captures.append({'action_index': index, 'world_index': world_index, 'action': action['action'],
                'capture_reference': copy.deepcopy(slot['capture_reference']),
                'observation': {**{key: copy.deepcopy(obs[key]) for key in
                    ('metadata', 'marks', 'mark_labels_by_name', 'input_text', 'history')}, 'files': image_files}})
    review_views = []
    for world_index in range(2):
        action_name = report[f'anchor_visible_actions_w{world_index}'][0]
        action = next(item for item in report['actions'] if item['action'] == action_name)
        obs = loaded[(action['action_index'], world_index)]
        if int(obs['masks_by_name'].get(anchor, np.zeros((255, 255), bool)).sum()) < 250:
            raise ValueError('Review visibility assertion is not supported by its actual mask')
        review_views.append({'world_index': world_index, 'action_index': action['action_index'], 'action': action_name})
    proof = {'schema_version': 1, 'search': report, 'captures': captures, 'raw_json_files': json_files,
             'review_views': review_views,
             'scope': 'all enumerated observation captures and directly referenced native JSON; original raw retained separately'}
    proof_ref = save((json.dumps(proof, indent=2, sort_keys=True, allow_nan=False) + '\n').encode(), 'look_solvability.json')
    return files, proof_ref, review_views, loaded


def review_sheet(path, row, prefix, after, targets, review_views, loaded):
    from PIL import Image, ImageDraw
    size, top, label_height = 320, 64, 32
    sheet = Image.new('RGB', (size * 3, top + (size + label_height) * 2), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 6), row['pair_id'] + ' | ' + row['kind'], fill='black')
    draw.text((8, 25), row['instruction'], fill='black')
    for world_index in range(2):
        selected = review_views[world_index]
        shown = [prefix[world_index], after[world_index], loaded[(selected['action_index'], world_index)]]
        labels = ['v0 prefix', 'selected common observation', 'X visible: ' + selected['action']]
        for column, (obs, label) in enumerate(zip(shown, labels)):
            image = Image.fromarray(obs['marked']).resize((size, size))
            painter = ImageDraw.Draw(image)
            target_label = obs['mark_labels_by_name'].get(targets[world_index])
            for mark in obs['marks']:
                if mark['label'] == target_label:
                    painter.rectangle(tuple(round(x * size / 255) for x in mark['bounding_box']), outline='lime', width=3)
            x, y = size * column, top + (size + label_height) * world_index
            draw.text((x + 5, y + 5), f'w{world_index} / {label}', fill='black')
            sheet.paste(image, (x, y + label_height))
    with path.open('xb') as stream:
        sheet.save(stream, format='PNG')
    return {'file': str(path), 'sha256': digest(path.read_bytes())}


def export_pair(original_export, config, row, worlds, prefix, after, witnesses, constructions):
    """Scoped hooks run only in the new producer's private worker process."""
    from . import pair_v2_shard as base
    from .pair_v2_solvable_policy import SOLVABILITY_VERSION, SOLVABILITY_POLICY, solvability_summary
    row.update(solvability_version=SOLVABILITY_VERSION, solvability_policy=copy.deepcopy(SOLVABILITY_POLICY))
    if not row['kind'].startswith('look'):
        return original_export(config, row, worlds, prefix, after, witnesses, constructions)
    report = row['observation_search']
    checked = solvability_summary(report)
    if not checked['complete'] or not checked['solvable_w0'] or not checked['solvable_w1']:
        raise ValueError('Incomplete per-world observation solvability cannot be exported as passed')
    row['look_solvability'] = copy.deepcopy(report)
    original_review, original_write = base.review_sheet, base.write_json
    extra = []
    def review(path, current, pre, post, targets):
        nonlocal extra
        extra, reference, selected, loaded = prepare_portable(config, row, Path(path).parent, base.ROOT)
        row['look_solvability_reference'] = reference
        row['look_review_views'] = selected
        return review_sheet(Path(path), current, pre, post, targets, selected, loaded)
    def written(path, value):
        if Path(path).name == 'pair.json' and value is row:
            if not extra:
                raise ValueError('Extra solvability observations were not materialized before final pair write')
            value['portable_files'].extend(extra)
        return original_write(path, value)
    try:
        base.review_sheet, base.write_json = review, written
        return original_export(config, row, worlds, prefix, after, witnesses, constructions)
    finally:
        base.review_sheet, base.write_json = original_review, original_write


def verify_look_solvability(row, worlds, after, proof, read_json, read_portable):
    """Independently bind every world/action mask to its native action/domain.

The caller authenticates portable files and supplies a SHA-bound raw JSON map.
No saved `passed`, mask count, or action-membership flag is an authority here.
"""
    import numpy as np
    from .pair_v2_construct import adjacent_moves
    from .pair_simulator import path_complete, point
    from .pair_v2_policy import restore_state_check, public_exact
    from .pair_v2_solvable_policy import SOLVABILITY_VERSION, SOLVABILITY_POLICY, solvability_summary, native_path_known
    from .pair_v2_shard import action_text
    report = proof['search']
    if (proof.get('schema_version') != 1 or report != row.get('observation_search')
            or report != row.get('look_solvability') or report != row.get('evidence', {}).get('look_solvability')
            or report.get('solvability_version') != SOLVABILITY_VERSION
            or report.get('solvability_policy') != SOLVABILITY_POLICY
            or report.get('anchor_name') != row['identities']['anchor_name']):
        raise ValueError('Portable per-world solvability report/version differs from its measured pair')
    anchor = row['identities']['anchor_name']

    def restored(world_index, restoration):
        actual = read_json(restoration['restored_snapshot_reference'])
        world = worlds[world_index]
        check = restore_state_check(world['objects'], actual['objects'], world['agent'], actual['agent'])
        if check['restore_deterministic'] is not True or restoration.get('restore_deterministic') is not True:
            raise ValueError('Observation did not start from the authenticated world snapshot')
        return actual

    def check_query(reference, source, target, world_index):
        native = read_json(reference)
        expected = {'position': source, 'target': target, 'allowedError': 0.05}
        if native.get('action') != 'GetShortestPathToPoint' or native.get('arguments') != expected:
            raise ValueError('Observation domain native query arguments differ from the fixed adjacent proposal')
        before = native['before_metadata']
        bound = restore_state_check(worlds[world_index]['objects'], before['objects'],
                                    worlds[world_index]['agent'], before['agent'])
        if not bound['restore_deterministic']:
            raise ValueError('Native observation path query belongs to a different world or agent state')
        metadata = native.get('after_metadata')
        if not isinstance(metadata, dict) or type(metadata.get('lastActionSuccess')) is not bool:
            raise ValueError('Unknown native path result cannot prove action unavailability')
        if metadata['lastActionSuccess'] and not native_path_known(metadata.get('actionReturn')):
            raise ValueError('Successful path query with unknown payload is not proof of an unavailable action')
        return (native.get('exception') is None and metadata['lastActionSuccess']
                and path_complete(metadata.get('actionReturn'), point(source), point(target), 0.05))

    def public(action):
        return {key: copy.deepcopy(action[key]) for key in ('kind', 'index', 'position') if key in action}

    domains = report['domains']
    if len(domains) != 2 or [domain['world_index'] for domain in domains] != [0, 1]:
        raise ValueError('Require original action domains measured separately in both worlds')
    by_world, union = [], {}
    for index, domain in enumerate(domains):
        restored(index, domain['restoration'])
        reachable = read_json(domain['reachable_reference'])
        if (reachable.get('action') != 'GetReachablePositions' or reachable.get('exception') is not None
                or reachable['after_metadata'].get('lastActionSuccess') is not True
                or reachable['after_metadata'].get('actionReturn') != domain['reachable_positions']):
            raise ValueError('Action domain reachable positions differ from the original native query')
        expected = [{'kind': 'turn_left'}, {'kind': 'turn_right'}]
        points = adjacent_moves(domain['reachable_positions'], worlds[index]['agent']['position'])
        moves = {item['index']: item for item in domain['actions'] if item['kind'] == 'move'}
        unavailable = domain['unavailable_moves']
        used_unavailable = set()
        if len(moves) != sum(item['kind'] == 'move' for item in domain['actions']):
            raise ValueError('Duplicate adjacent move index')
        for move_index, destination in enumerate(points, 1):
            action = moves.get(move_index)
            if action is not None:
                path = action['path']
                native = read_json(path['query_reference'])
                if (action['position'] != destination or path.get('source') != worlds[index]['agent']['position']
                        or path.get('target') != destination or path.get('endpoint_tolerance_m') != 0.05
                        or path.get('raw') != native['after_metadata'].get('actionReturn')
                        or path.get('complete') is not True
                        or not check_query(path['query_reference'], worlds[index]['agent']['position'], destination, index)):
                    raise ValueError('Declared domain move lacks its native complete path')
                expected.append(public(action))
            else:
                matches = [(number, item) for number, item in enumerate(unavailable) if item.get('position') == destination]
                if len(matches) != 1:
                    raise ValueError('Missing domain move lacks a unique native unavailable observation')
                number, item = matches[0]
                reference = item.get('query_reference') or (item.get('path') or {}).get('query_reference')
                if not reference or check_query(reference, worlds[index]['agent']['position'], destination, index):
                    raise ValueError('Unavailable move is missing native evidence or was actually complete')
                native = read_json(reference)
                expected_outcome = {'action': native.get('action'), 'arguments': native.get('arguments'),
                    'exception': native.get('exception'), 'last_action_success': native['after_metadata'].get('lastActionSuccess'),
                    'action_return': native['after_metadata'].get('actionReturn')}
                if item.get('query_outcome') != expected_outcome or (item.get('path') is not None
                        and item['path'].get('raw') != expected_outcome['action_return']):
                    raise ValueError('Unavailable path copied outcome differs from authenticated native metadata')
                used_unavailable.add(number)
        if ([public(action) for action in domain['actions']] != expected
                or used_unavailable != set(range(len(unavailable)))):
            raise ValueError('Declared action domain omitted, reordered, or added an original observation action')
        domain_actions = {action_text(action): action for action in domain['actions']}
        by_world.append(domain_actions)
        for text, action in domain_actions.items():
            if text in union and union[text] != public(action):
                raise ValueError('A public action means different motion in the two world domains')
            union[text] = public(action)
    enumerated = report['enumerated_actions']
    if (len(enumerated) != len(union) or len({action_text(action) for action in enumerated}) != len(union)
            or {action_text(action): public(action) for action in enumerated} != union):
        raise ValueError('Solvability search is not the full union of both original action domains')
    captures = {(item['action_index'], item['world_index']): item for item in proof['captures']}
    if len(captures) != len(proof['captures']):
        raise ValueError('Duplicate portable action/world capture')
    seen, visible, observations = set(), [[], []], {}
    if len(report['actions']) != len(enumerated):
        raise ValueError('Not all enumerated actions were assessed')
    for index, action_row in enumerate(report['actions']):
        text = action_text(enumerated[index])
        if (action_row.get('action_index') != index or action_row.get('action') != text
                or public(action_row['requested_action']) != union[text]
                or [slot['world_index'] for slot in action_row['worlds']] != [0, 1]):
            raise ValueError('Action completion ledger differs from its enumeration')
        for world_index, slot in enumerate(action_row['worlds']):
            key = (index, world_index)
            if text not in by_world[world_index]:
                if (slot.get('status') != 'not_in_domain' or slot.get('anchor_pixels') is not None
                        or slot.get('anchor_visible') is not None or slot.get('capture_reference') is not None or key in captures):
                    raise ValueError('Unavailable action was assigned fabricated visibility or capture')
                continue
            if slot.get('status') != 'observed' or key not in captures:
                raise ValueError('An available original action lacks its actual observed capture')
            actual = restored(world_index, slot['restoration'])
            record = captures[key]
            if record['capture_reference'] != slot['capture_reference'] or record['action'] != text:
                raise ValueError('Portable capture belongs to a different action/world')
            image_files = record['observation']['files']
            def raw_capture(reference):
                if reference['file'] == slot['capture_reference']['files']['metadata']['file']:
                    return json.dumps(read_json(reference)).encode()
                kind = next(key for key in ('rgb', 'masks') if reference == slot['capture_reference']['files'][key])
                if image_files[kind]['sha256'] != reference['sha256']:
                    raise ValueError('Portable image/mask is not a byte copy of the native capture')
                return read_portable(image_files[kind])
            observation = capture_observation(slot['capture_reference'], raw_capture, row['identities'], row['instruction'])
            if any(record['observation'].get(key) != observation[key] for key in
                   ('metadata', 'marks', 'mark_labels_by_name', 'input_text', 'history')):
                raise ValueError('Portable public observation metadata/marks changed from its native capture')
            from PIL import Image
            with Image.open(io.BytesIO(read_portable(image_files['marked']))) as marked:
                if not np.array_equal(np.asarray(marked), observation['marked']):
                    raise ValueError('Portable marked observation differs from deterministic actual-image marks')
            native = read_json(slot['action_reference'])
            executed = slot['executed_action']
            if public(executed) != union[text]:
                raise ValueError('Executed motion changed the public observation action')
            if executed['kind'] == 'move':
                if not check_query(executed['path']['query_reference'], actual['agent']['position'], executed['position'], world_index):
                    raise ValueError('Executed move lacks its own restored-world native path')
                expected_action = 'TeleportFull'
                expected_args = {'position': executed['position'], 'rotation': actual['agent']['rotation'],
                    'horizon': actual['agent']['horizon'], 'standing': actual['agent']['standing'], 'forceAction': False}
            else:
                expected_action = 'RotateLeft' if executed['kind'] == 'turn_left' else 'RotateRight'
                expected_args = {'degrees': 45}
            before = native['before_metadata']
            before_check = restore_state_check(worlds[world_index]['objects'], before['objects'],
                                               worlds[world_index]['agent'], before['agent'])
            if (not before_check['restore_deterministic'] or native.get('exception') is not None
                    or native.get('action') != expected_action or native.get('arguments') != expected_args
                    or native.get('after_metadata') != observation['metadata']
                    or native['after_metadata'].get('lastActionSuccess') is not True
                    or native.get('capture') != slot['capture_reference']
                    or native.get('index') != slot['capture_reference'].get('action_index')
                    or native.get('action') != slot['capture_reference'].get('source_action')):
                raise ValueError('Observation pixels are not the recorded legal native action output')
            pixels = int(observation['masks_by_name'].get(anchor, np.zeros((255, 255), bool)).sum())
            if slot.get('anchor_pixels') != pixels or slot.get('anchor_visible') is not (pixels >= 250):
                raise ValueError('Reported visible anchor count differs from its measured instance mask')
            if pixels >= 250:
                visible[world_index].append(text)
            observations[key] = observation
            seen.add(key)
    if set(captures) != seen or report.get('complete') is not True:
        raise ValueError('Solvability capture proof is incomplete or contains unmatched frames')
    for world_index in range(2):
        if not visible[world_index] or report[f'anchor_visible_actions_w{world_index}'] != visible[world_index]:
            raise ValueError(f'look_not_solvable_w{world_index}: no independently verified visible action set')
    selected = report['selected_action']
    if any(plan[0] != selected for plan in row['witness_actions']):
        raise ValueError('Selected common observation differs from the actual public witness prefix')
    chosen = next((index for index, action in enumerate(report['actions']) if action['action'] == selected), None)
    if chosen is None or any((chosen, index) not in observations or not public_exact(observations[(chosen, index)], after[index])
                             for index in range(2)):
        raise ValueError('Chosen common evidence frame is not its authenticated all-action capture')
    for index in range(2):
        actual, saved = observations[(chosen, index)], after[index]
        if (actual['metadata'] != saved['metadata'] or set(actual['masks_by_name']) != set(saved['masks_by_name'])
                or any(not np.array_equal(mask, saved['masks_by_name'][name]) for name, mask in actual['masks_by_name'].items())):
            raise ValueError('Selected evidence metadata/masks differ from the authenticated native observation')
    expected_review = [{'world_index': index,
        'action_index': next(i for i, action in enumerate(report['actions']) if action['action'] == visible[index][0]),
        'action': visible[index][0]} for index in range(2)]
    if proof['review_views'] != expected_review or row.get('look_review_views') != expected_review:
        raise ValueError('Review visibility columns do not identify each world seeing X')
    checked = solvability_summary(report)
    if not checked['complete'] or not checked['solvable_w0'] or not checked['solvable_w1']:
        raise ValueError('Independently verified report failed the declared solvability policy')
    return {'passed': True, 'summary': checked, 'report': report, 'native_captures_verified': len(captures)}

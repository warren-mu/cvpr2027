"""Read-only authenticated T01 sources for the separately owned T02 renderer.

No Controller, filesystem output, GPU action, or human-review authorization is
created here. A checkpoint is a source of checked construction evidence only.
"""
import copy
import json
import math
from pathlib import Path
import re

from src.tools import merge_pair_v2 as merge
from src.tools.t01_partition import audited_result
from src.tools.t01v2_posthoc_solvability import verified_passed_rows


def _point(value):
    if (not isinstance(value, dict) or set(value) != set('xyz')
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in value.values())):
        raise ValueError('Require original finite xyz coordinates')
    return copy.deepcopy(value)


def canonical_start(row, worlds):
    if len(worlds) != 2 or worlds[0]['agent'] != worlds[1]['agent']:
        raise ValueError('Original worlds have different actual initial agent poses')
    commands = [world.get('agent_setup_command') for world in worlds]
    if (not isinstance(commands[0], dict) or commands[0] != commands[1]
            or set(commands[0]) != {'position', 'rotation', 'horizon', 'standing', 'forceAction'}
            or commands[0]['forceAction'] is not False or type(commands[0]['standing']) is not bool):
        raise ValueError('Require identical original canonical agent setup commands')
    command, view = commands[0], row['view']
    position, rotation = _point(command['position']), _point(command['rotation'])
    yaw = rotation['y']
    if (position != view['position'] or rotation['x'] != 0 or rotation['z'] != 0
            or yaw != view['yaw'] or command['horizon'] != view['horizon']
            or type(yaw) not in (int, float) or yaw not in tuple(index * 45 for index in range(8))
            or type(command['horizon']) not in (int, float) or not math.isfinite(command['horizon'])):
        raise ValueError('Canonical command differs from the original discrete view; no rounding allowed')
    return {'initial_position': position, 'initial_heading': int(yaw / 45),
        'agent_setup_command': copy.deepcopy(command), 'initial_actual_agent': copy.deepcopy(worlds[0]['agent'])}


def native_witness(row, world_index, replay, read):
    """Convert actual native action targets, never infer positions from move k."""
    from .pair_simulator import path_complete, point
    public = row['witness_actions'][world_index]
    actions, captures = replay['actions'], replay['captures']
    if (not public or replay.get('planned_actions') != public or replay.get('executed_actions') != public
            or len(actions) != len(public) or len(captures) != len(public)
            or replay.get('status') != 'completed' or replay.get('cleanup_errors') != []
            or replay.get('passed') is not True or replay.get('success') is not True):
        raise ValueError('Witness is not a complete successful original execution')
    mapped, previous, raw_index = [], read(captures[0]['files']['metadata']), None
    restored = read(replay['restoration']['restored_snapshot_reference'])
    agent = previous['agent']
    actual_agent = {'position': agent['position'], 'rotation': agent['rotation'],
                    'horizon': agent['cameraHorizon'], 'standing': agent['isStanding']}
    if previous['objects'] != restored['objects'] or actual_agent != restored['agent']:
        raise ValueError('Witness initial capture differs from its authenticated actual restored state')
    for index, (text, action) in enumerate(zip(public, actions)):
        if action.get('success') is not True:
            raise ValueError('Original witness contains an unsuccessful action')
        if text.startswith('select '):
            target = row[f'target_w{world_index}']
            if (re.fullmatch(r'select [1-9][0-9]*', text) is None
                    or text != f'select {action.get("label")}'
                    or index != len(public) - 1 or action.get('kind') != 'select'
                    or action.get('selected_object_name') != target or replay.get('selected_name') != target):
                raise ValueError('Witness terminal selection is not its actual target name')
            mapped.append({'kind': 'select', 'target_name': target, 'original_public_action': text})
            continue
        native = read(action['action_reference'])
        capture = captures[index + 1]
        current = read(capture['files']['metadata'])
        if (native.get('exception') is not None or native.get('capture') != capture
                or native.get('after_metadata') != current or current.get('lastActionSuccess') is not True
                or native.get('index') != capture.get('action_index')):
            raise ValueError('Witness native action is not the recorded successful capture')
        query_reference = None
        if text in ('turn_left', 'turn_right'):
            expected = 'RotateLeft' if text == 'turn_left' else 'RotateRight'
            if (action.get('kind') != text or native.get('action') != expected
                    or native.get('arguments') != {'degrees': 45} or native.get('before_metadata') != previous):
                raise ValueError('Witness turn differs from its original action/capture chain')
            translated = {'kind': text}
        elif re.fullmatch(r'move [1-9][0-9]*', text):
            before, arguments = native['before_metadata'], native['arguments']
            target = _point(arguments['position'])
            agent = previous['agent']
            expected = {'position': target, 'rotation': agent['rotation'],
                'horizon': agent['cameraHorizon'], 'standing': agent['isStanding'], 'forceAction': False}
            if action.get('kind') != 'move' or native.get('action') != 'TeleportFull' or arguments != expected:
                raise ValueError('Witness move lacks its literal native target and preserved camera')
            if raw_index is None:
                raw_index = read(replay['raw_evidence'])
            query_path = Path(action['action_reference']['file']).with_name(f'action_{native["index"] - 1:06d}.json')
            matches = [ref for ref in raw_index['files'] if Path(ref['file']) == query_path]
            if len(matches) != 1:
                raise ValueError('Witness move lacks a unique preceding native path certificate')
            query_reference = matches[0]
            query = read(query_reference)
            start = agent['position']
            if (query.get('action') != 'GetShortestPathToPoint' or query.get('exception') is not None
                    or query.get('arguments') != {'position': start, 'target': target, 'allowedError': .05}
                    or query.get('before_metadata') != previous or query.get('after_metadata') != before
                    or before.get('lastActionSuccess') is not True
                    or not path_complete(before.get('actionReturn'), point(start), point(target), .05)):
                raise ValueError('Witness move is not bound to its actual complete path from the prior capture')
            translated = {'kind': 'move', 'position': target}
        else:
            raise ValueError('Unsupported original witness action')
        mapped.append({**translated, 'original_public_action': text,
            'action_reference': copy.deepcopy(action['action_reference']),
            'capture_reference': copy.deepcopy(capture), 'path_reference': query_reference})
        previous = current
    if mapped[-1]['kind'] != 'select':
        raise ValueError('Witness has no terminal select')
    return {'actions': mapped, 'original_public_actions': copy.deepcopy(public),
            'move_mapping': 'literal authenticated native target; original public index is diagnostic only'}


def _container(path):
    data, reference = audited_result(path)
    meta, measured, cfg = data['_meta'], data['metrics'], data['config']
    name = Path(reference['file']).stem
    if (meta.get('exit_code') != 0 or not meta.get('end') or meta.get('name') != name or cfg.get('name') != name
            or any('could not run git' in item for item in reference['audit'].get('warn', []))):
        raise ValueError('Require a successfully ended audited checkpoint or final pilot')
    if measured.get('kind') == 't01v2_review_checkpoint':
        module = 'src.tools.t01v2_review_checkpoint'
        if (measured.get('status') != 'checkpoint_ready_for_human_review' or measured.get('errors') != []
                or measured.get('final_pilot_acceptance') is not False):
            raise ValueError('Checkpoint did not publish independently checked candidates')
        entries = measured['candidate_pairs']
    elif measured.get('kind') == 'pair_v2_merged_pilot':
        module = 'src.tools.merge_pair_v2'
        if measured.get('status') != 'pilot_ready_for_review' or measured.get('acceptance_passed') is not True:
            raise ValueError('Final pilot did not complete its independent checks')
        entries = measured['selected_pairs']
        dataset = merge.reference_path(measured['dataset'], within=merge.ROOT / 'data/world_pairs')
        dataset_rows = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
        if [item.get('merge_source') for item in dataset_rows] != entries:
            raise ValueError('Final dataset differs from its hashed selected-pair list')
    else:
        raise ValueError('Source is neither a final pilot nor an independent review checkpoint')
    arguments = merge.producer_arguments(meta.get('command'), module)
    declared_name = merge.option(arguments, '--name') if '--name' in arguments else (
        'pilot_pairs_v2' if module == 'src.tools.merge_pair_v2' else None)
    if declared_name != name:
        raise ValueError('Container producer/name mismatch')
    if not isinstance(entries, list) or not entries:
        raise ValueError('No published candidate pairs')
    return data, reference, entries


def load_reviewed_pair(source_result, pair_id, *, pair_file=None):
    """Return immutable evidence inputs; caller separately owns review/GPU gates.

    ``pair_file`` disambiguates repeated pair IDs across explicit source runs.
    A Direct pair has no measured all-action catalogue: reachable=None means
    the renderer must make and authenticate a fresh native query, not invent it.
    """
    container, container_ref, entries = _container(source_result)
    matches = [entry for entry in entries if entry.get('pair_id') == pair_id
               and (pair_file is None or entry['pair_reference']['file'] == pair_file)]
    if len(matches) != 1:
        raise ValueError('Require exactly one explicitly identified published pair')
    entry = matches[0]
    data, source_ref = merge.load_source(merge.ROOT / entry['source_file'])
    cfg, measured = data['config'], data['metrics']
    if (source_ref['metrics_sha256'] != entry['source_metrics_sha256']
            or ('source_sha256' in entry and source_ref['sha256'] != entry['source_sha256'])):
        raise ValueError('Published pair names a changed source result')
    declared = [item for item in container.get('sources', []) if item.get('file') == source_ref['file']]
    if len(declared) != 1 or any(declared[0].get(key) != source_ref[key] for key in ('sha256', 'metrics_sha256', 'git_commit')):
        raise ValueError('Container source manifest does not bind the current audited scene source')
    eligibility, journal_ref, passed = verified_passed_rows(data)
    if not eligibility['eligible']:
        raise ValueError('Source no longer supplies complete pairs with known clean cleanup')
    matches = [(compact, ref, path) for compact, ref, path in passed if ref == entry['pair_reference']]
    if len(matches) != 1:
        raise ValueError('Published pair is absent from the authenticated whole scene journal')
    compact, pair_ref, pair_path = matches[0]
    row = merge.read_reference(pair_ref, within=pair_path.parent)
    if (any(row.get(key) != compact.get(key) for key in ('pair_id', 'scene', 'kind', 'status', 'discard_reasons'))
            or any(row.get(key) != entry.get(key) for key in ('pair_id', 'scene', 'kind', 'target_w0', 'target_w1'))
            or not merge.final_pair_eligible(row.get('kind'), cfg.get('solvability_version'))):
        raise ValueError('Published pair identity or required current construction policy differs')
    merge.reference_path(measured['scene_preflight_reference'], within=cfg['raw_evidence_dir'])
    if row['scene_preflight_reference']['sha256'] != measured['scene_preflight_reference']['sha256']:
        raise ValueError('Pair preflight differs from its source')
    ledger = None
    read = lambda ref: merge.read_reference(ref, within=cfg['raw_evidence_dir'])
    if cfg.get('search_version') is not None:
        ledger = merge.verify_search_ledger(measured, merge.search_source_inventory(measured, read))
    checked = merge.check_pair(row, pair_path, raw_root=cfg['raw_evidence_dir'],
        clone_enabled=cfg.get('enable_candidate_clone') is True, preflight_version=cfg.get('preflight_version'),
        evidence_version=cfg.get('evidence_version'), search_version=cfg.get('search_version'), search_policy=cfg.get('search_policy'),
        search_clone_source=(ledger or {}).get('successful_source_name'),
        solvability_version=cfg.get('solvability_version'), solvability_policy=cfg.get('solvability_policy'),
        generation_version=cfg.get('generation_version'), generation_policy=cfg.get('generation_policy'),
        attempt_setup_version=cfg.get('attempt_setup_version'), attempt_setup_policy=cfg.get('attempt_setup_policy'),
        candidate_policy=cfg.get('candidate_policy'), candidate_policy_spec=cfg.get('candidate_policy_spec'),
        projection_pool_version=cfg.get('projection_pool_version'), projection_pool_policy=cfg.get('projection_pool_policy'))
    if checked['proposal_identity'] != entry['proposal_identity']:
        raise ValueError('Published proposal identity differs from independent reconstruction')
    portable = {Path(ref['file']).name: ref for ref in row['portable_files']}
    read_portable = lambda filename: merge.read_reference(portable[filename], within=pair_path.parent)
    worlds = [read_portable(f'snapshot_w{i}.json') for i in (0, 1)]
    replays = [read_portable(f'witness_w{i}.json') for i in (0, 1)]
    # restore_snapshot reads the original source RGB for its diagnostic. A
    # portable prefix copy does not make a missing runtime input available.
    for world in worlds:
        files = world['capture_reference']['files']
        for name in ('rgb', 'metadata', 'masks'):
            merge.reference_path(files[name], within=cfg['raw_evidence_dir'])
    witnesses = [native_witness(row, index, replay, read) for index, replay in enumerate(replays)]
    t01 = {'pair_id': pair_id, 'witnesses': witnesses}
    reachable, reachable_ref, original_domains = None, None, None
    if row['kind'].startswith('look_'):
        report = row['observation_search']  # Entire catalogue already independently rechecked above.
        domains = report['domains']
        if (domains[0]['reachable_positions'] != domains[1]['reachable_positions']
                or domains[0]['reachable_reference'] != domains[1]['reachable_reference']):
            raise ValueError('Worlds did not use the same actual fixed reachable query')
        reachable, reachable_ref = copy.deepcopy(domains[0]['reachable_positions']), copy.deepcopy(domains[0]['reachable_reference'])
        original_domains = copy.deepcopy(domains)
        t01.update(selected_action=report['selected_action'],
            anchor_visible_actions_w0=copy.deepcopy(report['anchor_visible_actions_w0']),
            anchor_visible_actions_w1=copy.deepcopy(report['anchor_visible_actions_w1']))
    else:
        t01.update(selected_action=None, anchor_visible_actions_w0=None, anchor_visible_actions_w1=None)
    start = canonical_start(row, worlds)
    return {'schema_version': 1, 'pair_id': pair_id, 'scene': row['scene'], 'kind': row['kind'],
        'instruction': row['instruction'], 'anchor_name': row['identities']['anchor_name'],
        'candidate_names': copy.deepcopy(row['identities']['candidate_names']),
        'target_names': [row['target_w0'], row['target_w1']], **start,
        'worlds': worlds, 'row': row, 't01': t01, 'reachable_positions': reachable, 'reachable_reference': reachable_ref,
        'reachable_scope': 'original authenticated fixed native query' if reachable is not None else 'not_recorded_for_direct; require fresh native query',
        'original_observation_domains': original_domains, 'clone_context': copy.deepcopy(worlds[0].get('clone_context')),
        'raw_evidence_root': cfg['raw_evidence_dir'], 'source_config': copy.deepcopy(cfg),
        'evidence_roots': [str(Path(cfg['raw_evidence_dir']).resolve()), str(pair_path.parent.parent.resolve())],
        'sources': [container_ref, source_ref],
        'source_provenance': {'container': container_ref, 'scene': source_ref, 'pair': pair_ref, 'journal': journal_ref},
        'independent_verification': checked, 'partial_source': eligibility['partial_source'],
        'human_review_status': 'not_inferred', 'human_review_approved': None,
        'execution_authorized': False, 'full_t02_acceptance': None}


load_pair_source = load_reviewed_pair

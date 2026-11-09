"""Supplement-3 search adapter; frozen first/clone/stable producers are unchanged.

All Controllers share one absolute scene deadline. Only their search order and
raw-retention behavior change; physical, image, and fresh-witness tests do not.
Run this entry only through the audited multi-GPU launcher and run_logged.
"""
import copy
import json
import os
from pathlib import Path
import time
import traceback

from . import pair_v2_shard as base
from . import pair_v2_clone_shard as clone
from . import pair_v2_stable_shard as stable
from . import pair_v2_simulator as engine
from .pair_v2_clone import prepare_clone_baseline, clone_replay_worker
from .pair_v2_evidence import SearchEvidenceSession
from .pair_v2_search_policy import (SEARCH_VERSION, SEARCH_POLICY, eligible_clone_sources,
    eligible_moved_object, object_inventory, union_identity_sets, RoundRobinScheduler)
from .pair_v2_stable import StableCloneSimulator
from .thor_common import process_identity

_STABLE_ENABLE = stable.enable_config
_STABLE_GENERATOR = stable.generator_worker
_STABLE_RETENTION = stable.RetentionPolicy
RAW_BUDGET_BYTES = 2 * 1024 * 1024 * 1024
TERMINAL_RESERVE_BYTES = 8 * 1024 * 1024
_EXTERNAL_TERMINAL = None


class EvidenceIncomplete(Exception):
    """Expected candidate outcome, never a physical rejection or worker failure."""


def enable_config(config):
    _STABLE_ENABLE(config)
    config.update(search_version=SEARCH_VERSION, search_policy=copy.deepcopy(SEARCH_POLICY))
    config['proposal_order'] = 'fixed sources; fresh Controllers; union identities; round-robin proposals'
    config['search_raw_budget'] = {
        'raw_budget_bytes': RAW_BUDGET_BYTES, 'terminal_reserve_bytes': TERMINAL_RESERVE_BYTES,
        'scope': 'one scene session including all construction Controllers and returned replay payloads',
        'on_cap': 'stop raw writes; continue measured search; incomplete evidence cannot be accepted',
        'compact_and_index_overhead_outside_cap': True, 'disk_deletion': False}
    config['clone_raw_storage_limit'].update(scope='one scene session; raw payload cap plus reserved terminal proof',
        on_limit='stop raw materialization and continue search; no evidence-incomplete pair admission')
    config['raw_evidence_policy'].update(version='search_capacity_retention_v1',
        black_frame='bounded measured terminal proof; full attempt may remain incomplete',
        on_cap='continue search; compact records remain; incomplete evidence never accepted')
    config['clone_rule'] = {**config['clone_rule'], 'version': SEARCH_VERSION,
        'selection': 'fixed eligible singleton list from first passed original preflight; at most six',
        'natural_fallback': 'fresh Controller after all clone failures or a successful clone without eligible groups',
        'failure': 'retain ledger; next source uses a fresh Controller; first physical success stops cloning'}
    return config


def session_factory(scratch, *, construction_root=None, **kwargs):
    # Explicit future marked runs reserve only their original Controller
    # identity metadata. Legacy/replay callers retain their previous session.
    if construction_root is not None:
        from .pair_v2_construction_metadata import ConstructionMetadataSession
        return ConstructionMetadataSession(scratch, construction_root=construction_root,
            raw_budget_bytes=RAW_BUDGET_BYTES, terminal_reserve_bytes=TERMINAL_RESERVE_BYTES, **kwargs)
    return SearchEvidenceSession(scratch, raw_budget_bytes=RAW_BUDGET_BYTES,
                                 terminal_reserve_bytes=TERMINAL_RESERVE_BYTES, **kwargs)


class SearchRetention(_STABLE_RETENTION):
    def decide(self, row, error=None, *, black=False):
        if black:
            stable._SESSION.mark_terminal(error.reference)
        return super().decide(row, error, black=black)


def replay_worker(config, connection):
    """Return buffered actual replay evidence, or preserve only terminal black proof."""
    config = copy.deepcopy(config)
    imported = config.pop('_private_source_image_bytes', {})
    # Normal replay data stays in RAM and is charged to the parent on absorb.
    # A terminal child may write only the parent's explicitly reserved slice.
    session = SearchEvidenceSession(config['runtime']['scratch'],
        raw_budget_bytes=TERMINAL_RESERVE_BYTES, terminal_reserve_bytes=TERMINAL_RESERVE_BYTES,
        buffer_all=True, imports=imported)
    original = engine.EvidenceStore
    try:
        engine.EvidenceStore = session.store
        with session.reading():
            function = clone_replay_worker if config['snapshot'].get('clone_context') is not None else engine.replay_worker
            result = function(config, connection)
        if result.get('exit_code') == 86:
            terminal = session.minimal_black_flush(result['black_frame_reference'])
            result.update(terminal_evidence=terminal, buffered_raw_complete=False,
                          _private_evidence_bundle=None)
            if terminal.get('preserved') is not True:
                result.setdefault('cleanup_errors', []).append('Terminal black evidence exceeded reserved raw capacity')
        else:
            result['_private_evidence_bundle'] = session.bundle()
            result['buffered_raw_complete'] = True
        return result
    finally:
        engine.EvidenceStore = original


def wait_replay(*args, **kwargs):
    result = stable.wait_replay(*args, **kwargs)
    # build_pair raises BlackFrame before adding this witness to its row.
    # Preserve child-reported cleanup/evidence failures in the generator too.
    for error in result.get('cleanup_errors', []):
        if error not in clone._REPLAY_CLEANUP:
            clone._REPLAY_CLEANUP.append(error)
    if result.get('terminal_evidence') is not None and _EXTERNAL_TERMINAL is not None:
        reference = result.get('black_frame_reference') or {}
        _EXTERNAL_TERMINAL[reference.get('file', 'unknown')] = {
            'black_frame_reference': copy.deepcopy(reference),
            'terminal_evidence': copy.deepcopy(result['terminal_evidence'])}
    if result.get('cleanup_errors'):
        # Never start a second replay while the first child's ownership/cleanup
        # is uncertain. Black evidence remains truthful, but cleanup overrides86.
        if result.get('exit_code') == 86:
            raise engine.BlackFrame(result.get('black_frame_reference'))
        raise SystemExit('Fresh witness cleanup uncertain; stop scene search')
    return result


def update_w0(counts, anchor_type, row):
    """Only explicit measured booleans enter the physical denominator."""
    item = counts.setdefault(anchor_type, {'measured_attempts': 0, 'passed': 0,
                                         'rejected': 0, 'measurement_unavailable': 0})
    value = (row.get('physical_w0') or {}).get('valid_physical')
    if type(value) is bool:
        item['measured_attempts'] += 1
        item['passed' if value else 'rejected'] += 1
    else:
        item['measurement_unavailable'] += 1


def physical_rejected(row):
    return any((row.get(key) or {}).get('valid_physical') is False
               for key in ('physical_w0', 'physical_w1'))


def reject_export(row, reason):
    row.update(status='rejected', discard_reasons=[reason])
    if 'pair_file' in row or 'pair_sha256' in row:
        row['incomplete_export'] = {key: row.pop(key) for key in ('pair_file', 'pair_sha256') if key in row}


def proposal_iterator(sim, baseline, identity, reachable, config):
    """Frozen native proposal filters, lazily enumerated once per combination."""
    # Another combination may have left its movable receptacle/anchor in w1.
    # Native coordinates must be queried from the same authenticated baseline
    # that construct_world will restore, never from that previous trial state.
    _, restored = sim.restore_snapshot(baseline)
    sim.store.save_json(f'proposal_setup_{sim.action_count:06d}.json', {
        'purpose': 'common-baseline setup before native placement query; not a w0 physics trial',
        'identity': copy.deepcopy(identity), 'baseline_reference': baseline.get('reference'),
        'restoration': copy.deepcopy(restored)})
    if restored.get('restore_deterministic') is not True:
        raise RuntimeError('Common baseline state restoration failed before native proposal queries')
    objects = engine.objects_by_name(baseline['objects'])
    a, b, x = (objects[name] for name in (*identity['candidate_names'], identity['anchor_name']))
    receptacles = sorted((obj for obj in objects.values() if obj.get('receptacle')
        and obj['name'] not in (*identity['candidate_names'], identity['anchor_name'])
        and not (obj.get('openable') and not obj.get('isOpen'))),
        key=lambda obj: (min(base.distance(obj['position'], candidate['position']) for candidate in (a, b)), obj['name']))
    points = []
    for obj in receptacles[:config['max_receptacles']]:
        try:
            for token in sim.spawn_points(obj['name']):
                points.append({'position': copy.deepcopy(token['selected_position']),
                               'receptacle_name': token['receptacle_name'], 'spawn': token})
        except (TimeoutError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:
            sim.store.save_json(f'spawn_failure_{sim.action_count:06d}.json',
                                {'receptacle': obj['name'], 'error': str(error)})
    left, right = base.placement_groups(points, a, b, x, maximum_per_side=config['max_points_per_side'])
    views = base.view_proposals(reachable, a, b, maximum_positions=config['max_view_positions'])
    for proposal in base.construction_pairs(left, right):
        for view in views:
            yield proposal, view


def run_round_robin(sim, config, baseline, groups, state, journal, connection):
    scheduler = RoundRobinScheduler(groups)
    iterators = {}
    state['identity_groups'] = copy.deepcopy(groups)
    reachable = sim.reachable_positions()['positions']
    while True:
        sim.check_deadline()
        current = base.summarize(state['rows'], state['excluded'], state['status'], config)
        if all(current['passed_by_class'][kind] >= config['scene_pairs_per_class'] for kind in base.KINDS):
            state['status'] = 'completed_quota'
            break
        if len(state['rows']) >= config['max_attempts']:
            state['search_stop_reason'] = 'configured_attempt_limit'
            break
        identity = scheduler.next()
        if identity is None:
            state['search_stop_reason'] = 'all_combinations_exhausted_or_skipped'
            break
        key = scheduler.key(identity)
        if key not in iterators:
            iterators[key] = proposal_iterator(sim, baseline, identity, reachable, config)
        try:
            proposal, view = next(iterators[key])
        except StopIteration:
            scheduler.exhaust(key)
            state['scheduler'] = scheduler.snapshot()
            continue
        pair_id = f"{config['scene'].lower()}_{len(state['rows']):05d}"
        directory = Path(config['raw_evidence_dir']) / pair_id
        directory.mkdir(exist_ok=False)
        row = {'pair_id': pair_id, 'scene': config['scene'], 'status': 'rejected', 'discard_reasons': [],
               'search_version': SEARCH_VERSION, 'search_policy': copy.deepcopy(SEARCH_POLICY),
               'search_candidate_origin': identity['search_candidate_origin'],
               'search_controller_id': state['selected_controller_id']}
        started = time.monotonic()
        try:
            row = base.build_pair(sim, config, baseline, identity, proposal, view,
                                  pair_id, directory, reachable, attempt_row=row)
            if row.get('status') == 'passed' and (row.get('evidence_retention') or {}).get('evidence_complete_for_acceptance') is not True:
                reject_export(row, 'evidence_incomplete_storage_cap')
        except EvidenceIncomplete:
            reject_export(row, 'evidence_incomplete_storage_cap')
            row['evidence_complete_for_acceptance'] = False
        except (TimeoutError, KeyboardInterrupt, SystemExit):
            reject_export(row, 'deadline_or_interrupt')
            raise
        except Exception as error:
            reject_export(row, f'{type(error).__name__}: {error}')
            traceback.print_exc()
        finally:
            row['generation_seconds'] = time.monotonic() - started
            update_w0(state['w0_by_anchor'], identity['anchor_type'], row)
            scheduler.observe(identity, physical_rejected=physical_rejected(row))
            state['scheduler'] = scheduler.snapshot()
            raw = base.write_json(directory / 'attempt.json', row)
            compact = {key: row.get(key) for key in ('pair_id', 'scene', 'kind', 'status', 'discard_reasons',
                'generation_seconds', 'pair_file', 'pair_sha256', 'search_version', 'search_candidate_origin',
                'search_controller_id', 'evidence_complete_for_acceptance')}
            compact.update(raw_attempt=raw, anchor_type=identity['anchor_type'],
                w0_valid_physical=(row.get('physical_w0') or {}).get('valid_physical'),
                w1_valid_physical=(row.get('physical_w1') or {}).get('valid_physical'))
            state['rows'].append(compact)
            journal.write(json.dumps(compact, sort_keys=True, allow_nan=False) + '\n')
            journal.flush()
            print('T01V2_ATTEMPT: ' + json.dumps(compact, sort_keys=True, allow_nan=False), flush=True)
            connection.send(('progress', summary(state, config)))


def summary(state, config):
    result = base.summarize(state['rows'], state['excluded'], state['status'], config)
    for key in ('w0_by_anchor', 'clone_source_order', 'clone_attempts', 'controller_visits',
                'identity_groups', 'scheduler', 'selected_controller_id', 'search_stop_reason'):
        result[key] = copy.deepcopy(state.get(key))
    result.update(search_version=SEARCH_VERSION, search_policy=copy.deepcopy(SEARCH_POLICY),
        w0_count_scope='measured_attempts requires explicit bool physical_w0.valid_physical; unavailable is separate',
        clone_preparation=copy.deepcopy(clone._PREPARATION),
        search_raw_storage=stable._SESSION.status(),
        controller_policy='sequential fresh Unity Controllers; one absolute scene deadline; at most one sequential fresh replay')
    result['cleanup_errors'].extend(state['cleanup_errors'])
    external = list((_EXTERNAL_TERMINAL or {}).values())
    result['search_raw_storage']['external_terminal_proofs'] = copy.deepcopy(external)
    sizes = [row['terminal_evidence'].get('packet_bytes') for row in external]
    external_bytes = sum(sizes) if all(type(size) is int for size in sizes) else None
    result['search_raw_storage'].update(external_terminal_packet_bytes=external_bytes,
        parent_stored_payload_scope='parent session stores; returned ordinary replay bundles included; separately persisted terminal child proof excluded',
        external_terminal_packet_scope='authenticated closure bytes; inspect each preserved flag before treating the packet as retained')
    return result


def raw_reference(reference):
    if reference is None:
        return None
    result = copy.deepcopy(reference)
    # Path.is_file is deliberately virtual inside session.reading(); os.stat's
    # real filesystem view distinguishes retained bytes from memory-only refs.
    result['materialized'] = os.path.isfile(result['file'])
    result['availability'] = ('retained_raw_file' if result['materialized']
                              else 'not_retained_raw; digest identifies observed in-memory bytes only')
    return result


def compact_preflight(record):
    diagnostics = record.get('diagnostics') or {}
    initial = diagnostics.get('initial_settling') or {}
    window = diagnostics.get('stability_window') or {}

    def maximum(rows):
        values = [item['baseline_displacement_m'] for row in rows
                  for item in (row.get('measurements') or {}).get('objects', {}).values()]
        return max(values) if values else None

    initial_rows, window_rows = initial.get('observations'), window.get('observations')
    measured_initial = list(initial_rows or [])
    if diagnostics.get('pause_measurements') is not None:
        measured_initial.insert(0, {'measurements': diagnostics['pause_measurements']})
    return {'passed': record.get('passed'), 'reasons': copy.deepcopy(record.get('reasons')),
        'error': record.get('error'), 'preflight_version': record.get('preflight_version'),
        'initial_settled': initial.get('settled'), 'settling_criterion': initial.get('criterion'),
        'observed_initial_steps': len(initial_rows) if isinstance(initial_rows, list) else None,
        'initial_max_displacement_m': maximum(measured_initial),
        'window_completed': window.get('completed'),
        'observed_window_steps': len(window_rows) if isinstance(window_rows, list) else None,
        'window_max_displacement_m': maximum(window_rows or []),
        'scope': 'compact actual preflight measurements; not a substitute for full raw acceptance evidence'}


def compact_clone(record):
    measured = record.get('clone_record') or {}
    checked = measured.get('clone_baseline_check') or {}
    physical = measured.get('construction_physical') or {}
    return {'baseline_passed': checked.get('passed'),
        'creation_passed': ((measured.get('creation') or {}).get('lineage') or {}).get('passed'),
        'physical_reasons': copy.deepcopy(checked.get('reasons')),
        'physical_edits': copy.deepcopy((checked.get('physical') or {}).get('edits')),
        'native_is_scene_at_rest': physical.get('native_is_scene_at_rest'),
        'scope': 'recorded setup measurements only; missing measurements remain unknown'}


def preflight_reference(sim):
    path = sim.store.directory / 'scene_preflight.json'
    return raw_reference({'file': str(path), 'sha256': base.sha(path)})


def clone_attempt(sim, preflight, source, visit, source_index):
    """One native clone preparation, never a second proposal in this Controller."""
    record = {'enabled': True, 'status': 'preflight_failed', 'passed': False,
        'rule': copy.deepcopy(SEARCH_POLICY), 'clone_attempted': False,
        'original_scene_preflight_passed': preflight.get('passed'),
        'source_name': source['name'], 'source_type': source['objectType'], 'lineage': None, 'error': None,
        'controller_id': visit['controller_id'], 'fixed_source_index': source_index,
        'preflight_reference': copy.deepcopy(visit['preflight_reference'])}
    prepared = None
    try:
        if preflight.get('passed') is not True:
            return record, None
        indexed = object_inventory(preflight['snapshot']['objects'])
        actual = indexed.get(source['name'])
        if (actual is None or actual['objectType'] != source['objectType'] or not eligible_moved_object(actual)
                or sum(obj['objectType'] == actual['objectType'] for obj in indexed.values()) != 1):
            record['status'] = 'source_unavailable'
            return record, None
        token = clone.select_clone_token(sim, preflight['snapshot'], actual, record)
        if token is None:
            record['status'] = 'clone_no_native_proposal'
            return record, None
        record.update(status='clone_setup_started', clone_attempted=True)
        prepared = prepare_clone_baseline(sim, preflight['snapshot'], actual['name'], token)
        record.update(status='clone_ready' if prepared['passed'] is True else 'clone_setup_failed',
            passed=prepared['passed'] is True, lineage=copy.deepcopy(prepared.get('lineage')),
            baseline_validation_reference=raw_reference(prepared.get('record_reference')),
            clone_record=copy.deepcopy(prepared['record']))
        return record, prepared
    except (TimeoutError, KeyboardInterrupt, SystemExit):
        record.update(status='clone_interrupted', error='terminal interruption; see retained parent error')
        raise
    except Exception as error:
        record.update(status='clone_setup_failed', error=f'{type(error).__name__}: {error}')
        return record, None
    finally:
        record['record_reference'] = raw_reference(sim.store.save_json('clone_selection.json', record))
        # This is deliberately attached before any exception propagates.
        visit['clone_record'] = record


def search_body(config, connection):
    state = {'rows': [], 'excluded': None, 'status': 'completed_search', 'w0_by_anchor': {},
        'clone_source_order': [], 'clone_attempts': [], 'controller_visits': [], 'identity_groups': [],
        'scheduler': [], 'selected_controller_id': None, 'search_stop_reason': None, 'cleanup_errors': []}
    journal = base.ROOT / config['attempt_journal']
    journal.parent.mkdir(parents=True, exist_ok=True)
    fixed_sources, natural_available = None, False
    next_source, fallback, finished = 0, False, False

    def save_visit(visit):
        record = visit.pop('clone_record', None)
        if record is not None:
            item = {key: copy.deepcopy(record.get(key)) for key in ('source_name', 'source_type',
                'fixed_source_index', 'controller_id', 'status', 'passed', 'clone_attempted', 'error',
                'baseline_validation_reference')}
            item.update(attempt_index=len(state['clone_attempts']), worker_pid=visit.get('worker_pid'),
                unity_identity=copy.deepcopy(visit.get('unity_identity')),
                preflight_passed=visit.get('preflight_passed'), preflight_reference=visit.get('preflight_reference'),
                selection_reference=raw_reference(record.get('record_reference')),
                measurements=compact_clone(record))
            state['clone_attempts'].append(item)
        connection.send(('progress', summary(state, config)))

    with journal.open('x') as handle:
        try:
            while not finished:
                if time.time() >= config['deadline']:
                    raise TimeoutError('Absolute scene deadline before next fresh Controller')
                number = len(state['controller_visits'])
                visit = {'controller_id': f'search_controller_{number:02d}', 'worker_pid': os.getpid(),
                         'role': 'natural_fallback' if fallback else 'clone_source_or_discovery',
                         'unity_identity': None, 'preflight_passed': None, 'preflight_reference': None,
                         'cleanup_errors': []}
                state['controller_visits'].append(visit)
                directory = Path(config['raw_evidence_dir']) / visit['controller_id'] / 'construction'
                visit['raw_directory'] = str(directory)
                sim, normal_exit = None, False
                try:
                    with base.controller_session(config, visit['controller_id'], segmentation=True, connection=connection) as controller:
                        if getattr(getattr(controller, '_build', None), 'commit_id', None) != engine.BUILD:
                            raise RuntimeError('Unity build differs from reviewed fixed source')
                        visit['unity_identity'] = process_identity(controller.unity_pid)
                        if visit['unity_identity'] is None:
                            raise RuntimeError('Cannot authenticate fresh Unity process identity')
                        previous = [item['unity_identity'] for item in state['controller_visits'][:-1]]
                        if visit['unity_identity'] in previous:
                            raise RuntimeError('Fresh Controller reused an earlier Unity process identity')
                        sim = StableCloneSimulator(controller, config, directory)
                        preflight = sim.preflight_scene()
                        visit.update(preflight_passed=preflight.get('passed'),
                                     preflight_reference=preflight_reference(sim),
                                     preflight_measurements=compact_preflight(preflight),
                                     preflight_reasons=copy.deepcopy(preflight.get('reasons')))
                        config['scene_preflight_reference'] = copy.deepcopy(visit['preflight_reference'])
                        if fixed_sources is None:
                            if preflight.get('passed') is not True:
                                state['excluded'] = {'scene': config['scene'],
                                    'reason': 'settled_baseline_window_preflight_failed',
                                    'reasons': preflight.get('reasons'), 'evidence': visit['preflight_reference']}
                                finished = True
                                continue
                            fixed_sources = eligible_clone_sources(preflight['snapshot']['objects'])
                            state['clone_source_order'] = [{'name': obj['name'], 'objectType': obj['objectType']}
                                                           for obj in fixed_sources]
                            natural_available = bool(union_identity_sets(preflight['snapshot']['objects']))
                        baseline, groups, ready = None, [], False
                        if fallback or not fixed_sources:
                            if preflight.get('passed') is True:
                                baseline = preflight['snapshot']
                                groups = union_identity_sets(baseline['objects'])
                                clone._PREPARATION = {'enabled': True, 'status': 'natural_fallback',
                                    'passed': bool(groups), 'clone_attempted': False,
                                    'original_scene_preflight_passed': True, 'lineage': None,
                                    'controller_id': visit['controller_id'],
                                    'original_preflight_reference': visit['preflight_reference']}
                                ready = bool(groups)
                            finished = True
                        else:
                            record, prepared = clone_attempt(sim, preflight, fixed_sources[next_source], visit, next_source)
                            next_source += 1
                            if record['passed'] is True:
                                baseline = prepared['snapshot']
                                groups = union_identity_sets(baseline['objects'], prepared['lineage'])
                                clone._PREPARATION = record
                                ready = bool(groups)
                                # First physical clone success always ends clone attempts.
                                # An expanded inventory without a unique anchor cannot
                                # erase a valid original natural search opportunity.
                                if not groups:
                                    visit['clone_group_outcome'] = 'clone_ready_no_eligible_identity'
                                fallback = bool(not ready and natural_available)
                                finished = ready or not fallback
                            elif next_source >= len(fixed_sources):
                                fallback = natural_available
                                finished = not fallback
                            # Publish the completed clone attempt before pair search:
                            # a later deadline must not hide its success/failure.
                            save_visit(visit)
                        if ready:
                            state['selected_controller_id'] = visit['controller_id']
                            run_round_robin(sim, config, baseline, groups, state, handle, connection)
                            finished = True
                        elif finished and state['excluded'] is None:
                            state['excluded'] = {'scene': config['scene'],
                                'reason': 'no_eligible_union_after_clone_attempts_or_fresh_natural_fallback',
                                'evidence': visit['preflight_reference']}
                    normal_exit = True
                finally:
                    if sim is not None:
                        # A deadline/black frame may interrupt preflight before it
                        # returns; its own finally still writes the actual trace.
                        preflight_path = sim.store.directory / 'scene_preflight.json'
                        if visit.get('preflight_reference') is None and preflight_path.is_file():
                            visit['preflight_reference'] = preflight_reference(sim)
                            visit['preflight_passed'] = (sim.scene_preflight or {}).get('passed')
                            visit['preflight_measurements'] = compact_preflight(sim.scene_preflight or {})
                        try:
                            visit['raw_index'] = raw_reference(sim.finish())
                        except BaseException as error:
                            visit['cleanup_errors'].append(f'evidence finalization: {type(error).__name__}: {error}')
                    identity = visit.get('unity_identity')
                    if identity is not None and process_identity(identity['pid']) == identity:
                        visit['cleanup_errors'].append('Original Unity process remains after Controller cleanup')
                    state['cleanup_errors'].extend(visit['cleanup_errors'])
                    save_visit(visit)
                    if normal_exit and not visit['cleanup_errors']:
                        stable._SESSION.release_runtime_references([])
                if state['cleanup_errors']:
                    raise RuntimeError('Controller cleanup uncertain; refuse to start another Controller')
            return summary(state, config)
        except engine.BlackFrame as error:
            terminal = stable._SESSION.minimal_black_flush(error.reference)
            result = summary(state, config)
            result.update(status='black_frame', black_frame_detected=True,
                black_frame_reference=error.reference, terminal_evidence=terminal, exit_code=86)
            if terminal.get('preserved') is not True:
                result['cleanup_errors'].append('Terminal black evidence could not be preserved within reserve')
            return result
        except (TimeoutError, KeyboardInterrupt, SystemExit) as error:
            result = summary(state, config)
            result.update(status='incomplete', error=f'{type(error).__name__}: {error}')
            return result
        except Exception as error:
            traceback.print_exc()
            result = summary(state, config)
            result.update(status='worker_failed', error=f'{type(error).__name__}: {error}')
            return result


def generator_worker(config, connection):
    global _EXTERNAL_TERMINAL
    if config.get('search_version') != SEARCH_VERSION or config.get('search_policy') != SEARCH_POLICY:
        raise ValueError('Search worker requires its explicit version and exact policy')
    old = (stable.EvidenceSession, stable.RetentionPolicy, stable._CLONE_GENERATOR,
           stable.replay_worker, base.wait_replay, base.export_pair, clone._PREPARATION, clone._REPLAY_CLEANUP,
           _EXTERNAL_TERMINAL)
    archive = None

    def exporter(child_config, row, worlds, prefix, after, witnesses, constructions):
        nonlocal archive
        origin = row['identities'].get('search_candidate_origin')
        actual = union_identity_sets(worlds[0]['objects'], (worlds[0].get('clone_context') or {}).get('lineage'))
        if row['identities'] not in actual or origin not in ('natural', 'cloned'):
            raise ValueError('Export identities do not belong to the actual natural/clone union')
        row.update(search_version=SEARCH_VERSION, search_policy=copy.deepcopy(SEARCH_POLICY),
                   search_candidate_origin=origin)
        plan = stable._SESSION.retention_plan()
        row['search_evidence_capacity_plan'] = plan
        if not plan['can_materialize_complete']:
            raise EvidenceIncomplete('Raw package cannot be completely materialized')
        if archive is None:
            archive = clone.JsonProofArchive(config)
        try:
            return clone.export_pair(old[5], archive, child_config, row, worlds, prefix, after, witnesses, constructions)
        except ValueError as error:
            if str(error) in ('Clone JSON proof archive exceeded declared size/file limit',
                              'Clone JSON proof archive exceeded total byte limit',
                              'Clone proof index exceeded the declared archive byte limit'):
                raise EvidenceIncomplete('Portable JSON proof archive is full') from error
            raise

    def body(child_config, channel):
        # Common references remain readable in RAM after raw capacity is reached.
        with stable._SESSION.reading():
            return search_body(child_config, channel)

    try:
        stable.EvidenceSession, stable.RetentionPolicy = session_factory, SearchRetention
        stable._CLONE_GENERATOR, stable.replay_worker = body, replay_worker
        base.wait_replay, base.export_pair = wait_replay, exporter
        clone._PREPARATION, clone._REPLAY_CLEANUP = None, []
        _EXTERNAL_TERMINAL = {}
        result = _STABLE_GENERATOR(config, connection)
        result.setdefault('cleanup_errors', []).extend(clone._REPLAY_CLEANUP)
        result.update(search_version=SEARCH_VERSION, search_policy=copy.deepcopy(SEARCH_POLICY),
                      enable_candidate_clone=True)
        result['raw_evidence_policy'] = {**result['raw_evidence_policy'],
            'version': 'search_capacity_retention_v1',
            'black_frame': 'bounded measured terminal proof is retained; full attempt may be incomplete',
            'on_cap': 'continue search; no evidence-incomplete pair is accepted'}
        return result
    finally:
        (stable.EvidenceSession, stable.RetentionPolicy, stable._CLONE_GENERATOR,
         stable.replay_worker, base.wait_replay, base.export_pair, clone._PREPARATION, clone._REPLAY_CLEANUP,
         _EXTERNAL_TERMINAL) = old


def main():
    original_generator, original_enable, original_writer = stable.generator_worker, stable.enable_config, base.write_result

    def writer(name, metrics, config):
        enable_config(config)
        metrics.update(search_version=SEARCH_VERSION, search_policy=copy.deepcopy(SEARCH_POLICY))
        return original_writer(name, metrics, config)

    try:
        stable.generator_worker, stable.enable_config, base.write_result = generator_worker, enable_config, writer
        return stable.main()
    finally:
        stable.generator_worker, stable.enable_config, base.write_result = original_generator, original_enable, original_writer


if __name__ == '__main__':
    raise SystemExit(main())

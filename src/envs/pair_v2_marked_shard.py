"""Opt-in supplements 6+7 collector; previous producers are never modified.

Natural marked A/B candidates, projection-only Look/Direct pools, safe native
point retries and the existing exact physical/evidence/witness checks. Launch
only through run_logged and the approved allocation gate.
"""
import copy
from collections import Counter
import json
import os
from pathlib import Path
import time
import traceback

from . import pair_v2_shard as base
from . import pair_v2_simulator as engine
from . import pair_v2_policy as original_policy
from . import pair_v2_marked_policy as candidates
from . import pair_v2_projection_pool as pools
from . import pair_v2_role_policy as roles
from . import pair_v2_solvable_policy as solvable
from . import pair_v2_solvable_shard as old_solvable
from . import pair_v2_stable_shard as stable
from . import pair_v2_search_shard as search
from . import pair_v2_clone_shard as clone
from . import pair_v2_isolated_shard as isolated
from . import pair_v2_terminal as terminal
from .pair_v2_stable import PREFLIGHT_VERSION, PREFLIGHT_POLICY, augment_physical_result
from .pair_v2_coverage_order import view_index_order
from .thor_common import process_identity
from .pair_v2_native_retry import (NATIVE_RETRY_VERSION, FROZEN_RENDER_VERSION, FROZEN_RENDER_POLICY,
                                   record_restore_readback)
from .pair_v2_fov import declared_fields, bind_source_fields

PRODUCER_MODULE = 'src.envs.pair_v2_marked_shard'
SAFE_RETRY_VERSION = NATIVE_RETRY_VERSION
_BASE_BUILD = base.build_pair
_BASE_EXPORT = base.export_pair
_BASE_PHYSICAL = base.physical_measurement
_BASE_WRITE = base.write_json


def enable_config(config):
    shard = config.get('shard', {})
    fields = declared_fields(shard)
    if config.get('field_of_view', 100) != fields['field_of_view']:
        raise ValueError('Marked source camera differs from its immutable shard')
    config.update(fields)
    bind_source_fields(config, shard)
    old_solvable.enable_config(config)
    for key in ('clone_rule', 'clone_adapter_source', 'clone_portable_archive',
                'proposal_order_version', 'proposal_order_policy'):
        config.pop(key, None)
    config.update(candidate_policy=candidates.CANDIDATE_POLICY,
        candidate_policy_spec=copy.deepcopy(candidates.CANDIDATE_POLICY_SPEC),
        projection_pool_version=pools.POOL_VERSION, projection_pool_policy=copy.deepcopy(pools.POOL_POLICY),
        enable_candidate_clone=False, search_version=None, search_policy=None,
        attempt_setup_version=isolated.ATTEMPT_SETUP_VERSION,
        attempt_setup_policy=copy.deepcopy(isolated.ATTEMPT_SETUP_POLICY),
        safe_retry_version=SAFE_RETRY_VERSION, producer_module=PRODUCER_MODULE,
        minimum_eligible_attempts=100, max_attempts=max(100, config.get('max_attempts', 160)),
        initial_agent_setup='actual canonical TeleportFull before settled preflight; retain command and actual response',
        proposal_order='existing view domain; same-class marked candidates first per actual view; independent pool role streams; global Look/Direct alternation',
        controller_restart='only when native failed placement changes reported pose/identity; previous Controller must be confirmed stopped',
        retry_point_limit=40, scientific_thresholds_unchanged=True)
    config.update(frozen_render_version=FROZEN_RENDER_VERSION,
                  frozen_render_policy=copy.deepcopy(FROZEN_RENDER_POLICY))
    config['clone_raw_storage_limit']['scope'] = 'disabled cloning; scene-wide raw storage cap retained as infrastructure only'
    return config


def tags():
    return {'candidate_policy': candidates.CANDIDATE_POLICY,
        'candidate_policy_spec': copy.deepcopy(candidates.CANDIDATE_POLICY_SPEC),
        'projection_pool_version': pools.POOL_VERSION, 'projection_pool_policy': copy.deepcopy(pools.POOL_POLICY),
        'enable_candidate_clone': False, 'search_version': None, 'search_policy': None,
        'attempt_setup_version': isolated.ATTEMPT_SETUP_VERSION,
        'attempt_setup_policy': copy.deepcopy(isolated.ATTEMPT_SETUP_POLICY),
        'preflight_version': PREFLIGHT_VERSION, 'preflight_policy': copy.deepcopy(PREFLIGHT_POLICY),
        'evidence_version': stable.stable_policy.EVIDENCE_VERSION,
        'evidence_policy': copy.deepcopy(stable.stable_policy.EVIDENCE_POLICY),
        'solvability_version': solvable.SOLVABILITY_VERSION,
        'solvability_policy': copy.deepcopy(solvable.SOLVABILITY_POLICY),
        'generation_version': roles.ROLE_VERSION, 'generation_policy': copy.deepcopy(roles.ROLE_POLICY),
        'safe_retry_version': SAFE_RETRY_VERSION, 'producer_module': PRODUCER_MODULE,
        'frozen_render_version': FROZEN_RENDER_VERSION, 'frozen_render_policy': copy.deepcopy(FROZEN_RENDER_POLICY),
        'p_backend_solvability_verified': None}


def canonical_initial_agent_setup(sim):
    """Actual action, not rewriting baseline numbers or weakening equality."""
    metadata = copy.deepcopy(sim.controller.last_event.metadata)
    agent = metadata['agent']
    sim.position_agent(copy.deepcopy(agent['position']), agent['rotation']['y'], agent['cameraHorizon'])
    record = {'requested_from_initial_metadata': agent,
        'actual_command': copy.deepcopy(sim.agent_setup_command),
        'actual_agent': copy.deepcopy(sim.agent_setup_actual),
        'action_reference': copy.deepcopy(sim.last_action_reference),
        'scope': 'executed once before preflight; later restore reuses this saved command'}
    reference = sim.store.save_json('initial_canonical_agent_setup.json', record)
    sim.canonical_initial_setup_reference = copy.deepcopy(reference)
    sim.canonical_initial_setup_record = copy.deepcopy(record)
    return reference


def replay_worker(config, connection):
    """Spawn entry: install marked semantics explicitly in this fresh process."""
    config = copy.deepcopy(config)
    if config.get('candidate_policy') != candidates.CANDIDATE_POLICY or config['snapshot'].get('clone_context') is not None:
        raise ValueError('Marked replay requires its declared natural candidate policy')
    config['enable_candidate_clone'] = False
    previous = original_policy.closest_target
    original_restore = engine.V2Simulator.restore_snapshot
    def restore(sim, snapshot):
        capture, measured = original_restore(sim, snapshot)
        if config.get('frozen_render_version') == FROZEN_RENDER_VERSION:
            record_restore_readback(sim, snapshot, measured)
        return capture, measured
    try:
        original_policy.closest_target = candidates.closest_target
        engine.V2Simulator.restore_snapshot = restore
        return search.replay_worker(config, connection)
    finally:
        original_policy.closest_target = previous
        engine.V2Simulator.restore_snapshot = original_restore


def save_successful_prefix(sim, row, beginning):
    actions = copy.deepcopy(sim.query_log[beginning:])
    if not actions or actions[-1]['action'] != 'PlaceObjectAtPoint':
        raise ValueError('Successful native prefix lacks its final placement')
    record = {'version': isolated.ATTEMPT_SETUP_VERSION, 'safe_retry_version': SAFE_RETRY_VERSION,
        'pair_id': row['pair_id'], 'scene': row['scene'], 'setup_reference': row['attempt_setup_reference'],
        'action_references': actions, 'first_place_reference': actions[-1]['reference'],
        'scope': 'complete actual setup/action chain through the first successful w0 placement, including safe failed points'}
    row['attempt_setup_native_prefix_reference'] = sim.store.save_json(
        f'marked_setup_prefix_{row["pair_id"]}_{sim.action_count:06d}.json', record)


def prefix_observer(original, sim, row, beginning, arguments):
    original_action = sim.action
    own = vars(sim).get('action')
    had = 'action' in vars(sim)
    def action(name, **kwargs):
        result = original_action(name, **kwargs)
        if name == 'PlaceObjectAtPoint' and not row.get('attempt_setup_native_prefix_reference'):
            if result.metadata.get('lastActionSuccess') is not True:
                raise ValueError('Returned PlaceObjectAtPoint did not explicitly succeed')
            save_successful_prefix(sim, row, beginning)
        return result
    try:
        sim.action = action
        return original(*arguments, attempt_row=row)
    finally:
        if had:
            sim.action = own
        else:
            del sim.action


def construct_observer(sim, proposal, role, record, row, *, frozen=False, physical_context=None):
    """Return a temporary method; native retry adapter owns safe mutation rules."""
    original = sim.construct_world
    realized, constructions = {}, {}
    context = {} if physical_context is None else physical_context
    def construct(baseline, key, placements, require_agent=None):
        world = 'w0' if key.endswith('_w0') else 'w1' if key.endswith('_w1') else None
        if world is None or len(placements) != 1:
            raise ValueError('Marked construction requires exactly one anchor in an explicit world')
        side = role[world + '_source_side']
        alternatives = pools.placement_alternatives(record, proposal['projection_pool']['pool'], side,
            proposal[world], exclude_positions=([proposal['w0']['position']] if world == 'w1' else ()))
        if not alternatives:
            from .pair_v2_native_retry import NativePlacementExhausted
            raise NativePlacementExhausted({'status': 'exhausted', 'failed_point_count': 0,
                'attempts': [], 'reason': 'no_distinct_same_side_pool_point_after_w0_native_retry'})
        requested = copy.deepcopy(placements)
        requested[0]['spawn'] = copy.deepcopy(alternatives[0]['spawn'])
        requested[0]['alternatives'] = [item['spawn'] for item in alternatives[1:]]
        result = original(baseline, key, requested, require_agent=require_agent)
        name = requested[0]['object_name']
        actual = result['successful_tokens_by_name'][name]
        matches = [item for item in alternatives if item['spawn'] == actual]
        if len(matches) != 1:
            raise ValueError('Successful native token does not belong to this declared side/pool')
        proposal[world] = copy.deepcopy(matches[0])
        row['proposal'] = proposal
        row.setdefault('native_retry_records', {})[world] = copy.deepcopy(result.get('retry_records'))
        realized[world] = result['snapshot']
        constructions[world] = result
        if frozen and world == 'w1' and result['passed'] and constructions['w0']['passed']:
            physical_worlds = [realized['w0'], realized['w1']]
            row['physical_world_references'] = [copy.deepcopy(w['reference']) for w in physical_worlds]
            measured = sim.freeze_render_pair(physical_worlds, name, row['pair_id'])
            row['frozen_render_reference'] = copy.deepcopy(measured['reference'])
            row['frozen_render_readbacks'] = copy.deepcopy(measured['proof']['readbacks'])
            row['nonanchor_cross_world_readback'] = copy.deepcopy(measured['proof']['nonanchor_cross_world_readback'])
            row['physical_w1'] = augment_physical_result(_BASE_PHYSICAL(result), result['physical'])
            if not measured['passed']:
                raise RuntimeError('frozen_render_restore_failed')
            for index, label in enumerate(('w0', 'w1')):
                # Keep physical dictionaries untouched. All target, pixel,
                # distance and witness checks below use these actual renders.
                constructions[label]['snapshot'] = measured['worlds'][index]
                constructions[label]['capture'] = measured['captures'][index]
            context.update(physical_worlds=physical_worlds, render_worlds=measured['worlds'])
        if world == 'w1' and proposal['mode'] == 'stay':
            row['stay_horizontal_displacement'] = candidates.stay_horizontal_displacement(
                [constructions['w0']['snapshot'], result['snapshot']], name)
        return result
    return original, construct


def measured_build(sim, config, baseline, identities, proposal, view, pair_id, directory, reachable, attempt_row=None):
    row = {} if attempt_row is None else attempt_row
    row.update(tags())
    # orient_proposal copies its input. The live pool is not part of a selected
    # proposal and may contain thousands of full native coordinate tokens.
    record = proposal['_private_pool_record']
    public_proposal = {key: value for key, value in proposal.items() if key != '_private_pool_record'}
    oriented, role = roles.orient_proposal(public_proposal, config['scene'], old_solvable.pair_attempt_index(pair_id, config['scene']), seed=0)
    role.update(candidate_a_name=identities['candidate_names'][0], candidate_b_name=identities['candidate_names'][1])
    row['proposal_role'] = role
    row['projection_pool_reference'] = copy.deepcopy(record['reference'])
    row['candidate_types'], row['same_class'] = copy.deepcopy(identities['candidate_types']), identities['same_class']
    frozen = config.get('frozen_render_version') == FROZEN_RENDER_VERSION
    physical_context = {}
    original_construct, new_construct = construct_observer(sim, oriented, role, record, row, frozen=frozen,
                                                          physical_context=physical_context)
    had, own = 'construct_world' in vars(sim), vars(sim).get('construct_world')
    old_observer = isolated.build_with_prefix_observer
    original_save = sim.store.save_json
    store_had, store_own = 'save_json' in vars(sim.store), vars(sim.store).get('save_json')
    def save(name, value):
        # Safe native exhaustion does not consume the pair counter. Its next
        # proposal still needs an independent immutable setup record.
        if name == f'attempt_setup_{pair_id}.json':
            name = f'attempt_setup_{Path(directory).name}.json'
        return original_save(name, value)
    solvable.reset_report()
    original_single_edit = original_policy.single_edit_check
    def single_edit(first, requested, anchor):
        if frozen:
            physical = physical_context.get('physical_worlds')
            rendered = physical_context.get('render_worlds')
            if not physical or not rendered or first != rendered[0]['poses'] or anchor != identities['anchor_name']:
                raise ValueError('Frozen requested-edit check lacks its original physical w0 and actual render')
            first = physical[0]['poses']
        return original_single_edit(first, requested, anchor)
    try:
        sim.construct_world = new_construct
        sim.store.save_json = save
        original_policy.single_edit_check = single_edit
        isolated.build_with_prefix_observer = prefix_observer
        result = isolated.isolated_build(_BASE_BUILD, sim, config, baseline, identities, oriented, view,
            pair_id, directory, reachable, attempt_row=row)
        split_prefix_rejection(result)
        return result
    finally:
        original_policy.single_edit_check = original_single_edit
        isolated.build_with_prefix_observer = old_observer
        if store_had:
            sim.store.save_json = store_own
        else:
            del sim.store.save_json
        if had:
            sim.construct_world = own
        else:
            del sim.construct_world
        report = solvable.last_report()
        if report is not None:
            row['look_solvability'] = report
            row['observation_search'] = report
            for key in ('anchor_visible_actions_w0', 'anchor_visible_actions_w1'):
                row[key] = copy.deepcopy(report.get(key))
        solvable.reset_report()


def split_prefix_rejection(row):
    if row.get('discard_reasons') != ['neither_hidden_shared_prefix_nor_direct_evidence']:
        return
    pre = row.get('prefix_measurement') or {}
    masks = pre.get('prefix_edit_mask_pixels')
    if not isinstance(masks, list) or len(masks) != 1 or not isinstance(masks[0], dict):
        return
    pixels = [masks[0].get(key) for key in ('w0', 'w1')]
    if not all(type(value) is int and value >= 0 for value in pixels):
        return
    if any(pixels):
        row['discard_reasons'] = ['anchor_visible_at_v0_but_direct_evidence_failed']
    elif type(pre.get('prefix_max_abs_diff')) is int and pre['prefix_max_abs_diff'] > 3:
        row['discard_reasons'] = ['anchor_hidden_but_prefix_rgb_diff_gt3']


def export_pair(config, row, worlds, prefix, after, witnesses, constructions):
    if any(world.get('clone_context') is not None for world in worlds):
        raise ValueError('Default marked producer cannot export a cloned inventory')
    names = row['identities']['candidate_names']
    for observation in prefix:
        if set(observation['mark_labels_by_name']) != set(names):
            raise ValueError('Only the two marked candidate identities may receive labels')
    if row['kind'].startswith('look'):
        row['look_solvability'] = copy.deepcopy(row.get('observation_search') or solvable.last_report())
        if not solvable.solvability_summary(row['look_solvability'])['passed']:
            raise ValueError('Look lacks complete per-world one-step visibility evidence')
    capacity = stable._SESSION.retention_plan()
    row['search_evidence_capacity_plan'] = capacity
    if not capacity['can_materialize_complete']:
        raise search.EvidenceIncomplete('Full marked candidate evidence exceeds retained raw capacity')
    return old_solvable.export_pair(_BASE_EXPORT, config, row, worlds, prefix, after, witnesses, constructions)


def compact_native_retry(record):
    """Keep measured outcomes/references without duplicating raw pose maps.

    References identify original records, not a promise that rejected raw
    payloads survived the existing retention cap. Retained full attempts and
    successful pairs still contain their unmodified native ledger.
    """
    if record is None:
        return None
    result = {key: copy.deepcopy(record[key]) for key in (
        'version', 'scene', 'record_key', 'object_name', 'rotation', 'status',
        'max_points', 'supplied_point_count', 'attempted_point_count', 'failed_point_count',
        'arguments', 'action_reference', 'fresh_controller_required',
        'pair_attempt_counter_modified', 'error', 'reason', 'ledger_reference') if key in record}
    result['successful_token_reference'] = pools.native_token_reference(record.get('successful_token'))
    result['attempts'] = []
    for attempt in record.get('attempts', []):
        item = {key: copy.deepcopy(attempt[key]) for key in (
            'token_index', 'arguments', 'action_index', 'action_reference', 'native_success',
            'error', 'pose_read_error') if key in attempt}
        item['spawn_token_reference'] = pools.native_token_reference(attempt.get('spawn_token'))
        check = attempt.get('unchanged_check')
        item['unchanged_check'] = None if check is None else {key: copy.deepcopy(check[key]) for key in (
            'verified', 'unchanged', 'inventory_exact', 'identity_changed_names', 'pose_changed_names',
            'missing_names', 'added_names', 'representation_only_names', 'scope', 'error') if key in check}
        result['attempts'].append(item)
    result['storage_scope'] = 'compact measured retry outcomes; referenced raw availability follows the original retention decision'
    return result


def compact_attempt(row):
    result = stable.compact_attempt(row)
    for key in ('candidate_policy', 'candidate_types', 'same_class', 'projection_pool_reference',
                'proposal_role', 'stay_horizontal_displacement',
                'attempt_setup_reference', 'attempt_setup_native_prefix_reference',
                'frozen_render_reference', 'physical_world_references', 'frozen_render_readbacks',
                'nonanchor_cross_world_readback',
                'counts_as_pair_attempt'):
        result[key] = copy.deepcopy(row.get(key))
    records = row.get('native_retry_records')
    result['native_retry_records'] = None if records is None else {
        world: {name: compact_native_retry(record) for name, record in values.items()}
        for world, values in records.items()}
    result['native_retry_failure'] = compact_native_retry(row.get('native_retry_failure'))
    result['look_solvability'] = solvable.compact_report(row.get('look_solvability'))
    return result


def summary(state, config):
    result = base.summarize(state['rows'], state['excluded'], state['status'], config)
    result.update(tags())
    result.update(declared_fields(config))
    for key in ('w0_by_anchor', 'controller_visits', 'selected_controller_id', 'search_stop_reason',
                'pool_catalogs', 'placement_exhausted_proposals', 'placement_point_failures',
                'controller_restarts', 'discovery_failures', 'proposal_trials', 'failed_native_point_count'):
        result[key] = copy.deepcopy(state[key])
    eligible = sum(type(row.get('w0_valid_physical')) is bool for row in state['rows'])
    result.update(clone_attempts=[], clone_source_order=[], cloning_attempt_count=0,
        minimum_eligible_attempts=100, eligible_pair_attempts=eligible, minimum_attempt_count_met=eligible >= 100,
        minimum_count_scope='recorded trials with a measured w0 physical result; native failed points/setup errors do not increment this count',
        search_raw_storage=stable._SESSION.status(), cleanup_errors=copy.deepcopy(state['cleanup_errors']))
    result['proposed_pool_attempts'] = {pool: {
        'attempts': sum(row.get('proposed_pool') == pool for row in state['rows']),
        'discard_reason_counts': dict(Counter(reason for row in state['rows'] if row.get('proposed_pool') == pool
                                              for reason in row.get('discard_reasons', [])))}
        for pool in ('look', 'direct')}
    external = list((search._EXTERNAL_TERMINAL or {}).values())
    result['search_raw_storage']['external_terminal_proofs'] = copy.deepcopy(external)
    result['search_raw_storage']['external_terminal_packet_bytes'] = sum(
        item['terminal_evidence']['packet_bytes'] for item in external) if all(
        type(item['terminal_evidence'].get('packet_bytes')) is int for item in external) else None
    return result


def existing_views(baseline, reachable, config):
    seen, views = set(), []
    for a, b in candidates.all_candidate_pairs(baseline['objects']):
        for view in base.view_proposals(reachable, a, b, maximum_positions=config['max_view_positions']):
            key = engine.encoded(view)
            if key not in seen:
                seen.add(key); views.append(view)
    return [views[index] for index in view_index_order(views)]


def discover_view(sim, baseline, reachable, view, config, state):
    """Actual v0 discovery; native points are queried only from the same baseline."""
    _, restored = sim.restore_snapshot(baseline)
    if restored.get('restore_deterministic') is not True:
        raise RuntimeError('Canonical common baseline restoration failed before marked view discovery')
    capture = sim.position_agent(view['position'], view['yaw'], view['horizon'])
    groups = candidates.identity_sets_at_view(capture)
    source = sim.store.save_json(f'marked_discovery_{sim.action_count:06d}.json',
        {'view': view, 'identities': groups, 'capture': capture['evidence'], 'restoration': restored})
    objects = engine.objects_by_name(baseline['objects'])
    for identity in groups:
        # A previous yielded group may have left w1; queries never use that state.
        _, restoration = sim.restore_snapshot(baseline)
        if restoration.get('restore_deterministic') is not True:
            raise RuntimeError('Canonical common baseline restoration failed before native proposal queries')
        actual = sim.position_agent(view['position'], view['yaw'], view['horizon'])
        if identity not in candidates.identity_sets_at_view(actual):
            state['discovery_failures'].append({'identity': identity, 'view': view,
                'reason': 'actual_v0_candidate_eligibility_changed', 'source_reference': source})
            continue
        a, b, x = (objects[name] for name in (*identity['candidate_names'], identity['anchor_name']))
        receptacles = sorted((obj for obj in objects.values() if obj.get('receptacle')
            and obj['name'] not in (*identity['candidate_names'], identity['anchor_name'])
            and not (obj.get('openable') and not obj.get('isOpen'))),
            key=lambda obj: (min(base.distance(obj['position'], item['position']) for item in (a, b)), obj['name']))
        points = []
        for obj in receptacles[:config['max_receptacles']]:
            try:
                for token in sim.spawn_points(obj['name']):
                    points.append({'position': copy.deepcopy(token['selected_position']),
                        'receptacle_name': token['receptacle_name'], 'spawn': token})
            except (engine.BlackFrame, TimeoutError, KeyboardInterrupt, SystemExit):
                raise
            except Exception as error:
                failure = {'receptacle_name': obj['name'], 'identity': identity, 'view': view,
                    'error': f'{type(error).__name__}: {error}',
                    'action_reference': copy.deepcopy(sim.last_action_reference)}
                failure['reference'] = sim.store.save_json(f'spawn_failure_{sim.action_count:06d}.json', failure)
                state['discovery_failures'].append(failure)
        left, right = base.placement_groups(points, a, b, x, maximum_per_side=len(points))
        actions, unavailable = base.observation_actions(sim, reachable, sim.snapshot('projection_action_source'))
        camera = pools.camera_from_metadata(actual['metadata'], actual['evidence']['files']['metadata'])
        record = pools.build_view_pools(left, right, a, b, x, camera=camera, view=view,
            observation_cameras=[pools.camera_after_action(camera, action) for action in actions])
        record.update(identity=copy.deepcopy(identity), discovery_reference=source,
                      unavailable_observation_actions=unavailable)
        record['reference'] = sim.store.save_json(f'projection_pools_{sim.action_count:06d}.json',
                                                  pools.pool_evidence_record(record))
        state['pool_catalogs'].append({'identity': identity, 'view': view, 'counts': record['counts'],
                                      'reference': search.raw_reference(record['reference'])})
        yield identity, record


def run_search(sim, config, baseline, state, journal, connection, seen):
    from .pair_v2_native_retry import NativePlacementExhausted, FreshControllerRequired
    reachable = sim.reachable_positions()['positions']
    def discovered_domains():
        for view in existing_views(baseline, reachable, config):
            yield from discover_view(sim, baseline, reachable, view, config, state)
    discovery = iter(discovered_domains())
    active, discovering, next_pool = [], True, 'look'
    while discovering or active:
        sim.check_deadline()
        if len(state['rows']) >= config['max_attempts']:
            state['search_stop_reason'] = 'configured_attempt_limit'; return
        counts = base.summarize(state['rows'], None, state['status'], config)['passed_by_class']
        if sum(type(row.get('w0_valid_physical')) is bool for row in state['rows']) >= 100 and all(counts[kind] >= config['scene_pairs_per_class'] for kind in base.KINDS):
            state['status'] = 'completed_quota'; return
        if discovering:
            try:
                # Discover one actual candidate/anchor domain, then attempt a
                # proposal. Do not query every identity in a crowded view
                # before the first constructive attempt can run.
                identity, record = next(discovery)
                active.append({'identity': identity, 'record': record, 'stream': pools.PoolProposalIterator([record]),
                               'consecutive_physical_rejections': 0})
            except StopIteration:
                discovering = False
        # Select one source stream that actually contains the wanted pool.
        # Other streams keep their cursor; a locally empty pool cannot starve
        # an available desired pool in a different candidate/view stream.
        if active:
            selected = next((index for index, item in enumerate(active)
                if not item['stream'].summary()['pools'][next_pool]['all_exhausted']), 0)
            chosen = [active.pop(selected)]
        else:
            chosen = []
        remaining = active
        for item in chosen:
            sim.check_deadline()
            if len(state['rows']) >= config['max_attempts']:
                remaining.append(item); continue
            stream = item['stream']; stream.next_pool = next_pool
            try:
                proposal, view, _ = next(stream)
            except StopIteration:
                continue
            identity, record = item['identity'], item['record']
            signature = engine.encoded([identity, proposal['mode'], proposal['projection_pool']['pool'], view,
                [[proposal[key]['receptacle_name'], proposal[key]['position']] for key in ('w0', 'w1')]])
            if signature in seen:
                remaining.append(item); continue
            seen.add(signature)
            next_pool = 'direct' if proposal['projection_pool']['pool'] == 'look' else 'look'
            proposal['_private_pool_record'] = record
            pair_id = f'{config["scene"].lower()}_{len(state["rows"]):05d}'
            trial = state['proposal_trials']; state['proposal_trials'] += 1
            directory = Path(config['raw_evidence_dir']) / f'{pair_id}_trial{trial:06d}'
            directory.mkdir(exist_ok=False)
            row = {'pair_id': pair_id, 'scene': config['scene'], 'status': 'rejected', 'discard_reasons': [],
                   'search_controller_id': state['selected_controller_id'], 'proposed_pool': proposal['projection_pool']['pool']}
            started, counted, terminal_error = time.monotonic(), True, None
            try:
                row = base.build_pair(sim, config, baseline, identity, proposal, view, pair_id, directory, reachable, attempt_row=row)
                if row.get('status') == 'passed' and (row.get('evidence_retention') or {}).get('evidence_complete_for_acceptance') is not True:
                    search.reject_export(row, 'evidence_incomplete_storage_cap')
            except (NativePlacementExhausted, FreshControllerRequired) as error:
                counted = False
                state['placement_exhausted_proposals'] += 1
                record_error = copy.deepcopy(error.result)
                row.update(status='rejected', discard_reasons=['native_placement_points_exhausted_or_state_changed'],
                           native_retry_failure=record_error)
                if isinstance(error, FreshControllerRequired):
                    terminal_error = error
            except search.EvidenceIncomplete:
                search.reject_export(row, 'evidence_incomplete_storage_cap')
            except (engine.BlackFrame, TimeoutError, KeyboardInterrupt, SystemExit) as error:
                search.reject_export(row, 'terminal_interruption'); terminal_error = error
            except Exception as error:
                retry_error = getattr(error, 'native_retry_result', None)
                if isinstance(retry_error, dict) and retry_error.get('fresh_controller_required') is True:
                    counted = False
                    row.update(status='rejected', discard_reasons=['native_placement_state_unknown'],
                        native_retry_failure=copy.deepcopy(retry_error),
                        original_native_error=f'{type(error).__name__}: {error}')
                    state['placement_exhausted_proposals'] += 1
                    terminal_error = FreshControllerRequired(retry_error)
                else:
                    search.reject_export(row, f'{type(error).__name__}: {error}')
                    traceback.print_exc()
            finally:
                row['generation_seconds'] = time.monotonic() - started
                row['counts_as_pair_attempt'] = counted
                native_records = [native for world in (row.get('native_retry_records') or {}).values()
                                  for native in world.values()]
                if row.get('native_retry_failure'):
                    native_records.append(row['native_retry_failure'])
                state['failed_native_point_count'] += sum(record.get('failed_point_count', 0) for record in native_records)
                raw = base.write_json(directory / 'attempt.json', row)
                if counted:
                    search.update_w0(state['w0_by_anchor'], identity['anchor_type'], row)
                    compact = {key: copy.deepcopy(row.get(key)) for key in ('pair_id', 'scene', 'kind', 'status',
                        'discard_reasons', 'generation_seconds', 'pair_file', 'pair_sha256', 'proposed_pool',
                        'candidate_types', 'same_class', 'stay_horizontal_displacement')}
                    compact.update(raw_attempt=raw, anchor_type=identity['anchor_type'],
                        w0_valid_physical=(row.get('physical_w0') or {}).get('valid_physical'),
                        w1_valid_physical=(row.get('physical_w1') or {}).get('valid_physical'))
                    state['rows'].append(compact)
                    journal.write(json.dumps(compact, sort_keys=True, allow_nan=False) + '\n'); journal.flush()
                    print('T01V2_ATTEMPT: ' + json.dumps(compact, sort_keys=True, allow_nan=False), flush=True)
                else:
                    state['placement_point_failures'].append({'pair_id_not_consumed': pair_id,
                        'trial': trial, 'raw_attempt': raw,
                        'native_retry_failure': compact_native_retry(row.get('native_retry_failure'))})
                connection.send(('progress', summary(state, config)))
            if terminal_error is not None:
                raise terminal_error
            rejected = search.physical_rejected(row)
            item['consecutive_physical_rejections'] = item['consecutive_physical_rejections'] + 1 if rejected else 0
            if item['consecutive_physical_rejections'] < 10:
                remaining.append(item)
        active = remaining
    state['search_stop_reason'] = 'all_existing_views_and_pool_domains_exhausted_or_skipped'


def search_body(config, connection):
    from .pair_v2_native_retry import RetryPlacementSimulator, FreshControllerRequired
    state = {'rows': [], 'excluded': None, 'status': 'completed_search', 'w0_by_anchor': {},
        'controller_visits': [], 'selected_controller_id': None, 'search_stop_reason': None,
        'cleanup_errors': [], 'pool_catalogs': [], 'placement_exhausted_proposals': 0,
        'placement_point_failures': [], 'controller_restarts': 0, 'discovery_failures': [], 'proposal_trials': 0,
        'failed_native_point_count': 0}
    seen = set()
    journal = base.ROOT / config['attempt_journal']; journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open('x') as handle:
        try:
            while True:
                if time.time() >= config['deadline']:
                    raise TimeoutError('Absolute scene deadline before fresh natural Controller')
                number = len(state['controller_visits'])
                visit = {'controller_id': f'marked_controller_{number:03d}', 'worker_pid': os.getpid(),
                         'role': 'natural_marked', 'unity_identity': None, 'cleanup_errors': []}
                state['controller_visits'].append(visit)
                sim, restart, normal_exit = None, False, False
                try:
                    with base.controller_session(config, visit['controller_id'], segmentation=True, connection=connection) as controller:
                        if getattr(getattr(controller, '_build', None), 'commit_id', None) != engine.BUILD:
                            raise RuntimeError('Unity build differs from reviewed pinned build')
                        visit['unity_identity'] = process_identity(controller.unity_pid)
                        if visit['unity_identity'] is None or visit['unity_identity'] in [v['unity_identity'] for v in state['controller_visits'][:-1]]:
                            raise RuntimeError('Fresh Controller identity missing or reused')
                        directory = Path(config['raw_evidence_dir']) / visit['controller_id'] / 'construction'
                        sim = RetryPlacementSimulator(controller, config, directory)
                        visit['initial_agent_setup_reference'] = canonical_initial_agent_setup(sim)
                        preflight = sim.preflight_scene()
                        visit.update(preflight_passed=preflight.get('passed'),
                            preflight_reference=search.preflight_reference(sim),
                            preflight_measurements=search.compact_preflight(preflight))
                        config['scene_preflight_reference'] = copy.deepcopy(visit['preflight_reference'])
                        if preflight.get('passed') is not True:
                            state['excluded'] = {'scene': config['scene'], 'reason': 'settled_baseline_window_preflight_failed',
                                'reasons': preflight.get('reasons'), 'evidence': visit['preflight_reference']}
                            break
                        state['selected_controller_id'] = visit['controller_id']
                        run_search(sim, config, preflight['snapshot'], state, handle, connection, seen)
                    normal_exit = True
                except FreshControllerRequired:
                    restart = True; state['controller_restarts'] += 1
                finally:
                    if sim is not None:
                        try:
                            visit['raw_index'] = search.raw_reference(sim.finish())
                        except BaseException as error:
                            visit['cleanup_errors'].append(f'evidence finalization: {type(error).__name__}: {error}')
                    identity = visit.get('unity_identity')
                    if identity and process_identity(identity['pid']) == identity:
                        visit['cleanup_errors'].append('Owned Unity remains after Controller cleanup')
                    state['cleanup_errors'].extend(visit['cleanup_errors'])
                    connection.send(('progress', summary(state, config)))
                    if (normal_exit or restart) and not state['cleanup_errors']:
                        stable._SESSION.release_runtime_references([])
                if state['cleanup_errors']:
                    raise RuntimeError('Controller cleanup uncertain; refusing another Controller')
                if not restart:
                    break
            return summary(state, config)
        except engine.BlackFrame as error:
            retained = stable._SESSION.minimal_black_flush(error.reference)
            result = summary(state, config)
            result.update(status='black_frame', black_frame_detected=True, black_frame_reference=error.reference,
                          terminal_evidence=retained, exit_code=86)
            if retained.get('preserved') is not True:
                result['cleanup_errors'].append('Terminal black proof could not be preserved')
            return result
        except BaseException as error:
            result = summary(state, config)
            result.update(status='incomplete' if isinstance(error, (TimeoutError, KeyboardInterrupt, SystemExit)) else 'worker_failed',
                          error=f'{type(error).__name__}: {error}')
            return result


def generator_worker(config, connection):
    if (config.get('candidate_policy') != candidates.CANDIDATE_POLICY
            or config.get('candidate_policy_spec') != candidates.CANDIDATE_POLICY_SPEC
            or config.get('enable_candidate_clone') is not False
            or config.get('projection_pool_policy') != pools.POOL_POLICY):
        raise ValueError('Marked producer requires its exact explicit natural/projection policies')
    session, retention = search.session_factory(config['runtime']['scratch']), search.SearchRetention()
    previous = (engine.EvidenceStore, stable._SESSION, stable._REPLAY_RAW_COMPLETE,
        stable.replay_worker, clone._REPLAY_CLEANUP, search._EXTERNAL_TERMINAL,
        base.build_pair, base.write_json, base.export_pair, base.wait_replay, base.physical_measurement,
        base.choose_look, original_policy.closest_target, original_policy.classify_pair)
    retention_rows = []
    def build(*args, **kwargs):
        row = kwargs.get('attempt_row')
        if row is None:
            row = {}; kwargs['attempt_row'] = row
        error = None; session.begin(); stable._REPLAY_RAW_COMPLETE = True
        try:
            with session.reading():
                return measured_build(*args, **kwargs)
        except BaseException as caught:
            error = caught; raise
        finally:
            keep, decision = retention.decide(row, error, black=isinstance(error, engine.BlackFrame))
            row['raw_complete'] = bool(stable._REPLAY_RAW_COMPLETE)
            try:
                stored = session.finish(keep)
            except BaseException as storage_error:
                row.update(raw_complete=False, evidence_retention={**decision, 'full_raw_retained': False,
                    'storage_error': f'{type(storage_error).__name__}: {storage_error}',
                    'partial_disk_evidence_retained': True})
                session.pending.clear(); session.active = False
                raise SystemExit('Evidence materialization failed; retain partial files') from storage_error
            row['evidence_retention'] = {**decision, **stored, 'raw_complete': row['raw_complete'],
                                        'omitted_payloads_existed_only_in_memory': not keep}
            row['thumbnail'] = stable.thumbnail(args[0], args[7])
            retention_rows.append({'pair_id': row.get('pair_id'), **row['evidence_retention']})
    def written(path, value):
        if Path(path).name == 'attempt.json' and value.get('status') != 'passed' and (value.get('evidence_retention') or {}).get('full_raw_retained') is False:
            value = compact_attempt(value)
        return _BASE_WRITE(path, value)
    try:
        engine.EvidenceStore, stable._SESSION = session.store, session
        stable.replay_worker = replay_worker
        clone._REPLAY_CLEANUP, search._EXTERNAL_TERMINAL = [], {}
        base.build_pair, base.write_json, base.export_pair, base.wait_replay = build, written, export_pair, search.wait_replay
        base.physical_measurement = lambda result: augment_physical_result(_BASE_PHYSICAL(result), result['physical'])
        base.choose_look = solvable.choose_look
        original_policy.closest_target, original_policy.classify_pair = candidates.closest_target, solvable.classify_pair
        with session.reading():
            result = search_body(config, connection)
        result.setdefault('cleanup_errors', []).extend(clone._REPLAY_CLEANUP)
        result['raw_evidence_policy'] = copy.deepcopy(config['raw_evidence_policy'])
        result['evidence_retention'] = {'attempts': retention_rows,
            'complete_rejection_reason_counts': dict(retention.reason_counts),
            'physical_rejection_distribution': dict(retention.physical_distribution), 'disk_files_deleted': 0}
        return result
    finally:
        (engine.EvidenceStore, stable._SESSION, stable._REPLAY_RAW_COMPLETE,
         stable.replay_worker, clone._REPLAY_CLEANUP, search._EXTERNAL_TERMINAL,
         base.build_pair, base.write_json, base.export_pair, base.wait_replay, base.physical_measurement,
         base.choose_look, original_policy.closest_target, original_policy.classify_pair) = previous
        solvable.reset_report()


def main():
    previous = base.generator_worker, base.Worker, base.write_result
    active, publication = [], {'verified': False}
    def worker(target, config):
        enable_config(config)
        instance = terminal.TerminalWorker(target, config)
        active.append(instance); return instance
    def writer(name, metrics, config):
        enable_config(config); metrics.update(tags())
        metrics.update(declared_fields(config))
        final, verified = terminal.final_metrics(metrics, config, active[0] if len(active) == 1 else None)
        final.update(tags())
        final.update(declared_fields(config))
        publication['verified'] = verified
        return previous[2](name, final, config)
    try:
        base.generator_worker, base.Worker, base.write_result = generator_worker, worker, writer
        code = base.main()
        return code if publication['verified'] else 1
    finally:
        base.generator_worker, base.Worker, base.write_result = previous


if __name__ == '__main__':
    raise SystemExit(main())

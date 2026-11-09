"""Add-only discovery cadence. Native construction and acceptance stay unchanged.

One newly discovered combination starts each round. Every active combination
at that instant receives one visit before the next expensive discovery. A visit
may end in exhaustion, a duplicate, a native failure, or a physical skip; none
of those creates a measured attempt. The original attempt counter owns that.
"""
import copy
import json
from pathlib import Path
import time
import traceback
from . import pair_v2_marked_shard as core

base, engine, pools, search = core.base, core.engine, core.pools, core.search
VERSION = 'one_discovery_then_full_active_round_v1'
POLICY = {
    'version': VERSION,
    'round': 'discover at most one new combination, then process each combination active at that round start once',
    'empty_active': 'continue actual discovery until a proposal domain is available or discovery is exhausted',
    'within_round': 'unchanged preference for a stream containing the requested Look/Direct pool; then original active order',
    'exhaustion': 'StopIteration, seen proposal, native failure, and ten consecutive physical rejections consume a domain visit; other round domains still proceed',
    'native_queries': 'unchanged native discovery and original authenticated queries; no cached or inferred coordinates',
    'unchanged': 'all domain contents, seed0 assignment, global counted attempt IDs, seen set, safe native retries, native failure counting, physical10-skip, thresholds and cleanup',
    'scope': 'search scheduling only; fewer discoveries per active round, no speedup or acceptance claim',
}
FIELDS = ('discovery_schedule_version', 'discovery_schedule_policy')


def fields():
    return {FIELDS[0]: VERSION, FIELDS[1]: copy.deepcopy(POLICY)}


def validate_fields(value):
    if any(value.get(key) != expected for key, expected in fields().items()):
        raise ValueError('Missing/altered full-active-round discovery schedule')


class DiscoveryRound:
    """Snapshot finite active work; never duplicate a visit within one round."""
    def __init__(self, active):
        self.pending = list(active)

    def __bool__(self):
        return bool(self.pending)

    def pop(self, requested_pool):
        if requested_pool not in ('look', 'direct'):
            raise ValueError('Unknown requested projection pool')
        if not self.pending:
            raise StopIteration
        selected = next((index for index, item in enumerate(self.pending)
            if not item['stream'].summary()['pools'][requested_pool]['all_exhausted']), 0)
        return self.pending.pop(selected)


def run_search(sim, config, baseline, state, journal, connection, seen):
    from .pair_v2_native_retry import NativePlacementExhausted, FreshControllerRequired
    reachable = sim.reachable_positions()['positions']
    def discovered_domains():
        for view in core.existing_views(baseline, reachable, config):
            yield from core.discover_view(sim, baseline, reachable, view, config, state)
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
                # Discover one actual candidate/anchor domain at round start;
                # do not enumerate the whole view before trying placements.
                identity, record = next(discovery)
                active.append({'identity': identity, 'record': record, 'stream': pools.PoolProposalIterator([record]),
                               'consecutive_physical_rejections': 0})
            except StopIteration:
                discovering = False
        # Only discovery cadence changes: one new domain at round start,
        # then every domain in that snapshot is processed at most once.
        # A stream with no required Flip can stop without blocking its peers.
        current = DiscoveryRound(active)
        remaining = []
        while current:
            item = current.pop(next_pool)
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
                        'native_retry_failure': core.compact_native_retry(row.get('native_retry_failure'))})
                connection.send(('progress', core.summary(state, config)))
            if terminal_error is not None:
                raise terminal_error
            rejected = search.physical_rejected(row)
            item['consecutive_physical_rejections'] = item['consecutive_physical_rejections'] + 1 if rejected else 0
            if item['consecutive_physical_rejections'] < 10:
                remaining.append(item)
        active = remaining
    state['search_stop_reason'] = 'all_existing_views_and_pool_domains_exhausted_or_skipped'

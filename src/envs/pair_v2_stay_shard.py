"""Isolated supplement 11 producer: unchanged validation, far-first Stay search."""
import copy
from contextlib import contextmanager

from . import pair_v2_marked_shard as marked
from . import pair_v2_native_retry as native
from . import pair_v2_projection_pool as pools
from . import pair_v2_stay_order as order

_ORIGINAL_CONFIG = marked.enable_config
_ORIGINAL_TAGS = marked.tags
_ORIGINAL_OBSERVER = marked.construct_observer
_ORIGINAL_COMPACT = marked.compact_attempt
_ORIGINAL_GENERATOR = marked.generator_worker


def enable_config(config):
    order.validate_fields(config['shard'])
    if config['shard'].get('producer_module') != order.PRODUCER:
        raise ValueError('Stay worker requires its explicit immutable producer assignment')
    _ORIGINAL_CONFIG(config)
    config.update(order.fields(), producer_module=order.PRODUCER)
    return config


def tags():
    return {**_ORIGINAL_TAGS(), **order.fields(), 'producer_module': order.PRODUCER}


def compact_attempt(row):
    result = _ORIGINAL_COMPACT(row)
    for key in (*order.FIELDS, 'stay_w1_search_order'):
        result[key] = copy.deepcopy(row.get(key))
    return result


def construct_observer(sim, proposal, role, record, row, **kwargs):
    """Intercept native inputs while preserving the complete original observer.

    The ledger hook sees successful placements before later physical settling
    can fail, and partial failed-placement ledgers before their exceptions pass
    through. Offered but unexecuted alternatives are never marked as tried.
    """
    original = sim.construct_world
    planned_w0 = pools._point_key(proposal['w0'])
    w0_result = {}
    had, own = 'construct_world' in vars(sim), vars(sim).get('construct_world')

    def tracked(baseline, key, placements, require_agent=None):
        world = 'w0' if key.endswith('_w0') else 'w1' if key.endswith('_w1') else None
        if proposal['mode'] != 'stay' or world != 'w1':
            result = original(baseline, key, placements, require_agent=require_agent)
            if world == 'w0':
                name = placements[0]['object_name']
                w0_result.update(common_baseline_reference=copy.deepcopy(baseline['reference']),
                    snapshot_reference=copy.deepcopy(result['snapshot']['reference']),
                    position=copy.deepcopy(marked.engine.objects_by_name(result['snapshot']['objects'])[name]['position']),
                    token=copy.deepcopy(result['successful_tokens_by_name'][name]))
            return result
        if not w0_result:
            raise ValueError('Stay w1 requires the actual successful physical w0 result')
        name, pool, side = placements[0]['object_name'], proposal['projection_pool']['pool'], role['w1_source_side']
        actual_w0 = proposal['w0']  # the original observer already wrote back its actual native fallback
        if actual_w0['spawn'] != w0_result['token']:
            raise ValueError('Stay w0 proposal differs from the actual successful native token')
        key_state = order.state_key({'reference': w0_result['common_baseline_reference']},
            record, role, pool, side, name, actual_w0)
        if not hasattr(sim, '_stay_w1_tried'):
            sim._stay_w1_tried = {}
        tried = sim._stay_w1_tried.setdefault(key_state, set())
        fell_back = pools._point_key(actual_w0) != planned_w0
        options = order.alternatives(record, pool, side, proposal['w1'], actual_w0,
            w0_result['position'], tried, w0_fell_back=fell_back)
        proof = {'version': order.VERSION, 'policy': copy.deepcopy(order.POLICY),
            'common_baseline_reference': w0_result['common_baseline_reference'],
            'w0_snapshot_reference': w0_result['snapshot_reference'],
            'w0_successful_token_reference': pools.native_token_reference(w0_result['token']),
            'realized_w0_anchor_position': w0_result['position'], 'w0_fell_back': fell_back,
            'pool': pool, 'side': side, 'previously_attempted_native_keys': [list(value) for value in sorted(tried)],
            'ordered_token_references': [pools.native_token_reference(item['spawn']) for item in options],
            'attempted_token_references': [], 'native_retry_ledger_references': []}
        row['stay_w1_search_order'] = proof
        if not options:
            raise native.NativePlacementExhausted({'status': 'exhausted', 'failed_point_count': 0,
                'attempts': [], 'reason': 'no_untried_distinct_stay_w1_point_for_actual_w0'})
        requested = copy.deepcopy(placements)
        requested[0]['spawn'] = copy.deepcopy(options[0]['spawn'])
        requested[0]['alternatives'] = [copy.deepcopy(item['spawn']) for item in options[1:]]
        prior_place = native.place_native_with_retries

        def remember(ledger):
            if not isinstance(ledger, dict) or ledger.get('object_name') != name:
                return
            reference = ledger.get('ledger_reference')
            if reference is not None:
                proof['native_retry_ledger_references'].append(copy.deepcopy(reference))
            for attempted in ledger.get('attempts', []):
                token = attempted['spawn_token']
                matches = [item for item in options if item['spawn'] == token]
                if len(matches) != 1:
                    raise ValueError('Actual Stay retry is outside the offered native domain')
                tried.add(pools._point_key(matches[0]))
                proof['attempted_token_references'].append(pools.native_token_reference(token))

        def place(*args, **kw):
            try:
                outcome = prior_place(*args, **kw)
            except BaseException as error:
                remember(getattr(error, 'native_retry_result', None))
                raise
            remember(outcome)
            return outcome

        native.place_native_with_retries = place
        try:
            return original(baseline, key, requested, require_agent=require_agent)
        finally:
            native.place_native_with_retries = prior_place

    try:
        sim.construct_world = tracked
        _, observed = _ORIGINAL_OBSERVER(sim, proposal, role, record, row, **kwargs)
    finally:
        if had:
            sim.construct_world = own
        else:
            del sim.construct_world
    return original, observed


@contextmanager
def installed():
    previous = (marked.enable_config, marked.tags, marked.compact_attempt,
                marked.construct_observer, marked.generator_worker, pools._PoolRoles)
    try:
        marked.enable_config, marked.tags, marked.compact_attempt = enable_config, tags, compact_attempt
        marked.construct_observer, marked.generator_worker = construct_observer, generator_worker
        pools._PoolRoles = order.FarFirstPoolRoles
        yield
    finally:
        (marked.enable_config, marked.tags, marked.compact_attempt,
         marked.construct_observer, marked.generator_worker, pools._PoolRoles) = previous


def generator_worker(config, connection):
    # multiprocessing spawn imports this module afresh: install in the child.
    order.validate_source(config)
    with installed():
        return _ORIGINAL_GENERATOR(config, connection)


def main():
    with installed():
        return marked.main()


if __name__ == '__main__':
    raise SystemExit(main())

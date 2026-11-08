"""Versioned per-attempt baseline setup; historical producers stay unchanged.

Setup runs inside the existing attempt evidence session, before the frozen
builder sets v0. Nothing is inserted between that builder's w0 and w1.
"""
import copy
import hashlib
import json
from pathlib import Path

from . import pair_v2_solvable_shard as previous
from . import pair_v2_simulator as engine
from . import pair_v2_coverage_order as coverage
from . import pair_v2_terminal as terminal

base, stable, search = previous.base, previous.stable, previous.search
ATTEMPT_SETUP_VERSION = 'common_baseline_before_each_attempt_v1'
ATTEMPT_SETUP_POLICY = {
    'version': ATTEMPT_SETUP_VERSION,
    'timing': 'restore common baseline before each frozen build_pair positions the agent at v0',
    'baseline': 'same validated settled or clone common baseline supplied to every attempt',
    'verification': 'unchanged restoration_state: all movable/pickupable positions <=1cm and encoded full agent pose exact',
    'signed_zero': 'unchanged encoded comparison; signed-zero differences remain mismatches',
    'paired_prefix': 'no additional restore/reset/agent action between w0 and w1',
    'setup_failure': 'record rejected attempt; no w0 physical measurement invented',
    'terminal_failure': 'black frame, deadline and process interrupt propagate through existing cleanup',
    'retention': 'existing accepted/first-three-reason/terminal evidence policy and byte caps unchanged',
}
_ENABLE = previous.enable_config
_GENERATOR = previous.generator_worker


def enable_config(config):
    _ENABLE(config)
    coverage.enable_config(config)
    config.update(attempt_setup_version=ATTEMPT_SETUP_VERSION,
                  attempt_setup_policy=copy.deepcopy(ATTEMPT_SETUP_POLICY),
                  producer_module='src.envs.pair_v2_isolated_shard')
    return config


def compact_setup(record):
    result = {key: copy.deepcopy(record.get(key)) for key in
        ('version', 'pair_id', 'status', 'setup_passed', 'baseline_reference',
         'restored_snapshot_reference', 'error', 'proof_save_error')}
    for key in ('restoration', 'independent_state'):
        measured = record.get(key)
        result[key] = None if measured is None else {name: copy.deepcopy(measured.get(name)) for name in
            ('restore_deterministic', 'restored_positions_within_1cm', 'agent_pose_exact',
             'agent_requested', 'agent_realized')}
    return result


def save_native_prefix(sim, row, beginning):
    actions = copy.deepcopy(sim.query_log[beginning:])
    found = [index for index, action in enumerate(actions) if action.get('action') == 'PlaceObjectAtPoint']
    if not found or not row.get('attempt_setup_reference'):
        raise ValueError('First w0 native placement has no observed setup/action reference')
    actions = actions[:found[0] + 1]
    record = {'version': ATTEMPT_SETUP_VERSION, 'pair_id': row['pair_id'], 'scene': row['scene'],
        'setup_reference': copy.deepcopy(row['attempt_setup_reference']),
        'action_references': actions, 'first_place_reference': copy.deepcopy(actions[-1]['reference']),
        'scope': 'actual ordered native actions through first w0 PlaceObjectAtPoint; no actions inserted'}
    row['attempt_setup_native_prefix_reference'] = sim.store.save_json(
        f'attempt_setup_native_prefix_{row["pair_id"]}.json', record)


def build_with_prefix_observer(original, sim, row, beginning, arguments):
    """Observe one existing action; record its refs before export capacity check."""
    had_instance_method = 'action' in vars(sim)
    old_instance_method, original_action = vars(sim).get('action'), sim.action

    def observed(action, **kwargs):
        result, failure, proof_error = None, None, None
        try:
            result = original_action(action, **kwargs)
        except BaseException as error:
            failure = error
        finally:
            if action == 'PlaceObjectAtPoint' and not row.get('attempt_setup_native_prefix_reference'):
                try:
                    save_native_prefix(sim, row, beginning)
                except BaseException as error:
                    proof_error = error
                    row['attempt_setup_native_prefix_error'] = f'{type(error).__name__}: {error}'
        if failure is not None:
            raise failure
        if proof_error is not None:
            raise proof_error
        return result

    try:
        sim.action = observed
        return original(*arguments, attempt_row=row)
    finally:
        if had_instance_method:
            sim.action = old_instance_method
        else:
            del sim.action


def isolated_build(original, sim, config, baseline, identities, proposal, view,
                   pair_id, directory, reachable, attempt_row=None):
    row = {} if attempt_row is None else attempt_row
    row.update(pair_id=pair_id, scene=config['scene'], identities=copy.deepcopy(identities),
        proposal=copy.deepcopy(proposal), view=copy.deepcopy(view), mode=proposal['mode'],
        status='rejected', discard_reasons=[], attempt_setup_version=ATTEMPT_SETUP_VERSION,
        attempt_setup_policy=copy.deepcopy(ATTEMPT_SETUP_POLICY))
    record = {'version': ATTEMPT_SETUP_VERSION, 'policy': copy.deepcopy(ATTEMPT_SETUP_POLICY),
        'pair_id': pair_id, 'scene': config['scene'], 'phase': 'before_v0_and_w0',
        'baseline_reference': copy.deepcopy(baseline.get('reference')),
        'status': 'running', 'setup_passed': False, 'restoration': None,
        'restored_snapshot_reference': None, 'independent_state': None,
        'action_references': [], 'error': None}
    beginning, terminal = len(sim.query_log), None
    reason = 'attempt_baseline_restore_error'
    try:
        reference = baseline['reference']
        raw = Path(reference['file']).read_bytes()
        if (hashlib.sha256(raw).hexdigest() != reference['sha256']
                or engine.encoded(json.loads(raw)) != engine.encoded({k: v for k, v in baseline.items() if k != 'reference'})
                or baseline['scene'] != config['scene']):
            raise ValueError('Common baseline differs from its authenticated native snapshot')
        _, restoration = sim.restore_snapshot(baseline)
        record['restoration'] = copy.deepcopy(restoration)
        # snapshot reads current actual metadata; it sends no Unity action.
        realized = sim.snapshot(pair_id + '_attempt_baseline_ready')
        record['restored_snapshot_reference'] = copy.deepcopy(realized['reference'])
        measured = engine.restoration_state(baseline, realized['objects'], realized['agent'])
        record['independent_state'] = measured
        record['setup_passed'] = bool(restoration.get('restore_deterministic') is True
                                      and measured['restore_deterministic'] is True)
        record['status'] = 'passed' if record['setup_passed'] else 'mismatch'
        reason = 'attempt_baseline_restore_mismatch'
    except BaseException as error:
        record['error'] = f'{type(error).__name__}: {error}'
        if isinstance(error, (TimeoutError, KeyboardInterrupt, SystemExit)):
            terminal = error
            record['status'] = 'interrupted'
        elif isinstance(error, Exception):
            record['status'] = 'error'
        else:
            terminal = error
            record['status'] = 'interrupted'
    finally:
        record['action_references'] = copy.deepcopy(sim.query_log[beginning:])
        row['attempt_setup'] = record
        try:
            row['attempt_setup_reference'] = sim.store.save_json(f'attempt_setup_{pair_id}.json', record)
        except BaseException as error:
            # Preserve the original terminal failure, especially BlackFrame86.
            # An ordinary failed proof write can never authorize construction.
            record['proof_save_error'] = f'{type(error).__name__}: {error}'
            record['setup_passed'] = False
            record['status'] = 'proof_error'
            reason = 'attempt_baseline_evidence_unavailable'
            if terminal is None and (isinstance(error, (TimeoutError, KeyboardInterrupt, SystemExit))
                                     or not isinstance(error, Exception)):
                terminal = error
    if terminal is not None:
        row['discard_reasons'] = ['attempt_baseline_setup_interrupted']
        raise terminal
    if record['setup_passed'] is not True:
        row['discard_reasons'] = [reason]
        return row
    return build_with_prefix_observer(original, sim, row, beginning,
        (sim, config, baseline, identities, proposal, view, pair_id, directory, reachable))


def generator_worker(config, connection):
    if (config.get('attempt_setup_version') != ATTEMPT_SETUP_VERSION
            or config.get('attempt_setup_policy') != ATTEMPT_SETUP_POLICY
            or config.get('proposal_order_version') != coverage.ORDER_VERSION
            or config.get('proposal_order_policy') != coverage.ORDER_POLICY):
        raise ValueError('Isolated worker requires its explicit setup version and exact policy')
    old = base.build_pair, stable.compact_attempt, search.summary, base.export_pair

    def build(*args, **kwargs):
        return isolated_build(old[0], *args, **kwargs)

    def compact(row):
        result = old[1](row)
        result.update(attempt_setup_version=row.get('attempt_setup_version'),
            attempt_setup_policy=copy.deepcopy(row.get('attempt_setup_policy')),
            attempt_setup_reference=copy.deepcopy(row.get('attempt_setup_reference')),
            attempt_setup_native_prefix_reference=copy.deepcopy(row.get('attempt_setup_native_prefix_reference')),
            attempt_setup_native_prefix_error=row.get('attempt_setup_native_prefix_error'),
            attempt_setup=compact_setup(row['attempt_setup']) if row.get('attempt_setup') else None)
        return result

    def export(config, row, worlds, prefix, after, witnesses, constructions):
        reference = row.get('attempt_setup_native_prefix_reference')
        if not reference or row.get('attempt_setup', {}).get('setup_passed') is not True:
            raise ValueError('Accepted pair lacks its actual pre-w0 baseline/native action proof')
        raw = Path(reference['file']).read_bytes()
        proof = json.loads(raw)
        geometry = constructions[0]['placement_evidence_by_name'][row['identities']['anchor_name']]['geometry']
        if (hashlib.sha256(raw).hexdigest() != reference['sha256']
                or proof.get('setup_reference') != row['attempt_setup_reference']
                or proof.get('first_place_reference') != geometry['placement_reference']):
            raise ValueError('Accepted pair native prefix does not bind its first actual w0 placement')
        return old[3](config, row, worlds, prefix, after, witnesses, constructions)

    def summary(state, child_config):
        result = old[2](state, child_config)
        result.update(attempt_setup_version=ATTEMPT_SETUP_VERSION,
                      attempt_setup_policy=copy.deepcopy(ATTEMPT_SETUP_POLICY),
                      proposal_order_version=coverage.ORDER_VERSION,
                      proposal_order_policy=copy.deepcopy(coverage.ORDER_POLICY))
        return result

    try:
        base.build_pair, stable.compact_attempt, search.summary, base.export_pair = build, compact, summary, export
        with coverage.install_coverage_order():
            result = _GENERATOR(config, connection)
        result.update(attempt_setup_version=ATTEMPT_SETUP_VERSION,
                      attempt_setup_policy=copy.deepcopy(ATTEMPT_SETUP_POLICY),
                      proposal_order_version=coverage.ORDER_VERSION,
                      proposal_order_policy=copy.deepcopy(coverage.ORDER_POLICY))
        return result
    finally:
        base.build_pair, stable.compact_attempt, search.summary, base.export_pair = old


def main():
    old = previous.generator_worker, previous.enable_config, base.write_result, base.Worker
    active, publication = [], {'verified': False}

    def worker(target, config):
        instance = terminal.TerminalWorker(target, config)
        active.append(instance)
        return instance

    def writer(name, metrics, config):
        enable_config(config)
        metrics.update(attempt_setup_version=ATTEMPT_SETUP_VERSION,
                       attempt_setup_policy=copy.deepcopy(ATTEMPT_SETUP_POLICY),
                       proposal_order_version=coverage.ORDER_VERSION,
                       proposal_order_policy=copy.deepcopy(coverage.ORDER_POLICY))
        final, verified = terminal.final_metrics(metrics, config, active[0] if len(active) == 1 else None)
        publication['verified'] = verified
        return old[2](name, final, config)

    try:
        previous.generator_worker, previous.enable_config, base.write_result = generator_worker, enable_config, writer
        base.Worker = worker
        code = previous.main()
        # Frozen base.main computes its exit code before invoking its writer.
        # A post-cleanup seal mismatch must never leak an earlier 0 or 86.
        return code if publication['verified'] else 1
    finally:
        previous.generator_worker, previous.enable_config, base.write_result, base.Worker = old


if __name__ == '__main__':
    raise SystemExit(main())

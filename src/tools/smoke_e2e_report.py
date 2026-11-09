"""Report finalized E2E raw-output parsing and actual per-update timings.

Run through run_logged after committing this code. Original-model policies and
each training update remain separate; no paper endpoint or timing is inferred.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

from src.envs.paired_env import parse_action
from src.envs.thor_common import ROOT
from src.tools.t01_partition import audited_result
from src.tools.smoke_e2e_training_contract import available_actions


TIMING_KEYS = ('generation_seconds', 'scoring_seconds', 'update_seconds',
               'critic_update_seconds', 'actor_update_seconds', 'total_seconds')


def checked_rows(ref, root=ROOT):
    path = Path(ref['file'])
    if not path.is_absolute():
        path = root / path
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref['sha256'] or len(raw) != ref['bytes']:
        raise ValueError('Raw journal no longer matches finalized result: ' + str(path))
    return [json.loads(line) for line in raw.decode().splitlines() if line]


def summarize_decisions(decisions):
    """Deduplicate physical outputs, retaining every logical world decision."""
    physical, failures, unavailable, decision_ids = {}, [], [], set()
    for row in decisions:
        if (not isinstance(row['response'], str) or not isinstance(row['sample_id'], str)
                or not row['sample_id'] or not isinstance(row['available'], bool)):
            raise ValueError('Require actual raw response, sample identity and action availability')
        if row['decision_id'] in decision_ids:
            raise ValueError('Duplicate logical decision')
        decision_ids.add(row['decision_id'])
        try:
            parse_action(row['response'])
            parsed = True
        except ValueError:
            parsed = False
        if not parsed:
            failures.append(row['decision_id'])
        elif not row['available']:
            unavailable.append(row['decision_id'])
        old = physical.setdefault(row['sample_id'], (row['response'], parsed))
        if old != (row['response'], parsed):
            raise ValueError('One physical sample ID has conflicting outputs')
    physical_failures = sum(not parsed for _, parsed in physical.values())
    return {
        'logical_model_decisions': len(decisions),
        'logical_syntax_parse_failures': len(failures),
        'logical_syntax_parse_failure_rate': len(failures) / len(decisions) if decisions else None,
        'syntax_valid_but_unavailable_logical_actions': len(unavailable),
        'physical_model_outputs': len(physical),
        'physical_syntax_parse_failures': physical_failures,
        'physical_syntax_parse_failure_rate': physical_failures / len(physical) if physical else None,
        'parse_failure_decision_ids': failures,
        'syntax_valid_unavailable_decision_ids': unavailable}


def summarize_evaluation(evaluation, rows):
    groups, forced = defaultdict(list), defaultdict(int)
    for row in rows:
        if row.get('kind') != 'decision':
            continue  # Episode summaries repeat traces; count each decision once.
        strategy, sample = row['strategy'], row['sample']
        if sample['forced_observation']:
            forced[strategy] += 1
            continue
        groups[strategy].append({
            'decision_id': row['episode_id'] + ':' + str(row['decision_index']),
            'sample_id': sample['sample_id'], 'response': sample['response'],
            'available': sample['action'] in available_actions(row['public_text'])})
    return {strategy: {**summarize_decisions(groups[strategy]),
        'forced_observations_excluded': forced[strategy],
        'decoding': 'unconstrained greedy' if strategy == 'diag_base' else 'public-select-constrained greedy',
        'availability_basis': 'membership in recorded complete public Available actions'}
        for strategy in evaluation['config']['strategies']}


def summarize_training_updates(rows):
    groups, timing, events = defaultdict(list), {}, defaultdict(list)
    for row in rows:
        update = row['update']
        if row['kind'] == 'decision':
            groups[update].append({
                'decision_id': row['episode_id'] + ':' + str(row['decision_index']),
                'sample_id': row['sample_id'], 'response': row['response'],
                'available': row['after']['action_valid']})
        elif row['kind'] == 'update_complete':
            if update in timing:
                raise ValueError('Duplicate completed update')
            timing[update] = {key: row[key] for key in TIMING_KEYS}
        elif row['kind'] in ('update_begin', 'update_failed'):
            events[update].append(row['kind'])
    return {str(update): {**summarize_decisions(groups[update]), 'timing_seconds': timing.get(update),
            'events': events[update], 'completed': update in timing}
            for update in sorted(set(groups) | set(timing) | set(events))}


def build_report(eval_path, train_paths, root=ROOT):
    evaluation, ref = audited_result(root / eval_path)
    sources = [ref]
    eval_summary = summarize_evaluation(evaluation, checked_rows(evaluation['metrics']['episode_log'], root))
    training_summary = {}
    for result_path in train_paths:
        result, ref = audited_result(root / result_path)
        sources.append(ref)
        rows = []
        for journal_ref in result['metrics']['training_logs']:
            path = Path(journal_ref['file'])
            if path.name.startswith('update_') and path.suffix == '.jsonl':
                rows.extend(checked_rows(journal_ref, root))
        name = result['_meta']['name']
        if name in training_summary:
            raise ValueError('Repeated training source')
        training_summary[name] = {
            'arm': result['config']['arm'], 'source_exit_code': result['_meta']['exit_code'],
            'reported_actual_completed_updates': result['metrics'].get('actual_completed_updates'),
            'step4_passed': result['metrics'].get('step4_passed'),
            'step5_passed': result['metrics'].get('step5_passed'),
            'decoding': 'training policy: temperature0.7, top_p0.95, no action grammar constraint',
            'availability_basis': 'recorded after.action_valid, separately from syntax parsing',
            'updates': summarize_training_updates(rows)}
    metrics = {'original_model_step3_by_strategy': eval_summary,
               'training_by_run_and_update': training_summary, 'paper_result': False}
    config = {'eval': eval_path, 'training': train_paths,
        'parser': 'src.envs.paired_env.parse_action; failure means ValueError on actual raw response',
        'denominators': 'physical outputs deduplicate actual sample_id within strategy/update; logical decisions count both paired worlds',
        'separation': 'Original-model greedy strategy rates and each training update rate remain separate; unavailable canonical actions are not syntax failures',
        'sources_must_be_finalized': True}
    return {'metrics': metrics, 'config': config, 'sources': sources}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--eval', required=True)
    parser.add_argument('--train', action='append', required=True)
    args = parser.parse_args(argv)
    if not re.fullmatch(r'smoke_e2e_[A-Za-z0-9_]+', args.name):
        parser.error('Require a unique smoke_e2e_ result name')
    out = ROOT / 'results' / (args.name + '.json')
    if out.exists():
        raise FileExistsError(out)
    result = build_report(args.eval, args.train)
    with out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('FINAL_METRICS_JSON: ' + json.dumps(result['metrics'], sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

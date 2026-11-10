"""DIAG01 D2/D4: original episode journals and original audited pool metadata.

No training, rendering, re-selection, scientific acceptance or threshold changes.
D1's separately authenticated independent mask index supplies the visibility facts.
"""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import signal
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
KINDS = ('direct_flip', 'direct_stay', 'look_flip', 'look_stay')
ARMS = ('T0', 'T4')
SOURCES = {
    'T0': {'file': 'results/smoke_e2e_pilot_T0_s0_01_retry4.json', 'sha256': '17188af3dff53d6bb07be4b682fdcc4f3f9a3b8f369d12d93d32c3cb195698f2'},
    'T4': {'file': 'results/smoke_e2e_pilot_T4_s0_01_retry4.json', 'sha256': '994e3a7ba47ec29c06891a2ff1b1449439b0b337fc7ea2eef4118e23d2dc8717'},
    'inventory': {'file': 'results/gen_train_look_inventory_first_unit_01.json', 'sha256': 'cc77a010e7eef375a942354e8e89d5403f681b72a77fc44a53506aa41a5dff01'},
}
VERSION = 'diag01_learning_pool_v1'


def need(value, message):
    if not value:
        raise ValueError(message)


def identity(ref):
    return {key: ref[key] for key in ('file', 'sha256')}


def read_bytes(ref, *, root=ROOT, within=None):
    path = (root / ref['file']).resolve()
    path.relative_to((within or root).resolve())
    raw = path.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'Changed evidence bytes: ' + str(path))
    need('bytes' not in ref or len(raw) == ref['bytes'], 'Changed evidence length: ' + str(path))
    return raw


def audit_source(ref):
    from src.tools.t01_partition import audited_result
    data, actual = audited_result(ROOT / ref['file'])
    need(identity(actual) == identity(ref), 'Original result identity changed')
    need(data['_meta'].get('end') and data['_meta']['exit_code'] == 0, 'Unfinished or failed diagnostic source')
    need(data['metrics'].get('errors', []) == [], 'Source errors are not silently dropped')
    return data, actual


def rate(n, d):
    return n / d if d else None


def aggregate(rows):
    n = len(rows)
    success = sum(row['success'] for row in rows)
    reward = math.fsum(row['task_reward_sum'] for row in rows)
    return {'episode_count': n, 'success_count': success, 'success_rate': rate(success, n),
            'task_reward_sum': reward, 'mean_task_reward': rate(reward, n)}


def class_windows(episodes):
    """Episode-weighted diagnostic windows, explicitly not E2E's decision rule."""
    result = []
    for start in (1, 11, 21, 31, 41):
        rows = [r for r in episodes if start <= r['update'] < start + 10]
        result.append({'updates': [start, start + 9], 'by_class': {
            kind: aggregate([r for r in rows if r['kind'] == kind]) for kind in KINDS},
            'aggregation': 'sum of original episode numerators / all episodes in this class and window'})
    return result


def pair_diagnostics(episodes, population):
    result = []
    for key, entry in sorted(population.items()):
        if not entry['kind'].startswith('look_'):
            continue
        all_rows = [r for r in episodes if r['dataset_key'] == key]
        groups = [('all', 1, 50)] + [(f'{s:02d}-{s+9:02d}', s, s+9) for s in (1, 11, 21, 31, 41)]
        windows = []
        for label, lo, hi in groups:
            rows = [r for r in all_rows if lo <= r['update'] <= hi]
            worlds, selections = {}, []
            for world in ('w0', 'w1'):
                subset = [r for r in rows if r['original_world_id'] == world]
                chosen = Counter(r['selected_name'] for r in subset if r['valid_terminal_selection'])
                selections.extend(chosen)
                worlds[world] = {**aggregate(subset), 'candidate_object_counts': dict(sorted(chosen.items())),
                    'invalid_or_no_legal_select_count': sum(not r['valid_terminal_selection'] for r in subset)}
            both = all(worlds[w]['episode_count'] > 0 for w in ('w0', 'w1'))
            # Literal observed universality, not a learned-memory/accuracy threshold.
            same = (len(set(selections)) == 1 and all(r['valid_terminal_selection'] for r in rows)) if both else None
            paired = defaultdict(list)
            for row in rows:
                if row['sampling_design'] == 'both_worlds':
                    paired[row['pair_instance_id']].append(row)
            paired_n = paired_same = paired_valid = 0
            for pair in paired.values():
                need(len(pair) == 2 and {r['original_world_id'] for r in pair} == {'w0', 'w1'},
                     'Incomplete actual paired draw')
                paired_n += 1
                if all(r['valid_terminal_selection'] for r in pair):
                    paired_valid += 1
                    paired_same += pair[0]['selected_name'] == pair[1]['selected_name']
            total = aggregate(rows)
            windows.append({'window': label, 'updates': [lo, hi], **total, 'original_worlds': worlds,
                'both_worlds_observed': both, 'always_same_candidate_in_both_worlds': same,
                'actual_paired_draw_count': paired_n, 'paired_draws_with_two_legal_selections': paired_valid,
                'paired_same_candidate_count': paired_same, 'paired_same_candidate_rate': rate(paired_same, paired_valid),
                'success_minus_half': total['success_rate'] - .5 if total['success_rate'] is not None else None,
                'success_minus_one': total['success_rate'] - 1 if total['success_rate'] is not None else None})
        result.append({'dataset_key': key, 'original_pair_id': entry['original_pair_id'],
            'scene': entry['scene'], 'kind': entry['kind'], 'windows': windows})
    return {'pairs': result, 'scope': 'Observed object choices, not screen-label equality; world IDs are original table worlds. No cutoff defines near-one/near-half and no causal memorization conclusion is inferred.'}


def extract_journal(raw, *, arm, update, population, world_orders, expected_summary):
    """Verify original per-episode accounting and retain only the small diagnosis fields."""
    from src.tools.smoke_e2e_pilot_contract import episode_observation, classified_observations
    decisions, episodes, completed = defaultdict(list), [], []
    for line in raw.splitlines():
        row = json.loads(line)
        if row.get('kind') == 'decision':
            need(row['update'] == update, 'Decision update changed')
            decisions[row['episode_id']].append(row)
        if row.get('record_kind') == 'pilot_episode':
            episodes.append(row)
        if row.get('record_kind') == 'pilot_update_complete':
            completed.append(row)
        need(row.get('record_kind') != 'pilot_update_failed', 'Failed update in purported complete journal')
    need(len(episodes) == 64 and len({r['episode_id'] for r in episodes}) == 64,
         'Expected exactly 64 distinct original episodes')
    need(set(decisions) == {r['episode_id'] for r in episodes}, 'Decision/episode coverage mismatch')
    need(len(completed) == 1 and completed[0]['update'] == update and completed[0]['actual_completed_updates'] == update,
         'Missing original completed update')
    need(completed[0]['by_class'] == expected_summary, 'Journal completion summary differs from sealed parent')
    compact, s0 = [], []
    for row in episodes:
        traces = decisions[row['episode_id']]
        key = row['dataset_key']
        need(key in population and row['pair_id'] == key and row['update'] == update,
             'Episode escaped fixed training population/update')
        entry = population[key]
        need(row['kind'] == entry['kind'] and row['original_pair_id'] == entry['original_pair_id'], 'Episode class/source identity changed')
        need(row['world_id'] in ('w0', 'w1') and row['world_order'] == world_orders[key]
             and row['original_world_id'] == 'w' + str(world_orders[key][int(row['world_id'][1])]), 'World mapping changed')
        need(type(row['success']) is bool and isinstance(row['task_reward_sum'], (int,float))
             and math.isfinite(row['task_reward_sum']), 'Invalid episode outcome')
        final = traces[-1]['after']
        need(final['success'] == row['success'] and final['selected_name'] == row['selected_name'], 'Final decision outcome differs')
        plan = {k: v for k, v in row.items() if k in traces[0] and k not in ('kind', 'update')}
        plan.update(episode_id=row['episode_id'], pair_id=key)
        rebuilt = episode_observation(plan, traces, kind=row['kind'], final_info=final,
                                      selected_name=row['selected_name'], success=row['success'])
        for field in ('task_reward_sum', 'success', 'selected_name', 'actual_terminal_selection',
                      'valid_terminal_selection', 'valid_observation_actions', 'logical_decisions', 'seen_before_select'):
            need(rebuilt[field] == row[field], 'Original episode accounting differs: ' + field)
        if row['valid_terminal_selection']:
            need(row['selected_name'] in {c['object_name'] for c in traces[-1]['before']['candidates'] if c['success_eligible']},
                 'Selected candidate is not original eligible object')
        fields = ('episode_id','dataset_key','original_pair_id','kind','world_id','original_world_id',
                  'world_order','pair_instance_id','sampling_design','success','task_reward_sum','selected_name',
                  'valid_terminal_selection','actual_terminal_selection','valid_observation_actions','logical_decisions')
        compact.append({**{k: copy.deepcopy(row[k]) for k in fields}, 'arm': arm, 'phase': 'train', 'update': update})
        if row['kind'].startswith('look_'):
            first = traces[0]
            need(first['decision_index'] == 0 and len(first['input']['images']) == 1, 'Ambiguous original s0 model image')
            image = first['input']['images'][0]
            need(image['mode'] == 'RGB' and len(image['sha256']) == 64, 'Missing actual input image proof')
            cap = first['before']['capture']
            s0.append({'arm': arm, 'dataset_key': key, 'episode_id': row['episode_id'], 'update': update,
                'original_world_id': row['original_world_id'], 'state_id': cap['state_id'],
                'raw_rgb_sha256': cap['raw_rgb_sha256'], 'marked_rgb_sha256': cap['marked_rgb_sha256'],
                'actual_input_rgb_sha256': image['sha256'], 'actual_input_size': image['size'],
                'image_proof_source': 'original policy-input proof recorded before generation'})
    need(classified_observations(episodes) == expected_summary, 'Recomputed class metrics differ from original parent')
    return compact, s0


def s0_uniqueness(records, population):
    """Actual policy-input hashes and native/marked hashes are never conflated."""
    expected = {key for key, row in population.items() if row['kind'].startswith('look_')}
    groups = {}
    per_pair = {key: {'dataset_key': key, 'original_pair_id': population[key]['original_pair_id'],
        'scene': population[key]['scene'], 'kind': population[key]['kind'], 'worlds': {}} for key in sorted(expected)}
    for mode in ('raw_rgb_sha256', 'marked_rgb_sha256', 'actual_input_rgb_sha256'):
        index = defaultdict(set)
        for row in records:
            need(row['dataset_key'] in expected, 's0 proof escaped training Look population')
            token = row[mode]
            if mode == 'actual_input_rgb_sha256':
                token = json.dumps([row['actual_input_size'], token], separators=(',', ':'))
            index[token].add(row['dataset_key'])
        groups[mode] = {'hash_group_count': len(index), 'cross_pair_duplicate_groups': [
            {'hash_or_size_hash': value, 'dataset_keys': sorted(keys)} for value, keys in sorted(index.items()) if len(keys) > 1]}
    for key, item in per_pair.items():
        for world in ('w0', 'w1'):
            rows = [r for r in records if r['dataset_key'] == key and r['original_world_id'] == world]
            values = {mode: sorted({r[mode] for r in rows}) for mode in groups}
            others = set()
            for row in rows:
                others.update(r['dataset_key'] for r in records if r['dataset_key'] != key
                    and r['actual_input_size'] == row['actual_input_size']
                    and r['actual_input_rgb_sha256'] == row['actual_input_rgb_sha256'])
            item['worlds'][world] = {'observed_episode_count': len(rows), **values,
                'actual_input_sizes': sorted({tuple(r['actual_input_size']) for r in rows}),
                'actual_input_same_as_other_pairs': sorted(others),
                'all_observed_actual_inputs_unique_to_pair': not others if rows else None}
    return {'pairs': list(per_pair.values()), 'groups': groups,
        'scope': 'Exact byte-hash equality only, not perceptual similarity or evidence that the model memorized. Actual model-input hashes come from the SHA-bound original journals; raw/marked 255 images are separate. Unobserved pair/world is unknown, not unique.'}


def shadow_diagnostics(episodes, independent_rows):
    """Consume D1 independent masks; original post-tau semantics are step >= tau."""
    index = {}
    for row in independent_rows:
        if row['phase'] != 'train': continue
        token = (row['arm'], row['update'], row['episode_id'])
        need(token not in index, 'Duplicate D1 episode')
        index[token] = row
    output = []
    for arm in ARMS:
        for lo in (1, 11, 21, 31, 41):
            selected = [r for r in episodes if r['arm'] == arm and r['kind'] == 'look_flip'
                        and lo <= r['update'] < lo + 10 and r['valid_observation_actions'] > 0]
            states_all, matching, strict, evidence_episodes = [], [], [], 0
            for episode in selected:
                token = (arm, episode['update'], episode['episode_id'])
                need(token in index, 'D1 omitted an observed LF episode')
                actual = index[token]
                need(actual['dataset_key'] == episode['dataset_key']
                     and actual['original_world_id'] == episode['original_world_id'], 'D1 episode/world identity differs')
                tau = actual['independent_first_difference_step']
                states = actual['states']
                need([s['observation_index'] for s in states] == list(range(len(states))), 'Incomplete D1 state history')
                found = []
                for state in states:
                    need(type(state['own_anchor_pixels']) is int and state['own_anchor_pixels'] >= 0
                         and type(state['shared']) is bool, 'Invalid D1 independent state measurement')
                    ref = {'episode_id': episode['episode_id'], 'dataset_key': episode['dataset_key'],
                           'original_world_id': episode['original_world_id'], **state}
                    states_all.append(ref)
                    if tau is not None and state['observation_index'] >= tau and state['own_anchor_pixels'] == 0:
                        found.append(ref); matching.append(ref)
                        if state['observation_index'] > tau: strict.append(ref)
                evidence_episodes += bool(found)
            output.append({'arm': arm, 'updates': [lo, lo + 9], 'observing_LF_episode_count': len(selected),
                'visited_state_occurrences': len(states_all), 'post_tau_ownX_zero_state_occurrences': len(matching),
                'strictly_after_tau_ownX_zero_state_occurrences': len(strict),
                'episodes_with_post_tau_ownX_zero': evidence_episodes,
                'unique_dataset_world_states': len({(r['dataset_key'],r['original_world_id'],r['state_id']) for r in matching}),
                'matching_state_occurrences': matching})
    return {'windows': output, 'scope': 'All observed states including s0 and after-action terminal/repeated states; occurrences and unique states separately. Post-tau is >= first independently nonshared state (original pre_evidence complement); strict > is auxiliary. Only LF episodes with an actually valid observation action. No shadow-to-success causal claim.'}


def pool_diagnostic(inventory, frozen, *, postfilter_fn, minimum):
    rows = inventory['qualified_rows']
    need(len(rows) == 83 and len({r['dataset_key'] for r in rows}) == 83, 'D4 requires the complete fixed 83 population')
    heldout = set(frozen['selected_scenes'])
    need(inventory['additional_cache_sources'] == [] and inventory['known_rejection_sources'] == [], 'D4 original cutoff changed')
    kept = [r for r in rows if r['scene'] not in heldout]
    displaced = [r for r in kept if not r['kind'].endswith('stay') or r['stay_displacement']['distance_m'] >= minimum]
    original = postfilter_fn(rows, heldout)
    for field in ('confirmed_by_kind', 'selected_dataset_keys', 'filter_omissions', 'balance'):
        need(original[field] == inventory[field], 'Original postfilter replay differs: ' + field)
    selected = [r for r in rows if r['dataset_key'] in set(original['selected_dataset_keys'])]
    def stage(name, values):
        return {'stage': name, 'counts_by_kind': {k: sum(r['kind'] == k for r in values) for k in KINDS},
                'look_stay_dataset_keys': sorted(r['dataset_key'] for r in values if r['kind'] == 'look_stay')}
    rejected = [r for r in kept if r['kind'] == 'look_stay' and r['stay_displacement']['distance_m'] < minimum]
    numbers = sorted(r['stay_displacement']['distance_m'] for r in rejected)
    need(all(type(v) in (int,float) and math.isfinite(v) and v >= 0 for v in numbers), 'Missing LS displacement')
    return {'stages': [stage('original83_T02_and_cache_certified', rows), stage('fixed_holdout_excluded', kept),
            stage('original_Stay_displacement_filter', displaced), stage('original_scene_AB_balance_seed0', selected)],
        'stay_threshold_metres': minimum, 'threshold_rule': 'retain distance_m >= threshold; exact boundary accepted',
        'rejected_LS_displacements_metres': numbers, 'rejected_LS_distribution': {'count': len(numbers),
            'minimum': min(numbers) if numbers else None, 'maximum': max(numbers) if numbers else None,
            'median': statistics.median(numbers) if numbers else None,
            'mean': math.fsum(numbers)/len(numbers) if numbers else None},
        'rejected_LS_pairs': [{k: copy.deepcopy(r[k]) for k in ('dataset_key','original_pair_id','scene','pair_reference','stay_displacement')} for r in rejected],
        'all_LS_inputs': [copy.deepcopy(r) for r in rows if r['kind'] == 'look_stay'],
        'filter_omissions': original['filter_omissions'], 'balance': original['balance'],
        'measurement_sources': inventory['input_metadata_references'],
        'scope': 'Original audited inventory measurements; replay unchanged postfilter and balance only. No full native reauthentication, no newly frozen formal T04 pool.',
        'formal_T04_pool_authorized': False}


def main():
    sys.path.insert(0, str(ROOT))
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True); p.add_argument('--d1-result', required=True)
    p.add_argument('--timeout-seconds', type=int, default=300)
    a = p.parse_args()
    need(a.name.startswith('diag01_') and a.name.replace('_','').isalnum(), 'Unique diag01 name required')
    need(0 < a.timeout_seconds <= 600, 'CPU stop bound must be at most 600 seconds')
    out = ROOT/'logs/review'/a.name
    need(not out.exists() and not (ROOT/'results'/(a.name+'.json')).exists(), 'Preserve previous attempts')
    out.mkdir(parents=True)
    config = {'name': a.name, 'gpus': 0, 'version': VERSION, 'original_sources': SOURCES,
              'd1_result': json.loads(a.d1_result), 'timeout_seconds': a.timeout_seconds}
    metrics = {'kind': VERSION, 'complete': False, 'errors': [], 'producer_exit_code': 1,
               'native_scientific_revalidation_performed': False, 'paper_result': False}
    sources = []; code = 1; started = time.monotonic()
    def expired(*_): raise TimeoutError('DIAG01 CPU stop bound exceeded')
    signal.signal(signal.SIGALRM, expired); signal.alarm(a.timeout_seconds)
    try:
        from src.tools import g01_train_inventory as inventory_module
        from src.tools.train_holdout_freeze import load_frozen_holdout, assert_pilot_split
        data = {}
        for key, ref in SOURCES.items():
            data[key], actual = audit_source(ref); sources.append(actual)
        inv = data['inventory']['metrics']['inventory']
        need(data['inventory']['metrics']['passed'] is True and inv['formal_pool_authorized'] is False,
             'Original inventory not a successful diagnostic receipt')
        cfg = data['T0']['config']; manifest = json.loads(read_bytes(cfg['manifest_reference']))
        frozen, frozen_proof = load_frozen_holdout(cfg['holdout_reference'])
        assert_pilot_split(manifest['entries'], manifest['split'], frozen)
        need(identity(inv['manifest_reference']) == identity(cfg['manifest_reference'])
             and identity(inv['holdout_reference']) == identity(cfg['holdout_reference']), 'D4 uses another population/holdout')
        train_keys = set(data['T0']['metrics']['dataset_proof']['train_keys'])
        population = {r['dataset_key']: r for r in manifest['entries'] if r['dataset_key'] in train_keys}
        need(set(population) == train_keys and len(train_keys) == 63, 'Training population incomplete')
        all_episodes, all_s0, journal_refs, arms = [], [], [], {}
        for arm in ARMS:
            d = data[arm]; m = d['metrics']
            need(d['config']['arm'] == arm and m['pilot_completed'] is True and m['actual_completed_updates'] == 50
                 and m['errors'] == [] and m['cleanup_errors'] == [], 'Original arm not complete')
            need(d['config']['manifest_reference'] == cfg['manifest_reference']
                 and d['config']['holdout_reference'] == cfg['holdout_reference']
                 and m['common_pool_hash'] == data['T0']['metrics']['common_pool_hash']
                 and m['common_config_sha256'] == data['T0']['metrics']['common_config_sha256'], 'Arm population/method differs')
            episodes = []
            need([r['update'] for r in m['updates']] == list(range(1,51)), 'Missing update')
            for update in m['updates']:
                ref = update['episode_log']; raw = read_bytes(ref, within=ROOT/'logs/runs')
                rows, s0 = extract_journal(raw, arm=arm, update=update['update'], population=population,
                    world_orders=m['dataset_proof']['world_orders'], expected_summary=update['by_class'])
                episodes.extend(rows); all_s0.extend(s0); journal_refs.append(ref)
            all_episodes.extend(episodes)
            arms[arm] = {'ten_update_windows': class_windows(episodes), 'pair_diagnostics': pair_diagnostics(episodes,population)}
        d1, actual_d1 = audit_source(config['d1_result']); sources.append(actual_d1)
        need(d1['metrics']['complete'] is True, 'D1 index is incomplete')
        # D1 diagnostic report is immutable and contains independently read masks.
        d1_report = json.loads(read_bytes(d1['metrics']['report']))
        for arm in ARMS:
            need(any(identity(ref)==identity(SOURCES[arm]) for ref in d1_report['original_sources']), 'D1 does not cover same sealed arm')
        need(identity(d1_report['population_manifest']) == identity(cfg['manifest_reference']), 'D1 population differs')
        d1_s0 = {(r['dataset_key'],r['original_world_id']): r for r in d1_report['look_s0']}
        for row in all_s0:
            independent = d1_s0[(row['dataset_key'],row['original_world_id'])]
            need(independent['train_or_eval'] == 'train'
                 and all(independent[k] == row[k] for k in ('raw_rgb_sha256','marked_rgb_sha256'))
                 and {'mode':'RGB','size':row['actual_input_size'],'sha256':row['actual_input_rgb_sha256']}
                     in independent['actual_input_images'], 'D1 actual s0 bytes differ from original policy-input record')
        report = {'version': VERSION, 'sources': sources, 'original_journals': journal_refs,
            'common_pool_hash': data['T0']['metrics']['common_pool_hash'],
            'common_config_sha256': data['T0']['metrics']['common_config_sha256'],
            'manifest_reference': cfg['manifest_reference'], 'holdout_reference': cfg['holdout_reference'],
            'holdout_proof': frozen_proof, 'D2': {'arms': arms, 's0_mask_RGB_index_source': d1['metrics']['report'],
                's0_raw_marked_bytes_independently_verified_by_D1': True,
                'shadow_states': shadow_diagnostics(all_episodes,d1_report['episodes']),
                's0_inputs': s0_uniqueness(all_s0,population)},
            'D4': pool_diagnostic(inv, frozen, postfilter_fn=inventory_module.postfilter, minimum=inventory_module.MIN_STAY_METRES),
            'limitations': ['Exact s0 differences allow pair identification in principle but do not prove model memorization.',
                'Per-pair observed accuracy is descriptive; no post-hoc near-one/half threshold.',
                'Diagnostic results neither change acceptance nor authorize the main experiment.']}
        from src.tools.t02_export import save_json
        metrics.update(report=save_json(out/'report.json',report), complete=True, producer_exit_code=0)
        code = 0
    except BaseException as error:
        metrics['errors'].append(type(error).__name__ + ': ' + str(error)); traceback.print_exc()
    finally:
        signal.alarm(0); metrics['elapsed_seconds'] = time.monotonic()-started
        from src.tools.t02_export import save_json
        save_json(ROOT/'results'/(a.name+'.json'), {'config': config, 'metrics': metrics, 'sources': sources})
        print('FINAL_METRICS_JSON: '+json.dumps(metrics,sort_keys=True,allow_nan=False),flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())

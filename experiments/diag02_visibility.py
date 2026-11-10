"""DIAG02 E1/E2: original pilot regression and measured table co-visibility.

No renderer, model, new table authority, historical-result edit or pool freeze.
"""
import argparse
from collections import Counter, deque
import json
from pathlib import Path
import re
import signal
import time
import traceback

from experiments.diag01_visibility import (QUALIFICATION, PARENTS, Reads, TableIndex,
    load_journal, need, reference)
from src.envs.thor_common import ROOT
from src.envs.pair_table import eligible_names, label_assignment
from src.tools import smoke_e2e_pilot_certificates as certificates
from src.tools import smoke_e2e_pilot_contract as pilot
from src.tools import train_holdout_freeze as holdout
from src.tools.smoke_e2e_pilot import completed_worker
from src.tools.t02_export import save_json

VERSION = 'diag02_seen_regression_and_cov_v1'
DIAG01 = {'file': 'results/diag01_visibility_02.json',
          'sha256': '2cbe8598cacbdcaee1a2ae2679a4a13dad449f54e5e6e2f6252fb6d0ab885ed8'}


def regression_summary(rows):
    return {'look_episodes': len(rows),
        'historical_seen': sum(r['historical_seen'] for r in rows),
        'corrected_seen': sum(r['corrected_seen'] for r in rows),
        'independent_diag01_seen': sum(r['independent_diag01_seen'] for r in rows),
        'corrected_vs_independent_mismatches': sum(r['corrected_seen'] != r['independent_diag01_seen'] for r in rows),
        'changed_episode_count': sum(r['historical_seen'] != r['corrected_seen'] for r in rows),
        'denominator': 'all original Look episodes in this arm/phase/window, including no parsed select'}


def recompute_observation(episode, traces, independent):
    """Call the repaired production function on the original complete journal."""
    actual = pilot.episode_observation(episode, traces, kind=episode['kind'],
        final_info=traces[-1]['after'], selected_name=episode['selected_name'], success=episode['success'])
    changed_fields = {'seen_before_select', 'seen_before_select_reason'}
    for field, value in actual.items():
        if field not in changed_fields:
            need(episode[field] == value, 'E1 changed a non-seen outcome: ' + field)
    need(actual['seen_before_select'] == independent['seen_before_select'],
         'Repaired production seen differs from independent raw-mask DIAG01')
    need(episode['seen_before_select'] == independent['logged_seen_before_select']
         and episode['x_seen_step'] == independent['logged_x_seen_step'], 'Original episode differs from DIAG01')
    return {'episode_id': episode['episode_id'], 'dataset_key': episode['dataset_key'],
        'kind': episode['kind'], 'world_id': episode['world_id'],
        'x_seen_step': episode['x_seen_step'], 'select_decision_index': actual['select_decision_index'],
        'historical_seen': episode['seen_before_select'], 'corrected_seen': actual['seen_before_select'],
        'independent_diag01_seen': independent['seen_before_select'],
        'historical_reason': episode['seen_before_select_reason'],
        'corrected_reason': actual['seen_before_select_reason'],
        'all_other_episode_fields_unchanged': True}


def seen_regression(manifest, manifest_ref, reads, sources):
    diag, diag_ref = certificates._audited(DIAG01, ROOT)
    need(diag['_meta']['exit_code'] == 0 and diag['metrics']['complete'] is True, 'DIAG01 incomplete')
    sources.append(diag_ref); reads.read(DIAG01)
    independent = reads.json(diag['metrics']['report'])
    need(independent['population_manifest'] == manifest_ref, 'DIAG01 population differs')
    by_episode = {(r['arm'], r['phase'], r['update'], r['episode_id']): r for r in independent['episodes']}
    need(len(by_episode) == len(independent['episodes']), 'Duplicate independent episode')
    rows = []; consumed = set()
    for arm, ref in PARENTS.items():
        parent, actual = certificates._audited(ref, ROOT)
        completed_worker(parent['metrics'], parent['config'])
        need(parent['_meta']['exit_code'] == 0 and parent['metrics']['pilot_completed'] is True
             and parent['metrics']['errors'] == [] and parent['metrics']['cleanup_errors'] == []
             and parent['config']['arm'] == arm and parent['config']['updates'] == 50
             and parent['config']['manifest_reference'] == manifest_ref, 'Original complete pilot changed')
        frozen, frozen_ref = holdout.load_frozen_holdout(parent['config']['holdout_reference'], root=ROOT)
        holdout.assert_pilot_split(manifest['entries'], manifest['split'], frozen)
        sources.append(actual); reads.read(ref); reads.read(frozen_ref)
        for phase, items, marker in [('train', parent['metrics']['updates'], 'pilot_update_complete'),
                ('holdout', parent['metrics']['holdout_evaluations'], 'pilot_holdout_complete')]:
            for item in items:
                episodes, decisions = load_journal(item, reads, marker)
                for episode in episodes:
                    if not episode['kind'].startswith('look_'): continue
                    key = (arm, phase, item['update'], episode['episode_id'])
                    need(key in by_episode and key not in consumed, 'Missing/duplicate independent episode')
                    consumed.add(key)
                    row = recompute_observation(episode, decisions[episode['episode_id']], by_episode[key])
                    rows.append({'arm': arm, 'phase': phase, 'update': item['update'], **row})
            print('DIAG02_E1_PHASE:', arm, phase, flush=True)
    need(consumed == set(by_episode), 'Did not regress every DIAG01 episode')
    windows = []
    for arm in PARENTS:
        for phase, spans in [('train', [(n,n+9) for n in range(1,51,10)]),
                             ('holdout', [(n,n) for n in (0,20,40,50)])]:
            for low, high in spans:
                chosen = [r for r in rows if r['arm'] == arm and r['phase'] == phase and low <= r['update'] <= high]
                windows.append({'arm': arm, 'phase': phase, 'first_update': low, 'last_update': high,
                                **regression_summary(chosen)})
    return {'production_function': reference(ROOT/'src/tools/smoke_e2e_pilot_contract.py'),
        'independent_diag01': diag['metrics']['report'], 'episode_rows': rows, 'windows': windows,
        'totals': [{'arm': arm, 'phase': phase, **regression_summary([r for r in rows
                   if r['arm'] == arm and r['phase'] == phase])} for arm in PARENTS for phase in ('train','holdout')],
        'historical_results_modified': False, 'all_non_seen_episode_fields_unchanged': True,
        'scope': 'Repaired production function on original journals, compared to original independent mask-derived DIAG01; original learning-decision results are not overwritten.'}


def shortest_paths(header, depth=4):
    """Shortest actual transition paths; fixed lexical tie break, no outcome sorting."""
    need(depth == 4, 'This diagnostic uses the fixed D=4 table')
    initial = header['initial_state']; states = header['states']
    need(initial in states, 'Missing initial state')
    paths = {initial: []}; queue = deque([initial])
    while queue:
        state = queue.popleft(); path = paths[state]
        if len(path) == depth: continue
        for action, destination in sorted(states[state]['transitions'].items()):
            need(destination in states, 'Transition destination missing from original table')
            if destination not in paths:
                paths[destination] = path + [action]; queue.append(destination)
    return paths


class EligibilityProjection:
    """Only the real mask counts/distances used by the original eligible_names."""
    def __init__(self, counts, distances):
        self.counts, self.distances = counts, distances

    def count(self, name):
        return self.counts[name]


def cov_measurement(index, key, state_id, original_world):
    header = index.header(key); state = header['states'][state_id]
    anchor = index.state(key, state_id, original_world)
    public_world = 0 if state['shared'] else original_world
    capture = state['captures'][public_world]
    names = header['candidate_names']; labels = label_assignment(header['pair_id'], names)
    refs = {name: index.payload(key, capture['masks'][name]) for name in names}
    counts = {name: index.reads.mask(refs[name]) for name in names}
    eligible = eligible_names(EligibilityProjection(counts, capture['distances']), names)
    both = len(eligible) == 2
    return {'state_id': state_id, 'own_anchor_pixels': anchor['own_anchor_pixels'],
        'anchor_mask_reference': anchor['anchor_mask_reference'],
        'anchor_visible_250': anchor['own_anchor_pixels'] >= 250,
        'public_capture_original_world_id': 'w'+str(public_world),
        'candidate_measurements': [{'name': name, 'mask_pixels': counts[name],
            'distance_m': capture['distances'].get(name), 'label_if_marked': labels[name] if name in eligible else None,
            'success_eligible': name in eligible, 'mask_reference': refs[name]} for name in names],
        'both_candidates_marked': both,
        'anchor_and_both_marked': anchor['own_anchor_pixels'] >= 250 and both}


def describe_world(header, original_world, measurement):
    paths = shortest_paths(header)
    measured = {state: measurement(state) for state in paths}
    first = [{'action': action, **measured[state]} for action, state in sorted(
        header['states'][header['initial_state']]['transitions'].items())]
    reveals = [row for row in first if row['anchor_visible_250']]
    cov = [{'actions': path, 'observation_steps': len(path), **measured[state]}
           for state, path in paths.items() if measured[state]['anchor_and_both_marked']]
    ordered_cov = sorted(cov, key=lambda r: (r['observation_steps'], r['actions'], r['state_id']))
    return {'original_world_id': 'w'+str(original_world), 'all_one_step_actions': first,
        'one_step_revealing_actions': reveals,
        'has_one_step_reveal': bool(reveals),
        'has_one_step_reveal_with_both_marked': any(r['both_candidates_marked'] for r in reveals),
        'all_one_step_reveals_have_both_marked': bool(reveals) and all(r['both_candidates_marked'] for r in reveals),
        'reachable_within_D': len(paths), 'has_cov_state_within_D': bool(cov),
        'minimum_observation_steps_to_cov': ordered_cov[0]['observation_steps'] if cov else None,
        'shortest_cov_witness': ordered_cov[0] if cov else None,
        'all_cov_states': ordered_cov,
        'reachable_state_measurements': [{'shortest_actions': paths[state], **value} for state,value in measured.items()]}


def cov_summary(rows):
    n = len(rows); reveals = [a for row in rows for a in row['one_step_revealing_actions']]
    count = lambda name: sum(row[name] for row in rows)
    ratio = lambda a,b: a/b if b else None
    steps = Counter(row['minimum_observation_steps_to_cov'] for row in rows)
    return {'worlds': n,
        **{name+'_count': count(name) for name in ('has_one_step_reveal', 'has_one_step_reveal_with_both_marked',
           'all_one_step_reveals_have_both_marked', 'has_cov_state_within_D')},
        'one_step_cov_fraction_all_look_worlds': ratio(count('has_one_step_reveal_with_both_marked'),n),
        'within_D_cov_fraction_all_look_worlds': ratio(count('has_cov_state_within_D'),n),
        'one_step_revealing_action_count': len(reveals),
        'one_step_revealing_actions_with_both_marked': sum(r['both_candidates_marked'] for r in reveals),
        'fraction_revealing_actions_with_both_marked': ratio(sum(r['both_candidates_marked'] for r in reveals),len(reveals)),
        'minimum_observation_steps_histogram': {str(step): steps[step] for step in range(5)},
        'no_cov_state_within_D': steps[None],
        'world_denominator': 'each original world of every Look pair in the entire fixed83 population, including holdout',
        'action_denominator': 'all original s0 transitions whose own-world X mask has at least250 pixels; no chosen direction or favorable state selection'}


def cov_diagnosis(manifest, reads):
    index = TableIndex(manifest['entries'], reads); rows = []
    for key, entry in index.entries.items():
        if not entry['kind'].startswith('look_'): continue
        header = index.header(key)
        for original in (0,1):
            row = describe_world(header, original, lambda state: cov_measurement(index,key,state,original))
            rows.append({'dataset_key': key, 'original_pair_id': header['pair_id'],
                'scene': header['scene'], 'kind': header['kind'], 'world_order_in_pilot': entry['world_order'], **row})
        print('DIAG02_E2_PAIR:', header['pair_id'], flush=True)
    return {'D': 4, 'worlds': rows, 'summary': cov_summary(rows),
        'by_class': {kind: cov_summary([r for r in rows if r['kind'] == kind]) for kind in ('look_flip','look_stay')},
        'marking_definition': 'Original eligible_names: actual public candidate mask >0 and measured distance <=1.5m; stable original label_assignment. Own-world X mask >=250.',
        'minimum_steps_definition': 'Breadth-first search over the complete original directed transition table, depth<=4 observation actions. One following select fits the existing D+1 model-call budget.',
        'payload_scope': 'Actual original training-table X and candidate binary masks plus distances; no RGB rerender, no new export/cache authority or dataset changes.'}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--name',required=True)
    parser.add_argument('--part',choices=('e1','e2'),required=True)
    parser.add_argument('--timeout-seconds',type=int,default=300)
    args = parser.parse_args()
    need(re.fullmatch(r'diag02_[A-Za-z0-9_]+',args.name),'Unique DIAG02 result name required')
    need(30 <= args.timeout_seconds <= 300,'Bounded CPU metadata/mask diagnosis only')
    output = ROOT/'results'/(args.name+'.json'); need(not output.exists(),'Preserve prior result')
    directory = ROOT/'logs/review/diag02'/args.name; directory.mkdir(parents=True,exist_ok=False)
    def expired(*_): raise TimeoutError('DIAG02 CPU bound reached; retain partial output')
    signal.signal(signal.SIGALRM,expired); signal.alarm(args.timeout_seconds)
    started=time.monotonic(); reads=Reads(); sources=[]
    metrics={'kind':VERSION,'part':args.part,'complete':False,'errors':[],'gpus':0}; code=1
    try:
        manifest,proof,source,manifest_ref = certificates.authenticate_certificate_qualification(QUALIFICATION)
        sources.append(source); reads.read(manifest_ref); reads.read(QUALIFICATION)
        report={'version':VERSION,'part':args.part,'population_manifest':manifest_ref,
            'qualification':source,'population_pairs':len(manifest['entries'])}
        if args.part == 'e1':
            report['E1']=seen_regression(manifest,manifest_ref,reads,sources)
            metrics['totals']=report['E1']['totals']; metrics['windows']=report['E1']['windows']
        else:
            report['E2']=cov_diagnosis(manifest,reads)
            metrics['summary']=report['E2']['summary']; metrics['by_class']=report['E2']['by_class']
        # Each recorded payload was SHA/size checked when used. Results and
        # population identity are rechecked; do not re-read gigabytes needlessly.
        reads.read(manifest_ref); reads.read(QUALIFICATION)
        for source in sources: reads.read(source)
        report['source_files_read']=list(reads.refs.values())
        report['elapsed_seconds']=time.monotonic()-started
        metrics.update(complete=True,report=save_json(directory/'report.json',report),
            original_results_modified=False,new_acceptance_authority=False)
        code=0
    except BaseException as exc:
        traceback.print_exc(); metrics['errors'].append(type(exc).__name__+': '+str(exc))
    finally: signal.alarm(0)
    metrics.update(producer_exit_code=code,elapsed_seconds=time.monotonic()-started)
    save_json(output,{'metrics':metrics,'config':{**vars(args),'version':VERSION,'gpus':0},'sources':sources})
    print('FINAL_METRICS_JSON: '+json.dumps(metrics,sort_keys=True,allow_nan=False),flush=True)
    return code


if __name__ == '__main__': raise SystemExit(main())

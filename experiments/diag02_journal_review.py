"""Independent CPU review of sealed DIAG02 journals against fixed83 and E2.

Incomplete source runs remain incomplete. No new GPU/native/cache authority.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import signal
import time
import traceback

from src.envs.thor_common import ROOT
from src.envs.paired_env import parse_action
from src.tools.t01_partition import audited_result
from src.tools.t02_export import save_json

STRATEGIES = ('obs_never','obs_trigger','obs_trigger_cov')
KINDS = ('direct_flip','direct_stay','look_flip','look_stay')
TURNS = ('turn_left','turn_right','turn_right','turn_left')
E2 = {'file':'results/diag02_cov_01.json',
      'sha256':'078006b31464bb37db93373eb4a83c8cb25dfbd1b153a3cf7bc219a3d455dc8f'}
VERSION = 'diag02_sealed_journal_e2_cross_review_v1'


def need(value, message):
    if not value: raise ValueError(message)


def read(ref):
    path=(ROOT/ref['file']).resolve(); raw=path.read_bytes()
    need(hashlib.sha256(raw).hexdigest()==ref['sha256'] and
         ('bytes' not in ref or len(raw)==ref['bytes']), 'Referenced bytes changed: '+str(path))
    return raw


def ref_key(ref):
    return str((ROOT/ref['file']).resolve()),ref['sha256']


def output_identity(sample):
    return {name:sample[name] for name in ('response','action','response_token_ids','finish_reason','stop_reason')}


def expected_forced(strategy, step, first_seen, anchor_pixels, both_marked):
    if strategy=='obs_never': return None
    need(strategy in STRATEGIES,'Unknown strategy')
    if strategy=='obs_trigger_cov' and anchor_pixels>=250 and both_marked: return None
    if step>=4 or (first_seen is not None and step%2==0): return None
    return TURNS[step]


def e2_world_index(report):
    worlds={}
    for row in report['E2']['worlds']:
        key=(row['dataset_key'],row['original_world_id'])
        need(key not in worlds,'Duplicate E2 original world')
        states={s['state_id']:s for s in row['reachable_state_measurements']}
        need(len(states)==len(row['reachable_state_measurements']),'Duplicate E2 state')
        worlds[key]={'world':row,'states':states}
    return worlds


def bind_e2_state(state, known):
    need(state['state_id'] in known['states'],'Observed state not present in E2 D4 graph')
    measured=known['states'][state['state_id']]
    need(state['anchor_pixels']==measured['own_anchor_pixels'] and
         state['anchor_mask_sha256']==measured['anchor_mask_reference']['sha256'],
         'GPU mask count/SHA differs from independent E2 original bytes')
    return measured


def cov_crosscheck(row, known):
    actual=bool(row['selected'] and row['selection_frame_anchor_pixels'] is not None and
                row['selection_frame_anchor_pixels']>=250 and row['selection_frame_both_candidates_marked'])
    state=row['states'][-2]
    measured=bind_e2_state(state,known)
    need(row['selection_frame_anchor_pixels']==(state['anchor_pixels'] if row['selected'] else None)
         and row['selection_frame_both_candidates_marked']==(measured['both_candidates_marked'] if row['selected'] else None),
         'Selection frame fields disagree with E2 original state')
    if actual:
        need(measured['anchor_and_both_marked'] and known['world']['has_cov_state_within_D'],
             'Actual cov selection is absent from E2 graph existence')
        one_step={s['state_id'] for s in known['world']['all_one_step_actions'] if s['anchor_and_both_marked']}
        need(state['state_id'] in one_step,'Fixed L/R/R/L cov selected outside one-step reveal states')
    return {'dataset_key':row['dataset_key'],'pair_id':row['pair_id'],'kind':row['pair_kind'],
        'world_id':row['world_id'],'original_world_id':row['original_world_id'],
        'selected_state_id':state['state_id'],'selection_frame_anchor_pixels':row['selection_frame_anchor_pixels'],
        'selection_frame_both_candidates_marked':row['selection_frame_both_candidates_marked'],
        'actual_cov_selection':actual,'e2_has_cov_state_within_D':known['world']['has_cov_state_within_D'],
        'e2_has_one_step_cov':known['world']['has_one_step_reveal_with_both_marked'],
        'e2_minimum_observation_steps':known['world']['minimum_observation_steps_to_cov'],
        'actual_observation_count':row['observation_count'],'success':row['success']}


def inspect_episode(row, decisions, strategy, entries, e2_index):
    key=(row['dataset_key'],row['world_id']); need(key[0] in entries and key[1] in ('w0','w1'),'Unknown original episode identity')
    entry=entries[key[0]]; original='w'+str(entry['world_order'][int(key[1][-1])])
    need(row['pair_id']==entry['original_pair_id'] and row['pair_kind']==entry['kind'] and
         row['world_order']==entry['world_order'] and row['original_world_id']==original,'Original world mapping/class differs')
    need(1<=len(decisions)<=5 and [d['decision_index'] for d in decisions]==list(range(len(decisions)))
         and all((d['dataset_key'],d['world_id'])==key for d in decisions)
         and all(d['done'] is False for d in decisions[:-1]) and decisions[-1]['done'] is True,
         'Incomplete decision sequence or wrong terminal order')
    states=[decisions[0]['before_state']]+[d['after_state'] for d in decisions]
    need(states==row['states'] and [s['step'] for s in states]==list(range(len(states))) and
         all(d['before_state']==states[i] and d['after_state']==states[i+1] for i,d in enumerate(decisions)),
         'Episode state chain differs')
    known=e2_index.get((key[0],original))
    need(not entry['kind'].startswith('look_') or known is not None,'Look original world missing from E2')
    for state in states:
        need(state['original_world_id']==original and type(state['anchor_pixels']) is int and
             0<=state['anchor_pixels']<=255*255 and re.fullmatch('[0-9a-f]{64}',state['anchor_mask_sha256']),
             'Invalid original-world mask record')
        if known is not None: bind_e2_state(state,known)
    first=next((s['step'] for s in states if s['anchor_pixels']>=250),None)
    outputs=[]; syntax=observations=attempts=0; reward=cost=0.; parsed=[]
    for i,d in enumerate(decisions):
        sample=d['sample']; prior=next((s['step'] for s in states[:i+1] if s['anchor_pixels']>=250),None)
        if known is not None:
            need(d['both_candidates_marked_before']==bind_e2_state(states[i],known)['both_candidates_marked'],
                 'GPU marks differ from actual E2 eligibility')
        forced=expected_forced(strategy,i,prior,states[i]['anchor_pixels'],d['both_candidates_marked_before'])
        need(sample['forced_observation'] is (forced is not None),'Strategy observation/selection schedule differs')
        if forced is not None:
            need(sample['action']==sample['response']==forced and sample['sample_id'] is None and
                 sample['response_token_ids']==[] and sample['response_token_count']==0,'Forced observation changed')
        else:
            need(sample['sample_id'] and sample['response_token_count']==len(sample['response_token_ids']) and
                 0<sample['response_token_count']<=256 and sample['input']['public_image_sha256']==
                 [x['public_rgb_sha256'] for x in decisions[:i+1]],'Actual model output/history incomplete')
            need(sample['request_descriptor']['input']==sample['input'],'Descriptor input proof differs')
            need(sample['action'] in sample['input']['guided_select_choices'],'Generated action not a public guided choice')
            outputs.append(sample)
        try: action=parse_action(sample['action'])
        except ValueError: action=None; syntax+=int(not sample['forced_observation'])
        parsed.append(action); reward+=d['reward']
        if action is not None and action.kind in ('turn_left','turn_right','move'):
            attempts+=1; observations+=int(i<4 and d['action_valid']); cost+=max(0.,-d['reward'])
        if not sample['forced_observation']:
            need(action is not None and action.kind=='select' and d['action_valid'] is True,'Forced-select output invalid')
    selected=parsed[-1] is not None and parsed[-1].kind=='select'
    need(row['actions']==[d['sample']['response'] for d in decisions] and row['steps']==len(decisions) and
         row['selected'] is selected and row['selected_object']==decisions[-1]['selected_object'] and
         row['success'] is bool(decisions[-1]['success']),'Episode result differs from final decision')
    need(row['independent_first_seen_step']==first and row['seen_before_select'] is
         bool(selected and first is not None and first<len(decisions)) and row['seen_anywhere'] is (first is not None)
         and row['positive_anchor_anywhere'] is any(s['anchor_pixels']>0 for s in states),'Seen boundary differs')
    need(row['model_outputs']==row['model_calls']==len(outputs) and row['syntax_failures']==syntax and
         row['observation_count']==observations and row['observation_attempts']==attempts and
         row['observation_reward_cost']==cost and row['task_reward']==reward,'Episode cost/call totals differ')
    need(row['selection_frame_anchor_pixels']==(states[-2]['anchor_pixels'] if selected else None) and
         row['selection_frame_both_candidates_marked']==(decisions[-1]['both_candidates_marked_before'] if selected else None),
         'Selection frame differs from before-select decision state')
    return outputs, cov_crosscheck(row,known) if known is not None and strategy=='obs_trigger_cov' else None


def summarize(rows):
    result={}
    for kind in (*KINDS,'look_all','all'):
        selected=[r for r in rows if kind=='all' or r['pair_kind']==kind or
                  (kind=='look_all' and r['pair_kind'].startswith('look_'))]
        n=len(selected); ratio=lambda value:value/n if n else None
        result[kind]={'episodes':n,'successes':sum(r['success'] for r in selected),
            'success_rate':ratio(sum(r['success'] for r in selected)),
            'observation_count_mean':ratio(sum(r['observation_count'] for r in selected)),
            'observation_reward_cost_mean':ratio(sum(r['observation_reward_cost'] for r in selected)),
            'task_reward_mean':ratio(sum(r['task_reward'] for r in selected)),
            'seen_before_select_count':sum(r['seen_before_select'] for r in selected),
            'seen_before_select_rate':ratio(sum(r['seen_before_select'] for r in selected)),
            'seen_anywhere_count':sum(r['seen_anywhere'] for r in selected),
            'syntax_failures':sum(r['syntax_failures'] for r in selected),
            'model_outputs':sum(r['model_outputs'] for r in selected)}
    return result


def inspect_records(records,strategy,entries,e2_index,*,require_complete):
    pending=defaultdict(list); finished=set(); episodes=[]; by_identity={}; samples=[]; cov=[]
    for record in records:
        need(record.get('strategy')==strategy,'Unexpected journal strategy')
        eid=record['episode_id']; need(eid not in finished,'Duplicate terminal or decision after terminal episode')
        if record['record_kind']=='decision': pending[eid].append(record); continue
        need(record['record_kind']=='episode' and eid in pending,'Episode lacks actual decisions')
        ds=pending.pop(eid); finished.add(eid); key=(record['dataset_key'],record['world_id'])
        need(key not in by_identity,'Duplicate pair/world episode')
        actual_samples,actual_cov=inspect_episode(record,ds,strategy,entries,e2_index)
        episodes.append(record); by_identity[key]=ds; samples.extend(actual_samples)
        if actual_cov is not None: cov.append(actual_cov)
    expected={(key,world) for key in entries for world in ('w0','w1')}
    complete=not pending and set(by_identity)==expected and len(episodes)==len(expected)
    if require_complete: need(complete,'Source claimed complete but strategy population is incomplete')
    return {'episodes':episodes,'decisions_by_identity':by_identity,'samples':samples,'cov':cov,
        'coverage':{'complete':complete,'observed_completed_episodes':len(episodes),
            'expected_episodes':len(expected),'unfinished_episode_ids':sorted(pending),
            'missing_identities':[{'dataset_key':key,'world_id':world} for key,world in sorted(expected-set(by_identity))]},
        'summary':summarize(episodes)}


def cache_review(samples,*,require_complete):
    groups=defaultdict(list)
    for sample in samples:
        descriptor=sample['request_descriptor']
        canonical=json.dumps(descriptor,sort_keys=True,separators=(',',':'),allow_nan=False)
        key=hashlib.sha256(canonical.encode()).hexdigest(); cache=sample['exact_input_cache']
        need(key==cache['key'] and cache['origin_sample_id']==sample['sample_id'],'Cache identity differs')
        need(type(cache['reused_generation']) is bool,'Cache reuse marker is not boolean')
        groups[key].append(sample)
    for key,rows in groups.items():
        need(all(row['request_descriptor']==rows[0]['request_descriptor'] and
                 output_identity(row)==output_identity(rows[0]) and row['sample_id']==rows[0]['sample_id'] for row in rows),
             'Same complete request has inconsistent outputs or origin')
        origins=sum(row['exact_input_cache']['reused_generation'] is False for row in rows)
        need(origins<=1,'Same request generated more than once in strategy journals')
        if require_complete: need(origins==1,'Complete source has no generation origin')
    return {'observed_logical_requests':len(samples),'observed_unique_request_keys':len(groups),
        'same_request_different_outputs':0,'complete_origin_coverage_required':require_complete,
        'scope':'Original three strategy journals only; uncached E5 probes are separate; within one model run.'}


def direct_review(inspected,entries,*,require_complete):
    expected={(key,world) for key,row in entries.items() if row['kind'].startswith('direct_') for world in ('w0','w1')}
    rows=[]; missing=[]
    for key in sorted(expected):
        if not all(key in inspected[s]['decisions_by_identity'] for s in STRATEGIES):
            missing.append({'dataset_key':key[0],'world_id':key[1]}); continue
        samples={}
        for strategy in STRATEGIES:
            ds=inspected[strategy]['decisions_by_identity'][key]
            need(len(ds)==1 and not ds[0]['sample']['forced_observation'],'Direct control unexpectedly observed')
            samples[strategy]=ds[0]['sample']
        values=list(samples.values())
        same_input=all(s['request_descriptor']==values[0]['request_descriptor'] for s in values)
        same_output=all(output_identity(s)==output_identity(values[0]) for s in values)
        need(same_input and same_output,'Direct negative control is not exactly input/output consistent')
        rows.append({'dataset_key':key[0],'world_id':key[1],'same_complete_request':same_input,
            'same_output':same_output,'request_keys':{s:v['exact_input_cache']['key'] for s,v in samples.items()}})
    if require_complete: need(not missing,'Complete source is missing Direct negative controls')
    return {'expected_controls':len(expected),'observed_complete_controls':len(rows),'missing_identities':missing,
        'full_coverage':not missing,'same_request_different_output':sum(not r['same_output'] for r in rows),'rows':rows}


def uncached_review(report,inspected):
    direct={(r['dataset_key'],r['world_id']) for r in inspected['obs_never']['episodes'] if r['pair_kind'].startswith('direct_')}
    baseline={key:ds[0]['sample'] for key,ds in inspected['obs_never']['decisions_by_identity'].items()
        if key in direct and len(ds)==1 and not ds[0]['sample']['forced_observation']}
    previous=None;results=[];names=set()
    fields=('action','response','response_token_ids','finish_reason','stop_reason')
    for probe in report['uncached_probes']:
        name=probe['name'];need(name not in names,'Duplicate uncached probe');names.add(name)
        if probe['status']!='completed':
            results.append({'name':name,'status':probe['status'],'observed_completed_comparisons':0});continue
        comparisons=probe['comparisons'];current={(r['dataset_key'],r['world_id']):r['sample'] for r in comparisons}
        need(len(current)==len(comparisons)==probe['episodes'] and set(current)==direct,'Duplicate/missing uncached observations')
        rows=[]
        for row in comparisons:
            key=(row['dataset_key'],row['world_id']);sample=current[key]
            need(key in baseline and sample['request_descriptor']==baseline[key]['request_descriptor'],
                 'Uncached request differs from its original strategy request')
            descriptor_key=hashlib.sha256(json.dumps(sample['request_descriptor'],sort_keys=True,
                separators=(',',':'),allow_nan=False).encode()).hexdigest()
            need(descriptor_key==row['request_key'],'Uncached request SHA differs')
            comparisons_out={}
            for tag,other in [('mixed_origin',baseline[key]),('first_same_order',None if previous is None else previous[key])]:
                if other is None: comparisons_out[tag]=None;continue
                differences={field:sample[field]!=other[field] for field in fields}
                differences['complete_output_identity']=output_identity(sample)!=output_identity(other)
                comparisons_out[tag]=differences
            need(row['different_from_mixed_batch_origin']==comparisons_out['mixed_origin']['complete_output_identity'] and
                 row['different_from_prior_same_order']==(None if previous is None else comparisons_out['first_same_order']['complete_output_identity']),
                 'Uncached raw comparison flag differs')
            rows.append({'dataset_key':key[0],'world_id':key[1],**comparisons_out})
        counts={tag:None if tag=='first_same_order' and previous is None else
            {field:sum(row[tag][field] for row in rows) for field in (*fields,'complete_output_identity')}
            for tag in ('mixed_origin','first_same_order')}
        need(probe['mixed_batch_origin_differences']==counts['mixed_origin']['complete_output_identity'] and
             probe['prior_same_order_differences']==(None if previous is None else counts['first_same_order']['complete_output_identity']),
             'Official uncached summary differs')
        results.append({'name':name,'status':'completed','observed_completed_comparisons':len(rows),
            'differences':counts,'rows_with_any_difference':[row for row in rows if row['mixed_origin']['complete_output_identity']
                or (row['first_same_order'] is not None and row['first_same_order']['complete_output_identity'])]})
        if name=='direct_same_order_1':previous=current
    return {'probes':results,'scope':'Actual uncached calls at current runtime settings; action/text/token and finish/stop differences separated. Not a causal diagnosis of historical DIAG01 runtime.'}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--name',required=True);p.add_argument('--source',required=True)
    p.add_argument('--timeout-seconds',type=int,default=180); args=p.parse_args()
    need(re.fullmatch('diag02_[A-Za-z0-9_]+',args.name),'Unique review name required')
    need(30<=args.timeout_seconds<=300,'Bounded CPU review required')
    out=ROOT/'results'/(args.name+'.json'); need(not out.exists(),'Preserve previous review')
    folder=ROOT/'logs/review/diag02'/args.name;folder.mkdir(parents=True,exist_ok=False)
    def expired(*_):raise TimeoutError('DIAG02 CPU journal review time bound reached')
    signal.signal(signal.SIGALRM,expired);signal.alarm(args.timeout_seconds)
    metrics={'review_completed':False,'source_run_completed':False,'errors':[],'gpus':0,'paper_result':False}
    sources=[];start=time.monotonic();code=1
    try:
        requested=json.loads(args.source); parent,pref=audited_result(ROOT/requested['file']);sources.append(pref)
        need(ref_key(pref)==ref_key(requested) and parent['_meta'].get('end'),'Need the exact sealed source')
        cfg,pm=parent['config'],parent['metrics']
        need(cfg['version']=='diag02_fixed83_cov_full_input_once_v1' and cfg['strategies']==list(STRATEGIES)
             and cfg['model_size'] in ('3b','7b','32b'),'Not an original DIAG02 source')
        complete=pm.get('completed') is True and parent['_meta']['exit_code']==0 and pm.get('errors')==[] and pm.get('cleanup_errors')==[]
        e2,eref=audited_result(ROOT/E2['file']);sources.append(eref)
        need(ref_key(eref)==ref_key(E2) and e2['_meta']['exit_code']==0 and e2['metrics']['complete'] is True,'Original E2 not complete')
        e2_report=json.loads(read(e2['metrics']['report'])); e2_index=e2_world_index(e2_report)
        need(ref_key(cfg['manifest_reference'])==ref_key(e2_report['population_manifest']),'GPU population differs from E2 fixed83')
        manifest=json.loads(read(cfg['manifest_reference']));entries={r['dataset_key']:r for r in manifest['entries']}
        need(len(entries)==len(manifest['entries'])==83 and manifest['partition']=='train','Population is not fixed83 training-scene set')
        expected_names={s+'.jsonl' for s in STRATEGIES}|{'progress.jsonl'}
        refs={Path(r['file']).name:r for r in pm.get('journals',[])}
        need(len(refs)==len(pm.get('journals',[])) and set(refs)<=expected_names,'Duplicate/unexpected journals')
        if complete:need(set(refs)==expected_names,'Complete source journal missing')
        if 'progress.jsonl' in refs: read(refs['progress.jsonl'])
        inspected={}
        for strategy in STRATEGIES:
            ref=refs.get(strategy+'.jsonl');records=[]
            if ref is not None:
                need((ROOT/ref['file']).resolve().parent==Path(cfg['episode_directory']).resolve(),'Journal escaped producer directory')
                records=[json.loads(line) for line in read(ref).splitlines() if line.strip()]
            inspected[strategy]=inspect_records(records,strategy,entries,e2_index,require_complete=complete)
        summaries={s:r['summary'] for s,r in inspected.items()}
        official=pm.get('strategies'); matched=None
        if official is not None:
            need(summaries==official,'Official summary differs from original journal recount');matched=True
        if complete:need(matched is True,'Complete source lacks official strategy summary')
        cache=cache_review([sample for s in STRATEGIES for sample in inspected[s]['samples']],require_complete=complete)
        controls=direct_review(inspected,entries,require_complete=complete)
        need(controls['expected_controls']==86,'Fixed Direct domain differs')
        probes=None
        if complete:
            original_e5=json.loads(read(pm['determinism_report']))
            need(original_e5['direct_control_count']==controls['observed_complete_controls'] and
                 original_e5['same_request_different_output']==controls['same_request_different_output'],
                 'Official E5 Direct summary differs')
            official_cache=pm['unique_input_generation']
            need(official_cache['logical_requests']==cache['observed_logical_requests'] and
                 official_cache['actual_generations']==cache['observed_unique_request_keys'] and
                 official_cache['reused_generations']==cache['observed_logical_requests']-cache['observed_unique_request_keys'],
                 'Official cache counts differ from complete strategy journals')
            probes=uncached_review(original_e5,inspected)
        cov=inspected['obs_trigger_cov']['cov'];cov_summary={}
        for kind in ('look_flip','look_stay','look_all'):
            rows=[r for r in cov if kind=='look_all' or r['kind']==kind];n=len(rows)
            triggered=sum(r['actual_cov_selection'] for r in rows)
            cov_summary[kind]={'observed_ended_look_episodes':n,'actual_cov_selections':triggered,
                'actual_cov_selection_fraction':triggered/n if n else None,
                'expected_look_worlds':2*sum(r['kind'].startswith('look_') if kind=='look_all' else r['kind']==kind for r in entries.values()),
                'source_run_complete':complete,'not_an_E2_existence_rate':True}
        report={'version':VERSION,'source':pref,'source_run_completed':complete,'original_source_exit_code':parent['_meta']['exit_code'],
            'model_size':cfg['model_size'],'E2_source':eref,'E2_report':e2['metrics']['report'],
            'strategy_summaries_from_observed_complete_episodes':summaries,'official_summary_exactly_reproduced':matched,
            'coverage':{s:r['coverage'] for s,r in inspected.items()},'cov_actual_summary':cov_summary,
            'cov_actual_vs_E2_worlds':cov,'direct_negative_control':controls,'full_input_cache':cache,'uncached_probe_decomposition':probes,
            'scope':'Sealed source journal review. Look mask count/SHA and candidate marks cross-bound to independently read E2 raw bytes. No fresh GPU/native replay; incomplete runs are not promoted to complete.'}
        metrics.update(review_completed=True,source_run_completed=complete,report=save_json(folder/'report.json',report),
            cov_actual_summary=cov_summary,direct_control_summary={k:v for k,v in controls.items() if k not in ('rows','missing_identities')},
            all_strategies_have_full_coverage=all(r['coverage']['complete'] for r in inspected.values()),
            official_summary_exactly_reproduced=matched)
        code=0
    except BaseException as exc:
        traceback.print_exc();metrics['errors'].append(type(exc).__name__+': '+str(exc))
    finally:signal.alarm(0)
    metrics.update(producer_exit_code=code,elapsed_seconds=time.monotonic()-start)
    save_json(out,{'metrics':metrics,'config':{**vars(args),'version':VERSION,'gpus':0},'sources':sources})
    print('FINAL_METRICS_JSON: '+json.dumps(metrics,sort_keys=True,allow_nan=False),flush=True)
    return code


if __name__=='__main__':raise SystemExit(main())

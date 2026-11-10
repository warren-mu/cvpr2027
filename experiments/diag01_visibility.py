"""DIAG01 D1: read-only independent mask/action visibility diagnosis.

No training or metric changes, native replay, renderer, or new table authority.
The logged x_seen_step is used only after raw-mask trajectory recomputation.
"""
import argparse
import ast
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import re
import signal
import time
import traceback

from src.envs.thor_common import ROOT
from src.envs.paired_env import RGBFrame, parse_action
from src.envs.pair_table import RenderedView, mark_view, label_assignment
from src.tools import smoke_e2e_pilot_certificates as certificates
from src.tools import train_holdout_freeze as holdout
from src.tools.smoke_e2e_pilot import completed_worker
from src.tools.t02_export import save_json

VERSION = 'diag01_independent_mask_visibility_v1'
QUALIFICATION = {'file':'results/smoke_e2e_pilot_pool_83_certificates_01.json',
 'sha256':'0b704d1e8c1579b1f76a968084af6e919bb65ec009abcf7d91b96e02e63d91f0'}
PARENTS = {
 'T0': {'file':'results/smoke_e2e_pilot_T0_s0_01_retry4.json','sha256':'17188af3dff53d6bb07be4b682fdcc4f3f9a3b8f369d12d93d32c3cb195698f2'},
 'T4': {'file':'results/smoke_e2e_pilot_T4_s0_01_retry4.json','sha256':'994e3a7ba47ec29c06891a2ff1b1449439b0b337fc7ea2eef4118e23d2dc8717'}}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def reference(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return {'file':str(path), 'sha256':hashlib.sha256(raw).hexdigest(), 'bytes':len(raw)}


class Reads:
    """Per-run reference/size authentication, with count-only mask memoization."""
    def __init__(self, root=ROOT):
        self.root = Path(root); self.refs = {}; self.counts = {}

    def read(self, ref):
        path = (self.root / ref['file']).resolve()
        actual = {'file':str(path), **{k:ref[k] for k in ('sha256','bytes') if k in ref}}
        old = self.refs.get(str(path))
        need(old is None or old == actual, 'Conflicting immutable reference: '+str(path))
        raw = path.read_bytes()
        need(hashlib.sha256(raw).hexdigest() == ref['sha256'] and ('bytes' not in ref or len(raw) == ref['bytes']),
             'Original bytes changed: '+str(path))
        self.refs[str(path)] = actual
        return raw

    def json(self, ref):
        return json.loads(self.read(ref))

    def mask(self, ref):
        key = (str((self.root/ref['file']).resolve()), ref['sha256'], ref['bytes'])
        if key not in self.counts:
            raw = self.read(ref)
            need(len(raw) == 255*255 and set(raw).issubset({0,1}), 'Not an original binary 255x255 mask')
            self.counts[key] = raw.count(1)
        return self.counts[key]

    def recheck(self):
        for ref in list(self.refs.values()):
            self.read(ref)


class TableIndex:
    def __init__(self, entries, reads):
        self.entries = {r['dataset_key']:r for r in entries}; self.reads = reads
        self.headers = {}; self.state_index = {}; self.s0 = {}

    def header(self, key):
        if key not in self.headers:
            entry = self.entries[key]; h = self.reads.json(entry['table_manifest'])
            from src.envs.pair_table_identity import identity_fields
            identity = identity_fields(h['pair_id'],h['source']['t01']['pair'])
            need(identity['dataset_key'] == key and h['pair_id'] == entry['original_pair_id']
                 and h['scene'] == entry['scene'] and h['kind'] == entry['kind'], 'Header/pool identity changed')
            self.headers[key] = h
        return self.headers[key]

    def payload(self, key, ref):
        directory = (self.reads.root/self.entries[key]['table_manifest']['file']).resolve().parent
        path = (directory/ref['file']).resolve()
        need(path.parent == directory and path.name != 'table.json' and 'encoding' not in ref,
             'Require original fixed training payload inside its table directory')
        return {**ref,'file':str(path)}

    def state(self, key, state_id, original):
        lookup=(key,state_id,original)
        if lookup not in self.state_index:
            h=self.header(key); s=h['states'][state_id]; c=s['captures'][original]
            mask=self.payload(key,c['masks'][h['anchor_name']]); pixels=self.reads.mask(mask)
            public=0 if s['shared'] else original
            self.state_index[lookup]={'dataset_key':key,'original_world_id':f'w{original}','state_id':state_id,
                'own_anchor_pixels':pixels,'anchor_mask_reference':mask,'shared':s['shared'],
                'shadow_divergent':s['shadow_divergent'],'frame_max_abs_diff':s['frame_max_abs_diff'],
                'public_capture_original_world_id':f'w{public}',
                'raw_rgb_reference':self.payload(key,s['captures'][public]['rgb']),
                'stored_anchor_pixels':s['anchor_pixels'][original],
                'stored_anchor_count_matches_bytes':pixels == s['anchor_pixels'][original]}
        return self.state_index[lookup]

    def marked(self,key,state_id,original):
        h=self.header(key);s=h['states'][state_id];public=0 if s['shared'] else original;c=s['captures'][public]
        rgb=self.reads.read(self.payload(key,c['rgb']))
        masks={n:self.reads.read(self.payload(key,c['masks'][n])) for n in (h['anchor_name'],*h['candidate_names'])}
        view=RenderedView(RGBFrame(rgb,255,255),masks,c['distances'],c['agent'],c['process_id'],c['render_id'],c['source'],c['synthetic'])
        return mark_view(view,label_assignment(h['pair_id'],h['candidate_names']),h['candidate_names'])[0]

    def first_actions(self,key,original):
        h=self.header(key); rows=[]
        for action,state in sorted(h['states'][h['initial_state']]['transitions'].items()):
            measurement=self.state(key,state,original)
            rows.append({'action':action,'state_id':state,'own_anchor_pixels':measurement['own_anchor_pixels'],
                         'sees_250':measurement['own_anchor_pixels']>=250,
                         'destination_viewpoint':h['states'][state]['view']['viewpoint_id'],
                         'destination_position':h['positions'][h['states'][state]['view']['viewpoint_id']]})
        return rows


def bind_info(info,key,logical,order,state,index):
    original=order[int(logical[-1])]; h=index.header(key); s=h['states'][state]; public=0 if s['shared'] else original
    capture=info['capture']
    need(info['state_id']==state and info['dataset_key']==key and info['world_id']==logical
         and info['world_order']==order and info['original_world_id']==f'w{original}'
         and capture['state_id']==state and capture['pair_id']==key and capture['world_id']==logical
         and capture['raw_rgb_sha256']==s['captures'][public]['rgb']['sha256']
         and info['table_temporal_proof']['table_manifest']==index.entries[key]['table_manifest'],
         'Journal state/world/public capture differs from original table')


def recompute_episode(episode,traces,index,*,arm,phase,update,journal):
    key=episode['dataset_key']; h=index.header(key); order=index.entries[key]['world_order']; logical=episode['world_id']
    need(logical in ('w0','w1') and order in ([0,1],[1,0]) and episode['world_order']==order
         and episode['original_world_id']==f'w{order[int(logical[-1])]}' and h['kind']==episode['kind'],
         'Episode world mapping differs from fixed83')
    original=order[int(logical[-1])]
    need(traces and [t['decision_index'] for t in traces]==list(range(len(traces))) and len(traces)<=5,
         'Incomplete/duplicate episode decision trace')
    current=h['initial_state']; states=[{**index.state(key,current,original),'observation_index':0}]
    actions=[]; selection=None; first_observe=None; marked_hashes={}; input_hashes=[]
    for i,t in enumerate(traces):
        need(t['episode_id']==episode['episode_id'] and t['dataset_key']==key and t['world_id']==logical,
             'Decision belongs to another episode/world')
        bind_info(t['before'],key,logical,order,current,index)
        need(t['before']['steps']==i and t['after']['steps']==i+1, 'Environment step/event indices differ')
        marked_hashes[current]=t['before']['capture']['marked_rgb_sha256']
        if i==0:
            input_hashes=copy.deepcopy(t['input']['images'])
        try: parsed=parse_action(t['response'].strip())
        except ValueError: parsed=None
        action=parsed.text() if parsed is not None else None
        need((parsed is not None)==t['parse_valid'], 'Original parser disagrees with saved parse flag')
        before=current
        if parsed is not None and parsed.kind=='select':
            selection=i
        elif parsed is not None and i<4 and action in h['states'][current]['transitions']:
            if current==h['initial_state'] and first_observe is None:
                first_observe={'decision_index':i,'action':action,
                    'destination_viewpoint':h['states'][h['states'][current]['transitions'][action]]['view']['viewpoint_id']}
            current=h['states'][current]['transitions'][action]
        bind_info(t['after'],key,logical,order,current,index)
        need(t['after']['done'] is (selection is not None or i==4), 'Terminal state differs from action/budget semantics')
        need(selection is None or i==len(traces)-1, 'Decisions continue after parsed terminal select')
        state={**index.state(key,current,original),'observation_index':i+1}
        states.append(state)
        marked_hashes[current]=t['after']['capture']['marked_rgb_sha256']
        actions.append({'decision_index':i,'action_number':i+1,'raw_response':t['response'],'parsed_action':action,
            'action_kind':parsed.kind if parsed else 'invalid','before_state_id':before,'after_state_id':current,
            'transition_performed':current!=before,'saved_action_valid':t['action_valid'],
            'saved_action_success':t['after'].get('action_success'),
            'before_marked_rgb_sha256':t['before']['capture']['marked_rgb_sha256'],
            'after_marked_rgb_sha256':t['after']['capture']['marked_rgb_sha256'],
            'actual_input_images':copy.deepcopy(t['input']['images'])})
    need(traces[-1]['after']['done'] and len(traces)==episode['logical_decisions'], 'Episode was not completely recorded')
    first250=next((s['observation_index'] for s in states if s['own_anchor_pixels']>=250),None)
    firstpos=next((s['observation_index'] for s in states if s['own_anchor_pixels']>0),None)
    tau=next((s['observation_index'] for s in states if not s['shared']),None)
    # State observation k occurs before action k+1; parsed select index is zero-based.
    seen=selection is not None and any(s['own_anchor_pixels']>=250 for s in states[:selection+1])
    wrong_logged_seen=bool(seen)!=episode['seen_before_select']
    literal_logged_formula=selection is not None and first250 is not None and first250<selection
    correct=index.first_actions(key,original)
    if first_observe is not None:
        first_observe['sees_250']=first_observe['action'] in [r['action'] for r in correct if r['sees_250']]
    return {'arm':arm,'phase':phase,'update':update,'episode_id':episode['episode_id'],'dataset_key':key,
        'original_pair_id':h['pair_id'],'scene':h['scene'],'kind':h['kind'],'world_id':logical,
        'original_world_id':f'w{original}','world_order':order,'journal':journal,'states':states,'actions':actions,
        'select_completed_env_steps':selection+1 if selection is not None else None,
        'select_decision_index':selection,'select_action_number':selection+1 if selection is not None else None,
        'independent_first_250_state_index':first250,'independent_first_positive_state_index':firstpos,
        'independent_first_difference_step':tau,'seen_before_select':seen,'ever_250':first250 is not None,
        'ever_positive':firstpos is not None,'logged_seen_before_select':episode['seen_before_select'],
        'seen_count_mismatch':wrong_logged_seen,'logged_x_seen_step':episode['x_seen_step'],
        'x_seen_step_mismatch':first250!=episode['x_seen_step'],
        'recomputed_literal_logged_formula':literal_logged_formula,
        'literal_formula_matches_logged':literal_logged_formula==episode['seen_before_select'],
        's0_first_observation':first_observe,'contains_executed_observation':any(a['transition_performed'] for a in actions),
        'saved_success':episode['success'],'saved_task_reward_sum':episode['task_reward_sum'],
        'saved_selected_name':episode['selected_name'],'target_name':h['target_names'][original],
        's0_actual_input_images':input_hashes,'marked_rgb_sha256_by_state':marked_hashes}


def summarize(rows):
    n=len(rows)
    fields=('seen_before_select','ever_250','ever_positive','seen_count_mismatch','x_seen_step_mismatch')
    return {'episodes':n,**{field:sum(r[field] for r in rows) for field in fields},
        'logged_seen_before_select':sum(r['logged_seen_before_select'] is True for r in rows),
        'literal_formula_matches_logged':sum(r['literal_formula_matches_logged'] for r in rows),
        'denominator':'all Look episodes in the stated phase/window, including episodes without parsed select'}


def load_journal(row,reads,expected_kind):
    records=[json.loads(line) for line in reads.read(row['episode_log']).splitlines() if line.strip()]
    ends=[r for r in records if r.get('record_kind')==expected_kind]
    need(len(ends)==1 and ends[0]['update']==row['update'] and ends[0]['by_class']==row['by_class'],
         'Original complete journal marker differs from parent result')
    episodes=[r for r in records if r.get('record_kind')=='pilot_episode']
    need(len({r['episode_id'] for r in episodes})==len(episodes)==row['episodes'], 'Episode inventory incomplete')
    decisions=defaultdict(list)
    for r in records:
        if r.get('kind')=='decision':decisions[r['episode_id']].append(r)
    need(set(decisions)=={r['episode_id'] for r in episodes}, 'Decision/episode inventory differs')
    return episodes,decisions


def png(path,data,size):
    from PIL import Image
    image=Image.frombytes('RGB',size,data);image.save(path,format='PNG')
    with Image.open(path) as actual:need(actual.tobytes()==data and actual.size==size,'PNG pixels changed')
    return {**reference(path),'decoded_pixels_sha256':hashlib.sha256(data).hexdigest(),'size':list(size)}


def original_image_preprocessor(parents, reads):
    """Execute only the original pure image function, avoiding Torch/model imports."""
    upstream=parents['T0']['metrics']['upstream']['verl']
    need(upstream==parents['T4']['metrics']['upstream']['verl']
         and upstream['commit']=='3f55021e08b98315213e10d10b709c7f6bbe84f7', 'Different original preprocessing source')
    path=Path(upstream['path'])/'verl/utils/dataset/rl_dataset.py'
    ref=reference(path);raw=reads.read(ref);tree=ast.parse(raw,filename=str(path))
    matches=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='process_image']
    need(len(matches)==1 and not matches[0].decorator_list, 'Original pure process_image missing')
    node=matches[0]; namespace={}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace['process_image'],{'source':ref,'upstream':upstream,'first_line':node.lineno,'last_line':node.end_lineno,
        'scope':'Original process_image function extracted unchanged; no other upstream module code or model loaded'}


def code_comparisons(reads):
    wanted=[('src/tools/smoke_e2e_pilot_contract.py','seen >= selections',
             'Learning metric compares state count with zero-based decision index using strict <.'),
            ('src/rl/t4_consistency.py',"select['decision_index'] >= select['x_seen_step']",
             'T4 eligibility permits equality; the immediately preceding observation is eligible.'),
            ('src/tools/smoke_e2e_training_contract.py','return tau is None or decision_index < tau',
             'Pre-evidence excludes equality; the next action after a difference is post-evidence.')]
    output=[]
    for file,pattern,meaning in wanted:
        ref=reference(reads.root/file);lines=reads.read(ref).decode().splitlines()
        matched=[{'line':n,'text':text} for n,text in enumerate(lines,1) if pattern in text]
        need(len(matched)==1,'Diagnostic source comparison changed: '+file)
        output.append({'source':ref,'matched':matched[0],'interpretation':meaning})
    return output


def export_examples(rows,index,directory,process_image=None):
    candidates=[r for r in rows if r['contains_executed_observation'] and not r['ever_250']][:3]
    outputs=[]
    for number,row in enumerate(candidates):
        target=directory/f'example_{number+1:02d}';target.mkdir(parents=True,exist_ok=False)
        frames=[]
        for s in row['states']:
            marked=index.marked(row['dataset_key'],s['state_id'],int(row['original_world_id'][-1]))
            need(marked.sha256==row['marked_rgb_sha256_by_state'][s['state_id']], 'Reconstructed actual marked input differs')
            frame={'observation_index':s['observation_index'],'state_id':s['state_id'],
                'own_anchor_pixels':s['own_anchor_pixels'],
                'environment_input_255':png(target/f"state_{s['observation_index']:02d}_marked.png",marked.data,(255,255))}
            # Terminal after-state was not sent to the actor; do not invent a model input for it.
            if process_image is not None and s['observation_index']<len(row['actions']):
                from PIL import Image
                image=process_image(Image.frombytes('RGB',(255,255),marked.data))
                image_proof={'mode':image.mode,'size':list(image.size),'sha256':hashlib.sha256(image.tobytes()).hexdigest()}
                need(row['actions'][s['observation_index']]['actual_input_images']==[image_proof],
                     'Original processed model input differs from saved actual policy input')
                frame['actual_model_input']=png(target/f"state_{s['observation_index']:02d}_model.png",image.tobytes(),image.size)
            frames.append(frame)
        output={'episode_id':row['episode_id'],'arm':row['arm'],'phase':row['phase'],'update':row['update'],
                'dataset_key':row['dataset_key'],'world_id':row['world_id'],'original_world_id':row['original_world_id'],
                'frames':frames,'actions':row['actions'],'scope':'Original marked environment inputs; original pure preprocessing reconstructs actual model inputs with exact saved pixel SHA; terminal after-state has no invented model input'}
        ref=save_json(target/'manifest.json',output);outputs.append(ref)
    return outputs


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True);parser.add_argument('--timeout-seconds',type=int,default=240)
    args=parser.parse_args();need(re.fullmatch(r'diag01_[A-Za-z0-9_]+',args.name),'Use unique DIAG01 name')
    need(30<=args.timeout_seconds<=300,'Bounded CPU metadata/mask diagnosis only')
    output=ROOT/'results'/(args.name+'.json');need(not output.exists(),'Preserve original diagnostic result')
    directory=ROOT/'logs/review/diag01'/args.name;directory.mkdir(parents=True,exist_ok=False)
    def expired(*_):raise TimeoutError('DIAG01 CPU bound reached; retain partial artifacts')
    signal.signal(signal.SIGALRM,expired);signal.alarm(args.timeout_seconds)
    sources=[];metrics={'kind':VERSION,'complete':False,'errors':[],'gpus':0};started=time.monotonic();code=1
    try:
        manifest,proof,source,manifest_ref=certificates.authenticate_certificate_qualification(QUALIFICATION)
        sources.append(source);reads=Reads();reads.read(manifest_ref);reads.read(QUALIFICATION)
        parents={}
        for arm,ref in PARENTS.items():
            parent,actual=certificates._audited(ref,ROOT);completed_worker(parent['metrics'],parent['config'])
            need(parent['_meta']['end'] and parent['_meta']['exit_code']==0 and parent['metrics']['pilot_completed'] is True
                 and parent['metrics']['errors']==[] and parent['metrics']['cleanup_errors']==[]
                 and parent['config']['arm']==arm and parent['config']['updates']==50
                 and parent['config']['manifest_reference']==manifest_ref,'Require original complete fixed83 main pilot')
            frozen,actual_frozen=holdout.load_frozen_holdout(parent['config']['holdout_reference'],root=ROOT)
            holdout.assert_pilot_split(manifest['entries'],manifest['split'],frozen)
            reads.read(ref);reads.read(actual_frozen);parents[arm]=parent;sources.append(actual)
        process_image,preprocessing=original_image_preprocessor(parents,reads)
        comparisons=code_comparisons(reads)
        index=TableIndex(manifest['entries'],reads);episodes=[];correct=[]
        for key,entry in index.entries.items():
            if not entry['kind'].startswith('look_'):continue
            h=index.header(key)
            for world in (0,1):
                actions=index.first_actions(key,world)
                correct.append({'dataset_key':key,'original_pair_id':h['pair_id'],'scene':h['scene'],'kind':h['kind'],
                    'original_world_id':f'w{world}','world_order':entry['world_order'],'actions':actions})
        print('DIAG01_PHASE: original metadata and correct actions complete',flush=True)
        for arm,parent in parents.items():
            metrics_parent=parent['metrics']
            for phase,items,marker in [('train',metrics_parent['updates'],'pilot_update_complete'),
                                       ('holdout',metrics_parent['holdout_evaluations'],'pilot_holdout_complete')]:
                for item in items:
                    logged,decisions=load_journal(item,reads,marker)
                    for ep in logged:
                        if ep['kind'].startswith('look_'):
                            episodes.append(recompute_episode(ep,decisions[ep['episode_id']],index,
                                arm=arm,phase=phase,update=item['update'],journal=item['episode_log']))
                print(f'DIAG01_PHASE: {arm} {phase} journals complete',flush=True)
        s0=[]
        for key,entry in index.entries.items():
            if not entry['kind'].startswith('look_'):continue
            h=index.header(key)
            for world in (0,1):
                state=index.state(key,h['initial_state'],world);marked=index.marked(key,h['initial_state'],world)
                matches=[ep for ep in episodes if ep['dataset_key']==key and ep['original_world_id']==f'w{world}']
                need(all(ep['actions'][0]['before_marked_rgb_sha256']==marked.sha256 for ep in matches),
                     'Actual reconstructed s0 marked pixels differ from sealed journal')
                actual_hashes={json.dumps(image,sort_keys=True):image for ep in matches for image in ep['s0_actual_input_images']}
                s0.append({'dataset_key':key,'original_pair_id':h['pair_id'],'original_world_id':f'w{world}',
                    'raw_rgb_sha256':state['raw_rgb_reference']['sha256'],'marked_rgb_sha256':marked.sha256,
                    'actual_input_images':list(actual_hashes.values()),'train_or_eval':'train' if key in proof['train_keys'] else 'eval',
                    'scope':'Actual raw and 255 marked pixels read/reconstructed; processed model image hashes from sealed journals'})
        grouped=[];directions=[]
        for arm in parents:
            for phase,windows in [('train',[(x,x+9) for x in range(1,51,10)]),('holdout',[(x,x) for x in (0,20,40,50)])]:
                for low,high in windows:
                    rows=[r for r in episodes if r['arm']==arm and r['phase']==phase and low<=r['update']<=high]
                    grouped.append({'arm':arm,'phase':phase,'first_update':low,'last_update':high,**summarize(rows)})
                    counts=Counter((r['kind'],r['s0_first_observation']['action'],r['s0_first_observation']['sees_250'])
                        for r in rows if r['s0_first_observation'] is not None)
                    directions.append({'arm':arm,'phase':phase,'first_update':low,'last_update':high,'all_look_episodes':len(rows),
                        'no_executed_s0_observation':sum(r['s0_first_observation'] is None for r in rows),
                        'counts':[{'kind':k,'action':a,'correct_direction':c,'episodes':n} for (k,a,c),n in sorted(counts.items())]})
        examples=export_examples(episodes,index,directory,process_image);reads.recheck()
        report={'version':VERSION,'code_comparisons':comparisons,'example_preprocessing':preprocessing,'population_manifest':manifest_ref,'original_sources':sources,
            'independent_seen_definition':'Some actual own-world input/state before parsed select action has mask>=250; observation state index k precedes action number k+1. No select => false.',
            'logged_comparison_definition':'Original metric uses first observed state index < zero-based select decision_index; retained only as a comparison, not used to derive masks or seen.',
            'episodes':episodes,'summaries':grouped,'first_observation_directions':directions,'correct_s0_actions':correct,
            'look_s0':s0,'visited_state_index':list(index.state_index.values()),'examples':examples,
            'payload_scope':'All Look s0 and reached/one-step own-X masks; s0 RGB/candidate masks and three example frames only. No new native/cache acceptance.',
            'source_files_read':list(reads.refs.values()),'elapsed_seconds':time.monotonic()-started}
        report_ref=save_json(directory/'report.json',report)
        metrics.update(complete=True,report=report_ref,summaries=grouped,examples=examples,
            source_reauthentication_performed=False,original_results_modified=False)
        code=0
    except BaseException as e:
        traceback.print_exc();metrics['errors'].append(f'{type(e).__name__}: {e}')
    finally:signal.alarm(0)
    metrics.update(producer_exit_code=code,elapsed_seconds=time.monotonic()-started)
    save_json(output,{'metrics':metrics,'config':{**vars(args),'version':VERSION,'gpus':0},'sources':sources})
    print('FINAL_METRICS_JSON: '+json.dumps(metrics,sort_keys=True,allow_nan=False),flush=True)
    return code


if __name__=='__main__':raise SystemExit(main())

"""Read-only, explicitly unpaired alternatives to the unavailable Kettle figure."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import traceback

from PIL import Image, ImageDraw, ImageFont

from experiments import fig01_export as fig
from src.envs.thor_common import ROOT
from src.envs.pair_table import label_assignment
from src.envs.pair_v2_configuration import from_pair
from src.tools import smoke_e2e_pilot_certificates as certificates
from src.tools.t01_partition import audited_result
from src.tools.t02_export import save_json


QUALIFICATION = {'file': 'results/smoke_e2e_pilot_pool_83_certificates_01.json',
                 'sha256': '0b704d1e8c1579b1f76a968084af6e919bb65ec009abcf7d91b96e02e63d91f0'}


def rank_actions(header, order):
    initial = header['states'][header['initial_state']]
    fig.need(initial['shared'] and initial['anchor_pixels'] == [0, 0], 'Look start must hide the anchor')
    transitions = [(a, header['states'][state]['anchor_pixels'])
                   for a, state in initial['transitions'].items()]
    first = [(a,p) for a,p in transitions if p[order[0]] >= 250]
    second = [(a,p) for a,p in transitions if p[order[1]] >= 250]
    fig.need(first and second, 'Original one-step visibility unavailable')
    look, px = min(first, key=lambda ap: (ap[1][order[1]] != 0, -ap[1][order[0]], ap[0]))
    w1_action, w1_px = min(second, key=lambda ap: (-ap[1][order[1]], ap[0]))
    return {'look': look, 'w1_from_s0': w1_action, 'first_w0_pixels': px[order[0]],
            'same_action_w1_pixels': px[order[1]], 'w1_one_step_pixels': w1_px[order[1]],
            'rank_min_seen': min(px[order[0]], w1_px[order[1]])}


def capture(table, key, order, world, actions, out):
    env = fig.environment(table, key, order)
    try:
        obs = env.reset(key, world)
        for action in actions:
            obs, _, done, info = env.step(action)
            fig.need(info['action_success'] and not done, 'Required one-step action failed')
        info = env.info
        state = table.states[info['state_id']]
        actual = env.backend.world
        public = 0 if state.shared else actual
        public_capture = state.captures[public]
        mask = public_capture.masks[table.anchor_name]
        inds = [i for i,b in enumerate(mask) if b]
        box = [0,0,0,0]
        if inds:
            xs, ys = [i%255 for i in inds], [i//255 for i in inds]
            box = [min(xs),min(ys),max(xs)-min(xs)+1,max(ys)-min(ys)+1]
        fig.need(env.frame.raw_rgb == public_capture.rgb, 'Policy public capture differs')
        raw = obs.image.data
        image = Image.frombytes('RGB',(255,255),raw)
        row = {'world':world, 'original_world':actual, 'public_capture_world':public,
               'state_id':info['state_id'], 'actions':list(actions),
               'tau':info['tau'], 'tau_seen':info['x_seen_step'], 'mask_pixels':len(inds),
               'anchor_bbox_xywh':box, 'mask_sha256':hashlib.sha256(mask).hexdigest(),
               'marks':[vars(m) for m in obs.marks], 'decoded_rgb_sha256':obs.image.sha256,
               'source_rgb_sha256':public_capture.rgb.sha256, 'render_id':public_capture.render_id}
        if out is not None:
            row['png'] = fig.png(out,raw,'RGB')
        return image,row
    finally:
        env.close()


def flip_scan(header, order):
    """Preset mirrored sweeps. Graph and masks certify examples after defining each sweep."""
    states = header['states']
    s0 = header['initial_state']
    start = states[s0]
    if not start['shared'] or start['anchor_pixels'] != [0, 0]:
        return None
    labels = label_assignment(header['pair_id'], header['candidate_names'])
    if [labels[header['target_names'][i]] for i in order] != [1, 2]:
        return None
    plans = []
    for look, back in [('turn_left','turn_right'), ('turn_right','turn_left')]:
        first = states[start['transitions'][look]]
        if first['anchor_pixels'][order[0]] < 250 or first['anchor_pixels'][order[1]] != 0:
            continue
        returned = first['transitions'][back]
        fig.need(returned == s0, 'Reversing the original camera turn did not restore s0')
        seen_state = states[states[returned]['transitions'][back]]
        if seen_state['anchor_pixels'][order[1]] < 250:
            continue
        one_step_visible = [a for a,dest in first['transitions'].items()
                            if states[dest]['anchor_pixels'][order[1]] >= 250]
        if one_step_visible:
            continue
        plans.append({'look':look,'more':[back,back],'min_steps_to_w1_seen':2,
                      'one_step_visible_w1_actions_from_look':one_step_visible,
                      'w0_px':first['anchor_pixels'][order[0]],'w1_px':seen_state['anchor_pixels'][order[1]]})
    return max(plans,key=lambda row:min(row['w0_px'],row['w1_px'])) if plans else None


def metadata(table, record, key, order):
    pair=record['package']['row']; cfg=from_pair(pair)
    anchor=pair['identities']['anchor_type']
    return {'id':table.pair_id,'dataset_key':key,'scene':table.scene,'kind':table.kind,
            'anchor_type':anchor,'anchor_name':table.anchor_name,'configuration_id':cfg['configuration_id'],
            'configuration_fields':cfg['key'],'world_order_original_indices':order,
            'candidate_names':list(table.candidate_names),'fixed_labels':table.labels,
            'target_names_agent_worlds':[table.target_names[i] for i in order],
            'target_labels_agent_worlds':[table.labels[table.target_names[i]] for i in order],
            'table':table.manifest_reference,'cache_certificate':record['evidence']['source']}


def board(path, option, flip, stay, frames):
    canvas=Image.new('RGB',(805,695),'white'); draw=ImageDraw.Draw(canvas)
    try:
        title=ImageFont.truetype('DejaVuSans.ttf',17); font=ImageFont.truetype('DejaVuSans.ttf',14)
    except OSError:
        title=font=ImageFont.load_default()
    draw.text((10,5),f"Option {option}    FLIP   {flip['scene']} / {flip['anchor_type']}    +    STAY   {stay['scene']} / {stay['anchor_type']}",font=title,fill='black')
    draw.text((10,31),'Original 255 x 255 agent inputs. FLIP and STAY are independent training pairs.',font=font,fill='#444444')
    locations=[]
    for i,(name,(im,record)) in enumerate(frames.items()):
        x=10+(i%3)*265;y=60+(i//3)*315
        canvas.paste(im,(x,y));locations.append((x,y,im))
        draw.text((x,y+259),name,font=title,fill='#12304a' if i<4 else '#5a2913')
        acts=record['actions']; action_text='s0' if not acts else ' + '.join(a.replace('turn_','') for a in acts)
        draw.text((x,y+282),f"{action_text}  |  anchor {record['mask_pixels']} px",font=font,fill='#333333')
    canvas.save(path)
    with Image.open(path) as reopened:
        for x,y,im in locations:
            fig.need(reopened.crop((x,y,x+255,y+255)).tobytes()==im.tobytes(),'Preview changed original frame bytes')


def write_option(directory, code, fkey, skey, selected, records, entries, routes, stay_actions):
    directory.mkdir(); flip,stay=selected[fkey],selected[skey]; fo=entries[fkey]['world_order'];so=entries[skey]['world_order']
    fm=metadata(flip,records[fkey],fkey,fo);sm=metadata(stay,records[skey],skey,so);plan=routes[fkey]
    f0=plan['look'];s0=stay_actions[skey]['look'];s1=stay_actions[skey]['w1_from_s0']
    specs=[('s0',flip,fkey,fo,'w0',[]),('w0_look',flip,fkey,fo,'w0',[f0]),
           ('w1_look',flip,fkey,fo,'w1',[f0]),('w1_seen',flip,fkey,fo,'w1',[f0,*plan['more']]),
           ('stay_w0_seen',stay,skey,so,'w0',[s0]),('stay_w1_seen',stay,skey,so,'w1',[s1])]
    frames={name:capture(table,key,order,world,actions,directory/(name+'.png'))
            for name,table,key,order,world,actions in specs}
    other,other_record=capture(flip,fkey,fo,'w1',[],None)
    initial,initial_record=frames['s0'];fig.need(initial.tobytes()==other.tobytes(),'Worlds differ in actual public start')
    fig.need([v['label'] for v in initial_record['marks']]==[1,2],'Both start candidates must actually bear their marks')
    fig.need(initial_record['mask_pixels']==other_record['mask_pixels']==frames['w1_look'][1]['mask_pixels']==0,
             'A specified hidden frame has positive anchor pixels')
    for name,step in [('w0_look',1),('w1_seen',3),('stay_w0_seen',1),('stay_w1_seen',1)]:
        r=frames[name][1];fig.need(r['mask_pixels']>=250 and r['tau_seen']==step,'Expected first-seen observation differs')
    def public_row(name,r):
        return {'file':name+'.png','anchor':r['anchor_bbox_xywh'],'mask_px':r['mask_pixels'],
                'seen':r['mask_pixels']>=250,'state_id':r['state_id'],'actions':r['actions'],
                'tau':r['tau'],'tau_seen':r['tau_seen'],'dimensions':[255,255],
                'rgb_sha256':r['decoded_rgb_sha256'],'png_sha256':r['png']['sha256']}
    fields={name:public_row(name,row) for name,(_,row) in frames.items()}
    import io
    buf=io.BytesIO();other.save(buf,format='PNG');other_png_sha=hashlib.sha256(buf.getvalue()).hexdigest()
    fig.need(other_png_sha==fields['s0']['png_sha256'],'Independent world s0 PNG differs')
    fields['_actions']={'look':f0,'w1_more':' '.join([plan['more'][0],'x2']),'w1_more_n':len(plan['more']),
                        'w1_more_actions':plan['more'],'stay_w0':s0,'stay_w1':s1}
    fields['_meta']={'option':code,'scene':fm['scene'],'pair_ids':{'flip':fm['id'],'stay':sm['id']},
        'flip':fm,'stay':sm,'tau':{k:v['tau'] for k,v in fields.items() if not k.startswith('_')},
        'tau_seen':{'w0':fields['w0_look']['tau_seen'],'w1':fields['w1_seen']['tau_seen'],
                    'stay_w0':fields['stay_w0_seen']['tau_seen'],'stay_w1':fields['stay_w1_seen']['tau_seen']},
        'anchor_mask_px':{k:v['mask_px'] for k,v in fields.items() if not k.startswith('_')},
        's0_rgb_sha256':{'w0':initial_record['decoded_rgb_sha256'],'w1':other_record['decoded_rgb_sha256']},
        's0_png_sha256':{'w0':fields['s0']['png_sha256'],'w1':other_png_sha},
        'empty_mask_bbox_convention':'[0,0,0,0] for an empty mask; otherwise inclusive extrema converted to x,y,width,height',
        'sweep_policy':'Preset left-first and right-first mirrored sweeps. After the first public turn, if the anchor is absent, reverse the previous turn to s0 and continue once in that direction. No unseen state or world identity chooses a subsequent action.',
        'shortestness_postcheck':plan,'overview':'No same-snapshot third-party camera view used.',
        'serialization':'Original table RGB with TableObservationBackend training marks; one lossless PNG serialization, verified identical after decoding.'}
    save_json(directory/'frames.json',fields)
    with (directory/'checks.txt').open('x') as out:
        for world in ('w0','w1'):
            out.write(f"FLIP s0 {world} RGB SHA-256 {fields['_meta']['s0_rgb_sha256'][world]}\n")
            out.write(f"FLIP s0 {world} PNG SHA-256 {fields['_meta']['s0_png_sha256'][world]}\n")
        out.write('s0 byte equality: true (decoded RGB and serialized PNG)\n')
        for name,(_,r) in frames.items():out.write(f"{name}.png {r['png']['width']}x{r['png']['height']} RGB\n")
    board(directory/'preview.png',code,fm,sm,frames)
    return {'option':code,'flip':fm,'stay':sm,'frames':fields,'preview':fig.file_ref(directory/'preview.png'),
            'checks':fig.file_ref(directory/'checks.txt'),'manifest':fig.file_ref(directory/'frames.json')}


def execute(directory):
    src,src_ref=audited_result('results/diag_fig_kettle_inventory_01.json')
    fig.need(src['_meta']['exit_code']==0 and src['metrics']['complete'],'Need complete inventory')
    manifest,proof,qualification,_=certificates.authenticate_certificate_qualification(QUALIFICATION)
    entries={r['dataset_key']:r for r in manifest['entries']};kitchen={f'FloorPlan{i}' for i in range(1,21)}
    routes={};stay_actions={}
    for key in proof['train_keys']:
        entry=entries[key];kind=entry['kind']
        if kind=='look_flip':
            plan=flip_scan(fig.read_reference(entry['table_manifest']),entry['world_order'])
            if plan:routes[key]=plan
        elif kind=='look_stay' and entry['scene'] in kitchen:
            stay_actions[key]=rank_actions(fig.read_reference(entry['table_manifest']),entry['world_order'])
    def score(k):return (-min(routes[k]['w0_px'],routes[k]['w1_px']),k)
    first=sorted((k for k in routes if entries[k]['scene'] in kitchen),key=score)
    other=sorted((k for k in routes if entries[k]['scene'] not in kitchen),key=score)
    fig.need(first and len(stay_actions)>=3,'No primary kitchen sweep or insufficient distinct kitchen controls')
    choices=(first[:2]+other)[:3]
    while len(choices)<3:choices.append(first[0])
    stays=sorted(stay_actions,key=lambda k:(-stay_actions[k]['rank_min_seen'],k))
    selected_keys=set(choices+stays[:3]);records={};accessed=[]
    for reference in manifest['cache_sources']:
        ev=certificates.certificate_metadata(reference,root=ROOT)
        wanted=[r for r in ev['complete'] if r['dataset_key'] in selected_keys]
        if not wanted:continue
        headers=fig.certified_headers(ev,entries,accessed)
        for entry in wanted:
            key=entry['dataset_key'];records[key]=fig.accepted_pair_metadata(ev,entry,headers[key],accessed)
    tables={key:fig.validate_selected_table(records[key],expected_kind=entries[key]['kind'])[0] for key in sorted(selected_keys)}
    options=[write_option(directory/code,code,f,s,tables,records,entries,routes,stay_actions)
             for code,f,s in zip('ABC',choices,stays)]
    return {'options':options,'inventory':src_ref,'qualification':qualification,'scope':'Three alternatives with independent certified training pairs.'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True);a=parser.parse_args()
    fig.need(re.fullmatch('diag_fig_kettle_previews_[A-Za-z0-9_]+',a.name),'Fresh preview name required')
    dest=ROOT/'logs/review'/a.name;dest.mkdir(exist_ok=False)
    out=ROOT/'results'/(a.name+'.json');fig.need(not out.exists(),'Preserve prior run')
    metrics={'complete':False,'errors':[],'gpus':0};rc=1
    try:
        report=execute(dest);metrics.update(complete=True,report=save_json(dest/'report.json',report),
            alternatives=[{'option':r['option'],'flip':r['flip']['id'],'stay':r['stay']['id']} for r in report['options']]);rc=0
    except Exception as e:
        traceback.print_exc();metrics['errors'].append(f'{type(e).__name__}: {e}')
    save_json(out,{'metrics':metrics,'config':{'name':a.name,'gpus':0}})
    print('FINAL_METRICS_JSON: '+json.dumps(metrics,sort_keys=True),flush=True);return rc


if __name__=='__main__':raise SystemExit(main())

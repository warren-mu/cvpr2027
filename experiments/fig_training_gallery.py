"""CPU-only figure asset replay from certified, non-holdout original training tables."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re
import traceback

from PIL import Image

from experiments import fig01_export as fig
from experiments.fig_kettle_alternatives import QUALIFICATION
from experiments.fig_selected_frames import TRAIN_SCENES, scan_training
from src.envs.pair_table import mark_view
from src.envs.pair_v2_configuration import from_pair
from src.envs.thor_common import ROOT
from src.tools import smoke_e2e_pilot_certificates as certificates
from src.tools.t02_export import save_json


VERSION = 'certified_training_figure_gallery_20261113_v1'
KINDS = ('look_flip', 'look_stay', 'direct_flip', 'direct_stay')
HOLDOUT = {'file':'data/world_pairs/train_holdout_scenes.json',
           'sha256':'6420085a04a5338c4e6fda52fd8f6549465e8a2092eb98c2d2eb9062a27c59fa'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def choose(rows, limit):
    remaining = list(rows)
    used = {k:Counter() for k in ('kind','room','scene','anchor','combination')}
    configurations = set()
    chosen = []
    def rank(row):
        return (used['kind'][row['kind']], used['room'][row['room']], used['scene'][row['scene']],
                used['anchor'][row['anchor']], used['combination'][row['candidate_types']], row['key'])
    while remaining and len(chosen) < limit:
        row = min(remaining,key=rank)
        remaining.remove(row)
        if row['configuration'] in configurations:
            continue
        configurations.add(row['configuration']);chosen.append(row)
        for k in ('kind','room','scene','anchor'):
            used[k][row[k]] += 1
        used['combination'][row['candidate_types']] += 1
    return chosen


def sequence(table, order):
    """Choose an illustrative path from this graph; both worlds receive identical actions."""
    states = table.states
    start = table.initial_state
    candidates = []
    def walk(current, visited, actions):
        if len(actions) == 3:
            first = states[visited[1]].anchor_pixels
            counts = [max(states[s].anchor_pixels[w] for s in visited) for w in order]
            seen = sum(n >= 250 for n in counts)
            opposite = int(first[order[0]] >= 250 and first[order[1]] == 0)
            coords = [states[s].view.viewpoint_id for s in visited]
            # This scores real observational views for the gallery, not policy performance.
            score = (seen,opposite if table.kind.startswith('look') else 0,
                     len(set(visited)),len(set(coords)),min(counts),sum(counts))
            candidates.append((score, tuple(actions)))
            return
        for action,next_state in sorted(states[current].transitions.items()):
            walk(next_state,visited+[next_state],actions+[action])
    walk(start,[start],[])
    fig.need(candidates,'No common three-observation path')
    best = max(sc for sc,_ in candidates)
    return list(min(path for sc,path in candidates if sc==best))


def png(root, relative, body, mode, files):
    # Public reference masks are 0/1; the lossless PNG uses 0/255 for direct viewing.
    frame = Image.frombytes(mode,(255,255),body)
    stream = io.BytesIO();frame.save(stream,format='PNG',compress_level=3)
    data = stream.getvalue()
    with Image.open(io.BytesIO(data)) as restored:
        fig.need(restored.mode==mode and restored.size==(255,255) and restored.tobytes()==body,
                 'PNG round trip did not reproduce source pixels')
    path = root/relative
    with path.open('xb') as f:f.write(data)
    files[relative] = {'sha256':sha(data),'bytes':len(data),'mode':mode,'decoded_sha256':sha(body)}
    return relative


def write_capture(root, stem, capture, roles, files):
    raw = png(root,stem+'_raw.png',capture.rgb.data,'RGB',files)
    masks,counts,boxes,hashes = {},{},{},{}
    for role,obj in roles.items():
        bits = capture.masks[obj]
        data = bits.translate(bytes.maketrans(b'\x00\x01',b'\x00\xff'))
        masks[role] = png(root,stem+'_mask_'+role+'.png',data,'L',files)
        box = Image.frombytes('L',(255,255),data).getbbox()
        boxes[role] = [box[0],box[1],box[2]-box[0],box[3]-box[1]] if box else [0,0,0,0]
        counts[role] = bits.count(1);hashes[role] = sha(bits)
    return {'raw':raw,'masks':masks,'mask_px':counts,'bbox':boxes,'mask_sha256_01':hashes,
            'rgb_sha256':capture.rgb.sha256,'render_id':capture.render_id,
            'object_distance_to_agent_m':{role:capture.distances.get(obj) for role,obj in roles.items()}}


def export_group(root, row, record, order, table):
    group = row['id'];folder=root/group;folder.mkdir();paths={};frames=[]
    roles = {'X':table.anchor_name,'A':table.candidate_names[0],'B':table.candidate_names[1]}
    actions = sequence(table, order)
    for wi, original in enumerate(order):
        world = f'w{wi}';(folder/world).mkdir()
        env = fig.environment(table,row['key'],order)
        try:
            obs = env.reset(row['key'],world)
            for step in range(4):
                if step:
                    obs,_,done,info = env.step(actions[step-1])
                    fig.need(info['action_success'] and not done,'Illustration path failed in original environment')
                info=env.info;state=table.states[info['state_id']]
                public = 0 if state.shared else original
                cap=state.captures[public];native=state.captures[original]
                fig.need(env.frame.raw_rgb==cap.rgb,'Wrong public capture')
                marked,original_marks=mark_view(cap,table.labels,table.candidate_names)
                fig.need(marked==obs.image and original_marks==obs.marks,'Public observation differs from training marking')
                stem=f'{group}/{world}/t{step:02d}'
                pixel_data=write_capture(root,stem,cap,roles,paths)
                nat=write_capture(root,stem+'_native',native,roles,paths) if original!=public else pixel_data
                label_marks={};mark_patches={}
                for mark in original_marks:
                    ax,ay=mark.anchor_pixel;x0=min(247,max(0,ax-4));y0=min(243,max(0,ay-6))
                    label_marks[str(mark.label)]=[x0+4,y0+6];mark_patches[str(mark.label)]=[x0,y0,8,12]
                frames.append({**pixel_data,'world':world,'step':step,'action_prefix':actions[:step],
                    'action':None if step==0 else actions[step-1], 'state_id':info['state_id'],'dimensions':[255,255],
                    'marked':png(root,stem+'_marked.png',marked.data,'RGB',paths),'marked_rgb_sha256':marked.sha256,
                    'native_raw':nat['raw'],'native_masks':nat['masks'],'native_mask_px':nat['mask_px'],
                    'native_rgb_sha256':nat['rgb_sha256'],'native_bbox':nat['bbox'],'native_render_id':native.render_id,
                    'marks':label_marks,'mark_patch_xywh':mark_patches,'shared':state.shared,'shadow_divergent':state.shadow_divergent,
                    'tau':info['tau'],'tau_seen':info['x_seen_step'],'public_capture_original_world':public,
                    'original_world':original,'camera_agent_pose':native.agent,'viewpoint':state.view.viewpoint_id,
                    'heading_index':state.view.heading_index,'viewpoint_position':table.positions[state.view.viewpoint_id],
                    'frame_max_abs_interworld_diff':state.frame_max_abs_diff})
        finally:env.close()
    for i in range(4):
        fig.need(frames[i]['state_id']==frames[4+i]['state_id'],'Paired paths diverged in camera state')
        if frames[i]['shared']:
            fig.need(frames[i]['marked_rgb_sha256']==frames[4+i]['marked_rgb_sha256'],'Shared inputs differ')
    package=record['package'];src=package['row'];worlds={}
    for i,w in enumerate(order):
        snapshot=package['worlds'][w];objects={x['name']:x for x in snapshot['objects']}
        worlds[f'w{i}']={'original_world':w,'correct_target':table.target_names[w],
            'correct_label':table.labels[table.target_names[w]],'snapshot_reference':snapshot['reference'],
            'objects':{role:{k:objects[n][k] for k in ('name','objectType','position','rotation') if k in objects[n]}
                       for role,n in roles.items()},'measured_target_geometry':src['target_measurements'][w]}
    others = [f['world'] for f in frames if f['step']==0]
    fig.need(others==['w0','w1'],'Incorrect frame order')
    summary={'id':group,'dataset_key':row['key'],'pair_id':table.pair_id,'scene':table.scene,'kind':table.kind,
             'room_type':row['room'],'anchor':row['anchor'],'anchor_name':roles['X'],
             'candidate_names':list(table.candidate_names),'candidate_types':list(row['candidate_types']),
             'roles':roles,'fixed_labels':table.labels,'meta':group+'/meta.json','actions':actions,'frames':frames}
    document={**summary,'version':VERSION,'configuration':from_pair(src),'world_order':order,'worlds':worlds,
              'instruction':table.instruction,'table_manifest':table.manifest_reference,'certificate':record['evidence']['source'],
              'export_source':record['export_source'],'files':paths,'path_base':'Paths are relative to the gallery catalog',
              'sequence_source':'Shared actions selected from the original D4 table to illustrate visual evidence. No learned policy was run.',
              'color_and_masks':'255x255 source RGB, lossless PNG. X/A/B masks preserve original binary pixels as 0/255. Public RGB is before model marks; native fields identify each physical world capture.'}
    save_json(folder/'meta.json',document)
    return summary


def collect(name, start, count, limit):
    manifest,proof,qualification,_=certificates.authenticate_certificate_qualification(QUALIFICATION)
    heldout=set(fig.read_reference(HOLDOUT)['selected_scenes'])
    entries,_,_,evidences=scan_training(manifest,proof,heldout)
    original_keys={r['dataset_key'] for r in manifest['entries']};rows=[]
    for key,e in sorted(entries.items()):
        if e['scene'] in heldout or e['scene'] not in TRAIN_SCENES or (key in original_keys and key not in proof['train_keys']):continue
        if e['kind'] not in KINDS:continue
        src=fig.read_reference(e['pair_reference']);cfg=from_pair(src);ids=src['identities']
        room=['kitchen','living','bedroom','bathroom'][sum(int(e['scene'][9:])>=x for x in (200,300,400))]
        rows.append({'key':key,'kind':e['kind'],'room':room,'scene':e['scene'],'anchor':ids['anchor_type'],
                     'candidate_types':tuple(ids['candidate_types']),'configuration':cfg['configuration_id'],
                     'id':f"{src['pair_id']}_{e['kind']}_{key[5:13]}"})
    plan=choose(rows,limit);batch=plan[start:start+count];fig.need(batch,'No examples in chosen slice')
    root=ROOT/'logs/review'/name;root.mkdir(exist_ok=False);targets={r['key']:r for r in batch};accessed=[];groups={}
    for ev in evidences:
        candidates=[r for r in ev['complete'] if r['dataset_key'] in targets and r['dataset_key'] not in groups]
        if not candidates:continue
        headers=fig.certified_headers(ev,entries,accessed)
        for entry in candidates:
            key=entry['dataset_key'];row=targets[key]
            record=fig.accepted_pair_metadata(ev,entry,headers[key],accessed)
            table,_=fig.validate_selected_table(record,expected_kind=row['kind'])
            groups[key]=export_group(root,row,record,entries[key]['world_order'],table)
    fig.need(set(groups)==set(targets),'Missing selected certified tables')
    catalog={'version':VERSION,'groups':[groups[r['key']] for r in batch], 'qualification':qualification,
             'holdout_reference':HOLDOUT,'excluded_scenes':sorted(heldout),'available_certified_training':len(rows),
             'configuration_distinct':True,'planned_count':len(plan),'batch_start':start,
             'full_selection':[r['key'] for r in plan],'scope':'Illustration samples from cache-verified, non-holdout training scenes. No formal test scene or simulator.'}
    ref=save_json(root/'catalog.json',catalog)
    return {'catalog':ref,'groups':len(batch),'frames':sum(len(g['frames']) for g in groups.values()),
            'classes':dict(Counter(r['kind'] for r in batch)),'scenes':sorted({r['scene'] for r in batch})}


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--start',type=int,required=True)
    p.add_argument('--count',type=int,required=True);p.add_argument('--limit',type=int,default=24);a=p.parse_args()
    fig.need(re.fullmatch('diag_fig_training_gallery_[A-Za-z0-9_]+',a.name) and 0<=a.start<a.limit<=24 and 1<=a.count<=8,'Invalid batch')
    dest=ROOT/'results'/(a.name+'.json');fig.need(not dest.exists(),'Use a fresh run')
    metrics={'complete':False,'gpus':0,'errors':[]};status=1
    try:metrics.update(complete=True,**collect(a.name,a.start,a.count,a.limit));status=0
    except Exception as exc:traceback.print_exc();metrics['errors'].append(f'{type(exc).__name__}: {exc}')
    save_json(dest,{'metrics':metrics,'config':vars(a)|{'version':VERSION,'GPU':False}})
    print('FINAL_METRICS_JSON: '+json.dumps(metrics,sort_keys=True),flush=True);return status


if __name__=='__main__':raise SystemExit(main())

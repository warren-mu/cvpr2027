"""Deliver the chosen Box flip with a visible Box stay, or the selected Bread control."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re
import traceback
import zipfile

from PIL import Image

from experiments import fig01_export as fig
from experiments import fig_kettle_alternatives as previews
from src.envs.pair_table import _DIGITS
from src.envs.thor_common import ROOT
from src.tools import smoke_e2e_pilot_certificates as cert
from src.tools.t01_partition import audited_result
from src.tools.t02_export import save_json


FRAMES = ('s0','w0_look','w1_look','w1_seen','stay_w0_seen','stay_w1_seen')
TRAIN_SCENES = {f'FloorPlan{base+i}' for base in (0,200,300,400) for i in range(1,21)}


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def scan_training(manifest, proof, heldout):
    entries={r['dataset_key']:r for r in manifest['entries']}
    sources={Path(r['file']).name:r for r in manifest['cache_sources']}
    for path in sorted((ROOT/'results').glob('t02_export_cache*.json')):
        if 'test' in path.stem or path.name in sources:
            continue
        d=json.loads(path.read_bytes());m=d.get('metrics',{})
        if d['_meta'].get('exit_code')==0 and m.get('kind')=='t02_formal_cache_check' and m.get('cache_matches_live') is True:
            sources[path.name]={'file':str(path.relative_to(ROOT)),'sha256':digest(path.read_bytes())}
    by_key={};inventory=[];seen=set();evidences=[]
    for reference in sources.values():
        ev=cert.certificate_metadata(reference,root=ROOT);evidences.append(ev)
        for row in ev['complete']:
            key=row['dataset_key'];ref=ev['certificate']['tables'][key]
            if key not in entries:
                h=fig.read_reference(ref)
                entries[key]={'dataset_key':key,'scene':h['scene'],'kind':h['kind'],'original_pair_id':h['pair_id'],
                    'pair_reference':h['source']['t01']['pair'],'table_manifest':ref,'world_order':[0,1]}
            e=entries[key]
            fig.need(e['table_manifest']==ref,'Conflicting certified table identity')
            by_key.setdefault(key,(ev,row))
            if key in seen or e['kind']!='look_stay' or e['scene'] in heldout or e['scene'] not in TRAIN_SCENES:
                continue
            if key in {r['dataset_key'] for r in manifest['entries']}:
                fig.need(key in proof['train_keys'],'Validation row entered training inventory')
            seen.add(key);p=fig.read_reference(e['pair_reference']);ident=p['identities']
            inventory.append({'dataset_key':key,'pair_id':p['pair_id'],'scene':p['scene'],
                'anchor':ident['anchor_type'],'anchor_name':ident['anchor_name'],'pair_reference':e['pair_reference']})
    return entries,by_key,inventory,evidences


def marks(image, rows):
    pixels=image.load();centers={};details={}
    for r in rows:
        label=str(r['label']);ax,ay=r['anchor_pixel']
        x0=min(247,max(0,ax-4));y0=min(243,max(0,ay-6))
        for dy in range(12):
            for dx in range(8):
                white=(1<=dx<=6 and 1<=dy<=10 and _DIGITS[label][(dy-1)//2][(dx-1)//2]=='1')
                v=255 if white else 0
                fig.need(pixels[x0+dx,y0+dy]==(v,v,v),'Displayed digit patch does not match training glyph')
        centers[label]=[x0+4,y0+6]
        details[label]={'patch_xywh':[x0,y0,8,12],'mask_anchor_pixel':[ax,ay],'patch_rgb_verified':True}
    return centers,details


def execute(run, directory):
    original,preview_ref=audited_result('results/diag_fig_kettle_previews_01.json')
    fig.need(original['_meta']['exit_code']==0 and original['metrics']['complete'],'Need sealed preview results')
    report=fig.read_reference(original['metrics']['report']);options={o['option']:o for o in report['options']}
    inv,_=audited_result('results/diag_fig_kettle_inventory_01.json');heldout=set(fig.read_reference(inv['metrics']['report'])['holdout_scenes'])
    manifest,proof,_,_=cert.authenticate_certificate_qualification(previews.QUALIFICATION)
    entries,by_key,inventory,evidences=scan_training(manifest,proof,heldout)
    fkey=options['C']['flip']['dataset_key'];fallback=options['B']['stay']['dataset_key']
    boxes=[]
    for r in inventory:
        if r['anchor']=='Box':
            e=entries[r['dataset_key']];plan=previews.rank_actions(fig.read_reference(e['table_manifest']),e['world_order'])
            boxes.append((-plan['rank_min_seen'],r['dataset_key']))
    skey=min(boxes)[1] if boxes else fallback
    fig.need(any(r['dataset_key']==skey for r in inventory),'Chosen control lacks a training cache certificate')
    selected={fkey,skey};tables={};identities={};accessed=[]
    for ev in evidences:
        wanted=[r for r in ev['complete'] if r['dataset_key'] in selected and r['dataset_key'] not in tables]
        if not wanted:continue
        headers=fig.certified_headers(ev,entries,accessed)
        for r in wanted:
            key=r['dataset_key'];record=fig.accepted_pair_metadata(ev,r,headers[key],accessed)
            table,_=fig.validate_selected_table(record,expected_kind=entries[key]['kind']);tables[key]=table
            meta=previews.metadata(table,record,key,entries[key]['world_order'])
            meta.update(anchor=meta['anchor_type'],objectType=meta['anchor_type'],pair_id=meta['id'],
                configuration={'id':meta['configuration_id'],**meta['configuration_fields']})
            if key==skey:meta['anchor_horizontal_displacement']=record['package']['row']['stay_horizontal_displacement']
            identities[key]=meta
    fig.need(identities[fkey]['anchor']=='Box' and identities[fkey]['target_labels_agent_worlds']==[1,2],
             'Primary example changed')
    fig.need(len(set(identities[skey]['target_names_agent_worlds']))==1,'Control target changed')
    if not boxes:fig.need(identities[skey]['scene']=='FloorPlan7' and identities[skey]['anchor']=='Bread','Bread fallback changed')
    flip=options['C']['frames'];stay=options['B']['frames'] if skey==fallback else None
    splan=previews.rank_actions(fig.read_reference(entries[skey]['table_manifest']),entries[skey]['world_order'])
    docs={};blobs={};verification={};base=Path(original['metrics']['report']['file']).parent
    for name in FRAMES:
        is_stay=name.startswith('stay_'); key=skey if is_stay else fkey; world='w1' if 'w1' in name else 'w0'
        prior=stay if is_stay else flip
        actions=prior[name]['actions'] if prior else [splan['w1_from_s0' if world=='w1' else 'look']]
        image,row=previews.capture(tables[key],key,entries[key]['world_order'],world,actions,None)
        if prior:
            reference=prior[name];png=(base/('B' if is_stay else 'C')/reference['file']).read_bytes()
            fig.need(digest(png)==reference['png_sha256'] and row['anchor_bbox_xywh']==reference['anchor']
                     and row['mask_pixels']==reference['mask_px'] and row['state_id']==reference['state_id'],
                     'Selected frame or instance mask differs from original preview')
        else:
            buf=io.BytesIO();image.save(buf,format='PNG');png=buf.getvalue()
        with Image.open(io.BytesIO(png)) as restored:
            fig.need(restored.mode=='RGB' and restored.size==(255,255) and restored.tobytes()==image.tobytes(),
                     'Delivered bytes differ from the actual environment observation')
        centers,patches=marks(image,row['marks'])
        docs[name]={'file':name+'.png','anchor':row['anchor_bbox_xywh'],'marks':centers,
            'mask_px':row['mask_pixels'],'seen':row['mask_pixels']>=250,'world':world,'actions':actions,
            'state_id':row['state_id'],'tau':row['tau'],'tau_seen':row['tau_seen'],'dimensions':[255,255],
            'rgb_sha256':row['decoded_rgb_sha256'],'png_sha256':digest(png)}
        blobs[name+'.png']=png;verification[name]={'marks':patches,'mask_sha256':row['mask_sha256'],'render_id':row['render_id']}
    w1,match=previews.capture(tables[fkey],fkey,entries[fkey]['world_order'],'w1',[],None)
    fig.need(match['mask_pixels']==0 and docs['s0']['rgb_sha256']==digest(w1.tobytes()),'World s0 input differs')
    buf=io.BytesIO();w1.save(buf,format='PNG');fig.need(buf.getvalue()==blobs['s0.png'],'Independent s0 PNG differs')
    acts=flip['_actions']; docs['_actions']={**acts,'stay_w0':docs['stay_w0_seen']['actions'][0],'stay_w1':docs['stay_w1_seen']['actions'][0]}
    for key,names in [(fkey,('w0_look','w1_seen')),(skey,('stay_w0_seen','stay_w1_seen'))]:
        for field in ['tau','tau_seen','actions']:
            identities[key][field]={w:docs[n][field] for w,n in zip(('w0','w1'),names)}
    meta={'flip':identities[fkey],'stay':identities[skey], 'scene':identities[fkey]['scene'],
          'pair_ids':{'flip':identities[fkey]['pair_id'],'stay':identities[skey]['pair_id']},
          'selection':{'flip':'C','stay':'B fallback' if not boxes else 'accepted Box control',
                       'box_stay_count':len(boxes),'certified_training_stay_counts_by_anchor':dict(Counter(r['anchor'] for r in inventory))},
          'tau':{k:identities[v]['tau'] for k,v in [('flip',fkey),('stay',skey)]},
          'tau_seen':{k:identities[v]['tau_seen'] for k,v in [('flip',fkey),('stay',skey)]},
          'anchor_mask_px':{n:docs[n]['mask_px'] for n in FRAMES},
          'empty_mask_bbox_convention':'[0,0,0,0] when empty. Other boxes use inclusive mask extrema expressed as x,y,width,height.',
          'marks_center_convention':'Origin is top-left. For each drawn numeric 8x12 patch starting at clamped (x0,y0), the center is [x0+4,y0+6], rounded toward right/down. Omitted labels were not drawn.',
          's0_rgb_sha256':{'w0':docs['s0']['rgb_sha256'],'w1':digest(w1.tobytes())},
          's0_png_sha256':{'w0':docs['s0']['png_sha256'],'w1':digest(buf.getvalue())},
          'sweep_policy':flip['_meta']['sweep_policy'],'shortestness_postcheck':flip['_meta']['shortestness_postcheck'],
          'source_preview_result':preview_ref,'verification':verification,'overview':'No verified third-party camera view in the selected original artifacts.'}
    docs['_meta']=meta
    blobs['frames.json']=(json.dumps(docs,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
    checks=['FLIP independent world s0 input checks']
    for w in ('w0','w1'):
        checks.extend([f"{w} RGB SHA-256 {meta['s0_rgb_sha256'][w]}",f"{w} PNG SHA-256 {meta['s0_png_sha256'][w]}"])
    checks.append('Both independent world inputs and PNG byte streams are identical.')
    for n in FRAMES:checks.append(f"{n}.png: 255x255 RGB; PNG SHA-256 {docs[n]['png_sha256']}; decoded RGB SHA-256 {docs[n]['rgb_sha256']}; drawn marks {docs[n]['marks']}")
    blobs['checks.txt']=('\n'.join(checks)+'\n').encode()
    archive=directory/'frames.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_STORED) as z:
        for name,blob in blobs.items():
            entry=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));entry.external_attr=0o100644<<16;z.writestr(entry,blob)
    with zipfile.ZipFile(archive) as zipped:
        fig.need(zipped.testzip() is None and list(blobs)==zipped.namelist() and all(zipped.read(n)==b for n,b in blobs.items()),'ZIP verification failed')
    print(blobs['checks.txt'].decode(),end='')
    return {'selection':meta['selection'],'flip':meta['flip'],'stay':meta['stay'],'archive':fig.file_ref(archive),
            'original_stays_checked':inventory,'frames_manifest':save_json(directory/'frames.json',docs),
            's0_rgb_sha256':meta['s0_rgb_sha256'],'s0_png_sha256':meta['s0_png_sha256']}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True);a=parser.parse_args()
    fig.need(re.fullmatch('diag_fig_selected_frames_[A-Za-z0-9_]+',a.name),'Fresh final export name required')
    directory=ROOT/'logs/review'/a.name;directory.mkdir(exist_ok=False)
    out=ROOT/'results'/(a.name+'.json');fig.need(not out.exists(),'Preserve prior result')
    metrics={'complete':False,'gpus':0,'errors':[]};code=1
    try:
        report=execute(a.name,directory);metrics.update(complete=True,report=save_json(directory/'report.json',report),
            archive=report['archive'],selection=report['selection']);code=0
    except Exception as exc:
        traceback.print_exc();metrics['errors'].append(f'{type(exc).__name__}: {exc}')
    save_json(out,{'metrics':metrics,'config':{'name':a.name,'gpus':0,'selection':'C primary; Box stay preferred, B fallback'}})
    print('FINAL_METRICS_JSON: '+json.dumps(metrics,sort_keys=True),flush=True);return code


if __name__=='__main__':raise SystemExit(main())

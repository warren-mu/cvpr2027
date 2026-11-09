"""Construct and replay one fixed T01v2 scene inside a verified PBS shard.

Run only through run_logged. Geometry orders proposals; only measured policy
checks and fresh-process witnesses admit a pair. Rejections are append-only.
"""
import argparse
import copy
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

from .thor_common import (ROOT, Worker, controller_session, deadline_after,
                          install_signals, prepare_runtime, write_result)
from .pair_v2_construct import (identity_sets, view_proposals, placement_groups,
                                construction_pairs, adjacent_moves, distance)

KINDS = ('direct_flip', 'direct_stay', 'look_flip', 'look_stay')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return {'file': str(path), 'sha256': sha(path)}


def action_text(action):
    if action['kind'] in ('turn_left', 'turn_right'):
        return action['kind']
    if action['kind'] == 'move':
        return 'move ' + str(action['index'])
    if action['kind'] == 'select':
        return 'select ' + str(action['label'])
    raise ValueError('Unknown public action')


def marked(sim, capture, identities):
    from .pair_v2_policy import mark_observation
    instruction = f"Select the {identities['target_type']} closest to the {identities['anchor_type']}."
    return mark_observation(capture, identities['candidate_names'], instruction,
        public_context='Actions: move <index>; turn_left; turn_right; select <mark>.')


def eligible_select(sim, observation, target):
    row = sim.select(target)
    label = observation['mark_labels_by_name'].get(target)
    return bool(row['success'] and label is not None), label, row


def physical_measurement(result):
    from .pair_v2_policy import physical_check
    raw = result['physical']
    return physical_check(raw['before_objects'], raw['requested_poses'], raw['realized_objects'],
        raw['edited_names'], native_is_scene_at_rest=raw['native_is_scene_at_rest'],
        placement_evidence_by_name=raw['placement_evidence_by_name'])


def wait_replay(config, snapshot, actions, target, identities, directory):
    from .pair_v2_simulator import replay_worker
    child_config = {**config, 'snapshot': snapshot, 'actions': actions,
        'target_name': target, **identities,
        'instruction': f"Select the {identities['target_type']} closest to the {identities['anchor_type']}.",
        'raw_dir': str(directory)}
    child = Worker(replay_worker, child_config)
    result = None
    try:
        while result is None:
            child.receive()
            if child.result is not None:
                result = child.result
            elif child.error:
                result = {'status': 'worker_failed', 'error': child.error, 'success': False}
            elif time.time() >= config['deadline'] - 10:
                raise TimeoutError('Witness replay deadline')
            else:
                time.sleep(.05)
    finally:
        cleanup = child.stop()
    result.setdefault('cleanup_errors', []).extend(cleanup)
    if cleanup:
        result['success'] = False
    return result


def observation_actions(sim, reachable, start):
    actions = [{'kind': 'turn_left'}, {'kind': 'turn_right'}]
    rejected = []
    for index, point in enumerate(adjacent_moves(reachable, start['agent']['position']), 1):
        try:
            path = sim.shortest_path(start['agent']['position'], point)
            if path['complete']:
                actions.append({'kind': 'move', 'index': index, 'position': point, 'path': path})
            else:
                rejected.append({'position': point, 'reason': 'native_path_incomplete', 'path': path})
        except (TimeoutError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:
            rejected.append({'position': point, 'reason': f'{type(error).__name__}: {error}'})
    return actions, rejected


def inverse_action(sim, action, start):
    if action['kind'] == 'turn_left':
        return {'kind': 'turn_right'}
    if action['kind'] == 'turn_right':
        return {'kind': 'turn_left'}
    current = sim.snapshot('inverse_path_source')
    path = sim.shortest_path(current['agent']['position'], start['agent']['position'])
    return {'kind': 'move', 'index': 1, 'position': start['agent']['position'], 'path': path}


def choose_look(sim, worlds, targets, identities, reachable, prefix):
    """Evaluate actual common one-action evidence, then shortest valid continuation.

    The selected graph is the adjacent native grid and 45-degree rotations.
    Every start candidate is already selectable. Thus, given a first observation,
    select is either legal now (2 actions) or an actually verified inverse reaches
    a legal select (3). We compare all qualified first actions unless both worlds
    reach the absolute two-action lower bound.
    """
    from .pair_v2_policy import compare_view, area_passes
    actions, unavailable = observation_actions(sim, reachable, worlds[0])
    choices, attempts = [], []
    relevant = [*identities['candidate_names'], identities['anchor_name']]
    for action in actions:
        observed, plans, details = [], [], []
        try:
            for world, target in zip(worlds, targets):
                _, restoration = sim.restore_snapshot(world)
                if restoration.get('restore_deterministic') is not True:
                    raise ValueError('Planner snapshot restoration failed the state-level gate')
                world_action = copy.deepcopy(action)
                if action['kind'] == 'move':
                    world_action['path'] = sim.shortest_path(world['agent']['position'], action['position'])
                    if world_action['path'].get('complete') is not True:
                        raise ValueError('Common public move is not feasible in this restored world')
                capture = sim.observe(world_action)
                obs = marked(sim, capture, identities)
                observed.append(obs)
                possible, label, selection = eligible_select(sim, obs, target)
                plan = [world_action]
                if not possible:
                    reverse = inverse_action(sim, action, world)
                    returned = marked(sim, sim.observe(reverse), identities)
                    possible, label, selection = eligible_select(sim, returned, target)
                    plan.append(reverse)
                if not possible:
                    raise ValueError('Inverse observation did not reach a valid select')
                plan.append({'kind': 'select', 'label': label})
                plans.append(plan)
                details.append(selection)
            evidence = compare_view(observed[0], observed[1], relevant)
            anchor_counts = [int(o['masks_by_name'].get(identities['anchor_name'],
                __import__('numpy').zeros((255,255), dtype=bool)).sum()) for o in observed]
            passed = (evidence['evidence_passed'] and evidence['marked_evidence_passed']
                      and all(area_passes(n,255,255) for n in anchor_counts))
            attempts.append({'action': action_text(action), 'evidence': evidence,
                             'anchor_pixels': anchor_counts, 'passed': passed,
                             'plan_lengths': [len(p) for p in plans]})
            if passed:
                choices.append({'action': action, 'after': observed, 'plans': plans,
                    'evidence': evidence, 'anchor_pixels': anchor_counts,
                    'planning': {'method': 'shortest measured continuation after a common one-action evidence acquisition',
                        'action_counts': [len(p) for p in plans], 'selection_measurements': details,
                        'lower_bound_including_evidence_and_select': 2}})
                if all(len(plan) == 2 for plan in plans):
                    break
        except (TimeoutError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:
            attempts.append({'action': action_text(action), 'passed': False,
                             'error': f'{type(error).__name__}: {error}'})
    if not choices:
        return None, {'actions': attempts, 'unavailable_moves': unavailable}
    choice = min(choices, key=lambda row: (sum(map(len,row['plans'])),
        max(map(len,row['plans'])), action_text(row['action'])))
    choice['planning'].update(first_action_search=attempts, unavailable_moves=unavailable,
                             global_two_action_lower_bound_attained=all(len(p)==2 for p in choice['plans']))
    return choice, {'actions': attempts, 'unavailable_moves': unavailable}


def review_sheet(path, row, prefix, after, targets):
    from PIL import Image, ImageDraw
    panel_size = 384
    sheet = Image.new('RGB', (panel_size*2, (panel_size+26)*2+60), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((8,5), row['pair_id']+' | '+row['kind'], fill='black')
    draw.text((8,24), row['instruction'], fill='black')
    for i, obs in enumerate([*prefix,*after]):
        image = Image.fromarray(obs['marked']).resize((panel_size,panel_size))
        mark_label = obs['mark_labels_by_name'].get(targets[i % 2])
        painter = ImageDraw.Draw(image)
        for mark in obs['marks']:
            if mark['label'] == mark_label:
                painter.rectangle(tuple(round(v*panel_size/255) for v in mark['bounding_box']), outline='lime', width=3)
        x, y = (i%2)*panel_size, 60+(i//2)*(panel_size+26)
        draw.text((x+5,y+5), ('v0 prefix' if i<2 else 'after observation')+f' / w{i%2}', fill='black')
        sheet.paste(image,(x,y+26))
    path.parent.mkdir(parents=True,exist_ok=True)
    sheet.save(path,format='PNG')
    return {'file':str(path),'sha256':sha(path)}


def export_pair(config, row, worlds, prefix, after, witnesses, constructions):
    """Small portable training/review essentials; complete raw attempts stay scratch."""
    from PIL import Image
    import numpy as np
    directory = ROOT/'data/world_pairs/v2'/config['name']/row['pair_id']
    directory.mkdir(parents=True,exist_ok=False)
    files = []
    observations = {'prefix': [], 'after': []}
    preflight_ref=config['scene_preflight_reference']
    if sha(preflight_ref['file'])!=preflight_ref['sha256']:
        raise ValueError('Original scene preflight evidence changed')
    preflight_path=directory/'scene_preflight.json'
    with Path(preflight_ref['file']).open('rb') as incoming,preflight_path.open('xb') as outgoing:
        shutil.copyfileobj(incoming,outgoing)
    row['scene_preflight_reference']={'file':str(preflight_path.relative_to(ROOT)),'sha256':sha(preflight_path)}
    files.append(row['scene_preflight_reference'])
    for label, captures in [('prefix',prefix),('after',after)]:
        for index, obs in enumerate(captures):
            entry={k:copy.deepcopy(obs[k]) for k in ('metadata','marks','mark_labels_by_name','input_text','history')}
            entry['files']={}
            for variant in ('rgb','marked'):
                path = directory/f'{label}_w{index}_{variant}.png'
                Image.fromarray(obs[variant]).save(path)
                ref={'file':str(path.relative_to(ROOT)),'sha256':sha(path)}
                files.append(ref);entry['files'][variant]=ref
            path=directory/f'{label}_w{index}_masks.npz'
            names=sorted(obs['masks_by_name'])
            np.savez_compressed(path,names=np.asarray(names,dtype=str),
                masks=np.stack([obs['masks_by_name'][n] for n in names]) if names else np.zeros((0,255,255),bool))
            ref={'file':str(path.relative_to(ROOT)),'sha256':sha(path)}
            files.append(ref);entry['files']['masks']=ref
            observations[label].append(entry)
    if row['kind'].startswith('look'):
        # Both branches refer to this SAME public prefix; no branch-dependent copy.
        row['shared_prefix_file'] = str((directory/'prefix_w0_marked.png').relative_to(ROOT))
        row['shared_prefix_sha256'] = sha(directory/'prefix_w0_marked.png')
        first=observations['prefix'][0]
        public={k:copy.deepcopy(first[k]) for k in ('input_text','marks','history')}
        public.update(rgb=first['files']['rgb'],marked=first['files']['marked'])
        shared_path=directory/'shared_public_prefix.json';write_json(shared_path,public)
        shared_ref={'file':str(shared_path.relative_to(ROOT)),'sha256':sha(shared_path)}
        files.append(shared_ref)
        row['shared_public_prefix_by_world']={'w0':shared_ref,'w1':copy.deepcopy(shared_ref)}
    for index, (world, replay) in enumerate(zip(worlds,witnesses)):
        restored_ref=(replay.get('restoration') or {}).get('restored_snapshot_reference')
        if not restored_ref or sha(restored_ref['file'])!=restored_ref['sha256']:
            raise ValueError('Replay restored-snapshot evidence missing or changed')
        restored=json.loads(Path(restored_ref['file']).read_text())
        for label,value in [('snapshot',world),('witness',replay),('restored_snapshot',restored),('construction',constructions[index])]:
            p=directory/f'{label}_w{index}.json';write_json(p,value)
            files.append({'file':str(p.relative_to(ROOT)),'sha256':sha(p)})
    p=directory/'observations.json';write_json(p,observations)
    files.append({'file':str(p.relative_to(ROOT)),'sha256':sha(p)})
    image = review_sheet(directory/'review.png',row,prefix,after,[row['target_w0'],row['target_w1']])
    row['review_image']={'file':str(Path(image['file']).relative_to(ROOT)),'sha256':image['sha256']}
    row['portable_files']=files+[row['review_image']]
    write_json(directory/'pair.json',row)
    return str((directory/'pair.json').relative_to(ROOT))


def build_pair(sim, config, baseline, identities, proposal, view, pair_id, directory, reachable, attempt_row=None):
    import numpy as np
    from .pair_v2_policy import (closest_target, compare_view, prefix_check, public_copy,
                                single_edit_check, witness_check, classify_pair)
    row = {} if attempt_row is None else attempt_row
    row.update({'pair_id':pair_id,'scene':config['scene'],'mode':proposal['mode'],
         'instruction':f"Select the {identities['target_type']} closest to the {identities['anchor_type']}.",
         'identities':identities,'proposal':proposal,'view':view,'status':'rejected','discard_reasons':[]})
    sim.position_agent(view['position'],view['yaw'],view['horizon'])
    agent=copy.deepcopy(sim.snapshot(pair_id+'_v0')['agent'])
    w0=sim.construct_world(baseline,pair_id+'_w0',
        [{'object_name':identities['anchor_name'],'spawn':proposal['w0']['spawn']}],require_agent=agent)
    phys0=physical_measurement(w0)
    row['physical_w0']=phys0
    if not w0['passed'] or not phys0['valid_physical']:
        row['discard_reasons']=['w0_physical'];return row
    w1=sim.construct_world(w0['snapshot'],pair_id+'_w1',
        [{'object_name':identities['anchor_name'],'spawn':proposal['w1']['spawn']}],require_agent=agent)
    phys1=physical_measurement(w1)
    row['physical_w1']=phys1
    if not w1['passed'] or not phys1['valid_physical']:
        row['discard_reasons']=['w1_physical'];return row
    worlds=[w0['snapshot'],w1['snapshot']]
    targets=[closest_target(world['objects'],identities['candidate_names'],identities['anchor_name']) for world in worlds]
    row['target_measurements']=targets
    if not all(t['passed'] for t in targets):
        row['discard_reasons']=['target_uniqueness_or_ratio'];return row
    target_names=[t['target_name'] for t in targets]
    row.update(target_w0=target_names[0],target_w1=target_names[1])
    if (target_names[0]!=target_names[1]) != (proposal['mode']=='flip'):
        row['discard_reasons']=['realized_target_mode'];return row
    prefix=[marked(sim,world['capture'],identities) for world in (w0,w1)]
    pre=prefix_check([prefix[0]],[prefix[1]],identities['candidate_names'],identities['anchor_name'],
        same_process=worlds[0]['worker_pid']==worlds[1]['worker_pid'],
        agent_unchanged=worlds[0]['agent']==worlds[1]['agent'])
    row['prefix_measurement']=pre
    if not pre['candidates_visible_and_marked']:
        row['discard_reasons']=['v0_candidates_visibility'];return row
    # Constructions are restricted to start viewpoints where either select is
    # possible. This makes the shortest-witness proof bounded and checkable.
    for world,obs in zip(worlds,prefix):
        for name in identities['candidate_names']:
            obj=next(o for o in world['objects'] if o['name']==name)
            if distance(world['agent']['position'],obj['position'])>1.5:
                row['discard_reasons']=['v0_candidate_outside_select_range'];return row
    if pre['render_qualified_for_shared_prefix']:
        row['kind']='look_'+proposal['mode']
        shared=[public_copy(prefix[0])]
        pre=prefix_check([prefix[0]],[prefix[1]],identities['candidate_names'],identities['anchor_name'],
            same_process=True,agent_unchanged=True,shared_prefix_w0=shared,shared_prefix_w1=copy.deepcopy(shared))
        choice,search=choose_look(sim,worlds,target_names,identities,reachable,prefix)
        row['observation_search']=search
        if choice is None:
            row['discard_reasons']=['one_action_evidence_unavailable'];return row
        after,plans,evidence,anchor_pixels=choice['after'],choice['plans'],choice['evidence'],choice['anchor_pixels']
        common=action_text(choice['action']);row['planning']=choice['planning']
    else:
        row['kind']='direct_'+proposal['mode']
        evidence=compare_view(*prefix,[*identities['candidate_names'],identities['anchor_name']])
        anchor_pixels=[int(obs['masks_by_name'].get(identities['anchor_name'],np.zeros((255,255),bool)).sum()) for obs in prefix]
        if not evidence['evidence_passed'] or not evidence['marked_evidence_passed'] or min(anchor_pixels)<=0:
            row['discard_reasons']=['neither_hidden_shared_prefix_nor_direct_evidence'];return row
        plans=[[{'kind':'select','label':obs['mark_labels_by_name'][target]}] for obs,target in zip(prefix,target_names)]
        common=None
        # Review sheets include a real common observation even for direct cases;
        # it is illustrative and is not used to establish t=0 evidence.
        after=[]
        for world in worlds:
            sim.restore_snapshot(world)
            after.append(marked(sim,sim.observe({'kind':'turn_right'}),identities))
        row['planning']={'method':'terminal select at start; global one-action lower bound attained','action_counts':[1,1]}
    row.update(prefix_measurement=pre,evidence=evidence,anchor_pixels=anchor_pixels,
               witness_actions=[[action_text(a) for a in plan] for plan in plans])
    single=single_edit_check(w0['snapshot']['poses'],w1['physical']['requested_poses'],identities['anchor_name'])
    row['single_edit']=single
    if not single['passed']:
        row['discard_reasons']=['more_than_anchor_requested_edit'];return row
    witnesses=[]
    for index,(world,plan,target) in enumerate(zip(worlds,plans,target_names)):
        replay=wait_replay(config,world,plan,target,identities,directory/f'replay_w{index}')
        if replay.get('exit_code')==86:
            from .pair_v2_simulator import BlackFrame
            raise BlackFrame(replay.get('black_frame_reference'))
        witnesses.append(replay)
    checked=[witness_check(record,target) for record,target in zip(witnesses,target_names)]
    restored=all(record.get('restore_deterministic') is True for record in witnesses)
    pixel=[record.get('restore_max_abs_diff') for record in witnesses]
    verdict=classify_pair(row['kind'],target_w0=targets[0],target_w1=targets[1],prefix=pre,evidence=evidence,
        physical_w0=phys0,physical_w1=phys1,single_edit=single,witness_w0=checked[0],witness_w1=checked[1],
        restore_deterministic=restored,restore_max_abs_diff=pixel,
        anchor_pixels_w0=anchor_pixels[0],anchor_pixels_w1=anchor_pixels[1],common_observation_action=common)
    row.update(validity=verdict,restore_deterministic=restored,restore_max_abs_diff=pixel,
               witnesses=checked,discard_reasons=verdict['reasons'])
    if verdict.get('acceptance_passed') is True and not any(w.get('cleanup_errors') for w in witnesses):
        row['status']='passed'
        row['pair_file']=export_pair(config,row,worlds,prefix,after,witnesses,[w0['physical'],w1['physical']])
        row['pair_sha256']=sha(ROOT/row['pair_file'])
    return row


def summarize(rows,excluded,status,config):
    counts={kind:sum(row.get('kind')==kind and row['status']=='passed' for row in rows) for kind in KINDS}
    reasons=Counter(reason for row in rows for reason in row.get('discard_reasons',[]))
    times=[row['generation_seconds'] for row in rows if 'generation_seconds' in row]
    return {'kind':'pair_v2_shard','status':status,'acceptance_passed':False,
        'passed_by_class':counts,'attempts':len(rows),'discard_reason_counts':dict(reasons),
        'scene_excluded':excluded,'scene_exclusion_count':int(excluded is not None),
        'mean_attempt_generation_seconds':sum(times)/len(times) if times else None,
        'timing_denominator':'all recorded pair construction attempts including rejections',
        'passed_pair_files':[{'file':row['pair_file'],'sha256':row['pair_sha256']} for row in rows if row.get('pair_file')],
        'attempt_journal':config['attempt_journal'],'cleanup_errors':[], 'black_frame_detected':False,
        'scene_preflight_reference':config.get('scene_preflight_reference')}


def generator_worker(config,connection):
    from .pair_v2_simulator import V2Simulator,BlackFrame,BUILD,objects_by_name
    rows,excluded,sim=[],None,None
    journal=ROOT/config['attempt_journal'];journal.parent.mkdir(parents=True,exist_ok=True)
    status='completed_search'
    with journal.open('x') as handle:
        try:
            with controller_session(config,'v2_construction',segmentation=True,connection=connection) as controller:
                if getattr(getattr(controller,'_build',None),'commit_id',None)!=BUILD:
                    raise RuntimeError('Unity build differs from reviewed fixed source')
                sim=V2Simulator(controller,config,Path(config['raw_evidence_dir'])/'construction')
                preflight=sim.preflight_scene()
                preflight_path=sim.store.directory/'scene_preflight.json'
                config['scene_preflight_reference']={'file':str(preflight_path),'sha256':sha(preflight_path)}
                if not preflight['passed']:
                    preflight_path=sim.store.directory/'scene_preflight.json'
                    excluded={'scene':config['scene'],'reason':'original_scene_not_settled_within_1cm',
                              'evidence':{'file':str(preflight_path),'sha256':sha(preflight_path)}}
                else:
                    baseline=preflight['snapshot'];objects=objects_by_name(baseline['objects'])
                    reachable=sim.reachable_positions()['positions']
                    identities=identity_sets(baseline['objects'])
                    if not identities:
                        excluded={'scene':config['scene'],'reason':'no_natural_two_candidate_unique_anchor_structure',
                                  'clone_not_implemented_in_initial_run':True}
                    for identity in identities[:config['max_identity_sets']]:
                        a,b,x=(objects[n] for n in (*identity['candidate_names'],identity['anchor_name']))
                        receptacles=sorted((o for o in objects.values() if o.get('receptacle') and o['name'] not in (*identity['candidate_names'],identity['anchor_name'])
                            and not (o.get('openable') and not o.get('isOpen'))),key=lambda o:(min(distance(o['position'],p['position']) for p in (a,b)),o['name']))
                        points=[]
                        for obj in receptacles[:config['max_receptacles']]:
                            try:
                                for token in sim.spawn_points(obj['name']):
                                    points.append({'position':copy.deepcopy(token['selected_position']),
                                        'receptacle_name':token['receptacle_name'],'spawn':token})
                            except (TimeoutError,KeyboardInterrupt,SystemExit):raise
                            except Exception as error:
                                sim.store.save_json(f'spawn_failure_{sim.action_count:06d}.json',{'receptacle':obj['name'],'error':str(error)})
                        left,right=placement_groups(points,a,b,x,maximum_per_side=config['max_points_per_side'])
                        views=view_proposals(reachable,a,b,maximum_positions=config['max_view_positions'])
                        for proposal in construction_pairs(left,right):
                            for view in views:
                                sim.check_deadline()
                                if len(rows)>=config['max_attempts'] or all(summarize(rows,excluded,status,config)['passed_by_class'][k]>=config['scene_pairs_per_class'] for k in KINDS):
                                    status='completed_quota';return summarize(rows,excluded,status,config)
                                pair_id=f"{config['scene'].lower()}_{len(rows):05d}"
                                directory=Path(config['raw_evidence_dir'])/pair_id;directory.mkdir(exist_ok=False)
                                started=time.monotonic()
                                row={'pair_id':pair_id,'scene':config['scene'],'status':'rejected','discard_reasons':[]}
                                try:
                                    row=build_pair(sim,config,baseline,identity,proposal,view,pair_id,directory,reachable,attempt_row=row)
                                except (TimeoutError,KeyboardInterrupt,SystemExit):
                                    row.update(status='rejected',discard_reasons=['deadline_or_interrupt'])
                                    if 'pair_file' in row or 'pair_sha256' in row:
                                        row['incomplete_export']={key:row.pop(key) for key in ('pair_file','pair_sha256') if key in row}
                                    raise
                                except Exception as error:
                                    row.update(status='rejected',discard_reasons=[f'{type(error).__name__}: {error}'])
                                    if 'pair_file' in row or 'pair_sha256' in row:
                                        row['incomplete_export']={key:row.pop(key) for key in ('pair_file','pair_sha256') if key in row}
                                    traceback.print_exc()
                                finally:
                                    row['generation_seconds']=time.monotonic()-started
                                    raw=write_json(directory/'attempt.json',row)
                                    compact={key:row.get(key) for key in ('pair_id','scene','kind','status','discard_reasons','generation_seconds','pair_file','pair_sha256')}
                                    compact['raw_attempt']=raw
                                    rows.append(compact);handle.write(json.dumps(compact,sort_keys=True)+'\n');handle.flush()
                                    print('T01V2_ATTEMPT: '+json.dumps(compact,sort_keys=True),flush=True)
                                    connection.send(('progress',summarize(rows,excluded,status,config)))
        except BlackFrame as error:
            result=summarize(rows,excluded,'black_frame',config)
            result.update(black_frame_detected=True,black_frame_reference=error.reference,exit_code=86)
            return result
        except (TimeoutError,KeyboardInterrupt,SystemExit) as error:
            result=summarize(rows,excluded,'incomplete',config);result['error']=f'{type(error).__name__}: {error}';return result
        except Exception as error:
            traceback.print_exc();result=summarize(rows,excluded,'worker_failed',config);result['error']=f'{type(error).__name__}: {error}';return result
        finally:
            if sim is not None:
                sim.finish()
    return summarize(rows,excluded,status,config)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True);parser.add_argument('--scene',required=True)
    parser.add_argument('--shard-config',type=Path,required=True)
    parser.add_argument('--timeout-seconds',type=float,required=True)
    args=parser.parse_args()
    shard=json.loads(args.shard_config.read_text())
    config={'name':args.name,'scene':args.scene,'shard':shard,'seed':0,'resolution':255,
        'grid_size':.25,'visibility_distance':10,'field_of_view':shard.get('field_of_view',100),
        'settle_max_steps':50,'physics_time_step':.02,'physics_seconds_per_step':.1,
        'max_attempts':160,'max_identity_sets':8,'max_receptacles':4,'max_points_per_side':6,
        'max_view_positions':8,'scene_pairs_per_class':5,'max_raw_bytes':512*1024*1024,
        'raw_evidence_dir':shard['raw_evidence_dir'],
        'attempt_journal':f'data/world_pairs/v2/{args.name}_attempts.jsonl',
        'metrics_spec_sha256':sha(ROOT/'docs/METRICS_SPEC.md'),
        'review_policy':'four classes measured; final first ten/class selected by deterministic aggregation',
        'restore_policy':'state-level 1cm plus exact agent and both witnesses; native RGB difference is diagnostic',
        'instances_per_gpu_max':2,'proposal_order':'stable native identity/coordinate/view order; no random seed selection'}
    metrics={'kind':'pair_v2_shard','status':'failed_before_generation','acceptance_passed':False,'cleanup_errors':[],'black_frame_detected':False}
    worker=None;code=1;install_signals()
    try:
        if (shard['name']!=args.name or shard['scene']!=args.scene or shard['seed']!=0 or shard['resolution']!=255
                or shard['pbs_jobid']!=os.environ.get('PBS_JOBID')
                or shard['child_cuda_visible_devices']!=os.environ.get('CUDA_VISIBLE_DEVICES')
                or shard['parent_cuda_visible_devices']!=os.environ.get('T01_PARENT_CUDA_VISIBLE_DEVICES')):
            raise ValueError('Shard identity/environment differs from fixed manifest')
        cpus=shard['cpu_affinity']
        if len(set(cpus))!=12 or not set(cpus)<=os.sched_getaffinity(0):
            raise ValueError('Require twelve allocated CPU cores')
        os.sched_setaffinity(0,cpus)
        config['deadline']=deadline_after(args.timeout_seconds-10)
        config['runtime']=prepare_runtime(args.name)
        if config['runtime']['gpu_uuid']!=shard['gpu_identity']['gpu_uuid']:
            raise ValueError('Runtime UUID does not match gate')
        worker=Worker(generator_worker,config)
        while True:
            worker.receive()
            if worker.latest:metrics=copy.deepcopy(worker.latest)
            if worker.result is not None:
                metrics=worker.result;break
            if worker.error:
                metrics.update(status='worker_failed',error=worker.error);break
            if time.time()>=config['deadline']:
                raise TimeoutError('Shard parent deadline')
            time.sleep(.1)
        code=86 if metrics.get('black_frame_detected') else (0 if metrics.get('status') in ('completed_search','completed_quota') else 1)
    except BaseException as error:
        metrics.update(status='shard_failed',error=f'{type(error).__name__}: {error}');traceback.print_exc()
    finally:
        if worker:metrics.setdefault('cleanup_errors',[]).extend(worker.stop())
        if metrics.get('cleanup_errors'):code=1
        if metrics.get('black_frame_detected') and not metrics.get('cleanup_errors'):
            try:
                from src.tools.t01v2_black_evidence import portable_black_stop
                metrics['black_frame_stop']=portable_black_stop(metrics['black_frame_reference'],shard,ROOT)
                metrics['source_black_frame_detected']=metrics['black_frame_stop'].get('stage')=='source'
                metrics['capture_black_frame_detected']=not metrics['source_black_frame_detected']
                code=86
            except Exception as error:
                metrics['black_frame_evidence_error']=f'{type(error).__name__}: {error}'
                code=1
        journal=ROOT/config['attempt_journal']
        if journal.exists():metrics['attempt_journal_sha256']=sha(journal)
        write_result(args.name,metrics,config)
    return code


if __name__=='__main__':
    raise SystemExit(main())

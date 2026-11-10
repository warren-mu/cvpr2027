"""Identify existing training tables for the requested paired Kettle illustration."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import traceback

from experiments.fig01_export import file_ref, need, read_reference
from src.envs.pair_v2_configuration import from_pair
from src.envs.thor_common import ROOT
from src.tools.t01_partition import audited_result
from src.tools.t02_export import save_json


KITCHENS = {f'FloorPlan{i}' for i in range(1, 21)}
LOOK = {'look_flip', 'look_stay'}


def item(pair_ref, expected_scene, expected_kind):
    pair = read_reference(pair_ref)
    need((pair['scene'], pair['kind']) == (expected_scene, expected_kind), 'Pair identity changed')
    configuration = from_pair(pair)
    return {'pair': pair_ref, 'configuration': configuration['configuration_id'],
            'configuration_fields': configuration['key'], 'scene': expected_scene,
            'kind': expected_kind, 'anchor_type': pair['identities']['anchor_type'],
            'id': pair['pair_id'], 'labels_in_source_pair': pair.get('candidate_label_assignment')}


def joint_configurations(rows):
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row['kind'] in LOOK:
            groups[row['configuration']][row['kind']].append(row)
    return [dict(by_kind) for _, by_kind in sorted(groups.items()) if LOOK <= by_kind.keys()]


def inventory():
    e6, e6ref = audited_result('results/diag02_pool_options_01.json')
    need(e6['_meta']['exit_code'] == 0 and e6['metrics']['complete'], 'Missing complete pool inventory')
    scan = read_reference(e6['metrics']['report'])
    holdout = set(scan['fixed_holdout_scenes'])
    manifest_ref = file_ref(ROOT/'configs/t02_export_current_train.json')
    current = read_reference(manifest_ref)
    need(current['partition'] == 'train', 'Expected the training inventory')
    proposals = [item(r['pair'], r['scene'], r['kind']) for r in current['entries'] if r['scene'] in KITCHENS]
    certified = []
    for r in scan['rows_before_holdout']:
        if r['scene'] in KITCHENS or (r['kind'] in LOOK and r['scene'] not in holdout):
            certified.append({**item(r['pair_reference'], r['scene'], r['kind']), 'dataset_key': r['dataset_key']})
    tables = {}
    for result_path in sorted((ROOT/'results').glob('t02_export_*.json')):
        if any(tag in result_path.stem for tag in ('test', 'cache', '_gate')):
            continue
        data = json.loads(result_path.read_bytes())
        metrics = data.get('metrics', {})
        if metrics.get('kind') != 't02_table_export' or data['_meta'].get('exit_code') != 0:
            continue
        for r in metrics['pairs']:
            if r.get('scene') not in KITCHENS or r.get('status') != 'exported' or not r.get('table_complete'):
                continue
            table = read_reference(r['table_manifest'])
            need((table['scene'],table['kind'],table['pair_id']) == (r['scene'],r['kind'],r['pair_id']),
                 'Exported header identity differs')
            pair_ref = table['source']['t01']['pair']
            row = item(pair_ref, table['scene'], table['kind'])
            need(row['configuration_fields']['anchor_name'] == table['anchor_name'] and
                 row['configuration_fields']['candidate_names'] == sorted(table['candidate_names']),
                 'Exported table and original instance IDs differ')
            tables[(str(pair_ref['file']), pair_ref['sha256'])] = {
                **row, 'table': r['table_manifest'], 'export_source': file_ref(result_path)}
    actual = list(tables.values())
    train = [r for r in certified if r['scene'] not in holdout]
    def kitchen_kettles(rows):
        return [r for r in rows if r['scene'] in KITCHENS - holdout and r['anchor_type'] == 'Kettle']
    catalogs = {'certified_training_kitchen': [r for r in train if r['scene'] in KITCHENS],
                'kitchen_proposals_including_validation_scenes': proposals,
                'unique_kitchen_T02_exports_including_validation_scenes': actual}
    matches = {key: joint_configurations(kitchen_kettles(rows)) for key,rows in catalogs.items()}
    return {'catalogs': catalogs, 'exact_matching_configuration_sets': matches,
            'train_only_other_anchor_or_room_shared_configurations': joint_configurations(train),
            'summary': {key:{'look_pair_count': sum(r['kind'] in LOOK for r in rows),
                             'non_holdout_kettle_count_by_class': dict(Counter(r['kind'] for r in kitchen_kettles(rows))),
                             'exact_kitchen_kettle_configurations': len(matches[key])} for key,rows in catalogs.items()},
            'holdout_scenes': sorted(holdout), 'current_all_training_proposals': len(current['entries']),
            'e6_snapshot': scan['scan_started_at_utc'], 'e6_source': e6ref,
            'current_training_source': manifest_ref,
            'scope': 'Training inventory, training-named T02 single-export receipts and per-table metadata. No frozen test data, GPU, simulator or edits to any pair.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    need(re.fullmatch(r'diag_fig_kettle_inventory_[a-zA-Z0-9_]+', args.name), 'Fresh figure inventory name required')
    out = ROOT/'results'/(args.name+'.json')
    need(not out.exists(), 'Previous inventory result exists')
    metrics = {'complete': False, 'errors': [], 'gpus': 0}
    code = 1
    try:
        report = inventory()
        directory = ROOT/'logs/review'/args.name
        directory.mkdir(parents=True, exist_ok=False)
        metrics.update(complete=True, report=save_json(directory/'report.json',report), summary=report['summary'])
        code = 0
    except Exception as exc:
        traceback.print_exc()
        metrics['errors'].append(f'{type(exc).__name__}: {exc}')
    save_json(out, {'metrics': metrics, 'config': {'name': args.name, 'mode': 'read existing training sources'}})
    print('FINAL_METRICS_JSON: '+json.dumps(metrics, sort_keys=True), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())

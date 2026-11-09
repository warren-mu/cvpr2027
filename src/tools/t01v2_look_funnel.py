"""Supplement 12's requested Look funnel, from ended audited unit journals.

This reports recorded gates, not independent pair acceptance. Physics executes
first; absent later measurements remain unknown even though the requested
presentation puts physics fifth. No simulator, job or accounting operation.
"""
import argparse
import copy
import json
import re

from src.tools import merge_pair_v2 as merge
from src.tools.t01v2_posthoc_solvability import verified_passed_rows

STAGES = ('hidden_both', 'prefix_le3', 'first_evidence_ge250',
          'solvable_both', 'physical_both', 'witness_both')
LABELS = ('X 两边不可见', '前缀差 ≤3', '同一首步证据 ≥250', '两边可解', '物理检查', '见证回放')


def boolean(value):
    return value if type(value) is bool else None


def all_known(values):
    if any(value is False for value in values):
        return False
    return True if values and all(value is True for value in values) else None


def count(value, upper=255 * 255):
    return value if type(value) is int and 0 <= value <= upper else None


def compact_evidence(value):
    if not isinstance(value, dict):
        return None
    return {key: copy.deepcopy(value.get(key)) for key in (
        'evidence_pixels', 'marked_evidence_pixels', 'evidence_passed', 'marked_evidence_passed',
        'resolution', 'relevant_mask_union_pixels', 'anchor_pixels_w0', 'anchor_pixels_w1')}


def action_gate(evidence):
    if not isinstance(evidence, dict) or evidence.get('resolution') != [255, 255]:
        return None
    raw, marked = (count(evidence.get(key)) for key in ('evidence_pixels', 'marked_evidence_pixels'))
    if raw is None or marked is None:
        return None
    measured = raw >= 250 and marked >= 250
    for key, expected in (('evidence_passed', raw >= 250), ('marked_evidence_passed', marked >= 250)):
        if type(evidence.get(key)) is bool and evidence[key] is not expected:
            raise ValueError('Recorded evidence flag contradicts its actual pixel count')
    return measured


def measurements(row):
    prefix = row.get('prefix_measurement') or {}
    masks = prefix.get('prefix_edit_mask_pixels')
    pixels = ([count(item.get(world)) for item in masks for world in ('w0', 'w1')]
              if isinstance(masks, list) and masks and all(isinstance(item, dict) for item in masks) else [])
    hidden = (all(value == 0 for value in pixels) if pixels and all(value is not None for value in pixels) else None)
    maximum = count(prefix.get('prefix_max_abs_diff'), 255)
    report = row.get('look_solvability') or row.get('observation_search') or {}
    actions = []
    for action in report.get('actions') or []:
        actions.append({'action': action.get('action'), 'evidence': compact_evidence(action.get('evidence')),
            'worlds': [{key: copy.deepcopy(world.get(key)) for key in
                       ('world_index', 'status', 'anchor_pixels', 'anchor_visible', 'error')}
                       for world in action.get('worlds') or []]})
    results = [action_gate(action['evidence']) if len(action['worlds']) == 2
               and {world.get('world_index') for world in action['worlds']} == {0, 1}
               and all(world.get('status') == 'observed' for world in action['worlds']) else None
               for action in actions]
    if any(value is True for value in results):
        evidence_gate = True
    elif report.get('complete') is True:
        # An action unavailable in either world is not a common first action.
        # A common observed action with omitted pixels remains unknown.
        missing = any(result is None and not any(world.get('status') == 'not_in_domain'
                      for world in action['worlds']) for action, result in zip(actions, results))
        evidence_gate = None if missing else False
    else:
        evidence_gate = None
    own_solvable = [boolean(report.get('solvable_w0')), boolean(report.get('solvable_w1'))]
    solvable = all_known(own_solvable) if report.get('complete') is True else None
    physical = {}
    for world in ('physical_w0', 'physical_w1'):
        physical[world] = copy.deepcopy(row.get(world) or (row.get('physical_key_numbers') or {}).get(world) or {})
    physical_gate = all_known([boolean(value.get('valid_physical')) for value in physical.values()])
    witnesses = row.get('witnesses')
    witness_flags = ([boolean(value.get('passed')) for value in witnesses]
                     if isinstance(witnesses, list) and len(witnesses) == 2 else [])
    gates = dict(zip(STAGES, (hidden, None if maximum is None else maximum <= 3,
                            evidence_gate, solvable, physical_gate, all_known(witness_flags))))
    key_values = {'prefix_anchor_pixels': copy.deepcopy(masks), 'prefix_max_abs_diff': maximum,
        'candidate_prefix_visibility': boolean(prefix.get('candidates_visible_and_marked')),
        'first_actions': actions, 'catalog_complete': boolean(report.get('complete')),
        'solvable_by_world': own_solvable,
        'physical_by_world': {key: {name: copy.deepcopy(value.get(name)) for name in
            ('valid_physical', 'reasons', 'native_is_scene_at_rest', 'edits', 'max_other_position_change_m')}
            for key, value in physical.items()},
        'witnesses': [{key: copy.deepcopy(value.get(key)) for key in
            ('passed', 'reasons', 'executed_action_count')} for value in witnesses] if isinstance(witnesses, list) else None}
    return gates, key_values


def summarize(rows):
    """All rows belong to one provided FOV/mode group; never impute gates."""
    stages = {key: {'gate_pass': 0, 'gate_fail': 0, 'gate_unknown': 0,
                    'cumulative_pass': 0, 'cumulative_blocked': 0, 'cumulative_unknown': 0,
                    'rejected_examples': []} for key in STAGES}
    passed = 0
    for row in rows:
        gates, numbers = measurements(row)
        passed += int(row.get('status') == 'passed' and str(row.get('kind', '')).startswith('look_'))
        prefix = []
        for stage in STAGES:
            value, cell = gates[stage], stages[stage]
            earlier = all_known(prefix) if prefix else True
            cell['gate_pass' if value is True else 'gate_fail' if value is False else 'gate_unknown'] += 1
            prefix.append(value)
            cumulative = all_known(prefix)
            cell['cumulative_pass' if cumulative is True else 'cumulative_blocked' if cumulative is False
                 else 'cumulative_unknown'] += 1
            if value is False and row.get('status') == 'rejected' and len(cell['rejected_examples']) < 3:
                cell['rejected_examples'].append({'pair_id': row['pair_id'], 'scene': row['scene'],
                    'source_file': row.get('source_file'), 'raw_attempt': copy.deepcopy(row.get('raw_attempt')),
                    'earlier_requested_gates_passed': earlier, 'discard_reasons': copy.deepcopy(row.get('discard_reasons')),
                    'values': copy.deepcopy(numbers)})
    return {'observed_attempts': len(rows), 'source_reported_look_passed': passed, 'stages': stages,
            'example_selection': 'first up to three recorded rejected attempts with an observed failing gate; no padding'}


def load_unit(path, kitchen):
    from src.tools.t01v2_look_unit_review_checkpoint import source_support
    with source_support():
        data, reference = merge.load_source(path)
    cfg, meta, metrics = data['config'], data['_meta'], data['metrics']
    if (cfg.get('producer_module') != 'src.envs.pair_v2_look_unit_shard'
            or cfg.get('look_unit_kitchen_scene') != kitchen
            or cfg.get('scene') not in ('FloorPlan203', 'FloorPlan204', kitchen)
            or cfg.get('field_of_view') not in (60, 100)
            or not meta.get('end') or type(meta.get('exit_code')) is not int):
        raise ValueError('Require an ended explicit supplement 12 Look unit from the fixed six cells')
    _, journal_ref, _ = verified_passed_rows(data)
    unit = {'scene': cfg['scene'], 'field_of_view': cfg['field_of_view'], 'source': reference,
        'status': metrics.get('status'), 'exit_code': meta['exit_code'], 'journal': journal_ref,
        'scene_excluded': copy.deepcopy(metrics.get('scene_excluded')),
        'native_placement_failures_not_counted_as_pair_attempts': metrics.get('placement_exhausted_proposals'),
        'records': None}
    if journal_ref is None:
        return unit
    rows = []
    path = merge.reference_path(journal_ref, within=merge.ROOT / 'data/world_pairs/v2')
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        journal = json.loads(line)
        raw = merge.read_reference(journal['raw_attempt'], within=cfg['raw_evidence_dir'])
        if raw.get('proposed_pool', journal.get('proposed_pool')) != 'look':
            raise ValueError('Look-only unit journal contains a non-Look proposal')
        row = {**raw, 'source_file': reference['file'], 'raw_attempt': journal['raw_attempt']}
        for world in ('w0', 'w1'):
            reported = journal.get(world + '_valid_physical')
            actual = (raw.get('physical_' + world) or (raw.get('physical_key_numbers') or {}).get('physical_' + world) or {}).get('valid_physical')
            if type(reported) is bool and reported is not actual:
                raise ValueError('Journal physics differs from its authenticated attempt')
        rows.append(row)
    unit['records'] = rows
    return unit


def aggregate(units, kitchen):
    groups = []
    for fov in (100, 60):
        selected = [unit for unit in units if unit['field_of_view'] == fov]
        known = [unit for unit in selected if unit['records'] is not None]
        for mode in ('flip', 'stay'):
            rows = [row for unit in known for row in unit['records'] if row.get('mode') == mode]
            groups.append({'field_of_view': fov, 'mode': mode,
                'provided_sources': [unit['source']['file'] for unit in selected],
                'unobserved_journal_sources': [unit['source']['file'] for unit in selected if unit['records'] is None],
                'known_records_with_unknown_mode': sum(row.get('mode') not in ('flip', 'stay') for unit in known for row in unit['records']),
                'funnel': summarize(rows) if known else None})
    expected = [(scene, fov) for scene in ('FloorPlan203', 'FloorPlan204', kitchen) for fov in (100, 60)]
    observed = {(unit['scene'], unit['field_of_view']) for unit in units}
    return {'groups': groups, 'unprovided_units': [{'scene': scene, 'field_of_view': fov}
            for scene, fov in expected if (scene, fov) not in observed],
        'units': [{key: copy.deepcopy(value) for key, value in unit.items() if key != 'records'} for unit in units],
        'scope': 'source-recorded gate diagnostics; no independent pair acceptance or permission to expand',
        'execution_order': 'physics executes before prefix/first-action/witness; requested funnel order is presentation only',
        'missing_policy': 'unmeasured gates are unknown; cumulative pass requires every preceding gate observed true',
        'evidence_rule': 'raw and marked evidence each >=250 at255x255 in the same common first action; no combining maxima across actions',
        'counts_cover_only_counted_pair_attempts': True, 'acceptance_passed': None}


def markdown(metrics):
    def value(item):
        return '?' if item is None else str(item)
    def sample(example, stage):
        numbers = example['values']
        masks = numbers['prefix_anchor_pixels']
        mask_text = ('/'.join(value(masks[0].get(key)) for key in ('w0', 'w1'))
                     if isinstance(masks, list) and len(masks) == 1 else '?')
        observed = [item for item in numbers['first_actions'] if item['evidence'] is not None
                    and count(item['evidence'].get('evidence_pixels')) is not None
                    and len(item['worlds']) == 2 and all(world.get('status') == 'observed' for world in item['worlds'])]
        best = max(observed, key=lambda item: item['evidence']['evidence_pixels']) if observed else None
        evidence = ('?' if best is None else f"{best['action']}:{best['evidence']['evidence_pixels']}/"
                    f"{value(best['evidence'].get('marked_evidence_pixels'))}")
        text = f"{example['pair_id']}: X={mask_text}, Δ={value(numbers['prefix_max_abs_diff'])}, e={evidence}"
        if stage == 'solvable_both':
            text += ', S=' + '/'.join(map(value, numbers['solvable_by_world']))
        if stage == 'physical_both':
            text += ', P=' + '/'.join(value(numbers['physical_by_world'][key]['valid_physical'])
                                      for key in ('physical_w0', 'physical_w1'))
        if stage == 'witness_both':
            text += ', W=' + ('?' if numbers['witnesses'] is None else '/'.join(value(item['passed']) for item in numbers['witnesses']))
        return text.replace('|', '/')
    paragraph = ('仅汇总已结束、已审计来源的记录，不代表独立验收。物理检查实际先执行；未测的图像或后续关卡保留未知。'
                 '连续通过要求本关及前关均实测通过；样例是每关最先出现的至多三个实际失败，可能有未知前关。'
                 'X 为两世界 v0 锚点像素，Δ 为前缀最大差，e 为同一共同首步的原图/标号图证据像素（展示原图证据最多的动作）；? 为未测。')
    if metrics['unprovided_units']:
        paragraph += ' 尚未提供：' + '、'.join(f"{unit['scene']}/{unit['field_of_view']}°" for unit in metrics['unprovided_units']) + '。'
    if metrics.get('source_errors'):
        paragraph += ' 来源读取有错误，详见结果 JSON；表中只含成功核对的来源。'
    lines = [paragraph, '',
        '| FOV | 模式 | 关卡 | 连续通过 | 累计未知 | 本关通过/失败/未知 | 拒绝样例关键值 |',
        '|---|---|---|---:|---:|---|---|']
    for group in metrics['groups']:
        if not group['provided_sources']:
            continue
        funnel = group['funnel']
        for stage, label in zip(STAGES, LABELS):
            row = None if funnel is None else funnel['stages'][stage]
            values = ('未观测',) * 4 if row is None else (row['cumulative_pass'], row['cumulative_unknown'],
                f"{row['gate_pass']}/{row['gate_fail']}/{row['gate_unknown']}",
                '<br>'.join(sample(example, stage) for example in row['rejected_examples']) or '无已观测失败样例')
            lines.append(f"| {group['field_of_view']} | {group['mode']} | {label} | " + ' | '.join(map(str, values)) + ' |')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--source', nargs='+', required=True)
    parser.add_argument('--kitchen', required=True)
    args = parser.parse_args(argv)
    if re.fullmatch(r'diag_[A-Za-z0-9_]+', args.name) is None or args.kitchen not in {f'FloorPlan{i}' for i in range(1, 21)}:
        parser.error('Use a unique diag_* name and the already selected training kitchen')
    output = merge.ROOT / 'results' / (args.name + '.json')
    report = merge.ROOT / 'logs/review' / (args.name + '.md')
    if any(path.exists() for path in (output, report)):
        raise FileExistsError('Preserve previous reports; choose a new name')
    units, errors, seen = [], [], set()
    for source in args.source:
        try:
            path = (merge.ROOT / source).resolve()
            if path in seen:
                raise ValueError('Duplicate explicit source')
            seen.add(path)
            units.append(load_unit(path, args.kitchen))
        except Exception as error:
            errors.append({'source_file': source, 'error': f'{type(error).__name__}: {error}'})
    metrics = {'kind': 't01v2_look_funnel', **aggregate(units, args.kitchen), 'source_errors': errors,
               'status': 'reported' if not errors else 'incomplete_source_evidence'}
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open('x') as stream:
        stream.write(markdown(metrics))
    metrics['report'] = {'file': str(report.relative_to(merge.ROOT)), 'sha256': merge.sha_file(report)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump({'metrics': metrics, 'config': {'name': args.name, 'explicit_sources': args.source, 'selected_kitchen': args.kitchen},
                   'sources': [unit['source'] for unit in units]}, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('FINAL_METRICS_JSON: ' + json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())

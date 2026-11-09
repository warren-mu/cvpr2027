"""Read-only CPU diagnosis of the frozen first-round A/B construction logs.

No simulator, acceptance decision, new threshold or source mutation. Transported
inputs preserve original paths under INPUT/{repo,abs}; every used reference is
SHA-checked. Missing retained geometry stays unknown, never reconstructed.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath
import statistics
import traceback

KINDS = ('look_stay', 'direct_flip')
FUNNEL = 'logs/review/gen_test_round2_prepare_01_retry3_funnel/funnel.json'


def digest(body):
    return hashlib.sha256(body).hexdigest()


class Inputs:
    def __init__(self, root):
        self.root = root.resolve()
        self.support = {}
        self.missing = {}
        self.identities = {}

    def path(self, filename):
        part = 'abs' + filename if filename.startswith('/') else 'repo/' + filename
        p = (self.root / part).resolve()
        p.relative_to(self.root)
        return p

    def read(self, ref, optional=False):
        assert isinstance(ref, dict) and isinstance(ref.get('file'), str)
        assert re.fullmatch(r'[0-9a-f]{64}', ref.get('sha256', '')), 'Invalid original SHA reference'
        old_identity = self.identities.setdefault(ref['file'], ref['sha256'])
        assert old_identity == ref['sha256'], 'Conflicting original source identities'
        if optional:
            old = self.support.setdefault(ref['file'], ref)
            assert old['sha256'] == ref['sha256'], 'Conflicting source identities'
        path = self.path(ref['file'])
        if not path.is_file() and optional:
            self.missing[ref['file']] = ref
            return None
        body = path.read_bytes()
        assert digest(body) == ref['sha256'], 'Original SHA differs: ' + ref['file']
        if 'bytes' in ref:
            assert len(body) == ref['bytes'], 'Original byte count differs'
        return json.loads(body)


def physical(row, world):
    return row.get('physical_' + world) or (row.get('physical_key_numbers') or {}).get('physical_' + world) or {}


def physical_gate(row):
    flags = [physical(row, w).get('valid_physical') for w in ('w0', 'w1')]
    assert all(value is None or type(value) is bool for value in flags), 'Physical flag is not a measured bool/null'
    return False if any(value is False for value in flags) else True if all(value is True for value in flags) else None


def objects(snapshot):
    if not isinstance(snapshot, dict):
        return {}
    rows = snapshot.get('objects', [])
    assert isinstance(rows, list) and all(isinstance(o, dict) and isinstance(o.get('name'), str) for o in rows)
    result = {o['name']: o for o in rows}
    assert len(result) == len(rows), 'Duplicate baseline object identity'
    return result


def point(value):
    if not isinstance(value, dict):
        return None
    values = [value.get(a) for a in 'xyz']
    return values if all(type(v) in (int, float) and math.isfinite(v) for v in values) else None


def distance(a, b):
    a, b = point(a), point(b)
    return math.dist(a, b) if a is not None and b is not None else None


def world_details(row, world, inputs, source_config=None, source_metrics=None):
    p = physical(row, world)
    edits = p.get('edits') or {}
    anchors = list(edits)
    assert len(anchors) <= 1, 'This diagnostic requires the original single anchor'
    anchor = anchors[0] if anchors else (row.get('identities') or {}).get('anchor_name')
    common = p.get('common_baseline_check') or {}
    compact = p.get('common_baseline_key_numbers') or {}
    measurements = (common.get('measurements') or {}).get('objects') or {}
    baseline_ref = common.get('baseline_reference') or compact.get('baseline_reference')
    snapshot = inputs.read(baseline_ref, optional=True) if baseline_ref else None
    baseline = {}
    controller_verified = False
    construction_root = None
    association_missing = []
    if baseline_ref and snapshot is not None:
        reference_path = PurePosixPath(baseline_ref['file'])
        assert reference_path.is_absolute() and '..' not in reference_path.parts, 'Noncanonical baseline path'
        raw_root = (source_config or {}).get('raw_evidence_dir')
        if not raw_root:
            association_missing.append('source raw_evidence_dir absent')
        else:
            relative = reference_path.relative_to(PurePosixPath(raw_root))
            assert (len(relative.parts) >= 3 and re.fullmatch(r'marked_controller_[0-9]{3}', relative.parts[0])
                    and relative.parts[1] == 'construction'), 'Baseline is outside its source Controller construction root'
            controller = relative.parts[0]
            if row.get('search_controller_id') is not None:
                assert row['search_controller_id'] == controller, 'Baseline belongs to another Controller'
            visits = [visit for visit in (source_metrics or {}).get('controller_visits', [])
                      if visit.get('controller_id') == controller]
            assert len(visits) <= 1, 'Duplicate source Controller visit'
            if not visits:
                association_missing.append('matching source Controller visit absent')
            else:
                visit = visits[0]
                if any(value is None for value in (snapshot.get('scene'), row.get('scene'), (source_config or {}).get('scene'))):
                    association_missing.append('scene association absent')
                else:
                    assert snapshot['scene'] == row['scene'] == source_config['scene'], 'Baseline belongs to another scene'
                pids = (visit.get('worker_pid'), (visit.get('unity_identity') or {}).get('pid'),
                        snapshot.get('worker_pid'), snapshot.get('unity_pid'))
                if any(type(value) is not int for value in pids):
                    association_missing.append('Controller process identity absent')
                else:
                    assert pids[0] == pids[2] and pids[1] == pids[3], 'Baseline belongs to another Controller process'
                if not association_missing:
                    construction_root = PurePosixPath(raw_root) / controller / 'construction'
                    baseline = objects(snapshot)
                    controller_verified = True
    edits_set = set(p.get('edited_names') or anchors)
    local = dict(p.get('other_position_changes_above_1cm') or {})
    for name, change in (p.get('all_object_changes') or {}).items():
        delta = change.get('position_change_m')
        if name not in edits_set and type(delta) in (int, float) and delta > .01:
            local[name] = delta
    common_violations = dict(compact.get('violations') or {})
    for name, value in measurements.items():
        delta = value.get('baseline_displacement_m')
        if name not in edits_set and type(delta) in (int, float) and delta > .01:
            common_violations[name] = delta
    assert all(type(v) in (int, float) and math.isfinite(v) and v > .01 for v in [*local.values(), *common_violations.values()])
    ledger = ((row.get('native_retry_records') or {}).get(world) or {}).get(anchor, {})
    request = (ledger.get('arguments') or {}).get('position')
    if request is not None:
        if ledger.get('object_name') is None or ledger.get('scene') is None or ledger.get('status') is None:
            association_missing.append('successful native placement identity absent')
            request = None
        else:
            assert ledger['object_name'] == anchor and ledger['scene'] == row['scene'], 'Requested placement identity differs'
            if ledger['status'] != 'placed':
                request = None
        action_ref = ledger.get('action_reference')
        if not construction_root or not action_ref:
            association_missing.append('successful native placement Controller reference unavailable')
            request = None
        else:
            action_path = PurePosixPath(action_ref['file'])
            assert action_path.is_absolute() and '..' not in action_path.parts, 'Noncanonical placement action path'
            action_path.relative_to(construction_root)
            assert re.fullmatch(r'[0-9a-f]{64}', action_ref.get('sha256', '')), 'Invalid native placement SHA reference'
    anchor_current = (measurements.get(anchor) or {}).get('position')
    other_rows = []
    for name in sorted(set(local) | set(common_violations)):
        obj = baseline.get(name, {})
        actual = distance((measurements.get(name) or {}).get('position'), anchor_current)
        proposed = distance(obj.get('position'), request)
        other_rows.append({'object_name': name, 'object_type': obj.get('objectType'),
            'local_displacement_m': local.get(name), 'common_baseline_displacement_m': common_violations.get(name),
            'post_settling_center_distance_to_anchor_m': actual,
            'baseline_object_to_requested_anchor_distance_m': proposed,
            'type_source': baseline_ref if obj.get('objectType') else None,
            'same_controller_baseline_verified': controller_verified,
            'actual_distance_source': 'same-world common-baseline measurements' if actual is not None else None,
            'requested_distance_scope': 'same-controller baseline object versus logged requested placement; NOT post-settling distance' if proposed is not None else None})
    errors = [e.get('position_error_m') for e in edits.values()]
    finite = lambda value: type(value) in (int, float) and math.isfinite(value) and value >= 0
    anchor_over = True if any(finite(v) and v > .02 for v in errors) else (
        False if anchor in edits and bool(errors) and all(finite(v) for v in errors) else None)
    changes = p.get('all_object_changes')
    local_known = (isinstance(changes, dict) and bool(changes)
        and all(finite(change.get('position_change_m')) for name, change in changes.items() if name not in edits_set))
    local_known = local_known or finite(p.get('max_other_position_change_m'))
    common_known = (bool(measurements) and (common.get('measurements') or {}).get('inventory_exact') is True
        and all(finite(value.get('baseline_displacement_m')) for name, value in measurements.items() if name not in edits_set))
    common_known = common_known or finite(compact.get('max_other_displacement_m'))
    other_over = True if other_rows else False if local_known and common_known else None
    return {'world': world, 'anchor_name': anchor, 'anchor_position_errors_m': errors,
        'anchor_above_2cm': anchor_over, 'other_above_1cm': other_over,
        'other_objects': other_rows, 'valid_physical': p.get('valid_physical'),
        'recorded_reasons': p.get('reasons') or [], 'physics_record_present': bool(p),
        'metadata_association_missing': association_missing}


def native_calls(row):
    ledgers = [v for world in (row.get('native_retry_records') or {}).values() for v in world.values()]
    if row.get('native_retry_failure'):
        ledgers.append(row['native_retry_failure'])
    seen, identities = {}, {}
    for ledger in ledgers:
        for a in ledger.get('attempts', []):
            ref = a.get('action_reference')
            if not ref:
                # Keep records without an action receipt separately, not as a known call.
                continue
            assert isinstance(ref.get('file'), str) and re.fullmatch(r'[0-9a-f]{64}', ref.get('sha256', ''))
            old = identities.setdefault(ref['file'], ref['sha256'])
            assert old == ref['sha256'], 'Native call reference has conflicting SHA identities'
            key = (ref['file'], ref['sha256'])
            item = {k: a.get(k) for k in ('native_success', 'error', 'pose_read_error', 'action_reference')}
            if key in seen:
                assert seen[key] == item, 'Repeated native call disagrees'
            seen[key] = item
    return seen


def summary_distribution(values):
    return {'observed': len(values), 'min_m': min(values) if values else None,
        'median_m': statistics.median(values) if values else None, 'max_m': max(values) if values else None}


def diagnose(inputs, funnel_result_sha256):
    parent_path = 'results/gen_test_round2_prepare_01_retry3_funnel.json'
    assert re.fullmatch(r'[0-9a-f]{64}', funnel_result_sha256), 'Require the frozen original funnel-result SHA'
    parent_ref = {'file': parent_path, 'sha256': funnel_result_sha256}
    parent = inputs.read(parent_ref)
    assert parent['_meta']['exit_code'] == 0 and parent['_meta'].get('end')
    assert parent['_meta']['name'] == parent['config']['name'] == Path(parent_path).stem
    assert digest(json.dumps(parent['metrics'], sort_keys=True).encode()) == parent['_meta']['metrics_sha256']
    parent_ref['metrics_sha256'] = parent['_meta']['metrics_sha256']
    # The frozen report is named by the completed producer, not selected from results.
    def find_ref(value):
        if isinstance(value, dict):
            if value.get('file') == FUNNEL and 'sha256' in value:
                return value
            for v in value.values():
                found = find_ref(v)
                if found:
                    return found
        elif isinstance(value, list):
            for v in value:
                found = find_ref(v)
                if found:
                    return found
        return None
    ref = find_ref(parent)
    assert ref, 'Completed original report reference is missing'
    funnel = inputs.read(ref)
    assert not funnel['errors']
    groups = defaultdict(list)
    uncounted = defaultdict(list)
    expected_groups = {}
    for group in funnel['groups']:
        key = (group['batch'], group['actual_proposed_class'])
        if key[1] in KINDS:
            assert key not in expected_groups, 'Duplicate original funnel group'
            expected_groups[key] = group
            groups[key]  # Preserve observed zero groups and uncounted-only sources.
    sources = [parent_ref]
    seen_sources = set()
    for unit in funnel['sources']:
        assert unit['source']['file'] not in seen_sources, 'Repeated scene-source denominator'
        seen_sources.add(unit['source']['file'])
        source = inputs.read(unit['source']); sources.append(unit['source'])
        assert source['_meta']['end'] and unit['scan_complete']
        assert source['config']['scene'] == unit['scene']
        metrics = source['metrics']
        assert digest(json.dumps(metrics, sort_keys=True).encode()) == source['_meta']['metrics_sha256'] == unit['source']['metrics_sha256']
        mode_counts = Counter()
        body = b''
        if not unit['journal']:
            assert metrics['attempts'] == 0
        else:
            journal_path = inputs.path(unit['journal']['file'])
            body = journal_path.read_bytes()
            assert digest(body) == unit['journal']['sha256'] and (not body or body.endswith(b'\n'))
        seen_pairs = set()
        for line in body.decode().splitlines():
            if not line.strip():
                continue
            journal = json.loads(line); row = inputs.read(journal['raw_attempt'])
            assert row['scene'] == unit['scene'] and row['pair_id'] not in seen_pairs
            seen_pairs.add(row['pair_id'])
            assert row.get('counts_as_pair_attempt') is True
            assert all(row.get(k) == journal.get(k) for k in ('pair_id', 'scene', 'kind', 'status', 'discard_reasons', 'proposed_pool'))
            kind = row['proposed_pool'] + '_' + row['mode']; mode_counts[kind] += 1
            if kind not in KINDS:
                continue
            worlds = [world_details(row, w, inputs, source['config'], metrics) for w in ('w0', 'w1')]
            groups[(unit['batch'], kind)].append({'pair_id': row['pair_id'], 'scene': row['scene'],
                'source': unit['source']['file'], 'raw_attempt': journal['raw_attempt'],
                'physical_gate': physical_gate(row), 'worlds': worlds,
                'native_calls': list(native_calls(row).values())})
        assert sum(mode_counts.values()) == metrics['attempts']
        assert dict(mode_counts) == {k: v for k, v in (unit['observed_by_proposed_pool_mode'] or {}).items() if v}
        seen_uncounted = set()
        for entry in metrics.get('placement_point_failures', []):
            raw_ref = entry.get('raw_attempt')
            assert raw_ref, 'Original uncounted placement entry lacks raw reference'
            if raw_ref['file'] in seen_uncounted:
                continue
            seen_uncounted.add(raw_ref['file'])
            row = inputs.read(raw_ref)
            assert row.get('counts_as_pair_attempt') is False and row.get('scene') == unit['scene']
            kind = str(row.get('proposed_pool')) + '_' + str(row.get('mode'))
            if kind in KINDS:
                uncounted[(unit['batch'], kind)].append({'raw_attempt': raw_ref,
                    'native_calls': list(native_calls(row).values()), 'scene': unit['scene']})
    output = []
    for (batch, kind), rows in sorted(groups.items()):
        expected = expected_groups[(batch, kind)]
        assert len(rows) == expected['observed_attempts']
        failed = [r for r in rows if r['physical_gate'] is False]
        flags = Counter({key: 0 for key in ('anchor_above_2cm', 'other_above_1cm', 'both_anchor_and_other', 'neither_of_these_measured_displacements')})
        unknown = Counter({'anchor_displacement_unknown': 0, 'other_displacement_unknown': 0})
        unresolved_without_excess = 0
        any_measured = lambda values: True if any(v is True for v in values) else None if any(v is None for v in values) else False
        for r in failed:
            a = any_measured([w['anchor_above_2cm'] for w in r['worlds']])
            o = any_measured([w['other_above_1cm'] for w in r['worlds']])
            flags['anchor_above_2cm'] += a is True; flags['other_above_1cm'] += o is True
            flags['both_anchor_and_other'] += a is True and o is True
            flags['neither_of_these_measured_displacements'] += a is False and o is False
            unknown['anchor_displacement_unknown'] += a is None
            unknown['other_displacement_unknown'] += o is None
            unresolved_without_excess += a is not True and o is not True and (a is None or o is None)
        types = defaultdict(list)
        for r in failed:
            for world in r['worlds']:
                for other in world['other_objects']:
                    types[other['object_type'] or 'UNKNOWN'].append((r, world, other))
        type_rows = []
        for typ, events in types.items():
            type_rows.append({'object_type': typ, 'world_object_events': len(events),
                'distinct_counted_proposals': len({e[0]['raw_attempt']['file'] for e in events}),
                'post_settling_center_distance': summary_distribution([e[2]['post_settling_center_distance_to_anchor_m'] for e in events if e[2]['post_settling_center_distance_to_anchor_m'] is not None]),
                'requested_anchor_to_baseline_object_distance': summary_distribution([e[2]['baseline_object_to_requested_anchor_distance_m'] for e in events if e[2]['baseline_object_to_requested_anchor_distance_m'] is not None])})
        calls, call_identities = {}, {}
        for row in rows + uncounted[(batch, kind)]:
            for call in row['native_calls']:
                key = (call['action_reference']['file'], call['action_reference']['sha256'])
                old = call_identities.setdefault(key[0], key[1])
                assert old == key[1], 'Native call SHA conflicts across original proposals'
                if key in calls:
                    assert calls[key] == call
                calls[key] = call
        call_status = Counter('failed' if a['native_success'] is False else 'passed' if a['native_success'] is True else 'unknown' for a in calls.values())
        output.append({'batch': batch, 'proposal_class': kind, 'counted_proposals': len(rows),
            'physical_failures': len(failed), 'physical_unknown': sum(r['physical_gate'] is None for r in rows),
            'failure_subtypes_nonexclusive': dict(flags), 'failure_subtypes_unknown': dict(unknown),
            'physical_failures_without_positive_excess_and_with_unknown_measurement': unresolved_without_excess,
            'subtype_fraction_of_physical_failures': {k: v / len(failed) for k, v in flags.items()} if failed else {},
            'other_object_types': sorted(type_rows, key=lambda r: (-r['distinct_counted_proposals'], r['object_type'])),
            'uncounted_placement_proposals': len(uncounted[(batch, kind)]), 'native_call_status': dict(call_status),
            'counted_proposals_without_native_call_records': sum(not r['native_calls'] for r in rows),
            'counted_proposals_with_native_call_records': sum(bool(r['native_calls']) for r in rows),
            'counted_proposals_with_failed_native_call': sum(any(a['native_success'] is False for a in r['native_calls']) for r in rows),
            'failed_physics_rows': failed,
            'remaining_physics_reasons': dict(Counter(reason for r in failed if not any(w['anchor_above_2cm'] or w['other_above_1cm'] for w in r['worlds']) for w in r['worlds'] for reason in w['recorded_reasons'])),
            'uncounted_placement_rows': uncounted[(batch, kind)]})
    return {'status': 'complete', 'groups': output, 'support_references': list(inputs.support.values()),
        'missing_support_references': list(inputs.missing.values()), 'source_results': sources,
        'scope': 'Source-recorded physical failures; CPU only; no new acceptance or causal collision claim.',
        'native_call_scope': 'Statuses are original SHA-bound retry-ledger observations keyed by action reference; referenced action payloads are not revalidated here. Proposals without referenced calls are reported separately. Ledger attempts without action_reference are not counted as known calls; native_call_status is not an exhaustive count when such evidence is absent.',
        'denominators': 'Physical subtypes overlap and use physical-rejected counted proposals. Native failed calls include successful retries and uncounted exhausted proposals; they are NOT added to the pair-attempt denominator. Missing displacement measurements stay unknown and never enter the measured-neither subtype.',
        'distance_scope': 'Actual distance only from both same-world realized positions. Requested distance is a separate geometric diagnostic, using the exact referenced same-controller baseline and native placement argument. Missing geometry stays null.'}


def markdown_report(report):
    if report.get('status') != 'complete':
        return '# 已有物理记录诊断\n\n诊断未完成，错误原文见 physics.json；未补齐任何数字。\n'
    lines = ['# 已有物理记录诊断', '',
        '仅解释原日志，不重新验收、不推断碰撞因果。物理细分的分母为原物理检查失败的计次提议；锚点和其他物体超限可以重叠。未测量的 w1 或缺失字段保留未知。原生调用按原 action 引用去重，包含不计次的放置失败，与提议分母分开。', '',
        '| 批次/提议 | 计次提议 | 物理失败 | 锚点 >2cm | 其他 >1cm | 二者重叠 | 已测其余 / 尚未知 | 原生失败调用 / 有引用调用 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for group in report['groups']:
        flags = group['failure_subtypes_nonexclusive']; calls = group['native_call_status']
        lines.append('| {batch}/{kind} | {attempts} | {failed} | {anchor} | {other} | {both} | {neither} / {unknown} | {native_failed} / {native_total} |'.format(
            batch=group['batch'], kind=group['proposal_class'], attempts=group['counted_proposals'], failed=group['physical_failures'],
            anchor=flags['anchor_above_2cm'], other=flags['other_above_1cm'], both=flags['both_anchor_and_other'],
            neither=flags['neither_of_these_measured_displacements'], unknown=group['physical_failures_without_positive_excess_and_with_unknown_measurement'],
            native_failed=calls.get('failed', 0), native_total=sum(calls.values())))
    lines += ['', '“已测其余”仅指两项位移均已测量且未超限，仍可能因其他原物理原因失败；“尚未知”指无已观测超限且至少一项未知。原生调用的状态未独立重验 action 文件。', '',
        '| 批次/提议 | 常见其他物体类型 | world/object 事件 | 实际落定距离可用事件 | 请求位置距离可用事件 |',
        '|---|---|---:|---:|---:|']
    for group in report['groups']:
        for row in group['other_object_types'][:5]:
            lines.append('| {} / {} | {} | {} | {} | {} |'.format(group['batch'], group['proposal_class'], row['object_type'],
                row['world_object_events'], row['post_settling_center_distance']['observed'], row['requested_anchor_to_baseline_object_distance']['observed']))
    lines += ['', '同一世界同一物体跨两个基准只计一次；w0/w1 是不同物理观察。实际距离只使用同一世界内两个原落定位姿；请求位置距离另列，不能替代实际距离。类型和请求距离只使用已绑定原场景、Controller 路径及 PID 的快照。缺失快照原引用见 physics.json。', '']
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--name', required=True); ap.add_argument('--input-root', type=Path, required=True)
    ap.add_argument('--funnel-result-sha256', required=True)
    args = ap.parse_args(); inputs = Inputs(args.input_root)
    code = 0
    try:
        report = diagnose(inputs, args.funnel_result_sha256)
    except Exception:
        code = 1; report = {'status': 'failed', 'error': traceback.format_exc()}
    root = Path(__file__).resolve().parents[1]
    out = root / 'logs/review' / args.name; out.mkdir(parents=True, exist_ok=False)
    body = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n'
    (out / 'physics.json').write_text(body)
    markdown = markdown_report(report)
    (out / 'report.md').write_text(markdown)
    metrics = {'status': report['status'], 'report': {'file': str((out / 'physics.json').relative_to(root)), 'sha256': digest(body.encode())},
        'markdown_report': {'file': str((out / 'report.md').relative_to(root)), 'sha256': digest(markdown.encode())},
        'groups': [{k: v for k, v in g.items() if k not in ('failed_physics_rows', 'uncounted_placement_rows')} for g in report.get('groups', [])]}
    result = {'metrics': metrics, 'config': {'name': args.name, 'input_root': str(args.input_root), 'gpu_used': False, 'frozen_original_funnel': FUNNEL, 'funnel_result_sha256': args.funnel_result_sha256},
        'sources': report.get('source_results', [])}
    (root / 'results').mkdir(exist_ok=True)
    (root / 'results' / (args.name + '.json')).write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print('FINAL_METRICS_JSON: ' + json.dumps(metrics, sort_keys=True), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())

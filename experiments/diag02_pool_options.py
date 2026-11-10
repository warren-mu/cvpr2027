"""DIAG02 E6: read-only alternatives from original training/cache receipts.

No pool is frozen and no acceptance rule is changed. The alternatives explicitly
omit the Stay displacement filter; the second applies the original seed0 A/B
balancer without an additional ratio or left-position selection.
"""
import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
KINDS = ('direct_flip', 'direct_stay', 'look_flip', 'look_stay')
VERSION = 'diag02_pool_alternatives_metadata_v1'
BASE = {'file': 'results/gen_train_look_inventory_first_unit_01.json',
        'sha256': 'cc77a010e7eef375a942354e8e89d5403f681b72a77fc44a53506aa41a5dff01'}


def need(value, message):
    if not value:
        raise ValueError(message)


def file_ref(path, root=ROOT):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {'file': str(path.relative_to(Path(root).resolve())),
            'sha256': hashlib.sha256(raw).hexdigest()}


def combine_rows(original, additional):
    """Reuse exact identities; conflicting or duplicate-proposal variants fail."""
    by_key = {}
    for row in [*original, *additional]:
        key = row['dataset_key']
        if key in by_key:
            old = by_key[key]
            fields = ('pair_reference', 'scene', 'kind', 'candidate_names',
                      'target_names', 's0_target_left', 'proposal_identity')
            need(all(old[k] == row[k] for k in fields), 'Conflicting repeated cache identity: ' + key)
            need(old.get('stay_displacement') == row.get('stay_displacement'),
                 'Conflicting repeated Stay metadata: ' + key)
        else:
            by_key[key] = copy.deepcopy(row)
    rows = [by_key[k] for k in sorted(by_key)]
    need(len({r['proposal_identity'] for r in rows}) == len(rows),
         'Distinct dataset keys duplicate one proposal; do not double count')
    return rows


def summary(rows):
    kinds = {}
    for kind in KINDS:
        selected = [r for r in rows if r['kind'] == kind]
        scenes = sorted({r['scene'] for r in selected})
        kinds[kind] = {'pair_count': len(selected), 'scene_count': len(scenes),
                       'scenes': scenes, 'dataset_keys': sorted(r['dataset_key'] for r in selected)}
    flip, stay = kinds['look_flip']['pair_count'], kinds['look_stay']['pair_count']
    scenes = sorted({r['scene'] for r in rows})
    return {'pair_count': len(rows), 'scene_count': len(scenes), 'scenes': scenes,
            'by_class': kinds, 'look_flip_to_look_stay': {'numerator': flip,
                'denominator': stay, 'ratio': flip / stay if stay else None}}


def alternatives(rows, heldout, balance_fn):
    excluded = [r for r in rows if r['scene'] in set(heldout)]
    train = [r for r in rows if r['scene'] not in set(heldout)]
    # Do not call postfilter(): it applies the original 0.15m gate before balance.
    balanced = balance_fn(train, seed=0)
    keys = {r['dataset_key'] for r in balanced['selected']}
    selected = [r for r in train if r['dataset_key'] in keys]
    return {'no_stay_displacement_no_AB_balance': summary(train),
            'only_original_scene_AB_balance_seed0': summary(selected),
            'holdout_excluded': summary(excluded), 'original_balance': balanced,
            'stay_displacement_filter_applied': False,
            'ratio_or_left_position_selection_applied': False,
            'pool_frozen': False, 'formal_training_authorized': False}


def scan_cache_receipts(root, covered):
    """Record all observed original cache results; admit only complete receipts.

    Covered original batches are already authenticated by load_receipt. Failed,
    black, incomplete and unrelated result records are retained in the scan but
    cannot become new accepted training rows. A new mixed train/test certificate
    stops this projection instead of silently dropping part of its population.
    """
    from src.tools import smoke_e2e_pilot_certificates as certificates
    from src.tools import g01_train_inventory as inv
    from src.tools import g01_quota_plan as quota
    from src.tools.t01_partition import audited_result
    known = {(str((root / r['file']).resolve()), r['sha256']) for r in covered}
    observed, extra, publications = [], [], []
    paths = sorted((root / 'results').glob('t02_export_cache*.json'))
    for path in paths:
        ref = file_ref(path, root)
        raw = json.loads(path.read_bytes())
        meta, metrics = raw.get('_meta', {}), raw.get('metrics', {})
        record = {'source': ref, 'observed_end': meta.get('end'),
                  'observed_exit_code': meta.get('exit_code'),
                  'observed_cache_matches_live': metrics.get('cache_matches_live'),
                  'observed_downstream_allowed': metrics.get('downstream_allowed')}
        if metrics.get('kind') != 't02_formal_cache_check':
            record['disposition'] = 'not_an_original_formal_cache_result'
        elif not meta.get('end'):
            record['disposition'] = 'unsealed_not_eligible'
        elif meta.get('exit_code') != 0 or metrics.get('cache_matches_live') is not True \
                or metrics.get('downstream_allowed') is not True:
            # Audit original failures too; they are exclusions, never acceptance.
            _, actual = audited_result(path)
            need(actual['sha256'] == ref['sha256'], 'Cache result changed during scan')
            publications.append(actual)
            record['disposition'] = 'failed_or_incomplete_not_eligible'
        elif (str(path.resolve()), ref['sha256']) in known:
            record['disposition'] = 'already_in_fixed83_certificates'
        else:
            evidence = certificates.certificate_metadata(ref, root=root)
            publications.append(evidence['source'])
            scenes = set()
            for item in evidence['complete']:
                # Table header bytes are bound by the exact original certificate;
                # binary payload and native reauthentication are not repeated here.
                table_ref = item['table_manifest']
                table = inv.read(table_ref, root, Path(table_ref['file']).resolve().parent)
                scenes.add(table['scene'])
            flags = {quota.scene_allowed(scene, 'train') for scene in scenes}
            need(flags and len(flags) == 1, 'Mixed or empty new cache certificate; explicit review required')
            record['scenes'] = sorted(scenes)
            record['certified_dataset_keys'] = sorted(r['dataset_key'] for r in evidence['complete'])
            if True in flags:
                record['disposition'] = 'additional_training_cache_certificate'
                extra.append(ref)
            else:
                need(all(quota.scene_allowed(scene, 'test') for scene in scenes),
                     'Cache certificate contains an unknown scene partition')
                record['disposition'] = 'test_certificate_excluded'
        observed.append(record)
    return observed, extra, publications


def main():
    sys.path.insert(0, str(ROOT))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', required=True)
    p.add_argument('--timeout-seconds', type=int, default=300)
    a = p.parse_args()
    need(a.name.startswith('diag02_') and a.name.replace('_', '').isalnum(), 'Unique diag02 name required')
    need(0 < a.timeout_seconds <= 600, 'Metadata-only CPU stop bound required')
    out = ROOT / 'logs/review' / a.name
    need(not out.exists() and not (ROOT / 'results' / (a.name + '.json')).exists(), 'Preserve previous attempts')
    out.mkdir(parents=True)
    config = {'name': a.name, 'version': VERSION, 'gpus': 0, 'seed': 0,
              'base_inventory': BASE, 'timeout_seconds': a.timeout_seconds}
    metrics = {'kind': VERSION, 'complete': False, 'errors': [], 'producer_exit_code': 1,
               'pool_frozen': False, 'formal_training_authorized': False,
               'native_scientific_revalidation_performed': False}
    sources = []; code = 1; started = time.monotonic()
    def expired(*_):
        raise TimeoutError('DIAG02 E6 metadata deadline expired')
    signal.signal(signal.SIGALRM, expired); signal.alarm(a.timeout_seconds)
    try:
        from src.tools import g01_train_inventory as inv
        from src.tools.train_holdout_freeze import load_frozen_holdout
        from src.envs.pair_table_balance import balance_pairs
        from src.tools.t02_export import save_json
        baseline, original = inv.load_receipt(BASE, root=ROOT)
        sources.append(original)
        frozen, freeze_proof = load_frozen_holdout(baseline['holdout_reference'], root=ROOT)
        base_rows = baseline['qualified_rows']
        need(len(base_rows) == 83 and len({r['dataset_key'] for r in base_rows}) == 83,
             'Original qualified83 population changed')
        base_train = [r for r in base_rows if r['scene'] not in frozen['selected_scenes']]
        need({k: summary(base_train)['by_class'][k]['pair_count'] for k in KINDS}
             == {'direct_flip': 6, 'direct_stay': 26, 'look_flip': 18, 'look_stay': 13},
             'Pilot training part differs from DIAG02 declared baseline')
        cutoff = datetime.datetime.now(datetime.timezone.utc).isoformat()
        observed, extra_refs, observed_sources = scan_cache_receipts(ROOT, baseline['cache_sources'])
        sources.extend(observed_sources)
        extra_rows, extra_sources, extra_proofs, extra_inputs = inv.additional_cache_rows(extra_refs, root=ROOT)
        sources.extend(extra_sources)
        combined = combine_rows(base_rows, extra_rows)
        base_keys = {r['dataset_key'] for r in base_rows}
        new = [r for r in combined if r['dataset_key'] not in base_keys]
        report = {'version': VERSION, 'scan_started_at_utc': cutoff, 'sources': sources,
            'baseline_inventory': original, 'baseline_training': summary(base_train),
            'holdout_reference': baseline['holdout_reference'], 'holdout_proof': freeze_proof,
            'fixed_holdout_scenes': frozen['selected_scenes'],
            'cache_result_scan': observed, 'additional_cache_sources': extra_refs,
            'additional_cache_proofs': extra_proofs, 'additional_input_metadata': extra_inputs,
            'new_unique_cache_qualified_before_holdout': summary(new),
            'new_unique_cache_qualified_after_holdout': summary([r for r in new if r['scene'] not in frozen['selected_scenes']]),
            'rows_before_holdout': combined, 'options': alternatives(combined, frozen['selected_scenes'], balance_pairs),
            'scope': 'Read-only metadata reuse of original T02 and cache certificates. A scenario count is not a newly frozen or approved formal training pool. Only exact certified original training identities; exports without cache cannot enter. No Stay displacement, ratio, or left-position selection.'}
        metrics.update(report=save_json(out / 'report.json', report), complete=True, producer_exit_code=0)
        code = 0
    except BaseException as error:
        metrics['errors'].append(type(error).__name__ + ': ' + str(error)); traceback.print_exc()
    finally:
        signal.alarm(0); metrics['elapsed_seconds'] = time.monotonic() - started
        from src.tools.t02_export import save_json
        unique = {(r['file'], r['sha256']): r for r in sources}
        save_json(ROOT / 'results' / (a.name + '.json'), {'config': config, 'metrics': metrics,
            'sources': [unique[k] for k in sorted(unique)]})
        print('FINAL_METRICS_JSON: ' + json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())

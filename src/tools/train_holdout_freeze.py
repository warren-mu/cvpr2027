"""Freeze the approved T04 scene holdout once, after actual pilot qualification.

This is a metadata-only consumer of the completed original qualification. It
never replaces native/table/cache authentication or selects a new holdout when
a later pool grows. Run the producer through scripts/run_logged.py.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
import traceback

from src.envs.thor_common import ROOT
from src.tools.t01_partition import audited_result, canonical_hash
from src.tools.merge_pair_v2 import producer_arguments
from src.tools.t02_export import save_json

VERSION = 'train_holdout_scenes_v1'
CERTIFICATE_VERSION = 'train_holdout_scenes_v2_original_certificates83'
CERTIFICATE_MODE = 'approved_original_certificates83_after02_v1'
CERTIFICATE_RESULT = 'results/smoke_e2e_pilot_pool_83_certificates_01.json'
POLICY = 'room_stratified_scene_hash_seed0_ten_percent_v1'
NAME = 'train_holdout_freeze'
DESTINATION = 'data/world_pairs/train_holdout_scenes.json'
QUALIFICATION_NAME = 'smoke_e2e_pilot_pool_83_01'
QUALIFICATION_FILE = 'results/' + QUALIFICATION_NAME + '.json'
MANIFEST_FILE = 'data/world_pairs/e2e_pilot_pool_83_01.json'
KINDS = ('direct_flip', 'direct_stay', 'look_flip', 'look_stay')
ROOMS = ('kitchen', 'living', 'bedroom', 'bathroom')
CACHE31_FILE = 'results/t02_export_cache_six09_14_01_retry4.json'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ref_identity(reference):
    require(isinstance(reference, dict) and isinstance(reference.get('file'), str)
            and re.fullmatch(r'[0-9a-f]{64}', str(reference.get('sha256', ''))),
            'Require an explicit original file/SHA reference')
    return {key: reference[key] for key in ('file', 'sha256')}


def validate_reference(reference, *, root=ROOT, expected=None):
    """Read original bytes only; this helper alone is not freeze authorization."""
    reference = ref_identity(reference)
    path = (Path(root) / reference['file']).resolve()
    path.relative_to(Path(root).resolve())
    if expected is not None:
        require(path == (Path(root) / expected).resolve(), 'Unexpected fixed artifact path')
    body = path.read_bytes()
    require(hashlib.sha256(body).hexdigest() == reference['sha256'], 'Artifact SHA changed')
    return json.loads(body)


def values(argv, flag):
    return [argv[i + 1] for i, value in enumerate(argv[:-1]) if value == flag]


def room_type(scene):
    # Exact approved train scene ranges; no rounding, inference or test scenes.
    match = re.fullmatch(r'FloorPlan([0-9]+)', scene)
    require(match is not None, 'Unknown scene identifier')
    number = int(match[1])
    for base, room in ((0, 'kitchen'), (200, 'living'), (300, 'bedroom'), (400, 'bathroom')):
        if base + 1 <= number <= base + 20:
            return room
    raise ValueError('Holdout contains a test or non-training scene')


def compact_population(entries):
    require(isinstance(entries, list) and entries, 'Missing actual qualified population')
    rows = []
    for entry in entries:
        require(isinstance(entry.get('dataset_key'), str) and entry['dataset_key']
                and re.fullmatch(r'[0-9a-f]{64}', str(entry.get('proposal_identity', '')))
                and entry.get('kind') in KINDS, 'Invalid qualified pair identity/class')
        rows.append({key: entry[key] for key in ('dataset_key', 'proposal_identity', 'scene', 'kind')})
        room_type(entry['scene'])
    for field in ('dataset_key', 'proposal_identity'):
        require(len(rows) == len({row[field] for row in rows}), 'Duplicate qualified population identity')
    return sorted(rows, key=lambda row: row['dataset_key'])


def counts(entries):
    counted = Counter(row['kind'] for row in entries)
    return {kind: counted[kind] for kind in KINDS}


def _audited(reference, root):
    expected = ref_identity(reference)
    data, actual = audited_result(Path(root) / expected['file'])
    require(ref_identity(actual) == expected, 'Original logged result SHA changed')
    return data, actual


def authenticate_qualification(reference, *, root=ROOT):
    """Audit completed qualification + immutable metadata; no image/native scan."""
    from src.tools import smoke_e2e_pilot_dataset as dataset
    reference = ref_identity(reference)
    require(reference['file'] == QUALIFICATION_FILE, 'Use the actual original normal83 qualification')
    data, source = _audited(reference, root)
    cfg, meta, metrics = data['config'], data['_meta'], data['metrics']
    argv = producer_arguments(meta['command'], 'src.tools.smoke_e2e_pilot_dataset')
    declared = cfg.get('cache_source')
    fixed = validate_reference(dataset.FALLBACK52_POLICY, root=root)
    require(isinstance(declared, list) and len(declared) == 4
            and declared[:3] == fixed['cache_sources']
            and ref_identity(declared[3])['file'] == CACHE31_FILE,
            'The original three cache batches and fixed31 fourth batch must be unchanged')
    require(meta.get('end') and type(meta.get('exit_code')) is int and meta['exit_code'] == 0
            and meta.get('name') == QUALIFICATION_NAME and cfg.get('name') == QUALIFICATION_NAME
            and cfg.get('version') == dataset.VERSION and cfg.get('deadline_fallback_52') is False
            and '--deadline-fallback-52' not in argv
            and cfg.get('split_policy') == POLICY and cfg.get('out') == MANIFEST_FILE
            and values(argv, '--name') == [QUALIFICATION_NAME]
            and values(argv, '--out') == [MANIFEST_FILE]
            and values(argv, '--split-policy') == [POLICY]
            and [json.loads(v) for v in values(argv, '--cache-source')] == declared
            and metrics.get('kind') == 'smoke_e2e_pilot_dataset_preflight'
            and metrics.get('passed') is True and metrics.get('errors') == []
            and metrics.get('producer_exit_code') == 0,
            'Require a completed successful original qualification, not a pending or fallback manifest')
    manifest_ref = ref_identity(metrics['manifest'])
    manifest = validate_reference(manifest_ref, root=root, expected=MANIFEST_FILE)
    proof = metrics['dataset_proof']
    unhashed = copy.deepcopy(manifest); unhashed.pop('common_pool_hash', None)
    population = compact_population(manifest['entries'])
    original_split = dataset.split_entries(manifest['entries'], POLICY)
    require(manifest.get('version') == dataset.VERSION and manifest.get('partition') == 'train'
            and manifest.get('field_of_view') == 60 and type(manifest.get('seed')) is int and manifest['seed'] == 0
            and manifest.get('cache_sources') == declared and 'qualification' not in manifest
            and 'qualification' not in proof and len(population) == 83
            and manifest.get('counts_by_kind') == counts(population) and all(counts(population).values())
            and proof.get('version') == dataset.VERSION and proof.get('cache_authorized') is True
            and proof.get('counts_by_kind') == counts(population)
            and proof.get('split_policy') == POLICY
            and proof.get('split') == manifest.get('split') == original_split
            and proof.get('train_keys') == original_split['train_keys']
            and proof.get('eval_keys') == original_split['eval_keys']
            and proof.get('world_orders') == {r['dataset_key']: r['world_order'] for r in manifest['entries']}
            and proof.get('common_pool_hash') == manifest.get('common_pool_hash') == canonical_hash(unhashed)
            and [ref_identity(r) for r in proof.get('sources', [])] == declared
            and [ref_identity(r['source']) for r in proof.get('formal_cache_batches', [])] == declared,
            'Original qualified manifest/population/split/proof differs')
    batch_keys = []
    for batch, cache_ref in zip(proof['formal_cache_batches'], declared):
        cache_data, audited = _audited(cache_ref, root)
        cache_meta, cm = cache_data['_meta'], cache_data['metrics']
        require(cache_meta.get('end') and cache_meta.get('exit_code') == 0
                and cm.get('producer_exit_code') == 0 and cm.get('errors') == []
                and cm.get('cleanup_errors') == [] and cm.get('cache_matches_live') is True
                and cm.get('sample_complete') is True and cm.get('downstream_allowed') is True
                and cm.get('stop_downstream') is False
                and batch['source'].get('metrics_sha256') == audited['metrics_sha256'],
                'Qualification references an incomplete or changed original cache batch')
        require(all(ref_identity(batch[field]) == ref_identity(cm[field])
                    for field in ('authorization_manifest', 'validation_report', 'sampling_plan')),
                'Original cache proof metadata references differ')
        batch_keys.extend(batch['dataset_keys'])
    require(len(batch_keys) == len(set(batch_keys)) == len(population)
            and sorted(batch_keys) == [row['dataset_key'] for row in population],
            'Qualification omitted or added a cache-authorized pair')
    return manifest, proof, source, manifest_ref


def authenticate_population(reference, *, certificates_after_02=False, root=ROOT):
    """Dispatch authority without relabeling certificate-only data as rechecked.

    The certificate loader authenticates its original successful activation
    receipt. It does not re-evaluate a later clock or the later 709593 outcome.
    """
    require(type(certificates_after_02) is bool, 'Explicit population authority required')
    if not certificates_after_02:
        return authenticate_qualification(reference, root=root)
    from src.tools.smoke_e2e_pilot_certificates import authenticate_certificate_qualification
    require(ref_identity(reference)['file'] == CERTIFICATE_RESULT,
            'Certificate mode requires its explicit original population result')
    manifest, proof, source, manifest_ref = authenticate_certificate_qualification(reference, root=root)
    authority = manifest.get('qualification')
    require(isinstance(authority, dict) and authority.get('mode') == CERTIFICATE_MODE
            and authority.get('extra_source_reauthentication') == {'status': 'not_awaited', 'passed': None}
            and proof.get('qualification') == authority,
            'Certificate authority must not claim an additional source reauthentication PASS')
    from src.tools.smoke_e2e_pilot_dataset import split_entries
    population = compact_population(manifest['entries'])
    require(len(population) == 83 and manifest.get('partition') == 'train'
            and manifest.get('field_of_view') == 60 and type(manifest.get('seed')) is int
            and manifest['seed'] == 0 and manifest.get('split') == split_entries(manifest['entries'], POLICY)
            and manifest.get('counts_by_kind') == counts(population) and all(counts(population).values()),
            'Certificate population must preserve all original83 and the approved initial scene selection')
    return manifest, proof, source, manifest_ref


def frozen_document(manifest, source, manifest_ref, *, certificates_after_02=False):
    """Called after explicit population authentication; stores the original split."""
    population = compact_population(manifest['entries'])
    split = copy.deepcopy(manifest['split'])
    held = split['selected_scenes']
    train = [row for row in population if row['scene'] not in held]
    evaluation = [row for row in population if row['scene'] in held]
    document = {'version': VERSION, 'approved_revision': '629d4a03', 'policy': POLICY, 'seed': 0,
        'frozen_once': True, 'selected_scenes': held,
        'scenes': [{'scene_id': scene, 'room_type': room_type(scene)} for scene in held],
        'population': {'dataset_keys': [row['dataset_key'] for row in population],
            'dataset_keys_sha256': canonical_hash([row['dataset_key'] for row in population]),
            'entries': population, 'common_pool_hash': manifest['common_pool_hash'],
            'manifest_reference': manifest_ref, 'qualification_result': copy.deepcopy(source)},
        'split': split, 'counts_by_partition': {'train': counts(train), 'holdout': counts(evaluation)},
        'missing_room_types': [room for room in ROOMS if room not in {room_type(r['scene']) for r in population}],
        'scope': 'Fixed scene exclusion for all future training and fixed pilot evaluation; not a final main-pool or test approval'}
    if certificates_after_02:
        document.update(version=CERTIFICATE_VERSION, authorization_revision='55753931',
                        population_authority=copy.deepcopy(manifest['qualification']))
    return document


def validate_frozen(frozen):
    require(frozen.get('version') in (VERSION, CERTIFICATE_VERSION) and frozen.get('approved_revision') == '629d4a03'
            and frozen.get('policy') == POLICY and type(frozen.get('seed')) is int and frozen['seed'] == 0
            and frozen.get('frozen_once') is True, 'Unknown frozen holdout policy')
    if frozen['version'] == CERTIFICATE_VERSION:
        authority = frozen.get('population_authority', {})
        require(frozen.get('authorization_revision') == '55753931'
                and authority.get('mode') == CERTIFICATE_MODE
                and authority.get('extra_source_reauthentication') == {'status': 'not_awaited', 'passed': None},
                'Certificate freeze must preserve its explicit non-reauthenticated authority')
    else:
        require('population_authority' not in frozen and 'authorization_revision' not in frozen,
                'Original successful-qualification freeze cannot claim certificate-only authority')
    population = frozen['population']
    entries = compact_population(population['entries'])
    keys = [row['dataset_key'] for row in entries]
    held = frozen['selected_scenes']
    require(isinstance(held, list) and held == sorted(set(held)) and held
            and set(held) <= {r['scene'] for r in entries}
            and population['entries'] == entries and population['dataset_keys'] == keys
            and population['dataset_keys_sha256'] == canonical_hash(keys)
            and frozen['scenes'] == [{'scene_id': scene, 'room_type': room_type(scene)} for scene in held],
            'Frozen scene/population identity differs')
    train, evaluation = partition_entries(entries, frozen, _validated=True)
    split = frozen['split']
    require(train and evaluation and split.get('policy') == POLICY and type(split.get('seed')) is int
            and split['seed'] == 0 and split.get('selected_scenes') == held
            and split.get('holdout_fraction_requested') == .1
            and split.get('train_keys') == sorted(row['dataset_key'] for row in train)
            and split.get('eval_keys') == sorted(row['dataset_key'] for row in evaluation)
            and frozen['counts_by_partition'] == {'train': counts(train), 'holdout': counts(evaluation)}
            and frozen['missing_room_types'] == [room for room in ROOMS if room not in {room_type(r['scene']) for r in entries}],
            'Frozen split/counts differ from whole-scene assignment')
    return frozen


def partition_entries(entries, frozen, *, _validated=False):
    """Apply existing scenes to future pools. Never selects or adds holdout scenes."""
    if not _validated:
        validate_frozen(frozen)
    held = set(frozen['selected_scenes'])
    compact_population(entries)
    return ([row for row in entries if row['scene'] not in held],
            [row for row in entries if row['scene'] in held])


def is_holdout_scene(scene, frozen):
    validate_frozen(frozen)
    room_type(scene)
    return scene in frozen['selected_scenes']


def assert_pilot_split(entries, split, frozen):
    validate_frozen(frozen)
    require(compact_population(entries) == frozen['population']['entries']
            and split == frozen['split'], 'Pilot population or scene split differs from the permanent holdout')
    train, evaluation = partition_entries(entries, frozen)
    require(sorted(r['dataset_key'] for r in train) == split['train_keys']
            and sorted(r['dataset_key'] for r in evaluation) == split['eval_keys'],
            'Pilot train/evaluation contains a frozen-scene leak')


def freeze(reference, *, certificates_after_02=False, root=ROOT):
    destination = Path(root) / DESTINATION
    require(not destination.exists(), 'The permanent holdout already exists; never recompute or overwrite it')
    manifest, _, source, manifest_ref = authenticate_population(reference, certificates_after_02=certificates_after_02, root=root)
    document = frozen_document(manifest, source, manifest_ref, certificates_after_02=certificates_after_02)
    validate_frozen(document)
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = save_json(destination, document)  # xb; racing writers cannot overwrite.
    destination.chmod(0o444)
    return document, written, source


def load_frozen_holdout(reference, *, root=ROOT):
    """Require original freeze log/provenance, not just a self-consistent JSON."""
    frozen = validate_reference(reference, root=root, expected=DESTINATION)
    validate_frozen(frozen)
    receipt_path = Path(root) / 'results' / (NAME + '.json')
    receipt, receipt_ref = audited_result(receipt_path)
    meta, cfg, metrics = receipt['_meta'], receipt['config'], receipt['metrics']
    argv = producer_arguments(meta['command'], 'src.tools.train_holdout_freeze')
    qualification = ref_identity(frozen['population']['qualification_result'])
    certificates = frozen['version'] == CERTIFICATE_VERSION
    require(meta.get('end') and meta.get('exit_code') == 0 and meta.get('name') == NAME
            and cfg.get('name') == NAME and cfg.get('version') == frozen['version']
            and cfg.get('certificates_after_02', False) is certificates
            and argv.count('--certificates-after-02') == int(certificates)
            and cfg.get('qualification_result') == qualification
            and values(argv, '--name') == [NAME]
            and [json.loads(v) for v in values(argv, '--qualification-result')] == [qualification]
            and metrics.get('kind') == 'train_holdout_freeze' and metrics.get('passed') is True
            and metrics.get('producer_exit_code') == 0 and metrics.get('errors') == []
            and ref_identity(metrics['frozen_holdout']) == ref_identity(reference)
            and metrics.get('counts_by_partition') == frozen['counts_by_partition']
            and (not certificates or (metrics.get('population_authority') == frozen['population_authority']
                 and 'extra_source_reauthentication_passed' in metrics
                 and metrics['extra_source_reauthentication_passed'] is None))
            and [ref_identity(r) for r in receipt.get('sources', [])] == [qualification],
            'Frozen file has no matching successful original logged freeze')
    manifest, _, source, manifest_ref = authenticate_population(qualification, certificates_after_02=certificates, root=root)
    expected = frozen_document(manifest, frozen['population']['qualification_result'], manifest_ref, certificates_after_02=certificates)
    require(frozen == expected and ref_identity(source) == qualification,
            'Permanent freeze differs from actual original qualification')
    return frozen, ref_identity(reference)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True, choices=(NAME,))
    parser.add_argument('--qualification-result', required=True, help='Actual completed original population {file,sha256}')
    parser.add_argument('--certificates-after-02', action='store_true',
                        help='55753931 original83 certificate authority; never an extra source reauthentication PASS')
    args = parser.parse_args(argv)
    output = ROOT / 'results' / (NAME + '.json')
    require(not output.exists(), 'Preserve the previous freeze result')
    cfg = {'name': args.name, 'version': CERTIFICATE_VERSION if args.certificates_after_02 else VERSION,
           'certificates_after_02': args.certificates_after_02,
           'qualification_result': json.loads(args.qualification_result), 'gpus': 0}
    metrics = {'kind': 'train_holdout_freeze', 'passed': False, 'errors': [], 'frozen_holdout': None,
               'scope': 'Metadata-only fixed scene holdout; original native/cache qualification remains authoritative'}
    code, sources = 1, []
    try:
        frozen, written, source = freeze(cfg['qualification_result'], certificates_after_02=args.certificates_after_02)
        sources = [source]
        metrics.update(passed=True, frozen_holdout=written, counts_by_partition=frozen['counts_by_partition'],
                       missing_room_types=frozen['missing_room_types'], selected_scenes=frozen['selected_scenes'],
                       population_dataset_keys_sha256=frozen['population']['dataset_keys_sha256'])
        if args.certificates_after_02:
            metrics['population_authority'] = frozen['population_authority']
            metrics['extra_source_reauthentication_passed'] = None
        code = 0
    except Exception as error:
        traceback.print_exc()
        metrics['errors'].append(f'{type(error).__name__}: {error}')
    finally:
        metrics['producer_exit_code'] = code
        output.parent.mkdir(parents=True, exist_ok=True)
        save_json(output, {'metrics': metrics, 'config': cfg, 'sources': sources})
        print('FINAL_METRICS_JSON: ' + json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())

"""Bind an ended Look pair to its recorded Controller, never the last visit.

Association is additional provenance validation. It does not certify a pair;
callers must still execute the unchanged independent merge.check_pair.
"""
import copy
import json
from pathlib import Path
import re
from src.envs import pair_v2_look_unit_policy as policy
from src.tools import merge_pair_v2 as merge

VERSION = 'look_pair_controller_preflight_association_v1'
POLICY = {
    'version': VERSION,
    'selection': 'identical authenticated pair/raw search_controller_id; exactly one audited controller visit',
    'required': 'indexed original preflight, original baseline, canonical setup, attempt setup, physical world snapshots and matching worker/Unity identities',
    'source_final_preflight': 'preserved and authenticated as the selected final visit; never replaced in source data',
    'validation': 'original verify_preflight plus unchanged full check_pair; no acceptance threshold changes',
    'fallback': 'none; no hash-only or nearest-visit inference',
}


def same(first, second):
    encode = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return encode(first) == encode(second)


def verify_pair_controller(data, source_ref, journal, pair, pair_path, *, root=None):
    root = Path(merge.ROOT if root is None else root).resolve()
    config, metrics = data['config'], data['metrics']
    policy.validate_source(config, metrics)
    raw_root = Path(config['raw_evidence_dir']).resolve()
    pair_dir = Path(pair_path).resolve().parent
    if pair.get('kind') not in ('look_flip', 'look_stay') or pair.get('status') != 'passed':
        raise ValueError('Per-controller adapter is restricted to explicit passed Look candidates')
    raw_ref = journal['raw_attempt']
    raw = merge.read_reference(raw_ref, root=root, within=raw_root)
    for key in ('pair_id', 'scene', 'kind', 'status', 'discard_reasons'):
        if not same(raw.get(key), journal.get(key)) or not same(pair.get(key), journal.get(key)):
            raise ValueError('Pair/raw/journal identities or outcomes differ')
    for key in ('pair_file', 'pair_sha256'):
        if not same(raw.get(key), journal.get(key)):
            raise ValueError('Raw attempt does not name its authenticated journal pair')
    if merge.reference_path({'file': journal['pair_file'], 'sha256': journal['pair_sha256']},
                            root=root, within=pair_dir) != Path(pair_path).resolve():
        raise ValueError('Raw/journal pair reference differs from the loaded portable pair')
    controller = pair.get('search_controller_id')
    if (not isinstance(controller, str) or re.fullmatch(r'marked_controller_[0-9]{3}', controller) is None
            or raw.get('search_controller_id') != controller):
        raise ValueError('Pair/raw require the same explicit Controller ID')
    if raw.get('mode') != pair.get('mode') or pair.get('scene') != config['scene']:
        raise ValueError('Pair/raw mode or source scene differs')
    visits = metrics.get('controller_visits')
    if not isinstance(visits, list) or not visits:
        raise ValueError('Audited source lacks its Controller visits')
    names = [visit.get('controller_id') for visit in visits]
    if (any(not isinstance(name, str) or re.fullmatch(r'marked_controller_[0-9]{3}', name) is None for name in names)
            or len(names) != len(set(names)) or controller not in names):
        raise ValueError('Missing, duplicate or invalid audited Controller identity')
    visit = visits[names.index(controller)]
    selected = metrics.get('selected_controller_id')
    if selected not in names:
        raise ValueError('Source final selected Controller is absent from its visit ledger')
    final_visit = visits[names.index(selected)]
    final_ref = metrics['scene_preflight_reference']
    final_path = merge.reference_path(final_ref, root=root, within=raw_root)
    selected_path = merge.reference_path(final_visit['preflight_reference'], root=root, within=raw_root)
    if final_path != selected_path or final_ref['sha256'] != final_visit['preflight_reference']['sha256']:
        raise ValueError('Original final source preflight does not bind the selected final visit')
    if visit.get('role') != 'natural_marked' or visit.get('cleanup_errors') != [] or visit.get('preflight_passed') is not True:
        raise ValueError('Selected pair Controller has unknown role, preflight or cleanup')
    controller_dir = raw_root / controller / 'construction'
    index_ref = visit['raw_index']
    index_path = merge.reference_path(index_ref, root=root, within=controller_dir)
    if index_path != controller_dir / 'evidence_index.json':
        raise ValueError('Controller evidence index does not occupy its declared construction directory')
    index = merge.read_reference(index_ref, root=root, within=controller_dir)
    files = index.get('files')
    if not isinstance(files, list) or not files:
        raise ValueError('Controller evidence index lacks its original file entries')
    indexed = {}
    for ref in files:
        if (not isinstance(ref, dict) or not isinstance(ref.get('file'), str)
                or re.fullmatch(r'[0-9a-f]{64}', str(ref.get('sha256', ''))) is None):
            raise ValueError('Malformed original Controller index entry')
        path = (root / ref['file']).resolve()
        if not path.is_relative_to(controller_dir) or path in indexed:
            raise ValueError('Duplicate or foreign file in the Controller evidence index')
        indexed[path] = ref

    def bound_reference(ref):
        path = merge.reference_path(ref, root=root, within=controller_dir)
        item = indexed.get(path)
        if item is None or item['sha256'] != ref['sha256']:
            raise ValueError('Controller reference is absent from its authenticated raw index')
        if 'bytes' in item and (type(item['bytes']) is not int or item['bytes'] != path.stat().st_size):
            raise ValueError('Indexed native file size differs from its actual bytes')
        return path

    def read(ref):
        return json.loads(bound_reference(ref).read_text())

    def same_reference(first, second):
        return bound_reference(first) == bound_reference(second) and first['sha256'] == second['sha256']

    preflight_ref = visit['preflight_reference']
    preflight_path = bound_reference(preflight_ref)
    if preflight_path != controller_dir / 'scene_preflight.json':
        raise ValueError('Visit preflight is not that Controller original preflight file')
    preflight = read(preflight_ref)
    portable_ref = pair['scene_preflight_reference']
    portable_path = merge.reference_path(portable_ref, root=root, within=pair_dir)
    if portable_ref['sha256'] != preflight_ref['sha256'] or portable_path.read_bytes() != preflight_path.read_bytes():
        raise ValueError('Portable preflight differs from the exact pair Controller original bytes')
    # This reuses the full original action/window/canonical-setup validation.
    # The subsequent unchanged check_pair independently executes it again.
    verified_preflight = merge.verify_preflight(preflight, read, canonical_setup=True)
    baseline = preflight['snapshot']
    baseline_ref = baseline['reference']
    if not same(read(baseline_ref), {key: value for key, value in baseline.items() if key != 'reference'}):
        raise ValueError('Controller baseline differs from its original indexed snapshot')
    if not same_reference(preflight['initial_agent_setup_reference'], visit['initial_agent_setup_reference']):
        raise ValueError('Controller preflight differs from its recorded canonical setup')
    read(visit['initial_agent_setup_reference'])
    setup_ref = pair['attempt_setup_reference']
    if not same_reference(setup_ref, raw['attempt_setup_reference']):
        raise ValueError('Pair/raw attempt setup references differ')
    setup = read(setup_ref)
    if not same(setup, pair.get('attempt_setup')) or not same(setup, raw.get('attempt_setup')):
        raise ValueError('Pair/raw inline attempt setup differs from its indexed bytes')
    if (setup.get('pair_id') != pair['pair_id'] or setup.get('scene') != config['scene']
            or setup.get('status') != 'passed' or setup.get('setup_passed') is not True
            or setup.get('error') is not None
            or not same_reference(setup['baseline_reference'], baseline_ref)):
        raise ValueError('Attempt setup does not bind the exact passed Controller baseline')
    worker, unity = visit.get('worker_pid'), (visit.get('unity_identity') or {}).get('pid')
    if any(type(value) is not int or value <= 0 for value in (worker, unity)):
        raise ValueError('Controller visit lacks actual worker/Unity process identities')
    def snapshot_identity(snapshot):
        if (snapshot.get('scene') != config['scene'] or snapshot.get('worker_pid') != worker
                or snapshot.get('unity_pid') != unity):
            raise ValueError('Indexed native snapshot belongs to another scene or Controller process')
    snapshot_identity(baseline)
    for key in ('restored_snapshot_reference',):
        snapshot_identity(read(setup[key]))
    snapshot_identity(read(setup['restoration']['restored_snapshot_reference']))
    worlds = pair.get('physical_world_references')
    if not isinstance(worlds, list) or len(worlds) != 2 or not same(worlds, raw.get('physical_world_references')):
        raise ValueError('Frozen Look pair/raw lack matching original physical world references')
    for ref in worlds:
        snapshot_identity(read(ref))
    return {'version': VERSION, 'policy': copy.deepcopy(POLICY),
        'source_reference': copy.deepcopy(source_ref), 'controller_id': controller,
        'source_final_controller_id': selected, 'original_source_final_preflight_reference': copy.deepcopy(final_ref),
        'pair_preflight_reference': copy.deepcopy(portable_ref), 'controller_preflight_reference': copy.deepcopy(preflight_ref),
        'controller_index_reference': copy.deepcopy(index_ref), 'raw_attempt_reference': copy.deepcopy(raw_ref),
        'baseline_reference': copy.deepcopy(baseline_ref), 'canonical_setup_reference': copy.deepcopy(visit['initial_agent_setup_reference']),
        'attempt_setup_reference': copy.deepcopy(setup_ref), 'worker_pid': worker, 'unity_pid': unity,
        'preflight_validation': verified_preflight, 'source_record_mutated': False,
        'scope': 'Controller provenance only; unchanged full check_pair is still required'}

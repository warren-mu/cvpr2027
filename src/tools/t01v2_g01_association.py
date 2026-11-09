"""Bind an ended mixed G01 pair to its recorded Controller, never the last visit.

Association is additional provenance validation. It does not certify a pair;
callers must still execute the unchanged independent merge.check_pair.
"""
import copy
import json
from pathlib import Path
import re
from src.envs import pair_v2_g01_policy as policy
from src.envs import pair_v2_g01_search as search
from src.tools import merge_pair_v2 as merge

VERSION = 'g01_pair_controller_preflight_association_v1'
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


def verify_final_preflight_binding(config, metrics, visits, *, root, raw_root):
    """Distinguish a final failed attempt from the last selected success.

    Inputs are from the caller's audited source/log. The original equal-reference
    path is unchanged. The additional path authenticates every visit's indexed
    preflight and the exact final exclusion; it never substitutes source fields.
    """
    names = [visit['controller_id'] for visit in visits]
    selected = metrics.get('selected_controller_id')
    if selected not in names:
        raise ValueError('Source final selected Controller is absent from its visit ledger')
    selected_visit = visits[names.index(selected)]
    passed_names = [visit['controller_id'] for visit in visits if visit.get('preflight_passed') is True]
    if not passed_names or selected != passed_names[-1]:
        raise ValueError('Source selected Controller is not the last preflight-passed visit')
    final_ref = metrics['scene_preflight_reference']
    final_path = merge.reference_path(final_ref, root=root, within=raw_root)
    selected_path = merge.reference_path(selected_visit['preflight_reference'], root=root, within=raw_root)
    if (final_path == selected_path and final_ref['sha256'] == selected_visit['preflight_reference']['sha256']
            and visits[-1].get('preflight_passed') is not False):
        return None

    message = 'Original final source preflight does not bind the selected final visit'
    if (len(visits) < 2 or names != [f'marked_controller_{index:03d}' for index in range(len(visits))]
            or selected != names[-2] or metrics.get('status') != 'completed_search'
            or metrics.get('cleanup_errors') != []):
        raise ValueError(message + '; no authenticated final-failed-visit case')
    exclusion = metrics.get('scene_excluded')
    if (not isinstance(exclusion, dict) or exclusion.get('scene') != config['scene']
            or exclusion.get('reason') != 'settled_baseline_window_preflight_failed'
            or not same(exclusion.get('evidence'), final_ref)):
        raise ValueError(message + '; missing exact final preflight exclusion')

    verified = []
    for index, visit in enumerate(visits):
        directory = raw_root / visit['controller_id'] / 'construction'
        expected_passed = index < len(visits) - 1
        worker, unity = visit.get('worker_pid'), (visit.get('unity_identity') or {}).get('pid')
        if (visit.get('role') != 'natural_marked' or visit.get('cleanup_errors') != []
                or visit.get('preflight_passed') is not expected_passed
                or any(type(pid) is not int or pid <= 0 for pid in (worker, unity))):
            raise ValueError(message + '; visit order/verdict/ownership/cleanup is not proven')
        index_ref = visit['raw_index']
        index_path = merge.reference_path(index_ref, root=root, within=directory)
        if index_path != directory / 'evidence_index.json':
            raise ValueError('Final preflight visit index is outside its exact Controller path')
        files = merge.read_reference(index_ref, root=root, within=directory).get('files')
        if not isinstance(files, list) or not files:
            raise ValueError('Final preflight visit lacks its original index entries')
        indexed = {}
        for ref in files:
            if (not isinstance(ref, dict) or not isinstance(ref.get('file'), str)
                    or re.fullmatch(r'[0-9a-f]{64}', str(ref.get('sha256', ''))) is None):
                raise ValueError('Malformed final preflight Controller index entry')
            path = (root / ref['file']).resolve()
            if not path.is_relative_to(directory) or path in indexed:
                raise ValueError('Duplicate or foreign final preflight Controller index entry')
            indexed[path] = ref

        def read_indexed(ref):
            path = merge.reference_path(ref, root=root, within=directory)
            member = indexed.get(path)
            if member is None or member['sha256'] != ref['sha256']:
                raise ValueError('Final preflight reference lacks original index membership')
            if 'bytes' in member and (type(member['bytes']) is not int or member['bytes'] != path.stat().st_size):
                raise ValueError('Final preflight indexed size differs from actual bytes')
            return json.loads(path.read_text()), path

        ref = visit['preflight_reference']
        preflight, path = read_indexed(ref)
        if path != directory / 'scene_preflight.json' or preflight.get('passed') is not expected_passed:
            raise ValueError('Final preflight native verdict differs from visit declaration')
        setup = visit['initial_agent_setup_reference']
        if not same(preflight.get('initial_agent_setup_reference'), setup):
            raise ValueError('Final preflight canonical setup differs from its original visit')
        read_indexed(setup)
        if not expected_passed:
            if (path != final_path or ref['sha256'] != final_ref['sha256']
                    or not isinstance(preflight.get('reasons'), list) or not preflight['reasons']
                    or not same(preflight['reasons'], exclusion.get('reasons'))):
                raise ValueError('Final failed preflight does not bind the exact original exclusion')
        verified.append({'controller_id': visit['controller_id'], 'preflight_passed': expected_passed,
            'preflight_reference': copy.deepcopy(ref), 'controller_index_reference': copy.deepcopy(index_ref)})
    return {'version': 'g01_final_attempt_vs_selected_preflight_binding_v1',
        'selected_controller_id': selected, 'last_attempted_controller_id': names[-1],
        'original_final_preflight_reference': copy.deepcopy(final_ref),
        'scene_excluded': copy.deepcopy(exclusion), 'authenticated_visit_preflights': verified,
        'source_record_mutated': False,
        'scope': 'source final failed preflight provenance only; pair Controller and all scientific checks remain required'}


def verify_pair_controller(data, source_ref, journal, pair, pair_path, *, root=None):
    root = Path(merge.ROOT if root is None else root).resolve()
    config, metrics = data['config'], data['metrics']
    policy.validate_source(config, metrics)
    raw_root = Path(config['raw_evidence_dir']).resolve()
    pair_dir = Path(pair_path).resolve().parent
    if pair.get('kind') not in merge.KINDS or pair.get('status') != 'passed':
        raise ValueError('Per-controller adapter is restricted to explicit passed G01 candidates')
    raw_ref = journal['raw_attempt']
    raw = merge.read_reference(raw_ref, root=root, within=raw_root)
    for value in (pair, raw):
        if (value.get('producer_module') != policy.PRODUCER
                or any(value.get(key) != item for key, item in policy.fields(config['g01_partition'], config.get('g01_quota_version') is not None,
                    config.get('g01_configuration_version') is not None).items())):
            raise ValueError('Pair/raw differs from its authenticated G01 source policy')
        if config.get('g01_quota_version') is not None and any(
                value.get(key) != config.get(key) for key in policy.quota_plan.FIELDS):
            raise ValueError('Pair/raw differs from its authenticated static quota assignment')
        search.validate_fields(value)
    pool = pair.get('proposed_pool')
    if (pool not in ('look', 'direct') or raw.get('proposed_pool') != pool
            or any(value.get('proposal', {}).get('projection_pool', {}).get('pool') != pool
                   for value in (pair, raw))):
        raise ValueError('Pair/raw G01 proposal pool differs from its original construction domain')
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
    final_ref = metrics['scene_preflight_reference']
    final_binding = verify_final_preflight_binding(config, metrics, visits, root=root, raw_root=raw_root)
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
        raise ValueError('Frozen G01 pair/raw lack matching original physical world references')
    for ref in worlds:
        snapshot_identity(read(ref))
    admission = metrics.get('g01_witness_admission')
    if (not isinstance(admission, dict) or admission.get('owner', {}).get('pid') != worker
            or not isinstance(admission.get('lease_records'), list)):
        raise ValueError('G01 Controller lacks its authenticated generator admission owner')
    leases = admission['lease_records']
    portable = pair.get('portable_files')
    if not isinstance(portable, list):
        raise ValueError('G01 pair lacks its portable witness manifest')
    admission_links = []
    for world in (0, 1):
        matches = [ref for ref in portable if Path(ref['file']).name == f'witness_w{world}.json']
        if len(matches) != 1:
            raise ValueError('G01 pair needs exactly one authenticated witness file per world')
        replay = merge.read_reference(matches[0], root=root, within=pair_dir)
        pid, unity_pid = replay.get('process_id'), replay.get('unity_pid')
        if any(type(value) is not int or value <= 0 for value in (pid, unity_pid)):
            raise ValueError('G01 witness lacks its actual worker/Unity identity')
        matched = [lease for lease in leases if lease.get('worker_identity', {}).get('pid') == pid
                   and lease.get('unity_identity', {}).get('pid') == unity_pid]
        if len(matched) != 1 or matched[0].get('owner') != admission['owner']:
            raise ValueError('G01 witness does not bind one actual admitted resource lease')
        lease = matched[0]
        if (lease.get('release_confirmed') is not True or lease.get('cleanup_errors') != []
                or lease.get('remaining_owned_group') is not False):
            raise ValueError('G01 witness resource lease lacks confirmed cleanup')
        admission_links.append({'world_index': world, 'witness_reference': copy.deepcopy(matches[0]),
            'lease_token': lease['token'], 'worker_identity': copy.deepcopy(lease['worker_identity']),
            'unity_identity': copy.deepcopy(lease['unity_identity'])})
    if admission_links[0]['lease_token'] == admission_links[1]['lease_token']:
        raise ValueError('Independent world witnesses cannot reuse one resource lease')
    proof = {'version': VERSION, 'policy': copy.deepcopy(POLICY),
        'source_reference': copy.deepcopy(source_ref), 'controller_id': controller,
        'source_final_controller_id': selected, 'original_source_final_preflight_reference': copy.deepcopy(final_ref),
        'pair_preflight_reference': copy.deepcopy(portable_ref), 'controller_preflight_reference': copy.deepcopy(preflight_ref),
        'controller_index_reference': copy.deepcopy(index_ref), 'raw_attempt_reference': copy.deepcopy(raw_ref),
        'baseline_reference': copy.deepcopy(baseline_ref), 'canonical_setup_reference': copy.deepcopy(visit['initial_agent_setup_reference']),
        'attempt_setup_reference': copy.deepcopy(setup_ref), 'worker_pid': worker, 'unity_pid': unity,
        'witness_resource_admission': {'owner': copy.deepcopy(admission['owner']), 'worlds': admission_links},
        'proposal_class_binding': {'proposed_pool': pool, 'actual_kind': pair['kind'],
            'class_cross_over': pair['kind'].split('_', 1)[0] != pool,
            'scope': 'proposal domain versus actual rendered class; no new acceptance threshold'},
        'preflight_validation': verified_preflight, 'source_record_mutated': False,
        'scope': 'Controller provenance only; unchanged full check_pair is still required'}
    if final_binding is not None:
        proof['source_final_preflight_binding'] = final_binding
        proof['source_last_attempted_controller_id'] = final_binding['last_attempted_controller_id']
        proof['policy']['source_final_preflight'] = (
            'preserved as the authenticated last attempted failed visit; selected Controller remains the last success')
    return proof

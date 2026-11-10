"""Explicit selected-test storage option; original source/native checks remain mandatory."""
import copy
import hashlib
import json
from pathlib import Path
import re

from src.envs.thor_common import ROOT
from src.envs import pair_table_codec as codec
from src.envs import pair_v2_configuration as configurations
from src.tools.t01_partition import audited_result


def reference_equal(a, b, root=ROOT):
    return (isinstance(a, dict) and isinstance(b, dict)
            and (Path(root) / a.get('file', '')).resolve() == (Path(root) / b.get('file', '')).resolve()
            and a.get('sha256') == b.get('sha256'))


def selection_reference(value):
    ref = json.loads(value) if isinstance(value, str) else copy.deepcopy(value)
    if (not isinstance(ref, dict) or set(ref) != {'file', 'sha256', 'logged_result'}
            or not isinstance(ref['file'], str) or not re.fullmatch('[0-9a-f]{64}', str(ref['sha256']))
            or not isinstance(ref['logged_result'], dict) or set(ref['logged_result']) != {'file', 'sha256'}
            or not re.fullmatch('[0-9a-f]{64}', str(ref['logged_result']['sha256']))):
        raise ValueError('Test compression needs the exact selected snapshot and original logged result')
    return ref


def load_selection(reference, *, root=ROOT):
    ref = selection_reference(reference)
    root = Path(root).resolve()
    path = (root / ref['file']).resolve()
    if not path.is_relative_to(root / 'logs/review'):
        raise ValueError('Selected snapshot is outside repository review evidence')
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != ref['sha256']:
        raise ValueError('Selected snapshot SHA differs')
    data, logged = audited_result(root / ref['logged_result']['file'])
    if (not reference_equal(logged, ref['logged_result'], root)
            or data['_meta'].get('exit_code') != 0 or not data['_meta'].get('end')
            or not re.fullmatch('diag_current_test_selected_[0-9]+', data['_meta'].get('name', ''))
            or data['metrics'].get('command_exit_code') != 0):
        raise ValueError('Current selected snapshot lacks its successful original logged producer')
    log = root / (data['_meta'].get('log_path') or ('logs/runs/' + data['_meta']['name'] + '.log'))
    markers = re.findall(r'^CURRENT_TEST_SELECTED_JSON:\s*(.+)\s*$', log.read_text(), re.M)
    marker = json.loads(markers[0]) if len(markers) == 1 else {}
    if len(markers) != 1 or not reference_equal(marker.get('report'), ref, root):
        raise ValueError('Selected snapshot is not the unique original logged output')
    report = json.loads(body)
    if any(marker.get(key) != report.get(key) for key in ('counts_by_kind', 'scene_counts_by_kind', 'selected_count', 'smoke_candidate')):
        raise ValueError('Selected snapshot summary differs from its original logged output')
    proof = report['proof']
    selected = configurations.select(proof['unique_proposals'])
    if (proof.get('count_basis_authenticated') is not True
            or selected != proof['configuration_selection']
            or report.get('selected_count') != len(selected['selected'])):
        raise ValueError('Original seed-zero selected representatives differ from the snapshot')
    return {'reference': ref, 'logged_reference': logged, 'selected': selected['selected'],
            'scope': 'current incremental selected test candidates; not a final test freeze or T02 pass'}


def selected_entry(selection, *, source, pair, scene, proposal_identity, kind=None, root=ROOT):
    rows = [row for row in selection['selected'] if reference_equal(row['checkpoint_reference'], source, root)
            and reference_equal(row['pair_reference'], pair, root)]
    if (len(rows) != 1 or rows[0]['scene'] != scene or rows[0]['proposal_identity'] != proposal_identity
            or (kind is not None and rows[0]['kind'] != kind)):
        raise ValueError('Pair is not its exact selected test representative')
    return copy.deepcopy(rows[0])


def validate_package(selection, package, *, root=ROOT):
    codec.require_test_table(package['scene'], package['source_config'])
    row = selected_entry(selection, source=package['source_provenance']['container'],
        pair=package['source_provenance']['pair'], scene=package['scene'],
        proposal_identity=package['independent_verification']['proposal_identity'], kind=package['kind'], root=root)
    if (configurations.from_pair(package['row'], package['worlds']) != row['configuration']
            or list(package['target_names']) != row['target_names']):
        raise ValueError('Authenticated original configuration or fixed targets differ from selected input')
    return {'version': codec.VERSION, 'selection_reference': selection['reference'],
            'selected_entry': row, 'scope': selection['scope']}


def validate_entries(selection, manifest, *, root=ROOT):
    if manifest.get('partition') != 'test':
        raise ValueError('Compression is only for new selected test exports')
    seen = set()
    for entry in manifest['entries']:
        row = selected_entry(selection, source=entry['source'], pair=entry['pair'], scene=entry['scene'],
                             proposal_identity=entry['proposal_identity'], root=root)
        key = (row['configuration']['configuration_id'], row['kind'])
        if key in seen:
            raise ValueError('Repeated selected configuration/class in new test exports')
        seen.add(key)


def arguments(parser, *, dual=True):
    parser.add_argument('--compress-test-tables', action='store_true')
    parser.add_argument('--test-selection', help='Exact selected report + logged-result JSON reference')
    if dual:
        parser.add_argument('--compression-smoke-dual', action='store_true',
                            help='Save plain and compressed from the same renderer, for the one-pair engineering smoke')


def option_values(args):
    enabled = bool(getattr(args, 'compress_test_tables', False))
    dual = bool(getattr(args, 'compression_smoke_dual', False))
    raw = getattr(args, 'test_selection', None)
    if enabled != (raw is not None) or (dual and not enabled):
        raise ValueError('Compression/selection must be explicit together; dual smoke requires compression')
    return selection_reference(raw) if enabled else None


def command_suffix(reference):
    return [] if reference is None else ['--compress-test-tables', '--test-selection',
                                         json.dumps(selection_reference(reference), sort_keys=True, separators=(',', ':'))]


def compare_tables(plain, compressed):
    """Byte/graph equality proves identical tau and seen predicates on every path.

    This is a serialization comparison, not cache authorization; flags remain false.
    """
    fields = ('pair_id', 'scene', 'kind', 'instruction', 'anchor_name', 'candidate_names',
              'target_names', 'initial_state', 'positions', 'labels', 'source', 'states',
              'export_measurements')
    for name in fields:
        if getattr(plain, name) != getattr(compressed, name):
            raise ValueError('Decoded test table differs: ' + name)
    return {'passed': True, 'states': len(plain.states), 'world_views': 2 * len(plain.states),
        'rgb_bytes_equal': True, 'mask_bytes_equal': True, 'labels_equal': True,
        'tau_and_tau_seen_equal': True,
        'temporal_proof': 'identical initial state, all graph edges, shared flags and anchor pixels; original temporal predicates unchanged',
        'cache_matches_live': None, 'scope': 'serialization equality only; full fresh cache remains required'}

"""Immutable disjoint test tickets shared by one-GPU and packed execution.

The bank partitions an already authenticated full40 plan. Other unit tickets
remain reserved even when queued, unclaimed, failed, or partly collected. This
is capacity accounting only; no reservation is a scientific acceptance.
"""
import copy
import hashlib
import json
from pathlib import Path
import re

from src.tools import g01_quota_plan as quota

VERSION = 'g01_fixed40_disjoint_ticket_bank_v1'
RESERVATION_SCOPE = 'preallocated_disjoint_ticket_upper_bound'
SIZES = (6, 6, 6, 6, 6, 6, 4)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read_reference(reference, root):
    reference = quota.reference(reference)
    root = Path(root).resolve()
    path = (root / reference['file']).resolve()
    if not path.is_relative_to(root) or path.is_symlink():
        raise ValueError('Ticket bank references must remain immutable repository files')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference['sha256']:
        raise ValueError('Ticket bank reference SHA changed')
    return json.loads(raw)


def base_identity(plan):
    """Technical parent renames never create a second scientific ticket."""
    return canonical({key: value for key, value in plan.items() if key != 'parent_run_name'})


def validate_bank(bank, *, root):
    if (not isinstance(bank, dict) or set(bank) != {'version', 'base_plan_reference', 'units'}
            or bank['version'] != VERSION):
        raise ValueError('Unknown or incomplete disjoint test ticket bank')
    base = read_reference(bank['base_plan_reference'], root)
    quota.validate_structure(base)
    expected_scenes = [f'FloorPlan{offset + number}' for number in range(21, 31)
                       for offset in (0, 200, 300, 400)]
    if (base['version'] != quota.CONFIG_VERSION or base['partition'] != 'test'
            or base['outstanding_reservations'] != [] or 'shared_ticket_bank' in base
            or [row['scene'] for row in base['assignments']] != expected_scenes
            or not isinstance(bank['units'], list) or len(bank['units']) != len(SIZES)):
        raise ValueError('Bank requires the original ordered complete40 test configuration plan')
    names, cursor = set(), 0
    for index, (unit, size) in enumerate(zip(bank['units'], SIZES), 1):
        if (set(unit) != {'slice_id', 'parent_run_name', 'scenes'}
                or unit['slice_id'] != f's{index:02d}'
                or not re.fullmatch(r'gen_test_[A-Za-z0-9_]+', unit['parent_run_name'])
                or unit['parent_run_name'] in names
                or unit['scenes'] != expected_scenes[cursor:cursor + size]):
            raise ValueError('Bank slices must be exact unique ordered6/6/6/6/6/6/4 scene assignments')
        names.add(unit['parent_run_name']); cursor += size
    return base


def expected_member(bank, bank_ref, slice_id, *, root):
    base = validate_bank(bank, root=root)
    candidates = [unit for unit in bank['units'] if unit['slice_id'] == slice_id]
    if len(candidates) != 1:
        raise ValueError('Unknown bank slice')
    unit = candidates[0]
    by_scene = {row['scene']: row for row in base['assignments']}
    other = {row['scene']: copy.deepcopy(row['actual_output_caps'])
             for row in base['assignments'] if row['scene'] not in unit['scenes']}
    result = copy.deepcopy(base)
    result.update(parent_run_name=unit['parent_run_name'],
        assignments=[copy.deepcopy(by_scene[scene]) for scene in unit['scenes']],
        outstanding_reservations=[{'reservation_id': base_identity(base) + ':other_slices:' + slice_id,
            'source': quota.reference(bank_ref), 'scope': RESERVATION_SCOPE,
            'by_scene': other, 'by_kind': quota.summed(other.values())}],
        shared_ticket_bank={'version': VERSION, 'reference': quota.reference(bank_ref), 'slice_id': slice_id})
    quota.validate_structure(result)
    return result


def verify_member(plan, *, root, authenticate=False):
    declaration = plan.get('shared_ticket_bank')
    if (not isinstance(declaration, dict) or set(declaration) != {'version', 'reference', 'slice_id'}
            or declaration['version'] != VERSION):
        raise ValueError('Nonempty shared reservations need the exact ticket-bank declaration')
    bank = read_reference(declaration['reference'], root)
    expected = expected_member(bank, declaration['reference'], declaration['slice_id'], root=root)
    if plan != expected:
        raise ValueError('Slice changed original confirmed inventory, assigned tickets, or other-unit reservations')
    base = validate_bank(bank, root=root)
    proof = quota.authenticate_count_basis(base) if authenticate else None
    return {'version': VERSION, 'bank_reference': copy.deepcopy(declaration['reference']),
        'base_plan_reference': copy.deepcopy(bank['base_plan_reference']),
        'allocation_id': base_identity(base), 'slice_id': declaration['slice_id'],
        'count_basis_proof': proof,
        'scope': 'disjoint preallocated upper bounds; no new independent scientific acceptance'}


def scientific_identity(plan, *, root):
    proof = verify_member(plan, root=root, authenticate=False)
    identity = {'version': VERSION, 'allocation_id': proof['allocation_id'],
        'partition': 'test', 'assignments': copy.deepcopy(plan['assignments']),
        'scene_timeout_seconds': 1200, 'concurrency': 6, 'field_of_view': 60, 'seed': 0}
    keys = [canonical({'allocation_id': proof['allocation_id'], 'assignment': row})
            for row in plan['assignments']]
    return identity, sorted(keys)

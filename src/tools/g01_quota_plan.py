"""Immutable, conservative scene/class tickets; no live/global quota service.

Test confirmed counts are reconstructed from successfully published G01
checkpoints and the unchanged full source loader. Training post-T02 counts and
nonempty reservation ledgers remain fail-closed until their real adapters exist.
"""
import copy
import hashlib
import json
from pathlib import Path
import re

from src.envs.thor_common import ROOT

VERSION = 'g01_static_scene_kind_tickets_v1'
KINDS = ('direct_flip', 'direct_stay', 'look_flip', 'look_stay')
FIELDS = ('g01_quota_version', 'g01_quota_assignment', 'allowed_kinds', 'actual_output_caps')
_ACTIVE_COUNT_PLANS = set()

STAGES = {'test': 'independent_unique_construction_before_T02_filtering',
          'train': 'T02_accepted_then_Stay_displacement_and_Look_ratio_0.8_to_1.25'}


def counts(value):
    if (not isinstance(value, dict) or set(value) != set(KINDS)
            or any(type(number) is not int or number < 0 for number in value.values())):
        raise ValueError('Require explicit nonnegative integer counts for every actual class')
    return dict(value)


def reference(value):
    if (not isinstance(value, dict) or set(value) != {'file', 'sha256'}
            or not isinstance(value['file'], str) or not value['file']
            or re.fullmatch(r'[0-9a-f]{64}', str(value['sha256'])) is None):
        raise ValueError('Require an exact immutable file/SHA reference')
    return dict(value)


def scene_allowed(scene, partition):
    low, high = (1, 20) if partition == 'train' else (21, 30)
    return scene in {f'FloorPlan{base + i}' for base in (0, 200, 300, 400) for i in range(low, high + 1)}


def by_scene(value, partition):
    if not isinstance(value, dict) or any(not scene_allowed(scene, partition) for scene in value):
        raise ValueError('Scene counts must stay in the explicit fixed partition')
    return {scene: counts(row) for scene, row in value.items()}


def summed(rows):
    return {kind: sum(row[kind] for row in rows) for kind in KINDS}


def validate_structure(plan):
    if (not isinstance(plan, dict) or plan.get('version') != VERSION
            or plan.get('partition') not in STAGES):
        raise ValueError('Unknown static quota plan/partition')
    partition = plan['partition']
    if not re.fullmatch(r'gen_' + partition + r'_[A-Za-z0-9_]+', str(plan.get('parent_run_name'))):
        raise ValueError('Quota parent name must match the declared partition')
    target = {kind: (60 if partition == 'test' else 0 if kind.startswith('direct_') else 40) for kind in KINDS}
    if counts(plan.get('targets')) != target or plan.get('count_stage') != STAGES[partition]:
        raise ValueError('Quota target or scientific counting stage differs from the approved task')
    confirmed = counts(plan.get('confirmed_by_kind'))
    scene_confirmed = by_scene(plan.get('confirmed_by_scene'), partition)
    if summed(scene_confirmed.values()) != confirmed:
        raise ValueError('Confirmed per-scene totals do not equal the declared global counts')
    basis = plan.get('count_basis_sources')
    if not isinstance(basis, list) or not basis:
        raise ValueError('Even zero confirmed counts require an explicit authenticated inventory cutoff')
    for ref in basis:
        reference(ref)
    reservations = plan.get('outstanding_reservations')
    if not isinstance(reservations, list):
        raise ValueError('Unverified/in-flight output must have an explicit separate reservation ledger')
    ids, reserved_rows, reserved_scenes = set(), [], {}
    for item in reservations:
        identity = item.get('reservation_id')
        if not isinstance(identity, str) or not identity or identity in ids:
            raise ValueError('Duplicate or missing outstanding reservation identity')
        ids.add(identity)
        reference(item.get('source'))
        if item.get('scope') not in ('unverified_inventory_upper_bound', 'submitted_job_upper_bound'):
            raise ValueError('Reservation is an upper bound, never a confirmed scientific pass')
        per_scene = by_scene(item.get('by_scene'), partition)
        declared = counts(item.get('by_kind'))
        if summed(per_scene.values()) != declared:
            raise ValueError('Reservation per-scene/global upper bounds differ')
        reserved_rows.append(declared)
        for scene, row in per_scene.items():
            accumulated = reserved_scenes.setdefault(scene, {kind: 0 for kind in KINDS})
            for kind in KINDS:
                accumulated[kind] += row[kind]
    reserved = summed(reserved_rows)
    remaining = {kind: max(0, target[kind] - confirmed[kind] - reserved[kind]) for kind in KINDS}
    assigned, scenes = [], set()
    rows = plan.get('assignments')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Do not submit a quota plan without any scene tickets')
    cap = 8 if partition == 'train' else 6
    for row in rows:
        scene = row.get('scene')
        if not scene_allowed(scene, partition) or scene in scenes:
            raise ValueError('Each quota scene must occur once in the selected partition')
        scenes.add(scene)
        limits = counts(row.get('actual_output_caps'))
        if row.get('allowed_kinds') != [kind for kind in KINDS if limits[kind] > 0] or not any(limits.values()):
            raise ValueError('Allowed proposal kinds must be exactly the positive scene tickets')
        if partition == 'train' and any(limits[kind] for kind in KINDS if kind.startswith('direct_')):
            raise ValueError('New training collection must not propose or export Direct classes')
        for kind in KINDS:
            available = max(0, cap - scene_confirmed.get(scene, {}).get(kind, 0)
                            - reserved_scenes.get(scene, {}).get(kind, 0))
            if limits[kind] > available:
                raise ValueError('New scene ticket exceeds the original aggregate scene/class capacity')
        assigned.append(limits)
    allocation = summed(assigned)
    if any(allocation[kind] > remaining[kind] for kind in KINDS):
        raise ValueError('Scene tickets exceed conservative global remaining capacity')
    return {'confirmed_by_kind': confirmed, 'reserved_upper_bound_by_kind': reserved,
            'remaining_before_new_tickets': remaining, 'new_tickets_by_kind': allocation,
            'count_stage': STAGES[partition], 'count_basis_authenticated': False,
            'scope': 'structural arithmetic only; reservations are not scientific acceptance'}


def authenticate_count_basis(plan):
    """Recheck every published test pair; raw or failed checkpoint counts never count.

    This grants collection capacity, not image-review or training permission.
    Training requires a separate actual post-T02/Stay/ratio adapter. Nonempty
    reservations require authentic submission/inventory upper-bound certificates;
    trusting caller-supplied integers would not establish global safety.
    """
    if plan['partition'] != 'test':
        raise ValueError('Training scientific count-basis adapter is not implemented; production dispatch is blocked')
    if plan['outstanding_reservations']:
        raise ValueError('Nonempty reservation source adapter is not implemented; production dispatch is blocked')
    from src.envs import pair_table_source as sources
    identities, publications, duplicates, source_refs = {}, [], [], set()
    measured = {}
    for expected in plan['count_basis_sources']:
        reference(expected)
        key = (expected['file'], expected['sha256'])
        if key in source_refs:
            raise ValueError('Duplicate checkpoint in the explicit quota cutoff')
        source_refs.add(key)
        container, actual_ref, entries = sources._container(ROOT / expected['file'])
        if (any(actual_ref.get(field) != expected[field] for field in ('file', 'sha256'))
                or sources._checkpoint_module(container['_meta']['command']) != sources.G01_CHECKPOINT_MODULE):
            raise ValueError('Test quota basis must be the exact successfully published G01 checkpoint')
        if (container['metrics'].get('published_pair_count') != len(entries)
                or container['metrics'].get('status') != 'checkpoint_ready_for_human_review'):
            raise ValueError('Test checkpoint publication is incomplete')
        local_counts = {kind: 0 for kind in KINDS}
        for entry in entries:
            package = sources.load_reviewed_pair(ROOT / expected['file'], entry['pair_id'],
                                                pair_file=entry['pair_reference']['file'])
            kind, scene = package['kind'], package['scene']
            identity = package['independent_verification']['proposal_identity']
            if (kind not in KINDS or not scene_allowed(scene, 'test')
                    or package['source_config'].get('g01_partition') != 'test'
                    or entry.get('proposal_identity') != identity
                    or entry.get('kind') != kind or entry.get('scene') != scene
                    or package['source_provenance']['pair'] != entry['pair_reference']):
                raise ValueError('Published test quota pair differs from its authenticated scene/proposal/class')
            local_counts[kind] += 1
            record = {'proposal_identity': identity, 'scene': scene, 'kind': kind,
                'target_names': copy.deepcopy(package['target_names']),
                'checkpoint_reference': copy.deepcopy(actual_ref),
                'pair_reference': copy.deepcopy(package['source_provenance']['pair']),
                'source_reference': copy.deepcopy(package['source_provenance']['scene'])}
            # Same proposal across retries consumes one ticket. Conflicting
            # independently measured outcomes are not resolved by cherry-picking.
            if identity in identities:
                previous = identities[identity]
                if any(previous[field] != record[field] for field in ('scene', 'kind', 'target_names')):
                    raise ValueError('Duplicate test proposal has conflicting independent outcomes')
                duplicates.append(record)
                continue
            identities[identity] = record
            row = measured.setdefault(scene, {name: 0 for name in KINDS})
            row[kind] += 1
        if container['metrics'].get('published_by_class') != local_counts:
            raise ValueError('Checkpoint publication counts differ from its complete authenticated entries')
        publications.append(copy.deepcopy(actual_ref))
    if measured != plan['confirmed_by_scene'] or summed(measured.values()) != plan['confirmed_by_kind']:
        raise ValueError('Plan confirmed counts differ from independent unique test construction evidence')
    return {'count_basis_authenticated': True, 'count_stage': STAGES['test'],
        'confirmed_by_kind': summed(measured.values()), 'confirmed_by_scene': measured,
        'checkpoint_references': publications, 'unique_proposals': list(identities.values()),
        'duplicate_occurrences': duplicates, 'outstanding_reservations': [],
        'scope': 'independently rechecked test construction; before T02; no human image approval or training permission inferred'}


def load_plan(ref, *, authenticate=True):
    ref = reference(ref)
    path = (ROOT / ref['file']).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('Quota plan must be an explicit repository artifact')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref['sha256']:
        raise ValueError('Immutable quota plan bytes changed')
    plan = json.loads(raw)
    arithmetic = validate_structure(plan)
    proof = None
    if authenticate:
        key = (str(path), ref['sha256'])
        if key in _ACTIVE_COUNT_PLANS:
            raise ValueError('Cyclic quota count-basis dependency')
        _ACTIVE_COUNT_PLANS.add(key)
        try:
            proof = authenticate_count_basis(plan)
        finally:
            _ACTIVE_COUNT_PLANS.remove(key)
    return plan, arithmetic, proof


def assignment(plan, ref, scene, child_run_name):
    validate_structure(plan)
    rows = [row for row in plan['assignments'] if row['scene'] == scene]
    if len(rows) != 1 or not re.fullmatch(re.escape(plan['parent_run_name']) + r'_c[246]_' + re.escape(scene.lower()), child_run_name):
        raise ValueError('Child must match one explicit parent scene assignment')
    return {'version': VERSION, 'plan_reference': reference(ref), 'parent_run_name': plan['parent_run_name'],
            'partition': plan['partition'], 'scene': scene, 'child_run_name': child_run_name,
            'allowed_kinds': copy.deepcopy(rows[0]['allowed_kinds']),
            'actual_output_caps': copy.deepcopy(rows[0]['actual_output_caps'])}


def declared_fields(value):
    if not isinstance(value, dict) or value.get('version') != VERSION:
        raise ValueError('Unknown quota assignment')
    reference(value.get('plan_reference'))
    limits = counts(value.get('actual_output_caps'))
    if value.get('allowed_kinds') != [kind for kind in KINDS if limits[kind] > 0] or not any(limits.values()):
        raise ValueError('Assignment proposal kinds differ from positive actual output tickets')
    return {'g01_quota_version': VERSION, 'g01_quota_assignment': copy.deepcopy(value),
            'allowed_kinds': list(value['allowed_kinds']), 'actual_output_caps': limits}


def validate_assignment(config, *, authenticate=False):
    """Check immutable runtime assignment; authenticate counts at parent/checker.

    A fresh witness process must not rerun the entire historic scientific pool.
    Parent dispatch and independent verify_collection call the authenticated path;
    child policy validation checks exact plan bytes/structure/assignment only.
    """
    value = config.get('g01_quota_assignment')
    fields = declared_fields(value)
    if any(config.get(key) != item for key, item in fields.items()):
        raise ValueError('Source quota fields differ from the explicit assignment')
    plan, arithmetic, proof = load_plan(value['plan_reference'], authenticate=authenticate)
    expected = assignment(plan, value['plan_reference'], config['scene'], config['name'])
    if value != expected or config.get('g01_partition') != plan['partition']:
        raise ValueError('Source differs from its immutable quota scene/partition')
    return fields, arithmetic, proof

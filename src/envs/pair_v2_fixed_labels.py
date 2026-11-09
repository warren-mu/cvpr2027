"""Opt-in supplement 13 labels; historical spatial-label code stays unchanged.

Only the immutable pair ID and stable candidate identities choose numbers. A
spawned replay must explicitly restore the same declared context. This module
never rewrites old pair evidence or silently upgrades its label policy.
"""
import copy
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from . import pair_v2_policy as legacy
from .pair_table import label_assignment

VERSION = 'pair_stable_object_labels_v1'
POLICY = {
    'version': VERSION, 'approved_task_commit': '5a50999',
    'assignment': 'existing T02 label_assignment(pair_id, stable candidate names)',
    'scope': 'one object keeps its number across both worlds and all views of one pair',
    'visible_subset': 'draw only currently visible declared candidates; never renumber a subset',
    'public_identity': 'numbers and mask-derived geometry only; stable object names remain private',
    'legacy': 'absent policy means the original spatial policy; no retrospective upgrades',
}
FIELDS = ('fixed_label_version', 'fixed_label_policy')
PAIR_FIELDS = (*FIELDS, 'fixed_label_pair_id', 'candidate_label_assignment')
_LEGACY_MARK = legacy.mark_observation
_CONTEXT = ContextVar('t01v2_fixed_pair_labels', default=None)


def fields():
    return {'fixed_label_version': VERSION, 'fixed_label_policy': copy.deepcopy(POLICY)}


def validate_fields(value):
    if any(value.get(key) != expected for key, expected in fields().items()):
        raise ValueError('Missing or altered explicitly declared fixed-object label policy')


def pair_fields(pair_id, candidate_names):
    names = tuple(candidate_names)
    if not isinstance(pair_id, str) or not pair_id or any(not isinstance(name, str) or not name for name in names):
        raise ValueError('Fixed labels require an explicit pair ID and stable object names')
    return {**fields(), 'fixed_label_pair_id': pair_id,
            'candidate_label_assignment': label_assignment(pair_id, names)}


def validate_pair(value, pair_id, candidate_names):
    expected = pair_fields(pair_id, candidate_names)
    assignment = value.get('candidate_label_assignment')
    if (not isinstance(assignment, dict) or any(type(label) is not int for label in assignment.values())
            or any(value.get(key) != item for key, item in expected.items())):
        raise ValueError('Fixed-label pair identity or stable candidate assignment differs')
    return copy.deepcopy(expected['candidate_label_assignment'])


@contextmanager
def pair_context(pair_id, candidate_names, *, declared):
    validate_fields(declared)
    expected = pair_fields(pair_id, candidate_names)
    if any(key in declared for key in PAIR_FIELDS[2:]):
        validate_pair(declared, pair_id, candidate_names)
    token = _CONTEXT.set((pair_id, tuple(sorted(candidate_names)), expected['candidate_label_assignment']))
    try:
        yield copy.deepcopy(expected)
    finally:
        _CONTEXT.reset(token)


def mark_observation(capture, candidate_names, instruction, *, public_context='', history=()):
    """Reuse actual mask geometry, replacing only the visible label numbering."""
    from PIL import Image, ImageDraw
    import numpy as np
    context = _CONTEXT.get()
    if context is None or tuple(sorted(candidate_names)) != context[1]:
        raise ValueError('Fixed marks require their explicit matching pair/candidate context')
    # The legacy routine still validates masks and supplies its exact box/anchor
    # geometry. Its spatially numbered pixels are not retained or modified.
    result = _LEGACY_MARK(capture, candidate_names, instruction,
                          public_context=public_context, history=history)
    old_marks = {item['label']: item for item in result['marks']}
    labels = {name: context[2][name] for name in result['mark_labels_by_name']}
    marks = []
    for name, old_label in result['mark_labels_by_name'].items():
        marks.append({**copy.deepcopy(old_marks[old_label]), 'label': labels[name]})
    marks.sort(key=lambda item: item['label'])
    image = Image.fromarray(result['rgb'])
    draw = ImageDraw.Draw(image)
    for item in marks:
        anchor, text = tuple(item['anchor_pixel']), str(item['label'])
        draw.rectangle(draw.textbbox(anchor, text), fill='white')
        draw.text(anchor, text, fill='black')
    result.update(marked=np.asarray(image).copy(), marks=marks, mark_labels_by_name=labels)
    return result


@contextmanager
def installed():
    """Process-local only; new producers/checkers opt in explicitly."""
    previous = legacy.mark_observation
    legacy.mark_observation = mark_observation
    try:
        yield
    finally:
        legacy.mark_observation = previous


def verify_witness_labels(row, replay, read):
    """Rebuild terminal selection marks from the actual native capture bytes."""
    from .pair_v2_solvable_export import capture_observation
    names = row['identities']['candidate_names']
    expected = validate_pair(row, row['pair_id'], names)
    validate_pair(replay, row['pair_id'], names)
    with installed(), pair_context(row['pair_id'], names, declared=row):
        terminal = capture_observation(replay['captures'][-1], read, row['identities'], row['instruction'])
    action = replay['actions'][-1]
    selected = replay['selected_name']
    if (action.get('kind') != 'select' or selected not in expected
            or action.get('selected_object_name') != selected
            or action.get('label') != expected[selected]
            or action.get('mark_labels_by_name') != terminal['mark_labels_by_name']
            or terminal['mark_labels_by_name'].get(selected) != expected[selected]
            or replay['planned_actions'][-1] != f'select {expected[selected]}'
            or replay['executed_actions'][-1] != f'select {expected[selected]}'):
        raise ValueError('Witness selection does not use its fixed physical-object label')
    return {'pair_id': row['pair_id'], 'assignment': expected,
            'terminal_visible_assignment': terminal['mark_labels_by_name'], 'selected_name': selected}


def checked_pair(original_check, row, pair_path, *, declared_source, **kwargs):
    """New-source adapter; all original scientific/provenance checks still run."""
    from src.tools import merge_pair_v2 as merge
    validate_fields(declared_source)
    names = row['identities']['candidate_names']
    validate_pair(row, row['pair_id'], names)
    with installed(), pair_context(row['pair_id'], names, declared=row):
        checked = original_check(row, pair_path, **kwargs)
    root = Path(kwargs.get('root', merge.ROOT)).resolve()
    raw_root = kwargs.get('raw_root')
    if raw_root is None:
        raise ValueError('Fixed-label witness validation requires its authenticated raw root')
    portable = {Path(ref['file']).name: ref for ref in row['portable_files']}
    read = lambda ref: merge.reference_path(ref, root=root, within=raw_root).read_bytes()
    witnesses = []
    for index in (0, 1):
        replay = merge.read_reference(portable[f'witness_w{index}.json'], root=root,
                                      within=Path(pair_path).resolve().parent)
        witnesses.append(verify_witness_labels(row, replay, read))
    return {**checked, 'fixed_labels_verified': {**pair_fields(row['pair_id'], names),
                                               'witnesses': witnesses}}


def validate_table_package(package):
    """Check new source/T02 mapping agreement without changing historical data."""
    row, config = package['row'], package['source_config']
    present = [any(key in value for key in PAIR_FIELDS) for value in (row, config)]
    if not any(present):
        return {'fixed_label_version': None, 'scope': 'historical source; no fixed T01 label claim'}
    if not all(present):
        raise ValueError('Table source and pair disagree on fixed-label version')
    validate_fields(config)
    expected = validate_pair(row, package['pair_id'], package['candidate_names'])
    if row['pair_id'] != package['pair_id'] or sorted(row['identities']['candidate_names']) != sorted(package['candidate_names']):
        raise ValueError('Table candidate identities differ from its fixed-label source pair')
    return {**pair_fields(package['pair_id'], package['candidate_names']),
            't01_t02_assignment_equal': expected == label_assignment(package['pair_id'], package['candidate_names'])}

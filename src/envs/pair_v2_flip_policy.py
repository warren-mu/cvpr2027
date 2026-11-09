"""Explicit supplement13/14 single-scene Look iteration at the selected60 FOV."""
import copy
from contextlib import contextmanager
from . import pair_v2_fov as fov
from . import pair_v2_stay_order as stay
from . import pair_v2_flip_search as search
from . import pair_v2_fixed_labels as labels

VERSION = 'look_flip_single_scene_20min_selected60_v1'
PRODUCER = search.PRODUCER
POLICY = {
    'version': VERSION, 'approved_task_commit': '47ea528',
    'search_approval': '5a50999', 'field_of_view': 60,
    'camera_selection': 'completed same-scene supplement12 comparison; selected60; do not repeat the contrast',
    'unit_gpu_count': 1, 'unit_timeout_seconds': 1200,
    'parent_timeout_seconds': 1350, 'gate_timeout_seconds': 120,
    'allocation_walltime_seconds': 1380,
    'timing_scope': 'one full1200-second scene grant; native deadline1190s; gate and finalization separate',
    'partial_grant_policy': 'fail before launch if full1200-second child grant is unavailable',
    'scene_scope': 'one explicit scene from the fixed40 training scenes',
    'minimum_eligible_attempts': None,
    'search': search.VERSION, 'fixed_labels': labels.VERSION,
    'scope': 'new-version independent acceptance only; historical spatial-label pairs are not upgraded',
}
FIELDS = ('flip_unit_version', 'flip_unit_policy', 'flip_unit_scene', 'flip_unit_plan')
_ORIGINAL_DECLARED = fov.declared_fields


def camera_fields(camera, scene):
    allowed = {f'FloorPlan{i}' for i in (*range(1, 21), *range(201, 221))}
    if type(camera) is not int or camera != 60 or scene not in allowed:
        raise ValueError('New Flip units require selected60 and one fixed training scene')
    return {'field_of_view': camera, 'fov_contrast_version': VERSION,
        'fov_contrast_policy': copy.deepcopy(POLICY), 'fov_contrast_group': 'selected60_flip_unit',
        'flip_unit_version': VERSION, 'flip_unit_policy': copy.deepcopy(POLICY),
        'flip_unit_scene': scene,
        'flip_unit_plan': [{'scene': scene, 'field_of_view': camera, 'gpu_count': 1, 'timeout_seconds': 1200}],
        **search.fields(camera), **labels.fields()}


def declared_fields(config):
    if config.get('fov_contrast_version') != VERSION and not any(key in config for key in FIELDS):
        return _ORIGINAL_DECLARED(config)
    expected = camera_fields(config.get('field_of_view'), config.get('flip_unit_scene'))
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('Missing/altered selected-camera Flip unit policy')
    scenes = [config['scene']] if 'scene' in config else config.get('pilot_scenes', [])
    if scenes != [expected['flip_unit_scene']]:
        raise ValueError('Flip source differs from its single explicitly selected scene')
    return expected


def validate_source(config, metrics=None):
    actual, shard = declared_fields(config), declared_fields(config['shard'])
    if (actual.get('flip_unit_version') != VERSION or actual != shard
            or config.get('producer_module') != PRODUCER or config['shard'].get('producer_module') != PRODUCER):
        raise ValueError('Flip source differs from immutable opt-in producer/scene/camera')
    for value in (config, config['shard']):
        stay.validate_fields(value); search.validate_fields(value); labels.validate_fields(value)
    if metrics is not None:
        if any(metrics.get(key) != value for key, value in actual.items()) or metrics.get('producer_module') != PRODUCER:
            raise ValueError('Flip result differs from declared source unit')
        stay.validate_fields(metrics); search.validate_fields(metrics); labels.validate_fields(metrics)


@contextmanager
def camera_support():
    previous = fov.declared_fields
    fov.declared_fields = declared_fields
    try:
        yield
    finally:
        fov.declared_fields = previous

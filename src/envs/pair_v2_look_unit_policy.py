"""Explicit supplement 12 bounded, same-scene Look-only camera experiment."""
import copy
from contextlib import contextmanager
from . import pair_v2_fov as fov
from . import pair_v2_stay_order as stay

VERSION = 'look_only_same_scene_20min_units_v1'
PRODUCER = 'src.envs.pair_v2_look_unit_shard'
# The kitchen is supplied once by the human-authorized historical-attempt choice
# and carried unchanged through the parent, immutable shard and source result.
POLICY = {
    'version': VERSION, 'approved_task_commit': 'be43fc4',
    'scenes': 'FloorPlan203, FloorPlan204 and one explicitly recorded historical-highest-attempt kitchen',
    'cameras': [100, 60], 'assignment': 'both cameras on each of the same three scenes; overrides parity only for this explicit experiment',
    'search': 'Look projection pool only; both Flip and Stay; actual non-Look prefix is rejected before the Direct branch',
    'stay_search_order_version': stay.VERSION,
    'unit_gpu_count': 1, 'unit_timeout_seconds': 1200,
    'allocation_gpu_count': 1, 'allocation_walltime_seconds': 3900,
    'unit_scene_order': 'FloorPlan203, FloorPlan204, then the recorded kitchen',
    'parent_timeout_seconds': 3870, 'gate_timeout_seconds': 120, 'scene_timeout_seconds': 1200,
    'timing_scope': 'three sequential full1200-second scene grants on one GPU; native scene deadline1190s; gate and finalization overhead are separate within the65-minute allocation',
    'partial_grant_policy': 'do not start a unit if less than1200 seconds can be granted; retain completed units and report the unstarted one',
    'minimum_eligible_attempts': None,
    'unchanged': 'seed0; safe native retries; frozen render; physical, target, prefix, evidence, solvability and independent witness thresholds',
    'unreached_measurements': 'null/unknown; preserve physical-first execution order',
}
FIELDS = ('look_unit_version', 'look_unit_policy', 'look_unit_kitchen_scene', 'look_unit_plan')
_ORIGINAL_DECLARED = fov.declared_fields


def unit_fields(kitchen):
    if kitchen not in {f'FloorPlan{i}' for i in range(1, 21)}:
        raise ValueError('Look unit kitchen must be an explicitly selected training kitchen')
    return {'look_unit_version': VERSION, 'look_unit_policy': copy.deepcopy(POLICY),
        'look_unit_kitchen_scene': kitchen,
        'look_unit_plan': [{'scene': scene, 'field_of_view': camera, 'gpu_count': 1, 'timeout_seconds': 1200}
                           for scene in ('FloorPlan203', 'FloorPlan204', kitchen) for camera in (100, 60)]}


def camera_fields(camera, kitchen):
    if type(camera) is not int or camera not in (60, 100):
        raise ValueError('Look units require exactly60 or100 degrees')
    return {'field_of_view': camera, 'fov_contrast_version': VERSION,
        'fov_contrast_policy': copy.deepcopy(POLICY),
        'fov_contrast_group': f'look_unit_{camera}', **unit_fields(kitchen)}


def declared_fields(config):
    if (config.get('fov_contrast_version') != VERSION
            and not any(key in config for key in FIELDS)):
        return _ORIGINAL_DECLARED(config)
    expected = camera_fields(config.get('field_of_view'), config.get('look_unit_kitchen_scene'))
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('Altered or incomplete Look-only unit/FOV policy')
    scenes = [config['scene']] if 'scene' in config else config.get('pilot_scenes', [])
    allowed = {item['scene'] for item in expected['look_unit_plan']}
    if not scenes or any(scene not in allowed for scene in scenes):
        raise ValueError('Look-only source is outside its fixed same-scene unit plan')
    return expected


def validate_source(config, metrics=None):
    actual, shard = declared_fields(config), declared_fields(config['shard'])
    if actual != shard or config.get('producer_module') != PRODUCER or config['shard'].get('producer_module') != PRODUCER:
        raise ValueError('Look-only source differs from its immutable unit/FOV assignment')
    stay.validate_fields(config); stay.validate_fields(config['shard'])
    if metrics is not None:
        if any(metrics.get(key) != value for key, value in actual.items()) or metrics.get('producer_module') != PRODUCER:
            raise ValueError('Look-only source metrics differ from declared unit/FOV policy')
        stay.validate_fields(metrics)


@contextmanager
def camera_support():
    previous = fov.declared_fields
    fov.declared_fields = declared_fields
    try:
        yield
    finally:
        fov.declared_fields = previous

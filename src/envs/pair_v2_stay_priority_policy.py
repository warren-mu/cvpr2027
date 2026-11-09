"""Independent twenty-minute Stay-priority trial over the full ranked source."""
import copy
from contextlib import contextmanager
from . import pair_v2_fov as fov
from . import pair_v2_flip_ranked_policy as ranked
from . import pair_v2_stay_priority as priority

VERSION = 'look_stay_min_world_predicted_area_priority_selected60_v1'
PRODUCER = 'src.envs.pair_v2_stay_priority_shard'
POLICY = {**copy.deepcopy(ranked.POLICY), 'version': VERSION,
    'previous_source_version': ranked.VERSION, 'stay_priority': priority.VERSION,
    'approval': 'human-authorized independent Stay-order trial after the recorded supplement15 diagnosis',
    'scope': 'existing pending Stay domain order only; full original ranked geometry/physics/render/witness checks unchanged'}
FIELDS = (*ranked.FIELDS, *priority.FIELDS)


def camera_fields(camera, scene):
    if scene not in ('FloorPlan203', 'FloorPlan204'):
        raise ValueError('Stay priority trial is restricted to the two stated training scenes')
    result = ranked.camera_fields(camera, scene)
    result.update(fov_contrast_version=VERSION, fov_contrast_policy=copy.deepcopy(POLICY),
                  flip_unit_version=VERSION, flip_unit_policy=copy.deepcopy(POLICY))
    return {**result, **priority.fields()}


def declared_fields(config):
    if config.get('fov_contrast_version') != VERSION and not any(key in config for key in priority.FIELDS):
        return ranked._ORIGINAL_DECLARED(config)
    expected = camera_fields(config.get('field_of_view'), config.get('flip_unit_scene'))
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('Missing/altered explicit Stay priority source policy')
    scenes = [config['scene']] if 'scene' in config else config.get('pilot_scenes', [])
    if scenes != [expected['flip_unit_scene']]:
        raise ValueError('Stay priority trial differs from its single planned scene')
    return expected


def validate_source(config, metrics=None):
    actual, shard = declared_fields(config), declared_fields(config['shard'])
    if (actual.get('flip_unit_version') != VERSION or actual != shard
            or config.get('producer_module') != PRODUCER or config['shard'].get('producer_module') != PRODUCER):
        raise ValueError('Stay priority source requires its own immutable producer identity')
    modules = (ranked.stay, ranked.search, ranked.labels, ranked.cadence,
               ranked.sharing, ranked.rank, ranked.geometry, priority)
    for value in (config, config['shard']):
        for module in modules:
            module.validate_fields(value)
    if metrics is not None:
        if any(metrics.get(key) != value for key, value in actual.items()) or metrics.get('producer_module') != PRODUCER:
            raise ValueError('Stay priority measured result differs from the declared source')
        for module in modules:
            module.validate_fields(metrics)


@contextmanager
def camera_support():
    previous = fov.declared_fields
    fov.declared_fields = declared_fields
    try:
        yield
    finally:
        fov.declared_fields = previous

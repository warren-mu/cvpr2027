"""Explicit G01 mixed source policy; no evaluation scene or clone fallback."""
import copy
from contextlib import contextmanager
from . import pair_v2_fov as fov
from . import pair_v2_stay_order as stay
from . import pair_v2_fixed_labels as labels
from . import pair_v2_g01_search as search
from . import pair_v2_flip_rounds_search as cadence
from . import pair_v2_spawn_sharing as sharing
from . import pair_v2_bbox_rank as rank
from . import pair_v2_flip_geometry as geometry

VERSION = 'g01_mixed_train80_complete_look_geometry_selected60_v2'
PRODUCER = 'src.envs.pair_v2_g01_shard'
TRAIN_SCENES = frozenset(f'FloorPlan{i}' for start in (1, 201, 301, 401) for i in range(start, start + 20))
POLICY = {
    'version': VERSION, 'task': 'G01', 'approval': 'supplement14; execution still requires the separate observed gate and budget check',
    'partition': 'training only:1–20/201–220/301–320/401–420', 'field_of_view': 60,
    'candidates': 'approved natural two-marked-pickupables; cloning disabled',
    'classes': list(search.KINDS), 'per_scene_class_cap': 8,
    'scene_timeout_seconds': 1200, 'cpu_cores_per_gpu': 12,
    'source_instances_max': 2, 'resource_scope': 'one construction plus at most one sequential witness; aggregate multi-lane limit requires a separate parent policy',
    'fixed_labels': labels.VERSION, 'search': search.VERSION,
    'look_flip_geometry': geometry.VERSION, 'geometry_scope': 'Look-pool Flip proposals only, regardless of final measured class; Direct-pool Flip and all Stay retain their previous domains',
    'actual_class': 'unchanged rendered prefix/evidence classification may cross proposal pools; class caps use measured kind while Look attempt balance uses proposed_pool and mode',
    'native_token_storage': sharing.VERSION, 'discovery_schedule': cadence.VERSION,
    'look_proposal_ordering': rank.VERSION,
    'minimum_eligible_attempts': None, 'scope': 'new producer outputs only; no retrospective upgrade of historical label or search rules',
}
FIELDS = ('g01_version', 'g01_policy', 'g01_partition')
_ORIGINAL_DECLARED = fov.declared_fields


def fields():
    return {'g01_version': VERSION, 'g01_policy': copy.deepcopy(POLICY), 'g01_partition': 'train'}


def declared(scene):
    if scene not in TRAIN_SCENES:
        raise ValueError('G01 mixed producer is restricted to the fixed80 training scenes')
    return {'field_of_view': 60, 'fov_contrast_version': VERSION,
        'fov_contrast_policy': copy.deepcopy(POLICY), 'fov_contrast_group': 'g01_selected60',
        **fields(),
        **search.fields(), **labels.fields(), **cadence.fields(),
        **sharing.fields(), **rank.fields(), **geometry.fields()}


def declared_fields(config):
    if config.get('fov_contrast_version') != VERSION and not any(key in config for key in FIELDS):
        return _ORIGINAL_DECLARED(config)
    expected = declared(config.get('scene'))
    if any(config.get(key) != item for key, item in expected.items()):
        raise ValueError('Missing/altered explicit G01 camera/partition/search policy')
    return expected


def validate_source(config, metrics=None):
    expected, shard = declared_fields(config), declared_fields(config['shard'])
    if (expected.get('g01_version') != VERSION or expected != shard or config['scene'] != config['shard']['scene']
            or config.get('producer_module') != PRODUCER or config['shard'].get('producer_module') != PRODUCER):
        raise ValueError('G01 source differs from its immutable training scene/producer policy')
    if config.get('enable_candidate_clone') is not False or config.get('scene_pairs_per_class') != search.PER_CLASS:
        raise ValueError('G01 source must retain natural candidates and the task scene/class cap8')
    for value in (config, config['shard']):
        stay.validate_fields(value); labels.validate_fields(value); search.validate_fields(value)
        for module in (cadence, sharing, rank, geometry):
            module.validate_fields(value)
        if any(value.get(key) is not None for key in ('look_unit_version', 'look_unit_policy',
                'flip_unit_version', 'flip_unit_policy', 'flip_search_version', 'flip_search_policy')):
            raise ValueError('Mixed G01 source cannot also claim a historical Look-only producer policy')
    if metrics is not None:
        if any(metrics.get(key) != item for key, item in expected.items()) or metrics.get('producer_module') != PRODUCER:
            raise ValueError('G01 measured result differs from declared source policy')
        if metrics.get('enable_candidate_clone') is not False:
            raise ValueError('G01 measured source claims a nonnatural candidate policy')
        stay.validate_fields(metrics); labels.validate_fields(metrics); search.validate_fields(metrics)
        for module in (cadence, sharing, rank, geometry):
            module.validate_fields(metrics)


@contextmanager
def camera_support():
    previous = fov.declared_fields
    fov.declared_fields = declared_fields
    try:
        yield
    finally:
        fov.declared_fields = previous

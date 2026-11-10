"""Explicit G01 mixed source policy; explicit isolated training/evaluation partitions; no clone fallback."""
import copy
import re
from contextlib import contextmanager
from . import pair_v2_fov as fov
from . import pair_v2_stay_order as stay
from . import pair_v2_fixed_labels as labels
from . import pair_v2_g01_search as search
from . import pair_v2_flip_rounds_search as cadence
from . import pair_v2_spawn_sharing as sharing
from . import pair_v2_bbox_rank as rank
from . import pair_v2_flip_geometry as geometry
from . import pair_v2_nearby_physics as nearby_physics
from . import pair_v2_raw_sampling as raw_sampling

VERSION = 'g01_mixed_train80_complete_look_geometry_selected60_v2'
TEST_VERSION = 'g01_mixed_test40_frozen_generator_selected60_v1'
QUOTA_VERSIONS = {'train': 'g01_train_static_tickets_selected60_v1',
                  'test': 'g01_test_static_tickets_selected60_v1'}
CONFIG_QUOTA_VERSION = 'g01_test_configuration_tickets_selected60_v2'
from src.tools import g01_quota_plan as quota_plan
PRODUCER = 'src.envs.pair_v2_g01_shard'
TRAIN_SCENES = frozenset(f'FloorPlan{i}' for start in (1, 201, 301, 401) for i in range(start, start + 20))
TEST_SCENES = frozenset(f'FloorPlan{i}' for start in (21, 221, 321, 421) for i in range(start, start + 10))
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
SMOKE = {'version': 'g01_s16_single_scene_600s_v1', 'scene_timeout_seconds': 600,
    'formal_collection_eligible': False, 'purpose': 'engineering smoke only; no quota occupancy'}
BASE_VERSIONS = (VERSION, TEST_VERSION, *QUOTA_VERSIONS.values(), CONFIG_QUOTA_VERSION)
NEARBY_VERSIONS = tuple(v + '_nearby_v1' for v in BASE_VERSIONS)
SMOKE_VERSION = CONFIG_QUOTA_VERSION + '_nearby_v1_smoke600_v1'
UNSAMPLED_VERSIONS = (*BASE_VERSIONS, *NEARBY_VERSIONS, SMOKE_VERSION)
ALL_VERSIONS = (*UNSAMPLED_VERSIONS, *(v + "_sampled_raw_v1" for v in UNSAMPLED_VERSIONS))
_ORIGINAL_DECLARED = fov.declared_fields


def split_scenes(partition):
    if partition not in ('train', 'test'):
        raise ValueError('Require explicit train/test partition')
    if TRAIN_SCENES & TEST_SCENES:
        raise ValueError('Training and evaluation scene inventories overlap')
    return TRAIN_SCENES if partition == 'train' else TEST_SCENES


def class_cap(partition):
    split_scenes(partition)
    return 8 if partition == 'train' else 6


def fields(partition='train', quota=False, configuration=False, nearby=False, smoke=False, sampled=False):
    split_scenes(partition)
    spec = copy.deepcopy(POLICY)
    if partition == 'test':
        spec.update(version=TEST_VERSION,
            approval='human G01/T03 revision 2026-11-09; fixed generator, no evaluation-driven tuning',
            partition='evaluation only:21–30/221–230/321–330/421–430',
            per_scene_class_cap=6,
            purpose='frozen evaluation collection only; never training, tuning or generator pilots')
    if quota:
        spec.update(version=QUOTA_VERSIONS[partition], approval='cef90e80 explicit static scene/class tickets',
            classes='explicit allowed_kinds in the immutable scene assignment',
            search=search.QUOTA_VERSION, quota=quota_plan.VERSION,
            actual_class='unchanged scientific classification; only actual classes with reserved tickets are exported')
    if configuration:
        if partition != 'test' or not quota:
            raise ValueError('Configuration collection requires explicit test tickets')
        spec.update(version=CONFIG_QUOTA_VERSION, approval='83828652 and 002d747c human approved',
            proposal_priority=list(search.CONFIG_PRIORITY),
            discovery_schedule_scope='base finite discovery rounds; explicit configuration search policy overrides within-round domain preference only',
            quota=quota_plan.CONFIG_VERSION, search=search.CONFIG_VERSION,
            configuration=copy.deepcopy(quota_plan.configurations.POLICY))
    extra = {}
    if nearby:
        spec.update(version=spec['version'] + '_nearby_v1', physics=copy.deepcopy(nearby_physics.POLICY),
            physics_approval='38b15551 supplement16; explicit new production version')
        extra.update(nearby_physics.fields())
    if smoke:
        if not (partition == 'test' and quota and configuration and nearby):
            raise ValueError('Engineering smoke requires explicit test configuration tickets and supplement16')
        spec.update(version=SMOKE_VERSION, scene_timeout_seconds=600, purpose=SMOKE['purpose'])
        extra['g01_engineering_smoke'] = copy.deepcopy(SMOKE)
    if sampled:
        spec.update(version=spec['version'] + '_sampled_raw_v1', raw_sampling=copy.deepcopy(raw_sampling.POLICY))
        extra.update(raw_sampling.fields())
    return {'g01_version': spec['version'], 'g01_policy': spec, 'g01_partition': partition, **extra}


def declared(scene, partition='train', quota_assignment=None, nearby=False, smoke=False, sampled=False):
    if scene not in split_scenes(partition):
        raise ValueError('G01 scene is outside its explicitly declared train/test partition')
    configuration = quota_assignment is not None and quota_assignment.get('version') == quota_plan.CONFIG_VERSION
    values = fields(partition, quota_assignment is not None, configuration, nearby, smoke, sampled)
    quota_fields = {} if quota_assignment is None else quota_plan.declared_fields(quota_assignment)
    return {'field_of_view': 60, 'fov_contrast_version': values['g01_version'],
        'fov_contrast_policy': copy.deepcopy(values['g01_policy']), 'fov_contrast_group': 'g01_selected60',
        **values,
        **quota_fields, **search.fields(quota_assignment is not None, configuration), **labels.fields(), **cadence.fields(),
        **sharing.fields(), **rank.fields(), **geometry.fields()}


def declared_fields(config):
    if config.get('fov_contrast_version') not in ALL_VERSIONS and not any(key in config for key in FIELDS):
        return _ORIGINAL_DECLARED(config)
    ticketed = any(key in config for key in quota_plan.FIELDS)
    expected = declared(config.get('scene'), config.get('g01_partition'),
                        config.get('g01_quota_assignment') if ticketed else None,
                        config.get('nearby_physics_version') is not None,
                        config.get('g01_engineering_smoke') is not None,
                        config.get('raw_sampling_version') is not None)
    if ticketed and config.get('g01_quota_assignment') is None:
        raise ValueError('Partial quota fields cannot upgrade an old source')
    if any(key in config for key in nearby_physics.FIELDS) and config.get('nearby_physics_version') is None:
        raise ValueError('Partial nearby physics declaration cannot upgrade historical source')
    if any(key in config for key in raw_sampling.FIELDS):
        raw_sampling.validate_fields(config)
    if any(config.get(key) != item for key, item in expected.items()):
        raise ValueError('Missing/altered explicit G01 camera/partition/search policy')
    return expected


def validate_source(config, metrics=None):
    expected, shard = declared_fields(config), declared_fields(config['shard'])
    if (expected.get('g01_version') not in ALL_VERSIONS or expected != shard or config['scene'] != config['shard']['scene']
            or config.get('producer_module') != PRODUCER or config['shard'].get('producer_module') != PRODUCER):
        raise ValueError('G01 source differs from its immutable train/test scene/producer policy')
    if config.get('enable_candidate_clone') is not False or config.get('scene_pairs_per_class') != class_cap(expected['g01_partition']):
        raise ValueError('G01 source must retain natural candidates and the declared partition scene/class cap')
    if expected['g01_partition'] == 'test':
        if (not re.fullmatch(r'gen_test_[A-Za-z0-9_]+', config.get('name', ''))
                or config['shard'].get('name') != config['name']):
            raise ValueError('Evaluation source requires its isolated gen_test_ namespace')
    if config.get('g01_quota_version') is not None:
        quota_plan.validate_assignment(config)
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

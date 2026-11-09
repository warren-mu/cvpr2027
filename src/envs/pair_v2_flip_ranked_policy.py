"""Explicit combined Flip source: prior cadence, exact native values, box ranking."""
import copy
from contextlib import contextmanager
from . import pair_v2_fov as fov
from . import pair_v2_flip_rounds_policy as rounds
from . import pair_v2_flip_rounds_search as cadence
from . import pair_v2_flip_search as search
from . import pair_v2_fixed_labels as labels
from . import pair_v2_stay_order as stay
from . import pair_v2_spawn_sharing as sharing
from . import pair_v2_bbox_rank as rank

VERSION = 'look_flip_rounds_shared_native_ranked_box_selected60_v1'
PRODUCER = 'src.envs.pair_v2_flip_ranked_shard'
POLICY = {**copy.deepcopy(rounds.POLICY), 'version': VERSION,
    'previous_source_version': rounds.VERSION,
    'native_token_storage': sharing.VERSION, 'look_proposal_ordering': rank.VERSION,
    'proposal_domain': 'build the complete prior native domain first; rank only Look entries; preserve all tokens, Direct and historical Stay domains',
    'rank_deadline': 'check the original config.deadline before every ranked point; timeout propagates without a partial pool',
    'scope': 'new producer only; value-equivalent native storage and predicted ordering; all prior physical/rendered/witness acceptance checks remain unchanged'}
FIELDS = (*rounds.FIELDS, *sharing.FIELDS, *rank.FIELDS)
_ORIGINAL_DECLARED = rounds._ORIGINAL_DECLARED


def camera_fields(camera, scene):
    result = rounds.camera_fields(camera, scene)
    result.update(fov_contrast_version=VERSION, fov_contrast_policy=copy.deepcopy(POLICY),
                  flip_unit_version=VERSION, flip_unit_policy=copy.deepcopy(POLICY))
    result.update(sharing.fields(), **rank.fields())
    return result


def declared_fields(config):
    if config.get('fov_contrast_version') != VERSION and not any(key in config for key in FIELDS):
        return _ORIGINAL_DECLARED(config)
    expected = camera_fields(config.get('field_of_view'), config.get('flip_unit_scene'))
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('Missing/altered combined Flip camera/cadence/storage/rank policy')
    scenes = [config['scene']] if 'scene' in config else config.get('pilot_scenes', [])
    if scenes != [expected['flip_unit_scene']]:
        raise ValueError('Combined Flip source differs from its explicit training scene')
    return expected


def validate_source(config, metrics=None):
    actual, shard = declared_fields(config), declared_fields(config['shard'])
    if (actual.get('flip_unit_version') != VERSION or actual != shard
            or config.get('producer_module') != PRODUCER or config['shard'].get('producer_module') != PRODUCER):
        raise ValueError('Combined Flip source requires its own immutable producer identity')
    for value in (config, config['shard']):
        for module in (stay, search, labels, cadence, sharing, rank):
            module.validate_fields(value)
    if metrics is not None:
        if any(metrics.get(key) != value for key, value in actual.items()) or metrics.get('producer_module') != PRODUCER:
            raise ValueError('Combined Flip metrics differ from their declared source')
        for module in (stay, search, labels, cadence, sharing, rank):
            module.validate_fields(metrics)


@contextmanager
def camera_support():
    previous = fov.declared_fields
    fov.declared_fields = declared_fields
    try:
        yield
    finally:
        fov.declared_fields = previous

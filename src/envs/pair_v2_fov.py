"""Explicit supplement-10 camera contrast; no scientific threshold changes."""
import copy

FOV_VERSION = 'odd100_even60_v1'
FOV_POLICY = {
    'version': FOV_VERSION, 'approved_task_commit': '0451b8b',
    'scenes': [f'FloorPlan{i}' for i in (*range(1, 21), *range(201, 221))],
    'assignment': 'odd scene number A100; even scene number B60; fixed before collection',
    'field_of_view_degrees': {'A100': 100, 'B60': 60},
    'minimum_eligible_attempts_per_scene': 100,
    'selection_order': ['larger total qualified Look pairs',
                        'only if both Look totals are zero: larger total qualified Direct pairs',
                        'otherwise tied: retain 100 degrees and report the tie'],
    'scope': 'same generator and thresholds; camera field of view is the only contrast parameter',
}
CONTRAST_KEYS = ('fov_contrast_version', 'fov_contrast_policy', 'fov_contrast_group')


def scene_field_of_view(scene):
    if scene not in FOV_POLICY['scenes']:
        raise ValueError('FOV contrast requires one of the fixed forty training scenes')
    return 100 if int(scene.removeprefix('FloorPlan')) % 2 else 60


def contrast_fields(field_of_view):
    if type(field_of_view) is not int or field_of_view not in (60, 100):
        raise ValueError('FOV contrast allows only 60 or 100 degrees')
    return {'field_of_view': field_of_view, 'fov_contrast_version': FOV_VERSION,
            'fov_contrast_policy': copy.deepcopy(FOV_POLICY),
            'fov_contrast_group': 'A100' if field_of_view == 100 else 'B60'}


def declared_fields(config):
    """Old missing declarations mean the original 100, never a new arm."""
    value = config.get('field_of_view', 100)
    if type(value) is not int or value not in (60, 100):
        raise ValueError('Unknown declared camera field of view')
    active = any(key in config for key in CONTRAST_KEYS)
    if not active:
        if value != 100:
            raise ValueError('Nonlegacy FOV requires the explicit approved contrast policy')
        return {'field_of_view': 100}
    expected = contrast_fields(value)
    if any(config.get(key) != expected[key] for key in CONTRAST_KEYS):
        raise ValueError('Unknown or altered FOV contrast declaration')
    scenes = ([config['scene']] if 'scene' in config else config.get('pilot_scenes', []))
    if any(scene_field_of_view(scene) != value for scene in scenes):
        raise ValueError('Scene does not match its fixed odd100/even60 FOV assignment')
    return expected


def bind_source_fields(config, shard):
    actual, expected = declared_fields(config), declared_fields(shard)
    if actual != expected:
        raise ValueError('Source camera/FOV contrast differs from its immutable shard')
    return actual


def verify_native_fov(metadata, field_of_view):
    actual = metadata.get('fov')
    if type(actual) not in (int, float) or actual != field_of_view:
        raise ValueError('Actual native camera FOV differs from its declared source')


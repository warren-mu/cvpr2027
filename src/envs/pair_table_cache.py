"""Pure T02 cache checks; this module neither renders nor certifies synthetic data.

The 2% denominator is the cached mask count (the reference being checked),
recorded explicitly. A failed sampled state blocks the entire downstream cache;
it is never a suggestion to filter out that pair and keep training.
"""
from dataclasses import dataclass
import hashlib
import json
import random
import re

from .pair_table import RenderedView, classify_views, eligible_names, label_assignment

CACHE_VERSION = 'paired_cache_reference_count_shadow_v2'
CACHE_POLICY = {
    'minimum_distinct_pair_states': 200,
    'resolution': 255,
    'mask_count_relative_tolerance': 0.02,
    'mask_count_reference': 'cached_count',
    'mask_count_absolute_tolerance_pixels': 10,
    'minimum_mask_iou': 0.9,
    'either_positive_requires_iou': True,
    'fresh_pair_render_process': 'one_new_process_same_exact_camera_pose',
    'rgb_difference': 'diagnostic_only_except_existing_shared_state_definition',
    'non_anchor_divergence': 'Both Look and Direct fork with own world RGB; record shadow counts; never exclude by this predicate',
    'clean_subset': 'Look tables with zero shadow-divergent states; diagnostic subset only, not a cache acceptance gate',
    'failure_scope': 'whole_downstream_pipeline',
}


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _key(value):
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(not isinstance(part, str) or not part for part in value)):
        raise ValueError('State key must be an explicit nonempty (pair_id, state_id)')
    return tuple(value)


def sample_state_keys(population, *, seed, sample_size=200):
    """Select without replacement from the complete caller-supplied population.

    The exporter must bind that population to its authenticated table inventory;
    this pure function cannot establish that no table was omitted by its caller.
    """
    if type(seed) is not int or type(sample_size) is not int or sample_size < 200:
        raise ValueError('Explicit integer seed and at least 200 distinct states required')
    keys = tuple(sorted(_key(item) for item in population))
    if len(keys) != len(set(keys)) or len(keys) < sample_size:
        raise ValueError('Population must contain enough unique pair/state keys')
    selected = random.Random(seed).sample(keys, sample_size)
    return {'version': CACHE_VERSION, 'algorithm': 'python_random_sample_without_replacement',
            'seed': seed, 'sample_size': sample_size, 'population_size': len(keys),
            'population_sha256': hashlib.sha256(_encoded(keys)).hexdigest(),
            'selected_keys': [list(key) for key in selected]}


@dataclass(frozen=True)
class CacheSample:
    pair_id: str
    state_id: str
    anchor_name: str
    candidate_names: tuple
    cached: tuple  # w0, w1 RenderedView, before shared-pixel substitution
    fresh: tuple   # both worlds actually re-rendered, never a cached image copy
    cached_labels: tuple  # per world: {stable candidate name: public numeric label}
    fresh_labels: tuple
    kind: str  # bound to the source table before cache authorization
    dataset_key: str = None

    @property
    def population_key(self):
        return self.dataset_key or self.pair_id


def _labels(view, actual, pair_id, candidates):
    if not isinstance(actual, dict) or any(type(label) is not int for label in actual.values()):
        raise ValueError('Require actual per-world stable-name/public-label mappings')
    assignments = label_assignment(pair_id, candidates)
    if any(view.distances.get(name) is None for name in candidates):
        raise ValueError('Candidate distance is unknown; label eligibility cannot be checked')
    expected = {name: assignments[name] for name in eligible_names(view, candidates)}
    if actual != expected:
        raise ValueError('Reported label set differs from measured masks/distances and pair labels')
    return dict(sorted(actual.items()))


def _source(view, sample, world):
    source = view.source
    if (source.get('pair_id') != sample.pair_id or source.get('state_id') != sample.state_id
            or source.get('world_id') != world
            or not re.fullmatch(r'[0-9a-f]{64}', str(source.get('snapshot_sha256', '')))):
        raise ValueError('Capture source does not bind the exact pair/state/world snapshot')
    return {'process_id': view.process_id, 'render_id': view.render_id,
            'snapshot_sha256': source['snapshot_sha256'],
            'rgb_sha256': view.rgb.sha256,
            'source': source}


def _mask_check(cached, fresh):
    old_count, new_count = cached.count(1), fresh.count(1)
    delta = abs(old_count - new_count)
    intersection = sum(a & b for a, b in zip(cached, fresh))
    union = old_count + new_count - intersection
    # Integer inequalities implement the registered boundaries without rounding.
    count_passed = delta * 50 <= max(old_count, 500)
    iou_passed = union == 0 or intersection * 10 >= union * 9
    return {'cached_pixels': old_count, 'fresh_pixels': new_count,
            'absolute_count_difference': delta,
            'allowed_count_difference': max(0.02 * old_count, 10),
            'count_reference': 'cached_count', 'count_passed': count_passed,
            'intersection_pixels': intersection, 'union_pixels': union,
            'iou': intersection / union if union else None,
            'iou_required': union > 0, 'iou_passed': iou_passed,
            'passed': count_passed and iou_passed,
            'cached_mask_sha256': hashlib.sha256(cached).hexdigest(),
            'fresh_mask_sha256': hashlib.sha256(fresh).hexdigest()}


def check_cached_state(sample):
    """Return measured checks for BOTH worlds; malformed/missing evidence raises.

    `checks_passed` may be true for a CPU fixture. `cache_matches_live` remains
    null for any synthetic capture, which cannot authorize training.
    """
    if not isinstance(sample, CacheSample):
        raise ValueError('An explicit CacheSample is required')
    _key((sample.pair_id, sample.state_id))
    candidates = tuple(sample.candidate_names)
    names = (sample.anchor_name, *candidates)
    if (len(candidates) != 2 or len(set(names)) != 3
            or any(not isinstance(name, str) or not name for name in names)):
        raise ValueError('Require distinct stable X/A/B identities')
    for views in (sample.cached, sample.fresh):
        if len(views) != 2 or any(not isinstance(view, RenderedView) for view in views):
            raise ValueError('Both worlds require complete actual render records')
        for view in views:
            for name in names:
                view.count(name)  # Missing mask is unknown, not an all-zero mask.
    if len(sample.cached_labels) != 2 or len(sample.fresh_labels) != 2:
        raise ValueError('Both worlds require actual label sets')
    views = (*sample.cached, *sample.fresh)
    if len({view.synthetic for view in views}) != 1:
        raise ValueError('Cannot mix synthetic and real evidence')
    cached_processes = {view.process_id for view in sample.cached}
    if any(view.process_id in cached_processes for view in sample.fresh):
        raise ValueError('Fresh render process must differ from the cached export process')
    if len({(view.process_id, view.render_id) for view in views}) != 4:
        raise ValueError('Require four distinct originating renders')
    cached_class = classify_views(sample.cached, sample.anchor_name, kind=sample.kind)
    if (sample.fresh[0].process_id != sample.fresh[1].process_id
            or _encoded(sample.fresh[0].agent) != _encoded(sample.fresh[1].agent)):
        raise ValueError('Both fresh world renders require one process and the exact same camera pose')
    fresh_pixels = tuple(view.count(sample.anchor_name) for view in sample.fresh)
    fresh_difference = max(abs(a - b) for a, b in zip(sample.fresh[0].rgb.data, sample.fresh[1].rgb.data))
    fresh_shared = fresh_pixels == (0, 0) and fresh_difference <= 3
    fresh_divergent = fresh_pixels == (0, 0) and fresh_difference > 3
    reasons, worlds = [], {}
    for index, world in enumerate(('w0', 'w1')):
        cached, fresh = sample.cached[index], sample.fresh[index]
        cached_source, fresh_source = _source(cached, sample, world), _source(fresh, sample, world)
        if cached_source['snapshot_sha256'] != fresh_source['snapshot_sha256']:
            raise ValueError('Fresh render did not restore the cached world snapshot')
        old_labels = _labels(cached, sample.cached_labels[index], sample.pair_id, candidates)
        new_labels = _labels(fresh, sample.fresh_labels[index], sample.pair_id, candidates)
        masks = {name: _mask_check(cached.masks[name], fresh.masks[name]) for name in names}
        for name, check in masks.items():
            if not check['passed']:
                reasons.append(world + ':mask_mismatch:' + name)
        if old_labels != new_labels:
            reasons.append(world + ':label_set_changed')
        worlds[world] = {'masks': masks, 'cached_labels': old_labels, 'fresh_labels': new_labels,
                         'labels_unchanged': old_labels == new_labels,
                         'cache_max_abs_diff': max(abs(a - b) for a, b in zip(cached.rgb.data, fresh.rgb.data)),
                         'agent_pose_exact_diagnostic': _encoded(cached.agent) == _encoded(fresh.agent),
                         'cached_source': cached_source, 'fresh_source': fresh_source}
    if cached_class['shared'] != fresh_shared:
        reasons.append('shared_fork_classification_changed')
    passed, synthetic = not reasons, views[0].synthetic
    return {'pair_id': sample.pair_id, 'state_id': sample.state_id, 'kind': sample.kind, 'version': CACHE_VERSION,
            **({'dataset_key': sample.dataset_key, 'original_pair_id': sample.pair_id} if sample.dataset_key is not None else {}),
            'synthetic': synthetic, 'checks_passed': passed,
            'cache_matches_live': None if synthetic else passed,
            'reasons': reasons, 'worlds': worlds,
            'cached_classification': {'shared': cached_class['shared'],
                                      'shadow_divergent': cached_class['shadow_divergent'],
                                      'non_anchor_divergent': cached_class['non_anchor_divergent'],
                                      'frame_max_abs_diff': cached_class['frame_max_abs_diff']},
            'fresh_classification': {'shared': fresh_shared, 'non_anchor_divergent': fresh_divergent,
                                     'shadow_divergent': fresh_divergent,
                                     'anchor_pixels': fresh_pixels, 'frame_max_abs_diff': fresh_difference}}


def validate_cache_sample(samples, *, population, sampling_plan):
    """Validate the full predeclared sample. No failed-pair filtering is allowed.

    This helper does not audit external files: the caller must authenticate the
    complete table inventory and each capture before constructing these inputs.
    """
    expected = sample_state_keys(population, seed=sampling_plan['seed'],
                                 sample_size=sampling_plan['sample_size'])
    if sampling_plan != expected:
        raise ValueError('Sampling plan does not reproduce the declared population/seed')
    records = list(samples)
    keys = [_key((sample.population_key, sample.state_id)) for sample in records]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate pair/state records cannot increase the sample size')
    requested = {_key(key) for key in expected['selected_keys']}
    if set(keys) - requested:
        raise ValueError('Observed records include states outside the predeclared sample')
    results = []
    all_cached_processes = {view.process_id for sample in records for view in sample.cached}
    fresh_render_identities = [(view.process_id, view.render_id) for sample in records for view in sample.fresh]
    if len(fresh_render_identities) != len(set(fresh_render_identities)):
        raise ValueError('One fresh originating render cannot stand for multiple sampled pair/states')
    for sample in records:
        try:
            if any(view.process_id in all_cached_processes for view in sample.fresh):
                raise ValueError('Fresh process is also an export process elsewhere in the sample')
            result = check_cached_state(sample)
        except (ValueError, TypeError, KeyError) as error:
            result = {'pair_id': sample.pair_id, 'state_id': sample.state_id,
                      **({'dataset_key': sample.dataset_key, 'original_pair_id': sample.pair_id} if sample.dataset_key is not None else {}),
                      'checks_passed': False, 'cache_matches_live': None,
                      'synthetic': any(view.synthetic for view in (*sample.cached, *sample.fresh)),
                      'reasons': ['missing_or_invalid_evidence'],
                      'error': str(error), 'worlds': None}
        results.append(result)
    missing = sorted(requested - set(keys))
    complete = not missing and len(keys) >= 200
    passed = complete and all(row['checks_passed'] for row in results)
    synthetic = any(row['synthetic'] for row in results)
    live_passed = passed and not synthetic
    return {'version': CACHE_VERSION, 'policy': dict(CACHE_POLICY), 'sampling_plan': expected,
            'observed_distinct_states': len(keys), 'sample_complete': complete,
            'missing_state_keys': [list(key) for key in missing], 'states': results,
            'checks_passed': passed, 'cache_matches_live': None if synthetic else live_passed,
            'downstream_allowed': live_passed, 'stop_downstream': not live_passed,
            'synthetic_fixture': synthetic,
            'scope': 'whole_export_cache_sample_not_per_pair_filtering',
            'provenance_scope': 'caller_must_authenticate_inventory_and_capture_sources'}


def require_cache_validation(validation):
    """Fail closed at the training/export continuation boundary."""
    if (validation.get('version') != CACHE_VERSION
            or validation.get('policy') != CACHE_POLICY
            or validation.get('sample_complete') is not True
            or validation.get('checks_passed') is not True
            or type(validation.get('observed_distinct_states')) is not int
            or validation['observed_distinct_states'] < 200
            or len(validation.get('states', ())) != validation['observed_distinct_states']
            or any(row.get('cache_matches_live') is not True for row in validation.get('states', ()))
            or validation.get('cache_matches_live') is not True
            or validation.get('downstream_allowed') is not True
            or validation.get('stop_downstream') is not False
            or validation.get('synthetic_fixture') is not False):
        raise ValueError('Whole cache validation is incomplete or failed; downstream use is blocked')

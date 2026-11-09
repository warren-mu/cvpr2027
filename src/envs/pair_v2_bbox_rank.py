"""Optional, versioned proposal ordering; never a visibility measurement.

Translate the original anchor AABB dimensions to an existing proposal centre.
The projected rectangle is an upper bound on the box silhouette, not an object
mask. Placement, pivot offsets, settling, rotation and occlusion can invalidate
these predictions; every existing physical/rendered acceptance check remains.
"""
import copy
import itertools
from . import pair_v2_projection_pool as pools

VERSION = 'predicted_anchor_aabb_hidden_then_one_step_area_v1'
POLICY = {
    'version': VERSION,
    'geometry': 'original anchor world-axis AABB size translated to the existing projected proposal centre; no new placement coordinates',
    'projection': 'all eight box corners through the existing camera model; any nonpositive corner depth makes that camera prediction unknown',
    'start_margin': 'signed rectangle separation from the starting frame; positive means the entire predicted rectangle is outside',
    'one_step_area': 'maximum clipped projected rectangle area among supplied existing one-action cameras; not mask/evidence pixels or action availability',
    'ordering': 'known start and one-step prediction first, predicted whole-box outside first, larger one-step rectangle area, larger signed starting margin, original input order',
    'domain': 'retain every original entry and native token, including unknown or low-ranked predictions; no eligibility filter or cap',
    'deadline': 'optional original soft-deadline callback before each point; exceptions propagate before any returned ranked pool, without mutating the input',
    'unchanged': 'all actual physical, target, mask, RGB, evidence and witness checks; side/seed assignment, counters and schedules',
    'scope': 'proposal prediction only; no acceptance result or measured visibility is produced',
}
FIELDS = ('bbox_proposal_rank_version', 'bbox_proposal_rank_policy')


def fields():
    return {FIELDS[0]: VERSION, FIELDS[1]: copy.deepcopy(POLICY)}


def validate_fields(value):
    if any(value.get(key) != item for key, item in fields().items()):
        raise ValueError('Missing or altered predicted-box proposal ordering policy')


def project_box(center, size, camera):
    """A finite front-facing box gets a conservative projected rectangle."""
    center, size = pools._xyz(center), pools._xyz(size)
    if any(value < 0 for value in size.values()):
        raise ValueError('Negative source anchor AABB size')
    corners = [pools.project_point({axis: center[axis] + sign * size[axis] / 2
                                   for axis, sign in zip('xyz', signs)}, camera)
               for signs in itertools.product((-1, 1), repeat=3)]
    result = {'scope': POLICY['scope'], 'projectable': False, 'rectangle_px': None,
        'signed_outside_margin_px': None, 'whole_rectangle_outside_prediction': None,
        'clipped_rectangle_area_px2': None}
    if not all(corner['projectable'] for corner in corners):
        return {**result, 'unknown_reason': 'box_intersects_or_is_behind_camera_plane'}
    left, right = min(corner['u'] for corner in corners), max(corner['u'] for corner in corners)
    top, bottom = min(corner['v'] for corner in corners), max(corner['v'] for corner in corners)
    width, height = camera['width'], camera['height']
    margin = max(-right, left - width, -bottom, top - height)
    area = max(0., min(right, width) - max(left, 0.)) * max(0., min(bottom, height) - max(top, 0.))
    return {**result, 'projectable': True,
        'rectangle_px': {'left': left, 'right': right, 'top': top, 'bottom': bottom},
        'signed_outside_margin_px': margin, 'whole_rectangle_outside_prediction': margin > 0,
        'clipped_rectangle_area_px2': area}


def predict(item, anchor, camera, observation_cameras):
    """Retain the existing centre verbatim; do not infer actual settled poses."""
    result = {**fields(), 'scope': POLICY['scope'], 'anchor_name': anchor.get('name'),
        'projection_center': None, 'source_anchor_aabb_size': None, 'start': None,
        'after_actions': [], 'best_one_step_rectangle_area_px2': None,
        'ranking_prediction_known': False}
    try:
        center = pools._xyz(item['projection_point'])
        size = pools._xyz(anchor['axisAlignedBoundingBox']['size'])
        result.update(projection_center=center, source_anchor_aabb_size=size,
                      start=project_box(center, size, camera))
        for supplied in observation_cameras:
            result['after_actions'].append({'action': copy.deepcopy(supplied['action']),
                'projection': project_box(center, size, supplied['camera'])})
    except (KeyError, TypeError, ValueError) as error:
        # Incomplete geometric metadata must not erase a previously legal point.
        return {**result, 'unknown_reason': f'{type(error).__name__}: {error}'}
    known = [item['projection']['clipped_rectangle_area_px2'] for item in result['after_actions']
             if item['projection']['projectable']]
    if known:
        result['best_one_step_rectangle_area_px2'] = max(known)
    result['ranking_prediction_known'] = result['start']['projectable'] and bool(known)
    if not result['ranking_prediction_known']:
        result['unknown_reason'] = 'no_known_start_and_one_step_box_projection'
    return result


def rank_key(prediction):
    """No new cutoff: continuous area and signed margin order all known points."""
    if not prediction['ranking_prediction_known']:
        return (1, 0, 0., 0.)
    start = prediction['start']
    return (0, int(not start['whole_rectangle_outside_prediction']),
            -prediction['best_one_step_rectangle_area_px2'], -start['signed_outside_margin_px'])


def rank_look_points(points, anchor, camera, observation_cameras, *, check_deadline=None):
    """Return shallow entry copies in stable order, sharing unchanged tokens."""
    predicted = []
    for item in points:
        if check_deadline is not None:
            check_deadline()
        predicted.append({**item, 'bbox_proposal_prediction': predict(item, anchor, camera, observation_cameras)})
    return sorted(predicted, key=lambda item: rank_key(item['bbox_proposal_prediction']))


def rank_pool(record, anchor, *, check_deadline=None):
    """Opt-in adapter for an existing full pool; Stay/Direct domains stay intact.

    A future producer must explicitly declare fields() and authenticate these
    pool records. This helper does not install itself in any active module.
    """
    points = dict(record['points'])
    points['look'] = {side: rank_look_points(record['points']['look'][side], anchor,
        record['camera'], record['observation_cameras'], check_deadline=check_deadline) for side in ('A', 'B')}
    return {**record, **fields(), 'points': points}

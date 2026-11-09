"""Opt-in value-equivalent native tokens without quadratic copying/hashing.

The registered digest is exactly the original complete JSON token digest.
Native queries, literal point order, validate_spawn and placement membership
checks are unchanged. No scientific gate or native coordinate is inferred.
"""
import copy
import hashlib
from contextlib import contextmanager

from . import pair_v2_simulator as engine

VERSION = 'native_token_shared_json_exact_sha256_v1'
POLICY = {
    'version': VERSION,
    'native_query': 'same GetSpawnCoordinatesAboveReceptacle(objectId, anywhere=True), complete original actionReturn in original order',
    'coordinates': 'one detached read-only JSON-equivalent coordinate tree per actual query; deepcopy shares only that tree',
    'registration': 'every original token is registered before return with exactly SHA256(engine.encoded(token)); hash common canonical prefix once then copy hash state per selected point',
    'use_validation': 'unchanged validate_spawn and fresh complete-token SHA256 registry membership at actual placement; no cached caller-token digest',
    'deadline': 'original simulator soft deadline checked before query, during coordinate copy and before each point validation/registration',
    'scope': 'representation and CPU work only; no native domain, ordering, seed, physics, pixel, target or witness threshold changes',
}
FIELDS = ('spawn_token_storage_version', 'spawn_token_storage_policy')
_ORIGINAL_SPAWN = engine.V2Simulator.spawn_points


def fields():
    return {FIELDS[0]: VERSION, FIELDS[1]: copy.deepcopy(POLICY)}


def validate_fields(value):
    if any(value.get(key) != item for key, item in fields().items()):
        raise ValueError('Shared native tokens require their explicit exact-value storage policy')


def _readonly(*args, **kwargs):
    raise TypeError('Shared native coordinate evidence is read-only')


class ReadOnlyMap(dict):
    """JSON dict with ordinary mutation blocked; registration never trusts this alone."""
    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _readonly

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self

    def __reduce_ex__(self, protocol):
        return (ReadOnlyMap, (dict(self),))


class ReadOnlyList(list):
    __setitem__ = __delitem__ = append = clear = extend = insert = pop = remove = reverse = sort = _readonly
    __iadd__ = __imul__ = _readonly

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self

    def __reduce_ex__(self, protocol):
        return (ReadOnlyList, (list(self),))


class ReadOnlyTuple(tuple):
    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self

    def __reduce_ex__(self, protocol):
        return (ReadOnlyTuple, (tuple(self),))


def detached_readonly(value, deadline):
    """Preserve JSON values and list/tuple equality, detaching original metadata."""
    deadline()
    if isinstance(value, dict):
        return ReadOnlyMap((key, detached_readonly(item, deadline)) for key, item in value.items())
    if isinstance(value, list):
        return ReadOnlyList(detached_readonly(item, deadline) for item in value)
    if isinstance(value, tuple):
        return ReadOnlyTuple(detached_readonly(item, deadline) for item in value)
    return copy.deepcopy(value)


def canonical_parts(template):
    """Exact default-json framing around selected_position; no sentinel replacement.

    The fixed token schema has only string keys. Each field uses the original
    encoder, including its Unicode escaping, floating point and signed-zero
    representation. The giant coordinates field precedes selected_position.
    """
    if 'selected_position' not in template or any(type(key) is not str for key in template):
        raise ValueError('Expected the original string-keyed native token schema')
    keys = sorted(template)
    middle = keys.index('selected_position')
    entry = lambda key: engine.encoded(key) + b': ' + engine.encoded(template[key])
    before = b', '.join(entry(key) for key in keys[:middle])
    after = b', '.join(entry(key) for key in keys[middle + 1:])
    prefix = b'{' + before + (b', ' if before else b'') + engine.encoded('selected_position') + b': '
    suffix = (b', ' if after else b'') + after + b'}'
    return prefix, suffix


def spawn_points(sim, receptacle_name):
    validate_fields(sim.config)
    sim.check_deadline()
    obj = engine.objects_by_name(sim.controller.last_event.metadata['objects'])[receptacle_name]
    if obj.get('receptacle') is not True:
        raise ValueError('Destination is not a native receptacle')
    before_position = copy.deepcopy(obj['position'])
    event = sim.action('GetSpawnCoordinatesAboveReceptacle', objectId=obj['objectId'], anywhere=True)
    values = event.metadata.get('actionReturn')
    if not isinstance(values, list):
        raise RuntimeError('Unknown native spawn-coordinate schema')
    if not values:
        return []
    coordinates = detached_readonly(values, sim.check_deadline)
    template = {'action': 'GetSpawnCoordinatesAboveReceptacle', 'success': True,
        'scene': sim.scene, 'receptacle_name': receptacle_name, 'receptacle_id_at_query': obj['objectId'],
        'receptacle_position_at_query': before_position, 'selected_position': None,
        'coordinates': coordinates, 'query_reference': copy.deepcopy(sim.last_action_reference),
        'source': engine.SOURCE + 'PhysicsRemoteFPSAgentController.cs#L3423',
        'scope': 'native receptacle samples; placement requires subsequent settling/membership checks'}
    sim.check_deadline()
    prefix, suffix = canonical_parts(template)
    common_hash = hashlib.sha256(prefix)
    rows = []
    for point in values:
        sim.check_deadline()
        engine.distance(point, point)
        selected = copy.deepcopy(point)
        digest = common_hash.copy()
        digest.update(engine.encoded(selected)); digest.update(suffix)
        token = copy.deepcopy(template)
        token['selected_position'] = selected
        # The registry retains an independent small header/selected position.
        # Shared coordinates are detached/read-only; even a deliberate base-
        # class mutation changes the fresh full hash and cannot gain admission.
        sim.spawn_tokens[digest.hexdigest()] = copy.deepcopy(token)
        rows.append(token)
    return rows


@contextmanager
def installed():
    """Only a future explicitly versioned producer may opt into this helper."""
    previous = engine.V2Simulator.spawn_points
    engine.V2Simulator.spawn_points = spawn_points
    try:
        yield
    finally:
        engine.V2Simulator.spawn_points = previous

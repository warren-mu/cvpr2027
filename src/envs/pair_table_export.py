"""Offline D4 table construction; importing this module never starts THOR.

The caller must supply an approved pair and own PBS allocation/controller,
deadlines, cleanup, run_logged and result provenance. This module provides no
login-node or automatic GPU entry point. Export alone does not unlock training.
"""
from collections import deque
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Protocol

from .pair_core import ViewState
from .paired_env import RGBFrame
from .pair_table import (PairTable, RenderedView, TableState, TABLE_VERSION,
                         classify_views, encoded)


class PairRenderer(Protocol):
    def domain(self, world: int, state: ViewState) -> dict:
        """Return moves with literal native index/position and full path evidence."""
        ...
    def render_pair(self, state: ViewState) -> tuple:
        """Two fresh captures, one process, no agent movement between worlds."""
        ...


def point_id(point):
    if set(point) != {'x', 'y', 'z'} or any(type(v) not in (int, float) or not math.isfinite(v) for v in point.values()):
        raise ValueError('Finite literal xyz viewpoint required')
    return 'v_' + hashlib.sha256(encoded(point)).hexdigest()[:20]


def state_id(view):
    return f'{view.viewpoint_id}:h{view.heading_index}'


def directions(position, heading, moves):
    words = ('front', 'front-right', 'right', 'back-right', 'back', 'back-left', 'left', 'front-left')
    result = []
    for move in moves:
        point = move['position']
        yaw = math.degrees(math.atan2(point['x'] - position['x'], point['z'] - position['z']))
        relative = (yaw - heading * 45) % 360
        result.append((move['index'], words[int(math.floor((relative + 22.5) / 45)) % 8], point_id(point)))
    return tuple(result)


def domain_moves(domain, position):
    if domain.get('complete') is not True or not isinstance(domain.get('evidence'), dict):
        raise ValueError('Native graph domain unavailable; not an empty graph')
    moves = domain.get('moves')
    if not isinstance(moves, list):
        raise ValueError('Native move list missing')
    seen = set()
    for move in moves:
        key = move.get('index')
        point_id(move['position'])
        delta = {k: move['position'][k] - position[k] for k in ('x', 'y', 'z')}
        if (type(key) is not int or key < 1 or key in seen or move.get('verified') is not True
                or not isinstance(move.get('path_reference'), dict)
                or not 1e-6 < math.sqrt(sum(v*v for v in delta.values())) <= .26
                or abs(delta['y']) > .01 or min(abs(delta['x']), abs(delta['z'])) > .01):
            raise ValueError('Move differs from an authenticated T01 adjacent native point')
        seen.add(key)
    return sorted(moves, key=lambda move: move['index'])


def build_table(renderer, *, pair_id, scene, kind, instruction, anchor_name,
                candidate_names, target_names, initial_position, initial_heading,
                source=None):
    """BFS D4; query both worlds at every state before accepting a common graph."""
    started = time.perf_counter()
    initial = ViewState(point_id(initial_position), initial_heading)
    positions = {initial.viewpoint_id: copy.deepcopy(initial_position)}
    queue, distances = deque([initial]), {initial: 0}
    intermediate = {}
    while queue:
        view = queue.popleft()
        domains = tuple(renderer.domain(world, view) for world in (0, 1))
        moves = [domain_moves(domain, positions[view.viewpoint_id]) for domain in domains]
        signatures = [[(m['index'], m['position']) for m in rows] for rows in moves]
        if encoded(signatures[0]) != encoded(signatures[1]):
            raise ValueError('graph_mismatch')
        neighbors = directions(positions[view.viewpoint_id], view.heading_index, moves[0])
        successors = {'turn_left': ViewState(view.viewpoint_id, (view.heading_index - 1) % 8),
                      'turn_right': ViewState(view.viewpoint_id, (view.heading_index + 1) % 8)}
        for move in moves[0]:
            key = point_id(move['position'])
            positions.setdefault(key, copy.deepcopy(move['position']))
            successors[f'move {move["index"]}'] = ViewState(key, view.heading_index)
        if distances[view] < 4:
            for following in successors.values():
                if following not in distances:
                    distances[following] = distances[view] + 1
                    queue.append(following)
        captures = tuple(renderer.render_pair(view))
        measurement = classify_views(captures, anchor_name)
        intermediate[view] = (successors, neighbors, domains, captures, measurement)
    states = {}
    for view, (successors, neighbors, domains, captures, measured) in intermediate.items():
        key = state_id(view)
        # D4 boundary destinations outside the export cannot be observed within
        # the episode budget. Existing in-table return edges are retained.
        transitions = {a: state_id(target) for a, target in successors.items() if target in intermediate}
        states[key] = TableState(key, view, captures, transitions, neighbors, domains,
                                measured['shared'], measured['frame_max_abs_diff'], measured['anchor_pixels'])
    table = PairTable(pair_id=pair_id, scene=scene, kind=kind, instruction=instruction,
        anchor_name=anchor_name, candidate_names=candidate_names, target_names=target_names,
        initial_state=state_id(initial), positions=positions, states=states, source=source,
        export_verified=False, cache_verified=False)
    table.export_measurements = {'state_count': len(states), 'elapsed_seconds': time.perf_counter() - started,
        'includes': 'native graph queries and paired capture construction; excludes caller startup and serialization',
        'synthetic_only': table.synthetic}
    return table


class SimulatorTableRenderer:
    """Adapter for an already allocated, privately owned V2Simulator.

    A clone-expanded pair must already have its authenticated full inventory
    recreated by the caller. This adapter refuses missing/extra pose names via
    the existing complete_pose_list guard; it never invents clones.
    """
    def __init__(self, sim, worlds, reachable, initial_position, anchor_name, candidate_names,
                 *, pair_id, reachable_reference):
        self.sim, self.worlds, self.reachable = sim, copy.deepcopy(worlds), copy.deepcopy(reachable)
        self.positions = {point_id(point): point for point in [initial_position, *reachable]}
        self.names = (anchor_name, *candidate_names)
        self.pair_id, self.reachable_reference = pair_id, copy.deepcopy(reachable_reference)
        from .thor_common import process_identity
        identity = {'worker': process_identity(os.getpid()),
                    'unity': process_identity(sim.controller.unity_pid), 'instance_nonce': os.urandom(16).hex()}
        if identity['worker'] is None or identity['unity'] is None:
            raise ValueError('Actual exporter/Unity process identities are unavailable')
        self.process_reference = sim.store.save_json('table_process_' + identity['instance_nonce'] + '.json', identity)
        self.process_id = self.process_reference['sha256']
        self.serial = 0

    def setup(self, world, state):
        from .pair_v2_simulator import complete_pose_list
        sim = self.sim
        snapshot = self.worlds[world]
        sim.action('PausePhysicsAutoSim')
        sim.action('SetObjectPoses', objectPoses=complete_pose_list(
            sim.controller.last_event.metadata['objects'], snapshot['poses']), placeStationary=True)
        self.world_pose_action = copy.deepcopy(sim.last_action_reference)
        sim.action('PhysicsSyncTransforms')
        # Feed the originally executed arguments back to THOR; reported float32
        # horizon values can accumulate Euler drift when reused as new inputs.
        agent = copy.deepcopy(snapshot.get('agent_setup_command'))
        if not isinstance(agent, dict) or not all(k in agent for k in ('position', 'rotation', 'horizon', 'standing')):
            raise ValueError('Canonical source camera setup command is unavailable')
        agent['position'] = copy.deepcopy(self.positions[state.viewpoint_id])
        agent['rotation']['y'] = state.heading_index * 45
        agent['forceAction'] = False
        sim.action('TeleportFull', **agent)
        self.agent_setup_action = copy.deepcopy(sim.last_action_reference)
        return sim.snapshot('table_setup')

    def domain(self, world, state):
        from .pair_v2_shard import observation_actions
        snapshot = self.setup(world, state)
        beginning = len(self.sim.query_log)
        actions, unavailable = observation_actions(self.sim, self.reachable, snapshot)
        # Unexpected API exceptions are unknown. A native complete=False path
        # remains explicit evidence rather than silently becoming a usable edge.
        unknown = [row for row in unavailable if not isinstance(row.get('path'), dict)]
        moves = [{'index': a['index'], 'position': a['position'], 'verified': a['path']['complete'] is True,
                  'path_reference': a['path']['query_reference']} for a in actions if a['kind'] == 'move']
        return {'complete': not unknown, 'moves': moves, 'evidence': {'world': world,
            'setup': snapshot['reference'], 'queries': copy.deepcopy(self.sim.query_log[beginning:]),
            'unavailable': unavailable, 'reachable_positions': self.reachable,
            'reachable_reference': self.reachable_reference,
            'world_snapshot_reference': self.worlds[world]['reference'],
            'world_pose_action': copy.deepcopy(self.world_pose_action),
            'agent_setup_action': copy.deepcopy(self.agent_setup_action)}}

    def capture(self, world, state, sequence):
        from .thor_checks import agent_pose
        from .pair_v2_simulator import objects_by_name, distance
        sim = self.sim
        sim.action('Pass')
        observation = sim.last_capture
        objects = objects_by_name(sim.controller.last_event.metadata['objects'])
        masks = {}
        for name in self.names:
            if name not in objects:
                raise ValueError('Table inventory lacks anchor/candidate: ' + name)
            mask = observation['masks_by_name'].get(name)
            # The complete metadata inventory proves absence from segmentation
            # means a real empty mask, rather than an unknown missing identity.
            masks[name] = bytes(255 * 255) if mask is None else mask.astype('uint8').tobytes()
        agent = agent_pose(sim.controller.last_event)
        camera = {k: copy.deepcopy(sim.controller.last_event.metadata.get(k)) for k in
                  ('cameraPosition', 'fov', 'screenWidth', 'screenHeight')}
        self.serial += 1
        return RenderedView(RGBFrame(observation['rgb'].tobytes(), 255, 255), masks,
            {name: distance(agent['position'], objects[name]['position']) for name in self.names},
            {'agent': agent, 'camera': camera}, self.process_id, f'{self.process_id}:{self.serial}',
            {'capture': observation['evidence'], 'render_action': sim.last_action_reference,
             'pair_id': self.pair_id, 'world_id': f'w{world}', 'world_index': world,
             'state_id': state_id(state), 'sequence': sequence,
             'snapshot_reference': self.worlds[world]['reference'],
             'snapshot_sha256': self.worlds[world]['reference']['sha256'],
             'process_reference': self.process_reference,
             'agent_setup_action': copy.deepcopy(self.agent_setup_action),
             'world_pose_action': copy.deepcopy(self.world_pose_action)}, False)

    def render_pair(self, state):
        from .pair_v2_simulator import complete_pose_list
        self.setup(0, state)
        left = self.capture(0, state, 'w0_then_w1_without_agent_action')
        sim = self.sim
        begin = len(sim.query_log)
        sim.action('SetObjectPoses', objectPoses=complete_pose_list(
            sim.controller.last_event.metadata['objects'], self.worlds[1]['poses']), placeStationary=True)
        self.world_pose_action = copy.deepcopy(sim.last_action_reference)
        sim.action('PhysicsSyncTransforms')
        right = self.capture(1, state, 'w0_then_w1_without_agent_action')
        # Keep every intervening action, not only an asserted same-camera flag.
        right.source['between_renders'] = copy.deepcopy(sim.query_log[begin:])
        return left, right


def save_table(table, directory, *, max_file_bytes=48 * 1024 * 1024, max_total_bytes=512 * 1024 * 1024):
    """Content-addressed immutable export. Shared *public* RGB stored once.

    Both original RGB measurements remain in evidence for independent audits;
    public_rgb references explicitly alias w0 only for verified shared states.
    No data deletion or overwriting, including after a partial failure.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    written, total = {}, 0

    def put(data, suffix):
        nonlocal total
        sha = hashlib.sha256(data).hexdigest()
        name = sha + suffix
        if name not in written:
            if len(data) > max_file_bytes or total + len(data) > max_total_bytes:
                raise ValueError('Table export byte cap exceeded; partial evidence retained')
            (root / name).write_bytes(data)
            total += len(data)
            written[name] = {'file': name, 'sha256': sha, 'bytes': len(data)}
        return written[name]

    rows = {}
    for key, state in table.states.items():
        captures = []
        for view in state.captures:
            captures.append({'rgb': put(view.rgb.data, '.rgb'),
                'masks': {name: put(mask, '.mask') for name, mask in view.masks.items()},
                'distances': dict(view.distances), 'agent': view.agent,
                'process_id': view.process_id, 'render_id': view.render_id,
                'source': view.source, 'synthetic': view.synthetic})
        rows[key] = {'view': vars(state.view), 'captures': captures, 'transitions': dict(state.transitions),
            'neighbors': state.neighbors, 'graph_evidence': state.graph_evidence, 'shared': state.shared,
            'frame_max_abs_diff': state.frame_max_abs_diff, 'anchor_pixels': state.anchor_pixels,
            'public_rgb': [captures[0]['rgb'], captures[0 if state.shared else 1]['rgb']]}
    manifest = {key: getattr(table, key) for key in ('pair_id', 'scene', 'kind', 'instruction', 'anchor_name',
        'candidate_names', 'target_names', 'initial_state', 'positions', 'depth', 'source')}
    manifest.update(version=TABLE_VERSION, states=rows, files=list(written.values()),
        export_verified=False, cache_verified=False,
        status='exported_evidence_pending_independent_checks', stored_payload_bytes=total)
    manifest['export_measurements'] = copy.deepcopy(getattr(table, 'export_measurements', None))
    content = encoded(manifest)
    if len(content) > max_file_bytes or total + len(content) > max_total_bytes:
        raise ValueError('Table manifest cap exceeded; partial evidence retained')
    path = root / 'table.json'
    path.write_bytes(content)
    reference = {'file': str(path.resolve()), 'sha256': hashlib.sha256(content).hexdigest(),
                 'bytes': len(content), 'payload_bytes': total}
    table.manifest_reference = copy.deepcopy(reference)
    return reference


def load_table(path, *, expected_sha256):
    """Hash-verified data loading; never infer real approval from a saved boolean."""
    path = Path(path).resolve()
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('Table manifest changed')
    manifest = json.loads(raw)
    if manifest['version'] != TABLE_VERSION:
        raise ValueError('Unknown table version')

    def read(ref):
        target = (path.parent / ref['file']).resolve()
        if target.parent != path.parent or target == path:
            raise ValueError('Table payload path escapes immutable package')
        data = target.read_bytes()
        if len(data) != ref['bytes'] or hashlib.sha256(data).hexdigest() != ref['sha256']:
            raise ValueError('Table payload changed')
        return data

    states = {}
    for key, row in manifest['states'].items():
        captures = tuple(RenderedView(RGBFrame(read(c['rgb']), 255, 255),
            {name: read(mask) for name, mask in c['masks'].items()}, c['distances'], c['agent'],
            c['process_id'], c['render_id'], c['source'], c['synthetic']) for c in row['captures'])
        view = ViewState(row['view']['viewpoint_id'], row['view']['heading_index'], tuple(row['view']['opened_containers']))
        expected_public = [row['captures'][0]['rgb'], row['captures'][0 if row['shared'] else 1]['rgb']]
        if row['public_rgb'] != expected_public:
            raise ValueError('Invalid shared public RGB alias')
        states[key] = TableState(key, view, captures, row['transitions'],
            tuple(tuple(n) for n in row['neighbors']), tuple(row['graph_evidence']), row['shared'],
            row['frame_max_abs_diff'], tuple(row['anchor_pixels']))
    args = {key: manifest[key] for key in ('pair_id', 'scene', 'kind', 'instruction', 'anchor_name',
        'candidate_names', 'target_names', 'initial_state', 'positions', 'depth', 'source')}
    table = PairTable(**args, states=states, export_verified=False, cache_verified=False)
    table.export_measurements = manifest.get('export_measurements')
    table.manifest_reference = {'file': str(path), 'sha256': expected_sha256, 'bytes': len(raw)}
    return table

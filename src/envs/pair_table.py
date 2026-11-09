"""CPU P-backend: authenticated paired renders, never an implicit live renderer.

Synthetic tables exercise contracts only. Production tables need independently
verified export evidence and the dataset-wide fresh-cache check before rollout.
"""
from dataclasses import dataclass, replace
import copy
import hashlib
import json
import math
from typing import Mapping
import uuid

from .pair_core import Action, ViewState
from .paired_env import (BackendFrame, Candidate, CaptureEvidence, EpisodeBudget,
                         PairEnvironment, PublicMark, RGBFrame, RewardSpec,
                         parse_action, public_observation)

TABLE_VERSION = 'paired_view_table_d4_v1'
DEPTH = 4
SEEN_PIXELS = 250
RESOLUTION = 255


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


@dataclass(frozen=True)
class RenderedView:
    rgb: RGBFrame
    masks: Mapping[str, bytes]
    distances: Mapping[str, float]
    agent: dict
    process_id: str  # process identity must include start identity, not a reusable PID alone
    render_id: str
    source: dict
    synthetic: bool = False

    def __post_init__(self):
        if self.rgb.shape != (255, 255, 3) or type(self.synthetic) is not bool:
            raise ValueError('P tables require explicit 255x255 uint8 RGB')
        if not self.agent or not self.process_id or not self.render_id or not isinstance(self.source, dict):
            raise ValueError('Missing capture identity, full camera pose or source evidence')
        if any(not isinstance(name, str) or not name or type(mask) is not bytes
               or len(mask) != 255 * 255 or set(mask) - {0, 1} for name, mask in self.masks.items()):
            raise ValueError('Masks require complete binary pixels keyed by stable name')
        if any(value is not None and (type(value) not in (float, int) or not math.isfinite(value) or value < 0)
               for value in self.distances.values()):
            raise ValueError('Candidate distances must be actual finite measurements or unknown')

    def count(self, name):
        if name not in self.masks:
            raise ValueError('Missing mask is unknown, not an empty mask: ' + name)
        return self.masks[name].count(1)


def classify_views(views, anchor_name, *, kind):
    """Apply the human-approved Look-only no-anchor-leak rule (supplement 8)."""
    if kind not in ('look_flip', 'look_stay', 'direct_flip', 'direct_stay'):
        raise ValueError('Explicit registered pair kind is required for state classification')
    if len(views) != 2 or any(not isinstance(view, RenderedView) for view in views):
        raise ValueError('Two actual render records required')
    a, b = views
    if (a.synthetic != b.synthetic or a.process_id != b.process_id or a.render_id == b.render_id
            or encoded(a.agent) != encoded(b.agent)):
        raise ValueError('Paired export must use two renders in one process without changing camera pose')
    pixels = (a.count(anchor_name), b.count(anchor_name))
    difference = max(abs(x - y) for x, y in zip(a.rgb.data, b.rgb.data))
    divergent = pixels == (0, 0) and difference > 3
    if divergent and kind.startswith('look_'):
        raise ValueError('non_anchor_divergent')
    return {'shared': pixels == (0, 0) and difference <= 3, 'anchor_pixels': pixels,
            'frame_max_abs_diff': difference, 'non_anchor_divergent': divergent}


def label_assignment(pair_id, candidate_names):
    """Pair-specific ordering of stable identities, independent of A/B input order."""
    names = tuple(sorted(candidate_names))
    if not pair_id or len(names) != 2 or len(set(names)) != 2:
        raise ValueError('Require pair ID and two distinct stable candidates')
    order = sorted(names, key=lambda name: hashlib.sha256(encoded([pair_id, name])).digest())
    return {name: index for index, name in enumerate(order, 1)}


def eligible_names(view, names):
    return tuple(name for name in names if view.count(name) > 0
                 and view.distances.get(name) is not None and view.distances[name] <= 1.5)


# Small built-in decimal glyphs avoid a font dependency or nondeterministic fonts.
_DIGITS = {'1': ('010', '110', '010', '010', '111'),
           '2': ('110', '001', '010', '100', '111')}


def mark_view(view, labels, candidate_names):
    data, marks = bytearray(view.rgb.data), []
    for name in eligible_names(view, candidate_names):
        pixels = [index for index, bit in enumerate(view.masks[name]) if bit]
        xs, ys = [p % 255 for p in pixels], [p // 255 for p in pixels]
        box = (min(xs), min(ys), max(xs), max(ys))
        n, sx, sy = len(pixels), sum(xs), sum(ys)
        center = min(pixels, key=lambda p: ((p % 255 * n - sx) ** 2 + (p // 255 * n - sy) ** 2, p))
        x, y = center % 255, center // 255
        marks.append(PublicMark(labels[name], (x, y), box))
        # A contrasting box + numeral is public policy RGB; masks remain raw.
        x0, y0 = min(247, max(0, x - 4)), min(243, max(0, y - 6))
        for dy in range(12):
            for dx in range(8):
                index = ((y0 + dy) * 255 + x0 + dx) * 3
                data[index:index + 3] = b'\x00\x00\x00'
        for dy, row in enumerate(_DIGITS[str(labels[name])]):
            for dx, bit in enumerate(row):
                if bit == '1':
                    for yy in range(2):
                        for xx in range(2):
                            index = ((y0 + 1 + dy * 2 + yy) * 255 + x0 + 1 + dx * 2 + xx) * 3
                            data[index:index + 3] = b'\xff\xff\xff'
    return RGBFrame(bytes(data), 255, 255), tuple(sorted(marks, key=lambda mark: mark.label))


@dataclass(frozen=True)
class TableState:
    state_id: str
    view: ViewState
    captures: tuple
    transitions: Mapping[str, str]
    neighbors: tuple  # (public move index, relative direction, destination viewpoint ID)
    graph_evidence: tuple  # per-world actual path/domain records
    shared: bool
    frame_max_abs_diff: int
    anchor_pixels: tuple


class PairTable:
    def __init__(self, *, pair_id, scene, kind, instruction, anchor_name, candidate_names,
                 target_names, initial_state, positions, states, depth=DEPTH,
                 export_verified=False, cache_verified=False, source=None):
        self.pair_id, self.scene, self.kind, self.instruction = pair_id, scene, kind, instruction
        self.anchor_name, self.candidate_names = anchor_name, tuple(sorted(candidate_names))
        self.target_names, self.initial_state = tuple(target_names), initial_state
        self.positions, self.states, self.depth = copy.deepcopy(positions), dict(states), depth
        self.labels = label_assignment(pair_id, self.candidate_names)
        self.source = copy.deepcopy(source or {})
        self.manifest_reference = None
        self.export_verified, self.cache_verified = export_verified, cache_verified
        if (depth != 4 or kind not in ('look_flip', 'look_stay', 'direct_flip', 'direct_stay')
                or initial_state not in states or not instruction or len(self.target_names) != 2
                or any(name not in self.candidate_names for name in self.target_names)):
            raise ValueError('Malformed fixed-D4 table identity or target')
        synthetic = set()
        for key, state in self.states.items():
            measurement = classify_views(state.captures, anchor_name, kind=kind)
            if (key != state.state_id or state.view.opened_containers or state.view.viewpoint_id not in positions
                    or state.shared is not measurement['shared']
                    or state.frame_max_abs_diff != measurement['frame_max_abs_diff']
                    or state.anchor_pixels != measurement['anchor_pixels']):
                raise ValueError('Table measurement/state identity mismatch')
            for capture in state.captures:
                synthetic.add(capture.synthetic)
                for name in (*self.candidate_names, anchor_name):
                    capture.count(name)
            if state.shared and eligible_names(state.captures[0], self.candidate_names) != eligible_names(state.captures[1], self.candidate_names):
                raise ValueError('Shared-state selectable labels disagree between actual worlds')
            if len(state.graph_evidence) != 2:
                raise ValueError('Both world graph proofs required')
            for action, destination in state.transitions.items():
                parsed = parse_action(action)
                if parsed.kind in ('select', 'open') or destination not in self.states:
                    raise ValueError('Invalid table transition')
                following = self.states[destination].view
                if parsed.kind.startswith('turn_'):
                    delta = -1 if parsed.kind == 'turn_left' else 1
                    if following != ViewState(state.view.viewpoint_id, (state.view.heading_index + delta) % 8):
                        raise ValueError('Table turn changes more than its 45-degree heading')
                elif (following.heading_index != state.view.heading_index
                      or (parsed.argument, following.viewpoint_id) not in [(n[0], n[2]) for n in state.neighbors]):
                    raise ValueError('Move does not match its public neighbor')
        if len(synthetic) != 1:
            raise ValueError('Synthetic and real captures cannot be mixed')
        self.synthetic = synthetic.pop()
        if self.synthetic and (export_verified or cache_verified):
            raise ValueError('CPU fixtures cannot certify real export or cache validation')

    def selectable_names(self, state_id, world):
        state = self.states[state_id]
        return eligible_names(state.captures[0 if state.shared else world], self.candidate_names)


class TableObservationBackend:
    """ObservationBackend implementation; no renderer, training or GPU imports."""
    def __init__(self, tables, *, allow_synthetic=False, world_orders=None):
        self.tables = dict(tables)
        self.world_orders = dict(world_orders or {})
        if set(self.world_orders) - set(self.tables) or any(
                list(order) not in ([0, 1], [1, 0]) for order in self.world_orders.values()):
            raise ValueError('World label assignment must be an explicit original-world permutation')
        if not self.tables or any(key != table.pair_id for key, table in self.tables.items()):
            raise ValueError('Explicit pair table mapping required')
        for table in self.tables.values():
            if table.synthetic:
                if not allow_synthetic:
                    raise ValueError('Synthetic tables require explicit CPU-test mode')
            elif not (table.export_verified and table.cache_verified):
                raise ValueError('Real P rollout blocked until export and dataset-wide cache validation pass')
            elif not isinstance(table.manifest_reference, dict):
                raise ValueError('Real P rollout requires its immutable table manifest reference')
        self.serial = 0
        self.public_cache = {}

    def reset(self, pair_id, world):
        if world not in ('w0', 'w1'):
            raise ValueError('Unknown world')
        self.table, self.requested_world = self.tables[pair_id], int(world[-1])
        self.world_order = tuple(self.world_orders.get(pair_id, (0, 1)))
        self.world = self.world_order[self.requested_world]
        self.state_id = self.table.initial_state
        self.active = True
        return self._frame()

    def _frame(self):
        self.serial += 1
        table, state = self.table, self.table.states[self.state_id]
        capture = state.captures[0 if state.shared else self.world]
        cache_key = (table.pair_id, self.state_id, 0 if state.shared else self.world)
        if cache_key not in self.public_cache:
            self.public_cache[cache_key] = mark_view(capture, table.labels, table.candidate_names)
        marked, marks = self.public_cache[cache_key]
        allowed = table.selectable_names(self.state_id, self.world)
        candidates = tuple(Candidate(name, table.labels[name] if name in allowed else None,
            capture.count(name), capture.distances.get(name), True, True) for name in table.candidate_names)
        proof = CaptureEvidence(f'{table.pair_id}:{self.world}:{self.serial}:{uuid.uuid4().hex}',
            table.pair_id, f'w{self.requested_world}', self.state_id, 'mock' if table.synthetic else 'P',
            capture.render_id, capture.rgb.sha256, marked.sha256, table.synthetic)
        return BackendFrame(capture.rgb, marked, table.instruction,
            tuple((n[0], n[1]) for n in state.neighbors),
            tuple(parse_action(action) for action in state.transitions), marks, candidates,
            table.target_names[self.world], table.kind, proof)

    def step(self, action):
        if not self.active:
            raise RuntimeError('Inactive table backend')
        self.state_id = self.table.states[self.state_id].transitions[action.text()]
        return self._frame()

    def verify_capture(self, frame):
        state = self.table.states[self.state_id]
        capture = state.captures[0 if state.shared else self.world]
        marked, _ = self.public_cache[(self.table.pair_id, self.state_id, 0 if state.shared else self.world)]
        return (self.table.export_verified and self.table.cache_verified
                and frame.raw_rgb == capture.rgb and frame.marked_rgb == marked
                and frame.capture.state_id == self.state_id and frame.capture.source_id == capture.render_id)

    def verify_evidence(self, comparison):
        return False  # P uses authenticated state classification, not the old live-difference object.

    def close(self):
        self.active = False


class TableEnvironment(PairEnvironment):
    def __init__(self, backend, max_response_tokens=256):
        if not isinstance(backend, TableObservationBackend):
            raise TypeError('TableEnvironment requires its P backend')
        super().__init__(backend, EpisodeBudget(5, 5, max_response_tokens), RewardSpec(.05, 0.0))

    def reset(self, pair_id, world):
        self.tau = self.x_seen_step = None
        return super().reset(pair_id, world)

    def _accept_frame(self, frame):
        if not frame.capture.synthetic and not self.backend.verify_capture(frame):
            raise ValueError('P capture not verified')
        self.frame = frame
        state = self.backend.table.states[self.backend.state_id]
        if self.tau is None and not state.shared:
            self.tau = self.steps
        if self.x_seen_step is None and state.anchor_pixels[self.backend.world] >= 250:
            self.x_seen_step = self.steps
        self.evidence_entered = self.tau is not None
        self.evidence_status = 'entered' if self.evidence_entered else 'not_entered_verified'
        self.evidence_proofs.append({'version': TABLE_VERSION, 'state_id': state.state_id,
            'decision_index': self.steps, 'shared': state.shared, 'tau': self.tau,
            'x_seen_step': self.x_seen_step, 'original_world_index': self.backend.world,
            'anchor_pixels': state.anchor_pixels, 'frame_max_abs_diff': state.frame_max_abs_diff,
            'table_manifest': copy.deepcopy(self.backend.table.manifest_reference),
            'captures': [{'render_id': view.render_id, 'process_id': view.process_id,
                'rgb_sha256': view.rgb.sha256, 'snapshot_sha256': view.source.get('snapshot_sha256'),
                'anchor_mask_sha256': hashlib.sha256(view.masks[self.backend.table.anchor_name]).hexdigest()}
                for view in state.captures], 'synthetic': self.backend.table.synthetic})

    @property
    def info(self):
        state = self.backend.table.states[self.backend.state_id]
        return {**super().info, 'backend_version': TABLE_VERSION, 'state_id': state.state_id,
            'shared': state.shared, 'tau': self.tau, 'x_seen_step': self.x_seen_step,
            'anchor_pixels_w0': state.anchor_pixels[self.backend.world_order[0]],
            'anchor_pixels_w1': state.anchor_pixels[self.backend.world_order[1]],
            'world_order': self.backend.world_order, 'original_world_id': f'w{self.backend.world}',
            'frame_max_abs_diff': state.frame_max_abs_diff,
            'selectable_labels': sorted(row.mark_label for row in self.frame.candidates if row.success_eligible),
            'prefix_verified_identical': state.shared and self.backend.table.export_verified,
            'cache_matches_live': None if self.backend.table.synthetic else self.backend.table.cache_verified,
            'table_temporal_proof': copy.deepcopy(self.evidence_proofs[-1]),
            'evidence_provenance_verified': self.backend.table.export_verified}

    def step(self, action):
        try:
            parsed = parse_action(action)
        except ValueError:
            parsed = None
        # The fifth turn cannot introduce an unexported fifth observation. It
        # still consumes its turn and observation cost, with zero terminal SR.
        if (not self.done and self.steps == 4 and parsed is not None
                and parsed.kind in ('turn_left', 'turn_right', 'move')):
            self.steps += 1
            self.done = True
            return public_observation(self.frame), -.05, True, {**self.info,
                'action_valid': parsed in self.frame.observation_actions, 'action_success': False,
                'terminal_reward': 0.0, 'budget_exhausted': True}
        observation, reward, done, info = super().step(action)
        if self.steps == 4 and not done:
            self.frame = replace(self.frame, observation_actions=())
            observation = public_observation(self.frame)
            info = {**info, 'final_select_turn': True}
        return observation, reward, done, info

"""T02 public interface, with the approved P backend available by lazy factory.

No simulator, training framework, image library, or GPU is imported. Only an
explicit backend supplies observations. Only authenticated P shared states may
give w0 pixels/text to both worlds after same-process paired-render checks.
Mock captures cannot certify a real prefix or backend consistency.
"""
from dataclasses import dataclass
import copy
import hashlib
import json
import math
import re
from typing import Optional, Protocol, Tuple
import uuid

from src.envs.pair_core import Action

KINDS = ("direct_flip", "direct_stay", "look_flip", "look_stay")
WORLDS = ("w0", "w1")


def positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(name + " must be a positive integer")


@dataclass(frozen=True)
class RGBFrame:
    data: bytes
    width: int
    height: int
    dtype: str = "uint8"
    channels: int = 3

    def __post_init__(self):
        positive_integer(self.width, "width")
        positive_integer(self.height, "height")
        if (type(self.data) is not bytes or self.dtype != "uint8" or self.channels != 3
                or len(self.data) != self.width * self.height * 3):
            raise ValueError("Require immutable, complete uint8 RGB pixels")

    @property
    def shape(self):
        return self.height, self.width, self.channels

    @property
    def sha256(self):
        return hashlib.sha256(self.data).hexdigest()


@dataclass(frozen=True)
class PublicMark:
    label: int
    anchor_pixel: Tuple[int, int]
    bounding_box: Tuple[int, int, int, int]

    def __post_init__(self):
        positive_integer(self.label, "mark label")
        if (type(self.anchor_pixel) is not tuple or len(self.anchor_pixel) != 2
                or type(self.bounding_box) is not tuple or len(self.bounding_box) != 4
                or any(type(value) is not int or value < 0 for value in (*self.anchor_pixel, *self.bounding_box))):
            raise ValueError("Marks require integer image coordinates")
        x, y = self.anchor_pixel
        left, top, right, bottom = self.bounding_box
        if not left <= x <= right or not top <= y <= bottom:
            raise ValueError("Mark anchor must lie within its inclusive bounding box")


@dataclass(frozen=True)
class PublicObservation:
    image: RGBFrame  # Actual marked RGB supplied by the backend, not drawn here.
    text: str
    marks: Tuple[PublicMark, ...]

    def __post_init__(self):
        if (not isinstance(self.image, RGBFrame) or not isinstance(self.text, str) or not isinstance(self.marks, tuple)
                or any(not isinstance(mark, PublicMark) for mark in self.marks)):
            raise ValueError("Malformed public observation")
        if len({mark.label for mark in self.marks}) != len(self.marks):
            raise ValueError("Public labels must be unique")
        if any(mark.bounding_box[2] >= self.image.width or mark.bounding_box[3] >= self.image.height for mark in self.marks):
            raise ValueError("Public mark lies outside the supplied image")

    def policy_dict(self):
        # Exhaustive public representation: no privileged fields or IDs.
        return {"image": {"data": self.image.data, "shape": self.image.shape, "dtype": self.image.dtype},
                "text": self.text, "marks": [{"label": mark.label, "anchor_pixel": mark.anchor_pixel,
                                                "bounding_box": mark.bounding_box} for mark in self.marks]}


@dataclass(frozen=True)
class CaptureEvidence:
    capture_id: str
    pair_id: str
    world_id: str
    state_id: str
    backend_kind: str  # mock, L or P; no L/P implementation is supplied here.
    source_id: str  # Independent originating render, including for a P cache hit.
    raw_rgb_sha256: str
    marked_rgb_sha256: str
    synthetic: bool

    def __post_init__(self):
        if self.world_id not in WORLDS or self.backend_kind not in ("mock", "L", "P"):
            raise ValueError("Unknown world or backend")
        if any(not isinstance(value, str) or not value for value in
               (self.capture_id, self.pair_id, self.state_id, self.source_id)):
            raise ValueError("Capture identities must be explicit")
        if type(self.synthetic) is not bool or (self.backend_kind == "mock") != self.synthetic:
            raise ValueError("Mock evidence must remain explicitly synthetic")
        if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in (self.raw_rgb_sha256, self.marked_rgb_sha256)):
            raise ValueError("Capture requires exact raw and marked image hashes")


@dataclass(frozen=True)
class EvidenceComparison:
    left: RGBFrame
    right: RGBFrame
    left_capture: CaptureEvidence
    right_capture: CaptureEvidence
    relevant_mask_union_pixels: Tuple[int, ...]  # Flattened y*width+x, not visibility metadata.
    marked_left: RGBFrame  # Actual policy pixels, not merely an unmarked renderer frame.
    marked_right: RGBFrame

    def __post_init__(self):
        if not self.left.shape == self.right.shape == self.marked_left.shape == self.marked_right.shape:
            raise ValueError("Evidence images must have the same resolution")
        a, b = self.left_capture, self.right_capture
        if (a.pair_id != b.pair_id or {a.world_id, b.world_id} != set(WORLDS)
                or a.capture_id == b.capture_id or a.source_id == b.source_id
                or a.raw_rgb_sha256 != self.left.sha256 or b.raw_rgb_sha256 != self.right.sha256
                or a.marked_rgb_sha256 != self.marked_left.sha256 or b.marked_rgb_sha256 != self.marked_right.sha256
                or a.synthetic != b.synthetic):
            raise ValueError("Evidence requires independently captured opposite worlds of the same pair")
        pixels = self.relevant_mask_union_pixels
        if (type(pixels) is not tuple or len(set(pixels)) != len(pixels)
                or any(type(index) is not int or not 0 <= index < self.left.width * self.left.height for index in pixels)):
            raise ValueError("Relevant instance-mask union must contain unique in-image pixels")

    @property
    def changed_pixels(self):
        return sum(any(abs(self.left.data[index * 3 + channel] - self.right.data[index * 3 + channel]) > 10
                       for channel in range(3)) for index in self.relevant_mask_union_pixels)

    @property
    def required_pixels(self):
        return (250 * self.left.width * self.left.height + 255 * 255 - 1) // (255 * 255)

    @property
    def entered(self):
        return self.changed_pixels >= self.required_pixels and self.marked_input_changed_pixels >= self.required_pixels

    @property
    def marked_input_changed_pixels(self):
        return sum(any(abs(self.marked_left.data[index * 3 + channel] - self.marked_right.data[index * 3 + channel]) > 10
                       for channel in range(3)) for index in self.relevant_mask_union_pixels)

    def summary(self):
        return {"capture_ids": [self.left_capture.capture_id, self.right_capture.capture_id],
                "source_ids": [self.left_capture.source_id, self.right_capture.source_id],
                "raw_rgb_sha256": [self.left.sha256, self.right.sha256],
                "relevant_mask_union_sha256": hashlib.sha256(json.dumps(self.relevant_mask_union_pixels).encode()).hexdigest(),
                "changed_pixels": self.changed_pixels, "required_pixels": self.required_pixels,
                "marked_input_changed_pixels": self.marked_input_changed_pixels,
                "channel_difference_strictly_greater_than": 10, "entered": self.entered,
                "synthetic": self.left_capture.synthetic}


@dataclass(frozen=True)
class Candidate:
    object_name: str  # Stable privileged identity; never placed in public obs.
    mark_label: Optional[int]
    mask_pixel_count: int
    distance_m: Optional[float]
    legal_target: bool
    public_candidate: bool  # Public mark role; explicit and independent of success metadata.

    def __post_init__(self):
        if (not isinstance(self.object_name, str) or not self.object_name or type(self.legal_target) is not bool
                or type(self.public_candidate) is not bool):
            raise ValueError("Candidate identity/category must be explicit")
        if type(self.mask_pixel_count) is not int or self.mask_pixel_count < 0:
            raise ValueError("Use measured instance-mask pixel counts")
        if self.mark_label is not None:
            positive_integer(self.mark_label, "candidate mark")
        if self.distance_m is not None and (isinstance(self.distance_m, bool) or not math.isfinite(self.distance_m) or self.distance_m < 0):
            raise ValueError("Distance must be a nonnegative measured value or unknown")

    @property
    def public_selectable(self):
        return self.public_candidate and self.mark_label is not None and self.mask_pixel_count > 0

    @property
    def success_eligible(self):
        return (self.public_selectable and self.legal_target
                and self.distance_m is not None and self.distance_m <= 1.5)


@dataclass(frozen=True)
class BackendFrame:
    raw_rgb: RGBFrame  # Privileged strict audit; policy sees marked_rgb only.
    marked_rgb: RGBFrame
    instruction: str
    neighbor_directions: Tuple[Tuple[int, str], ...]
    observation_actions: Tuple[Action, ...]
    marks: Tuple[PublicMark, ...]
    candidates: Tuple[Candidate, ...]
    target_name: str
    kind: str
    capture: CaptureEvidence
    evidence: Optional[EvidenceComparison] = None
    action_success: bool = True

    def __post_init__(self):
        if (self.kind not in KINDS or self.raw_rgb.shape != self.marked_rgb.shape
                or not isinstance(self.instruction, str) or not self.instruction
                or any(type(value) is not tuple for value in (self.neighbor_directions, self.observation_actions, self.marks, self.candidates))
                or any(not isinstance(value, Action) for value in self.observation_actions)
                or any(not isinstance(value, Candidate) for value in self.candidates)
                or type(self.action_success) is not bool):
            raise ValueError("Invalid backend observation/task")
        if self.capture.raw_rgb_sha256 != self.raw_rgb.sha256 or self.capture.marked_rgb_sha256 != self.marked_rgb.sha256:
            raise ValueError("Backend pixels differ from their capture evidence")
        PublicObservation(self.marked_rgb, self.instruction, self.marks)
        if (len({row.object_name for row in self.candidates}) != len(self.candidates)
                or self.target_name not in {row.object_name for row in self.candidates if row.legal_target}):
            raise ValueError("Require unique stable candidates and one legal correct target")
        labels = {mark.label for mark in self.marks}
        assigned = [row.mark_label for row in self.candidates if row.mark_label is not None]
        if (len(set(assigned)) != len(assigned) or not set(assigned) <= labels
                or any(row.mask_pixel_count > self.raw_rgb.width * self.raw_rgb.height for row in self.candidates)):
            raise ValueError("Candidate labels/mask counts conflict with the supplied image")
        neighbor_indices = [index for index, _ in self.neighbor_directions]
        if len(set(neighbor_indices)) != len(neighbor_indices):
            raise ValueError("Duplicate neighbor index")
        for index, direction in self.neighbor_directions:
            positive_integer(index, "neighbor index")
            if direction not in ("front", "front-right", "right", "back-right", "back", "back-left", "left", "front-left"):
                raise ValueError("Neighbor text must use public relative directions")
        if len(set(self.observation_actions)) != len(self.observation_actions):
            raise ValueError("Duplicate backend action")
        for action in self.observation_actions:
            if (action.kind == "select" or (action.kind == "move" and action.argument not in neighbor_indices)
                    or (action.kind == "open" and action.argument not in labels)):
                raise ValueError("Backend observation action lacks public action evidence")
        if self.evidence is not None and not any(capture == self.capture and image == self.raw_rgb for capture, image in
                ((self.evidence.left_capture, self.evidence.left), (self.evidence.right_capture, self.evidence.right))):
            raise ValueError("Discriminative evidence must include this actually returned capture")


class ObservationBackend(Protocol):
    def reset(self, pair_id: str, world: str) -> BackendFrame: ...
    def step(self, action: Action) -> BackendFrame: ...
    def verify_capture(self, frame: BackendFrame) -> bool:
        """Real backends must verify persisted independent render provenance, not IDs alone."""
        ...
    def verify_evidence(self, comparison: EvidenceComparison) -> bool:
        """Verify both source captures/masks at the matched viewpoint/orientation/container state."""
        ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class EpisodeBudget:
    max_steps: int
    max_model_calls: int
    max_response_tokens: int

    def __post_init__(self):
        for key, value in vars(self).items():
            positive_integer(value, key)


@dataclass(frozen=True)
class RewardSpec:
    observation_cost: float
    invalid_action_reward: float  # Explicit pending T04 format-penalty interpretation.

    def __post_init__(self):
        if any(isinstance(value, bool) or not math.isfinite(value) for value in vars(self).values()) or self.observation_cost < 0:
            raise ValueError("Reward values must be finite and observation cost nonnegative")


def parse_action(value):
    if isinstance(value, Action):
        return value
    if not isinstance(value, str):
        raise ValueError("One discrete action text is required")
    text = value.strip()
    if text in ("turn_left", "turn_right"):
        return Action(text)
    match = re.fullmatch(r"(move|open|select) ([1-9]\d*)", text)
    if not match:
        raise ValueError("Exactly one canonical discrete action is allowed per decision")
    return Action(match[1], int(match[2]))


def public_observation(frame):
    actions = [action.text() for action in frame.observation_actions]
    actions += [f"select {row.mark_label}" for row in frame.candidates if row.public_selectable]
    marks = [{"label": mark.label, "anchor_pixel": mark.anchor_pixel, "bounding_box": mark.bounding_box} for mark in frame.marks]
    text = (frame.instruction + "\nNeighbors: " + json.dumps(frame.neighbor_directions, separators=(",", ":"))
            + "\nAvailable actions: " + json.dumps(actions, separators=(",", ":"))
            + "\nMarks: " + json.dumps(marks, sort_keys=True, separators=(",", ":")))
    return PublicObservation(frame.marked_rgb, text, frame.marks)


class PairEnvironment:
    def __init__(self, backend: ObservationBackend, budget: EpisodeBudget, rewards: RewardSpec):
        self.backend, self.budget, self.rewards = backend, budget, rewards
        self.frame = None
        self.done = True

    def reset(self, pair_id, world):
        if world not in WORLDS or not isinstance(pair_id, str) or not pair_id:
            raise ValueError("reset requires an explicit pair_id and w0/w1")
        frame = self.backend.reset(pair_id, world)
        if (frame.capture.pair_id, frame.capture.world_id) != (pair_id, world):
            raise ValueError("Backend returned another pair/world")
        self.pair_id, self.world_id = pair_id, world
        self.episode_id = uuid.uuid4().hex
        self.steps = self.model_calls = self.obs_count = 0
        self.call_ids = set()
        self.selected = self.success = self.done = self.evidence_entered = False
        self.selected_name = None
        self.evidence_proofs = []
        self.evidence_status = "unknown"
        self.frame = None
        self._accept_frame(frame)
        if frame.kind.startswith("direct_") and not self.evidence_entered:
            self.done = True
            raise ValueError("Direct labels require qualifying evidence in the initial actual input")
        return public_observation(frame)

    def _accept_frame(self, frame):
        if self.frame is not None and not self.done:
            previous = self.frame
            if (frame.capture.pair_id != self.pair_id or frame.capture.world_id != self.world_id
                    or frame.kind != previous.kind or frame.target_name != previous.target_name
                    or frame.capture.backend_kind != previous.capture.backend_kind):
                raise ValueError("Backend changed episode identity or target")
        if not frame.capture.synthetic and not self.backend.verify_capture(frame):
            raise ValueError("Real backend capture provenance was not verified")
        if (frame.evidence is not None and not frame.capture.synthetic
                and not self.backend.verify_evidence(frame.evidence)):
            raise ValueError("Real discriminative evidence provenance was not verified")
        self.frame = frame
        if frame.evidence is not None:
            self.evidence_proofs.append(frame.evidence.summary())
            self.evidence_entered |= frame.evidence.entered
            self.evidence_status = "entered" if self.evidence_entered else "not_entered_verified"
        elif not self.evidence_entered:
            # An earlier non-discriminative frame cannot certify this new frame.
            self.evidence_status = "unknown"

    def record_model_call(self, call_id):
        if self.done or self.model_calls >= self.budget.max_model_calls:
            raise RuntimeError("Episode model-call budget exhausted")
        if not isinstance(call_id, str) or not call_id or call_id in self.call_ids:
            raise ValueError("Each physical policy call needs a unique explicit ID per episode")
        self.call_ids.add(call_id)
        self.model_calls += 1

    @property
    def info(self):
        if self.frame is None:
            raise RuntimeError("Call reset first")
        return {"pair_id": self.pair_id, "world_id": self.world_id, "episode_id": self.episode_id,
                "decision_index": self.steps, "kind": self.frame.kind, "target_name": self.frame.target_name,
                "evidence_entered": self.evidence_entered, "evidence_status": self.evidence_status,
                "evidence_proof": copy.deepcopy(tuple(self.evidence_proofs)), "obs_count": self.obs_count,
                "selected": self.selected, "selected_name": self.selected_name, "success": self.success,
                "steps": self.steps, "model_calls": self.model_calls, "done": self.done,
                "capture": dict(vars(self.frame.capture)), "synthetic_backend": self.frame.capture.synthetic,
                "capture_provenance_verified": not self.frame.capture.synthetic,
                "evidence_provenance_verified": bool(not self.frame.capture.synthetic and
                    (self.evidence_entered or self.frame.evidence is not None)),
                "raw_rgb_sha256": self.frame.raw_rgb.sha256,
                "candidates": [{"object_name": row.object_name, "mark_label": row.mark_label,
                    "action_text": f"select {row.mark_label}" if row.public_selectable else None,
                    "action_legal": row.public_selectable, "success_eligible": row.success_eligible,
                    "public_candidate": row.public_candidate, "mask_pixel_count": row.mask_pixel_count,
                    "distance_m": row.distance_m} for row in self.frame.candidates]}

    def step(self, action):
        if self.done or self.frame is None or self.steps >= self.budget.max_steps:
            raise RuntimeError("Episode is inactive or its step budget is exhausted")
        self.steps += 1
        valid, effective, reward = False, False, self.rewards.invalid_action_reward
        try:
            parsed = parse_action(action)
        except ValueError:
            parsed = None
        if parsed is not None and parsed.kind == "select":
            candidates = [row for row in self.frame.candidates if row.mark_label == parsed.argument]
            chosen = candidates[0] if len(candidates) == 1 else None
            self.selected = self.done = True
            self.selected_name = chosen.object_name if chosen else None
            valid = bool(chosen and chosen.public_selectable)
            effective = valid
            self.success = valid and chosen.success_eligible and self.selected_name == self.frame.target_name
            reward = float(self.success) if valid else self.rewards.invalid_action_reward
        elif parsed is not None and parsed in self.frame.observation_actions:
            old_capture = self.frame.capture.capture_id
            try:
                following = self.backend.step(parsed)
                if following.capture.capture_id == old_capture:
                    raise ValueError("Observation action did not return fresh backend capture evidence")
                self._accept_frame(following)
            except Exception:
                self.done = True  # A failed backend may have changed state; never act on stale RGB.
                raise
            self.obs_count += 1
            valid, effective, reward = True, following.action_success, -self.rewards.observation_cost
        if self.steps >= self.budget.max_steps or self.model_calls >= self.budget.max_model_calls:
            self.done = True
        return public_observation(self.frame), reward, self.done, {**self.info, "action_valid": valid,
                                                                 "action_success": effective or self.success}

    def close(self):
        self.backend.close()
        self.done = True


def compare_prefix(left: PairEnvironment, right: PairEnvironment):
    """Compare actual inputs; validated P shared-state aliases are explicit."""
    a, b = left.frame, right.frame
    if a is None or b is None:
        raise ValueError("Both backends must have independently returned an observation")
    from .pair_table import TableEnvironment
    if isinstance(left, TableEnvironment) or isinstance(right, TableEnvironment):
        if not isinstance(left, TableEnvironment) or not isinstance(right, TableEnvironment):
            raise ValueError('Cannot compare live and table observation contracts')
        first, second = left.backend.table, right.backend.table
        equal = public_observation(a) == public_observation(b)
        same_state = (first.pair_id == second.pair_id and a.capture.state_id == b.capture.state_id
            and {left.world_id, right.world_id} == set(WORLDS)
            and left.backend.world_order == right.backend.world_order)
        shared = same_state and first.states[a.capture.state_id].shared and second.states[b.capture.state_id].shared
        verified = bool(shared and equal and first.export_verified and second.export_verified
            and first.manifest_reference == second.manifest_reference
            and left.backend.verify_capture(a) and right.backend.verify_capture(b))
        return {'same_policy_input': equal, 'raw_rgb_exact': a.raw_rgb == b.raw_rgb,
            'same_input_at_t_strict': equal and a.raw_rgb == b.raw_rgb,
            'prefix_verified_identical': verified, 'synthetic_only': first.synthetic or second.synthetic,
            'independent_capture_ids': a.capture.capture_id != b.capture.capture_id,
            'shared_public_source': shared, 'capture_ids': [a.capture.capture_id, b.capture.capture_id],
            'note': 'P shared alias uses prior paired-render proof; current equality is not fresh simulator evidence.'}
    independent = (a.capture.pair_id == b.capture.pair_id and {a.capture.world_id, b.capture.world_id} == set(WORLDS)
                   and a.capture.capture_id != b.capture.capture_id and a.capture.source_id != b.capture.source_id)
    equal = public_observation(a) == public_observation(b)
    real_verified = (independent and not a.capture.synthetic and not b.capture.synthetic
                     and left.backend.verify_capture(a) and right.backend.verify_capture(b))
    return {"same_policy_input": equal, "independent_capture_ids": independent,
            "raw_rgb_exact": a.raw_rgb == b.raw_rgb,
            "same_input_at_t_strict": bool(equal and a.raw_rgb == b.raw_rgb),
            "prefix_verified_identical": bool(equal and a.raw_rgb == b.raw_rgb and real_verified),
            "synthetic_only": a.capture.synthetic or b.capture.synthetic,
            "capture_ids": [a.capture.capture_id, b.capture.capture_id],
            "note": "Equality in a CPU mock is not real-backend prefix verification; full history comparison belongs to the paired rollout controller."}


def table_environment(tables, *, allow_synthetic=False, max_response_tokens=256, world_orders=None):
    """Explicit P backend; real use remains blocked until T02 checks are passed."""
    from .pair_table import TableEnvironment, TableObservationBackend
    return TableEnvironment(TableObservationBackend(tables, allow_synthetic=allow_synthetic, world_orders=world_orders),
                            max_response_tokens=max_response_tokens)

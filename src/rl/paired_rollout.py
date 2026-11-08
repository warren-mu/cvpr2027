"""Backend-independent T05 coupling state machine; no simulator/model imports.

Ports carry privileged state. The policy receives ONLY PolicyInput (public RGB,
text, marks, prior responses/actions and the shared prompt contract). Equality is
byte/structure equality, never a similarity threshold or hash-only comparison.
Any divergence permanently forks the pair, even if later observations converge.
Forking on rendering noise does not change the backend's evidence annotation.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Callable, Mapping, Protocol

from .credit import CreditContractError, finite


@dataclass(frozen=True)
class PublicMark:
    label: int
    anchor_pixel: tuple[int, int]
    bounding_box: tuple[int, int, int, int]


@dataclass(frozen=True)
class PublicObservation:
    rgb: bytes
    width: int
    height: int
    dtype: str
    channels: int
    text: str
    marks: tuple[PublicMark, ...]

    def __post_init__(self):
        if (not isinstance(self.rgb, bytes) or self.dtype != "uint8" or self.channels != 3
                or type(self.width) is not int or type(self.height) is not int
                or self.width <= 0 or self.height <= 0 or len(self.rgb) != self.width * self.height * self.channels
                or not isinstance(self.text, str) or not isinstance(self.marks, tuple)
                or any(not isinstance(mark, PublicMark) for mark in self.marks)):
            raise CreditContractError("Invalid immutable public observation")
        if len({mark.label for mark in self.marks}) != len(self.marks):
            raise CreditContractError("Public marks must have unique labels")

    def record(self):
        return {"rgb_sha256": hashlib.sha256(self.rgb).hexdigest(), "width": self.width,
                "height": self.height, "dtype": self.dtype, "channels": self.channels,
                "text": self.text, "marks": [asdict(mark) for mark in self.marks]}


def from_t02_observation(observation):
    """Explicit whitelist adapter; no info/name/target/evidence enters the policy."""
    frame = observation.image
    marks = tuple(PublicMark(mark.label, tuple(mark.anchor_pixel), tuple(mark.bounding_box)) for mark in observation.marks)
    return PublicObservation(bytes(frame.data), frame.width, frame.height, frame.dtype, frame.channels,
                             observation.text, marks)


@dataclass(frozen=True)
class PolicyInput:
    observations: tuple[PublicObservation, ...]
    response_history: tuple[str, ...]
    action_history: tuple[str, ...]
    prompt_contract: str

    def __post_init__(self):
        if (not self.prompt_contract or len(self.observations) != len(self.action_history) + 1
                or len(self.response_history) != len(self.action_history)):
            raise CreditContractError("Complete public history and fixed prompt contract are required")

    def record(self):
        return {"observations": [obs.record() for obs in self.observations],
                "response_history": list(self.response_history), "action_history": list(self.action_history),
                "prompt_contract": self.prompt_contract}

    @property
    def sha256(self):
        # Actual equality uses this dataclass's raw bytes, not this audit digest.
        return hashlib.sha256(json.dumps(self.record(), sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class PolicySample:
    action_text: str
    response_text: str
    policy_id: str
    sample_id: str
    response_token_count: int  # Actual tokenizer count, not characters/whitespace.


@dataclass(frozen=True)
class EpisodeState:
    public: PublicObservation
    privileged: Mapping


@dataclass(frozen=True)
class Transition:
    next_state: EpisodeState
    reward: float
    done: bool


class EpisodePort(Protocol):
    def reset(self) -> EpisodeState: ...
    def record_model_call(self, call_id: str) -> None: ...
    def step(self, action_text: str) -> Transition: ...


class T02EpisodePort:
    """Duck-typed adapter for PairEnvironment; assigns the planned episode ID."""
    def __init__(self, environment, *, pair_id, world_id, episode_id):
        self.environment, self.pair_id, self.world_id, self.episode_id = environment, pair_id, world_id, episode_id

    def _state(self, public, info):
        private = dict(info)
        private.update(backend_episode_id=private["episode_id"], episode_id=self.episode_id)
        return EpisodeState(from_t02_observation(public), private)

    def reset(self):
        public = self.environment.reset(self.pair_id, self.world_id)
        return self._state(public, self.environment.info)

    def record_model_call(self, call_id):
        self.environment.record_model_call(call_id)

    def step(self, action_text):
        public, reward, done, info = self.environment.step(action_text)
        return Transition(self._state(public, info), reward, done)


def validate_private(state, world, pair_id=None, episode_id=None, *, allow_synthetic=False):
    info = state.privileged
    if (info.get("world_id") != world or not info.get("pair_id") or not info.get("episode_id")
            or type(info.get("decision_index")) is not int or info["decision_index"] < 0
            or type(info.get("evidence_entered")) is not bool
            or info.get("kind") not in ("direct_flip", "direct_stay", "look_flip", "look_stay")
            or "evidence_proof" not in info or info.get("evidence_status") not in ("unknown", "not_entered_verified", "entered")):
        raise CreditContractError("Backend must supply explicit privileged identity/evidence metadata")
    synthetic = info.get("synthetic_backend") is True
    if (synthetic and not allow_synthetic) or (not synthetic and info.get("capture_provenance_verified") is not True):
        raise CreditContractError("Real capture provenance is required; synthetic fixtures need explicit opt-in")
    if bool(info["evidence_entered"]) != (info["evidence_status"] == "entered"):
        raise CreditContractError("Evidence boolean and explicit status disagree")
    if (pair_id is not None and info["pair_id"] != pair_id) or (episode_id is not None and info["episode_id"] != episode_id):
        raise CreditContractError("Backend changed the episode or pair identity")
    if info["kind"].startswith("direct_") and info["evidence_entered"] is not True:
        raise CreditContractError("Direct pairs have no pre-evidence segment")


def collect_pair(ports: Mapping[str, EpisodePort], policy: Callable[[PolicyInput], PolicySample],
                 *, max_decisions: int, max_response_tokens: int, prompt_contract: str,
                 pair_instance_id: str, allow_synthetic=False):
    """Collect two realized episodes; shared prefix uses one physical policy call.

    Observations are never copied from one world to the other. The backend must
    independently realize each capture and expose its provenance in privileged.
    Interactions remain two episodes even when one shared policy call supplies
    both actions. Exceptions are not converted into successful trajectories.
    """
    if (set(ports) != {"w0", "w1"} or type(max_decisions) is not int or max_decisions <= 0
            or type(max_response_tokens) is not int or max_response_tokens <= 0 or not pair_instance_id):
        raise CreditContractError("Require two ports and an explicit positive decision cap")
    states = {world: ports[world].reset() for world in ("w0", "w1")}
    for world, state in states.items():
        validate_private(state, world, allow_synthetic=allow_synthetic)
        if state.privileged["decision_index"] != 0:
            raise CreditContractError("A new episode must start at decision zero")
    pair_id = states["w0"].privileged["pair_id"]
    if (states["w1"].privileged["pair_id"] != pair_id
            or states["w0"].privileged["kind"] != states["w1"].privileged["kind"]
            or states["w0"].privileged["episode_id"] == states["w1"].privileged["episode_id"]):
        raise CreditContractError("Ports do not represent distinct episodes of the same pair")
    observations = {world: [states[world].public] for world in ports}
    raw_hashes = {world: [states[world].privileged.get("raw_rgb_sha256")] for world in ports}
    responses, actions, rows = ({world: [] for world in ports} for _ in range(3))
    done = {world: False for world in ports}
    forked, first_fork, sample_calls = False, None, 0
    seen_samples = set()
    for decision in range(max_decisions):
        active = [world for world in ("w0", "w1") if not done[world]]
        if not active:
            break
        contexts = {world: PolicyInput(tuple(observations[world]), tuple(responses[world]), tuple(actions[world]), prompt_contract)
                    for world in active}
        exact = (len(active) == 2 and contexts["w0"] == contexts["w1"]
                 and raw_hashes["w0"] == raw_hashes["w1"] and all(raw_hashes["w0"]))
        if not forked and not exact:
            forked = True
            first_fork = {"decision_index": decision, "reason": "public_input_difference" if len(active) == 2 else "branch_termination",
                          "input_sha256_by_world": {world: context.sha256 for world, context in contexts.items()},
                          "evidence_entered_by_world": {world: states[world].privileged["evidence_entered"] for world in active},
                          "noise_is_not_evidence": True}
        shared = exact and not forked
        if shared:
            captures = [states[world].privileged.get("capture", {}) for world in ("w0", "w1")]
            if any(not capture.get(key) for capture in captures for key in ("capture_id", "source_id")) or any(
                    captures[0][key] == captures[1][key] for key in ("capture_id", "source_id")):
                raise CreditContractError("Shared sampling needs two independently sourced captures; no copied prefix")
        samples = {}
        for world in ([active[0]] if shared else active):
            sample = policy(contexts[world])  # privileged state is deliberately not an argument
            sample_calls += 1
            if (not isinstance(sample, PolicySample) or not isinstance(sample.action_text, str)
                    or not isinstance(sample.response_text, str) or not sample.policy_id or not sample.sample_id
                    or sample.sample_id in seen_samples or type(sample.response_token_count) is not int
                    or sample.response_token_count < 0):
                raise CreditContractError("Policy must return an identifiable fresh sample")
            seen_samples.add(sample.sample_id)
            samples[world] = sample
        if shared:
            samples["w1"] = samples["w0"]
        for world in active:
            current, sample, context = states[world], samples[world], contexts[world]
            ports[world].record_model_call(sample.sample_id)  # logical per-episode budget; one shared physical sample
            exceeded = sample.response_token_count > max_response_tokens
            executed_action = "" if exceeded else sample.action_text
            transition = ports[world].step(executed_action)
            if not isinstance(transition, Transition) or type(transition.done) is not bool:
                raise CreditContractError("Backend step must return an explicit Transition")
            finite(transition.reward, "realized reward")
            validate_private(transition.next_state, world, pair_id, current.privileged["episode_id"], allow_synthetic=allow_synthetic)
            if transition.next_state.privileged["decision_index"] != decision + 1:
                raise CreditContractError("Backend decision index is not aligned with the realized action")
            if current.privileged["evidence_entered"] and not transition.next_state.privileged["evidence_entered"]:
                raise CreditContractError("Evidence already in history cannot disappear")
            rows[world].append({"pair_id": pair_id, "pair_instance_id": pair_instance_id,
                "episode_id": current.privileged["episode_id"],
                "world_id": world, "decision_index": decision,
                "public_input": context.record(), "input_sha256": context.sha256,
                "response_text": sample.response_text, "action_text": executed_action,
                "proposed_action_text": sample.action_text, "response_token_count": sample.response_token_count,
                "response_token_budget_exceeded": exceeded,
                "policy_id": sample.policy_id, "sample_id": sample.sample_id,
                "shared_sample_id": sample.sample_id if shared else None,
                "shared_input_verified_exact": bool(shared and not current.privileged.get("synthetic_backend")),
                "shared_public_input_exact": shared, "synthetic_shared_input_exact": bool(shared and current.privileged.get("synthetic_backend")),
                "forked_before_action": forked,
                "pre_evidence": (None if current.privileged["evidence_status"] == "unknown" else not current.privileged["evidence_entered"]),
                "privileged_before": dict(current.privileged),
                "privileged_after": dict(transition.next_state.privileged),
                "reward": transition.reward, "done": transition.done})
            actions[world].append(executed_action)
            responses[world].append(sample.response_text)
            observations[world].append(transition.next_state.public)
            raw_hashes[world].append(transition.next_state.privileged.get("raw_rgb_sha256"))
            states[world], done[world] = transition.next_state, transition.done
    return {"pair_id": pair_id, "pair_instance_id": pair_instance_id,
            "decisions_by_world": rows, "forked": forked, "first_fork": first_fork,
            "physical_policy_sampling_calls": sample_calls,
            "logical_model_calls_by_world": {world: len(actions[world]) for world in ports},
            "environment_steps_by_world": {world: len(rows[world]) for world in ports},
            "done_by_world": done, "truncated_worlds": [world for world in ports if not done[world]],
            "complete": all(done.values()), "max_decisions": max_decisions, "max_response_tokens": max_response_tokens,
            "notes": ["Sampling calls exclude teacher-forced Q scoring calls; record those separately.",
                      "Input equality was checked on actual bytes/text/marks/full public history.",
                      "Forking is permanent; an RGB/text/mark difference does not itself certify task evidence."]}

"""T05 decision-level credit and auxiliary-loss contracts.

Production T0/T1 and post-evidence steps keep the upstream masked-GAE output.
This module never changes critic targets, normalizes advantages, or decides the
still-unfrozen token-to-decision adapter policy. Revised T05 explicitly conditions
Q_commit on the complete public set of selectable candidate action texts.
All floats supplied here are detached targets; differentiable auxiliary helpers
are separate and import torch only when explicitly called.
"""
from dataclasses import asdict, dataclass
import math
import random
from typing import Mapping, Optional, Sequence

ARMS = ("T0", "T1", "T2", "T3", "T4")
PAIRED_ARMS = ("T1", "T3", "T4")
REQUIRED_ARMS = ("T0", "T2", "T3", "T4")
SHOULD_DO_ARMS = ("T1",)
# Validation of floating-point arithmetic only; never clips/rescales a score.
PROBABILITY_CHECK_TOLERANCE = 1e-6


class CreditContractError(ValueError):
    pass


class UnsupportedCredit(CreditContractError):
    """An unresolved information/distribution convention must not become a label."""


def finite(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise CreditContractError(f"{label} must be a finite scalar")
    return float(value)


def prior(rho):
    rho = finite(rho, "rho")
    if not 0 <= rho <= 1:
        raise CreditContractError("rho must lie in [0,1]")
    return {"w0": 1 - rho, "w1": rho}


@dataclass(frozen=True)
class CandidateScore:
    object_name: str
    action_text: str
    log_probability: float
    token_log_probabilities: tuple[float, ...]


@dataclass(frozen=True)
class ActionScores:
    candidates: tuple[CandidateScore, ...]
    input_sha256: str
    policy_id: str
    scoring_id: str
    scoring_contract: str
    complete_action_events: bool
    probability_space: str = "unconditional_model_action"
    length_normalized: bool = False
    scoring_model_calls: int = 1
    public_select_action_texts: tuple[str, ...] = ()
    candidate_domain_source: str = ""

    def validate(self):
        if (not self.input_sha256 or not self.policy_id or not self.scoring_id or not self.scoring_contract
                or not self.complete_action_events or self.probability_space != "unconditional_model_action"
                or self.length_normalized or not self.candidate_domain_source
                or not isinstance(self.public_select_action_texts, tuple)):
            raise CreditContractError("Require complete, unconditioned action-event probabilities with explicit scoring provenance")
        if type(self.scoring_model_calls) is not int or self.scoring_model_calls < 0:
            raise CreditContractError("Record actual scoring calls, including zero for an identified cached score")
        names, texts, total = set(), set(), 0.0
        for item in self.candidates:
            if (not item.object_name or not item.action_text.startswith("select ")
                    or item.object_name in names or item.action_text in texts):
                raise CreditContractError("Candidate names and complete select action events must be unique")
            names.add(item.object_name)
            texts.add(item.action_text)
            logp = finite(item.log_probability, "action log probability")
            if logp > 0 or not item.token_log_probabilities:
                raise CreditContractError("Require a nonpositive log probability and complete token scores, including action termination")
            token_logs = [finite(value, "token log probability") for value in item.token_log_probabilities]
            if any(value > 0 for value in token_logs) or not math.isclose(sum(token_logs), logp,
                    rel_tol=PROBABILITY_CHECK_TOLERANCE, abs_tol=PROBABILITY_CHECK_TOLERANCE):
                raise CreditContractError("Action log probability must equal the sum, not mean, of its scored token log probabilities")
            total += math.exp(logp)
        if total > 1 + PROBABILITY_CHECK_TOLERANCE:
            raise CreditContractError("Disjoint select events have probability mass above one")
        if (len(set(self.public_select_action_texts)) != len(self.public_select_action_texts)
                or set(self.public_select_action_texts) != texts):
            raise CreditContractError("Score every publicly selectable candidate exactly once; no hidden subset normalization")
        return self

    def record(self):
        self.validate()
        result = asdict(self)
        result["candidate_probability_mass"] = sum(math.exp(item.log_probability) for item in self.candidates)
        result["probabilities_by_stable_name"] = {item.object_name: math.exp(item.log_probability) for item in self.candidates}
        result["select_conditional_probabilities_by_stable_name"] = select_distribution(self)
        result["q_commit_normalization"] = "complete_public_select_candidate_set"
        return result


def select_distribution(scores: ActionScores):
    """Stable select-only normalization; an observed empty action domain is None."""
    scores.validate()
    if not scores.candidates:
        return None
    largest = max(item.log_probability for item in scores.candidates)
    masses = {item.object_name: math.exp(item.log_probability - largest) for item in scores.candidates}
    total = math.fsum(masses.values())
    return {name: mass / total for name, mass in masses.items()}


def q_commit(scores: ActionScores, target_name: str):
    """Revised T05 conditional select expectation; no public candidates => undefined."""
    if not target_name:
        raise CreditContractError("A unique privileged target name is required")
    distribution = select_distribution(scores)
    return None if distribution is None else distribution.get(target_name, 0.0)


def reference_gae(rewards, values, terminated, *, gamma, gae_lambda, bootstrap_value):
    """CPU scalar reference, no whitening; production uses upstream masked GAE.

    A truncated nonterminal tail bootstraps from bootstrap_value. A true terminal
    cuts both the TD bootstrap and backward GAE recursion. Parameters are explicit.
    """
    if not (len(rewards) == len(values) == len(terminated)):
        raise CreditContractError("GAE trajectory fields have different lengths")
    gamma, gae_lambda = finite(gamma, "gamma"), finite(gae_lambda, "gae_lambda")
    if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1 or any(type(x) is not bool for x in terminated):
        raise CreditContractError("Invalid GAE discount, lambda or termination mask")
    next_value, running = finite(bootstrap_value, "bootstrap_value"), 0.0
    advantages = [0.0] * len(rewards)
    for index in reversed(range(len(rewards))):
        reward, value = finite(rewards[index], "reward"), finite(values[index], "value")
        continuation = 0.0 if terminated[index] else 1.0
        delta = reward + gamma * continuation * next_value - value
        running = delta + gamma * gae_lambda * continuation * running
        advantages[index], next_value = running, value
    return tuple(advantages)


@dataclass(frozen=True)
class CreditStep:
    episode_id: str
    pair_id: str
    pair_instance_id: str
    world_id: str
    decision_index: int
    rho: float
    action_text: str
    action_kind: str
    pre_evidence: bool
    input_sha256: str
    shared_sample_id: Optional[str]
    shared_input_verified_exact: bool
    gae_advantage: float
    realized_return: float
    immediate_reward: float
    target_name: str
    scores: Optional[ActionScores]
    gae_source: str
    return_definition: str
    evidence_proof: Mapping
    evidence_status: str
    evidence_provenance_verified: bool
    tau: Optional[int]
    x_seen_step: Optional[int]
    temporal_proof: Mapping
    synthetic_fixture: bool = False

    @property
    def key(self):
        return (self.episode_id, self.decision_index)

    def validate(self):
        if (not self.episode_id or not self.pair_id or not self.pair_instance_id or self.world_id not in ("w0", "w1")
                or type(self.decision_index) is not int or self.decision_index < 0
                or type(self.pre_evidence) is not bool or not self.input_sha256
                or self.action_kind not in ("observe", "select", "invalid")
                or not self.gae_source or not self.return_definition):
            raise CreditContractError("Incomplete decision-credit provenance")
        prior(self.rho)
        if (any(value is not None and (type(value) is not int or value < 0)
                for value in (self.tau, self.x_seen_step)) or not self.temporal_proof
                or self.pre_evidence != (self.tau is None or self.decision_index < self.tau)
                or (self.x_seen_step is not None and (self.tau is None or self.x_seen_step < self.tau))):
            raise UnsupportedCredit("Require recorded T02 tau/x_seen_step; pre-evidence is t<tau, seen is t>=x_seen_step")
        if (self.evidence_status != ("not_entered_verified" if self.pre_evidence else "entered")
                or not self.evidence_proof
                or (self.evidence_provenance_verified is not True and not self.synthetic_fixture)):
            raise UnsupportedCredit("Unknown/unverified evidence status cannot define a pre/post-evidence training segment")
        for label in ("gae_advantage", "realized_return", "immediate_reward"):
            finite(getattr(self, label), label)
        if self.scores is not None:
            self.scores.validate()
            if self.scores.input_sha256 != self.input_sha256:
                raise CreditContractError("Q score input differs from the actual decision input")
        return self


def same_world_credit(step):
    step.validate()
    if step.action_kind == "invalid":
        return step.gae_advantage
    if step.scores is None:
        raise CreditContractError("Counterfactual credit requires actual policy action scores")
    q = q_commit(step.scores, step.target_name)
    if q is None:
        return step.gae_advantage
    outcome = step.realized_return if step.action_kind == "observe" else step.immediate_reward
    return outcome - q


def trajectory_loss_weight(arm, world_id, rho):
    """Separate token-loss weight; never multiply it into the shared advantage."""
    if arm not in ARMS or world_id not in ("w0", "w1"):
        raise CreditContractError("Unknown arm/world for trajectory loss weighting")
    weights = prior(rho)
    return 2 * weights[world_id] if arm in PAIRED_ARMS else 1.0


def advantages(arm: str, steps: Sequence[CreditStep]):
    """Return detached decision advantages; T0/T1/post-evidence GAE is unchanged."""
    if arm not in ARMS:
        raise CreditContractError("Unknown training arm")
    for step in steps:
        step.validate()
    if len({step.key for step in steps}) != len(steps):
        raise CreditContractError("Repeated episode/decision identity")
    output = {step.key: step.gae_advantage for step in steps}
    if arm in ("T0", "T1"):
        return output
    if arm == "T2":
        for step in steps:
            if step.pre_evidence:
                output[step.key] = same_world_credit(step)
        return output
    pairs = {}
    for step in steps:
        if step.pre_evidence:
            pairs.setdefault((step.pair_instance_id, step.decision_index), []).append(step)
    for pair, members in pairs.items():
        by_world = {step.world_id: step for step in members}
        if len(members) != 2 or set(by_world) != {"w0", "w1"}:
            raise UnsupportedCredit(f"Cannot align both pre-evidence worlds at {pair}")
        first, second = by_world["w0"], by_world["w1"]
        if (first.pair_id != second.pair_id or first.rho != second.rho or first.tau != second.tau
                or not first.shared_input_verified_exact or not second.shared_input_verified_exact
                or first.input_sha256 != second.input_sha256 or not first.shared_sample_id
                or first.shared_sample_id != second.shared_sample_id or first.action_text != second.action_text
                or first.action_kind != second.action_kind):
            raise UnsupportedCredit("Pre-evidence information set/action is not verified equal; noise divergence is not evidence")
        if first.action_kind == "invalid":
            weights = prior(first.rho)
            shared = sum(weights[world] * by_world[world].gae_advantage for world in ("w0", "w1"))
            for step in members:
                output[step.key] = shared
            continue
        if first.scores is None or second.scores is None:
            raise CreditContractError("Both-world Q scores are required")
        public_scores = lambda s: {item.action_text: item.log_probability for item in s.candidates}
        if (first.scores.policy_id != second.scores.policy_id
                or public_scores(first.scores) != public_scores(second.scores)):
            raise CreditContractError("Identical public input must use the same policy action probabilities, aligned by action text")
        weights = prior(first.rho)
        shared = sum(weights[world] * same_world_credit(by_world[world]) for world in ("w0", "w1"))
        # Assign the same computed scalar object; no per-world whitening follows here.
        for step in members:
            output[step.key] = shared
    return output


def episode_plan(arm, pair_ids, *, rho, rng: random.Random):
    """Caller fixes pool selection; this only enforces the 64-episode allocation."""
    if arm not in ARMS:
        raise CreditContractError("Unknown training arm")
    prior(rho)
    expected = 32 if arm in PAIRED_ARMS else 64
    if len(pair_ids) != expected or any(not value for value in pair_ids):
        raise CreditContractError(f"{arm} requires {expected} explicit pool draws, making 64 episodes")
    rows = []
    for index, pair in enumerate(pair_ids):
        worlds = ("w0", "w1") if arm in PAIRED_ARMS else ("w1" if rng.random() < rho else "w0",)
        for world in worlds:
            rows.append({"episode_id": f"draw_{index:03d}_{world}", "pair_id": pair, "pair_instance_id": f"draw_{index:03d}:{pair}",
                         "pool_pair_id": pair, "world_id": world, "rho": rho,
                         "trajectory_loss_weight": trajectory_loss_weight(arm, world, rho),
                         "sampling_design": "both_worlds" if arm in PAIRED_ARMS else "bernoulli_prior"})
    return rows


def flip_hinges(log_probs_by_world, targets, *, margin=1.0):
    """Unreduced branch losses; keys are stable names, logs are full action logs."""
    if set(log_probs_by_world) != {"w0", "w1"} or set(targets) != {"w0", "w1"} or targets["w0"] == targets["w1"]:
        raise CreditContractError("Flip requires two distinct stable targets and both worlds")
    margin = finite(margin, "margin")
    if margin != 1.0:
        raise CreditContractError("T05 fixes margin at 1.0")
    losses = {}
    for world, other in (("w0", "w1"), ("w1", "w0")):
        scores = log_probs_by_world[world]
        if targets[world] not in scores or targets[other] not in scores:
            raise UnsupportedCredit("Both Flip targets need actual scoreable public select actions; never invent invisible marks")
        correct, alternative = finite(scores[targets[world]], "correct log probability"), finite(scores[targets[other]], "other log probability")
        if correct > 0 or alternative > 0:
            raise CreditContractError("Log probabilities must be nonpositive")
        losses[world] = max(0.0, margin - correct + alternative)
    return losses


def symmetric_kl(first, second, *, distribution_source, convention):
    """No automatic conditioning/intersection/epsilon. Caller supplies a distribution.

    Return both directions and the explicitly chosen symmetric convention. This
    makes sum versus half-sum reviewable without silently freezing that choice.
    """
    if not distribution_source or convention not in ("sum", "half_sum"):
        raise UnsupportedCredit("Stay distribution provenance and symmetric-KL convention must be explicitly approved")
    if not first or set(first) != set(second):
        raise UnsupportedCredit("Stay supports must contain the same stable names; no silent intersection")
    for distribution in (first, second):
        values = [finite(value, "selection probability") for value in distribution.values()]
        if any(value < 0 for value in values) or not math.isclose(sum(values), 1.0,
                rel_tol=PROBABILITY_CHECK_TOLERANCE, abs_tol=PROBABILITY_CHECK_TOLERANCE):
            raise UnsupportedCredit("Low-level Stay helper requires an already normalized candidate distribution")
    def kl(left, right):
        total = 0.0
        for name, p in left.items():
            q = right[name]
            if p == 0:
                continue
            if q == 0:
                raise UnsupportedCredit("KL has an infinite support mismatch; no unapproved epsilon smoothing")
            total += p * math.log(p / q)
        return total
    forward, reverse = kl(first, second), kl(second, first)
    return {"forward": forward, "reverse": reverse,
            "symmetric": (forward + reverse) / (2 if convention == "half_sum" else 1),
            "convention": convention, "distribution_source": distribution_source}


def t4_eligible_worlds(arm, steps):
    """Each Flip branch uses its OWN inclusive t >= tau_seen boundary."""
    if arm != "T4":
        return {"w0": False, "w1": False}
    if len(steps) != 2 or {step.world_id for step in steps} != {"w0", "w1"}:
        raise UnsupportedCredit("T4 auxiliary loss requires both branch select observations")
    for step in steps:
        step.validate()
    if len({(step.pair_id, step.pair_instance_id, step.rho) for step in steps}) != 1:
        raise CreditContractError("T4 branch records must describe the same sampled world pair/prior")
    return {step.world_id: (step.action_kind == "select" and step.x_seen_step is not None
                           and step.decision_index >= step.x_seen_step) for step in steps}


def t4_eligibility(arm, steps, *, forked=None):
    # Kept as a compatibility query; a historical fork boolean is not tau_seen.
    return any(t4_eligible_worlds(arm, steps).values())


def consistency_components(arm, steps, *, forked=None, pair_change,
                           stay_distributions=None, distribution_source=None, kl_convention=None):
    """Eligibility-enforced scalar audit of T4 terms, with no hidden reduction."""
    eligibility = t4_eligible_worlds(arm, steps)
    if not any(eligibility.values()):
        return {"eligible": False, "eligible_by_world": eligibility, "weighted_terms": None,
                "reason": "no_T4_select_at_or_after_own_x_seen_step"}
    by_world = {step.world_id: step for step in steps}
    targets = {world: step.target_name for world, step in by_world.items()}
    if pair_change == "flip":
        if targets["w0"] == targets["w1"]:
            raise CreditContractError("Flip needs distinct stable targets")
        if all(eligibility.values()) and steps[0].input_sha256 == steps[1].input_sha256:
            raise UnsupportedCredit("Cannot impose distinct Flip targets on the same actual policy input")
        terms = {}
        for world, eligible in eligibility.items():
            if not eligible:
                continue
            scores = by_world[world].scores
            if scores is None:
                raise UnsupportedCredit("Eligible Flip branch requires actual candidate action scores")
            logs = {score.object_name: score.log_probability for score in scores.candidates}
            other = "w1" if world == "w0" else "w0"
            if targets[world] not in logs or targets[other] not in logs:
                raise UnsupportedCredit("Flip cannot invent an invisible public select action")
            terms[world] = max(0.0, 1.0 - logs[targets[world]] + logs[targets[other]])
        return {"eligible": True, "kind": "flip", "margin": 1.0, "beta": 0.1,
                "eligible_by_world": eligibility,
                "unweighted_terms": terms, "weighted_terms": {world: 0.1 * value for world, value in terms.items()},
                "reduction": "unreduced per-world terms; training reduction awaits frozen common configuration"}
    if pair_change != "stay" or targets["w0"] != targets["w1"]:
        raise CreditContractError("Stay requires the same stable target; unknown pair-change label")
    if not all(eligibility.values()):
        return {"eligible": False, "kind": "stay", "eligible_by_world": eligibility,
                "weighted_terms": None, "reason": "Stay_requires_both_selects_at_or_after_own_x_seen_step"}
    if any(step.scores is None for step in steps):
        raise UnsupportedCredit('Stay requires actual complete public candidate scores in both worlds')
    measured = {world: select_distribution(step.scores) for world, step in by_world.items()}
    if any(distribution is None for distribution in measured.values()):
        raise UnsupportedCredit('No selectable candidates means no Stay distribution')
    if stay_distributions is not None and stay_distributions != measured:
        raise CreditContractError('Supplied Stay distribution differs from actual normalized select scores')
    stay_distributions = measured
    distribution_source = distribution_source or ('complete_public_select_action_scores:' +
                           ','.join(by_world[world].scores.scoring_id for world in ('w0', 'w1')))
    terms = symmetric_kl(stay_distributions["w0"], stay_distributions["w1"],
                         distribution_source=distribution_source, convention=kl_convention)
    return {"eligible": True, "kind": "stay", "beta": 0.1, "eligible_by_world": eligibility, "unweighted_terms": terms,
            "weighted_terms": 0.1 * terms["symmetric"], "reduction": "one supplied pair term"}


def torch_flip_hinges(log_probs_by_world, targets, *, margin=1.0, beta=0.1):
    """Differentiable, unreduced per-world terms; no model or CUDA initialization."""
    if beta != 0.1:
        raise CreditContractError("T05 fixes beta at 0.1")
    detached = {world: {name: float(value.detach()) for name, value in scores.items()}
                for world, scores in log_probs_by_world.items()}
    flip_hinges(detached, targets, margin=margin)
    return {world: beta * (margin - log_probs_by_world[world][targets[world]]
                          + log_probs_by_world[world][targets[other]]).clamp_min(0)
            for world, other in (("w0", "w1"), ("w1", "w0"))}


def torch_symmetric_kl(first, second, *, distribution_source, convention, beta=0.1):
    """Differentiable Stay term with the same explicit distribution contract."""
    if beta != 0.1:
        raise CreditContractError("T05 fixes beta at 0.1")
    raw_first, raw_second = ({name: float(value.detach()) for name, value in mapping.items()} for mapping in (first, second))
    symmetric_kl(raw_first, raw_second, distribution_source=distribution_source, convention=convention)
    # Preserve gradients; skip only mathematical zero/zero support terms.
    terms = [(first[name] - second[name]) * (first[name].log() - second[name].log())
             for name in first if raw_first[name] or raw_second[name]]
    zero = next(iter(first.values())) * 0
    return beta * sum(terms, zero) / (2 if convention == "half_sum" else 1)
